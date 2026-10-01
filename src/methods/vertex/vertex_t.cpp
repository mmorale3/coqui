/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */


#include <cmath>
#include <cstdlib>        // getenv: the read-only R-decay diagnostic switch
#include <random>          // deterministic sketch contractions of the Sigma-dyn dump
#include <unordered_set>
#include <vector>

#include "methods/vertex/vertex_debug.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"     // before nda/lapack.hpp: its CUDA branch (geqrf_batched) names nda::blas::device,
                            // which only blas/interface/cublas_interface.hpp declares (CUDA build, gcc 13)
#include "nda/lapack.hpp"
#include "nda/linalg/eigenelements.hpp"

#include "utilities/check.hpp"
#include "utilities/omp_threads.hpp"   // OpenMP / BLAS thread-count control
#include "utilities/blas_threads.hpp"
#include "utilities/proc_grid_partition.hpp"  // {1,nP,nQ} grid for the distributed Z fold
#include "numerics/sparse/csr_blas.hpp"   // csrmm for the symmetry D-matrix blocks
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/ERI/thc.h"    // restricted-range ISDF point selection
#include "methods/GW/g0_div_utils.hpp"  // eps_inv_head_w at i nu = 0 (the q->0 head machinery)
#include "vertex_t.h"
#include "vertex_wannier_detail.hpp"  // MLWF-frame helpers, needed by vertex_ladder.icc::ladder_inputs
#include "h5/h5.hpp"
#include "nda/h5.hpp"
#include "utilities/kpoint_utils.hpp"
#include "vertex_secondary_fold.hpp"  // distributed downfold of dW (no full Np^2 gather)
#include "vertex_pi.icc"
#include "vertex_sigma.icc"  // fused G^3 W^2 Sigma^C kernel
#include "vertex_sigma_r.icc" // static-vertex response cut Sigma^{C,r}
#include "vertex_ladder.icc"  // ladder polarization (scGW-tilde)
#include "vertex_dynbse.icc"  // full-frequency dynamic-rung BSE driver
#include "vertex_sigma_pair.icc"  // pair-resolved static-ladder vertex in Sigma

namespace methods {
namespace solvers {

  namespace vertex_rdecay_detail {

    /**
     * R-DECAY DIAGNOSTIC. Read-only, enabled by env COQUI_VERTEX_RDECAY=1: measures the
     * lattice (R-space) decay of the coarse-grid interpolants, which decides whether the
     * vertex can be evaluated on a coarse k-mesh and Wannier-interpolated to the fine one.
     * Never touches any physics array.
     *
     * A(R) = (1/Nk) sum_k e^{-2 pi i k.R} A(k) on the mesh-dual minimal-image R
     * lattice; shells are Chebyshev (max_i |n_i|). Logged per shell: max|A(R)|, rms,
     * rel-to-shell-0, and the CUMULATIVE TAIL l2 fraction from that shell outward --
     * an estimate of the truncation error of a coarse mesh resolving shells < s.
     */
    inline void log_rshell_decay_k(auto MF, nda::ArrayOfRank<5> auto const &A,
                                   std::string const &label) {
      const long nt = A.shape(0), ns_ = A.shape(1), nk = A.shape(2);
      const long na = A.shape(3), nb = A.shape(4);
      auto kpts = MF->kpts_crystal();
      auto grid = MF->kp_grid();
      const long g0 = grid(0), g1 = grid(1), g2 = grid(2);
      if (g0 * g1 * g2 != nk) {
        app_log(1, "  [RDECAY] {}: kp_grid {}x{}x{} != nk {} -- table skipped.",
                label, g0, g1, g2, nk);
        return;
      }
      const long nshell = std::max({g0, g1, g2}) / 2 + 1;
      std::vector<double> smax(nshell, 0.0), s2(nshell, 0.0);
      std::vector<long> scnt(nshell, 0);
      for (long n0 = 0; n0 < g0; ++n0)
        for (long n1 = 0; n1 < g1; ++n1)
          for (long n2 = 0; n2 < g2; ++n2) {
            const long m0 = (2 * n0 > g0) ? n0 - g0 : n0;
            const long m1 = (2 * n1 > g1) ? n1 - g1 : n1;
            const long m2 = (2 * n2 > g2) ? n2 - g2 : n2;
            const long sh = std::max({std::labs(m0), std::labs(m1), std::labs(m2)});
            double amax = 0.0, a2 = 0.0;
            for (long it = 0; it < nt; ++it)
              for (long is = 0; is < ns_; ++is)
                for (long a = 0; a < na; ++a)
                  for (long b = 0; b < nb; ++b) {
                    ComplexType acc(0.0);
                    for (long k = 0; k < nk; ++k) {
                      const double ph = -2.0 * M_PI * (kpts(k, 0) * double(m0) +
                                                       kpts(k, 1) * double(m1) +
                                                       kpts(k, 2) * double(m2));
                      acc += ComplexType(std::cos(ph), std::sin(ph)) * A(it, is, k, a, b);
                    }
                    const double v = std::abs(acc) / double(nk);
                    amax = std::max(amax, v);
                    a2 += v * v;
                  }
            smax[sh] = std::max(smax[sh], amax);
            s2[sh] += a2;
            scnt[sh] += 1;
          }
      double tot2 = 0.0;
      for (long s = 0; s < nshell; ++s) tot2 += s2[s];
      app_log(1, "  [RDECAY] {} on {}x{}x{}: shell |R|_inf | n_R | max|A(R)| | "
                 "rms | rel-to-0 | tail-l2-frac(>= shell)",
              label, g0, g1, g2);
      double tail2 = tot2;
      for (long s = 0; s < nshell; ++s) {
        const double rms = (scnt[s] > 0)
            ? std::sqrt(s2[s] / double(scnt[s] * nt * ns_ * na * nb)) : 0.0;
        app_log(1, "  [RDECAY] {}:   {}   {:4d}   {:.4e}   {:.4e}   {:.4e}   {:.4e}",
                label, s, scnt[s], smax[s],
                rms, (smax[0] > 0.0 ? smax[s] / smax[0] : 0.0),
                (tot2 > 0.0 ? std::sqrt(tail2 / tot2) : 0.0));
        tail2 -= s2[s];
      }
    }

    /**
     * The q-side companion for a per-q aux matrix D(q, P, Q) on the FULL transfer mesh
     * (nosym runs only -- IBZ-stored q-objects do not star-unfold elementwise in the
     * aux frame). Scalar channels are transformed: the aux trace plus three fixed
     * deterministic probe bilinears u^dag D(q) v; for Np <= 512 the full matrix is
     * transformed too. The q vectors are validated as crystal-integer multiples of the
     * mesh (MF->Qpts() is cartesian, not crystal) and the table is skipped otherwise.
     */
    inline void log_rshell_decay_q(auto MF, nda::ArrayOfRank<3> auto const &D,
                                   std::string const &label) {
      const long nq = D.shape(0), Np = D.shape(1);
      auto qcart = MF->Qpts();          // cartesian coordinates
      auto lat = MF->lattv();
      auto grid = MF->kp_grid();
      const long g0 = grid(0), g1 = grid(1), g2 = grid(2);
      if (g0 * g1 * g2 != nq) {
        app_log(1, "  [RDECAY] {}: kp_grid {}x{}x{} != nq {} -- table skipped.",
                label, g0, g1, g2, nq);
        return;
      }
      // cartesian -> crystal via the direct lattice, q_crys = q_cart . a / (2 pi);
      // both matrix orientations are tried and the mesh-integrality check adjudicates
      // (self-validating -- a wrong convention skips the table instead of lying).
      const long gv[3] = {g0, g1, g2};
      nda::array<double, 2> qpts(nq, 3);
      bool ok = false;
      for (int orient = 0; orient < 2 and not ok; ++orient) {
        for (long q = 0; q < nq; ++q)
          for (int c = 0; c < 3; ++c) {
            double acc = 0.0;
            for (int d = 0; d < 3; ++d)
              acc += qcart(q, d) * (orient == 0 ? lat(c, d) : lat(d, c));
            qpts(q, c) = acc / (2.0 * M_PI);
          }
        ok = true;
        for (long q = 0; q < nq and ok; ++q)
          for (int c = 0; c < 3 and ok; ++c) {
            const double x = qpts(q, c) * double(gv[c]);
            if (std::abs(x - std::round(x)) > 1e-6) ok = false;
          }
      }
      if (not ok) {
        app_log(1, "  [RDECAY] {}: Qpts do not reduce to crystal multiples of the "
                   "{}x{}x{} mesh in either lattv orientation -- table skipped.",
                label, g0, g1, g2);
        return;
      }
      // channels: trace + three fixed probes (deterministic xorshift)
      const int nch = 4;
      nda::array<ComplexType, 2> uu(nch, Np), vv(nch, Np);
      {
        unsigned long st = 88172645463325252ull;
        auto rnd = [&st]() {  // xorshift, deterministic across platforms
          st ^= st << 13; st ^= st >> 7; st ^= st << 17;
          return double(st % 1000003ull) / 1000003.0 - 0.5;
        };
        for (int c = 1; c < nch; ++c) {
          double nu = 0.0, nv = 0.0;
          for (long P = 0; P < Np; ++P) {
            uu(c, P) = ComplexType(rnd(), rnd());
            vv(c, P) = ComplexType(rnd(), rnd());
            nu += std::norm(uu(c, P));
            nv += std::norm(vv(c, P));
          }
          for (long P = 0; P < Np; ++P) {
            uu(c, P) /= std::sqrt(nu);
            vv(c, P) /= std::sqrt(nv);
          }
        }
      }
      nda::array<ComplexType, 2> ch(nch, nq);
      ch() = ComplexType(0.0);
      for (long q = 0; q < nq; ++q) {
        for (long P = 0; P < Np; ++P) ch(0, q) += D(q, P, P);
        for (int c = 1; c < nch; ++c)
          for (long P = 0; P < Np; ++P)
            for (long Q = 0; Q < Np; ++Q)
              ch(c, q) += std::conj(uu(c, P)) * D(q, P, Q) * vv(c, Q);
      }
      const long nshell = std::max({g0, g1, g2}) / 2 + 1;
      std::vector<double> smax(nshell, 0.0);
      std::vector<long> scnt(nshell, 0);
      for (long n0 = 0; n0 < g0; ++n0)
        for (long n1 = 0; n1 < g1; ++n1)
          for (long n2 = 0; n2 < g2; ++n2) {
            const long m0 = (2 * n0 > g0) ? n0 - g0 : n0;
            const long m1 = (2 * n1 > g1) ? n1 - g1 : n1;
            const long m2 = (2 * n2 > g2) ? n2 - g2 : n2;
            const long sh = std::max({std::labs(m0), std::labs(m1), std::labs(m2)});
            for (int c = 0; c < nch; ++c) {
              ComplexType acc(0.0);
              for (long q = 0; q < nq; ++q) {
                const double ph = -2.0 * M_PI * (qpts(q, 0) * double(m0) +
                                                 qpts(q, 1) * double(m1) +
                                                 qpts(q, 2) * double(m2));
                acc += ComplexType(std::cos(ph), std::sin(ph)) * ch(c, q);
              }
              smax[sh] = std::max(smax[sh], std::abs(acc) / double(nq));
            }
            scnt[sh] += 1;
          }
      app_log(1, "  [RDECAY] {} (trace+probe channels) on {}x{}x{}: shell | n_R | "
                 "max|c(R)| | rel-to-0", label, g0, g1, g2);
      for (long s = 0; s < nshell; ++s)
        app_log(1, "  [RDECAY] {}:   {}   {:4d}   {:.4e}   {:.4e}", label, s, scnt[s],
                smax[s], (smax[0] > 0.0 ? smax[s] / smax[0] : 0.0));
      if (Np <= 512) {
        auto Dv = nda::reshape(D, std::array<long, 5>{1, 1, nq, Np, Np});
        log_rshell_decay_k(MF, Dv, label + " (full matrix)");
      }
    }

  }  // vertex_rdecay_detail

  namespace vertex_timer_detail {

    /**
     * RAII scope timer over a utils::TimerManager slot.
     *
     * Used instead of bare start/stop pairs so a stage cannot be left running on any exit
     * path of the (long) vertex entry points; a mismatched stop would silently corrupt every
     * subsequent reading of that slot rather than failing loudly.
     *
     * The slot id is resolved ONCE in the constructor (TimerManager::add is a map lookup);
     * start/stop then go through the integer overloads, so the per-call overhead is two
     * steady_clock reads. The finest stage timed here (SIG_KERNEL's per-call wrapper) is
     * entered once per eval, not per tuple; inner contraction loops are deliberately not
     * timed, since instrumenting the tuple loop would cost more than it measures.
     */
    struct scoped_timer {
      utils::TimerManager& tm;
      const int id;
      scoped_timer(utils::TimerManager& t, std::string const& name)
        : tm(t), id(t.add(name)) { tm.start(id); }
      ~scoped_timer() { tm.stop(id); }
      scoped_timer(scoped_timer const&) = delete;
      scoped_timer& operator=(scoped_timer const&) = delete;
    };

  }  // namespace vertex_timer_detail

  namespace vertex_head_detail {

    /**
     * The analytic q->0 head of the rung at the Gamma cell, in the STORED-ARRAY
     * convention:
     *
     *   H_PQ = N_k * madelung * conj(chi_P(Gamma)) * chi_Q(Gamma)
     *
     * with chi = thc.basis_head() (the G = 0 plane-wave components of the aux basis)
     * and madelung = MF->madelung() (the Gygi-Baldereschi / probe-charge-Ewald
     * constant, PRB 80, 085114 (2009)). Consistent with the GW code: consuming
     * H_PQ through the GW Hadamard reproduces Sigma_div_correction (thc_gw.icc)
     * exactly (dynamic piece, weight Re[eps_inv_head(tau)]), and through the exchange
     * reproduces HF_K_correction (hf_t.cpp) (bare piece, weight 1).
     *
     * Returns false (H untouched) when the head data are unusable: madelung == 0
     * (model systems) or chi_head not populated (some ERI read paths, see
     * thc_reader_t.hpp). The caller logs and proceeds without insertion.
     */
    /**
     * `scale` is the head-strength factor lambda (vertex_t::_bl_head_scale), applied to xi.
     * lambda = 1 is the untouched head. lambda = 0 makes xi vanish and therefore trips the
     * SAME `xi == 0.0` guard below that a system without a madelung constant does, so the
     * caller takes exactly the no-head ("ignore_g0") branch -- structurally, not merely
     * numerically. Every other head site in vertex_t applies the identical factor; scaling
     * some but not all of them would introduce a W0-vs-W head-weight mismatch.
     */
    template<THC_ERI thc_t>
    bool build_head_rank1(thc_t const& thc, long iq_gamma, long nkpts,
                          nda::array<ComplexType, 2>& H_PQ, double scale = 1.0) {
      auto MF = thc.MF();
      const double xi = MF->madelung() * scale;
      auto chi = thc.basis_head();   // (nqpts_ibz, Np)
      const long Np = H_PQ.shape(0);
      utils::check(chi.shape(0) > iq_gamma and chi.shape(1) == Np,
                   "vertex_head_detail::build_head_rank1: basis_head shape mismatch "
                   "(({}, {}) vs iq_gamma = {}, Np = {}).",
                   chi.shape(0), chi.shape(1), iq_gamma, Np);
      double chi_max = 0.0;
      for (long P = 0; P < Np; ++P) chi_max = std::max(chi_max, std::abs(chi(iq_gamma, P)));
      if (xi == 0.0 or chi_max == 0.0) return false;
      for (long P = 0; P < Np; ++P)
        for (long Q = 0; Q < Np; ++Q)
          H_PQ(P, Q) = double(nkpts) * xi
                       * std::real(std::conj(chi(iq_gamma, P)) * chi(iq_gamma, Q));

      // ---- WHY THE REAL PART (the `Re` above is required, not a safeguard) -------------
      // Gamma is a SELF-INVERSE transfer, and every rung of the Sigma^C kernel must obey
      // W_PQ(q) = W_QP(-q) -- that relation is what makes the diagram's four G-cuts equal,
      // hence what makes Sigma^C equal dPhi/dG. At q = -q it reads W(q) = W(q)^T, which
      // TOGETHER WITH Hermiticity forces the block to be REAL. The exact Coulomb matrix
      // obeys this identically: Z_PQ(0) = sum_G v(G) conj(chi_P(G)) chi_Q(G) satisfies
      // Z_QP(0) = conj(Z_PQ(0)) and Z_PQ(0) = Z_QP(-0) = Z_QP(0), so Z(Gamma) IS real.
      //
      // The rank-1 head conj(chi_P) chi_Q is Hermitian but is real only when chi(Gamma) is
      // real up to ONE global phase -- a property of the auxiliary basis that nothing
      // enforces, and it does not hold in general.
      // Taking the real part is also exactly the +-q microcell average the Gygi /
      // probe-charge construction already implies: the Gamma cell is inversion symmetric
      // and chi_P(-q) = conj(chi_P(q)) for a real basis, so averaging conj(chi_P(q))chi_Q(q)
      // over +-q gives Re[conj(chi_P) chi_Q]. It stays positive semidefinite
      // (Re = a a^T + b b^T with chi = a + i b), so the head keeps its sign.
      //
      // Keeping the antisymmetric part makes Sigma^C non-Hermitian: Im(e_corr), which must
      // vanish, becomes nonzero, and in B-L (which feeds P^{C,L} back into the Dyson
      // equation for W, unlike B-S) it grows from iteration to iteration.
      {
        double d = 0.0, sc = 0.0;
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            const ComplexType raw = std::conj(chi(iq_gamma, P)) * chi(iq_gamma, Q);
            d = std::max(d, std::abs(raw.imag()));
            sc = std::max(sc, std::abs(raw));
          }
        const double rel = (sc > 0.0) ? d / sc : 0.0;
        app_log(2, "  vertex head: discarded antisymmetric part |Im conj(chi_P)chi_Q| / "
                   "|conj(chi_P)chi_Q| at Gamma = {:.3e}", rel);
        if (rel > 1e-8)
          app_log(1, "  vertex head: chi(Gamma) is NOT real up to a global phase "
                     "(|Im| / |.| = {:.3e}).\n"
                     "            The antisymmetric part is DISCARDED -- it is illegal at a "
                     "self-inverse transfer\n"
                     "            (it would make Sigma^C non-Hermitian and, in B-L, "
                     "compound through the Dyson\n"
                     "            equation). Keeping it is what made Im(e_corr) grow on Si.",
                  rel);
      }
      return true;
    }

  } // vertex_head_detail

  /**
   * W-redistribution helpers.
   *
   * The dynamic screened interaction dW lives on the RPA (t,q,P,Q) proc grid
   * (MBState::dW_qtPQ, a distributed_array). The vertex kernels (eval_sigma_C_g3w2,
   * pi_c_accumulate_w) consume a FULLY-REPLICATED (nqpts_ibz, nt_half, Np, Np) tau
   * slab and index arbitrary q: the qy/q_ext inner loops are not distributed, so each
   * rank runs the full inner sums and needs every q of W.
   *
   * gather_dW_replicated is the single gather site (Sigma, Pi, cache_w). dW is a
   * PARTITION of the global array -- every global element lives on exactly one source
   * rank, the rest is zero -- so the all_reduce(plus) of the zero-padded local block is
   * a pure GATHER with no floating-point reassociation: the result is BIT-IDENTICAL on
   * every rank. (A q-owned distributed result would save memory only once the kernels
   * consume q-owned tiles; math::nda::redistribute provides that layout and is covered
   * by test_vertex_wredist.cpp.)
   */
  namespace vertex_redist_detail {

    // Gather the RPA-grid distributed dW into a replicated (nq, nt_half, Np, Np) array.
    template<typename dArray_t, typename comm_t>
    nda::array<ComplexType, 4>
    gather_dW_replicated(dArray_t const& dW, comm_t& comm,
                         long nqpts_ibz, long nt_half, long Np) {
      auto gs = dW.global_shape();
      utils::check(gs[0] == nqpts_ibz and gs[1] == nt_half and gs[2] == Np and gs[3] == Np,
                   "vertex_redist_detail::gather_dW_replicated: unexpected dW global "
                   "shape ({}, {}, {}, {}); expected ({}, {}, {}, {}).",
                   gs[0], gs[1], gs[2], gs[3], nqpts_ibz, nt_half, Np, Np);
      nda::array<ComplexType, 4> W_qtPQ(nqpts_ibz, nt_half, Np, Np);
      W_qtPQ() = ComplexType(0.0);
      W_qtPQ(dW.local_range(0), dW.local_range(1), dW.local_range(2), dW.local_range(3)) =
          dW.local();
      comm.all_reduce_in_place_n(W_qtPQ.data(), W_qtPQ.size(), std::plus<>{});
      return W_qtPQ;
    }

    // Gather ONE global q slice of the RPA-grid distributed dW into a replicated
    // (nt_half, Np, Np) array (per-q tau-domain gather). Returns EXACTLY gather_dW_replicated(dW, ...)( iq, :, :, : ):
    // the q axis (axis 0) is a partition like every other, so this rank owns q=iq iff
    // iq lies in its local_range(0); if so it writes its (t,P,Q) block at
    // local_range(1..3) (indexing dW.local() at the local q offset iq - origin(0)),
    // else it contributes zero. The all_reduce(plus) over the zero-padded buffer is the
    // pure GATHER of that one slice -- bit-identical on every rank, and to slicing the
    // all-q replicated array. Lets the secondary fold hold only one q of tau-domain W.
    template<typename dArray_t, typename comm_t>
    nda::array<ComplexType, 3>
    gather_dW_one_q(dArray_t const& dW, comm_t& comm, long iq, long nt_half, long Np) {
      auto gs = dW.global_shape();
      utils::check(gs[1] == nt_half and gs[2] == Np and gs[3] == Np and
                   iq >= 0 and iq < gs[0],
                   "vertex_redist_detail::gather_dW_one_q: bad dW global shape "
                   "({}, {}, {}, {}) or iq = {}; expected (*, {}, {}, {}) with 0 <= iq "
                   "< {}.", gs[0], gs[1], gs[2], gs[3], iq, nt_half, Np, Np, gs[0]);
      nda::array<ComplexType, 3> W_q(nt_half, Np, Np);
      W_q() = ComplexType(0.0);
      const long q0 = dW.origin()[0];
      const long nq_loc = dW.local_shape()[0];
      if (iq >= q0 and iq < q0 + nq_loc)
        W_q(dW.local_range(1), dW.local_range(2), dW.local_range(3)) =
            dW.local()(iq - q0, nda::ellipsis{});
      comm.all_reduce_in_place_n(W_q.data(), W_q.size(), std::plus<>{});
      return W_q;
    }

    /**
     * REDUCE-SCATTER. The Pi^C kernel produces a PARTIAL replicated
     * `part` (each rank holds its round-robin tuple/q_ext contribution to the WHOLE
     * (t, q, P, Q) array). The RPA output grid dPi splits t (ntpools) and P,Q (np_P x
     * np_Q) but NOT q, so every output block is owned by exactly ONE rank. The correct
     * sum-scatter is therefore a reduce onto each block owner -- an MPI_Reduce_scatter,
     * which the mpi3 wrapper lacks, so it is composed from per-owner MPI_Reduce.
     *
     * No full-array all_reduce is needed, and only ONE transient full `part` (freed by
     * the caller) + the owned block are alive.
     *
     * `part` is the FULL-shape partial (upfolded + tau-converted on the partial -- valid
     * because upfold+tau are LINEAR and commute with the rank sum). dPi.local() receives
     * the summed owned block. rank r gets sum_over_ranks part[block_r]. On one rank this
     * is a bit-identical copy.
     */
    template<typename dArray_t, typename comm_t>
    void reduce_scatter_into(nda::array<ComplexType, 4> const& part, dArray_t& dPi,
                             comm_t& comm) {
      const long np = comm.size();
      // gather every rank's owned block (origin[4] + local_shape[4]) -- 8 longs each.
      std::array<long, 8> mine{};
      for (int d = 0; d < 4; ++d) { mine[d] = dPi.origin()[d]; mine[4 + d] = dPi.local_shape()[d]; }
      nda::array<long, 2> boxes(np, 8);
      comm.all_gather_n(mine.data(), 8, boxes.data());
      for (long r = 0; r < np; ++r) {
        nda::range r0(boxes(r, 0), boxes(r, 0) + boxes(r, 4));
        nda::range r1(boxes(r, 1), boxes(r, 1) + boxes(r, 5));
        nda::range r2(boxes(r, 2), boxes(r, 2) + boxes(r, 6));
        nda::range r3(boxes(r, 3), boxes(r, 3) + boxes(r, 7));
        // contiguous copy of r's block from THIS rank's partial, reduce onto root r.
        nda::array<ComplexType, 4> buf = part(r0, r1, r2, r3);
        comm.reduce_in_place_n(buf.data(), buf.size(), std::plus<>{}, int(r));
        if (r == comm.rank()) dPi.local() = buf;
      }
    }

    /**
     * SLAB variant (for the Pi^C slab accumulator): `part` holds ONLY this
     * rank's owned q rows -- axis 1 is the slab, `slab_of` maps global iq -> slab row
     * (-1 when not owned). Each destination block is packed from the slab where owned
     * and ZEROS elsewhere -- the same math as a full-shape partial, whose non-owned rows
     * are structurally zero -- then reduced onto its owner exactly as
     * reduce_scatter_into. On one rank (every row owned) this is a bit-identical copy.
     */
    template<typename dArray_t, typename comm_t>
    void reduce_scatter_slab_into(nda::array<ComplexType, 4> const& part,
                                  std::vector<long> const& slab_of,
                                  dArray_t& dPi, comm_t& comm) {
      decltype(nda::range::all) all;
      const long np = comm.size();
      std::array<long, 8> mine{};
      for (int d = 0; d < 4; ++d) { mine[d] = dPi.origin()[d]; mine[4 + d] = dPi.local_shape()[d]; }
      nda::array<long, 2> boxes(np, 8);
      comm.all_gather_n(mine.data(), 8, boxes.data());
      for (long r = 0; r < np; ++r) {
        nda::range r0(boxes(r, 0), boxes(r, 0) + boxes(r, 4));
        nda::range r2(boxes(r, 2), boxes(r, 2) + boxes(r, 6));
        nda::range r3(boxes(r, 3), boxes(r, 3) + boxes(r, 7));
        nda::array<ComplexType, 4> buf(boxes(r, 4), boxes(r, 5), boxes(r, 6), boxes(r, 7));
        buf() = ComplexType(0.0);
        for (long jq = 0; jq < boxes(r, 5); ++jq) {
          const long gq = boxes(r, 1) + jq;
          const long sq = slab_of[gq];
          if (sq < 0) continue;                       // not owned: stays zero
          buf(all, jq, all, all) = part(r0, sq, r2, r3);
        }
        comm.reduce_in_place_n(buf.data(), buf.size(), std::plus<>{}, int(r));
        if (r == comm.rank()) dPi.local() = buf;
      }
    }

  } // vertex_redist_detail

  /**
   * Secondary-basis helpers: the secondary ISDF basis on the correlated subspace C and
   * the per-q transfer maps
   *   s(q) = B(q)^dag B(q),  t(q) = s(q)^+ B(q)^dag C(q),
   *   downfold Wbar = t W t^dag, upfold Pi = t^dag Pibar t (mutual adjoints, so
   *   <Pibar, Wbar> = <Pi, W> holds algebraically: nothing leaks between the two bases).
   */
  namespace vertex_secondary_detail {

    /**
     * Pair-collocation matrix at transfer q, in the kernels' in/out convention
     * (P-side pairs, k_in = k - q):
     *   rows I = ((is*nk + ik)*nc + o)*nc + i, o/i in the window [orb0, orb0 + nc):
     *   A(I, u) = X(is, kmq(iq, ik), u, orb0 + i) * conj(X(is, ik, u, orb0 + o)).
     * The same routine builds B(q) (from the secondary collocation, orb0 = 0) and
     * C(q) (from the global collocation, orb0 = C.first()) -- one code path, so the
     * two matrices are convention-consistent by construction.
     */
    inline void build_pair_matrix(nda::MemoryArrayOfRank<4> auto const& X_skua,
                                  long orb0, long nc,
                                  nda::array<long, 2> const& kmq, long iq,
                                  nda::array<ComplexType, 2>& A_Iu) {
      const long ns = X_skua.shape(0), nk = X_skua.shape(1), naux = X_skua.shape(2);
      utils::check(A_Iu.shape(0) == ns * nk * nc * nc and A_Iu.shape(1) == naux,
                   "vertex_secondary_detail::build_pair_matrix: shape mismatch.");
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nk; ++ik) {
          const long ikin = kmq(iq, ik);
          for (long o = 0; o < nc; ++o)
            for (long i = 0; i < nc; ++i) {
              const long I = ((is * nk + ik) * nc + o) * nc + i;
              for (long u = 0; u < naux; ++u)
                A_Iu(I, u) = X_skua(is, ikin, u, orb0 + i) *
                             std::conj(X_skua(is, ik, u, orb0 + o));
            }
        }
    }

    // fold the core: out(m, n) = [t A t^dag](m, n); tmp is (N_m, Np)
    inline void fold_core(nda::MemoryArrayOfRank<2> auto const& t_mP,
                          nda::MemoryArrayOfRank<2> auto const& A_PQ,
                          nda::array<ComplexType, 2>& tmp_mQ,
                          nda::MemoryArrayOfRank<2> auto&& out_mn) {
      nda::blas::gemm(t_mP, A_PQ, tmp_mQ);
      nda::blas::gemm(tmp_mQ, nda::dagger(t_mP), out_mn);
    }

    // upfold: out(P, Q) = [t^dag Pibar t](P, Q); tmp is (Np, N_m)
    inline void upfold_core(nda::MemoryArrayOfRank<2> auto const& t_mP,
                            nda::MemoryArrayOfRank<2> auto const& Pi_mn,
                            nda::array<ComplexType, 2>& tmp_Pn,
                            nda::MemoryArrayOfRank<2> auto&& out_PQ) {
      nda::blas::gemm(nda::dagger(t_mP), Pi_mn, tmp_Pn);
      nda::blas::gemm(tmp_Pn, t_mP, out_PQ);
    }

    /**
     * Downfold residual eta(q, .) for one core matrix A:
     *   eta = || B (t A t^dag) B^dag - C A C^dag ||_F / || C A C^dag ||_F,
     * with B t A t^dag B^dag = (B t) A (B t)^dag. Test-scale diagnostic
     * (N_pair x N_pair matrices are formed).
     */
    inline double eta_of(nda::array<ComplexType, 2> const& B_Im,
                         nda::array<ComplexType, 2> const& C_IP,
                         nda::MemoryArrayOfRank<2> auto const& t_mP,
                         nda::MemoryArrayOfRank<2> auto const& A_PQ) {
      const long Npair = B_Im.shape(0), Np = C_IP.shape(1);
#if defined(ENABLE_DEVICE)
      // Device path: at large N_pair this diagnostic costs O(N_pair^2 Np) complex MACs per (q, slice), replicated on
      // every rank. On the device: the same four gemms, D = WA - WC by a beta = -1 gemm, the two Frobenius norms as
      // dotc. Agrees with the host path to rounding.
      if (vertex_debug::number("eta_device", 1.0) != 0.0) {   // vertex_debug: eta_device
        using dmat_t = memory::array<DEVICE_MEMORY, ComplexType, 2>;
        auto Bd = memory::to_memory_space<DEVICE_MEMORY>(B_Im);
        auto Cd = memory::to_memory_space<DEVICE_MEMORY>(C_IP);
        nda::array<ComplexType, 2> th(t_mP), Ah(A_PQ);
        auto td = memory::to_memory_space<DEVICE_MEMORY>(th);
        auto Ad = memory::to_memory_space<DEVICE_MEMORY>(Ah);
        dmat_t Cfd(Npair, Np), Ed(Npair, Np), Dd(Npair, Npair);
        nda::blas::gemm(Bd, td, Cfd);                                                   // fitted pair rows B t
        nda::blas::gemm(Cd, Ad, Ed);
        nda::blas::gemm(Ed, nda::dagger(Cd), Dd);                                       // WC
        auto Df = nda::reshape(Dd, std::array<long, 1>{Npair * Npair});
        const double den = std::real(nda::blas::dotc(Df, Df));
        nda::blas::gemm(Cfd, Ad, Ed);
        nda::blas::gemm(ComplexType(1.0), Ed, nda::dagger(Cfd), ComplexType(-1.0), Dd);  // WA - WC
        const double num = std::real(nda::blas::dotc(Df, Df));
        return std::sqrt(num) / std::max(std::sqrt(den), 1e-300);
      }
#endif
      nda::array<ComplexType, 2> Cf(Npair, Np);        // fitted pair rows B t
      nda::blas::gemm(B_Im, t_mP, Cf);
      nda::array<ComplexType, 2> E(Npair, Np), WC(Npair, Npair), WA(Npair, Npair);
      nda::blas::gemm(C_IP, A_PQ, E);
      nda::blas::gemm(E, nda::dagger(C_IP), WC);
      nda::blas::gemm(Cf, A_PQ, E);
      nda::blas::gemm(E, nda::dagger(Cf), WA);
      double num = 0.0, den = 0.0;
      for (long I = 0; I < Npair; ++I)
        for (long J = 0; J < Npair; ++J) {
          num += std::norm(WA(I, J) - WC(I, J));
          den += std::norm(WC(I, J));
        }
      return std::sqrt(num) / std::max(std::sqrt(den), 1e-300);
    }

    /**
     * eta(q) sweep over all q for one labeled core slice family (core(iq) must return
     * a (Np, Np) view of the GLOBAL rung array actually consumed by the kernels --
     * head-augmented under the gygi policy). Per-q values at app_log(3), the max at
     * app_log(2). Returns max_q eta.
     */
    template<nda::MemoryArrayOfRank<4> XArr, typename CoreF>
    double eta_max_over_q(char const* label,
                          XArr const& X_skPa, long orb0, long nc,
                          nda::array<ComplexType, 4> const& Xb_skma,
                          nda::array<ComplexType, 3> const& t_qmP,
                          nda::array<long, 2> const& kmq, CoreF&& core) {
      decltype(nda::range::all) all;
      const long nq = t_qmP.shape(0), Nm = t_qmP.shape(1), Np = t_qmP.shape(2);
      const long ns = X_skPa.shape(0), nk = X_skPa.shape(1);
      const long Npair = ns * nk * nc * nc;
      nda::array<ComplexType, 2> B_Im(Npair, Nm), C_IP(Npair, Np);
      double mx = 0.0;
      for (long iq = 0; iq < nq; ++iq) {
        build_pair_matrix(Xb_skma, 0, nc, kmq, iq, B_Im);
        build_pair_matrix(X_skPa, orb0, nc, kmq, iq, C_IP);
        double e = eta_of(B_Im, C_IP, t_qmP(iq, all, all), core(iq));
        app_log(3, "    Refinement 2 eta[{}](q = {}) = {}", label, iq, e);
        mx = std::max(mx, e);
      }
      app_log(2, "  Refinement 2 downfold residual (Eq. 40): max_q eta[{}] = {}", label, mx);
      return mx;
    }

  } // vertex_secondary_detail

  /**
   * IBZ k-point symmetry helpers.
   */
  namespace vertex_ibz_detail {

    // crystal-coordinate comparison mod integer G-vectors (same class as the
    // matching in generate_qsymm_maps, symmetry.hpp)
    inline bool same_kpt_mod_G(nda::ArrayOfRank<1> auto const& a,
                               nda::ArrayOfRank<1> auto const& b) {
      for (int i = 0; i < 3; ++i) {
        double d = a(i) - b(i);
        d -= std::round(d);
        if (std::abs(d) > 1e-6) return false;
      }
      return true;
    }

    /**
     * G-rotation consistency diagnostic: for each symmetry
     * position js >= 1 and a sample of full-BZ k, measure
     *   || G_CC(k) - Dc(js,k)^dag G_CC(krot(js,k)) Dc(js,k) ||_F / ||G_CC(k)||_F
     * on one tau slice. O(leakage) is expected (window truncation of the exact
     * covariance); a much larger value indicates a map/conjugation bug. Non-trev
     * k only (the trev gauge composition is exercised by the kernels themselves).
     */
    inline double g_rotation_check(vertex_sym::sym_ctx const& ctx,
                                 nda::MemoryArrayOfRank<5> auto const& G_full,
                                 nda::ArrayOfRank<1> auto const& kp_trev) {
      decltype(nda::range::all) all;
      const long nc = ctx.nc;
      const long it = G_full.shape(0) / 2, is = 0;
      nda::array<ComplexType, 2> T1(nc, nc), T2(nc, nc);
      double worst = 0.0;
      for (long js = 1; js < ctx.nsym; ++js) {
        for (long k = 0; k < ctx.nk_full; ++k) {
          if (kp_trev(k) or ctx.cjg(js, k)) continue;
          auto Dc = ctx.Dc(js, k, all, all);
          auto Gk = G_full(it, is, k, all, all);
          auto Gr = G_full(it, is, ctx.krot(js, k), all, all);
          nda::blas::gemm(ComplexType(1.0), nda::dagger(Dc), Gr, ComplexType(0.0), T1);
          nda::blas::gemm(ComplexType(1.0), T1, Dc, ComplexType(0.0), T2);
          double num = 0.0, den = 0.0;
          for (long a = 0; a < nc; ++a)
            for (long b = 0; b < nc; ++b) {
              num += std::norm(T2(a, b) - Gk(a, b));
              den += std::norm(Gk(a, b));
            }
          if (den > 1e-24) worst = std::max(worst, std::sqrt(num / den));
        }
      }
      app_log(2, "  IBZ symmetry: G-rotation consistency residual (C block, one tau "
                 "slice): max = {} (O(D-leakage) expected)", worst);
      return worst;
    }

  } // vertex_ibz_detail

  // vertex_wannier_detail: see vertex_wannier_detail.hpp (included before the .icc kernels)

  void vertex_t::set_wannier_projector(methods::projector_t const &proj, bool loewdin) {
    // The projector defines the C subspace for EITHER the Sigma^C vertex (enabled()) OR the
    // pol-vertex ladder (pol_vertex_enabled()); the two are mutually exclusive (they would double count),
    // and pol_vertex="ladder" is the coarse->fine interpolation path, so accept a pol-vertex-only vertex.
    utils::check(enabled() or pol_vertex_enabled(),
                 "vertex_t::set_wannier_projector: neither the Sigma^C vertex (vertex_type) nor the "
                 "pol-vertex ladder is configured; nothing to project onto.");
    utils::check(proj.nImps() == 1,
                 "vertex_t::set_wannier_projector: only a single impurity is supported "
                 "(nImps = {}); merge the shells in the wan.h5 reader.", proj.nImps());
    decltype(nda::range::all) all;

    // U = dagger(proj_mat) on the W_rng rows: the code's
    // proj_mat _C_skIai(s,k,0,a,i) = C_{a,i} downfolds as O_loc = C O_WW C^dag, so
    // U_{i,a} = conj(C_{a,i}) is the isometry with O_loc = U^dag O U.
    auto C = proj.C_skIai();                       // (ns, nk, 1, M, nOrbs_W)
    auto const &W_rng = proj.W_rng()[0];
    const long ns = C.shape(0), nk = C.shape(1), M = C.shape(3), nW = C.shape(4);
    utils::check(nW == W_rng.size(),
                 "vertex_t::set_wannier_projector: proj_mat window dim {} != W_rng size "
                 "{}.", nW, long(W_rng.size()));
    utils::check(M > 0 and M <= nW,
                 "vertex_t::set_wannier_projector: invalid projector rank M = {} "
                 "(window size {}).", M, nW);
    // Wannier mode REPLACES the band window by the projector's W_rng. An explicitly set (non-empty) window that differs
    // from W_rng is a contradiction in the input and aborts. An empty (default) window is filled, and the effective
    // window logged.
    utils::check(_band_window.size() == 0 or
                     (_band_window.first() == W_rng.first() and _band_window.last() == W_rng.last()),
                 "vertex_t::set_wannier_projector: vertex_band_window = [{}, {}) differs from the Wannier projector's band "
                 "window W_rng = [{}, {}) of {} (audit D8): in Wannier mode C is the projector's span on W_rng. Set "
                 "vertex_band_window = [{}, {}) or leave it empty.", _band_window.first(), _band_window.last(),
                 W_rng.first(), W_rng.last(), proj.C_file(), W_rng.first(), W_rng.last());
    // the pol-vertex ladder's window (pol_vertex_band_window, inheriting vertex_band_window) is the C window of the readout
    // instance that adopts this projector (scr_coulomb_t::ensure_pol_vertex -> adopt_wannier): same contradiction, caught here.
    utils::check(not pol_vertex_active() or
                     (_pol_band_window.first() == W_rng.first() and _pol_band_window.last() == W_rng.last()),
                 "vertex_t::set_wannier_projector: pol_vertex_band_window = [{}, {}) differs from the Wannier projector's band "
                 "window W_rng = [{}, {}) of {} (audit D8): the ladder readout runs on the projector's span on W_rng. Set "
                 "pol_vertex_band_window = [{}, {}).", _pol_band_window.first(), _pol_band_window.last(),
                 W_rng.first(), W_rng.last(), proj.C_file(), W_rng.first(), W_rng.last());
    if (_band_window.size() == 0)
      app_log(1, "  [audit D8] vertex_band_window was empty: the effective vertex window is the Wannier projector's W_rng = "
                 "[{}, {}).", W_rng.first(), W_rng.last());

    _wannier = true;
    _M = M;
    _band_window = W_rng;                            // injection/storage support
    _wannier_file = proj.C_file();
    _U_skia = nda::array<ComplexType, 4>(ns, nk, nW, M);
    for (long is = 0; is < ns; ++is)
      for (long ik = 0; ik < nk; ++ik)
        for (long i = 0; i < nW; ++i)
          for (long a = 0; a < M; ++a)
            _U_skia(is, ik, i, a) = std::conj(C(is, ik, 0, a, i));

    // Loewdin-orthonormalize per (s,k); measure ||U^dag U - 1|| before the correction
    double defect_max = 0.0;
    for (long is = 0; is < ns; ++is)
      for (long ik = 0; ik < nk; ++ik) {
        double d = vertex_wannier_detail::loewdin_block(_U_skia(is, ik, all, all), loewdin);
        defect_max = std::max(defect_max, d);
      }
    _iso_defect = defect_max;

    app_log(1, "\n  Vertex subspace C = Wannier projector P = U U^dag "
               "(notes/wannier_projector_theory.md)\n"
               "  ------------------------------------------------------------------\n"
               "  Wannier file             = {}\n"
               "  Subspace rank M          = {} orbitals\n"
               "  Injection window W_rng   = [{}, {})  ({} bands)\n"
               "  Isometry defect max_sk ||U^dag U - 1||_F (before Loewdin) = {:.3e}\n"
               "  Loewdin orthonormalize   = {} (owner ruling Q1)\n",
            _wannier_file.empty() ? "(in-memory projector)" : _wannier_file,
            M, _band_window.first(), _band_window.last(), nW, defect_max,
            loewdin ? "yes" : "no");
    if (not loewdin and defect_max > 1e-8)
      app_log(1, "  [WARNING] set_wannier_projector: Loewdin skipped and the raw "
                 "isometry defect is {:.3e} > 1e-8:\n"
                 "            P is only approximately idempotent, so the subspace "
                 "interpretation, the\n"
                 "            q->0 head delta_ab reduction, and gauge invariance "
                 "degrade at O(defect)\n"
                 "            (memo section 1.3). Prefer the default Loewdin path.\n",
              defect_max);
  }

  void vertex_t::build_sym_ctx(THC_ERI auto const &thc,
                               nda::MemoryArrayOfRank<4> auto const &X_w,
                               long C0_global,
                               std::optional<vertex_sym::sym_ctx> &slot,
                               nda::array<ComplexType, 4> const *U_skia) {
    if (slot.has_value()) return;   // geometry-fixed
    // After the lazy guard (see SEC_BASIS): counts REAL builds only. INCLUSIVE in the
    // entry point that triggered it. Note build_sym_ctx is called for TWO slots (global
    // and secondary), so ncalls up to 2 per run is expected, not a leak.
    vertex_timer_detail::scoped_timer _tm_total(_Timer, "SYM_CTX");
    decltype(nda::range::all) all;
    auto MF = thc.MF();
    const bool wan = (U_skia != nullptr);

    vertex_sym::sym_ctx ctx;
    ctx.active = true;
    ctx.ns = X_w.shape(0);
    ctx.nk_full = MF->nkpts();
    ctx.nk_ibz = MF->nkpts_ibz();
    ctx.nq_full = MF->nqpts();
    ctx.nq_ibz = MF->nqpts_ibz();
    ctx.naux = X_w.shape(2);
    ctx.nc = X_w.shape(3);                 // = M (Wannier) or window size (window mode)
    utils::check(X_w.shape(1) == ctx.nk_full,
                 "vertex_t::build_sym_ctx: X_w must carry the FULL BZ k axis "
                 "({} vs {}).", X_w.shape(1), ctx.nk_full);
    const long nbnd = MF->nbnd();
    const long nc = ctx.nc;
    // nW = the D-window band span. WINDOW: nW = nc (the C rows in band basis).
    // WANNIER: nW = W_rng.size(), the injection support, and the
    // C-sector rotation is d = U(Sk)^dag D_win U(k) (M x M), NOT the band block.
    const long nW = wan ? U_skia->shape(2) : nc;
    utils::check(C0_global >= 0 and C0_global + nW <= nbnd,
                 "vertex_t::build_sym_ctx: invalid window [{}, {}).", C0_global, C0_global + nW);
    if (wan)
      utils::check(U_skia->shape(0) == ctx.ns and U_skia->shape(1) == ctx.nk_full and
                   U_skia->shape(3) == nc,
                   "vertex_t::build_sym_ctx: U_skia shape mismatch with X_w.");

    auto qsymms = MF->qsymms();
    ctx.nsym = qsymms.extent(0);
    auto kp_trev_pair = MF->kp_trev_pair();

    // ---- q'-access tables ------------------------------------------------------------
    ctx.q_isym = nda::array<long, 1>(ctx.nq_full);
    ctx.q_star = nda::array<long, 1>(ctx.nq_full);
    ctx.q_trev = nda::array<bool, 1>(ctx.nq_full);
    for (long iq = 0; iq < ctx.nq_full; ++iq) {
      const int sidx = MF->qp_symm(iq);
      long js = -1;
      for (long i = 0; i < ctx.nsym; ++i)
        if (qsymms(i) == sidx) { js = i; break; }
      utils::check(js >= 0, "vertex_t::build_sym_ctx: qp_symm({}) = {} not found in "
                            "qsymms.", iq, sidx);
      ctx.q_isym(iq) = js;
      ctx.q_star(iq) = MF->qp_to_ibz(iq);
      ctx.q_trev(iq) = MF->qp_trev(iq);
    }
    for (long iq = 0; iq < ctx.nq_ibz; ++iq)
      utils::check(ctx.q_star(iq) == iq and not ctx.q_trev(iq) and ctx.q_isym(iq) == 0,
                   "vertex_t::build_sym_ctx: IBZ q-point {} is not identity-mapped "
                   "(star = {}, trev = {}, isym = {}).",
                   iq, ctx.q_star(iq), int(ctx.q_trev(iq)), ctx.q_isym(iq));

    // ---- momentum map: krot = ks_to_k (full-BZ rows; direction checked below) ---------
    ctx.krot = nda::array<long, 2>(ctx.nsym, ctx.nk_full);
    for (long is = 0; is < ctx.nsym; ++is)
      for (long ik = 0; ik < ctx.nk_full; ++ik)
        ctx.krot(is, ik) = MF->ks_to_k(int(is), int(ik));
    ctx.ktrev_pair = nda::array<long, 1>(ctx.nk_full);
    for (long ik = 0; ik < ctx.nk_full; ++ik) ctx.ktrev_pair(ik) = long(kp_trev_pair(ik));
    {
      // the complete -k map (crystal coordinates, mod G): kp_trev_pair is -1 wherever the point is not a time-reversal image
      auto kc = MF->kpts_crystal();
      ctx.kminus = nda::array<long, 1>(ctx.nk_full);
      ctx.kminus() = -1;
      for (long ik = 0; ik < ctx.nk_full; ++ik) {
        for (long jk = 0; jk < ctx.nk_full and ctx.kminus(ik) < 0; ++jk) {
          double d = 0.0;
          for (int a = 0; a < 3; ++a) { const double x = kc(ik, a) + kc(jk, a); d += std::abs(x - std::round(x)); }
          if (d < 1e-8) ctx.kminus(ik) = jk;
        }
        utils::check(ctx.kminus(ik) >= 0, "vertex_t::build_sym_ctx: the k mesh has no -k for k-point {} (not a Gamma-centered lattice).", ik);
        utils::check(kp_trev_pair(ik) < 0 or long(kp_trev_pair(ik)) == ctx.kminus(ik),
                     "vertex_t::build_sym_ctx: kp_trev_pair({}) = {} is not -k = {}.", ik, kp_trev_pair(ik), ctx.kminus(ik));
      }
    }
    ctx.qminus = nda::array<long, 1>(ctx.nq_full);
    {
      auto qm = MF->qminus();
      for (long iq = 0; iq < ctx.nq_full; ++iq) ctx.qminus(iq) = long(qm(iq));
    }

    // direction self-check: the same map on the Q mesh must send q' -> +/- qs.
    // (slist = find_inverse_symmetry(qsymms) in the MF makes the D-pair point of k
    //  exactly ks_to_k(js, k); assert rather than trust.)
    // NOTE: symm_op.R acts on CRYSTAL coordinates (generate_dmatrix works on
    // kpts_crystal, symmetry.hpp); MF->Qpts() is CARTESIAN, so the crystal
    // q list is built self-consistently from kpts_crystal differences via qk_to_k2
    // (bz convention Qpts[q] + G = kpts[a] - kpts[b], bz_symmetry.hpp).
    {
      auto slist_ops = MF->symm_list();
      auto kcrys = MF->kpts_crystal();
      nda::array<double, 2> qcrys(ctx.nq_full, 3);
      for (long iq = 0; iq < ctx.nq_full; ++iq) {
        const long k2 = MF->qk_to_k2(int(iq), 0);   // k0 - q (mod G)
        for (int i = 0; i < 3; ++i) qcrys(iq, i) = kcrys(0, i) - kcrys(k2, i);
      }
      nda::stack_array<double, 3> qrot_v, qtgt;
      for (long iq = 0; iq < ctx.nq_full; ++iq) {
        const long js = ctx.q_isym(iq);
        const long qs = ctx.q_star(iq);
        if (js == 0 and not ctx.q_trev(iq)) continue;
        auto const& R = slist_ops[qsymms(js)].R;
        // image = q' * R (row-vector right action, as in the generate_qsymm_maps
        // matching, symmetry.hpp)
        nda::blas::gemv(1.0, nda::transpose(R), qcrys(iq, all), 0.0, qrot_v);
        const double sgn = ctx.q_trev(iq) ? -1.0 : 1.0;
        for (int i = 0; i < 3; ++i) qtgt(i) = sgn * qcrys(qs, i);
        utils::check(vertex_ibz_detail::same_kpt_mod_G(qrot_v, qtgt),
                     "vertex_t::build_sym_ctx: rung-transfer direction check FAILED at "
                     "q' = {} (isym pos {}, qs = {}, trev = {}): q'*R (crystal) = "
                     "({}, {}, {}) vs target ({}, {}, {}). The MF symmetry conventions "
                     "deviate from the derivation in notes/vertex_ibz_symmetry.md "
                     "section 3.1 -- refusing to rotate the wrong way.",
                     iq, js, qs, int(ctx.q_trev(iq)),
                     qrot_v(0), qrot_v(1), qrot_v(2), qtgt(0), qtgt(1), qtgt(2));
      }
    }

    // ---- effective columns Xhat + C-window D blocks + leakage diagnostic -------------
    // Xhat is NODE-SHARED (one copy per NUMA node) and the (js, ik) build loop is
    // DISTRIBUTED across node_comm -- each (js, ik) tile is computed on exactly one
    // node-rank and written into the shared window (a partition => exact GATHER).
    // Dc/cjg stay per-rank (small: nsym*nk*nc^2) but are ALSO filled only on the owning
    // rank and node-gathered (zero-init + all_reduce = exact). The leakage scalars are a
    // per-(js,ik) sum, node-reduced. On one rank per node this is bit-identical.
    auto mpi = thc.mpi();
    ctx.Xhat_shm = std::make_shared<math::shm::shared_array<nda::array_view<ComplexType, 5>>>(
        math::shm::make_shared_array<nda::array_view<ComplexType, 5>>(
            *mpi, std::array<long, 5>{ctx.ns, ctx.nsym, ctx.nk_full, ctx.naux, nc}));
    ctx.Xhat.rebind(ctx.Xhat_shm->local());   // ctor zero-inits; identity slot filled below
    ctx.Dc = nda::array<ComplexType, 4>(ctx.nsym, ctx.nk_full, nc, nc);
    ctx.Dc() = ComplexType(0.0);
    ctx.cjg = nda::array<bool, 2>(ctx.nsym, ctx.nk_full);
    ctx.cjg() = false;
    // identity slot js = 0: node-shared write, sharded over node_comm ranks by ik.
    ctx.Xhat_shm->win().fence();
    for (long ik = mpi->node_comm.rank(); ik < ctx.nk_full; ik += mpi->node_comm.size())
      for (long is = 0; is < ctx.ns; ++is)
        ctx.Xhat(is, 0, ik, all, all) = X_w(is, ik, all, all);
    ctx.Xhat_shm->win().fence();

    double leak_max = 0.0, leak_sum = 0.0, dunit_max = 0.0;
    long leak_cnt = 0;
    {
      // column selector E(nbnd, nW) of the W-window band block; Dcols = D * E (nbnd, nW)
      nda::array<ComplexType, 2> E(nbnd, nW), Dcols(nbnd, nW);
      nda::array<ComplexType, 2> base(ctx.naux, nc);
      // Wannier scratch: DU(nbnd, M) = Dcols . U(k) (rows W_rng), d(M,M) = U(Sk)^dag DU
      nda::array<ComplexType, 2> DU(nbnd, nc), dW_win(nW, nc);
      E() = ComplexType(0.0);
      for (long j = 0; j < nW; ++j) E(C0_global + j, j) = ComplexType(1.0);
      using math::sparse::csrmm;
      // distribute the (js, ik) tiles (js >= 1) over node_comm; write Xhat into the shared
      // window, Dc/cjg into the (zeroed) per-rank arrays -- each tile touched once.
      ctx.Xhat_shm->win().fence();
      const long njk = (ctx.nsym - 1) * ctx.nk_full;
      for (long jk = mpi->node_comm.rank(); jk < njk; jk += mpi->node_comm.size()) {
        const long js = 1 + jk / ctx.nk_full;
        const long ik = jk % ctx.nk_full;
        {
          auto [cj, Dsp] = MF->symmetry_rotation(js, ik);
          ctx.cjg(js, ik) = cj;
          csrmm<'N'>(ComplexType(1.0), *Dsp, E, ComplexType(0.0), Dcols);
          const long ksrc = ctx.krot(js, cj ? long(kp_trev_pair(ik)) : ik);
          // C-window leakage of this rotation; the PLAIN block is kept, with no extra
          // normalization (as in projector_boson_t.cpp).
          // WANNIER: the projector-level leakage ||(1 - P(Sk)) D U(k)|| = mass of D U(k)
          // falling outside range(P(Sk)); 0 for a symmetry-closed Wannier set.
          auto Dc = ctx.Dc(js, ik, all, all);
          if (not wan) {
            double m_in = 0.0, m_all = 0.0;
            for (long a = 0; a < nbnd; ++a)
              for (long j = 0; j < nc; ++j) {
                const double w = std::norm(Dcols(a, j));
                m_all += w;
                if (a >= C0_global and a < C0_global + nc) m_in += w;
              }
            if (m_all > 1e-24) {
              const double leak = 1.0 - m_in / m_all;
              leak_max = std::max(leak_max, leak);
              leak_sum += leak;
              ++leak_cnt;
            }
            for (long a = 0; a < nc; ++a)
              for (long j = 0; j < nc; ++j) Dc(a, j) = Dcols(C0_global + a, j);
          }
          // UNITARITY defect of the C-sector rotation actually used, ||Dc^dag Dc - 1||_F.
          // This is NOT the same as the leakage above: leak measures how much of D.E
          // falls outside the window among the RETAINED nbnd rows and is normalized by
          // that retained mass, so it is blind to weight lost past the nbnd truncation
          // and to the row-renormalization generate_dmatrix applies there
          // (symmetry.hpp). Xhat = X(ksrc).Dc enters EIGHT collocation legs of
          // Sigma^C and four of Pi^C, so this defect is the accuracy floor of the whole
          // symmetry path.
          {
            double d2 = 0.0;
            for (long a = 0; a < nc; ++a)
              for (long b = 0; b < nc; ++b) {
                ComplexType s(0.0, 0.0);
                for (long p = 0; p < nc; ++p) s += std::conj(Dc(p, a)) * Dc(p, b);
                d2 += std::norm(s - ((a == b) ? ComplexType(1.0) : ComplexType(0.0)));
              }
            dunit_max = std::max(dunit_max, std::sqrt(d2));
          }
          // effective columns: base collocation at the D-pair point
          // (the trev pair's rotation for trev k -- the API redirect), conj for trev.
          // WANNIER: base = X_bar(ksrc) . d(k;S), d = U(ksrc)^dag D_win U(ik) (M x M).
          nda::array<ComplexType, 2> dloc(nc, nc);   // per-spin C-sector rotation
          for (long is = 0; is < ctx.ns; ++is) {
            if (wan) {
              // DU(nbnd, M) = Dcols(nbnd, nW) . U(ik)(nW, M)
              // For a CONJUGATED rotation (cj) the kernel applies conj to the whole effective column X_bar(ksrc) dloc, so
              // the C-sector block must be dloc = U(ksrc)^dag D conj(U(k')): with U(k') unconjugated the Wannier point frame
              // would lose its gauge invariance for any complex U on meshes with time-reversal images (real U and TRIM-only
              // meshes would be unaffected).
              if (cj) {
                auto Uc = nda::make_regular(nda::conj((*U_skia)(is, ik, all, all)));
                nda::blas::gemm(Dcols, Uc, DU);
              } else {
                nda::blas::gemm(Dcols, (*U_skia)(is, ik, all, all), DU);
              }
              // d(M, M) = U(ksrc)^dag(M, nW) . DU[W_rng rows](nW, M)
              for (long p = 0; p < nW; ++p)
                for (long a = 0; a < nc; ++a) dW_win(p, a) = DU(C0_global + p, a);
              nda::blas::gemm(nda::dagger((*U_skia)(is, ksrc, all, all)), dW_win, dloc);
              // projector-level leakage: 1 - ||P(ksrc) DU||^2 / ||DU||^2
              if (is == 0) {
                Dc = dloc;                               // store the is=0 rotation
                double m_all = 0.0, m_in = 0.0;
                for (long a = 0; a < nc; ++a) {
                  for (long p = 0; p < nbnd; ++p) m_all += std::norm(DU(p, a));
                  for (long b = 0; b < nc; ++b) m_in += std::norm(dloc(b, a));
                }
                if (m_all > 1e-24) {
                  const double leak = std::max(0.0, 1.0 - m_in / m_all);
                  leak_max = std::max(leak_max, leak);
                  leak_sum += leak; ++leak_cnt;
                }
              }
            }
            if (wan)
              nda::blas::gemm(ComplexType(1.0), X_w(is, ksrc, all, all), dloc,
                              ComplexType(0.0), base);
            else if (vertex_debug::flag("sym_dt"))
              // DIAGNOSTIC: the TRANSPOSED C-sector rotation in the effective columns, Xhat = X(ksrc) . Dc^T
              // (the degenerate-block convention question: identical for diagonal Dc; see vertex_sym.hpp sym_fold_dt)
              nda::blas::gemm(ComplexType(1.0), X_w(is, ksrc, all, all), nda::transpose(Dc),
                              ComplexType(0.0), base);
            else
              nda::blas::gemm(ComplexType(1.0), X_w(is, ksrc, all, all), Dc,
                              ComplexType(0.0), base);
            if (cj)
              for (long P = 0; P < ctx.naux; ++P)
                for (long j = 0; j < nc; ++j)
                  ctx.Xhat(is, js, ik, P, j) = std::conj(base(P, j));
            else
              ctx.Xhat(is, js, ik, all, all) = base;
          }
        }
      }
      ctx.Xhat_shm->win().fence();   // publish the node-shared Xhat tiles
    }
    // node-gather Dc/cjg (each tile written on one rank; zero-init + sum = exact GATHER)
    // and reduce the leakage scalars over the node (diagnostic only).
    if (mpi->node_comm.size() > 1) {
      mpi->node_comm.all_reduce_in_place_n(ctx.Dc.data(), ctx.Dc.size(), std::plus<>{});
      // cjg is bool; reduce via an int scratch with logical OR (each tile set once)
      nda::array<int, 2> cjg_i(ctx.nsym, ctx.nk_full);
      for (long a = 0; a < ctx.nsym; ++a)
        for (long b = 0; b < ctx.nk_full; ++b) cjg_i(a, b) = ctx.cjg(a, b) ? 1 : 0;
      mpi->node_comm.all_reduce_in_place_n(cjg_i.data(), cjg_i.size(), std::plus<>{});
      for (long a = 0; a < ctx.nsym; ++a)
        for (long b = 0; b < ctx.nk_full; ++b) ctx.cjg(a, b) = (cjg_i(a, b) != 0);
      leak_max = mpi->node_comm.all_reduce_value(leak_max, boost::mpi3::max<>{});
      dunit_max = mpi->node_comm.all_reduce_value(dunit_max, boost::mpi3::max<>{});
      leak_sum = mpi->node_comm.all_reduce_value(leak_sum, std::plus<>{});
      leak_cnt = mpi->node_comm.all_reduce_value(leak_cnt, std::plus<>{});
    }
    ctx.d_unitarity_max = dunit_max;
    _sym_d_unitarity_max = std::max(_sym_d_unitarity_max, dunit_max);
    ctx.leak_max = leak_max;
    ctx.leak_mean = (leak_cnt > 0) ? leak_sum / double(leak_cnt) : 0.0;
    _sym_leak_max = std::max(_sym_leak_max, ctx.leak_max);
    _sym_leak_mean = ctx.leak_mean;

    app_log(1, "\n  IBZ symmetry context READY (notes/vertex_ibz_symmetry.md): "
               "nk {} -> {} IBZ, nq {} -> {} IBZ, {} symmetry ops, naux = {}\n"
               "  C-window D-matrix leakage out of C = [{}, {}): max = {:.3e}, "
               "mean = {:.3e}\n"
               "  [NOTE] expected to be small; symmetry-unfolded vertex quantities "
               "carry O(leakage)\n"
               "         relative error -- the C-window analogue of the nbnd "
               "truncation warning in\n"
               "         generate_dmatrix (symmetry.hpp:1084-1092). No abort "
               "(theory-owner ruling).\n",
            ctx.nk_full, ctx.nk_ibz, ctx.nq_full, ctx.nq_ibz, ctx.nsym, ctx.naux,
            C0_global, C0_global + nW, ctx.leak_max, ctx.leak_mean);
    app_log(1, "  C-sector rotation UNITARITY defect max ||Dc^dag Dc - 1||_F = {:.3e}\n"
               "  [NOTE] Xhat = X(ksrc).Dc feeds 8 collocation legs of Sigma^C and 4 of "
               "Pi^C, so this is\n"
               "         the accuracy floor of the symmetry path (distinct from the "
               "leakage above).\n", ctx.d_unitarity_max);
    if (wan)
      app_log(1, "  (Wannier mode: leakage is the projector-level "
                 "||(1 - P(Sk)) D U(k)||^2; 0 for a symmetry-closed set, memo 2.8)\n");
    if (ctx.leak_max > 1e-2)
      app_log(1, "  [WARNING] C-window D-matrix leakage max = {:.3e} > 1e-2: the "
                 "window cuts deeply\n"
                 "            through an irreducible/degenerate block; consider a "
                 "window aligned with\n"
                 "            degenerate sets if higher symmetry fidelity is needed.\n",
              ctx.leak_max);

    slot = std::move(ctx);
  }

  void vertex_t::check_rung_implemented(std::string_view where) const {
    // All three rung modes are implemented, so there is nothing to reject. Each mode
    // assembles ALL of its own cuts in one place (eval_Sigma_C), so the non-conserving
    // half-theories are structurally unrepresentable -- e.g. Sigma^{C,x} without
    // Sigma^{C,r} is not the derivative of any functional once W0 is rebuilt from the
    // current G each iteration:
    //   dynamic : Sigma^C (G^3W^2) + Pi^C (G^4W)
    //   static  : Sigma^{C,x} + Sigma^{C,r};  P = P_RPA (Pi^C injection off at the update_w seam)
    //   linear  : the three explicit terms + Sigma^{L,r};  P = P_RPA + P^{C,L}
    (void)where;
  }

  void vertex_t::set_div_treatment(std::string div) {
    const std::unordered_set<std::string> exact = {"ignore_g0", "v1_skip"};
    utils::check(exact.count(div) > 0 or div.find("gygi") != std::string::npos,
                 "vertex_t: unknown vertex div_treatment: {}. Valid options are "
                 "\"ignore_g0\" (v2 default), \"gygi\"-class, and \"v1_skip\".", div);
    _div_treatment = std::move(div);
  }

  vertex_t::vertex_t(const imag_axes_ft::IAFT *ft,
                     std::string vertex_type,
                     nda::range band_window,
                     long nbnd,
                     std::string div_treatment,
                     std::string isdf_mode,
                     long isdf_rank,
                     double isdf_svd_tol,
                     double isdf_thresh,
                     double isdf_cond_max,
                     std::string rung):
    _ft(ft), _vertex_type(std::move(vertex_type)), _rung(string_to_vertex_rung_enum(rung)),
    _band_window(band_window),
    _isdf_mode(std::move(isdf_mode)), _isdf_rank(isdf_rank), _isdf_svd_tol(isdf_svd_tol),
    _isdf_thresh(isdf_thresh), _isdf_cond_max(isdf_cond_max) {

    const std::unordered_set<std::string> valid_vertex_types = {"none", "2nd_exchange"};
    utils::check(valid_vertex_types.find(_vertex_type) != valid_vertex_types.end(),
                 "vertex_t: unknown vertex_type: {}. Valid options are \"none\" and \"2nd_exchange\".",
                 _vertex_type);
    utils::check(_isdf_mode == "global" or _isdf_mode == "secondary",
                 "vertex_t: unknown vertex_isdf mode: {}. Valid options are \"global\" "
                 "(the original path) and \"secondary\" (Refinement 2, "
                 "notes/refinement2_optionA.md).", _isdf_mode);
    utils::check(_isdf_svd_tol >= 0.0 and _isdf_svd_tol < 1.0,
                 "vertex_t: invalid vertex_isdf_svd_tol = {}. Expect 0 <= tol < 1.",
                 _isdf_svd_tol);
    set_div_treatment(std::move(div_treatment));
    if (not enabled()) return;

    utils::check(_ft != nullptr, "vertex_t: IAFT instance is required when the vertex is enabled.");
    utils::check(_band_window.first() >= 0 and _band_window.first() <= _band_window.last(),
                 "vertex_t: invalid vertex_band_window = [{}, {}). Expect 0 <= first <= last.",
                 _band_window.first(), _band_window.last());
    utils::check(_band_window.last() <= nbnd,
                 "vertex_t: invalid vertex_band_window = [{}, {}). "
                 "The window must be within the primary basis: last <= nbnd = {}.",
                 _band_window.first(), _band_window.last(), nbnd);

    if (active()) {
      app_log(1, "\n"
                 "  Second-order exchange vertex correction (ISDF-Vertex)\n"
                 "  ------------------------------------------------------\n"
                 "  Vertex type              = {}\n"
                 "  Rung mode                = {} (notes/static_vertex_implementation_plan.md)\n"
                 "  Subspace C band window   = [{}, {})\n"
                 "  Subspace C size          = {} orbitals (nbnd = {})\n"
                 "  Cuts                     = Sigma^C (G3W2) + Pi^C (G4W), always both\n"
                 "  q->0 rung policy         = {} (notes/q0_head_treatment.md)\n"
                 "  Auxiliary basis          = {}{}\n"
                 "  Status                   = kernels ACTIVE for this rung mode\n",
              _vertex_type, rung_str(), _band_window.first(), _band_window.last(),
              _band_window.size(), nbnd, _div_treatment, _isdf_mode,
              secondary() ? std::string(" (Refinement 2: requested N_m = ") +
                            (_isdf_rank > 0 ? std::to_string(_isdf_rank)
                                            : std::string("auto = nc^2*nk")) +
                            ", svd_tol(B) = " + std::to_string(_isdf_svd_tol) +
                            "; notes/refinement2_optionA.md)"
                          : std::string(" (global THC, dimension Np)"));
      // Announce which theory is active. Each rung mode assembles ALL of its own cuts,
      // so a half-theory cannot be configured.
      if (_rung != dynamic_rung)
        app_log(1, "  [NOTE] vertex_rung = \"{}\": {}. P = {}, and the self-energy "
                   "carries\n"
                   "         {} -- always together (Phi-derivability).\n",
                rung_str(),
                (_rung == static_rung ? "B-S, the iv = 0 statically screened truncation"
                                      : "B-L, the tangent completion, first order in "
                                        "dW = W - W0"),
                (_rung == static_rung ? "P_RPA (no Pi^C injection at all)"
                                      : "P_RPA + P^{C,L} at full weight"),
                (_rung == static_rung ? "Sigma^{C,x} + Sigma^{C,r}"
                                      : "three explicit terms + Sigma^{L,r}"));
    } else {
      // The CLASS keeps the exact empty-C no-op (test_vertex_noop checks it bitwise); a USER input
      // that requests a vertex with an empty window is rejected by the MBPT drivers (mbpt_vertex_audit::check_vertex_requests)
      // after an optional Wannier projector has had the chance to define C.
      app_log(1, "\nvertex_t: vertex_type = \"{}\" (vertex_rung = \"{}\") with an empty "
                 "vertex_band_window: C = empty set, so the vertex contributes nothing in "
                 "ANY rung mode and the calculation reduces to plain scGW exactly\n"
                 "          (unless a Wannier projector defines C next; the MBPT drivers abort on a vertex request whose C "
                 "stays empty -- audit D7).\n",
              _vertex_type, rung_str());
    }
  }

  void vertex_t::build_secondary_basis(THC_ERI auto const &thc,
                                       nda::MemoryArrayOfRank<4> auto const &X_glob, long orb0,
                                       nda::array<long, 2> const &kmq, long iq_gamma) {
    if (_secondary_ready) return;
    // Placed AFTER the lazy guard on purpose: this slot should count REAL builds only, so
    // its ncalls is the number of times the geometry-fixed basis was actually constructed
    // (expected 1 per run). It is INCLUSIVE in whichever entry point triggered it.
    vertex_timer_detail::scoped_timer _tm_total(_Timer, "SEC_BASIS");
    decltype(nda::range::all) all;
    auto mpi = thc.mpi();
    auto MF = thc.MF();
    const long ns = X_glob.shape(0), nkpts = X_glob.shape(1), Np = X_glob.shape(2);
    // t(q) is built at IBZ q ONLY: the kernels source non-IBZ transfers from the
    // IBZ-stored folded cores through the symmetry context. On symmetry-free meshes
    // nqpts_ibz == nqpts.
    const long nqpts = MF->nqpts_ibz();
    utils::check(kmq.shape(0) >= nqpts,
                 "vertex_t::build_secondary_basis: kmq must cover the IBZ q range.");
    const long nc = subspace_rank();
    const long Npair = ns * nkpts * nc * nc;   // the pair index carries momentum
    long Nm_req = (_isdf_rank > 0) ? _isdf_rank : nc * nc * nkpts;
    utils::check(Nm_req <= Npair,
                 "vertex_t::build_secondary_basis: vertex_isdf_rank = {} exceeds the "
                 "subspace pair rank N_pair = ns*nk*nc^2 = {}; the secondary basis "
                 "cannot usefully exceed the space it represents.", Nm_req, Npair);
    // Cap the secondary rank at the GLOBAL basis size Np: the secondary ISDF lives inside
    // the span of the global THC basis, so it must never request more interpolating vectors
    // than the global basis has (out-ranking it selects near-null directions and makes the
    // secondary metric s = B^dag B ill-conditioned -- the companion guard to sec_thresh below).
    // Both the requested and the capped rank are logged: a [WARNING] when the user gave the rank explicitly
    // (vertex_isdf_rank / pol_vertex_isdf_rank > 0), a plain line for the auto default (nc^2 nk), which is a target, not
    // a request.
    const long Nm_asked = Nm_req;
    Nm_req = std::min(Nm_req, (long)thc.Np());
    if (Nm_req < Nm_asked) {
      if (_isdf_rank > 0)
        app_log(1, "  [WARNING] Refinement 2: vertex_isdf_rank = {} exceeds the global THC basis size Np = {}; the secondary "
                   "basis is capped at N_m = {} (the secondary ISDF lives in the span of the global basis).",
                Nm_asked, thc.Np(), Nm_req);
      else
        app_log(1, "  Refinement 2: auto N_m = nc^2 * nk = {} exceeds the global THC basis size Np = {}; using N_m = {}.",
                Nm_asked, thc.Np(), Nm_req);
    }

    // Secondary-ISDF point-selection threshold. It DEFAULTS to the SAME thresh used for
    // the GLOBAL THC basis (thc.thresh()) unless vertex_isdf_thresh (>0) overrides it.
    // Over-resolving the C pair-density metric (e.g. a fixed 1e-13) selects
    // interpolating vectors that leave the span of the global basis, so the transfer
    // t(q) = pinv(B) picks up near-null directions and s becomes ill-conditioned
    // (N_m can exceed the global Np). thc.thresh() is -1.0 when the
    // global THC was built via the nIpts-only path (no thresh set); fall back to a sane
    // 1e-6 in that case so the pivoted Cholesky still has a meaningful stop criterion.
    double sec_thresh = (_isdf_thresh > 0.0) ? _isdf_thresh : thc.thresh();
    if (sec_thresh <= 0.0) {
      app_log(1, "  Refinement 2: global THC thresh is unset (nIpts-only path); "
                 "defaulting secondary-ISDF selection thresh to 1e-6.");
      sec_thresh = 1e-6;
    }

    app_log(1, "\n  Refinement 2: building the secondary ISDF basis on the subspace C "
               "(rank M = {}, {})\n"
               "  requested N_m = {} ({}), used N_m target = {}, svd_tol(B) = {}, sec_thresh = {} (global THC thresh = {}), "
               "N_pair (per q, spin-stacked) = {}\n",
            nc, _wannier ? "Wannier projector" : "band window",
            Nm_asked, (_isdf_rank > 0 ? "explicit" : "auto = nc^2*nk"), Nm_req, _isdf_svd_tol, sec_thresh, thc.thresh(),
            Npair);

    // ---- restricted-range ISDF point selection (collective on thc.mpi()->comm) --------
    // Private methods::thc builder on the SAME MF/mpi context using sec_thresh above; the
    // greedy pivot order makes rank scans nested (first N of a larger selection = a
    // selection of N).
    {
      ptree pt;
      pt.put("thresh", sec_thresh);
      // the blocked pivoted Cholesky is not robust at near-zero thresholds (thc.icc
      // forces block_size = 1 itself when thresh == 0.0; at very tight thresh with the
      // default block 8 it produces NaN residuals) -- use the serial pivot order,
      // which is also the exactly-nested greedy order the rank scans rely on
      pt.put("chol_block_size", 1);
      // This private builder does not see the input's distr_tol and uses the class default
      // 0.2, which limits the number of MPI ranks the secondary selection can run on (the
      // generic "increase distr_tol" advice does not reach it). When vertex_isdf_distr_tol
      // is set (> 0) it is passed through; a larger value allows more ranks. The default
      // (-1) keeps the class default.
      if (_isdf_distr_tol > 0.0) pt.put("distr_tol", _isdf_distr_tol);
      methods::thc builder(MF.get(), *mpi, pt, /*print_metadata*/ false);
      // WINDOW: the band-range overload (Wannier=window). WANNIER: the rotated overload
      // interpolating_points(C_skai, iq, max), fed the zero-padded U as
      // C_skai(s,k,a,i) = conj(U_ia) on the W_rng band columns (the overload rotates the
      // real-space orbitals by conj(C_skai), see thc.icc, so this yields exactly the
      // Wannier orbitals w_a). Its metric resolves (Wannier x all-band) pairs -- a
      // superset of the (Wannier x Wannier) pairs the vertex needs; eta(q,nu) certifies
      // adequacy. The rotated overload requires nkpts == nkpts_ibz (thc.cpp) --
      // Wannier+secondary is nosym only.
      nda::array<long, 1> ipts;
      nda::array<ComplexType, 4> Xa(ns, nkpts, nc, 0);   // (ns, nk, nc, Nm), filled below
      const bool frozen = not _isdf_points_file.empty();
      if (frozen) {
        // FROZEN secondary points (coarse->fine interpolation): the point list of the coarse run is reused on THIS
        // mesh -- no selection. The collocation is gathered at the given points in the selection's convention
        // (thc::collocation_at_points), rotated by U in Wannier mode (X_bar = X U); the transfer t(q) below is
        // rebuilt on this mesh as usual. The points are density-FFT-grid indices, so the FFT mesh must match.
        // Symmetric meshes: the gather builds the image k-points from the IBZ orbitals exactly as the ISDF
        // selection path does.
        nda::array<long, 1> mesh_in;
        long nW_in = 0, W0_in = -1, wan_in = -1;
        {
          h5::file f(_isdf_points_file, 'r');
          h5::group g(f);
          nda::h5_read(g, "ipts", ipts);
          nda::h5_read(g, "fft_mesh", mesh_in);
          h5::h5_read(g, "window_size", nW_in);
          h5::h5_read(g, "window_first", W0_in);
          if (g.has_dataset("wannier")) h5::h5_read(g, "wannier", wan_in);   // absent in older dumps
        }
        // The points were selected for EITHER the window orbitals OR the Wannier orbitals
        // (rotated overload); the dump stores which. A mismatch puts the frozen points in the wrong frame.
        if (wan_in >= 0)
          utils::check((wan_in != 0) == _wannier,
                       "vertex_t::build_secondary_basis: the frozen points of {} were selected in {} mode, this run is in {} "
                       "mode (audit A19): re-dump the points with the same projector setting (vertex_wannier_file).",
                       _isdf_points_file, wan_in ? "Wannier" : "band-window", _wannier ? "Wannier" : "band-window");
        else
          app_log(1, "  [WARNING] the frozen points file {} carries no \"wannier\" flag (an old dump): its window-vs-Wannier "
                     "selection mode cannot be checked against this run ({} mode).", _isdf_points_file,
                  _wannier ? "Wannier" : "band-window");
        // Frozen points mean NO point selection, so the selection knobs the user set explicitly do nothing here.
        // (vertex_isdf_svd_tol / _cond_max still act: they regularize the per-q transfer solve t(q) below.)
        if (_isdf_rank > 0)
          app_log(1, "  [WARNING] vertex_isdf_rank (pol_vertex_isdf_rank) = {} is IGNORED: the secondary points are frozen "
                     "from {} (N_m = {} = the stored point count).", _isdf_rank, _isdf_points_file, ipts.extent(0));
        if (_isdf_thresh > 0.0)
          app_log(1, "  [WARNING] vertex_isdf_thresh (pol_vertex_isdf_thresh) = {} is IGNORED: the secondary points are frozen "
                     "from {} (no point selection on this mesh).", _isdf_thresh, _isdf_points_file);
        auto mesh_now = builder.rho_mesh();
        utils::check(mesh_in.size() == 3 and mesh_in(0) == mesh_now(0) and mesh_in(1) == mesh_now(1) and
                     mesh_in(2) == mesh_now(2),
                     "vertex_t::build_secondary_basis: frozen points were selected on FFT mesh {} x {} x {}, "
                     "this run's density grid is {} x {} x {} (same cell + same THC ecut required).",
                     mesh_in(0), mesh_in(1), mesh_in(2), mesh_now(0), mesh_now(1), mesh_now(2));
        utils::check(nW_in == long(_band_window.size()) and W0_in == long(_band_window.first()),
                     "vertex_t::build_secondary_basis: frozen points belong to the band window [{}, {}), this "
                     "run's is [{}, {}).", W0_in, W0_in + nW_in, _band_window.first(), _band_window.last());
        const long Nm = ipts.extent(0);
        auto Xw = builder.collocation_at_points(ipts, nda::range(0, nkpts), _band_window);   // (ns, nk, nW, Nm)
        Xa = nda::array<ComplexType, 4>(ns, nkpts, nc, Nm);
        Xa() = ComplexType(0.0);
        const long nW = long(_band_window.size());
        for (long is = 0; is < ns; ++is)
          for (long ik = 0; ik < nkpts; ++ik)
            for (long a = 0; a < nc; ++a)
              for (long m = 0; m < Nm; ++m) {
                if (_wannier) {
                  ComplexType acc(0.0);
                  for (long i = 0; i < nW; ++i) acc += Xw(is, ik, i, m) * _U_skia(is, ik, i, a);
                  Xa(is, ik, a, m) = acc;
                } else {
                  Xa(is, ik, a, m) = Xw(is, ik, a, m);
                }
              }
        app_log(1, "  [W-int] secondary ISDF points FROZEN from {}: N_m = {} (FFT mesh {} x {} x {}, window [{}, {}), "
                   "{}); no point selection on this mesh.",
                _isdf_points_file, Nm, mesh_now(0), mesh_now(1), mesh_now(2), _band_window.first(),
                _band_window.last(), _wannier ? "X_bar = X U" : "window");
      } else if (not _wannier) {
        auto [ip, dXa, dXb] = builder.interpolating_points<HOST_MEMORY>(
            int(iq_gamma), int(Nm_req), _band_window, _band_window);
        (void)dXb;   // empty optional for a_range == b_range at Gamma (single_psi path)
        const long Nm = ip.extent(0);
        auto gs = dXa.global_shape();
        utils::check(gs[0] == ns and gs[1] == nkpts and gs[2] == nc and gs[3] == Nm,
                     "vertex_t::build_secondary_basis: unexpected collocation shape "
                     "({}, {}, {}, {}); expected ({}, {}, {}, {}).",
                     gs[0], gs[1], gs[2], gs[3], ns, nkpts, nc, Nm);
        Xa = nda::array<ComplexType, 4>(ns, nkpts, nc, Nm);
        Xa() = ComplexType(0.0);
        Xa(dXa.local_range(0), dXa.local_range(1), dXa.local_range(2), dXa.local_range(3)) =
            dXa.local();
        ipts = std::move(ip);
      } else {
        // Guard must match the sym_mesh predicate of the eval paths (eval_Sigma_C /
        // eval_Pi_C): a mesh reduced by TIME REVERSAL alone still has
        // nkpts == nkpts_ibz, but the eval paths then take the symmetry route and
        // the secondary sym ctx is never U-rotated -- letting Wannier+secondary
        // through on such a mesh would mix a rotated basis with a window-mode ctx.
        bool sym_mesh = (MF->nqpts() != MF->nqpts_ibz()) or
                        (MF->nkpts() != MF->nkpts_ibz());
        {
          auto kp_trev = MF->kp_trev();
          for (long ik = 0; ik < MF->nkpts(); ++ik)
            if (kp_trev(ik)) { sym_mesh = true; break; }
        }
        utils::check(not sym_mesh,
                     "vertex_t::build_secondary_basis: the Wannier rotated point-selection "
                     "overload does not support symmetry-reduced k-meshes (thc.cpp:207), "
                     "including meshes reduced by time reversal alone. "
                     "Use vertex_isdf = \"global\" for Wannier + symmetry runs.");
        const long nbnd = MF->nbnd();
        nda::array<ComplexType, 4> C_skai(ns, nkpts, nc, nbnd);
        C_skai() = ComplexType(0.0);
        for (long is = 0; is < ns; ++is)
          for (long ik = 0; ik < nkpts; ++ik)
            for (long a = 0; a < nc; ++a)
              for (long i = 0; i < _band_window.size(); ++i)
                C_skai(is, ik, a, _band_window.first() + i) =
                    std::conj(_U_skia(is, ik, i, a));
        auto [ip, dXa, dXb] =
            builder.interpolating_points<HOST_MEMORY>(C_skai, int(iq_gamma), int(Nm_req));
        (void)dXb;
        const long Nm = ip.extent(0);
        auto gs = dXa.global_shape();
        utils::check(gs[0] == ns and gs[1] == nkpts and gs[2] == nc and gs[3] == Nm,
                     "vertex_t::build_secondary_basis: unexpected rotated collocation "
                     "shape ({}, {}, {}, {}); expected ({}, {}, {}, {}).",
                     gs[0], gs[1], gs[2], gs[3], ns, nkpts, nc, Nm);
        Xa = nda::array<ComplexType, 4>(ns, nkpts, nc, Nm);
        Xa() = ComplexType(0.0);
        Xa(dXa.local_range(0), dXa.local_range(1), dXa.local_range(2), dXa.local_range(3)) =
            dXa.local();
        ipts = std::move(ip);
      }
      const long Nm = ipts.extent(0);
      utils::check(Nm > 0,
                   "vertex_t::build_secondary_basis: point selection returned 0 points.");
      if (Nm < Nm_req and not frozen)
        app_log(1, "  [NOTE] Refinement 2: point selection stopped at N_m = {} "
                   "(< target {}):\n"
                   "         the C pair-density metric is numerically rank-deficient below "
                   "the selection thresh = {};\n"
                   "         using the returned rank.", Nm, Nm_req, sec_thresh);
      // gather the distributed collocation (already assembled into Xa above), then
      // transpose to the kernels' (aux, orb) layout. Any fixed per-point phase/scale
      // convention of the selection output is absorbed by the least-squares transfer.
      if (not frozen) mpi->comm.all_reduce_in_place_n(Xa.data(), Xa.size(), std::plus<>{});   // frozen: replicated
      _Xb_skma = nda::array<ComplexType, 4>(ns, nkpts, Nm, nc);
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nkpts; ++ik)
          for (long a = 0; a < nc; ++a)
            for (long m = 0; m < Nm; ++m)
              _Xb_skma(is, ik, m, a) = Xa(is, ik, a, m);
      _Nm = Nm;
      _sec_ipts = ipts;
      if (_isdf_points_dump and mpi->comm.root()) {
        // the point list for a fine-mesh run to freeze (pol_vertex_isdf_points_file)
        const std::string fn = _run_prefix + ".secpts.h5";
        h5::file f(fn, 'w');
        h5::group g(f);
        nda::h5_write(g, "ipts", ipts);
        nda::h5_write(g, "fft_mesh", builder.rho_mesh());
        h5::h5_write(g, "window_size", long(_band_window.size()));
        h5::h5_write(g, "window_first", long(_band_window.first()));
        h5::h5_write(g, "wannier", long(_wannier ? 1 : 0));
        h5::h5_write(g, "nm", Nm);
        app_log(1, "  [W-int] secondary ISDF points written to {} (N_m = {}) -- freeze them on the fine mesh with "
                   "pol_vertex_isdf_points_file.", fn, Nm);
      }
    }

    // ---- conditioning cap (vertex_isdf_cond_max): applied PER Q in the transfer solve ---
    // The cond(s) blowup is Q-SPECIFIC and typically NOT at Gamma (the selection is done at
    // Gamma, where s is well conditioned). The interpolating POINTS are shared across
    // all q, so pruning the shared set CANNOT bound the worst-q conditioning -- pruning only
    // removes points redundant at EVERY q, while the worst q's ill-conditioning comes from
    // points that nearly coincide THERE but separate elsewhere (why both were selected). The
    // robust cap is therefore applied per q in the least-squares solve below: the gelss
    // rcond truncates B(q)'s near-null directions so each q's downfold t(q) is conditioned
    // to <= _isdf_cond_max (rcond_eff = 1/sqrt(cond_max), floored by _isdf_svd_tol).
    // Disabled (_isdf_cond_max <= 0) => rcond_eff = _isdf_svd_tol.

    // ---- per-q transfer t(q) = s(q)^+ B(q)^dag C(q) -----------------------------------
    // Solved as the truncated-SVD least squares min || B t - C ||_F directly on B
    // (numerically equivalent, better conditioned: rcond acts on sv(B); the metric
    // s = B^dag B is thereby regularized at rcond^2). The explicit s^{-1} is REQUIRED:
    // the code's THC body contractions are metric-free.
    //
    // The q loop is distributed over mpi->comm (round-robin): each rank solves its q
    // subset into the zero-initialized _t_qmP, then a single all_reduce(plus) GATHERS the
    // rows (a partition: each q is written by exactly one rank, so the result is exact and
    // independent of the rank count). Per-rank solve work drops ~1/P. _t_qmP stays
    // replicated because its consumers (Pi upfold, Sigma, cache_w) loop over every IBZ q.
    // Diagnostics (cond/fit/disc maxima) are reduced with max.
    _t_qmP = nda::array<ComplexType, 3>(nqpts, _Nm, Np);
    _t_qmP() = ComplexType(0.0);
    double cond_s_max = 0.0, cond_eff_max = 0.0, fit_max = 0.0;
    long disc_max = 0;
    long my_nsolve = 0;
    // conditioning cap: rcond truncates B(q)'s near-null directions so each q's downfold is
    // conditioned to <= _isdf_cond_max. Floored by _isdf_svd_tol; disabled => svd_tol only.
    const double rcond_eff = (_isdf_cond_max > 0.0)
        ? std::max(_isdf_svd_tol, 1.0 / std::sqrt(_isdf_cond_max))
        : _isdf_svd_tol;
    {
      nda::array<ComplexType, 2> B_Im(Npair, _Nm), C_IP(Npair, Np);
      // gelss needs F-layout: keep the TRANSPOSES in C-layout and pass transposed views
      nda::array<ComplexType, 2> BT(_Nm, Npair), CT(Np, Npair), Cf(Npair, Np);
      nda::array<double, 1> sv(std::min(Npair, _Nm));
      for (long iq = mpi->comm.rank(); iq < nqpts; iq += mpi->comm.size()) {
        ++my_nsolve;
        vertex_secondary_detail::build_pair_matrix(_Xb_skma, 0, nc, kmq, iq, B_Im);
        vertex_secondary_detail::build_pair_matrix(X_glob, orb0, nc, kmq, iq, C_IP);
        for (long I = 0; I < Npair; ++I) {
          for (long m = 0; m < _Nm; ++m) BT(m, I) = B_Im(I, m);
          for (long P = 0; P < Np; ++P) CT(P, I) = C_IP(I, P);
        }
        int rank = 0;
        int info = nda::lapack::gelss(nda::transpose(BT), nda::transpose(CT), sv,
                                      rcond_eff, rank);
        utils::check(info == 0, "vertex_t::build_secondary_basis: gelss failed "
                                "(info = {}) at iq = {}.", info, iq);
        // solution rows live in the first N_m rows of the (transposed-view) rhs
        for (long m = 0; m < _Nm; ++m)
          for (long P = 0; P < Np; ++P) _t_qmP(iq, m, P) = CT(P, m);
        // diagnostics: cond_s = raw metric cond(B)^2; cond_eff = conditioning the
        // regularized solve actually sees (smallest RETAINED sv) <= 1/rcond_eff^2 = the cap
        const double smax = sv(0), smin = sv(sv.size() - 1);   // descending order
        const double cond_s = (smax / std::max(smin, 1e-300)) *
                              (smax / std::max(smin, 1e-300));
        const double smin_kept = (rank > 0) ? sv(rank - 1) : smin;
        const double cond_eff = (smax / std::max(smin_kept, 1e-300)) *
                                (smax / std::max(smin_kept, 1e-300));
        const long discarded = _Nm - rank;
        double num = 0.0, den = 0.0;
        nda::blas::gemm(B_Im, _t_qmP(iq, all, all), Cf);
        for (long I = 0; I < Npair; ++I)
          for (long P = 0; P < Np; ++P) {
            num += std::norm(Cf(I, P) - C_IP(I, P));
            den += std::norm(C_IP(I, P));
          }
        const double fit = std::sqrt(num) / std::max(std::sqrt(den), 1e-300);
        app_log(3, "    Refinement 2 t(q = {}): sv(B) in [{}, {}], cond(s) = {}, "
                   "rank = {}/{}, discarded = {}, ||Bt - C||_F/||C||_F = {}",
                iq, smin, smax, cond_s, rank, _Nm, discarded, fit);
        cond_s_max = std::max(cond_s_max, cond_s);
        cond_eff_max = std::max(cond_eff_max, cond_eff);
        disc_max = std::max(disc_max, discarded);
        fit_max = std::max(fit_max, fit);
      }
    }
    // GATHER the q-distributed t(q) rows (partition => exact) and reduce the diagnostics
    // across ranks.
    mpi->comm.all_reduce_in_place_n(_t_qmP.data(), _t_qmP.size(), std::plus<>{});
    cond_s_max = mpi->comm.all_reduce_value(cond_s_max, boost::mpi3::max<>{});
    cond_eff_max = mpi->comm.all_reduce_value(cond_eff_max, boost::mpi3::max<>{});
    fit_max    = mpi->comm.all_reduce_value(fit_max, boost::mpi3::max<>{});
    disc_max   = mpi->comm.all_reduce_value(disc_max, boost::mpi3::max<>{});
    const long total_solve = mpi->comm.all_reduce_value(my_nsolve, std::plus<>{});
    app_log(1, "  Refinement 2 secondary basis READY: N_m = {} (pair rank {} per q), "
               "max_q cond(s) = {} (raw metric), {} (regularized solve, rcond = {}),\n"
               "  max_q discarded sv = {}, max_q fit residual ||Bt - C||_F/||C||_F = {}\n"
               "  t(q) solve distributed over {} ranks: this rank ran {} of {} gelss "
               "solves (~1/P work)\n",
            _Nm, Npair, cond_s_max, cond_eff_max, rcond_eff, disc_max, fit_max,
            mpi->comm.size(), my_nsolve, total_solve);
    // store the REGULARIZED conditioning (what the cap controls); cond_eff_max <= the cap.
    _cond_s_max = cond_eff_max;
    if (_isdf_cond_max > 0.0)
      app_log(1, "  Refinement 2: conditioning cap vertex_isdf_cond_max = {} -> per-q "
                 "downfold conditioning bounded to max_q {} (rcond = {}); the raw metric "
                 "cond(s) = {} is regularized in the solve.",
              _isdf_cond_max, cond_eff_max, rcond_eff, cond_s_max);
    _secondary_ready = true;
  }

  void vertex_t::check_iaft_backend(std::string_view where) const {
    if (_ft->basis() == imag_axes_ft::dlr_basis) return;
    if (_rung == dynamic_rung) {
      utils::check(false,
                   "{}: the fused G3W2 kernel requires the DLR IAFT backend "
                   "(iaft basis = \"dlr\"); the IR backend is not supported.", where);
    } else {
      // The static rungs need no pole algebra, so this requirement is NOT structural like
      // the dynamic one: it stands because the Pi^{C,0}(tau = 0) interpolation row is not
      // available on the IR driver.
      utils::check(false,
                   "{}: vertex_rung = \"{}\" also requires the DLR IAFT backend "
                   "(iaft basis = \"dlr\") for now. The static rungs themselves need no "
                   "pole algebra; what is missing on IR is the Pi^{{C,0}}(tau = 0) "
                   "interpolation row (decision D3, open until increment S4).",
                   where, rung_str());
    }
  }

  void vertex_t::print_vertex_timers() const {
    // getOrAdd, NOT getPos: TimerManager::elapsed(name) APP_ABORTs on an unregistered
    // slot, and a run legitimately skips whole stages (SIG_RESP_* are linear-only;
    // SIG_W_GATHER is dynamic-only; CACHE_W only exists on the secondary path). Reading
    // through add() materializes a zeroed slot instead of aborting the run from inside a
    // diagnostic printer -- a printer must never be able to kill a multi-hour job.
    auto T = [&](const char* n) { return _Timer.elapsed(_Timer.add(n)); };
    auto A = [&](const char* n) { return _Timer.average(_Timer.add(n)); };
    auto N = [&](const char* n) { return _Timer.number_of_calls(_Timer.add(n)); };

    // The four TOP-LEVEL entry points are disjoint (none calls another), so their sum is
    // this rank's total time inside vertex routines. SEC_BASIS / SYM_CTX are nested
    // inside them and are reported separately below.
    const double t_sig = T("SIGMA_C"), t_pi = T("PI_C");
    const double t_cw  = T("CACHE_W"), t_w0 = T("BUILD_W0");
    const double total = t_sig + t_pi + t_cw + t_w0;
    if (total <= 0.0) return;   // vertex never ran (C = empty, or inactive)
    const double sc = 100.0 / total;

    auto row = [&](const char* label, const char* slot, int indent) {
      const double e = T(slot);
      if (e <= 0.0 and N(slot) == 0) return;   // stage not exercised in this rung mode
      app_log(2, "  {:{}}{:<26}{:>12.3f}{:>12.4f}{:>8d}{:>8.1f}", "", indent,
              label, e, A(slot), N(slot), e * sc);
    };
    // A negative remainder would mean overlapping start/stop pairs; print it either way
    // rather than clamping, because the sign is the diagnostic.
    auto remainder = [&](double parent, std::initializer_list<const char*> parts) {
      double s = 0.0;
      for (auto p : parts) s += T(p);
      return parent - s;
    };

    app_log(2, "\n  ISDF-Vertex timers  (rung = {}, rank-local wall time)", rung_str());
    app_log(2, "  ----------------------------------------------------------------------");
    app_log(2, "  {:<28}{:>12}{:>12}{:>8}{:>8}",
            "operation", "elapsed(s)", "avg(s)", "calls", "% vtx");
    app_log(2, "  {:<28}{:>12.3f}{:>12}{:>8}{:>8.1f}", "TOTAL (vertex routines)",
            total, "-", "-", 100.0);

    row("eval_Sigma_C", "SIGMA_C", 2);
    row("setup + Z(q)",        "SIG_SETUP",     4);
    row("dW gather/unfold",    "SIG_W_GATHER",  4);
    row("secondary fold",      "SIG_SECONDARY", 4);
    row("IBZ sym ctx",         "SIG_SYMCTX",    4);
    row("KERNEL (G^3W^2)",     "SIG_KERNEL",    4);
    row("response Sigma^{C,r}","SIG_RESPONSE",  4);
    // inclusive sub-stages of SIG_RESPONSE -- indented further and NOT part of the sum
    row("|- Pi^{C,0} (no poles)",  "SIG_RESP_PI0",    6);
    row("|- Wdyn tau->inu",        "SIG_RESP_WDYNW",  6);
    row("|- pi^dyn factorized",    "SIG_RESP_PIDYNF", 6);
    row("|- Pi^{C,dyn} @tau=0 *",  "SIG_RESP_PIDYN",  6);
    row("barrier (skew)",      "SIG_BARRIER",   4);
    {
      const double r = remainder(t_sig, {"SIG_SETUP", "SIG_W_GATHER", "SIG_SECONDARY",
                                         "SIG_SYMCTX", "SIG_KERNEL", "SIG_RESPONSE",
                                         "SIG_BARRIER"});
      if (std::abs(r) > 1e-6)
        app_log(2, "  {:4}{:<26}{:>12.3f}{:>12}{:>8}{:>8.1f}", "", "(unattributed)",
                r, "-", "-", r * sc);
    }

    row("eval_Pi_C", "PI_C", 2);
    row("setup + W materialize","PI_SETUP",         4);
    row("secondary fold",       "PI_SECONDARY",     4);
    row("IBZ sym ctx",          "PI_SYMCTX",        4);
    row("KERNEL (G^4W)",        "PI_KERNEL",        4);
    row("upfold + reduce",      "PI_UPFOLD_REDUCE", 4);
    row("barrier (skew)",       "PI_BARRIER",       4);
    {
      const double r = remainder(t_pi, {"PI_SETUP", "PI_SECONDARY", "PI_SYMCTX",
                                        "PI_KERNEL", "PI_UPFOLD_REDUCE", "PI_BARRIER"});
      if (std::abs(r) > 1e-6)
        app_log(2, "  {:4}{:<26}{:>12.3f}{:>12}{:>8}{:>8.1f}", "", "(unattributed)",
                r, "-", "-", r * sc);
    }

    row("cache_w",  "CACHE_W",  2);

    row("build_w0", "BUILD_W0", 2);
    row("(P,Q) layout",        "W0_LAYOUT",   4);
    row("Pi_RPA i.nu=0 row",   "W0_PI0_ROW",  4);
    row("1-freq THC Dyson",    "W0_DYSON",    4);
    row("q->0 head policy",    "W0_HEAD",     4);
    row("W0 = Z + dW0",        "W0_ASSEMBLE", 4);
    row("W0bar = t W0 t^dag",  "W0_FOLD",     4);
    row("barrier (skew)",      "W0_BARRIER",  4);
    {
      const double r = remainder(t_w0, {"W0_LAYOUT", "W0_PI0_ROW", "W0_DYSON", "W0_HEAD",
                                        "W0_ASSEMBLE", "W0_FOLD", "W0_BARRIER"});
      if (std::abs(r) > 1e-6)
        app_log(2, "  {:4}{:<26}{:>12.3f}{:>12}{:>8}{:>8.1f}", "", "(unattributed)",
                r, "-", "-", r * sc);
    }

    // Lazy, geometry-fixed, built once: already counted inside whichever entry point
    // triggered them, so they are listed for visibility and NOT added to TOTAL.
    if (T("SEC_BASIS") > 0.0 or T("SYM_CTX") > 0.0) {
      app_log(2, "  {:<28}", "lazy builds (incl. above, not re-added):");
      row("secondary ISDF basis", "SEC_BASIS", 4);
      row("IBZ symmetry ctx",     "SYM_CTX",   4);
    }
    if (T("SIG_RESP_PIDYN") > 0.0) {
      app_log(2, "  * Pi^{{C,dyn}} @tau=0 runs the FULL dynamic kernel (incl. the aux pole\n"
                 "    algebra) and keeps only the tau = 0 row. It is REPLACED by the\n"
                 "    factorized eq:pibardynfact row above unless vertex_pidyn = \"kernel\"\n"
                 "    or \"check\"; when both rows are present their ratio IS the win.");
      if (T("SIG_RESP_PIDYNF") > 0.0 and T("SIG_RESP_PIDYN") > 0.0)
        app_log(2, "    measured pi^dyn speedup: {:.1f}x (kernel {:.3f} s vs factorized "
                   "{:.3f} s)", T("SIG_RESP_PIDYN") / T("SIG_RESP_PIDYNF"),
                T("SIG_RESP_PIDYN"), T("SIG_RESP_PIDYNF"));
    }
    app_log(2, "  ----------------------------------------------------------------------\n");
    app_log_flush();
  }

  void vertex_t::eval_Sigma_C(MBState &mb_state, THC_ERI auto const &thc) {
    vertex_timer_detail::scoped_timer _tm_total(_Timer, "SIGMA_C");
    // Stage timers PARTITION SIGMA_C: each stop is immediately followed by the next
    // start, so the six SIG_* slots sum to SIGMA_C up to the print's rounding. They are
    // explicit start/stop (not scoped_timer) because the stages are sequential regions of
    // one flat function body -- brace-scoping them would break the variable lifetimes the
    // later stages depend on.
    _Timer.start("SIG_SETUP");
    utils::check(active(), "vertex_t::eval_Sigma_C: called while the vertex is inactive. "
                           "Callers must guard vertex calls with vertex_t::active().");
    check_rung_implemented("vertex_t::eval_Sigma_C");
    // STATIC-rung mode (B-S). Sigma^{C,x} is the doubly-instantaneous
    // reduction of the SAME kernel with both rungs = W0bar. Nothing
    // dynamical is consumed: no Z build, no head re-insertion (build_w0 already applied
    // the policy to W0 -- "one policy, one W0, every appearance"), no dW gather, no
    // secondary fold of Z/dW, and no pole machinery.
    const bool stat = (_rung != dynamic_rung);
    // B-L still consumes the SAME-ITERATION dynamic W: its two mixed terms are
    // W0_x dW_y + dW_x W0_y with dW = W - W0. So the bare core and dW are built for
    // "dynamic" and "linear", and skipped only for B-S (which is purely tau-local).
    const bool lin = (_rung == linear_rung);
    const bool need_dyn = (_rung != static_rung);
    utils::check(mb_state.sG_tskij.has_value(),
                 "vertex_t::eval_Sigma_C: sG_tskij is not initialized in MBState.");
    utils::check(mb_state.sSigma_tskij.has_value(),
                 "vertex_t::eval_Sigma_C: sSigma_tskij is not initialized in MBState.");
    utils::check(stat or mb_state.dW_qtPQ.has_value(),
                 "vertex_t::eval_Sigma_C: dW_qtPQ is not initialized in MBState.");
    utils::check(not stat or _W0b_qmm.has_value(),
                 "vertex_t::eval_Sigma_C: vertex_rung = \"{}\" needs the static rung "
                 "W0bar, which update_w builds (vertex_t::build_w0). It is absent -- the "
                 "update_w seam did not run for this iteration.", rung_str());

    decltype(nda::range::all) all;
    auto mpi = thc.mpi();
    auto MF = thc.MF();
    const long nkpts = MF->nkpts();
    const long nqpts = MF->nqpts();
    const long nkpts_ibz = MF->nkpts_ibz();
    const long nqpts_ibz = MF->nqpts_ibz();
    const long Np = thc.Np();
    const long nbnd = MF->nbnd();

    // IBZ SYMMETRY: on symmetry-reduced meshes the external k axis stays IBZ-resident,
    // all internal sums run over the full BZ, and the rungs are sourced from the
    // IBZ-stored W/Z through the symmetry context. Symmetry-free meshes take the plain
    // full-BZ path.
    bool sym_mesh = (nqpts != nqpts_ibz) or (nkpts != nkpts_ibz);
    {
      auto kp_trev = MF->kp_trev();
      for (long ik = 0; ik < nkpts; ++ik)
        if (kp_trev(ik)) { sym_mesh = true; break; }
    }
    utils::check(nqpts == nkpts,
                 "vertex_t::eval_Sigma_C: expected a full transfer mesh with nqpts == "
                 "nkpts (got {} vs {}).", nqpts, nkpts);
    utils::check(MF->npol() == 1, "vertex_t::eval_Sigma_C: npol != 1 is not supported.");
    check_iaft_backend("vertex_t::eval_Sigma_C");

    auto G_tskij = mb_state.sG_tskij.value().local();
    auto& sSigma_tskij = mb_state.sSigma_tskij.value();
    const long nt = G_tskij.shape(0);
    const long ns = G_tskij.shape(1);
    const long nt_half = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;
    const long nk_ext = sym_mesh ? nkpts_ibz : nkpts;   // external Sigma^C k axis
    utils::check(G_tskij.shape(2) == nk_ext,
                 "vertex_t::eval_Sigma_C: G_tskij k axis = {} != {} ({}).",
                 G_tskij.shape(2), nk_ext, sym_mesh ? "nkpts_ibz" : "nkpts");
    utils::check(nt == _ft->nt_f(), "vertex_t::eval_Sigma_C: G time axis != nt_f.");
    { // the W(beta-tau)=W(tau) unfolding below requires a tau mesh symmetric about beta/2
      auto tau_mesh = _ft->tau_mesh();
      for (long it = 0; it < nt; ++it)
        utils::check(std::abs(std::abs(tau_mesh(it)) - std::abs(tau_mesh(nt - it - 1))) <= 1e-6,
                     "vertex_t::eval_Sigma_C: IAFT tau grid is not particle-hole symmetric.");
    }

    app_log(1, "\n  ISDF-Vertex: evaluating Sigma^C (G^3 W^2, double bosonic convolution)\n"
               "  ---------------------------------------------------------------------\n"
               "  Subspace C band window = [{}, {})  ({} orbitals)\n"
               "  nbnd = {}, Np = {}, nkpts = {}, prefactor = +1 (sign_crossing_report)\n",
            _band_window.first(), _band_window.last(), _band_window.size(),
            nbnd, Np, nkpts);

    // ---- collocation matrices (q-independent X, polarization 0) ----------------------
    // Node-share X_skPa (the ns*nk*Np*nbnd collocation, a dominant memory term) -- one
    // copy per NUMA node, not one per rank. Values are copied from the already-node-shared
    // thc.X (data-location change only). All downstream X consumers (kernel, build_Xbar,
    // build_sym_ctx, build_secondary_basis, eta_max_over_q, X_C slice) are templated to
    // bind the shared_array .local() view.
    auto sX_skPa = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(
        *mpi, std::array<long, 4>{ns, nkpts, Np, nbnd});
    sX_skPa.win().fence();
    if (mpi->node_comm.root())
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nkpts; ++ik)
          sX_skPa.local()(is, ik, all, all) = thc.X(is, 0, ik);
    sX_skPa.win().fence();
    auto X_skPa = sX_skPa.local();

    // ---- q = Gamma index (crystal coordinates: all components integer mod G) ----------
    long iq_gamma = -1;
    {
      auto Qpts = MF->Qpts();
      for (long iq = 0; iq < nqpts; ++iq) {
        double d = 0.0;
        for (long i = 0; i < 3; ++i) {
          double x = Qpts(iq, i);
          d += std::abs(x - std::round(x));
        }
        if (d < 1e-8) {
          utils::check(iq_gamma < 0,
                       "vertex_t::eval_Sigma_C: multiple Gamma q-points found ({} and {}).",
                       iq_gamma, iq);
          iq_gamma = iq;
        }
      }
      utils::check(iq_gamma >= 0, "vertex_t::eval_Sigma_C: no Gamma q-point found.");
      utils::check(iq_gamma < nqpts_ibz,
                   "vertex_t::eval_Sigma_C: Gamma q-point index {} is outside the IBZ "
                   "range [0, {}).", iq_gamma, nqpts_ibz);
    }

    // ---- q->0 rung policy -------------------------------------------------------------
    const bool skip_rung_gamma = (_div_treatment == "v1_skip");
    bool head_insertion = (_div_treatment.find("gygi") != std::string::npos);
    if (head_insertion and nqpts_ibz == 1) {
      app_log(1, "  [WARNING] Sigma^C: nqpts_ibz == 1 while vertex div_treatment is "
                 "gygi-class; the q->0\n"
                 "            extrapolation is meaningless on a Gamma-only mesh -- taking "
                 "\"ignore_g0\" instead\n"
                 "            (same downgrade as GW's Sigma_div_correction).");
      head_insertion = false;
    }
    if (skip_rung_gamma)
      app_log(1, "  [NOTE] Sigma^C q->0 policy: v1_skip -- the q = Gamma (iq = {}) cell is "
                 "DROPPED on both rung\n"
                 "         transfers (qx and qy; bare Z and dynamic dW). Fallback mode; "
                 "O(1/N_k) finite-size error.\n", iq_gamma);
    else
      app_log(1, "  [NOTE] Sigma^C q->0 policy: {} -- the q = Gamma (iq = {}) cell of both "
                 "rung transfers is\n"
                 "         INCLUDED with the stored regularized W(Gamma) (v(G=0) zeroed at "
                 "ERI build){}.\n", _div_treatment, iq_gamma,
              head_insertion ? ",\n         PLUS the analytic rank-1 head insertion "
                               "Nk*madelung*[1 | Re eps_inv_head(tau)]*chi chi^dag"
                             : "; no analytic head (GW ignore_g0 analogue)");

    // ---- bare coulomb Z(q): the instantaneous part of the rungs (collective call) -----
    // IBZ rows under symmetry (the kernels source non-IBZ transfers via the sym ctx)
    nda::array<ComplexType, 3> Z_qPQ(need_dyn ? nqpts_ibz : 0, Np, Np);
    if (need_dyn)
      for (long iq = 0; iq < nqpts_ibz; ++iq)
        Z_qPQ(iq, all, all) = thc.Z(int(iq));

    // head insertion, bare piece (weight 1) into Z(Gamma). STATIC modes: the head policy
    // was already applied to W0 inside build_w0, so re-applying it
    // here would double-count it.
    nda::array<ComplexType, 2> H_PQ(need_dyn ? Np : 0, need_dyn ? Np : 0);
    bool head_ok = false;
    if (head_insertion and need_dyn) {
      head_ok = vertex_head_detail::build_head_rank1(thc, iq_gamma, nkpts, H_PQ,
                                                                    _bl_head_scale);
      if (head_ok) {
        if (_bl_head_static_all and lin) {
          // ---- THE BALANCED FIRST-ORDER HEAD (see _bl_head_static_all) --------------
          // The FULL STATIC-weight head c*(1 + eps_inv_head(i.nu=0)) goes into the
          // INSTANTANEOUS slot, using build_w0's OWN weight so it cancels against
          // W0(Gamma)'s head in the fluctuation dW = [Z + dW(i.nu)] - W0 (dWw_lin's
          // broadcast term below) AND in pi^dyn's rung difference against Pi^{C,0}.
          // The dynamic-slot piece is then NOT added (the branch below), so dW carries
          // no analytic head at all: delta W_head == 0, the head is part of the
          // expansion point instead of the fluctuation.
          utils::check(_w0_head_applied,
                       "vertex_t::eval_Sigma_C: vertex_bl_head_static_all requires "
                       "build_w0's head weight (_w0_eps_head), but build_w0 did not "
                       "apply a head this iteration -- inconsistent q->0 policy state.");
          Z_qPQ(iq_gamma, all, all) += ComplexType(1.0 + _w0_eps_head) * H_PQ;
          double h_max = 0.0;
          for (auto const& v : H_PQ) h_max = std::max(h_max, std::abs(v));
          app_log(1, "  Sigma^C head insertion [H1 STATIC]: madelung = {}, |H|_max = {}, "
                     "weight 1 + eps_inv_head(i.nu=0) = {:.6e} applied to Z(Gamma); the "
                     "dynamic piece is NOT added (dW = W - W0 is analytic-head-free).",
                  MF->madelung(), h_max, 1.0 + _w0_eps_head);
        } else {
        Z_qPQ(iq_gamma, all, all) += H_PQ;
        double h_max = 0.0;
        for (auto const& v : H_PQ) h_max = std::max(h_max, std::abs(v));
        app_log(1, "  Sigma^C head insertion: madelung = {}, |H|_max = {} (bare piece "
                   "applied to Z(Gamma))", MF->madelung(), h_max);
        }
      } else if (head_unusable_continue("vertex_t::eval_Sigma_C")) {   // aborts unless allowed / explicit
        app_log(1, "  [WARNING] Sigma^C: gygi head insertion requested but head data are "
                   "unusable\n"
                   "            (madelung == 0 or empty basis_head) -- proceeding WITHOUT "
                   "the analytic head\n"
                   "            (equivalent to policy \"ignore_g0\").");
      }
    }

    _Timer.stop("SIG_SETUP");
    _Timer.start("SIG_W_GATHER");
    // ---- dynamic W(tau): replicate and unfold nt_half storage to the full tau mesh ----
    // dW_qtPQ is dynamic-only (bare Z subtracted, scr_coulomb_t.cpp); W is
    // PH-symmetric in tau, W(beta-t) = W(t). IBZ rows under symmetry.
    //
    // LEAN PATH (global B-L): replicating this slab per rank -- and the two nu-domain
    // all-q Np^2 arrays derived from it (dWw_lin, Wdyn_w) -- costs O(nq_ibz nt Np^2) per
    // rank and exhausts memory on dense k-meshes. The kernel never consumes W(tau) at all
    // (its ONLY use is staging the tau->nu transform: vertex_sigma.icc builds/receives
    // the nu-domain rung), so for B-L on the global path the slab is skipped entirely and
    // ONE node-shared nu window is staged per-q straight from the DISTRIBUTED
    // mb_state.dW_qtPQ (gather_dW_one_q is a collective pure gather -- bit-identical to
    // slicing the all-q gather). The per-element op chain (head-add on the tau-half
    // slice -> mirror unfold -> tau->nu gemm -> +(Z - W0)) is the same as on the slab
    // path, so the results are identical. The window is staged TWICE per eval -- with the
    // (Z - W0) broadcast for the kernel rung, without it for the response's pi^dyn rung
    // -- trading one extra per-q gather sweep for never holding two copies.
    // The dynamic and secondary paths use the replicated slab.
    const bool lean = lin and not secondary();
    nda::array<ComplexType, 4> Wt_qtPQ((need_dyn and not lean) ? nqpts_ibz : 0, nt, Np, Np);
    if (need_dyn and not lean) {
      // gather the RPA-grid dW into the replicated tau slab the kernel needs
      nda::array<ComplexType, 4> W_half = vertex_redist_detail::gather_dW_replicated(
          mb_state.dW_qtPQ.value(), mpi->comm, nqpts_ibz, nt_half, Np);

      // head insertion, dynamic piece (weight Re[eps_inv_head(tau)]) into dW(Gamma, tau).
      // eps_inv_head = eps^-1_00(q->0, tau) - 1, stored on nt_half by scr_coulomb
      // (scr_coulomb_t.cpp); same Re[.] convention as Sigma_div_correction.
      if (head_ok) {
        if (_bl_head_static_all and lin) {
          // Balanced head: NO dynamic-slot head. The full static-weight head already sits in the
          // instantaneous slot (Z(Gamma) above), so the fluctuation dW = W - W0 -- and
          // with it dWw_lin, pi^dyn's rung, and every downstream consumer -- carries no
          // analytic head. See _bl_head_static_all.
          app_log(1, "  Sigma^C head insertion [H1 STATIC]: dynamic piece SKIPPED "
                     "(the static-weight head is in the instantaneous slot; dW is "
                     "analytic-head-free).");
        } else if (mb_state.eps_inv_head.has_value()) {
          auto& eps = mb_state.eps_inv_head.value();
          utils::check(eps.shape(0) == nt_half,
                       "vertex_t::eval_Sigma_C: eps_inv_head size {} != nt_half = {}.",
                       eps.shape(0), nt_half);
          for (long it = 0; it < nt_half; ++it)
            W_half(iq_gamma, it, all, all) += ComplexType(eps(it).real()) * H_PQ;
          app_log(1, "  Sigma^C head insertion: dynamic piece applied to dW(Gamma, tau) "
                     "with eps_inv_head(tau=0) = {}", eps(0).real());
        } else {
          dyn_head_missing("vertex_t::eval_Sigma_C");   // aborts unless vertex_allow_missing_head
          app_log(1, "  [WARNING] Sigma^C: dW is present but eps_inv_head is not in MBState "
                     "-- the DYNAMIC head\n"
                     "            piece is skipped (bare piece applied).");
        }
      }

      for (long it = 0; it < nt; ++it) {
        long ith = std::min(it, nt - it - 1);
        Wt_qtPQ(all, it, all, all) = W_half(all, ith, all, all);
      }
    }

    // ---- LEAN staging: the node-shared nu window + its builder ------------------------
    // ONE (nq_ibz, nw_b, Np, Np) window per NUMA node. Builder: for each q, ALL ranks run the collective
    // per-q gather (bit-identical to slicing the all-q gather); exactly one writer
    // per node per q head-augments Gamma, mirror-unfolds, tau->nu gemms into the
    // window row, and (with_cq) broadcast-adds the nu-constant (Z - W0). Content is
    // deterministic, so every node's window is identical.
    std::optional<vertex_pi::iaft_tools> wtls;
    std::optional<math::shm::shared_array<nda::array_view<ComplexType, 4>>> sWw;
    const bool lean_head_dyn = lean and head_ok and not _bl_head_static_all
                               and mb_state.eps_inv_head.has_value();
    if (lean) {
      wtls.emplace(*_ft);
      sWw.emplace(math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(
          *mpi, std::array<long, 4>{nqpts_ibz, wtls->nw_b, Np, Np}));
      if (head_ok) {
        if (_bl_head_static_all) {
          app_log(1, "  Sigma^C head insertion [H1 STATIC]: dynamic piece SKIPPED "
                     "(the static-weight head is in the instantaneous slot; dW is "
                     "analytic-head-free).");
        } else if (mb_state.eps_inv_head.has_value()) {
          utils::check(mb_state.eps_inv_head.value().shape(0) == nt_half,
                       "vertex_t::eval_Sigma_C: eps_inv_head size {} != nt_half = {}.",
                       mb_state.eps_inv_head.value().shape(0), nt_half);
          app_log(1, "  Sigma^C head insertion: dynamic piece applied to dW(Gamma, tau) "
                     "with eps_inv_head(tau=0) = {}",
                  mb_state.eps_inv_head.value()(0).real());
        } else {
          dyn_head_missing("vertex_t::eval_Sigma_C (lean staging)");   // aborts unless vertex_allow_missing_head
          app_log(1, "  [WARNING] Sigma^C: dW is present but eps_inv_head is not in MBState "
                     "-- the DYNAMIC head\n"
                     "            piece is skipped (bare piece applied).");
        }
      }
    }
    auto stage_lean_Ww = [&](bool with_cq) {
      auto W = sWw->local();
      auto& tl = wtls.value();
      auto const& W0b_l = _W0b_qmm.value();
      const long n2 = Np * Np;
      sWw->win().fence();
      nda::array<ComplexType, 2> Wt2;   // per-q unfolded tau slice, writer ranks only
      nda::array<ComplexType, 2> cq;
      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        auto W_q = vertex_redist_detail::gather_dW_one_q(
            mb_state.dW_qtPQ.value(), mpi->comm, iq, nt_half, Np);   // COLLECTIVE
        if (iq % mpi->node_comm.size() != mpi->node_comm.rank()) continue;
        if (lean_head_dyn and iq == iq_gamma) {
          auto& eps = mb_state.eps_inv_head.value();
          for (long it = 0; it < nt_half; ++it)
            W_q(it, all, all) += ComplexType(eps(it).real()) * H_PQ;
        }
        if (Wt2.size() == 0) Wt2 = nda::array<ComplexType, 2>(nt, n2);
        for (long it = 0; it < nt; ++it) {
          const long ith = std::min(it, nt - it - 1);
          Wt2(it, all) = nda::reshape(W_q(ith, all, all), std::array<long, 1>{n2});
        }
        auto out = nda::reshape(W(iq, all, all, all), std::array<long, 2>{tl.nw_b, n2});
        nda::blas::gemm(tl.Twt_bb, Wt2, out);
        if (with_cq) {
          if (cq.size() == 0) cq = nda::array<ComplexType, 2>(Np, Np);
          for (long M = 0; M < Np; ++M)
            for (long N = 0; N < Np; ++N)
              cq(M, N) = Z_qPQ(iq, M, N) - W0b_l(iq, M, N);
          for (long m = 0; m < tl.nw_b; ++m) W(iq, m, all, all) += cq;
        }
      }
      sWw->win().fence();
    };

    // ---- momentum maps (symmetry-free mesh) -------------------------------------------
    nda::array<long, 2> kmq(nqpts, nkpts);
    nda::array<long, 1> qmin(nqpts);
    for (long iq = 0; iq < nqpts; ++iq) {
      qmin(iq) = MF->qminus()(iq);
      for (long ik = 0; ik < nkpts; ++ik) kmq(iq, ik) = MF->qk_to_k2(iq, ik);
    }

    _Timer.stop("SIG_W_GATHER");
    _Timer.start("SIG_SECONDARY");
    // ---- optional secondary-basis substitution -----------------------------------------
    // The SAME kernel runs on the input set
    // (Xb, Zbar = t Z t^dag, Wbar = t dW t^dag, G_CC, window [0, nc)) -- fold-the-core;
    // the head-augmented Gamma cells above downfold automatically through t (rank-1
    // t H t^dag = (t conj(chi))(t conj(chi))^dag). Sigma^C externals are a, b in C and
    // land in the C-C block; NO upfold.
    const bool sec = secondary();
    const bool wan = _wannier;
    const long nc = subspace_rank();     // = M (Wannier) or _band_window.size() (window)
    nda::array<ComplexType, 3> Zb_qmm;
    nda::array<ComplexType, 4> Wb_qtmm;
    // STRICT C-C EXTERNALS: in Phi_2^C ALL FOUR G-lines -- including the cut one -- are
    // C-restricted, so Sigma^C = dPhi/dG is nonzero ONLY on the C-C block (window) /
    // range(P) (Wannier). BOTH paths run the kernel with C-restricted externals
    // (G_CC + the C columns of the collocation); the full-range extension of the
    // kernel formula is well-defined but is NOT dPhi/dG.
    // WINDOW: G_CC = the W-window block; WANNIER: G_CC = U^dag G U.
    // On the FULL BZ: image points are gauge copies of the IBZ blocks
    // (identity D by convention, symmetry.hpp); trev points are the tau-pointwise
    // TRANSPOSE (thc_solver_comm.hpp; == conj for the hermitian G). No tau-mirror
    // anywhere.
    // G_CC is node-shared (one copy per NUMA node, not one per rank). It is built from
    // the already node-shared sG_tskij and is READ (never written) by the kernel and
    // g_rotation_check, which take it via a templated array param -- so the
    // shared_array .local() view binds without any kernel change.
    auto sG_CC = math::shm::make_shared_array<nda::array_view<ComplexType, 5>>(
        *mpi, std::array<long, 5>{nt, ns, nkpts, nc, nc});
    sG_CC.win().fence();
    if (mpi->node_comm.root()) {
      auto G_CC = sG_CC.local();
      if (wan) {
        vertex_wannier_detail::build_Gbar_fullbz(G_tskij, _U_skia, _band_window, sym_mesh,
                                                 MF->kp_to_ibz(), MF->kp_trev(), G_CC);
      } else if (not sym_mesh) {
        G_CC = G_tskij(all, all, all, _band_window, _band_window);
      } else {
        auto kp_to_ibz = MF->kp_to_ibz();
        auto kp_trev = MF->kp_trev();
        for (long kp = 0; kp < nkpts; ++kp) {
          const long kib = kp_to_ibz(kp);
          if (not kp_trev(kp)) {
            G_CC(all, all, kp, all, all) = G_tskij(all, all, kib, _band_window, _band_window);
          } else {
            for (long it = 0; it < nt; ++it)
              for (long is = 0; is < ns; ++is)
                for (long a = 0; a < nc; ++a)
                  for (long b = 0; b < nc; ++b)
                    G_CC(it, is, kp, a, b) =
                        G_tskij(it, is, kib, _band_window.first() + b, _band_window.first() + a);
          }
        }
      }
    }
    sG_CC.win().fence();
    auto G_CC = sG_CC.local();
    app_log(2, "  Sigma^C externals restricted to {} a, b in [0, {}) "
               "(strict Phi cut; notes/refinement2_optionA.md DECISION 2).",
            wan ? "range(P) (Wannier labels)" : "the C-C block", nc);
    // effective window collocation: WINDOW = X(:,C); WANNIER = X_bar = X.U (Np x M).
    // (also the sym-ctx input; secondary uses Xb). orb0 of the pair matrices is 0 in
    // Wannier mode (X_C already carries exactly the M subspace columns).
    // X_C is node-shared too (one copy per node; built on node root from
    // the node-shared X_skPa). X_glob is a plain view selecting X_C (Wannier) / X_skPa
    // (window) -- both are array_view<ComplexType,4> so the ternary binds.
    auto sX_C = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(
        *mpi, std::array<long, 4>{ns, nkpts, Np, nc});
    sX_C.win().fence();
    if (mpi->node_comm.root()) {
      auto X_C_loc = sX_C.local();
      if (wan)
        X_C_loc = vertex_wannier_detail::build_Xbar(X_skPa, _U_skia, _band_window);
      else
        X_C_loc = X_skPa(all, all, all, _band_window);
    }
    sX_C.win().fence();
    auto X_C = sX_C.local();
    // the "global collocation" the secondary C(q)/eta refer to: X_bar (orb0=0) in
    // Wannier mode, X_skPa (orb0=C.first()) in window mode.
    auto X_glob = wan ? X_C : X_skPa;
    const long orb0_glob = wan ? 0 : _band_window.first();
    if (sec) {
      build_secondary_basis(thc, X_glob, orb0_glob, kmq, iq_gamma);
      app_log(1, "  Refinement 2: Sigma^C runs in the SECONDARY basis (N_m = {} vs "
                 "Np = {}); externals a, b in C.", _Nm, Np);
      // eta diagnostics on the rung arrays ACTUALLY consumed (test scale only: N_pair <= 4096)
      if (not need_dyn) {
        app_log(2, "  Refinement 2: eta diagnostic skipped in static-rung mode "
                   "(the only rung is W0bar, downfolded by build_w0).");
      } else if (ns * nkpts * nc * nc <= 4096) {
        vertex_secondary_detail::eta_max_over_q(
            "Z", X_glob, orb0_glob, nc, _Xb_skma, _t_qmP, kmq,
            [&](long iq) { return Z_qPQ(iq, all, all); });
        vertex_secondary_detail::eta_max_over_q(
            "dW(tau_0)", X_glob, orb0_glob, nc, _Xb_skma, _t_qmP, kmq,
            [&](long iq) { return Wt_qtPQ(iq, 0, all, all); });
        vertex_secondary_detail::eta_max_over_q(
            "dW(tau_mid)", X_glob, orb0_glob, nc, _Xb_skma, _t_qmP, kmq,
            [&](long iq) { return Wt_qtPQ(iq, nt / 2, all, all); });
      } else {
        app_log(2, "  Refinement 2: eta diagnostic skipped (N_pair = {} > 4096).",
                ns * nkpts * nc * nc);
      }
      // fold the cores at IBZ q (frequency-slice-wise; t is frequency-independent;
      // non-IBZ transfers are sourced through the sym ctx).
      // STATIC modes: W0bar is ALREADY the downfolded rung (folded in build_w0), and no
      // dynamic core exists -- nothing to fold here.
      if (not need_dyn) {
        // nothing to fold
      } else {
      Zb_qmm = nda::array<ComplexType, 3>(nqpts_ibz, _Nm, _Nm);
      Wb_qtmm = nda::array<ComplexType, 4>(nqpts_ibz, nt, _Nm, _Nm);
      nda::array<ComplexType, 2> tmp(_Nm, Np);
      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        auto t_q = _t_qmP(iq, all, all);
        vertex_secondary_detail::fold_core(t_q, Z_qPQ(iq, all, all), tmp,
                                           Zb_qmm(iq, all, all));
        for (long it = 0; it < nt; ++it)
          vertex_secondary_detail::fold_core(t_q, Wt_qtPQ(iq, it, all, all), tmp,
                                             Wb_qtmm(iq, it, all, all));
      }
      }
    }

    _Timer.stop("SIG_SECONDARY");
    _Timer.start("SIG_SYMCTX");
    // ---- IBZ symmetry context (trivial/null on symmetry-free meshes) ------------------
    // WANNIER: thread U through build_sym_ctx so the C-sector
    // rotation is d = U(Sk)^dag D U(k) and sym + Wannier compose. Secondary + Wannier +
    // symmetry is blocked by the rotated point-selection overload (nosym only), so the
    // secondary sym ctx is never U-rotated here.
    vertex_sym::sym_ctx const* symc = nullptr;
    if (sym_mesh) {
      if (sec) {
        build_sym_ctx(thc, _Xb_skma, _band_window.first(), _sym_secondary);
        symc = &_sym_secondary.value();
      } else {
        build_sym_ctx(thc, X_C, _band_window.first(), _sym_global,
                      wan ? &_U_skia : nullptr);
        symc = &_sym_global.value();
      }
      _g_rot_max = std::max(_g_rot_max,
                            vertex_ibz_detail::g_rotation_check(*symc, G_CC, MF->kp_trev()));
    }

    _Timer.stop("SIG_SYMCTX");
    _Timer.start("SIG_KERNEL");
    // ---- fused kernel (round-robin over (s,k,qx); result all-reduced inside) ----------
    // Both paths: C-restricted externals; the ONLY difference is the auxiliary input
    // set -- (X_C, W, Z, Np) global vs (Xb, Wbar, Zbar, N_m) secondary.
    nda::array<ComplexType, 5> Sigma_C(nt, ns, nk_ext, nc, nc);
    if (stat) {
      // B-S: BOTH rungs are W0bar. The kernel's doubly-instantaneous reduction S3 is
      // Sigma^{C,x}; families I-V and S1/S2 are identically zero and are
      // skipped, as is every pole-fit call. W0bar carries N_m (secondary) or Np (global)
      // -- the same array serves both paths -- and the dynamic W stub is empty.
      auto const& W0b = _W0b_qmm.value();
      const long naux = W0b.shape(1);
      utils::check(W0b.shape(0) == nqpts_ibz and W0b.shape(2) == naux,
                   "vertex_t::eval_Sigma_C: W0bar shape ({}, {}, {}) is inconsistent with "
                   "nqpts_ibz = {}.", W0b.shape(0), naux, W0b.shape(2), nqpts_ibz);
      nda::array<ComplexType, 4> Wstub(nqpts_ibz, 0, naux, naux);
      const int rmode = lin ? 2 : 1;
      // ---- B-L: the DYNAMIC FLUCTUATION rung, handed over in FREQUENCY --------------
      //   dW(q, i.nu) := W(q, i.nu) - W0(q) = [Zbar + dWbar(i.nu)] - W0bar
      // It vanishes at nu = 0 by construction (W0 IS the nu = 0 slice) and cannot be
      // routed through the kernel's tau slot: W0 differs from the bare core Zbar by the
      // CONSTANT dWbar(0), and a nu-constant is a delta(tau). With Z_slot = W0bar and
      // this as the dynamic rung, the kernel's own reductions ARE B-L's three terms:
      //   S3 = W0_x W0_y,  S1 = W0_x dW_y,  S2 = dW_x W0_y,
      // and families I-V are precisely the dropped dW_x dW_y of Phi^(2).
      nda::array<ComplexType, 4> dWw_lin;   // SECONDARY path only (N_m^2 -- small)
      std::optional<nda::array_view<ComplexType, 4>> Ww_v;
      nda::array_view<ComplexType, 4> const *wwp = nullptr;
      if (lin and not sec) {
        // LEAN: pass 1 of the node-shared window -- dW(nu) + (Z - W0), the kernel rung.
        stage_lean_Ww(/*with_cq=*/true);
        Ww_v.emplace(sWw->local());
        wwp = &Ww_v.value();
      }
      if (lin and sec) {
        vertex_pi::iaft_tools tls(*_ft);
        dWw_lin = nda::array<ComplexType, 4>(nqpts_ibz, tls.nw_b, naux, naux);
        // A per-q GEMM, not a 5-deep scalar loop with the reduction on the STRIDED `it`
        // axis. Both sources are contiguous C-order (nq, nt, naux, naux), so a q-slice
        // reshapes to (nt, naux^2) legally, and the m-INDEPENDENT (Z - W0) term is a
        // broadcast add AFTER the gemm instead of being recomputed inside the innermost
        // loop.
        {
          const long n2 = naux * naux;
          nda::array<ComplexType, 2> cq(naux, naux);
          for (long iq = 0; iq < nqpts_ibz; ++iq) {
            auto out = nda::reshape(dWw_lin(iq, all, all, all),
                                    std::array<long, 2>{tls.nw_b, n2});
            if (sec) {
              auto src = nda::reshape(Wb_qtmm(iq, all, all, all),
                                      std::array<long, 2>{nt, n2});
              nda::blas::gemm(tls.Twt_bb, src, out);
            } else {
              auto src = nda::reshape(Wt_qtPQ(iq, all, all, all),
                                      std::array<long, 2>{nt, n2});
              nda::blas::gemm(tls.Twt_bb, src, out);
            }
            for (long M = 0; M < naux; ++M)
              for (long N = 0; N < naux; ++N)
                cq(M, N) = (sec ? Zb_qmm(iq, M, N) : Z_qPQ(iq, M, N)) - W0b(iq, M, N);
            for (long m = 0; m < tls.nw_b; ++m)
              dWw_lin(iq, m, all, all) += cq;
          }
        }
        Ww_v.emplace(dWw_lin());
        wwp = &Ww_v.value();
      }
      if (lin) {
        vertex_pi::iaft_tools tls(*_ft);
        auto const& Wdw = *wwp;   // the nu-domain rung, whichever storage backs it
        double z0 = 0.0, zs = 0.0, zmax = 0.0;
        for (long iq = 0; iq < nqpts_ibz; ++iq)
          for (long M = 0; M < naux; ++M)
            for (long N = 0; N < naux; ++N) {
              z0 = std::max(z0, std::abs(Wdw(iq, tls.m0, M, N)));
              zs = std::max(zs, std::abs(W0b(iq, M, N)));
            }
        // THE EXPANSION PARAMETER, measured over ALL frequencies -- not the i.nu = 0 slice
        // reported below it. B-L is first order in dW, so |dW|/|W0| is what says whether
        // the tangent expansion is controlled, and the nu = 0 value CANNOT answer that:
        // W0 IS the nu = 0 slice, so dW is small there BY CONSTRUCTION and for a reason
        // that has nothing to do with convergence. dW(i.nu -> infinity) -> v - W0, i.e.
        // the whole of the screening; the nu = 0 value can be small while the first-order
        // (mixed) terms are comparable to or larger than the zeroth-order one.
        for (long iq = 0; iq < nqpts_ibz; ++iq)
          for (long m = 0; m < tls.nw_b; ++m)
            for (long M = 0; M < naux; ++M)
              for (long N = 0; N < naux; ++N)
                zmax = std::max(zmax, std::abs(Wdw(iq, m, M, N)));
        _diag_dw_rel = (zs > 0.0 ? zmax / zs : -1.0);
        app_log(1, "  Sigma^C [rung = linear]: EXPANSION PARAMETER max_nu |dW|/|W0| = "
                   "{:.4f}  (max_nu |dW| = {:.3e}, |W0| = {:.3e}). B-L is FIRST ORDER in "
                   "dW, so this -- not the i.nu = 0 value below -- is the validity meter; "
                   "dW is small at nu = 0 by construction (W0 is that slice) and tends to "
                   "v - W0 at large nu. A value near or above 1 means the tangent "
                   "expansion is NOT controlled.",
                _diag_dw_rel, zmax, zs);

        // ---- dW's OWN HEAD CHANNEL ------------------------------------------------
        // The meter above is a MAX-NORM, and a max-norm cannot see a rank-1 head: the Gamma
        // head can change the first-order mixed terms substantially while barely moving
        // max|dW|/|W0|. So the question is NOT how big dW is, it is whether the head
        // SURVIVES the W - W0 subtraction. Measure that directly, in the channel the head
        // lives in:
        //     h_A(q) := chi(q)^dag A(q) chi(q) / ||chi(q)||^2 ,  chi = thc.basis_head()
        //     ratio(q) := max_nu |h_dW(q, i.nu)| / |h_W0(q)|
        // The ratio alone is not diagnostic: it is of similar size at every q (including
        // the head-free q != Gamma), so the head does not make it anomalous; ignore_g0
        // instead makes Gamma anomalously QUIET, because with v(G = 0) zeroed there is
        // barely any G = 0 content to fluctuate. The two ABSOLUTE meters are reported:
        //   _diag_dw_head_abs -- how much dW sits in that one direction, in a.u., which is
        //     directly comparable across q -> 0 policies (a ratio is not);
        //   _diag_dw_head_coh -- |h_dW| / max|dW(Gamma)|, the ALIGNMENT with the rank-1
        //     direction, against the chi-aligned ceiling. A max-norm gate is structurally
        //     blind to it: a perturbation that is small element-wise but coherent is summed
        //     by the kernel over N_p^2 terms in phase.
        // The worst q != Gamma is a WITHIN-RUN head-free control -- no head is ever inserted
        // there, so it is the same channel at the same G in the same iteration.
        _diag_dw_head_rel = -1.0;
        _diag_dw_head_bg = -1.0;
        _diag_dw_head_abs = -1.0;
        _diag_dw_head_coh = -1.0;
        _diag_dw_head_nu0 = -1.0;
        {
          auto chi_h = thc.basis_head();            // (nqpts_ibz, Np)
          if (chi_h.shape(0) < nqpts_ibz or chi_h.shape(1) != Np
              or (not sec and naux != Np)) {
            app_log(2, "  Sigma^C [rung = linear]: head-channel meter SKIPPED -- "
                       "basis_head shape ({}, {}) is unusable for nqpts_ibz = {}, Np = {}.",
                    chi_h.shape(0), chi_h.shape(1), nqpts_ibz, Np);
          } else {
            // the largest |nu| on the bosonic mesh: where dW -> v - W0 and the head is at
            // its most bare. Reported so the nu-DEPENDENCE of the survival is visible and
            // not just its max.
            long mhi = 0;
            for (long m = 0; m < tls.nw_b; ++m)
              if (std::abs(tls.wn_b(m)) > std::abs(tls.wn_b(mhi))) mhi = m;
            nda::array<ComplexType, 1> p(naux);
            auto quad = [&](auto const& A) {                // chi^dag A chi (unnormalized)
              ComplexType s(0.0, 0.0);
              for (long M = 0; M < naux; ++M) {
                const ComplexType pm = std::conj(p(M));
                for (long N = 0; N < naux; ++N) s += pm * A(M, N) * p(N);
              }
              return s;
            };
            long bg_q = -1;
            for (long iq = 0; iq < nqpts_ibz; ++iq) {
              // the probe vector, expressed in the basis dW ACTUALLY lives in.
              //   GLOBAL   : chi itself.
              //   SECONDARY: the fold is Abar = t A t^dag, so w^dag Abar w =
              //              (t^dag w)^dag A (t^dag w). The probe representing chi is
              //              therefore w = t chi, which lands on the projection of chi
              //              onto the row space of t -- the only part of chi the
              //              secondary basis retains, which is the honest thing to
              //              measure since it is also the only part the kernel sees.
              //   (NOTE: this is NOT the head fold t conj(chi) of head_block_add. That is
              //    the image of the head MATRIX; this is a probe VECTOR and carries the
              //    opposite conjugation. Mixing them up silently measures nothing.)
              if (sec) {
                auto t_q = _t_qmP(iq, all, all);
                for (long m = 0; m < naux; ++m) {
                  ComplexType s(0.0, 0.0);
                  for (long P = 0; P < Np; ++P) s += t_q(m, P) * chi_h(iq, P);
                  p(m) = s;
                }
              } else {
                for (long P = 0; P < Np; ++P) p(P) = chi_h(iq, P);
              }
              double pn = 0.0, pinf = 0.0;
              for (long M = 0; M < naux; ++M) {
                pn += std::norm(p(M));
                pinf = std::max(pinf, std::abs(p(M)));
              }
              if (pn <= 0.0) continue;                      // no G = 0 content at this q
              const double hw0 = std::abs(quad(W0b(iq, all, all))) / pn;
              if (hw0 <= 0.0) continue;
              double hmax = 0.0, h0 = 0.0, hhi = 0.0, dwmax = 0.0;
              long m_at = -1;
              for (long m = 0; m < tls.nw_b; ++m) {
                // the quadratic form and the per-q max in ONE pass over the slice: dWw_lin
                // is the largest array in this routine, so the diagnostic sweeps it once.
                ComplexType s(0.0, 0.0);
                for (long M = 0; M < naux; ++M) {
                  const ComplexType pm = std::conj(p(M));
                  for (long N = 0; N < naux; ++N) {
                    const ComplexType a = Wdw(iq, m, M, N);
                    s += pm * a * p(N);
                    dwmax = std::max(dwmax, std::abs(a));
                  }
                }
                const double h = std::abs(s) / pn;
                if (h > hmax) { hmax = h; m_at = m; }
                if (m == tls.m0) h0 = h;
                if (m == mhi) hhi = h;
              }
              // COHERENCE. For a chi-aligned rank-1 matrix c p p^dag the channel value is
              // |c| ||p||^2 while the max element is |c| max_M|p_M|^2, so the ratio hits the
              // ceiling ||p||^2 / max_M|p_M|^2 (<= naux, = naux for a flat p). A matrix with
              // no preferred alignment gives O(1). This is the quantity a max-norm gate
              // cannot see, and it is what makes a "small" perturbation dominate a kernel
              // that sums N_p^2 terms in phase.
              const double coh = (dwmax > 0.0 ? hmax / dwmax : 0.0);
              const double coh_ceil = (pinf > 0.0 ? pn / (pinf * pinf) : 0.0);
              if (iq == iq_gamma) {
                _diag_dw_head_rel = hmax / hw0;
                _diag_dw_head_abs = hmax;
                _diag_dw_head_coh = (coh_ceil > 0.0 ? coh / coh_ceil : 0.0);
                _diag_dw_head_nu0 = h0;
                double w0max = 0.0;
                for (long M = 0; M < naux; ++M)
                  for (long N = 0; N < naux; ++N)
                    w0max = std::max(w0max, std::abs(W0b(iq, M, N)));
                app_log(1, "  Sigma^C [rung = linear]: HEAD CHANNEL at q = Gamma "
                           "(h_A := chi^dag A chi / ||chi||^2, the G = 0 / rank-1 head "
                           "direction):\n"
                           "      h_W0 = {:.4e}  (max|W0(Gamma)| = {:.4e}; the channel "
                           "carries {:.1f} % of the block)\n"
                           "      h_dW : nu = 0 {:.4e} | MAX {:.4e} (n = {}) | largest "
                           "|nu| (n = {}) {:.4e}   -> the nu = 0 slice understates by "
                           "{:.1f}x\n"
                           "      ratio max_nu |h_dW| / |h_W0| = {:.4f}   vs the max-norm "
                           "meter {:.4f} ({:.2f}x)\n"
                           "      COHERENCE |h_dW| / max|dW(Gamma)| = {:.2f} of the "
                           "chi-aligned rank-1 ceiling {:.2f}  ->  {:.3f}\n"
                           "      Read the ABSOLUTE and the COHERENCE, not the ratio: the "
                           "ratio is ~0.4 at EVERY q\n"
                           "      (see the control below), so it is not what the head "
                           "changes. What the head changes is\n"
                           "      how much sits in ONE rank-1 direction that the kernel "
                           "then sums over N_p^2 terms in phase.\n"
                           "      COHERENCE -> 1 means dW(Gamma) IS c chi chi^dag, i.e. the "
                           "head did not cancel in W - W0 at all.",
                        hw0, w0max, (w0max > 0.0 ? 100.0 * hw0 / w0max : 0.0),
                        h0, hmax, tls.wn_b(m_at >= 0 ? m_at : 0), tls.wn_b(mhi), hhi,
                        (h0 > 0.0 ? hmax / h0 : 0.0),
                        _diag_dw_head_rel, _diag_dw_rel,
                        (_diag_dw_rel > 0.0 ? (hmax / hw0) / _diag_dw_rel : 0.0),
                        coh, coh_ceil, _diag_dw_head_coh);
              } else {
                app_log(2, "  Sigma^C [rung = linear]: head channel q = {}: h_W0 = {:.4e}, "
                           "max_nu |h_dW| = {:.4e}, ratio {:.4f}, coherence {:.2f} / {:.2f}",
                        iq, hw0, hmax, hmax / hw0, coh, coh_ceil);
                if (hmax / hw0 > _diag_dw_head_bg) {
                  _diag_dw_head_bg = hmax / hw0;
                  bg_q = iq;
                }
              }
            }
            if (bg_q >= 0)
              app_log(1, "  Sigma^C [rung = linear]: head-channel CONTROL, worst q != "
                         "Gamma (q = {}): max_nu |h_dW| / |h_W0| = {:.4f}. No head is "
                         "inserted at q != Gamma,\n"
                         "      so this is the same channel WITHOUT the analytic head, at "
                         "the same G and the same iteration.",
                      bg_q, _diag_dw_head_bg);
          }
        }

        // |dW(i.nu = 0)| = |W(q,0) - W0(q)| is the VERTEX CORRECTION TO THE STATIC
        // SCREEN, and in B-L it is nonzero BY DESIGN: the kernel W0[G] is the RPA-static
        // screen, while the run's own W carries P^{C,L}. The self-slice identity
        // W(q,0) == W0(q) holds in B-S (where P = P_RPA) but NOT in B-L. This is a
        // definition matching standard BSE practice (the BSE kernel is the RPA-screened
        // static W, not a self-consistently excitonic-screened one), and the difference
        // is O(vertex^2), beyond the order of the theory. Reported as a diagnostic; a
        // LARGE value means the vertex is strongly reshaping its own kernel and the
        // tangent expansion is being pushed.
        app_log(1, "  Sigma^C [rung = linear]: vertex correction to the static screen "
                   "|W(q,0) - W0(q)|_max = {:.3e} (|W0| = {:.3e}, ratio {:.4f}). Nonzero "
                   "by design in B-L (W0 is RPA-static); it is identically zero in B-S.",
                z0, zs, (zs > 0.0 ? z0 / zs : 0.0));
        app_log(1, "  Sigma^C [rung = linear]: three explicit terms "
                   "W0_x W_y + W_x W0_y - W0_x W0_y via the kernel's S1/S2/S3 reductions "
                   "(ONE bosonic convolution each; families I-V = Phi^(2) skipped).");
      } else {
        app_log(1, "  Sigma^C [rung = static]: Sigma^(C,x) only "
                   "(tau-local G^3 (W0)^2; no convolution at all).");
      }
      if (sec)
        vertex_detail::eval_sigma_C_g3w2(*_ft, mpi->comm, nda::range(0, nc), G_CC,
                                         _Xb_skma, Wstub, W0b, kmq, qmin, iq_gamma,
                                         skip_rung_gamma, rmode, wwp, symc, Sigma_C);
      else
        vertex_detail::eval_sigma_C_g3w2(*_ft, mpi->comm, nda::range(0, nc), G_CC,
                                         X_C, Wstub, W0b, kmq, qmin, iq_gamma,
                                         skip_rung_gamma, rmode, wwp, symc, Sigma_C);
    } else if (sec)
      vertex_detail::eval_sigma_C_g3w2(*_ft, mpi->comm, nda::range(0, nc), G_CC,
                                       _Xb_skma, Wb_qtmm, Zb_qmm, kmq, qmin,
                                       iq_gamma, skip_rung_gamma, 0,
                                       static_cast<nda::array<ComplexType, 4> const*>(nullptr),
                                       symc, Sigma_C);
    else
      vertex_detail::eval_sigma_C_g3w2(*_ft, mpi->comm, nda::range(0, nc), G_CC,
                                       X_C, Wt_qtPQ, Z_qPQ, kmq, qmin, iq_gamma,
                                       skip_rung_gamma, 0,
                                       static_cast<nda::array<ComplexType, 4> const*>(nullptr),
                                       symc, Sigma_C);
    {
      double max_abs = 0.0;
      long n_bad = 0;
      for (auto const& v : Sigma_C) {
        double a = std::abs(v);
        if (not std::isfinite(a)) { ++n_bad; continue; }
        max_abs = std::max(max_abs, a);
      }
      utils::check(n_bad == 0,
                   "vertex_t::eval_Sigma_C: Sigma^C contains {} NaN/Inf entries -- aborting.", n_bad);
      app_log(2, "  Sigma^C(tau) max|.| = {}\n", max_abs);
    }

    _Timer.stop("SIG_KERNEL");
    _Timer.start("SIG_RESPONSE");
    // ---- the RESPONSE cut Sigma^{C,r} -------------------------------------------------
    // Phi-derivability of B-S requires Sigma^{C,x} and Sigma^{C,r} TOGETHER: W0 is an
    // explicit functional of the CURRENT G, so differentiating Phi produces this chain-
    // rule term as well. The routing (the transposed, symmetrized sandwich) is checked
    // end-to-end against finite differences by test_vertex_fdoracle.
    nda::array<ComplexType, 5> Sigma_r;
    if (stat) {
      // IBZ (symmetry-adapted): Sigma^{C,r} follows the GW construction --
      // gw_t::eval_Sigma_all_k_impl / "Low-Scaling algorithms for GW and cRPA using
      // symmetry-adapted ISDF". The two-body rung Delta w is built and stored at IBZ q
      // ONLY; the aux-dressed Gt (which is NOT symmetric) is rebuilt on the full BZ per
      // tau from the IBZ orbital G; and the D matrices rotate the BAND indices of the
      // finished self-energy. This works for Sigma^{C,r} -- unlike Sigma^C / Pi^C, which
      // must rotate collocation legs via Xhat -- because it carries a single transfer and
      // both external legs sit at the same k.
      auto const& W0b_r = _W0b_qmm.value();
      vertex_pi::iaft_tools tools(*_ft);
      nda::array<long, 2> kpq(nqpts, nkpts);
      for (long iq = 0; iq < nqpts; ++iq)
        for (long ik = 0; ik < nkpts; ++ik) kpq(iq, ik) = kmq(qmin(iq), ik);

      // (1) Pi^{C,0}(q, i.nu): the instantaneous (Z) phase with the rung W0bar and
      //     NO dynamic rung -- pi_c_accumulate_w returns right after phase 1 on nullptr.
      //     Fed the C-C block: the kernel CONTRACTS its external orbital legs into the aux
      //     indices, so their range is part of the object (all eight labels of Phi are in C).
      const long Naux_pi = sec ? _Nm : Np;
      // The same slab as eval_Pi_C's accumulator: on the global path a full-shape
      // response-stage accumulator would be the (nw_b, nq_ibz, Np, Np) array per rank,
      // which bounds the memory of the Sigma stage of every STATIC theory (B-S included,
      // which the lean-W staging does not touch: need_dyn is false there). Only the owned
      // +-q-orbit rows are stored; the (linear) tau = 0 row is applied to the slab below
      // and the SMALL Pi0 is what gets reduced. The secondary path (N_m^2) stays
      // full-shape.
      std::optional<vertex_pi::pi_qext_plan> pi0_plan;
      if (not sec)
        pi0_plan.emplace(vertex_pi::make_pi_qext_plan(mpi->comm.rank(), mpi->comm.size(),
                                                      ns * nkpts * nqpts, nqpts_ibz,
                                                      qmin));
      const long n_pi0_rows = pi0_plan ? long(pi0_plan->owned.size()) : nqpts_ibz;
      nda::array<ComplexType, 4> Pi_wq(tools.nw_b, n_pi0_rows, Naux_pi, Naux_pi);
      Pi_wq() = ComplexType(0.0);
      // symc MUST be threaded through: on an IBZ mesh the kernel's external q axis is
      // nqpts_ibz while kmq/kpq carry the FULL transfer mesh, and it sources non-IBZ
      // rung transfers through Xhat.
      // SUB-STAGE (inclusive in SIG_RESPONSE): the INSTANTANEOUS phase. Wdyn is
      // nullptr, so pi_c_accumulate_w returns after phase 1 and never touches the aux
      // pole basis. Compare against SIG_RESP_PIDYN below -- the ratio is the cost of the
      // dynamic phase.
      _Timer.start("SIG_RESP_PI0");
      if (sec)
        vertex_pi::pi_c_accumulate_w(*_ft, tools, G_CC, _Xb_skma, W0b_r, static_cast<nda::array<ComplexType, 4> const*>(nullptr),
                                     kmq, kpq, nda::range(0, nc), Pi_wq,
                                     mpi->comm.rank(), mpi->comm.size(),
                                     skip_rung_gamma, nullptr, symc);
      else
        vertex_pi::pi_c_accumulate_w(*_ft, tools, G_CC, X_C, W0b_r, static_cast<nda::array<ComplexType, 4> const*>(nullptr),
                                     kmq, kpq, nda::range(0, nc), Pi_wq,
                                     mpi->comm.rank(), mpi->comm.size(),
                                     skip_rung_gamma, nullptr, symc, nullptr,
                                     &pi0_plan.value());
      // The full-array all_reduce is used only on the (small, N_m^2) secondary path.
      // On the global path the tau = 0 row is applied to the SLAB partial below and the
      // reduction moves to the small Pi0 -- the full array is never materialized, let
      // alone summed.
      if (sec)
        mpi->comm.all_reduce_in_place_n(Pi_wq.data(), Pi_wq.size(), std::plus<>{});
      _Timer.stop("SIG_RESP_PI0");

      // (2) the tau = 0 row (the LEGAL evaluation of (1/beta) sum_nu; sparse nodes are
      //     fitting nodes, not Fourier points)
      auto R0 = vertex_w0_detail::tau0_transform_row(*_ft);
      // The reduction runs over the LEADING axis of Pw, i.e. inner stride
      // nqpts_ibz * Naux_pi^2 -- the worst possible access pattern for a loop. Written as
      // ONE gemm on the (nw_b, nq * Naux_pi^2) reshape, with R0 as a 1 x nw_b row so no
      // transpose is needed.
      const long tau0_ncol = nqpts_ibz * Naux_pi * Naux_pi;
      auto R0row = nda::reshape(R0, std::array<long, 2>{1, tools.nw_b});
      auto tau0_of = [&](nda::array<ComplexType, 4> const &Pw,
                         nda::array<ComplexType, 3> &out) {
        auto A = nda::reshape(Pw, std::array<long, 2>{tools.nw_b, tau0_ncol});
        auto y = nda::reshape(out, std::array<long, 2>{1, tau0_ncol});
        nda::blas::gemm(R0row, A, y);
      };
      nda::array<ComplexType, 3> Pi0(nqpts_ibz, Naux_pi, Naux_pi);
      if (pi0_plan) {
        // SLAB: the same per-cell m-sum gemm as tau0_of, on the owned rows only. tau0
        // is linear, so applying it before the rank sum and reducing Pi0 instead of
        // Pi_wq agrees up to the summation order (rounding level).
        nda::array<ComplexType, 3> Pi0_slab(n_pi0_rows, Naux_pi, Naux_pi);
        {
          const long ncol = n_pi0_rows * Naux_pi * Naux_pi;
          auto A = nda::reshape(Pi_wq, std::array<long, 2>{tools.nw_b, ncol});
          auto y = nda::reshape(Pi0_slab, std::array<long, 2>{1, ncol});
          nda::blas::gemm(R0row, A, y);
        }
        Pi0() = ComplexType(0.0);
        for (long r = 0; r < n_pi0_rows; ++r)
          Pi0(pi0_plan->owned[r], all, all) = Pi0_slab(r, all, all);
        mpi->comm.all_reduce_in_place_n(Pi0.data(), Pi0.size(), std::plus<>{});
      } else {
        tau0_of(Pi_wq, Pi0);
      }

      nda::array<ComplexType, 3> PiStat;   // DIAGNOSTIC: B-S middle factor, kept for comparison
      // ---- B-L's response middle factor ---------------------------------------------
      // B-S sandwiches Pi^{C,0}(tau=0); B-L sandwiches the DIFFERENCE
      //     Pi^L = pi^dyn - Pi^{C,0}(tau = 0),   pi^dyn = Pi^{C,dyn}(q, tau = 0),
      // because the rung derivative of the tangent functional is
      //     X^L = -(1/2)[PiBar^dyn - PiBar^0]  (transposed/symmetrized as for B-S).
      // X^L therefore VANISHES when the screening is genuinely static: it is a built-in,
      // per-q meter of the static-kernel approximation itself, logged below.
      //
      // pi^dyn IS THE EQUAL-TIME VALUE ONLY, so it is evaluated by a factorized formula:
      // the external frequency sum closes the (12)/(34) G-pairs and leaves ONE bosonic
      // pairing of two ordinary bubbles against W, with no twisted pairs and no pole
      // algebra at all (vertex_pi::pi_dyn_factorized). The alternative route -- run the
      // FULL dynamic-rung Pi^C over every nw_b frequency, then keep the tau = 0 row -- is
      // far more expensive and is B-L's only contact with the aux pole basis; it stays
      // reachable as vertex_pidyn = "kernel", and "check" runs both and gates their
      // agreement.
      if (lin) {
        // SUB-STAGE (inclusive in SIG_RESPONSE): the tau -> i.nu transform of the dynamic
        // rung (a per-q GEMM).
        _Timer.start("SIG_RESP_WDYNW");
        nda::array<ComplexType, 4> Wdyn_w;   // SECONDARY path only (N_m^2 -- small)
        std::optional<nda::array_view<ComplexType, 4>> Wdyn_v;
        // SECONDARY: per-q gemm on the (nt, Naux_pi^2) reshape, as for dWw_lin above.
        // LEAN (global): pass 2 of the node-shared window -- the SAME per-q staging
        // WITHOUT the (Z - W0) broadcast, overwriting pass 1 in place. This is the pure
        // tau->nu transform of the (head-augmented) dW, at zero additional memory.
        if (sec) {
          Wdyn_w = nda::array<ComplexType, 4>(nqpts_ibz, tools.nw_b, Naux_pi, Naux_pi);
          const long n2 = Naux_pi * Naux_pi;
          for (long iq = 0; iq < nqpts_ibz; ++iq) {
            auto out = nda::reshape(Wdyn_w(iq, all, all, all),
                                    std::array<long, 2>{tools.nw_b, n2});
            auto src = nda::reshape(Wb_qtmm(iq, all, all, all),
                                    std::array<long, 2>{nt, n2});
            nda::blas::gemm(tools.Twt_bb, src, out);
          }
          Wdyn_v.emplace(Wdyn_w());
        } else {
          stage_lean_Ww(/*with_cq=*/false);
          Wdyn_v.emplace(sWw->local());
        }
        auto& Wdyn = Wdyn_v.value();
        _Timer.stop("SIG_RESP_WDYNW");

        // ---- DIAGNOSTIC KNOB: freeze the Gamma head's frequency dependence -------------
        // The Gamma rung head is inserted as Re[eps^-1_head(tau)] * H into dW(Gamma,tau),
        // so in frequency its weight runs from eps^-1(i.nu=0) (fully screened) to 1
        // (unscreened) -- it is the MOST frequency-dependent part of the whole rung. Pi^0
        // uses a rung whose head is FROZEN at eps^-1(i.nu=0) (W0), and Pi^0 satisfies the
        // q->0 head suppression exactly; pi^dyn uses the varying one and does NOT. This
        // knob replaces the varying weight by the constant i.nu=0 value inside pi^dyn's
        // rung ONLY, which is precisely the difference between the two objects.
        //
        // If <H, pi^dyn> collapses under this, the head insertion's nu-dependence is what
        // breaks conservation, and the head-channel projection is masking a real defect
        // rather than cleaning a residue. Diagnostic only -- freezing the head is NOT
        // physical (W's head really is strongly retarded); default OFF.
        // (interlock: under the balanced head (_bl_head_static_all) no dynamic head is inserted,
        //  so there is nothing to freeze -- adding the difference here would CREATE a
        //  spurious head. The two knobs are mutually exclusive by construction.)
        if (_bl_static_head and head_ok and not sec and not _bl_head_static_all and
            mb_state.eps_inv_head.has_value()) {
          auto& eps = mb_state.eps_inv_head.value();
          nda::array<ComplexType, 2> epst(nt, 1), epsw(tools.nw_b, 1);
          for (long it = 0; it < nt; ++it) {
            const long ith = std::min(it, nt - it - 1);
            epst(it, 0) = ComplexType(eps(ith).real());
          }
          nda::blas::gemm(tools.Twt_bb, epst, epsw);       // tau -> i.nu, same map as W
          const ComplexType w0v = epsw(tools.m0, 0);
          double dmax = 0.0;
          // this knob is global-path only (`not sec` above), so Wdyn is the node-shared
          // window: exactly one writer per node, fenced.
          if (sWw.has_value()) sWw->win().fence();
          for (long l = 0; l < tools.nw_b; ++l) {
            const ComplexType d = w0v - epsw(l, 0);        // freeze at the i.nu=0 value
            dmax = std::max(dmax, std::abs(d));
            if (not mpi->node_comm.root()) continue;
            for (long P = 0; P < Np; ++P)
              for (long Q = 0; Q < Np; ++Q)
                Wdyn(iq_gamma, l, P, Q) += d * H_PQ(P, Q);
          }
          if (sWw.has_value()) sWw->win().fence();
          app_log(1, "  [STATICHEAD] pi^dyn's Gamma head FROZEN at its i.nu=0 weight "
                     "{:.6e}; max|eps^-1(i.nu=0) - eps^-1(i.nu)| = {:.4e} (DIAGNOSTIC -- "
                     "not physical; tests whether the head's nu-dependence is what breaks "
                     "the q->0 head suppression of pi^dyn)",
                  w0v.real(), dmax);
        }
        // ---- DIAGNOSTIC KNOB: THE CONSTANT-RUNG ABSOLUTE PIN --------------------------
        // Overwrite the dynamic rung with the frequency-INDEPENDENT  W0bar - Z, so that
        // pi^dyn's total rung  Z + Wdyn_w(i.nu)  becomes exactly W0bar at every i.nu --
        // bit-for-bit the rung Pi^{C,0} was built with above. The two
        // objects are then the same integral evaluated by two different routes, and
        //     X^L -> 0        <H, pi^dyn> -> <H, Pi^{C,0}>
        // must follow to the DLR representability floor. Anything O(1) surviving here is a
        // defect in the equal-time path that Pi^{C,0} does not share.
        //
        // NOTE: NOT "zero the dynamic rung": that would leave pi^dyn with the BARE rung Z
        // while Pi^{C,0} keeps W0, so X^L would stay O(1) for a reason that says nothing
        // about the equal-time path. See _bl_pidyn_const_rung.
        if (_bl_pidyn_const_rung) {
          utils::check(W0b_r.shape(0) == nqpts_ibz and W0b_r.shape(1) == Naux_pi
                           and W0b_r.shape(2) == Naux_pi,
                       "vertex_t::eval_Sigma_C: const-rung pin: W0bar shape ({}, {}, {}) "
                       "does not match pi^dyn's rung ({}, {}, {}).",
                       W0b_r.shape(0), W0b_r.shape(1), W0b_r.shape(2),
                       nqpts_ibz, Naux_pi, Naux_pi);
          double dmax = 0.0, w0max = 0.0;
          // sec: per-rank array, every rank writes its own copy.
          // lean: node-shared window -- one writer per node, fenced; the scalars are
          // computed on every rank (deterministic) so rank 0 always has them to log.
          if (sWw.has_value()) sWw->win().fence();
          const bool wmut = sec or mpi->node_comm.root();
          for (long iq = 0; iq < nqpts_ibz; ++iq)
            for (long M = 0; M < Naux_pi; ++M)
              for (long N = 0; N < Naux_pi; ++N) {
                const ComplexType d =
                    W0b_r(iq, M, N) - (sec ? Zb_qmm(iq, M, N) : Z_qPQ(iq, M, N));
                dmax = std::max(dmax, std::abs(d));
                w0max = std::max(w0max, std::abs(W0b_r(iq, M, N)));
                // Total rung becomes Z + (W0bar - Z) = W0bar at every frequency. Which
                // slot carries the static content is irrelevant -- the rung enters as
                // Zc + Wd(i.nu); see _bl_pidyn_const_rung.
                if (wmut)
                  for (long l = 0; l < tools.nw_b; ++l) Wdyn(iq, l, M, N) = d;
              }
          if (sWw.has_value()) sWw->win().fence();
          app_log(1, "  [CONSTRUNG] pi^dyn's rung forced STATIC and equal to W0bar at "
                     "every i.nu:\n"
                     "            max|W0bar - Z| = {:.4e}, max|W0bar| = {:.4e}. "
                     "DIAGNOSTIC -- not physical.\n"
                     "            THE PIN: pi^dyn and Pi^(C,0) are now ONE integral by two "
                     "routes, so the X^L line\n"
                     "            below must collapse to the grid's representability floor "
                     "and [HEADPROJ] must show\n"
                     "            |<H,pi^dyn>| ~ |<H,Pi^0>|. Anything larger is a defect in "
                     "the equal-time path.",
                  dmax, w0max);
        }
        nda::array<ComplexType, 3> const& Zpi_rung = (sec ? Zb_qmm : Z_qPQ);
        nda::array<ComplexType, 3> PiDyn0(nqpts_ibz, Naux_pi, Naux_pi);
        // ---- ROUTE A: the factorized equal-time primitive (default) --------------------
        // SUB-STAGE (inclusive in SIG_RESPONSE). Pole-free: two bubble builds and one
        // bosonic pairing per (q, k, qx), with the externals folded onto (M, N) ONCE after
        // the nu_x sum. Checked against ROUTE B on identical inputs by
        // test_methods_vertex_pibardynfact.
        if (_pidyn_mode != 1) {
          _Timer.start("SIG_RESP_PIDYNF");
          PiDyn0() = ComplexType(0.0);
          if (sec)
            vertex_pi::pi_dyn_factorized(tools, G_CC, _Xb_skma, Zpi_rung, &Wdyn,
                                         kmq, kpq, nda::range(0, nc), R0, PiDyn0,
                                         mpi->comm.rank(), mpi->comm.size(),
                                         skip_rung_gamma, nullptr, symc);
          else
            vertex_pi::pi_dyn_factorized(tools, G_CC, X_C, Zpi_rung, &Wdyn,
                                         kmq, kpq, nda::range(0, nc), R0, PiDyn0,
                                         mpi->comm.rank(), mpi->comm.size(),
                                         skip_rung_gamma, nullptr, symc);
          mpi->comm.all_reduce_in_place_n(PiDyn0.data(), PiDyn0.size(), std::plus<>{});
          _Timer.stop("SIG_RESP_PIDYNF");
        }
        // ---- ROUTE B: the full dynamic-rung kernel, kept as a cross-check -------------
        // Runs the FULL dynamic-rung Pi^C -- including phase 2's twisted-pair pole algebra
        // over all nw_b frequencies -- and keeps only the tau = 0 row. Its timer against
        // SIG_RESP_PIDYNF gives the speedup of the factorized route, and it is the ONLY
        // thing that exposes B-L to the aux pole basis.
        if (_pidyn_mode != 0) {
          _Timer.start("SIG_RESP_PIDYN");
          // Full-shape BY DESIGN: route B is a default-off cross-check, tau0_of below
          // wants the full q axis, and slabbing a diagnostic route buys nothing.
          nda::array<ComplexType, 4> Pid_wq(tools.nw_b, nqpts_ibz, Naux_pi, Naux_pi);
          Pid_wq() = ComplexType(0.0);
          if (sec)
            vertex_pi::pi_c_accumulate_w(*_ft, tools, G_CC, _Xb_skma, Zpi_rung, &Wdyn,
                                         kmq, kpq, nda::range(0, nc), Pid_wq,
                                         mpi->comm.rank(), mpi->comm.size(),
                                         skip_rung_gamma, nullptr, symc);
          else
            vertex_pi::pi_c_accumulate_w(*_ft, tools, G_CC, X_C, Zpi_rung, &Wdyn,
                                         kmq, kpq, nda::range(0, nc), Pid_wq,
                                         mpi->comm.rank(), mpi->comm.size(),
                                         skip_rung_gamma, nullptr, symc);
          mpi->comm.all_reduce_in_place_n(Pid_wq.data(), Pid_wq.size(), std::plus<>{});
          _Timer.stop("SIG_RESP_PIDYN");
          if (_pidyn_mode == 1) {
            tau0_of(Pid_wq, PiDyn0);
          } else {
            // CHECK: both routes ran. WHAT THIS CAN AND CANNOT GATE:
            // the two routes are exact Matsubara sums of DIFFERENT integrands read through
            // the same tau = 0 row, so their agreement floor is the bosonic
            // REPRESENTABILITY of each integrand. That floor is NOT a fixed multiple of eps:
            // its prefactor grows with beta*wmax (see
            // test_vertex_pibardynfact/production_grid_attribution) AND it is data
            // dependent (it can change between scf iterations, and is large at
            // prec = "low"). So an eps-derived ABORT threshold is unreachable by
            // construction and would only produce flaky failures.
            //
            // What this check really discriminates is a ROUTING or PLUMBING break, and every
            // known mis-routing gives an O(1) deviation. So: WARN
            // whenever the deviation exceeds the grid floor (that is a real and actionable
            // statement -- pi^dyn is grid-limited, tighten iaft prec), and ABORT only above
            // a hard O(1) bar that no representability effect can reach. An explicit
            // vertex_pidyn_tol overrides the abort bar for callers who want it strict.
            const double warn_at = std::max(1e-8, 1e2 * _ft->eps());
            const double ctol = (_pidyn_check_tol > 0.0) ? _pidyn_check_tol : 0.25;
            nda::array<ComplexType, 3> PiK(nqpts_ibz, Naux_pi, Naux_pi);
            tau0_of(Pid_wq, PiK);
            double dnum = 0.0, dden = 0.0;
            for (long iq = 0; iq < nqpts_ibz; ++iq)
              for (long M = 0; M < Naux_pi; ++M)
                for (long N = 0; N < Naux_pi; ++N) {
                  dnum = std::max(dnum, std::abs(PiDyn0(iq, M, N) - PiK(iq, M, N)));
                  dden = std::max(dden, std::abs(PiK(iq, M, N)));
                }
            const double drel = (dden > 0.0) ? dnum / dden : dnum;
            _pidyn_check_max = std::max(_pidyn_check_max, drel);
            app_log(1, "  vertex_pidyn = check: |pi^dyn(factorized) - pi^dyn(kernel)| = "
                       "{:.4e}, max|pi^dyn(kernel)| = {:.4e}, rel = {:.3e} "
                       "(warn > {:.1e}, abort > {:.1e}; iaft eps = {:.1e})",
                    dnum, dden, drel, warn_at, ctol, _ft->eps());
            if (drel > warn_at)
              app_warning("vertex_t::eval_Sigma_C: pi^dyn is GRID-LIMITED -- the factorized "
                          "and kernel routes agree only to rel = {:.3e} at iaft eps = {:.1e}. "
                          "Both are exact in the continuum, so this is the DLR "
                          "representability floor of the tau = 0 read, and it bounds pi^dyn's "
                          "accuracy BY EITHER ROUTE, not just the factorized one. If B-L "
                          "needs pi^dyn tighter, the lever is iaft prec (\"medium\" = 1e-10, "
                          "\"high\" = 1e-13), NOT vertex_pidyn.", drel, _ft->eps());
            utils::check(drel <= ctol,
                         "vertex_t::eval_Sigma_C: the eq:pibardynfact factorized pi^dyn "
                         "disagrees with the dynamic-rung kernel at tau = 0 by rel = {}, "
                         "above the O(1) abort bar {} (iaft eps = {}). This is too large to be "
                         "the representability floor -- the closest mis-routing the routing "
                         "pin rejects sits at 1.24 -- so suspect a ROUTING or PLUMBING break, "
                         "not the grid. Do NOT raise this bar to make a run proceed: confirm "
                         "first with test_methods_vertex_pibardynfact, whose "
                         "production_grid_attribution section separates the two (the floor "
                         "FALLS with iaft eps; a routing bug does not).",
                         drel, ctol, _ft->eps());
          }
        }
        // ---- DIAGNOSTIC: pair symmetry of the two middle factors -----------------------
        // build_delta_w's assume_reflection path takes the PLAIN TRANSPOSE Pi(q)^T in
        // place of 1/2[Pi(q)^T + Pi(-q)], justified by Pi(-q) = Pi(q)^T. At a SELF-INVERSE
        // transfer (every q of a Gamma-centred 2x2x2 mesh) that identity reads
        // Pi(q) = Pi(q)^T, i.e. the block must be SYMMETRIC. pi^dyn comes from a DIFFERENT
        // algorithm (the factorized (12)/(34) rung-frequency grouping) than Pi^0's
        // (14)/(23) external-frequency kernel, so both are measured, per q.
        {
          auto qm_map = MF->qminus();
          for (long iq = 0; iq < nqpts_ibz; ++iq) {
            double a0 = 0.0, s0 = 0.0, ad = 0.0, sd = 0.0;
            for (long M = 0; M < Naux_pi; ++M)
              for (long N = 0; N < Naux_pi; ++N) {
                a0 = std::max(a0, std::abs(Pi0(iq, M, N) - Pi0(iq, N, M)));
                s0 = std::max(s0, std::abs(Pi0(iq, M, N)));
                ad = std::max(ad, std::abs(PiDyn0(iq, M, N) - PiDyn0(iq, N, M)));
                sd = std::max(sd, std::abs(PiDyn0(iq, M, N)));
              }
            app_log(1, "  [PAIRSYM] q = {} (qminus = {}, self-inverse = {}): "
                       "asym(Pi^0) = {:.3e} / {:.3e} = {:.3e};  "
                       "asym(pi^dyn) = {:.3e} / {:.3e} = {:.3e}",
                    iq, qm_map(iq), (qm_map(iq) == iq ? "YES" : "no"),
                    a0, s0, (s0 > 0.0 ? a0 / s0 : 0.0),
                    ad, sd, (sd > 0.0 ? ad / sd : 0.0));
          }
        }
        // keep the B-S middle factor so the two sandwiches can be compared below
        PiStat = nda::array<ComplexType, 3>(Pi0);
        double xl = 0.0, p0 = 0.0;
        for (long iq = 0; iq < nqpts_ibz; ++iq)
          for (long M = 0; M < Naux_pi; ++M)
            for (long N = 0; N < Naux_pi; ++N) {
              const ComplexType d = PiDyn0(iq, M, N) - Pi0(iq, M, N);
              xl = std::max(xl, std::abs(d));
              p0 = std::max(p0, std::abs(Pi0(iq, M, N)));
              Pi0(iq, M, N) = d;              // Pi^L = pi^dyn - Pi^{C,0}(tau = 0)
            }
        _diag_xl_rel = (p0 > 0.0 ? xl / p0 : 0.0);
        app_log(1, "  X^L diagnostic: max|pi^dyn - Pi^(C,0)(tau=0)| = {:.4e}, "
                   "relative to max|Pi^(C,0)(tau=0)| = {:.4e}  -> X^L/Pi^0 = {:.4f} "
                   "(vanishes iff the screening is truly static; theory diagnostic O3)",
                xl, p0, (p0 > 0.0 ? xl / p0 : 0.0));
      }

      // (3) Sigma^{C,r} is a GLOBAL-aux object (its Gt and externals are full-space), so
      //     the secondary-basis Pi is upfolded, Pi_hat = t^dag Pibar t.
      nda::array<ComplexType, 3> Pi0g(nqpts_ibz, Np, Np), W0g(nqpts_ibz, Np, Np);
      if (sec) {
        nda::array<ComplexType, 2> tmp_Pn(Np, _Nm);
        for (long iq = 0; iq < nqpts_ibz; ++iq)
          vertex_secondary_detail::upfold_core(_t_qmP(iq, all, all), Pi0(iq, all, all),
                                               tmp_Pn, Pi0g(iq, all, all));
        // the global static screen: zero-pad + all_reduce GATHER of the (P,Q)-distributed
        // W0 (the gather_dW_replicated pattern; every element lives on exactly one rank,
        // so this is a pure gather with no reassociation).
        // NOTE: this replicates an (nq, Np, Np) object per rank, which is prohibitive at
        // large Np; a distributed (P,Q) sandwich would avoid it.
        auto const& dW0 = _W0_qPQ.value();
        W0g() = ComplexType(0.0);
        W0g(dW0.local_range(0), dW0.local_range(1), dW0.local_range(2)) = dW0.local();
        mpi->comm.all_reduce_in_place_n(W0g.data(), W0g.size(), std::plus<>{});
      } else {
        Pi0g = Pi0;
        W0g = W0b_r;         // global path: W0bar IS the global W0 (N_m == Np)
      }

      // (4) the response rung and the +-q Hadamard pair
      nda::array<ComplexType, 3> Dw(nqpts_ibz, Np, Np);
      // ---- ENFORCE THE q -> 0 HEAD SUPPRESSION OF THE RESPONSE MIDDLE FACTOR -----------
      // Delta w(q) = W0(q) Pi(q) W0(q) with W0 ~ v0 ~ 1/q^2 is finite as q -> 0 ONLY
      // because a CONSERVING polarization has a head that vanishes like q^2 (f-sum rule /
      // Ward). Without that suppression the Gamma microcell of the external q-sum behaves
      // like 1/q^4, whose cell integral DIVERGES in 3D -- unlike the 1/q^2 of an ordinary
      // rung sum, which is integrable.
      //
      // B-S's middle factor Pi^{C,0} has this suppression to round-off. B-L's Pi^L need
      // not: its offending component (<H,Pi>/||H||_F^2) H can be a sizeable fraction of
      // the object. It is carried by pi^dyn and is the same on BOTH pi^dyn routes
      // (vertex_pidyn = "factorized" and "kernel"), so it is not an artifact of the
      // factorization.
      //
      // WHAT THIS DOES AND DOES NOT CLAIM. Removing the chi chi^dag component enforces a
      // property the exact object HAS and that the sandwich REQUIRES; it is the same class
      // of exact-symmetry projection as build_delta_w's +-q symmetrization and eval_Pi_C's
      // pair-symmetry projection. It is NOT a repair of whatever upstream defect lets
      // pi^dyn acquire the component in the first place, and the amount removed is
      // logged every call so it stays visible rather than silent.
      // B-S is essentially unaffected (its component is at round-off level).
      if (head_ok and _bl_head_projection) {
        ComplexType hp(0.0, 0.0);
        double hn = 0.0, pmax = 0.0;
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            hp += std::conj(H_PQ(P, Q)) * Pi0g(iq_gamma, P, Q);
            hn += std::norm(H_PQ(P, Q));
            pmax = std::max(pmax, std::abs(Pi0g(iq_gamma, P, Q)));
          }
        if (hn > 0.0) {
          const ComplexType c = hp / ComplexType(hn);
          double dmax = 0.0;
          for (long P = 0; P < Np; ++P)
            for (long Q = 0; Q < Np; ++Q) {
              const ComplexType d = c * H_PQ(P, Q);
              dmax = std::max(dmax, std::abs(d));
              Pi0g(iq_gamma, P, Q) -= d;
            }
          _diag_head_removed = (pmax > 0.0 ? dmax / pmax : 0.0);
          app_log(1, "  [WARNING] {} head-channel projection at q = Gamma is ENABLED: "
                     "removed |<H,Pi>|/||H||^2 = {:.4e},\n"
                     "            max|removed| = {:.4e} vs max|Pi(Gamma)| = {:.4e} "
                     "({:.2f} %).\n"
                     "            🚨 THIS BREAKS PHI-DERIVABILITY. The B-L G-side oracle "
                     "shows that deleting only\n"
                     "            20 % of this channel takes the eq:eulerBL1 residual from "
                     "3.3e-11 to 1.6e-01 --\n"
                     "            worse than the untransposed-sandwich control the same "
                     "test exists to reject.\n"
                     "            It is also applied to the SIGMA CUT ONLY (eval_Pi_C's "
                     "P^{{C,L}} keeps its head),\n"
                     "            so Sigma and P are no longer two cuts of one Phi. "
                     "DIAGNOSTIC USE ONLY -- the\n"
                     "            resulting energies are NOT conserving. See "
                     "notes/bl_head_channel_diagnosis.md.",
                  (lin ? "Sigma^(L,r)" : "Sigma^(C,r)"), std::abs(c), dmax, pmax,
                  (pmax > 0.0 ? 100.0 * dmax / pmax : 0.0));
        }
      }

      {
        // On an IBZ mesh build_delta_w replaces Pi(-q) by Pi(q)^T. CHECK it on the rows that
        // allow it (self-inverse q; stored +-q pairs) and abort above a relative 1e-8 unless
        // vertex_allow_unchecked_reflection. Pi0g is replicated, so every rank measures the same number.
        vertex_detail::reflection_check refl;
        vertex_detail::build_delta_w(W0g, Pi0g, qmin, Dw, /*assume_reflection*/ sym_mesh, sym_mesh ? &refl : nullptr);
        if (sym_mesh) {
          constexpr double refl_tol = 1e-8;
          app_log(1, "  [audit D10] {} reflection identity Pi(-q) = Pi(q)^T on the IBZ mesh: checked on {} of {} transfers "
                     "(self-inverse or stored +-q pair), max relative violation = {:.3e} (q = {}); {} transfer(s) without a "
                     "stored -q are not checkable.", (lin ? "Delta_w^L" : "Delta_w"), refl.n_checked, nqpts_ibz,
                  refl.rel_max, refl.iq_worst, refl.n_unchecked);
          utils::check(refl.rel_max <= refl_tol or _allow_unchecked_reflection,
                       "vertex_t::eval_Sigma_C: the response rung on this IBZ mesh assumes Pi(-q) = Pi(q)^T (build_delta_w, "
                       "assume_reflection), but the {} middle factor violates it by a relative {:.3e} > {:.0e} at q = {} "
                       "(audit D10): the symmetrized middle factor would be replaced by a plain transpose that is not equal to "
                       "it. Run on a symmetry-free (nosym) mesh, or set vertex_allow_unchecked_reflection = true to continue.",
                       (lin ? "Pi^L (pi^dyn - Pi^(C,0))" : "Pi^(C,0)"), refl.rel_max, refl_tol, refl.iq_worst);
          if (refl.rel_max > refl_tol)
            app_log(1, "  [WARNING] reflection identity violated by a relative {:.3e} > {:.0e} at q = {}; continuing "
                       "because vertex_allow_unchecked_reflection = true (the transpose is used as is).",
                    refl.rel_max, refl_tol, refl.iq_worst);
        }
      }

      // R-DECAY DIAGNOSTIC, q-side (read-only; env COQUI_VERTEX_RDECAY=1): Delta_w
      // on the full transfer mesh. Full-q only (nosym runs) -- an IBZ-stored aux-frame
      // q-object does not star-unfold elementwise (the collocation rotation intervenes).
      if (vertex_debug::flag("vertex_rdecay") and Dw.shape(0) == nqpts and mb_state.mpi->comm.root())   // vertex_debug: vertex_rdecay
        vertex_rdecay_detail::log_rshell_decay_q(MF, Dw,
                                                 lin ? "Delta_w^L" : "Delta_w");

      // ---- DIAGNOSTIC: B-L vs B-S response sandwich, per q ----------------------------
      // Sigma^r is exactly linear in the middle factor and B-S/B-L share this code, so
      // max|Dw^L(q)| / max|Dw^S(q)| must track X^L/Pi^0 at EVERY q. If it does
      // not, the excess is produced by the SANDWICH -- i.e. by how the rank-1 Gamma head
      // of W0 projects the two middle factors -- and not by Pi^L itself. Also reports the
      // head projection chi^dag Pi chi directly, which is the scalar the head amplifies.
      if (lin and not sec and PiStat.shape(0) == nqpts_ibz) {
        nda::array<ComplexType, 3> DwS(nqpts_ibz, Np, Np);
        vertex_detail::build_delta_w(W0g, PiStat, qmin, DwS, sym_mesh);
        for (long iq = 0; iq < nqpts_ibz; ++iq) {
          double dl = 0.0, ds = 0.0, pl = 0.0, ps = 0.0;
          for (long P = 0; P < Np; ++P)
            for (long Q = 0; Q < Np; ++Q) {
              dl = std::max(dl, std::abs(Dw(iq, P, Q)));
              ds = std::max(ds, std::abs(DwS(iq, P, Q)));
              pl = std::max(pl, std::abs(Pi0g(iq, P, Q)));
              ps = std::max(ps, std::abs(PiStat(iq, P, Q)));
            }
          app_log(1, "  [SANDWICH] q = {}: max|Pi^L| / max|Pi^0| = {:.4f}   ||   "
                     "max|Dw^L| = {:.4e}, max|Dw^S| = {:.4e}, RATIO = {:.4f}",
                  iq, (ps > 0.0 ? pl / ps : 0.0), dl, ds, (ds > 0.0 ? dl / ds : 0.0));
        }
        // The scalar the rank-1 head actually picks out: the overlap of Pi with the head
        // direction, <H, Pi> = sum_PQ conj(H_PQ) Pi_PQ. A CONSERVING polarization must
        // have a vanishing head as q -> 0 (f-sum rule / Ward), so <H, Pi(Gamma)> should be
        // strongly suppressed. If Pi^0 is suppressed and pi^dyn is not, their difference
        // inherits pi^dyn's violation and the W0(Gamma) head sandwich amplifies it.
        if (head_ok) {
          ComplexType hl(0.0, 0.0), hs(0.0, 0.0);
          double hn = 0.0;
          for (long P = 0; P < Np; ++P)
            for (long Q = 0; Q < Np; ++Q) {
              const ComplexType h = std::conj(H_PQ(P, Q));
              hl += h * Pi0g(iq_gamma, P, Q);
              hs += h * PiStat(iq_gamma, P, Q);
              hn += std::norm(H_PQ(P, Q));
            }
          _diag_head_hl = std::abs(hl);
          _diag_head_hs = std::abs(hs);
          app_log(1, "  [HEADPROJ] q = Gamma: |<H, Pi^L>| = {:.6e}, |<H, Pi^0>| = {:.6e}, "
                     "RATIO = {:.4e};  |H|_F^2 = {:.4e} (a conserving Pi must SUPPRESS this)",
                  std::abs(hl), std::abs(hs),
                  (std::abs(hs) > 0.0 ? std::abs(hl) / std::abs(hs) : 0.0), hn);
        }
      }
      Sigma_r = nda::array<ComplexType, 5>(nt, ns, nk_ext, nbnd, nbnd);
      if (sym_mesh)
        vertex_detail::eval_sigma_C_response_sym(mpi->comm, thc, G_tskij, Dw, Sigma_r);
      else
        vertex_detail::eval_sigma_C_response(mpi->comm, G_tskij, X_skPa, Dw, kmq, qmin,
                                             Sigma_r);
      double rmax = 0.0, xmax = 0.0;
      for (auto const& v : Sigma_r) rmax = std::max(rmax, std::abs(v));
      for (auto const& v : Sigma_C) xmax = std::max(xmax, std::abs(v));
      utils::check(std::isfinite(rmax),
                   "vertex_t::eval_Sigma_C: Sigma^(C,r) contains NaN/Inf -- aborting.");
      _diag_resp_share = (xmax > 0.0 ? rmax / xmax : 0.0);
      app_log(1, "  Sigma^({}): max|.| = {:.4e}; response share "
                 "||Sigma^(C,r)||/||Sigma^(C,x)|| = {:.4f} (large => the deleted rung "
                 "dynamics likely matters; theory diagnostic O3)",
              (lin ? "L,r" : "C,r"), rmax, (xmax > 0.0 ? rmax / xmax : 0.0));
    }

    // R-DECAY DIAGNOSTIC (read-only; env COQUI_VERTEX_RDECAY=1): lattice-decay tables
    // of the coarse-grid interpolants. Sigma_C's externals are IBZ-resident; unfold to
    // the full BZ by the SAME star-copy / trev-transpose convention as G_CC above (image
    // points are gauge copies, identity D; symmetry.hpp) -- exactly the rule a
    // coarse-grid interpolation would use. G_CC is logged as the long-range CONTRAST
    // (Sigma/Pi are the interpolation targets, never G).
    if (vertex_debug::flag("vertex_rdecay") and mb_state.mpi->comm.root()) {   // vertex_debug: vertex_rdecay
      nda::array<ComplexType, 5> Sfull(nt, ns, nkpts, nc, nc);
      if (not sym_mesh) {
        Sfull = Sigma_C;
      } else {
        auto kp_to_ibz = MF->kp_to_ibz();
        auto kp_trev = MF->kp_trev();
        for (long kp = 0; kp < nkpts; ++kp) {
          const long kib = kp_to_ibz(kp);
          if (not kp_trev(kp)) {
            Sfull(all, all, kp, all, all) = Sigma_C(all, all, kib, all, all);
          } else {
            for (long it = 0; it < nt; ++it)
              for (long is = 0; is < ns; ++is)
                for (long a = 0; a < nc; ++a)
                  for (long b = 0; b < nc; ++b)
                    Sfull(it, is, kp, a, b) = Sigma_C(it, is, kib, b, a);
          }
        }
      }
      vertex_rdecay_detail::log_rshell_decay_k(
          MF, Sfull, lin  ? "Sigma^(C,L-explicit) C-block"
                     : stat ? "Sigma^(C,x) C-block"
                            : "Sigma^C C-block");
      vertex_rdecay_detail::log_rshell_decay_k(MF, G_CC, "G_CC contrast");
    }

    // accumulate on top of the GW self-energy: Sigma <- Sigma + Sigma^C
    // (Sigma_C is identical on every rank after the kernel's all_reduce; hermitization
    //  stays downstream in scf_driver).
    //   WINDOW MODE: Sigma_C is already in band labels on the C-C block -> drop in.
    //   WANNIER MODE: Sigma_C lives in Wannier labels; inject the
    //     operator sandwich Sigma^C_ij = [U Sigma_bar U^dag]_ij over i,j in W_rng
    //     (projector_t::upfold primitive). External k axis is IBZ-resident; the IBZ
    //     k-points are [0, nk_ext), so _U_skia(is, ik_ext) is the right U.
    const double lam_s = vertex_scale();
    if (lam_s != 1.0) {
      Sigma_C *= ComplexType(lam_s);
      app_log(1, "  [ISDF-Vertex] Sigma^C scaled by lambda = {:.4f} (Phi_2^C -> lambda "
                 "Phi_2^C; BOTH cuts carry the same lambda, so conservation is exact).",
              lam_s);
    }
    if (stat and lam_s != 1.0) Sigma_r *= ComplexType(lam_s);
    if (mb_state.mpi->node_comm.root()) {
      // Sigma^{C,r} is FULL-SPACE (its lines live in W0's unprojected RPA bubble), so it
      // is added over the whole band range -- unlike Sigma^{C,x}, which is strictly C-C.
      if (_bl_drop != 0)
        app_log(1, "  [DROP] vertex_bl_drop = {} -- {} is being OMITTED from the "
                   "accumulated Sigma.\n"
                   "         DIAGNOSTIC ONLY: one cut without the other is NOT "
                   "Phi-derivable and these\n"
                   "         energies do not conserve. Used to read off each piece's exact "
                   "energy share.",
                _bl_drop, (_bl_drop == 1 ? "Sigma^(L,r), the response term"
                           : _bl_drop == 2 ? "Sigma^(C,x), the kernel term"
                                           : "BOTH Sigma^C pieces (P^{C,L} still injected, "
                                             "so this isolates its effect via Sigma_GW)"));
      if (stat and _bl_drop != 1 and _bl_drop != 3) sSigma_tskij.local() += Sigma_r;
      if (_bl_drop == 2 or _bl_drop == 3) {
        // kernel term dropped -- nothing to accumulate on the C-C block
      } else if (not wan) {
        sSigma_tskij.local()(all, all, all, _band_window, _band_window) += Sigma_C;
      } else {
        const long nW = _band_window.size();
        nda::array<ComplexType, 2> tmp(nW, nc);
        auto S = sSigma_tskij.local();
        for (long it = 0; it < nt; ++it)
          for (long is = 0; is < ns; ++is)
            for (long ik = 0; ik < nk_ext; ++ik)
              vertex_wannier_detail::upfold_Sigma(
                  _U_skia(is, ik, all, all), Sigma_C(it, is, ik, all, all), tmp,
                  S(it, is, ik, _band_window, _band_window));
      }
    }
    _Timer.stop("SIG_RESPONSE");
    // Timed separately: this barrier absorbs the load imbalance of everything above, so
    // folding it into SIG_RESPONSE would misattribute other ranks' skew to the response
    // cut. A large SIG_BARRIER is the signal that the (s,k,qx) x qy split is uneven.
    _Timer.start("SIG_BARRIER");
    mb_state.mpi->comm.barrier();
    _Timer.stop("SIG_BARRIER");
  }

  auto vertex_t::eval_Pi_C(MBState &mb_state, THC_ERI auto const &thc,
                           shape_t<4> pi_pgrid, shape_t<4> pi_bsize, shape_t<4> pi_gshape)
  -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>
  {
    vertex_timer_detail::scoped_timer _tm_total(_Timer, "PI_C");
    _Timer.start("PI_SETUP");
    decltype(nda::range::all) all;
    utils::check(active(), "vertex_t::eval_Pi_C: called while the vertex is inactive. "
                           "Callers must guard vertex calls with vertex_t::active().");
    // B-S has NO Pi^C injection at all (scr_coulomb_t never calls this in that mode);
    // B-L's P^{C,L} is the static-rung Z-phase.
    check_rung_implemented("vertex_t::eval_Pi_C");
    // THE FORBIDDEN HYBRID: B-S's W-cut vanishes identically, so P = P_RPA. Injecting a
    // static-rung Pi^C while using B-S's Sigma would pair the G-cut of Phi_2^{C,0} with
    // the W-cut of a DIFFERENT functional and break conservation exactly as "Sigma^C
    // without P^C" would in the dynamic theory. The
    // update_w seam already returns before reaching here (scr_coulomb_t), so this is a
    // structural tripwire, not a user-facing path.
    utils::check(_rung != static_rung,
                 "vertex_t::eval_Pi_C: reached with vertex_rung = \"static\". B-S has NO "
                 "polarization injection (P = P_RPA); combining it with Sigma^{C,x} is the "
                 "forbidden hybrid. This is an internal wiring bug -- the update_w seam "
                 "should have skipped the Pi^C hook.");
    // one scf iteration = one eval_Pi_C followed by one eval_Sigma_C, so counting here
    // and READING (not advancing) in eval_Sigma_C gives both cuts the same lambda.
    ++_vertex_iter;
    utils::check(mb_state.sG_tskij.has_value(),
                 "vertex_t::eval_Pi_C: sG_tskij is not initialized in MBState.");

    auto mpi = thc.mpi();
    auto MF = thc.MF();
    long nkpts = MF->nkpts();
    long nqpts = MF->nqpts();
    long nqpts_ibz = MF->nqpts_ibz();
    long nkpts_ibz = MF->nkpts_ibz();
    long Np = thc.Np();
    long nbnd = MF->nbnd();

    // IBZ SYMMETRY: the external q axis of Pi^C is IBZ-resident (as the output grid
    // already is); the internal (k, qx) sums run over the full BZ, sourcing the rung
    // from IBZ-stored W/Z via the symmetry context. Symmetry-free meshes take the plain
    // full-BZ path.
    bool sym_mesh = (nqpts != nqpts_ibz) or (nkpts != nkpts_ibz);
    {
      auto kp_trev = MF->kp_trev();
      for (long ik = 0; ik < nkpts; ++ik)
        if (kp_trev(ik)) { sym_mesh = true; break; }
    }

    auto G_tskij = mb_state.sG_tskij.value().local();
    long nt_f = G_tskij.shape(0);
    long ns = G_tskij.shape(1);
    long nt_half = (nt_f % 2 == 0) ? nt_f / 2 : nt_f / 2 + 1;
    utils::check(pi_gshape[0] == nt_half and pi_gshape[1] == nqpts_ibz and
                 pi_gshape[2] == Np and pi_gshape[3] == Np,
                 "vertex_t::eval_Pi_C: unexpected Pi grid shape ({}, {}, {}, {}); "
                 "expected ({}, {}, {}, {}).",
                 pi_gshape[0], pi_gshape[1], pi_gshape[2], pi_gshape[3],
                 nt_half, nqpts_ibz, Np, Np);

    app_log(1, "\n  ISDF-Vertex: evaluating Pi^C (G^4 W, single rung)\n"
               "  -------------------------------------------------\n"
               "  Subspace C band window = [{}, {})  ({} orbitals)\n"
               "  Grid (nt_half, nq, Np, Np) = ({}, {}, {}, {})\n",
            _band_window.first(), _band_window.last(), _band_window.size(),
            nt_half, nqpts_ibz, Np, Np);
    if (_ft->basis() != imag_axes_ft::dlr_basis)
      app_log(1, "  [NOTE] Pi^C requires off-grid Matsubara interpolation from the imaginary-\n"
                 "         axis backend (IAFT::construct_w_interpolate_matrix). The DLR backend\n"
                 "         provides it; the IR driver does not implement it yet and will abort\n"
                 "         inside the backend if this run proceeds.\n");

    vertex_pi::iaft_tools tools(*_ft);

    // ---- q = Gamma index (crystal coordinates: all components integer mod G) ----------
    long iq_gamma = -1;
    {
      auto Qpts = MF->Qpts();
      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        double d = 0.0;
        for (long i = 0; i < 3; ++i) {
          double x = Qpts(iq, i);
          d += std::abs(x - std::round(x));
        }
        if (d < 1e-8) {
          utils::check(iq_gamma < 0,
                       "vertex_t::eval_Pi_C: multiple Gamma q-points found ({} and {}).",
                       iq_gamma, iq);
          iq_gamma = iq;
        }
      }
      utils::check(iq_gamma >= 0, "vertex_t::eval_Pi_C: no Gamma q-point found.");
    }

    // ---- q->0 rung policy -------------------------------------------------------------
    const bool skip_rung_gamma = (_div_treatment == "v1_skip");
    bool head_insertion = (_div_treatment.find("gygi") != std::string::npos);
    if (head_insertion and nqpts_ibz == 1) {
      app_log(1, "  [WARNING] Pi^C: nqpts_ibz == 1 while vertex div_treatment is "
                 "gygi-class; the q->0\n"
                 "            extrapolation is meaningless on a Gamma-only mesh -- taking "
                 "\"ignore_g0\" instead\n"
                 "            (same downgrade as GW's Sigma_div_correction).");
      head_insertion = false;
    }
    if (skip_rung_gamma)
      app_log(1, "  [NOTE] Pi^C q->0 policy: v1_skip -- the qx = Gamma (iq = {}) cell of "
                 "the internal rung\n"
                 "         transfer is DROPPED (bare Z and dynamic dW). Fallback mode; "
                 "O(1/N_k) finite-size error.\n", iq_gamma);
    else
      app_log(1, "  [NOTE] Pi^C q->0 policy: {} -- the qx = Gamma (iq = {}) cell of the "
                 "internal rung transfer\n"
                 "         is INCLUDED with the stored regularized W(Gamma) (v(G=0) zeroed "
                 "at ERI build){}.\n"
                 "         The external q axis is regular and computed at all q.\n",
              _div_treatment, iq_gamma,
              head_insertion ? ",\n         PLUS the analytic rank-1 head insertion "
                               "Nk*madelung*[1 | Re eps_inv_head(tau)]*chi chi^dag"
                             : "; no analytic head (GW ignore_g0 analogue)");

    // ---- collocation matrices (q-independent X, polarization 0) -----------------------
    // node-share X_skPa (one copy per node; see eval_Sigma_C).
    auto sX_skPa = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(
        *mpi, std::array<long, 4>{ns, nkpts, Np, nbnd});
    sX_skPa.win().fence();
    if (mpi->node_comm.root())
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nkpts; ++ik)
          sX_skPa.local()(is, ik, all, all) = thc.X(is, 0, ik);
    sX_skPa.win().fence();
    auto X_skPa = sX_skPa.local();

    // ---- bare coulomb Z(q) (thc.Z is collective: call uniformly on all ranks) ---------
    // GLOBAL path: build the full replicated (nq, Np, Np) Z_qPQ (the kernel + tripwire read
    // it). SECONDARY path: do NOT materialize the replicated all-q Z (prohibitive at large
    // Np) -- it is folded DISTRIBUTED from thc.dZ below. Z_qPQ stays a
    // default-empty (0-sized) array; the only secondary-path reads of the full Z are the
    // TEST-SCALE gated diagnostics (eta[Z], no-leak tripwire), which pull the small
    // replicated thc.Z(iq) locally inside their own gated branches.
    const bool sec_z = secondary();
    nda::array<ComplexType, 3> Z_qPQ;
    if (not sec_z) {
      Z_qPQ = nda::array<ComplexType, 3>(nqpts_ibz, Np, Np);
      for (long iq = 0; iq < nqpts_ibz; ++iq)
        Z_qPQ(iq, all, all) = thc.Z(int(iq));
    }

    // head insertion, bare piece (weight 1) into Z(Gamma). The head is
    // EXACTLY rank-1 (build_head_rank1): H_PQ = N_k * madelung * conj(chi_g) chi_g^T with
    // chi_g = thc.basis_head()(iq_gamma, :). The GLOBAL path materializes the dense (Np x Np)
    // H_PQ and adds it into Z_qPQ(Gamma) here. The SECONDARY path NEVER materializes
    // the dense Np^2 head: it captures the Np-vector chi_g and
    // the scalar c = N_k * madelung, and rebuilds each (P,Q) block on the fly from those
    // (vertex_secondary_detail::head_block_add) inside the distributed W/Z fold closures and
    // the test-scale diagnostics -- bit-identical per element to the dense-H slice.
    nda::array<ComplexType, 2> H_PQ;                  // GLOBAL path only (0-sized in secondary)
    nda::array<ComplexType, 1> chi_g;                 // SECONDARY path: chi(iq_gamma, :)
    ComplexType head_c = ComplexType(0.0);            // SECONDARY path: c = N_k * madelung
    bool head_ok = false;
    if (head_insertion) {
      if (not sec_z) {
        // GLOBAL path: dense rank-1 head.
        H_PQ = nda::array<ComplexType, 2>(Np, Np);
        head_ok = vertex_head_detail::build_head_rank1(thc, iq_gamma, nkpts, H_PQ,
                                                                    _bl_head_scale);
      } else {
        // SECONDARY path: replicate build_head_rank1's skip logic EXACTLY (madelung == 0 or an
        // all-zero chi(iq_gamma, :) => head_ok = false, no head) WITHOUT the dense Np^2 matrix.
        // "EXACTLY" includes the head-strength factor lambda: this site must carry the same
        // factor as build_head_rank1 or the global and secondary paths would scale the head
        // differently, and the secondary path would silently ignore the knob.
        const double xi = MF->madelung() * _bl_head_scale;
        auto chi = thc.basis_head();                 // (nqpts_ibz, Np)
        utils::check(chi.shape(0) > iq_gamma and chi.shape(1) == Np,
                     "vertex_t::eval_Pi_C: basis_head shape mismatch (({}, {}) vs iq_gamma = "
                     "{}, Np = {}).", chi.shape(0), chi.shape(1), iq_gamma, Np);
        double chi_max = 0.0;
        for (long P = 0; P < Np; ++P) chi_max = std::max(chi_max, std::abs(chi(iq_gamma, P)));
        if (xi != 0.0 and chi_max != 0.0) {
          chi_g = nda::array<ComplexType, 1>(chi(iq_gamma, all));   // Np-vector copy (~KB)
          head_c = ComplexType(double(nkpts) * xi);
          head_ok = true;
        }
      }
      if (head_ok) {
        if (not sec_z) Z_qPQ(iq_gamma, all, all) += H_PQ;
        double h_max = 0.0;
        if (not sec_z) {
          for (auto const& v : H_PQ) h_max = std::max(h_max, std::abs(v));
        } else {
          // |H_PQ|_max = |c| * (max_P |chi_g(P)|)^2 (rank-1, no dense matrix needed).
          double cg_max = 0.0;
          for (long P = 0; P < Np; ++P) cg_max = std::max(cg_max, std::abs(chi_g(P)));
          h_max = std::abs(head_c) * cg_max * cg_max;
        }
        app_log(1, "  Pi^C head insertion: madelung = {}, |H|_max = {} (bare piece "
                   "applied to Z(Gamma))", MF->madelung(), h_max);
      } else if (head_unusable_continue("vertex_t::eval_Pi_C")) {   // aborts unless allowed / explicit
        app_log(1, "  [WARNING] Pi^C: gygi head insertion requested but head data are "
                   "unusable\n"
                   "            (madelung == 0 or empty basis_head) -- proceeding WITHOUT "
                   "the analytic head\n"
                   "            (equivalent to policy \"ignore_g0\").");
      }
    }

    // ---- dynamic W on the full bosonic Matsubara mesh ---------------------------------
    // mb_state.dW_qtPQ is the dynamic-only screened interaction (bare Z subtracted,
    // scr_coulomb_t.cpp) on (nq, nt_half, Np, Np). Source selection:
    //   - SECONDARY path with a FILLED W-bar cache: the previous iteration's rung was
    //     already downfolded at update_w time (cache_w) -- the global-basis Wdyn is
    //     NOT rebuilt here (the scf driver frees dW unconditionally in this mode:
    //     plain-GW memory profile). Same one-iteration lag as the retained-dW path.
    //   - otherwise (global path; or secondary with a retained dW): fold at
    //     consumption from mb_state.dW_qtPQ.
    //   - neither present: FIRST ITERATION -- the rung reduces to the bare
    //     interaction Z.
    const bool sec = secondary();
    const bool use_wcache = sec and has_cached_w();
    // A dynamic-W source is present iff the W-bar cache is NOT consumed and mb_state
    // carries dW. In the GLOBAL path the full replicated Wdyn_qwPQ (Np^2) is built below;
    // in the SECONDARY path the tau -> nu transform is DEFERRED into the per-q
    // distributed fold, so the replicated (nq, nw_b, Np, Np) array is never materialized.
    // B-L's rung is the STATIC screen W0bar, so no dynamic rung is folded
    // here at all (the mixed Sigma terms consume dW separately, in eval_Sigma_C).
    const bool dyn_src = (_rung == dynamic_rung) and (not use_wcache)
                         and mb_state.dW_qtPQ.has_value();
    std::optional<nda::array<ComplexType, 4>> W_qtPQ;  // tau-domain all-q, GLOBAL path only
    std::optional<nda::array<ComplexType, 4>> Wdyn_qwPQ;  // nu-domain, GLOBAL path only
    if (use_wcache and mb_state.dW_qtPQ.has_value())
      app_log(2, "  [NOTE] Pi^C: both the W-bar cache and mb_state.dW_qtPQ are present "
                 "-- consuming the CACHE\n"
                 "         (identical content when both were produced by the same "
                 "update_w).");
    // head insertion, dynamic piece, into dW(Gamma, tau) for one q's tau slab (weight
    // Re[eps_inv_head(tau)]; same Re[.] convention as Sigma_div_correction, thc_gw.icc):
    // add eps(it).real()*H_PQ into rows [0,nt_half) IN PLACE. Applied to the all-q slab at
    // iq_gamma (GLOBAL) or, in the SECONDARY path, to the per-q gathered slab / distributed
    // block when iq == iq_gamma --
    // identical arithmetic either way. head_dyn_ok says whether the dynamic piece is added.
    const bool head_dyn_ok = head_ok and dyn_src and mb_state.eps_inv_head.has_value();
    if (dyn_src and head_ok) {
      if (mb_state.eps_inv_head.has_value())
        utils::check(mb_state.eps_inv_head.value().shape(0) == nt_half,
                     "vertex_t::eval_Pi_C: eps_inv_head size {} != nt_half = {}.",
                     mb_state.eps_inv_head.value().shape(0), nt_half);
      else {
        dyn_head_missing("vertex_t::eval_Pi_C");   // aborts unless vertex_allow_missing_head
        app_log(1, "  [WARNING] Pi^C: dW is present but eps_inv_head is not in MBState "
                   "-- the DYNAMIC head\n"
                   "            piece is skipped (bare piece applied).");
      }
    }
    // adds the dynamic head into the (nt_half, Np, Np) tau slab of q = iq_gamma, in place.
    auto add_head_tau = [&](nda::MemoryArrayOfRank<3> auto&& W_t_gamma) {
      auto& eps = mb_state.eps_inv_head.value();
      for (long it = 0; it < nt_half; ++it)
        W_t_gamma(it, all, all) += ComplexType(eps(it).real()) * H_PQ;
    };
    if (head_dyn_ok)
      app_log(1, "  Pi^C head insertion: dynamic piece applied to dW(Gamma, tau) "
                 "with eps_inv_head(tau=0) = {}", mb_state.eps_inv_head.value()(0).real());
    if (dyn_src and not sec) {
      // GLOBAL path: gather the all-q tau slab (bit-identical), head-augment iq_gamma,
      // then materialize the full nu-domain Wdyn_qwPQ. The SECONDARY path instead folds
      // distributed blocks below, so neither the all-q tau slab nor the replicated Np^2
      // nu-array is built there.
      W_qtPQ.emplace(vertex_redist_detail::gather_dW_replicated(
          mb_state.dW_qtPQ.value(), mpi->comm, nqpts_ibz, nt_half, Np));
      if (head_dyn_ok)
        add_head_tau(W_qtPQ.value()(iq_gamma, nda::ellipsis{}));
      long nw_b = tools.nw_b;
      long nw_half = (nw_b % 2 == 0) ? nw_b / 2 : nw_b / 2 + 1;
      Wdyn_qwPQ.emplace(nda::array<ComplexType, 4>(nqpts_ibz, nw_b, Np, Np));
      nda::array<ComplexType, 3> W_wpos(nw_half, Np, Np);
      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        auto W_t = W_qtPQ.value()(iq, nda::ellipsis{});
        _ft->tau_to_w_PHsym(W_t, W_wpos);
        // unfold to the full mesh assuming W(-nu) = W(nu) (PH-symmetric storage, same
        // assumption as the SOSEX cache folding, thc_sosex.icc)
        for (long l = 0; l < nw_b; ++l) {
          long lpos = std::max(l, tools.w_mirror_b(l)) - nw_b / 2;
          Wdyn_qwPQ.value()(iq, l, all, all) = W_wpos(lpos, all, all);
        }
      }
    } else if (_rung == dynamic_rung and not dyn_src and not use_wcache) {
      // NO SCREENED RUNG AVAILABLE. The bare-rung fallback is not a harmless startup
      // detail: it is a large perturbation the scf trajectory need not recover from.
      // scr_coulomb_t::update_w bootstraps an RPA W before the first vertex-attached pass,
      // so reaching this branch means the bootstrap did NOT run -- e.g. a caller invoking
      // eval_Pi_C outside update_w. COUNTED so a test can assert it never happens in a
      // normal scf loop.
      // ABORTS unless pol_vertex_allow_bare_rung (the class default keeps the WARNING for the direct-call unit tests,
      // e.g. test_vertex_wcache's first-iteration semantics; every MBPT driver sets it from the input, default false).
      // Only the DYNAMIC rung reaches this branch: B-L (linear) never consumes a dynamic rung here -- its P^{C,L} rung
      // is W0bar (below).
      utils::check(_allow_bare_rung,
                   "vertex_t::eval_Pi_C: no dynamic W in MBState{} -- the dynamic-rung Pi^C would fall back to the BARE "
                   "rung W = Z, a large uncontrolled perturbation (audit D4). The update_w RPA bootstrap was bypassed (a caller "
                   "outside the scf loop?). Set pol_vertex_allow_bare_rung = true only to reproduce the historic behaviour.",
                   sec ? " and no cached Wbar" : "");
      ++_bare_rung_uses;
      app_log(1, "  [WARNING] Pi^C: no dynamic W in MBState{} -- falling back to the "
                 "BARE-interaction rung W = Z.\n"
                 "            Pi^C is a functional of the SCREENED W; this fallback is a "
                 "large, uncontrolled\n"
                 "            perturbation, not a small startup detail. Expected only if "
                 "the update_w\n"
                 "            RPA bootstrap was bypassed.\n",
              sec ? " and no cached Wbar" : "");
    }

    // ---- momentum maps on the FULL transfer mesh --------------------------------------
    // (rows beyond nqpts_ibz feed the internal qx sums under symmetry; on
    //  symmetry-free meshes nqpts == nqpts_ibz)
    nda::array<long, 2> kmq(nqpts, nkpts), kpq(nqpts, nkpts);
    for (long iq = 0; iq < nqpts; ++iq) {
      for (long ik = 0; ik < nkpts; ++ik) kmq(iq, ik) = MF->qk_to_k2(iq, ik);
      for (long ik = 0; ik < nkpts; ++ik) kpq(iq, kmq(iq, ik)) = ik;   // inverse: (k-q)+q = k
    }

    _Timer.stop("PI_SETUP");
    _Timer.start("PI_SECONDARY");
    // ---- optional secondary-basis substitution -----------------------------------------
    // The SAME kernel runs on the input set
    // (Xb, Zbar = t Z t^dag, Wbar_dyn = t dW t^dag, G_CC, window [0, nc)); the
    // head-augmented Gamma cells above downfold automatically through t. Pibar^C is
    // produced in (N_m x N_m) and UPFOLDED with the adjoint of the same t -- the no-leak
    // identity <Pibar, Wbar> = <Pi, W> is checked below as a transposition tripwire.
    // (`sec`/`use_wcache` are resolved above, at the dynamic-W source selection.)
    const bool wan = _wannier;
    const long nc = subspace_rank();
    nda::array<ComplexType, 3> Zb_qmm;
    std::optional<nda::array<ComplexType, 4>> Wbdyn_qwmm;
    // STRICT C-C EXTERNALS: dPhi_2^C/dW vanishes unless ALL FOUR pair orbital slots are
    // in range(P), so the external legs of Pi^C are C-restricted in BOTH paths via the
    // input projection -- exactly the kernel on G~ = P G P.
    // WINDOW: G_CC = the W-window block; WANNIER: G_CC = U^dag G U.
    // On the FULL BZ: image points are gauge copies of the IBZ
    // blocks; trev points are the tau-pointwise transpose (see eval_Sigma_C).
    utils::check(G_tskij.shape(2) == (sym_mesh ? nkpts_ibz : nkpts),
                 "vertex_t::eval_Pi_C: G_tskij k axis = {} != {}.",
                 G_tskij.shape(2), sym_mesh ? nkpts_ibz : nkpts);
    // node-share G_CC (one copy per node; see eval_Sigma_C for the rationale -- built
    // from node-shared sG_tskij, read-only in the templated-param kernel, so the
    // shared_array view binds directly).
    auto sG_CC = math::shm::make_shared_array<nda::array_view<ComplexType, 5>>(
        *mpi, std::array<long, 5>{nt_f, ns, nkpts, nc, nc});
    sG_CC.win().fence();
    if (mpi->node_comm.root()) {
      auto G_CC = sG_CC.local();
      if (wan) {
        vertex_wannier_detail::build_Gbar_fullbz(G_tskij, _U_skia, _band_window, sym_mesh,
                                                 MF->kp_to_ibz(), MF->kp_trev(), G_CC);
      } else if (not sym_mesh) {
        G_CC = G_tskij(all, all, all, _band_window, _band_window);
      } else {
        auto kp_to_ibz = MF->kp_to_ibz();
        auto kp_trev = MF->kp_trev();
        for (long kp = 0; kp < nkpts; ++kp) {
          const long kib = kp_to_ibz(kp);
          if (not kp_trev(kp)) {
            G_CC(all, all, kp, all, all) = G_tskij(all, all, kib, _band_window, _band_window);
          } else {
            for (long it = 0; it < nt_f; ++it)
              for (long is = 0; is < ns; ++is)
                for (long a = 0; a < nc; ++a)
                  for (long b = 0; b < nc; ++b)
                    G_CC(it, is, kp, a, b) =
                        G_tskij(it, is, kib, _band_window.first() + b, _band_window.first() + a);
          }
        }
      }
    }
    sG_CC.win().fence();
    auto G_CC = sG_CC.local();
    app_log(2, "  Pi^C external legs restricted to {} [0, {}) (strict Phi cut; "
               "notes/refinement2_optionA.md DECISION 2).",
            wan ? "range(P) (Wannier labels)" : "the C window", nc);
    // effective window collocation: WINDOW = X(:,C); WANNIER = X_bar = X.U (Np x M).
    // node-share X_C (one copy per node; see eval_Sigma_C).
    auto sX_C = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(
        *mpi, std::array<long, 4>{ns, nkpts, Np, nc});
    sX_C.win().fence();
    if (mpi->node_comm.root()) {
      auto X_C_loc = sX_C.local();
      if (wan)
        X_C_loc = vertex_wannier_detail::build_Xbar(X_skPa, _U_skia, _band_window);
      else
        X_C_loc = X_skPa(all, all, all, _band_window);
    }
    sX_C.win().fence();
    auto X_C = sX_C.local();
    auto X_glob = wan ? X_C : X_skPa;
    const long orb0_glob = wan ? 0 : _band_window.first();
    if (sec) {
      build_secondary_basis(thc, X_glob, orb0_glob, kmq, iq_gamma);
      app_log(1, "  Refinement 2: Pi^C runs in the SECONDARY basis (N_m = {} vs Np = {}); "
                 "upfold Pi^C = t^dag Pibar t.", _Nm, Np);
      // In the SECONDARY path neither the all-q tau slab W_qtPQ (nq*nt*Np^2) nor
      // the full nu-domain Wdyn_qwPQ (nq*nw_b*Np^2) is materialized. For each q we gather
      // ONE tau slab W_q (nt_half, Np, Np) from the distributed dW (bit-identical to the
      // all-q gather sliced at that q), head-augment it if iq == iq_gamma, then defer the
      // tau -> nu transform into the fold below (reusing ONE W_wpos per q). Bitwise-
      // equivalent to "gather all-q + augment iq_gamma + build full Wdyn_qwPQ then fold":
      // each (iq, l) folds exactly W_wpos(lpos) with the SAME PH-unfold map
      // lpos = max(l, w_mirror_b(l)) - nw_b/2 the global build applies.
      const long nw_b_sec = tools.nw_b;
      const long nw_half_sec = (nw_b_sec % 2 == 0) ? nw_b_sec / 2 : nw_b_sec / 2 + 1;
      // SECONDARY-path rank-1 tau head: adds the dynamic head into the (nt_half, Np,
      // Np) tau slab of q = iq_gamma from chi_g + head_c, WITHOUT the dense Np^2 H_PQ. Per
      // element head_block_add reproduces add_head_tau's `+= ComplexType(eps(it).real()) * H_PQ`
      // bit-for-bit (weight * (c*conj(chi_g)*chi_g) == weight * H_PQ(P,Q)).
      const nda::range head_all_P(0, Np), head_all_Q(0, Np);
      auto add_head_tau_sec = [&](nda::MemoryArrayOfRank<3> auto&& W_t_gamma) {
        auto& eps = mb_state.eps_inv_head.value();
        for (long it = 0; it < nt_half; ++it)
          vertex_secondary_detail::head_block_add(
              chi_g, head_c, ComplexType(eps(it).real()), head_all_P, head_all_Q,
              W_t_gamma(it, all, all));
      };
      // per-q tau slab source: gather q = iq from the distributed dW into the reused W_q_src
      // and head-augment at iq_gamma -- the exact tau slab the all-q path fed to tau_to_w.
      nda::array<ComplexType, 3> W_q_src(nt_half, Np, Np);
      long W_q_src_q = -1;
      auto gather_W_q = [&](long iq) -> nda::array_view<ComplexType, 3> {
        if (W_q_src_q != iq) {
          W_q_src = vertex_redist_detail::gather_dW_one_q(
              mb_state.dW_qtPQ.value(), mpi->comm, iq, nt_half, Np);
          if (head_dyn_ok and iq == iq_gamma) add_head_tau_sec(W_q_src());
          W_q_src_q = iq;
        }
        return W_q_src();
      };
      // eta W-accessor: returns the Np x Np nu-slice at bosonic index l. Global path reads
      // the prebuilt Wdyn_qwPQ; secondary path transforms the per-q gathered W_q(iq) on the
      // fly (test-scale gate only: N_pair <= 4096). Reuses one scratch W_wpos across calls.
      nda::array<ComplexType, 3> W_wpos_eta(nw_half_sec, Np, Np);
      long W_wpos_eta_q = -1;
      auto eta_W_slice = [&](long iq, long l) -> nda::array_view<ComplexType, 2> {
        const long lpos = std::max(l, tools.w_mirror_b(l)) - nw_b_sec / 2;
        if (dyn_src) {
          if (W_wpos_eta_q != iq) {
            _ft->tau_to_w_PHsym(gather_W_q(iq), W_wpos_eta);
            W_wpos_eta_q = iq;
          }
          return W_wpos_eta(lpos, all, all);
        }
        return Wdyn_qwPQ.value()(iq, l, all, all);  // (secondary path never hits this)
      };
      // TEST-SCALE ONLY diagnostics (eta[Z], no-leak tripwire) still need the full
      // replicated Z(iq). In the secondary path Z_qPQ is NOT materialized, so pull
      // the small replicated thc.Z(iq) locally, head-augmented at Gamma exactly as the global
      // build did (Z(iq_gamma) += H_PQ). thc.Z(iq) is collective -- these branches are gated
      // on a GLOBAL condition and loop over all iq uniformly, so every rank calls it in lockstep.
      auto z_local = [&](long iq) {
        nda::array<ComplexType, 2> Zq = thc.Z(int(iq));
        if (head_ok and iq == iq_gamma)
          vertex_secondary_detail::head_block_add(chi_g, head_c, ComplexType(1.0),
                                                  head_all_P, head_all_Q, Zq());
        return Zq;
      };
      if (ns * nkpts * nc * nc <= 4096) {
        vertex_secondary_detail::eta_max_over_q(
            "Z", X_glob, orb0_glob, nc, _Xb_skma, _t_qmP, kmq,
            [&](long iq) { return z_local(iq); });
        if (dyn_src) {
          vertex_secondary_detail::eta_max_over_q(
              "dW(nu_0)", X_glob, orb0_glob, nc, _Xb_skma, _t_qmP, kmq,
              [&](long iq) { return eta_W_slice(iq, tools.m0); });
          vertex_secondary_detail::eta_max_over_q(
              "dW(nu_max)", X_glob, orb0_glob, nc, _Xb_skma, _t_qmP, kmq,
              [&](long iq) { return eta_W_slice(iq, tools.nw_b - 1); });
        }
      } else {
        app_log(2, "  Refinement 2: eta diagnostic skipped (N_pair = {} > 4096).",
                ns * nkpts * nc * nc);
      }
      // --- bare core Zbar = t Z t^dag: DISTRIBUTED downfold ------------------------------
      // thc.dZ({1, nP, nQ}) gives Z distributed over (P,Q) with q NOT split;
      // fold_Z_distributed folds each rank's own (P,Q) block with fold_core_block and sums
      // the disjoint-block partials with one final comm all_reduce. No rank ever holds the
      // full Np^2. Simpler than fold_dW_distributed: Z has no t axis => no t-pool, no
      // tau->nu, no PH-unfold. With nP = nQ = 1 there is one (P,Q) block == the whole array
      // and this is BIT-IDENTICAL to the replicated fold_core (disjoint-block sum is exact).
      // The Gamma head is added into the gamma block via z_head_add (same
      // Z(iq_gamma) += H_PQ semantics, built per block from the rank-1 factors).
      Zb_qmm = nda::array<ComplexType, 3>(nqpts_ibz, _Nm, _Nm);
      {
        // {1, nP, nQ} grid: q unsplit, (P,Q) balanced over comm; nP*nQ == comm.size().
        const long np_ranks = mpi->comm.size();
        std::array<long, 3> z_pgrid = {1, 1, 1};
        z_pgrid[1] = utils::find_proc_grid_min_diff(np_ranks, Np, Np);
        z_pgrid[2] = np_ranks / z_pgrid[1];
        auto dZ = thc.dZ(z_pgrid);
        auto z_head_add = [&](nda::MemoryArrayOfRank<2> auto&& A_PQ_block,
                              nda::range const& P_rng, nda::range const& Q_rng) {
          // bare piece, weight 1 (matches the global Z += H); rank-1 block, no dense
          // H_PQ -- head_block_add reproduces H_PQ(P,Q) bit-for-bit per element.
          vertex_secondary_detail::head_block_add(chi_g, head_c, ComplexType(1.0),
                                                  P_rng, Q_rng, A_PQ_block);
        };
        vertex_secondary_detail::fold_Z_distributed(
            dZ, _t_qmP, nqpts_ibz, Np, _Nm, iq_gamma, head_ok, z_head_add, Zb_qmm, mpi->comm);
      }
      // --- dynamic rung Wbar = t Wdyn(q,nu) t^dag: DISTRIBUTED downfold ------------------
      // fold_dW_distributed assembles ONLY this rank's (P,Q) block over all t (a t-pool
      // all_reduce over the disjoint t-partition -- exact), applies the Gamma head +
      // tau->nu + PH-unfold on that block, folds it with fold_core_block, and sums the
      // (P,Q)-block partials with one final comm all_reduce. NO full Np^2 slab is ever
      // held (a per-q gather would hold a full (nt_half, Np, Np) tau slab). With
      // np_P = np_Q = 1 there is one (P,Q) block and this reduces to the replicated fold
      // BIT-IDENTICALLY (disjoint-block sum is exact). The forced (P,Q) split is exercised
      // by test_vertex_dfold.
      if (dyn_src) {
        Wbdyn_qwmm.emplace(nda::array<ComplexType, 4>(nqpts_ibz, tools.nw_b, _Nm, _Nm));
        auto head_add = [&](nda::MemoryArrayOfRank<2> auto&& W_bt_block, long it,
                            nda::range const& P_rng, nda::range const& Q_rng) {
          // dynamic piece, weight Re[eps_inv_head(tau)] (matches the global path's
          // += ComplexType(eps(it).real()) * H_PQ); rank-1 block, no dense H_PQ.
          auto& eps = mb_state.eps_inv_head.value();
          vertex_secondary_detail::head_block_add(chi_g, head_c,
                                                  ComplexType(eps(it).real()),
                                                  P_rng, Q_rng, W_bt_block);
        };
        auto xform = [&](nda::MemoryArrayOfRank<3> auto&& W_bt_block,
                         nda::MemoryArrayOfRank<3> auto&& W_bw_block) {
          _ft->tau_to_w_PHsym(W_bt_block, W_bw_block);
        };
        vertex_secondary_detail::fold_dW_distributed(
            mb_state.dW_qtPQ.value(), _t_qmP, nqpts_ibz, nt_half, Np, _Nm,
            tools.nw_b, nw_half_sec, tools.w_mirror_b, iq_gamma, head_dyn_ok,
            head_add, xform, Wbdyn_qwmm.value(), mpi->comm);
      }
      // W-bar iteration cache consumption: the dynamic rung was
      // folded at update_w time on the positive half mesh; reconstruct the kernel's
      // full bosonic mesh with the SAME mirror map the uncached path applies BEFORE
      // its fold (W(-nu) = W(nu); the mirror is a pure copy, so fold-then-mirror is
      // bitwise identical to mirror-then-fold). eta[dW] diagnostics for this rung
      // were logged at fill time (cache_w); eta[Z] above covers the bare core.
      if (use_wcache) {
        auto Wbh = wb_cache();
        const long nw_half = (tools.nw_b % 2 == 0) ? tools.nw_b / 2 : tools.nw_b / 2 + 1;
        utils::check(Wbh.shape(0) == nqpts_ibz and Wbh.shape(1) == nw_half and
                     Wbh.shape(2) == _Nm and Wbh.shape(3) == _Nm,
                     "vertex_t::eval_Pi_C: cached Wbar shape ({}, {}, {}, {}) != "
                     "(nq, nw_half, N_m, N_m) = ({}, {}, {}, {}).",
                     Wbh.shape(0), Wbh.shape(1), Wbh.shape(2), Wbh.shape(3),
                     nqpts_ibz, nw_half, _Nm, _Nm);
        app_log(1, "  Refinement 2: Pi^C dynamic rung from the CACHED Wbar (previous "
                   "iteration's W, downfolded\n"
                   "  at update_w time; the same one-iteration lag as the retained-dW "
                   "path it replaces).");
        Wbdyn_qwmm.emplace(nda::array<ComplexType, 4>(nqpts_ibz, tools.nw_b, _Nm, _Nm));
        for (long iq = 0; iq < nqpts_ibz; ++iq)
          for (long l = 0; l < tools.nw_b; ++l) {
            long lpos = std::max(l, tools.w_mirror_b(l)) - tools.nw_b / 2;
            Wbdyn_qwmm.value()(iq, l, all, all) = Wbh(iq, lpos, all, all);
          }
      }
    }

    _Timer.stop("PI_SECONDARY");
    _Timer.start("PI_SYMCTX");
    // ---- IBZ symmetry context (trivial/null on symmetry-free meshes) ------------------
    // WANNIER: thread U so d = U(Sk)^dag D U(k) (sym + Wannier compose).
    vertex_sym::sym_ctx const* symc = nullptr;
    if (sym_mesh) {
      if (sec) {
        build_sym_ctx(thc, _Xb_skma, _band_window.first(), _sym_secondary);
        symc = &_sym_secondary.value();
      } else {
        build_sym_ctx(thc, X_C, _band_window.first(), _sym_global,
                      wan ? &_U_skia : nullptr);
        symc = &_sym_global.value();
      }
      _g_rot_max = std::max(_g_rot_max,
                            vertex_ibz_detail::g_rotation_check(*symc, G_CC, MF->kp_trev()));
    }

    _Timer.stop("PI_SYMCTX");
    _Timer.start("PI_KERNEL");
    // ---- kernel: accumulate Pi^C(inu) over this rank's (s,k,qx) tuples ----------------
    // q->0 policy resolved above (skip_rung_gamma / head-augmented inputs). Both paths:
    // C-restricted externals; the ONLY difference is the auxiliary input set --
    // (X_C, W, Z, Np) global vs (Xb, Wbar, Zbar, N_m) secondary.
    const long naux = sec ? _Nm : Np;
    nda::array<double, 1> qx_diag(nqpts);
    // THE Pi^C SLAB ACCUMULATOR (global path). A per-rank full-shape partial
    // (nw_b, nq_ibz, Np, Np) would be the largest replicated array, for B-S, B-L AND the
    // dynamic theory alike -- eval_Pi_C is the same call for all three. Each rank's
    // partial is structurally zero outside its q_ext stride, so store ONLY the owned
    // rows. Ownership is +-q-ORBIT-closed so the pair-symmetry projection below stays
    // rank-local; the plan's qext-first preference divides the accumulator by
    // qext_size at unchanged per-rank work (see vertex_pi::pi_qext_plan). The
    // SECONDARY path stays full-shape (N_m^2 is small).
    std::optional<vertex_pi::pi_qext_plan> qplan;
    if (not sec)
      qplan.emplace(vertex_pi::make_pi_qext_plan(mpi->comm.rank(), mpi->comm.size(),
                                                 ns * nkpts * nqpts, nqpts_ibz,
                                                 MF->qminus()));
    const long n_qrows = qplan ? long(qplan->owned.size()) : nqpts_ibz;
    if (qplan)
      app_log(2, "  Pi^C slab accumulator: qext_size = {} x tup_size = {}; this rank "
                 "stores {} of {} q rows ({:.2f} GB instead of {:.2f} GB)",
              qplan->qext_size, qplan->tup_size, n_qrows, nqpts_ibz,
              double(tools.nw_b * n_qrows * naux * naux) * 16.0 / 1.0e9,
              double(tools.nw_b * nqpts_ibz * naux * naux) * 16.0 / 1.0e9);
    nda::array<ComplexType, 4> Pi_wqMN(tools.nw_b, n_qrows, naux, naux);
    Pi_wqMN() = ComplexType(0.0);
    nda::array<double, 1> phase_diag(4);
    phase_diag() = 0.0;
    // B-L's W-cut. The tangent functional's dynamical W appears LINEARLY,
    // so its W-derivative kills BOTH the momentum and the frequency sum of the cut rung:
    //     P^{C,L}(q, i.nu) = -2 dPhi^L/dW = Pi^{C,0}(q, i.nu)
    // at FULL parent-normalized weight, with complete external frequency dependence. That
    // full weight is the whole point: the naive "make one rung static" functional has only
    // ONE W appearance and gives HALF the weight -- an error that only a W-side
    // (dPhi/dW) consistency check detects, where it shows up as exactly a factor 2.
    // Mechanically this is the SAME kernel with the rung Z -> W0bar and NO dynamic rung:
    // the internal convolution collapses into two decoupled bubbles (the instantaneous
    // Z-phase), so B-L needs no pole algebra here either.
    const bool lin = (_rung == linear_rung);
    if (lin) {
      auto const& W0b_r = _W0b_qmm.value();
      utils::check(W0b_r.shape(0) == nqpts_ibz and W0b_r.shape(1) == naux,
                   "vertex_t::eval_Pi_C: W0bar shape ({}, {}, {}) does not match the "
                   "kernel basis (nq = {}, naux = {}).", W0b_r.shape(0), W0b_r.shape(1),
                   W0b_r.shape(2), nqpts_ibz, naux);
      app_log(1, "  Pi^C [rung = linear]: P^(C,L) = Pi^(C,0) with the STATIC rung W0bar "
                 "at FULL parent weight (instantaneous phase only; no pole algebra).");
      if (sec)
        vertex_pi::pi_c_accumulate_w(*_ft, tools, G_CC, _Xb_skma, W0b_r, static_cast<nda::array<ComplexType, 4> const*>(nullptr),
                                     kmq, kpq, nda::range(0, nc), Pi_wqMN,
                                     mpi->comm.rank(), mpi->comm.size(),
                                     skip_rung_gamma, &qx_diag, symc, &phase_diag);
      else
        vertex_pi::pi_c_accumulate_w(*_ft, tools, G_CC, X_C, W0b_r, static_cast<nda::array<ComplexType, 4> const*>(nullptr),
                                     kmq, kpq, nda::range(0, nc), Pi_wqMN,
                                     mpi->comm.rank(), mpi->comm.size(),
                                     skip_rung_gamma, &qx_diag, symc, &phase_diag,
                                     &qplan.value());
    } else if (sec)
      vertex_pi::pi_c_accumulate_w(*_ft, tools, G_CC, _Xb_skma, Zb_qmm,
                                   Wbdyn_qwmm.has_value() ? &Wbdyn_qwmm.value() : nullptr,
                                   kmq, kpq, nda::range(0, nc), Pi_wqMN,
                                   mpi->comm.rank(), mpi->comm.size(),
                                   skip_rung_gamma, &qx_diag, symc, &phase_diag);
    else
      vertex_pi::pi_c_accumulate_w(*_ft, tools, G_CC, X_C, Z_qPQ,
                                   Wdyn_qwPQ.has_value() ? &Wdyn_qwPQ.value() : nullptr,
                                   kmq, kpq, nda::range(0, nc), Pi_wqMN,
                                   mpi->comm.rank(), mpi->comm.size(),
                                   skip_rung_gamma, &qx_diag, symc, &phase_diag,
                                   &qplan.value());
    _Timer.stop("PI_KERNEL");
    _Timer.start("PI_UPFOLD_REDUCE");
    // DO NOT all_reduce the full partial Pi_wqMN. It is a PARTIAL (this rank's
    // round-robin tuple/q_ext contribution); the upfold (t^dag Pibar t) and the tau
    // conversion are LINEAR and commute with the rank sum, so they are applied to the
    // PARTIAL and the result is REDUCE-SCATTERED directly into the RPA grid. This avoids
    // a full-array all_reduce and any persistent full replicated Pi_up / Pi_tqMN. Only
    // qx_diag (tiny) is all_reduced.
    mpi->comm.all_reduce_in_place_n(qx_diag.data(), qx_diag.size(), std::plus<>{});

    // ---- PROJECT P^C ONTO ITS EXACT SYMMETRY CLASS ------------------------------------
    //   P_PQ(q) = P_QP(-q)     (exact; the same relation every rung obeys)
    // The computed P^C satisfies it only to round-off. That would be harmless -- except
    // that B-L (and the dynamic theory) INJECT P^C into the Dyson equation
    //   W = v + v (P_RPA + P^C) W,
    // which closes a loop W -> Sigma -> G -> P -> W. The loop gain for the non-Hermitian
    // component exceeds 1, so a round-off seed GROWS geometrically from iteration to
    // iteration (visible as a growing Im(e_corr)/Re(e_corr)) unless the illegal component
    // is projected out. B-S, where P stays RPA, has no such feedback path. Delta w is
    // projected the same way (vertex_detail::build_delta_w).
    //
    // Applied to the PARTIAL: the projection is LINEAR, so it commutes with the
    // round-robin rank sum exactly as the upfold and the tau conversion do (same identity
    // cited above). The q axis is full on every rank, so this costs no communication.
    {
      auto qminus_map = MF->qminus();
      long n_done = 0, n_skip = 0;
      double d_max = 0.0, sc_max = 0.0;
      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        const long iqm = qminus_map(iq);
        // Under an IBZ mesh -q need not be a stored row; those q are left alone (and
        // counted). NOTE a Gamma-centred EVEN mesh (e.g. 2x2x2) has 2q = G for every q,
        // i.e. every transfer is SELF-INVERSE -- there iqm == iq and coverage is complete.
        if (iqm >= nqpts_ibz) { ++n_skip; continue; }
        if (iqm < iq) continue;                    // already handled with its partner
        ++n_done;                                  // global count -- every rank agrees
        // SLAB: each rank projects only the rows it owns. Ownership is
        // +-q-ORBIT-closed, so iq and iqm are always co-resident: both slab rows exist
        // or neither does, and the projection needs no communication either way.
        const long rA = qplan ? qplan->slab_of[iq] : iq;
        if (rA < 0) continue;                      // not my orbit
        const long rB = qplan ? qplan->slab_of[iqm] : iqm;
        utils::check(rB >= 0,
                     "vertex_t::eval_Pi_C: slab ownership is not orbit-closed "
                     "(iq = {} owned, -q partner {} not).", iq, iqm);
        for (long l = 0; l < tools.nw_b; ++l) {
          auto A = Pi_wqMN(l, rA, all, all);
          if (iqm == iq) {
            // self-inverse transfer: the relation reduces to P(q) = P(q)^T
            for (long M = 0; M < naux; ++M)
              for (long N = M; N < naux; ++N) {
                const ComplexType s = 0.5 * (A(M, N) + A(N, M));
                d_max = std::max(d_max, std::abs(A(M, N) - A(N, M)));
                sc_max = std::max(sc_max, std::abs(A(M, N)));
                A(M, N) = s;  A(N, M) = s;
              }
          } else {
            // P_sym(q)_MN == P_sym(-q)_NM, so each (M,N) writes exactly the two slots it
            // reads -- safe in place, no temporary.
            auto B = Pi_wqMN(l, rB, all, all);
            for (long M = 0; M < naux; ++M)
              for (long N = 0; N < naux; ++N) {
                const ComplexType s = 0.5 * (A(M, N) + B(N, M));
                d_max = std::max(d_max, std::abs(A(M, N) - B(N, M)));
                sc_max = std::max(sc_max, std::abs(A(M, N)));
                A(M, N) = s;  B(N, M) = s;
              }
          }
        }
      }
      double gl[2] = {d_max, sc_max};
      mpi->comm.all_reduce_in_place_n(gl, 2, boost::mpi3::max<>{});
      app_log(2, "  Pi^C pair-symmetry projection: {} of {} stored q projected ({} left "
                 "(no stored -q); rank-local |P_PQ(q) - P_QP(-q)| = {:.3e}, scale {:.3e})",
              n_done, nqpts_ibz, n_skip, gl[0], gl[1]);
      // The INJECTION predicate. eval_Pi_C runs only for the rungs that inject P^C into the
      // Dyson equation (dynamic: Pi^C; linear: P^{C,L}; static is rejected at the top), and its one production caller
      // (scr_coulomb_t's add_vertex_Pi_C) adds the result to P. So an unprojected transfer here IS injected: abort unless
      // pol_vertex_allow_unprojected (class default true keeps the WARNING for the direct-call readout / unit tests).
      const bool pi_c_injected = (_rung == dynamic_rung or _rung == linear_rung);
      utils::check(n_skip == 0 or not pi_c_injected or _allow_unprojected,
                   "vertex_t::eval_Pi_C: {} of {} stored transfers have no stored -q partner (IBZ mesh), so the pair-symmetry "
                   "projection P_PQ(q) = P_QP(-q) cannot be applied there, and this Pi^C (vertex_rung = \"{}\") is injected into "
                   "the Dyson equation, where the unprojected component is amplified by the self-consistency loop (audit D5). "
                   "Use a symmetry-free (nosym) k-mesh, or set pol_vertex_allow_unprojected = true to continue with a WARNING.",
                   n_skip, nqpts_ibz, rung_str());
      if (n_skip > 0)
        app_log(1, "  [WARNING] Pi^C: {} of {} stored transfers have no stored -q partner "
                   "(IBZ mesh), so the\n"
                   "            pair-symmetry projection could not be applied there. In a "
                   "theory that injects\n"
                   "            P^C into the Dyson equation (linear / dynamic rung) the "
                   "unprojected component is\n"
                   "            amplified by the self-consistency loop.", n_skip, nqpts_ibz);
    }

    // ---- KERNEL SCALES (blow-up diagnostic) -------------------------------------------
    // Pi^C is MULTILINEAR in (G,G,G,G,W): bounded inputs => Lipschitz, so a small change in
    // G cannot produce a huge change in Pi^C. With G and W bounded, a blow-up of Pi^C is
    // therefore INTERNAL to this routine. These norms split it three ways:
    //   Zbar/Wbar huge  => the DOWNFOLD (t W t^dag) is at fault
    //   Pibar huge with bounded inputs => the CONTRACTION is
    //   only the upfolded Pi^C huge => t / the UPFOLD is
    // NB Pibar is this rank's round-robin PARTIAL, so the reduction is a max over partials,
    // not the max of the sum -- fine for spotting an explosion, not a physical norm.
    {
      auto amax = [](auto const &A) {
        double m = 0.0;
        for (auto const &v : A) m = std::max(m, std::abs(v));
        return m;
      };
      double g_m = amax(G_CC), t_m = amax(_t_qmP), pib = amax(Pi_wqMN);
      double zb_m = sec ? amax(Zb_qmm) : 0.0;
      double wb_m = (sec and Wbdyn_qwmm.has_value()) ? amax(Wbdyn_qwmm.value()) : 0.0;
      pib = mpi->comm.all_reduce_value(pib, boost::mpi3::max<>{});
      g_m = mpi->comm.all_reduce_value(g_m, boost::mpi3::max<>{});
      mpi->comm.all_reduce_in_place_n(phase_diag.data(), phase_diag.size(),
                                      boost::mpi3::max<>{});
      app_log(1, "  [ISDF-Vertex] kernel scales: max|G_CC| = {:.4e}  max|t| = {:.4e}  "
                 "max|Zbar| = {:.4e}  max|Wbar| = {:.4e}  max|Pibar(partial)| = {:.4e}",
              g_m, t_m, zb_m, wb_m, pib);
      // Phase 1 is the pole-free instantaneous Z rung; Phase 2 is the ONLY part running the
      // DLR pole algebra. If Pibar is already huge after Phase 1 the fault is in the exact
      // bubble contraction; if it is small there and huge at the end, it is the pole algebra
      // -- and then max|z| vs max|pole residue| says whether pole_coeffs is the amplifier.
      app_log(1, "  [ISDF-Vertex] phase split: max|Pibar after Phase 1 (pole-free)| = {:.4e}  "
                 "max|z| = {:.4e}  max|DLR residue of z| = {:.4e}  pole-fit rel err = {:.4e}"
                 "  -> final = {:.4e}",
              phase_diag(0), phase_diag(2), phase_diag(1), phase_diag(3), pib);
      if (phase_diag(3) > 1e-3)
        app_log(1, "  [WARNING] the auxiliary DLR pole fit is NOT reproducing the z objects "
                   "(rel err {:.2e}).\n"
                   "            Its residues then enter the twisted-pair algebra as products, "
                   "so this is\n"
                   "            squared into Pi^C. See notes/vertex_divergence_diagnosis.md.",
                phase_diag(3));
    }

    // per-qx rung diagnostics (sum of rank-local maxima -- order-of-magnitude indicator
    // for the q->0 head pathology; Gamma reads 0 when skipped)
    {
      long iqg = -1;
      for (long iq = 0; iq < nqpts; ++iq) {
        bool isg = true;
        for (long ik = 0; ik < nkpts; ++ik)
          if (kmq(iq, ik) != ik) { isg = false; break; }
        if (isg) { iqg = iq; break; }
      }
      double g_val = (iqg >= 0) ? qx_diag(iqg) : -1.0;
      double other = 0.0;
      for (long iqx = 0; iqx < nqpts; ++iqx) {
        app_log(3, "  Pi^C rung diagnostics: qx = {}  max|contribution| = {}", iqx, qx_diag(iqx));
        if (iqx != iqg) other = std::max(other, qx_diag(iqx));
      }
      app_log(2, "  Pi^C rung per-qx |contribution|: Gamma(iq={}) = {}, max(other qx) = {}\n",
              iqg, g_val, other);
    }

    // ---- no-leak tripwire (<Pibar, Zbar> = <Pi, Z>; DIAGNOSTIC only) -- needs the REDUCED Pi
    // The tripwire compares upfolded-vs-downfolded traces at the nu = 0 node, which needs
    // the SUMMED Pi_bar. all_reduce ONLY the m0 slice (nq * naux^2 -- small), upfold that
    // one slice, and check. This keeps the diagnostic exact without reducing the full array.
    if (sec) {
      nda::array<ComplexType, 3> Pibar_m0(nqpts_ibz, _Nm, _Nm);
      Pibar_m0() = Pi_wqMN(tools.m0, all, all, all);
      mpi->comm.all_reduce_in_place_n(Pibar_m0.data(), Pibar_m0.size(), std::plus<>{});
      nda::array<ComplexType, 2> Pi_up0(Np, Np), tmp(Np, _Nm);
      double leak_max = 0.0;
      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        // full replicated Z(iq) for the bare-Z pairing: pulled locally (the secondary path
        // does not materialize Z_qPQ). thc.Z is collective and this loop is uniform
        // across ranks. Head-augment at Gamma exactly as the global build (Z += H_PQ).
        nda::array<ComplexType, 2> Zq = thc.Z(int(iq));
        if (head_ok and iq == iq_gamma)
          vertex_secondary_detail::head_block_add(chi_g, head_c, ComplexType(1.0),
                                                  nda::range(0, Np), nda::range(0, Np), Zq());
        vertex_secondary_detail::upfold_core(_t_qmP(iq, all, all), Pibar_m0(iq, all, all),
                                             tmp, Pi_up0);
        ComplexType S_up(0.0), S_bar(0.0);
        for (long M = 0; M < Np; ++M)
          for (long N = 0; N < Np; ++N) S_up += Pi_up0(M, N) * Zq(N, M);
        for (long m = 0; m < _Nm; ++m)
          for (long n = 0; n < _Nm; ++n) S_bar += Pibar_m0(iq, m, n) * Zb_qmm(iq, n, m);
        leak_max = std::max(leak_max, std::abs(S_up - S_bar) /
                                      std::max(std::abs(S_bar), 1e-300));
      }
      app_log(2, "  Refinement 2 no-leak residual (Eq. 39; nu = 0 node, bare-Z pairing): "
                 "max_q = {}", leak_max);
    }

    // ---- upfold + tau conversion, then materialize the RPA-distributed Pi^C -------------
    // The upfold (t^dag Pibar t) and the tau conversion are LINEAR and commute with the
    // round-robin rank sum, so both may be applied to the PARTIAL Pi_wqMN.
    auto dPi_C_tqPQ = math::nda::make_distributed_array<memory::array<HOST_MEMORY, ComplexType, 4>>(
        mpi->comm, pi_pgrid, pi_gshape, pi_bsize);
    if (sec) {
      // SECONDARY: distribute the Np^2 upfold over the RPA (P,Q) grid (adjoint of
      // fold_dW_distributed). The full Np^2 upfold partial is never materialized; each rank
      // upfolds ONLY its owned (P,Q) block directly into dPi_C_tqPQ.local(). First sum the
      // SMALL N_m^2 partial across comm (upfold+tau are linear, so the rank sum before the
      // upfold == reduce-scatter after).
      mpi->comm.all_reduce_in_place_n(Pi_wqMN.data(), Pi_wqMN.size(), std::plus<>{});

      // this rank's block ranges into the global (nt_half, nq, Np, Np) grid. q (axis 1) is
      // NOT split; only t (axis 0) and P,Q (axes 2,3) are.
      auto grd = dPi_C_tqPQ.grid();
      utils::check(grd[1] == 1,
                   "vertex_t::eval_Pi_C: the q axis of the Pi^C grid must NOT be split "
                   "(grid[1] = {} != 1).", grd[1]);
      auto t_range = dPi_C_tqPQ.local_range(0);
      auto P_range = dPi_C_tqPQ.local_range(2);
      auto Q_range = dPi_C_tqPQ.local_range(3);
      const long P_bs = dPi_C_tqPQ.local_shape()[2];
      const long Q_bs = dPi_C_tqPQ.local_shape()[3];

      // upfold ONLY my (P,Q) block over the full bosonic mesh, then tau-convert the block.
      nda::array<ComplexType, 4> Pi_up_blk(tools.nw_b, nqpts_ibz, P_bs, Q_bs);
      nda::array<ComplexType, 2> tmp(P_bs, _Nm);
      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        auto t_q = _t_qmP(iq, all, all);                 // (N_m x Np)
        auto t_qP = t_q(all, P_range);                   // (N_m x P_bs)
        auto t_qQ = t_q(all, Q_range);                   // (N_m x Q_bs)
        for (long l = 0; l < tools.nw_b; ++l)
          vertex_secondary_detail::upfold_core_block(t_qP, t_qQ, Pi_wqMN(l, iq, all, all),
                                                     tmp, Pi_up_blk(l, iq, all, all));
      }
      // tau-convert the BLOCK (pi_w_to_code_tau transforms the w<->t axis independent of the
      // last two dims -- it reads Np from shape(2)/shape(3), so block shapes work as-is).
      nda::array<ComplexType, 4> Pi_t_blk(nt_half, nqpts_ibz, P_bs, Q_bs);
      vertex_pi::pi_w_to_code_tau(*_ft, tools, Pi_up_blk, Pi_t_blk);
      // write my t-slice of the block into the owned local() (q axis is full: local q == 0..nq).
      dPi_C_tqPQ.local() = Pi_t_blk(t_range, all, all, all);
    } else {
      // GLOBAL: Pi_wqMN is the OWNED-ROW SLAB of the partial. tau-convert the slab --
      // upfold-free here, and pi_w_to_code_tau's internal (nt, nq, ., .) transient shrinks
      // with it -- and reduce-scatter into the RPA grid, packing ZEROS for the rows this
      // rank does not own: the same math as a full-shape partial, whose non-owned rows are
      // structurally zero. On one rank this is a bit-identical copy.
      nda::array<ComplexType, 4> Pi_tqMN(nt_half, n_qrows, Np, Np);
      vertex_pi::pi_w_to_code_tau(*_ft, tools, Pi_wqMN, Pi_tqMN);
      vertex_redist_detail::reduce_scatter_slab_into(Pi_tqMN, qplan->slab_of,
                                                     dPi_C_tqPQ, mpi->comm);
    }

    {
      // VERTEX RAMP / SCALE. Both cuts carry the SAME lambda, which is exactly
      // Phi_2^C -> lambda Phi_2^C: the approximation acts on the GENERATING FUNCTIONAL,
      // not on the already-cut Sigma/P, so Phi-derivability and the conservation
      // identity survive at every lambda.
      // Used to walk the vertex in continuously: P^C is not sign-definite, so a full-
      // strength vertex can push eps = I - Z.Pi through zero and break the W-Dyson
      // solve. Ramping finds the
      // largest lambda whose solution still has a positive-definite eps.
      const double lam = vertex_scale();
      if (lam != 1.0) {
        dPi_C_tqPQ.local() *= ComplexType(lam);
        app_log(1, "  [ISDF-Vertex] Pi^C scaled by lambda = {:.4f} (ramp iteration {} of "
                   "{})", lam, _vertex_iter, _ramp_iters);
      }
    }
    {
      // NaN/Inf guard on THIS rank's owned block (the full array is never materialized)
      double max_abs = 0.0;
      long n_bad = 0;
      for (auto const& v : dPi_C_tqPQ.local()) {
        double a = std::abs(v);
        if (not std::isfinite(a)) { ++n_bad; continue; }
        max_abs = std::max(max_abs, a);
      }
      n_bad = mpi->comm.all_reduce_value(n_bad, std::plus<>{});
      max_abs = mpi->comm.all_reduce_value(max_abs, boost::mpi3::max<>{});
      utils::check(n_bad == 0,
                   "vertex_t::eval_Pi_C: Pi^C contains {} NaN/Inf entries -- aborting.", n_bad);
      app_log(2, "  Pi^C(tau) max|.| = {}\n", max_abs);
    }
    _Timer.stop("PI_UPFOLD_REDUCE");
    // See SIG_BARRIER: timed apart so cross-rank skew is not charged to the upfold.
    _Timer.start("PI_BARRIER");
    mpi->comm.barrier();
    _Timer.stop("PI_BARRIER");

    return dPi_C_tqPQ;
  }

  void vertex_t::cache_w(MBState &mb_state, THC_ERI auto const &thc) {
    vertex_timer_detail::scoped_timer _tm_total(_Timer, "CACHE_W");
    decltype(nda::range::all) all;
    utils::check(active() and secondary(),
                 "vertex_t::cache_w: requires an ACTIVE vertex in isdf mode \"secondary\" "
                 "(the global path retains the full dW instead; notes/wbar_cache.md).");
    // On the DEVICE path the source is the device W (mb_state.dW_qtPQ_dev) whenever it is present;
    // update_w then keeps no host mirror for this consumer (vertex_t::reads_host_W).
#if defined(ENABLE_DEVICE)
    const bool have_dev_W = mb_state.dW_qtPQ_dev.has_value();
#else
    const bool have_dev_W = false;
#endif
    utils::check(mb_state.dW_qtPQ.has_value() or have_dev_W,
                 "vertex_t::cache_w: dW_qtPQ is not initialized in MBState -- cache_w "
                 "must run at the scr_coulomb_t::update_w tail, after the new W is stored.");
    utils::check(mb_state.sG_tskij.has_value(),
                 "vertex_t::cache_w: sG_tskij is not initialized in MBState.");

    auto mpi = thc.mpi();
    auto MF = thc.MF();
    const long nkpts = MF->nkpts();
    const long nqpts = MF->nqpts();
    const long nqpts_ibz = MF->nqpts_ibz();
    const long nkpts_ibz = MF->nkpts_ibz();
    const long Np = thc.Np();
    const long nbnd = MF->nbnd();
    const long ns = mb_state.sG_tskij.value().local().shape(1);
    const bool wan = _wannier;
    const long nc = subspace_rank();

    // IBZ symmetry: the fill runs over IBZ q only -- which is exactly the cache's
    // q-keyed first axis; consumption at non-IBZ transfers goes through the kernels'
    // symmetry context.
    (void)nqpts; (void)nkpts; (void)nkpts_ibz;

    vertex_pi::iaft_tools tools(*_ft);
    const long nw_b = tools.nw_b;
    const long nw_half = (nw_b % 2 == 0) ? nw_b / 2 : nw_b / 2 + 1;
#if defined(ENABLE_DEVICE)
    auto gs = mb_state.dW_qtPQ.has_value() ? mb_state.dW_qtPQ.value().global_shape() : mb_state.dW_qtPQ_dev.value().global_shape();
#else
    auto gs = mb_state.dW_qtPQ.value().global_shape();
#endif
    const long nt_half = gs[1];
    utils::check(gs[0] == nqpts_ibz and gs[2] == Np and gs[3] == Np,
                 "vertex_t::cache_w: unexpected dW_qtPQ global shape ({}, {}, {}, {}).",
                 gs[0], gs[1], gs[2], gs[3]);

    app_log(1, "\n  Refinement 2: caching the downfolded rung Wbar = t dW t^dag "
               "(notes/wbar_cache.md)\n"
               "  -- filled at update_w time from THIS iteration's (dW, eps_inv_head); "
               "consumed by the NEXT\n"
               "  iteration's Pi^C (one-iteration lag); dW itself is then freed by the "
               "scf driver.");

    // ---- collocation + momentum maps (for the lazy basis build + diagnostics) --------
    // node-share X_skPa (one copy per node; see eval_Sigma_C).
    auto sX_skPa = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(
        *mpi, std::array<long, 4>{ns, nkpts, Np, nbnd});
    sX_skPa.win().fence();
    if (mpi->node_comm.root())
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nkpts; ++ik)
          sX_skPa.local()(is, ik, all, all) = thc.X(is, 0, ik);
    sX_skPa.win().fence();
    auto X_skPa = sX_skPa.local();
    nda::array<long, 2> kmq(nqpts_ibz, nkpts);
    for (long iq = 0; iq < nqpts_ibz; ++iq)
      for (long ik = 0; ik < nkpts; ++ik) kmq(iq, ik) = MF->qk_to_k2(iq, ik);
    // effective global collocation for the secondary fit (WANNIER: X_bar = X.U, orb0=0).
    // node-shared so X_glob is a plain view (both branches array_view<ComplexType,4>).
    auto sXbar_glob = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(
        *mpi, std::array<long, 4>{ns, nkpts, Np, wan ? subspace_rank() : 1});
    if (wan) {
      sXbar_glob.win().fence();
      if (mpi->node_comm.root())
        sXbar_glob.local() = vertex_wannier_detail::build_Xbar(X_skPa, _U_skia, _band_window);
      sXbar_glob.win().fence();
    }
    auto X_glob = wan ? sXbar_glob.local() : X_skPa;
    const long orb0_glob = wan ? 0 : _band_window.first();

    // ---- q = Gamma index (crystal coordinates: all components integer mod G) ----------
    long iq_gamma = -1;
    {
      auto Qpts = MF->Qpts();
      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        double d = 0.0;
        for (long i = 0; i < 3; ++i) {
          double x = Qpts(iq, i);
          d += std::abs(x - std::round(x));
        }
        if (d < 1e-8) {
          utils::check(iq_gamma < 0,
                       "vertex_t::cache_w: multiple Gamma q-points found ({} and {}).",
                       iq_gamma, iq);
          iq_gamma = iq;
        }
      }
      utils::check(iq_gamma >= 0, "vertex_t::cache_w: no Gamma q-point found.");
    }

    // (idempotent; in the production flow eval_Pi_C already built it this iteration)
    build_secondary_basis(thc, X_glob, orb0_glob, kmq, iq_gamma);

    // ---- q->0 rung policy: SAME resolution as eval_Pi_C ------------------------------
    // Only the gygi head insertion matters here (v1_skip acts in the kernel, not on
    // the stored W content). eps_inv_head is captured NOW -- the same iteration as W.
    bool head_insertion = (_div_treatment.find("gygi") != std::string::npos);
    if (head_insertion and nqpts_ibz == 1) {
      app_log(1, "  [WARNING] cache_w: nqpts_ibz == 1 with a gygi-class vertex "
                 "div_treatment -- taking \"ignore_g0\" instead (same downgrade as "
                 "eval_Pi_C).");
      head_insertion = false;
    }
    nda::array<ComplexType, 2> H_PQ(Np, Np);
    bool head_ok = false;
    if (head_insertion) {
      head_ok = vertex_head_detail::build_head_rank1(thc, iq_gamma, nkpts, H_PQ,
                                                                    _bl_head_scale);
      if (not head_ok and _bl_head_scale == 0.0)
        app_log(1, "  [W-int-3] cache_w: vertex_bl_head_scale = 0 -- the dynamic rung W-bar(q, i nu) carries NO analytic "
                   "q -> 0 head (body-only vertex kernel).");
      else if (not head_ok and head_unusable_continue("vertex_t::cache_w"))   // aborts unless allowed
        app_log(1, "  [WARNING] cache_w: gygi head insertion requested but head data "
                   "are unusable\n"
                   "            (madelung == 0 or empty basis_head) -- caching WITHOUT "
                   "the analytic head\n"
                   "            (equivalent to policy \"ignore_g0\").");
    }
    // Decided HERE, collectively (the per-q head insertion below runs on the owner rank of Gamma only), whether
    // the dynamic head piece can be built; abort unless vertex_allow_missing_head (then the in-loop WARNING says it).
    if (head_ok and not (_bl_head_static_all and _rung == linear_rung) and not mb_state.eps_inv_head.has_value())
      dyn_head_missing("vertex_t::cache_w");

    // ---- per q: gather dW(tau), augment the Gamma head, transform to the half nu mesh, fold
    // Identical arithmetic to the fold-at-consumption path (eval_Pi_C):
    // augment BEFORE tau_to_w_PHsym, per-q transform on the same (nt_half, Np, Np)
    // tau-storage slices.
    // MEMORY-LEAN: every q slice is obtained on its own (gather_dW_one_q is bit-identical to
    // slicing a replicated array), transformed, folded and dropped, so only ONE q of tau-
    // and omega-domain W is alive per rank -- never the replicated (nq x nt_half x Np^2)
    // tau slab or (nq x nw_half x Np^2) omega slab.
    const bool eta_diag = (ns * nkpts * nc * nc <= 4096);
    const long lpos0 = std::max(tools.m0, tools.w_mirror_b(tools.m0)) - nw_b / 2;
    const long lposm = std::max(nw_b - 1, tools.w_mirror_b(nw_b - 1)) - nw_b / 2;
    // the two omega slices the eta diagnostic reads, for every q (replicated, small)
    nda::array<ComplexType, 4> W_diag(eta_diag ? nqpts_ibz : 0, 2, eta_diag ? Np : 0, eta_diag ? Np : 0);
    if (eta_diag) W_diag() = ComplexType(0.0);
    // The storage: one array per rank ("replicated": each q folded by ONE rank globally and
    // gathered by a zero-padded all_reduce), or one node-shared window per NUMA node ("shared": each q folded by one rank
    // PER NODE into the shared window, no all_reduce; the same fold_core gemms, bitwise the same values).
    const bool shm_cache = (_wcache == "shared");
    _Wb_qwmm.reset(); _Wb_shm.reset();
    if (shm_cache) {
      _Wb_shm = std::make_shared<math::shm::shared_array<nda::array_view<ComplexType, 4>>>(
          math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(*mpi, std::array<long, 4>{nqpts_ibz, nw_half, _Nm, _Nm}));
      _Wb_shm->win().fence();
      if (mpi->node_comm.root()) _Wb_shm->local()() = ComplexType(0.0);
      _Wb_shm->win().fence();
    } else {
      _Wb_qwmm.emplace(nda::array<ComplexType, 4>(nqpts_ibz, nw_half, _Nm, _Nm));
      _Wb_qwmm.value()() = ComplexType(0.0);
    }
    nda::array_view<ComplexType, 4> Wb = shm_cache ? _Wb_shm->local() : nda::array_view<ComplexType, 4>(_Wb_qwmm.value());
    const bool diag_writer = shm_cache ? (mpi->internode_comm.rank() == 0) : true;   // the eta slices: one writer per q globally
    {
      bool head_logged = false;
      long my_nfold = 0;
      // The replicated-cache path REDISTRIBUTES the distributed dW once into whole-q slabs (q-only grid, one all-to-all)
      // instead of one zero-padded all-reduce per q, and every rank folds the q it then owns -- on the host with the same
      // transform + fold_core calls (bitwise the same values: the data only moves), or under DEVICE on the device (upload
      // per q, the PH-sym tau -> nu transform and the two fold gemms through cuBLAS; only (nw_half, N_m, N_m) per q comes
      // down). vertex_debug cache_w_redist = 0 selects the per-q gather loop, cache_w_device = 0 keeps the redistributed
      // folds on the host. The node-shared cache uses the gather loop.
      bool fill_dev = false;
      const bool fill_redist = (not shm_cache and nqpts_ibz >= long(mpi->comm.size()) and
                                vertex_debug::number("cache_w_redist", 1.0) != 0.0);   // vertex_debug: cache_w_redist
#if defined(ENABLE_DEVICE)
      if (fill_redist) fill_dev = (vertex_debug::number("cache_w_device", 1.0) != 0.0);   // vertex_debug: cache_w_device
#endif
      // the source of the redistribution: the device W (device-to-device, no host mirror) when the folds run there too
      const bool src_dev = fill_dev and have_dev_W;
      // every other path reads the host mirror: materialize it on demand when update_w kept none
#if defined(ENABLE_DEVICE)
      if (not src_dev and not mb_state.dW_qtPQ.has_value()) {
        app_log(1, "  cache_w: the fold runs on the host -- materializing the host mirror of W from the device W.");
        (void)mb_state.W_host();
      }
#endif
      if (fill_redist) {
        nda::array<ComplexType, 3> W_head;                // the Gamma slab with the head inserted (one q)
        nda::array<ComplexType, 3> W_w(fill_dev ? 0 : nw_half, fill_dev ? 0 : Np, fill_dev ? 0 : Np);
        nda::array<ComplexType, 2> tmp(fill_dev ? 0 : _Nm, fill_dev ? 0 : Np);
#if defined(ENABLE_DEVICE)
        std::optional<memory::array<DEVICE_MEMORY, ComplexType, 3>> Ww_d, out_d;
        std::optional<memory::array<DEVICE_MEMORY, ComplexType, 2>> tmp_d;
        if (fill_dev) { Ww_d.emplace(nw_half, Np, Np); out_d.emplace(nw_half, _Nm, _Nm); tmp_d.emplace(_Nm, Np); }
#endif
        // the folds of the whole-q slabs this rank owns; loc is the host or (src_dev) the device slab view
        auto fold_slabs = [&](auto loc, long q0, long nql) {
          constexpr bool loc_host = nda::mem::on_host<std::decay_t<decltype(loc)>>;
          for (long iql = 0; iql < nql; ++iql) {
            const long iq = q0 + iql;
            ++my_nfold;
            bool use_head = false;
            if (head_ok and iq == iq_gamma) {
              if (_bl_head_static_all and _rung == linear_rung) {
                app_log(1, "  cache_w head insertion [H1 STATIC]: dynamic piece SKIPPED for the "
                           "cached B-L rung (dW is analytic-head-free).");
              } else if (mb_state.eps_inv_head.has_value()) {
                auto& eps = mb_state.eps_inv_head.value();
                utils::check(eps.shape(0) == nt_half,
                             "vertex_t::cache_w: eps_inv_head size {} != nt_half = {}.",
                             eps.shape(0), nt_half);
                if constexpr (loc_host) W_head = nda::array<ComplexType, 3>(loc(iql, all, all, all));
                else W_head = nda::array<ComplexType, 3>(memory::to_memory_space<HOST_MEMORY>(loc(iql, all, all, all)));
                for (long it = 0; it < nt_half; ++it)
                  W_head(it, all, all) += ComplexType(eps(it).real()) * H_PQ;
                use_head = true;
                app_log(1, "  cache_w head insertion: dynamic piece applied to dW(Gamma, tau) "
                           "with eps_inv_head(tau=0) = {}\n"
                           "  (SAME-iteration eps_inv_head, captured at fill time)",
                        eps(0).real());
              } else {
                app_log(1, "  [WARNING] cache_w: dW is present but eps_inv_head is not in "
                           "MBState -- the DYNAMIC head\n"
                           "            piece is skipped for the cached rung.");
              }
              head_logged = true;
            }
            auto t_q = _t_qmP(iq, all, all);
            if (not fill_dev) {
              if constexpr (loc_host) {
                if (use_head) _ft->tau_to_w_PHsym(W_head, W_w);
                else _ft->tau_to_w_PHsym(loc(iql, all, all, all), W_w);
                if (eta_diag and diag_writer) {
                  W_diag(iq, 0, all, all) = W_w(lpos0, all, all);
                  W_diag(iq, 1, all, all) = W_w(lposm, all, all);
                }
                for (long lp = 0; lp < nw_half; ++lp)
                  vertex_secondary_detail::fold_core(t_q, W_w(lp, all, all), tmp, Wb(iq, lp, all, all));
              }
            }
#if defined(ENABLE_DEVICE)
            else {
              // the tau slab on the device: uploaded (host source, or the Gamma slab with its head), else the slab itself
              std::optional<memory::array<DEVICE_MEMORY, ComplexType, 3>> Wt_up;
              if (use_head) Wt_up.emplace(memory::to_memory_space<DEVICE_MEMORY>(W_head));
              else if constexpr (loc_host)
                Wt_up.emplace(memory::to_memory_space<DEVICE_MEMORY>(nda::array<ComplexType, 3>(loc(iql, all, all, all))));
              if (Wt_up.has_value()) _ft->tau_to_w_PHsym(*Wt_up, *Ww_d);
              else if constexpr (not loc_host) _ft->tau_to_w_PHsym(loc(iql, all, all, all), *Ww_d);
              if (eta_diag and diag_writer) {
                W_diag(iq, 0, all, all) = nda::array<ComplexType, 2>(memory::to_memory_space<HOST_MEMORY>((*Ww_d)(lpos0, all, all)));
                W_diag(iq, 1, all, all) = nda::array<ComplexType, 2>(memory::to_memory_space<HOST_MEMORY>((*Ww_d)(lposm, all, all)));
              }
              nda::array<ComplexType, 2> t_h(t_q);
              auto t_d = memory::to_memory_space<DEVICE_MEMORY>(t_h);
              for (long lp = 0; lp < nw_half; ++lp) {       // fold_core's two gemms: (t W) then (t W) t^dag
                nda::blas::gemm(t_d, (*Ww_d)(lp, all, all), *tmp_d);
                nda::blas::gemm(*tmp_d, nda::dagger(t_d), (*out_d)(lp, all, all));
              }
              Wb(iq, all, all, all) = nda::array<ComplexType, 3>(memory::to_memory_space<HOST_MEMORY>(*out_d));
            }
#endif
          }
        };
#if defined(ENABLE_DEVICE)
        if (src_dev) {
          auto dWq = math::nda::make_distributed_array<memory::array<DEVICE_MEMORY, ComplexType, 4>>(
              mpi->comm, {long(mpi->comm.size()), 1l, 1l, 1l}, {nqpts_ibz, nt_half, Np, Np}, {1l, 1l, 1l, 1l});
          math::nda::redistribute(mb_state.dW_qtPQ_dev.value(), dWq);
          fold_slabs(dWq.local(), dWq.origin()[0], dWq.local_shape()[0]);
        } else
#endif
        {
          auto dWq = math::nda::make_distributed_array<nda::array<ComplexType, 4>>(
              mpi->comm, {long(mpi->comm.size()), 1l, 1l, 1l}, {nqpts_ibz, nt_half, Np, Np}, {1l, 1l, 1l, 1l});
          math::nda::redistribute(mb_state.dW_qtPQ.value(), dWq);
          fold_slabs(dWq.local(), dWq.origin()[0], dWq.local_shape()[0]);
        }
      }
      if (not fill_redist) {
      nda::array<ComplexType, 3> W_w(nw_half, Np, Np);
      nda::array<ComplexType, 2> tmp(_Nm, Np);
      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        // the collective per-q gather (every rank participates; the owner keeps the slab)
        nda::array<ComplexType, 3> W_t = vertex_redist_detail::gather_dW_one_q(
            mb_state.dW_qtPQ.value(), mpi->comm, iq, nt_half, Np);
        if (shm_cache ? (iq % mpi->node_comm.size() != mpi->node_comm.rank()) : (iq % mpi->comm.size() != mpi->comm.rank())) continue;
        ++my_nfold;
        if (head_ok and iq == iq_gamma) {
          if (_bl_head_static_all and _rung == linear_rung) {
            // Balanced head (see _bl_head_static_all): the cached B-L rung must be analytic-head-free
            // in its dynamic part, exactly like eval_Sigma_C's Wt_qtPQ -- the static-weight
            // head rides the instantaneous slot at the consumer.
            app_log(1, "  cache_w head insertion [H1 STATIC]: dynamic piece SKIPPED for the "
                       "cached B-L rung (dW is analytic-head-free).");
          } else if (mb_state.eps_inv_head.has_value()) {
            auto& eps = mb_state.eps_inv_head.value();
            utils::check(eps.shape(0) == nt_half,
                         "vertex_t::cache_w: eps_inv_head size {} != nt_half = {}.",
                         eps.shape(0), nt_half);
            for (long it = 0; it < nt_half; ++it)
              W_t(it, all, all) += ComplexType(eps(it).real()) * H_PQ;
            app_log(1, "  cache_w head insertion: dynamic piece applied to dW(Gamma, tau) "
                       "with eps_inv_head(tau=0) = {}\n"
                       "  (SAME-iteration eps_inv_head, captured at fill time)",
                    eps(0).real());
          } else {
            app_log(1, "  [WARNING] cache_w: dW is present but eps_inv_head is not in "
                       "MBState -- the DYNAMIC head\n"
                       "            piece is skipped for the cached rung.");
          }
          head_logged = true;
        }
        // tau -> omega on the PH-sym half mesh (per q, on its owner)
        _ft->tau_to_w_PHsym(W_t, W_w);
        if (eta_diag and diag_writer) {
          W_diag(iq, 0, all, all) = W_w(lpos0, all, all);
          W_diag(iq, 1, all, all) = W_w(lposm, all, all);
        }
        // fold on the half mesh: Wbar(q, nu) = t(q) Wdyn(q, nu) t(q)^dag (same fold_core gemms)
        auto t_q = _t_qmP(iq, all, all);
        for (long lp = 0; lp < nw_half; ++lp)
          vertex_secondary_detail::fold_core(t_q, W_w(lp, all, all), tmp,
                                             Wb(iq, lp, all, all));
      }
      }
      (void)head_logged;
      if (shm_cache) {
        _Wb_shm->win().fence();   // publish the node-shared cache (every q written by exactly one rank of the node)
      } else {
        // exact partition gathers (zero-padded all_reduce): bit-identical
        mpi->comm.all_reduce_in_place_n(_Wb_qwmm.value().data(), _Wb_qwmm.value().size(),
                                        std::plus<>{});
      }
      if (eta_diag) mpi->comm.all_reduce_in_place_n(W_diag.data(), W_diag.size(), std::plus<>{});
      const long total_fold = mpi->comm.all_reduce_value(my_nfold, std::plus<>{});
      app_log(2, "  Refinement 2 W-bar fold distributed over {} ranks: this rank folded "
                 "{} of {} q-points (~1/P work; one q of W alive per rank){}.", mpi->comm.size(), my_nfold,
              total_fold, fill_redist ? (src_dev ? " -- the device W redistributed to whole-q device slabs, folded ON THE DEVICE"
                                     : fill_dev ? " -- redistributed to whole-q slabs, folded ON THE DEVICE" : " -- redistributed to whole-q slabs") : "");
    }

    // ---- eta(q) diagnostics on the rung ACTUALLY cached (test scale: N_pair <= 4096) --
    if (eta_diag) {
      vertex_secondary_detail::eta_max_over_q(
          "dW(nu_0)", X_glob, orb0_glob, nc, _Xb_skma, _t_qmP, kmq,
          [&](long iq) { return W_diag(iq, 0, all, all); });
      vertex_secondary_detail::eta_max_over_q(
          "dW(nu_max)", X_glob, orb0_glob, nc, _Xb_skma, _t_qmP, kmq,
          [&](long iq) { return W_diag(iq, 1, all, all); });
    } else {
      app_log(2, "  Refinement 2: eta diagnostic skipped (N_pair = {} > 4096).",
              ns * nkpts * nc * nc);
    }

    // ---- REPLACE the cache by an externally approximated rung (accuracy studies of a factorized W-bar) ----
    // vertex_debug wbar_load = <file.h5> reads Wbar_qwmm (nq_ibz, nw_half, N_m, N_m) (the wbar_dump layout) into the cache.
    // The file's METADATA (the datasets wbar_dump writes next to Wbar_qwmm) is validated against
    // THIS run -- the shape alone cannot tell a rung of another window / basis / temperature / k-mesh. Required: window_first,
    // window_size, nm, beta, nw_half, kpts (abort when absent: copy them from the wbar_dump file into the approximated one);
    // Xb_skma is compared when present (the secondary collocation fixes the basis the rung is expressed in).
    if (auto wl = vertex_debug::get("wbar_load"); wl and not wl->empty()) {
      nda::array<ComplexType, 4> Wl;
      {
        h5::file f(*wl, 'r');
        h5::group g(f);
        nda::h5_read(g, "Wbar_qwmm", Wl);
        for (auto const *key : {"window_first", "window_size", "nm", "beta", "nw_half", "kpts"})
          utils::check(g.has_dataset(key),
                       "vertex_t::cache_w: wbar_load {} carries no \"{}\": its W-bar cannot be validated against this run "
                       "(audit A18). Copy the metadata datasets window_first, window_size, nm, beta, nw_half, kpts (and "
                       "Xb_skma) of the wbar_dump file into it.", *wl, key);
        long w0_in = -1, nw_in = -1, nm_in = -1, nwh_in = -1;
        double beta_in = 0.0;
        nda::array<double, 2> kpts_in;
        h5::h5_read(g, "window_first", w0_in);
        h5::h5_read(g, "window_size", nw_in);
        h5::h5_read(g, "nm", nm_in);
        h5::h5_read(g, "beta", beta_in);
        h5::h5_read(g, "nw_half", nwh_in);
        nda::h5_read(g, "kpts", kpts_in);
        utils::check(w0_in == long(_band_window.first()) and nw_in == long(_band_window.size()),
                     "vertex_t::cache_w: wbar_load {} was dumped on the window [{}, {}), this run's is [{}, {}) (audit A18).",
                     *wl, w0_in, w0_in + nw_in, _band_window.first(), _band_window.last());
        utils::check(nm_in == _Nm, "vertex_t::cache_w: wbar_load {} has N_m = {}, this run's secondary basis has N_m = {} "
                                   "(audit A18).", *wl, nm_in, _Nm);
        utils::check(std::abs(beta_in - tools.beta) <= 1e-10 * std::max(1.0, std::abs(tools.beta)),
                     "vertex_t::cache_w: wbar_load {} was dumped at beta = {}, this run's beta = {} (audit A18).",
                     *wl, beta_in, tools.beta);
        utils::check(nwh_in == nw_half, "vertex_t::cache_w: wbar_load {} has nw_half = {}, this run's = {} (audit A18).",
                     *wl, nwh_in, nw_half);
        nda::array<double, 2> kpts_now(MF->kpts());
        bool kpts_ok = (kpts_in.shape(0) == kpts_now.shape(0) and kpts_in.shape(1) == kpts_now.shape(1));
        double kdev = 0.0;
        if (kpts_ok) {
          for (long i = 0; i < kpts_now.shape(0); ++i)
            for (long d = 0; d < kpts_now.shape(1); ++d) kdev = std::max(kdev, std::abs(kpts_in(i, d) - kpts_now(i, d)));
          kpts_ok = (kdev <= 1e-8);
        }
        utils::check(kpts_ok, "vertex_t::cache_w: wbar_load {} was dumped on a different k-mesh ({} k-points vs {}; max "
                              "|dk| = {:.3e}) (audit A18).", *wl, kpts_in.shape(0), kpts_now.shape(0), kdev);
        if (g.has_dataset("Xb_skma")) {
          nda::array<ComplexType, 4> Xb_in;
          nda::h5_read(g, "Xb_skma", Xb_in);
          bool xb_ok = (Xb_in.shape() == _Xb_skma.shape());
          double xdev = 0.0, xsc = 0.0;
          if (xb_ok) {
            auto const *a = Xb_in.data();
            auto const *b = _Xb_skma.data();
            for (long i = 0; i < long(_Xb_skma.size()); ++i) {
              xdev = std::max(xdev, std::abs(a[i] - b[i]));
              xsc = std::max(xsc, std::abs(b[i]));
            }
            xb_ok = (xdev <= 1e-8 * std::max(xsc, 1e-300));
          }
          utils::check(xb_ok, "vertex_t::cache_w: wbar_load {}: the stored secondary collocation Xb_skma differs from this "
                              "run's (max |dXb| = {:.3e} vs max |Xb| = {:.3e}, or a different shape): the W-bar is expressed in "
                              "another secondary basis (audit A18).", *wl, xdev, xsc);
        } else {
          app_log(1, "  [WARNING] vertex_t::cache_w: wbar_load {} carries no Xb_skma -- the secondary BASIS of the loaded W-bar "
                     "cannot be compared with this run's (window / N_m / beta / k-mesh match).", *wl);
        }
      }
      utils::check(Wl.shape() == Wb.shape(), "vertex_t::cache_w: wbar_load {} has shape ({}, {}, {}, {}) != the cache's.", *wl,
                   Wl.shape(0), Wl.shape(1), Wl.shape(2), Wl.shape(3));
      if (shm_cache) {
        _Wb_shm->win().fence();
        if (mpi->node_comm.root()) Wb() = Wl;
        _Wb_shm->win().fence();
      } else {
        Wb() = Wl;
      }
      // Loud on purpose: this runs on EVERY cache_w call, so the rung of every iteration is the file's: the dynamic rung is
      // FROZEN at the loaded W-bar and does not follow this run's W.
      app_log(1, "  [WARNING] [factorize-vertex] the W-bar cache (the dynamic rung) is REPLACED by the file {} (vertex_debug "
                 "wbar_load; metadata validated). This happens at EVERY cache fill, so the rung is FROZEN at the file's W-bar "
                 "and does not follow the self-consistent W of this run.", *wl);
    }

    // ---- dump the cached rung for offline factorization studies -----------------------------
    // vertex_debug wbar_dump = 1 writes <prefix>.wbar.h5 (rank 0): the cache Wbar_dyn(q, nu >= 0) on the PH-sym half mesh,
    // the secondary collocation Xb(s, k, N, a), the momentum maps, the bosonic grid + transforms, and the analytic q -> 0
    // head separately (Hbar = t H t^dag at Gamma and its tau-weights eps_inv_head, already INCLUDED in Wbar at Gamma).
    // wbar_dump_exit = 1 ends the run right after the dump (skips the readout / Sigma of the iteration).
    if (vertex_debug::flag("wbar_dump")) {
      nda::array<ComplexType, 2> Hbar(_Nm, _Nm);
      Hbar() = ComplexType(0.0);
      if (head_ok) {
        nda::array<ComplexType, 2> tmp(_Nm, Np);
        vertex_secondary_detail::fold_core(_t_qmP(iq_gamma, all, all), H_PQ, tmp, Hbar);
      }
      if (mpi->comm.root()) {
        const std::string fn = mb_state.coqui_prefix + ".wbar.h5";
        h5::file f(fn, 'w');
        h5::group g(f);
        nda::h5_write(g, "Wbar_qwmm", nda::array<ComplexType, 4>(Wb));   // (nq_ibz, nw_half, N_m, N_m)
        nda::h5_write(g, "Xb_skma", _Xb_skma);                            // (ns, nk, N_m, nc)
        nda::h5_write(g, "kmq", kmq);                                     // (nq_ibz, nk): k - q
        nda::h5_write(g, "Qpts", nda::array<double, 2>(MF->Qpts()));
        nda::h5_write(g, "kpts", nda::array<double, 2>(MF->kpts()));
        nda::h5_write(g, "wn_b", tools.wn_b);                             // (nw_b) bosonic Matsubara integers
        nda::h5_write(g, "Ttw_bb", tools.Ttw_bb);                         // (nt, nw_b)
        nda::h5_write(g, "Twt_bb", tools.Twt_bb);                         // (nw_b, nt)
        nda::h5_write(g, "tau", tools.s_phys);                            // (nt) physical tau
        nda::h5_write(g, "Hbar_gamma", Hbar);
        if (mb_state.eps_inv_head.has_value())
          nda::h5_write(g, "eps_inv_head_t", nda::array<ComplexType, 1>(mb_state.eps_inv_head.value()));
        h5::h5_write(g, "beta", tools.beta);
        h5::h5_write(g, "nw_half", nw_half);
        h5::h5_write(g, "iq_gamma", iq_gamma);
        h5::h5_write(g, "head_ok", long(head_ok ? 1 : 0));
        h5::h5_write(g, "window_first", long(_band_window.first()));
        h5::h5_write(g, "window_size", long(_band_window.size()));
        h5::h5_write(g, "nm", _Nm);
        app_log(1, "  [factorize-vertex] Wbar cache dumped to {} ((nq, nw_half, N_m, N_m) = ({}, {}, {}, {}), head {})",
                fn, nqpts_ibz, nw_half, _Nm, _Nm, head_ok);
      }
      mpi->comm.barrier();
      if (vertex_debug::flag("wbar_dump_exit")) {
        // a debug request, never quiet -- the iteration's readout / Sigma and every later step are skipped
        app_log(1, "  [WARNING] [factorize-vertex] exiting after the W-bar dump by request (vertex_debug wbar_dump_exit): "
                   "MPI_Finalize + exit(0) now -- the rest of this iteration (readout, Sigma) and of the run is NOT computed.");
        app_log_flush();
        MPI_Finalize();
        std::_Exit(0);
      }
    }

    // ---- footprint: the cache vs the retained dW it replaces -------------------------
    const double to_mb = 16.0 / (1024.0 * 1024.0);   // complex<double>
    const double cache_mb = double(nqpts_ibz) * double(nw_half) * double(_Nm) * double(_Nm) * to_mb;
    const double dw_mb = double(nqpts_ibz) * double(nt_half) * double(Np) * double(Np) * to_mb;
    app_log(2, "  Refinement 2 W-bar cache FILLED: (nq, nw_half, N_m, N_m) = "
               "({}, {}, {}, {}) = {:.3f} MB (replicated store; fold WORK distributed "
               "over q, M3 item #7)\n"
               "  vs the retained dW it replaces: (nq, nt_half, Np, Np) = "
               "({}, {}, {}, {}) = {:.3f} MB -- ratio {:.3e}\n",
            nqpts_ibz, nw_half, _Nm, _Nm, cache_mb,
            nqpts_ibz, nt_half, Np, Np, dw_mb, cache_mb / dw_mb);
    mpi->comm.barrier();
  }

  long vertex_t::ensure_secondary_basis(MBState &mb_state, THC_ERI auto const &thc) {
    decltype(nda::range::all) all;
    _run_prefix = mb_state.coqui_prefix;   // for the <prefix>.secpts.h5 / .pol_nu0 dumps
    auto mpi = thc.mpi();
    auto MF = thc.MF();
    const long nkpts = MF->nkpts();
    const long nqpts_ibz = MF->nqpts_ibz();
    const long Np = thc.Np();
    const long nbnd = MF->nbnd();
    utils::check(mb_state.sG_tskij.has_value(),
                 "vertex_t::ensure_secondary_basis: sG_tskij is not initialized in MBState.");
    const long ns = mb_state.sG_tskij.value().local().shape(1);
    const bool wan = _wannier;

    // ---- q = Gamma index (crystal coordinates: all components integer mod G) ----------
    long iq_gamma = -1;
    {
      auto Qpts = MF->Qpts();
      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        double d = 0.0;
        for (long i = 0; i < 3; ++i) {
          double x = Qpts(iq, i);
          d += std::abs(x - std::round(x));
        }
        if (d < 1e-8) {
          utils::check(iq_gamma < 0,
                       "vertex_t::ensure_secondary_basis: multiple Gamma q-points found "
                       "({} and {}).", iq_gamma, iq);
          iq_gamma = iq;
        }
      }
      utils::check(iq_gamma >= 0,
                   "vertex_t::ensure_secondary_basis: no Gamma q-point found.");
    }
    if (_secondary_ready or not secondary()) return iq_gamma;

    // ---- collocation + momentum maps for the lazy secondary build (as in cache_w) -----
    auto sX_skPa = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(
        *mpi, std::array<long, 4>{ns, nkpts, Np, nbnd});
    sX_skPa.win().fence();
    if (mpi->node_comm.root())
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nkpts; ++ik)
          sX_skPa.local()(is, ik, all, all) = thc.X(is, 0, ik);
    sX_skPa.win().fence();
    auto X_skPa = sX_skPa.local();
    nda::array<long, 2> kmq(nqpts_ibz, nkpts);
    for (long iq = 0; iq < nqpts_ibz; ++iq)
      for (long ik = 0; ik < nkpts; ++ik) kmq(iq, ik) = MF->qk_to_k2(iq, ik);
    // effective global collocation the secondary basis fits against (WANNIER: X_bar = X.U)
    auto sXbar_glob = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(
        *mpi, std::array<long, 4>{ns, nkpts, Np, wan ? subspace_rank() : 1});
    if (wan) {
      sXbar_glob.win().fence();
      if (mpi->node_comm.root())
        sXbar_glob.local() = vertex_wannier_detail::build_Xbar(X_skPa, _U_skia, _band_window);
      sXbar_glob.win().fence();
    }
    auto X_glob = wan ? sXbar_glob.local() : X_skPa;
    const long orb0_glob = wan ? 0 : _band_window.first();

    build_secondary_basis(thc, X_glob, orb0_glob, kmq, iq_gamma);
    return iq_gamma;
  }

  template<typename dArray_t>
  void vertex_t::build_w0(MBState &mb_state, THC_ERI auto const &thc,
                          dArray_t const &dPi_rpa_tqPQ, bool nu0_partial) {
    vertex_timer_detail::scoped_timer _tm_total(_Timer, "BUILD_W0");
    decltype(nda::range::all) all;
    using Arr3 = nda::array<ComplexType, 3>;
    using Arr4 = nda::array<ComplexType, 4>;
    using Arr2 = nda::array<ComplexType, 2>;
    using math::nda::make_distributed_array;
    utils::check(active(),
                 "vertex_t::build_w0: called while the vertex is inactive. Callers must "
                 "guard with vertex_t::active() (or needs_w0()).");
    utils::check(_ft != nullptr, "vertex_t::build_w0: IAFT instance is required.");

    auto mpi = thc.mpi();
    auto MF = thc.MF();
    const long nkpts = MF->nkpts();
    const long nqpts_ibz = MF->nqpts_ibz();
    const long Np = thc.Np();
    auto gs = dPi_rpa_tqPQ.global_shape();
    const long nt_half_ft = (_ft->nt_b() % 2 == 0) ? _ft->nt_b() / 2 : _ft->nt_b() / 2 + 1;
    utils::check(gs[1] == nqpts_ibz and gs[2] == Np and gs[3] == Np and
                 (nu0_partial ? gs[0] == dPi_rpa_tqPQ.grid()[0] : gs[0] == nt_half_ft),
                 "vertex_t::build_w0: unexpected Pi_RPA global shape ({}, {}, {}, {}); "
                 "expected ({}, {}, {}, {}).",
                 gs[0], gs[1], gs[2], gs[3], nu0_partial ? dPi_rpa_tqPQ.grid()[0] : nt_half_ft, nqpts_ibz, Np, Np);

    // ITERATION-LOCAL lifetime: drop last iteration's objects up front, so no static-rung
    // state can ever be read across an iteration boundary (a stale W0 would silently
    // break the Phi-derivability of the response terms).
    reset_w0();

    app_log(1, "\n  [ISDF-Vertex] static rung W0[G] (increment S2; "
               "notes/static_vertex_implementation_plan.md section 2.2)\n"
               "  W0(q) = Z(q) + dW(q, i.nu = 0) from the SAME-ITERATION RPA "
               "polarizability -- no lag,\n"
               "  no Pi^C content (decision D2). Grid (nt_half, nq, Np) = ({}, {}, {}), "
               "rung mode = {}.",
            nt_half_ft, nqpts_ibz, Np, rung_str());

    // Stage timers PARTITION BUILD_W0 (see the SIG_* note in eval_Sigma_C).
    _Timer.start("W0_LAYOUT");
    // ---- (P,Q)-block layout: q unsplit, (P,Q) over ALL ranks (thc.dZ({1,nP,nQ}) ------
    // layout; the one fold_Z_distributed and the slate 2D ops both accept, and one that
    // keeps the nq*Np^2 object distributed).
    const long np_ranks = mpi->comm.size();
    std::array<long, 3> w0_pgrid = {1, 1, 1};
    w0_pgrid[1] = utils::find_proc_grid_min_diff(np_ranks, Np, Np);
    w0_pgrid[2] = np_ranks / w0_pgrid[1];
    // BLOCK SIZE: mirror scr_coulomb_t::W_omega_proc_grid (scr_coulomb_t.h) -- a
    // SQUARE block of min(1024, Np/nP, Np/nQ) on the (P,Q) axes.
    //
    // With 1x1 SLATE tiles, slate_ops::multiply / ::inverse below would pay per-tile
    // overhead Np^2 times for a small O(Np^3) solve, which dominates build_w0. The
    // reference path (scr_coulomb_t::dyson_W_in_place, which this code deliberately
    // mirrors) sizes its blocks the same way.
    //
    // max(1, ...) guards Np < grid (tiny/toy meshes), where Np/grid truncates to 0 and a
    // zero block size is invalid.
    std::array<long, 3> w0_bsize = {1, 1, 1};
    w0_bsize[1] = std::min({static_cast<long>(1024),
                            std::max(1l, Np / w0_pgrid[1]),
                            std::max(1l, Np / w0_pgrid[2])});
    w0_bsize[2] = w0_bsize[1];
    // The block size changes only the SLATE tiling, not the algebra.
    const std::array<long, 4> f0_pgrid = {1, 1, w0_pgrid[1], w0_pgrid[2]};
    const std::array<long, 4> f0_bsize = {1, 1, w0_bsize[1], w0_bsize[2]};

    _Timer.stop("W0_LAYOUT");
    _Timer.start("W0_PI0_ROW");
    // ---- step 1: the i.nu = 0 row of Pi_RPA on that layout ---------------------------
    auto dW0_1qPQ = make_distributed_array<Arr4>(
        mpi->comm, f0_pgrid, {1, nqpts_ibz, Np, Np}, f0_bsize);
    if (nu0_partial) {        // the caller formed step (a) (on the device): finish the row from its partial
      vertex_w0_detail::finish_nu0_row(const_cast<dArray_t &>(dPi_rpa_tqPQ), dW0_1qPQ);
    } else {
      auto R_t = vertex_w0_detail::nu0_transform_row(*_ft);
      vertex_w0_detail::extract_nu0_row(dPi_rpa_tqPQ, R_t, dW0_1qPQ);
    }

    _Timer.stop("W0_PI0_ROW");
    _Timer.start("W0_DYSON");
    // ---- step 2: the SINGLE-FREQUENCY THC Dyson, per q ------------------------------
    // dW0(q) = ([I - Z(q).Pi0(q)]^{-1} - I) Z(q): the scr_coulomb_t::dyson_W_in_place
    // algebra (scr_coulomb_t.cpp) with the frequency loop removed. Same slate
    // primitives, same operand order, same in-place convention (the array that came in
    // holding Pi0 goes out holding dW0), so the plain-GW self-slice identity
    // W0(q) == W(q, i.nu = 0) holds to machine precision.
    auto dZ = thc.dZ(w0_pgrid, w0_bsize);
    auto P_rng = dW0_1qPQ.local_range(2);
    auto Q_rng = dW0_1qPQ.local_range(3);
    utils::check(dZ.local_range(1) == P_rng and dZ.local_range(2) == Q_rng,
                 "vertex_t::build_w0: Z and Pi0 do not share the (P,Q) block partition.");
    // Under CUDA the per-q Dysons run on the device -- q split over the ranks (one redistribute of
    // the Pi0 row there and of dW0 back, nq Np^2 each), M = I - Z Pi0 by one gemm, getrf + getrs against the identity
    // for eps^{-1} (its max-abs is the conditioning meter), dW0 = (eps^{-1} - I) Z by one gemm. The host path's SLATE
    // multiply / inverse / multiply on the (P,Q) grid computes the same quantities in another rounding.
    // vertex_debug w0_dyson_device = 0 keeps SLATE.
#if defined(ENABLE_CUDA)
    const bool dyson_dev = (nqpts_ibz >= np_ranks) and
                           vertex_debug::number("w0_dyson_device", 1.0) != 0.0;   // vertex_debug: w0_dyson_device
#else
    const bool dyson_dev = false;
#endif
    if (dyson_dev) {
#if defined(ENABLE_CUDA)
      namespace dcu = methods::solvers::dynbse_cuda;
      auto dq = make_distributed_array<Arr4>(mpi->comm, {1, np_ranks, 1, 1}, {1, nqpts_ibz, Np, Np}, {1, 1, 1, 1});
      math::nda::redistribute(dW0_1qPQ, dq);
      auto dZq = thc.template dZ<DEVICE_MEMORY>({np_ranks, 1, 1}, {1, 1, 1});
      utils::check(dZq.local_range(0) == dq.local_range(1),
                   "vertex_t::build_w0: the q-split Z and Pi0 do not share the q partition.");
      auto qloc = dq.local();
      const long q0 = dq.origin()[1], nql = dq.local_shape()[1];
      nda::array<ComplexType, 2> Ih(Np, Np);
      nda::matrix<ComplexType, nda::F_layout> IhF(Np, Np);
      Ih() = ComplexType(0.0); IhF() = ComplexType(0.0);
      for (long P = 0; P < Np; ++P) { Ih(P, P) = ComplexType(1.0); IhF(P, P) = ComplexType(1.0); }
      auto Id = memory::to_memory_space<DEVICE_MEMORY>(Ih);
      memory::array<DEVICE_MEMORY, ComplexType, 2> Md(Np, Np), Od(Np, Np);
      memory::array<DEVICE_MEMORY, int, 1> ipd(Np);
      double epsinv_max = 0.0;
      long epsinv_q = -1;
      for (long iql = 0; iql < nql; ++iql) {
        auto Zd = dZq.local()(iql, all, all);
        auto Pd = memory::to_memory_space<DEVICE_MEMORY>(Arr2(qloc(0, iql, all, all)));
        Md() = Id;
        nda::blas::gemm(ComplexType(-1.0), Zd, Pd, ComplexType(1.0), Md);   // M = I - Z.Pi0
        int info = nda::lapack::getrf(Md, ipd);
        utils::check(info == 0, "vertex_t::build_w0: device getrf of I - Z.Pi0 failed at q = {} (info {}).", q0 + iql, info);
        auto Xd = memory::to_memory_space<DEVICE_MEMORY>(IhF);
        info = nda::lapack::getrs(Md, Xd, ipd);                            // X = eps^{-1}
        utils::check(info == 0, "vertex_t::build_w0: device getrs failed at q = {} (info {}).", q0 + iql, info);
        const double m = dcu::dev_maxabs(Xd.data(), Np * Np);
        if (m > epsinv_max) { epsinv_max = m; epsinv_q = q0 + iql; }
        dcu::dev_add_diag(Xd.data(), Np, Np, -1.0);                        // eps^{-1} - I
        nda::blas::gemm(Xd, Zd, Od);                                       // dW0 = (eps^{-1} - I) Z
        qloc(0, iql, all, all) = Arr2(memory::to_memory_space<HOST_MEMORY>(Od));
      }
      math::nda::redistribute(dq, dW0_1qPQ);
      double gmax = mpi->comm.all_reduce_value(epsinv_max, boost::mpi3::max<>{});
      long q_of_max = (epsinv_max == gmax) ? epsinv_q : -1;
      q_of_max = mpi->comm.all_reduce_value(q_of_max, boost::mpi3::max<>{});
      app_log(2, "  [ISDF-Vertex] W0 conditioning: max_q ||[I - Z.Pi_RPA(i.nu=0)]^-1||_max "
                 "= {:.4e} (worst q = {}; the single-frequency Dysons on the device)", gmax, q_of_max);
#endif
    } else {
      auto pgrid2 = std::array<long, 2>{w0_pgrid[1], w0_pgrid[2]};
      auto bsize2 = std::array<long, 2>{w0_bsize[1], w0_bsize[2]};
      auto dPi_PQ = make_distributed_array<Arr2>(mpi->comm, pgrid2, {Np, Np}, bsize2, true);
      auto dZ_PQ = make_distributed_array<Arr2>(mpi->comm, pgrid2, {Np, Np}, bsize2, true);
      auto dA_PQ = make_distributed_array<Arr2>(mpi->comm, pgrid2, {Np, Np}, bsize2, true);
      utils::check(dPi_PQ.local_range(0) == P_rng and dPi_PQ.local_range(1) == Q_rng,
                   "vertex_t::build_w0: the 2D solve grid does not match the (P,Q) blocks.");
      auto Pi_PQ = dPi_PQ.local();
      auto Z_PQ = dZ_PQ.local();
      auto A_PQ = dA_PQ.local();
      // diagonal entries owned by this rank (the "-I" of I - Z.Pi and of eps^{-1} - I)
      std::vector<std::pair<long, long> > diag_idx;
      for (long iP = 0; iP < Pi_PQ.shape(0); ++iP)
        for (long iQ = 0; iQ < Pi_PQ.shape(1); ++iQ)
          if (P_rng.first() + iP == Q_rng.first() + iQ) diag_idx.push_back({iP, iQ});

      double epsinv_max = 0.0;
      long epsinv_q = -1;
      auto W0_loc = dW0_1qPQ.local();
      for (long iq = 0; iq < nqpts_ibz; ++iq) {     // q is NOT split: every rank loops all q
        Z_PQ = dZ.local()(iq, all, all);
        Pi_PQ = W0_loc(0, iq, all, all);
        math::nda::slate_ops::multiply(dZ_PQ, dPi_PQ, dA_PQ);           // A = Z.Pi0
        for (auto idx : diag_idx) A_PQ(idx.first, idx.second) -= ComplexType(1.0);
        A_PQ *= -1.0;                                                   // A = I - Z.Pi0
        math::nda::slate_ops::inverse(dA_PQ);                           // A = eps^{-1}
        for (auto const &v : A_PQ)
          if (std::abs(v) > epsinv_max) { epsinv_max = std::abs(v); epsinv_q = iq; }
        for (auto idx : diag_idx) A_PQ(idx.first, idx.second) -= ComplexType(1.0);
        math::nda::slate_ops::multiply(dA_PQ, dZ_PQ, dPi_PQ);           // dW0 = (eps^-1 - I) Z
        W0_loc(0, iq, all, all) = Pi_PQ;
      }
      {
        double gmax = mpi->comm.all_reduce_value(epsinv_max, boost::mpi3::max<>{});
        long q_of_max = (epsinv_max == gmax) ? epsinv_q : -1;
        q_of_max = mpi->comm.all_reduce_value(q_of_max, boost::mpi3::max<>{});
        app_log(2, "  [ISDF-Vertex] W0 conditioning: max_q ||[I - Z.Pi_RPA(i.nu=0)]^-1||_max "
                   "= {:.4e} (worst q = {})", gmax, q_of_max);
      }
    }   // SLATE path

    _Timer.stop("W0_DYSON");
    _Timer.start("W0_HEAD");
    // ---- step 3: the q->0 head policy AT i.nu = 0 -------------------------------------
    // ONE policy, ONE W0, so every later appearance of the rung carries the same head.
    // "v1_skip"/"ignore_g0" store the regularized body only; the
    // gygi class additionally inserts the analytic rank-1 head, whose i.nu = 0 dynamic
    // weight Re[eps^{-1}_head(i.nu=0)] is extracted from THIS RPA dW0 -- so the rung and
    // its head factor belong to the same iteration by construction,
    // instead of the previous iteration's mb_state.eps_inv_head that is still standing
    // at this point of update_w.
    // the Gamma index + (in the secondary path) the lazy secondary transfer maps. Built
    // HERE, not at the first kernel call: update_w runs before any kernel, so this is
    // the earliest point the fold below can rely on _t_qmP existing.
    const long iq_gamma = ensure_secondary_basis(mb_state, thc);

    bool head_insertion = (_div_treatment.find("gygi") != std::string::npos);
    if (head_insertion and nqpts_ibz == 1) {
      app_log(1, "  [WARNING] W0: nqpts_ibz == 1 with a gygi-class vertex div_treatment -- "
                 "taking \"ignore_g0\"\n"
                 "            instead (same downgrade as eval_Pi_C / cache_w).");
      head_insertion = false;
    }
    nda::array<ComplexType, 1> chi_g;
    if (head_insertion) {
      // SAME skip logic as vertex_head_detail::build_head_rank1 (madelung == 0 or an
      // all-zero chi(Gamma, :) => no head), rank-1 form, no dense Np^2 head.
      // THE head-strength lambda MUST BE APPLIED HERE TOO. W0 is the object B-L expands
      // AROUND (dW = W - W0), so scaling the head in the rungs but not in W0 would not
      // weaken the head -- it would create a W0-vs-W head-weight MISMATCH, which is a
      // different effect. One lambda, every site.
      const double xi = MF->madelung() * _bl_head_scale;
      auto chi = thc.basis_head();                              // (nqpts_ibz, Np)
      utils::check(chi.shape(0) > iq_gamma and chi.shape(1) == Np,
                   "vertex_t::build_w0: basis_head shape mismatch (({}, {}) vs iq_gamma = "
                   "{}, Np = {}).", chi.shape(0), chi.shape(1), iq_gamma, Np);
      double chi_max = 0.0;
      for (long P = 0; P < Np; ++P) chi_max = std::max(chi_max, std::abs(chi(iq_gamma, P)));
      if (xi != 0.0 and chi_max != 0.0) {
        chi_g = nda::array<ComplexType, 1>(chi(iq_gamma, all));
        // The ladder kernel's q -> 0 head scale. The default _ladder_head_scale = 1.0
        // multiplies by EXACTLY 1.0 (IEEE), so the default path is unchanged bitwise. This is
        // the ONLY site the knob acts: one W0, one policy, and W0 is the ladder rung W-bar_0.
        _w0_head_c = ComplexType(double(nkpts) * xi * _ladder_head_scale);
        _w0_head_applied = true;
        if (_ladder_head_scale != 1.0)
          app_log(1, "  [DA D-4] ladder_head_scale = {:.6g}: the analytic rank-1 q -> 0 "
                     "head of the static rung W0(Gamma) -- i.e. the head INSIDE the ladder "
                     "kernel W-bar_0 -- is scaled by this factor. The loop's own RPA W and "
                     "its div_treatment are untouched.", _ladder_head_scale);
      } else if (_bl_head_scale == 0.0) {
        app_log(1, "  [W-int-3] W0: vertex_bl_head_scale = 0 -- the static rung W-bar_0 carries NO analytic q -> 0 head "
                   "(body-only vertex kernel; the loop's RPA W keeps its {} head).", _div_treatment);
      } else if (head_unusable_continue("vertex_t::build_w0")) {   // aborts unless allowed (scale 0 handled above)
        app_log(1, "  [WARNING] W0: gygi head insertion requested but the head data are "
                   "unusable\n"
                   "            (madelung == 0 or empty basis_head) -- proceeding WITHOUT "
                   "the analytic head\n"
                   "            (equivalent to policy \"ignore_g0\").");
      }
      if (_w0_head_applied) {
        // the DYNAMIC head weight at i.nu = 0, from the freshly built RPA dW0 (the GW head
        // machinery evaluated at one frequency: eps_inv_head_w takes a (nw, nq, Np, Np)
        // distributed array, and ours has nw == 1).
        auto [eps_inv_w, eps_inv_q0_w] =
            div_utils::eps_inv_head_w(dW0_1qPQ, thc, *MF, _div_treatment);
        (void)eps_inv_w;
        _w0_eps_head = eps_inv_q0_w(0).real();
        app_log(1, "  [ISDF-Vertex] W0 head insertion at i.nu = 0: madelung = {}, "
                   "Nk*madelung = {:.6e},\n"
                   "  Re[eps^-1_head(i.nu=0) - 1] = {:.6e}  =>  epsilon_inf(RPA, W0) = "
                   "{:.6f}",
                xi, _w0_head_c.real(), _w0_eps_head, 1.0 / (1.0 + _w0_eps_head));

        // ---- DIAGNOSTIC: take W0's Gamma head weight from the SAME eps^-1 that W uses --
        // WHY THIS KNOB EXISTS. B-L expands in dW = W - W0, and W0 is the RPA-STATIC
        // screen by definition, while the run's own W carries P^{C,L}. That is a deliberate
        // choice and the residue is nominally O(vertex^2). But the head weights
        // Re[eps^-1 - 1] of W0 (RPA only) and of W (vertex-corrected) can differ by several
        // percent, and since the head is rank-1 that offset can make up most of
        // |W(q,0) - W0(q)| at Gamma while being nearly invisible in max-norm (a max-norm
        // cannot see a coherent rank-1 channel). The W0 . Pi . W0 sandwich then amplifies
        // exactly that channel by c^2 ||chi||^4.
        //
        // Turning this on makes the head part of dW(Gamma, i.nu = 0) vanish (up to the
        // one-iteration lag below), leaving W0's BODY RPA-static as the theory specifies.
        // It isolates whether the head-channel residue is what drives B-L's instability.
        //
        // NOT THE DEFAULT, and note the iteration lag it implies: build_w0 runs immediately
        // after Pi_RPA and BEFORE Pi^C is added (scr_coulomb_t.cpp), so
        // mb_state.eps_inv_head is still the PREVIOUS iteration's vertex-corrected head.
        // That one-iteration lag is the very thing _w0_eps_head avoids; it vanishes at the
        // fixed point, so it is acceptable for a diagnostic and would need thought before
        // ever becoming a default.
        if (_bl_w0_head_from_w) {
          if (mb_state.eps_inv_head.has_value()) {
            auto &eih = mb_state.eps_inv_head.value();
            const long nw_half =
                (_ft->nw_b() % 2 == 0) ? _ft->nw_b() / 2 : _ft->nw_b() / 2 + 1;
            nda::array<ComplexType, 2> eih_w(nw_half, 1);
            auto eih_t = nda::reshape(eih, std::array<long, 2>{eih.shape(0), 1});
            _ft->tau_to_w_PHsym(eih_t, eih_w);   // i.nu = 0 is index 0 of the PH-sym half
            const double from_w = eih_w(0, 0).real();
            app_log(1, "  [W0HEADFROMW] W0's Gamma head weight OVERRIDDEN: RPA-static "
                       "{:.6e} -> W's own {:.6e} (difference {:.6e}, {:.2f} % of the "
                       "RPA-static value). DIAGNOSTIC: makes the head part of "
                       "W(Gamma,0) - W0(Gamma) vanish up to a one-iteration lag; W0's "
                       "BODY stays RPA-static.",
                    _w0_eps_head, from_w, from_w - _w0_eps_head,
                    (_w0_eps_head != 0.0 ? 100.0 * (from_w - _w0_eps_head) / _w0_eps_head
                                         : 0.0));
            _w0_eps_head = from_w;
          } else {
            app_log(1, "  [WARNING] vertex_bl_w0_head_from_w is set but MBState carries no "
                       "eps_inv_head yet\n"
                       "            (first update of a cold run) -- keeping the RPA-static "
                       "head weight this iteration.");
          }
        }
      }
    }
    if (not head_insertion)
      app_log(1, "  [ISDF-Vertex] W0 q->0 policy: {} -- W0(Gamma) is the stored regularized "
                 "body\n"
                 "  Z(Gamma) + dW0(Gamma) (v(G=0) zeroed at ERI build); no analytic head{}.",
              _div_treatment,
              w0_skip_gamma() ? ", and the Gamma cell of the rung transfer will be "
                                "DROPPED by the S3+ kernels (v1_skip fallback)"
                              : " (GW ignore_g0 analogue)");

    _Timer.stop("W0_HEAD");
    _Timer.start("W0_ASSEMBLE");
    // ---- W0 = Z + dW0 (+ head at Gamma), (P,Q)-block-distributed ---------------------
    _W0_qPQ.emplace(make_distributed_array<Arr3>(
        mpi->comm, w0_pgrid, {nqpts_ibz, Np, Np}, w0_bsize));
    {
      auto W0 = _W0_qPQ.value().local();
      auto dW0 = dW0_1qPQ.local();
      auto Zl = dZ.local();
      for (long iq = 0; iq < nqpts_ibz; ++iq)
        for (long ip = 0; ip < W0.shape(1); ++ip)
          for (long jq = 0; jq < W0.shape(2); ++jq)
            W0(iq, ip, jq) = Zl(iq, ip, jq) + dW0(0, iq, ip, jq);
      if (_w0_head_applied) {
        // The head in its two pieces, applied in the same order and with the same
        // per-element arithmetic as the Z / dW augmentations: bare weight 1 into
        // the Z part, dynamic weight Re[eps^{-1}_head(i.nu=0)] into the dW0 part.
        vertex_secondary_detail::head_block_add(chi_g, _w0_head_c, ComplexType(1.0),
                                                P_rng, Q_rng, W0(iq_gamma, all, all));
        vertex_secondary_detail::head_block_add(chi_g, _w0_head_c,
                                                ComplexType(_w0_eps_head),
                                                P_rng, Q_rng, W0(iq_gamma, all, all));
      }
    }
    dW0_1qPQ.reset();

    _Timer.stop("W0_ASSEMBLE");
    _Timer.start("W0_FOLD");
    // ---- W0bar = t W0 t^dag: the DISTRIBUTED one-row fold ---------------------------
    // fold_Z_distributed IS the one-row variant of fold_dW_distributed (Z has no tau
    // axis => no t-pool, no tau->nu, no PH-unfold), and W0 has exactly Z's shape and
    // layout, so the fold "restricted to the single i.nu = 0 row" is this call verbatim.
    // head_at_gamma = false: the head is ALREADY inside W0 (one W0, one policy), so
    // re-adding it here would double count.
    auto no_head = [](nda::MemoryArrayOfRank<2> auto&&, nda::range const&,
                      nda::range const&) {};
    if (secondary()) {
      _W0b_qmm.emplace(Arr3(nqpts_ibz, _Nm, _Nm));
      vertex_secondary_detail::fold_Z_distributed(
          _W0_qPQ.value(), _t_qmP, nqpts_ibz, Np, _Nm, iq_gamma, false, no_head,
          _W0b_qmm.value(), mpi->comm);
      {
        // DIAGNOSTIC: the Hermiticity and the imaginary content of the folded static rung per IBZ q -- the
        // time-reversed transfers read it PQ-transposed (= conj for a Hermitian core); a non-Hermitian core breaks that identity
        auto const &Wb = _W0b_qmm.value();
        double herm_max = 0.0, im_max = 0.0;
        for (long iq = 0; iq < nqpts_ibz; ++iq) {
          double dh = 0.0, nn = 0.0, im = 0.0;
          for (long P = 0; P < _Nm; ++P)
            for (long Q = 0; Q < _Nm; ++Q) { dh += std::norm(Wb(iq, P, Q) - std::conj(Wb(iq, Q, P))); nn += std::norm(Wb(iq, P, Q)); im += Wb(iq, P, Q).imag() * Wb(iq, P, Q).imag(); }
          herm_max = std::max(herm_max, std::sqrt(dh / std::max(nn, 1e-300)));
          im_max = std::max(im_max, std::sqrt(im / std::max(nn, 1e-300)));
        }
        app_log(1, "  [W0 fold] W0bar per IBZ q: max ||W - W^dag||_F/||W||_F = {:.3e}, max ||Im W||_F/||W||_F = {:.3e}", herm_max, im_max);
      }
    } else {
      // GLOBAL-aux reference path (small scale only): the "secondary"
      // rung IS the global one, N_m == Np, t = identity. Gather the distributed blocks
      // (zero-pad + all_reduce over a PARTITION = an exact gather, no reassociation) --
      // the same replication class this path already accepts for its Z_qPQ.
      _W0b_qmm.emplace(Arr3(nqpts_ibz, Np, Np));
      auto &Wb = _W0b_qmm.value();
      Wb() = ComplexType(0.0);
      Wb(_W0_qPQ.value().local_range(0), P_rng, Q_rng) = _W0_qPQ.value().local();
      mpi->comm.all_reduce_in_place_n(Wb.data(), Wb.size(), std::plus<>{});
    }

    // ---- the PRE- vs POST-FOLD head meter ----------------------------------------------
    // A secondary basis optimized for subspace pair densities may represent the q -> 0 head
    // poorly; this meter quantifies it.
    // PURE OBSERVER -- nothing below is written back into W0 or W0bar.
    //
    // The inserted head is EXACTLY rank-1, H = c_eff chi chi^dag with
    //   c_eff = _w0_head_c * (1 + _w0_eps_head)     (the two head_block_add weights),
    // so ||H||_F = |c_eff| ||chi||^2 pre-fold and ||t H t^dag||_F = |c_eff| ||t chi||^2
    // post-fold -- no refold needed. What the ladder kernel actually sees is the head's
    // weight RELATIVE to the body it rides on, so the reported meter is each one's share of
    // its own rung's Frobenius norm and the RATIO of the two shares: attenuation < 1 means
    // the secondary basis represents the head worse than it represents the body.
    if (_ladder_qnu_meter and _w0_head_applied and secondary() and chi_g.size() > 0) {
      const ComplexType c_eff = _w0_head_c * ComplexType(1.0 + _w0_eps_head);
      double chi2 = 0.0;
      for (long P = 0; P < Np; ++P) chi2 += std::norm(chi_g(P));
      // t(Gamma) . chi   (N_m); _t_qmP is replicated (nq, N_m, Np)
      double tchi2 = 0.0;
      if (_t_qmP.shape(0) > iq_gamma and _t_qmP.shape(2) == Np) {
        for (long m = 0; m < _t_qmP.shape(1); ++m) {
          ComplexType acc(0.0);
          for (long P = 0; P < Np; ++P) acc += _t_qmP(iq_gamma, m, P) * chi_g(P);
          tchi2 += std::norm(acc);
        }
      }
      // ||W0(Gamma)||_F over the (P,Q) block distribution, and ||W0bar(Gamma)||_F
      double w0g2_loc = 0.0;
      {
        auto W0 = _W0_qPQ.value().local();
        for (long ip = 0; ip < W0.shape(1); ++ip)
          for (long jq = 0; jq < W0.shape(2); ++jq)
            w0g2_loc += std::norm(W0(iq_gamma, ip, jq));
      }
      const double w0g2 = mpi->comm.all_reduce_value(w0g2_loc, std::plus<>{});
      double wbg2 = 0.0;
      {
        auto const &Wb = _W0b_qmm.value();
        for (long m = 0; m < Wb.shape(1); ++m)
          for (long n = 0; n < Wb.shape(2); ++n) wbg2 += std::norm(Wb(iq_gamma, m, n));
      }
      const double h_pre = std::abs(c_eff) * chi2;
      const double h_post = std::abs(c_eff) * tchi2;
      const double n_pre = std::sqrt(std::max(w0g2, 0.0));
      const double n_post = std::sqrt(std::max(wbg2, 0.0));
      _w0_head_share_pre = h_pre / std::max(n_pre, 1e-300);
      _w0_head_share_post = h_post / std::max(n_post, 1e-300);
      _w0_head_atten = _w0_head_share_post / std::max(_w0_head_share_pre, 1e-300);
      app_log(1, "  [DA D-7 head] pre/post-fold head meter at Gamma (H1b): "
                 "|c_eff| = {:.6e}; ||chi||^2 = {:.6e} -> ||t.chi||^2 = {:.6e} "
                 "(kept {:.4f});\n"
                 "    ||H||_F = {:.6e} of ||W0(G)||_F = {:.6e}  => head share PRE  = "
                 "{:.6e}\n"
                 "    ||H_bar||_F = {:.6e} of ||W0bar(G)||_F = {:.6e}  => head share POST = "
                 "{:.6e}\n"
                 "    ATTENUATION (post share / pre share) = {:.6f}   [< 1 = the fold "
                 "represents the head worse than the body = H1b]",
              std::abs(c_eff), chi2, tchi2, (chi2 > 0.0 ? tchi2 / chi2 : 0.0),
              h_pre, n_pre, _w0_head_share_pre, h_post, n_post, _w0_head_share_post,
              _w0_head_atten);
    }

    {
      double w0_max = 0.0, wb_max = 0.0;
      for (auto const &v : _W0_qPQ.value().local()) w0_max = std::max(w0_max, std::abs(v));
      w0_max = mpi->comm.all_reduce_value(w0_max, boost::mpi3::max<>{});
      for (auto const &v : _W0b_qmm.value()) wb_max = std::max(wb_max, std::abs(v));
      const long Nm_eff = _W0b_qmm.value().shape(1);
      const double to_mb = 16.0 / (1024.0 * 1024.0);
      app_log(1, "  [ISDF-Vertex] W0 BUILT: max|W0| = {:.4e} (distributed {} x {} x {}, "
                 "{:.3f} MB total),\n"
                 "  max|W0bar| = {:.4e} (replicated {} x {} x {}, {:.3f} MB/rank); "
                 "iteration-local -- both\n"
                 "  are dropped at the next build (plan section 2.3).\n",
              w0_max, nqpts_ibz, Np, Np,
              double(nqpts_ibz) * double(Np) * double(Np) * to_mb,
              wb_max, nqpts_ibz, Nm_eff, Nm_eff,
              double(nqpts_ibz) * double(Nm_eff) * double(Nm_eff) * to_mb);
      utils::check(std::isfinite(w0_max) and std::isfinite(wb_max),
                   "vertex_t::build_w0: the static rung contains NaN/Inf -- aborting.");
    }
    _Timer.stop("W0_FOLD");
    // See SIG_BARRIER: skew charged to its own slot, not to the fold.
    _Timer.start("W0_BARRIER");
    mpi->comm.barrier();
    _Timer.stop("W0_BARRIER");
    // the stage walls (the head stage includes the one-time secondary-basis build)
    app_log(1, "  [build_w0 wall, cumulative] layout {:.1f} s, Pi0 row {:.1f} s, Dysons {:.1f} s, head (+ the lazy secondary basis) "
               "{:.1f} s, assemble {:.1f} s, fold {:.1f} s, barrier {:.1f} s", _Timer.elapsed("W0_LAYOUT"), _Timer.elapsed("W0_PI0_ROW"),
            _Timer.elapsed("W0_DYSON"), _Timer.elapsed("W0_HEAD"), _Timer.elapsed("W0_ASSEMBLE"), _Timer.elapsed("W0_FOLD"),
            _Timer.elapsed("W0_BARRIER"));
  }

  // template instantiations
  template void vertex_t::eval_Sigma_C(MBState&, const thc_reader_t&);
  template void vertex_t::cache_w(MBState&, const thc_reader_t&);
  template long vertex_t::ensure_secondary_basis(MBState&, const thc_reader_t&);
  template void vertex_t::build_w0(
      MBState&, const thc_reader_t&,
      memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator> const&, bool);

  template memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>
  vertex_t::eval_Pi_C(MBState&, const thc_reader_t&,
                      std::array<long, 4>, std::array<long, 4>, std::array<long, 4>);

  // ladder polarization (vertex_ladder.icc)
  template nda::array<ComplexType, 4> vertex_t::eval_pol_pi0(MBState&, thc_reader_t&);
  template vertex_t::ladder_l1_diag vertex_t::ladder_l1_gates(MBState&, thc_reader_t&);
  template nda::array<ComplexType, 3> vertex_t::eval_pol_ladder_nu0(MBState&, thc_reader_t&,
                                                                      nda::array<ComplexType, 3>*);
  template vertex_t::ladder_p4_diag vertex_t::ladder_p4_gates(MBState&, thc_reader_t&);
  template vertex_t::ladder_sym_diag vertex_t::ladder_sym_gate(MBState&, thc_reader_t&);
  template vertex_t::ladder_p3_diag vertex_t::ladder_p3_gate(MBState&, thc_reader_t&);

  // ladder polarization on the half bosonic mesh (vertex_ladder.icc)
  template nda::array<ComplexType, 4>
  vertex_t::eval_pol_ladder_whalf(MBState&, thc_reader_t&, nda::array<double, 1>*,
                                  nda::array<ComplexType, 4>*);
  // Ward-identity legs check (vertex_ladder.icc)
  template vertex_t::ward_legs_diag vertex_t::ward_legs_gate(MBState&, thc_reader_t&);
  // full-frequency dynamic-rung BSE (vertex_dynbse.icc) and the pair-resolved Sigma vertex (vertex_sigma_pair.icc)
  template vertex_t::dynbse_diag vertex_t::dynbse_gate(MBState&, thc_reader_t&, bool);
  template vertex_t::dynbse_nu0_result vertex_t::eval_pol_dynbse_nu0(MBState&, thc_reader_t&, long);
  template void vertex_t::eval_sigma_pair(MBState&, thc_reader_t&, vertex_t::sigma_pair_opts const&,
                                          nda::array<ComplexType, 5>&, vertex_t::sigma_pair_meter*);
  template void vertex_t::eval_sigma_pair_dyn(MBState&, thc_reader_t&, vertex_t::sigma_pair_opts const&,
                                          nda::array<ComplexType, 5>&, vertex_t::sigma_pair_meter*);
  template long vertex_t::arm_shared_sigma_hook(MBState&, thc_reader_t&, vertex_t::sigma_pair_opts const&, std::vector<long> const&);
  template vertex_t::dynbse_cut_result vertex_t::eval_pol_dynbse_cut(MBState&, thc_reader_t&, std::vector<long> const&,
                                                                     std::vector<long> const&, long, bool);
  template vertex_t::ladder_whalf_diag
  vertex_t::ladder_whalf_gate(MBState&, thc_reader_t&, double);

  // orbital-local ladder polarization (vertex_ladder.icc)
  template nda::array<ComplexType, 4>
  vertex_t::eval_pol_ladder_loc_whalf(MBState&, thc_reader_t&,
                                      nda::array<ComplexType, 4> const&,
                                      nda::array<ComplexType, 4>*);
  template vertex_t::ladder_loc_diag
  vertex_t::ladder_loc_gate(MBState&, thc_reader_t&, nda::array<ComplexType, 4> const&);

}  // solvers
}  // methods
