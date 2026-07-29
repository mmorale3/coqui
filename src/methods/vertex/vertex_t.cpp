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
#include <unordered_set>

#include "nda/lapack.hpp"
#include "nda/linalg/eigenelements.hpp"

#include "utilities/check.hpp"
#include "utilities/proc_grid_partition.hpp"  // Impl 2b: {1,nP,nQ} grid for distributed Z fold
#include "numerics/sparse/csr_blas.hpp"   // csrmm for the symmetry D-matrix blocks
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/ERI/thc.h"    // Refinement 2: restricted-range ISDF point selection
#include "methods/GW/g0_div_utils.hpp"  // S2: eps_inv_head_w at i.nu = 0 (the v2 head machinery)
#include "vertex_t.h"
#include "vertex_secondary_fold.hpp"  // Impl 2: distributed downfold of dW (no full Np^2 gather)
#include "vertex_pi.icc"
#include "vertex_sigma.icc"  // ISDF-Vertex Phase 1c: fused G^3 W^2 Sigma^C kernel
#include "vertex_sigma_r.icc" // INCREMENT S5: the static-vertex response cut Sigma^{C,r}

namespace methods {
namespace solvers {

  namespace vertex_head_detail {

    /**
     * The analytic q->0 head of the rung at the Gamma cell, in the STORED-ARRAY
     * convention (notes/q0_head_treatment.md, (H1) and section 1.5):
     *
     *   H_PQ = N_k * madelung * conj(chi_P(Gamma)) * chi_Q(Gamma)
     *
     * with chi = thc.basis_head() (the G = 0 plane-wave components of the aux basis)
     * and madelung = MF->madelung() (the Gygi-Baldereschi / probe-charge-Ewald
     * constant, PRB 80, 085114 (2009)). Pinned against existing code: consuming
     * H_PQ through the GW Hadamard reproduces Sigma_div_correction
     * (thc_gw.icc:444-525) exactly (dynamic piece, weight Re[eps_inv_head(tau)]),
     * and through the exchange reproduces HF_K_correction (hf_t.cpp:64-100)
     * (bare piece, weight 1).
     *
     * Returns false (H untouched) when the head data are unusable: madelung == 0
     * (model systems) or chi_head not populated (some ERI read paths,
     * thc_reader_t.hpp:380-381). The caller logs and proceeds without insertion.
     */
    template<THC_ERI thc_t>
    bool build_head_rank1(thc_t const& thc, long iq_gamma, long nkpts,
                          nda::array<ComplexType, 2>& H_PQ) {
      auto MF = thc.MF();
      const double xi = MF->madelung();
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
          H_PQ(P, Q) = double(nkpts) * xi * std::conj(chi(iq_gamma, P)) * chi(iq_gamma, Q);
      return true;
    }

  } // vertex_head_detail

  /**
   * W-redistribution helper (vertex parallelization M1, change-list item #1;
   * notes/vertex_parallelization_M1.md, notes/vertex_parallelization_analysis.md
   * section 1.7 / Appendix A).
   *
   * The dynamic screened interaction dW lives on the RPA (t,q,P,Q) proc grid
   * (MBState::dW_qtPQ, a distributed_array). The vertex kernels (eval_sigma_C_g3w2,
   * pi_c_accumulate_w) consume a FULLY-REPLICATED (nqpts_ibz, nt_half, Np, Np) tau
   * slab and index arbitrary q -- because in M1 the qy/q_ext inner loops are NOT yet
   * distributed (that is M2, change-list items #3/#4), each rank still runs the full
   * serial inner sums and therefore needs every q of W.
   *
   * gather_dW_replicated centralizes the three previously-duplicated gather sites
   * (was: vertex_t.cpp:1066 Sigma, :1427 Pi, :1822 cache_w). dW is a PARTITION of the
   * global array -- every global element lives on exactly one source rank, the rest is
   * zero -- so the all_reduce(plus) of the zero-padded local block is a pure GATHER
   * with no floating-point reassociation: the result is BIT-IDENTICAL on every rank
   * and bit-identical to the pre-M1 code. (A genuinely q-owned distributed result --
   * the eventual memory win -- is only useful once the kernels consume q-owned tiles,
   * i.e. M2; the proven math::nda::redistribute path for that is exercised bit-exactly
   * by the round-trip unit test, notes section 4.2 #4, test_vertex_wredist.cpp.)
   */
  namespace vertex_redist_detail {

    // Gather the RPA-grid distributed dW into a replicated (nq, nt_half, Np, Np) array.
    // Bit-identical to the former in-line "alloc + zero + copy local_range + all_reduce".
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
    // (nt_half, Np, Np) array (notes/vertex_parallelization_v2_plan.md Step 1: per-q
    // tau-domain gather). Returns EXACTLY gather_dW_replicated(dW, ...)( iq, :, :, : ):
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
     * REDUCE-SCATTER (vertex parallelization M3, change-list item #8;
     * notes/vertex_parallelization_M3.md). The Pi^C kernel produces a PARTIAL replicated
     * `part` (each rank holds its round-robin tuple/q_ext contribution to the WHOLE
     * (t, q, P, Q) array). The RPA output grid dPi splits t (ntpools) and P,Q (np_P x
     * np_Q) but NOT q, so every output block is owned by exactly ONE rank. The correct
     * sum-scatter is therefore a reduce onto each block owner -- an MPI_Reduce_scatter,
     * which the mpi3 wrapper lacks, so it is composed from per-owner MPI_Reduce.
     *
     * This REPLACES the former "all_reduce the full Pi_wqMN + build full replicated
     * Pi_up/Pi_tqMN + copy the local block": no full-array all_reduce, and only ONE
     * transient full `part` (freed by the caller) + the owned block survive.
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

  } // vertex_redist_detail

  /**
   * Refinement 2 helpers (notes/refinement2_optionA.md): the secondary ISDF basis on
   * the correlated subspace C and the per-q Option-A transfer maps
   *   s(q) = B(q)^dag B(q),  t(q) = s(q)^+ B(q)^dag C(q)  (theoryB Eq. 35/36),
   *   downfold Wbar = t W t^dag, upfold Pi = t^dag Pibar t (Eq. 38; mutual adjoints
   *   => the no-leak identity Eq. 39 holds algebraically).
   */
  namespace vertex_secondary_detail {

    /**
     * Pair-collocation matrix at transfer q, in the kernels' pinned in/out rule
     * (pi_c_kernel_design.md section 2 rule 1; P-side pairs, k_in = k - q):
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

    // fold-the-core (theoryB Eq. 36): out(m, n) = [t A t^dag](m, n); tmp is (N_m, Np)
    inline void fold_core(nda::MemoryArrayOfRank<2> auto const& t_mP,
                          nda::MemoryArrayOfRank<2> auto const& A_PQ,
                          nda::array<ComplexType, 2>& tmp_mQ,
                          nda::MemoryArrayOfRank<2> auto&& out_mn) {
      nda::blas::gemm(t_mP, A_PQ, tmp_mQ);
      nda::blas::gemm(tmp_mQ, nda::dagger(t_mP), out_mn);
    }

    // upfold (theoryB Eq. 38): out(P, Q) = [t^dag Pibar t](P, Q); tmp is (Np, N_m)
    inline void upfold_core(nda::MemoryArrayOfRank<2> auto const& t_mP,
                            nda::MemoryArrayOfRank<2> auto const& Pi_mn,
                            nda::array<ComplexType, 2>& tmp_Pn,
                            nda::MemoryArrayOfRank<2> auto&& out_PQ) {
      nda::blas::gemm(nda::dagger(t_mP), Pi_mn, tmp_Pn);
      nda::blas::gemm(tmp_Pn, t_mP, out_PQ);
    }

    /**
     * Downfold residual eta(q, .) of theoryB Eq. 40 for one core matrix A:
     *   eta = || B (t A t^dag) B^dag - C A C^dag ||_F / || C A C^dag ||_F,
     * with B t A t^dag B^dag = (B t) A (B t)^dag. Test-scale diagnostic
     * (N_pair x N_pair matrices are formed).
     */
    inline double eta_of(nda::array<ComplexType, 2> const& B_Im,
                         nda::array<ComplexType, 2> const& C_IP,
                         nda::MemoryArrayOfRank<2> auto const& t_mP,
                         nda::MemoryArrayOfRank<2> auto const& A_PQ) {
      const long Npair = B_Im.shape(0), Np = C_IP.shape(1);
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
   * IBZ k-point symmetry helpers (notes/vertex_ibz_symmetry.md sections 3-4, 6).
   */
  namespace vertex_ibz_detail {

    // crystal-coordinate comparison mod integer G-vectors (same class as the
    // matching in generate_qsymm_maps, symmetry.hpp:534-549)
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
     * G-rotation consistency diagnostic (memo section 6): for each symmetry
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

  /**
   * WANNIER-projector helpers (notes/wannier_projector_theory.md section 0-2).
   * The whole substitution is the linearity lemma (memo section 2.0): with the
   * fixed Norb x M isometry U(s,k) the four input-slice sites become
   *   G_bar = U^dag G U   (M x M),   X_bar = X . U   (Np x M),
   * fed to the ALREADY projector-general kernels; the Sigma^C cut comes out in
   * Wannier labels and is injected back as the operator sandwich U Sigma_bar U^dag
   * (memo C2/C3). All rotations act on the W_rng rows only (U is zero elsewhere),
   * so the U arrays carry exactly W_rng.size() rows.
   */
  namespace vertex_wannier_detail {

    // Loewdin orthonormalization of one Norb x M block: U_orth = U (U^dag U)^{-1/2},
    // via the Hermitian eig of the M x M Gram s = U^dag U (owner ruling Q1). Returns
    // ||s - 1_M||_F measured BEFORE the correction. If loewdin == false the block is
    // left raw (the caller warns). nrow = W_rng.size(), M = columns.
    inline double loewdin_block(nda::MemoryArrayOfRank<2> auto &&U, bool loewdin) {
      const long nrow = U.shape(0), M = U.shape(1);
      nda::matrix<ComplexType> s(M, M);
      nda::blas::gemm(ComplexType(1.0), nda::dagger(U), U, ComplexType(0.0), s);
      double defect = 0.0;
      for (long a = 0; a < M; ++a)
        for (long b = 0; b < M; ++b)
          defect += std::norm(s(a, b) - ((a == b) ? ComplexType(1.0) : ComplexType(0.0)));
      defect = std::sqrt(defect);
      if (not loewdin) return defect;
      // s = V diag(lam) V^dag (Hermitian, lam ascending; eigenvectors in COLUMNS,
      // diis_alg.hpp convention); s^{-1/2} = V diag(lam^{-1/2}) V^dag.
      auto [lam, V] = nda::linalg::eigenelements(s);
      utils::check(lam(0) > 1e-12,
                   "vertex_wannier_detail::loewdin_block: the Wannier Gram U^dag U is "
                   "numerically singular (min eigenvalue {} <= 1e-12); the projector "
                   "columns are linearly dependent.", lam(0));
      nda::matrix<ComplexType> sinvhalf(M, M), tmp(M, M);
      for (long a = 0; a < M; ++a)
        for (long b = 0; b < M; ++b)
          tmp(a, b) = V(a, b) / std::sqrt(lam(b));   // V diag(lam^{-1/2})
      nda::blas::gemm(ComplexType(1.0), tmp, nda::dagger(V), ComplexType(0.0), sinvhalf);
      nda::matrix<ComplexType> Uc(nrow, M);
      Uc() = U;
      nda::blas::gemm(ComplexType(1.0), Uc, sinvhalf, ComplexType(0.0), U);
      return defect;
    }

    // G_bar(s,k) = U(s,k)^dag G(s,k)|_{W_rng,W_rng} U(s,k)  (M x M), one tau slice
    // handled by the caller. Gw is the (W_rng x W_rng) band-basis block.
    inline void downfold_G(nda::MemoryArrayOfRank<2> auto const &U,     // (nrow, M)
                           nda::MemoryArrayOfRank<2> auto const &Gw,    // (nrow, nrow)
                           nda::array<ComplexType, 2> &tmp,             // (M, nrow)
                           nda::MemoryArrayOfRank<2> auto &&Gbar) {     // (M, M)
      nda::blas::gemm(nda::dagger(U), Gw, tmp);       // U^dag G
      nda::blas::gemm(tmp, U, Gbar);                  // (U^dag G) U
    }

    // Sigma^C injection (C3, memo section 2.3). CHAIN-RULE form (memo section 1.2),
    // PINNED-BY-TEST by the gauge check + the kernel-level phase razor (memo section 6.2):
    //   Sigma^C_ij += sum_ab conj(U_ia) Sigma_bar_ab U_jb = [conj(U) Sigma_bar U^T]_ij
    // over i,j in W_rng. The Sigma kernel emits Sigma_bar(a,b) with the external index a
    // carrying the NON-conjugated collocation leg (X_bar, phase phi_a) and b the
    // CONJUGATED leg (conj(X_bar), phase conj(phi_b)); so under U -> U V the kernel output
    // transforms as Sigma_bar -> V^T Sigma_bar conj(V), and only the chain-rule sandwich
    // conj(U) Sigma_bar U^T is invariant (the operator sandwich U Sigma_bar U^dag leaks at
    // O(1) under a COMPLEX gauge -- the vertex_sigma_toy "wannier_gauge" oracle). This
    // equals Sigma_bar for U = I, so the window / degenerate-U bit-identity is preserved
    // (conj(I) Sigma_bar I^T = Sigma_bar; NO transpose of Sigma_bar). Sw is the W_rng
    // destination block. Implemented as conj(U) . Sigma_bar . transpose(U).
    inline void upfold_Sigma(nda::MemoryArrayOfRank<2> auto const &U,      // (nrow, M)
                             nda::MemoryArrayOfRank<2> auto const &Sbar,   // (M, M)
                             nda::array<ComplexType, 2> &tmp,              // (nrow, M)
                             nda::MemoryArrayOfRank<2> auto &&Sw) {        // (nrow, nrow)
      // conj(U) cannot be passed alone to gemm (BLAS has no conj-only op); materialize it.
      auto Uc = nda::make_regular(nda::conj(U));       // (nrow x M)
      nda::blas::gemm(Uc, Sbar, tmp);                  // conj(U) Sigma_bar   (nrow x M)
      nda::blas::gemm(ComplexType(1.0), tmp, nda::transpose(U), ComplexType(1.0), Sw);
    }

    // X_bar(s,k) = X(s,k) . U(s,k) : the rotated collocation (Np x M). X_skPa carries
    // the FULL band range on its last axis; U acts on the W_rng rows. k axis is full BZ.
    inline nda::array<ComplexType, 4>
    build_Xbar(nda::MemoryArrayOfRank<4> auto const &X_skPa,   // (ns, nk, Np, nbnd)
               nda::array<ComplexType, 4> const &U_skia,   // (ns, nk, nW, M)
               nda::range W_rng) {
      decltype(nda::range::all) all;
      const long ns = X_skPa.shape(0), nk = X_skPa.shape(1), Np = X_skPa.shape(2);
      const long M = U_skia.shape(3);
      nda::array<ComplexType, 4> Xbar(ns, nk, Np, M);
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nk; ++ik)
          nda::blas::gemm(X_skPa(is, ik, all, W_rng), U_skia(is, ik, all, all),
                          Xbar(is, ik, all, all));   // (Np x nW)(nW x M) = (Np x M)
      return Xbar;
    }

    // G_bar(t,s,k) = U(s,k)^dag G(t,s,k)|_{W_rng,W_rng} U(s,k) on the FULL BZ k axis.
    // Under symmetry the band-basis block is sourced with (G1)/(G2): non-trev k is a
    // pure copy of the IBZ block, trev k the tau-pointwise TRANSPOSE (the same gauge
    // gather the window path uses; memo section 2.8 / 3.5). G_ibz has the IBZ k axis.
    inline void build_Gbar_fullbz(nda::MemoryArrayOfRank<5> auto const &G_ibz,  // (nt,ns,nk_src,nbnd,nbnd)
                                  nda::array<ComplexType, 4> const &U_skia,
                                  nda::range W_rng, bool sym_mesh,
                                  nda::ArrayOfRank<1> auto const &kp_to_ibz,
                                  nda::ArrayOfRank<1> auto const &kp_trev,
                                  nda::MemoryArrayOfRank<5> auto &&Gbar) {      // (nt,ns,nk,M,M)
      decltype(nda::range::all) all;
      const long nt = Gbar.shape(0), ns = Gbar.shape(1), nk = Gbar.shape(2);
      const long M = Gbar.shape(3), nW = W_rng.size(), W0 = W_rng.first();
      nda::array<ComplexType, 2> Gw(nW, nW), tmp(M, nW);
      for (long ik = 0; ik < nk; ++ik) {
        const long ksrc = sym_mesh ? long(kp_to_ibz(ik)) : ik;
        const bool trev = sym_mesh and bool(kp_trev(ik));
        for (long is = 0; is < ns; ++is)
          for (long it = 0; it < nt; ++it) {
            auto Gsrc = G_ibz(it, is, ksrc, all, all);
            if (not trev) {
              for (long a = 0; a < nW; ++a)
                for (long b = 0; b < nW; ++b) Gw(a, b) = Gsrc(W0 + a, W0 + b);
            } else {
              for (long a = 0; a < nW; ++a)
                for (long b = 0; b < nW; ++b) Gw(a, b) = Gsrc(W0 + b, W0 + a);  // transpose
            }
            downfold_G(U_skia(is, ik, all, all), Gw, tmp, Gbar(it, is, ik, all, all));
          }
      }
    }

  } // vertex_wannier_detail

  void vertex_t::set_wannier_projector(methods::projector_t const &proj, bool loewdin) {
    utils::check(enabled(),
                 "vertex_t::set_wannier_projector: the vertex is disabled (vertex_type = "
                 "\"none\"); nothing to project onto.");
    utils::check(proj.nImps() == 1,
                 "vertex_t::set_wannier_projector: only a single impurity is supported "
                 "(nImps = {}); merge the shells in the wan.h5 reader.", proj.nImps());
    decltype(nda::range::all) all;

    // U = dagger(proj_mat) on the W_rng rows (memo section 0 pin): the code's
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
    // WANNIER (memo section 2.8): nW = W_rng.size(), the injection support, and the
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

    // ---- momentum map: krot = ks_to_k (full-BZ rows; memo (R1) direction pin) --------
    ctx.krot = nda::array<long, 2>(ctx.nsym, ctx.nk_full);
    for (long is = 0; is < ctx.nsym; ++is)
      for (long ik = 0; ik < ctx.nk_full; ++ik)
        ctx.krot(is, ik) = MF->ks_to_k(int(is), int(ik));

    // direction self-check: the same map on the Q mesh must send q' -> +/- qs.
    // (Derivation memo section 3.1: slist = find_inverse_symmetry(qsymms) in the MF
    //  makes the D-pair point of k exactly ks_to_k(js, k); assert rather than trust.)
    // NOTE: symm_op.R acts on CRYSTAL coordinates (generate_dmatrix works on
    // kpts_crystal, symmetry.hpp:827+1003); MF->Qpts() is CARTESIAN, so the crystal
    // q list is built self-consistently from kpts_crystal differences via qk_to_k2
    // (bz convention Qpts[q] + G = kpts[a] - kpts[b], bz_symmetry.hpp:540-544).
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
        // image = q' * R (row-vector right action; the generate_qsymm_maps matching,
        // symmetry.hpp:588)
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
    // M3 item #9 (notes/vertex_parallelization_M3.md): Xhat is NODE-SHARED (one copy per
    // NUMA node) and the (js, ik) build loop is DISTRIBUTED across node_comm -- each
    // (js, ik) tile is computed on exactly one node-rank and written into the shared
    // window (a partition => GATHER, bit-identical to the former serial replicated build).
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
          csrmm(ComplexType(1.0), *Dsp, E, ComplexType(0.0), Dcols);
          const long ksrc = ctx.krot(js, cj ? long(kp_trev_pair(ik)) : ik);
          // C-window leakage of this rotation (memo (C-leak)); PLAIN block kept --
          // no extra normalization (consumer precedent, projector_boson_t.cpp:108-121).
          // WANNIER: the projector-level leakage ||(1 - P(Sk)) D U(k)|| = mass of D U(k)
          // falling outside range(P(Sk)); 0 for a symmetry-closed Wannier set (memo 2.8).
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
          // (symmetry.hpp:1067-1092). Xhat = X(ksrc).Dc enters EIGHT collocation legs of
          // Sigma^C and four of Pi^C, so this defect is the accuracy floor of the whole
          // symmetry path -- and it was previously unmeasured.
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
          // effective columns (memo (X-hat)): base collocation at the D-pair point
          // (the trev pair's rotation for trev k -- the API redirect), conj for trev.
          // WANNIER: base = X_bar(ksrc) . d(k;S), d = U(ksrc)^dag D_win U(ik) (M x M).
          nda::array<ComplexType, 2> dloc(nc, nc);   // per-spin C-sector rotation
          for (long is = 0; is < ctx.ns; ++is) {
            if (wan) {
              // DU(nbnd, M) = Dcols(nbnd, nW) . U(ik)(nW, M)
              nda::blas::gemm(Dcols, (*U_skia)(is, ik, all, all), DU);
              // d(M, M) = U(ksrc)^dag(M, nW) . DU[W_rng rows](nW, M)
              for (long p = 0; p < nW; ++p)
                for (long a = 0; a < nc; ++a) dW_win(p, a) = DU(C0_global + p, a);
              nda::blas::gemm(nda::dagger((*U_skia)(is, ksrc, all, all)), dW_win, dloc);
              // projector-level leakage: 1 - ||P(ksrc) DU||^2 / ||DU||^2 (memo 2.8)
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
    // and reduce the leakage scalars over the node (a diagnostic sum -- no gate).
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
    // As of increment S3 the STATIC path of Sigma^C is wired and unit-tested (the
    // doubly-instantaneous reduction with both rungs = W0bar). The guard nonetheless
    // stays CLOSED, and for a stronger reason than "not implemented": B-S's
    // non-negotiable invariant is "Sigma^{C,x} and Sigma^{C,r} TOGETHER or neither"
    // (theoryB_static.pdf section 6.4). Sigma^{C,x} alone is not the derivative of any
    // functional once W0 is rebuilt from the current G each iteration, so a run
    // producing it alone would be silently non-conserving -- exactly the failure mode
    // the parent theory's "both cuts or neither" rule exists to prevent. The gate opens
    // at S5 (static) / S9 (linear), when the response term lands.
    // B-S (static_rung) is COMPLETE as of increment S5: Sigma = Sigma^{C,x} + Sigma^{C,r},
    // both cuts always together (they are assembled in one place, eval_Sigma_C, so the
    // half-theory is structurally unrepresentable), and P = P_RPA with the Pi^C injection
    // switched off at the update_w seam. B-L (linear_rung) still lacks its mixed Sigma
    // terms and its own response term (increments S8/S9).
    // All three modes are implemented as of increment S9. Each assembles ALL of its own
    // cuts in one place, so the non-conserving half-theories remain unrepresentable:
    //   dynamic : Sigma^C (G^3W^2) + Pi^C (G^4W)
    //   static  : Sigma^{C,x} + Sigma^{C,r};  P = P_RPA (no injection)
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
      // Announce which theory is active. All three rung modes are implemented
      // (increments S3-S9); each assembles ALL of its own cuts, so a half-theory cannot
      // be configured.
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
      app_log(1, "\nvertex_t: vertex_type = \"{}\" (vertex_rung = \"{}\") with an empty "
                 "vertex_band_window: C = empty set, so the vertex contributes nothing in "
                 "ANY rung mode and the calculation reduces to plain scGW exactly.\n",
              _vertex_type, rung_str());
    }
  }

  void vertex_t::build_secondary_basis(THC_ERI auto const &thc,
                                       nda::MemoryArrayOfRank<4> auto const &X_glob, long orb0,
                                       nda::array<long, 2> const &kmq, long iq_gamma) {
    if (_secondary_ready) return;
    decltype(nda::range::all) all;
    auto mpi = thc.mpi();
    auto MF = thc.MF();
    const long ns = X_glob.shape(0), nkpts = X_glob.shape(1), Np = X_glob.shape(2);
    // t(q) is built at IBZ q ONLY (notes/vertex_ibz_symmetry.md section 3.7): the
    // kernels source non-IBZ transfers from the IBZ-stored folded cores through the
    // symmetry context. On symmetry-free meshes nqpts_ibz == nqpts (historic path).
    const long nqpts = MF->nqpts_ibz();
    utils::check(kmq.shape(0) >= nqpts,
                 "vertex_t::build_secondary_basis: kmq must cover the IBZ q range.");
    const long nc = subspace_rank();
    const long Npair = ns * nkpts * nc * nc;   // the pair index carries momentum
                                               // (CLAUDE.md section 2, invariant 4)
    long Nm_req = (_isdf_rank > 0) ? _isdf_rank : nc * nc * nkpts;
    utils::check(Nm_req <= Npair,
                 "vertex_t::build_secondary_basis: vertex_isdf_rank = {} exceeds the "
                 "subspace pair rank N_pair = ns*nk*nc^2 = {}; the secondary basis "
                 "cannot usefully exceed the space it represents.", Nm_req, Npair);
    // Cap the secondary rank at the GLOBAL basis size Np: the secondary ISDF lives inside
    // the span of the global THC basis, so it must never request more interpolating vectors
    // than the global basis has (out-ranking it selects near-null directions and makes the
    // secondary metric s = B^dag B ill-conditioned -- the companion guard to sec_thresh below).
    Nm_req = std::min(Nm_req, (long)thc.Np());

    // Secondary-ISDF point-selection threshold. It DEFAULTS to the SAME thresh used for
    // the GLOBAL THC basis (thc.thresh()) unless vertex_isdf_thresh (>0) overrides it.
    // Over-resolving the C pair-density metric (e.g. the old hardcoded 1e-13) selects
    // interpolating vectors that leave the span of the global basis, so the transfer
    // t(q) = pinv(B) picks up near-null directions and s becomes ill-conditioned
    // (kp444/M8: N_m=2075 > global Np=1086, cond(s)~1e25). thc.thresh() is -1.0 when the
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
               "  requested N_m = {}, svd_tol(B) = {}, sec_thresh = {} (global THC thresh = {}), "
               "N_pair (per q, spin-stacked) = {}\n",
            nc, _wannier ? "Wannier projector" : "band window",
            Nm_req, _isdf_svd_tol, sec_thresh, thc.thresh(), Npair);

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
      methods::thc builder(MF.get(), *mpi, pt, /*print_metadata*/ false);
      // WINDOW: the band-range overload (Wannier=window). WANNIER (owner ruling Q2):
      // the committed rotated overload interpolating_points(C_skai, iq, max), fed the
      // zero-padded U as C_skai(s,k,a,i) = conj(U_ia) on the W_rng band columns (memo
      // C5/section 2.6; the overload rotates the real-space orbitals by conj(C_skai),
      // thc.icc:831-858, so this yields exactly the Wannier orbitals w_a). Its metric
      // resolves (Wannier x all-band) pairs -- a superset of the (Wannier x Wannier)
      // pairs the vertex needs; eta(q,nu) certifies adequacy. The rotated overload
      // requires nkpts == nkpts_ibz (thc.cpp:207) -- Wannier+secondary is nosym only.
      nda::array<long, 1> ipts;
      nda::array<ComplexType, 4> Xa(ns, nkpts, nc, 0);   // (ns, nk, nc, Nm), filled below
      if (not _wannier) {
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
        utils::check(MF->nkpts() == MF->nkpts_ibz(),
                     "vertex_t::build_secondary_basis: the Wannier rotated point-selection "
                     "overload does not support symmetry-reduced k-meshes (thc.cpp:207). "
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
      if (Nm < Nm_req)
        app_log(1, "  [NOTE] Refinement 2: point selection stopped at N_m = {} "
                   "(< requested {}):\n"
                   "         the C pair-density metric is numerically rank-deficient below "
                   "thresh = 1e-13;\n"
                   "         using the returned rank.", Nm, Nm_req);
      // gather the distributed collocation (already assembled into Xa above), then
      // transpose to the kernels' (aux, orb) layout. Any fixed per-point phase/scale
      // convention of the selection output is absorbed by the least-squares transfer.
      mpi->comm.all_reduce_in_place_n(Xa.data(), Xa.size(), std::plus<>{});
      _Xb_skma = nda::array<ComplexType, 4>(ns, nkpts, Nm, nc);
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nkpts; ++ik)
          for (long a = 0; a < nc; ++a)
            for (long m = 0; m < Nm; ++m)
              _Xb_skma(is, ik, m, a) = Xa(is, ik, a, m);
      _Nm = Nm;
    }

    // ---- conditioning cap (vertex_isdf_cond_max): applied PER Q in the transfer solve ---
    // The cond(s) blowup is Q-SPECIFIC and NOT at Gamma (measured: Gamma cond ~1e4 but
    // max_q cond ~1e19 at a non-Gamma transfer). The interpolating POINTS are shared across
    // all q, so pruning the shared set CANNOT bound the worst-q conditioning -- pruning only
    // removes points redundant at EVERY q, while the worst q's ill-conditioning comes from
    // points that nearly coincide THERE but separate elsewhere (why both were selected). The
    // robust cap is therefore applied per q in the least-squares solve below: the gelss
    // rcond truncates B(q)'s near-null directions so each q's downfold t(q) is conditioned
    // to <= _isdf_cond_max (rcond_eff = 1/sqrt(cond_max), floored by _isdf_svd_tol).
    // Disabled (_isdf_cond_max <= 0) => rcond_eff = _isdf_svd_tol => BIT-IDENTICAL legacy.

    // ---- per-q Option-A transfer t(q) = s(q)^+ B(q)^dag C(q) (theoryB Eq. 36) ---------
    // Solved as the truncated-SVD least squares min || B t - C ||_F directly on B
    // (numerically equivalent, better conditioned: rcond acts on sv(B); the metric
    // s = B^dag B is thereby regularized at rcond^2). The explicit s^{-1} is REQUIRED:
    // the code's THC body contractions are metric-free (coqui_conventions_confirmed.md).
    //
    // M3 item #6 (notes/vertex_parallelization_M3.md): the per-q gelss solve was
    // FULLY SERIAL AND REDUNDANT (every rank formed the full Npair-row B/C and solved
    // ALL q identically). Distribute the q loop over mpi->comm -- each rank solves its
    // q subset (round-robin), zero-init _t_qmP, then a single all_reduce(plus) GATHERS
    // the rows (a partition: each q written by exactly one rank => bit-identical to the
    // former serial solve for each q, only WHICH rank computed it changes). Per-rank
    // solve work (gelss calls) drops ~1/P. _t_qmP stays replicated (its all-q consumers
    // -- Pi upfold, Sigma, cache_w -- loop every IBZ q); the redundant SOLVE was the
    // actionable win. Diagnostics (cond/fit/disc maxima) are reduced with max.
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
    // M3 item #6: GATHER the q-distributed t(q) rows (partition => exact, bit-identical
    // per q to the serial solve) and reduce the diagnostics across ranks.
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
      // Decision D3 (notes/static_vertex_implementation_plan.md section 6): the static
      // rungs need no pole algebra, so this requirement is NOT structural like the
      // dynamic one -- it stands only until the tau = 0 interpolation row (section 2.4)
      // is shown to be available on the IR driver, which is checked at increment S4.
      utils::check(false,
                   "{}: vertex_rung = \"{}\" also requires the DLR IAFT backend "
                   "(iaft basis = \"dlr\") for now. The static rungs themselves need no "
                   "pole algebra; what is missing on IR is the Pi^{{C,0}}(tau = 0) "
                   "interpolation row (decision D3, open until increment S4).",
                   where, rung_str());
    }
  }

  void vertex_t::eval_Sigma_C(MBState &mb_state, THC_ERI auto const &thc) {
    utils::check(active(), "vertex_t::eval_Sigma_C: called while the vertex is inactive. "
                           "Callers must guard vertex calls with vertex_t::active().");
    // this file only implements the DYNAMIC-rung (Formulation B) kernel; B-S/B-L land at S3+
    check_rung_implemented("vertex_t::eval_Sigma_C");
    // INCREMENT S3: STATIC-rung mode (B-S). Sigma^{C,x} is the doubly-instantaneous
    // reduction of the SAME kernel with both rungs = W0bar (plan section 1). Nothing
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

    // IBZ SYMMETRY (notes/vertex_ibz_symmetry.md): on symmetry-reduced meshes the
    // external k axis stays IBZ-resident, all internal sums run over the full BZ,
    // and the rungs are sourced from the IBZ-stored W/Z through the symmetry
    // context. Symmetry-free meshes take the historic path bit-identically.
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
    // M2 item #5-finish: node-share X_skPa (the ns*nk*Np*nbnd collocation, spec section
    // 1.1's memory wall) -- one copy per NUMA node, not one per rank. Values are copied
    // from the already-node-shared thc.X (data-location change only => bit-identical).
    // All downstream X consumers (kernel, build_Xbar, build_sym_ctx, build_secondary_basis,
    // eta_max_over_q, X_C slice) were templated to bind the shared_array .local() view.
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

    // ---- q->0 rung policy (notes/q0_head_treatment.md section 3) ----------------------
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
    // was already applied to W0 inside build_w0 (section 2.2 step 3), so re-applying it
    // here would double-count it.
    nda::array<ComplexType, 2> H_PQ(need_dyn ? Np : 0, need_dyn ? Np : 0);
    bool head_ok = false;
    if (head_insertion and need_dyn) {
      head_ok = vertex_head_detail::build_head_rank1(thc, iq_gamma, nkpts, H_PQ);
      if (head_ok) {
        Z_qPQ(iq_gamma, all, all) += H_PQ;
        double h_max = 0.0;
        for (auto const& v : H_PQ) h_max = std::max(h_max, std::abs(v));
        app_log(1, "  Sigma^C head insertion: madelung = {}, |H|_max = {} (bare piece "
                   "applied to Z(Gamma))", MF->madelung(), h_max);
      } else {
        app_log(1, "  [WARNING] Sigma^C: gygi head insertion requested but head data are "
                   "unusable\n"
                   "            (madelung == 0 or empty basis_head) -- proceeding WITHOUT "
                   "the analytic head\n"
                   "            (equivalent to policy \"ignore_g0\").");
      }
    }

    // ---- dynamic W(tau): replicate and unfold nt_half storage to the full tau mesh ----
    // dW_qtPQ is dynamic-only (bare Z subtracted, scr_coulomb_t.cpp:217); W is
    // PH-symmetric in tau, W(beta-t) = W(t). IBZ rows under symmetry.
    nda::array<ComplexType, 4> Wt_qtPQ(need_dyn ? nqpts_ibz : 0, nt, Np, Np);
    if (need_dyn) {
      // M1 item #1: gather the RPA-grid dW into the replicated tau slab the kernel
      // needs (bit-identical to the former in-line allreduce; helper centralizes it).
      nda::array<ComplexType, 4> W_half = vertex_redist_detail::gather_dW_replicated(
          mb_state.dW_qtPQ.value(), mpi->comm, nqpts_ibz, nt_half, Np);

      // head insertion, dynamic piece (weight Re[eps_inv_head(tau)]) into dW(Gamma, tau).
      // eps_inv_head = eps^-1_00(q->0, tau) - 1, stored on nt_half by scr_coulomb
      // (scr_coulomb_t.cpp:106-108); same Re[.] convention as Sigma_div_correction.
      if (head_ok) {
        if (mb_state.eps_inv_head.has_value()) {
          auto& eps = mb_state.eps_inv_head.value();
          utils::check(eps.shape(0) == nt_half,
                       "vertex_t::eval_Sigma_C: eps_inv_head size {} != nt_half = {}.",
                       eps.shape(0), nt_half);
          for (long it = 0; it < nt_half; ++it)
            W_half(iq_gamma, it, all, all) += ComplexType(eps(it).real()) * H_PQ;
          app_log(1, "  Sigma^C head insertion: dynamic piece applied to dW(Gamma, tau) "
                     "with eps_inv_head(tau=0) = {}", eps(0).real());
        } else {
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

    // ---- momentum maps (symmetry-free mesh) -------------------------------------------
    nda::array<long, 2> kmq(nqpts, nkpts);
    nda::array<long, 1> qmin(nqpts);
    for (long iq = 0; iq < nqpts; ++iq) {
      qmin(iq) = MF->qminus()(iq);
      for (long ik = 0; ik < nkpts; ++ik) kmq(iq, ik) = MF->qk_to_k2(iq, ik);
    }

    // ---- Refinement 2: optional secondary-basis substitution --------------------------
    // (notes/refinement2_optionA.md section 4). The SAME kernel runs on the input set
    // (Xb, Zbar = t Z t^dag, Wbar = t dW t^dag, G_CC, window [0, nc)) -- fold-the-core;
    // the head-augmented Gamma cells above downfold automatically through t (rank-1
    // t H t^dag = (t conj(chi))(t conj(chi))^dag, memo section 3). Sigma^C externals
    // are a, b in C (theoryB 11.5) and land in the C-C block; NO upfold (Eq. 38 text).
    const bool sec = secondary();
    const bool wan = _wannier;
    const long nc = subspace_rank();     // = M (Wannier) or _band_window.size() (window)
    nda::array<ComplexType, 3> Zb_qmm;
    nda::array<ComplexType, 4> Wb_qtmm;
    // STRICT C-C EXTERNALS (theory-owner ruling, notes/refinement2_optionA.md
    // DECISION 2): in Phi_2^C ALL FOUR G-lines -- including the cut one -- are
    // C-restricted, so Sigma^C = dPhi/dG is nonzero ONLY on the C-C block (window) /
    // range(P) (Wannier). BOTH paths run the kernel with C-restricted externals
    // (G_CC + the C columns of the collocation); the full-range extension of the
    // kernel formula is well-defined but is NOT dPhi/dG.
    // WINDOW: G_CC = the W-window block; WANNIER: G_CC = U^dag G U (memo C2/section 2.2).
    // On the FULL BZ (memo (G1)/(G2)): image points are gauge copies of the IBZ blocks
    // (identity D by convention, symmetry.hpp:910); trev points are the tau-pointwise
    // TRANSPOSE (thc_solver_comm.hpp:443-447; == conj for the hermitian G). No tau-mirror
    // anywhere (memo section 3.5).
    // M1 item #5: node-share G_CC (one copy per NUMA node, not one per rank). G_CC is
    // built from the already node-shared sG_tskij and is READ (never written) by the
    // kernel and g_rotation_check, which take it via a templated array param -- so the
    // shared_array .local() view binds without any kernel change. Values are identical
    // to the per-rank array (data-location change only) => bit-identical outputs.
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
    // M2 item #5-finish: node-share X_C too (one copy per node; built on node root from
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
      // eta diagnostics (Eq. 40) on the rung arrays ACTUALLY consumed (test-scale gate)
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
      // non-IBZ transfers are sourced through the sym ctx, memo section 3.7).
      // STATIC modes: W0bar is ALREADY the downfolded rung (build_w0 step 4), and no
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

    // ---- IBZ symmetry context (trivial/null on symmetry-free meshes) ------------------
    // WANNIER (memo section 2.8): thread U through build_sym_ctx so the C-sector
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

    // ---- fused kernel (round-robin over (s,k,qx); result all-reduced inside) ----------
    // Both paths: C-restricted externals; the ONLY difference is the auxiliary input
    // set -- (X_C, W, Z, Np) global vs (Xb, Wbar, Zbar, N_m) secondary.
    nda::array<ComplexType, 5> Sigma_C(nt, ns, nk_ext, nc, nc);
    if (stat) {
      // B-S: BOTH rungs are W0bar. The kernel's doubly-instantaneous reduction S3 is
      // Sigma^{C,x} (eq:sigmaxtau); families I-V and S1/S2 are identically zero and are
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
      nda::array<ComplexType, 4> dWw_lin;
      nda::array<ComplexType, 4> const *wwp = nullptr;
      if (lin) {
        vertex_pi::iaft_tools tls(*_ft);
        dWw_lin = nda::array<ComplexType, 4>(nqpts_ibz, tls.nw_b, naux, naux);
        dWw_lin() = ComplexType(0.0);
        for (long iq = 0; iq < nqpts_ibz; ++iq)
          for (long m = 0; m < tls.nw_b; ++m)
            for (long M = 0; M < naux; ++M)
              for (long N = 0; N < naux; ++N) {
                ComplexType acc(0.0);
                for (long it = 0; it < nt; ++it)
                  acc += tls.Twt_bb(m, it) * (sec ? Wb_qtmm(iq, it, M, N)
                                                  : Wt_qtPQ(iq, it, M, N));
                dWw_lin(iq, m, M, N) = acc
                    + (sec ? Zb_qmm(iq, M, N) : Z_qPQ(iq, M, N)) - W0b(iq, M, N);
              }
        wwp = &dWw_lin;
        double z0 = 0.0, zs = 0.0;
        for (long iq = 0; iq < nqpts_ibz; ++iq)
          for (long M = 0; M < naux; ++M)
            for (long N = 0; N < naux; ++N) {
              z0 = std::max(z0, std::abs(dWw_lin(iq, tls.m0, M, N)));
              zs = std::max(zs, std::abs(W0b(iq, M, N)));
            }
        // |dW(i.nu = 0)| = |W(q,0) - W0(q)| is the VERTEX CORRECTION TO THE STATIC
        // SCREEN, and in B-L it is nonzero BY DESIGN: the kernel W0[G] is the RPA-static
        // screen, while the run's own W carries P^{C,L}. The self-slice identity
        // W(q,0) == W0(q) holds in B-S (where P = P_RPA) but NOT in B-L -- see
        // theoryB_static.pdf, "Kernel choice, heads, and lifetimes": this is a
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
                                       iq_gamma, skip_rung_gamma, 0, nullptr, symc, Sigma_C);
    else
      vertex_detail::eval_sigma_C_g3w2(*_ft, mpi->comm, nda::range(0, nc), G_CC,
                                       X_C, Wt_qtPQ, Z_qPQ, kmq, qmin, iq_gamma,
                                       skip_rung_gamma, 0, nullptr, symc, Sigma_C);
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

    // ---- INCREMENT S5: the RESPONSE cut Sigma^{C,r} -----------------------------------
    // Phi-derivability of B-S requires Sigma^{C,x} and Sigma^{C,r} TOGETHER: W0 is an
    // explicit functional of the CURRENT G, so differentiating Phi produces this chain-
    // rule term as well. Routing verified mechanically (the transposed, symmetrized
    // sandwich) and pinned end-to-end by test_vertex_fdoracle.
    nda::array<ComplexType, 5> Sigma_r;
    if (stat) {
      // IBZ: NOT YET IMPLEMENTED for Sigma^{C,r}, and note WHY the obvious route does
      // not work. Every other vertex kernel is C-restricted, so the vertex symmetry
      // context carries only the C-WINDOW effective collocation columns
      // (vertex_sym::sym_ctx::Xhat is (ns, nsym, nk, naux, nc)). Sigma^{C,r} is the one
      // object with FULL-SPACE legs -- its Gt and its externals live in W0's unprojected
      // RPA bubble -- so Xhat is the wrong tool for it. It needs the PLAIN GW pattern
      // instead (gw_t::eval_Sigma_all_kspace: accumulate at IBZ k in the aux basis, then
      // unfold with MF->symmetry_rotation over MF->qsymms()), which is cheap here because
      // Sigma^{C,r} carries a SINGLE transfer q -- no two-rung unfolding.
      // Gate for that work: sym-vs-nosym agreement, the vertex_ibz gold pattern.
      utils::check(not sym_mesh,
                   "vertex_t::eval_Sigma_C: vertex_rung = \"{}\" on a SYMMETRY-REDUCED "
                   "mesh is not supported yet. Sigma^(C,r) has FULL-SPACE legs, so it "
                   "cannot use the vertex symmetry context (whose Xhat is C-window only); "
                   "it needs the plain GW unfolding path. Run on a symmetry-free mesh "
                   "(nosym = noinv = .true. in the QE nscf) until that lands.",
                   rung_str());
      auto const& W0b_r = _W0b_qmm.value();
      vertex_pi::iaft_tools tools(*_ft);
      nda::array<long, 2> kpq(nqpts, nkpts);
      for (long iq = 0; iq < nqpts; ++iq)
        for (long ik = 0; ik < nkpts; ++ik) kpq(iq, ik) = kmq(qmin(iq), ik);

      // (1) Pi^{C,0}(q, i.nu): the pinned instantaneous (Z) phase with the rung W0bar and
      //     NO dynamic rung -- pi_c_accumulate_w returns right after phase 1 on nullptr.
      //     Fed the C-C block: the kernel CONTRACTS its external orbital legs into the aux
      //     indices, so their range is part of the object (all eight labels of Phi are in C).
      const long Naux_pi = sec ? _Nm : Np;
      nda::array<ComplexType, 4> Pi_wq(tools.nw_b, nqpts_ibz, Naux_pi, Naux_pi);
      Pi_wq() = ComplexType(0.0);
      if (sec)
        vertex_pi::pi_c_accumulate_w(*_ft, tools, G_CC, _Xb_skma, W0b_r, nullptr,
                                     kmq, kpq, nda::range(0, nc), Pi_wq,
                                     mpi->comm.rank(), mpi->comm.size());
      else
        vertex_pi::pi_c_accumulate_w(*_ft, tools, G_CC, X_C, W0b_r, nullptr,
                                     kmq, kpq, nda::range(0, nc), Pi_wq,
                                     mpi->comm.rank(), mpi->comm.size());
      mpi->comm.all_reduce_in_place_n(Pi_wq.data(), Pi_wq.size(), std::plus<>{});

      // (2) the tau = 0 row (the LEGAL evaluation of (1/beta) sum_nu; sparse nodes are
      //     fitting nodes, not Fourier points)
      auto R0 = vertex_w0_detail::tau0_transform_row(*_ft);
      auto tau0_of = [&](nda::array<ComplexType, 4> const &Pw,
                         nda::array<ComplexType, 3> &out) {
        for (long iq = 0; iq < nqpts_ibz; ++iq)
          for (long M = 0; M < Naux_pi; ++M)
            for (long N = 0; N < Naux_pi; ++N) {
              ComplexType acc(0.0);
              for (long m = 0; m < tools.nw_b; ++m) acc += R0(m) * Pw(m, iq, M, N);
              out(iq, M, N) = acc;
            }
      };
      nda::array<ComplexType, 3> Pi0(nqpts_ibz, Naux_pi, Naux_pi);
      tau0_of(Pi_wq, Pi0);

      // ---- INCREMENT S9: B-L's response middle factor -------------------------------
      // B-S sandwiches Pi^{C,0}(tau=0); B-L sandwiches the DIFFERENCE
      //     Pi^L = pi^dyn - Pi^{C,0}(tau = 0),   pi^dyn = Pi^{C,dyn}(q, tau = 0),
      // because the rung derivative of the tangent functional is
      //     X^L = -(1/2)[PiBar^dyn - PiBar^0]  (transposed/symmetrized as in S5).
      // X^L therefore VANISHES when the screening is genuinely static: it is a built-in,
      // per-q meter of the static-kernel approximation itself, logged below.
      //
      // NOTE (performance, not correctness): pi^dyn is obtained here from the FULL
      // dynamic-rung Pi^C kernel and then evaluated at tau = 0. Theory eq:pibardynfact
      // shows the frequency SUM factorizes into a single bosonic pairing of two bubbles
      // against W -- much cheaper, and it would avoid invoking the twisted-pair pole
      // algebra (whose conditioning is the parent theory's open issue) for what is only
      // an equal-time value. Implementing that factorized primitive is the natural
      // follow-up; correctness is unaffected.
      if (lin) {
        nda::array<ComplexType, 4> Wdyn_w(nqpts_ibz, tools.nw_b, Naux_pi, Naux_pi);
        Wdyn_w() = ComplexType(0.0);
        for (long iq = 0; iq < nqpts_ibz; ++iq)
          for (long m = 0; m < tools.nw_b; ++m)
            for (long M = 0; M < Naux_pi; ++M)
              for (long N = 0; N < Naux_pi; ++N) {
                ComplexType acc(0.0);
                for (long it = 0; it < nt; ++it)
                  acc += tools.Twt_bb(m, it) * (sec ? Wb_qtmm(iq, it, M, N)
                                                    : Wt_qtPQ(iq, it, M, N));
                Wdyn_w(iq, m, M, N) = acc;
              }
        nda::array<ComplexType, 4> Pid_wq(tools.nw_b, nqpts_ibz, Naux_pi, Naux_pi);
        Pid_wq() = ComplexType(0.0);
        if (sec)
          vertex_pi::pi_c_accumulate_w(*_ft, tools, G_CC, _Xb_skma, Zb_qmm, &Wdyn_w,
                                       kmq, kpq, nda::range(0, nc), Pid_wq,
                                       mpi->comm.rank(), mpi->comm.size());
        else
          vertex_pi::pi_c_accumulate_w(*_ft, tools, G_CC, X_C, Z_qPQ, &Wdyn_w,
                                       kmq, kpq, nda::range(0, nc), Pid_wq,
                                       mpi->comm.rank(), mpi->comm.size());
        mpi->comm.all_reduce_in_place_n(Pid_wq.data(), Pid_wq.size(), std::plus<>{});
        nda::array<ComplexType, 3> PiDyn0(nqpts_ibz, Naux_pi, Naux_pi);
        tau0_of(Pid_wq, PiDyn0);
        double xl = 0.0, p0 = 0.0;
        for (long iq = 0; iq < nqpts_ibz; ++iq)
          for (long M = 0; M < Naux_pi; ++M)
            for (long N = 0; N < Naux_pi; ++N) {
              const ComplexType d = PiDyn0(iq, M, N) - Pi0(iq, M, N);
              xl = std::max(xl, std::abs(d));
              p0 = std::max(p0, std::abs(Pi0(iq, M, N)));
              Pi0(iq, M, N) = d;              // Pi^L = pi^dyn - Pi^{C,0}(tau = 0)
            }
        app_log(1, "  X^L diagnostic: max|pi^dyn - Pi^(C,0)(tau=0)| = {:.4e}, "
                   "relative to max|Pi^(C,0)(tau=0)| = {:.4e}  -> X^L/Pi^0 = {:.4f} "
                   "(vanishes iff the screening is truly static; theory diagnostic O3)",
                xl, p0, (p0 > 0.0 ? xl / p0 : 0.0));
      }

      // (3) Sigma^{C,r} is a GLOBAL-aux object (its Gt and externals are full-space), so
      //     the secondary-basis Pi is upfolded, Pi_hat = t^dag Pibar t (Eq. 38).
      nda::array<ComplexType, 3> Pi0g(nqpts_ibz, Np, Np), W0g(nqpts_ibz, Np, Np);
      if (sec) {
        nda::array<ComplexType, 2> tmp_Pn(Np, _Nm);
        for (long iq = 0; iq < nqpts_ibz; ++iq)
          vertex_secondary_detail::upfold_core(_t_qmP(iq, all, all), Pi0(iq, all, all),
                                               tmp_Pn, Pi0g(iq, all, all));
        // the global static screen: zero-pad + all_reduce GATHER of the (P,Q)-distributed
        // W0 (the gather_dW_replicated pattern; every element lives on exactly one rank,
        // so this is a pure gather with no reassociation).
        // NOTE (production): this replicates an (nq, Np, Np) object. At Np = 20k that is
        // the 320 GB class the v2 rule forbids -- the distributed (P,Q) sandwich of plan
        // section 3 is the S11 hardening item. Validation scales are unaffected.
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
      vertex_detail::build_delta_w(W0g, Pi0g, qmin, Dw);
      Sigma_r = nda::array<ComplexType, 5>(nt, ns, nkpts, nbnd, nbnd);
      vertex_detail::eval_sigma_C_response(mpi->comm, G_tskij, X_skPa, Dw, kmq, qmin,
                                           Sigma_r);
      double rmax = 0.0, xmax = 0.0;
      for (auto const& v : Sigma_r) rmax = std::max(rmax, std::abs(v));
      for (auto const& v : Sigma_C) xmax = std::max(xmax, std::abs(v));
      utils::check(std::isfinite(rmax),
                   "vertex_t::eval_Sigma_C: Sigma^(C,r) contains NaN/Inf -- aborting.");
      app_log(1, "  Sigma^({}): max|.| = {:.4e}; response share "
                 "||Sigma^(C,r)||/||Sigma^(C,x)|| = {:.4f} (large => the deleted rung "
                 "dynamics likely matters; theory diagnostic O3)",
              (lin ? "L,r" : "C,r"), rmax, (xmax > 0.0 ? rmax / xmax : 0.0));
    }

    // accumulate on top of the GW self-energy: Sigma <- Sigma + Sigma^C
    // (Sigma_C is identical on every rank after the kernel's all_reduce; hermitization
    //  stays downstream in scf_driver).
    //   WINDOW MODE: Sigma_C is already in band labels on the C-C block -> drop in.
    //   WANNIER MODE (memo C3/section 2.3): Sigma_C lives in Wannier labels; inject the
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
      if (stat) sSigma_tskij.local() += Sigma_r;
      if (not wan) {
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
    mb_state.mpi->comm.barrier();
  }

  auto vertex_t::eval_Pi_C(MBState &mb_state, THC_ERI auto const &thc,
                           shape_t<4> pi_pgrid, shape_t<4> pi_bsize, shape_t<4> pi_gshape)
  -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>
  {
    decltype(nda::range::all) all;
    utils::check(active(), "vertex_t::eval_Pi_C: called while the vertex is inactive. "
                           "Callers must guard vertex calls with vertex_t::active().");
    // this file only implements the DYNAMIC-rung (Formulation B) G^4W cut. B-S has NO
    // Pi^C injection at all (plan section 2.1 -- scr_coulomb_t never calls us in that
    // mode); B-L's P^{C,L} is the static-rung Z-phase and lands at S7.
    check_rung_implemented("vertex_t::eval_Pi_C");
    // THE FORBIDDEN HYBRID (theoryB_static.pdf, the W-cut section): B-S's W-cut vanishes
    // identically, so P = P_RPA. Injecting a static-rung Pi^C while using B-S's Sigma
    // would pair the G-cut of Phi_2^{C,0} with the W-cut of a DIFFERENT functional and
    // break conservation exactly as "Sigma^C without P^C" did in the parent theory. The
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

    // IBZ SYMMETRY (notes/vertex_ibz_symmetry.md): the external q axis of Pi^C is
    // IBZ-resident (as the output grid already is); the internal (k, qx) sums run
    // over the full BZ, sourcing the rung from IBZ-stored W/Z via the symmetry
    // context. Symmetry-free meshes take the historic path bit-identically.
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

    // ---- q->0 rung policy (notes/q0_head_treatment.md section 3) ----------------------
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
    // M2 item #5-finish: node-share X_skPa (one copy per node; see eval_Sigma_C).
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
    // it). SECONDARY path (Impl 2b): do NOT materialize the replicated all-q Z (320 GB @
    // production Np = 20000) -- it is folded DISTRIBUTED from thc.dZ below. Z_qPQ stays a
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

    // head insertion, bare piece (weight 1) into Z(Gamma) -- (H1) of the memo. The head is
    // EXACTLY rank-1 (build_head_rank1): H_PQ = N_k * madelung * conj(chi_g) chi_g^T with
    // chi_g = thc.basis_head()(iq_gamma, :). The GLOBAL path materializes the dense (Np x Np)
    // H_PQ and adds it into Z_qPQ(Gamma) here. The SECONDARY path (Impl 2c) NEVER materializes
    // the dense Np^2 head (6.4 GB @ production Np = 20000): it captures the Np-vector chi_g and
    // the scalar c = N_k * madelung, and rebuilds each (P,Q) block on the fly from those
    // (vertex_secondary_detail::head_block_add) inside the distributed W/Z fold closures and
    // the test-scale diagnostics -- bit-identical per element to the dense-H slice.
    nda::array<ComplexType, 2> H_PQ;                  // GLOBAL path only (0-sized in secondary)
    nda::array<ComplexType, 1> chi_g;                 // SECONDARY path: chi(iq_gamma, :)
    ComplexType head_c = ComplexType(0.0);            // SECONDARY path: c = N_k * madelung
    bool head_ok = false;
    if (head_insertion) {
      if (not sec_z) {
        // GLOBAL path: dense rank-1 head (unchanged from the legacy build).
        H_PQ = nda::array<ComplexType, 2>(Np, Np);
        head_ok = vertex_head_detail::build_head_rank1(thc, iq_gamma, nkpts, H_PQ);
      } else {
        // SECONDARY path: replicate build_head_rank1's skip logic EXACTLY (madelung == 0 or an
        // all-zero chi(iq_gamma, :) => head_ok = false, no head) WITHOUT the dense Np^2 matrix.
        const double xi = MF->madelung();
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
      } else {
        app_log(1, "  [WARNING] Pi^C: gygi head insertion requested but head data are "
                   "unusable\n"
                   "            (madelung == 0 or empty basis_head) -- proceeding WITHOUT "
                   "the analytic head\n"
                   "            (equivalent to policy \"ignore_g0\").");
      }
    }

    // ---- dynamic W on the full bosonic Matsubara mesh ---------------------------------
    // mb_state.dW_qtPQ is the dynamic-only screened interaction (bare Z subtracted,
    // scr_coulomb_t.cpp:217) on (nq, nt_half, Np, Np). Source selection
    // (notes/wbar_cache.md):
    //   - SECONDARY path with a FILLED W-bar cache: the previous iteration's rung was
    //     already downfolded at update_w time (cache_w) -- the global-basis Wdyn is
    //     NOT rebuilt here (the scf driver frees dW unconditionally in this mode:
    //     plain-GW memory profile). Same one-iteration lag as the retained-dW path.
    //   - otherwise (global path; or secondary with a retained dW -- the legacy /
    //     compat branch, kept verbatim as the machine-identity reference): fold at
    //     consumption from mb_state.dW_qtPQ.
    //   - neither present: FIRST ITERATION -- the rung reduces to the bare
    //     interaction Z.
    const bool sec = secondary();
    const bool use_wcache = sec and _Wb_qwmm.has_value();
    // Step 1a (notes/vertex_parallelization_v2_plan.md Step 1): a dynamic-W source is
    // present iff we are NOT consuming the W-bar cache and mb_state carries dW. In the
    // GLOBAL path we build the full replicated Wdyn_qwPQ (Np^2) below; in the SECONDARY
    // path we KEEP only the tau-domain W_qtPQ here (with the head augmented) and DEFER
    // the tau -> nu transform into the per-q fold loop, reusing one W_wpos(nw_half, Np, Np)
    // buffer per q -- so the replicated (nq, nw_b, Np, Np) array is never materialized
    // (16 TB @ production Np=20k).
    // INCREMENT S7: B-L's rung is the STATIC screen W0bar, so no dynamic rung is folded
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
    // Step 1b (notes/vertex_parallelization_v2_plan.md Step 1: per-q tau-domain gather).
    // head insertion, dynamic piece, into dW(Gamma, tau) for one q's tau slab (weight
    // Re[eps_inv_head(tau)]; same Re[.] convention as Sigma_div_correction, thc_gw.icc:506):
    // add eps(it).real()*H_PQ into rows [0,nt_half) IN PLACE. Applied to the all-q slab at
    // iq_gamma (GLOBAL) or to the per-q gathered slab when iq == iq_gamma (SECONDARY) --
    // identical arithmetic either way. Returns whether the dynamic piece was actually added.
    const bool head_dyn_ok = head_ok and dyn_src and mb_state.eps_inv_head.has_value();
    if (dyn_src and head_ok) {
      if (mb_state.eps_inv_head.has_value())
        utils::check(mb_state.eps_inv_head.value().shape(0) == nt_half,
                     "vertex_t::eval_Pi_C: eps_inv_head size {} != nt_half = {}.",
                     mb_state.eps_inv_head.value().shape(0), nt_half);
      else
        app_log(1, "  [WARNING] Pi^C: dW is present but eps_inv_head is not in MBState "
                   "-- the DYNAMIC head\n"
                   "            piece is skipped (bare piece applied).");
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
      // then materialize the full nu-domain Wdyn_qwPQ. The SECONDARY path instead gathers
      // ONE q at a time inside the fold loop below (Step 1b), so neither the all-q tau
      // slab nor the replicated Np^2 nu-array is built there (16 TB @ production Np=20k).
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
        // assumption as the SOSEX cache folding, thc_sosex.icc:970-976)
        for (long l = 0; l < nw_b; ++l) {
          long lpos = std::max(l, tools.w_mirror_b(l)) - nw_b / 2;
          Wdyn_qwPQ.value()(iq, l, all, all) = W_wpos(lpos, all, all);
        }
      }
    } else if (not dyn_src and not use_wcache) {
      // NO SCREENED RUNG AVAILABLE. This used to happen on the first update of every
      // run and was treated as a harmless startup caveat; it is not (Si kp444 C = [0,8):
      // iteration-1 eps_inf 19.6 against a converged RPA 5.35, and the trajectory never
      // recovers). scr_coulomb_t::update_w now bootstraps an RPA W before the first
      // vertex-attached pass, so reaching this branch means the bootstrap did NOT run --
      // e.g. a caller invoking eval_Pi_C outside update_w. COUNTED so a test can assert
      // it never happens in a normal scf loop (notes/vertex_divergence_diagnosis.md s3).
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
    //  symmetry-free meshes nqpts == nqpts_ibz and this is the historic table)
    nda::array<long, 2> kmq(nqpts, nkpts), kpq(nqpts, nkpts);
    for (long iq = 0; iq < nqpts; ++iq) {
      for (long ik = 0; ik < nkpts; ++ik) kmq(iq, ik) = MF->qk_to_k2(iq, ik);
      for (long ik = 0; ik < nkpts; ++ik) kpq(iq, kmq(iq, ik)) = ik;   // inverse: (k-q)+q = k
    }

    // ---- Refinement 2: optional secondary-basis substitution --------------------------
    // (notes/refinement2_optionA.md section 4). The SAME kernel runs on the input set
    // (Xb, Zbar = t Z t^dag, Wbar_dyn = t dW t^dag, G_CC, window [0, nc)); the
    // head-augmented Gamma cells above downfold automatically through t. Pibar^C is
    // produced in (N_m x N_m) and UPFOLDED with the adjoint of the same t (Eq. 38) --
    // the no-leak identity (Eq. 39) is checked below as a transposition tripwire.
    // (`sec`/`use_wcache` are resolved above, at the dynamic-W source selection.)
    const bool wan = _wannier;
    const long nc = subspace_rank();
    nda::array<ComplexType, 3> Zb_qmm;
    std::optional<nda::array<ComplexType, 4>> Wbdyn_qwmm;
    // STRICT C-C EXTERNALS (theory-owner ruling, notes/refinement2_optionA.md
    // DECISION 2): dPhi_2^C/dW vanishes unless ALL FOUR pair orbital slots are in
    // range(P), so the external legs of Pi^C are C-restricted in BOTH paths via the
    // input projection -- exactly the kernel on G~ = P G P (conservation notes 1.2).
    // WINDOW: G_CC = the W-window block; WANNIER: G_CC = U^dag G U (memo C2).
    // On the FULL BZ (memo (G1)/(G2)): image points are gauge copies of the IBZ
    // blocks; trev points are the tau-pointwise transpose (see eval_Sigma_C).
    utils::check(G_tskij.shape(2) == (sym_mesh ? nkpts_ibz : nkpts),
                 "vertex_t::eval_Pi_C: G_tskij k axis = {} != {}.",
                 G_tskij.shape(2), sym_mesh ? nkpts_ibz : nkpts);
    // M1 item #5: node-share G_CC (one copy per node; see eval_Sigma_C for the rationale
    // -- built from node-shared sG_tskij, read-only by the templated-param kernel, so
    // the shared_array view binds and the values are bit-identical).
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
    // M2 item #5-finish: node-share X_C (one copy per node; see eval_Sigma_C).
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
      // Step 1a/1b (notes/vertex_parallelization_v2_plan.md Step 1: per-q tau-domain
      // gather): in the SECONDARY path neither the all-q tau slab W_qtPQ (nq*nt*Np^2) nor
      // the full nu-domain Wdyn_qwPQ (nq*nw_b*Np^2) is materialized. For each q we gather
      // ONE tau slab W_q (nt_half, Np, Np) from the distributed dW (bit-identical to the
      // all-q gather sliced at that q), head-augment it if iq == iq_gamma, then defer the
      // tau -> nu transform into the fold below (reusing ONE W_wpos per q). Bitwise-
      // equivalent to "gather all-q + augment iq_gamma + build full Wdyn_qwPQ then fold":
      // each (iq, l) folds exactly W_wpos(lpos) with the SAME PH-unfold map
      // lpos = max(l, w_mirror_b(l)) - nw_b/2 the global build applies.
      const long nw_b_sec = tools.nw_b;
      const long nw_half_sec = (nw_b_sec % 2 == 0) ? nw_b_sec / 2 : nw_b_sec / 2 + 1;
      // SECONDARY-path rank-1 tau head (Impl 2c): adds the dynamic head into the (nt_half, Np,
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
      // replicated Z(iq). In the secondary path Z_qPQ is NOT materialized (Impl 2b), so pull
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
      // --- bare core Zbar = t Z t^dag: DISTRIBUTED downfold (Impl 2b) ----------------
      // Replaces the former replicated (nq, Np, Np) Z_qPQ build + fold_core loop (320 GB @
      // production Np = 20000). thc.dZ({1, nP, nQ}) gives Z distributed over (P,Q) with q
      // NOT split; fold_Z_distributed folds each rank's own (P,Q) block with fold_core_block
      // and sums the disjoint-block partials with one final comm all_reduce. No rank ever
      // holds the full Np^2. Simpler than fold_dW_distributed: Z has no t axis => no t-pool,
      // no tau->nu, no PH-unfold. At Si test scale (nP = nQ = 1) there is one (P,Q) block ==
      // the whole array and this is BIT-IDENTICAL to the replicated fold_core (disjoint-block
      // sum is exact). The Gamma head is added into the gamma block via head_add (same
      // Z(iq_gamma) += H_PQ semantics; H_PQ is still the replicated Np^2 head, block-sliced --
      // rank-1/distributing it is a separate later micro-step).
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
          // bare piece, weight 1 (matches the global Z += H); Impl 2c rank-1 block, no dense
          // H_PQ -- head_block_add reproduces H_PQ(P,Q) bit-for-bit per element.
          vertex_secondary_detail::head_block_add(chi_g, head_c, ComplexType(1.0),
                                                  P_rng, Q_rng, A_PQ_block);
        };
        vertex_secondary_detail::fold_Z_distributed(
            dZ, _t_qmP, nqpts_ibz, Np, _Nm, iq_gamma, head_ok, z_head_add, Zb_qmm, mpi->comm);
      }
      // --- dynamic rung Wbar = t Wdyn(q,nu) t^dag: DISTRIBUTED downfold (Impl 2) --------
      // Replaces the former per-q gather+fold of dW (gather_dW_one_q built a full
      // (nt_half, Np, Np) tau slab per q -- 6.4 GB per (t,q) block @ production). Instead
      // fold_dW_distributed assembles ONLY this rank's (P,Q) block over all t (a t-pool
      // all_reduce over the disjoint t-partition -- exact), applies the Gamma head +
      // tau->nu + PH-unfold on that block, folds it with fold_core_block, and sums the
      // (P,Q)-block partials with one final comm all_reduce. NO full Np^2 slab is ever
      // held. At Si test scale (np_P = np_Q = 1) there is one (P,Q) block and this reduces
      // to the replicated fold BIT-IDENTICALLY (disjoint-block sum is exact). The forced
      // (P,Q) split is exercised only by test_vertex_dfold. (notes ... section 6b.)
      if (dyn_src) {
        Wbdyn_qwmm.emplace(nda::array<ComplexType, 4>(nqpts_ibz, tools.nw_b, _Nm, _Nm));
        auto head_add = [&](nda::MemoryArrayOfRank<2> auto&& W_bt_block, long it,
                            nda::range const& P_rng, nda::range const& Q_rng) {
          // dynamic piece, weight Re[eps_inv_head(tau)] (matches the legacy
          // += ComplexType(eps(it).real()) * H_PQ); Impl 2c rank-1 block, no dense H_PQ.
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
      // W-bar iteration cache consumption (notes/wbar_cache.md): the dynamic rung was
      // folded at update_w time on the positive half mesh; reconstruct the kernel's
      // full bosonic mesh with the SAME mirror map the legacy path applies BEFORE
      // its fold (W(-nu) = W(nu); the mirror is a pure copy, so fold-then-mirror is
      // bitwise identical to mirror-then-fold). eta[dW] diagnostics for this rung
      // were logged at fill time (cache_w); eta[Z] above covers the bare core.
      if (use_wcache) {
        auto const& Wbh = _Wb_qwmm.value();
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

    // ---- IBZ symmetry context (trivial/null on symmetry-free meshes) ------------------
    // WANNIER (memo section 2.8): thread U so d = U(Sk)^dag D U(k) (sym + Wannier compose).
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

    // ---- kernel: accumulate Pi^C(inu) over this rank's (s,k,qx) tuples ----------------
    // q->0 policy resolved above (skip_rung_gamma / head-augmented inputs). Both paths:
    // C-restricted externals; the ONLY difference is the auxiliary input set --
    // (X_C, W, Z, Np) global vs (Xb, Wbar, Zbar, N_m) secondary.
    const long naux = sec ? _Nm : Np;
    nda::array<double, 1> qx_diag(nqpts);
    nda::array<ComplexType, 4> Pi_wqMN(tools.nw_b, nqpts_ibz, naux, naux);
    Pi_wqMN() = ComplexType(0.0);
    nda::array<double, 1> phase_diag(4);
    phase_diag() = 0.0;
    // INCREMENT S7 -- B-L's W-cut. The tangent functional's dynamical W appears LINEARLY,
    // so its W-derivative kills BOTH the momentum and the frequency sum of the cut rung:
    //     P^{C,L}(q, i.nu) = -2 dPhi^L/dW = Pi^{C,0}(q, i.nu)
    // at FULL parent-normalized weight, with complete external frequency dependence. That
    // full weight is the whole point: the naive "make one rung static" functional has only
    // ONE W appearance and gives HALF the weight -- the Variant-F trap, which ONLY the
    // W-side oracle detects (verified: it breaks that oracle by exactly 2,
    // verification/static_vertex_routing_report.md section 2.3).
    // Mechanically this is the SAME kernel with the rung Z -> W0bar and NO dynamic rung:
    // the internal convolution collapses into two decoupled bubbles (the already-pinned
    // instantaneous Z-phase), so B-L needs no pole algebra here either.
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
        vertex_pi::pi_c_accumulate_w(*_ft, tools, G_CC, _Xb_skma, W0b_r, nullptr,
                                     kmq, kpq, nda::range(0, nc), Pi_wqMN,
                                     mpi->comm.rank(), mpi->comm.size(),
                                     skip_rung_gamma, &qx_diag, symc, &phase_diag);
      else
        vertex_pi::pi_c_accumulate_w(*_ft, tools, G_CC, X_C, W0b_r, nullptr,
                                     kmq, kpq, nda::range(0, nc), Pi_wqMN,
                                     mpi->comm.rank(), mpi->comm.size(),
                                     skip_rung_gamma, &qx_diag, symc, &phase_diag);
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
                                   skip_rung_gamma, &qx_diag, symc, &phase_diag);
    // M3 item #8 (notes/vertex_parallelization_M3.md): DO NOT all_reduce the full partial
    // Pi_wqMN. It is a PARTIAL (this rank's round-robin tuple/q_ext contribution); the
    // upfold (t^dag Pibar t) and the tau conversion are LINEAR and commute with the rank
    // sum, so they are applied to the PARTIAL and the result is REDUCE-SCATTERED directly
    // into the RPA grid (reduce_scatter_into). This removes the full-array all_reduce and
    // the persistent full replicated Pi_up / Pi_tqMN. Only qx_diag (tiny) is all_reduced.
    mpi->comm.all_reduce_in_place_n(qx_diag.data(), qx_diag.size(), std::plus<>{});

    // ---- KERNEL SCALES (2026-07-27 divergence hunt) -----------------------------------
    // Pi^C is MULTILINEAR in (G,G,G,G,W): bounded inputs => Lipschitz, so a ~1e-4 change in
    // G cannot produce a 1e8 change in Pi^C. Measured on the Si kp444 C=[0,4) break, the
    // checkpointed G feeding the exploding iteration has max|G_CC| = 0.98259 against
    // 0.98246 the iteration before, and W is bounded there (||eps^-1|| = 1.11). The
    // explosion is therefore INTERNAL to this routine. These norms split it three ways:
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

    // ---- no-leak tripwire (Eq. 39; DIAGNOSTIC, no gate) -- needs the REDUCED Pi ---------
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
        // full replicated Z(iq) for the bare-Z pairing: pulled locally (secondary path no
        // longer materializes Z_qPQ; Impl 2b). thc.Z is collective and this loop is uniform
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
      // fold_dW_distributed). The full Np^2 upfold partial (~5-11 GB/rank at production Np)
      // is never materialized; each rank upfolds ONLY its owned (P,Q) block directly into
      // dPi_C_tqPQ.local(). First sum the SMALL N_m^2 partial across comm (upfold+tau are
      // linear, so the rank sum before the upfold == reduce-scatter after -- same identity
      // the removed reduce_scatter_into cited).
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
      // GLOBAL: Pi_wqMN is already Np^2 (a partial). tau-convert the partial and reduce-
      // scatter the full Np^2 partial into the RPA grid: every output block is summed across
      // ranks and lands ONLY on its owner (no full-array all_reduce). On one rank this is a
      // bit-identical copy.
      nda::array<ComplexType, 4> Pi_tqMN(nt_half, nqpts_ibz, Np, Np);
      vertex_pi::pi_w_to_code_tau(*_ft, tools, Pi_wqMN, Pi_tqMN);
      vertex_redist_detail::reduce_scatter_into(Pi_tqMN, dPi_C_tqPQ, mpi->comm);
    }

    {
      // VERTEX RAMP / SCALE. Both cuts carry the SAME lambda, which is exactly
      // Phi_2^C -> lambda Phi_2^C: the approximation acts on the GENERATING FUNCTIONAL,
      // not on the already-cut Sigma/P, so Phi-derivability and the conservation
      // identity survive at every lambda (notes/CLAUDE.md section 12 / section 8).
      // Used to walk the vertex in continuously: P^C is not sign-definite, so a full-
      // strength vertex can push eps = I - Z.Pi through zero and break the W-Dyson
      // solve (notes/vertex_divergence_diagnosis.md section 2). Ramping finds the
      // largest lambda whose solution still has a positive-definite eps.
      const double lam = vertex_scale();
      if (lam != 1.0) {
        dPi_C_tqPQ.local() *= ComplexType(lam);
        app_log(1, "  [ISDF-Vertex] Pi^C scaled by lambda = {:.4f} (ramp iteration {} of "
                   "{})", lam, _vertex_iter, _ramp_iters);
      }
    }
    {
      // NaN/Inf guard on THIS rank's owned block (the full array is no longer materialized)
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
    mpi->comm.barrier();

    return dPi_C_tqPQ;
  }

  void vertex_t::cache_w(MBState &mb_state, THC_ERI auto const &thc) {
    decltype(nda::range::all) all;
    utils::check(active() and secondary(),
                 "vertex_t::cache_w: requires an ACTIVE vertex in isdf mode \"secondary\" "
                 "(the global path retains the full dW instead; notes/wbar_cache.md).");
    utils::check(mb_state.dW_qtPQ.has_value(),
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

    // IBZ symmetry (notes/vertex_ibz_symmetry.md section 3.7): the fill runs over
    // IBZ q only -- which is exactly the cache's q-keyed first axis; consumption at
    // non-IBZ transfers goes through the kernels' symmetry context. No further
    // change is needed here (the layout anticipated this extension).
    (void)nqpts; (void)nkpts; (void)nkpts_ibz;

    vertex_pi::iaft_tools tools(*_ft);
    const long nw_b = tools.nw_b;
    const long nw_half = (nw_b % 2 == 0) ? nw_b / 2 : nw_b / 2 + 1;
    auto gs = mb_state.dW_qtPQ.value().global_shape();
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
    // M2 item #5-finish: node-share X_skPa (one copy per node; see eval_Sigma_C).
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

    // ---- q->0 rung policy: SAME resolution as eval_Pi_C (q0_head_treatment.md) --------
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
      head_ok = vertex_head_detail::build_head_rank1(thc, iq_gamma, nkpts, H_PQ);
      if (not head_ok)
        app_log(1, "  [WARNING] cache_w: gygi head insertion requested but head data "
                   "are unusable\n"
                   "            (madelung == 0 or empty basis_head) -- caching WITHOUT "
                   "the analytic head\n"
                   "            (equivalent to policy \"ignore_g0\").");
    }

    // ---- replicate dW(tau), augment the Gamma head, transform to the half nu mesh ----
    // Identical arithmetic to the legacy fold-at-consumption path (eval_Pi_C):
    // augment BEFORE tau_to_w_PHsym, per-q transform on the same (nt_half, Np, Np)
    // tau-storage slices.
    nda::array<ComplexType, 4> W_wq(nqpts_ibz, nw_half, Np, Np);
    {
      // M1 item #1: gather the RPA-grid dW into the replicated tau slab (bit-identical).
      nda::array<ComplexType, 4> W_qtPQ = vertex_redist_detail::gather_dW_replicated(
          mb_state.dW_qtPQ.value(), mpi->comm, nqpts_ibz, nt_half, Np);

      if (head_ok) {
        if (mb_state.eps_inv_head.has_value()) {
          auto& eps = mb_state.eps_inv_head.value();
          utils::check(eps.shape(0) == nt_half,
                       "vertex_t::cache_w: eps_inv_head size {} != nt_half = {}.",
                       eps.shape(0), nt_half);
          for (long it = 0; it < nt_half; ++it)
            W_qtPQ(iq_gamma, it, all, all) += ComplexType(eps(it).real()) * H_PQ;
          app_log(1, "  cache_w head insertion: dynamic piece applied to dW(Gamma, tau) "
                     "with eps_inv_head(tau=0) = {}\n"
                     "  (SAME-iteration eps_inv_head, captured at fill time)",
                  eps(0).real());
        } else {
          app_log(1, "  [WARNING] cache_w: dW is present but eps_inv_head is not in "
                     "MBState -- the DYNAMIC head\n"
                     "            piece is skipped for the cached rung.");
        }
      }

      // M3 item #7 (notes/vertex_parallelization_M3.md): the per-q tau_to_w_PHsym
      // transform was SERIAL AND REDUNDANT on every rank. Distribute over mpi->comm
      // (each rank transforms its q subset), then GATHER W_wq (zero-init + all_reduce =
      // exact partition gather; the eta diagnostic below reads the full-q W_wq). The fold
      // is likewise distributed below. Per-rank transform+fold WORK drops ~1/P.
      W_wq() = ComplexType(0.0);
      for (long iq = mpi->comm.rank(); iq < nqpts_ibz; iq += mpi->comm.size()) {
        auto W_t = W_qtPQ(iq, nda::ellipsis{});
        auto W_w = W_wq(iq, nda::ellipsis{});
        _ft->tau_to_w_PHsym(W_t, W_w);
      }
      mpi->comm.all_reduce_in_place_n(W_wq.data(), W_wq.size(), std::plus<>{});
    }

    // ---- eta(q) diagnostics on the rung ACTUALLY cached (test-scale gate) ------------
    // (moved here from the consumption site: the global-basis Wdyn no longer exists
    //  at eval time in the cached mode; same labels/slices as before)
    if (ns * nkpts * nc * nc <= 4096) {
      const long lpos0 = std::max(tools.m0, tools.w_mirror_b(tools.m0)) - nw_b / 2;
      const long lposm = std::max(nw_b - 1, tools.w_mirror_b(nw_b - 1)) - nw_b / 2;
      vertex_secondary_detail::eta_max_over_q(
          "dW(nu_0)", X_glob, orb0_glob, nc, _Xb_skma, _t_qmP, kmq,
          [&](long iq) { return W_wq(iq, lpos0, all, all); });
      vertex_secondary_detail::eta_max_over_q(
          "dW(nu_max)", X_glob, orb0_glob, nc, _Xb_skma, _t_qmP, kmq,
          [&](long iq) { return W_wq(iq, lposm, all, all); });
    } else {
      app_log(2, "  Refinement 2: eta diagnostic skipped (N_pair = {} > 4096).",
              ns * nkpts * nc * nc);
    }

    // ---- fold on the half mesh: Wbar(q, nu) = t(q) Wdyn(q, nu) t(q)^dag --------------
    // (same fold_core gemms on the same values as the legacy full-mesh fold; the
    //  mirrored nu points are pure copies, reconstructed at consumption)
    _Wb_qwmm.emplace(nda::array<ComplexType, 4>(nqpts_ibz, nw_half, _Nm, _Nm));
    {
      // M3 item #7: distribute the per-q fold over mpi->comm; each q is folded on exactly
      // ONE rank (so the fold reduction is byte-identical to serial FOR THAT q -- no
      // per-q reassociation), then GATHER (zero-init + all_reduce = exact partition). The
      // ONLY reduction is the zero-padded gather (bit-identical). Per-rank fold work ~1/P.
      _Wb_qwmm.value()() = ComplexType(0.0);
      nda::array<ComplexType, 2> tmp(_Nm, Np);
      long my_nfold = 0;
      for (long iq = mpi->comm.rank(); iq < nqpts_ibz; iq += mpi->comm.size()) {
        ++my_nfold;
        auto t_q = _t_qmP(iq, all, all);
        for (long lp = 0; lp < nw_half; ++lp)
          vertex_secondary_detail::fold_core(t_q, W_wq(iq, lp, all, all), tmp,
                                             _Wb_qwmm.value()(iq, lp, all, all));
      }
      mpi->comm.all_reduce_in_place_n(_Wb_qwmm.value().data(), _Wb_qwmm.value().size(),
                                      std::plus<>{});
      const long total_fold = mpi->comm.all_reduce_value(my_nfold, std::plus<>{});
      app_log(2, "  Refinement 2 W-bar fold distributed over {} ranks: this rank folded "
                 "{} of {} q-points (~1/P work).", mpi->comm.size(), my_nfold, total_fold);
    }

    // ---- footprint: the memory point of the exercise ---------------------------------
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
                          dArray_t const &dPi_rpa_tqPQ) {
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
    utils::check(gs[1] == nqpts_ibz and gs[2] == Np and gs[3] == Np and gs[0] == nt_half_ft,
                 "vertex_t::build_w0: unexpected Pi_RPA global shape ({}, {}, {}, {}); "
                 "expected ({}, {}, {}, {}).",
                 gs[0], gs[1], gs[2], gs[3], nt_half_ft, nqpts_ibz, Np, Np);

    // ITERATION-LOCAL lifetime (plan section 2.3): drop last iteration's objects up
    // front, so no static-rung state can ever be read across an iteration boundary
    // (a stale W0 would silently break the FD oracles of S5/S9).
    reset_w0();

    app_log(1, "\n  [ISDF-Vertex] static rung W0[G] (increment S2; "
               "notes/static_vertex_implementation_plan.md section 2.2)\n"
               "  W0(q) = Z(q) + dW(q, i.nu = 0) from the SAME-ITERATION RPA "
               "polarizability -- no lag,\n"
               "  no Pi^C content (decision D2). Grid (nt_half, nq, Np) = ({}, {}, {}), "
               "rung mode = {}.",
            gs[0], nqpts_ibz, Np, rung_str());

    // ---- (P,Q)-block layout: q unsplit, (P,Q) over ALL ranks (thc.dZ({1,nP,nQ}) ------
    // layout; the one fold_Z_distributed and the slate 2D ops both accept, and the one
    // the plan's section 3 mandates for a 320 GB-class nq*Np^2 object).
    const long np_ranks = mpi->comm.size();
    std::array<long, 3> w0_pgrid = {1, 1, 1};
    w0_pgrid[1] = utils::find_proc_grid_min_diff(np_ranks, Np, Np);
    w0_pgrid[2] = np_ranks / w0_pgrid[1];
    const std::array<long, 3> w0_bsize = {1, 1, 1};
    const std::array<long, 4> f0_pgrid = {1, 1, w0_pgrid[1], w0_pgrid[2]};
    const std::array<long, 4> f0_bsize = {1, 1, w0_bsize[1], w0_bsize[2]};

    // ---- step 1: the i.nu = 0 row of Pi_RPA on that layout ---------------------------
    auto dW0_1qPQ = make_distributed_array<Arr4>(
        mpi->comm, f0_pgrid, {1, nqpts_ibz, Np, Np}, f0_bsize);
    {
      auto R_t = vertex_w0_detail::nu0_transform_row(*_ft);
      vertex_w0_detail::extract_nu0_row(dPi_rpa_tqPQ, R_t, dW0_1qPQ);
    }

    // ---- step 2: the SINGLE-FREQUENCY THC Dyson, per q ------------------------------
    // dW0(q) = ([I - Z(q).Pi0(q)]^{-1} - I) Z(q): the scr_coulomb_t::dyson_W_in_place
    // algebra (scr_coulomb_t.cpp:310-338) with the frequency loop removed. Same slate
    // primitives, same operand order, same in-place convention (the array that came in
    // holding Pi0 goes out holding dW0), so the plain-GW self-slice identity holds to
    // machine precision -- that is the S2 gate (i).
    auto dZ = thc.dZ(w0_pgrid, w0_bsize);
    auto P_rng = dW0_1qPQ.local_range(2);
    auto Q_rng = dW0_1qPQ.local_range(3);
    utils::check(dZ.local_range(1) == P_rng and dZ.local_range(2) == Q_rng,
                 "vertex_t::build_w0: Z and Pi0 do not share the (P,Q) block partition.");
    {
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
    }

    // ---- step 3: the q->0 head policy AT i.nu = 0 (q0_head_treatment.md section 3) ---
    // ONE policy, ONE W0, so every later appearance of the rung carries the same head
    // (plan section 2.2). "v1_skip"/"ignore_g0" store the regularized body only; the
    // gygi class additionally inserts the analytic rank-1 head, whose i.nu = 0 dynamic
    // weight Re[eps^{-1}_head(i.nu=0)] is extracted from THIS RPA dW0 -- so the rung and
    // its head factor carry the same iteration tag by construction (memo section 1.6),
    // instead of the previous iteration's mb_state.eps_inv_head that is still standing
    // at this point of update_w.
    // the Gamma index + (in the secondary path) the lazy Option-A transfer maps. Built
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
      const double xi = MF->madelung();
      auto chi = thc.basis_head();                              // (nqpts_ibz, Np)
      utils::check(chi.shape(0) > iq_gamma and chi.shape(1) == Np,
                   "vertex_t::build_w0: basis_head shape mismatch (({}, {}) vs iq_gamma = "
                   "{}, Np = {}).", chi.shape(0), chi.shape(1), iq_gamma, Np);
      double chi_max = 0.0;
      for (long P = 0; P < Np; ++P) chi_max = std::max(chi_max, std::abs(chi(iq_gamma, P)));
      if (xi != 0.0 and chi_max != 0.0) {
        chi_g = nda::array<ComplexType, 1>(chi(iq_gamma, all));
        _w0_head_c = ComplexType(double(nkpts) * xi);            // build_head_rank1's c
        _w0_head_applied = true;
      } else {
        app_log(1, "  [WARNING] W0: gygi head insertion requested but the head data are "
                   "unusable\n"
                   "            (madelung == 0 or empty basis_head) -- proceeding WITHOUT "
                   "the analytic head\n"
                   "            (equivalent to policy \"ignore_g0\").");
      }
      if (_w0_head_applied) {
        // the DYNAMIC head weight at i.nu = 0, from the freshly built RPA dW0 (the v2
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
        // (H1) in the memo's two pieces, applied in the same order and with the same
        // per-element arithmetic as the pinned Z / dW augmentations: bare weight 1 into
        // the Z part, dynamic weight Re[eps^{-1}_head(i.nu=0)] into the dW0 part.
        vertex_secondary_detail::head_block_add(chi_g, _w0_head_c, ComplexType(1.0),
                                                P_rng, Q_rng, W0(iq_gamma, all, all));
        vertex_secondary_detail::head_block_add(chi_g, _w0_head_c,
                                                ComplexType(_w0_eps_head),
                                                P_rng, Q_rng, W0(iq_gamma, all, all));
      }
    }
    dW0_1qPQ.reset();

    // ---- W0bar = t W0 t^dag: the DISTRIBUTED one-row fold ---------------------------
    // fold_Z_distributed IS the one-row variant of fold_dW_distributed (Z has no tau
    // axis => no t-pool, no tau->nu, no PH-unfold), and W0 has exactly Z's shape and
    // layout, so the "restricted to the single i.nu = 0 row" fold of plan section 2.2 is
    // this call verbatim. head_at_gamma = false: the head is ALREADY inside W0 (one W0,
    // one policy), so re-adding it here would double count.
    auto no_head = [](nda::MemoryArrayOfRank<2> auto&&, nda::range const&,
                      nda::range const&) {};
    if (secondary()) {
      _W0b_qmm.emplace(Arr3(nqpts_ibz, _Nm, _Nm));
      vertex_secondary_detail::fold_Z_distributed(
          _W0_qPQ.value(), _t_qmP, nqpts_ibz, Np, _Nm, iq_gamma, false, no_head,
          _W0b_qmm.value(), mpi->comm);
    } else {
      // GLOBAL-aux reference path (small scale only, plan section 2.5): the "secondary"
      // rung IS the global one, N_m == Np, t = identity. Gather the distributed blocks
      // (zero-pad + all_reduce over a PARTITION = an exact gather, no reassociation) --
      // the same replication class this path already accepts for its Z_qPQ.
      _W0b_qmm.emplace(Arr3(nqpts_ibz, Np, Np));
      auto &Wb = _W0b_qmm.value();
      Wb() = ComplexType(0.0);
      Wb(_W0_qPQ.value().local_range(0), P_rng, Q_rng) = _W0_qPQ.value().local();
      mpi->comm.all_reduce_in_place_n(Wb.data(), Wb.size(), std::plus<>{});
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
    mpi->comm.barrier();
  }

  // template instantiations
  template void vertex_t::eval_Sigma_C(MBState&, const thc_reader_t&);
  template void vertex_t::cache_w(MBState&, const thc_reader_t&);
  template long vertex_t::ensure_secondary_basis(MBState&, const thc_reader_t&);
  template void vertex_t::build_w0(
      MBState&, const thc_reader_t&,
      memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator> const&);

  template memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>
  vertex_t::eval_Pi_C(MBState&, const thc_reader_t&,
                      std::array<long, 4>, std::array<long, 4>, std::array<long, 4>);

}  // solvers
}  // methods
