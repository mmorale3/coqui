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

#include "utilities/check.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/ERI/thc.h"    // Refinement 2: restricted-range ISDF point selection
#include "vertex_t.h"
#include "vertex_pi.icc"
#include "vertex_sigma.icc"  // ISDF-Vertex Phase 1c: fused G^3 W^2 Sigma^C kernel

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
    inline void build_pair_matrix(nda::array<ComplexType, 4> const& X_skua,
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
    template<typename CoreF>
    double eta_max_over_q(char const* label,
                          nda::array<ComplexType, 4> const& X_skPa, long orb0, long nc,
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
                     double isdf_svd_tol):
    _ft(ft), _vertex_type(std::move(vertex_type)), _band_window(band_window),
    _isdf_mode(std::move(isdf_mode)), _isdf_rank(isdf_rank), _isdf_svd_tol(isdf_svd_tol) {

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
                 "  Subspace C band window   = [{}, {})\n"
                 "  Subspace C size          = {} orbitals (nbnd = {})\n"
                 "  Cuts                     = Sigma^C (G3W2) + Pi^C (G4W), always both\n"
                 "  q->0 rung policy         = {} (notes/q0_head_treatment.md)\n"
                 "  Auxiliary basis          = {}{}\n"
                 "  Status                   = Pi^C kernel ACTIVE (Phase 1d); Sigma^C status is\n"
                 "                             reported by eval_Sigma_C at evaluation time\n",
              _vertex_type, _band_window.first(), _band_window.last(),
              _band_window.size(), nbnd, _div_treatment, _isdf_mode,
              secondary() ? std::string(" (Refinement 2: requested N_m = ") +
                            (_isdf_rank > 0 ? std::to_string(_isdf_rank)
                                            : std::string("auto = nc^2*nk")) +
                            ", svd_tol(B) = " + std::to_string(_isdf_svd_tol) +
                            "; notes/refinement2_optionA.md)"
                          : std::string(" (global THC, dimension Np)"));
    } else {
      app_log(1, "\nvertex_t: vertex_type = \"{}\" with an empty vertex_band_window: "
                 "C = empty set, so the vertex contributes nothing and the "
                 "calculation reduces to plain scGW exactly.\n", _vertex_type);
    }
  }

  void vertex_t::build_secondary_basis(THC_ERI auto const &thc,
                                       nda::array<ComplexType, 4> const &X_skPa,
                                       nda::array<long, 2> const &kmq, long iq_gamma) {
    if (_secondary_ready) return;
    decltype(nda::range::all) all;
    auto mpi = thc.mpi();
    auto MF = thc.MF();
    const long ns = X_skPa.shape(0), nkpts = X_skPa.shape(1), Np = X_skPa.shape(2);
    const long nqpts = kmq.shape(0);
    const long nc = _band_window.size();
    const long Npair = ns * nkpts * nc * nc;   // the pair index carries momentum
                                               // (CLAUDE.md section 2, invariant 4)
    const long Nm_req = (_isdf_rank > 0) ? _isdf_rank : nc * nc * nkpts;
    utils::check(Nm_req <= Npair,
                 "vertex_t::build_secondary_basis: vertex_isdf_rank = {} exceeds the "
                 "subspace pair rank N_pair = ns*nk*nc^2 = {}; the secondary basis "
                 "cannot usefully exceed the space it represents.", Nm_req, Npair);

    app_log(1, "\n  Refinement 2: building the secondary ISDF basis on C = [{}, {}) "
               "(notes/refinement2_optionA.md)\n"
               "  requested N_m = {}, svd_tol(B) = {}, N_pair (per q, spin-stacked) = {}\n",
            _band_window.first(), _band_window.last(), Nm_req, _isdf_svd_tol, Npair);

    // ---- restricted-range ISDF point selection (collective on thc.mpi()->comm) --------
    // Private methods::thc builder on the SAME MF/mpi context; thresh = 1e-13 makes the
    // pivoted Cholesky STOP cleanly at the numerical rank of the C pair-density metric
    // instead of hard-aborting at the 1e-14 guard (memo DECISION 4). The greedy pivot
    // order also makes rank scans nested (first N of a larger selection = selection of N).
    {
      ptree pt;
      pt.put("thresh", 1e-13);
      // the blocked pivoted Cholesky is not robust at near-zero thresholds (thc.icc
      // forces block_size = 1 itself when thresh == 0.0; at thresh = 1e-13 with the
      // default block 8 it produces NaN residuals) -- use the serial pivot order,
      // which is also the exactly-nested greedy order the rank scans rely on
      pt.put("chol_block_size", 1);
      methods::thc builder(MF.get(), *mpi, pt, /*print_metadata*/ false);
      auto [ipts, dXa, dXb] = builder.interpolating_points<HOST_MEMORY>(
          int(iq_gamma), int(Nm_req), _band_window, _band_window);
      (void)dXb;   // empty optional for a_range == b_range at Gamma (single_psi path)
      const long Nm = ipts.extent(0);
      utils::check(Nm > 0,
                   "vertex_t::build_secondary_basis: point selection returned 0 points.");
      if (Nm < Nm_req)
        app_log(1, "  [NOTE] Refinement 2: point selection stopped at N_m = {} "
                   "(< requested {}):\n"
                   "         the C pair-density metric is numerically rank-deficient below "
                   "thresh = 1e-13;\n"
                   "         using the returned rank.", Nm, Nm_req);
      auto gs = dXa.global_shape();
      utils::check(gs[0] == ns and gs[1] == nkpts and gs[2] == nc and gs[3] == Nm,
                   "vertex_t::build_secondary_basis: unexpected collocation shape "
                   "({}, {}, {}, {}); expected ({}, {}, {}, {}).",
                   gs[0], gs[1], gs[2], gs[3], ns, nkpts, nc, Nm);
      // gather the distributed collocation, then transpose to the kernels' (aux, orb)
      // layout. Any fixed per-point phase/scale convention of the selection output is
      // absorbed by the least-squares transfer (memo section 2.4).
      nda::array<ComplexType, 4> Xa(ns, nkpts, nc, Nm);
      Xa() = ComplexType(0.0);
      Xa(dXa.local_range(0), dXa.local_range(1), dXa.local_range(2), dXa.local_range(3)) =
          dXa.local();
      mpi->comm.all_reduce_in_place_n(Xa.data(), Xa.size(), std::plus<>{});
      _Xb_skma = nda::array<ComplexType, 4>(ns, nkpts, Nm, nc);
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nkpts; ++ik)
          for (long a = 0; a < nc; ++a)
            for (long m = 0; m < Nm; ++m)
              _Xb_skma(is, ik, m, a) = Xa(is, ik, a, m);
      _Nm = Nm;
    }

    // ---- per-q Option-A transfer t(q) = s(q)^+ B(q)^dag C(q) (theoryB Eq. 36) ---------
    // Solved as the truncated-SVD least squares min || B t - C ||_F directly on B
    // (numerically equivalent, better conditioned: rcond acts on sv(B); the metric
    // s = B^dag B is thereby regularized at rcond^2). The explicit s^{-1} is REQUIRED:
    // the code's THC body contractions are metric-free (coqui_conventions_confirmed.md).
    _t_qmP = nda::array<ComplexType, 3>(nqpts, _Nm, Np);
    double cond_s_max = 0.0, fit_max = 0.0;
    long disc_max = 0;
    {
      nda::array<ComplexType, 2> B_Im(Npair, _Nm), C_IP(Npair, Np);
      // gelss needs F-layout: keep the TRANSPOSES in C-layout and pass transposed views
      nda::array<ComplexType, 2> BT(_Nm, Npair), CT(Np, Npair), Cf(Npair, Np);
      nda::array<double, 1> sv(std::min(Npair, _Nm));
      for (long iq = 0; iq < nqpts; ++iq) {
        vertex_secondary_detail::build_pair_matrix(_Xb_skma, 0, nc, kmq, iq, B_Im);
        vertex_secondary_detail::build_pair_matrix(X_skPa, _band_window.first(), nc,
                                                   kmq, iq, C_IP);
        for (long I = 0; I < Npair; ++I) {
          for (long m = 0; m < _Nm; ++m) BT(m, I) = B_Im(I, m);
          for (long P = 0; P < Np; ++P) CT(P, I) = C_IP(I, P);
        }
        int rank = 0;
        int info = nda::lapack::gelss(nda::transpose(BT), nda::transpose(CT), sv,
                                      _isdf_svd_tol, rank);
        utils::check(info == 0, "vertex_t::build_secondary_basis: gelss failed "
                                "(info = {}) at iq = {}.", info, iq);
        // solution rows live in the first N_m rows of the (transposed-view) rhs
        for (long m = 0; m < _Nm; ++m)
          for (long P = 0; P < Np; ++P) _t_qmP(iq, m, P) = CT(P, m);
        // diagnostics: cond(s) = cond(B)^2, discarded sv, fit residual ||Bt-C||/||C||
        const double smax = sv(0), smin = sv(sv.size() - 1);   // descending order
        const double cond_s = (smax / std::max(smin, 1e-300)) *
                              (smax / std::max(smin, 1e-300));
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
        disc_max = std::max(disc_max, discarded);
        fit_max = std::max(fit_max, fit);
      }
    }
    app_log(1, "  Refinement 2 secondary basis READY: N_m = {} (pair rank {} per q), "
               "max_q cond(s) = {},\n"
               "  max_q discarded sv = {}, max_q fit residual ||Bt - C||_F/||C||_F = {}\n",
            _Nm, Npair, cond_s_max, disc_max, fit_max);
    _secondary_ready = true;
  }

  void vertex_t::eval_Sigma_C(MBState &mb_state, THC_ERI auto const &thc) {
    utils::check(active(), "vertex_t::eval_Sigma_C: called while the vertex is inactive. "
                           "Callers must guard vertex calls with vertex_t::active().");
    utils::check(mb_state.sG_tskij.has_value(),
                 "vertex_t::eval_Sigma_C: sG_tskij is not initialized in MBState.");
    utils::check(mb_state.sSigma_tskij.has_value(),
                 "vertex_t::eval_Sigma_C: sSigma_tskij is not initialized in MBState.");
    utils::check(mb_state.dW_qtPQ.has_value(),
                 "vertex_t::eval_Sigma_C: dW_qtPQ is not initialized in MBState.");

    decltype(nda::range::all) all;
    auto mpi = thc.mpi();
    auto MF = thc.MF();
    const long nkpts = MF->nkpts();
    const long nqpts = MF->nqpts();
    const long nkpts_ibz = MF->nkpts_ibz();
    const long nqpts_ibz = MF->nqpts_ibz();
    const long Np = thc.Np();
    const long nbnd = MF->nbnd();

    // SYMMETRY POLICY (Phase 1c): symmetry-free meshes only; abort loudly rather than
    // produce silently wrong results (the audit-sec.7 IBZ unfolding is a follow-up).
    utils::check(nqpts == nqpts_ibz and nkpts == nkpts_ibz and nqpts == nkpts,
                 "vertex_t::eval_Sigma_C: vertex + k-space symmetry is not supported yet "
                 "(nqpts = {}, nqpts_ibz = {}, nkpts = {}, nkpts_ibz = {}). "
                 "Rerun with symmetry disabled in the mean-field input.",
                 nqpts, nqpts_ibz, nkpts, nkpts_ibz);
    {
      auto kp_trev = MF->kp_trev();
      for (long ik = 0; ik < nkpts; ++ik)
        utils::check(not kp_trev(ik),
                     "vertex_t::eval_Sigma_C: vertex + time-reversal-reduced meshes are not supported yet.");
    }
    utils::check(MF->npol() == 1, "vertex_t::eval_Sigma_C: npol != 1 is not supported.");
    utils::check(_ft->basis() == imag_axes_ft::dlr_basis,
                 "vertex_t::eval_Sigma_C: the fused G3W2 kernel requires the DLR IAFT backend "
                 "(iaft basis = \"dlr\"); the IR backend is not supported.");

    auto G_tskij = mb_state.sG_tskij.value().local();
    auto& sSigma_tskij = mb_state.sSigma_tskij.value();
    const long nt = G_tskij.shape(0);
    const long ns = G_tskij.shape(1);
    const long nt_half = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;
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
    nda::array<ComplexType, 4> X_skPa(ns, nkpts, Np, nbnd);
    for (long is = 0; is < ns; ++is)
      for (long ik = 0; ik < nkpts; ++ik)
        X_skPa(is, ik, all, all) = thc.X(is, 0, ik);

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
    nda::array<ComplexType, 3> Z_qPQ(nqpts, Np, Np);
    for (long iq = 0; iq < nqpts; ++iq)
      Z_qPQ(iq, all, all) = thc.Z(int(iq));

    // head insertion, bare piece (weight 1) into Z(Gamma)
    nda::array<ComplexType, 2> H_PQ(Np, Np);
    bool head_ok = false;
    if (head_insertion) {
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
    // PH-symmetric in tau, W(beta-t) = W(t).
    nda::array<ComplexType, 4> Wt_qtPQ(nqpts, nt, Np, Np);
    {
      auto& dW = mb_state.dW_qtPQ.value();
      auto gs = dW.global_shape();
      utils::check(gs[0] == nqpts_ibz and gs[1] == nt_half and gs[2] == Np and gs[3] == Np,
                   "vertex_t::eval_Sigma_C: unexpected dW_qtPQ global shape.");
      nda::array<ComplexType, 4> W_half(nqpts, nt_half, Np, Np);
      W_half() = ComplexType(0.0);
      W_half(dW.local_range(0), dW.local_range(1), dW.local_range(2), dW.local_range(3)) = dW.local();
      mpi->comm.all_reduce_in_place_n(W_half.data(), W_half.size(), std::plus<>{});

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
    const long nc = _band_window.size();
    nda::array<ComplexType, 3> Zb_qmm;
    nda::array<ComplexType, 4> Wb_qtmm;
    // STRICT C-C EXTERNALS (theory-owner ruling, notes/refinement2_optionA.md
    // DECISION 2): in Phi_2^C ALL FOUR G-lines -- including the cut one -- are
    // C-restricted, so Sigma^C = dPhi/dG is nonzero ONLY on the C-C block. BOTH paths
    // run the kernel with C-restricted externals (G_CC + the C columns of the
    // collocation); the full-range extension of the kernel formula is well-defined but
    // is NOT dPhi/dG, and accumulating it would break strict Phi-derivability.
    nda::array<ComplexType, 5> G_CC(nt, ns, nkpts, nc, nc);
    G_CC = G_tskij(all, all, all, _band_window, _band_window);
    app_log(2, "  Sigma^C externals restricted to the C-C block a, b in [{}, {}) "
               "(strict Phi cut; notes/refinement2_optionA.md DECISION 2).",
            _band_window.first(), _band_window.last());
    if (sec) {
      build_secondary_basis(thc, X_skPa, kmq, iq_gamma);
      app_log(1, "  Refinement 2: Sigma^C runs in the SECONDARY basis (N_m = {} vs "
                 "Np = {}); externals a, b in C.", _Nm, Np);
      // eta diagnostics (Eq. 40) on the rung arrays ACTUALLY consumed (test-scale gate)
      if (ns * nkpts * nc * nc <= 4096) {
        vertex_secondary_detail::eta_max_over_q(
            "Z", X_skPa, _band_window.first(), nc, _Xb_skma, _t_qmP, kmq,
            [&](long iq) { return Z_qPQ(iq, all, all); });
        vertex_secondary_detail::eta_max_over_q(
            "dW(tau_0)", X_skPa, _band_window.first(), nc, _Xb_skma, _t_qmP, kmq,
            [&](long iq) { return Wt_qtPQ(iq, 0, all, all); });
        vertex_secondary_detail::eta_max_over_q(
            "dW(tau_mid)", X_skPa, _band_window.first(), nc, _Xb_skma, _t_qmP, kmq,
            [&](long iq) { return Wt_qtPQ(iq, nt / 2, all, all); });
      } else {
        app_log(2, "  Refinement 2: eta diagnostic skipped (N_pair = {} > 4096).",
                ns * nkpts * nc * nc);
      }
      // fold the cores (frequency-slice-wise; t is frequency-independent)
      Zb_qmm = nda::array<ComplexType, 3>(nqpts, _Nm, _Nm);
      Wb_qtmm = nda::array<ComplexType, 4>(nqpts, nt, _Nm, _Nm);
      nda::array<ComplexType, 2> tmp(_Nm, Np);
      for (long iq = 0; iq < nqpts; ++iq) {
        auto t_q = _t_qmP(iq, all, all);
        vertex_secondary_detail::fold_core(t_q, Z_qPQ(iq, all, all), tmp,
                                           Zb_qmm(iq, all, all));
        for (long it = 0; it < nt; ++it)
          vertex_secondary_detail::fold_core(t_q, Wt_qtPQ(iq, it, all, all), tmp,
                                             Wb_qtmm(iq, it, all, all));
      }
    }

    // ---- fused kernel (round-robin over (s,k,qx); result all-reduced inside) ----------
    // Both paths: C-restricted externals; the ONLY difference is the auxiliary input
    // set -- (X_C, W, Z, Np) global vs (Xb, Wbar, Zbar, N_m) secondary.
    nda::array<ComplexType, 5> Sigma_C(nt, ns, nkpts, nc, nc);
    if (sec)
      vertex_detail::eval_sigma_C_g3w2_nosym(*_ft, mpi->comm, nda::range(0, nc), G_CC,
                                             _Xb_skma, Wb_qtmm, Zb_qmm, kmq, qmin,
                                             iq_gamma, skip_rung_gamma, Sigma_C);
    else {
      nda::array<ComplexType, 4> X_C(ns, nkpts, Np, nc);
      X_C = X_skPa(all, all, all, _band_window);
      vertex_detail::eval_sigma_C_g3w2_nosym(*_ft, mpi->comm, nda::range(0, nc), G_CC,
                                             X_C, Wt_qtPQ, Z_qPQ, kmq, qmin, iq_gamma,
                                             skip_rung_gamma, Sigma_C);
    }
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

    // accumulate on top of the GW self-energy: Sigma <- Sigma + Sigma^C
    // (Sigma_C is identical on every rank after the kernel's all_reduce; hermitization
    //  stays downstream in scf_driver). Strict Phi cut: the C-C block only, both paths.
    if (mb_state.mpi->node_comm.root())
      sSigma_tskij.local()(all, all, all, _band_window, _band_window) += Sigma_C;
    mb_state.mpi->comm.barrier();
  }

  auto vertex_t::eval_Pi_C(MBState &mb_state, THC_ERI auto const &thc,
                           shape_t<4> pi_pgrid, shape_t<4> pi_bsize, shape_t<4> pi_gshape)
  -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>
  {
    decltype(nda::range::all) all;
    utils::check(active(), "vertex_t::eval_Pi_C: called while the vertex is inactive. "
                           "Callers must guard vertex calls with vertex_t::active().");
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

    // SYMMETRY POLICY (Phase 1d): only symmetry-free meshes are supported; anything else
    // aborts loudly rather than producing silently wrong results (audit-sec.7 unfolding
    // is a follow-up).
    utils::check(nqpts == nqpts_ibz and nkpts == nkpts_ibz,
                 "vertex_t::eval_Pi_C: vertex + k-space symmetry is not supported yet "
                 "(nqpts = {}, nqpts_ibz = {}, nkpts = {}, nkpts_ibz = {}). "
                 "Rerun with symmetry disabled in the mean-field input.",
                 nqpts, nqpts_ibz, nkpts, nkpts_ibz);
    {
      auto kp_trev = MF->kp_trev();
      for (long ik = 0; ik < nkpts; ++ik)
        utils::check(not kp_trev(ik),
                     "vertex_t::eval_Pi_C: vertex + time-reversal-reduced meshes are not supported yet.");
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
    nda::array<ComplexType, 4> X_skPa(ns, nkpts, Np, nbnd);
    for (long is = 0; is < ns; ++is)
      for (long ik = 0; ik < nkpts; ++ik)
        X_skPa(is, ik, all, all) = thc.X(is, 0, ik);

    // ---- bare coulomb Z(q) (thc.Z is collective: call uniformly on all ranks) ---------
    nda::array<ComplexType, 3> Z_qPQ(nqpts_ibz, Np, Np);
    for (long iq = 0; iq < nqpts_ibz; ++iq)
      Z_qPQ(iq, all, all) = thc.Z(int(iq));

    // head insertion, bare piece (weight 1) into Z(Gamma) -- (H1) of the memo
    nda::array<ComplexType, 2> H_PQ(Np, Np);
    bool head_ok = false;
    if (head_insertion) {
      head_ok = vertex_head_detail::build_head_rank1(thc, iq_gamma, nkpts, H_PQ);
      if (head_ok) {
        Z_qPQ(iq_gamma, all, all) += H_PQ;
        double h_max = 0.0;
        for (auto const& v : H_PQ) h_max = std::max(h_max, std::abs(v));
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
    // scr_coulomb_t.cpp:217) on (nq, nt_half, Np, Np). Absent on the very first
    // iteration: the rung then reduces to the bare interaction Z.
    std::optional<nda::array<ComplexType, 4>> Wdyn_qwPQ;
    if (mb_state.dW_qtPQ.has_value()) {
      auto& dW = mb_state.dW_qtPQ.value();
      auto gs = dW.global_shape();
      utils::check(gs[0] == nqpts_ibz and gs[1] == nt_half and gs[2] == Np and gs[3] == Np,
                   "vertex_t::eval_Pi_C: unexpected dW_qtPQ global shape.");
      nda::array<ComplexType, 4> W_qtPQ(nqpts_ibz, nt_half, Np, Np);
      W_qtPQ() = ComplexType(0.0);
      W_qtPQ(dW.local_range(0), dW.local_range(1), dW.local_range(2), dW.local_range(3)) = dW.local();
      mpi->comm.all_reduce_in_place_n(W_qtPQ.data(), W_qtPQ.size(), std::plus<>{});

      // head insertion, dynamic piece (weight Re[eps_inv_head(tau)]) into dW(Gamma, tau),
      // BEFORE the tau -> Matsubara transform (so the bosonic-mesh rung inherits it).
      // Same Re[.] convention as Sigma_div_correction (thc_gw.icc:506).
      if (head_ok) {
        if (mb_state.eps_inv_head.has_value()) {
          auto& eps = mb_state.eps_inv_head.value();
          utils::check(eps.shape(0) == nt_half,
                       "vertex_t::eval_Pi_C: eps_inv_head size {} != nt_half = {}.",
                       eps.shape(0), nt_half);
          for (long it = 0; it < nt_half; ++it)
            W_qtPQ(iq_gamma, it, all, all) += ComplexType(eps(it).real()) * H_PQ;
          app_log(1, "  Pi^C head insertion: dynamic piece applied to dW(Gamma, tau) "
                     "with eps_inv_head(tau=0) = {}", eps(0).real());
        } else {
          app_log(1, "  [WARNING] Pi^C: dW is present but eps_inv_head is not in MBState "
                     "-- the DYNAMIC head\n"
                     "            piece is skipped (bare piece applied).");
        }
      }

      long nw_b = tools.nw_b;
      long nw_half = (nw_b % 2 == 0) ? nw_b / 2 : nw_b / 2 + 1;
      Wdyn_qwPQ.emplace(nda::array<ComplexType, 4>(nqpts_ibz, nw_b, Np, Np));
      nda::array<ComplexType, 3> W_wpos(nw_half, Np, Np);
      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        auto W_t = W_qtPQ(iq, nda::ellipsis{});
        _ft->tau_to_w_PHsym(W_t, W_wpos);
        // unfold to the full mesh assuming W(-nu) = W(nu) (PH-symmetric storage, same
        // assumption as the SOSEX cache folding, thc_sosex.icc:970-976)
        for (long l = 0; l < nw_b; ++l) {
          long lpos = std::max(l, tools.w_mirror_b(l)) - nw_b / 2;
          Wdyn_qwPQ.value()(iq, l, all, all) = W_wpos(lpos, all, all);
        }
      }
    } else {
      app_log(1, "  [NOTE] Pi^C: no dynamic W in MBState (first iteration) -- "
                 "using the bare-interaction rung W = Z only.\n");
    }

    // ---- momentum maps (symmetry-free mesh) -------------------------------------------
    nda::array<long, 2> kmq(nqpts_ibz, nkpts), kpq(nqpts_ibz, nkpts);
    for (long iq = 0; iq < nqpts_ibz; ++iq) {
      for (long ik = 0; ik < nkpts; ++ik) kmq(iq, ik) = MF->qk_to_k2(iq, ik);
      for (long ik = 0; ik < nkpts; ++ik) kpq(iq, kmq(iq, ik)) = ik;   // inverse: (k-q)+q = k
    }

    // ---- Refinement 2: optional secondary-basis substitution --------------------------
    // (notes/refinement2_optionA.md section 4). The SAME kernel runs on the input set
    // (Xb, Zbar = t Z t^dag, Wbar_dyn = t dW t^dag, G_CC, window [0, nc)); the
    // head-augmented Gamma cells above downfold automatically through t. Pibar^C is
    // produced in (N_m x N_m) and UPFOLDED with the adjoint of the same t (Eq. 38) --
    // the no-leak identity (Eq. 39) is checked below as a transposition tripwire.
    const bool sec = secondary();
    const long nc = _band_window.size();
    nda::array<ComplexType, 3> Zb_qmm;
    std::optional<nda::array<ComplexType, 4>> Wbdyn_qwmm;
    // STRICT C-C EXTERNALS (theory-owner ruling, notes/refinement2_optionA.md
    // DECISION 2): dPhi_2^C/dW vanishes unless ALL FOUR pair orbital indices are in C,
    // so the external legs of Pi^C (the zL/zR one-sided transforms, which previously
    // summed the full band range) are C-restricted in BOTH paths via the input
    // projection (G_CC + C columns of the collocation) -- exactly the kernel on
    // G~ = P_C G P_C, the exact all-C cut (conservation notes section 1.2).
    nda::array<ComplexType, 5> G_CC(nt_f, ns, nkpts, nc, nc);
    G_CC = G_tskij(all, all, all, _band_window, _band_window);
    app_log(2, "  Pi^C external legs restricted to C = [{}, {}) (strict Phi cut; "
               "notes/refinement2_optionA.md DECISION 2).",
            _band_window.first(), _band_window.last());
    if (sec) {
      build_secondary_basis(thc, X_skPa, kmq, iq_gamma);
      app_log(1, "  Refinement 2: Pi^C runs in the SECONDARY basis (N_m = {} vs Np = {}); "
                 "upfold Pi^C = t^dag Pibar t.", _Nm, Np);
      if (ns * nkpts * nc * nc <= 4096) {
        vertex_secondary_detail::eta_max_over_q(
            "Z", X_skPa, _band_window.first(), nc, _Xb_skma, _t_qmP, kmq,
            [&](long iq) { return Z_qPQ(iq, all, all); });
        if (Wdyn_qwPQ.has_value()) {
          vertex_secondary_detail::eta_max_over_q(
              "dW(nu_0)", X_skPa, _band_window.first(), nc, _Xb_skma, _t_qmP, kmq,
              [&](long iq) { return Wdyn_qwPQ.value()(iq, tools.m0, all, all); });
          vertex_secondary_detail::eta_max_over_q(
              "dW(nu_max)", X_skPa, _band_window.first(), nc, _Xb_skma, _t_qmP, kmq,
              [&](long iq) { return Wdyn_qwPQ.value()(iq, tools.nw_b - 1, all, all); });
        }
      } else {
        app_log(2, "  Refinement 2: eta diagnostic skipped (N_pair = {} > 4096).",
                ns * nkpts * nc * nc);
      }
      Zb_qmm = nda::array<ComplexType, 3>(nqpts_ibz, _Nm, _Nm);
      nda::array<ComplexType, 2> tmp(_Nm, Np);
      if (Wdyn_qwPQ.has_value())
        Wbdyn_qwmm.emplace(nda::array<ComplexType, 4>(nqpts_ibz, tools.nw_b, _Nm, _Nm));
      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        auto t_q = _t_qmP(iq, all, all);
        vertex_secondary_detail::fold_core(t_q, Z_qPQ(iq, all, all), tmp,
                                           Zb_qmm(iq, all, all));
        if (Wdyn_qwPQ.has_value())
          for (long l = 0; l < tools.nw_b; ++l)
            vertex_secondary_detail::fold_core(t_q, Wdyn_qwPQ.value()(iq, l, all, all),
                                               tmp, Wbdyn_qwmm.value()(iq, l, all, all));
      }
    }

    // ---- kernel: accumulate Pi^C(inu) over this rank's (s,k,qx) tuples ----------------
    // q->0 policy resolved above (skip_rung_gamma / head-augmented inputs). Both paths:
    // C-restricted externals; the ONLY difference is the auxiliary input set --
    // (X_C, W, Z, Np) global vs (Xb, Wbar, Zbar, N_m) secondary.
    const long naux = sec ? _Nm : Np;
    nda::array<double, 1> qx_diag(nqpts_ibz);
    nda::array<ComplexType, 4> Pi_wqMN(tools.nw_b, nqpts_ibz, naux, naux);
    Pi_wqMN() = ComplexType(0.0);
    if (sec)
      vertex_pi::pi_c_accumulate_w(*_ft, tools, G_CC, _Xb_skma, Zb_qmm,
                                   Wbdyn_qwmm.has_value() ? &Wbdyn_qwmm.value() : nullptr,
                                   kmq, kpq, nda::range(0, nc), Pi_wqMN,
                                   mpi->comm.rank(), mpi->comm.size(),
                                   skip_rung_gamma, &qx_diag);
    else {
      nda::array<ComplexType, 4> X_C(ns, nkpts, Np, nc);
      X_C = X_skPa(all, all, all, _band_window);
      vertex_pi::pi_c_accumulate_w(*_ft, tools, G_CC, X_C, Z_qPQ,
                                   Wdyn_qwPQ.has_value() ? &Wdyn_qwPQ.value() : nullptr,
                                   kmq, kpq, nda::range(0, nc), Pi_wqMN,
                                   mpi->comm.rank(), mpi->comm.size(),
                                   skip_rung_gamma, &qx_diag);
    }
    mpi->comm.all_reduce_in_place_n(Pi_wqMN.data(), Pi_wqMN.size(), std::plus<>{});
    mpi->comm.all_reduce_in_place_n(qx_diag.data(), qx_diag.size(), std::plus<>{});

    // per-qx rung diagnostics (sum of rank-local maxima -- order-of-magnitude indicator
    // for the q->0 head pathology; Gamma reads 0 when skipped)
    {
      long iqg = -1;
      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        bool isg = true;
        for (long ik = 0; ik < nkpts; ++ik)
          if (kmq(iq, ik) != ik) { isg = false; break; }
        if (isg) { iqg = iq; break; }
      }
      double g_val = (iqg >= 0) ? qx_diag(iqg) : -1.0;
      double other = 0.0;
      for (long iqx = 0; iqx < nqpts_ibz; ++iqx) {
        app_log(3, "  Pi^C rung diagnostics: qx = {}  max|contribution| = {}", iqx, qx_diag(iqx));
        if (iqx != iqg) other = std::max(other, qx_diag(iqx));
      }
      app_log(2, "  Pi^C rung per-qx |contribution|: Gamma(iq={}) = {}, max(other qx) = {}\n",
              iqg, g_val, other);
    }

    // ---- Refinement 2: upfold Pi^C = t^dag Pibar^C t (Eq. 38) + no-leak tripwire ------
    if (sec) {
      nda::array<ComplexType, 4> Pi_up(tools.nw_b, nqpts_ibz, Np, Np);
      nda::array<ComplexType, 2> tmp(Np, _Nm);
      double leak_max = 0.0;
      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        auto t_q = _t_qmP(iq, all, all);
        for (long l = 0; l < tools.nw_b; ++l)
          vertex_secondary_detail::upfold_core(t_q, Pi_wqMN(l, iq, all, all), tmp,
                                               Pi_up(l, iq, all, all));
        // no-leak (Eq. 39) at the nu = 0 node, against the bare core (always present):
        // sum_MN [t^dag Pibar t]_MN Z_NM must equal sum_mn Pibar_mn Zbar_nm EXACTLY --
        // a residual here means a transposition/conjugation bug in t or the upfold.
        ComplexType S_up(0.0), S_bar(0.0);
        for (long M = 0; M < Np; ++M)
          for (long N = 0; N < Np; ++N)
            S_up += Pi_up(tools.m0, iq, M, N) * Z_qPQ(iq, N, M);
        for (long m = 0; m < _Nm; ++m)
          for (long n = 0; n < _Nm; ++n)
            S_bar += Pi_wqMN(tools.m0, iq, m, n) * Zb_qmm(iq, n, m);
        leak_max = std::max(leak_max, std::abs(S_up - S_bar) /
                                      std::max(std::abs(S_bar), 1e-300));
      }
      app_log(2, "  Refinement 2 no-leak residual (Eq. 39; nu = 0 node, bare-Z pairing): "
                 "max_q = {}", leak_max);
      Pi_wqMN = std::move(Pi_up);
    }

    // ---- to the code's tau storage convention -----------------------------------------
    nda::array<ComplexType, 4> Pi_tqMN(nt_half, nqpts_ibz, Np, Np);
    vertex_pi::pi_w_to_code_tau(*_ft, tools, Pi_wqMN, Pi_tqMN);

    {
      double max_abs = 0.0;
      long n_bad = 0;
      for (auto const& v : Pi_tqMN) {
        double a = std::abs(v);
        if (not std::isfinite(a)) { ++n_bad; continue; }
        max_abs = std::max(max_abs, a);
      }
      utils::check(n_bad == 0,
                   "vertex_t::eval_Pi_C: Pi^C contains {} NaN/Inf entries -- aborting.", n_bad);
      app_log(2, "  Pi^C(tau) max|.| = {}\n", max_abs);
    }

    // ---- scatter into the caller's distributed layout ---------------------------------
    auto dPi_C_tqPQ = math::nda::make_distributed_array<memory::array<HOST_MEMORY, ComplexType, 4>>(
        mpi->comm, pi_pgrid, pi_gshape, pi_bsize);
    dPi_C_tqPQ.local() = Pi_tqMN(dPi_C_tqPQ.local_range(0), dPi_C_tqPQ.local_range(1),
                                 dPi_C_tqPQ.local_range(2), dPi_C_tqPQ.local_range(3));
    mpi->comm.barrier();

    return dPi_C_tqPQ;
  }

  // template instantiations
  template void vertex_t::eval_Sigma_C(MBState&, const thc_reader_t&);

  template memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>
  vertex_t::eval_Pi_C(MBState&, const thc_reader_t&,
                      std::array<long, 4>, std::array<long, 4>, std::array<long, 4>);

}  // solvers
}  // methods
