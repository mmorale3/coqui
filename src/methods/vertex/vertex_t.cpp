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
#include "numerics/sparse/csr_blas.hpp"   // csrmm for the symmetry D-matrix blocks
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
    inline void g_rotation_check(vertex_sym::sym_ctx const& ctx,
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
    build_Xbar(nda::array<ComplexType, 4> const &X_skPa,   // (ns, nk, Np, nbnd)
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
                                  nda::array<ComplexType, 5> &Gbar) {           // (nt,ns,nk,M,M)
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
                               nda::array<ComplexType, 4> const &X_w,
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
    ctx.Xhat = nda::array<ComplexType, 5>(ctx.ns, ctx.nsym, ctx.nk_full, ctx.naux, nc);
    ctx.Dc = nda::array<ComplexType, 4>(ctx.nsym, ctx.nk_full, nc, nc);
    ctx.Dc() = ComplexType(0.0);
    ctx.cjg = nda::array<bool, 2>(ctx.nsym, ctx.nk_full);
    ctx.cjg() = false;
    for (long is = 0; is < ctx.ns; ++is)
      for (long ik = 0; ik < ctx.nk_full; ++ik)
        ctx.Xhat(is, 0, ik, all, all) = X_w(is, ik, all, all);

    double leak_max = 0.0, leak_sum = 0.0;
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
      for (long js = 1; js < ctx.nsym; ++js) {
        for (long ik = 0; ik < ctx.nk_full; ++ik) {
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
    }
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
                                       nda::array<ComplexType, 4> const &X_glob, long orb0,
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
    const long Nm_req = (_isdf_rank > 0) ? _isdf_rank : nc * nc * nkpts;
    utils::check(Nm_req <= Npair,
                 "vertex_t::build_secondary_basis: vertex_isdf_rank = {} exceeds the "
                 "subspace pair rank N_pair = ns*nk*nc^2 = {}; the secondary basis "
                 "cannot usefully exceed the space it represents.", Nm_req, Npair);

    app_log(1, "\n  Refinement 2: building the secondary ISDF basis on the subspace C "
               "(rank M = {}, {})\n"
               "  requested N_m = {}, svd_tol(B) = {}, N_pair (per q, spin-stacked) = {}\n",
            nc, _wannier ? "Wannier projector" : "band window",
            Nm_req, _isdf_svd_tol, Npair);

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
        vertex_secondary_detail::build_pair_matrix(X_glob, orb0, nc, kmq, iq, C_IP);
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
    utils::check(_ft->basis() == imag_axes_ft::dlr_basis,
                 "vertex_t::eval_Sigma_C: the fused G3W2 kernel requires the DLR IAFT backend "
                 "(iaft basis = \"dlr\"); the IR backend is not supported.");

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
    nda::array<ComplexType, 3> Z_qPQ(nqpts_ibz, Np, Np);
    for (long iq = 0; iq < nqpts_ibz; ++iq)
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
    // PH-symmetric in tau, W(beta-t) = W(t). IBZ rows under symmetry.
    nda::array<ComplexType, 4> Wt_qtPQ(nqpts_ibz, nt, Np, Np);
    {
      auto& dW = mb_state.dW_qtPQ.value();
      auto gs = dW.global_shape();
      utils::check(gs[0] == nqpts_ibz and gs[1] == nt_half and gs[2] == Np and gs[3] == Np,
                   "vertex_t::eval_Sigma_C: unexpected dW_qtPQ global shape.");
      nda::array<ComplexType, 4> W_half(nqpts_ibz, nt_half, Np, Np);
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
    nda::array<ComplexType, 5> G_CC(nt, ns, nkpts, nc, nc);
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
    app_log(2, "  Sigma^C externals restricted to {} a, b in [0, {}) "
               "(strict Phi cut; notes/refinement2_optionA.md DECISION 2).",
            wan ? "range(P) (Wannier labels)" : "the C-C block", nc);
    // effective window collocation: WINDOW = X(:,C); WANNIER = X_bar = X.U (Np x M).
    // (also the sym-ctx input; secondary uses Xb). orb0 of the pair matrices is 0 in
    // Wannier mode (X_C already carries exactly the M subspace columns).
    nda::array<ComplexType, 4> X_C(ns, nkpts, Np, nc);
    if (wan)
      X_C = vertex_wannier_detail::build_Xbar(X_skPa, _U_skia, _band_window);
    else
      X_C = X_skPa(all, all, all, _band_window);
    // the "global collocation" the secondary C(q)/eta refer to: X_bar (orb0=0) in
    // Wannier mode, X_skPa (orb0=C.first()) in window mode.
    nda::array<ComplexType, 4> const &X_glob = wan ? X_C : X_skPa;
    const long orb0_glob = wan ? 0 : _band_window.first();
    if (sec) {
      build_secondary_basis(thc, X_glob, orb0_glob, kmq, iq_gamma);
      app_log(1, "  Refinement 2: Sigma^C runs in the SECONDARY basis (N_m = {} vs "
                 "Np = {}); externals a, b in C.", _Nm, Np);
      // eta diagnostics (Eq. 40) on the rung arrays ACTUALLY consumed (test-scale gate)
      if (ns * nkpts * nc * nc <= 4096) {
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
      // non-IBZ transfers are sourced through the sym ctx, memo section 3.7)
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
      vertex_ibz_detail::g_rotation_check(*symc, G_CC, MF->kp_trev());
    }

    // ---- fused kernel (round-robin over (s,k,qx); result all-reduced inside) ----------
    // Both paths: C-restricted externals; the ONLY difference is the auxiliary input
    // set -- (X_C, W, Z, Np) global vs (Xb, Wbar, Zbar, N_m) secondary.
    nda::array<ComplexType, 5> Sigma_C(nt, ns, nk_ext, nc, nc);
    if (sec)
      vertex_detail::eval_sigma_C_g3w2(*_ft, mpi->comm, nda::range(0, nc), G_CC,
                                       _Xb_skma, Wb_qtmm, Zb_qmm, kmq, qmin,
                                       iq_gamma, skip_rung_gamma, symc, Sigma_C);
    else
      vertex_detail::eval_sigma_C_g3w2(*_ft, mpi->comm, nda::range(0, nc), G_CC,
                                       X_C, Wt_qtPQ, Z_qPQ, kmq, qmin, iq_gamma,
                                       skip_rung_gamma, symc, Sigma_C);
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
    //  stays downstream in scf_driver).
    //   WINDOW MODE: Sigma_C is already in band labels on the C-C block -> drop in.
    //   WANNIER MODE (memo C3/section 2.3): Sigma_C lives in Wannier labels; inject the
    //     operator sandwich Sigma^C_ij = [U Sigma_bar U^dag]_ij over i,j in W_rng
    //     (projector_t::upfold primitive). External k axis is IBZ-resident; the IBZ
    //     k-points are [0, nk_ext), so _U_skia(is, ik_ext) is the right U.
    if (mb_state.mpi->node_comm.root()) {
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
    std::optional<nda::array<ComplexType, 4>> Wdyn_qwPQ;
    if (use_wcache and mb_state.dW_qtPQ.has_value())
      app_log(2, "  [NOTE] Pi^C: both the W-bar cache and mb_state.dW_qtPQ are present "
                 "-- consuming the CACHE\n"
                 "         (identical content when both were produced by the same "
                 "update_w).");
    if (not use_wcache and mb_state.dW_qtPQ.has_value()) {
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
    } else if (not use_wcache) {
      // FIRST ITERATION (loud): nothing to screen with yet, in either path.
      app_log(1, "  [NOTE] Pi^C: no dynamic W in MBState{} (first iteration) -- "
                 "using the bare-interaction rung W = Z only.\n",
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
    nda::array<ComplexType, 5> G_CC(nt_f, ns, nkpts, nc, nc);
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
    app_log(2, "  Pi^C external legs restricted to {} [0, {}) (strict Phi cut; "
               "notes/refinement2_optionA.md DECISION 2).",
            wan ? "range(P) (Wannier labels)" : "the C window", nc);
    // effective window collocation: WINDOW = X(:,C); WANNIER = X_bar = X.U (Np x M).
    nda::array<ComplexType, 4> X_C(ns, nkpts, Np, nc);
    if (wan)
      X_C = vertex_wannier_detail::build_Xbar(X_skPa, _U_skia, _band_window);
    else
      X_C = X_skPa(all, all, all, _band_window);
    nda::array<ComplexType, 4> const &X_glob = wan ? X_C : X_skPa;
    const long orb0_glob = wan ? 0 : _band_window.first();
    if (sec) {
      build_secondary_basis(thc, X_glob, orb0_glob, kmq, iq_gamma);
      app_log(1, "  Refinement 2: Pi^C runs in the SECONDARY basis (N_m = {} vs Np = {}); "
                 "upfold Pi^C = t^dag Pibar t.", _Nm, Np);
      if (ns * nkpts * nc * nc <= 4096) {
        vertex_secondary_detail::eta_max_over_q(
            "Z", X_glob, orb0_glob, nc, _Xb_skma, _t_qmP, kmq,
            [&](long iq) { return Z_qPQ(iq, all, all); });
        if (Wdyn_qwPQ.has_value()) {
          vertex_secondary_detail::eta_max_over_q(
              "dW(nu_0)", X_glob, orb0_glob, nc, _Xb_skma, _t_qmP, kmq,
              [&](long iq) { return Wdyn_qwPQ.value()(iq, tools.m0, all, all); });
          vertex_secondary_detail::eta_max_over_q(
              "dW(nu_max)", X_glob, orb0_glob, nc, _Xb_skma, _t_qmP, kmq,
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
      vertex_ibz_detail::g_rotation_check(*symc, G_CC, MF->kp_trev());
    }

    // ---- kernel: accumulate Pi^C(inu) over this rank's (s,k,qx) tuples ----------------
    // q->0 policy resolved above (skip_rung_gamma / head-augmented inputs). Both paths:
    // C-restricted externals; the ONLY difference is the auxiliary input set --
    // (X_C, W, Z, Np) global vs (Xb, Wbar, Zbar, N_m) secondary.
    const long naux = sec ? _Nm : Np;
    nda::array<double, 1> qx_diag(nqpts);
    nda::array<ComplexType, 4> Pi_wqMN(tools.nw_b, nqpts_ibz, naux, naux);
    Pi_wqMN() = ComplexType(0.0);
    if (sec)
      vertex_pi::pi_c_accumulate_w(*_ft, tools, G_CC, _Xb_skma, Zb_qmm,
                                   Wbdyn_qwmm.has_value() ? &Wbdyn_qwmm.value() : nullptr,
                                   kmq, kpq, nda::range(0, nc), Pi_wqMN,
                                   mpi->comm.rank(), mpi->comm.size(),
                                   skip_rung_gamma, &qx_diag, symc);
    else
      vertex_pi::pi_c_accumulate_w(*_ft, tools, G_CC, X_C, Z_qPQ,
                                   Wdyn_qwPQ.has_value() ? &Wdyn_qwPQ.value() : nullptr,
                                   kmq, kpq, nda::range(0, nc), Pi_wqMN,
                                   mpi->comm.rank(), mpi->comm.size(),
                                   skip_rung_gamma, &qx_diag, symc);
    mpi->comm.all_reduce_in_place_n(Pi_wqMN.data(), Pi_wqMN.size(), std::plus<>{});
    mpi->comm.all_reduce_in_place_n(qx_diag.data(), qx_diag.size(), std::plus<>{});

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
    nda::array<ComplexType, 4> X_skPa(ns, nkpts, Np, nbnd);
    for (long is = 0; is < ns; ++is)
      for (long ik = 0; ik < nkpts; ++ik)
        X_skPa(is, ik, all, all) = thc.X(is, 0, ik);
    nda::array<long, 2> kmq(nqpts_ibz, nkpts);
    for (long iq = 0; iq < nqpts_ibz; ++iq)
      for (long ik = 0; ik < nkpts; ++ik) kmq(iq, ik) = MF->qk_to_k2(iq, ik);
    // effective global collocation for the secondary fit (WANNIER: X_bar = X.U, orb0=0)
    nda::array<ComplexType, 4> Xbar_glob;
    if (wan) Xbar_glob = vertex_wannier_detail::build_Xbar(X_skPa, _U_skia, _band_window);
    nda::array<ComplexType, 4> const &X_glob = wan ? Xbar_glob : X_skPa;
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
      auto& dW = mb_state.dW_qtPQ.value();
      nda::array<ComplexType, 4> W_qtPQ(nqpts_ibz, nt_half, Np, Np);
      W_qtPQ() = ComplexType(0.0);
      W_qtPQ(dW.local_range(0), dW.local_range(1), dW.local_range(2), dW.local_range(3)) = dW.local();
      mpi->comm.all_reduce_in_place_n(W_qtPQ.data(), W_qtPQ.size(), std::plus<>{});

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

      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        auto W_t = W_qtPQ(iq, nda::ellipsis{});
        auto W_w = W_wq(iq, nda::ellipsis{});
        _ft->tau_to_w_PHsym(W_t, W_w);
      }
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
      nda::array<ComplexType, 2> tmp(_Nm, Np);
      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        auto t_q = _t_qmP(iq, all, all);
        for (long lp = 0; lp < nw_half; ++lp)
          vertex_secondary_detail::fold_core(t_q, W_wq(iq, lp, all, all), tmp,
                                             _Wb_qwmm.value()(iq, lp, all, all));
      }
    }

    // ---- footprint: the memory point of the exercise ---------------------------------
    const double to_mb = 16.0 / (1024.0 * 1024.0);   // complex<double>
    const double cache_mb = double(nqpts_ibz) * double(nw_half) * double(_Nm) * double(_Nm) * to_mb;
    const double dw_mb = double(nqpts_ibz) * double(nt_half) * double(Np) * double(Np) * to_mb;
    app_log(2, "  Refinement 2 W-bar cache FILLED: (nq, nw_half, N_m, N_m) = "
               "({}, {}, {}, {}) = {:.3f} MB (replicated)\n"
               "  vs the retained dW it replaces: (nq, nt_half, Np, Np) = "
               "({}, {}, {}, {}) = {:.3f} MB -- ratio {:.3e}\n",
            nqpts_ibz, nw_half, _Nm, _Nm, cache_mb,
            nqpts_ibz, nt_half, Np, Np, dw_mb, cache_mb / dw_mb);
    mpi->comm.barrier();
  }

  // template instantiations
  template void vertex_t::eval_Sigma_C(MBState&, const thc_reader_t&);
  template void vertex_t::cache_w(MBState&, const thc_reader_t&);

  template memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>
  vertex_t::eval_Pi_C(MBState&, const thc_reader_t&,
                      std::array<long, 4>, std::array<long, 4>, std::array<long, 4>);

}  // solvers
}  // methods
