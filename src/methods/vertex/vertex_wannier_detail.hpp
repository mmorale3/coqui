/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team
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
#ifndef COQUI_VERTEX_WANNIER_DETAIL_HPP
#define COQUI_VERTEX_WANNIER_DETAIL_HPP

// The Wannier-projector helpers of vertex_t (Loewdin, G_bar = U^dag G U, X_bar = X U, the chain-rule
// Sigma injection). Extracted from vertex_t.cpp (W-int-1) so the .icc kernels included BEFORE the
// vertex_t.cpp body (vertex_ladder.icc::ladder_inputs) can rotate their inputs to the MLWF frame too.
#include "nda/nda.hpp"
#include "nda/linalg/eigenelements.hpp"
#include "utilities/check.hpp"
#include "vertex_t.h"

namespace methods {
namespace solvers {

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

} // solvers
} // methods

#endif // COQUI_VERTEX_WANNIER_DETAIL_HPP
