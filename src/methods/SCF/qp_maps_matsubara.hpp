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

#ifndef COQUI_QP_MAPS_MATSUBARA_HPP
#define COQUI_QP_MAPS_MATSUBARA_HPP

/**
 * Project 2 (qpGW+BSE+EDMFT on the imaginary axis) increment Q1
 * (notes/qpgw_edmft_implementation_plan.md; spec notes/qpgw_bse_edmft_option2.pdf
 * section 4): MATSUBARA-NATIVE quasiparticle maps -- the AC-free alternatives to
 * the Pade route of solve_qp_eqn / qp_approx (qp_scf_common.cpp:275/489). Both
 * consume Sigma_c(iw_n) in the MO basis on the fermionic Matsubara mesh; neither
 * touches the real axis. V_ref^xc is handled STRUCTURALLY: the qp loop recomputes
 * HF each iteration and feeds these maps the correlation-only Sigma (the
 * add_qpscf_vcorr convention), so no explicit V_ref subtraction appears.
 *
 * map (i) "mats_lin" -- omega ~ 0 linearization (spec eq 13, Kutepov-type):
 *   Zt = [1 - dSigma/d(iw)|_0]^{-1}, the derivative from the FIRST Matsubara node
 *   via the exact odd-part identity Sigma(-iw) = Sigma(iw)^dag:
 *       S1 = [Sigma(iw0) - Sigma(iw0)^dag] / (2 i w0)          (Hermitian);
 *   Sigma(0) by EVEN quadratic extrapolation of the Hermitian part from the two
 *   lowest nodes (S(w) ~ S0 + c w^2):
 *       S0 = (w1^2 H(w0) - w0^2 H(w1)) / (w1^2 - w0^2);
 *   V = herm( Zt^{1/2} S0 Zt^{1/2} ),  Zt^{1/2} by eigendecomposition of the
 *   Hermitian S1 with z_a = 1/(1 - s_a) clamped to [z_floor, 1] (a physical-Z
 *   window; clamps are counted and reported). Resolution floor = pi*T (the first
 *   node), documented and inherent to the map.
 *
 * map (ii) "mats_gmatch" -- variational Green's-function matching (spec eq 14):
 *   H = argmin_{H = H^dag} sum_n w_n || G_n - (iw_n + mu - H)^{-1} ||_F^2.
 *   Projected, Levenberg-damped Gauss-Newton. Writing M = K^{-1},
 *   K = (iw + mu) I - H, the first-order change of M under H -> H + d is
 *   M d M = sum_{ij} d_ij (M e_i)(e_j^T M), an OUTER PRODUCT per basis element,
 *   which collapses the normal equations to small-matrix products:
 *       A[(ij),(kl)]  = sum_n w_n (M^dag M)_{ik} (M M^dag)_{lj},
 *       b[(ij)]       = sum_n w_n [ M^dag (G_n - M) M^dag ]_{ij},
 *   solve A d = b (A is Hermitian PSD; + Levenberg lambda), Hermitize the step,
 *   backtracking line search on the true residual. Weights w_n = (w_min/w_n)^wpow
 *   (reportable; spec section 4 "the weights are an explicit, reportable choice").
 *
 * Both maps return the QP-ized CORRELATION matrix V (map i) / the full static
 * H_eff block (map ii) in the MO basis; the callers compose with the static part
 * exactly as qp_approx does (Hermitization + back-transform).
 */

#include <cmath>
#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/linalg.hpp"
#include "nda/linalg/eigenelements.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"

namespace methods {
namespace qp_matsubara {

  namespace detail {

    // Hermitian part
    inline nda::array<ComplexType, 2> herm(nda::MemoryArrayOfRank<2> auto const &A) {
      const long n = A.shape(0);
      nda::array<ComplexType, 2> H(n, n);
      for (long i = 0; i < n; ++i)
        for (long j = 0; j < n; ++j) H(i, j) = 0.5 * (A(i, j) + std::conj(A(j, i)));
      return H;
    }

    // f(S) for Hermitian S via eigendecomposition (used for Zt^{1/2} with clamping)
    inline nda::array<ComplexType, 2> zt_sqrt_from_S1(nda::array<ComplexType, 2> const &S1,
                                                      double z_floor, long &n_clamped) {
      const long n = S1.shape(0);
      auto [ev, U] = nda::linalg::eigenelements(nda::matrix<ComplexType>(S1));
      nda::array<ComplexType, 2> Z12(n, n);
      Z12() = ComplexType(0.0);
      nda::array<double, 1> zh(n);
      for (long a = 0; a < n; ++a) {
        double z = 1.0 / (1.0 - ev(a));
        if (z < z_floor or z > 1.0 or not std::isfinite(z)) {
          z = std::min(1.0, std::max(z_floor, std::isfinite(z) ? z : 1.0));
          ++n_clamped;
        }
        zh(a) = std::sqrt(z);
      }
      nda::array<ComplexType, 2> T(n, n);
      for (long i = 0; i < n; ++i)
        for (long a = 0; a < n; ++a) T(i, a) = U(i, a) * zh(a);
      nda::blas::gemm(ComplexType(1.0), T, nda::dagger(U), ComplexType(0.0), Z12);
      return Z12;
    }

  } // detail

  /**
   * map (i), matrix form (the qsGW/qp_approx analogue). Sigma_wab holds the
   * correlation self-energy on the POSITIVE fermionic nodes in ascending |w|
   * (Sigma(iw0), Sigma(iw1), ...); wn the matching Matsubara values (> 0).
   * Returns V (Hermitian, nb x nb). n_clamped accumulates Z-window clamps.
   */
  inline nda::array<ComplexType, 2> qp_lin_matrix(nda::MemoryArrayOfRank<3> auto const &Sigma_wab,
                                                  nda::MemoryArrayOfRank<1> auto const &wn,
                                                  long &n_clamped, double z_floor = 0.05) {
    const long nw = Sigma_wab.shape(0), nb = Sigma_wab.shape(1);
    utils::check(nw >= 2 and wn.shape(0) >= 2 and wn(0) > 0 and wn(1) > wn(0),
                 "qp_lin_matrix: need >= 2 ascending positive Matsubara nodes.");
    decltype(nda::range::all) all;
    const double w0 = wn(0), w1 = wn(1);
    // S1 = [Sigma(iw0) - Sigma(iw0)^dag] / (2 i w0)  (Hermitian by construction)
    nda::array<ComplexType, 2> S1(nb, nb);
    for (long i = 0; i < nb; ++i)
      for (long j = 0; j < nb; ++j)
        S1(i, j) = (Sigma_wab(0, i, j) - std::conj(Sigma_wab(0, j, i))) /
                   (ComplexType(0.0, 2.0 * w0));
    // S0: even quadratic extrapolation of the Hermitian parts at w0, w1
    auto H0 = detail::herm(Sigma_wab(0, all, all));
    auto H1 = detail::herm(Sigma_wab(1, all, all));
    nda::array<ComplexType, 2> S0(nb, nb);
    const double d = w1 * w1 - w0 * w0;
    for (long i = 0; i < nb; ++i)
      for (long j = 0; j < nb; ++j)
        S0(i, j) = (w1 * w1 * H0(i, j) - w0 * w0 * H1(i, j)) / d;
    auto Z12 = detail::zt_sqrt_from_S1(S1, z_floor, n_clamped);
    nda::array<ComplexType, 2> T(nb, nb), V(nb, nb);
    nda::blas::gemm(ComplexType(1.0), Z12, S0, ComplexType(0.0), T);
    nda::blas::gemm(ComplexType(1.0), T, Z12, ComplexType(0.0), V);
    return detail::herm(V);
  }

  /** map (i), scalar (the evGW/solve_qp_eqn analogue): E = Vhf + Zt * ReSigma(0). */
  inline double qp_lin_diagonal(nda::MemoryArrayOfRank<1> auto const &Sigma_w,
                                nda::MemoryArrayOfRank<1> auto const &wn,
                                double Vhf, long &n_clamped, double z_floor = 0.05) {
    nda::array<ComplexType, 3> S(2, 1, 1);
    S(0, 0, 0) = Sigma_w(0);
    S(1, 0, 0) = Sigma_w(1);
    auto V = qp_lin_matrix(S, wn, n_clamped, z_floor);
    return Vhf + V(0, 0).real();
  }

  struct gmatch_opts {
    double tol = 1e-10;    // relative residual-decrease stop
    int maxiter = 200;
    double wpow = 2.0;     // w_n = (w_min/w_n)^wpow  (reportable)
    double lm = 1e-10;     // Levenberg floor added to A's diagonal
    double z_floor = 0.05; // passed to the map-(i) initializer when used
  };
  struct gmatch_info {
    int iters = 0;
    double r0 = 0.0, r = 0.0;   // initial / final weighted residual (Frobenius^2)
    bool converged = false;
  };

  /**
   * map (ii): per-(s,k) block. G_wab = the TARGET Green's function on the positive
   * fermionic nodes (ascending), built from Sigma^GW only (spec principle 2);
   * H = Hermitian initial guess in, the matched static H_eff out.
   */
  inline gmatch_info qp_gmatch_block(nda::MemoryArrayOfRank<3> auto const &G_wab,
                                     nda::MemoryArrayOfRank<1> auto const &wn,
                                     double mu, nda::array<ComplexType, 2> &H,
                                     gmatch_opts const &opt = {}) {
    decltype(nda::range::all) all;
    const long nw = G_wab.shape(0), nb = G_wab.shape(1), n2 = nb * nb;
    gmatch_info info;
    nda::array<double, 1> w(nw);
    for (long n = 0; n < nw; ++n) w(n) = std::pow(wn(0) / wn(n), opt.wpow);

    auto resid = [&](nda::array<ComplexType, 2> const &Ht) {
      double r2 = 0.0;
      nda::matrix<ComplexType> K(nb, nb);
      for (long n = 0; n < nw; ++n) {
        K() = -Ht;
        for (long i = 0; i < nb; ++i) K(i, i) += ComplexType(mu, wn(n));
        nda::inverse_in_place(K);
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j) r2 += w(n) * std::norm(G_wab(n, i, j) - K(i, j));
      }
      return r2;
    };

    info.r0 = info.r = resid(H);
    nda::matrix<ComplexType> K(nb, nb);
    nda::array<ComplexType, 2> P(nb, nb), Q(nb, nb), Bm(nb, nb), T(nb, nb);
    nda::matrix<ComplexType> A(n2, n2);
    nda::array<ComplexType, 1> b(n2);
    nda::array<ComplexType, 2> Ht(nb, nb);
    for (int it = 1; it <= opt.maxiter; ++it) {
      info.iters = it;
      A() = ComplexType(0.0);
      b() = ComplexType(0.0);
      for (long n = 0; n < nw; ++n) {
        K() = -H;
        for (long i = 0; i < nb; ++i) K(i, i) += ComplexType(mu, wn(n));
        nda::inverse_in_place(K);                       // M
        // P = M^dag M, Q = M M^dag, Bm = M^dag (G - M) M^dag
        nda::blas::gemm(ComplexType(1.0), nda::dagger(K), K, ComplexType(0.0), P);
        nda::blas::gemm(ComplexType(1.0), K, nda::dagger(K), ComplexType(0.0), Q);
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j) T(i, j) = G_wab(n, i, j) - K(i, j);
        nda::array<ComplexType, 2> MG(nb, nb);
        nda::blas::gemm(ComplexType(1.0), nda::dagger(K), T, ComplexType(0.0), MG);
        nda::blas::gemm(ComplexType(1.0), MG, nda::dagger(K), ComplexType(0.0), Bm);
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j) {
            b(i * nb + j) += w(n) * Bm(i, j);
            for (long k = 0; k < nb; ++k)
              for (long l = 0; l < nb; ++l)
                A(i * nb + j, k * nb + l) += w(n) * P(i, k) * Q(l, j);
          }
      }
      for (long p = 0; p < n2; ++p) A(p, p) += ComplexType(opt.lm * (1.0 + std::abs(A(p, p))));
      nda::inverse_in_place(A);
      nda::array<ComplexType, 1> d(n2);
      nda::blas::gemv(ComplexType(1.0), A, b, ComplexType(0.0), d);
      // Hermitize the step (the projected-GN step onto H = H^dag)
      nda::array<ComplexType, 2> D(nb, nb);
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j)
          D(i, j) = 0.5 * (d(i * nb + j) + std::conj(d(j * nb + i)));
      // backtracking line search on the true residual
      double s = 1.0, rnew = 0.0;
      bool accepted = false;
      for (int ls = 0; ls < 12; ++ls) {
        Ht = H;
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j) Ht(i, j) += s * D(i, j);
        rnew = resid(Ht);
        if (rnew < info.r) { accepted = true; break; }
        s *= 0.5;
      }
      if (not accepted) break;                          // stalled at a minimum
      const double drop = (info.r - rnew) / std::max(info.r, 1e-300);
      H = Ht;
      info.r = rnew;
      if (drop < opt.tol) { info.converged = true; break; }
    }
    if (info.iters >= opt.maxiter)
      app_log(2, "  [qp_gmatch] hit maxiter = {} (r0 = {:.3e} -> r = {:.3e})",
              opt.maxiter, info.r0, info.r);
    else
      info.converged = true;
    return info;
  }

} // qp_matsubara
} // methods

#endif // COQUI_QP_MAPS_MATSUBARA_HPP
