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

#ifndef COQUI_SIGMA_CD_LINE_HPP
#define COQUI_SIGMA_CD_LINE_HPP

/**
 * ===========================================================================
 * THE eq-1 CD ASSEMBLY  (increment TC-3 / spec M4)
 * ===========================================================================
 *
 * Sigma^c from W^c evaluated at POINTS -- an imaginary-axis integral plus a
 * residue sum -- which is the one form the tilted-contour route can feed,
 * because the contour delivers W^c(z) at a target list and NOT a pole
 * representation. (`sigma_route_b::sigma_cd`, the tau/nu/spectral routes'
 * evaluator, is the complementary object: exact finite-T in closed form, but
 * it requires poles.)
 *
 *      Sigma^c_nn(w) = -(1/2pi) Int dnu  sum_m <nm|W^c(i nu)|mn> / (w + i nu - eps_m)
 *                    +           sum_m sigma_m <nm|W^c(w - eps_m + i delta)|mn>
 *
 * ---------------------------------------------------------------------------
 * 1. THE sigma_m SIGNS -- DERIVED, AND IN THEIR FINITE-T FORM
 * ---------------------------------------------------------------------------
 * results section 2.3 records that the spec's stated sigma_m are the OPPOSITE
 * of the ones that reproduce Sigma^c, and fixes them to
 *
 *      sigma_m = +1   for  mu < eps_m < w     (empty, below w)
 *      sigma_m = -1   for  w < eps_m < mu     (occupied, above w)
 *      sigma_m =  0   otherwise.
 *
 * Re-deriving that here mechanically produces something slightly stronger.
 * For ONE G pole at eps_m (weight 1) and a W^c pole set {(w_j, om_j)}, closing
 * the nu integral in the upper half plane gives, with A = w - eps_m,
 *
 *      I(A) = [A>0] sum_j w_j/(A+om_j)  -  sum_{om_j<0} w_j/(A+om_j),      (I)
 *
 * while the EXACT finite-T self-energy is the closed form of
 * sigma_route_b.hpp, Sigma^c = sum_j w_j [n_B(om_j) + f(eps_m)]/(A + om_j).
 * Subtracting, and using that a particle-hole-symmetric W^c is EVEN so that
 * sum_j w_j/(A+om_j) = -W^c(A):
 *
 *      Sigma^c - I = [ theta(A) - f(eps_m) ] W^c(A)
 *                    + sum_j w_j [ n_B(om_j) + theta(-om_j) ] / (A + om_j).   (R)
 *
 * The first bracket IS sigma_m, in a form valid at any temperature:
 *
 *      sigma_m = theta(w - eps_m) - f(eps_m).                          (SIGMA_M)
 *
 * Its beta -> infinity limit is exactly the three cases above (f -> a step at
 * mu), so this is the DERIVED convention, not a different one -- but it also
 * carries the fractional occupations that a metal actually has, at no cost,
 * since f is already in hand. `sigma_m_weight` implements it.
 *
 * The leftover in (R) is purely BOSONIC and state-independent:
 * n_B(om) + theta(-om) equals n_B(om) for om > 0 and -n_B(-om) for om < 0, so
 * it is bounded by exp(-beta*om_min) where om_min is the bottom of W^c's
 * spectral support. It is the ONE approximation the eq-1 form makes relative to
 * the exact finite-T closed form, it is reported by `thermal_leftover_bound`,
 * and on any gapped W^c at production beta it is astronomically small
 * (measured 1e-221 at beta = 1e3, om_p ~ 0.5 in the unit pin).
 * [verified: the pin `tc_sigma_cd_single_pole` scores (SIGMA_M) against the
 *  exact finite-T Lehmann at 1e-14, with NO quadrature -- the integral term
 *  (I) is evaluated in closed form there.]
 *
 * ---------------------------------------------------------------------------
 * 2. WHAT FEEDS EACH TERM
 * ---------------------------------------------------------------------------
 *  * The RESIDUE term needs W^c on the LINE Im z = delta, at z = w - eps_m +
 *    i delta -- one target per (evaluation energy, internal state m). This is
 *    the contour route's product (p_contour + wc_line).
 *  * The IMAGINARY-AXIS term needs W^c(i nu) only. That is what the production
 *    Matsubara solver already computes and stores, so it "reuses the existing
 *    machinery" (spec M4) and never asks the contour for small nu -- which is
 *    exactly where results section 5.5 says the contour is invalid
 *    (nu >= gamma only). The caller supplies the nu nodes, their weights and
 *    the contracted <nm|W^c(i nu)|mn>; `imag_axis_term` does the rest.
 *
 * `tan_quadrature` provides the campaign's own nu grid (Gauss-Legendre in
 * nu = L tan(u)) for the unit pins and for any caller that wants a continuous
 * rule rather than the stored bosonic mesh.
 */

#include <cmath>
#include <complex>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"

namespace methods {
namespace sigma_cd_line {

  using dcomplex = std::complex<double>;

  /** Fermi factor, numerically stable (same algebra as sigma_route_b::stable_nF). */
  inline double stable_nF(double beta, double e) {
    if (e >= 0.0) {
      const double x = std::exp(-beta * e);
      return x / (1.0 + x);
    }
    return 1.0 / (1.0 + std::exp(beta * e));
  }
  inline double stable_nB(double beta, double e) {
    if (e > 0.0) {
      const double x = std::exp(-beta * e);
      return x / (1.0 - x);
    }
    return -1.0 / (1.0 - std::exp(beta * e));
  }

  /**
   * (SIGMA_M): sigma_m = theta(w - eps_m) - f(eps_m).
   *
   * At beta -> infinity this is +1 for mu < eps_m < w, -1 for w < eps_m < mu,
   * and 0 otherwise -- the DERIVED signs of results section 2.3.
   */
  inline double sigma_m_weight(double omega, double eps_m, double mu, double beta) {
    const double theta = (omega > eps_m) ? 1.0 : 0.0;
    return theta - stable_nF(beta, eps_m - mu);
  }

  /**
   * The size of the bosonic leftover in (R): max_j |n_B(om_j) + theta(-om_j)|
   * over a W^c pole set. Reported, never corrected -- correcting it would need
   * the pole set, which the contour route does not have.
   */
  inline double thermal_leftover_bound(double beta,
                                       nda::MemoryArrayOfRank<1> auto const &om) {
    double worst = 0.0;
    for (long j = 0; j < om.shape(0); ++j) {
      const double o = om(j);
      const double t = (o > 0.0) ? stable_nB(beta, o) : -stable_nB(beta, -o);
      worst = std::max(worst, std::abs(t));
    }
    return worst;
  }

  // =========================================================================
  //  the imaginary-axis quadrature
  // =========================================================================

  struct quad_t {
    nda::array<double, 1> nu;    ///< nodes on the FULL axis (-inf, inf)
    nda::array<double, 1> w;     ///< weights
  };

  /**
   * Gauss-Legendre in the substituted variable nu = L tan(u), u in (-pi/2, pi/2).
   * `L` must exceed the plasmon range or the eq-1 identity becomes
   * quadrature-limited rather than sign-limited.
   * [port of tc_validation/models.py::SigmaModel.cd_assembly's _ivcache block]
   */
  inline quad_t tan_quadrature(long n, double L = 50.0) {
    utils::check(n >= 2, "sigma_cd_line::tan_quadrature: n = {} must be >= 2.", n);
    quad_t q;
    q.nu.resize(n);
    q.w.resize(n);
    // Gauss-Legendre nodes/weights on [-1,1] by Newton iteration on P_n.
    for (long i = 0; i < n; ++i) {
      double x = std::cos(M_PI * (double(i) + 0.75) / (double(n) + 0.5));
      double dp = 0.0;
      for (int it = 0; it < 100; ++it) {
        double p0 = 1.0, p1 = 0.0;
        for (long k = 0; k < n; ++k) {
          const double p2 = p1;
          p1 = p0;
          p0 = ((2.0 * double(k) + 1.0) * x * p1 - double(k) * p2) / (double(k) + 1.0);
        }
        dp = double(n) * (x * p0 - p1) / (x * x - 1.0);
        const double dx = -p0 / dp;
        x += dx;
        if (std::abs(dx) < 1e-15) break;
      }
      const double wgl = 2.0 / ((1.0 - x * x) * dp * dp);
      const double t = 0.5 * M_PI * x * 0.999999;
      q.nu(i) = L * std::tan(t);
      q.w(i) = wgl * 0.5 * M_PI * 0.999999 * L / (std::cos(t) * std::cos(t));
    }
    return q;
  }

  /**
   * The imaginary-axis term of eq 1 for ONE internal state m:
   *
   *     -(1/2pi) Int dnu  Wnm(i nu) / (w + i nu - eps_m)
   *
   * @param nu, wq  the quadrature (any rule; `tan_quadrature` supplies one).
   * @param Wnm     (n_nu) the contracted <nm|W^c(i nu)|mn>.
   */
  inline dcomplex imag_axis_term(double omega, double eps_m,
                                 nda::MemoryArrayOfRank<1> auto const &nu,
                                 nda::MemoryArrayOfRank<1> auto const &wq,
                                 nda::MemoryArrayOfRank<1> auto const &Wnm) {
    const long n = nu.shape(0);
    utils::check(wq.shape(0) == n and Wnm.shape(0) == n,
                 "sigma_cd_line::imag_axis_term: nu has {} nodes, weights {}, W {}.",
                 n, wq.shape(0), Wnm.shape(0));
    dcomplex acc(0.0, 0.0);
    for (long i = 0; i < n; ++i)
      acc += wq(i) * Wnm(i) / dcomplex(omega - eps_m, nu(i));
    return -acc / (2.0 * M_PI);
  }

  /**
   * The closed-form integral term (I) for a W^c given as an explicit pole set --
   * the unit-pin path, exact and quadrature-free:
   *
   *     I(A) = [A>0] sum_j w_j/(A+om_j)  -  sum_{om_j<0} w_j/(A+om_j).
   */
  inline dcomplex imag_axis_term_poles(double A,
                                       nda::MemoryArrayOfRank<1> auto const &w,
                                       nda::MemoryArrayOfRank<1> auto const &om) {
    const long n = om.shape(0);
    dcomplex all(0.0, 0.0), neg(0.0, 0.0);
    for (long j = 0; j < n; ++j) {
      const dcomplex t = dcomplex(w(j)) / (A + om(j));
      all += t;
      if (om(j) < 0.0) neg += t;
    }
    return (A > 0.0 ? all : dcomplex(0.0, 0.0)) - neg;
  }

  /**
   * Assemble Sigma^c_nn(w) from the two terms.
   *
   * @param omega     the evaluation energy (real part; the caller applies any
   *                  i*eta itself through `z_res`).
   * @param eps       (n_m) internal-state energies, ABSOLUTE.
   * @param W_iv      (n_m, n_nu) contracted <nm|W^c(i nu)|mn>.
   * @param W_line    (n_m) contracted <nm|W^c(w - eps_m + i delta)|mn>; only the
   *                  entries with a nonzero sigma_m are read.
   * @param st_sigma  (n_m) OPTIONAL out: the sigma_m actually used.
   */
  inline dcomplex assemble(double omega, double mu, double beta,
                           nda::MemoryArrayOfRank<1> auto const &eps,
                           nda::MemoryArrayOfRank<1> auto const &nu,
                           nda::MemoryArrayOfRank<1> auto const &wq,
                           nda::MemoryArrayOfRank<2> auto const &W_iv,
                           nda::MemoryArrayOfRank<1> auto const &W_line,
                           nda::array<double, 1> *st_sigma = nullptr) {
    const long nm = eps.shape(0);
    utils::check(W_iv.shape()[0] == nm and W_line.shape(0) == nm,
                 "sigma_cd_line::assemble: {} states, W_iv is {}x{}, W_line has {}.",
                 nm, W_iv.shape()[0], W_iv.shape()[1], W_line.shape(0));
    if (st_sigma != nullptr) st_sigma->resize(nm);
    dcomplex s(0.0, 0.0);
    nda::array<dcomplex, 1> row(nu.shape(0));
    for (long m = 0; m < nm; ++m) {
      for (long i = 0; i < nu.shape(0); ++i) row(i) = W_iv(m, i);
      s += imag_axis_term(omega, double(std::real(ComplexType(eps(m)))), nu, wq, row);
      const double sg = sigma_m_weight(omega, double(std::real(ComplexType(eps(m)))),
                                       mu, beta);
      if (st_sigma != nullptr) (*st_sigma)(m) = sg;
      if (sg != 0.0) s += sg * W_line(m);
    }
    return s;
  }

  /** The residue-target list eq 1 needs at one evaluation energy. */
  inline void residue_targets(double omega, double delta, double mu, double beta,
                              nda::MemoryArrayOfRank<1> auto const &eps,
                              nda::array<dcomplex, 1> &z,
                              nda::array<double, 1> &sg) {
    const long nm = eps.shape(0);
    z.resize(nm);
    sg.resize(nm);
    for (long m = 0; m < nm; ++m) {
      const double e = double(std::real(ComplexType(eps(m))));
      z(m) = dcomplex(omega - e, delta);
      sg(m) = sigma_m_weight(omega, e, mu, beta);
    }
  }

} // namespace sigma_cd_line
} // namespace methods

#endif // COQUI_SIGMA_CD_LINE_HPP
