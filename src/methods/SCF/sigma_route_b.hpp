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

#ifndef COQUI_SIGMA_ROUTE_B_HPP
#define COQUI_SIGMA_ROUTE_B_HPP

/**
 * Project 2 (qpGW+BSE+EDMFT) increment QM2 -- ROUTE B: the contour-deformation (CD)
 * self-energy kernel (spec notes/qm2_route_b_finite_t_spec.md; parent
 * notes/qsgw_matsubara_plan.pdf section 3). Third sibling of qp_maps_matsubara.hpp
 * (AC-free Matsubara-native maps) and sigma_real_axis.hpp (route A, Taylor
 * re-expansion from imaginary-direction samples): this file evaluates Sigma^c at
 * ARBITRARY complex z -- real quasiparticle energies included -- in CLOSED FORM,
 * with no analytic continuation and no fit, given pole representations of G and W^c.
 *
 * THIS IS THE FINITE-TEMPERATURE FORMULA, NOT THE TEXTBOOK T=0 ONE
 * ----------------------------------------------------------------
 * CoQuí is finite-T exclusively (user ruling 2026-08-12), so the deliverable is the
 * exact closed form of the BOSONIC MATSUBARA SUM at temperature 1/beta. The T = 0
 * contour-deformation expression of the parent plan (its eq 5: an imaginary-axis
 * quadrature plus the residues of the G poles swept by the deformation) is only the
 * beta -> infinity LIMIT of what is implemented here, and appears in this increment
 * solely as gate QM2-a(iii). Nothing in the code path below deforms a contour at
 * run time; the contour argument is what DERIVES the formula, once, on paper.
 *
 * DEFINITION AND DERIVATION (verify mechanically; the gates pin every sign)
 * ------------------------------------------------------------------------
 *     Sigma^c(z) = -(1/beta) sum_m G(z + i nu_m) W^c(i nu_m),     nu_m = 2 pi m / beta
 *
 * with the pole representations
 *
 *     G(z)   = sum_l P_l / (z - eps_l)        (QP-pole G, P_l >= 0 diagonal weights)
 *     W^c(z) = sum_j w_j / (z - omega_j)      (ANY pole set; the w_j signs are mixed)
 *
 * Insert both and close the bosonic sum by residues. n_B(zeta) = 1/(e^{beta zeta} - 1)
 * has simple poles exactly at zeta = i nu_m with residue 1/beta, and the summand
 * h(zeta) = 1/[(zeta + z - eps_l)(zeta - omega_j)] decays as |zeta|^-2, so the large-
 * circle integral of n_B h vanishes and the residues sum to zero:
 *
 *     (1/beta) sum_m h(i nu_m) = - sum_{poles of h} Res[ n_B h ].
 *
 * h has poles at zeta = omega_j and at zeta = eps_l - z. On the fermionic axis
 * z = i w_n + mu one has e^{-i beta w_n} = -1, hence
 *
 *     n_B(eps_l - z) = 1/(-e^{beta(eps_l - mu)} - 1) = -f(eps_l),
 *
 * and both residues carry the SAME denominator up to a sign,
 * 1/(omega_j + z - eps_l) = +1/(z - (eps_l - omega_j)) and
 * 1/(eps_l - z - omega_j) = -1/(z - (eps_l - omega_j)). The two minus signs (the one
 * in front of the definition and the one from the residue theorem) cancel, leaving
 *
 *     Sigma^c(z) = sum_l sum_j P_l w_j [ n_B(omega_j) + f(eps_l) ] / (z - (eps_l - omega_j))
 *
 *     f(eps) = 1/(1 + e^{beta(eps - mu)}),   n_B(omega) = 1/(e^{beta omega} - 1).
 *
 * It is RATIONAL in z with simple real poles at eps_l - omega_j, therefore analytic off
 * the real axis and evaluable anywhere, including at real QP energies -- which is the
 * whole point of route B. Being rational with the right decay, agreement with the
 * directly summed Sigma^c at the fermionic nodes IS uniqueness of the continuation;
 * that is what gate QM2-a(i) measures (rel < 1e-10 at beta = 100, 1000, 10000).
 *
 * CONVENTION FOR mu (matters, and the mu = 0 fixtures cannot see it). The pole
 * energies eps_l are ABSOLUTE, so the Matsubara evaluation point is z = i w_n + mu:
 * G(i w_n + mu) = sum_l P_l / (i w_n - (eps_l - mu)). That is the reading under which
 * n_B(eps_l - z) = -f(eps_l) holds with f measured from mu, and it is the reading the
 * caller must use when feeding Matsubara points.
 *
 * NEGATIVE omega_j and the T -> 0 limit. Negative poles need no special case:
 * n_B(-omega) = -1 - n_B(omega) is already built into the formula. For a W^c given as
 * explicit PH-symmetric pairs (+Omega, +d) and (-Omega, -d) the l-th contribution reads
 *
 *     d (n_B(Omega) + f) / (z - eps_l + Omega) + d (n_B(Omega) + 1 - f) / (z - eps_l - Omega),
 *
 * whose beta -> infinity limit (n_B -> 0, f -> theta(mu - eps_l)) is the familiar T = 0
 * plasmon-pole self-energy: occupied states shifted DOWN by Omega, empty states UP.
 *
 * SMALL-|omega_j| GUARD (cd_opts::bw_floor)
 * -----------------------------------------
 * n_B(omega_j) diverges as 1/(beta omega_j), so |beta omega_j| is checked against a
 * floor. Production pole grids satisfy this by construction (imag_axes_ft::dlr_pole_fit
 * asserts min |hw_l| > 1e-12 at build), and fixtures with explicit +/- pairs must keep
 * Omega away from zero. Worth knowing WHY the production convention is safe rather than
 * merely legal: the bosonic residues of iaft_dconv.hpp carry a tanh factor,
 * w_l = tanh(hw_l/2) * coeff_l, and
 *
 *     w_l n_B(omega_l) = coeff_l tanh(hw_l/2)/(e^{hw_l} - 1) = coeff_l n_F(omega_l),
 *
 * bounded by |coeff_l| -- the 1/omega of n_B is cancelled EXACTLY by the tanh. The
 * divergence that survives at small nodes is not this one but the DENOMINATOR
 * z - (eps_l - omega_j) when z is real; see the caveat below.
 *
 * FITTED W^c AT REAL z -- THE STANDING CAVEAT (measured, gate QM2-b)
 * -----------------------------------------------------------------
 * When (w_j, omega_j) come from a DLR pole fit of W^c rather than from exact poles, the
 * fit is exact-to-eps ON THE IMAGINARY AXIS only (the QM1-e lesson, recorded in
 * test_qp_maps_matsubara.cpp). This kernel at real z evaluates the fitted measure at the
 * REAL arguments eps_l - z: the f-weighted part of the sum is exactly -f(eps_l) W^c_fit(eps_l - z).
 * If the fit has spurious residues on auxiliary nodes near eps_l - z, they are divided by
 * a vanishing gap. The cure, pinned by gate QM2-b, is to fit on the SUPPORT of W^c
 * (auxiliary nodes inside the PH gap dropped), which is legitimate because it is prior
 * physical information about W^c, not a tuned regularization. QM3 must carry that
 * constraint into the production fit.
 */

#include <cmath>
#include <complex>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"

namespace methods {
namespace sigma_route_b {

  /** Kernel options. */
  struct cd_opts {
    double bw_floor = 1e-8;   // floor on |beta * omega_j|: n_B diverges as 1/(beta omega)
  };

  /**
   * Stable occupations. Same algebra as imag_axes_ft::dconv_detail::stable_nF/nB;
   * repeated here (three lines each) so the kernel carries no dependency on the
   * double-convolution / DLR headers -- it is usable with any pole source.
   */
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
   * The kernel: Sigma^c(z) = sum_{l,j} P_l w(l,j) [n_B(om_j) + f(eps_l)] / (z - (eps_l - om_j)).
   *
   * eps(l), P(l): energies and weights of the G poles (eps ABSOLUTE, see the header note
   * on mu). om(j): the W^c pole energies, REAL. w(l,j): the W^c residues seen by internal
   * state l -- l-independent for the unit-level toys, the fitted residues of
   * W^c_{il,lj}(i nu) in production (QM3). w may be real or complex.
   *
   * A weighted double loop by design: this is a scalar / per-matrix-element kernel and
   * the production contraction over (i, j, l) is QM3's business, not this file's.
   */
  inline ComplexType sigma_cd(ComplexType z,
                              nda::MemoryArrayOfRank<1> auto const &eps,
                              nda::MemoryArrayOfRank<1> auto const &P,
                              nda::MemoryArrayOfRank<2> auto const &w,
                              nda::MemoryArrayOfRank<1> auto const &om,
                              double beta, double mu, cd_opts const &opt = {}) {
    const long nl = eps.shape(0), nj = om.shape(0);
    utils::check(beta > 0.0, "sigma_route_b::sigma_cd: beta = {} must be > 0.", beta);
    utils::check(P.shape(0) == nl,
                 "sigma_route_b::sigma_cd: {} weights for {} G poles.", P.shape(0), nl);
    utils::check(w.shape(0) == nl and w.shape(1) == nj,
                 "sigma_route_b::sigma_cd: residue block is {} x {}, needs {} x {} "
                 "(G poles x W^c poles).", w.shape(0), w.shape(1), nl, nj);

    nda::array<double, 1> nb(nj);
    for (long j = 0; j < nj; ++j) {
      utils::check(std::abs(beta * om(j)) >= opt.bw_floor,
                   "sigma_route_b::sigma_cd: |beta*omega_j| = {:.3e} at j = {} (omega_j = "
                   "{:.6e}) is below the floor {:.1e}; n_B(omega_j) diverges there. Production "
                   "pole grids guarantee a nonzero node; fixtures must keep Omega off zero.",
                   std::abs(beta * om(j)), j, om(j), opt.bw_floor);
      nb(j) = stable_nB(beta, om(j));
    }

    ComplexType s(0.0);
    for (long l = 0; l < nl; ++l) {
      const double f = stable_nF(beta, eps(l) - mu);
      for (long j = 0; j < nj; ++j)
        s += P(l) * w(l, j) * (nb(j) + f) / (z - (eps(l) - om(j)));
    }
    return s;
  }

  /** Convenience overload for an l-INDEPENDENT residue vector w(j) (the toys, and any
   *  diagonal production case where W^c does not resolve the internal state). */
  inline ComplexType sigma_cd(ComplexType z,
                              nda::MemoryArrayOfRank<1> auto const &eps,
                              nda::MemoryArrayOfRank<1> auto const &P,
                              nda::MemoryArrayOfRank<1> auto const &w,
                              nda::MemoryArrayOfRank<1> auto const &om,
                              double beta, double mu, cd_opts const &opt = {}) {
    const long nl = eps.shape(0), nj = om.shape(0);
    utils::check(w.shape(0) == nj,
                 "sigma_route_b::sigma_cd: {} residues for {} W^c poles.", w.shape(0), nj);
    nda::array<std::decay_t<decltype(w(0))>, 2> W(nl, nj);
    for (long l = 0; l < nl; ++l)
      for (long j = 0; j < nj; ++j) W(l, j) = w(j);
    return sigma_cd(z, eps, P, W, om, beta, mu, opt);
  }

  /** one-line provenance for the run log. */
  inline void log_kernel(int level, std::string_view who, long nl, long nj,
                         double beta, double mu, double min_abs_bw) {
    app_log(level, "  {}: route-B CD kernel, {} G poles x {} W^c poles, beta = {:.4g}, "
                   "mu = {:.6g}, min |beta*omega_j| = {:.3e}",
            who, nl, nj, beta, mu, min_abs_bw);
  }

} // sigma_route_b
} // methods

#endif // COQUI_SIGMA_ROUTE_B_HPP
