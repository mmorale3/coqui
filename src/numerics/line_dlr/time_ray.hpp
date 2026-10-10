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

#ifndef COQUI_NUMERICS_LINE_DLR_TIME_RAY_HPP
#define COQUI_NUMERICS_LINE_DLR_TIME_RAY_HPP

/**
 * Complex-time rays and the transform to the line (notes section 4.1-4.2, Eq. laplace):
 *
 *   particle ray t = s e^{-i theta_t}, hole ray t = s e^{+i theta_t},  0 < theta_t < theta,
 *   X^{>,<}(zeta) = -i int_0^inf dt e^{i zeta t} X^{>,<}(t) = sum_m F(zeta, m) X(t_m),
 *   F(zeta, m) = -i phase ws_m e^{i zeta t_m}.
 *
 * Generic quadrature: composite Gauss-Legendre in s on log-spaced panels, the FIRST panel starting at s = 0,
 * edges [0] ++ exp(linspace(log smin, log smax, int(log(smax/smin) per_efold) + 2)).
 * Transcription of coqui/cayley/cayley/line/timeray.py::TimeRay.
 *
 * Finite temperature (S8b, notes section 11.3): `guarded` = the ray truncated at the beta guard S_T = beta / sin(theta_t)
 * (Im t = -tau in [0, beta]) with log panels wider than hmax split evenly (python TimeRay(..., hmax)); the GL sum on
 * [0, S_T] is then the truncated transform T_S of Eq. fT_TS. `tau_half` = the first half of the imaginary-time grid of the
 * finite-T tau leg (python timeray.tau_grid on [0, beta / 2]) as a "ray" at theta_t = pi / 2 (t = -i tau on the particle
 * side, +i tau on the hole side): the standard transform_matrix at zeta = i nu_n is then -w e^{i nu tau} (particle) /
 * w e^{-i nu tau} (hole), the Matsubara transform of the tau leg (polarization.hpp, thermal.hpp).
 */

#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include "numerics/line_dlr/line_dlr_utils.hpp"

namespace numerics::line_dlr {

struct time_ray_t {
  double theta_t = 0.0;
  sector_t sector = sector_t::particle;
  ComplexType phase = ComplexType(1.0, 0.0);   ///< dt/ds: e^{-i theta_t} (particle) or e^{+i theta_t} (hole)
  nda::array<double, 1> s;                     ///< (nt) real ray coordinate
  nda::array<double, 1> ws;                    ///< (nt) quadrature weights in s
  nda::array<ComplexType, 1> t;                ///< (nt) complex times s * phase

  time_ray_t() = default;

  /// hmax > 0 (S8b): log panels wider than hmax are split into ceil(width / hmax) equal panels (python TimeRay hmax)
  time_ray_t(double theta_t_, double smax, double smin = 1e-5, double per_efold = 3.0, long nn = 16,
             sector_t sector_ = sector_t::particle, double hmax = 0.0)
     : theta_t(theta_t_), sector(sector_) {
    utils::check(sector != sector_t::both, "time_ray_t: sector must be particle or hole");
    utils::check(smax > smin and smin > 0.0, "time_ray_t: need 0 < smin < smax (got {} {})", smin, smax);
    nda::array<double, 1> xg, wg;
    detail::gauss_legendre(nn, xg, wg);
    const long ne = long(std::log(smax / smin) * per_efold) + 2;   // python int() truncates (argument > 0)
    auto lg       = detail::logspace(smin, smax, ne);
    nda::array<double, 1> edges(ne + 1);
    edges(0) = 0.0;
    for (long i = 0; i < ne; ++i) edges(i + 1) = lg(i);
    if (hmax > 0.0) {   // python: concatenate([[e0]] + [linspace(a, b, ceil((b - a) / hmax) + 1)[1:] for a, b in panels])
      std::vector<double> ed{edges(0)};
      for (long i = 0; i < ne; ++i) {
        const double a = edges(i), b = edges(i + 1);
        const long m   = long(std::ceil((b - a) / hmax));
        auto ls        = detail::linspace(a, b, m + 1);
        for (long j = 1; j <= m; ++j) ed.push_back(ls(j));
      }
      edges = nda::array<double, 1>(long(ed.size()));
      for (long i = 0; i < long(ed.size()); ++i) edges(i) = ed[i];
    }
    const long npan = edges.size() - 1;
    s  = nda::array<double, 1>(npan * nn);
    ws = nda::array<double, 1>(npan * nn);
    for (long p = 0; p < npan; ++p) {
      const double mid = (edges(p + 1) + edges(p)) / 2.0, h = (edges(p + 1) - edges(p)) / 2.0;
      for (long k = 0; k < nn; ++k) {
        s(p * nn + k)  = mid + h * xg(k);
        ws(p * nn + k) = h * wg(k);
      }
    }
    phase = (sector == sector_t::particle) ? std::exp(ComplexType(0.0, -theta_t)) : std::exp(ComplexType(0.0, theta_t));
    t     = nda::array<ComplexType, 1>(s.size());
    for (long m = 0; m < s.size(); ++m) t(m) = s(m) * phase;
  }

  /// Nodes given explicitly (s, ws): t = s e^{-+ i theta_t}
  static time_ray_t from_nodes(double theta_t, sector_t sector, nda::array<double, 1> const &s_, nda::array<double, 1> const &ws_) {
    utils::check(sector != sector_t::both and s_.size() == ws_.size(), "time_ray_t::from_nodes: invalid input");
    time_ray_t r;
    r.theta_t = theta_t;
    r.sector  = sector;
    r.s       = s_;
    r.ws      = ws_;
    r.phase   = (sector == sector_t::particle) ? std::exp(ComplexType(0.0, -theta_t)) : std::exp(ComplexType(0.0, theta_t));
    r.t       = nda::array<ComplexType, 1>(r.s.size());
    for (long m = 0; m < r.s.size(); ++m) r.t(m) = r.s(m) * r.phase;
    return r;
  }

  /// S8b: finite-T ray truncated at the beta guard S_T = beta / sin(theta_t) (Eq. fT_guard); hmax <= 0: python's default
  /// pi / (E_T cos theta_t) (one period of the fastest window-window pair |E| = 2 E_T per panel) with E_T = c_T / beta
  static time_ray_t guarded(double theta_t, double beta, double E_T, double smin = 1e-5, double per_efold = 3.0, long nn = 16,
                            sector_t sector = sector_t::particle, double hmax = -1.0) {
    if (hmax <= 0.0) hmax = std::numbers::pi / (E_T * std::cos(theta_t));
    return time_ray_t(theta_t, beta / std::sin(theta_t), smin, per_efold, nn, sector, hmax);
  }

  /**
   * S8b tau leg: python timeray.tau_grid(beta, emax, nn, per_efold, x0) restricted to [0, beta / 2] (its first half; the
   * second half is the mirror tau -> beta - tau, used through Pi(q, beta - tau) = Pi(-q, tau)^T): composite GL with panel
   * edges {0} U {x0 / emax e^{j / per_efold}} up to beta / 2. As a ray at theta_t = pi / 2 (see the file header).
   */
  static time_ray_t tau_half(double beta, double emax, long nn = 12, double per_efold = 2.0, double x0 = 0.02,
                             sector_t sector = sector_t::particle) {
    nda::array<double, 1> xg, wg;
    detail::gauss_legendre(nn, xg, wg);
    const double half = 0.5 * beta, a0 = x0 / emax;
    nda::array<double, 1> edges;
    if (a0 >= half) {
      edges = nda::array<double, 1>(2);
      edges(0) = 0.0;
      edges(1) = half;
    } else {
      const long ne = long(std::ceil(std::log(half / a0) * per_efold)) + 1;
      auto lg       = detail::logspace(a0, half, ne);
      edges         = nda::array<double, 1>(ne + 1);
      edges(0)      = 0.0;
      for (long i = 0; i < ne; ++i) edges(i + 1) = lg(i);
    }
    const long npan = edges.size() - 1;
    nda::array<double, 1> s(npan * nn), ws(npan * nn);
    for (long p = 0; p < npan; ++p) {
      const double mid = (edges(p + 1) + edges(p)) / 2.0, h = (edges(p + 1) - edges(p)) / 2.0;
      for (long k = 0; k < nn; ++k) {
        s(p * nn + k)  = mid + h * xg(k);
        ws(p * nn + k) = h * wg(k);
      }
    }
    return from_nodes(std::numbers::pi / 2.0, sector, s, ws);
  }

  /// smax from the smallest |pole energy| emin: e^{-emin smax sin(theta_t)} = e^{-decades}.
  static time_ray_t for_spectrum(double theta_t, double emin, double decades = 40.0, double smin = 1e-5,
                                 double per_efold = 3.0, long nn = 16, sector_t sector = sector_t::particle) {
    return time_ray_t(theta_t, decades / (emin * std::sin(theta_t)), smin, per_efold, nn, sector);
  }

  long size() const { return s.size(); }

  /// e^{-i E_p t_m} -> [nt, np]
  nda::array<ComplexType, 2> exponentials(nda::array<double, 1> const &E) const {
    const long nt = size(), np = E.size();
    nda::array<ComplexType, 2> X(nt, np);
    for (long m = 0; m < nt; ++m)
      for (long p = 0; p < np; ++p) X(m, p) = std::exp(ComplexType(0.0, -E(p)) * t(m));
    return X;
  }

  /// F[nz, nt] with X(zeta) = F X(t):  F = -i phase ws e^{i zeta t}
  nda::array<ComplexType, 2> transform_matrix(nda::array<ComplexType, 1> const &zeta) const {
    const long nz = zeta.size(), nt = size();
    const ComplexType I(0.0, 1.0), pre = -I * phase;
    nda::array<ComplexType, 2> F(nz, nt);
    for (long i = 0; i < nz; ++i)
      for (long m = 0; m < nt; ++m) F(i, m) = pre * ws(m) * std::exp(I * zeta(i) * t(m));
    return F;
  }
};

} // namespace numerics::line_dlr

#endif
