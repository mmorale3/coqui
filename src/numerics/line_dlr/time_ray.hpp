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
 */

#include <cmath>
#include <complex>

#include "numerics/line_dlr/line_dlr_utils.hpp"

namespace numerics::line_dlr {

struct time_ray_t {
  double theta_t = 0.0;
  sector_t sector = sector_t::particle;
  ComplexType phase = ComplexType(1.0, 0.0);   ///< dt/ds: e^{-i theta_t} (particle) or e^{+i theta_t} (hole)
  nda::array<double, 1> s;                     ///< (nt) real ray coordinate
  nda::array<double, 1> ws;                    ///< (nt) quadrature weights in s
  nda::array<ComplexType, 1> t;                ///< (nt) complex times s * phase

  time_ray_t(double theta_t_, double smax, double smin = 1e-5, double per_efold = 3.0, long nn = 16,
             sector_t sector_ = sector_t::particle)
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
    const long npan = ne;
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
