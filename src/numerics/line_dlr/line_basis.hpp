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

#ifndef COQUI_NUMERICS_LINE_DLR_LINE_BASIS_HPP
#define COQUI_NUMERICS_LINE_DLR_LINE_BASIS_HPP

/**
 * Fermionic real-pole basis on the tilted line zeta = mu + t e^{i theta} (notes section 3.2, Eqs. fkernel, frep).
 * Transcription of coqui/cayley/cayley/line/line_dlr.py::LineBasis.
 *
 * Data live on the two upper rays (angles theta and pi - theta). Candidate poles on log grids over
 * [-lam, -gap_minus] U [gap_plus, lam]; poles = column-pivoted QR of the column-normalized kernel K = 1/(zeta_i - w)
 * on a dense line grid (rank at eps); line nodes = row-pivoted QR of K(., w). A function is represented as
 * X(zeta) = sum_l c_l / (zeta - w_l), c_l signed; coefficients from a least-squares fit of samples.
 * All energies are mu-relative.
 *
 * Dense-node rule (notes 3.2): fits must use samples covering well beyond the pole support, e.g.
 * dense_nodes(theta, 1e-3, 60, 120); fits on the r QR nodes alone pin the total function but not the weight
 * distribution of the far poles (moments then fail at 1e-3).
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

#include "numerics/line_dlr/line_dlr_utils.hpp"

namespace numerics::line_dlr {

/// 2 n_per_ray mu-relative nodes: [t e^{i theta}, t e^{i (pi - theta)}], t = logspace(tmin, tmax, n_per_ray).
inline nda::array<ComplexType, 1> dense_nodes(double theta, double tmin, double tmax, long n_per_ray) {
  auto t = detail::logspace(tmin, tmax, n_per_ray);
  const ComplexType ep = std::exp(ComplexType(0.0, theta));
  const ComplexType em = std::exp(ComplexType(0.0, std::numbers::pi - theta));
  nda::array<ComplexType, 1> z(2 * n_per_ray);
  for (long i = 0; i < n_per_ray; ++i) {
    z(i)             = t(i) * ep;
    z(n_per_ray + i) = t(i) * em;
  }
  return z;
}

struct line_basis_t {
  double theta = 0.0, lam = 0.0, eps = 0.0;
  std::array<double, 2> gap = {0.0, 0.0};      ///< (D_minus, D_plus): no poles in (-D_minus, D_plus)
  long rank = 0;
  nda::array<double, 1> w;                     ///< (rank) real poles, sorted ascending (mu-relative)
  nda::array<ComplexType, 1> zeta_nodes;       ///< (rank) row-pivot line nodes (mu-relative)
  nda::array<ComplexType, 1> zeta_dense;       ///< (2 nline) the dense selection grid

  /**
   * @param theta     tilt angle (rad)
   * @param lam       pole range |w| <= lam
   * @param gap_minus / gap_plus  no poles in (-gap_minus, gap_plus); a gap >= lam removes that side entirely
   * @param tmin,tmax |t| range of the dense line grid; < 0 -> 1e-4 lam, 20 lam
   */
  line_basis_t(double theta_, double lam_, double eps_, double gap_minus, double gap_plus, double tmin = -1.0,
               double tmax = -1.0, long nline = 1200, long npole = 1500)
     : theta(theta_), lam(lam_), eps(eps_), gap{gap_minus, gap_plus} {
    utils::check(lam > 0.0 and eps > 0.0 and nline > 1 and npole > 1, "line_basis_t: invalid parameters");
    if (tmin < 0.0) tmin = 1e-4 * lam;
    if (tmax < 0.0) tmax = 20.0 * lam;
    zeta_dense = dense_nodes(theta, tmin, tmax, nline);
    const long nd = zeta_dense.size();

    // candidate poles (python: gp, gm, w = [-gm[::-1], (0,) gp])
    const double Dm = gap_minus, Dp = gap_plus;
    nda::array<double, 1> gp, gm;
    if (Dp < lam) gp = detail::logspace(std::max(Dp, 1e-4 * lam), lam, npole / 2);
    if (Dm < lam) gm = detail::logspace(std::max(Dm, 1e-4 * lam), lam, npole / 2);
    std::vector<double> wc;
    for (long i = gm.size() - 1; i >= 0; --i) wc.push_back(-gm(i));
    if (Dm <= 0.0 and Dp <= 0.0) wc.push_back(0.0);
    for (long i = 0; i < gp.size(); ++i) wc.push_back(gp(i));
    const long nc = wc.size();
    utils::check(nc > 0, "line_basis_t: no candidate poles (both gaps >= lam)");

    // column-normalized kernel, column-pivoted QR -> rank and poles
    nda::matrix<ComplexType, nda::F_layout> Kn(nd, nc);
    for (long j = 0; j < nc; ++j) {
      double nrm = 0.0;
      for (long i = 0; i < nd; ++i) {
        Kn(i, j) = 1.0 / (zeta_dense(i) - wc[j]);
        nrm += std::norm(Kn(i, j));
      }
      nrm = std::sqrt(nrm);
      for (long i = 0; i < nd; ++i) Kn(i, j) /= nrm;
    }
    auto q = detail::pivoted_qr(Kn);
    rank   = detail::qr_rank(q, eps);
    std::vector<double> ws(rank);
    for (long l = 0; l < rank; ++l) ws[l] = wc[q.piv[l]];
    std::sort(ws.begin(), ws.end());
    w = nda::array<double, 1>(rank);
    for (long l = 0; l < rank; ++l) w(l) = ws[l];

    // row-pivoted QR of K(., w): geqp3 of K^T (rank x nd) -> nodes = zeta_dense at the sorted first rank pivots
    nda::matrix<ComplexType, nda::F_layout> KT(rank, nd);
    for (long i = 0; i < nd; ++i)
      for (long l = 0; l < rank; ++l) KT(l, i) = 1.0 / (zeta_dense(i) - w(l));
    auto q2 = detail::pivoted_qr(KT);
    std::vector<long> rows(q2.piv.begin(), q2.piv.begin() + rank);
    std::sort(rows.begin(), rows.end());
    zeta_nodes = nda::array<ComplexType, 1>(rank);
    for (long l = 0; l < rank; ++l) zeta_nodes(l) = zeta_dense(rows[l]);
  }

  long size() const { return rank; }
  bool is_particle(long l) const { return w(l) > 0.0; }   ///< sector of pole l (python: pos = w > 0)

  /// K[i, l] = 1/(zeta_i - w_l)
  nda::array<ComplexType, 2> kernel(nda::array<ComplexType, 1> const &zeta) const {
    const long nz = zeta.size();
    nda::array<ComplexType, 2> K(nz, rank);
    for (long i = 0; i < nz; ++i)
      for (long l = 0; l < rank; ++l) K(i, l) = 1.0 / (zeta(i) - w(l));
    return K;
  }

  /// LS coefficients c[r, ncol] from samples X[nz, ncol] at mu-relative zeta (gelss, rcond = DBL_EPSILON max(nz, r)).
  nda::array<ComplexType, 2> fit(nda::array<ComplexType, 1> const &zeta, nda::array<ComplexType, 2> const &X,
                                 double rcond = -1.0) const {
    utils::check(X.extent(0) == zeta.size(), "line_basis_t::fit: X has {} rows for {} points", X.extent(0), zeta.size());
    return detail::lstsq(kernel(zeta), X, rcond);
  }

  /// [nz, n, n] -> [r, n, n]
  nda::array<ComplexType, 3> fit(nda::array<ComplexType, 1> const &zeta, nda::array<ComplexType, 3> const &X,
                                 double rcond = -1.0) const {
    const long nz = X.extent(0), n1 = X.extent(1), n2 = X.extent(2);
    nda::array<ComplexType, 2> X2 = nda::reshape(X, std::array<long, 2>{nz, n1 * n2});
    auto c = fit(zeta, X2, rcond);
    return nda::array<ComplexType, 3>(nda::reshape(c, std::array<long, 3>{rank, n1, n2}));
  }

  /// X(zeta)[nz, ncol] = sum_l c_l / (zeta - w_l)
  nda::array<ComplexType, 2> eval(nda::array<ComplexType, 2> const &c, nda::array<ComplexType, 1> const &zeta) const {
    utils::check(c.extent(0) == rank, "line_basis_t::eval: c has {} rows, rank {}", c.extent(0), rank);
    return detail::matmul(kernel(zeta), c);
  }

  nda::array<ComplexType, 3> eval(nda::array<ComplexType, 3> const &c, nda::array<ComplexType, 1> const &zeta) const {
    const long n1 = c.extent(1), n2 = c.extent(2), nz = zeta.size();
    nda::array<ComplexType, 2> c2 = nda::reshape(c, std::array<long, 2>{c.extent(0), n1 * n2});
    auto X = eval(c2, zeta);
    return nda::array<ComplexType, 3>(nda::reshape(X, std::array<long, 3>{nz, n1, n2}));
  }

  /// indices of the hole (w <= 0, python ~pos) and particle (w > 0) poles
  std::vector<long> hole_indices() const {
    std::vector<long> v;
    for (long l = 0; l < rank; ++l) if (not is_particle(l)) v.push_back(l);
    return v;
  }
  std::vector<long> particle_indices() const {
    std::vector<long> v;
    for (long l = 0; l < rank; ++l) if (is_particle(l)) v.push_back(l);
    return v;
  }

  /// (c_hole, c_particle): the rows of c for the poles below / above the centre (sector split; python split).
  /// Pair with hole_poles() / particle_poles() to evaluate a sector.
  template <int R>
  std::pair<nda::array<ComplexType, R>, nda::array<ComplexType, R>> split(nda::array<ComplexType, R> const &c) const {
    static_assert(R >= 1);
    utils::check(c.extent(0) == rank, "line_basis_t::split: c has {} rows, rank {}", c.extent(0), rank);
    auto pick = [&](std::vector<long> const &idx) {
      auto shp = c.shape();
      shp[0]   = long(idx.size());
      nda::array<ComplexType, R> out(shp);
      const long blk = (rank > 0) ? c.size() / rank : 0;
      for (long k = 0; k < long(idx.size()); ++k)
        std::copy_n(c.data() + idx[k] * blk, blk, out.data() + k * blk);
      return out;
    };
    return {pick(hole_indices()), pick(particle_indices())};
  }

  nda::array<double, 1> hole_poles() const {
    auto idx = hole_indices();
    nda::array<double, 1> e(long(idx.size()));
    for (long k = 0; k < long(idx.size()); ++k) e(k) = w(idx[k]);
    return e;
  }
  nda::array<double, 1> particle_poles() const {
    auto idx = particle_indices();
    nda::array<double, 1> e(long(idx.size()));
    for (long k = 0; k < long(idx.size()); ++k) e(k) = w(idx[k]);
    return e;
  }
};

} // namespace numerics::line_dlr

#endif
