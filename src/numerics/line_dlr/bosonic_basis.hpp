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

#ifndef COQUI_NUMERICS_LINE_DLR_BOSONIC_BASIS_HPP
#define COQUI_NUMERICS_LINE_DLR_BOSONIC_BASIS_HPP

/**
 * Odd-symmetric real-pole basis for bosonic functions with B(-nu) = -B(nu)^T (notes section 3.3, Eqs. brep, bfit):
 *
 *   W_PQ(zeta) = sum_j [ w_j,PQ / (zeta - nu_j) - w_j,QP / (zeta + nu_j) ],   nu_j > 0.
 *
 * The hole part is tied to the particle part, so the sector split is exact. Poles by pivoted QR of the stacked kernel
 * [K^-; -K^+] on the two upper rays; line nodes (2r) by row-pivoted QR of the odd kernel; the fit solves the coupled
 * (PQ, QP) system for all unordered pairs with ONE gelss factorization.
 * Transcription of coqui/cayley/cayley/line/line_dlr.py::BosonicLineBasis.
 */

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "numerics/line_dlr/line_dlr_utils.hpp"
#include "numerics/line_dlr/line_basis.hpp"

namespace numerics::line_dlr {

struct bosonic_basis_t {
  double theta = 0.0, lam = 0.0, eps = 0.0, gap = 0.0;
  long rank = 0;
  nda::array<double, 1> nu;                    ///< (rank) positive poles, sorted ascending
  nda::array<ComplexType, 1> zeta_nodes;       ///< (min(2 rank, 2 nline)) line nodes (mu-relative)
  nda::array<ComplexType, 1> zeta_dense;       ///< (2 nline) the dense selection grid

  bosonic_basis_t(double theta_, double lam_, double eps_, double gap_, double tmin = -1.0, double tmax = -1.0,
                  long nline = 1200, long npole = 800)
     : theta(theta_), lam(lam_), eps(eps_), gap(gap_) {
    utils::check(lam > 0.0 and eps > 0.0 and nline > 1 and npole > 1, "bosonic_basis_t: invalid parameters");
    if (tmin < 0.0) tmin = 1e-4 * lam;
    if (tmax < 0.0) tmax = 20.0 * lam;
    zeta_dense = dense_nodes(theta, tmin, tmax, nline);
    const long nd = zeta_dense.size();
    auto nuc = detail::logspace(std::max(gap, 1e-4 * lam), lam, npole);

    // stacked kernel [1/(z - nu) ; -1/(z + nu)], column-normalized, column-pivoted QR
    nda::matrix<ComplexType, nda::F_layout> Kn(2 * nd, npole);
    for (long j = 0; j < npole; ++j) {
      double nrm = 0.0;
      for (long i = 0; i < nd; ++i) {
        Kn(i, j)      = 1.0 / (zeta_dense(i) - nuc(j));
        Kn(nd + i, j) = -1.0 / (zeta_dense(i) + nuc(j));
        nrm += std::norm(Kn(i, j)) + std::norm(Kn(nd + i, j));
      }
      nrm = std::sqrt(nrm);
      for (long i = 0; i < 2 * nd; ++i) Kn(i, j) /= nrm;
    }
    auto q = detail::pivoted_qr(Kn);
    rank   = detail::qr_rank(q, eps);
    std::vector<double> ns(rank);
    for (long l = 0; l < rank; ++l) ns[l] = nuc(q.piv[l]);
    std::sort(ns.begin(), ns.end());
    nu = nda::array<double, 1>(rank);
    for (long l = 0; l < rank; ++l) nu(l) = ns[l];

    // line nodes: row-pivoted QR of the odd kernel 1/(z - nu) - 1/(z + nu); min(2r, nd) nodes
    nda::matrix<ComplexType, nda::F_layout> KT(rank, nd);
    for (long i = 0; i < nd; ++i)
      for (long l = 0; l < rank; ++l) KT(l, i) = 1.0 / (zeta_dense(i) - nu(l)) - 1.0 / (zeta_dense(i) + nu(l));
    auto q2       = detail::pivoted_qr(KT);
    const long nn = std::min(2 * rank, nd);
    std::vector<long> rows(q2.piv.begin(), q2.piv.begin() + nn);
    std::sort(rows.begin(), rows.end());
    zeta_nodes = nda::array<ComplexType, 1>(nn);
    for (long l = 0; l < nn; ++l) zeta_nodes(l) = zeta_dense(rows[l]);
  }

  long size() const { return rank; }

  /// (Km, Kp): Km[i, j] = 1/(zeta_i - nu_j), Kp[i, j] = 1/(zeta_i + nu_j)
  std::pair<nda::array<ComplexType, 2>, nda::array<ComplexType, 2>> kernels(nda::array<ComplexType, 1> const &zeta) const {
    const long nz = zeta.size();
    nda::array<ComplexType, 2> Km(nz, rank), Kp(nz, rank);
    for (long i = 0; i < nz; ++i)
      for (long j = 0; j < rank; ++j) {
        Km(i, j) = 1.0 / (zeta(i) - nu(j));
        Kp(i, j) = 1.0 / (zeta(i) + nu(j));
      }
    return {std::move(Km), std::move(Kp)};
  }

  /**
   * Residues w[r, N, N] of the positive poles from samples W[nz, N, N] at mu-relative zeta (Eq. bfit):
   *   [[Km, -Kp], [-Kp, Km]] [w_PQ; w_QP] = [W_PQ(zeta); W_QP(zeta)]   for all P <= Q (row-major upper triangle),
   * one gelss with all pairs as right-hand sides, rcond = DBL_EPSILON max(2 nz, 2 r).
   */
  nda::array<ComplexType, 3> fit(nda::array<ComplexType, 1> const &zeta, nda::array<ComplexType, 3> const &W,
                                 double rcond = -1.0) const {
    const long nz = W.extent(0), N = W.extent(1);
    utils::check(nz == zeta.size() and W.extent(2) == N, "bosonic_basis_t::fit: W shape mismatch");
    auto [Km, Kp] = kernels(zeta);
    const long r = rank;
    nda::array<ComplexType, 2> A(2 * nz, 2 * r);
    for (long i = 0; i < nz; ++i)
      for (long j = 0; j < r; ++j) {
        A(i, j)          = Km(i, j);
        A(i, r + j)      = -Kp(i, j);
        A(nz + i, j)     = -Kp(i, j);
        A(nz + i, r + j) = Km(i, j);
      }
    std::vector<std::pair<long, long>> pairs;
    for (long P = 0; P < N; ++P)
      for (long Q = P; Q < N; ++Q) pairs.emplace_back(P, Q);
    const long np = pairs.size();
    nda::array<ComplexType, 2> data(2 * nz, np);
    for (long k = 0; k < np; ++k) {
      auto [P, Q] = pairs[k];
      for (long i = 0; i < nz; ++i) {
        data(i, k)      = W(i, P, Q);
        data(nz + i, k) = W(i, Q, P);
      }
    }
    auto sol = detail::lstsq(A, data, rcond);                    // (2r, np)
    nda::array<ComplexType, 3> w(r, N, N);
    w() = ComplexType(0.0);
    for (long k = 0; k < np; ++k) {
      auto [P, Q] = pairs[k];
      for (long j = 0; j < r; ++j) w(j, P, Q) = sol(j, k);
      for (long j = 0; j < r; ++j) w(j, Q, P) = sol(r + j, k);  // python order: the QP write wins on the diagonal
    }
    return w;
  }

  /// W(zeta)[nz, N, N]; sector particle: Km w only; hole: -Kp w^T only (transpose on (P, Q)); both: the sum.
  nda::array<ComplexType, 3> eval(nda::array<ComplexType, 3> const &w, nda::array<ComplexType, 1> const &zeta,
                                  sector_t sector = sector_t::both) const {
    const long r = w.extent(0), N = w.extent(1), nz = zeta.size();
    utils::check(r == rank and w.extent(2) == N, "bosonic_basis_t::eval: w shape mismatch");
    auto [Km, Kp] = kernels(zeta);
    nda::array<ComplexType, 2> out(nz, N * N);
    out() = ComplexType(0.0);
    if (sector != sector_t::hole) {
      nda::array<ComplexType, 2> wp = nda::reshape(w, std::array<long, 2>{r, N * N});
      out += detail::matmul(Km, wp);
    }
    if (sector != sector_t::particle) {
      nda::array<ComplexType, 2> wt(r, N * N);
      for (long j = 0; j < r; ++j)
        for (long P = 0; P < N; ++P)
          for (long Q = 0; Q < N; ++Q) wt(j, P * N + Q) = w(j, Q, P);
      out -= detail::matmul(Kp, wt);
    }
    return nda::array<ComplexType, 3>(nda::reshape(out, std::array<long, 3>{nz, N, N}));
  }

  /// [nt, r]: e^{-i nu_j t} (particle, W^>(t) = sum_j w_j e^{-i nu_j t}) or -e^{+i nu_j t} (hole, W^<(t) = sum_j w_j^T (.)).
  nda::array<ComplexType, 2> time_exponentials(nda::array<ComplexType, 1> const &t, sector_t sector) const {
    utils::check(sector != sector_t::both, "bosonic_basis_t::time_exponentials: sector must be particle or hole");
    const long nt = t.size();
    nda::array<ComplexType, 2> E(nt, rank);
    const ComplexType I(0.0, 1.0);
    for (long i = 0; i < nt; ++i)
      for (long j = 0; j < rank; ++j)
        E(i, j) = (sector == sector_t::particle) ? std::exp(-I * nu(j) * t(i)) : -std::exp(I * nu(j) * t(i));
    return E;
  }
};

} // namespace numerics::line_dlr

#endif
