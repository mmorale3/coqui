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
 * Odd-symmetric real-pole basis for bosonic functions with W(q, -zeta) = W(-q, zeta)^T (notes section 3.3, Eqs. brep, bfit):
 *
 *   W_PQ(q, zeta) = sum_j [ w_j(q)_PQ / (zeta - nu_j) - w_j(-q)_QP / (zeta + nu_j) ],   nu_j > 0,
 *
 * i.e. the negative-frequency residues of W(q) are the TRANSPOSED positive-frequency residues of W(-q) (Pi and W of the
 * density fluctuation rho_q: B(q) = A(-q)^T). The hole part is tied to the particle part of the partner -q, so the sector
 * split is exact. For a self-inverse q (q = -q mod G) this is the per-q form w_j(q)_QP; for q != -q it is NOT (the per-q
 * form was the bug of 2026-10-04, Si 4x4x4). fit(zeta, W) / eval(w, zeta) are the self-inverse forms; fit(zeta, W, Wm) /
 * eval(w, wm, zeta) the paired forms (Wm = W(-q) at the same nodes, wm = w(-q)).
 * Poles by pivoted QR of the stacked kernel [K^-; -K^+] on the two upper rays; line nodes (2r) by row-pivoted QR of the
 * odd kernel; the fit solves the coupled (PQ, QP) system for all pairs with ONE gelss factorization.
 * Transcription of coqui/cayley/cayley/line/line_dlr.py::BosonicLineBasis.
 *
 * Mirror-symmetric line nodes (perf 7.1 (b)/(c), notes section 5): W(-q, -conj zeta) = conj W(q, zeta) (exact; Z(-q) =
 * conj Z(q), Pi(-q, -conj zeta) = conj Pi(q, zeta)), so on a node set closed under zeta -> -conj zeta the Dyson equation
 * is solved on the ray-1 nodes only and the ray-2 values are conjugates. node_factor > 0 (default: env
 * COQUI_GWLINE_BOS_NODES, else default_node_factor) selects n1 = ceil(node_factor r / 2) ray-1 nodes and stores
 *   zeta_nodes = [zeta_1 .. zeta_n1 (ray 1, ascending |zeta|), -conj zeta_1 .. -conj zeta_n1]   (nz = 2 n1),
 * by a greedy GROUP selection on the pair system of Eq. bfit: a ray-1 candidate zeta contributes the four rows
 * {a(zeta), b(zeta), conj b(zeta), conj a(zeta)} (a = [K^-, -K^+], b = [-K^+, K^-]; the last two are the rows of the
 * mirror node -conj zeta), the candidate with the largest residual group norm is taken and its rows are projected out
 * (block modified Gram-Schmidt, twice); when the pair system's rank is exhausted a fresh pass continues on the remaining
 * candidates (oversampling). node_factor <= 0: the python node selection above (asymmetric; [parity] / bases files).
 * mirror_half(zeta) detects the symmetric layout of any node set (0 if not symmetric).
 *
 * Finite temperature (S8b, notes section 11.5 sec:fT_W):
 *  - from_data(theta, D, lam, eps_b): the basis selected ON the bosonic data set D (unmasked line nodes, wedge band, i nu_n,
 *    nu_0 = 0): candidate poles on the log grid [numin, lam] (numin = 1e-4 lam), column-pivoted QR of the column-normalized
 *    stacked odd kernel [K^-; -K^+] evaluated at D, tolerance eps_b; zeta_nodes = zeta_dense = D (any layout; a layout
 *    [z_1..z_n1, -conj z_1..-conj z_n1] with z_i in the closed upper-right quadrant keeps the mirror paths, mirror_half).
 *    split_fit: the pair fit of screened.hpp is the decoupled odd / even fit of Eq. fT_splitfit (bosonic_fit_t), cutoffs
 *    cut_odd / cut_even relative to the largest singular value of the column-scaled kernels (fit_split below = the host form).
 *  - with_bose(beta, E_T): the Sigma basis of Eq. fT_W: the rows j < rank_fit() are the fitted poles with W(t) weights
 *    tw_j = 1 + n_j, the n_bose rows after them are copies of the window poles (nu_j <= E_T, n_j > 0) whose residue rows hold
 *    w_j(-q)^T (bosonic_fit_t computes them from the same data, screened.hpp) with weights n_j and the opposite exponential:
 *      particle: tw_j e^{-i nu_j t} (fitted rows), n_j e^{+i nu_j t} (extra rows);
 *      hole:    -tw_j e^{+i nu_j t},              -n_j e^{-i nu_j t},
 *    so W^>_T(q, t) = sum_rows E^>(t)_j w_j(q) and W^<_T(q, t)^T = sum_rows E^<(t)_j w_j(-q) with the unchanged kernels.
 *    tw empty (T = 0): time_exponentials is the T = 0 formula (bitwise).
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

#include "nda/blas.hpp"

#include "numerics/line_dlr/line_dlr_utils.hpp"
#include "numerics/line_dlr/line_basis.hpp"

namespace numerics::line_dlr {

struct bosonic_basis_t {
  double theta = 0.0, lam = 0.0, eps = 0.0, gap = 0.0;
  long rank = 0;
  nda::array<double, 1> nu;                    ///< (rank) positive poles, sorted ascending
  nda::array<ComplexType, 1> zeta_nodes;       ///< (min(2 rank, 2 nline)) line nodes (mu-relative)
  nda::array<ComplexType, 1> zeta_dense;       ///< (2 nline) the dense selection grid

  long n_mirror = 0;                           ///< n1 > 0: zeta_nodes is mirror-symmetric in the layout above, else 0
  double node_factor = 0.0;                    ///< nz / rank of the symmetric selection (0: python nodes)

  // S8b finite temperature (file header)
  bool split_fit  = false;                     ///< decoupled odd / even pair fit (Eq. fT_splitfit)
  double cut_odd  = 1e-13, cut_even = 1e-10;   ///< relative SVD cutoffs of the split fit
  long n_bose     = 0;                         ///< extra residue rows (window poles, w_j(-q)^T) after the rank_fit() fitted rows
  std::vector<long> bose_src;                  ///< (n_bose) fitted row j of every extra row
  nda::array<double, 1> tw;                    ///< (rank) W(t) weights (1 + n_j fitted, n_j extra); empty: T = 0
  double beta = 0.0, E_T = 0.0;                ///< of with_bose (diagnostics)
  long rank_fit() const { return rank - n_bose; }

  bosonic_basis_t() = default;

  /// default nz / r of the symmetric node selection (perf 7.1 scan, notes section 5)
  static constexpr double default_node_factor = 1.5;   // perf 7.1 scan: 1.1-1.5 as accurate as 2 r vs Casida; 1.5 for margin
  /// env COQUI_GWLINE_BOS_NODES (a number; <= 0 selects the python nodes), else default_node_factor
  static double env_node_factor() {
    char const *v = std::getenv("COQUI_GWLINE_BOS_NODES");
    return (v != nullptr and *v != '\0') ? std::strtod(v, nullptr) : default_node_factor;
  }

  bosonic_basis_t(double theta_, double lam_, double eps_, double gap_, double tmin = -1.0, double tmax = -1.0,
                  long nline = 1200, long npole = 800, double node_factor_ = env_node_factor())
     : theta(theta_), lam(lam_), eps(eps_), gap(gap_), node_factor(node_factor_) {
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

    if (node_factor > 0.0) {   // mirror-symmetric nodes (see the file header)
      select_mirror_nodes(nline);
      return;
    }
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
  static long mirror_half_(nda::array<ComplexType, 1> const &zeta);

  /// S8b: the basis selected on the data set D (file header; python BosonicLineBasis.from_data)
  static bosonic_basis_t from_data(double theta_, nda::array<ComplexType, 1> const &zdata, double lam_, double eps_ = 1e-12,
                                   long npole = 800, double numin = -1.0, double cut_odd_ = 1e-13, double cut_even_ = 1e-10) {
    utils::check(lam_ > 0.0 and eps_ > 0.0 and npole > 1 and zdata.size() > 0, "bosonic_basis_t::from_data: invalid parameters");
    bosonic_basis_t b;
    b.theta = theta_; b.lam = lam_; b.eps = eps_; b.gap = 0.0;
    b.split_fit = true; b.cut_odd = cut_odd_; b.cut_even = cut_even_;
    if (numin <= 0.0) numin = 1e-4 * lam_;
    const long nd = zdata.size();
    auto nuc      = detail::logspace(numin, lam_, npole);
    nda::matrix<ComplexType, nda::F_layout> Kn(2 * nd, npole);
    for (long j = 0; j < npole; ++j) {
      double nrm = 0.0;
      for (long i = 0; i < nd; ++i) {
        Kn(i, j)      = 1.0 / (zdata(i) - nuc(j));
        Kn(nd + i, j) = -1.0 / (zdata(i) + nuc(j));
        nrm += std::norm(Kn(i, j)) + std::norm(Kn(nd + i, j));
      }
      nrm = std::sqrt(nrm);
      for (long i = 0; i < 2 * nd; ++i) Kn(i, j) /= nrm;
    }
    auto q = detail::pivoted_qr(Kn);
    b.rank = detail::qr_rank(q, eps_);
    std::vector<double> ns(b.rank);
    for (long l = 0; l < b.rank; ++l) ns[l] = nuc(q.piv[l]);
    std::sort(ns.begin(), ns.end());
    b.nu = nda::array<double, 1>(b.rank);
    for (long l = 0; l < b.rank; ++l) b.nu(l) = ns[l];
    b.zeta_nodes = zdata;
    b.zeta_dense = zdata;
    b.n_mirror   = mirror_half_(zdata);
    return b;
  }

  /// S8b: the same fitted poles nu (e.g. injected) on the data set D (no selection)
  static bosonic_basis_t with_poles(double theta_, nda::array<ComplexType, 1> const &zdata, double lam_, nda::array<double, 1> const &nu_,
                                    double eps_ = 1e-12, double cut_odd_ = 1e-13, double cut_even_ = 1e-10) {
    bosonic_basis_t b;
    b.theta = theta_; b.lam = lam_; b.eps = eps_; b.gap = 0.0;
    b.split_fit = true; b.cut_odd = cut_odd_; b.cut_even = cut_even_;
    b.nu = nu_;
    b.rank = nu_.size();
    b.zeta_nodes = zdata;
    b.zeta_dense = zdata;
    b.n_mirror   = mirror_half_(zdata);
    return b;
  }

  /// S8b: the Sigma basis of Eq. fT_W (file header): n_j = 1 / (e^{beta nu_j} - 1) for nu_j <= E_T (0 beyond)
  bosonic_basis_t with_bose(double beta_, double E_T_) const {
    utils::check(n_bose == 0, "bosonic_basis_t::with_bose: already augmented");
    bosonic_basis_t b = *this;
    b.beta = beta_;
    b.E_T  = E_T_;
    const long r = rank;
    std::vector<double> n(r, 0.0);
    for (long j = 0; j < r; ++j) {
      const double x = beta_ * nu(j);
      if (nu(j) <= E_T_ and x <= 700.0) n[j] = 1.0 / std::expm1(x);
      if (n[j] > 0.0) b.bose_src.push_back(j);
    }
    b.n_bose = long(b.bose_src.size());
    b.rank   = r + b.n_bose;
    b.nu     = nda::array<double, 1>(b.rank);
    b.tw     = nda::array<double, 1>(b.rank);
    for (long j = 0; j < r; ++j) {
      b.nu(j) = nu(j);
      b.tw(j) = 1.0 + n[j];
    }
    for (long i = 0; i < b.n_bose; ++i) {
      b.nu(r + i) = nu(b.bose_src[i]);
      b.tw(r + i) = n[b.bose_src[i]];
    }
    return b;
  }

  /// the fitted poles only (rows < rank_fit())
  nda::array<double, 1> nu_fit() const { return nda::array<double, 1>(nu(nda::range(rank_fit()))); }

  /**
   * S8b host form of the decoupled pair fit (Eq. fT_splitfit; python BosonicLineBasis.fit_split): returns {w(q), w(-q)}
   * (rank_fit, N, N) from W(q), W(-q) at zeta (W(-q) = W(q) for a self-inverse q).
   */
  std::pair<nda::array<ComplexType, 3>, nda::array<ComplexType, 3>> fit_split(nda::array<ComplexType, 1> const &zeta,
                                                                              nda::array<ComplexType, 3> const &W,
                                                                              nda::array<ComplexType, 3> const &Wm) const {
    const long nz = W.extent(0), N = W.extent(1), r = rank_fit();
    nda::array<ComplexType, 2> s_(r, N * N), d_(r, N * N);
    for (int sec = 0; sec < 2; ++sec) {
      nda::matrix<ComplexType, nda::F_layout> A(nz, r);
      std::vector<double> cn(r, 0.0);
      for (long j = 0; j < r; ++j) {
        for (long i = 0; i < nz; ++i) {
          const ComplexType km = 1.0 / (zeta(i) - nu(j)), kp = 1.0 / (zeta(i) + nu(j));
          A(i, j) = sec == 0 ? km - kp : km + kp;
          cn[j] += std::norm(A(i, j));
        }
        cn[j] = std::sqrt(cn[j]);
        for (long i = 0; i < nz; ++i) A(i, j) /= cn[j];
      }
      const long dm = std::min(nz, r);
      nda::matrix<ComplexType, nda::F_layout> U(nz, nz), VT(r, r);
      nda::array<double, 1> sv(dm);
      nda::lapack::gesvd(A, sv, U, VT);
      const double cut = sec == 0 ? cut_odd : cut_even;
      long k = 0;
      for (long i = 0; i < dm; ++i)
        if (sv(i) > cut * sv(0)) ++k;
      nda::array<ComplexType, 2> Y(k, N * N);
      Y() = 0.0;
      for (long i = 0; i < k; ++i)
        for (long z = 0; z < nz; ++z) {
          const ComplexType u = std::conj(U(z, i));
          for (long P = 0; P < N; ++P)
            for (long Q = 0; Q < N; ++Q) {
              const ComplexType dz = sec == 0 ? W(z, P, Q) + Wm(z, Q, P) : W(z, P, Q) - Wm(z, Q, P);
              Y(i, P * N + Q) += u * dz;
            }
        }
      auto &X = sec == 0 ? s_ : d_;
      X()     = 0.0;
      for (long j = 0; j < r; ++j)
        for (long i = 0; i < k; ++i) {
          const ComplexType c = std::conj(VT(i, j)) / sv(i) / cn[j];
          for (long c0 = 0; c0 < N * N; ++c0) X(j, c0) += c * Y(i, c0);
        }
    }
    nda::array<ComplexType, 3> w(r, N, N), wm(r, N, N);
    for (long j = 0; j < r; ++j)
      for (long P = 0; P < N; ++P)
        for (long Q = 0; Q < N; ++Q) {
          w(j, P, Q)  = 0.5 * (s_(j, P * N + Q) + d_(j, P * N + Q));
          wm(j, Q, P) = 0.5 * (s_(j, P * N + Q) - d_(j, P * N + Q));
        }
    return {std::move(w), std::move(wm)};
  }

  /**
   * n1 = ceil(node_factor r / 2) ray-1 nodes by the greedy group selection of the file header (zeta_dense(0:nline) is ray 1,
   * zeta_dense(nline + i) = -conj zeta_dense(i) is its mirror).
   */
  void select_mirror_nodes(long nline) {
    const long r = rank, n2r = 2 * r;
    const long n1 = std::min(nline, std::max(1L, long(std::ceil(0.5 * node_factor * double(r) - 1e-9))));
    // candidate rows (4 per ray-1 point), row-major (4 nline, 2r)
    nda::array<ComplexType, 2> C(4 * nline, n2r);
    for (long c = 0; c < nline; ++c) {
      const ComplexType z = zeta_dense(c);
      for (long j = 0; j < r; ++j) {
        const ComplexType km = 1.0 / (z - nu(j)), kp = 1.0 / (z + nu(j));
        C(4 * c + 0, j) = km;             C(4 * c + 0, r + j) = -kp;              // a(z)
        C(4 * c + 1, j) = -kp;            C(4 * c + 1, r + j) = km;               // b(z)
        C(4 * c + 2, j) = -std::conj(kp); C(4 * c + 2, r + j) = std::conj(km);    // conj b(z) = a(-conj z)
        C(4 * c + 3, j) = std::conj(km);  C(4 * c + 3, r + j) = -std::conj(kp);   // conj a(z) = b(-conj z)
      }
    }
    std::vector<char> taken(nline, 0);
    std::vector<long> sel;
    nda::array<ComplexType, 2> R = C;   // residual rows
    double s0 = 0.0;
    auto group_norm = [&](long c) {
      double s = 0.0;
      for (long i = 0; i < 4; ++i)
        for (long j = 0; j < n2r; ++j) s += std::norm(R(4 * c + i, j));
      return s;
    };
    for (long c = 0; c < nline; ++c) s0 = std::max(s0, group_norm(c));
    long nbasis = 0;
    while (long(sel.size()) < n1) {
      long best = -1;
      double bs = -1.0;
      for (long c = 0; c < nline; ++c) {
        if (taken[c]) continue;
        const double s = group_norm(c);
        if (s > bs) { bs = s; best = c; }
      }
      if (best < 0) break;
      if (bs <= 1e-26 * s0 or nbasis >= n2r) {   // rank exhausted: fresh pass on the remaining candidates
        R      = C;
        nbasis = 0;
        for (long c : sel)
          for (long i = 0; i < 4; ++i) R(4 * c + i, nda::range::all) = ComplexType(0.0);
        continue;
      }
      taken[best] = 1;
      sel.push_back(best);
      // orthonormal directions of the group's residual rows (MGS twice), projected out of every candidate row
      std::vector<nda::array<ComplexType, 1>> qv;
      for (long i = 0; i < 4; ++i) {
        nda::array<ComplexType, 1> v = R(4 * best + i, nda::range::all);
        for (int pass = 0; pass < 2; ++pass)
          for (auto const &u : qv) {
            ComplexType d = 0.0;
            for (long j = 0; j < n2r; ++j) d += std::conj(u(j)) * v(j);
            for (long j = 0; j < n2r; ++j) v(j) -= d * u(j);
          }
        double nv = 0.0;
        for (long j = 0; j < n2r; ++j) nv += std::norm(v(j));
        if (nv > 1e-24 * s0) {
          v /= std::sqrt(nv);
          qv.push_back(v);
        }
      }
      if (qv.empty()) continue;
      nda::array<ComplexType, 2> Qm(long(qv.size()), n2r), D(4 * nline, long(qv.size()));
      for (long i = 0; i < long(qv.size()); ++i) Qm(i, nda::range::all) = qv[i];
      for (int pass = 0; pass < 2; ++pass) {   // R -= (R Q^dagger) Q
        nda::blas::gemm(ComplexType(1.0), R, nda::dagger(Qm), ComplexType(0.0), D);
        nda::blas::gemm(ComplexType(-1.0), D, Qm, ComplexType(1.0), R);
      }
      for (long i = 0; i < 4; ++i) R(4 * best + i, nda::range::all) = ComplexType(0.0);
      nbasis += long(qv.size());
    }
    std::sort(sel.begin(), sel.end());   // ray-1 dense points are ascending in |zeta|
    n_mirror   = long(sel.size());
    zeta_nodes = nda::array<ComplexType, 1>(2 * n_mirror);
    for (long l = 0; l < n_mirror; ++l) {
      zeta_nodes(l)            = zeta_dense(sel[l]);
      zeta_nodes(n_mirror + l) = -std::conj(zeta_dense(sel[l]));
    }
  }

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
   * Self-inverse q only (W(-q) = W(q)). Residues w[r, N, N] of the positive poles from samples W[nz, N, N] at mu-relative
   * zeta (Eq. bfit):
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

  /**
   * Paired fit (q != -q): residues w(q) [r, N, N] from W(q) and Wm = W(-q) at the same nodes (Eq. bfit with the partner):
   *   [[Km, -Kp], [-Kp, Km]] [w(q)_PQ; w(-q)_QP] = [W(q)_PQ(zeta); W(-q)_QP(zeta)]   for all ordered (P, Q),
   * one gelss with all N^2 right-hand sides. With Wm = W this is the self-inverse fit above (up to the non-unique
   * near-threshold directions of the residues).
   */
  nda::array<ComplexType, 3> fit(nda::array<ComplexType, 1> const &zeta, nda::array<ComplexType, 3> const &W,
                                 nda::array<ComplexType, 3> const &Wm, double rcond = -1.0) const {
    const long nz = W.extent(0), N = W.extent(1);
    utils::check(nz == zeta.size() and W.extent(2) == N, "bosonic_basis_t::fit: W shape mismatch");
    utils::check(Wm.extent(0) == nz and Wm.extent(1) == N and Wm.extent(2) == N, "bosonic_basis_t::fit: W(-q) shape mismatch");
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
    nda::array<ComplexType, 2> data(2 * nz, N * N);
    for (long i = 0; i < nz; ++i)
      for (long P = 0; P < N; ++P)
        for (long Q = 0; Q < N; ++Q) {
          data(i, P * N + Q)      = W(i, P, Q);
          data(nz + i, P * N + Q) = Wm(i, Q, P);
        }
    auto sol = detail::lstsq(A, data, rcond);   // (2r, N^2)
    nda::array<ComplexType, 3> w(r, N, N);
    for (long j = 0; j < r; ++j)
      for (long P = 0; P < N; ++P)
        for (long Q = 0; Q < N; ++Q) w(j, P, Q) = sol(j, P * N + Q);
    return w;
  }

  /// Paired evaluation W(q, zeta)[nz, N, N] = Km w(q) - Kp w(-q)^T (sector particle: the first term, hole: the second).
  nda::array<ComplexType, 3> eval(nda::array<ComplexType, 3> const &w, nda::array<ComplexType, 3> const &wm,
                                  nda::array<ComplexType, 1> const &zeta, sector_t sector = sector_t::both) const {
    utils::check(wm.extent(0) == w.extent(0) and wm.extent(1) == w.extent(1) and wm.extent(2) == w.extent(2),
                 "bosonic_basis_t::eval: w(-q) shape mismatch");
    nda::array<ComplexType, 3> out(zeta.size(), w.extent(1), w.extent(2));
    out() = ComplexType(0.0);
    if (sector != sector_t::hole) out += eval(w, zeta, sector_t::particle);
    if (sector != sector_t::particle) out += eval(wm, zeta, sector_t::hole);
    return out;
  }

  /// W(zeta)[nz, N, N] for a self-inverse q (w(-q) = w(q)); sector particle: Km w only; hole: -Kp w^T only (transpose on
  /// (P, Q)); both: the sum. For q != -q use eval(w(q), w(-q), zeta).
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

  /// [nt, r]: e^{-i nu_j t} (particle, W^>(q,t) = sum_j w_j(q) e^{-i nu_j t}) or -e^{+i nu_j t} (hole,
  /// W^<(q,t) = sum_j w_j(-q)^T (.): the residues of the partner -q).
  nda::array<ComplexType, 2> time_exponentials(nda::array<ComplexType, 1> const &t, sector_t sector) const {
    utils::check(sector != sector_t::both, "bosonic_basis_t::time_exponentials: sector must be particle or hole");
    const long nt = t.size();
    nda::array<ComplexType, 2> E(nt, rank);
    const ComplexType I(0.0, 1.0);
    if (tw.size() == rank) {   // S8b Eq. fT_W (file header)
      const long rf = rank_fit();
      for (long i = 0; i < nt; ++i)
        for (long j = 0; j < rank; ++j) {
          const double sg = (j < rf) ? 1.0 : -1.0;   // extra rows: the opposite exponential
          E(i, j) = (sector == sector_t::particle) ? tw(j) * std::exp(-I * sg * nu(j) * t(i)) : -tw(j) * std::exp(I * sg * nu(j) * t(i));
        }
      return E;
    }
    for (long i = 0; i < nt; ++i)
      for (long j = 0; j < rank; ++j)
        E(i, j) = (sector == sector_t::particle) ? std::exp(-I * nu(j) * t(i)) : -std::exp(I * nu(j) * t(i));
    return E;
  }
};

/**
 * n1 if zeta = [z_1 .. z_n1, -conj z_1 .. -conj z_n1] with every z_i in the upper right quadrant (the mirror-symmetric
 * layout of bosonic_basis_t), else 0. Any node set (e.g. read from a bases file) can be tested.
 */
inline long mirror_half(nda::array<ComplexType, 1> const &zeta) {
  const long nz = zeta.size();
  if (nz < 2 or nz % 2 != 0) return 0;
  const long n1 = nz / 2;
  for (long i = 0; i < n1; ++i) {
    // S8b: the CLOSED quadrant (Matsubara points i nu_n and nu_0 = 0 of the finite-T data set are their own mirrors)
    if (not(zeta(i).real() >= 0.0 and zeta(i).imag() >= 0.0)) return 0;
    if (std::abs(zeta(n1 + i) + std::conj(zeta(i))) > 1e-14 * std::abs(zeta(i))) return 0;
  }
  return n1;
}

inline long bosonic_basis_t::mirror_half_(nda::array<ComplexType, 1> const &zeta) { return mirror_half(zeta); }

} // namespace numerics::line_dlr

#endif
