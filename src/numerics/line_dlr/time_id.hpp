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

#ifndef COQUI_NUMERICS_LINE_DLR_TIME_ID_HPP
#define COQUI_NUMERICS_LINE_DLR_TIME_ID_HPP

/**
 * Time-node interpolative decomposition (notes section 4.3, Eqs. gram and F): the compressed replacement of the
 * generic Gauss-Legendre ray quadrature of time_ray.hpp (~1000 nodes) by r_t ~ 100-200 nodes.
 *
 * A sector's time functions are sums of exponentials X(t) = sum_p c_p e^{-i E_p t} with SUMMED energies
 * |E_p| in [Emin, Emax] (E > 0 on the particle ray t = s e^{-i theta_t}, E < 0 on the hole ray t = s e^{+i theta_t}).
 * The construction:
 *  1. fine energy grid e_k = log-spaced on [Emin, Emax] (nE_per_efold points per e-fold), E_k = +-e_k;
 *     candidate times s_m = {0} u logspace(smin_fac/Emax, smax_fac ln(1/eps)/(Emin sin theta_t)) with trapezoid
 *     weights in log s (ws_m = s_m dlog s; ws_0 = smin);
 *  2. selection matrix M(k, m) = sqrt(2 a(e_k)) e^{-i E_k t(s_m)} sqrt(ws_m), a(e) = e sin(theta_t):
 *     rows normalized to unit L2(ds) norm (the unit-diagonal Gram of Eq. gram), columns carrying the L2(ds) measure,
 *     so M M^dagger is the discretized normalized Gram and its singular values are sqrt(Gram eigenvalues);
 *     column-pivoted QR (geqp3) of M -> eps-rank = #{|R_ll| > eps |R_00|} (= Gram eigenvalues > eps^2 lambda_max),
 *     and the first r_t = ceil(oversample * rank) pivots, sorted by s, are the time nodes;
 *  3. transform weights (Eq. F) by least squares over the fine E grid, rows weighted by w_k = sqrt(e_k):
 *        min || diag(w) [ A F(zeta,.)^T - b(zeta) ] ||,  A(k, j) = e^{-i E_k t_j},  b_k(zeta) = 1/(zeta - E_k).
 *     The weight is zeta-independent, so ONE SVD A_w = U S V^dagger serves every target. It is applied FACTORED,
 *        F(zeta, .) = [ (w b)^T conj(U) ] diag(1/S) conj(V^dagger),
 *     never as an explicit pseudo-inverse: forming V S^{-1} U^dagger explicitly costs cond(A) in the residual (measured
 *     1e-6 instead of 1e-12 at cond ~ 1e14), the factored product is backward stable as long as |F| stays O(10-100).
 *     Singular values below rcond S_0 (default 1e-14) are dropped.
 *
 * Lessons from the prototype (scratch study, numbers in notes/line_gw_progress.md, S7a):
 *  - the L2(ds) column weights sqrt(ws) in the selection are essential: unweighted candidate columns on a log grid
 *    under-select the long times and the transform stalls at 1e3-1e5 eps;
 *  - oversampling 1.25 brings the transform error from ~5 eps to ~0.05-0.1 eps and lowers max|F|;
 *  - the E grid needs >= 80 points per e-fold for the LS at eps = 1e-10 (the rank is converged already at 40);
 *  - measured scaling (theta_t = 10 deg, 39 cases): r_t = 8.7 + 1.07 ln(Emax/Emin) ln(1/eps) within 3%;
 *  - the LS is unconstrained outside [Emin, Emax] (E = 0.8 Emin / 1.2 Emax: 1e-3..3e-1 error): pass a margin,
 *    opts.pad = 1.25 restores <= 3 eps there for ~8% more nodes.
 * Recommended production setting: eps = 1e-10, oversample = 1.0, pad = 1.25 (r_t ~ 145 for Pi [0.04, 6] and
 * Sigma [0.06, 10], per-pole error <= 4e-10, mixtures ~1e-11; GL ray: 976-992 nodes).
 *
 * Interface: same consumer members as time_ray_t (t, s, size(), sector, theta_t, phase, transform_matrix(zeta)).
 * `time_nodes_t` below is a type-erased view constructible from either, so a kernel signature
 * `time_ray_t const &` -> `time_nodes_t const &` accepts both without further changes.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <memory>
#include <numbers>
#include <type_traits>
#include <vector>

#include "nda/lapack/gesvd.hpp"
#include "numerics/line_dlr/line_dlr_utils.hpp"
#include "numerics/line_dlr/time_ray.hpp"

namespace numerics::line_dlr {

/// Construction knobs of time_id_t (defaults: the converged values of the S7a study).
struct time_id_opts_t {
  double oversample   = 1.0;    ///< r_t = ceil(oversample * eps-rank), capped by the candidate count
  double nE_per_efold = 120.0;  ///< fine energy grid density (LS rows); rank converged at 40, LS at >= 80
  double ns_per_efold = 40.0;   ///< candidate s grid density
  double smin_fac     = 1e-3;   ///< smallest nonzero candidate s = smin_fac / Emax (plus s = 0)
  double smax_fac     = 1.5;    ///< largest candidate s = smax_fac ln(1/eps) / (Emin sin theta_t)
  double rcond        = 1e-14;  ///< LS: singular values below rcond * S_0 are dropped
  long rank_force     = -1;     ///< > 0: take exactly this many pivots (diagnostic / accuracy-vs-nodes studies)
  double pad          = 1.0;    ///< safety margin: the ID is built for [Emin/pad, Emax*pad] (the LS is unconstrained
                                ///< outside its range: E = 0.8 Emin or 1.2 Emax cost 1e-3..1e-1 at pad = 1)
};

struct time_id_t {
  double theta_t = 0.0;
  sector_t sector = sector_t::particle;
  ComplexType phase = ComplexType(1.0, 0.0);   ///< dt/ds: e^{-i theta_t} (particle) or e^{+i theta_t} (hole)
  double Emin = 0.0, Emax = 0.0, eps = 0.0;    ///< nominal design range (|E|) and tolerance (before opts.pad)
  time_id_opts_t opts;
  long rank = 0;                               ///< eps-rank of the normalized kernel (|R_ll| > eps |R_00|)
  long n_cand = 0;                             ///< number of candidate s
  nda::array<double, 1> s;                     ///< (r_t) selected real ray coordinates, ascending
  nda::array<ComplexType, 1> t;                ///< (r_t) complex times s * phase
  nda::array<double, 1> rdiag;                 ///< |R_ll| / |R_00| of the selection QR (first min(nE, n_cand))

  // LS factorization (fine grid and the factored pseudo-inverse)
  nda::array<double, 1> E;                     ///< (nE) signed fine energies
  nda::array<double, 1> w;                     ///< (nE) row weights sqrt(|E|)
  nda::array<ComplexType, 2> Uc;               ///< (nE, k) conj(U(:, :k))
  nda::array<ComplexType, 2> Vs;               ///< (k, r_t) diag(1/S) conj(V^dagger(:k, :))
  nda::array<double, 1> sv;                    ///< (min(nE, r_t)) singular values of diag(w) A
  long ls_rank = 0;                            ///< k: number of kept singular values

  time_id_t() = default;

  time_id_t(double theta_t_, sector_t sector_, double Emin_, double Emax_, double eps_, time_id_opts_t const &o = {})
     : theta_t(theta_t_), sector(sector_), Emin(Emin_), Emax(Emax_), eps(eps_), opts(o) {
    utils::check(sector != sector_t::both, "time_id_t: sector must be particle or hole");
    utils::check(Emin > 0.0 and Emax > Emin, "time_id_t: need 0 < Emin < Emax (got {} {})", Emin, Emax);
    utils::check(eps > 0.0 and eps < 1.0, "time_id_t: eps = {} out of (0, 1)", eps);
    utils::check(theta_t > 0.0 and theta_t < std::numbers::pi / 2, "time_id_t: theta_t = {} out of (0, pi/2)", theta_t);
    utils::check(opts.pad >= 1.0, "time_id_t: pad = {} must be >= 1", opts.pad);
    const double sgn = (sector == sector_t::particle) ? 1.0 : -1.0;
    const double st  = std::sin(theta_t);
    const double elo = Emin / opts.pad, ehi = Emax * opts.pad;   // design range actually covered
    phase            = std::exp(ComplexType(0.0, -sgn * theta_t));

    // 1. fine energy grid and candidate s grid
    const long nE = long(std::ceil(opts.nE_per_efold * std::log(ehi / elo))) + 20;
    auto e        = detail::logspace(elo, ehi, nE);
    E             = nda::array<double, 1>(nE);
    w             = nda::array<double, 1>(nE);
    for (long k = 0; k < nE; ++k) {
      E(k) = sgn * e(k);
      w(k) = std::sqrt(e(k));
    }
    const double smin = opts.smin_fac / ehi, smax = opts.smax_fac * std::log(1.0 / eps) / (elo * st);
    utils::check(smax > smin, "time_id_t: empty candidate s range");
    const long nsl = std::max(long(opts.ns_per_efold * std::log(smax / smin)), 2L);
    auto sl        = detail::logspace(smin, smax, nsl);
    const double h = std::log(smax / smin) / double(nsl - 1);
    n_cand         = nsl + 1;
    nda::array<double, 1> sc(n_cand), wsc(n_cand);
    sc(0)  = 0.0;
    wsc(0) = smin;
    for (long m = 0; m < nsl; ++m) {
      sc(m + 1)  = sl(m);
      wsc(m + 1) = sl(m) * h;
    }

    // 2. selection: column-pivoted QR of the normalized kernel
    nda::matrix<ComplexType, nda::F_layout> M(nE, n_cand);
    for (long m = 0; m < n_cand; ++m) {
      const ComplexType mt = ComplexType(0.0, -1.0) * sc(m) * phase;   // -i t_m
      const double cw      = std::sqrt(wsc(m));
      for (long k = 0; k < nE; ++k) M(k, m) = std::sqrt(2.0 * e(k) * st) * cw * std::exp(E(k) * mt);
    }
    auto qr = detail::pivoted_qr(M);
    rank    = detail::qr_rank(qr, eps);
    rdiag   = nda::array<double, 1>(long(qr.rdiag.size()));
    for (long l = 0; l < rdiag.size(); ++l) rdiag(l) = qr.rdiag[l] / qr.rdiag[0];
    long r = (opts.rank_force > 0) ? opts.rank_force : long(std::ceil(opts.oversample * double(rank) - 1e-9));
    r      = std::min(r, std::min(n_cand, nE));
    std::vector<long> J(qr.piv.begin(), qr.piv.begin() + r);
    std::sort(J.begin(), J.end());
    s = nda::array<double, 1>(r);
    t = nda::array<ComplexType, 1>(r);
    for (long j = 0; j < r; ++j) {
      s(j) = sc(J[j]);
      t(j) = s(j) * phase;
    }

    // 3. LS factorization: diag(w) A = U S V^dagger
    nda::matrix<ComplexType, nda::F_layout> A(nE, r), U(nE, nE), VT(r, r);
    for (long j = 0; j < r; ++j) {
      const ComplexType mt = ComplexType(0.0, -1.0) * t(j);
      for (long k = 0; k < nE; ++k) A(k, j) = w(k) * std::exp(E(k) * mt);
    }
    const long kmax = std::min(nE, r);
    sv              = nda::array<double, 1>(kmax);
    nda::lapack::gesvd(A, sv, U, VT);
    ls_rank = 0;
    for (long l = 0; l < kmax; ++l)
      if (sv(l) > opts.rcond * sv(0)) ++ls_rank;
    Uc = nda::array<ComplexType, 2>(nE, ls_rank);
    Vs = nda::array<ComplexType, 2>(ls_rank, r);
    for (long k = 0; k < nE; ++k)
      for (long l = 0; l < ls_rank; ++l) Uc(k, l) = std::conj(U(k, l));
    for (long l = 0; l < ls_rank; ++l)
      for (long j = 0; j < r; ++j) Vs(l, j) = std::conj(VT(l, j)) / sv(l);
  }

  long size() const { return s.size(); }
  long nE() const { return E.size(); }

  /// e^{-i E_p t_j} -> [r_t, np]
  nda::array<ComplexType, 2> exponentials(nda::array<double, 1> const &Ep) const {
    const long nt = size(), np = Ep.size();
    nda::array<ComplexType, 2> X(nt, np);
    for (long j = 0; j < nt; ++j)
      for (long p = 0; p < np; ++p) X(j, p) = std::exp(ComplexType(0.0, -Ep(p)) * t(j));
    return X;
  }

  /// F[nz, r_t] with X(zeta) = F X(t) (Eq. F); optionally the max over targets of the relative weighted LS residual
  /// || diag(w) (A F(zeta,.)^T - b(zeta)) || / || diag(w) b(zeta) ||.
  nda::array<ComplexType, 2> transform_matrix(nda::array<ComplexType, 1> const &zeta, double *ls_residual = nullptr) const {
    const long nz = zeta.size(), ne = nE();
    nda::array<ComplexType, 2> B(nz, ne);
    for (long i = 0; i < nz; ++i)
      for (long k = 0; k < ne; ++k) B(i, k) = w(k) / (zeta(i) - E(k));
    auto F = detail::matmul(detail::matmul(B, Uc), Vs);
    if (ls_residual) {
      // R = F A_w^T - B with A_w(k, j) = w_k e^{-i E_k t_j}
      nda::array<ComplexType, 2> AwT(size(), ne);
      for (long j = 0; j < size(); ++j)
        for (long k = 0; k < ne; ++k) AwT(j, k) = w(k) * std::exp(ComplexType(0.0, -E(k)) * t(j));
      auto R       = detail::matmul(F, AwT);
      double rmax  = 0.0;
      for (long i = 0; i < nz; ++i) {
        double n2 = 0.0, d2 = 0.0;
        for (long k = 0; k < ne; ++k) {
          n2 += std::norm(R(i, k) - B(i, k));
          d2 += std::norm(B(i, k));
        }
        rmax = std::max(rmax, std::sqrt(n2 / d2));
      }
      *ls_residual = rmax;
    }
    return F;
  }

  /// condition number of the kept part of the LS matrix
  double ls_cond() const { return sv(0) / sv(ls_rank - 1); }
};

/**
 * Normalized-Gram eps-rank of Eq. gram on the log grid of n points in [Emin, Emax] (diagnostic cross-check of the
 * QR rank; O(n^3)): G(E, E') = 1 / (a(E) + a(E') - i (E - E') cos theta_t), a = |E| sin theta_t, unit diagonal,
 * rank = #{lambda > eps^2 lambda_max}.
 */
inline long gram_rank(double theta_t, double Emin, double Emax, double eps, long n) {
  auto e         = detail::logspace(Emin, Emax, n);
  const double st = std::sin(theta_t), ct = std::cos(theta_t);
  nda::matrix<ComplexType, nda::F_layout> G(n, n);
  for (long i = 0; i < n; ++i)
    for (long j = 0; j < n; ++j)
      G(i, j) = std::sqrt(4.0 * e(i) * e(j)) * st / ComplexType((e(i) + e(j)) * st, -(e(i) - e(j)) * ct);
  // eigenvalues of the Hermitian PSD G = singular values; use gesvd (no eigen wrapper needed here)
  nda::matrix<ComplexType, nda::F_layout> U(n, n), VT(n, n);
  nda::array<double, 1> lam(n);
  nda::lapack::gesvd(G, lam, U, VT);
  long r = 0;
  for (long i = 0; i < n; ++i)
    if (lam(i) > eps * eps * lam(0)) ++r;
  return r;
}

/**
 * Type-erased time-node set: the consumer view shared by time_ray_t (GL quadrature) and time_id_t (ID nodes).
 * Implicitly constructible from both (a copy of the nodes plus a shared handle on the source for the transform), so
 * a kernel taking `time_nodes_t const &` accepts either.
 */
struct time_nodes_t {
  double theta_t = 0.0;
  sector_t sector = sector_t::particle;
  ComplexType phase = ComplexType(1.0, 0.0);
  nda::array<double, 1> s;
  nda::array<ComplexType, 1> t;
  std::function<nda::array<ComplexType, 2>(nda::array<ComplexType, 1> const &)> transform_fn;

  template <typename Nodes>
    requires(not std::is_same_v<std::decay_t<Nodes>, time_nodes_t>)
  time_nodes_t(Nodes const &n) : theta_t(n.theta_t), sector(n.sector), phase(n.phase), s(n.s), t(n.t) {   // NOLINT
    auto keep    = std::make_shared<Nodes const>(n);
    transform_fn = [keep](nda::array<ComplexType, 1> const &z) { return keep->transform_matrix(z); };
  }
  long size() const { return s.size(); }
  nda::array<ComplexType, 2> transform_matrix(nda::array<ComplexType, 1> const &zeta) const { return transform_fn(zeta); }
};

} // namespace numerics::line_dlr

#endif
