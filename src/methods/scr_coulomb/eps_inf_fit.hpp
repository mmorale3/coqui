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

#ifndef COQUI_EPS_INF_FIT_HPP
#define COQUI_EPS_INF_FIT_HPP

/**
 * P25 / G32 (notes/vertex_perf_plan.md): epsilon_inf from the SMALL-q FIT of the static
 * macroscopic dielectric function, reported next to the stored q -> 0 head.
 *
 * The head of W(q) carries the bare 1/q^2 divergence, and every div_treatment (gygi,
 * gygi_smallest_q, cvv, ...) is a recipe for the q = 0 value of eps^-1_{00}. The physical
 * check of that recipe is that the static dielectric function on the smallest |q| points
 * of the mesh extrapolates smoothly to q = 0:
 *
 *     eps(q, i nu = 0) = eps_inf + A |q|^2 (+ B |q|^4),                             (1)
 *
 * i.e. a polynomial in |q|^2 (eps is even in q). Given |q| and eps(q) at the mesh transfers,
 * fit_eps_inf takes the n_fit smallest NONZERO |q| (q = 0 excluded: that slot is the stored
 * head itself), fits (1) by least squares -- degree 1 in |q|^2 for 2-3 points, degree 2 for
 * >= 4 -- and returns eps_inf = the constant term, the coefficients, the RMS misfit and the
 * |q| used.
 *
 * Pure functions, no MPI, no nda: the caller hands in std::vector<double>s. The least-squares
 * solve is a Householder QR on the |q|^2-Vandermonde matrix with the variable scaled to
 * x = |q|^2 / max|q|^2 (columns O(1)), so an exact polynomial is recovered to round-off.
 *
 * Approximations the CALLER makes (documented at the call site in scr_coulomb_t::update_w):
 * the fit is in |q| only -- for a non-cubic cell eps(q -> 0) depends on the direction of q
 * and the |q|-only fit averages over whichever directions the smallest IBZ transfers sample.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace methods {
namespace solvers {
namespace eps_fit {

  struct eps_inf_fit_t {
    bool ok = false;               // false: fewer than 2 usable (nonzero, distinct) |q|
    int degree = 0;                // polynomial degree in |q|^2 actually used
    double eps_inf = 0.0;          // c_0 = eps(q -> 0)
    std::vector<double> coeffs;    // c_0 .. c_degree:  eps(q) = sum_k c_k |q|^{2k}; c_1 = A
    double residual = 0.0;         // RMS of eps_fit(q_i) - eps(q_i) over the points used
    std::vector<double> q_used;    // the |q| used, ascending
    std::vector<double> eps_used;  // eps(q) at q_used
  };

  /**
   * Indices of the n smallest NONZERO |q| (|q| > q_zero_tol), ascending in |q|
   * (ties keep the input order). n <= 0 or n > #nonzero returns all nonzero points.
   */
  inline std::vector<std::size_t> smallest_nonzero_q(std::vector<double> const &q_abs, long n,
                                                     double q_zero_tol = 1e-10) {
    std::vector<std::size_t> idx;
    idx.reserve(q_abs.size());
    for (std::size_t i = 0; i < q_abs.size(); ++i)
      if (q_abs[i] > q_zero_tol) idx.push_back(i);
    std::stable_sort(idx.begin(), idx.end(),
                     [&](std::size_t a, std::size_t b) { return q_abs[a] < q_abs[b]; });
    if (n > 0 and std::size_t(n) < idx.size()) idx.resize(std::size_t(n));
    return idx;
  }

  /** The fit degree in |q|^2 for a given number of points: 1 for 2-3 points, 2 for >= 4. */
  inline int fit_degree(std::size_t npts) { return npts >= 4 ? 2 : 1; }

  /**
   * Least-squares polynomial fit y(x) = sum_{k=0}^{degree} c_k x^k (Householder QR on the
   * Vandermonde matrix; x is scaled internally to max|x| = 1 for conditioning and the
   * coefficients are returned in the ORIGINAL variable). Requires x.size() == y.size() >=
   * degree + 1. Returns the coefficients; *rms (optional) receives the RMS misfit.
   */
  inline std::vector<double> polyfit_ls(std::vector<double> const &x, std::vector<double> const &y,
                                        int degree, double *rms = nullptr) {
    const std::size_t m = x.size();
    const std::size_t n = std::size_t(degree) + 1;
    std::vector<double> c(n, 0.0);
    if (degree < 0 or m != y.size() or m < n) {
      if (rms != nullptr) *rms = 0.0;
      return c;
    }
    double xmax = 0.0;
    for (double v : x) xmax = std::max(xmax, std::abs(v));
    if (xmax <= 0.0) xmax = 1.0;
    // A (m x n) column-major, b (m)
    std::vector<double> A(m * n, 0.0), b(y);
    for (std::size_t i = 0; i < m; ++i) {
      double p = 1.0;
      const double xs = x[i] / xmax;
      for (std::size_t k = 0; k < n; ++k) { A[k * m + i] = p; p *= xs; }
    }
    // Householder QR: A = Q R, b <- Q^T b
    for (std::size_t k = 0; k < n; ++k) {
      double norm2 = 0.0;
      for (std::size_t i = k; i < m; ++i) norm2 += A[k * m + i] * A[k * m + i];
      const double norm = std::sqrt(norm2);
      if (norm == 0.0) continue;                       // rank-deficient column: leave it
      const double akk = A[k * m + k];
      const double alpha = (akk > 0.0) ? -norm : norm;
      std::vector<double> v(m - k, 0.0);
      for (std::size_t i = k; i < m; ++i) v[i - k] = A[k * m + i];
      v[0] -= alpha;
      double vnorm2 = 0.0;
      for (double t : v) vnorm2 += t * t;
      if (vnorm2 == 0.0) continue;
      // apply H = I - 2 v v^T / (v^T v) to the columns k..n-1 of A and to b
      for (std::size_t j = k; j < n; ++j) {
        double dot = 0.0;
        for (std::size_t i = k; i < m; ++i) dot += v[i - k] * A[j * m + i];
        const double s = 2.0 * dot / vnorm2;
        for (std::size_t i = k; i < m; ++i) A[j * m + i] -= s * v[i - k];
      }
      {
        double dot = 0.0;
        for (std::size_t i = k; i < m; ++i) dot += v[i - k] * b[i];
        const double s = 2.0 * dot / vnorm2;
        for (std::size_t i = k; i < m; ++i) b[i] -= s * v[i - k];
      }
    }
    // back substitution R c = (Q^T b)[0..n)
    std::vector<double> cs(n, 0.0);
    for (std::size_t kk = n; kk-- > 0;) {
      double s = b[kk];
      for (std::size_t j = kk + 1; j < n; ++j) s -= A[j * m + kk] * cs[j];
      const double rkk = A[kk * m + kk];
      cs[kk] = (rkk != 0.0) ? s / rkk : 0.0;
    }
    // unscale: y = sum_k cs_k (x/xmax)^k  ->  c_k = cs_k / xmax^k
    double scale = 1.0;
    for (std::size_t k = 0; k < n; ++k) { c[k] = cs[k] / scale; scale *= xmax; }
    if (rms != nullptr) {
      double ss = 0.0;
      for (std::size_t i = 0; i < m; ++i) {
        double p = 0.0, xp = 1.0;
        for (std::size_t k = 0; k < n; ++k) { p += c[k] * xp; xp *= x[i]; }
        ss += (p - y[i]) * (p - y[i]);
      }
      *rms = std::sqrt(ss / double(m));
    }
    return c;
  }

  /** Evaluate the fitted polynomial eps(q) = sum_k c_k |q|^{2k} at |q|. */
  inline double eval_fit(std::vector<double> const &coeffs, double q_abs) {
    const double x = q_abs * q_abs;
    double p = 0.0, xp = 1.0;
    for (double ck : coeffs) { p += ck * xp; xp *= x; }
    return p;
  }

  /**
   * The small-q fit of eps_inf. q_abs and eps_q are indexed alike (one entry per transfer;
   * q = 0 entries are excluded by |q| <= q_zero_tol). n_fit = the number of smallest nonzero
   * |q| to use (default 3; clamped to >= 2 and to the number available). The degree in
   * |q|^2 follows fit_degree(#points), reduced further if the points carry fewer distinct
   * |q|^2 values than degree + 1 (ties in |q| between IBZ transfers). ok = false when fewer
   * than 2 nonzero |q| (or fewer than 2 distinct |q|) exist -- then nothing is fitted.
   */
  inline eps_inf_fit_t fit_eps_inf(std::vector<double> const &q_abs, std::vector<double> const &eps_q,
                                   long n_fit = 3, double q_zero_tol = 1e-10) {
    eps_inf_fit_t r;
    if (q_abs.size() != eps_q.size()) return r;
    if (n_fit < 2) n_fit = 2;
    auto idx = smallest_nonzero_q(q_abs, n_fit, q_zero_tol);
    if (idx.size() < 2) return r;
    r.q_used.reserve(idx.size());
    r.eps_used.reserve(idx.size());
    std::vector<double> x;
    x.reserve(idx.size());
    for (auto i : idx) {
      r.q_used.push_back(q_abs[i]);
      r.eps_used.push_back(eps_q[i]);
      x.push_back(q_abs[i] * q_abs[i]);
    }
    // distinct |q|^2 values (relative tolerance) bound the degree
    std::size_t ndistinct = 1;
    for (std::size_t i = 1; i < x.size(); ++i)
      if (std::abs(x[i] - x[i - 1]) > 1e-12 * std::max(std::abs(x[i]), std::abs(x[i - 1]))) ++ndistinct;
    if (ndistinct < 2) return r;
    r.degree = std::min(fit_degree(x.size()), int(ndistinct) - 1);
    r.coeffs = polyfit_ls(x, r.eps_used, r.degree, &r.residual);
    r.eps_inf = r.coeffs.empty() ? 0.0 : r.coeffs[0];
    r.ok = true;
    return r;
  }

}  // eps_fit
}  // solvers
}  // methods

#endif  // COQUI_EPS_INF_FIT_HPP
