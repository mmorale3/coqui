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

#ifndef COQUI_TILTED_CONTOUR_HPP
#define COQUI_TILTED_CONTOUR_HPP

/**
 * ===========================================================================
 * THE TILTED CONTOUR (increment TC-1 of notes/tc_coqui_impl_spec.md)
 * ===========================================================================
 *
 * OFFLINE geometry only: no physics, no MPI, no allocation of anything that
 * scales with the basis. Inputs are six numbers (Delta_min, Delta_max,
 * W_target, delta, eps, rho); outputs are the contour nodes {s_j}, the tilt
 * theta, and a least-squares transform F that maps CONTOUR SAMPLES to values
 * of the same function at a caller-supplied target list {z}:
 *
 *      P(z)  ~  sum_j F[z,j] P(s_j),        t_j = s_j e^{-i theta}.
 *
 * REFERENCE IMPLEMENTATION: tc_validation/contour.py. Every formula below is a
 * line-by-line port of that file, whose module docstring carries the mechanical
 * derivation and whose [verified: ...] tags name the numerical pin that fixes
 * it. The pins themselves are ported into
 * src/numerics/tests/test_tilted_contour.cpp (gate TC-1-b) and the campaign's
 * exported reference numbers are pinned there too (gate TC-1-a).
 *
 * ---------------------------------------------------------------------------
 * THE DERIVATION, in the reference implementation's own words
 * ---------------------------------------------------------------------------
 * Contour   t(s) = s e^{-i th},  s in [0, S].   Target z = w + i d, d > 0.
 * Pole      R(z) = 1/(z - D) for a transition energy D > 0.
 *
 *   1/(z-D) = -i e^{-i th} Int_0^inf ds K(s, D; z),
 *   K(s, D; z) = exp[i (z - D) s e^{-i th}]
 *              = exp[-a(D) s] exp[i((w-D) cos th + d sin th) s],
 *   a(D) = (D - w) sin th + d cos th.                                   (eq 2)
 * [verified: contour.py module docstring; pin1 below]
 *
 * Convergence for every transition D >= Dmin and every target w in
 * [Dmin, Dmin+W] needs a > 0 at the worst case (D = Dmin, w = Dmin+W):
 *   tan th < d / W,   rho := tan th * W / d in [0,1),                   (eq 3)
 *   gamma  = d cos th (1 - rho) = a_worst,   S = ln(1/eps_tr)/gamma.    (eq 4)
 * [verified: pin_eq34]
 *
 * Gram (x = D - Dmin, worst-case target so that D - w = x - W):
 *   G(x,x') = 1 / [ a(x) + a(x') - i (x - x') cos th ],                 (eq 5)
 *   a(x)    = (x - W) sin th + d cos th.
 * [verified: pin2_gram_closed_form]
 *
 * Endpoint limits:  th = 0 -> G = 1/[2d - i(x-x')];
 *                   th = pi/2, W = 0 -> G = 1/(x+x')  (Cauchy).
 * [verified: pin3_endpoints]
 *
 * ---------------------------------------------------------------------------
 * THE FOUR BINDING CORRECTIONS (notes/tilted_contour_validation_results.md
 * sections 2 and 7.2 -- each one costs accuracy or correctness if dropped)
 * ---------------------------------------------------------------------------
 * (1) RANK AT lambda > eps^2 lambda_max, NOT eps lambda_max. The Gram is
 *     G = K^* K, so its eigenvalues are the SQUARES of the kernel's singular
 *     values; the spec's own threshold keeps singular values only down to
 *     sqrt(eps) and delivers sqrt(eps) accuracy. Measured ladder (results
 *     section 2.1): rank 188 -> F rel err 3.9e-02 at eps lambda_max versus
 *     rank 376 -> 2.3e-05 at eps^2 lambda_max. `rank_eps` is reported for
 *     comparison with the spec's tables and is NEVER used.
 * (2) THE CONJUGATE-MIRROR TARGET ROWS. Eq 9 builds P from G^> G^<, which
 *     carries only the RESONANT half R(z) = sum_k w_k/(z - D_k). The physical
 *     P is R(z) + R(-z), and the anti-resonant poles sit BELOW the targets
 *     where a < 0 for essentially the whole spectrum -- the contour integral
 *     diverges. For real D_k and Hermitian residues R(-z) = [R(-conj z)]^dag,
 *     so the anti-resonant half comes from the SAME contour samples through
 *     extra F rows at z' = -conj(z) = -w + i d. `mirror_target` and
 *     `build_transform(..., with_mirror = true)` implement this; the nodes and
 *     the rank are unchanged. [verified: pin_mirror]
 * (3) eps_tr = eps^2 IN THE CONTOUR LENGTH. The spec's Table-2 S column is
 *     reproduced only with eps_tr = 1e-12 at eps = 1e-6; reading eq 4 with
 *     eps_tr = eps truncates the contour at HALF the needed length
 *     (results section 1 item 4). `eps_tr < 0` selects eps*eps.
 * (4) THE RANK IS CAPPED AT THE DOUBLE-PRECISION CONDITIONING CEILING.
 *     Measured LS conditioning (results section 5.1): ~1e4 at the spec rank,
 *     ~1e7 at the eps^2 rank (which is where we run and which is FINE), and
 *     ~3.5e12 at the fully resolved rank -- beyond N_s^sigma(1e-8) the
 *     transform is precision-limited, not rank-limited. The cap is
 *     #{lambda > `rank_ceiling` lambda_max} with rank_ceiling = 1e-16; at
 *     eps = 1e-6 it never binds, which is the point.
 *
 * ---------------------------------------------------------------------------
 * ROW WEIGHTING: `gram`, i.e. row k of the LS system scaled by the kernel
 * column norm ||K(.,x_k)||^{-1} = sqrt(2 a(x_k)). This is z-INDEPENDENT, so
 * ONE factorization serves every target, every q and every SCF iteration --
 * the spec's reuse claim. Measured equal in accuracy to the per-target
 * `relative` weighting at the eps^2 rank (1.16e-04 vs 1.18e-04, results
 * section 5.1), so nothing is given up. The alternative is not implemented.
 * ---------------------------------------------------------------------------
 */

#include <cmath>
#include <complex>
#include <functional>
#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/lapack.hpp"
#include "nda/linalg.hpp"
#include "utilities/check.hpp"

namespace tilted_contour {

  using dcomplex = std::complex<double>;

  /** 1 Hartree in eV / hbar in eV.fs (CODATA 2018) -- the reference's constants. */
  inline constexpr double hbar_eV_fs = 0.6582119569;
  inline constexpr double ha_to_eV   = 27.211386245988;

  // =========================================================================
  //  Inputs
  // =========================================================================

  /**
   * The tilted contour + the target-window geometry it must serve.
   * Energies are in ONE consistent unit; the campaign works in eV, the CoQui
   * consumer (TC-2) works in Hartree. Nothing here cares which.
   */
  struct contour_params_t {
    double dmin  = 0.0;    ///< Delta_min, bottom of the particle-hole continuum
    double dmax  = 0.0;    ///< Delta_max, top (set by nbnd)
    double W     = 0.0;    ///< target-window width above dmin: targets w in [dmin, dmin+W]
    double delta = 0.0;    ///< Im z of the target line (the flat value / the floor)
    double rho   = 0.65;   ///< tan(theta) * W / delta, in [0, 1)
    double eps   = 1e-6;   ///< rank tolerance
    double eps_tr = -1.0;  ///< contour-truncation tolerance; < 0 selects eps*eps (BINDING 3)
    long   nx    = 2500;   ///< adapted Delta-grid points (spec 4.1 step 2)
    double kappa = 0.35;   ///< s-grid grading constant (fraction of an oscillation period)
    double rank_ceiling = 1e-16;  ///< conditioning ceiling on the rank (BINDING 4)
    long   rank_max = -1;  ///< hard cap on the rank; < 0 = none (knob for the caller)
    long   rank_force = -1;  ///< DIAGNOSTIC: force this rank, bypassing the rule; < 0 = off
    /**
     * Worst-case damping rate gamma, overriding eq 4's flat-profile closed form
     * delta cos(theta) (1 - rho). < 0 = use the closed form. REQUIRED whenever an
     * `a_fun` profile is supplied: the growing-delta profile's worst case is
     * a_fun(0), not the flat expression, and S = ln(1/eps_tr)/gamma sets the whole
     * contour length. [the campaign does the same:
     *  tc_validation/tests/common.py::Geom, profile != 'flat' branch]
     */
    double gamma_override = -1.0;
  };

  /** Derived geometry: theta, gamma, S. Pure function of contour_params_t. */
  struct geometry_t {
    double tan_theta = 0.0, theta = 0.0, gamma = 0.0, S = 0.0, eps_tr = 0.0;
  };

  inline geometry_t derive_geometry(contour_params_t const &p) {
    utils::check(p.W > 0.0, "tilted_contour: W = {} must be > 0.", p.W);
    utils::check(p.delta > 0.0, "tilted_contour: delta = {} must be > 0.", p.delta);
    utils::check(p.rho >= 0.0 and p.rho < 1.0,
                 "tilted_contour: rho = {} must be in [0, 1).", p.rho);
    utils::check(p.eps > 0.0 and p.eps < 1.0,
                 "tilted_contour: eps = {} must be in (0, 1).", p.eps);
    geometry_t g;
    g.eps_tr    = (p.eps_tr < 0.0) ? p.eps * p.eps : p.eps_tr;   // BINDING 3
    utils::check(g.eps_tr > 0.0 and g.eps_tr < 1.0,
                 "tilted_contour: eps_tr = {} must be in (0, 1).", g.eps_tr);
    g.tan_theta = p.rho * p.delta / p.W;                          // eq 3/4
    g.theta     = std::atan(g.tan_theta);
    g.gamma     = (p.gamma_override > 0.0) ? p.gamma_override
                                           : p.delta * std::cos(g.theta) * (1.0 - p.rho);
    utils::check(g.gamma > 0.0,
                 "tilted_contour: gamma = {} must be > 0 (eq 4).", g.gamma);
    g.S         = std::log(1.0 / g.eps_tr) / g.gamma;             // eq 4
    return g;
  }

  /** a(x) of eq 5 (flat-delta worst-case target). x = Delta - Delta_min. */
  inline double a_of_x(double x, double W, double delta, double theta) {
    return (x - W) * std::sin(theta) + delta * std::cos(theta);
  }

  /**
   * Semiclassical point count, spec eq 6:
   *   N_s ~= ln(1/eps) W / (4 pi rho d) * ln[1 + rho (Dmax-Dmin)/((1-rho) W)],
   * with the rho -> 0 limit ln(1/eps)(Dmax-Dmin)/(4 pi d).
   * [verified: contour.py::eq6_Ns; pins.py::pin_eq6_table1 reproduces the spec's
   *  Table-1 "Eq. (6)" row to <1 count]
   */
  inline double eq6_Ns(double dmin, double dmax, double W, double delta,
                       double rho, double eps) {
    const double L = std::log(1.0 / eps);
    const double span = dmax - dmin;
    if (rho < 1e-12) return L * span / (4.0 * M_PI * delta);
    return (L * W / (4.0 * M_PI * rho * delta))
         * std::log1p(rho * span / ((1.0 - rho) * W));
  }

  // =========================================================================
  //  Grids
  // =========================================================================

  /**
   * Grid on [0, xmax] whose local spacing is proportional to a(x)
   * (spec 4.1 step 2). du = dx/a(x); for the closed-form a of eq 5 with
   * sin th > 0, u(x) = ln(a(x)/a(0))/sin th, invertible in closed form. For
   * sin th = 0 the grid is uniform. A general profile `a_fun` (the growing-delta
   * profile of TC-2) is handled by numerical inversion of the same cumulative map.
   * [verified: contour.py::adapted_grid; pins.py::pin_grid -- rank is grid
   *  converged to <=1 count between n = 750 and n = 6000]
   */
  inline nda::array<double, 1> adapted_grid(
      double xmax, double W, double delta, double theta, long n,
      std::function<double(double)> const &a_fun = {}) {
    utils::check(n >= 2, "tilted_contour::adapted_grid: n = {} must be >= 2.", n);
    nda::array<double, 1> out(n);
    const double st = std::sin(theta), ct = std::cos(theta);
    if (not a_fun) {
      if (st < 1e-14) {
        for (long i = 0; i < n; ++i)
          out(i) = xmax * double(i) / double(n - 1);
        return out;
      }
      const double a0   = delta * ct - W * st;
      const double amax = (xmax - W) * st + delta * ct;
      utils::check(a0 > 0.0 and amax > 0.0,
                   "tilted_contour::adapted_grid: a(0) = {} and a(xmax) = {} must both "
                   "be > 0 -- the tilt violates eq 3 (tan theta < delta/W).", a0, amax);
      const double umax = std::log(amax / a0) / st;
      for (long i = 0; i < n; ++i) {
        const double u = umax * double(i) / double(n - 1);
        const double a = a0 * std::exp(u * st);
        out(i) = (a - delta * ct) / st + W;
      }
      return out;
    }
    // general profile: integrate du = dx/a(x) on a fine uniform grid, then invert
    const long nf = std::max(20000L, 8 * n);
    nda::array<double, 1> xf(nf), u(nf);
    for (long i = 0; i < nf; ++i) xf(i) = xmax * double(i) / double(nf - 1);
    u(0) = 0.0;
    for (long i = 1; i < nf; ++i) {
      const double af0 = a_fun(xf(i - 1)), af1 = a_fun(xf(i));
      utils::check(af0 > 0.0 and af1 > 0.0,
                   "tilted_contour::adapted_grid: the supplied a(x) profile is not "
                   "positive (a = {} at x = {}).", std::min(af0, af1), xf(i));
      u(i) = u(i - 1) + 0.5 * (1.0 / af1 + 1.0 / af0) * (xf(i) - xf(i - 1));
    }
    long j = 0;
    for (long i = 0; i < n; ++i) {
      const double ut = u(nf - 1) * double(i) / double(n - 1);
      while (j + 2 < nf and u(j + 1) < ut) ++j;
      const double du = u(j + 1) - u(j);
      const double w  = (du > 0.0) ? (ut - u(j)) / du : 0.0;
      out(i) = xf(j) + w * (xf(j + 1) - xf(j));
    }
    return out;
  }

  /**
   * Fine s-grid for the ID, graded by band truncation along the contour
   * (spec 4.2 item 2). At contour position s only transitions with
   * a(D) s <~ ln(1/eps) survive, D <~ D_alive(s) = w + ln(1/eps)/(s sin th);
   * the integrand oscillates as exp(-i(D-w) s cos th), so the local sampling
   * requirement is h(s) <= kappa 2 pi / ((D_alive(s) - w) cos th), capped at
   * s = 0 by the full band width.
   * [verified: contour.py::graded_s_grid; pins.py::pin_id_grid -- doubling the
   *  grid density does not move the reconstruction error]
   *
   * NOTE the `eps` argument is the contour-truncation tolerance eps_tr, which
   * is what every campaign call site passes (tc_validation/tests/run_*.py).
   */
  inline nda::array<double, 1> graded_s_grid(double S, double dmax, double theta,
                                             double eps_tr, double kappa = 0.35,
                                             long smin_pts = 8) {
    const double st = std::sin(theta), ct = std::cos(theta);
    const double L = std::log(1.0 / eps_tr);
    const double hmin = kappa * 2.0 * M_PI / std::max(dmax * ct, 1e-12);
    std::vector<double> out{0.0};
    double s = 0.0;
    while (s < S) {
      double h;
      if (st < 1e-14) {
        h = hmin;
      } else {
        const double d_alive = L / std::max(s, 1e-30) / st;
        h = kappa * 2.0 * M_PI / std::max(std::min(d_alive, dmax) * ct, 1e-12);
        h = std::max(h, hmin);
      }
      s += h;
      out.push_back(std::min(s, S));
    }
    if (long(out.size()) < smin_pts) {
      out.assign(smin_pts, 0.0);
      for (long i = 0; i < smin_pts; ++i) out[i] = S * double(i) / double(smin_pts - 1);
    }
    nda::array<double, 1> g(long(out.size()));
    for (long i = 0; i < g.size(); ++i) g(i) = out[i];
    return g;
  }

  // =========================================================================
  //  Kernels
  // =========================================================================

  /** A[k,j] = exp(-i Delta_k s_j e^{-i th}) -- the RAW contour data kernel. */
  template<typename D_t, typename S_t>
  nda::array<dcomplex, 2> time_kernel(D_t const &D, S_t const &s, double theta) {
    const long nD = D.size(), nS = s.size();
    nda::array<dcomplex, 2> A(nD, nS);
    const dcomplex rot = std::exp(dcomplex(0.0, -theta));
    for (long k = 0; k < nD; ++k)
      for (long j = 0; j < nS; ++j)
        A(k, j) = std::exp(dcomplex(0.0, -1.0) * double(D(k)) * double(s(j)) * rot);
    return A;
  }

  /** phi_z(s) = exp(i z s e^{-i th}) -- the target factor of K. */
  template<typename S_t>
  nda::array<dcomplex, 1> target_phase(S_t const &s, dcomplex z, double theta) {
    const long nS = s.size();
    nda::array<dcomplex, 1> p(nS);
    const dcomplex rot = std::exp(dcomplex(0.0, -theta));
    for (long j = 0; j < nS; ++j)
      p(j) = std::exp(dcomplex(0.0, 1.0) * z * double(s(j)) * rot);
    return p;
  }

  /** Exact Gram, eq 5. Hermitian by construction. Row/col index = x-grid index. */
  template<typename X_t, typename A_t>
  nda::matrix<dcomplex> gram(X_t const &x, A_t const &a, double theta) {
    const long n = x.size();
    const double ct = std::cos(theta);
    nda::matrix<dcomplex> G(n, n);
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j)
        G(i, j) = 1.0 / dcomplex(a(i) + a(j), -(x(i) - x(j)) * ct);
    return G;
  }

  /** Unit-diagonal normalization Ghat = G / sqrt(G_xx G_x'x') (spec 2.2). */
  inline void normalize_gram_in_place(nda::matrix<dcomplex> &G) {
    const long n = G.extent(0);
    nda::array<double, 1> d(n);
    for (long i = 0; i < n; ++i) d(i) = std::sqrt(std::real(G(i, i)));
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) G(i, j) /= (d(i) * d(j));
  }

  // =========================================================================
  //  The contour
  // =========================================================================

  struct contour_t {
    contour_params_t p;
    geometry_t g;
    nda::array<double, 1> x;        ///< (nx) adapted grid, x = Delta - dmin
    nda::array<double, 1> a;        ///< (nx) a(x)
    nda::array<double, 1> eig;      ///< (nx) normalized-Gram eigenvalues, DESCENDING
    long rank_eps  = 0;             ///< #{lambda > eps   lambda_max} -- REPORTED ONLY
    long rank_eps2 = 0;             ///< #{lambda > eps^2 lambda_max} -- the accuracy rank
    long rank_ceil = 0;             ///< #{lambda > rank_ceiling lambda_max} -- the cap
    long rank      = 0;             ///< min(rank_eps2, rank_ceil, rank_max)
    nda::array<double, 1> s_grid;   ///< the graded fine s-grid the ID selects from
    nda::array<double, 1> s;        ///< (rank) selected nodes, ASCENDING
    nda::array<long, 1>   idx;      ///< (rank) their indices into s_grid, same order
    nda::array<dcomplex, 2> T;      ///< (rank, n_s_grid) ID coefficients (pin4); optional
    double id_identity = 0.0;       ///< max |T[:,J] - I| (0 by construction; pin4)

    double theta() const { return g.theta; }
    double S()     const { return g.S; }
    long   nnode() const { return rank; }
  };

  /**
   * Column-pivoted QR on K(s,x)^T -> `rank` contour nodes {s_j} (spec 4.1 step 4).
   *
   *   M[i,j] = K(s_j, x_i; z_worst) / ||K(., x_i)||_{L2(ds)},
   *   ||K(., x)||_{L2} = sqrt(G(x,x)) = sqrt(1/(2 a(x))),
   *
   * i.e. relative accuracy at every Delta, matching the unit-diagonal Gram
   * normalization the rank is read from. Column-pivoted QR of M selects `rank`
   * COLUMNS = s-nodes.
   * [verified: contour.py::id_nodes; pins.py::pin4_F_exactness]
   *
   * @param want_T  also build the ID coefficient matrix T with M ~ M[:,J] T
   *                (only the pin needs it; production does not).
   */
  inline void id_nodes(contour_t &c, bool want_T = false) {
    const long nx = c.x.size(), nS = c.s_grid.size();
    const long r  = c.rank;
    utils::check(r > 0 and r <= nS,
                 "tilted_contour::id_nodes: rank {} out of range for a {}-point s grid.",
                 r, nS);

    const dcomplex z_worst(c.p.dmin + c.p.W, c.p.delta);
    nda::array<double, 1> D(nx);
    for (long i = 0; i < nx; ++i) D(i) = c.p.dmin + c.x(i);

    // M in F_layout for geqp3 (geqp3 destroys its input, and M is not needed after)
    nda::matrix<dcomplex, nda::F_layout> A(nx, nS);
    {
      auto ph = target_phase(c.s_grid, z_worst, c.g.theta);
      const dcomplex rot = std::exp(dcomplex(0.0, -c.g.theta));
      for (long i = 0; i < nx; ++i) {
        const double nrm = std::sqrt(1.0 / (2.0 * c.a(i)));
        for (long j = 0; j < nS; ++j)
          A(i, j) = std::exp(dcomplex(0.0, -1.0) * D(i) * c.s_grid(j) * rot)
                  * ph(j) / nrm;
      }
    }

    nda::array<int, 1> jpvt(nS);
    jpvt() = 0;                                      // all columns free
    nda::array<dcomplex, 1> tau(std::min(nx, nS));
    nda::array<dcomplex, 1> work(1);                 // nda's 3-arg overload passes an
    nda::lapack::geqp3(A, jpvt, tau, work);          // EMPTY work, which trips its own
                                                     // min_stride EXPECTS under NDEBUG-off

    // nda::lapack::geqp3 already shifts jpvt to 0-based ("column j of A.P was column
    // jpvt(j) of A"), so no further -1 here.
    std::vector<long> piv(nS);
    for (long j = 0; j < nS; ++j) piv[j] = long(jpvt(j));

    // the selected nodes, sorted ascending (a column permutation of the LS system,
    // which the minimum-norm solution is equivariant under)
    std::vector<long> J(piv.begin(), piv.begin() + r);
    std::sort(J.begin(), J.end());
    c.idx = nda::array<long, 1>(r);
    c.s   = nda::array<double, 1>(r);
    for (long i = 0; i < r; ++i) { c.idx(i) = J[i]; c.s(i) = c.s_grid(J[i]); }

    if (not want_T) return;

    // T = [ I at piv[:r] ; R11^{-1} R12 at piv[r:] ]:  M ~ M[:,J] T.
    // R lives in the upper triangle of A (F_layout, so A(i,j) is R(i,j) for i<=j).
    nda::matrix<dcomplex> T12(r, nS - r);
    for (long j = 0; j < nS - r; ++j)
      for (long i = 0; i < r; ++i) T12(i, j) = A(i, r + j);
    for (long j = 0; j < nS - r; ++j)                 // back substitution, R11 upper
      for (long i = r - 1; i >= 0; --i) {
        dcomplex acc = T12(i, j);
        for (long k2 = i + 1; k2 < r; ++k2) acc -= A(i, k2) * T12(k2, j);
        T12(i, j) = acc / A(i, i);
      }
    c.T = nda::array<dcomplex, 2>(r, nS);
    c.T() = dcomplex(0.0, 0.0);
    // rows of T follow the PIVOT order; c.s is sorted, so map pivot slot -> sorted slot
    std::vector<long> slot(r);
    for (long i = 0; i < r; ++i)
      slot[i] = long(std::lower_bound(J.begin(), J.end(), piv[i]) - J.begin());
    for (long i = 0; i < r; ++i) c.T(slot[i], piv[i]) = dcomplex(1.0, 0.0);
    for (long j = 0; j < nS - r; ++j)
      for (long i = 0; i < r; ++i) c.T(slot[i], piv[r + j]) = T12(i, j);
    c.id_identity = 0.0;
    for (long i = 0; i < r; ++i)
      for (long j = 0; j < r; ++j) {
        const dcomplex e = c.T(i, c.idx(j)) - dcomplex(i == j ? 1.0 : 0.0, 0.0);
        c.id_identity = std::max(c.id_identity, std::abs(e));
      }
  }

  /**
   * Build the whole contour: geometry -> adapted Delta grid -> Gram (eq 5) ->
   * Hermitian eigensolve -> rank (BINDING 1 + BINDING 4) -> graded s-grid ->
   * column-pivoted QR -> nodes.
   *
   * Cost is dominated by the (nx x nx) Hermitian eigensolve; nx = 2500 is the
   * spec's own resolution and pin_grid shows the rank moves by <= 1 count
   * between 750 and 6000, so nx is a safe cost knob.
   *
   * @param a_fun  optional a(x) profile (the growing-delta profile). When empty
   *               the closed form of eq 5 is used. Any positive a(x) still
   *               gives a genuine PSD Gram -- see the factorization note in
   *               contour.py's docstring.
   */
  inline contour_t build_contour(contour_params_t const &p,
                                 std::function<double(double)> const &a_fun = {},
                                 bool want_T = false) {
    utils::check(p.dmax > p.dmin,
                 "tilted_contour: dmax = {} must exceed dmin = {}.", p.dmax, p.dmin);
    contour_t c;
    c.p = p;
    c.g = derive_geometry(p);

    c.x = adapted_grid(p.dmax - p.dmin, p.W, p.delta, c.g.theta, p.nx, a_fun);
    c.a = nda::array<double, 1>(p.nx);
    for (long i = 0; i < p.nx; ++i)
      c.a(i) = a_fun ? a_fun(c.x(i)) : a_of_x(c.x(i), p.W, p.delta, c.g.theta);

    {
      auto G = gram(c.x, c.a, c.g.theta);
      normalize_gram_in_place(G);
      auto ev = nda::linalg::eigenvalues(G);          // ascending
      c.eig = nda::array<double, 1>(p.nx);
      for (long i = 0; i < p.nx; ++i)
        c.eig(i) = std::max(0.0, ev(p.nx - 1 - i));   // descending, clipped at 0
    }
    const double lmax = c.eig(0);
    auto count = [&](double thr) {
      long n = 0;
      for (long i = 0; i < c.eig.size(); ++i) if (c.eig(i) > thr * lmax) ++n;
      return n;
    };
    c.rank_eps  = count(p.eps);
    c.rank_eps2 = count(p.eps * p.eps);               // BINDING 1
    c.rank_ceil = count(p.rank_ceiling);              // BINDING 4
    c.rank = std::min(c.rank_eps2, c.rank_ceil);
    if (p.rank_max > 0) c.rank = std::min(c.rank, p.rank_max);
    if (p.rank_force > 0) c.rank = p.rank_force;      // diagnostic override (pin4)
    utils::check(c.rank > 0, "tilted_contour: the eps-rank came out 0.");

    c.s_grid = graded_s_grid(c.g.S, p.dmax, c.g.theta, c.g.eps_tr, p.kappa);
    c.rank = std::min(c.rank, c.s_grid.size());
    id_nodes(c, want_T);
    return c;
  }

  // =========================================================================
  //  The transform F
  // =========================================================================

  /** z' = -conj(z): the conjugate-mirror target of BINDING 2. */
  inline dcomplex mirror_target(dcomplex z) { return -std::conj(z); }

  struct transform_t {
    nda::array<dcomplex, 2> F;      ///< (ntarget_total, rank)
    nda::array<dcomplex, 1> z;      ///< (ntarget_total) the target list actually solved
    long n_res = 0;                 ///< rows [0, n_res) are the resonant targets;
                                    ///< rows [n_res, 2*n_res) their conjugate mirrors
    double cond = 0.0;              ///< 2-norm condition number of the LS matrix
    nda::array<double, 1> relres;   ///< (ntarget_total) LS relative residual per target
    double relres_max = 0.0;
  };

  /**
   * Least-squares transform (spec 4.1 step 5) with the `gram` row weighting:
   *
   *   min_F || diag(rw) [ A0 F^T - B0 ] ||_F,
   *   A0[k,j] = exp(-i D_k s_j e^{-i th}),   B0[k,t] = 1/(z_t - D_k),
   *   rw_k    = sqrt(2 a(x_k))  ( = 1/||K(., x_k)|| ).
   *
   * The weight is z-INDEPENDENT, so ONE gelss factorization serves every target,
   * every q and every SCF iteration.
   * [verified: contour.py::build_F, weighting = 'gram']
   *
   * @param with_mirror  append the conjugate-mirror rows z' = -conj(z) (BINDING 2).
   *                     The physical response is then
   *                        P(z_t) = R_t + [R_{t + n_res}]^dag
   *                     for matrix-valued residues, or R_t + conj(R_{t+n_res})
   *                     for scalars -- see `combine_mirror`.
   */
  inline transform_t build_transform(contour_t const &c,
                                     nda::array<dcomplex, 1> const &z_res,
                                     bool with_mirror = true) {
    const long nD = c.x.size(), r = c.rank, nz = z_res.size();
    utils::check(nz > 0, "tilted_contour::build_transform: empty target list.");
    transform_t out;
    out.n_res = nz;
    const long nt = with_mirror ? 2 * nz : nz;
    out.z = nda::array<dcomplex, 1>(nt);
    for (long t = 0; t < nz; ++t) out.z(t) = z_res(t);
    if (with_mirror)
      for (long t = 0; t < nz; ++t) out.z(nz + t) = mirror_target(z_res(t));

    nda::array<double, 1> D(nD), rw(nD);
    for (long k = 0; k < nD; ++k) {
      D(k)  = c.p.dmin + c.x(k);
      rw(k) = std::sqrt(2.0 * c.a(k));
    }

    nda::matrix<dcomplex, nda::F_layout> A(nD, r);
    {
      const dcomplex rot = std::exp(dcomplex(0.0, -c.g.theta));
      for (long k = 0; k < nD; ++k)
        for (long j = 0; j < r; ++j)
          A(k, j) = std::exp(dcomplex(0.0, -1.0) * D(k) * c.s(j) * rot) * rw(k);
    }
    nda::matrix<dcomplex, nda::F_layout> B(nD, nt), B0(nD, nt);
    for (long t = 0; t < nt; ++t)
      for (long k = 0; k < nD; ++k) {
        B0(k, t) = 1.0 / (out.z(t) - D(k));
        B(k, t)  = B0(k, t) * rw(k);
      }
    nda::matrix<dcomplex, nda::F_layout> Acopy(A);

    // numpy's lstsq(rcond=None) uses rcond = eps_mach * max(m, n); match it so the
    // C++ and python transforms are the same object, not merely the same formula.
    const double rcond = std::numeric_limits<double>::epsilon() * double(std::max(nD, r));
    nda::array<double, 1> sv(std::min(nD, r));
    int lsrank = 0;
    nda::lapack::gelss(Acopy, B, sv, rcond, lsrank);
    out.cond = (sv(sv.size() - 1) > 0.0) ? sv(0) / sv(sv.size() - 1)
                                         : std::numeric_limits<double>::infinity();

    out.F = nda::array<dcomplex, 2>(nt, r);
    for (long t = 0; t < nt; ++t)
      for (long j = 0; j < r; ++j) out.F(t, j) = B(j, t);   // gelss writes X into B

    out.relres = nda::array<double, 1>(nt);
    out.relres_max = 0.0;
    for (long t = 0; t < nt; ++t) {
      double num = 0.0, den = 0.0;
      for (long k = 0; k < nD; ++k) {
        dcomplex acc(0.0, 0.0);
        for (long j = 0; j < r; ++j) acc += A(k, j) * out.F(t, j);
        const dcomplex b = B0(k, t) * rw(k);
        num += std::norm(acc - b);
        den += std::norm(b);
      }
      out.relres(t) = (den > 0.0) ? std::sqrt(num / den) : 0.0;
      out.relres_max = std::max(out.relres_max, out.relres(t));
    }
    return out;
  }

  /**
   * The mirror combination for a SCALAR response:
   *   P(z_t) = R(z_t) + conj(R(z'_t)),   z'_t = -conj(z_t).
   * For a MATRIX response with Hermitian residues the conjugation becomes the
   * conjugate TRANSPOSE, R(z_t) + R(z'_t)^dag -- the same distinction the RW-2
   * quadrature had to make (rw2_report section 3.7). Callers holding matrices
   * must use the dagger; this helper is the scalar case only.
   * [verified: contour.py / models.py, pins.py::pin_mirror; the campaign's B1
   *  combination is `Rf = got[:n] + conj(got[n:])`]
   */
  inline dcomplex combine_mirror(dcomplex R_res, dcomplex R_mir) {
    return R_res + std::conj(R_mir);
  }

} // namespace tilted_contour

#endif // COQUI_TILTED_CONTOUR_HPP
