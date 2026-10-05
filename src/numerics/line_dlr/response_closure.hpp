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

#ifndef COQUI_NUMERICS_LINE_DLR_RESPONSE_CLOSURE_HPP
#define COQUI_NUMERICS_LINE_DLR_RESPONSE_CLOSURE_HPP

/**
 * Real-axis closure of a SCALAR bosonic response sampled on the tilted line (plan S9b; design study
 * notes/bosonic_closure_design.md section 7; notes section "Optics from the line").
 *
 * Every scalar response of the optics, r(zeta) in {h = eps^-1_00 - 1, m = 1 - eps_M = h / (1 + h)}, is an ODD function with a
 * POSITIVE particle measure:
 *     r(zeta) = sum_j r_j [ 1/(zeta - Omega_j) - 1/(zeta + Omega_j) ],   Omega_j > 0,  r_j >= 0,
 * so that -Im r(omega + i eta) >= 0 for omega, eta > 0 (loss = -Im h, eps_2 = Im eps = -Im m). Sign convention: we close
 * m = 1 - eps_M (the study's wording), whose measure is >= 0 in the convention above; eps_M - 1 = -m has the negative one.
 *
 * Starting representation (as the SCF code): the odd least-squares fit of the line data on the bosonic basis poles nu_j with
 * REAL unknowns (odd_fit; r(conj z) = conj r(z) built in). Its residues are signed and not unique (near-threshold directions);
 * the pole function is accurate to the data. Two real-axis estimates are built from it / from the data:
 *
 *  MB (multi-scale blend, the primary estimate): for each scale w_i (log-spaced, 3-6 of them, 0.12-1 Ha), the one-sided
 *     Cayley moments about the line crossing (centre 0, NEVER shifted)
 *         C_i^(n) = sum_j r_j u_i(nu_j)^n,   u_i(nu) = (nu + i w_i) / (nu - i w_i),   n = 0..K+1,
 *     are closed with the 1x1 upfold_block (cayley.hpp) -> poles d_il > 0, weights a_il = |W_il|^2; the evaluated functions
 *     are blended with a cos^2 partition of unity in log omega peaked at the scales (partition_weights):
 *         r(omega + i eta) = sum_i chi_i(omega) sum_l a_il [ 1/(z - d_il) - 1/(z + d_il) ].
 *     Every closure keeps its moments exact; -Im r >= 0 as a convex combination. Weight at d <= 0 (<= 1e-5 at the K rule)
 *     is dropped and reported.
 *  G (NNLS): a positive fit of the line data on 1500 log-spaced candidate poles in [bos_gap, lam_b] (Lawson-Hanson on the
 *     real system [Re; Im] of the odd kernel, column-scaled). |r_MB - r_G| is the a-posteriori error bar (it tracks the true
 *     error of MB within 2x in 85% of the study's cases, section 3.4).
 *
 * Order and Gram cut from the data accuracy delta (relative, max-norm; estimated from the odd-fit residual at the nodes,
 * floor 1e-10): K = min(K_cap, floor(ln(0.1 / delta) / ln(1 / r))), r = tan(pi/4 - theta/2) (the lens bound of a centre
 * on the line crossing: moment noise grows as delta r^-n), K_cap = 48 ln(1/r_20deg) / ln(1/r) (48 at 20 deg, 97 at 10 deg,
 * 195 at 5 deg); tol_gram = max(1e-10, 10 delta).
 * Statics and sum rules come from the line FIT, not from a closure: r(0) = -2 sum_j r_j / nu_j, f-sum = lim z^2 r / 2 ... i.e.
 * sum_j 2 r_j nu_j (= omega_p^2 for both h and m; int_0^inf omega eps_2 d omega = (pi/2) omega_p^2).
 *
 * Optical quantities from eps(omega) (optical_quantities): eps1, eps2, N = n + i kappa = sqrt(eps) (Im N >= 0),
 * alpha = omega eps2 / (n c) = 2 omega kappa / c, R = |(N - 1)/(N + 1)|^2, loss = -Im 1/eps,
 * sigma = -i omega (eps - 1) / (4 pi) (sigma1 = omega eps2 / 4 pi). UNITS: omega in Ha (x 27.211386 = eV), c = 137.035999
 * a.u., alpha in 1/bohr (x ALPHA_AU_TO_CM = 1.8897261e8 -> cm^-1), sigma in atomic units e^2/(hbar a0)
 * (x SIGMA_AU_TO_S_CM = 4.599848e4 -> S/cm).
 *
 * Transcription of scripts/closure_design/{cd_lib.py (odd_fit, closure_B, partition_weights, nnls_fit), run_designs.py
 * (multiscale_blend, k_auto), optics.py (quantities)}; the NNLS is the Lawson-Hanson algorithm (Solving Least Squares
 * Problems, ch. 23: NNLS with H12 / G1 / G2), as scipy.optimize.nnls. Host only, pure math.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

#include "numerics/line_dlr/line_dlr_utils.hpp"
#include "numerics/line_dlr/cayley.hpp"

namespace numerics::line_dlr::response {

constexpr double HA_EV            = 27.211386245988;
constexpr double C_AU             = 137.035999;          ///< speed of light, atomic units
constexpr double ALPHA_AU_TO_CM   = 1.0 / 0.529177210903e-8;   ///< 1/bohr -> cm^-1
constexpr double SIGMA_AU_TO_S_CM = 4.599848e4;          ///< e^2/(hbar a0) -> S/cm

/// odd kernel 1/(z - nu) - 1/(z + nu)
inline ComplexType odd_k(ComplexType z, double nu) { return 1.0 / (z - nu) - 1.0 / (z + nu); }

/// r(z) = sum_j r_j [1/(z - W_j) - 1/(z + W_j)]
struct odd_measure_t {
  nda::array<double, 1> W, r;

  odd_measure_t() = default;
  odd_measure_t(nda::array<double, 1> W_, nda::array<double, 1> r_) : W(std::move(W_)), r(std::move(r_)) {
    utils::check(W.size() == r.size(), "response::odd_measure_t: {} poles vs {} residues", W.size(), r.size());
  }
  long size() const { return W.size(); }
  ComplexType operator()(ComplexType z) const {
    ComplexType s(0.0);
    for (long j = 0; j < W.size(); ++j) s += r(j) * odd_k(z, W(j));
    return s;
  }
  nda::array<ComplexType, 1> operator()(nda::array<ComplexType, 1> const &z) const {
    nda::array<ComplexType, 1> o(z.size());
    for (long i = 0; i < z.size(); ++i) o(i) = (*this)(z(i));
    return o;
  }
  /// lim z^2 r(z) = sum 2 r_j W_j (f-sum: omega_p^2)
  double fsum() const {
    double s = 0.0;
    for (long j = 0; j < W.size(); ++j) s += 2.0 * r(j) * W(j);
    return s;
  }
  /// r(0) = -2 sum r_j / W_j
  double static_value() const {
    double s = 0.0;
    for (long j = 0; j < W.size(); ++j) s -= 2.0 * r(j) / W(j);
    return s;
  }
  /// sum_j r_j (the zeroth moment of the particle measure)
  double weight() const {
    double s = 0.0;
    for (long j = 0; j < W.size(); ++j) s += r(j);
    return s;
  }
};

// ------------------------------------------------------------------------------------------------------------------
// odd least-squares fit (cd_lib.odd_fit) and the noise estimate
// ------------------------------------------------------------------------------------------------------------------
/// real residues f_j on the poles nu from data at zeta: min || [Re K; Im K] f - [Re d; Im d] || (numpy lstsq cutoff)
inline nda::array<double, 1> odd_fit(nda::array<ComplexType, 1> const &zeta, nda::array<ComplexType, 1> const &data,
                                     nda::array<double, 1> const &nu, double rcond = -1.0) {
  const long nz = zeta.size(), r = nu.size();
  utils::check(data.size() == nz, "response::odd_fit: {} data vs {} nodes", data.size(), nz);
  nda::array<ComplexType, 2> A(2 * nz, r), b(2 * nz, 1);
  for (long i = 0; i < nz; ++i) {
    for (long j = 0; j < r; ++j) {
      const ComplexType k = odd_k(zeta(i), nu(j));
      A(i, j)      = k.real();
      A(nz + i, j) = k.imag();
    }
    b(i, 0)      = data(i).real();
    b(nz + i, 0) = data(i).imag();
  }
  auto x = detail::lstsq(A, b, rcond);
  nda::array<double, 1> f(r);
  for (long j = 0; j < r; ++j) f(j) = x(j, 0).real();
  return f;
}

/// max |sum_j f_j K(zeta_i, nu_j) - d_i| / max |d_i|: the relative accuracy of the data as seen by the fit
inline double fit_residual(nda::array<ComplexType, 1> const &zeta, nda::array<ComplexType, 1> const &data,
                           nda::array<double, 1> const &nu, nda::array<double, 1> const &f) {
  double e = 0.0, s = 0.0;
  odd_measure_t m(nu, f);
  for (long i = 0; i < zeta.size(); ++i) {
    e = std::max(e, std::abs(m(zeta(i)) - data(i)));
    s = std::max(s, std::abs(data(i)));
  }
  return s > 0.0 ? e / s : e;
}

// ------------------------------------------------------------------------------------------------------------------
// NNLS (Lawson-Hanson)
// ------------------------------------------------------------------------------------------------------------------
struct nnls_result_t {
  nda::array<double, 1> x;
  double rnorm = 0.0;
  long iter = 0;
  int mode = 1;   ///< 1 = success, 3 = iteration limit
};

namespace detail_nnls {
/// Householder H12 (Lawson-Hanson): mode 1 constructs (and applies), mode 2 applies. 0-based pivot lp, rows l1..m-1 of u.
/// c: first vector to transform, ncv vectors with stride icv (element stride 1).
inline void h12(int mode, long lp, long l1, long m, double *u, double &up, double *c, long icv, long ncv) {
  if (lp < 0 or lp >= l1 or l1 >= m) return;
  double cl = std::abs(u[lp]);
  if (mode == 1) {
    for (long j = l1; j < m; ++j) cl = std::max(std::abs(u[j]), cl);
    if (cl <= 0.0) return;
    const double clinv = 1.0 / cl;
    double sm = (u[lp] * clinv) * (u[lp] * clinv);
    for (long j = l1; j < m; ++j) sm += (u[j] * clinv) * (u[j] * clinv);
    cl = cl * std::sqrt(sm);
    if (u[lp] > 0.0) cl = -cl;
    up    = u[lp] - cl;
    u[lp] = cl;
  } else if (cl <= 0.0) {
    return;
  }
  if (ncv <= 0) return;
  double b = up * u[lp];
  if (b >= 0.0) return;
  b = 1.0 / b;
  for (long j = 0; j < ncv; ++j) {
    double *cj = c + j * icv;
    double sm  = cj[lp] * up;
    for (long i = l1; i < m; ++i) sm += cj[i] * u[i];
    if (sm != 0.0) {
      sm *= b;
      cj[lp] += sm * up;
      for (long i = l1; i < m; ++i) cj[i] += sm * u[i];
    }
  }
}
/// Givens G1: (c, s, sig) with [c s; -s c] (a, b)^T = (sig, 0)^T
inline void g1(double a, double b, double &c, double &s, double &sig) {
  if (std::abs(a) > std::abs(b)) {
    const double xr = b / a, yr = std::sqrt(1.0 + xr * xr);
    c   = std::copysign(1.0 / yr, a);
    s   = c * xr;
    sig = std::abs(a) * yr;
  } else if (b != 0.0) {
    const double xr = a / b, yr = std::sqrt(1.0 + xr * xr);
    s   = std::copysign(1.0 / yr, b);
    c   = s * xr;
    sig = std::abs(b) * yr;
  } else {
    sig = 0.0;
    c   = 0.0;
    s   = 1.0;
  }
}
inline void g2(double c, double s, double &x, double &y) {
  const double xr = c * x + s * y;
  y               = -s * x + c * y;
  x               = xr;
}
} // namespace detail_nnls

/**
 * min ||A x - b||_2 subject to x >= 0 (Lawson & Hanson, NNLS; A (m x n) and b are copied and overwritten internally).
 * itmax < 0: 3 n (scipy's default).
 */
inline nnls_result_t nnls(nda::matrix<double, nda::F_layout> A, nda::array<double, 1> b, long itmax = -1) {
  using namespace detail_nnls;
  const long m = A.extent(0), n = A.extent(1);
  utils::check(b.size() == m, "response::nnls: b has {} rows, A {}", b.size(), m);
  if (itmax < 0) itmax = 3 * n;
  nnls_result_t res;
  res.x = nda::array<double, 1>(n);
  res.x() = 0.0;
  auto &x = res.x;
  std::vector<double> w(n, 0.0), zz(m, 0.0);
  std::vector<long> index(n);
  for (long i = 0; i < n; ++i) index[i] = i;
  long iz1 = 0, iz2 = n - 1, nsetp = 0, npp1 = 0, iter = 0;
  const double factor = 0.01;
  double up = 0.0;
  auto col = [&](long j) { return &A(0, j); };
  auto solve = [&]() {   // triangular solve of the passive set: zz(0..nsetp-1)
    long jj = -1;
    for (long l = 0; l < nsetp; ++l) {
      const long ip = nsetp - 1 - l;
      if (l != 0)
        for (long ii = 0; ii <= ip; ++ii) zz[ii] -= A(ii, jj) * zz[ip + 1];
      jj = index[ip];
      zz[ip] /= A(ip, jj);
    }
  };
  bool done = false;
  while (not done) {
    // main loop (label 30)
    if (iz1 > iz2 or nsetp >= m) break;
    for (long iz = iz1; iz <= iz2; ++iz) {
      const long j = index[iz];
      double sm    = 0.0;
      for (long l = npp1; l < m; ++l) sm += A(l, j) * b(l);
      w[j] = sm;
    }
    long iz = -1, j = -1;
    while (true) {   // label 60: the most positive dual whose column is admissible
      double wmax = 0.0;
      long izmax  = -1;
      for (long k = iz1; k <= iz2; ++k) {
        const long jk = index[k];
        if (w[jk] > wmax) {
          wmax  = w[jk];
          izmax = k;
        }
      }
      if (wmax <= 0.0) {
        done = true;
        break;
      }
      iz = izmax;
      j  = index[iz];
      const double asave = A(npp1, j);
      h12(1, npp1, npp1 + 1, m, col(j), up, nullptr, 0, 0);
      double unorm = 0.0;
      for (long l = 0; l < nsetp; ++l) unorm += A(l, j) * A(l, j);
      unorm = std::sqrt(unorm);
      if ((unorm + std::abs(A(npp1, j)) * factor) - unorm > 0.0) {
        for (long l = 0; l < m; ++l) zz[l] = b(l);
        h12(2, npp1, npp1 + 1, m, col(j), up, zz.data(), m, 1);
        const double ztest = zz[npp1] / A(npp1, j);
        if (ztest > 0.0) break;   // label 140
      }
      A(npp1, j) = asave;
      w[j]       = 0.0;
    }
    if (done) break;
    // label 140: move j to the passive set
    for (long l = 0; l < m; ++l) b(l) = zz[l];
    index[iz]  = index[iz1];
    index[iz1] = j;
    ++iz1;
    nsetp = npp1 + 1;
    ++npp1;
    if (iz1 <= iz2)
      for (long jz = iz1; jz <= iz2; ++jz) h12(2, nsetp - 1, npp1, m, col(j), up, col(index[jz]), m, 1);
    if (nsetp != m)
      for (long l = npp1; l < m; ++l) A(l, j) = 0.0;
    w[j] = 0.0;
    solve();
    // inner loop (label 210)
    while (true) {
      ++iter;
      if (iter > itmax) {
        res.mode = 3;
        done     = true;
        break;
      }
      double alpha = 2.0;
      long jj      = -1;
      for (long ip = 0; ip < nsetp; ++ip) {
        const long l = index[ip];
        if (zz[ip] <= 0.0) {
          const double t = -x(l) / (zz[ip] - x(l));
          if (alpha > t) {
            alpha = t;
            jj    = ip;
          }
        }
      }
      if (alpha == 2.0) {   // label 330: all positive
        for (long ip = 0; ip < nsetp; ++ip) x(index[ip]) = zz[ip];
        break;
      }
      for (long ip = 0; ip < nsetp; ++ip) {
        const long l = index[ip];
        x(l)         = x(l) + alpha * (zz[ip] - x(l));
      }
      long i = index[jj];
      while (true) {   // label 260: move i (at position jj) back to the active set
        x(i) = 0.0;
        if (jj != nsetp - 1) {
          for (long jp = jj + 1; jp < nsetp; ++jp) {
            const long ii = index[jp];
            index[jp - 1] = ii;
            double cc = 0.0, ss = 0.0, sig = 0.0;
            g1(A(jp - 1, ii), A(jp, ii), cc, ss, sig);
            A(jp - 1, ii) = sig;
            A(jp, ii)     = 0.0;
            for (long l = 0; l < n; ++l)
              if (l != ii) g2(cc, ss, A(jp - 1, l), A(jp, l));
            g2(cc, ss, b(jp - 1), b(jp));
          }
        }
        npp1 = nsetp - 1;
        --nsetp;
        --iz1;
        index[iz1] = i;
        long bad = -1;
        for (long q = 0; q < nsetp; ++q)
          if (x(index[q]) <= 0.0) {
            bad = q;
            break;
          }
        if (bad < 0) break;
        jj = bad;
        i  = index[jj];
      }
      for (long l = 0; l < m; ++l) zz[l] = b(l);
      solve();
    }
  }
  double sm = 0.0;
  for (long l = npp1; l < m; ++l) sm += b(l) * b(l);
  res.rnorm = std::sqrt(sm);
  res.iter  = iter;
  return res;
}

/// candidate poles of the NNLS fit: n log-spaced in [lo, hi] (the study: 1500 in [0.02, 12] Ha = [bos_gap, lam_b])
inline nda::array<double, 1> nnls_grid(double lo, double hi, long n = 1500) { return detail::logspace(lo, hi, n); }

struct nnls_fit_t {
  odd_measure_t measure;   ///< the positive poles only (x > 0)
  nda::array<double, 1> x; ///< all residues on the grid
  double rnorm = 0.0;      ///< residual 2-norm of the column-scaled real system (= unscaled residual)
  double resid_rel = 0.0;  ///< max |fit - data| / max |data| at the nodes
  long iter = 0;
  int mode = 1;
};

/// G: positive residues on the pole grid by NNLS on the line data (cd_lib.nnls_fit, rel_ridge = 0). itmax < 0: 50 n (study)
inline nnls_fit_t nnls_odd_fit(nda::array<ComplexType, 1> const &zeta, nda::array<ComplexType, 1> const &data,
                               nda::array<double, 1> const &grid, long itmax = -1) {
  const long nz = zeta.size(), n = grid.size();
  nda::matrix<double, nda::F_layout> A(2 * nz, n);
  nda::array<double, 1> sc(n), b(2 * nz);
  for (long j = 0; j < n; ++j) {
    double s = 0.0;
    for (long i = 0; i < nz; ++i) {
      const ComplexType k = odd_k(zeta(i), grid(j));
      A(i, j)      = k.real();
      A(nz + i, j) = k.imag();
      s += std::norm(k);
    }
    sc(j) = std::sqrt(s);
    for (long i = 0; i < 2 * nz; ++i) A(i, j) /= sc(j);
  }
  for (long i = 0; i < nz; ++i) {
    b(i)      = data(i).real();
    b(nz + i) = data(i).imag();
  }
  auto r = nnls(A, b, itmax < 0 ? 50 * n : itmax);
  nnls_fit_t out;
  out.x = nda::array<double, 1>(n);
  long npos = 0;
  for (long j = 0; j < n; ++j) {
    out.x(j) = r.x(j) / sc(j);
    if (out.x(j) > 0.0) ++npos;
  }
  nda::array<double, 1> W(npos), rr(npos);
  for (long j = 0, l = 0; j < n; ++j)
    if (out.x(j) > 0.0) {
      W(l)  = grid(j);
      rr(l) = out.x(j);
      ++l;
    }
  out.measure   = odd_measure_t(std::move(W), std::move(rr));
  out.rnorm     = r.rnorm;
  out.iter      = r.iter;
  out.mode      = r.mode;
  out.resid_rel = 0.0;
  double s = 0.0;
  for (long i = 0; i < nz; ++i) {
    out.resid_rel = std::max(out.resid_rel, std::abs(out.measure(zeta(i)) - data(i)));
    s             = std::max(s, std::abs(data(i)));
  }
  if (s > 0.0) out.resid_rel /= s;
  return out;
}

// ------------------------------------------------------------------------------------------------------------------
// rules: lens bound, K, Gram cut, scales
// ------------------------------------------------------------------------------------------------------------------
/// r = tan(pi/4 - theta/2): min |u| of the Cayley image of the data rays for a centre on the line crossing
inline double lens_r(double theta_rad) { return std::tan(std::numbers::pi / 4.0 - 0.5 * theta_rad); }

/// K_cap = 48 ln(1/r_20) / ln(1/r) (48 at 20 deg)
inline long k_cap(double theta_rad) {
  const double r20 = lens_r(20.0 * std::numbers::pi / 180.0);
  return long(48.0 * std::log(1.0 / r20) / std::log(1.0 / lens_r(theta_rad)));
}

/// K = min(K_cap, floor(ln(tau / max(delta, 1e-10)) / ln(1/r)))   (notes/bosonic_closure_design.md 7.1, item 4)
inline long k_rule(double delta, double theta_rad, double tau = 0.1, long kcap = -1) {
  if (kcap < 0) kcap = k_cap(theta_rad);
  const double de = std::max(delta, 1e-10);
  const long k    = long(std::floor(std::log(tau / de) / std::log(1.0 / lens_r(theta_rad))));
  return std::max(2L, std::min(kcap, k));
}

/// tol_gram = max(1e-10, 10 delta)
inline double tol_gram_rule(double delta) { return std::max(1e-10, 10.0 * delta); }

/// n log-spaced scales in [lo, hi] (np.exp(np.linspace(log lo, log hi, n)))
inline std::vector<double> log_scales(double lo, double hi, long n) {
  auto s = detail::logspace(lo, hi, n);
  return std::vector<double>(s.begin(), s.end());
}

/**
 * Automatic scales (7.1 item 2): Omega_lo = max(lo_floor, nu at 1% of the cumulative sum |r_j|), Omega_hi = min(hi_cap, the
 * nu below which 95% of sum |r_j| nu_j lies); n log-spaced scales in [Omega_lo, Omega_hi] (one scale if Omega_hi <= 1.05
 * Omega_lo). The measure should be the positive one (NNLS) when available.
 */
inline std::vector<double> auto_scales(odd_measure_t const &m, long n = 4, double lo_floor = 0.12, double hi_cap = 1.0) {
  const long N = m.size();
  std::vector<long> o(N);
  for (long j = 0; j < N; ++j) o[j] = j;
  std::sort(o.begin(), o.end(), [&](long a, long b) { return m.W(a) < m.W(b); });
  double s0 = 0.0, s1 = 0.0;
  for (long j = 0; j < N; ++j) {
    s0 += std::abs(m.r(j));
    s1 += std::abs(m.r(j)) * m.W(j);
  }
  double lo = lo_floor, hi = hi_cap;
  if (N > 0 and s0 > 0.0) {
    double c = 0.0, nlo = m.W(o[N - 1]);
    for (long j = 0; j < N; ++j) {
      c += std::abs(m.r(o[j]));
      if (c >= 0.01 * s0) { nlo = m.W(o[j]); break; }
    }
    c = 0.0;
    double nhi = m.W(o[N - 1]);
    for (long j = 0; j < N; ++j) {
      c += std::abs(m.r(o[j])) * m.W(o[j]);
      if (c >= 0.95 * s1) { nhi = m.W(o[j]); break; }
    }
    lo = std::max(lo_floor, nlo);
    hi = std::min(hi_cap, nhi);
  }
  if (hi <= 1.05 * lo or n <= 1) return {lo};
  return log_scales(lo, hi, n);
}

/// cos^2 partition of unity in log x peaked at the (sorted) centres (cd_lib.partition_weights), (ns, nx)
inline nda::array<double, 2> partition_weights(nda::array<double, 1> const &x, std::vector<double> const &centres) {
  const long ns = centres.size(), nx = x.size();
  nda::array<double, 2> P(ns, nx);
  P() = 0.0;
  std::vector<double> lc(ns);
  for (long i = 0; i < ns; ++i) lc[i] = std::log(centres[i]);
  for (long k = 0; k < nx; ++k) {
    const double lx = std::log(std::max(x(k), 1e-300));
    if (ns == 1 or lx <= lc[0]) { P(0, k) = 1.0; continue; }
    if (lx >= lc[ns - 1]) { P(ns - 1, k) = 1.0; continue; }
    for (long i = 0; i + 1 < ns; ++i)
      if (lx >= lc[i] and lx < lc[i + 1]) {
        const double t = (lx - lc[i]) / (lc[i + 1] - lc[i]);
        P(i, k)        = std::cos(0.5 * std::numbers::pi * t) * std::cos(0.5 * std::numbers::pi * t);
        P(i + 1, k)    = std::sin(0.5 * std::numbers::pi * t) * std::sin(0.5 * std::numbers::pi * t);
        break;
      }
  }
  return P;
}

// ------------------------------------------------------------------------------------------------------------------
// MB closure
// ------------------------------------------------------------------------------------------------------------------
struct closure_opts_t {
  long nphi         = 16;      ///< coarse terminal-phase scan of the 1x1 upfolding (study: 16)
  double tol_c0     = 1e-14;   ///< as cd_lib.upfold_scalar
  double tol_svd    = 1e-12;
  bool drop_nonpositive = true;   ///< drop output poles d <= 0 (weight reported); false = keep them (python parity)
};

/// one scale: closure_B(nu, f, centre 0, w, K) of the study
struct scale_closure_t {
  double w = 0.0;
  long K = 0, r_gram = 0;
  nda::array<double, 1> d, a;   ///< poles (sorted) and weights |W|^2
  double residual = 0.0;        ///< held-out moment error of the upfolding
  long nneg = 0;                ///< poles at d <= 0 (dropped if drop_nonpositive)
  double wneg = 0.0;            ///< their weight / total weight
  double fsum() const {
    double s = 0.0;
    for (long l = 0; l < d.size(); ++l) s += 2.0 * a(l) * d(l);
    return s;
  }
};

inline scale_closure_t close_scale(nda::array<double, 1> const &nu, nda::array<double, 1> const &f, double w, long K,
                                   double tol_gram, closure_opts_t const &o = {}) {
  const long r = nu.size();
  nda::array<ComplexType, 3> g(r, 1, 1);
  for (long j = 0; j < r; ++j) g(j, 0, 0) = f(j);
  auto C = moments_from_poles(nu, g, w, K + 1);
  upfold_opts_t uo;
  uo.tol_c0   = o.tol_c0;
  uo.tol_gram = tol_gram;
  uo.tol_svd  = o.tol_svd;
  uo.nphi     = o.nphi;
  auto up     = upfold_block(C, K, w, uo);
  scale_closure_t s;
  s.w        = w;
  s.K        = K;
  s.r_gram   = up.r_gram;
  s.residual = up.residual;
  const long np = up.d.size();
  double wt = 0.0, wn = 0.0;
  long keep = 0;
  for (long l = 0; l < np; ++l) {
    const double al = std::norm(up.W(0, l));
    wt += al;
    if (up.d(l) <= 0.0) {
      wn += al;
      ++s.nneg;
    }
    if (up.d(l) > 0.0 or not o.drop_nonpositive) ++keep;
  }
  s.wneg = wt > 0.0 ? wn / wt : 0.0;
  s.d    = nda::array<double, 1>(keep);
  s.a    = nda::array<double, 1>(keep);
  for (long l = 0, k = 0; l < np; ++l)   // up.d is sorted ascending
    if (up.d(l) > 0.0 or not o.drop_nonpositive) {
      s.d(k) = up.d(l);
      s.a(k) = std::norm(up.W(0, l));
      ++k;
    }
  return s;
}

struct mb_closure_t {
  std::vector<double> scales;
  std::vector<scale_closure_t> parts;
  long K = 0;
  double tol_gram = 0.0;

  /// r_i(z) of scale i
  ComplexType part(long i, ComplexType z) const {
    auto const &p = parts[i];
    ComplexType s(0.0);
    for (long l = 0; l < p.d.size(); ++l) s += p.a(l) * odd_k(z, p.d(l));
    return s;
  }
  /// blended r(z) at the points z (partition in log |Re z|)
  nda::array<ComplexType, 1> operator()(nda::array<ComplexType, 1> const &z) const {
    const long nx = z.size();
    nda::array<double, 1> x(nx);
    for (long k = 0; k < nx; ++k) x(k) = std::abs(z(k).real());
    auto P = partition_weights(x, scales);
    nda::array<ComplexType, 1> o(nx);
    o() = ComplexType(0.0);
    for (long i = 0; i < long(parts.size()); ++i)
      for (long k = 0; k < nx; ++k)
        if (P(i, k) != 0.0) o(k) += P(i, k) * part(i, z(k));
    return o;
  }
  long npoles() const {
    long n = 0;
    for (auto const &p : parts) n += p.d.size();
    return n;
  }
  double max_wneg() const {
    double m = 0.0;
    for (auto const &p : parts) m = std::max(m, p.wneg);
    return m;
  }
};

/// MB: one closure per scale (centre 0), evaluated functions blended (run_designs.multiscale_blend)
inline mb_closure_t mb_close(nda::array<double, 1> const &nu, nda::array<double, 1> const &f, std::vector<double> const &scales,
                             long K, double tol_gram, closure_opts_t const &o = {}) {
  utils::check(not scales.empty(), "response::mb_close: no scales");
  for (size_t i = 1; i < scales.size(); ++i)
    utils::check(scales[i] > scales[i - 1], "response::mb_close: scales must be increasing");
  mb_closure_t m;
  m.scales   = scales;
  m.K        = K;
  m.tol_gram = tol_gram;
  for (double w : scales) m.parts.push_back(close_scale(nu, f, w, K, tol_gram, o));
  return m;
}

// ------------------------------------------------------------------------------------------------------------------
// optical quantities (optics.py::quantities + sigma)
// ------------------------------------------------------------------------------------------------------------------
struct optics_t {
  nda::array<double, 1> eps1, eps2, n, kappa, alpha, R, loss, sigma1, sigma2;   ///< alpha in 1/bohr, sigma in a.u.
  static std::vector<std::string> names() { return {"eps1", "eps2", "n", "kappa", "alpha", "R", "loss", "sigma1", "sigma2"}; }
  nda::array<double, 1> const &get(std::string const &q) const {
    if (q == "eps1") return eps1;
    if (q == "eps2") return eps2;
    if (q == "n") return n;
    if (q == "kappa") return kappa;
    if (q == "alpha") return alpha;
    if (q == "R") return R;
    if (q == "loss") return loss;
    if (q == "sigma1") return sigma1;
    utils::check(q == "sigma2", "optics_t::get: unknown quantity {}", q);
    return sigma2;
  }
};

/// eps(omega) on the real grid omega (Ha) -> the optical quantities (see the file header for the units)
inline optics_t optical_quantities(nda::array<ComplexType, 1> const &eps, nda::array<double, 1> const &omega) {
  const long nw = eps.size();
  utils::check(omega.size() == nw, "response::optical_quantities: {} eps vs {} omega", nw, omega.size());
  optics_t q;
  for (auto *a : {&q.eps1, &q.eps2, &q.n, &q.kappa, &q.alpha, &q.R, &q.loss, &q.sigma1, &q.sigma2}) *a = nda::array<double, 1>(nw);
  constexpr double fpi = 4.0 * std::numbers::pi;
  for (long i = 0; i < nw; ++i) {
    const ComplexType e = eps(i);
    ComplexType N       = std::sqrt(e);
    if (N.imag() < 0.0) N = -N;
    q.eps1(i)   = e.real();
    q.eps2(i)   = e.imag();
    q.n(i)      = N.real();
    q.kappa(i)  = N.imag();
    q.alpha(i)  = omega(i) * e.imag() / (N.real() * C_AU);
    q.R(i)      = std::norm((N - 1.0) / (N + 1.0));
    q.loss(i)   = -(1.0 / e).imag();
    q.sigma1(i) = omega(i) * e.imag() / fpi;
    q.sigma2(i) = -omega(i) * (e.real() - 1.0) / fpi;
  }
  return q;
}

} // namespace numerics::line_dlr::response

#endif
