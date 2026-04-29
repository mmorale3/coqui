/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_QP_UTILS_HPP
#define COQUI_REAL_AXIS_QP_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <complex>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"

#include "methods/GW_real_axis/real_freq_grid.hpp"

namespace methods {
namespace real_axis {

/**
 * Linear interpolation helpers on a strictly sorted, possibly non-uniform
 * grid. Used by the QP solver to evaluate Re Sigma^c (and its derivative)
 * at arbitrary omega values produced by root-finding / peak-search.
 *
 * Convention reminder: the real-axis pipeline stores Sigma_c on the grid
 * `state.grid->w()`, with the same omega convention as `dyson_G_one_kw`:
 * the absolute frequency at index iw is `grid.w()(iw) + grid.mu_chem()`.
 * Equivalently, `grid.w()` is the chemical-potential-relative frequency.
 *
 * The QP equation is `eps_QP = H_eff_nn + Re Sigma_c_nn(eps_QP)`. The
 * solver therefore converts `eps_QP` to the relative coordinate
 * `w_rel = eps_QP - mu_chem` before calling these helpers.
 */

/**
 * Locate the bracketing index for value `x` in a strictly sorted ascending
 * 1D grid. Returns the largest index i such that g(i) <= x (clamped to
 * [0, N-2]).
 */
inline long lower_bracket(nda::array<double, 1> const& g, double x)
{
  const long N = g.shape()[0];
  if (N <= 1) return 0;
  if (x <= g(0))     return 0;
  if (x >= g(N - 1)) return N - 2;
  // std::lower_bound finds first element >= x; we want the index of the
  // first element <= x, which is one less.
  auto it = std::lower_bound(g.begin(), g.end(), x);
  long i  = std::distance(g.begin(), it);
  if (i > 0 && g(i) > x) --i;
  if (i > N - 2) i = N - 2;
  if (i < 0)     i = 0;
  return i;
}

/**
 * Linear interpolation of a complex-valued function f sampled on g at
 * the target value x. Out-of-range x is clamped to the boundary and
 * extrapolation is NOT performed (constant continuation beyond the grid).
 */
inline ComplexType linear_interp_complex(nda::array<double, 1> const& g,
                                         nda::array<ComplexType, 1> const& f,
                                         double x)
{
  utils::check(g.shape()[0] == f.shape()[0],
               "linear_interp_complex: grid and value shape mismatch ({} vs {})",
               g.shape()[0], f.shape()[0]);
  const long N = g.shape()[0];
  if (x <= g(0))     return f(0);
  if (x >= g(N - 1)) return f(N - 1);
  const long i = lower_bracket(g, x);
  const double t = (x - g(i)) / (g(i + 1) - g(i));
  return (1.0 - t) * f(i) + t * f(i + 1);
}

/**
 * Linear interpolation of Re Sigma^c at orbital component (i, j) for a
 * given (s, k) on the real-w grid, evaluated at the relative frequency
 * `w_rel = omega_abs - mu_chem`. ReSigma is read from the supplied 5D
 * view; expected layout (N_w, ns, nkpts, nbnd, nbnd).
 *
 * Returns ComplexType because the caller often wants the full complex
 * Sigma^c stored in (Re + i*Im) form -- but this routine only reads the
 * real part of `ReSigma_wskij` and the imaginary part of `ImSigma_wskij`
 * if both are passed. For Re-only sampling (the common case) call with
 * a default-constructed empty `ImSigma` view (size 0), which the helper
 * detects and treats as zero.
 */
inline ComplexType
interp_Sigma_orbital_at_w(nda::array<double, 1> const& w_grid,
                          nda::ArrayOfRank<5> auto const& ReSigma_wskij,
                          nda::ArrayOfRank<5> auto const& ImSigma_wskij,
                          long s, long k, long i, long j,
                          double w_rel)
{
  const long N_w = w_grid.shape()[0];
  utils::check(ReSigma_wskij.shape()[0] == N_w,
               "interp_Sigma_orbital_at_w: ReSigma N_w mismatch");
  if (w_rel <= w_grid(0))   {
    const double re = ReSigma_wskij(0, s, k, i, j).real();
    const double im = ImSigma_wskij(0, s, k, i, j).real();
    return ComplexType(re, im);
  }
  if (w_rel >= w_grid(N_w - 1)) {
    const double re = ReSigma_wskij(N_w - 1, s, k, i, j).real();
    const double im = ImSigma_wskij(N_w - 1, s, k, i, j).real();
    return ComplexType(re, im);
  }
  const long iw = lower_bracket(w_grid, w_rel);
  const double t = (w_rel - w_grid(iw)) / (w_grid(iw + 1) - w_grid(iw));
  const double re = (1.0 - t) * ReSigma_wskij(iw,     s, k, i, j).real()
                  +        t  * ReSigma_wskij(iw + 1, s, k, i, j).real();
  const double im = (1.0 - t) * ImSigma_wskij(iw,     s, k, i, j).real()
                  +        t  * ImSigma_wskij(iw + 1, s, k, i, j).real();
  return ComplexType(re, im);
}

/**
 * Two-sided finite-difference derivative dRe Sigma_c / domega at relative
 * frequency `w_rel`, for orbital component (s, k, i, j). Uses the bracketing
 * grid points directly: in interval [w(iw), w(iw+1)], the slope is
 * (f(iw+1) - f(iw)) / (w(iw+1) - w(iw)). At grid boundaries, returns the
 * one-sided slope. Sufficient for the linearized QP equation Z-factor.
 */
inline double
interp_dReSigma_orbital_at_w(nda::array<double, 1> const& w_grid,
                             nda::ArrayOfRank<5> auto const& ReSigma_wskij,
                             long s, long k, long i, long j,
                             double w_rel)
{
  const long N_w = w_grid.shape()[0];
  if (N_w < 2) return 0.0;
  long iw = lower_bracket(w_grid, w_rel);
  if (iw < 0)         iw = 0;
  if (iw > N_w - 2)   iw = N_w - 2;
  const double dw = w_grid(iw + 1) - w_grid(iw);
  const double f0 = ReSigma_wskij(iw,     s, k, i, j).real();
  const double f1 = ReSigma_wskij(iw + 1, s, k, i, j).real();
  return (f1 - f0) / dw;
}

/**
 * Build the diagonal of Sigma_c (Re and Im) in the QP / MO basis at all
 * w grid points, for a single (s, k, n).
 *
 *     Sigma_c_nn(w) = sum_ij conj(MO(i, n)) * Sigma_c_ij(w) * MO(j, n)
 *
 * Output: complex 1D array of length N_w with .real() = Re Sigma_c_nn(w)
 * and .imag() = Im Sigma_c_nn(w). The caller passes flat 4D views into
 * Sigma_c (per (s, k) slice).
 *
 * Implemented as a single matrix-vector reduction per omega; cheap enough
 * to do per band on the fly for the QP equation.
 */
inline void
diag_Sigma_in_QP_basis(nda::ArrayOfRank<3> auto const& ReSigma_wij,
                       nda::ArrayOfRank<3> auto const& ImSigma_wij,
                       nda::ArrayOfRank<1> auto const& MO_n,
                       nda::array<ComplexType, 1>      & Sigma_w_out)
{
  const long N_w  = ReSigma_wij.shape()[0];
  const long nbnd = MO_n.shape()[0];
  utils::check(ReSigma_wij.shape()[1] == nbnd and ReSigma_wij.shape()[2] == nbnd,
               "diag_Sigma_in_QP_basis: ReSigma orbital dimension mismatch");
  utils::check(ImSigma_wij.shape() == ReSigma_wij.shape(),
               "diag_Sigma_in_QP_basis: ReSigma / ImSigma shape mismatch");
  utils::check(Sigma_w_out.shape()[0] == N_w,
               "diag_Sigma_in_QP_basis: output N_w mismatch");

  for (long iw = 0; iw < N_w; ++iw) {
    ComplexType acc(0.0, 0.0);
    for (long i = 0; i < nbnd; ++i) {
      const ComplexType ci = std::conj(MO_n(i));
      ComplexType row_acc(0.0, 0.0);
      for (long j = 0; j < nbnd; ++j) {
        const double re = ReSigma_wij(iw, i, j).real();
        const double im = ImSigma_wij(iw, i, j).real();
        row_acc += ComplexType(re, im) * MO_n(j);
      }
      acc += ci * row_acc;
    }
    Sigma_w_out(iw) = acc;
  }
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_QP_UTILS_HPP
