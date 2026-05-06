/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_REAL_FREQ_GRID_HPP
#define COQUI_REAL_AXIS_REAL_FREQ_GRID_HPP

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"

namespace methods {
namespace real_axis {

/**
 * Real-frequency grid for finite-temperature real-axis GW.
 *
 * Holds three independent grids:
 *   - Fermionic frequencies w_j  (non-uniform, full window [-w_max, w_max])
 *   - Bosonic frequencies Omega_l (non-uniform, half window [Omega_min, Omega_max],
 *                                  exploits Im W^R odd in Omega)
 *   - Conjugate uniform "time" grid t_k = (k - N_t/2) * dt,  k=0..N_t-1
 *     where dt = T_window / N_t. The time grid is the convergence parameter
 *     for the NUFFT-based convolutions and Hilbert transforms.
 *
 * Finite-temperature parameters beta and mu_chem are first-class members and
 * are used by the f(w) and n_B(Omega) kernels. The Bose function n_B is
 * evaluated through a numerically stable expression to avoid catastrophic
 * cancellation for small beta*Omega; the grid constructor refuses Omega=0
 * exactly to prevent the n_B singularity.
 *
 * No fermionic half-grid reduction is implemented: at finite temperature the
 * Fermi factor breaks the would-be reflection symmetry around mu_chem.
 */
class real_freq_grid_t {
public:

  /// Direct construction. Caller supplies the physical grids; this constructor
  /// validates them and computes trapezoidal quadrature weights.
  /// @param beta      inverse temperature (1/Hartree)
  /// @param mu_chem   chemical potential (Hartree)
  /// @param w_grid    fermionic frequency grid, sorted, full window
  /// @param Omega_grid bosonic frequency grid, sorted, ALL ENTRIES > 0
  /// @param N_t       number of points on the conjugate uniform time grid
  /// @param T_window  total length of the time window
  real_freq_grid_t(double beta,
                   double mu_chem,
                   nda::array<double,1> w_grid,
                   nda::array<double,1> Omega_grid,
                   long N_t,
                   double T_window)
    : _beta(beta)
    , _mu_chem(mu_chem)
    , _w(std::move(w_grid))
    , _Omega(std::move(Omega_grid))
    , _N_t(N_t)
    , _T_window(T_window)
  {
    utils::check(_beta > 0.0,
                 "real_freq_grid_t: beta must be > 0 (got {})", _beta);
    utils::check(_w.shape()[0] >= 2,
                 "real_freq_grid_t: fermionic grid must have at least 2 points");
    utils::check(_Omega.shape()[0] >= 2,
                 "real_freq_grid_t: bosonic grid must have at least 2 points");
    utils::check(_N_t >= 4 and (_N_t % 2 == 0),
                 "real_freq_grid_t: N_t must be an even integer >= 4 (got {})", _N_t);
    utils::check(_T_window > 0.0,
                 "real_freq_grid_t: T_window must be > 0 (got {})", _T_window);

    check_sorted_strictly(_w, "fermionic frequency");
    check_sorted_strictly(_Omega, "bosonic frequency");
    utils::check(_Omega(0) > 0.0,
                 "real_freq_grid_t: bosonic grid must satisfy Omega>0 everywhere "
                 "(first point = {} <= 0). The Bose function diverges at Omega=0; "
                 "exclude Omega=0 from the grid by construction.", _Omega(0));

    _w_weights = make_trapezoid_weights(_w);
    _Omega_weights = make_trapezoid_weights(_Omega);

    // Time grid: t_k = (k - N_t/2) * dt, dt = T_window / N_t.
    // Range: [-T/2, T/2 - dt]. t=0 occurs exactly at index k = N_t/2.
    _t = nda::array<double,1>(_N_t);
    const double dt = _T_window / static_cast<double>(_N_t);
    for (long k = 0; k < _N_t; ++k)
      _t(k) = (static_cast<double>(k) - 0.5 * static_cast<double>(_N_t)) * dt;

    // Sanity: the time-window choice must resolve the maximum frequency,
    // i.e. dt < pi / w_max so that the rescaled coordinate w_j * dt is in
    // [-pi, pi]. Issue an error rather than warn, since a violated condition
    // produces silent aliasing in the NUFFT.
    const double w_max = std::max(std::abs(_w(0)), std::abs(_w(_w.shape()[0]-1)));
    const double Omega_max = _Omega(_Omega.shape()[0]-1);
    const double freq_max = std::max(w_max, Omega_max);
    utils::check(freq_max * dt <= M_PI + 1e-12,
                 "real_freq_grid_t: Nyquist condition violated. "
                 "max(|w|,Omega) * dt = {} > pi. "
                 "Reduce T_window or increase N_t (current dt = {}, freq_max = {}).",
                 freq_max * dt, dt, freq_max);
  }

  // -------------------------------------------------------------------
  // Factories
  // -------------------------------------------------------------------

  /// Uniform fermionic grid on [-w_max, w_max] with N_w points (excluding
  /// 0 if N_w is even; otherwise mu_chem is shifted off-grid by the caller).
  /// Bosonic grid is uniform on [dOmega, Omega_max] with N_Omega points,
  /// dOmega = Omega_max / N_Omega so that Omega=0 is excluded.
  static real_freq_grid_t make_uniform(double beta,
                                       double mu_chem,
                                       double w_max,
                                       long   N_w,
                                       double Omega_max,
                                       long   N_Omega,
                                       long   N_t,
                                       double T_window)
  {
    utils::check(w_max > 0.0,    "make_uniform: w_max must be > 0");
    utils::check(N_w >= 2,       "make_uniform: N_w must be >= 2");
    utils::check(Omega_max > 0.0,"make_uniform: Omega_max must be > 0");
    utils::check(N_Omega >= 2,   "make_uniform: N_Omega must be >= 2");

    nda::array<double,1> w(N_w);
    {
      const double h = 2.0 * w_max / static_cast<double>(N_w - 1);
      for (long j = 0; j < N_w; ++j)
        w(j) = -w_max + h * static_cast<double>(j);
    }

    nda::array<double,1> Omega(N_Omega);
    {
      const double h = Omega_max / static_cast<double>(N_Omega);
      for (long l = 0; l < N_Omega; ++l)
        Omega(l) = h * static_cast<double>(l + 1);
    }

    return real_freq_grid_t(beta, mu_chem,
                            std::move(w), std::move(Omega),
                            N_t, T_window);
  }

  // Non-uniform fermionic grid. Linear-dense block of `N_dense` points
  // covering [-w_dense, +w_dense] (chemical-potential-relative, so the
  // dense region surrounds mu_chem in absolute coordinates), with the
  // remaining `N_w - N_dense` points log-spaced into the two tails out
  // to ±w_max. The Bosonic grid stays uniform on [dOmega, Omega_max]
  // (Omega is absolute, not mu-relative; n_B's structure is around 0
  // not mu).
  //
  // Use case: dense sampling of the QP region and the f(w)*A integrand
  // near mu, sparse in the deep valence / high-conduction tails where
  // A is essentially zero.
  static real_freq_grid_t make_nonuniform_log(double beta,
                                              double mu_chem,
                                              double w_max,
                                              long   N_w,
                                              double w_dense,
                                              long   N_dense,
                                              double Omega_max,
                                              long   N_Omega,
                                              long   N_t,
                                              double T_window)
  {
    utils::check(w_max > 0.0,    "make_nonuniform_log: w_max must be > 0");
    utils::check(w_dense > 0.0,
                 "make_nonuniform_log: w_dense must be > 0 (got {})", w_dense);
    utils::check(w_dense < w_max,
                 "make_nonuniform_log: w_dense ({}) must be < w_max ({})",
                 w_dense, w_max);
    utils::check(N_w >= 4,       "make_nonuniform_log: N_w must be >= 4");
    utils::check(N_dense >= 3,
                 "make_nonuniform_log: N_dense must be >= 3 (got {})", N_dense);
    utils::check(N_dense + 2 <= N_w,
                 "make_nonuniform_log: N_dense ({}) leaves no room for tails "
                 "in N_w ({}); need N_dense + 2 <= N_w.", N_dense, N_w);
    utils::check((N_w - N_dense) % 2 == 0,
                 "make_nonuniform_log: N_w - N_dense must be even so tails "
                 "are symmetric (got N_w={}, N_dense={}, diff={}).",
                 N_w, N_dense, N_w - N_dense);

    const long n_tail = (N_w - N_dense) / 2;

    // Dense block: uniform on [-w_dense, +w_dense] with N_dense points.
    const double h_dense = 2.0 * w_dense / static_cast<double>(N_dense - 1);

    // Log-spaced tails: span [w_dense + h_dense, w_max] with n_tail points.
    // Use the dense-edge spacing as the inner anchor so the grid spacing is
    // monotone non-decreasing as |w| grows past w_dense.
    const double w_tail_inner = w_dense + h_dense;
    utils::check(w_tail_inner < w_max,
                 "make_nonuniform_log: dense block already reaches w_max "
                 "(w_dense + h_dense = {} >= w_max = {}); reduce N_dense or "
                 "w_dense, or increase w_max.", w_tail_inner, w_max);
    const double log_step = (std::log(w_max) - std::log(w_tail_inner))
                           / static_cast<double>(n_tail - 1);

    nda::array<double,1> w(N_w);
    // Negative tail (descending magnitude in w; build then mirror).
    for (long i = 0; i < n_tail; ++i) {
      const double mag = std::exp(std::log(w_tail_inner)
                                  + static_cast<double>(n_tail - 1 - i) * log_step);
      w(i) = -mag;
    }
    // Dense block.
    for (long j = 0; j < N_dense; ++j)
      w(n_tail + j) = -w_dense + h_dense * static_cast<double>(j);
    // Positive tail (ascending magnitude).
    for (long i = 0; i < n_tail; ++i) {
      const double mag = std::exp(std::log(w_tail_inner)
                                  + static_cast<double>(i) * log_step);
      w(n_tail + N_dense + i) = mag;
    }

    nda::array<double,1> Omega(N_Omega);
    {
      const double h = Omega_max / static_cast<double>(N_Omega);
      for (long l = 0; l < N_Omega; ++l)
        Omega(l) = h * static_cast<double>(l + 1);
    }

    return real_freq_grid_t(beta, mu_chem,
                            std::move(w), std::move(Omega),
                            N_t, T_window);
  }

  // -------------------------------------------------------------------
  // Accessors
  // -------------------------------------------------------------------

  double beta()    const { return _beta; }
  double mu_chem() const { return _mu_chem; }
  long   N_w()     const { return _w.shape()[0]; }
  long   N_Omega() const { return _Omega.shape()[0]; }
  long   N_t()     const { return _N_t; }
  double T_window() const { return _T_window; }
  double dt()      const { return _T_window / static_cast<double>(_N_t); }

  nda::array<double,1> const& w()             const { return _w; }
  nda::array<double,1> const& Omega()         const { return _Omega; }
  nda::array<double,1> const& t()             const { return _t; }
  nda::array<double,1> const& w_weights()     const { return _w_weights; }
  nda::array<double,1> const& Omega_weights() const { return _Omega_weights; }

  // -------------------------------------------------------------------
  // Numerically stable finite-T kernels
  //
  // fermi(w):   1 / (exp(beta*(w-mu)) + 1), evaluated to avoid overflow.
  // bose(Omega): 1 / (exp(beta*Omega) - 1), evaluated via expm1 for stability.
  //              Caller is responsible for not passing Omega=0 exactly.
  // -------------------------------------------------------------------

  static inline double fermi(double w, double mu_chem, double beta) {
    const double x = beta * (w - mu_chem);
    if (x >= 0.0) {
      const double e = std::exp(-x);
      return e / (1.0 + e);
    } else {
      return 1.0 / (1.0 + std::exp(x));
    }
  }

  /// 1 - fermi(w).
  static inline double fermi_bar(double w, double mu_chem, double beta) {
    const double x = beta * (w - mu_chem);
    if (x >= 0.0) {
      return 1.0 / (1.0 + std::exp(-x));
    } else {
      const double e = std::exp(x);
      return e / (1.0 + e);
    }
  }

  static inline double bose(double Omega, double beta) {
    // 1/(exp(beta*Omega) - 1) = 1/expm1(beta*Omega), stable for small Omega
    // up to the floating point scale of beta*Omega itself. Caller must avoid
    // Omega=0 exactly; we emit a NaN as a tripwire if it occurs.
    const double y = beta * Omega;
    if (y == 0.0) return std::nan("bose-Omega-zero");
    return 1.0 / std::expm1(y);
  }

  inline double fermi(double w)        const { return fermi(w, _mu_chem, _beta); }
  inline double fermi_bar(double w)    const { return fermi_bar(w, _mu_chem, _beta); }
  inline double bose(double Omega)     const { return bose(Omega, _beta); }

private:

  static void check_sorted_strictly(nda::array<double,1> const& g,
                                    std::string const& label) {
    for (long i = 1; i < g.shape()[0]; ++i)
      utils::check(g(i) > g(i-1),
                   "real_freq_grid_t: {} grid must be strictly sorted "
                   "ascending (failed at i={}).", label, i);
  }

  /// Trapezoidal quadrature weights on a non-uniform 1-D grid.
  static nda::array<double,1> make_trapezoid_weights(nda::array<double,1> const& g) {
    const long N = g.shape()[0];
    nda::array<double,1> w(N);
    if (N == 1) { w(0) = 0.0; return w; }
    w(0)   = 0.5 * (g(1) - g(0));
    w(N-1) = 0.5 * (g(N-1) - g(N-2));
    for (long i = 1; i < N - 1; ++i)
      w(i) = 0.5 * (g(i+1) - g(i-1));
    return w;
  }

  double               _beta;
  double               _mu_chem;
  nda::array<double,1> _w;
  nda::array<double,1> _Omega;
  long                 _N_t;
  double               _T_window;
  nda::array<double,1> _t;
  nda::array<double,1> _w_weights;
  nda::array<double,1> _Omega_weights;
};

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_REAL_FREQ_GRID_HPP
