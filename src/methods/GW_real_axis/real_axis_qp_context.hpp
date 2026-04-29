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

#ifndef COQUI_REAL_AXIS_QP_CONTEXT_HPP
#define COQUI_REAL_AXIS_QP_CONTEXT_HPP

#include <string>

namespace methods {
namespace real_axis {

/**
 * Configuration for the real-axis quasiparticle solver.
 *
 * Mirrors methods::qp_context_t (imag-axis side) but drops the analytic-
 * continuation knobs (`ac_alg`, `Nfit`) -- on the real axis Re Sigma_c is
 * already on a real-frequency grid, so the QP equation reduces to direct
 * interpolation. Eta is retained as the small imaginary shift used by the
 * spectral / linearized algorithms.
 *
 * qp_type
 *     "bisection"  -- bracketed bisection on the real residual
 *                     omega - H_eff_nn - Re Sigma_c_nn(omega).
 *     "linearized" -- one-step Z-factor evaluation at omega = eps0.
 *     "secant"     -- Newton-secant iteration on the residual.
 *     "spectral"   -- argmax of |Im G^R| in a window around eps0.
 *
 * off_diag_mode (only used by compute_V_corr / QSGW path):
 *     "qp_energy"  -- Faleev "Mode A":
 *                     V_{ab} = 0.5 * [Re Sigma_c_{ab}(eps_a) + Re Sigma_c_{ab}(eps_b)]
 *                     followed by 0.5 * (V + V^dagger) hermitization.
 *     "fermi"      -- Diagonal at eps_a, off-diagonal at omega = mu_chem
 *                     (i.e., at the chemical potential).
 *
 * eta
 *     Imaginary shift used by `linearized` / `secant` / `spectral`. Must be
 *     small enough that the QP peak is resolved but large enough to keep
 *     numerical derivatives stable (default 1e-3).
 *
 * tol
 *     Convergence tolerance on the QP residual (bisection / secant) or on
 *     the peak-search step size (spectral).
 */
struct real_axis_qp_context_t {
  std::string qp_type        = "bisection";
  std::string off_diag_mode  = "qp_energy";
  double      eta            = 1e-3;
  double      tol            = 1e-8;
  long        secant_maxiter = 200;

  real_axis_qp_context_t() = default;
  real_axis_qp_context_t(std::string qpt, std::string odm,
                         double e = 1e-3, double t = 1e-8)
    : qp_type(std::move(qpt)), off_diag_mode(std::move(odm)),
      eta(e), tol(t) {}
};

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_QP_CONTEXT_HPP
