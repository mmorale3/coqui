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

#ifndef COQUI_REAL_AXIS_QP_A_HPP
#define COQUI_REAL_AXIS_QP_A_HPP

/**
 * Shared QP-pole inputs of the real-axis W chain (increment RW-2; promoted out of the RW-1
 * gate harness test_real_axis_w_lehmann.cpp, which now calls these).
 *
 *   (1) `size_grids(eta, w_max, Omega_max)` -- the DERIVED grid sizing of RW-1 section 3.
 *   (2) `build_A_from_QP_poles(...)`        -- the QP-pole Lorentzian spectral function.
 *
 * Both are pure functions of their arguments; nothing here allocates state or touches MPI.
 *
 * -------------------------------------------------------------------------------------
 * (1) GRID SIZING. Everything is derived from eta, not tuned (RW-1 report section 3,
 *     "Grid sizing (derived, not tuned)"):
 *
 *   * w grid   : A is a sum of Lorentzians of width eta. Trapezoid quadrature of a
 *                Lorentzian with spacing h converges as exp(-2 pi eta / h), so h <= eta/2
 *                buys e^{-4 pi} ~ 3e-6. Uniform over the whole [-w_max, w_max] window, since
 *                the QP poles are spread over the full bandwidth.
 *   * t window : both legs of the bubble decay as exp(-eta |t|), the product as
 *                exp(-2 eta |t|), so truncating at |t| = T/2 costs ~exp(-eta T);
 *                T = 9.2/eta gives ~1e-4.
 *   * dt       : real_freq_grid_t enforces max(|w|, Omega) dt <= pi as a HARD error; we take
 *                the branch's own safety factor 2, dt = 0.5 pi / freq_max.
 *   * Omega    : the features of Im W^c inherit the ~2 eta width of the bubble and the
 *                forward spectral integral uses the grid's own trapezoid weights, so
 *                dOmega <= eta (again exponential). NOTE the branch's own default
 *                (N_Omega = 64 on [0, 2 w_max]) does NOT satisfy this at production eta --
 *                that is a property of the criterion, not of the ported code.
 *
 * -------------------------------------------------------------------------------------
 * (2) THE SPECTRAL FUNCTION. `build_A_from_QP_poles` reproduces the branch recipe
 *     (real_axis_qp_scf_driver.hpp:249-278, which is NOT part of the RW-1 port):
 *
 *         A_{ij}(w; s, k) = sum_a C_{i a} * (1/pi) eta / ((w_abs - E_a)^2 + eta^2)
 *                                          * conj(C_{j a}),
 *         w_abs = grid.w()(iw) + grid.mu_chem()          [the grid's w is measured from mu]
 *
 *     with C = the MO coefficients and E_a the ABSOLUTE quasiparticle energies. Passing a
 *     null MO pointer selects the identity-MO fill, which is the RW-1 gate's construction
 *     (licensed there by convention pin P1: H0 + F = diag(eps_KS) and S = 1) and is written
 *     out separately so that path stays bit-for-bit what RW-1 measured.
 *
 *     The array is filled at IBZ k -- real_axis_scr_coulomb_t::update_w hard-checks
 *     A.shape()[2] == MF.nkpts_ibz() and does the FBZ expansion itself.
 */

#include <cmath>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"
#include "methods/GW_real_axis/real_freq_grid.hpp"

namespace methods {
namespace real_axis {

  /** Derived grid sizes for one eta. See the header comment. */
  struct grid_sizing_t {
    long   N_w = 0;
    long   N_Omega = 0;
    long   N_t = 0;
    double T_window = 0.0;
    double dw = 0.0;
    double dOmega = 0.0;
    double dt = 0.0;
  };

  inline grid_sizing_t size_grids(double eta, double w_max, double Omega_max) {
    utils::check(eta > 0.0, "real_axis::size_grids: eta = {} must be > 0.", eta);
    utils::check(w_max > 0.0 and Omega_max > 0.0,
                 "real_axis::size_grids: w_max = {}, Omega_max = {} must be > 0.",
                 w_max, Omega_max);
    grid_sizing_t g;
    const double freq_max = std::max(w_max, Omega_max);
    g.dt = 0.5 * M_PI / freq_max;
    const double T_target = 9.2 / eta;
    g.N_t = 2;
    while (static_cast<double>(g.N_t) * g.dt < T_target) g.N_t *= 2;
    g.T_window = g.dt * static_cast<double>(g.N_t);
    const double h_target = 0.5 * eta;
    g.N_w = static_cast<long>(std::ceil(2.0 * w_max / h_target)) + 1;
    if (g.N_w % 2 == 0) ++g.N_w;            // keep w = 0 on the grid
    g.dw = 2.0 * w_max / static_cast<double>(g.N_w - 1);
    g.N_Omega = static_cast<long>(std::ceil(Omega_max / eta));
    g.dOmega  = Omega_max / static_cast<double>(g.N_Omega);
    return g;
  }

  /**
   * Fill A(iw, s, k, i, j) with QP-pole Lorentzians of width eta.
   *
   * @param A     (N_w, ns, nkpts_ibz, nbnd, nbnd) -- OVERWRITTEN.
   * @param grid  supplies w() and mu_chem().
   * @param E     (ns, nkpts_ibz, nbnd) ABSOLUTE quasiparticle energies (real part used).
   * @param MO    (ns, nkpts_ibz, nbnd, nbnd) MO coefficients, column a = MO a; may be null,
   *              which selects the identity-MO (diagonal) fill.
   * @param eta   Lorentzian half width, a.u.
   */
  template<typename A_t, typename E_t, typename MO_t>
  void build_A_from_QP_poles(A_t &&A, real_freq_grid_t const &grid,
                             E_t const &E, MO_t const *MO, double eta) {
    const long N_w  = A.shape()[0];
    const long ns   = A.shape()[1];
    const long nk   = A.shape()[2];
    const long nbnd = A.shape()[3];
    utils::check(A.shape()[4] == nbnd,
                 "real_axis::build_A_from_QP_poles: A is not square in the orbital indices "
                 "({} x {}).", nbnd, A.shape()[4]);
    utils::check(grid.N_w() == N_w,
                 "real_axis::build_A_from_QP_poles: grid has N_w = {} but A has {}.",
                 grid.N_w(), N_w);
    utils::check(E.shape()[0] == ns and E.shape()[1] == nk and E.shape()[2] == nbnd,
                 "real_axis::build_A_from_QP_poles: E has shape ({}, {}, {}), A wants "
                 "({}, {}, {}).", E.shape()[0], E.shape()[1], E.shape()[2], ns, nk, nbnd);
    utils::check(eta > 0.0, "real_axis::build_A_from_QP_poles: eta = {} must be > 0.", eta);

    A = ComplexType(0.0, 0.0);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < nk; ++k) {
        if (MO == nullptr) {
          // identity MO: the RW-1 gate construction, kept textually separate so that path is
          // bit-for-bit what the RW-1 eta series measured.
          for (long n = 0; n < nbnd; ++n) {
            const double e_n = std::real(ComplexType(E(s, k, n)));
            for (long iw = 0; iw < N_w; ++iw) {
              const double w_abs = grid.w()(iw) + grid.mu_chem();
              A(iw, s, k, n, n) = ComplexType(
                  (1.0 / M_PI) * eta
                      / ((w_abs - e_n) * (w_abs - e_n) + eta * eta), 0.0);
            }
          }
        } else {
          for (long n = 0; n < nbnd; ++n) {
            const double e_n = std::real(ComplexType(E(s, k, n)));
            for (long iw = 0; iw < N_w; ++iw) {
              const double w_abs = grid.w()(iw) + grid.mu_chem();
              const double L = (1.0 / M_PI) * eta
                             / ((w_abs - e_n) * (w_abs - e_n) + eta * eta);
              for (long i = 0; i < nbnd; ++i) {
                const ComplexType ci = ComplexType((*MO)(s, k, i, n)) * L;
                if (ci == ComplexType(0.0, 0.0)) continue;
                for (long j = 0; j < nbnd; ++j)
                  A(iw, s, k, i, j) += ci * std::conj(ComplexType((*MO)(s, k, j, n)));
              }
            }
          }
        }
      }
  }

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_QP_A_HPP
