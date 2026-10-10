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

#ifndef COQUI_METHODS_GW_LINE_THERMAL_MU_HPP
#define COQUI_METHODS_GW_LINE_THERMAL_MU_HPP

/**
 * Finite temperature (S8b, notes section 11): the parameters (thermal_params_t, [gw_line] keys of driver.hpp) and the
 * chemical-potential rule (notes section 11.6, Eq. fT_nth; python line/closure.py::chemical_potential_auto, reference
 * dev/s8b_mu_rule.py). Light header (cayley.hpp only): used by closure.hpp, spectra and thermal.hpp.
 *   mu_rule "gap"    : the widest admissible gap midpoint mu_g (the T = 0 rule);
 *           "number" : N(mu) = N_el with the Fermi function (Eq. fT_mu, bisection to the root);
 *           "auto"   : mu_g iff |dN| <= mu_dn_max and |dN| > mu_th_factor n_th(mu_g) (dN = N_T(mu_g) - N_el, n_th = the thermal
 *                      carriers across mu_g), else "number". With NO pole within E_T of mu_g (empty window: the T = 0 path,
 *                      notes section 11.1) the midpoint is kept ("gap(T=0)"), so that a run whose windows stay empty is
 *                      bitwise the T = 0 run (gate T4).
 */

#include <cmath>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"
#include "numerics/line_dlr/cayley.hpp"

namespace methods::gw_line {

/// Finite-temperature parameters ([gw_line] keys, driver.hpp) and the derived scales.
struct thermal_params_t {
  double beta        = 0.0;     ///< 0: T = 0
  double thermal_tol = 1e-8;    ///< tau_T
  double c_zeta      = 30.0;    ///< thermal_floor (bosonic node floor rho beta |zeta| >= c_zeta, band bottom)
  double c_f         = 30.0;    ///< thermal_floor_f (fermionic node floor)
  double theta = 0.0, theta_t = 0.0;
  double wp_floor    = 15.0;    ///< omega_p >= wp_floor zeta_T
  std::string mu_rule = "auto";
  double mu_dn_max = 0.1, mu_th_factor = 10.0;
  long band_heights = 8, band_x = 21;
  double band_c = -1.0;         ///< <= 0: c_zeta
  double band_top = 4.0, mats_factor = 4.0;
  double eps_b = 1e-12, lam_b = 0.0, bos_eps_T = 1e-10;   ///< D-selected basis tolerance, candidate range, line basis eps
  long npole_b = 800;
  double cut_odd = 1e-13, cut_even = 1e-10, deg_tol = 1e-8;
  long tau_nn = 12;
  double tau_per_efold = 2.0, tau_x0 = 0.02;
  std::string tau_grid = "gl";  ///< "gl" (composite GL, python) | "id" (finite-interval time ID at theta_t = pi / 2)
  double tau_eps = 1e-12;       ///< tau ID tolerance
  bool mirror_D = true;         ///< D in the mirror layout (when the line nodes allow it)

  bool on() const { return beta > 0.0; }
  double c_T() const { return std::log(1.0 / thermal_tol); }
  double E_T() const { return c_T() / beta; }
  double rho() const { return std::sin(theta - theta_t) / std::sin(theta_t); }
  double zeta_T() const { return c_zeta / (rho() * beta); }
  double zeta_Tf() const { return c_f / (rho() * beta); }
  double S_T() const { return beta / std::sin(theta_t); }
};

/// result of the chemical-potential rule
struct mu_rule_out_t {
  double mu = 0.0;          ///< shift (same reference as e)
  std::string rule = "gap"; ///< rule used: "gap" | "number" | "gap(T=0)" (empty window at mu_g)
  double dN = 0.0, n_th = 0.0, N = 0.0;   ///< N_T(mu_g) - N_el, thermal carriers at mu_g, N_T at the chosen mu
  numerics::line_dlr::chemical_potential_t gap;   ///< the widest admissible gap
  bool window = false;      ///< a pole within E_T of the chosen mu
};

/// mu_rule (file header) on Lehmann (e, v) per k; window_at: any pole within E_T of mu_g (else T = 0: the midpoint)
inline mu_rule_out_t mu_rule_apply(std::vector<nda::array<double, 1>> const &e, std::vector<nda::array<ComplexType, 2>> const &v,
                                   double nelec, std::vector<double> const &k_weight, thermal_params_t const &tp,
                                   double dropped = 0.0) {
  using namespace numerics::line_dlr;
  mu_rule_out_t r;
  r.gap = chemical_potential(e, v, nelec, k_weight);
  const double ET = tp.E_T();
  auto win_at = [&](double mu) {
    for (auto const &ek : e)
      for (long m = 0; m < ek.size(); ++m)
        if (std::abs(ek(m) - mu) <= ET) return true;
    return false;
  };
  r.dN   = electron_count_T(e, v, tp.beta, r.gap.mu, k_weight, dropped) - nelec;
  r.n_th = thermal_carriers(e, v, tp.beta, r.gap.mu, k_weight);
  bool use_gap = false;
  if (tp.mu_rule == "gap")
    use_gap = true, r.rule = "gap";
  else if (tp.mu_rule == "number")
    use_gap = false;
  else {
    utils::check(tp.mu_rule == "auto", "gw_line: mu_rule must be \"auto\", \"gap\" or \"number\" (got \"{}\")", tp.mu_rule);
    if (not win_at(r.gap.mu)) {
      use_gap = true;
      r.rule  = "gap(T=0)";
    } else if (std::abs(r.dN) <= tp.mu_dn_max and std::abs(r.dN) > tp.mu_th_factor * r.n_th) {
      use_gap = true;
      r.rule  = "gap";
    }
  }
  if (use_gap) {
    r.mu = r.gap.mu;
    r.N  = r.dN + nelec;
  } else {
    auto [mu, N] = chemical_potential_T(e, v, nelec, tp.beta, k_weight, dropped);
    r.mu         = mu;
    r.N          = N;
    r.rule       = "number";
  }
  r.window = win_at(r.mu);
  return r;
}

} // namespace methods::gw_line

#endif
