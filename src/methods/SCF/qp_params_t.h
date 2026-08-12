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


#ifndef COQUI_QP_CONTEXT_H
#define COQUI_QP_CONTEXT_H

namespace methods {

struct qp_params_t {
  std::string qp_type = "sc";
  std::string ac_alg = "pade";
  int Nfit = 18;
  double eta = 0.0001;
  double tol = 1e-8;

  // SCF mode selector:
  // - evscf: update only QP energies and keep QP wavefunctions fixed to mean-field ones.
  // - qpscf: update both QP energies and QP wavefunctions.
  std::string qp_scf_mode = "qpscf";

  // whether to update dynamically screened interaction W in evscf.
  bool keep_scr_coulomb_fixed = false;

  // off-diagonal mode defined in T. Kotani et. al., Phys. Rev. B 76, 165106 (2007)
  // "fermi": evaluate off-diagonal elements of self-energy at the Fermi level;
  // "qp_energy": evaluate off-diagonal elements of self-energy at the quasiparticle energy
  // (defined as the average of the two diagonal elements)
  std::string off_diag_mode = "fermi";

  double mu_tolerance = 1e-9;
  std::string mu_update_alg = "bisection";

  // Project 2 increment Q0 (notes/qpgw_edmft_implementation_plan.md): the
  // quasiparticle-map selector.
  // - "ac_pade":     today's route -- Pade AC of Sigma(iw) evaluated near the
  //                  real axis (solve_qp_eqn / qp_approx unchanged). DEFAULT.
  // - "mats_lin":    Matsubara-native omega~0 linearization (spec eq 13;
  //                  qp_maps_matsubara.hpp map (i)) -- no analytic continuation.
  // - "mats_gmatch": Matsubara-native variational Green's-function matching
  //                  (spec eq 14; map (ii)) -- no analytic continuation.
  // Wired into the solvers at increment Q2; parsed and validated from Q0 so the
  // default path is pinned bitwise before any dispatch lands.
  std::string qp_map = "ac_pade";
};

} // methods

#endif //COQUI_QP_CONTEXT_H
