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

  // mats_gmatch weight exponent: w_n = (w0/w_n)^qp_map_wpow on the positive
  // fermionic nodes. 2.0 emphasizes omega -> 0 (Z-weighted, Kutepov-like
  // behavior); 0.0 weights all nodes equally, exposing the QP-pole window
  // omega ~ |eps - mu| (closer to the real-axis Sigma(eps_QP) map). The
  // residual's leading 1/(i omega) tails cancel, so any wpow >= 0 is
  // well-posed. Scanned against the real-axis qsGW references (Q2-c).
  double qp_map_wpow = 2.0;

  // ---- Project 2 increment QM3: qp_map = "mode_a" (notes/qm3_mode_a_loop_spec.md) ----
  // NOTE ON ORDER: qp_params_t is an aggregate and the drivers use parenthesized aggregate
  // initialization positionally (MBPT_drivers.cpp), so new members must be APPENDED.
  //
  // Sigma^c evaluator for mode_a:
  // - "cd":        the QM2 contour-deformation closed form (sigma_route_b::sigma_cd) fed by
  //                the state-resolved W^c band elements. PRODUCTION default.
  // - "expansion": solve the same map with the route-A z0 = 0 re-expansion of the stored
  //                Sigma(iw) only (sigma_real_axis). Pure diagnostic -- needs no W data.
  std::string qp_modea_route = "cd";
  // inner QP-consistency cap at FIXED Sigma data (spec section 4). Non-convergence at the
  // cap is a physical multi-solution flag (app_warning), never a hard error.
  long qp_modea_nconsist = 5;
  // inner-consistency tolerance on max_a |d eps|, a.u.
  double qp_modea_consist_tol = 1e-8;
  // evaluation offset i*eta for the real-axis evaluation. Default 0.0; the QM3-c judge
  // protocol runs eta = 0. Stress knob only.
  double qp_modea_eta = 0.0;
  // W^c support constraint (BINDING, QM2 measurement): auxiliary pole nodes inside the
  // particle-hole gap are dropped from the fit -- prior physical information, not
  // regularization. "auto" takes the indirect gap of the CURRENT QP spectrum; "off"
  // disables it (the plain fit is O(1e4) wrong at real z -- do not use in production);
  // any other value is parsed as an explicit gap edge in a.u.
  std::string qp_modea_wsupp = "auto";
  // pole-fit route: "tau" = the QM2-b tested chain (bosonic mesh -> Ttw_bb -> fermionic tau
  // kernel -> residues x tanh(hw/2)); "nu" = the support-constrained LS directly on the
  // bosonic Matsubara nodes. Both are MEASURED by gate QM3-b.
  std::string qp_modea_wfit = "tau";
  // truncated-SVD cut of the SUPPORT-CONSTRAINED W^c pole fit, relative to the largest
  // singular value. Negative selects the shared doctrine value, imag_axes_ft::
  // dlr_pole_fit_rel_tol = 1e-8, which is the DEFAULT and reproduces the QM2-b chain.
  //
  // WHY IT IS EXPOSED (measured, QM3-b on lih222): 1e-8 maximizes IMAGINARY-axis accuracy and
  // is the right doctrine for the residue algebras of project 1, which never leave that axis.
  // Route B does: it evaluates the same rational function at REAL quasiparticle energies,
  // where a near-interpolatory fit is a wild function with thousands of poles and residues
  // 1e2-1e4 times the data it represents. Measured on lih222/q=0 (bosonic-mesh reconstruction,
  // residue ratio): tau route 1e-8 -> (4.4e-3, 6.0e3), 1e-6 -> (1.4e-2, 1.2e2),
  // 1e-4 -> (5.0e-2, 9.2); nu route 1e-8 -> (1.2e-3, 4.3e2), 1e-6 -> (3.0e-3, 1.2e1),
  // 1e-4 -> (8.5e-3, 7.7e-1). The knob is the imaginary-accuracy vs real-axis-smoothness
  // trade-off; the production value is a spec decision, NOT an agent default.
  double qp_modea_wrtol = -1.0;

  // W^c RESIDUE-SLAB COMPRESSION (stage 1b of wc_band_elements.hpp). Relative eigenvalue cut
  // on each Hermitian residue slab W^(p)_PQ: the mode-A sandwich then costs r/Np of the dense
  // Np^2 one, which is what makes the production (nbnd, Np) reachable -- see the flop model in
  // that file's header. <= 0 disables the factorization and takes the dense reference path.
  // The default is a numerical-noise cut, three orders below the W-fit reconstruction error
  // that bounds the whole map's accuracy; the achieved rank AND the truncation residual of
  // every (q,p) slab are logged, so this is never a silent accuracy change.
  double qp_modea_wrank = 1e-10;
  // factorization backend: 0 = automatic (LAPACK heev up to Np = 600, randomized Nystrom
  // sketching above it), > 0 = force the sketch with that initial block size, < 0 = force
  // heev. Exposed because the sketch is the production path but only the small fixtures can
  // cross-check it against the exact one.
  long qp_modea_wsketch = 0;

  // ---- spec rev 4 (2026-08-13): GRADED-eta FAR-STATE EVALUATION ----
  // Imaginary offset, in a.u., applied to the evaluation energies of states OUTSIDE the
  // analyticity strip (VBM - 0.95 E_PH, CBM + 0.95 E_PH):
  //
  //     eta_far  = 0   -> out-of-strip states are evaluated at z = mu   (rev 3.1, THE DEFAULT)
  //     eta_far  > 0   -> out-of-strip states are evaluated at z = eps + i eta_far
  //
  // In-strip states are unaffected and stay exact (eta -> 0). Applies to mode_a (both indices
  // of 1/2[Sigma(eps_a) + Sigma(eps_b)]) and to the mode_b diagonal.
  //
  // WHY (QM3-c judge verdict, kp222, matched heads): mode_a == mode_b = 3.714/1.219 eV against
  // a real-axis reference series of 3.10-3.37/0.70-0.94 eV with a tau oracle of 2.6e-08, i.e.
  // the whole 0.35/0.45 eV offset is the mu fallback of the 471 out-of-strip evaluations. The
  // reference's own far-state object is Re Sigma(eps + i eta), so ours must be too.
  //
  // VALIDITY FLOOR (logged, warned, never fatal): eta_far must exceed ~3x the local fitted-pole
  // spacing (~1e-3 a.u. at kp222) or the evaluation rides single poles of the fit instead of
  // the eta-smoothed spectral density. The measured spacing is reported every outer iteration.
  double qp_modea_eta_far = 0.0;
};

} // methods

#endif //COQUI_QP_CONTEXT_H
