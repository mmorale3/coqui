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
  // RW-2 (notes/rw_real_axis_w_spec.md): qp_modea_wfit = "spectral" replaces the least-
  // squares pole fit above by a SIGN-DEFINITE quadrature of the computed Im W^c(Omega, q)
  // from the real-axis module (needs -DENABLE_FINUFFT=ON). Motivation: on a metal the LS
  // residues are mixed sign and the contracted Sigma^c needs 4-5.5 digits of cancellation
  // (notes/qpgw_metal_mode_m0.md section 8b). The two knobs below are FLAGGED defaults:
  //   spectral_eta   -- Lorentzian width of the QP-pole A(w), a.u. Drives the whole grid
  //                     stack; smaller is more accurate and quadratically more expensive.
  //   spectral_npole -- target positive-Omega node count after coarsening (pole count is
  //                     2x this plus the head sector). <= 0 keeps every Omega node.
  //   spectral_gamma -- "spectral" (default; FABLE RULING 2026-08-21 on the section-4.5d
  //                     A/B, flipping the earlier protective ruling now that the data is
  //                     in) takes the Gamma BODY from the quadrature and leaves only the
  //                     scalar head on LS poles; "ls" keeps the whole q = Gamma column on
  //                     an appended support-constrained LS pole set. Measured on SVO
  //                     (notes/rw2_report.md section 4.5d): "ls" reintroduces the
  //                     mixed-sign LS cancellation on the ONE transfer that dominates
  //                     Sabs (100% head share, 49.97% negative numerators, transient
  //                     U(0) = -15 eV at the first map) while "spectral" passes every
  //                     RW-2-c gate (Sabs/|Sigma| 2.4-3.6 at map 2, neg share 0.11%,
  //                     U(0) positive both maps, causality at machine zero). The Gamma
  //                     hole "ls" guarded against does not exist on this path -- the
  //                     Gamma body is COMPUTED on the real axis (the section-3.3 fix),
  //                     not dropped. Known open item: the "spectral" mode's scalar-head
  //                     LS fit is unconstrained (6 sub-3pi/beta poles on SVO
  //                     contaminating eta=0 PROBE rows only, not the map) -- the support
  //                     constraint there is the filed follow-up, report section 7 item 1.
  std::string qp_modea_spectral_gamma = "spectral";
  double qp_modea_spectral_eta = 0.0125;
  long   qp_modea_spectral_npole = 64;
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

  // W^c UNION SUBSPACE (stage 1c of wc_band_elements.hpp): ONE orthonormal basis per q,
  // shared by the npk residue slabs of that q, so the Np axis of the sandwich is contracted
  // R_q times per (k,q,n) instead of sum_p r_p times -- the dominant stage-2 term goes from
  // npk*r*Np to npk*r*R + Np*R. Cut on the not-yet-spanned part of a retained slab direction,
  // weighted by its residue and scaled by the LARGEST slab of that q (see the normalization
  // argument at detail::union_build -- it bounds the absolute error of the residue sum, which
  // is what the sandwich accumulates, and it is NOT the per-slab relative measure of wrank):
  //     < 0  -> the restructure is OFF (the per-slab stage-1b path)   <-- THE DEFAULT
  //     = 0  -> take qp_modea_wrank
  //     > 0  -> that tolerance
  // WHY THE DEFAULT IS OFF (measured, qe_lih222 / mode_a / qpscf, the scan case
  // "qp_map_modea_wunion_scan" plus the rank ladder in wc_band_elements.hpp): the achieved
  // basis is R/Np = 1.00 at wunion = 1e-10 and still 0.81 at 1e-8, so at the accuracy class
  // the QM3 gates require the restructure has nothing to compress -- it only adds the
  // stage-1c build (measured +0.46 s against a 1.70 s stage 2 on the fixture). It becomes a
  // real speedup only where R << Np, i.e. at cuts of 1e-6 and looser, which is a spec
  // decision and not an agent default. It is a TRUNCATION trade, tau-anchor interlocked like
  // every other cut, and R, R/Np and the projection residual are logged on every build.
  double qp_modea_wunion = -1.0;

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

  // ---- increment TC-2 (notes/tc_coqui_impl_spec.md): P ON THE TILTED CONTOUR ----
  // Reached only through qp_modea_wfit = "contour", which is a SIBLING of the RW-2
  // "spectral" route: same knob family, same G provenance (the current QP spectrum
  // and MOs), same downstream consumption. ENABLE-flag-free -- it is plain complex
  // arithmetic in the existing THC kernels. With the knob absent every value below
  // is inert and the tau/nu/spectral paths are bit-for-bit unchanged (gate TC-2-b).
  //
  //   qp_tc_eps      rank tolerance of the contour builder. The rank is taken at
  //                  lambda > eps^2 lambda_max (BINDING correction 1 of
  //                  notes/tilted_contour_validation_results.md section 2.1 -- the
  //                  spec's own lambda > eps lambda_max delivers only sqrt(eps)),
  //                  and the contour length uses eps_tr = eps^2 (correction 4).
  //   qp_tc_delta    Im z of the target line, a.u. 0 selects the eq-8 recipe
  //                  delta = eta_targ with the mesh floor 1.2 W_band / N_k.
  //                  FLAGGED CONSTANT: the spec writes 0.7; results section 5.4
  //                  measured the 1 %-crossing at 0.81-3.99 x that prediction,
  //                  median 1.69, i.e. a fitted constant of ~1.2. And the adopted
  //                  sub-meV tier samples at eta_targ, NOT 3.5 eta_targ
  //                  (section 7.2 item 5), so there is no 3.5x continuation here.
  //   qp_tc_rho      tan(theta) W_target / delta, in [0, 1). 0.65 is the spec's
  //                  no-tuning value; the campaign confirmed rho* = 0.60-0.80 at
  //                  production meshes and rho = 0.65 within 15 % of optimal for
  //                  the semiconductor and the metal, 26 % for the wide-gap
  //                  insulator (section 4.1a).
  //   qp_tc_profile  "flat" (the adopted tier) or "growing" (the eq-8 growing-eta
  //                  profile). MEASURED gain 1.00x at 8^3 rising to 1.25-2.79x at
  //                  24^3 -- it pays only at >= 16^3 (section 4.2), so the default
  //                  is flat.
  //   qp_tc_trunc    band truncation along the contour: at node s only transitions
  //                  with a(Delta) s <~ ln(1/eps) survive, so the band sums shrink.
  //                  Measured 1.2-1.6x in the campaign, 2.1-2.6x here on the unit
  //                  fixtures (gate TC-2-c); the deviation from the samples it
  //                  drops is at the qp_tc_eps class.
  double      qp_tc_eps = 1e-6;
  double      qp_tc_delta = 0.0;
  double      qp_tc_rho = 0.65;
  std::string qp_tc_profile = "flat";
  bool        qp_tc_trunc = false;
  // ---- increment TC-3: the line solver and the eq-1 residue band-factor store ----
  //   qp_tc_krylov      warm-started GMRES instead of the dense inverse for the
  //                     contracted <nm|W^c|mn>. The economics (measured, notes/
  //                     tc3_report.md): the DIAGONAL path needs nbnd right-hand sides
  //                     per (q, z) and Krylov wins at ~5 iterations per solve; a full
  //                     qpscf block needs nbnd^2 and the dense inverse amortizes.
  //   qp_tc_krylov_tol  its relative-residual target. dense == Krylov measured 3.3e-13.
  //   qp_tc_bstore_gb   cap, in GB per owned (s,k) block, on the eq-1 residue term's
  //                     band-factor STORE B_J(P,a) -- nJ x Np x nbnd complex, ~1 MB on
  //                     qe_lih222 but ~1.25 GB at (64 k-points, Np 364, nbnd 60). It is
  //                     0 by default, which under qp_tc_bfactor = "auto" selects the
  //                     recompute path.
  // ---- increment TC-4: the band-factor representation and the residue batching ----
  //   qp_tc_bfactor     "auto" (default) | "store" | "recompute". The eq-1 residue term
  //                     needs the band-pair factor B_J(P,a) for every internal state it
  //                     visits. "recompute" keeps only B's two factors -- XCe at
  //                     nsym x Np x nbnd per owned block and XCi at ns*nkpts x Np x nbnd
  //                     shared by every block on the rank (22 MB at the production sizes
  //                     above, against 1.25 GB PER BLOCK for the store) -- and forms B_J
  //                     on demand at Np*nbnd flops, i.e. 0.06 % of the Np^3 Dyson solve
  //                     that consumes it. "store" materializes B and needs a
  //                     qp_tc_bstore_gb large enough to hold it. "auto" stores when the
  //                     cap admits it and recomputes otherwise, so the DEFAULT (cap 0)
  //                     is the recompute path. Both produce the same expression term by
  //                     term and agree bitwise.
  //   qp_tc_batch_mb    residue-evaluation batch budget, MB. It sizes the evaluator's
  //                     PERSISTENT work space (allocated once per residue source, grown
  //                     never shrunk), not a per-call allocation.
  //                     It caps the number of residue targets one batched call carries,
  //                     hence the (nt x Np^2) transform buffer inside the contour source
  //                     and the (nt x nbnd x nbnd) sandwich buffer in the assembly; 64 MB
  //                     is ~15 targets at (Np 364, nbnd 60) and thousands at fixture
  //                     scale. Raising it deepens the batch and the gemms. The
  //                     batching is what turns the transform contraction
  //                     R(z) = sum_j F(z,j) Pi(q, s_j) into one gemm per (q, chunk);
  //                     results do not depend on the value beyond gemm reassociation.
  bool        qp_tc_krylov = false;
  double      qp_tc_krylov_tol = 1e-12;
  double      qp_tc_bstore_gb = 0.0;
  std::string qp_tc_bfactor = "auto";
  double      qp_tc_batch_mb = 64.0;
};

} // methods

#endif //COQUI_QP_CONTEXT_H
