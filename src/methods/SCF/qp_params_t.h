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

  // Quasiparticle-map selector.
  // - "ac_pade":     Pade AC of Sigma(iw) evaluated near the real axis
  //                  (solve_qp_eqn / qp_approx). DEFAULT.
  // - "mats_lin":    Matsubara-native omega ~ 0 linearization
  //                  (qp_maps_matsubara.hpp map (i)) -- no analytic continuation.
  // - "mats_gmatch": Matsubara-native variational Green's-function matching
  //                  (qp_maps_matsubara.hpp map (ii)) -- no analytic continuation.
  // - "mode_a" / "mode_b": real-axis evaluation of Sigma^c at the quasiparticle
  //                  energies from a pole representation of W^c (qp_modea.hpp).
  std::string qp_map = "ac_pade";

  // mats_gmatch weight exponent: w_n = (w0/w_n)^qp_map_wpow on the positive
  // fermionic nodes. 2.0 emphasizes omega -> 0 (Z-weighted, Kutepov-like
  // behavior); 0.0 weights all nodes equally, exposing the QP-pole window
  // omega ~ |eps - mu| (closer to the real-axis Sigma(eps_QP) map). The
  // residual's leading 1/(i omega) tails cancel, so any wpow >= 0 is
  // well-posed.
  double qp_map_wpow = 2.0;

  // ---- qp_map = "mode_a" ----
  // NOTE ON ORDER: qp_params_t is an aggregate and the drivers use parenthesized aggregate
  // initialization positionally (MBPT_drivers.cpp), so new members must be APPENDED.
  //
  // Sigma^c evaluator for mode_a:
  // - "cd":        the contour-deformation closed form (sigma_route_b::sigma_cd) fed by
  //                the state-resolved W^c band elements. Production default.
  // - "expansion": solve the same map with the z0 = 0 re-expansion of the stored
  //                Sigma(iw) only (sigma_real_axis). Pure diagnostic -- needs no W data.
  std::string qp_modea_route = "cd";
  // inner QP-consistency cap at FIXED Sigma data. Non-convergence at the cap is a
  // physical multi-solution flag (app_warning), never a hard error.
  long qp_modea_nconsist = 5;
  // inner-consistency tolerance on max_a |d eps|, a.u.
  double qp_modea_consist_tol = 1e-8;
  // evaluation offset i*eta for the real-axis evaluation. Default 0.0; stress knob only.
  double qp_modea_eta = 0.0;
  // W^c support constraint: auxiliary pole nodes inside the particle-hole gap are dropped
  // from the fit -- prior physical information, not regularization. "auto" takes the
  // indirect gap of the CURRENT QP spectrum; "off" disables it (the unconstrained fit can be
  // orders of magnitude wrong at real z -- do not use in production); any other value is
  // parsed as an explicit gap edge in a.u.
  std::string qp_modea_wsupp = "auto";
  // pole-fit route: "tau" = bosonic mesh -> Ttw_bb -> fermionic tau kernel -> residues x
  // tanh(hw/2); "nu" = the support-constrained LS directly on the bosonic Matsubara nodes;
  // "spectral" and "contour" are described below.
  std::string qp_modea_wfit = "tau";
  // qp_modea_wfit = "spectral" replaces the least-squares pole fit above by a SIGN-DEFINITE
  // quadrature of the computed Im W^c(Omega, q) from the real-axis W module (needs
  // -DENABLE_FINUFFT=ON). Motivation: on a metal the LS residues are mixed sign and the
  // contracted Sigma^c needs several digits of cancellation. Knobs:
  //   spectral_eta   -- Lorentzian width of the QP-pole A(w), a.u. Drives the whole grid
  //                     stack; smaller is more accurate and quadratically more expensive.
  //   spectral_npole -- target positive-Omega node count after coarsening (pole count is
  //                     2x this plus the head sector). <= 0 keeps every Omega node.
  //   spectral_gamma -- "spectral" (default) takes the Gamma BODY from the quadrature and
  //                     leaves only the scalar head on LS poles; "ls" keeps the whole
  //                     q = Gamma column on an appended support-constrained LS pole set.
  //                     "ls" reintroduces the mixed-sign LS cancellation on the q = Gamma
  //                     transfer, which typically dominates the absolute sum on a metal;
  //                     the Gamma body is computed on the real axis on the "spectral"
  //                     path, so nothing is dropped there. Caveat: the scalar-head LS fit
  //                     of the "spectral" mode is not support-constrained, so it can carry
  //                     poles below 3 pi / beta; these affect eta = 0 probe rows only, not
  //                     the map.
  std::string qp_modea_spectral_gamma = "spectral";
  double qp_modea_spectral_eta = 0.0125;
  long   qp_modea_spectral_npole = 64;
  // truncated-SVD cut of the SUPPORT-CONSTRAINED W^c pole fit, relative to the largest
  // singular value. Negative selects the shared default, imag_axes_ft::
  // dlr_pole_fit_rel_tol = 1e-8.
  //
  // Why it is exposed: 1e-8 maximizes IMAGINARY-axis accuracy, which is what residue
  // algebras that never leave that axis need. The mode-A map instead evaluates the same
  // rational function at REAL quasiparticle energies, where a near-interpolatory fit is a
  // wild function with very many poles and residues orders of magnitude larger than the
  // data it represents. Loosening the cut (1e-6, 1e-4) trades imaginary-axis
  // reconstruction accuracy for much smaller residues; the knob is that
  // imaginary-accuracy vs real-axis-smoothness trade-off.
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
  // Why the default is off: at tight cuts (1e-10 .. 1e-8) the union basis spans nearly all
  // of Np, so the restructure has nothing to compress and only adds the stage-1c build. It
  // pays only where R << Np, i.e. at cuts of 1e-6 and looser. It is a truncation trade like
  // every other cut, and R, R/Np and the projection residual are logged on every build.
  double qp_modea_wunion = -1.0;

  // ---- GRADED-eta FAR-STATE EVALUATION ----
  // Imaginary offset, in a.u., applied to the evaluation energies of states OUTSIDE the
  // analyticity strip (VBM - 0.95 E_PH, CBM + 0.95 E_PH):
  //
  //     eta_far  = 0   -> out-of-strip states are evaluated at z = mu   (THE DEFAULT)
  //     eta_far  > 0   -> out-of-strip states are evaluated at z = eps + i eta_far
  //
  // In-strip states are unaffected and stay exact (eta -> 0). Applies to mode_a (both indices
  // of 1/2[Sigma(eps_a) + Sigma(eps_b)]) and to the mode_b diagonal.
  //
  // Why: evaluating out-of-strip states at mu biases band edges and gaps; a real-axis
  // reference evaluates far states as Re Sigma(eps + i eta), and this knob does the same.
  //
  // VALIDITY FLOOR (logged, warned, never fatal): eta_far must exceed ~3x the local fitted-pole
  // spacing or the evaluation rides single poles of the fit instead of the eta-smoothed
  // spectral density. The pole spacing is reported every outer iteration.
  double qp_modea_eta_far = 0.0;

  // ---- P ON THE TILTED CONTOUR ----
  // Reached only through qp_modea_wfit = "contour", which is a SIBLING of the "spectral"
  // route: same knob family, same G provenance (the current QP spectrum and MOs), same
  // downstream consumption. Needs no build flag -- it is plain complex arithmetic in the
  // existing THC kernels. With the knob absent every value below is inert and the
  // tau/nu/spectral paths are unchanged.
  //
  //   qp_tc_eps      rank tolerance of the contour builder. The rank is taken at
  //                  lambda > eps^2 lambda_max (a cut at lambda > eps lambda_max
  //                  delivers only sqrt(eps) accuracy), and the contour length uses
  //                  eps_tr = eps^2.
  //   qp_tc_delta    Im z of the target line, a.u. 0 selects the default recipe
  //                  delta = eta_targ with the mesh floor 1.2 W_band / N_k (the
  //                  constant 1.2 is empirical). The target line is sampled at
  //                  eta_targ itself, with no further continuation factor.
  //   qp_tc_rho      tan(theta) W_target / delta, in [0, 1). 0.65 is a no-tuning
  //                  value, close to optimal (optimum ~0.60-0.80 at production
  //                  meshes) for semiconductors and metals and somewhat less so for
  //                  wide-gap insulators.
  //   qp_tc_profile  "flat" (default) or "growing" (growing-eta profile along the
  //                  contour). The growing profile pays only on dense k meshes
  //                  (>= 16^3), so the default is flat.
  //   qp_tc_trunc    band truncation along the contour: at node s only transitions
  //                  with a(Delta) s <~ ln(1/eps) survive, so the band sums shrink.
  //                  The deviation from the samples it drops is at the qp_tc_eps
  //                  class.
  double      qp_tc_eps = 1e-6;
  double      qp_tc_delta = 0.0;
  double      qp_tc_rho = 0.65;
  std::string qp_tc_profile = "flat";
  bool        qp_tc_trunc = false;
  // ---- the line solver and the residue band-factor store ----
  //   qp_tc_krylov      warm-started GMRES instead of the dense inverse for the
  //                     contracted <nm|W^c|mn>. The DIAGONAL path needs nbnd right-hand
  //                     sides per (q, z), where Krylov wins at a few iterations per solve;
  //                     a full qpscf block needs nbnd^2 and the dense inverse amortizes.
  //   qp_tc_krylov_tol  its relative-residual target.
  //   qp_tc_bstore_gb   cap, in GB per owned (s,k) block, on the residue term's
  //                     band-factor STORE B_J(P,a) -- nJ x Np x nbnd complex, ~1 MB on
  //                     a small fixture but ~1.25 GB at (64 k-points, Np 364, nbnd 60). It
  //                     is 0 by default, which under qp_tc_bfactor = "auto" selects the
  //                     recompute path.
  // ---- the band-factor representation and the residue batching ----
  //   qp_tc_bfactor     "auto" (default) | "store" | "recompute". The residue term
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
  // ---- THE EXPLICIT STRIP WINDOW (mode-A CD route) ----
  //   qp_modea_strip_lo / qp_modea_strip_hi   HALF-WIDTHS below and above mu, a.u.
  //   Both 0 (DEFAULT) = unset = the E_PH-derived strip
  //   (VBM - 0.95 E_PH, CBM + 0.95 E_PH) exactly. Both > 0 replaces it with
  //   [mu - strip_lo, mu + strip_hi] and forces the strip active. Exactly one set is a
  //   parse error -- a one-sided window is never intended and pairing it with an E_PH
  //   bound would hide the mistake.
  //
  //   ⚠ WHY IT EXISTS. The E_PH strip is a window of order the GAP, so on a gapped system
  //   with a wide band window it admits only a few percent of the states, and the BAND
  //   EDGES can be clamped to mu, i.e. the reported VBM/CBM/gap then read Sigma^c(mu), not
  //   the contour. The default is correct for a METAL and wrong for any insulator QP or
  //   band-structure study.
  //
  //   It overrides the STRIP ONLY. gap_edge, the W^c support constraint, the retained pole
  //   set, the fit and the contour geometry are untouched -- an evaluation-coverage knob,
  //   not a representation knob. qp_modea_wsupp is NOT an alternative: it drives the
  //   support constraint and the strip TOGETHER, and widening it to cover a valence
  //   manifold guts the constrained fit (and is silently disabled past the outermost
  //   auxiliary node).
  //
  //   SIZING, for a QP calculation: cover the full valence manifold plus the conduction
  //   bands of interest, with margin --
  //       strip_lo ~ (mu - VBM) + valence bandwidth + margin
  //       strip_hi ~ (CBM - mu) + (top of the needed conduction bands - CBM) + margin
  //   The resolved window and the census are printed in the banner; check them, because a
  //   mis-sized window fails silently as a clamp artefact rather than as an error.
  //
  //   ⚠ CAVEAT: states outside the window are still clamped and still feed H_eff through
  //   self-consistency. That residual is second order for energy DIFFERENCES at a fixed
  //   clamp policy; a band-structure calculation must WIDEN THE WINDOW rather than rely on
  //   qp_modea_eta_far, whose cost is ~10^3 x at a 60-band window (the residue-target count
  //   grows with |eps - mu| and the deep conduction tail dominates).
  double      qp_modea_strip_lo = 0.0;
  double      qp_modea_strip_hi = 0.0;
  // ---- THE AMORTIZED W^c TILE CACHE (methods/SCF/wc_grid.hpp) ----
  //   ⚠ THE KNOB IS THE ACCURACY TARGET, NOT THE GRID SPACING.
  //   qp_tc_wgrid_mev   the ABSOLUTE residue accuracy target in meV
  //                     (default 1.0). h is derived from the empirical sizing law
  //                        dSigma[meV] = K (h/delta)^p / delta[eV],  p = 2.81,
  //                     with K = 32.4 (a conservative constant used for every
  //                     system; there is no metallicity auto-detection), a 3x
  //                     safety factor for the spread of the fit, and h clamped to
  //                     delta/2 (past which the 3-point stencil overshoots and
  //                     can be worse than linear).
  //                     0 DISABLES the cache and uses the per-target Np^3 Dyson
  //                     path exactly -- the reference for the cache.
  //   qp_tc_wgrid_h     EXPERT override: h directly, a.u. > 0 bypasses the law.
  //   qp_tc_wgrid_audit samples per (q, outer iteration) at which W^c is
  //                     evaluated EXACTLY and compared with the interpolation
  //                     (default 16; 0 = off, NOT recommended).
  //                     ⚠ WHY IT EXISTS: the error constant varies by orders of
  //                     magnitude between systems. On a spectrum whose weight is
  //                     CONCENTRATED on a single in-range pole a fixed constant is
  //                     badly wrong and NOTHING in the sizing inputs reveals it --
  //                     the grid is silently under-resolved and Sigma still looks
  //                     plausible. The law sizes; the sample proves.
  //   qp_tc_wgrid_audit_hard  (default true) ABORT when the measured error
  //                     exceeds 10x the target. Failure behaviour is defined:
  //                     always log predicted-vs-measured and the worst (q, Re z);
  //                     WARN and continue up to 10x; HARD ABORT above it, because
  //                     an order-of-magnitude breach means every downstream number
  //                     is untrustworthy. Set false to push through deliberately
  //                     on a diagnostic run.
  //                     The audit result is carried into the qpGW iteration
  //                     summary line as
  //                         wgrid_aud = <measured>/<predicted> meV (worst q = ..,
  //                                     Re z = ..)
  //                     -- measured FIRST -- so a breach is actionable from the
  //                     log alone and can be grepped as `wgrid_aud`.
  //                     Both values read -1 when the cache or the audit is off.
  double      qp_tc_wgrid_mev = 1.0;
  double      qp_tc_wgrid_h = 0.0;
  long        qp_tc_wgrid_audit = 16;
  bool        qp_tc_wgrid_audit_hard = true;
};

} // methods

#endif //COQUI_QP_CONTEXT_H
