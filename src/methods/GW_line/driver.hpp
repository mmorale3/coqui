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

#ifndef COQUI_METHODS_GW_LINE_DRIVER_HPP
#define COQUI_METHODS_GW_LINE_DRIVER_HPP

/**
 * Self-consistent GW on the tilted line (notes sections 5-7; plan S6; python coqui/cayley/cayley/line/driver.py::LineSCGW
 * and scripts/si222c_scgw_line.py). TOML block [gw_line] (plan section 5), read by gw_line_params_t::from_ptree:
 *
 *   theta_deg = 20      line angle (deg); the time rays use theta_t = theta / 2
 *   eps = 1e-10         tolerance of all real-pole bases (fermionic, bosonic)
 *   lam = 6.0           fermionic pole range (Ha)
 *   lam_b = auto        bosonic pole range (Ha); must cover Pi's spectrum e^> + |e^<| of the G poles (S7c: lam_b = 4 with
 *                       G poles up to 6 Ha gave W residues 1e5 x W and a 1e5 x amplification of the time-grid error in
 *                       Sigma); <= 0 or absent: 2 g_emax (lehmann) / 2 lam (compressed). A warning is logged per iteration
 *                       if the poles exceed it.
 *   sigma_gap = 0.02    Sigma-basis gap on each side (Ha). < 0: auto, gap_p = 0.8 (e_lumo + nu_min),
 *                       gap_h = 0.8 (|e_homo| + nu_min), e_homo/e_lumo the current QP edges (mu-relative), nu_min = bos gap
 *   bos_gap = 0.02      gap of the bosonic basis (Ha). < 0: auto, 0.5 x the current QP gap
 *   g_gap = 0.0         gap of the G compression bases (notes 6.4: keep 0; g_repr = "compressed" only)
 *   g_repr = "lehmann"  representation of G between iterations (S7c): "lehmann" = the Lehmann (e_m, v_m) of the
 *                       closure, factorized residues v v^dagger, pruned only (closure.hpp g_repr_params_t):
 *                         g_emax = lam      drop |e_m| > g_emax (moment-truncation artefacts; logged as "dropped")
 *                         g_wtol = 1e-12    drop poles of weight |v_m|^2 < g_wtol (count and weight logged)
 *                         g_emin_frac = 0.5, g_wsmall = 1e-4   drop in-gap poles: |e_m| < g_emin_frac x (QP half gap) AND
 *                                           weight < g_wsmall (count and weight logged; 0 = off). lih222 (K 24): ~280 such
 *                                           poles, total weight 4e-6..8e-6, each <= 8e-7, down to |e| = 3e-6 Ha; they
 *                                           set the ray length / the ID E_min (Pi nodes 235-291 -> 146, GL 1184-1440 -> 992)
 *                       "compressed" = per-sector refit on the gapless G bases (signed matrix coefficients; S6; parity)
 *   nodes_per_ray = 120, node_tmin = 1e-3, node_tmax = 60   dense fermionic nodes (log grid per ray); node_tmax is also
 *                       the tmax of the fermionic bases (python node_range)
 *   wp = 0.11, K = 24, tol_gram = 1e-10, nphi = 8           Cayley closure
 *   tol_gram_eps = 1    S7f: the Gram cut actually used is max(tol_gram, tol_gram_eps x eps): the moments come from fits on
 *                       eps-bases, so Gram eigenvalues below ~eps lambda_max are noise; keeping them amplifies roundoff into
 *                       G ~1e5 x (lih222 K 8, eps 1e-8). No-op at the production settings (eps = tol_gram = 1e-10); 0 = off
 *                       (python: tol_gram only; the [parity] test pins 0)
 *   tol_svd = 1e-12, closure_cut = "hard", closure_svd_cut = "hard", closure_cut_window = 10, phase_keep = 0
 *                       S7f closure options (closure.hpp closure_params_t, cayley.hpp upfold_opts_t): SVD cutoff, Gram /
 *                       SVD cut placement ("gap": at the largest eigenvalue ratio within a window of the tolerance;
 *                       "smooth": smooth-step weights of the Gram rows in the window), phase continuity (> 0)
 *   closure_threads = -1, closure_svd = "gesdd", closure_ueig = "cayley"   (pre-S7g: "gesvd", "schur")
 *                       S7g closure performance: BLAS threads of the per-k host closure (-1 = auto: the cores of the rank
 *                       (SLURM_CPUS_PER_TASK, else affinity / node ranks) in device runs, untouched in host runs where the
 *                       ranks fill the cores; 0 = untouched; n > 0 = n), SVD driver of D+ D-^dagger ("gesdd" = divide and
 *                       conquer), eigenvectors of the unitary U ("cayley" = Hermitian eigenproblem of i (U-1)^{-1} (U+1)
 *                       with Rayleigh-Ritz refinement and a Schur fallback; cayley.hpp unitary_eig_cayley)
 *   closure_k_workers = -1  S7g: the rank's k processed by this many concurrent host threads (std::thread; each with
 *                       closure_threads / closure_k_workers BLAS threads; no MPI inside); -1 = auto: closure_threads / 2
 *                       in device runs (pairs of cores; si222c bench 4 k x 2 threads 0.59 s per k vs 0.81 s for 1 k x 8),
 *                       1 in host runs
 *   closure_device = "auto", closure_dev_svd = "gesvdp"
 *                       S7g: the Gram / Cayley / Lehmann Hermitian eigenproblems, the LU solve of the "cayley" path and the
 *                       SVD on the GPU (cuSOLVER, cuda/gw_line_lapack.cu; CUDA builds; "auto" = in device runs), each call
 *                       falling back to the host LAPACK on any device failure (counted in the closure profile line)
 *   debug_noise_h0 = 0, debug_noise_seed = 0, debug_noise_sigma = 0, debug_noise_iter = 1, bases_file = ""
 *                       diagnostics: relative Hermitian noise on H0, or relative noise on Sigma at the nodes of iteration
 *                       debug_noise_iter (noise-floor meter, test [.scf_noise]); real-pole bases read from a file instead of built (parity test: the
 *                       pivoted-QR pole selection differs between LAPACKs; keys sigma_{particle,hole}_w, g_{particle,hole}_w,
 *                       bos_nu, bos_zeta_nodes)
 *   niter = 12          TOTAL number of iterations (a restart continues until niter iterations are done)
 *   mixing = 1.0        linear mixing of Sigma^{>/<} at the nodes (F is not mixed, as python). perf 7.2: default 1.0 (was 0.5;
 *                       the map Sigma_in -> Sigma[G] contracts by 0.15-0.25 per step for Si/LiH, so 0.5 made every
 *                       iteration halve the error: 16-24 iterations), with damp_below = 3e-4 (damped tail, below)
 *   conv_thr = 3e-5     stop when max|dSigma| at the nodes (after mixing, as python) < conv_thr (perf 7.2: default 3e-5, was
 *                       1e-5: the closure-noise floor of the residual is 2e-5..1e-4 for Si; in the damped tail dSigma =
 *                       0.5 resid, so 3e-5 = resid 6e-5, reached at iteration 9 for Si 2x2x2); with mixing_alg = "diis"
 *                       also mixing x max|Sigma[G] - Sigma_in| < conv_thr. Never in an iteration without a previous Sigma
 *                       (the first iteration, the first after a qp start or a level change of the multilevel schedule)
 *   mixing_alg = "linear"   perf 7.2 (scf_mixing.hpp): "linear" (above) | "diis" (Anderson/Pulay on Sigma^{>/<} at the nodes
 *                       of all k): diis_hist = 6, diis_start = 2 (iterations before it: linear with `mixing`), diis_beta = 1
 *                       (x = sum c (x_i + beta r_i)), diis_reg = 1e-10, diis_cmax = 10, diis_grow = 10 (resets to a linear
 *                       step), diis_mix_F = false (F in the vector; the next closure uses the extrapolated F), diis_wF = -1
 *                       (weight of the F elements; < 0: the number of fermionic nodes). The history is not checkpointed:
 *                       after a restart it is rebuilt (first step x + beta r). Every iteration logs the residual
 *                       max|Sigma[G] - Sigma_in| ("resid"; linear: dSigma / mixing) and max|F[D] - F_in|.
 *   damp_below = 3e-4, damp_mixing = 0.5   perf 7.2 damped tail (any mixing_alg): once resid < damp_below, linear steps with
 *                       damp_mixing (sticky, also across a restart). Undamped steps keep hopping at the closure's
 *                       discrete-decision noise (~1e-4 in Sigma for Si); damped steps let the decisions lock.
 *   start = "ks"        perf 7.2 initial G: "ks" (below) | "qp_diag": one Pi -> W -> Sigma pass on the KS poles (no SCF closure, not
 *                       counted as an iteration), diagonal G0W0 QP equation E_n = (H0 + F - mu)_nn + Re Sigma_nn(E_n + i
 *                       start_eta, default 1e-3 Ha) per k and band, Sigma_c from the upfolded closure representation of each k (Newton;
 *                       linearized fallback), then KS vectors with the QP energies (same density, F unchanged), mu = QP
 *                       mid-gap; the history of the run starts after it | "qp_file": the same with the energies
 *                       (absolute, Ha, band order, first nbnd bands) read from start_file (dataset start_dataset; default:
 *                       scf/iter<final_iter>/qp_approx/E_ska of a CoQui mbpt.h5, else "E_ska" / "qp_energies" at the root)
 *   coarse = { niter = 0, eps = 1e-8, K = 16, nodes_per_ray = 80, time_eps = coarse.eps }   perf 7.2 multilevel schedule: iterations
 *                       1..niter at these settings (bases, closure, fermionic nodes, time grids), then the production ones;
 *                       the Sigma of the coarse level is dropped at the switch (the first production iteration takes Sigma[G]
 *                       unmixed), so the run ends with >= 2 production iterations (niter >= coarse.niter + 2 required)
 *   t_chunk = 0 (auto), ray_decades = 36   time chunk of the ray products; ray length e^{-emin smax sin theta_t} = e^{-decades}
 *   time_grid = "id"    time nodes of the ray products (S7b): "id" = time-node ID (time_grids.hpp; four grids rebuilt every
 *                       iteration from the current poles and bosonic poles, ~100-170 nodes each), "gl" = the generic
 *                       Gauss-Legendre rays for_spectrum(theta_t, emin, ray_decades) (~1000 nodes; the python reference)
 *   time_eps = eps, time_pad = 1.25, time_oversample = 1.0   ID tolerance, energy-range margin [Emin/pad, pad Emax] and
 *                       node oversampling (time_id_opts_t); ignored for "gl"
 *   time_snap = 0       S7f: > 0 = the ID |E| ranges widened to a geometric grid of time_snap points per octave, so the
 *                       grids do not change with roundoff-level changes of the poles (line_time_grids_t)
 *   restart = false     resume from <output>.gw_line.h5:/scf_line/final_iter (bitwise identical continuation)
 *   checkpoint_sigma = "last"   Sigma at the nodes in the checkpoint (S7e): "last" = only the last iteration's, in the
 *                       separate file <output>.gw_line.sigma.h5 rewritten every iteration (constant size); "all" = every
 *                       iteration's in iter<N>/Sigma_{p,h} (the pre-S7e layout; 2 N_k N_zeta nb^2 x 16 B per iteration)
 *   sigma_kdist = true  Sigma at the nodes k-distributed over the ranks (S7e, k_dist.hpp: owner(k) = k mod np, the closure's
 *                       ownership; reduce-scatter in the self-energy); false = replicated on every rank (pre-S7e)
 *   mem_budget_gb = 0, dev_mem_budget_gb = 0, mem_frac = 0.8, q_group_size = 0   perf 7.4b (q_plan.hpp): q groups of the
 *                       Pi -> W stage chosen automatically as the largest group whose plan-6.7 model fits mem_frac of the free
 *                       host memory per rank (MemAvailable / cgroup, at the plan) and of the free device memory; the budgets
 *                       in GB per rank override the measurement; q_group_size > 0 (or env COQUI_GWLINE_QGROUP) fixes the size
 *   ibz = true          perf 7.3: IBZ reduction (ibz.hpp) when the mean field is symmetric; false = full BZ (env COQUI_GWLINE_IBZ)
 *   theta_t_frac = 0.5  S8b: the ray angle theta_t = theta_t_frac theta (was fixed at 1/2)
 *   beta = 0            S8b finite temperature (notes section 11; thermal.hpp): 0 = T = 0. beta > 0: thermal_tol = 1e-8 (tau_T,
 *                       window E_T = ln(1/tau_T)/beta), thermal_floor = 30 (c_zeta: bosonic node floor rho beta |zeta| >= c_zeta,
 *                       band bottom), thermal_floor_f = thermal_floor (c_f: fermionic node floor of the closure fit), wp_floor = 15
 *                       (omega_p := max(wp, wp_floor zeta_T)), mu_rule = "auto" ("auto" | "gap" | "number"; mu_dn_max = 0.1,
 *                       mu_th_factor = 10, notes Eq. fT_nth), band_heights = 8, band_x = 21, band_top = 4, mats_factor = 4 (data set
 *                       D), bos_eps_T = 1e-12 (D-selected basis), bos_line_eps = 1e-10 (the line basis of D's line nodes),
 *                       cut_odd = 1e-13, cut_even = 1e-10 (split fit), tau_grid = "gl" | "id", tau_eps = 1e-12 (tau leg),
 *                       spectra.occupation = false, thermal_bases_file (parity: D, nu_b, the Sigma basis injected).
 *                       An iteration is THERMAL iff a pole lies within E_T of mu; otherwise it is the T = 0 code (bitwise).
 *                       Thermal iterations need g_repr = "lehmann"; sigma_gap / bos_gap are not used there (two-sided gapless
 *                       Sigma basis, D-selected bosonic basis); start = "ks"; no multilevel, no optics.
 *   output / outdir + prefix   checkpoint stem (MBPT_drivers resolve_mbpt_output_stem; the driver reads "output")
 *   div_treatment = "ignore_g0"   q -> 0 divergence of Sigma_c (S9a, head.hpp): "ignore_g0" (Z(Gamma) without its G = 0 term,
 *                       no head term) or a gygi variant of CoQui ("gygi" = axis-folded polynomial extrapolation of the head
 *                       eps^-1_00(q) - 1 to q -> 0, "gygi_order_N", "gygi_perdir", "gygi_2d", "gygi_smallest_q", "gygi_average"):
 *                       dSigma_c = madelung x (extrapolated head) x T G T^dagger per sector (gw_t::Sigma_div_correction).
 *                       nqpts == 1 forces "ignore_g0" for Sigma_c (as CoQui).
 *   hf_div_treatment    exchange: "ignore_g0" | "gygi" (Sigma_x -= madelung D, hf_t::HF_K_correction with S = 1); default
 *                       "ignore_g0" if div_treatment is "ignore_g0", else "gygi"
 *   head_extrapolation = "gygi"   q -> 0 variant of the head data (checkpoint, eps_inf) when div_treatment = "ignore_g0";
 *                       with a gygi div_treatment the head uses div_treatment itself
 *   spectra = { enable = true, eta = [0.004, 0.01], wmin = -0.45, wmax = 0.45, nw = 601 }   A(k,w) at the end
 *   optics = { enable, wmin = 0, wmax = 1.5, nw = 1501, eta = [0.01], eta_rel = [0.05], scales = "auto" | [..], nscales = 4,
 *              K = "auto" | int, q0 = true, finite_q = true, theta_deg = <flatter final line(s), deg>, time_grid = "id",
 *              nline, npole, mem_gb = 2, nnls_n = 1500, delta_floor = 1e-10,
 *              q0_variants = ["gygi_perdir", "gygi_smallest_q", "gygi_average"] }   (delta_floor: lower bound of the
 *                       fit-residual noise level delta used by the optics closure)
 *                       S9b real-axis optics (optics.hpp, head_pass.hpp) after the loop (also from a restart with nothing
 *                       left to iterate): loss, eps1, eps2, n, kappa, alpha, R, sigma for q -> 0 and every mesh q != Gamma
 *                       from the head at the SCF angle (the last iteration of this run, else the checkpoint's last head
 *                       group, else recomputed from the final poles: head_pass) and, for every theta_deg, from ONE extra
 *                       Pi -> W -> head pass on a flatter bosonic line (own basis, time rays at theta_deg / 2).
 *                       h5 group optics/theta<deg>/{q0 = q -> 0, q0_<variant>, iq<n> = mesh q n} of the checkpoint (layout: optics.hpp write_optics).
 *                       optics.poles = "final" (default) | "initial" (perf 7.2): the G of every optics line: the final poles
 *                       (SCF angle: the head of the last iteration) or the initial ones (iteration 0: KS or the qp start;
 *                       SCF angle: the head of iteration 1, computed from them), e.g. RPA@PBE next to G0W0 with niter = 1.
 *
 * Spectra (perf 7.2): A(k, w) of the G whose poles are stored, G_N = [w - (H0 + F_N^cl - mu) - Sigma_N]^{-1} with Sigma_N the
 * (mixed) Sigma input of the last closure and F_N^cl the F of that closure (scf_line/iter<N>/F_closure; = F[D_{N-1}], the
 * density of the previous iteration, unless diis_mix_F). Before 7.2 the spectra used F[D_N], inconsistent with G_N by
 * F[D_N] - F[D_{N-1}] (G0W0, niter = 1: 0.15-0.23 eV peak shifts). niter = 1 from the KS start: the spectra are those of the
 * G0W0@KS G (F[D_KS] + Sigma[G_KS]) and the poles are its Lehmann poles.
 *
 * Initial guess (as si222c_scgw_line.py): KS poles e_n(k) - mu0 with unit residues in the KS band basis, mu0 = KS
 * mid-gap (python uses CoQui's Matsubara mu; the gap midpoint is used here, no imaginary-axis checkpoint needed), and
 * F = V_H + Sigma_x of the KS density matrix. H0 = the non-interacting one-body Hamiltonian (kinetic + pseudopotential, no
 * Hartree, no xc) in the KS band basis, built exactly as the imaginary-axis SCF builds it (hamilt::set_H0, hermitized).
 *
 * Loop (python LineSCGW.iterate): rays from the current poles -> Pi at the bosonic nodes -> W residues -> Sigma^{>/<} at
 * the dense nodes -> mixing -> closure with H_stat = H0 + F - mu (closure.hpp) -> new mu and compressed poles -> D, F.
 * Checkpoint: <output>.gw_line.h5 (own file), written by the root after every iteration:
 *   system/{nkpts, nbnd, Np, nelec, H0, eigval, qk_to_k2, mu0}, input/{parameters, fermionic_nodes},
 *   scf_line/final_iter, scf_line/iter<N>/{mu, mu_sigma, dmu, e_homo, e_lumo, F, Sigma_p, Sigma_h (N >= 1),
 *   poles/{particle,hole}_{counts,e,coef | v}, history scalars (incl. time_grid and the node counts nt_{pi,sigma}_{p,h})},
 *   poles: matrix coefficients {s}_coef (sum M, nb, nb) or, factorized (g_repr = "lehmann"), {s}_v (sum M, nb) = the
 *   Lehmann vectors v_m as rows (S6/S7b checkpoints with coefficients stay readable),
 *   iter0 = the initial state; spectra/ at the end.
 *   S9a: scf_line/iter<N>/head/ (N >= 1, every div_treatment): h_nodes (nq, nz_b) = eps^-1_00(q, zeta_i) - 1 at the bosonic
 *   nodes from W; h_res, h_res_hole (nq, r_b) scalar residues of the particle (+nu_j) and hole (-nu_j) parts,
 *   h(q, z) = sum_j h_res_j / (z - nu_j) - h_res_hole_j / (z + nu_j) (raw fit residues: the pole function is determined, the
 *   individual residues are not, see head.hpp); h0_nodes (nz_b), h0_res, h0_res_hole (r_b): the q -> 0 extrapolation
 *   (weights q_weights (nq) of the variant `extrapolation`); zeta (nz_b), nu (r_b), qpts (nq, 3), madelung, eps_inf
 *   = 1 / (1 + Re h0(0)), div_treatment, hf_div_treatment.
 */

#include <optional>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "mean_field/MF.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/spectra.hpp"
#include "methods/GW_line/optics.hpp"
#include "methods/GW_line/scf_mixing.hpp"

namespace methods::gw_line {

struct gw_line_params_t {
  double theta_deg = 20.0, eps = 1e-10, lam = 6.0, lam_b = -1.0;   ///< lam_b <= 0: auto (2 g_emax / 2 lam), S7c
  bool lam_b_auto = false;
  double sigma_gap = 0.02, bos_gap = 0.02, g_gap = 0.0;
  std::string g_repr = "lehmann";        ///< "lehmann" | "compressed" (S7c)
  double g_emax = -1.0, g_wtol = 1e-12, g_emin_frac = 0.5, g_wsmall = 1e-4;   ///< lehmann pruning (g_emax < 0: lam)
  long nodes_per_ray = 120;
  double node_tmin = 1e-3, node_tmax = 60.0;
  double wp = 0.11;
  long K = 24;
  double tol_gram = 1e-10;
  long nphi = 8;
  double tol_svd = 1e-12;                ///< S7f: relative SVD cutoff of the upfolding (python 1e-12)
  double tol_gram_eps = 1.0;             ///< S7f: Gram cut >= tol_gram_eps x eps (the noise level of the moments); 0 = off
  std::string closure_cut = "hard";      ///< S7f: Gram cut "hard" (python) | "gap" | "smooth" (cayley.hpp upfold_opts_t)
  std::string closure_svd_cut = "hard";  ///< S7f: SVD cut "hard" | "gap"
  double closure_cut_window = 10.0;      ///< S7f: window factor of the gap / smooth cuts
  double phase_keep = 0.0;               ///< S7f: > 0 = phase continuity (keep the previous phi* basin within this factor)
  long closure_k_workers = -1;           ///< S7g: concurrent host threads over the rank's k (-1 auto: device runs threads / 2, host 1)
  long closure_threads = -1;             ///< S7g: BLAS threads of the host closure (-1 auto: device runs the rank's cores, host runs untouched; 0 untouched)
  std::string closure_svd  = "gesdd";    ///< S7g: SVD driver of the upfolding "gesdd" (default) | "gesvd" (python / pre-S7g)
  std::string closure_ueig = "cayley";   ///< S7g: eigenvectors of U "cayley" (default; Hermitian Cayley image) | "schur" (zgees, pre-S7g)
  std::string closure_device = "auto";   ///< S7g: cuSOLVER eigensolvers/SVD in the closure "auto" (device runs) | "on" | "off"
  std::string closure_dev_svd = "gesvdp"; ///< S7g: device SVD "gesvdp" (polar decomposition, default) | "gesvd"
  double debug_noise_h0 = 0.0;           ///< diagnostics: relative Hermitian Gaussian noise on H0 (seed debug_noise_seed)
  long debug_noise_seed = 0;
  double debug_noise_sigma = 0.0;        ///< diagnostics: relative noise on Sigma at the nodes (before mixing) ...
  long debug_noise_iter = 1;             ///< ... in this iteration
  std::string bases_file;                ///< diagnostics/parity: real-pole bases read from this file (gen_lih222_scf_ref.py)
  long niter = 12;
  double mixing = 1.0, conv_thr = 3e-5;   ///< perf 7.2: defaults mixing 1.0 (was 0.5) + damped tail, conv_thr 3e-5 (was 1e-5)
  mixing_params_t mix;                   ///< perf 7.2: mixing algorithm (scf_mixing.hpp); mix.mixing == mixing
  std::string start = "ks";              ///< perf 7.2: initial G "ks" | "qp_diag" | "qp_file"
  std::string start_file, start_dataset; ///< perf 7.2: QP energies for start = "qp_file"
  double start_eta = 1e-3;               ///< perf 7.2: broadening of the diagonal QP equation (Ha)
  long coarse_niter = 0;                 ///< perf 7.2: multilevel schedule, iterations 1..coarse_niter at coarse settings
  double coarse_eps = 1e-8, coarse_time_eps = 1e-8;
  long coarse_K = 16, coarse_nodes_per_ray = 80;
  std::string optics_poles = "final";    ///< perf 7.2: G of the optics passes "final" | "initial"
  long t_chunk = 0;   // 0 = automatic (host: 32, COQUI_GWLINE_HOST_TCHUNK; device: from the free memory, capped)
  double ray_decades = 36.0;
  std::string time_grid = "id";          ///< "id" (time-node ID) or "gl" (Gauss-Legendre rays)
  double time_eps = 1e-10, time_pad = 1.25, time_oversample = 1.0;   ///< time_eps defaults to eps
  double time_snap = 0.0;                ///< S7f: ID |E| ranges snapped to a geometric grid (points per octave; 0 = off)
  bool restart = false;
  std::string output = "./gw_line";
  std::string checkpoint_sigma = "last";   ///< "last" | "all" (S7e)
  bool sigma_kdist = true;                 ///< k-distributed Sigma at the nodes (S7e)
  bool ibz = true;                         ///< perf 7.3: use the symmetry reduction of a symmetric mean field (ibz.hpp); env COQUI_GWLINE_IBZ
  double mem_budget_gb = 0.0;              ///< perf 7.4b: host memory per rank for the GW_line arrays (GB; 0 = mem_frac of the free memory)
  double dev_mem_budget_gb = 0.0;          ///< perf 7.4b: device memory per rank (GB; 0 = mem_frac of the free device memory)
  double mem_frac = 0.8;                   ///< perf 7.4b: fraction of the available memory the q plan may fill
  long q_group_size = 0;                   ///< perf 7.4b: > 0 fixes the largest q-group size (0 = automatic, q_plan.hpp)
  bool do_spectra = true;
  spectra_params_t spectra;
  std::string div_treatment = "ignore_g0";      ///< S9a: Sigma_c head term ("ignore_g0" | gygi variants, head.hpp)
  std::string hf_div_treatment = "ignore_g0";   ///< S9a: exchange Madelung term ("ignore_g0" | "gygi")
  std::string head_extrapolation = "gygi";      ///< S9a: q -> 0 variant of the head data when div_treatment = "ignore_g0"
  optics_params_t optics;                       ///< S9b: real-axis optics after the loop (optics.hpp)
  // S8b finite temperature (notes section 11; thermal.hpp, thermal_mu.hpp)
  double theta_t_frac = 0.5;                    ///< theta_t = theta_t_frac theta (also at T = 0)
  double beta = 0.0;                            ///< 0: T = 0
  double thermal_tol = 1e-8, thermal_floor = 30.0, thermal_floor_f = -1.0;   ///< tau_T, c_zeta, c_f (< 0: c_zeta)
  double wp_floor = 15.0;                       ///< omega_p >= wp_floor zeta_T in thermal iterations
  std::string mu_rule = "auto";                 ///< "auto" | "gap" | "number"
  double mu_dn_max = 0.1, mu_th_factor = 10.0;  ///< rule "auto" (Eq. fT_nth)
  long band_heights = 8, band_x = 21;           ///< wedge band of the bosonic data set D
  double mats_factor = 4.0, band_top = 4.0;     ///< N_M = ceil(mats_factor zeta_T beta / 2 pi); band top
  double bos_eps_T = 1e-12;                     ///< eps_b of the D-selected bosonic basis
  double bos_line_eps = 1e-10;                  ///< eps of the gapless bosonic line basis whose unmasked nodes enter D
  double cut_odd = 1e-13, cut_even = 1e-10;     ///< split pair fit cutoffs
  std::string tau_grid = "gl";                  ///< tau leg nodes "gl" | "id"
  double tau_eps = 1e-12;                       ///< tau ID tolerance
  bool spectra_occupation = false;              ///< spectra.occupation: also f(w - mu) A(k, w)
  std::string thermal_bases_file;               ///< parity: D (D_zeta_re/_im), nu_b, sigma_basis_w injected (python reference)

  static gw_line_params_t from_ptree(ptree const &pt);
  void log() const;
};

/// One line of the iteration table (python history record).
struct gw_line_iter_t {
  long iter = 0;
  double dSigma = 0.0, mu = 0.0, dmu = 0.0, gap = 0.0, e_homo = 0.0, e_lumo = 0.0;
  double nelec = 0.0, nelec_lehmann = 0.0, N_mu = 0.0, dropped = 0.0, heldout_max = 0.0;
  long npoles_min = 0, npoles_max = 0;
  double bos_gap = 0.0, sigma_gap_p = 0.0, sigma_gap_h = 0.0, time = 0.0;
  std::string time_grid = "gl";          ///< time grid used in this iteration
  long nt_pi_p = 0, nt_pi_h = 0, nt_sig_p = 0, nt_sig_h = 0;   ///< node counts (Pi / Sigma, particle / hole ray)
  std::string g_repr = "compressed";     ///< representation of the poles produced by this iteration (S7c)
  long ng_min = 0, ng_max = 0;           ///< retained G poles per k and sector (min, max)
  double g_emin = 0.0;                   ///< smallest retained |e_m| (sets the ID E_min of the next iteration)
  long pruned_w = 0, pruned_near = 0;    ///< lehmann: poles pruned by weight / near-mu rule (all k)
  double pruned_w_weight = 0.0, pruned_near_weight = 0.0;
  double resid = 0.0, residF = 0.0;      ///< perf 7.2: max|Sigma[G] - Sigma_in| (total Sigma), max|F[D] - F_in| (0 before 7.2)
  std::string mix = "none";              ///< perf 7.2: mixing step of this iteration (none | linear | diis | reset)
  long ndiis = 0;                        ///< perf 7.2: DIIS history entries used
  long level = 0;                        ///< perf 7.2: 0 = production settings, 1 = coarse (multilevel schedule)
  // S8b (written in iter<N>/thermal/, not in the broadcast history table)
  long thermal = 0;                      ///< 1: thermal iteration (window non-empty at its start)
  std::string mu_rule = "";              ///< rule the closure used
  double dN = 0.0, n_th = 0.0, wp_used = 0.0;
  long nD = 0, rank_b = 0, ntau = 0, nwin = 0;
};

struct gw_line_result_t {
  std::vector<gw_line_iter_t> history;   ///< all iterations (including those read on restart)
  bool converged = false;
  double mu = 0.0, mu_sigma = 0.0;       ///< final centre; centre at which Sigma_p/h were sampled
  pole_data_t poles;                     ///< final poles (mu-relative; compressed or factorized Lehmann)
  nda::array<ComplexType, 3> F;          ///< final V_H + Sigma_x
  nda::array<ComplexType, 3> F_closure;  ///< perf 7.2: the F of the closure that built the final poles (spectra use it)
  double start_time = 0.0;               ///< perf 7.2: seconds of the qp_diag start pass (0 otherwise)
  nda::array<ComplexType, 3> H0;         ///< one-body Hamiltonian (KS band basis)
  nda::array<ComplexType, 4> Sig_p, Sig_h;   ///< last mixed Sigma at the nodes (empty if no iteration was done); with
                                             ///< sigma_kdist (default) the rows of the k owned by this rank (k mod np)
  nda::array<ComplexType, 1> zeta;       ///< fermionic nodes (mu-relative)
  std::optional<spectra_out_t> spectra;
  std::vector<double> eps_inf;           ///< S9a: 1 / (1 + Re h0(0)) of the iterations done in this run
  nda::array<ComplexType, 1> head_hp0, head_hh0;   ///< S9a: extrapolated head residues of the last iteration of this run
  std::vector<double> optics_theta;                ///< S9b: angles of the optics lines (SCF angle first)
  std::vector<optics_q_t> optics_q0;               ///< S9b: q -> 0 optics per line (same order)
  long q_group_size = 0, q_ngroups = 0;           ///< perf 7.4b: the q plan of the Pi -> W stage (last one made)
  bool q_wR_inplace = false;                      ///< perf 7.4b: Sigma's real-space residues overwrote w
  double q_budget_host = -1.0, q_model_host = 0.0, q_model_host_all = 0.0, q_model_host_min = 0.0;   ///< bytes per rank
};

/// Non-interacting one-body Hamiltonian (no xc) in the KS band basis, (nk, nb, nb), hermitized; collective.
nda::array<ComplexType, 3> one_body_h0(mf::MF &mf);

/// Root rank writes system/{nkpts, nbnd, Np, nelec, H0, eigval, qk_to_k2, mu0} into `file` ('w' if truncate, else 'a').
void write_system_h5(boost::mpi3::communicator &comm, std::string const &file, mf::MF &mf, long Np,
                     nda::array<ComplexType, 3> const &H0, double mu0, bool truncate);

/// The line scGW driver. Collective over thc.mpi()->comm.
template <MEMORY_SPACE MEM> gw_line_result_t gw_line_scf(methods::thc_reader_t &thc, mf::MF &mf, ptree const &pt);

extern template gw_line_result_t gw_line_scf<HOST_MEMORY>(methods::thc_reader_t &, mf::MF &, ptree const &);
#if defined(ENABLE_DEVICE)
extern template gw_line_result_t gw_line_scf<DEVICE_MEMORY>(methods::thc_reader_t &, mf::MF &, ptree const &);
#endif

} // namespace methods::gw_line

#endif
