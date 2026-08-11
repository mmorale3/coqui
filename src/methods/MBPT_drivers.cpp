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



#include <string>

#include "configuration.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "cxxopts.hpp"

#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"

#include "hamiltonian/pseudo/pseudopot.h"
#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "mean_field/mf_utils.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/mb_state/mb_state.hpp"
#include "methods/SCF/dca_dyson.h"
#include "methods/SCF/simple_dyson.h"
#include "methods/embedding/embed_t.h"
#include "methods/embedding/embed_eri_t.h"
#include "methods/vertex/vertex_t.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/iter_scf/iter_scf_utils.hpp"

#include "SCF/scf_driver.hpp"
#include "MBPT_drivers.h"

namespace mpi3 = boost::mpi3;
namespace methods
{

inline std::string resolve_mbpt_output_stem(ptree const& pt) {
  auto output_opt = pt.get_optional<std::string>("output");
  if (output_opt and !output_opt->empty()) return output_opt.value();

  std::string err = std::string("Incorrect input - ");
  auto outdir = io::get_value_with_default<std::string>(pt, "outdir", "./");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  if (prefix.empty()) {
    utils::check(false, "Incorrect input - prefix cannot be empty string.");
  }
  return outdir + "/" + prefix;
}

// Helper function to prepare checkpoint file for downfold_coulomb
inline void ensure_checkpoint(std::shared_ptr<mf::MF> mf, std::string const& output, 
                              std::string const& greens_func_source, ptree const& pt) {
  
  if (greens_func_source == "mf" and std::filesystem::exists(output+".mbpt.h5")) {
    
    app_log(1, "");
    app_log(1, "╔═════════════════════════════════════════════════════════════╗");
    app_log(1, "║ [ NOTE ]                                                    ║");
    app_log(1, "║ greens_func_source is set to \"mf\", while a CoQuí checkpoint ║");
    app_log(1, "║ HDF5 with the same prefix has been detected. CoQuí will     ║");
    app_log(1, "║ read \"scf/iter0\" h5 group as the input, which should be     ║");
    app_log(1, "║ equivalent to the mean-field solution.                      ║");
    app_log(1, "╚═════════════════════════════════════════════════════════════╝\n");

  } else if (greens_func_source == "mf" and not std::filesystem::exists(output+".mbpt.h5")) {
    
    imag_axes_ft::IAFT ft(pt, false, mf::wmax_from_mf(*mf));
    hamilt::pseudopot psp(*mf);
    write_mf_data(*mf, ft, psp, output);
  
  } else if (greens_func_source == "scf" or greens_func_source == "embed") {
  
    utils::check(std::filesystem::exists(output+".mbpt.h5"),
                 "MBPT_drivers::ensure_checkpoint: greens_func_source == \"{}\" while the coqui h5, {}.mbpt.h5, does not exist!", 
                 greens_func_source, output);

  } else {

    utils::check(false, "MBPT_drivers::ensure_checkpoint: invalid greens_func_source = {}. Valid options are \"mf\", \"scf\", and \"embed\".", 
                 greens_func_source);

  }
}

/**
 * Many-body perturbation calculations from a given mean-field and ERI objects with arguments in property tree.
 * Optional arguments (with default values):
 *  - beta: "1000" Inverse temperature (a.u.)
 *  - wmax: Optional. Frequency cutoff for the IAFT grids (a.u.).
 *          If not provided, wmax is estimated from mean_field. 
 *  - iaft_prec: "high" Precision of IAFT grids. {choices: "high", "medium", "low"}
 *  - div_treatment: "gygi" Divergent treatment for Coulomb kernel. {choices: "ignore_g0",
 *                 "gygi", "cvv"}. "cvv" (gw solver only; scGW-tilde,
 *                 notes/scgwt_implementation_plan.md) replaces the gygi/stored q->0
 *                 EXTRAPOLATION of eps_inv_head by the covariant-velocity O(q^2) head,
 *                 Pi_ab(inu) = -(2/(beta Nk V)) sum_k,iw tr[v~_a G v~_b G] with
 *                 v~ = d_k(H0 + F + Sigma). LIVE since increment C4: update_w fills
 *                 mb_state.eps_inv_head from the SUBTRACTED head coefficient
 *                 Phead(inu) = [Pi^jj(inu) - Pi^jj(0)]/(inu)^2 (zero-Drude-weight
 *                 identity; insulators), wings dropped; all downstream consumers
 *                 single-source that array. Logs the T-d meter v(q).P00 per iteration.
 *  - hf_div_treatment: "gygi" Divergent treatment for Coulomb kernel in HF. {choices: "ignore_g0", "gygi"}
 *  - niter: "1" Number of iterations in the self-consistent loop.
 *  - conv_thr: "1e-9" Convergence threshold for the self-consistent loop.
 *  - const_mu: "false" Fix the chemical potential during the self-consistent loop.
 *  - output: Optional legacy output flag. If present, this is used directly.
 *  - outdir: "./" Output directory used when output is not provided.
 *  - prefix: "bdft.mbpt" Prefix used when output is not provided.
 *  - restart: "false" Restart from a previous bdft.scf calculation.
 *  - t_prescreen_thresh: "0.0" Threshold for prescreening in time (GF2 only for now)
 *  - vertex_type: "none" Vertex correction on top of the gw solver.
 *                 {choices: "none", "2nd_exchange"}. "2nd_exchange" enables BOTH cuts of the
 *                 Phi-derivable second-order-exchange functional: Sigma^C (G3W2) and Pi^C (G4W).
 *  - vertex_rung: "dynamic" Rung mode of the vertex correction (gw solver only;
 *                 notes/static_vertex_implementation_plan.md section 2.1).
 *                 {choices: "dynamic", "linear", "static"}. "dynamic" (default) is the
 *                 parent Formulation B (dynamic W rungs, G3W2 + G4W). "static" (B-S) and
 *                 "linear" (B-L) are the two conserving static-rung truncations, in which
 *                 the rungs are the iv = 0 statically screened W0[G]. ONE vertex_t drives
 *                 all cuts of the selected mode, so mixed half-theories cannot be
 *                 configured. All other vertex_* keys apply to every mode. All three
 *                 modes are fully implemented (plan increments S0-S10 complete); an
 *                 empty window is a no-op in every mode.
 *  - vertex_band_window: [i0, i1) Contiguous 0-based orbital range defining the vertex
 *                 subspace C (gw solver only). Absent/empty window means C is the empty set,
 *                 which reproduces plain scGW exactly.
 *                 Requirements with an active vertex: DLR IAFT backend (iaft basis "dlr");
 *                 screen_type "rpa" or "rpa_k". Symmetry-reduced (IBZ) k-meshes are
 *                 supported (notes/vertex_ibz_symmetry.md): external axes stay
 *                 IBZ-resident and the rung transfers are sourced from IBZ-stored W/Z;
 *                 the C-window D-matrix leakage of the symmetry rotations is measured
 *                 and logged (expected small; O(leakage) relative error on the vertex).
 *                 Note: Pi^C uses the PREVIOUS iteration's screened W (one-iteration lag;
 *                 first iteration uses the bare-Z rung), and dW stays resident across the
 *                 iteration boundary (memory tradeoff). The vertex inherits the run's
 *                 div_treatment for its q->0 rung policy: "ignore_g0" includes the stored
 *                 regularized W(Gamma) body; "gygi" additionally applies the analytic
 *                 rank-1 head insertion (madelung x basis_head x eps_inv_head). At coarse
 *                 k-meshes the gygi head can dominate the vertex rung sums (O(Nk^-1/3)
 *                 convergence) -- check mesh convergence of the head fraction.
 *  - vertex_isdf: "global" Auxiliary basis of the vertex kernels (gw solver only;
 *                 Refinement 2, notes/refinement2_optionA.md). {choices: "global",
 *                 "secondary"}. "global" (default) runs the kernels in the global THC
 *                 basis (dimension Np) -- the original path, bit-identical. "secondary"
 *                 builds a dedicated secondary ISDF basis on the subspace C by re-running
 *                 the restricted-range point selection (once per geometry) and runs both
 *                 kernels with the auxiliary dimension N_m = O(nc^2 nk) << Np: the rung
 *                 cores are downfolded with the frequency-independent Option-A transfer
 *                 t(q) = s(q)^+ B(q)^dag C(q); Sigma^C lands directly in the C-C block;
 *                 Pi^C is upfolded with the adjoint of the same t (no-leak). The q->0
 *                 gygi head insertion downfolds automatically through t. Downfold
 *                 fidelity is reported per q as eta(q, nu) (theoryB Eq. 40) at verbosity
 *                 >= 2 (test scale).
 *  - vertex_isdf_rank: "-1" Secondary basis size N_m ("secondary" only). -1 selects the
 *                 full subspace pair rank nc^2 * nk (eta -> 0 limit); smaller values
 *                 trade accuracy (monitored by eta) for cost. The point selection may
 *                 return fewer points when the C pair-density metric is numerically
 *                 rank-deficient; the returned count is used and logged.
 *  - vertex_isdf_svd_tol: "1e-8" Relative SVD cutoff on the secondary pair collocation
 *                 B(q) in the truncated pseudo-inverse solve for t(q) (the secondary
 *                 metric s = B^dag B is regularized at the square of this value).
 *  - vertex_isdf_thresh: "-1" Point-selection threshold for the secondary ISDF pivoted
 *                 Cholesky ("secondary" only). -1 defaults to the GLOBAL THC thresh
 *                 ([interaction.thc] thresh) so the selected interpolating vectors stay
 *                 in the span of the global basis; a tighter override over-resolves the
 *                 C pair-density metric and yields an ill-conditioned secondary metric.
 *  - vertex_div_treatment: "" q->0 rung policy for the VERTEX only; empty inherits
 *                 div_treatment. {choices: "ignore_g0", "gygi", "v1_skip"}. Lets the
 *                 analytic rank-1 head inserted into the vertex rungs be switched off
 *                 without changing the GW/HF divergence treatment -- that head is the one
 *                 piece the Sigma^C-vs-GF2-exchange absolute cross-check does not cover,
 *                 and notes/q0_head_treatment.md measures it at ~2.4x the body scale at
 *                 N_k = 8.
 *  - vertex_pidyn: "factorized" Route for B-L's equal-time dynamic-rung polarization
 *                 pi^dyn (vertex_rung = "linear" only; ignored otherwise).
 *                 {choices: "factorized", "kernel", "check"}. "factorized" evaluates
 *                 eq:pibardynfact directly -- ONE bosonic pairing of two bubbles against W,
 *                 no twisted pairs, no pole algebra. "kernel" restores the historic route
 *                 (run the full dynamic-rung Pi^C over every bosonic node, keep only the
 *                 tau = 0 row): it measured 98.9 % of B-L's vertex time and is B-L's only
 *                 contact with the aux pole basis, so it is a diagnostic, not a
 *                 recommendation. "check" runs BOTH and aborts if they disagree by more
 *                 than vertex_pidyn_tol -- the production-scale version of the refactor
 *                 gate that test_methods_vertex_pibardynfact runs on a toy.
 *  - vertex_pidyn_tol: "-1" ABORT bar for vertex_pidyn = "check". <=0 uses 0.25, an O(1)
 *                 routing bar. It is deliberately NOT tied to iaft eps: the two routes are
 *                 exact Matsubara sums of DIFFERENT integrands read through the same tau = 0
 *                 row, so their agreement floor is a REPRESENTABILITY floor whose prefactor
 *                 grows with beta*wmax (MEASURED ~30*eps at 160, ~2000*eps at 6000) AND is
 *                 data dependent (LiH-222 at prec = "low": 3.6e-03 in scf iteration 1,
 *                 2.1e-02 in iteration 2). An eps-derived abort would therefore be flaky by
 *                 construction. What the check DOES discriminate is a routing/plumbing break,
 *                 and every mis-routing the routing pin rejects is O(1) (closest control
 *                 1.24). Exceeding max(1e-8, 100*iaft eps) instead emits a WARNING, which is
 *                 the actionable statement: pi^dyn is grid-limited -- at prec = "low" it is
 *                 only good to ~1e-2 BY EITHER ROUTE -- and the lever is iaft prec, not
 *                 vertex_pidyn.
 *  - vertex_scale: "1.0" Scale BOTH cuts by lambda, i.e. Phi_2^C -> lambda Phi_2^C.
 *                 Conservation stays exact at every lambda because the scaling acts on the
 *                 generating functional, not on the already-cut Sigma/P.
 *  - vertex_ramp_iters: "0" If > 0, walk lambda up to vertex_scale over this many scf
 *                 iterations (lambda_n = vertex_scale * min(1, n/ramp)). P^C carries no
 *                 sign guarantee, so a full-strength vertex can drive eps = I - Z.Pi
 *                 through zero and break the W-Dyson solve; ramping locates the largest
 *                 lambda whose solution still has a positive-definite eps. Watch the
 *                 "dielectric conditioning" line in the update_w log.
 *  - vertex_isdf_cond_max: "-1" Per-q conditioning cap on the secondary downfold
 *                 ("secondary" only). <=0 disables it (solve uses vertex_isdf_svd_tol). >0
 *                 sets the per-q least-squares rcond = 1/sqrt(cond_max) so each transfer q's
 *                 downfold is conditioned to <= cond_max. The ill-conditioning is q-specific
 *                 (not at Gamma) with a shared point set, so it is capped in the SOLVE, not
 *                 by pruning interpolating vectors.
 *  - vertex_wannier_file: "" Path to a TRIQS-compatible wan.h5 (proj_mat + band_window;
 *                 gw solver only). When set, the vertex subspace C becomes the span of
 *                 the M Wannier orbitals |w_a(k)> = sum_i U_ia(k)|psi_i(k)> read from the
 *                 file (a general fixed projector P = U U^dag, notes/
 *                 wannier_projector_theory.md), replacing the vertex_band_window C. The
 *                 committed theory is already projector-general; window mode is the
 *                 U = 1_window limit and stays bit-identical. U is FIXED for the whole
 *                 SCF loop (re-Wannierization = a restart). If a gw_edmft embedding
 *                 projector is also active it must be the SAME file (demand D2). The
 *                 rotated point selection of vertex_isdf = "secondary" is nosym-only.
 *  - vertex_wannier_loewdin: "true" Loewdin-orthonormalize U at load so U^dag U = 1_M
 *                 exactly (deterministic, gauge-covariant; the correction norm is
 *                 logged). false = proceed with the raw disentangled U (warn; P then
 *                 only approximately idempotent).
 *  - cvv_rspace_tol: "1e-6" R-shell truncation tolerance of the CVV head's
 *                 Sigma(R, iw) store (div_treatment = "cvv" only; increment C1). The
 *                 default is calibrated from the T6 R-decay diagnostic.
 *  - pol_vertex: "none" scGW-tilde ladder polarization (gw solver only;
 *                 notes/scgwt_implementation_plan.md L1-L3). {choices: "none", "ladder"}.
 *                 "ladder" resums the density-channel BSE with the static screened
 *                 kernel in the secondary-ISDF pair basis,
 *                 Pi-bar = [1 - Pi-bar^0 K-bar]^-1 Pi-bar^0, injected into P beside the
 *                 RPA bubble. P-ONLY: Sigma stays GW-form, so the production loop is NOT
 *                 Phi-derivable (deliberate; user ruling 2026-08-10 -- accurate screening
 *                 over Phi-derivability). Excludes an ACTIVE vertex_type (double-count
 *                 guard, ruling R5) and requires the DLR IAFT backend. An empty ladder
 *                 C-window is an exact no-op. Scaffolded (increment C0); an ACTIVE
 *                 ladder ABORTS until L1-L3 land.
 *  - pol_vertex_kernel: "w0_prev" Ladder kernel source (ruling R4). {choices: "w0_prev",
 *                 "w0_frozen"}. "w0_prev" takes K-bar = W-bar_0 from the previous
 *                 iteration's W (matches the static-rung convention); "w0_frozen" keeps
 *                 the RPA@KS W_0 (scGW_0-flavored).
 *  - pol_vertex_band_window, pol_vertex_isdf_rank, pol_vertex_isdf_svd_tol,
 *    pol_vertex_isdf_thresh, pol_vertex_isdf_cond_max, pol_vertex_isdf_distr_tol:
 *                 the ladder's C-window and secondary-basis knobs. Each key ABSENT
 *                 inherits the corresponding vertex_* value, so a ladder run on top of
 *                 an existing vertex input needs only pol_vertex = "ladder".
 */
template<typename eri_t>
void mbpt(std::string solver_type, eri_t &eri, ptree const& pt)
{
  auto mf = eri.corr_eri->get().MF();
  auto& mpi = eri.corr_eri->get().mpi();
  if (mpi->comm.size()%mpi->node_comm.size()!=0) {
    APP_ABORT("MBPT: number of processors on each node should be the same.");
  }
  std::string err = std::string("mbpt - Incorrect input - ");
  auto div_treatment = io::get_value_with_default<std::string>(pt, "div_treatment", "gygi");
  auto hf_div_treatment = io::get_value_with_default<std::string>(pt, "hf_div_treatment", "gygi");
  io::tolower(div_treatment);
  io::tolower(hf_div_treatment);

  auto niter = io::get_value_with_default<int>(pt,"niter",1);
  auto conv_thr = io::get_value_with_default<double>(pt,"conv_thr",1e-8);
  auto const_mu = io::get_value_with_default<bool>(pt,"const_mu",false);
  auto mu_tol = io::get_value_with_default<double>(pt,"mu_tolerance", 1e-9);
  auto output = resolve_mbpt_output_stem(pt);
  auto mu_update_alg = io::get_value_with_default<std::string>(pt, "mu_update_alg", "midpoint");

  auto restart = io::get_value_with_default<bool>(pt,"restart",false);
  auto greens_func_source = io::get_value_with_default<std::string>(pt,"greens_func_source", "scf");
  auto greens_func_iteration = io::get_value_with_default<long>(pt, "greens_func_iteration", -1);

  bool chkpt_exist = std::filesystem::exists(output + ".mbpt.h5");
  if (restart and !chkpt_exist) {
    restart = false;
    app_log(1, "");
    app_log(1, "╔══════════════════════════════════════════════════════════╗");
    app_log(1, "║ [ WARNING ]                                              ║");
    app_log(1, "║ Running in restart mode while the checkpoint HDF5 does   ║");
    app_log(1, "║ not exist. Switching to the start-from-scratch mode.     ║");
    app_log(1, "╚══════════════════════════════════════════════════════════╝\n");
  } else if (not restart and chkpt_exist) {
    app_log(1, "");
    app_log(1, "╔══════════════════════════════════════════════════════════╗");
    app_log(1, "║ [ WARNING ]                                              ║");
    app_log(1, "║ An existing CoQuí checkpoint HDF5 with the same prefix   ║");
    app_log(1, "║ has been detected even though CoQuí is running in the    ║");
    app_log(1, "║ start-from-scratch mode. --> The old checkpoint will be  ║");
    app_log(1, "║ overwritten. Considering move the old HDF5 or change the ║");
    app_log(1, "║ prefix next time.                                        ║");
    app_log(1, "╚══════════════════════════════════════════════════════════╝\n");
  }

  imag_axes_ft::IAFT ft(
    !restart ? imag_axes_ft::IAFT(pt, false, mf::wmax_from_mf(*mf))
             : imag_axes_ft::read_iaft(output+".mbpt.h5", false)
  );

  std::unique_ptr<iter_scf::iter_scf_t> iter_solver;

  using namespace solvers;
  hf_t hf(hf_div_treatment);
  if(solver_type == "rpa") {

    simple_dyson dyson(mf.get(), &ft, mu_tol, mu_update_alg);
    gw_t gw(&ft, div_treatment, output);
    MBState mb_state(mpi, ft, output);
    rpa_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf,&gw));

  } else if(solver_type == "hf") {

    simple_dyson dyson(mf.get(), &ft, mu_tol, mu_update_alg);
    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt));
    } else {
      iter_solver = nullptr;
    }
    MBState mb_state(mpi, ft, output);
    scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf),
             iter_solver.get(), niter, restart, conv_thr, const_mu,
             greens_func_source, greens_func_iteration);

  } else if(solver_type == "gw") {
    auto screen_type = io::get_value_with_default<std::string>(pt,"screen_type", "rpa");

    // optional second-order-exchange vertex correction (ISDF-Vertex)
    auto vertex_type = io::get_value_with_default<std::string>(pt,"vertex_type","none");
    io::tolower(vertex_type);
    // Rung mode of the vertex (notes/static_vertex_implementation_plan.md section 2.1):
    // "dynamic" (default) = the parent Formulation B; "static"/"linear" = the B-S/B-L
    // static-rung truncations. Validated in the vertex_t constructor, which also aborts
    // for an ACTIVE static/linear vertex until its kernels land (increment S2+).
    auto vertex_rung = io::get_value_with_default<std::string>(pt,"vertex_rung","dynamic");
    io::tolower(vertex_rung);
    auto vertex_band_window = io::get_value_with_default<nda::range>(pt,"vertex_band_window",nda::range(0,0));
    // Refinement 2 (secondary ISDF on C; notes/refinement2_optionA.md)
    auto vertex_isdf = io::get_value_with_default<std::string>(pt,"vertex_isdf","global");
    io::tolower(vertex_isdf);
    auto vertex_isdf_rank = io::get_value_with_default<long>(pt,"vertex_isdf_rank",-1);
    auto vertex_isdf_svd_tol = io::get_value_with_default<double>(pt,"vertex_isdf_svd_tol",1e-8);
    // Secondary-ISDF point-selection thresh (-1 = default to the global THC thresh)
    auto vertex_isdf_thresh = io::get_value_with_default<double>(pt,"vertex_isdf_thresh",-1.0);
    // distr_tol for the SECONDARY basis' private thc builder (rank-cap lift; <= 0 =
    // builder default 0.2, today's behavior). 1.0 lifts kp444/M8 to 260 ranks.
    auto vertex_isdf_distr_tol = io::get_value_with_default<double>(pt,"vertex_isdf_distr_tol",-1.0);
    // Conditioning cap on the secondary metric s(q) (<=0 = disabled). >0 prunes the
    // near-dependent tail of the selected basis so cond(s) stays under this value.
    auto vertex_isdf_cond_max = io::get_value_with_default<double>(pt,"vertex_isdf_cond_max",-1.0);
    // Phi-scaling of the vertex: BOTH cuts by the same lambda == Phi_2^C -> lambda Phi_2^C,
    // so conservation is exact at every lambda (the approximation acts on Phi, not on the
    // cuts). vertex_ramp_iters > 0 walks lambda from vertex_scale/ramp up to vertex_scale
    // over that many scf iterations. Used to keep eps = I - Z.Pi positive definite: P^C is
    // not sign-definite (notes/vertex_divergence_diagnosis.md).
    // q->0 rung policy for the VERTEX, independent of the GW/HF one. The gygi head
    // inserted into the vertex rungs is a rank-1 Nk*madelung*chi.chi^dag block that the
    // q0 memo measured at ~2.4x the body scale on LiH-222 (Nk = 8), large enough to flip
    // the sign of the traced Phi_2^C. It is also the ONE component the Sigma^C-vs-GF2
    // absolute cross-check does not cover (that test runs "ignore_g0"). Being able to
    // switch it WITHOUT changing the GW/HF divergence treatment makes it separable.
    // Empty string = inherit div_treatment (previous behavior).
    auto vertex_div_treatment = io::get_value_with_default<std::string>(pt,"vertex_div_treatment","");
    io::tolower(vertex_div_treatment);
    auto vertex_scale = io::get_value_with_default<double>(pt,"vertex_scale",1.0);
    auto vertex_ramp_iters = io::get_value_with_default<long>(pt,"vertex_ramp_iters",0);
    // B-L's pi^dyn route (eq:pibardynfact). "factorized" is the production route; "kernel"
    // restores the historic full-dynamic-Pi^C-then-tau=0 path; "check" runs both and gates.
    auto vertex_pidyn = io::get_value_with_default<std::string>(pt,"vertex_pidyn","factorized");
    io::tolower(vertex_pidyn);
    auto vertex_pidyn_tol = io::get_value_with_default<double>(pt,"vertex_pidyn_tol",-1.0);
    // Project the rank-1 head channel out of the response middle factor at q = Gamma.
    // 🚨 DEFAULT false SINCE 2026-07-31: it BREAKS PHI-DERIVABILITY (the B-L G-side oracle
    // goes from 3.3e-11 to 1.6e-01 when only 20 % of the channel is removed) and is applied
    // to the Sigma cut only, leaving eval_Pi_C's P^{C,L} untouched. It does control the
    // COLD-START basin, so it survives as a diagnostic. See vertex_t.h and
    // notes/bl_head_channel_diagnosis.md.
    auto vertex_bl_head_projection =
        io::get_value_with_default<bool>(pt,"vertex_bl_head_projection",false);
    // DIAGNOSTIC, default false, NOT physical: freeze pi^dyn's Gamma rung head at its
    // i.nu = 0 weight. Tests whether the head's retardation is what breaks pi^dyn's
    // q -> 0 head suppression (which Pi^{C,0}, whose head IS frozen, satisfies exactly).
    auto vertex_bl_static_head =
        io::get_value_with_default<bool>(pt,"vertex_bl_static_head",false);
    // DIAGNOSTIC, default false, changes the KERNEL DEFINITION: take W0's Gamma head
    // weight from the same (vertex-corrected) eps^-1 that W's head uses, instead of from
    // W0's own RPA-only Dyson. Makes the head part of W(Gamma,0) - W0(Gamma) vanish; that
    // head residue is 94 % of the whole measured |W(q,0) - W0(q)| on Si. See vertex_t.h.
    auto vertex_bl_w0_head_from_w =
        io::get_value_with_default<bool>(pt,"vertex_bl_w0_head_from_w",false);
    // DIAGNOSTIC, default false, NOT physical: THE CONSTANT-RUNG ABSOLUTE PIN. Replaces
    // pi^dyn's dynamic rung by the frequency-independent W0bar - Z, so pi^dyn's rung IS
    // Pi^{C,0}'s rung and X^L must collapse to the DLR representability floor. A residual
    // O(1) X^L convicts the equal-time path itself. See vertex_t.h.
    auto vertex_bl_pidyn_const_rung =
        io::get_value_with_default<bool>(pt,"vertex_bl_pidyn_const_rung",false);
    // H1, the BALANCED FIRST-ORDER HEAD (default false pending the Gate-0/Gate-1 record
    // and a defaults ruling; notes/bl_head_balance_theory_and_plan.md). In B-L only:
    // every W input of the vertex functional carries W0's STATIC Gamma-head weight
    // (instantaneous slot, 1 + eps_inv_head(i.nu=0); no dynamic-slot head), so the
    // fluctuation dW = W - W0 carries no analytic head. CONSERVING (a modified
    // interaction in Phi -- both cuts differentiate). B-S is bit-identical; the parent
    // theory keeps its retarded head.
    auto vertex_bl_head_static_all =
        io::get_value_with_default<bool>(pt,"vertex_bl_head_static_all",false);
    // Wannier-projector subspace C (notes/wannier_projector_theory.md): when set, the
    // vertex subspace is span{ w_a(k) } from a TRIQS-compatible wan.h5 (proj_mat +
    // band_window) instead of the band window; U is Loewdin-orthonormalized at load.
    auto vertex_wannier_file = io::get_value_with_default<std::string>(pt,"vertex_wannier_file","");
    auto vertex_wannier_loewdin = io::get_value_with_default<bool>(pt,"vertex_wannier_loewdin",true);
    // scGW-tilde knob surface (notes/scgwt_implementation_plan.md section 1).
    // div_treatment = "cvv" is live since increment C4 (the covariant-velocity head
    // fill in update_w). The pol_vertex_* basis knobs inherit the vertex_* values when
    // their keys are absent, so a ladder run on top of an existing vertex input needs
    // only pol_vertex = "ladder".
    auto cvv_rspace_tol = io::get_value_with_default<double>(pt,"cvv_rspace_tol",1e-6);
    auto pol_vertex = io::get_value_with_default<std::string>(pt,"pol_vertex","none");
    io::tolower(pol_vertex);
    auto pol_vertex_kernel = io::get_value_with_default<std::string>(pt,"pol_vertex_kernel","w0_prev");
    io::tolower(pol_vertex_kernel);
    auto pol_vertex_band_window = io::get_value_with_default<nda::range>(pt,"pol_vertex_band_window",vertex_band_window);
    auto pol_vertex_isdf_rank = io::get_value_with_default<long>(pt,"pol_vertex_isdf_rank",vertex_isdf_rank);
    auto pol_vertex_isdf_svd_tol = io::get_value_with_default<double>(pt,"pol_vertex_isdf_svd_tol",vertex_isdf_svd_tol);
    auto pol_vertex_isdf_thresh = io::get_value_with_default<double>(pt,"pol_vertex_isdf_thresh",vertex_isdf_thresh);
    auto pol_vertex_isdf_cond_max = io::get_value_with_default<double>(pt,"pol_vertex_isdf_cond_max",vertex_isdf_cond_max);
    auto pol_vertex_isdf_distr_tol = io::get_value_with_default<double>(pt,"pol_vertex_isdf_distr_tol",vertex_isdf_distr_tol);

    simple_dyson dyson(mf.get(), &ft, mu_tol, mu_update_alg);
    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt));
    } else {
      iter_solver = nullptr;
    }
    solvers::scr_coulomb_t scr_eri(&ft, screen_type, div_treatment);
    solvers::gw_t gw(&ft, div_treatment, output);

    // vertex_t must outlive the scf_loop below. Both cuts (Sigma^C and Pi^C)
    // are switched together through this single object -- never one alone.
    solvers::vertex_t vertex(&ft, vertex_type, vertex_band_window, mf->nbnd(), div_treatment,
                             vertex_isdf, vertex_isdf_rank, vertex_isdf_svd_tol, vertex_isdf_thresh,
                             vertex_isdf_cond_max, vertex_rung);
    vertex.set_vertex_scale(vertex_scale, vertex_ramp_iters);
    vertex.set_pidyn_mode(vertex_pidyn, vertex_pidyn_tol);
    vertex.set_bl_head_projection(vertex_bl_head_projection);
    vertex.set_bl_static_head(vertex_bl_static_head);
    vertex.set_bl_w0_head_from_w(vertex_bl_w0_head_from_w);
    vertex.set_bl_pidyn_const_rung(vertex_bl_pidyn_const_rung);
    vertex.set_bl_head_static_all(vertex_bl_head_static_all);
    vertex.set_isdf_distr_tol(vertex_isdf_distr_tol);
    if (not vertex_div_treatment.empty()) vertex.set_div_treatment(vertex_div_treatment);
    // scGW-tilde (C0): validate + store the ladder knobs (double-count guard and the
    // not-implemented abort for an ACTIVE ladder live in the setter) and hand the CVV
    // R-shell tolerance to the W builder for increment C4.
    vertex.set_pol_vertex(pol_vertex, pol_vertex_kernel, pol_vertex_band_window,
                          pol_vertex_isdf_rank, pol_vertex_isdf_svd_tol,
                          pol_vertex_isdf_thresh, pol_vertex_isdf_cond_max,
                          pol_vertex_isdf_distr_tol);
    scr_eri.set_cvv_rspace_tol(cvv_rspace_tol);
    if (vertex.enabled()) {
      utils::check(screen_type == "rpa" or screen_type == "rpa_k",
                   "vertex_type = \"{}\" currently requires screen_type = \"rpa\" or \"rpa_k\" "
                   "(got \"{}\"): combining the vertex correction with cRPA/EDMFT screening "
                   "is not validated (Phi-derivability of the combination unestablished).",
                   vertex.vertex_type(), screen_type);
      // WANNIER MODE (notes/wannier_projector_theory.md P1): build the projector from
      // wan.h5 and install U. Demand D2 (one U per run): if a gw_edmft embedding
      // projector is also active, the vertex must consume the SAME wan.h5 or abort.
      if (not vertex_wannier_file.empty()) {
        if (screen_type.substr(0,8) == "gw_edmft") {
          auto embed_file = io::get_value_with_default<std::string>(pt,"wannier_file","");
          utils::check(embed_file == vertex_wannier_file,
                       "vertex_wannier_file = \"{}\" differs from the gw_edmft embedding "
                       "wannier_file = \"{}\": one projector P per run is required (demand "
                       "D2, notes/wannier_projector_theory.md section 1.5); use the SAME "
                       "wan.h5 for both.", vertex_wannier_file, embed_file);
        }
        auto vtx_trans_home = io::get_value_with_default<bool>(pt,"translate_home_cell",false);
        methods::projector_t proj(*mf, vertex_wannier_file, vtx_trans_home);
        vertex.set_wannier_projector(proj, vertex_wannier_loewdin);
      }
      scr_eri.set_vertex(&vertex);
      gw.set_vertex(&vertex);
    } else if (vertex.pol_vertex_enabled()) {
      // scGW-tilde L2: a pol-vertex-only run (vertex_type = "none") attaches the knob
      // carrier to scr_eri so update_w can run the ladder READOUT. Never attached to
      // gw -- Sigma stays GW-form, and has_active_vertex() stays false (no injection).
      scr_eri.set_vertex(&vertex);
    }

    if (screen_type.substr(0,8)=="gw_edmft") {

      auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
      auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);

      MBState mb_state(ft, output, mf, wannier_file, trans_home_cell);
      scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gw, &scr_eri),
               iter_solver.get(), niter, restart, conv_thr, const_mu,
               greens_func_source, greens_func_iteration);

      auto dump_w_to_h5 = io::get_value_with_default<bool>(pt,"dump_w_to_h5", false);
      if (dump_w_to_h5) {
        auto& W_qtPQ = mb_state.dW_qtPQ.value();
        if (mb_state.mpi->comm.root()) {
          h5::file file("thc_screened_interaction.h5", 'w');
          h5::group grp(file);
          math::nda::h5_write(grp, "W_qtPQ", W_qtPQ);
        } else {
          h5::group grp;
          math::nda::h5_write(grp, "W_qtPQ", W_qtPQ);
        }
      }
    } else {

      MBState mb_state(mpi, ft, output);
      scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gw, &scr_eri),
               iter_solver.get(), niter, restart, conv_thr, const_mu,
               greens_func_source, greens_func_iteration);

      auto dump_w_to_h5 = io::get_value_with_default<bool>(pt,"dump_w_to_h5", false);
      if (dump_w_to_h5) {
        auto& W_qtPQ = mb_state.dW_qtPQ.value();
        if (mb_state.mpi->comm.root()) {
          h5::file file("thc_screened_interaction.h5", 'w');
          h5::group grp(file);
          math::nda::h5_write(grp, "W_qtPQ", W_qtPQ);
        } else {
          h5::group grp;
          math::nda::h5_write(grp, "W_qtPQ", W_qtPQ);
        }
      }
    }

  } else if(solver_type == "gf2") {

    auto gf2_direct_type = io::get_value_with_default<std::string>(pt,"gf2_direct_type","gf2");
    auto gf2_exchange_alg = io::get_value_with_default<std::string>(pt,"gf2_exchange_alg","orb");
    auto gf2_exchange_type = io::get_value_with_default<std::string>(pt,"gf2_exchange_type","gf2");
    auto gf2_save_C = io::get_value_with_default<bool>(pt,"gf2_save_C",true);
    auto gf2_sosex_save_memory = io::get_value_with_default<bool>(pt,"gf2_sosex_save_memory",true);
    auto t_prescreen_thresh = io::get_value_with_default<double>(pt,"t_prescreen_thresh",0.0);

    simple_dyson dyson(mf.get(), &ft, mu_tol, mu_update_alg);
    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt));
    } else {
      iter_solver = nullptr;
    }
    solvers::gf2_t gf2(mf.get(), &ft, div_treatment,
                       gf2_direct_type, gf2_exchange_alg, gf2_exchange_type, output,
                       gf2_save_C, gf2_sosex_save_memory);
    gf2.t_thresh() = t_prescreen_thresh;

    MBState mb_state(mpi, ft, output);

    if (gf2_direct_type == "gf2") {
      scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gf2),
               iter_solver.get(), niter, restart, conv_thr, const_mu,
               greens_func_source, greens_func_iteration);
    } else {
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", div_treatment);
      scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gf2, &scr_eri),
               iter_solver.get(), niter, restart, conv_thr, const_mu,
               greens_func_source, greens_func_iteration);
    }

  } else if(solver_type == "gw_dca") {

    utils::check(false, "mbpt: gw_dca is not implemented!");
    /*ptree dca_pt = io::find_child(pt, "gw_mean_field");
    std::string mf_type = (mf.mf_type()==mf::qe_source)?
        "qe" : (mf.mf_type()==mf::pyscf_source)? "pyscf" : "bdft";
    mf::MF dca_mf(mf::make_MF(mpi, dca_pt, mf_type));
    dca_dyson dyson(mpi, &mf, &ft, dca_mf);
    solvers::gw_t gw(&ft, div_treatment, output);
    scf_loop(dyson, eri, ft, mb_solver_t(&hf,&gw), nullptr,
             output, niter, restart, conv_thr, const_mu);*/

  } else if (solver_type == "qphf") {

    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt));
    } else {
      iter_solver = nullptr;
    }
    MBState mb_state(mpi, ft, output);
    qp_params_t qp_params;
    qp_params.mu_tolerance = mu_tol;
    qp_params.mu_update_alg = mu_update_alg;
    qp_scf_loop(mb_state, eri, ft, qp_params, mb_solver_t(&hf), iter_solver.get(),
                niter, restart, conv_thr);

  } else if (solver_type == "evgw") {

    auto keep_scr_coulomb_fixed = io::get_value_with_default<bool>(pt,"keep_scr_coulomb_fixed", false);
    auto qp_type = io::get_value_with_default<std::string>(pt,"qp_type","sc");
    auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
    auto eta     = io::get_value_with_default<double>(pt,"eta", M_PI/ft.beta());
    auto Nfit    = io::get_value_with_default<int>(pt,"Nfit",18);
    io::tolower(ac_alg);
    io::tolower(qp_type);
    qp_params_t qp_params(qp_type, ac_alg, Nfit, eta, conv_thr, "evscf", keep_scr_coulomb_fixed,
                          "fermi", mu_tol, mu_update_alg);
    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt, 0.7, true));
    } else {
      iter_solver = nullptr;
    }
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", div_treatment);
    solvers::gw_t gw(&ft, div_treatment, output);
    MBState mb_state(mpi, ft, output);
    qp_scf_loop(mb_state, eri, ft, qp_params, mb_solver_t(&hf,&gw,&scr_eri), iter_solver.get(),
                niter, restart, conv_thr);

  } else if (solver_type == "qpgw") {

    auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
    auto eta     = io::get_value_with_default<double>(pt,"eta", M_PI/ft.beta());
    auto Nfit    = io::get_value_with_default<int>(pt,"Nfit",18);
    auto off_diag_mode = io::get_value_with_default<std::string>(pt,"off_diag_mode","fermi");
    io::tolower(ac_alg);
    io::tolower(off_diag_mode);
    utils::check(off_diag_mode=="fermi" or off_diag_mode=="qp_energy",
                 "unknown off_diag_mode: {}. Valid options are \"fermi\" and \"qp_energy\"");
    qp_params_t qp_params("sc", ac_alg, Nfit, eta, 1e-8, "qpscf", false, off_diag_mode,
                          mu_tol, mu_update_alg);
    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt));
    } else {
      iter_solver = nullptr;
    }
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", div_treatment);
    solvers::gw_t gw(&ft, div_treatment, output);
    MBState mb_state(mpi, ft, output);
    qp_scf_loop(mb_state, eri, ft, qp_params, mb_solver_t(&hf,&gw,&scr_eri), iter_solver.get(),
                niter, restart, conv_thr);

  } else
    APP_ABORT("mbpt: Unknown solver type: {}",solver_type);
}


template<typename eri_t>
void mbpt(std::string solver_type, eri_t &eri, ptree const& pt,
          nda::array<ComplexType, 5> const& projector_ksIai,
          nda::array<long, 3> const& band_window,
          nda::array<RealType, 2> const& kpts_crys,
          std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities)
{
  auto mf = eri.corr_eri->get().MF();
  auto& mpi = eri.corr_eri->get().mpi();
  if (mpi->comm.size()%mpi->node_comm.size()!=0) {
    APP_ABORT("MBPT: number of processors on each node should be the same.");
  }
  std::string err = std::string("mbpt - Incorrect input - ");
  auto div_treatment = io::get_value_with_default<std::string>(pt, "div_treatment", "gygi");
  auto hf_div_treatment = io::get_value_with_default<std::string>(pt, "hf_div_treatment", "gygi");
  io::tolower(div_treatment);
  io::tolower(hf_div_treatment);

  auto niter = io::get_value_with_default<int>(pt,"niter",1);
  auto conv_thr = io::get_value_with_default<double>(pt,"conv_thr",1e-8);
  auto const_mu = io::get_value_with_default<bool>(pt,"const_mu",false);
  auto mu_tol = io::get_value_with_default<double>(pt,"mu_tolerance", 1e-9);
  auto output = resolve_mbpt_output_stem(pt);
  auto mu_update_alg = io::get_value_with_default<std::string>(pt, "mu_update_alg", "midpoint");

  auto restart = io::get_value_with_default<bool>(pt,"restart",false);
  auto greens_func_source = io::get_value_with_default<std::string>(pt,"greens_func_source", "scf");
  auto greens_func_iteration = io::get_value_with_default<long>(pt, "greens_func_iteration", -1);
  bool chkpt_exist = std::filesystem::exists(output + ".mbpt.h5");
  if (restart and !chkpt_exist) {
    restart = false;
    app_log(1, "");
    app_log(1, "╔══════════════════════════════════════════════════════════╗");
    app_log(1, "║ [ WARNING ]                                              ║");
    app_log(1, "║ Running in restart mode while the checkpoint HDF5 does   ║");
    app_log(1, "║ not exist. Switching to the start-from-scratch mode.     ║");
    app_log(1, "╚══════════════════════════════════════════════════════════╝\n");
  } else if (not restart and chkpt_exist) {
    app_log(1, "");
    app_log(1, "╔══════════════════════════════════════════════════════════╗");
    app_log(1, "║ [ WARNING ]                                              ║");
    app_log(1, "║ An existing CoQuí checkpoint HDF5 with the same prefix   ║");
    app_log(1, "║ has been detected even though CoQuí is running in the    ║");
    app_log(1, "║ start-from-scratch mode. --> The old checkpoint will be  ║");
    app_log(1, "║ overwritten. Considering move the old HDF5 or change the ║");
    app_log(1, "║ prefix next time.                                        ║");
    app_log(1, "╚══════════════════════════════════════════════════════════╝\n");
  }

  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);

  imag_axes_ft::IAFT ft(
    !restart ? imag_axes_ft::IAFT(pt, false, mf::wmax_from_mf(*mf))
             : imag_axes_ft::read_iaft(output+".mbpt.h5", false)
  );

  std::unique_ptr<iter_scf::iter_scf_t> iter_solver;

  using namespace solvers;
  hf_t hf(hf_div_treatment);
  if (solver_type == "gw") {

    auto screen_type = io::get_value_with_default<std::string>(pt,"screen_type", "rpa");

    // optional second-order-exchange vertex correction (ISDF-Vertex)
    auto vertex_type = io::get_value_with_default<std::string>(pt,"vertex_type","none");
    io::tolower(vertex_type);
    // Rung mode of the vertex (notes/static_vertex_implementation_plan.md section 2.1):
    // "dynamic" (default) = the parent Formulation B; "static"/"linear" = the B-S/B-L
    // static-rung truncations. Validated in the vertex_t constructor, which also aborts
    // for an ACTIVE static/linear vertex until its kernels land (increment S2+).
    auto vertex_rung = io::get_value_with_default<std::string>(pt,"vertex_rung","dynamic");
    io::tolower(vertex_rung);
    auto vertex_band_window = io::get_value_with_default<nda::range>(pt,"vertex_band_window",nda::range(0,0));
    // Refinement 2 (secondary ISDF on C; notes/refinement2_optionA.md)
    auto vertex_isdf = io::get_value_with_default<std::string>(pt,"vertex_isdf","global");
    io::tolower(vertex_isdf);
    auto vertex_isdf_rank = io::get_value_with_default<long>(pt,"vertex_isdf_rank",-1);
    auto vertex_isdf_svd_tol = io::get_value_with_default<double>(pt,"vertex_isdf_svd_tol",1e-8);
    // Secondary-ISDF point-selection thresh (-1 = default to the global THC thresh)
    auto vertex_isdf_thresh = io::get_value_with_default<double>(pt,"vertex_isdf_thresh",-1.0);
    // distr_tol for the SECONDARY basis' private thc builder (rank-cap lift; <= 0 =
    // builder default 0.2, today's behavior). 1.0 lifts kp444/M8 to 260 ranks.
    auto vertex_isdf_distr_tol = io::get_value_with_default<double>(pt,"vertex_isdf_distr_tol",-1.0);
    // Conditioning cap on the secondary metric s(q) (<=0 = disabled). >0 prunes the
    // near-dependent tail of the selected basis so cond(s) stays under this value.
    auto vertex_isdf_cond_max = io::get_value_with_default<double>(pt,"vertex_isdf_cond_max",-1.0);
    // Phi-scaling of the vertex: BOTH cuts by the same lambda == Phi_2^C -> lambda Phi_2^C,
    // so conservation is exact at every lambda (the approximation acts on Phi, not on the
    // cuts). vertex_ramp_iters > 0 walks lambda from vertex_scale/ramp up to vertex_scale
    // over that many scf iterations. Used to keep eps = I - Z.Pi positive definite: P^C is
    // not sign-definite (notes/vertex_divergence_diagnosis.md).
    // q->0 rung policy for the VERTEX, independent of the GW/HF one. The gygi head
    // inserted into the vertex rungs is a rank-1 Nk*madelung*chi.chi^dag block that the
    // q0 memo measured at ~2.4x the body scale on LiH-222 (Nk = 8), large enough to flip
    // the sign of the traced Phi_2^C. It is also the ONE component the Sigma^C-vs-GF2
    // absolute cross-check does not cover (that test runs "ignore_g0"). Being able to
    // switch it WITHOUT changing the GW/HF divergence treatment makes it separable.
    // Empty string = inherit div_treatment (previous behavior).
    auto vertex_div_treatment = io::get_value_with_default<std::string>(pt,"vertex_div_treatment","");
    io::tolower(vertex_div_treatment);
    auto vertex_scale = io::get_value_with_default<double>(pt,"vertex_scale",1.0);
    auto vertex_ramp_iters = io::get_value_with_default<long>(pt,"vertex_ramp_iters",0);
    // B-L's pi^dyn route (eq:pibardynfact). "factorized" is the production route; "kernel"
    // restores the historic full-dynamic-Pi^C-then-tau=0 path; "check" runs both and gates.
    auto vertex_pidyn = io::get_value_with_default<std::string>(pt,"vertex_pidyn","factorized");
    io::tolower(vertex_pidyn);
    auto vertex_pidyn_tol = io::get_value_with_default<double>(pt,"vertex_pidyn_tol",-1.0);
    // Project the rank-1 head channel out of the response middle factor at q = Gamma.
    // 🚨 DEFAULT false SINCE 2026-07-31: it BREAKS PHI-DERIVABILITY (the B-L G-side oracle
    // goes from 3.3e-11 to 1.6e-01 when only 20 % of the channel is removed) and is applied
    // to the Sigma cut only, leaving eval_Pi_C's P^{C,L} untouched. It does control the
    // COLD-START basin, so it survives as a diagnostic. See vertex_t.h and
    // notes/bl_head_channel_diagnosis.md.
    auto vertex_bl_head_projection =
        io::get_value_with_default<bool>(pt,"vertex_bl_head_projection",false);
    // DIAGNOSTIC, default false, NOT physical: freeze pi^dyn's Gamma rung head at its
    // i.nu = 0 weight. Tests whether the head's retardation is what breaks pi^dyn's
    // q -> 0 head suppression (which Pi^{C,0}, whose head IS frozen, satisfies exactly).
    auto vertex_bl_static_head =
        io::get_value_with_default<bool>(pt,"vertex_bl_static_head",false);
    // DIAGNOSTIC, default false, changes the KERNEL DEFINITION: take W0's Gamma head
    // weight from the same (vertex-corrected) eps^-1 that W's head uses, instead of from
    // W0's own RPA-only Dyson. Makes the head part of W(Gamma,0) - W0(Gamma) vanish; that
    // head residue is 94 % of the whole measured |W(q,0) - W0(q)| on Si. See vertex_t.h.
    auto vertex_bl_w0_head_from_w =
        io::get_value_with_default<bool>(pt,"vertex_bl_w0_head_from_w",false);
    // DIAGNOSTIC, default false, NOT physical: THE CONSTANT-RUNG ABSOLUTE PIN. Replaces
    // pi^dyn's dynamic rung by the frequency-independent W0bar - Z, so pi^dyn's rung IS
    // Pi^{C,0}'s rung and X^L must collapse to the DLR representability floor. A residual
    // O(1) X^L convicts the equal-time path itself. See vertex_t.h.
    auto vertex_bl_pidyn_const_rung =
        io::get_value_with_default<bool>(pt,"vertex_bl_pidyn_const_rung",false);
    // H1, the BALANCED FIRST-ORDER HEAD (default false pending the Gate-0/Gate-1 record
    // and a defaults ruling; notes/bl_head_balance_theory_and_plan.md). In B-L only:
    // every W input of the vertex functional carries W0's STATIC Gamma-head weight
    // (instantaneous slot, 1 + eps_inv_head(i.nu=0); no dynamic-slot head), so the
    // fluctuation dW = W - W0 carries no analytic head. CONSERVING (a modified
    // interaction in Phi -- both cuts differentiate). B-S is bit-identical; the parent
    // theory keeps its retarded head.
    auto vertex_bl_head_static_all =
        io::get_value_with_default<bool>(pt,"vertex_bl_head_static_all",false);
    // Wannier-projector subspace C (notes/wannier_projector_theory.md): when set, the
    // vertex subspace is span{ w_a(k) } from a TRIQS-compatible wan.h5 (proj_mat +
    // band_window) instead of the band window; U is Loewdin-orthonormalized at load.
    auto vertex_wannier_file = io::get_value_with_default<std::string>(pt,"vertex_wannier_file","");
    auto vertex_wannier_loewdin = io::get_value_with_default<bool>(pt,"vertex_wannier_loewdin",true);
    // scGW-tilde knob surface (notes/scgwt_implementation_plan.md section 1).
    // div_treatment = "cvv" is live since increment C4 (the covariant-velocity head
    // fill in update_w). The pol_vertex_* basis knobs inherit the vertex_* values when
    // their keys are absent, so a ladder run on top of an existing vertex input needs
    // only pol_vertex = "ladder".
    auto cvv_rspace_tol = io::get_value_with_default<double>(pt,"cvv_rspace_tol",1e-6);
    auto pol_vertex = io::get_value_with_default<std::string>(pt,"pol_vertex","none");
    io::tolower(pol_vertex);
    auto pol_vertex_kernel = io::get_value_with_default<std::string>(pt,"pol_vertex_kernel","w0_prev");
    io::tolower(pol_vertex_kernel);
    auto pol_vertex_band_window = io::get_value_with_default<nda::range>(pt,"pol_vertex_band_window",vertex_band_window);
    auto pol_vertex_isdf_rank = io::get_value_with_default<long>(pt,"pol_vertex_isdf_rank",vertex_isdf_rank);
    auto pol_vertex_isdf_svd_tol = io::get_value_with_default<double>(pt,"pol_vertex_isdf_svd_tol",vertex_isdf_svd_tol);
    auto pol_vertex_isdf_thresh = io::get_value_with_default<double>(pt,"pol_vertex_isdf_thresh",vertex_isdf_thresh);
    auto pol_vertex_isdf_cond_max = io::get_value_with_default<double>(pt,"pol_vertex_isdf_cond_max",vertex_isdf_cond_max);
    auto pol_vertex_isdf_distr_tol = io::get_value_with_default<double>(pt,"pol_vertex_isdf_distr_tol",vertex_isdf_distr_tol);

    simple_dyson dyson(mf.get(), &ft, mu_tol, mu_update_alg);
    if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
      iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt));
    } else {
      iter_solver = nullptr;
    }
    solvers::scr_coulomb_t scr_eri(&ft, screen_type, div_treatment);
    solvers::gw_t gw(&ft, div_treatment, output);

    // vertex_t must outlive the scf_loop below. Both cuts (Sigma^C and Pi^C)
    // are switched together through this single object -- never one alone.
    solvers::vertex_t vertex(&ft, vertex_type, vertex_band_window, mf->nbnd(), div_treatment,
                             vertex_isdf, vertex_isdf_rank, vertex_isdf_svd_tol, vertex_isdf_thresh,
                             vertex_isdf_cond_max, vertex_rung);
    vertex.set_vertex_scale(vertex_scale, vertex_ramp_iters);
    vertex.set_pidyn_mode(vertex_pidyn, vertex_pidyn_tol);
    vertex.set_bl_head_projection(vertex_bl_head_projection);
    vertex.set_bl_static_head(vertex_bl_static_head);
    vertex.set_bl_w0_head_from_w(vertex_bl_w0_head_from_w);
    vertex.set_bl_pidyn_const_rung(vertex_bl_pidyn_const_rung);
    vertex.set_bl_head_static_all(vertex_bl_head_static_all);
    vertex.set_isdf_distr_tol(vertex_isdf_distr_tol);
    if (not vertex_div_treatment.empty()) vertex.set_div_treatment(vertex_div_treatment);
    // scGW-tilde (C0): validate + store the ladder knobs (double-count guard and the
    // not-implemented abort for an ACTIVE ladder live in the setter) and hand the CVV
    // R-shell tolerance to the W builder for increment C4.
    vertex.set_pol_vertex(pol_vertex, pol_vertex_kernel, pol_vertex_band_window,
                          pol_vertex_isdf_rank, pol_vertex_isdf_svd_tol,
                          pol_vertex_isdf_thresh, pol_vertex_isdf_cond_max,
                          pol_vertex_isdf_distr_tol);
    scr_eri.set_cvv_rspace_tol(cvv_rspace_tol);
    if (vertex.enabled()) {
      utils::check(screen_type == "rpa" or screen_type == "rpa_k",
                   "vertex_type = \"{}\" currently requires screen_type = \"rpa\" or \"rpa_k\" "
                   "(got \"{}\"): combining the vertex correction with cRPA/EDMFT screening "
                   "is not validated (Phi-derivability of the combination unestablished).",
                   vertex.vertex_type(), screen_type);
      // WANNIER MODE (notes/wannier_projector_theory.md P1): build the projector from
      // wan.h5 and install U. Demand D2 (one U per run): if a gw_edmft embedding
      // projector is also active, the vertex must consume the SAME wan.h5 or abort.
      if (not vertex_wannier_file.empty()) {
        if (screen_type.substr(0,8) == "gw_edmft") {
          auto embed_file = io::get_value_with_default<std::string>(pt,"wannier_file","");
          utils::check(embed_file == vertex_wannier_file,
                       "vertex_wannier_file = \"{}\" differs from the gw_edmft embedding "
                       "wannier_file = \"{}\": one projector P per run is required (demand "
                       "D2, notes/wannier_projector_theory.md section 1.5); use the SAME "
                       "wan.h5 for both.", vertex_wannier_file, embed_file);
        }
        auto vtx_trans_home = io::get_value_with_default<bool>(pt,"translate_home_cell",false);
        methods::projector_t proj(*mf, vertex_wannier_file, vtx_trans_home);
        vertex.set_wannier_projector(proj, vertex_wannier_loewdin);
      }
      scr_eri.set_vertex(&vertex);
      gw.set_vertex(&vertex);
    } else if (vertex.pol_vertex_enabled()) {
      // scGW-tilde L2: a pol-vertex-only run (vertex_type = "none") attaches the knob
      // carrier to scr_eri so update_w can run the ladder READOUT. Never attached to
      // gw -- Sigma stays GW-form, and has_active_vertex() stays false (no injection).
      scr_eri.set_vertex(&vertex);
    }

    MBState mb_state(ft, output, mf, projector_ksIai, band_window, kpts_crys, trans_home_cell, false);
    if (local_polarizabilities) {
      mb_state.set_local_polarizabilities(std::move(local_polarizabilities.value()));
      local_polarizabilities.reset();
    }

    scf_loop(mb_state, dyson, eri, ft, mb_solver_t(&hf, &gw, &scr_eri),
             iter_solver.get(), niter, restart, conv_thr, const_mu,
             greens_func_source, greens_func_iteration);

  } else
    APP_ABORT("mbpt: Unknown solver type: {}",solver_type);
}

// FIXME this function requires HDF5_USE_FILE_LOCKING=FALSE.
void downfolding_1e(std::shared_ptr<mf::MF> mf, ptree const& pt) {
  std::string err = std::string("downfolding_1e - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);
  auto qp_selfenergy = io::get_value_with_default<bool>(pt,"qp_selfenergy",false);

  embed_t embed(*mf, wannier_file, trans_home_cell);

  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(outdir+"/"+prefix+".mbpt.h5", false));

  MBState mb_state(ft, outdir+"/"+prefix, mf, wannier_file, trans_home_cell, false);

  if (qp_selfenergy) {
    auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
    auto off_diag_mode = io::get_value_with_default<std::string>(pt,"off_diag_mode","qp_energy");
    io::tolower(ac_alg);
    io::tolower(off_diag_mode);
    utils::check(off_diag_mode=="fermi" or off_diag_mode=="qp_energy",
                 "unknown off_diag_mode: {}. Valid options are \"fermi\" and \"qp_energy\"");
    qp_params_t qp_params("sc", ac_alg,
                io::get_value_with_default<int>(pt,"Nfit",30),
                io::get_value_with_default<double>(pt,"eta", M_PI/ft.beta()),
                1e-8, "qpscf", false, off_diag_mode);
    embed.downfolding(mb_state, pt, &qp_params);
  } else {
    embed.downfolding(mb_state, pt);
  }
}

auto downfold_gloc_impl(std::shared_ptr<mf::MF> mf,
                        MBState&& mb_state,
                        ptree const& pt)
-> nda::array<ComplexType, 5> {
  std::string err = std::string("downfold_gloc_impl - Incorrect input - ");
  auto greens_func_source = io::get_value<std::string>(pt, "greens_func_source", err+"greens_func_source");
  auto greens_func_iteration = io::get_value_with_default<long>(pt, "greens_func_iteration", -1);
  auto force_real = io::get_value_with_default<bool>(pt, "force_real", true);
  embed_t embed(*mf);
  return embed.downfold_gloc(mb_state, force_real, greens_func_source, greens_func_iteration);
}

auto downfold_gloc(std::shared_ptr<mf::MF> mf, ptree const& pt,
                  nda::array<ComplexType, 5> const& projector_ksIai,
                  nda::array<long, 3> const& band_window,
                  nda::array<RealType, 2> const& kpts_crys)
  -> nda::array<ComplexType, 5> {
  std::string err = std::string("downfold_gloc - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt, "prefix", err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt, "outdir", "./");
  auto trans_home_cell = io::get_value_with_default<bool>(pt, "translate_home_cell", false);
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(outdir+"/"+prefix+".mbpt.h5", false));
  return downfold_gloc_impl(
      mf, MBState(ft, outdir+"/"+prefix, mf, projector_ksIai, band_window, kpts_crys, trans_home_cell, false), pt);
}

auto downfold_gloc_with_projector_from_h5(std::shared_ptr<mf::MF> mf, ptree const& pt)
-> nda::array<ComplexType, 5> {
  std::string err = std::string("downfold_gloc - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt, "prefix", err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt, "outdir", "./");
  auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
  auto trans_home_cell = io::get_value_with_default<bool>(pt, "translate_home_cell", false);
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(outdir+"/"+prefix+".mbpt.h5", false));
  return downfold_gloc_impl(
      mf, MBState(ft, outdir+"/"+prefix, mf, wannier_file, trans_home_cell, false), pt);
}

template<typename eri_t>
std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_coulomb_impl(eri_t &eri, MBState&& mb_state, ptree const& pt, 
                   std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
  std::string err = std::string("downfold_coulomb_impl - Incorrect input - ");
  auto greens_func_source = io::tolower_copy(io::get_value<std::string>(pt, "greens_func_source"));
  greens_func_source = (greens_func_source == "mf") ? "scf" : greens_func_source;
  auto greens_func_iteration = io::get_value_with_default<long>(pt, "greens_func_iteration", -1);
  auto screen_type = io::get_value<std::string>(
      pt, "screen_type", err+"screen_type. This parameter determines the type of screened interactions for the downfolded Hamiltonian. "
                             "Valid types are \"crpa\", \"crpa_ks\", \"crpa_vasp\", "
                             "\"gw_edmft\", \"gw_edmft_rpa\", and \"gw_edmft_density\"");
  io::tolower(screen_type);
  auto permut_symm = io::get_value_with_default<bool>(pt, "permut_symm", true);
  auto force_real = io::get_value_with_default<bool>(pt, "force_real", true);
  auto div_treatment = io::tolower_copy(io::get_value_with_default<std::string>(pt, "div_treatment", "gygi"));
  auto bare_div_treatment = io::tolower_copy(io::get_value_with_default<std::string>(pt, "bare_div_treatment", "gygi"));
  auto output_in_tau = io::get_value_with_default<bool>(pt, "output_in_tau", false);
  bool write_to_hdf5 = io::get_value_with_default<bool>(pt, "write_to_hdf5", true);
  bool q_dependent_output = io::get_value_with_default<bool>(pt, "q_dependent_output", false);

  if (q_dependent_output) write_to_hdf5 = true;

  auto mf = eri.MF();

  // set local polarizabilities if provided
  if (local_polarizabilities) {
    mb_state.set_local_polarizabilities(std::move(local_polarizabilities.value()));
    local_polarizabilities.reset();
  }
  embed_eri_t embed_eri(*mf, div_treatment, bare_div_treatment, "default");
  return (output_in_tau)?
    embed_eri.compute_downfolded_coulomb_tensors<true>(
      eri, mb_state, screen_type, permut_symm, force_real, mb_state.ft, 
      greens_func_source, greens_func_iteration, write_to_hdf5, q_dependent_output) :
    embed_eri.compute_downfolded_coulomb_tensors<false>(
      eri, mb_state, screen_type, permut_symm, force_real, mb_state.ft, 
      greens_func_source, greens_func_iteration, write_to_hdf5, q_dependent_output);
}

template<typename eri_t>
std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_coulomb_with_projector_from_h5(eri_t &eri, ptree const& pt,
              std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
  std::string err = std::string("downfold_coulomb - Incorrect input - ");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);
  auto greens_func_source = io::tolower_copy(io::get_value<std::string>(pt, "greens_func_source",
      err+"greens_func_source. This parameter defines the source of input Green's function. Valid types are \"mf\", \"scf\", and \"embed\"."));

  auto mf = eri.MF();
  std::string output = outdir + "/" + prefix;
  
  ensure_checkpoint(mf, output, greens_func_source, pt);

  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(output+".mbpt.h5", false));
  return downfold_coulomb_impl(
    eri, MBState(ft, output, mf, wannier_file, trans_home_cell, false),
    pt, local_polarizabilities);
}

template<typename eri_t>
std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_coulomb(eri_t &eri, ptree const& pt,
              nda::array<ComplexType, 5> const& projector_ksIai,
              nda::array<long, 3> const& band_window,
              nda::array<RealType, 2> const& kpts_crys,
              std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
  std::string err = std::string("downfold_coulomb - Incorrect input - ");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);
  auto greens_func_source = io::tolower_copy(io::get_value<std::string>(pt,"greens_func_source",
      err+"greens_func_source. This parameter defines the source of input Green's function. Valid types are \"mf\", \"scf\", and \"embed\". "));

  auto mf = eri.MF();
  std::string output = outdir + "/" + prefix;
  
  ensure_checkpoint(mf, output, greens_func_source, pt);

  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(output+".mbpt.h5", false));
  return downfold_coulomb_impl(
    eri, MBState(ft, output, mf, projector_ksIai, band_window, kpts_crys, trans_home_cell, false),
    pt, local_polarizabilities);
}

/**
 * Downfolds the two-electron Hamiltonian with arguments in property tree.
 * Required arguments:
 *  - prefix: Prefix of the output and input files.
 *  - wannier_file: h5 file in which the Wannier transformation matrices are stored.
 *  - screen_type: Screening types for the partially screened interaction u(iw). {choices: "bare", "crpa", "edmft"}
 * Optional arguments (with default values):
 *  - outdir: "./" Directory where the source and output files are.
 *  - div_treatment: "gygi" Divergent treatment for Coulomb kernel. {choices: "ignore_g0", "gygi"}
 *  - bare_div_treatment: "gygi" Divergent treatment for the bare Coulomb kernel. {choices: "ignore_g0", "gygi"}
 * Optional arguments used only when outdir/prefix.mbpt.h5 does not exist:
 *  - beta: "1000" Inverse temperature (a.u.)
 *  - wmax: "12.0" Frequency cutoff for the IAFT grid (a.u.)
 *  - iaft_prec: "high" Precision of IAFT grids. {choices: "high", "medium", "low"}
 */
template<typename eri_t>
void downfolding_2e(eri_t &eri, ptree const& pt,
               std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
  auto outdir = io::get_value_with_default<std::string>(pt, "outdir", "./");
  auto prefix = io::get_value<std::string>(pt, "prefix", "downfolding_2e - Incorrect input - prefix");
  auto wannier_file = io::get_value<std::string>(pt, "wannier_file", "downfolding_2e - Incorrect input - wannier_file");
  auto trans_home_cell = io::get_value_with_default<bool>(pt, "translate_home_cell", false);
  auto screen_type = io::tolower_copy(io::get_value<std::string>(pt, "screen_type", "downfolding_2e - Incorrect input - screen_type"));
  auto greens_func_source = io::tolower_copy(io::get_value<std::string>(pt, "greens_func_source", "downfolding_2e - Incorrect input - greens_func_source"));
  auto div_treatment = io::tolower_copy(io::get_value_with_default<std::string>(pt, "div_treatment", "gygi"));
  auto bare_div_treatment = io::tolower_copy(io::get_value_with_default<std::string>(pt, "bare_div_treatment", "gygi"));

  auto mf = eri.MF();
  std::string output = outdir + "/" + prefix;
  
  ensure_checkpoint(mf, output, greens_func_source, pt);

  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(output+".mbpt.h5", false));
  MBState mb_state(ft, output, mf, wannier_file, trans_home_cell, false);

  if (local_polarizabilities) {
    mb_state.set_local_polarizabilities(std::move(local_polarizabilities.value()));
    local_polarizabilities.reset();
  }

  embed_eri_t embed_eri(*mf, div_treatment, bare_div_treatment, "default");

  if (screen_type.substr(0, 8) == "gw_edmft") {
    embed_eri.downfolding_edmft(eri, mb_state, pt, screen_type);
  } else {
    embed_eri.downfolding_crpa(eri, mb_state, pt, screen_type);
  }
}

/**
 * Generates a downfolded Hamiltonian at the Hartree-Fock (HF) level. Bare 2-electron integrals
 * are calculated in the local basis, as defined by the provided projection matrix. 
 * HF frozen core contributions are added to the bare 1-body Hamiltonian in the local basis. 
 * The results are consistent with screen_type=bare and dc_type=hf in downfold_2e/downfold_1e routines.
 * Output is written in a format suitable to be read back by the mbpt modules, e.g. can be used in the 
 * mean_field and interaction sections.
 * Required arguments:
 *  - prefix: Prefix of the generated output mbpt and model files.
 *  - wannier_file: h5 file in which the Wannier transformation matrices are stored.
 * Optional arguments (with default values):
 *  - outdir: "./" Directory where the resulting prefix.mbpt.h5 and prefix.model.h5 files will be placed.
 *  - hf_div_treatment: "gygi" Divergent treatment for the bare Coulomb kernel. {choices: "ignore_g0", "gygi"}
 *  - permut_symm: false. If true, applies 4-/8-fold permutation symmetry to 2-electron interaction. Only
 applies if factorization="none".
 *  - force_real: false. If true, forces the 2-electron interaction tensor to be real.
 *  - factorization_type: "cholesky", Type of factorization. {choices: "none", "cholesky", "cholesky_high_memory", "choleksy_from_4index", "thc"}
 *  - thresh: 1e-6. Threshold used if factorization is requested.
 * Optional arguments used only when outdir/prefix.mbpt.h5 does not exist:
 *  - beta: "1000" Inverse temperature (a.u.)
 *  - wmax: "12.0" Frequency cutoff for the IAFT grids (a.u.)
 *  - iaft_prec: "high" Precision of IAFT grids. {choices: "high", "medium", "low"}
 */
template<typename eri_t>
void hf_downfold(eri_t &eri, ptree const& pt) {
  std::string err = std::string("hf_downfold - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);

  // two-body downfolding options
  auto hf_div_treatment = io::get_value_with_default<std::string>(pt, "hf_div_treatment", "gygi");
  auto factorization_type = io::get_value_with_default<std::string>(pt, "factorization_type", "cholesky");
  io::tolower(hf_div_treatment);
  io::tolower(factorization_type);

  utils::check( factorization_type=="none"                  or
                factorization_type=="cholesky"              or
                factorization_type=="cholesky_high_memory"  or
                factorization_type=="cholesky_from_4index"  or
                factorization_type=="thc",
                " downfold_2e: Invalid factorization_type: {}", factorization_type);

  auto mf = eri.MF();

  // mbpt and model outputs
  std::string output = outdir + "/" + prefix;

  // initialize
  imag_axes_ft::IAFT ft(pt, true, mf::wmax_from_mf(*mf));
  hamilt::pseudopot psp(*mf);
  write_mf_data(*mf, ft, psp, output);
  MBState mb_state(ft, output, mf, wannier_file, trans_home_cell, false);

  // Two-body Hamiltonian
  embed_eri_t embed_eri(*mf, "ignore_g0", hf_div_treatment, "model_static");
  embed_eri.downfolding_crpa(eri, mb_state, pt, "bare", factorization_type,
                             io::get_value_with_default<double>(pt, "thresh", 1e-6));

  // One-body Hamiltonian
  embed_t embed(*mf, wannier_file, trans_home_cell);
  embed.hf_downfolding(outdir, prefix, eri, ft,
                       io::get_value_with_default<bool>(pt, "force_real", true),
                       hf_div_treatment);

}

/**
 * Generates a downfolded Hamiltonian at the GW level. cRPA Screened 2-electron integrals
 * are calculated in the local basis, as defined by the provided projection matrix. 
 * A quasi-particle approximation to the GW self-energy is applied to generate a downfolded
 * 1-body Hamiltonian in the local basis. 
 * The results are consistent with screen_type=crpa and dc_type=gw in downfold_2e/downfold_1e routines.
 * Output is written in a format suitable to be read back by the mbpt modules,
 * e.g. model Hamiltonian type mean-field chkpt file and ERI-compatible h5 chkpt file,
 * which can be used in the mean_field and interaction sections.
 * Required arguments:
 *  - prefix: Prefix of the generated output mbpt and model files.
 *  - wannier_file: h5 file in which the Wannier transformation matrices are stored.
 * Optional arguments (with default values):
 *  - outdir: "./" Directory where the resulting prefix.model.h5 files will be placed.
 *  - div_treatment: "gygi" Divergent treatment for Coulomb kernel. {choices: "ignore_g0", "gygi"}
 *  - hf_div_treatment: "gygi" Divergent treatment for the bare Coulomb kernel. {choices: "ignore_g0", "gygi"}
 *  - permut_symm: false. If true, applies 4-/8-fold permutation symmetry to 2-electron interaction. Only applies if factorization="none".
 *  - force_real: false. If true, forces the 2-electron interaction tensor to be real. 
 *  - factorization_type: "cholesky", Type of factorization. {choices: "none", "cholesky", "cholesky_high_memory", "choleksy_from_4index", "thc"}
 *  - thresh: 1e-6. Threshold used if factorization is requested.
 *  Parameters used by quasiparticle algorithm:
 *  - ac_alg: Algorithm for analytic continuation, default:pade {choices: pade}
 *  - eta: Smearing parameter: default:1e-6
 *  - Nfit: Number of terms in AC fit, default: 30
 *  - off_diag_mode: Off diagonal treatment, default: qp_energy. {choices: fermi, qp_energy} 
 */
template<typename eri_t>
void gw_downfold(eri_t &eri, ptree &pt) {
  std::string err = std::string("gw_downfold - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto wannier_file = io::get_value<std::string>(pt,"wannier_file",err+"wannier_file");
  auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);

  // two-body downfolding options
  auto div_treatment = io::get_value_with_default<std::string>(pt, "div_treatment", "gygi");
  auto hf_div_treatment = io::get_value_with_default<std::string>(pt, "hf_div_treatment", "gygi");
  auto factorization_type = io::get_value_with_default<std::string>(pt, "factorization_type", "cholesky");
  io::tolower(div_treatment);
  io::tolower(hf_div_treatment);
  io::tolower(factorization_type);

  utils::check( factorization_type=="none"                  or
                factorization_type=="cholesky"              or
                factorization_type=="cholesky_high_memory"  or
                factorization_type=="cholesky_from_4index"  or
                factorization_type=="thc",
                " downfold_2e: Invalid factorization_type: {}", factorization_type);

  auto mf = eri.MF();

  // mbpt and model output
  std::string output = outdir + "/" + prefix;

  utils::check(std::filesystem::exists(output+".mbpt.h5"),
               "gw_downfolding: {}.mbpt.h5, does not exist!", output);

  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(output+".mbpt.h5", false));
  // create MBstate object to store the state of the downfolding
  MBState mb_state(ft, output, mf, wannier_file, trans_home_cell, false);

  // Two-body Hamiltonian
  embed_eri_t embed_eri(*mf, div_treatment, hf_div_treatment, "model_static");
  embed_eri.downfolding_crpa(eri, mb_state, pt, "crpa", factorization_type,
                             io::get_value_with_default<double>(pt, "thresh", 1e-6));

  // one body hamiltonian
  auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
  auto off_diag_mode = io::get_value_with_default<std::string>(pt,"off_diag_mode","qp_energy");
  io::tolower(ac_alg);
  io::tolower(off_diag_mode);
  utils::check(off_diag_mode=="fermi" or off_diag_mode=="qp_energy",
               "unknown off_diag_mode: {}. Valid options are \"fermi\" and \"qp_energy\"");
  qp_params_t qp_params(
      "sc", ac_alg,
      io::get_value_with_default<int>(pt,"Nfit",30),
      io::get_value_with_default<double>(pt,"eta", M_PI/ft.beta()),
      1e-8, "qpscf", false, off_diag_mode);
  embed_t embed(*mf, wannier_file, trans_home_cell);
  pt.put("update_dc", true);
  pt.put("dc_type", "gw");
  embed.downfolding(mb_state, pt, &qp_params, "model_static");
}

void dmft_embed_with_projector_from_h5(std::shared_ptr<mf::MF> mf, ptree const& pt,
                std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials,
                std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies) {
  std::string err = std::string("dmft_embed - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");

  std::unique_ptr<iter_scf::iter_scf_t> iter_solver;
  if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
    iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt, 1.0));
  } else {
    iter_solver = nullptr;
  }
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(outdir+"/"+prefix+".mbpt.h5", false));
  MBState mb_state(ft, outdir+"/"+prefix, mf,
                   io::get_value<std::string>(pt,"wannier_file",err+"wannier_file"),
                   io::get_value_with_default<bool>(pt,"translate_home_cell",false), false);
  if (local_hf_potentials and local_selfenergies) {
    mb_state.set_local_hf_potentials(std::move(local_hf_potentials.value()));
    mb_state.set_local_selfenergies(std::move(local_selfenergies.value()));
    local_hf_potentials.reset();
    local_selfenergies.reset();
  }

  auto dyson = simple_dyson(mf.get(), &ft, mb_state.coqui_prefix,
                            io::get_value_with_default<double>(pt,"mu_tolerance", 1e-9),
                            io::get_value_with_default<std::string>(pt, "mu_update_alg", "midpoint"));

  embed_t embed(*mf);
  embed.dmft_embed(mb_state, dyson, iter_solver.get(),
                   io::get_value_with_default<bool>(pt,"qp_approx_mbpt",false),
                   io::get_value_with_default<bool>(pt,"corr_only",false));
}

void dmft_embed(std::shared_ptr<mf::MF> mf, ptree const& pt,
                nda::array<ComplexType, 5> const& projector_ksIai,
                nda::array<long, 3> const& band_window,
                nda::array<RealType, 2> const& kpts_crys,
                std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials,
                std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies) {
  std::string err = std::string("dmft_embed - Incorrect input - ");
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");

  std::unique_ptr<iter_scf::iter_scf_t> iter_solver;
  if (io::get_value_with_default<bool>(pt,"iter_alg.enable", true)) {
    iter_solver = std::make_unique<iter_scf::iter_scf_t>(iter_scf::make_iter_scf(pt, 1.0));
  } else {
    iter_solver = nullptr;
  }
  imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(outdir+"/"+prefix+".mbpt.h5", false));
  MBState mb_state(ft, outdir+"/"+prefix, mf,
                   projector_ksIai, band_window, kpts_crys,
                   io::get_value_with_default<bool>(pt,"translate_home_cell",false), false);
  if (local_hf_potentials and local_selfenergies) {
    mb_state.set_local_hf_potentials(std::move(local_hf_potentials.value()));
    mb_state.set_local_selfenergies(std::move(local_selfenergies.value()));
    local_hf_potentials.reset();
    local_selfenergies.reset();
  }

  auto dyson = simple_dyson(mf.get(), &ft, mb_state.coqui_prefix,
                            io::get_value_with_default<double>(pt,"mu_tolerance", 1e-9),
                            io::get_value_with_default<std::string>(pt, "mu_update_alg", "midpoint"));

  embed_t embed(*mf);
  embed.dmft_embed(mb_state, dyson, iter_solver.get(),
                   io::get_value_with_default<bool>(pt,"qp_approx_mbpt",false),
                   io::get_value_with_default<bool>(pt,"corr_only",false));
}


// instantiations
using mpi3::communicator;

template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_coulomb_impl(thc_reader_t &, MBState&& mb_state, ptree const& pt,
                   std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities);

template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_coulomb_with_projector_from_h5(
  thc_reader_t &, ptree const&, std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_coulomb(thc_reader_t &, ptree const&,
              nda::array<ComplexType, 5> const&,
              nda::array<long, 3> const&,
              nda::array<RealType, 2> const&,
              std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

template void downfolding_2e(
     thc_reader_t&, ptree const&, std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

template void hf_downfold(thc_reader_t&, ptree const&);
template void gw_downfold(thc_reader_t&, ptree&);

#define MBPT_INST(HF, HARTREE, EXCHANGE, CORR) \
template void mbpt(std::string, \
     mb_eri_t<HF, HARTREE, EXCHANGE, CORR>&,    \
     ptree const&);                             \
template void mbpt(std::string, \
     mb_eri_t<HF, HARTREE, EXCHANGE, CORR>&, \
     ptree const&,                             \
     nda::array<ComplexType, 5> const&,  \
     nda::array<long, 3> const&,   \
     nda::array<RealType, 2> const&,           \
     std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

// All combinations of thc/chol for 4 eri slots
  MBPT_INST(thc_reader_t, thc_reader_t, thc_reader_t, thc_reader_t)
  MBPT_INST(thc_reader_t, thc_reader_t, thc_reader_t, chol_reader_t)
  MBPT_INST(thc_reader_t, thc_reader_t, chol_reader_t, thc_reader_t)
  MBPT_INST(thc_reader_t, thc_reader_t, chol_reader_t, chol_reader_t)
  MBPT_INST(thc_reader_t, chol_reader_t, thc_reader_t, thc_reader_t)
  MBPT_INST(thc_reader_t, chol_reader_t, thc_reader_t, chol_reader_t)
  MBPT_INST(thc_reader_t, chol_reader_t, chol_reader_t, thc_reader_t)
  MBPT_INST(thc_reader_t, chol_reader_t, chol_reader_t, chol_reader_t)
  MBPT_INST(chol_reader_t, thc_reader_t, thc_reader_t, thc_reader_t)
  MBPT_INST(chol_reader_t, thc_reader_t, thc_reader_t, chol_reader_t)
  MBPT_INST(chol_reader_t, thc_reader_t, chol_reader_t, thc_reader_t)
  MBPT_INST(chol_reader_t, thc_reader_t, chol_reader_t, chol_reader_t)
  MBPT_INST(chol_reader_t, chol_reader_t, thc_reader_t, thc_reader_t)
  MBPT_INST(chol_reader_t, chol_reader_t, thc_reader_t, chol_reader_t)
  MBPT_INST(chol_reader_t, chol_reader_t, chol_reader_t, thc_reader_t)
  MBPT_INST(chol_reader_t, chol_reader_t, chol_reader_t, chol_reader_t)

#undef MBPT_INST

}
