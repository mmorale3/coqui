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
 *   tol_svd = 1e-12, closure_cut = "hard", closure_svd_cut = "hard", closure_cut_window = 10, phase_keep = 0
 *                       S7f closure options (closure.hpp closure_params_t, cayley.hpp upfold_opts_t): SVD cutoff, Gram /
 *                       SVD cut placement ("gap": at the largest eigenvalue ratio within a window of the tolerance;
 *                       "smooth": smooth-step weights of the Gram rows in the window), phase continuity (> 0)
 *   debug_noise_h0 = 0, debug_noise_seed = 0, debug_noise_sigma = 0, debug_noise_iter = 1, bases_file = ""
 *                       diagnostics: relative Hermitian noise on H0, or relative noise on Sigma at the nodes of iteration
 *                       debug_noise_iter (noise-floor meter, test [.scf_noise]); real-pole bases read from a file instead of built (parity test: the
 *                       pivoted-QR pole selection differs between LAPACKs; keys sigma_{particle,hole}_w, g_{particle,hole}_w,
 *                       bos_nu, bos_zeta_nodes)
 *   niter = 12          TOTAL number of iterations (a restart continues until niter iterations are done)
 *   mixing = 0.5        linear mixing of Sigma^{>/<} at the nodes (F is not mixed, as python)
 *   conv_thr = 1e-5     stop when max|dSigma| at the nodes (after mixing, as python) < conv_thr
 *   t_chunk = 0 (auto), ray_decades = 36   time chunk of the ray products; ray length e^{-emin smax sin theta_t} = e^{-decades}
 *   time_grid = "id"    time nodes of the ray products (S7b): "id" = time-node ID (time_grids.hpp; four grids rebuilt every
 *                       iteration from the current poles and bosonic poles, ~100-170 nodes each), "gl" = the generic
 *                       Gauss-Legendre rays for_spectrum(theta_t, emin, ray_decades) (~1000 nodes; the python reference)
 *   time_eps = eps, time_pad = 1.25, time_oversample = 1.0   ID tolerance, energy-range margin [Emin/pad, pad Emax] and
 *                       node oversampling (time_id_opts_t); ignored for "gl"
 *   restart = false     resume from <output>.gw_line.h5:/scf_line/final_iter (bitwise identical continuation)
 *   checkpoint_sigma = "last"   Sigma at the nodes in the checkpoint (S7e): "last" = only the last iteration's, in the
 *                       separate file <output>.gw_line.sigma.h5 rewritten every iteration (constant size); "all" = every
 *                       iteration's in iter<N>/Sigma_{p,h} (the pre-S7e layout; 2 N_k N_zeta nb^2 x 16 B per iteration)
 *   sigma_kdist = true  Sigma at the nodes k-distributed over the ranks (S7e, k_dist.hpp: owner(k) = k mod np, the closure's
 *                       ownership; reduce-scatter in the self-energy); false = replicated on every rank (pre-S7e)
 *   output / outdir + prefix   checkpoint stem (MBPT_drivers resolve_mbpt_output_stem; the driver reads "output")
 *   div_treatment       must be absent or "ignore_g0" (Z(Gamma) without its G = 0 term, no Madelung/head correction)
 *   spectra = { enable = true, eta = [0.004, 0.01], wmin = -0.45, wmax = 0.45, nw = 601 }   A(k,w) at the end
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
  std::string closure_cut = "hard";      ///< S7f: Gram cut "hard" (python) | "gap" | "smooth" (cayley.hpp upfold_opts_t)
  std::string closure_svd_cut = "hard";  ///< S7f: SVD cut "hard" | "gap"
  double closure_cut_window = 10.0;      ///< S7f: window factor of the gap / smooth cuts
  double phase_keep = 0.0;               ///< S7f: > 0 = phase continuity (keep the previous phi* basin within this factor)
  double debug_noise_h0 = 0.0;           ///< diagnostics: relative Hermitian Gaussian noise on H0 (seed debug_noise_seed)
  long debug_noise_seed = 0;
  double debug_noise_sigma = 0.0;        ///< diagnostics: relative noise on Sigma at the nodes (before mixing) ...
  long debug_noise_iter = 1;             ///< ... in this iteration
  std::string bases_file;                ///< diagnostics/parity: real-pole bases read from this file (gen_lih222_scf_ref.py)
  long niter = 12;
  double mixing = 0.5, conv_thr = 1e-5;
  long t_chunk = 0;   // 0 = automatic (host: 32, COQUI_GWLINE_HOST_TCHUNK; device: from the free memory, capped)
  double ray_decades = 36.0;
  std::string time_grid = "id";          ///< "id" (time-node ID) or "gl" (Gauss-Legendre rays)
  double time_eps = 1e-10, time_pad = 1.25, time_oversample = 1.0;   ///< time_eps defaults to eps
  bool restart = false;
  std::string output = "./gw_line";
  std::string checkpoint_sigma = "last";   ///< "last" | "all" (S7e)
  bool sigma_kdist = true;                 ///< k-distributed Sigma at the nodes (S7e)
  bool do_spectra = true;
  spectra_params_t spectra;

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
};

struct gw_line_result_t {
  std::vector<gw_line_iter_t> history;   ///< all iterations (including those read on restart)
  bool converged = false;
  double mu = 0.0, mu_sigma = 0.0;       ///< final centre; centre at which Sigma_p/h were sampled
  pole_data_t poles;                     ///< final poles (mu-relative; compressed or factorized Lehmann)
  nda::array<ComplexType, 3> F;          ///< final V_H + Sigma_x
  nda::array<ComplexType, 3> H0;         ///< one-body Hamiltonian (KS band basis)
  nda::array<ComplexType, 4> Sig_p, Sig_h;   ///< last mixed Sigma at the nodes (empty if no iteration was done); with
                                             ///< sigma_kdist (default) the rows of the k owned by this rank (k mod np)
  nda::array<ComplexType, 1> zeta;       ///< fermionic nodes (mu-relative)
  std::optional<spectra_out_t> spectra;
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
