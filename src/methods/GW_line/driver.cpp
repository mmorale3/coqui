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

/**
 * gw_line_scf: the self-consistency loop of the line GW (see driver.hpp for the options, the initial guess, the loop and
 * the checkpoint layout). Python oracle: coqui/cayley/cayley/line/driver.py::LineSCGW.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numbers>
#include <random>
#include <string>
#include <vector>
#include <fstream>
#include <sys/resource.h>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/check.hpp"
#include "utilities/Timer.hpp"
#include "utilities/mpi_context.h"
#include "utilities/h5_background_writer.hpp"
#include "numerics/shared_array/nda.hpp"
#include "hamiltonian/one_body_hamiltonian.hpp"
#include "hamiltonian/pseudo/pseudopot.h"

#include "numerics/line_dlr/line_basis.hpp"
#include "numerics/line_dlr/bosonic_basis.hpp"
#include "numerics/line_dlr/time_ray.hpp"
#include "numerics/line_dlr/time_id.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/polarization.hpp"
#include "methods/GW_line/screened.hpp"
#include "methods/GW_line/self_energy.hpp"
#include "methods/GW_line/static_part.hpp"
#include "methods/GW_line/closure.hpp"
#include "methods/GW_line/closure_device.hpp"
#include "methods/GW_line/spectra.hpp"
#include "methods/GW_line/time_grids.hpp"
#include "methods/GW_line/head.hpp"
#include "methods/GW_line/head_pass.hpp"
#include "methods/GW_line/optics.hpp"
#include "methods/GW_line/ibz.hpp"
#include "methods/GW_line/self_energy_ibz.hpp"
#include "methods/GW_line/static_ibz.hpp"
#include "methods/GW_line/q_plan.hpp"
#include "methods/GW_line/k_dist.hpp"
#include "methods/GW_line/scf_mixing.hpp"
#include "methods/GW_line/warm_start.hpp"
#include "methods/GW_line/thermal.hpp"
#include "methods/GW_line/hybrid.hpp"
#include "methods/GW_line/driver.hpp"

namespace methods::gw_line {

using numerics::line_dlr::bosonic_basis_t;
using numerics::line_dlr::line_basis_t;
using numerics::line_dlr::time_ray_t;
using numerics::line_dlr::time_nodes_t;

static constexpr double HA_EV = 27.211386;

// ------------------------------------------------------------------------------------------------------------------
// parameters
// ------------------------------------------------------------------------------------------------------------------
gw_line_params_t gw_line_params_t::from_ptree(ptree const &pt) {
  gw_line_params_t p;
  p.theta_deg     = io::get_value_with_default<double>(pt, "theta_deg", p.theta_deg);
  p.eps           = io::get_value_with_default<double>(pt, "eps", p.eps);
  p.lam           = io::get_value_with_default<double>(pt, "lam", p.lam);
  p.lam_b         = io::get_value_with_default<double>(pt, "lam_b", p.lam_b);
  p.sigma_gap     = io::get_value_with_default<double>(pt, "sigma_gap", p.sigma_gap);
  p.bos_gap       = io::get_value_with_default<double>(pt, "bos_gap", p.bos_gap);
  p.g_gap         = io::get_value_with_default<double>(pt, "g_gap", p.g_gap);
  p.g_repr        = io::get_value_with_default<std::string>(pt, "g_repr", p.g_repr);
  io::tolower(p.g_repr);
  p.g_emax        = io::get_value_with_default<double>(pt, "g_emax", p.g_emax);
  p.g_wtol        = io::get_value_with_default<double>(pt, "g_wtol", p.g_wtol);
  p.g_emin_frac   = io::get_value_with_default<double>(pt, "g_emin_frac", p.g_emin_frac);
  p.g_wsmall      = io::get_value_with_default<double>(pt, "g_wsmall", p.g_wsmall);
  p.nodes_per_ray = io::get_value_with_default<long>(pt, "nodes_per_ray", p.nodes_per_ray);
  p.node_tmin     = io::get_value_with_default<double>(pt, "node_tmin", p.node_tmin);
  p.node_tmax     = io::get_value_with_default<double>(pt, "node_tmax", p.node_tmax);
  p.wp            = io::get_value_with_default<double>(pt, "wp", p.wp);
  p.K             = io::get_value_with_default<long>(pt, "K", p.K);
  p.tol_gram      = io::get_value_with_default<double>(pt, "tol_gram", p.tol_gram);
  p.nphi          = io::get_value_with_default<long>(pt, "nphi", p.nphi);
  p.tol_svd       = io::get_value_with_default<double>(pt, "tol_svd", p.tol_svd);
  p.tol_gram_eps  = io::get_value_with_default<double>(pt, "tol_gram_eps", p.tol_gram_eps);
  p.closure_cut   = io::get_value_with_default<std::string>(pt, "closure_cut", p.closure_cut);
  io::tolower(p.closure_cut);
  p.closure_svd_cut = io::get_value_with_default<std::string>(pt, "closure_svd_cut", p.closure_svd_cut);
  io::tolower(p.closure_svd_cut);
  p.closure_cut_window = io::get_value_with_default<double>(pt, "closure_cut_window", p.closure_cut_window);
  p.phase_keep    = io::get_value_with_default<double>(pt, "phase_keep", p.phase_keep);
  p.closure_threads = io::get_value_with_default<long>(pt, "closure_threads", p.closure_threads);
  p.closure_k_workers = io::get_value_with_default<long>(pt, "closure_k_workers", p.closure_k_workers);
  p.closure_svd   = io::get_value_with_default<std::string>(pt, "closure_svd", p.closure_svd);
  io::tolower(p.closure_svd);
  p.closure_ueig  = io::get_value_with_default<std::string>(pt, "closure_ueig", p.closure_ueig);
  io::tolower(p.closure_ueig);
  p.closure_device = io::get_value_with_default<std::string>(pt, "closure_device", p.closure_device);
  io::tolower(p.closure_device);
  p.closure_dev_svd = io::get_value_with_default<std::string>(pt, "closure_dev_svd", p.closure_dev_svd);
  io::tolower(p.closure_dev_svd);
  p.debug_noise_h0   = io::get_value_with_default<double>(pt, "debug_noise_h0", p.debug_noise_h0);
  p.debug_noise_seed = io::get_value_with_default<long>(pt, "debug_noise_seed", p.debug_noise_seed);
  p.debug_noise_sigma = io::get_value_with_default<double>(pt, "debug_noise_sigma", p.debug_noise_sigma);
  p.debug_noise_iter  = io::get_value_with_default<long>(pt, "debug_noise_iter", p.debug_noise_iter);
  p.bases_file    = io::get_value_with_default<std::string>(pt, "bases_file", p.bases_file);
  p.niter         = io::get_value_with_default<long>(pt, "niter", p.niter);
  p.mixing        = io::get_value_with_default<double>(pt, "mixing", p.mixing);
  p.conv_thr      = io::get_value_with_default<double>(pt, "conv_thr", p.conv_thr);
  p.t_chunk       = io::get_value_with_default<long>(pt, "t_chunk", p.t_chunk);
  p.ray_decades   = io::get_value_with_default<double>(pt, "ray_decades", p.ray_decades);
  p.restart       = io::get_value_with_default<bool>(pt, "restart", p.restart);
  p.time_grid     = io::get_value_with_default<std::string>(pt, "time_grid", p.time_grid);
  io::tolower(p.time_grid);
  p.time_eps        = io::get_value_with_default<double>(pt, "time_eps", p.eps);
  p.time_pad        = io::get_value_with_default<double>(pt, "time_pad", p.time_pad);
  p.time_oversample = io::get_value_with_default<double>(pt, "time_oversample", p.time_oversample);
  p.time_snap       = io::get_value_with_default<double>(pt, "time_snap", p.time_snap);
  p.sigma_kdist     = io::get_value_with_default<bool>(pt, "sigma_kdist", p.sigma_kdist);
  p.ibz             = io::get_value_with_default<bool>(pt, "ibz", p.ibz);
  p.mem_budget_gb     = io::get_value_with_default<double>(pt, "mem_budget_gb", p.mem_budget_gb);
  p.dev_mem_budget_gb = io::get_value_with_default<double>(pt, "dev_mem_budget_gb", p.dev_mem_budget_gb);
  p.mem_frac          = io::get_value_with_default<double>(pt, "mem_frac", p.mem_frac);
  p.q_group_size      = io::get_value_with_default<long>(pt, "q_group_size", p.q_group_size);
  utils::check(p.mem_frac > 0.0 and p.mem_frac <= 1.0, "gw_line: mem_frac = {} must be in (0, 1]", p.mem_frac);
  if (char const *e = std::getenv("COQUI_GWLINE_IBZ"); e != nullptr and *e != '\0') p.ibz = std::strtol(e, nullptr, 10) != 0;
  p.checkpoint_sigma = io::get_value_with_default<std::string>(pt, "checkpoint_sigma", p.checkpoint_sigma);
  io::tolower(p.checkpoint_sigma);
  {
    auto o = pt.get_optional<std::string>("output");
    if (o and not o->empty()) p.output = *o;
    else {
      auto outdir = io::get_value_with_default<std::string>(pt, "outdir", "./");
      auto prefix = io::get_value_with_default<std::string>(pt, "prefix", "gw_line");
      p.output    = outdir + "/" + prefix;
    }
  }
  p.do_spectra    = io::get_value_with_default<bool>(pt, "spectra.enable", true);
  p.spectra.eta   = io::get_array_with_default<double>(pt, "spectra.eta", p.spectra.eta);
  p.spectra.wmin  = io::get_value_with_default<double>(pt, "spectra.wmin", p.spectra.wmin);
  p.spectra.wmax  = io::get_value_with_default<double>(pt, "spectra.wmax", p.spectra.wmax);
  p.spectra.nw    = io::get_value_with_default<long>(pt, "spectra.nw", p.spectra.nw);

  // S9a: q -> 0 divergence (head.hpp). Defaults: ignore_g0 (unchanged); hf_div_treatment follows div_treatment
  p.div_treatment = io::get_value_with_default<std::string>(pt, "div_treatment", p.div_treatment);
  io::tolower(p.div_treatment);
  head_check_variant(p.div_treatment);
  p.hf_div_treatment = io::get_value_with_default<std::string>(pt, "hf_div_treatment",
                                                               p.div_treatment == "ignore_g0" ? "ignore_g0" : "gygi");
  io::tolower(p.hf_div_treatment);
  utils::check(p.hf_div_treatment == "ignore_g0" or p.hf_div_treatment == "gygi",
               "gw_line: hf_div_treatment must be \"ignore_g0\" or \"gygi\" (got \"{}\")", p.hf_div_treatment);
  p.head_extrapolation = io::get_value_with_default<std::string>(pt, "head_extrapolation", p.head_extrapolation);
  io::tolower(p.head_extrapolation);
  if (head_div_is_gygi(p.div_treatment)) p.head_extrapolation = p.div_treatment;
  head_check_variant(p.head_extrapolation);
  utils::check(head_div_is_gygi(p.head_extrapolation), "gw_line: head_extrapolation must be a gygi variant (got \"{}\")",
               p.head_extrapolation);
  // S8b finite temperature
  p.theta_t_frac    = io::get_value_with_default<double>(pt, "theta_t_frac", p.theta_t_frac);
  p.beta            = io::get_value_with_default<double>(pt, "beta", p.beta);
  p.thermal_tol     = io::get_value_with_default<double>(pt, "thermal_tol", p.thermal_tol);
  p.thermal_floor   = io::get_value_with_default<double>(pt, "thermal_floor", p.thermal_floor);
  p.thermal_floor_f = io::get_value_with_default<double>(pt, "thermal_floor_f", p.thermal_floor);
  p.wp_floor        = io::get_value_with_default<double>(pt, "wp_floor", p.wp_floor);
  p.mu_rule         = io::get_value_with_default<std::string>(pt, "mu_rule", p.mu_rule);
  io::tolower(p.mu_rule);
  p.mu_dn_max       = io::get_value_with_default<double>(pt, "mu_dn_max", p.mu_dn_max);
  p.mu_th_factor    = io::get_value_with_default<double>(pt, "mu_th_factor", p.mu_th_factor);
  p.band_heights    = io::get_value_with_default<long>(pt, "band_heights", p.band_heights);
  p.band_x          = io::get_value_with_default<long>(pt, "band_x", p.band_x);
  p.band_top        = io::get_value_with_default<double>(pt, "band_top", p.band_top);
  p.mats_factor     = io::get_value_with_default<double>(pt, "mats_factor", p.mats_factor);
  p.bos_eps_T       = io::get_value_with_default<double>(pt, "bos_eps_T", p.bos_eps_T);
  p.bos_line_eps    = io::get_value_with_default<double>(pt, "bos_line_eps", p.bos_line_eps);
  p.cut_odd         = io::get_value_with_default<double>(pt, "cut_odd", p.cut_odd);
  p.cut_even        = io::get_value_with_default<double>(pt, "cut_even", p.cut_even);
  p.tau_grid        = io::get_value_with_default<std::string>(pt, "tau_grid", p.tau_grid);
  io::tolower(p.tau_grid);
  p.tau_eps         = io::get_value_with_default<double>(pt, "tau_eps", p.tau_eps);
  p.spectra_occupation = io::get_value_with_default<bool>(pt, "spectra.occupation", p.spectra_occupation);
  p.thermal_bases_file = io::get_value_with_default<std::string>(pt, "thermal_bases_file", p.thermal_bases_file);
  p.scf_density = io::get_value_with_default<std::string>(pt, "scf_density", p.scf_density);   // S8b.3
  io::tolower(p.scf_density);
  p.hyb_wmax    = io::get_value_with_default<double>(pt, "hyb_wmax", p.hyb_wmax);
  p.hyb_tau_eps = io::get_value_with_default<double>(pt, "hyb_tau_eps", p.hyb_tau_eps);
  utils::check(p.scf_density == "closure" or p.scf_density == "matsubara",
               "gw_line: scf_density must be \"closure\" or \"matsubara\" (got \"{}\")", p.scf_density);
  utils::check(p.hyb_wmax > 0.0 and p.hyb_tau_eps > 0.0 and p.hyb_tau_eps < 1.0, "gw_line: invalid hyb_wmax / hyb_tau_eps");
  utils::check(p.theta_t_frac > 0.0 and p.theta_t_frac < 1.0, "gw_line: theta_t_frac must be in (0, 1)");
  utils::check(p.beta >= 0.0, "gw_line: beta must be >= 0");
  utils::check(p.thermal_tol > 0.0 and p.thermal_tol < 1.0 and p.thermal_floor > 0.0 and p.thermal_floor_f > 0.0 and p.wp_floor >= 0.0,
               "gw_line: invalid thermal_tol / thermal_floor / thermal_floor_f / wp_floor");
  utils::check(p.mu_rule == "auto" or p.mu_rule == "gap" or p.mu_rule == "number",
               "gw_line: mu_rule must be \"auto\", \"gap\" or \"number\" (got \"{}\")", p.mu_rule);
  utils::check(p.band_heights >= 2 and p.band_x >= 3 and p.mats_factor > 0.0 and p.bos_eps_T > 0.0 and p.cut_odd > 0.0 and p.cut_even > 0.0,
               "gw_line: invalid finite-T W-step parameters");
  utils::check(p.tau_grid == "gl" or p.tau_grid == "id", "gw_line: tau_grid must be \"gl\" or \"id\" (got \"{}\")", p.tau_grid);
  p.optics = optics_params_t::from_ptree(pt);   // S9b
  p.optics_poles = io::get_value_with_default<std::string>(pt, "optics.poles", p.optics_poles);   // perf 7.2
  io::tolower(p.optics_poles);
  utils::check(p.optics_poles == "final" or p.optics_poles == "initial",
               "gw_line: optics.poles must be \"final\" or \"initial\" (got \"{}\")", p.optics_poles);
  // perf 7.2: mixing algorithm, warm start, multilevel schedule
  p.mix.alg    = io::get_value_with_default<std::string>(pt, "mixing_alg", p.mix.alg);
  io::tolower(p.mix.alg);
  p.mix.mixing = p.mixing;
  p.mix.hist   = io::get_value_with_default<long>(pt, "diis_hist", p.mix.hist);
  p.mix.start  = io::get_value_with_default<long>(pt, "diis_start", p.mix.start);
  p.mix.beta   = io::get_value_with_default<double>(pt, "diis_beta", p.mix.beta);
  p.mix.reg    = io::get_value_with_default<double>(pt, "diis_reg", p.mix.reg);
  p.mix.cmax   = io::get_value_with_default<double>(pt, "diis_cmax", p.mix.cmax);
  p.mix.grow   = io::get_value_with_default<double>(pt, "diis_grow", p.mix.grow);
  p.mix.mix_F  = io::get_value_with_default<bool>(pt, "diis_mix_F", p.mix.mix_F);
  p.mix.wF     = io::get_value_with_default<double>(pt, "diis_wF", p.mix.wF);
  p.mix.damp_below  = io::get_value_with_default<double>(pt, "damp_below", p.mix.damp_below);
  p.mix.damp_mixing = io::get_value_with_default<double>(pt, "damp_mixing", p.mix.damp_mixing);
  utils::check(p.mix.damp_below >= 0.0 and p.mix.damp_mixing > 0.0 and p.mix.damp_mixing <= 1.0,
               "gw_line: need damp_below >= 0 and damp_mixing in (0, 1]");
  utils::check(p.mix.alg == "linear" or p.mix.alg == "diis", "gw_line: mixing_alg must be \"linear\" or \"diis\" (got \"{}\")",
               p.mix.alg);
  utils::check(p.mix.hist >= 1 and p.mix.beta > 0.0 and p.mix.reg >= 0.0 and p.mix.cmax > 1.0 and p.mix.grow > 1.0,
               "gw_line: need diis_hist >= 1, diis_beta > 0, diis_reg >= 0, diis_cmax > 1, diis_grow > 1");
  p.start         = io::get_value_with_default<std::string>(pt, "start", p.start);
  io::tolower(p.start);
  p.start_file    = io::get_value_with_default<std::string>(pt, "start_file", p.start_file);
  p.start_dataset = io::get_value_with_default<std::string>(pt, "start_dataset", p.start_dataset);
  p.start_eta     = io::get_value_with_default<double>(pt, "start_eta", p.start_eta);
  utils::check(p.start == "ks" or p.start == "qp_diag" or p.start == "qp_file",
               "gw_line: start must be \"ks\", \"qp_diag\" or \"qp_file\" (got \"{}\")", p.start);
  utils::check(p.start != "qp_file" or not p.start_file.empty(), "gw_line: start = \"qp_file\" needs start_file");
  utils::check(p.start_eta > 0.0, "gw_line: start_eta must be > 0");
  p.coarse_niter         = io::get_value_with_default<long>(pt, "coarse.niter", p.coarse_niter);
  p.coarse_eps           = io::get_value_with_default<double>(pt, "coarse.eps", p.coarse_eps);
  p.coarse_K             = io::get_value_with_default<long>(pt, "coarse.K", p.coarse_K);
  p.coarse_nodes_per_ray = io::get_value_with_default<long>(pt, "coarse.nodes_per_ray", p.coarse_nodes_per_ray);
  p.coarse_time_eps      = io::get_value_with_default<double>(pt, "coarse.time_eps", p.coarse_eps);
  utils::check(p.coarse_niter >= 0 and p.coarse_eps > 0.0 and p.coarse_eps < 1.0 and p.coarse_K >= 1 and
                   p.coarse_nodes_per_ray > 1 and p.coarse_time_eps > 0.0 and p.coarse_time_eps < 1.0,
               "gw_line: invalid coarse = {{ niter, eps, K, nodes_per_ray, time_eps }}");
  utils::check(p.coarse_niter == 0 or p.bases_file.empty(), "gw_line: coarse.niter > 0 is incompatible with bases_file");
  utils::check(p.coarse_niter == 0 or p.niter >= p.coarse_niter + 2,
               "gw_line: niter ({}) must be >= coarse.niter + 2 ({}): the run ends with >= 2 production iterations", p.niter,
               p.coarse_niter + 2);
  utils::check(p.theta_deg > 0.0 and p.theta_deg < 90.0, "gw_line: theta_deg must be in (0, 90)");
  utils::check(p.eps > 0.0 and p.lam > 0.0, "gw_line: eps, lam must be > 0");
  utils::check(p.g_gap >= 0.0 and p.g_gap < p.lam, "gw_line: g_gap must be in [0, lam)");
  utils::check(p.g_repr == "lehmann" or p.g_repr == "compressed", "gw_line: g_repr must be \"lehmann\" or \"compressed\" (got \"{}\")",
               p.g_repr);
  if (p.g_emax < 0.0) p.g_emax = p.lam;
  // S7c: the bosonic basis must cover the spectrum of Pi, i.e. the summed energies e^> + |e^<| of the G poles (up to
  // 2 g_emax / 2 lam); a narrower lam_b makes the W fit residues ill-conditioned (|w_j| ~ 1e5 x W, measured on lih222)
  // and amplifies every time-grid error in Sigma ~1e5 x. lam_b <= 0 (the default): 2 g_emax (lehmann) / 2 lam (compressed)
  p.lam_b_auto = (p.lam_b <= 0.0);
  if (p.lam_b_auto) p.lam_b = 2.0 * (p.g_repr == "lehmann" ? p.g_emax : p.lam);
  utils::check(p.g_wtol >= 0.0 and p.g_emin_frac >= 0.0 and p.g_wsmall >= 0.0, "gw_line: g_wtol, g_emin_frac, g_wsmall must be >= 0");
  utils::check(p.nodes_per_ray > 1 and p.node_tmin > 0.0 and p.node_tmax > p.node_tmin, "gw_line: invalid node grid");
  utils::check(p.K >= 1 and p.nphi >= 1 and p.wp > 0.0 and p.tol_gram > 0.0, "gw_line: invalid closure parameters");
  utils::check(p.tol_gram_eps >= 0.0, "gw_line: tol_gram_eps must be >= 0");
  utils::check(p.tol_svd > 0.0 and p.closure_cut_window >= 1.0 and p.phase_keep >= 0.0 and p.debug_noise_h0 >= 0.0 and
                   p.debug_noise_sigma >= 0.0,
               "gw_line: invalid tol_svd / closure_cut_window / phase_keep / debug_noise_h0");
  utils::check(p.closure_cut == "hard" or p.closure_cut == "gap" or p.closure_cut == "smooth",
               "gw_line: closure_cut must be \"hard\", \"gap\" or \"smooth\" (got \"{}\")", p.closure_cut);
  utils::check(p.closure_svd_cut == "hard" or p.closure_svd_cut == "gap",
               "gw_line: closure_svd_cut must be \"hard\" or \"gap\" (got \"{}\")", p.closure_svd_cut);
  utils::check(p.closure_svd == "gesvd" or p.closure_svd == "gesdd", "gw_line: closure_svd must be \"gesvd\" or \"gesdd\" (got \"{}\")",
               p.closure_svd);
  utils::check(p.closure_ueig == "schur" or p.closure_ueig == "cayley",
               "gw_line: closure_ueig must be \"schur\" or \"cayley\" (got \"{}\")", p.closure_ueig);
  utils::check(p.closure_k_workers >= 1 or p.closure_k_workers == -1, "gw_line: closure_k_workers must be >= 1 or -1 (auto)");
  utils::check(p.closure_device == "auto" or p.closure_device == "on" or p.closure_device == "off",
               "gw_line: closure_device must be \"auto\", \"on\" or \"off\" (got \"{}\")", p.closure_device);
  utils::check(p.closure_dev_svd == "gesvd" or p.closure_dev_svd == "gesvdp",
               "gw_line: closure_dev_svd must be \"gesvd\" or \"gesvdp\" (got \"{}\")", p.closure_dev_svd);
  utils::check(p.niter >= 0 and p.t_chunk >= 0 and p.ray_decades > 0.0, "gw_line: invalid niter / t_chunk / ray_decades");
  utils::check(p.mixing > 0.0 and p.mixing <= 1.0, "gw_line: mixing must be in (0, 1]");
  utils::check(p.time_grid == "id" or p.time_grid == "gl", "gw_line: time_grid must be \"id\" or \"gl\" (got \"{}\")",
               p.time_grid);
  utils::check(p.checkpoint_sigma == "last" or p.checkpoint_sigma == "all",
               "gw_line: checkpoint_sigma must be \"last\" or \"all\" (got \"{}\")", p.checkpoint_sigma);
  utils::check(p.time_eps > 0.0 and p.time_eps < 1.0 and p.time_pad >= 1.0 and p.time_oversample >= 1.0 and p.time_snap >= 0.0,
               "gw_line: need 0 < time_eps < 1, time_pad >= 1, time_oversample >= 1");
  if (p.beta > 0.0) {   // S8b: what thermal iterations need
    utils::check(p.g_repr == "lehmann", "gw_line: beta > 0 needs g_repr = \"lehmann\" (per-pole thermal weights)");
    utils::check(p.start == "ks", "gw_line: beta > 0 supports start = \"ks\" only");
    utils::check(p.coarse_niter == 0, "gw_line: beta > 0 does not support the multilevel schedule (coarse.niter)");
    utils::check(not p.optics.enable, "gw_line: beta > 0 does not support the optics passes");
  }
  if (p.scf_density == "matsubara") {   // S8b.3 hybrid
    utils::check(p.beta > 0.0, "gw_line: scf_density = \"matsubara\" needs beta > 0");
    utils::check(p.mix.alg == "linear", "gw_line: scf_density = \"matsubara\" needs mixing_alg = \"linear\"");
    utils::check(p.div_treatment == "ignore_g0", "gw_line: scf_density = \"matsubara\" supports div_treatment = \"ignore_g0\" only");
  }
  return p;
}

void gw_line_params_t::log() const {
  app_log(1, "  gw_line parameters:");
  app_log(1, "    theta = {} deg (theta_t = {} deg), eps = {:.1e}, lam = {} Ha, lam_b = {} Ha{}", theta_deg, theta_deg / 2.0, eps,
          lam, lam_b, lam_b_auto ? " (auto: 2 x the G pole range)" : "");
  app_log(1, "    sigma_gap = {}{}, bos_gap = {}{}, g_gap = {}", sigma_gap, sigma_gap < 0.0 ? " (auto)" : "", bos_gap,
          bos_gap < 0.0 ? " (auto)" : "", g_gap);
  if (g_repr == "lehmann")
    app_log(1, "    G representation: lehmann (factorized v v^dagger; prune |e| > {} Ha, weight < {:.1e}, |e| < {} x half gap "
               "with weight < {:.1e})",
            g_emax, g_wtol, g_emin_frac, g_wsmall);
  else
    app_log(1, "    G representation: compressed (gapless per-sector refit, g_gap = {})", g_gap);
  app_log(1, "    fermionic nodes: {} per ray, |t| in [{}, {}] Ha", nodes_per_ray, node_tmin, node_tmax);
  app_log(1, "    closure: wp = {} Ha, K = {}, tol_gram = {:.1e} (used: {:.1e} = max(tol_gram, {} x eps)), nphi = {}, tol_svd = {:.1e}, "
             "cuts {} / {} (window {}), phase continuity {}",
          wp, K, tol_gram, std::max(tol_gram, tol_gram_eps * eps), tol_gram_eps, nphi, tol_svd, closure_cut, closure_svd_cut, closure_cut_window,
          phase_keep > 0.0 ? "on (x " + std::to_string(phase_keep) + ")" : std::string("off"));
  app_log(1, "    closure linear algebra: SVD {}, U eigenvectors {}, BLAS threads {}, k workers {}, device {} (SVD {})", closure_svd,
          closure_ueig, closure_threads < 0 ? std::string("auto") : std::to_string(closure_threads), closure_k_workers,
          closure_device, closure_dev_svd);
  if (debug_noise_h0 > 0.0)
    app_log(1, "    DIAGNOSTIC: relative noise {:.1e} on H0 (seed {})", debug_noise_h0, debug_noise_seed);
  if (debug_noise_sigma > 0.0)
    app_log(1, "    DIAGNOSTIC: relative noise {:.1e} on Sigma at the nodes in iteration {} (seed {})", debug_noise_sigma,
            debug_noise_iter, debug_noise_seed);
  if (not bases_file.empty()) app_log(1, "    DIAGNOSTIC: real-pole bases read from {}", bases_file);
  app_log(1, "    niter = {} (total), mixing = {}, conv_thr = {:.1e}, t_chunk = {}, ray_decades = {}", niter, mixing, conv_thr,
          t_chunk, ray_decades);
  if (mix.alg == "diis")
    app_log(1, "    mixing algorithm: DIIS (Anderson/Pulay on Sigma at the nodes{}): history {}, from iteration {} (before: linear "
               "{}), beta {}, reg {:.1e}, reset if max|c| > {} or |r| > {} x the best",
            mix.mix_F ? " and F" : "", mix.hist, mix.start, mixing, mix.beta, mix.reg, mix.cmax, mix.grow);
  else
    app_log(1, "    mixing algorithm: linear (Sigma at the nodes, F unmixed)");
  if (mix.damp_below > 0.0)
    app_log(1, "    damped tail: linear mixing {} once max|Sigma[G] - Sigma_in| < {:.1e}", mix.damp_mixing, mix.damp_below);
  if (start == "ks") app_log(1, "    start: KS poles");
  else if (start == "qp_diag")
    app_log(1, "    start: qp_diag (one Pi -> W -> Sigma pass on the KS poles, diagonal QP equation, eta {:.1e} Ha; KS vectors + QP "
               "energies)", start_eta);
  else app_log(1, "    start: qp_file (KS vectors + the QP energies of {}{})", start_file, start_dataset.empty() ? "" : ":" + start_dataset);
  if (coarse_niter > 0)
    app_log(1, "    multilevel: iterations 1..{} at eps {:.1e}, K {}, {} nodes per ray, time_eps {:.1e}; then production", coarse_niter,
            coarse_eps, coarse_K, coarse_nodes_per_ray, coarse_time_eps);
  if (time_grid == "id")
    app_log(1, "    time grid: ID (time_eps = {:.1e}, time_pad = {}, time_oversample = {}, time_snap = {}), rebuilt every iteration",
            time_eps, time_pad, time_oversample, time_snap);
  else
    app_log(1, "    time grid: GL rays (ray_decades = {}, 3 panels/e-fold, 16 nodes/panel)", ray_decades);
  app_log(1, "    restart = {}, checkpoint = {}.gw_line.h5 (Sigma at the nodes: {})", restart, output,
          checkpoint_sigma == "last" ? "last iteration only, in " + output + ".gw_line.sigma.h5" : "every iteration");
  app_log(1, "    IBZ reduction of a symmetric mean field: {} (ibz; env COQUI_GWLINE_IBZ)", ibz ? "on" : "off");
  app_log(1, "    Sigma at the nodes {} (sigma_kdist = {})", sigma_kdist ? "k-distributed (owner k mod np)" : "replicated on every rank",
          sigma_kdist);
  app_log(1, "    divergence: div_treatment = {} (Sigma_c head term {}), hf_div_treatment = {}; head extrapolation {}", div_treatment,
          div_treatment == "ignore_g0" ? "off" : "on", hf_div_treatment, head_extrapolation);
  if (do_spectra) {
    std::string e;
    for (auto x : spectra.eta) e += std::to_string(x) + " ";
    app_log(1, "    spectra: eta = [ {}] Ha, w - mu in [{}, {}] Ha, nw = {}", e, spectra.wmin, spectra.wmax, spectra.nw);
  } else {
    app_log(1, "    spectra: off");
  }
  app_log(1, "    theta_t = {} theta", theta_t_frac);
  if (beta > 0.0)
    app_log(1, "    finite temperature: beta = {} (T = {:.1f} K), thermal_tol = {:.1e} (E_T = {:.5f} Ha), c_zeta = {}, c_f = {}, wp_floor = {}, "
               "mu_rule = {} (mu_dn_max {}, mu_th_factor {}), band {} x {}, mats_factor {}, eps_b {:.1e} (line basis {:.1e}), split "
               "cuts {:.0e} / {:.0e}, tau grid {} (tau_eps {:.0e}){}",
            beta, 315775.02 / beta, thermal_tol, std::log(1.0 / thermal_tol) / beta, thermal_floor, thermal_floor_f, wp_floor, mu_rule,
            mu_dn_max, mu_th_factor, band_heights, band_x, mats_factor, bos_eps_T, bos_line_eps, cut_odd, cut_even, tau_grid, tau_eps,
            thermal_bases_file.empty() ? "" : "; DIAGNOSTIC: D / nu_b / Sigma basis from " + thermal_bases_file);
  if (scf_density == "matsubara")
    app_log(1, "    scf_density = matsubara (S8b.3 hybrid): tau-leg Sigma_c(i w_n), w_max {} Ha, tau ID {:.0e}; D, N, mu from the Matsubara "
               "Dyson equation", hyb_wmax, hyb_tau_eps);
  if (optics.enable) {
    optics.log();
    app_log(1, "    optics G: {} poles", optics_poles);
  } else app_log(1, "    optics: off");
}

// ------------------------------------------------------------------------------------------------------------------
// H0 and the system group
// ------------------------------------------------------------------------------------------------------------------
nda::array<ComplexType, 3> one_body_h0(mf::MF &mf) {
  auto mpi       = mf.mpi();
  const long nkI = mf.nkpts_ibz(), nb = mf.nbnd(), ns = mf.nspin();
  auto psp       = hamilt::make_pseudopot(mf);
  auto sH0       = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(*mpi, {ns, nkI, nb, nb});
  hamilt::set_H0(mf, psp.get(), sH0);
  auto H0l = sH0.local();
  nda::array<ComplexType, 3> H0(nkI, nb, nb);
  for (long ik = 0; ik < nkI; ++ik)
    for (long i = 0; i < nb; ++i)
      for (long j = 0; j < nb; ++j) H0(ik, i, j) = 0.5 * (H0l(0, ik, i, j) + std::conj(H0l(0, ik, j, i)));   // hermitize
  mpi->comm.barrier();
  return H0;
}

void write_system_h5(boost::mpi3::communicator &comm, std::string const &file, mf::MF &mf, long Np,
                     nda::array<ComplexType, 3> const &H0, double mu0, bool truncate) {
  if (comm.root()) {
    utils::h5_quiesce();
    h5::file f(file, truncate ? 'w' : 'a');
    h5::group g(f);
    auto s        = g.create_group("system");
    const long nk = mf.nkpts(), nb = H0.extent(1);
    nda::array<double, 2> eig(nk, nb);
    for (long ik = 0; ik < nk; ++ik)
      for (long n = 0; n < nb; ++n) eig(ik, n) = mf.eigval()(0, ik, n);
    h5::h5_write(s, "nkpts", nk);
    h5::h5_write(s, "nbnd", nb);
    h5::h5_write(s, "Np", Np);
    h5::h5_write(s, "nelec", double(mf.nelec()));
    h5::h5_write(s, "mu0", mu0);
    nda::h5_write(s, "H0", H0, false);
    nda::h5_write(s, "eigval", eig, false);
    nda::h5_write(s, "qk_to_k2", mf.qk_to_k2(), false);
    nda::h5_write(s, "kpoints", mf.kpts(), false);
    // perf 7.3: the k of the poles / Sigma / spectra are the first nkpts_ibz of kpoints; every k maps to kp_to_ibz (time-
    // reversed orbitals where kp_trev: G(k) = conj G(k_ibz))
    h5::h5_write(s, "nkpts_ibz", long(H0.extent(0)));
    nda::array<long, 1> k2i(nk), ktr(nk);
    for (long ik = 0; ik < nk; ++ik) {
      k2i(ik) = (H0.extent(0) < nk) ? long(mf.kp_to_ibz()(ik)) : ik;
      ktr(ik) = (H0.extent(0) < nk and mf.kp_trev()(ik)) ? 1 : 0;
    }
    nda::h5_write(s, "kp_to_ibz", k2i, false);
    nda::h5_write(s, "kp_trev", ktr, false);
  }
  comm.barrier();
}

namespace {

template <typename T, int R> void bcast_array(boost::mpi3::communicator &comm, nda::array<T, R> &A) {
  std::array<long, R> shp{};
  if (comm.root()) shp = A.shape();
  comm.broadcast_n(shp.data(), R, 0);
  if (not comm.root()) A.resize(shp);
  if (A.size() > 0) comm.broadcast_n(A.data(), A.size(), 0);
}

/// ragged pole storage per sector: counts (nk), flat e (sum M), and the residues: flat coef (sum M, nb, nb) (matrix form)
/// or flat v (sum M, nb) (factorized form; row = v_m)
void write_poles(h5::group &g, pole_data_t const &pd) {
  auto pg = g.create_group("poles");
  for (auto s : {sector_t::particle, sector_t::hole}) {
    const std::string nm = (s == sector_t::particle) ? "particle" : "hole";
    bool fact = true;
    nda::array<long, 1> cnt(pd.nk);
    long tot = 0;
    for (long ik = 0; ik < pd.nk; ++ik) {
      tot += (cnt(ik) = pd(ik, s).size());
      fact = fact and pd(ik, s).is_factorized();
    }
    nda::array<double, 1> e(tot);
    nda::array<ComplexType, 3> c(fact ? 0 : tot, pd.nb, pd.nb);
    nda::array<ComplexType, 2> v(fact ? tot : 0, pd.nb);
    long o = 0;
    for (long ik = 0; ik < pd.nk; ++ik) {
      auto const &ps = pd(ik, s);
      for (long m = 0; m < ps.size(); ++m, ++o) {
        e(o) = ps.e(m);
        if (fact)
          for (long i = 0; i < pd.nb; ++i) v(o, i) = ps.v(i, m);
        else
          c(o, nda::ellipsis{}) = ps.coef(m, nda::ellipsis{});
      }
    }
    nda::h5_write(pg, nm + "_counts", cnt, false);
    nda::h5_write(pg, nm + "_e", e, false);
    if (fact) nda::h5_write(pg, nm + "_v", v, false);
    else nda::h5_write(pg, nm + "_coef", c, false);
  }
}

pole_data_t read_poles(h5::group &g, long nk, long nb) {
  auto pg = g.open_group("poles");
  pole_data_t pd;
  pd.nk = nk;
  pd.nb = nb;
  pd.part.resize(nk);
  pd.hole.resize(nk);
  for (auto s : {sector_t::particle, sector_t::hole}) {
    const std::string nm = (s == sector_t::particle) ? "particle" : "hole";
    const bool fact = pg.has_dataset(nm + "_v");
    nda::array<long, 1> cnt;
    nda::array<double, 1> e;
    nda::array<ComplexType, 3> c;
    nda::array<ComplexType, 2> v;
    nda::h5_read(pg, nm + "_counts", cnt);
    nda::h5_read(pg, nm + "_e", e);
    if (fact) {
      nda::h5_read(pg, nm + "_v", v);
      utils::check(cnt.size() == nk and v.extent(1) == nb and v.extent(0) == e.size(), "gw_line restart: pole data shape mismatch");
    } else {
      nda::h5_read(pg, nm + "_coef", c);
      utils::check(cnt.size() == nk and c.extent(1) == nb, "gw_line restart: pole data shape mismatch");
    }
    long o = 0;
    for (long ik = 0; ik < nk; ++ik) {
      auto &ps = (s == sector_t::particle) ? pd.part[ik] : pd.hole[ik];
      nda::array<double, 1> es(e(nda::range(o, o + cnt(ik))));
      if (fact) {
        nda::array<ComplexType, 2> vs(nb, cnt(ik));
        for (long m = 0; m < cnt(ik); ++m)
          for (long i = 0; i < nb; ++i) vs(i, m) = v(o + m, i);
        ps = pole_sector_t::factorized_form(std::move(es), std::move(vs));
      } else {
        ps = pole_sector_t(std::move(es), nda::array<ComplexType, 3>(c(nda::range(o, o + cnt(ik)), nda::range::all, nda::range::all)));
      }
      o += cnt(ik);
    }
  }
  return pd;
}

void bcast_poles(boost::mpi3::communicator &comm, pole_data_t &pd) {
  std::array<long, 2> d = {pd.nk, pd.nb};
  comm.broadcast_n(d.data(), 2, 0);
  if (not comm.root()) {
    pd.nk = d[0];
    pd.nb = d[1];
    pd.part.resize(pd.nk);
    pd.hole.resize(pd.nk);
  }
  for (long ik = 0; ik < pd.nk; ++ik) {
    for (auto *ps : {&pd.part[ik], &pd.hole[ik]}) {
      int f = ps->factorized ? 1 : 0;
      comm.broadcast_n(&f, 1, 0);
      ps->factorized = (f != 0);
      bcast_array(comm, ps->e);
      if (ps->factorized) bcast_array(comm, ps->v);
      else bcast_array(comm, ps->coef);
    }
  }
}

void write_history(h5::group &g, gw_line_iter_t const &r) {
  auto hg = g.create_group("history");
  h5::h5_write(hg, "iter", r.iter);
  h5::h5_write(hg, "dSigma", r.dSigma);
  h5::h5_write(hg, "mu", r.mu);
  h5::h5_write(hg, "dmu", r.dmu);
  h5::h5_write(hg, "gap", r.gap);
  h5::h5_write(hg, "e_homo", r.e_homo);
  h5::h5_write(hg, "e_lumo", r.e_lumo);
  h5::h5_write(hg, "nelec", r.nelec);
  h5::h5_write(hg, "nelec_lehmann", r.nelec_lehmann);
  h5::h5_write(hg, "N_mu", r.N_mu);
  h5::h5_write(hg, "dropped_weight", r.dropped);
  h5::h5_write(hg, "heldout_max", r.heldout_max);
  h5::h5_write(hg, "npoles_min", r.npoles_min);
  h5::h5_write(hg, "npoles_max", r.npoles_max);
  h5::h5_write(hg, "bos_gap", r.bos_gap);
  h5::h5_write(hg, "sigma_gap_p", r.sigma_gap_p);
  h5::h5_write(hg, "sigma_gap_h", r.sigma_gap_h);
  h5::h5_write(hg, "time", r.time);
  h5::h5_write(hg, "time_grid", r.time_grid);
  h5::h5_write(hg, "nt_pi_p", r.nt_pi_p);
  h5::h5_write(hg, "nt_pi_h", r.nt_pi_h);
  h5::h5_write(hg, "nt_sigma_p", r.nt_sig_p);
  h5::h5_write(hg, "nt_sigma_h", r.nt_sig_h);
  h5::h5_write(hg, "g_repr", r.g_repr);
  h5::h5_write(hg, "ng_min", r.ng_min);
  h5::h5_write(hg, "ng_max", r.ng_max);
  h5::h5_write(hg, "g_emin", r.g_emin);
  h5::h5_write(hg, "pruned_w", r.pruned_w);
  h5::h5_write(hg, "pruned_w_weight", r.pruned_w_weight);
  h5::h5_write(hg, "pruned_near", r.pruned_near);
  h5::h5_write(hg, "pruned_near_weight", r.pruned_near_weight);
  h5::h5_write(hg, "resid", r.resid);   // perf 7.2
  h5::h5_write(hg, "residF", r.residF);
  h5::h5_write(hg, "mix", r.mix);
  h5::h5_write(hg, "ndiis", r.ndiis);
  h5::h5_write(hg, "level", r.level);
  if (not r.mu_rule.empty()) {   // S8b
    auto tg = hg.create_group("thermal");
    h5::h5_write(tg, "thermal", r.thermal);
    h5::h5_write(tg, "mu_rule", r.mu_rule);
    h5::h5_write(tg, "dN", r.dN);
    h5::h5_write(tg, "n_th", r.n_th);
    h5::h5_write(tg, "wp_used", r.wp_used);
    h5::h5_write(tg, "nD", r.nD);
    h5::h5_write(tg, "rank_b", r.rank_b);
    h5::h5_write(tg, "ntau", r.ntau);
    h5::h5_write(tg, "nwin", r.nwin);
    if (r.nfreq > 0) {   // S8b.3
      h5::h5_write(tg, "N_trace", r.N_trace);
      h5::h5_write(tg, "N_closure", r.N_closure);
      h5::h5_write(tg, "D_cl_err", r.D_cl_err);
      h5::h5_write(tg, "dmu_closure", r.dmu_closure);
      h5::h5_write(tg, "rule_closure", r.rule_closure);
      h5::h5_write(tg, "tail_max", r.tail_max);
      h5::h5_write(tg, "nfreq", r.nfreq);
    }
  }
}

gw_line_iter_t read_history(h5::group &g) {
  auto hg = g.open_group("history");
  gw_line_iter_t r;
  h5::h5_read(hg, "iter", r.iter);
  h5::h5_read(hg, "dSigma", r.dSigma);
  h5::h5_read(hg, "mu", r.mu);
  h5::h5_read(hg, "dmu", r.dmu);
  h5::h5_read(hg, "gap", r.gap);
  h5::h5_read(hg, "e_homo", r.e_homo);
  h5::h5_read(hg, "e_lumo", r.e_lumo);
  h5::h5_read(hg, "nelec", r.nelec);
  h5::h5_read(hg, "nelec_lehmann", r.nelec_lehmann);
  h5::h5_read(hg, "N_mu", r.N_mu);
  h5::h5_read(hg, "dropped_weight", r.dropped);
  h5::h5_read(hg, "heldout_max", r.heldout_max);
  h5::h5_read(hg, "npoles_min", r.npoles_min);
  h5::h5_read(hg, "npoles_max", r.npoles_max);
  h5::h5_read(hg, "bos_gap", r.bos_gap);
  h5::h5_read(hg, "sigma_gap_p", r.sigma_gap_p);
  h5::h5_read(hg, "sigma_gap_h", r.sigma_gap_h);
  h5::h5_read(hg, "time", r.time);
  if (hg.has_dataset("time_grid")) {   // absent in S6 checkpoints (GL rays)
    h5::h5_read(hg, "time_grid", r.time_grid);
    h5::h5_read(hg, "nt_pi_p", r.nt_pi_p);
    h5::h5_read(hg, "nt_pi_h", r.nt_pi_h);
    h5::h5_read(hg, "nt_sigma_p", r.nt_sig_p);
    h5::h5_read(hg, "nt_sigma_h", r.nt_sig_h);
  }
  if (hg.has_dataset("g_repr")) {   // absent before S7c (compressed poles)
    h5::h5_read(hg, "g_repr", r.g_repr);
    h5::h5_read(hg, "ng_min", r.ng_min);
    h5::h5_read(hg, "ng_max", r.ng_max);
    h5::h5_read(hg, "g_emin", r.g_emin);
    h5::h5_read(hg, "pruned_w", r.pruned_w);
    h5::h5_read(hg, "pruned_w_weight", r.pruned_w_weight);
    h5::h5_read(hg, "pruned_near", r.pruned_near);
    h5::h5_read(hg, "pruned_near_weight", r.pruned_near_weight);
  }
  if (hg.has_dataset("resid")) {   // absent before perf 7.2
    h5::h5_read(hg, "resid", r.resid);
    h5::h5_read(hg, "residF", r.residF);
    h5::h5_read(hg, "mix", r.mix);
    h5::h5_read(hg, "ndiis", r.ndiis);
    h5::h5_read(hg, "level", r.level);
  }
  if (hg.has_subgroup("thermal")) {   // S8b
    auto tg = hg.open_group("thermal");
    h5::h5_read(tg, "thermal", r.thermal);
    h5::h5_read(tg, "mu_rule", r.mu_rule);
    h5::h5_read(tg, "dN", r.dN);
    h5::h5_read(tg, "n_th", r.n_th);
    h5::h5_read(tg, "wp_used", r.wp_used);
  }
  return r;
}

void write_input(h5::group &g, gw_line_params_t const &p, nda::array<ComplexType, 1> const &zeta) {
  auto ig = g.create_group("input");
  h5::h5_write(ig, "theta_deg", p.theta_deg);
  h5::h5_write(ig, "eps", p.eps);
  h5::h5_write(ig, "lam", p.lam);
  h5::h5_write(ig, "lam_b", p.lam_b);
  h5::h5_write(ig, "sigma_gap", p.sigma_gap);
  h5::h5_write(ig, "bos_gap", p.bos_gap);
  h5::h5_write(ig, "g_gap", p.g_gap);
  h5::h5_write(ig, "g_repr", p.g_repr);
  h5::h5_write(ig, "g_emax", p.g_emax);
  h5::h5_write(ig, "g_wtol", p.g_wtol);
  h5::h5_write(ig, "g_emin_frac", p.g_emin_frac);
  h5::h5_write(ig, "g_wsmall", p.g_wsmall);
  h5::h5_write(ig, "nodes_per_ray", p.nodes_per_ray);
  h5::h5_write(ig, "node_tmin", p.node_tmin);
  h5::h5_write(ig, "node_tmax", p.node_tmax);
  h5::h5_write(ig, "wp", p.wp);
  h5::h5_write(ig, "K", p.K);
  h5::h5_write(ig, "tol_gram", p.tol_gram);
  h5::h5_write(ig, "nphi", p.nphi);
  h5::h5_write(ig, "tol_svd", p.tol_svd);
  h5::h5_write(ig, "tol_gram_eps", p.tol_gram_eps);
  h5::h5_write(ig, "closure_cut", p.closure_cut);
  h5::h5_write(ig, "closure_svd_cut", p.closure_svd_cut);
  h5::h5_write(ig, "closure_cut_window", p.closure_cut_window);
  h5::h5_write(ig, "phase_keep", p.phase_keep);
  h5::h5_write(ig, "closure_svd", p.closure_svd);
  h5::h5_write(ig, "closure_ueig", p.closure_ueig);
  h5::h5_write(ig, "closure_device", p.closure_device);
  h5::h5_write(ig, "closure_k_workers", p.closure_k_workers);
  h5::h5_write(ig, "closure_dev_svd", p.closure_dev_svd);
  h5::h5_write(ig, "mixing", p.mixing);
  h5::h5_write(ig, "conv_thr", p.conv_thr);
  h5::h5_write(ig, "t_chunk", p.t_chunk);
  h5::h5_write(ig, "ray_decades", p.ray_decades);
  h5::h5_write(ig, "time_grid", p.time_grid);
  h5::h5_write(ig, "time_eps", p.time_eps);
  h5::h5_write(ig, "time_pad", p.time_pad);
  h5::h5_write(ig, "time_oversample", p.time_oversample);
  h5::h5_write(ig, "time_snap", p.time_snap);
  h5::h5_write(ig, "checkpoint_sigma", p.checkpoint_sigma);
  h5::h5_write(ig, "div_treatment", p.div_treatment);
  h5::h5_write(ig, "hf_div_treatment", p.hf_div_treatment);
  h5::h5_write(ig, "head_extrapolation", p.head_extrapolation);
  h5::h5_write(ig, "mixing_alg", p.mix.alg);   // perf 7.2
  h5::h5_write(ig, "diis_hist", p.mix.hist);
  h5::h5_write(ig, "diis_start", p.mix.start);
  h5::h5_write(ig, "diis_beta", p.mix.beta);
  h5::h5_write(ig, "diis_mix_F", long(p.mix.mix_F ? 1 : 0));
  h5::h5_write(ig, "start", p.start);
  h5::h5_write(ig, "start_file", p.start_file);
  h5::h5_write(ig, "coarse_niter", p.coarse_niter);
  h5::h5_write(ig, "coarse_eps", p.coarse_eps);
  h5::h5_write(ig, "coarse_K", p.coarse_K);
  h5::h5_write(ig, "coarse_nodes_per_ray", p.coarse_nodes_per_ray);
  h5::h5_write(ig, "coarse_time_eps", p.coarse_time_eps);
  h5::h5_write(ig, "theta_t_frac", p.theta_t_frac);   // S8b
  h5::h5_write(ig, "beta", p.beta);
  h5::h5_write(ig, "thermal_tol", p.thermal_tol);
  h5::h5_write(ig, "thermal_floor", p.thermal_floor);
  h5::h5_write(ig, "thermal_floor_f", p.thermal_floor_f);
  h5::h5_write(ig, "wp_floor", p.wp_floor);
  h5::h5_write(ig, "mu_rule", p.mu_rule);
  h5::h5_write(ig, "mu_dn_max", p.mu_dn_max);
  h5::h5_write(ig, "mu_th_factor", p.mu_th_factor);
  h5::h5_write(ig, "tau_grid", p.tau_grid);
  h5::h5_write(ig, "scf_density", p.scf_density);   // S8b.3
  h5::h5_write(ig, "hyb_wmax", p.hyb_wmax);
  h5::h5_write(ig, "hyb_tau_eps", p.hyb_tau_eps);
  if (p.beta > 0.0) {   // S8b: the derived scales (E_T window, zeta_T bosonic floor, S_T guard)
    const double th = p.theta_deg * std::numbers::pi / 180.0, tt = p.theta_t_frac * th;
    const double rho = std::sin(th - tt) / std::sin(tt);
    h5::h5_write(ig, "E_T", std::log(1.0 / p.thermal_tol) / p.beta);
    h5::h5_write(ig, "zeta_T", p.thermal_floor / (rho * p.beta));
    h5::h5_write(ig, "zeta_T_f", p.thermal_floor_f / (rho * p.beta));
    h5::h5_write(ig, "S_T", p.beta / std::sin(tt));
  }
  nda::h5_write(ig, "fermionic_nodes", zeta, false);
}

/// The SCF state carried from one iteration to the next (exactly what a restart needs).
struct state_t {
  long iter = 0;
  double mu = 0.0, mu_sigma = 0.0, dmu = 0.0, e_homo = 0.0, e_lumo = 0.0;
  nda::array<ComplexType, 3> F;
  nda::array<ComplexType, 3> F_cl;   ///< perf 7.2: the F of the closure that built `poles` (iteration 0: F)
  pole_data_t poles;
  bool have_sigma = false;
  nda::array<ComplexType, 4> Sig_p, Sig_h;
  nda::array<double, 1> phi;   ///< terminal phase phi* of the last closure per k (S7f phase continuity; empty = none)
  bool have_stau = false;      ///< S8b.3 hybrid: the mixed tau-leg nodal Sigma (hybrid.hpp), k-distributed like Sig_p
  nda::array<ComplexType, 4> Stau_p, Stau_h;
};

/// S9a: the head data of one iteration (scf_line/iter<N>/head/, see driver.hpp)
struct head_out_t {
  nda::array<ComplexType, 2> h_nodes, h_res, h_res_hole;   ///< (nq, nz_b), (nq, r_b), (nq, r_b)
  nda::array<ComplexType, 1> h0_nodes, h0_res, h0_res_hole, zeta;
  nda::array<double, 1> nu, q_weights;
  nda::array<double, 2> qpts;
  double madelung = 0.0, eps_inf = 1.0;
  std::string extrapolation, div_treatment, hf_div_treatment;
};

void write_head(h5::group &it, head_out_t const &h) {
  auto hg = it.create_group("head");
  nda::h5_write(hg, "h_nodes", h.h_nodes, false);
  nda::h5_write(hg, "h_res", h.h_res, false);
  nda::h5_write(hg, "h_res_hole", h.h_res_hole, false);
  nda::h5_write(hg, "h0_nodes", h.h0_nodes, false);
  nda::h5_write(hg, "h0_res", h.h0_res, false);
  nda::h5_write(hg, "h0_res_hole", h.h0_res_hole, false);
  nda::h5_write(hg, "zeta", h.zeta, false);
  nda::h5_write(hg, "nu", h.nu, false);
  nda::h5_write(hg, "q_weights", h.q_weights, false);
  nda::h5_write(hg, "qpts", h.qpts, false);
  h5::h5_write(hg, "madelung", h.madelung);
  h5::h5_write(hg, "eps_inf", h.eps_inf);
  h5::h5_write(hg, "extrapolation", h.extrapolation);
  h5::h5_write(hg, "div_treatment", h.div_treatment);
  h5::h5_write(hg, "hf_div_treatment", h.hf_div_treatment);
}

/// S9b: the head group of scf_line/iter<it>/ (root reads, broadcast); false if the checkpoint has none (pre-S9a)
bool read_head_nodes(boost::mpi3::communicator &comm, std::string const &file, long it, nda::array<ComplexType, 2> &h,
                     nda::array<ComplexType, 1> &zeta, nda::array<double, 1> &nu) {
  long ok = 0;
  if (comm.root()) {
    utils::h5_quiesce();
    h5::file f(file, 'r');
    h5::group g(f);
    if (g.has_subgroup("scf_line")) {
      auto sg = g.open_group("scf_line");
      const std::string nm = "iter" + std::to_string(it);
      if (sg.has_subgroup(nm) and sg.open_group(nm).has_subgroup("head")) {
        auto hg = sg.open_group(nm).open_group("head");
        nda::h5_read(hg, "h_nodes", h);
        nda::h5_read(hg, "zeta", zeta);
        nda::h5_read(hg, "nu", nu);
        ok = 1;
      }
    }
  }
  comm.broadcast_n(&ok, 1, 0);
  if (ok == 0) return false;
  bcast_array(comm, h);
  bcast_array(comm, zeta);
  bcast_array(comm, nu);
  return true;
}

/// Sigma file of checkpoint_sigma = "last" (S7e): <output>.gw_line.sigma.h5, rewritten every iteration (tmp + rename)
std::string sigma_file(std::string const &chk) { return chk.substr(0, chk.size() - 3) + ".sigma.h5"; }

/**
 * Root writes iter<N>. Sigma (k-distributed when kd != nullptr: gathered to the root first, collective) goes to
 * iter<N>/Sigma_{p,h} (sigma_all) or, checkpoint_sigma = "last" (S7e), to the separate file sigma_file(file) that is
 * rewritten every iteration (constant size; before S7e the checkpoint grew by 2 N_k N_zeta nb^2 x 16 B per iteration,
 * 1.8 GB for Si 4x4x4 nb 60).
 */
void write_state(boost::mpi3::communicator &comm, std::string const &file, state_t const &st,
                 gw_line_iter_t const *rec, k_dist_t const *kd = nullptr, bool sigma_all = true,
                 head_out_t const *head = nullptr) {
  nda::array<ComplexType, 4> Sp_full, Sh_full;
  if (st.have_sigma and kd != nullptr) {
    Sp_full = kd_gather_full(comm, *kd, st.Sig_p);
    Sh_full = kd_gather_full(comm, *kd, st.Sig_h);
  }
  auto const &Sp = (kd != nullptr) ? Sp_full : st.Sig_p;
  auto const &Sh = (kd != nullptr) ? Sh_full : st.Sig_h;
  nda::array<ComplexType, 4> Tp_full, Th_full;   // S8b.3
  if (st.have_stau and kd != nullptr) {
    Tp_full = kd_gather_full(comm, *kd, st.Stau_p);
    Th_full = kd_gather_full(comm, *kd, st.Stau_h);
  }
  auto const &Tp = (kd != nullptr) ? Tp_full : st.Stau_p;
  auto const &Th = (kd != nullptr) ? Th_full : st.Stau_h;
  if (comm.root() and st.have_sigma and not sigma_all) {
    utils::h5_quiesce();
    const std::string sf = sigma_file(file), tmp = sf + ".tmp";
    {
      h5::file f(tmp, 'w');
      h5::group g(f);
      h5::h5_write(g, "iter", st.iter);
      nda::h5_write(g, "Sigma_p", Sp, false);
      nda::h5_write(g, "Sigma_h", Sh, false);
      if (st.have_stau) {
        nda::h5_write(g, "Sigma_tau_p", Tp, false);
        nda::h5_write(g, "Sigma_tau_h", Th, false);
      }
    }
    std::filesystem::rename(tmp, sf);
  }
  if (comm.root()) {
    utils::h5_quiesce();
    h5::file f(file, 'a');
    h5::group g(f);
    auto sg = g.has_subgroup("scf_line") ? g.open_group("scf_line") : g.create_group("scf_line");
    auto it = sg.create_group("iter" + std::to_string(st.iter));
    h5::h5_write(it, "mu", st.mu);
    h5::h5_write(it, "mu_sigma", st.mu_sigma);
    h5::h5_write(it, "dmu", st.dmu);
    h5::h5_write(it, "e_homo", st.e_homo);
    h5::h5_write(it, "e_lumo", st.e_lumo);
    nda::h5_write(it, "F", st.F, false);
    nda::h5_write(it, "F_closure", st.F_cl, false);   // perf 7.2
    if (st.have_sigma and sigma_all) {
      nda::h5_write(it, "Sigma_p", Sp, false);
      nda::h5_write(it, "Sigma_h", Sh, false);
      if (st.have_stau) {
        nda::h5_write(it, "Sigma_tau_p", Tp, false);
        nda::h5_write(it, "Sigma_tau_h", Th, false);
      }
    }
    h5::h5_write(it, "has_sigma", long(st.have_sigma ? 1 : 0));
    if (st.phi.size() > 0) nda::h5_write(it, "closure_phi", st.phi, false);
    write_poles(it, st.poles);
    if (rec != nullptr) write_history(it, *rec);
    if (head != nullptr) write_head(it, *head);
    h5::h5_write(sg, "final_iter", st.iter);
  }
  comm.barrier();
}

static constexpr int NHIST = 38;   ///< columns of the broadcast history table
static const std::vector<std::string> mix_names = {"none", "linear", "diis", "reset", "damped"};
double mix_code(std::string const &m) {
  auto it = std::find(mix_names.begin(), mix_names.end(), m);
  return it == mix_names.end() ? 0.0 : double(std::distance(mix_names.begin(), it));
}
std::string mix_name(long c) { return (c >= 0 and c < long(mix_names.size())) ? mix_names[c] : std::string("none"); }

/// Root reads scf_line/final_iter (+ the history of iterations 1..final_iter), everything is broadcast.
state_t read_state(boost::mpi3::communicator &comm, std::string const &file, long nk, long nb,
                   nda::array<ComplexType, 1> const &zeta, std::vector<gw_line_iter_t> &history, k_dist_t const *kd = nullptr) {
  state_t st;
  long nhist = 0;
  nda::array<double, 2> hist;
  if (comm.root()) {
    utils::h5_quiesce();
    h5::file f(file, 'r');
    h5::group g(f);
    {
      auto ig = g.open_group("input");
      nda::array<ComplexType, 1> z;
      nda::h5_read(ig, "fermionic_nodes", z);
      utils::check(z.size() == zeta.size() and nda::max_element(nda::abs(z - zeta)) <= 1e-14 * nda::max_element(nda::abs(zeta)),
                   "gw_line restart: the fermionic nodes of {} differ from the input (node grid / theta changed)", file);
    }
    auto sg = g.open_group("scf_line");
    h5::h5_read(sg, "final_iter", st.iter);
    auto it = sg.open_group("iter" + std::to_string(st.iter));
    h5::h5_read(it, "mu", st.mu);
    h5::h5_read(it, "mu_sigma", st.mu_sigma);
    h5::h5_read(it, "dmu", st.dmu);
    h5::h5_read(it, "e_homo", st.e_homo);
    h5::h5_read(it, "e_lumo", st.e_lumo);
    nda::h5_read(it, "F", st.F);
    if (it.has_dataset("F_closure")) nda::h5_read(it, "F_closure", st.F_cl);   // perf 7.2
    else {
      app_log(1, "  restart: {} iteration {} has no F_closure (pre-7.2 checkpoint): the spectra use F[D] of the final poles", file,
              st.iter);
      st.F_cl = st.F;
    }
    st.have_sigma = it.has_dataset("Sigma_p");
    if (st.have_sigma) {
      nda::h5_read(it, "Sigma_p", st.Sig_p);
      nda::h5_read(it, "Sigma_h", st.Sig_h);
      st.have_stau = it.has_dataset("Sigma_tau_p");
      if (st.have_stau) {
        nda::h5_read(it, "Sigma_tau_p", st.Stau_p);
        nda::h5_read(it, "Sigma_tau_h", st.Stau_h);
      }
    } else if (it.has_dataset("has_sigma")) {   // S7e checkpoint_sigma = "last": the separate Sigma file
      long hs = 0;
      h5::h5_read(it, "has_sigma", hs);
      if (hs != 0) {
        const std::string sf = sigma_file(file);
        utils::check(std::filesystem::exists(sf), "gw_line restart: {} has no Sigma for iteration {} and {} is missing", file,
                     st.iter, sf);
        h5::file fs(sf, 'r');
        h5::group gs(fs);
        long si = -1;
        h5::h5_read(gs, "iter", si);
        utils::check(si == st.iter, "gw_line restart: {} holds Sigma of iteration {}, the checkpoint ends at {}", sf, si, st.iter);
        nda::h5_read(gs, "Sigma_p", st.Sig_p);
        nda::h5_read(gs, "Sigma_h", st.Sig_h);
        st.have_sigma = true;
        st.have_stau  = gs.has_dataset("Sigma_tau_p");
        if (st.have_stau) {
          nda::h5_read(gs, "Sigma_tau_p", st.Stau_p);
          nda::h5_read(gs, "Sigma_tau_h", st.Stau_h);
        }
      }
    }
    st.poles = read_poles(it, nk, nb);
    if (it.has_dataset("closure_phi")) nda::h5_read(it, "closure_phi", st.phi);
    utils::check(st.F.extent(0) == nk and st.F.extent(1) == nb, "gw_line restart: F shape mismatch");
    nhist = st.iter;
    hist  = nda::array<double, 2>(nhist, NHIST);
    for (long i = 1; i <= st.iter; ++i) {
      auto gi = sg.open_group("iter" + std::to_string(i));
      auto r  = read_history(gi);
      double v[NHIST] = {double(r.iter), r.dSigma, r.mu, r.dmu, r.gap, r.e_homo, r.e_lumo, r.nelec, r.nelec_lehmann, r.N_mu,
                         r.dropped, r.heldout_max, double(r.npoles_min), double(r.npoles_max), r.bos_gap, r.sigma_gap_p,
                         r.sigma_gap_h, r.time, r.time_grid == "id" ? 1.0 : 0.0, double(r.nt_pi_p), double(r.nt_pi_h),
                         double(r.nt_sig_p), double(r.nt_sig_h), r.g_repr == "lehmann" ? 1.0 : 0.0, double(r.ng_min),
                         double(r.ng_max), r.g_emin, double(r.pruned_w), r.pruned_w_weight, double(r.pruned_near),
                         r.pruned_near_weight, r.resid, r.residF, mix_code(r.mix), double(r.ndiis), double(r.level),
                         double(r.thermal), r.wp_used};
      for (int j = 0; j < NHIST; ++j) hist(i - 1, j) = v[j];
    }
  }
  std::array<double, 8> sc = {double(st.iter), st.mu, st.mu_sigma, st.dmu, st.e_homo, st.e_lumo, st.have_sigma ? 1.0 : 0.0,
                              st.have_stau ? 1.0 : 0.0};
  comm.broadcast_n(sc.data(), sc.size(), 0);
  st.have_stau  = sc[7] > 0.5;
  st.iter       = long(std::llround(sc[0]));
  st.mu         = sc[1];
  st.mu_sigma   = sc[2];
  st.dmu        = sc[3];
  st.e_homo     = sc[4];
  st.e_lumo     = sc[5];
  st.have_sigma = sc[6] > 0.5;
  bcast_array(comm, st.F);
  bcast_array(comm, st.F_cl);
  if (st.have_sigma and kd != nullptr) {   // k-distributed: every rank receives the rows of its k only
    st.Sig_p = kd_scatter_full(comm, *kd, st.Sig_p);
    st.Sig_h = kd_scatter_full(comm, *kd, st.Sig_h);
  } else if (st.have_sigma) {
    bcast_array(comm, st.Sig_p);
    bcast_array(comm, st.Sig_h);
  }
  if (st.have_stau and kd != nullptr) {   // S8b.3
    st.Stau_p = kd_scatter_full(comm, *kd, st.Stau_p);
    st.Stau_h = kd_scatter_full(comm, *kd, st.Stau_h);
  } else if (st.have_stau) {
    bcast_array(comm, st.Stau_p);
    bcast_array(comm, st.Stau_h);
  }
  bcast_poles(comm, st.poles);
  bcast_array(comm, st.phi);
  bcast_array(comm, hist);
  history.clear();
  for (long i = 0; i < hist.extent(0); ++i) {
    gw_line_iter_t r;
    r.iter = long(std::llround(hist(i, 0)));
    r.dSigma = hist(i, 1); r.mu = hist(i, 2); r.dmu = hist(i, 3); r.gap = hist(i, 4); r.e_homo = hist(i, 5);
    r.e_lumo = hist(i, 6); r.nelec = hist(i, 7); r.nelec_lehmann = hist(i, 8); r.N_mu = hist(i, 9); r.dropped = hist(i, 10);
    r.heldout_max = hist(i, 11); r.npoles_min = long(std::llround(hist(i, 12))); r.npoles_max = long(std::llround(hist(i, 13)));
    r.bos_gap = hist(i, 14); r.sigma_gap_p = hist(i, 15); r.sigma_gap_h = hist(i, 16); r.time = hist(i, 17);
    r.time_grid = hist(i, 18) > 0.5 ? "id" : "gl";
    r.nt_pi_p = long(std::llround(hist(i, 19))); r.nt_pi_h = long(std::llround(hist(i, 20)));
    r.nt_sig_p = long(std::llround(hist(i, 21))); r.nt_sig_h = long(std::llround(hist(i, 22)));
    r.g_repr = hist(i, 23) > 0.5 ? "lehmann" : "compressed";
    r.ng_min = long(std::llround(hist(i, 24))); r.ng_max = long(std::llround(hist(i, 25))); r.g_emin = hist(i, 26);
    r.pruned_w = long(std::llround(hist(i, 27))); r.pruned_w_weight = hist(i, 28);
    r.pruned_near = long(std::llround(hist(i, 29))); r.pruned_near_weight = hist(i, 30);
    r.resid = hist(i, 31); r.residF = hist(i, 32); r.mix = mix_name(long(std::llround(hist(i, 33))));
    r.ndiis = long(std::llround(hist(i, 34))); r.level = long(std::llround(hist(i, 35)));
    r.thermal = long(std::llround(hist(i, 36))); r.wp_used = hist(i, 37);   // S8b
    history.push_back(r);
  }
  return st;
}

void write_spectra(boost::mpi3::communicator &comm, std::string const &file, spectra_out_t const &s, double mu) {
  if (comm.root()) {
    utils::h5_quiesce();
    h5::file f(file, 'a');
    h5::group g(f);
    auto sg = g.create_group("spectra");
    h5::h5_write(sg, "mu", mu);
    nda::h5_write(sg, "omega", s.omega, false);
    nda::h5_write(sg, "eta", s.eta, false);
    nda::h5_write(sg, "A_k_w_diag", s.A_diag, false);
    nda::h5_write(sg, "A_k_w_trace", s.A_trace, false);
    h5::h5_write(sg, "e_homo", s.e_homo);
    h5::h5_write(sg, "e_lumo", s.e_lumo);
    h5::h5_write(sg, "vbm", mu + s.e_homo);
    h5::h5_write(sg, "cbm", mu + s.e_lumo);
    h5::h5_write(sg, "gap", s.e_lumo - s.e_homo);
  }
  comm.barrier();
}

/// retained G poles per k and sector (min, max) and the smallest retained |e|
void pole_counts(pole_data_t const &pd, long &nmin, long &nmax, double &emin) {
  nmin = std::numeric_limits<long>::max();
  nmax = 0;
  for (long ik = 0; ik < pd.nk; ++ik)
    for (auto const *ps : {&pd.part[ik], &pd.hole[ik]}) {
      nmin = std::min(nmin, ps->size());
      nmax = std::max(nmax, ps->size());
    }
  emin = pd.emin();
}


// perf 7.4b: q_plan_t / choose_q_plan moved to q_plan.hpp (automatic q groups from the 6.7 memory model)

// ------------------------------------------------------------------------------------------------------------------
// S7e instrumentation: per-iteration phase timers (min / avg / max over ranks) and host / device high-water memory
// ------------------------------------------------------------------------------------------------------------------
/// high-water resident set of this process (bytes): getrusage ru_maxrss (Linux: kB, = VmHWM of /proc/self/status;
/// macOS: bytes)
double host_hwm_bytes() {
  struct rusage ru {};
  getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
  return double(ru.ru_maxrss);
#else
  return double(ru.ru_maxrss) * 1024.0;
#endif
}
/// current resident set (bytes): VmRSS of /proc/self/status (0 where unavailable)
double host_rss_bytes() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("VmRSS:", 0) == 0) return 1024.0 * std::strtod(line.c_str() + 6, nullptr);
  return 0.0;
}

/// perf 7.4b: env COQUI_GWLINE_MEMTRACE = 1: VmRSS / VmHWM per rank (max, min over the ranks; the rank of the max) and the
/// smallest MemAvailable over the nodes at a point of the iteration (collective; diagnostics of the memory model)
template <typename ncomm_t>
void mem_trace(boost::mpi3::communicator &comm, ncomm_t &node_comm, std::string const &tag) {
  static const bool on = [] {
    char const *v = std::getenv("COQUI_GWLINE_MEMTRACE");
    return v != nullptr and *v != '\0' and std::strtol(v, nullptr, 10) != 0;
  }();
  if (not on) return;
  const double GB = 1073741824.0;
  double rss = host_rss_bytes(), hwm = host_hwm_bytes(), av = 1e300;
  if (node_comm.rank() == 0) {
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f, line))
      if (line.rfind("MemAvailable:", 0) == 0) av = 1024.0 * std::strtod(line.c_str() + 13, nullptr);
  }
  double v[4] = {rss, hwm, -rss, -av}, mx[4];
  comm.all_reduce_n(v, 4, mx, boost::mpi3::max<>{});
  double who = (rss == mx[0]) ? double(comm.rank()) : -1.0, wmax = 0.0;
  comm.all_reduce_n(&who, 1, &wmax, boost::mpi3::max<>{});
  app_log(1, "  [memtrace] {:<14s} VmRSS max {:.3f} GB (rank {}) min {:.3f} GB; VmHWM max {:.3f} GB; MemAvailable min over nodes {:.1f} GB",
          tag, mx[0] / GB, long(wmax), -mx[2] / GB, mx[1] / GB, -mx[3] / GB);
}

/// the phases of one iteration, in print order (indented names are sub-timers of the preceding phase)
static const std::vector<std::string> phase_names = {
    "time_grid",     "bases",          "phase_Pi",        "G_tilde",         "Pi_hadamard",    "Pi_ft_fwd",
    "Pi_ft_prod",    "Pi_ft_back",     "Pi_transform",
    "phase_W",       "W_redistribute", "W_dyson",         "W_fit",           "phase_Sigma",    "Sigma_G_tilde",
    "Sigma_W_time",  "Sig_ft_wR",      "Sigma_hadamard",  "Sig_ft_G",        "Sig_ft_W",       "Sig_ft_prod",
    "Sig_ft_back",   "Sigma_contract", "Sigma_allreduce", "Sigma_transform", "Sigma_mix",
    "phase_closure", "closure_upfold", "closure_kloop",   "closure_wait",    "closure_scan",   "closure_gather",
    "closure_mu",    "closure_compress", "phase_F",
    "checkpoint",    "iteration"};

std::vector<double> phase_snapshot(utils::TimerManager &T) {
  std::vector<double> v;
  v.reserve(phase_names.size());
  for (auto const &nm : phase_names) v.push_back(T.elapsed(nm));
  return v;
}

/**
 * Per-iteration report (level 1): the time of every phase in this iteration, min / avg / max over the ranks (the
 * max/avg ratio is the load imbalance), and the high-water memory (host VmHWM max / min over ranks; device: the
 * largest drop of the free device memory since the start of the run, max over ranks) next to the memory model.
 */
void report_iteration(boost::mpi3::communicator &comm, long iter, std::vector<double> const &t0, std::vector<double> const &t1,
                      double rss0, double model_host, double model_dev, bool device) {
  const long n = long(phase_names.size()), np = comm.size();
  std::vector<double> d(n), mn(n), mx(n), sm(n);
  for (long i = 0; i < n; ++i) d[i] = t1[i] - t0[i];
  comm.all_reduce_n(d.data(), n, mn.data(), boost::mpi3::min<>{});
  comm.all_reduce_n(d.data(), n, mx.data(), boost::mpi3::max<>{});
  comm.all_reduce_n(d.data(), n, sm.data(), std::plus<>{});
  app_log(1, "  phase timers, iteration {} (s over {} ranks; min / avg / max, max/avg):", iter, np);
  for (long i = 0; i < n; ++i) {
    if (mx[i] <= 0.0) continue;
    const bool phase = phase_names[i].rfind("phase_", 0) == 0 or phase_names[i] == "time_grid" or
                       phase_names[i] == "checkpoint" or phase_names[i] == "iteration" or phase_names[i] == "bases";
    const double avg = sm[i] / double(np);
    app_log(1, "    {}{:<18s} {:9.3f} {:9.3f} {:9.3f}  {:5.2f}", phase ? "" : "  ", phase_names[i], mn[i], avg, mx[i],
            avg > 0.0 ? mx[i] / avg : 1.0);
  }
  const double GB = 1024.0 * 1024.0 * 1024.0;
  double hw = host_hwm_bytes(), hmax = 0.0, hmin = 0.0;
  comm.all_reduce_n(&hw, 1, &hmax, boost::mpi3::max<>{});
  comm.all_reduce_n(&hw, 1, &hmin, boost::mpi3::min<>{});
  app_log(1, "  memory, iteration {}: host VmHWM per rank max {:.3f} GB, min {:.3f} GB (RSS before the GW_line arrays {:.3f} GB "
             "max; model: that + {:.3f} GB = {:.3f} GB)",
          iter, hmax / GB, hmin / GB, rss0 / GB, model_host / GB, (rss0 + model_host) / GB);
  if (device) {
    double dh = device_high_water_bytes(), dmax = 0.0;
    comm.all_reduce_n(&dh, 1, &dmax, boost::mpi3::max<>{});
    app_log(1, "  memory, iteration {}: device high-water per rank max {:.3f} GB (model {:.3f} GB)", iter, dmax / GB, model_dev / GB);
  }
}

void print_line(gw_line_iter_t const &r, double tPi, double tW, double tS, double tC, double tF) {
  app_log(1,
          "iter {:3d}: dSigma {:.2e}  mu {:.6f} (dmu {:+.4f} eV)  QP gap {:.4f} eV  nelec {:.6f} (Lehmann {:.6f}, N(mu) {:.6f}, "
          "dropped {:.1e})  npoles {}-{}  held-out {:.1e}  t-nodes ({}) Pi {}+{} Sigma {}+{}  [Pi {:.1f}s W {:.1f}s Sigma "
          "{:.1f}s closure {:.1f}s F {:.1f}s total {:.1f}s]",
          r.iter, r.dSigma, r.mu, r.dmu * HA_EV, r.gap * HA_EV, r.nelec, r.nelec_lehmann, r.N_mu, r.dropped, r.npoles_min,
          r.npoles_max, r.heldout_max, r.time_grid, r.nt_pi_p, r.nt_pi_h, r.nt_sig_p, r.nt_sig_h, tPi, tW, tS, tC, tF, r.time);
  if (r.g_repr == "lehmann")
    app_log(1, "          G lehmann: {}-{} poles per k and sector, min|e| {:.3e} Ha; pruned: weight rule {} ({:.1e}), near-mu rule {} "
               "({:.1e})",
            r.ng_min, r.ng_max, r.g_emin, r.pruned_w, r.pruned_w_weight, r.pruned_near, r.pruned_near_weight);
  else
    app_log(2, "          G compressed: {}-{} poles per k and sector, min|e| {:.3e} Ha", r.ng_min, r.ng_max, r.g_emin);
}

} // namespace

// ------------------------------------------------------------------------------------------------------------------
// driver
// ------------------------------------------------------------------------------------------------------------------
template <MEMORY_SPACE MEM> gw_line_result_t gw_line_scf(methods::thc_reader_t &thc, mf::MF &mf, ptree const &pt) {
  using arr4_t = memory::array<MEM, ComplexType, 4>;
  auto prm     = gw_line_params_t::from_ptree(pt);
  auto &mpi    = *thc.mpi();
  auto &comm   = mpi.comm;
  const std::string chk = prm.output + ".gw_line.h5";

  // perf 7.3: a symmetric mean field runs on its IBZ (ibz.hpp); the full-BZ path needs a nosym mean field (the THC reader
  // of a symmetric mean field holds Z(q) for the IBZ q only)
  const bool sym_mf = (mf.nkpts() != mf.nkpts_ibz() or mf.nqpts() != mf.nqpts_ibz());
  utils::check(not sym_mf or prm.ibz, "gw_line: symmetric mean field (nkpts {} nkpts_ibz {}) with ibz = false: the full-BZ path needs a "
                                      "nosym mean field", mf.nkpts(), mf.nkpts_ibz());
  const ibz_t ibz(mf, thc.nbnd(), sym_mf);
  utils::check(mf.nspin() == 1 and mf.npol() == 1 and thc.ns() == 1 and thc.npol() == 1,
               "gw_line: spin-restricted collinear only (nspin {}, npol {})", mf.nspin(), mf.npol());
  utils::check(thc.nbnd() == mf.nbnd(), "gw_line: THC nbnd {} != MF nbnd {}", thc.nbnd(), mf.nbnd());
  // nk: the k of the poles, Sigma, closure, mixing and checkpoints (the IBZ k on a symmetric mesh); nkF: all k (X, G~)
  const long nk = ibz.nkI, nkF = mf.nkpts(), nq = mf.nqpts(), nb = thc.nbnd(), Np = thc.Np();
  const double nelec = double(mf.nelec());
  const long nocc    = long(std::llround(nelec / 2.0));
  const k_dist_t kd(nk, comm);                         // owner of Sigma(k) (S7e: k-distributed Sigma)
  k_dist_t const *kdp = prm.sigma_kdist ? &kd : nullptr;
  const bool sig_all  = (prm.checkpoint_sigma == "all");
  utils::check(std::abs(nelec - 2.0 * nocc) < 1e-8 and nocc > 0 and nocc < nb,
               "gw_line: need an even electron count with 0 < nelec/2 < nbnd (nelec {}, nbnd {})", nelec, nb);

  utils::TimerManager Timer;
  for (auto nm : {"total", "H0", "bases", "time_grid", "phase_Pi", "phase_W", "phase_Sigma", "phase_closure", "phase_F",
                  "checkpoint", "spectra", "iteration", "Sigma_mix"})
    Timer.add(nm);
  Timer.start("total");
  // S7e: memory baseline (MF, THC, node-shared arrays) before any GW_line array; device high-water from here on
  double rss0 = host_rss_bytes();
  rss0        = comm.all_reduce_value(rss0, boost::mpi3::max<>{});
  if constexpr (MEM != HOST_MEMORY) device_mem_reset();

  app_log(1, "\n╔══════════════════════════════════════════════════════════╗");
  app_log(1, "║  CoQuí: self-consistent GW on the tilted frequency line  ║");
  app_log(1, "╚══════════════════════════════════════════════════════════╝");
  app_log(1, "  nkpts = {}, nqpts = {}, nbnd = {}, Np = {}, nelec = {}, ranks = {}, memory space = {}", nkF, nq, nb, Np, nelec,
          comm.size(), MEM == HOST_MEMORY ? "host" : "device");
  prm.log();
  ibz.log(1);

  // one-body Hamiltonian, KS spectrum, initial centre
  Timer.start("H0");
  auto H0 = one_body_h0(mf);
  if (prm.debug_noise_h0 > 0.0) {   // diagnostic (noise-floor meter): the same Hermitian noise on every rank
    const double h0max = nda::max_element(nda::abs(H0));
    std::mt19937_64 gen(0x5eedull + 7919ull * std::uint64_t(prm.debug_noise_seed));
    std::normal_distribution<double> N01;
    for (long ik = 0; ik < H0.extent(0); ++ik)
      for (long i = 0; i < H0.extent(1); ++i)
        for (long j = 0; j <= i; ++j) {
          const ComplexType x = prm.debug_noise_h0 * h0max * ComplexType(N01(gen), i == j ? 0.0 : N01(gen));
          H0(ik, i, j) += x;
          if (j != i) H0(ik, j, i) += std::conj(x);
        }
  }
  Timer.stop("H0");
  nda::array<double, 2> eig(nk, nb);
  double homo = -1e300, lumo = 1e300;
  for (long ik = 0; ik < nk; ++ik)
    for (long n = 0; n < nb; ++n) {
      eig(ik, n) = mf.eigval()(0, ik, n);
      if (n < nocc) homo = std::max(homo, eig(ik, n));
      else lumo = std::min(lumo, eig(ik, n));
    }
  utils::check(lumo > homo, "gw_line: the KS spectrum has no gap (homo {} lumo {})", homo, lumo);
  const double mu0 = 0.5 * (homo + lumo);

  // fixed grids and bases
  const double theta = prm.theta_deg * std::numbers::pi / 180.0, theta_t = prm.theta_t_frac * theta;   // S8b: theta_t_frac
  auto zeta          = numerics::line_dlr::dense_nodes(theta, prm.node_tmin, prm.node_tmax, prm.nodes_per_ray);
  long nz            = zeta.size();
  const auto zeta_prod = zeta;   // perf 7.2: the production nodes (checkpoint input group; zeta may be a coarse level's)
  Timer.start("bases");
  line_basis_t gp(theta, prm.lam, prm.eps, prm.lam, prm.g_gap, -1.0, prm.node_tmax);
  line_basis_t gh(theta, prm.lam, prm.eps, prm.g_gap, prm.lam, -1.0, prm.node_tmax);
  // diagnostics / parity: poles (and bosonic nodes) of the bases from a file (every rank reads it)
  auto read_w = [&](std::string const &key) {
    nda::array<double, 1> w;
    h5::file f(prm.bases_file, 'r');
    h5::group g(f);
    nda::h5_read(g, key, w);
    return w;
  };
  auto set_line_basis = [&](line_basis_t &b, std::string const &key) {
    if (prm.bases_file.empty()) return;
    b.w    = read_w(key);
    b.rank = b.w.size();
  };
  set_line_basis(gp, "g_particle_w");
  set_line_basis(gh, "g_hole_w");
  Timer.stop("bases");
  closure_params_t cprm{prm.wp, prm.K, std::max(prm.tol_gram, prm.tol_gram_eps * prm.eps), prm.nphi};
  cprm.tol_svd    = prm.tol_svd;
  if (ibz.active)   // perf 7.3: star weights of the IBZ k in the electron count (empty = uniform, the nosym path unchanged)
    for (long k = 0; k < ibz.nkI; ++k) cprm.k_weight.push_back(ibz.kw(k));
  cprm.gram_cut   = prm.closure_cut;
  cprm.svd_cut    = prm.closure_svd_cut;
  cprm.cut_window = prm.closure_cut_window;
  cprm.phase_keep = prm.phase_keep;
  cprm.svd_driver = prm.closure_svd;
  cprm.ueig       = prm.closure_ueig;
  // S7g: BLAS threads of the host closure. Device runs have one rank per GPU and idle cores; host runs fill the cores
  cprm.blas_threads = prm.closure_threads >= 0 ? prm.closure_threads
                                               : (MEM != HOST_MEMORY ? cores_per_rank(long(mpi.node_comm.size())) : 0);
  {
    const bool dev = (prm.closure_device == "on") or (prm.closure_device == "auto" and MEM != HOST_MEMORY);
    cprm.hooks     = dev ? device_lapack_hooks(prm.closure_dev_svd == "gesvdp" ? 1 : 0) : nullptr;
    utils::check(not(prm.closure_device == "on" and cprm.hooks == nullptr),
                 "gw_line: closure_device = \"on\" needs a CUDA build");
  }
  cprm.k_workers = prm.closure_k_workers > 0 ? prm.closure_k_workers
                                             : (MEM != HOST_MEMORY ? std::max(1L, cprm.blas_threads / 2) : 1L);
  app_log(1, "  closure: BLAS threads per rank {} ({}), k workers {} (BLAS threads each {}), dense eigensolvers/SVD on the {}",
          cprm.blas_threads,
          cprm.blas_threads > 0 ? (prm.closure_threads >= 0 ? "closure_threads" : "auto: cores of the rank") : "untouched",
          cprm.k_workers, cprm.k_workers > 1 ? std::max(1L, cprm.blas_threads / cprm.k_workers) : cprm.blas_threads,
          cprm.hooks ? "GPU (cuSOLVER, host fallback)" : "host");
  g_repr_params_t grepr{prm.g_repr, prm.g_emax, prm.g_wtol, prm.g_emin_frac, prm.g_wsmall};

  // S8b finite temperature (thermal.hpp): an iteration is thermal iff a pole of its G lies within E_T of mu; then the
  // kernels run on the thermal sector lists, the W step on the data set D with the D-selected (Bose-augmented) basis, the
  // closure fits the total Sigma on the two-sided gapless basis and mu follows mu_rule. Otherwise the T = 0 code (bitwise).
  thermal_params_t tpar;
  tpar.beta = prm.beta; tpar.thermal_tol = prm.thermal_tol; tpar.c_zeta = prm.thermal_floor; tpar.c_f = prm.thermal_floor_f;
  tpar.theta = theta; tpar.theta_t = theta_t; tpar.wp_floor = prm.wp_floor; tpar.mu_rule = prm.mu_rule;
  tpar.mu_dn_max = prm.mu_dn_max; tpar.mu_th_factor = prm.mu_th_factor; tpar.band_heights = prm.band_heights;
  tpar.band_x = prm.band_x; tpar.band_top = prm.band_top; tpar.mats_factor = prm.mats_factor; tpar.eps_b = prm.bos_eps_T;
  tpar.lam_b = prm.lam_b; tpar.bos_eps_T = prm.bos_line_eps; tpar.cut_odd = prm.cut_odd; tpar.cut_even = prm.cut_even;
  tpar.tau_grid = prm.tau_grid; tpar.tau_eps = prm.tau_eps;
  const double E_T = tpar.on() ? tpar.E_T() : 0.0;
  // S8b: Lehmann (e, v) per k of factorized poles (both sectors, ascending) and the density matrix of a state
  auto lehmann_lists = [&](pole_data_t const &p) {
    std::vector<nda::array<double, 1>> e(p.nk);
    std::vector<nda::array<ComplexType, 2>> v(p.nk);
    for (long k = 0; k < p.nk; ++k) {
      const long M = p.hole[k].size() + p.part[k].size();
      e[k] = nda::array<double, 1>(M);
      v[k] = nda::array<ComplexType, 2>(p.nb, M);
      long j = 0;
      for (auto const *ps : {&p.hole[k], &p.part[k]})
        for (long m = 0; m < ps->size(); ++m, ++j) {
          e[k](j) = ps->e(m);
          v[k](nda::range::all, j) = ps->v(nda::range::all, m);
        }
    }
    return std::make_pair(std::move(e), std::move(v));
  };
  auto density_of = [&](pole_data_t const &p) {
    if (tpar.on() and window_active(p, E_T)) return density_matrix(thermal_lists(p, prm.beta, E_T));
    return density_matrix(p);
  };
  // state: restart or KS start
  state_t st;
  gw_line_result_t res;
  // the decision of the root (S8b.2 fix: evaluated per rank, a slow rank saw the checkpoint the root had just created and
  // went into read_state while the others wrote it: deadlock with restart = true and no checkpoint at 64 ranks)
  long restart_l = (comm.root() and prm.restart and std::filesystem::exists(chk)) ? 1 : 0;
  comm.broadcast_n(&restart_l, 1, 0);
  const bool restart = restart_l != 0;
  if (prm.restart and not restart) app_log(1, "  restart requested but {} does not exist: starting from the KS poles", chk);
  aux_grid_t grid(mpi, Np);
  propagator_t<MEM> prop(thc, grid);
  prop.set_ibz(&ibz);   // perf 7.3: IBZ poles unfolded to every k (no-op without symmetry)
  if (restart) {
    st = read_state(comm, chk, nk, nb, zeta_prod, res.history, kdp);
    app_log(1, "  resumed from {}: {} iterations done, mu {:.6f} Ha", chk, st.iter, st.mu);
  } else {
    st.iter     = 0;
    st.mu       = mu0;
    st.mu_sigma = mu0;
    st.e_homo   = homo - mu0;
    st.e_lumo   = lumo - mu0;
    st.poles    = pole_data_t::from_ks(eig, mu0);
    if (tpar.on()) {   // S8b: mu_0 of the KS poles by mu_rule at the run's beta (an empty window keeps the midpoint)
      auto [le, lv] = lehmann_lists(st.poles);
      auto tp0      = tpar;
      if (prm.scf_density == "matsubara") tp0.mu_rule = "number";   // S8b.3: N(mu_0) = N_el exactly (python LineSCGW)
      auto r        = mu_rule_apply(le, lv, nelec, cprm.k_weight, tp0);
      if (r.rule != "gap(T=0)") {
        st.mu       = mu0 + r.mu;
        st.mu_sigma = st.mu;
        st.e_homo   = homo - st.mu;
        st.e_lumo   = lumo - st.mu;
        st.poles    = pole_data_t::from_ks(eig, st.mu);
      }
      app_log(1, "  finite T start: KS mu_0 = {:.12f} Ha (rule {}; gap midpoint {:.12f}, dN {:.2e}, n_th {:.2e}, N(mu_0) {:.12f}), "
                 "window poles {}",
              st.mu, r.rule, mu0, r.dN, r.n_th, r.N, [&] { long c = 0; for (long x : window_counts(st.poles, E_T)) c += x; return c; }());
    }
    write_system_h5(comm, chk, mf, Np, H0, mu0, true);
    if (comm.root()) {
      h5::file f(chk, 'a');
      h5::group g(f);
      write_input(g, prm, zeta_prod);
    }
    comm.barrier();
  }

  // bosonic basis (rebuilt when its gap changes), Sigma bases (rebuilt when their gaps change)
  std::optional<bosonic_basis_t> bos;
  std::optional<line_basis_t> bp, bh;
  double lv_eps = prm.eps, lv_time_eps = prm.time_eps;   // perf 7.2: eps / time_eps of the current level
  auto update_bases = [&]() {
    Timer.start("bases");
    const double bgap = prm.bos_gap >= 0.0 ? prm.bos_gap : 0.5 * (st.e_lumo - st.e_homo);
    if (not bos or bos->gap != bgap) {
      bos.emplace(theta, prm.lam_b, lv_eps, bgap);
      if (not prm.bases_file.empty()) {
        bos->nu   = read_w("bos_nu");
        bos->rank = bos->nu.size();
        h5::file f(prm.bases_file, 'r');
        h5::group g(f);
        nda::h5_read(g, "bos_zeta_nodes", bos->zeta_nodes);
      }
    }
    double gpp = prm.sigma_gap, ghh = prm.sigma_gap;
    if (prm.sigma_gap < 0.0) {
      gpp = 0.8 * (st.e_lumo + bos->gap);
      ghh = 0.8 * (std::abs(st.e_homo) + bos->gap);
    }
    if (not bp or bp->gap[1] != gpp) {
      bp.emplace(theta, prm.lam, lv_eps, prm.lam, gpp, -1.0, prm.node_tmax);
      set_line_basis(*bp, "sigma_particle_w");
    }
    if (not bh or bh->gap[0] != ghh) {
      bh.emplace(theta, prm.lam, lv_eps, ghh, prm.lam, -1.0, prm.node_tmax);
      set_line_basis(*bh, "sigma_hole_w");
    }
    Timer.stop("bases");
  };
  update_bases();
  app_log(1, "  bases: bosonic rank {} ({} nodes, gap {:.4f}), Sigma {}+{}, G {}+{}, fermionic nodes {}", bos->rank,
          bos->zeta_nodes.size(), bos->gap, bp->rank, bh->rank, gp.rank, gh.rank, nz);

  std::optional<bosonic_basis_t> bosT;   // the D-selected bosonic basis with the Bose rows (fixed for the run)
  // S8b.3 hybrid: Bose rows (w(-q)^T) for every pole with beta nu <= 700 (weight 0 on the rays beyond E_T), the tau nodes of the
  // Sigma leg (fixed range g_emax + lam_b)
  const bool hybrid   = (prm.scf_density == "matsubara");
  const double E_rows = hybrid ? 700.0 / prm.beta : -1.0;
  std::optional<hybrid_nodes_t> hyb;
  std::optional<bosonic_basis_t> bosX;   // bosT with the exact Bose weights (tau leg)
  std::optional<line_basis_t> bt;        // the two-sided gapless Sigma basis of the thermal closure
  bos_data_t Dset;
  closure_thermal_t cth;
  bosonic_basis_t const *bcur = &*bos;   // the bosonic basis of the current iteration
  auto ensure_thermal = [&]() {
    if (bosT) return;
    Timer.start("bases");
    nda::array<double, 1> nub, sbw;
    if (not prm.thermal_bases_file.empty()) {   // parity: the python reference's D, nu_b, Sigma basis
      h5::file f(prm.thermal_bases_file, 'r');
      h5::group g(f);
      nda::array<double, 1> zr, zi;
      nda::array<signed char, 1> kd;
      nda::h5_read(g, "D_zeta_re", zr);
      nda::h5_read(g, "D_zeta_im", zi);
      nda::h5_read(g, "D_kind", kd);
      nda::h5_read(g, "nu_b", nub);
      nda::h5_read(g, "sigma_basis_w", sbw);
      Dset.z = nda::array<ComplexType, 1>(zr.size());
      Dset.kind.assign(zr.size(), 0);
      for (long i = 0; i < zr.size(); ++i) {
        Dset.z(i)    = ComplexType(zr(i), zi(i));
        Dset.kind[i] = kd(i);
      }
      Dset.n_line = Dset.count(0);
      Dset.n_band = Dset.count(1);
      Dset.n_mats = Dset.count(2);
      bosT.emplace(bosonic_basis_t::with_poles(theta, Dset.z, prm.lam_b, nub, prm.bos_eps_T, prm.cut_odd, prm.cut_even)
                       .with_bose(prm.beta, E_T, E_rows));
    } else {
      bosonic_basis_t bl(theta, prm.lam_b, prm.bos_line_eps, 0.0);
      Dset = make_bos_data(bl.zeta_nodes, tpar);
      bosT.emplace(bosonic_basis_t::from_data(theta, Dset.z, prm.lam_b, prm.bos_eps_T, 800, -1.0, prm.cut_odd, prm.cut_even)
                       .with_bose(prm.beta, E_T, E_rows));
    }
    bt.emplace(theta, prm.lam, lv_eps, 0.0, 0.0, -1.0, prm.node_tmax);
    if (sbw.size() > 0) {
      bt->w    = sbw;
      bt->rank = sbw.size();
    }
    cth.bt   = &*bt;
    cth.tp   = tpar;
    if (hybrid) {
      bosX.emplace(bosT->with_exact_bose());
      hyb.emplace(make_hybrid_nodes(prm.beta, prm.g_emax + prm.lam_b, prm.hyb_tau_eps, comm));
      app_log(1, "  hybrid (S8b.3): Sigma tau ID {} nodes per leg on [0, beta / 2] (E in [{:.4f}, {:.3f}] Ha, rank {}, candidates {}), "
                 "{} Bose rows (beta nu <= 700), dense Matsubara set N = {} (w_max {} Ha)",
              hyb->ntau, -hyb->Eneg, hyb->Emax, hyb->idp.rank, hyb->idp.n_cand, bosT->n_bose, matsubara_set(prm.beta, prm.hyb_wmax).size(),
              prm.hyb_wmax);
    }
    cth.mask.assign(zeta.size(), 0);
    long nkeep = 0;
    for (long i = 0; i < zeta.size(); ++i) {
      cth.mask[i] = (tpar.rho() * tpar.beta * std::abs(zeta(i)) >= tpar.c_f) ? 1 : 0;
      nkeep += cth.mask[i];
    }
    Timer.stop("bases");
    app_log(1, "  finite T: E_T = {:.5f} Ha, zeta_T = {:.5f} Ha (rho {:.3f}), S_T = {:.2f}; data set D: {} points (line {}, band {}, "
               "Matsubara {}, nu_0 x{}{}), D-selected bosonic basis rank {} + {} Bose rows (nu_j <= E_T), two-sided Sigma basis {} "
               "poles, fermionic nodes kept {} / {} (c_f = {})",
            E_T, tpar.zeta_T(), tpar.rho(), tpar.S_T(), Dset.z.size(), Dset.n_line, Dset.n_band, Dset.n_mats, Dset.count(3),
            Dset.mirror ? ", mirror layout" : "", bosT->rank_fit(), bosT->n_bose, bt->rank, nkeep, zeta.size(), tpar.c_f);
  };

  const long nqR = ibz.nrows();   // perf 7.3: Pi / W rows (all q without symmetry)
  dyson_layout_t lay(comm.size(), comm.rank(), nqR, bos->zeta_nodes.size(), Np);
  // q groups of the Pi -> W stage (S7e; perf 7.4b: automatic from the 6.7 model and the host / device budgets, q_plan.hpp)
  const bool dev_fused   = (MEM != HOST_MEMORY) and detail::fused_hadamard();
  const long tc_host0    = (prm.t_chunk > 0 ? prm.t_chunk : detail::env_long("COQUI_GWLINE_HOST_TCHUNK", detail::host_t_chunk_default));
  const long tc_model    = (MEM == HOST_MEMORY ? tc_host0 : (prm.t_chunk > 0 ? prm.t_chunk : 64));
  // perf 7.4b relief levels of the q plan (q_plan.hpp), tried in order when no q grouping fits the budget:
  //   full BZ: 0 none, 1 Sigma's real-space residues in place of w, 2 + host t_chunk / 2, 3 + host t_chunk / 4 (>= 8);
  //   IBZ    : 0 none, 1 host t_chunk / 2, 2 host t_chunk / 4. The device chooses its chunk itself (no t_chunk levels);
  //   an explicit t_chunk disables the t_chunk levels.
  const bool ip_ok      = not ibz.active and detail::env_long("COQUI_GWLINE_WR_INPLACE", -1) != 0;
  const long ntc_levels = (MEM == HOST_MEMORY and prm.t_chunk <= 0) ? 2 : 0;
  const long nlevels    = 1 + (ip_ok ? 1 : 0) + ntc_levels;
  auto level_inplace    = [&](long L) { return ip_ok and (L >= 1 or detail::env_long("COQUI_GWLINE_WR_INPLACE", -1) == 1); };
  auto level_tc         = [&](long L) {   // the host kernels' t_chunk at level L (0: the default)
    const long halvings = std::max(0L, L - (ip_ok ? 1 : 0));
    if (MEM != HOST_MEMORY or prm.t_chunk > 0 or halvings == 0) return prm.t_chunk;
    return std::max(8L, tc_host0 >> halvings);
  };
  const double sig_bytes = 16.0 * double(prm.sigma_kdist ? kd.nloc(0) : nk) * double(nz) * double(nb * nb);
  // host model (S7e): the kernel arrays (= the 6.7 model on the host path) + the Sigma arrays of the driver (Sig_p, Sig_h,
  // Sp_new, Sh_new: 4 N_k N_zeta_f nb^2) + the per-chunk Sigma reduce buffers (2 N_k t_chunk nb^2) + the full Z(q) of the
  // Dyson slab of the grouping (max over the ranks: the plan is collective)
  // the kernels' stage-wise peak of the 6.7 model (aux_grid_t::model) + what it does not contain (perf 7.4b, measured on
  // si444 IBZ: VmHWM 2.77 GB vs model 1.48 GB): the A^ cache of Pi's real-space transform (propagators.hpp, N_k x N_t
  // blocks when enabled; N_t estimated by the bosonic node count) and, on the IBZ, the class-sum arrays of self_energy_ibz
  // (G~, G^ of all k, W(t) of the rows R, the back-transformed class sums: (2 N_k + |R| + n_cls nk_ibz) t_chunk blocks)
  // device: the kernels size their chunk from the free memory at the call (40%, >= 8): the plan needs only the minimum chunk
  auto tc_plan_of = [&](long L) -> long {
    if (MEM != HOST_MEMORY) return prm.t_chunk > 0 ? prm.t_chunk : 8L;
    const long t = level_tc(L);
    return t > 0 ? t : tc_model;
  };
  // a grouping is feasible when the Dyson layout of every group exists (np <= q pools x zeta pools; dyson_layout_t)
  auto feasible = [&](long g, long nzb_) {
    q_groups_t qgg(ibz.rows, g, ibz.qminus);
    for (long G = 0; G < qgg.n; ++G)
      if (not dyson_layout_t::valid(comm.size(), qgg.size(G), nzb_)) return false;
    return true;
  };
  auto kernels_model = [&](long g, long nzb_, long L) {
    q_groups_t qgg(ibz.rows, g, ibz.qminus);
    const long tc_plan = tc_plan_of(L);
    const auto mm    = grid.model(nkF, nqR, nzb_, bcur->rank, tc_plan, nb, qgg.max_size(), dev_fused, level_inplace(L));
    const double b16 = 16.0 * double(grid.max_block_size());
    double stage     = mm.peak_stage;
    if (ibz.active) {
      const double sg = (2.0 * nkF + nqR + double(ibz.nclasses()) * nk + (MEM == HOST_MEMORY ? 0.0 : double(nkF + nqR))) *
                        double(tc_plan) * b16;
      stage           = std::max({mm.res + mm.pi_t, mm.res + mm.w_t, mm.res - mm.pig + sg});
    }
    const double ahat = double(nkF) * double(nzb_) * b16;
    if (detail::gt_cache_enabled<MEM>(ahat)) stage += ahat;
    return stage;
  };
  // host model (S7e): the kernel arrays (host path) + the Sigma arrays of the driver (Sig_p, Sig_h, Sp_new, Sh_new:
  // 4 N_k N_zeta_f nb^2) + the per-chunk Sigma reduce buffers (2 N_k t_chunk nb^2) + the full Z(q) of the Dyson slab of
  // the grouping (max over the ranks: the plan is collective)
  // the Sigma reduce of a chunk: the send buffer + MPI_Reduce_scatter's temporaries (~2 more of the same size; the 8x8x8 node
  // ran out of memory entering Sigma with 3.2 GB per rank free against the 2.8 GB of the arrays alone): 3 N_k t_chunk nb^2
  auto qmodel_host = [&](long g, long nzb_, long L) {
    if (not feasible(g, nzb_)) return 1e300;   // same on every rank (no collective skipped unevenly)
    q_groups_t qgg(ibz.rows, g, ibz.qminus);
    const double zf = 16.0 * double(qgg.dyson_q_list(comm.size(), comm.rank(), nzb_, Np).size()) * double(Np) * double(Np);
    const long tcr  = (MEM == HOST_MEMORY) ? tc_plan_of(L) : detail::host_t_chunk_default;
    double m = (MEM == HOST_MEMORY ? kernels_model(g, nzb_, L) : 0.0) + 4.0 * sig_bytes +
               3.0 * 16.0 * double(nkF) * double(tcr) * double(nb * nb) + zf;
    return comm.all_reduce_value(m, boost::mpi3::max<>{});
  };
  auto qmodel_dev = [&](long g, long nzb_, long L) {
    if (not feasible(g, nzb_)) return 1e300;
    double m = kernels_model(g, nzb_, L);
    return comm.all_reduce_value(m, boost::mpi3::max<>{});
  };
  auto make_qplan = [&](long nzb_) {
    auto qp = choose_q_plan<MEM>(comm, mpi.node_comm, grid, ibz.rows, ibz.qminus, nkF, nzb_, bcur->rank, nb,
                                 q_budget_params_t{prm.mem_budget_gb, prm.dev_mem_budget_gb, prm.mem_frac, prm.q_group_size},
                                 [&](long g, long L) { return qmodel_host(g, nzb_, L); },
                                 [&](long g, long L) { return qmodel_dev(g, nzb_, L); }, not ibz.active, nlevels,
                                 // the root alone: the checkpoint's gather of Sigma (write_state: two sectors, owner + k order)
                                 (prm.sigma_kdist and comm.size() > 1) ? 3.0 * 16.0 * double(nk) * double(nz) * double(nb * nb) : 0.0);
    qp.wR_inplace = level_inplace(qp.level);
    qp.t_chunk    = level_tc(qp.level);
    if (qp.level > 0 or qp.wR_inplace)
      app_log(1, "  q plan relief: Sigma's real-space residues {}, kernel t_chunk {}", qp.wR_inplace ? "in place of w" : "beside w",
              qp.t_chunk > 0 ? std::to_string(qp.t_chunk) : std::string("default"));
    return qp;
  };
  q_plan_t qplan = make_qplan(long(bos->zeta_nodes.size()));
  auto record_qplan = [&]() {
    res.q_group_size     = qplan.g;
    res.q_ngroups        = qplan.ngroups;
    res.q_wR_inplace     = qplan.wR_inplace;
    res.q_budget_host    = qplan.budget_host;
    res.q_model_host     = qplan.model_host;
    res.q_model_host_all = qplan.model_host_all;
    res.q_model_host_min = qplan.model_host_min;
  };
  record_qplan();
  utils::check(not(ibz.active and qplan.w_host), "gw_line: host-resident residues (q_plan_t::w_host) are not implemented with the IBZ");
  q_groups_t qg(ibz.rows, qplan.g, ibz.qminus);   // pair-closed groups of the rows R (W pairing q <-> -q, screened.hpp)
  long zb_nzb = long(bos->zeta_nodes.size());     // perf 7.2: bosonic node count the Dyson slab (Zb) was built for
  std::vector<long> zb_list = qg.dyson_q_list(comm.size(), comm.rank(), zb_nzb, Np);
  coulomb_blocks_t<MEM> Zb(thc, grid, zb_list, Timer);
  // F = V_H + Sigma_x: the IBZ class sums (static_ibz.hpp) on a symmetric mesh
  auto static_F = [&](nda::array<ComplexType, 3> const &D, nda::array<ComplexType, 3> &F) {
    if (ibz.active) hartree_exchange_ibz<MEM>(prop, Zb, D, mf, ibz, grid, comm, F, Timer);
    else hartree_exchange<MEM>(prop, Zb, D, mf, grid, mpi, F, Timer);
  };
  if (qg.n > 1) app_log(1, "  q groups of the Pi -> W stage: {} groups of <= {} q (Pi group of all q does not fit)", qg.n, qg.max_size());
  if (qplan.w_host)
    app_log(1, "  residues w of all q on the HOST ({:.3f} GB per rank); Sigma streams them in groups of {} q", 16.0 * double(nq) *
                   bos->rank * grid.max_block_size() / 1073741824.0, qplan.gs_sigma);
  const double model_dev = grid.log(nkF, nqR, bos->zeta_nodes.size(), bos->rank, tc_model, nb, qg.max_size(), dev_fused);
  lay.log();
  const double model_host = (MEM == HOST_MEMORY ? model_dev : 0.0) + 4.0 * sig_bytes +
                            2.0 * 16.0 * double(nkF) * double(detail::host_t_chunk_default) * double(nb * nb) +
                            16.0 * double(Zb.Z_full.extent(0)) * double(Np) * double(Np);
  app_log(2, "    driver host arrays: Sigma 4 x {:.4f} GB ({}), full Z(q) {} x {:.4f} GB; host model {:.4f} GB above the "
             "baseline RSS {:.4f} GB",
          sig_bytes / 1073741824.0, prm.sigma_kdist ? "k-distributed, <= ceil(N_k / np) rows" : "replicated on every rank",
          Zb.Z_full.extent(0), 16.0 * double(Np) * Np / 1073741824.0, model_host / 1073741824.0, rss0 / 1073741824.0);

  // S9a: the Coulomb head (head.hpp): per-q heads eps^-1_00(q) - 1 from W every iteration (checkpoint, eps_inf), the
  // q -> 0 extrapolation, and the Madelung terms of div_treatment (Sigma_c) / hf_div_treatment (exchange)
  const double madelung = mf.madelung();
  const head_basis_t hbasis(thc, mf, grid);
  const head_extrapolation_t hextra(mf, prm.head_extrapolation, ibz.active);   // perf 7.3: full mesh, unfolded heads
  bool sig_div = head_div_is_gygi(prm.div_treatment);
  if (sig_div and nq == 1) {
    app_log(1, "  gw_line: nqpts == 1 while div_treatment = {}: the Sigma_c head term is skipped (ignore_g0), as CoQui does",
            prm.div_treatment);
    sig_div = false;
  }
  const bool hf_div = (prm.hf_div_treatment == "gygi");
  nda::array<ComplexType, 3> Thead;   // T(k) = X^dagger diag(conj chi_head(Gamma)) X (Sigma_div_correction)
  if (sig_div) Thead = head_overlap_T(thc, gamma_index(mf));
  {
    double dT = 0.0;
    for (long ik = 0; ik < Thead.extent(0); ++ik)
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) dT = std::max(dT, std::abs(Thead(ik, i, j) - ComplexType(i == j ? 1.0 : 0.0)));
    app_log(1, "  head: madelung = {:.8f} Ha; Sigma_c head term {}; exchange Madelung term {}{}", madelung,
            sig_div ? "ON (" + prm.div_treatment + ")" : std::string("off"), hf_div ? "ON" : "off",
            sig_div ? ", max|T - 1| = " + std::to_string(dT) : std::string(""));
    hextra.log(1);
  }
  std::vector<long> q_all = ibz.rows;   // the residue rows (all q without symmetry)
  std::vector<long> k_rows;   // global k of the rows of Sigma on this rank
  for (long l = 0; l < (prm.sigma_kdist ? kd.nloc() : nk); ++l) k_rows.push_back(prm.sigma_kdist ? kd.global(l, kd.rank) : l);

  // perf 7.2 warm starts: KS vectors with QP energies (same density matrix: F[D_KS] unchanged), mu = QP mid-gap
  auto apply_qp_start = [&](nda::array<double, 2> const &Eabs, std::string const &what) {
    double h = -1e300, l = 1e300, sh = 0.0;
    for (long ik = 0; ik < nk; ++ik)
      for (long n = 0; n < nb; ++n) {
        if (n < nocc) h = std::max(h, Eabs(ik, n));
        else l = std::min(l, Eabs(ik, n));
        sh = std::max(sh, std::abs(Eabs(ik, n) - eig(ik, n)));
      }
    utils::check(l > h, "gw_line: start {}: the QP energies have no gap in the KS band order (homo {} lumo {})", what, h, l);
    const double muq = 0.5 * (h + l);
    st.mu       = muq;
    st.mu_sigma = muq;
    st.dmu      = 0.0;
    st.e_homo   = h - muq;
    st.e_lumo   = l - muq;
    st.poles    = pole_data_t::from_ks(Eabs, muq);
    app_log(1, "  start {}: KS vectors + QP energies, mu = {:.6f} Ha (QP mid-gap), QP gap {:.4f} eV (KS {:.4f} eV), max|E_QP - e_KS| "
               "{:.4f} eV; F = F[D_KS] (same density)",
            what, muq, (l - h) * HA_EV, (lumo - homo) * HA_EV, sh * HA_EV);
  };
  auto write_start_done = [&](long done) {   // scf_line/start_done: 0 while the qp_diag start pass is pending
    if (comm.root()) {
      utils::h5_quiesce();
      h5::file f(chk, 'a');
      h5::group g(f);
      auto sg = g.has_subgroup("scf_line") ? g.open_group("scf_line") : g.create_group("scf_line");
      h5::h5_write(sg, "start_done", done);
    }
    comm.barrier();
  };
  bool qp_pending = false;   // perf 7.2: the qp_diag start pass is still to be done
  if (not restart) {
    Timer.start("phase_F");
    // S8b: the thermal hole list's density in thermal mode; S8b.3 hybrid: the exact f(H) of the KS poles (Sigma_c = 0)
    auto D = prm.scf_density == "matsubara" ? density_fermi(st.poles, prm.beta) : density_of(st.poles);
    static_F(D, st.F);
    if (hf_div) exchange_head_correction(st.F, D, madelung);
    Timer.stop("phase_F");
    st.F_cl = st.F;
    if (prm.start == "qp_file") apply_qp_start(read_qp_energies(comm, prm.start_file, prm.start_dataset, nk, nb), "qp_file");
    qp_pending = (prm.start == "qp_diag" and prm.niter > 0);
    Timer.start("checkpoint");
    write_state(comm, chk, st, nullptr, kdp, sig_all);
    write_start_done(qp_pending ? 0 : 1);
    Timer.stop("checkpoint");
    if (prm.start != "qp_file")
      app_log(1, "  start: KS poles, mu0 = {:.6f} Ha (KS mid-gap), KS gap {:.4f} eV, F = V_H + Sigma_x[D_KS]{}", mu0,
              (lumo - homo) * HA_EV, qp_pending ? "; qp_diag start pass first" : "");
  } else if (st.iter == 0 and prm.start == "qp_diag" and prm.niter > 0) {   // resumed before the start pass completed?
    long done = 1;
    if (comm.root()) {
      h5::file f(chk, 'r');
      h5::group g(f);
      auto sg = g.open_group("scf_line");
      if (sg.has_dataset("start_done")) h5::h5_read(sg, "start_done", done);
    }
    comm.broadcast_n(&done, 1, 0);
    qp_pending = (done == 0);
  }
  if (st.F_cl.size() == 0) st.F_cl = st.F;

  // perf 7.2: Coulomb blocks of the Dyson slab follow the bosonic node count (multilevel levels, auto bos_gap)
  auto ensure_zb = [&]() {
    const long nzb = long(bcur->zeta_nodes.size());
    if (nzb == zb_nzb) return;
    qplan    = make_qplan(nzb);
    record_qplan();
    qg       = q_groups_t(ibz.rows, qplan.g, ibz.qminus);
    auto lst = qg.dyson_q_list(comm.size(), comm.rank(), nzb, Np);
    long chg = (lst != zb_list) ? 1 : 0;
    chg      = comm.all_reduce_value(chg, boost::mpi3::max<>{});
    zb_nzb   = nzb;
    if (chg) {   // collective (lockstep Z gather)
      Zb      = coulomb_blocks_t<MEM>(thc, grid, lst, Timer);
      zb_list = std::move(lst);
      app_log(1, "  Coulomb blocks rebuilt for {} bosonic nodes (Dyson slab changed)", nzb);
    }
  };
  // perf 7.2 multilevel schedule: level of iteration it (1-based): 1 = coarse for it <= coarse_niter, else 0 = production
  auto level_of  = [&](long it) -> long { return (it >= 1 and it <= prm.coarse_niter) ? 1 : 0; };
  long cur_level = 0;
  auto set_level = [&](long L) {
    if (L == cur_level) return;
    cur_level   = L;
    lv_eps      = L ? prm.coarse_eps : prm.eps;
    lv_time_eps = L ? prm.coarse_time_eps : prm.time_eps;
    zeta = numerics::line_dlr::dense_nodes(theta, prm.node_tmin, prm.node_tmax, L ? prm.coarse_nodes_per_ray : prm.nodes_per_ray);
    nz   = zeta.size();
    Timer.start("bases");
    gp = line_basis_t(theta, prm.lam, lv_eps, prm.lam, prm.g_gap, -1.0, prm.node_tmax);
    gh = line_basis_t(theta, prm.lam, lv_eps, prm.g_gap, prm.lam, -1.0, prm.node_tmax);
    Timer.stop("bases");
    bos.reset();
    bp.reset();
    bh.reset();
    cprm.K        = L ? prm.coarse_K : prm.K;
    cprm.tol_gram = std::max(prm.tol_gram, prm.tol_gram_eps * lv_eps);
    cprm.phi_prev.clear();
    update_bases();
    ensure_zb();
    app_log(1, "  level {}: eps {:.1e}, K {}, {} fermionic nodes, time_eps {:.1e}; bosonic rank {} ({} nodes), Sigma {}+{}",
            L ? "coarse" : "production", lv_eps, cprm.K, nz, lv_time_eps, bos->rank, bos->zeta_nodes.size(), bp->rank, bh->rank);
  };
  scf_mixer_t mixer(prm.mix);
  if (not res.history.empty() and res.history.back().mix == "damped") mixer.damped = true;   // restart in the damped tail
  head_out_t hout_first;   // perf 7.2: the head of iteration 1 (computed from the initial poles; optics.poles = "initial")
  bool have_hout_first = false;

  // ---------------------------------------------------------------------------------------------- the loop
  arr4_t Pi, w, Wn, wI;
  memory::array<HOST_MEMORY, ComplexType, 4> w_h;   // host-resident residues (q_plan_t::w_host)
  Timer.add("W_head");
  Timer.add("Sigma_head");
  nda::array<ComplexType, 4> Sp_new, Sh_new;
  bool converged = false;
  head_out_t hout_last;   // S9b: the head of the last iteration of this run (optics)
  bool have_hout = false;
  while (st.iter < prm.niter) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto ph0 = phase_snapshot(Timer);
    Timer.start("iteration");
    {   // perf 7.2: the level of this iteration; a Sigma of another level is dropped (no mixing across node grids)
      const long L = level_of(st.iter + 1);
      set_level(L);
      if (st.have_sigma and level_of(st.iter) != L) {
        st.have_sigma = false;
        mixer.reset();
        app_log(1, "  level change: the Sigma of iteration {} (other node grid) is dropped; this iteration takes Sigma[G] unmixed",
                st.iter);
      }
    }
    update_bases();
    // S8b: thermal iteration iff a pole lies within E_T of mu (window); its kernels see the thermal sector lists
    const bool thermal = tpar.on() and (hybrid or window_active(st.poles, E_T));   // S8b.3: the hybrid always runs thermal
    if (thermal) ensure_thermal();
    bcur = thermal ? &*bosT : &*bos;
    bosonic_basis_t const &BB = *bcur;
    pole_data_t kp_store;
    if (thermal) kp_store = thermal_lists(st.poles, prm.beta, E_T);
    pole_data_t const &kp = thermal ? kp_store : st.poles;
    long nwin = 0;
    if (tpar.on())
      for (long c : window_counts(st.poles, E_T)) nwin += c;
    ensure_zb();
    // time nodes of the ray products: GL rays (both kernels) or the four ID grids of the current poles
    Timer.start("time_grid");
    std::optional<time_nodes_t> pi_p, pi_h, sig_p, sig_h;
    tau_nodes_t taun;
    if (thermal) {   // S8b: rays guarded at S_T (GL) or the finite-interval ID; the tau leg's nodes
      double pmax = -1e300, hmin = 1e300, emax_all = 0.0;
      for (long ik = 0; ik < kp.nk; ++ik) {
        for (long m = 0; m < kp.part[ik].size(); ++m) pmax = std::max(pmax, kp.part[ik].e(m));
        for (long m = 0; m < kp.hole[ik].size(); ++m) hmin = std::min(hmin, kp.hole[ik].e(m));
        for (auto const *ps : {&st.poles.part[ik], &st.poles.hole[ik]})
          for (long m = 0; m < ps->size(); ++m) emax_all = std::max(emax_all, std::abs(ps->e(m)));
      }
      const double numax = nda::max_element(BB.nu);
      if (prm.time_grid == "gl") {
        auto ray_p = time_ray_t::guarded(theta_t, prm.beta, E_T, 1e-5, 3.0, 16, sector_t::particle);
        auto ray_h = time_ray_t::guarded(theta_t, prm.beta, E_T, 1e-5, 3.0, 16, sector_t::hole);
        app_log(2, "  rays (finite T, guarded at S_T = {:.2f}): {} + {} time nodes", tpar.S_T(), ray_p.size(), ray_h.size());
        pi_p.emplace(ray_p);
        pi_h.emplace(ray_h);
        sig_p.emplace(ray_p);
        sig_h.emplace(ray_h);
      } else {
        numerics::line_dlr::time_id_opts_t topt;
        topt.pad        = prm.time_pad;
        topt.oversample = prm.time_oversample;
        const double Emax = std::max({pmax - hmin, pmax + numax, -hmin + numax});
        auto tg = line_time_grids_t::thermal(theta_t, 2.0 * E_T, Emax, tpar.S_T(), lv_time_eps, topt, BB.zeta_nodes, zeta, comm);
        tg.log(1);
        pi_p.emplace(tg.pi_p);
        pi_h.emplace(tg.pi_h);
        sig_p.emplace(tg.sig_p);
        sig_h.emplace(tg.sig_h);
      }
      taun = make_tau_nodes(tpar, emax_all, 2.0 * emax_all, std::addressof(comm));
      app_log(2, "  tau leg: {} nodes on [0, beta / 2] ({}{})", taun.size(), taun.kind,
              taun.kind == "id" ? ", rank " + std::to_string(taun.rank) : std::string(""));
    } else if (prm.time_grid == "gl") {
      const double emin = st.poles.emin();
      auto ray_p = time_ray_t::for_spectrum(theta_t, emin, prm.ray_decades, 1e-5, 3.0, 16, sector_t::particle);
      auto ray_h = time_ray_t::for_spectrum(theta_t, emin, prm.ray_decades, 1e-5, 3.0, 16, sector_t::hole);
      if (st.iter == 0 or res.history.empty())
        app_log(2, "  rays: emin {:.3e} Ha -> {} + {} time nodes", emin, ray_p.size(), ray_h.size());
      pi_p.emplace(ray_p);
      pi_h.emplace(ray_h);
      sig_p.emplace(ray_p);
      sig_h.emplace(ray_h);
    } else {
      numerics::line_dlr::time_id_opts_t topt;
      topt.pad        = prm.time_pad;
      topt.oversample = prm.time_oversample;
      line_time_grids_t tg(st.poles, BB.nu, theta_t, lv_time_eps, topt, BB.zeta_nodes, zeta, comm, prm.time_snap);
      tg.log(1);
      pi_p.emplace(tg.pi_p);
      pi_h.emplace(tg.pi_h);
      sig_p.emplace(tg.sig_p);
      sig_h.emplace(tg.sig_h);
    }
    Timer.stop("time_grid");
    {
      // S7c: Pi's spectrum (summed pole energies) vs the bosonic range (see from_ptree)
      auto pr          = pole_ranges_t::from(st.poles);
      const double emx = pr.p_max + pr.h_max;
      if (emx > bos->lam * (1.0 + 1e-12))
        app_log(1, "  WARNING gw_line: the G poles give Pi transitions up to {:.3f} Ha > lam_b = {:.3f} Ha: the W residues are "
                   "ill-conditioned and amplify the time-grid error in Sigma (S7c); use lam_b >= {:.1f} (or lam_b <= 0: auto)",
                emx, bos->lam, emx);
    }

    auto tic = [&](char const *nm) { Timer.start(nm); return Timer.elapsed(nm); };
    auto toc = [&](char const *nm, double e0) { Timer.stop(nm); return Timer.elapsed(nm) - e0; };

    // 1. Pi at the bosonic nodes, W residues; S9a: the heads h(q, zeta_i) from W at the nodes (the Pi buffer, moved out by
    //    screened_interaction and freed right after) and the scalar head residues from w
    double tPi = 0.0, tW = 0.0, e0 = 0.0;
    const long nzb = BB.zeta_nodes.size();
    nda::array<ComplexType, 2> Hn(nq, nzb), hres_p(nq, BB.rank), hres_h(nq, BB.rank);
    Hn()     = ComplexType(0.0);
    hres_p() = ComplexType(0.0);
    hres_h() = ComplexType(0.0);
    for (long G = 0; G < qg.n; ++G) {   // one group (all q) unless the Pi group does not fit
      e0 = tic("phase_Pi");
      arr4_t Pi_tau;
      if (thermal) {   // S8b tau leg first (the ray call below refills the propagator's A^ cache): dynamic Pi(q, 0)
        nda::array<ComplexType, 1> z0(1);
        z0(0) = ComplexType(0.0);
        pi_tau_leg<MEM>(prop, st.poles, mf, ibz, grid, tpar, taun, z0, qplan.t_chunk, Pi_tau, Timer, qg.rows(G));
      }
      polarization<MEM>(prop, kp, mf, grid, BB.zeta_nodes, *pi_p, *pi_h, qplan.t_chunk, Pi, Timer, sector_t::both,
                        qg.rows(G));
      if (thermal)   // the nu_0 points of D (z = 0, outside the wedge) from the tau leg
        for (long i = 0; i < nzb; ++i)
          if (BB.zeta_nodes(i) == ComplexType(0.0))
            for (long r = 0; r < Pi.extent(0); ++r)
              Pi(r, i, nda::range::all, nda::range::all) = Pi_tau(r, 0, nda::range::all, nda::range::all);
      tPi += toc("phase_Pi", e0);
      mem_trace(comm, mpi.node_comm, "Pi group " + std::to_string(G));
      e0 = tic("phase_W");
      screened_interaction<MEM>(Pi, Zb, BB, grid, mpi, w, Timer, &Wn, qg.rows(G), qplan.w_host or ibz.active);
      mem_trace(comm, mpi.node_comm, "W group " + std::to_string(G));
      Timer.start("W_head");
      head_nodes_partial<MEM>(Wn, qg.rows(G), hbasis, Hn);
      Wn = arr4_t{};
      if (qplan.w_host) head_residues_partial<MEM>(w, qg.rows(G), hbasis, hres_p, hres_h);
      Timer.stop("W_head");
      if (qplan.w_host) {   // this group's rows -> the host-resident residues of all q
        if (w_h.extent(0) != nq or w_h.extent(1) != BB.rank)
          w_h = memory::array<HOST_MEMORY, ComplexType, 4>(nq, BB.rank, grid.nP, grid.nQ);
        auto wg = memory::to_memory_space<HOST_MEMORY>(w);
        for (long i = 0; i < qg.size(G); ++i)
          w_h(qg.rows(G)[i], nda::range::all, nda::range::all, nda::range::all) = wg(i, nda::range::all, nda::range::all, nda::range::all);
      }
      if (ibz.active) {   // perf 7.3: this group's rows -> the residues of the rows R (ibz.rows order)
        if (wI.extent(0) != nqR or wI.extent(1) != BB.rank) wI = arr4_t(nqR, BB.rank, grid.nP, grid.nQ);
        for (long i = 0; i < qg.size(G); ++i)
          wI(ibz.rpos[qg.rows(G)[i]], nda::range::all, nda::range::all, nda::range::all) =
              w(i, nda::range::all, nda::range::all, nda::range::all);
      }
      tW += toc("phase_W", e0);
    }
    if (ibz.active) std::swap(w, wI);   // w: rows R in ibz.rows order (head residues, Sigma)
    head_out_t hout;
    {
      e0 = tic("phase_W");
      Timer.start("W_head");
      if (not qplan.w_host) head_residues_partial<MEM>(w, q_all, hbasis, hres_p, hres_h);
      head_reduce(comm, {&Hn, &hres_p, &hres_h});
      if (thermal) {   // S8b: the head residues of the fitted poles only (the Bose rows are w(-q)^T copies)
        const auto rf = nda::range(BB.rank_fit());
        hres_p = nda::array<ComplexType, 2>(hres_p(nda::range::all, rf));
        hres_h = nda::array<ComplexType, 2>(hres_h(nda::range::all, rf));
      }
      hout.h_nodes       = std::move(Hn);
      hout.h_res         = std::move(hres_p);
      hout.h_res_hole    = std::move(hres_h);
      if (ibz.active) {   // perf 7.3: the IBZ heads to every q of the mesh (hextra weights the full mesh)
        hout.h_nodes    = unfold_heads(hout.h_nodes, mf);
        hout.h_res      = unfold_heads(hout.h_res, mf);
        hout.h_res_hole = unfold_heads(hout.h_res_hole, mf);
      }
      hout.h0_nodes      = hextra.apply(hout.h_nodes);
      hout.h0_res        = hextra.apply(hout.h_res);
      hout.h0_res_hole   = hextra.apply(hout.h_res_hole);
      hout.zeta          = BB.zeta_nodes;
      hout.nu            = thermal ? BB.nu_fit() : bos->nu;
      hout.q_weights     = hextra.c;
      hout.qpts          = ibz.active ? nda::array<double, 2>(mf.Qpts()) : nda::array<double, 2>(mf.Qpts_ibz());
      hout.madelung      = madelung;
      hout.eps_inf       = head_eps_inf(hout.h0_res, hout.h0_res_hole, hout.nu);
      hout.extrapolation = prm.head_extrapolation;
      hout.div_treatment = sig_div ? prm.div_treatment : std::string("ignore_g0");
      hout.hf_div_treatment = prm.hf_div_treatment;
      Timer.stop("W_head");
      tW += toc("phase_W", e0);
      const auto h0r = head_eval(hout.h0_res, hout.h0_res_hole, hout.nu, hout.zeta);
      double dh = 0.0, sh = 0.0;
      for (long i = 0; i < nzb; ++i) {
        dh = std::max(dh, std::abs(h0r(i) - hout.h0_nodes(i)));
        sh = std::max(sh, std::abs(hout.h0_nodes(i)));
      }
      app_log(1, "  head: eps_inf = {:.6f} (q -> 0 {}; 1 / (1 + Re h0(0)) from the residues), max|h0| at the nodes {:.4e}, "
                 "residues vs nodes {:.1e}",
              hout.eps_inf, prm.head_extrapolation, sh, sh > 0.0 ? dh / sh : dh);
      res.eps_inf.push_back(hout.eps_inf);
      hout_last = hout;
      have_hout = true;
      res.head_hp0 = hout.h0_res;
      res.head_hh0 = hout.h0_res_hole;
    }

    // 2. Sigma per sector at the dense nodes, mixing
    e0 = tic("phase_Sigma");
    if (qplan.w_host) w = arr4_t{};   // only the last group's rows: free them
    auto const *whp = qplan.w_host ? &w_h : nullptr;
    // both sectors in one call (perf 7.1: the real-space residues are transformed once): particle -> Sp_new, hole -> Sh_new
    if (ibz.active)   // perf 7.3: Sigma at the IBZ k from the class sums (self_energy_ibz.hpp)
      self_energy_ibz<MEM>(prop, kp, w, BB, mf, ibz, grid, comm, zeta, *sig_p, *sig_h, qplan.t_chunk, Sp_new, Timer,
                           sector_t::both, prm.sigma_kdist, &Sh_new);
    else
      self_energy<MEM>(prop, kp, w, BB, mf, grid, mpi, zeta, *sig_p, *sig_h, qplan.t_chunk, Sp_new, Timer, sector_t::both,
                       prm.sigma_kdist, whp, qplan.gs_sigma, &Sh_new, (qplan.wR_inplace and not hybrid) ? &w : nullptr);   // 7.4b: w consumed
    nda::array<ComplexType, 4> Stp_new, Sth_new;   // S8b.3 hybrid: the tau-leg nodal Sigma (hybrid.hpp), same rows as Sp_new
    if (hybrid) {
      Timer.add("Sigma_tau_leg");
      Timer.start("Sigma_tau_leg");
      auto tl = tau_lists(st.poles, prm.beta);
      if (ibz.active)
        self_energy_ibz<MEM>(prop, tl, w, *bosX, mf, ibz, grid, comm, hyb->zdummy, *hyb->kp, *hyb->kh, qplan.t_chunk, Stp_new, Timer,
                             sector_t::both, prm.sigma_kdist, &Sth_new);
      else
        self_energy<MEM>(prop, tl, w, *bosX, mf, grid, mpi, hyb->zdummy, *hyb->kp, *hyb->kh, qplan.t_chunk, Stp_new, Timer,
                         sector_t::both, prm.sigma_kdist, whp, qplan.gs_sigma, &Sth_new, nullptr);
      Timer.stop("Sigma_tau_leg");
    }
    mem_trace(comm, mpi.node_comm, "Sigma");
    if (sig_div) {   // S9a: the q -> 0 head term of Sigma_c (head.hpp), per sector, on the rows of this rank
      Timer.start("Sigma_head");
      if (thermal)   // S8b: weights (1 - f + n), (f + n) over all poles
        head_sigma_thermal(kp, st.poles, Thead, hout.nu, hout.h0_res, hout.h0_res_hole, madelung, prm.beta, E_T, zeta, k_rows, Sp_new,
                           Sh_new);
      else
        head_sigma_correction(st.poles, Thead, bos->nu, hout.h0_res, hout.h0_res_hole, madelung, zeta, k_rows, Sp_new, Sh_new);
      Timer.stop("Sigma_head");
    }
    if (prm.debug_noise_sigma > 0.0 and st.iter + 1 == prm.debug_noise_iter and not qp_pending) {
      // diagnostic (noise-floor meter): relative complex Gaussian noise on the new Sigma, seeded per GLOBAL k (rank-count
      // independent), scale max|Sigma^> + Sigma^<| over all k and nodes
      double smax = nda::max_element(nda::abs(Sp_new + Sh_new));
      if (prm.sigma_kdist) smax = comm.all_reduce_value(smax, boost::mpi3::max<>{});
      for (long l = 0; l < Sp_new.extent(0); ++l) {
        const long k = prm.sigma_kdist ? kd.global(l, kd.rank) : l;
        std::mt19937_64 gen(0xabcdull + 1000003ull * std::uint64_t(prm.debug_noise_seed) + std::uint64_t(k));
        std::normal_distribution<double> N01;
        for (auto *S : {&Sp_new, &Sh_new})
          for (long iz = 0; iz < nz; ++iz)
            for (long i = 0; i < nb; ++i)
              for (long j = 0; j < nb; ++j) (*S)(l, iz, i, j) += prm.debug_noise_sigma * smax * ComplexType(N01(gen), N01(gen));
      }
    }
    if (qp_pending) {   // perf 7.2 qp_diag start: diagonal QP energies from this Sigma[G_KS], no closure, not an iteration
      nda::array<ComplexType, 3> Hq(nk, nb, nb);
      nda::array<double, 2> eks(nk, nb);
      for (long ik = 0; ik < nk; ++ik) {
        for (long n = 0; n < nb; ++n) eks(ik, n) = eig(ik, n) - st.mu;
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j) Hq(ik, i, j) = H0(ik, i, j) + st.F(ik, i, j) - (i == j ? st.mu : 0.0);
      }
      auto qp = qp_diag_energies(comm, eks, Hq, Sp_new, Sh_new, k_rows, prm.sigma_kdist, zeta, *bp, *bh, cprm, prm.start_eta);
      nda::array<double, 2> Eabs(nk, nb);
      for (long ik = 0; ik < nk; ++ik)
        for (long n = 0; n < nb; ++n) Eabs(ik, n) = qp.e(ik, n) + st.mu;
      app_log(1, "  qp_diag start pass: diagonal QP equation solved for {} (k, band) by Newton, {} linearized; max shift {:.4f} eV",
              qp.n_newton, qp.n_lin, qp.max_shift * HA_EV);
      apply_qp_start(Eabs, "qp_diag");
      qp_pending = false;
      res.eps_inf.pop_back();   // the head of the KS pass is not an iteration's
      have_hout = false;
      Timer.stop("phase_Sigma");
      Timer.start("checkpoint");
      write_state(comm, chk, st, nullptr, kdp, sig_all);
      write_start_done(1);
      Timer.stop("checkpoint");
      Timer.stop("iteration");
      res.start_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      app_log(1, "  qp_diag start pass: {:.1f} s (Pi {:.1f}s W {:.1f}s Sigma {:.1f}s)", res.start_time, tPi, tW,
              Timer.elapsed("phase_Sigma") - e0);
      continue;
    }
    Timer.start("Sigma_mix");
    // perf 7.2: linear (bitwise the pre-7.2 arithmetic) or DIIS mixing (scf_mixing.hpp); F_next = the F of this closure
    mix_info_t mi;
    nda::array<ComplexType, 3> F_next = st.F;
    const bool mixed = st.have_sigma;
    if (mixed)
      mi = mixer.step(comm, st.iter + 1, nk, k_rows, prm.sigma_kdist, st.Sig_p, st.Sig_h, Sp_new, Sh_new, st.F_cl, st.F, F_next);
    else
      for (long a = 0; a < st.F.size(); ++a) mi.residF = std::max(mi.residF, std::abs(st.F.data()[a] - st.F_cl.data()[a]));
    const double dS = mi.dS;
    st.Sig_p      = Sp_new;
    st.Sig_h      = Sh_new;
    st.have_sigma = true;
    if (hybrid) {   // S8b.3: the tau-leg Sigma mixed with the same linear factor (python mixes Sigma(i w_n), S1, S2: linear)
      if (mixed and st.have_stau) {
        utils::check(mi.kind == "linear" or mi.kind == "damped", "gw_line: hybrid mixing needs a linear step (got {})", mi.kind);
        const double a = mi.kind == "damped" ? prm.mix.damp_mixing : prm.mixing;
        for (long x = 0; x < Stp_new.size(); ++x) {
          Stp_new.data()[x] = a * Stp_new.data()[x] + (1.0 - a) * st.Stau_p.data()[x];
          Sth_new.data()[x] = a * Sth_new.data()[x] + (1.0 - a) * st.Stau_h.data()[x];
        }
      }
      st.Stau_p    = std::move(Stp_new);
      st.Stau_h    = std::move(Sth_new);
      st.have_stau = true;
    }
    Timer.stop("Sigma_mix");
    const double tS = toc("phase_Sigma", e0);

    // 3. closure with H_stat - mu = H0 + F - mu
    e0 = tic("phase_closure");
    nda::array<ComplexType, 3> Hrel(nk, nb, nb);
    for (long ik = 0; ik < nk; ++ik)
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) Hrel(ik, i, j) = H0(ik, i, j) + F_next(ik, i, j) - (i == j ? st.mu : 0.0);
    st.F_cl = F_next;   // perf 7.2: the F of the closure that builds the new poles (spectra, restart)
    cprm.phi_prev.assign(st.phi.begin(), st.phi.end());   // S7f phase continuity (used only if phase_keep > 0)
    // S8b: thermal closure (total fit, omega_p floor, mu rule) or, with beta > 0 and an empty window, the T = 0 fits with the
    // rule (which keeps the gap midpoint while the window stays empty: bitwise the T = 0 path)
    closure_params_t cprm_it = cprm;
    double wp_used = cprm.wp;
    if (thermal) {
      wp_used    = std::max(cprm.wp, prm.wp_floor * tpar.zeta_T());
      cprm_it.wp = wp_used;
      cth.active = true;
      // S8b thermal closure perf: the terminal-phase scan the omega_p floor makes necessary at every k (n_free > 0) is done
      // "lazy" (cayley.hpp: poly errors + eigenvalue-only unity screen, serial on the owner with its lent cores) instead of
      // the distributed eigen scan; env COQUI_GWLINE_THERMAL_SCAN = parallel restores the S8b.2 path
      cprm_it.scan = detail::env_string("COQUI_GWLINE_THERMAL_SCAN", "lazy");
    } else
      cth.active = false;
    if (tpar.on() and not thermal) cth.tp = tpar;
    auto co = closure(comm, Hrel, st.Sig_p, st.Sig_h, zeta, *bp, *bh, gp, gh, thermal ? cprm_it : cprm, nelec, Timer, grepr,
                      tpar.on() ? &cth : nullptr);
    st.phi = nda::array<double, 1>(nk);
    for (long ik = 0; ik < nk; ++ik) st.phi(ik) = co.diag[ik].phi;
    {   // S7f: how close the hard decisions of the upfolding are to flipping (closure noise floor)
      long gn = 0, sn = 0, kept = 0;
      double gm = 1e300, sm = 1e300, tie = 1e300;
      for (auto const &d : co.diag) {
        gn += d.gram_near; sn += d.svd_near; kept += d.phi_kept;
        gm = std::min(gm, d.gram_margin); sm = std::min(sm, d.svd_margin); tie = std::min(tie, d.phi_tie);
      }
      app_log(2, "          closure decisions: Gram eigenvalues within x10 of the cut {} (min margin {:.2e} dec), singular values "
                 "within x10 of tol_svd {} (min margin {:.2e} dec), closest phase-basin tie {:.3f}, phase basins kept {}",
              gn, gm, sn, sm, tie, kept);
    }
    const double tC = toc("phase_closure", e0);
    mem_trace(comm, mpi.node_comm, "closure");
    st.mu_sigma = st.mu;
    // S8b.3 hybrid: mu, D and N from the Matsubara Dyson equation; the closure's poles re-centred at that mu
    std::optional<hybrid_density_t> hd;
    double hyb_dmu = co.dmu, t_hyb = 0.0, N_cl = 0.0, Dcl_err = 0.0;
    if (hybrid) {
      e0 = tic("phase_closure");
      Timer.add("hybrid_density");
      Timer.start("hybrid_density");
      hd.emplace(matsubara_density(comm, Hrel, st.Stau_p, st.Stau_h, k_rows, cprm.k_weight, *hyb, prm.beta, prm.hyb_wmax, nelec));
      Timer.stop("hybrid_density");
      t_hyb   = toc("phase_closure", e0);
      hyb_dmu = hd->dmu;
      const double sh = co.dmu - hyb_dmu;   // closure energies (about mu_old + co.dmu) -> about mu_old + hyb_dmu
      N_cl = numerics::line_dlr::electron_count_T(co.leh.e, co.leh.v, prm.beta, -sh, cprm.k_weight);
      auto [le, lv] = lehmann_lists(co.poles);
      for (auto &e : le) e += sh;
      co.poles = pole_data_t::from_lehmann(le, lv);
      co.e_homo += sh;
      co.e_lumo += sh;
      auto Dcl = density_fermi(co.poles, prm.beta);
      for (long x = 0; x < Dcl.size(); ++x) Dcl_err = std::max(Dcl_err, std::abs(Dcl.data()[x] - hd->D.data()[x]));
    }
    st.mu += hyb_dmu;
    st.dmu    = hyb_dmu;
    st.e_homo = co.e_homo;
    st.e_lumo = co.e_lumo;
    st.poles  = std::move(co.poles);

    // 4. static part of the new poles
    e0 = tic("phase_F");
    // S8b: the thermal hole list's density when the new G has a window; S8b.3 hybrid: the Matsubara density
    auto D = hybrid ? nda::array<ComplexType, 3>(hd->D) : density_of(st.poles);
    static_F(D, st.F);
    if (hf_div) exchange_head_correction(st.F, D, madelung);
    const double tF = toc("phase_F", e0);
    mem_trace(comm, mpi.node_comm, "F");
    {
      const double np_tot = double(st.poles.total_poles()), MB = 1024.0 * 1024.0;
      app_log(2, "          pole residues ({}): {:.0f} poles, {:.3f} MB host + the same mirrored by the propagator (matrix form "
                 "would be {:.3f} MB)",
              st.poles.is_factorized() ? "factorized nb x M" : "matrix M x nb^2", np_tot, st.poles.residue_bytes() / MB,
              np_tot * double(nb * nb) * 16.0 / MB);
    }

    st.iter += 1;
    gw_line_iter_t rec;
    rec.iter          = st.iter;
    rec.dSigma        = dS;
    rec.mu            = st.mu;
    rec.dmu           = co.dmu;
    rec.gap           = co.e_lumo - co.e_homo;
    rec.e_homo        = co.e_homo;
    rec.e_lumo        = co.e_lumo;
    rec.nelec         = co.nel_compressed;
    rec.nelec_lehmann = co.nel_lehmann;
    rec.N_mu          = co.N_mu;
    rec.dropped       = co.dropped;
    rec.heldout_max   = *std::max_element(co.heldout.begin(), co.heldout.end());
    rec.npoles_min    = *std::min_element(co.npoles.begin(), co.npoles.end());
    rec.npoles_max    = *std::max_element(co.npoles.begin(), co.npoles.end());
    rec.bos_gap       = bos->gap;
    rec.sigma_gap_p   = bp->gap[1];
    rec.sigma_gap_h   = bh->gap[0];
    rec.time_grid     = prm.time_grid;
    rec.nt_pi_p       = pi_p->size();
    rec.nt_pi_h       = pi_h->size();
    rec.nt_sig_p      = sig_p->size();
    rec.nt_sig_h      = sig_h->size();
    rec.g_repr        = co.repr;
    pole_counts(st.poles, rec.ng_min, rec.ng_max, rec.g_emin);
    rec.pruned_w           = co.pruned_w;
    rec.pruned_w_weight    = co.pruned_w_weight;
    rec.pruned_near        = co.pruned_near;
    rec.pruned_near_weight = co.pruned_near_weight;
    rec.resid              = mi.resid;   // perf 7.2
    rec.residF             = mi.residF;
    rec.mix                = mixed ? mi.kind : std::string("none");
    rec.ndiis              = mi.m;
    rec.level              = cur_level;
    if (tpar.on()) {   // S8b
      rec.thermal = thermal ? 1 : 0;
      rec.mu_rule = co.mu_rule;
      rec.dN      = co.dN;
      rec.n_th    = co.n_th;
      rec.wp_used = wp_used;
      rec.nD      = BB.zeta_nodes.size();
      rec.rank_b  = BB.rank_fit();
      rec.ntau    = taun.size();
      rec.nwin    = nwin;
      if (hybrid) {   // S8b.3
        rec.mu_rule      = "matsubara";
        rec.rule_closure = co.mu_rule;
        rec.N_mu         = hd->N;
        rec.N_trace      = hd->N_trace;
        rec.N_closure    = N_cl;
        rec.D_cl_err     = Dcl_err;
        rec.dmu_closure  = co.dmu - hyb_dmu;
        rec.tail_max     = hd->tail_max;
        rec.nfreq        = hd->nfreq;
        rec.dmu          = hyb_dmu;
        rec.nelec        = hd->N;
        app_log(1, "          hybrid: mu {:.12f} Ha (dmu {:+.4e} Ha), N(mu) {:.15f} (trace {:.15f}), {} frequencies, {} bisection steps, tail "
                   "max {:.1e}, {:.2f} s | closure at this mu: N {:.10f} (dN {:+.2e}), max|D_cl - D| {:.2e}, own mu {:+.3e} Ha ({})",
                st.mu, hyb_dmu, hd->N, hd->N_trace, hd->nfreq, hd->nbisect, hd->tail_max, t_hyb, N_cl, N_cl - nelec, Dcl_err,
                co.dmu - hyb_dmu, co.mu_rule);
      }
      app_log(1, "          finite T: {} iteration (window poles {}), mu rule {} (dN = N_T(mu_g) - N_el {:+.3e}, n_th {:.3e}), N(mu) {:.12f}, "
                 "omega_p {:.4f}, |D| {}, rank_b {}, tau nodes {}; next window {}",
              thermal ? "thermal" : "T = 0", nwin, co.mu_rule, co.dN, co.n_th, co.N_mu, wp_used, rec.nD, rec.rank_b, rec.ntau,
              co.window_next ? "non-empty" : "empty");
    }
    rec.time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    res.history.push_back(rec);
    print_line(rec, tPi, tW, tS, tC, tF);
    app_log(1, "          mixing: {}{}, residual max|Sigma[G] - Sigma_in| {:.3e} (||r|| {:.3e}), max|F[D] - F_in| {:.3e}{}{}", rec.mix,
            rec.mix == "diis" ? " (" + std::to_string(mi.m) + " entries, max|c| " + std::to_string(mi.cmax).substr(0, 6) + ")" : "",
            rec.resid, mi.rnorm, rec.residF, cur_level ? ", level coarse" : "",
            mixer.size() > 0 ? ", DIIS history " + std::to_string(mixer.bytes() / 1048576.0).substr(0, 7) + " MB per rank" : "");
    if (st.iter == 1) {
      hout_first      = hout;
      have_hout_first = true;
    }

    Timer.start("checkpoint");
    write_state(comm, chk, st, &rec, kdp, sig_all, &hout);
    mem_trace(comm, mpi.node_comm, "checkpoint");
    Timer.stop("checkpoint");
    Timer.stop("iteration");
    if (comm.root() and std::filesystem::exists(chk)) {
      const auto ic = std::distance(phase_names.begin(), std::find(phase_names.begin(), phase_names.end(), "checkpoint"));
      const double sf = std::filesystem::exists(sigma_file(chk)) ? double(std::filesystem::file_size(sigma_file(chk))) : 0.0;
      app_log(1, "  checkpoint {}: {:.3f} GB (+ Sigma file {:.3f} GB) after iteration {} (write {:.2f} s)", chk,
              double(std::filesystem::file_size(chk)) / 1073741824.0, sf / 1073741824.0, st.iter, Timer.elapsed("checkpoint") - ph0[ic]);
    }
    // + the pole residues mirrored by the propagator and its host XV factors (S7e; they depend on the poles of the iteration)
    report_iteration(comm, st.iter, ph0, phase_snapshot(Timer), rss0, model_host + prop.pole_bytes() + prop.xv_bytes(), model_dev,
                     MEM != HOST_MEMORY);
    {
      const double xv = comm.all_reduce_value(prop.xv_bytes(), boost::mpi3::max<>{});
      const long nxv  = comm.all_reduce_value(prop.n_xv, boost::mpi3::max<>{});
      if (MEM == HOST_MEMORY)
        app_log(2, "  propagator: G~ blocks in the XV form for {} of {} (k, sector) pairs (XV factors <= {:.3f} GB per rank)", nxv,
                2 * nk, xv / 1073741824.0);
    }

    // perf 7.2: only after a mixing step (an unmixed iteration has dS = 0); DIIS also needs mixing x residual < conv_thr
    if (rec.iter > 1 and mixed and dS < prm.conv_thr and
        (mi.kind == "linear" or mi.kind == "damped" or prm.mixing * mi.resid < prm.conv_thr)) {
      converged = true;
      app_log(1, "  converged: max|dSigma| = {:.2e} < conv_thr = {:.1e}", dS, prm.conv_thr);
      break;
    }
  }
  if (not converged and not res.history.empty() and res.history.back().iter > 1 and res.history.back().dSigma < prm.conv_thr) {
    auto const &h = res.history.back();
    converged     = (h.mix != "none" or (h.resid == 0.0 and h.dSigma > 0.0)) and
                (h.mix != "diis" and h.mix != "reset" ? true : prm.mixing * h.resid < prm.conv_thr);
  }

  // ---------------------------------------------------------------------------------------------- spectra
  if (prm.do_spectra and st.have_sigma) {
    Timer.start("spectra");
    nda::array<ComplexType, 3> Hrel(nk, nb, nb);
    for (long ik = 0; ik < nk; ++ik)
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) Hrel(ik, i, j) = H0(ik, i, j) + st.F_cl(ik, i, j) - (i == j ? st.mu : 0.0);
    // perf 7.2: the G of the stored poles: F of their closure (F_closure), not F[D] of the poles themselves
    double dFc = 0.0;
    for (long a = 0; a < st.F.size(); ++a) dFc = std::max(dFc, std::abs(st.F.data()[a] - st.F_cl.data()[a]));
    app_log(1, "  spectra: G of iteration {} (the stored poles): H0 + F_closure(iteration {}) + Sigma_c(iteration {}){}; max|F[D_{}] - "
               "F_closure| = {:.3e} Ha (not used)",
            st.iter, st.iter, st.iter,
            st.iter == 1 and prm.start == "ks" ? " = G0W0@KS (F[D_KS] + Sigma[G_KS])"
                                               : (prm.mix.mix_F and not res.history.empty() and res.history.back().mix == "diis" ? " (F_closure = the DIIS-extrapolated F)" : " (F_closure = F[D_" + std::to_string(st.iter - 1) + "])"),
            st.iter, dFc);
    // S8b: a thermal last iteration -> the total fit of its closure (two-sided basis, mask, omega_p floor)
    const bool th_last = tpar.on() and not res.history.empty() and res.history.back().thermal == 1;
    closure_params_t cprm_sp = cprm;
    if (th_last) {
      ensure_thermal();
      cth.active = true;
      cprm_sp.wp = res.history.back().wp_used;
    }
    auto sp = line_spectra(comm, Hrel, st.Sig_p, st.Sig_h, zeta, st.mu - st.mu_sigma, *bp, *bh, th_last ? cprm_sp : cprm, nelec,
                           prm.spectra, th_last ? &cth : nullptr);
    write_spectra(comm, chk, sp, st.mu);
    if (prm.spectra_occupation and prm.beta > 0.0 and comm.root()) {   // S8b: f(w - mu) on the spectra grid
      utils::h5_quiesce();
      h5::file f(chk, 'a');
      h5::group g(f);
      auto sg = g.open_group("spectra");
      nda::array<double, 1> fw(sp.omega.size());
      for (long i = 0; i < fw.size(); ++i) fw(i) = numerics::line_dlr::fermi(sp.omega(i), prm.beta);
      nda::h5_write(sg, "fermi_w", fw, false);
    }
    comm.barrier();
    Timer.stop("spectra");
    app_log(1, "  spectra: A(k, w) for {} eta x {} k x {} w written to {}:/spectra ({:.1f} s)", sp.eta.size(), nk, sp.omega.size(),
            chk, Timer.elapsed("spectra"));
    app_log(1, "  final QP edges: VBM {:.6f} Ha, CBM {:.6f} Ha (mu {:.6f} Ha) -> QP gap {:.4f} eV", st.mu + sp.e_homo,
            st.mu + sp.e_lumo, st.mu, (sp.e_lumo - sp.e_homo) * HA_EV);
    res.spectra = std::move(sp);
  } else if (prm.do_spectra) {
    app_log(1, "  spectra: no Sigma in the state (no iteration done), skipped");
  }

  // ---------------------------------------------------------------------------------------------- optics (S9b)
  if (prm.optics.enable) {
    Timer.add("optics");
    Timer.add("optics_pass");
    Timer.start("optics");
    Pi = arr4_t{};   // the SCF's W-stage buffers are not needed any more
    w  = arr4_t{};
    Wn = arr4_t{};
    w_h = memory::array<HOST_MEMORY, ComplexType, 4>{};
    auto const &op = prm.optics;
    auto make_line = [&](double th_deg) {
      optics_line_t L;
      L.theta_deg = th_deg;
      L.q_weights = hextra.c;
      L.qpts      = ibz.active ? nda::array<double, 2>(mf.Qpts()) : nda::array<double, 2>(mf.Qpts_ibz());
      L.lattv     = nda::array<double, 2>(mf.lattv());
      L.qminus    = qminus_list(mf);
      L.qfac.resize(nq);
      for (long q = 0; q < nq; ++q) {   // perf 7.3: f(q) of every q of the mesh (hbasis holds only the rows R)
        const double q2 = L.qpts(q, 0) * L.qpts(q, 0) + L.qpts(q, 1) * L.qpts(q, 1) + L.qpts(q, 2) * L.qpts(q, 2);
        L.qfac[q]       = ibz.active ? q2 / (4.0 * std::numbers::pi) * mf.volume() : hbasis.fac(q);
      }
      for (auto const &v : op.q0_variants) {   // sensitivity of q0 to the extrapolation variant
        if (v == prm.head_extrapolation) continue;
        head_extrapolation_t hv(mf, v, ibz.active);
        L.variant_names.push_back(v);
        L.variant_weights.push_back(hv.c);
      }
      return L;
    };
    auto pass_params = [&](double th_deg, bool flat) {
      head_pass_params_t hp;
      hp.theta_deg = th_deg;
      hp.lam_b     = prm.lam_b;
      hp.eps       = prm.eps;
      hp.bos_gap   = bos->gap;
      hp.time_grid = flat ? op.time_grid : prm.time_grid;
      hp.time_eps  = prm.time_eps;
      hp.time_pad  = prm.time_pad;
      hp.time_oversample = prm.time_oversample;
      hp.ray_decades = prm.ray_decades;
      hp.t_chunk   = prm.t_chunk;
      hp.mem_gb    = op.mem_gb;
      hp.ibz       = &ibz;
      if (flat) {
        hp.nline = op.nline;
        hp.npole = op.npole;
      } else {   // the SCF basis (bosonic_basis_t defaults)
        hp.nline = 1200;
        hp.npole = 800;
      }
      return hp;
    };
    // perf 7.2: which G the optics lines use (optics.poles): the final poles or those of iteration 0 (KS / qp start)
    const bool initial = (prm.optics_poles == "initial");
    pole_data_t poles0;
    if (initial) {
      if (comm.root()) {
        utils::h5_quiesce();
        h5::file f(chk, 'r');
        h5::group g(f);
        auto it0 = g.open_group("scf_line").open_group("iter0");
        poles0   = read_poles(it0, nk, nb);
      }
      bcast_poles(comm, poles0);
    }
    pole_data_t const &opoles = initial ? poles0 : st.poles;
    const std::string gname   = initial ? "initial poles (iteration 0)" : "final poles (iteration " + std::to_string(st.iter) + ")";
    app_log(1, "  optics: G = {} (optics.poles = {})", gname, prm.optics_poles);
    auto run_pass = [&](optics_line_t &L, bool flat) {
      Timer.start("optics_pass");
      auto hpo = head_pass<MEM>(thc, mf, mpi, grid, prop, opoles, hbasis, pass_params(L.theta_deg, flat));
      Timer.stop("optics_pass");
      L.zeta      = hpo.zeta;
      L.nu        = hpo.nu;
      L.h_nodes   = std::move(hpo.h_nodes);
      L.pass_time = hpo.t_total;
      L.nt_p      = hpo.nt_p;
      L.nt_h      = hpo.nt_h;
      L.time_grid = hpo.time_grid;
      app_log(1, "  optics: head pass at {} deg from the {}: bosonic rank {} ({} nodes), Pi time nodes {} + {} ({}), {} q "
                 "group(s), {:.1f} s (Z {:.1f}, Pi {:.1f}, W {:.1f})",
              L.theta_deg, gname, hpo.rank, hpo.nz, hpo.nt_p, hpo.nt_h, hpo.time_grid, hpo.ngroups, hpo.t_total, hpo.t_Z, hpo.t_Pi, hpo.t_W);
    };
    std::vector<optics_line_t> lines;
    {   // the SCF angle
      auto L = make_line(prm.theta_deg);
      L.time_grid = prm.time_grid;
      // The head of iteration N is Pi[G_{N-1}] (the poles BEFORE its closure). initial: the head of iteration 1 = Pi[G_0].
      // final: the head of the last iteration if the run converged (G_{N-1} = G_N to the SCF tolerance), else (e.g. G0W0,
      // niter = 1) one pass from the final poles, so that every optics line uses the same G (perf 7.2; before: always
      // the last iteration's head, i.e. Pi[G_KS] at the SCF angle next to flatter lines from G_1 in a niter = 1 run)
      const long ih = initial ? 1 : st.iter;
      if (not initial and not converged and st.iter >= 1) {
        L.source = "recomputed from the " + gname + " (run not converged: the last head is Pi[G_" + std::to_string(st.iter - 1) + "])";
        run_pass(L, false);
      } else if (initial and have_hout_first) {
        L.source  = "head of iteration 1 of this run (from the initial poles)";
        L.zeta    = hout_first.zeta;
        L.nu      = hout_first.nu;
        L.h_nodes = hout_first.h_nodes;
      } else if (not initial and have_hout) {
        L.source  = "head of the last iteration of this run (iteration " + std::to_string(st.iter) + ")";
        L.zeta    = hout_last.zeta;
        L.nu      = hout_last.nu;
        L.h_nodes = hout_last.h_nodes;
      } else if (st.iter >= 1 and read_head_nodes(comm, chk, ih, L.h_nodes, L.zeta, L.nu)) {
        L.source = "checkpoint scf_line/iter" + std::to_string(ih) + "/head";
      } else {
        L.source = "recomputed from the " + gname + " (no head group)";
        run_pass(L, false);
      }
      lines.push_back(std::move(L));
    }
    for (double th : op.theta_deg) {
      auto L   = make_line(th);
      L.source = "flatter-line pass from the " + gname;
      run_pass(L, true);
      lines.push_back(std::move(L));
    }
    const double wp2 = wp2_valence(nelec, mf.volume());
    for (auto const &L : lines) {
      const auto t0 = std::chrono::steady_clock::now();
      auto R        = optics_line(comm, L, op);
      char tag[32];
      std::snprintf(tag, sizeof(tag), "theta%.1f", L.theta_deg);
      write_optics(comm, chk, tag, L, R, op, nelec, mf.volume(),
                   "div_treatment " + prm.div_treatment + ", head extrapolation " + prm.head_extrapolation + ".");
      log_optics(L, R, wp2);
      app_log(1, "  optics {} deg ({}): {} curves x {} broadenings x {} omega in {:.1f} s -> {}:/optics/{}", L.theta_deg, L.source,
              R.size(), op.nbroad(), op.nw, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), chk, tag);
      res.optics_theta.push_back(L.theta_deg);
      for (auto const &o : R)
        if (o.iq == -1) res.optics_q0.push_back(o);
    }
    Timer.stop("optics");
  }

  Timer.stop("total");
  app_log(1, "\n  gw_line timers (s, all iterations of this run):");
  for (auto nm : {"total", "H0", "bases", "time_grid", "phase_Pi", "phase_W", "phase_Sigma", "phase_closure", "phase_F",
                  "checkpoint", "spectra"})
    app_log(1, "    {:<20s} {:10.3f}", nm, Timer.elapsed(nm));
  if (prm.optics.enable)
    app_log(1, "    {:<20s} {:10.3f} (of which head passes {:.3f})", "optics", Timer.elapsed("optics"), Timer.elapsed("optics_pass"));
  app_log(1, "  kernel sub-timers:");
  for (auto const &nm : Timer.timer_names()) {
    if (nm == "total" or nm == "H0" or nm == "bases" or nm == "time_grid" or nm == "checkpoint" or nm == "spectra" or
        nm.rfind("phase_", 0) == 0 or nm.rfind("optics", 0) == 0)
      continue;
    app_log(1, "    {:<20s} {:10.3f}", nm, Timer.elapsed(nm));
  }

  res.converged = converged;
  res.mu        = st.mu;
  res.mu_sigma  = st.mu_sigma;
  res.poles     = std::move(st.poles);
  res.F         = std::move(st.F);
  res.F_closure = std::move(st.F_cl);
  res.H0        = std::move(H0);
  res.Sig_p     = std::move(st.Sig_p);
  res.Sig_h     = std::move(st.Sig_h);
  res.zeta      = zeta;
  return res;
}

template gw_line_result_t gw_line_scf<HOST_MEMORY>(methods::thc_reader_t &, mf::MF &, ptree const &);
#if defined(ENABLE_DEVICE)
template gw_line_result_t gw_line_scf<DEVICE_MEMORY>(methods::thc_reader_t &, mf::MF &, ptree const &);
#endif

} // namespace methods::gw_line
