/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 *
 * Real-axis GW dispatch entry point. Translates the ptree configuration
 * received by methods::mbpt(...) into the real-axis solver pipeline and
 * runs real_axis_scf_loop. Mirrors the imag-axis dispatch in
 * methods::mbpt::"gw" but on the real-frequency grid via THC.
 */

#ifndef COQUI_REAL_AXIS_MBPT_DISPATCH_HPP
#define COQUI_REAL_AXIS_MBPT_DISPATCH_HPP

#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"

#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"

#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_mb_state.hpp"
#include "methods/GW_real_axis/real_axis_scr_coulomb_t.h"
#include "methods/GW_real_axis/real_axis_gw_t.h"
#include "methods/GW_real_axis/real_axis_hf_t.h"
#include "methods/GW_real_axis/real_axis_dyson_t.h"
#include "methods/GW_real_axis/real_axis_scf_driver.hpp"
#include "methods/GW_real_axis/real_axis_scf.hpp"
#include "methods/GW_real_axis/real_axis_qp_context.hpp"
#include "methods/GW_real_axis/real_axis_qp_solver_t.h"
#include "methods/GW_real_axis/real_axis_qp_scf_driver.hpp"
#include "methods/HF/hf_t.h"
#include "hamiltonian/one_body_hamiltonian.hpp"
#include "hamiltonian/pseudo/pseudopot.h"

namespace methods {
namespace real_axis {

/**
 * Default real-frequency grid parameters derived from the MF spectrum.
 * Mirrors the convention used in tests/test_real_axis_*.cpp.
 *
 *   w_max     = max(|e_min|, |e_max|) + 2.0
 *   Omega_max = 2 * w_max
 *   freq_max  = max(w_max, Omega_max)
 *   dt        = 0.5 * pi / freq_max     (Nyquist with safety factor 2)
 *   T_window  = dt * N_t
 */
struct grid_params {
  double beta;
  double mu0;
  double w_max;
  double Omega_max;
  long   N_w;
  long   N_Omega;
  long   N_t;
  double T_window;
};

inline grid_params derive_grid_params(mf::MF const& mf, double beta,
                                      long N_w_user      = 0,
                                      long N_Omega_user  = 0,
                                      long N_t_user      = 0,
                                      double w_max_user  = 0.0,
                                      double Omega_max_user = 0.0)
{
  auto eigval = mf.eigval();
  const long ns_  = mf.nspin();
  const long nbnd = mf.nbnd();
  double e_min =  std::numeric_limits<double>::infinity();
  double e_max = -std::numeric_limits<double>::infinity();
  for (long s = 0; s < ns_; ++s)
    for (long k = 0; k < mf.nkpts_ibz(); ++k)
      for (long n = 0; n < nbnd; ++n) {
        const double e = eigval(s, k, n);
        e_min = std::min(e_min, e);
        e_max = std::max(e_max, e);
      }
  if (e_max - e_min < 1e-3) { e_min -= 1.0; e_max += 1.0; }

  const long n_homo = static_cast<long>(mf.nelec() / 2 - 1);
  const long n_lumo = n_homo + 1;
  const double eps_homo = eigval(0, 0, n_homo);
  const double eps_lumo = eigval(0, 0, std::min(n_lumo, nbnd - 1));

  grid_params p;
  p.beta      = beta;
  p.mu0       = 0.5 * (eps_homo + eps_lumo);
  p.w_max     = (w_max_user > 0.0)
                 ? w_max_user
                 : std::max(std::abs(e_min), std::abs(e_max)) + 2.0;
  p.Omega_max = (Omega_max_user > 0.0) ? Omega_max_user : 2.0 * p.w_max;
  p.N_w       = (N_w_user      > 0) ? N_w_user     : 129;
  p.N_Omega   = (N_Omega_user  > 0) ? N_Omega_user : 64;
  p.N_t       = (N_t_user      > 0) ? N_t_user     : 256;
  const double freq_max = std::max(p.w_max, p.Omega_max);
  const double dt       = 0.5 * M_PI / freq_max;
  p.T_window  = dt * static_cast<double>(p.N_t);
  return p;
}

/**
 * Entry point for the real-axis GW dispatcher. Called from
 * methods::mbpt(solver_type == "real_axis_gw", eri, pt).
 *
 * Currently THC-only (the real-axis pipeline does not implement Cholesky).
 * Caller must pass a THC-backed eri.
 */
template<methods::THC_ERI THC_t>
inline void run_real_axis_gw(THC_t& thc, ptree const& pt)
{
  using cval_t = ComplexType;

  auto mf  = thc.MF();
  auto mpi = thc.mpi();

  // ---- Read ptree params --------------------------------------------------
  // Imag-axis-style names where they apply; real-axis-specific knobs flagged
  // explicitly.
  const auto beta       = io::get_value_with_default<double>(pt, "beta", 1000.0);
  const auto niter      = io::get_value_with_default<int>   (pt, "niter", 1);
  const auto conv_thr   = io::get_value_with_default<double>(pt, "conv_thr", 1e-6);
  const auto mu_tol     = io::get_value_with_default<double>(pt, "mu_tolerance", 1e-9);
  const auto const_mu   = io::get_value_with_default<bool>  (pt, "const_mu", false);
  const auto verbose    = io::get_value_with_default<bool>  (pt, "verbose", false);
  auto       div_treat  = io::get_value_with_default<std::string>(pt, "div_treatment", "ignore_g0");
  auto       hf_div_t   = io::get_value_with_default<std::string>(pt, "hf_div_treatment", "ignore_g0");
  auto       screen_t   = io::get_value_with_default<std::string>(pt, "screen_type", "rpa");
  io::tolower(div_treat);
  io::tolower(hf_div_t);
  io::tolower(screen_t);

  // Real-axis-specific:
  const auto N_w_p      = io::get_value_with_default<long>  (pt, "N_w", 0);
  const auto N_Omega_p  = io::get_value_with_default<long>  (pt, "N_Omega", 0);
  const auto N_t_p      = io::get_value_with_default<long>  (pt, "N_t", 0);
  const auto wmax_p     = io::get_value_with_default<double>(pt, "wmax", 0.0);
  const auto Omegamax_p = io::get_value_with_default<double>(pt, "Omega_max", 0.0);
  const auto eta_p      = io::get_value_with_default<double>(pt, "eta", 0.05);
  const auto eps_nufft  = io::get_value_with_default<double>(pt, "eps_nufft", 1e-10);
  auto       mix_kind_s = io::get_value_with_default<std::string>(pt, "mix_kind", "diis");
  const auto alpha_mix  = io::get_value_with_default<double>(pt, "alpha_mix", 0.7);
  const auto diis_win   = io::get_value_with_default<long>  (pt, "diis_window", 8);
  // Non-uniform fermionic-grid options. grid_kind="uniform" (default,
  // back-compat) or "nonuniform_log" (linear-dense around mu, log tails).
  // w_dense / N_dense control the dense block; ignored for uniform.
  auto       grid_kind_s = io::get_value_with_default<std::string>(pt, "grid_kind", "uniform");
  const auto w_dense_p   = io::get_value_with_default<double>(pt, "w_dense", 0.0);
  const auto N_dense_p   = io::get_value_with_default<long>  (pt, "N_dense", 0);
  io::tolower(mix_kind_s);
  io::tolower(grid_kind_s);

  // ---- Validation ---------------------------------------------------------
  // Accepted div_treatment values:
  //   "ignore_g0"        -- skip q=Gamma, no head correction
  //   "gygi_smallest_q"  -- use eps_inv_head from smallest-|q| in IBZ
  //   "gygi"             -- alias for gygi_smallest_q until polynomial-fit
  //                          extrapolation is ported from g0_div_utils
  if (div_treat == "gygi") {
    if (mpi->comm.root())
      app_log(2, "real_axis_gw: div_treatment=\"gygi\" mapped to "
                 "\"gygi_smallest_q\" -- polynomial-fit extrapolation is "
                 "open work; the smallest-|q| estimate is used.");
    div_treat = "gygi_smallest_q";
  }
  utils::check(div_treat == "ignore_g0" or div_treat == "gygi_smallest_q",
               "real_axis_gw: div_treatment must be \"ignore_g0\" or "
               "\"gygi_smallest_q\" (got \"{}\"). Polynomial-fit \"gygi\" "
               "extrapolation is not yet ported from the imag-axis side.",
               div_treat);
  utils::check(screen_t == "rpa",
               "real_axis_gw: only screen_type=\"rpa\" is supported (got "
               "\"{}\"). EDMFT/cRPA hooks are imag-axis only at present.",
               screen_t);
  utils::check(mix_kind_s == "linear" or mix_kind_s == "diis",
               "real_axis_gw: mix_kind must be \"linear\" or \"diis\" (got \"{}\")",
               mix_kind_s);
  utils::check(grid_kind_s == "uniform" or grid_kind_s == "nonuniform_log",
               "real_axis_gw: grid_kind must be \"uniform\" or \"nonuniform_log\" (got \"{}\")",
               grid_kind_s);

  // ---- Grid + state -------------------------------------------------------
  auto p = derive_grid_params(*mf, beta, N_w_p, N_Omega_p, N_t_p,
                              wmax_p, Omegamax_p);
  // Nonuniform-grid defaults: dense halfwidth defaults to max(8*eta, 0.4 Ha)
  // (covers the QP / band-edge region for typical solids); N_dense defaults
  // to ~half of N_w. Both are user-overridable.
  const double w_dense_eff = (w_dense_p > 0.0)
                              ? w_dense_p
                              : std::max(8.0 * eta_p, 0.4);
  const long   N_dense_eff = (N_dense_p > 0)
                              ? N_dense_p
                              : (((p.N_w / 2) | 1) /* odd if N_w even */);
  auto grid = (grid_kind_s == "nonuniform_log")
              ? real_freq_grid_t::make_nonuniform_log(
                  p.beta, p.mu0, p.w_max, p.N_w,
                  w_dense_eff, N_dense_eff,
                  p.Omega_max, p.N_Omega, p.N_t, p.T_window)
              : real_freq_grid_t::make_uniform(
                  p.beta, p.mu0, p.w_max, p.N_w,
                  p.Omega_max, p.N_Omega, p.N_t, p.T_window);

  real_axis_mb_state_t state(grid);
  state.mpi = mpi;

  if (verbose and mpi->comm.root()) {
    app_log(1, "");
    app_log(1, "╔══════════════════════════════════════════════════════════╗");
    app_log(1, "║                Real-axis GW (THC)                        ║");
    app_log(1, "╚══════════════════════════════════════════════════════════╝");
    app_log(2, "  beta        = {}", p.beta);
    app_log(2, "  mu0         = {:.6f}  (gap midpoint)", p.mu0);
    app_log(2, "  w_max       = {:.4f}", p.w_max);
    app_log(2, "  Omega_max   = {:.4f}", p.Omega_max);
    app_log(2, "  N_w         = {}", p.N_w);
    app_log(2, "  N_Omega     = {}", p.N_Omega);
    app_log(2, "  N_t         = {}", p.N_t);
    app_log(2, "  eta         = {}", eta_p);
    app_log(2, "  eps_nufft   = {}", eps_nufft);
    app_log(2, "  mix_kind    = {}", mix_kind_s);
    app_log(2, "  alpha_mix   = {}", alpha_mix);
    app_log(2, "  niter       = {}", niter);
    app_log(2, "  conv_thr    = {}", conv_thr);
    app_log(2, "  grid_kind   = {}", grid_kind_s);
    if (grid_kind_s == "nonuniform_log") {
      app_log(2, "  w_dense     = {:.4f}", w_dense_eff);
      app_log(2, "  N_dense     = {}", N_dense_eff);
    }
  }

  // ---- Build H_MF = diag(eps_KS) at IBZ k. Star expansion to FBZ k
  //      happens inside the BZ-pair kernels via X(FBZ k); orbital-basis
  //      arrays live at IBZ k (matches imag-axis pattern).
  const long ns      = mf->nspin();
  const long Nk_ibz  = mf->nkpts_ibz();
  const long nbnd    = mf->nbnd();
  auto eigval = mf->eigval();
  nda::array<cval_t, 4> H_MF(ns, Nk_ibz, nbnd, nbnd);
  H_MF = cval_t(0.0, 0.0);
  for (long s = 0; s < ns; ++s)
    for (long k = 0; k < Nk_ibz; ++k)
      for (long n = 0; n < nbnd; ++n)
        H_MF(s, k, n, n) = cval_t(eigval(s, k, n), 0.0);

  // ---- Solver bundle + Dyson + mixing config ----------------------------
  real_axis_hf_t          hf(&grid, hf_div_t);
  real_axis_scr_coulomb_t scr(&grid, screen_t, div_treat, eps_nufft);
  methods::solvers::real_axis_gw_t gw(grid, /*max_iter*/ 1, /*mix*/ alpha_mix,
                                       /*eps_nufft*/ eps_nufft, /*ntrans*/ 1);
  real_axis_dyson_t       dyson(std::move(H_MF), &grid, eta_p, mu_tol);
  real_axis_mb_solver_t   mb_solver{&hf, &scr, &gw};

  scgw_config cfg;
  cfg.max_iter    = niter;
  cfg.alpha_mix   = alpha_mix;
  cfg.tol         = conv_thr;
  cfg.eta         = eta_p;
  cfg.eps_nufft   = eps_nufft;
  cfg.update_mu   = !const_mu;
  cfg.verbose     = verbose;
  cfg.mix_kind    = (mix_kind_s == "diis") ? scgw_mix_kind::diis
                                            : scgw_mix_kind::linear;
  cfg.diis_window = diis_win;

  // ---- BZ weights, electron count ---------------------------------------
  // k_weight is IBZ-sized with multiplicity-weighted entries summing to 1.
  nda::array<double, 1> k_weights(Nk_ibz);
  auto kw = mf->k_weight();
  for (long ik = 0; ik < Nk_ibz; ++ik)
    k_weights(ik) = kw(ik);
  const double N_elec = static_cast<double>(mf->nelec());

  // ---- Run SCF ----------------------------------------------------------
  auto res = real_axis_scf_loop(state, dyson, thc, mb_solver, cfg,
                                k_weights, N_elec);

  if (mpi->comm.root()) {
    app_log(1, "");
    app_log(1, "real_axis_gw: iter_used={}  ||dA||={:.3e}  mu={:.6f}  converged={}",
            res.iter_used, res.final_diff, res.final_mu,
            res.converged ? "yes" : "no");

    // Direct gap at k=Gamma (k=0 in the FBZ): peak-pick the diagonal
    // spectral function A(omega; s, k=0, n=HOMO) below mu and
    // A(omega; s, k=0, n=LUMO) above mu. Dump to A_gamma.dat for
    // offline analysis.
    if (state.A_wskij.has_value()) {
      auto A    = state.A_wskij->local();
      auto wgrid = grid.w();
      const long Nw   = A.shape()[0];
      const long n_homo = static_cast<long>(mf->nelec() / 2 - 1);
      const long n_lumo = std::min<long>(n_homo + 1, nbnd - 1);
      double w_homo_peak = 0.0, A_homo_peak = -1.0;
      double w_lumo_peak = 0.0, A_lumo_peak = -1.0;
      const double mu_abs = res.final_mu;
      std::ofstream of("A_gamma.dat");
      of << "# w_abs  A_homo(w)  A_lumo(w)\n";
      for (long iw = 0; iw < Nw; ++iw) {
        const double w = wgrid(iw) + mu_abs;  // grid is omega-mu
        const double aH = std::abs(A(iw, 0, 0, n_homo, n_homo).imag()) / M_PI;
        const double aL = std::abs(A(iw, 0, 0, n_lumo, n_lumo).imag()) / M_PI;
        of << w << "  " << aH << "  " << aL << "\n";
        if (w < mu_abs and aH > A_homo_peak) {
          A_homo_peak = aH; w_homo_peak = w;
        }
        if (w > mu_abs and aL > A_lumo_peak) {
          A_lumo_peak = aL; w_lumo_peak = w;
        }
      }
      of.close();
      app_log(1, "real_axis_gw: direct gap @ Gamma : HOMO_peak = {:.4f} Ha, "
                  "LUMO_peak = {:.4f} Ha, gap = {:.4f} Ha = {:.4f} eV",
              w_homo_peak, w_lumo_peak,
              w_lumo_peak - w_homo_peak,
              (w_lumo_peak - w_homo_peak) * 27.211386245988);
    }
  }
}

/**
 * Entry point for the real-axis QP-SCF dispatcher (QSGW / evGW). Called
 * from methods::mbpt(solver_type == "real_axis_qpgw", eri, pt).
 *
 * Reads ptree key `mode` ("qsgw" or "evgw"). For evGW with niter==1 this
 * is the standard real-axis G0W0-QP. THC-only.
 */
template<methods::THC_ERI THC_t>
inline void run_real_axis_qpgw(THC_t& thc, ptree const& pt)
{
  using cval_t = ComplexType;

  auto mf  = thc.MF();
  auto mpi = thc.mpi();

  // ---- ptree params -------------------------------------------------------
  const auto beta       = io::get_value_with_default<double>(pt, "beta", 1000.0);
  const auto niter      = io::get_value_with_default<int>   (pt, "niter", 1);
  const auto conv_thr   = io::get_value_with_default<double>(pt, "conv_thr", 1e-4);
  const auto verbose    = io::get_value_with_default<bool>  (pt, "verbose", false);
  auto       mode_s     = io::get_value_with_default<std::string>(pt, "mode", "qsgw");
  auto       div_treat  = io::get_value_with_default<std::string>(pt, "div_treatment", "ignore_g0");
  auto       hf_div_t   = io::get_value_with_default<std::string>(pt, "hf_div_treatment", "ignore_g0");
  auto       screen_t   = io::get_value_with_default<std::string>(pt, "screen_type", "rpa");
  auto       qp_type    = io::get_value_with_default<std::string>(pt, "qp_type", "bisection");
  auto       odm_s      = io::get_value_with_default<std::string>(pt, "off_diag_mode", "qp_energy");
  auto       mix_kind_s = io::get_value_with_default<std::string>(pt, "mix_kind", "diis");
  io::tolower(mode_s); io::tolower(div_treat); io::tolower(hf_div_t);
  io::tolower(screen_t); io::tolower(qp_type); io::tolower(odm_s);
  io::tolower(mix_kind_s);

  const auto N_w_p      = io::get_value_with_default<long>  (pt, "N_w", 0);
  const auto N_Omega_p  = io::get_value_with_default<long>  (pt, "N_Omega", 0);
  const auto N_t_p      = io::get_value_with_default<long>  (pt, "N_t", 0);
  const auto wmax_p     = io::get_value_with_default<double>(pt, "wmax", 0.0);
  const auto Omegamax_p = io::get_value_with_default<double>(pt, "Omega_max", 0.0);
  const auto eta_p      = io::get_value_with_default<double>(pt, "eta", 0.05);
  const auto qp_eta_p   = io::get_value_with_default<double>(pt, "qp_eta", 1e-3);
  const auto qp_tol     = io::get_value_with_default<double>(pt, "qp_tol", 1e-8);
  const auto eps_nufft  = io::get_value_with_default<double>(pt, "eps_nufft", 1e-8);
  const auto alpha_mix  = io::get_value_with_default<double>(pt, "alpha_mix", 0.7);
  const auto diis_win   = io::get_value_with_default<long>  (pt, "diis_window", 8);
  const auto update_W   = io::get_value_with_default<bool>  (pt, "update_W", true);
  // MO Procrustes alignment to remove rotation drift across iters in
  // degenerate / near-degenerate ε-clusters. Default on; helps both
  // ‖dH_eff‖_F as a meaningful metric and DIIS convergence speed.
  const auto align_mo_p   = io::get_value_with_default<bool>  (pt, "align_mo", true);
  const auto dE_cluster_p = io::get_value_with_default<double>(pt, "dE_cluster_align", 1e-3);
  // Rotation-invariant alternative stopping criteria (in addition to
  // ||dH_eff||_F < conv_thr). Set to 0 to disable.
  const auto tol_max_de_p = io::get_value_with_default<double>(pt, "tol_max_de", 1e-3);
  const auto tol_dDm_p    = io::get_value_with_default<double>(pt, "tol_dDm",    1e-3);
  // Non-uniform fermionic-grid options (see [real_axis_gw] section docs).
  auto       grid_kind_s = io::get_value_with_default<std::string>(pt, "grid_kind", "uniform");
  const auto w_dense_p   = io::get_value_with_default<double>(pt, "w_dense", 0.0);
  const auto N_dense_p   = io::get_value_with_default<long>  (pt, "N_dense", 0);
  io::tolower(grid_kind_s);
  // Checkpointing options. write_chkpt: enable per-iter h5 dump
  // ({prefix}.mbpt.h5). restart: pick up from existing h5 final_iter rather
  // than starting fresh (and don't overwrite metadata). Both default off so
  // legacy invocations are unchanged. Outer-SCF wrappers reusing a prefix
  // MUST set restart=true on calls 2+ to avoid wiping the file.
  const auto write_chkpt_p = io::get_value_with_default<bool>(pt, "write_chkpt", false);
  const auto restart_p     = io::get_value_with_default<bool>(pt, "restart", false);
  const auto output_p      = io::get_value_with_default<std::string>(pt, "output",
                              "coqui_real_axis");

  utils::check(mode_s == "qsgw" or mode_s == "evgw",
               "real_axis_qpgw: mode must be \"qsgw\" or \"evgw\" (got \"{}\")", mode_s);
  if (div_treat == "gygi") {
    if (mpi->comm.root())
      app_log(2, "real_axis_qpgw: div_treatment=\"gygi\" mapped to \"gygi_smallest_q\".");
    div_treat = "gygi_smallest_q";
  }
  utils::check(div_treat == "ignore_g0" or div_treat == "gygi_smallest_q",
               "real_axis_qpgw: div_treatment must be \"ignore_g0\" or \"gygi_smallest_q\" (got \"{}\")",
               div_treat);
  utils::check(screen_t == "rpa",
               "real_axis_qpgw: only screen_type=\"rpa\" supported (got \"{}\")", screen_t);
  utils::check(mix_kind_s == "linear" or mix_kind_s == "diis",
               "real_axis_qpgw: mix_kind must be \"linear\" or \"diis\" (got \"{}\")", mix_kind_s);
  utils::check(grid_kind_s == "uniform" or grid_kind_s == "nonuniform_log",
               "real_axis_qpgw: grid_kind must be \"uniform\" or \"nonuniform_log\" (got \"{}\")", grid_kind_s);

  // ---- Grid + state ------------------------------------------------------
  auto p = derive_grid_params(*mf, beta, N_w_p, N_Omega_p, N_t_p,
                              wmax_p, Omegamax_p);
  const double w_dense_eff = (w_dense_p > 0.0)
                              ? w_dense_p
                              : std::max(8.0 * eta_p, 0.4);
  const long   N_dense_eff = (N_dense_p > 0)
                              ? N_dense_p
                              : (((p.N_w / 2) | 1));
  auto grid = (grid_kind_s == "nonuniform_log")
              ? real_freq_grid_t::make_nonuniform_log(
                  p.beta, p.mu0, p.w_max, p.N_w,
                  w_dense_eff, N_dense_eff,
                  p.Omega_max, p.N_Omega, p.N_t, p.T_window)
              : real_freq_grid_t::make_uniform(
                  p.beta, p.mu0, p.w_max, p.N_w,
                  p.Omega_max, p.N_Omega, p.N_t, p.T_window);

  real_axis_mb_state_t state(grid);
  state.mpi = mpi;
  state.coqui_prefix = output_p;

  if (verbose and mpi->comm.root()) {
    app_log(1, "");
    app_log(1, "╔══════════════════════════════════════════════════════════╗");
    app_log(1, "║                Real-axis QP-SCF (THC) -- {}              ║",
            mode_s);
    app_log(1, "╚══════════════════════════════════════════════════════════╝");
    app_log(2, "  beta        = {}", p.beta);
    app_log(2, "  mu0         = {:.6f}  (gap midpoint)", p.mu0);
    app_log(2, "  w_max       = {:.4f}", p.w_max);
    app_log(2, "  Omega_max   = {:.4f}", p.Omega_max);
    app_log(2, "  N_w         = {}", p.N_w);
    app_log(2, "  N_Omega     = {}", p.N_Omega);
    app_log(2, "  N_t         = {}", p.N_t);
    app_log(2, "  eta (W,Sig) = {}", eta_p);
    app_log(2, "  qp_type     = {}", qp_type);
    app_log(2, "  off_diag    = {}", odm_s);
    app_log(2, "  niter       = {}", niter);
    app_log(2, "  div_treat   = {}", div_treat);
    app_log(2, "  grid_kind   = {}", grid_kind_s);
    if (grid_kind_s == "nonuniform_log") {
      app_log(2, "  w_dense     = {:.4f}", w_dense_eff);
      app_log(2, "  N_dense     = {}", N_dense_eff);
    }
    app_log(2, "  align_mo    = {}", align_mo_p ? "true" : "false");
    if (align_mo_p)
      app_log(2, "  dE_cluster  = {:.2e}", dE_cluster_p);
    app_log(2, "  tol_max_de  = {:.1e}", tol_max_de_p);
    app_log(2, "  tol_dDm     = {:.1e}", tol_dDm_p);
  }

  // ---- Build sH_0, sS, sFock from MF -------------------------------------
  // Real-axis stack stores orbital-basis arrays at IBZ k (matches imag-axis
  // pattern in scf_driver.cpp). Star expansion to the full BZ happens
  // inside the BZ-pair kernels via X(FBZ k) + MF.symmetry_rotation, never
  // at the storage layer.
  const long ns      = mf->nspin();
  const long Nk_ibz  = mf->nkpts_ibz();
  const long nbnd    = mf->nbnd();

  auto sH0_ibz = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(
      *mpi, std::array<long, 4>{ns, Nk_ibz, nbnd, nbnd});
  auto sS_ibz  = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(
      *mpi, std::array<long, 4>{ns, Nk_ibz, nbnd, nbnd});
  auto sF_ibz  = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(
      *mpi, std::array<long, 4>{ns, Nk_ibz, nbnd, nbnd});
  auto psp = hamilt::make_pseudopot(*mf);
  hamilt::set_H0  (*mf, psp.get(), sH0_ibz);
  hamilt::set_ovlp(*mf, sS_ibz);
  hamilt::set_fock(*mf, psp.get(), sF_ibz, /*exclude_H0=*/false);
  mpi->comm.barrier();

  // Pass IBZ-shaped one-body data to the QP-SCF loop.
  nda::array<cval_t, 4> H_0_skij (std::array<long, 4>{ns, Nk_ibz, nbnd, nbnd});
  nda::array<cval_t, 4> S_skij   (std::array<long, 4>{ns, Nk_ibz, nbnd, nbnd});
  nda::array<cval_t, 4> Fock_skij(std::array<long, 4>{ns, Nk_ibz, nbnd, nbnd});
  auto H0L = sH0_ibz.local();
  auto SL  = sS_ibz.local();
  auto FL  = sF_ibz.local();
  for (long s = 0; s < ns; ++s)
    for (long k = 0; k < Nk_ibz; ++k)
      for (long i = 0; i < nbnd; ++i)
        for (long j = 0; j < nbnd; ++j) {
          H_0_skij (s, k, i, j) = H0L(s, k, i, j);
          S_skij   (s, k, i, j) = SL (s, k, i, j);
          Fock_skij(s, k, i, j) = FL (s, k, i, j);
        }

  // Seed state.H_eff = KS Fock so iter-1 diagonalization reproduces KS
  // orbitals (the imag-axis qp_scf_loop convention; required for evGW
  // to be a clean G0W0-QP starting point).
  // Restart path: read final_iter + Heff from existing h5 instead.
  using sA4 = real_axis_mb_state_t::sArray_t<nda::array_view<ComplexType, 4>>;
  state.H_eff_skij.emplace(*mpi, std::array<long, 4>{ns, Nk_ibz, nbnd, nbnd});
  long init_it = 0;
  double mu_init = 0.0;
  if (write_chkpt_p and restart_p) {
    init_it = chkpt::read_qpscf(mpi->node_comm, state.H_eff_skij.value(),
                                 mu_init, state.coqui_prefix);
    if (verbose and mpi->comm.root())
      app_log(1, "real_axis_qpgw: restart from {} at iter {}",
              state.coqui_prefix + ".mbpt.h5", init_it);
  } else {
    if (state.H_eff_skij->node_comm()->root())
      state.H_eff_skij->local() = Fock_skij;
    state.H_eff_skij->node_sync();
  }

  // ---- Solver bundle -----------------------------------------------------
  methods::solvers::hf_t        hf(hf_div_t);
  real_axis_scr_coulomb_t       scr(&grid, screen_t, div_treat, eps_nufft);
  methods::solvers::real_axis_gw_t
                                gw(grid, /*max_iter*/ 1, /*mix*/ alpha_mix,
                                   /*eps_nufft*/ eps_nufft, /*ntrans*/ 1);
  real_axis_qp_context_t        qctx{qp_type, odm_s, qp_eta_p, qp_tol};
  real_axis_qp_solver_t         qp(&grid, qctx);
  real_axis_qp_mb_solver_t      mb_solver(&hf, &scr, &gw, &qp);

  qp_scgw_config cfg;
  cfg.max_iter    = niter;
  cfg.alpha_mix   = alpha_mix;
  cfg.conv_tol    = conv_thr;
  cfg.mix_kind    = (mix_kind_s == "diis") ? qp_mix_kind::diis
                                            : qp_mix_kind::linear;
  cfg.diis_window = diis_win;
  cfg.eta         = eta_p;
  cfg.eps_nufft   = eps_nufft;
  cfg.update_W    = update_W;
  cfg.verbose     = verbose;
  cfg.align_mo    = align_mo_p;
  cfg.dE_cluster_align = dE_cluster_p;
  cfg.tol_max_de  = tol_max_de_p;
  cfg.tol_dDm     = tol_dDm_p;

  // ---- BZ weights, electron count ---------------------------------------
  // k_weight is IBZ-sized with multiplicity-weighted entries that sum to 1.
  nda::array<double, 1> k_weights(Nk_ibz);
  auto kw = mf->k_weight();
  for (long k = 0; k < Nk_ibz; ++k) k_weights(k) = kw(k);
  const double N_elec    = mf->nelec();
  const long   ns_factor = (ns == 1 and mf->npol() == 1) ? 2 : 1;

  const qp_mode mode = (mode_s == "qsgw") ? qp_mode::qsgw : qp_mode::evgw;

  // Pre-loop checkpoint: write metadata + iter-0 dump (initial MO/E/Dm
  // canonicalized from the seeded H_eff). Skipped on restart (file already
  // has metadata; iter labels continue from final_iter+1). Mirrors imag-axis
  // qp_scf_loop pattern (scf_driver.cpp:285-294).
  if (write_chkpt_p and !restart_p) {
    // Allocate MO/E/Dm sArrays so the initial canonicalize can write into them.
    state.MO_skia.emplace(*mpi,
        std::array<long, 4>{ns, Nk_ibz, nbnd, nbnd});
    state.E_ska.emplace(*mpi,
        std::array<long, 3>{ns, Nk_ibz, nbnd});
    state.Dm_skij.emplace(*mpi,
        std::array<long, 4>{ns, Nk_ibz, nbnd, nbnd});
    auto& sHe = state.H_eff_skij.value();
    auto& sMO = state.MO_skia.value();
    auto& sE  = state.E_ska.value();
    auto& sDm = state.Dm_skij.value();
    if (sMO.node_comm()->root()) {
      detail_qp::diagonalize_H_eff(sHe.local(), S_skij, sE.local(), sMO.local());
    }
    sMO.node_sync();
    sE.node_sync();
    mu_init = detail_qp::find_mu_from_QP(sE.local(), k_weights,
                                          grid.beta(), N_elec, ns_factor);
    if (sDm.node_comm()->root()) {
      detail_qp::update_Dm_from_QP(sE.local(), sMO.local(), mu_init,
                                    grid.beta(), sDm.local());
    }
    sDm.node_sync();

    chkpt::write_metadata_real_axis(mpi->comm, *mf, grid, sH0_ibz, sS_ibz,
                                     state.coqui_prefix);
    chkpt::dump_scf(mpi->comm, /*iter*/ 0, sDm, sHe, sMO, sE, mu_init,
                    state.coqui_prefix);
  }

  auto res = real_axis_qp_scf_loop(state, H_0_skij, S_skij, thc,
                                   mb_solver, mode, cfg,
                                   k_weights, N_elec, ns_factor,
                                   /*init_it*/ init_it,
                                   /*write_chkpt*/ write_chkpt_p);

  if (mpi->comm.root()) {
    app_log(1, "");
    app_log(1, "real_axis_qpgw[{}]: iter_used={}  ||dH_eff||={:.3e}  mu={:.6f}  converged={}",
            mode_s, res.iter_used, res.final_diff, res.final_mu,
            res.converged ? "yes" : "no");

    // Indirect QP gap from state.E_ska (E_ska are absolute energies, NOT
    // shifted by mu). Dump per-k HOMO/LUMO + the indirect gap.
    if (state.E_ska.has_value()) {
      auto E = state.E_ska->local();
      const long n_homo = static_cast<long>(mf->nelec() / 2 - 1);
      const long n_lumo = std::min<long>(n_homo + 1, nbnd - 1);
      double max_homo = -1e30;
      double min_lumo =  1e30;
      long   k_homo = 0, k_lumo = 0;
      std::ofstream of("E_QP.dat");
      of << "# s  k  HOMO  LUMO\n";
      for (long s = 0; s < ns; ++s) {
        for (long k = 0; k < Nk_ibz; ++k) {
          const double e_h = E(s, k, n_homo).real();
          const double e_l = E(s, k, n_lumo).real();
          of << s << "  " << k << "  " << e_h << "  " << e_l << "\n";
          if (e_h > max_homo) { max_homo = e_h; k_homo = k; }
          if (e_l < min_lumo) { min_lumo = e_l; k_lumo = k; }
        }
      }
      of.close();
      app_log(1, "real_axis_qpgw[{}]: indirect gap : HOMO(k={}) = {:.4f} Ha, "
                  "LUMO(k={}) = {:.4f} Ha, gap = {:.4f} Ha = {:.4f} eV",
              mode_s, k_homo, max_homo, k_lumo, min_lumo,
              min_lumo - max_homo,
              (min_lumo - max_homo) * 27.211386245988);
    }
  }
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_MBPT_DISPATCH_HPP
