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
  io::tolower(mix_kind_s);

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

  // ---- Grid + state -------------------------------------------------------
  auto p = derive_grid_params(*mf, beta, N_w_p, N_Omega_p, N_t_p,
                              wmax_p, Omegamax_p);
  auto grid = real_freq_grid_t::make_uniform(
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
  }

  // ---- Build H_MF = diag(eps_KS) ----------------------------------------
  const long ns   = mf->nspin();
  const long Nk   = mf->nkpts();
  const long nbnd = mf->nbnd();
  auto eigval = mf->eigval();
  auto kp2ibz = mf->kp_to_ibz();
  nda::array<cval_t, 4> H_MF(ns, Nk, nbnd, nbnd);
  H_MF = cval_t(0.0, 0.0);
  for (long s = 0; s < ns; ++s)
    for (long k = 0; k < Nk; ++k) {
      const long kibz = kp2ibz(k);
      for (long n = 0; n < nbnd; ++n)
        H_MF(s, k, n, n) = cval_t(eigval(s, kibz, n), 0.0);
    }

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
  nda::array<double, 1> k_weights(Nk);
  for (long ik = 0; ik < Nk; ++ik)
    k_weights(ik) = 1.0 / static_cast<double>(Nk);
  const double N_elec = static_cast<double>(mf->nelec());

  // ---- Run SCF ----------------------------------------------------------
  auto res = real_axis_scf_loop(state, dyson, thc, mb_solver, cfg,
                                k_weights, N_elec);

  if (mpi->comm.root()) {
    app_log(1, "");
    app_log(1, "real_axis_gw: iter_used={}  ||dA||={:.3e}  mu={:.6f}  converged={}",
            res.iter_used, res.final_diff, res.final_mu,
            res.converged ? "yes" : "no");
  }
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_MBPT_DISPATCH_HPP
