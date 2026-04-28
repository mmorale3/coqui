/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 *
 * Real-axis SCF driver: orchestrates {scr_coulomb, gw, sigma_x, dyson} per
 * iteration. Mirrors `methods::scf_loop` (imag-axis side) in shape:
 *
 *     real_axis_mb_solver_t mb_solver{ &scr_eri, &gw };
 *     real_axis_scf_loop(state, dyson, thc, mb_solver, cfg,
 *                        k_weights, N_elec);   // comm is read from state.mpi
 *
 * Once the real-axis HF class is split off from `evaluate_Sigma_x_serial`
 * the mb_solver_t struct will gain an hf member; for now the SCF loop calls
 * the free function directly.
 */

#ifndef COQUI_REAL_AXIS_SCF_DRIVER_HPP
#define COQUI_REAL_AXIS_SCF_DRIVER_HPP

#include <complex>
#include <iostream>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"

#include "methods/ERI/detail/concepts.hpp"
#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_mb_state.hpp"
#include "methods/GW_real_axis/real_axis_scr_coulomb_t.h"
#include "methods/GW_real_axis/real_axis_gw_t.h"
#include "methods/GW_real_axis/real_axis_hf_t.h"
#include "methods/GW_real_axis/real_axis_dyson_t.h"
#include "methods/GW_real_axis/real_axis_sigma_x.hpp"
#include "methods/GW_real_axis/real_axis_dyson_G.hpp"
#include "methods/GW_real_axis/real_axis_diis.hpp"
#include "methods/GW_real_axis/real_axis_scf.hpp"

namespace methods {
namespace real_axis {

/**
 * Solver bundle for the real-axis SCF loop. Mirrors
 * methods::solvers::mb_solver_t (the imag-axis equivalent).
 *
 * Pointers are non-owning; the caller maintains object lifetime.
 *
 * `hf` may be null if the SCF loop is run without a static-exchange piece
 * (e.g. pure correlation-only G_0W_0); in that case state.Sigma_x_skij is
 * left at zero.
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
struct real_axis_mb_solver_base_t {
  real_axis_hf_base_t<MEM>*          hf      = nullptr;
  real_axis_scr_coulomb_base_t<MEM>* scr_eri = nullptr;
  methods::solvers::real_axis_gw_t*  gw      = nullptr;

  real_axis_mb_solver_base_t() = default;
  real_axis_mb_solver_base_t(real_axis_scr_coulomb_base_t<MEM>* s,
                             methods::solvers::real_axis_gw_t*  g)
    : hf(nullptr), scr_eri(s), gw(g) {}
  real_axis_mb_solver_base_t(real_axis_hf_base_t<MEM>*          h,
                             real_axis_scr_coulomb_base_t<MEM>* s,
                             methods::solvers::real_axis_gw_t*  g)
    : hf(h), scr_eri(s), gw(g) {}
};

using real_axis_mb_solver_t = real_axis_mb_solver_base_t<HOST_MEMORY>;

/**
 * Real-axis self-consistent GW driver. Mirrors `evaluate(MBState, eri)`
 * dispatch + `update_G` Dyson update of `methods::scf_loop` (imag-axis).
 *
 * Inputs:
 *   state       reads/writes A_wskij, Sigma_x_skij, {Im,Re}Sigma_wskij.
 *               The MPI communicator is read from state.mpi->comm.
 *               If state.A_wskij is unset/zero the loop builds an initial
 *               Lorentzian A from H_MF (taken from `dyson.H_MF()`).
 *   dyson       supplies H_MF, eta, mu_tol; wraps the Dyson + mu-update.
 *   thc         THC ERI object.
 *   mb_solver   {scr_eri, gw} pointers.
 *   cfg         iteration / mixing config (scgw_config from real_axis_scf.hpp).
 *   k_weights   (Nk,) BZ weights used for the mu bisection N_elec integral.
 *   N_elec      target electron count for the mu update.
 *
 * Returns: scgw_result with iter_used, final_diff, final_mu, converged.
 */
template<MEMORY_SPACE MEM = HOST_MEMORY,
         methods::THC_ERI THC_t>
inline scgw_result real_axis_scf_loop(real_axis_mb_state_t& state,
                                       real_axis_dyson_base_t<MEM>& dyson,
                                       THC_t const& thc,
                                       real_axis_mb_solver_base_t<MEM> mb_solver,
                                       scgw_config const& cfg,
                                       nda::array<double, 1> const& k_weights,
                                       double N_elec)
{
  static_assert(MEM == HOST_MEMORY,
                "real_axis_scf_loop<DEVICE>: device path not yet supported "
                "in the underlying solver classes / Dyson update / mixer.");
  utils::check(state.grid != nullptr,
               "real_axis_scf_loop: state.grid not bound");
  utils::check(state.grid == &dyson.grid(),
               "real_axis_scf_loop: state.grid disagrees with dyson.grid()");
  utils::check(state.mpi != nullptr,
               "real_axis_scf_loop: state.mpi not bound");
  utils::check(mb_solver.scr_eri != nullptr,
               "real_axis_scf_loop: mb_solver.scr_eri must not be null");
  utils::check(mb_solver.gw != nullptr,
               "real_axis_scf_loop: mb_solver.gw must not be null");
  auto& comm = state.mpi->comm;

  using nda::range;
  const auto _ = range::all;

  auto const& grid_in = *state.grid;
  auto const& MF      = *thc.MF();
  auto const& H_MF    = dyson.H_MF();

  const long ns   = H_MF.shape()[0];
  const long Nk   = H_MF.shape()[1];
  const long nbnd = H_MF.shape()[2];
  const long N_w  = grid_in.N_w();

  utils::check(static_cast<long>(MF.nspin()) == ns and
               static_cast<long>(MF.nkpts()) == Nk and
               static_cast<long>(MF.nbnd()) == nbnd,
               "real_axis_scf_loop: H_MF shape disagrees with MF accessors");
  utils::check(k_weights.shape()[0] == Nk,
               "real_axis_scf_loop: k_weights size mismatch");

  // ---- Initial A: Lorentzian per H_MF diagonal if state.A_wskij is empty.
  bool A_is_zero = true;
  if (state.A_wskij.has_value()) {
    auto const* d = state.A_wskij->data();
    for (long i = 0; i < state.A_wskij->size(); ++i)
      if (std::abs(d[i]) > 0.0) { A_is_zero = false; break; }
  }
  if (!state.A_wskij.has_value() or A_is_zero) {
    state.A_wskij = nda::array<ComplexType, 5>(N_w, ns, Nk, nbnd, nbnd);
    auto& A = *state.A_wskij;
    A = ComplexType(0.0, 0.0);
    const double eta_init = std::max(cfg.eta, 1e-2);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k)
        for (long m = 0; m < nbnd; ++m) {
          const double e = H_MF(s, k, m, m).real();
          for (long iw = 0; iw < N_w; ++iw) {
            const double wl = grid_in.w()(iw) + grid_in.mu_chem();
            const double v = (1.0 / M_PI) * eta_init
                           / ((wl - e)*(wl - e) + eta_init*eta_init);
            A(iw, s, k, m, m) = ComplexType(v, 0.0);
          }
        }
  }

  // Sigma_x storage on state (read by dyson.solve_dyson). Allocate so even
  // a HF-less SCF (mb_solver.hf == nullptr) can pass it to dyson.
  if (!state.Sigma_x_skij.has_value())
    state.Sigma_x_skij = nda::array<ComplexType, 4>(ns, Nk, nbnd, nbnd);
  *state.Sigma_x_skij = ComplexType(0.0, 0.0);

  // Use R-space if the THC fixture is periodic (Nk > 1).
  const bool use_rspace = (Nk > 1);

  // DIIS mixer (always allocated; consulted only when mix_kind == diis).
  diis_mixer_t diis(cfg.diis_window);

  // Working mu (rebuilt grid is owned by dyson.solve_dyson).
  double mu_cur = grid_in.mu_chem();

  scgw_result res;
  res.iter_used  = 0;
  res.final_diff = 0.0;
  res.final_mu   = mu_cur;
  res.converged  = false;

  for (long it = 0; it < cfg.max_iter; ++it) {
    // ---- 1. update W (scr_coulomb) ----
    mb_solver.scr_eri->update_w(state, thc,
                                /*verbose*/ false, use_rspace);

    // ---- 2. Sigma^c (gw) ----
    mb_solver.gw->evaluate(state, thc, cfg.eps_nufft,
                           "ignore_g0", /*verbose*/ false, use_rspace);

    // Causality projection on Im Sigma_c (skwij layout).
    {
      auto& ImS = *state.ImSigma_wskij;
      // Repack to (ns, Nk, N_w, nbnd, nbnd) for project_causality_ImSigma,
      // then back. Cheap relative to the kernel cost.
      nda::array<ComplexType, 5> ImS_skwij(ns, Nk, N_w, nbnd, nbnd);
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k)
          for (long iw = 0; iw < N_w; ++iw)
            for (long mu = 0; mu < nbnd; ++mu)
              for (long nu = 0; nu < nbnd; ++nu)
                ImS_skwij(s, k, iw, mu, nu) = ImS(iw, s, k, mu, nu);
      project_causality_ImSigma(ImS_skwij);
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k)
          for (long iw = 0; iw < N_w; ++iw)
            for (long mu = 0; mu < nbnd; ++mu)
              for (long nu = 0; nu < nbnd; ++nu)
                ImS(iw, s, k, mu, nu) = ImS_skwij(s, k, iw, mu, nu);
    }

    // ---- 3. Sigma^x (HF) ----
    if (mb_solver.hf != nullptr)
      mb_solver.hf->evaluate(state, thc, mu_cur);
    // (else: state.Sigma_x_skij stays zero; correlation-only SCF.)

    // ---- 4. Dyson update -> A_full (scratch) ----
    nda::array<ComplexType, 5> A_old = *state.A_wskij;
    dyson.solve_dyson(state, mu_cur);
    auto A_full = *state.A_wskij;

    // ---- 5. Mix A_old <- (A_old, A_full). ----
    const double diff = frobenius_diff(A_old, A_full);
    if (cfg.mix_kind == scgw_mix_kind::diis) {
      nda::array<ComplexType, 5> A_next(A_old.shape());
      diis.mix(A_old, A_full, cfg.alpha_mix, A_next);
      *state.A_wskij = A_next;
    } else {
      const double a = cfg.alpha_mix;
      *state.A_wskij = nda::map([a](ComplexType old_v, ComplexType new_v) {
        return (1.0 - a) * old_v + a * new_v;
      })(A_old, A_full);
    }

    // ---- 6. mu update ----
    if (cfg.update_mu) {
      mu_cur = dyson.find_mu_chem(state, k_weights, N_elec);
      state.mu_chem = mu_cur;
    }

    if (cfg.verbose and comm.root()) {
      app_log(2, "[real_axis_scf_loop] iter={}  ||dA||={:.3e}  mu={:.6f}",
              it + 1, diff, mu_cur);
    }

    res.iter_used  = it + 1;
    res.final_diff = diff;
    res.final_mu   = mu_cur;
    if (diff < cfg.tol) { res.converged = true; break; }
  }

  return res;
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_SCF_DRIVER_HPP
