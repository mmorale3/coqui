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
#include "methods/tools/chkpt_utils.h"

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

namespace detail_scgw {
  /**
   * Compute orbital-basis density matrix from the spectral function via
   * the sum-rule integral
   *     Dm_{ij}(s, k) = sum_w  w_weights(w) * f(w - mu_chem) * A_{ij}(w, s, k)
   * where w-grid and weights come from the real_freq_grid_t and f is the
   * Fermi-Dirac factor. A_wskij is stored componentwise (-(i/π) G^R_ij), so
   * this integral on the real axis recovers the standard definition.
   * Writes on node-root then node_sync.
   */
  template<typename sArray5_t, typename sArray4_t>
  inline void compute_Dm_from_A(real_freq_grid_t const& grid,
                                sArray5_t        const& sA_wskij,
                                double                  mu_chem,
                                sArray4_t            && sDm_skij)
  {
    if (sDm_skij.node_comm()->root()) {
      auto A   = sA_wskij.local();
      auto Dm  = sDm_skij.local();
      const long N_w  = A.shape()[0];
      const long ns   = A.shape()[1];
      const long Nk   = A.shape()[2];
      const long nbnd = A.shape()[3];
      Dm = ComplexType(0.0, 0.0);
      auto w_grid = grid.w();
      auto w_wts  = grid.w_weights();
      const double beta = grid.beta();
      for (long iw = 0; iw < N_w; ++iw) {
        const double f  = real_freq_grid_t::fermi(w_grid(iw), mu_chem, beta);
        const double cf = f * w_wts(iw);
        for (long s = 0; s < ns; ++s)
          for (long k = 0; k < Nk; ++k)
            for (long i = 0; i < nbnd; ++i)
              for (long j = 0; j < nbnd; ++j)
                Dm(s, k, i, j) += ComplexType(cf, 0.0) * A(iw, s, k, i, j);
      }
    }
    sDm_skij.node_sync();
  }
} // namespace detail_scgw

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
                                       double N_elec,
                                       long  init_it    = 0,
                                       bool  write_chkpt = false,
                                       long  chkpt_freq = 1)
{
  // The state lives on host (sArrays / dArrays<HOST_MEMORY>) so MEM is a
  // template marker for this driver. The underlying solver classes
  // (real_axis_dyson_t, real_axis_scr_coulomb_t, real_axis_gw_t,
  // real_axis_hf_t) all have host bodies; their MEM template parameters
  // are no-ops or device-delegating wrappers. Phase-2 device acceleration
  // would push specific kernels (Pi cross-correlation, Hilbert) onto the
  // device via the already-MEM-aware accumulate_ImPi_one_kq /
  // RePi_from_ImPi / primary_to_aux_one_k routines.
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

  // Orbital-basis arrays (A, Sigma, etc.) live at IBZ k. Star expansion to
  // FBZ happens inside BZ-pair kernels via X(FBZ k). H_MF is IBZ-shaped.
  utils::check(static_cast<long>(MF.nspin()) == ns and
               static_cast<long>(MF.nkpts_ibz()) == Nk and
               static_cast<long>(MF.nbnd()) == nbnd,
               "real_axis_scf_loop: H_MF shape disagrees with MF accessors "
               "(expected IBZ k)");
  utils::check(k_weights.shape()[0] == Nk,
               "real_axis_scf_loop: k_weights size mismatch (must be IBZ)");

  // ---- Initial A: Lorentzian per H_MF diagonal if state.A_wskij is empty.
  bool A_is_zero = true;
  if (state.A_wskij.has_value()) {
    auto A_loc = state.A_wskij->local();
    auto const* d = A_loc.data();
    for (long i = 0; i < A_loc.size(); ++i)
      if (std::abs(d[i]) > 0.0) { A_is_zero = false; break; }
  }
  if (!state.A_wskij.has_value() or A_is_zero) {
    if (!state.A_wskij.has_value()) {
      state.A_wskij.emplace(*state.mpi,
          std::array<long, 5>{N_w, ns, Nk, nbnd, nbnd});
    }
    if (state.A_wskij->node_comm()->root()) {
      auto A = state.A_wskij->local();
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
    state.A_wskij->node_sync();
  }

  // Sigma_x storage on state (read by dyson.solve_dyson). Allocate so even
  // a HF-less SCF (mb_solver.hf == nullptr) can pass it to dyson.
  if (!state.Sigma_x_skij.has_value())
    state.Sigma_x_skij.emplace(*state.mpi,
        std::array<long, 4>{ns, Nk, nbnd, nbnd});
  if (state.Sigma_x_skij->node_comm()->root())
    state.Sigma_x_skij->local() = ComplexType(0.0, 0.0);
  state.Sigma_x_skij->node_sync();

  // Density matrix storage on state. Populated each iter (sum-rule integral
  // on A) so a per-iter chkpt dump can record it. Cheap (single ω quadrature
  // on a 5D nbnd² array) compared to the GW work.
  if (!state.Dm_skij.has_value())
    state.Dm_skij.emplace(*state.mpi,
        std::array<long, 4>{ns, Nk, nbnd, nbnd});

  // Use R-space if the THC fixture is periodic (Nk > 1).
  const bool use_rspace = (Nk > 1);

  // DIIS mixer (always allocated; consulted only when mix_kind == diis).
  diis_mixer_t diis(state.mpi, cfg.diis_window);

  // Pre-allocate per-iteration scratch as sArrays (one copy per node).
  // Reused across all iterations.
  std::array<long, 5> A_shape =
      {N_w, ns, Nk, nbnd, nbnd};
  using sArray5 = real_axis_mb_state_t::sArray_t<nda::array_view<ComplexType, 5>>;
  sArray5 A_old_shm(*state.mpi, A_shape);
  sArray5 A_new_shm(*state.mpi, A_shape);

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
                                cfg.verbose, use_rspace);

    // P5* lite (memory redesign): free intermediate bosonic state arrays
    // not needed downstream. evaluate (Sigma_c) and hf::evaluate (Sigma_x)
    // read only state.ImW_qPQO; ImPi/RePi/ReW were scratch for the Dyson
    // solve + eps_inv_head and can be released. The next SCF iteration's
    // update_w reallocates them.
    state.free_intermediate_bosonic();

    // ---- 2. Sigma^c (gw) ----
    // Forward the divergence treatment from scr_coulomb so the head
    // correction in gw_t::evaluate sees the same setting that produced
    // state.eps_inv_head_O. Thread MEM through so MEM=DEVICE_MEMORY
    // routes the GW work to device.
    mb_solver.gw->template evaluate<MEM>(state, thc, cfg.eps_nufft,
                                          mb_solver.scr_eri->div_treatment(),
                                          cfg.verbose, use_rspace);

    // Causality projection on Im Sigma_c (skwij layout). Write only on
    // node-root, then node_sync.
    {
      auto ImS = state.ImSigma_wskij->local();
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
      if (state.ImSigma_wskij->node_comm()->root()) {
        for (long s = 0; s < ns; ++s)
          for (long k = 0; k < Nk; ++k)
            for (long iw = 0; iw < N_w; ++iw)
              for (long mu = 0; mu < nbnd; ++mu)
                for (long nu = 0; nu < nbnd; ++nu)
                  ImS(iw, s, k, mu, nu) = ImS_skwij(s, k, iw, mu, nu);
      }
      state.ImSigma_wskij->node_sync();
    }

    // ---- 3. Sigma^x (HF) ----
    if (mb_solver.hf != nullptr)
      mb_solver.hf->evaluate(state, thc, mu_cur);
    // (else: state.Sigma_x_skij stays zero; correlation-only SCF.)

    // ---- 4. Snapshot A^(k) into A_old_shm; Dyson updates state.A_wskij in
    //         place to A_full = Dyson(Sigma[A^(k)]). Both are sArrays.
    if (A_old_shm.node_comm()->root())
      A_old_shm.local() = state.A_wskij->local();
    A_old_shm.node_sync();
    dyson.solve_dyson(state, mu_cur);
    auto& A_full_shm = *state.A_wskij;

    // ---- 5. Mix(A_old_shm, A_full_shm) -> A_new_shm. Then copy back
    //         to state.A_wskij. ||dA||_F = sqrt(<R, R>) where R = A_full - A_old.
    double diff = 0.0;
    if (cfg.mix_kind == scgw_mix_kind::diis) {
      diff = diis.mix(A_old_shm, A_full_shm, cfg.alpha_mix, A_new_shm);
    } else {
      // Linear mix on node root.
      const double a = cfg.alpha_mix;
      if (A_new_shm.node_comm()->root()) {
        auto An = A_new_shm.local();
        auto Ao = A_old_shm.local();
        auto Af = A_full_shm.local();
        const long N = An.size();
        auto * dn = An.data();
        auto const* da = Ao.data();
        auto const* df = Af.data();
        for (long i = 0; i < N; ++i)
          dn[i] = (1.0 - a) * da[i] + a * df[i];
      }
      A_new_shm.node_sync();
      // Compute ||A_full - A_old||_F using shared-memory reads.
      auto Ao = A_old_shm.local();
      auto Af = A_full_shm.local();
      double acc = 0.0;
      const long N = Ao.size();
      auto const* da = Ao.data();
      auto const* df = Af.data();
      for (long i = 0; i < N; ++i) acc += std::norm(df[i] - da[i]);
      diff = std::sqrt(acc);
    }
    if (state.A_wskij->node_comm()->root())
      state.A_wskij->local() = A_new_shm.local();
    state.A_wskij->node_sync();

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
    const bool stop = (diff < cfg.tol);

    // Per-iter checkpoint: populate Dm from A, then write {Dm, A, Sigma_x,
    // ImSigma, ReSigma, mu} to the SCF h5 file. Iter label is offset by
    // init_it (file-derived for restart). Throttled by chkpt_freq; the
    // final iter (stop=true) is always dumped if write_chkpt is on so the
    // converged state is persisted.
    if (write_chkpt) {
      const long it_label = init_it + it + 1;
      const bool dump_now = stop or (chkpt_freq > 0 and (it_label % chkpt_freq == 0));
      if (dump_now) {
        detail_scgw::compute_Dm_from_A(grid_in, state.A_wskij.value(),
                                        mu_cur, state.Dm_skij.value());
        chkpt::dump_scf_real_axis(
            comm, it_label,
            state.Dm_skij.value(), state.A_wskij.value(),
            state.Sigma_x_skij.value(),
            state.ImSigma_wskij.value(), state.ReSigma_wskij.value(),
            mu_cur, state.coqui_prefix);
      }
    }

    if (stop) { res.converged = true; break; }
  }

  return res;
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_SCF_DRIVER_HPP
