/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_QP_SCF_DRIVER_HPP
#define COQUI_REAL_AXIS_QP_SCF_DRIVER_HPP

#include <cmath>
#include <complex>
#include <memory>
#include <string>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/linalg.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"

#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_mb_state.hpp"
#include "methods/GW_real_axis/real_axis_diis.hpp"
#include "methods/GW_real_axis/real_axis_dyson_G.hpp"   // project_causality_ImSigma
#include "methods/GW_real_axis/real_axis_qp_context.hpp"
#include "methods/GW_real_axis/real_axis_qp_solver_t.h"
#include "methods/GW_real_axis/real_axis_scr_coulomb_t.h"
#include "methods/GW_real_axis/real_axis_gw_t.h"
#include "methods/HF/hf_t.h"

namespace methods {
namespace real_axis {

/**
 * Quasiparticle self-consistency mode.
 *
 *   evGW  : eigenvalue self-consistent GW. Orbitals (sMO) are held fixed
 *           at the input H_0 KS basis; only the eigenvalues E_ska iterate.
 *           Each iter updates sH_eff <- sMO * diag(eps_QP) * sMO^dagger.
 *   qsgw  : full Faleev-Schilfgaarde-Kotani QSGW. Both orbitals AND
 *           eigenvalues iterate via diagonalization of
 *               sH_eff = sH_0 + sSigma_x + sV_corr
 *           where sV_corr is the static, hermitized Faleev correction.
 */
enum class qp_mode { evgw, qsgw };

/**
 * Mixing flavor for the QP-SCF outer loop.
 */
enum class qp_mix_kind { linear, diis };

struct qp_scgw_config {
  long          max_iter      = 20;
  double        alpha_mix     = 0.7;
  double        conv_tol      = 1e-4;
  qp_mix_kind   mix_kind      = qp_mix_kind::diis;
  long          diis_window   = 8;
  double        eta           = 0.05;     // Lorentzian width for QP A.
  double        eps_nufft     = 1e-8;
  bool          update_W      = true;     // false freezes W = W^(0).
  bool          verbose       = false;
};

struct qp_scgw_result {
  long   iter_used  = 0;
  double final_diff = 0.0;
  double final_mu   = 0.0;
  bool   converged  = false;
};

/**
 * Solver bundle for the QP-SCF loop. Pointers to:
 *   hf      : production HF solver (`methods::HF::hf_t`). Computes
 *             V_H + Sigma_x from the density matrix Dm at each iter --
 *             the same convention as the imag-axis qp_scf_loop. The
 *             real-axis-specific `real_axis_hf_t` (which integrates
 *             A_wskij) is NOT used here: QSGW wants Dm-based exchange.
 *   scr_eri : real-axis screened-Coulomb solver (W from A).
 *   gw      : real-axis GW solver (Sigma_c from A and W).
 *   qp      : real-axis quasiparticle solver (V_corr from Sigma_c).
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
struct real_axis_qp_mb_solver_base_t {
  methods::solvers::hf_t*                  hf      = nullptr;
  real_axis_scr_coulomb_base_t<MEM>*  scr_eri = nullptr;
  methods::solvers::real_axis_gw_t*   gw      = nullptr;
  real_axis_qp_solver_base_t<MEM>*    qp      = nullptr;

  real_axis_qp_mb_solver_base_t() = default;
  real_axis_qp_mb_solver_base_t(methods::solvers::hf_t*                 h,
                                real_axis_scr_coulomb_base_t<MEM>* s,
                                methods::solvers::real_axis_gw_t*  g,
                                real_axis_qp_solver_base_t<MEM>*   q)
    : hf(h), scr_eri(s), gw(g), qp(q) {}
};

using real_axis_qp_mb_solver_t = real_axis_qp_mb_solver_base_t<HOST_MEMORY>;

namespace detail_qp {

  /// Diagonalize hermitian H per (s, k); write eigenvalues to E and
  /// eigenvectors (column-major) to MO. Each rank does its share of (s, k)
  /// and a final allreduce makes everything visible. Operates on local()
  /// views directly (assumes per-rank fully-replicated H).
  inline void diagonalize_H_eff(nda::ArrayOfRank<4> auto const& H_skij,
                                nda::ArrayOfRank<4> auto const& S_skij,
                                nda::ArrayOfRank<3> auto      && E_ska,
                                nda::ArrayOfRank<4> auto      && MO_skia)
  {
    const long ns   = H_skij.shape()[0];
    const long Nk   = H_skij.shape()[1];
    const long nbnd = H_skij.shape()[2];
    nda::matrix<ComplexType> Hsk(nbnd, nbnd), Ssk(nbnd, nbnd);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k) {
        for (long i = 0; i < nbnd; ++i)
          for (long j = 0; j < nbnd; ++j) {
            Hsk(i, j) = H_skij(s, k, i, j);
            Ssk(i, j) = S_skij(s, k, i, j);
          }
        auto [evals, evecs] = nda::linalg::eigenelements(Hsk, Ssk);
        for (long n = 0; n < nbnd; ++n) E_ska(s, k, n) = ComplexType(evals(n), 0.0);
        for (long i = 0; i < nbnd; ++i)
          for (long n = 0; n < nbnd; ++n)
            MO_skia(s, k, i, n) = evecs(i, n);
      }
  }

  /// Find mu such that
  ///     sum_{s,k,n} k_weight(k) * f(eps_n, mu, beta) [* 2 if (ns==1, npol==1)]
  ///                = N_elec
  /// Bisection on [eps_min - 5, eps_max + 5].
  inline double find_mu_from_QP(nda::ArrayOfRank<3> auto const& E_ska,
                                nda::array<double, 1> const& k_weights,
                                double beta, double N_elec, long ns_factor,
                                double tol = 1e-9, long max_iter = 200)
  {
    const long ns = E_ska.shape()[0];
    const long Nk = E_ska.shape()[1];
    const long nb = E_ska.shape()[2];
    auto N_of_mu = [&](double mu) {
      double n = 0.0;
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k) {
          double sum = 0.0;
          for (long a = 0; a < nb; ++a) {
            const double e = E_ska(s, k, a).real();
            const double x = beta * (e - mu);
            const double f = (x >= 0) ? std::exp(-x) / (1.0 + std::exp(-x))
                                      : 1.0 / (1.0 + std::exp(x));
            sum += f;
          }
          n += k_weights(k) * sum;
        }
      return ns_factor * n;
    };
    double e_min =  std::numeric_limits<double>::infinity();
    double e_max = -std::numeric_limits<double>::infinity();
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k)
        for (long a = 0; a < nb; ++a) {
          const double e = E_ska(s, k, a).real();
          e_min = std::min(e_min, e);
          e_max = std::max(e_max, e);
        }
    double mu_lo = e_min - 5.0;
    double mu_hi = e_max + 5.0;
    double f_lo = N_of_mu(mu_lo) - N_elec;
    double f_hi = N_of_mu(mu_hi) - N_elec;
    utils::check(f_lo * f_hi <= 0.0,
                 "find_mu_from_QP: target N_elec={} not bracketed in "
                 "[{} -> N={}, {} -> N={}]",
                 N_elec, mu_lo, f_lo + N_elec, mu_hi, f_hi + N_elec);
    for (long it = 0; it < max_iter; ++it) {
      const double mu_mid = 0.5 * (mu_lo + mu_hi);
      const double f_mid  = N_of_mu(mu_mid) - N_elec;
      if (std::abs(f_mid) < tol) return mu_mid;
      if (f_lo * f_mid <= 0.0) { mu_hi = mu_mid; f_hi = f_mid; }
      else                     { mu_lo = mu_mid; f_lo = f_mid; }
    }
    return 0.5 * (mu_lo + mu_hi);
  }

  /// Update the density matrix from MO/E using Fermi-Dirac occupation:
  ///     Dm_ij(s, k) = sum_n MO(i, n) * f(eps_n - mu) * conj(MO(j, n))
  /// Mirrors `methods::update_Dm` from the imag-axis side. Distributed
  /// over (s, k) and written on node-root then synced.
  inline void update_Dm_from_QP(nda::ArrayOfRank<3> auto  const& E_ska,
                                nda::ArrayOfRank<4> auto  const& MO_skia,
                                double                           mu,
                                double                           beta,
                                nda::ArrayOfRank<4> auto      && Dm_skij)
  {
    const long ns   = E_ska.shape()[0];
    const long Nk   = E_ska.shape()[1];
    const long nbnd = E_ska.shape()[2];
    nda::array<double, 1> occ(nbnd);
    nda::matrix<ComplexType> Cdag(nbnd, nbnd);
    nda::matrix<ComplexType> Csk (nbnd, nbnd);
    nda::matrix<ComplexType> Dsk (nbnd, nbnd);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k) {
        for (long n = 0; n < nbnd; ++n) {
          const double x = beta * (E_ska(s, k, n).real() - mu);
          occ(n) = (x >= 0) ? std::exp(-x) / (1.0 + std::exp(-x))
                            : 1.0 / (1.0 + std::exp(x));
        }
        for (long i = 0; i < nbnd; ++i)
          for (long n = 0; n < nbnd; ++n)
            Csk(i, n) = MO_skia(s, k, i, n);
        // Cdag(n, j) = occ(n) * conj(MO(j, n))
        for (long n = 0; n < nbnd; ++n)
          for (long j = 0; j < nbnd; ++j)
            Cdag(n, j) = occ(n) * std::conj(MO_skia(s, k, j, n));
        nda::blas::gemm(Csk, Cdag, Dsk);
        for (long i = 0; i < nbnd; ++i)
          for (long j = 0; j < nbnd; ++j)
            Dm_skij(s, k, i, j) = Dsk(i, j);
      }
  }

  /// Build state.A_wskij as the QP spectral function: a sum of Lorentzians
  /// at QP energies, projected through the current MO orbitals.
  ///
  ///     A_{ij}(s, k, w_abs) = (1/pi) * eta
  ///                         * sum_n MO(i, n) * conj(MO(j, n))
  ///                                / [(w_abs - eps_n)^2 + eta^2]
  ///
  /// w_abs = grid.w()(iw) + grid.mu_chem() (relative-grid convention).
  /// Writes on node-root then node_sync. MO and E come from the just-
  /// diagonalized state.MO_skia / state.E_ska.
  inline void build_A_from_QP_poles(real_freq_grid_t          const& grid,
                                    nda::ArrayOfRank<3> auto  const& E_ska,
                                    nda::ArrayOfRank<4> auto  const& MO_skia,
                                    double                           eta,
                                    real_axis_mb_state_t::sArray_t<
                                        nda::array_view<ComplexType, 5>>& sA)
  {
    const long ns   = E_ska.shape()[0];
    const long Nk   = E_ska.shape()[1];
    const long nbnd = E_ska.shape()[2];
    const long N_w  = grid.N_w();
    if (sA.node_comm()->root()) {
      auto A = sA.local();
      A = ComplexType(0.0, 0.0);
      const double mu = grid.mu_chem();
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k)
          for (long iw = 0; iw < N_w; ++iw) {
            const double w_abs = grid.w()(iw) + mu;
            for (long n = 0; n < nbnd; ++n) {
              const double e_n   = E_ska(s, k, n).real();
              const double denom = (w_abs - e_n) * (w_abs - e_n) + eta * eta;
              const double w_n   = (1.0 / M_PI) * eta / denom;
              for (long i = 0; i < nbnd; ++i)
                for (long j = 0; j < nbnd; ++j)
                  A(iw, s, k, i, j) +=
                      ComplexType(w_n, 0.0)
                    * MO_skia(s, k, i, n) * std::conj(MO_skia(s, k, j, n));
            }
          }
    }
    sA.node_sync();
  }

  /// In-place causality projection on Im Sigma_c stored as (w, s, k, i, j).
  /// Calls the existing project_causality_ImSigma with a single repack
  /// (matches real_axis_scf_loop).
  inline void project_causality_inplace(real_axis_mb_state_t& state)
  {
    auto ImS = state.ImSigma_wskij->local();
    const long N_w  = ImS.shape()[0];
    const long ns   = ImS.shape()[1];
    const long Nk   = ImS.shape()[2];
    const long nbnd = ImS.shape()[3];
    nda::array<ComplexType, 5> ImS_skwij(ns, Nk, N_w, nbnd, nbnd);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k)
        for (long iw = 0; iw < N_w; ++iw)
          for (long i = 0; i < nbnd; ++i)
            for (long j = 0; j < nbnd; ++j)
              ImS_skwij(s, k, iw, i, j) = ImS(iw, s, k, i, j);
    project_causality_ImSigma(ImS_skwij);
    if (state.ImSigma_wskij->node_comm()->root()) {
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k)
          for (long iw = 0; iw < N_w; ++iw)
            for (long i = 0; i < nbnd; ++i)
              for (long j = 0; j < nbnd; ++j)
                ImS(iw, s, k, i, j) = ImS_skwij(s, k, iw, i, j);
    }
    state.ImSigma_wskij->node_sync();
  }

} // namespace detail_qp

/**
 * Real-axis quasiparticle self-consistent SCF loop. Mirrors the imag-axis
 * `methods::qp_scf_loop` directly: the static potential at each iter is
 *
 *     H_eff = H_0 + V_H(Dm) + Sigma_x(Dm) + V_corr
 *
 * where H_0 = T + V_ext (one-body, supplied via `sH_0_skij`), V_H + Sigma_x
 * are computed from the density matrix Dm by the production HF solver
 * `methods::HF::hf_t`, and V_corr is the static QSGW correction (or the
 * evGW projector). This convention adds Hartree + Exchange directly --
 * V_xc^KS does not appear, so no V_xc subtraction is needed.
 *
 * Per-iteration sequence:
 *   1. Diagonalize state.H_eff_skij -> state.MO_skia, state.E_ska
 *      (uses sS_skij if non-identity).
 *   2. Update mu from a Fermi sum over current QP eigenvalues.
 *   3. Update state.Dm_skij from MO/E (FD occupation).
 *   4. Build state.A_wskij from QP poles (Lorentzian sum, broadening
 *      cfg.eta) projected through state.MO_skia.
 *   5. mb_solver.scr_eri->update_w(state, thc)            -- W
 *   6. mb_solver.gw->evaluate(state, thc, ...)            -- Sigma_c
 *      (causality projection on Im Sigma_c diagonal)
 *   7. mb_solver.hf->evaluate(sHF, Dm, thc, sS, true, true) -- V_H + Sigma_x
 *   8. Form the next H_eff:
 *        QSGW (mode == qsgw):
 *           V_corr   = mb_solver.qp->compute_V_corr(...)
 *           H_eff_new = sH_0 + sHF + V_corr
 *        evGW (mode == evgw):
 *           sH_static = sH_0 + sHF
 *           sE_QP     = mb_solver.qp->solve_qp_diag(state, sH_static, MO, sE)
 *           H_eff_new = MO * diag(sE_QP) * MO^dagger
 *   9. Mix sH_eff (linear or DIIS) and compute residual ||dH_eff||_F.
 *  10. Convergence: residual < cfg.conv_tol.
 *
 * State on entry:
 *   state.H_eff_skij   may be unset (will be allocated and initialized to
 *                      `sH_0_skij + sHF(Dm_KS)` for a sensible starting
 *                      point). If allocated and non-zero, used as the
 *                      starting H_eff (restart). For a fresh run with no
 *                      better guess, supply Dm_KS via state.Dm_skij or let
 *                      the loop init H_eff = sH_0 (caller's choice).
 *   state.MO_skia, state.E_ska, state.Dm_skij  may be unset.
 *   state.A_wskij, state.{Im,Re}Sigma_wskij, state.Sigma_x_skij
 *                      may be unset (allocated by the loop).
 *
 * State on exit:
 *   - All sArrays above populated.
 *   - state.H_eff_skij = converged H_eff (or last iterate).
 *   - state.MO_skia, state.E_ska reflect eigenelements of state.H_eff_skij.
 *
 * For the orthogonal-basis case (sS = identity, current default), MO^{-1}
 * = MO^dagger and the QP solver's rotation back is exact. For non-trivial
 * overlap, `nda::linalg::eigenelements(F, S)` solves the generalized
 * problem; extending the QP solver's MO inverse to (S, MO) is out of scope
 * for this iteration but the SCF infrastructure threads sS through.
 */
template<MEMORY_SPACE MEM = HOST_MEMORY, methods::THC_ERI THC_t,
         nda::ArrayOfRank<4> H0_t, nda::ArrayOfRank<4> S_t>
qp_scgw_result
real_axis_qp_scf_loop(real_axis_mb_state_t                          & state,
                      H0_t                                    const& sH_0_skij,
                      S_t                                     const& sS_skij,
                      THC_t                                        & thc,
                      real_axis_qp_mb_solver_base_t<MEM>             mb_solver,
                      qp_mode                                        mode,
                      qp_scgw_config                          const& cfg,
                      nda::array<double, 1>                   const& k_weights,
                      double                                         N_elec,
                      long                                           ns_factor)
{
  static_assert(MEM == HOST_MEMORY,
                "real_axis_qp_scf_loop<DEVICE>: device path not yet supported");
  utils::check(state.grid != nullptr, "real_axis_qp_scf_loop: state.grid not bound");
  utils::check(state.mpi  != nullptr, "real_axis_qp_scf_loop: state.mpi not bound");
  utils::check(mb_solver.scr_eri != nullptr, "real_axis_qp_scf_loop: mb_solver.scr_eri null");
  utils::check(mb_solver.gw      != nullptr, "real_axis_qp_scf_loop: mb_solver.gw null");
  utils::check(mb_solver.qp      != nullptr, "real_axis_qp_scf_loop: mb_solver.qp null");
  utils::check(mb_solver.hf      != nullptr, "real_axis_qp_scf_loop: mb_solver.hf null");

  auto const& grid = *state.grid;
  auto& comm = state.mpi->comm;
  const long ns   = sH_0_skij.shape()[0];
  const long Nk   = sH_0_skij.shape()[1];
  const long nbnd = sH_0_skij.shape()[2];
  utils::check(sH_0_skij.shape()[3] == nbnd,
               "real_axis_qp_scf_loop: sH_0 not square in (i, j)");
  const long N_w  = grid.N_w();

  // ---- Allocate state buffers if needed. ----
  using sA5 = real_axis_mb_state_t::sArray_t<nda::array_view<ComplexType, 5>>;
  using sA4 = real_axis_mb_state_t::sArray_t<nda::array_view<ComplexType, 4>>;
  using sA3 = real_axis_mb_state_t::sArray_t<nda::array_view<ComplexType, 3>>;
  if (!state.A_wskij.has_value())
    state.allocate_fermionic(ns, Nk, nbnd);
  if (!state.H_eff_skij.has_value())
    state.H_eff_skij.emplace(*state.mpi,
        std::array<long, 4>{ns, Nk, nbnd, nbnd});
  if (!state.MO_skia.has_value())
    state.MO_skia.emplace(*state.mpi,
        std::array<long, 4>{ns, Nk, nbnd, nbnd});
  if (!state.E_ska.has_value())
    state.E_ska.emplace(*state.mpi,
        std::array<long, 3>{ns, Nk, nbnd});
  if (!state.Dm_skij.has_value())
    state.Dm_skij.emplace(*state.mpi,
        std::array<long, 4>{ns, Nk, nbnd, nbnd});

  // Production HF buffer: sH_HF = V_H(Dm) + Sigma_x(Dm), overwritten each iter.
  sA4 sH_HF(*state.mpi, std::array<long, 4>{ns, Nk, nbnd, nbnd});

  // ---- Initialize H_eff from sH_0 if it's all zero. ----
  {
    auto& sHe = state.H_eff_skij.value();
    bool is_zero = true;
    if (sHe.node_comm()->root()) {
      auto H = sHe.local();
      const long N = H.size();
      auto const* d = H.data();
      for (long i = 0; i < N; ++i) {
        if (std::abs(d[i]) > 0.0) { is_zero = false; break; }
      }
    }
    int iz_int = is_zero ? 1 : 0, iz_glob = 0;
    comm.all_reduce_n(&iz_int, 1, &iz_glob, std::logical_and<>{});
    if (iz_glob) {
      if (sHe.node_comm()->root()) {
        auto He = sHe.local();
        for (long s = 0; s < ns; ++s)
          for (long k = 0; k < Nk; ++k)
            for (long i = 0; i < nbnd; ++i)
              for (long j = 0; j < nbnd; ++j)
                He(s, k, i, j) = sH_0_skij(s, k, i, j);
      }
      sHe.node_sync();
    }
  }

  // ---- DIIS mixer over a phantom rank-5 sArray (1, ns, Nk, nbnd, nbnd). ----
  std::array<long, 5> Hwrap_shape = {1, ns, Nk, nbnd, nbnd};
  diis_mixer_t diis(state.mpi, cfg.diis_window);
  sA5 Hwrap_old(*state.mpi, Hwrap_shape);
  sA5 Hwrap_full(*state.mpi, Hwrap_shape);
  sA5 Hwrap_new (*state.mpi, Hwrap_shape);

  // ---- Local scratch (per-rank). ----
  nda::array<ComplexType, 4> H_full_skij(ns, Nk, nbnd, nbnd);
  nda::array<ComplexType, 4> Vc_skij    (ns, Nk, nbnd, nbnd);
  nda::array<ComplexType, 3> E_QP_ska   (ns, Nk, nbnd);

  const bool use_rspace = (Nk > 1);

  qp_scgw_result res;
  res.iter_used  = 0;
  res.final_diff = 0.0;
  res.final_mu   = grid.mu_chem();
  res.converged  = false;

  if (cfg.verbose and comm.root()) {
    app_log(1, "real_axis_qp_scf_loop: mode = {}, mix = {}, niter_max = {}, "
                "alpha = {}, tol = {:.1e}, eta = {}",
            (mode == qp_mode::qsgw ? "qsgw" : "evgw"),
            (cfg.mix_kind == qp_mix_kind::diis ? "diis" : "linear"),
            cfg.max_iter, cfg.alpha_mix, cfg.conv_tol, cfg.eta);
  }

  for (long it = 0; it < cfg.max_iter; ++it) {
    auto& sHe = state.H_eff_skij.value();
    auto& sMO = state.MO_skia.value();
    auto& sE  = state.E_ska.value();

    // ---- 1. Diagonalize H_eff -> MO, E. ----
    if (sMO.node_comm()->root()) {
      detail_qp::diagonalize_H_eff(sHe.local(), sS_skij, sE.local(), sMO.local());
    }
    sMO.node_sync();
    sE.node_sync();

    // ---- 2. Update mu. ----
    const double mu_cur = detail_qp::find_mu_from_QP(
        sE.local(), k_weights, grid.beta(), N_elec, ns_factor);
    res.final_mu = mu_cur;

    // ---- 3. Update density matrix from MO/E. ----
    auto& sDm = state.Dm_skij.value();
    if (sDm.node_comm()->root()) {
      detail_qp::update_Dm_from_QP(sE.local(), sMO.local(), mu_cur,
                                   grid.beta(), sDm.local());
    }
    sDm.node_sync();

    // ---- 4. Build A from QP poles. ----
    detail_qp::build_A_from_QP_poles(grid, sE.local(), sMO.local(), cfg.eta,
                                     state.A_wskij.value());

    // ---- 5. update W (scr_coulomb). ----
    if (cfg.update_W or it == 0) {
      mb_solver.scr_eri->update_w(state, thc, /*verbose*/ false, use_rspace);
    }

    // ---- 6. Sigma_c (gw). ----
    mb_solver.gw->evaluate(state, thc, cfg.eps_nufft,
                           mb_solver.scr_eri->div_treatment(),
                           /*verbose*/ false, use_rspace);
    detail_qp::project_causality_inplace(state);

    // ---- 7. V_H(Dm) + Sigma_x(Dm) via the production HF solver. ----
    //         Overwrites sH_HF with the Dm-based static self-energy
    //         (matches the imag-axis qp_scf_loop convention). hf_t::evaluate
    //         is explicitly instantiated for (sArray<Arrv4D>, Arrv4D, ...,
    //         Arrv4D) so we pass S as a view; sS_skij is the user-supplied
    //         owning nda::array<4>.
    nda::array_view<ComplexType, 4> sS_view(sS_skij);
    mb_solver.hf->evaluate(sH_HF, sDm.local(), thc, sS_view,
                           /*hartree*/ true, /*exchange*/ true);

    // ---- 8. Form next H_eff. ----
    if (mode == qp_mode::qsgw) {
      mb_solver.qp->compute_V_corr(state, sMO.local(), sE.local(), mu_cur,
                                    Vc_skij);
      // H_eff_new = sH_0 + sH_HF + V_corr
      auto HF = sH_HF.local();
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k)
          for (long i = 0; i < nbnd; ++i)
            for (long j = 0; j < nbnd; ++j)
              H_full_skij(s, k, i, j) =
                  sH_0_skij(s, k, i, j) + HF(s, k, i, j) + Vc_skij(s, k, i, j);
    } else {  // evGW
      // For evGW the static potential serving as Vhf_n in the QP equation
      // is H_0 + V_H + Sigma_x (one-body + Dm-based static self-energy).
      // We feed it as sH_static into solve_qp_diag.
      auto HF = sH_HF.local();
      nda::array<ComplexType, 4> H_static_skij(ns, Nk, nbnd, nbnd);
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k)
          for (long i = 0; i < nbnd; ++i)
            for (long j = 0; j < nbnd; ++j)
              H_static_skij(s, k, i, j) = sH_0_skij(s, k, i, j) + HF(s, k, i, j);

      mb_solver.qp->solve_qp_diag(state, H_static_skij, sMO.local(), sE.local(),
                                   mu_cur, E_QP_ska);
      // H_eff_new = MO * diag(eps_QP) * MO^dagger.
      nda::array<ComplexType, 2> Cdiag(nbnd, nbnd);
      nda::array<ComplexType, 2> Cdag (nbnd, nbnd);
      nda::array<ComplexType, 2> Hsk  (nbnd, nbnd);
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k) {
          for (long i = 0; i < nbnd; ++i)
            for (long n = 0; n < nbnd; ++n)
              Cdiag(i, n) = sMO.local()(s, k, i, n) * E_QP_ska(s, k, n);
          for (long n = 0; n < nbnd; ++n)
            for (long j = 0; j < nbnd; ++j)
              Cdag(n, j) = std::conj(sMO.local()(s, k, j, n));
          nda::blas::gemm(Cdiag, Cdag, Hsk);
          for (long i = 0; i < nbnd; ++i)
            for (long j = 0; j < nbnd; ++j)
              H_full_skij(s, k, i, j) = Hsk(i, j);
        }
    }

    // ---- 8. Mix sH_eff via the rank-5 phantom wrapper. ----
    if (Hwrap_old.node_comm()->root()) {
      auto Ho = Hwrap_old.local()(0, nda::ellipsis{});
      auto Hf = Hwrap_full.local()(0, nda::ellipsis{});
      Ho = sHe.local();
      Hf = H_full_skij;
    }
    Hwrap_old .node_sync();
    Hwrap_full.node_sync();

    double diff = 0.0;
    if (cfg.mix_kind == qp_mix_kind::diis) {
      diff = diis.mix(Hwrap_old, Hwrap_full, cfg.alpha_mix, Hwrap_new);
    } else {
      const double a = cfg.alpha_mix;
      if (Hwrap_new.node_comm()->root()) {
        auto Hn = Hwrap_new.local();
        auto Ho = Hwrap_old.local();
        auto Hf = Hwrap_full.local();
        const long N = Hn.size();
        auto * dn = Hn.data();
        auto const* da = Ho.data();
        auto const* df = Hf.data();
        for (long i = 0; i < N; ++i)
          dn[i] = (1.0 - a) * da[i] + a * df[i];
      }
      Hwrap_new.node_sync();
      auto Ho = Hwrap_old.local();
      auto Hf = Hwrap_full.local();
      double acc = 0.0;
      const long N = Ho.size();
      auto const* da = Ho.data();
      auto const* df = Hf.data();
      for (long i = 0; i < N; ++i) acc += std::norm(df[i] - da[i]);
      diff = std::sqrt(acc);
    }

    // Copy back to sHe (state.H_eff_skij).
    if (sHe.node_comm()->root()) {
      auto Hn = Hwrap_new.local()(0, nda::ellipsis{});
      sHe.local() = Hn;
    }
    sHe.node_sync();

    res.iter_used  = it + 1;
    res.final_diff = diff;
    if (cfg.verbose and comm.root()) {
      app_log(1, "  iter {:3}  ||dH_eff||_F = {:.4e}  mu = {:.6f}",
              it + 1, diff, mu_cur);
    }
    if (std::abs(diff) < std::abs(cfg.conv_tol)) {
      res.converged = true;
      break;
    }
  }

  // Final diagonalize: state.E_ska / state.MO_skia reflect the eigenelements
  // of the *final* state.H_eff_skij (post-mix), so callers can read QP
  // energies directly without an extra rotation. Mirrors the imag-axis
  // pattern where chkpt::dump_scf records the post-iter (E, MO).
  {
    auto& sHe = state.H_eff_skij.value();
    auto& sMO = state.MO_skia.value();
    auto& sE  = state.E_ska.value();
    if (sMO.node_comm()->root()) {
      detail_qp::diagonalize_H_eff(sHe.local(), sS_skij, sE.local(), sMO.local());
    }
    sMO.node_sync();
    sE.node_sync();
    res.final_mu = detail_qp::find_mu_from_QP(
        sE.local(), k_weights, grid.beta(), N_elec, ns_factor);
  }

  return res;
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_QP_SCF_DRIVER_HPP
