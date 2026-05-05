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

#ifndef COQUI_REAL_AXIS_QP_SOLVER_T_H
#define COQUI_REAL_AXIS_QP_SOLVER_T_H

#include <cmath>
#include <complex>
#include <tuple>
#include <utility>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"

#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_mb_state.hpp"
#include "methods/GW_real_axis/real_axis_qp_context.hpp"
#include "methods/GW_real_axis/real_axis_qp_utils.hpp"

namespace methods {
namespace real_axis {

/**
 * Real-axis quasiparticle solver.
 *
 * Two operating modes mirror the imag-axis `methods::qp_scf_loop` flavors:
 *
 *   evGW (eigenvalue self-consistent GW):
 *     `solve_qp_diag(state, sH_eff, sMO, mu, sE_QP_out)`
 *     -- iterates only the eigenvalues. Solves
 *           eps_QP_n = H_eff_nn(MO basis) + Re Sigma_c_nn(eps_QP_n)
 *        per (s, k, n). Algorithms (selected by qp_context.qp_type):
 *          bisection : bracketed bisection on the residual
 *          linearized: Z-factor evaluated at eps0
 *          secant    : Newton-secant iteration
 *          spectral  : argmax of |Im G^R| in a window around eps0
 *
 *   QSGW (full quasiparticle self-consistent GW, Faleev-Schilfgaarde-Kotani):
 *     `compute_V_corr(state, sE, sMO, mu, sV_corr_out)`
 *     -- builds a static, hermitian effective potential V_corr in the
 *        primary (orbital) basis. Off-diagonal modes:
 *          "qp_energy" (Faleev "Mode A"):
 *              V_{ab} = 0.5 * [Re Sigma_c_{ab}(eps_a) + Re Sigma_c_{ab}(eps_b)]
 *          "fermi":
 *              V_{ab} = Re Sigma_c_{aa}(eps_a)              if a == b
 *                     = Re Sigma_c_{ab}(omega = mu_chem)    if a != b
 *        followed by hermitization V <- 0.5*(V + V^dagger) and rotation
 *        back to the primary basis.
 *
 * The solver does NOT perform analytic continuation: Re Sigma_c lives on
 * the real-omega grid `state.grid->w()` already, and we sample it via
 * linear interpolation from `real_axis_qp_utils.hpp`. This is the
 * principal robustness advantage over the imag-axis QP solver.
 *
 * Frequency convention: `state.{Im,Re}Sigma_wskij` is stored on
 * `grid.w()`, which represents (omega - mu_chem) -- the same convention
 * as `dyson_G_one_kw`. Targets like `eps_QP - mu` are passed to the
 * interpolation directly.
 *
 * MPI: each rank does a per-rank-disjoint partition of the (s, k, n)
 * (or (s, k, a, b)) flat index, then a single allreduce on the result
 * sArray. State Sigma reads happen against the node-local sArray
 * (`state.ReSigma_wskij->local()`) so no comm traffic in the inner loops.
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
class real_axis_qp_solver_base_t {
public:
  // The QP solver work is small per-(s, k, n) bisection / linearized
  // root-finds plus a (nbnd x nbnd) hermitization. State Sigma reads happen
  // against host sArrays (`state.{Re,Im}Sigma_wskij->local()`). MEM is a
  // template marker so the QP-SCF driver can dispatch uniformly.

  using state_t = real_axis_mb_state_t;

  real_axis_qp_solver_base_t(real_freq_grid_t const* grid,
                             real_axis_qp_context_t  ctx)
    : _grid(grid), _ctx(std::move(ctx))
  {
    utils::check(_grid != nullptr,
                 "real_axis_qp_solver_t: grid pointer must not be null");
    utils::check(_ctx.qp_type == "bisection"  or _ctx.qp_type == "linearized" or
                 _ctx.qp_type == "secant"     or _ctx.qp_type == "spectral",
                 "real_axis_qp_solver_t: unknown qp_type \"{}\". "
                 "Valid: bisection / linearized / secant / spectral.",
                 _ctx.qp_type);
    utils::check(_ctx.off_diag_mode == "qp_energy" or _ctx.off_diag_mode == "fermi",
                 "real_axis_qp_solver_t: unknown off_diag_mode \"{}\". "
                 "Valid: qp_energy / fermi.", _ctx.off_diag_mode);
  }

  real_axis_qp_context_t const& context() const { return _ctx; }

  // --------------------------------------------------------------------
  // Phase 1: evGW -- diagonal QP equation per (s, k, n).
  // --------------------------------------------------------------------

  /**
   * Solve the diagonal QP equation for every (s, k, n) and write the
   * resulting QP energies into `sE_QP_out_ska`.
   *
   * Inputs:
   *   state       carries grid pointer (must equal _grid) and
   *               state.{Re,Im}Sigma_wskij allocated by gw_t::evaluate.
   *   sH_eff_skij effective one-body Hamiltonian in the PRIMARY (orbital)
   *               basis. For a "first iter" call, this is typically
   *               H_0 + Sigma_x; in later iterations it equals the prior
   *               iter's mixed H_eff.
   *   sMO_skia    MO coefficients (column a is the a-th MO in primary
   *               basis). For a first iter, these are the KS MOs;
   *               subsequently, eigvecs of the mixed H_eff.
   *   sE0_ska     starting eigenvalues (used as initial guess / bracket
   *               center). Typically eigvals of sH_eff at the start of
   *               the SCF iter.
   *   mu          chemical potential (absolute energy).
   *   sE_QP_out_ska  OUTPUT: QP eigenvalues, absolute energy.
   */
  void solve_qp_diag(state_t                         & state,
                     nda::ArrayOfRank<4> auto  const& sH_eff_skij_loc,
                     nda::ArrayOfRank<4> auto  const& sMO_skia_loc,
                     nda::ArrayOfRank<3> auto  const& sE0_ska_loc,
                     double                           mu,
                     nda::ArrayOfRank<3> auto       & sE_QP_out_ska_loc) const
  {
    require_grid_match(state);
    utils::check(state.ReSigma_wskij.has_value() and state.ImSigma_wskij.has_value(),
                 "solve_qp_diag: state.{Re,Im}Sigma_wskij not allocated -- "
                 "call gw_t::evaluate before solve_qp_diag.");
    auto& comm = state.mpi->comm;
    auto const& w_grid = _grid->w();
    auto const& ReS = state.ReSigma_wskij->local();
    auto const& ImS = state.ImSigma_wskij->local();
    const long N_w  = w_grid.shape()[0];
    const long ns   = ReS.shape()[1];
    const long Nk   = ReS.shape()[2];
    const long nbnd = ReS.shape()[3];
    utils::check(ReS.shape()[4] == nbnd, "solve_qp_diag: Sigma not square");
    utils::check(sH_eff_skij_loc.shape()[0] == ns and
                 sH_eff_skij_loc.shape()[1] == Nk and
                 sH_eff_skij_loc.shape()[2] == nbnd and
                 sH_eff_skij_loc.shape()[3] == nbnd,
                 "solve_qp_diag: sH_eff shape ({},{},{},{}) mismatches ({},{},{},{})",
                 sH_eff_skij_loc.shape()[0], sH_eff_skij_loc.shape()[1],
                 sH_eff_skij_loc.shape()[2], sH_eff_skij_loc.shape()[3],
                 ns, Nk, nbnd, nbnd);
    utils::check(sMO_skia_loc.shape()[0] == ns and
                 sMO_skia_loc.shape()[1] == Nk and
                 sMO_skia_loc.shape()[2] == nbnd and
                 sMO_skia_loc.shape()[3] == nbnd,
                 "solve_qp_diag: sMO shape mismatches expected (ns,Nk,nbnd,nbnd)");
    utils::check(sE0_ska_loc.shape()[0] == ns and
                 sE0_ska_loc.shape()[1] == Nk and
                 sE0_ska_loc.shape()[2] == nbnd,
                 "solve_qp_diag: sE0 shape mismatches expected (ns,Nk,nbnd)");
    utils::check(sE_QP_out_ska_loc.shape() == sE0_ska_loc.shape(),
                 "solve_qp_diag: sE_QP_out shape mismatches sE0");

    sE_QP_out_ska_loc = ComplexType(0.0, 0.0);

    nda::array<ComplexType, 1> Sigma_diag_w(N_w);
    nda::array<ComplexType, 1> MO_n(nbnd);

    const long total = ns * Nk * nbnd;
    const long rank  = comm.rank();
    const long size  = comm.size();
    long n_solved = 0, n_failed = 0;

    for (long flat = rank; flat < total; flat += size) {
      const long s   = flat / (Nk * nbnd);
      const long kn  = flat % (Nk * nbnd);
      const long k   = kn / nbnd;
      const long n   = kn % nbnd;

      // 1. Diagonal of H_eff in MO basis at this (s, k, n).
      //    Vhf_n = sum_ij conj(MO_in) * H_eff_ij * MO_jn
      const double Vhf_n = vhf_in_QP_basis(sH_eff_skij_loc, sMO_skia_loc, s, k, n);

      // 2. Diagonal of Sigma_c in MO basis on the full omega grid for
      //    band n. Re only is used by the residual; Im is kept for the
      //    spectral algorithm.
      for (long i = 0; i < nbnd; ++i) MO_n(i) = sMO_skia_loc(s, k, i, n);
      auto ReS_sk = ReS(nda::range::all, s, k, nda::range::all, nda::range::all);
      auto ImS_sk = ImS(nda::range::all, s, k, nda::range::all, nda::range::all);
      diag_Sigma_in_QP_basis(ReS_sk, ImS_sk, MO_n, Sigma_diag_w);

      // 3. Solve for eps_QP per the chosen algorithm.
      const double eps0 = sE0_ska_loc(s, k, n).real();
      double eps_qp = eps0;
      bool   ok     = true;
      if      (_ctx.qp_type == "bisection")  std::tie(eps_qp, ok) = solve_bisection (Vhf_n, Sigma_diag_w, eps0, mu);
      else if (_ctx.qp_type == "linearized") std::tie(eps_qp, ok) = solve_linearized(Vhf_n, Sigma_diag_w, eps0, mu);
      else if (_ctx.qp_type == "secant")     std::tie(eps_qp, ok) = solve_secant    (Vhf_n, Sigma_diag_w, eps0, mu);
      else                                    std::tie(eps_qp, ok) = solve_spectral  (Vhf_n, Sigma_diag_w, eps0, mu);

      if (!ok) {
        ++n_failed;
        eps_qp = eps0;  // safe fallback
      } else {
        ++n_solved;
      }
      sE_QP_out_ska_loc(s, k, n) = ComplexType(eps_qp, 0.0);

      // Diagnostic: dump per-band QP solve at k=0 for the first 4 bands
      // when the env var COQUI_QP_VERBOSE=1 is set. Shows Vhf, eps0,
      // eps_qp, residual to help debug the bisection convergence.
      if (rank == 0 and k == 0 and n < 6 and s == 0) {
        const char * v = std::getenv("COQUI_QP_VERBOSE");
        if (v != nullptr and std::string(v) == "1") {
          const ComplexType s_at = linear_interp_complex(_grid->w(), Sigma_diag_w,
                                                         eps_qp - mu);
          app_log(1, "  [QP] s={} k={} n={}  eps0={:.6f}  Vhf_n={:.6f}  "
                      "eps_qp={:.6f}  Re Sigma_c(@eps_qp)={:.6f}  ok={}",
                  s, k, n, eps0, Vhf_n, eps_qp, s_at.real(), ok ? "Y" : "N");
        }
      }
    }

    // Reduce across ranks (each (s, k, n) is owned by exactly one rank).
    comm.all_reduce_in_place_n(sE_QP_out_ska_loc.data(),
                               sE_QP_out_ska_loc.size(), std::plus<>{});
    long g_solved = 0, g_failed = 0;
    comm.all_reduce_n(&n_solved, 1, &g_solved, std::plus<>{});
    comm.all_reduce_n(&n_failed, 1, &g_failed, std::plus<>{});

    if (comm.root()) {
      app_log(2, "real_axis_qp_solver: qp_type={}, off_diag_mode={}, "
                  "solved {}/{} bands ({} failed -- fallback to eps_KS).",
              _ctx.qp_type, _ctx.off_diag_mode,
              g_solved, g_solved + g_failed, g_failed);
    }
  }

  // --------------------------------------------------------------------
  // Phase 2: full QSGW -- static V_corr in the primary basis.
  // --------------------------------------------------------------------

  /**
   * Build the static QSGW correction potential V_corr_{ij}(s, k) and
   * write it into `sV_corr_out_skij`.
   *
   * Algorithm:
   *   1. For each (s, k), rotate Re Sigma_c_wskij(s, k) into the QP/MO basis
   *      to get ReSigma_c_wskab on the omega grid.
   *   2. Sample at the QP energies per `_ctx.off_diag_mode`:
   *        "qp_energy": V_{ab} = 0.5 * [Re Sigma_c_{ab}(eps_a)
   *                                   + Re Sigma_c_{ab}(eps_b)]
   *        "fermi"    : V_{aa} = Re Sigma_c_{aa}(eps_a)
   *                     V_{ab} = Re Sigma_c_{ab}(omega = mu)  (a != b)
   *   3. Hermitize: V <- 0.5 * (V + V^dagger).
   *   4. Rotate back to primary basis: V_corr_{ij} = sum_ab MO_{ia} * V_{ab}
   *      * conj(MO_{jb}). This requires MO^{-1} = MO^dagger * S only if
   *      MO is non-orthogonal; for the orthogonal-basis case (current
   *      real-axis pipeline assumption) MO^{-1} = MO^dagger.
   *
   * Re-only output is enforced by storing only the .real() part in V_corr;
   * the imag-axis equivalent does the same hermitization. V_corr is real-
   * symmetric in the orbital basis when MO is real; complex-hermitian in
   * general.
   */
  void compute_V_corr(state_t                         & state,
                      nda::ArrayOfRank<4> auto  const& sMO_skia_loc,
                      nda::ArrayOfRank<3> auto  const& sE_ska_loc,
                      double                           mu,
                      nda::ArrayOfRank<4> auto       & sV_corr_out_skij_loc) const
  {
    require_grid_match(state);
    utils::check(state.ReSigma_wskij.has_value() and state.ImSigma_wskij.has_value(),
                 "compute_V_corr: state.{Re,Im}Sigma_wskij not allocated -- "
                 "call gw_t::evaluate before compute_V_corr.");
    auto& comm = state.mpi->comm;
    auto const& w_grid = _grid->w();
    auto const& ReS = state.ReSigma_wskij->local();
    const long N_w  = w_grid.shape()[0];
    const long ns   = ReS.shape()[1];
    const long Nk   = ReS.shape()[2];
    const long nbnd = ReS.shape()[3];
    utils::check(ReS.shape()[4] == nbnd, "compute_V_corr: Sigma not square");
    utils::check(sMO_skia_loc.shape()[0] == ns and
                 sMO_skia_loc.shape()[1] == Nk and
                 sMO_skia_loc.shape()[2] == nbnd and
                 sMO_skia_loc.shape()[3] == nbnd,
                 "compute_V_corr: sMO shape mismatches expected (ns,Nk,nbnd,nbnd)");
    utils::check(sE_ska_loc.shape()[0] == ns and
                 sE_ska_loc.shape()[1] == Nk and
                 sE_ska_loc.shape()[2] == nbnd,
                 "compute_V_corr: sE shape mismatches expected (ns,Nk,nbnd)");
    utils::check(sV_corr_out_skij_loc.shape()[0] == ns and
                 sV_corr_out_skij_loc.shape()[1] == Nk and
                 sV_corr_out_skij_loc.shape()[2] == nbnd and
                 sV_corr_out_skij_loc.shape()[3] == nbnd,
                 "compute_V_corr: sV_corr_out shape mismatches");

    sV_corr_out_skij_loc = ComplexType(0.0, 0.0);

    // Distributed (s, k) loop with allreduce. Per (s, k) the work is one
    // rotation of Sigma into the MO basis (N_w * 2 small gemms) plus a
    // pointwise sample / hermitize / rotate-back.
    const long total = ns * Nk;
    const long rank  = comm.rank();
    const long size  = comm.size();

    nda::array<ComplexType, 3> ReS_mo_wab(N_w, nbnd, nbnd);
    nda::array<ComplexType, 2> tmp_ib(nbnd, nbnd);  // Sigma_ij * MO_jb
    nda::array<ComplexType, 2> Cdag_ai(nbnd, nbnd);
    nda::array<ComplexType, 2> Sigma_ij_w(nbnd, nbnd);  // ReS at one omega
    nda::array<ComplexType, 2> V_ab(nbnd, nbnd);
    nda::array<ComplexType, 2> V_ab_h(nbnd, nbnd);
    nda::array<ComplexType, 2> tmp_ab(nbnd, nbnd);
    nda::array<ComplexType, 2> V_ij(nbnd, nbnd);

    for (long flat = rank; flat < total; flat += size) {
      const long s = flat / Nk;
      const long k = flat % Nk;

      auto MO  = sMO_skia_loc(s, k, nda::range::all, nda::range::all);

      // --- Rotate Re Sigma_c_wij into MO basis: ReS_mo_wab = MO^dagger * ReS_w * MO
      Cdag_ai = nda::transpose(nda::conj(MO));
      for (long iw = 0; iw < N_w; ++iw) {
        for (long i = 0; i < nbnd; ++i)
          for (long j = 0; j < nbnd; ++j)
            Sigma_ij_w(i, j) = ComplexType(ReS(iw, s, k, i, j).real(), 0.0);
        nda::blas::gemm(Sigma_ij_w, MO, tmp_ib);
        nda::blas::gemm(Cdag_ai, tmp_ib, V_ab);
        ReS_mo_wab(iw, nda::range::all, nda::range::all) = V_ab;
      }

      // --- Sample at QP energies.
      for (long a = 0; a < nbnd; ++a) {
        const double eps_a = sE_ska_loc(s, k, a).real();
        const double w_a   = eps_a - mu;
        for (long b = 0; b < nbnd; ++b) {
          const double eps_b = sE_ska_loc(s, k, b).real();
          const double w_b   = eps_b - mu;
          double v_re = 0.0;
          if (_ctx.off_diag_mode == "qp_energy") {
            const double r_a = interp_real_at_w(w_grid,
                                                 ReS_mo_wab, a, b, w_a);
            const double r_b = interp_real_at_w(w_grid,
                                                 ReS_mo_wab, a, b, w_b);
            v_re = 0.5 * (r_a + r_b);
          } else {  // "fermi"
            if (a == b) {
              v_re = interp_real_at_w(w_grid, ReS_mo_wab, a, b, w_a);
            } else {
              // off-diagonal at omega = mu  =>  w_rel = 0
              v_re = interp_real_at_w(w_grid, ReS_mo_wab, a, b, 0.0);
            }
          }
          V_ab(a, b) = ComplexType(v_re, 0.0);
        }
      }

      // --- Hermitize.
      V_ab_h = 0.5 * (V_ab + nda::transpose(nda::conj(V_ab)));

      // --- Rotate back to primary basis: V_ij = sum_ab MO_ia V_ab MO*_jb
      //     = MO * V * MO^dagger.
      nda::blas::gemm(V_ab_h, Cdag_ai, tmp_ab);
      nda::blas::gemm(MO, tmp_ab, V_ij);

      sV_corr_out_skij_loc(s, k, nda::range::all, nda::range::all) = V_ij;
    }

    comm.all_reduce_in_place_n(sV_corr_out_skij_loc.data(),
                               sV_corr_out_skij_loc.size(), std::plus<>{});
  }

private:

  void require_grid_match(state_t const& state) const {
    utils::check(state.grid != nullptr,
                 "real_axis_qp_solver: state.grid not bound");
    utils::check(state.grid == _grid,
                 "real_axis_qp_solver: state.grid disagrees with the grid the "
                 "solver was constructed with");
    utils::check(state.mpi != nullptr,
                 "real_axis_qp_solver: state.mpi not bound");
  }

  template<typename ArrH, typename ArrMO>
  static double vhf_in_QP_basis(ArrH const& sH_eff_skij_loc,
                                ArrMO const& sMO_skia_loc,
                                long s, long k, long n)
  {
    const long nbnd = sH_eff_skij_loc.shape()[2];
    ComplexType acc(0.0, 0.0);
    for (long i = 0; i < nbnd; ++i) {
      ComplexType row_acc(0.0, 0.0);
      for (long j = 0; j < nbnd; ++j) {
        row_acc += sH_eff_skij_loc(s, k, i, j) * sMO_skia_loc(s, k, j, n);
      }
      acc += std::conj(sMO_skia_loc(s, k, i, n)) * row_acc;
    }
    return acc.real();
  }

  /// Linear interp of Re part of an (N_w, nbnd, nbnd) array at (a, b)
  /// on the relative-w grid. Out-of-range -> nearest boundary.
  template<typename Arr3>
  static double interp_real_at_w(nda::array<double, 1> const& w_grid,
                                 Arr3 const& A_wab,
                                 long a, long b, double w_rel)
  {
    const long N_w = w_grid.shape()[0];
    if (w_rel <= w_grid(0))      return A_wab(0,       a, b).real();
    if (w_rel >= w_grid(N_w-1))  return A_wab(N_w - 1, a, b).real();
    const long i = lower_bracket(w_grid, w_rel);
    const double t = (w_rel - w_grid(i)) / (w_grid(i + 1) - w_grid(i));
    return (1.0 - t) * A_wab(i,     a, b).real()
         +        t  * A_wab(i + 1, a, b).real();
  }

  /// QP equation residual: f(w) = w - Vhf - Re Sigma_diag(w - mu).
  /// Sigma_diag_w is sampled on _grid->w() (relative coordinate).
  static double qp_residual(double Vhf, double w_abs, double mu,
                            nda::array<double, 1> const& w_grid,
                            nda::array<ComplexType, 1> const& Sigma_diag_w)
  {
    const double w_rel = w_abs - mu;
    const ComplexType s = linear_interp_complex(w_grid, Sigma_diag_w, w_rel);
    return w_abs - Vhf - s.real();
  }

  std::tuple<double, bool>
  solve_bisection(double Vhf, nda::array<ComplexType, 1> const& Sigma_w,
                  double eps0, double mu) const
  {
    auto const& w = _grid->w();
    const double tol = _ctx.tol;
    auto f = [&](double e) { return qp_residual(Vhf, e, mu, w, Sigma_w); };
    double r0 = f(eps0);
    if (std::abs(r0) < tol) return {eps0, true};

    double e_lo, e_hi;
    double delta = 0.05;  // a.u.; covers typical QP shifts
    if (r0 >= 0) {
      e_hi = eps0;  e_lo = eps0 - delta;
      double r = f(e_lo);
      long max_expand = 200;
      while (r > 0 and max_expand-- > 0) { e_lo -= delta; r = f(e_lo); }
      if (r > 0) return {eps0, false};
    } else {
      e_lo = eps0;  e_hi = eps0 + delta;
      double r = f(e_hi);
      long max_expand = 200;
      while (r < 0 and max_expand-- > 0) { e_hi += delta; r = f(e_hi); }
      if (r < 0) return {eps0, false};
    }
    long max_iter = 200;
    while (max_iter-- > 0) {
      const double e_mid = 0.5 * (e_lo + e_hi);
      const double r_mid = f(e_mid);
      if (std::abs(r_mid) < tol) return {e_mid, true};
      if (r_mid >= 0) e_hi = e_mid;
      else            e_lo = e_mid;
    }
    return {0.5 * (e_lo + e_hi), false};
  }

  std::tuple<double, bool>
  solve_linearized(double Vhf, nda::array<ComplexType, 1> const& Sigma_w,
                   double eps0, double /*mu*/) const
  {
    // eps_QP = eps0 + Z * (Vhf + Re Sigma(eps0 - mu) - eps0)
    // Z = 1 / (1 - dRe Sigma / domega | eps0 - mu)
    auto const& w = _grid->w();
    // Note: derivative is taken on the relative axis; chain rule trivial
    // (d/d omega = d/d w_rel since w_rel = omega - mu).
    const double w_rel = eps0 - _grid->mu_chem();
    const ComplexType s0 = linear_interp_complex(w, Sigma_w, w_rel);
    // Finite-diff derivative on the bracketing interval.
    const long N_w = w.shape()[0];
    long iw = lower_bracket(w, w_rel);
    if (iw < 0) iw = 0;
    if (iw > N_w - 2) iw = N_w - 2;
    const double dw = w(iw + 1) - w(iw);
    const double dS = Sigma_w(iw + 1).real() - Sigma_w(iw).real();
    const double dSdw = dS / dw;
    const double Z = 1.0 / (1.0 - dSdw);
    const double eps_qp = eps0 + Z * (Vhf + s0.real() - eps0);
    return {eps_qp, std::isfinite(eps_qp) and Z > 0.0};
  }

  std::tuple<double, bool>
  solve_secant(double Vhf, nda::array<ComplexType, 1> const& Sigma_w,
               double eps0, double mu) const
  {
    auto const& w = _grid->w();
    auto f = [&](double e) { return qp_residual(Vhf, e, mu, w, Sigma_w); };
    double p0 = eps0;
    double p1 = (eps0 >= 0) ? eps0 * (1.0 + 1e-4) + 1e-4
                            : eps0 * (1.0 + 1e-4) - 1e-4;
    double q0 = f(p0), q1 = f(p1);
    long it = 0;
    for (; it < _ctx.secant_maxiter; ++it) {
      double p;
      if (std::abs(q1) > std::abs(q0))
        p = (-q0 / q1 * p1 + p0) / (1.0 - q0 / q1);
      else
        p = (-q1 / q0 * p0 + p1) / (1.0 - q1 / q0);
      if (!std::isfinite(p)) return {eps0, false};
      if (std::abs(p - p1) < _ctx.tol) return {p, true};
      p0 = p1; q0 = q1;
      p1 = p;  q1 = f(p1);
    }
    return {p1, std::abs(q1) < 10.0 * _ctx.tol};
  }

  std::tuple<double, bool>
  solve_spectral(double Vhf, nda::array<ComplexType, 1> const& Sigma_w,
                 double eps0, double mu) const
  {
    // Find argmax of |Im G^R(omega)| in a window [eps0-1, eps0+1].
    // G^R(omega) = 1 / (omega - Vhf - Sigma(omega-mu))
    auto const& w = _grid->w();
    const double eta = _ctx.eta;
    auto Aw = [&](double omega) {
      const ComplexType S = linear_interp_complex(w, Sigma_w, omega - mu);
      const ComplexType denom = ComplexType(omega - Vhf - S.real(), eta - S.imag());
      const ComplexType G = 1.0 / denom;
      return std::abs(G.imag());
    };
    const double w_min = eps0 - 1.0;
    const double w_max = eps0 + 1.0;
    const long Nw = 1000;
    double best_w = eps0;
    double best_A = Aw(eps0);
    for (long i = 0; i < Nw; ++i) {
      const double omega = w_min + (w_max - w_min) * static_cast<double>(i) / static_cast<double>(Nw - 1);
      const double A = Aw(omega);
      if (A > best_A) { best_A = A; best_w = omega; }
    }
    // Refine within tolerance.
    double step = _ctx.tol;
    double a_plus  = Aw(best_w + step);
    if (a_plus > best_A) {
      while (a_plus > best_A) {
        best_A = a_plus; best_w += step; a_plus = Aw(best_w + step);
      }
    } else {
      double a_minus = Aw(best_w - step);
      while (a_minus > best_A) {
        best_A = a_minus; best_w -= step; a_minus = Aw(best_w - step);
      }
    }
    return {best_w, std::isfinite(best_w) and std::isfinite(best_A)};
  }

  real_freq_grid_t const* _grid;
  real_axis_qp_context_t  _ctx;
};

using real_axis_qp_solver_t = real_axis_qp_solver_base_t<HOST_MEMORY>;

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_QP_SOLVER_T_H
