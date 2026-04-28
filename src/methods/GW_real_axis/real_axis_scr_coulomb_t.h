/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_SCR_COULOMB_T_H
#define COQUI_REAL_AXIS_SCR_COULOMB_T_H

#include <chrono>
#include <complex>
#include <memory>
#include <string>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/check.hpp"
#include "utilities/kpoint_utils.hpp"
#include "numerics/shared_array/nda.hpp"

#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_conv.hpp"
#include "methods/GW_real_axis/real_axis_mb_state.hpp"
#include "methods/GW_real_axis/real_axis_thc_project.hpp"
#include "methods/GW_real_axis/real_axis_pi.hpp"
#include "methods/GW_real_axis/real_axis_dyson.hpp"

namespace methods {
namespace real_axis {

/**
 * Real-axis screened-interaction solver. Mirrors methods::solvers::scr_coulomb_t
 * (the imag-axis class) but operates on the real-frequency grid and a
 * real_axis_mb_state_t.
 *
 * Contract:
 *   update_w reads `state.A_wskij`, computes Im/Re Pi via the THC-auxiliary
 *   cross-correlation kernel + Hilbert transform, solves the per-(q,Omega)
 *   bosonic Dyson W = (I - V Pi)^{-1} V, and writes
 *
 *     state.ImPi_qPQO, state.RePi_qPQO   -- polarization (Naux, Naux, N_Omega)
 *     state.ImW_qPQO,  state.ReW_qPQO    -- screened interaction
 *
 * with iq_gamma zeroed when div_treatment == "ignore_g0".
 *
 * Matches Steps 1-4 of `evaluate_serial` in `real_axis_gw_driver.hpp`. The
 * production driver duplicates this logic for now; once the SCF refactor
 * lands the driver collapses to a thin wrapper around update_w + the
 * forthcoming real_axis_gw_t::evaluate.
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
class real_axis_scr_coulomb_base_t {
public:
  using mpi_communicator_t = boost::mpi3::communicator;

  real_axis_scr_coulomb_base_t(real_freq_grid_t const* grid,
                               std::string screen_type   = "rpa",
                               std::string div_treatment = "ignore_g0",
                               double      eps_nufft     = 1e-10)
    : _grid(grid),
      _screen_type(std::move(screen_type)),
      _div_treatment(std::move(div_treatment)),
      _eps_nufft(eps_nufft)
  {
    utils::check(_grid != nullptr,
                 "real_axis_scr_coulomb_t: grid pointer must not be null");
    utils::check(_screen_type == "rpa",
                 "real_axis_scr_coulomb_t: only screen_type=\"rpa\" is supported "
                 "in this version (got \"{}\")", _screen_type);
  }

  ~real_axis_scr_coulomb_base_t() = default;

  std::string screen_type()   const { return _screen_type; }
  std::string div_treatment() const { return _div_treatment; }
  double      eps_nufft()     const { return _eps_nufft; }

  /**
   * Compute the dynamic screened interaction and store it in the MBState.
   *
   * @param state       reads state.A_wskij; writes state.{Im,Re}{Pi,W}_qPQO.
   *                    The MPI communicator is read from state.mpi->comm.
   * @param thc         THC ERI object (provides X, Z=V_PQ, MF accessors).
   * @param verbose     emit per-step timings on rank 0.
   * @param use_rspace  when true and Nk>1, run Π in R-space (Nk-fold reduction
   *                    in NUFFT calls) rather than k-space.
   */
  template<methods::THC_ERI THC_t>
  void update_w(real_axis_mb_state_t& state,
                THC_t const& thc,
                bool verbose    = false,
                bool use_rspace = false);

private:
  real_freq_grid_t const* _grid = nullptr;
  std::string _screen_type;
  std::string _div_treatment;
  double      _eps_nufft = 1e-10;
};

using real_axis_scr_coulomb_t = real_axis_scr_coulomb_base_t<HOST_MEMORY>;

// --------------------------------------------------------------------------
// update_w implementation. Header-inline for now (matches the rest of the
// real-axis module). Body parallels Steps 1-4 of `evaluate_serial`.
// --------------------------------------------------------------------------
template<MEMORY_SPACE MEM>
template<methods::THC_ERI THC_t>
void real_axis_scr_coulomb_base_t<MEM>::update_w(
    real_axis_mb_state_t& state,
    THC_t const& thc,
    bool verbose,
    bool use_rspace)
{
  static_assert(MEM == HOST_MEMORY,
                "real_axis_scr_coulomb_t<DEVICE>: device-side instantiation "
                "is not yet supported. The conv engine and kernels are "
                "MEM-aware but the per-element buffer copies in this body "
                "are still host-only.");
  utils::check(state.A_wskij.has_value(),
               "real_axis_scr_coulomb_t::update_w: state.A_wskij not allocated");
  utils::check(state.grid != nullptr,
               "real_axis_scr_coulomb_t::update_w: state.grid not bound");
  utils::check(state.grid == _grid,
               "real_axis_scr_coulomb_t::update_w: state.grid disagrees with "
               "the grid the solver was constructed with");
  utils::check(state.mpi != nullptr,
               "real_axis_scr_coulomb_t::update_w: state.mpi not bound");
  auto& comm = state.mpi->comm;

  using nda::range;
  const auto _ = range::all;
  using clock_t = std::chrono::steady_clock;
  auto t_now = []{ return clock_t::now(); };
  auto sec_since = [](clock_t::time_point t0) {
    return std::chrono::duration<double>(clock_t::now() - t0).count();
  };
  const auto t_total = t_now();

  auto const& grid = *_grid;
  auto const& MF   = *thc.MF();

  const long ns    = MF.nspin();
  const long nbnd  = MF.nbnd();
  const long Nk    = MF.nkpts();
  const long Nq    = MF.nqpts();
  const long Naux  = thc.Np();
  const long N_w   = grid.N_w();
  const long N_O   = grid.N_Omega();

  utils::check(MF.npol() == 1,
               "real_axis_scr_coulomb_t::update_w: npol={} not supported (need 1)",
               MF.npol());

  auto const& A_in = *state.A_wskij;
  utils::check(A_in.shape()[0] == N_w and A_in.shape()[1] == ns and
               A_in.shape()[2] == Nk and A_in.shape()[3] == nbnd and
               A_in.shape()[4] == nbnd,
               "real_axis_scr_coulomb_t::update_w: state.A_wskij shape mismatch");

  // (Re)allocate bosonic state fields.
  state.allocate_bosonic(Nq, Naux);
  auto & ImPi = *state.ImPi_qPQO;
  auto & RePi = *state.RePi_qPQO;
  auto & ImW  = *state.ImW_qPQO;
  auto & ReW  = *state.ReW_qPQO;
  ImPi = ComplexType(0.0, 0.0);
  RePi = ComplexType(0.0, 0.0);
  ImW  = ComplexType(0.0, 0.0);
  ReW  = ComplexType(0.0, 0.0);

  // Repack input A from (N_w, ns, nkpts, nbnd, nbnd) to driver layout
  // (ns, nkpts, N_w, nbnd, nbnd).
  nda::array<ComplexType, 5> A(ns, Nk, N_w, nbnd, nbnd);
  for (long s = 0; s < ns; ++s)
    for (long k = 0; k < Nk; ++k)
      for (long iw = 0; iw < N_w; ++iw)
        for (long mu = 0; mu < nbnd; ++mu)
          for (long nu = 0; nu < nbnd; ++nu)
            A(s, k, iw, mu, nu) = A_in(iw, s, k, mu, nu);

  // Marshal X(s, k, P, mu) from THC reader.
  nda::array<ComplexType, 4> X(ns, Nk, Naux, nbnd);
  for (long s = 0; s < ns; ++s)
    for (long k = 0; k < Nk; ++k) {
      auto Xsk = thc.X(static_cast<int>(s), /*ip*/ 0, static_cast<int>(k));
      for (long P = 0; P < Naux; ++P)
        for (long mu = 0; mu < nbnd; ++mu)
          X(s, k, P, mu) = Xsk(P, mu);
    }

  // V(iq, P, Q) from THC.Z(iq).
  nda::array<ComplexType, 3> V(Nq, Naux, Naux);
  for (long iq = 0; iq < Nq; ++iq) {
    auto Zq = thc.Z(static_cast<int>(iq));
    for (long P = 0; P < Naux; ++P)
      for (long Q = 0; Q < Naux; ++Q)
        V(iq, P, Q) = Zq(P, Q);
  }

  // BZ closure maps in shared memory. kpq(ik, iq) = ik+iq via qk_to_k2(qminus(iq), ik).
  math::shm::shared_array<nda::array_view<long, 2>> skpq(*state.mpi, {Nk, Nq});
  {
    if (skpq.node_comm()->root()) {
      auto kpq_loc = skpq.local();
      auto const& qk_to_k2 = MF.qk_to_k2();
      auto const& qm       = MF.qminus();
      for (long iq = 0; iq < Nq; ++iq)
        for (long ik = 0; ik < Nk; ++ik)
          kpq_loc(ik, iq) = qk_to_k2(qm(iq), ik);
    }
    skpq.node_sync();
  }
  auto kpq = skpq.local();

  // Identify the Gamma q-point if div_treatment == "ignore_g0".
  long iq_gamma = -1;
  if (_div_treatment == "ignore_g0") {
    auto Qp = MF.Qpts();
    if (Qp.shape()[0] >= 1) {
      double norm0 = 0.0;
      for (long c = 0; c < Qp.shape()[1]; ++c) norm0 += std::abs(Qp(0, c));
      if (norm0 < 1e-10) iq_gamma = 0;
    }
  }

  // R-space FT matrices for Pi, when requested. One copy per node.
  std::optional<math::shm::shared_array<nda::array_view<ComplexType, 2>>> sf_Rk_opt;
  std::optional<math::shm::shared_array<nda::array_view<ComplexType, 2>>> sf_qR_opt;
  long NR = 0;
  if (use_rspace and Nk > 1) {
    auto kp_grid = MF.kp_grid();
    auto lattv   = MF.lattv();
    const long nx = kp_grid(0);
    const long ny = kp_grid(1);
    const long nz = kp_grid(2);
    NR = nx * ny * nz;
    utils::check(NR == Nk,
                 "real_axis_scr_coulomb_t::update_w: R-space path expects NR ({}) "
                 "== Nk ({})", NR, Nk);

    nda::array<long, 2> Rpts_idx(NR, 3);
    for (long p = 0; p < NR; ++p) {
      long a = p / (ny * nz);
      long b = (p / nz) % ny;
      long c = p % nz;
      if (a > nx / 2) a -= nx;
      if (b > ny / 2) b -= ny;
      if (c > nz / 2) c -= nz;
      Rpts_idx(p, 0) = a;
      Rpts_idx(p, 1) = b;
      Rpts_idx(p, 2) = c;
    }
    nda::array<long, 1> Rpts_weights(NR);
    Rpts_weights() = 1;

    sf_Rk_opt.emplace(*state.mpi, std::array<long,2>{NR, Nk});
    sf_qR_opt.emplace(*state.mpi, std::array<long,2>{Nq, NR});
    utils::k_to_R_coefficients(comm, Rpts_idx, MF.kpts(), lattv, *sf_Rk_opt);
    utils::R_to_k_coefficients(comm, Rpts_idx, Rpts_weights, MF.Qpts(), lattv,
                               *sf_qR_opt);
  }
  const bool do_rspace = (NR > 0);

  // NUFFT engine sized to the largest batched cross-correlation (Naux*Naux).
  const auto t_conv0 = t_now();
  real_axis_conv_t conv(grid, /*ntrans*/ Naux*Naux, _eps_nufft);
  const double dt_conv = sec_since(t_conv0);

  // ----------------------------------------------------------------
  // Step 1: project A(k) -> A_aux_skPQw with iw innermost.
  // ----------------------------------------------------------------
  const auto t1 = t_now();
  nda::array<ComplexType, 5> A_aux_skPQw(ns, Nk, Naux, Naux, N_w);
  for (long s = 0; s < ns; ++s)
    for (long ik = 0; ik < Nk; ++ik) {
      auto X_view     = X(s, ik, _, _);
      auto A_view     = A(s, ik, _, _, _);
      auto A_aux_view = A_aux_skPQw(s, ik, _, _, _);
      primary_to_aux_one_k(X_view, A_view, A_aux_view);
    }
  const double dt1 = sec_since(t1);

  // ----------------------------------------------------------------
  // Step 2: Im Pi via spectral cross-correlation. k-space or R-space.
  // ----------------------------------------------------------------
  const auto t2 = t_now();

  if (do_rspace) {
    auto f_Rk = sf_Rk_opt->local();
    auto f_qR = sf_qR_opt->local();

    // FT A_aux from k-space to R-space, per s.
    nda::array<ComplexType, 5> A_aux_sRPQw(ns, NR, Naux, Naux, N_w);
    for (long s = 0; s < ns; ++s) {
      auto A_in_2D  = nda::reshape(A_aux_skPQw(s, _, _, _, _),
                                   std::array<long,2>{Nk, Naux*Naux*N_w});
      auto A_out_2D = nda::reshape(A_aux_sRPQw(s, _, _, _, _),
                                   std::array<long,2>{NR, Naux*Naux*N_w});
      nda::blas::gemm(ComplexType(1.0, 0.0), f_Rk, A_in_2D,
                      ComplexType(0.0, 0.0), A_out_2D);
    }

    // Per-R cross-correlation, distributed over R.
    nda::array<ComplexType, 4> ImPi_RPQO(NR, Naux, Naux, N_O);
    ImPi_RPQO = ComplexType(0.0, 0.0);
    {
      const int rank = comm.rank();
      const int size = comm.size();
      for (long iR = rank; iR < NR; iR += size) {
        for (long s = 0; s < ns; ++s) {
          auto A_view      = A_aux_sRPQw(s, iR, _, _, _);
          auto ImPi_R_view = ImPi_RPQO(iR, _, _, _);
          accumulate_ImPi_one_kq(conv, A_view, A_view, ImPi_R_view, 1.0);
        }
      }
      if (size > 1)
        comm.all_reduce_in_place_n(ImPi_RPQO.data(), ImPi_RPQO.size(), std::plus<>{});
    }

    // FT ImPi from R-space to q-space.
    {
      auto ImPi_R_2D = nda::reshape(ImPi_RPQO, std::array<long,2>{NR, Naux*Naux*N_O});
      auto ImPi_q_2D = nda::reshape(ImPi, std::array<long,2>{Nq, Naux*Naux*N_O});
      nda::blas::gemm(ComplexType(1.0, 0.0), f_qR, ImPi_R_2D,
                      ComplexType(0.0, 0.0), ImPi_q_2D);
    }

    if (iq_gamma >= 0) {
      for (long P = 0; P < Naux; ++P)
        for (long Q = 0; Q < Naux; ++Q)
          for (long iO = 0; iO < N_O; ++iO)
            ImPi(iq_gamma, P, Q, iO) = ComplexType(0.0, 0.0);
    }
  } else {
    const double k_weight = 1.0 / static_cast<double>(Nk);
    const long total_qsk = Nq * ns * Nk;
    const int rank = comm.rank();
    const int size = comm.size();
    for (long idx = rank; idx < total_qsk; idx += size) {
      const long iq  = idx / (ns * Nk);
      const long rem = idx % (ns * Nk);
      const long s   = rem / Nk;
      const long ik  = rem % Nk;
      if (iq == iq_gamma) continue;
      const long ikq = kpq(ik, iq);
      auto Ak_view   = A_aux_skPQw(s, ik,  _, _, _);
      auto Akq_view  = A_aux_skPQw(s, ikq, _, _, _);
      auto ImPi_view = ImPi(iq, _, _, _);
      accumulate_ImPi_one_kq(conv, Ak_view, Akq_view, ImPi_view, k_weight);
    }
    if (size > 1)
      comm.all_reduce_in_place_n(ImPi.data(), ImPi.size(), std::plus<>{});
  }
  const double dt2 = sec_since(t2);

  // ----------------------------------------------------------------
  // Step 3: Re Pi via batched Hilbert transform on the bosonic grid.
  // Distributed over iq, allreduced.
  // ----------------------------------------------------------------
  const auto t3 = t_now();
  {
    const int rank = comm.rank();
    const int size = comm.size();
    nda::array<double, 3> ImPi_PQ_O(Naux, Naux, N_O);
    nda::array<double, 3> RePi_PQ_O(Naux, Naux, N_O);
    for (long iq = rank; iq < Nq; iq += size) {
      if (iq == iq_gamma) continue;
      auto ImPi_view = ImPi(iq, _, _, _);
      for (long P = 0; P < Naux; ++P)
        for (long Q = 0; Q < Naux; ++Q)
          for (long iO = 0; iO < N_O; ++iO)
            ImPi_PQ_O(P, Q, iO) = ImPi_view(P, Q, iO).real();
      RePi_from_ImPi(conv, ImPi_PQ_O, RePi_PQ_O);
      auto RePi_view = RePi(iq, _, _, _);
      for (long P = 0; P < Naux; ++P)
        for (long Q = 0; Q < Naux; ++Q)
          for (long iO = 0; iO < N_O; ++iO)
            RePi_view(P, Q, iO) = ComplexType(RePi_PQ_O(P, Q, iO), 0.0);
    }
    if (size > 1)
      comm.all_reduce_in_place_n(RePi.data(), RePi.size(), std::plus<>{});
  }
  const double dt3 = sec_since(t3);

  // ----------------------------------------------------------------
  // Step 4: Solve Dyson W per (q, Omega). Distributed over iq, allreduced.
  // ----------------------------------------------------------------
  const auto t4 = t_now();
  {
    const int rank = comm.rank();
    const int size = comm.size();
    nda::array<ComplexType, 2> Vmat(Naux, Naux);
    nda::array<ComplexType, 2> Pi(Naux, Naux);
    nda::array<ComplexType, 2> W(Naux, Naux);
    for (long iq = rank; iq < Nq; iq += size) {
      // ignore_g0: leave W at iq_gamma exactly zero (consistent with Pi=0
      // there). The Sigma kernel skips iq_gamma anyway, so this is
      // semantically equivalent to leaving W = V_bare at gamma -- but the
      // explicit zero gives a cleaner contract for downstream consumers
      // (e.g. eps_inv_head, divergence corrections).
      if (iq == iq_gamma) continue;
      for (long P = 0; P < Naux; ++P)
        for (long Q = 0; Q < Naux; ++Q)
          Vmat(P, Q) = V(iq, P, Q);
      for (long iO = 0; iO < N_O; ++iO) {
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q)
            Pi(P, Q) = ComplexType(RePi(iq, P, Q, iO).real(),
                                   ImPi(iq, P, Q, iO).real());
        solve_dyson_W_aux(Vmat, Pi, W);
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q) {
            ReW(iq, P, Q, iO) = ComplexType(W(P, Q).real(), 0.0);
            ImW(iq, P, Q, iO) = ComplexType(W(P, Q).imag(), 0.0);
          }
      }
    }
    if (size > 1) {
      comm.all_reduce_in_place_n(ReW.data(), ReW.size(), std::plus<>{});
      comm.all_reduce_in_place_n(ImW.data(), ImW.size(), std::plus<>{});
    }
  }
  const double dt4 = sec_since(t4);

  if (verbose and comm.root()) {
    const double dt_total = sec_since(t_total);
    app_log(2, "[real_axis_scr_coulomb::update_w] Naux={}, N_w={}, N_O={}, "
                "Nk={}, Nq={}, ns={}, nbnd={}",
            Naux, N_w, N_O, Nk, Nq, ns, nbnd);
    app_log(2, "[real_axis_scr_coulomb::update_w]   conv_t setup     : {0:8.3f}", dt_conv);
    app_log(2, "[real_axis_scr_coulomb::update_w]   step 1 project A : {0:8.3f}", dt1);
    app_log(2, "[real_axis_scr_coulomb::update_w]   step 2 Im Pi     : {0:8.3f}", dt2);
    app_log(2, "[real_axis_scr_coulomb::update_w]   step 3 Re Pi (Hi): {0:8.3f}", dt3);
    app_log(2, "[real_axis_scr_coulomb::update_w]   step 4 Dyson W   : {0:8.3f}", dt4);
    app_log(2, "[real_axis_scr_coulomb::update_w]   TOTAL            : {0:8.3f}", dt_total);
  }
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_SCR_COULOMB_T_H
