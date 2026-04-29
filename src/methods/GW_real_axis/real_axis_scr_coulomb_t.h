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
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/nda_utils.hpp"
#include "numerics/distributed_array/slate_ops.hpp"

#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_conv.hpp"
#include "methods/GW_real_axis/real_axis_mb_state.hpp"
#include "methods/GW_real_axis/real_axis_thc_project.hpp"
#include "methods/GW_real_axis/real_axis_pi.hpp"
#include "methods/GW_real_axis/real_axis_dyson.hpp"
#include "methods/GW_real_axis/real_axis_div_utils.hpp"

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

  auto A_in = state.A_wskij->local();
  utils::check(A_in.shape()[0] == N_w and A_in.shape()[1] == ns and
               A_in.shape()[2] == Nk and A_in.shape()[3] == nbnd and
               A_in.shape()[4] == nbnd,
               "real_axis_scr_coulomb_t::update_w: state.A_wskij shape mismatch");

  // (Re)allocate bosonic state dArrays (distributed over (P, Q)).
  state.allocate_bosonic(Nq, Naux);

  // Local (P_loc, Q_loc) ranges and shapes from the distributed state.
  auto Pr = state.ImPi_qPQO->local_range(1);
  auto Qr = state.ImPi_qPQO->local_range(2);
  const long Naux_loc_P = Pr.size();
  const long Naux_loc_Q = Qr.size();
  const long B_loc      = Naux_loc_P * Naux_loc_Q;
  auto ImPi_loc = state.ImPi_qPQO->local();
  auto RePi_loc = state.RePi_qPQO->local();
  auto ImW_loc  = state.ImW_qPQO->local();
  auto ReW_loc  = state.ReW_qPQO->local();

  // V(iq, P_loc, Q_loc) from THC.Z(iq) -- only this rank's local block.
  nda::array<ComplexType, 3> V_qPQ_loc(Nq, Naux_loc_P, Naux_loc_Q);
  for (long iq = 0; iq < Nq; ++iq) {
    auto Zq = thc.Z(static_cast<int>(iq));
    for (long iP = 0; iP < Naux_loc_P; ++iP)
      for (long iQ = 0; iQ < Naux_loc_Q; ++iQ)
        V_qPQ_loc(iq, iP, iQ) = Zq(Pr.first() + iP, Qr.first() + iQ);
  }
  ImPi_loc = ComplexType(0.0, 0.0);
  RePi_loc = ComplexType(0.0, 0.0);
  ImW_loc  = ComplexType(0.0, 0.0);
  ReW_loc  = ComplexType(0.0, 0.0);

  // Repack A from (N_w, ns, nkpts, nbnd, nbnd) to driver layout
  // (ns, nkpts, N_w, nbnd, nbnd) and symmetrize into the matrix-hermitian
  // physical spectral function:
  //
  //   A_phys_{ij} = 0.5 * (A_wskij_{ij} + conj(A_wskij_{ji}))
  //               = -(1/pi) (Im G^R)^matrix_{ij}
  //
  // state.A_wskij stores -(i/pi) G^R componentwise, which is NOT hermitian
  // off-diagonal. The above symmetrization recovers the matrix-valued
  // hermitian spectral function exactly. Diagonals are unchanged
  // (.imag(A_wskij_ii) = 0 for the spectral-function-on-diagonal); off-
  // diagonals pull from both Re and Im of A_wskij. After this, A_aux is
  // hermitian in (P, Q), and Pi/Sigma kernels see a physically correct
  // (matrix-hermitian) input.
  nda::array<ComplexType, 5> A(ns, Nk, N_w, nbnd, nbnd);
  for (long s = 0; s < ns; ++s)
    for (long k = 0; k < Nk; ++k)
      for (long iw = 0; iw < N_w; ++iw)
        for (long mu = 0; mu < nbnd; ++mu)
          for (long nu = 0; nu < nbnd; ++nu)
            A(s, k, iw, mu, nu) =
                ComplexType(0.5, 0.0) *
                (A_in(iw, s, k, mu, nu)
                 + std::conj(A_in(iw, s, k, nu, mu)));

  // Marshal X(s, k, P, mu) from THC reader into shared memory: one copy
  // per node since X is read-only and moderately large for production
  // (Naux * nbnd) per (s, k).
  math::shm::shared_array<nda::array_view<ComplexType, 4>>
      sX(*state.mpi, {ns, Nk, Naux, nbnd});
  if (sX.node_comm()->root()) {
    auto X_loc = sX.local();
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k) {
        auto Xsk = thc.X(static_cast<int>(s), /*ip*/ 0, static_cast<int>(k));
        for (long P = 0; P < Naux; ++P)
          for (long mu = 0; mu < nbnd; ++mu)
            X_loc(s, k, P, mu) = Xsk(P, mu);
      }
  }
  sX.node_sync();
  auto X = sX.local();

  // V is marshaled below as a local (Nq, NP_loc, NQ_loc) block once Pr/Qr
  // are known from state.ImPi_qPQO->local_range.

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

  // NUFFT engine sized to the local batched cross-correlation
  // (Naux_loc_P * Naux_loc_Q). Each rank only does the FFTs for its
  // (P_loc, Q_loc) block; the full Naux^2 batch is split across ranks.
  const auto t_conv0 = t_now();
  real_axis_conv_t conv(grid, /*ntrans*/ B_loc, _eps_nufft);
  const double dt_conv = sec_since(t_conv0);

  // ----------------------------------------------------------------
  // Step 1: project A(k) -> A_aux_skPQw with iw innermost. A_aux is
  // distributed over (P, Q) with the SAME proc grid as state.{Im,Re}Pi
  // so the kernel reads/writes line up. Each rank produces only its
  // local (P_loc, Q_loc, N_w) slice via the two-X-views overload of
  // primary_to_aux_one_k.
  // ----------------------------------------------------------------
  const auto t1 = t_now();
  using local_5d_t = memory::array<HOST_MEMORY, ComplexType, 5>;
  auto pgrid_4d = state.ImPi_qPQO->grid();
  auto bsize_4d = state.ImPi_qPQO->block_size();
  std::array<long, 5> pgrid_aux = {1, 1, pgrid_4d[1], pgrid_4d[2], 1};
  std::array<long, 5> bsize_aux = {1, 1, bsize_4d[1], bsize_4d[2], 1};
  std::array<long, 5> shape_aux = {ns, Nk, Naux, Naux, N_w};
  auto dA_aux_skPQw = math::nda::make_distributed_array<local_5d_t>(
      comm, pgrid_aux, shape_aux, bsize_aux);
  auto A_aux_loc = dA_aux_skPQw.local();  // (ns, Nk, Naux_loc_P, Naux_loc_Q, N_w)
  for (long s = 0; s < ns; ++s)
    for (long ik = 0; ik < Nk; ++ik) {
      auto X_P_slice  = X(s, ik, Pr, _);
      auto X_Q_slice  = X(s, ik, Qr, _);
      auto A_view     = A(s, ik, _, _, _);
      auto A_aux_view = A_aux_loc(s, ik, _, _, _);
      primary_to_aux_one_k(X_P_slice, X_Q_slice, A_view, A_aux_view);
    }
  const double dt1 = sec_since(t1);

  // ----------------------------------------------------------------
  // Step 2: Im Pi via spectral cross-correlation. k-space or R-space.
  // ----------------------------------------------------------------
  const auto t2 = t_now();

  // Step 2 distribution model: each rank works on its local (P_loc, Q_loc)
  // slice for ALL (s, k, q). The (P, Q) partitioning replaces the previous
  // (s, k, q) partitioning + allreduce. No comm in Step 2.
  if (do_rspace) {
    auto f_Rk = sf_Rk_opt->local();
    auto f_qR = sf_qR_opt->local();

    // FT A_aux from k-space to R-space, per s. Each rank does its OWN
    // local (P_loc, Q_loc, N_w) block: a single gemm of f_Rk by the
    // (Nk, B_loc * N_w) reshape of the rank's local A_aux slice. No comm.
    nda::array<ComplexType, 5> A_aux_sRPQw_loc(ns, NR, Naux_loc_P, Naux_loc_Q, N_w);
    for (long s = 0; s < ns; ++s) {
      auto A_in_2D  = nda::reshape(A_aux_loc(s, _, _, _, _),
                                   std::array<long,2>{Nk, B_loc * N_w});
      auto A_out_2D = nda::reshape(A_aux_sRPQw_loc(s, _, _, _, _),
                                   std::array<long,2>{NR, B_loc * N_w});
      nda::blas::gemm(ComplexType(1.0, 0.0), f_Rk, A_in_2D,
                      ComplexType(0.0, 0.0), A_out_2D);
    }

    // Per-R cross-correlation, into per-rank local (P_loc, Q_loc, N_O) slice.
    // Kernel reads only LOCAL (P, Q) blocks of A_aux on both legs.
    nda::array<ComplexType, 4> ImPi_RPQO_loc(NR, Naux_loc_P, Naux_loc_Q, N_O);
    ImPi_RPQO_loc = ComplexType(0.0, 0.0);
    for (long iR = 0; iR < NR; ++iR) {
      for (long s = 0; s < ns; ++s) {
        auto A_local     = A_aux_sRPQw_loc(s, iR, _, _, _);
        auto ImPi_R_view = ImPi_RPQO_loc(iR, _, _, _);
        accumulate_ImPi_one_kq(conv, A_local, A_local, ImPi_R_view, 1.0);
      }
    }

    // FT ImPi from R-space to q-space. Per-rank gemm on the local block;
    // f_qR is replicated, contracts over R.
    {
      auto ImPi_R_2D = nda::reshape(ImPi_RPQO_loc,
                                    std::array<long,2>{NR, B_loc * N_O});
      auto ImPi_q_2D = nda::reshape(ImPi_loc,
                                    std::array<long,2>{Nq, B_loc * N_O});
      nda::blas::gemm(ComplexType(1.0, 0.0), f_qR, ImPi_R_2D,
                      ComplexType(0.0, 0.0), ImPi_q_2D);
    }

    if (iq_gamma >= 0) {
      ImPi_loc(iq_gamma, nda::ellipsis{}) = ComplexType(0.0, 0.0);
    }
  } else {
    const double k_weight = 1.0 / static_cast<double>(Nk);
    for (long iq = 0; iq < Nq; ++iq) {
      if (iq == iq_gamma) continue;
      auto ImPi_q_view = ImPi_loc(iq, _, _, _);
      for (long s = 0; s < ns; ++s) {
        for (long ik = 0; ik < Nk; ++ik) {
          const long ikq = kpq(ik, iq);
          auto Ak_local  = A_aux_loc(s, ik,  _, _, _);
          auto Akq_local = A_aux_loc(s, ikq, _, _, _);
          accumulate_ImPi_one_kq(conv, Ak_local, Akq_local, ImPi_q_view,
                                 k_weight);
        }
      }
    }
  }
  const double dt2 = sec_since(t2);

  // ----------------------------------------------------------------
  // Step 3: Re Pi via batched Hilbert transform on the bosonic grid.
  // Each rank Hilbert-transforms its own (P_loc, Q_loc, N_O) slice for
  // every iq. No comm.
  // ----------------------------------------------------------------
  const auto t3 = t_now();
  {
    nda::array<double, 3> ImPi_loc_real(Naux_loc_P, Naux_loc_Q, N_O);
    nda::array<double, 3> RePi_loc_real(Naux_loc_P, Naux_loc_Q, N_O);
    for (long iq = 0; iq < Nq; ++iq) {
      if (iq == iq_gamma) continue;
      auto ImPi_q_view = ImPi_loc(iq, _, _, _);
      auto RePi_q_view = RePi_loc(iq, _, _, _);
      for (long iP = 0; iP < Naux_loc_P; ++iP)
        for (long iQ = 0; iQ < Naux_loc_Q; ++iQ)
          for (long iO = 0; iO < N_O; ++iO)
            ImPi_loc_real(iP, iQ, iO) = ImPi_q_view(iP, iQ, iO).real();
      RePi_from_ImPi(conv, ImPi_loc_real, RePi_loc_real);
      for (long iP = 0; iP < Naux_loc_P; ++iP)
        for (long iQ = 0; iQ < Naux_loc_Q; ++iQ)
          for (long iO = 0; iO < N_O; ++iO)
            RePi_q_view(iP, iQ, iO) =
                ComplexType(RePi_loc_real(iP, iQ, iO), 0.0);
    }
  }
  const double dt3 = sec_since(t3);

  // ----------------------------------------------------------------
  // Step 4: Solve Dyson W = (I - V Pi)^{-1} V per (q, Omega) via slate_ops
  // on (P, Q)-distributed matrices. The same proc grid as state.{Im,Re}Pi
  // is used so .local() blocks line up.
  // ----------------------------------------------------------------
  const auto t4 = t_now();
  {
    using Array_2D_t = memory::array<HOST_MEMORY, ComplexType, 2>;
    auto pgrid_full = state.ImPi_qPQO->grid();
    auto bsize_full = state.ImPi_qPQO->block_size();
    std::array<long, 2> pgrid_PQ = {pgrid_full[1], pgrid_full[2]};
    std::array<long, 2> bsize_PQ = {bsize_full[1], bsize_full[2]};
    auto dV_PQ  = math::nda::make_distributed_array<Array_2D_t>(
        comm, pgrid_PQ, {Naux, Naux}, bsize_PQ, true);
    auto dPi_PQ = math::nda::make_distributed_array<Array_2D_t>(
        comm, pgrid_PQ, {Naux, Naux}, bsize_PQ, true);
    auto dA_PQ  = math::nda::make_distributed_array<Array_2D_t>(
        comm, pgrid_PQ, {Naux, Naux}, bsize_PQ, true);
    auto dW_PQ  = math::nda::make_distributed_array<Array_2D_t>(
        comm, pgrid_PQ, {Naux, Naux}, bsize_PQ, true);
    auto V_loc  = dV_PQ.local();
    auto Pi_loc = dPi_PQ.local();
    auto A_loc  = dA_PQ.local();
    auto W_loc  = dW_PQ.local();

    // Identity-diagonal entries owned by this rank.
    std::vector<std::pair<long, long>> diag_idx;
    for (long iP = 0; iP < Naux_loc_P; ++iP) {
      const long P_g = Pr.first() + iP;
      for (long iQ = 0; iQ < Naux_loc_Q; ++iQ) {
        const long Q_g = Qr.first() + iQ;
        if (P_g == Q_g) diag_idx.push_back({iP, iQ});
      }
    }

    for (long iq = 0; iq < Nq; ++iq) {
      // ignore_g0: leave W at iq_gamma exactly zero (Pi was zeroed there
      // already). Σ kernel skips iq_gamma; explicit zero is a cleaner
      // contract for downstream consumers (eps_inv_head, divergence corrections).
      if (iq == iq_gamma) continue;

      // Fill V_loc (the slate-distributed scratch) and Pi_loc from the
      // local V_qPQ_loc and distributed state.Pi (both already local).
      for (long iP = 0; iP < Naux_loc_P; ++iP)
        for (long iQ = 0; iQ < Naux_loc_Q; ++iQ)
          V_loc(iP, iQ) = V_qPQ_loc(iq, iP, iQ);

      for (long iO = 0; iO < N_O; ++iO) {
        for (long iP = 0; iP < Naux_loc_P; ++iP)
          for (long iQ = 0; iQ < Naux_loc_Q; ++iQ)
            Pi_loc(iP, iQ) = ComplexType(RePi_loc(iq, iP, iQ, iO).real(),
                                         ImPi_loc(iq, iP, iQ, iO).real());

        // A = V * Pi
        math::nda::slate_ops::multiply(ComplexType(1.0, 0.0), dV_PQ, dPi_PQ,
                                       ComplexType(0.0, 0.0), dA_PQ);
        // A = -A
        A_loc *= ComplexType(-1.0, 0.0);
        // A = I - V*Pi (add identity to diagonal entries owned by this rank)
        for (auto const& d : diag_idx)
          A_loc(d.first, d.second) += ComplexType(1.0, 0.0);
        // A := A^{-1} = (I - V*Pi)^{-1}
        math::nda::slate_ops::inverse(dA_PQ);
        // A -= I
        for (auto const& d : diag_idx)
          A_loc(d.first, d.second) -= ComplexType(1.0, 0.0);
        // W = ((I - V*Pi)^{-1} - I) * V
        math::nda::slate_ops::multiply(ComplexType(1.0, 0.0), dA_PQ, dV_PQ,
                                       ComplexType(0.0, 0.0), dW_PQ);

        // Write into state.{Im,Re}W at this (iq, iO).
        for (long iP = 0; iP < Naux_loc_P; ++iP)
          for (long iQ = 0; iQ < Naux_loc_Q; ++iQ) {
            ReW_loc(iq, iP, iQ, iO) = ComplexType(W_loc(iP, iQ).real(), 0.0);
            ImW_loc(iq, iP, iQ, iO) = ComplexType(W_loc(iP, iQ).imag(), 0.0);
          }
      }
    }
  }
  const double dt4 = sec_since(t4);

  // ----------------------------------------------------------------
  // Step 5: head of eps^-1 at q->0, for the GW divergence correction.
  // Mirrors `g0_div_utils::eps_inv_head_t` on the imag-axis side. Computed
  // unconditionally; gw_t::evaluate only consumes it when div_treatment
  // is not "ignore_g0".
  // Requires the full (Nq, Naux, Naux, N_Omega) W to project through
  // chi_bar_head; gather from the distributed state on each rank since
  // the cost is small (Nq * N_O scalar accumulations of length Naux^2).
  // ----------------------------------------------------------------
  if (_div_treatment != "ignore_g0") {
    auto W_full = math::nda::all_gather_slow<HOST_MEMORY>(*state.ImW_qPQO);
    auto Re_full = math::nda::all_gather_slow<HOST_MEMORY>(*state.ReW_qPQO);
    // Combine Re + i Im into the complex W viewed by compute_eps_inv_head_O.
    // (Re_full and W_full have the same shape; same with ImW_qPQO.)
    nda::array<ComplexType, 4> W_complex(W_full.shape());
    {
      const long N = W_complex.size();
      auto * dst = W_complex.data();
      auto const* re = Re_full.data();
      auto const* im = W_full.data();
      for (long i = 0; i < N; ++i)
        dst[i] = ComplexType(re[i].real(), im[i].real());
    }
    auto Qpts = nda::array<double, 2>(MF.Qpts());
    auto chi_bar = nda::array<ComplexType, 2>(thc.basis_bar_head());
    nda::array<ComplexType, 2> eps_inv_qO(Nq, N_O);
    if (!state.eps_inv_head_O.has_value())
      state.eps_inv_head_O = nda::array<ComplexType, 1>(N_O);
    compute_eps_inv_head_O(W_complex, Qpts, chi_bar, MF.volume(),
                           eps_inv_qO, *state.eps_inv_head_O);
  }

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
