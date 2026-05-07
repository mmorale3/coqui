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

  // Lazy caches for SCF iterations:
  //   _conv: (cu)FINUFFT plans + Hilbert sgn(t) buffers — built once on
  //          first update_w call when B_loc (=Naux_loc_P*Naux_loc_Q) is
  //          known. Reused across all subsequent SCF iterations. (cu)FINUFFT
  //          plan setup is non-trivial (~10s of ms on host, possibly worse
  //          on device); avoiding 19+ rebuilds is a real wall-clock win.
  //   _cached_B_loc: invalidates _conv if the proc-grid changes between calls.
  //   _sX, _skpq, _sf_Rk, _sf_qR, _V_qPQ_loc: marshaled THC / BZ data that
  //          never changes within an SCF run. Built on first call, reused.
  //   _cached_use_rspace: invalidates the R-space FT caches if the use_rspace
  //          choice changes between calls.
  mutable std::unique_ptr<real_axis::detail::real_axis_conv_base_t<MEM>> _conv;
  mutable long _cached_B_loc = -1;
  mutable bool _cached_use_rspace = false;
  mutable std::optional<math::shm::shared_array<nda::array_view<ComplexType, 4>>> _sX;
  mutable std::optional<math::shm::shared_array<nda::array_view<long, 2>>>        _skpq;
  mutable std::optional<math::shm::shared_array<nda::array_view<ComplexType, 2>>> _sf_Rk;
  mutable std::optional<math::shm::shared_array<nda::array_view<ComplexType, 2>>> _sf_qR;
  mutable std::optional<nda::array<ComplexType, 3>> _V_qPQ_loc;
  // For MEM != HOST_MEMORY, we also keep MEM-side mirrors of the read-only
  // marshaled data (X, V, FT factors). These are re-pushed lazily when the
  // corresponding host cache changes.
  mutable std::optional<memory::array<MEM, ComplexType, 4>> _X_mem;
  mutable std::optional<memory::array<MEM, ComplexType, 3>> _V_qPQ_mem;
  mutable std::optional<memory::array<MEM, ComplexType, 2>> _f_Rk_mem;
  mutable std::optional<memory::array<MEM, ComplexType, 2>> _f_qR_mem;
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
  // The body is MEM-templated. State arrays (state.A_wskij,
  // state.{Im,Re}{Pi,W}_qPQO) live on host (sArrays / dArrays<HOST>);
  // their data is pushed to MEM at the entry boundary and pulled back at
  // the exit boundary. All scratch and intermediate buffers
  // (A_aux_skPQw, ImPi_RPQO_loc, slate matrices, etc.) live in MEM,
  // and the kernels (accumulate_ImPi_one_kq, RePi_from_ImPi, primary_to_aux_one_k,
  // slate_ops::multiply/inverse) all dispatch on MEM. The conv engine
  // (cu)FINUFFT plans are MEM-aware and cached as a class member.
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

  const long ns      = MF.nspin();
  const long nbnd    = MF.nbnd();
  // Storage scopes: orbital-basis state (A, Sigma) at IBZ k, bosonic aux
  // state (Pi, W) at IBZ q. The kernel BZ-pair sums still iterate over FBZ
  // (k, q) pairs (Nk, Nq) — star expansion of orbital quantities happens
  // inside the kernel via X(FBZ k) and the orbital rotations from
  // MF.symmetry_rotation. See feedback_realaxis_thc_x_nonsymmetric.md and
  // notes/realaxis_symmetry_audit.md.
  const long Nk      = MF.nkpts();
  const long Nq      = MF.nqpts();
  const long Nk_ibz  = MF.nkpts_ibz();
  const long Nq_ibz  = MF.nqpts_ibz();
  const long Naux    = thc.Np();
  const long N_w     = grid.N_w();
  const long N_O     = grid.N_Omega();

  utils::check(MF.npol() == 1,
               "real_axis_scr_coulomb_t::update_w: npol={} not supported (need 1)",
               MF.npol());

  auto A_in = state.A_wskij->local();
  utils::check(A_in.shape()[0] == N_w and A_in.shape()[1] == ns and
               A_in.shape()[2] == Nk_ibz and A_in.shape()[3] == nbnd and
               A_in.shape()[4] == nbnd,
               "real_axis_scr_coulomb_t::update_w: state.A_wskij shape mismatch "
               "(expected IBZ k = {}, got {})", Nk_ibz, A_in.shape()[2]);

  // (Re)allocate bosonic state dArrays at IBZ q (distributed over (P, Q)).
  state.allocate_bosonic(Nq_ibz, Naux);

  // Local (P_loc, Q_loc) ranges and shapes from the distributed state.
  auto Pr = state.ImPi_qPQO->local_range(1);
  auto Qr = state.ImPi_qPQO->local_range(2);
  const long Naux_loc_P = Pr.size();
  const long Naux_loc_Q = Qr.size();
  const long B_loc      = Naux_loc_P * Naux_loc_Q;

  // State views (host). All Pi/W work is staged into MEM-side buffers
  // below; we copy back to these host views at the end of update_w.
  auto h_ImPi_loc = state.ImPi_qPQO->local();
  auto h_RePi_loc = state.RePi_qPQO->local();
  auto h_ImW_loc  = state.ImW_qPQO->local();
  auto h_ReW_loc  = state.ReW_qPQO->local();
  // MEM-side staging buffers for Pi/W. For MEM=HOST these are extra
  // copies; for MEM=DEVICE they are the residence of all Pi/W work.
  // buffered_array reuses the static pool so the per-call alloc cost
  // is paid once and amortized across SCF iterations.
  memory::buffered_array<MEM, ComplexType, 4> ImPi_loc(h_ImPi_loc.shape());
  memory::buffered_array<MEM, ComplexType, 4> RePi_loc(h_RePi_loc.shape());
  memory::buffered_array<MEM, ComplexType, 4> ImW_loc (h_ImW_loc.shape());
  memory::buffered_array<MEM, ComplexType, 4> ReW_loc (h_ReW_loc.shape());

  // V(iq, P_loc, Q_loc) from THC.Z(iq) at IBZ q -- only this rank's local
  // block. W lives at IBZ q (Pi at IBZ q, Dyson per-q solve produces W at
  // the same q grid). Cached across SCF iterations since V never changes.
  if (!_V_qPQ_loc.has_value() or
      _V_qPQ_loc->shape() != std::array<long,3>{Nq_ibz, Naux_loc_P, Naux_loc_Q}) {
    _V_qPQ_loc = nda::array<ComplexType, 3>(Nq_ibz, Naux_loc_P, Naux_loc_Q);
    for (long iq = 0; iq < Nq_ibz; ++iq) {
      auto Zq = thc.Z(static_cast<int>(iq));
      for (long iP = 0; iP < Naux_loc_P; ++iP)
        for (long iQ = 0; iQ < Naux_loc_Q; ++iQ)
          (*_V_qPQ_loc)(iq, iP, iQ) = Zq(Pr.first() + iP, Qr.first() + iQ);
    }
  }
  auto& V_qPQ_loc = *_V_qPQ_loc;
  ImPi_loc = ComplexType(0.0, 0.0);
  RePi_loc = ComplexType(0.0, 0.0);
  ImW_loc  = ComplexType(0.0, 0.0);
  ReW_loc  = ComplexType(0.0, 0.0);

  // Repack A from IBZ orbital storage (N_w, ns, Nk_ibz, nbnd, nbnd) to
  // FBZ-k driver layout (ns, Nk, N_w, nbnd, nbnd), symmetrized into the
  // matrix-hermitian physical spectral function:
  //
  //   A_phys_{ij} = 0.5 * (A_wskij_{ij} + conj(A_wskij_{ji}))
  //               = -(1/pi) (Im G^R)^matrix_{ij}
  //
  // state.A_wskij stores -(i/pi) G^R componentwise, which is NOT hermitian
  // off-diagonal. The above symmetrization recovers the matrix-valued
  // hermitian spectral function exactly.
  //
  // The FBZ-k expansion uses kp_to_ibz to look up the IBZ representative.
  // For the symmetry-adapted ISDF (X factor at FBZ k carries the orbital
  // rotation implicitly), this is the correct read. For trivial-IBZ
  // systems kp_to_ibz is identity → bit-identical to the prior behavior.
  // For kp_trev-flagged k (TR-pair partners): we project for the non-TR
  // partner only and fill the TR k's aux block by conjugation of the
  // partner's aux block (post-fix below). The orbital A at TR k positions
  // is computed-but-unused — small wasted work, much simpler than skipping.
  auto kp_to_ibz_arr   = MF.kp_to_ibz();
  auto kp_trev_arr     = MF.kp_trev();
  auto kp_trev_pair_arr = MF.kp_trev_pair();
  memory::array<MEM, ComplexType, 5> A;
  if constexpr (MEM == HOST_MEMORY) {
    A = nda::array<ComplexType, 5>(ns, Nk, N_w, nbnd, nbnd);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k) {
        const long kibz = kp_to_ibz_arr(k);
        for (long iw = 0; iw < N_w; ++iw)
          for (long mu = 0; mu < nbnd; ++mu)
            for (long nu = 0; nu < nbnd; ++nu)
              A(s, k, iw, mu, nu) =
                  ComplexType(0.5, 0.0) *
                  (A_in(iw, s, kibz, mu, nu)
                   + std::conj(A_in(iw, s, kibz, nu, mu)));
      }
  } else {
    nda::array<ComplexType, 5> A_h(ns, Nk, N_w, nbnd, nbnd);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k) {
        const long kibz = kp_to_ibz_arr(k);
        for (long iw = 0; iw < N_w; ++iw)
          for (long mu = 0; mu < nbnd; ++mu)
            for (long nu = 0; nu < nbnd; ++nu)
              A_h(s, k, iw, mu, nu) =
                  ComplexType(0.5, 0.0) *
                  (A_in(iw, s, kibz, mu, nu)
                   + std::conj(A_in(iw, s, kibz, nu, mu)));
      }
    A = memory::to_memory_space<MEM>(A_h);
  }

  // Marshal X(s, k, P, mu) from THC reader into shared memory: one copy
  // per node since X is read-only and moderately large for production.
  // Cached across SCF iterations since X never changes.
  if (!_sX.has_value() or
      _sX->shape() != std::array<long,4>{ns, Nk, Naux, nbnd}) {
    _sX.emplace(*state.mpi, std::array<long,4>{ns, Nk, Naux, nbnd});
    if (_sX->node_comm()->root()) {
      auto X_loc = _sX->local();
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k) {
          auto Xsk = thc.X(static_cast<int>(s), /*ip*/ 0, static_cast<int>(k));
          for (long P = 0; P < Naux; ++P)
            for (long mu = 0; mu < nbnd; ++mu)
              X_loc(s, k, P, mu) = Xsk(P, mu);
        }
    }
    _sX->node_sync();
  }
  auto X = _sX->local();

  // V is marshaled below as a local (Nq, NP_loc, NQ_loc) block once Pr/Qr
  // are known from state.ImPi_qPQO->local_range.

  // BZ closure maps in shared memory. kpq(ik, iq) = ik+iq via qk_to_k2(qminus(iq), ik).
  // Cached across SCF iterations since BZ maps never change.
  // NOTE: this map is sized at FBZ (Nk, Nq) and is currently used only by
  // the k-space kernel branch (Nk == 1). For non-trivial IBZ with Nk > 1
  // we go through the R-space path which doesn't use this map.
  if (!_skpq.has_value() or
      _skpq->shape() != std::array<long,2>{Nk, Nq}) {
    _skpq.emplace(*state.mpi, std::array<long,2>{Nk, Nq});
    if (_skpq->node_comm()->root()) {
      auto kpq_loc = _skpq->local();
      auto const& qk_to_k2 = MF.qk_to_k2();
      auto const& qm       = MF.qminus();
      for (long iq = 0; iq < Nq; ++iq)
        for (long ik = 0; ik < Nk; ++ik)
          kpq_loc(ik, iq) = qk_to_k2(qm(iq), ik);
    }
    _skpq->node_sync();
  }
  auto kpq = _skpq->local();

  // Identify the Gamma q-point in IBZ q indexing if div_treatment ==
  // "ignore_g0". Gamma is conventionally at iq=0 in the IBZ.
  long iq_gamma = -1;
  if (_div_treatment == "ignore_g0") {
    auto Qp = MF.Qpts_ibz();
    if (Qp.shape()[0] >= 1) {
      double norm0 = 0.0;
      for (long c = 0; c < Qp.shape()[1]; ++c) norm0 += std::abs(Qp(0, c));
      if (norm0 < 1e-10) iq_gamma = 0;
    }
  }

  // R-space FT matrices for Pi, when requested. One copy per node.
  // Cached across SCF iterations.
  long NR = 0;
  if (use_rspace and Nk > 1) {
    auto kp_grid = MF.kp_grid();
    NR = kp_grid(0) * kp_grid(1) * kp_grid(2);
    utils::check(NR == Nk,
                 "real_axis_scr_coulomb_t::update_w: R-space path expects NR ({}) "
                 "== Nk ({})", NR, Nk);
    if (!_sf_Rk.has_value() or _cached_use_rspace != use_rspace or
        _sf_Rk->shape() != std::array<long,2>{NR, Nk}) {
      auto lattv   = MF.lattv();
      const long nx = kp_grid(0);
      const long ny = kp_grid(1);
      const long nz = kp_grid(2);
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

      // sf_Rk: FBZ-k -> R FT (full BZ k as input).
      // sf_qR: R -> IBZ-q FT (output Pi lands at IBZ q).
      _sf_Rk.emplace(*state.mpi, std::array<long,2>{NR, Nk});
      _sf_qR.emplace(*state.mpi, std::array<long,2>{Nq_ibz, NR});
      utils::k_to_R_coefficients(comm, Rpts_idx, MF.kpts(), lattv, *_sf_Rk);
      utils::R_to_k_coefficients(comm, Rpts_idx, Rpts_weights,
                                 MF.Qpts_ibz(), lattv, *_sf_qR);
      _cached_use_rspace = use_rspace;
      if constexpr (MEM != HOST_MEMORY) {
        // Push MEM-side mirrors so the gemms in step 2 can run on device.
        _f_Rk_mem = memory::to_memory_space<MEM>(_sf_Rk->local());
        _f_qR_mem = memory::to_memory_space<MEM>(_sf_qR->local());
      }
    }
  }
  const bool do_rspace = (NR > 0);

  // For MEM != HOST_MEMORY, also keep MEM-side mirrors of X and V_qPQ_loc
  // so the per-call kernel work can read them on device. These are
  // re-pushed only when the host caches are invalidated (handled
  // implicitly by checking has_value after the host cache update).
  if constexpr (MEM != HOST_MEMORY) {
    if (!_X_mem.has_value() or
        _X_mem->shape() != std::array<long,4>{ns, Nk, Naux, nbnd}) {
      _X_mem = memory::to_memory_space<MEM>(_sX->local());
    }
    if (!_V_qPQ_mem.has_value() or
        _V_qPQ_mem->shape() != std::array<long,3>{Nq_ibz, Naux_loc_P, Naux_loc_Q}) {
      _V_qPQ_mem = memory::to_memory_space<MEM>(*_V_qPQ_loc);
    }
  }

  // NUFFT engine sized to the local batched cross-correlation
  // (Naux_loc_P * Naux_loc_Q). Each rank only does the FFTs for its
  // (P_loc, Q_loc) block; the full Naux^2 batch is split across ranks.
  // Cached across SCF iterations: (cu)FINUFFT plan setup is non-trivial
  // (~10s of ms host, possibly worse on device), and B_loc only changes
  // when the proc grid does. Invalidate if B_loc changes.
  const auto t_conv0 = t_now();
  if (!_conv or _cached_B_loc != B_loc) {
    _conv = std::make_unique<real_axis::detail::real_axis_conv_base_t<MEM>>(
        grid, /*ntrans*/ B_loc, _eps_nufft);
    _cached_B_loc = B_loc;
  }
  auto& conv = *_conv;
  const double dt_conv = sec_since(t_conv0);

  // ----------------------------------------------------------------
  // Step 1: project A(k) -> A_aux_skPQw with iw innermost. A_aux is
  // distributed over (P, Q) with the SAME proc grid as state.{Im,Re}Pi
  // so the kernel reads/writes line up. Each rank produces only its
  // local (P_loc, Q_loc, N_w) slice via the two-X-views overload of
  // primary_to_aux_one_k.
  // ----------------------------------------------------------------
  const auto t1 = t_now();
  using local_5d_t = memory::array<MEM, ComplexType, 5>;
  auto pgrid_4d = state.ImPi_qPQO->grid();
  auto bsize_4d = state.ImPi_qPQO->block_size();
  std::array<long, 5> pgrid_aux = {1, 1, pgrid_4d[1], pgrid_4d[2], 1};
  std::array<long, 5> bsize_aux = {1, 1, bsize_4d[1], bsize_4d[2], 1};
  std::array<long, 5> shape_aux = {ns, Nk, Naux, Naux, N_w};
  auto dA_aux_skPQw = math::nda::make_distributed_array<local_5d_t>(
      comm, pgrid_aux, shape_aux, bsize_aux);
  auto A_aux_loc = dA_aux_skPQw.local();  // (ns, Nk, Naux_loc_P, Naux_loc_Q, N_w)
  // Pass 1: project for non-TR k's only. Pass 2: fill TR-pair k's by conj-copy.
  if constexpr (MEM == HOST_MEMORY) {
    for (long s = 0; s < ns; ++s)
      for (long ik = 0; ik < Nk; ++ik) {
        if (kp_trev_arr(ik) != 0) continue;
        auto X_P_slice  = X(s, ik, Pr, _);
        auto X_Q_slice  = X(s, ik, Qr, _);
        auto A_view     = A(s, ik, _, _, _);
        auto A_aux_view = A_aux_loc(s, ik, _, _, _);
        primary_to_aux_one_k(X_P_slice, X_Q_slice, A_view, A_aux_view);
      }
  } else {
    auto X_dev = (*_X_mem)();
    for (long s = 0; s < ns; ++s)
      for (long ik = 0; ik < Nk; ++ik) {
        if (kp_trev_arr(ik) != 0) continue;
        auto X_P_slice  = X_dev(s, ik, Pr, _);
        auto X_Q_slice  = X_dev(s, ik, Qr, _);
        auto A_view     = A(s, ik, _, _, _);
        auto A_aux_view = A_aux_loc(s, ik, _, _, _);
        primary_to_aux_one_k<MEM>(X_P_slice, X_Q_slice, A_view, A_aux_view);
      }
  }
  // Pass 2: TR-pair fix-up. aux-A(k) = conj(aux-A(kp_trev_pair(k))) for
  // k where kp_trev != 0. Pattern from rpa_pi.icc:122-135.
  if constexpr (MEM == HOST_MEMORY) {
    for (long s = 0; s < ns; ++s)
      for (long ik = 0; ik < Nk; ++ik) {
        if (kp_trev_arr(ik) == 0) continue;
        const long ik_partner = kp_trev_pair_arr(ik);
        auto src = A_aux_loc(s, ik_partner, _, _, _);
        auto dst = A_aux_loc(s, ik,         _, _, _);
        const long lP = src.shape()[0], lQ = src.shape()[1], lW = src.shape()[2];
        for (long p = 0; p < lP; ++p)
          for (long q = 0; q < lQ; ++q)
            for (long w = 0; w < lW; ++w)
              dst(p, q, w) = std::conj(src(p, q, w));
      }
  } else {
    for (long s = 0; s < ns; ++s)
      for (long ik = 0; ik < Nk; ++ik) {
        if (kp_trev_arr(ik) == 0) continue;
        const long ik_partner = kp_trev_pair_arr(ik);
        auto src = A_aux_loc(s, ik_partner, _, _, _);
        auto dst = A_aux_loc(s, ik,         _, _, _);
        dst = src;
        nda::tensor::scale(ComplexType(1.0, 0.0), dst, nda::tensor::op::CONJ);
      }
  }
  const double dt1 = sec_since(t1);

  // ----------------------------------------------------------------
  // Step 2: Im Pi via spectral cross-correlation. k-space or R-space.
  // ----------------------------------------------------------------
  const auto t2 = t_now();

  // Step 2 distribution model: each rank works on its local (P_loc, Q_loc)
  // slice for ALL (s, k, q). The (P, Q) partitioning replaces the previous
  // (s, k, q) partitioning + allreduce. No comm in Step 2.
  double dt2_alloc = 0.0, dt2_ftR = 0.0, dt2_kernel = 0.0, dt2_ftq = 0.0;
  if (do_rspace) {
    // Use MEM-side R-space FT factors when device; host views otherwise.
    // (Note: nda::blas::gemm dispatches on the array memory space; mixing
    // host f_Rk with device A_aux would not work.)
    auto t2a = t_now();
    memory::buffered_array<MEM, ComplexType, 5> A_aux_sRPQw_loc(
        std::array<long,5>{ns, NR, Naux_loc_P, Naux_loc_Q, N_w});
    memory::buffered_array<MEM, ComplexType, 4> ImPi_RPQO_loc(
        std::array<long,4>{NR, Naux_loc_P, Naux_loc_Q, N_O});
    dt2_alloc += sec_since(t2a);
    auto t2b = t_now();
    for (long s = 0; s < ns; ++s) {
      auto A_in_2D  = nda::reshape(A_aux_loc(s, _, _, _, _),
                                   std::array<long,2>{Nk, B_loc * N_w});
      auto A_out_2D = nda::reshape(A_aux_sRPQw_loc(s, _, _, _, _),
                                   std::array<long,2>{NR, B_loc * N_w});
      if constexpr (MEM == HOST_MEMORY) {
        nda::blas::gemm(ComplexType(1.0, 0.0), _sf_Rk->local(), A_in_2D,
                        ComplexType(0.0, 0.0), A_out_2D);
      } else {
        nda::blas::gemm(ComplexType(1.0, 0.0), *_f_Rk_mem, A_in_2D,
                        ComplexType(0.0, 0.0), A_out_2D);
      }
    }
    dt2_ftR += sec_since(t2b);

    // Per-R cross-correlation, into per-rank local (P_loc, Q_loc, N_O) slice.
    // Kernel reads only LOCAL (P, Q) blocks of A_aux on both legs.
    auto t2c = t_now();
    ImPi_RPQO_loc() = ComplexType(0.0, 0.0);
    for (long iR = 0; iR < NR; ++iR) {
      for (long s = 0; s < ns; ++s) {
        auto A_local     = A_aux_sRPQw_loc(s, iR, _, _, _);
        auto ImPi_R_view = ImPi_RPQO_loc(iR, _, _, _);
        accumulate_ImPi_one_kq<MEM>(conv, A_local, A_local, ImPi_R_view, 1.0);
      }
    }
    dt2_kernel += sec_since(t2c);

    // FT ImPi from R-space to IBZ q-space. Per-rank gemm on the local block.
    auto t2d = t_now();
    {
      auto ImPi_R_2D = nda::reshape(ImPi_RPQO_loc,
                                    std::array<long,2>{NR, B_loc * N_O});
      auto ImPi_q_2D = nda::reshape(ImPi_loc,
                                    std::array<long,2>{Nq_ibz, B_loc * N_O});
      if constexpr (MEM == HOST_MEMORY) {
        nda::blas::gemm(ComplexType(1.0, 0.0), _sf_qR->local(), ImPi_R_2D,
                        ComplexType(0.0, 0.0), ImPi_q_2D);
      } else {
        nda::blas::gemm(ComplexType(1.0, 0.0), *_f_qR_mem, ImPi_R_2D,
                        ComplexType(0.0, 0.0), ImPi_q_2D);
      }
    }

    if (iq_gamma >= 0) {
      ImPi_loc(iq_gamma, nda::ellipsis{}) = ComplexType(0.0, 0.0);
    }
    dt2_ftq += sec_since(t2d);
  } else {
    // K-space branch: only used for Γ-only fixtures (Nk == 1, IBZ == FBZ).
    // For non-trivial IBZ the R-space branch is taken.
    utils::check(Nk == Nq_ibz,
                 "real_axis_scr_coulomb_t::update_w: k-space Pi branch only "
                 "supports IBZ == FBZ (got Nk={}, Nq_ibz={}, Nq={})",
                 Nk, Nq_ibz, Nq);
    const double k_weight = 1.0 / static_cast<double>(Nk);
    for (long iq = 0; iq < Nq_ibz; ++iq) {
      if (iq == iq_gamma) continue;
      auto ImPi_q_view = ImPi_loc(iq, _, _, _);
      for (long s = 0; s < ns; ++s) {
        for (long ik = 0; ik < Nk; ++ik) {
          const long ikq = kpq(ik, iq);
          auto Ak_local  = A_aux_loc(s, ik,  _, _, _);
          auto Akq_local = A_aux_loc(s, ikq, _, _, _);
          accumulate_ImPi_one_kq<MEM>(conv, Ak_local, Akq_local, ImPi_q_view,
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
    memory::buffered_array<MEM, double, 3> ImPi_loc_real(
        std::array<long,3>{Naux_loc_P, Naux_loc_Q, N_O});
    memory::buffered_array<MEM, double, 3> RePi_loc_real(
        std::array<long,3>{Naux_loc_P, Naux_loc_Q, N_O});
    for (long iq = 0; iq < Nq_ibz; ++iq) {
      if (iq == iq_gamma) continue;
      auto ImPi_q_view = ImPi_loc(iq, _, _, _);
      auto RePi_q_view = RePi_loc(iq, _, _, _);
      // Complex -> real lift on (P, Q, O). On host: per-element loop.
      // On device: to_real_view(ImPi_q_view) reinterprets the complex
      // (P, Q, O) view as (P, Q, O, 2) real; index 0 is the real slot.
      if constexpr (MEM == HOST_MEMORY) {
        for (long iP = 0; iP < Naux_loc_P; ++iP)
          for (long iQ = 0; iQ < Naux_loc_Q; ++iQ)
            for (long iO = 0; iO < N_O; ++iO)
              ImPi_loc_real(iP, iQ, iO) = ImPi_q_view(iP, iQ, iO).real();
      } else {
        auto ImPi_real_view = memory::to_real_view(ImPi_q_view);
        ImPi_loc_real() =
            ImPi_real_view(nda::range::all, nda::range::all,
                           nda::range::all, 0);
      }
      RePi_from_ImPi<MEM>(conv, ImPi_loc_real, RePi_loc_real);
      // Real -> complex lift back into RePi_q_view.
      if constexpr (MEM == HOST_MEMORY) {
        for (long iP = 0; iP < Naux_loc_P; ++iP)
          for (long iQ = 0; iQ < Naux_loc_Q; ++iQ)
            for (long iO = 0; iO < N_O; ++iO)
              RePi_q_view(iP, iQ, iO) =
                  ComplexType(RePi_loc_real(iP, iQ, iO), 0.0);
      } else {
        // Zero RePi_q_view first so imag slot is 0, then fill real slot.
        nda::tensor::scale(ComplexType(0.0, 0.0), RePi_q_view);
        auto RePi_real_view = memory::to_real_view(RePi_q_view);
        RePi_real_view(nda::range::all, nda::range::all,
                       nda::range::all, 0) = RePi_loc_real();
      }
    }
  }
  const double dt3 = sec_since(t3);

  // ----------------------------------------------------------------
  // Step 4: Solve Dyson W = (I - V Pi)^{-1} V per (q, Omega) on a Dyson grid
  // that pools (q, Omega) onto distinct ranks so each rank holds the full
  // Naux x Naux matrix locally and slate dispatches to local LAPACK (no
  // distributed inverse comms). Mirrors the imag-axis idiom in
  // scr_coulomb_t::W_omega_proc_grid + scr_coulomb_t::dyson_W_in_place.
  // ----------------------------------------------------------------
  const auto t4 = t_now();
  {
    using Array_2D_t = memory::array<MEM, ComplexType, 2>;
    using Array_4D_t = memory::array<MEM, ComplexType, 4>;
    using math::nda::make_distributed_array;

    // 4a. Pick the Dyson grid: maximize q_pool, then w_pool, leaving any
    // residual to (np_P, np_Q). For nproc=1 (Mac unit tests) this collapses
    // to (1, 1, 1, 1) — bit-identical to the previous single-rank path.
    long np = comm.size();
    long q_pool = utils::find_proc_grid_max_npools(np, Nq_ibz, 0.2);
    long np1    = np / q_pool;
    long w_pool = utils::find_proc_grid_max_npools(np1, N_O, 0.2);
    long np2    = np1 / w_pool;
    long np_P   = utils::find_proc_grid_min_diff(np2, 1, 1);
    long np_Q   = np2 / np_P;
    utils::check(q_pool * w_pool * np_P * np_Q == np,
                 "real_axis_scr_coulomb_t::update_w step 4: Dyson grid does "
                 "not factorize nproc ({} = {} * {} * {} * {})",
                 np, q_pool, w_pool, np_P, np_Q);

    std::array<long, 4> dyson_pgrid = {q_pool, np_P, np_Q, w_pool};
    std::array<long, 4> dyson_bsize;
    dyson_bsize.fill(1);
    dyson_bsize[1] = std::min<long>(1024L, std::max<long>(1L, Naux / np_P));
    dyson_bsize[2] = std::min<long>(1024L, std::max<long>(1L, Naux / np_Q));

    auto pgrid_cur = state.ImPi_qPQO->grid();
    auto bsize_cur = state.ImPi_qPQO->block_size();

    app_log(2, "[real_axis_scr_coulomb::update_w]   Dyson W grid     :"
               " (q, P, Q, O) = ({}, {}, {}, {})",
            dyson_pgrid[0], dyson_pgrid[1], dyson_pgrid[2], dyson_pgrid[3]);

    // 4b. Build complex Pi := RePi.real + i*ImPi.real on the current grid,
    // then redistribute to the Dyson grid. All q-axis sizes are IBZ.
    auto dPi_cur = make_distributed_array<Array_4D_t>(
        comm, pgrid_cur, {Nq_ibz, Naux, Naux, N_O}, bsize_cur);
    {
      auto Pi_cur = dPi_cur.local();
      if constexpr (MEM == HOST_MEMORY) {
        for (long iq = 0; iq < Nq_ibz; ++iq)
          for (long iP = 0; iP < Naux_loc_P; ++iP)
            for (long iQ = 0; iQ < Naux_loc_Q; ++iQ)
              for (long iO = 0; iO < N_O; ++iO)
                Pi_cur(iq, iP, iQ, iO) =
                    ComplexType(RePi_loc(iq, iP, iQ, iO).real(),
                                ImPi_loc(iq, iP, iQ, iO).real());
      } else {
        // Device: alias the buffered ComplexType arrays as (..., 2) real,
        // copy real-slot of RePi into Pi.real, real-slot of ImPi into Pi.imag.
        auto Pi_cur_real = memory::to_real_view(Pi_cur);
        auto RePi_real   = memory::to_real_view(RePi_loc);
        auto ImPi_real   = memory::to_real_view(ImPi_loc);
        Pi_cur_real(_, _, _, _, 0) = RePi_real(_, _, _, _, 0);
        Pi_cur_real(_, _, _, _, 1) = ImPi_real(_, _, _, _, 0);
      }
    }
    auto dPi_dyson = make_distributed_array<Array_4D_t>(
        comm, dyson_pgrid, {Nq_ibz, Naux, Naux, N_O}, dyson_bsize);
    math::nda::redistribute(dPi_cur, dPi_dyson);
    dPi_cur.reset();

    // 4c. V is q-only-distributed; carry a phantom Omega axis of size w_pool
    // so the same 4D Dyson grid can be used (V data is replicated across
    // the w_pool ranks on each (q, P, Q) tile). On the SOURCE side the
    // phantom axis is NOT distributed (grid[3]=1, lshape[3]=w_pool), so
    // each rank fills all w_pool replicas with the same V data; the
    // redistribute then carves them out per Dyson-rank.
    std::array<long, 4> pgrid_V_cur = {1, pgrid_cur[1], pgrid_cur[2], 1};
    std::array<long, 4> bsize_V_cur = {1, bsize_cur[1], bsize_cur[2], 1};
    auto dV_cur = make_distributed_array<Array_4D_t>(
        comm, pgrid_V_cur, {Nq_ibz, Naux, Naux, w_pool}, bsize_V_cur);
    {
      auto V_cur = dV_cur.local();   // shape (Nq_ibz, NP_loc, NQ_loc, w_pool)
      if constexpr (MEM == HOST_MEMORY) {
        for (long iq = 0; iq < Nq_ibz; ++iq)
          for (long iP = 0; iP < Naux_loc_P; ++iP)
            for (long iQ = 0; iQ < Naux_loc_Q; ++iQ)
              for (long iw = 0; iw < w_pool; ++iw)
                V_cur(iq, iP, iQ, iw) = V_qPQ_loc(iq, iP, iQ);
      } else {
        for (long iw = 0; iw < w_pool; ++iw)
          V_cur(_, _, _, iw) = *_V_qPQ_mem;
      }
    }
    std::array<long, 4> dyson_V_pgrid = {q_pool, np_P, np_Q, w_pool};
    std::array<long, 4> dyson_V_bsize = {1, dyson_bsize[1], dyson_bsize[2], 1};
    auto dV_dyson = make_distributed_array<Array_4D_t>(
        comm, dyson_V_pgrid, {Nq_ibz, Naux, Naux, w_pool}, dyson_V_bsize);
    math::nda::redistribute(dV_cur, dV_dyson);
    dV_cur.reset();

    // Output W on the Dyson grid (filled by local Dyson loop; redistributed
    // back at the end).
    auto dW_dyson = make_distributed_array<Array_4D_t>(
        comm, dyson_pgrid, {Nq_ibz, Naux, Naux, N_O}, dyson_bsize);

    // 4d. Slate work arrays on a wq_intra_comm sized (np_P * np_Q). When
    // np_P == np_Q == 1 the intra-comm has size 1 and slate runs as
    // local LAPACK — eliminating the per-(q, Omega) distributed inverse
    // comms that dominate the previous implementation.
    auto Pi_dyson_loc = dPi_dyson.local();
    auto V_dyson_loc  = dV_dyson.local();
    auto W_dyson_loc  = dW_dyson.local();
    auto pgrid_d  = dPi_dyson.grid();
    auto origin_d = dPi_dyson.origin();
    long nq_loc_d = pgrid_d[0] > 0 ? Pi_dyson_loc.shape()[0] : 0;
    long NPlocD   = Pi_dyson_loc.shape()[1];
    long NQlocD   = Pi_dyson_loc.shape()[2];
    long nO_loc_d = pgrid_d[3] > 0 ? Pi_dyson_loc.shape()[3] : 0;
    auto Pr_d = dPi_dyson.local_range(1);
    auto Qr_d = dPi_dyson.local_range(2);

    mpi3::communicator wq_intra_comm =
        comm.split(origin_d[0] * N_O + origin_d[3], comm.rank());
    utils::check(wq_intra_comm.size() == np_P * np_Q,
                 "real_axis_scr_coulomb_t::update_w step 4: wq_intra_comm.size "
                 "({}) != np_P * np_Q ({})",
                 long(wq_intra_comm.size()), np_P * np_Q);

    std::array<long, 2> work_pgrid = {np_P, np_Q};
    std::array<long, 2> work_bsize = {dyson_bsize[1], dyson_bsize[2]};
    auto dV_PQ  = make_distributed_array<Array_2D_t>(
        wq_intra_comm, work_pgrid, {Naux, Naux}, work_bsize, true);
    auto dPi_PQ = make_distributed_array<Array_2D_t>(
        wq_intra_comm, work_pgrid, {Naux, Naux}, work_bsize, true);
    auto dA_PQ  = make_distributed_array<Array_2D_t>(
        wq_intra_comm, work_pgrid, {Naux, Naux}, work_bsize, true);
    auto dW_PQ  = make_distributed_array<Array_2D_t>(
        wq_intra_comm, work_pgrid, {Naux, Naux}, work_bsize, true);
    auto V_PQ  = dV_PQ.local();
    auto Pi_PQ = dPi_PQ.local();
    auto A_PQ  = dA_PQ.local();
    auto W_PQ  = dW_PQ.local();

    // Identity diagonal owned by this rank's local (P, Q) tile.
    std::vector<std::pair<long, long>> diag_idx;
    for (long iP = 0; iP < NPlocD; ++iP) {
      const long P_g = Pr_d.first() + iP;
      for (long iQ = 0; iQ < NQlocD; ++iQ) {
        const long Q_g = Qr_d.first() + iQ;
        if (P_g == Q_g) diag_idx.push_back({iP, iQ});
      }
    }
    [[maybe_unused]] memory::array<MEM, ComplexType, 2> I_local;
    if constexpr (MEM != HOST_MEMORY) {
      nda::array<ComplexType, 2> I_h(NPlocD, NQlocD);
      I_h() = ComplexType(0.0, 0.0);
      for (auto const& d : diag_idx)
        I_h(d.first, d.second) = ComplexType(1.0, 0.0);
      I_local = memory::to_memory_space<MEM>(I_h);
    }

    // 4e. Local Dyson loop on the (q_loc, O_loc) tile.
    for (long iq_loc = 0; iq_loc < nq_loc_d; ++iq_loc) {
      const long iq = origin_d[0] + iq_loc;
      // ignore_g0: leave W(q=0) = 0 (Pi was zeroed in step 2; downstream
      // consumers expect explicit zero rather than a numerical I^{-1}-I).
      if (iq == iq_gamma) {
        for (long iO_loc = 0; iO_loc < nO_loc_d; ++iO_loc)
          W_dyson_loc(iq_loc, _, _, iO_loc) = ComplexType(0.0, 0.0);
        continue;
      }

      V_PQ = V_dyson_loc(iq_loc, _, _, 0);

      for (long iO_loc = 0; iO_loc < nO_loc_d; ++iO_loc) {
        Pi_PQ = Pi_dyson_loc(iq_loc, _, _, iO_loc);

        // A = V * Pi
        math::nda::slate_ops::multiply(ComplexType(1.0, 0.0), dV_PQ, dPi_PQ,
                                       ComplexType(0.0, 0.0), dA_PQ);
        // A = I - V * Pi
        nda::tensor::scale(ComplexType(-1.0, 0.0), A_PQ);
        if constexpr (MEM == HOST_MEMORY) {
          for (auto const& d : diag_idx)
            A_PQ(d.first, d.second) += ComplexType(1.0, 0.0);
        } else {
          nda::tensor::elementwise(ComplexType(1.0, 0.0), I_local, "PQ",
                                   ComplexType(1.0, 0.0), A_PQ,    "PQ",
                                   nda::tensor::op::SUM);
        }
        // A := A^{-1}
        math::nda::slate_ops::inverse(dA_PQ);
        // A -= I
        if constexpr (MEM == HOST_MEMORY) {
          for (auto const& d : diag_idx)
            A_PQ(d.first, d.second) -= ComplexType(1.0, 0.0);
        } else {
          nda::tensor::elementwise(ComplexType(-1.0, 0.0), I_local, "PQ",
                                   ComplexType(1.0, 0.0),  A_PQ,    "PQ",
                                   nda::tensor::op::SUM);
        }
        // W = (A^{-1} - I) * V
        math::nda::slate_ops::multiply(ComplexType(1.0, 0.0), dA_PQ, dV_PQ,
                                       ComplexType(0.0, 0.0), dW_PQ);

        W_dyson_loc(iq_loc, _, _, iO_loc) = W_PQ;
      }
    }

    // 4f. Redistribute W back to the (1, gridP, gridQ, 1) state grid and
    // unpack complex W into ReW_loc.real, ImW_loc.real (the Re/Im arrays
    // store real-valued data in the .real() slot of ComplexType).
    auto dW_cur = make_distributed_array<Array_4D_t>(
        comm, pgrid_cur, {Nq_ibz, Naux, Naux, N_O}, bsize_cur);
    math::nda::redistribute(dW_dyson, dW_cur);
    dPi_dyson.reset();
    dV_dyson.reset();
    dW_dyson.reset();

    {
      auto W_cur = dW_cur.local();
      if constexpr (MEM == HOST_MEMORY) {
        for (long iq = 0; iq < Nq_ibz; ++iq)
          for (long iP = 0; iP < Naux_loc_P; ++iP)
            for (long iQ = 0; iQ < Naux_loc_Q; ++iQ)
              for (long iO = 0; iO < N_O; ++iO) {
                ReW_loc(iq, iP, iQ, iO) =
                    ComplexType(W_cur(iq, iP, iQ, iO).real(), 0.0);
                ImW_loc(iq, iP, iQ, iO) =
                    ComplexType(W_cur(iq, iP, iQ, iO).imag(), 0.0);
              }
      } else {
        auto W_real   = memory::to_real_view(W_cur);
        auto ReW_real = memory::to_real_view(ReW_loc);
        auto ImW_real = memory::to_real_view(ImW_loc);
        nda::tensor::scale(ComplexType(0.0, 0.0), ReW_loc);
        nda::tensor::scale(ComplexType(0.0, 0.0), ImW_loc);
        ReW_real(_, _, _, _, 0) = W_real(_, _, _, _, 0);
        ImW_real(_, _, _, _, 0) = W_real(_, _, _, _, 1);
      }
    }
  }
  const double dt4 = sec_since(t4);

  // ----------------------------------------------------------------
  // Boundary: copy MEM-side staging buffers (ImPi_loc, RePi_loc, ImW_loc,
  // ReW_loc) back to the host state arrays. For MEM=HOST_MEMORY this is
  // a single host->host memcpy on each (essentially the same cost as
  // before since previously kernels wrote directly into these views and
  // there's no extra allocation thanks to buffered_array). For
  // MEM=DEVICE_MEMORY this is one device->host transfer per array. The
  // sizes are (Nq, NP_loc, NQ_loc, N_O) per array; small relative to
  // the kernel work above.
  // ----------------------------------------------------------------
  // State arrays are real-typed (memory-redesign step P0'); take .real()
  // of the complex MEM-side scratch values. By convention the MEM-side
  // buffers carry .imag()=0 throughout, so .real() captures the data.
  {
    auto copy_real = [](auto&& src_complex, auto&& dst_real) {
      const long N = src_complex.size();
      auto const* sp = src_complex.data();
      auto * dp = dst_real.data();
      for (long i = 0; i < N; ++i) dp[i] = sp[i].real();
    };
    if constexpr (MEM == HOST_MEMORY) {
      copy_real(ImPi_loc, h_ImPi_loc);
      copy_real(RePi_loc, h_RePi_loc);
      copy_real(ImW_loc,  h_ImW_loc);
      copy_real(ReW_loc,  h_ReW_loc);
    } else {
      auto ImPi_h = nda::to_host(ImPi_loc);
      auto RePi_h = nda::to_host(RePi_loc);
      auto ImW_h  = nda::to_host(ImW_loc);
      auto ReW_h  = nda::to_host(ReW_loc);
      copy_real(ImPi_h, h_ImPi_loc);
      copy_real(RePi_h, h_RePi_loc);
      copy_real(ImW_h,  h_ImW_loc);
      copy_real(ReW_h,  h_ReW_loc);
    }
  }

  // ----------------------------------------------------------------
  // Step 5: head of eps^-1 at q->0, for the GW divergence correction.
  // Mirrors `g0_div_utils::eps_inv_head_t` on the imag-axis side. Computed
  // unconditionally; gw_t::evaluate only consumes it when div_treatment
  // is not "ignore_g0".
  // Requires the (Nq_ibz, Naux, Naux, N_Omega) W to project through
  // chi_bar_head; gather from the distributed state on each rank since
  // the cost is small (Nq_ibz * N_O scalar accumulations of length Naux^2).
  // TODO: replace gather with a local (P,Q)-contraction + Allreduce of
  // (Nq_ibz, N_O) reals (memory-redesign P6).
  // ----------------------------------------------------------------
  if (_div_treatment != "ignore_g0") {
    auto W_full  = math::nda::all_gather_slow<HOST_MEMORY>(*state.ImW_qPQO);
    auto Re_full = math::nda::all_gather_slow<HOST_MEMORY>(*state.ReW_qPQO);
    // State arrays are real-typed (P0'); combine into complex W for the
    // eps_inv_head computation.
    nda::array<ComplexType, 4> W_complex(W_full.shape());
    {
      const long N = W_complex.size();
      auto * dst = W_complex.data();
      auto const* re = Re_full.data();
      auto const* im = W_full.data();
      for (long i = 0; i < N; ++i)
        dst[i] = ComplexType(re[i], im[i]);
    }
    auto Qpts = nda::array<double, 2>(MF.Qpts_ibz());
    auto chi_bar = nda::array<ComplexType, 2>(thc.basis_bar_head());
    nda::array<ComplexType, 2> eps_inv_qO(Nq_ibz, N_O);
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
    if (do_rspace) {
      app_log(2, "[real_axis_scr_coulomb::update_w]     2a alloc       : {0:8.3f}", dt2_alloc);
      app_log(2, "[real_axis_scr_coulomb::update_w]     2b FT k->R     : {0:8.3f}", dt2_ftR);
      app_log(2, "[real_axis_scr_coulomb::update_w]     2c kernel (NUF): {0:8.3f}", dt2_kernel);
      app_log(2, "[real_axis_scr_coulomb::update_w]     2d FT R->q     : {0:8.3f}", dt2_ftq);
    }
    app_log(2, "[real_axis_scr_coulomb::update_w]   step 3 Re Pi (Hi): {0:8.3f}", dt3);
    app_log(2, "[real_axis_scr_coulomb::update_w]   step 4 Dyson W   : {0:8.3f}", dt4);
    app_log(2, "[real_axis_scr_coulomb::update_w]   TOTAL            : {0:8.3f}", dt_total);
  }
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_SCR_COULOMB_T_H
