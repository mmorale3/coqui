/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_GW_T_H
#define COQUI_REAL_AXIS_GW_T_H

#include <chrono>
#include <string>
#include <memory>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "mpi3/communicator.hpp"
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
#include "methods/GW_real_axis/real_axis_sigma.hpp"
#include "methods/GW_real_axis/real_axis_div_utils.hpp"

namespace methods {
namespace solvers {

/**
 * Finite-temperature real-axis ISDF/THC GW solver.
 *
 * Mirrors the public-API shape of methods::solvers::gw_t but operates on
 * methods::real_axis::real_axis_mb_state_t and a real_freq_grid_t (real
 * frequencies + uniform conjugate time grid + finite-T parameters beta,
 * mu_chem). The single SCF iteration follows Algorithm 1 in the
 * accompanying methods notes (notes/isdf_gw_prb_draft_v2.tex).
 *
 * Self-consistency policy:
 *   - max_iter = 1 (default) corresponds to one-shot G_0 W_0;
 *   - max_iter > 1 with mixing performs scGW.
 *
 * Causality projection is applied at the end of each iteration:
 *   - Im Sigma^{c,R}(w) <- min(Im Sigma^{c,R}(w), 0)
 *   - sgn(Omega) Im Pi^R(Omega) <- min(.,0)
 * before recomputing the corresponding real parts via Hilbert transforms.
 *
 * NOTE on integration with the rest of CoQui:
 *   The full self-consistency loop requires the THC ERI (X, Y, V_PQ) and an
 *   MPI-distributed iteration over k and q points. The standalone class here
 *   provides:
 *     - one-iteration structure (compute Pi -> W -> Sigma_c -> update G)
 *     - causality projection
 *     - hooks for Z (THC) tensor contractions
 *   Connecting the THC factor inputs and the BZ/IBZ machinery is the next
 *   integration step (see TODO markers in the body of the iteration). For
 *   testing purposes the kernels (accumulate_ImPi_one_kq,
 *   accumulate_ImSigma_one_kq, solve_dyson_W_aux) are exercised directly via
 *   their dedicated unit tests.
 */
class real_axis_gw_t {
public:

  /**
   * @param grid        real-frequency grid (finite T, mu_chem, time grid)
   * @param max_iter    maximum number of SCF iterations (1 -> G_0W_0)
   * @param mix         linear mixing coefficient for Sigma updates
   * @param eps_nufft   FINUFFT accuracy tolerance
   * @param ntrans_aux  ntrans of the NUFFT plans, sized to the largest
   *                    Naux*Naux batched cross-correlation.
   * @param output      filename prefix for diagnostic outputs
   */
  real_axis_gw_t(real_axis::real_freq_grid_t const& grid,
                 long max_iter   = 1,
                 double mix      = 0.5,
                 double eps_nufft = 1e-10,
                 long ntrans_aux  = 1,
                 std::string output = "coqui_real_axis")
    : _grid(&grid)
    , _max_iter(max_iter)
    , _mix(mix)
    , _output(std::move(output))
    , _conv(std::make_shared<real_axis::real_axis_conv_t>(
              grid, ntrans_aux, eps_nufft))
  {}

  ~real_axis_gw_t() = default;

  long max_iter() const  { return _max_iter; }
  double mix()    const  { return _mix; }
  std::string const& output() const { return _output; }

  /// Apply causality projection to a fermionic Im Sigma array
  /// (any leading dims, last dim = N_w). Modifies in place.
  template<typename ArrayT>
  void apply_causality_fermionic(ArrayT && Im_w) const {
    // Im Sigma^{c,R}(w) must be <= 0 on the real axis. Clip positive noise.
    auto shape = Im_w.shape();
    auto * data = Im_w.data();
    long total = 1;
    for (long d = 0; d < (long)shape.size(); ++d) total *= shape[d];
    for (long i = 0; i < total; ++i) {
      if (data[i].real() > 0.0)
        data[i] = ComplexType(0.0, data[i].imag());
    }
  }

  /// Apply causality projection to a bosonic Im Pi array (layout (q,P,Q,Omega)).
  /// The diagonal of Im Pi must satisfy sgn(Omega) Im Pi(Omega) <= 0 on the
  /// diagonal; we clip violations.
  template<typename ArrayT>
  void apply_causality_bosonic_diag(ArrayT && ImPi_q_PQ_O,
                                    nda::array<double,1> const& Omega) const {
    auto sh = ImPi_q_PQ_O.shape();
    const long Nq   = sh[0];
    const long Naux = sh[1];
    const long NO   = sh[3];
    for (long iq = 0; iq < Nq; ++iq)
      for (long iO = 0; iO < NO; ++iO) {
        const double sO = (Omega(iO) > 0 ? 1.0 : -1.0);
        for (long P = 0; P < Naux; ++P) {
          auto v = ImPi_q_PQ_O(iq, P, P, iO).real();
          if (sO * v > 0.0)
            ImPi_q_PQ_O(iq, P, P, iO) =
                ComplexType(0.0, ImPi_q_PQ_O(iq, P, P, iO).imag());
        }
      }
  }

  /**
   * Solve the bosonic Dyson equation W = (I - V Pi)^{-1} V across all (q, Omega)
   * grid points using state.RePi_qPQO + i state.ImPi_qPQO as input
   * and writing state.ReW_qPQO + i state.ImW_qPQO as output.
   *
   * @param V_qPQ   (Nq, Naux, Naux) auxiliary Coulomb matrices, complex.
   */
  void solve_W(real_axis::real_axis_mb_state_t & state,
               nda::array<ComplexType, 3> const& V_qPQ) const
  {
    utils::check(state.ImPi_qPQO.has_value() and state.RePi_qPQO.has_value(),
                 "real_axis_gw_t::solve_W: state Pi arrays not allocated");
    utils::check(state.ImW_qPQO.has_value() and state.ReW_qPQO.has_value(),
                 "real_axis_gw_t::solve_W: state W arrays not allocated");

    // Phase 1A.1: gather distributed Pi locally on every rank, compute W
    // redundantly, then each rank writes its own (P_loc, Q_loc) slice
    // back to the distributed state.W arrays. Phase 1A.2 will replace
    // this with a slate_ops Dyson on (P, Q)-distributed inputs.
    auto ImPi = math::nda::all_gather_slow<HOST_MEMORY>(*state.ImPi_qPQO);
    auto RePi = math::nda::all_gather_slow<HOST_MEMORY>(*state.RePi_qPQO);

    const long Nq    = ImPi.shape()[0];
    const long Naux  = ImPi.shape()[1];
    const long NOm   = ImPi.shape()[3];

    nda::array<ComplexType, 4> ImW_full(Nq, Naux, Naux, NOm);
    nda::array<ComplexType, 4> ReW_full(Nq, Naux, Naux, NOm);
    ImW_full() = ComplexType(0.0, 0.0);
    ReW_full() = ComplexType(0.0, 0.0);

    nda::array<ComplexType, 2> Vmat(Naux, Naux);
    nda::array<ComplexType, 2> Pi(Naux, Naux);
    nda::array<ComplexType, 2> W(Naux, Naux);

    for (long iq = 0; iq < Nq; ++iq) {
      for (long P = 0; P < Naux; ++P)
        for (long Q = 0; Q < Naux; ++Q)
          Vmat(P, Q) = V_qPQ(iq, P, Q);
      for (long iO = 0; iO < NOm; ++iO) {
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q)
            Pi(P, Q) = ComplexType(RePi(iq, P, Q, iO).real(),
                                   ImPi(iq, P, Q, iO).real());
        real_axis::solve_dyson_W_aux(Vmat, Pi, W);
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q) {
            ReW_full(iq, P, Q, iO) = ComplexType(W(P, Q).real(), 0.0);
            ImW_full(iq, P, Q, iO) = ComplexType(W(P, Q).imag(), 0.0);
          }
      }
    }

    // Write each rank's local (P_loc, Q_loc) slice into the distributed
    // state arrays.
    auto write_slice = [](nda::array<ComplexType, 4> const& src,
                          real_axis::real_axis_mb_state_t::bosonic_dArray_t& dst) {
      auto Pr = dst.local_range(1);
      auto Qr = dst.local_range(2);
      auto loc = dst.local();
      const long Nq_  = src.shape()[0];
      const long NOm_ = src.shape()[3];
      for (long iq = 0; iq < Nq_; ++iq)
        for (long iP = 0; iP < Pr.size(); ++iP)
          for (long iQ = 0; iQ < Qr.size(); ++iQ)
            for (long iO = 0; iO < NOm_; ++iO)
              loc(iq, iP, iQ, iO) =
                  src(iq, Pr.first() + iP, Qr.first() + iQ, iO);
    };
    write_slice(ImW_full, *state.ImW_qPQO);
    write_slice(ReW_full, *state.ReW_qPQO);
  }

  real_axis::real_axis_conv_t & conv() { return *_conv; }

  real_axis::real_freq_grid_t const& grid() const { return *_grid; }

  /**
   * Evaluate the GW correlation self-energy from a real-axis MBState.
   *
   * Reads `state.A_wskij` (orbital-basis fermionic spectral function) and
   * `state.{Im,Re}W_qPQO` (auxiliary-basis screened interaction; produced
   * upstream by real_axis_scr_coulomb_t::update_w). Writes
   * `state.{Im,Re}Sigma_wskij` (orbital-basis correlation self-energy).
   *
   * Internally:
   *   1. project A from orbital -> aux basis (one BLAS-3 GEMM pair per (s,k))
   *   2. B = -Im W / pi (bosonic spectral function on Omega>=0 grid)
   *   3. Im Sigma_PQ via NUFFT-based fermionic*bosonic convolution
   *      (k-space loop over (s,k,q), or R-space FT k->R + q->R + per-R conv)
   *   4. Re Sigma_PQ via batched Hilbert transform
   *   5. back-project to orbital basis
   *
   * MPI: the convolution loop is rank-distributed and allreduced; result is
   * fully replicated. Mirrors `evaluate_thc_serial` Steps 5-8.
   *
   * @param state       reads A and W; writes Sigma.
   *                    The MPI communicator is read from state.mpi->comm.
   * @param thc         THC ERI (provides X and MF accessors).
   * @param eps_nufft   FINUFFT accuracy for the conv engine.
   * @param div_treatment  "ignore_g0" zeroes iq_gamma in the Sigma sum.
   * @param verbose     emit per-step timings on rank 0.
   * @param use_rspace  when true and Nk>1, run the conv in R-space.
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY,
           methods::THC_ERI THC_t>
  void evaluate(real_axis::real_axis_mb_state_t & state,
                THC_t const& thc,
                double eps_nufft = 1e-10,
                std::string div_treatment = "ignore_g0",
                bool verbose = false,
                bool use_rspace = false);

private:
  real_axis::real_freq_grid_t const* _grid;
  long _max_iter;
  double _mix;
  std::string _output;
  std::shared_ptr<real_axis::real_axis_conv_t> _conv;
  // SCF iteration caches (MEM-keyed via separate slots; only the slot for
  // the actually-used MEM gets populated). Mirrors the pattern used in
  // real_axis_scr_coulomb_t. Lazy-initialized on first evaluate() call
  // to avoid paying the (cu)FINUFFT plan setup + THC marshaling cost on
  // every SCF iteration.
  template<MEMORY_SPACE MEM>
  struct evaluate_caches_t {
    std::unique_ptr<real_axis::detail::real_axis_conv_base_t<MEM>> conv;
    long B_loc = -1;
    bool use_rspace_cached = false;
    std::optional<math::shm::shared_array<nda::array_view<ComplexType, 4>>> sX;
    std::optional<math::shm::shared_array<nda::array_view<long, 2>>>        skmq;
    std::optional<math::shm::shared_array<nda::array_view<ComplexType, 2>>> sf_Rk;
    std::optional<math::shm::shared_array<nda::array_view<ComplexType, 2>>> sf_Rq;
    std::optional<math::shm::shared_array<nda::array_view<ComplexType, 2>>> sf_kR;
    std::optional<memory::array<MEM, ComplexType, 4>> X_mem;
    std::optional<memory::array<MEM, ComplexType, 2>> f_Rk_mem;
    std::optional<memory::array<MEM, ComplexType, 2>> f_Rq_mem;
    std::optional<memory::array<MEM, ComplexType, 2>> f_kR_mem;
  };
  // One cache slot per supported MEM. Adding more is mechanical.
  mutable evaluate_caches_t<HOST_MEMORY>   _ev_cache_host;
  mutable evaluate_caches_t<DEVICE_MEMORY> _ev_cache_dev;

  // Helper: return the cache for the requested MEM.
  template<MEMORY_SPACE MEM>
  evaluate_caches_t<MEM>& evaluate_cache_for() const {
    if constexpr (MEM == HOST_MEMORY)
      return _ev_cache_host;
    else
      return _ev_cache_dev;
  }
};

// --------------------------------------------------------------------------
// real_axis_gw_t::evaluate implementation. Body parallels Steps 5-8 of
// `evaluate_serial`. Header-inline (matches the rest of the real-axis module).
// --------------------------------------------------------------------------
template<MEMORY_SPACE MEM, methods::THC_ERI THC_t>
void real_axis_gw_t::evaluate(real_axis::real_axis_mb_state_t & state,
                              THC_t const& thc,
                              double eps_nufft,
                              std::string div_treatment,
                              bool verbose,
                              bool use_rspace)
{
  using namespace methods::real_axis;
  auto& cache = evaluate_cache_for<MEM>();
  utils::check(state.A_wskij.has_value(),
               "real_axis_gw_t::evaluate: state.A_wskij not allocated");
  utils::check(state.ImW_qPQO.has_value() and state.ReW_qPQO.has_value(),
               "real_axis_gw_t::evaluate: state.{Im,Re}W_qPQO not allocated; "
               "call real_axis_scr_coulomb_t::update_w first");
  utils::check(state.mpi != nullptr,
               "real_axis_gw_t::evaluate: state.mpi not bound");
  auto& comm = state.mpi->comm;
  utils::check(state.grid != nullptr,
               "real_axis_gw_t::evaluate: state.grid not bound");
  utils::check(state.grid == _grid,
               "real_axis_gw_t::evaluate: state.grid disagrees with the grid "
               "the solver was constructed with");

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
               "real_axis_gw_t::evaluate: npol={} not supported (need 1)",
               MF.npol());

  auto A_in = state.A_wskij->local();
  utils::check(A_in.shape()[0] == N_w and A_in.shape()[1] == ns and
               A_in.shape()[2] == Nk and A_in.shape()[3] == nbnd and
               A_in.shape()[4] == nbnd,
               "real_axis_gw_t::evaluate: state.A_wskij shape mismatch");
  utils::check(state.ImW_qPQO->global_shape()[0] == Nq and
               state.ImW_qPQO->global_shape()[1] == Naux and
               state.ImW_qPQO->global_shape()[2] == Naux and
               state.ImW_qPQO->global_shape()[3] == N_O,
               "real_axis_gw_t::evaluate: state.ImW_qPQO shape mismatch");

  // Local (P_loc, Q_loc) ranges and shapes from the distributed state.W.
  auto Pr = state.ImW_qPQO->local_range(1);
  auto Qr = state.ImW_qPQO->local_range(2);
  const long Naux_loc_P = Pr.size();
  const long Naux_loc_Q = Qr.size();
  const long B_loc      = Naux_loc_P * Naux_loc_Q;
  // Read state.W directly via .local(); no gather, no replicated buffer.
  auto ImW_loc = state.ImW_qPQO->local();

  // Allocate Sigma outputs in state as sArrays (one copy per node) if
  // not already allocated.
  if (!state.ImSigma_wskij.has_value() or !state.ReSigma_wskij.has_value()) {
    state.ImSigma_wskij.emplace(*state.mpi,
        std::array<long, 5>{N_w, ns, Nk, nbnd, nbnd});
    state.ReSigma_wskij.emplace(*state.mpi,
        std::array<long, 5>{N_w, ns, Nk, nbnd, nbnd});
  }
  auto ImSigma_out = state.ImSigma_wskij->local();
  auto ReSigma_out = state.ReSigma_wskij->local();

  // Repack A from (N_w, ns, Nk, nbnd, nbnd) to (ns, Nk, N_w, nbnd, nbnd),
  // taking the matrix-hermitian symmetrization. For MEM=HOST we fill the
  // MEM-resident A directly; for MEM=DEVICE we fill a host scratch and
  // push (one H2D transfer per call).
  memory::array<MEM, ComplexType, 5> A;
  if constexpr (MEM == HOST_MEMORY) {
    A = nda::array<ComplexType, 5>(ns, Nk, N_w, nbnd, nbnd);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k)
        for (long iw = 0; iw < N_w; ++iw)
          for (long mu = 0; mu < nbnd; ++mu)
            for (long nu = 0; nu < nbnd; ++nu)
              A(s, k, iw, mu, nu) =
                  ComplexType(0.5, 0.0) *
                  (A_in(iw, s, k, mu, nu)
                   + std::conj(A_in(iw, s, k, nu, mu)));
  } else {
    nda::array<ComplexType, 5> A_h(ns, Nk, N_w, nbnd, nbnd);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k)
        for (long iw = 0; iw < N_w; ++iw)
          for (long mu = 0; mu < nbnd; ++mu)
            for (long nu = 0; nu < nbnd; ++nu)
              A_h(s, k, iw, mu, nu) =
                  ComplexType(0.5, 0.0) *
                  (A_in(iw, s, k, mu, nu)
                   + std::conj(A_in(iw, s, k, nu, mu)));
    A = memory::to_memory_space<MEM>(A_h);
  }

  // Marshal X(s, k, P, mu) into shared memory: one copy per node.
  // Cached across SCF iterations.
  if (!cache.sX.has_value() or
      cache.sX->shape() != std::array<long,4>{ns, Nk, Naux, nbnd}) {
    cache.sX.emplace(*state.mpi, std::array<long,4>{ns, Nk, Naux, nbnd});
    if (cache.sX->node_comm()->root()) {
      auto X_loc = cache.sX->local();
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k) {
          auto Xsk = thc.X(static_cast<int>(s), /*ip*/ 0, static_cast<int>(k));
          for (long P = 0; P < Naux; ++P)
            for (long mu = 0; mu < nbnd; ++mu)
              X_loc(s, k, P, mu) = Xsk(P, mu);
        }
    }
    cache.sX->node_sync();
    if constexpr (MEM != HOST_MEMORY)
      cache.X_mem = memory::to_memory_space<MEM>(cache.sX->local());
  }
  auto X = cache.sX->local();

  // BZ closure: kmq(ik, iq) = ik - iq, in shared memory. Cached.
  if (!cache.skmq.has_value() or
      cache.skmq->shape() != std::array<long,2>{Nk, Nq}) {
    cache.skmq.emplace(*state.mpi, std::array<long,2>{Nk, Nq});
    if (cache.skmq->node_comm()->root()) {
      auto kmq_loc = cache.skmq->local();
      auto const& qk_to_k2 = MF.qk_to_k2();
      for (long iq = 0; iq < Nq; ++iq)
        for (long ik = 0; ik < Nk; ++ik)
          kmq_loc(ik, iq) = qk_to_k2(iq, ik);
    }
    cache.skmq->node_sync();
  }
  auto kmq = cache.skmq->local();

  // q-mesh weights (uniform 1/Nq).
  nda::array<double, 1> qw(Nq);
  for (long iq = 0; iq < Nq; ++iq) qw(iq) = 1.0 / static_cast<double>(Nq);

  // Identify Gamma.
  long iq_gamma = -1;
  if (div_treatment == "ignore_g0") {
    auto Qp = MF.Qpts();
    if (Qp.shape()[0] >= 1) {
      double norm0 = 0.0;
      for (long c = 0; c < Qp.shape()[1]; ++c) norm0 += std::abs(Qp(0, c));
      if (norm0 < 1e-10) iq_gamma = 0;
    }
  }

  // R-space FT matrices for Sigma. One copy per node. Cached.
  long NR = 0;
  if (use_rspace and Nk > 1) {
    auto kp_grid = MF.kp_grid();
    NR = kp_grid(0) * kp_grid(1) * kp_grid(2);
    utils::check(NR == Nk,
                 "real_axis_gw_t::evaluate: R-space path expects NR ({}) == Nk ({})",
                 NR, Nk);
    if (!cache.sf_Rk.has_value() or cache.use_rspace_cached != use_rspace or
        cache.sf_Rk->shape() != std::array<long,2>{NR, Nk}) {
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

      cache.sf_Rk.emplace(*state.mpi, std::array<long,2>{NR, Nk});
      cache.sf_Rq.emplace(*state.mpi, std::array<long,2>{NR, Nq});
      cache.sf_kR.emplace(*state.mpi, std::array<long,2>{Nk, NR});
      utils::k_to_R_coefficients(comm, Rpts_idx, MF.kpts(), lattv, *cache.sf_Rk);
      utils::k_to_R_coefficients(comm, Rpts_idx, MF.Qpts(), lattv, *cache.sf_Rq);
      utils::R_to_k_coefficients(comm, Rpts_idx, Rpts_weights, MF.kpts(), lattv,
                                 *cache.sf_kR);
      cache.use_rspace_cached = use_rspace;
      if constexpr (MEM != HOST_MEMORY) {
        cache.f_Rk_mem = memory::to_memory_space<MEM>(cache.sf_Rk->local());
        cache.f_Rq_mem = memory::to_memory_space<MEM>(cache.sf_Rq->local());
        cache.f_kR_mem = memory::to_memory_space<MEM>(cache.sf_kR->local());
      }
    }
  }
  const bool do_rspace = (NR > 0);

  // NUFFT engine sized to the local (P_loc, Q_loc) block. Cached.
  const auto t_conv0 = t_now();
  if (!cache.conv or cache.B_loc != B_loc) {
    cache.conv = std::make_unique<detail::real_axis_conv_base_t<MEM>>(
        grid, /*ntrans*/ B_loc, eps_nufft);
    cache.B_loc = B_loc;
  }
  auto& conv = *cache.conv;
  const double dt_conv = sec_since(t_conv0);

  // ----------------------------------------------------------------
  // Step 1: project A(k) -> A_aux distributed over (P, Q). Same proc
  // grid as state.W so .local() blocks line up.
  // ----------------------------------------------------------------
  const auto t1 = t_now();
  using local_5d_t = memory::array<MEM, ComplexType, 5>;
  auto pgrid_4d = state.ImW_qPQO->grid();
  auto bsize_4d = state.ImW_qPQO->block_size();
  std::array<long, 5> pgrid_aux = {1, 1, pgrid_4d[1], pgrid_4d[2], 1};
  std::array<long, 5> bsize_aux = {1, 1, bsize_4d[1], bsize_4d[2], 1};
  std::array<long, 5> shape_aux = {ns, Nk, Naux, Naux, N_w};
  auto dA_aux_skPQw = math::nda::make_distributed_array<local_5d_t>(
      comm, pgrid_aux, shape_aux, bsize_aux);
  auto A_aux_loc = dA_aux_skPQw.local();
  if constexpr (MEM == HOST_MEMORY) {
    for (long s = 0; s < ns; ++s)
      for (long ik = 0; ik < Nk; ++ik) {
        auto X_P_slice  = X(s, ik, Pr, _);
        auto X_Q_slice  = X(s, ik, Qr, _);
        auto A_view     = A(s, ik, _, _, _);
        auto A_aux_view = A_aux_loc(s, ik, _, _, _);
        primary_to_aux_one_k(X_P_slice, X_Q_slice, A_view, A_aux_view);
      }
  } else {
    auto X_dev = (*cache.X_mem)();
    for (long s = 0; s < ns; ++s)
      for (long ik = 0; ik < Nk; ++ik) {
        auto X_P_slice  = X_dev(s, ik, Pr, _);
        auto X_Q_slice  = X_dev(s, ik, Qr, _);
        auto A_view     = A(s, ik, _, _, _);
        auto A_aux_view = A_aux_loc(s, ik, _, _, _);
        primary_to_aux_one_k<MEM>(X_P_slice, X_Q_slice, A_view, A_aux_view);
      }
  }
  const double dt1 = sec_since(t1);

  // ----------------------------------------------------------------
  // Step 5: B = -Im W / pi on the local (P_loc, Q_loc, Omega) block.
  // For MEM=DEVICE, ImW_loc is host-resident (state lives on host); we
  // pull to MEM via to_memory_space and apply the scale on device.
  // ----------------------------------------------------------------
  const auto t5 = t_now();
  memory::buffered_array<MEM, ComplexType, 4> B_qPQO_loc(
      std::array<long,4>{Nq, Naux_loc_P, Naux_loc_Q, N_O});
  if constexpr (MEM == HOST_MEMORY) {
    B_qPQO_loc() = nda::map([](ComplexType w) {
      return ComplexType(-w.real() / M_PI, 0.0);
    })(ImW_loc);
  } else {
    // Push host ImW_loc to MEM, then scale by -1/pi on MEM. ImW_loc per
    // convention has imag()=0; B inherits that, then scale acts on both
    // slots (which is correct since 0 * anything = 0).
    nda::array<ComplexType, 4> ImW_h(ImW_loc.shape());
    ImW_h = ImW_loc;
    auto ImW_mem = memory::to_memory_space<MEM>(ImW_h);
    B_qPQO_loc = ImW_mem;  // memcpy in MEM
    nda::tensor::scale(ComplexType(-1.0 / M_PI, 0.0), B_qPQO_loc);
  }
  const double dt5 = sec_since(t5);

  // ----------------------------------------------------------------
  // Step 6: Im Sigma^c_PQ(k, w). Each rank computes its (P_loc, Q_loc)
  // block for ALL (s, k, q); no allreduce.
  // ----------------------------------------------------------------
  const auto t6 = t_now();
  memory::buffered_array<MEM, ComplexType, 5> ImSigma_aux_skPQw_loc(
      std::array<long,5>{ns, Nk, Naux_loc_P, Naux_loc_Q, N_w});
  ImSigma_aux_skPQw_loc() = ComplexType(0.0, 0.0);

  if (do_rspace) {
    if (iq_gamma >= 0) {
      B_qPQO_loc(iq_gamma, nda::ellipsis{}) = ComplexType(0.0, 0.0);
    }

    // FT B from q-space to R-space (per-rank on local block).
    memory::buffered_array<MEM, ComplexType, 4> B_RPQO_loc(
        std::array<long,4>{NR, Naux_loc_P, Naux_loc_Q, N_O});
    {
      auto B_q_2D = nda::reshape(B_qPQO_loc,
                                 std::array<long,2>{Nq, B_loc * N_O});
      auto B_R_2D = nda::reshape(B_RPQO_loc,
                                 std::array<long,2>{NR, B_loc * N_O});
      if constexpr (MEM == HOST_MEMORY) {
        nda::blas::gemm(ComplexType(1.0, 0.0), cache.sf_Rq->local(), B_q_2D,
                        ComplexType(0.0, 0.0), B_R_2D);
      } else {
        nda::blas::gemm(ComplexType(1.0, 0.0), *cache.f_Rq_mem, B_q_2D,
                        ComplexType(0.0, 0.0), B_R_2D);
      }
    }

    // FT A from k-space to R-space (per-rank on local block).
    memory::buffered_array<MEM, ComplexType, 5> A_aux_sRPQw_loc(
        std::array<long,5>{ns, NR, Naux_loc_P, Naux_loc_Q, N_w});
    for (long s = 0; s < ns; ++s) {
      auto A_in_2D  = nda::reshape(A_aux_loc(s, _, _, _, _),
                                   std::array<long,2>{Nk, B_loc * N_w});
      auto A_out_2D = nda::reshape(A_aux_sRPQw_loc(s, _, _, _, _),
                                   std::array<long,2>{NR, B_loc * N_w});
      if constexpr (MEM == HOST_MEMORY) {
        nda::blas::gemm(ComplexType(1.0, 0.0), cache.sf_Rk->local(), A_in_2D,
                        ComplexType(0.0, 0.0), A_out_2D);
      } else {
        nda::blas::gemm(ComplexType(1.0, 0.0), *cache.f_Rk_mem, A_in_2D,
                        ComplexType(0.0, 0.0), A_out_2D);
      }
    }

    // Per-R Sigma kernel on local (P_loc, Q_loc) blocks. No allreduce.
    memory::buffered_array<MEM, ComplexType, 5> ImSigma_aux_sRPQw_loc(
        std::array<long,5>{ns, NR, Naux_loc_P, Naux_loc_Q, N_w});
    ImSigma_aux_sRPQw_loc() = ComplexType(0.0, 0.0);
    for (long iR = 0; iR < NR; ++iR) {
      for (long s = 0; s < ns; ++s) {
        auto A_view   = A_aux_sRPQw_loc(s, iR, _, _, _);
        auto B_view   = B_RPQO_loc(iR, _, _, _);
        auto Sig_view = ImSigma_aux_sRPQw_loc(s, iR, _, _, _);
        accumulate_ImSigma_one_kq_nufft<MEM>(conv, A_view, B_view, Sig_view, 1.0);
      }
    }

    // FT Sigma from R-space to k-space (per-rank on local block).
    for (long s = 0; s < ns; ++s) {
      auto Sig_R_2D = nda::reshape(ImSigma_aux_sRPQw_loc(s, _, _, _, _),
                                   std::array<long,2>{NR, B_loc * N_w});
      auto Sig_k_2D = nda::reshape(ImSigma_aux_skPQw_loc(s, _, _, _, _),
                                   std::array<long,2>{Nk, B_loc * N_w});
      if constexpr (MEM == HOST_MEMORY) {
        nda::blas::gemm(ComplexType(1.0, 0.0), cache.sf_kR->local(), Sig_R_2D,
                        ComplexType(0.0, 0.0), Sig_k_2D);
      } else {
        nda::blas::gemm(ComplexType(1.0, 0.0), *cache.f_kR_mem, Sig_R_2D,
                        ComplexType(0.0, 0.0), Sig_k_2D);
      }
    }
  } else {
    for (long s = 0; s < ns; ++s) {
      for (long ik = 0; ik < Nk; ++ik) {
        for (long iq = 0; iq < Nq; ++iq) {
          if (iq == iq_gamma) continue;
          const long ikmq = kmq(ik, iq);
          auto A_view   = A_aux_loc(s, ikmq, _, _, _);
          auto B_view   = B_qPQO_loc(iq, _, _, _);
          auto Sig_view = ImSigma_aux_skPQw_loc(s, ik, _, _, _);
          accumulate_ImSigma_one_kq_nufft<MEM>(conv, A_view, B_view, Sig_view, qw(iq));
        }
      }
    }
  }
  const double dt6 = sec_since(t6);

  // ----------------------------------------------------------------
  // Step 7: Re Sigma^c via batched Hilbert transform (per-rank on local block).
  // ----------------------------------------------------------------
  const auto t7 = t_now();
  memory::buffered_array<MEM, ComplexType, 5> ReSigma_aux_skPQw_loc(
      std::array<long,5>{ns, Nk, Naux_loc_P, Naux_loc_Q, N_w});
  for (long s = 0; s < ns; ++s)
    for (long ik = 0; ik < Nk; ++ik) {
      auto Im_view = ImSigma_aux_skPQw_loc(s, ik, _, _, _);
      auto Re_view = ReSigma_aux_skPQw_loc(s, ik, _, _, _);
      ReSigma_from_ImSigma_aux<MEM>(conv, Im_view, Re_view);
    }
  const double dt7 = sec_since(t7);

  // ----------------------------------------------------------------
  // Step 8: back-project Sigma_PQ_loc(k, w) -> Sigma_munu(k, w).
  // Each rank computes a PARTIAL contribution from its (P_loc, Q_loc)
  // block. The full orbital-basis Sigma is the sum across (P, Q) ranks,
  // obtained via a single allreduce on the orbital-shape Sigma at the end.
  // ----------------------------------------------------------------
  const auto t8 = t_now();
  memory::buffered_array<MEM, ComplexType, 5> ImSigma_mem(
      std::array<long,5>{ns, Nk, N_w, nbnd, nbnd});
  memory::buffered_array<MEM, ComplexType, 5> ReSigma_mem(
      std::array<long,5>{ns, Nk, N_w, nbnd, nbnd});
  ImSigma_mem() = ComplexType(0.0, 0.0);
  ReSigma_mem() = ComplexType(0.0, 0.0);
  if constexpr (MEM == HOST_MEMORY) {
    for (long s = 0; s < ns; ++s)
      for (long ik = 0; ik < Nk; ++ik) {
        auto X_P_slice = X(s, ik, Pr, _);
        auto X_Q_slice = X(s, ik, Qr, _);
        auto Im_view = ImSigma_aux_skPQw_loc(s, ik, _, _, _);
        auto Re_view = ReSigma_aux_skPQw_loc(s, ik, _, _, _);
        auto ImOut   = ImSigma_mem(s, ik, _, _, _);
        auto ReOut   = ReSigma_mem(s, ik, _, _, _);
        aux_to_primary_one_k(X_P_slice, X_Q_slice, Im_view, ImOut);
        aux_to_primary_one_k(X_P_slice, X_Q_slice, Re_view, ReOut);
      }
  } else {
    auto X_dev = (*cache.X_mem)();
    for (long s = 0; s < ns; ++s)
      for (long ik = 0; ik < Nk; ++ik) {
        auto X_P_slice = X_dev(s, ik, Pr, _);
        auto X_Q_slice = X_dev(s, ik, Qr, _);
        auto Im_view = ImSigma_aux_skPQw_loc(s, ik, _, _, _);
        auto Re_view = ReSigma_aux_skPQw_loc(s, ik, _, _, _);
        auto ImOut   = ImSigma_mem(s, ik, _, _, _);
        auto ReOut   = ReSigma_mem(s, ik, _, _, _);
        aux_to_primary_one_k<MEM>(X_P_slice, X_Q_slice, Im_view, ImOut);
        aux_to_primary_one_k<MEM>(X_P_slice, X_Q_slice, Re_view, ReOut);
      }
  }
  // Pull MEM-side Sigmas to host for the allreduce + state write.
  // (The allreduce is over the global comm; orbital basis is small.)
  nda::array<ComplexType, 5> ImSigma, ReSigma;
  if constexpr (MEM == HOST_MEMORY) {
    ImSigma = ImSigma_mem;
    ReSigma = ReSigma_mem;
  } else {
    ImSigma = nda::to_host(ImSigma_mem);
    ReSigma = nda::to_host(ReSigma_mem);
  }
  // Allreduce the partial sums across (P, Q) ranks to get the full Sigma.
  if (comm.size() > 1) {
    comm.all_reduce_in_place_n(ImSigma.data(), ImSigma.size(), std::plus<>{});
    comm.all_reduce_in_place_n(ReSigma.data(), ReSigma.size(), std::plus<>{});
  }

  // Repack into state.{Im,Re}Sigma_wskij with (N_w, ns, Nk, nbnd, nbnd) layout.
  // sArray writes only on node-root (else races on shared memory).
  if (state.ImSigma_wskij->node_comm()->root()) {
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k)
        for (long iw = 0; iw < N_w; ++iw)
          for (long mu = 0; mu < nbnd; ++mu)
            for (long nu = 0; nu < nbnd; ++nu) {
              ImSigma_out(iw, s, k, mu, nu) = ImSigma(s, k, iw, mu, nu);
              ReSigma_out(iw, s, k, mu, nu) = ReSigma(s, k, iw, mu, nu);
            }
  }
  state.ImSigma_wskij->node_sync();
  state.ReSigma_wskij->node_sync();
  const double dt8 = sec_since(t8);

  // ----------------------------------------------------------------
  // Step 9: Sigma^head divergence correction (Gygi-Baldereschi style).
  // Mirrors gw_t::Sigma_div_correction on the imag-axis side. Only applied
  // when scr_coulomb_t::update_w has populated state.eps_inv_head_O
  // (i.e. div_treatment != "ignore_g0").
  //
  // Direct quadrature over (s, k, w, w'), per-rank local nda::array
  // scratch; then write back into state.{Im,Re}Sigma_wskij on node-root +
  // node_sync so the sArrays remain consistent.
  // ----------------------------------------------------------------
  const auto t9 = t_now();
  if (state.eps_inv_head_O.has_value() and div_treatment != "ignore_g0") {
    if constexpr (MEM == HOST_MEMORY) {
      const double madelung = MF.madelung();
      auto chi_head_qP = thc.basis_head();    // (Nq, Naux), local copy
      nda::array<ComplexType, 2> chi_head_buf(chi_head_qP);  // contiguous local

      // Snapshot the orbital-basis Sigma into per-rank scratch buffers
      // (so we can call the in-place updater locally and node-write the
      // result via the standard pattern).
      nda::array<ComplexType, 5> ImSig_local(N_w, ns, Nk, nbnd, nbnd);
      nda::array<ComplexType, 5> ReSig_local(N_w, ns, Nk, nbnd, nbnd);
      ImSig_local() = state.ImSigma_wskij->local();
      ReSig_local() = state.ReSigma_wskij->local();

      nda::array<ComplexType, 5> A_buf(state.A_wskij->local().shape());
      A_buf() = state.A_wskij->local();

      real_axis::apply_sigma_head_correction_real_axis(
          conv, grid, madelung, *state.eps_inv_head_O,
          chi_head_buf, X, A_buf,
          ImSig_local, ReSig_local);

      if (state.ImSigma_wskij->node_comm()->root()) {
        state.ImSigma_wskij->local() = ImSig_local;
        state.ReSigma_wskij->local() = ReSig_local;
      }
      state.ImSigma_wskij->node_sync();
      state.ReSigma_wskij->node_sync();
    } else {
      // Step 9 div correction not yet device-aware; would require a
      // device-resident chi_head and an apply_sigma_head_correction
      // variant on MEM. The standard "ignore_g0" path skips this entirely.
      utils::check(false,
                   "real_axis_gw_t::evaluate<DEVICE>: Sigma_div correction "
                   "(div_treatment != \"ignore_g0\") is host-only. Use "
                   "ignore_g0 or instantiate the host path.");
    }
  }
  const double dt9 = sec_since(t9);

  if (verbose and comm.root()) {
    const double dt_total = sec_since(t_total);
    app_log(2, "[real_axis_gw_t::evaluate] Naux={}, N_w={}, N_O={}, "
                "Nk={}, Nq={}, ns={}, nbnd={}",
            Naux, N_w, N_O, Nk, Nq, ns, nbnd);
    app_log(2, "[real_axis_gw_t::evaluate]   conv_t setup     : {0:8.3f}", dt_conv);
    app_log(2, "[real_axis_gw_t::evaluate]   step 1 project A : {0:8.3f}", dt1);
    app_log(2, "[real_axis_gw_t::evaluate]   step 5 B=-ImW/pi : {0:8.3f}", dt5);
    app_log(2, "[real_axis_gw_t::evaluate]   step 6 Im Sigma  : {0:8.3f}", dt6);
    app_log(2, "[real_axis_gw_t::evaluate]   step 7 Re Sig (H): {0:8.3f}", dt7);
    app_log(2, "[real_axis_gw_t::evaluate]   step 8 backproj  : {0:8.3f}", dt8);
    app_log(2, "[real_axis_gw_t::evaluate]   step 9 div corr  : {0:8.3f}", dt9);
    app_log(2, "[real_axis_gw_t::evaluate]   TOTAL            : {0:8.3f}", dt_total);
  }
}

} // namespace solvers
} // namespace methods

#endif // COQUI_REAL_AXIS_GW_T_H
