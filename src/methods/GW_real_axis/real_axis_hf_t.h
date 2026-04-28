/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_HF_T_H
#define COQUI_REAL_AXIS_HF_T_H

#include <string>
#include <utility>

#include "configuration.hpp"
#include "numerics/shared_array/nda.hpp"
#include "nda/nda.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/check.hpp"

#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_mb_state.hpp"
#include "methods/GW_real_axis/real_axis_sigma_x.hpp"

namespace methods {
namespace real_axis {

/**
 * Real-axis Hartree-Fock (static exchange) solver. Wraps
 * `evaluate_Sigma_x_serial` with a state/THC API matching the rest of the
 * real-axis solver classes (`real_axis_scr_coulomb_t`, `real_axis_gw_t`).
 *
 * Mirrors `methods::solvers::hf_t` in role; the signature differs because
 * the real-axis side computes Sigma_x directly from the spectral function
 * A (via the auxiliary-basis density n_aux = X.A.X^dag integrated against
 * f(w)) rather than from a separately-stored density matrix Dm.
 *
 * Reads:  `state.A_wskij`, MF accessors via `thc.MF()`.
 * Writes: `state.Sigma_x_skij`.
 *
 * Configuration:
 *   div_treatment: "ignore_g0" zeroes the q=Gamma contribution.
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
class real_axis_hf_base_t {
public:
  using mpi_communicator_t = boost::mpi3::communicator;

  real_axis_hf_base_t(real_freq_grid_t const* grid,
                      std::string div_treatment = "ignore_g0")
    : _grid(grid),
      _div_treatment(std::move(div_treatment))
  {
    static_assert(MEM == HOST_MEMORY,
                  "real_axis_hf_t<DEVICE>: device path not yet supported in "
                  "evaluate_Sigma_x_serial.");
    utils::check(_grid != nullptr,
                 "real_axis_hf_t: grid pointer must not be null");
  }

  ~real_axis_hf_base_t() = default;

  std::string div_treatment() const { return _div_treatment; }

  /**
   * Evaluate the static-exchange self-energy Sigma_x from state.A_wskij.
   *
   * The mu used for the f(w) integration in `evaluate_Sigma_x_serial` is
   * taken from `mu` (NOT grid->mu_chem()), so the SCF loop can pass the
   * current mu without rebuilding the grid. The MPI communicator is read
   * from state.mpi->comm.
   */
  template<methods::THC_ERI THC_t>
  void evaluate(real_axis_mb_state_t& state,
                THC_t const& thc,
                double mu)
  {
    utils::check(state.grid != nullptr,
                 "real_axis_hf_t::evaluate: state.grid not bound");
    utils::check(state.grid == _grid,
                 "real_axis_hf_t::evaluate: state.grid disagrees with the "
                 "grid the solver was constructed with");
    utils::check(state.A_wskij.has_value(),
                 "real_axis_hf_t::evaluate: state.A_wskij not allocated");
    utils::check(state.mpi != nullptr,
                 "real_axis_hf_t::evaluate: state.mpi not bound");
    auto& comm = state.mpi->comm;

    auto const& grid_in = *_grid;
    auto const& MF      = *thc.MF();
    auto A_in           = state.A_wskij->local();

    const long ns   = MF.nspin();
    const long Nk   = MF.nkpts();
    const long Nq   = MF.nqpts();
    const long nbnd = MF.nbnd();
    const long Naux = thc.Np();
    const long N_w  = grid_in.N_w();

    utils::check(MF.npol() == 1,
                 "real_axis_hf_t::evaluate: npol={} not supported (need 1)",
                 MF.npol());

    // Allocate Sigma_x output sArray (one copy per node).
    if (!state.Sigma_x_skij.has_value())
      state.Sigma_x_skij.emplace(*state.mpi,
          std::array<long, 4>{ns, Nk, nbnd, nbnd});
    if (state.Sigma_x_skij->node_comm()->root())
      state.Sigma_x_skij->local() = ComplexType(0.0, 0.0);
    state.Sigma_x_skij->node_sync();

    // Repack A from (N_w, ns, Nk, nbnd, nbnd) to driver layout, and apply
    // the matrix-hermitian symmetrization that recovers the physical
    // matrix-valued spectral function:
    //
    //   A_phys_{ij} = 0.5 * (A_wskij_{ij} + conj(A_wskij_{ji}))
    //
    // (See the longer comment in real_axis_scr_coulomb_t.h::update_w.)
    nda::array<ComplexType, 5> A_drv(ns, Nk, N_w, nbnd, nbnd);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k)
        for (long iw = 0; iw < N_w; ++iw)
          for (long mu_i = 0; mu_i < nbnd; ++mu_i)
            for (long nu = 0; nu < nbnd; ++nu)
              A_drv(s, k, iw, mu_i, nu) =
                  ComplexType(0.5, 0.0) *
                  (A_in(iw, s, k, mu_i, nu)
                   + std::conj(A_in(iw, s, k, nu, mu_i)));

    // Marshal X, V, kmq from THC. X is moderately large (Naux x nbnd per
    // s, k); put it in shared memory (one copy per node). V has 2 aux
    // indices; marshal only this rank's local (P_loc, Q_loc) block from
    // each thc.Z(iq) -- evaluate_Sigma_x_serial accepts this directly.
    math::shm::shared_array<nda::array_view<ComplexType, 4>>
        sX_skPmu(*state.mpi, {ns, Nk, Naux, nbnd});
    if (sX_skPmu.node_comm()->root()) {
      auto X_loc = sX_skPmu.local();
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k) {
          auto Xsk = thc.X(static_cast<int>(s), 0, static_cast<int>(k));
          for (long P = 0; P < Naux; ++P)
            for (long mu_i = 0; mu_i < nbnd; ++mu_i)
              X_loc(s, k, P, mu_i) = Xsk(P, mu_i);
        }
    }
    sX_skPmu.node_sync();

    // Determine this rank's (P_loc, Q_loc) block via the same proc-grid
    // convention used by evaluate_Sigma_x_serial internally.
    const long nproc = static_cast<long>(comm.size());
    auto pgrid_PQ_hf = real_axis::square_factor_capped(nproc, Naux);
    const long gridP_hf = pgrid_PQ_hf[0];
    const long gridQ_hf = pgrid_PQ_hf[1];
    const long ip_hf = static_cast<long>(comm.rank());
    const long iP_block_hf = (ip_hf / gridQ_hf) % gridP_hf;
    const long iQ_block_hf = ip_hf % gridQ_hf;
    auto chunk_hf = [](long N, long G, long i) {
      long base = N / G;
      long rem  = N % G;
      long start = i * base + std::min(i, rem);
      long sz    = base + (i < rem ? 1 : 0);
      return std::array<long, 2>{start, sz};
    };
    auto [P0_hf, NP_loc_hf] = chunk_hf(Naux, gridP_hf, iP_block_hf);
    auto [Q0_hf, NQ_loc_hf] = chunk_hf(Naux, gridQ_hf, iQ_block_hf);
    nda::array<ComplexType, 3> V_qPQ_loc(Nq, NP_loc_hf, NQ_loc_hf);
    for (long iq = 0; iq < Nq; ++iq) {
      auto Zq = thc.Z(static_cast<int>(iq));
      for (long iP = 0; iP < NP_loc_hf; ++iP)
        for (long iQ = 0; iQ < NQ_loc_hf; ++iQ)
          V_qPQ_loc(iq, iP, iQ) = Zq(P0_hf + iP, Q0_hf + iQ);
    }
    auto X_skPmu = sX_skPmu.local();
    math::shm::shared_array<nda::array_view<long, 2>> skmq(*state.mpi, {Nk, Nq});
    {
      if (skmq.node_comm()->root()) {
        auto kmq_loc = skmq.local();
        auto const& qk_to_k2 = MF.qk_to_k2();
        for (long iq = 0; iq < Nq; ++iq)
          for (long ik = 0; ik < Nk; ++ik)
            kmq_loc(ik, iq) = qk_to_k2(iq, ik);
      }
      skmq.node_sync();
    }
    auto kmq_to_kp = skmq.local();

    long iq_gamma = -1;
    if (_div_treatment == "ignore_g0") {
      auto Qp = MF.Qpts();
      if (Qp.shape()[0] >= 1) {
        double norm0 = 0.0;
        for (long c = 0; c < Qp.shape()[1]; ++c) norm0 += std::abs(Qp(0, c));
        if (norm0 < 1e-10) iq_gamma = 0;
      }
    }

    // Build a grid at the requested mu (Sigma_x integrates the Fermi factor).
    auto grid_at_mu = real_freq_grid_t(grid_in.beta(), mu,
                                       nda::array<double,1>(grid_in.w()),
                                       nda::array<double,1>(grid_in.Omega()),
                                       grid_in.N_t(), grid_in.T_window());

    // evaluate_Sigma_x_serial allreduces Sigma_x to per-rank-replicated;
    // we then copy into the sArray on node root and sync.
    nda::array<ComplexType, 4> Sigma_x_local(ns, Nk, nbnd, nbnd);
    Sigma_x_local() = ComplexType(0.0, 0.0);
    evaluate_Sigma_x_serial(comm, grid_at_mu, A_drv, X_skPmu, V_qPQ_loc,
                            kmq_to_kp, Sigma_x_local, iq_gamma);
    if (state.Sigma_x_skij->node_comm()->root())
      state.Sigma_x_skij->local() = Sigma_x_local;
    state.Sigma_x_skij->node_sync();

    state.mu_chem = mu;
  }

  real_freq_grid_t const& grid() const noexcept { return *_grid; }

private:
  real_freq_grid_t const* _grid;
  std::string             _div_treatment;
};

using real_axis_hf_t = real_axis_hf_base_t<HOST_MEMORY>;

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_HF_T_H
