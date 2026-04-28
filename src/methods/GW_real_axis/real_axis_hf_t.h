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
   * current mu without rebuilding the grid.
   */
  template<methods::THC_ERI THC_t>
  void evaluate(mpi_communicator_t& comm,
                real_axis_mb_state_t& state,
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

    auto const& grid_in = *_grid;
    auto const& MF      = *thc.MF();
    auto const& A_in    = *state.A_wskij;

    const long ns   = MF.nspin();
    const long Nk   = MF.nkpts();
    const long Nq   = MF.nqpts();
    const long nbnd = MF.nbnd();
    const long Naux = thc.Np();
    const long N_w  = grid_in.N_w();

    utils::check(MF.npol() == 1,
                 "real_axis_hf_t::evaluate: npol={} not supported (need 1)",
                 MF.npol());

    // Allocate Sigma_x output.
    if (!state.Sigma_x_skij.has_value())
      state.Sigma_x_skij = nda::array<ComplexType, 4>(ns, Nk, nbnd, nbnd);
    *state.Sigma_x_skij = ComplexType(0.0, 0.0);

    // Repack A from (N_w, ns, Nk, nbnd, nbnd) to driver layout.
    nda::array<ComplexType, 5> A_drv(ns, Nk, N_w, nbnd, nbnd);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k)
        for (long iw = 0; iw < N_w; ++iw)
          for (long mu_i = 0; mu_i < nbnd; ++mu_i)
            for (long nu = 0; nu < nbnd; ++nu)
              A_drv(s, k, iw, mu_i, nu) =
                  ComplexType(A_in(iw, s, k, mu_i, nu).real(), 0.0);

    // Marshal X, V, kmq from THC.
    nda::array<ComplexType, 4> X_skPmu(ns, Nk, Naux, nbnd);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k) {
        auto Xsk = thc.X(static_cast<int>(s), 0, static_cast<int>(k));
        for (long P = 0; P < Naux; ++P)
          for (long mu_i = 0; mu_i < nbnd; ++mu_i)
            X_skPmu(s, k, P, mu_i) = Xsk(P, mu_i);
      }
    nda::array<ComplexType, 3> V_qPQ(Nq, Naux, Naux);
    for (long iq = 0; iq < Nq; ++iq) {
      auto Zq = thc.Z(static_cast<int>(iq));
      for (long P = 0; P < Naux; ++P)
        for (long Q = 0; Q < Naux; ++Q)
          V_qPQ(iq, P, Q) = Zq(P, Q);
    }
    nda::array<long, 2> kmq_to_kp(Nk, Nq);
    auto const& qk_to_k2 = MF.qk_to_k2();
    for (long iq = 0; iq < Nq; ++iq)
      for (long ik = 0; ik < Nk; ++ik)
        kmq_to_kp(ik, iq) = qk_to_k2(iq, ik);

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

    evaluate_Sigma_x_serial(comm, grid_at_mu, A_drv, X_skPmu, V_qPQ,
                            kmq_to_kp, *state.Sigma_x_skij, iq_gamma);

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
