/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_DYSON_T_H
#define COQUI_REAL_AXIS_DYSON_T_H

#include <complex>
#include <utility>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"

#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_mb_state.hpp"
#include "methods/GW_real_axis/real_axis_dyson_G.hpp"

namespace methods {
namespace real_axis {

/**
 * Real-axis Dyson solver. Mirrors methods::simple_dyson's role on the
 * imag-axis side: owns the one-body mean-field Hamiltonian H_MF, exposes
 *
 *    solve_dyson(state, mu)        : A(w) = -(1/pi) Im [(w + mu + i eta) I
 *                                          - H_MF - Sigma_x - Sigma^c(w)]^{-1}
 *    find_mu_chem(state, ...)      : bisection on N_elec from state.A_wskij
 *    project_causality(state)      : in-place clip of positive Im Sigma^c diag
 *
 * The H_MF is supplied at construction. For the current periodic-SCF tests
 * H_MF = diag(eps_KS) in the KS basis; a future MF-driven constructor can
 * delegate to hamilt::set_H0 (mirroring simple_dyson).
 *
 * The grid pointer is non-owning; lifetime is the caller's responsibility.
 *
 * Free-function callers (`dyson_G_one_kw`, `dyson_update_A`,
 * `find_mu_chem`, `project_causality_ImSigma`) in
 * `real_axis_dyson_G.hpp` continue to be valid; this class is a thin
 * stateful wrapper that the SCF loop will use.
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
class real_axis_dyson_base_t {
public:
  /// Construct from an explicit H_MF in skij layout.
  real_axis_dyson_base_t(nda::array<ComplexType, 4> H_MF_skij,
                         real_freq_grid_t const* grid,
                         double eta    = 1e-3,
                         double mu_tol = 1e-9)
    : _H_MF(std::move(H_MF_skij)),
      _grid(grid),
      _eta(eta),
      _mu_tol(mu_tol)
  {
    // The dyson math is small per-(s, k, w) matrix solves on host; state
    // arrays it reads/writes (Sigma_x, ImSigma_c, ReSigma_c, A) are
    // sArrays (host node-shared memory) regardless of MEM. The MEM template
    // parameter is therefore a no-op for this class -- it exists only so
    // the SCF driver can be templated uniformly on MEM.
    utils::check(_grid != nullptr,
                 "real_axis_dyson_t: grid pointer must not be null");
    utils::check(_H_MF.shape().size() == 4,
                 "real_axis_dyson_t: H_MF must be (ns, Nk, nbnd, nbnd)");
    utils::check(_H_MF.shape()[2] == _H_MF.shape()[3],
                 "real_axis_dyson_t: H_MF not square in (i, j)");
  }

  ~real_axis_dyson_base_t() = default;

  /**
   * Solve the Dyson equation across all (s, k, w). Reads
   *   state.Sigma_x_skij      (allocated; zeros allowed for G0W0)
   *   state.ImSigma_wskij     (allocated by upstream gw_t::evaluate)
   *   state.ReSigma_wskij     (allocated by upstream gw_t::evaluate)
   * Writes state.A_wskij with the canonical (N_w, ns, Nk, nbnd, nbnd) layout.
   *
   * The chemical potential is taken from `mu` (NOT grid->mu_chem()), so the
   * caller can update mu between SCF iterations without rebuilding the grid.
   */
  void solve_dyson(real_axis_mb_state_t & state, double mu) const
  {
    utils::check(state.grid != nullptr,
                 "real_axis_dyson_t::solve_dyson: state.grid not bound");
    utils::check(state.grid == _grid,
                 "real_axis_dyson_t::solve_dyson: state.grid disagrees with "
                 "the grid the solver was constructed with");
    utils::check(state.Sigma_x_skij.has_value(),
                 "real_axis_dyson_t::solve_dyson: state.Sigma_x_skij not allocated");
    utils::check(state.ImSigma_wskij.has_value() and
                 state.ReSigma_wskij.has_value(),
                 "real_axis_dyson_t::solve_dyson: state.{Im,Re}Sigma_wskij not "
                 "allocated; call real_axis_gw_t::evaluate first");

    auto const& grid = *_grid;
    const long ns   = _H_MF.shape()[0];
    const long Nk   = _H_MF.shape()[1];
    const long nbnd = _H_MF.shape()[2];
    const long N_w  = grid.N_w();

    // Allocate / resize A as sArray (one copy per node).
    if (!state.A_wskij.has_value()
        or state.A_wskij->shape() != std::array<long, 5>{N_w, ns, Nk, nbnd, nbnd}) {
      state.A_wskij.emplace(*state.mpi,
          std::array<long, 5>{N_w, ns, Nk, nbnd, nbnd});
    }

    // Build a temporary grid with the requested mu_chem so dyson_update_A
    // (which reads grid.mu_chem()) sees the right value. Beta and the
    // physical grids are unchanged.
    auto grid_at_mu = real_freq_grid_t(grid.beta(), mu,
                                        nda::array<double,1>(grid.w()),
                                        nda::array<double,1>(grid.Omega()),
                                        grid.N_t(), grid.T_window());

    // Compute A into a per-rank scratch buffer (dyson_update_A is purely
    // local: every rank does the same redundant work). Then write to the
    // node-shared sArray on node-root and node_sync.
    nda::array<ComplexType, 5> A_local(N_w, ns, Nk, nbnd, nbnd);
    nda::array<ComplexType, 4> Sx_local(ns, Nk, nbnd, nbnd);
    nda::array<ComplexType, 5> ReSc_local(N_w, ns, Nk, nbnd, nbnd);
    nda::array<ComplexType, 5> ImSc_local(N_w, ns, Nk, nbnd, nbnd);
    Sx_local()   = state.Sigma_x_skij->local();
    ReSc_local() = state.ReSigma_wskij->local();
    ImSc_local() = state.ImSigma_wskij->local();
    dyson_update_A(grid_at_mu, _H_MF, Sx_local, ReSc_local, ImSc_local,
                   _eta, A_local);
    if (state.A_wskij->node_comm()->root())
      state.A_wskij->local() = A_local;
    state.A_wskij->node_sync();

    state.mu_chem = mu;
  }

  /**
   * Bisection on mu_chem so that
   *   sum_{s,k,m} k_weight(k) * int dw f(w-mu) A_{mm}(s,k,w) = N_elec
   * given the current A in state. The grid carries beta; mu_chem is updated
   * to the new value before integration. Returns the new mu.
   *
   * Note: f(w) is evaluated using grid.beta() and the candidate mu via
   * `real_freq_grid_t::fermi(w, mu, beta)` -- see find_mu_chem in
   * real_axis_dyson_G.hpp. The grid's own mu_chem is not used.
   */
  double find_mu_chem(real_axis_mb_state_t const& state,
                       nda::array<double, 1> const& k_weights,
                       double N_elec,
                       long max_iter = 100) const
  {
    utils::check(state.A_wskij.has_value(),
                 "real_axis_dyson_t::find_mu_chem: state.A_wskij not allocated");
    auto A_loc = state.A_wskij->local();
    return real_axis::find_mu_chem(*_grid, A_loc, k_weights, N_elec,
                                    _mu_tol, max_iter);
  }

  /// In-place causality projection on Im Sigma^c diagonal.
  /// Operates on a (ns, Nk, N_w, nbnd, nbnd) array; the caller is responsible
  /// for repacking to/from state.ImSigma_wskij if needed.
  void project_causality(nda::array<ComplexType, 5> & ImSigma_skwij) const
  {
    project_causality_ImSigma(ImSigma_skwij);
  }

  nda::array<ComplexType, 4> const& H_MF() const noexcept { return _H_MF; }
  real_freq_grid_t const& grid() const noexcept { return *_grid; }
  double eta()    const noexcept { return _eta; }
  double mu_tol() const noexcept { return _mu_tol; }

private:
  nda::array<ComplexType, 4> _H_MF;
  real_freq_grid_t const*    _grid;
  double                     _eta;
  double                     _mu_tol;
};

using real_axis_dyson_t = real_axis_dyson_base_t<HOST_MEMORY>;

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_DYSON_T_H
