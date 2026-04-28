/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 * ==========================================================================
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "utilities/test_common.hpp"
#include "numerics/distributed_array/nda_utils.hpp"
#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_mb_state.hpp"
#include "methods/GW_real_axis/real_axis_gw_t.h"

#include <complex>

namespace gw_real_axis_tests
{

using methods::real_axis::real_freq_grid_t;
using methods::real_axis::real_axis_mb_state_t;
using methods::solvers::real_axis_gw_t;

// =============================================================================
// End-to-end smoke test: instantiate the solver, allocate a state, solve W
// from a synthetic Pi, and verify outputs are finite. Exercises the public
// API and the full Pi -> W pipeline at the SCF-loop level.
// =============================================================================
TEST_CASE("real_axis_gw_endtoend_smoke", "[real_axis][gw][e2e]")
{
  const double w_max     = 8.0;
  const long   N_w       = 65;
  const double Omega_max = 4.0;
  const long   N_Omega   = 32;
  const long   N_t       = 128;
  const double T_window  = 16.0;
  const double beta = 50.0;
  const double mu   = 0.0;

  auto grid = real_freq_grid_t::make_uniform(
      beta, mu, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

  // Create solver. Naux=2 chosen so the W inversion is non-trivial.
  const long Naux = 2;
  real_axis_gw_t gw(grid,
                    /*max_iter*/ 1,
                    /*mix*/      0.5,
                    /*eps_nufft*/ 1e-10,
                    /*ntrans*/   Naux*Naux,
                    /*output*/   "test_real_axis_gw");

  // Set up state. State.mpi must be bound for the dArray allocation.
  auto& mpi_context = utils::make_unit_test_mpi_context();
  const long Nq = 1;
  real_axis_mb_state_t state(grid);
  state.mpi = mpi_context;
  state.allocate_bosonic(Nq, Naux);

  // Synthetic, causal Im Pi: small negative diagonal contribution. Build
  // the deterministic full array on every rank, then write each rank's
  // local (P_loc, Q_loc) slice into the distributed state.
  nda::array<std::complex<double>, 4> ImPi_full(Nq, Naux, Naux, N_Omega);
  nda::array<std::complex<double>, 4> RePi_full(Nq, Naux, Naux, N_Omega);
  ImPi_full() = std::complex<double>(0.0, 0.0);
  RePi_full() = std::complex<double>(0.0, 0.0);
  // Small Lorentzian-like contribution on the diagonal at Omega = 1.
  for (long iq = 0; iq < Nq; ++iq)
    for (long iO = 0; iO < N_Omega; ++iO) {
      const double O = grid.Omega()(iO);
      const double v = -0.05 / (1.0 + (O - 1.0)*(O - 1.0));
      for (long P = 0; P < Naux; ++P)
        ImPi_full(iq, P, P, iO) = std::complex<double>(v, 0.0);
    }
  {
    auto fill = [&](nda::array<std::complex<double>, 4> const& src,
                    real_axis_mb_state_t::bosonic_dArray_t& dst) {
      auto Pr = dst.local_range(1);
      auto Qr = dst.local_range(2);
      auto loc = dst.local();
      for (long iq = 0; iq < Nq; ++iq)
        for (long iP = 0; iP < Pr.size(); ++iP)
          for (long iQ = 0; iQ < Qr.size(); ++iQ)
            for (long iO = 0; iO < N_Omega; ++iO)
              loc(iq, iP, iQ, iO) =
                  src(iq, Pr.first() + iP, Qr.first() + iQ, iO);
    };
    fill(ImPi_full, *state.ImPi_qPQO);
    fill(RePi_full, *state.RePi_qPQO);
  }

  // Bare Coulomb V: identity scaled by 1.0
  nda::array<std::complex<double>, 3> V(Nq, Naux, Naux);
  V = std::complex<double>(0.0, 0.0);
  for (long iq = 0; iq < Nq; ++iq)
    for (long P = 0; P < Naux; ++P)
      V(iq, P, P) = std::complex<double>(1.0, 0.0);

  gw.solve_W(state, V);

  // Outputs must exist and be finite. Gather to a replicated array for
  // convenient global-index inspection.
  REQUIRE(state.ImW_qPQO.has_value());
  REQUIRE(state.ReW_qPQO.has_value());
  auto ImW = math::nda::all_gather_slow<HOST_MEMORY>(*state.ImW_qPQO);
  auto ReW = math::nda::all_gather_slow<HOST_MEMORY>(*state.ReW_qPQO);
  for (long iq = 0; iq < Nq; ++iq)
    for (long P = 0; P < Naux; ++P)
      for (long Q = 0; Q < Naux; ++Q)
        for (long iO = 0; iO < N_Omega; ++iO) {
          REQUIRE(std::isfinite(ImW(iq, P, Q, iO).real()));
          REQUIRE(std::isfinite(ReW(iq, P, Q, iO).real()));
        }

  // Sanity: at this Im Pi, ReW should be close to V (~1 on diagonal, ~0 off).
  for (long iq = 0; iq < Nq; ++iq)
    for (long iO = 0; iO < N_Omega; ++iO) {
      for (long P = 0; P < Naux; ++P)
        REQUIRE(ReW(iq, P, P, iO).real() == Approx(1.0).margin(0.1));
      REQUIRE(std::abs(ReW(iq, 0, 1, iO).real()) < 0.05);
    }
}

// =============================================================================
// Causality-projection test: starting from Im Sigma with positive noise,
// the projection clips to zero on the offending entries.
// =============================================================================
TEST_CASE("real_axis_gw_causality_projection", "[real_axis][gw][causality]")
{
  const double w_max     = 8.0;
  const long   N_w       = 17;
  const double Omega_max = 4.0;
  const long   N_Omega   = 16;
  const long   N_t       = 64;
  const double T_window  = 16.0;
  auto grid = real_freq_grid_t::make_uniform(
      50.0, 0.0, w_max, N_w, Omega_max, N_Omega, N_t, T_window);
  real_axis_gw_t gw(grid, 1, 0.5, 1e-10, 1, "tag");

  // Positive Im Sigma: should be clipped to 0.
  nda::array<std::complex<double>, 1> Im(N_w);
  for (long l = 0; l < N_w; ++l) Im(l) = std::complex<double>(0.5, 0.0);
  gw.apply_causality_fermionic(Im);
  for (long l = 0; l < N_w; ++l)
    REQUIRE(Im(l).real() <= 0.0);

  // Mixed sign: only positives clipped.
  for (long l = 0; l < N_w; ++l)
    Im(l) = std::complex<double>(l - N_w/2.0, 0.0);  // -8..+8
  gw.apply_causality_fermionic(Im);
  for (long l = 0; l < N_w; ++l)
    REQUIRE(Im(l).real() <= 0.0);
}

} // namespace gw_real_axis_tests
