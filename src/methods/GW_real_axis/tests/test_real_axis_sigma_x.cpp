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

#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"

#include "nda/nda.hpp"
#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_sigma_x.hpp"

#include <cmath>
#include <complex>

namespace gw_real_axis_tests
{

using methods::real_axis::real_freq_grid_t;
using methods::real_axis::evaluate_Sigma_x_serial;
using cval_t = std::complex<double>;

// =============================================================================
// Smoke test: zero spectral function gives zero Sigma_x.
// =============================================================================
TEST_CASE("real_axis_sigma_x_zero_input", "[real_axis][sigma_x]")
{
  const double w_max     = 8.0;
  const long   N_w       = 33;
  const double Omega_max = 4.0;
  const long   N_Omega   = 16;
  const long   N_t       = 64;
  const double T_window  = 16.0;
  auto grid = real_freq_grid_t::make_uniform(
      50.0, 0.0, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

  const long ns = 1, Nk = 1, Nq = 1, Naux = 2, nbnd = 2;
  nda::array<cval_t, 5> A(ns, Nk, N_w, nbnd, nbnd);
  A = cval_t(0.0, 0.0);
  nda::array<cval_t, 4> X(ns, Nk, Naux, nbnd);
  X = cval_t(0.0, 0.0);
  for (long P = 0; P < Naux; ++P) X(0, 0, P, P) = cval_t(1.0, 0.0);
  nda::array<cval_t, 3> V(Nq, Naux, Naux);
  V = cval_t(0.0, 0.0);
  for (long P = 0; P < Naux; ++P) V(0, P, P) = cval_t(1.0, 0.0);
  nda::array<long, 2> kmq(Nk, Nq);  kmq(0, 0) = 0;
  nda::array<cval_t, 4> Sx(ns, Nk, nbnd, nbnd);

  auto& mpi_context = utils::make_unit_test_mpi_context();
  evaluate_Sigma_x_serial(mpi_context->comm, grid, A, X, V, kmq, Sx);
  for (long mu = 0; mu < nbnd; ++mu)
    for (long nu = 0; nu < nbnd; ++nu)
      REQUIRE(std::abs(Sx(0, 0, mu, nu)) < 1e-12);
}

// =============================================================================
// Non-trivial case: unit V, identity X. Then Sigma_x_aux = - n_aux, and
// after back-projection Sigma_x_orb = -n_orb. Verify the magnitude and sign.
// =============================================================================
TEST_CASE("real_axis_sigma_x_identity_check", "[real_axis][sigma_x]")
{
  const double w_max     = 8.0;
  const long   N_w       = 257;
  const double Omega_max = 4.0;
  const long   N_Omega   = 16;
  const long   N_t       = 64;
  const double T_window  = 16.0;
  const double beta = 100.0;
  const double mu   = 0.0;
  auto grid = real_freq_grid_t::make_uniform(
      beta, mu, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

  const long ns = 1, Nk = 1, Nq = 1, Naux = 2, nbnd = 2;

  // Spectral function: sharp Gaussians at -1 (occupied) and +1 (virtual),
  // each normalized to 1.
  nda::array<cval_t, 5> A(ns, Nk, N_w, nbnd, nbnd);
  A = cval_t(0.0, 0.0);
  for (long iw = 0; iw < N_w; ++iw) {
    const double w_l = grid.w()(iw);
    const double sigma2 = 0.05;  // narrow
    const double a0 = std::exp(-0.5 * (w_l + 1.0)*(w_l + 1.0) / sigma2)
                    / std::sqrt(2.0 * M_PI * sigma2);
    const double a1 = std::exp(-0.5 * (w_l - 1.0)*(w_l - 1.0) / sigma2)
                    / std::sqrt(2.0 * M_PI * sigma2);
    A(0, 0, iw, 0, 0) = cval_t(a0, 0.0);
    A(0, 0, iw, 1, 1) = cval_t(a1, 0.0);
  }

  // X = identity (Naux = nbnd).
  nda::array<cval_t, 4> X(ns, Nk, Naux, nbnd);
  X = cval_t(0.0, 0.0);
  for (long P = 0; P < Naux; ++P) X(0, 0, P, P) = cval_t(1.0, 0.0);

  // V = identity.
  nda::array<cval_t, 3> V(Nq, Naux, Naux);
  V = cval_t(0.0, 0.0);
  for (long P = 0; P < Naux; ++P) V(0, P, P) = cval_t(1.0, 0.0);

  nda::array<long, 2> kmq(Nk, Nq);  kmq(0, 0) = 0;
  nda::array<cval_t, 4> Sx(ns, Nk, nbnd, nbnd);

  auto& mpi_context = utils::make_unit_test_mpi_context();
  evaluate_Sigma_x_serial(mpi_context->comm, grid, A, X, V, kmq, Sx);

  // Expected: n_orb diagonal with n_00 ~ 1 (occupied), n_11 ~ 0 (virtual),
  // Sigma_x_orb diagonal with Sx_00 ~ -1, Sx_11 ~ 0.
  REQUIRE(Sx(0, 0, 0, 0).real() == Approx(-1.0).margin(0.05));
  REQUIRE(std::abs(Sx(0, 0, 1, 1).real()) < 0.05);
  REQUIRE(std::abs(Sx(0, 0, 0, 1)) < 0.05);
  REQUIRE(std::abs(Sx(0, 0, 1, 0)) < 0.05);
  // Sigma_x must be Hermitian: Sx_munu = conj(Sx_numu)
  REQUIRE(std::abs(Sx(0, 0, 0, 1) - std::conj(Sx(0, 0, 1, 0))) < 1e-10);
}

} // namespace gw_real_axis_tests
