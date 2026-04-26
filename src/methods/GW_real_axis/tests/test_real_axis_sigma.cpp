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
#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_conv.hpp"
#include "methods/GW_real_axis/real_axis_sigma.hpp"

#include <cmath>
#include <complex>

namespace gw_real_axis_tests
{

using methods::real_axis::real_freq_grid_t;
using methods::real_axis::real_axis_conv_t;
using methods::real_axis::accumulate_ImSigma_one_kq;

// =============================================================================
// Smoke test: trivial inputs (zero spectral functions) produce zero output.
// =============================================================================
TEST_CASE("real_axis_sigma_zero_inputs", "[real_axis][sigma]")
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
  real_axis_conv_t conv(grid, 4, 1e-10);

  const long Naux = 2;
  nda::array<std::complex<double>, 2> A(Naux, N_w);
  nda::array<std::complex<double>, 3> B(Naux, Naux, N_Omega);
  A = std::complex<double>(0.0, 0.0);
  B = std::complex<double>(0.0, 0.0);

  nda::array<std::complex<double>, 2> ImSigma(Naux, N_w);
  ImSigma = std::complex<double>(0.0, 0.0);
  accumulate_ImSigma_one_kq(conv, A, B, ImSigma, 1.0);

  for (long P = 0; P < Naux; ++P)
    for (long l = 0; l < N_w; ++l) {
      REQUIRE(std::abs(ImSigma(P, l).real()) < 1e-12);
      REQUIRE(std::abs(ImSigma(P, l).imag()) < 1e-12);
    }
}

// =============================================================================
// Smoke test: non-trivial diagonal inputs produce a non-trivial Im Sigma.
// We do not check the value precisely; we check that the result is finite,
// real-valued (since A and B are real-valued in this test), and non-zero in
// the bulk of the fermionic window.
// =============================================================================
TEST_CASE("real_axis_sigma_non_trivial", "[real_axis][sigma]")
{
  const double w_max     = 8.0;
  const long   N_w       = 129;
  const double Omega_max = 4.0;
  const long   N_Omega   = 64;
  const long   N_t       = 256;
  const double T_window  = 16.0;
  const double beta = 50.0;
  const double mu   = 0.0;

  auto grid = real_freq_grid_t::make_uniform(
      beta, mu, w_max, N_w, Omega_max, N_Omega, N_t, T_window);
  real_axis_conv_t conv(grid, 1, 1e-10);

  const long Naux = 1;
  nda::array<std::complex<double>, 2> A(Naux, N_w);
  nda::array<std::complex<double>, 3> B(Naux, Naux, N_Omega);

  // Single-band Gaussian spectral function at ε = -0.5 (occupied).
  for (long l = 0; l < N_w; ++l) {
    const double w_l = grid.w()(l);
    A(0, l) = std::complex<double>(std::exp(-0.5 * (w_l + 0.5)*(w_l + 0.5)), 0.0);
  }
  // Bosonic spectral function: simple peaked at Omega=2 (one plasmon-like mode).
  for (long iO = 0; iO < N_Omega; ++iO) {
    const double O = grid.Omega()(iO);
    B(0, 0, iO) = std::complex<double>(O * std::exp(-0.5 * (O - 2.0)*(O - 2.0)), 0.0);
  }

  nda::array<std::complex<double>, 2> ImSigma(Naux, N_w);
  ImSigma = std::complex<double>(0.0, 0.0);
  accumulate_ImSigma_one_kq(conv, A, B, ImSigma, 1.0);

  // Im Sigma must be finite everywhere.
  for (long l = 0; l < N_w; ++l) {
    REQUIRE(std::isfinite(ImSigma(0, l).real()));
    REQUIRE(std::isfinite(ImSigma(0, l).imag()));
  }

  // Im Sigma must be approximately real (B and A are both real).
  double max_imag = 0.0;
  for (long l = 0; l < N_w; ++l)
    max_imag = std::max(max_imag, std::abs(ImSigma(0, l).imag()));
  REQUIRE(max_imag < 1e-8);

  // Im Sigma must not be zero in the bulk.
  double max_abs = 0.0;
  for (long l = 0; l < N_w; ++l)
    max_abs = std::max(max_abs, std::abs(ImSigma(0, l).real()));
  REQUIRE(max_abs > 1e-3);
}

} // namespace gw_real_axis_tests
