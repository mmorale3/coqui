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
using methods::real_axis::accumulate_ImSigma_one_kq_nufft;
using methods::real_axis::ReSigma_from_ImSigma_aux;

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
  nda::array<std::complex<double>, 3> A(Naux, Naux, N_w);
  nda::array<std::complex<double>, 3> B(Naux, Naux, N_Omega);
  A = std::complex<double>(0.0, 0.0);
  B = std::complex<double>(0.0, 0.0);

  nda::array<std::complex<double>, 3> ImSigma(Naux, Naux, N_w);
  ImSigma = std::complex<double>(0.0, 0.0);
  accumulate_ImSigma_one_kq(conv, A, B, ImSigma, 1.0);

  for (long P = 0; P < Naux; ++P)
    for (long Q = 0; Q < Naux; ++Q)
      for (long l = 0; l < N_w; ++l) {
        REQUIRE(std::abs(ImSigma(P, Q, l).real()) < 1e-12);
        REQUIRE(std::abs(ImSigma(P, Q, l).imag()) < 1e-12);
      }
}

// =============================================================================
// Smoke test: non-trivial diagonal inputs produce a non-trivial Im Sigma.
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
  nda::array<std::complex<double>, 3> A(Naux, Naux, N_w);
  nda::array<std::complex<double>, 3> B(Naux, Naux, N_Omega);

  // Single-band Gaussian spectral function at ε = -0.5 (occupied).
  for (long l = 0; l < N_w; ++l) {
    const double w_l = grid.w()(l);
    A(0, 0, l) = std::complex<double>(std::exp(-0.5 * (w_l + 0.5)*(w_l + 0.5)), 0.0);
  }
  // Bosonic spectral function: simple peaked at Omega=2 (one plasmon-like mode).
  for (long iO = 0; iO < N_Omega; ++iO) {
    const double O = grid.Omega()(iO);
    B(0, 0, iO) = std::complex<double>(O * std::exp(-0.5 * (O - 2.0)*(O - 2.0)), 0.0);
  }

  nda::array<std::complex<double>, 3> ImSigma(Naux, Naux, N_w);
  ImSigma = std::complex<double>(0.0, 0.0);
  accumulate_ImSigma_one_kq(conv, A, B, ImSigma, 1.0);

  // Im Sigma must be finite everywhere.
  for (long l = 0; l < N_w; ++l) {
    REQUIRE(std::isfinite(ImSigma(0, 0, l).real()));
    REQUIRE(std::isfinite(ImSigma(0, 0, l).imag()));
  }

  // Im Sigma must be approximately real (B and A are both real).
  double max_imag = 0.0;
  for (long l = 0; l < N_w; ++l)
    max_imag = std::max(max_imag, std::abs(ImSigma(0, 0, l).imag()));
  REQUIRE(max_imag < 1e-8);

  // Im Sigma must not be zero in the bulk.
  double max_abs = 0.0;
  for (long l = 0; l < N_w; ++l)
    max_abs = std::max(max_abs, std::abs(ImSigma(0, 0, l).real()));
  REQUIRE(max_abs > 1e-3);
}

// =============================================================================
// Hilbert round-trip on the auxiliary-basis Sigma.
// =============================================================================
TEST_CASE("real_axis_sigma_hilbert_runs", "[real_axis][sigma]")
{
  const double w_max     = 30.0;
  const long   N_w       = 257;
  const double Omega_max = 1.0;
  const long   N_Omega   = 8;
  const long   N_t       = 512;
  const double T_window  = 16.0;

  auto grid = real_freq_grid_t::make_uniform(
      50.0, 0.0, w_max, N_w, Omega_max, N_Omega, N_t, T_window);
  real_axis_conv_t conv(grid, 4, 1e-10);

  const long Naux = 2;
  nda::array<std::complex<double>, 3> ImSigma(Naux, Naux, N_w);
  nda::array<std::complex<double>, 3> ReSigma(Naux, Naux, N_w);
  // Set up a simple synthetic Im Sigma: Lorentzian dip on the diagonal.
  for (long P = 0; P < Naux; ++P)
    for (long Q = 0; Q < Naux; ++Q)
      for (long l = 0; l < N_w; ++l) {
        const double w_l = grid.w()(l);
        const double v = (P == Q ? -0.5 / (1.0 + w_l*w_l) : 0.0);
        ImSigma(P, Q, l) = std::complex<double>(v, 0.0);
      }
  ReSigma_from_ImSigma_aux(conv, ImSigma, ReSigma);
  for (long P = 0; P < Naux; ++P)
    for (long Q = 0; Q < Naux; ++Q)
      for (long l = 0; l < N_w; ++l)
        REQUIRE(std::isfinite(ReSigma(P, Q, l).real()));

  // Off-diagonal must be zero (linearity).
  for (long l = 0; l < N_w; ++l) {
    REQUIRE(std::abs(ReSigma(0, 1, l).real()) < 1e-8);
    REQUIRE(std::abs(ReSigma(1, 0, l).real()) < 1e-8);
  }
}

// =============================================================================
// Regression test: the NUFFT-accelerated and direct-quadrature versions of
// accumulate_ImSigma_one_kq should agree (modulo discretization error from
// resampling B onto the fermionic grid).
// =============================================================================
TEST_CASE("real_axis_sigma_nufft_matches_direct", "[real_axis][sigma][nufft]")
{
  const double w_max     = 12.0;
  const long   N_w       = 257;
  const double Omega_max = 6.0;
  const long   N_Omega   = 64;
  const long   N_t       = 512;
  const double T_window  = 16.0;
  const double beta = 50.0;
  const double mu   = 0.0;

  auto grid = real_freq_grid_t::make_uniform(
      beta, mu, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

  const long Naux = 2;
  real_axis_conv_t conv(grid, Naux*Naux, 1e-10);

  // Synthetic A: diagonal Gaussians at +/- 0.5.
  nda::array<std::complex<double>, 3> A(Naux, Naux, N_w);
  A = std::complex<double>(0.0, 0.0);
  for (long iw = 0; iw < N_w; ++iw) {
    const double w_l = grid.w()(iw);
    const double a0 = std::exp(-(w_l + 0.5)*(w_l + 0.5));
    const double a1 = std::exp(-(w_l - 0.5)*(w_l - 0.5));
    A(0, 0, iw) = std::complex<double>(a0, 0.0);
    A(1, 1, iw) = std::complex<double>(a1, 0.0);
  }
  // Synthetic B: diagonal, B_PQ(O) = O * exp(-(O - 1)^2) (odd-in-O on diagonal).
  nda::array<std::complex<double>, 3> B(Naux, Naux, N_Omega);
  B = std::complex<double>(0.0, 0.0);
  for (long iO = 0; iO < N_Omega; ++iO) {
    const double O = grid.Omega()(iO);
    const double v = O * std::exp(-(O - 1.0)*(O - 1.0));
    B(0, 0, iO) = std::complex<double>(v, 0.0);
    B(1, 1, iO) = std::complex<double>(0.5 * v, 0.0);
  }

  nda::array<std::complex<double>, 3> ImSigma_dir (Naux, Naux, N_w);
  nda::array<std::complex<double>, 3> ImSigma_nuf(Naux, Naux, N_w);
  ImSigma_dir = std::complex<double>(0.0, 0.0);
  ImSigma_nuf = std::complex<double>(0.0, 0.0);
  accumulate_ImSigma_one_kq      (conv, A, B, ImSigma_dir, 1.0);
  accumulate_ImSigma_one_kq_nufft(conv, A, B, ImSigma_nuf, 1.0);

  // Restrict comparison to the central window (boundary effects from
  // resampling onto the fermionic grid + finite time window contaminate
  // the outer ring).
  long n_central = 0, n_pass = 0;
  double max_dir = 0.0;
  for (long iw = 0; iw < N_w; ++iw) {
    if (std::abs(grid.w()(iw)) > 0.7 * w_max) continue;
    for (long P = 0; P < Naux; ++P)
      max_dir = std::max(max_dir, std::abs(ImSigma_dir(P, P, iw).real()));
  }
  for (long iw = 0; iw < N_w; ++iw) {
    if (std::abs(grid.w()(iw)) > 0.7 * w_max) continue;
    for (long P = 0; P < Naux; ++P)
      for (long Q = 0; Q < Naux; ++Q) {
        ++n_central;
        const double d = std::abs(ImSigma_dir(P, Q, iw).real()
                                  - ImSigma_nuf(P, Q, iw).real());
        if (d < 0.05 * max_dir + 1e-3) ++n_pass;
      }
  }
  // The two methods are not byte-identical (NUFFT path uses an interpolated
  // B on the fermionic grid; direct path uses native bosonic quadrature),
  // but they should agree within a few percent of the bulk magnitude.
  REQUIRE(n_pass > (90 * n_central) / 100);
  // Also: both must be approximately real-valued.
  for (long iw = 0; iw < N_w; ++iw)
    for (long P = 0; P < Naux; ++P)
      for (long Q = 0; Q < Naux; ++Q) {
        REQUIRE(std::abs(ImSigma_dir(P, Q, iw).imag()) < 1e-4);
        REQUIRE(std::abs(ImSigma_nuf(P, Q, iw).imag()) < 1e-4);
      }
}

} // namespace gw_real_axis_tests
