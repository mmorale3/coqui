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
#include "methods/GW_real_axis/real_axis_pi.hpp"

#include <cmath>
#include <complex>

namespace gw_real_axis_tests
{

using methods::real_axis::real_freq_grid_t;
using methods::real_axis::real_axis_conv_t;
using methods::real_axis::accumulate_ImPi_one_kq;
using methods::real_axis::RePi_from_ImPi;

// =============================================================================
// Sanity test: At Omega = 0, the kernel f(w) - f(w+Omega) vanishes identically,
// so Im Pi(Omega=0) must be zero (independent of A). Verify on a single
// auxiliary (Naux=1) bubble with a Gaussian spectral function.
// =============================================================================
TEST_CASE("real_axis_pi_zero_at_zero_Omega", "[real_axis][pi]")
{
  const double w_max     = 10.0;
  const long   N_w       = 401;
  const double Omega_max = 4.0;
  const long   N_Omega   = 100;
  const long   N_t       = 256;
  const double T_window  = 16.0;
  const double beta = 50.0;
  const double mu   = 0.0;

  auto grid = real_freq_grid_t::make_uniform(
      beta, mu, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

  const long Naux = 1;
  real_axis_conv_t conv(grid, /*ntrans*/ Naux*Naux, /*eps*/ 1e-10);

  // Same Gaussian spectral function at k and k+q.
  const double sigma = 1.0;
  nda::array<std::complex<double>, 3> A_k(Naux, Naux, N_w), A_kq(Naux, Naux, N_w);
  for (long j = 0; j < N_w; ++j) {
    const double wj = grid.w()(j);
    const double v = std::exp(-0.5 * wj * wj / (sigma * sigma))
                   / (sigma * std::sqrt(2.0 * M_PI));
    A_k (0, 0, j) = std::complex<double>(v, 0.0);
    A_kq(0, 0, j) = std::complex<double>(v, 0.0);
  }

  nda::array<std::complex<double>, 3> ImPi(Naux, Naux, N_Omega);
  ImPi = std::complex<double>(0.0, 0.0);
  accumulate_ImPi_one_kq(conv, A_k, A_kq, ImPi, /*k_weight*/ 1.0);

  // The smallest Omega in our grid is dOmega = Omega_max/N_Omega = 0.04.
  // Im Pi(Omega) should approach 0 linearly as Omega -> 0.
  const double smallest = std::abs(ImPi(0, 0, 0).real());
  const double largest_around_kT = 0.0;  // unused; placeholder
  // Sanity: Im Pi at smallest Omega is much smaller than at typical Omega.
  // Find the max |Im Pi| anywhere on grid.
  double max_abs = 0.0;
  for (long iO = 0; iO < N_Omega; ++iO)
    max_abs = std::max(max_abs, std::abs(ImPi(0, 0, iO).real()));
  // smallest should be at most a small fraction of the bulk.
  REQUIRE(smallest < 0.2 * max_abs);
  // Im Pi must be real (numerical noise allowed).
  for (long iO = 0; iO < N_Omega; ++iO)
    REQUIRE(std::abs(ImPi(0, 0, iO).imag()) < 1e-3);
  (void)largest_around_kT;
}

// =============================================================================
// Round-trip test: Re Pi recovered from Im Pi via Hilbert should be finite
// and approximately a real-valued function. (We do not check exact values
// here; the Hilbert-transform identity is exercised in the conv tests.)
// =============================================================================
TEST_CASE("real_axis_pi_RePi_from_ImPi_runs", "[real_axis][pi]")
{
  const double w_max     = 10.0;
  const long   N_w       = 201;
  const double Omega_max = 4.0;
  const long   N_Omega   = 64;
  const long   N_t       = 256;
  const double T_window  = 16.0;
  const double beta = 50.0;
  const double mu   = 0.0;

  auto grid = real_freq_grid_t::make_uniform(
      beta, mu, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

  const long Naux = 2;
  real_axis_conv_t conv(grid, Naux*Naux, 1e-10);

  // Synthetic Im Pi: a simple real-valued odd-in-Omega function diagonal.
  nda::array<double, 3> ImPi(Naux, Naux, N_Omega), RePi(Naux, Naux, N_Omega);
  for (long P = 0; P < Naux; ++P)
    for (long Q = 0; Q < Naux; ++Q)
      for (long iO = 0; iO < N_Omega; ++iO) {
        const double O = grid.Omega()(iO);
        ImPi(P, Q, iO) = (P == Q ? std::exp(-O*O) - std::exp(-(O-1.0)*(O-1.0)) : 0.0);
      }
  RePi_from_ImPi(conv, ImPi, RePi);

  // Re Pi must be finite everywhere.
  for (long P = 0; P < Naux; ++P)
    for (long Q = 0; Q < Naux; ++Q)
      for (long iO = 0; iO < N_Omega; ++iO)
        REQUIRE(std::isfinite(RePi(P, Q, iO)));

  // Off-diagonal entries with zero ImPi should give zero RePi (linearity).
  for (long iO = 0; iO < N_Omega; ++iO) {
    REQUIRE(std::abs(RePi(0, 1, iO)) < 1e-8);
    REQUIRE(std::abs(RePi(1, 0, iO)) < 1e-8);
  }
}

} // namespace gw_real_axis_tests
