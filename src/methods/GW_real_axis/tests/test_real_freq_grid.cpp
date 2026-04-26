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

#include <cmath>

namespace gw_real_axis_tests
{

using methods::real_axis::real_freq_grid_t;

TEST_CASE("real_freq_grid_uniform_construction", "[real_axis][grid]")
{
  const double beta = 100.0;
  const double mu   = 0.5;
  const double w_max = 5.0;
  const long   N_w = 65;        // odd so 0 is included; we test both
  const double Omega_max = 4.0;
  const long   N_Omega   = 32;
  const long   N_t       = 64;
  // Nyquist: dt < pi/freq_max  =>  T_window/N_t < pi/freq_max.
  // freq_max = max(w_max, Omega_max) = 5.0; require dt <= pi/5.0 ~ 0.628.
  // Choose T_window so that dt = T_window/N_t = 0.5 (well under bound).
  const double T_window = 32.0;

  auto g = real_freq_grid_t::make_uniform(
      beta, mu, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

  REQUIRE(g.beta()    == Approx(beta));
  REQUIRE(g.mu_chem() == Approx(mu));
  REQUIRE(g.N_w()     == N_w);
  REQUIRE(g.N_Omega() == N_Omega);
  REQUIRE(g.N_t()     == N_t);
  REQUIRE(g.dt()      == Approx(T_window / static_cast<double>(N_t)));

  // Fermionic grid: spans full window
  REQUIRE(g.w()(0)        == Approx(-w_max));
  REQUIRE(g.w()(N_w - 1)  == Approx(+w_max));

  // Bosonic grid: strictly positive
  REQUIRE(g.Omega()(0) > 0.0);
  REQUIRE(g.Omega()(N_Omega - 1) == Approx(Omega_max));

  // Time grid: symmetric around 0, t_{N_t/2} = 0
  REQUIRE(g.t()(N_t / 2) == Approx(0.0));
  REQUIRE(g.t()(0)         < 0.0);
  REQUIRE(g.t()(N_t - 1)   > 0.0);

  // Quadrature weights normalized to sum to grid width.
  double sum_w = 0.0;
  for (long j = 0; j < g.N_w(); ++j) sum_w += g.w_weights()(j);
  REQUIRE(sum_w == Approx(2.0 * w_max).epsilon(1e-10));

  // Bosonic grid: uniform with spacing h = Omega_max/N_Omega starting at h.
  // Trapezoidal weights sum to h*(N-1) = Omega_max*(N-1)/N for that grid.
  double sum_Omega = 0.0;
  for (long l = 0; l < g.N_Omega(); ++l) sum_Omega += g.Omega_weights()(l);
  const double h_O = Omega_max / static_cast<double>(N_Omega);
  REQUIRE(sum_Omega == Approx(h_O * static_cast<double>(N_Omega - 1)).epsilon(1e-10));
}

TEST_CASE("real_freq_grid_finite_T_kernels", "[real_axis][grid]")
{
  const double beta = 50.0;
  const double mu   = 0.0;
  // Sanity: low temperature limit, f -> step at mu, n_B -> 0 for Omega>>kT.
  REQUIRE(real_freq_grid_t::fermi(-1.0, mu, beta) == Approx(1.0).margin(1e-10));
  REQUIRE(real_freq_grid_t::fermi( 1.0, mu, beta) == Approx(0.0).margin(1e-10));
  REQUIRE(real_freq_grid_t::fermi( 0.0, mu, beta) == Approx(0.5).margin(1e-10));

  // Stable evaluation at large argument (no overflow).
  REQUIRE(std::isfinite(real_freq_grid_t::fermi(+100.0, mu, beta)));
  REQUIRE(std::isfinite(real_freq_grid_t::fermi(-100.0, mu, beta)));
  REQUIRE(real_freq_grid_t::fermi(+100.0, mu, beta) >= 0.0);
  REQUIRE(real_freq_grid_t::fermi(+100.0, mu, beta) <= 1.0);

  // Bose: at large argument decays exponentially, at small argument ~ 1/(beta*Omega)
  REQUIRE(real_freq_grid_t::bose(0.5, beta) == Approx(1.0/(std::exp(25.0)-1.0)).epsilon(1e-10));

  // Small-Omega stability check via expm1: 1/(exp(beta*Omega)-1) ~ 1/(beta*Omega) - 1/2
  const double Omega = 1e-6;
  const double n_exact = 1.0/(std::expm1(beta*Omega));
  REQUIRE(real_freq_grid_t::bose(Omega, beta) == Approx(n_exact).epsilon(1e-12));

  // Tripwire: bose(0) returns NaN
  REQUIRE(std::isnan(real_freq_grid_t::bose(0.0, beta)));

  // f + (1-f) == 1 exactly (numerical stability)
  for (double w : {-2.0, -0.1, 0.0, 0.1, 2.0}) {
    REQUIRE(real_freq_grid_t::fermi(w, mu, beta)
          + real_freq_grid_t::fermi_bar(w, mu, beta)
          == Approx(1.0).epsilon(1e-14));
  }
}

// Validation errors trigger APP_ABORT in the codebase rather than throwing,
// so they cannot be exercised from a Catch2 unit test. The validation logic
// in real_freq_grid_t is exercised at integration time via downstream tests.

} // namespace gw_real_axis_tests
