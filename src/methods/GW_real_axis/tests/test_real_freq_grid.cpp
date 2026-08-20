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

TEST_CASE("real_freq_grid_nonuniform_log_construction", "[real_axis][grid]")
{
  const double beta      = 100.0;
  const double mu        = 0.5;
  const double w_max     = 5.0;
  const long   N_w       = 65;
  const double w_dense   = 0.4;     // dense block on [-0.4, +0.4]
  const long   N_dense   = 21;      // odd, includes 0
  const double Omega_max = 4.0;
  const long   N_Omega   = 32;
  const long   N_t       = 64;
  const double T_window  = 32.0;

  auto g = real_freq_grid_t::make_nonuniform_log(
      beta, mu, w_max, N_w, w_dense, N_dense,
      Omega_max, N_Omega, N_t, T_window);

  REQUIRE(g.N_w() == N_w);

  // Endpoints span the full window.
  REQUIRE(g.w()(0)        == Approx(-w_max));
  REQUIRE(g.w()(N_w - 1)  == Approx(+w_max));

  // Strictly monotone increasing.
  for (long j = 1; j < N_w; ++j) {
    REQUIRE(g.w()(j) > g.w()(j-1));
  }

  // Dense block sits at the center, spanning [-w_dense, +w_dense] with
  // spacing h_dense = 2*w_dense/(N_dense-1).
  const long n_tail = (N_w - N_dense) / 2;
  REQUIRE(g.w()(n_tail)              == Approx(-w_dense).margin(1e-12));
  REQUIRE(g.w()(n_tail + N_dense -1) == Approx(+w_dense).margin(1e-12));
  const double h_dense = 2.0 * w_dense / static_cast<double>(N_dense - 1);
  for (long j = 1; j < N_dense; ++j) {
    REQUIRE((g.w()(n_tail + j) - g.w()(n_tail + j - 1))
            == Approx(h_dense).epsilon(1e-12));
  }

  // Tail spacing strictly increases away from the dense edge: log-spaced
  // tails have monotone-growing cell widths.
  for (long j = N_w - n_tail; j + 1 < N_w; ++j) {
    const double dj   = g.w()(j+1)   - g.w()(j);
    const double djm1 = g.w()(j)     - g.w()(j-1);
    REQUIRE(dj > djm1);
  }
  for (long j = 1; j < n_tail; ++j) {
    const double dj   = g.w()(j)   - g.w()(j-1);
    const double djp1 = g.w()(j+1) - g.w()(j);
    REQUIRE(dj > djp1);
  }

  // Mirror symmetry around 0.
  for (long j = 0; j < N_w; ++j) {
    REQUIRE(g.w()(j) == Approx(-g.w()(N_w - 1 - j)).margin(1e-12));
  }

  // Trapezoidal weights still sum to grid width (2 * w_max).
  double sum_w = 0.0;
  for (long j = 0; j < N_w; ++j) sum_w += g.w_weights()(j);
  REQUIRE(sum_w == Approx(2.0 * w_max).epsilon(1e-10));
}

TEST_CASE("real_freq_grid_nonuniform_log_lorentzian_quadrature",
          "[real_axis][grid]")
{
  // Integrate a normalized Lorentzian f(w) = (eta/pi) / (w^2 + eta^2)
  // (∫f = 1) using the grid's trapezoidal weights. Dense block centered
  // on the peak should resolve it; sparse tails carry negligible weight.
  const double eta   = 0.05;
  const double w_max = 5.0;
  const long   N_w   = 129;
  const double w_dense = 0.4;
  const long   N_dense = 65;
  const double Omega_max = 4.0;
  const long   N_Omega   = 32;
  const long   N_t       = 64;
  const double T_window  = 32.0;

  auto g_nu = real_freq_grid_t::make_nonuniform_log(
      100.0, 0.0, w_max, N_w, w_dense, N_dense,
      Omega_max, N_Omega, N_t, T_window);

  auto integrate = [&](real_freq_grid_t const& gg) {
    double s = 0.0;
    for (long j = 0; j < gg.N_w(); ++j) {
      const double w = gg.w()(j);
      s += gg.w_weights()(j) * (eta / M_PI) / (w*w + eta*eta);
    }
    return s;
  };
  const double I_nu = integrate(g_nu);

  // Target: the truncated-window analytic value (2/pi)*arctan(w_max/eta).
  // Full integral over R is 1; the tail outside ±w_max truncates ~0.006.
  const double I_exact = (2.0 / M_PI) * std::atan(w_max / eta);
  REQUIRE(I_nu == Approx(I_exact).margin(5e-4));

  // Same total points on uniform grid: dense-region spacing ~0.078, much
  // worse for Lorentzian peak resolution; integral biased low because
  // peak is undersampled. This documents the qualitative win: the
  // nonuniform grid resolves the peak with the same total point count.
  auto g_u = real_freq_grid_t::make_uniform(
      100.0, 0.0, w_max, N_w, Omega_max, N_Omega, N_t, T_window);
  const double I_u  = integrate(g_u);

  // The nonuniform-log error against the truncated-window value should
  // not exceed the uniform-grid error by more than rounding noise.
  REQUIRE(std::abs(I_nu - I_exact) <= std::abs(I_u - I_exact) + 1e-6);
}

// ---------------------------------------------------------------------------
// Bosonic-axis nonuniform construction. Same factory, but with non-default
// Omega_dense + N_Omega_dense activates a linear-dense + log-tail layout on
// the half-axis [h_dense, Omega_max].
// ---------------------------------------------------------------------------
TEST_CASE("real_freq_grid_nonuniform_log_bosonic_construction",
          "[real_axis][grid]")
{
  const double beta      = 100.0;
  const double mu        = 0.5;
  const double w_max     = 5.0;
  const long   N_w       = 65;
  const double w_dense   = 0.4;
  const long   N_dense   = 21;
  const double Omega_max = 8.0;
  const long   N_Omega   = 64;
  const double Omega_dense   = 1.0;     // dense on (h, 1.0]
  const long   N_Omega_dense = 32;      // dense pts
  const long   N_t       = 128;
  const double T_window  = 16.0;

  auto g = real_freq_grid_t::make_nonuniform_log(
      beta, mu, w_max, N_w, w_dense, N_dense,
      Omega_max, N_Omega, N_t, T_window,
      Omega_dense, N_Omega_dense);

  REQUIRE(g.N_Omega() == N_Omega);
  // Ω = 0 must NOT be on the grid (n_B singular).
  REQUIRE(g.Omega()(0) > 0.0);
  // Endpoint reaches Omega_max.
  REQUIRE(g.Omega()(N_Omega - 1) == Approx(Omega_max).epsilon(1e-12));

  // Strictly monotone increasing.
  for (long l = 1; l < N_Omega; ++l)
    REQUIRE(g.Omega()(l) > g.Omega()(l - 1));

  // Dense block: linear with spacing h_dense = Omega_dense / N_Omega_dense.
  const double h_dense = Omega_dense / static_cast<double>(N_Omega_dense);
  REQUIRE(g.Omega()(0)                   == Approx(h_dense).epsilon(1e-12));
  REQUIRE(g.Omega()(N_Omega_dense - 1)   == Approx(Omega_dense).epsilon(1e-12));
  for (long l = 1; l < N_Omega_dense; ++l) {
    REQUIRE((g.Omega()(l) - g.Omega()(l - 1))
            == Approx(h_dense).epsilon(1e-12));
  }

  // Log tail: spacing strictly increases.
  for (long l = N_Omega_dense + 1; l + 1 < N_Omega; ++l) {
    const double d_lo = g.Omega()(l)     - g.Omega()(l - 1);
    const double d_hi = g.Omega()(l + 1) - g.Omega()(l);
    REQUIRE(d_hi > d_lo);
  }

  // Trapezoidal weights are non-negative and sum to the half-axis width.
  double sum_O = 0.0;
  for (long l = 0; l < N_Omega; ++l) {
    REQUIRE(g.Omega_weights()(l) >= 0.0);
    sum_O += g.Omega_weights()(l);
  }
  // First trapezoidal weight covers half of the inner cell, last covers
  // half of the outer cell; total spans Omega_max - Omega(0).
  REQUIRE(sum_O == Approx(Omega_max - g.Omega()(0)).epsilon(1e-10));

  // Backward compat: passing Omega_dense=0 / N_Omega_dense=0 (defaults)
  // should produce the existing uniform-Ω grid.
  auto g_uniform_O = real_freq_grid_t::make_nonuniform_log(
      beta, mu, w_max, N_w, w_dense, N_dense,
      Omega_max, N_Omega, N_t, T_window);
  const double h_unif = Omega_max / static_cast<double>(N_Omega);
  for (long l = 0; l < N_Omega; ++l)
    REQUIRE(g_uniform_O.Omega()(l)
            == Approx(h_unif * static_cast<double>(l + 1)).epsilon(1e-12));
}

// ---------------------------------------------------------------------------
// Bosonic-axis Lorentzian quadrature (peak near small Ω). The nonuniform
// bosonic grid should resolve a peak at Ω ≈ Omega_dense/4 better than the
// uniform grid at the same total N_Omega.
// ---------------------------------------------------------------------------
TEST_CASE("real_freq_grid_nonuniform_log_bosonic_lorentzian",
          "[real_axis][grid]")
{
  // Lorentzian peaked at Omega_p = 0.5 with width gamma = 0.1.
  // The peak lives well inside the dense bosonic block.
  const double Omega_p = 0.5;
  const double gamma   = 0.1;
  const double w_max   = 5.0;
  const long   N_w     = 65;
  const double w_dense = 0.4;
  const long   N_dense = 21;
  const double Omega_max = 8.0;
  const long   N_Omega   = 32;
  const long   N_t       = 128;
  const double T_window  = 16.0;

  // Nonuniform bosonic: dense on (h, 2.0], tail to 8.0.
  const double Omega_dense_p   = 2.0;
  const long   N_Omega_dense_p = 16;

  auto g_nu = real_freq_grid_t::make_nonuniform_log(
      100.0, 0.0, w_max, N_w, w_dense, N_dense,
      Omega_max, N_Omega, N_t, T_window,
      Omega_dense_p, N_Omega_dense_p);
  auto g_un = real_freq_grid_t::make_nonuniform_log(
      100.0, 0.0, w_max, N_w, w_dense, N_dense,
      Omega_max, N_Omega, N_t, T_window);

  auto integrate = [&](real_freq_grid_t const& gg) {
    double s = 0.0;
    for (long l = 0; l < gg.N_Omega(); ++l) {
      const double O = gg.Omega()(l);
      const double v = (gamma / M_PI)
                     / ((O - Omega_p) * (O - Omega_p) + gamma * gamma);
      s += gg.Omega_weights()(l) * v;
    }
    return s;
  };

  // Truncated-window analytic Lorentzian half-window integral (one-sided).
  // ∫_{Omega(0)}^{Omega_max} (γ/π) / ((Ω-Ω_p)² + γ²) dΩ
  //   = (1/π) [arctan((Ω_max-Ω_p)/γ) - arctan((Ω(0)-Ω_p)/γ)]
  const double O_lo_nu = g_nu.Omega()(0);
  const double O_lo_un = g_un.Omega()(0);
  auto exact = [&](double O_lo) {
    return (1.0 / M_PI) * (std::atan((Omega_max - Omega_p) / gamma)
                          - std::atan((O_lo - Omega_p) / gamma));
  };
  const double I_nu_exact = exact(O_lo_nu);
  const double I_un_exact = exact(O_lo_un);

  const double I_nu = integrate(g_nu);
  const double I_un = integrate(g_un);

  const double err_nu = std::abs(I_nu - I_nu_exact);
  const double err_un = std::abs(I_un - I_un_exact);

  // Key claim of this test: the nonuniform bosonic axis resolves a
  // peak near small Ω substantially better than the uniform axis at the
  // same total N_Omega. The uniform grid here has dΩ = 0.25 ≫ γ = 0.1,
  // so the peak is heavily under-sampled and err_un is large; the
  // nonuniform grid (16 points dense on [h, 2.0]) brings dΩ_dense ≈
  // 0.125 ~ γ and recovers the peak to ≪ 0.1 error.
  REQUIRE(err_nu < err_un);
  REQUIRE(err_un > err_nu * 5.0);   // ≥ 5x improvement
  REQUIRE(err_nu < 2e-2);           // NU absolute bound
}

// ---------------------------------------------------------------------------
// Plasmon-mode bosonic axis: dense block centered at Omega_center > 0.
// ---------------------------------------------------------------------------
TEST_CASE("real_freq_grid_nonuniform_log_plasmon_construction",
          "[real_axis][grid]")
{
  const double beta      = 100.0;
  const double mu        = 0.0;
  const double w_max     = 5.0;
  const long   N_w       = 65;
  const double w_dense   = 0.4;
  const long   N_dense   = 21;
  const double Omega_max = 8.0;
  const long   N_Omega   = 64;
  const double Omega_dense   = 1.0;    // full width of dense block
  const long   N_Omega_dense = 32;
  const double Omega_center  = 2.0;    // dense block centered here
  const long   N_t       = 256;
  const double T_window  = 12.0;

  auto g = real_freq_grid_t::make_nonuniform_log(
      beta, mu, w_max, N_w, w_dense, N_dense,
      Omega_max, N_Omega, N_t, T_window,
      Omega_dense, N_Omega_dense, Omega_center);

  REQUIRE(g.N_Omega() == N_Omega);
  REQUIRE(g.Omega()(0) > 0.0);
  REQUIRE(g.Omega()(N_Omega - 1) == Approx(Omega_max).epsilon(1e-12));

  // Strictly monotone increasing.
  for (long l = 1; l < N_Omega; ++l)
    REQUIRE(g.Omega()(l) > g.Omega()(l - 1));

  // Symmetric tails: N_Omega - N_Omega_dense split equally between
  // inner and outer tails.
  const long n_tail = (N_Omega - N_Omega_dense) / 2;
  REQUIRE(n_tail * 2 + N_Omega_dense == N_Omega);

  // Dense block: linear with spacing h = Omega_dense/(N_Omega_dense-1),
  // covering [Omega_center - Omega_dense/2, Omega_center + Omega_dense/2].
  const double halfwidth = Omega_dense * 0.5;
  const double h_dense = Omega_dense / static_cast<double>(N_Omega_dense - 1);
  const double lower_edge = Omega_center - halfwidth;
  const double upper_edge = Omega_center + halfwidth;

  REQUIRE(g.Omega()(n_tail)                       == Approx(lower_edge).epsilon(1e-12));
  REQUIRE(g.Omega()(n_tail + N_Omega_dense - 1)  == Approx(upper_edge).epsilon(1e-12));
  for (long j = 1; j < N_Omega_dense; ++j) {
    REQUIRE((g.Omega()(n_tail + j) - g.Omega()(n_tail + j - 1))
            == Approx(h_dense).epsilon(1e-12));
  }

  // Inner tail: log-spaced, spacing strictly increases as Ω → lower_edge.
  for (long l = 1; l + 1 < n_tail; ++l) {
    const double d_lo = g.Omega()(l)     - g.Omega()(l - 1);
    const double d_hi = g.Omega()(l + 1) - g.Omega()(l);
    REQUIRE(d_hi > d_lo);
  }
  // Outer tail: spacing strictly increases as Ω → Omega_max.
  for (long l = n_tail + N_Omega_dense + 1; l + 1 < N_Omega; ++l) {
    const double d_lo = g.Omega()(l)     - g.Omega()(l - 1);
    const double d_hi = g.Omega()(l + 1) - g.Omega()(l);
    REQUIRE(d_hi > d_lo);
  }

  // Weights non-negative; sum = span of the grid.
  double sum_O = 0.0;
  for (long l = 0; l < N_Omega; ++l) {
    REQUIRE(g.Omega_weights()(l) >= 0.0);
    sum_O += g.Omega_weights()(l);
  }
  REQUIRE(sum_O == Approx(Omega_max - g.Omega()(0)).epsilon(1e-10));
}

// ---------------------------------------------------------------------------
// Plasmon-mode quadrature: a Lorentzian peaked at Omega_pl is resolved much
// better with Omega_center = Omega_pl than with the dense-at-zero layout.
// ---------------------------------------------------------------------------
TEST_CASE("real_freq_grid_nonuniform_log_plasmon_lorentzian",
          "[real_axis][grid]")
{
  // Sharper plasmon-like peak (γ = 0.04, FWHM = 0.08 = 2.2 eV at Si plasmon
  // energy ~16 eV). Whenever the peak is much narrower than the spacing
  // of the dense-at-zero log tail at Ω ≈ Omega_pl, only the centered grid
  // captures it.
  const double Omega_pl = 3.0;           // simulated plasmon frequency
  const double gamma   = 0.04;
  const double w_max   = 5.0;
  const long   N_w     = 65;
  const double w_dense = 0.4;
  const long   N_dense = 21;
  const double Omega_max = 8.0;
  const long   N_Omega   = 32;
  const long   N_t       = 256;
  const double T_window  = 12.0;

  // (a) dense at zero — wrong place for this peak
  auto g_zero = real_freq_grid_t::make_nonuniform_log(
      100.0, 0.0, w_max, N_w, w_dense, N_dense,
      Omega_max, N_Omega, N_t, T_window,
      /*Omega_dense*/ 1.0, /*N_Omega_dense*/ 16);
  // (b) dense at Omega_pl — should resolve the peak
  auto g_plas = real_freq_grid_t::make_nonuniform_log(
      100.0, 0.0, w_max, N_w, w_dense, N_dense,
      Omega_max, N_Omega, N_t, T_window,
      /*Omega_dense*/ 0.5, /*N_Omega_dense*/ 16,
      /*Omega_center*/ Omega_pl);

  auto integrate = [&](real_freq_grid_t const& gg) {
    double s = 0.0;
    for (long l = 0; l < gg.N_Omega(); ++l) {
      const double O = gg.Omega()(l);
      const double v = (gamma / M_PI)
                     / ((O - Omega_pl) * (O - Omega_pl) + gamma * gamma);
      s += gg.Omega_weights()(l) * v;
    }
    return s;
  };

  auto exact = [&](double O_lo) {
    return (1.0 / M_PI) * (std::atan((Omega_max - Omega_pl) / gamma)
                          - std::atan((O_lo - Omega_pl) / gamma));
  };

  const double err_zero = std::abs(integrate(g_zero) - exact(g_zero.Omega()(0)));
  const double err_plas = std::abs(integrate(g_plas) - exact(g_plas.Omega()(0)));

  // The plasmon-centered layout must resolve the peak substantially
  // better than the dense-at-zero one at the same N_Omega.
  REQUIRE(err_plas < err_zero);
  REQUIRE(err_zero > err_plas * 5.0);
}

} // namespace gw_real_axis_tests
