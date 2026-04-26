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

#include <cmath>
#include <complex>

namespace gw_real_axis_tests
{

using methods::real_axis::real_freq_grid_t;
using methods::real_axis::real_axis_conv_t;
using cval_t = std::complex<double>;

// =============================================================================
// Cross-correlation: Gaussian benchmark.
// For F(w) = exp(-alpha*(w-a)^2), G(w) = exp(-alpha*(w-b)^2) (real),
// (F * G)(Omega) = int dw F(w) G(w + Omega)
//                = sqrt(pi/(2*alpha)) * exp(-alpha/2 * (Omega - (b-a))^2).
// Peak at Omega = b - a.
// =============================================================================
TEST_CASE("real_axis_conv_gaussian_cross_correlation", "[real_axis][conv]")
{
  // Wide window to make the Gaussian integral tail negligible.
  const double w_max     = 12.0;
  const long   N_w       = 401;       // dense fermionic grid
  const double Omega_max = 8.0;
  const long   N_Omega   = 200;
  const long   N_t       = 256;
  const double T_window  = 64.0;       // dt = 0.25; freq_max*dt = 12*0.25 = 3.0 < pi
  // Nyquist requires freq_max*dt <= pi, but for accuracy we want freq_max*dt
  // well under pi to avoid coordinate aliasing in the NUFFT.
  const double beta = 50.0;
  const double mu   = 0.0;

  // The above choice of T_window=64, N_t=256 gives dt=0.25, freq_max*dt = 3.0.
  // freq_max*dt > pi violates Nyquist; reduce T_window.
  // Use T_window = 16.0, dt = 0.0625, freq_max*dt = 0.75 < pi. Safe.
  const double T_window_used = 16.0;

  auto grid = real_freq_grid_t::make_uniform(
      beta, mu, w_max, N_w, Omega_max, N_Omega, N_t, T_window_used);

  real_axis_conv_t conv(grid, /*ntrans*/ 1, /*eps*/ 1e-10);

  const double alpha = 1.0;
  const double a     = -0.5;
  const double b     = +1.5;

  nda::array<cval_t,2> F(1, N_w), G(1, N_w);
  for (long j = 0; j < N_w; ++j) {
    const double wj = grid.w()(j);
    F(0, j) = cval_t(std::exp(-alpha*(wj - a)*(wj - a)), 0.0);
    G(0, j) = cval_t(std::exp(-alpha*(wj - b)*(wj - b)), 0.0);
  }
  // cross_correlate applies quadrature weights internally.

  nda::array<cval_t,2> H(1, N_Omega);
  conv.cross_correlate(F, G, H,
                       real_axis_conv_t::grid_kind::fermionic,
                       real_axis_conv_t::grid_kind::bosonic);

  const double prefac = std::sqrt(M_PI / (2.0 * alpha));
  for (long l = 0; l < N_Omega; ++l) {
    const double Omega_l = grid.Omega()(l);
    const double dx      = Omega_l - (b - a);
    const double expected = prefac * std::exp(-0.5 * alpha * dx * dx);
    const double got_re   = H(0, l).real();
    // Loose tolerance because of NUFFT eps and finite-window truncation.
    REQUIRE(got_re == Approx(expected).margin(2e-3));
    REQUIRE(std::abs(H(0, l).imag()) < 5e-3);   // F, G real -> H real
  }
}

// =============================================================================
// Hilbert transform: Lorentzian benchmark.
// If Im X(w) = gamma / ((w - x0)^2 + gamma^2), then
//    Re X(w) = (w - x0) / ((w - x0)^2 + gamma^2),
// derived from analyticity of 1/(w - x0 + i*gamma).
// Equivalently:  Re X = (1/pi) PV int dw' Im X(w') / (w' - w).
// =============================================================================
TEST_CASE("real_axis_conv_lorentzian_hilbert", "[real_axis][conv]")
{
  const double w_max     = 60.0;       // wide window for Lorentzian tails
  const long   N_w       = 1024;
  const double Omega_max = 1.0;
  const long   N_Omega   = 8;
  const long   N_t       = 1024;
  const double T_window  = 32.0;       // dt = 1/32; freq_max*dt = 60*1/32 ~ 1.875 < pi
  const double beta = 50.0;
  const double mu   = 0.0;

  auto grid = real_freq_grid_t::make_uniform(
      beta, mu, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

  real_axis_conv_t conv(grid, /*ntrans*/ 1, /*eps*/ 1e-10);

  const double x0    = 0.7;
  const double gamma = 0.3;

  nda::array<double,2> ImX(1, N_w), ReX(1, N_w);
  for (long j = 0; j < N_w; ++j) {
    const double wj = grid.w()(j);
    const double dx = wj - x0;
    ImX(0, j) = gamma / (dx*dx + gamma*gamma);
  }
  // hilbert applies quadrature weights internally.

  conv.hilbert(ImX, ReX, real_axis_conv_t::grid_kind::fermionic);

  // Compare in the central window where boundary effects are smallest.
  // Window-tail truncation is O(gamma/w_max), which for w_max=60, gamma=0.3
  // produces an error ~ 0.005.
  long n_checked = 0, n_pass = 0;
  for (long j = 0; j < N_w; ++j) {
    const double wj = grid.w()(j);
    if (std::abs(wj) > w_max - 5.0) continue; // skip outer ring
    const double dx = wj - x0;
    const double expected = dx / (dx*dx + gamma*gamma);
    const double got      = ReX(0, j);
    ++n_checked;
    if (std::abs(got - expected) < std::max(2e-2,
        2e-2 * std::abs(expected) + 2e-2)) ++n_pass;
  }
  // Demand >97% of central points within tolerance.
  REQUIRE(n_pass > (97 * n_checked) / 100);
}

// =============================================================================
// Round-trip: applying Hilbert twice should give -ImX (since H^2 = -1 on
// functions vanishing at infinity).
// =============================================================================
TEST_CASE("real_axis_conv_hilbert_squared_negates", "[real_axis][conv]")
{
  const double w_max     = 80.0;
  const long   N_w       = 1024;
  const double Omega_max = 1.0;
  const long   N_Omega   = 8;
  const long   N_t       = 1024;
  const double T_window  = 32.0;
  const double beta = 50.0;
  const double mu   = 0.0;

  auto grid = real_freq_grid_t::make_uniform(
      beta, mu, w_max, N_w, Omega_max, N_Omega, N_t, T_window);
  real_axis_conv_t conv(grid, 1, 1e-10);

  const double x0 = 0.0, gamma = 0.5;
  nda::array<double,2> X(1, N_w), HX(1, N_w), HHX(1, N_w);
  for (long j = 0; j < N_w; ++j) {
    const double wj = grid.w()(j);
    X(0, j) = gamma / ((wj - x0)*(wj - x0) + gamma*gamma);
  }
  conv.hilbert(X,  HX,  real_axis_conv_t::grid_kind::fermionic);
  conv.hilbert(HX, HHX, real_axis_conv_t::grid_kind::fermionic);

  // H^2 X = -X for X with sufficient decay. Check central window.
  long n_checked = 0, n_pass = 0;
  for (long j = 0; j < N_w; ++j) {
    const double wj = grid.w()(j);
    if (std::abs(wj) > w_max - 10.0) continue;
    ++n_checked;
    if (std::abs(HHX(0, j) + X(0, j)) < 5e-2) ++n_pass;
  }
  REQUIRE(n_pass > (90 * n_checked) / 100);
}

} // namespace gw_real_axis_tests
