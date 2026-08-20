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
#include "utilities/test_common.hpp"

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

// =============================================================================
// Device-vs-host: exercise the cross_correlate / convolve / hilbert paths on
// the device backend (real_axis_conv_mem_t<DEVICE_MEMORY>) and require
// element-wise agreement with the host result. Gated on COQUI_HAVE_CUFINUFFT
// (which subsumes ENABLE_DEVICE / nda::cuarray availability) -- on host-only
// builds these cases are excluded at compile time.
//
// This is the end-to-end check on the lifted static_asserts in conv.hpp:
//   * weight broadcast (1D × 2D) via nda::tensor::elementwise on cuTENSOR
//   * forward / backward NUFFT via cuFINUFFT plan
//   * Hadamard / sgn-multiply via nda::map (MEM-agnostic)
//   * (dt / 2π) scalar scale via nda::map
//   * real()-extraction in hilbert via nda::map
// =============================================================================
#if defined(COQUI_HAVE_CUFINUFFT)

namespace {

template<methods::real_axis::grid_kind SrcKind,
         methods::real_axis::grid_kind DstKind>
void run_cross_correlate_devhost(real_freq_grid_t const& grid, long B)
{
  using namespace methods::real_axis;
  const long N_src = (SrcKind == grid_kind::fermionic ? grid.N_w() : grid.N_Omega());
  const long N_dst = (DstKind == grid_kind::fermionic ? grid.N_w() : grid.N_Omega());

  nda::array<cval_t,2> F_h(B, N_src), G_h(B, N_src), H_h(B, N_dst);
  utils::fillRandomArray(F_h);
  utils::fillRandomArray(G_h);

  detail::real_axis_conv_base_t<HOST_MEMORY> conv_h(grid, B, 1e-10);
  conv_h.cross_correlate(F_h, G_h, H_h, SrcKind, DstKind);

  auto F_d = memory::to_memory_space<DEVICE_MEMORY>(F_h);
  auto G_d = memory::to_memory_space<DEVICE_MEMORY>(G_h);
  memory::array<DEVICE_MEMORY, cval_t, 2> H_d(B, N_dst);
  detail::real_axis_conv_base_t<DEVICE_MEMORY> conv_d(grid, B, 1e-10);
  conv_d.cross_correlate(F_d, G_d, H_d, SrcKind, DstKind);

  auto H_d_h = nda::to_host(H_d);
  for (long b = 0; b < B; ++b)
    for (long l = 0; l < N_dst; ++l)
      REQUIRE(std::abs(H_h(b, l) - H_d_h(b, l)) < 1e-6);
}

void run_convolve_devhost(real_freq_grid_t const& grid, long B,
                          methods::real_axis::grid_kind kind)
{
  using namespace methods::real_axis;
  const long N = (kind == grid_kind::fermionic ? grid.N_w() : grid.N_Omega());

  nda::array<cval_t,2> F_h(B, N), G_h(B, N), H_h(B, N);
  utils::fillRandomArray(F_h);
  utils::fillRandomArray(G_h);

  detail::real_axis_conv_base_t<HOST_MEMORY> conv_h(grid, B, 1e-10);
  conv_h.convolve(F_h, G_h, H_h, kind);

  auto F_d = memory::to_memory_space<DEVICE_MEMORY>(F_h);
  auto G_d = memory::to_memory_space<DEVICE_MEMORY>(G_h);
  memory::array<DEVICE_MEMORY, cval_t, 2> H_d(B, N);
  detail::real_axis_conv_base_t<DEVICE_MEMORY> conv_d(grid, B, 1e-10);
  conv_d.convolve(F_d, G_d, H_d, kind);

  auto H_d_h = nda::to_host(H_d);
  for (long b = 0; b < B; ++b)
    for (long l = 0; l < N; ++l)
      REQUIRE(std::abs(H_h(b, l) - H_d_h(b, l)) < 1e-6);
}

void run_hilbert_devhost(real_freq_grid_t const& grid, long B,
                         methods::real_axis::grid_kind kind)
{
  using namespace methods::real_axis;
  const long N = (kind == grid_kind::fermionic ? grid.N_w() : grid.N_Omega());

  nda::array<double,2> ImX_h(B, N), ReX_h(B, N);
  utils::fillRandomArray(ImX_h);

  detail::real_axis_conv_base_t<HOST_MEMORY> conv_h(grid, B, 1e-10);
  conv_h.hilbert(ImX_h, ReX_h, kind);

  auto ImX_d = memory::to_memory_space<DEVICE_MEMORY>(ImX_h);
  memory::array<DEVICE_MEMORY, double, 2> ReX_d(B, N);
  detail::real_axis_conv_base_t<DEVICE_MEMORY> conv_d(grid, B, 1e-10);
  conv_d.hilbert(ImX_d, ReX_d, kind);

  auto ReX_d_h = nda::to_host(ReX_d);
  for (long b = 0; b < B; ++b)
    for (long j = 0; j < N; ++j)
      REQUIRE(std::abs(ReX_h(b, j) - ReX_d_h(b, j)) < 1e-6);
}

} // anonymous namespace

TEST_CASE("real_axis_conv_device_vs_host_cross_correlate", "[real_axis][conv][device]")
{
  using methods::real_axis::grid_kind;
  // Modest grid -- the goal is to validate the wiring, not the physics.
  const double w_max = 8.0;
  const long N_w = 64, N_Omega = 32, N_t = 128;
  const double Omega_max = 4.0, T_window = 16.0;
  auto grid = real_freq_grid_t::make_uniform(
      /*beta*/ 50.0, /*mu*/ 0.0, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

  // Two batch sizes: 1 (single transform) and 4 (batched).
  run_cross_correlate_devhost<grid_kind::fermionic, grid_kind::bosonic>(grid, 1);
  run_cross_correlate_devhost<grid_kind::fermionic, grid_kind::bosonic>(grid, 4);
}

TEST_CASE("real_axis_conv_device_vs_host_convolve", "[real_axis][conv][device]")
{
  using methods::real_axis::grid_kind;
  const double w_max = 8.0;
  const long N_w = 64, N_Omega = 32, N_t = 128;
  const double Omega_max = 4.0, T_window = 16.0;
  auto grid = real_freq_grid_t::make_uniform(
      50.0, 0.0, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

  run_convolve_devhost(grid, 1, grid_kind::fermionic);
  run_convolve_devhost(grid, 4, grid_kind::fermionic);
}

TEST_CASE("real_axis_conv_device_vs_host_hilbert", "[real_axis][conv][device]")
{
  using methods::real_axis::grid_kind;
  const double w_max = 8.0;
  const long N_w = 64, N_Omega = 32, N_t = 128;
  const double Omega_max = 4.0, T_window = 16.0;
  auto grid = real_freq_grid_t::make_uniform(
      50.0, 0.0, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

  run_hilbert_devhost(grid, 1, grid_kind::fermionic);
  run_hilbert_devhost(grid, 4, grid_kind::fermionic);
}

#endif // COQUI_HAVE_CUFINUFFT

} // namespace gw_real_axis_tests
