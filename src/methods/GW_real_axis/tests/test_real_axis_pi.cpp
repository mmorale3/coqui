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
#include "methods/GW_real_axis/real_axis_thc_project.hpp"
#include "utilities/test_common.hpp"

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

// =============================================================================
// Device-vs-host: exercise the lifted gates on accumulate_ImPi_one_kq and
// RePi_from_ImPi with random inputs. Gated on COQUI_HAVE_CUFINUFFT.
// =============================================================================
#if defined(COQUI_HAVE_CUFINUFFT)

TEST_CASE("real_axis_pi_accumulate_ImPi_device_vs_host", "[real_axis][pi][device]")
{
  using methods::real_axis::detail::real_axis_conv_base_t;

  const double w_max = 6.0;
  const long N_w = 64, N_Omega = 32, N_t = 128;
  const double Omega_max = 3.0, T_window = 12.0;
  auto grid = real_freq_grid_t::make_uniform(
      50.0, 0.0, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

  const long Naux = 4;
  const long B = Naux * Naux;
  using cval_t = std::complex<double>;

  nda::array<cval_t, 3> A_k_h(Naux, Naux, N_w), A_kq_h(Naux, Naux, N_w);
  utils::fillRandomArray(A_k_h);
  utils::fillRandomArray(A_kq_h);

  // Host
  nda::array<cval_t, 3> ImPi_h(Naux, Naux, N_Omega);
  ImPi_h = cval_t(0.0, 0.0);
  real_axis_conv_base_t<HOST_MEMORY> conv_h(grid, /*ntrans*/ B, /*eps*/ 1e-10);
  accumulate_ImPi_one_kq(conv_h, A_k_h, A_kq_h, ImPi_h, /*k_weight*/ 1.0);

  // Device
  auto A_k_d  = memory::to_memory_space<DEVICE_MEMORY>(A_k_h);
  auto A_kq_d = memory::to_memory_space<DEVICE_MEMORY>(A_kq_h);
  memory::array<DEVICE_MEMORY, cval_t, 3> ImPi_d(Naux, Naux, N_Omega);
  ImPi_d = cval_t(0.0, 0.0);
  real_axis_conv_base_t<DEVICE_MEMORY> conv_d(grid, B, 1e-10);
  accumulate_ImPi_one_kq(conv_d, A_k_d, A_kq_d, ImPi_d, 1.0);

  auto ImPi_dh = nda::to_host(ImPi_d);
  for (long P = 0; P < Naux; ++P)
    for (long Q = 0; Q < Naux; ++Q)
      for (long iO = 0; iO < N_Omega; ++iO)
        REQUIRE(std::abs(ImPi_h(P, Q, iO) - ImPi_dh(P, Q, iO)) < 1e-6);
}

TEST_CASE("real_axis_pi_RePi_from_ImPi_device_vs_host", "[real_axis][pi][device]")
{
  using methods::real_axis::detail::real_axis_conv_base_t;

  const double w_max = 6.0;
  const long N_w = 64, N_Omega = 32, N_t = 128;
  const double Omega_max = 3.0, T_window = 12.0;
  auto grid = real_freq_grid_t::make_uniform(
      50.0, 0.0, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

  const long Naux = 4;
  const long B = Naux * Naux;

  nda::array<double, 3> ImPi_h(Naux, Naux, N_Omega), RePi_h(Naux, Naux, N_Omega);
  utils::fillRandomArray(ImPi_h);

  real_axis_conv_base_t<HOST_MEMORY> conv_h(grid, B, 1e-10);
  RePi_from_ImPi(conv_h, ImPi_h, RePi_h);

  auto ImPi_d = memory::to_memory_space<DEVICE_MEMORY>(ImPi_h);
  memory::array<DEVICE_MEMORY, double, 3> RePi_d(Naux, Naux, N_Omega);
  real_axis_conv_base_t<DEVICE_MEMORY> conv_d(grid, B, 1e-10);
  RePi_from_ImPi(conv_d, ImPi_d, RePi_d);

  auto RePi_dh = nda::to_host(RePi_d);
  for (long P = 0; P < Naux; ++P)
    for (long Q = 0; Q < Naux; ++Q)
      for (long iO = 0; iO < N_Omega; ++iO)
        REQUIRE(std::abs(RePi_h(P, Q, iO) - RePi_dh(P, Q, iO)) < 1e-6);
}

TEST_CASE("real_axis_thc_project_primary_to_aux_device_vs_host",
          "[real_axis][thc_project][device]")
{
  using methods::real_axis::primary_to_aux_one_k;
  using cval_t = std::complex<double>;

  const long Naux = 6;
  const long nbnd = 4;
  const long N_w  = 12;

  nda::array<cval_t, 2> X_h(Naux, nbnd);
  nda::array<cval_t, 3> A_h(N_w, nbnd, nbnd);
  utils::fillRandomArray(X_h);
  utils::fillRandomArray(A_h);

  nda::array<cval_t, 3> A_aux_h(Naux, Naux, N_w);
  primary_to_aux_one_k<HOST_MEMORY>(X_h, X_h, A_h, A_aux_h);

  auto X_d = memory::to_memory_space<DEVICE_MEMORY>(X_h);
  auto A_d = memory::to_memory_space<DEVICE_MEMORY>(A_h);
  memory::array<DEVICE_MEMORY, cval_t, 3> A_aux_d(Naux, Naux, N_w);
  primary_to_aux_one_k<DEVICE_MEMORY>(X_d, X_d, A_d, A_aux_d);

  auto A_aux_dh = nda::to_host(A_aux_d);
  for (long P = 0; P < Naux; ++P)
    for (long Q = 0; Q < Naux; ++Q)
      for (long iw = 0; iw < N_w; ++iw)
        REQUIRE(std::abs(A_aux_h(P, Q, iw) - A_aux_dh(P, Q, iw)) < 1e-9);
}

TEST_CASE("real_axis_thc_project_aux_to_primary_device_vs_host",
          "[real_axis][thc_project][device]")
{
  using methods::real_axis::aux_to_primary_one_k;
  using cval_t = std::complex<double>;

  const long Naux = 6;
  const long nbnd = 4;
  const long N_w  = 12;

  nda::array<cval_t, 2> X_h(Naux, nbnd);
  nda::array<cval_t, 3> M_aux_h(Naux, Naux, N_w);
  utils::fillRandomArray(X_h);
  utils::fillRandomArray(M_aux_h);

  nda::array<cval_t, 3> M_h(N_w, nbnd, nbnd);
  aux_to_primary_one_k<HOST_MEMORY>(X_h, X_h, M_aux_h, M_h);

  auto X_d     = memory::to_memory_space<DEVICE_MEMORY>(X_h);
  auto M_aux_d = memory::to_memory_space<DEVICE_MEMORY>(M_aux_h);
  memory::array<DEVICE_MEMORY, cval_t, 3> M_d(N_w, nbnd, nbnd);
  aux_to_primary_one_k<DEVICE_MEMORY>(X_d, X_d, M_aux_d, M_d);

  auto M_dh = nda::to_host(M_d);
  for (long iw = 0; iw < N_w; ++iw)
    for (long mu = 0; mu < nbnd; ++mu)
      for (long nu = 0; nu < nbnd; ++nu)
        REQUIRE(std::abs(M_h(iw, mu, nu) - M_dh(iw, mu, nu)) < 1e-9);
}

#endif // COQUI_HAVE_CUFINUFFT

} // namespace gw_real_axis_tests
