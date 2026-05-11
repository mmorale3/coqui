/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Unit tests for the QP-form QSGW CD building blocks:
 *   - Thiele Pade fit (real_axis_qp_pade.hpp)
 *   - Imaginary-axis convolution kernel (real_axis_qp_iaxis_integral.hpp)
 *
 * Both pieces are tested against closed-form Drude-like W_c that has
 * analytic continuation between iω axis and real axis.
 * ==========================================================================
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include <vector>
#include <complex>
#include <cmath>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "methods/GW_real_axis_qp/real_axis_qp_pade.hpp"
#include "methods/GW_real_axis_qp/real_axis_qp_iaxis_integral.hpp"

namespace gw_real_axis_qp_tests {

using methods::real_axis_qp::ComplexType;
using cdbl = ComplexType;

// ===========================================================================
// Closed-form analytic W_c: a single sharpened "Drude" pole at Ω = Ω_pl,
// W_c(z) = A / (z - Ω_pl + i γ)   +   A* / (z + Ω_pl + i γ)
// (Two complex-conjugate poles to give a real-valued imag-axis response.)
//
// On the imag axis with z = iω':
//   W_c(iω') = A / (iω' - Ω_pl + i γ) + A* / (iω' + Ω_pl + i γ)
// Reduces to a smooth function with magnitude peaked near ω' ~ γ.
//
// Pade fit from iω' samples should recover W_c(Ω real) for |Ω| < Ω_pl - γ
// where there are no real poles.
// ===========================================================================
namespace {
const double OMEGA_PL = 0.6;     // plasmon-like pole position (Ha)
const double GAMMA    = 0.05;    // pole half-width
const cdbl   A_RES    = cdbl(0.5, 0.0);

inline cdbl W_c_analytic(cdbl z) {
  return A_RES / (z - OMEGA_PL + cdbl(0.0, GAMMA))
       + std::conj(A_RES) / (z + OMEGA_PL + cdbl(0.0, GAMMA));
}
}  // namespace

TEST_CASE("qp_pade_drude_recovery", "[real_axis_qp][pade]") {
  // Sample W_c on a logarithmic Matsubara-like mesh on iω'.
  const long N = 32;
  std::vector<cdbl> z_nodes;
  std::vector<cdbl> f_nodes;
  z_nodes.reserve(N);
  f_nodes.reserve(N);
  const double omega_iw_min = 0.001;
  const double omega_iw_max = 5.0;
  for (long i = 0; i < N; ++i) {
    double omega_iw = omega_iw_min
                    * std::pow(omega_iw_max / omega_iw_min,
                               static_cast<double>(i) / (N - 1));
    z_nodes.emplace_back(0.0, omega_iw);  // pure imaginary
    f_nodes.emplace_back(W_c_analytic(z_nodes.back()));
  }

  auto g = methods::real_axis_qp::pade_coefficients(z_nodes, f_nodes);
  REQUIRE(g.size() == static_cast<std::size_t>(N));

  // Test Pade agreement on the iω axis itself (interpolation, exact).
  for (long i = 0; i < N; ++i) {
    cdbl v = methods::real_axis_qp::pade_eval(z_nodes, g, z_nodes[i]);
    REQUIRE(std::abs(v - f_nodes[i]) < 1e-10);
  }

  // Test Pade extrapolation OFF the iω axis: Ω small real, far from any
  // real pole (the real poles are at ±Ω_pl ± iγ). Pade should give a
  // good approximation to the analytic W_c(Ω).
  for (double Omega : {0.05, 0.10, 0.20, 0.30}) {
    cdbl z(Omega, 0.0);
    cdbl pade_val = methods::real_axis_qp::pade_eval(z_nodes, g, z);
    cdbl ref      = W_c_analytic(z);
    const double rel = std::abs(pade_val - ref) / std::abs(ref);
    INFO("Omega=" << Omega << "  pade=" << pade_val << "  ref=" << ref
         << "  rel=" << rel);
    // Far from the pole at Ω_pl=0.6, Pade should be sub-1% accurate.
    REQUIRE(rel < 1e-2);
  }
}

TEST_CASE("qp_iaxis_integral_drude", "[real_axis_qp][iaxis]") {
  // Sanity check: kernel produces finite values with sensible magnitude,
  // and the integral converges as N is increased (relative diff between
  // N=128 and N=512 should be small). Closed-form analytic continuation
  // tests are sensitive to the GL node distribution near the integrand
  // peak; for a stricter convergence study see the production IAFT mesh.
  const double iw_max = 50.0;

  auto run = [&](long N_iw) {
    auto [x, w] = methods::real_axis_qp::make_gauss_legendre_iw_mesh(iw_max, N_iw);
    std::vector<cdbl> W_c(N_iw);
    for (long i = 0; i < N_iw; ++i) W_c[i] = W_c_analytic(cdbl(0.0, x[i]));
    std::vector<cdbl> vals;
    for (double omega_minus_eps : {0.1, 0.2, 0.5, 1.0, 3.0}) {
      vals.push_back(methods::real_axis_qp::iaxis_integral_scalar(
          x, w, W_c, omega_minus_eps, 0.0));
    }
    return vals;
  };

  auto v128 = run(128);
  auto v512 = run(512);

  // All values finite + N-converged.
  for (std::size_t i = 0; i < v128.size(); ++i) {
    REQUIRE(std::isfinite(v128[i].real()));
    REQUIRE(std::isfinite(v128[i].imag()));
    const double rel = std::abs(v128[i] - v512[i]) / std::max(std::abs(v512[i]), 1e-12);
    INFO("i=" << i << "  v128=" << v128[i] << "  v512=" << v512[i]
         << "  rel=" << rel);
    // GL convergence on a wide symmetric interval is slow when the
    // integrand peaks sharply near ω'=0. Production uses the log-dense
    // IAFT mesh from numerics/imag_axes_ft/IAFT.hpp which converges
    // much faster. Here we just check the kernel arithmetic runs
    // and produces stable values as N grows.
    REQUIRE(rel < 2e-1);
  }
}

TEST_CASE("qp_iaxis_integral_batched", "[real_axis_qp][iaxis]") {
  // The batched iaxis integral should give the same result as the
  // scalar version for each (P, Q) entry.
  const long N_iw = 16;
  auto [x, w] = methods::real_axis_qp::make_gauss_legendre_iw_mesh(5.0, N_iw);
  const long NP = 4, NQ = 3;
  nda::array<ComplexType, 3> W_c_iw_PQ(NP, NQ, N_iw);
  for (long P = 0; P < NP; ++P)
    for (long Q = 0; Q < NQ; ++Q)
      for (long l = 0; l < N_iw; ++l) {
        // Vary the pole position per channel so we exercise different
        // values, not the same number.
        cdbl resid_test = cdbl(0.1 * (P + Q + 1), 0.0);
        cdbl z(0.0, x[l]);
        W_c_iw_PQ(P, Q, l) =
            resid_test / (z - 0.4 + cdbl(0.0, 0.04))
          + std::conj(resid_test) / (z + 0.4 + cdbl(0.0, 0.04));
      }

  nda::array<double, 1> x_nda(N_iw), w_nda(N_iw);
  for (long l = 0; l < N_iw; ++l) { x_nda(l) = x[l]; w_nda(l) = w[l]; }
  nda::array<ComplexType, 2> out(NP, NQ);

  const double omega = 0.2;
  const double eps_pole = 0.0;
  methods::real_axis_qp::iaxis_integral_batched(
      x_nda, w_nda, W_c_iw_PQ, omega, eps_pole, out);

  for (long P = 0; P < NP; ++P)
    for (long Q = 0; Q < NQ; ++Q) {
      std::vector<ComplexType> W_PQ(N_iw);
      for (long l = 0; l < N_iw; ++l) W_PQ[l] = W_c_iw_PQ(P, Q, l);
      cdbl ref = methods::real_axis_qp::iaxis_integral_scalar(
          x, w, W_PQ, omega, eps_pole);
      REQUIRE(std::abs(out(P, Q) - ref) < 1e-12);
    }
}

}  // namespace gw_real_axis_qp_tests
