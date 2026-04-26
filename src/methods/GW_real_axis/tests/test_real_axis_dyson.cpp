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
#include "methods/GW_real_axis/real_axis_dyson.hpp"

#include <complex>

namespace gw_real_axis_tests
{

using cval_t = std::complex<double>;

// =============================================================================
// Scalar limit: V = v*I, Pi = p*I, expected W = v/(1-v*p) * I.
// =============================================================================
TEST_CASE("real_axis_dyson_scalar_identity", "[real_axis][dyson]")
{
  const long N = 5;
  nda::array<cval_t, 2> V(N, N), Pi(N, N), W(N, N);
  V = cval_t(0.0, 0.0);
  Pi = cval_t(0.0, 0.0);
  const cval_t v(0.7, 0.0);
  const cval_t p(0.3, 0.1);
  for (long P = 0; P < N; ++P) {
    V(P, P)  = v;
    Pi(P, P) = p;
  }
  methods::real_axis::solve_dyson_W_aux(V, Pi, W);

  const cval_t expected = v / (cval_t(1.0, 0.0) - v * p);
  for (long P = 0; P < N; ++P) {
    REQUIRE(W(P, P).real() == Approx(expected.real()).epsilon(1e-12));
    REQUIRE(W(P, P).imag() == Approx(expected.imag()).epsilon(1e-12));
    for (long Q = 0; Q < N; ++Q) if (Q != P)
      REQUIRE(std::abs(W(P, Q)) < 1e-12);
  }
}

// =============================================================================
// 2x2 analytic check.
//   V = [[a, b], [b*, c]],  Pi = [[p, 0], [0, q]]
//   M = I - V*Pi = [[1 - a*p, -b*q], [-b**p, 1 - c*q]]
//   W = M^{-1} * V
// We check via direct matrix multiplication that (I - V*Pi) * W == V.
// =============================================================================
TEST_CASE("real_axis_dyson_2x2_consistency", "[real_axis][dyson]")
{
  const long N = 2;
  nda::array<cval_t, 2> V(N, N), Pi(N, N), W(N, N);
  V(0,0) = cval_t(1.0, 0.0);
  V(0,1) = cval_t(0.4, 0.1);
  V(1,0) = std::conj(V(0,1));
  V(1,1) = cval_t(0.7, 0.0);
  Pi(0,0) = cval_t(0.2, 0.05);
  Pi(0,1) = cval_t(0.0, 0.0);
  Pi(1,0) = cval_t(0.0, 0.0);
  Pi(1,1) = cval_t(0.1, -0.05);

  methods::real_axis::solve_dyson_W_aux(V, Pi, W);

  // Verify (I - V*Pi) * W == V.
  nda::array<cval_t, 2> VP(N, N), I_VP(N, N), Mout(N, N);
  for (long i = 0; i < N; ++i)
    for (long j = 0; j < N; ++j) {
      cval_t acc(0.0, 0.0);
      for (long k = 0; k < N; ++k) acc += V(i, k) * Pi(k, j);
      VP(i, j) = acc;
      I_VP(i, j) = (i == j ? cval_t(1.0, 0.0) : cval_t(0.0, 0.0)) - VP(i, j);
    }
  for (long i = 0; i < N; ++i)
    for (long j = 0; j < N; ++j) {
      cval_t acc(0.0, 0.0);
      for (long k = 0; k < N; ++k) acc += I_VP(i, k) * W(k, j);
      Mout(i, j) = acc;
    }
  for (long i = 0; i < N; ++i)
    for (long j = 0; j < N; ++j) {
      REQUIRE(Mout(i, j).real() == Approx(V(i, j).real()).margin(1e-10));
      REQUIRE(Mout(i, j).imag() == Approx(V(i, j).imag()).margin(1e-10));
    }
}

// =============================================================================
// High-frequency limit: as Pi -> 0, W -> V.
// =============================================================================
TEST_CASE("real_axis_dyson_high_frequency_limit", "[real_axis][dyson]")
{
  const long N = 4;
  nda::array<cval_t, 2> V(N, N), Pi(N, N), W(N, N);
  for (long i = 0; i < N; ++i)
    for (long j = 0; j < N; ++j)
      V(i, j) = (i == j ? cval_t(1.0 + 0.1*i, 0.0) : cval_t(0.05*(i-j), 0.0));
  Pi = cval_t(0.0, 0.0);
  // Tiny Pi simulating high-frequency tail.
  for (long i = 0; i < N; ++i) Pi(i, i) = cval_t(1e-8, 1e-8);

  methods::real_axis::solve_dyson_W_aux(V, Pi, W);

  for (long i = 0; i < N; ++i)
    for (long j = 0; j < N; ++j) {
      REQUIRE(W(i, j).real() == Approx(V(i, j).real()).margin(1e-6));
      REQUIRE(std::abs(W(i, j).imag()) < 1e-6);
    }
}

} // namespace gw_real_axis_tests
