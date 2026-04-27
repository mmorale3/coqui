/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Unit test for the real-axis q->0 divergence helpers.
 * ==========================================================================
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "methods/GW_real_axis/real_axis_div_utils.hpp"

#include <cmath>
#include <complex>

namespace gw_real_axis_tests
{

using cval_t = std::complex<double>;

// ===========================================================================
// Reference: hand-compute the formula
//   eps^-1(q, O) = (|q|^2 / 4pi) * Vol * sum_PQ chi_bar(q,P) W(q,P,Q,O) conj(chi_bar(q,Q))
// for a tiny synthetic (Nq=2, Naux=2, N_Omega=2) system and compare against
// the helper output. Tolerance is machine precision.
// ===========================================================================
TEST_CASE("real_axis_eps_inv_head_O_synthetic",
          "[real_axis][div_utils][eps_inv]")
{
  const long Nq   = 2;
  const long Naux = 2;
  const long N_O  = 2;

  // Q-points: iq=0 is Gamma (|q|=0), iq=1 has |q|^2 = 0.25.
  nda::array<double, 2> Qpts(Nq, 3);
  Qpts(0, nda::range::all) = 0.0;
  Qpts(1, 0) = 0.5; Qpts(1, 1) = 0.0; Qpts(1, 2) = 0.0;

  nda::array<cval_t, 2> chi_bar_qu(Nq, Naux);
  chi_bar_qu(0, 0) = cval_t(1.0, 0.0);  chi_bar_qu(0, 1) = cval_t(0.5, 0.5);
  chi_bar_qu(1, 0) = cval_t(2.0, 1.0);  chi_bar_qu(1, 1) = cval_t(-1.0, 0.5);

  nda::array<cval_t, 4> W_qPQO(Nq, Naux, Naux, N_O);
  // Fill with a simple, distinguishable pattern.
  for (long iq = 0; iq < Nq; ++iq)
    for (long P = 0; P < Naux; ++P)
      for (long Q = 0; Q < Naux; ++Q)
        for (long iO = 0; iO < N_O; ++iO)
          W_qPQO(iq, P, Q, iO) = cval_t(0.1*(iq+1) + 0.2*P + 0.3*Q + 0.4*iO,
                                         0.05*(iq+P+Q+iO));

  const double volume = 17.5;

  nda::array<cval_t, 2> eps_inv_qO(Nq, N_O);
  nda::array<cval_t, 1> eps_inv_O(N_O);
  methods::real_axis::compute_eps_inv_head_O(W_qPQO, Qpts, chi_bar_qu, volume,
                                              eps_inv_qO, eps_inv_O);

  const double fpi = 4.0 * M_PI;

  // iq=0 (Gamma): formula prefactor (|q|^2 / 4pi) is zero, so output is 0.
  for (long iO = 0; iO < N_O; ++iO) {
    REQUIRE(std::abs(eps_inv_qO(0, iO)) < 1e-15);
  }

  // iq=1 (|q|^2 = 0.25): hand-compute the double sum.
  for (long iO = 0; iO < N_O; ++iO) {
    cval_t expected(0.0, 0.0);
    for (long P = 0; P < Naux; ++P)
      for (long Q = 0; Q < Naux; ++Q)
        expected += chi_bar_qu(1, P) * W_qPQO(1, P, Q, iO)
                                     * std::conj(chi_bar_qu(1, Q));
    expected *= (0.25 / fpi) * volume;
    REQUIRE(std::abs(eps_inv_qO(1, iO) - expected) < 1e-13);
  }

  // q->0 estimate: smallest-|q| (excluding Gamma) -> iq=1.
  for (long iO = 0; iO < N_O; ++iO) {
    REQUIRE(std::abs(eps_inv_O(iO) - eps_inv_qO(1, iO)) < 1e-15);
  }
}

// ===========================================================================
// Edge case: only Gamma is present (Nq=1). The smallest-|q| treatment has
// nothing to extrapolate from and should produce 0 in eps_inv_O.
// ===========================================================================
TEST_CASE("real_axis_eps_inv_head_O_gamma_only",
          "[real_axis][div_utils][eps_inv]")
{
  const long Nq   = 1;
  const long Naux = 2;
  const long N_O  = 3;

  nda::array<double, 2> Qpts(Nq, 3);
  Qpts() = 0.0;

  nda::array<cval_t, 2> chi_bar_qu(Nq, Naux);
  chi_bar_qu = cval_t(1.0, 0.0);

  nda::array<cval_t, 4> W_qPQO(Nq, Naux, Naux, N_O);
  W_qPQO = cval_t(2.0, -1.0);

  nda::array<cval_t, 2> eps_inv_qO(Nq, N_O);
  nda::array<cval_t, 1> eps_inv_O(N_O);
  methods::real_axis::compute_eps_inv_head_O(W_qPQO, Qpts, chi_bar_qu, 1.0,
                                              eps_inv_qO, eps_inv_O);

  // All zeros: Gamma contributes nothing by the formula, no other q to use.
  for (long iO = 0; iO < N_O; ++iO) {
    REQUIRE(std::abs(eps_inv_qO(0, iO)) < 1e-15);
    REQUIRE(std::abs(eps_inv_O(iO))     < 1e-15);
  }
}

} // namespace gw_real_axis_tests
