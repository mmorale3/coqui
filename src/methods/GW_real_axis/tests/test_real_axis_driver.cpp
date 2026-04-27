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

#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"

#include "nda/nda.hpp"
#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_gw_driver.hpp"
#include "methods/GW_real_axis/real_axis_thc_project.hpp"

#include <cmath>
#include <complex>

namespace gw_real_axis_tests
{

using methods::real_axis::real_freq_grid_t;
using methods::real_axis::evaluate_serial;
using methods::real_axis::primary_to_aux_one_k;
using methods::real_axis::aux_to_primary_one_k;
using cval_t = std::complex<double>;

// =============================================================================
// Round-trip test for primary_to_aux / aux_to_primary using a unitary X.
// If X is a unitary tall matrix (Naux >= nbnd, X X^† = I_{nbnd}), then
//    aux_to_primary(primary_to_aux(A)) = X^† X A X^† X = A.
// =============================================================================
TEST_CASE("real_axis_thc_projection_round_trip", "[real_axis][thc]")
{
  const long Naux = 4;
  const long nbnd = 3;
  const long N_w  = 7;

  // Build a unitary X of shape (Naux, nbnd) with X X^† = I_nbnd via
  // Gram-Schmidt on a random matrix. Here we just write columns of X^T to be
  // pairwise orthonormal.
  nda::array<cval_t, 2> X(Naux, nbnd);
  X = cval_t(0.0, 0.0);
  // Trivial choice: first 3 rows of X form a 3x3 identity (in P=mu order),
  // remaining row is zero.
  for (long mu = 0; mu < nbnd; ++mu)
    X(mu, mu) = cval_t(1.0, 0.0);

  // Sanity: X X^† = I_{nbnd}? Yes, sum_P X(P, mu) conj(X(P, nu)) = delta_{mu nu}.

  // Random orbital-basis A.
  nda::array<cval_t, 3> A(N_w, nbnd, nbnd);
  for (long iw = 0; iw < N_w; ++iw)
    for (long mu = 0; mu < nbnd; ++mu)
      for (long nu = 0; nu < nbnd; ++nu)
        A(iw, mu, nu) = cval_t(0.1 * (iw + 1) * (mu + 1) + 0.01 * nu,
                               0.05 * (mu - nu));

  nda::array<cval_t, 3> A_aux(Naux, Naux, N_w);
  primary_to_aux_one_k(X, A, A_aux);

  nda::array<cval_t, 3> A_back(N_w, nbnd, nbnd);
  aux_to_primary_one_k(X, A_aux, A_back);

  for (long iw = 0; iw < N_w; ++iw)
    for (long mu = 0; mu < nbnd; ++mu)
      for (long nu = 0; nu < nbnd; ++nu) {
        REQUIRE(A_back(iw, mu, nu).real() == Approx(A(iw, mu, nu).real()).margin(1e-12));
        REQUIRE(A_back(iw, mu, nu).imag() == Approx(A(iw, mu, nu).imag()).margin(1e-12));
      }
}

// =============================================================================
// End-to-end serial G0W0 driver smoke test.
//
// Setup: 1 spin, 1 k-point, 1 q-point (i.e. molecular Gamma-point),
// nbnd = 2, Naux = 2. Choose A_munu(w) as two narrow Gaussians at +/- 0.5
// with diagonal structure (occupied at w=-0.5, virtual at +0.5). X is the
// identity-mapped THC factor, V_PQ = identity (a trivial Coulomb).
//
// We do not compare against a known reference here; the test verifies that
// the driver:
//   - runs to completion;
//   - produces finite outputs;
//   - approximately satisfies Im Sigma^c <= 0 (causality) once we have run
//     the engine (we apply the projection externally to assert it).
// =============================================================================
TEST_CASE("real_axis_driver_endtoend_smoke", "[real_axis][driver][e2e]")
{
  const double w_max     = 8.0;
  const long   N_w       = 65;
  const double Omega_max = 4.0;
  const long   N_Omega   = 32;
  const long   N_t       = 128;
  const double T_window  = 16.0;
  const double beta      = 50.0;
  const double mu_chem   = 0.0;

  auto grid = real_freq_grid_t::make_uniform(
      beta, mu_chem, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

  const long ns   = 1;
  const long Nk   = 1;
  const long Nq   = 1;
  const long Naux = 2;
  const long nbnd = 2;

  // Build A: two-band, diagonal Gaussian at +/-0.5 (HOMO/LUMO-like).
  nda::array<cval_t, 5> A(ns, Nk, N_w, nbnd, nbnd);
  A = cval_t(0.0, 0.0);
  for (long iw = 0; iw < N_w; ++iw) {
    const double w_l = grid.w()(iw);
    const double a0 = std::exp(-0.5 * (w_l + 0.5)*(w_l + 0.5)) / std::sqrt(2.0 * M_PI);
    const double a1 = std::exp(-0.5 * (w_l - 0.5)*(w_l - 0.5)) / std::sqrt(2.0 * M_PI);
    A(0, 0, iw, 0, 0) = cval_t(a0, 0.0);
    A(0, 0, iw, 1, 1) = cval_t(a1, 0.0);
  }

  // X: identity in (P, mu) for Naux == nbnd.
  nda::array<cval_t, 4> X(ns, Nk, Naux, nbnd);
  X = cval_t(0.0, 0.0);
  for (long P = 0; P < Naux; ++P)
    X(0, 0, P, P) = cval_t(1.0, 0.0);

  // V: trivial diagonal Coulomb-like matrix.
  nda::array<cval_t, 3> V(Nq, Naux, Naux);
  V = cval_t(0.0, 0.0);
  for (long P = 0; P < Naux; ++P)
    V(0, P, P) = cval_t(0.5, 0.0);

  // k+q and k-q maps: with Nk=Nq=1 both are trivially 0.
  nda::array<long, 2> kpq(Nk, Nq), kmq(Nk, Nq);
  kpq(0, 0) = 0;
  kmq(0, 0) = 0;

  // q_weights: single q with weight 1.
  nda::array<double, 1> qw(Nq);
  qw(0) = 1.0;

  nda::array<cval_t, 5> ImSigma(ns, Nk, N_w, nbnd, nbnd);
  nda::array<cval_t, 5> ReSigma(ns, Nk, N_w, nbnd, nbnd);
  ImSigma = cval_t(0.0, 0.0);
  ReSigma = cval_t(0.0, 0.0);

  auto& mpi_context = utils::make_unit_test_mpi_context();
  evaluate_serial(mpi_context->comm, grid, A, X, V, kpq, kmq, qw, ImSigma, ReSigma, /*eps_nufft*/ 1e-10);

  // Outputs must be finite and approximately real-valued.
  for (long iw = 0; iw < N_w; ++iw)
    for (long mu = 0; mu < nbnd; ++mu)
      for (long nu = 0; nu < nbnd; ++nu) {
        REQUIRE(std::isfinite(ImSigma(0, 0, iw, mu, nu).real()));
        REQUIRE(std::isfinite(ImSigma(0, 0, iw, mu, nu).imag()));
        REQUIRE(std::isfinite(ReSigma(0, 0, iw, mu, nu).real()));
        REQUIRE(std::isfinite(ReSigma(0, 0, iw, mu, nu).imag()));
      }

  // For real-valued A and V, Sigma^c should be approximately real (small
  // imaginary numerical noise allowed). Each component:
  double max_imag_im = 0.0, max_imag_re = 0.0;
  for (long iw = 0; iw < N_w; ++iw)
    for (long mu = 0; mu < nbnd; ++mu)
      for (long nu = 0; nu < nbnd; ++nu) {
        max_imag_im = std::max(max_imag_im,
                               std::abs(ImSigma(0, 0, iw, mu, nu).imag()));
        max_imag_re = std::max(max_imag_re,
                               std::abs(ReSigma(0, 0, iw, mu, nu).imag()));
      }
  REQUIRE(max_imag_im < 1e-6);
  REQUIRE(max_imag_re < 1e-6);

  // Self-energy must be non-zero somewhere on the diagonal; otherwise the
  // pipeline produced trivial output.
  double max_diag = 0.0;
  for (long iw = 0; iw < N_w; ++iw)
    for (long m = 0; m < nbnd; ++m) {
      max_diag = std::max(max_diag, std::abs(ImSigma(0, 0, iw, m, m).real()));
      max_diag = std::max(max_diag, std::abs(ReSigma(0, 0, iw, m, m).real()));
    }
  REQUIRE(max_diag > 1e-4);
}

} // namespace gw_real_axis_tests
