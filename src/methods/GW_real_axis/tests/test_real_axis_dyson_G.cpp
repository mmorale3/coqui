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
#include "methods/GW_real_axis/real_axis_dyson_G.hpp"

#include <cmath>
#include <complex>

namespace gw_real_axis_tests
{

using methods::real_axis::real_freq_grid_t;
using methods::real_axis::dyson_G_one_kw;
using methods::real_axis::dyson_update_A;
using methods::real_axis::find_mu_chem;
using methods::real_axis::project_causality_ImSigma;
using methods::real_axis::linear_mix;
using methods::real_axis::frobenius_diff;
using cval_t = std::complex<double>;

// =============================================================================
// Single-orbital, no self-energy: G^R = 1/(w + mu - eps + i eta).
// A(w) = (1/pi) eta / ((w + mu - eps)^2 + eta^2).
// =============================================================================
TEST_CASE("real_axis_dyson_G_single_band", "[real_axis][dysonG]")
{
  const long nbnd = 1;
  const double eps = 0.5;
  const double mu  = 0.3;
  const double eta = 0.05;
  nda::array<cval_t, 2> H(nbnd, nbnd), Sx(nbnd, nbnd), Re(nbnd, nbnd), Im(nbnd, nbnd);
  nda::array<cval_t, 2> G(nbnd, nbnd), A(nbnd, nbnd);
  H(0, 0)  = cval_t(eps, 0.0);
  Sx(0, 0) = cval_t(0.0, 0.0);
  Re(0, 0) = cval_t(0.0, 0.0);
  Im(0, 0) = cval_t(0.0, 0.0);

  const double w = 0.7;
  dyson_G_one_kw(H, Sx, Re, Im, w, mu, eta, G, A);

  const double dx = w + mu - eps;
  const cval_t G_expected = cval_t(1.0, 0.0) / cval_t(dx, eta);
  REQUIRE(G(0, 0).real() == Approx(G_expected.real()).margin(1e-12));
  REQUIRE(G(0, 0).imag() == Approx(G_expected.imag()).margin(1e-12));
  const double A_expected = -G_expected.imag() / M_PI;
  REQUIRE(A(0, 0).real() == Approx(A_expected).margin(1e-12));
}

// =============================================================================
// find_mu_chem: at zero T (large beta), with a single band at eps and
// N_elec = 1 (one electron per spin in spin-restricted), mu should bracket
// eps from above. Verify that mu(N=1) is close to the spectral peak.
// =============================================================================
TEST_CASE("real_axis_find_mu_chem_single_band", "[real_axis][mu]")
{
  const double w_max     = 8.0;
  const long   N_w       = 401;
  const double Omega_max = 1.0;
  const long   N_Omega   = 8;
  const long   N_t       = 64;
  const double T_window  = 16.0;
  const double beta = 200.0;
  const double mu0  = 0.0;

  auto grid = real_freq_grid_t::make_uniform(
      beta, mu0, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

  // A(w) = (1/pi) eta / ((w - eps)^2 + eta^2).
  const long ns = 1, Nk = 1, nbnd = 1;
  const double eps = 0.5, eta = 0.02;
  nda::array<cval_t, 5> A(N_w, ns, Nk, nbnd, nbnd);
  for (long iw = 0; iw < N_w; ++iw) {
    const double w_l = grid.w()(iw);
    const double v = (1.0 / M_PI) * eta / ((w_l - eps)*(w_l - eps) + eta*eta);
    A(iw, 0, 0, 0, 0) = cval_t(v, 0.0);
  }

  nda::array<double, 1> kw(Nk);
  kw(0) = 1.0;
  // For the single band at eps, integral of f(w) A(w) over all w with
  // mu = eps gives 1/2 (Fermi function at eps is 1/2). For N_elec = 1
  // we need mu to be slightly above eps. For N_elec = 0.5 we expect mu = eps.
  const double mu_found = find_mu_chem(grid, A, kw, /*N_elec*/ 0.5);
  // Lorentzian tails at finite eta and finite window contribute, so a
  // 1e-2 margin is appropriate.
  REQUIRE(mu_found == Approx(eps).margin(2e-2));
}

// =============================================================================
// Causality projection: positive Im Sigma diagonal entries are clipped to 0.
// =============================================================================
TEST_CASE("real_axis_causality_projection", "[real_axis][causality]")
{
  const long ns = 1, Nk = 1, N_w = 5, nbnd = 2;
  nda::array<cval_t, 5> S(ns, Nk, N_w, nbnd, nbnd);
  for (long iw = 0; iw < N_w; ++iw)
    for (long mu = 0; mu < nbnd; ++mu)
      for (long nu = 0; nu < nbnd; ++nu)
        S(0, 0, iw, mu, nu) = cval_t(0.5 * (iw - 2), 0.0); // -1, -0.5, 0, 0.5, 1
  project_causality_ImSigma(S);
  for (long iw = 0; iw < N_w; ++iw) {
    REQUIRE(S(0, 0, iw, 0, 0).real() <= 0.0);
    REQUIRE(S(0, 0, iw, 1, 1).real() <= 0.0);
    // Off-diagonal preserved.
    REQUIRE(S(0, 0, iw, 0, 1).real() == Approx(0.5 * (iw - 2)));
  }
}

// =============================================================================
// Linear mixing.
// =============================================================================
TEST_CASE("real_axis_linear_mix", "[real_axis][mix]")
{
  const long ns = 1, Nk = 1, N_w = 3, nbnd = 1;
  nda::array<cval_t, 5> S_old(ns, Nk, N_w, nbnd, nbnd);
  nda::array<cval_t, 5> S_in (ns, Nk, N_w, nbnd, nbnd);
  for (long iw = 0; iw < N_w; ++iw) {
    S_old(0, 0, iw, 0, 0) = cval_t(1.0, 0.0);
    S_in (0, 0, iw, 0, 0) = cval_t(2.0, 0.0);
  }
  linear_mix(S_old, S_in, 0.5);
  for (long iw = 0; iw < N_w; ++iw)
    REQUIRE(S_old(0, 0, iw, 0, 0).real() == Approx(1.5));

  // Frobenius diff sanity.
  for (long iw = 0; iw < N_w; ++iw)
    S_in(0, 0, iw, 0, 0) = cval_t(1.5 + 0.1 * iw, 0.0);
  const double d = frobenius_diff(S_old, S_in);
  // sum (0.1*iw)^2 over iw=0..2 = 0 + 0.01 + 0.04 = 0.05; sqrt = ~0.2236
  REQUIRE(d == Approx(std::sqrt(0.05)).margin(1e-12));
}

} // namespace gw_real_axis_tests
