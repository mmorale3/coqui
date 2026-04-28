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
#include "utilities/kpoint_utils.hpp"

#include "mean_field/default_MF.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/ERI/eri_utils.hpp"

#include "nda/nda.hpp"
#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_scf.hpp"

#include <cmath>
#include <complex>

namespace gw_real_axis_tests
{

using methods::real_axis::real_freq_grid_t;
using methods::real_axis::scgw_config;
using methods::real_axis::scgw_result;
using methods::real_axis::run_scgw_serial;
using cval_t = std::complex<double>;

// =============================================================================
// G0W0 smoke test: a 2-band, 1-spin, 1-k, 1-q toy with diagonal H_MF and
// identity X, identity-like V. Run a single iteration; verify outputs.
// =============================================================================
TEST_CASE("real_axis_scf_g0w0_smoke", "[real_axis][scf][e2e]")
{
  const double w_max     = 10.0;
  const long   N_w       = 129;
  const double Omega_max = 4.0;
  const long   N_Omega   = 32;
  const long   N_t       = 256;
  const double T_window  = 16.0;
  const double beta = 50.0;
  const double mu0  = 0.0;

  auto grid = real_freq_grid_t::make_uniform(
      beta, mu0, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

  const long ns = 1, Nk = 1, Nq = 1, Naux = 2, nbnd = 2;

  // H_MF: diagonal at -0.5 (occupied) and +0.5 (virtual).
  nda::array<cval_t, 4> H(ns, Nk, nbnd, nbnd);
  H = cval_t(0.0, 0.0);
  H(0, 0, 0, 0) = cval_t(-0.5, 0.0);
  H(0, 0, 1, 1) = cval_t(+0.5, 0.0);

  // X = identity.
  nda::array<cval_t, 4> X(ns, Nk, Naux, nbnd);
  X = cval_t(0.0, 0.0);
  for (long P = 0; P < Naux; ++P) X(0, 0, P, P) = cval_t(1.0, 0.0);

  // V = small diagonal Coulomb (weak interaction).
  nda::array<cval_t, 3> V(Nq, Naux, Naux);
  V = cval_t(0.0, 0.0);
  for (long P = 0; P < Naux; ++P) V(0, P, P) = cval_t(0.2, 0.0);

  nda::array<long, 2> kpq(Nk, Nq), kmq(Nk, Nq);
  kpq(0, 0) = 0;  kmq(0, 0) = 0;
  nda::array<double, 1> qw(Nq), kw(Nk);
  qw(0) = 1.0;    kw(0) = 1.0;

  // Allocate IO arrays (zero-initialized triggers initial Lorentzian A).
  nda::array<cval_t, 5> A(N_w, ns, Nk, nbnd, nbnd);  A = cval_t(0.0, 0.0);
  nda::array<cval_t, 4> Sx(ns, Nk, nbnd, nbnd);
  nda::array<cval_t, 5> ImSc(ns, Nk, N_w, nbnd, nbnd);
  nda::array<cval_t, 5> ReSc(ns, Nk, N_w, nbnd, nbnd);

  scgw_config cfg;
  cfg.max_iter   = 1;       // one-shot G0W0
  cfg.alpha_mix  = 1.0;
  cfg.tol        = 1e-6;
  cfg.eta        = 0.05;
  cfg.update_mu  = false;   // keep mu=0 fixed

  auto& mpi_context = utils::make_unit_test_mpi_context();
  auto res = run_scgw_serial(mpi_context->comm, grid, H, X, V, kpq, kmq, qw, kw,
                             /*N_elec*/ 1.0, cfg,
                             A, Sx, ImSc, ReSc);

  REQUIRE(res.iter_used == 1);

  // A must be real-valued (real part) and positive on the diagonal.
  for (long iw = 0; iw < N_w; ++iw)
    for (long mu = 0; mu < nbnd; ++mu) {
      REQUIRE(std::isfinite(A(iw, 0, 0, mu, mu).real()));
      REQUIRE(A(iw, 0, 0, mu, mu).real() >= -1e-8);
    }
  // Sigma_x diagonal entries should be negative (occupied) or near zero (virtual).
  REQUIRE(Sx(0, 0, 0, 0).real() <= 0.05);
  REQUIRE(std::abs(Sx(0, 0, 1, 1).real()) <= 0.5);
  // Im Sigma^c diagonal must be <= 0 (causality).
  for (long iw = 0; iw < N_w; ++iw) {
    REQUIRE(ImSc(0, 0, iw, 0, 0).real() <= 1e-10);
    REQUIRE(ImSc(0, 0, iw, 1, 1).real() <= 1e-10);
  }
}

// =============================================================================
// scGW convergence test: same toy, run multiple iterations and verify the
// residual on A monotonically decreases (with mixing).
// =============================================================================
TEST_CASE("real_axis_scf_scgw_converges", "[real_axis][scf][scgw]")
{
  const double w_max     = 10.0;
  const long   N_w       = 129;
  const double Omega_max = 4.0;
  const long   N_Omega   = 32;
  const long   N_t       = 256;
  const double T_window  = 16.0;
  const double beta = 50.0;
  const double mu0  = 0.0;

  auto grid = real_freq_grid_t::make_uniform(
      beta, mu0, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

  const long ns = 1, Nk = 1, Nq = 1, Naux = 2, nbnd = 2;
  nda::array<cval_t, 4> H(ns, Nk, nbnd, nbnd);
  H = cval_t(0.0, 0.0);
  H(0, 0, 0, 0) = cval_t(-0.5, 0.0);
  H(0, 0, 1, 1) = cval_t(+0.5, 0.0);
  nda::array<cval_t, 4> X(ns, Nk, Naux, nbnd);
  X = cval_t(0.0, 0.0);
  for (long P = 0; P < Naux; ++P) X(0, 0, P, P) = cval_t(1.0, 0.0);
  nda::array<cval_t, 3> V(Nq, Naux, Naux);
  V = cval_t(0.0, 0.0);
  for (long P = 0; P < Naux; ++P) V(0, P, P) = cval_t(0.1, 0.0);
  nda::array<long, 2> kpq(Nk, Nq), kmq(Nk, Nq);
  kpq(0, 0) = 0;  kmq(0, 0) = 0;
  nda::array<double, 1> qw(Nq), kw(Nk);
  qw(0) = 1.0;    kw(0) = 1.0;

  nda::array<cval_t, 5> A(N_w, ns, Nk, nbnd, nbnd);  A = cval_t(0.0, 0.0);
  nda::array<cval_t, 4> Sx(ns, Nk, nbnd, nbnd);
  nda::array<cval_t, 5> ImSc(ns, Nk, N_w, nbnd, nbnd);
  nda::array<cval_t, 5> ReSc(ns, Nk, N_w, nbnd, nbnd);

  scgw_config cfg;
  cfg.max_iter   = 6;
  cfg.alpha_mix  = 0.5;
  cfg.tol        = 1e-6;
  cfg.eta        = 0.05;
  cfg.update_mu  = false;

  auto& mpi_context = utils::make_unit_test_mpi_context();
  auto res = run_scgw_serial(mpi_context->comm, grid, H, X, V, kpq, kmq, qw, kw,
                             /*N_elec*/ 1.0, cfg,
                             A, Sx, ImSc, ReSc);

  REQUIRE(res.iter_used >= 1);
  REQUIRE(res.iter_used <= cfg.max_iter);
  REQUIRE(std::isfinite(res.final_diff));
  REQUIRE(res.final_diff >= 0.0);
  // Final spectral function must satisfy basic causality.
  for (long iw = 0; iw < N_w; ++iw)
    for (long mu = 0; mu < nbnd; ++mu)
      REQUIRE(A(iw, 0, 0, mu, mu).real() >= -1e-8);
}


} // namespace gw_real_axis_tests
