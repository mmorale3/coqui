/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Regression test for real_axis_dyson_t. Drives the new class on the
 * qe_lih222 fixture: builds a Lorentzian A, runs scr_coulomb + gw to get
 * Sigma_c, then calls solve_dyson with H_MF = diag(eps_KS) and Sigma_x = 0
 * to update A. Verifies the new A is finite, has the right shape, and that
 * the spectral function trace integrates to nbnd per (s, k).
 * ==========================================================================
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"
#include "mean_field/default_MF.hpp"

#include "methods/ERI/thc_reader_t.hpp"
#include "methods/ERI/eri_utils.hpp"

#include "nda/nda.hpp"
#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_mb_state.hpp"
#include "methods/GW_real_axis/real_axis_scr_coulomb_t.h"
#include "methods/GW_real_axis/real_axis_gw_t.h"
#include "methods/GW_real_axis/real_axis_dyson_t.h"

#include <cmath>
#include <complex>

namespace bdft_tests {

  using namespace methods;
  using methods::real_axis::real_freq_grid_t;
  using methods::real_axis::real_axis_mb_state_t;
  using methods::real_axis::real_axis_scr_coulomb_t;
  using methods::real_axis::real_axis_dyson_t;
  using methods::solvers::real_axis_gw_t;
  using cval_t = std::complex<double>;

  // ===========================================================================
  // End-to-end: scr_coulomb + gw + dyson_t on the qe_lih222 fixture.
  // Verifies solve_dyson writes A_wskij with the right shape, finite entries,
  // and the trace-integral sum-rule sum_w trA(w) dw ~= nbnd per (s, k).
  // ===========================================================================
  TEST_CASE("real_axis_dyson_t_lih222_g0w0",
            "[real_axis][dyson][thc][qe]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(
                  mf::default_MF(mpi_context, "qe_lih222"));
    const int nIpts = mf->nbnd() * 8;
    thc_reader_t thc(mf, make_thc_reader_ptree(nIpts, "", "incore", "",
                                               "bdft", 1e-8,
                                               mf->ecutrho(), 1, 1024));

    const long ns   = mf->nspin();
    const long Nk   = mf->nkpts();
    const long nbnd = mf->nbnd();

    auto eigval = mf->eigval();
    double e_min = std::numeric_limits<double>::infinity();
    double e_max = -std::numeric_limits<double>::infinity();
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < mf->nkpts_ibz(); ++k)
        for (long n = 0; n < nbnd; ++n) {
          e_min = std::min(e_min, double(eigval(s, k, n)));
          e_max = std::max(e_max, double(eigval(s, k, n)));
        }
    if (e_max - e_min < 1e-3) { e_min -= 1.0; e_max += 1.0; }
    const double w_max = std::max(std::abs(e_min), std::abs(e_max)) + 2.0;
    const long n_homo = static_cast<long>(mf->nelec() / 2 - 1);
    const double mu0  = 0.5 * (eigval(0, 0, n_homo) + eigval(0, 0, n_homo + 1));

    const long   N_w       = 65;
    const long   N_Omega   = 32;
    const long   N_t       = 128;
    const double Omega_max = 2.0 * w_max;
    const double freq_max  = std::max(w_max, Omega_max);
    const double dt        = 0.5 * M_PI / freq_max;
    const double T_window  = dt * static_cast<double>(N_t);
    const double beta      = 50.0;

    auto grid = real_freq_grid_t::make_uniform(
                  beta, mu0, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

    real_axis_mb_state_t state(grid);
    state.mpi          = mpi_context;
    state.A_wskij      = nda::array<cval_t, 5>(N_w, ns, Nk, nbnd, nbnd);
    state.Sigma_x_skij = nda::array<cval_t, 4>(ns, Nk, nbnd, nbnd);
    auto& A0 = *state.A_wskij;
    A0 = cval_t(0.0, 0.0);
    *state.Sigma_x_skij = cval_t(0.0, 0.0);

    const double eta_lor = 0.05;
    auto kp2ibz = mf->kp_to_ibz();
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k) {
        const long kibz = kp2ibz(k);
        for (long n = 0; n < nbnd; ++n) {
          const double eps_n = eigval(s, kibz, n);
          for (long iw = 0; iw < N_w; ++iw) {
            const double w_l = grid.w()(iw) + grid.mu_chem();
            const double v = (1.0 / M_PI) * eta_lor
                           / ((w_l - eps_n)*(w_l - eps_n) + eta_lor*eta_lor);
            A0(iw, s, k, n, n) = cval_t(v, 0.0);
          }
        }
      }

    // Build Sigma_c via scr_coulomb + gw.
    real_axis_scr_coulomb_t scr_eri(&grid, "rpa", "ignore_g0", 1e-8);
    real_axis_gw_t          gw(grid, /*max_iter*/ 1, /*mix*/ 0.5,
                               /*eps_nufft*/ 1e-8, /*ntrans*/ 1);
    scr_eri.update_w(state, thc,
                     /*verbose*/ false, /*use_rspace*/ false);
    gw.evaluate(state, thc,
                /*eps_nufft*/ 1e-8, "ignore_g0",
                /*verbose*/ false, /*use_rspace*/ false);

    REQUIRE(state.ImSigma_wskij.has_value());
    REQUIRE(state.ReSigma_wskij.has_value());

    // Build H_MF = diag(eps_KS) (KS basis is diagonal in itself).
    nda::array<cval_t, 4> H_MF_skij(ns, Nk, nbnd, nbnd);
    H_MF_skij = cval_t(0.0, 0.0);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k) {
        const long kibz = kp2ibz(k);
        for (long n = 0; n < nbnd; ++n)
          H_MF_skij(s, k, n, n) = cval_t(eigval(s, kibz, n), 0.0);
      }

    real_axis_dyson_t dyson(std::move(H_MF_skij), &grid, /*eta*/ 0.05);
    dyson.solve_dyson(state, mu0);

    // A should be allocated, right shape, finite.
    REQUIRE(state.A_wskij.has_value());
    auto const& A = *state.A_wskij;
    REQUIRE(A.shape()[0] == N_w);
    REQUIRE(A.shape()[1] == ns);
    REQUIRE(A.shape()[2] == Nk);
    REQUIRE(A.shape()[3] == nbnd);
    REQUIRE(A.shape()[4] == nbnd);
    REQUIRE(state.mu_chem == mu0);

    bool finite = true;
    for (long i = 0; i < A.size(); ++i)
      if (!std::isfinite(A.data()[i].real())) finite = false;
    REQUIRE(finite);

    // Spectral sum rule: sum_iw trA(s, k, w) * dw ~= nbnd for each (s, k).
    // The Hilbert window (-w_max, w_max) is wide enough to capture all
    // KS eigenvalues + eta tails; tolerance 5% is generous given finite
    // grid + Sigma broadening.
    auto const& dw = grid.w_weights();
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k) {
        double sum_tr = 0.0;
        for (long iw = 0; iw < N_w; ++iw) {
          double tr = 0.0;
          for (long n = 0; n < nbnd; ++n)
            tr += A(iw, s, k, n, n).real();
          sum_tr += dw(iw) * tr;
        }
        // Allow 5% error from the eta=0.05 broadening leaking outside w_max
        // and from the small Sigma_c shift.
        REQUIRE(sum_tr == Approx(double(nbnd)).margin(0.05 * nbnd));
      }
  }

  // ===========================================================================
  // mu update: with H_MF = diag(eps_KS), Sigma_c = 0, and Sigma_x = 0, the
  // bisection should reproduce mu0 within tolerance.
  // ===========================================================================
  TEST_CASE("real_axis_dyson_t_lih222_find_mu_chem_g0",
            "[real_axis][dyson][thc][mu]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    (void) mpi_context;

    auto mf = std::make_shared<mf::MF>(
                  mf::default_MF(mpi_context, "qe_lih222"));

    const long ns   = mf->nspin();
    const long Nk   = mf->nkpts();
    const long nbnd = mf->nbnd();

    auto eigval = mf->eigval();
    double e_min = std::numeric_limits<double>::infinity();
    double e_max = -std::numeric_limits<double>::infinity();
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < mf->nkpts_ibz(); ++k)
        for (long n = 0; n < nbnd; ++n) {
          e_min = std::min(e_min, double(eigval(s, k, n)));
          e_max = std::max(e_max, double(eigval(s, k, n)));
        }
    const double w_max = std::max(std::abs(e_min), std::abs(e_max)) + 2.0;
    const long n_homo  = static_cast<long>(mf->nelec() / 2 - 1);
    const double mu0   = 0.5 * (eigval(0, 0, n_homo) + eigval(0, 0, n_homo + 1));

    const long   N_w       = 257;        // dense grid for accurate quadrature
    const long   N_Omega   = 32;
    const long   N_t       = 128;
    const double Omega_max = 2.0 * w_max;
    const double freq_max  = std::max(w_max, Omega_max);
    const double dt        = 0.5 * M_PI / freq_max;
    const double T_window  = dt * static_cast<double>(N_t);
    const double beta      = 200.0;       // sharper Fermi for tighter mu test

    auto grid = real_freq_grid_t::make_uniform(
                  beta, mu0, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

    real_axis_mb_state_t state(grid);
    state.Sigma_x_skij = nda::array<cval_t, 4>(ns, Nk, nbnd, nbnd);
    state.ImSigma_wskij = nda::array<cval_t, 5>(N_w, ns, Nk, nbnd, nbnd);
    state.ReSigma_wskij = nda::array<cval_t, 5>(N_w, ns, Nk, nbnd, nbnd);
    *state.Sigma_x_skij  = cval_t(0.0, 0.0);
    *state.ImSigma_wskij = cval_t(0.0, 0.0);
    *state.ReSigma_wskij = cval_t(0.0, 0.0);

    nda::array<cval_t, 4> H_MF_skij(ns, Nk, nbnd, nbnd);
    H_MF_skij = cval_t(0.0, 0.0);
    auto kp2ibz = mf->kp_to_ibz();
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k) {
        const long kibz = kp2ibz(k);
        for (long n = 0; n < nbnd; ++n)
          H_MF_skij(s, k, n, n) = cval_t(eigval(s, kibz, n), 0.0);
      }

    real_axis_dyson_t dyson(std::move(H_MF_skij), &grid, /*eta*/ 0.01);
    dyson.solve_dyson(state, mu0);

    nda::array<double, 1> k_weights(Nk);
    for (long k = 0; k < Nk; ++k)
      k_weights(k) = 1.0 / static_cast<double>(Nk);
    const double N_elec = static_cast<double>(mf->nelec());

    const double mu_solved = dyson.find_mu_chem(state, k_weights, N_elec);
    app_log(2, "[dyson_t mu_solved] mu0={:.6f}, mu_solved={:.6f}, gap=[{:.4f}, {:.4f}]",
            mu0, mu_solved,
            double(eigval(0, 0, n_homo)), double(eigval(0, 0, n_homo + 1)));

    // mu_solved should land inside the gap.
    const double homo = eigval(0, 0, n_homo);
    const double lumo = eigval(0, 0, n_homo + 1);
    REQUIRE(mu_solved > homo);
    REQUIRE(mu_solved < lumo);
  }

} // namespace bdft_tests
