/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Regression test for real_axis_scf_loop -- the class-API SCF driver that
 * mirrors methods::scf_loop on the imag-axis side. Drives the periodic
 * qe_lih222 fixture via {real_axis_scr_coulomb_t, real_axis_gw_t,
 * real_axis_dyson_t} and verifies it produces a convergent DIIS trajectory
 * with mu landing in the LiH gap.
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
#include "methods/GW_real_axis/real_axis_mb_state.hpp"
#include "methods/GW_real_axis/real_axis_scr_coulomb_t.h"
#include "methods/GW_real_axis/real_axis_gw_t.h"
#include "methods/GW_real_axis/real_axis_hf_t.h"
#include "methods/GW_real_axis/real_axis_dyson_t.h"
#include "methods/GW_real_axis/real_axis_scf_driver.hpp"

#include <cmath>
#include <complex>

namespace bdft_tests {

  using namespace methods;
  using methods::real_axis::real_freq_grid_t;
  using methods::real_axis::real_axis_mb_state_t;
  using methods::real_axis::real_axis_scr_coulomb_t;
  using methods::real_axis::real_axis_hf_t;
  using methods::real_axis::real_axis_dyson_t;
  using methods::real_axis::real_axis_mb_solver_t;
  using methods::real_axis::real_axis_scf_loop;
  using methods::real_axis::scgw_config;
  using methods::real_axis::scgw_mix_kind;
  using methods::solvers::real_axis_gw_t;
  using cval_t = std::complex<double>;

  // ===========================================================================
  // Periodic SCF on qe_lih222 driven via the new class API. Mirrors the
  // existing `real_axis_scgw_lih222_periodic` test in test_real_axis_scf.cpp
  // (which drives `run_scgw_serial`). Verifies convergence and mu landing.
  // ===========================================================================
  TEST_CASE("real_axis_scf_loop_lih222_periodic",
            "[real_axis][scf_loop][thc][qe][bdft][periodic]")
  {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto mf = std::make_shared<mf::MF>(
                  mf::default_MF(mpi_context, "qe_lih222"));
    const int nIpts = mf->nbnd() * 8;
    thc_reader_t thc(mf, make_thc_reader_ptree(nIpts, "", "incore", "",
                                               "bdft", 1e-8,
                                               mf->ecutrho(), 1, 1024));

    const long ns   = mf->nspin();
    const long Nk   = mf->nkpts();
    const long Nq   = mf->nqpts();
    const long nbnd = mf->nbnd();

    REQUIRE(Nk > 1);
    REQUIRE(Nq == Nk);

    auto eigval = mf->eigval();
    auto kp2ibz = mf->kp_to_ibz();

    double e_min =  std::numeric_limits<double>::infinity();
    double e_max = -std::numeric_limits<double>::infinity();
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < mf->nkpts_ibz(); ++k)
        for (long n = 0; n < nbnd; ++n) {
          const double e = eigval(s, k, n);
          e_min = std::min(e_min, e);
          e_max = std::max(e_max, e);
        }
    if (e_max - e_min < 1e-3) { e_min -= 1.0; e_max += 1.0; }
    const long n_homo = static_cast<long>(mf->nelec() / 2 - 1);
    const long n_lumo = n_homo + 1;
    const double eps_homo = eigval(0, 0, n_homo);
    const double eps_lumo = eigval(0, 0, n_lumo);
    const double mu0      = 0.5 * (eps_homo + eps_lumo);
    const double w_max    = std::max(std::abs(e_min), std::abs(e_max)) + 2.0;

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

    // H_MF = diag(eps_KS).
    nda::array<cval_t, 4> H_MF(ns, Nk, nbnd, nbnd);
    H_MF = cval_t(0.0, 0.0);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k) {
        const long kibz = kp2ibz(k);
        for (long n = 0; n < nbnd; ++n)
          H_MF(s, k, n, n) = cval_t(eigval(s, kibz, n), 0.0);
      }

    // k_weights for mu bisection.
    nda::array<double, 1> k_weights(Nk);
    for (long ik = 0; ik < Nk; ++ik) k_weights(ik) = 1.0 / static_cast<double>(Nk);

    // State: empty A (loop builds Lorentzian initial from H_MF).
    real_axis_mb_state_t state(grid);
    state.mpi = mpi_context;

    // Solver bundle.
    real_axis_hf_t          hf(&grid, "ignore_g0");
    real_axis_scr_coulomb_t scr_eri(&grid, "rpa", "ignore_g0", 1e-8);
    real_axis_gw_t          gw(grid, /*max_iter*/ 1, /*mix*/ 0.5,
                               /*eps_nufft*/ 1e-8, /*ntrans*/ 1);
    real_axis_dyson_t       dyson(std::move(H_MF), &grid, /*eta*/ 0.05);
    real_axis_mb_solver_t   mb_solver{&hf, &scr_eri, &gw};

    // Same DIIS config as the existing periodic test.
    scgw_config cfg;
    cfg.max_iter    = 20;
    cfg.alpha_mix   = 0.7;
    cfg.tol         = 1e-3;
    cfg.eta         = 0.05;
    cfg.eps_nufft   = 1e-8;
    cfg.update_mu   = true;
    cfg.mix_kind    = scgw_mix_kind::diis;
    cfg.diis_window = 8;

    auto res = real_axis_scf_loop(state, dyson, thc,
                                   mb_solver, cfg, k_weights,
                                   /*N_elec*/ static_cast<double>(mf->nelec()));

    app_log(2, "[scf_loop_lih222] iter_used={}  final_diff={:.3e}  final_mu={:.6f}",
            res.iter_used, res.final_diff, res.final_mu);

    REQUIRE(res.iter_used >= 1);
    REQUIRE(res.iter_used <= cfg.max_iter);
    REQUIRE(std::isfinite(res.final_diff));
    REQUIRE(res.final_diff >= 0.0);
    REQUIRE(std::isfinite(res.final_mu));

    // mu should land near the LiH gap.
    REQUIRE(res.final_mu > eps_homo - 0.5);
    REQUIRE(res.final_mu < eps_lumo + 0.5);
    REQUIRE(res.final_diff < 1.0);

    // State outputs were populated.
    REQUIRE(state.A_wskij.has_value());
    REQUIRE(state.Sigma_x_skij.has_value());
    REQUIRE(state.ImSigma_wskij.has_value());
    REQUIRE(state.ReSigma_wskij.has_value());
    REQUIRE(state.ImW_qPQO.has_value());

    // Causality on diag A.
    auto A = state.A_wskij->local();
    long n_total = 0, n_violations = 0;
    for (long iw = 0; iw < N_w; ++iw)
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k)
          for (long n = 0; n < nbnd; ++n) {
            ++n_total;
            if (A(iw, s, k, n, n).real() < -1e-3) ++n_violations;
          }
    REQUIRE(n_violations < n_total / 20);
  }

} // namespace bdft_tests
