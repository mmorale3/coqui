/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Smoke test for the symmetry-adapted (isym) path in real_axis_gw_t::evaluate
 * and real_axis_scr_coulomb_t::update_w.
 *
 * Uses the `qe_lih222_sym` fixture: same physical system as `qe_lih222`
 * (the trivial-IBZ fixture used by all other real-axis tests) but with
 * the QE space-group symmetry exposed → nkpts_ibz < nkpts and qsymms.size()
 * > 1. This is the smallest available fixture that exercises the
 * symmetry-adapted ISDF code path.
 *
 * Acceptance:
 *   - state arrays come out at IBZ shape (Nk_ibz, Nq_ibz).
 *   - update_w + gw_t::evaluate run without assertion failures, producing
 *     finite Pi, W, Sigma_c.
 *   - The diagonal Sigma values are sane (non-zero, finite).
 *
 * The full bit-comparison against the imag-axis Sigma at IBZ k is left to
 * a follow-up xvalidate test once the Sigma_x isym wrap is also in place
 * (this test runs only update_w + gw_t::evaluate, no HF).
 * ==========================================================================
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

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
#include "methods/GW_real_axis/real_axis_hf_t.h"
#include "methods/GW_real_axis/real_axis_dyson_t.h"
#include "methods/GW_real_axis/real_axis_scf.hpp"
#include "methods/GW_real_axis/real_axis_scf_driver.hpp"
#include "methods/GW_real_axis/real_axis_gw_thc.hpp"

// Matsubara branch dependencies for the cross-validation test.
#include "methods/mb_state/mb_state.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include "numerics/iter_scf/iter_scf_t.hpp"
#include "methods/SCF/mb_solver_t.h"
#include "methods/HF/hf_t.h"
#include "methods/GW/gw_t.h"
#include "methods/scr_coulomb/scr_coulomb_t.h"
#include "methods/ERI/mb_eri_context.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include <cstdio>

#include <cmath>
#include <complex>
#include <limits>

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

  TEST_CASE("real_axis_symmetry_lih222_sym",
            "[real_axis][thc][gw][qe][bdft][serial][symmetry]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    // Load the symmetry-aware LiH 2x2x2 fixture. Same physical system as
    // qe_lih222 but with non-trivial qsymms / nkpts_ibz < nkpts.
    auto mf = std::make_shared<mf::MF>(
                  mf::default_MF(mpi_context, "qe_lih222_sym"));
    const int nIpts = mf->nbnd() * 8;
    thc_reader_t thc(mf, make_thc_reader_ptree(nIpts, "", "incore", "",
                                               "bdft", 1e-8,
                                               mf->ecutrho(), 1, 1024));

    const long ns      = mf->nspin();
    const long Nk      = mf->nkpts();
    const long Nq      = mf->nqpts();
    const long Nk_ibz  = mf->nkpts_ibz();
    const long Nq_ibz  = mf->nqpts_ibz();
    const long nbnd    = mf->nbnd();
    const long Naux    = thc.Np();

    if (mpi_context->comm.root()) {
      app_log(2, "[symmetry_lih222_sym] Nk={}, Nk_ibz={}, Nq={}, Nq_ibz={}, "
                  "qsymms.size={}, ns={}, nbnd={}, Naux={}",
              Nk, Nk_ibz, Nq, Nq_ibz, mf->qsymms().shape()[0],
              ns, nbnd, Naux);
    }
    // Sanity: confirm the fixture actually has non-trivial symmetry. If
    // this fails the fixture is the wrong target — pick another.
    REQUIRE(Nk_ibz < Nk);
    REQUIRE(Nq_ibz < Nq);
    REQUIRE(mf->qsymms().shape()[0] > 1);

    // ---- Real-axis grid ----
    auto eigval = mf->eigval();
    double e_min =  std::numeric_limits<double>::infinity();
    double e_max = -std::numeric_limits<double>::infinity();
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk_ibz; ++k)
        for (long n = 0; n < nbnd; ++n) {
          e_min = std::min(e_min, eigval(s, k, n));
          e_max = std::max(e_max, eigval(s, k, n));
        }
    const double w_max     = std::max(std::abs(e_min), std::abs(e_max)) + 2.0;
    const long n_homo = static_cast<long>(mf->nelec() / 2 - 1);
    const long n_lumo = n_homo + 1;
    const double mu0       = 0.5 * (eigval(0, 0, n_homo) + eigval(0, 0, n_lumo));

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

    // ---- Build initial A at IBZ k. Diagonal Lorentzians per
    // (s, k_ibz, n) eigenvalue. ----
    real_axis_mb_state_t state(grid);
    state.mpi = mpi_context;
    state.A_wskij.emplace(*state.mpi,
        std::array<long, 5>{N_w, ns, Nk_ibz, nbnd, nbnd});
    if (state.A_wskij->node_comm()->root()) {
      auto A = state.A_wskij->local();
      A = cval_t(0.0, 0.0);
      const double eta = 0.05;
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk_ibz; ++k)
          for (long n = 0; n < nbnd; ++n) {
            const double e = eigval(s, k, n);
            for (long iw = 0; iw < N_w; ++iw) {
              const double wl = grid.w()(iw) + grid.mu_chem();
              const double v = (1.0 / M_PI) * eta
                              / ((wl - e)*(wl - e) + eta*eta);
              A(iw, s, k, n, n) = cval_t(v, 0.0);
            }
          }
    }
    state.A_wskij->node_sync();

    // ---- Run update_w (Pi/W build via symmetry-adapted projection) ----
    real_axis_scr_coulomb_t scr(&grid, "rpa", "ignore_g0", 1e-8);
    scr.update_w(state, thc, /*verbose*/ false, /*use_rspace*/ true);

    // Sanity: bosonic state lives at IBZ q.
    REQUIRE(state.ImPi_qPQO.has_value());
    REQUIRE(state.ImW_qPQO.has_value());
    REQUIRE(state.ImW_qPQO->global_shape()[0] == Nq_ibz);
    REQUIRE(state.ImW_qPQO->global_shape()[1] == Naux);
    REQUIRE(state.ImW_qPQO->global_shape()[3] == grid.N_Omega());

    // Check ImW is finite and non-zero somewhere.
    {
      auto ImW = state.ImW_qPQO->local();
      double sum_abs = 0.0;
      bool all_finite = true;
      for (long i = 0; i < ImW.size(); ++i) {
        const auto v = ImW.data()[i];
        sum_abs += std::abs(v);
        if (!std::isfinite(v.real()) or !std::isfinite(v.imag()))
          all_finite = false;
      }
      REQUIRE(all_finite);
      REQUIRE(sum_abs > 0.0);
    }

    // ---- Run gw_t::evaluate (Sigma build via isym loop) ----
    real_axis_gw_t gw(grid, /*max_iter*/ 1, /*mix*/ 0.5,
                      /*eps_nufft*/ 1e-8, /*ntrans*/ 1);
    gw.evaluate(state, thc, /*eps_nufft*/ 1e-8, /*div_treatment*/ "ignore_g0",
                /*verbose*/ false, /*use_rspace*/ false);

    // Sanity: Sigma lives at IBZ k.
    REQUIRE(state.ImSigma_wskij.has_value());
    REQUIRE(state.ReSigma_wskij.has_value());
    REQUIRE(state.ImSigma_wskij->shape()[0] == N_w);
    REQUIRE(state.ImSigma_wskij->shape()[1] == ns);
    REQUIRE(state.ImSigma_wskij->shape()[2] == Nk_ibz);
    REQUIRE(state.ImSigma_wskij->shape()[3] == nbnd);
    REQUIRE(state.ImSigma_wskij->shape()[4] == nbnd);

    // Check Sigma is finite, non-zero, and not crazy.
    {
      auto ImS = state.ImSigma_wskij->local();
      auto ReS = state.ReSigma_wskij->local();
      double max_abs_im = 0.0, max_abs_re = 0.0;
      double sum_abs = 0.0;
      bool all_finite = true;
      for (long i = 0; i < ImS.size(); ++i) {
        const auto vi = ImS.data()[i];
        const auto vr = ReS.data()[i];
        max_abs_im = std::max(max_abs_im, std::abs(vi));
        max_abs_re = std::max(max_abs_re, std::abs(vr));
        sum_abs += std::abs(vi) + std::abs(vr);
        if (!std::isfinite(vi.real()) or !std::isfinite(vr.real()))
          all_finite = false;
      }
      REQUIRE(all_finite);
      REQUIRE(sum_abs > 0.0);
      // Diagonal Im Sigma should be O(0.01-1) Hartree at most for LiH;
      // a value > 1e3 would indicate a bug (e.g. orbital rotation
      // applied incorrectly).
      REQUIRE(max_abs_im < 1.0e3);
      REQUIRE(max_abs_re < 1.0e3);

      if (mpi_context->comm.root()) {
        app_log(2, "[symmetry_lih222_sym] max|Im Sigma| = {:.3e}, "
                    "max|Re Sigma| = {:.3e}", max_abs_im, max_abs_re);
      }
    }

    // ---- Run hf::evaluate (Sigma_x via isym path) ----
    real_axis_hf_t hf(&grid, "ignore_g0");
    hf.evaluate(state, thc, mu0);

    // Sanity: Sigma_x lives at IBZ k.
    REQUIRE(state.Sigma_x_skij.has_value());
    REQUIRE(state.Sigma_x_skij->shape()[0] == ns);
    REQUIRE(state.Sigma_x_skij->shape()[1] == Nk_ibz);
    REQUIRE(state.Sigma_x_skij->shape()[2] == nbnd);
    REQUIRE(state.Sigma_x_skij->shape()[3] == nbnd);

    // Check Sigma_x is finite, non-zero, and not crazy.
    {
      auto Sx = state.Sigma_x_skij->local();
      double max_abs = 0.0, sum_abs = 0.0;
      bool all_finite = true;
      for (long i = 0; i < Sx.size(); ++i) {
        const auto v = Sx.data()[i];
        max_abs = std::max(max_abs, std::abs(v));
        sum_abs += std::abs(v);
        if (!std::isfinite(v.real()) or !std::isfinite(v.imag()))
          all_finite = false;
      }
      REQUIRE(all_finite);
      REQUIRE(sum_abs > 0.0);
      REQUIRE(max_abs < 1.0e3);
      if (mpi_context->comm.root())
        app_log(2, "[symmetry_lih222_sym] max|Sigma_x| = {:.3e}", max_abs);
    }
  }

  TEST_CASE("real_axis_scgw_lih222_sym",
            "[real_axis][thc][gw][qe][bdft][serial][symmetry][scf]") {
    // End-to-end scGW SCF on the symmetry-aware LiH 2x2x2 fixture. Drives
    // update_w + gw_t::evaluate + hf_t::evaluate + dyson through the
    // real_axis_scf_loop. Acceptance: SCF runs through the configured
    // iterations and produces finite, sane state arrays at IBZ k.
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto mf = std::make_shared<mf::MF>(
                  mf::default_MF(mpi_context, "qe_lih222_sym"));
    const int nIpts = mf->nbnd() * 8;
    thc_reader_t thc(mf, make_thc_reader_ptree(nIpts, "", "incore", "",
                                               "bdft", 1e-8,
                                               mf->ecutrho(), 1, 1024));

    const long ns      = mf->nspin();
    const long Nk_ibz  = mf->nkpts_ibz();
    const long nbnd    = mf->nbnd();
    REQUIRE(Nk_ibz < mf->nkpts());

    // Real-frequency grid.
    auto eigval = mf->eigval();
    double e_min =  std::numeric_limits<double>::infinity();
    double e_max = -std::numeric_limits<double>::infinity();
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk_ibz; ++k)
        for (long n = 0; n < nbnd; ++n) {
          e_min = std::min(e_min, eigval(s, k, n));
          e_max = std::max(e_max, eigval(s, k, n));
        }
    const double w_max  = std::max(std::abs(e_min), std::abs(e_max)) + 2.0;
    const long n_homo = static_cast<long>(mf->nelec() / 2 - 1);
    const long n_lumo = n_homo + 1;
    const double mu0    = 0.5 * (eigval(0, 0, n_homo) + eigval(0, 0, n_lumo));
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

    // H_MF = diag(eps_KS) at IBZ k. k_weights at IBZ k.
    nda::array<ComplexType, 4> H_MF(ns, Nk_ibz, nbnd, nbnd);
    H_MF = ComplexType(0.0, 0.0);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk_ibz; ++k)
        for (long n = 0; n < nbnd; ++n)
          H_MF(s, k, n, n) = ComplexType(eigval(s, k, n), 0.0);

    nda::array<double, 1> k_weights(Nk_ibz);
    auto kw = mf->k_weight();
    for (long k = 0; k < Nk_ibz; ++k) k_weights(k) = kw(k);

    // State (no pre-allocated A — driver builds Lorentzian initial).
    real_axis::real_axis_mb_state_t state(grid);
    state.mpi = mpi_context;

    // Solver bundle.
    real_axis_hf_t          hf(&grid, "ignore_g0");
    real_axis_scr_coulomb_t scr(&grid, "rpa", "ignore_g0", 1e-8);
    real_axis_gw_t          gw(grid, /*max_iter*/ 1, /*mix*/ 0.5,
                                /*eps_nufft*/ 1e-8, /*ntrans*/ 1);
    real_axis_dyson_t       dyson(std::move(H_MF), &grid,
                                  /*eta*/ 0.05, /*mu_tol*/ 1e-6);
    real_axis_mb_solver_t   mb_solver(&hf, &scr, &gw);

    scgw_config cfg;
    cfg.max_iter    = 3;
    cfg.alpha_mix   = 0.3;
    cfg.tol         = 1e-3;
    cfg.eta         = 0.05;
    cfg.eps_nufft   = 1e-8;
    cfg.update_mu   = false;
    cfg.verbose     = false;
    cfg.mix_kind    = scgw_mix_kind::linear;
    cfg.diis_window = 4;

    auto res = real_axis_scf_loop(state, dyson, thc, mb_solver, cfg,
                                  k_weights,
                                  static_cast<double>(mf->nelec()));
    (void)res;

    // Sanity: state arrays at IBZ shape after the SCF loop.
    REQUIRE(state.A_wskij.has_value());
    REQUIRE(state.A_wskij->shape()[2] == Nk_ibz);
    REQUIRE(state.ImSigma_wskij.has_value());
    REQUIRE(state.ImSigma_wskij->shape()[2] == Nk_ibz);
    REQUIRE(state.Sigma_x_skij.has_value());
    REQUIRE(state.Sigma_x_skij->shape()[1] == Nk_ibz);

    if (mpi_context->comm.root()) {
      app_log(2, "[scgw_lih222_sym] iter_used={}, ||dA||={:.3e}, mu={:.6f}, "
                  "converged={}",
              res.iter_used, res.final_diff, res.final_mu,
              res.converged ? "yes" : "no");
    }

    // Sanity: final dA is finite.
    REQUIRE(std::isfinite(res.final_diff));
    REQUIRE(res.iter_used > 0);
  }

  TEST_CASE("real_axis_vs_matsubara_lih222_sym_diagnostics",
            "[real_axis][thc][gw][qe][bdft][serial][xvalidate][symmetry]") {
    // Symmetry-aware twin of test_real_axis_xvalidate.cpp's main case.
    // Runs the real-axis G0W0 chain (update_w + gw_t::evaluate) on the
    // symmetry-aware qe_lih222_sym fixture, and compares the diagonal
    // ImΣ-derived Sigma_c(iw_n) against the imag-axis G0W0 reference at
    // the same IBZ k = Γ. Acceptance: max |Σ_real_to_im - Σ_mat| over the
    // lowest 8 Matsubara points stays under 5e-2 — same tolerance as the
    // qe_lih222 xvalidate baseline.
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(
                  mf::default_MF(mpi_context, "qe_lih222_sym"));
    const int nIpts = mf->nbnd() * 8;
    methods::thc_reader_t thc(mf,
        methods::make_thc_reader_ptree(nIpts, "", "incore", "",
                                        "bdft", 1e-10,
                                        mf->ecutrho(), 1, 1024));

    const long ns      = mf->nspin();
    const long Nk_ibz  = mf->nkpts_ibz();
    const long nbnd    = mf->nbnd();

    REQUIRE(Nk_ibz < mf->nkpts());

    auto eigval = mf->eigval();
    const long n_homo = static_cast<long>(mf->nelec() / 2 - 1);
    const long n_lumo = n_homo + 1;
    const double eps_homo = eigval(0, 0, n_homo);
    const double eps_lumo = eigval(0, 0, n_lumo);

    // Real-frequency grid (same shape as the qe_lih222 xvalidate test).
    double e_min =  std::numeric_limits<double>::infinity();
    double e_max = -std::numeric_limits<double>::infinity();
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk_ibz; ++k)
        for (long n = 0; n < nbnd; ++n) {
          e_min = std::min(e_min, eigval(s, k, n));
          e_max = std::max(e_max, eigval(s, k, n));
        }
    if (e_max - e_min < 1e-3) { e_min -= 1.0; e_max += 1.0; }
    const double w_max     = std::max(std::abs(e_min), std::abs(e_max)) + 2.0;
    const double mu0       = 0.5 * (eps_homo + eps_lumo);
    const long   N_w       = 129;
    const long   N_Omega   = 64;
    const long   N_t       = 256;
    const double Omega_max = 2.0 * w_max;
    const double freq_max  = std::max(w_max, Omega_max);
    const double dt        = 0.5 * M_PI / freq_max;
    const double T_window  = dt * static_cast<double>(N_t);
    const double beta      = 200.0;
    auto grid = real_freq_grid_t::make_uniform(
                  beta, mu0, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

    // Initial A: diagonal Lorentzians per (s, k_ibz, n) eigenvalue.
    real_axis_mb_state_t state(grid);
    state.mpi = mpi_context;
    state.A_wskij.emplace(*state.mpi,
        std::array<long, 5>{N_w, ns, Nk_ibz, nbnd, nbnd});
    if (state.A_wskij->node_comm()->root()) {
      auto A = state.A_wskij->local();
      A = cval_t(0.0, 0.0);
      const double eta = 0.05;
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk_ibz; ++k)
          for (long n = 0; n < nbnd; ++n) {
            const double e = eigval(s, k, n);
            for (long iw = 0; iw < N_w; ++iw) {
              const double w_l = grid.w()(iw) + grid.mu_chem();
              const double v = (1.0 / M_PI) * eta
                              / ((w_l - e)*(w_l - e) + eta*eta);
              A(iw, s, k, n, n) = cval_t(v, 0.0);
            }
          }
    }
    state.A_wskij->node_sync();

    // Run real-axis G0W0 (update_w + gw_t::evaluate via thc shim).
    // use_rspace=true is required for the symmetry-aware update_w (R-space
    // path); gw_t::evaluate's isym branch is k-space-only and ignores the
    // use_rspace flag.
    methods::real_axis::evaluate_thc_serial(
        state, thc, /*eps_nufft*/ 1e-10, "ignore_g0",
        /*verbose*/ false, /*use_rspace*/ true);

    REQUIRE(state.ImSigma_wskij.has_value());
    REQUIRE(state.ReSigma_wskij.has_value());
    auto ImS = state.ImSigma_wskij->local();
    auto A_view = state.A_wskij->local();
    REQUIRE(ImS.shape()[2] == Nk_ibz);

    // ---- Imag-axis G0W0 on the same fixture ----
    imag_axes_ft::IAFT ft_im(1000, 1.2, imag_axes_ft::ir_source);
    const std::string output_prefix = "coqui_xvalidate_sym";

    methods::solvers::hf_t            hf_im;
    methods::solvers::gw_t            gw_im(&ft_im, "ignore_g0", output_prefix);
    methods::solvers::scr_coulomb_t   scr_im(&ft_im, "rpa", "ignore_g0");
    methods::simple_dyson             dyson_im(mf.get(), &ft_im);
    auto                              eri = methods::mb_eri_t(thc, thc);
    iter_scf::iter_scf_t              iter_sol_im("damping");
    methods::MBState                  mb_state_im(mpi_context, ft_im, output_prefix);

    methods::scf_loop(mb_state_im, dyson_im, eri, ft_im,
                      methods::solvers::mb_solver_t(&hf_im, &gw_im, &scr_im),
                      &iter_sol_im, /*niter*/ 1, /*restart*/ false,
                      /*conv_tol*/ 1e-9, /*const_mu*/ true);

    REQUIRE(mb_state_im.sSigma_tskij.has_value());
    auto const& sS_im = mb_state_im.sSigma_tskij.value();
    auto Sigma_tskij_im = sS_im.local();   // (nt, ns, nkpts_ibz, nbnd, nbnd)
    const long nw_im = ft_im.nw_f();
    REQUIRE(Sigma_tskij_im.shape()[2] == Nk_ibz);

    // tau -> i omega.
    nda::array<cval_t, 5> Sigma_wskij_im(nw_im, ns, Nk_ibz, nbnd, nbnd);
    ft_im.tau_to_w(Sigma_tskij_im, Sigma_wskij_im, imag_axes_ft::fermi);

    auto wn = ft_im.wn_mesh();
    nda::array<cval_t, 1> iw_mesh(nw_im);
    for (long n = 0; n < nw_im; ++n) iw_mesh(n) = ft_im.omega(wn(n));

    // Forward-transform real-axis Im Σ^R(ω) onto the Matsubara mesh via
    // Σ_c(z) = -(1/π) ∫ dω' Im Σ^R(ω') / (z - ω').
    auto realaxis_to_matsubara_diag = [&](long n_band) {
      nda::array<cval_t, 1> result(nw_im);
      auto const& w_real    = grid.w();
      auto const& w_real_wq = grid.w_weights();
      const long Nw_real    = grid.N_w();
      for (long iw = 0; iw < nw_im; ++iw) {
        const cval_t z = iw_mesh(iw);
        cval_t acc(0.0, 0.0);
        for (long j = 0; j < Nw_real; ++j) {
          const double imS_v = ImS(j, 0, 0, n_band, n_band).real();
          const cval_t denom = z - cval_t(w_real(j), 0.0);
          acc += cval_t(w_real_wq(j) * imS_v, 0.0) / denom;
        }
        result(iw) = -acc / M_PI;
      }
      return result;
    };

    auto Sigma_r2i_homo = realaxis_to_matsubara_diag(n_homo);
    auto Sigma_r2i_lumo = realaxis_to_matsubara_diag(n_lumo);

    const long n_check = std::min<long>(8, nw_im);
    double max_diff_homo = 0.0, max_diff_lumo = 0.0;
    for (long iw = 0; iw < n_check; ++iw) {
      const cval_t mat_h = Sigma_wskij_im(iw, 0, 0, n_homo, n_homo);
      const cval_t mat_l = Sigma_wskij_im(iw, 0, 0, n_lumo, n_lumo);
      max_diff_homo = std::max(max_diff_homo, std::abs(Sigma_r2i_homo(iw) - mat_h));
      max_diff_lumo = std::max(max_diff_lumo, std::abs(Sigma_r2i_lumo(iw) - mat_l));
    }
    REQUIRE(std::isfinite(max_diff_homo));
    REQUIRE(std::isfinite(max_diff_lumo));

    if (mpi_context->comm.root()) {
      app_log(2, "[xvalidate_sym] LiH222_sym alpha=8 (real-axis), "
                  "Matsubara IR-Lambda=1000, k=Γ s=0");
      app_log(2, "[xvalidate_sym] Σ_c(iw_n) HOMO at the lowest {} Matsubara "
                  "points (s=0, k=Γ):", n_check);
      app_log(2, "[xvalidate_sym]   iw_n            mat (re,im)             "
                  "real->im (re,im)         |diff|");
      for (long iw = 0; iw < n_check; ++iw) {
        const cval_t mat = Sigma_wskij_im(iw, 0, 0, n_homo, n_homo);
        const cval_t r2i = Sigma_r2i_homo(iw);
        app_log(2, "[xvalidate_sym]   {0:+10.4f}   ({1:+10.5f},{2:+10.5f})  "
                    "({3:+10.5f},{4:+10.5f})   {5:+10.3e}",
                iw_mesh(iw).imag(), mat.real(), mat.imag(),
                r2i.real(), r2i.imag(), std::abs(r2i - mat));
      }
      app_log(2, "[xvalidate_sym] max |Σ_real_to_im - Σ_mat| over lowest "
                  "{0} iw_n: HOMO={1:+10.3e}  LUMO={2:+10.3e}",
              n_check, max_diff_homo, max_diff_lumo);
    }

    const double tol_xvalid = 5e-2;
    REQUIRE(max_diff_homo < tol_xvalid);
    REQUIRE(max_diff_lumo < tol_xvalid);

    if (mpi_context->comm.root()) {
      std::remove((output_prefix + ".mbpt.h5").c_str());
    }
    mpi_context->comm.barrier();
  }

  TEST_CASE("real_axis_inspect_trev_lih223_inv_only",
            "[real_axis][qe][bdft][trev][diagnostic]") {
    // Diagnostic: report kp_trev / qp_trev statistics for the inv-broken
    // LiH 2x2x3 fixture, used to scope the TR-pair handling work.
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto mf = mf::default_MF(mpi_context, "qe_lih223_inv");
    auto kp_trev = mf.kp_trev();
    auto qp_trev = mf.qp_trev();
    long n_kp_trev = 0, n_qp_trev = 0;
    for (long k = 0; k < kp_trev.shape()[0]; ++k)
      if (kp_trev(k) != 0) ++n_kp_trev;
    for (long q = 0; q < qp_trev.shape()[0]; ++q)
      if (qp_trev(q) != 0) ++n_qp_trev;
    if (mpi_context->comm.root()) {
      app_log(2, "[trev_inspect] qe_lih223_inv_only Nk={}, Nk_ibz={}, "
                  "Nq={}, Nq_ibz={}, qsymms={}",
              mf.nkpts(), mf.nkpts_ibz(), mf.nqpts(), mf.nqpts_ibz(),
              mf.qsymms().shape()[0]);
      app_log(2, "[trev_inspect]   #k with kp_trev != 0 = {} / {}",
              n_kp_trev, kp_trev.shape()[0]);
      app_log(2, "[trev_inspect]   #q with qp_trev != 0 = {} / {}",
              n_qp_trev, qp_trev.shape()[0]);
    }
    // Sanity that the fixture loaded.
    REQUIRE(mf.nkpts() > 0);
  }

  TEST_CASE("real_axis_inspect_si211",
            "[real_axis][qe][bdft][diagnostic][si]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto mf = mf::default_MF(mpi_context, "qe_si211");
    auto kp_trev = mf.kp_trev();
    auto qp_trev = mf.qp_trev();
    long n_kp_trev = 0, n_qp_trev = 0;
    for (long k = 0; k < kp_trev.shape()[0]; ++k)
      if (kp_trev(k) != 0) ++n_kp_trev;
    for (long q = 0; q < qp_trev.shape()[0]; ++q)
      if (qp_trev(q) != 0) ++n_qp_trev;
    if (mpi_context->comm.root()) {
      app_log(2, "[si211_inspect] qe_si211 Nk={}, Nk_ibz={}, "
                  "Nq={}, Nq_ibz={}, qsymms={}, npol={}, ns={}",
              mf.nkpts(), mf.nkpts_ibz(), mf.nqpts(), mf.nqpts_ibz(),
              mf.qsymms().shape()[0], mf.npol(), mf.nspin());
      app_log(2, "[si211_inspect]   #k with kp_trev != 0 = {} / {}",
              n_kp_trev, kp_trev.shape()[0]);
      app_log(2, "[si211_inspect]   #q with qp_trev != 0 = {} / {}",
              n_qp_trev, qp_trev.shape()[0]);
    }
    REQUIRE(mf.nkpts() > 0);
  }

  TEST_CASE("real_axis_update_w_lih223_inv_partial_trev",
            "[real_axis][thc][gw][qe][bdft][serial][trev]") {
    // Partial TR-pair validation: run only update_w (Pi/W build) on the
    // inv-broken LiH 2x2x3 fixture, which has 4/12 kp_trev and 4/12 qp_trev
    // entries set. update_w uses the R-space path with the kp_trev
    // conj-copy fix-up of aux-A at TR-pair k's; the output Pi/W lands at
    // IBZ q via the R→q FT, so qp_trev does NOT enter the kernel here.
    //
    // The full Sigma chain (gw_t::evaluate isym kernel) requires qp_trev
    // handling (currently asserted absent) and is deferred. This test
    // confirms the kp_trev half of the TR support produces finite Pi/W
    // on a non-inversion fixture.
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto mf = std::make_shared<mf::MF>(
                  mf::default_MF(mpi_context, "qe_lih223_inv"));
    const int nIpts = mf->nbnd() * 8;
    methods::thc_reader_t thc(mf,
        methods::make_thc_reader_ptree(nIpts, "", "incore", "",
                                        "bdft", 1e-8,
                                        mf->ecutrho(), 1, 1024));

    const long ns      = mf->nspin();
    const long Nk_ibz  = mf->nkpts_ibz();
    const long Nq_ibz  = mf->nqpts_ibz();
    const long nbnd    = mf->nbnd();
    const long Naux    = thc.Np();

    // Sanity that this fixture exercises kp_trev.
    long n_kp_trev = 0;
    auto kp_trev = mf->kp_trev();
    for (long k = 0; k < kp_trev.shape()[0]; ++k)
      if (kp_trev(k) != 0) ++n_kp_trev;
    REQUIRE(n_kp_trev > 0);

    auto eigval = mf->eigval();
    double e_min =  std::numeric_limits<double>::infinity();
    double e_max = -std::numeric_limits<double>::infinity();
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk_ibz; ++k)
        for (long n = 0; n < nbnd; ++n) {
          e_min = std::min(e_min, eigval(s, k, n));
          e_max = std::max(e_max, eigval(s, k, n));
        }
    const double w_max = std::max(std::abs(e_min), std::abs(e_max)) + 2.0;
    const long n_homo = static_cast<long>(mf->nelec() / 2 - 1);
    const long n_lumo = n_homo + 1;
    const double mu0  = 0.5 * (eigval(0, 0, n_homo) + eigval(0, 0, n_lumo));
    const long   N_w     = 65;
    const long   N_Omega = 32;
    const long   N_t     = 128;
    const double Omega_max = 2.0 * w_max;
    const double freq_max  = std::max(w_max, Omega_max);
    const double dt        = 0.5 * M_PI / freq_max;
    const double T_window  = dt * static_cast<double>(N_t);
    const double beta      = 50.0;
    auto grid = real_freq_grid_t::make_uniform(
                  beta, mu0, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

    real_axis_mb_state_t state(grid);
    state.mpi = mpi_context;
    state.A_wskij.emplace(*state.mpi,
        std::array<long, 5>{N_w, ns, Nk_ibz, nbnd, nbnd});
    if (state.A_wskij->node_comm()->root()) {
      auto A = state.A_wskij->local();
      A = cval_t(0.0, 0.0);
      const double eta = 0.05;
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk_ibz; ++k)
          for (long n = 0; n < nbnd; ++n) {
            const double e = eigval(s, k, n);
            for (long iw = 0; iw < N_w; ++iw) {
              const double w_l = grid.w()(iw) + grid.mu_chem();
              const double v = (1.0 / M_PI) * eta
                              / ((w_l - e)*(w_l - e) + eta*eta);
              A(iw, s, k, n, n) = cval_t(v, 0.0);
            }
          }
    }
    state.A_wskij->node_sync();

    // Run only update_w (R-space path; kp_trev conj-copy handles the
    // TR-pair k's, qp_trev does not enter the R→q FT projection).
    real_axis_scr_coulomb_t scr(&grid, "rpa", "ignore_g0", 1e-8);
    scr.update_w(state, thc, /*verbose*/ false, /*use_rspace*/ true);

    REQUIRE(state.ImW_qPQO.has_value());
    REQUIRE(state.ImW_qPQO->global_shape()[0] == Nq_ibz);

    // Sanity: ImW finite + non-zero.
    auto ImW = state.ImW_qPQO->local();
    double sum_abs = 0.0;
    bool all_finite = true;
    double max_abs = 0.0;
    for (long i = 0; i < ImW.size(); ++i) {
      const auto v = ImW.data()[i];
      sum_abs += std::abs(v);
      max_abs = std::max(max_abs, std::abs(v));
      if (!std::isfinite(v.real()) or !std::isfinite(v.imag()))
        all_finite = false;
    }
    REQUIRE(all_finite);
    REQUIRE(sum_abs > 0.0);
    REQUIRE(max_abs < 1.0e3);

    if (mpi_context->comm.root()) {
      app_log(2, "[trev_lih223_inv update_w] kp_trev fired for {}/{} k; "
                  "max|ImW| = {:.3e}",
              n_kp_trev, kp_trev.shape()[0], max_abs);
    }

    // Also run gw_t::evaluate (Sigma_c via isym kernel with qp_trev branch).
    // For our real-axis kernel B = -ImW/pi is purely real, so qp_trev=true
    // collapses to an index swap on the k-lookup (no W conjugation needed).
    real_axis_gw_t gw(grid, /*max_iter*/ 1, /*mix*/ 0.5,
                      /*eps_nufft*/ 1e-8, /*ntrans*/ 1);
    gw.evaluate(state, thc, /*eps_nufft*/ 1e-8, /*div_treatment*/ "ignore_g0",
                /*verbose*/ false, /*use_rspace*/ false);
    REQUIRE(state.ImSigma_wskij.has_value());
    REQUIRE(state.ImSigma_wskij->shape()[2] == Nk_ibz);
    auto ImS = state.ImSigma_wskij->local();
    auto ReS = state.ReSigma_wskij->local();
    double max_abs_S = 0.0;
    bool sigma_finite = true;
    for (long i = 0; i < ImS.size(); ++i) {
      const auto vi = ImS.data()[i];
      const auto vr = ReS.data()[i];
      max_abs_S = std::max({max_abs_S, std::abs(vi), std::abs(vr)});
      if (!std::isfinite(vi.real()) or !std::isfinite(vr.real()))
        sigma_finite = false;
    }
    REQUIRE(sigma_finite);
    REQUIRE(max_abs_S > 0.0);
    REQUIRE(max_abs_S < 1.0e3);
    if (mpi_context->comm.root())
      app_log(2, "[trev_lih223_inv evaluate] qp_trev fired for some q; "
                  "max|Sigma| = {:.3e}", max_abs_S);
  }

} // namespace bdft_tests
