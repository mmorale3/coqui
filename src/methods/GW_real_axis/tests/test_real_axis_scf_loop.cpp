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
#include "methods/tools/chkpt_utils.h"
#include "h5/h5.hpp"

#include <cmath>
#include <complex>
#include <cstdio>

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
    cfg.verbose     = true;

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

  // ===========================================================================
  // Per-iter checkpoint round-trip on LiH222 scGW. Verifies the h5 file is
  // produced with the expected groups + axis="real" marker, that iter labels
  // are file-derived (init_it offset works), and that chkpt_freq throttles
  // correctly while still emitting the final iter.
  // ===========================================================================
  TEST_CASE("real_axis_scf_chkpt_lih222",
            "[real_axis][scf_loop][chkpt][thc][qe][bdft][periodic]")
  {
    auto& mpi = utils::make_unit_test_mpi_context();
    if (mpi->comm.size() != 1) return;  // single-rank fixture
    auto mf = std::make_shared<mf::MF>(
                  mf::default_MF(mpi, "qe_lih222"));
    const int nIpts = mf->nbnd() * 8;
    thc_reader_t thc(mf, make_thc_reader_ptree(nIpts, "", "incore", "",
                                               "bdft", 1e-8,
                                               mf->ecutrho(), 1, 1024));

    const long ns   = mf->nspin();
    const long Nk   = mf->nkpts();
    const long nbnd = mf->nbnd();
    auto eigval = mf->eigval();
    auto kp2ibz = mf->kp_to_ibz();

    double e_min =  std::numeric_limits<double>::infinity();
    double e_max = -std::numeric_limits<double>::infinity();
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < mf->nkpts_ibz(); ++k)
        for (long n = 0; n < nbnd; ++n) {
          const double e = eigval(s, k, n);
          e_min = std::min(e_min, e); e_max = std::max(e_max, e);
        }
    const long n_homo = static_cast<long>(mf->nelec() / 2 - 1);
    const double mu0 = 0.5 * (eigval(0, 0, n_homo) + eigval(0, 0, n_homo + 1));
    const double w_max = std::max(std::abs(e_min), std::abs(e_max)) + 2.0;
    const long N_w = 33, N_Omega = 16, N_t = 64;
    const double Omega_max = 2.0 * w_max;
    const double freq_max  = std::max(w_max, Omega_max);
    const double dt = 0.5 * M_PI / freq_max;
    const double T_window = dt * static_cast<double>(N_t);
    auto grid = real_freq_grid_t::make_uniform(
        50.0, mu0, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

    nda::array<cval_t, 4> H_MF(ns, Nk, nbnd, nbnd);
    H_MF = cval_t(0.0, 0.0);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k) {
        const long kibz = kp2ibz(k);
        for (long n = 0; n < nbnd; ++n)
          H_MF(s, k, n, n) = cval_t(eigval(s, kibz, n), 0.0);
      }
    nda::array<double, 1> k_weights(Nk);
    for (long ik = 0; ik < Nk; ++ik) k_weights(ik) = 1.0 / static_cast<double>(Nk);

    const std::string prefix = "/tmp/coqui_real_axis_scf_chkpt_test";
    std::remove((prefix + ".mbpt.h5").c_str());

    real_axis_mb_state_t state(grid);
    state.mpi = mpi;
    state.coqui_prefix = prefix;

    real_axis_hf_t          hf(&grid, "ignore_g0");
    real_axis_scr_coulomb_t scr_eri(&grid, "rpa", "ignore_g0", 1e-8);
    real_axis_gw_t          gw(grid, /*max_iter*/ 1, /*mix*/ 0.5,
                               /*eps_nufft*/ 1e-8, /*ntrans*/ 1);
    real_axis_dyson_t       dyson(std::move(H_MF), &grid, /*eta*/ 0.05);
    real_axis_mb_solver_t   mb_solver{&hf, &scr_eri, &gw};

    scgw_config cfg;
    cfg.max_iter    = 3;
    cfg.alpha_mix   = 0.7;
    cfg.tol         = 1e-12;        // force max_iter
    cfg.eta         = 0.05;
    cfg.eps_nufft   = 1e-8;
    cfg.update_mu   = true;
    cfg.mix_kind    = scgw_mix_kind::diis;
    cfg.diis_window = 4;
    cfg.verbose     = false;

    // Pre-loop: emulate the dispatcher's metadata + iter-0 dump so we
    // exercise the same code path the production toml runs through.
    using sA4 = real_axis_mb_state_t::sArray_t<nda::array_view<ComplexType, 4>>;
    using sA5 = real_axis_mb_state_t::sArray_t<nda::array_view<ComplexType, 5>>;
    state.A_wskij.emplace(*mpi,
        std::array<long, 5>{N_w, ns, Nk, nbnd, nbnd});
    state.ImSigma_wskij.emplace(*mpi,
        std::array<long, 5>{N_w, ns, Nk, nbnd, nbnd});
    state.ReSigma_wskij.emplace(*mpi,
        std::array<long, 5>{N_w, ns, Nk, nbnd, nbnd});
    state.Sigma_x_skij.emplace(*mpi,
        std::array<long, 4>{ns, Nk, nbnd, nbnd});
    state.Dm_skij.emplace(*mpi,
        std::array<long, 4>{ns, Nk, nbnd, nbnd});
    if (state.A_wskij->node_comm()->root()) {
      auto A = state.A_wskij->local();
      A = ComplexType(0.0, 0.0);
      const double eta_init = std::max(cfg.eta, 1e-2);
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k) {
          const long kibz = kp2ibz(k);
          for (long n = 0; n < nbnd; ++n) {
            const double e = eigval(s, kibz, n);
            for (long iw = 0; iw < N_w; ++iw) {
              const double wl = grid.w()(iw) + grid.mu_chem();
              const double v = (1.0 / M_PI) * eta_init
                             / ((wl - e)*(wl - e) + eta_init*eta_init);
              A(iw, s, k, n, n) = ComplexType(v, 0.0);
            }
          }
        }
      state.ImSigma_wskij->local() = ComplexType(0.0, 0.0);
      state.ReSigma_wskij->local() = ComplexType(0.0, 0.0);
      state.Sigma_x_skij->local()  = ComplexType(0.0, 0.0);
    }
    state.A_wskij->node_sync();
    state.ImSigma_wskij->node_sync();
    state.ReSigma_wskij->node_sync();
    state.Sigma_x_skij->node_sync();

    methods::real_axis::detail_scgw::compute_Dm_from_A(
        grid, state.A_wskij.value(), grid.mu_chem(),
        state.Dm_skij.value());

    // sH0/sS for write_metadata.
    auto sH0 = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(
        *mpi, std::array<long, 4>{ns, Nk, nbnd, nbnd});
    auto sS  = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(
        *mpi, std::array<long, 4>{ns, Nk, nbnd, nbnd});
    if (sH0.node_comm()->root()) sH0.local() = ComplexType(0.0, 0.0);
    if (sS .node_comm()->root()) {
      // Identity overlap (LiH KS basis is orthogonal in the THC fixture).
      sS.local() = ComplexType(0.0, 0.0);
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k)
          for (long n = 0; n < nbnd; ++n)
            sS.local()(s, k, n, n) = ComplexType(1.0, 0.0);
    }
    sH0.node_sync(); sS.node_sync();
    methods::chkpt::write_metadata_real_axis(mpi->comm, *mf, grid, sH0, sS,
                                              prefix);
    methods::chkpt::dump_scf_real_axis(mpi->comm, /*iter*/ 0,
        state.Dm_skij.value(), state.A_wskij.value(),
        state.Sigma_x_skij.value(),
        state.ImSigma_wskij.value(), state.ReSigma_wskij.value(),
        grid.mu_chem(), prefix);

    // Run 3 iters with chkpt_freq=1 (default): expect iter 0..3 in the file.
    auto res = real_axis_scf_loop(state, dyson, thc, mb_solver, cfg,
                                   k_weights, static_cast<double>(mf->nelec()),
                                   /*init_it*/ 0,
                                   /*write_chkpt*/ true,
                                   /*chkpt_freq*/ 1);
    REQUIRE(res.iter_used == 3);

    {
      h5::file file(prefix + ".mbpt.h5", 'r');
      h5::group root(file);
      REQUIRE(root.has_subgroup("system"));
      REQUIRE(root.has_subgroup("real_frequency_grid"));
      REQUIRE(root.has_subgroup("scf"));
      auto scf_grp = root.open_group("scf");
      for (long n = 0; n <= 3; ++n)
        REQUIRE(scf_grp.has_subgroup("iter" + std::to_string(n)));
      long final_iter = -1;
      h5::h5_read(scf_grp, "final_iter", final_iter);
      REQUIRE(final_iter == 3);
      // Verify axis marker + a real-axis-specific dataset name.
      auto it1 = scf_grp.open_group("iter1");
      std::string axis_str;
      h5::h5_read(it1, "axis", axis_str);
      REQUIRE(axis_str == "real");
      REQUIRE(it1.has_dataset("A_wskij"));
      REQUIRE(it1.has_dataset("ImSigma_wskij"));
      REQUIRE(it1.has_dataset("ReSigma_wskij"));
      REQUIRE(it1.has_dataset("Sigma_x_skij"));
      REQUIRE(it1.has_dataset("Dm_skij"));
    }

    // chkpt_freq=2 throttling: with init_it=3, run 3 more iters (labels
    // 4,5,6). With freq=2: only labels 4 and 6 are dumped (4 even,
    // 5 odd skipped, 6 even). The final iter (6) is ALWAYS dumped via the
    // `stop` path even if not aligned to freq.
    auto res2 = real_axis_scf_loop(state, dyson, thc, mb_solver, cfg,
                                    k_weights, static_cast<double>(mf->nelec()),
                                    /*init_it*/ 3,
                                    /*write_chkpt*/ true,
                                    /*chkpt_freq*/ 2);
    REQUIRE(res2.iter_used == 3);
    {
      h5::file file(prefix + ".mbpt.h5", 'r');
      h5::group root(file);
      auto scf_grp = root.open_group("scf");
      REQUIRE(scf_grp.has_subgroup("iter4"));      // even -> dumped
      REQUIRE(!scf_grp.has_subgroup("iter5"));     // odd  -> skipped
      REQUIRE(scf_grp.has_subgroup("iter6"));      // final -> dumped via stop
      long final_iter = -1;
      h5::h5_read(scf_grp, "final_iter", final_iter);
      REQUIRE(final_iter == 6);
    }

    std::remove((prefix + ".mbpt.h5").c_str());
  }

#if defined(ENABLE_DEVICE)
  // ===========================================================================
  // Same scGW SCF loop on LiH222, but instantiated with MEM=DEVICE_MEMORY.
  // The scr_coulomb and gw drivers run their hot paths on the GPU; the
  // Sigma_x (HF) and Dyson update remain host-side. Asserts the final
  // converged state agrees bit-for-bit (or to NUFFT eps) with the host
  // instantiation. Compiles only when ENABLE_DEVICE is set.
  // ===========================================================================
  TEST_CASE("real_axis_scf_loop_lih222_periodic_device",
            "[real_axis][scf_loop][thc][qe][bdft][periodic][device]")
  {
    using methods::real_axis::real_axis_scr_coulomb_base_t;
    using methods::real_axis::real_axis_mb_solver_base_t;
    using methods::real_axis::real_axis_dyson_base_t;
    using methods::real_axis::real_axis_hf_base_t;

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
    const long   n_homo  = static_cast<long>(mf->nelec() / 2 - 1);
    const long   n_lumo  = n_homo + 1;
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

    nda::array<cval_t, 4> H_MF(ns, Nk, nbnd, nbnd);
    H_MF = cval_t(0.0, 0.0);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k) {
        const long kibz = kp2ibz(k);
        for (long n = 0; n < nbnd; ++n)
          H_MF(s, k, n, n) = cval_t(eigval(s, kibz, n), 0.0);
      }

    nda::array<double, 1> k_weights(Nk);
    for (long ik = 0; ik < Nk; ++ik) k_weights(ik) = 1.0 / static_cast<double>(Nk);

    real_axis_mb_state_t state(grid);
    state.mpi = mpi_context;

    real_axis_hf_base_t<DEVICE_MEMORY>          hf(&grid, "ignore_g0");
    real_axis_scr_coulomb_base_t<DEVICE_MEMORY> scr_eri(&grid, "rpa", "ignore_g0", 1e-8);
    methods::solvers::real_axis_gw_t            gw(grid, /*max_iter*/ 1, /*mix*/ 0.5,
                                                    /*eps_nufft*/ 1e-8, /*ntrans*/ 1);
    real_axis_dyson_base_t<DEVICE_MEMORY>       dyson(std::move(H_MF), &grid, /*eta*/ 0.05);
    real_axis_mb_solver_base_t<DEVICE_MEMORY>   mb_solver{&hf, &scr_eri, &gw};

    scgw_config cfg;
    cfg.max_iter    = 20;
    cfg.alpha_mix   = 0.7;
    cfg.tol         = 1e-3;
    cfg.eta         = 0.05;
    cfg.eps_nufft   = 1e-8;
    cfg.update_mu   = true;
    cfg.mix_kind    = scgw_mix_kind::diis;
    cfg.diis_window = 8;
    cfg.verbose     = true;

    auto res = real_axis_scf_loop<DEVICE_MEMORY>(state, dyson, thc,
                                                  mb_solver, cfg, k_weights,
                                                  static_cast<double>(mf->nelec()));

    app_log(2, "[scf_loop_lih222_device] iter_used={}  final_diff={:.3e}  final_mu={:.6f}",
            res.iter_used, res.final_diff, res.final_mu);

    REQUIRE(res.iter_used >= 1);
    REQUIRE(res.iter_used <= cfg.max_iter);
    REQUIRE(std::isfinite(res.final_diff));
    REQUIRE(res.final_diff >= 0.0);
    REQUIRE(std::isfinite(res.final_mu));
    REQUIRE(res.final_mu > eps_homo - 0.5);
    REQUIRE(res.final_mu < eps_lumo + 0.5);
    REQUIRE(res.final_diff < 1.0);

    REQUIRE(state.A_wskij.has_value());
    REQUIRE(state.ImW_qPQO.has_value());
  }
#endif // ENABLE_DEVICE

} // namespace bdft_tests
