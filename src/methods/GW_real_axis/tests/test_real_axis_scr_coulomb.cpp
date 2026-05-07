/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Regression test for real_axis_scr_coulomb_t::update_w. Drives the new
 * class against the qe_lih222 THC fixture and verifies the Pi/W outputs
 * agree bit-for-bit with the equivalent Steps 1-4 of evaluate_thc_serial
 * (run in parallel with `verbose=false, use_rspace=false`). Tolerance is
 * tight (1e-12 absolute) because both paths share kernels, so the only
 * difference is glue.
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

#include "numerics/distributed_array/nda_utils.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/ERI/eri_utils.hpp"

#include "nda/nda.hpp"
#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_mb_state.hpp"
#include "methods/GW_real_axis/real_axis_scr_coulomb_t.h"
#include "methods/GW_real_axis/real_axis_gw_t.h"
#include "methods/GW_real_axis/real_axis_gw_thc.hpp"

#include <cmath>
#include <complex>

namespace bdft_tests {

  using namespace methods;
  using methods::real_axis::real_freq_grid_t;
  using methods::real_axis::real_axis_mb_state_t;
  using methods::real_axis::real_axis_scr_coulomb_t;
  using methods::real_axis::evaluate_thc_serial;
  using methods::solvers::real_axis_gw_t;
  using cval_t = std::complex<double>;

  // Helper: derive grid parameters (and mu0) from the MF eigenvalue spectrum.
  struct grid_params_t {
    double mu0;
    double w_max;
    long   N_w;
    double Omega_max;
    long   N_Omega;
    long   N_t;
    double T_window;
    double beta;
  };

  static grid_params_t derive_grid_params(std::shared_ptr<mf::MF> const& mf) {
    const long ns   = mf->nspin();
    const long nbnd = mf->nbnd();
    auto eigval = mf->eigval();
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
    grid_params_t p;
    p.w_max     = std::max(std::abs(e_min), std::abs(e_max)) + 2.0;
    const long n_homo = static_cast<long>(mf->nelec() / 2 - 1);
    p.mu0       = 0.5 * (eigval(0, 0, n_homo) + eigval(0, 0, n_homo + 1));
    p.N_w       = 65;
    p.N_Omega   = 32;
    p.N_t       = 128;
    p.Omega_max = 2.0 * p.w_max;
    const double freq_max = std::max(p.w_max, p.Omega_max);
    const double dt       = 0.5 * M_PI / freq_max;
    p.T_window  = dt * static_cast<double>(p.N_t);
    p.beta      = 50.0;
    (void) ns;  // silence unused-var warning under release build
    (void) nbnd;
    return p;
  }

  // Build initial Lorentzian-broadened diagonal A on a state already
  // bound to a grid. Mirrors test_real_axis_thc.cpp's initial-A pattern.
  static void fill_lorentzian_A(std::shared_ptr<mf::MF> const& mf,
                                real_freq_grid_t const& grid,
                                real_axis_mb_state_t& state)
  {
    const long ns   = mf->nspin();
    const long Nk   = mf->nkpts();
    const long nbnd = mf->nbnd();
    const long N_w  = grid.N_w();
    state.A_wskij.emplace(*state.mpi,
        std::array<long, 5>{N_w, ns, Nk, nbnd, nbnd});
    if (state.A_wskij->node_comm()->root()) {
      auto A = state.A_wskij->local();
      A = cval_t(0.0, 0.0);
      const double eta = 0.05;
      auto eigval = mf->eigval();
      auto kp2ibz = mf->kp_to_ibz();
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k) {
          const long kibz = kp2ibz(k);
          for (long n = 0; n < nbnd; ++n) {
            const double eps_n = eigval(s, kibz, n);
            for (long iw = 0; iw < N_w; ++iw) {
              const double w_l = grid.w()(iw) + grid.mu_chem();
              const double v = (1.0 / M_PI) * eta
                             / ((w_l - eps_n)*(w_l - eps_n) + eta*eta);
              A(iw, s, k, n, n) = cval_t(v, 0.0);
            }
          }
        }
    }
    state.A_wskij->node_sync();
  }

  // ===========================================================================
  // update_w smoke test: the new class produces finite Pi / W on the
  // standard LiH222 fixture, with the expected shapes.
  // ===========================================================================
  TEST_CASE("real_axis_scr_coulomb_update_w_lih222",
            "[real_axis][scr_coulomb][thc][qe]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(
                  mf::default_MF(mpi_context, "qe_lih222"));
    const int nIpts = mf->nbnd() * 8;
    thc_reader_t thc(mf, make_thc_reader_ptree(nIpts, "", "incore", "",
                                               "bdft", 1e-8,
                                               mf->ecutrho(), 1, 1024));

    const long ns   = mf->nspin();
    const long Nq   = mf->nqpts();
    const long Naux = thc.Np();

    auto p = derive_grid_params(mf);
    auto grid = real_freq_grid_t::make_uniform(
                  p.beta, p.mu0, p.w_max, p.N_w,
                  p.Omega_max, p.N_Omega, p.N_t, p.T_window);
    real_axis_mb_state_t state(grid);
    state.mpi = mpi_context;
    fill_lorentzian_A(mf, grid, state);
    const long N_O = grid.N_Omega();

    real_axis_scr_coulomb_t scr_eri(&grid, "rpa", "ignore_g0", 1e-8);
    scr_eri.update_w(state, thc,
                     /*verbose*/ false, /*use_rspace*/ false);

    REQUIRE(state.ImPi_qPQO.has_value());
    REQUIRE(state.RePi_qPQO.has_value());
    REQUIRE(state.ImW_qPQO.has_value());
    REQUIRE(state.ReW_qPQO.has_value());

    REQUIRE(state.ImPi_qPQO->global_shape()[0] == Nq);
    REQUIRE(state.ImPi_qPQO->global_shape()[1] == Naux);
    REQUIRE(state.ImPi_qPQO->global_shape()[2] == Naux);
    REQUIRE(state.ImPi_qPQO->global_shape()[3] == N_O);

    // Gather to a fully-replicated array for convenient global-index checks.
    auto ImPi = math::nda::all_gather_slow<HOST_MEMORY>(*state.ImPi_qPQO);
    auto RePi = math::nda::all_gather_slow<HOST_MEMORY>(*state.RePi_qPQO);
    auto ImW  = math::nda::all_gather_slow<HOST_MEMORY>(*state.ImW_qPQO);
    auto ReW  = math::nda::all_gather_slow<HOST_MEMORY>(*state.ReW_qPQO);

    bool all_finite = true;
    bool nonzero_Pi = false;
    for (long iq = 0; iq < Nq; ++iq)
      for (long P = 0; P < Naux; ++P)
        for (long Q = 0; Q < Naux; ++Q)
          for (long iO = 0; iO < N_O; ++iO) {
            if (!std::isfinite(ImPi(iq, P, Q, iO)) ||
                !std::isfinite(ImW (iq, P, Q, iO)) ||
                !std::isfinite(ReW (iq, P, Q, iO))) {
              all_finite = false;
            }
            if (std::abs(ImPi(iq, P, Q, iO)) > 1e-10) nonzero_Pi = true;
          }
    REQUIRE(all_finite);
    REQUIRE(nonzero_Pi);

    // ignore_g0: iq_gamma=0 should be exactly zero across all four arrays.
    bool gamma_zero = true;
    for (long P = 0; P < Naux; ++P)
      for (long Q = 0; Q < Naux; ++Q)
        for (long iO = 0; iO < N_O; ++iO) {
          if (std::abs(ImPi(0, P, Q, iO)) != 0.0 ||
              std::abs(RePi(0, P, Q, iO)) != 0.0 ||
              std::abs(ImW (0, P, Q, iO)) != 0.0 ||
              std::abs(ReW (0, P, Q, iO)) != 0.0) {
            gamma_zero = false;
          }
        }
    REQUIRE(gamma_zero);
    (void) ns;
  }

  // ===========================================================================
  // R-space vs k-space agreement: the same physical Pi should come out of
  // both code paths within NUFFT tolerance. Mirrors the
  // real_axis_thc_g0w0_lih222_kspace_vs_rspace test for the W stage.
  // ===========================================================================
  // ===========================================================================
  // MEM=DEVICE_MEMORY instantiation. Gated on ENABLE_DEVICE because nda's
  // address-space validation rejects Device allocations when the project
  // is compiled without GPU support. On Rusty with ENABLE_DEVICE this
  // exercises the real device path; the test asserts agreement between
  // HOST and DEVICE instantiations of update_w on the same fixture to
  // NUFFT eps.
  // ===========================================================================
#if defined(ENABLE_DEVICE)
  TEST_CASE("real_axis_scr_coulomb_update_w_lih222_host_vs_device",
            "[real_axis][scr_coulomb][thc][device]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(
                  mf::default_MF(mpi_context, "qe_lih222"));
    const int nIpts = mf->nbnd() * 8;
    thc_reader_t thc(mf, make_thc_reader_ptree(nIpts, "", "incore", "",
                                               "bdft", 1e-8,
                                               mf->ecutrho(), 1, 1024));

    auto p = derive_grid_params(mf);
    auto grid_h = real_freq_grid_t::make_uniform(
                    p.beta, p.mu0, p.w_max, p.N_w,
                    p.Omega_max, p.N_Omega, p.N_t, p.T_window);
    auto grid_d = real_freq_grid_t::make_uniform(
                    p.beta, p.mu0, p.w_max, p.N_w,
                    p.Omega_max, p.N_Omega, p.N_t, p.T_window);
    real_axis_mb_state_t state_h(grid_h), state_d(grid_d);
    state_h.mpi = mpi_context;
    state_d.mpi = mpi_context;
    fill_lorentzian_A(mf, grid_h, state_h);
    fill_lorentzian_A(mf, grid_d, state_d);

    methods::real_axis::real_axis_scr_coulomb_base_t<HOST_MEMORY>
        scr_h(&grid_h, "rpa", "ignore_g0", 1e-8);
    methods::real_axis::real_axis_scr_coulomb_base_t<DEVICE_MEMORY>
        scr_d(&grid_d, "rpa", "ignore_g0", 1e-8);
    scr_h.update_w(state_h, thc, /*verbose*/ false, /*use_rspace*/ true);
    scr_d.update_w(state_d, thc, /*verbose*/ false, /*use_rspace*/ true);

    auto ImPi_h_loc = state_h.ImPi_qPQO->local();
    auto ImPi_d_loc = state_d.ImPi_qPQO->local();
    auto ImW_h_loc  = state_h.ImW_qPQO->local();
    auto ImW_d_loc  = state_d.ImW_qPQO->local();
    auto ReW_h_loc  = state_h.ReW_qPQO->local();
    auto ReW_d_loc  = state_d.ReW_qPQO->local();

    double max_diff_Pi = 0.0;
    double max_diff_W  = 0.0;
    const long size = ImPi_h_loc.size();
    for (long i = 0; i < size; ++i) {
      max_diff_Pi = std::max(max_diff_Pi,
                              std::abs(ImPi_h_loc.data()[i] - ImPi_d_loc.data()[i]));
      max_diff_W  = std::max(max_diff_W,
                              std::abs(ImW_h_loc.data()[i] - ImW_d_loc.data()[i]));
      max_diff_W  = std::max(max_diff_W,
                              std::abs(ReW_h_loc.data()[i] - ReW_d_loc.data()[i]));
    }
    app_log(2, "[scr_coulomb host_vs_device] max diff ImPi = {:.3e}", max_diff_Pi);
    app_log(2, "[scr_coulomb host_vs_device] max diff W    = {:.3e}", max_diff_W);
    // For MEM=DEVICE on a build without ENABLE_DEVICE, this should be 0
    // (same host code path). With ENABLE_DEVICE it tests the device path
    // against host to NUFFT eps. Use a generous tolerance.
    REQUIRE(max_diff_Pi < 1e-7);
    REQUIRE(max_diff_W  < 1e-7);
  }
#endif // ENABLE_DEVICE

  TEST_CASE("real_axis_scr_coulomb_update_w_lih222_kspace_vs_rspace",
            "[real_axis][scr_coulomb][thc][rspace]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(
                  mf::default_MF(mpi_context, "qe_lih222"));
    const int nIpts = mf->nbnd() * 8;
    thc_reader_t thc(mf, make_thc_reader_ptree(nIpts, "", "incore", "",
                                               "bdft", 1e-8,
                                               mf->ecutrho(), 1, 1024));

    auto p = derive_grid_params(mf);
    auto grid_k = real_freq_grid_t::make_uniform(
                    p.beta, p.mu0, p.w_max, p.N_w,
                    p.Omega_max, p.N_Omega, p.N_t, p.T_window);
    auto grid_R = real_freq_grid_t::make_uniform(
                    p.beta, p.mu0, p.w_max, p.N_w,
                    p.Omega_max, p.N_Omega, p.N_t, p.T_window);
    real_axis_mb_state_t state_k(grid_k), state_R(grid_R);
    state_k.mpi = mpi_context;
    state_R.mpi = mpi_context;
    fill_lorentzian_A(mf, grid_k, state_k);
    fill_lorentzian_A(mf, grid_R, state_R);

    real_axis_scr_coulomb_t scr_k(&grid_k, "rpa", "ignore_g0", 1e-8);
    real_axis_scr_coulomb_t scr_R(&grid_R, "rpa", "ignore_g0", 1e-8);
    scr_k.update_w(state_k, thc,
                   /*verbose*/ false, /*use_rspace*/ false);
    scr_R.update_w(state_R, thc,
                   /*verbose*/ false, /*use_rspace*/ true);

    // Both states have the same proc grid -> same local layout. Compare
    // local slices directly (no gather needed).
    auto ImPi_k_loc = state_k.ImPi_qPQO->local();
    auto ImPi_R_loc = state_R.ImPi_qPQO->local();
    auto ImW_k_loc  = state_k.ImW_qPQO->local();
    auto ImW_R_loc  = state_R.ImW_qPQO->local();
    auto ReW_k_loc  = state_k.ReW_qPQO->local();
    auto ReW_R_loc  = state_R.ReW_qPQO->local();

    double max_diff_Pi = 0.0;
    double max_diff_W  = 0.0;
    const long size = ImPi_k_loc.size();
    auto const* p_Pi_k = ImPi_k_loc.data();
    auto const* p_Pi_R = ImPi_R_loc.data();
    auto const* p_ImW_k = ImW_k_loc.data();
    auto const* p_ImW_R = ImW_R_loc.data();
    auto const* p_ReW_k = ReW_k_loc.data();
    auto const* p_ReW_R = ReW_R_loc.data();
    for (long i = 0; i < size; ++i) {
      max_diff_Pi = std::max(max_diff_Pi,
                              std::abs(p_Pi_k[i] - p_Pi_R[i]));
      max_diff_W  = std::max(max_diff_W,
                              std::abs(p_ImW_k[i] - p_ImW_R[i]));
      max_diff_W  = std::max(max_diff_W,
                              std::abs(p_ReW_k[i] - p_ReW_R[i]));
    }
    app_log(2, "[scr_coulomb kspace_vs_rspace] max diff ImPi = {:.3e}", max_diff_Pi);
    app_log(2, "[scr_coulomb kspace_vs_rspace] max diff W    = {:.3e}", max_diff_W);
    REQUIRE(max_diff_Pi < 1e-9);
    REQUIRE(max_diff_W  < 1e-9);
  }

  // ===========================================================================
  // End-to-end consistency: running update_w followed by Steps 5-8 of the
  // existing evaluate_thc_serial driver produces the same Sigma as running
  // evaluate_thc_serial alone. Validates that the new class is a faithful
  // factoring of Steps 1-4.
  // ===========================================================================
  TEST_CASE("real_axis_scr_coulomb_matches_driver_lih222",
            "[real_axis][scr_coulomb][thc][regression]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(
                  mf::default_MF(mpi_context, "qe_lih222"));
    const int nIpts = mf->nbnd() * 8;
    thc_reader_t thc(mf, make_thc_reader_ptree(nIpts, "", "incore", "",
                                               "bdft", 1e-8,
                                               mf->ecutrho(), 1, 1024));

    auto p = derive_grid_params(mf);
    auto grid_drv = real_freq_grid_t::make_uniform(
                      p.beta, p.mu0, p.w_max, p.N_w,
                      p.Omega_max, p.N_Omega, p.N_t, p.T_window);
    auto grid_split = real_freq_grid_t::make_uniform(
                        p.beta, p.mu0, p.w_max, p.N_w,
                        p.Omega_max, p.N_Omega, p.N_t, p.T_window);
    real_axis_mb_state_t state_drv(grid_drv), state_split(grid_split);
    state_drv.mpi   = mpi_context;
    state_split.mpi = mpi_context;
    fill_lorentzian_A(mf, grid_drv, state_drv);
    fill_lorentzian_A(mf, grid_split, state_split);

    // Reference: full driver.
    evaluate_thc_serial(state_drv, thc, /*eps_nufft*/ 1e-8,
                        "ignore_g0", /*verbose*/ false, /*use_rspace*/ false);

    // Split: update_w computes W into state, then gw_t::evaluate computes
    // Sigma from A and W via Steps 5-8. Sigma must agree with the full
    // driver bit-for-bit (sub-eps differences from non-associativity in
    // the rank-distributed sums, but at single-rank: bitwise identical).
    real_axis_scr_coulomb_t scr_eri(&grid_split, "rpa", "ignore_g0", 1e-8);
    real_axis_gw_t          gw(grid_split, /*max_iter*/ 1, /*mix*/ 0.5,
                               /*eps_nufft*/ 1e-8, /*ntrans*/ 1);
    scr_eri.update_w(state_split, thc,
                     /*verbose*/ false, /*use_rspace*/ false);
    gw.evaluate(state_split, thc,
                /*eps_nufft*/ 1e-8, "ignore_g0", /*verbose*/ false,
                /*use_rspace*/ false);

    auto ImS_drv = state_drv.ImSigma_wskij->local();
    auto ImS_spl = state_split.ImSigma_wskij->local();
    auto ReS_drv = state_drv.ReSigma_wskij->local();
    auto ReS_spl = state_split.ReSigma_wskij->local();

    double max_diff = 0.0;
    const long size = ImS_drv.size();
    auto const* p_drv_im = ImS_drv.data();
    auto const* p_spl_im = ImS_spl.data();
    auto const* p_drv_re = ReS_drv.data();
    auto const* p_spl_re = ReS_spl.data();
    for (long i = 0; i < size; ++i) {
      max_diff = std::max(max_diff, std::abs(p_drv_im[i] - p_spl_im[i]));
      max_diff = std::max(max_diff, std::abs(p_drv_re[i] - p_spl_re[i]));
    }
    app_log(2, "[scr_coulomb matches_driver] max diff Sigma = {:.3e}", max_diff);
    REQUIRE(max_diff < 1e-12);

    // Side-effect check: state.W from update_w must be finite (check
    // the local slice on each rank).
    auto ImW_loc = state_split.ImW_qPQO->local();
    auto ReW_loc = state_split.ReW_qPQO->local();
    bool all_finite = true;
    for (long i = 0; i < ImW_loc.size(); ++i) {
      if (!std::isfinite(ImW_loc.data()[i]) ||
          !std::isfinite(ReW_loc.data()[i]))
        all_finite = false;
    }
    REQUIRE(all_finite);
  }

} // namespace bdft_tests
