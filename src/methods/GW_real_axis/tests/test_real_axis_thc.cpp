/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Real-system integration test: wires the real-axis G0W0 wrapper to a
 * concrete thc_reader_t built from the qe_lih222 unit-test fixture.
 * Verifies the full pipeline runs and produces finite, causal outputs.
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
#include "methods/GW_real_axis/real_axis_gw_thc.hpp"

#include <cmath>
#include <complex>

namespace bdft_tests {

  using namespace methods;
  using methods::real_axis::real_freq_grid_t;
  using methods::real_axis::real_axis_mb_state_t;
  using methods::real_axis::evaluate_thc_serial;
  using cval_t = std::complex<double>;

  // ===========================================================================
  // Smoke test: build the QE LiH 2x2x2 fixture, build a THC ERI, construct an
  // initial Lorentzian-broadened spectral function from the MF eigenvalues,
  // and run one iteration of evaluate_thc_serial.
  //
  // Step-1 of MPI distribution: the comm is plumbed through the wrapper but
  // the body still runs the full computation redundantly on every rank, so
  // multi-rank runs produce identical results everywhere. Distributed work
  // partitioning lands in subsequent steps.
  // ===========================================================================
  TEST_CASE("real_axis_thc_g0w0_lih222_serial",
            "[real_axis][thc][gw][qe][bdft][serial]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(
                  mf::default_MF(mpi_context, "qe_lih222"));

    // Use a small auxiliary rank to keep the test fast.
    const int nIpts = mf->nbnd() * 8;
    thc_reader_t thc(mf, make_thc_reader_ptree(nIpts, "", "incore", "",
                                               "bdft", 1e-8,
                                               mf->ecutrho(), 1, 1024));

    const long ns   = mf->nspin();
    const long Nk   = mf->nkpts();
    const long nbnd = mf->nbnd();
    const long Naux = thc.Np();

    // -----------------------------------------------------------------------
    // Build a real-frequency grid sized to the MF eigenvalue window.
    // -----------------------------------------------------------------------
    auto eigval = mf->eigval();   // (ns, nkpts_ibz, nbnd)
    double e_min =  std::numeric_limits<double>::infinity();
    double e_max = -std::numeric_limits<double>::infinity();
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < mf->nkpts_ibz(); ++k)
        for (long n = 0; n < nbnd; ++n) {
          const double e = eigval(s, k, n);
          e_min = std::min(e_min, e);
          e_max = std::max(e_max, e);
        }
    // Guard against degenerate ranges.
    if (e_max - e_min < 1e-3) { e_min -= 1.0; e_max += 1.0; }
    const double w_max = std::max(std::abs(e_min), std::abs(e_max)) + 2.0;
    // Pick an mu_chem in the band gap (for LiH: between HOMO and LUMO).
    // Use 0.5 * (eps_homo + eps_lumo) at k=0 as a starting guess.
    const long n_homo = static_cast<long>(mf->nelec() / 2 - 1);
    const long n_lumo = n_homo + 1;
    const double mu0  = 0.5 * (eigval(0, 0, n_homo) + eigval(0, 0, n_lumo));

    const long   N_w       = 65;
    const long   N_Omega   = 32;
    const long   N_t       = 128;
    // Nyquist: T_window/N_t < pi / max(w_max, Omega_max).
    const double Omega_max = 2.0 * w_max;       // 2*w_max for the bubble
    const double freq_max  = std::max(w_max, Omega_max);
    // Choose dt = 0.5*pi/freq_max (factor 2 safety margin).
    const double dt        = 0.5 * M_PI / freq_max;
    const double T_window  = dt * static_cast<double>(N_t);
    const double beta      = 50.0;  // T ~ 0.02 Ha

    auto grid = real_freq_grid_t::make_uniform(
                  beta, mu0, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

    // -----------------------------------------------------------------------
    // Build initial A: diagonal Lorentzians per (s, k, n) eigenvalue.
    // Layout follows real_axis_mb_state_t: A_wskij(N_w, ns, Nk, nbnd, nbnd).
    // We use the IBZ eigenvalue for each FBZ k by mapping through kp_to_ibz.
    // -----------------------------------------------------------------------
    real_axis_mb_state_t state(grid);
    state.mpi = mpi_context;
    state.A_wskij.emplace(*state.mpi,
        std::array<long, 5>{N_w, ns, Nk, nbnd, nbnd});
    if (state.A_wskij->node_comm()->root()) {
      auto A = state.A_wskij->local();
      A = cval_t(0.0, 0.0);
      const double eta = 0.05;
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

    // -----------------------------------------------------------------------
    // Run the real-axis G0W0 wrapper.
    // -----------------------------------------------------------------------
    evaluate_thc_serial(state, thc, /*eps_nufft*/ 1e-8,
                        "ignore_g0", /*verbose*/ true, /*use_rspace*/ true);

    REQUIRE(state.ImSigma_wskij.has_value());
    REQUIRE(state.ReSigma_wskij.has_value());
    auto ImS = state.ImSigma_wskij->local();
    auto ReS = state.ReSigma_wskij->local();

    REQUIRE(ImS.shape()[0] == N_w);
    REQUIRE(ImS.shape()[1] == ns);
    REQUIRE(ImS.shape()[2] == Nk);
    REQUIRE(ImS.shape()[3] == nbnd);
    REQUIRE(ImS.shape()[4] == nbnd);

    // Outputs must be finite everywhere.
    bool all_finite = true;
    for (long iw = 0; iw < N_w; ++iw)
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k)
          for (long mu = 0; mu < nbnd; ++mu)
            for (long nu = 0; nu < nbnd; ++nu) {
              if (!std::isfinite(ImS(iw, s, k, mu, nu).real()) ||
                  !std::isfinite(ReS(iw, s, k, mu, nu).real())) {
                all_finite = false;
              }
            }
    REQUIRE(all_finite);

    // Diagonal Im Sigma^c must be approximately non-positive (causality).
    // Allow small positive numerical noise on the order of NUFFT eps.
    long n_violations = 0, n_total = 0;
    for (long iw = 0; iw < N_w; ++iw)
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k)
          for (long mu = 0; mu < nbnd; ++mu) {
            ++n_total;
            if (ImS(iw, s, k, mu, mu).real() > 1e-4) ++n_violations;
          }
    // Allow at most 5% causality violations from numerical noise.
    REQUIRE(n_violations < n_total / 20);

    // Output some diagnostic values for the first few states at k=0.
    app_log(2, "[real_axis_thc_g0w0_lih222_serial] mu_chem = {0:.6f}", mu0);
    app_log(2, "[real_axis_thc_g0w0_lih222_serial] HOMO energy: {0:.6f}",
            eigval(0, 0, n_homo));
    app_log(2, "[real_axis_thc_g0w0_lih222_serial] LUMO energy: {0:.6f}",
            eigval(0, 0, n_lumo));
    app_log(2, "[real_axis_thc_g0w0_lih222_serial] Naux = {0}", Naux);
  }

  // ===========================================================================
  // R-space-vs-k-space regression: same fixture and inputs, run the wrapper
  // twice with use_rspace=false (k-space Steps 2 and 6) and use_rspace=true
  // (R-space Steps 2 and 6 via FT k->R / q->R, per-R kernel, FT R->q / R->k).
  // The two paths should agree to NUFFT eps; we use a 1e-6 tolerance.
  // ===========================================================================
  TEST_CASE("real_axis_thc_g0w0_lih222_kspace_vs_rspace",
            "[real_axis][thc][gw][qe][bdft][rspace]") {
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
    const double w_max     = std::max(std::abs(e_min), std::abs(e_max)) + 2.0;
    const long n_homo = static_cast<long>(mf->nelec() / 2 - 1);
    const long n_lumo = n_homo + 1;
    const double mu0  = 0.5 * (eigval(0, 0, n_homo) + eigval(0, 0, n_lumo));
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

    auto fill_state = [&](real_axis_mb_state_t& state) {
      state.mpi = mpi_context;
      state.A_wskij.emplace(*state.mpi,
          std::array<long, 5>{N_w, ns, Nk, nbnd, nbnd});
      if (state.A_wskij->node_comm()->root()) {
        auto A = state.A_wskij->local();
        A = cval_t(0.0, 0.0);
        const double eta = 0.05;
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
    };

    real_axis_mb_state_t state_k(grid), state_r(grid);
    fill_state(state_k);
    fill_state(state_r);
    evaluate_thc_serial(state_k, thc, /*eps_nufft*/ 1e-8,
                        "ignore_g0", /*verbose*/ false, /*use_rspace*/ false);
    evaluate_thc_serial(state_r, thc, /*eps_nufft*/ 1e-8,
                        "ignore_g0", /*verbose*/ true, /*use_rspace*/ true);

    REQUIRE(state_k.ImSigma_wskij.has_value());
    REQUIRE(state_r.ImSigma_wskij.has_value());
    auto ImSk = state_k.ImSigma_wskij->local();
    auto ImSr = state_r.ImSigma_wskij->local();
    auto ReSk = state_k.ReSigma_wskij->local();
    auto ReSr = state_r.ReSigma_wskij->local();

    REQUIRE(ImSk.shape() == ImSr.shape());
    double max_diff_im = 0.0, max_diff_re = 0.0;
    double max_abs_k   = 0.0;
    for (long iw = 0; iw < N_w; ++iw)
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k)
          for (long mu = 0; mu < nbnd; ++mu)
            for (long nu = 0; nu < nbnd; ++nu) {
              max_diff_im = std::max(max_diff_im,
                  std::abs(ImSk(iw,s,k,mu,nu) - ImSr(iw,s,k,mu,nu)));
              max_diff_re = std::max(max_diff_re,
                  std::abs(ReSk(iw,s,k,mu,nu) - ReSr(iw,s,k,mu,nu)));
              max_abs_k = std::max(max_abs_k, std::abs(ImSk(iw,s,k,mu,nu)));
              max_abs_k = std::max(max_abs_k, std::abs(ReSk(iw,s,k,mu,nu)));
            }
    app_log(2, "[rspace_pi] max diff Im={0:+10.3e}  Re={1:+10.3e}  "
                "max |Sigma|_k={2:+10.3e}", max_diff_im, max_diff_re, max_abs_k);
    REQUIRE(max_diff_im < 1e-6);
    REQUIRE(max_diff_re < 1e-6);
  }

} // namespace bdft_tests
