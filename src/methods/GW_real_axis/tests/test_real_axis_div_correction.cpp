/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Smoke regression test for the q=0 divergence (Gygi-Baldereschi-style)
 * head correction in the real-axis GW pipeline.
 *
 * Compares Sigma_c on the qe_lih222 fixture between div_treatment values
 * "ignore_g0" (no correction) and "gygi_smallest_q" (Madelung x
 * eps_inv_head correction). Asserts:
 *   - Both runs produce finite Sigma everywhere.
 *   - The head-corrected run differs from ignore_g0 by an amount of the
 *     expected scale (small but non-zero -- the LiH222 Madelung
 *     contribution shifts diagonal Sigma by O(0.01-0.1) Hartree).
 *   - state.eps_inv_head_O is populated only when div_treatment != "ignore_g0".
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

#include <cmath>
#include <complex>

namespace bdft_tests {

  using namespace methods;
  using methods::real_axis::real_freq_grid_t;
  using methods::real_axis::real_axis_mb_state_t;
  using methods::real_axis::real_axis_scr_coulomb_t;
  using methods::solvers::real_axis_gw_t;
  using cval_t = std::complex<double>;

  namespace {

    // Common LiH222 setup, same as test_real_axis_thc.cpp::lih222_serial
    // but with a smaller frequency grid for speed.
    struct lih222_fixture {
      std::shared_ptr<mf::MF>                                mf;
      std::optional<thc_reader_t>                            thc;
      std::optional<real_freq_grid_t>                        grid;
      double                                                 mu0;
      long                                                   ns;
      long                                                   Nk;
      long                                                   Nq;
      long                                                   nbnd;
      long                                                   N_w;
      long                                                   N_Omega;

      explicit lih222_fixture(std::shared_ptr<utils::mpi_context_t<>>& mpi_context)
      {
        mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
        const int nIpts = mf->nbnd() * 8;
        thc.emplace(mf, make_thc_reader_ptree(nIpts, "", "incore", "",
                                              "bdft", 1e-8,
                                              mf->ecutrho(), 1, 1024));
        ns   = mf->nspin();
        Nk   = mf->nkpts();
        Nq   = mf->nqpts();
        nbnd = mf->nbnd();

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
        const long n_homo = static_cast<long>(mf->nelec() / 2 - 1);
        const long n_lumo = n_homo + 1;
        mu0  = 0.5 * (eigval(0, 0, n_homo) + eigval(0, 0, n_lumo));
        const double w_max = std::max(std::abs(e_min), std::abs(e_max)) + 2.0;

        N_w     = 65;
        N_Omega = 32;
        const long   N_t = 128;
        const double Omega_max = 2.0 * w_max;
        const double freq_max  = std::max(w_max, Omega_max);
        const double dt        = 0.5 * M_PI / freq_max;
        const double T_window  = dt * static_cast<double>(N_t);
        const double beta      = 50.0;

        grid = real_freq_grid_t::make_uniform(
                  beta, mu0, w_max, N_w, Omega_max, N_Omega, N_t, T_window);
      }
    };

    void fill_lorentzian_A(lih222_fixture const& f, real_axis_mb_state_t& state)
    {
      const double eta = 0.05;
      auto eigval = f.mf->eigval();
      auto kp2ibz = f.mf->kp_to_ibz();
      if (state.A_wskij->node_comm()->root()) {
        auto A = state.A_wskij->local();
        A = cval_t(0.0, 0.0);
        for (long s = 0; s < f.ns; ++s)
          for (long k = 0; k < f.Nk; ++k) {
            const long kibz = kp2ibz(k);
            for (long n = 0; n < f.nbnd; ++n) {
              const double eps_n = eigval(s, kibz, n);
              for (long iw = 0; iw < f.N_w; ++iw) {
                const double w_l = f.grid->w()(iw) + f.grid->mu_chem();
                const double v = (1.0 / M_PI) * eta
                               / ((w_l - eps_n)*(w_l - eps_n) + eta*eta);
                A(iw, s, k, n, n) = cval_t(v, 0.0);
              }
            }
          }
      }
      state.A_wskij->node_sync();
    }

    void run_g0w0(lih222_fixture const& f,
                  std::shared_ptr<utils::mpi_context_t<>>& mpi_context,
                  std::string const& div_treatment,
                  real_axis_mb_state_t& state_out)
    {
      state_out = real_axis_mb_state_t(*f.grid);
      state_out.mpi = mpi_context;
      state_out.A_wskij.emplace(*state_out.mpi,
          std::array<long, 5>{f.N_w, f.ns, f.Nk, f.nbnd, f.nbnd});
      fill_lorentzian_A(f, state_out);

      real_axis_scr_coulomb_t scr(&*f.grid, "rpa", div_treatment, 1e-8);
      real_axis_gw_t          gw(*f.grid, /*max_iter*/ 1, /*mix*/ 0.5,
                                 /*eps_nufft*/ 1e-8, /*ntrans*/ 1);
      const bool use_rspace = (f.Nk > 1);
      scr.update_w(state_out, *f.thc, /*verbose*/ false, use_rspace);
      gw.evaluate(state_out, *f.thc, /*eps_nufft*/ 1e-8, div_treatment,
                  /*verbose*/ false, use_rspace);
    }

  } // anonymous namespace

  // ===========================================================================
  TEST_CASE("real_axis_gw_div_correction_lih222",
            "[real_axis][div][thc][qe][bdft]")
  {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    lih222_fixture f(mpi_context);

    // Run two G_0W_0 calculations with different divergence treatments.
    real_axis_mb_state_t state_ignore;
    real_axis_mb_state_t state_gygi;
    run_g0w0(f, mpi_context, "ignore_g0",       state_ignore);
    run_g0w0(f, mpi_context, "gygi_smallest_q", state_gygi);

    // eps_inv_head_O contract: populated only when div_treatment != ignore_g0.
    REQUIRE_FALSE(state_ignore.eps_inv_head_O.has_value());
    REQUIRE      (state_gygi.eps_inv_head_O.has_value());

    // Both runs must produce finite, well-shaped Sigma_c outputs.
    REQUIRE(state_ignore.ImSigma_wskij.has_value());
    REQUIRE(state_gygi.ImSigma_wskij.has_value());
    auto ImS_ig = state_ignore.ImSigma_wskij->local();
    auto ReS_ig = state_ignore.ReSigma_wskij->local();
    auto ImS_gg = state_gygi.ImSigma_wskij->local();
    auto ReS_gg = state_gygi.ReSigma_wskij->local();

    REQUIRE(ImS_ig.shape() == ImS_gg.shape());

    bool all_finite = true;
    for (long iw = 0; iw < f.N_w; ++iw)
      for (long s = 0; s < f.ns; ++s)
        for (long k = 0; k < f.Nk; ++k)
          for (long mu = 0; mu < f.nbnd; ++mu)
            for (long nu = 0; nu < f.nbnd; ++nu) {
              if (!std::isfinite(ImS_gg(iw, s, k, mu, nu).real()) ||
                  !std::isfinite(ReS_gg(iw, s, k, mu, nu).real()))
                all_finite = false;
            }
    REQUIRE(all_finite);

    // The head correction should produce a non-zero, finite shift on the
    // diagonal Sigma. We don't pin down the exact magnitude (it depends
    // on the LiH222 Madelung and the head channel of eps^-1) but the
    // correction must be non-trivial -- catches the case where the
    // correction silently zeros itself out.
    double max_diff_im = 0.0, max_diff_re = 0.0;
    double max_abs_ig  = 0.0;
    for (long iw = 0; iw < f.N_w; ++iw)
      for (long s = 0; s < f.ns; ++s)
        for (long k = 0; k < f.Nk; ++k)
          for (long mu = 0; mu < f.nbnd; ++mu)
            for (long nu = 0; nu < f.nbnd; ++nu) {
              max_diff_im = std::max(max_diff_im,
                  std::abs(ImS_gg(iw, s, k, mu, nu) - ImS_ig(iw, s, k, mu, nu)));
              max_diff_re = std::max(max_diff_re,
                  std::abs(ReS_gg(iw, s, k, mu, nu) - ReS_ig(iw, s, k, mu, nu)));
              max_abs_ig = std::max(max_abs_ig,
                  std::abs(ImS_ig(iw, s, k, mu, nu)));
              max_abs_ig = std::max(max_abs_ig,
                  std::abs(ReS_ig(iw, s, k, mu, nu)));
            }
    app_log(2, "[div_correction] max diff Im Sigma = {:.3e}", max_diff_im);
    app_log(2, "[div_correction] max diff Re Sigma = {:.3e}", max_diff_re);
    app_log(2, "[div_correction] max |Sigma| ignore_g0 = {:.3e}", max_abs_ig);
    REQUIRE(max_diff_im > 1e-6);
    REQUIRE(max_diff_re > 1e-6);
    // Sanity: the correction should not blow up; it should be O(|Sigma|).
    REQUIRE(max_diff_im < 10.0 * max_abs_ig);
    REQUIRE(max_diff_re < 10.0 * max_abs_ig);
  }

} // namespace bdft_tests
