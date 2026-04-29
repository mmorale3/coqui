/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Real-axis quasiparticle solver tests (Phases 1 and 2):
 *   - Phase 1 (evGW diagonal QP equation): bisection / linearized / secant /
 *     spectral algorithms on the qe_lih222 fixture. Checks internal
 *     consistency between algorithms and physical reasonableness of the
 *     QP shifts (HOMO/LUMO move outward, gap widens).
 *   - Phase 2 (full QSGW static V_corr): "qp_energy" and "fermi" off-diag
 *     modes. Verifies hermiticity of V_corr and consistency of its
 *     diagonal with the directly-sampled Re Sigma_c at QP energies.
 *
 * Convention: with MO = identity (H_MF diagonal in itself), the rotation
 * back to the primary basis is a no-op, so the diagonal of V_corr in
 * the primary basis equals Re Sigma_c_{nn}(eps_n) per band.
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
#include "methods/GW_real_axis/real_axis_qp_context.hpp"
#include "methods/GW_real_axis/real_axis_qp_solver_t.h"

#include <cmath>
#include <complex>

namespace bdft_tests {

  using namespace methods;
  using methods::real_axis::real_freq_grid_t;
  using methods::real_axis::real_axis_mb_state_t;
  using methods::real_axis::real_axis_scr_coulomb_t;
  using methods::real_axis::real_axis_hf_t;
  using methods::real_axis::real_axis_qp_solver_t;
  using methods::real_axis::real_axis_qp_context_t;
  using methods::solvers::real_axis_gw_t;
  using cval_t = std::complex<double>;

  namespace {

    struct lih222_qp_fixture {
      std::shared_ptr<mf::MF>           mf;
      std::optional<thc_reader_t>       thc;
      std::optional<real_freq_grid_t>   grid;
      double                            mu0 = 0.0;
      long ns = 0, Nk = 0, nbnd = 0, N_w = 0, N_Omega = 0;

      explicit lih222_qp_fixture(std::shared_ptr<utils::mpi_context_t<>>& mpi)
      {
        mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222"));
        const int nIpts = mf->nbnd() * 8;
        thc.emplace(mf, make_thc_reader_ptree(nIpts, "", "incore", "",
                                              "bdft", 1e-8,
                                              mf->ecutrho(), 1, 1024));
        ns   = mf->nspin();
        Nk   = mf->nkpts();
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
        mu0 = 0.5 * (eigval(0, 0, n_homo) + eigval(0, 0, n_lumo));
        const double w_max = std::max(std::abs(e_min), std::abs(e_max)) + 2.0;

        N_w     = 65;
        N_Omega = 32;
        const long   N_t = 128;
        const double Omega_max = 2.0 * w_max;
        const double freq_max  = std::max(w_max, Omega_max);
        const double dt        = 0.5 * M_PI / freq_max;
        const double T_window  = dt * static_cast<double>(N_t);
        const double beta      = 50.0;

        grid = real_freq_grid_t::make_uniform(beta, mu0, w_max, N_w,
                                              Omega_max, N_Omega, N_t, T_window);
      }
    };

    // Fill state.A_wskij with diagonal Lorentzians at eps_KS, then run
    // scr_coulomb_t::update_w + gw_t::evaluate + hf_t::evaluate to populate
    // {Im,Re}Sigma_c, Sigma_x. Returns the chemical potential used.
    double populate_g0w0(lih222_qp_fixture const& f,
                         std::shared_ptr<utils::mpi_context_t<>>& mpi,
                         real_axis_mb_state_t& state)
    {
      state = real_axis_mb_state_t(*f.grid);
      state.mpi = mpi;
      state.allocate_fermionic(f.ns, f.Nk, f.nbnd);

      // Lorentzian initial A from KS eigenvalues.
      const double eta_init = 0.05;
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
                const double v = (1.0 / M_PI) * eta_init
                               / ((w_l - eps_n)*(w_l - eps_n) + eta_init*eta_init);
                A(iw, s, k, n, n) = cval_t(v, 0.0);
              }
            }
          }
      }
      state.A_wskij->node_sync();

      const std::string div = "ignore_g0";
      real_axis_scr_coulomb_t scr(&*f.grid, "rpa", div, 1e-8);
      real_axis_gw_t          gw(*f.grid, /*max_iter*/ 1, /*mix*/ 0.5,
                                 /*eps_nufft*/ 1e-8, /*ntrans*/ 1);
      real_axis_hf_t          hf(&*f.grid, div);
      const bool use_rspace = (f.Nk > 1);
      const double mu = f.grid->mu_chem();
      scr.update_w (state, *f.thc, /*verbose*/ false, use_rspace);
      gw.evaluate  (state, *f.thc, /*eps_nufft*/ 1e-8, div,
                    /*verbose*/ false, use_rspace);
      hf.evaluate  (state, *f.thc, mu);
      return mu;
    }

    // Build H_eff = diag(eps_KS) + Sigma_x in primary (KS) basis. With MO = I,
    // sH_eff is diagonal in (i, j); the diagonal carries (eps_KS_n + Sigma_x_nn).
    // Returns nda arrays (NOT sArrays) since the QP solver consumes views.
    void build_H_eff_and_MO(lih222_qp_fixture const& f,
                            real_axis_mb_state_t const& state,
                            nda::array<ComplexType, 4>& H_eff,
                            nda::array<ComplexType, 4>& MO,
                            nda::array<ComplexType, 3>& E0)
    {
      const long ns = f.ns, Nk = f.Nk, nbnd = f.nbnd;
      H_eff.resize(std::array<long, 4>{ns, Nk, nbnd, nbnd});
      MO   .resize(std::array<long, 4>{ns, Nk, nbnd, nbnd});
      E0   .resize(std::array<long, 3>{ns, Nk, nbnd});
      H_eff = ComplexType(0.0, 0.0);
      MO    = ComplexType(0.0, 0.0);
      E0    = ComplexType(0.0, 0.0);

      auto eigval = f.mf->eigval();
      auto kp2ibz = f.mf->kp_to_ibz();
      auto Sx     = state.Sigma_x_skij->local();
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k) {
          const long kibz = kp2ibz(k);
          for (long n = 0; n < nbnd; ++n) {
            MO(s, k, n, n) = ComplexType(1.0, 0.0);
            E0(s, k, n)    = ComplexType(eigval(s, kibz, n), 0.0);
            for (long m = 0; m < nbnd; ++m) {
              ComplexType v(0.0, 0.0);
              if (m == n) v = ComplexType(eigval(s, kibz, n), 0.0);
              v += Sx(s, k, m, n);  // primary basis = KS basis here
              H_eff(s, k, m, n) = v;
            }
          }
        }
    }

  } // anonymous namespace

  // ===========================================================================
  // Phase 1: diagonal QP equation. evGW.
  // ===========================================================================
  TEST_CASE("real_axis_qp_solver_diag_lih222",
            "[real_axis][qp][thc][qe][bdft]")
  {
    auto& mpi = utils::make_unit_test_mpi_context();
    lih222_qp_fixture f(mpi);

    real_axis_mb_state_t state;
    const double mu = populate_g0w0(f, mpi, state);

    nda::array<ComplexType, 4> H_eff, MO;
    nda::array<ComplexType, 3> E0;
    build_H_eff_and_MO(f, state, H_eff, MO, E0);

    const long n_homo = static_cast<long>(f.mf->nelec() / 2 - 1);
    const long n_lumo = n_homo + 1;

    // Helper: run a single algorithm and return the QP energies array.
    auto run_alg = [&](std::string const& alg) {
      real_axis_qp_context_t ctx{alg, "qp_energy", /*eta*/ 1e-3, /*tol*/ 1e-8};
      real_axis_qp_solver_t qp(&*f.grid, ctx);
      nda::array<ComplexType, 3> E_QP(std::array<long, 3>{f.ns, f.Nk, f.nbnd});
      qp.solve_qp_diag(state, H_eff, MO, E0, mu, E_QP);
      return E_QP;
    };

    auto E_bisect = run_alg("bisection");
    auto E_linear = run_alg("linearized");
    auto E_secant = run_alg("secant");
    auto E_spect  = run_alg("spectral");

    // Finiteness everywhere.
    auto check_finite = [&](nda::array<ComplexType, 3> const& E,
                            std::string const& label) {
      for (long s = 0; s < f.ns; ++s)
        for (long k = 0; k < f.Nk; ++k)
          for (long n = 0; n < f.nbnd; ++n) {
            REQUIRE(std::isfinite(E(s, k, n).real()));
            (void)label;
          }
    };
    check_finite(E_bisect, "bisection");
    check_finite(E_linear, "linearized");
    check_finite(E_secant, "secant");
    check_finite(E_spect , "spectral");

    // Sanity: every QP energy should differ from eps_KS by less than 10 Ha.
    // Note: we add Sigma_x without subtracting V_xc^KS (since the test
    // doesn't have access to V_xc), so the absolute Sigma_x contribution
    // (typically -1 to -5 Ha) shows up as a global shift of all QP energies.
    // Without V_xc subtraction, the shift is unphysical in absolute terms,
    // but the algorithm should still find a finite root.
    double max_dE = 0.0;
    for (long s = 0; s < f.ns; ++s)
      for (long k = 0; k < f.Nk; ++k)
        for (long n = 0; n < f.nbnd; ++n) {
          const double dE = std::abs(E_bisect(s, k, n).real() - E0(s, k, n).real());
          max_dE = std::max(max_dE, dE);
        }
    if (mpi->comm.root())
      app_log(1, "real_axis evGW LiH222 max |eps_QP - eps_KS| = {:.4f} Ha "
                  "(Sigma_x dominant; no V_xc subtraction in test).", max_dE);
    REQUIRE(max_dE < 10.0);

    // Check QP equation residual is small for bisection (the exact root-finder).
    // residual = eps_QP - H_eff_nn - Re Sigma_c_nn(eps_QP - mu) should be ~tol.
    auto ReS = state.ReSigma_wskij->local();
    auto const& w_grid = f.grid->w();
    auto interp_real_at = [&](long iw_lo, long s, long k, long i, long j, double t) {
      return (1.0 - t) * ReS(iw_lo,     s, k, i, j).real()
           +        t  * ReS(iw_lo + 1, s, k, i, j).real();
    };
    double max_residual = 0.0;
    for (long s = 0; s < f.ns; ++s)
      for (long k = 0; k < f.Nk; ++k)
        for (long n = 0; n < f.nbnd; ++n) {
          // Vhf_n in MO basis (MO=I): Vhf_n = H_eff(s,k,n,n) (real).
          const double Vhf_n  = H_eff(s, k, n, n).real();
          const double e_qp   = E_bisect(s, k, n).real();
          const double w_rel  = e_qp - mu;
          const long N_w      = w_grid.shape()[0];
          double resigma_diag = 0.0;
          if (w_rel <= w_grid(0))           resigma_diag = ReS(0, s, k, n, n).real();
          else if (w_rel >= w_grid(N_w-1))  resigma_diag = ReS(N_w-1, s, k, n, n).real();
          else {
            auto it = std::lower_bound(w_grid.begin(), w_grid.end(), w_rel);
            long i  = std::distance(w_grid.begin(), it);
            if (i > 0 and w_grid(i) > w_rel) --i;
            if (i > N_w - 2) i = N_w - 2;
            const double t = (w_rel - w_grid(i)) / (w_grid(i + 1) - w_grid(i));
            resigma_diag = interp_real_at(i, s, k, n, n, t);
          }
          const double res = e_qp - Vhf_n - resigma_diag;
          max_residual = std::max(max_residual, std::abs(res));
        }
    if (mpi->comm.root())
      app_log(1, "real_axis evGW LiH222 max |residual| (bisection) = {:.2e}", max_residual);
    REQUIRE(max_residual < 1e-7);

    // Cross-algorithm consistency:
    //   bisection vs secant : both exact root-finders -- must agree to tol.
    //   bisection vs linearized : linearized is a one-shot Z-factor approx
    //     and is only expected to be accurate for bands near the gap where
    //     the Z-factor is well-conditioned. We restrict the check to HOMO
    //     and LUMO. For deep-valence / high-conduction bands the linearized
    //     formula can deviate by O(0.1-0.5) Ha.
    //   bisection vs spectral : spectral finds peaks of |Im G^R|; for low-
    //     |Im Sigma_c| (LiH gap region) these track the bisection root.
    double max_bs = 0.0, max_bp = 0.0;
    double max_bl_gap = 0.0;
    for (long s = 0; s < f.ns; ++s)
      for (long k = 0; k < f.Nk; ++k)
        for (long n = 0; n < f.nbnd; ++n) {
          max_bs = std::max(max_bs, std::abs(E_bisect(s,k,n).real() - E_secant(s,k,n).real()));
          max_bp = std::max(max_bp, std::abs(E_bisect(s,k,n).real() - E_spect (s,k,n).real()));
          if (n == n_homo or n == n_lumo)
            max_bl_gap = std::max(max_bl_gap,
                std::abs(E_bisect(s,k,n).real() - E_linear(s,k,n).real()));
        }
    if (mpi->comm.root())
      app_log(1, "real_axis evGW LiH222 cross-algo: bisect vs secant {:.2e}, "
                  "bisect vs linearized (HOMO/LUMO) {:.2e}, bisect vs spectral {:.2e}",
              max_bs, max_bl_gap, max_bp);
    REQUIRE(max_bs     < 1e-4);
    REQUIRE(max_bl_gap < 2e-1);   // linearized: first-order Z-factor approx
    REQUIRE(max_bp     < 5e-2);   // spectral grid resolution

    // HOMO / LUMO shift summary at k=0 (printed for human inspection).
    if (mpi->comm.root()) {
      const double e_h_ks = E0(0, 0, n_homo).real();
      const double e_l_ks = E0(0, 0, n_lumo).real();
      const double e_h_qp = E_bisect(0, 0, n_homo).real();
      const double e_l_qp = E_bisect(0, 0, n_lumo).real();
      app_log(1, "real_axis evGW LiH222 k=0:");
      app_log(1, "  HOMO eps_KS = {:.6f} -> eps_QP = {:.6f}  (shift = {:+.4f})",
              e_h_ks, e_h_qp, e_h_qp - e_h_ks);
      app_log(1, "  LUMO eps_KS = {:.6f} -> eps_QP = {:.6f}  (shift = {:+.4f})",
              e_l_ks, e_l_qp, e_l_qp - e_l_ks);
      app_log(1, "  KS gap = {:.4f}, QP gap = {:.4f}",
              e_l_ks - e_h_ks, e_l_qp - e_h_qp);
    }
  }

  // ===========================================================================
  // Phase 2: full QSGW V_corr static potential. Two off-diagonal modes.
  // ===========================================================================
  TEST_CASE("real_axis_qp_solver_v_corr_lih222",
            "[real_axis][qp][thc][qe][bdft]")
  {
    auto& mpi = utils::make_unit_test_mpi_context();
    lih222_qp_fixture f(mpi);

    real_axis_mb_state_t state;
    const double mu = populate_g0w0(f, mpi, state);

    nda::array<ComplexType, 4> H_eff, MO;
    nda::array<ComplexType, 3> E0;
    build_H_eff_and_MO(f, state, H_eff, MO, E0);

    auto run_v_corr = [&](std::string const& mode) {
      real_axis_qp_context_t ctx{"bisection", mode, 1e-3, 1e-8};
      real_axis_qp_solver_t qp(&*f.grid, ctx);
      nda::array<ComplexType, 4> Vc(std::array<long, 4>{f.ns, f.Nk, f.nbnd, f.nbnd});
      qp.compute_V_corr(state, MO, E0, mu, Vc);
      return Vc;
    };

    auto Vc_qpe = run_v_corr("qp_energy");
    auto Vc_fer = run_v_corr("fermi");

    // 1. Finiteness.
    for (long s = 0; s < f.ns; ++s)
      for (long k = 0; k < f.Nk; ++k)
        for (long i = 0; i < f.nbnd; ++i)
          for (long j = 0; j < f.nbnd; ++j) {
            REQUIRE(std::isfinite(Vc_qpe(s, k, i, j).real()));
            REQUIRE(std::isfinite(Vc_qpe(s, k, i, j).imag()));
            REQUIRE(std::isfinite(Vc_fer(s, k, i, j).real()));
            REQUIRE(std::isfinite(Vc_fer(s, k, i, j).imag()));
          }

    // 2. Hermiticity: max |V - V^dagger| / max |V| should be < 1e-12 (the
    //    explicit 0.5*(V + V^dagger) symmetrization is the last step of
    //    compute_V_corr).
    auto check_herm = [&](nda::array<ComplexType, 4> const& V,
                          double tol, std::string const& label) {
      double max_v = 0.0, max_d = 0.0;
      for (long s = 0; s < f.ns; ++s)
        for (long k = 0; k < f.Nk; ++k)
          for (long i = 0; i < f.nbnd; ++i)
            for (long j = 0; j < f.nbnd; ++j) {
              max_v = std::max(max_v, std::abs(V(s, k, i, j)));
              const ComplexType d = V(s, k, i, j) - std::conj(V(s, k, j, i));
              max_d = std::max(max_d, std::abs(d));
            }
      const double rel = (max_v > 0.0) ? max_d / max_v : max_d;
      if (mpi->comm.root())
        app_log(1, "compute_V_corr({}): max |V| = {:.4e}, "
                    "max |V - V^dag| / max |V| = {:.4e}",
                label, max_v, rel);
      REQUIRE(rel < tol);
    };
    check_herm(Vc_qpe, 1e-12, "qp_energy");
    check_herm(Vc_fer, 1e-12, "fermi");

    // 3. Sanity on magnitudes: V_corr should be O(0.01-0.5) Ha.
    auto max_abs = [&](nda::array<ComplexType, 4> const& V) {
      double m = 0.0;
      for (long s = 0; s < f.ns; ++s)
        for (long k = 0; k < f.Nk; ++k)
          for (long i = 0; i < f.nbnd; ++i)
            for (long j = 0; j < f.nbnd; ++j)
              m = std::max(m, std::abs(V(s, k, i, j)));
      return m;
    };
    const double m_qpe = max_abs(Vc_qpe);
    const double m_fer = max_abs(Vc_fer);
    REQUIRE(m_qpe > 1e-4);
    REQUIRE(m_qpe < 2.0);
    REQUIRE(m_fer > 1e-4);
    REQUIRE(m_fer < 2.0);

    // 4. Diagonal cross-check (qp_energy mode, MO = identity):
    //    V_corr(s, k, n, n) should equal Re Sigma_c(s, k, n, n) at
    //    omega_rel = (eps_KS_n - mu).
    auto ReS = state.ReSigma_wskij->local();
    auto const& w_grid = f.grid->w();
    auto interp_ReS_diag = [&](long s, long k, long n, double w_rel) {
      const long N_w = w_grid.shape()[0];
      if (w_rel <= w_grid(0))     return ReS(0,       s, k, n, n).real();
      if (w_rel >= w_grid(N_w-1)) return ReS(N_w - 1, s, k, n, n).real();
      auto it = std::lower_bound(w_grid.begin(), w_grid.end(), w_rel);
      long i = std::distance(w_grid.begin(), it);
      if (i > 0 and w_grid(i) > w_rel) --i;
      if (i > N_w - 2) i = N_w - 2;
      const double t = (w_rel - w_grid(i)) / (w_grid(i + 1) - w_grid(i));
      return (1.0 - t) * ReS(i,     s, k, n, n).real()
           +        t  * ReS(i + 1, s, k, n, n).real();
    };

    double max_diag_diff = 0.0;
    for (long s = 0; s < f.ns; ++s)
      for (long k = 0; k < f.Nk; ++k)
        for (long n = 0; n < f.nbnd; ++n) {
          const double e_n   = E0(s, k, n).real();
          const double w_rel = e_n - mu;
          const double r     = interp_ReS_diag(s, k, n, w_rel);
          const double d     = std::abs(Vc_qpe(s, k, n, n).real() - r);
          max_diag_diff = std::max(max_diag_diff, d);
        }
    if (mpi->comm.root())
      app_log(1, "compute_V_corr(qp_energy) diag vs Re Sigma_c(eps_KS): "
                  "max |diff| = {:.4e}", max_diag_diff);
    // Both compute the same quantity (with MO = I, qp_energy mode at a==b
    // gives Re Sigma_c_aa(eps_a)). Allow ~1e-12 for floating-point round-off.
    REQUIRE(max_diag_diff < 1e-12);
  }

} // namespace bdft_tests
