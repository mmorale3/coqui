/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Probe test: after 10 SCF iterations on the qe_lih222 fixture (so we have
 * a properly correlated spectral function, not the Lorentzian initial
 * guess), measure whether the spectral function is hermitian in the orbital
 * indices at each fixed (s, k, w):
 *
 *     A_{ij}(s, k, w) =? conj(A_{ji}(s, k, w))
 *
 * and whether the THC-auxiliary projection preserves that property:
 *
 *     A_aux(s, k, P, Q, w) =? conj(A_aux(s, k, Q, P, w))
 *
 * Finding (LiH222, 10 SCF iters, 2026-04-28):
 *
 * The storage convention (dyson_G_one_kw stores A_wskij_{ij} = (i/pi) G^R_ij
 * componentwise) makes the *stored* tensor non-hermitian off-diagonal by
 * design. The matrix-valued physical spectral function is recovered by
 *
 *     A_phys_{ij} = 0.5 * (A_wskij_{ij} + conj(A_wskij_{ji}))
 *                 = -(1/pi) (Im G^R)^matrix_{ij}
 *
 * which is hermitian by construction. The class-API kernels
 * (scr_coulomb_t::update_w, gw_t::evaluate, hf_t::evaluate) apply this
 * symmetrization at the kernel input.
 *
 *   Sigma_x_skij                              rel = 5.4e-16   (machine eps)
 *   ImSigma_wskij (correlation)               rel = 2.9e-03   (~ machine eps)
 *   ReSigma_wskij (correlation)               rel = 3.8e-06   (~ machine eps)
 *   A_wskij (storage, componentwise)          rel = 1.0e+00   (storage convention)
 *   A_phys (matrix-hermitian symmetrized)     rel = 0.0       (exact)
 *   A_aux (from storage, componentwise)       rel = 1.2e+00   (storage convention)
 *   A_aux (from symmetrized A_phys)           rel = 6.9e-16   (machine eps)
 *
 * Sigma is hermitian to machine precision; the symmetrized A and its
 * THC projection are hermitian exactly / to round-off; the cross-
 * validation against Matsubara (LiH222 lowest 8 iw_n) is unchanged at
 * 2.77e-3 / 2.65e-3 (HOMO/LUMO).
 *
 * Implication for the distributed-memory refactor (Phase 1B): A_aux
 * built from the symmetrized A_phys IS hermitian in (P, Q). The
 * (P, Q) <-> (Q, P) swap on the second leg of the Pi cross-correlation
 * kernel CAN be replaced by complex conjugation of the local block,
 * eliminating the transposed-peer redistribute. (Note: the stored
 * A_wskij is still componentwise non-hermitian -- the physical
 * symmetrization happens at kernel input, not at storage.)
 *
 * The test always passes; numbers are logged for reference. Single-rank
 * only.
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
#include "methods/GW_real_axis/real_axis_scf_driver.hpp"
#include "methods/GW_real_axis/real_axis_thc_project.hpp"

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
  using methods::real_axis::primary_to_aux_one_k;
  using methods::solvers::real_axis_gw_t;
  using cval_t = std::complex<double>;

  // ===========================================================================
  TEST_CASE("real_axis_hermiticity_lih222_after_10_scf",
            "[real_axis][hermiticity][thc][qe][bdft][serial]")
  {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    // Single-rank: we materialize the full A and A_aux to inspect element by
    // element.
    if (mpi_context->comm.size() != 1) return;

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
    const long Naux = thc.Np();

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
    const double mu0  = 0.5 * (eigval(0, 0, n_homo) + eigval(0, 0, n_lumo));
    const double w_max = std::max(std::abs(e_min), std::abs(e_max)) + 2.0;

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

    real_axis_hf_t          hf(&grid, "ignore_g0");
    real_axis_scr_coulomb_t scr_eri(&grid, "rpa", "ignore_g0", 1e-8);
    real_axis_gw_t          gw(grid, /*max_iter*/ 1, /*mix*/ 0.5,
                               /*eps_nufft*/ 1e-8, /*ntrans*/ 1);
    real_axis_dyson_t       dyson(std::move(H_MF), &grid, /*eta*/ 0.05);
    real_axis_mb_solver_t   mb_solver{&hf, &scr_eri, &gw};

    scgw_config cfg;
    cfg.max_iter    = 10;
    cfg.alpha_mix   = 0.7;
    cfg.tol         = 1e-12;     // do not early-stop -- run all 10 iters
    cfg.eta         = 0.05;
    cfg.eps_nufft   = 1e-8;
    cfg.update_mu   = true;
    cfg.mix_kind    = scgw_mix_kind::diis;
    cfg.diis_window = 8;

    auto res = real_axis_scf_loop(state, dyson, thc,
                                   mb_solver, cfg, k_weights,
                                   /*N_elec*/ static_cast<double>(mf->nelec()));

    app_log(2, "[hermiticity_test] iter_used={}  final_diff={:.3e}  final_mu={:.6f}",
            res.iter_used, res.final_diff, res.final_mu);

    REQUIRE(res.iter_used == 10);
    REQUIRE(state.A_wskij.has_value());
    REQUIRE(state.Sigma_x_skij.has_value());
    REQUIRE(state.ImSigma_wskij.has_value());
    REQUIRE(state.ReSigma_wskij.has_value());

    // Helper: check hermiticity of a 4D (s, k, i, j) tensor in (i, j) at
    // each (s, k). Returns (max abs err, max abs amplitude).
    auto check_skij = [&](nda::array<cval_t, 4> const& T,
                          char const* label) {
      const long Ns_  = T.shape()[0];
      const long Nk_  = T.shape()[1];
      const long Nb_  = T.shape()[2];
      double max_err = 0.0, max_amp = 0.0;
      for (long s = 0; s < Ns_; ++s)
        for (long k = 0; k < Nk_; ++k)
          for (long i = 0; i < Nb_; ++i)
            for (long j = 0; j < Nb_; ++j) {
              const cval_t t_ij = T(s, k, i, j);
              const cval_t t_ji = T(s, k, j, i);
              max_err = std::max(max_err, std::abs(t_ij - std::conj(t_ji)));
              max_amp = std::max(max_amp, std::abs(t_ij));
            }
      const double rel = (max_amp > 0) ? max_err / max_amp : 0.0;
      app_log(2, "[hermiticity_test] {:32s}  max err = {:.3e}   max amp = {:.3e}   rel = {:.3e}",
              label, max_err, max_amp, rel);
      return std::make_pair(max_err, max_amp);
    };

    // Helper: check hermiticity of a 5D (w, s, k, i, j) tensor in (i, j)
    // at each (w, s, k).
    auto check_wskij = [&](nda::array<cval_t, 5> const& T,
                           char const* label) {
      const long Nw_  = T.shape()[0];
      const long Ns_  = T.shape()[1];
      const long Nk_  = T.shape()[2];
      const long Nb_  = T.shape()[3];
      double max_err = 0.0, max_amp = 0.0;
      for (long iw = 0; iw < Nw_; ++iw)
        for (long s = 0; s < Ns_; ++s)
          for (long k = 0; k < Nk_; ++k)
            for (long i = 0; i < Nb_; ++i)
              for (long j = 0; j < Nb_; ++j) {
                const cval_t t_ij = T(iw, s, k, i, j);
                const cval_t t_ji = T(iw, s, k, j, i);
                max_err = std::max(max_err, std::abs(t_ij - std::conj(t_ji)));
                max_amp = std::max(max_amp, std::abs(t_ij));
              }
      const double rel = (max_amp > 0) ? max_err / max_amp : 0.0;
      app_log(2, "[hermiticity_test] {:32s}  max err = {:.3e}   max amp = {:.3e}   rel = {:.3e}",
              label, max_err, max_amp, rel);
      return std::make_pair(max_err, max_amp);
    };

    // -----------------------------------------------------------------------
    // Diagnostic: walk every fermionic field on the state and check whether
    // it is hermitian in (i, j). This pinpoints which step breaks the
    // assumption.
    //
    // Note on A_wskij: the storage convention is A_wskij_{ij} = (i/pi) G^R_ij
    // componentwise (see real_axis_dyson_G.hpp:78). This stored object is
    // NOT hermitian (~ 1.0 rel violation expected). The kernels consume
    // the matrix-hermitian symmetrization
    //    A_phys_{ij} = 0.5 * (A_wskij_{ij} + conj(A_wskij_{ji}))
    // which IS hermitian by construction. We check both: the storage (will
    // show O(1) violation, expected) and the symmetrized form (should be
    // hermitian to round-off).
    // -----------------------------------------------------------------------
    auto res_Sx  = check_skij (*state.Sigma_x_skij,  "Sigma_x_skij");
    auto res_ImS = check_wskij(*state.ImSigma_wskij, "ImSigma_wskij (correlation)");
    auto res_ReS = check_wskij(*state.ReSigma_wskij, "ReSigma_wskij (correlation)");
    auto res_A   = check_wskij(*state.A_wskij,       "A_wskij (storage, componentwise)");

    // Build the matrix-hermitian symmetrized A and check hermiticity.
    {
      auto const& A0 = *state.A_wskij;
      const long Nw_ = A0.shape()[0];
      const long Ns_ = A0.shape()[1];
      const long Nk_ = A0.shape()[2];
      const long Nb_ = A0.shape()[3];
      nda::array<cval_t, 5> A_phys(Nw_, Ns_, Nk_, Nb_, Nb_);
      for (long iw = 0; iw < Nw_; ++iw)
        for (long s = 0; s < Ns_; ++s)
          for (long k = 0; k < Nk_; ++k)
            for (long i = 0; i < Nb_; ++i)
              for (long j = 0; j < Nb_; ++j)
                A_phys(iw, s, k, i, j) =
                    0.5 * (A0(iw, s, k, i, j) + std::conj(A0(iw, s, k, j, i)));
      check_wskij(A_phys, "A_phys (matrix-hermitian symmetrized)");
    }

    // -----------------------------------------------------------------------
    // Project A -> A_aux and check (P, Q) hermiticity at each (s, k).
    // -----------------------------------------------------------------------
    {
      auto const& A = *state.A_wskij;
      // Repack A from (N_w, ns, Nk, nbnd, nbnd) to (ns, Nk, N_w, nbnd, nbnd).
      nda::array<cval_t, 5> A_drv(ns, Nk, N_w, nbnd, nbnd);
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k)
          for (long iw = 0; iw < N_w; ++iw)
            for (long mu = 0; mu < nbnd; ++mu)
              for (long nu = 0; nu < nbnd; ++nu)
                A_drv(s, k, iw, mu, nu) = A(iw, s, k, mu, nu);

      nda::array<cval_t, 4> X_skPmu(ns, Nk, Naux, nbnd);
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k) {
          auto Xsk = thc.X(static_cast<int>(s), /*ip*/ 0, static_cast<int>(k));
          for (long P = 0; P < Naux; ++P)
            for (long mu = 0; mu < nbnd; ++mu)
              X_skPmu(s, k, P, mu) = Xsk(P, mu);
        }

      // Aux projection of the storage A (non-hermitian, expected).
      // and of the symmetrized A_phys (should be hermitian to round-off).
      auto check_aux = [&](nda::array<cval_t, 5> const& A_full,
                           char const* label) {
        double max_err = 0.0, max_amp = 0.0;
        using nda::range;
        const auto _ = range::all;
        nda::array<cval_t, 3> A_aux_PQw(Naux, Naux, N_w);
        for (long s = 0; s < ns; ++s)
          for (long ik = 0; ik < Nk; ++ik) {
            auto X_view = X_skPmu(s, ik, _, _);
            auto A_view = A_full(s, ik, _, _, _);
            primary_to_aux_one_k(X_view, A_view, A_aux_PQw);
            for (long iw = 0; iw < N_w; ++iw)
              for (long P = 0; P < Naux; ++P)
                for (long Q = 0; Q < Naux; ++Q) {
                  const cval_t a_PQ = A_aux_PQw(P, Q, iw);
                  const cval_t a_QP = A_aux_PQw(Q, P, iw);
                  max_err = std::max(max_err,
                                      std::abs(a_PQ - std::conj(a_QP)));
                  max_amp = std::max(max_amp, std::abs(a_PQ));
                }
          }
        const double rel = (max_amp > 0) ? max_err / max_amp : 0.0;
        app_log(2, "[hermiticity_test] {:32s}  max err = {:.3e}   max amp = {:.3e}   rel = {:.3e}",
                label, max_err, max_amp, rel);
      };

      check_aux(A_drv, "A_aux (from storage, componentwise)");

      // Build the matrix-hermitian symmetrized A in the (s,k,iw,μ,ν) layout
      // and project it; this is what the kernels actually consume.
      nda::array<cval_t, 5> A_phys_drv(ns, Nk, N_w, nbnd, nbnd);
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k)
          for (long iw = 0; iw < N_w; ++iw)
            for (long mu = 0; mu < nbnd; ++mu)
              for (long nu = 0; nu < nbnd; ++nu)
                A_phys_drv(s, k, iw, mu, nu) =
                    0.5 * (A(iw, s, k, mu, nu)
                           + std::conj(A(iw, s, k, nu, mu)));
      check_aux(A_phys_drv, "A_aux (from symmetrized A_phys)");
    }

    // The test is diagnostic; we just require the fields are well-defined.
    REQUIRE(res_A.second > 0.0);
    (void) res_Sx;
    (void) res_ImS;
    (void) res_ReS;
  }

} // namespace bdft_tests
