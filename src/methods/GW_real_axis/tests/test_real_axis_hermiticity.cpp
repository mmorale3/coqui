/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * RW-1 ADAPTED PORT of origin/real_axis's test_real_axis_hermiticity.cpp.
 *
 * DEVIATION FROM THE BRANCH (flagged, notes/rw1_port_report.md): the branch
 * version obtains its spectral function from TEN real-axis scGW iterations
 * (real_axis_scf_loop + real_axis_gw_t + real_axis_hf_t + real_axis_dyson_t),
 * none of which are in the ported leaf slice. Reproducing it verbatim would
 * mean porting the whole Sigma/SCF half of the module, which RW-1 explicitly
 * excludes. What is preserved here is the IDENTITY THE PI KERNEL DEPENDS ON --
 * the one the branch test exists to certify (real_axis_pi.hpp:70-79: the second
 * leg of the bare bubble is taken as conj of the LOCAL (P,Q) block instead of
 * the transposed peer, which is legal only if the aux-projected spectral
 * function is hermitian in (P,Q)):
 *
 *     A_phys_{ij}   = 1/2 (A_{ij} + conj(A_{ji}))          hermitian by construction
 *     A_aux_{PQ}    = sum_{mu,nu} X_{P mu} A_phys_{mu nu} conj(X_{Q nu})
 *                   =? conj(A_aux_{QP})                    must hold to round-off
 *
 * The spectral function here is a QP-pole Lorentzian sum through a NON-TRIVIAL
 * unitary MO rotation (so A is not diagonal and the (P,Q) check has content),
 * plus a deliberately injected anti-hermitian component that reproduces the
 * branch's "storage convention is componentwise non-hermitian" situation:
 * A_store = A_phys + iD with D real symmetric is O(1) non-hermitian
 * componentwise, and the kernel-input symmetrization must remove it exactly.
 *
 * Branch reference numbers (LiH222, 10 SCF iters, 2026-04-28) for context:
 *   A_wskij (storage, componentwise)          rel = 1.0e+00   (storage convention)
 *   A_phys  (matrix-hermitian symmetrized)    rel = 0.0       (exact)
 *   A_aux   (from storage, componentwise)     rel = 1.2e+00   (storage convention)
 *   A_aux   (from symmetrized A_phys)         rel = 6.9e-16   (machine eps)
 *
 * Single-rank only; numbers are logged.
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
#include "methods/GW_real_axis/real_axis_thc_project.hpp"

#include <cmath>
#include <complex>

namespace bdft_tests {

  using namespace methods;
  using methods::real_axis::real_freq_grid_t;
  using methods::real_axis::primary_to_aux_one_k;
  using cval_t = std::complex<double>;

  // ===========================================================================
  TEST_CASE("real_axis_hermiticity_lih222_qp_poles",
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
    const long nbnd = mf->nbnd();
    const long Naux = thc.Np();

    REQUIRE(Nk > 1);

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
    const double mu0   = 0.5 * (eigval(0, 0, n_homo) + eigval(0, 0, n_homo + 1));
    const double w_max = std::max(std::abs(e_min), std::abs(e_max)) + 2.0;

    const long   N_w       = 33;
    const long   N_Omega   = 16;
    const long   N_t       = 64;
    const double Omega_max = 2.0 * w_max;
    const double freq_max  = std::max(w_max, Omega_max);
    const double dt        = 0.5 * M_PI / freq_max;
    const double T_window  = dt * static_cast<double>(N_t);
    const double beta      = 50.0;

    auto grid = real_freq_grid_t::make_uniform(
                  beta, mu0, w_max, N_w, Omega_max, N_Omega, N_t, T_window);

    // ---------------------------------------------------------------------
    // A deterministic non-trivial unitary MO rotation per (s,k): a chain of
    // Givens rotations with an extra phase, so A_ij is dense and complex.
    // ---------------------------------------------------------------------
    nda::array<cval_t, 4> MO(ns, Nk, nbnd, nbnd);
    MO() = cval_t(0.0, 0.0);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k) {
        for (long i = 0; i < nbnd; ++i) MO(s, k, i, i) = cval_t(1.0, 0.0);
        for (long n = 0; n + 1 < nbnd; ++n) {
          const double th = 0.11 * static_cast<double>(n + 1)
                          + 0.07 * static_cast<double>(k + 1);
          const double ph = 0.23 * static_cast<double>(n + 1);
          const double c = std::cos(th), sn = std::sin(th);
          const cval_t eip = std::exp(cval_t(0.0, ph));
          for (long i = 0; i < nbnd; ++i) {
            const cval_t a = MO(s, k, i, n), b = MO(s, k, i, n + 1);
            MO(s, k, i, n)     =  c * a + sn * eip * b;
            MO(s, k, i, n + 1) = -sn * std::conj(eip) * a + c * b;
          }
        }
      }

    // QP-pole spectral function through those orbitals (the
    // build_A_from_QP_poles recipe, real_axis_qp_scf_driver.hpp:249-278).
    const double eta = 0.05;
    nda::array<cval_t, 5> A_skwij(ns, Nk, N_w, nbnd, nbnd);
    A_skwij() = cval_t(0.0, 0.0);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k) {
        const long kibz = kp2ibz(k);
        for (long iw = 0; iw < N_w; ++iw) {
          const double w_abs = grid.w()(iw) + grid.mu_chem();
          for (long n = 0; n < nbnd; ++n) {
            const double e_n = eigval(s, kibz, n);
            const double w_n = (1.0 / M_PI) * eta
                             / ((w_abs - e_n) * (w_abs - e_n) + eta * eta);
            for (long i = 0; i < nbnd; ++i)
              for (long j = 0; j < nbnd; ++j)
                A_skwij(s, k, iw, i, j) +=
                    cval_t(w_n, 0.0) * MO(s, k, i, n) * std::conj(MO(s, k, j, n));
          }
        }
      }

    // "Storage convention" surrogate: add i * (real symmetric) so the stored
    // tensor is O(1) non-hermitian componentwise, exactly like the branch's
    // A_wskij = (i/pi) G^R storage.
    nda::array<cval_t, 5> A_store(A_skwij);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k)
        for (long iw = 0; iw < N_w; ++iw)
          for (long i = 0; i < nbnd; ++i)
            for (long j = 0; j < nbnd; ++j) {
              const double d = 0.5 * (A_skwij(s, k, iw, i, j).real()
                                    + A_skwij(s, k, iw, j, i).real());
              A_store(s, k, iw, i, j) += cval_t(0.0, d);
            }

    auto check_herm5 = [&](nda::array<cval_t, 5> const& T, char const* label) {
      double max_err = 0.0, max_amp = 0.0;
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k)
          for (long iw = 0; iw < N_w; ++iw)
            for (long i = 0; i < nbnd; ++i)
              for (long j = 0; j < nbnd; ++j) {
                max_err = std::max(max_err, std::abs(T(s, k, iw, i, j)
                                        - std::conj(T(s, k, iw, j, i))));
                max_amp = std::max(max_amp, std::abs(T(s, k, iw, i, j)));
              }
      const double rel = (max_amp > 0) ? max_err / max_amp : 0.0;
      app_log(2, "[hermiticity_test] {:34s}  max err = {:.3e}   max amp = {:.3e}   rel = {:.3e}",
              label, max_err, max_amp, rel);
      return rel;
    };

    const double rel_store = check_herm5(A_store, "A storage (componentwise)");
    const double rel_phys  = check_herm5(A_skwij, "A_phys (matrix-hermitian)");
    // The storage surrogate must actually be non-hermitian (positive control),
    // and A_phys must be hermitian to round-off.
    REQUIRE(rel_store > 1e-2);
    REQUIRE(rel_phys  < 1e-12);

    // ---------------------------------------------------------------------
    // Project A -> A_aux and check (P, Q) hermiticity at each (s, k).
    // ---------------------------------------------------------------------
    nda::array<cval_t, 4> X_skPmu(ns, Nk, Naux, nbnd);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k) {
        auto Xsk = thc.X(static_cast<int>(s), /*ip*/ 0, static_cast<int>(k));
        for (long P = 0; P < Naux; ++P)
          for (long mu = 0; mu < nbnd; ++mu)
            X_skPmu(s, k, P, mu) = Xsk(P, mu);
      }

    auto check_aux = [&](nda::array<cval_t, 5> const& A_full, char const* label) {
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
                max_err = std::max(max_err, std::abs(A_aux_PQw(P, Q, iw)
                                        - std::conj(A_aux_PQw(Q, P, iw))));
                max_amp = std::max(max_amp, std::abs(A_aux_PQw(P, Q, iw)));
              }
        }
      const double rel = (max_amp > 0) ? max_err / max_amp : 0.0;
      app_log(2, "[hermiticity_test] {:34s}  max err = {:.3e}   max amp = {:.3e}   rel = {:.3e}",
              label, max_err, max_amp, rel);
      return rel;
    };

    const double rel_aux_store = check_aux(A_store,  "A_aux (from storage)");
    const double rel_aux_phys  = check_aux(A_skwij,  "A_aux (from symmetrized A_phys)");

    // THE LOAD-BEARING ASSERT: this is what licenses the conj-second-leg
    // shortcut in real_axis_pi.hpp:70-79.
    REQUIRE(rel_aux_phys < 1e-12);
    // Positive control: the un-symmetrized storage does NOT satisfy it.
    REQUIRE(rel_aux_store > 1e-2);
  }

} // namespace bdft_tests
