/**
 * scGW-tilde Tier 1.5, increment T15-b (notes/tier15_ward_legs_plan.md sections 5-6): the
 * WIRING of the discrete-Ward leg vertex into the ladder readout, on the qe_lih222 fixture
 * with the production pol-only attachment (vertex_type = "none", pol_vertex = "ladder").
 *
 *  ward_wiring_readout
 *    G-i fixture leg: at scf iteration 1 the stored Sigma is ZERO, so legs = "ward" must
 *    reproduce legs = "bare" BITWISE (loop energies, RPA and +ladder readouts, and the
 *    +DeltaLambda column collapses onto the RPA one). At iteration 2 Sigma != 0: the loop
 *    stays bitwise (the readout is report-only), the RPA column is bitwise, and the
 *    Lambda legs move the +ladder column and populate a finite +DeltaLambda column.
 *
 *  ward_wiring_gates (vertex_t::ward_legs_gate on the readout instance, C = all 16 bands)
 *    bub_pin      : the pair kernel's bare zero-rung bubble IS the secondary-basis RPA
 *                   bubble (machine class) -- the normalization of Delta P^Lambda;
 *    fit errors   : the aux-grid DLR pole fits of G / Sigma_c (reported; hard gate 1e-2);
 *    G-g fixture  : the vertex-traced pair propagator at Gamma, bare O(1) vs Lambda --
 *                   fit class at the full window (asserted < 1e-2 of bare, logged);
 *    pole-vs-tau  : the pole route's representation error on real data (logged);
 *    Hermiticity  : the (M,N) asymmetry before the projection (logged).
 *    A second run at C = [0, 4) reports the window-truncated G-g meter (no assertion).
 */

#undef NDEBUG

#include <cmath>
#include <limits>
#include <complex>
#include <string>
#include <tuple>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"
#include "IO/app_loggers.h"

#include "numerics/imag_axes_ft/IAFT.hpp"

#include "methods/vertex/vertex_pi.icc"

#include "mean_field/default_MF.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include "methods/vertex/vertex_t.h"
#include "methods/scr_coulomb/cvv_head.hpp"

namespace bdft_tests {

  using namespace methods;

  TEST_CASE("ward_wiring_readout", "[methods][vertex][scgwt][tier15]") {
#ifndef ENABLE_DLR
    SUCCEED("ward_wiring_readout skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_t15_wiring";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    auto run = [&](std::string const &legs, int niter) {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd());
      vtx.set_pol_vertex("ladder", "w0_prev", nda::range(0, 2), -1, 1e-8, -1.0, -1.0, -1.0);
      vtx.set_ladder_legs(legs);
      scr_eri.set_vertex(&vtx);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     niter, false, 1e-9, true);
      auto [er, el] = scr_eri.pol_eps_readout();
      const double ed = scr_eri.pol_eps_dlam();
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return std::make_tuple(e_hf, e_corr, er, el, ed);
    };

    // ---- G-i fixture leg: iteration 1 sees Sigma = 0 => ward == bare BITWISE ------------
    auto [h0, c0, r0, l0, d0] = run("bare", 1);
    auto [h1, c1, r1, l1, d1] = run("ward", 1);
    app_log(1, "ward_wiring_readout iter 1: bare e_corr {} eps RPA {} +ladder {} ; ward e_corr "
               "{} eps RPA {} +DeltaLambda {} +ladder {}", c0, r0, l0, c1, r1, d1, l1);
    REQUIRE(h1 == h0);
    REQUIRE(c1 == c0);
    REQUIRE(r1 == r0);
    // Sigma = 0 => Delta chi0 = 0 exactly, so the two readouts are the same arithmetic. They are NOT
    // bitwise reproducible across two scf_loop runs in one process, though: the eps readout wobbles
    // at the ulp level run to run on the pre-merge tree already (ba23dbf, OMP_NUM_THREADS=1, 2 ranks,
    // measured 2026-09-24: eps RPA 1.753605413982423 / ...4253 / ...4253 and +ladder ...5243 / ...526
    // / ...5267 over three runs; source not identified, upstream of the readout since e_corr is
    // stable to 17 digits). Under ctest load the bitwise form caught that wobble (2 ulp). A few ulp
    // is the gate; a routing defect would show at the size of the ladder correction (~2e-8 here).
    REQUIRE(std::abs(l1 - l0) <= 8.0 * std::numeric_limits<double>::epsilon() * std::abs(l0));
    REQUIRE(d0 == -1.0);          // the bare run reports no +DeltaLambda column
    REQUIRE(d1 == r1);            // the Lambda column collapses onto RPA (adds exact zeros)

    // ---- iteration 2: Sigma != 0 -- loop bitwise (report-only), RPA bitwise, ward moves ---
    auto [h2, c2, r2, l2, d2] = run("bare", 2);
    auto [h3, c3, r3, l3, d3] = run("ward", 2);
    app_log(1, "ward_wiring_readout iter 2: bare eps RPA {} +ladder {} (Delta {:+.6f}) ; ward "
               "+DeltaLambda {} (Delta {:+.6f}) +ladder(Lambda) {} (Delta {:+.6f})",
            r2, l2, l2 - r2, d3, d3 - r3, l3, l3 - r3);
    REQUIRE(h3 == h2);
    REQUIRE(c3 == c2);
    REQUIRE(r3 == r2);
    REQUIRE(std::isfinite(l3));
    REQUIRE(std::isfinite(d3));
    REQUIRE(l3 > 0.0);
    REQUIRE(d3 > 0.0);
    REQUIRE(l3 != l2);            // the Lambda legs changed the composite
    REQUIRE(d3 != r3);            // and chi0_Lambda alone differs from RPA
#endif
  }

  TEST_CASE("ward_wiring_gates", "[methods][vertex][scgwt][tier15]") {
#ifndef ENABLE_DLR
    SUCCEED("ward_wiring_gates skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_t15_gates";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    auto gate_at = [&](nda::range window, int niter) {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd());
      vtx.set_pol_vertex("ladder", "w0_prev", window, -1, 1e-8, -1.0, -1.0, -1.0);
      vtx.set_ladder_legs("ward");
      scr_eri.set_vertex(&vtx);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     niter, false, 1e-9, true);
      (void)e_hf; (void)e_corr;
      auto *pv = scr_eri.pol_vertex_instance();
      REQUIRE(pv != nullptr);
      auto g = pv->ward_legs_gate(mb_state, thc);
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return g;
    };

    // full band window: no truncation, so the G-g meter is pure fit class
    auto g = gate_at(nda::range(0, mf->nbnd()), 2);
    app_log(1, "ward_wiring_gates [C = all {} bands]: bub_pin {:.3e}; fit_err G {:.3e} Sigma "
               "{:.3e} (rr {:.3g} / {:.3g}); Gamma-trace bare/Lambda nu0 {:.3e}/{:.3e}, nu1 "
               "{:.3e}/{:.3e}, nu2 {:.3e}/{:.3e}; pole-vs-tau {:.3e}; asym ladder {:.3e} dlam "
               "{:.3e}; |dlam| {:.3e} |ladder| {:.3e}",
            mf->nbnd(), g.bub_pin, g.fit_err_G, g.fit_err_S, g.rr_G, g.rr_S,
            g.gamma_bare[0], g.gamma_lam[0], g.gamma_bare[1], g.gamma_lam[1],
            g.gamma_bare[2], g.gamma_lam[2], g.pole_vs_tau, g.asym_ladder, g.asym_dlam,
            g.dlam_max, g.ladder_max);
    REQUIRE(g.bub_pin >= 0.0);
    REQUIRE(g.bub_pin < 1e-7);                        // the zero-rung normalization pin (7.5e-9 measured: the PH-sym transform floor)
    REQUIRE(g.fit_err_G < 1e-2);
    REQUIRE(g.fit_err_S < 1e-2);
    REQUIRE(g.gamma_bare[1] > 0.0);                   // the bare bubble violates C1 at O(1)
    REQUIRE(g.gamma_lam[1] < 1e-2 * g.gamma_bare[1]); // Lambda restores it to fit class
    REQUIRE(g.gamma_lam[2] < 1e-2 * g.gamma_bare[2]);
    REQUIRE(g.dlam_max > 0.0);
    REQUIRE(std::isfinite(g.ladder_max));

    // a proper window: the G-g meter now reports the window truncation's C1 violation
    auto gw4 = gate_at(nda::range(0, 4), 2);
    app_log(1, "ward_wiring_gates [C = [0,4) of {}]: bub_pin {:.3e}; Gamma-trace bare/Lambda "
               "nu1 {:.3e}/{:.3e} (ratio {:.3e}); pole-vs-tau {:.3e}; |dlam| {:.3e}",
            mf->nbnd(), gw4.bub_pin, gw4.gamma_bare[1], gw4.gamma_lam[1],
            gw4.gamma_lam[1] / gw4.gamma_bare[1], gw4.pole_vs_tau, gw4.dlam_max);
    REQUIRE(gw4.bub_pin < 1e-7);

    // the state after ONE iteration (the first GW Sigma and the G it produced -- NOT the
    // Sigma = 0 start: that is the update_w-time readout leg above, which is bitwise). The
    // second, independent (G, Sigma) pair for the fixture G-g and the pins.
    auto g1 = gate_at(nda::range(0, mf->nbnd()), 1);
    app_log(1, "ward_wiring_gates [C = all, after 1 iteration]: bub_pin {:.3e}; fit_err G "
               "{:.3e}; pole-vs-tau {:.3e}; Gamma-trace bare/Lambda nu1 {:.3e}/{:.3e}; |dlam| {:.3e}",
            g1.bub_pin, g1.fit_err_G, g1.pole_vs_tau, g1.gamma_bare[1], g1.gamma_lam[1], g1.dlam_max);
    REQUIRE(g1.bub_pin < 1e-7);
    REQUIRE(g1.gamma_lam[1] < 1e-2 * g1.gamma_bare[1]);
    REQUIRE(g1.dlam_max > 0.0);
#endif
  }

} // namespace bdft_tests
