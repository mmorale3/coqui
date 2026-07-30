/**
 * INCREMENT S6 -- Formulation B-S end to end on a real system (LiH-222, symmetry-free).
 *
 * B-S is the Phi-derivable second-order-exchange functional with statically screened
 * (i.nu = 0) rungs. This target runs the WHOLE loop with vertex_rung = "static":
 *   update_w:  G -> Pi_RPA -> [NEW] W0[G] (one-frequency Dyson) -> W-Dyson (P = RPA)
 *   corr:      Sigma^GW + Sigma^{C,x}(tau-local G^3 (W0)^2) + Sigma^{C,r}(response)
 * and checks that it is finite, stable, and physically active.
 *
 * What this pins that the unit tests cannot:
 *   - the update_w SEAM: W0 is built from the SAME-ITERATION RPA polarizability, before
 *     any vertex piece runs, so a physically screened rung exists from iteration 1 and
 *     the parent's bare-rung bootstrap is structurally unnecessary (plan section 2.2);
 *   - that P really stays RPA (B-S has NO polarization injection -- the forbidden hybrid
 *     is unrepresentable, and eval_Pi_C carries a tripwire);
 *   - that BOTH self-energy cuts are present (Phi-derivability) and that the run is not
 *     silently identical to plain scGW;
 *   - the three q->0 head policies, with the head applied to ONE W0 in every appearance.
 *
 * Correctness of the pieces is pinned elsewhere and is not re-derived here:
 *   test_vertex_fdoracle (the FD oracle -- Phi-derivability itself),
 *   test_vertex_sigma/static_rung_W0, test_vertex_pi/static_rung_W0, test_vertex_w0.
 */

#undef NDEBUG

#include <complex>
#include <cmath>
#include <string>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"

#include "numerics/imag_axes_ft/IAFT.hpp"
#include "mean_field/default_MF.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/HF/hf_t.h"
#include "methods/GW/gw_t.h"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include "methods/vertex/vertex_t.h"

namespace bdft_tests {

  using namespace methods;

  TEST_CASE("vertex_static_e2e", "[methods][vertex][static][e2e]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_static_e2e skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    // wmax headroom: B-S's intermediates are ordinary GG/GGG products (no twisted pairs,
    // no double convolution), so the dynamic theory's 3-4x requirement is relaxed -- but
    // the same basis is kept here so the comparison against the dynamic path is like for
    // like.
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_static_e2e";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    struct res_t { double e_hf, e_corr; };

    // rung = "" means no vertex at all (the plain-scGW reference)
    auto run = [&](std::string const &rung, std::string const &policy) -> res_t {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");

      const bool with_vertex = not rung.empty();
      // C straddles the LiH gap: HOMO = 1, LUMO = 2 (4 electrons, 16 bands)
      solvers::vertex_t vtx(&ft, with_vertex ? "2nd_exchange" : "none",
                            with_vertex ? nda::range(1, 3) : nda::range(0, 0), mf->nbnd(),
                            with_vertex ? policy : "ignore_g0",
                            "global", -1, 1e-8, -1.0, -1.0,
                            with_vertex ? rung : "dynamic");
      if (vtx.enabled()) {
        // BOTH cuts, as production does (MBPT_drivers.cpp:372-373). For B-S the
        // scr_coulomb attachment is what builds W0 inside update_w; the gw attachment is
        // what evaluates Sigma^{C,x} + Sigma^{C,r}.
        scr_eri.set_vertex(&vtx);
        gw.set_vertex(&vtx);
      }
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     2, false, 1e-9, true);
      mpi_context->comm.barrier();
      return {e_hf, e_corr};
    };

    // ---- plain scGW reference ----------------------------------------------------------
    auto ref = run("", "");
    app_log(1, "vertex_static_e2e: plain scGW      e_hf = {:.12f}, e_corr = {:.12f}",
            ref.e_hf, ref.e_corr);
    REQUIRE(std::isfinite(ref.e_hf));
    REQUIRE(std::isfinite(ref.e_corr));

    // ---- B-S, the three q->0 head policies ---------------------------------------------
    for (std::string policy : {std::string("ignore_g0"), std::string("v1_skip"),
                               std::string("gygi")}) {
      auto bs = run("static", policy);
      app_log(1, "vertex_static_e2e: B-S [{:>9}]  e_hf = {:.12f}, e_corr = {:.12f}, "
                 "d(e_corr) vs scGW = {:+.3e}",
              policy, bs.e_hf, bs.e_corr, bs.e_corr - ref.e_corr);
      REQUIRE(std::isfinite(bs.e_hf));
      REQUIRE(std::isfinite(bs.e_corr));
      // e_hf is evaluated on the SELF-CONSISTENT G, so adding a correlation term does
      // move it -- through the density, not through the functional. It must move only
      // slightly (measured: ~7e-5 relative), never by a correlation-sized amount.
      REQUIRE(std::abs(bs.e_hf - ref.e_hf) < 1e-3 * std::abs(ref.e_hf));
      // ... and the vertex must actually DO something.
      REQUIRE(std::abs(bs.e_corr - ref.e_corr) > 1e-9);
      // sanity band: a subspace vertex on 2 of 16 bands must not swamp the correlation
      // energy (an uncontrolled run in the parent theory reached O(1e15)).
      REQUIRE(std::abs(bs.e_corr - ref.e_corr) < 0.5 * std::abs(ref.e_corr));
    }

    // ---- B-L (linear rung): the tangent completion ---------------------------------------
    // Adds, on top of B-S: the two MIXED Sigma terms (one bosonic convolution each) and
    // the full-weight P^{C,L} injection into the Dyson equation -- so unlike B-S, B-L's
    // screening IS vertex-corrected. Its response term sandwiches pi^dyn - Pi^{C,0}(tau=0),
    // whose size is the built-in X^L meter of the static-kernel approximation.
    auto bl = run("linear", "ignore_g0");
    {
      app_log(1, "vertex_static_e2e: B-L [ignore_g0]  e_hf = {:.12f}, e_corr = {:.12f}, "
                 "d(e_corr) vs scGW = {:+.3e}", bl.e_hf, bl.e_corr, bl.e_corr - ref.e_corr);
      REQUIRE(std::isfinite(bl.e_hf));
      REQUIRE(std::isfinite(bl.e_corr));
      REQUIRE(std::abs(bl.e_hf - ref.e_hf) < 1e-3 * std::abs(ref.e_hf));
      REQUIRE(std::abs(bl.e_corr - ref.e_corr) > 1e-9);
      REQUIRE(std::abs(bl.e_corr - ref.e_corr) < 0.5 * std::abs(ref.e_corr));
    }

    // ---- THE PRODUCTION-SCALE REFACTOR GATE for eq:pibardynfact --------------------------
    // B-L again with vertex_pidyn = "check": BOTH pi^dyn routes -- the factorized single
    // bosonic pairing and the historic full-dynamic-Pi^C-then-tau=0 kernel -- run on
    // IDENTICAL inputs every iteration, and vertex_t gates their agreement internally.
    // test_vertex_pibardynfact pins the algebra on a toy; this pins it on the real LiH path
    // (real DLR grid, real Zbar/W0bar rungs, the real response chain, both scf iterations),
    // which is where a representability failure of either integrand would first show. The
    // timer table this run prints carries the measured speedup (both rows present).
    {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(1, 3), mf->nbnd(),
                            "ignore_g0", "global", -1, 1e-8, -1.0, -1.0, "linear");
      vtx.set_pidyn_mode("check");
      REQUIRE(vtx.pidyn_mode() == 2);
      scr_eri.set_vertex(&vtx);
      gw.set_vertex(&vtx);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     2, false, 1e-9, true);
      mpi_context->comm.barrier();
      const double dev = vtx.pidyn_check_max();
      app_log(1, "vertex_static_e2e: B-L pidyn CHECK  e_corr = {:.12f}, "
                 "max rel |pi^dyn(factorized) - pi^dyn(kernel)| = {:.4e}", e_corr, dev);
      // liveness: both routes really ran and really are different code. Exact zero here
      // would mean one route silently supplied the other's answer -- and in this project
      // suspiciously exact agreement across a supposed code change has twice been the tell
      // for a stale artifact, so it is a failure, not a pass.
      REQUIRE(dev > 0.0);
      // MEASURED 2026-07-30 at this grid (beta = 1000, wmax = 6, prec = "low", eps = 1e-6):
      // 3.58e-03 in scf iteration 1 and 2.11e-02 in iteration 2. That is the DLR
      // representability floor of reading two different exact Matsubara sums through the same
      // tau = 0 row, and it is DATA DEPENDENT -- which is exactly why vertex_t warns on it
      // instead of aborting, and why the abort bar is an O(1) routing bar. It is NOT the pole
      // algebra: test_vertex_pibardynfact/production_grid_attribution measures the pole
      // contribution at a factor 0.80, i.e. none.
      //
      // So gate here only on what is meaningful: the deviation must stay far below the O(1)
      // scale at which a mis-routing would show (the closest control the routing pin rejects
      // is 1.24). The PHYSICS statement is the shift assertion in the next block.
      REQUIRE(dev < 0.25);
      // ... and record when pi^dyn is grid-limited, which at prec = "low" it is.
      if (dev > 1e2 * ft.eps())
        app_log(1, "vertex_static_e2e: NOTE pi^dyn is grid-limited at iaft eps = {:.1e} "
                   "(dev = {:.3e} >> 100*eps). This bounds pi^dyn BY EITHER ROUTE; the lever "
                   "is iaft prec, not vertex_pidyn.", ft.eps(), dev);
      // the two routes must also give the same PHYSICS: check mode consumes the factorized
      // value, so e_corr must reproduce the default B-L run exactly.
      REQUIRE(e_corr == bl.e_corr);
      REQUIRE(e_hf == bl.e_hf);
    }

    // ---- and the historic route must land on the same physics ----------------------------
    // vertex_pidyn = "kernel" is what every B-L number before 2026-07-30 was computed with.
    // It must agree with the factorized route to the same floor -- this is the statement
    // "the switch-over did not move B-L's answer", made as an assertion on e_corr rather
    // than an assertion on an intermediate.
    {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(1, 3), mf->nbnd(),
                            "ignore_g0", "global", -1, 1e-8, -1.0, -1.0, "linear");
      vtx.set_pidyn_mode("kernel");
      scr_eri.set_vertex(&vtx);
      gw.set_vertex(&vtx);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     2, false, 1e-9, true);
      mpi_context->comm.barrier();
      const double shift_f = bl.e_corr - ref.e_corr;
      const double shift_k = e_corr - ref.e_corr;
      app_log(1, "vertex_static_e2e: B-L pidyn KERNEL e_corr = {:.12f} (factorized "
                 "{:.12f}); vertex shift {:+.4e} vs {:+.4e}, rel diff of the SHIFT = {:.3e}",
              e_corr, bl.e_corr, shift_k, shift_f,
              std::abs(shift_k - shift_f) / std::max(std::abs(shift_k), 1e-30));
      REQUIRE(std::isfinite(e_corr));
      // BOTH bars are relative to the VERTEX SHIFT, never to e_corr's own magnitude. e_corr
      // is ~84x the shift here (0.095 vs 1.13e-03), so an e_corr-relative bar is a bar three
      // orders TIGHTER on the quantity actually under test -- tighter than the measured DLR
      // floor of the object feeding it, hence meaningless. MEASURED 2026-07-30:
      //   e_corr  -0.095078681891 (kernel) vs -0.095078649549 (factorized) => 3.23e-08
      //   vertex shift  +1.1343e-03 both ways => rel diff of the SHIFT = 2.85e-05
      REQUIRE(std::abs(e_corr - bl.e_corr) <= 1e-3 * std::abs(shift_k));
      REQUIRE(std::abs(shift_k - shift_f) <= 1e-4 * std::abs(shift_k));
    }

    // ---- C = empty set must reproduce plain scGW EXACTLY in static mode -----------------
    // (the primary regression of the whole project: with no subspace the vertex is inert)
    {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(0, 0), mf->nbnd(),
                            "ignore_g0", "global", -1, 1e-8, -1.0, -1.0, "static");
      REQUIRE(not vtx.active());
      scr_eri.set_vertex(&vtx);
      gw.set_vertex(&vtx);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     2, false, 1e-9, true);
      mpi_context->comm.barrier();
      app_log(1, "vertex_static_e2e: B-S with C = {{}}  e_hf = {:.12f}, e_corr = {:.12f} "
                 "(must equal plain scGW bit-for-bit)", e_hf, e_corr);
      REQUIRE(e_hf == ref.e_hf);
      REQUIRE(e_corr == ref.e_corr);
    }
#endif
  }

  // ======================================================================================
  // THE GOLD CHECK for the symmetry-adapted Sigma^{C,r}: the SAME physical LiH-222 state
  // driven through the nosym (qe_lih222) and sym (qe_lih222_sym) variants. Plain scGW
  // supplies the cross-variant baseline (the two datasets are not bit-identical), and the
  // B-S vertex must not add deviation beyond it -- mirroring vertex_ibz_gold, which does
  // the same for the dynamic vertex.
  TEST_CASE("vertex_static_ibz_gold", "[methods][vertex][static][ibz][smoke]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_static_ibz_gold skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_static_gold";

    auto run = [&](std::string const &mf_name, nda::range C, std::string const &rung) {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, mf_name));
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      const bool with_vertex = (C.size() > 0);
      solvers::vertex_t vtx(&ft, with_vertex ? "2nd_exchange" : "none", C, mf->nbnd(),
                            "ignore_g0", "global", -1, 1e-8, -1.0, -1.0,
                            with_vertex ? rung : "dynamic");
      if (vtx.enabled()) { scr_eri.set_vertex(&vtx); gw.set_vertex(&vtx); }
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     2, false, 1e-9, true);
      mpi_context->comm.barrier();
      return std::make_pair(e_hf, e_corr);
    };

    // cross-variant baseline: plain scGW on the two datasets
    auto [hf_ns, ec_ns] = run("qe_lih222", nda::range(0, 0), "dynamic");
    auto [hf_s, ec_s]   = run("qe_lih222_sym", nda::range(0, 0), "dynamic");
    const double d_plain = std::abs(ec_ns - ec_s);
    app_log(1, "static ibz gold: PLAIN scGW  e_corr {:.12f} (nosym) vs {:.12f} (sym), "
               "|D| = {:.3e}", ec_ns, ec_s, d_plain);

    // B-S with C = [1,3) on both
    auto [hfv_ns, ecv_ns] = run("qe_lih222", nda::range(1, 3), "static");
    auto [hfv_s, ecv_s]   = run("qe_lih222_sym", nda::range(1, 3), "static");
    const double d_vert = std::abs(ecv_ns - ecv_s);
    const double shift = std::abs(ecv_ns - ec_ns);
    app_log(1, "static ibz gold: B-S C=[1,3)  e_corr {:.12f} (nosym) vs {:.12f} (sym), "
               "|D| = {:.3e}; vertex shift = {:.3e}", ecv_ns, ecv_s, d_vert, shift);
    app_log(1, "static ibz gold: attribution -- |D e_corr(B-S)| = {:.3e} vs baseline "
               "{:.3e}; excess = {:.3e} ({:.2f}% of the vertex shift)",
            d_vert, d_plain, d_vert - d_plain,
            100.0 * (d_vert - d_plain) / std::max(shift, 1e-30));
    REQUIRE(std::isfinite(ecv_ns));
    REQUIRE(std::isfinite(ecv_s));
    REQUIRE(shift > 1e-9);            // the vertex must be doing something
    // the symmetry path must not add deviation beyond the dataset baseline, up to the
    // C-window leakage margin (same tolerance structure as vertex_ibz_gold)
    REQUIRE(d_vert <= d_plain + 0.05 * shift + 1e-8);

    // B-L must clear the same bar: it adds the P^{C,L} injection (which goes through the
    // Pi^C symmetry path) on top of B-S's Sigma^{C,r}.
    auto [hfl_ns, ecl_ns] = run("qe_lih222", nda::range(1, 3), "linear");
    auto [hfl_s, ecl_s]   = run("qe_lih222_sym", nda::range(1, 3), "linear");
    const double d_lin = std::abs(ecl_ns - ecl_s);
    const double shift_l = std::abs(ecl_ns - ec_ns);
    app_log(1, "static ibz gold: B-L C=[1,3)  e_corr {:.12f} (nosym) vs {:.12f} (sym), "
               "|D| = {:.3e}; vertex shift = {:.3e}; excess = {:.3e} ({:.2f}% of shift)",
            ecl_ns, ecl_s, d_lin, shift_l, d_lin - d_plain,
            100.0 * (d_lin - d_plain) / std::max(shift_l, 1e-30));
    REQUIRE(std::isfinite(ecl_s));
    REQUIRE(shift_l > 1e-9);
    REQUIRE(d_lin <= d_plain + 0.05 * shift_l + 1e-8);
#endif
  }

} // bdft_tests
