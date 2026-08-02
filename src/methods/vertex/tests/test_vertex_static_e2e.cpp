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
#include <vector>

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
// P0 (vertex_bl_mixed_term_split) reaches the kernel's TEST-ONLY S1/S2/S3 term gate,
// vertex_detail::sigma_C_slot_probe.only_term. Header-only; the probe is an inline global.
#include "methods/vertex/vertex_sigma.icc"

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
  // P1 of the B-L plan: DOES THE Gamma HEAD-CHANNEL PROJECTION DO ANYTHING ON LiH?
  //
  // Everything ever measured about vertex_bl_head_projection was measured on Si -- where
  // B-S and B-L disagree in the SIGN of d(e_corr) (+7.118e-03 vs -6.850e-03). On LiH they
  // AGREE, in sign and to 1.7 % (+1.153e-03 vs +1.134e-03). So the discriminating question
  // is what the projection is worth HERE. If it is ~a no-op on LiH and worth 1.17e-02 on Si
  // (1.6x the entire B-S vertex correction), the B-S/B-L discrepancy localises to the Gamma
  // head channel and the search narrows to one rank-1 object.
  //
  // Both arms are COLD and differ ONLY in the knob. That is deliberate: confounding the
  // projection with the starting point is exactly what produced the retracted 2026-07-30
  // conclusion (notes/bl_head_channel_diagnosis.md, "2026-07-31 RETRACTION").
  //
  // The meters are READ FROM vertex_t, not scraped from the log, and the head meters are
  // chi-channel projections <H,.> -- never max-norms. Three independent max-norm gates have
  // already been passed by objects differing ~10x in the only channel that matters.
  //
  // ⚠ THE POLICY MUST BE "gygi". The projection acts on the rank-1 Gamma head, and the head
  // only EXISTS under a gygi-class div treatment: eval_Sigma_C sets
  // head_insertion = (_div_treatment contains "gygi"), and head_ok gates the projection on
  // it. A first run of this A/B used "ignore_g0" (the default elsewhere in this file) and
  // reported a perfectly bit-identical ON/OFF pair -- because the knob was STRUCTURALLY
  // INERT, not because the projection does nothing. That measured null is kept as a fact
  // (at ignore_g0 the projection is exactly inert on LiH, e_corr -0.095078649549 both ways,
  // resp share 0.0260 both ways), and the liveness REQUIREs below exist so that failure mode
  // can never again be misread as a physics result. Every Si number on file is gygi.
  //
  // HIDDEN ([.]): a diagnostic A/B, not coverage -- an unfiltered ctest run skips it.
  //   cd build/tests/bin && OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  //     mpiexec -n 2 --oversubscribe ./test_methods_vertex_static_e2e "[blheadab]"
  //
  // 🚨 The projection is NOT physical: it breaks Phi-derivability (see vertex_t.h,
  // _bl_head_projection). This measures what it MOVES; it does not endorse using it.
  TEST_CASE("vertex_bl_head_projection_lih_ab", "[.][methods][vertex][static][blheadab]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_bl_head_projection_lih_ab skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_blheadab";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    struct res_t {
      double e_hf, e_corr;
      double hl, hs, removed, resp_share;   // the four A/B meters, last iteration
    };

    // The head-carrying policy -- see the ⚠ above. Not a parameter: at any other value the
    // A/B measures nothing.
    const std::string policy = "gygi";

    // rung = "" is the plain-scGW reference; proj is only read when rung == "linear".
    auto run = [&](std::string const &rung, bool proj) -> res_t {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");

      const bool with_vertex = not rung.empty();
      solvers::vertex_t vtx(&ft, with_vertex ? "2nd_exchange" : "none",
                            with_vertex ? nda::range(1, 3) : nda::range(0, 0), mf->nbnd(),
                            with_vertex ? policy : "ignore_g0",
                            "global", -1, 1e-8, -1.0, -1.0,
                            with_vertex ? rung : "dynamic");
      vtx.set_bl_head_projection(proj);
      REQUIRE(vtx.bl_head_projection() == proj);
      if (vtx.enabled()) {
        scr_eri.set_vertex(&vtx);
        gw.set_vertex(&vtx);
      }
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     2, false, 1e-9, true);
      mpi_context->comm.barrier();
      return {e_hf, e_corr, vtx.diag_head_hl(), vtx.diag_head_hs(),
              vtx.diag_head_removed_frac(), vtx.diag_resp_share()};
    };

    auto ref = run("", false);
    auto bs  = run("static", false);   // B-S is untouched by the knob (head_ok needs need_dyn)
    auto off = run("linear", false);
    auto on  = run("linear", true);

    app_log(1, "\n================ B-L Gamma-HEAD PROJECTION A/B on LiH-222 ================");
    app_log(1, "  vertex div_treatment = \"{}\" (the head only exists under gygi), "
               "C = [1,3), 2 COLD iterations", policy);
    app_log(1, "  plain scGW            e_corr = {:.12f}", ref.e_corr);
    app_log(1, "  B-S                   e_corr = {:.12f}, d vs scGW = {:+.4e}",
            bs.e_corr, bs.e_corr - ref.e_corr);
    app_log(1, "  B-L projection OFF    e_corr = {:.12f}, d vs scGW = {:+.4e}",
            off.e_corr, off.e_corr - ref.e_corr);
    app_log(1, "  B-L projection ON     e_corr = {:.12f}, d vs scGW = {:+.4e}",
            on.e_corr, on.e_corr - ref.e_corr);
    app_log(1, "  --> THE PROJECTION IS WORTH  {:+.4e}  in e_corr  ({:.2f} % of the "
               "OFF vertex shift)", on.e_corr - off.e_corr,
            (std::abs(off.e_corr - ref.e_corr) > 0.0
                 ? 100.0 * std::abs(on.e_corr - off.e_corr)
                       / std::abs(off.e_corr - ref.e_corr) : 0.0));
    app_log(1, "  Si reference for the same quantity: 1.17e-02, i.e. 1.6x the whole B-S "
               "vertex correction (+7.118e-03).");
    app_log(1, "  ---- head meters (last iteration) ----");
    app_log(1, "    |<H, Pi^L(Gamma)>|      OFF {:.6e}   ON {:.6e}", off.hl, on.hl);
    app_log(1, "    |<H, Pi^{{C,0}}(Gamma)>|  OFF {:.6e}   ON {:.6e}", off.hs, on.hs);
    app_log(1, "    max|removed|/max|Pi^L|  OFF {:.4f}         ON {:.4f}   "
               "(Si: 0.6590 at the converged solution)", off.removed, on.removed);
    app_log(1, "    resp share |S^(C,r)|/|S^(C,x)|  OFF {:.4f}   ON {:.4f}   "
               "(Si: 1.0126 OFF / 0.1037 ON; LiH range on file 0.12-0.24)",
            off.resp_share, on.resp_share);
    app_log(1, "  ---- THE SIGN QUESTION, at ONE policy on ONE system ----");
    app_log(1, "    d(e_corr):  B-S {:+.4e}   B-L(honest, projection OFF) {:+.4e}   "
               "-> signs {}",
            bs.e_corr - ref.e_corr, off.e_corr - ref.e_corr,
            (((bs.e_corr - ref.e_corr) * (off.e_corr - ref.e_corr)) > 0.0
                 ? "AGREE" : "DISAGREE"));
    app_log(1, "    Si at the same policy: B-S +7.118e-03, B-L OFF -6.850e-03 -> DISAGREE.");
    app_log(1, "    The +1.153e-03 / +1.134e-03 LiH pair on file that motivated \"LiH "
               "agrees, Si does not\"");
    app_log(1, "    was measured at div_treatment = \"ignore_g0\", i.e. WITH NO GAMMA HEAD "
               "AT ALL.");
    app_log(1, "==========================================================================\n");

    REQUIRE(std::isfinite(ref.e_corr));
    REQUIRE(std::isfinite(off.e_corr));
    REQUIRE(std::isfinite(on.e_corr));
    // LIVENESS -- without these the A/B reports "no effect" whenever the knob never fired,
    // which is exactly what happened on the first attempt (div_treatment = "ignore_g0" =>
    // no Gamma head => head_ok false => the projection block never entered, and the two arms
    // came out bit-identical). A failure of `on.removed > 0` means THE CONFIG IS WRONG, not
    // that the projection is harmless.
    REQUIRE(off.removed == 0.0);   // OFF really did not touch Pi^L
    REQUIRE(on.removed > 0.0);     // ON really did -- and the head really exists
    // ... and the head meters really were measured (-1 = never evaluated != a clean head).
    REQUIRE(off.hl >= 0.0);
    REQUIRE(on.hl >= 0.0);
    REQUIRE(off.resp_share >= 0.0);
#endif
  }

  // ======================================================================================
  // THE CONSTANT-RUNG ABSOLUTE PIN (P3): is pi^dyn's Gamma-head violation retarded-rung
  // PHYSICS, or a DEFECT in the equal-time path that Pi^{C,0} does not share?
  //
  // X^L = pi^dyn - Pi^{C,0}(tau=0) is supposed to vanish for a genuinely static screen. It
  // does not: on LiH at gygi it sits at X^L/Pi^0 = 0.34, and |<H,pi^dyn>| exceeds
  // |<H,Pi^{C,0}>| by 7.7e+04 at iteration 1. Nothing so far separates "the rung really is
  // retarded" from "the two objects are computed by paths that differ in more than the rung".
  //
  // ⚠ WHY NOT "just zero the dynamic rung". The two paths do NOT share an instantaneous
  // rung: Pi^{C,0} = pi_c_accumulate_w(rung = W0bar, Wdyn = nullptr), while
  // pi^dyn = pi_dyn_factorized(rung = Z (BARE), Wdyn = Wdyn_w). Zeroing Wdyn_w leaves
  // pi^dyn on the bare rung, so X^L stays O(1) for a reason that says nothing about the
  // equal-time path -- a vacuous test. vertex_bl_pidyn_const_rung instead sets
  // Wdyn_w(i.nu) := W0bar - Z at every frequency, making pi^dyn's TOTAL rung identically
  // W0bar: the same rung, constant in nu. Then the two objects are one integral by two
  // routes, and X^L must collapse.
  //
  // FLOOR: the DLR representability of the two integrands, ~1e-10 on the toy
  // (test_vertex_pibardynfact/static_rung), NOT machine epsilon. The gate is set far above
  // that floor and far below the 0.34 it must destroy.
  //
  // HIDDEN ([.]): diagnostic. Run with:
  //   cd build/tests/bin && OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  //     mpiexec -n 2 --oversubscribe ./test_methods_vertex_static_e2e "[construng]"
  TEST_CASE("vertex_bl_pidyn_const_rung_pin", "[.][methods][vertex][static][construng]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_bl_pidyn_const_rung_pin skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    std::string output = "coqui_vertex_construng";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    struct res_t { double e_corr, xl, hl, hs; };

    // "gygi" for the same reason as the A/B above: the head meters only exist there.
    // eps is the IAFT (DLR) tolerance -- the axis the precision scan sweeps.
    auto run = [&](double eps, bool const_rung) -> res_t {
      imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, eps);
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(1, 3), mf->nbnd(),
                            "gygi", "global", -1, 1e-8, -1.0, -1.0, "linear");
      vtx.set_bl_pidyn_const_rung(const_rung);
      REQUIRE(vtx.bl_pidyn_const_rung() == const_rung);
      scr_eri.set_vertex(&vtx);
      gw.set_vertex(&vtx);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     2, false, 1e-9, true);
      (void) e_hf;
      mpi_context->comm.barrier();
      return {e_corr, vtx.diag_xl_rel(), vtx.diag_head_hl(), vtx.diag_head_hs()};
    };

    // THE DISCRIMINATOR is a PRECISION SCAN, not a single number. pi^dyn is known to be
    // grid-limited at prec = "low" (~1e-3 by either route -- test_vertex_pibardynfact's
    // production_grid_attribution and the pidyn CHECK in vertex_static_e2e above), so an
    // absolute value alone cannot tell "representability floor" from "defect". What can:
    //   - the PINNED X^L must FALL with eps          (=> it is representability)
    //   - the CONTROL X^L must NOT fall with eps     (=> it is physics, not the grid)
    // Two eps, four runs.
    const double eps_lo = 1e-6, eps_hi = 1e-10;
    auto ctl_lo = run(eps_lo, false);
    auto pin_lo = run(eps_lo, true);
    auto ctl_hi = run(eps_hi, false);
    auto pin_hi = run(eps_hi, true);

    app_log(1, "\n========== CONSTANT-RUNG ABSOLUTE PIN (pi^dyn vs Pi^{{C,0}}) ==========");
    app_log(1, "  LiH-222, gygi, C = [1,3), 2 COLD iterations; meters from the LAST one.");
    app_log(1, "  {:<26} {:>15} {:>15} {:>15} {:>15}", "", "control eps=1e-6",
            "PINNED eps=1e-6", "control eps=1e-10", "PINNED eps=1e-10");
    app_log(1, "  {:<26} {:>15.6e} {:>15.6e} {:>15.6e} {:>15.6e}", "X^L / Pi^0",
            ctl_lo.xl, pin_lo.xl, ctl_hi.xl, pin_hi.xl);
    // NB hl is |<H, Pi^L>| -- the head overlap of the DIFFERENCE pi^dyn - Pi^{C,0}, which
    // is what Sigma^{L,r} actually sandwiches -- NOT |<H, pi^dyn>|. Under the pin it must
    // go to zero along with X^L; Pi^{C,0}'s own head overlap (hs) is untouched by the knob
    // and does NOT vanish -- once G is dressed, Pi^{C,0} does not satisfy the q->0 head
    // suppression exactly either.
    app_log(1, "  {:<26} {:>15.6e} {:>15.6e} {:>15.6e} {:>15.6e}", "|<H, Pi^L(Gamma)>|",
            ctl_lo.hl, pin_lo.hl, ctl_hi.hl, pin_hi.hl);
    app_log(1, "  {:<26} {:>15.6e} {:>15.6e} {:>15.6e} {:>15.6e}", "|<H, Pi^(C,0)(Gamma)>|",
            ctl_lo.hs, pin_lo.hs, ctl_hi.hs, pin_hi.hs);
    app_log(1, "  {:<26} {:>15.4e} {:>15.4e} {:>15.4e} {:>15.4e}", "head ratio Pi^L/Pi^0",
            (ctl_lo.hs > 0.0 ? ctl_lo.hl / ctl_lo.hs : 0.0),
            (pin_lo.hs > 0.0 ? pin_lo.hl / pin_lo.hs : 0.0),
            (ctl_hi.hs > 0.0 ? ctl_hi.hl / ctl_hi.hs : 0.0),
            (pin_hi.hs > 0.0 ? pin_hi.hl / pin_hi.hs : 0.0));
    app_log(1, "  {:<26} {:>15.9f} {:>15.9f} {:>15.9f} {:>15.9f}", "e_corr",
            ctl_lo.e_corr, pin_lo.e_corr, ctl_hi.e_corr, pin_hi.e_corr);
    app_log(1, "  ---- READING ----");
    app_log(1, "    pinned X^L fell {:.1f}x from eps 1e-6 to 1e-10   (representability "
               "=> the equal-time path is CLEAN)",
            (pin_hi.xl > 0.0 ? pin_lo.xl / pin_hi.xl : 0.0));
    app_log(1, "    control X^L moved {:.3f}x over the same range     (~1 => pi^dyn's head "
               "violation is PHYSICS, not the grid)",
            (ctl_hi.xl > 0.0 ? ctl_lo.xl / ctl_hi.xl : 0.0));
    app_log(1, "======================================================================\n");

    REQUIRE(std::isfinite(pin_lo.e_corr));
    REQUIRE(std::isfinite(pin_hi.e_corr));
    // LIVENESS: the control must actually show the violation this pin exists to explain.
    REQUIRE(ctl_lo.xl > 1e-2);
    // The head overlap of the DIFFERENCE must dominate Pi^{C,0}'s own. MEASURED at the
    // LAST iteration: 16.2 (eps 1e-6) / 16.9 (eps 1e-10). ⚠ The 7.7e+04 on file is an
    // ITERATION-1 number -- these meters carry the last iteration, and a bar set from the
    // wrong iteration is what first failed this test.
    REQUIRE(ctl_lo.hl > 10.0 * ctl_lo.hs);
    // THE PIN, part 1: forcing the rung static must destroy X^L at BOTH grids.
    REQUIRE(pin_lo.xl < 1e-2 * ctl_lo.xl);
    REQUIRE(pin_hi.xl < 1e-2 * ctl_hi.xl);
    // THE PIN, part 2 -- the part with teeth: the residue must be REPRESENTABILITY, i.e.
    // it must FALL when the DLR tolerance is tightened by four orders. A residue that
    // plateaus is a defect in the equal-time path, and that is the finding this test
    // exists to produce.
    REQUIRE(pin_hi.xl < 0.1 * pin_lo.xl);
    // ... while the CONTROL's X^L must NOT fall -- it is the retarded rung, not the grid.
    REQUIRE(ctl_hi.xl > 0.5 * ctl_lo.xl);
    // and the head channel of the difference must collapse with it.
    REQUIRE(pin_hi.hl < 1e-3 * ctl_hi.hl);
#endif
  }

  // ======================================================================================
  // P2: WHICH TERM CARRIES B-L's NEGATIVE SIGN?
  //
  // B-L's d(e_corr) is negative where B-S's is positive (LiH gygi: -3.8642e-03 vs
  // +1.1986e-03). B-L's self-energy is Sigma^(C,x) + Sigma^(L,r), and the strong prior is
  // Sigma^(L,r): its response share ||Sigma^(C,r)||/||Sigma^(C,x)|| is 0.64 here (Si: 1.01),
  // i.e. the "linear correction" is comparable to the term it corrects.
  //
  // ⚠ THESE SHARES ARE ABLATIONS WITH FEEDBACK -- THEY DO NOT ADD UP, AND THAT IS CORRECT.
  // eval_corr_energy is (-0.5 * spin_factor) * sum k_weight <Sigma, G>, linear in Sigma at
  // FIXED G -- but scf_driver runs update_G (the Dyson step) BEFORE evaluating it
  // (scf_driver.cpp:188 vs :200), so each arm's energy is taken at its OWN post-Dyson G.
  // The arms therefore never share a G, not even at iteration 1, and no additivity identity
  // holds. Two earlier versions of this test asserted one and failed:
  //   two-way   sum -8.7380e-03 vs d(full) -6.6760e-03   residual 2.062e-03
  //   three-way sum -7.7916e-03 vs d(full) -6.6760e-03   residual 1.116e-03
  // The three-way residual (1.116e-03, ~17 % of d(full)) is the MEASURED nonlinearity of
  // G's response to Sigma. It is reported, not asserted away.
  //
  // WHAT IS ASSERTED INSTEAD is assumption-free and needs no additivity: removing
  // Sigma^(C,x) FLIPS THE SIGN of d(e_corr), and removing Sigma^(L,r) does not. That is the
  // whole question P2 asks, and it survives the nonlinearity.
  //
  // HIDDEN ([.]). Run with:
  //   cd build/tests/bin && OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  //     mpiexec -n 2 --oversubscribe ./test_methods_vertex_static_e2e "[bldecomp]"
  TEST_CASE("vertex_bl_term_decomposition", "[.][methods][vertex][static][bldecomp]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_bl_term_decomposition skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_bldecomp";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    // vertex = false is the plain-scGW baseline; drop is only read when vertex is true.
    auto run_impl = [&](bool vertex, int drop, long niter, std::string const &rung) -> double {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, vertex ? "2nd_exchange" : "none",
                            vertex ? nda::range(1, 3) : nda::range(0, 0), mf->nbnd(),
                            vertex ? "gygi" : "ignore_g0",
                            "global", -1, 1e-8, -1.0, -1.0, vertex ? rung : "dynamic");
      vtx.set_bl_drop(drop);
      REQUIRE(vtx.bl_drop() == drop);
      if (vtx.enabled()) { scr_eri.set_vertex(&vtx); gw.set_vertex(&vtx); }
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     niter, false, 1e-9, true);
      (void) e_hf;
      mpi_context->comm.barrier();
      return e_corr;
    };
    auto run   = [&](bool vertex, int drop, long niter) {
      return run_impl(vertex, drop, niter, "linear");
    };
    // B-S at the SAME policy and iteration count -- the like-for-like sign comparison. The
    // +1.1986e-03 on file is a 2-iteration number and must not be compared to niter = 1.
    auto runBS = [&](long niter) { return run_impl(true, 0, niter, "static"); };

    auto report = [&](long niter, bool exact) {
      const double ref   = run(false, 0, niter);
      const double bs    = runBS(niter);
      const double full  = run(true,  0, niter);
      const double dropR = run(true,  1, niter);   // no Sigma^(L,r)
      const double dropK = run(true,  2, niter);   // no Sigma^(C,x)
      const double dropB = run(true,  3, niter);   // neither -- P^{C,L} via Sigma_GW alone
      const double d_full = full  - ref;
      const double d_b    = dropB - ref;           // the P^{C,L} share, MEASURED
      const double x      = dropR - dropB;         // Sigma^(C,x), clean of d_b
      const double r      = dropK - dropB;         // Sigma^(L,r), clean of d_b
      app_log(1, "\n===== B-L TERM ABLATION, niter = {} ({}) =====", niter,
              exact ? "1 iteration" : "2 iterations");
      app_log(1, "  Shares are ABLATIONS WITH FEEDBACK: the Dyson update precedes the energy "
                 "evaluation, so each\n"
                 "  arm is measured at its OWN G and the pieces do NOT add. The residual "
                 "below IS that nonlinearity.");
      app_log(1, "  plain scGW                       e_corr = {:.12f}", ref);
      app_log(1, "  B-S                              e_corr = {:.12f}   d = {:+.4e}",
              bs, bs - ref);
      app_log(1, "  B-L full                         e_corr = {:.12f}   d = {:+.4e}",
              full, d_full);
      app_log(1, "  ---- the THREE pieces (drop-both is the common P^{{C,L}} baseline) ----");
      app_log(1, "    P^(C,L) via Sigma_GW                 {:+.4e}   (Dyson injection; "
                 "present in EVERY B-L arm)", d_b);
      app_log(1, "    Sigma^(C,x)  [kernel]                {:+.4e}", x);
      app_log(1, "    Sigma^(L,r)  [response]              {:+.4e}", r);
      app_log(1, "    sum {:+.4e}  vs  d(full) {:+.4e}   -> G-RESPONSE NONLINEARITY "
                 "{:.3e} ({:.1f} % of d(full))",
              d_b + x + r, d_full, std::abs(d_b + x + r - d_full),
              (std::abs(d_full) > 0.0
                   ? 100.0 * std::abs(d_b + x + r - d_full) / std::abs(d_full) : 0.0));
      app_log(1, "  ---- THE ASSUMPTION-FREE STATEMENT (no additivity needed) ----");
      app_log(1, "    d(full)              = {:+.4e}", d_full);
      app_log(1, "    d(full) WITHOUT Sigma^(C,x) = {:+.4e}   <- sign {}", dropK - ref,
              (((dropK - ref) * d_full) > 0.0 ? "UNCHANGED" : "FLIPS"));
      app_log(1, "    d(full) WITHOUT Sigma^(L,r) = {:+.4e}   <- sign {}", dropR - ref,
              (((dropR - ref) * d_full) > 0.0 ? "UNCHANGED" : "FLIPS"));
      const double mx = std::max({std::abs(d_b), std::abs(x), std::abs(r)});
      app_log(1, "  --> signs: P^(C,L) {}, Sigma^(C,x) {}, Sigma^(L,r) {}. "
                 "LARGEST piece = {}.",
              (d_b < 0.0 ? "NEG" : "pos"), (x < 0.0 ? "NEG" : "pos"),
              (r < 0.0 ? "NEG" : "pos"),
              (mx == std::abs(x) ? "Sigma^(C,x)"
               : mx == std::abs(r) ? "Sigma^(L,r)" : "P^(C,L)"));
      app_log(1, "  NOTE B-S's Sigma^(C,x) uses W0_x W0_y; B-L's uses the three mixed terms "
                 "W0_x W_y + W_x W0_y - W0_x W0_y,");
      app_log(1, "       so B-S vs B-L on this line is the MIXED-TERM question, not a "
                 "different diagram.");
      app_log(1, "=========================================================\n");
      return std::array<double, 6>{d_full, d_b, x, r, dropK - ref, dropR - ref};
    };

    // ---- the EXACT decomposition: one iteration, one G, additivity must hold ------------
    auto e1 = report(1, true);
    REQUIRE(std::isfinite(e1[0]));
    // liveness: all three pieces must actually be doing something.
    REQUIRE(std::abs(e1[1]) > 1e-12);
    REQUIRE(std::abs(e1[2]) > 1e-12);
    REQUIRE(std::abs(e1[3]) > 1e-12);
    // THE RESULT, stated without any additivity assumption.
    // (a) B-L's shift is negative ...
    REQUIRE(e1[0] < 0.0);
    // (b) ... removing Sigma^(C,x) FLIPS it positive => that term carries the sign ...
    REQUIRE(e1[4] > 0.0);
    // (c) ... while removing Sigma^(L,r) leaves it negative, i.e. the response term is NOT
    // the carrier -- it partially CANCELS the sign. This refutes the standing prior.
    REQUIRE(e1[5] < 0.0);
    // and the kernel term is the dominant piece by magnitude.
    REQUIRE(std::abs(e1[2]) > std::abs(e1[3]));
    REQUIRE(std::abs(e1[2]) > std::abs(e1[1]));

    // ---- the self-consistent view (NOT additive -- reported, not asserted additive) -----
    auto e2 = report(2, false);
    REQUIRE(std::isfinite(e2[0]));
#endif
  }

  // ======================================================================================
  // P0: WHICH OF THE THREE MIXED REDUCTIONS S1/S2/S3 FLIPS THE SIGN?
  //
  // WHERE P2 LEFT THE INVESTIGATION. Sigma^(C,x) carries B-L's negative d(e_corr) (removing
  // it flips the sign; removing Sigma^(L,r) does not). But B-S's Sigma^(C,x) is POSITIVE for
  // the SAME DIAGRAM. The two differ ONLY in the rung combination:
  //
  //     B-S:  W0_x W0_y
  //     B-L:  W0_x W_y + W_x W0_y - W0_x W0_y
  //         =  W0_x W0_y  +  W0_x dW_y  +  dW_x W0_y        (dW := W - W0)
  //         =      S3     +      S1     +      S2           of the kernel's own reductions
  //
  // so the flip must live in S1/S2, the DYNAMIC-FLUCTUATION mixed terms -- or S3 is not what
  // it is claimed to be. This test decides which, by gating the kernel to ONE reduction at a
  // time through vertex_detail::sigma_C_slot_probe.only_term (vertex_sigma.icc:992: the gate
  // is already there, hoisted out of the fill loops, and needs no production code).
  //
  // THE POSITIVE CONTROL that makes the whole reading trustworthy: S3 is, term for term, the
  // rung combination B-S evaluates -- both feed Z_qPQ = W0bar into the doubly-instantaneous
  // reduction. So "B-L restricted to S3" and "B-S's kernel term" must agree in SIGN. If they
  // do not, the premise above is wrong and nothing else here means anything.
  //
  // ⚠ ABLATIONS WITH FEEDBACK, exactly as in [bldecomp]: update_G precedes the energy
  // evaluation (scf_driver.cpp:188 vs :200), so every arm is measured at its OWN post-Dyson
  // G and the pieces do NOT add. Only SIGN-level statements survive the ~17 % nonlinearity.
  // Each kernel share is therefore taken against the drop = 3 arm (P^{C,L} via Sigma_GW
  // alone), which is the ONE baseline common to every B-L arm, and Sigma^(L,r) is switched
  // off (drop = 1) so the isolated object really is the kernel term.
  //
  // AND THEN THE GAMMA-HEAD A/B. The split is run at BOTH div_treatment policies, because
  // the split alone raises a question it cannot answer: the mixed terms are FIRST order in
  // dW and S3 is ZEROTH, yet |S1+S2|/|S3| = 3.23 while the expansion parameter
  // max_nu|dW|/|W0| is 0.28 -- naive power counting predicts 0.56, so something amplifies
  // dW ~6x beyond its magnitude. The standing suspect is the Gamma head (coherent, rank-1,
  // invisible to a max-norm -- trap 2), and "gygi" vs "ignore_g0" toggles exactly that: the
  // former augments the Gamma cell with the analytic rank-1 head insertion, the latter has
  // no head at all. gw and scr_coulomb stay at "ignore_g0" in BOTH halves, so the two
  // differ in one thing and share one plain-scGW reference.
  //
  // Protocol: LiH-222, C = [1,3), 2 COLD iterations -- the same protocol as P1/P2, so these
  // numbers are directly comparable to the -3.8642e-03 / +1.1986e-03 pair on file.
  //
  // HIDDEN ([.]). Run with:
  //   cd build/tests/bin && OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  //     mpiexec -n 2 --oversubscribe ./test_methods_vertex_static_e2e "[blmixed]"
  TEST_CASE("vertex_bl_mixed_term_split", "[.][methods][vertex][static][blmixed]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_bl_mixed_term_split skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_blmixed";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    const long NIT = 2;

    // `only` gates the kernel to one reduction: 0 = all, 1 = S1, 2 = S2, 3 = S3. It is a
    // process-global; every rank runs this same code, and it is CLEARED after the arm so it
    // can never leak into the next one.
    double last_dw_rel = -1.0;   // B-L's expansion parameter from the arm just run
    auto run_impl = [&](bool vertex, int drop, int only, std::string const &rung,
                        std::string const &policy) -> double {
      solvers::vertex_detail::sigma_C_slot_probe.only_term = only;
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, vertex ? "2nd_exchange" : "none",
                            vertex ? nda::range(1, 3) : nda::range(0, 0), mf->nbnd(),
                            vertex ? policy : "ignore_g0",
                            "global", -1, 1e-8, -1.0, -1.0, vertex ? rung : "dynamic");
      vtx.set_bl_drop(drop);
      REQUIRE(vtx.bl_drop() == drop);
      if (vtx.enabled()) { scr_eri.set_vertex(&vtx); gw.set_vertex(&vtx); }
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     NIT, false, 1e-9, true);
      (void) e_hf;
      mpi_context->comm.barrier();
      solvers::vertex_detail::sigma_C_slot_probe.clear();
      last_dw_rel = vtx.diag_dw_rel();
      return e_corr;
    };

    // The plain-scGW reference is SHARED by both policies: gw and scr_coulomb are always
    // constructed at "ignore_g0", and the policy under test is the VERTEX's div_treatment
    // only. So one reference arm serves both halves of the A/B -- and, more importantly,
    // the two halves differ in exactly one thing.
    const double ref = run_impl(false, 0, 0, "dynamic", "ignore_g0");

    struct split_t {
      double bs, bsk, full, baseP, k_all, k_S1, k_S2, k_S3, dw_rel;
      double d_full, d_b, x_all, x_S1, x_S2, x_S3, x_BS;
      double ratio;   // |S1+S2| / |S3|: first order over zeroth
    };
    auto measure = [&](std::string const &policy) -> split_t {
      split_t s{};
      auto BL = [&](int drop, int only) {
        return run_impl(true, drop, only, "linear", policy);
      };
      s.bs    = run_impl(true, 0, 0, "static", policy);   // B-S, both cuts
      s.bsk   = run_impl(true, 1, 0, "static", policy);   // B-S kernel term alone
      s.full  = BL(0, 0);                                 // the honest B-L
      s.dw_rel = last_dw_rel;      // the expansion parameter, max over ALL i.nu
      s.baseP = BL(3, 0);                                 // P^{C,L} via Sigma_GW alone
      s.k_all = BL(1, 0);                                 // S1+S2+S3 (+ P^{C,L})
      s.k_S1  = BL(1, 1);
      s.k_S2  = BL(1, 2);
      s.k_S3  = BL(1, 3);
      // kernel shares, clean of the common P^{C,L} baseline
      s.d_full = s.full  - ref;
      s.d_b    = s.baseP - ref;
      s.x_all  = s.k_all - s.baseP;
      s.x_S1   = s.k_S1  - s.baseP;
      s.x_S2   = s.k_S2  - s.baseP;
      s.x_S3   = s.k_S3  - s.baseP;
      s.x_BS   = s.bsk   - ref;    // B-S's kernel term; its common baseline is ref
      s.ratio  = (std::abs(s.x_S3) > 0.0
                      ? std::abs(s.x_S1 + s.x_S2) / std::abs(s.x_S3) : 0.0);
      return s;
    };

    auto sgn = [](double v) { return v < 0.0 ? "NEG" : "pos"; };

    auto report = [&](std::string const &policy, split_t const &s) {
      app_log(1, "\n===== B-L MIXED-TERM SPLIT (S1/S2/S3), div_treatment = \"{}\" =====",
              policy);
      app_log(1, "  LiH-222, C = [1,3), {} COLD iterations, projection OFF.", NIT);
      app_log(1, "  Shares are ABLATIONS WITH FEEDBACK -- update_G precedes the energy, so");
      app_log(1, "  each arm sits at its OWN G and the pieces do NOT add. SIGN-level only.");
      app_log(1, "  plain scGW                     e_corr = {:.12f}", ref);
      app_log(1, "  B-S (both cuts)                e_corr = {:.12f}   d = {:+.4e}",
              s.bs, s.bs - ref);
      app_log(1, "  B-L full (both cuts)           e_corr = {:.12f}   d = {:+.4e}",
              s.full, s.d_full);
      app_log(1, "  B-L drop=3 (P^(C,L) only)      e_corr = {:.12f}   d = {:+.4e}",
              s.baseP, s.d_b);
      app_log(1, "  ---- the KERNEL term Sigma^(C,x), gated to one reduction (drop = 1) ----");
      app_log(1, "    B-L all three  S1+S2+S3        {:+.4e}   [{}]", s.x_all, sgn(s.x_all));
      app_log(1, "    S1 = W0_x dW_y                 {:+.4e}   [{}]", s.x_S1, sgn(s.x_S1));
      app_log(1, "    S2 = dW_x W0_y                 {:+.4e}   [{}]", s.x_S2, sgn(s.x_S2));
      app_log(1, "    S3 = W0_x W0_y                 {:+.4e}   [{}]", s.x_S3, sgn(s.x_S3));
      app_log(1, "    sum(S1,S2,S3) {:+.4e}  vs  all-three {:+.4e}   -> G-RESPONSE "
                 "NONLINEARITY {:.3e}", s.x_S1 + s.x_S2 + s.x_S3, s.x_all,
              std::abs(s.x_S1 + s.x_S2 + s.x_S3 - s.x_all));
      app_log(1, "  ---- POSITIVE CONTROL: S3 is the rung combination B-S evaluates ----");
      app_log(1, "    B-S kernel term alone (drop=1)  {:+.4e}   [{}]", s.x_BS, sgn(s.x_BS));
      app_log(1, "    B-L restricted to S3            {:+.4e}   [{}]   -> {}",
              s.x_S3, sgn(s.x_S3),
              ((s.x_S3 * s.x_BS) > 0.0 ? "signs AGREE (control holds)"
                                       : "DISAGREE -- S3 is NOT B-S's term, premise broken"));
      app_log(1, "  ---- PAIR SYMMETRY: S1 and S2 are the two orderings ----");
      app_log(1, "    |S1 - S2| = {:.4e}   |S1 + S2| = {:.4e}   ratio {:.3e}",
              std::abs(s.x_S1 - s.x_S2), std::abs(s.x_S1 + s.x_S2),
              (std::abs(s.x_S1 + s.x_S2) > 0.0
                   ? std::abs(s.x_S1 - s.x_S2) / std::abs(s.x_S1 + s.x_S2) : 0.0));
      app_log(1, "  ---- IS THE TANGENT EXPANSION CONTROLLED? ----");
      app_log(1, "    MIXED (S1+S2) = {:+.4e}   STATIC (S3) = {:+.4e}",
              s.x_S1 + s.x_S2, s.x_S3);
      app_log(1, "    |S1+S2| / |S3|      = {:.3f}   <- FIRST order over ZEROTH order",
              s.ratio);
      app_log(1, "    max_nu |dW| / |W0|  = {:.4f}   <- the expansion parameter, ALL "
                 "frequencies", s.dw_rel);
      app_log(1, "    NAIVE power counting predicts |S1+S2|/|S3| ~ 2 * {:.4f} = {:.3f}; "
                 "measured {:.3f}\n"
                 "    -> UNEXPLAINED AMPLIFICATION {:.2f}x",
              s.dw_rel, 2.0 * s.dw_rel, s.ratio,
              (s.dw_rel > 0.0 ? s.ratio / (2.0 * s.dw_rel) : 0.0));
      app_log(1, "    d(e_corr) signs:  B-S {}   B-L {}   -> {}",
              sgn(s.bs - ref), sgn(s.d_full),
              (((s.bs - ref) * s.d_full) > 0.0 ? "AGREE" : "DISAGREE"));
      app_log(1, "==============================================================\n");
    };

    // The two halves differ in ONE thing: whether the vertex's Gamma cell carries the
    // analytic rank-1 head insertion. gygi does, ignore_g0 has no head at all.
    const auto g = measure("gygi");
    report("gygi", g);
    const auto n = measure("ignore_g0");
    report("ignore_g0", n);

    app_log(1, "\n===== THE GAMMA-HEAD A/B ON THE MIXED TERMS =====");
    app_log(1, "  Is the Gamma head what makes the FIRST-order terms outweigh the ZEROTH?");
    app_log(1, "  quantity                        gygi (head)     ignore_g0 (no head)");
    app_log(1, "    STATIC   S3                   {:+.4e}      {:+.4e}", g.x_S3, n.x_S3);
    app_log(1, "    MIXED    S1+S2                {:+.4e}      {:+.4e}",
            g.x_S1 + g.x_S2, n.x_S1 + n.x_S2);
    app_log(1, "    |S1+S2| / |S3|                {:9.3f}       {:9.3f}", g.ratio, n.ratio);
    app_log(1, "    max_nu |dW| / |W0|            {:9.4f}       {:9.4f}", g.dw_rel, n.dw_rel);
    app_log(1, "    d(e_corr) B-S                 {:+.4e}      {:+.4e}",
            g.bs - ref, n.bs - ref);
    app_log(1, "    d(e_corr) B-L                 {:+.4e}      {:+.4e}", g.d_full, n.d_full);
    app_log(1, "    B-S vs B-L signs              {:>9}       {:>9}",
            (((g.bs - ref) * g.d_full) > 0.0 ? "AGREE" : "DISAGREE"),
            (((n.bs - ref) * n.d_full) > 0.0 ? "AGREE" : "DISAGREE"));
    app_log(1, "  -> the head changes the mixed/static ratio by {:.2f}x",
            (n.ratio > 0.0 ? g.ratio / n.ratio : 0.0));
    app_log(1, "=================================================\n");

    REQUIRE(std::isfinite(ref));
    for (auto const *s : {&g, &n}) {
      REQUIRE(std::isfinite(s->full));
      REQUIRE(std::isfinite(s->k_S1));
      REQUIRE(std::isfinite(s->k_S2));
      REQUIRE(std::isfinite(s->k_S3));
      // LIVENESS -- the term gate must actually have FIRED, at BOTH policies. A structurally
      // inert knob returns identical arms and reads as "no effect"; that is trap 7 of the
      // plan, and it is exactly how the head projection behaved at ignore_g0. Three distinct
      // restrictions of a nonzero kernel cannot coincide with each other or with the
      // unrestricted kernel.
      REQUIRE(std::abs(s->x_all) > 1e-12);
      REQUIRE(std::abs(s->k_S1 - s->k_all) > 1e-12);
      REQUIRE(std::abs(s->k_S2 - s->k_all) > 1e-12);
      REQUIRE(std::abs(s->k_S3 - s->k_all) > 1e-12);
      REQUIRE(std::abs(s->k_S1 - s->k_S3) > 1e-12);
      REQUIRE(std::abs(s->k_S2 - s->k_S3) > 1e-12);

      // ---- THE POSITIVE CONTROL, at both policies ------------------------------------
      // S3 IS B-S's rung combination, so it must not merely share B-S's sign -- it must
      // reproduce its VALUE. At gygi: +1.9648e-03 vs +1.9517e-03, 0.67 %. The residual is
      // the G-response feedback (the B-L arm's G additionally carries P^{C,L}), not a
      // difference of diagram. Every reading here rests on this.
      REQUIRE(s->x_BS * s->x_S3 > 0.0);
      REQUIRE(std::abs(s->x_S3 - s->x_BS) < 0.05 * std::abs(s->x_BS));
      // ... and S3, being B-S, is positive at both policies.
      REQUIRE(s->x_S3 > 0.0);

      // ---- THE PAIR-SYMMETRY DEGENERACY (a new gate, not a restatement) --------------
      // S1 = W0_x dW_y and S2 = dW_x W0_y are the two orderings of ONE rung pair, so the
      // self-inverse-transfer relation W_PQ(q) = W_QP(-q) (see rung_pair_symmetry) forces
      // them to contribute EQUALLY to the energy. Measured |S1 - S2| = 2.6e-14 on an
      // |S1 + S2| of 6.3e-03 -- 4e-12 relative, i.e. FP accumulation, not a coincidence.
      // ⚠ Agreement this exact is normally the stale-artifact tell (trap 6). It is not one
      // here, and the LIVENESS block above carries the burden of proof: the same gate that
      // leaves S1 and S2 degenerate moves S3 by 5.1e-03. A break here means the mixed-term
      // routing has lost the pair symmetry -- the exact failure mode behind the retracted
      // "eq:mixgw is wrong".
      REQUIRE(std::abs(s->x_S1 - s->x_S2) < 1e-9);
    }

    // ---- THE RESULT, at gygi ----------------------------------------------------------
    // B-L's shift is negative, the static reduction is positive (it is B-S), and BOTH mixed
    // reductions are negative and together outweigh it. That is the sign flip, located.
    REQUIRE(g.d_full < 0.0);
    REQUIRE(g.x_all < 0.0);
    REQUIRE(g.x_S1 < 0.0);
    REQUIRE(g.x_S2 < 0.0);
    REQUIRE(std::abs(g.x_S1 + g.x_S2) > std::abs(g.x_S3));

    // ---- THE EXPANSION PARAMETER IS NOT THE WHOLE STORY -------------------------------
    // B-L is FIRST order in dW and S3 is ZEROTH, so naive power counting predicts
    // |S1+S2|/|S3| ~ 2 * max_nu|dW|/|W0|. At gygi that is 2 * 0.2775 = 0.56 against a
    // measured 3.23 -- a ~6x amplification that magnitude alone does not explain. (The
    // i.nu = 0 meter the log has always carried reads 0.02-0.06 and would have made this
    // look like ~100x; it is measured at the one frequency where dW vanishes by
    // construction, which is why diag_dw_rel exists.) Recorded, not explained here.
    REQUIRE(g.dw_rel > 0.0);
    REQUIRE(g.ratio > 2.0 * g.dw_rel);
#endif
  }

  // ======================================================================================
  // P0.1: DOES THE GAMMA HEAD CANCEL IN  dW = W - W0  ?
  //
  // vertex_bl_mixed_term_split (above) located the B-S/B-L sign flip in the MIXED (first-
  // order) reductions S1/S2 and showed the gygi Gamma head is what flips them: it moves
  // |S1+S2|/|S3| from 0.035 to 3.228 (93x) while moving the max-norm expansion parameter
  // max_nu|dW|/|W0| by only 1.81x (0.153 -> 0.278). So MAGNITUDE does not explain it, and
  // "the head does it" is still a correlation, not a mechanism.
  //
  // THE METERS: h_A := chi^dag A(Gamma) chi / ||chi||^2 with chi = thc.basis_head(), i.e.
  // the G = 0 direction the rank-1 head is BUILT from, so this is its own channel.
  //   diag_dw_head_rel() = max_nu |h_dW| / |h_W0|          (the ratio)
  //   diag_dw_head_bg()  = the same at the worst q != Gamma -- a WITHIN-RUN head-free
  //                        control, since no head is ever inserted away from Gamma
  //   diag_dw_head_abs() = max_nu |h_dW| in a.u.           (comparable across policies)
  //   diag_dw_head_coh() = that over max|dW(Gamma)|        (alignment with the rank-1 dir)
  //
  // ⚠ THE RATIO IS NOT THE ANSWER, AND THE FIRST READING OF THIS TEST SAID SO. The prior was
  // that the head enters W with weight eps^-1_00(q->0,i.nu) and W0 with only its static
  // weight, so the ratio would run to eps_M - 1 >> 1. Measured: 0.408 at Gamma with the head,
  // 0.016 without -- but the head-free q != Gamma control reads 0.39-0.41 at BOTH policies.
  // ~0.4 is just what dW/W0 looks like in the G = 0 channel anywhere on this mesh, so the
  // head does not make the RATIO anomalous; ignore_g0 makes Gamma anomalously QUIET (0.016),
  // because with v(G = 0) zeroed there is hardly any G = 0 content there to fluctuate.
  //
  // WHAT THE HEAD DOES CHANGE is the ABSOLUTE content of that one coherent rank-1 direction
  // (madelung ~ 1/q^2) and hence the COHERENCE of dW -- and the kernel sums that direction
  // over N_p^2 terms IN PHASE. That is the same mechanism as the [SANDWICH] and head-
  // projection findings, and it is exactly what a max-norm gate cannot see (trap 2): between
  // the two policies max|dW|/|W0| moves 1.81x while |S1+S2|/|S3| moves 93x.
  //
  // THREE ARMS, so the reading cannot be an artifact of the meter itself:
  //   B-L @ gygi      -- the head is present
  //   B-L @ ignore_g0 -- no head at all (the policy A/B of [blmixed], same protocol)
  //   B-S @ gygi      -- no dW exists, so both meters must stay at -1 (NEVER MEASURED,
  //                      deliberately distinct from a measured 0; trap 7 in reverse)
  //
  // HIDDEN ([.]). Run with:
  //   cd build/tests/bin && OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  //     mpiexec -n 2 --oversubscribe ./test_methods_vertex_static_e2e "[bldwhead]"
  TEST_CASE("vertex_bl_dw_head_channel", "[.][methods][vertex][static][bldwhead]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_bl_dw_head_channel skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_bldwhead";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    struct dw_t {
      double e_corr;
      double dw_rel;     // max-norm expansion parameter, ALL i.nu
      double head_rel;   // the same in the chi channel at q = Gamma
      double head_bg;    // ... and at the worst q != Gamma (no head there)
      double head_abs;   // max_nu |chi^dag dW(Gamma) chi| / ||chi||^2 in a.u.
      double head_coh;   // alignment with the rank-1 direction, 1 = dW(Gamma) IS chi chi^dag
      double head_nu0;   // the same channel at i.nu = 0 -- where the log's old meter looked
    };

    // Same protocol as [blmixed]: LiH-222, C = [1,3), 2 COLD iterations, gw and scr_coulomb
    // pinned at ignore_g0 so the only thing the policy argument moves is the VERTEX's own
    // q -> 0 treatment, i.e. whether the Gamma cell carries the analytic rank-1 head.
    auto run = [&](std::string const &rung, std::string const &policy) -> dw_t {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(1, 3), mf->nbnd(), policy,
                            "global", -1, 1e-8, -1.0, -1.0, rung);
      scr_eri.set_vertex(&vtx);
      gw.set_vertex(&vtx);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     2, false, 1e-9, true);
      (void) e_hf;
      mpi_context->comm.barrier();
      return {e_corr, vtx.diag_dw_rel(), vtx.diag_dw_head_rel(), vtx.diag_dw_head_bg(),
              vtx.diag_dw_head_abs(), vtx.diag_dw_head_coh(), vtx.diag_dw_head_nu0()};
    };

    const auto g = run("linear", "gygi");
    const auto n = run("linear", "ignore_g0");
    const auto s = run("static", "gygi");

    app_log(1, "\n========== P0.1: DOES THE GAMMA HEAD CANCEL IN dW = W - W0? ==========");
    app_log(1, "  LiH-222, C = [1,3), 2 COLD iterations, meters from the LAST eval_Sigma_C.");
    app_log(1, "  chi = thc.basis_head() (the G = 0 components of the aux basis) -- the");
    app_log(1, "  direction the rank-1 head is built from, so this is ITS channel.");
    app_log(1, "  quantity                                gygi (head)   ignore_g0 (no head)");
    app_log(1, "    max-norm      max_nu |dW| / |W0|      {:11.4f}   {:11.4f}",
            g.dw_rel, n.dw_rel);
    app_log(1, "    CHI CHANNEL   at q = Gamma            {:11.4f}   {:11.4f}",
            g.head_rel, n.head_rel);
    app_log(1, "    CHI CHANNEL   worst q != Gamma        {:11.4f}   {:11.4f}",
            g.head_bg, n.head_bg);
    app_log(1, "    Gamma / (worst other q)               {:11.2f}   {:11.2f}",
            (g.head_bg > 0.0 ? g.head_rel / g.head_bg : 0.0),
            (n.head_bg > 0.0 ? n.head_rel / n.head_bg : 0.0));
    app_log(1, "  ---- and the two that carry the result ----");
    app_log(1, "    ABSOLUTE  |h_dW(Gamma)|  [a.u.]       {:11.4e}   {:11.4e}   -> {:.1f}x",
            g.head_abs, n.head_abs,
            (n.head_abs > 0.0 ? g.head_abs / n.head_abs : 0.0));
    app_log(1, "    COHERENCE (1 = pure chi chi^dag)      {:11.3f}   {:11.3f}   -> {:.1f}x",
            g.head_coh, n.head_coh,
            (n.head_coh > 0.0 ? g.head_coh / n.head_coh : 0.0));
    app_log(1, "  ---- and why the log's i.nu = 0 meter could never see any of it ----");
    app_log(1, "    |h_dW| at i.nu = 0    [a.u.]          {:11.4e}   {:11.4e}",
            g.head_nu0, n.head_nu0);
    app_log(1, "    the nu = 0 slice understates by       {:11.1f}x  {:11.1f}x",
            (g.head_nu0 > 0.0 ? g.head_abs / g.head_nu0 : 0.0),
            (n.head_nu0 > 0.0 ? n.head_abs / n.head_nu0 : 0.0));
    app_log(1, "    e_corr                                {:11.6f}   {:11.6f}",
            g.e_corr, n.e_corr);
    app_log(1, "  B-S control @ gygi: dw_rel = {:.4f}, head_rel = {:.4f}, head_bg = {:.4f} "
               "(-1 = NEVER MEASURED, as it must be: B-S has no dW)",
            s.dw_rel, s.head_rel, s.head_bg);
    app_log(1, "  Reference from [blmixed] on the same protocol: |S1+S2|/|S3| = 3.228 "
               "(gygi) vs 0.035 (ignore_g0).");
    app_log(1, "======================================================================\n");

    REQUIRE(std::isfinite(g.e_corr));
    REQUIRE(std::isfinite(n.e_corr));

    // ---- THE B-S CONTROL: the meter is B-L-only and did not fire where there is no dW ----
    // -1 is the NEVER-MEASURED sentinel and is deliberately distinct from a measured 0. If
    // this ever reads >= 0, the meter is picking something up on a path that has no dW and
    // every number above is suspect.
    REQUIRE(s.dw_rel == -1.0);
    REQUIRE(s.head_rel == -1.0);
    REQUIRE(s.head_bg == -1.0);
    REQUIRE(s.head_abs == -1.0);
    REQUIRE(s.head_coh == -1.0);
    REQUIRE(s.head_nu0 == -1.0);

    // ---- LIVENESS: the meter really ran on both B-L arms (trap 7) -------------------------
    // basis_head can be absent on some ERI read paths, in which case the meter silently
    // skips and returns -1 -- which would make every comparison below vacuous.
    REQUIRE(g.head_rel > 0.0);
    REQUIRE(n.head_rel > 0.0);
    REQUIRE(g.head_bg > 0.0);
    REQUIRE(n.head_bg > 0.0);
    REQUIRE(g.dw_rel > 0.0);
    REQUIRE(n.dw_rel > 0.0);
    REQUIRE(g.head_abs > 0.0);
    REQUIRE(n.head_abs > 0.0);
    REQUIRE(g.head_coh > 0.0);
    REQUIRE(n.head_coh > 0.0);

    // ---- THE HEAD-FREE CONTROL IS WHAT MAKES THE RATIO UNINTERPRETABLE -------------------
    // ~0.4 in the G = 0 channel is generic: the worst q != Gamma sits there at BOTH policies,
    // including the one with no head anywhere on the mesh. Pinned loosely (it is a physical
    // screening ratio, not a constant) but tightly enough that a collapse or a blow-up of the
    // control -- which would invalidate every comparison here -- fails the test.
    REQUIRE(g.head_bg > 0.15);
    REQUIRE(g.head_bg < 1.0);
    REQUIRE(n.head_bg > 0.15);
    REQUIRE(n.head_bg < 1.0);
    // ... and with the head, Gamma is UNREMARKABLE in ratio: it joins the other q. Without
    // it, Gamma is the outlier -- anomalously QUIET, not loud. Whatever the head does, it
    // does not show up here, and any future claim resting on this ratio must clear this.
    REQUIRE(g.head_rel > 0.5 * g.head_bg);
    REQUIRE(n.head_rel < 0.2 * n.head_bg);

    // ---- WHAT THE HEAD ACTUALLY CHANGES: the ABSOLUTE coherent content -------------------
    // Same system, same basis, same iteration count -- so these two are directly comparable
    // and the ratio between them is the head's whole effect on the rung B-L expands in.
    // Measured 8.3877e-03 vs 4.9288e-06 = 1702x, against 1.81x for max|dW|/|W0| and 93x for
    // |S1+S2|/|S3|. Gated at 100x: far above any plausible drift, far below the measurement.
    REQUIRE(g.head_abs > 100.0 * n.head_abs);
    // ... and it is COHERENT. 1 means dW(Gamma) IS the chi-aligned rank-1 matrix c chi
    // chi^dag; 1/N_p means no alignment at all. Measured 0.980 vs 0.009 -- i.e. WITH the
    // head, dW(Gamma) is 98 % that rank-1 object, and the head therefore does not cancel in
    // W - W0 in any meaningful sense. This is the number no max-norm gate can see (trap 2),
    // and it is the mechanism the mixed terms S1/S2 amplify by summing N_p^2 terms in phase.
    REQUIRE(g.head_coh > 0.9);
    REQUIRE(n.head_coh < 0.05);
    // ---- AND WHY EVERY EARLIER METER MISSED IT ------------------------------------------
    // The head channel is smallest at i.nu = 0 -- dW vanishes there by construction, W0 IS
    // that slice -- and grows monotonically to the mesh cutoff, where W -> v (bare,
    // unscreened). So the |W(q,0) - W0(q)| line the run log has carried since the beginning
    // was looking at the one frequency where the effect is absent: it understates the head
    // channel by 13.6x at gygi and 428x at ignore_g0. Same defect as the max-norm dw meter
    // that diag_dw_rel replaced, in the other axis.
    REQUIRE(g.head_nu0 > 0.0);
    REQUIRE(n.head_nu0 > 0.0);
    REQUIRE(g.head_abs > 5.0 * g.head_nu0);
    REQUIRE(n.head_abs > 5.0 * n.head_nu0);
#endif
  }

  // ======================================================================================
  // GATE 0 OF THE H1 REPAIR (notes/bl_head_balance_theory_and_plan.md section 5):
  // vertex_bl_head_static_all gives every W input of the B-L functional W0's STATIC
  // Gamma-head weight (instantaneous slot; no dynamic-slot head), so the fluctuation
  // dW = W - W0 carries NO analytic head. The theory analysis says the whole gygi B-L
  // anomaly is that head: a first-order functional handed an O(1) coherent rank-1
  // fluctuation. Falsifiable predictions asserted here:
  //   P1  dW(Gamma)'s coherent head content COLLAPSES (diag_dw_head_abs ~1702x down,
  //       coherence 0.98 -> ignore_g0 class ~0.01);
  //   P2  B-S is BIT-IDENTICAL under the knob (its only W input is W0);
  //   P3  B-L's vertex shift returns to the healthy class: POSITIVE, B-S-magnitude,
  //       sign-AGREEING with B-S (the sign disagreement was the head's doing);
  //   P4  the max-norm expansion parameter dw_rel falls toward its head-free value.
  // If P1 or P3 fail, the imbalance mechanism of the theory note is WRONG -- stop and
  // re-derive from the failing meter (that is what makes this a gate).
  //
  // HIDDEN ([.]), 4 runs, ~10 min. Run with:
  //   cd build/tests/bin && OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  //     mpiexec -n 2 --oversubscribe ./test_methods_vertex_static_e2e "[blh1]"
  TEST_CASE("vertex_bl_head_static_all_gate0", "[.][methods][vertex][static][blh1]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_bl_head_static_all_gate0 skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_blh1";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    struct h1_t {
      double e_corr;
      double dw_rel;     // max-norm expansion parameter, ALL i.nu
      double head_abs;   // max_nu |chi^dag dW(Gamma) chi| / ||chi||^2 in a.u.
      double head_coh;   // alignment with the rank-1 direction, 1 = dW(Gamma) IS chi chi^dag
    };

    // Same protocol as [blmixed]/[bldwhead]: LiH-222, C = [1,3), 2 COLD iterations, gw and
    // scr_coulomb pinned at ignore_g0 so the only mover is the VERTEX's own q -> 0 policy.
    auto run = [&](std::string const &rung, std::string const &policy, bool h1) -> h1_t {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(1, 3), mf->nbnd(), policy,
                            "global", -1, 1e-8, -1.0, -1.0, rung);
      vtx.set_bl_head_static_all(h1);
      scr_eri.set_vertex(&vtx);
      gw.set_vertex(&vtx);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     2, false, 1e-9, true);
      (void) e_hf;
      mpi_context->comm.barrier();
      return {e_corr, vtx.diag_dw_rel(), vtx.diag_dw_head_abs(), vtx.diag_dw_head_coh()};
    };

    // The plain-scGW reference and the two gygi baselines of this exact protocol, on file
    // since [blheadab]/[blmixed] and re-pinned by [blhlambda] (lambda = 1 pin).
    const double E_SCGW = -0.096212991792;
    const double E_BS_G = -0.095014419061;   // B-S @ gygi     -> d = +1.1986e-03
    const double E_BL_G = -0.100077224200;   // B-L @ gygi     -> d = -3.8642e-03

    const auto bs_off = run("static", "gygi", false);
    const auto bs_on  = run("static", "gygi", true);
    const auto bl_off = run("linear", "gygi", false);
    const auto bl_on  = run("linear", "gygi", true);

    const double d_bs  = bs_off.e_corr - E_SCGW;
    const double d_off = bl_off.e_corr - E_SCGW;
    const double d_on  = bl_on.e_corr - E_SCGW;

    app_log(1, "\n========== GATE 0: THE H1 STATIC-HEAD VERTEX (delta W_head == 0) ==========");
    app_log(1, "  LiH-222, C = [1,3), 2 COLD iterations, meters from the LAST eval_Sigma_C.");
    app_log(1, "  arm                      e_corr          d(e_corr)      dw_rel   head_abs     head_coh");
    app_log(1, "    B-S gygi   H1 off   {:.12f}   {:+.4e}   {:7.4f}  {:10.4e}  {:8.3f}",
            bs_off.e_corr, d_bs, bs_off.dw_rel, bs_off.head_abs, bs_off.head_coh);
    app_log(1, "    B-S gygi   H1 ON    {:.12f}   {:+.4e}   (must be bit-identical to the row above)",
            bs_on.e_corr, bs_on.e_corr - E_SCGW);
    app_log(1, "    B-L gygi   H1 off   {:.12f}   {:+.4e}   {:7.4f}  {:10.4e}  {:8.3f}",
            bl_off.e_corr, d_off, bl_off.dw_rel, bl_off.head_abs, bl_off.head_coh);
    app_log(1, "    B-L gygi   H1 ON    {:.12f}   {:+.4e}   {:7.4f}  {:10.4e}  {:8.3f}",
            bl_on.e_corr, d_on, bl_on.dw_rel, bl_on.head_abs, bl_on.head_coh);
    app_log(1, "  head_abs collapse: {:.1f}x   coherence: {:.3f} -> {:.3f}   d(B-L,H1)/d(B-S) = {:.3f}",
            (bl_on.head_abs > 0.0 ? bl_off.head_abs / bl_on.head_abs : -1.0),
            bl_off.head_coh, bl_on.head_coh, (d_bs != 0.0 ? d_on / d_bs : 0.0));
    app_log(1, "  References: B-L @ ignore_g0 d = +1.1343e-03; [blmixed] |S1+S2|/|S3| = 3.228 (gygi).");
    app_log(1, "============================================================================\n");

    // ---- HARNESS INTEGRITY: the H1-off arms reproduce the numbers on file ----------------
    REQUIRE(std::abs(bs_off.e_corr - E_BS_G) < 1e-8);
    REQUIRE(std::abs(bl_off.e_corr - E_BL_G) < 1e-8);
    // ... and the baseline pathology is present in the off arm (else the collapse below is
    // vacuous -- trap 7, assert the knob HAD something to remove).
    REQUIRE(bl_off.head_abs > 5.0e-3);
    REQUIRE(bl_off.head_coh > 0.9);

    // ---- P2: B-S IS BIT-IDENTICAL UNDER THE KNOB -----------------------------------------
    // B-S's only W input is W0 (need_dyn false: no H1 site is ever reached). Any difference
    // means the knob leaks outside the B-L functional.
    REQUIRE(bs_on.e_corr == bs_off.e_corr);
    REQUIRE(bs_on.dw_rel == -1.0);        // B-S never measures the dW meters
    REQUIRE(bs_on.head_abs == -1.0);
    REQUIRE(bs_on.head_coh == -1.0);

    // ---- P1: THE COHERENT HEAD CONTENT OF dW(Gamma) COLLAPSES ----------------------------
    // With the head part of the expansion point instead of the fluctuation, dW's chi channel
    // must fall to the head-free (body) class. MEASURED 2026-08-01: 8.3877e-03 -> 4.7662e-06
    // (1759.8x, right at ignore_g0's 4.9288e-06), coherence 0.980 -> 0.009. The knob
    // controls the channel COMPLETELY. Gates far from both sides.
    REQUIRE(bl_on.head_abs >= 0.0);                       // measured, not the -1 sentinel
    REQUIRE(bl_on.head_abs < 1.0e-4);
    REQUIRE(bl_off.head_abs > 50.0 * bl_on.head_abs);
    REQUIRE(bl_on.head_coh < 0.15);

    // ---- 🚨 THE GATE-0 VERDICT (2026-08-01): THE ENERGY DOES NOT CARE -------------------
    // The theory note's section-2 prediction (P3: "remove the fluctuation head and the
    // vertex shift returns to the B-S class, positive") is REFUTED BY THIS MEASUREMENT:
    //     d(B-L, H1 on)  = -3.8835e-03     d(B-L, H1 off) = -3.8642e-03
    // i.e. with dW's coherent head channel PROVABLY EMPTY (the P1 gates above), the energy
    // stays at the anomalous sign-flipped value. The fluctuation head is worth -1.9e-05 of
    // a -5.0e-03 effect; the damage is carried by the STATIC head weight eps^-1(0) in
    // B-L's W0-consuming structures (H1-vs-ignore_g0 = -5.02e-03 = the whole lambda-scan
    // damage). The lambda scan could not separate the two -- both are linear in lambda --
    // and P0.1's coherence spotlight was on the wrong object. Localization: [blh1split].
    // These gates now pin the MEASURED behavior so the falsification stays on record; if
    // the energy ever starts responding to the fluctuation head, this fires and the
    // investigation reopens.
    REQUIRE(std::abs(bl_on.e_corr - bl_off.e_corr) < 1.0e-3);   // energy-insensitive
    REQUIRE(d_on < 0.0);                                        // still sign-flipped
    REQUIRE(std::abs(d_on - d_off) < 0.2 * std::abs(d_off));    // same anomaly class
    (void) d_bs;

    // ---- P4: THE EXPANSION PARAMETER FALLS TOWARD ITS HEAD-FREE VALUE --------------------
    // Directional gate only (the max-norm is body-dominated; ignore_g0 measured 0.1533
    // against gygi's 0.2775; H1 measured 0.0948).
    REQUIRE(bl_on.dw_rel > 0.0);
    REQUIRE(bl_on.dw_rel < 0.9 * bl_off.dw_rel);
#endif
  }

  // ======================================================================================
  // GATE 0.5: WHERE DOES THE STATIC-HEAD DAMAGE LIVE? The [blh1] measurement moved the
  // question: the -5.0e-03 gygi damage is NOT dW's coherent head (removing it moves the
  // energy -1.9e-05); it is proportional to the STATIC head weight eps^-1(0) carried by
  // B-L's W0-consuming structures. Under H1 those are, exhaustively:
  //   (i)   S1/S2's static rung  (W0(Gamma) head x the dW BODY convolution),
  //   (ii)  S3 = B-S's own kernel term (known healthy, +1.9e-03 class),
  //   (iii) P^{C,L} = Pi^{C,0}[W0-head rung] injected into the Dyson (enters every arm
  //         through Sigma_GW; the drop knob never touches it),
  //   (iv)  Sigma^{L,r} via the W0(Gamma)-head sandwich Delta w^L = W0 [X^L] W0.
  // This case decomposes the H1 theory with the SAME machinery as [blmixed]/[bldecomp]:
  // the S1/S2/S3 slot probe x the bl_drop knob, all arms at gygi + H1. Ablations with
  // feedback (trap 8): SIGN-level statements only, ~17 % nonlinearity.
  // On-file comparison columns ([blmixed], same protocol, no H1):
  //   gygi      : S3 +1.9648e-03, S1 = S2 = -3.1708e-03, |S1+S2|/|S3| = 3.228
  //   ignore_g0 : S3 +1.8911e-03, S1 = S2 = +3.2730e-05, |S1+S2|/|S3| = 0.035
  //
  // HIDDEN ([.]), 7 runs, ~18 min. Run with:
  //   cd build/tests/bin && OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  //     mpiexec -n 2 --oversubscribe ./test_methods_vertex_static_e2e "[blh1split]"
  TEST_CASE("vertex_bl_h1_static_head_split", "[.][methods][vertex][static][blh1split]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_bl_h1_static_head_split skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_blh1split";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    // [blmixed]'s run_impl + the H1 knob. All arms: B-L, gygi, H1 ON, 2 COLD iterations.
    auto run = [&](int drop, int only) -> double {
      solvers::vertex_detail::sigma_C_slot_probe.only_term = only;
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(1, 3), mf->nbnd(), "gygi",
                            "global", -1, 1e-8, -1.0, -1.0, "linear");
      vtx.set_bl_head_static_all(true);
      vtx.set_bl_drop(drop);
      scr_eri.set_vertex(&vtx);
      gw.set_vertex(&vtx);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     2, false, 1e-9, true);
      (void) e_hf;
      mpi_context->comm.barrier();
      solvers::vertex_detail::sigma_C_slot_probe.clear();
      return e_corr;
    };

    const double E_SCGW  = -0.096212991792;   // plain scGW, this protocol, on file
    const double E_FULL0 = -0.100096475753;   // [blh1]'s bl_on: full B-L @ gygi + H1

    const double e_full = run(0, 0);   // everything
    const double e_d3   = run(3, 0);   // P^{C,L} via Sigma_GW alone (the shares' baseline)
    const double e_d2   = run(2, 0);   // without Sigma^{C,x}  (P^{C,L} + Sigma^{L,r})
    const double e_d1   = run(1, 0);   // without Sigma^{L,r}  (P^{C,L} + full Sigma^{C,x})
    const double e_s1   = run(1, 1);   // ... Sigma^{C,x} restricted to S1
    const double e_s2   = run(1, 2);   // ... restricted to S2
    const double e_s3   = run(1, 3);   // ... restricted to S3

    const double sh_pcl = e_d3 - E_SCGW;    // P^{C,L}-via-Sigma_GW share (vs plain scGW)
    const double sh_lr  = e_d2 - e_d3;      // Sigma^{L,r} share
    const double sh_cx  = e_d1 - e_d3;      // full Sigma^{C,x} share
    const double S1 = e_s1 - e_d3, S2 = e_s2 - e_d3, S3 = e_s3 - e_d3;

    app_log(1, "\n========== GATE 0.5: THE STATIC-HEAD DAMAGE, LOCALIZED (all arms H1) ==========");
    app_log(1, "  B-L @ gygi + H1, LiH-222, C = [1,3), 2 COLD iterations. Shares are");
    app_log(1, "  ABLATIONS WITH FEEDBACK (trap 8): signs and orders only.");
    app_log(1, "    full theory            e = {:.12f}   d = {:+.4e}", e_full, e_full - E_SCGW);
    app_log(1, "    P^(C,L) alone (drop 3) e = {:.12f}   share = {:+.4e}", e_d3, sh_pcl);
    app_log(1, "    Sigma^(L,r)  share (drop2 - drop3)   = {:+.4e}", sh_lr);
    app_log(1, "    Sigma^(C,x)  share (drop1 - drop3)   = {:+.4e}", sh_cx);
    app_log(1, "    S1 = W0_x dW_y share                 = {:+.4e}", S1);
    app_log(1, "    S2 = dW_x W0_y share                 = {:+.4e}", S2);
    app_log(1, "    S3 = W0_x W0_y share                 = {:+.4e}", S3);
    app_log(1, "    |S1 + S2| / |S3|                     = {:.4f}",
            (S3 != 0.0 ? std::abs(S1 + S2) / std::abs(S3) : -1.0));
    app_log(1, "  On-file columns, same protocol, NO H1 ([blmixed]):");
    app_log(1, "    gygi:      S3 +1.9648e-03, S1 = S2 = -3.1708e-03, ratio 3.228");
    app_log(1, "    ignore_g0: S3 +1.8911e-03, S1 = S2 = +3.2730e-05, ratio 0.035");
    app_log(1, "  READ:  S1/S2 still ~ -3e-03 -> the damage is W0's STATIC head x the dW BODY");
    app_log(1, "         (the mixed terms' Gamma_x cell). S1/S2 collapsed to +3e-05 class ->");
    app_log(1, "         the damage lives in P^(C,L) injection and/or Sigma^(L,r) instead.");
    app_log(1, "===============================================================================\n");

    REQUIRE(std::isfinite(e_full));
    REQUIRE(std::isfinite(e_d3));
    REQUIRE(std::isfinite(e_d2));
    REQUIRE(std::isfinite(e_d1));
    REQUIRE(std::isfinite(e_s1));
    REQUIRE(std::isfinite(e_s2));
    REQUIRE(std::isfinite(e_s3));
    // Harness integrity: the full arm reproduces [blh1]'s bl_on number.
    REQUIRE(std::abs(e_full - E_FULL0) < 1e-8);
    // Liveness of the slot probe (trap 7): restricting to S3 must differ from full Sig^{C,x}.
    REQUIRE(std::abs(e_s3 - e_d1) > 1e-6);
    // The pair-symmetry gate: the two orderings of one rung pair contribute equally
    // (W_PQ(q) = W_QP(-q) end to end; [blmixed] measured 2.6e-14).
    REQUIRE(std::abs(e_s1 - e_s2) < 1e-9);
#endif
  }

  // ======================================================================================
  // GATE 1 OF THE H1 REPAIR: THE PARENT DISCRIMINATOR
  // (notes/bl_head_balance_theory_and_plan.md section 5, Gate 1.)
  //
  // The PARENT theory (dynamic rung) = B-L + Phi^(2), with the IDENTICAL gygi head
  // treatment. The theory note's order counting (its section 2.2) shows the Phi^(2) that
  // B-L discards contains head x body cross terms that are ALSO first order in the head
  // -- the lambda scan cannot separate them from B-L's kept O(lambda) terms -- so B-L's
  // anomalous linear head response says nothing about the parent until the parent is
  // measured. The discriminating number is the head GAP at fixed everything,
  //     gap := d(e_corr)[gygi] - d(e_corr)[ignore_g0],
  // which partially cancels the parent's known pole-fit defect (the same-binary,
  // same-protocol difference). Two outcomes, both informative:
  //   (a) gap is B-S-class (small):  the cross terms CANCEL B-L's kept head term; the
  //       gygi B-L anomaly is pure truncation; H1 is not merely balanced but ACCURATE,
  //       and H2's value is only the (small) retarded-head content.
  //   (b) gap is large/negative (B-L-like):  head dominance of the Gamma cell is REAL
  //       parent physics at N_k = 8 (the 2026-07-16 caveat measured the head flipping
  //       Phi_2^C's sign at this mesh) -- a program-wide coarse-mesh issue; production
  //       stays on H1/ignore_g0 and any retarded-head claim needs H2 + the rusty ladder.
  //
  // This is a MEASUREMENT, not a prediction: no outcome assertion beyond finiteness and
  // liveness. Read the verdict from the logged table. CAVEATS: 2 cold iterations (signs
  // and orders only, trap 10), and the parent's dynamic path carries the pre-existing
  // pole-fit defect (notes/pole_fit_repair_STATUS.md) -- do not quote absolute energies.
  //
  // HIDDEN ([.]), 2 parent runs (the dconv kernel makes these the slow ones). Run with:
  //   cd build/tests/bin && OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  //     mpiexec -n 2 --oversubscribe ./test_methods_vertex_static_e2e "[blparent]"
  TEST_CASE("vertex_parent_head_gap", "[.][methods][vertex][static][blparent]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_parent_head_gap skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_blparent";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    // Same protocol as [blmixed]/[bldwhead]/[blh1]: LiH-222, C = [1,3), 2 COLD
    // iterations, gw and scr_coulomb pinned at ignore_g0.
    auto run = [&](std::string const &rung, std::string const &policy) -> double {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(1, 3), mf->nbnd(), policy,
                            "global", -1, 1e-8, -1.0, -1.0, rung);
      scr_eri.set_vertex(&vtx);
      gw.set_vertex(&vtx);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     2, false, 1e-9, true);
      (void) e_hf;
      mpi_context->comm.barrier();
      return e_corr;
    };

    const double E_SCGW = -0.096212991792;   // plain scGW, this protocol, on file
    const double p_g  = run("dynamic", "gygi");
    const double p_n  = run("dynamic", "ignore_g0");
    const double d_g  = p_g - E_SCGW;
    const double d_n  = p_n - E_SCGW;
    const double gap  = d_g - d_n;

    app_log(1, "\n========== GATE 1: THE PARENT (DYNAMIC-RUNG) HEAD GAP ==========");
    app_log(1, "  LiH-222, C = [1,3), 2 COLD iterations. On-file companions, same protocol:");
    app_log(1, "    B-S  d = +1.1986e-03 (gygi) / +1.1529e-03 (ig0)  -> gap = +4.57e-05");
    app_log(1, "    B-L  d = -3.8642e-03 (gygi) / +1.1343e-03 (ig0)  -> gap = -5.00e-03");
    app_log(1, "  PARENT:");
    app_log(1, "    e_corr(gygi)      = {:.12f}   d = {:+.4e}", p_g, d_g);
    app_log(1, "    e_corr(ignore_g0) = {:.12f}   d = {:+.4e}", p_n, d_n);
    app_log(1, "    gap               = {:+.4e}   gap/gap_BS = {:+.1f}   gap/gap_BL = {:+.3f}",
            gap, gap / 4.57e-05, gap / -5.00e-03);
    app_log(1, "  VERDICT KEY: |gap| ~ 5e-05 class -> (a) B-L's anomaly is pure truncation");
    app_log(1, "               (H1 accurate; the Phi^(2) cross terms cancel the head).");
    app_log(1, "               gap ~ -5e-03 class  -> (b) head dominance is REAL parent");
    app_log(1, "               physics at N_k = 8 (program-wide coarse-mesh issue).");
    app_log(1, "================================================================\n");

    REQUIRE(std::isfinite(p_g));
    REQUIRE(std::isfinite(p_n));
    // LIVENESS (trap 7): the two arms must differ -- at ignore_g0 there is no head at all,
    // so equality would mean the gygi arm never inserted one (a config null, not physics).
    REQUIRE(std::abs(p_g - p_n) > 1.0e-9);
#endif
  }

  // ======================================================================================
  // P0.2: IS THE Gamma HEAD A LEGITIMATE ACCELERATOR, OR DOES IT MOVE THE N_k -> inf LIMIT?
  //
  // The analytic head insertion is a FINITE-SIZE device: "gygi" and "ignore_g0" differ only in
  // how the q = Gamma microcell of a q-sum is treated, and both must converge to the SAME
  // answer as N_k -> inf. That is the whole justification for inserting it. So the quantity to
  // watch is not either arm but the GAP between them,
  //     gap(N_k) := d(e_corr)[gygi] - d(e_corr)[ignore_g0],
  // which MUST SHRINK with N_k for a legitimate accelerator. On LiH-222 the B-L gap is
  // -5.0e-03, LARGER than either arm's own vertex shift, while the B-S gap is 4.6e-05 (3.9 %).
  //
  // WHY B-L IS THE SUSPECT -- ORDER COUNTING. The head is derived for the SINGLE-rung q -> 0
  // limit of Sigma_GW, where the Gamma microcell contributes int_cell d^3q / q^2 ~ O(N_k^-1/3):
  // slowly vanishing, which is exactly why an analytic accelerator is worth having. B-L's
  // Sigma^{C,x} is a TWO-rung kernel in which the same Gamma cell can sit on BOTH transfers,
  // giving int d^3q / q^4 -- which DIVERGES in 3D (notes/head_corrections.pdf sections 2-3,
  // the same non-integrability that forces the response middle factor to have a q^2-suppressed
  // head). P0.1 then measured that the leg carrying the pathology, dW(Gamma), is 98 % a
  // coherent rank-1 chi chi^dag. B-S has the same two rungs but both STATIC, so it is the
  // built-in control: its gap is known to be small and must stay small.
  //
  // THREE OUTCOMES, ALL INFORMATIVE:
  //   (1) the B-L gap SHRINKS with N_k  -> the head is fine; P0.2 refuted, look elsewhere.
  //   (2) it stays FLAT or GROWS        -> the head insertion is biased in a two-rung kernel:
  //                                        the two policies do not share an N_k -> inf limit.
  //   (3) ignore_g0 B-L itself DRIFTS   -> the coincident-Gamma cell is genuinely
  //       systematically with N_k          non-integrable and B-L needs a real q -> 0
  //                                        treatment, not a different head.
  //
  // ⚠ WHAT THIS LOCAL LADDER CAN AND CANNOT DO. The unit-test meanfields give N_k = 1, 2 (Si)
  // and 8, 12 (LiH) -- two materials, two rungs each. That is enough to measure the DIRECTION
  // of gap(N_k) within a material; it is NOT enough for a convergence exponent, and Si and LiH
  // numbers must never be compared to each other. A real convergence statement needs larger
  // grids on rusty. ⚠ And the vertex does not scale well: 32 ranks is the measured ceiling,
  // upfold+reduce anti-scales, build_delta_w is fully replicated (see the parallel-scaling
  // section of notes/bl_head_channel_diagnosis.md).
  //
  // 🚨 `qe_si111` (N_k = 1) WOULD BE THE FREE CONTROL -- nqpts_ibz == 1 is where eval_Sigma_C
  // DOWNGRADES gygi to ignore_g0 by itself, so the two arms must come out bit-identical, a
  // positive test of the downgrade path. IT CANNOT BE RUN HERE: on a Gamma-only mesh at 2
  // ranks the ISDF fit leaves one rank with no work and aborts in
  // math::fft::create_plan_many ("howmany=0", numerics/fft/nda.hpp:174) before the vertex is
  // ever reached. That is a pre-existing THC/ISDF limitation, NOT a vertex defect, and it is
  // fatal (APP_ABORT), so it cannot be guarded around from here. Excluded deliberately.
  //
  // HIDDEN ([.]), ~40 min. Run with:
  //   cd build/tests/bin && OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  //     mpiexec -n 2 --oversubscribe ./test_methods_vertex_static_e2e "[blnkconv]"
  TEST_CASE("vertex_bl_head_nk_convergence", "[.][methods][vertex][static][blnkconv]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_bl_head_nk_convergence skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");

    struct nk_t {
      std::string sys;
      long nkpts;
      double ref;
      double bs_g, bs_n, bl_g, bl_n;        // e_corr per arm
      double head_abs, head_coh;            // the P0.1 meters, gygi arm
      double madelung;
      double d_bs_g, d_bs_n, d_bl_g, d_bl_n, gap_bs, gap_bl;
    };

    // One system, five arms: plain scGW + {B-S, B-L} x {gygi, ignore_g0}. gw and scr_coulomb
    // stay at ignore_g0 throughout, so the policy argument moves the VERTEX's q -> 0 treatment
    // and nothing else -- the same isolation as [blmixed] and [bldwhead].
    auto measure = [&](std::string const &sys) -> nk_t {
      nk_t r{};
      r.sys = sys;
      std::string output = "coqui_vertex_blnkconv";
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, sys));
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      r.nkpts = mf->nkpts();
      r.madelung = mf->madelung();

      auto run = [&](std::string const &rung, std::string const &policy,
                     double *abs_out, double *coh_out) -> double {
        solvers::hf_t hf;
        solvers::gw_t gw(&ft, "ignore_g0", output);
        solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
        simple_dyson dyson(mf.get(), &ft);
        MBState mb_state(mpi_context, ft, output);
        iter_scf::iter_scf_t iter_sol("damping");
        const bool with_vertex = not rung.empty();
        solvers::vertex_t vtx(&ft, with_vertex ? "2nd_exchange" : "none",
                              with_vertex ? nda::range(1, 3) : nda::range(0, 0), mf->nbnd(),
                              with_vertex ? policy : "ignore_g0",
                              "global", -1, 1e-8, -1.0, -1.0,
                              with_vertex ? rung : "dynamic");
        if (vtx.enabled()) { scr_eri.set_vertex(&vtx); gw.set_vertex(&vtx); }
        auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                       solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                       2, false, 1e-9, true);
        (void) e_hf;
        mpi_context->comm.barrier();
        if (abs_out) *abs_out = vtx.diag_dw_head_abs();
        if (coh_out) *coh_out = vtx.diag_dw_head_coh();
        return e_corr;
      };

      r.ref  = run("", "ignore_g0", nullptr, nullptr);
      r.bs_g = run("static", "gygi", nullptr, nullptr);
      r.bs_n = run("static", "ignore_g0", nullptr, nullptr);
      r.bl_g = run("linear", "gygi", &r.head_abs, &r.head_coh);
      r.bl_n = run("linear", "ignore_g0", nullptr, nullptr);

      r.d_bs_g = r.bs_g - r.ref;   r.d_bs_n = r.bs_n - r.ref;
      r.d_bl_g = r.bl_g - r.ref;   r.d_bl_n = r.bl_n - r.ref;
      r.gap_bs = r.d_bs_g - r.d_bs_n;
      r.gap_bl = r.d_bl_g - r.d_bl_n;
      return r;
    };

    // Ordered by N_k WITHIN a material. Si and LiH are different systems and their numbers are
    // NOT comparable to each other -- only the trend inside each pair means anything.
    std::vector<nk_t> res;
    for (auto const *sys : {"qe_si211", "qe_lih222", "qe_lih223"})
      res.push_back(measure(sys));

    app_log(1, "\n===== P0.2: DOES THE Gamma HEAD MOVE THE N_k -> inf LIMIT? =====");
    app_log(1, "  C = [1,3), 2 COLD iterations per arm, 5 arms per system.");
    app_log(1, "  gap := d(e_corr)[gygi] - d(e_corr)[ignore_g0]. A legitimate finite-size");
    app_log(1, "  accelerator must have gap -> 0 as N_k grows. B-S is the control (same two");
    app_log(1, "  rungs, both STATIC); B-L is the two-rung-with-a-dynamic-leg case.");
    app_log(1, "  system      N_k   d_BS(gygi)   d_BS(ig0)     gap_BS   |  d_BL(gygi)   "
               "d_BL(ig0)     gap_BL   gap_BL/|d_BL(ig0)|");
    for (auto const &r : res)
      app_log(1, "  {:<10} {:4}  {:+.4e}  {:+.4e}  {:+.4e}  |  {:+.4e}  {:+.4e}  {:+.4e}  "
                 "{:8.2f}",
              r.sys, r.nkpts, r.d_bs_g, r.d_bs_n, r.gap_bs, r.d_bl_g, r.d_bl_n, r.gap_bl,
              (std::abs(r.d_bl_n) > 0.0 ? std::abs(r.gap_bl) / std::abs(r.d_bl_n) : 0.0));
    app_log(1, "  ---- the P0.1 head-channel meters on the B-L gygi arm ----");
    app_log(1, "  system      N_k   madelung      |h_dW(Gamma)|   coherence (1 = pure "
               "chi chi^dag)");
    for (auto const &r : res)
      app_log(1, "  {:<10} {:4}  {:+.4e}   {:.4e}      {:.3f}",
              r.sys, r.nkpts, r.madelung, r.head_abs, r.head_coh);
    app_log(1, "  ---- ⭐ WITHOUT THE HEAD, DOES B-L TRACK B-S? ----");
    app_log(1, "  system      N_k   d_BS(ig0)    d_BL(ig0)    |difference|   relative");
    for (auto const &r : res)
      app_log(1, "  {:<10} {:4}  {:+.4e}  {:+.4e}  {:.4e}   {:8.2f} %",
              r.sys, r.nkpts, r.d_bs_n, r.d_bl_n, std::abs(r.d_bl_n - r.d_bs_n),
              (std::abs(r.d_bs_n) > 0.0
                   ? 100.0 * std::abs(r.d_bl_n - r.d_bs_n) / std::abs(r.d_bs_n) : 0.0));
    app_log(1, "  ---- the trend, WITHIN each material ----");
    for (size_t i = 1; i < res.size(); ++i) {
      if (res[i].sys.substr(0, 6) != res[i - 1].sys.substr(0, 6)) continue;
      app_log(1, "    {} (N_k {} -> {}):  gap_BS {:+.4e} -> {:+.4e} ({:.2f}x)   ||   "
                 "gap_BL {:+.4e} -> {:+.4e} ({:.2f}x)  -> {}",
              res[i].sys.substr(0, 6), res[i - 1].nkpts, res[i].nkpts,
              res[i - 1].gap_bs, res[i].gap_bs,
              (std::abs(res[i - 1].gap_bs) > 0.0
                   ? std::abs(res[i].gap_bs) / std::abs(res[i - 1].gap_bs) : 0.0),
              res[i - 1].gap_bl, res[i].gap_bl,
              (std::abs(res[i - 1].gap_bl) > 0.0
                   ? std::abs(res[i].gap_bl) / std::abs(res[i - 1].gap_bl) : 0.0),
              (std::abs(res[i].gap_bl) < std::abs(res[i - 1].gap_bl)
                   ? "B-L gap SHRINKS (accelerator behaving)"
                   : "B-L gap DOES NOT SHRINK (the two policies are not converging together)"));
    }
    app_log(1, "  ⚠ Two rungs per material only -- this measures the DIRECTION of gap(N_k),");
    app_log(1, "    NOT a convergence exponent. Si and LiH numbers are not comparable.");
    app_log(1, "===============================================================\n");

    for (auto const &r : res) {
      REQUIRE(std::isfinite(r.ref));
      REQUIRE(std::isfinite(r.bs_g));
      REQUIRE(std::isfinite(r.bl_g));
      REQUIRE(std::isfinite(r.bl_n));
    }

    // ---- LIVENESS: on every mesh here the Gamma head really fired --------------------
    // Every system in the ladder has nqpts_ibz > 1, so none of them takes the Gamma-only
    // downgrade and the head must be live in all of them. Without this a config null reads
    // exactly like a physics null (trap 7) -- which is how the head projection first
    // "showed no effect".
    for (size_t i = 0; i < res.size(); ++i) {
      REQUIRE(res[i].nkpts > 1);
      REQUIRE(res[i].head_abs > 0.0);
      REQUIRE(res[i].head_coh > 0.0);
      // ⭐ P0.1 GENERALIZES. dW(Gamma) is a near-pure rank-1 chi chi^dag on BOTH materials and
      // every mesh here (0.939 / 0.980 / 0.976), so the coherence finding is not a LiH-222
      // accident and does not wash out as the mesh is refined.
      REQUIRE(res[i].head_coh > 0.85);
    }

    // ---- ⚠ si211 IS EXCLUDED FROM THE PHYSICS GATES, AND THAT IS ITSELF A FINDING --------
    // At N_k = 2 the Gamma cell is HALF the mesh, so every arm is Gamma-dominated and the
    // whole vertex shift is ~1e-05/1e-06 -- three orders below LiH. There B-S's own gap is
    // 7.3x its own shift. So **B-S is not intrinsically head-insensitive**: its 3-4 %
    // insensitivity on LiH is a statement about the MESH, not about the diagram, and any
    // future argument of the form "B-S proves the head is harmless" must say at which N_k.
    REQUIRE(res[0].nkpts == 2);
    REQUIRE(std::abs(res[0].gap_bs) > std::abs(res[0].d_bs_n));

    // ---- THE GATES, on the LiH pair (N_k = 8, 12) ---------------------------------------
    for (size_t i = 1; i < res.size(); ++i) {
      // (a) the B-S control holds where the mesh is not Gamma-dominated: 4.0 % / 2.7 %.
      REQUIRE(std::abs(res[i].gap_bs) < 0.15 * std::abs(res[i].d_bs_n));
      // (b) and B-L's gap dwarfs it -- 109x / 123x. Same head, same mesh, same q-sum; the
      //     ONLY difference is that B-L's kernel has a dynamic rung leg for the head to
      //     survive into (P0.1: that leg is 98 % rank-1).
      REQUIRE(std::abs(res[i].gap_bl) > 20.0 * std::abs(res[i].gap_bs));
      // (c) ⭐ WITHOUT THE HEAD, B-L TRACKS B-S: 1.6 % at N_k = 8, 3.1 % at N_k = 12, and
      //     both drift a well-behaved +14-15 % over the refinement. So the head-free B-L is
      //     an ordinary, convergent quantity -- outcome (3) is NOT what is happening, and
      //     B-L is not intrinsically sick. Only the head-corrected arm is anomalous.
      REQUIRE(std::abs(res[i].d_bl_n - res[i].d_bs_n) < 0.20 * std::abs(res[i].d_bs_n));
      REQUIRE(res[i].d_bl_n * res[i].d_bs_n > 0.0);
      // (d) ... while WITH the head, B-L takes the OPPOSITE SIGN from B-S at both meshes.
      REQUIRE(res[i].d_bl_g * res[i].d_bs_g < 0.0);
      // (e) ... and the disagreement is still LARGER than the signal at the finest mesh
      //     available here: |gap_BL| / |d_BL(ig0)| = 4.41 (N_k = 8), 3.44 (N_k = 12).
      REQUIRE(std::abs(res[i].gap_bl) > 2.0 * std::abs(res[i].d_bl_n));
    }

    // ---- THE TREND -- AND WHAT IT CANNOT SETTLE -----------------------------------------
    // gap_BL goes -4.9986e-03 -> -4.4343e-03, i.e. 0.89x over N_k 8 -> 12. A legitimate
    // O(N_k^-1/3) finite-size residual would predict 0.87x, and "flat" would be 1.00x --
    // and this ladder CANNOT TELL THOSE APART: 8 -> 12 is only 1.5x in N_k (and 222 -> 223
    // refines ONE direction, not three), against a ~7 % run-to-run spread on a 2-cold-
    // iteration delta (trap 10). So the honest assertion is only that the gap does not GROW.
    // Distinguishing outcome (1) from (2) needs 2x2x2 -> 3x3x3 -> 4x4x4 on rusty, where
    // N_k^1/3 doubles. ⚠ Do not quote the 0.89x as evidence of convergence.
    REQUIRE(std::abs(res[2].gap_bl) < 1.05 * std::abs(res[1].gap_bl));
    // Meanwhile the head's own channel content does NOT fall with the mesh -- |h_dW(Gamma)|
    // goes 8.3877e-03 -> 1.3066e-02 (1.56x) while madelung falls 0.2971 -> 0.2505. Recorded,
    // not interpreted: the contribution to the energy carries a 1/N_k cell weight that this
    // meter does not include, so 1.56x against 1.5x in N_k is suggestive of a FLAT
    // contribution, not proof of one.
    REQUIRE(res[2].head_abs > res[1].head_abs);
#endif
  }

  // ======================================================================================
  // P0.3: SEPARATE THE ONE-RUNG Gamma TERM FROM THE COINCIDENT-Gamma ONE, BY SCALING THE HEAD
  //
  // The N_k ladder ([blnkconv]) could not settle whether the head moves the N_k -> inf limit:
  // its leverage is only N_k^1/3, and over 8 -> 12 a legitimate O(N_k^-1/3) residual (0.87x)
  // and a flat one (1.00x) are indistinguishable against a ~7 % delta spread. Scaling the
  // HEAD ITSELF has full leverage at FIXED mesh, FIXED G and FIXED cost.
  //
  // THE SEPARATION. Sigma^{C,x} is a TWO-rung kernel, so the Gamma cell enters it two ways:
  //   * ONE rung transfer at Gamma, the other elsewhere  -> scales as lambda
  //   * BOTH rung transfers at Gamma (the coincident cell) -> scales as lambda^2
  // so fitting  d(e_corr)(lambda) = a*lambda + b*lambda^2  over lambda in {0, 1/4, 1/2,
  // 3/4, 1} SEPARATES them. `b` is the coefficient of the coincident-Gamma cell -- the
  // int d^3q/q^4 that sections 2-3 of notes/head_corrections.pdf show is NON-INTEGRABLE in
  // 3D, unlike the single-rung int d^3q/q^2 the analytic head is actually derived for. P0.1
  // says the leg carrying the pathology (dW(Gamma)) is 98 % a coherent rank-1 chi chi^dag.
  //
  // B-S IS THE CONTROL, and it is a sharp one: B-S has the SAME two rungs and the same
  // Gamma cell, but both rungs are STATIC W0. If a large |b|/|a| showed up in B-S too, then
  // quadratic-in-head behaviour is just what a two-rung kernel does and it says nothing about
  // B-L. If B-S is essentially linear and B-L is not, the coincident-Gamma cell is entering
  // B-L through the DYNAMIC leg specifically -- which is exactly what P0/P0.1 measured.
  //
  // 🚨 MEASURED 2026-08-01 -- AND THE COINCIDENT-Gamma HYPOTHESIS IS REFUTED. The damage is
  // LINEAR in the head, not quadratic:
  //     B-L :  a = -7.1978e-03   b = +2.1677e-03   |b|/|a| = 0.301   (2-term resid 1.3 %)
  //     B-S :  a = +4.0045e-05   b = +5.7883e-06   |b|/|a| = 0.145   (2-term resid 0.5 %)
  // The lambda^2 term is only 30 % of the linear one AND CARRIES THE OPPOSITE SIGN (it
  // partially cancels), and B-L's curvature ratio is a mere 2x B-S's -- no qualitative
  // difference. So the int d^3q/q^4 coincident-Gamma cell is NOT the mechanism.
  //
  // ⭐ WHAT IS ANOMALOUS IS `a` -- THE ORDINARY ONE-RUNG-AT-Gamma TERM. B-L's linear
  // response to the head is **180x B-S's, with the opposite sign**. That is precisely what
  // P0.1 predicts: ONE head, entering ONE leg (the dynamic rung dW), surviving the W - W0
  // subtraction with near-full bare weight and leaving that leg 98 % rank-1. A single-head
  // effect is a LINEAR effect. P0.1 and P0.3 agree.
  //
  // ⇒ the open question is no longer "is the two-rung order counting wrong" but "why does a
  // SINGLE head insertion move B-L 180x more than B-S", and P0.1 has already named the
  // answer's ingredients: it survives into dW, and it is coherent.
  //
  // ⚠ THE FIT IS DESCRIPTIVE, NOT A DECOMPOSITION. e_corr is not evaluated at a common G
  // (update_G precedes it, scf_driver.cpp:188 vs :200), so a/b absorb some G-response
  // nonlinearity, and the curve need not be exactly quadratic. The claim to draw is the
  // SHAPE -- is B-L's lambda-dependence strongly curved where B-S's is not -- not the value
  // of b. The residual of the 2-term fit is reported so that curvature beyond lambda^2
  // (which would mean the picture is incomplete) is visible rather than absorbed.
  //
  // ⭐ TWO PINS MAKE THE KNOB TRUSTWORTHY, and they are the first thing this test asserts:
  //   lambda = 1  ==  today's gygi           (the knob is inert at its default)
  //   lambda = 0  ==  div_treatment ignore_g0 (it really is the head being removed, and
  //                   nothing else -- xi * 0 trips the same `xi == 0` guard every head site
  //                   already has, so lambda = 0 takes the no-head branch STRUCTURALLY)
  // Without both, a lambda scan measures an unknown mixture of the head and whatever else
  // the knob perturbed.
  //
  // HIDDEN ([.]), ~25 min. Run with:
  //   cd build/tests/bin && OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  //     mpiexec -n 2 --oversubscribe ./test_methods_vertex_static_e2e "[blhlambda]"
  TEST_CASE("vertex_bl_head_lambda_scan", "[.][methods][vertex][static][blhlambda]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_bl_head_lambda_scan skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_blhlambda";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    // Same protocol as [blmixed] / [bldwhead] / [blnkconv]: LiH-222, C = [1,3), 2 COLD
    // iterations, gw and scr_coulomb pinned at ignore_g0 so the vertex's q -> 0 treatment is
    // the only thing that moves.
    auto run = [&](std::string const &rung, std::string const &policy, double lambda,
                   double *coh_out = nullptr) -> double {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      const bool with_vertex = not rung.empty();
      solvers::vertex_t vtx(&ft, with_vertex ? "2nd_exchange" : "none",
                            with_vertex ? nda::range(1, 3) : nda::range(0, 0), mf->nbnd(),
                            with_vertex ? policy : "ignore_g0",
                            "global", -1, 1e-8, -1.0, -1.0,
                            with_vertex ? rung : "dynamic");
      vtx.set_bl_head_scale(lambda);
      REQUIRE(vtx.bl_head_scale() == lambda);
      if (vtx.enabled()) { scr_eri.set_vertex(&vtx); gw.set_vertex(&vtx); }
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     2, false, 1e-9, true);
      (void) e_hf;
      mpi_context->comm.barrier();
      if (coh_out) *coh_out = vtx.diag_dw_head_coh();
      return e_corr;
    };

    const double ref = run("", "ignore_g0", 1.0);

    // ---- PIN 1 and PIN 2 -----------------------------------------------------------------
    const double bl_g1 = run("linear", "gygi", 1.0);        // lambda = 1 at gygi
    const double bl_g0 = run("linear", "gygi", 0.0);        // lambda = 0 at gygi
    const double bl_ig = run("linear", "ignore_g0", 1.0);   // the real ignore_g0

    app_log(1, "\n===== P0.3: THE Gamma-HEAD STRENGTH SCAN =====");
    app_log(1, "  LiH-222, C = [1,3), 2 COLD iterations. plain scGW e_corr = {:.12f}", ref);
    app_log(1, "  ---- the two pins ----");
    app_log(1, "    lambda = 1 @ gygi      e_corr = {:.12f}   (must equal the -3.8642e-03 "
               "shift on file)", bl_g1);
    app_log(1, "    lambda = 0 @ gygi      e_corr = {:.12f}", bl_g0);
    app_log(1, "    div_treatment ignore_g0 e_corr = {:.12f}   |difference| = {:.3e}",
            bl_ig, std::abs(bl_g0 - bl_ig));

    // lambda = 0 must reproduce ignore_g0 to MACHINE PRECISION, not merely closely: it takes
    // the same code branch, so any difference at all means some head site was missed and the
    // scan below is measuring a mixture.
    REQUIRE(std::abs(bl_g0 - bl_ig) < 1e-12);
    // ... and lambda = 1 must NOT equal it -- otherwise the knob is inert and every point
    // below is the same run (trap 7).
    REQUIRE(std::abs(bl_g1 - bl_ig) > 1e-6);

    // ---- THE SCAN ------------------------------------------------------------------------
    const std::vector<double> lam = {0.0, 0.25, 0.5, 0.75, 1.0};
    std::vector<double> d_bl(lam.size()), d_bs(lam.size()), coh(lam.size(), -1.0);
    for (size_t i = 0; i < lam.size(); ++i) {
      d_bl[i] = (i == 0 ? bl_g0 : (i + 1 == lam.size() ? bl_g1
                                                       : run("linear", "gygi", lam[i], &coh[i])))
                - ref;
      d_bs[i] = run("static", "gygi", lam[i]) - ref;
    }

    // least squares for d(lambda) = a*lambda + b*lambda^2 (no constant: d(0) is subtracted,
    // so the curve passes through the lambda = 0 point by construction and a, b describe the
    // head's OWN contribution). Two unknowns, five points -> the residual is meaningful.
    auto fit = [&](std::vector<double> const &d) {
      double s11 = 0, s12 = 0, s22 = 0, t1 = 0, t2 = 0;
      for (size_t i = 0; i < lam.size(); ++i) {
        const double x = lam[i], y = d[i] - d[0];
        s11 += x * x;       s12 += x * x * x;   s22 += x * x * x * x;
        t1  += x * y;       t2  += x * x * y;
      }
      const double det = s11 * s22 - s12 * s12;
      const double a = (t1 * s22 - t2 * s12) / det;
      const double b = (t2 * s11 - t1 * s12) / det;
      double res = 0.0, scale = 0.0;
      for (size_t i = 0; i < lam.size(); ++i) {
        const double x = lam[i], y = d[i] - d[0];
        res = std::max(res, std::abs(y - (a * x + b * x * x)));
        scale = std::max(scale, std::abs(y));
      }
      return std::array<double, 3>{a, b, (scale > 0.0 ? res / scale : 0.0)};
    };
    const auto fbl = fit(d_bl);
    const auto fbs = fit(d_bs);

    app_log(1, "  ---- d(e_corr) vs the head strength lambda ----");
    app_log(1, "    lambda      B-L d(e_corr)     B-S d(e_corr)    dW(Gamma) coherence "
               "(-1 = not recorded)");
    for (size_t i = 0; i < lam.size(); ++i)
      app_log(1, "    {:5.2f}     {:+.6e}     {:+.6e}     {:6.3f}", lam[i], d_bl[i], d_bs[i],
              coh[i]);
    app_log(1, "  ---- fit  d(lambda) - d(0) = a*lambda + b*lambda^2 ----");
    app_log(1, "    B-L :  a = {:+.4e}   b = {:+.4e}   |b|/|a| = {:.3f}   max resid "
               "{:.1e} (rel)", fbl[0], fbl[1], (fbl[0] != 0.0 ? std::abs(fbl[1] / fbl[0]) : 0.0),
            fbl[2]);
    app_log(1, "    B-S :  a = {:+.4e}   b = {:+.4e}   |b|/|a| = {:.3f}   max resid "
               "{:.1e} (rel)", fbs[0], fbs[1], (fbs[0] != 0.0 ? std::abs(fbs[1] / fbs[0]) : 0.0),
            fbs[2]);
    app_log(1, "    b is the COINCIDENT-Gamma (both rung transfers at Gamma) coefficient --");
    app_log(1, "    the int d^3q/q^4 cell. a is the ordinary one-rung-at-Gamma term.");
    app_log(1, "==============================================\n");

    for (size_t i = 0; i < lam.size(); ++i) {
      REQUIRE(std::isfinite(d_bl[i]));
      REQUIRE(std::isfinite(d_bs[i]));
    }
    // MONOTONE LIVENESS: the knob must actually move B-L across the scan, or the fit is
    // fitting noise.
    REQUIRE(std::abs(d_bl.back() - d_bl.front()) > 1e-4);
    // the 2-term fit must actually describe the curve -- if the residual is a large fraction
    // of the swing, d(lambda) is not lambda + lambda^2 and the one-rung/two-rung reading
    // does not hold. Measured 1.3 % (B-L) and 0.5 % (B-S).
    REQUIRE(fbl[2] < 0.25);
    REQUIRE(fbs[2] < 0.25);

    // ---- 🚨 THE COINCIDENT-Gamma HYPOTHESIS IS REFUTED ---------------------------------
    // If the int d^3q/q^4 cell (BOTH rung transfers at Gamma) were the mechanism, B-L would
    // be dominated by lambda^2. It is not: |b|/|a| = 0.301, the quadratic term carries the
    // OPPOSITE sign from the linear one (it partially cancels), and B-S -- which has the very
    // same two-rung Gamma cell -- shows a comparable 0.145. Two-rung order counting is not
    // what distinguishes them.
    REQUIRE(std::abs(fbl[1]) < 0.6 * std::abs(fbl[0]));
    REQUIRE(fbl[0] * fbl[1] < 0.0);
    REQUIRE(std::abs(fbl[1] / fbl[0]) < 4.0 * std::abs(fbs[1] / fbs[0]));

    // ---- ⭐ WHAT IS ANOMALOUS IS THE ONE-RUNG (LINEAR) TERM ------------------------------
    // B-L's linear response to the head is 180x B-S's and of OPPOSITE SIGN, on the same mesh,
    // the same Gamma cell and the same q-sum. A single head insertion is a LINEAR effect, so
    // this is the same object P0.1 measured: the head survives W - W0 into the dynamic rung
    // leg with near-full bare weight and leaves it 98 % rank-1.
    REQUIRE(std::abs(fbl[0]) > 50.0 * std::abs(fbs[0]));
    REQUIRE(fbl[0] * fbs[0] < 0.0);
    // ... and B-S stays positive and nearly head-independent across the whole scan, so the
    // sign flip is B-L's alone: +1.1529e-03 -> +1.1986e-03, a 4 % drift over the full range.
    REQUIRE(d_bs.front() > 0.0);
    REQUIRE(d_bs.back() > 0.0);
    REQUIRE(std::abs(d_bs.back() - d_bs.front()) < 0.10 * std::abs(d_bs.front()));
    // ... while B-L crosses zero inside the scan: it AGREES with B-S at lambda = 0 and is
    // sign-flipped by lambda = 1. The head strength alone drives the disagreement.
    REQUIRE(d_bl.front() > 0.0);
    REQUIRE(d_bl.back() < 0.0);

    // ---- the coherence rises with the head, and is already high at QUARTER strength ------
    // 0.881 at lambda = 0.25, 0.960 at 0.50, 0.973 at 0.75 (vs 0.009 with no head at all).
    // The rank-1 alignment of dW(Gamma) is not something that only switches on at full
    // strength -- a quarter of the head already produces most of it.
    REQUIRE(coh[1] > 0.75);
    REQUIRE(coh[3] > coh[1]);
#endif
  }

  // ======================================================================================
  // P4 SCOPING: does the COLD-START divergence reproduce on LiH at gygi -- and does damping
  // fix it without touching a cut?
  //
  // The divergence has only ever been seen on Si (e_corr -2.46 / -2.02 / -5.97 at iterations
  // 4-8 of three cold runs). Every LiH run in this file stops at 2 iterations, which is too
  // early to see it. Now that LiH at gygi reproduces the head pathology, the first question
  // is whether it reproduces the INSTABILITY too -- because if it does, P4 becomes a local
  // problem like P1-P3, and if it does not, a remedy cannot be developed here at all.
  //
  // The head projection is the one knob known to control the basin, and it is Phi-BREAKING
  // (vertex_t.h). P4 asks for a remedy that does NOT modify a cut; damping is the cheapest
  // such option, since it changes only the ITERATION MAP, never Phi or either cut, so a
  // converged fixed point is unchanged by construction.
  //
  // HIDDEN ([.]). Run with:
  //   cd build/tests/bin && OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  //     mpiexec -n 2 --oversubscribe ./test_methods_vertex_static_e2e "[blcold]"
  TEST_CASE("vertex_bl_cold_start_damping", "[.][methods][vertex][static][blcold]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_bl_cold_start_damping skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_blcold";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    const long NIT = 10;   // Si diverged at iterations 4-8, so 10 is enough to see it

    auto run = [&](double mixing, bool projection) -> double {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol{iter_scf::damp_t(mixing)};
      solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(1, 3), mf->nbnd(),
                            "gygi", "global", -1, 1e-8, -1.0, -1.0, "linear");
      vtx.set_bl_head_projection(projection);
      scr_eri.set_vertex(&vtx);
      gw.set_vertex(&vtx);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     NIT, false, 1e-9, true);
      (void) e_hf;
      mpi_context->comm.barrier();
      return e_corr;
    };

    const double m07 = run(0.7, false);   // the production map, projection OFF (the honest one)
    const double m03 = run(0.3, false);   // P4's cheapest legal remedy
    const double proj = run(0.7, true);   // the Phi-breaking knob, for reference only

    // ---- DOES THE SIGN SURVIVE ITERATION COUNT? -----------------------------------------
    // Everything concluded about the B-S/B-L sign was measured at 2 COLD iterations, but the
    // trajectory above moves e_corr by 2.6e-02 between iterations 1 and 10 -- seven times the
    // vertex shift itself. So the 2-iteration d(e_corr) is a snapshot on a moving curve, and
    // the sign has to be re-checked against a MATCHED reference at the same iteration count.
    auto run_plain = [&](std::string const &rung) -> double {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol{iter_scf::damp_t(0.7)};
      const bool with_vertex = not rung.empty();
      solvers::vertex_t vtx(&ft, with_vertex ? "2nd_exchange" : "none",
                            with_vertex ? nda::range(1, 3) : nda::range(0, 0), mf->nbnd(),
                            with_vertex ? "gygi" : "ignore_g0",
                            "global", -1, 1e-8, -1.0, -1.0,
                            with_vertex ? rung : "dynamic");
      if (vtx.enabled()) { scr_eri.set_vertex(&vtx); gw.set_vertex(&vtx); }
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     NIT, false, 1e-9, true);
      (void) e_hf;
      mpi_context->comm.barrier();
      return e_corr;
    };
    const double ref10 = run_plain("");
    const double bs10  = run_plain("static");

    app_log(1, "\n===== P4: COLD B-L ON LiH AT gygi, {} ITERATIONS =====", NIT);
    app_log(1, "  mixing 0.7, projection OFF   e_corr = {:.9f}", m07);
    app_log(1, "  mixing 0.3, projection OFF   e_corr = {:.9f}   <- damping only; touches "
               "the ITERATION MAP, not Phi", m03);
    app_log(1, "  mixing 0.7, projection ON    e_corr = {:.9f}   <- Phi-BREAKING, reference "
               "only", proj);
    app_log(1, "  Si cold reference: e_corr reached -2.46 / -2.02 / -5.97 at iterations 4-8.");
    app_log(1, "  plain-scGW scale here is ~-0.096, so |e_corr| >> 1 means DIVERGED.");
    app_log(1, "  --> LiH cold at mixing 0.7 {} reproduce the Si instability.",
            (std::abs(m07) > 1.0 or not std::isfinite(m07)) ? "DOES" : "does NOT");
    app_log(1, "  NOTE mixing 0.3 lags 0.7 at a FIXED iteration count because it converges "
               "slower -- damping cannot");
    app_log(1, "       move the fixed point, it only changes the path. Neither arm hit the "
               "1e-9 tolerance in {} iters.", NIT);
    app_log(1, "  ---- THE SIGN, RE-CHECKED AT {} ITERATIONS (matched reference) ----", NIT);
    app_log(1, "    plain scGW  e_corr = {:.9f}", ref10);
    app_log(1, "    B-S         e_corr = {:.9f}   d = {:+.4e}", bs10, bs10 - ref10);
    app_log(1, "    B-L         e_corr = {:.9f}   d = {:+.4e}", m07, m07 - ref10);
    app_log(1, "    at 2 iterations these were  B-S +1.1986e-03 / B-L -3.8642e-03 -> DISAGREE");
    app_log(1, "    at {} iterations they {}.", NIT,
            (((bs10 - ref10) * (m07 - ref10)) > 0.0 ? "AGREE -- the 2-iteration reading "
                                                      "does NOT survive"
                                                    : "still DISAGREE -- the sign is robust "
                                                      "to iteration count"));
    app_log(1, "=================================================\n");

    // No pass/fail on the physics -- this case exists to MEASURE whether the instability is
    // reproducible here. The only hard requirement is that the runs completed.
    REQUIRE(not std::isnan(m07));
    REQUIRE(not std::isnan(m03));
    REQUIRE(not std::isnan(proj));
    REQUIRE(std::isfinite(ref10));
    REQUIRE(std::isfinite(bs10));
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

  // ======================================================================================
  // T1 STEP 1 -- THE PER-q_x-RESOLVED S1 METER (the qx_diag analogue on the Sigma side)
  //
  // P0.4's verdict (notes/bl_head_balance_theory_and_plan.md section 5b-5e) localized the
  // gygi damage to the mixed terms' coupling of W0's STATIC Gamma-head (static rung, q_x)
  // to the RETARDED dW-body convolution (dynamic rung, all q_y). The new theory target is
  // the validity of the rank-1 microcell estimate xi_M eps^-1 chi chi^dag when the Gamma
  // cell couples coherently to a retarded second-order kernel -- and its FIRST step is to
  // measure the S1 integrand's q_x profile directly, not infer it from energy ablations.
  //
  // The meter: sigma_C_slot_probe.qx_St -- the kernel's own per-(q_x, external s,k)
  // decomposition of the instantaneous reductions, gated to S1 by only_term = 1, captured
  // from the LAST eval (iteration NIT's kernel, trap 9) together with the exact G it
  // consumed (gA -- NOT the post-update_G G, trap 8). Each q_x slice is contracted
  // against gA in the eval_corr_energy convention (same pipeline, same k weights, same
  // -0.5*spin prefactor, times the kernel's 1/N_k^2), giving T(q_x): the q_x-resolved
  // S1 contribution to Tr[Sigma G] in e_corr-like units.
  //
  // THREE ARMS, one knob (the VERTEX div policy; gw/scr stay ignore_g0 as in every gate):
  //   gygi      -- the analytic rank-1 head augments the Gamma cell of W0_x AND W(Gamma)
  //   ignore_g0 -- no head anywhere: the pure body integrand
  //   v1_skip   -- the STRUCTURAL pin: the combo loop drops q_x = Gamma entirely, so the
  //                meter's Gamma row must be EXACTLY zero (attribution gate: if anything
  //                lands there, the meter's q_x bookkeeping is wrong)
  //
  // WHAT THE PROFILE DECIDES: the head enters the S1 kernel at the q_x = Gamma cell (the
  // static-rung W0_x head) and at q_y = Gamma inside every q_x row (the dW_y head, proven
  // energetically irrelevant by [blh1]). So the gygi-vs-ignore_g0 difference profile
  // should be CONCENTRATED at q_x = Gamma -- the per-q_x replication of the H1
  // falsification by an entirely different meter -- and the ignore_g0 body profile is the
  // first direct look at the integrand's q_x -> 0 behavior under a retarded partner,
  // the quantity the microcell comparison (T1 step 2) needs.
  //
  // HIDDEN ([.]). Run with:
  //   cd build/tests/bin && OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  //     mpiexec -n 2 --oversubscribe ./test_methods_vertex_static_e2e "[blqx]"
  TEST_CASE("vertex_bl_s1_qx_profile", "[.][methods][vertex][static][blqx]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_bl_s1_qx_profile skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_blqx";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    const long NIT = 2;   // the standing gate protocol: 2 COLD iterations

    // one (qx_St, gA) pair per arm; resized by the kernel, all_reduced here
    nda::array<ComplexType, 5> qxS_g, qxS_n, qxS_v, gA_g, gA_n, gA_v;

    auto run = [&](std::string const &policy, nda::array<ComplexType, 5> *qx,
                   nda::array<ComplexType, 5> *g) -> double {
      solvers::vertex_detail::sigma_C_slot_probe.only_term = 1;   // S1 ONLY
      solvers::vertex_detail::sigma_C_slot_probe.qx_St = qx;
      solvers::vertex_detail::sigma_C_slot_probe.gA = g;
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(1, 3), mf->nbnd(), policy,
                            "global", -1, 1e-8, -1.0, -1.0, "linear");
      scr_eri.set_vertex(&vtx);
      gw.set_vertex(&vtx);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     NIT, false, 1e-9, true);
      (void) e_hf;
      mpi_context->comm.barrier();
      solvers::vertex_detail::sigma_C_slot_probe.clear();
      // the kernel leaves rank-local partial sums over its (combo, qy) split
      mpi_context->comm.all_reduce_in_place_n(qx->data(), qx->size(), std::plus<>{});
      return e_corr;
    };

    const double e_g = run("gygi", &qxS_g, &gA_g);
    const double e_n = run("ignore_g0", &qxS_n, &gA_n);
    const double e_v = run("v1_skip", &qxS_v, &gA_v);

    // ---- locate Gamma on the transfer mesh (== the k mesh; kernel asserts nq == nk) ----
    auto kpts = mf->kpts();
    const long nq = qxS_g.shape(0);
    long iq_gamma = 0;
    {
      double best = 1e30;
      for (long iq = 0; iq < nq; ++iq) {
        double n2 = 0.0;
        for (int d = 0; d < 3; ++d) n2 += kpts(iq, d) * kpts(iq, d);
        if (n2 < best) { best = n2; iq_gamma = iq; }
      }
      REQUIRE(best < 1e-12);   // the transfer mesh must contain Gamma exactly
    }

    // ---- T(q_x): contract each q_x slice against the captured G, eval_corr_energy
    //      convention (tau_to_w -> k-weighted band dot -> w_to_tau -> tau_to_beta,
    //      times the kernel's own 1/N_k^2) --------------------------------------------
    auto kw = mf->k_weight();
    auto trace_qx = [&](nda::array<ComplexType, 5> const &qxS,
                        nda::array<ComplexType, 5> const &g) {
      auto shp = qxS.shape();
      const long nqs = shp[0], nsk = shp[1], ntq = shp[2], nb = shp[3];
      const long ns = g.shape(1);
      const long nk_ext = nsk / ns;
      const long nw = ft.nw_f();
      REQUIRE(g.shape(3) == nb);            // both are C-window blocks
      REQUIRE(g.shape(0) == ntq);
      const double norm = 1.0 / double(nqs * nqs);    // the kernel's (1/N_k)^2
      const double spin = (ns == 2) ? 1.0 : 2.0;
      nda::array<ComplexType, 1> T(nqs);
      nda::array<ComplexType, 4> S_tski(ntq, ns, nk_ext, nb), G_tski(ntq, ns, nk_ext, nb);
      nda::array<ComplexType, 4> S_wski(nw, ns, nk_ext, nb), G_wski(nw, ns, nk_ext, nb);
      for (long iq = 0; iq < nqs; ++iq) {
        nda::array<ComplexType, 2> SG_ws(nw, ns);
        SG_ws() = ComplexType(0.0);
        for (long i = 0; i < nb; ++i) {   // band-row i of Sigma against band-col i of G
          for (long it = 0; it < ntq; ++it)
            for (long s = 0; s < ns; ++s)
              for (long k = 0; k < nk_ext; ++k)
                for (long b = 0; b < nb; ++b) {
                  S_tski(it, s, k, b) = qxS(iq, s * nk_ext + k, it, i, b);
                  G_tski(it, s, k, b) = g(it, s, k, b, i);
                }
          ft.tau_to_w(S_tski, S_wski, imag_axes_ft::fermion);
          ft.tau_to_w(G_tski, G_wski, imag_axes_ft::fermion);
          for (long w = 0; w < nw; ++w)
            for (long s = 0; s < ns; ++s)
              for (long k = 0; k < nk_ext; ++k) {
                ComplexType acc(0.0);
                for (long b = 0; b < nb; ++b) acc += S_wski(w, s, k, b) * G_wski(w, s, k, b);
                SG_ws(w, s) += kw(k) * acc;
              }
        }
        nda::array<ComplexType, 2> SG_ts(ntq, ns);
        nda::array<ComplexType, 1> SG_b(ns);
        ft.w_to_tau(SG_ws, SG_ts, imag_axes_ft::fermion);
        ft.tau_to_beta(SG_ts, SG_b);
        ComplexType tot(0.0);
        for (long s = 0; s < ns; ++s) tot += SG_b(s);
        T(iq) = ComplexType(-0.5 * spin * norm) * tot;
      }
      return T;
    };

    auto T_g = trace_qx(qxS_g, gA_g);
    auto T_n = trace_qx(qxS_n, gA_n);
    auto T_v = trace_qx(qxS_v, gA_v);

    // ---- the table -------------------------------------------------------------------
    app_log(1, "\n========== T1 STEP 1: THE PER-q_x S1 PROFILE (iteration-{} kernel) ==========", NIT);
    app_log(1, "  LiH-222, C = [1,3), S1 = W0_x dW_y only (only_term = 1); gw/scr at ignore_g0.");
    app_log(1, "  T(q_x) = the q_x slice of Tr[Sigma^(S1) G] in the eval_corr_energy convention.");
    app_log(1, "  e_corr(S1-only arms): gygi {:.9f}, ignore_g0 {:.9f}, v1_skip {:.9f}", e_g, e_n, e_v);
    app_log(1, "  {:>4} {:>9}   {:>13} {:>13} {:>13}   {:>13}", "qx", "|q|", "T_gygi",
            "T_ig0", "T_v1skip", "dT(g-i)");
    double body_absdiff = 0.0, body_abs_n = 0.0, qmin_body = 1e30;
    long iq_body = -1;
    for (long iq = 0; iq < nq; ++iq) {
      double qn = 0.0;
      for (int d = 0; d < 3; ++d) qn += kpts(iq, d) * kpts(iq, d);
      qn = std::sqrt(qn);
      const double dT = T_g(iq).real() - T_n(iq).real();
      app_log(1, "  {:>4} {:>9.5f}   {:>+13.5e} {:>+13.5e} {:>+13.5e}   {:>+13.5e}{}",
              iq, qn, T_g(iq).real(), T_n(iq).real(), T_v(iq).real(), dT,
              iq == iq_gamma ? "   <-- Gamma" : "");
      if (iq != iq_gamma) {
        body_absdiff += std::abs(dT);
        body_abs_n += std::abs(T_n(iq).real());
        if (qn < qmin_body) { qmin_body = qn; iq_body = iq; }
      }
    }
    const double dT_gamma = T_g(iq_gamma).real() - T_n(iq_gamma).real();
    app_log(1, "  Gamma-concentration of the head: |dT(Gamma)| = {:.4e} vs sum_body |dT| = {:.4e} "
               "(ratio {:.1f})", std::abs(dT_gamma), body_absdiff,
            std::abs(dT_gamma) / std::max(body_absdiff, 1e-300));
    app_log(1, "  integrand q->0 look (ignore_g0 body): T(Gamma) = {:+.4e} vs T(|q|min body, qx={}) "
               "= {:+.4e}", T_n(iq_gamma).real(), iq_body, T_n(iq_body).real());
    app_log(1, "==============================================================================\n");

    // ---- gates -----------------------------------------------------------------------
    // liveness (trap 7): all three arms produced a meter, and the policy knob FIRED
    REQUIRE(nda::max_element(nda::abs(qxS_g)) > 0.0);
    REQUIRE(nda::max_element(nda::abs(qxS_n)) > 0.0);
    REQUIRE(nda::max_element(nda::abs(qxS_v)) > 0.0);
    REQUIRE(std::abs(dT_gamma) > 1e-12);

    // ATTRIBUTION (the structural pin): v1_skip drops q_x = Gamma in the combo loop, so
    // the meter's Gamma row must be EXACTLY zero -- not small, zero.
    REQUIRE(nda::max_element(nda::abs(qxS_v(iq_gamma, nda::ellipsis{}))) == 0.0);

    // the policy difference is where the theory says it is: concentrated at q_x = Gamma
    // (the q_y = Gamma dW-head leakage into the body rows is the [blh1]-irrelevant piece)
    REQUIRE(std::abs(dT_gamma) > body_absdiff);

    // ---- MEASURED 2026-08-01 (first run of this meter) -- pinned as regressions ------
    //   T(Gamma): gygi -1.96123e-03 vs ignore_g0 -7.75672e-06  -> the head multiplies
    //   the Gamma-cell S1 integrand 253x, while the neighbouring BODY cells sit at
    //   1.2e-05/4.5e-05 -- single-rung 1/q^2 microcell counting would put the Gamma
    //   cell at a few x the neighbouring cells for this mesh, not 2.5e+02. That excess
    //   weight is the integrand-level face of the static-head x retarded-partner
    //   mechanism, and it is exactly what T1 step 2's fine-q microcell integration
    //   must compare against the analytic insertion.
    //   Gamma-concentration of the A/B: |dT(Gamma)| / sum_body |dT| = 54.3 (98.2 %) --
    //   the [blh1] falsification (dW-head irrelevant) replicated per-q_x.
    REQUIRE(std::abs(dT_gamma) > 20.0 * body_absdiff);                    // measured 54.3
    REQUIRE(std::abs(T_g(iq_gamma).real() / T_n(iq_gamma).real()) > 100.0);  // measured 253
    REQUIRE(std::abs(T_g(iq_gamma).real() - (-1.96123e-03)) < 2e-4);
    REQUIRE(std::abs(T_n(iq_gamma).real() - (-7.75672e-06)) < 2e-6);
    // e_corr of the S1-only arms, same protocol class as [blmixed] (loose 1e-5 bars:
    // 2 cold iterations, DLR prec "low")
    REQUIRE(std::abs(e_g - (-0.099123246)) < 1e-5);
    REQUIRE(std::abs(e_n - (-0.096997903)) < 1e-5);
    REQUIRE(std::abs(e_v - (-0.096978884)) < 1e-5);

    // finite energies, and the arms genuinely differ (no config null)
    REQUIRE(std::isfinite(e_g));
    REQUIRE(std::isfinite(e_n));
    REQUIRE(std::isfinite(e_v));
    REQUIRE(std::abs(e_g - e_n) > 1e-9);
#endif
  }

} // bdft_tests
