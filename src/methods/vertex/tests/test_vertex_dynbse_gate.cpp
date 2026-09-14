/**
 * scGW-tilde Tier 2 full frequency, increment D2 (notes/dynbse_plan.md section 9): the production
 * driver's gates on the qe_lih222 fixture with the production pol-only attachment (vertex_type =
 * "none", pol_vertex = "ladder"). After two damped iterations the W-bar cache is filled by hand
 * (update_w publishes dW; cache_w folds it -- the static rung mode never fills the cache), then
 * vertex_t::dynbse_gate runs:
 *   (A0) the THC rung operator vs the explicit Kbig                      -- machine class
 *   (A)  the static limit vs the sign-corrected L2 resolvent -ladder(-W0) -- 1e-12 class
 *   (B)  the one bare dynamic rung vs pi_c_accumulate_w(Z = 0, W_dyn - W_dyn(0)) -- fit class
 *   (C)  GMRES vs Neumann on the resummed vertex; Hermiticity; the watchdog and meters.
 */

#undef NDEBUG

#include <cstdlib>
#include <filesystem>
#include <cmath>
#include <complex>
#include <string>

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

  TEST_CASE("dynbse_gate", "[methods][vertex][scgwt][dynbse]") {
#ifndef ENABLE_DLR
    SUCCEED("dynbse_gate skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis,
                          std::getenv("COQUI_DYNBSE_TEST_PREC") ? std::string(std::getenv("COQUI_DYNBSE_TEST_PREC")) : std::string("low"));
    std::string output = "coqui_d2_gates";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    auto gate_at = [&](nda::range window, int niter, bool quick) {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd());
      vtx.set_pol_vertex("ladder", "w0_prev", window, -1, 1e-8, -1.0, -1.0, -1.0);
      scr_eri.set_vertex(&vtx);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     niter, false, 1e-9, true);
      (void)e_hf; (void)e_corr;
      auto *pv = scr_eri.pol_vertex_instance();
      REQUIRE(pv != nullptr);
      // the W-bar cache: the static-rung mode never fills it; publish dW once more and fold it
      scr_eri.update_w(mb_state, thc, -1);
      REQUIRE(mb_state.dW_qtPQ.has_value());
      pv->cache_w(mb_state, thc);
      REQUIRE(pv->has_cached_w());
      auto g = pv->dynbse_gate(mb_state, thc, quick);
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return g;
    };

    if (std::getenv("COQUI_DYNBSE_QUICK") != nullptr) {
      // the quick leg (A0 / A0-FT / A0' only): the mesh-Fourier k-sum IS the direct rung operator
      auto gq = gate_at(nda::range(0, 4), 1, true);
      app_log(1, "dynbse_gate quick: A0 {:.3e}; A0-FT {:.3e} (mesh check {:.2e})", gq.a0_resid, gq.a0_ft_resid, gq.a0_ft_check);
      REQUIRE(gq.a0_resid < 1e-12);
      REQUIRE(gq.a0_ft_check >= 0.0);
      REQUIRE(gq.a0_ft_check < 1e-10);
      REQUIRE(gq.a0_ft_resid >= 0.0);
      REQUIRE(gq.a0_ft_resid < 1e-11);
      return;
    }
    auto g = gate_at(nda::range(0, 4), 2, false);
    app_log(1, "dynbse_gate [C = [0,4) of {}]: A0 {:.3e}; A {:.3e} (vs as-implemented L2 {:.3e}); B {:.3e} "
               "(|anchor| {:.3e}, |static| {:.3e}); GMRES vs Neumann {:.3e}, Gamma1 {:.3e}; Ritz {:.3f} / "
               "contraction {:.3f}; applications {} / {}; converged {}; refit {:.3e}; herm {:.3e}; dyn vs static "
               "{:.3e}, Gamma1 vs static {:.3e}; G fit {:.3e} rr {:.3g}; Dsq {:.3e}; W(s) sym {:.3e}",
            mf->nbnd(), g.a0_resid, g.a_resid, g.a_l2_diff, g.b_resid, g.onerung_max, g.static_max,
            g.gmres_vs_neumann, g.gam1_consistency, g.ritz_max, g.contraction_max, g.it_max, g.it_max_neumann,
            g.all_converged, g.refit_err, g.herm, g.dyn_vs_static, g.gam1_vs_static, g.fit_err_G, g.rr_G,
            g.dsq_err, g.wtau_sym);
    REQUIRE(g.a0_resid >= 0.0);
    REQUIRE(g.a0_resid < 1e-12);                  // the THC rung operator IS Kbig
    REQUIRE(g.a0_ft_resid >= 0.0);
    REQUIRE(g.a0_ft_resid < 1e-11);               // (A0-FT) the mesh-Fourier k-sum IS the direct rung operator
    REQUIRE(g.a_resid < 1e-10);                   // the static limit IS the (sign-corrected) L2 resolvent
    REQUIRE(g.b_resid < 1e-3);                    // one dynamic rung = the anchor at inu = 0 (fit class)
    REQUIRE(g.b_continuity < 1e-2);               // and continuous into the first positive node
    // the resummed solve at inu = 0: both solvers agree, converged, contractive
    REQUIRE(g.gmres_vs_neumann < 1e-6);
    REQUIRE(g.block_resid >= 0.0);
    REQUIRE(g.block_resid < 1e-6);                // (C'') RHS column blocking = the unblocked solve (tol class)
    REQUIRE(g.nu1_gfit >= 0.0);
    REQUIRE(g.nu1_gfit < 1e-4);                   // (C') the union grid without a mask represents G
    REQUIRE(g.nu1_done);                          // and the first positive node converges (GMRES(12), readout stop)
    REQUIRE(std::abs(g.nu1_resid - g.dyn_vs_static) < 0.05 * g.dyn_vs_static);   // nu-continuity of the resummed correction
    REQUIRE(g.gam1_consistency < 1e-12);
    REQUIRE(g.all_converged);
    REQUIRE(g.ritz_max < 1.0);
    REQUIRE(g.dyn_vs_static > 0.0);
    REQUIRE(std::isfinite(g.dyn_max));
#endif
  }

  TEST_CASE("dynbse_readout", "[methods][vertex][scgwt][dynbse]") {
#ifndef ENABLE_DLR
    SUCCEED("dynbse_readout skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis,
                          std::getenv("COQUI_DYNBSE_TEST_PREC") ? std::string(std::getenv("COQUI_DYNBSE_TEST_PREC")) : std::string("low"));
    std::string output = "coqui_d3_readout";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    auto run = [&](std::string const &rung, int niter, bool dump = false, bool dense = true, long ustride = 1) {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd());
      vtx.set_pol_vertex("ladder", "w0_prev", nda::range(0, 4), -1, 1e-8, -1.0, -1.0, -1.0);
      vtx.set_ladder_rung(rung, 1e-8, 30, 12, -1.0);
      vtx.set_eps_cut(1, 3);                    // the eps(q_i, i nu) cut report (report-only; q_min and the 3 lowest nodes
                                                //  for the dynamic columns keep the test short)
      vtx.set_ladder_dyn_dump(dump);            // per-unit dump / restart files of the dynamic solves
      vtx.set_ladder_dyn_dense(dense);          // the dense per-tau rung (default) vs the THC streaming route
      vtx.set_ladder_dyn_union_stride(ustride); // the inu != 0 union grid's G-node stride
      if (char const *vp = std::getenv("COQUI_DYNBSE_TEST_VPREC")) vtx.set_ladder_dyn_iaft_prec(vp);   // vertex-local DLR precision
      if (char const *tf = std::getenv("COQUI_DYNBSE_TEST_TFOLD")) vtx.set_ladder_dyn_tfold(std::atof(tf));   // the small-nu fold ratio
      if (std::getenv("COQUI_DYNBSE_TEST_G1")) vtx.set_ladder_dyn_gamma1_only(true);   // Gamma_1 only (skip the resummation GMRES)
      scr_eri.set_vertex(&vtx);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     niter, false, 1e-9, true);
      auto [er, el] = scr_eri.pol_eps_readout();
      auto ed = scr_eri.pol_eps_dyn();
      const double ritz = scr_eri.pol_dyn_ritz();
      nda::array<double, 2> cut(scr_eri.pol_eps_cut_qmin());
      const double eloop = scr_eri.pol_eps_loop();
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return std::make_tuple(e_hf, e_corr, er, el, ed, ritz, cut, eloop);
    };
    auto [h0, c0, r0, l0, d0, z0, cut0, eloop0] = run("static", 2);
    auto [h1, c1, r1, l1, d1, z1, cut1, eloop1] = run("dynamic", 2, true);     // writes the unit dump files
    auto [h2, c2, r2, l2, d2, z2, cut2, eloop2] = run("dynamic", 2, true);     // loads them: no unit is re-solved
    auto [h3, c3, r3, l3, d3, z3, cut3, eloop3] = run("dynamic", 2, false, false);   // the THC streaming rung
    auto [h4, c4, r4, l4, d4, z4, cut4, eloop4] = run("dynamic", 2, false, true, 2);  // union stride 2 at inu != 0
    if (mpi_context->comm.root()) {
      app_log(1, "dynbse_readout: union stride 2 vs 1 at nodes 1, 2: resummed {} vs {}, {} vs {}", cut4(1, 7), cut1(1, 7),
              cut4(2, 7), cut1(2, 7));
      for (long j = 1; j < 3; ++j)
        for (int c = 4; c < 8; ++c) REQUIRE(std::abs(cut4(j, c) - cut1(j, c)) < 2e-3 * std::abs(cut1(j, c) - 1.0));
    }
    {
      // the dense per-tau rung and the THC streaming route are the same operator (solve-tolerance class)
      app_log(1, "dynbse_readout: dense vs THC rung: resummed {} vs {}, Gamma1 {} vs {}", d1[3], d3[3], d1[2], d3[2]);
      for (int c = 0; c < 4; ++c) REQUIRE(std::abs(d3[c] - d1[c]) < 1e-7 * std::max(1.0, std::abs(d1[c] - 1.0)));
      if (mpi_context->comm.root())
        for (long j = 0; j < 3; ++j)
          for (int c = 4; c < 8; ++c) REQUIRE(std::abs(cut3(j, c) - cut1(j, c)) < 1e-6 * std::max(1.0, std::abs(cut1(j, c) - 1.0)));
    }
    {
      // the restart path: every dynamic column is bitwise what the solving run produced
      for (int c = 0; c < 4; ++c) REQUIRE(d2[c] == d1[c]);
      if (mpi_context->comm.root())
        for (long j = 0; j < cut1.shape(0); ++j)
          for (int c = 4; c < 8; ++c) REQUIRE(cut2(j, c) == cut1(j, c));
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) {
        namespace fs = std::filesystem;
        for (auto const &e : fs::directory_iterator(fs::current_path())) {
          const std::string fn = e.path().filename().string();
          if (fn.rfind(output + ".dynunits.", 0) == 0) fs::remove(e.path());
        }
      }
      mpi_context->comm.barrier();
    }
    // the eps(q_i, i nu) cut (rank 0): node 0 of the q_min cut IS the readout (same rows, same
    // Dyson; the ladder node 0 of the whalf pass vs eval_pol_ladder_nu0 = the node-map class),
    // the loop-side column at node 0 IS the Q3 loop-side value, every node is finite and the
    // static screening decays along i nu (eps_M(i nu_max) < eps_M(0)).
    if (mpi_context->comm.root()) {
      REQUIRE(cut0.shape(0) > 1);
      REQUIRE(cut0.shape(1) == 8);
      app_log(1, "dynbse_readout: eps-cut q_min node 0: RPA {} (readout {}), +ladder {} (readout {}), loop {} (Q3 {}); "
                 "last node: RPA {} +ladder {} loop {}", cut0(0, 0), r0, cut0(0, 1), l0, cut0(0, 3), eloop0,
              cut0(cut0.shape(0) - 1, 0), cut0(cut0.shape(0) - 1, 1), cut0(cut0.shape(0) - 1, 3));
      REQUIRE(std::abs(cut0(0, 0) - r0) < 1e-10);
      REQUIRE(std::abs(cut0(0, 1) - l0) < 1e-8);
      REQUIRE(std::abs(cut0(0, 3) - eloop0) < 1e-10);
      for (long j = 0; j < cut0.shape(0); ++j) {
        REQUIRE(std::isfinite(cut0(j, 0))); REQUIRE(cut0(j, 0) > 0.0);
        REQUIRE(std::isfinite(cut0(j, 1))); REQUIRE(cut0(j, 1) > 0.0);
        REQUIRE(std::isfinite(cut0(j, 3))); REQUIRE(cut0(j, 3) > 0.0);
        REQUIRE(cut0(j, 2) == -1.0);            // legs = bare: no DeltaLambda column
      }
      REQUIRE(cut0(cut0.shape(0) - 1, 0) < cut0(0, 0));
      REQUIRE(cut0(cut0.shape(0) - 1, 1) < cut0(0, 1));
      // the cut is report-only: bitwise across the rung modes
      REQUIRE(cut1.shape(0) == cut0.shape(0));
      for (long j = 0; j < cut0.shape(0); ++j)
        for (long c = 0; c < 4; ++c) REQUIRE(cut1(j, c) == cut0(j, c));
      // the dynamic-rung cut (union basis, GMRES(12), all half nodes): node 0 agrees with the
      // inu = 0 readout columns (shared grid) to the refit class, every node is finite, and the
      // resummed column is nu-continuous at the first positive node
      app_log(1, "dynbse_readout: dynamic cut q_min node 0: static {} (nu0 {}), +dyn1 {} (nu0 {}), Gamma1 {} (nu0 {}), "
                 "resummed {} (nu0 {}); node 1 resummed {}; last node resummed {}",
              cut1(0, 4), d1[0], cut1(0, 5), d1[1], cut1(0, 6), d1[2], cut1(0, 7), d1[3], cut1(1, 7),
              cut1(cut1.shape(0) - 1, 7));
      for (long j = 0; j < 3; ++j)
        for (int c = 4; c < 8; ++c) { REQUIRE(std::isfinite(cut1(j, c))); REQUIRE(cut1(j, c) > 0.0); }
      for (long j = 3; j < cut1.shape(0); ++j)
        for (int c = 4; c < 8; ++c) REQUIRE(cut1(j, c) == -1.0);        // beyond the 3 requested nodes
      for (int c = 0; c < 4; ++c) REQUIRE(std::abs(cut1(0, 4 + c) - d1[c]) < 1e-3 * std::abs(d1[c] - 1.0));
      REQUIRE(std::abs(cut1(1, 7) - cut1(0, 7)) < 0.05 * std::abs(cut1(0, 7) - 1.0));
      REQUIRE(cut1(2, 7) < cut1(0, 7));
      for (long j = 0; j < cut0.shape(0); ++j)
        for (int c = 4; c < 8; ++c) REQUIRE(cut0(j, c) == -1.0);
    }
    app_log(1, "dynbse_readout: static rung: e_corr {} eps RPA {} +ladder {} ; dynamic rung: e_corr {} eps RPA {} "
               "+ladder(L2) {} ; +static(sign-corr.) {} +static+Pi^C_dyn {} +Gamma1 {} +resummed {} ; Ritz {}",
            c0, r0, l0, c1, r1, l1, d1[0], d1[1], d1[2], d1[3], z1);
    // the loop and the historic columns are bitwise (the dynamic rung is a readout-only column)
    REQUIRE(h1 == h0);
    REQUIRE(c1 == c0);
    REQUIRE(r1 == r0);
    REQUIRE(l1 == l0);
    REQUIRE(d0[3] == -1.0);
    for (double v : d1) { REQUIRE(std::isfinite(v)); REQUIRE(v > 0.0); }
    REQUIRE(z1 >= 0.0);
    REQUIRE(z1 < 1.0);
    // 2026-09-11: pair_space_ladder resums the derived resolvent (1 + Xh Kt)^-1, so the dynbse
    // driver's static column IS the L2 readout (gate A class, through the same upfold + Dyson)
    REQUIRE(std::abs(d1[0] - l1) < 1e-8);
    // the dynamic rungs move eps_M away from the static ladder
    REQUIRE(d1[3] != d1[0]);
#endif
  }

} // namespace bdft_tests
