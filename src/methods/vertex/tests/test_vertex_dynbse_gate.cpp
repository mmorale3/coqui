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
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_d2_gates";

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
      auto g = pv->dynbse_gate(mb_state, thc);
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return g;
    };

    auto g = gate_at(nda::range(0, 4), 2);
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
    REQUIRE(g.a_resid < 1e-10);                   // the static limit IS the (sign-corrected) L2 resolvent
    REQUIRE(g.b_resid < 1e-3);                    // one dynamic rung = the anchor at inu = 0 (fit class)
    REQUIRE(g.b_continuity < 1e-2);               // and continuous into the first positive node
    // the resummed solve at inu = 0: both solvers agree, converged, contractive
    REQUIRE(g.gmres_vs_neumann < 1e-6);
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
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_d3_readout";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    auto run = [&](std::string const &rung, int niter) {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd());
      vtx.set_pol_vertex("ladder", "w0_prev", nda::range(0, 4), -1, 1e-8, -1.0, -1.0, -1.0);
      vtx.set_ladder_rung(rung, 1e-8, 30, 4, -1.0);
      scr_eri.set_vertex(&vtx);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     niter, false, 1e-9, true);
      auto [er, el] = scr_eri.pol_eps_readout();
      auto ed = scr_eri.pol_eps_dyn();
      const double ritz = scr_eri.pol_dyn_ritz();
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return std::make_tuple(e_hf, e_corr, er, el, ed, ritz);
    };
    auto [h0, c0, r0, l0, d0, z0] = run("static", 2);
    auto [h1, c1, r1, l1, d1, z1] = run("dynamic", 2);
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
    // the sign-corrected static column differs from the as-implemented L2 (the even-order sign)
    REQUIRE(d1[0] != l1);
    // the dynamic rungs move eps_M away from the static ladder
    REQUIRE(d1[3] != d1[0]);
#endif
  }

} // namespace bdft_tests
