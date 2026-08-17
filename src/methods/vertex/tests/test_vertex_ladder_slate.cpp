/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"
#include "mean_field/default_MF.hpp"

#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/scf_common.hpp"
#include "methods/SCF/scf_driver.hpp"
#include "methods/scr_coulomb/scr_coulomb_t.h"
#include "methods/vertex/vertex_t.h"
#include "methods/mb_state/mb_state.hpp"

/**
 * INCREMENT B (notes/ladder_b_integration_design.md; spec notes/ladder_opt_spec.md
 * "Increment B", Gate B): the DISTRIBUTED (SLATE) solve path of the pair-space ladder.
 *
 * The knob is ladder_solve_grid = g, the number of ranks that cooperate on ONE (s,q,nu)
 * resolvent. g = 1 is the historic per-rank LAPACK path; g > 1 tiles the (D,D) operator
 * over a g-rank process grid, builds the owned tiles locally (each owner RECOMPUTING its
 * (N_m x nc2) legs rather than communicating them), runs ONE slate_ops::lu_solve per
 * (q,nu) with both RHS families as a single multi-RHS block, and estimates lambda_max by
 * the same 20-step power iteration made distributed.
 *
 * Gates here (measure first, then gate):
 *   B-1  eval_pol_ladder_whalf at g = 2 vs g = 1: max|dP| / max|P| <= 1e-12. This is
 *        Gate B's "fixture ladder observables" comparison; g = 1 vs the pre-B tree is the
 *        bitwise-class statement pinned by the commit-point suites (nothing on the g = 1
 *        path is touched -- the dispatch resolves g and falls through).
 *   B-2  the WHALF GATE cloned at g = 2: ladder_whalf_gate's node_map_resid compares the
 *        half-grid evaluator (SLATE at g = 2) against a FULL-mesh pair_space_ladder
 *        reference (always the per-rank kernel), so it is an independent A/B of the two
 *        solve paths through the production entry point. Bitwise 0 at g = 1.
 *   B-3  lambda_max: the SAME 20-step protocol, distributed. Reported, and gated only at
 *        the protocol's own resolution -- the iteration breaks on a 1e-3 RELATIVE change,
 *        so the estimate is a 1e-3-converged number by construction, not a 1e-12 one.
 *   B-4  the Q4-C3b E-family (the second RHS family) at g = 2 vs g = 1, on synthetic MLWF
 *        legs: the multi-RHS block carries [D | E] through ONE factorization.
 *
 * ==========================================================================================
 * HOW TO RUN (Catch2 v2 traps) -- MEASURED, do not "improve" the command
 * ==========================================================================================
 *
 *     KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
 *       mpirun -np 2 <build>/tests/bin/test_methods_vertex_ladder_slate
 *
 * i.e. THE BARE BINARY under mpirun, no filter (a "~[tag]" spec does NOT exclude hidden
 * cases, it RUNS them). At np = 1 there is no g = 2 to test and the case SKIPS with a
 * message -- ctest's default CTEST_NPROC is 1, so the distributed legs are a MANUAL /
 * multi-rank-CI run, exactly like the rest of the multi-rank vertex coverage.
 */

namespace bdft_tests {

  using namespace methods;

  TEST_CASE("ladder_slate_whalf_lih222", "[methods][vertex][ladder][ladder_slate]") {
#ifndef ENABLE_DLR
    SUCCEED("ladder_slate_whalf_lih222 skipped: build has ENABLE_DLR=OFF.");
#else
    auto& mpi_context = utils::make_unit_test_mpi_context();
    if (mpi_context->comm.size() < 2) {
      SUCCEED("ladder_slate_whalf_lih222 skipped: needs >= 2 MPI ranks for g = 2 "
              "(run under mpirun -np 2).");
      return;
    }
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    const std::string output = "vertex_ladder_slate";
    const std::string div = "ignore_g0";
    const nda::range window(1, 3);

    // ---- one short qpGW run with the ladder READOUT on (the qpgw_bse recipe, trimmed to
    // 3 iterations: every gate below is an A/B of two solve paths on ONE state, so a
    // converged loop buys nothing) --------------------------------------------------
    solvers::hf_t hf;
    solvers::gw_t gw(&ft, div, output);
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", div);
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*12, "", "incore", "",
                                               output, 1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);
    qp_params_t qp_params("sc", "pade", 18, 0.0001, 1e-8, "qpscf");
    qp_params.qp_map = "ac_pade";
    solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd(), div);
    vtx.set_pol_vertex("ladder", "w0_prev", window, -1, 1e-8, -1.0, -1.0, -1.0, "none");
    scr_eri.set_vertex(&vtx);
    iter_scf::iter_scf_t iter_sol(iter_scf::damp_t(0.7));
    MBState mb_state(mpi_context, ft, output);
    qp_scf_loop(mb_state, eri, ft, qp_params,
                solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 3, false, 1e-6);
    mpi_context->comm.barrier();

    // the extra screening step: puts G and W0bar on the SAME state, which is what the
    // ladder entry points read (the qpgw_bse/c3b recipe)
    {
      using math::shm::make_shared_array;
      double mu = update_mu(0.0, *mf, mb_state.sE_ska.value(), ft.beta());
      mb_state.sG_tskij.emplace(make_shared_array<Array_view_5D_t>(
          *mpi_context, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
      update_G(mb_state.sG_tskij.value(), mb_state.sMO_skia.value(),
               mb_state.sE_ska.value(), mu, ft);
      scr_eri.update_w(mb_state, thc, -1);
    }
    auto *pv = scr_eri.pol_vertex_instance();
    REQUIRE(pv != nullptr);

    auto relmax = [](auto const &A, auto const &B) {
      double d = 0.0, s = 0.0;
      auto ia = A.begin();
      auto ib = B.begin();
      for (; ia != A.end(); ++ia, ++ib) {
        d = std::max(d, std::abs(*ia - *ib));
        s = std::max(s, std::abs(*ia));
      }
      return std::make_pair(d / std::max(s, 1e-300), s);
    };

    // ====================================================================================
    // B-1: the production evaluator, g = 1 vs g = 2
    // ====================================================================================
    nda::array<double, 1> lam1;
    pv->set_ladder_solve(1, 8.0);
    REQUIRE(pv->ladder_solve_grid() == 1);
    auto P1 = pv->eval_pol_ladder_whalf(mb_state, thc, &lam1);

    // every g that DIVIDES the communicator: g = 2 is a 2 x 1 (row-split) grid, g = 4 a
    // genuine 2 x 2 one (both row and column splits, so the tile-owner arithmetic and the
    // multi-column RHS distribution are both exercised).
    std::vector<long> gs;
    for (long g = 2; g <= mpi_context->comm.size(); ++g)
      if (mpi_context->comm.size() % g == 0) gs.push_back(g);
    for (long g : gs) {
      nda::array<double, 1> lam2;
      pv->set_ladder_solve(g, 8.0);
      REQUIRE(pv->ladder_solve_grid() == g);
      auto P2 = pv->eval_pol_ladder_whalf(mb_state, thc, &lam2);
      REQUIRE(P2.shape(0) == P1.shape(0));
      REQUIRE(P2.shape(1) == P1.shape(1));
      REQUIRE(P2.shape(2) == P1.shape(2));
      auto [rel_P, scale_P] = relmax(P1, P2);
      app_log(1, "@@LADSLATE B-1: eval_pol_ladder_whalf g = {} vs g = 1 -- max|dP|/max|P| "
                 "= {:.3e} (max|P| = {:.6e}; {} nu nodes x {} q x {} x {})",
              g, rel_P, scale_P, P1.shape(0), P1.shape(1), P1.shape(2), P1.shape(3));
      REQUIRE(scale_P > 0.0);         // non-vacuous: the ladder is not identically zero
      REQUIRE(rel_P <= 1e-12);

      // B-3: the watchdog. The 20-step power iteration breaks on a 1e-3 relative change,
      // so the estimate is only defined to that resolution -- gate at 1e-2 relative and
      // REPORT the measured deviation.
      double dlam = 0.0, lscale = 0.0;
      for (long j = 0; j < lam1.shape(0); ++j) {
        dlam = std::max(dlam, std::abs(lam1(j) - lam2(j)));
        lscale = std::max(lscale, std::abs(lam1(j)));
      }
      app_log(1, "@@LADSLATE B-3: lambda_max g = {} vs g = 1 -- max|dlambda| = {:.3e} at "
                 "max lambda = {:.6f} (relative {:.3e}); the 20-step protocol's own break "
                 "test is 1e-3 relative", g, dlam, lscale,
              dlam / std::max(lscale, 1e-300));
      REQUIRE(lscale > 0.0);
      REQUIRE(dlam <= 1e-2 * std::max(lscale, 1e-300));
    }

    // B-1b: the AUTO fit test (ladder_solve_grid = 0). A budget far above the per-rank
    // footprint must resolve to g = 1 (and reproduce P1 BITWISE, since it IS the g = 1
    // path); a budget far below it must resolve to some g > 1 and still agree to 1e-12.
    {
      pv->set_ladder_solve(0, 1e6);
      auto Pa = pv->eval_pol_ladder_whalf(mb_state, thc, nullptr);
      auto [rel_a, scale_a] = relmax(P1, Pa);
      pv->set_ladder_solve(0, 1e-9);
      auto Pb = pv->eval_pol_ladder_whalf(mb_state, thc, nullptr);
      auto [rel_b, scale_b] = relmax(P1, Pb);
      app_log(1, "@@LADSLATE B-1b: AUTO grid -- huge budget (=> g = 1) rel = {:.3e}, tiny "
                 "budget (=> widest g that fits) rel = {:.3e} (scale {:.6e})",
              rel_a, rel_b, scale_a);
      REQUIRE(rel_a == 0.0);
      REQUIRE(rel_b <= 1e-12);
      REQUIRE(scale_b > 0.0);
    }

    // ====================================================================================
    // B-2: the WHALF GATE cloned at g = 2 (node_map_resid is the SLATE-vs-per-rank A/B)
    // ====================================================================================
    pv->set_ladder_solve(1, 8.0);
    auto d1 = pv->ladder_whalf_gate(mb_state, thc, 2.0);
    REQUIRE(d1.node_map_resid == 0.0);      // g = 1 IS the reference kernel: bitwise
    for (long g : gs) {
      pv->set_ladder_solve(g, 8.0);
      auto d2 = pv->ladder_whalf_gate(mb_state, thc, 2.0);
      app_log(1, "@@LADSLATE B-2: whalf gate node_map_resid -- g = 1: {:.3e} (bitwise), "
                 "g = {}: {:.3e} ; ph_sym_resid {:.3e} vs {:.3e} ; ladder_max {:.6e} vs "
                 "{:.6e}", d1.node_map_resid, g, d2.node_map_resid, d1.ph_sym_resid,
              d2.ph_sym_resid, d1.ladder_max, d2.ladder_max);
      REQUIRE(d2.node_map_resid <= 1e-12);
      REQUIRE(d2.ladder_max == d1.ladder_max);  // the full-mesh reference leg is untouched
    }

    // ====================================================================================
    // B-4: the second RHS family (Q4-C3b E legs) through the SAME factorization
    // ====================================================================================
    {
      const long ns = mf->nspin(), nk = mf->nkpts(), nc = window.size();
      const long norb = 2;
      nda::array<ComplexType, 4> U_syn(ns, nk, norb, nc);
      unsigned long s = 20260816ul;
      auto rnd = [&s]() {
        s = s * 6364136223846793005ul + 1442695040888963407ul;
        return double((s >> 33) % 100000ul) / 100000.0 - 0.5;
      };
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nk; ++ik)
          for (long m = 0; m < norb; ++m)
            for (long a = 0; a < nc; ++a) U_syn(is, ik, m, a) = ComplexType(rnd(), rnd());
      pv->set_ladder_solve(1, 8.0);
      auto L1 = pv->eval_pol_ladder_loc_whalf(mb_state, thc, U_syn);
      for (long g : gs) {
        pv->set_ladder_solve(g, 8.0);
        auto L2 = pv->eval_pol_ladder_loc_whalf(mb_state, thc, U_syn);
        auto [rel_L, scale_L] = relmax(L1, L2);
        app_log(1, "@@LADSLATE B-4: eval_pol_ladder_loc_whalf (E legs, {} orbitals) g = {} "
                   "vs g = 1 -- max|dP|/max|P| = {:.3e} (max|P| = {:.6e})", norb, g, rel_L,
                scale_L);
        REQUIRE(scale_L > 0.0);
        REQUIRE(rel_L <= 1e-12);
      }
    }

    pv->set_ladder_solve(1, 8.0);
    mpi_context->comm.barrier();
    if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
    mpi_context->comm.barrier();
#endif
  }

} // bdft_tests
