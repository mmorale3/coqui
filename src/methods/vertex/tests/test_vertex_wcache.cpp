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

/**
 * ISDF-Vertex Refinement 2: W-bar iteration cache (notes/wbar_cache.md).
 *
 * In the secondary path the vertex caches the DOWNFOLDED dynamic rung
 * Wbar(q, nu) = t(q) [dW folded to the bosonic mesh] t(q)^dag at update_w time
 * (head-augmented under gygi with the SAME-iteration eps_inv_head), so the scf
 * driver can free mb_state.dW_qtPQ unconditionally (plain-GW memory profile);
 * the cache is consumed by the NEXT iteration's eval_Pi_C -- the identical
 * one-iteration lag as the retained-dW behavior it replaces.
 *
 *  - vertex_wcache_e2e: 2-iteration LiH-222 scGW (both cuts), secondary path with
 *    the cache vs the legacy retained-dW semantics (set_w_cache_enabled(false) --
 *    the verbatim pre-cache code path): e_hf/e_corr identical to <= 1e-14
 *    (bitwise expected: same arithmetic, different scheduling). Memory behavior:
 *    after the loop mb_state.dW_qtPQ is FREED with the cache, RETAINED in legacy
 *    mode and on the global path (unchanged).
 *  - vertex_wcache_identity: eval_Pi_C on one physical (G, W) LiH state, cached
 *    vs legacy consumption, bitwise; under ignore_g0 AND gygi (the cached
 *    head-augmented Wbar(Gamma) vs the legacy at-consumption augmentation).
 *    First-iteration semantics: empty cache + no dW -> bare-Zbar rung, finite.
 */

#undef NDEBUG

#include <cmath>
#include <complex>
#include <tuple>

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
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include "methods/vertex/vertex_t.h"

namespace bdft_tests {

  using namespace methods;
  using cplx = ComplexType;

  // ====================================================================================
  // End-to-end: 2-iteration scGW with BOTH cuts (the second iteration runs the DYNAMIC
  // rung in Pi^C: cached Wbar vs the legacy retained-dW fold-at-consumption).
  TEST_CASE("vertex_wcache_e2e", "[methods][vertex][wcache][smoke]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_wcache_e2e skipped: build has ENABLE_DLR=OFF.");
#else
    auto& mpi_context = utils::make_unit_test_mpi_context();
    // wmax = 6.0: the vertex [A-comp] headroom requirement (pi design section 4b)
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_wcache_e2e";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    // returns (e_hf, e_corr, dW retained after the loop, actual N_m)
    auto run = [&](std::string const& isdf_mode, bool wcache_on) {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(1, 3), mf->nbnd(),
                            "ignore_g0", isdf_mode, 32, 1e-8);
      REQUIRE(vtx.active());
#ifdef VERTEX_WCACHE_API
      if (not wcache_on) vtx.set_w_cache_enabled(false);
#else
      (void)wcache_on;
#endif
      scr_eri.set_vertex(&vtx);
      gw.set_vertex(&vtx);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     2, false, 1e-12, true);
      mpi_context->comm.barrier();
      bool dw_retained = mb_state.dW_qtPQ.has_value();
      long nm = vtx.secondary_rank();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return std::make_tuple(e_hf, e_corr, dw_retained, nm);
    };

    auto [e_hf_leg, e_corr_leg, dw_leg, nm_leg] = run("secondary", false);
    app_log(1, "wcache e2e: secondary LEGACY (retained dW): e_hf = {:.17g}, "
               "e_corr = {:.17g}, dW retained = {}, N_m = {}",
            e_hf_leg, e_corr_leg, dw_leg, nm_leg);

    auto [e_hf_g, e_corr_g, dw_g, nm_g] = run("global", true);
    (void)nm_g;
    app_log(1, "wcache e2e: global (reference path):        e_hf = {:.17g}, "
               "e_corr = {:.17g}, dW retained = {}", e_hf_g, e_corr_g, dw_g);

    for (double e : {e_hf_leg, e_corr_leg, e_hf_g, e_corr_g}) REQUIRE(std::isfinite(e));

#ifdef VERTEX_WCACHE_API
    auto [e_hf_c, e_corr_c, dw_c, nm_c] = run("secondary", true);
    app_log(1, "wcache e2e: secondary CACHED Wbar:          e_hf = {:.17g}, "
               "e_corr = {:.17g}, dW retained = {}, N_m = {}",
            e_hf_c, e_corr_c, dw_c, nm_c);
    app_log(1, "wcache e2e: |D e_hf| = {}, |D e_corr| = {} (cache vs legacy; bitwise "
               "expected -- notes/wbar_cache.md section 2)",
            std::abs(e_hf_c - e_hf_leg), std::abs(e_corr_c - e_corr_leg));
    REQUIRE(nm_c == nm_leg);
    // MACHINE-IDENTITY: the cached consumption is algebraically identical to the
    // retained-dW consumption (same dW, same eps_inv_head, same transform, same
    // fold order; the mirror copy commutes with the fold bitwise) -- see the memo.
    REQUIRE(std::abs(e_hf_c - e_hf_leg) <= 1e-14 * std::abs(e_hf_leg));
    REQUIRE(std::abs(e_corr_c - e_corr_leg) <= 1e-14);
    // MEMORY BEHAVIOR at the retention site (scf_driver.cpp): with the cache the
    // secondary path frees dW every iteration (plain-GW profile); the legacy switch
    // and the global path retain it (unchanged).
    REQUIRE(not dw_c);
    REQUIRE(dw_leg);
    REQUIRE(dw_g);
#else
    // pre-change recording build: the legacy run above IS the current production
    // behavior; both modes retain dW.
    REQUIRE(dw_leg);
    REQUIRE(dw_g);
#endif
#endif  // ENABLE_DLR
  }

#if defined(ENABLE_DLR) && defined(VERTEX_WCACHE_API)
  // ====================================================================================
  // Kernel-output identity on one physical (G, W) state: eval_Pi_C consuming the CACHE
  // vs the LEGACY retained-dW branch, bitwise; ignore_g0 and gygi head policies.
  TEST_CASE("vertex_wcache_identity", "[methods][vertex][wcache][smoke]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_wcache_id";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    // physical state: one plain scGW iteration, then an RPA-W rebuild so that
    // mb_state holds a CONSISTENT (G, dW, eps_inv_head) triple (the same isolation
    // as the refinement2/conservation tests)
    solvers::hf_t hf;
    solvers::gw_t gw(&ft, "ignore_g0", output);
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
    simple_dyson dyson(mf.get(), &ft);
    MBState mb_state(mpi_context, ft, output);
    iter_scf::iter_scf_t iter_sol("damping");
    auto [e_hf_0, e_corr_0] = scf_loop(mb_state, dyson, eri, ft,
                                       solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                       1, false, 1e-9, true);
    mpi_context->comm.barrier();
    REQUIRE(std::isfinite(e_hf_0));
    REQUIRE(std::isfinite(e_corr_0));
    REQUIRE(not mb_state.dW_qtPQ.has_value());   // freed by the driver (no vertex)

    auto MF_ptr = thc.MF();
    const long nqpts_ibz = MF_ptr->nqpts_ibz();
    const long Np = thc.Np();
    const long nt = mb_state.sG_tskij.value().local().shape(0);
    const long nt_half = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;
    const std::array<long, 4> pgrid = {1, 1, 1, mpi_context->comm.size()};
    const std::array<long, 4> bsize = {1, 1, 1, 1};
    const std::array<long, 4> gshape = {nt_half, nqpts_ibz, Np, Np};

    auto max_abs = [](auto const& a) {
      double m = 0.0;
      for (auto const& v : a) m = std::max(m, std::abs(v));
      return m;
    };
    auto max_diff = [](auto const& a, auto const& b) {
      double m = 0.0;
      auto ita = a.begin();
      auto itb = b.begin();
      for (; ita != a.end(); ++ita, ++itb) m = std::max(m, std::abs(*ita - *itb));
      return m;
    };

    auto check_policy = [&](std::string const& div) {
      solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(1, 3), mf->nbnd(),
                            div, "secondary", 32, 1e-8);
      REQUIRE(vtx.active());
      REQUIRE(not vtx.has_cached_w());

      // ---- first-iteration semantics: empty cache + no dW -> bare-Zbar rung --------
      if (div == "ignore_g0") {
        REQUIRE(not mb_state.dW_qtPQ.has_value());
        auto dPi_bare = vtx.eval_Pi_C(mb_state, thc, pgrid, bsize, gshape);
        double m0 = max_abs(dPi_bare.local());
        double m0g = m0;
        mpi_context->comm.all_reduce_in_place_n(&m0g, 1, mpi3::max<>{});
        app_log(1, "wcache id: bare-Zbar rung (empty cache, no dW): max|Pi^C| = {}", m0g);
        REQUIRE(std::isfinite(m0));
        REQUIRE(m0g > 1e-12);
        REQUIRE(not vtx.has_cached_w());   // eval never fills the cache
      }

      // ---- build W once (dW + eps_inv_head now in mb_state) -------------------------
      if (not mb_state.dW_qtPQ.has_value()) {
        scr_eri.update_w(mb_state, thc, -1);   // no vertex attached: no cache fill
        REQUIRE(mb_state.dW_qtPQ.has_value());
        REQUIRE(mb_state.eps_inv_head.has_value());
      }

      // ---- A: LEGACY consumption (cache empty, dW present -> fold at eval) ----------
      auto dPi_legacy = vtx.eval_Pi_C(mb_state, thc, pgrid, bsize, gshape);
      REQUIRE(not vtx.has_cached_w());

      // ---- fill the cache from the SAME mb_state (the update_w-tail seam) -----------
      vtx.cache_w(mb_state, thc);
      REQUIRE(vtx.has_cached_w());

      // ---- B: CACHED consumption on the identical state -----------------------------
      auto dPi_cached = vtx.eval_Pi_C(mb_state, thc, pgrid, bsize, gshape);

      double scale = max_abs(dPi_legacy.local());
      double diff = max_diff(dPi_legacy.local(), dPi_cached.local());
      double g[2] = {scale, diff};
      mpi_context->comm.all_reduce_in_place_n(g, 2, mpi3::max<>{});
      app_log(1, "wcache id [{}]: max|Pi^C(legacy)| = {}, max|cached - legacy| = {} "
                 "(bitwise expected)", div, g[0], g[1]);
      REQUIRE(g[0] > 1e-12);
      // MACHINE-IDENTITY of the consumption (memo section 2): same arithmetic on the
      // same data; only the scheduling moved. 1e-14 headroom per the task spec --
      // the measured value is expected to be exactly 0.
      REQUIRE(g[1] <= 1e-14 * g[0]);

      // reset_w_cache restores the legacy branch (A reproduced)
      vtx.reset_w_cache();
      REQUIRE(not vtx.has_cached_w());
    };

    check_policy("ignore_g0");
    // gygi: the cached Wbar(Gamma) carries the head augmentation folded at FILL time
    // with the same-iteration eps_inv_head -- must be bitwise the legacy at-eval
    // augmentation (q0_head_treatment.md; wbar_cache.md section 2).
    check_policy("gygi");

    if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
    mpi_context->comm.barrier();
  }
#endif  // ENABLE_DLR && VERTEX_WCACHE_API

} // namespace bdft_tests
