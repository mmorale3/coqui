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

// scGW-tilde increment C0, gate C0-a (notes/scgwt_implementation_plan.md): with the
// scGW-tilde knobs OFF (or present but inert) the scGW loop reproduces the pre-scgwt
// tree BIT-FOR-BIT. Mirrors test_vertex_noop: the scaffolding guarantees inertness
// structurally (pol_vertex = "none" or an empty ladder C-window stores knobs and touches
// nothing; cvv_rspace_tol is stored and unconsumed until C4; div_treatment = "cvv"
// aborts at parse), and this test asserts it END TO END on LiH-222 by running the
// identical scGW loop three ways --
//   (a) baseline: no scgwt call at all,
//   (b) inert ladder, production wiring for vertex_type = "none" (vertex NOT attached),
//   (c) inert ladder on an enabled-but-empty vertex, attached to BOTH solver seams --
// and requiring exact (bitwise) equality of e_hf and e_corr. Any FP-level perturbation
// of the knobs-off path fails this test.

#include <cmath>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"

#include "numerics/imag_axes_ft/IAFT.hpp"
// pulls the same known-good include closure as the sibling vertex tests
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

  TEST_CASE("scgwt_noop_knobs_off", "[methods][vertex][scgwt][noop]") {
#ifndef ENABLE_DLR
    SUCCEED("scgwt_noop_knobs_off skipped: build has ENABLE_DLR=OFF.");
#else
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_scgwt_noop";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    { // knob round trip: modes/kernels store and report; the empty C-window ladder is
      // INERT (pol_vertex_active() false) -- the C = empty convention of the vertex.
      solvers::vertex_t v(&ft, "none", nda::range(0, 0), mf->nbnd());
      REQUIRE(not v.pol_vertex_enabled());
      REQUIRE(not v.pol_vertex_active());
      v.set_pol_vertex("none", "w0_frozen", nda::range(0, 0), -1, 1e-8, -1.0, -1.0, -1.0);
      REQUIRE(v.pol_vertex() == "none");
      REQUIRE(v.pol_vertex_kernel() == "w0_frozen");
      REQUIRE(not v.pol_vertex_enabled());
      v.set_pol_vertex("ladder", "w0_prev", nda::range(0, 0), -1, 1e-8, -1.0, -1.0, -1.0);
      REQUIRE(v.pol_vertex() == "ladder");
      REQUIRE(v.pol_vertex_kernel() == "w0_prev");
      REQUIRE(v.pol_vertex_enabled());
      REQUIRE(not v.pol_vertex_active());   // empty window => inert, no abort

      // the CVV head scaffold stores its knob
      solvers::cvv_head_t cvv(&ft, 3.14e-7);
      REQUIRE(cvv.rspace_tol() == 3.14e-7);
    }

    // wiring == the production driver's gw branch; 2 scf iterations, damping.
    auto run = [&](bool with_inert_ladder, bool with_empty_vertex) {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");

      solvers::vertex_t vtx(&ft, with_empty_vertex ? "2nd_exchange" : "none",
                            nda::range(0, 0), mf->nbnd());
      if (with_inert_ladder) {
        // enabled-but-EMPTY ladder + a non-default (unconsumed until C4) cvv tolerance:
        // both must be exact no-ops.
        vtx.set_pol_vertex("ladder", "w0_prev", nda::range(0, 0), -1, 1e-8, -1.0, -1.0, -1.0);
        scr_eri.set_cvv_rspace_tol(3.14e-7);
      }
      if (vtx.enabled()) {   // production attachment rule (MBPT_drivers gw branch)
        scr_eri.set_vertex(&vtx);
        gw.set_vertex(&vtx);
      }
      REQUIRE(not scr_eri.has_active_vertex());

      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     2, false, 1e-9, true);
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return std::make_pair(e_hf, e_corr);
    };

    auto [e_hf_0, e_corr_0] = run(false, false);
    app_log(1, "scgwt_noop: baseline                     e_hf = {}, e_corr = {}",
            e_hf_0, e_corr_0);

    // BIT-IDENTITY: the runs execute the identical instruction stream (only stored
    // knob values differ), so the results must be exactly equal -- not merely close.
    auto [e_hf_1, e_corr_1] = run(true, false);
    app_log(1, "scgwt_noop: inert ladder, no vertex      e_hf = {}, e_corr = {}",
            e_hf_1, e_corr_1);
    REQUIRE(e_hf_1 == e_hf_0);
    REQUIRE(e_corr_1 == e_corr_0);

    auto [e_hf_2, e_corr_2] = run(true, true);
    app_log(1, "scgwt_noop: inert ladder, empty vertex   e_hf = {}, e_corr = {}",
            e_hf_2, e_corr_2);
    REQUIRE(e_hf_2 == e_hf_0);
    REQUIRE(e_corr_2 == e_corr_0);
#endif
  }

} // bdft_tests
