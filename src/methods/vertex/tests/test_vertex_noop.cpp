/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
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

// ISDF-Vertex Phase 2 regression: C = empty set reproduces plain scGW BIT-FOR-BIT.
//
// The primary correctness gate of the vertex plan (theory notes CLAUDE.md section 2,
// invariant 5): enabling vertex_type = "2nd_exchange" with an EMPTY subspace window
// must be an exact no-op. The scaffold guarantees this structurally (vertex_t::active()
// is false => every injection guard is false => no allocation, no arithmetic, and the
// scf driver's W-lifetime exception does not trigger since has_active_vertex() is
// false). This test asserts the guarantee END TO END on the LiH-222 nosym system by
// running the identical scGW loop twice -- vertex absent vs vertex enabled-but-empty,
// attached to BOTH solver seams exactly like the production driver -- and requiring
// exact (bitwise) equality of e_hf and e_corr. Any FP-level perturbation of the
// disabled path (reordered arithmetic, stray allocation-induced nondeterminism, a
// guard evaluated on the wrong object) fails this test.
//
// INCREMENT S1 (notes/static_vertex_implementation_plan.md sections 2.1, 4) extends the
// gate to ALL THREE rung modes: with C = empty set the vertex is inert whichever theory
// was requested, so "dynamic" (Formulation B), "static" (B-S) and "linear" (B-L) must
// each reproduce the vertex-absent run BITWISE -- even though the B-S/B-L kernels do not
// exist yet. This is what makes the mode plumbing provably free of behavior change and
// pins the ordering of the two constructor guards (the C = empty inert path is reached
// BEFORE the "kernels not implemented" abort).

#include <cmath>
#include <random>
#include <vector>

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

namespace bdft_tests {

  using namespace methods;

  TEST_CASE("vertex_noop_c_empty", "[methods][vertex][noop]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_noop_c_empty skipped: build has ENABLE_DLR=OFF.");
#else
    auto& mpi_context = utils::make_unit_test_mpi_context();
    // Same imaginary-axis settings as the vertex smoke tests (the vertex never runs
    // here, but identical settings keep this comparable with the smoke logs).
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_noop";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    { // the vertex_rung default is "dynamic" (key absent == parent behavior), and the
      // string <-> enum map round-trips
      solvers::vertex_t vdef(&ft, "2nd_exchange", nda::range(0, 0), mf->nbnd());
      REQUIRE(vdef.rung() == solvers::dynamic_rung);
      REQUIRE(vdef.rung_str() == "dynamic");
      REQUIRE(solvers::string_to_vertex_rung_enum("static") == solvers::static_rung);
      REQUIRE(solvers::string_to_vertex_rung_enum("linear") == solvers::linear_rung);
      REQUIRE(solvers::vertex_rung_enum_to_string(solvers::static_rung) == "static");
      REQUIRE(solvers::vertex_rung_enum_to_string(solvers::linear_rung) == "linear");
    }

    auto run = [&](bool with_empty_vertex, std::string rung) {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");

      // enabled-but-EMPTY vertex: vertex_type set, C = empty set. active() == false.
      // The rung mode travels the driver's constructor path (all intermediate arguments
      // at their driver defaults); with C = empty it must never reach a kernel guard.
      solvers::vertex_t vtx(&ft, with_empty_vertex ? "2nd_exchange" : "none",
                            nda::range(0, 0), mf->nbnd(), "ignore_g0", "global",
                            -1, 1e-8, -1.0, -1.0, rung);
      REQUIRE(vtx.enabled() == with_empty_vertex);
      REQUIRE(not vtx.active());
      REQUIRE(vtx.rung() == solvers::string_to_vertex_rung_enum(rung));
      REQUIRE(vtx.rung_str() == rung);
      if (vtx.enabled()) {
        // production-like attachment: BOTH seams (MBPT_drivers gw branch pattern).
        scr_eri.set_vertex(&vtx);
        gw.set_vertex(&vtx);
      }
      REQUIRE(not scr_eri.has_active_vertex());   // W-lifetime exception must not trigger

      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     2, false, 1e-9, true);
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return std::make_pair(e_hf, e_corr);
    };

    auto [e_hf_0, e_corr_0] = run(false, "dynamic");
    app_log(1, "vertex_noop_c_empty: vertex off               e_hf = {}, e_corr = {}",
            e_hf_0, e_corr_0);

    // BIT-IDENTITY: every one of these runs must execute the identical instruction stream
    // (guards only differ), so the results must be exactly equal -- not merely close.
    for (std::string rung : {"dynamic", "static", "linear"}) {
      auto [e_hf_1, e_corr_1] = run(true, rung);
      app_log(1, "vertex_noop_c_empty: on, C=empty, {:<8} e_hf = {}, e_corr = {}",
              rung, e_hf_1, e_corr_1);
      REQUIRE(e_hf_1 == e_hf_0);
      REQUIRE(e_corr_1 == e_corr_0);
    }
#endif
  }

} // bdft_tests
