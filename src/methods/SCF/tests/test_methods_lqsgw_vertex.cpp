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

#include <cstdio>
#include <cmath>

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

/**
 * The linearized quasiparticle self-consistent GW loop (qp_approx = "lqp") with the vertex, wired as the [lqsgw]
 * driver does it (MBPT_drivers.cpp): a pol-vertex knob carrier attached to scr_eri carries the ladder in P
 * (pol_vertex = "ladder", pol_vertex_inject = "ladder_n2") and, optionally, the pair-resolved ladder vertex in Sigma
 * (pol_vertex_sigma = "pair"). LiH 2x2x2 (the lqsgw_lih222 setup of test_scf_common), C = [1, 3), 3 iterations:
 *   - plain LQSGW, vertex in P, vertex in P + Sigma;
 *   - the vertex is active: each leg moves the energy and the gap away from the previous one;
 *   - regression references for the three legs.
 * The vertex in Sigma switches on the causal projection of Sigma_c before the linearization (the default rule of
 * the Dyson loop), so the P + Sigma leg also exercises that stage.
 */

namespace bdft_tests {

  using namespace methods;

  namespace lqsgw_vertex_detail {

    constexpr double HA2EV = 27.211386245988;

    struct lq_row {
      double e_tot = 0.0;
      double e_homo = 0.0, e_lumo = 0.0;
      long final_iter = -1;
      double lam_nu0 = -1.0;
      double gap_eV() const { return (e_lumo - e_homo) * HA2EV; }
    };

    inline lq_row run_lqsgw(auto &mpi_context, std::shared_ptr<mf::MF> &mf, imag_axes_ft::IAFT &ft,
                            std::string const &tag, std::string const &pol_mode, std::string const &inject,
                            nda::range window, bool sigma_pair, int niter) {
      const std::string output = "lqsgw_vertex_" + tag;
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");

      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*8, "", "incore", "", output, 1e-8, mf->ecutrho(), 1, 512));
      auto eri = mb_eri_t(thc, thc);
      qp_context_t qp_params{.qp_approx = "lqp", .qp_scf_mode = "qpscf",
                             .lqp = {.n_fit = 4, .fit_order = -1, .symmetric_window = true, .fit_resid_tol = 1e-8}};

      // the [lqsgw] driver's knob carrier, attached to scr_eri only; it must outlive qp_scf_loop
      solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd(), "ignore_g0");
      vtx.set_pol_vertex(pol_mode, "w0_prev", window, -1, 1e-8, -1.0, -1.0, -1.0, inject);
      if (sigma_pair) vtx.set_sigma_pair(true, "static", "dynamic", 1.0, true, false);
      if (vtx.pol_vertex_enabled()) scr_eri.set_vertex(&vtx);
      REQUIRE(not scr_eri.has_active_vertex());      // no Sigma^C / Pi^C vertex on this path
      REQUIRE(scr_eri.sigma_vertex_active() == sigma_pair);

      iter_scf::iter_scf_t iter_sol(iter_scf::damp_t(0.7));
      MBState mb_state(mpi_context, ft, output);
      lq_row row;
      row.e_tot = qp_scf_loop(mb_state, eri, ft, qp_params, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                              niter, false, 1e-6);
      mpi_context->comm.barrier();
      row.lam_nu0 = scr_eri.pol_lambda_nu0();

      nda::array<ComplexType, 3> E_ska;
      {
        h5::file file(output + ".mbpt.h5", 'r');
        auto scf_grp = h5::group(file).open_group("scf");
        h5::h5_read(scf_grp, "final_iter", row.final_iter);
        nda::h5_read(scf_grp.open_group("iter" + std::to_string(row.final_iter)), "E_ska", E_ska);
      }
      mpi_context->comm.barrier();
      const int homo = int(mf->nelec()/2 - 1), lumo = int(mf->nelec()/2);
      row.e_homo = -1e9;
      row.e_lumo = 1e9;
      for (long ik = 0; ik < E_ska.shape(1); ++ik) {
        row.e_homo = std::max(row.e_homo, E_ska(0, ik, homo).real());
        row.e_lumo = std::min(row.e_lumo, E_ska(0, ik, lumo).real());
      }
      if (mpi_context->comm.root()) std::remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      std::printf("lqsgw_vertex [%s]: final iter %ld, e_tot = %.12f, gap = %.6f eV, lambda_max(nu = 0) = %.6f\n",
                  tag.c_str(), row.final_iter, row.e_tot, row.gap_eV(), row.lam_nu0);
      return row;
    }

  } // lqsgw_vertex_detail

  TEST_CASE("lqsgw_vertex_lih222", "[methods][lqsgw][vertex]") {
#ifndef ENABLE_DLR
    SUCCEED("lqsgw_vertex_lih222 skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace lqsgw_vertex_detail;
    auto &mpi_context = utils::make_unit_test_mpi_context();
    // the setup of test_scf_common's lqsgw_lih222 (beta = 100, wmax = 7 covers the spectrum; the plain leg reproduces
    // its reference)
    imag_axes_ft::IAFT ft(100.0, 7.0, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    constexpr int niter = 3;

    auto r0 = run_lqsgw(mpi_context, mf, ft, "plain", "none", "none", nda::range(0, 0), false, niter);
    auto r1 = run_lqsgw(mpi_context, mf, ft, "p", "ladder", "ladder_n2", nda::range(1, 3), false, niter);
    auto r2 = run_lqsgw(mpi_context, mf, ft, "ps", "ladder", "ladder_n2", nda::range(1, 3), true, niter);

    REQUIRE(r0.final_iter == niter);
    REQUIRE(r1.final_iter == niter);
    REQUIRE(r2.final_iter == niter);
    // the ladder ran (w0_prev: from the second update on) and each vertex leg moves the solution
    REQUIRE(r1.lam_nu0 > 0.0);
    REQUIRE(std::abs(r1.e_tot - r0.e_tot) > 1e-6);
    REQUIRE(std::abs(r2.e_tot - r1.e_tot) > 1e-6);
    REQUIRE(std::abs(r1.gap_eV() - r0.gap_eV()) > 1e-4);
    REQUIRE(std::abs(r2.gap_eV() - r1.gap_eV()) > 1e-4);

    // Regression references (macOS, clang + OpenBLAS, 2 ranks). The tolerances cover the platform spread of the
    // quasiparticle loops (1e-6 Ha class) and the roundoff amplification of the causal projection in the P + Sigma leg.
    // The plain leg agrees with lqsgw_lih222 of test_scf_common (-4.2355209995661243) to 1e-10.
    constexpr double e_ref[3] = {-4.235520999502, -4.235212678811, -4.236097567226};
    constexpr double gap_ref[3] = {11.752218, 11.746762, 11.754767};
    lq_row const *rows[3] = {&r0, &r1, &r2};
    for (int i = 0; i < 3; ++i) {
      REQUIRE(std::abs(rows[i]->e_tot - e_ref[i]) < 2e-5);
      REQUIRE(std::abs(rows[i]->gap_eV() - gap_ref[i]) < 1e-3);
    }
#endif
  }

} // bdft_tests
