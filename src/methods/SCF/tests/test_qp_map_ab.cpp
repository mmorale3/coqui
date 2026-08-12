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
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"

/**
 * Project 2 increment Q2 (notes/qpgw_edmft_implementation_plan.md): the A/B
 * surrogate-spread deliverable. The full qp_scf_loop runs with each of the
 * three quasiparticle maps (ac_pade / mats_lin / mats_gmatch) on the same
 * mean field and the same THC factorization, so every difference in the band
 * edges is the surrogate spread of the static map itself (spec section 4
 * "residual ambiguity" -- REPORTED, not converged away). Assertions are loose
 * tripwires against gross breakage; the table in the log is the deliverable.
 */

namespace bdft_tests {

  using namespace methods;

  namespace qp_map_ab_detail {

    constexpr double HA2EV = 27.211386245988;

    struct ab_row {
      std::string map;
      double e_homo, e_lumo;              // band edges over the k mesh
      nda::array<double, 1> Ek0;          // bands [homo-1, lumo+2) at k = 0
    };

    // One qp-scf run with the given map; returns band edges from the
    // checkpoint's final iteration.
    inline ab_row run_map(auto &mpi_context, std::shared_ptr<mf::MF> &mf,
                          imag_axes_ft::IAFT &ft,
                          const std::string &map, const std::string &mode,
                          int thc_prefactor, double thc_tol,
                          int niter, double conv_tol) {
      const std::string output = "qp_map_ab_" + map + "_" + mode;
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");

      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*thc_prefactor, "", "incore", "", output,
                                                 thc_tol, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      qp_params_t qp_params("sc", "pade", 18, 0.0001, 1e-8, mode);
      qp_params.qp_map = map;
      iter_scf::iter_scf_t iter_sol(iter_scf::damp_t(0.7));
      MBState mb_state(mpi_context, ft, output);
      [[maybe_unused]] double e_hf = qp_scf_loop(mb_state, eri, ft, qp_params,
                                                 solvers::mb_solver_t(&hf, &gw, &scr_eri),
                                                 &iter_sol, niter, false, conv_tol);
      mpi_context->comm.barrier();

      nda::array<ComplexType, 3> E_ska;
      long final_it = -1;
      {
        h5::file file(output + ".mbpt.h5", 'r');
        auto scf_grp = h5::group(file).open_group("scf");
        h5::h5_read(scf_grp, "final_iter", final_it);
        auto iter_grp = scf_grp.open_group("iter" + std::to_string(final_it));
        nda::h5_read(iter_grp, "E_ska", E_ska);
      }
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();

      const long nkpts = E_ska.shape(1);
      const int homo = int(mf->nelec()/2 - 1);
      const int lumo = int(mf->nelec()/2);
      ab_row row;
      row.map = map + "(" + mode + ")";
      row.e_homo = -1e9;
      row.e_lumo = 1e9;
      for (long ik = 0; ik < nkpts; ++ik) {
        row.e_homo = std::max(row.e_homo, E_ska(0, ik, homo).real());
        row.e_lumo = std::min(row.e_lumo, E_ska(0, ik, lumo).real());
      }
      row.Ek0.resize(4);
      for (int b = 0; b < 4; ++b) row.Ek0(b) = E_ska(0, 0, homo - 1 + b).real();
      app_log(1, "qp_map_ab [{}]: final iter {}, gap = {:.6f} eV (homo {:.6f}, lumo {:.6f} Ha)",
              row.map, final_it, (row.e_lumo - row.e_homo) * HA2EV, row.e_homo, row.e_lumo);
      return row;
    }

    // THE deliverable: the per-map band-edge table + the per-band spread at
    // k = 0 (the Q1 finding predicts the spread GROWS with |E - mu|).
    inline void report_and_check(std::vector<ab_row> const &rows) {
      app_log(1, "\n== qp_map A/B surrogate-spread table ==");
      app_log(1, "{:<24} {:>12} {:>12} {:>12}", "map", "E_homo (Ha)", "E_lumo (Ha)", "gap (eV)");
      for (auto const &r : rows)
        app_log(1, "{:<24} {:>12.6f} {:>12.6f} {:>12.4f}",
                r.map, r.e_homo, r.e_lumo, (r.e_lumo - r.e_homo) * HA2EV);
      app_log(1, "-- per-band values at k = 0 (eV, band index relative to homo-1) --");
      REQUIRE(rows.size() == 3);
      for (int b = 0; b < 4; ++b) {
        double lo = 1e9, hi = -1e9;
        for (auto const &r : rows) {
          lo = std::min(lo, r.Ek0(b));
          hi = std::max(hi, r.Ek0(b));
        }
        app_log(1, "band {:+d}: {:>12.4f} {:>12.4f} {:>12.4f}  | spread {:.4f} eV", b - 1,
                rows[0].Ek0(b) * HA2EV, rows[1].Ek0(b) * HA2EV, rows[2].Ek0(b) * HA2EV,
                (hi - lo) * HA2EV);
      }

      const double gap_ref = rows[0].e_lumo - rows[0].e_homo;
      for (auto const &r : rows) {
        const double gap = r.e_lumo - r.e_homo;
        REQUIRE(std::isfinite(gap));
        REQUIRE(gap > 0.0);
        REQUIRE(gap < 2.0);
        // loose tripwire (~1.4 eV) against gross map breakage; the expected
        // surrogate class is ~0.1-0.2 eV and is REPORTED above, not asserted.
        REQUIRE(std::abs(gap - gap_ref) < 0.05);
      }
    }

  } // qp_map_ab_detail

  TEST_CASE("qp_map_ab_lih222", "[methods][qpgw][qp_map_ab]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));

    SECTION("qpscf") {
      std::vector<ab_row> rows;
      for (auto const &map : {"ac_pade", "mats_lin", "mats_gmatch"})
        rows.push_back(run_map(mpi_context, mf, ft, map, "qpscf", 12, 1e-10, 20, 1e-6));
      report_and_check(rows);
    }
    SECTION("evscf") {
      std::vector<ab_row> rows;
      for (auto const &map : {"ac_pade", "mats_lin", "mats_gmatch"})
        rows.push_back(run_map(mpi_context, mf, ft, map, "evscf", 12, 1e-10, 20, 1e-6));
      report_and_check(rows);
    }
  }

  TEST_CASE("qp_map_ab_si222", "[methods][qpgw][qp_map_ab]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    // bdft_si222 is commented out of default_MF -- pyscf_si222 is the plan's
    // named alternative (notes/qpgw_edmft_implementation_plan.md section 4).
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_si222"));

    std::vector<ab_row> rows;
    for (auto const &map : {"ac_pade", "mats_lin", "mats_gmatch"})
      rows.push_back(run_map(mpi_context, mf, ft, map, "qpscf", 10, 1e-8, 12, 1e-5));
    report_and_check(rows);
  }

} // bdft_tests
