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

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"
#include <filesystem>

#include "mean_field/default_MF.hpp"
#include "methods/GW/g0_div_utils.hpp"   // the q -> 0 head: the reduction-independence gate below

#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"

// TODO add lih_223 unit tests

namespace bdft_tests {

  using namespace methods;

  TEST_CASE("thc_g0w0_qe_bdft", "[methods][thc][gw][qe][bdft]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_basis, "high");
    std::string output = "coqui";

    auto solve_thc_g0w0 = [&](std::shared_ptr<mf::MF> &mf) {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");

      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*24, "", "incore", "", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      qp_params_t qp_params("sc", "pade", 18, 0.0001, 1e-8, "evscf");
      iter_scf::iter_scf_t iter_sol("damping");
      MBState mb_state(mpi_context, ft, output);
      [[maybe_unused]] double e_hf = qp_scf_loop(mb_state, eri, ft, qp_params,
                                      solvers::mb_solver_t(&hf,&gw,&scr_eri), &iter_sol, 1, false, 1e-8);
      mpi_context->comm.barrier();

      nda::array<ComplexType, 3> E_ska;
      {
        h5::file file(output+".mbpt.h5", 'r');
        auto scf_grp = h5::group(file).open_group("scf");
        auto iter_grp = scf_grp.open_group("iter1");
        nda::h5_read(iter_grp, "E_ska", E_ska);
      }
      /**
       * Reference value is obtained from Chol-G0W0 with ERIs converge to 1e-10.
       * The accuracy is roughly 1e-5 at alpha=24 for this system in the presence of AC.
       **/
      int homo = int(mf->nelec()/2 - 1);
      int lumo = int(mf->nelec()/2);
      app_log(2, "E_ska at k = 0: {0:.12f}, {1:.12f}, {2:.12f}, {3:.12f}",
              E_ska(0,0,homo-1).real(), E_ska(0,0,homo).real(),
              E_ska(0,0,lumo).real(), E_ska(0,0,lumo+1).real());
      VALUE_EQUAL(E_ska(0,0,homo-1).real(), -1.959166853350, 1e-5);
      VALUE_EQUAL(E_ska(0,0,homo).real(), -0.343590135344, 1e-5);
      VALUE_EQUAL(E_ska(0,0,lumo).real(), 0.769452793794, 1e-5);
      VALUE_EQUAL(E_ska(0,0,lumo+1).real(), 0.819356108320, 1e-5);

      app_log(2, "E_ska at k = 1: {0:.12f}, {1:.12f}, {2:.12f}, {3:.12f}",
              E_ska(0,1,homo-1).real(), E_ska(0,1,homo).real(),
              E_ska(0,1,lumo).real(), E_ska(0,1,lumo+1).real());
      VALUE_EQUAL(E_ska(0,1,homo-1).real(), -1.949608656698, 1e-5);
      VALUE_EQUAL(E_ska(0,1,homo).real(), -0.234561625134, 1e-5);
      VALUE_EQUAL(E_ska(0,1,lumo).real(), 0.332168314756, 1e-5);
      VALUE_EQUAL(E_ska(0,1,lumo+1).real(), 0.691491471197, 1e-5);
      mpi_context->comm.barrier();

      if (mpi_context->comm.root()) {
        remove((output+".mbpt.h5").c_str());
      }
      mpi_context->comm.barrier();
    };

    SECTION("nosym_qe") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
      solve_thc_g0w0(mf);
    }
    SECTION("sym_qe") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_sym"));
      solve_thc_g0w0(mf);
    }
    SECTION("nosym_bdft") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "bdft_lih222"));
      solve_thc_g0w0(mf);
    }
    SECTION("sym_bdft") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "bdft_lih222_sym"));
      solve_thc_g0w0(mf);
    }
  }

#if defined(ENABLE_DEVICE)
  // Smoke test for imag-axis GW DEVICE_MEMORY path: after a HOST scf_loop
  // initializes mb_state, call scr_coulomb_t::update_w<DEVICE> and
  // gw_t::evaluate<DEVICE> directly to validate the device path runs
  // end-to-end (no host-vs-device validation in this lightweight smoke).
  TEST_CASE("thc_gw_qe_device_smoke", "[methods][thc][gw][qe][device]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_basis, "high");
    std::string output = "coqui_dev";

    solvers::hf_t hf;
    solvers::gw_t gw(&ft, "ignore_g0", output);
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
    simple_dyson dyson(mf.get(), &ft);
    MBState mb_state(mpi_context, ft, output);

    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 20, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);
    iter_scf::iter_scf_t iter_sol("damping");

    // One HOST iter initializes mb_state.
    scf_loop(mb_state, dyson, eri, ft,
             solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
             1, false, 1e-9, true);

    // Now drive the DEVICE_MEMORY explicit instantiations.
    scr_eri.update_w<DEVICE_MEMORY>(mb_state, thc, 0);
    gw.evaluate<DEVICE_MEMORY>(mb_state, thc, true);

    mpi_context->comm.barrier();
    if (mpi_context->comm.root()) {
      remove((output+".mbpt.h5").c_str());
    }
    mpi_context->comm.barrier();
  }
#endif

  TEST_CASE("thc_gw_qe", "[methods][thc][gw][qe]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto solve_thc_gw = [&](
      std::shared_ptr<mf::MF> &mf, double wmax, bool chol_eri_hf=false) {

      imag_axes_ft::IAFT ft(1000, wmax, imag_axes_ft::ir_basis, "high");
      std::string output = "coqui";

      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);

      /**
       * Reference value is obtained from Chol-GW with ERIs converge to 1e-10.
       * The accuracy is roughly 1e-5 at alpha=20 for this system.
       **/
      double e_hf;
      double e_corr;
      if (!chol_eri_hf) {
        thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 20, "", "incore", "", "bdft",
                                                   1e-10, mf->ecutrho(), 1, 1024));
        auto eri = mb_eri_t(thc, thc);
        iter_scf::iter_scf_t iter_sol("damping");
        std::tie(e_hf, e_corr) = scf_loop(mb_state, dyson, eri, ft,
                                          solvers::mb_solver_t(&hf,&gw,&scr_eri), &iter_sol,
                                          1, false, 1e-9, true);
      } else {
        thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 20, "", "incore", "", "bdft",
                                                   1e-10, mf->ecutrho(), 1, 1024));
        chol_reader_t chol(mf, methods::make_chol_reader_ptree(1e-10, mf->ecutrho(), 32, "./"));
        iter_scf::iter_scf_t iter_sol("damping");
        auto eri = mb_eri_t(chol, thc);
        std::tie(e_hf, e_corr) = scf_loop(mb_state, dyson, eri, ft,
                                          solvers::mb_solver_t(&hf,&gw,&scr_eri), &iter_sol,
                                          1, false, 1e-9, true);
      }

      VALUE_EQUAL(e_hf, -4.224737908935479, 1e-5);
      VALUE_EQUAL(e_corr, -0.11256940748889475, 1e-5);
      mpi_context->comm.barrier();

      if (mpi_context->comm.root()) {
        remove((output+".mbpt.h5").c_str());
        if (chol_eri_hf) {
          remove("chol_info.h5");
          for (size_t ik = 0; ik < mf->nqpts(); ++ik) {
            std::string fname = "Vq"+std::to_string(ik)+".h5";
            remove(fname.c_str());
          }
        }
      }
      mpi_context->comm.barrier();
    };

    SECTION("nosym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
      solve_thc_gw(mf, 1.2);
    }
    SECTION("sym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_sym"));
      solve_thc_gw(mf, 1.2);
      solve_thc_gw(mf, 12.0);
    }
    SECTION("nosym_mix_eri") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
      solve_thc_gw(mf, 1.2, true);
    }

  }

  TEST_CASE("thc_rpa_qe", "[methods][thc][rpa][qe]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto solve_thc_rpa = [&](std::shared_ptr<mf::MF> &mf, double wmax) {
      imag_axes_ft::IAFT ft(1000, wmax, imag_axes_ft::ir_basis, "high");

      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "gygi_smallest_q");

      /**
       * Reference value is obtained from Chol-RPA with ERIs converge to 1e-10.
       * The accuracy is roughly 1e-5 at alpha=20 for this system.
       **/
      simple_dyson dyson(mf.get(), &ft);
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*20, "", "incore", "", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      MBState mb_state(mpi_context, ft, "bdft");
      double e_rpa = rpa_loop(mb_state, dyson, eri, ft, solvers::mb_solver_t(&hf, &gw));
      VALUE_EQUAL(e_rpa, -0.07295472568310496, 1e-5);
      mpi_context->comm.barrier();

      if (mpi_context->comm.root()) {
        remove("./thc_eri.h5");
        remove("./bdft.mbpt.h5");
      }
      mpi_context->comm.barrier();
    };

    SECTION("nosym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
      solve_thc_rpa(mf, 1.2);
    }
    SECTION("sym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_sym"));
      solve_thc_rpa(mf, 1.2);
      solve_thc_rpa(mf, 12.0);
    }
  }


  TEST_CASE("thc_gw_pyscf", "[methods][thc][gw][pyscf]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    std::string output = "coqui";
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, mf::pyscf_source));
    imag_axes_ft::IAFT ft(1000, 12.0, imag_axes_ft::ir_basis, "high");
    solvers::hf_t hf;
    solvers::gw_t gw(&ft, "ignore_g0", output);

    /**
     * Reference value is obtained from Chol-GW with ERIs converge to 1e-10.
     * The accuracy is roughly 1e-6 at alpha=25 for this system.
     **/
    { // incore thc-gw
      MBState mb_state(mpi_context, ft, output);
      simple_dyson dyson(mf.get(), &ft);
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*25, "", "incore", "./thc_eri.h5", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf,&gw,&scr_eri), &iter_sol,
                                     1, false, 1e-9, true);

      // Reference value is obtained from Chol-GW with ERIs converge to 1e-10
      VALUE_EQUAL(e_hf, 0.9096946909052888, 1e-6);
      VALUE_EQUAL(e_corr, -0.11439719195215467, 1e-6);
      mpi_context->comm.barrier();
    }

    { // outcore thc-gw from the precomputed thc-eri
      MBState mb_state(mpi_context, ft, output);
      simple_dyson dyson(mf.get(), &ft);
      thc_reader_t thc(mf, "outcore", "./thc_eri.h5");
      auto eri = mb_eri_t(thc, thc);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf,&gw,&scr_eri), &iter_sol,
                                     1, false, 1e-9, true);

      // Reference value is obtained from Chol-GW with ERIs converge to 1e-10
      VALUE_EQUAL(e_hf, 0.9096946909052888, 1e-6);
      VALUE_EQUAL(e_corr, -0.11439719195215467, 1e-6);
      if (mpi_context->comm.root()) {
        remove("./thc_eri.h5");
        remove((output+".mbpt.h5").c_str());
      }
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) {
        remove("./thc_eri.h5");
        remove((output+".mbpt.h5").c_str());
      }
    }
  }

  TEST_CASE("thc_rpa_pyscf", "[methods][thc][rpa][pyscf]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, mf::pyscf_source));
    imag_axes_ft::IAFT ft(1000, 12.0, imag_axes_ft::ir_basis, "high");
    solvers::hf_t hf;
    solvers::gw_t gw(&ft, "gygi_smallest_q");
    solvers::mb_solver_t mb_solver(&hf, &gw);

    { // incore thc-rpa
      simple_dyson dyson(mf.get(), &ft);
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*25, "", "incore", "", "bdft", 1e-10, mf->ecutrho(),
                       1, 1024));
      auto eri = mb_eri_t(thc, thc);
      MBState mb_state(mpi_context, ft, "bdft");
      double e_rpa = rpa_loop(mb_state, dyson, eri, ft, mb_solver);
      VALUE_EQUAL(e_rpa, -0.06481111309877628, 1e-6);
      mpi_context->comm.barrier();
    }

    { // outcore thc-rpa
      simple_dyson dyson(mf.get(), &ft);
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*25, "", "outcore", "./thc_eri.h5", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      MBState mb_state(mpi_context, ft, "bdft");
      double e_rpa = rpa_loop(mb_state, dyson, eri, ft, mb_solver);
      VALUE_EQUAL(e_rpa, -0.06481111309877628, 1e-6);
      if (mpi_context->comm.root()) {
        remove("./thc_eri.h5");
        remove("./bdft.mbpt.h5");
      }
      mpi_context->comm.barrier();
    }
  }

  TEST_CASE("thc_gw_mol", "[methods][thc][gw][pyscf][mol]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    std::string output = "coqui";
    imag_axes_ft::IAFT ft(2000, 6.0, imag_axes_ft::ir_basis, "high");

    auto solve_gdf_thc_gw = [&](std::shared_ptr<mf::MF> &mf, std::string gdf_dir) {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      /**
       * References are obtained from the same GDF-THC-GW with alpha=12
       * The accuracy is roughly 1e-4 at Np=280 (alpha~11.67) for this system.
       **/
      thc_reader_t thc(mf, make_thc_reader_ptree(280, gdf_dir, "incore", "", "bdft",
                                                 0.0, mf->ecutrho(), 1, 1024));
      simple_dyson dyson(mf.get(), &ft);
      auto eri = mb_eri_t(thc, thc);
      iter_scf::iter_scf_t iter_sol("damping");
      MBState mb_state(mpi_context, ft, output);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf,&gw,&scr_eri), &iter_sol,
                                     1, false, 1e-9, false);
      VALUE_EQUAL(e_hf, -84.66602711500559, 1e-4);
      VALUE_EQUAL(e_corr, -0.41696395032933564, 1e-4);

      if (mpi_context->comm.root())
        remove((output+".mbpt.h5").c_str());
      mpi_context->comm.barrier();
    };

    std::string gdf_dir = std::string(PROJECT_SOURCE_DIR)+"/tests/unit_test_files/pyscf/h2o_mol/gdf_eri/";
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_h2o_mol"));
    solve_gdf_thc_gw(mf, gdf_dir);
  }

#ifdef ENABLE_DLR
  TEST_CASE("thc_gw_dlr_vs_ir", "[methods][thc][gw][qe][iaft][dlr][ir]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto solve_thc_gw = [&](std::shared_ptr<mf::MF> &mf, double wmax) {

      imag_axes_ft::IAFT ft(1000, wmax, imag_axes_ft::dlr_basis, "high");
      std::string output = "coqui";

      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);

      /**
       * Reference value is obtained from Chol-GW with ERIs converge to 1e-10 with IR basis.
       * The accuracy is roughly 1e-5 at alpha=20 for this system.
       **/
      double e_hf;
      double e_corr;
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 20, "", "incore", "", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      iter_scf::iter_scf_t iter_sol("damping");
      std::tie(e_hf, e_corr) = scf_loop(mb_state, dyson, eri, ft,
                                        solvers::mb_solver_t(&hf,&gw,&scr_eri), &iter_sol,
                                        1, false, 1e-9, true);

      VALUE_EQUAL(e_hf, -4.224737908935479, 1e-5);
      VALUE_EQUAL(e_corr, -0.11256940748889475, 1e-5);
      mpi_context->comm.barrier();

      if (mpi_context->comm.root()) {
        remove((output+".mbpt.h5").c_str());
      }
      mpi_context->comm.barrier();
    };

    SECTION("sym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_sym"));
      solve_thc_gw(mf, 10.0);
      solve_thc_gw(mf, 100.0);
    }
  }
#endif

} // bdft_tests

namespace bdft_tests {

  // ====================================================================================
  // THE q -> 0 HEAD MUST NOT DEPEND ON HOW THE MESH WAS REDUCED (user ruling 2026-09-22:
  // "we should always use the extrapolated head"; notes/vertex_perf_plan.md, the Si 4^3
  // time-reversal hunt). eps^-1(q) = eps^-1(-q), so +b_i and -b_i carry the SAME physical
  // sample: a head built from the IBZ q LIST with a per-DIRECTION fit order sees different
  // point counts on differently reduced meshes of the same crystal, and moves (Si 4^3:
  // 6.78 on the full mesh vs 8.27 with the time-reversal reduction, which the ladder turns
  // into 7 % of its correction). The axis-folded default merges the two sides and fits once.
  //
  // Driven on a MODEL eps^-1(q) = -1 + 1/(1 + a |q|^2) evaluated on each mesh's own IBZ q
  // list, so the exact head is known analytically (-1 + 1 = 0 ... the q -> 0 limit is 0 in
  // this parameterization, i.e. eps_inv(0) = 0) and the three meshes must agree to the fit's
  // own accuracy. Requires the untracked Si 4^3 fixtures (qe_si444_*); skipped without them.
  // ====================================================================================
  TEST_CASE("gw_head_reduction_independence", "[methods][gw][head]") {
    auto &mpi_context = utils::make_unit_test_mpi_context();
    const std::vector<std::string> fx = {"qe_si444_sym", "qe_si444_noinv", "qe_si444_trevonly"};
    {
      auto [outdir, prefix] = utils::utest_filename(fx[0]);
      if (not std::filesystem::exists(outdir + prefix + ".coqui.h5")) {
        SUCCEED("gw_head_reduction_independence skipped: the Si 4^3 fixtures are not present.");
        return;
      }
    }
    const double a_model = 3.0;
    std::vector<double> head_axis, head_perdir;
    std::vector<long> nq_ibz;
    for (auto const &f : fx) {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, f));
      const long nq = mf->nqpts_ibz();
      nq_ibz.push_back(nq);
      nda::array<ComplexType, 2> eps_inv(1, nq);
      for (long iq = 0; iq < nq; ++iq) {
        auto q = mf->Qpts_ibz(iq);
        const double q2 = q(0) * q(0) + q(1) * q(1) + q(2) * q(2);
        eps_inv(0, iq) = ComplexType(-1.0 + 1.0 / (1.0 + a_model * q2), 0.0);   // smooth, even in q, exact value 0 at q = 0
      }
      head_axis.push_back(methods::solvers::div_utils::extrapolate_eps_inv_q0(eps_inv, *mf, "gygi")(0).real());
      head_perdir.push_back(methods::solvers::div_utils::extrapolate_eps_inv_q0(eps_inv, *mf, "gygi_perdir")(0).real());
    }
    double spread_axis = 0.0, spread_perdir = 0.0;
    for (size_t i = 1; i < head_axis.size(); ++i) {
      spread_axis = std::max(spread_axis, std::abs(head_axis[i] - head_axis[0]));
      spread_perdir = std::max(spread_perdir, std::abs(head_perdir[i] - head_perdir[0]));
    }
    app_log(1, "gw head: the model eps^-1 head at q -> 0 (exact 0) on {} ({} IBZ q), {} ({}), {} ({}): "
               "AXIS-FOLDED (the default) {:.6e} / {:.6e} / {:.6e} -> spread {:.2e}; per-direction (historic) "
               "{:.6e} / {:.6e} / {:.6e} -> spread {:.2e}",
            fx[0], nq_ibz[0], fx[1], nq_ibz[1], fx[2], nq_ibz[2],
            head_axis[0], head_axis[1], head_axis[2], spread_axis,
            head_perdir[0], head_perdir[1], head_perdir[2], spread_perdir);
    REQUIRE(spread_axis < 1e-10);            // the ruling: one head, whatever the reduction
    REQUIRE(spread_perdir > 1e-4);           // the defect it replaces is real, not a rounding difference
    // ... and on a mesh with ONE distinct |q| per axis (every 2x2x2 fixture) the two forms are the SAME fit, so
    // changing the default cannot move any small-fixture number. Measured, not assumed.
    {
      auto mf2 = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_si222_nosym"));
      const long nq2 = mf2->nqpts_ibz();
      nda::array<ComplexType, 2> e2(1, nq2);
      for (long iq = 0; iq < nq2; ++iq) {
        auto q = mf2->Qpts_ibz(iq);
        const double q2 = q(0) * q(0) + q(1) * q(1) + q(2) * q(2);
        e2(0, iq) = ComplexType(-1.0 + 1.0 / (1.0 + a_model * q2), 0.0);
      }
      const double h_ax = methods::solvers::div_utils::extrapolate_eps_inv_q0(e2, *mf2, "gygi")(0).real();
      const double h_pd = methods::solvers::div_utils::extrapolate_eps_inv_q0(e2, *mf2, "gygi_perdir")(0).real();
      app_log(1, "gw head: on qe_si222_nosym ({} IBZ q, one distinct |q| per axis) the two forms agree: {:.12e} vs {:.12e}",
              nq2, h_ax, h_pd);
      REQUIRE(std::abs(h_ax - h_pd) < 1e-14);
    }
    mpi_context->comm.barrier();
  }

} // bdft_tests
