#undef NDEBUG

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"

#include "mean_field/MF.hpp"
#include "mean_field/default_MF.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/simple_dyson.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/SCF/scf_driver.hpp"

namespace bdft_tests {

  using namespace methods;

  TEST_CASE("chol_gw_qe", "[methods][chol][gw][qe]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    imag_axes_ft::IAFT ft(1000, 1.2e3, imag_axes_ft::ir_source);
    chol_reader_t chol(mf, methods::make_chol_reader_ptree(1e-10, mf->ecutrho(), 32, "./"));
    auto eri = mb_eri_t(chol, chol);
    solvers::gw_t gw(std::addressof(ft));
    solvers::hf_t hf;
    simple_dyson dyson(mf.get(), std::addressof(ft));
    MBState mb_state(mpi_context, ft, "bdft");

    iter_scf::iter_scf_t iter_sol("damping");
    auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                   solvers::mb_solver_t(&hf,&gw), &iter_sol,
                                   1, false, 1e-9, true);

    // Reference value is obtained from Chol-GW with ERIs converge to 1e-10
    VALUE_EQUAL(e_hf, -4.224737908935479, 1e-10);
    VALUE_EQUAL(e_corr, -0.11256940748889475, 1e-10);
    if (mpi_context->comm.root()) {
      remove("chol_info.h5");
      for (size_t ik = 0; ik < mf->nqpts(); ++ik) {
        std::string fname = "Vq"+std::to_string(ik)+".h5";
        remove(fname.c_str());
      }
      remove("bdft.mbpt.h5");
    }
    mpi_context->comm.barrier();
  }

  TEST_CASE("chol_rpa_qe", "[methods][chol][rpa][qe]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));

    imag_axes_ft::IAFT ft(1000, 1.2e3, imag_axes_ft::ir_source);
    chol_reader_t chol(mf, methods::make_chol_reader_ptree(1e-10, mf->ecutrho(), 32, "./"));
    auto eri = mb_eri_t(chol, chol);
    solvers::gw_t gw(std::addressof(ft));
    solvers::hf_t hf;
    solvers::mb_solver_t mb_solver(&hf, &gw);
    simple_dyson dyson(mf.get(), std::addressof(ft));
    MBState mb_state(mpi_context, ft, "bdft");

    double e_rpa = rpa_loop(mb_state, dyson, eri, ft, mb_solver);

    // Reference value is obtained from Chol-RPA with ERIs converge to 1e-10
    VALUE_EQUAL(e_rpa, -0.07295472568310496, 1e-10);
    if (mpi_context->comm.root()) {
      remove("chol_info.h5");
      for (size_t ik = 0; ik < mf->nqpts(); ++ik) {
        std::string fname = "Vq"+std::to_string(ik)+".h5";
        remove(fname.c_str());
      }
      remove("bdft.mbpt.h5");
    }
    mpi_context->comm.barrier();
  }

  TEST_CASE("chol_gw_pyscf", "[methods][chol][gw][pyscf]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/pyscf/si_kp222_krhf/";

    auto mf = std::make_shared<mf::MF>(mf::MF(mf::pyscf::pyscf_readonly(mpi_context, filepath, "pyscf")));

    imag_axes_ft::IAFT ft(1000, 1.2e4, imag_axes_ft::ir_source);
    chol_reader_t chol(mf, methods::make_chol_reader_ptree(1e-10, mf->ecutrho(), 32, "./"));
    auto eri = mb_eri_t(chol, chol);
    solvers::gw_t gw(std::addressof(ft));
    solvers::hf_t hf;
    simple_dyson dyson(mf.get(), std::addressof(ft));
    MBState mb_state(mpi_context, ft, "bdft");

    iter_scf::iter_scf_t iter_sol("damping");
    auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                   solvers::mb_solver_t(&hf,&gw), &iter_sol,
                                   1, false, 1e-9, true);

    // Reference value is obtained from Chol-GW with ERIs converge to 1e-10
    VALUE_EQUAL(e_hf, 0.9096946909052888, 1e-10);
    VALUE_EQUAL(e_corr, -0.11439719195215467, 1e-10);
    if (mpi_context->comm.root()) {
      remove("chol_info.h5");
      for (size_t ik = 0; ik < mf->nqpts(); ++ik) {
        std::string fname = "Vq"+std::to_string(ik)+".h5";
        remove(fname.c_str());
      }
      remove("bdft.mbpt.h5");
    }
    mpi_context->comm.barrier();
  }

  TEST_CASE("chol_rpa_pyscf", "[methods][chol][rpa][pyscf]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/pyscf/si_kp222_krhf/";

    auto mf = std::make_shared<mf::MF>(mf::MF(mf::pyscf::pyscf_readonly(mpi_context, filepath, "pyscf")));

    imag_axes_ft::IAFT ft(1000, 1.2e4, imag_axes_ft::ir_source);
    chol_reader_t chol(mf, methods::make_chol_reader_ptree(1e-10, mf->ecutrho(), 32, "./"));
    auto eri = mb_eri_t(chol, chol);
    solvers::gw_t gw(std::addressof(ft));
    solvers::hf_t hf;
    solvers::mb_solver_t mb_solver(&hf, &gw);
    simple_dyson dyson(mf.get(), std::addressof(ft));
    MBState mb_state(mpi_context, ft, "bdft");

    double e_rpa = rpa_loop(mb_state, dyson, eri, ft, mb_solver);

    // Reference value is obtained from Chol-RPA with ERIs converge to 1e-10
    VALUE_EQUAL(e_rpa, -0.06481111309877628, 1e-10);
    if (mpi_context->comm.root()) {
      remove("chol_info.h5");
      for (size_t ik = 0; ik < mf->nqpts(); ++ik) {
        std::string fname = "Vq"+std::to_string(ik)+".h5";
        remove(fname.c_str());
      }
    }
    mpi_context->comm.barrier();
  }

  TEST_CASE("chol_gw_mol", "[methods][chol][gw][pyscf][mol]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    std::string gdf_dir = std::string(PROJECT_SOURCE_DIR)+"/tests/unit_test_files/pyscf/h2o_mol/gdf_eri/";
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_h2o_mol"));

    imag_axes_ft::IAFT ft(2000, 1.2e4, imag_axes_ft::ir_source);

    chol_reader_t chol(mf, gdf_dir);
    auto eri = mb_eri_t(chol, chol);
    solvers::gw_t gw(&ft);
    solvers::hf_t hf;
    simple_dyson dyson(mf.get(), &ft);
    MBState mb_state(mpi_context, ft, "bdft");

    iter_scf::iter_scf_t iter_sol("damping");
    auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                   solvers::mb_solver_t(&hf,&gw), &iter_sol,
                                   1, false, 1e-9, false);
    VALUE_EQUAL(e_hf, -84.66602711500559, 1e-8);
    VALUE_EQUAL(e_corr, -0.41696395032933564, 1e-8);

    if (mpi_context->comm.root()) {
      remove("bdft.mbpt.h5");
    }
    mpi_context->comm.barrier();
  }

} // bdft_tests
