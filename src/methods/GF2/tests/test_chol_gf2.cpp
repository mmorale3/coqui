#undef NDEBUG

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"
#include "mean_field/default_MF.hpp"

#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"

#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"


namespace bdft_tests {

  using namespace methods;

  TEST_CASE("chol_gf2", "[methods]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_h2_222"));
    imag_axes_ft::IAFT ft(1000, 1200, imag_axes_ft::ir_source);
    solvers::hf_t hf;
    solvers::gf2_t gf2(mf.get(), &ft);

    simple_dyson dyson(mf.get(), &ft);

    chol_reader_t chol(mf, methods::make_chol_reader_ptree(1e-8, mf->ecutrho(), 32, "./"));
    auto eri = mb_eri_t(chol, chol);

    iter_scf::iter_scf_t iter_sol("damping");
    MBState mb_state(mpi_context, ft, "bdft");
    auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                   solvers::mb_solver_t(&hf,&gf2), &iter_sol,
                                   1, false, 1e-9, true);

    VALUE_EQUAL(e_corr, -0.24464049055256362, 1e-8); //CD-GF2, tol 1e-8
    VALUE_EQUAL(e_corr, -0.24464049043549288, 1e-5); //THC-GF2 self-check
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


  TEST_CASE("chol_gf2_u", "[methods]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_li_222u"));
    imag_axes_ft::IAFT ft(1000, 1200, imag_axes_ft::ir_source);
    solvers::hf_t hf;
    solvers::gf2_t gf2(mf.get(), &ft);

    simple_dyson dyson(mf.get(), &ft);

    chol_reader_t chol(mf, methods::make_chol_reader_ptree(1e-8, mf->ecutrho(), 32, "./"));
    auto eri = mb_eri_t(chol, chol);

    iter_scf::iter_scf_t iter_sol("damping");
    MBState mb_state(mpi_context, ft, "bdft");
    auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                   solvers::mb_solver_t(&hf,&gf2), &iter_sol,
                                   1, false, 1e-9, true);

    VALUE_EQUAL(e_corr, -0.008663905137277621, 1e-8); //CD-GF2, tol 1e-8
    VALUE_EQUAL(e_corr, -0.008663272247268823, 1e-5); //THC-GF2 self-check
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

} // bdft_tests
