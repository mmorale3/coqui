#undef NDEBUG

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"
#include "mean_field/default_MF.hpp"

#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"


namespace bdft_tests {

  using namespace methods;

  TEST_CASE("thc_gf2_pyscf", "[methods][thc][gf2][pyscf]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    std::string output = "coqui";
    imag_axes_ft::IAFT ft(1000, 1200, imag_axes_ft::ir_source);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_h2_222"));
    solvers::hf_t hf;
    solvers::gf2_t gf2(mf.get(), &ft, string_to_div_enum("gygi"),
                       "gf2", "orb", "gf2", output);

    { // incore thc-gf2
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*10, "", "incore", "./thc_eri.h5", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      simple_dyson dyson(mf.get(), &ft);
      auto eri = mb_eri_t(thc, thc);
      iter_scf::iter_scf_t iter_sol("damping");
      MBState mb_state(mpi_context, ft, output);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft, solvers::mb_solver_t(&hf,&gf2),
                                     std::addressof(iter_sol), 1, false, 1e-9, true);

      VALUE_EQUAL(e_corr, -0.24464049055256362, 1e-5); //CD-GF2, tol 1e-8
      VALUE_EQUAL(e_corr, -0.24464049043549288, 1e-8); //THC-GF2 self-check
      mpi_context->comm.barrier();
    }
    { // outcore thc-gf2
      simple_dyson dyson(mf.get(), &ft);
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*10, "", "outcore", "./thc_eri.h5", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      iter_scf::iter_scf_t iter_sol("damping");
      MBState mb_state(mpi_context, ft, output);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft, solvers::mb_solver_t(&hf,&gf2),
                                     std::addressof(iter_sol), 1, false, 1e-9, true);


      VALUE_EQUAL(e_corr, -0.24464049055256362, 1e-5); //CD-GF2, tol 1e-8
      VALUE_EQUAL(e_corr, -0.24464049043549288, 1e-8); //THC-GF2 self-check
      if (mpi_context->comm.root()) {
        remove("./thc_eri.h5");
        remove((output+".mbpt.h5").c_str());
      }
      mpi_context->comm.barrier();
    }

  }

  TEST_CASE("thc_gf2_u_pyscf", "[methods][thc][gf2][pyscf]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_li_222u"));
    imag_axes_ft::IAFT ft(1000, 1200, imag_axes_ft::ir_source);
    solvers::hf_t hf;
    solvers::gf2_t gf2(mf.get(), &ft);

    { // incore thc-gf2
      simple_dyson dyson(mf.get(), &ft);
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*10, "", "incore", "./thc_eri.h5", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      iter_scf::iter_scf_t iter_sol("damping");
      MBState mb_state(mpi_context, ft, "bdft");
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf,&gf2), &iter_sol,
                                     1, false, 1e-9, true);


      VALUE_EQUAL(e_corr, -0.008663905137277621, 1e-5); //CD-GF2, tol 1e-8
      VALUE_EQUAL(e_corr, -0.008663272247268823, 1e-8); //THC-GF2 self-check
      mpi_context->comm.barrier();
    }
    { // outcore thc-gf2
      simple_dyson dyson(mf.get(), &ft);
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*10, "", "outcore", "./thc_eri.h5", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      iter_scf::iter_scf_t iter_sol("damping");
      MBState mb_state(mpi_context, ft, "bdft");
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf,&gf2), &iter_sol,
                                     1, false, 1e-9, true);


      VALUE_EQUAL(e_corr, -0.008663905137277621, 1e-5); //CD-GF2, tol 1e-8
      VALUE_EQUAL(e_corr, -0.008663272247268823, 1e-8); //THC-GF2 self-check
      if (mpi_context->comm.root()) {
        remove("./thc_eri.h5");
        remove("./bdft.mbpt.h5");
      }
      mpi_context->comm.barrier();
    }

  }


} // bdft_tests
