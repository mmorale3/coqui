/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Smoke test for the real-axis GW dispatcher (methods::mbpt with
 * solver_type=="real_axis_gw"). Builds an mb_eri_t from a THC reader
 * on the qe_lih222 fixture, populates a ptree with G_0W_0 settings,
 * and invokes methods::mbpt. Asserts that the dispatch path runs to
 * completion -- the underlying physics is exercised by
 * test_real_axis_scf_loop_lih222_periodic and the xvalidate test.
 * ==========================================================================
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"

#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"

#include "mean_field/default_MF.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/MBPT_drivers.h"

#include <string>

namespace bdft_tests {

  using namespace methods;

  // ===========================================================================
  // Smoke test: methods::mbpt("real_axis_gw", thc_eri, pt) runs through
  // a single G_0W_0 iteration on qe_lih222 without error.
  // ===========================================================================
  TEST_CASE("real_axis_gw_dispatch_lih222_g0w0",
            "[real_axis][dispatch][thc][qe][bdft]")
  {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(
                  mf::default_MF(mpi_context, "qe_lih222"));
    const int nIpts = mf->nbnd() * 8;
    thc_reader_t thc(mf, make_thc_reader_ptree(nIpts, "", "incore", "",
                                               "bdft", 1e-8,
                                               mf->ecutrho(), 1, 1024));
    auto mb_eri = mb_eri_t<thc_reader_t>(thc);

    ptree pt;
    pt.put<double>("beta",          50.0);
    pt.put<int>   ("niter",         1);          // G_0W_0
    pt.put<double>("conv_thr",      1e-3);
    pt.put<std::string>("div_treatment",     "ignore_g0");
    pt.put<std::string>("hf_div_treatment",  "ignore_g0");
    pt.put<std::string>("screen_type",       "rpa");
    pt.put<std::string>("mix_kind",          "linear");
    pt.put<double>("alpha_mix",     0.5);
    pt.put<double>("eta",           0.05);
    pt.put<double>("eps_nufft",     1e-8);
    pt.put<long>  ("N_w",           65);
    pt.put<long>  ("N_Omega",       32);
    pt.put<long>  ("N_t",           128);
    pt.put<bool>  ("update_mu",     true);
    pt.put<bool>  ("verbose",       false);

    // Should run to completion without throwing / aborting.
    methods::mbpt(std::string("real_axis_gw"), mb_eri, pt);

    // No state artifact is exposed by methods::mbpt(); the dispatcher just
    // logs the SCF result. Reaching this point means dispatch succeeded.
    REQUIRE(true);
  }

} // namespace bdft_tests
