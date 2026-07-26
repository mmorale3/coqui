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


#undef NDEBUG

#include "catch2/catch.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"
#include "utilities/mpi_context.h"

#include "nda/h5.hpp"

#include "IO/ptree/ptree_utilities.hpp"
#include "mean_field/default_MF.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/mb_solver_t.h"
#include "methods/SCF/simple_dyson.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/SCF/scf_driver.hpp"

namespace bdft_tests {

  using namespace methods;
  using math::shm::make_shared_array;
  using math::shm::shared_array;

  namespace {
    template<int N>
    double max_abs_diff_gw(nda::array<ComplexType,N> const& A,
                           nda::array<ComplexType,N> const& B) {
      REQUIRE(A.shape() == B.shape());
      double m = 0.0;
      auto const* pa = A.data();
      auto const* pb = B.data();
      for (long i = 0; i < A.size(); ++i)
        m = std::max(m, std::abs(pa[i] - pb[i]));
      return m;
    }
  }

  TEST_CASE("hamilt_gw_hf_slot", "[methods][hamilt][gw][qe][paw]") {
    // Static-route plan phase 3.3: one Dyson GW iteration with the hf slot on
    // the direct route vs THC. The static Σ (F_skij) must agree to the THC
    // fitting tolerance; the DYNAMIC self-energy at iteration 1 is built from
    // the mean-field G with the SAME thc corr object in both runs, so it must
    // be identical to numerical determinism (route independence of the
    // dynamics, invariant I7).
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_source);

    auto check_gw = [&](std::string mf_id) {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, mf_id));
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*20, "", "incore", "", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      hamilt_eval_t heval(mf, ptree{});

      struct gw_run { double e_hf, e_corr; nda::array<ComplexType,4> F;
                      nda::array<ComplexType,5> Sigma; };
      auto run_one_iter = [&](auto& eri, std::string prefix) {
        solvers::hf_t hf(methods::ignore_g0);
        solvers::gw_t gw(&ft, string_to_div_enum("ignore_g0"), prefix);
        solvers::scr_coulomb_t scr_eri(&ft, "rpa", string_to_div_enum("ignore_g0"));
        simple_dyson dyson(mf.get(), &ft);
        iter_scf::iter_scf_t iter_sol("damping");
        MBState mb_state(mpi_context, ft, prefix);
        gw_run r;
        std::tie(r.e_hf, r.e_corr) = scf_loop(mb_state, dyson, eri, ft,
                                              solvers::mb_solver_t(&hf, &gw, &scr_eri),
                                              &iter_sol, 1, false, 1e-9, true);
        {
          h5::file file(prefix + ".mbpt.h5", 'r');
          auto g1 = h5::group(file).open_group("scf").open_group("iter1");
          nda::h5_read(g1, "F_skij", r.F);
          nda::h5_read(g1, "Sigma_tskij", r.Sigma);
        }
        mpi_context->comm.barrier();
        if (mpi_context->comm.root())
          remove((prefix + ".mbpt.h5").c_str());
        mpi_context->comm.barrier();
        return r;
      };

      auto eri_ref = mb_eri_t(thc, thc);
      auto ref = run_one_iter(eri_ref, "cq_gw_thchf");
      auto eri_ham = mb_eri_t(heval, thc);
      auto ham = run_one_iter(eri_ham, "cq_gw_hamhf");

      app_log(1, "[gw hf-slot {}] e_hf thc = {:.10f}  hamilt = {:.10f}  |diff| = {:.3e}",
              mf_id, ref.e_hf, ham.e_hf, std::abs(ref.e_hf - ham.e_hf));
      app_log(1, "[gw hf-slot {}] e_corr |diff| = {:.3e}  max|dF| = {:.3e}  max|dSigma| = {:.3e}",
              mf_id, std::abs(ref.e_corr - ham.e_corr),
              max_abs_diff_gw<4>(ref.F, ham.F), max_abs_diff_gw<5>(ref.Sigma, ham.Sigma));

      // Static route difference: THC fitting scale.
      VALUE_EQUAL(ham.e_hf, ref.e_hf, 1e-4);
      ARRAY_EQUAL(ham.F, ref.F, 1e-4);
      // Dynamics: same G0, same corr ERIs — route-independent to determinism.
      ARRAY_EQUAL(ham.Sigma, ref.Sigma, 1e-11);
      // e_corr is evaluated on the UPDATED G (a function of F through Dyson),
      // so it inherits the static-route THC-scale difference in damped form
      // (measured ~7e-8 here) — the dynamics invariant is the Sigma assertion.
      VALUE_EQUAL(ham.e_corr, ref.e_corr, 1e-5);
    };

    SECTION("lih222_paw")     { check_gw("qe_lih222_paw"); }
    // Symmetric mesh (static-route plan 4.3b; phase-4 View-2 nij lift).
    SECTION("lih222_paw_sym") { check_gw("qe_lih222_paw_sym"); }
  }

} // bdft_tests
