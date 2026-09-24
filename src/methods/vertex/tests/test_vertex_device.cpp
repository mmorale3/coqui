/*
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

/**
 * ISDF-Vertex on the DEVICE path -- the gate of increment G-1, the host bridge
 * (notes/gpu_port_plan.md section 4). Built only with ENABLE_DEVICE.
 *
 * The same two-iteration LiH-222 scGW is run through scf_loop<HOST_MEMORY> and
 * scf_loop<DEVICE_MEMORY>, once without a vertex and once with an ACTIVE dynamic-rung
 * 2nd-exchange vertex on the window [0, 2). On the device path the GW solver keeps W and
 * Pi on the device while the vertex runs on host copies (keep_host_W, the Pi mirror of
 * eval_Pi_qdep's hooks). The gate: the device run reproduces the host run -- e_corr and
 * the full Sigma(tau) -- to the same tolerance with the vertex ON as it does with the
 * vertex OFF (the vertex-OFF pair measures the device GW's own agreement, which is not
 * this increment's to answer).
 */

#undef NDEBUG
#include <complex>
#include <cmath>
#include <tuple>
#include "catch2/catch.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "mean_field/default_MF.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include "methods/vertex/vertex_t.h"

namespace bdft_tests {

  using namespace methods;
  using cplx = ComplexType;

  TEST_CASE("vertex_device_bridge_lih", "[methods][vertex][device]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_device";
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    // one two-iteration scGW; returns (e_hf, e_corr, Sigma(tau) as a plain host array)
    auto run = [&](auto mem_tag, bool with_vertex) {
      constexpr MEMORY_SPACE MEM = decltype(mem_tag)::value;
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, with_vertex ? "2nd_exchange" : "none",
                            with_vertex ? nda::range(0, 2) : nda::range(0, 0), mf->nbnd(),
                            "ignore_g0", "global", -1, 1e-8, -1.0, -1.0, "dynamic");
      REQUIRE(vtx.active() == with_vertex);
      if (vtx.enabled()) {
        scr_eri.set_vertex(&vtx);
        gw.set_vertex(&vtx);
      }
      auto [e_hf, e_corr] = scf_loop<MEM>(mb_state, dyson, eri, ft,
                                          solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                          2, false, 1e-9, true);
      REQUIRE(mb_state.sSigma_tskij.has_value());
      nda::array<cplx, 5> S(mb_state.sSigma_tskij.value().local());
      if constexpr (MEM != HOST_MEMORY) {
        // the bridge's contract: the host mirror of W is kept exactly when a vertex is attached
        REQUIRE(mb_state.keep_host_W == with_vertex);
      }
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return std::make_tuple(e_hf, e_corr, std::move(S));
    };
    auto max_rel = [&](nda::array<cplx, 5> const& A, nda::array<cplx, 5> const& B) {
      double d = 0.0, m = 0.0;
      for (long i = 0; i < A.size(); ++i) {
        d = std::max(d, std::abs(A.data()[i] - B.data()[i]));
        m = std::max(m, std::abs(A.data()[i]));
      }
      d = mpi_context->comm.all_reduce_value(d, boost::mpi3::max<>{});
      m = mpi_context->comm.all_reduce_value(m, boost::mpi3::max<>{});
      return d / std::max(m, 1e-300);
    };
    using host_t = std::integral_constant<MEMORY_SPACE, HOST_MEMORY>;
    using dev_t  = std::integral_constant<MEMORY_SPACE, DEVICE_MEMORY>;

    // the device GW's own host-vs-device agreement (vertex off)
    auto [eh0_h, ec0_h, S0_h] = run(host_t{}, false);
    auto [eh0_d, ec0_d, S0_d] = run(dev_t{},  false);
    const double dS0 = max_rel(S0_h, S0_d);
    app_log(1, "vertex_device_bridge_lih [vertex OFF]: HOST e_hf = {}, e_corr = {};  DEVICE e_hf = {}, e_corr = {};  "
               "max rel |dSigma| = {:.3e}", eh0_h, ec0_h, eh0_d, ec0_d, dS0);

    // the bridge: vertex on, the vertex on host copies of the device-resident W / Pi
    auto [eh1_h, ec1_h, S1_h] = run(host_t{}, true);
    auto [eh1_d, ec1_d, S1_d] = run(dev_t{},  true);
    const double dS1 = max_rel(S1_h, S1_d);
    app_log(1, "vertex_device_bridge_lih [vertex ON ]: HOST e_hf = {}, e_corr = {};  DEVICE e_hf = {}, e_corr = {};  "
               "max rel |dSigma| = {:.3e}", eh1_h, ec1_h, eh1_d, ec1_d, dS1);
    app_log(1, "vertex_device_bridge_lih: the vertex moves e_corr by {:.6e} (host) / {:.6e} (device)",
            ec1_h - ec0_h, ec1_d - ec0_d);

    constexpr double tol = 1e-6;
    REQUIRE(std::abs(ec0_d - ec0_h) < tol);
    REQUIRE(dS0 < tol);
    REQUIRE(std::abs(ec1_d - ec1_h) < tol);
    REQUIRE(dS1 < tol);
    // the vertex must actually have done something on both paths (an inert bridge would pass the lines above)
    REQUIRE(std::abs(ec1_h - ec0_h) > 1e-10);
    REQUIRE(std::abs(ec1_d - ec0_d) > 1e-10);
  }

} // namespace bdft_tests
