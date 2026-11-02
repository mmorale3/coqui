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

/**
 * The STATIC-RUNG vertex self-consistency (pol_vertex_sc with pol_vertex_rung = "static"): the sublattice ladder resummed
 * with the static rung W-bar_0 alone, injected into P. It is the static column of the dynamic-rung pass, so the two chains
 *   - "dyn":    pol_vertex_rung = "dynamic", pol_vertex_sc_col = "static" (the static column of the dynamic driver),
 *   - "static": pol_vertex_rung = "static" (the static-only driver: no W-bar cache, no dynamic rung, no solve),
 * run end to end through scf_loop on identical inputs (pyscf_si222, the identity sublattice = "direct"), must agree to
 * round-off: the stored vertex polarization, G, Sigma and the energies. The static chain must never fill the W-bar cache.
 *
 * The LINEARIZED rung (pol_vertex_rung = "linearized") on the same chain (the fit weights of an LQSGW window set directly,
 * no Z^1/2): with its dynamic remainder set to zero (vertex_debug lin_static_limit) it reproduces the static rung; with the
 * real W-bar it runs on the same cache as the dynamic rung (the refresh cost is logged next to the static one).
 */

#undef NDEBUG

#include <chrono>
#include <cmath>
#include <complex>
#include <filesystem>
#include <string>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"
#include "IO/app_loggers.h"

#include "numerics/imag_axes_ft/IAFT.hpp"
#include "mean_field/default_MF.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/HF/hf_t.h"
#include "methods/GW/gw_t.h"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include "methods/vertex/vertex_t.h"
#include "methods/vertex/vertex_debug.hpp"
#include "methods/SCF/qp/linearized_qp.hpp"
#include "h5/h5.hpp"
#include "nda/h5.hpp"

namespace bdft_tests {

  using namespace methods;
  using cplx = std::complex<double>;

  namespace {
    struct sc_run {
      double e_hf = 0.0, e_corr = 0.0, wall = 0.0;
      bool cached_w = false;
      nda::array<cplx, 5> G, Sigma;
      nda::array<cplx, 4> Pi;
    };
    /** max |A - B| and max |B| over the elements */
    template <typename A_t>
    std::pair<double, double> max_diff(A_t const &A, A_t const &B) {
      REQUIRE(A.shape() == B.shape());
      double d = 0.0, s = 0.0;
      for (long i = 0; i < A.size(); ++i) {
        d = std::max(d, std::abs(A.data()[i] - B.data()[i]));
        s = std::max(s, std::abs(B.data()[i]));
      }
      return {d, s};
    }
  }

  TEST_CASE("vertex_sc_static_rung", "[methods][vertex][vertex_sc]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_sc_static_rung skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_si222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft", 1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);
    const nda::range C(2, 6);   // two valence and two conduction bands of the 8 (4 occupied)

    auto run = [&](std::string const &rung, std::string const &tag = "", std::string const &dbg = "") {
      const std::string out = "coqui_sc_static_" + rung + tag;
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", out);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, out);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd(), "ignore_g0", "global", -1, 1e-8, -1.0, -1.0, "dynamic");
      vtx.set_pol_vertex("ladder", "w0_prev", C, -1, 1e-8, -1.0, -1.0, -1.0, "ladder_n2");
      vtx.set_ladder_rung(rung, 1e-8, 30, 12, -1.0);
      if (rung == "dynamic") vtx.set_ladder_dyn_gamma1_only(true);   // (the injected static column does not depend on it)
      if (rung == "linearized") {   // the weights of an LQSGW window (n_fit = 6, symmetric, exact order), no Z^1/2
        auto nw = lqp::fit_window(6, true);
        std::vector<long> n(nw.begin(), nw.end());
        lqp::fit_operator_t op;
        const long nn = long(n.size());
        op.fit_order = int(nn) - 1;
        op.M = nda::array<cplx, 2>(nn, nn);
        for (long j = 0; j < nn; ++j) {
          const cplx ix(0.0, double(n[size_t(j)]) / 11.0);
          cplx x = 1.0;
          for (long q = 0; q < nn; ++q) { op.M(j, q) = x; x *= ix; }
        }
        vtx.set_lin_fit(n, lqp::zeroth_order_row(op));
        vtx.set_lin_z(false);
      }
      solvers::vertex_sc::config sc;
      sc.mode = "hybrid";
      sc.mesh = {2, 2, 2};   // the identity sublattice: runs as "direct"
      sc.col = "static";
      vtx.set_pol_sc(sc);
      scr_eri.set_vertex(&vtx);
      sc_run r;
      vertex_debug::set(dbg);
      const auto t0 = std::chrono::steady_clock::now();
      std::tie(r.e_hf, r.e_corr) = scf_loop(mb_state, dyson, eri, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                            2, false, 1e-9, true);
      mpi_context->comm.barrier();
      r.wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      vertex_debug::set("");
      REQUIRE(scr_eri.pol_vertex_instance() != nullptr);
      r.cached_w = scr_eri.pol_vertex_instance()->has_cached_w();
      r.G = nda::array<cplx, 5>(mb_state.sG_tskij.value().local());
      r.Sigma = nda::array<cplx, 5>(mb_state.sSigma_tskij.value().local());
      {
        h5::file f(out + ".pol_sc.h5", 'r');
        h5::group g(f);
        nda::h5_read(g, "Pi", r.Pi);
      }
      mpi_context->comm.barrier();
      if (mpi_context->comm.root())
        for (auto const &e : std::filesystem::directory_iterator("."))
          if (e.path().filename().string().rfind(out + ".", 0) == 0) std::filesystem::remove(e.path());
      mpi_context->comm.barrier();
      return r;
    };

    auto rd = run("dynamic");
    auto rs = run("static");
    auto [dP, sP] = max_diff(rs.Pi, rd.Pi);
    auto [dG, sG] = max_diff(rs.G, rd.G);
    auto [dS, sS] = max_diff(rs.Sigma, rd.Sigma);
    app_log(1, "vertex_sc_static_rung: e_hf {:.12f} / {:.12f}, e_corr {:.12f} / {:.12f} (dynamic rung, col static / static rung); "
               "max |Pi_s - Pi_d| = {:.3e} (|Pi| {:.3e}), max |G_s - G_d| = {:.3e} (|G| {:.3e}), max |Sigma_s - Sigma_d| = {:.3e} "
               "(|Sigma| {:.3e}); wall {:.1f} s (dynamic rung) vs {:.1f} s (static rung); W-bar cache filled: {} / {}",
            rd.e_hf, rs.e_hf, rd.e_corr, rs.e_corr, dP, sP, dG, sG, dS, sS, rd.wall, rs.wall, rd.cached_w, rs.cached_w);
    // (b) the static rung never fills the W-bar cache (the dynamic-rung chain does: the check is not vacuous)
    REQUIRE(rd.cached_w);
    REQUIRE(not rs.cached_w);
    // (a) the same static column: identical arithmetic per unit, the unit reduction order aside
    const double tol = 1e-10;
    REQUIRE(sP > 1e-10);
    REQUIRE(dP < tol * sP);
    REQUIRE(dG < tol * sG);
    REQUIRE(dS < tol * sS);
    REQUIRE(std::abs(rs.e_hf - rd.e_hf) < tol * std::abs(rd.e_hf));
    REQUIRE(std::abs(rs.e_corr - rd.e_corr) < tol * std::abs(rd.e_corr));

    // the linearized rung: its static limit (the stored polarization is the Hermitian part of the static one's), then the real W-bar
    auto rl0 = run("linearized", "_limit", "lin_static_limit=1");
    auto rl = run("linearized");
    nda::array<cplx, 4> Ph(rs.Pi);
    for (long jn = 0; jn < Ph.shape(0); ++jn)
      for (long iq = 0; iq < Ph.shape(1); ++iq)
        for (long M = 0; M < Ph.shape(2); ++M)
          for (long N = 0; N < Ph.shape(3); ++N) Ph(jn, iq, M, N) = 0.5 * (rs.Pi(jn, iq, M, N) + std::conj(rs.Pi(jn, iq, N, M)));
    auto [dL, sL] = max_diff(rl0.Pi, Ph);
    auto [dR, sR] = max_diff(rl.Pi, Ph);
    app_log(1, "vertex_sc_lin_rung: static limit vs the static rung: max |Pi| diff {:.3e} (|Pi| {:.3e}), e_corr {:.12f} / {:.12f}; "
               "linearized vs static: max |Pi| diff {:.3e}, e_corr {:.12f}; wall {:.1f} s (static) / {:.1f} s (linearized, limit) / "
               "{:.1f} s (linearized)", dL, sL, rl0.e_corr, rs.e_corr, dR, rl.e_corr, rs.wall, rl0.wall, rl.wall);
    REQUIRE(dL < 1e-10 * sL);
    REQUIRE(std::abs(rl0.e_corr - rs.e_corr) < tol * std::abs(rs.e_corr));
    REQUIRE(dR > 1e-6 * sL);
    REQUIRE(rl.cached_w);
#endif
  }

} // bdft_tests
