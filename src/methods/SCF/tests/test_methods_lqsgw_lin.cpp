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

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

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
#include "methods/vertex/vertex_debug.hpp"

/**
 * The linearized vertex rung (pol_vertex_rung = "linearized") in the LQSGW loop, end to end: LiH 2x2x2 (the lqsgw_lih222
 * setup), C = [1, 3), the vertex self-consistency on the identity sublattice (pol_vertex_sc = "direct"), the first refresh
 * after the first quasiparticle solve (pol_vertex_sc_start = 2: the Z^1/2 of that solve exists), 3 iterations:
 *   - the static limit: the linearized rung with its dynamic remainder set to zero (vertex_debug lin_static_limit) and
 *     pol_vertex_lin_z = false reproduces the static-rung run -- the stored vertex polarization and the energy;
 *   - the linearized rung with the real W-bar, without and with Z^1/2: the Ward meter -- the head u^dag P u of the stored
 *     polarization at q = 0 relative to the smallest finite |q| -- stays at the static rung's level;
 *   - on a device build, the device-resident linearized unit against the host path (vertex_debug dynbse_dev_unit = 0).
 */

namespace bdft_tests {

  using namespace methods;

  namespace lqsgw_lin_detail {

    struct lin_row {
      double e_tot = 0.0, wall = 0.0;
      long final_iter = -1;
      nda::array<ComplexType, 4> Pi;          // the stored vertex polarization (nw_h, nq, Nm, Nm)
      double head_gamma = 0.0, head_q1 = 0.0; // |u^dag P(i nu_0) u| at Gamma and at the smallest finite |q|
    };

    inline lin_row run(auto &mpi_context, std::shared_ptr<mf::MF> &mf, imag_axes_ft::IAFT &ft, std::string const &tag,
                       std::string const &rung, bool lin_z, std::string const &dbg) {
      const std::string output = "lqsgw_lin_" + tag;
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*8, "", "incore", "", output, 1e-8, mf->ecutrho(), 1, 512));
      auto eri = mb_eri_t(thc, thc);
      qp_context_t qp_params{.qp_approx = "lqp", .qp_scf_mode = "qpscf",
                             .lqp = {.n_fit = 4, .fit_order = -1, .symmetric_window = true, .fit_resid_tol = 1e-8}};
      solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd(), "ignore_g0");
      vtx.set_pol_vertex("ladder", "w0_prev", nda::range(1, 3), -1, 1e-8, -1.0, -1.0, -1.0, "ladder_n2");
      vtx.set_ladder_rung(rung, 1e-8, 30, 12, -1.0);
      vtx.set_lin_z(lin_z);
      solvers::vertex_sc::config sc;
      sc.mode = "direct";
      sc.mesh = {2, 2, 2};
      sc.col = "static";
      sc.start = 2;
      vtx.set_pol_sc(sc);
      scr_eri.set_vertex(&vtx);
      vertex_debug::set(dbg);
      iter_scf::iter_scf_t iter_sol(iter_scf::damp_t(0.7));
      MBState mb_state(mpi_context, ft, output);
      lin_row row;
      const auto t0 = std::chrono::steady_clock::now();
      row.e_tot = qp_scf_loop(mb_state, eri, ft, qp_params, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 3, false, 1e-6);
      mpi_context->comm.barrier();
      row.wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      vertex_debug::set("");
      {
        h5::file f(output + ".pol_sc.h5", 'r');
        h5::group g(f);
        nda::h5_read(g, "Pi", row.Pi);
      }
      {
        h5::file file(output + ".mbpt.h5", 'r');
        h5::h5_read(h5::group(file).open_group("scf"), "final_iter", row.final_iter);
      }
      // the Ward meter: u(q) = t(q) chi(q) (the plane-wave head in the secondary frame), h = u^dag P(q, i nu_0) u
      auto *pv = scr_eri.pol_vertex_instance();
      REQUIRE(pv != nullptr);
      auto const &tmap = pv->secondary_transfer();      // (nq, Nm, Np)
      auto chi = thc.basis_head();                       // (nq, Np)
      const long Nm = row.Pi.shape(2), Np = tmap.shape(2);
      double q2min = 1e300;
      long iq1 = -1;
      for (long iq = 0; iq < row.Pi.shape(1); ++iq) {
        auto qv = mf->Qpts_ibz(iq);
        const double q2 = qv(0) * qv(0) + qv(1) * qv(1) + qv(2) * qv(2);
        if (q2 > 1e-12 and q2 < q2min) { q2min = q2; iq1 = iq; }
      }
      REQUIRE(iq1 >= 0);
      auto head = [&](long iq) {
        nda::array<ComplexType, 1> u(Nm);
        for (long m = 0; m < Nm; ++m) {
          ComplexType s = 0.0;
          for (long P = 0; P < Np; ++P) s += tmap(iq, m, P) * chi(iq, P);
          u(m) = s;
        }
        ComplexType h = 0.0;
        for (long M = 0; M < Nm; ++M)
          for (long N = 0; N < Nm; ++N) h += std::conj(u(M)) * row.Pi(0, iq, M, N) * u(N);
        return std::abs(h);
      };
      row.head_gamma = head(0);
      row.head_q1 = head(iq1);
      mpi_context->comm.barrier();
      if (mpi_context->comm.root())
        for (auto const &e : std::filesystem::directory_iterator("."))
          if (e.path().filename().string().rfind(output + ".", 0) == 0) std::filesystem::remove(e.path());
      mpi_context->comm.barrier();
      std::printf("lqsgw_lin [%s]: final iter %ld, e_tot = %.12f, |h(Gamma)| = %.3e, |h(q1)| = %.3e, wall %.1f s\n", tag.c_str(),
                  row.final_iter, row.e_tot, row.head_gamma, row.head_q1, row.wall);
      return row;
    }

    inline std::pair<double, double> max_diff(nda::array<ComplexType, 4> const &A, nda::array<ComplexType, 4> const &B) {
      REQUIRE(A.shape() == B.shape());
      double d = 0.0, s = 0.0;
      for (long i = 0; i < A.size(); ++i) {
        d = std::max(d, std::abs(A.data()[i] - B.data()[i]));
        s = std::max(s, std::abs(B.data()[i]));
      }
      return {d, s};
    }

  } // lqsgw_lin_detail

  TEST_CASE("lqsgw_lin_lih222", "[methods][lqsgw][vertex][lin]") {
#ifndef ENABLE_DLR
    SUCCEED("lqsgw_lin_lih222 skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace lqsgw_lin_detail;
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(100.0, 7.0, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));

    auto rs = run(mpi_context, mf, ft, "static", "static", false, "");
    auto r0 = run(mpi_context, mf, ft, "lin_limit", "linearized", false, "lin_static_limit=1");
    auto r1 = run(mpi_context, mf, ft, "lin_noz", "linearized", false, "");
    auto r2 = run(mpi_context, mf, ft, "lin_z", "linearized", true, "");
    for (auto const *r : {&rs, &r0, &r1, &r2}) REQUIRE(r->final_iter == 3);

    // (a) the static limit; the linearized rung stores the Hermitian part of its polarization (the static rung's is Hermitian
    // only to the solve's rounding, ~4e-8 here)
    nda::array<ComplexType, 4> Ps(rs.Pi);
    for (long jn = 0; jn < Ps.shape(0); ++jn)
      for (long iq = 0; iq < Ps.shape(1); ++iq)
        for (long M = 0; M < Ps.shape(2); ++M)
          for (long N = 0; N < Ps.shape(3); ++N) Ps(jn, iq, M, N) = 0.5 * (rs.Pi(jn, iq, M, N) + std::conj(rs.Pi(jn, iq, N, M)));
    auto [dP, sP] = max_diff(r0.Pi, Ps);
    std::printf("lqsgw_lin: static limit vs the static rung: max |Pi| diff %.3e (|Pi| %.3e), e_tot diff %.3e\n", dP, sP,
                std::abs(r0.e_tot - rs.e_tot));
    REQUIRE(sP > 1e-8);
    REQUIRE(dP < 1e-10 * sP);
    REQUIRE(std::abs(r0.e_tot - rs.e_tot) < 1e-10);
    // the linearized rung moves the polarization and the energy away from the static one; Z^1/2 moves them again
    REQUIRE(max_diff(r1.Pi, rs.Pi).first > 1e-6 * sP);
    REQUIRE(max_diff(r2.Pi, r1.Pi).first > 1e-6 * sP);
    // (b) the Ward meter: the window block of the quasiparticle G is not a resolvent, so the q = 0 head of a ladder correction
    // does not vanish exactly (it is the bubble's leakage u^dag Cb != 0 times the vertex); it stays small next to the head at
    // the smallest finite |q| (measured: static 2.4e-5, linearized 4.8e-4 / 4.5e-4 of it; the exact statement on a
    // quasiparticle resolvent is vertex_lin_toy (b))
    std::printf("lqsgw_lin: Ward meter h(Gamma) / h(q1): static %.3e, linearized %.3e, linearized + Z^1/2 %.3e\n",
                rs.head_gamma / rs.head_q1, r1.head_gamma / r1.head_q1, r2.head_gamma / r2.head_q1);
    for (auto const *r : {&rs, &r1, &r2}) REQUIRE(r->head_gamma < 2e-3 * r->head_q1);
#if defined(ENABLE_CUDA)
    // (c) the device-resident linearized unit against the host path
    auto rh = run(mpi_context, mf, ft, "lin_z_host", "linearized", true, "dynbse_dev_unit=0");
    auto [dH, sH] = max_diff(r2.Pi, rh.Pi);
    std::printf("lqsgw_lin: device vs host linearized unit: max |Pi| diff %.3e (|Pi| %.3e), e_tot diff %.3e\n", dH, sH,
                std::abs(r2.e_tot - rh.e_tot));
    REQUIRE(dH < 1e-9 * sH);
    REQUIRE(std::abs(r2.e_tot - rh.e_tot) < 1e-9);
#endif
#endif
  }

} // bdft_tests
