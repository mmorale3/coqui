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

// scGW-tilde increments C1+ (notes/scgwt_implementation_plan.md): the CVV R-space
// engine and covariant velocity.
//
//   C1-a  toy tight-binding oracle: for H(k) = sum_R t(R) e^{ik.R} with t supported
//         STRICTLY INSIDE the WS supercell, the k->R->velocity pipeline reproduces the
//         analytic  d_k H = sum_R iR t(R) e^{ik.R}  to FP accuracy (the WS interpolant
//         is exact when no boundary aliasing occurs), and the value row reproduces
//         H(k) on the mesh (round-trip pin).
//   C1-b  hermiticity: hermitian-paired t (t(-R) = t(R)^dag) gives v_a(k)^dag = v_a(k)
//         exactly; time reversal: REAL hoppings give v_a(-k) = -v_a(k)^T.
//   C1-c  finite-difference-in-k cross-check of the derivative rows against the value
//         rows of the SAME WS interpolant, at mesh and generic k.
//   C1-s  class-level smoke on LiH-222: build cvv_head_t from a real 2-iteration scGW
//         state (H0 + F + Sigma(tau)), log the R-decay/truncation, evaluate velocities;
//         the Sigma = 0 control (KS/HF) velocity is exactly hermitian.

#include <cmath>
#include <random>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"

#include "numerics/imag_axes_ft/IAFT.hpp"
// pulls the same known-good include closure as the sibling vertex tests
#include "methods/vertex/vertex_pi.icc"

#include "mean_field/default_MF.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include "methods/scr_coulomb/cvv_head.hpp"

namespace bdft_tests {

  using namespace methods;

  namespace {

    // rows of the reciprocal lattice: b_i . a_j = 2 pi delta_ij
    inline nda::stack_array<double, 3, 3> recip_lattv(nda::stack_array<double, 3, 3> const &a) {
      nda::stack_array<double, 3, 3> b;
      const double det = a(0,0)*(a(1,1)*a(2,2)-a(1,2)*a(2,1))
                       - a(0,1)*(a(1,0)*a(2,2)-a(1,2)*a(2,0))
                       + a(0,2)*(a(1,0)*a(2,1)-a(1,1)*a(2,0));
      const double f = 2.0 * M_PI / det;
      // b rows = f * (a_j x a_k) for cyclic (i, j, k)
      for (int i = 0; i < 3; ++i) {
        const int j = (i + 1) % 3, k = (i + 2) % 3;
        b(i, 0) = f * (a(j,1)*a(k,2) - a(j,2)*a(k,1));
        b(i, 1) = f * (a(j,2)*a(k,0) - a(j,0)*a(k,2));
        b(i, 2) = f * (a(j,0)*a(k,1) - a(j,1)*a(k,0));
      }
      return b;
    }

    struct toy_tb {
      // hoppings t(R) on lattice-unit R indices, hermitian-paired: t(-R) = t(R)^dag
      std::vector<std::array<long, 3>> Rs;
      std::vector<nda::array<ComplexType, 2>> ts;
      nda::stack_array<double, 3, 3> lattv;
      long nb;

      nda::array<double, 1> rcart(std::array<long, 3> const &R) const {
        nda::array<double, 1> rc(3);
        for (int a = 0; a < 3; ++a)
          rc(a) = R[0]*lattv(0,a) + R[1]*lattv(1,a) + R[2]*lattv(2,a);
        return rc;
      }
      nda::array<ComplexType, 2> H(nda::MemoryArrayOfRank<1> auto const &kvec) const {
        nda::array<ComplexType, 2> h(nb, nb); h() = ComplexType(0.0);
        for (size_t m = 0; m < Rs.size(); ++m) {
          auto rc = rcart(Rs[m]);
          const double kR = kvec(0)*rc(0) + kvec(1)*rc(1) + kvec(2)*rc(2);
          h += std::exp(ComplexType(0.0, kR)) * ts[m];
        }
        return h;
      }
      // analytic velocity d_k_a H
      nda::array<ComplexType, 3> V(nda::MemoryArrayOfRank<1> auto const &kvec) const {
        nda::array<ComplexType, 3> v(3, nb, nb); v() = ComplexType(0.0);
        for (size_t m = 0; m < Rs.size(); ++m) {
          auto rc = rcart(Rs[m]);
          const double kR = kvec(0)*rc(0) + kvec(1)*rc(1) + kvec(2)*rc(2);
          const ComplexType ph = std::exp(ComplexType(0.0, kR));
          for (int a = 0; a < 3; ++a)
            v(a, nda::range::all, nda::range::all) += ComplexType(0.0, rc(a)) * ph * ts[m];
        }
        return v;
      }
    };

    // hermitian-paired toy on R in {0, +-e1, +-e2, +-e3, +-(1,1,0)}; real_t restricts
    // to real symmetric hoppings (the time-reversal-symmetric case)
    toy_tb make_toy(long nb, bool real_t, unsigned seed) {
      toy_tb toy;
      toy.nb = nb;
      toy.lattv = nda::stack_array<double, 3, 3>{{5.0, 0.3, 0.0},
                                                 {0.0, 5.2, 0.1},
                                                 {0.2, 0.0, 4.8}};
      std::mt19937 gen(seed);
      std::uniform_real_distribution<double> dis(-0.5, 0.5);
      auto rnd_mat = [&](bool herm) {
        nda::array<ComplexType, 2> t(nb, nb);
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j)
            t(i, j) = real_t ? ComplexType(dis(gen), 0.0) : ComplexType(dis(gen), dis(gen));
        if (herm) {  // hermitize (real case: symmetrize)
          nda::array<ComplexType, 2> th(nb, nb);
          for (long i = 0; i < nb; ++i)
            for (long j = 0; j < nb; ++j) th(i, j) = 0.5 * (t(i, j) + std::conj(t(j, i)));
          return th;
        }
        return t;
      };
      auto add_pair = [&](std::array<long, 3> R) {
        auto t = rnd_mat(false);
        if (real_t) { // t real; pair rule t(-R) = t(R)^T keeps H(k) hermitian and real-symmetric
          toy.Rs.push_back(R); toy.ts.push_back(t);
          nda::array<ComplexType, 2> tt(toy.nb, toy.nb);
          for (long i = 0; i < toy.nb; ++i)
            for (long j = 0; j < toy.nb; ++j) tt(i, j) = t(j, i);
          toy.Rs.push_back({-R[0], -R[1], -R[2]}); toy.ts.push_back(tt);
        } else {      // t(-R) = t(R)^dag
          toy.Rs.push_back(R); toy.ts.push_back(t);
          nda::array<ComplexType, 2> td(toy.nb, toy.nb);
          for (long i = 0; i < toy.nb; ++i)
            for (long j = 0; j < toy.nb; ++j) td(i, j) = std::conj(t(j, i));
          toy.Rs.push_back({-R[0], -R[1], -R[2]}); toy.ts.push_back(td);
        }
      };
      toy.Rs.push_back({0, 0, 0}); toy.ts.push_back(rnd_mat(true));
      add_pair({1, 0, 0}); add_pair({0, 1, 0}); add_pair({0, 0, 1}); add_pair({1, 1, 0});
      return toy;
    }

    double max_abs_diff(nda::MemoryArray auto const &A, nda::MemoryArray auto const &B) {
      double m = 0.0;
      nda::for_each(A.shape(), [&](auto... i) { m = std::max(m, std::abs(A(i...) - B(i...))); });
      return m;
    }

  } // anonymous

  TEST_CASE("cvv_velocity_toy", "[methods][scgwt][cvv]") {
    decltype(nda::range::all) all;
    const long nb = 3, nmesh = 6;
    const long nk = nmesh * nmesh * nmesh;

    for (bool real_t : {false, true}) {
      auto toy = make_toy(nb, real_t, real_t ? 77u : 13u);
      auto b = recip_lattv(toy.lattv);

      // full-BZ cartesian mesh k = (i/n) b1 + (j/n) b2 + (l/n) b3
      nda::array<double, 2> kpts(nk, 3);
      {
        long ik = 0;
        for (long i = 0; i < nmesh; ++i)
          for (long j = 0; j < nmesh; ++j)
            for (long l = 0; l < nmesh; ++l, ++ik)
              for (int a = 0; a < 3; ++a)
                kpts(ik, a) = (double(i)/nmesh)*b(0,a) + (double(j)/nmesh)*b(1,a) +
                              (double(l)/nmesh)*b(2,a);
      }

      // H(k) on the mesh -> WS R store (the exact pipeline cvv_head_t::build runs)
      nda::array<long, 1> mesh(3); mesh() = nmesh;
      auto [rw, rp] = utils::WS_rgrid(toy.lattv, mesh);
      const long nR = rp.shape(0);
      nda::array<ComplexType, 2> f_Rk(nR, nk);
      utils::k_to_R_coefficients(rp, kpts, toy.lattv, f_Rk);

      nda::array<ComplexType, 2> H_k(nk, nb * nb);
      for (long ik = 0; ik < nk; ++ik) {
        auto h = toy.H(kpts(ik, all));
        H_k(ik, all) = nda::reshape(h, std::array<long, 1>{nb * nb});
      }
      nda::array<ComplexType, 2> h_R(nR, nb * nb);
      nda::blas::gemm(f_Rk, H_k, h_R);

      auto Rcart = solvers::cvv_detail::rcart_from_idx(rp, toy.lattv);

      double err_val = 0.0, err_vel = 0.0, err_herm = 0.0, err_fd = 0.0, err_trev = 0.0;
      // probe: every 7th mesh point + one generic off-mesh k
      std::vector<nda::array<double, 1>> probes;
      for (long ik = 0; ik < nk; ik += 7) {
        nda::array<double, 1> kv(3); kv() = kpts(ik, all); probes.push_back(kv);
      }
      { nda::array<double, 1> kv(3);
        for (int a = 0; a < 3; ++a) kv(a) = 0.137*b(0,a) + 0.291*b(1,a) - 0.054*b(2,a);
        probes.push_back(kv); }

      const double dk = 1e-4;
      for (auto const &kv : probes) {
        auto P = solvers::cvv_detail::phase_rows(Rcart, rw, kv, true);   // (4, nR)
        nda::array<ComplexType, 2> out(4, nb * nb);
        nda::blas::gemm(P, h_R, out);

        // value row == H(k) (exact for interior-supported t; also at generic k)
        auto h_ref = toy.H(kv);
        auto h_num = nda::reshape(out(3, all), std::array<long, 2>{nb, nb});
        err_val = std::max(err_val, max_abs_diff(h_num, h_ref));

        // C1-a: derivative rows == analytic velocity
        auto v_ref = toy.V(kv);
        for (int a = 0; a < 3; ++a) {
          auto va = nda::reshape(out(a, all), std::array<long, 2>{nb, nb});
          err_vel = std::max(err_vel, max_abs_diff(va, v_ref(a, all, all)));
          // C1-b hermiticity: v_a(k)^dag = v_a(k)
          for (long i = 0; i < nb; ++i)
            for (long j = 0; j < nb; ++j)
              err_herm = std::max(err_herm, std::abs(va(i, j) - std::conj(va(j, i))));
        }

        // C1-b time reversal (real hoppings): v_a(-k) = -v_a(k)^T
        if (real_t) {
          nda::array<double, 1> km(3); for (int a = 0; a < 3; ++a) km(a) = -kv(a);
          auto Pm = solvers::cvv_detail::phase_rows(Rcart, rw, km, false);
          nda::array<ComplexType, 2> outm(3, nb * nb);
          nda::blas::gemm(Pm, h_R, outm);
          for (int a = 0; a < 3; ++a) {
            auto vm = nda::reshape(outm(a, all), std::array<long, 2>{nb, nb});
            auto vp = nda::reshape(out(a, all), std::array<long, 2>{nb, nb});
            for (long i = 0; i < nb; ++i)
              for (long j = 0; j < nb; ++j)
                err_trev = std::max(err_trev, std::abs(vm(i, j) + vp(j, i)));
          }
        }

        // C1-c: central finite difference of the value rows vs the derivative rows
        for (int a = 0; a < 3; ++a) {
          nda::array<double, 1> kp_(3), km_(3);
          kp_() = kv; km_() = kv; kp_(a) += dk; km_(a) -= dk;
          auto Pp = solvers::cvv_detail::phase_rows(Rcart, rw, kp_, true);
          auto Pm = solvers::cvv_detail::phase_rows(Rcart, rw, km_, true);
          nda::array<ComplexType, 2> op(4, nb * nb), om(4, nb * nb);
          nda::blas::gemm(Pp, h_R, op);
          nda::blas::gemm(Pm, h_R, om);
          for (long f = 0; f < nb * nb; ++f) {
            const ComplexType fd = (op(3, f) - om(3, f)) / (2.0 * dk);
            err_fd = std::max(err_fd, std::abs(fd - out(a, f)));
          }
        }
      }

      app_log(1, "cvv_velocity_toy[real_t={}]: err_val = {:.3e}, err_vel = {:.3e}, "
                 "err_herm = {:.3e}, err_trev = {:.3e}, err_fd = {:.3e}",
              real_t, err_val, err_vel, err_herm, err_trev, err_fd);
      REQUIRE(err_val < 1e-11);    // round-trip pin: interpolant == H on and off mesh
      REQUIRE(err_vel < 1e-10);    // C1-a: exact analytic derivative
      REQUIRE(err_herm < 1e-11);   // C1-b: hermiticity
      if (real_t) REQUIRE(err_trev < 1e-11);   // C1-b: time reversal
      REQUIRE(err_fd < 1e-5);      // C1-c: O(dk^2) finite-difference agreement
    }
  }

  TEST_CASE("cvv_build_lih222", "[methods][scgwt][cvv]") {
#ifndef ENABLE_DLR
    SUCCEED("cvv_build_lih222 skipped: build has ENABLE_DLR=OFF.");
#else
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_scr_cvv";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    solvers::hf_t hf;
    solvers::gw_t gw(&ft, "ignore_g0", output);
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
    simple_dyson dyson(mf.get(), &ft);
    MBState mb_state(mpi_context, ft, output);
    iter_scf::iter_scf_t iter_sol("damping");
    auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                   solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                   2, false, 1e-9, true);
    app_log(1, "cvv_build_lih222: scGW state e_hf = {}, e_corr = {}", e_hf, e_corr);
    REQUIRE(mb_state.sF_skij.has_value());
    REQUIRE(mb_state.sSigma_tskij.has_value());

    // C1-s: build from the real scGW state; log R decay + truncation; velocities run
    solvers::cvv_head_t cvv(&ft, 1e-6);
    cvv.build(*mf, dyson.H0(), mb_state.sF_skij.value().local(),
              mb_state.sSigma_tskij.value().local());
    REQUIRE(cvv.built());
    REQUIRE(cvv.nR() > 0);
    REQUIRE(cvv.nR_kept() > 0);
    REQUIRE(cvv.nR_kept() <= cvv.nR());
    auto v = cvv.velocity(0, 0);
    REQUIRE(v.shape(0) == 3);
    REQUIRE(v.shape(1) == cvv.nw());
    // STRUCTURAL PIN (measured 2026-08-11, first C1 run): on a Gamma-centered 2x2x2
    // mesh EVERY mesh k is a time-reversal-invariant momentum -- all k.R phases are
    // 0/pi (real), so the WS interpolant is a cosine sum and its derivative VANISHES
    // IDENTICALLY at mesh points (the +-R images at the WS boundary cancel pairwise).
    // v(mesh k) == 0 is therefore exact at 2^3, for ANY stored h. Meshes >= 3 per
    // direction carry nonzero mesh-point velocities (the 6^3 toy above proves the
    // machinery). Consequence: 2^3 fixtures cannot discriminate the CVV head (C2-c
    // note); the head needs denser meshes.
    double vmax = 0.0;
    nda::for_each(v.shape(), [&](auto... i) { vmax = std::max(vmax, std::abs(v(i...))); });
    app_log(1, "cvv_build_lih222: nR = {}, nR_kept = {}, max|v| (2^3 TRIM zero) = {:.3e}",
            cvv.nR(), cvv.nR_kept(), vmax);
    REQUIRE(std::isfinite(vmax));
    REQUIRE(vmax < 1e-10);   // the structural 2^3 zero, pinned

    // Sigma = 0 control (KS/HF): the velocity is iw-independent and exactly hermitian
    solvers::cvv_head_t cvv0(&ft, 1e-6);
    nda::array<ComplexType, 5> sig_empty(0, 0, 0, 0, 0);
    cvv0.build(*mf, dyson.H0(), mb_state.sF_skij.value().local(), sig_empty);
    auto v0 = cvv0.velocity(0, 1);
    double err_herm = 0.0, err_wdep = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (long i = 0; i < v0.shape(2); ++i)
        for (long j = 0; j < v0.shape(3); ++j) {
          err_herm = std::max(err_herm, std::abs(v0(a, 0, i, j) - std::conj(v0(a, 0, j, i))));
          err_wdep = std::max(err_wdep, std::abs(v0(a, v0.shape(1) - 1, i, j) - v0(a, 0, i, j)));
        }
    }
    app_log(1, "cvv_build_lih222: Sigma=0 control err_herm = {:.3e}, err_wdep = {:.3e}",
            err_herm, err_wdep);
    REQUIRE(err_herm < 1e-10);
    REQUIRE(err_wdep == 0.0);

    mpi_context->comm.barrier();
    if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
    mpi_context->comm.barrier();
#endif
  }

} // bdft_tests
