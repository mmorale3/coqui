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

/*
 * Static-route selection plan, phase 3 (notes/static_route_selection_plan.md):
 * in-SCF I7 acceptance — the direct ("hamilt") route in the static ERI slots
 * must reproduce the factorized-ERI route through the UNTOUCHED scf_loop seam.
 * The route difference is bounded by the THC fitting error; the wiring errors
 * this battery pins (double normalization, ns_scl/N_k, slot composition, sign
 * of K, div correction) are all gross by comparison.
 *
 * Guard behavior (plan 3.5) that ends in utils::check -> APP_ABORT -> MPI_Abort
 * (exx-options conflict, hamilt in the dynamic slot) is NOT testable
 * in-process — consistent with the codebase-wide absence of THROWS-style
 * tests. The positive complements are covered here instead; the dynamic-slot
 * rejection lives in main.cpp (utils::check on eri_type=="hamilt") and is
 * additionally enforced at compile time: no scf_loop instantiation has
 * hamilt_eval_t in corr.
 *
 * Phase 4 (symmetry in the nij path): the interim sym-mesh exchange guard is
 * RETIRED — the route-equivalence battery runs on the symmetric mesh too, and
 * hamilt_hf_sym_vs_nosym pins the View-2 general-nij lift (becsum + v_x,
 * notes/static_route_nij_symmetry_note.md) against the nosym fixture.
 */

namespace bdft_tests {

  using namespace methods;
  using mpi_context_t = utils::mpi_context_t<mpi3::communicator,mpi3::shared_communicator>;
  using array_view_4d_t = nda::array_view<ComplexType, 4>;
  using math::shm::make_shared_array;
  using math::shm::shared_array;

  namespace {

    // max_i |a_i - b_i| over two same-shape contiguous rank-4 arrays
    inline double max_abs_diff(nda::array<ComplexType,4> const& A,
                               nda::array<ComplexType,4> const& B) {
      REQUIRE(A.shape() == B.shape());
      double m = 0.0;
      auto const* pa = A.data();
      auto const* pb = B.data();
      for (long i = 0; i < A.size(); ++i)
        m = std::max(m, std::abs(pa[i] - pb[i]));
      return m;
    }

    struct hf_scf_result {
      double e_hf;
      nda::array<ComplexType,4> F_first;   // scf/iter<first>/F_skij
      nda::array<ComplexType,4> F_final;   // scf/iter<final_iter>/F_skij
    };

    // Fixed-iteration Dyson HF-SCF through the production scf_loop seam
    // (scf_driver.cpp): conv_tol below reach so BOTH routes run exactly
    // `niter` iterations and the per-iteration trajectories are comparable.
    template<typename ERI_t>
    hf_scf_result run_hf_scf(std::shared_ptr<mpi_context_t>& mpi_context,
                             std::shared_ptr<mf::MF>& mf, ERI_t& eri,
                             solvers::hf_t& hf, imag_axes_ft::IAFT& ft,
                             std::string prefix, int niter) {
      simple_dyson dyson(mf.get(), &ft);
      iter_scf::iter_scf_t iter_sol("damping");
      MBState mb_state(mpi_context, ft, prefix);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf), &iter_sol,
                                     niter, false, 1e-13, false);
      (void)e_corr;
      hf_scf_result res;
      res.e_hf = e_hf;
      {
        h5::file file(prefix + ".mbpt.h5", 'r');
        auto scf_grp = h5::group(file).open_group("scf");
        int it;
        h5::h5_read(scf_grp, "final_iter", it);
        auto g1 = scf_grp.open_group("iter1");
        nda::h5_read(g1, "F_skij", res.F_first);
        auto gf = scf_grp.open_group("iter" + std::to_string(it));
        nda::h5_read(gf, "F_skij", res.F_final);
      }
      mpi_context->comm.barrier();
      if (mpi_context->comm.root())
        remove((prefix + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return res;
    }

  } // anonymous namespace

  TEST_CASE("hamilt_hf_route_equivalence", "[methods][hamilt][hf][qe][paw]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_source);
    int const niter = 3;

    // ERI-static vs hamilt-static with matched div treatment (ignore_g0,
    // plan 3.1); tolerance = THC fitting scale for these fixtures/ranks.
    auto check_routes = [&](std::string mf_id, double tol) {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, mf_id));
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*20, "", "incore", "", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      hamilt_eval_t heval(mf, ptree{});
      solvers::hf_t hf(methods::ignore_g0);

      auto eri_ref = mb_eri_t(thc, thc);
      auto ref = run_hf_scf(mpi_context, mf, eri_ref, hf, ft, "cq_thcref", niter);

      auto eri_ham = mb_eri_t(heval, thc);
      auto ham = run_hf_scf(mpi_context, mf, eri_ham, hf, ft, "cq_hamhf", niter);

      app_log(1, "[{}] e_hf thc = {:.10f}  hamilt = {:.10f}  |diff| = {:.3e}",
              mf_id, ref.e_hf, ham.e_hf, std::abs(ref.e_hf - ham.e_hf));
      app_log(1, "[{}] max|dF| iter1 = {:.3e}  final = {:.3e}", mf_id,
              max_abs_diff(ref.F_first, ham.F_first),
              max_abs_diff(ref.F_final, ham.F_final));
      VALUE_EQUAL(ham.e_hf, ref.e_hf, tol);
      ARRAY_EQUAL(ham.F_first, ref.F_first, tol);   // first iteration ...
      ARRAY_EQUAL(ham.F_final, ref.F_final, tol);   // ... and end of the run

      // Per-term mixing (plan 3.4) falls out of the slot structure: hartree
      // direct + exchange THC, and the inverse, against the all-THC reference.
      auto eri_mixJ = mb_eri_t(heval, thc, thc);  // hartree = hamilt
      auto mixJ = run_hf_scf(mpi_context, mf, eri_mixJ, hf, ft, "cq_mixJ", niter);
      auto eri_mixK = mb_eri_t(thc, heval, thc);  // exchange = hamilt
      auto mixK = run_hf_scf(mpi_context, mf, eri_mixK, hf, ft, "cq_mixK", niter);
      app_log(1, "[{}] mixing: |de| J-direct = {:.3e}  K-direct = {:.3e}", mf_id,
              std::abs(mixJ.e_hf - ref.e_hf), std::abs(mixK.e_hf - ref.e_hf));
      VALUE_EQUAL(mixJ.e_hf, ref.e_hf, tol);
      VALUE_EQUAL(mixK.e_hf, ref.e_hf, tol);
      ARRAY_EQUAL(mixJ.F_final, ref.F_final, tol);
      ARRAY_EQUAL(mixK.F_final, ref.F_final, tol);
    };

    SECTION("lih222_paw")  { check_routes("qe_lih222_paw",  1e-4); }
    SECTION("lih222_uspp") { check_routes("qe_lih222_uspp", 1e-4); }
    SECTION("lih222_ncpp") { check_routes("qe_lih222",      1e-4); }
    // Symmetry-reduced mesh (plan 4.3b): same battery; iterations >= 2 have a
    // non-diagonal Dm, exercising the View-2 general-nij lift in BOTH the
    // direct Hartree (becsum) and the direct exchange.
    SECTION("lih222_paw_sym") { check_routes("qe_lih222_paw_sym", 1e-4); }
  }

  TEST_CASE("hamilt_hf_sym_vs_nosym", "[methods][hamilt][hf][qe][paw]") {
    // Plan 4.3a: sym-vs-nosym invariance of the PURELY direct static route
    // (hf slot = hamilt; the thc corr slot is typed but never evaluated by the
    // HF-only solver). Same physical system, symmetry on/off — e_hf must
    // agree to the cross-fixture consistency of the two QE runs (1e-5, the
    // same scale the THC tests use for shared sym/nosym references).
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_source);
    int const niter = 3;

    auto run_direct = [&](std::string mf_id, std::string prefix) {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, mf_id));
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*20, "", "incore", "", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      hamilt_eval_t heval(mf, ptree{});
      solvers::hf_t hf(methods::ignore_g0);
      auto eri = mb_eri_t(heval, thc);
      return run_hf_scf(mpi_context, mf, eri, hf, ft, prefix, niter).e_hf;
    };

    double e_nosym = run_direct("qe_lih222_paw",     "cq_dir_nosym");
    double e_sym   = run_direct("qe_lih222_paw_sym", "cq_dir_sym");
    app_log(1, "[sym-vs-nosym direct] e_hf nosym = {:.10f}  sym = {:.10f}  |diff| = {:.3e}",
            e_nosym, e_sym, std::abs(e_nosym - e_sym));
    VALUE_EQUAL(e_sym, e_nosym, 1e-5);
  }

  TEST_CASE("hamilt_hf_gygi_parity", "[methods][hamilt][hf][qe][paw]") {
    // Plan 3.6 (closes 0.1): the gygi finite-size treatment is applied at the
    // operator level by the shared, route-free HF_K_correction
    // (notes/static_route_gygi_note.md) — direct-route gygi must match
    // ERI-route gygi to the same THC tolerance as the ignore_g0 battery.
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_source);

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_paw"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*20, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    hamilt_eval_t heval(mf, ptree{});
    solvers::hf_t hf_gygi;  // default div treatment = gygi

    auto eri_ref = mb_eri_t(thc, thc);
    auto ref = run_hf_scf(mpi_context, mf, eri_ref, hf_gygi, ft, "cq_gygi_thc", 1);
    auto eri_ham = mb_eri_t(heval, thc);
    auto ham = run_hf_scf(mpi_context, mf, eri_ham, hf_gygi, ft, "cq_gygi_ham", 1);

    app_log(1, "[gygi] e_hf thc = {:.10f}  hamilt = {:.10f}  |diff| = {:.3e}",
            ref.e_hf, ham.e_hf, std::abs(ref.e_hf - ham.e_hf));
    app_log(1, "[gygi] max|dF| iter1 = {:.3e}", max_abs_diff(ref.F_first, ham.F_first));
    VALUE_EQUAL(ham.e_hf, ref.e_hf, 1e-4);
    ARRAY_EQUAL(ham.F_first, ref.F_first, 1e-4);
  }

  TEST_CASE("hamilt_hf_sym_hartree", "[methods][hamilt][hf][qe][paw]") {
    // Component-level route equivalence on a symmetry-reduced mesh with a
    // diagonal (occupation) density matrix: direct HARTREE (full-BZ becsum
    // lift) and — since the phase-4 View-2 nij lift — direct EXCHANGE must
    // both match their THC counterparts.
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_paw_sym"));
    REQUIRE(mf->nkpts_ibz() < mf->nkpts());

    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*20, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    hamilt_eval_t heval(mf, ptree{});
    solvers::hf_t hf(methods::ignore_g0);

    long ns = mf->nspin(), nk = mf->nkpts_ibz(), nb = mf->nbnd();
    auto sS_skij = make_shared_array<array_view_4d_t>(*mpi_context, {ns, nk, nb, nb});
    hamilt::set_ovlp(*mf, sS_skij);

    nda::array<ComplexType,4> occ4d(ns, nk, nb, nb);
    occ4d() = ComplexType(0.0);
    auto mfocc = mf->occ();
    for (int s = 0; s < ns; s++)
      for (int k = 0; k < nk; k++)
        for (int a = 0; a < nb; a++)
          occ4d(s, k, a, a) = mfocc(s, k, a);

    auto sJ_thc = make_shared_array<array_view_4d_t>(*mpi_context, {ns, nk, nb, nb});
    auto sJ_ham = make_shared_array<array_view_4d_t>(*mpi_context, {ns, nk, nb, nb});
    hf.evaluate(sJ_thc, occ4d, thc, sS_skij.local(), true, false);
    hf.evaluate(sJ_ham, occ4d, heval, sS_skij.local(), true, false);

    nda::array<ComplexType,4> J_thc(sJ_thc.local()), J_ham(sJ_ham.local());
    app_log(1, "[sym hartree] max|dJ| = {:.3e}", max_abs_diff(J_thc, J_ham));
    ARRAY_EQUAL(J_ham, J_thc, 1e-4);

    auto sK_thc = make_shared_array<array_view_4d_t>(*mpi_context, {ns, nk, nb, nb});
    auto sK_ham = make_shared_array<array_view_4d_t>(*mpi_context, {ns, nk, nb, nb});
    hf.evaluate(sK_thc, occ4d, thc, sS_skij.local(), false, true);
    hf.evaluate(sK_ham, occ4d, heval, sS_skij.local(), false, true);

    nda::array<ComplexType,4> K_thc(sK_thc.local()), K_ham(sK_ham.local());
    app_log(1, "[sym exchange] max|dK| = {:.3e}", max_abs_diff(K_thc, K_ham));
    ARRAY_EQUAL(K_ham, K_thc, 1e-4);
  }

  TEST_CASE("hamilt_exx_options_sharing", "[methods][hamilt][hf][qe][paw]") {
    // Plan 0.2 positive direction: two hamilt evaluators with IDENTICAL
    // non-default exx options must share the one pseudopot instance without
    // tripping the conflict check (the conflict direction is APP_ABORT and
    // cannot be exercised in-process).
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_paw"));

    ptree pt;
    pt.put("vv_compensation", "shape");
    hamilt_eval_t heval1(mf, pt);
    hamilt_eval_t heval2(mf, pt);
    auto* p1 = heval1.psp();  // sets shape options on the shared pseudopot
    auto* p2 = heval2.psp();  // identical options: accepted, same instance
    REQUIRE(p1 == p2);
    REQUIRE(p1->exx_options().vv_compensation == hamilt::vv_compensation_e::shape);
  }

} // bdft_tests
