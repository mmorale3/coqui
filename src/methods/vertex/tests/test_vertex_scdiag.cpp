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

// ISDF-Vertex self-consistency diagnostics.
//
// The IBZ tests compare the secondary vs the global basis over a few iterations, so a
// symmetry defect shared by both paths -- or one that lives in the plain scGW
// baseline -- is invisible to them. This file supplies the controls:
//   0. vertex_scdiag_transfer_map_consistency: the k-point maps used by the kernels
//      carry the stored momentum transfer.
//   1. vertex_scdiag_baseline_symmetry: the little-group residual of the self-
//      consistent G with the vertex off. G(k) must commute with the representation
//      of the little group of k -- this is a property of any correct Sigma, so a
//      large value here points to the plain scGW path, not the vertex.
//   2. vertex_scdiag_vertex_symmetry: the same residual with the vertex on, at
//      1-4 iterations, global basis, on a symmetry-closed window (no D-matrix
//      leakage), so the window-truncation confound is removed.
//   3. vertex_scdiag_trajectory: sym vs nosym e_corr trajectory over several
//      iterations. A symmetry defect that is invisible in a 2-iteration energy
//      comparison shows up as a growing sym-vs-nosym gap.
//   4. vertex_scdiag_dielectric_stability: e_corr as the window C grows
//      (characterization of the stable envelope of eps = I - Z.Pi).
//
// The residual is measured on the full band block (the vertex's own diagnostic uses
// the C block only), so it is independent of the C-window truncation.

#include <cmath>
#include <complex>
#include <tuple>
#include <vector>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"

#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/sparse/csr_blas.hpp"

#include "mean_field/default_MF.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include "methods/vertex/vertex_t.h"

namespace bdft_tests {

  using namespace methods;
  using cplx = ComplexType;
  decltype(nda::range::all) constexpr all_r = nda::range::all;

  namespace scdiag_detail {

    /**
     * Little-group consistency residual of an IBZ-stored G.
     *
     * CoQui convention (thc_solver_comm::_primary_to_aux_impl): the full-BZ value of
     * any band-basis operator is the IBZ block VERBATIM (transposed for trev k) --
     * the stored orbitals at an image point are the canonically rotated IBZ orbitals.
     * MF->symmetry_rotation(js, k) is d(R, k)(a,b) = <psi(k.R, a) | S psi(k, b)>
     * (utilities/symmetry.hpp); composing the canonical rotations of k and of
     * ks_to_k(js, k) makes it the band representation of an element of the LITTLE
     * GROUP of kp_to_ibz(k). Any correct self-energy therefore satisfies
     *
     *     G(k_ibz) = D^dag G(k_ibz) D           for every such D,
     *
     * up to the nbnd-truncation leakage of D. Measured in the Frobenius norm on one
     * tau slice over the [b0, b1) band block; returns the worst (relative) value.
     */
    inline double little_group_residual(mf::MF &mf,
                                        nda::MemoryArrayOfRank<5> auto const &G_tskij,
                                        long b0, long b1) {
      const long nk = mf.nkpts();
      const long nbnd = mf.nbnd();
      const long nb = b1 - b0;
      const long it = G_tskij.shape(0) / 2, is = 0;
      auto qsymms = mf.qsymms();
      const long nsym = qsymms.extent(0);
      auto kp_to_ibz = mf.kp_to_ibz();
      auto kp_trev = mf.kp_trev();
      nda::array<cplx, 2> E(nbnd, nb), Dcols(nbnd, nb), T1(nb, nb), T2(nb, nb), Gsrc(nb, nb);
      E() = cplx(0.0);
      for (long j = 0; j < nb; ++j) E(b0 + j, j) = cplx(1.0);
      double worst = 0.0;
      using math::sparse::csrmm;
      for (long js = 1; js < nsym; ++js)
        for (long k = 0; k < nk; ++k) {
          if (kp_trev(k)) continue;
          auto [cj, Dsp] = mf.symmetry_rotation(js, k);
          if (cj) continue;
          const long ksrc = mf.ks_to_k(int(js), int(k));
          const long kib = kp_to_ibz(k), kib_src = kp_to_ibz(ksrc);
          // both points sit in the same star -- assert the map rather than trust it
          utils::check(kib == kib_src,
                       "scdiag: ks_to_k({}, {}) = {} left the star ({} vs {}).",
                       js, k, ksrc, kib, kib_src);
          csrmm<'N'>(cplx(1.0), *Dsp, E, cplx(0.0), Dcols);   // Dcols(nbnd, nb) = D . E
          // restrict to the measured band block (exact when the block is D-closed)
          nda::array<cplx, 2> Dblk(nb, nb);
          for (long a = 0; a < nb; ++a)
            for (long j = 0; j < nb; ++j) Dblk(a, j) = Dcols(b0 + a, j);
          Gsrc() = G_tskij(it, is, kib_src, nda::range(b0, b1), nda::range(b0, b1));
          nda::blas::gemm(cplx(1.0), nda::dagger(Dblk), Gsrc, cplx(0.0), T1);
          nda::blas::gemm(cplx(1.0), T1, Dblk, cplx(0.0), T2);
          double num = 0.0, den = 0.0;
          for (long a = 0; a < nb; ++a)
            for (long b = 0; b < nb; ++b) {
              num += std::norm(T2(a, b) - G_tskij(it, is, kib, b0 + a, b0 + b));
              den += std::norm(G_tskij(it, is, kib, b0 + a, b0 + b));
            }
          if (den > 1e-24) worst = std::max(worst, std::sqrt(num / den));
        }
      return worst;
    }

    struct scf_result {
      double e_hf, e_corr;
      double grot_full;     // little-group residual, full band block
      double grot_win;      // little-group residual, C window
      double vtx_grot;      // the vertex's own C-block diagnostic (running max)
      long bare_rung;       // times Pi^C fell back to the BARE rung (must be 0)
    };

    // one scGW run; vertex active iff window.size() > 0.
    template<typename ctx_t>
    scf_result run_scf(ctx_t &mpi_context, imag_axes_ft::IAFT &ft,
                       std::string const &mf_name, nda::range window, long n_iter,
                       std::string const &isdf_mode, nda::range meas_win) {
      std::string output = "coqui_vertex_scdiag";
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, mf_name));
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, window.size() > 0 ? "2nd_exchange" : "none", window,
                            mf->nbnd(), "ignore_g0", isdf_mode, 32, 1e-8, -1.0, -1.0);
      if (vtx.enabled()) {
        scr_eri.set_vertex(&vtx);
        gw.set_vertex(&vtx);
      }
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     n_iter, false, 1e-12, true);
      mpi_context->comm.barrier();
      scf_result r{e_hf, e_corr, 0.0, 0.0, vtx.g_rotation_max(), vtx.bare_rung_uses()};
      if (mf->nkpts() != mf->nkpts_ibz()) {
        auto G = mb_state.sG_tskij.value().local();
        r.grot_full = little_group_residual(*mf, G, 0, mf->nbnd());
        r.grot_win = little_group_residual(*mf, G, meas_win.first(), meas_win.last());
      }
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return r;
    }

  } // scdiag_detail

  // ====================================================================================
  // MAP CONSISTENCY: the rung-transfer momentum identity the Sigma^C / Pi^C kernels rely
  // on, expressed ONLY in terms of the maps they actually index with.
  //
  // For a full-BZ transfer q' the kernels pair the effective columns
  //   Xhat(js, k)  ~  collocation at krot(js, k) = ks_to_k(js, k)
  //   Xhat(js, k-q') ~ collocation at krot(js, k-q')
  // and contract them with the IBZ-STORED core at q_star(q'). That is only legitimate
  // if the rotated legs really carry the stored transfer:
  //
  //   krot(js,k) - krot(js, k-q')  ==  (+/-) q_star(q')      (mod G),
  //   sign = -1 for a time-reversal-mapped transfer.
  //
  // build_sym_ctx asserts the direction on the Q MESH ALONE (q'*R vs q_star, using
  // symm_list[qsymms(js)].R) but never checks that ks_to_k -- a DIFFERENT map, built in
  // the MF from find_inverse_symmetry(qsymms) -- rotates the k legs the same way. If the
  // two differ by an inversion the legs are paired against the wrong core, and the error
  // is a rotation of the rung, i.e. exactly a little-group-covariance violation.
  // ====================================================================================
  TEST_CASE("vertex_scdiag_transfer_map_consistency", "[methods][vertex][scdiag]") {
    auto &mpi_context = utils::make_unit_test_mpi_context();
    for (auto name : {"qe_lih222_sym", "qe_lih223_sym"}) {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, name));
      if (mf->nkpts() == mf->nkpts_ibz()) continue;
      auto kcrys = mf->kpts_crystal();
      auto qsymms = mf->qsymms();
      const long nsym = qsymms.extent(0);
      const long nq = mf->nqpts(), nk = mf->nkpts();

      // crystal q list, built the same way build_sym_ctx does (Qpts() is cartesian)
      nda::array<double, 2> qcrys(nq, 3);
      for (long iq = 0; iq < nq; ++iq) {
        const long k2 = mf->qk_to_k2(int(iq), 0);
        for (int i = 0; i < 3; ++i) qcrys(iq, i) = kcrys(0, i) - kcrys(k2, i);
      }
      auto same_mod_G = [](double a, double b) {
        double d = a - b; d -= std::round(d); return std::abs(d) < 1e-6;
      };

      long bad = 0, tot = 0;
      for (long iq = 0; iq < nq; ++iq) {
        const int sidx = mf->qp_symm(int(iq));
        long js = -1;
        for (long i = 0; i < nsym; ++i) if (qsymms(i) == sidx) { js = i; break; }
        REQUIRE(js >= 0);
        const long qs = mf->qp_to_ibz(int(iq));
        const double sgn = mf->qp_trev(int(iq)) ? -1.0 : 1.0;
        for (long ik = 0; ik < nk; ++ik) {
          const long ikmq = mf->qk_to_k2(int(iq), int(ik));
          const long r1 = mf->ks_to_k(int(js), int(ik));
          const long r2 = mf->ks_to_k(int(js), int(ikmq));
          ++tot;
          for (int i = 0; i < 3; ++i)
            if (not same_mod_G(kcrys(r1, i) - kcrys(r2, i), sgn * qcrys(qs, i))) {
              ++bad; break;
            }
        }
      }
      app_log(1, "scdiag TRANSFER-MAP ({}): krot(js,k) - krot(js,k-q') vs (+/-)q_star: "
                 "{} of {} (q', k) pairs INCONSISTENT", name, bad, tot);
      REQUIRE(bad == 0);
    }
  }

  // ====================================================================================
  // CONTROL 1: does PLAIN scGW keep the self-consistent G little-group symmetric?
  // ====================================================================================
  TEST_CASE("vertex_scdiag_baseline_symmetry", "[methods][vertex][scdiag]") {
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    const nda::range meas(1, 3);

    for (long nit : {1L, 2L, 4L}) {
      auto r = scdiag_detail::run_scf(mpi_context, ft, "qe_lih222_sym", nda::range(0, 0),
                                      nit, "global", meas);
      app_log(1, "scdiag BASELINE (vertex OFF) niter = {}: e_corr = {:.12f}; "
                 "little-group residual: full band block = {:.3e}, C = [1,3) = {:.3e}",
              nit, r.e_corr, r.grot_full, r.grot_win);
      REQUIRE(std::isfinite(r.e_corr));
      // Loose bound only: a correct Sigma is little-group covariant, but the diagnostic
      // is limited by the accuracy of the D matrices themselves (numerically computed
      // overlaps, truncated at nbnd). On LiH-222 that floor is ~1e-3. The value is
      // reported so the vertex run can be compared against it.
      REQUIRE(r.grot_full < 1.0);
    }
  }

  // ====================================================================================
  // CONTROL 2: does the VERTEX keep it? Symmetry-closed window (no D-matrix leakage on
  // qe_lih222_sym for C = [1,3)), global basis -- no window leakage, no secondary-basis
  // conditioning confound. Any growth vs CONTROL 1 is due to the vertex.
  // ====================================================================================
  TEST_CASE("vertex_scdiag_vertex_symmetry", "[methods][vertex][scdiag]") {
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    const nda::range C(1, 3);

    std::vector<double> resid;
    for (long nit : {1L, 2L, 3L, 4L}) {
      auto r = scdiag_detail::run_scf(mpi_context, ft, "qe_lih222_sym", C, nit,
                                      "global", C);
      app_log(1, "scdiag VERTEX (global, C = [1,3)) niter = {}: e_corr = {:.12f}; "
                 "little-group residual: full = {:.3e}, C = {:.3e}; vertex C-block "
                 "running max = {:.3e}",
              nit, r.e_corr, r.grot_full, r.grot_win, r.vtx_grot);
      REQUIRE(std::isfinite(r.e_corr));
      // Pi^C is a functional of the screened W; update_w bootstraps W before the first
      // Pi^C evaluation, so the bare-rung fallback must never be taken (using the bare
      // rung grossly overestimates the screening on the first iteration and spoils the
      // whole trajectory). Zero tolerance.
      REQUIRE(r.bare_rung == 0);
      resid.push_back(r.grot_full);
    }
    // the vertex must not produce a symmetry violation orders of magnitude above the
    // plain-scGW floor of vertex_scdiag_baseline_symmetry (~1e-3 on LiH-222).
    REQUIRE(resid.back() < 1e-1);
  }

  // ====================================================================================
  // CONTROL 3: sym vs nosym e_corr over a longer trajectory. A defect that compounds
  // over iterations shows up here even if a 2-iteration comparison misses it.
  // ====================================================================================
  TEST_CASE("vertex_scdiag_trajectory", "[methods][vertex][scdiag]") {
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    const nda::range C(1, 3);

    for (long nit : {1L, 2L, 4L, 6L}) {
      auto ns = scdiag_detail::run_scf(mpi_context, ft, "qe_lih222", C, nit, "global", C);
      auto sy = scdiag_detail::run_scf(mpi_context, ft, "qe_lih222_sym", C, nit,
                                       "global", C);
      auto p_ns = scdiag_detail::run_scf(mpi_context, ft, "qe_lih222", nda::range(0, 0),
                                         nit, "global", C);
      const double shift = std::abs(ns.e_corr - p_ns.e_corr);
      app_log(1, "scdiag TRAJECTORY niter = {}: e_corr nosym {:.12f} vs sym {:.12f} "
                 "(|D| = {:.3e}); vertex shift = {:.3e}; ratio |D|/shift = {:.3e}; "
                 "sym little-group residual = {:.3e}",
              nit, ns.e_corr, sy.e_corr, std::abs(ns.e_corr - sy.e_corr), shift,
              std::abs(ns.e_corr - sy.e_corr) / std::max(shift, 1e-30), sy.grot_full);
      REQUIRE(std::isfinite(ns.e_corr));
      REQUIRE(std::isfinite(sy.e_corr));
    }
  }


  // ====================================================================================
  // DIELECTRIC STABILITY vs the size of the vertex.
  //
  // P^C -- unlike the RPA polarization, which is negative semi-definite on the imaginary
  // axis -- can push eps = I - Z.Pi through zero, after which dyson_W_in_place's inverse
  // is meaningless and Sigma explodes on the next iteration. This instability is driven
  // by the size of the vertex, so widening C on a small system (LiH-222) probes it.
  //
  // Reports, per window and per iteration: e_corr, and (from the update_w log) the
  // dielectric conditioning max_(q,i.nu) ||[I - Z.Pi]^-1||_max. A window that stays
  // O(1) is inside the stable envelope; one that blows up is outside it.
  // ====================================================================================
  TEST_CASE("vertex_scdiag_dielectric_stability", "[methods][vertex][scdiag]") {
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");

    // widen C on the SAME system: the vertex grows, everything else is held fixed
    for (auto w : std::vector<nda::range>{nda::range(0, 0), nda::range(1, 3),
                                          nda::range(0, 4), nda::range(0, 6)}) {
      auto r = scdiag_detail::run_scf(mpi_context, ft, "qe_lih222", w, 5, "global",
                                      nda::range(1, 3));
      app_log(1, "scdiag DIELECTRIC-STABILITY C = [{}, {}) (nc = {}), 5 iterations: "
                 "e_corr = {:.12f} {}",
              w.first(), w.last(), w.size(), r.e_corr,
              std::isfinite(r.e_corr) ? "" : "  <-- NOT FINITE");
      // Characterization only: widening C deliberately leaves the regime where
      // eps = I - Z.Pi stays positive definite, so a window that blows up is expected,
      // not a regression. Only the vertex-off case is required to be finite; the rest is
      // reported, with the per-iteration "dielectric conditioning" log line.
      if (w.size() == 0) REQUIRE(std::isfinite(r.e_corr));
    }
  }

} // bdft_tests
