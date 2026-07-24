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

// ISDF-Vertex: IBZ k-point symmetry support (notes/vertex_ibz_symmetry.md).
//
// Validation ladder (memo section 7):
//   1. vertex_ibz_leakage_diag: the C-window D-matrix leakage diagnostic computed
//      independently from MF->symmetry_rotation for a family of windows, tied to
//      the measured eigenvalue degeneracy structure (a window boundary slicing a
//      degenerate set <=> nonzero leakage). MF-only, fast.
//   2. vertex_ibz_gold: THE GOLD CHECK -- the same physical LiH-222 state driven
//      through the nosym (qe_lih222) and sym (qe_lih222_sym) variants with the
//      vertex on: e_hf/e_corr must agree to (cross-variant class) + O(leakage);
//      both error sources measured and reported separately (theory-owner item 3a).
//      Includes the near-closed-window control (item 3c) and the secondary-basis
//      path on the sym mesh.
//   3. vertex_ibz_conservation_sym: the conservation identity S_SigmaG + S_PW = 0
//      evaluated with star-weighted IBZ pairings (memo section 3.6) on the sym
//      mesh; sign-flip control at O(1).
//   4. vertex_ibz_noop_sym: C = empty set reproduces plain sym-scGW bitwise.

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

  namespace ibz_test_detail {

    // Independent reimplementation of the C-window leakage (memo (C-leak)):
    // for every qsymms position >= 1 and full-BZ k, the D-mass of the window
    // columns outside the window. Returns (max, mean).
    inline std::pair<double, double> window_leakage(mf::MF &mf, long c0, long c1) {
      const long nk = mf.nkpts();
      const long nbnd = mf.nbnd();
      const long nc = c1 - c0;
      auto qsymms = mf.qsymms();
      const long nsym = qsymms.extent(0);
      nda::array<cplx, 2> E(nbnd, nc), Dcols(nbnd, nc);
      E() = cplx(0.0);
      for (long j = 0; j < nc; ++j) E(c0 + j, j) = cplx(1.0);
      double mx = 0.0, sum = 0.0;
      long cnt = 0;
      using math::sparse::csrmm;
      for (long js = 1; js < nsym; ++js)
        for (long ik = 0; ik < nk; ++ik) {
          auto [cj, Dsp] = mf.symmetry_rotation(js, ik);
          (void)cj;
          csrmm(cplx(1.0), *Dsp, E, cplx(0.0), Dcols);
          double m_in = 0.0, m_all = 0.0;
          for (long a = 0; a < nbnd; ++a)
            for (long j = 0; j < nc; ++j) {
              const double w = std::norm(Dcols(a, j));
              m_all += w;
              if (a >= c0 and a < c1) m_in += w;
            }
          if (m_all > 1e-24) {
            const double l = 1.0 - m_in / m_all;
            mx = std::max(mx, l);
            sum += l;
            ++cnt;
          }
        }
      return {mx, (cnt > 0) ? sum / double(cnt) : 0.0};
    }

    // does the window boundary [c0, c1) slice through a degenerate eigenvalue set
    // at any (spin, k)? (the same 1e-4 degeneracy resolution generate_dmatrix uses,
    // symmetry.hpp:1039)
    inline bool window_splits_degeneracy(mf::MF &mf, long c0, long c1) {
      auto eig = mf.eigval();
      const long ns = eig.shape(0), nk = eig.shape(1), nb = eig.shape(2);
      auto split_at = [&](long b) {  // boundary between b-1 and b
        if (b <= 0 or b >= nb) return false;
        for (long is = 0; is < ns; ++is)
          for (long ik = 0; ik < nk; ++ik)
            if (std::abs(eig(is, ik, b) - eig(is, ik, b - 1)) < 1e-4) return true;
        return false;
      };
      return split_at(c0) or split_at(c1);
    }

  } // ibz_test_detail

  // ====================================================================================
  TEST_CASE("vertex_ibz_leakage_diag", "[methods][vertex][ibz]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_sym"));
    REQUIRE(mf->nkpts() != mf->nkpts_ibz());   // this IS a symmetry-reduced mesh

    struct row { long c0, c1; double mx, mean; bool splits; };
    std::vector<row> table;
    for (auto [c0, c1] : std::vector<std::pair<long,long>>{
             {0, 1}, {0, 2}, {1, 3}, {1, 2}, {0, 4}, {1, 4}, {2, 3}}) {
      auto [mx, mean] = ibz_test_detail::window_leakage(*mf, c0, c1);
      bool sp = ibz_test_detail::window_splits_degeneracy(*mf, c0, c1);
      table.push_back({c0, c1, mx, mean, sp});
      app_log(1, "ibz leakage: window [{}, {}): max = {:.3e}, mean = {:.3e}, "
                 "splits degenerate set = {}", c0, c1, mx, mean, sp);
    }
    // the diagnostic must track the degeneracy structure (theory-owner item 3b):
    // a window that does NOT split any degenerate set must be (near-)closed; a
    // window that does must show correspondingly larger leakage.
    double leak_closed_max = 0.0, leak_split_min = 1e300;
    bool have_closed = false, have_split = false;
    for (auto const& r : table) {
      if (r.splits) { have_split = true; leak_split_min = std::min(leak_split_min, r.mx); }
      else          { have_closed = true; leak_closed_max = std::max(leak_closed_max, r.mx); }
    }
    app_log(1, "ibz leakage: max over non-splitting windows = {:.3e}; "
               "min over splitting windows = {:.3e}", leak_closed_max,
            have_split ? leak_split_min : -1.0);
    if (have_closed) REQUIRE(leak_closed_max < 1e-6);
    if (have_closed and have_split) REQUIRE(leak_split_min > 10.0 * leak_closed_max);

    // probe the trev-carrying mesh used by the gold check (223: 2x2x3)
    {
      auto mf3 = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih223_sym"));
      long ntrev = mf3->nkpts_trev_pairs();
      long nqtrev = 0;
      for (long q = 0; q < mf3->nqpts(); ++q)
        if (mf3->qp_trev(q)) ++nqtrev;
      auto [l13, l13m] = ibz_test_detail::window_leakage(*mf3, 1, 3);
      (void)l13m;
      app_log(1, "ibz leakage: qe_lih223_sym: nk {} -> {} IBZ, nq {} -> {} IBZ, "
                 "trev k-pairs = {}, trev-mapped q = {}, leak[1,3) = {:.3e}",
              mf3->nkpts(), mf3->nkpts_ibz(), mf3->nqpts(), mf3->nqpts_ibz(),
              ntrev, nqtrev, l13);
    }
  }

  // ====================================================================================
  TEST_CASE("vertex_ibz_gold", "[methods][vertex][ibz][smoke]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_ibz_gold skipped: build has ENABLE_DLR=OFF.");
#else
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_ibz_gold";

    // one (variant, run) driver: n_iter scGW, optional vertex window, returns
    // (e_hf, e_corr, sym_leakage_max). Side channel: last_grot holds the most-recent
    // run's max G_CC G-rotation residual (see the REPRO block below).
    double last_grot = 0.0;
    auto run = [&](std::string const& mf_name, nda::range window, long n_iter,
                   std::string const& isdf_mode) {
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
                            mf->nbnd(), "ignore_g0", isdf_mode, 32, 1e-8);
      if (vtx.enabled()) {
        scr_eri.set_vertex(&vtx);
        gw.set_vertex(&vtx);
      }
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     n_iter, false, 1e-12, true);
      mpi_context->comm.barrier();
      double leak = vtx.sym_leakage_max();
      last_grot = vtx.g_rotation_max();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return std::make_tuple(e_hf, e_corr, leak);
    };

    // ---- cross-variant baseline: plain scGW (vertex off), 2 iterations ---------------
    auto [ehf_p_ns, ec_p_ns, l0a] = run("qe_lih222", nda::range(0, 0), 2, "global");
    auto [ehf_p_s, ec_p_s, l0b] = run("qe_lih222_sym", nda::range(0, 0), 2, "global");
    (void)l0a; (void)l0b;
    const double d_plain_hf = std::abs(ehf_p_ns - ehf_p_s);
    const double d_plain_ec = std::abs(ec_p_ns - ec_p_s);
    app_log(1, "ibz gold: PLAIN scGW cross-variant baseline: e_hf {:.12f} vs {:.12f} "
               "(|D| = {:.3e}); e_corr {:.12f} vs {:.12f} (|D| = {:.3e})",
            ehf_p_ns, ehf_p_s, d_plain_hf, ec_p_ns, ec_p_s, d_plain_ec);

    // ---- GOLD: vertex on, production window C = [1, 3), both cuts, 2 iterations ------
    auto [ehf_v_ns, ec_v_ns, lv_ns] = run("qe_lih222", nda::range(1, 3), 2, "global");
    auto [ehf_v_s, ec_v_s, lv_s] = run("qe_lih222_sym", nda::range(1, 3), 2, "global");
    (void)lv_ns;
    const double d_vert_hf = std::abs(ehf_v_ns - ehf_v_s);
    const double d_vert_ec = std::abs(ec_v_ns - ec_v_s);
    // the vertex effect (the physical signal the symmetry path must reproduce)
    const double shift_ns = std::abs(ec_v_ns - ec_p_ns);
    const double shift_s = std::abs(ec_v_s - ec_p_s);
    app_log(1, "ibz gold: VERTEX C=[1,3): e_hf {:.12f} vs {:.12f} (|D| = {:.3e}); "
               "e_corr {:.12f} vs {:.12f} (|D| = {:.3e})",
            ehf_v_ns, ehf_v_s, d_vert_hf, ec_v_ns, ec_v_s, d_vert_ec);
    app_log(1, "ibz gold: vertex e_corr shift: nosym {:.3e}, sym {:.3e}; "
               "measured C-window leakage (sym) = {:.3e}", shift_ns, shift_s, lv_s);
    app_log(1, "ibz gold: attribution: |D e_corr(vertex)| = {:.3e} vs baseline "
               "|D e_corr(plain)| = {:.3e} + O(leakage)*shift = {:.3e}",
            d_vert_ec, d_plain_ec, lv_s * shift_s);
    for (double e : {ehf_v_ns, ec_v_ns, ehf_v_s, ec_v_s}) REQUIRE(std::isfinite(e));
    REQUIRE(shift_ns > 1e-6);            // the vertex actually did something
    REQUIRE(shift_s > 1e-6);
    // the two variants must agree on the vertex physics: the sym-vs-nosym deviation
    // is bounded by (cross-variant baseline) + (leakage scale on the vertex shift)
    // + kernel headroom. A conjugation/rotation bug shows up at O(shift) instead.
    REQUIRE(d_vert_ec <= d_plain_ec + std::max(0.25 * shift_ns, 5.0 * lv_s * shift_s) + 1e-6);
    REQUIRE(d_vert_hf <= d_plain_hf + std::max(0.25 * shift_ns, 5.0 * lv_s * shift_s) + 1e-6);
    // the shifts themselves must agree to the same class
    REQUIRE(std::abs(shift_ns - shift_s) <=
            d_plain_ec + std::max(0.25 * shift_ns, 5.0 * lv_s * shift_s) + 1e-6);

    // NOTE (measured, vertex_ibz_leakage_diag): C = [1,3) is EXACTLY symmetry-closed
    // on qe_lih222_sym (leak = 0) -- the gold comparison above is therefore the
    // clean-separation case of theory-owner item 3c: pure kernel/cross-variant
    // class, no leakage contribution.

    // ---- LEAKY-WINDOW control (theory-owner item 3b): C = [1, 4) splits a
    // degenerate conduction set (measured leak ~0.33). The deviation may grow to
    // O(leak * shift) but must remain finite and controlled.
    {
      auto [ehf_l_ns, ec_l_ns, ll_ns] = run("qe_lih222", nda::range(1, 4), 1, "global");
      auto [ehf_l_s, ec_l_s, ll_s] = run("qe_lih222_sym", nda::range(1, 4), 1, "global");
      (void)ll_ns; (void)ehf_l_ns; (void)ehf_l_s;
      const double shift_l = 0.5 * (std::abs(ec_l_ns - ec_p_ns) + std::abs(ec_l_s - ec_p_s));
      const double d_l = std::abs(ec_l_ns - ec_l_s);
      app_log(1, "ibz gold: LEAKY-WINDOW control C=[1,4): measured leakage = {:.3e}; "
                 "e_corr {:.12f} vs {:.12f}: |D| = {:.3e} vs shift = {:.3e} "
                 "(closed-window |D| = {:.3e})", ll_s, ec_l_ns, ec_l_s, d_l, shift_l, d_vert_ec);
      REQUIRE(std::isfinite(ec_l_ns));
      REQUIRE(std::isfinite(ec_l_s));
      REQUIRE(ll_s > 0.1);                     // the diagnostic sees the deep cut
      // controlled: bounded by the leakage scale on the vertex signal (+ baseline)
      REQUIRE(d_l <= d_plain_ec + 3.0 * ll_s * shift_l + 0.25 * shift_l + 1e-6);
    }

    // ---- TIME-REVERSAL gold (qe_lih223: 4 trev k-pairs, 4 trev-mapped q; the
    // trev-leg conj and the PQ-transpose transfer branches are exercised here;
    // C = [1,3) is measured closed on this mesh too) -------------------------------
    {
      auto [ehf3_p_ns, ec3_p_ns, l3a] = run("qe_lih223", nda::range(0, 0), 1, "global");
      auto [ehf3_p_s, ec3_p_s, l3b] = run("qe_lih223_sym", nda::range(0, 0), 1, "global");
      (void)l3a; (void)l3b;
      auto [ehf3_v_ns, ec3_v_ns, l3c] = run("qe_lih223", nda::range(1, 3), 1, "global");
      auto [ehf3_v_s, ec3_v_s, l3d] = run("qe_lih223_sym", nda::range(1, 3), 1, "global");
      (void)l3c;
      const double d3_plain = std::abs(ec3_p_ns - ec3_p_s);
      const double d3_vert = std::abs(ec3_v_ns - ec3_v_s);
      const double shift3 = 0.5 * (std::abs(ec3_v_ns - ec3_p_ns) + std::abs(ec3_v_s - ec3_p_s));
      app_log(1, "ibz gold (223/trev): plain |D e_corr| = {:.3e}; vertex e_corr "
                 "{:.12f} vs {:.12f}: |D| = {:.3e}, shift = {:.3e}, leak = {:.3e}",
              d3_plain, ec3_v_ns, ec3_v_s, d3_vert, shift3, l3d);
      for (double e : {ehf3_v_ns, ec3_v_ns, ehf3_v_s, ec3_v_s}) REQUIRE(std::isfinite(e));
      REQUIRE(shift3 > 1e-7);
      REQUIRE(d3_vert <= d3_plain + std::max(0.25 * shift3, 5.0 * l3d * shift3) + 1e-6);
      REQUIRE(std::abs(ehf3_v_ns - ehf3_v_s) <=
              std::abs(ehf3_p_ns - ehf3_p_s) + 0.25 * shift3 + 1e-6);
    }

    // ---- secondary basis on the sym mesh (Refinement 2 under symmetry) ---------------
    {
      auto [ehf_sec, ec_sec, lsec] = run("qe_lih222_sym", nda::range(1, 3), 1, "secondary");
      auto [ehf_glo, ec_glo, lglo] = run("qe_lih222_sym", nda::range(1, 3), 1, "global");
      (void)lsec; (void)lglo;
      app_log(1, "ibz gold: sym-mesh secondary vs global (1 iteration): e_corr "
                 "{:.12f} vs {:.12f} (|D| = {:.3e}); e_hf |D| = {:.3e}",
              ec_sec, ec_glo, std::abs(ec_sec - ec_glo), std::abs(ehf_sec - ehf_glo));
      REQUIRE(std::isfinite(ec_sec));
      // at the numerical full pair rank the secondary path tracks global to the
      // downfold class (refinement2 memo 10.2: machine-level on the nosym mesh;
      // allow the svd_tol/kernel class here)
      REQUIRE(std::abs(ec_sec - ec_glo) <= 1e-5 + 0.05 * std::abs(ec_glo - ec_p_s));
    }

    // ---- REPRODUCTION (secondary + sym mesh, TWO iterations): does the self-consistent
    // G_CC stay symmetry-consistent past iter-1 in the secondary path? C=[1,3) is
    // symmetry-CLOSED on qe_lih222_sym (D-leak = 0, gold block above) and LiH-222
    // secondary tracks global to ~1e-5 at 1 iter (block above) -- so BOTH the window-
    // leakage and the basis-crudeness confounds are removed. A large secondary
    // G-rotation residual here (vs global's ~machine value) isolates a secondary-path
    // symmetry-unfolding defect in the self-consistent vertex feedback -- the same
    // signature seen in the sec_scgwvtx_M8 production run (iter-2 residual 5e-9 -> 0.49).
    {
      auto [ehf_sec2, ec_sec2, lsec2] = run("qe_lih222_sym", nda::range(1, 3), 2, "secondary");
      const double grot_sec2 = last_grot;
      auto [ehf_glo2, ec_glo2, lglo2] = run("qe_lih222_sym", nda::range(1, 3), 2, "global");
      const double grot_glo2 = last_grot;
      (void)ehf_sec2; (void)ehf_glo2; (void)lsec2; (void)lglo2;
      app_log(1, "ibz REPRO (2-iter, C=[1,3) closed): e_corr secondary {:.12f} vs global "
                 "{:.12f} (|D| = {:.3e}); G-rotation residual secondary = {:.3e}, global = "
                 "{:.3e}; window D-leak secondary = {:.3e}, global = {:.3e}",
              ec_sec2, ec_glo2, std::abs(ec_sec2 - ec_glo2), grot_sec2, grot_glo2, lsec2, lglo2);
      REQUIRE(std::isfinite(ec_sec2));
      REQUIRE(std::isfinite(ec_glo2));
      // EXPECT (if the secondary sym multi-iter path is correct): the secondary
      // G-rotation residual stays the same small class as global's on this closed window,
      // and the self-consistent e_corr tracks global to the downfold class it held at 1 iter.
      REQUIRE(grot_sec2 <= std::max(1e-6, 10.0 * grot_glo2));
      REQUIRE(std::abs(ec_sec2 - ec_glo2) <= 1e-4 + 0.05 * std::abs(ec_glo2 - ec_p_s));
    }
#endif  // ENABLE_DLR
  }

  // ====================================================================================
  TEST_CASE("vertex_ibz_conservation_sym", "[methods][vertex][ibz][smoke]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_ibz_conservation_sym skipped: build has ENABLE_DLR=OFF.");
#else
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_ibz_cons";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_sym"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    // physical state: one plain scGW iteration + RPA-W rebuild (consistent (G, W))
    solvers::hf_t hf;
    solvers::gw_t gw(&ft, "ignore_g0", output);
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
    simple_dyson dyson(mf.get(), &ft);
    MBState mb_state(mpi_context, ft, output);
    iter_scf::iter_scf_t iter_sol("damping");
    auto [e_hf_0, e_corr_0] = scf_loop(mb_state, dyson, eri, ft,
                                       solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                       1, false, 1e-9, true);
    REQUIRE(std::isfinite(e_hf_0));
    REQUIRE(std::isfinite(e_corr_0));
    if (not mb_state.dW_qtPQ.has_value()) scr_eri.update_w(mb_state, thc, -1);
    REQUIRE(mb_state.dW_qtPQ.has_value());
    mpi_context->comm.barrier();

    auto MFp = thc.MF();
    const long nkpts = MFp->nkpts();
    const long nkpts_ibz = MFp->nkpts_ibz();
    const long nqpts_ibz = MFp->nqpts_ibz();
    const long Np = thc.Np();
    const long nbnd = MFp->nbnd();
    nda::range C(1, 3);

    auto G_loc = mb_state.sG_tskij.value().local();
    const long nt = G_loc.shape(0);
    const long ns = G_loc.shape(1);
    const long nt_half = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;
    REQUIRE(G_loc.shape(2) == nkpts_ibz);

    // star multiplicities (memo section 3.6)
    nda::array<double, 1> m_k(nkpts_ibz), m_q(nqpts_ibz);
    m_k() = 0.0; m_q() = 0.0;
    for (long k = 0; k < nkpts; ++k) m_k(MFp->kp_to_ibz(k)) += 1.0;
    for (long q = 0; q < MFp->nqpts(); ++q) m_q(MFp->qp_to_ibz(q)) += 1.0;

    // ---- Sigma^C alone: zero the accumulator, eval, read back ------------------------
    solvers::vertex_t vtx(&ft, "2nd_exchange", C, nbnd, "ignore_g0", "global");
    REQUIRE(vtx.active());
    mb_state.sSigma_tskij.value().set_zero();
    mpi_context->comm.barrier();
    vtx.eval_Sigma_C(mb_state, thc);
    nda::array<cplx, 5> Sig(nt, ns, nkpts_ibz, nbnd, nbnd);
    Sig = mb_state.sSigma_tskij.value().local();
    app_log(1, "ibz cons: sym leakage max = {:.3e}", vtx.sym_leakage_max());

    // ---- Pi^C on the IBZ grid (code tau storage) -------------------------------------
    const std::array<long, 4> pgrid = {1, 1, 1, mpi_context->comm.size()};
    const std::array<long, 4> bsize = {1, 1, 1, 1};
    const std::array<long, 4> gshape = {nt_half, nqpts_ibz, Np, Np};
    auto dPi = vtx.eval_Pi_C(mb_state, thc, pgrid, bsize, gshape);
    nda::array<cplx, 4> Pi_code(nt_half, nqpts_ibz, Np, Np);
    Pi_code() = cplx(0.0);
    Pi_code(dPi.local_range(0), dPi.local_range(1), dPi.local_range(2), dPi.local_range(3)) =
        dPi.local();
    mpi_context->comm.all_reduce_in_place_n(Pi_code.data(), Pi_code.size(), std::plus<>{});

    // ---- Wdyn(tau) replicated + Z(q) -------------------------------------------------
    nda::array<cplx, 4> Wt(nqpts_ibz, nt_half, Np, Np);
    {
      auto& dW = mb_state.dW_qtPQ.value();
      Wt() = cplx(0.0);
      Wt(dW.local_range(0), dW.local_range(1), dW.local_range(2), dW.local_range(3)) =
          dW.local();
      mpi_context->comm.all_reduce_in_place_n(Wt.data(), Wt.size(), std::plus<>{});
    }
    nda::array<cplx, 3> Zq(nqpts_ibz, Np, Np);
    for (long iq = 0; iq < nqpts_ibz; ++iq) Zq(iq, all_r, all_r) = thc.Z(int(iq));

    // ---- pairing machinery (conservation_validation.md section 1.6) -------------------
    auto Twt_bb = ft.Twt_bb();
    long m0 = -1;
    {
      auto wnb = ft.wn_mesh_b();
      for (long m = 0; m < ft.nw_b(); ++m)
        if (wnb(m) == 0) { m0 = m; break; }
    }
    REQUIRE(m0 >= 0);
    // tau = 0 interpolation row (x = -1)
    nda::array<double, 1> x0(1);
    x0(0) = -1.0;
    auto Row0_arr = ft.construct_tau_interpolate_matrix(x0);   // (1, nt)
    const double spin = (ns == 1) ? 2.0 : 1.0;

    // S_SigmaG = -(spin/Nk) sum_{s,k in IBZ} m_k Twt_bb(m0,:) . f_k,
    //   f_k(tau) = sum_{ab in C} Sig_ab(k, tau) G_ab(k, beta - tau)
    cplx S_SG(0.0);
    {
      nda::array<cplx, 1> f(nt);
      for (long is = 0; is < ns; ++is)
        for (long k = 0; k < nkpts_ibz; ++k) {
          f() = cplx(0.0);
          for (long it = 0; it < nt; ++it) {
            const long itm = nt - it - 1;
            cplx acc(0.0);
            for (long a = C.first(); a < C.last(); ++a)
              for (long b = C.first(); b < C.last(); ++b)
                acc += Sig(it, is, k, a, b) * G_loc(itm, is, k, a, b);
            f(it) = acc;
          }
          cplx row(0.0);
          for (long it = 0; it < nt; ++it) row += Twt_bb(m0, it) * f(it);
          S_SG += m_k(k) * row;
        }
      S_SG *= cplx(-spin / double(nkpts));
    }

    // S_PW = +(1/Nk) sum_{q in IBZ} m_q [ sum_MN Pi(q, tau=0) Z_NM
    //                                     + Twt_bb(m0,:) . g_q ],
    //   g_q(tau) = sum_MN Pi_notes(q,tau) Wdyn_NM(q, beta-tau); both PH-symmetric,
    //   Pi_notes on the full grid from the code storage via the PH mirror
    //   (pi design section 2 rule 3: code(it) = notes(beta - tau_it), notes PH-sym).
    cplx S_PW(0.0);
    {
      nda::array<cplx, 2> Pi_full_t(nt, Np * Np);   // notes-tau, one q at a time
      for (long q = 0; q < nqpts_ibz; ++q) {
        for (long it = 0; it < nt; ++it) {
          const long ih = std::min(it, nt - it - 1);   // PH-symmetric storage
          auto src = Pi_code(ih, q, all_r, all_r);
          for (long M = 0; M < Np; ++M)
            for (long N = 0; N < Np; ++N) Pi_full_t(it, M * Np + N) = src(M, N);
        }
        // tau = 0 value from the DLR interpolation row
        cplx SZ(0.0);
        {
          nda::array<cplx, 1> Pi0(Np * Np);
          Pi0() = cplx(0.0);
          for (long it = 0; it < nt; ++it)
            for (long MN = 0; MN < Np * Np; ++MN)
              Pi0(MN) += cplx(Row0_arr(0, it)) * Pi_full_t(it, MN);
          for (long M = 0; M < Np; ++M)
            for (long N = 0; N < Np; ++N) SZ += Pi0(M * Np + N) * Zq(q, N, M);
        }
        // dynamic part
        cplx SW(0.0);
        for (long it = 0; it < nt; ++it) {
          const long ihw = std::min(it, nt - it - 1);   // Wdyn(beta-tau) = Wdyn(tau)
          cplx g(0.0);
          for (long M = 0; M < Np; ++M)
            for (long N = 0; N < Np; ++N)
              g += Pi_full_t(it, M * Np + N) * Wt(q, ihw, N, M);
          SW += Twt_bb(m0, it) * g;
        }
        S_PW += m_q(q) * (SZ + SW);
      }
      S_PW *= cplx(1.0 / double(nkpts));
    }

    const double scale = std::max(std::abs(S_SG), std::abs(S_PW));
    const double rel = std::abs(S_SG + S_PW) / std::max(scale, 1e-300);
    const double ctrl = std::abs(S_SG - S_PW) / std::max(scale, 1e-300);
    app_log(1, "ibz cons: S_SigmaG = ({:.10e}, {:.2e}), S_PW = ({:.10e}, {:.2e})",
            S_SG.real(), S_SG.imag(), S_PW.real(), S_PW.imag());
    app_log(1, "ibz cons: |S_SG + S_PW| / scale = {:.3e} (sign-flip control = {:.3f}; "
               "leakage = {:.3e})", rel, ctrl, vtx.sym_leakage_max());
    REQUIRE(scale > 1e-8);
    // kernel-accuracy + D-matrix-accuracy + O(leakage) class (memo section 7 item 2;
    // measured 5.05e-5 on qe_lih222_sym with leak = 0 -- ~13x the nosym identity's
    // 3.78e-6, the D-overlap accuracy class); the sign-flip control breaks at O(1)
    REQUIRE(rel < 5e-3);
    REQUIRE(ctrl > 1.5);

    if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
    mpi_context->comm.barrier();
#endif  // ENABLE_DLR
  }

  // ====================================================================================
  TEST_CASE("vertex_ibz_noop_sym", "[methods][vertex][ibz][smoke]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_ibz_noop_sym skipped: build has ENABLE_DLR=OFF.");
#else
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_ibz_noop";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_sym"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    auto run = [&](bool with_empty_vertex) {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, with_empty_vertex ? "2nd_exchange" : "none",
                            nda::range(0, 0), mf->nbnd());
      REQUIRE(not vtx.active());
      if (vtx.enabled()) {
        scr_eri.set_vertex(&vtx);
        gw.set_vertex(&vtx);
      }
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     1, false, 1e-12, true);
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return std::make_pair(e_hf, e_corr);
    };

    auto [ehf0, ec0] = run(false);
    auto [ehf1, ec1] = run(true);
    app_log(1, "ibz noop: plain sym-scGW e_hf = {:.17g}, e_corr = {:.17g}", ehf0, ec0);
    app_log(1, "ibz noop: empty-vertex   e_hf = {:.17g}, e_corr = {:.17g}", ehf1, ec1);
    REQUIRE(ehf0 == ehf1);   // bitwise
    REQUIRE(ec0 == ec1);     // bitwise
#endif  // ENABLE_DLR
  }

} // bdft_tests
