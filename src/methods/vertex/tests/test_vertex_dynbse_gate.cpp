/**
 * scGW-tilde Tier 2 full frequency, increment D2 (notes/dynbse_plan.md section 9): the production
 * driver's gates on the qe_lih222 fixture with the production pol-only attachment (vertex_type =
 * "none", pol_vertex = "ladder"). After two damped iterations the W-bar cache is filled by hand
 * (update_w publishes dW; cache_w folds it -- the static rung mode never fills the cache), then
 * vertex_t::dynbse_gate runs:
 *   (A0) the THC rung operator vs the explicit Kbig                      -- machine class
 *   (A)  the static limit vs the sign-corrected L2 resolvent -ladder(-W0) -- 1e-12 class
 *   (B)  the one bare dynamic rung vs pi_c_accumulate_w(Z = 0, W_dyn - W_dyn(0)) -- fit class
 *   (C)  GMRES vs Neumann on the resummed vertex; Hermiticity; the watchdog and meters.
 */

#undef NDEBUG

#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <cmath>
#include <complex>
#include <string>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"
#include "IO/app_loggers.h"

#include "numerics/imag_axes_ft/IAFT.hpp"

#include "methods/vertex/vertex_pi.icc"

#include "mean_field/default_MF.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include "methods/vertex/vertex_t.h"
#include "methods/embedding/projector_t.h"
#include <filesystem>
#include <fstream>
#include <map>
#include <array>
#include <cmath>
#include "nda/blas.hpp"
#include "h5/h5.hpp"
#include "nda/h5.hpp"
#include "nda/linalg/eigenelements.hpp"
#include "methods/scr_coulomb/cvv_head.hpp"
#include "methods/HF/thc_exchange_kernel.hpp"
#include "utilities/proc_grid_partition.hpp"

namespace bdft_tests {

  using namespace methods;

  // proj_mat convention (test_vertex_wannier.cpp): |w_a> = sum_i V_{i,a} |psi_{W0+i}>, C_{a,i} = conj(V_{i,a});
  // V = nullptr is the degenerate identity projector (window physics in the band gauge).
  // trs_images: on a symmetric mesh give the time-reversal images (the last nkpts_trev_pairs k) U(-k) = conj(U(k)),
  // the relation real MLWFs satisfy (a k-independent complex V is NOT time-reversal consistent).
  inline projector_t make_degenerate_projector(mf::MF &mf, long W0, long M,
                                               nda::array<std::complex<double>, 2> const *V = nullptr,
                                               bool trs_images = false) {
    using cplx = std::complex<double>;
    const long nk = mf.nkpts(), ns = mf.nspin(), ntrev = mf.nkpts_trev_pairs();
    nda::array<cplx, 5> C_ksIai(nk, ns, 1, M, M); C_ksIai() = cplx(0.0);
    // trs_pairs: a fully time-reversal-consistent complex gauge, U(-k) = conj(U(k)) for EVERY (k, -k) pair of the
    // mesh and U real at the TRIM points (the relation real MLWFs satisfy; a k-independent complex V cannot).
    const bool trs_pairs = std::getenv("COQUI_DYNBSE_TEST_WINT_V") and std::string(std::getenv("COQUI_DYNBSE_TEST_WINT_V")) == "trs2";
    nda::array<double, 2> kcr(mf.kpts_crystal());
    auto minus_k = [&](long ik) {   // index of -k (mod 1), -1 if absent
      for (long j = 0; j < nk; ++j) {
        bool same = true;
        for (int d = 0; d < 3 and same; ++d) { const double x = kcr(j, d) + kcr(ik, d); same = std::abs(x - std::round(x)) < 1e-6; }
        if (same) return j;
      }
      return -1L;
    };
    for (long ik = 0; ik < nk; ++ik) for (long is = 0; is < ns; ++is)
      for (long a = 0; a < M; ++a)
        for (long i = 0; i < M; ++i) {
          cplx u = V ? (*V)(i, a) : cplx(a == i ? 1.0 : 0.0);
          if (trs_images and ik >= nk - ntrev) u = std::conj(u);
          if (trs_pairs and V) {
            const long jk = minus_k(ik);
            if (jk == ik) u = cplx(u.real(), 0.0);          // TRIM point: real part of V (re-orthonormalized by Loewdin at load)
            else if (jk >= 0 and jk < ik) u = std::conj(u);   // the second member of the pair gets conj(V)
          }
          C_ksIai(ik, is, 0, a, i) = std::conj(u);
        }
    nda::array<long, 3> bw(1, 1, 2); bw(0, 0, 0) = W0 + 1; bw(0, 0, 1) = W0 + M;
    auto kc = nda::make_regular(mf.kpts_crystal());
    return projector_t(mf, C_ksIai, bw, kc, false, false);
  }

  TEST_CASE("dynbse_gate", "[methods][vertex][scgwt][dynbse]") {
#ifndef ENABLE_DLR
    SUCCEED("dynbse_gate skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis,
                          std::getenv("COQUI_DYNBSE_TEST_PREC") ? std::string(std::getenv("COQUI_DYNBSE_TEST_PREC")) : std::string("low"));
    std::string output = "coqui_d2_gates";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    auto gate_at = [&](nda::range window, int niter, bool quick) {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd());
      vtx.set_pol_vertex("ladder", "w0_prev", window, -1, 1e-8, -1.0, -1.0, -1.0);
      if (auto *rm = std::getenv("COQUI_DYNBSE_TEST_RESOLVENT")) vtx.set_ladder_dyn_resolvent(rm);   // P7: inverse | lu
      if (auto *ac = std::getenv("COQUI_DYNBSE_TEST_SIGDYN_ACC")) vtx.set_sigma_dyn_acc(ac);   // P4-C14: split | single
      if (auto *wc = std::getenv("COQUI_DYNBSE_TEST_WCACHE")) vtx.set_wcache_mode(wc);   // P19: replicated | shared
      scr_eri.set_vertex(&vtx);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     niter, false, 1e-9, true);
      (void)e_hf; (void)e_corr;
      auto *pv = scr_eri.pol_vertex_instance();
      REQUIRE(pv != nullptr);
      // the W-bar cache: the static-rung mode never fills it; publish dW once more and fold it
      scr_eri.update_w(mb_state, thc, -1);
      REQUIRE(mb_state.dW_qtPQ.has_value());
      pv->cache_w(mb_state, thc);
      REQUIRE(pv->has_cached_w());
      auto g = pv->dynbse_gate(mb_state, thc, quick);
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return g;
    };

    if (std::getenv("COQUI_DYNBSE_QUICK") != nullptr) {
      // the quick leg (A0 / A0-FT / A0' only): the mesh-Fourier k-sum IS the direct rung operator
      auto gq = gate_at(nda::range(0, 4), 1, true);
      app_log(1, "dynbse_gate quick: A0 {:.3e}; A0-FT {:.3e} (mesh check {:.2e})", gq.a0_resid, gq.a0_ft_resid, gq.a0_ft_check);
      REQUIRE(gq.a0_resid < 1e-12);
      REQUIRE(gq.a0_ft_check >= 0.0);
      REQUIRE(gq.a0_ft_check < 1e-10);
      REQUIRE(gq.a0_ft_resid >= 0.0);
      REQUIRE(gq.a0_ft_resid < 1e-11);
      return;
    }
    auto g = gate_at(nda::range(0, 4), 2, false);
    app_log(1, "dynbse_gate [C = [0,4) of {}]: A0 {:.3e}; A {:.3e} (vs as-implemented L2 {:.3e}); B {:.3e} "
               "(|anchor| {:.3e}, |static| {:.3e}); GMRES vs Neumann {:.3e}, Gamma1 {:.3e}; Ritz {:.3f} / "
               "contraction {:.3f}; applications {} / {}; converged {}; refit {:.3e}; herm {:.3e}; dyn vs static "
               "{:.3e}, Gamma1 vs static {:.3e}; G fit {:.3e} rr {:.3g}; Dsq {:.3e}; W(s) sym {:.3e}",
            mf->nbnd(), g.a0_resid, g.a_resid, g.a_l2_diff, g.b_resid, g.onerung_max, g.static_max,
            g.gmres_vs_neumann, g.gam1_consistency, g.ritz_max, g.contraction_max, g.it_max, g.it_max_neumann,
            g.all_converged, g.refit_err, g.herm, g.dyn_vs_static, g.gam1_vs_static, g.fit_err_G, g.rr_G,
            g.dsq_err, g.wtau_sym);
    REQUIRE(g.a0_resid >= 0.0);
    REQUIRE(g.a0_resid < 1e-12);                  // the THC rung operator IS Kbig
    REQUIRE(g.a0_ft_resid >= 0.0);
    REQUIRE(g.a0_ft_resid < 1e-11);               // (A0-FT) the mesh-Fourier k-sum IS the direct rung operator
    REQUIRE(g.a_resid < 1e-10);                   // the static limit IS the (sign-corrected) L2 resolvent
    REQUIRE(g.b_resid < 1e-3);                    // one dynamic rung = the anchor at inu = 0 (fit class)
    REQUIRE(g.b_continuity < 1e-2);               // and continuous into the first positive node
    // the resummed solve at inu = 0: both solvers agree, converged, contractive
    REQUIRE(g.gmres_vs_neumann < 1e-6);
    REQUIRE(g.block_resid >= 0.0);
    REQUIRE(g.block_resid < 1e-6);                // (C'') RHS column blocking = the unblocked solve (tol class)
    REQUIRE(g.nu1_gfit >= 0.0);
    REQUIRE(g.nu1_gfit < 1e-4);                   // (C') the union grid without a mask represents G
    REQUIRE(g.nu1_done);                          // and the first positive node converges (GMRES(12), readout stop)
    REQUIRE(std::abs(g.nu1_resid - g.dyn_vs_static) < 0.05 * g.dyn_vs_static);   // nu-continuity of the resummed correction
    REQUIRE(g.gam1_consistency < 1e-12);
    REQUIRE(g.all_converged);
    REQUIRE(g.ritz_max < 1.0);
    REQUIRE(g.dyn_vs_static > 0.0);
    REQUIRE(std::isfinite(g.dyn_max));
#endif
  }

  TEST_CASE("dynbse_readout", "[methods][vertex][scgwt][dynbse]") {
#ifndef ENABLE_DLR
    SUCCEED("dynbse_readout skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis,
                          std::getenv("COQUI_DYNBSE_TEST_PREC") ? std::string(std::getenv("COQUI_DYNBSE_TEST_PREC")) : std::string("low"));
    std::string output = "coqui_d3_readout";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    // W-int-0 empirical gate: the full Wannier-vertex DUMP chain (pol-vertex "ladder" + wannier ->
    // set_wannier_projector -> scr_coulomb adopt_wannier + dump path -> dynbse E-leg -> Pi_loc dumped).
    if (std::getenv("COQUI_DYNBSE_TEST_WAN")) {
      // W-int-0/1 empirical gate: the full Wannier-vertex DUMP chain (pol-vertex "ladder" + wannier ->
      // set_wannier_projector -> scr_coulomb adopt_wannier + dump path -> ladder_inputs G_bar + the rotated
      // X_bar -> identity pair legs -> Pi_loc dumped), run for the SAME C (window [0,4)) in two gauges: the
      // degenerate identity projector and a fixed complex unitary mix V of the window bands. Pi_loc(q) then
      // differs by the pair-frame rotation only, so its gauge INVARIANTS (tr H, tr H^2, tr H^3 of the
      // Hermitized block, per q) must agree -- the W-int-1 frame-consistency oracle (W-int-0 mixed a
      // band-frame G with the MLWF-frame X_bar and applied U twice; invisible for V = 1).
      using cplx = std::complex<double>;
      auto run_wan = [&](std::string const &tag, nda::array<cplx, 2> const *V) {
        const std::string out = "coqui_d3_wan_" + tag;
        solvers::hf_t hf; solvers::gw_t gw(&ft, "ignore_g0", out);
        solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
        simple_dyson dyson(mf.get(), &ft); MBState mb_state(mpi_context, ft, out);
        iter_scf::iter_scf_t iter_sol("damping");
        solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd());
        vtx.set_pol_vertex("ladder", "w0_prev", nda::range(0, 4), -1, 1e-8, -1.0, -1.0, -1.0);
        vtx.set_ladder_rung("dynamic", 1e-8, 30, 12, -1.0);
        vtx.set_ladder_dyn_gamma1_only(true); vtx.set_ladder_dyn_dump(true);
        auto proj = make_degenerate_projector(*mf, 0, 4, V); vtx.set_wannier_projector(proj, true);
        REQUIRE(vtx.wannier()); REQUIRE(vtx.subspace_rank() == 4); REQUIRE(vtx.isometry_defect() < 1e-10);
        if (auto *rm = std::getenv("COQUI_DYNBSE_TEST_RESOLVENT")) vtx.set_ladder_dyn_resolvent(rm);   // P7: inverse | lu
        if (auto *ac = std::getenv("COQUI_DYNBSE_TEST_SIGDYN_ACC")) vtx.set_sigma_dyn_acc(ac);   // P4-C14: split | single
        if (auto *wc = std::getenv("COQUI_DYNBSE_TEST_WCACHE")) vtx.set_wcache_mode(wc);   // P19: replicated | shared
        scr_eri.set_vertex(&vtx);
        auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                       solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 2, false, 1e-9, true);
        app_log(1, "dynbse_readout WANNIER dump smoke [{}]: e_hf {:.8f}, e_corr {:.8f} (Pi_loc dumped, nab = 16)",
                tag, e_hf, e_corr);
        REQUIRE(std::isfinite(e_hf)); REQUIRE(std::isfinite(e_corr));
        mpi_context->comm.barrier();
        // read back the dumped Gamma_1 blocks: record = 5 longs {is, iq, m, nout, 4} + 4 nout^2 complex
        std::map<std::pair<long, long>, nda::array<cplx, 2>> blocks;
        for (auto const &e : std::filesystem::directory_iterator(".")) {
          const std::string fn = e.path().filename().string();
          if (fn.rfind(out + ".dynunits.nu0.g", 0) != 0 or fn.substr(fn.size() - 4) != ".bin") continue;
          std::ifstream in(e.path(), std::ios::binary);
          long hdr[5];
          while (in.read(reinterpret_cast<char *>(hdr), sizeof(hdr))) {
            const long n = hdr[3];
            std::vector<cplx> blk(size_t(4 * n * n));
            if (not in.read(reinterpret_cast<char *>(blk.data()), std::streamsize(blk.size() * sizeof(cplx)))) break;
            nda::array<cplx, 2> P(n, n);
            for (long a = 0; a < n; ++a)
              for (long b = 0; b < n; ++b) P(a, b) = blk[size_t((2 * n + a) * n + b)];   // column 2 = gam1
            blocks[{hdr[0], hdr[1]}] = P;
          }
        }
        mpi_context->comm.barrier();
        if (mpi_context->comm.root()) {
          remove((out + ".mbpt.h5").c_str());
          for (auto const &e : std::filesystem::directory_iterator("."))
            if (e.path().filename().string().rfind(out + ".dynunits.", 0) == 0) std::filesystem::remove(e.path());
        }
        mpi_context->comm.barrier();
        return std::make_tuple(e_hf, e_corr, blocks);
      };
      // a fixed complex unitary on the 4 window bands: a product of complex Givens rotations
      nda::array<cplx, 2> V(4, 4); V() = cplx(0.0);
      for (long a = 0; a < 4; ++a) V(a, a) = cplx(1.0);
      auto givens = [&](long p, long q, double th, double ph) {   // V <- V . G(p, q; th, ph), G unitary
        const double c = std::cos(th), s = std::sin(th); const cplx eph(std::cos(ph), std::sin(ph));
        for (long i = 0; i < 4; ++i) {
          const cplx vp = V(i, p), vq = V(i, q);
          V(i, p) = vp * c - vq * s * std::conj(eph);
          V(i, q) = vp * s * eph + vq * c;
        }
      };
      givens(0, 1, 0.7, 0.3); givens(1, 2, 1.1, -0.8); givens(2, 3, 0.4, 1.9); givens(0, 3, 0.9, 0.5);
      auto [h_id, c_id, b_id] = run_wan("id", nullptr);
      auto [h_V, c_V, b_V] = run_wan("V", &V);
      // the SAME C: the loop energies do not depend on the gauge (the pol-vertex readout is not fed back)
      app_log(1, "dynbse_readout WANNIER gauge: |D e_hf| = {:.2e}, |D e_corr| = {:.2e}, {} / {} Pi_loc blocks",
              std::abs(h_id - h_V), std::abs(c_id - c_V), b_id.size(), b_V.size());
      REQUIRE(std::abs(h_id - h_V) < 1e-8); REQUIRE(std::abs(c_id - c_V) < 1e-8);
      REQUIRE(b_id.size() == b_V.size()); REQUIRE(b_id.size() > 0);
      double worst = 0.0, fro_worst = 0.0;
      for (auto const &[key, P1] : b_id) {
        REQUIRE(b_V.count(key) == 1);
        auto const &P2 = b_V.at(key);
        const long n = P1.shape(0);
        auto invariants = [&](nda::array<cplx, 2> const &P) {
          nda::array<cplx, 2> H(n, n), H2(n, n), H3(n, n);
          for (long a = 0; a < n; ++a) for (long b = 0; b < n; ++b) H(a, b) = 0.5 * (P(a, b) + std::conj(P(b, a)));
          nda::blas::gemm(H, H, H2); nda::blas::gemm(H2, H, H3);
          double t1 = 0.0, t2 = 0.0, t3 = 0.0, f = 0.0;
          for (long a = 0; a < n; ++a) { t1 += H(a, a).real(); t2 += H2(a, a).real(); t3 += H3(a, a).real(); }
          for (long a = 0; a < n; ++a) for (long b = 0; b < n; ++b) f += std::norm(P(a, b));
          return std::array<double, 4>{t1, t2, t3, std::sqrt(f)};
        };
        auto i1 = invariants(P1), i2 = invariants(P2);
        for (int c = 0; c < 3; ++c) worst = std::max(worst, std::abs(i1[c] - i2[c]) / std::max(std::abs(i1[c]), 1e-300));
        fro_worst = std::max(fro_worst, std::abs(i1[3] - i2[3]) / i1[3]);
      }
      app_log(1, "dynbse_readout WANNIER gauge oracle: max rel mismatch of (tr H, tr H^2, tr H^3) over (s, q) = {:.3e}, "
                 "of |Pi_loc|_F = {:.3e}  (gauge-invariant; the identity-vs-V frame test)", worst, fro_worst);
      REQUIRE(worst < 1e-6); REQUIRE(fro_worst < 1e-6);
      return;
    }

    // W-int-1b/4 gate as a lambda over the mesh: nosym (qe_lih222) and, with COQUI_DYNBSE_TEST_WINT_SYM, the SYMMETRIC
    // qe_lih222_sym fixture (W-int-4s: the frozen-point gather builds the image k-points from the IBZ orbitals).
    auto wint_gate = [&](std::shared_ptr<mf::MF> mfw, auto &eriw, std::string const &mesh_tag) {
      // W-int-1b/4 gate (notes/wannier_coarse_vertex_plan.md): the frozen-point aux frame + the consumer.
      //  A  window, dynamic Gamma_1, dumps <A>.secpts.h5 + <A>.pol_nu0.g2.h5
      //  B  window, the points FROZEN from A          -> the dumped Pi(q)_{MN} == A's, eps readout == A's
      //  V  a unitary MLWF mix V of the same window, frame "aux", points frozen from A -> Pi == A's: the point
      //     frame is GAUGE-INVARIANT (the C-space alone defines it)
      //  C  window, static rung, points frozen from A, the eps readout CONSUMES A's dumped static column
      //     (pol_vertex_interp_file) -> eps_M(ladder) == A's (the consumer V0: coarse = fine)
      using cplx = std::complex<double>;
      struct res_t { double e_corr, er, el; nda::array<cplx, 3> Ps, Pg; };
      bool trs_images = false;   // set with the V flavour below (captured by reference)
      bool stream_cur = false;   // P6: the streaming THC rung (per symmetry class on the symmetric mesh) instead of the dense one
      auto run_w = [&](std::string const &tag, std::string const &rung, std::string const &points, std::string const &interp,
                       nda::array<cplx, 2> const *V) {
        const std::string out = "coqui_d3_wint_" + mesh_tag + "_" + tag;
        solvers::hf_t hf; solvers::gw_t gw(&ft, "ignore_g0", out);
        solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
        simple_dyson dyson(mfw.get(), &ft); MBState mb_state(mpi_context, ft, out);
        iter_scf::iter_scf_t iter_sol("damping");
        solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mfw->nbnd());
        vtx.set_pol_vertex("ladder", "w0_prev", nda::range(0, 4), -1, 1e-8, -1.0, -1.0, -1.0);
        vtx.set_ladder_rung(rung, 1e-8, 30, 12, -1.0);
        vtx.set_ladder_dyn_gamma1_only(true); vtx.set_ladder_dyn_dump(rung == "dynamic");
        vtx.set_isdf_points(points, points.empty()); vtx.set_wannier_frame("aux");
        vtx.set_pol_interp(interp, "static");
        vtx.set_ladder_dyn_dense(not stream_cur);
        if (V) { auto proj = make_degenerate_projector(*mfw, 0, 4, V, trs_images); vtx.set_wannier_projector(proj, true); }
        if (auto *rm = std::getenv("COQUI_DYNBSE_TEST_RESOLVENT")) vtx.set_ladder_dyn_resolvent(rm);   // P7: inverse | lu
        if (auto *ac = std::getenv("COQUI_DYNBSE_TEST_SIGDYN_ACC")) vtx.set_sigma_dyn_acc(ac);   // P4-C14: split | single
        if (auto *wc = std::getenv("COQUI_DYNBSE_TEST_WCACHE")) vtx.set_wcache_mode(wc);   // P19: replicated | shared
        scr_eri.set_vertex(&vtx);
        auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eriw, ft,
                                       solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 2, false, 1e-9, true);
        auto [er, el] = scr_eri.pol_eps_readout();
        res_t r{e_corr, er, el, {}, {}};
        mpi_context->comm.barrier();
        if (rung == "dynamic") {
          h5::file f(out + ".pol_nu0.g2.h5", 'r'); h5::group g(f);
          nda::h5_read(g, "Pi_static", r.Ps); nda::h5_read(g, "Pi_gam1", r.Pg);
        }
        app_log(1, "dynbse_readout W-int gate {} [{}]: e_corr {:.10f}, eps_M RPA {:.8f} ladder {:.8f}{}", mesh_tag, tag, e_corr, er, el,
                rung == "dynamic" ? " (Pi dumped)" : "");
        mpi_context->comm.barrier();
        if (mpi_context->comm.root() and tag != "A") {
          remove((out + ".mbpt.h5").c_str());
          for (auto const &e : std::filesystem::directory_iterator("."))
            if (e.path().filename().string().rfind(out + ".", 0) == 0) std::filesystem::remove(e.path());
        }
        mpi_context->comm.barrier();
        return r;
      };
      auto relmax = [](nda::array<cplx, 3> const &A, nda::array<cplx, 3> const &B) {
        double d = 0.0, n = 0.0;
        for (long i = 0; i < A.size(); ++i) { d = std::max(d, std::abs(A.data()[i] - B.data()[i])); n = std::max(n, std::abs(B.data()[i])); }
        return d / n;
      };
      nda::array<cplx, 2> V(4, 4); V() = cplx(0.0);
      for (long a = 0; a < 4; ++a) V(a, a) = cplx(1.0);
      auto givens = [&](long p, long q, double th, double ph) {
        const double c = std::cos(th), s = std::sin(th); const cplx eph(std::cos(ph), std::sin(ph));
        for (long i = 0; i < 4; ++i) { const cplx vp = V(i, p), vq = V(i, q); V(i, p) = vp * c - vq * s * std::conj(eph); V(i, q) = vp * s * eph + vq * c; }
      };
      const std::string vflavour = std::getenv("COQUI_DYNBSE_TEST_WINT_V") ? std::getenv("COQUI_DYNBSE_TEST_WINT_V") : "complex";
      if (vflavour == "real") { givens(0, 1, 0.7, 0.0); givens(1, 2, 1.1, 0.0); givens(2, 3, 0.4, 0.0); givens(0, 3, 0.9, 0.0); }
      else { givens(0, 1, 0.7, 0.3); givens(1, 2, 1.1, -0.8); givens(2, 3, 0.4, 1.9); givens(0, 3, 0.9, 0.5); }
      trs_images = (vflavour == "trs");   // "trs2" = the fully TRS-consistent k-dependent gauge (handled in make_degenerate_projector)
      app_log(1, "dynbse_readout W-int gate ({}): Wannier V flavour = {} (real | trs = conj(V) on the trev images | complex)", mesh_tag, vflavour);
      const std::string pts = "coqui_d3_wint_" + mesh_tag + "_A.secpts.h5", nu0 = "coqui_d3_wint_" + mesh_tag + "_A.pol_nu0.g2.h5";
      auto A = run_w("A", "dynamic", "", "", nullptr);
      REQUIRE(std::filesystem::exists(pts)); REQUIRE(std::filesystem::exists(nu0));
      auto B = run_w("B", "dynamic", pts, "", nullptr);
      {   // P6: the streaming THC rung (on the symmetric mesh: one pass per symmetry class of the transfers, legs from Xhat,
          // the IBZ-stored W at q_star, transposed on time-reversal transfers) is the same operator as the dense per-tau rung:
          // the static column and the one-rung Gamma_1 column agree to rounding (frozen points, the same aux frame as A)
        stream_cur = true;
        auto AS = run_w("AS", "dynamic", pts, "", nullptr);
        stream_cur = false;
        const double dS = relmax(AS.Pg, A.Pg), dSs = relmax(AS.Ps, A.Ps);
        app_log(1, "dynbse_readout W-int gate ({}): P6 streaming THC rung vs the dense per-tau rung: |dPi_gam1| {:.2e} |dPi_static| {:.2e}; "
                   "e_corr {:.10f} vs {:.10f}", mesh_tag, dS, dSs, AS.e_corr, A.e_corr);
        REQUIRE(dS < 1e-9); REQUIRE(dSs < 1e-9);
      }
      auto W = run_w("V", "dynamic", pts, "", &V);
      auto C = run_w("C", "static", pts, nu0, nullptr);
      //  CW the PRODUCTION consumer: Wannier mode (a unitary mix V of the same window, X_bar = X U on the frozen
      //     points), static rung, consuming A's (window-run) static column -- the point frame is gauge-invariant,
      //     so eps_M(ladder) == A's again
      auto CW = run_w("CW", "static", pts, nu0, &V);
      const double dB = relmax(B.Pg, A.Pg), dW = relmax(W.Pg, A.Pg), dBs = relmax(B.Ps, A.Ps), dWs = relmax(W.Ps, A.Ps);
      app_log(1, "dynbse_readout W-int gate ({}): FROZEN points reproduce the selection: |dPi_gam1| {:.2e} |dPi_static| {:.2e}; "
                 "the point frame is gauge-invariant (unitary V, aux frame): {:.2e} / {:.2e}; eps_M(ladder) A {:.10f} B {:.10f} "
                 "CONSUMER window {:.10f} (|d| = {:.2e}) Wannier {:.10f} (|d| = {:.2e}); e_corr A-B {:.1e} A-V {:.1e} A-C {:.1e} A-CW {:.1e} "
                 "(a Wannier-mode DYNAMIC run dumps and returns before the eps readout: V.el = {:.1f} by design)",
              mesh_tag, dB, dBs, dW, dWs, A.el, B.el, C.el, std::abs(C.el - A.el), CW.el, std::abs(CW.el - A.el),
              std::abs(A.e_corr - B.e_corr), std::abs(A.e_corr - W.e_corr), std::abs(A.e_corr - C.e_corr),
              std::abs(A.e_corr - CW.e_corr), W.el);
      REQUIRE(dB < 1e-10); REQUIRE(dBs < 1e-10);
      REQUIRE(dW < 1e-8); REQUIRE(dWs < 1e-8);   // W-int-4w fixed (conj(U) on the conjugated rotations of build_sym_ctx): holds with trev images too
      REQUIRE(std::abs(B.el - A.el) < 1e-9);
      REQUIRE(std::abs(C.el - A.el) < 1e-8); REQUIRE(std::abs(C.er - A.er) < 1e-10);
      REQUIRE(std::abs(CW.el - A.el) < 1e-8); REQUIRE(std::abs(CW.er - A.er) < 1e-10);
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) {
        remove(("coqui_d3_wint_" + mesh_tag + "_A.mbpt.h5").c_str());
        for (auto const &e : std::filesystem::directory_iterator("."))
          if (e.path().filename().string().rfind("coqui_d3_wint_" + mesh_tag + "_A.", 0) == 0) std::filesystem::remove(e.path());
      }
      mpi_context->comm.barrier();
    };
    if (std::getenv("COQUI_DYNBSE_TEST_WINT_CROSS")) {
      // STAR-CLOSURE probe (kp888 finding 2026-09-16): points selected on the NOSYM mesh, frozen on the SYMMETRIC mesh.
      // The vertex's IBZ machinery assumes the secondary point set is closed under the group (the sym selection
      // builds it from irreducible r-grid sectors). Runs: A_nosym (window, dynamic, dumps points + Pi), A_sym (own
      // star-closed selection: the direct sym reference), C_sym (A_nosym's points frozen on the sym mesh, consuming
      // A_nosym's static column). Reported: eps_M(ladder, q_min) of the three -- C_sym vs A_sym beyond the ~1e-4
      // sym/nosym leakage class means the non-star-closed frozen points break the symmetric consumer.
      using cplx = std::complex<double>;
      // env value = the NOSYM fixture ("1" = qe_lih222); its symmetric partner is <fixture>_sym
      std::string fxn = std::getenv("COQUI_DYNBSE_TEST_WINT_CROSS"); if (fxn == "1" or fxn.empty()) fxn = "qe_lih222";
      auto mfn = std::make_shared<mf::MF>(mf::default_MF(mpi_context, fxn));
      thc_reader_t thcn(mfn, make_thc_reader_ptree(mfn->nbnd() * 8, "", "incore", "", "bdft", 1e-10, mfn->ecutrho(), 1, 1024));
      auto erin = mb_eri_t(thcn, thcn);
      auto mfs = std::make_shared<mf::MF>(mf::default_MF(mpi_context, fxn + "_sym"));
      thc_reader_t thcs(mfs, make_thc_reader_ptree(mfs->nbnd() * 8, "", "incore", "", "bdft", 1e-10, mfs->ecutrho(), 1, 1024));
      auto eris = mb_eri_t(thcs, thcs);
      app_log(1, "dynbse_readout W-int CROSS on {} (nk {}) vs {}_sym (nk {}, IBZ {}, trev pairs {})", fxn, mfn->nkpts(), fxn,
              mfs->nkpts(), mfs->nkpts_ibz(), mfs->nkpts_trev_pairs());
      auto run_x = [&](std::string const &tag, std::shared_ptr<mf::MF> mfw, auto &eriw, std::string const &rung, bool dump,
                       std::string const &points, std::string const &interp) {
        const std::string out = "coqui_d3_wcross_" + tag;
        solvers::hf_t hf; solvers::gw_t gw(&ft, "ignore_g0", out);
        solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
        simple_dyson dyson(mfw.get(), &ft); MBState mb_state(mpi_context, ft, out);
        iter_scf::iter_scf_t iter_sol("damping");
        solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mfw->nbnd());
        vtx.set_pol_vertex("ladder", "w0_prev", nda::range(0, 4), -1, 1e-8, -1.0, -1.0, -1.0);
        vtx.set_ladder_rung(rung, 1e-8, 30, 12, -1.0);
        vtx.set_ladder_dyn_gamma1_only(true); vtx.set_ladder_dyn_dump(dump);
        vtx.set_isdf_points(points, dump); vtx.set_pol_interp(interp, "static");
        if (auto *rm = std::getenv("COQUI_DYNBSE_TEST_RESOLVENT")) vtx.set_ladder_dyn_resolvent(rm);   // P7: inverse | lu
        if (auto *ac = std::getenv("COQUI_DYNBSE_TEST_SIGDYN_ACC")) vtx.set_sigma_dyn_acc(ac);   // P4-C14: split | single
        if (auto *wc = std::getenv("COQUI_DYNBSE_TEST_WCACHE")) vtx.set_wcache_mode(wc);   // P19: replicated | shared
        scr_eri.set_vertex(&vtx);
        auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eriw, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 2, false, 1e-9, true);
        auto [er, el] = scr_eri.pol_eps_readout();
        app_log(1, "dynbse_readout W-int CROSS [{}]: e_corr {:.10f}, eps_M RPA {:.8f} ladder {:.8f}", tag, e_corr, er, el);
        mpi_context->comm.barrier();
        if (mpi_context->comm.root() and tag != "A_nosym") {
          remove((out + ".mbpt.h5").c_str());
          for (auto const &e : std::filesystem::directory_iterator("."))
            if (e.path().filename().string().rfind(out + ".", 0) == 0) std::filesystem::remove(e.path());
        }
        mpi_context->comm.barrier();
        return std::make_pair(er, el);
      };
      auto [ra, la] = run_x("A_nosym", mfn, erin, "dynamic", true, "", "");
      auto [rs, ls] = run_x("A_sym", mfs, eris, "static", false, "", "");
      auto [rc, lc] = run_x("C_sym", mfs, eris, "static", false, "coqui_d3_wcross_A_nosym.secpts.h5", "coqui_d3_wcross_A_nosym.pol_nu0.g2.h5");
      app_log(1, "dynbse_readout W-int CROSS: eps_M(ladder, q_min): nosym direct {:.8f} | sym direct (star-closed points) {:.8f} | "
                 "sym consumer on NOSYM-selected frozen points {:.8f}  => C_sym - A_sym = {:+.2e} (rel {:.1e}); RPA sym {:.8f} nosym {:.8f}",
              la, ls, lc, lc - ls, std::abs(lc - ls) / std::abs(ls - rs), rs, ra);
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) {
        remove("coqui_d3_wcross_A_nosym.mbpt.h5");
        for (auto const &e : std::filesystem::directory_iterator("."))
          if (e.path().filename().string().rfind("coqui_d3_wcross_A_nosym.", 0) == 0) std::filesystem::remove(e.path());
      }
      mpi_context->comm.barrier();
      return;
    }
    if (std::getenv("COQUI_DYNBSE_TEST_WINT_INJ")) {
      // W-int-4f V0 gate (notes/wannier_coarse_vertex_plan.md): the FULL-FREQUENCY W-Dyson feed. A injects the direct
      // resummed ladder at every PH-sym half node (pol_vertex_inject = ladder_n2, the L3 object) and dumps it
      // (<A>.pol_wh.g1.h5 + <A>.secpts.h5); B runs the same loop with the injection READ from that file on A's frozen
      // points (never solving the ladder). Same G -> same P_RPA + P^{C,L} -> same W, Sigma: e_hf / e_corr / the loop-side
      // eps_M / the readout must be IDENTICAL (bit level). C = the same with a fixture of the SYMMETRIC mesh's own dump.
      std::string fxn = std::getenv("COQUI_DYNBSE_TEST_WINT_INJ"); if (fxn == "1" or fxn.empty()) fxn = "qe_lih222";
      auto mfn = std::make_shared<mf::MF>(mf::default_MF(mpi_context, fxn));
      thc_reader_t thcn(mfn, make_thc_reader_ptree(mfn->nbnd() * 8, "", "incore", "", "bdft", 1e-10, mfn->ecutrho(), 1, 1024));
      auto erin = mb_eri_t(thcn, thcn);
      auto section_m = [&]() {
      {   // LFF-aux L-6 (Route 2) gate, section M: the PAIR-RESOLVED static-ladder vertex in Sigma (vertex_sigma_pair.icc).
            // R0: plain GW; XS: B-S Sigma^{C,x} (vertex_type 2nd_exchange, rung static, secondary frame at run G's frozen
            // points, bl_drop 1 drops Sigma^{C,r}); P1: pair col static1 + outer static = the SAME diagram through the pair
            // machinery (the exact identity, G2); P0: pair at scale 0 (bit-identical to R0); PS: col static + outer dynamic
            // (the production object; G1: its Pi-check == the all-nu dump's Pi_static column at the half nodes); PH: PS at
            // scale 1/2 (linearity).
          using S5 = nda::array<std::complex<double>, 5>;
          auto read_sig = [&](std::string const &fn, S5 &S) {
            h5::file f(fn, 'r'); h5::group g(f);
            auto it = g.open_group("scf").open_group("iter1");
            nda::h5_read(it, "Sigma_tskij", S);
          };
          long m0b = -1;
          { auto wb = ft.wn_mesh_b(); for (long l = 0; l < wb.shape(0); ++l) if (wb(l) == 0) m0b = l; }
          REQUIRE(m0b >= 0);
          std::string side_cur = "right";   // the junction of the next kind-2 run (section N sets it)
          bool sd_dump = false; std::vector<long> sd_nodes; std::string sd_fit; long sd_rank = 0;   // L-8: the next kind-2 run's sampled-mode knobs
          bool sd_ckpt = false;   // P20: the next kind-2 run checkpoints its Sigma accumulators after every unit (and keeps the files)
          long sd_auto = 0;       // P14b: the next kind-2 run chooses this many sampled nodes from the dump (in place of sd_nodes)
          std::string sd_acc = std::getenv("COQUI_DYNBSE_TEST_SIGDYN_ACC") ? std::getenv("COQUI_DYNBSE_TEST_SIGDYN_ACC") : "split";   // P4-C14: split | single
          auto run_p = [&](std::string const &tag, int kind, std::string const &col, std::string const &outer, double scale, S5 &Sig) {
            // kind: 0 = plain GW, 1 = B-S Sigma^{C,x} (static rung), 2 = the pair vertex, 3 = the DYNAMIC-rung B-S Sigma^C (G^3 W^2)
            const std::string out = "coqui_d3_winj_" + tag;
            solvers::hf_t hf; solvers::gw_t gw(&ft, "ignore_g0", out);
            solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
            simple_dyson dyson(mfn.get(), &ft); MBState mb_state(mpi_context, ft, out);
            iter_scf::iter_scf_t iter_sol("damping");
            std::array<double, 4> m{0.0, 0.0, 0.0, 0.0}; std::array<double, 2> d{0.0, 0.0};
            double e_corr = 0.0, pichk = -1.0;
            if (kind == 1 or kind == 3) {
              solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(0, 4), mfn->nbnd(), "ignore_g0", "secondary", -1, 1e-8, -1.0, -1.0,
                                    kind == 1 ? "static" : "dynamic");
              vtx.set_isdf_points("coqui_d3_winj_G.secpts.h5", false);
              vtx.set_bl_drop(1);
              if (kind == 3) vtx.set_skip_pi_c(true);   // the G^3 W^2 cut alone on the RPA W: the reference for the one-bare-rung pair column (N2)
              if (auto *rm = std::getenv("COQUI_DYNBSE_TEST_RESOLVENT")) vtx.set_ladder_dyn_resolvent(rm);   // P7: inverse | lu
              if (auto *ac = std::getenv("COQUI_DYNBSE_TEST_SIGDYN_ACC")) vtx.set_sigma_dyn_acc(ac);   // P4-C14: split | single
              if (auto *wc = std::getenv("COQUI_DYNBSE_TEST_WCACHE")) vtx.set_wcache_mode(wc);   // P19: replicated | shared
              scr_eri.set_vertex(&vtx); gw.set_vertex(&vtx);
              e_corr = std::get<1>(scf_loop(mb_state, dyson, erin, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 1, false, 1e-9, true));
            } else if (kind == 2) {
              solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mfn->nbnd());
              vtx.set_pol_vertex("ladder", "w0_prev", nda::range(0, 4), -1, 1e-8, -1.0, -1.0, -1.0, "none");
              vtx.set_ladder_rung("static", 1e-8, 30, 12, -1.0);
              vtx.set_isdf_points("coqui_d3_winj_G.secpts.h5", false);
              vtx.set_sigma_pair(true, col, outer, scale, true, true, side_cur);
              vtx.set_sigma_dyn(sd_dump, sd_nodes, sd_fit, sd_rank);
              if (sd_auto > 0) vtx.set_sigma_dyn_auto_nodes(sd_auto);   // P14b
              if (sd_ckpt) vtx.set_sigma_dyn_ckpt_minutes(1e-9);   // P20: a checkpoint after every unit
              if (auto *rf = std::getenv("COQUI_DYNBSE_TEST_SIGDYN_REFIT")) vtx.set_sigma_dyn_refit(rf);   // P12: fit | union
              vtx.set_sigma_dyn_acc(sd_acc);   // P4-C14: split | single
              if (auto *rm = std::getenv("COQUI_DYNBSE_TEST_RESOLVENT")) vtx.set_ladder_dyn_resolvent(rm);   // P7: inverse | lu
              if (auto *wc = std::getenv("COQUI_DYNBSE_TEST_WCACHE")) vtx.set_wcache_mode(wc);   // P19: replicated | shared
              scr_eri.set_vertex(&vtx);
              e_corr = std::get<1>(scf_loop(mb_state, dyson, erin, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 1, false, 1e-9, true));
              m = scr_eri.sigma_pair_meter(); d = gw.sigma_pair_dsigma();
              if (col == "static" or col == "static1") {   // G1: the P side's own object from the same amplitudes vs the dump's static column
                nda::array<std::complex<double>, 4> Pc, Ps;
                { h5::file f(out + ".sigpair.h5", 'r'); h5::group g(f); nda::h5_read(g, "Pi_check", Pc); }
                { h5::file f("coqui_d3_winj_G.pol_wh_dyn.g1.h5", 'r'); h5::group g(f); nda::h5_read(g, "Pi_static", Ps); }
                REQUIRE(Pc.shape(1) == Ps.shape(1)); REQUIRE(Pc.shape(2) == Ps.shape(2));
                double dd = 0.0, nn = 0.0;
                for (long j = 0; j < Ps.shape(0); ++j)
                  for (long iq = 0; iq < Ps.shape(1); ++iq)
                    for (long M = 0; M < Ps.shape(2); ++M)
                      for (long N = 0; N < Ps.shape(3); ++N) { dd += std::norm(Pc(m0b + j, iq, M, N) - Ps(j, iq, M, N)); nn += std::norm(Ps(j, iq, M, N)); }
                pichk = std::sqrt(dd / nn);
              }
            } else {
              e_corr = std::get<1>(scf_loop(mb_state, dyson, erin, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 1, false, 1e-9, true));
            }
            mpi_context->comm.barrier();
            read_sig(out + ".mbpt.h5", Sig);
            app_log(1, "dynbse_readout LFF-Sigma pair [{}]: kind {} col {} outer {} scale {}: e_corr {:.12f}, max|dSigma| {:.6e} (anti-Hermitian {:.2e}, "
                       "|K_s - K_s^dag|/|K_s| {:.2e}, wall {:.1f} s), added {:.6e} vs max|Sigma^GW| {:.4e}; Pi-check vs the dump's static column {:.3e}",
                    tag, kind, col, outer, scale, e_corr, m[0], m[1], m[2], m[3], d[0], d[1], pichk);
            mpi_context->comm.barrier();
            if (mpi_context->comm.root()) {
              remove((out + ".mbpt.h5").c_str());
              for (auto const &e : std::filesystem::directory_iterator("."))
                if (e.path().filename().string().rfind(out + ".", 0) == 0 and e.path().extension() != ".h5" ) std::filesystem::remove(e.path());
              for (auto const &e : std::filesystem::directory_iterator("."))
                if (e.path().filename().string().rfind(out + ".", 0) == 0 and e.path().filename().string().find(".sigdyn.h5") == std::string::npos
                    and not (sd_ckpt and e.path().filename().string().find(".sigdyn_ckpt.") != std::string::npos)) std::filesystem::remove(e.path());
            }
            mpi_context->comm.barrier();
            return std::make_tuple(e_corr, m, d, pichk);
          };
          if (std::getenv("COQUI_DYNBSE_TEST_SIGDYN_SHARE")) {
            // ---- P3 gate: one solve feeding the P readout and the Sigma deposits ---------------------------------------------
            // SH0 / SH1: the P-side all-nu dynamic readout (every half node) AND the dynamic pair vertex in Sigma (col dyn1, outer
            // dynamic, every node) in one run, without / with pol_vertex_sigma_share: the 21 nu >= 0 nodes are then deposited by
            // the P-side solve and the Sigma-side call solves the 20 others. Sigma and the P dump must agree (the same units on
            // the same W-bar cache): Pi_gam1 bitwise, Sigma to rounding.
            auto run_sh = [&](std::string const &tag, bool share, S5 &Sig, nda::array<std::complex<double>, 4> &Pg) {
              const std::string out = "coqui_d3_winj_" + tag;
              solvers::hf_t hf; solvers::gw_t gw(&ft, "ignore_g0", out);
              solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
              simple_dyson dyson(mfn.get(), &ft); MBState mb_state(mpi_context, ft, out);
              iter_scf::iter_scf_t iter_sol("damping");
              solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mfn->nbnd());
              vtx.set_pol_vertex("ladder", "w0_prev", nda::range(0, 4), -1, 1e-8, -1.0, -1.0, -1.0, "none");
              vtx.set_ladder_rung("dynamic", 1e-8, 30, 12, -1.0);
              vtx.set_ladder_dyn_gamma1_only(true); vtx.set_ladder_dyn_all_nu(true);
              vtx.set_isdf_points("coqui_d3_winj_G.secpts.h5", false);
              vtx.set_sigma_pair(true, "dyn1", "dynamic", 1.0, true, false, "right");
              vtx.set_sigma_share(share);
              scr_eri.set_vertex(&vtx);
              const double e_corr = std::get<1>(scf_loop(mb_state, dyson, erin, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 1, false, 1e-9, true));
              mpi_context->comm.barrier();
              read_sig(out + ".mbpt.h5", Sig);
              { h5::file f(out + ".pol_wh_dyn.g1.h5", 'r'); h5::group g(f); nda::h5_read(g, "Pi_gam1", Pg); }
              app_log(1, "dynbse_readout LFF-Sigma SHARE [{}]: share {}: e_corr {:.12f}, max|dSigma| {:.6e}, wall {:.1f} s", tag, share, e_corr,
                      scr_eri.sigma_pair_meter()[0], scr_eri.sigma_pair_meter()[3]);
              mpi_context->comm.barrier();
              if (mpi_context->comm.root()) {
                remove((out + ".mbpt.h5").c_str());
                for (auto const &e : std::filesystem::directory_iterator("."))
                  if (e.path().filename().string().rfind(out + ".", 0) == 0) std::filesystem::remove(e.path());
              }
              mpi_context->comm.barrier();
              return e_corr;
            };
            // COQUI_DYNBSE_TEST_SIGDYN_SHARE=repro runs SH0 twice instead (the run-to-run floor of the dynamically scheduled P solve)
            const bool repro = (std::string(std::getenv("COQUI_DYNBSE_TEST_SIGDYN_SHARE")) == "repro");
            S5 S0, S1;
            nda::array<std::complex<double>, 4> P0, P1;
            const double c0 = run_sh("SH0", false, S0, P0), c1 = run_sh("SH1", not repro, S1, P1);
            double ds = 0.0, ns_ = 0.0, dp = 0.0, np_ = 0.0;
            for (long i = 0; i < S0.size(); ++i) { ds = std::max(ds, std::abs(S0.data()[i] - S1.data()[i])); ns_ = std::max(ns_, std::abs(S0.data()[i])); }
            for (long i = 0; i < P0.size(); ++i) { dp = std::max(dp, std::abs(P0.data()[i] - P1.data()[i])); np_ = std::max(np_, std::abs(P0.data()[i])); }
            app_log(1, "dynbse_readout LFF-Sigma SHARE gate (P3): {}: |dSigma| {:.2e} (max |Sigma| {:.3e}), "
                       "|dPi_gam1| {:.2e} (max |Pi_gam1| {:.3e}); e_corr {:+.12f} vs {:+.12f}",
                    repro ? "the separate-solve run repeated (run-to-run floor)" : "one solve for P and Sigma vs separate solves", ds, ns_, dp, np_, c1, c0);
            REQUIRE(dp < 1e-9 * np_);
            REQUIRE(ds < 1e-11 * ns_);
            mpi_context->comm.barrier();
            return;
          }
          if (std::getenv("COQUI_DYNBSE_TEST_SIGDYN_CKPT")) {
            // ---- P20 gate: the Sigma-accumulator checkpoint / restart of the dynamic Sigma solve ---------------------------------
            // CK1 (a) solves every unit and checkpoints after each; (b) the same prefix again: every unit is loaded, none solved;
            // (c) rank 1's checkpoint removed: rank 1's units are re-solved, rank 0's loaded. All three dSigma must agree.
            S5 S_a, S_b, S_c;
            sd_ckpt = true;
            auto [ca_, ma_, da_, ka_] = run_p("CK1", 2, "dyn1_bare", "dynamic", 1.0, S_a);
            REQUIRE(std::filesystem::exists("coqui_d3_winj_CK1.sigdyn_ckpt.r0.h5"));
            auto [cb_, mb_, db_, kb_] = run_p("CK1", 2, "dyn1_bare", "dynamic", 1.0, S_b);
            mpi_context->comm.barrier();
            if (mpi_context->comm.root()) remove("coqui_d3_winj_CK1.sigdyn_ckpt.r1.h5");
            mpi_context->comm.barrier();
            auto [cc_, mc_, dc_, kc_] = run_p("CK1", 2, "dyn1_bare", "dynamic", 1.0, S_c);
            sd_ckpt = false;
            double dab = 0.0, dac = 0.0, na = 0.0;
            for (long i = 0; i < S_a.size(); ++i) {
              dab = std::max(dab, std::abs(S_a.data()[i] - S_b.data()[i])); dac = std::max(dac, std::abs(S_a.data()[i] - S_c.data()[i]));
              na = std::max(na, std::abs(S_a.data()[i]));
            }
            app_log(1, "dynbse_readout LFF-Sigma CKPT gate (P20): solve + checkpoint vs full restart |dSigma| {:.2e}, vs the partial restart (rank 1 re-solved) {:.2e} "
                       "(max |Sigma| {:.3e}); e_corr {:+.12f} {:+.12f} {:+.12f}", dab, dac, na, ca_, cb_, cc_);
            REQUIRE(dab == 0.0);
            REQUIRE(dac < 1e-12 * na);
            mpi_context->comm.barrier();
            if (mpi_context->comm.root())
              for (auto const &e : std::filesystem::directory_iterator("."))
                if (e.path().filename().string().rfind("coqui_d3_winj_CK1.", 0) == 0) std::filesystem::remove(e.path());
            mpi_context->comm.barrier();
            return;
          }
          if (auto const *only = std::getenv("COQUI_DYNBSE_TEST_N_ONLY"); only != nullptr) {
            // diagnostics: run ONE dynamic-path column (dyn1_bare | dyn1 | static_dyn) and stop -- for the route / family
            // cross-checks of vertex_sigma_dyn.icc (COQUI_SIGDYN_ROUTE, COQUI_SIGDYN_FAMILIES); the driver logs |dSigma|_F
            S5 S_one;
            auto [c1, m1, d1, k1] = run_p("D1B", 2, only, "dynamic", 1.0, S_one);
            app_log(1, "dynbse_readout LFF-Sigma N_ONLY {}: e_corr {:+.12f}, max|dSigma| {:.6e}, anti-Hermitian {:.3e}", only, c1, m1[0], m1[1]);
            mpi_context->comm.barrier();
            return;
          }
          if (std::getenv("COQUI_DYNBSE_TEST_SIGDYN_ACC1")) {
            // ---- P4-C14 gate: the single tau-resolved accumulator vs the split (node-resolved) ones -------------------------------
            // A1 / A2: col dyn1, outer dynamic on all nodes, split (writing the per-node dump) / single; then the SAMPLED mode (the
            // 11-node recipe of the SAMPLED gate on A1's dump) split / single. The two paths apply the same fixed linear maps in
            // another order: rounding x cond(refit) -- reported; REQUIRE < 1e-6 relative on the C block.
            S5 S_a1, S_a2, S_a3, S_a4;
            sd_acc = "split"; sd_dump = true;
            auto [ca1, ma1, da1, ka1] = run_p("A1", 2, "dyn1", "dynamic", 1.0, S_a1);
            sd_dump = false;
            REQUIRE(std::filesystem::exists("coqui_d3_winj_A1.sigdyn.h5"));
            sd_acc = "single";
            auto [ca2, ma2, da2, ka2] = run_p("A2", 2, "dyn1", "dynamic", 1.0, S_a2);
            auto rel4 = [&](S5 const &A, S5 const &B) {
              double num = 0.0, den = 0.0, mx = 0.0;
              for (long it = 0; it < A.shape(0); ++it)
                for (long is = 0; is < A.shape(1); ++is)
                  for (long ik = 0; ik < A.shape(2); ++ik)
                    for (long i = 0; i < 4; ++i)
                      for (long j = 0; j < 4; ++j) { num += std::norm(A(it, is, ik, i, j) - B(it, is, ik, i, j)); den += std::norm(A(it, is, ik, i, j)); mx = std::max(mx, std::abs(A(it, is, ik, i, j) - B(it, is, ik, i, j))); }
              return std::make_pair(std::sqrt(num) / std::max(std::sqrt(den), 1e-300), mx);
            };
            auto [r12, m12] = rel4(S_a1, S_a2);
            app_log(1, "dynbse_readout LFF-Sigma ACC gate (P4-C14), all nodes: single vs split accumulators: rel Frobenius {:.3e} (max |d| {:.3e}); "
                       "e_corr {:+.12f} vs {:+.12f}; max|dSigma| {:.6e} vs {:.6e}", r12, m12, ca2, ca1, ma2[0], ma1[0]);
            REQUIRE(r12 < 1e-6);
            const long nwb = ft.wn_mesh_b().shape(0), hm = nwb / 2;
            sd_nodes = {m0b}; for (long j : {4l, 8l, 12l, 16l, hm}) { sd_nodes.push_back(m0b + j); sd_nodes.push_back(m0b - j); }
            sd_fit = "coqui_d3_winj_A1.sigdyn.h5"; sd_rank = 6;
            sd_acc = "split";
            auto [ca3, ma3, da3, ka3] = run_p("A3", 2, "dyn1", "dynamic", 1.0, S_a3);
            sd_acc = "single";
            auto [ca4, ma4, da4, ka4] = run_p("A4", 2, "dyn1", "dynamic", 1.0, S_a4);
            sd_nodes.clear(); sd_fit.clear(); sd_rank = 0; sd_acc = "split";
            auto [r34, m34] = rel4(S_a3, S_a4);
            app_log(1, "dynbse_readout LFF-Sigma ACC gate (P4-C14), sampled (11 of {} nodes, K = 6): single vs split accumulators: rel Frobenius {:.3e} (max |d| {:.3e}); "
                       "e_corr {:+.12f} vs {:+.12f}", nwb, r34, m34, ca4, ca3);
            REQUIRE(r34 < 1e-6);
            if (mpi_context->comm.root()) remove("coqui_d3_winj_A1.sigdyn.h5");
            mpi_context->comm.barrier();
            return;
          }
          if (std::getenv("COQUI_DYNBSE_TEST_SIGDYN_SAMPLED")) {
            // ---- L-8 gate: the nu-SAMPLED dynamic Sigma vertex vs the all-node one -------------------------------------------
            // D1D: col dyn1 on ALL nodes, writing the per-node objects; D1S: col dyn1 on 9 nodes (the S1 recipe's structure:
            // nu = 0, three interior pivots and the tail node, both signs) with the nu-bases learned from D1D (K = 9:
            // interpolation, exact at the samples). Reported: the relative Frobenius distance of the two dSigma on the C block.
            S5 S_r0s, S_d1d, S_d1s;
            auto [cr0s, mr0s, dr0s, kr0s] = run_p("R0", 0, "static", "static", 1.0, S_r0s);
            sd_dump = true;
            auto [cd1d, md1d, dd1d, kd1d] = run_p("D1D", 2, "dyn1", "dynamic", 1.0, S_d1d);
            sd_dump = false;
            REQUIRE(std::filesystem::exists("coqui_d3_winj_D1D.sigdyn.h5"));
            if (std::getenv("COQUI_DYNBSE_TEST_SIGDYN_DUMP_ONLY")) return;   // keep the dump for offline analysis
            const long nwb = ft.wn_mesh_b().shape(0), hm = nwb / 2;   // LiH: 41 nodes, nu = 0 at 20
            // an evenly spread set (in node index ~ log nu): nu = 0 and +-4, 8, 12, 16, 20 -> 11 of 41 nodes; K = 6 per basis (least squares)
            // COQUI_DYNBSE_TEST_SIGDYN_SAMPLED=auto: the 11 nodes chosen by the dump's own nu-modes instead (P14b)
            const bool auto_mode = (std::string(std::getenv("COQUI_DYNBSE_TEST_SIGDYN_SAMPLED")) == "auto");
            sd_fit = "coqui_d3_winj_D1D.sigdyn.h5"; sd_rank = 6;
            if (auto_mode) sd_auto = 11;
            else { sd_nodes = {m0b}; for (long j : {4l, 8l, 12l, 16l, hm}) { sd_nodes.push_back(m0b + j); sd_nodes.push_back(m0b - j); } }
            auto [cd1s, md1s, dd1s, kd1s] = run_p("D1S", 2, "dyn1", "dynamic", 1.0, S_d1s);
            sd_nodes.clear(); sd_fit.clear(); sd_rank = 0; sd_auto = 0;
            double num = 0.0, den = 0.0, mx = 0.0;
            for (long it = 0; it < S_d1d.shape(0); ++it)
              for (long is = 0; is < S_d1d.shape(1); ++is)
                for (long ik = 0; ik < S_d1d.shape(2); ++ik)
                  for (long i = 0; i < 4; ++i)
                    for (long j = 0; j < 4; ++j) {
                      const auto da = S_d1s(it, is, ik, i, j) - S_r0s(it, is, ik, i, j), db = S_d1d(it, is, ik, i, j) - S_r0s(it, is, ik, i, j);
                      num += std::norm(da - db); den += std::norm(db); mx = std::max(mx, std::abs(da - db));
                    }
            app_log(1, "dynbse_readout LFF-Sigma SAMPLED gate: dyn1 on {} of {} nodes ({}; K = {}, per-p U/T bases) vs all nodes: rel Frobenius {:.3e} (max |d| {:.3e}); "
                       "e_corr all {:+.10f} sampled {:+.10f} (R0 {:+.10f}); max|dSigma| all {:.6e} sampled {:.6e}; anti-Hermitian all {:.2e} sampled {:.2e}",
                    11, nwb, auto_mode ? "chosen by the dump's nu-modes, P14b" : "the fixed evenly spread set", 6,
                    std::sqrt(num) / std::max(std::sqrt(den), 1e-300), mx, cd1d, cd1s, cr0s, md1d[0], md1s[0], md1d[1], md1s[1]);
            REQUIRE(den > 0.0);
            REQUIRE(std::sqrt(num) / std::sqrt(den) < 5e-2);
            if (mpi_context->comm.root()) remove("coqui_d3_winj_D1D.sigdyn.h5");
            mpi_context->comm.barrier();
            return;
          }
          S5 S_r0, S_xs, S_p1, S_p0, S_ps, S_ph;
          auto [cr0, mr0, dr0, kr0] = run_p("R0", 0, "static", "static", 1.0, S_r0);
          auto [cxs, mxs, dxs, kxs] = run_p("XS", 1, "static", "static", 1.0, S_xs);
          auto [cp1, mp1, dp1, kp1] = run_p("P1", 2, "static1", "static", 1.0, S_p1);
          auto [cp0, mp0, dp0, kp0] = run_p("P0", 2, "static", "dynamic", 0.0, S_p0);
          auto [cps, mps, dps, kps] = run_p("PS", 2, "static", "dynamic", 1.0, S_ps);
          auto [cph, mph, dph, kph] = run_p("PH", 2, "static", "dynamic", 0.5, S_ph);
          S5 S_pt;
          auto [cpt, mpt, dpt, kpt] = run_p("PT", 2, "static", "static", 1.0, S_pt);   // the resummed ladder with the static outer W (a complete diagram set: Hermitian?)
          REQUIRE(S_xs.shape() == S_r0.shape()); REQUIRE(S_p1.shape() == S_r0.shape());
          const long nb = 4;
          auto win_diff = [&](S5 const &A, S5 const &B, double fac, S5 const &Cc, double &num, double &den, double &mx) {
            // num = || (A - R0) - fac (B - R0) ||_F on the C block, den = || (B - R0) ||_F, mx = max abs of the same difference
            num = den = mx = 0.0;
            for (long it = 0; it < A.shape(0); ++it)
              for (long is = 0; is < A.shape(1); ++is)
                for (long ik = 0; ik < A.shape(2); ++ik)
                  for (long i = 0; i < nb; ++i)
                    for (long j = 0; j < nb; ++j) {
                      const auto da = A(it, is, ik, i, j) - Cc(it, is, ik, i, j), db = B(it, is, ik, i, j) - Cc(it, is, ik, i, j);
                      num += std::norm(da - fac * db); den += std::norm(db); mx = std::max(mx, std::abs(da - fac * db));
                    }
            num = std::sqrt(num); den = std::sqrt(den);
          };
          double n1, d1, x1, n0, d0, x0, nh, dh_, xh, nfull, dfull, xfull;
          win_diff(S_p1, S_xs, 1.0, S_r0, n1, d1, x1);          // G2: the exact identity
          win_diff(S_p0, S_r0, 1.0, S_r0, n0, d0, x0);          // G3: scale 0 -> bitwise R0
          win_diff(S_ph, S_ps, 0.5, S_r0, nh, dh_, xh);         // linearity
          win_diff(S_ps, S_xs, 1.0, S_r0, nfull, dfull, xfull); // the production object vs the one-rung static diagram (a size, not a gate)
          // the off-window rows of Sigma must be untouched by the pair vertex (it is C-C by construction)
          double off = 0.0;
          for (long it = 0; it < S_ps.shape(0); ++it)
            for (long is = 0; is < S_ps.shape(1); ++is)
              for (long ik = 0; ik < S_ps.shape(2); ++ik)
                for (long i = 0; i < S_ps.shape(3); ++i)
                  for (long j = 0; j < S_ps.shape(4); ++j)
                    if (i >= nb or j >= nb) off = std::max(off, std::abs(S_ps(it, is, ik, i, j) - S_r0(it, is, ik, i, j)));
          app_log(1, "dynbse_readout LFF-Sigma pair gate: EXACT IDENTITY one static rung + static outer W vs B-S Sigma^(C,x): rel Frobenius {:.3e} "
                     "(max |d| {:.3e}, |Sigma^(C,x)|_F {:.4e}, max |dSigma_pair| {:.4e}); e_corr XS {:+.10f} P1 {:+.10f} (R0 {:+.10f}); "
                     "scale 0: max |Sigma(P0) - Sigma(R0)| {:.2e}; linearity |dSigma(1/2) - dSigma(1)/2| {:.2e} rel {:.2e}; resummed + dynamic outer W vs "
                     "the static diagram: rel {:.3e}; off-window leakage {:.2e}; Pi-check(PS) {:.3e}; anti-Hermitian residual before Hermitization "
                     "(P1 one rung + static W {:.2e}, PT resummed + static W {:.2e}, PS resummed + dynamic W {:.2e}); K_s hermiticity {:.2e}; "
                     "e_corr: R0 {:+.8f} P1 {:+.8f} PT {:+.8f} PS {:+.8f}",
                  n1 / d1, x1, d1, mp1[0], cxs, cp1, cr0, x0, xh, (dh_ > 0.0 ? nh / dh_ : 0.0), nfull / std::max(dfull, 1e-300), off, kps, mp1[1], mpt[1], mps[1], mps[2],
                  cr0, cp1, cpt, cps);
          REQUIRE(d1 > 0.0);
          REQUIRE(n1 / d1 < 1e-8);                 // the exact identity (3e-12 measured; the leg/W variants miss it by 39-59 %)
          REQUIRE(x0 == 0.0);                      // scale 0: bit-identical to plain GW
          REQUIRE(nh <= 1e-12 * dh_);              // linear in the scale
          REQUIRE(off == 0.0);                     // C-C only
          REQUIRE(kps >= 0.0); REQUIRE(kps < 1e-8); // the P side's own object from the same amplitudes == the dump's static column
          REQUIRE(mp1[1] < 1e-8);                  // the one-rung static diagram is Hermitian by itself (the one-sided insertion with a
                                                   // dynamic outer W is not -- 6.7e-2 measured, Hermitized, logged above)
          REQUIRE(std::abs(cps - cr0) > 1e-10);    // the production object changes the correlation energy
          if (std::getenv("COQUI_DYNBSE_TEST_SKIP_N") == nullptr) {
            // ---- section N (2026-09-20): L-6b the junction side + L-7 the DYNAMIC-rung vertex in Sigma (vertex_sigma_dyn.icc) ----
            // N1: the dynamic path with y = 0 (col static_dyn) == the static path (col static): the same amplitude K_s Gsum0 = T_s Cb D
            //     (Cb_cst route) through the tau-closure of vertex_sigma_dyn -- an identity to the Cb round trip;
            // N2: the ONE BARE dynamic rung (col dyn1_bare, T_s = 0) with the dynamic outer W == the G^3 W^2 second-order exchange
            //     Sigma^C of the dynamic-rung B-S theory (vertex_type 2nd_exchange, rung dynamic, secondary frame, bl_drop 1):
            //     the four terms [W_0 W_0] + [W_d W_0] + [W_0 W_d] + [W_d W_d] at one rung -- the exact identity of the dynamic path
            //     (the twisted family, the E refit, the nu accumulation over both signs), to the pole-fit / refit class;
            // N3: dyn1 (Gamma_1) and dyn (resummed): sizes, linearity in the scale, C-C only, Hermiticity residual;
            // N4: the LEFT junction: (a) one rung + static W on the left leg == Sigma^{C,x} too (the crossed diagram is unique);
            //     (b) resummed + dynamic W: |Sigma_L - Sigma_R^dag| / |Sigma_R| (is the mirror insertion the Hermitian conjugate?);
            //     (c) side both == (Sigma_R + Sigma_L) / 2 (linearity) and its distance from the Hermitized one-sided object.
            S5 S_xd, S_sd, S_d1b, S_d1, S_d1h, S_dr, S_pl1, S_pls, S_pb;
            auto [cxd, mxd, dxd, kxd] = run_p("XD", 3, "static", "static", 1.0, S_xd);
            auto [csd, msd, dsd, ksd] = run_p("SD", 2, "static_dyn", "dynamic", 1.0, S_sd);
            auto [cd1b, md1b, dd1b, kd1b] = run_p("D1B", 2, "dyn1_bare", "dynamic", 1.0, S_d1b);
            auto [cd1, md1, dd1, kd1] = run_p("D1", 2, "dyn1", "dynamic", 1.0, S_d1);
            auto [cd1h, md1h, dd1h, kd1h] = run_p("D1H", 2, "dyn1", "dynamic", 0.5, S_d1h);
            // COQUI_DYNBSE_TEST_SKIP_DR: reuse the Gamma_1 column in place of the resummed one (39 min on 2 ranks; measured 2026-09-19:
            // e_corr -0.10505671, max|dSigma| 1.3601e-02, anti-Hermitian 4.75e-02 vs Gamma_1 -0.10531543 / 1.3606e-02 / 3.99e-02)
            const bool skip_dr = (std::getenv("COQUI_DYNBSE_TEST_SKIP_DR") != nullptr);
            auto [cdr, mdr, ddr, kdr] = skip_dr ? std::make_tuple(cd1, md1, dd1, kd1) : run_p("DR", 2, "dyn", "dynamic", 1.0, S_dr);
            if (skip_dr) S_dr = S_d1;
            side_cur = "left";
            auto [cpl1, mpl1, dpl1, kpl1] = run_p("PL1", 2, "static1", "static", 1.0, S_pl1);
            auto [cpls, mpls, dpls, kpls] = run_p("PLS", 2, "static", "dynamic", 1.0, S_pls);
            side_cur = "both";
            auto [cpb, mpb, dpb, kpb] = run_p("PB", 2, "static", "dynamic", 1.0, S_pb);
            side_cur = "right";
            double n_sd, d_sd, x_sd, n_xd, d_xd, x_xd, n_l1, d_l1, x_l1, n_h, d_h, x_h, n_b, d_b, x_b;
            win_diff(S_sd, S_ps, 1.0, S_r0, n_sd, d_sd, x_sd);            // N1
            win_diff(S_d1b, S_xd, 1.0, S_r0, n_xd, d_xd, x_xd);           // N2
            win_diff(S_pl1, S_xs, 1.0, S_r0, n_l1, d_l1, x_l1);           // N4a
            win_diff(S_d1h, S_d1, 0.5, S_r0, n_h, d_h, x_h);              // N3 linearity
            // N4b: the left insertion vs the Hermitian conjugate of the right one (both un-Hermitized would be the clean
            // statement; the runs are Hermitized, so compare the Hermitized objects: equal iff the mirror identity holds)
            double n_lr = 0.0, d_lr = 0.0, n_bh = 0.0;
            for (long it = 0; it < S_ps.shape(0); ++it)
              for (long is = 0; is < S_ps.shape(1); ++is)
                for (long ik = 0; ik < S_ps.shape(2); ++ik)
                  for (long i = 0; i < nb; ++i)
                    for (long j = 0; j < nb; ++j) {
                      const auto dr_ = S_ps(it, is, ik, i, j) - S_r0(it, is, ik, i, j);
                      const auto dl_ = S_pls(it, is, ik, i, j) - S_r0(it, is, ik, i, j);
                      const auto db_ = S_pb(it, is, ik, i, j) - S_r0(it, is, ik, i, j);
                      n_lr += std::norm(dl_ - dr_); d_lr += std::norm(dr_); n_bh += std::norm(db_ - 0.5 * (dr_ + dl_));
                    }
            n_lr = std::sqrt(n_lr); d_lr = std::sqrt(d_lr); n_bh = std::sqrt(n_bh);
            double off_d = 0.0;
            for (long it = 0; it < S_dr.shape(0); ++it)
              for (long is = 0; is < S_dr.shape(1); ++is)
                for (long ik = 0; ik < S_dr.shape(2); ++ik)
                  for (long i = 0; i < S_dr.shape(3); ++i)
                    for (long j = 0; j < S_dr.shape(4); ++j)
                      if (i >= nb or j >= nb) off_d = std::max(off_d, std::abs(S_dr(it, is, ik, i, j) - S_r0(it, is, ik, i, j)));
            app_log(1, "dynbse_readout LFF-Sigma DYN gate: N1 static limit of the dynamic path vs the static path: rel {:.3e} (max |d| {:.3e}); "
                       "N2 EXACT IDENTITY one bare dynamic rung + dynamic outer W vs the DYNAMIC B-S Sigma^C (G^3 W^2): rel Frobenius {:.3e} "
                       "(max |d| {:.3e}, |Sigma^C_dyn|_F {:.4e}); e_corr XD {:+.10f} D1B {:+.10f} (R0 {:+.10f}, XS {:+.10f}); "
                       "N3 sizes: max|dSigma| dyn1 {:.4e} dyn {:.4e} (static {:.4e}), anti-Hermitian residual dyn1 {:.2e} dyn {:.2e} (static {:.2e}), "
                       "e_corr dyn1 {:+.8f} dyn {:+.8f} (PS {:+.8f}), linearity |dSigma(1/2) - dSigma(1)/2| rel {:.2e}, off-window {:.2e}; "
                       "N4a left one rung + static W vs Sigma^(C,x): rel {:.3e}; N4b |Sigma_L - Sigma_R| / |Sigma_R| (resummed + dynamic W, Hermitized) "
                       "{:.3e} (e_corr PL {:+.8f} PLS {:+.8f}); N4c |both - (R + L)/2| {:.2e}",
                    n_sd / d_sd, x_sd, n_xd / std::max(d_xd, 1e-300), x_xd, d_xd, cxd, cd1b, cr0, cxs,
                    md1[0], mdr[0], mps[0], md1[1], mdr[1], mps[1], cd1, cdr, cps, (d_h > 0.0 ? n_h / d_h : 0.0), off_d,
                    n_l1 / d_l1, (d_lr > 0.0 ? n_lr / d_lr : 0.0), cpl1, cpls, n_bh);
            REQUIRE(n_sd / d_sd < 1e-8);           // N1: the dynamic path reproduces the static-ladder Sigma vertex
            REQUIRE(d_xd > 0.0);
            REQUIRE(n_xd / d_xd < 1e-3);           // N2: the exact identity with the dynamic Sigma^C (pole-fit / refit class)
            REQUIRE(n_h <= 1e-10 * d_h);           // N3: linear in the scale
            REQUIRE(off_d == 0.0);                 // C-C only
            REQUIRE(n_l1 / d_l1 < 1e-8);           // N4a: the left one-rung insertion is the same crossed diagram
            REQUIRE(n_bh < 1e-10);                 // N4c: both == the average of the two one-sided insertions
            REQUIRE(std::abs(cd1 - cr0) > 1e-10);  // the dynamic vertex changes the correlation energy
          }
        }
};
      if (std::getenv("COQUI_DYNBSE_TEST_M_ONLY")) { section_m(); return; }   // re-run section M alone on an existing run-G dump
      auto run_i = [&](std::string const &tag, std::string const &points, std::string const &interp, int niter) {
        const std::string out = "coqui_d3_winj_" + tag;
        solvers::hf_t hf; solvers::gw_t gw(&ft, "ignore_g0", out);
        solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
        simple_dyson dyson(mfn.get(), &ft); MBState mb_state(mpi_context, ft, out);
        iter_scf::iter_scf_t iter_sol("damping");
        solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mfn->nbnd());
        vtx.set_pol_vertex("ladder", "w0_prev", nda::range(0, 4), -1, 1e-8, -1.0, -1.0, -1.0, "ladder_n2");
        vtx.set_ladder_rung("static", 1e-8, 30, 12, -1.0);
        vtx.set_ladder_dyn_dump(interp.empty());          // A dumps the injection object
        vtx.set_isdf_points(points, points.empty());       // A dumps its points, B freezes them
        vtx.set_pol_interp(interp, "ladder");
        if (auto *rm = std::getenv("COQUI_DYNBSE_TEST_RESOLVENT")) vtx.set_ladder_dyn_resolvent(rm);   // P7: inverse | lu
        if (auto *ac = std::getenv("COQUI_DYNBSE_TEST_SIGDYN_ACC")) vtx.set_sigma_dyn_acc(ac);   // P4-C14: split | single
        if (auto *wc = std::getenv("COQUI_DYNBSE_TEST_WCACHE")) vtx.set_wcache_mode(wc);   // P19: replicated | shared
        scr_eri.set_vertex(&vtx);
        auto [e_hf, e_corr] = scf_loop(mb_state, dyson, erin, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, niter, false, 1e-9, true);
        auto [er, el] = scr_eri.pol_eps_readout();
        const double eloop = scr_eri.pol_eps_loop();
        app_log(1, "dynbse_readout W-int-4f INJ [{}]: e_hf {:.12f} e_corr {:.12f} eps_M readout RPA {:.10f} +ladder {:.10f} loop-side {:.10f}",
                tag, e_hf, e_corr, er, el, eloop);
        mpi_context->comm.barrier();
        if (mpi_context->comm.root() and tag != "A") {
          remove((out + ".mbpt.h5").c_str());
          for (auto const &e : std::filesystem::directory_iterator("."))
            if (e.path().filename().string().rfind(out + ".", 0) == 0) std::filesystem::remove(e.path());
        }
        mpi_context->comm.barrier();
        return std::make_tuple(e_hf, e_corr, er, el, eloop);
      };
      auto [ha, ca, ra, la, ea] = run_i("A", "", "", 1);
      REQUIRE(std::filesystem::exists("coqui_d3_winj_A.pol_wh.g1.h5")); REQUIRE(std::filesystem::exists("coqui_d3_winj_A.secpts.h5"));
      auto [hb, cb, rb, lb, eb] = run_i("B", "coqui_d3_winj_A.secpts.h5", "coqui_d3_winj_A.pol_wh.g1.h5", 1);
      app_log(1, "dynbse_readout W-int-4f INJ gate: |d e_hf| {:.2e} |d e_corr| {:.2e} |d eps_M(readout +ladder)| {:.2e} |d eps_M(loop)| {:.2e}",
              std::abs(ha - hb), std::abs(ca - cb), std::abs(la - lb), std::abs(ea - eb));
      REQUIRE(std::abs(ha - hb) < 1e-12); REQUIRE(std::abs(ca - cb) < 1e-12);
      REQUIRE(std::abs(la - lb) < 1e-10); REQUIRE(std::abs(ea - eb) < 1e-10);
      REQUIRE(std::abs(la - ra) > 1e-4);   // the injection did something
      // G: the coarse-side Gamma_1 UPGRADE -- the dynamic rung at ALL q x ALL half nodes (pol_vertex_dyn_all_nu) dumped
      // as <G>.pol_wh_dyn.g1.h5; D: the loop with Gamma_1 injected from that file. Checks: the dump's static column ==
      // the injected static-rung ladder (same object, union-grid route: refit class), and D's loop-side eps_M == the
      // dynamic readout's Gamma_1 column of G (the same P^{C,L} through two consumers).
      auto run_g = [&](std::string const &tag, bool all_nu, std::string const &points, std::string const &interp, std::string const &col,
                       bool cut_r1 = true, bool bubble_only = false, std::vector<long> nodes = {}, std::string const &fit_file = "",
                       std::string const &mu_file = "") {
        const std::string out = "coqui_d3_winj_" + tag;
        solvers::hf_t hf; solvers::gw_t gw(&ft, "ignore_g0", out);
        solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
        simple_dyson dyson(mfn.get(), &ft); MBState mb_state(mpi_context, ft, out);
        iter_scf::iter_scf_t iter_sol("damping");
        solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mfn->nbnd());
        vtx.set_pol_vertex("ladder", "w0_prev", nda::range(0, 4), -1, 1e-8, -1.0, -1.0, -1.0, all_nu ? "none" : "ladder_n2");
        vtx.set_ladder_rung(all_nu ? "dynamic" : "static", 1e-8, 30, 12, -1.0);
        vtx.set_ladder_dyn_gamma1_only(true); vtx.set_ladder_dyn_dump(all_nu); vtx.set_ladder_dyn_all_nu(all_nu);
        vtx.set_ladder_dyn_cut_r1(cut_r1);
        vtx.set_ladder_dyn_bubble_only(bubble_only); vtx.set_ladder_dyn_all_nu_nodes(nodes); vtx.set_ladder_dyn_fit(fit_file, 0);
        vtx.set_ladder_dyn_resum_mu_file(mu_file);
        vtx.set_isdf_points(points, points.empty()); vtx.set_pol_interp(interp, col);
        if (auto *rm = std::getenv("COQUI_DYNBSE_TEST_RESOLVENT")) vtx.set_ladder_dyn_resolvent(rm);   // P7: inverse | lu
        if (auto *ac = std::getenv("COQUI_DYNBSE_TEST_SIGDYN_ACC")) vtx.set_sigma_dyn_acc(ac);   // P4-C14: split | single
        if (auto *wc = std::getenv("COQUI_DYNBSE_TEST_WCACHE")) vtx.set_wcache_mode(wc);   // P19: replicated | shared
        scr_eri.set_vertex(&vtx);
        auto [e_hf, e_corr] = scf_loop(mb_state, dyson, erin, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 1, false, 1e-9, true);
        auto ed = scr_eri.pol_eps_dyn();
        const double eloop = scr_eri.pol_eps_loop();
        app_log(1, "dynbse_readout W-int-4f INJ [{}]: e_corr {:.12f}, dynamic readout (static {:.10f}, one rung {:.10f}, Gamma_1 {:.10f}), loop-side {:.10f}",
                tag, e_corr, ed[0], ed[1], ed[2], eloop);
        mpi_context->comm.barrier();
        if (mpi_context->comm.root() and tag != "G" and tag != "G2" and tag != "GB" and tag != "GS" and tag != "GF") {
          remove((out + ".mbpt.h5").c_str());
          for (auto const &e : std::filesystem::directory_iterator("."))
            if (e.path().filename().string().rfind(out + ".", 0) == 0) std::filesystem::remove(e.path());
        }
        mpi_context->comm.barrier();
        return std::make_pair(ed, eloop);
      };
      auto [edg, elg] = run_g("G", true, "", "", "gam1");
      REQUIRE(std::filesystem::exists("coqui_d3_winj_G.pol_wh_dyn.g1.h5"));
      {   // the dump's static column vs the injected whalf ladder of run A (both dumps carry all nodes)
        nda::array<std::complex<double>, 4> Ps, Pl;
        { h5::file f("coqui_d3_winj_G.pol_wh_dyn.g1.h5", 'r'); h5::group g(f); nda::h5_read(g, "Pi_static", Ps); }
        { h5::file f("coqui_d3_winj_A.pol_wh.g1.h5", 'r'); h5::group g(f); nda::h5_read(g, "Pi_ladder", Pl); }
        REQUIRE(Ps.shape() == Pl.shape());
        double d = 0.0, n = 0.0;
        for (long i = 0; i < Ps.size(); ++i) { d += std::norm(Ps.data()[i] - Pl.data()[i]); n += std::norm(Pl.data()[i]); }
        app_log(1, "dynbse_readout W-int-4f INJ: all-nu dump's static column vs the injected static-rung ladder: rel Frobenius {:.2e}", std::sqrt(d / n));
        REQUIRE(std::sqrt(d / n) < 1e-8);
      }
      {   // G2: the production setting pol_vertex_dyn_cut_r1 = false (the one-bare-rung pass skipped): the Gamma_1 and
          // static columns are bitwise those of G (the r1 pass is a separate solve), Pi_dyn1 repeats Pi_static
        auto [edg2, elg2] = run_g("G2", true, "", "", "gam1", false);
        REQUIRE(std::abs(edg2[0] - edg[0]) < 1e-12); REQUIRE(std::abs(edg2[2] - edg[2]) < 1e-12);
        nda::array<std::complex<double>, 4> Pg, Pg2, Ps2, P12;
        { h5::file f("coqui_d3_winj_G.pol_wh_dyn.g1.h5", 'r'); h5::group g(f); nda::h5_read(g, "Pi_gam1", Pg); }
        { h5::file f("coqui_d3_winj_G2.pol_wh_dyn.g1.h5", 'r'); h5::group g(f); nda::h5_read(g, "Pi_gam1", Pg2);
          nda::h5_read(g, "Pi_static", Ps2); nda::h5_read(g, "Pi_dyn1", P12); }
        REQUIRE(Pg2.shape() == Pg.shape());
        double dg = 0.0, d1 = 0.0, n = 0.0;
        for (long i = 0; i < Pg.size(); ++i) {
          dg = std::max(dg, std::abs(Pg2.data()[i] - Pg.data()[i])); d1 = std::max(d1, std::abs(P12.data()[i] - Ps2.data()[i]));
          n = std::max(n, std::abs(Pg.data()[i]));
        }
        app_log(1, "dynbse_readout W-int-4f INJ: cut_r1 = false vs true: max |d Pi_gam1| {:.2e} (max |Pi_gam1| {:.2e}); max |Pi_dyn1 - Pi_static| {:.2e}", dg, n, d1);
        // the two dumps come from separate solves under DYNAMIC unit scheduling (a shared task counter): the (s, q, nu)
        // -> rank assignment is timing-dependent, so the all_reduce order of the unit results can differ between runs
        // (measured 2.5e-8 on 8e2 = 3e-11 relative, 2026-09-19); the identity holds to reduction-order noise.
        REQUIRE(dg <= 1e-9 * n); REQUIRE(d1 == 0.0);
        mpi_context->comm.barrier();
        if (mpi_context->comm.root()) {
          remove("coqui_d3_winj_G2.mbpt.h5");
          for (auto const &e : std::filesystem::directory_iterator("."))
            if (e.path().filename().string().rfind("coqui_d3_winj_G2.", 0) == 0) std::filesystem::remove(e.path());
        }
        mpi_context->comm.barrier();
      }
      {   // H (LFF-aux L-0, notes/lff_aux_plan.md): the window-bubble column and the sampled-node subset.
          // GB: bubble_only -> the file carries Pi_bub only, bitwise the Pi_bub column of the full dump G, Hermitian and
          // sign-definite at every (node, q). GS: nodes {0, 2} -> nu_sampled == {0, 2}, the dynamic columns equal G's
          // at those nodes and are zero elsewhere, Pi_bub still at every node.
        auto rd4 = [](std::string const &fn, std::string const &ds, nda::array<std::complex<double>, 4> &A) {
          h5::file f(fn, 'r'); h5::group g(f); nda::h5_read(g, ds, A); };
        nda::array<std::complex<double>, 4> PbG, PbB, PgG, PgS, PbS, PsS;
        rd4("coqui_d3_winj_G.pol_wh_dyn.g1.h5", "Pi_bub", PbG);
        run_g("GB", true, "", "", "gam1", false, true);
        {
          h5::file f("coqui_d3_winj_GB.pol_wh_dyn.g1.h5", 'r'); h5::group g(f);
          REQUIRE(g.has_dataset("Pi_bub")); REQUIRE(not g.has_dataset("Pi_gam1"));
          nda::array<long, 1> ns; nda::h5_read(g, "nu_sampled", ns);
          REQUIRE(ns.size() == PbG.shape(0));
        }
        rd4("coqui_d3_winj_GB.pol_wh_dyn.g1.h5", "Pi_bub", PbB);
        REQUIRE(PbB.shape() == PbG.shape());
        double db = 0.0, nb = 0.0, dh = 0.0, evpos = 0.0, evneg = 0.0;
        long npos = 0, nneg = 0;
        for (long i = 0; i < PbB.size(); ++i) { db = std::max(db, std::abs(PbB.data()[i] - PbG.data()[i])); nb = std::max(nb, std::abs(PbG.data()[i])); }
        for (long j = 0; j < PbG.shape(0); ++j)
          for (long iq = 0; iq < PbG.shape(1); ++iq) {
            nda::array<std::complex<double>, 2> H(PbG(j, iq, nda::ellipsis{}));
            for (long M = 0; M < H.shape(0); ++M)
              for (long N = 0; N < H.shape(1); ++N) dh = std::max(dh, std::abs(H(M, N) - std::conj(H(N, M))));
            nda::array<std::complex<double>, 2> Hh(H);
            for (long M = 0; M < H.shape(0); ++M)
              for (long N = 0; N < H.shape(1); ++N) Hh(M, N) = 0.5 * (H(M, N) + std::conj(H(N, M)));
            auto ev = nda::linalg::eigenvalues(Hh);
            double emax = 0.0;
            for (auto v : ev) emax = std::max(emax, std::abs(v));
            for (auto v : ev) { if (v > 1e-10 * emax) ++npos; if (v < -1e-10 * emax) ++nneg; }
            for (auto v : ev) { evpos = std::max(evpos, v / emax); evneg = std::min(evneg, v / emax); }
          }
        // the tau-route bubble carries the DLR class (prec low here: ~4e-9 relative Hermiticity noise); the window
        // bubble is negative semi-definite to that class (its largest positive eigenvalue relative to |B| is noise)
        app_log(1, "dynbse_readout LFF-aux H: bubble_only Pi_bub vs the full dump's column: max |d| {:.2e} (max |Pi_bub| {:.2e}); "
                   "Hermiticity max |P - P^dag| {:.2e} ({:.1e} relative); eigenvalues > 0: {}, < 0: {} (over all nodes x q), "
                   "largest positive / most negative relative to |ev|_max: {:.2e} / {:.2e}", db, nb, dh, dh / nb, npos, nneg, evpos, evneg);
        REQUIRE(db == 0.0); REQUIRE(nb > 0.0); REQUIRE(dh < 1e-7 * nb);
        REQUIRE(evneg < -0.5);            // the bubble's sign: negative semi-definite
        REQUIRE(evpos < 2e-5);            // no positive eigenvalue above the DLR class (prec low: 5.7e-6 measured)
        run_g("GS", true, "", "", "gam1", false, false, std::vector<long>{0, 2});
        rd4("coqui_d3_winj_G.pol_wh_dyn.g1.h5", "Pi_gam1", PgG);
        rd4("coqui_d3_winj_GS.pol_wh_dyn.g1.h5", "Pi_gam1", PgS);
        rd4("coqui_d3_winj_GS.pol_wh_dyn.g1.h5", "Pi_bub", PbS);
        {
          h5::file f("coqui_d3_winj_GS.pol_wh_dyn.g1.h5", 'r'); h5::group g(f);
          nda::array<long, 1> ns; nda::h5_read(g, "nu_sampled", ns);
          REQUIRE(ns.size() == 2); REQUIRE(ns(0) == 0); REQUIRE(ns(1) == 2);
        }
        REQUIRE(PgS.shape() == PgG.shape());
        double ds_in = 0.0, ds_out = 0.0, ng = 0.0, dbs = 0.0;
        for (long j = 0; j < PgG.shape(0); ++j) {
          const bool in = (j == 0 or j == 2);
          for (long iq = 0; iq < PgG.shape(1); ++iq)
            for (long M = 0; M < PgG.shape(2); ++M)
              for (long N = 0; N < PgG.shape(3); ++N) {
                const double d = std::abs(PgS(j, iq, M, N) - PgG(j, iq, M, N));
                if (in) ds_in = std::max(ds_in, d); else ds_out = std::max(ds_out, std::abs(PgS(j, iq, M, N)));
                ng = std::max(ng, std::abs(PgG(j, iq, M, N)));
                dbs = std::max(dbs, std::abs(PbS(j, iq, M, N) - PbG(j, iq, M, N)));
              }
        }
        app_log(1, "dynbse_readout LFF-aux H: sampled nodes {{0, 2}}: max |d Pi_gam1| at the sampled nodes {:.2e}, max |Pi_gam1| at the "
                   "unsampled nodes {:.2e} (max |Pi_gam1| {:.2e}); Pi_bub vs the full dump {:.2e}", ds_in, ds_out, ng, dbs);
        // ds_in: two separate GMRES solves under dynamic unit scheduling -> reduction-order noise (1.2e-11 relative measured)
        REQUIRE(ds_in < 1e-9 * ng); REQUIRE(ds_out == 0.0); REQUIRE(dbs == 0.0);
        {   // L-3: the on-demand fit. GF: nodes {0, 2, last} + fit_file = G's full dump (self-trained basis, 3 modes) -> the
            // written Pi_gam1 equals G's at the sampled nodes (least squares exact there) and approximates it elsewhere.
          const long nwh = PgG.shape(0);
          const std::vector<long> fnodes{0, nwh / 4, nwh / 2, nwh - 1};   // nu = 0, two interior nodes, the tail node
          if (mpi_context->comm.root()) {   // a constant mu table (2.0 at every node): Pi_dyn must come out as 2 Pi_gam1
            std::ofstream mf("coqui_d3_winj_mu.txt");
            mf << "# j nu mu (test: constant 2)\n";
            for (long j = 0; j < nwh; ++j) mf << j << " 0 2.0\n";
          }
          mpi_context->comm.barrier();
          run_g("GF", true, "", "", "gam1", false, false, fnodes, "coqui_d3_winj_G.pol_wh_dyn.g1.h5", "coqui_d3_winj_mu.txt");
          {
            nda::array<std::complex<double>, 4> Pg_, Pd_;
            rd4("coqui_d3_winj_GF.pol_wh_dyn.g1.h5", "Pi_gam1", Pg_); rd4("coqui_d3_winj_GF.pol_wh_dyn.g1.h5", "Pi_dyn", Pd_);
            double dm = 0.0, nm = 0.0;
            for (long i = 0; i < Pg_.size(); ++i) { dm = std::max(dm, std::abs(Pd_.data()[i] - 2.0 * Pg_.data()[i])); nm = std::max(nm, std::abs(Pg_.data()[i])); }
            app_log(1, "dynbse_readout LFF mu: Pi_dyn vs 2 x Pi_gam1 (constant mu table): max |d| {:.2e} (max |Pi_gam1| {:.2e})", dm, nm);
            REQUIRE(dm <= 1e-12 * nm);
          }
          nda::array<std::complex<double>, 4> PgF;
          rd4("coqui_d3_winj_GF.pol_wh_dyn.g1.h5", "Pi_gam1", PgF);
          REQUIRE(PgF.shape() == PgG.shape());
          double d_in = 0.0, n_in = 0.0, d_out = 0.0, n_out = 0.0;
          for (long j = 0; j < nwh; ++j) {
            const bool in = (std::find(fnodes.begin(), fnodes.end(), j) != fnodes.end());
            double dj = 0.0, nj = 0.0;
            for (long iq = 0; iq < PgG.shape(1); ++iq)
              for (long M = 0; M < PgG.shape(2); ++M)
                for (long N = 0; N < PgG.shape(3); ++N) { dj += std::norm(PgF(j, iq, M, N) - PgG(j, iq, M, N)); nj += std::norm(PgG(j, iq, M, N)); }
            if (in) { d_in += dj; n_in += nj; } else { d_out += dj; n_out += nj; }
          }
          app_log(1, "dynbse_readout LFF-aux L-3: on-demand fit ({} sampled nodes, self-trained basis): rel Frobenius error at the sampled "
                     "nodes {:.2e} (= the source dump's own non-Hermiticity, the written object is Hermitized), at the {} unsampled "
                     "nodes {:.2e}", fnodes.size(), std::sqrt(d_in / n_in), nwh - long(fnodes.size()), std::sqrt(d_out / n_out));
          REQUIRE(std::sqrt(d_in / n_in) < 1e-4);      // exact at the sampled nodes up to the Hermitization (K modes = K nodes)
          REQUIRE(std::sqrt(d_out / n_out) < 0.15);    // the 4-mode reconstruction of the 21-node object (printed)
          REQUIRE(n_out > 0.0);
        }
        mpi_context->comm.barrier();
        if (mpi_context->comm.root())
          for (auto const &t : {std::string("GB"), std::string("GS"), std::string("GF")}) {
            remove("coqui_d3_winj_mu.txt");
            remove(("coqui_d3_winj_" + t + ".mbpt.h5").c_str());
            for (auto const &e : std::filesystem::directory_iterator("."))
              if (e.path().filename().string().rfind("coqui_d3_winj_" + t + ".", 0) == 0) std::filesystem::remove(e.path());
          }
        mpi_context->comm.barrier();
      }
      auto [edd, eld] = run_g("D", false, "coqui_d3_winj_G.secpts.h5", "coqui_d3_winj_G.pol_wh_dyn.g1.h5", "gam1");
      app_log(1, "dynbse_readout W-int-4f INJ: Gamma_1 consumed in the W-Dyson: loop-side eps_M {:.10f} vs the dynamic readout's Gamma_1 column {:.10f} (|d| = {:.2e})",
              eld, edg[2], std::abs(eld - edg[2]));
      REQUIRE(std::abs(eld - edg[2]) < 1e-5);   // the nu -> tau -> nu round trip of the injection (1.6e-7 here; 3e-4 on Si kp888)
      {   // LFF-Sigma (Route 1) gate, section L: the local-field-factor vertex in SIGMA from the same dump (col gam1 +
          // Pi_bub), switched separately from the P-side injection. SP: the injection alone; S0: + the Sigma vertex at
          // scale 0 (bit-identical to SP); S1: scale 1 (window bubble); Sh: scale 1/2 (the vertex self-energy is LINEAR in
          // the scale); SF: the "full" bubble (a different, diluted vertex); SN: the Sigma side WITHOUT the P side.
        double xk_diff = -1.0;
        auto run_s = [&](std::string const &tag, std::string const &inject, std::string const &mode, std::string const &bub, double scale, bool with_static = true) {
          const std::string out = "coqui_d3_winj_" + tag;
          solvers::hf_t hf; solvers::gw_t gw(&ft, "ignore_g0", out);
          solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
          simple_dyson dyson(mfn.get(), &ft); MBState mb_state(mpi_context, ft, out);
          iter_scf::iter_scf_t iter_sol("damping");
          solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mfn->nbnd());
          vtx.set_pol_vertex("ladder", "w0_prev", nda::range(0, 4), -1, 1e-8, -1.0, -1.0, -1.0, inject);
          vtx.set_ladder_rung("static", 1e-8, 30, 12, -1.0);
          vtx.set_ladder_dyn_gamma1_only(true); vtx.set_ladder_dyn_cut_r1(false);
          vtx.set_isdf_points("coqui_d3_winj_G.secpts.h5", false); vtx.set_pol_interp("coqui_d3_winj_G.pol_wh_dyn.g1.h5", "gam1");
          vtx.set_sigma_lff(mode, bub, scale, 1e-3, 1.0, "", with_static);   // the strong-mode cutoff (1e-8 admits the bubble's null directions: |Gamma - 1| ~ 1e4)
          if (auto *rm = std::getenv("COQUI_DYNBSE_TEST_RESOLVENT")) vtx.set_ladder_dyn_resolvent(rm);   // P7: inverse | lu
          if (auto *ac = std::getenv("COQUI_DYNBSE_TEST_SIGDYN_ACC")) vtx.set_sigma_dyn_acc(ac);   // P4-C14: split | single
          if (auto *wc = std::getenv("COQUI_DYNBSE_TEST_WCACHE")) vtx.set_wcache_mode(wc);   // P19: replicated | shared
          scr_eri.set_vertex(&vtx);
          auto [e_hf, e_corr] = scf_loop(mb_state, dyson, erin, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 1, false, 1e-9, true);
          auto [dmax, smax, dfmax] = gw.sigma_lff_dsigma(); auto m = scr_eri.sigma_lff_meter();
          if (tag == "SD") {   // the exchange contraction with a custom kernel, driven with the bare Z, must reproduce hf_t's exchange
            auto &Dm = mb_state.sDm_skij.value();
            auto shp = Dm.shape();
            auto sF1 = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(*mpi_context, shp);
            auto sF2 = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(*mpi_context, shp);
            hf.evaluate(sF1, Dm.local(), thcn, dyson.sS_skij().local(), false, true);
            const long np = mpi_context->comm.size();
            const long npP = utils::find_proc_grid_min_diff(np, 1, 1), npQ = np / npP;
            auto dU = thcn.dZ({1, npP, npQ});
            solvers::lff_sigma_detail::exchange_with_kernel(Dm.local(), dU, sF2, thcn);
            hf.HF_K_correction(sF2, Dm.local(), dyson.sS_skij().local(), mfn->madelung());
            double d = 0.0, n = 0.0;
            if (mpi_context->node_comm.root()) {
              auto a = sF1.local(); auto b = sF2.local();
              for (long i = 0; i < a.size(); ++i) { d = std::max(d, std::abs(a.data()[i] - b.data()[i])); n = std::max(n, std::abs(a.data()[i])); }
            }
            d = mpi_context->comm.all_reduce_value(d, boost::mpi3::max<>{}); n = mpi_context->comm.all_reduce_value(n, boost::mpi3::max<>{});
            xk_diff = d;
            app_log(1, "dynbse_readout LFF-Sigma exchange-kernel identity: max |F_x(hf_t) - F_x(exchange_with_kernel(Z) + head)| = {:.3e} (max |F_x| = {:.3e})", d, n);
          }
          app_log(1, "dynbse_readout LFF-Sigma [{}]: inject {} sigma {} ({}, scale {}, static {}): e_hf {:.12f} e_corr {:.12f}, max|dSigma_dyn| {:.6e} (max|Sigma| {:.4e}), "
                     "max|dF| {:.6e}, local vertex(q_1, nu_0) {:+.4f}, |dW~_dyn|_F/|dW|_F {:.3e}, heads at nu_0: correction {:+.5f} vs loop {:+.5f}",
                  tag, inject, mode, bub, scale, with_static, e_hf, e_corr, dmax, smax, dfmax, m[0], m[1], m[2], m[3]);
          mpi_context->comm.barrier();
          if (mpi_context->comm.root()) {
            remove((out + ".mbpt.h5").c_str());
            for (auto const &e : std::filesystem::directory_iterator("."))
              if (e.path().filename().string().rfind(out + ".", 0) == 0) std::filesystem::remove(e.path());
          }
          mpi_context->comm.barrier();
          return std::make_tuple(e_corr, dmax, m, dfmax);
        };
        auto [cp, dp, mp, fp] = run_s("SP", "ladder_n2", "none", "window", 1.0);
        auto [c0, d0, m0, f0] = run_s("S0", "ladder_n2", "lff", "window", 0.0);
        auto [c1, d1, m1, f1] = run_s("S1", "ladder_n2", "lff", "window", 1.0);
        auto [ch, dh, mh, fh] = run_s("Sh", "ladder_n2", "lff", "window", 0.5);
        auto [cf, df, mf_, ff] = run_s("SF", "ladder_n2", "lff", "full", 1.0);
        auto [cn, dn, mn, fn] = run_s("SN", "none", "lff", "window", 1.0);
        auto [cd, dd, md, fd] = run_s("SD", "ladder_n2", "lff", "window", 1.0, false);   // the instantaneous part dropped
        app_log(1, "dynbse_readout LFF-Sigma gate: |e_corr(S0) - e_corr(SP)| {:.2e} (dSigma(S0) {:.2e}, dF(S0) {:.2e}); scale 1: d e_corr {:+.6e}, max|dSigma_dyn| {:.4e}, max|dF| {:.4e}; "
                   "linearity |dSigma(1/2) - dSigma(1)/2| {:.2e}, |dF(1/2) - dF(1)/2| {:.2e}; full bubble: d e_corr {:+.6e}, max|dSigma_dyn| {:.4e}; Sigma side alone: d e_corr {:+.6e}; "
                   "static dropped: max|dSigma_dyn| {:.4e} (== S1's), max|dF| {:.2e}; vertex(q_1, nu_0) {:+.4f}, head correction at nu_0 {:+.5f} (loop {:+.5f})",
                std::abs(c0 - cp), d0, f0, c1 - cp, d1, f1, std::abs(dh - 0.5 * d1), std::abs(fh - 0.5 * f1), cf - cp, df, cn - cp, dd, fd, m1[0], m1[2], m1[3]);
        REQUIRE(c0 == cp);                                   // scale 0: bit-identical to the injection alone
        REQUIRE(d0 == 0.0); REQUIRE(f0 == 0.0);
        REQUIRE(d1 > 0.0); REQUIRE(f1 > 0.0); REQUIRE(m1[1] > 0.0);
        REQUIRE(std::abs(dh - 0.5 * d1) <= 1e-12 * d1);     // the dynamic vertex self-energy is linear in the scale
        REQUIRE(std::abs(fh - 0.5 * f1) <= 1e-12 * f1);     // so is the instantaneous part
        REQUIRE(std::abs(c1 - cp) > 1e-10);                  // it changes the correlation energy
        REQUIRE(df > 0.0); REQUIRE(std::abs(df - d1) > 1e-10 * d1);   // the full bubble is a different (diluted) vertex
        REQUIRE(fd == 0.0); REQUIRE(dd == d1);               // dropping the instantaneous part leaves the dynamic one untouched
        REQUIRE(m1[0] > 0.0);                                // the LiH toy's ladder raises eps_M: the local vertex is positive at nu = 0
        REQUIRE(xk_diff >= 0.0); REQUIRE(xk_diff < 1e-10);   // exchange_with_kernel(Z) == the code's own exchange (K term + head correction)
        REQUIRE(dn > 0.0);                                   // the Sigma side runs without the P side
      }
      section_m();
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) {
        remove("coqui_d3_winj_G.mbpt.h5");
        for (auto const &e : std::filesystem::directory_iterator("."))
          if (e.path().filename().string().rfind("coqui_d3_winj_G.", 0) == 0) std::filesystem::remove(e.path());
        remove("coqui_d3_winj_A.mbpt.h5");
        for (auto const &e : std::filesystem::directory_iterator("."))
          if (e.path().filename().string().rfind("coqui_d3_winj_A.", 0) == 0) std::filesystem::remove(e.path());
      }
      mpi_context->comm.barrier();
      return;
    }
    if (std::getenv("COQUI_DYNBSE_TEST_SIGPAIR_SYMW")) {
      // ---- L-6c (2026-09-20): the pair-resolved Sigma vertex in WANNIER mode and on the SYMMETRIC mesh ----------------------
      // (W) nosym fixture: the one-rung static object (col static1, outer static) in the band frame vs in a Wannier frame that is a
      //     unitary mix V of the SAME window (M = |C|): the C-space object is gauge-invariant, so the band-frame dSigma must agree
      //     (the MLWF-frame contraction rotated back with U dSigma_bar U^dag); V = 1 must reproduce it to rounding.
      // (S) symmetric fixture (qe_lih222_sym): the same one-rung object on the IBZ externals vs the B-S Sigma^{C,x} on the symmetric
      //     mesh (its own IBZ path), both at the pair run's own (star-closed) points: the exact identity again.
      using S5 = nda::array<std::complex<double>, 5>;
      auto read_sig = [&](std::string const &fn, S5 &S) {
        h5::file f(fn, 'r'); h5::group g(f);
        auto it = g.open_group("scf").open_group("iter1");
        nda::h5_read(it, "Sigma_tskij", S);
      };
      // kind: 0 plain GW, 1 B-S Sigma^{C,x} (static rung, bl_drop 1), 2 the pair vertex (col static1, outer static; the eps readout
      // CONSUMES `interp` -- the static nu = 0 ladder readout is window/nosym-only, exactly as the W-int gate's consumer runs),
      // 4 the dynamic band run that dumps the points and the nu0 column (2 iterations, as run_w's A)
      bool ibz_cur = false;   // P1: the IBZ solve + star fold of the Sigma-side vertex (sym meshes)
      std::string col_cur = "static1", outer_cur = "static";   // the Sigma-side column / outer W of the kind-2 runs
      // P1 (2026-09-21): the window of the (S) part. The historic window [0, 4) CUTS the Gamma triplet (bands 3-5 of LiH), so the
      // C-sector rotations of the symmetric path LEAK (D-matrix leakage 0.25, unitarity defect 0.99, G-rotation residual 4.5e-2
      // on qe_lih222_sym): the full-mesh sym path is then NOT the exact object (its identity with the B-S Sigma^{C,x} holds
      // because both use the same lossy rotated legs), while the IBZ path (identity-frame legs at every full-mesh point) is.
      // COQUI_DYNBSE_TEST_SYMW_NC = 3 | 6 selects a symmetry-closed window ([0, 3): three non-degenerate bands; [0, 6): the
      // triplet complete), for which the two paths must agree; with NC != 4 the (W) Wannier part (a 4 x 4 mix) is skipped.
      const long ncw = std::getenv("COQUI_DYNBSE_TEST_SYMW_NC") ? std::atol(std::getenv("COQUI_DYNBSE_TEST_SYMW_NC")) : 4;
      auto run_sw = [&](std::string const &tag, std::shared_ptr<mf::MF> mfw, auto &eriw, int kind, std::string const &points,
                        std::string const &interp, nda::array<std::complex<double>, 2> const *V, bool keep, S5 &Sig) {
        const std::string out = "coqui_d3_sigsw_" + tag;
        solvers::hf_t hf; solvers::gw_t gw(&ft, "ignore_g0", out);
        solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
        simple_dyson dyson(mfw.get(), &ft); MBState mb_state(mpi_context, ft, out);
        iter_scf::iter_scf_t iter_sol("damping");
        double e_corr = 0.0;
        std::array<double, 4> m{0.0, 0.0, 0.0, 0.0};
        if (kind == 1) {
          solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(0, ncw), mfw->nbnd(), "ignore_g0", "secondary", -1, 1e-8, -1.0, -1.0, "static");
          vtx.set_isdf_points(points, points.empty());
          vtx.set_bl_drop(1);
          if (auto *rm = std::getenv("COQUI_DYNBSE_TEST_RESOLVENT")) vtx.set_ladder_dyn_resolvent(rm);   // P7: inverse | lu
          if (auto *ac = std::getenv("COQUI_DYNBSE_TEST_SIGDYN_ACC")) vtx.set_sigma_dyn_acc(ac);   // P4-C14: split | single
          if (auto *wc = std::getenv("COQUI_DYNBSE_TEST_WCACHE")) vtx.set_wcache_mode(wc);   // P19: replicated | shared
          scr_eri.set_vertex(&vtx); gw.set_vertex(&vtx);
          e_corr = std::get<1>(scf_loop(mb_state, dyson, eriw, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 1, false, 1e-9, true));
        } else if (kind == 2) {
          solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mfw->nbnd());
          vtx.set_pol_vertex("ladder", "w0_prev", nda::range(0, ncw), -1, 1e-8, -1.0, -1.0, -1.0, "none");
          vtx.set_ladder_rung("static", 1e-8, 30, 12, -1.0);
          vtx.set_isdf_points(points, points.empty());
          vtx.set_wannier_frame("aux");
          vtx.set_pol_interp(interp, "static");
          if (V) { auto proj = make_degenerate_projector(*mfw, 0, ncw, V, false); vtx.set_wannier_projector(proj, true); }
          vtx.set_sigma_pair(true, col_cur, outer_cur, 1.0, true, false, "right");
          vtx.set_sigma_pair_ibz(ibz_cur);
          if (auto *rm = std::getenv("COQUI_DYNBSE_TEST_RESOLVENT")) vtx.set_ladder_dyn_resolvent(rm);   // P7: inverse | lu
          if (auto *ac = std::getenv("COQUI_DYNBSE_TEST_SIGDYN_ACC")) vtx.set_sigma_dyn_acc(ac);   // P4-C14: split | single
          if (auto *wc = std::getenv("COQUI_DYNBSE_TEST_WCACHE")) vtx.set_wcache_mode(wc);   // P19: replicated | shared
          scr_eri.set_vertex(&vtx);
          e_corr = std::get<1>(scf_loop(mb_state, dyson, eriw, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 1, false, 1e-9, true));
          m = scr_eri.sigma_pair_meter();
        } else if (kind == 4) {
          solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mfw->nbnd());
          vtx.set_pol_vertex("ladder", "w0_prev", nda::range(0, ncw), -1, 1e-8, -1.0, -1.0, -1.0, "none");
          vtx.set_ladder_rung("dynamic", 1e-8, 30, 12, -1.0);
          vtx.set_ladder_dyn_gamma1_only(true); vtx.set_ladder_dyn_dump(true);
          vtx.set_isdf_points("", true); vtx.set_wannier_frame("aux");
          if (auto *rm = std::getenv("COQUI_DYNBSE_TEST_RESOLVENT")) vtx.set_ladder_dyn_resolvent(rm);   // P7: inverse | lu
          if (auto *ac = std::getenv("COQUI_DYNBSE_TEST_SIGDYN_ACC")) vtx.set_sigma_dyn_acc(ac);   // P4-C14: split | single
          if (auto *wc = std::getenv("COQUI_DYNBSE_TEST_WCACHE")) vtx.set_wcache_mode(wc);   // P19: replicated | shared
          scr_eri.set_vertex(&vtx);
          e_corr = std::get<1>(scf_loop(mb_state, dyson, eriw, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 2, false, 1e-9, true));
        } else {
          e_corr = std::get<1>(scf_loop(mb_state, dyson, eriw, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 1, false, 1e-9, true));
        }
        mpi_context->comm.barrier();
        read_sig(out + ".mbpt.h5", Sig);
        app_log(1, "dynbse_readout LFF-Sigma pair sym/Wannier [{}]: kind {} points \"{}\" Wannier {}: e_corr {:.12f}, max|dSigma| {:.6e} (anti-Hermitian {:.2e}, wall {:.1f} s)",
                tag, kind, points, V != nullptr, e_corr, m[0], m[1], m[3]);
        mpi_context->comm.barrier();
        if (mpi_context->comm.root() and not keep) {
          remove((out + ".mbpt.h5").c_str());
          for (auto const &e : std::filesystem::directory_iterator("."))
            if (e.path().filename().string().rfind(out + ".", 0) == 0) std::filesystem::remove(e.path());
        }
        mpi_context->comm.barrier();
        return e_corr;
      };
      auto cdiff = [&](S5 const &A, S5 const &B, S5 const &R, long nb, long nk_cmp, double &rel, double &mx) {
        // || (A - R) - (B - R) ||_F / || B - R ||_F on the C block, the first nk_cmp k-points
        double num = 0.0, den = 0.0; mx = 0.0;
        for (long it = 0; it < A.shape(0); ++it)
          for (long is = 0; is < A.shape(1); ++is)
            for (long ik = 0; ik < nk_cmp; ++ik)
              for (long i = 0; i < nb; ++i)
                for (long j = 0; j < nb; ++j) {
                  const auto da = A(it, is, ik, i, j) - R(it, is, ik, i, j), db = B(it, is, ik, i, j) - R(it, is, ik, i, j);
                  num += std::norm(da - db); den += std::norm(db); mx = std::max(mx, std::abs(da - db));
                }
        rel = std::sqrt(num) / std::max(std::sqrt(den), 1e-300);
      };
      const long nb = ncw;
      // (W) the Wannier frame on the nosym fixture
      nda::array<std::complex<double>, 2> V(4, 4), I4(4, 4);
      V() = std::complex<double>(0.0); I4() = std::complex<double>(0.0);
      for (long a = 0; a < 4; ++a) { V(a, a) = 1.0; I4(a, a) = 1.0; }
      auto givens = [&](long p, long q, double th, double ph) {
        const double c = std::cos(th), s = std::sin(th); const std::complex<double> eph(std::cos(ph), std::sin(ph));
        for (long i = 0; i < 4; ++i) { const auto vp = V(i, p), vq = V(i, q); V(i, p) = vp * c - vq * s * std::conj(eph); V(i, q) = vp * s * eph + vq * c; }
      };
      givens(0, 1, 0.7, 0.3); givens(1, 2, 1.1, -0.8); givens(2, 3, 0.4, 1.9); givens(0, 3, 0.9, 0.5);
      S5 S_r0, S_p1, S_pi, S_pv, S_a;
      const std::string ptsn = "coqui_d3_sigsw_A.secpts.h5", nu0n = "coqui_d3_sigsw_A.pol_nu0.g2.h5";
      double relI = 0.0, mxI = 0.0, relV = 0.0, mxV = 0.0, cp1 = 0.0;
      const bool do_w = (ncw == 4);
      if (do_w) {
        double ca = run_sw("A", mf, eri, 4, "", "", nullptr, true, S_a);           // band, dynamic: dumps the points + the nu0 column
        REQUIRE(std::filesystem::exists(ptsn)); REQUIRE(std::filesystem::exists(nu0n));
        double cr0 = run_sw("R0", mf, eri, 0, "", "", nullptr, false, S_r0);
        cp1 = run_sw("P1", mf, eri, 2, ptsn, nu0n, nullptr, false, S_p1);  // band frame on A's points, readout consumes A's column
        double cpi = run_sw("PI", mf, eri, 2, ptsn, nu0n, &I4, false, S_pi);      // Wannier V = 1 on the same points
        double cpv = run_sw("PV", mf, eri, 2, ptsn, nu0n, &V, false, S_pv);       // Wannier unitary mix V
        (void)ca;
        cdiff(S_pi, S_p1, S_r0, nb, S_r0.shape(2), relI, mxI);
        cdiff(S_pv, S_p1, S_r0, nb, S_r0.shape(2), relV, mxV);
        app_log(1, "dynbse_readout LFF-Sigma pair WANNIER gate: band frame vs Wannier V = 1: rel {:.3e} (max |d| {:.2e}); vs a unitary mix V of the window: "
                   "rel {:.3e} (max |d| {:.2e}); e_corr band {:+.10f} V=1 {:+.10f} V {:+.10f} (R0 {:+.10f})", relI, mxI, relV, mxV, cp1, cpi, cpv, cr0);
      } else {
        app_log(1, "dynbse_readout LFF-Sigma pair SYMW: window [0, {}) (COQUI_DYNBSE_TEST_SYMW_NC): the (W) Wannier part is skipped", ncw);
      }
      // (S) the symmetric mesh
      std::string fx = std::getenv("COQUI_DYNBSE_TEST_SIGPAIR_SYMW"); if (fx == "1" or fx.empty()) fx = "qe_lih222_sym";
      auto mfs = std::make_shared<mf::MF>(mf::default_MF(mpi_context, fx));
      thc_reader_t thcs(mfs, make_thc_reader_ptree(mfs->nbnd() * 8, "", "incore", "", "bdft", 1e-10, mfs->ecutrho(), 1, 1024));
      auto eris = mb_eri_t(thcs, thcs);
      app_log(1, "dynbse_readout LFF-Sigma pair SYM gate on {}: nkpts {} (IBZ {}), nqpts {} (IBZ {})", fx, mfs->nkpts(), mfs->nkpts_ibz(), mfs->nqpts(), mfs->nqpts_ibz());
      S5 S_sr0, S_sp1, S_sxs, S_sa;
      const std::string ptss = "coqui_d3_sigsw_SA.secpts.h5", nu0s = "coqui_d3_sigsw_SA.pol_nu0.g2.h5";
      double csa = run_sw("SA", mfs, eris, 4, "", "", nullptr, true, S_sa);       // the sym mesh's own star-closed points + nu0 column
      REQUIRE(std::filesystem::exists(ptss)); REQUIRE(std::filesystem::exists(nu0s));
      double csr0 = run_sw("SR0", mfs, eris, 0, "", "", nullptr, false, S_sr0);
      double csp1 = run_sw("SP1", mfs, eris, 2, ptss, nu0s, nullptr, false, S_sp1);   // the pair vertex on the sym mesh (IBZ externals)
      double csxs = run_sw("SXS", mfs, eris, 1, ptss, "", nullptr, false, S_sxs);     // B-S Sigma^{C,x} on the sym mesh at the same points
      S5 S_sp1i;
      ibz_cur = true;
      double csp1i = run_sw("SP1I", mfs, eris, 2, ptss, nu0s, nullptr, false, S_sp1i);  // P1: the same object from the IBZ solve + star fold
      ibz_cur = false;
      (void)csa;
      REQUIRE(S_sp1.shape(2) == mfs->nkpts_ibz());
      double relS, mxS;
      cdiff(S_sp1, S_sxs, S_sr0, nb, mfs->nkpts_ibz(), relS, mxS);
      app_log(1, "dynbse_readout LFF-Sigma pair SYM gate: one rung + static W on the SYMMETRIC mesh vs B-S Sigma^(C,x) (sym): rel Frobenius {:.3e} "
                 "(max |d| {:.2e}); e_corr SP1 {:+.10f} SXS {:+.10f} (SR0 {:+.10f}); nosym P1 {:+.10f} XS-equivalent identity on nosym: see section M",
              relS, mxS, csp1, csxs, csr0, cp1);
      if (do_w) { REQUIRE(relI < 1e-9); REQUIRE(relV < 1e-7); }   // the C-space object is gauge-invariant (the W-int point-frame gate holds at 1e-8)
      REQUIRE(relS < 1e-6);   // the symmetric path's own accuracy floor is the C-sector rotation unitarity (~1e-8 class)
      double relIB, mxIB;
      cdiff(S_sp1i, S_sp1, S_sr0, nb, mfs->nkpts_ibz(), relIB, mxIB);
      app_log(1, "dynbse_readout LFF-Sigma pair IBZ gate (P1): the IBZ solve + star fold vs the full-mesh units on {}: rel {:.3e} (max |d| {:.2e}); "
                 "e_corr {:+.10f} vs {:+.10f}", fx, relIB, mxIB, csp1i, csp1);
      if (std::getenv("COQUI_DYNBSE_TEST_SYMW_XREF")) {
        // P1 ARBITER: the NOSYM fixture (its own orbitals at every k, no rotation anywhere) on the SYM run's star-closed points is
        // the exact full-mesh object in another gauge; the gauge-invariant eigenvalues of the Hermitized C block of dSigma(tau, k)
        // at the IBZ points (matched by crystal coordinates) decide which sym path -- the full-mesh units or the IBZ fold -- is
        // closer to it. The two mean fields differ at the 1e-6 level (their GW baselines: R0 vs SR0), the arbiter's floor.
        S5 S_xr0, S_xp1;
        double cxr0 = run_sw("XR0", mf, eri, 0, "", "", nullptr, false, S_xr0);
        double cxp1 = run_sw("XP1", mf, eri, 2, ptss, "", nullptr, false, S_xp1);     // the sym points frozen; the readout solves its own nu0 column
        auto kn = mf->kpts_crystal(); auto ks = mfs->kpts_crystal();
        auto eig_of = [&](S5 const &S, S5 const &R, long ik) {   // the sorted eigenvalues of the Hermitized (S - R) C block, every tau
          std::vector<double> ev;
          for (long it = 0; it < S.shape(0); ++it) {
            nda::matrix<std::complex<double>> M(nb, nb);
            for (long i = 0; i < nb; ++i)
              for (long j = 0; j < nb; ++j) M(i, j) = 0.5 * ((S(it, 0, ik, i, j) - R(it, 0, ik, i, j)) + std::conj(S(it, 0, ik, j, i) - R(it, 0, ik, j, i)));
            auto [lam, V] = nda::linalg::eigenelements(M);
            for (long i = 0; i < nb; ++i) ev.push_back(double(lam(i)));
          }
          return ev;
        };
        double d_full = 0.0, d_ibz = 0.0, scale = 0.0;
        for (long ki = 0; ki < mfs->nkpts_ibz(); ++ki) {
          long kmatch = -1;
          for (long kk = 0; kk < mf->nkpts() and kmatch < 0; ++kk) {
            bool same = true;
            for (int d = 0; d < 3; ++d) { double x = ks(ki, d) - kn(kk, d); x -= std::round(x); if (std::abs(x) > 1e-8) same = false; }
            if (same) kmatch = kk;
          }
          REQUIRE(kmatch >= 0);
          auto en = eig_of(S_xp1, S_xr0, kmatch), ef = eig_of(S_sp1, S_sr0, ki), ei = eig_of(S_sp1i, S_sr0, ki);
          for (size_t i = 0; i < en.size(); ++i) {
            d_full = std::max(d_full, std::abs(ef[i] - en[i])); d_ibz = std::max(d_ibz, std::abs(ei[i] - en[i])); scale = std::max(scale, std::abs(en[i]));
          }
        }
        app_log(1, "dynbse_readout LFF-Sigma pair IBZ ARBITER (P1): eigenvalues of the Hermitized dSigma C block at the IBZ k, nosym exact vs the sym "
                   "full-mesh units: max |d| {:.3e}, vs the IBZ fold: max |d| {:.3e} (scale {:.3e}); e_corr nosym {:+.10f} (R0 {:+.10f}) sym full "
                   "{:+.10f} ibz {:+.10f} (SR0 {:+.10f})", d_full, d_ibz, scale, cxp1, cxr0, csp1, csp1i, csr0);
      }
      // P1 tolerance: the two sym paths carry the D-matrix accuracy of symmetry_rotation differently (the full-mesh path in the
      // transported legs of every pair, the IBZ path in the fold of the externals); on qe_lih222_sym with a closed window the
      // established class of the symmetry machinery is 5e-5 (test_vertex_ibz.cpp: "kernel-accuracy + D-matrix-accuracy +
      // O(leakage) class", REQUIRE < 5e-3); measured here 1.5e-4 at [0, 3). With the leaking default window [0, 4) the full-mesh
      // path is off by O(leakage) = 1.8e-2 and the check is informational.
      const bool closed_window = (ncw == 3 or ncw == 6);
      if (std::getenv("COQUI_DYNBSE_TEST_SYMW_STATIC_ONLY")) { if (closed_window) REQUIRE(relIB < 5e-3); mpi_context->comm.barrier(); return; }
      // P1, the dynamic path on the sym mesh: col static_dyn (y = 0 through vertex_sigma_dyn.icc, IBZ solve + fold) must equal
      // the static path's resummed column (col static) with the same outer W and IBZ fold (the N1 identity on a sym mesh)
      S5 S_sps, S_spsd;
      ibz_cur = true; col_cur = "static"; outer_cur = "static";
      double csps = run_sw("SPS", mfs, eris, 2, ptss, nu0s, nullptr, false, S_sps);
      col_cur = "static_dyn";
      double cspsd = run_sw("SPSD", mfs, eris, 2, ptss, nu0s, nullptr, false, S_spsd);
      ibz_cur = false; col_cur = "static1"; outer_cur = "static";
      double relSD, mxSD;
      cdiff(S_spsd, S_sps, S_sr0, nb, mfs->nkpts_ibz(), relSD, mxSD);
      app_log(1, "dynbse_readout LFF-Sigma dyn IBZ gate (P1): the dynamic path (static_dyn, IBZ solve + fold) vs the static path (static, IBZ) on {}: "
                 "rel {:.3e} (max |d| {:.2e}); e_corr {:+.10f} vs {:+.10f}", fx, relSD, mxSD, cspsd, csps);
      if (closed_window) { REQUIRE(relIB < 5e-3); REQUIRE(relSD < 1e-8); }
      else app_log(1, "dynbse_readout LFF-Sigma IBZ gates (P1): informational at the leaking window [0, 4) (run with COQUI_DYNBSE_TEST_SYMW_NC=3 for the REQUIREs)");
      mpi_context->comm.barrier();
      if (mpi_context->comm.root())
        for (auto const &e : std::filesystem::directory_iterator("."))
          if (e.path().filename().string().rfind("coqui_d3_sigsw_", 0) == 0) std::filesystem::remove(e.path());
      mpi_context->comm.barrier();
      return;
    }
    if (std::getenv("COQUI_DYNBSE_TEST_WINT")) { wint_gate(mf, eri, "nosym"); return; }
    if (std::getenv("COQUI_DYNBSE_TEST_SIGINTERP")) {
      // ---- P16 (vertex_perf_plan.md): the Wannier-frame interpolation of the pair Sigma vertex, same-mesh identity -------------
      //  I0  the pair vertex (window [0, 2) = the LiH projector's window, col static1, outer static) that DUMPS the Wannier-frame
      //      object with its R grid (pol_vertex_sigma_interp_dump = lih_wan.h5)
      //  I1  the same run CONSUMING I0's dump on the same mesh with the same projector: k -> R -> k is the identity on the coarse
      //      mesh and the projector is unitary on its window, so dSigma (hence Sigma) must be reproduced to rounding
      using S5 = nda::array<std::complex<double>, 5>;
      auto [outdir_w, prefix_w] = utils::utest_filename("qe_lih222");
      const std::string wan = outdir_w + "/lih_wan.h5";
      REQUIRE(std::filesystem::exists(wan));
      auto run_si = [&](std::string const &tag, std::string const &dump_proj, std::string const &interp, std::string const &proj_f, S5 &Sig) {
        const std::string out = "coqui_d3_siginterp_" + tag;
        solvers::hf_t hf; solvers::gw_t gw(&ft, "ignore_g0", out);
        solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
        simple_dyson dyson(mf.get(), &ft); MBState mb_state(mpi_context, ft, out);
        iter_scf::iter_scf_t iter_sol("damping");
        solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd());
        vtx.set_pol_vertex("ladder", "w0_prev", nda::range(0, 2), -1, 1e-8, -1.0, -1.0, -1.0, "none");
        vtx.set_ladder_rung("static", 1e-8, 30, 12, -1.0);
        vtx.set_isdf_points("", true);
        vtx.set_sigma_pair(true, "static1", "static", 1.0, true, false, "right");
        vtx.set_sigma_interp(dump_proj, interp, proj_f);
        scr_eri.set_vertex(&vtx);
        const double e_corr = std::get<1>(scf_loop(mb_state, dyson, eri, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 1, false, 1e-9, true));
        mpi_context->comm.barrier();
        { h5::file f(out + ".mbpt.h5", 'r'); h5::group g(f); auto it = g.open_group("scf").open_group("iter1"); nda::h5_read(it, "Sigma_tskij", Sig); }
        app_log(1, "dynbse_readout LFF-Sigma INTERP [{}]: e_corr {:.12f}, max|dSigma| {:.6e}", tag, e_corr, scr_eri.sigma_pair_meter()[0]);
        mpi_context->comm.barrier();
        return e_corr;
      };
      S5 S0, S1;
      const double c0 = run_si("I0", wan, "", "", S0);
      REQUIRE(std::filesystem::exists("coqui_d3_siginterp_I0.sigpair_wan.h5"));
      const double c1 = run_si("I1", "", "coqui_d3_siginterp_I0.sigpair_wan.h5", wan, S1);
      double d = 0.0, n = 0.0;
      for (long it = 0; it < S0.shape(0); ++it)
        for (long is = 0; is < S0.shape(1); ++is)
          for (long ik = 0; ik < S0.shape(2); ++ik)
            for (long i = 0; i < 2; ++i)
              for (long j = 0; j < 2; ++j) { d = std::max(d, std::abs(S0(it, is, ik, i, j) - S1(it, is, ik, i, j))); n = std::max(n, std::abs(S0(it, is, ik, i, j))); }
      app_log(1, "dynbse_readout LFF-Sigma INTERP gate (P16): the Wannier-frame dump consumed on the same mesh vs the solve: max |dSigma| {:.2e} "
                 "(max |Sigma| {:.3e}); e_corr {:+.12f} vs {:+.12f} (|d| {:.1e})", d, n, c1, c0, std::abs(c1 - c0));
      REQUIRE(d < 1e-10 * n);
      REQUIRE(std::abs(c1 - c0) < 1e-10);
      mpi_context->comm.barrier();
      if (mpi_context->comm.root())
        for (auto const &e : std::filesystem::directory_iterator("."))
          if (e.path().filename().string().rfind("coqui_d3_siginterp_", 0) == 0) std::filesystem::remove(e.path());
      mpi_context->comm.barrier();
      return;
    }
    if (std::getenv("COQUI_DYNBSE_TEST_CHAIN")) {
      // ---- P18 (vertex_perf_plan.md, 2026-09-21): the in-process vertex chain == the scripted chain of one-iteration restarts.
      //  CA  the seed: a dynamic Gamma_1 run (1 iteration) that dumps its points and its all-nu object CA.g1
      //  CC  pol_vertex_chain = true, 2 iterations restarted from CA's checkpoint: iteration 1 injects CA.g1 (the seed) and dumps
      //      CC.g1; iteration 2 injects CC.g1 (this run's previous dump) and dumps CC.g2
      //  CS1 1 iteration restarted from CA's checkpoint injecting CA.g1 (= CC's iteration 1)          -> CS1.g1 == CC.g1
      //  CS2 1 iteration restarted from CS1's checkpoint injecting CS1.g1 (= CC's iteration 2)        -> CS2.g1 == CC.g2
      // The damping mixer reads the previous iteration from the checkpoint, so the restarted state is the in-memory one -- with
      // mu re-solved at every Dyson step (const_mu = false): a restart's first Dyson step re-solves mu unconditionally.
      using cplx = std::complex<double>;
      namespace fs = std::filesystem;
      auto run_chain = [&](std::string const &tag, int niter, bool chain, std::string const &interp, std::string const &points,
                           std::string const &restart_from) {
        const std::string out = "coqui_d3_chain_" + tag;
        if (not restart_from.empty()) {
          mpi_context->comm.barrier();
          if (mpi_context->comm.root()) fs::copy_file(restart_from + ".mbpt.h5", out + ".mbpt.h5", fs::copy_options::overwrite_existing);
          mpi_context->comm.barrier();
        }
        solvers::hf_t hf; solvers::gw_t gw(&ft, "ignore_g0", out);
        solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
        simple_dyson dyson(mf.get(), &ft); MBState mb_state(mpi_context, ft, out);
        iter_scf::iter_scf_t iter_sol("damping");
        solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd());
        vtx.set_pol_vertex("ladder", "w0_prev", nda::range(0, 4), -1, 1e-8, -1.0, -1.0, -1.0, "none");
        vtx.set_ladder_rung("dynamic", 1e-8, 30, 12, -1.0);
        vtx.set_ladder_dyn_gamma1_only(true); vtx.set_ladder_dyn_all_nu(true);
        vtx.set_isdf_points(points, points.empty()); vtx.set_pol_interp(interp, "gam1"); vtx.set_pol_chain(chain);
        scr_eri.set_vertex(&vtx);
        // const_mu = false: a restart's initial Dyson step always re-solves mu (scf_loop), so the in-process loop must too
        auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, niter,
                                       not restart_from.empty(), 1e-9, false);
        app_log(1, "dynbse_readout CHAIN [{}]: chain {} niter {} restart {}: e_corr {:.12f}", tag, chain, niter,
                restart_from.empty() ? "no" : restart_from, e_corr);
        mpi_context->comm.barrier();
        return e_corr;
      };
      auto rd4 = [&](std::string const &fn, nda::array<cplx, 4> &A) { h5::file f(fn, 'r'); h5::group g(f); nda::h5_read(g, "Pi_gam1", A); };
      auto relmax4 = [&](nda::array<cplx, 4> const &A, nda::array<cplx, 4> const &B) {
        double d = 0.0, n = 0.0;
        for (long i = 0; i < A.size(); ++i) { d = std::max(d, std::abs(A.data()[i] - B.data()[i])); n = std::max(n, std::abs(B.data()[i])); }
        return (n > 0.0) ? d / n : d;
      };
      const std::string pts = "coqui_d3_chain_CA.secpts.h5", seed = "coqui_d3_chain_CA.pol_wh_dyn.g1.h5";
      run_chain("CA", 1, false, "", "", "");
      REQUIRE(fs::exists(pts)); REQUIRE(fs::exists(seed));
      const double ecc = run_chain("CC", 2, true, seed, pts, "coqui_d3_chain_CA");
      REQUIRE(fs::exists("coqui_d3_chain_CC.pol_wh_dyn.g1.h5")); REQUIRE(fs::exists("coqui_d3_chain_CC.pol_wh_dyn.g2.h5"));
      run_chain("CS1", 1, false, seed, pts, "coqui_d3_chain_CA");
      const double ecs2 = run_chain("CS2", 1, false, "coqui_d3_chain_CS1.pol_wh_dyn.g1.h5", pts, "coqui_d3_chain_CS1");
      nda::array<cplx, 4> C1, C2, S1, S2;
      rd4("coqui_d3_chain_CC.pol_wh_dyn.g1.h5", C1); rd4("coqui_d3_chain_CC.pol_wh_dyn.g2.h5", C2);
      rd4("coqui_d3_chain_CS1.pol_wh_dyn.g1.h5", S1); rd4("coqui_d3_chain_CS2.pol_wh_dyn.g1.h5", S2);
      const double d1 = relmax4(C1, S1), d2 = relmax4(C2, S2), d12 = relmax4(C2, C1);
      app_log(1, "dynbse_readout CHAIN gate (P18): in-process chain vs one-iteration restarts on qe_lih222: |dPi_gam1| iteration 1 {:.2e}, "
                 "iteration 2 {:.2e} (the chain moved the object by {:.2e} between the iterations); e_corr chain {:.12f} restarts {:.12f} "
                 "(|d| {:.1e})", d1, d2, d12, ecc, ecs2, std::abs(ecc - ecs2));
      REQUIRE(d1 < 1e-12);
      REQUIRE(d2 < 1e-10);
      REQUIRE(d12 > 1e-6);                                   // the second iteration did consume a different (its own) object
      REQUIRE(std::abs(ecc - ecs2) < 1e-10);
      mpi_context->comm.barrier();
      if (mpi_context->comm.root())
        for (auto const &e : fs::directory_iterator("."))
          if (e.path().filename().string().rfind("coqui_d3_chain_", 0) == 0) fs::remove(e.path());
      mpi_context->comm.barrier();
      return;
    }
    if (const char *sw = std::getenv("COQUI_DYNBSE_TEST_SYMEPS")) {
      // ---- 2026-09-22: the P-side static-ladder eps_M readout, SYM vs NOSYM on LiH at window [0, nc) (nc = the env value) -------
      // A fast reproducer of the Si 4^3 finding (the symmetric ladder correction 7 % below the full-mesh one): one scGW iteration
      // from the DFT start on qe_lih222 and on qe_lih222_sym with the static ladder readout (its own point selection on each
      // mesh), the RPA and +ladder eps_M at q_min compared; the RPA agreement is the mean-field/THC floor, the ladder correction
      // is what the symmetry path must reproduce. COQUI_DYNBSE_TEST_SYMEPS_RUNG=dynamic adds the dynamic-rung columns.
      const long ncw = std::atol(sw) > 0 ? std::atol(sw) : 6;
      const std::string rung = std::getenv("COQUI_DYNBSE_TEST_SYMEPS_RUNG") ? std::getenv("COQUI_DYNBSE_TEST_SYMEPS_RUNG") : "static";
      auto run_se = [&](std::string const &fx, std::string const &tag) {
        auto mfw = std::make_shared<mf::MF>(mf::default_MF(mpi_context, fx));
        thc_reader_t thcw(mfw, make_thc_reader_ptree(mfw->nbnd() * 8, "", "incore", "", "bdft", 1e-10, mfw->ecutrho(), 1, 1024));
        auto eriw = mb_eri_t(thcw, thcw);
        const std::string out = "coqui_d3_symeps_" + tag;
        solvers::hf_t hf; solvers::gw_t gw(&ft, "ignore_g0", out);
        solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
        simple_dyson dyson(mfw.get(), &ft); MBState mb_state(mpi_context, ft, out);
        iter_scf::iter_scf_t iter_sol("damping");
        solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mfw->nbnd());
        // COQUI_DYNBSE_TEST_SYMEPS_THRESH: the SECONDARY selection threshold (default: the reader's; 1e-13 -> the complete pair
        // rank, so the two meshes share no truncation error); COQUI_DYNBSE_TEST_SYMEPS_FREEZE=1: the sym run takes the nosym
        // run's points (the same secondary frame on both meshes)
        const double sth = std::getenv("COQUI_DYNBSE_TEST_SYMEPS_THRESH") ? std::atof(std::getenv("COQUI_DYNBSE_TEST_SYMEPS_THRESH")) : -1.0;
        const bool freeze = (tag == "sym") and std::getenv("COQUI_DYNBSE_TEST_SYMEPS_FREEZE");
        vtx.set_pol_vertex("ladder", "w0_prev", nda::range(0, ncw), -1, 1e-8, sth, -1.0, -1.0, "none");
        vtx.set_ladder_rung(rung, 1e-8, 30, 12, -1.0);
        if (rung == "dynamic") vtx.set_ladder_dyn_gamma1_only(true);
        vtx.set_isdf_points(freeze ? "coqui_d3_symeps_nosym.secpts.h5" : "", not freeze);
        scr_eri.set_vertex(&vtx);
        const double e_corr = std::get<1>(scf_loop(mb_state, dyson, eriw, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 1, false, 1e-9, true));
        auto [er, el] = scr_eri.pol_eps_readout();
        auto ed = scr_eri.pol_eps_dyn();
        app_log(1, "dynbse_readout SYMEPS [{}] {} (nk {} IBZ {}, window [0, {}), rung {}): e_corr {:.10f}, eps_M(q_min) RPA {:.8f} +ladder {:.8f} (Delta {:+.8f}); dyn columns {:.6f} {:.6f} {:.6f} {:.6f}",
                tag, fx, mfw->nkpts(), mfw->nkpts_ibz(), ncw, rung, e_corr, er, el, el - er, ed[0], ed[1], ed[2], ed[3]);
        mpi_context->comm.barrier();
        if (mpi_context->comm.root()) { remove((out + ".mbpt.h5").c_str()); for (auto const &e : std::filesystem::directory_iterator(".")) if (e.path().filename().string().rfind(out + ".", 0) == 0 and e.path().filename().string().find(".secpts.h5") == std::string::npos) std::filesystem::remove(e.path()); }
        mpi_context->comm.barrier();
        return std::make_tuple(e_corr, er, el, ed);
      };
      // COQUI_DYNBSE_TEST_SYMEPS_FX = lih222 (default) | lih223 (a 2x2x3 mesh: non-TRIM k = +-1/3, time-reversal pairs when the
      // group lacks inversion -- qe_lih223_sym) | lih223inv (inversion only)
      const std::string fxs = std::getenv("COQUI_DYNBSE_TEST_SYMEPS_FX") ? std::getenv("COQUI_DYNBSE_TEST_SYMEPS_FX") : "lih222";
      // si333: the C3v x TIME-REVERSAL combination (6 IBZ k of 27) against its own full mesh -- the cheap reproducer of the
      // Si 4^3 production finding; si444trev: the production mesh reduced by time reversal ALONE (36 k of 64) against the
      // symmetric one (13 k) -- both added 2026-09-22 for the time-reversal hunt.
      const std::string fx_ns = (fxs == "lih222") ? "qe_lih222" : (fxs == "si222") ? "qe_si222_nosym"
                              : (fxs == "si333") ? "qe_si333_nosym" : (fxs == "si444trev") ? "qe_si444_trevonly" : "qe_lih223";
      const std::string fx_s = (fxs == "lih222") ? "qe_lih222_sym" : (fxs == "si222") ? "qe_si222_sym"
                             : (fxs == "si333") ? "qe_si333_sym" : (fxs == "si444trev") ? "qe_si444_sym"
                             : (fxs == "lih223inv") ? "qe_lih223_inv" : "qe_lih223_sym";
      auto [cn, rn, ln, dn] = run_se(fx_ns, "nosym");
      auto [cs, rs, ls, ds] = run_se(fx_s, "sym");
      app_log(1, "dynbse_readout SYMEPS gate (window [0, {}), rung {}): RPA eps_M sym vs nosym rel {:.3e}; the ladder correction Delta: nosym {:+.8f} sym {:+.8f} -> rel {:.3e} of the correction; "
                 "e_corr {:.10f} vs {:.10f}; Gamma_1 column sym/nosym {:.6f}/{:.6f}",
              ncw, rung, std::abs(rs - rn) / rn, ln - rn, ls - rs, std::abs((ls - rs) - (ln - rn)) / std::abs(ln - rn), cs, cn, ds[2], dn[2]);
      mpi_context->comm.barrier();
      return;
    }
    if (std::getenv("COQUI_DYNBSE_TEST_WINT_SYM")) {
      // the fixture name is the env value ("1" = qe_lih222_sym; use qe_lih223_sym for a mesh with non-TRIM k-points)
      std::string fx = std::getenv("COQUI_DYNBSE_TEST_WINT_SYM"); if (fx == "1" or fx.empty()) fx = "qe_lih222_sym";
      auto mfs = std::make_shared<mf::MF>(mf::default_MF(mpi_context, fx));
      thc_reader_t thcs(mfs, make_thc_reader_ptree(mfs->nbnd() * 8, "", "incore", "", "bdft", 1e-10, mfs->ecutrho(), 1, 1024));
      auto eris = mb_eri_t(thcs, thcs);
      app_log(1, "dynbse_readout W-int gate on the SYMMETRIC mesh: nkpts {} (IBZ {})", mfs->nkpts(), mfs->nkpts_ibz());
      wint_gate(mfs, eris, "sym"); return;
    }

    auto run = [&](std::string const &rung, int niter, bool dump = false, bool dense = true, long ustride = 1) {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd());
      vtx.set_pol_vertex("ladder", "w0_prev", nda::range(0, 4), -1, 1e-8, -1.0, -1.0, -1.0);
      vtx.set_ladder_rung(rung, 1e-8, 30, 12, -1.0);
      vtx.set_eps_cut(1, 3);                    // the eps(q_i, i nu) cut report (report-only; q_min and the 3 lowest nodes
                                                //  for the dynamic columns keep the test short)
      vtx.set_ladder_dyn_dump(dump);            // per-unit dump / restart files of the dynamic solves
      vtx.set_ladder_dyn_dense(dense);          // the dense per-tau rung (default) vs the THC streaming route
      vtx.set_ladder_dyn_union_stride(ustride); // the inu != 0 union grid's G-node stride
      if (char const *vp = std::getenv("COQUI_DYNBSE_TEST_VPREC")) vtx.set_ladder_dyn_iaft_prec(vp);   // vertex-local DLR precision
      if (char const *tf = std::getenv("COQUI_DYNBSE_TEST_TFOLD")) vtx.set_ladder_dyn_tfold(std::atof(tf));   // the small-nu fold ratio
      if (std::getenv("COQUI_DYNBSE_TEST_G1")) vtx.set_ladder_dyn_gamma1_only(true);   // Gamma_1 only (skip the resummation GMRES)
      if (auto *rm = std::getenv("COQUI_DYNBSE_TEST_RESOLVENT")) vtx.set_ladder_dyn_resolvent(rm);   // P7: inverse | lu
      if (auto *ac = std::getenv("COQUI_DYNBSE_TEST_SIGDYN_ACC")) vtx.set_sigma_dyn_acc(ac);   // P4-C14: split | single
      if (auto *wc = std::getenv("COQUI_DYNBSE_TEST_WCACHE")) vtx.set_wcache_mode(wc);   // P19: replicated | shared
      scr_eri.set_vertex(&vtx);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     niter, false, 1e-9, true);
      auto [er, el] = scr_eri.pol_eps_readout();
      auto ed = scr_eri.pol_eps_dyn();
      const double ritz = scr_eri.pol_dyn_ritz();
      nda::array<double, 2> cut(scr_eri.pol_eps_cut_qmin());
      const double eloop = scr_eri.pol_eps_loop();
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return std::make_tuple(e_hf, e_corr, er, el, ed, ritz, cut, eloop);
    };
    auto [h0, c0, r0, l0, d0, z0, cut0, eloop0] = run("static", 2);
    auto [h1, c1, r1, l1, d1, z1, cut1, eloop1] = run("dynamic", 2, true);     // writes the unit dump files
    auto [h2, c2, r2, l2, d2, z2, cut2, eloop2] = run("dynamic", 2, true);     // loads them: no unit is re-solved
    auto [h3, c3, r3, l3, d3, z3, cut3, eloop3] = run("dynamic", 2, false, false);   // the THC streaming rung
    auto [h4, c4, r4, l4, d4, z4, cut4, eloop4] = run("dynamic", 2, false, true, 2);  // union stride 2 at inu != 0
    if (mpi_context->comm.root()) {
      app_log(1, "dynbse_readout: union stride 2 vs 1 at nodes 1, 2: resummed {} vs {}, {} vs {}", cut4(1, 7), cut1(1, 7),
              cut4(2, 7), cut1(2, 7));
      for (long j = 1; j < 3; ++j)
        for (int c = 4; c < 8; ++c) REQUIRE(std::abs(cut4(j, c) - cut1(j, c)) < 2e-3 * std::abs(cut1(j, c) - 1.0));
    }
    {
      // the dense per-tau rung and the THC streaming route are the same operator (solve-tolerance class)
      app_log(1, "dynbse_readout: dense vs THC rung: resummed {} vs {}, Gamma1 {} vs {}", d1[3], d3[3], d1[2], d3[2]);
      for (int c = 0; c < 4; ++c) REQUIRE(std::abs(d3[c] - d1[c]) < 1e-7 * std::max(1.0, std::abs(d1[c] - 1.0)));
      if (mpi_context->comm.root())
        for (long j = 0; j < 3; ++j)
          for (int c = 4; c < 8; ++c) REQUIRE(std::abs(cut3(j, c) - cut1(j, c)) < 1e-6 * std::max(1.0, std::abs(cut1(j, c) - 1.0)));
    }
    {
      // the restart path: every dynamic column is bitwise what the solving run produced
      for (int c = 0; c < 4; ++c) REQUIRE(d2[c] == d1[c]);
      if (mpi_context->comm.root())
        for (long j = 0; j < cut1.shape(0); ++j)
          for (int c = 4; c < 8; ++c) REQUIRE(cut2(j, c) == cut1(j, c));
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) {
        namespace fs = std::filesystem;
        for (auto const &e : fs::directory_iterator(fs::current_path())) {
          const std::string fn = e.path().filename().string();
          if (fn.rfind(output + ".dynunits.", 0) == 0) fs::remove(e.path());
        }
      }
      mpi_context->comm.barrier();
    }
    // the eps(q_i, i nu) cut (rank 0): node 0 of the q_min cut IS the readout (same rows, same
    // Dyson; the ladder node 0 of the whalf pass vs eval_pol_ladder_nu0 = the node-map class),
    // the loop-side column at node 0 IS the Q3 loop-side value, every node is finite and the
    // static screening decays along i nu (eps_M(i nu_max) < eps_M(0)).
    if (mpi_context->comm.root()) {
      REQUIRE(cut0.shape(0) > 1);
      REQUIRE(cut0.shape(1) == 8);
      app_log(1, "dynbse_readout: eps-cut q_min node 0: RPA {} (readout {}), +ladder {} (readout {}), loop {} (Q3 {}); "
                 "last node: RPA {} +ladder {} loop {}", cut0(0, 0), r0, cut0(0, 1), l0, cut0(0, 3), eloop0,
              cut0(cut0.shape(0) - 1, 0), cut0(cut0.shape(0) - 1, 1), cut0(cut0.shape(0) - 1, 3));
      REQUIRE(std::abs(cut0(0, 0) - r0) < 1e-10);
      REQUIRE(std::abs(cut0(0, 1) - l0) < 1e-8);
      REQUIRE(std::abs(cut0(0, 3) - eloop0) < 1e-10);
      for (long j = 0; j < cut0.shape(0); ++j) {
        REQUIRE(std::isfinite(cut0(j, 0))); REQUIRE(cut0(j, 0) > 0.0);
        REQUIRE(std::isfinite(cut0(j, 1))); REQUIRE(cut0(j, 1) > 0.0);
        REQUIRE(std::isfinite(cut0(j, 3))); REQUIRE(cut0(j, 3) > 0.0);
        REQUIRE(cut0(j, 2) == -1.0);            // legs = bare: no DeltaLambda column
      }
      REQUIRE(cut0(cut0.shape(0) - 1, 0) < cut0(0, 0));
      REQUIRE(cut0(cut0.shape(0) - 1, 1) < cut0(0, 1));
      // the cut is report-only: bitwise across the rung modes
      REQUIRE(cut1.shape(0) == cut0.shape(0));
      for (long j = 0; j < cut0.shape(0); ++j)
        for (long c = 0; c < 4; ++c) REQUIRE(cut1(j, c) == cut0(j, c));
      // the dynamic-rung cut (union basis, GMRES(12), all half nodes): node 0 agrees with the
      // inu = 0 readout columns (shared grid) to the refit class, every node is finite, and the
      // resummed column is nu-continuous at the first positive node
      app_log(1, "dynbse_readout: dynamic cut q_min node 0: static {} (nu0 {}), +dyn1 {} (nu0 {}), Gamma1 {} (nu0 {}), "
                 "resummed {} (nu0 {}); node 1 resummed {}; last node resummed {}",
              cut1(0, 4), d1[0], cut1(0, 5), d1[1], cut1(0, 6), d1[2], cut1(0, 7), d1[3], cut1(1, 7),
              cut1(cut1.shape(0) - 1, 7));
      for (long j = 0; j < 3; ++j)
        for (int c = 4; c < 8; ++c) { REQUIRE(std::isfinite(cut1(j, c))); REQUIRE(cut1(j, c) > 0.0); }
      for (long j = 3; j < cut1.shape(0); ++j)
        for (int c = 4; c < 8; ++c) REQUIRE(cut1(j, c) == -1.0);        // beyond the 3 requested nodes
      for (int c = 0; c < 4; ++c) REQUIRE(std::abs(cut1(0, 4 + c) - d1[c]) < 1e-3 * std::abs(d1[c] - 1.0));
      REQUIRE(std::abs(cut1(1, 7) - cut1(0, 7)) < 0.05 * std::abs(cut1(0, 7) - 1.0));
      REQUIRE(cut1(2, 7) < cut1(0, 7));
      for (long j = 0; j < cut0.shape(0); ++j)
        for (int c = 4; c < 8; ++c) REQUIRE(cut0(j, c) == -1.0);
    }
    app_log(1, "dynbse_readout: static rung: e_corr {} eps RPA {} +ladder {} ; dynamic rung: e_corr {} eps RPA {} "
               "+ladder(L2) {} ; +static(sign-corr.) {} +static+Pi^C_dyn {} +Gamma1 {} +resummed {} ; Ritz {}",
            c0, r0, l0, c1, r1, l1, d1[0], d1[1], d1[2], d1[3], z1);
    // the loop and the historic columns are bitwise (the dynamic rung is a readout-only column)
    REQUIRE(h1 == h0);
    REQUIRE(c1 == c0);
    REQUIRE(r1 == r0);
    REQUIRE(l1 == l0);
    REQUIRE(d0[3] == -1.0);
    for (double v : d1) { REQUIRE(std::isfinite(v)); REQUIRE(v > 0.0); }
    REQUIRE(z1 >= 0.0);
    REQUIRE(z1 < 1.0);
    // 2026-09-11: pair_space_ladder resums the derived resolvent (1 + Xh Kt)^-1, so the dynbse
    // driver's static column IS the L2 readout (gate A class, through the same upfold + Dyson)
    REQUIRE(std::abs(d1[0] - l1) < 1e-8);
    // the dynamic rungs move eps_M away from the static ladder
    REQUIRE(d1[3] != d1[0]);
#endif
  }

} // namespace bdft_tests
