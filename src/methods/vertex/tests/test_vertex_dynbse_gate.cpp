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
        if (V) { auto proj = make_degenerate_projector(*mfw, 0, 4, V, trs_images); vtx.set_wannier_projector(proj, true); }
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
          auto run_p = [&](std::string const &tag, int kind, std::string const &col, std::string const &outer, double scale, S5 &Sig) {
            // kind: 0 = plain GW, 1 = B-S Sigma^{C,x}, 2 = the pair vertex
            const std::string out = "coqui_d3_winj_" + tag;
            solvers::hf_t hf; solvers::gw_t gw(&ft, "ignore_g0", out);
            solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
            simple_dyson dyson(mfn.get(), &ft); MBState mb_state(mpi_context, ft, out);
            iter_scf::iter_scf_t iter_sol("damping");
            std::array<double, 4> m{0.0, 0.0, 0.0, 0.0}; std::array<double, 2> d{0.0, 0.0};
            double e_corr = 0.0, pichk = -1.0;
            if (kind == 1) {
              solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(0, 4), mfn->nbnd(), "ignore_g0", "secondary", -1, 1e-8, -1.0, -1.0, "static");
              vtx.set_isdf_points("coqui_d3_winj_G.secpts.h5", false);
              vtx.set_bl_drop(1);
              scr_eri.set_vertex(&vtx); gw.set_vertex(&vtx);
              e_corr = std::get<1>(scf_loop(mb_state, dyson, erin, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 1, false, 1e-9, true));
            } else if (kind == 2) {
              solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mfn->nbnd());
              vtx.set_pol_vertex("ladder", "w0_prev", nda::range(0, 4), -1, 1e-8, -1.0, -1.0, -1.0, "none");
              vtx.set_ladder_rung("static", 1e-8, 30, 12, -1.0);
              vtx.set_isdf_points("coqui_d3_winj_G.secpts.h5", false);
              vtx.set_sigma_pair(true, col, outer, scale, true, true);
              scr_eri.set_vertex(&vtx);
              e_corr = std::get<1>(scf_loop(mb_state, dyson, erin, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 1, false, 1e-9, true));
              m = scr_eri.sigma_pair_meter(); d = gw.sigma_pair_dsigma();
              if (col == "static") {   // G1: the P side's own object from the same amplitudes vs the dump's static column
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
                if (e.path().filename().string().rfind(out + ".", 0) == 0) std::filesystem::remove(e.path());
            }
            mpi_context->comm.barrier();
            return std::make_tuple(e_corr, m, d, pichk);
          };
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
    if (std::getenv("COQUI_DYNBSE_TEST_WINT")) { wint_gate(mf, eri, "nosym"); return; }
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
