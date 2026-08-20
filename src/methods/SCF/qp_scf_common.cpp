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


#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "nda/linalg.hpp"
#include "nda/linalg/det_and_inverse.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/ac/AC_t.hpp"

#include "hamiltonian/one_body_hamiltonian.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "mean_field/MF.hpp"
#include "utilities/mpi_context.h"
#include "methods/tools/chkpt_utils.h"
#include "methods/ERI/mb_eri_context.h"
#include "methods/SCF/mb_solver_t.h"
#include "methods/SCF/scf_common.hpp"
#include "methods/SCF/qp_solvers.hpp"
#include "methods/SCF/qp_maps_matsubara.hpp"
#include "methods/SCF/qp_modea.hpp"
#include "methods/SCF/wc_band_elements.hpp"

namespace methods {

namespace {
  // The Matsubara-native maps (qp_map != "ac_pade") consume Sigma(iw) on the
  // POSITIVE fermionic nodes in ascending order; the sampling mesh interleaves
  // signs, so select and sort once. Returns (w_n values, row indices into the
  // wn_mesh-ordered frequency axis).
  auto positive_wn_nodes(nda::array<ComplexType, 1> const &iw_mesh) {
    std::vector<std::pair<double, long>> pw;
    for (long n = 0; n < iw_mesh.shape(0); ++n)
      if (iw_mesh(n).imag() > 0.0) pw.emplace_back(iw_mesh(n).imag(), n);
    std::sort(pw.begin(), pw.end());
    utils::check(pw.size() >= 2, "qp_map matsubara: found {} positive fermionic nodes, need >= 2.", pw.size());
    nda::array<double, 1> wp(pw.size());
    std::vector<long> idx(pw.size());
    for (size_t m = 0; m < pw.size(); ++m) {
      wp(m) = pw[m].first;
      idx[m] = pw[m].second;
    }
    return std::make_pair(std::move(wp), std::move(idx));
  }

  /**
   * Project 2 increment QM3: build the mode-A evaluator context, if and only if the map asks
   * for it. MUST be called inside the live-W window of the caller (mb_state.dW_qtPQ alive).
   * `need_diag` also produces the replicated DIAGONAL residues, which the evGW leg needs
   * because solve_qp_eqn distributes (s,k,a) on a processor grid of its own.
   */
  template<typename eri_t, typename corr_solver_t>
  void build_modea_ctx_if_needed(qp_modea::modea_ctx &ctx, MBState &mb_state,
                                 solvers::mb_solver_t<corr_solver_t> &mb_solver, eri_t &eri,
                                 const sArray_t<Array_view_4D_t> &sMO_skia,
                                 const sArray_t<Array_view_3D_t> &sE_ska,
                                 double mu, const imag_axes_ft::IAFT &FT,
                                 qp_params_t &qp_params, bool need_diag) {
    if (qp_params.qp_map != "mode_a" and qp_params.qp_map != "mode_b") return;
    qp_modea::modea_opts opts;
    opts.route = qp_params.qp_modea_route;
    opts.nconsist = qp_params.qp_modea_nconsist;
    opts.consist_tol = qp_params.qp_modea_consist_tol;
    opts.eta = qp_params.qp_modea_eta;
    opts.eta_far = qp_params.qp_modea_eta_far;
    opts.wsupp = qp_params.qp_modea_wsupp;
    opts.wfit = qp_params.qp_modea_wfit;
    opts.wrtol = qp_params.qp_modea_wrtol;
    opts.wrank = qp_params.qp_modea_wrank;
    opts.wsketch = qp_params.qp_modea_wsketch;
    opts.wunion = qp_params.qp_modea_wunion;
    opts.iter = 1;
    std::string div = "ignore_g0";
    if constexpr (requires { mb_solver.corr->iter(); })
      if (mb_solver.corr != nullptr) opts.iter = mb_solver.corr->iter();
    if constexpr (requires { mb_solver.corr->div_treatment(); })
      if (mb_solver.corr != nullptr) div = mb_solver.corr->div_treatment();
    utils::check(opts.route == "cd" or opts.route == "expansion",
                 "qp_modea: unknown qp_modea_route = {}. Valid: \"cd\", \"expansion\".",
                 opts.route);
    utils::check(opts.wfit == "tau" or opts.wfit == "nu",
                 "qp_modea: unknown qp_modea_wfit = {}. Valid: \"tau\", \"nu\".", opts.wfit);
    utils::check(opts.eta_far >= 0.0,
                 "qp_modea: qp_modea_eta_far = {} must be >= 0 (0 = the rev-3.1 mu fallback).",
                 opts.eta_far);

    app_log(2, "\n* {} quasiparticle map (Project 2 increment QM3): building the "
               "evaluator context", qp_params.qp_map);
    if (opts.eta_far > 0.0)
      app_log(2, "  - FAR-STATE EVALUATION (spec rev 4): states outside (VBM - 0.95 E_PH, "
                 "CBM + 0.95 E_PH) are evaluated at z = eps + i eta_far with eta_far = "
                 "{:.4e} a.u. ({:.4g} eV); in-strip states stay exact (eta = 0). Census and "
                 "the pole-spacing validity floor are logged below.",
              opts.eta_far, opts.eta_far * 27.211386245988);
    else if (qp_params.qp_map == "mode_a")
      app_log(2, "  - evaluation energies are STRIP-CLAMPED TO mu (spec rev 3.1, addendum "
                 "item 2): states inside (VBM - 0.95 E_PH, CBM + 0.95 E_PH) are exact mode A, "
                 "states outside it are evaluated at mu; census logged below.");
    if (opts.route == "expansion") {
      // pure diagnostic: the whole map is solved from the route-A z0 = 0 re-expansion of the
      // stored Sigma(iw), so no W data (and no THC contraction) is needed at all.
      ctx = qp_modea::modea_ctx{};
      ctx.opts = opts;
      ctx.active = true;
      ctx.have_cd = false;
      ctx.beta = FT.beta();
      ctx.mu = mu;
      ctx.eta = opts.eta;
      ctx.eta_far = opts.eta_far;
      ctx.ns = sE_ska.shape()[0];
      ctx.nk = sE_ska.shape()[1];
      ctx.nbnd = sE_ska.shape()[2];
      app_log(2, "  - route:                       expansion (DIAGNOSTIC -- Sigma^c comes "
                 "from the route-A z0 = 0 fit, no W^c contraction)");
      return;
    }
    if constexpr (std::is_same_v<std::decay_t<eri_t>, thc_reader_t>) {
      qp_modea::build_modea_context(ctx, mb_state, eri, sMO_skia, sE_ska, mu, FT, opts, div,
                                    need_diag);
    } else {
      utils::check(false, "qp_modea: qp_map = \"mode_a\" with qp_modea_route = \"cd\" needs a "
                          "THC ERI (the W^c band elements are collocation pair vectors). Use "
                          "a thc eri, or qp_modea_route = \"expansion\".");
    }
  }

  // ---------------------------------------------------------------------------------------
  // Project 2 increment QM3: the mode-A driver (notes/qm3_mode_a_loop_spec.md sections 4/5/7)
  //
  // Runs, per external (s,k) block owned by this rank:
  //   (1) the inner QP-consistency loop at FIXED Sigma data -> the LAST V^xc;
  //   (2) THE ANCHOR (gate QM3-b(ii)): route-B Sigma^c_ab at the first few FERMIONIC nodes
  //       vs the gathered solver Sigma(i w_n). Pins prefactor, spin, q-star/trev rule, the
  //       MO rotation and the head in one number;
  //   (3) the A/B cross-validation harness of spec section 5: delta_i = |Sigma^A_ii(eps_i) -
  //       Sigma^B_ii(eps_i)| against the A-side TRUNCATION CLASS |Sigma^A_(p+1) - Sigma^A_(p)|,
  //       plus one off-diagonal spot check.
  // Everything here is LOGGED; the only hard failure is a route/knob misuse.
  // ---------------------------------------------------------------------------------------
  template<typename comm_t>
  void modea_run(sArray_t<Array_view_4D_t> &sVcorr_skij,
                 const sArray_t<Array_view_5D_t> &sSigma_wskab,
                 const sArray_t<Array_view_5D_t> &sSigma_tskab,
                 const sArray_t<Array_view_4D_t> &sMO_skia,
                 const sArray_t<Array_view_3D_t> &sE_ska,
                 const sArray_t<Array_view_4D_t> &sHstat_skij,
                 nda::array<double, 1> const &wp, std::vector<long> const &widx,
                 double mu, const imag_axes_ft::IAFT &FT, std::string const &map_name,
                 const qp_modea::modea_ctx *ctx, comm_t &comm) {
    using namespace qp_modea;
    decltype(nda::range::all) all;
    constexpr double HA2EV = 27.211386245988;
    auto [ns, nkpts, nbnd] = sE_ska.shape();
    const long npos = wp.shape(0);
    const int lvl = 2;

    utils::check(ctx != nullptr and ctx->active,
                 "qp_approx: qp_map = \"mode_a\" reached the map without an evaluator "
                 "context. The context is built in the live W window of add_qpscf_vcorr / "
                 "add_evscf_vcorr and requires a THC ERI.");
    const bool cd = ctx->have_cd;
    const bool mode_b = (map_name == "mode_b");

    // owned blocks: the cd route inherits the context's partition, the diagnostic
    // "expansion" route round-robins the same way.
    std::vector<std::pair<long, long>> mine;
    if (cd) {
      for (auto const &b : ctx->blocks) mine.emplace_back(b.is, b.ik);
    } else {
      for (long sk = comm.rank(); sk < long(ns * nkpts); sk += comm.size())
        mine.emplace_back(sk / nkpts, sk % nkpts);
    }

    // the A-side window: order p and p+1 on the SAME node window (spec section 5)
    sigma_real_axis::fit_opts optA, optA1;
    optA.p = 2;
    optA1.p = 3;
    optA.m = optA1.m = std::min<long>(3 * (optA1.p), npos);
    utils::check(optA.m >= optA1.p + 1,
                 "qp_approx (mode_a): only {} positive fermionic nodes; the A-side "
                 "truncation-class fit needs at least {}.", npos, optA1.p + 1);

    double anchor_worst = 0.0, anchor_expect = ctx->diag.rec_rel_worst;
    double dmax_worst = 0.0, min_den_worst = 1e300, anti_worst = 0.0;
    double ratio_worst = 0.0, dev_off_worst = 0.0;
    double ratio_in_worst = 0.0, delta_in_worst = 0.0, class_in_worst = 0.0;
    double tau_dev_worst = 0.0;
    // WHERE the gate quantity is attained: the (s,k) block and the element of the probed set
    // that carries max|Sigma_B - Sigma^GW|. Reduced together with the value (see below) so
    // the abort names its own element -- the kp444 false fire of 2026-08-13 was invisible
    // precisely because the offending block was owned by a non-root rank and app_log is
    // root-only, while the gate maxes over every rank's blocks.
    long tau_dev_is = -1, tau_dev_ik = -1, tau_dev_a = -1, tau_dev_b = -1;
    // the per-isym anchor breakdown, kept for the gate message (see the TAU ISYM block)
    double isym_ratio_worst = 0.0;
    long isym_class_worst = -1;
    long iters_worst = 0, n_noconv = 0, n_flag = 0, n_flag_in = 0;
    long n_fallback = 0, n_fallback_win = 0, n_sanity_trip = 0;
    long n_homo_fb = 0, n_lumo_fb = 0, n_blocks = 0;
    // mode_a STRIP CLAMP census (spec rev 3 addendum item 2)
    long n_clamp = 0, n_clamp_win = 0, n_eval = 0, n_homo_cl = 0, n_lumo_cl = 0;
    double exc_lo_worst = 0.0, exc_hi_worst = 0.0;
    // rev 4: the graded-eta far-state census
    long n_eta = 0, n_anti_in = 0;
    double im_off_worst = 0.0, anti_in_worst = 0.0, spacing_worst = 0.0;
    const qp_modea::strip_t strip = qp_modea::strip_of(*ctx);
    const double eta_far = ctx->eta_far;

    nda::array<ComplexType, 2> tmp(nbnd, nbnd), Hstat_ab(nbnd, nbnd), V(nbnd, nbnd);
    nda::array<ComplexType, 3> Sw(optA.m, nbnd, nbnd);
    nda::array<ComplexType, 2> Sanch(nbnd, nbnd);
    nda::array<double, 1> eps(nbnd), eps_rel(nbnd);

    for (auto const &[is, ik] : mine) {
      auto MO = sMO_skia.local()(is, ik, all, all);
      nda::blas::gemm(ComplexType(1.0), sHstat_skij.local()(is, ik, nda::ellipsis{}), MO,
                      ComplexType(0.0), tmp);
      nda::blas::gemm(ComplexType(1.0), nda::dagger(MO), tmp, ComplexType(0.0), Hstat_ab);

      for (long m = 0; m < optA.m; ++m)
        Sw(m, all, all) = sSigma_wskab.local()(widx[m], is, ik, all, all);
      auto XA = sigma_real_axis::fit_matsubara(Sw, wp, optA);
      auto XA1 = sigma_real_axis::fit_matsubara(Sw, wp, optA1);

      // The RAW incoming QP spectrum. Everything DIAGNOSTIC below is read here: it is the
      // physical band structure of the current outer iteration, and it is what "gap window"
      // means. The route-A start refinement is applied afterwards, to the inner loop's
      // starting point only, so that a bad refinement cannot silently move the diagnostics.
      for (long a = 0; a < nbnd; ++a) eps(a) = sE_ska.local()(is, ik, a).real();

      const sk_block *blk = nullptr;
      if (cd) blk = std::addressof(ctx->blocks[ctx->block_index(is, ik)]);

      // ------- the gap window: two occupied + two empty states straddling mu -----------
      // Built from the INCOMING (route-A refined) spectrum, i.e. before the inner loop can
      // move anything: it is the physically meaningful window, and it must not depend on the
      // outcome of the loop it is used to diagnose.
      std::vector<long> win;
      {
        std::vector<long> occ, emp;
        for (long a = 0; a < nbnd; ++a) (eps(a) < mu ? occ : emp).push_back(a);
        std::sort(occ.begin(), occ.end(), [&](long x, long y) { return eps(x) > eps(y); });
        std::sort(emp.begin(), emp.end(), [&](long x, long y) { return eps(x) < eps(y); });
        for (size_t t = 0; t < std::min<size_t>(2, occ.size()); ++t) win.push_back(occ[t]);
        for (size_t t = 0; t < std::min<size_t>(2, emp.size()); ++t) win.push_back(emp[t]);
        std::sort(win.begin(), win.end());
      }

      // ------- THE ANCHOR (gate QM3-b(ii)) ---------------------------------------------
      // Evaluated at FERMIONIC nodes, so it is independent of eps; only the window is not.
      if (cd) {
        double num = 0.0, den = 0.0;
        const long nanch = std::min<long>(4, npos);
        for (long m = 0; m < nanch; ++m) {
          modea_sigma_at(*ctx, *blk, ComplexType(mu, wp(m)), Sanch);
          auto Sgw = sSigma_wskab.local()(widx[m], is, ik, all, all);
          for (long a : win)
            for (long b : win) {
              num = std::max(num, std::abs(Sanch(a, b) - Sgw(a, b)));
              den = std::max(den, std::abs(Sgw(a, b)));
            }
        }
        const double dev = (den > 0.0) ? num / den : 0.0;
        anchor_worst = std::max(anchor_worst, dev);
        app_log(lvl, "  mode_a ANCHOR (s,k) = ({},{}): max rel dev over {} nodes x the "
                     "gap window = {:.4e}  (max|Sigma^GW| = {:.4e}, W-fit reconstruction "
                     "class {:.4e})", is, ik, nanch, dev, den, ctx->diag.rec_rel_worst);
      }

      // ------- the A/B harness AT THE INCOMING ENERGIES (spec section 5) ----------------
      // This is the near-gap real-axis accuracy of route B: close to mu the A-side
      // truncation class is tight, so delta_i there IS the B-side error scale. It must be
      // read BEFORE the inner loop, whose divergence would otherwise evaluate both sides at
      // meaningless energies.
      {
        // Both sides are read AT THE ACTUAL EVALUATION POINT of the map, i.e. at the strip
        // evaluation point (rev 3 addendum item 2 / rev 4): comparing an out-of-strip A-side
        // extrapolation against a B side evaluated at mu would compare two different
        // arguments. The A side can only be evaluated at REAL energies, so with eta_far > 0
        // it is read at Re z (the same eps_i) while the B side carries the i eta_far.
        // Out-of-strip rows are marked "*"; eps_i - mu is printed RAW.
        app_log(lvl, "  mode_a delta_i [IN] (s,k) = ({},{}):  {:>4} {:>14} {:>12} {:>12} "
                     "{:>10}", is, ik, "i", "eps_i-mu (eV)", "delta_i", "class_i", "ratio");
        for (long i = 0; i < nbnd; ++i) {
          const bool in_win = std::find(win.begin(), win.end(), i) != win.end();
          const double er = eps(i) - mu;
          bool hit = false;
          const ComplexType zc = (mode_b or not cd) ? ComplexType(eps(i), ctx->eta)
                                                    : strip.zeval(eps(i), &hit);
          const ComplexType SA = XA.eval(i, i, zc.real() - mu);
          const ComplexType SA1 = XA1.eval(i, i, zc.real() - mu);
          const ComplexType SB = cd ? modea_sigma_diag(*ctx, *blk, i, zc) : SA;
          const double d = std::abs(SB - SA), cl = std::abs(SA1 - SA);
          const double ratio = (cl > 0.0) ? d / cl : 0.0;
          if (in_win) {
            ratio_in_worst = std::max(ratio_in_worst, ratio);
            delta_in_worst = std::max(delta_in_worst, d);
            class_in_worst = std::max(class_in_worst, cl);
            if (ratio > 10.0) ++n_flag_in;
          }
          app_log(in_win ? lvl : lvl + 1, "  mode_a delta_i [IN]   ({},{}):  {:>4} {:>14.4f} "
                                          "{:>12.4e} {:>12.4e} {:>10.3g}{}",
                  is, ik, i, er * HA2EV, d, cl, ratio,
                  hit ? (eta_far > 0.0 ? "  * eta_far" : "  * clamped") : "");
        }
      }

      // NOTE: the route-A z0 = 0 start refinement that the spec adopted as a default here
      // has been DELETED -- see the reversal note in qp_modea.hpp. The inner loop starts
      // unconditionally from the raw incoming sE_ska.
      // ------- THE TAU-DOMAIN ORACLE (coordinator request 2026-08-12) ------------------
      // Same elements, two domains: tau (no transform on either side) vs the first fermionic
      // nodes (the anchor). Runs on EVERY block this rank owns; only the root rank's rows
      // reach the log (app_log), while the gate below maxes over all ranks -- which is why
      // the gate quantity now carries its argmax (s,k,a,b) with it.
      //
      // NORMALIZATION -- GATE-SEMANTICS CORRECTION, 2026-08-13 (spec-author ruling; see
      // notes/qm3_mode_a_loop_spec.md rev 4 and "THE GATE'S NORMALIZATION" in
      // wc_band_elements.hpp). The GATE quantity is normalized ONCE PER BLOCK, by the largest
      // |Sigma^GW| of the probed set, exactly like the i w anchor it replaced (:270-283). The
      // per-element ratios below are KEPT as log lines -- they are what identified the kp444
      // false fire -- but they are diagnostics, not the gate: dividing each element by ITS OWN
      // magnitude lets a symmetry-suppressed off-diagonal (kp444 block (0,0): the largest
      // gap-window off-diagonal was 1860x below the diagonal) turn a uniform 5.6e-09 a.u.
      // absolute deviation into a 6.6e-05 "relative" one, and the gate fired on the smallness
      // of the element rather than on any error.
      if (cd) {
        nda::array<double, 1> tau_ph(FT.nt_f());
        {
          auto xm = FT.tau_mesh();
          for (long i = 0; i < FT.nt_f(); ++i) tau_ph(i) = (xm(i) + 1.0) * FT.beta() / 2.0;
        }
        nda::array<ComplexType, 1> Stau(FT.nt_f());
        // the gap-window diagonals, plus the largest off-diagonal of the raw map
        std::vector<std::pair<long, long>> elems;
        for (long i : win) elems.emplace_back(i, i);
        {
          long ba = 0, bb = 1;
          double vm = -1.0;
          for (long a : win)
            for (long b : win)
              if (a != b) {
                double m = 0.0;
                for (long i = 0; i < FT.nt_f(); ++i)
                  m = std::max(m, std::abs(sSigma_tskab.local()(i, is, ik, a, b)));
                if (m > vm) { vm = m; ba = a; bb = b; }
              }
          elems.emplace_back(ba, bb);
        }
        app_log(lvl, "  mode_a TAU ORACLE (s,k) = ({},{}): {:>6} {:>14} {:>14} {:>10}",
                is, ik, "(a,b)", "tau rel dev", "iw rel dev", "ratio");
        double worst_tn = -1.0, worst_td = 0.0, blk_den = 0.0;
        std::pair<long, long> worst_ab{0, 0};
        for (auto const &[a, b] : elems) {
          modea_sigma_tau(*ctx, *blk, a, b, tau_ph, Stau);
          double tn = 0.0, td = 0.0;
          for (long i = 0; i < FT.nt_f(); ++i) {
            const ComplexType ref = sSigma_tskab.local()(i, is, ik, a, b);
            tn = std::max(tn, std::abs(Stau(i) - ref));
            td = std::max(td, std::abs(ref));
          }
          double wn_ = 0.0, wd = 0.0;
          const long nanch = std::min<long>(4, npos);
          for (long m = 0; m < nanch; ++m) {
            modea_sigma_at(*ctx, *blk, ComplexType(mu, wp(m)), Sanch);
            const ComplexType ref = sSigma_wskab.local()(widx[m], is, ik, a, b);
            wn_ = std::max(wn_, std::abs(Sanch(a, b) - ref));
            wd = std::max(wd, std::abs(ref));
          }
          const double dtau = (td > 0.0) ? tn / td : 0.0;
          const double diw = (wd > 0.0) ? wn_ / wd : 0.0;
          if (tn > worst_tn) { worst_tn = tn; worst_ab = {a, b}; }
          blk_den = std::max(blk_den, td);
          app_log(lvl, "  mode_a TAU ORACLE     ({},{}): ({:>2},{:>2}) {:>14.4e} {:>14.4e} "
                       "{:>10.3g}{}", is, ik, a, b, dtau, diw,
                  (dtau > 0.0 ? diw / dtau : 0.0), (a == b) ? "" : "   <- off-diagonal");
        }
        // THE GATE QUANTITY of this block: ONE denominator for the probed set (above).
        const double dtau_blk = (blk_den > 0.0 and worst_tn > 0.0) ? worst_tn / blk_den : 0.0;
        worst_td = blk_den;
        app_log(lvl, "  mode_a TAU ORACLE     ({},{}): BLOCK = {:.4e}  [= max|Sigma_B - "
                     "Sigma^GW| {:.4e} over the probed set, at element ({},{}), / max|Sigma^GW| "
                     "{:.4e} of the SAME set -- THE GATE QUANTITY, normalized once per block "
                     "like the i w anchor. The per-element rows above divide by each element's "
                     "own magnitude and are diagnostics only.]",
                is, ik, dtau_blk, worst_tn, worst_ab.first, worst_ab.second, blk_den);
        if (dtau_blk > tau_dev_worst) {
          tau_dev_worst = dtau_blk;
          tau_dev_is = is; tau_dev_ik = ik;
          tau_dev_a = worst_ab.first; tau_dev_b = worst_ab.second;
        }
        // ---- THE PER-ISYM BREAKDOWN (permanent, level 2; the kp444 post-mortem) ---------
        // The route-B side of the element with the largest ABSOLUTE deviation, split over the
        // symmetry classes of the star loop (ctx->q_isym; the classes partition the full
        // transfer mesh, so the rows sum back to the total -- printed as the bookkeeping
        // check), against that same absolute deviation. Under the block normalization adopted
        // on 2026-08-13 that element IS the one that sets the gate (numerator and gate share
        // the same max), so the breakdown and the gate now describe the same thing -- they did
        // NOT before: the census picked the largest absolute deviation while the gate maxed
        // per-element ratios, so at kp444 the census described the diagonal (3,3) while the
        // gate fired on the suppressed off-diagonal (2,5). The reference cannot be split (the
        // solver sums the isym loop
        // internally), so there is no per-class deviation; what discriminates is each class's
        // own MAGNITUDE:
        //   share       = max_tau |Sigma_B^(isym)| / max_tau |Sigma^GW|
        //   dev/|class| = how wrong THAT CLASS ALONE would have to be, relatively, to carry
        //                 the entire deviation.
        // A structurally wrong class -- wrong rotation, wrong fractional-translation phase,
        // wrong trev branch -- is wrong by O(1), so exactly one row would read ~1. Rows that
        // all read << 1 EXCLUDE a per-class convention error, and by the symmetry derivation
        // in wc_band_elements.hpp the only object the two sides then still differ in is the W
        // representation (fit + stage-1b/1c truncation).
        if (ctx->nsym > 1 and ctx->q_isym.size() > 0 and worst_td > 0.0) {
          const long a = worst_ab.first, b = worst_ab.second;
          nda::array<ComplexType, 2> Scls(ctx->nsym, FT.nt_f());
          modea_sigma_tau_by_class(*ctx, *blk, a, b, tau_ph, Scls);
          modea_sigma_tau(*ctx, *blk, a, b, tau_ph, Stau);
          double chk = 0.0, worst_ratio = 0.0;
          long worst_s = -1;
          for (long i = 0; i < FT.nt_f(); ++i) {
            ComplexType tot(0.0);
            for (long s = 0; s < ctx->nsym; ++s) tot += Scls(s, i);
            chk = std::max(chk, std::abs(tot - Stau(i)));
          }
          app_log(lvl, "  mode_a TAU ISYM (s,k) = ({},{}): worst element ({},{}), absolute "
                       "deviation {:.4e} of max|Sigma^GW| {:.4e}; {:>5} {:>8} {:>14} {:>11} "
                       "{:>13}", is, ik, a, b, worst_tn, worst_td, "isym", "nq", "max|Sigma_B|",
                  "share", "dev/|class|");
          for (long s = 0; s < ctx->nsym; ++s) {
            double m = 0.0;
            for (long i = 0; i < FT.nt_f(); ++i) m = std::max(m, std::abs(Scls(s, i)));
            long nq_s = 0;
            for (long qp = 0; qp < ctx->q_isym.size(); ++qp) if (ctx->q_isym(qp) == s) ++nq_s;
            if (nq_s == 0) continue;      // a qsymms entry the star loop never uses
            const double ratio = (m > 0.0) ? worst_tn / m : -1.0;
            if (ratio > worst_ratio) { worst_ratio = ratio; worst_s = s; }
            app_log(lvl, "  mode_a TAU ISYM       ({},{}): {:>27} {:>5} {:>8} {:>14.4e} "
                         "{:>11.3e} {:>13.3g}", is, ik, "", s, nq_s, m,
                    (worst_td > 0.0 ? m / worst_td : 0.0), ratio);
          }
          app_log(lvl, "  mode_a TAU ISYM       ({},{}): worst dev/|class| = {:.3g} at class "
                       "{} -- O(1) there means THAT symmetry class is structurally wrong, "
                       "<< 1 for every class means the deviation is diffuse (the W "
                       "representation). Class sum vs total = {:.2e} (bookkeeping).",
                  is, ik, worst_ratio, worst_s, chk);
          if (worst_ratio > isym_ratio_worst) {
            isym_ratio_worst = worst_ratio;
            isym_class_worst = worst_s;
          }
        }
      }

      if (mode_b) {
        // ---- MODE B (spec rev 2, the user ruling): no inner-consistency loop ----
        auto br = modeb_vxc(*ctx, *blk, eps, win, V);
        sVcorr_skij.local()(is, ik, all, all) = V;
        n_fallback += br.n_fallback;
        n_fallback_win += br.n_fallback_win;
        n_sanity_trip += br.n_sanity_trip;
        if (br.homo_fallback) ++n_homo_fb;
        if (br.lumo_fallback) ++n_lumo_fb;
        ++n_blocks;
        min_den_worst = std::min(min_den_worst, br.min_den);
        anti_worst = std::max(anti_worst, br.anti_herm);
        n_eta += br.n_eta;
        im_off_worst = std::max(im_off_worst, br.im_off);
        spacing_worst = std::max(spacing_worst, br.spacing);
        app_log(lvl, "  mode_b (s,k) = ({},{}): max|V| = {:.4e}, min_den = {:.4e} a.u., "
                     "out-of-strip diagonals: {} of {} ({} in the gap window, {} evaluated at "
                     "eps + i eta_far, the rest at z = mu); per-k HOMO {}, LUMO {}",
                is, ik, br.vmax, br.min_den, br.n_fallback, nbnd, br.n_fallback_win, br.n_eta,
                br.homo_fallback ? "OUT OF STRIP" : "strip-exact",
                br.lumo_fallback ? "OUT OF STRIP" : "strip-exact");
        continue;
      }

      consist_result cr;
      qp_modea::clamp_census cc;   // the LAST sweep's census (the map that is returned)
      if (cd) {
        cr = inner_consistency(
            [&](nda::array<double, 1> const &e, nda::array<ComplexType, 2> &Vout, long *am) {
              return modea_vxc_cd(*ctx, *blk, e, Vout, am, std::addressof(cc),
                                  std::addressof(win));
            }, Hstat_ab, eps, V, ctx->opts.nconsist, ctx->opts.consist_tol);
      } else {
        cr = inner_consistency(
            [&](nda::array<double, 1> const &e, nda::array<ComplexType, 2> &Vout, long *am) {
              (void)am;
              nda::array<double, 1> er(e.shape(0));
              for (long a = 0; a < e.shape(0); ++a) er(a) = e(a) - mu;
              Vout = sigma_real_axis::assemble_vxc(XA, er);
              return 1e300;
            }, Hstat_ab, eps, V, ctx->opts.nconsist, ctx->opts.consist_tol);
      }
      sVcorr_skij.local()(is, ik, all, all) = V;
      app_log(lvl, "  mode_a inner consistency (s,k) = ({},{}): {} sweeps, max|d eps| = "
                       "{:.4e}, max|V| = {:.4e}, min_den = {:.4e} at state {} (eps - mu = "
                       "{:+.6f} a.u.)", is, ik, cr.iters, cr.dmax, cr.vmax, cr.min_den,
              cr.min_den_a, cr.min_den_ea - mu);
      iters_worst = std::max(iters_worst, cr.iters);
      dmax_worst = std::max(dmax_worst, cr.dmax);
      min_den_worst = std::min(min_den_worst, cr.min_den);
      anti_worst = std::max(anti_worst, cr.anti_herm);
      if (not cr.converged) ++n_noconv;

      // ------- THE STRIP CLAMP CENSUS (spec rev 3 addendum item 2) ---------------------
      // Read off the LAST inner sweep, i.e. the map actually returned to the caller.
      if (cd) {
        n_clamp += cc.n_clamp;
        n_clamp_win += cc.n_clamp_win;
        n_eval += cc.n_eval;
        if (cc.homo_clamp) ++n_homo_cl;
        if (cc.lumo_clamp) ++n_lumo_cl;
        exc_lo_worst = std::max(exc_lo_worst, cc.exc_lo);
        exc_hi_worst = std::max(exc_hi_worst, cc.exc_hi);
        n_eta += cc.n_eta;
        im_off_worst = std::max(im_off_worst, cc.im_off);
        spacing_worst = std::max(spacing_worst, cc.spacing);
        if (cc.anti_in >= 0.0) { anti_in_worst = std::max(anti_in_worst, cc.anti_in); ++n_anti_in; }
        ++n_blocks;
        app_log(lvl, "  mode_a STRIP CENSUS (s,k) = ({},{}): {} of {} evaluation energies "
                     "out of strip ({} in the gap window, {} evaluated at eps + i eta_far, "
                     "the rest at mu); per-k HOMO {}, LUMO {}; worst excursion below the "
                     "lower bound {:.4f} a.u., above the upper bound {:.4f} a.u.",
                is, ik, cc.n_clamp, cc.n_eval, cc.n_clamp_win, cc.n_eta,
                cc.homo_clamp ? "OUT OF STRIP" : "in strip",
                cc.lumo_clamp ? "OUT OF STRIP" : "in strip", cc.exc_lo, cc.exc_hi);
      }

      // ------- the same harness AT THE EXIT ENERGIES (reported, interpretable only if
      //         the inner loop converged) -------------------------------------------------
      {
        app_log(lvl + 1, "  mode_a delta_i [OUT] (s,k) = ({},{}):  {:>4} {:>14} {:>12} {:>12} "
                         "{:>10}", is, ik, "i", "eps_i-mu (eV)", "delta_i", "class_i", "ratio");
        for (long i = 0; i < nbnd; ++i) {
          const bool in_win = std::find(win.begin(), win.end(), i) != win.end();
          const double er = eps(i) - mu;
          bool hit = false;
          const ComplexType zc = cd ? strip.zeval(eps(i), &hit)
                                    : ComplexType(eps(i), ctx->eta);
          const ComplexType SA = XA.eval(i, i, zc.real() - mu);
          const ComplexType SA1 = XA1.eval(i, i, zc.real() - mu);
          const ComplexType SB = cd ? modea_sigma_diag(*ctx, *blk, i, zc) : SA;
          const double d = std::abs(SB - SA), cl = std::abs(SA1 - SA);
          const double ratio = (cl > 0.0) ? d / cl : 0.0;
          if (in_win) ratio_worst = std::max(ratio_worst, ratio);
          if (in_win and ratio > 10.0) ++n_flag;
          app_log(lvl + 1, "  mode_a delta_i [OUT]  ({},{}):  {:>4} {:>14.4f} "
                           "{:>12.4e} {:>12.4e} {:>10.3g}{}",
                  is, ik, i, er * HA2EV, d, cl, ratio,
                  hit ? (eta_far > 0.0 ? "  * eta_far" : "  * clamped") : "");
        }
        // one off-diagonal spot check: the largest |V^xc_ab|, a != b. The A side is built at
        // the SAME (clamped) energies the B side used, for the same reason as the table above.
        long ba = 0, bb = 0;
        double vmax = -1.0;
        for (long a = 0; a < nbnd; ++a)
          for (long b = 0; b < nbnd; ++b)
            if (a != b and std::abs(V(a, b)) > vmax) { vmax = std::abs(V(a, b)); ba = a; bb = b; }
        for (long a = 0; a < nbnd; ++a)
          eps_rel(a) = (cd ? strip.zeval(eps(a)).real() : eps(a)) - mu;
        auto VA = sigma_real_axis::assemble_vxc(XA, eps_rel);
        const double doff = std::abs(V(ba, bb) - VA(ba, bb)) / std::max(vmax, 1e-30);
        dev_off_worst = std::max(dev_off_worst, doff);
        app_log(lvl + 1, "  mode_a off-diagonal spot check (s,k) = ({},{}): largest |V_ab| at "
                         "({},{}) = {:.4e}; A-vs-B rel dev = {:.4e}", is, ik, ba, bb, vmax, doff);
      }
    }

    anchor_worst = comm.all_reduce_value(anchor_worst, boost::mpi3::max<>{});
    dmax_worst = comm.all_reduce_value(dmax_worst, boost::mpi3::max<>{});
    min_den_worst = comm.all_reduce_value(min_den_worst, boost::mpi3::min<>{});
    anti_worst = comm.all_reduce_value(anti_worst, boost::mpi3::max<>{});
    ratio_worst = comm.all_reduce_value(ratio_worst, boost::mpi3::max<>{});
    dev_off_worst = comm.all_reduce_value(dev_off_worst, boost::mpi3::max<>{});
    iters_worst = comm.all_reduce_value(iters_worst, boost::mpi3::max<>{});
    n_noconv = comm.all_reduce_value(n_noconv, std::plus<>{});
    n_flag = comm.all_reduce_value(n_flag, std::plus<>{});
    ratio_in_worst = comm.all_reduce_value(ratio_in_worst, boost::mpi3::max<>{});
    delta_in_worst = comm.all_reduce_value(delta_in_worst, boost::mpi3::max<>{});
    class_in_worst = comm.all_reduce_value(class_in_worst, boost::mpi3::max<>{});
    n_flag_in = comm.all_reduce_value(n_flag_in, std::plus<>{});

    // rev 4: the far-state census travels with the anti-Hermitian rescope below.
    n_eta = comm.all_reduce_value(n_eta, std::plus<>{});
    im_off_worst = comm.all_reduce_value(im_off_worst, boost::mpi3::max<>{});
    spacing_worst = comm.all_reduce_value(spacing_worst, boost::mpi3::max<>{});
    anti_in_worst = comm.all_reduce_value(anti_in_worst, boost::mpi3::max<>{});
    n_anti_in = comm.all_reduce_value(n_anti_in, std::plus<>{});
    // THE TRIPWIRE quantity. With eta_far > 0 the off-strip evaluations are complex BY
    // CONSTRUCTION (Im Sigma(eps + i eta) is eta times the smoothed spectral density), so an
    // O(1) anti-Hermitian residual there is physics, not a routing error; the tripwire is
    // rescoped to the elements both of whose evaluation points are real. mode_b needs no
    // rescope: its off-diagonals are all at z = mu and its diagonal takes Re explicitly.
    // n_anti_in = 0 means NO block had an in-strip pair at all (every state out of strip); the
    // rescoped number would then be vacuous, so the full-matrix one is reported instead.
    const bool anti_rescoped = (eta_far > 0.0 and not mode_b and cd and n_anti_in > 0);
    const double anti_gate = anti_rescoped ? anti_in_worst : anti_worst;

    app_log(lvl, "  - inner QP consistency:       worst count {} of {} (cap), max|d eps| at "
                 "exit = {:.3e} a.u., min_den = {:.4e} a.u.",
            iters_worst, ctx->opts.nconsist, dmax_worst, min_den_worst);
    app_log(lvl, "  - anti-Hermitian residual:    max|V - V^dag|/max|V| = {:.3e} over {} "
                 "(expected at the W-fit class {:.3e}; O(1) would be a routing error)",
            anti_gate, anti_rescoped ? "the IN-STRIP elements (rev 4 rescope; the full-matrix "
                                       "value below is dominated by the eta-broadened far "
                                       "states and is NOT an error)" : "all elements",
            ctx->diag.rec_rel_worst);
    if (eta_far > 0.0)
      app_log(lvl, "  - rev-4 far-state physics:    {} out-of-strip evaluations at eps + i "
                   "eta_far ({:.4e} a.u. = {:.4g} eV); max|Im Sigma^c| there = {:.4e} a.u. "
                   "({:.4g} eV) [= eta x the smoothed spectral density, a DIAGNOSTIC]; "
                   "full-matrix anti-Hermitian residual = {:.3e}; worst local fitted-pole "
                   "spacing at those points = {:.4e} a.u. (eta_far / spacing = {:.3g}, "
                   "floor {:.1f})",
              n_eta, eta_far, eta_far * HA2EV, im_off_worst, im_off_worst * HA2EV, anti_worst,
              spacing_worst, (spacing_worst > 0.0 ? eta_far / spacing_worst : 0.0),
              qp_modea::modea_eta_far_mult);
    if (eta_far > 0.0 and n_eta > 0 and spacing_worst > 0.0 and
        eta_far < qp_modea::modea_eta_far_mult * spacing_worst)
      app_warning("qp_approx ({}): qp_modea_eta_far = {:.4e} a.u. is BELOW {:.1f} x the "
                  "measured local fitted-pole spacing ({:.4e} a.u.) at the out-of-strip "
                  "evaluation points. Sigma^c(eps + i eta_far) is then dominated by individual "
                  "poles of the FIT rather than by the eta-smoothed spectral density, i.e. the "
                  "far-state values are representation artefacts. Raise eta_far above {:.4e} "
                  "a.u. ({:.4g} eV) or sharpen the W^c fit. See notes/qm3_mode_a_loop_spec.md "
                  "rev 4 (validity floor).",
                  map_name, eta_far, qp_modea::modea_eta_far_mult, spacing_worst,
                  qp_modea::modea_eta_far_mult * spacing_worst,
                  qp_modea::modea_eta_far_mult * spacing_worst * HA2EV);
    if (cd)
      app_log(lvl, "  - i w comparison (DIAGNOSTIC, never a gate): route-B Sigma^c vs the "
                   "solver Sigma(i w_n) over the gap window = {:.4e} [W-fit class {:.4e}]. "
                   "This number is dominated by the REFERENCE's tau -> i w transform of the "
                   "G.W product, whose spectral support exceeds the grid: it halves with "
                   "every DLR prec notch while the tau deviation does not. The gate is the "
                   "TAU anchor above.", anchor_worst, anchor_expect);
    {   // MAXLOC: reduce the gate quantity together with WHERE it was attained. Only the
        // root rank's oracle rows reach the log, so without this the offending block of a
        // multi-rank run is unnameable -- the 2026-08-13 kp444 post-mortem.
      const double local = tau_dev_worst;
      tau_dev_worst = comm.all_reduce_value(tau_dev_worst, boost::mpi3::max<>{});
      if (local < tau_dev_worst) { tau_dev_is = tau_dev_ik = tau_dev_a = tau_dev_b = -1; }
      tau_dev_is = comm.all_reduce_value(tau_dev_is, boost::mpi3::max<>{});
      tau_dev_ik = comm.all_reduce_value(tau_dev_ik, boost::mpi3::max<>{});
      tau_dev_a = comm.all_reduce_value(tau_dev_a, boost::mpi3::max<>{});
      tau_dev_b = comm.all_reduce_value(tau_dev_b, boost::mpi3::max<>{});
      const double li = isym_ratio_worst;
      isym_ratio_worst = comm.all_reduce_value(isym_ratio_worst, boost::mpi3::max<>{});
      if (li < isym_ratio_worst) isym_class_worst = -1;
      isym_class_worst = comm.all_reduce_value(isym_class_worst, boost::mpi3::max<>{});
    }
    app_log(lvl, "  - TAU ORACLE:                 max BLOCK-normalized dev of Sigma_B(tau) vs "
                 "the solver's Sigma^c(tau) (no transform on either side) = {:.4e}, attained "
                 "on block (s,k) = ({},{}) at element ({},{}); the same elements at the first "
                 "fermionic nodes deviate by {:.4e} (the anchor). [Both are normalized by the "
                 "largest |Sigma^GW| of the probed set -- gate-semantics correction of "
                 "2026-08-13, see wc_band_elements.hpp.]",
            tau_dev_worst, tau_dev_is, tau_dev_ik, tau_dev_a, tau_dev_b, anchor_worst);
    app_log(lvl, "  - A/B harness [IN, gap window]: max delta_i = {:.4e} a.u. ({:.4g} meV), "
                 "max class_i = {:.4e} a.u. ({:.4g} meV), worst ratio = {:.3g} ({} states "
                 "above 10x)",
            delta_in_worst, delta_in_worst * 27211.386, class_in_worst,
            class_in_worst * 27211.386, ratio_in_worst, n_flag_in);
    app_log(lvl, "  - A/B harness [OUT]:          worst delta_i/class_i over gap-window "
                 "states = {:.3g} ({} states above 10x); off-diagonal spot check rel dev = "
                 "{:.3e}", ratio_worst, n_flag, dev_off_worst);
    // ONE machine-greppable line per outer iteration: the knob-matrix harness scrapes this.
    app_log(1, "@@MODEA_CELL wfit={} wrtol={:.1e} eta={:.4e} | ratio={:.4e} rec={:.4e} "
               "gapedge={:.6g} npk={} | dmax={:.4e} iters={} minden={:.4e} anchor={:.4e} "
               "antiherm={:.4e} taudev={:.4e} | dIN={:.4e} clIN={:.4e} rIN={:.4e} | "
               "wrank={:.1e} Np={} rmax={} rmean={:.2f} wtrunc={:.3e} "
               "tfit={:.2f} tfac={:.2f} tsand={:.2f} | etafar={:.4e} neta={} imoff={:.4e} "
               "spacing={:.4e} antiin={:.4e}",
            ctx->opts.wfit, ctx->diag.gap_edge > -1 ? qp_modea::last_run().wrtol : -1.0,
            ctx->eta, ctx->diag.res_ratio_worst, ctx->diag.rec_rel_worst,
            ctx->diag.gap_edge, ctx->npk, dmax_worst, iters_worst, min_den_worst,
            anchor_worst, anti_gate, tau_dev_worst, delta_in_worst, class_in_worst,
            ratio_in_worst, ctx->opts.wrank, qp_modea::last_run().Np,
            ctx->diag.wrank_max, ctx->diag.wrank_mean, ctx->diag.wtrunc_worst,
            ctx->diag.t_fit, ctx->diag.t_fac, ctx->diag.t_sand,
            eta_far, n_eta, im_off_worst, spacing_worst, anti_in_worst);
    if (n_noconv > 0)
      app_warning("qp_approx (mode_a): the inner QP-consistency loop hit the cap ({}) on {} "
                  "(s,k) blocks with max|d eps| = {:.3e}. This is the physical "
                  "multi-solution flag of the spec, not an error.",
                  ctx->opts.nconsist, n_noconv, dmax_worst);

    // ---- THE STRIP CLAMP CENSUS, summary (spec rev 3 addendum item 2) ----
    // n_blocks is shared with the mode_b census below and is reduced HERE, once.
    n_blocks = comm.all_reduce_value(n_blocks, std::plus<>{});
    n_clamp = comm.all_reduce_value(n_clamp, std::plus<>{});
    n_clamp_win = comm.all_reduce_value(n_clamp_win, std::plus<>{});
    n_eval = comm.all_reduce_value(n_eval, std::plus<>{});
    n_homo_cl = comm.all_reduce_value(n_homo_cl, std::plus<>{});
    n_lumo_cl = comm.all_reduce_value(n_lumo_cl, std::plus<>{});
    exc_lo_worst = comm.all_reduce_value(exc_lo_worst, boost::mpi3::max<>{});
    exc_hi_worst = comm.all_reduce_value(exc_hi_worst, boost::mpi3::max<>{});
    if (cd and not mode_b) {
      app_log(lvl, "  - mode_a STRIP CLAMP:         strip = (VBM - 0.95 E_PH, CBM + 0.95 "
                   "E_PH) = ({:+.6f}, {:+.6f}) a.u. with VBM {:+.6f}, CBM {:+.6f}, E_PH "
                   "{:.6f} ({})", strip.lo, strip.hi, ctx->vbm, ctx->cbm,
              ctx->diag.gap_edge, strip.active ? "ACTIVE" : "INACTIVE (no support "
                                                            "constraint this iteration)");
      app_log(lvl, "  - mode_a strip census:        {} of {} evaluation energies out of strip "
                   "({} in the gap window; {} evaluated at eps + i eta_far, {} clamped to mu) "
                   "over {} (s,k) blocks; THE JUDGE STATES: per-k HOMO out of strip in {} of "
                   "{} blocks, LUMO in {} of {}; worst excursion {:.4f} a.u. below / {:.4f} "
                   "a.u. above",
              n_clamp, n_eval, n_clamp_win, n_eta, n_clamp - n_eta, n_blocks, n_homo_cl,
              n_blocks, n_lumo_cl, n_blocks, exc_lo_worst, exc_hi_worst);
      if (n_clamp_win > 0)
        app_warning("qp_approx (mode_a): {} GAP-WINDOW evaluation energies were OUT OF STRIP "
                    "(evaluated at {}). Near mu the particle-hole edge should guarantee a "
                    "clearance of E_PH = {:.4g} a.u., so this means the QP spectrum has moved "
                    "a band-edge state outside the analyticity strip -- the judge states are "
                    "then NOT exact mode A. Check the gap and the inner-consistency numbers "
                    "above.", n_clamp_win,
                    eta_far > 0.0 ? "eps + i eta_far" : "mu", ctx->diag.gap_edge);
    }

    auto &LR = qp_modea::last_run();
    LR.anchor = cd ? anchor_worst : -1.0;
    LR.n_clamp = n_clamp;
    LR.n_clamp_win = n_clamp_win;
    LR.n_eval = n_eval;
    LR.n_homo_clamp = n_homo_cl;
    LR.n_lumo_clamp = n_lumo_cl;
    LR.n_blocks = n_blocks;
    LR.converged_inner = (n_noconv == 0);
    LR.anchor_expect = anchor_expect;
    LR.ratio_worst = ratio_worst;
    LR.anti_herm = anti_gate;      // rev 4: IN-STRIP only once eta_far > 0 (see above)
    LR.eta_far = eta_far;
    LR.n_eta = n_eta;
    LR.im_off = im_off_worst;
    LR.anti_in = anti_in_worst;
    LR.spacing = spacing_worst;
    LR.dmax = dmax_worst;
    LR.min_den = min_den_worst;
    LR.iters = iters_worst;
    LR.tau_dev = tau_dev_worst;
    LR.n_fallback = n_fallback;
    LR.delta_in = delta_in_worst;
    LR.class_in = class_in_worst;
    LR.ratio_in = ratio_in_worst;

    n_fallback = comm.all_reduce_value(n_fallback, std::plus<>{});
    n_fallback_win = comm.all_reduce_value(n_fallback_win, std::plus<>{});
    n_sanity_trip = comm.all_reduce_value(n_sanity_trip, std::plus<>{});
    n_homo_fb = comm.all_reduce_value(n_homo_fb, std::plus<>{});
    n_lumo_fb = comm.all_reduce_value(n_lumo_fb, std::plus<>{});
    if (mode_b) {
      app_log(lvl, "  - mode_b STRIP TEST census:   strip = (VBM - 0.95 E_PH, CBM + 0.95 "
                   "E_PH) = ({:+.6f}, {:+.6f}) a.u. with VBM {:+.6f}, CBM {:+.6f}, E_PH "
                   "{:.6f}", ctx->vbm - 0.95 * ctx->diag.gap_edge,
              ctx->cbm + 0.95 * ctx->diag.gap_edge, ctx->vbm, ctx->cbm, ctx->diag.gap_edge);
      app_log(lvl, "  - mode_b out-of-strip diag:   {} states ({} in the gap window; {} "
                   "evaluated at eps + i eta_far, {} demoted to z = mu); THE JUDGE STATES: "
                   "per-k HOMO out of strip in {} of {} blocks, LUMO in {} of {}",
              n_fallback, n_fallback_win, n_eta, n_fallback - n_eta, n_homo_fb, n_blocks,
              n_lumo_fb, n_blocks);
      app_log(lvl, "  - mode_b strip-interior |ReSigma| trips: {} (expected 0 -- a nonzero "
                   "count FALSIFIES the strip criterion)", n_sanity_trip);
      if (n_fallback_win > 0)
        app_warning("qp_approx (mode_b): {} GAP-WINDOW diagonal states fell back to the "
                    "Fermi-static value because Sigma^c could not be resolved at their "
                    "quasiparticle energy. Near mu the support constraint should guarantee a "
                    "clearance of gap_edge; check min_den = {:.4e} a.u. and the W^c fit.",
                    n_fallback_win, min_den_worst);
    }

    // THE GATE (spec rev 2): the TAU-DOMAIN anchor. The tau image of the route-B pole rep is
    // compared against the solver's Sigma^c(tau) with NO transform on either side, so it
    // isolates the contraction from the reference's tau -> i w aliasing. Measured 2026-08-12:
    // tau agreement 5.6e-05 at DLR prec "low" (per-element normalization, retired 2026-08-13
    // -- the block-normalized number is smaller), two orders below that grid's W-fit class,
    // while the i w deviation was 2.3e-01 on the SAME elements and halved with every prec
    // notch. The i w comparison is therefore a LOGGED DIAGNOSTIC ONLY -- never a gate.
    //
    // NORMALIZED PER BLOCK since 2026-08-13 (spec-author ruling, rev 4 note): one denominator
    // for the probed set, exactly as the i w anchor at :270-283. See the oracle above.
    if (cd) {
      utils::check(tau_dev_worst < qp_modea::modea_tau_anchor_mult * ctx->diag.rec_rel_worst,
                   "qp_approx ({}): THE TAU ANCHOR FAILED -- the analytic tau image of the "
                   "route-B Sigma^c deviates from the solver's Sigma^c(tau) by {:.4e} on block "
                   "(s,k) = ({},{}) at element ({},{}) [block-normalized: max|Sigma_B - "
                   "Sigma^GW| over the gap window and the largest off-diagonal, divided by "
                   "max|Sigma^GW| of that same set], against a gate of {:.1f} x the "
                   "W-fit reconstruction class ({:.4e}). Neither side applies a Fourier "
                   "transform here, so the two sides differ ONLY in (a) the contraction "
                   "(prefactor, spin, q-star/trev rule, MO rotation, the Gamma head) and "
                   "(b) the W REPRESENTATION -- the support-constrained pole fit and the "
                   "stage-1b/1c truncations. Read them apart from the numbers this run "
                   "already printed:\n"
                   "  * per-isym breakdown (TAU ISYM lines): worst dev/|class| = {:.3g} at "
                   "class {}. A structurally wrong symmetry class is wrong by O(1), so a "
                   "value << 1 for EVERY class rules the symmetry path out -- and the "
                   "contraction is identical to the GW assembly term by term for any D, see "
                   "\"THE SYMMETRY PATH\" in wc_band_elements.hpp (measured: the lih223 "
                   "ladder, 6e-09 absolute).\n"
                   "  * the head: gated since 2026-08-13 by test_qp_map_ab "
                   "\"qp_map_modeb_head_anchor\" (agrees with Sigma_div_correction to 8e-09).\n"
                   "  * the W representation: its own ERROR BUDGET sum_q |dW_q| / sum_q "
                   "max|W_q| = {:.3e} is the anchor scale it predicts on its own -- compare "
                   "it with the {:.4e} above BEFORE reading this as a routing error. Backing "
                   "numbers: tau-domain fit residual {:.3e}, worst slab truncation {:.3e} "
                   "(Frobenius {:.3e}), union projection tail {:.3e}. The gate's yardstick is "
                   "the per-q RELATIVE bosonic-mesh class, which is a different object from "
                   "the budget.\n"
                   "See notes/qm3_mode_a_loop_spec.md rev 2 (and its rev-4 note on the "
                   "2026-08-13 block-normalization correction).",
                   map_name, tau_dev_worst, tau_dev_is, tau_dev_ik, tau_dev_a, tau_dev_b,
                   qp_modea::modea_tau_anchor_mult,
                   ctx->diag.rec_rel_worst, isym_ratio_worst, isym_class_worst,
                   ctx->diag.rec_budget, tau_dev_worst,
                   ctx->diag.fit_err_worst, ctx->diag.wtrunc_worst,
                   ctx->diag.wtrunc_frob_worst, ctx->diag.union_tail_worst);
    }
    if (false)
      utils::check(anchor_worst < qp_modea::modea_anchor_gate,
                   "qp_approx (mode_a): THE ANCHOR FAILED -- route-B Sigma^c deviates from the "
                   "solver Sigma(i w_n) by {:.4e} (gate {:.1g}) over the gap window, against a "
                   "W-fit reconstruction class of {:.4e}.\nIf the class is itself O(1) the W "
                   "pole representation collapsed, which normally means the INCOMING QP "
                   "spectrum already diverged -- check the previous iteration's inner-"
                   "consistency numbers (this iteration: worst count {}, max|d eps| = {:.3e} "
                   "a.u., min_den = {:.4e} a.u.).\nOtherwise this is a CONTRACTION ROUTING "
                   "error (prefactor, spin, q-star/trev rule, MO rotation, or the Gamma head), "
                   "not a tolerance to loosen. See notes/qm3_mode_a_loop_spec.md section 7(ii).",
                   anchor_worst, qp_modea::modea_anchor_gate, anchor_expect, iters_worst,
                   dmax_worst, min_den_worst);
  }
}

auto get_mf_MOs(utils::mpi_context_t<mpi3::communicator> &context, mf::MF &mf, hamilt::pseudopot &psp)
  -> std::tuple<sArray_t<Array_view_4D_t>, sArray_t<Array_view_3D_t> > {
  using math::shm::make_shared_array;
  auto sF_skij = make_shared_array<Array_view_4D_t>(context.comm, context.internode_comm, context.node_comm,
                                                    {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()});
  hamilt::set_fock(mf, std::addressof(psp), sF_skij, false);
  auto sS_skij = make_shared_array<Array_view_4D_t>(context.comm, context.internode_comm, context.node_comm,
                                                    {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()});
  hamilt::set_ovlp(mf, sS_skij);

  auto sMO_skij = make_shared_array<Array_view_4D_t>(context.comm, context.internode_comm, context.node_comm,
                                                     {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()});
  auto sE_ski = make_shared_array<Array_view_3D_t>(context.comm, context.internode_comm, context.node_comm,
                                                   {mf.nspin(), mf.nkpts_ibz(), mf.nbnd()});
  update_MOs(sMO_skij, sE_ski, sF_skij, sS_skij);

  return std::make_tuple(std::move(sMO_skij), std::move(sE_ski));
}

void update_MOs(sArray_t<Array_view_4D_t> &sMO_skij, sArray_t<Array_view_3D_t> &sE_ski,
                const sArray_t<Array_view_4D_t> &sF_skij, const sArray_t<Array_view_4D_t> &sS_skij) {
  auto [ns, nkpts, nbnd, nbnd2] = sF_skij.shape();
  sMO_skij.win().fence();
  sE_ski.win().fence();
  for (long sk = sF_skij.node_comm()->rank(); sk < ns*nkpts; sk+=sF_skij.node_comm()->size()) {
    long is = sk / nkpts;
    long ik = sk % nkpts;
    auto F         = sF_skij.local()(is, ik, nda::ellipsis{});
    auto S         = sS_skij.local()(is, ik, nda::ellipsis{});

    auto [mo_e, coeffs] = nda::linalg::eigenelements(F, S);
    sMO_skij.local()(is, ik, nda::ellipsis{}) = coeffs;
    for (long i = 0; i < sE_ski.shape()[2]; ++i) sE_ski.local()(is, ik, i) = mo_e(i);
  }
  sMO_skij.win().fence();
  sE_ski.win().fence();
}

void update_Dm(sArray_t<Array_view_4D_t> &sDm_skij,
               const sArray_t<Array_view_4D_t> &sMO_skij, const sArray_t<Array_view_3D_t> &sE_ski,
               const double mu, const double beta) {
  auto FD_occ = nda::map([&](ComplexType e) { return 1.0 / ( 1 + std::exp( (e.real()-mu) * beta ) ); });

  auto [ns, nkpts, nbnd, nbnd2] = sDm_skij.shape();
  nda::array<RealType, 1> occ(nbnd);
  nda::array<ComplexType, 2> fCdag(nbnd, nbnd);

  sDm_skij.win().fence();
  for (size_t sk = sDm_skij.node_comm()->rank(); sk < ns*nkpts; sk += sDm_skij.node_comm()->size()) {
    size_t is = sk / nkpts;
    size_t ik = sk % nkpts;

    occ = FD_occ(sE_ski.local()(is, ik, nda::range::all));
    fCdag = nda::transpose(nda::conj(sMO_skij.local()(is, ik, nda::ellipsis{})));
    for (size_t i = 0; i < nbnd; ++i) {
      fCdag(i, nda::range::all) *= occ(i);
    }

    auto C = sMO_skij.local()(is, ik, nda::ellipsis{});
    auto Dm = sDm_skij.local()(is, ik, nda::ellipsis{});
    nda::blas::gemm(C, fCdag, Dm);
  }
  sDm_skij.win().fence();
}

void update_G(sArray_t<Array_view_5D_t> &sG_tskij,
              const sArray_t<Array_view_4D_t> &sMO_skia, const sArray_t<Array_view_3D_t> &sE_ska,
              double mu, const imag_axes_ft::IAFT &FT) {
  double beta = FT.beta();
  auto compute_G0 = [&](double e, double t) {
    double x = e-mu;
    if (x > 0) {
      return -std::exp( -x*t ) / (1 + std::exp( -x*beta ));
    } else {
      return -std::exp( x*(beta-t) ) / (1 + std::exp( x*beta ));
    }
  };

  auto [nts, ns, nkpts, nbnd, nbnd2] = sG_tskij.shape();
  auto x_mesh = FT.tau_mesh();
  auto x_to_tau = nda::map([&](double x) { return (x+1) * FT.beta()/2.0; });
  nda::array<double, 1> tau_mesh = x_to_tau(x_mesh);
  nda::array<ComplexType, 2> GCdag_aj(nbnd, nbnd);

  sG_tskij.win().fence();
  for (size_t tsk = sG_tskij.node_comm()->rank(); tsk < nts*ns*nkpts; tsk += sG_tskij.node_comm()->size()) {
    size_t it = tsk / (ns*nkpts); // tsk = it*ns*nkpts + is*nkpts + ik
    size_t is = (tsk / nkpts) % ns;
    size_t ik = tsk % nkpts;

    GCdag_aj = nda::transpose(nda::conj(sMO_skia.local()(is, ik, nda::ellipsis{})));
    for (size_t a = 0; a < nbnd; ++a) {
      GCdag_aj(a, nda::range::all) *= compute_G0(sE_ska.local()(is, ik, a).real(), tau_mesh(it));
    }

    auto C_ia = sMO_skia.local()(is, ik, nda::ellipsis{});
    auto G_ij = sG_tskij.local()(it, is, ik, nda::ellipsis{});
    nda::blas::gemm(C_ia, GCdag_aj, G_ij);
  }
  sG_tskij.win().fence();
  sG_tskij.communicator()->barrier();
}

template<nda::ArrayOfRank<5> Array_base_t>
void compute_G_from_mf(h5::group iter_grp, imag_axes_ft::IAFT &ft,
                       sArray_t<Array_base_t> &sG_tskij) {
  using math::shm::make_shared_array;
  long ns = sG_tskij.shape()[1];
  long nkpts_ibz = sG_tskij.shape()[2];
  long nbnd = sG_tskij.shape()[3];
  // Construct the Green's function for a mean-field solution on-the-fly
  auto sMO_skia = make_shared_array<Array_view_4D_t>(
      *sG_tskij.communicator(), *sG_tskij.internode_comm(), *sG_tskij.node_comm(),
      {ns, nkpts_ibz, nbnd, nbnd});
  auto sE_ska = make_shared_array<Array_view_3D_t>(
      *sG_tskij.communicator(), *sG_tskij.internode_comm(), *sG_tskij.node_comm(),
      {ns, nkpts_ibz, nbnd});
  double mu;

  sMO_skia.win().fence();
  if (sG_tskij.node_comm()->root()) {
    auto MO_loc = sMO_skia.local();
    auto E_loc = sE_ska.local();
    nda::h5_read(iter_grp, "MO_skia", MO_loc);
    nda::h5_read(iter_grp, "E_ska", E_loc);
  }
  sMO_skia.win().fence();
  h5::h5_read(iter_grp, "mu", mu);

  update_G(sG_tskij, sMO_skia, sE_ska, mu, ft);
  sG_tskij.communicator()->barrier();
}

void solve_qp_eqn(sArray_t<Array_view_3D_t> &sE_ska,
                  const sArray_t<Array_view_5D_t> &sSigma_tskij,
                  const sArray_t<Array_view_4D_t> &sVhf_skij,
                  const sArray_t<Array_view_4D_t> &sMO_skia,
                  double mu,
                  const imag_axes_ft::IAFT &FT, qp_params_t &qp_params,
                  const qp_modea::modea_ctx *modea_ctx) {
  using math::shm::make_shared_array;
  using math::nda::make_distributed_array;
  using local_Array_4D_t = nda::array<ComplexType, 4>;
  using local_Array_3D_t = nda::array<ComplexType, 3>;

  auto comm = sE_ska.communicator();
  auto [ns, nkpts, nbnd, nbnd2] = sVhf_skij.shape();
  auto nt = FT.nt_f();
  auto nw = FT.nw_f();

  int np = comm->size();
  // nbnd > np -> nkpools = 1, np_a = np
  // np_a = np / nkpools <= nbnd -> nkpools > np/nbnd
  int nkpools = utils::find_min_col(np, nbnd, (np%nbnd==0)? np/nbnd : np/nbnd+1);
  int np_a    = np / nkpools;
  utils::check(np_a <= nbnd, "solve_qp_eqn: np_a({}) > nbnd({})", np_a, nbnd);
  utils::check(nkpools <= nkpts, "solv_qp_eqn: nkpools({}) > nkpts({})", nkpools, nkpts);
  utils::check(comm->size() % nkpools == 0, "solve_qp_eqn: comm.size({}) % nkpools({}) != 0", np, nkpools);

  auto dSigma_wska = make_distributed_array<local_Array_4D_t>(*comm, {1, 1, nkpools, np_a}, {nw, ns, nkpts, nbnd}, {1, 1, 1, 1});
  auto s_rng = dSigma_wska.local_range(1);
  auto k_rng = dSigma_wska.local_range(2);
  auto a_rng = dSigma_wska.local_range(3);
  auto [nw_loc, ns_loc, nk_loc, na_loc] = dSigma_wska.local_shape();


  // ------ basis transform from primary to MO basis ------
  {
    auto dSigma_tska = make_distributed_array<local_Array_4D_t>(
        *comm, {1, 1, nkpools, np_a}, {nt, ns, nkpts, nbnd}, {1, 1, 1, 1});
    auto Sigma_tska_loc = dSigma_tska.local();

    nda::array<ComplexType, 2> SigmaC_ia(nbnd, na_loc);
    nda::array<ComplexType, 2> Sigma_ab(na_loc, na_loc);
    for (size_t it = 0; it < nt; ++it) {
      for (auto [is_loc, is]: itertools::enumerate(s_rng)) {
        for (auto [ik_loc, ik]: itertools::enumerate(k_rng)) {

          nda::blas::gemm(sSigma_tskij.local()(it, is, ik, nda::ellipsis{}),
                          sMO_skia.local()(is, ik, nda::range::all, a_rng),
                          SigmaC_ia);
          nda::blas::gemm(ComplexType(1.0),
                          nda::dagger(sMO_skia.local()(is, ik, nda::range::all, a_rng)),
                          SigmaC_ia,
                          ComplexType(0.0),
                          Sigma_ab);
          Sigma_tska_loc(it, is_loc, ik_loc, nda::range::all) = nda::diagonal(Sigma_ab);
        }
      }
    }
    FT.tau_to_w(dSigma_tska.local(), dSigma_wska.local(), imag_axes_ft::fermion);
  }

  // ------ basis transformation from primary to MO basis ------
  auto dVhf_ska = make_distributed_array<local_Array_3D_t>(*comm, {1, nkpools, np_a}, {ns, nkpts, nbnd}, {1, 1, 1});
  auto dE_ska = make_distributed_array<local_Array_3D_t>(*comm, {1, nkpools, np_a}, {ns, nkpts, nbnd}, {1, 1, 1});
  {
    nda::array<ComplexType, 2> VC_ia(nbnd, na_loc);
    nda::array<ComplexType, 2> V_ab(na_loc, na_loc);
    auto Vhf_ska_loc = dVhf_ska.local();
    for (auto [is_loc, is]: itertools::enumerate(s_rng)) {
      for (auto [ik_loc, ik]: itertools::enumerate(k_rng)) {

        nda::blas::gemm(sVhf_skij.local()(is, ik, nda::ellipsis{}),
                        sMO_skia.local()(is, ik, nda::range::all, a_rng),
                        VC_ia);
        nda::blas::gemm(ComplexType(1.0),
                        nda::dagger(sMO_skia.local()(is, ik, nda::range::all, a_rng)),
                        VC_ia,
                        ComplexType(0.0),
                        V_ab);
        Vhf_ska_loc(is_loc, ik_loc, nda::range::all) = nda::diagonal(V_ab);
      }
    }

    auto E_loc = dE_ska.local();
    E_loc = sE_ska.local()(s_rng, k_rng, a_rng);
  }

  // ------ Solve quasi-particle equation for Heff ------
  {
    long dim1 = ns_loc * nk_loc * na_loc;
    auto E_loc_1D = nda::reshape(dE_ska.local(), std::array<long, 1>{dim1});
    auto Vhf_loc_1D = nda::reshape(dVhf_ska.local(), std::array<long, 1>{dim1});
    auto Sigma_loc_2D = nda::reshape(dSigma_wska.local(), std::array<long, 2>{nw, dim1});

    // bypass clang openmp error: error: capturing a structured binding is not yet supported in OpenMP
    auto nk_loc_ = nk_loc;
    auto na_loc_ = na_loc;
    auto I_to_ska = [&](size_t I) {
      // I = s_loc*nk_loc*na_loc + k_loc*na_loc + a_loc
      size_t s_loc = I / (nk_loc_*na_loc_);
      size_t k_loc = (I/na_loc_) % nk_loc_;
      size_t a_loc = I % na_loc_;
      return std::make_tuple(s_rng.first()+s_loc, k_rng.first()+k_loc, a_rng.first()+a_loc);
    };

    auto n_to_iw = nda::map([&](int n) { return FT.omega(n); });
    nda::array<ComplexType, 1> iw_mesh(n_to_iw(FT.wn_mesh()));

    if (qp_params.qp_map != "ac_pade") {

    // Project 2 increment Q2 (notes/qpgw_edmft_implementation_plan.md): the
    // Matsubara-native quasiparticle maps (qp_maps_matsubara.hpp), scalar form
    // -- the qp_type solvers belong to the AC route only.
    auto [wp, widx] = positive_wn_nodes(iw_mesh);
    const long npos = wp.shape(0);
    nda::array<double, 1> wn2(2);
    wn2(0) = wp(0);
    wn2(1) = wp(1);
    long n_clamped = 0, n_noconv = 0;
    app_log(2, "\n* Solving quasiparticle equation for given Sigma(iw): ");
    app_log(2, "  - processor grid for quasi-particle equation: (s, k, a) = ({}, {}, {})", 1, nkpools, np_a);
    app_log(2, "  - quasiparticle map:                          {} (Matsubara-native, no AC)", qp_params.qp_map);
    app_log(2, "  - positive fermionic nodes:                   {} (w0 = {:.6f}, w1 = {:.6f})", npos, wp(0), wp(1));
    if (qp_params.qp_map == "mode_a" or qp_params.qp_map == "mode_b") {

    // ---- Project 2 increment QM3: the evGW diagonal leg (spec section 4) ----
    // The QP equation E = Vhf + Sigma_ii(E) is solved by the EXISTING generic helpers; only
    // the sampler changes. The functor bridges to the route-B closed form on the cached
    // DIAGONAL residues (z = (w - mu) + mu is ABSOLUTE, the sigma_route_b mu rider).
    utils::check(modea_ctx != nullptr and modea_ctx->active,
                 "solve_qp_eqn: qp_map = \"mode_a\" reached the solver without an evaluator "
                 "context (built in the live W window of add_evscf_vcorr).");
    utils::check(modea_ctx->have_cd and modea_ctx->have_diag,
                 "solve_qp_eqn: qp_map = \"mode_a\" on the evGW leg requires "
                 "qp_modea_route = \"cd\" (the diagnostic \"expansion\" route has no "
                 "diagonal residue data on this processor grid).");
    utils::check(qp_params.qp_type == "sc" or qp_params.qp_type == "sc_bisection" or
                 qp_params.qp_type == "sc_newton" or qp_params.qp_type == "linearized",
                 "solve_qp_eqn: qp_map = \"mode_a\" supports qp_type in {{sc, sc_bisection, "
                 "sc_newton, linearized}}; \"{}\" is AC-specific (it needs a spectral "
                 "function, which route B does not produce).", qp_params.qp_type);
    app_log(2, "  - route-B (CD) diagonal sampler:              nJ x npk = {} poles, "
               "eta = {:.3g}", modea_ctx->nJ * modea_ctx->npk, modea_ctx->eta);
    // GUARD, spec rev 3 addendum item 4 (measured, 2026-08-12). This leg is reached only by
    // evGW (qp_mode = "evscf"); the qsGW leg goes through modea_run.
    app_warning("solve_qp_eqn (qp_map = {}): THE evGW LEG IS KNOWN-INCOMPLETE AND "
                "PATHOLOGICALLY SLOW -- measured >75 min per outer iteration on qe_lih222 "
                "against ~7 s for the qsGW (qpscf) leg of the same fixture, __divdc3-bound: "
                "the route-B diagonal sampler rebuilds the full nJ x npk = {} pole-weight "
                "vector at EVERY secant/bisection step of EVERY (s,k,a). It is neither "
                "optimized nor gated in this increment -- the live QM3 gates cover the qpscf "
                "leg only (the evscf fixture cases are hidden behind [.modeb_evscf]). Results "
                "from this path are diagnostic, not deliverable. See "
                "notes/qm3_mode_a_loop_spec.md rev 3 addendum item 4.",
                qp_params.qp_map, modea_ctx->nJ * modea_ctx->npk);

    struct modea_diag_sampler {
      const qp_modea::modea_ctx *ctx;
      long nkpts_ = 0;
      std::function<std::tuple<long,long,long>(size_t)> I_to;
      mutable nda::array<ComplexType, 1> w;
      ComplexType evaluate(ComplexType dz, long I) const {
        auto [is, ik, ia] = I_to(size_t(I));
        const ComplexType z = dz + ctx->mu;   // sigma_route_b: z ABSOLUTE
        ctx->pole_weights(is, ik, z, w);
        ComplexType s(0.0);
        const long nP = ctx->nJ * ctx->npk;
        for (long P = 0; P < nP; ++P) s += ctx->Mdiag(is, ik, ia, P) * w(P);
        return s;
      }
    };
    modea_diag_sampler S;
    S.ctx = modea_ctx;
    S.I_to = I_to_ska;
    S.w = nda::array<ComplexType, 1>(modea_ctx->nJ * modea_ctx->npk);

    double res;
    bool conv;
    long n_noconv_a = 0;
    const double eta_a = qp_params.qp_modea_eta;
    for (size_t I = 0; I < dim1; ++I) {
      if (qp_params.qp_type == "linearized") {
        E_loc_1D(I) = qp_eqn_linearized(Vhf_loc_1D(I).real(), S, I, mu, E_loc_1D(I).real(), eta_a);
      } else if (qp_params.qp_type == "sc_newton") {
        std::tie(E_loc_1D(I), res, conv) =
            qp_eqn_secant(Vhf_loc_1D(I).real(), S, I, mu, E_loc_1D(I).real(), 400,
                          qp_params.tol, eta_a);
        if (!conv) ++n_noconv_a;
      } else {
        std::tie(E_loc_1D(I), res) =
            qp_eqn_bisection(Vhf_loc_1D(I).real(), S, I, mu, E_loc_1D(I).real(),
                             qp_params.tol, eta_a);
      }
    }
    n_noconv_a = comm->all_reduce_value(n_noconv_a, std::plus<>{});
    if (n_noconv_a > 0)
      app_warning("solve_qp_eqn (mode_a): the qp equation did not converge on {} states.",
                  n_noconv_a);

    } else if (qp_params.qp_map == "mats_lin") {
      nda::array<ComplexType, 1> S2(2);
      for (size_t I = 0; I < dim1; ++I) {
        S2(0) = Sigma_loc_2D(widx[0], I);
        S2(1) = Sigma_loc_2D(widx[1], I);
        E_loc_1D(I) = qp_matsubara::qp_lin_diagonal(S2, wn2, Vhf_loc_1D(I).real(), n_clamped);
      }
    } else { // "mats_gmatch"
      qp_matsubara::gmatch_opts opt;
      opt.wpow = qp_params.qp_map_wpow;
      app_log(2, "  - gmatch weights (reportable):                w_n = (w0/w_n)^{}", opt.wpow);
      nda::array<ComplexType, 1> S2(2);
      nda::array<ComplexType, 3> G(npos, 1, 1);
      nda::array<ComplexType, 2> H(1, 1);
      double rmax = 0.0;
      for (size_t I = 0; I < dim1; ++I) {
        const double vhf = Vhf_loc_1D(I).real();
        for (long m = 0; m < npos; ++m)
          G(m, 0, 0) = 1.0 / (ComplexType(mu - vhf, wp(m)) - Sigma_loc_2D(widx[m], I));
        S2(0) = Sigma_loc_2D(widx[0], I);
        S2(1) = Sigma_loc_2D(widx[1], I);
        H(0, 0) = ComplexType(qp_matsubara::qp_lin_diagonal(S2, wn2, vhf, n_clamped, opt.z_floor));
        auto info = qp_matsubara::qp_gmatch_block(G, wp, mu, H, opt);
        if (not info.converged) ++n_noconv;
        rmax = std::max(rmax, info.r);
        E_loc_1D(I) = H(0, 0);
      }
      rmax = comm->all_reduce_value(rmax, boost::mpi3::max<>{});
      app_log(2, "  - gmatch final weighted residual (max over states): {:.3e}", rmax);
    }
    n_clamped = comm->all_reduce_value(n_clamped, std::plus<>{});
    n_noconv = comm->all_reduce_value(n_noconv, std::plus<>{});
    if (n_clamped > 0)
      app_log(2, "  - Z-window clamps: {} of {} states", n_clamped, ns * nkpts * nbnd);
    if (n_noconv > 0)
      app_warning("solve_qp_eqn: gmatch hit maxiter on {} states.", n_noconv);

    } else {

    analyt_cont::AC_t AC(qp_params.ac_alg);
    app_log(2, "\n* Solving quasiparticle equation for given Sigma(iw): ");
    app_log(2, "  - processor grid for quasi-particle equation: (s, k, a) = ({}, {}, {})", 1, nkpools, np_a);
    app_log(2, "  - quasi-particle equation algorithm:          {}", qp_params.qp_type);
    app_log(2, "  - ac algorithm:                               {}", qp_params.ac_alg);
    app_log(2, "  - eta:                                        {}", qp_params.eta);
    app_log(2, "  - tolerance for quasi-particle equation:      {}\n", qp_params.tol);
    AC.init(iw_mesh, Sigma_loc_2D, qp_params.Nfit);
    if (qp_params.qp_type == "sc" or qp_params.qp_type == "sc_bisection") {
      double res;
      for (size_t I = 0; I < dim1; ++I) {
        std::tie(E_loc_1D(I), res) =
            qp_eqn_bisection(Vhf_loc_1D(I).real(), AC, I, mu, E_loc_1D(I).real(), qp_params.tol, qp_params.eta);
      }
    } else if (qp_params.qp_type == "sc_newton") {
      bool conv;
      double res;
      for (size_t I = 0; I < dim1; ++I) {
        std::tie(E_loc_1D(I), res, conv) =
            qp_eqn_secant(Vhf_loc_1D(I).real(), AC, I, mu, E_loc_1D(I).real(), 400, qp_params.tol, qp_params.eta);
        if (!conv) {
          auto [is, ik, ia] = I_to_ska(I);
          app_warning("secant method fails to converge at (s,k,a) = ({},{},{}); residual = {}", is, ik, ia, res);
        }
      }
    } else if (qp_params.qp_type == "linearized") {
      for (size_t I = 0; I < dim1; ++I) {
        E_loc_1D(I) = qp_eqn_linearized(Vhf_loc_1D(I).real(), AC, I, mu, E_loc_1D(I).real(), qp_params.eta);
      }
    } else if (qp_params.qp_type == "spectral") {
      bool conv;
      for (size_t I = 0; I < dim1; ++I) {
        std::tie(E_loc_1D(I), conv) = qp_eqn_spectral(Vhf_loc_1D(I).real(), AC, I, mu, E_loc_1D(I).real(), qp_params.tol, qp_params.eta);
        if (!conv) {
          auto [is, ik, ia] = I_to_ska(I);
          app_warning("spectral method fails to converge at (s,k,a) = ({},{},{})", is, ik, ia);
        }
      }
    } else {
      utils::check(false, "solve_qp_eqn: unknown type of qp equation: {}", qp_params.qp_type);
    }

    } // qp_map dispatch
  }
  dSigma_wska.reset();
  dVhf_ska.reset();

  sE_ska.set_zero();
  {
    auto E_ska_loc = dE_ska.local();
    sE_ska.win().fence();
    sE_ska.local()(s_rng, k_rng, a_rng) = E_ska_loc;
    sE_ska.win().fence();
    sE_ska.all_reduce();
  }
  dE_ska.reset();
  comm->barrier();
}

template<typename eri_t, typename corr_solver_t>
void add_evscf_vcorr(MBState &mb_state,
                     double mu,
                     solvers::mb_solver_t<corr_solver_t> &mb_solver,
                     eri_t &eri,
                     const imag_axes_ft::IAFT &FT,
                     qp_params_t &qp_params, 
                     bool fixed_w) {
  using math::shm::make_shared_array;

  auto mpi = eri.mpi();

  auto& sHeff_skij = mb_state.sHeff_skij.value();
  auto& sMO_skia = mb_state.sMO_skia.value();
  auto& sE_ska = mb_state.sE_ska.value();
  
  auto [ns, nkpts, nbnd, nbnd2] = sHeff_skij.shape();
  auto nt = FT.nt_f();

  // Evaluate dynamic self-energy and solve the quasiparticle equation.
  mb_state.sSigma_tskij.emplace(make_shared_array<Array_view_5D_t>(*mpi, {nt, ns, nkpts, nbnd, nbnd}));
  mb_state.sG_tskij.emplace(make_shared_array<Array_view_5D_t>(*mpi, {nt, ns, nkpts, nbnd, nbnd}));
  {
    update_G(mb_state.sG_tskij.value(), sMO_skia, sE_ska, mu, FT);
    FT.check_leakage(mb_state.sG_tskij.value(), imag_axes_ft::fermion, "Green's function");
    
    // update dyanmically screened interaction in mb_state if necessary.
    if (not fixed_w or not mb_state.dW_qtPQ.has_value()) {
      utils::check(mb_solver.scr_eri!=nullptr, "add_evscf_vcorr: mb_solver.scr_eri == nullptr when update_W is true.");
      mb_solver.scr_eri->update_w(mb_state, eri, mb_solver.corr->iter());
    }
    
    mb_solver.corr->evaluate(mb_state, eri);
    FT.check_leakage(mb_state.sSigma_tskij.value(), imag_axes_ft::fermion, "Self-energy");
    mpi->comm.barrier();
  }

  // Project 2 increment QM3: the mode-A context must be built while dW is still alive
  // (it is freed just below when keep_scr_coulomb_fixed is off). evGW needs the DIAGONAL
  // residues replicated -- solve_qp_eqn distributes (s,k,a) on its own processor grid.
  qp_modea::modea_ctx modea_ctx;
  build_modea_ctx_if_needed(modea_ctx, mb_state, mb_solver, eri, sMO_skia, sE_ska, mu, FT,
                            qp_params, true);

  // deallocate dynamically screened interaction if it is not fixed for the next iteration.
  if (not fixed_w) {
    mb_state.dW_qtPQ.reset();
  }

  // Solve the QP equation to get new QP energies.
  solve_qp_eqn(sE_ska, mb_state.sSigma_tskij.value(), sHeff_skij, sMO_skia, mu, FT, qp_params,
               modea_ctx.active ? std::addressof(modea_ctx) : nullptr);
  mb_state.sG_tskij.reset();
  mb_state.sSigma_tskij.reset();

  /** Update sHeff_skij with the new QP energies while keeping MO coefficients the same */
  // Basis transformation back to the primary basis.
  auto sMOinv_skai = make_shared_array<Array_view_4D_t>(*mpi, {ns, nkpts, nbnd, nbnd});
  sMOinv_skai.win().fence();
  for (size_t sk = mpi->node_comm.rank(); sk < ns*nkpts; sk += mpi->node_comm.size()) {
    size_t is = sk / nkpts;
    size_t ik = sk % nkpts;
    auto MO = make_matrix_view(sMO_skia.local()(is, ik, nda::ellipsis{}));
    sMOinv_skai.local()(is, ik, nda::ellipsis{}) = nda::inverse(MO);
  }
  sMOinv_skai.win().fence();

  sHeff_skij.win().fence();
  if (mpi->node_comm.rank() < ns*nkpts) {
    nda::array<ComplexType, 2> C_aj(nbnd, nbnd);
    nda::array<ComplexType, 2> Cdag_ia(nbnd, nbnd);
    for (size_t sk = mpi->node_comm.rank(); sk < ns * nkpts; sk += mpi->node_comm.size()) {
      size_t is = sk / nkpts;
      size_t ik = sk % nkpts;
      C_aj = sMOinv_skai.local()(is, ik, nda::ellipsis{});
      Cdag_ia = nda::transpose(nda::conj(C_aj));
      for (size_t a = 0; a < nbnd; ++a) {
        C_aj(a, nda::range::all) *= sE_ska.local()(is, ik, a);
      }
      nda::blas::gemm(Cdag_ia, C_aj, sHeff_skij.local()(is, ik, nda::ellipsis{}));
    }
  }
  sHeff_skij.win().fence();
}

auto qp_approx(const sArray_t<Array_view_5D_t> &sSigma_tskij,
               const sArray_t<Array_view_4D_t> &sMO_skia,
               const sArray_t<Array_view_3D_t> &sE_ska, double mu,
               const imag_axes_ft::IAFT &FT, qp_params_t &qp_params,
               const sArray_t<Array_view_4D_t> *sHstat_skij,
               const qp_modea::modea_ctx *modea_ctx)
               -> sArray_t<Array_view_4D_t> {
  using math::shm::make_shared_array;
  using math::nda::make_distributed_array;
  using local_Array_5D_t = nda::array<ComplexType, 5>;

  auto comm = sSigma_tskij.communicator();
  auto internode_comm = sSigma_tskij.internode_comm();
  auto node_comm = sSigma_tskij.node_comm();
  auto [ns, nkpts, nbnd] = sE_ska.shape();
  auto nt = FT.nt_f();
  auto nw = FT.nw_f();

  int np = comm->size();
  long nkpools = utils::find_proc_grid_max_npools(np, nkpts, 0.2);
  np /= nkpools;
  long np_a = utils::find_proc_grid_min_diff(np, 1, 1);
  long np_b = np / np_a;
  utils::check(nkpools > 0 and nkpools <= nkpts,
               "qp_approx:: nkpools <= 0 or nkpools > nkpts. nkpools = {}", nkpools);
  utils::check(comm->size() % nkpools == 0, "qp_approx:: gcomm.size() % nkpools != 0");
  utils::check(np_a < nbnd and np_b < nbnd, "qp_approx: np_a({}) or np_b({}) > nbnd({})", np_a, np_b, nbnd);

  auto dSigma_wskab = make_distributed_array<local_Array_5D_t>(*comm, {1, 1, nkpools, np_a, np_b},
                                                               {nw, ns, nkpts, nbnd, nbnd}, {1, 1, 1, 1, 1});
  auto s_rng = dSigma_wskab.local_range(1);
  auto k_rng = dSigma_wskab.local_range(2);
  auto a_rng = dSigma_wskab.local_range(3);
  auto b_rng = dSigma_wskab.local_range(4);
  auto [nw_loc, ns_loc, nk_loc, na_loc, nb_loc] = dSigma_wskab.local_shape();

  // Project 2 increment QM3: the tau-domain ORACLE for the anchor discrepancy needs the
  // MO-basis Sigma^c(tau) BEFORE tau_to_w, i.e. the reference as the solver actually built
  // it, with no fermionic transform applied. Only allocated for mode_a.
  std::optional<sArray_t<Array_view_5D_t>> sSigma_tskab_shm;
  if (qp_params.qp_map == "mode_a" or qp_params.qp_map == "mode_b") {
    sSigma_tskab_shm.emplace(make_shared_array<Array_view_5D_t>(
        *comm, *internode_comm, *node_comm, {nt, ns, nkpts, nbnd, nbnd}));
    sSigma_tskab_shm->set_zero();
  }

  // ------ basis transform from primary to MO basis ------
  {
    auto dSigma_tskab = make_distributed_array<local_Array_5D_t>(*comm, {1, 1, nkpools, np_a, np_b},
                                                               {nt, ns, nkpts, nbnd, nbnd},
                                                               {1, 1, 1, 1, 1});
    auto Sigma_tskab_loc = dSigma_tskab.local();

    nda::array<ComplexType, 2> C_jb(nbnd, nb_loc);
    nda::array<ComplexType, 2> SigmaC_ib(nbnd, nb_loc);
    nda::array<ComplexType, 2> Cdag_ai(na_loc, nbnd);
    auto Sigma_ab = C_jb(nda::range(na_loc), nda::range::all);
    for (size_t it = 0; it < nt; ++it) {
      for (auto [is_loc, is]: itertools::enumerate(s_rng)) {
        for (auto [ik_loc, ik]: itertools::enumerate(k_rng)) {
          C_jb = sMO_skia.local()(is, ik, nda::range::all, b_rng);
          nda::blas::gemm(sSigma_tskij.local()(it, is, ik, nda::ellipsis{}), C_jb, SigmaC_ib);

          auto C_ia = sMO_skia.local()(is, ik, nda::range::all, a_rng);
          Cdag_ai = nda::transpose(nda::conj(C_ia));
          nda::blas::gemm(Cdag_ai, SigmaC_ib, Sigma_ab);
          Sigma_tskab_loc(it, is_loc, ik_loc, nda::range::all, nda::range::all) = Sigma_ab;
        }
      }
    }
    if (sSigma_tskab_shm.has_value()) {
      sSigma_tskab_shm->win().fence();
      sSigma_tskab_shm->local()(nda::range::all, s_rng, k_rng, a_rng, b_rng) =
          dSigma_tskab.local();
      sSigma_tskab_shm->win().fence();
      sSigma_tskab_shm->all_reduce();
    }
    FT.tau_to_w(dSigma_tskab.local(), dSigma_wskab.local(), imag_axes_ft::fermion);
  }

  // Static approximation for V_QPGW
  long dim1 = ns_loc * nk_loc * na_loc * nb_loc;
  // local I index to global [s, k, a, b]
  // bypass clang openmp error: error: capturing a structured binding is not yet supported in OpenMP
  auto nk_loc_ = nk_loc;
  auto na_loc_ = na_loc;
  auto nb_loc_ = nb_loc;
  auto I_to_skab = [&](size_t I) {
    // I = s_loc*nk_loc*na_loc*nb_loc + k_loc*na_loc*nb_loc + a_loc*nb_loc + b_loc
    size_t s_loc = I / (nk_loc_*na_loc_*nb_loc_);
    size_t k_loc = ( I / (na_loc_*nb_loc_) ) % nk_loc_;
    size_t a_loc = ( I / nb_loc_ ) % na_loc_;
    size_t b_loc = I % nb_loc_;
    return std::make_tuple(s_rng.first()+s_loc, k_rng.first()+k_loc, a_rng.first()+a_loc, b_rng.first()+b_loc);
  };

  auto n_to_iw = nda::map([&](int n) { return FT.omega(n); });
  nda::array<ComplexType, 1> iw_mesh(n_to_iw(FT.wn_mesh()));

  // ---- Project 2 increment Q6 (notes/q6_diagnostics_closeout_spec.md §1.3), part 1 ----
  // THE LINESHAPE METER, input half: the MO-basis DIAGONAL Sigma^c_aa at the first and the
  // highest positive fermionic node. Gathered HERE because this is the only point at which
  // dSigma_wskab is alive for EVERY map -- the Matsubara branch below resets it, and the
  // ac_pade branch consumes it in place. ADDITIVE: nothing downstream reads these arrays.
  nda::array<ComplexType, 3> q6_sig_w0(ns, nkpts, nbnd), q6_sig_wtop(ns, nkpts, nbnd);
  q6_sig_w0() = ComplexType(0.0);
  q6_sig_wtop() = ComplexType(0.0);
  double q6_w0 = -1.0, q6_wtop = -1.0;
  {
    // NOT positive_wn_nodes(): that helper hard-checks ">= 2 positive nodes" for the
    // Matsubara maps, and a REPORTING hook must never be able to abort the ac_pade path.
    auto w_rng = dSigma_wskab.local_range(0);
    long g_w0 = -1, g_wtop = -1;
    double lo = 1e300, hi = -1.0;
    for (long n = 0; n < iw_mesh.shape(0); ++n) {
      const double x = iw_mesh(n).imag();
      if (x <= 0.0) continue;
      if (x < lo) { lo = x; g_w0 = n; }
      if (x > hi) { hi = x; g_wtop = n; }
    }
    if (g_w0 >= 0 and g_wtop >= 0) {
      q6_w0 = lo;
      q6_wtop = hi;
      const bool have_w0 = (g_w0 >= w_rng.first() and g_w0 < w_rng.first() + long(w_rng.size()));
      const bool have_wt = (g_wtop >= w_rng.first() and g_wtop < w_rng.first() + long(w_rng.size()));
      for (auto [is_loc, is]: itertools::enumerate(s_rng))
        for (auto [ik_loc, ik]: itertools::enumerate(k_rng))
          for (auto [ia_loc, ia]: itertools::enumerate(a_rng))
            for (auto [ib_loc, ib]: itertools::enumerate(b_rng)) {
              if (ia != ib) continue;            // the (a,b) grid is disjoint => no double count
              if (have_w0)
                q6_sig_w0(is, ik, ia) = dSigma_wskab.local()(g_w0 - w_rng.first(),
                                                             is_loc, ik_loc, ia_loc, ib_loc);
              if (have_wt)
                q6_sig_wtop(is, ik, ia) = dSigma_wskab.local()(g_wtop - w_rng.first(),
                                                               is_loc, ik_loc, ia_loc, ib_loc);
            }
      comm->all_reduce_in_place_n(q6_sig_w0.data(), q6_sig_w0.size(), std::plus<>{});
      comm->all_reduce_in_place_n(q6_sig_wtop.data(), q6_sig_wtop.size(), std::plus<>{});
    }
  }

  auto sVcorr_skij = make_shared_array<Array_view_4D_t>(*comm, *internode_comm, *node_comm, {ns, nkpts, nbnd, nbnd});

  if (qp_params.qp_map != "ac_pade") {

  // Project 2 increment Q2 (notes/qpgw_edmft_implementation_plan.md): the
  // Matsubara-native maps (qp_maps_matsubara.hpp) act on whole (a,b) blocks per
  // (s,k), so gather the MO-basis Sigma(iw) tiles into a shared array (disjoint
  // writes + zeros-elsewhere all_reduce) and round-robin the blocks over ranks.
  utils::check(qp_params.qp_map == "mats_lin" or sHstat_skij != nullptr,
               "qp_approx: qp_map = {} needs the static Heff, which only the "
               "qp-scf loop call path provides; use ac_pade or mats_lin here.",
               qp_params.qp_map);
  auto sSigma_wskab = make_shared_array<Array_view_5D_t>(*comm, *internode_comm, *node_comm,
                                                         {nw, ns, nkpts, nbnd, nbnd});
  sSigma_wskab.set_zero();
  sSigma_wskab.win().fence();
  sSigma_wskab.local()(nda::range::all, s_rng, k_rng, a_rng, b_rng) = dSigma_wskab.local();
  sSigma_wskab.win().fence();
  sSigma_wskab.all_reduce();
  dSigma_wskab.reset();

  auto [wp, widx] = positive_wn_nodes(iw_mesh);
  const long npos = wp.shape(0);
  nda::array<double, 1> wn2(2);
  wn2(0) = wp(0);
  wn2(1) = wp(1);
  long n_clamped = 0, n_noconv = 0;
  decltype(nda::range::all) all;
  app_log(2, "\n* Applying the Matsubara-native static map to Sigma(iw): ");
  app_log(2, "  - quasiparticle map:        {} (no AC)", qp_params.qp_map);
  app_log(2, "  - positive fermionic nodes: {} (w0 = {:.6f}, w1 = {:.6f})", npos, wp(0), wp(1));
  if (qp_params.qp_map == "mode_a" or qp_params.qp_map == "mode_b")
    app_log(2, "  - mode knobs:             route = {}, nconsist = {}, consist_tol = {:.1e}, "
               "eta = {:.3g}, wsupp = {}, wfit = {}, wrtol = {:.2g}, wrank = {:.2g}, "
               "wsketch = {}, wunion = {:.2g}",
            qp_params.qp_modea_route, qp_params.qp_modea_nconsist,
            qp_params.qp_modea_consist_tol, qp_params.qp_modea_eta,
            qp_params.qp_modea_wsupp, qp_params.qp_modea_wfit, qp_params.qp_modea_wrtol,
            qp_params.qp_modea_wrank, qp_params.qp_modea_wsketch,
            qp_params.qp_modea_wunion);
  sVcorr_skij.set_zero();
  sVcorr_skij.win().fence();
  if (qp_params.qp_map == "mode_a" or qp_params.qp_map == "mode_b") {

  // ---- Project 2 increment QM3 (notes/qm3_mode_a_loop_spec.md): the MODE-A map ----
  // V^xc_ab = 1/2 [ Sigma^c_ab(eps_a) + Sigma^c_ab(eps_b) ] with Sigma^c from the QM2
  // contour-deformation kernel, driven to inner QP consistency at FIXED Sigma data. The
  // Hermitize + MO -> primary tail below is the existing one, unchanged.
  modea_run(sVcorr_skij, sSigma_wskab, sSigma_tskab_shm.value(), sMO_skia, sE_ska,
            *sHstat_skij, wp, widx, mu, FT, qp_params.qp_map, modea_ctx, *comm);

  } else if (qp_params.qp_map == "mats_lin") {
    nda::array<ComplexType, 3> S2(2, nbnd, nbnd);
    for (long sk = comm->rank(); sk < long(ns * nkpts); sk += comm->size()) {
      long is = sk / nkpts, ik = sk % nkpts;
      S2(0, all, all) = sSigma_wskab.local()(widx[0], is, ik, all, all);
      S2(1, all, all) = sSigma_wskab.local()(widx[1], is, ik, all, all);
      sVcorr_skij.local()(is, ik, all, all) = qp_matsubara::qp_lin_matrix(S2, wn2, n_clamped);
    }
  } else { // "mats_gmatch"
    qp_matsubara::gmatch_opts opt;
    opt.wpow = qp_params.qp_map_wpow;
    app_log(2, "  - gmatch weights (reportable): w_n = (w0/w_n)^{}", opt.wpow);
    nda::array<ComplexType, 3> S2(2, nbnd, nbnd), G(npos, nbnd, nbnd);
    nda::array<ComplexType, 2> Hstat_ab(nbnd, nbnd), H(nbnd, nbnd), tmp(nbnd, nbnd);
    nda::matrix<ComplexType> K(nbnd, nbnd);
    double rmax = 0.0;
    for (long sk = comm->rank(); sk < long(ns * nkpts); sk += comm->size()) {
      long is = sk / nkpts, ik = sk % nkpts;
      auto MO = sMO_skia.local()(is, ik, all, all); // (i, a)
      // the static part in the MO basis
      nda::blas::gemm(ComplexType(1.0), sHstat_skij->local()(is, ik, nda::ellipsis{}), MO,
                      ComplexType(0.0), tmp);
      nda::blas::gemm(ComplexType(1.0), nda::dagger(MO), tmp, ComplexType(0.0), Hstat_ab);
      // the target G^GW on the positive nodes (spec principle 2: Sigma^GW only)
      for (long m = 0; m < npos; ++m) {
        auto Sw = sSigma_wskab.local()(widx[m], is, ik, all, all);
        for (long i = 0; i < nbnd; ++i)
          for (long j = 0; j < nbnd; ++j) K(i, j) = -Hstat_ab(i, j) - Sw(i, j);
        for (long i = 0; i < nbnd; ++i) K(i, i) += ComplexType(mu, wp(m));
        nda::inverse_in_place(K);
        G(m, all, all) = K;
      }
      // map-(i) initializer
      S2(0, all, all) = sSigma_wskab.local()(widx[0], is, ik, all, all);
      S2(1, all, all) = sSigma_wskab.local()(widx[1], is, ik, all, all);
      H = Hstat_ab + qp_matsubara::qp_lin_matrix(S2, wn2, n_clamped, opt.z_floor);
      auto info = qp_matsubara::qp_gmatch_block(G, wp, mu, H, opt);
      if (not info.converged) ++n_noconv;
      rmax = std::max(rmax, info.r);
      sVcorr_skij.local()(is, ik, all, all) = H - Hstat_ab;
    }
    rmax = comm->all_reduce_value(rmax, boost::mpi3::max<>{});
    app_log(2, "  - gmatch final weighted residual (max over blocks): {:.3e}", rmax);
  }
  n_clamped = comm->all_reduce_value(n_clamped, std::plus<>{});
  n_noconv = comm->all_reduce_value(n_noconv, std::plus<>{});
  if (n_clamped > 0)
    app_log(2, "  - Z-window clamps: {} eigenvalues", n_clamped);
  if (n_noconv > 0)
    app_warning("qp_approx: gmatch hit maxiter on {} (s,k) blocks.", n_noconv);
  sVcorr_skij.win().fence();
  sVcorr_skij.all_reduce();

  } else {

  analyt_cont::AC_t AC(qp_params.ac_alg);
  app_log(2, "\n* Applying the static approximation (Phys. Rev. Lett. 93, 126406) to Sigma(w): ");
  app_log(2, "  - processor grid for V_QPGW : (s, k, a, b) = ({}, {}, {})", 1, nkpools, np_a, np_b);
  app_log(2, "  - ac algorithm:               {}", qp_params.ac_alg);
  app_log(2, "  - eta:                        {}", qp_params.eta);
  app_log(2, "  - off-diagonal mode:          {}\n", qp_params.off_diag_mode);
  auto Sigma_loc_2D = nda::reshape(dSigma_wskab.local(), std::array<long, 2>{nw, dim1});
  AC.init(iw_mesh, Sigma_loc_2D, qp_params.Nfit);

  sVcorr_skij.win().fence();
  for (size_t I = 0; I < dim1; ++I) {
    auto [s, k, a, b] = I_to_skab(I);
    if (qp_params.off_diag_mode == "qp_energy") {
      double eps_a = sE_ska.local()(s, k, a).real() - mu;
      double eps_b = sE_ska.local()(s, k, b).real() - mu;
      sVcorr_skij.local()(s, k, a, b) = 0.5 * ( AC.evaluate(ComplexType(eps_a, qp_params.eta), I)
                                                 + AC.evaluate(ComplexType(eps_b, qp_params.eta), I) );
    } else if (qp_params.off_diag_mode == "fermi") {
      double eps_a = (a == b)? sE_ska.local()(s, k, a).real() - mu : 0.0;
      sVcorr_skij.local()(s, k, a, b) = AC.evaluate(ComplexType(eps_a, qp_params.eta), I);
    } else {
      utils::check(false, "qp_approx: unknown off-diagonal mode: {}", qp_params.off_diag_mode);
    }
  }
  sVcorr_skij.win().fence();
  sVcorr_skij.all_reduce();

  } // qp_map dispatch

  // ---- Project 2 increment Q6 (spec §1.3), part 2: THE LINESHAPE METER, output half ----
  // sVcorr_skij still holds V^xc in the MO BASIS here (the Hermitize + MO -> primary tail is
  // below), i.e. the map's own output measured against the map's own input -- which is what
  // "what the static map discards" means. Reading it AFTER the Hermitization would only drop
  // Im on the diagonal, and that drop is itself part of the discard.
  sVcorr_skij.win().fence();
  {
    auto &LS = q6_lineshape();
    LS = q6_lineshape_t{};
    LS.w0 = q6_w0;
    LS.wtop = q6_wtop;
    if (q6_w0 > 0.0) {
      // eps_floor: the denominator guard of spec §1.3. At 1e-12 a.u. it sits ~10 orders below
      // any Sigma^c this map is applied to, so it fires only on an exact zero.
      constexpr double eps_floor = 1e-12;
      double m0 = 0.0, mt = 0.0, s0 = 0.0, st = 0.0;
      double a0 = 0.0, at = 0.0, b0 = 0.0, bt = 0.0;   // the ABSOLUTE discard, a.u.
      long cnt = 0;
      std::vector<long> occ, emp, win;
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nkpts; ++ik) {
          // The gap window, built with the SAME rule as the mode-A diagnostics
          // (qp_scf_common.cpp:255-267): the two highest occupied + the two lowest empty
          // states of the INCOMING qp spectrum. Map-independent by construction, so the
          // meter compares the same states across ac_pade / mats_* / mode_a / mode_b.
          occ.clear(); emp.clear(); win.clear();
          for (long a = 0; a < nbnd; ++a)
            ((sE_ska.local()(is, ik, a).real() < mu) ? occ : emp).push_back(a);
          std::sort(occ.begin(), occ.end(), [&](long x, long y) {
            return sE_ska.local()(is, ik, x).real() > sE_ska.local()(is, ik, y).real(); });
          std::sort(emp.begin(), emp.end(), [&](long x, long y) {
            return sE_ska.local()(is, ik, x).real() < sE_ska.local()(is, ik, y).real(); });
          for (size_t t = 0; t < std::min<size_t>(2, occ.size()); ++t) win.push_back(occ[t]);
          for (size_t t = 0; t < std::min<size_t>(2, emp.size()); ++t) win.push_back(emp[t]);
          for (long a : win) {
            const ComplexType V = sVcorr_skij.local()(is, ik, a, a);
            const ComplexType S0 = q6_sig_w0(is, ik, a);
            const ComplexType St = q6_sig_wtop(is, ik, a);
            const double d0 = std::abs(S0 - V), dt = std::abs(St - V);
            const double f0 = d0 / std::max(std::abs(S0), eps_floor);
            const double ft = dt / std::max(std::abs(St), eps_floor);
            m0 = std::max(m0, f0);
            mt = std::max(mt, ft);
            s0 += f0;
            st += ft;
            a0 = std::max(a0, d0);
            at = std::max(at, dt);
            b0 += d0;
            bt += dt;
            ++cnt;
          }
        }
      if (cnt > 0) {
        LS.frac_w0_max = m0;
        LS.frac_w0_mean = s0 / double(cnt);
        LS.frac_top_max = mt;
        LS.frac_top_mean = st / double(cnt);
        LS.abs_w0_max = a0;
        LS.abs_w0_mean = b0 / double(cnt);
        LS.abs_top_max = at;
        LS.abs_top_mean = bt / double(cnt);
        LS.n_states = cnt;
      }
    }
    app_log(2, "  [Q6] lineshape meter ({} gap-window diagonals): "
               "|Sigma^c_aa - V^xc_aa| / |Sigma^c_aa| at iw_0 = {:.6f} a.u.: max {:.6e}, "
               "mean {:.6e}; at iw_top = {:.6f} a.u.: max {:.6e}, mean {:.6e}. "
               "ABSOLUTE discard |Sigma^c_aa - V^xc_aa| (a.u.): iw_0 max {:.6e} mean {:.6e}, "
               "iw_top max {:.6e} mean {:.6e}",
            LS.n_states, LS.w0, LS.frac_w0_max, LS.frac_w0_mean,
            LS.wtop, LS.frac_top_max, LS.frac_top_mean,
            LS.abs_w0_max, LS.abs_w0_mean, LS.abs_top_max, LS.abs_top_mean);
  }

  // prepare for inverse transformation from MO to primary basis
  auto sMOinv_skai = make_shared_array<Array_view_4D_t>(*comm, *internode_comm, *node_comm, {ns, nkpts, nbnd, nbnd});
  sMOinv_skai.win().fence();
  for (size_t sk = node_comm->rank(); sk < ns*nkpts; sk += node_comm->size()) {
    size_t is = sk / nkpts;
    size_t ik = sk % nkpts;
    auto MO = make_matrix_view(sMO_skia.local()(is, ik, nda::ellipsis{}));
    sMOinv_skai.local()(is, ik, nda::ellipsis{}) = nda::inverse(MO);
  }
  sMOinv_skai.win().fence();

  // Hermitize V_QPGW and then do basis transformation from MO to primary basis
  sVcorr_skij.win().fence();
  if (node_comm->rank() < ns*nkpts) {
    nda::array<ComplexType, 2> V_QPGW_ab(nbnd, nbnd);
    nda::array<ComplexType, 2> VC_aj(nbnd, nbnd);
    nda::array<ComplexType, 2> Cdag_ia(nbnd, nbnd);
    for (size_t sk = node_comm->rank(); sk < ns*nkpts; sk += node_comm->size()) {
      size_t is = sk / nkpts;
      size_t ik = sk % nkpts;

      // Extract the Hermitian part of V_QPGW since V_QPGW in principle in non-Hermitian
      V_QPGW_ab = 0.5 * ( sVcorr_skij.local()(is, ik, nda::ellipsis{})
                          + nda::transpose(nda::conj(sVcorr_skij.local()(is, ik, nda::ellipsis{}))) );

      nda::blas::gemm(V_QPGW_ab, sMOinv_skai.local()(is,ik,nda::ellipsis{}), VC_aj);
      Cdag_ia = nda::transpose(nda::conj(sMOinv_skai.local()(is,ik,nda::ellipsis{})));
      nda::blas::gemm(Cdag_ia, VC_aj, sVcorr_skij.local()(is, ik, nda::ellipsis{}));
    }
  }
  sVcorr_skij.win().fence();
  return sVcorr_skij;
}

template<typename eri_t, typename corr_solver_t>
void add_qpscf_vcorr(MBState &mb_state,
                     double mu,
                     solvers::mb_solver_t<corr_solver_t> &mb_solver,
                     eri_t &eri,
                     const imag_axes_ft::IAFT &FT,
                     qp_params_t &qp_params,
                     const sArray_t<Array_view_5D_t> *sG_ext) {
  using math::shm::make_shared_array;

  auto& sHeff_skij = mb_state.sHeff_skij.value();
  auto& sMO_skia = mb_state.sMO_skia.value();
  auto& sE_ska = mb_state.sE_ska.value();
  auto mpi = eri.mpi();
  auto [ns, nkpts, nbnd, nbnd2] = sHeff_skij.shape();
  auto nt = FT.nt_f();

  auto &qpt = qp_stage_timer();   // T-1 item 2: instrument the previously untimed QP stage
  qpt.start("QP_G_BUILD");

  mb_state.sSigma_tskij.emplace(make_shared_array<Array_view_5D_t>(*mpi, {nt, ns, nkpts, nbnd, nbnd}));
  mb_state.sG_tskij.emplace(make_shared_array<Array_view_5D_t>(*mpi, {nt, ns, nkpts, nbnd, nbnd}));
  if (sG_ext == nullptr) {
    update_G(mb_state.sG_tskij.value(), sMO_skia, sE_ska, mu, FT);
  } else {
    // Project 2 increment Q5 (spec §1): the re-QP-ization step. The external G replaces the
    // analytic QP G for BOTH consumers below -- update_w (so W_corr screens with
    // P^RPA[G_ext] + P^lad + P_C(P_imp-P_dc)P_C^dag) and the Sigma^GW build. The MAP stage
    // downstream (qp_approx / the mode-A CD kernel) is untouched: it consumes the MO-basis
    // Sigma gather + mb_state.dW_qtPQ, both built from whatever G is here.
    // element-wise dims: rusty's bundled fmt has no std::array formatter
    auto gx = sG_ext->shape();
    auto ge = mb_state.sG_tskij.value().shape();
    utils::check(gx == ge,
                 "add_qpscf_vcorr: external Green's function shape ({},{},{},{},{}) != "
                 "expected ({},{},{},{},{}).",
                 gx[0], gx[1], gx[2], gx[3], gx[4], ge[0], ge[1], ge[2], ge[3], ge[4]);
    auto &sG = mb_state.sG_tskij.value();
    sG.win().fence();
    if (mpi->node_comm.root()) sG.local() = sG_ext->local();
    sG.win().fence();
    mpi->comm.barrier();
  }
  FT.check_leakage(mb_state.sG_tskij.value(), imag_axes_ft::fermion, "Green's function");
  qpt.stop("QP_G_BUILD");

  // compute screen interaction
  utils::check(mb_solver.scr_eri!=nullptr, "add_qpscf_vcorr: mb_solver.scr_eri == nullptr.");
  qpt.start("QP_SCR_COULOMB");
  mb_solver.scr_eri->update_w(mb_state, eri, mb_solver.corr->iter());
  qpt.stop("QP_SCR_COULOMB");

  // compute dynamic self-energy in the primary basis
  qpt.start("QP_SIGMA");
  mb_solver.corr->evaluate(mb_state, eri);
  FT.check_leakage(mb_state.sSigma_tskij.value(), imag_axes_ft::fermion, "Self-energy");
  qpt.stop("QP_SIGMA");

  // Project 2 increment QM3: build the mode-A evaluator context HERE -- this is the only
  // window in which mb_state.dW_qtPQ is alive (it is reset below).
  qpt.start("QP_MAP");
  qp_modea::modea_ctx modea_ctx;
  build_modea_ctx_if_needed(modea_ctx, mb_state, mb_solver, eri, sMO_skia, sE_ska, mu, FT,
                            qp_params, false);

  // Add the correlation contribution to the QP Hamiltonian. At this point
  // sHeff_skij holds the static (H0 + HF) part only -- the map-(ii) reference.
  auto sVcorr_skij = qp_approx(mb_state.sSigma_tskij.value(),  sMO_skia, sE_ska, mu, FT, qp_params,
                               std::addressof(sHeff_skij),
                               modea_ctx.active ? std::addressof(modea_ctx) : nullptr);
  if (mpi->node_comm.root()) sHeff_skij.local() += sVcorr_skij.local();
  mpi->comm.barrier();
  qpt.stop("QP_MAP");

  // deallocation
  mb_state.dW_qtPQ.reset();
  mb_state.sG_tskij.reset();
  mb_state.sSigma_tskij.reset();
  mpi->comm.barrier();

  print_qp_stage_timers();
}



double compute_Nelec(double mu, const mf::MF &mf, const sArray_t<Array_view_3D_t> &sE_ski, double beta) {
  auto [ns, nkpts, nbnd] = sE_ski.shape();
  auto FD_occ = nda::map([&](ComplexType e) { return 1.0 / ( 1 + std::exp( (e.real()-mu) * beta ) ); });
  //nda::array<double, 3> nel_ski(ns, nkpts, nbnd);
  //nel_ski = FD_occ(sE_ski.local());
  nda::array<double, 1> nel_i(nbnd);

  auto k_weight = mf.k_weight();
  double nel = 0.0;
  for (size_t s = 0; s < ns; ++s) {
    for (size_t k = 0; k < nkpts; ++k) {
      nel_i = FD_occ(sE_ski.local()(s, k, nda::range::all));
      nel += k_weight(k) * nda::sum(nel_i);
    }
  }
  if (ns == 1 and mf.npol()==1) nel *= 2.0;
  //double nel = (ns == 2)? nda::sum(nel_ski) / nkpts : 2 * nda::sum(nel_ski) / nkpts;
  return nel;
}

template<typename X_t>
double update_mu(double old_mu, const mf::MF &mf, const X_t &sE_ski, double beta,
                 double mu_tol, std::string mu_update_alg) {
  double nel_target = mf.nelec();
  double delta = 0.2;
  auto eval_f = [&](double mu) {
    return compute_Nelec(mu, mf, sE_ski, beta) - nel_target;
  };
  double nel_old = compute_Nelec(old_mu, mf, sE_ski, beta);
  app_log(2, "Initial chemical potential (mu) = {}, nelec = {}", old_mu, nel_old);

  if (mu_update_alg == "bisection") {
    auto [mu, f_mu] = detail::update_mu_bisection_impl(old_mu, mu_tol, delta, eval_f);
    double nel = f_mu + nel_target;
    app_log(2, "Chemical potential found (mu) = {} a.u.", mu);
    app_log(2, "Number of electrons per unit cell = {}", nel);
    return mu;
  } else if (mu_update_alg == "midpoint") {
    auto [mu, f_mu, mu_left, mu_right] =
        detail::update_mu_midpoint_impl(old_mu, mu_tol, delta, eval_f);
    double nel = f_mu + nel_target;
    app_log(2, "Chemical potential bounds found (mu_left, mu_right) = ({}, {}) a.u.",
            mu_left, mu_right);
    app_log(2, "Chemical potential found (mu) = {} a.u.", mu);
    app_log(2, "Number of electrons per unit cell = {}", nel);
    return mu;
  }
  utils::check(false,
               "qp_scf_common.cpp::update_mu: unknown mu update algorithm {}.",
               mu_update_alg);
  return old_mu;
}

template<typename comm_t, typename X_t>
double solve_iterative(utils::mpi_context_t<comm_t> &context, iter_scf::iter_scf_t& iter_solver,
                       long it, std::string h5_prefix, X_t &sHeff_skij, const X_t &sS_skij) {
  double conv = 0;
  // Which "scf/iter{ref_it}" holds the object this iteration is measured against, and is it
  // a QP Hamiltonian at all?
  //
  // A qpGW run that RESTARTS from a plain-GW checkpoint -- which is exactly what the
  // GW+EDMFT lattice stage does (dmft/io.py always emits 'restart': True for the [qpgw]
  // block, and the workflow REQUIRES a pre-existing GW checkpoint) -- starts at
  // it = init_it + 1 with init_it > 0, so it never sees the it == 1 branch below. The
  // else branch then asks the iterative solver for "scf/iter{it-1}/Heff_skij", but a
  // checkpoint written by the DYSON scf holds F_skij (+ system/H0_skij), never Heff_skij.
  // nda::h5_read then threw INSIDE the `node_comm.root()` guard, i.e. on one rank only:
  // the root unwound qp_scf_loop's shared arrays into MPI_Win_free while every other rank
  // sat in the node_comm.broadcast_n below -- a hard deadlock, no error message
  // (observed 2026-08-17 on svo kp444 restart, qp_map=mode_a, iteration 13).
  //
  // Treat "no QP Hamiltonian in the previous iteration" exactly like the first qp
  // iteration: there is nothing to damp against yet, so report the change w.r.t. the
  // reference Hamiltonian and (for DIIS) start the history here. it == 1 keeps ref_it = 0
  // and is therefore bit-identical to the previous behaviour.
  const long ref_it = (it == 1) ? 0 : it - 1;
  const std::string ref_grp_name = "scf/iter" + std::to_string(ref_it);
  bool prev_is_qp = false;
  if (context.comm.root()) {
    h5::file file(h5_prefix + ".mbpt.h5", 'r');
    h5::group grp(file);
    prev_is_qp = grp.has_subgroup(ref_grp_name) and
                 grp.open_group(ref_grp_name).has_dataset("Heff_skij");
  }
  context.comm.broadcast_n(&prev_is_qp, 1, 0);

  if (it == 1 or not prev_is_qp) {
    // Just check changes w.r.t. mf (or, on a dyson-checkpoint restart, w.r.t. the
    // effective one-body Hamiltonian F + H0 of the iteration we are continuing from).
    if (context.node_comm.root()) {
      auto H_mf = nda::make_regular(sHeff_skij.local());
      std::string filename = h5_prefix + ".mbpt.h5";
      h5::file file(filename, 'r');
      h5::group grp(file);
      if (grp.has_subgroup(ref_grp_name)) {
        auto iter_grp = grp.open_group(ref_grp_name);
        if (iter_grp.has_dataset("Heff_skij")) {
          // checkpoint from a qp scf
          nda::h5_read(iter_grp, "Heff_skij", H_mf);
        } else if (iter_grp.has_dataset("F_skij")) {
          // checkpoint from a dyson scf
          nda::h5_read(iter_grp, "F_skij", H_mf);
          nda::array<ComplexType, 4> H0(H_mf.shape());
          auto sys_grp = grp.open_group("system");
          nda::h5_read(sys_grp, "H0_skij", H0);
          H_mf += H0;
        }
      }
      H_mf -= sHeff_skij.local();
      auto max_iter = max_element(H_mf.data(), H_mf.data()+H_mf.size(),
                                   [](auto a, auto b) { return std::abs(a) < std::abs(b); });
      conv =  std::abs((*max_iter));
    }
    context.node_comm.broadcast_n(&conv, 1, 0);
    if (iter_solver.iter_alg() == iter_scf::DIIS and context.comm.root()) {
      // Initialize DIIS solver at the root process since the solver currently doesn't support MPI
      iter_solver.initialize(sHeff_skij.local(), sS_skij.local(), h5_prefix);
    }
    context.comm.barrier();
  } else {
    // prev_is_qp is true here, so "scf/iter{it-1}/Heff_skij" exists on every rank's view of
    // the file. The check is kept as a collective-safe assertion: everything below runs on
    // the node root ONLY, so anything that throws there deadlocks the other ranks in the
    // broadcast instead of failing. utils::check aborts the job with a message instead.
    utils::check(prev_is_qp,
                 "qp_scf_common::solve_iterative: \"{}/Heff_skij\" is missing in {}.mbpt.h5, "
                 "so the qp Hamiltonian of the previous iteration cannot be damped against.",
                 ref_grp_name, h5_prefix);
    iter_solver.metadata_log();
    if (context.node_comm.root()) {
      std::string filename = h5_prefix + ".mbpt.h5";
      h5::file file(filename, 'r');
      h5::group grp(file);
      auto scf_grp = grp.open_group("scf");
      conv = iter_solver.solve(sHeff_skij.local(), "Heff_skij", scf_grp, it);
    }
    context.node_comm.broadcast_n(&conv, 1, 0);
  }
  context.comm.barrier();
  return conv;
}

void write_mf_data(mf::MF &mf, const imag_axes_ft::IAFT &ft,
                   hamilt::pseudopot &psp, std::string output) {
  auto mpi = mf.mpi();
  sArray_t<Array_view_4D_t> sHeff_skij(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  sArray_t<Array_view_4D_t> sH0_skij(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  sArray_t<Array_view_4D_t> sS_skij(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  sArray_t<Array_view_4D_t> sDm_skij(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  sArray_t<Array_view_4D_t> sMO_skia(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  sArray_t<Array_view_3D_t> sE_ska(math::shm::make_shared_array<Array_view_3D_t>(
      *mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd()}));
  double mu = 0.0;

  hamilt::set_fock(mf, std::addressof(psp), sHeff_skij, false);
  hamilt::set_H0(mf, std::addressof(psp), sH0_skij);
  hamilt::set_ovlp(mf, sS_skij);
  update_MOs(sMO_skia, sE_ska, sHeff_skij, sS_skij);
  mu = update_mu(mu, mf, sE_ska, ft.beta());
  update_Dm(sDm_skij, sMO_skia, sE_ska, mu, ft.beta());

  chkpt::write_metadata(mpi->comm, mf, ft, sH0_skij, sS_skij, output);
  chkpt::dump_scf(mpi->comm, 0, sDm_skij, sHeff_skij, sMO_skia, sE_ska, mu, output);
}

/** Instantiation of public template **/

template double update_mu(double, const mf::MF&, const sArray_t<Array_view_3D_t>&, double, double, std::string);

template void add_evscf_vcorr(MBState&, double, solvers::mb_solver_t<>&, thc_reader_t&, const imag_axes_ft::IAFT&, qp_params_t&, bool);
template void add_evscf_vcorr(MBState&, double, solvers::mb_solver_t<>&, chol_reader_t&, const imag_axes_ft::IAFT&, qp_params_t&, bool);

template void add_qpscf_vcorr(MBState&, double, solvers::mb_solver_t<>&, thc_reader_t&, const imag_axes_ft::IAFT&, qp_params_t&, const sArray_t<Array_view_5D_t>*);
template void add_qpscf_vcorr(MBState&, double, solvers::mb_solver_t<>&, chol_reader_t&, const imag_axes_ft::IAFT&, qp_params_t&, const sArray_t<Array_view_5D_t>*);

template double qp_eqn_linearized(double, analyt_cont::AC_t &, long, double, double, double);
template std::tuple<double,double> qp_eqn_bisection(double, analyt_cont::AC_t &, long, double, double, double, double);
template std::tuple<double,double,bool> qp_eqn_secant(double, analyt_cont::AC_t &, long, double, double, int, double, double);
template std::tuple<double,bool> qp_eqn_spectral(double, analyt_cont::AC_t &, long, double, double, double, double);

template double solve_iterative(utils::mpi_context_t<mpi3::communicator>&, iter_scf::iter_scf_t&, long, std::string,
                                sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_4D_t>&);
template void compute_G_from_mf(h5::group, imag_axes_ft::IAFT&, sArray_t<nda::array_view<ComplexType, 5>>&);

} // methods
