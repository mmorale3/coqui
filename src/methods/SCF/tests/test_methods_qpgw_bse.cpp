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

/**
 * Project 2 increment Q3 (notes/q3_bse_tier_spec.md): the BSE (ladder) polarization tier
 * in the qpGW loop. The loop screens with
 *
 *     P_latt(q, i.nu) = P^RPA(q, i.nu) + P^lad(q, i.nu),   W = v + v P_latt W,
 *
 * behind pol_vertex_inject = "ladder_n2". P^lad is the resummed static-rung
 * electron-hole ladder (rungs >= 1 = chi0-factor counts >= 2), which IS eq 6's
 * [.]_{n >= 2} as implemented -- no subtraction anywhere.
 *
 * Gates (spec section 5), measure-first-then-gate throughout:
 *   Q3-a  the three noop legs (bitwise);
 *   Q3-b  (i) the loop-vs-readout eps_M identity + the ladder's DIRECTION;
 *         (iii) the mode_a gap with and without injection (the physics readout);
 *   Q3-c  the lambda_max = rho(chi0 Xi) watchdog: parsed NUMBER finite and < 1, plus the
 *         kernel-scaling linearity pin on the estimator;
 *   Q3-d  the s = 1 pin -- ladder_sym_gate (one-rung rebuild == the pi_c_accumulate_w
 *         anchor) re-run on the qpGW-loop state.
 *
 * ==========================================================================================
 * HOW TO RUN (Catch2 v2 traps, spec section 8) -- MEASURED, do not "improve" the command
 * ==========================================================================================
 *
 *     KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 <build>/tests/bin/test_methods_qpgw_bse
 *
 * i.e. THE BARE BINARY, no filter: a "~[tag]" spec does NOT exclude hidden cases, it RUNS
 * them, and two positional test names are silently concatenated into one (unmatched) name.
 */

namespace bdft_tests {

  using namespace methods;

  namespace qpgw_bse_detail {

    constexpr double HA2EV = 27.211386245988;

    struct q3_row {
      std::string tag;
      double e_hf = 0.0;
      double e_homo = 0.0, e_lumo = 0.0;
      long final_iter = -1, niter = -1;
      // the update_w readouts of the LAST screening step (-1 = never populated)
      double eps_rpa = -1.0, eps_ladder = -1.0, eps_loop = -1.0;
      double lam_nu0 = -1.0, lam_max = -1.0, r_rt = -1.0, lad_ratio = -1.0;
      // the post-loop gates (only when gates = true)
      double sym_resid = -1.0, sym_ladder_frac = -1.0;
      double node_map_resid = -1.0, ph_sym_resid = -1.0;
      double lam_scaled = -1.0, lam_scale = 1.0;
      bool sym_active = false;
      // DA Phase 2 (notes/qsgwhat_discrepancy_spec.md): the pre/post-secondary-fold head
      // meter of the LAST build_w0 of the run (-1 = never measured).
      double hs_pre = -1.0, hs_post = -1.0, hs_atten = -1.0;
      double gap_eV() const { return (e_lumo - e_homo) * HA2EV; }
    };

    /**
     * One qpGW run with the ladder knobs of (pol_mode, inject, window), then -- when
     * gates is set -- ONE more screening step outside the loop so the Q3-c/Q3-d gates
     * can run on the state the loop actually used.
     *
     * Why the state has to be rebuilt: qp_scf_loop frees mb_state.sG_tskij at the end of
     * every iteration (qp_scf_common.cpp:1684) while sMO_skia / sE_ska -- the qp solution
     * itself -- persist. update_mu + update_G are the loop's OWN functions applied to the
     * loop's OWN final spectrum, so the G below is the G the last update_w saw, and the
     * extra update_w reproduces that screening step exactly (h5_iter = -1: no dump).
     */
    inline q3_row run_qpgw(auto &mpi_context, std::shared_ptr<mf::MF> &mf,
                           imag_axes_ft::IAFT &ft, std::string const &tag,
                           std::string const &map, std::string const &pol_mode,
                           std::string const &inject, nda::range window,
                           int thc_prefactor, double thc_tol, int niter, double conv_tol,
                           bool gates = false, const std::string &div = "ignore_g0",
                           // DA Phase 2 knobs (D-1 / D-4 / D-7); the defaults are the
                           // committed values, so every pre-existing caller is untouched.
                           bool tda = false, double head_scale = 1.0, bool qnu = false) {
      const std::string output = "qpgw_bse_" + tag;
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, div, output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", div);

      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*thc_prefactor, "", "incore", "",
                                                 output, thc_tol, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      qp_params_t qp_params("sc", "pade", 18, 0.0001, 1e-8, "qpscf");
      qp_params.qp_map = map;

      // the driver's [qpgw] wiring (MBPT_drivers.cpp): a pure knob carrier, attached to
      // scr_eri only. It must outlive qp_scf_loop -- same stack frame.
      solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd(), div);
      vtx.set_pol_vertex(pol_mode, "w0_prev", window, -1, 1e-8, -1.0, -1.0, -1.0, inject);
      vtx.set_ladder_da(tda, head_scale, qnu);
      if (vtx.pol_vertex_enabled()) scr_eri.set_vertex(&vtx);
      REQUIRE(not scr_eri.has_active_vertex());     // Sigma stays GW-form on this path

      iter_scf::iter_scf_t iter_sol(iter_scf::damp_t(0.7));
      MBState mb_state(mpi_context, ft, output);
      q3_row row;
      row.tag = tag;
      row.niter = niter;
      row.e_hf = qp_scf_loop(mb_state, eri, ft, qp_params,
                             solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                             niter, false, conv_tol);
      mpi_context->comm.barrier();

      std::tie(row.eps_rpa, row.eps_ladder) = scr_eri.pol_eps_readout();
      row.eps_loop = scr_eri.pol_eps_loop();
      row.lam_nu0 = scr_eri.pol_lambda_nu0();
      row.lam_max = scr_eri.pol_lambda_max();
      row.r_rt = scr_eri.pol_round_trip();
      row.lad_ratio = scr_eri.pol_ladder_ratio();
      // DA D-7: the head meter of the loop's LAST build_w0 (reset_w0 clears it at the top
      // of every build, so what stands here is the final iteration's).
      if (auto *pvh = scr_eri.pol_vertex_instance(); pvh != nullptr) {
        row.hs_pre = pvh->w0_head_share_pre();
        row.hs_post = pvh->w0_head_share_post();
        row.hs_atten = pvh->w0_head_attenuation();
      }

      nda::array<ComplexType, 3> E_ska;
      {
        h5::file file(output + ".mbpt.h5", 'r');
        auto scf_grp = h5::group(file).open_group("scf");
        h5::h5_read(scf_grp, "final_iter", row.final_iter);
        auto iter_grp = scf_grp.open_group("iter" + std::to_string(row.final_iter));
        nda::h5_read(iter_grp, "E_ska", E_ska);
      }
      mpi_context->comm.barrier();
      const long nkpts = E_ska.shape(1);
      const int homo = int(mf->nelec()/2 - 1);
      const int lumo = int(mf->nelec()/2);
      row.e_homo = -1e9;
      row.e_lumo = 1e9;
      for (long ik = 0; ik < nkpts; ++ik) {
        row.e_homo = std::max(row.e_homo, E_ska(0, ik, homo).real());
        row.e_lumo = std::min(row.e_lumo, E_ska(0, ik, lumo).real());
      }

      if (gates) {
        using math::shm::make_shared_array;
        double mu = update_mu(0.0, *mf, mb_state.sE_ska.value(), ft.beta());
        mb_state.sG_tskij.emplace(make_shared_array<Array_view_5D_t>(
            *mpi_context, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
        update_G(mb_state.sG_tskij.value(), mb_state.sMO_skia.value(),
                 mb_state.sE_ska.value(), mu, ft);
        scr_eri.update_w(mb_state, thc, -1);
        // the meters of THIS screening step (the loop's last one is reproduced here)
        std::tie(row.eps_rpa, row.eps_ladder) = scr_eri.pol_eps_readout();
        row.eps_loop = scr_eri.pol_eps_loop();
        row.lam_nu0 = scr_eri.pol_lambda_nu0();
        row.lam_max = scr_eri.pol_lambda_max();
        row.r_rt = scr_eri.pol_round_trip();
        row.lad_ratio = scr_eri.pol_ladder_ratio();

        auto *pv = scr_eri.pol_vertex_instance();
        REQUIRE(pv != nullptr);
        auto dw = pv->ladder_whalf_gate(mb_state, thc, 2.0);
        row.node_map_resid = dw.node_map_resid;
        row.ph_sym_resid = dw.ph_sym_resid;
        row.lam_scaled = dw.lam_nu0_scaled;
        row.lam_scale = dw.scale;
        auto ds = pv->ladder_sym_gate(mb_state, thc);
        row.sym_resid = ds.l1b_resid;
        row.sym_ladder_frac = ds.ladder_frac;
        row.sym_active = ds.sym_active;
      }

      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      app_log(1, "qpgw_bse [{}]: final iter {} of {}, e_hf = {}, gap = {:.6f} eV "
                 "(homo {:.9f}, lumo {:.9f} Ha)", tag, row.final_iter, niter, row.e_hf,
              row.gap_eV(), row.e_homo, row.e_lumo);
      return row;
    }

  } // qpgw_bse_detail

  /**
   * ==========================================================================================
   * GATE Q3-a -- THE NOOP LEGS (bitwise)
   * ==========================================================================================
   * Leg (i)   knob ABSENT: the qpGW loop must still reproduce the stored QM3 reference. The
   *           BITWISE pin of that number is the commit-point suite test_methods_qp_map_ab
   *           (which runs the same fixture/settings); here it is re-checked against the
   *           stored ac_pade gap 11.8024 eV of qp_map_modea_lih222 at its own quoted
   *           precision, so a shared-seam edit that moves the loop cannot pass unnoticed.
   * Leg (ii)  pol_vertex_inject = "ladder_n2" with an EMPTY C window: the injection is a
   *           STRUCTURAL no-op (pol_vertex_active() false), so the run must be BITWISE
   *           identical to leg (i) -- the scgwt_noop pattern.
   * Leg (ii-b) the ladder READOUT on a NON-empty window with inject = "none": report-only,
   *           so still bitwise identical. This is what makes leg (iii)/Q3-b meaningful --
   *           it proves the injection, not the ladder machinery, is what moves the loop.
   * Leg (iii) the [gw] path: Project 1's test_vertex_scgwt_noop suite (shared seam), run as
   *           a commit-point suite -- not duplicated here.
   */
  TEST_CASE("qpgw_bse_noop_lih222", "[methods][qpgw][bse]") {
#ifndef ENABLE_DLR
    SUCCEED("qpgw_bse_noop_lih222 skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace qpgw_bse_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));

    auto r0 = run_qpgw(mpi_context, mf, ft, "noop_a", "ac_pade", "none", "none",
                       nda::range(0, 0), 12, 1e-10, 20, 1e-6);
    // (i) the stored QM3 reference (qp_map_modea_lih222: ac_pade 11.8024 eV)
    app_log(1, "@@Q3A leg (i): gap = {:.6f} eV vs the stored ac_pade reference 11.8024 eV "
               "(d = {:+.2e})", r0.gap_eV(), r0.gap_eV() - 11.8024);
    REQUIRE(std::abs(r0.gap_eV() - 11.8024) < 1e-4);
    REQUIRE(r0.eps_rpa == -1.0);        // no readout ran
    REQUIRE(r0.lam_nu0 == -1.0);        // ... and no injection

    // (ii) injection requested, EMPTY window => structurally inert
    auto r1 = run_qpgw(mpi_context, mf, ft, "noop_b", "ac_pade", "none", "ladder_n2",
                       nda::range(0, 0), 12, 1e-10, 20, 1e-6);
    app_log(1, "@@Q3A leg (ii): empty-window injection e_hf = {}, gap = {:.9f} eV "
               "(d_e_hf = {:.3e}, d_gap = {:.3e})", r1.e_hf, r1.gap_eV(),
            std::abs(r1.e_hf - r0.e_hf), std::abs(r1.gap_eV() - r0.gap_eV()));
    REQUIRE(r1.e_hf == r0.e_hf);
    REQUIRE(r1.e_homo == r0.e_homo);
    REQUIRE(r1.e_lumo == r0.e_lumo);
    REQUIRE(r1.final_iter == r0.final_iter);
    REQUIRE(r1.lam_nu0 == -1.0);        // inert: nothing was injected

    // (ii-b) the READOUT on a live window, injection off => still bitwise
    auto r2 = run_qpgw(mpi_context, mf, ft, "noop_c", "ac_pade", "ladder", "none",
                       nda::range(1, 3), 12, 1e-10, 20, 1e-6);
    app_log(1, "@@Q3A leg (ii-b): readout-only e_hf = {}, gap = {:.9f} eV; eps_M(q_min) "
               "RPA = {:.6f}, +ladder = {:.6f}", r2.e_hf, r2.gap_eV(), r2.eps_rpa,
            r2.eps_ladder);
    REQUIRE(r2.e_hf == r0.e_hf);
    REQUIRE(r2.e_homo == r0.e_homo);
    REQUIRE(r2.e_lumo == r0.e_lumo);
    REQUIRE(r2.eps_rpa > 0.0);          // the readout DID run (non-vacuous)
    REQUIRE(r2.lam_nu0 == -1.0);        // ... and injected nothing
#endif
  }

  /**
   * ==========================================================================================
   * GATES Q3-b(i), Q3-c, Q3-d -- the injection on qe_lih222, C = [1, 3)
   * ==========================================================================================
   * Q3-b(i) THE IDENTITY. Two routes to the same number: the loop's own q-resolved
   *   eps^-1 head at q_min (tau Dyson of the INJECTED Pi, then tau -> i.nu = 0) against the
   *   readout's single-frequency i.nu = 0 Dyson of P^RPA(nu=0) + P^lad(nu=0). Same G, same
   *   kernel, same q. The expected deviation class is the nu -> tau -> nu round trip r_rt of
   *   P^lad through the IAFT PH-sym pair, which the injection logs every update_w.
   *   MEASURED (2026-08-14, qe_lih222, C = [1,3), ac_pade, 20 iterations): see the numbers
   *   in the REQUIREs below; the gate is 10x the measured class, as the spec prescribes.
   *   Direction: the ladder must INCREASE eps_M (up at every Project-1 fixture/iteration).
   * Q3-c  the watchdog NUMBER (not the log line): finite and < 1, and the estimator reads
   *   the right matrix -- Xh Kt is linear in the rung, so scaling W-bar_0 by 2 must scale
   *   rho by exactly 2.
   * Q3-d  the s = 1 pin: the one-rung rebuild against the pi_c_accumulate_w anchor with the
   *   Q3 configuration live, on the qpGW G. Machine class (3.2e-16 / 4.7e-16 measured on
   *   scGW states); the qpGW G is a different G, the identity is the same.
   * The node-map / PH-symmetry residuals of the multi-nu evaluator (increment I1) are
   * checked here too: they are what licenses the PH-sym transform of P^lad at all.
   */
  TEST_CASE("qpgw_bse_inject_lih222", "[methods][qpgw][bse]") {
#ifndef ENABLE_DLR
    SUCCEED("qpgw_bse_inject_lih222 skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace qpgw_bse_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));

    auto r = run_qpgw(mpi_context, mf, ft, "inj_lih", "ac_pade", "ladder", "ladder_n2",
                      nda::range(1, 3), 12, 1e-10, 20, 1e-6, true);

    // ---- the multi-nu evaluator (I1) --------------------------------------------------
    app_log(1, "@@Q3I1 node-map resid = {:.3e}, PH-symmetry resid = {:.3e}",
            r.node_map_resid, r.ph_sym_resid);
    // the PH-sym half grid must address the SAME nodes as a full-mesh evaluation: same
    // kernel, same scheduling => bitwise 0 (a wrong half<->full map is O(1)). MEASURED
    // exactly 0.0 (qe_lih222, nw_b = 53 -> 27 half nodes; the map is full = nw_b/2 + j).
    REQUIRE(r.node_map_resid == 0.0);
    // ... and P^lad must be PH-symmetric in nu, or the PH-sym transform it is pushed
    // through is not the right one. This is NOT machine precision: MEASURED 4.28e-09
    // relative (2026-08-14) -- the DLR/solve class, not the FP class -- and the gate is
    // 10x that. It is the licence for the transform, and the Q3-b(i) identity below is
    // the falsifier that keeps it honest.
    REQUIRE(r.ph_sym_resid < 5e-8);

    // ---- Q3-b(i): the loop-vs-readout eps_M identity -----------------------------------
    const double dev = std::abs(r.eps_loop - r.eps_ladder);
    app_log(1, "@@Q3B(i) eps_M(q_min, inu = 0): loop route = {:.9f}, readout route "
               "(+ladder) = {:.9f}, deviation = {:.3e}; RPA baseline = {:.9f} "
               "(Delta_ladder = {:+.3e}); transform class r_rt = {:.3e}",
            r.eps_loop, r.eps_ladder, dev, r.eps_rpa, r.eps_ladder - r.eps_rpa, r.r_rt);
    REQUIRE(r.eps_loop > 0.0);
    REQUIRE(r.eps_ladder > 0.0);
    REQUIRE(std::isfinite(r.eps_loop));
    // MEASURED (2026-08-14, qe_lih222, C = [1,3), ac_pade, 20 iterations): loop route
    // 1.439266540 vs readout route 1.439266535, deviation 4.78e-09 -- exactly the
    // transform class the spec predicts (r_rt = 8.47e-08 measured on the same step).
    // The gate is 10x the measured deviation (spec section 5: measure, then gate).
    REQUIRE(dev < 4.8e-8);
    // DIRECTION: the ladder increases eps_M (Project-1 reference behaviour everywhere).
    // MEASURED here: 1.426163 -> 1.439267 at q_min, +0.0131 (+0.92 %).
    REQUIRE(r.eps_ladder > r.eps_rpa);
    // the round trip is a meter, not decoration: it must have been measured and stay in
    // its class (MEASURED 8.47e-08; gate 10x).
    REQUIRE(r.r_rt >= 0.0);
    REQUIRE(r.r_rt < 8.5e-7);

    // ---- Q3-c: the lambda_max watchdog (the NUMBER, not the log line) ------------------
    app_log(1, "@@Q3C lambda_max(inu = 0) = {:.9f}, max over nu = {:.9f}; rung x {:.3g} => "
               "{:.9f} (ratio {:.9f}); ||P^lad||/||P^RPA|| = {:.3e}",
            r.lam_nu0, r.lam_max, r.lam_scale, r.lam_scaled,
            r.lam_scaled / std::max(r.lam_nu0, 1e-300), r.lad_ratio);
    // MEASURED: lambda_max = 0.114764 at every nu node (the nu = 0 node carries the
    // maximum, as chi0 does), i.e. a wide margin to the eq-6 instability at 1.
    REQUIRE(std::isfinite(r.lam_nu0));
    REQUIRE(r.lam_nu0 > 0.0);
    REQUIRE(std::isfinite(r.lam_max));
    REQUIRE(r.lam_max < 1.0);
    REQUIRE(r.lam_max >= r.lam_nu0);
    // the estimator reads Xh Kt, which is LINEAR in the rung: MEASURED ratio 2.000000 at
    // scale 2 (a power iteration on any other matrix would not scale exactly)
    REQUIRE(std::abs(r.lam_scaled / r.lam_nu0 - r.lam_scale) < 1e-6 * r.lam_scale);
    // and the correction is a correction (MEASURED ||P^lad||/||P^RPA|| = 3.14e-04)
    REQUIRE(r.lad_ratio > 0.0);
    REQUIRE(r.lad_ratio < 1.0);

    // ---- Q3-d: the s = 1 pin on the qpGW-loop state ------------------------------------
    app_log(1, "@@Q3D ladder_sym_gate on the qpGW state (sym {}): one-rung rebuild vs the "
               "Pi^C anchor resid = {:.3e}; >= 2-rung content = {:.3e}",
            r.sym_active ? "ACTIVE" : "inactive", r.sym_resid, r.sym_ladder_frac);
    // MEASURED 4.07e-16 on the qpGW G (3.2e-16 / 4.7e-16 on the Project-1 scGW states):
    // the same machine-precision identity, so s = 1 in this configuration too.
    REQUIRE(r.sym_resid >= 0.0);
    REQUIRE(r.sym_resid < 1e-12);
    // the resummation is not trivially the anchor (MEASURED >= 2-rung content 8.64e-02)
    REQUIRE(r.sym_ladder_frac > 0.0);
#endif
  }

  /**
   * ==========================================================================================
   * GATE Q3-b(iii) -- THE PHYSICS READOUT: the mode-A gap with and without the BSE tier
   * ==========================================================================================
   * The first qpGW+BSE number of the project. Both legs must converge, stay Hermitian (the
   * mode-A driver's own anchor check aborts otherwise) and give finite, sane gaps; the SHIFT
   * is RECORDED, never gated -- there is no reference to gate it against yet.
   * pyscf_si222 with C = [2, 6) straddling the gap.
   */
  TEST_CASE("qpgw_bse_modea_si222", "[methods][qpgw][bse]") {
#ifndef ENABLE_DLR
    SUCCEED("qpgw_bse_modea_si222 skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace qpgw_bse_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_si222"));
    const long b1 = std::min<long>(6, mf->nbnd());
    nda::range window(2, b1);
    REQUIRE(window.size() > 0);

    auto r0 = run_qpgw(mpi_context, mf, ft, "si_modea_off", "mode_a", "none", "none",
                       nda::range(0, 0), 10, 1e-8, 12, 1e-5);
    auto r1 = run_qpgw(mpi_context, mf, ft, "si_modea_on", "mode_a", "ladder", "ladder_n2",
                       window, 10, 1e-8, 12, 1e-5, true);

    app_log(1, "@@Q3B(iii) pyscf_si222 / mode_a, C = [{}, {}): gap OFF = {:.6f} eV, "
               "ON = {:.6f} eV (shift {:+.6f} eV); eps_M(q_min) RPA = {:.6f} -> +ladder "
               "= {:.6f}; ||P^lad||/||P^RPA|| = {:.3e}, lambda_max = {:.6f}, "
               "r_rt = {:.3e}", window.first(), window.last(), r0.gap_eV(), r1.gap_eV(),
            r1.gap_eV() - r0.gap_eV(), r1.eps_rpa, r1.eps_ladder, r1.lad_ratio,
            r1.lam_max, r1.r_rt);
    for (auto const &r : {r0, r1}) {
      REQUIRE(std::isfinite(r.gap_eV()));
      REQUIRE(r.gap_eV() > 0.0);
      REQUIRE(r.gap_eV() < 30.0);
      REQUIRE(r.final_iter < r.niter);        // converged inside the cap
    }
    // Q3-c on the second fixture: the watchdog NUMBER
    REQUIRE(std::isfinite(r1.lam_max));
    REQUIRE(r1.lam_max > 0.0);
    REQUIRE(r1.lam_max < 1.0);
    REQUIRE(r1.r_rt < 8.5e-7);
    // Q3-b direction and Q3-d on the si222 state
    REQUIRE(r1.eps_ladder > r1.eps_rpa);
    REQUIRE(r1.sym_resid >= 0.0);
    REQUIRE(r1.sym_resid < 1e-12);
    REQUIRE(r1.node_map_resid == 0.0);
#endif
  }

  /**
   * ==========================================================================================
   * DA PHASE 2 -- the D-1 / D-4 / D-7 knob gates (notes/qsgwhat_discrepancy_spec.md)
   * ==========================================================================================
   * These three knobs exist to MEASURE factors in the Questaal-paper discrepancy analysis,
   * so the discipline is the standing one: knob-absent must be BITWISE the committed path,
   * and knob-present must be demonstrably non-vacuous (finding F-DA-1 was exactly a knob
   * that parsed, logged, and did nothing).
   *
   * Leg (a) baseline: inject on, gygi div (so a head EXISTS to scale), at the reduced
   *         kernel head the fixture needs (see the FIXTURE CHOICE note below).
   * Leg (b) D-7 ladder_qnu_meter = true: the meters are PURE OBSERVERS, so every physics
   *         number must be BITWISE leg (a). This is the strongest of the three gates --
   *         the (q,nu) accumulators sit inside the injection's innermost write loop.
   * Leg (c) D-1 ladder_tda = true: the kernel's resonant<->anti-resonant coupling block is
   *         zeroed. Must MOVE the answer (non-vacuous) and stay sane.
   * Leg (d) D-4 ladder_head_scale -> 0.0: the analytic q->0 head is removed from the ladder
   *         kernel W-bar_0. Must MOVE the answer -- this is the leg F-DA-1 proved did not
   *         exist before (vertex_div_treatment = "ignore_g0" was numerically inert at
   *         9.3e-08 eV on the LiF production runs).
   * All shifts are RECORDED, never gated on a value: the fixture is a toy and the physics
   * numbers belong to the production legs.
   */
  TEST_CASE("qpgw_bse_da_knobs_si222", "[methods][qpgw][bse][da]") {
#ifndef ENABLE_DLR
    SUCCEED("qpgw_bse_da_knobs_si222 skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace qpgw_bse_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    // FIXTURE CHOICE, MEASURED not assumed. The head knob needs a head, so div = "gygi"
    // (under "ignore_g0" no head is inserted at all and D-4 is CORRECTLY inert). But the
    // gygi head at a 2^3 mesh is enormous relative to the body -- N_k*madelung is an O(1/N_k)
    // object -- and it drives the ladder past the particle-hole instability on BOTH toy
    // fixtures: MEASURED rho(Xh Kt) = 1.688 (qe_lih222, C = [1,3)), 1.474 (pyscf_si222,
    // C = [2,6)) and 1.007 (pyscf_si222, C = [3,5)), i.e. the eq-6 resolvent abort fires
    // before any leg finishes. The DA legs therefore run at a REDUCED kernel head
    // (ladder_head_scale = HS0 = 0.25) and the D-4 non-vacuity leg drops it to 0. That
    // exercises the knob at TWO non-default values, which is strictly stronger evidence
    // that it reaches the kernel than a single on/off pair would be. The knob-ABSENT
    // bitwise requirement is carried by the three pre-existing test cases above, which run
    // with every DA knob at its committed default and reproduce their stored references.
    // The window is narrowed to C = [3,5) for the same reason (rho stays in regime at the
    // reduced head): it is the minimal window that still straddles the gap (Si nocc = 4:
    // band 3 occupied, band 4 empty), so the four pairs per k are (3,3) and (4,4) with
    // character 0 and (3,4)/(4,3) with characters +1/-1 -- both of the characters the D-1
    // mask distinguishes are present, which is what the gate needs.
    constexpr double HS0 = 0.25;
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_si222"));
    const long b1 = std::min<long>(5, mf->nbnd());
    nda::range window(3, b1);
    REQUIRE(window.size() == 2);

    auto ra = run_qpgw(mpi_context, mf, ft, "da_base", "ac_pade", "ladder", "ladder_n2",
                       window, 10, 1e-8, 12, 1e-5, false, "gygi", false, HS0, false);
    auto rb = run_qpgw(mpi_context, mf, ft, "da_meter", "ac_pade", "ladder", "ladder_n2",
                       window, 10, 1e-8, 12, 1e-5, false, "gygi", false, HS0, true);
    // gates = true on the TDA leg: ladder_whalf_gate re-earns, FOR THE TRUNCATED KERNEL,
    // the PH-symmetry licence that the injection's nu -> tau transform depends on.
    auto rc = run_qpgw(mpi_context, mf, ft, "da_tda", "ac_pade", "ladder", "ladder_n2",
                       window, 10, 1e-8, 12, 1e-5, true, "gygi", true, HS0, true);
    auto rd = run_qpgw(mpi_context, mf, ft, "da_head0", "ac_pade", "ladder", "ladder_n2",
                       window, 10, 1e-8, 12, 1e-5, false, "gygi", false, 0.0, true);

    // ---- leg (b): the D-7 meters are pure observers -> BITWISE --------------------------
    app_log(1, "@@DA D-7 pure-observer leg: e_hf d = {:.3e}, gap d = {:.3e} eV, "
               "eps_ladder d = {:.3e}, lambda_nu0 d = {:.3e}, lad_ratio d = {:.3e}",
            std::abs(rb.e_hf - ra.e_hf), std::abs(rb.gap_eV() - ra.gap_eV()),
            std::abs(rb.eps_ladder - ra.eps_ladder),
            std::abs(rb.lam_nu0 - ra.lam_nu0),
            std::abs(rb.lad_ratio - ra.lad_ratio));
    REQUIRE(rb.e_hf == ra.e_hf);
    REQUIRE(rb.e_homo == ra.e_homo);
    REQUIRE(rb.e_lumo == ra.e_lumo);
    REQUIRE(rb.final_iter == ra.final_iter);
    REQUIRE(rb.eps_rpa == ra.eps_rpa);
    REQUIRE(rb.eps_ladder == ra.eps_ladder);
    REQUIRE(rb.lam_nu0 == ra.lam_nu0);
    REQUIRE(rb.lam_max == ra.lam_max);
    REQUIRE(rb.lad_ratio == ra.lad_ratio);
    // ... and the meter it enables actually produced numbers (H1b, the fold attenuation)
    app_log(1, "@@DA D-7 head meter (H1b): share PRE-fold = {:.6e}, POST-fold = {:.6e}, "
               "attenuation = {:.6f}", rb.hs_pre, rb.hs_post, rb.hs_atten);
    REQUIRE(rb.hs_pre > 0.0);
    REQUIRE(rb.hs_post > 0.0);
    REQUIRE(rb.hs_atten > 0.0);
    REQUIRE(std::isfinite(rb.hs_atten));
    // ... and the baseline leg, which never asked for it, has none
    REQUIRE(ra.hs_pre == -1.0);

    // ---- leg (c): the D-1 TDA knob is non-vacuous ---------------------------------------
    app_log(1, "@@DA D-1 TDA leg: gap OFF = {:.6f} eV, TDA ON = {:.6f} eV "
               "(shift {:+.6f} eV); eps_M(q_min) {:.6f} -> {:.6f}; lambda_max {:.6f} -> "
               "{:.6f}; ||P^lad||/||P^RPA|| {:.3e} -> {:.3e}",
            ra.gap_eV(), rc.gap_eV(), rc.gap_eV() - ra.gap_eV(), ra.eps_ladder,
            rc.eps_ladder, ra.lam_max, rc.lam_max, ra.lad_ratio, rc.lad_ratio);
    REQUIRE(std::isfinite(rc.gap_eV()));
    REQUIRE(rc.gap_eV() > 0.0);
    REQUIRE(std::isfinite(rc.lam_max));
    REQUIRE(rc.lam_max < 1.0);
    REQUIRE(rc.lad_ratio > 0.0);
    REQUIRE(rc.gap_eV() != ra.gap_eV());          // the knob reached the kernel
    // The TDA mask is nu-INDEPENDENT and exchanges the two characters under nu -> -nu, so
    // the truncated P^lad should stay PH-symmetric -- otherwise the injection's PH-sym
    // transform would be illegitimate on this leg. MEASURE FIRST, THEN GATE: the residual
    // is 3.02e-06 relative under TDA against 1.08e-08 for the untruncated kernel, i.e. the
    // truncation DEGRADES the symmetry by ~300x without breaking it. That is expected and
    // recorded rather than explained away -- the res <-> ares mirror also flips q, so the
    // two blocks are exact mirrors only up to the q -> -q relabeling of the IBZ transfer
    // axis, and the character assignment itself is only as sharp as the C-window density
    // matrix is occupation-diagonal (this fixture measures crispness 0.45 / off-diagonal
    // 0.26 and the code WARNS about it). 3e-06 is four orders below the ladder effects
    // being measured, so the transform stays licensed; the gate is 10x the measurement.
    app_log(1, "@@DA D-1 PH-symmetry licence UNDER TDA: node-map resid = {:.3e}, "
               "PH-symmetry resid = {:.3e}; one-rung-vs-anchor (untruncated) = {:.3e}",
            rc.node_map_resid, rc.ph_sym_resid, rc.sym_resid);
    REQUIRE(rc.node_map_resid == 0.0);
    REQUIRE(rc.ph_sym_resid < 3e-5);

    // ---- leg (d): the D-4 kernel-head knob is non-vacuous (the F-DA-1 repair) -----------
    app_log(1, "@@DA D-4 head-off leg (kernel head scale {:.2f} -> 0): gap head-ON = "
               "{:.6f} eV, head-OFF = {:.6f} eV (shift {:+.6f} eV); eps_M(q_min) {:.6f} -> "
               "{:.6f}; lambda_max {:.6f} -> {:.6f}; head share PRE-fold {:.6e} -> {:.6e}",
            HS0, ra.gap_eV(), rd.gap_eV(), rd.gap_eV() - ra.gap_eV(),
            ra.eps_ladder, rd.eps_ladder, ra.lam_max, rd.lam_max, rb.hs_pre, rd.hs_pre);
    REQUIRE(std::isfinite(rd.gap_eV()));
    REQUIRE(rd.gap_eV() > 0.0);
    REQUIRE(std::isfinite(rd.lam_max));
    REQUIRE(rd.lam_max < 1.0);
    // F-DA-1's falsifier: the old zero-code route moved the LiF gap by 9.3e-08 eV. A knob
    // that actually reaches the ladder kernel's head must do far better than that.
    REQUIRE(std::abs(rd.gap_eV() - ra.gap_eV()) > 1e-6);
    // head_scale = 0 removes the head from W-bar_0, so the meter must report a ZERO head
    REQUIRE(rd.hs_pre == 0.0);
#endif
  }

} // bdft_tests
