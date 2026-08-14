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
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include <cstdlib>

#include "methods/SCF/qp_modea.hpp"

/**
 * Project 2 increment Q2 (notes/qpgw_edmft_implementation_plan.md): the A/B
 * surrogate-spread deliverable. The full qp_scf_loop runs with each of the
 * quasiparticle maps (ac_pade / mats_lin / mats_gmatch, and from increment QM3
 * also mode_a) on the same
 * mean field and the same THC factorization, so every difference in the band
 * edges is the surrogate spread of the static map itself (spec section 4
 * "residual ambiguity" -- REPORTED, not converged away). Assertions are loose
 * tripwires against gross breakage; the table in the log is the deliverable.
 *
 * ==========================================================================================
 * HOW TO RUN THIS SUITE (gate QM3-a) -- MEASURED, do not "improve" the command
 * ==========================================================================================
 *
 *     KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 <build>/tests/test_methods_qp_map_ab
 *
 * i.e. THE BARE BINARY, no filter. Catch2 v2 hides a test case tagged "[.foo]" only in the
 * DEFAULT (no test-spec) run: as soon as ANY test spec is supplied on the command line,
 * hidden cases are matched by it like any other, so a NEGATIVE filter such as `~[modea]` or
 * `~[.modeb_matrix]` does NOT exclude them -- it selects everything that does not carry that
 * tag, hidden cases included, and the hidden measurement/hold cases then run (measured here,
 * 2026-08-12: a "~[...]" suite command silently ran the hidden matrix case for >1 h).
 *
 * Consequences for this file:
 *   - the live gates carry ordinary tags and run under the bare binary;
 *   - the measurement / on-hold cases carry a leading-dot tag and are run ONE AT A TIME by
 *     naming them explicitly, e.g. `test_methods_qp_map_ab "qp_map_modeb_matrix" -c lih222_qpscf`;
 *   - never use a "~[tag]" spec to skip them.
 */

namespace bdft_tests {

  using namespace methods;

  namespace qp_map_ab_detail {

    constexpr double HA2EV = 27.211386245988;

    struct ab_row {
      std::string map;
      double e_homo, e_lumo;              // band edges over the k mesh
      nda::array<double, 1> Ek0;          // bands [homo-1, lumo+2) at k = 0
      qp_modea::last_run_t lr;            // mode_a only: anchor / delta_i / fit diagnostics
      long final_iter = -1, niter = -1;   // outer-loop trajectory (final_iter < niter = converged)
      double gap_eV() const { return (e_lumo - e_homo) * HA2EV; }
    };

    // One qp-scf run with the given map; returns band edges from the
    // checkpoint's final iteration.
    inline ab_row run_map(auto &mpi_context, std::shared_ptr<mf::MF> &mf,
                          imag_axes_ft::IAFT &ft,
                          const std::string &map, const std::string &mode,
                          int thc_prefactor, double thc_tol,
                          int niter, double conv_tol,
                          const std::string &wfit = "tau",
                          const std::string &tag = "", double eta = 0.0,
                          double wrtol = -1.0, double wrank = 1e-10, long wsketch = 0,
                          double eta_far = 0.0, bool gate = true, double wunion = -1.0,
                          const std::string &div = "ignore_g0") {
      const std::string output = "qp_map_ab_" + map + "_" + mode + tag;
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, div, output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", div);

      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*thc_prefactor, "", "incore", "", output,
                                                 thc_tol, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      qp_params_t qp_params("sc", "pade", 18, 0.0001, 1e-8, mode);
      qp_params.qp_map = map;
      qp_params.qp_modea_wfit = wfit;
      qp_params.qp_modea_eta = eta;
      qp_params.qp_modea_eta_far = eta_far;      // spec rev 4; 0 = the rev-3.1 mu fallback
      qp_params.qp_modea_wrtol = wrtol;
      qp_params.qp_modea_wrank = wrank;
      qp_params.qp_modea_wsketch = wsketch;
      qp_params.qp_modea_wunion = wunion;
      qp_modea::last_run() = qp_modea::last_run_t{};
      iter_scf::iter_scf_t iter_sol(iter_scf::damp_t(0.7));
      MBState mb_state(mpi_context, ft, output);
      [[maybe_unused]] double e_hf = qp_scf_loop(mb_state, eri, ft, qp_params,
                                                 solvers::mb_solver_t(&hf, &gw, &scr_eri),
                                                 &iter_sol, niter, false, conv_tol);
      mpi_context->comm.barrier();

      nda::array<ComplexType, 3> E_ska;
      long final_it = -1;
      {
        h5::file file(output + ".mbpt.h5", 'r');
        auto scf_grp = h5::group(file).open_group("scf");
        h5::h5_read(scf_grp, "final_iter", final_it);
        auto iter_grp = scf_grp.open_group("iter" + std::to_string(final_it));
        nda::h5_read(iter_grp, "E_ska", E_ska);
      }
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();

      const long nkpts = E_ska.shape(1);
      const int homo = int(mf->nelec()/2 - 1);
      const int lumo = int(mf->nelec()/2);
      ab_row row;
      row.map = map + "(" + mode + ")";
      row.e_homo = -1e9;
      row.e_lumo = 1e9;
      for (long ik = 0; ik < nkpts; ++ik) {
        row.e_homo = std::max(row.e_homo, E_ska(0, ik, homo).real());
        row.e_lumo = std::min(row.e_lumo, E_ska(0, ik, lumo).real());
      }
      row.Ek0.resize(4);
      for (int b = 0; b < 4; ++b) row.Ek0(b) = E_ska(0, 0, homo - 1 + b).real();
      row.lr = qp_modea::last_run();
      row.final_iter = final_it;
      row.niter = niter;
      app_log(1, "qp_map_ab [{}]: final iter {} of {}, gap = {:.6f} eV (homo {:.6f}, "
                 "lumo {:.6f} Ha)", row.map, final_it, niter,
              (row.e_lumo - row.e_homo) * HA2EV, row.e_homo, row.e_lumo);
      if (map == "mode_b" or map == "mode_a") {
        // Per-k band-edge geometry against the strip of qp_modea.hpp -- outside it mode_b
        // falls back to z = mu on the diagonal and mode_a evaluates BOTH indices at mu
        // (rev 3.1) -- evaluated on the FINAL stored spectrum. VBM/CBM here are the same
        // global edges the driver uses.
        const double E_PH = row.lr.gap_edge;
        const double lo = row.e_homo - 0.95 * E_PH, hi = row.e_lumo + 0.95 * E_PH;
        app_log(1, "@@STRIPGEO [{}{}] VBM = {:+.6f}, CBM = {:+.6f}, E_PH = {:.6f} a.u. "
                   "({:.4g} eV) => strip = ({:+.6f}, {:+.6f}) a.u.",
                map, tag, row.e_homo, row.e_lumo, E_PH, E_PH * HA2EV, lo, hi);
        for (long ik = 0; ik < nkpts; ++ik) {
          const double eh = E_ska(0, ik, homo).real(), el = E_ska(0, ik, lumo).real();
          app_log(1, "@@STRIPK [{}{}] k = {:2}: HOMO = {:+.6f} (dist to lower bound {:+.6f}, "
                     "{}), LUMO = {:+.6f} (dist to upper bound {:+.6f}, {})",
                  map, tag, ik, eh, eh - lo, (eh > lo ? "in strip" : "OUTSIDE"),
                  el, hi - el, (el < hi ? "in strip" : "OUTSIDE"));
        }
        // whole-spectrum census on the final spectrum
        long n_out = 0;
        for (long ik = 0; ik < nkpts; ++ik)
          for (long a = 0; a < E_ska.shape(2); ++a) {
            const double e = E_ska(0, ik, a).real();
            if (not (e > lo and e < hi)) ++n_out;
          }
        app_log(1, "@@STRIPCENSUS [{}{}] {} of {} states of the FINAL spectrum lie outside "
                   "the strip", map, tag, n_out, nkpts * E_ska.shape(2));
      }
      if (map == "mode_a" or map == "mode_b") {
        auto const &L = row.lr;
        // gate QM3-b(ii): the ANCHOR, with its expectation class. The hard < 1e-2 check is
        // enforced inside the driver at EVERY outer iteration (qp_scf_common.cpp); this line
        // reports the last iteration's numbers and re-asserts them here.
        app_log(1, "qp_map_ab [{}{}] QM3-b' diagnostics: TAU ANCHOR = {:.4e} (gate {:.1g} x "
                   "the W-fit reconstruction class {:.4e}, ratio {:.3g});  i w diagnostic "
                   "(NOT a gate, reference-aliasing dominated) = {:.4e};  anti-Hermitian "
                   "residual = {:.3e};  min_den = {:.4e} a.u.;  diagonal fallbacks = {}",
                map, tag, L.tau_dev, qp_modea::modea_tau_anchor_mult, L.anchor_expect,
                (L.anchor_expect > 0.0 ? L.tau_dev / L.anchor_expect : 0.0), L.anchor,
                L.anti_herm, L.min_den, L.n_fallback);
        app_log(1, "qp_map_ab [{}{}] W^c fit: route = {}, gap_edge = {:.6g} a.u. "
                   "({:.4g} eV), {} of {} auxiliary nodes retained, nJ = {}, npk = {}; "
                   "context build {:.2f} s, {:.1f} MB extra per rank",
                map, tag, L.wfit, L.gap_edge, L.gap_edge * HA2EV, L.n_support, L.np_total,
                L.nJ, L.npk, L.wall_s, L.mem_mb);
        REQUIRE(L.tau_dev >= 0.0);
        // THE GATE (spec rev 2) -- tau domain, not tunable. `gate = false` is used ONLY by the
        // hidden measurement harnesses, whose point is to REPORT a cell that misbehaves (a
        // REQUIRE there aborts the case and loses the rest of the table); every live case
        // leaves it true. The driver's own tau-anchor check still aborts the process, so this
        // cannot hide a contraction error.
        if (gate) REQUIRE(L.tau_dev < qp_modea::modea_tau_anchor_mult * L.anchor_expect);
      }
      if (map == "mode_a") {
        auto const &L = row.lr;
        // THE STRIP-CLAMP CENSUS of the LAST outer iteration (rev 3 addendum item 2) and the
        // inner-consistency convergence that the clamp is there to restore.
        app_log(1, "@@CLAMPCENSUS [{}{}] out of strip {} of {} evaluation energies ({} in the "
                   "gap window) over {} (s,k) blocks; per-k HOMO out of strip in {} blocks, "
                   "LUMO in {}; inner consistency: {} sweeps, max|d eps| = {:.4e} a.u. ({}); "
                   "rev4: etafar = {:.4e} a.u., eta-evaluated {} (mu-clamped {}), "
                   "max|Im Sigma| off strip = {:.4e} a.u., in-strip anti-herm = {:.3e}, "
                   "pole spacing = {:.4e} a.u.",
                map, tag, L.n_clamp, L.n_eval, L.n_clamp_win, L.n_blocks, L.n_homo_clamp,
                L.n_lumo_clamp, L.iters, L.dmax,
                L.converged_inner ? "converged" : "HIT THE CAP",
                L.eta_far, L.n_eta, L.n_clamp - L.n_eta, L.im_off, L.anti_in, L.spacing);
        // (i) of gate QM3-b: THE LOOP MUST CONVERGE. The rev-1 failure mode was
        // max|d eps| ~ 1e4-1e5 a.u. and the strip-BOUNDARY reading gave 2.4e+04; with the
        // clamp to mu every evaluation is bounded.
        //
        // The gate is the EXIT RESIDUAL of the last outer iteration, not the per-block
        // "met consist_tol within nconsist sweeps" flag. MEASURED (lih222, rev 3.1): 5 of 8
        // blocks reach 6.6e-09 in 2 sweeps and 3 blocks are still contracting at 9.3e-08
        // when the nconsist = 5 cap ends them -- a budget statement about the knob pair
        // (nconsist 5, consist_tol 1e-8), not a physics one, and 9.3e-08 a.u. is 2.5 neV.
        // Requiring the flag would gate on the knob; requiring 1e-6 a.u. (0.027 meV) gates
        // on the physics and still fails the boundary pathology by ten orders. The flag is
        // logged above and the driver warns on it every outer iteration.
        if (gate) {
          REQUIRE(std::isfinite(L.dmax));
          REQUIRE(L.dmax < 1e-6);
          // the census must be self-consistent (a nonzero count with no evaluations would mean
          // the census never ran, i.e. a vacuous check)
          REQUIRE(L.n_eval > 0);
          REQUIRE(L.n_blocks > 0);
        }
      }
      return row;
    }

    // THE deliverable: the per-map band-edge table + the per-band spread at
    // k = 0 (the Q1 finding predicts the spread GROWS with |E - mu|).
    inline void report_and_check(std::vector<ab_row> const &rows) {
      app_log(1, "\n== qp_map A/B surrogate-spread table ==");
      app_log(1, "{:<24} {:>12} {:>12} {:>12}", "map", "E_homo (Ha)", "E_lumo (Ha)", "gap (eV)");
      for (auto const &r : rows)
        app_log(1, "{:<24} {:>12.6f} {:>12.6f} {:>12.4f}",
                r.map, r.e_homo, r.e_lumo, (r.e_lumo - r.e_homo) * HA2EV);
      app_log(1, "-- per-band values at k = 0 (eV, band index relative to homo-1) --");
      for (int b = 0; b < 4; ++b) {
        double lo = 1e9, hi = -1e9;
        std::string vals;
        for (auto const &r : rows) {
          lo = std::min(lo, r.Ek0(b));
          hi = std::max(hi, r.Ek0(b));
          vals += std::format("{:>12.4f}", r.Ek0(b) * HA2EV);
        }
        app_log(1, "band {:+d}: {}  | spread {:.4f} eV", b - 1, vals, (hi - lo) * HA2EV);
      }

      REQUIRE(rows.size() >= 2);
      const double gap_ref = rows[0].e_lumo - rows[0].e_homo;
      for (auto const &r : rows) {
        const double gap = r.e_lumo - r.e_homo;
        REQUIRE(std::isfinite(gap));
        REQUIRE(gap > 0.0);
        REQUIRE(gap < 2.0);
        // loose tripwire (~1.4 eV) against gross map breakage; the expected
        // surrogate class is ~0.1-0.2 eV and is REPORTED above, not asserted.
        REQUIRE(std::abs(gap - gap_ref) < 0.05);
      }
    }

  } // qp_map_ab_detail

  TEST_CASE("qp_map_ab_lih222", "[methods][qpgw][qp_map_ab]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));

    SECTION("qpscf") {
      std::vector<ab_row> rows;
      for (auto const &map : {"ac_pade", "mats_lin", "mats_gmatch"})
        rows.push_back(run_map(mpi_context, mf, ft, map, "qpscf", 12, 1e-10, 20, 1e-6));
      report_and_check(rows);
    }
    SECTION("evscf") {
      std::vector<ab_row> rows;
      for (auto const &map : {"ac_pade", "mats_lin", "mats_gmatch"})
        rows.push_back(run_map(mpi_context, mf, ft, map, "evscf", 12, 1e-10, 20, 1e-6));
      report_and_check(rows);
    }
  }

  TEST_CASE("qp_map_ab_si222", "[methods][qpgw][qp_map_ab]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    // bdft_si222 is commented out of default_MF -- pyscf_si222 is the plan's
    // named alternative (notes/qpgw_edmft_implementation_plan.md section 4).
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_si222"));

    std::vector<ab_row> rows;
    for (auto const &map : {"ac_pade", "mats_lin", "mats_gmatch"})
      rows.push_back(run_map(mpi_context, mf, ft, map, "qpscf", 10, 1e-8, 12, 1e-5));
    report_and_check(rows);
  }


  /**
   * ==========================================================================================
   * GATE QM3-b -- THE mode_a FIXTURE GATE (spec rev 3 + its addendum: mode_a IS the
   * deliverable again, with the CD evaluator and the STRIP CLAMP).
   * ==========================================================================================
   * History, all measured on 2026-08-12 (kept because it is the justification for the clamp,
   * not decoration):
   *
   *   (ii) THE ANCHOR PASSED FROM THE START: route-B Sigma^c_ab at the first four fermionic
   *        nodes agreed with the gathered solver Sigma(i w_n) to 6.28e-03 over the gap window
   *        against a W-fit reconstruction class of 4.44e-03 (ratio 1.4), and the later tau
   *        oracle put the same elements at 5.6e-05 with NO transform on either side. The
   *        contraction (prefactor, spin, q-star/trev rule, MO rotation, head) IS CORRECT.
   *   (i)  THE LOOP DIVERGED (max|d eps| ~ 1e4-1e5 a.u.) because mode A needs V^xc for ALL
   *        nbnd states, and states OUTSIDE the analyticity strip -- lih222 empty states at
   *        eps - mu = 0.73 / 1.18 / 1.41 a.u., i.e. at and above the IAFT wmax = 1.2 a.u. --
   *        land on the fitted Sigma^c poles at eps_J - om_p (min_den down to 3e-07 a.u.).
   *        No knob cell cured it: eta 0 / 3.14e-3 / 3e-2 a.u. all stayed at 1e3-1e5;
   *        wfit nu improved the reconstruction (1.2e-3 vs 4.4e-3) and failed anyway;
   *        wrtol 1e-6 only moved 9.1e3 -> 8.1e2. The trade-off is STRUCTURAL: the SVD cut
   *        that maximizes imaginary-axis accuracy produces a rational function with thousands
   *        of poles and residues 1e2-1e4x the data, i.e. a wild function at real z.
   *
   * REV 3.1 ADDENDUM item 2 resolves it by CONVENTION, not by tuning: a state inside
   * (VBM - 0.95 E_PH, CBM + 0.95 E_PH) -- the same particle-hole-edge prior the W^c support
   * constraint and the mode_b strip test already use -- is exact mode A, and a state outside
   * it is evaluated at mu, mode_b's fallback. The inner-consistency loop is RETAINED (every
   * evaluation is now bounded), and the clamp census is the acceptance measurement: the judge
   * states (per-k HOMO/LUMO) must not be clamped.
   *
   * The intermediate reading of the addendum -- clamp to the strip BOUNDARY -- was measured
   * and REVERSED here on the same day; the boundary sits inside the fitted-pole pile-up and
   * collapsed lih222 to a -9.6e+03 eV gap. The diagnosis is in qp_modea.hpp, strip_t.
   *
   * This case therefore gates: the tau anchor (in run_map), inner-consistency convergence and
   * a non-vacuous clamp census (in run_map), and finite/sane gaps (report_and_check). The last
   * three matter because THE TAU ANCHOR IS VACUOUS in a collapsed-spectrum regime: it is
   * normalized against the W-fit reconstruction class, which blows up in step with it (that
   * boundary run "passed" the anchor at ratio 0.007 with a negative gap). The gap next to the
   * stored ac_pade value is REPORTED, never gated.
   * QPSCF ONLY -- the evGW leg's diagonal sampler is pathologically slow (rev 3 addendum
   * item 4); see the [.modeb_evscf] case below.
   */
  TEST_CASE("qp_map_modea_lih222", "[methods][qpgw][qp_map_ab][modea2]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));

    std::vector<ab_row> rows;
    for (auto const &map : {"ac_pade", "mode_a"})
      rows.push_back(run_map(mpi_context, mf, ft, map, "qpscf", 12, 1e-10, 20, 1e-6));
    report_and_check(rows);
    app_log(1, "@@MODEA_GAP lih222/qpscf: ac_pade = {:.4f} eV, mode_a = {:.4f} eV "
               "(d = {:+.4f} eV); stored references: ac_pade 11.8024, mode_b 11.8537",
            rows[0].gap_eV(), rows[1].gap_eV(), rows[1].gap_eV() - rows[0].gap_eV());
  }

  TEST_CASE("qp_map_modea_si222", "[methods][qpgw][qp_map_ab][modea2]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_si222"));

    std::vector<ab_row> rows;
    for (auto const &map : {"ac_pade", "mode_a"})
      rows.push_back(run_map(mpi_context, mf, ft, map, "qpscf", 10, 1e-8, 12, 1e-5));
    report_and_check(rows);
    app_log(1, "@@MODEA_GAP si222/qpscf: ac_pade = {:.4f} eV, mode_a = {:.4f} eV "
               "(d = {:+.4f} eV); stored references: ac_pade 9.2484, mode_b 9.2615",
            rows[0].gap_eV(), rows[1].gap_eV(), rows[1].gap_eV() - rows[0].gap_eV());
    // THE JUDGE STATES on the si222-class gap: the measured strip bounds are VBM + 0.155 /
    // CBM + 0.496 a.u. against band edges spread by ~0.34 a.u., so the per-k band edges are
    // deep inside the strip and must never be clamped (spec rev 3 addendum item 2).
    REQUIRE(rows[1].lr.n_homo_clamp == 0);
    REQUIRE(rows[1].lr.n_lumo_clamp == 0);
  }


  /**
   * GATE QM3-b' -- the mode_b fixture gate (spec rev 2, the user ruling of 2026-08-12; mode_b
   * is now the AUXILIARY map, rev 3, and stays gated because mode_a shares all of its
   * machinery). V^xc_ab = Re Sigma^c_ab(mu) off-diagonal, V^xc_aa = Re Sigma^c_aa(eps_a)
   * diagonal, with the CD evaluator. No inner-consistency loop. The loops must CONVERGE; the
   * tau anchor is the acceptance criterion; the diagonal-fallback count is logged (expected 0
   * near the gap). QPSCF ONLY -- see [.modeb_evscf] below.
   */
  TEST_CASE("qp_map_modeb_lih222", "[methods][qpgw][qp_map_ab][modeb]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));

    std::vector<ab_row> rows;
    for (auto const &map : {"ac_pade", "mats_lin", "mats_gmatch", "mode_b"})
      rows.push_back(run_map(mpi_context, mf, ft, map, "qpscf", 12, 1e-10, 20, 1e-6));
    report_and_check(rows);
  }

  /**
   * THE evGW (evscf) LEG OF THE CD MAPS -- HIDDEN, INCOMPLETE BY DESIGN (spec rev 3 addendum
   * item 4). It was a SECTION of the live gate above until it was measured, on 2026-08-12, to
   * take >75 min for a single outer iteration on qe_lih222 against ~7 s for the qsGW (qpscf)
   * leg of the same fixture, __divdc3-bound: solve_qp_eqn's modea_diag_sampler rebuilds the
   * whole nJ x npk pole-weight vector (thousands of complex divisions) at EVERY secant /
   * bisection step of EVERY (s,k,a), instead of caching the per-(s,k,a) residue slab and
   * evaluating incrementally. Until that is fixed the leg is neither gated nor deliverable;
   * the driver emits a flagged warning when it is entered. Run explicitly if you need it:
   *
   *     test_methods_qp_map_ab "qp_map_modeb_lih222_evscf"
   */
  TEST_CASE("qp_map_modeb_lih222_evscf", "[.modeb_evscf]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));

    std::vector<ab_row> rows;
    for (auto const &map : {"ac_pade", "mats_lin", "mats_gmatch", "mode_b"})
      rows.push_back(run_map(mpi_context, mf, ft, map, "evscf", 12, 1e-10, 20, 1e-6));
    report_and_check(rows);
  }

  TEST_CASE("qp_map_modeb_si222", "[methods][qpgw][qp_map_ab][modeb]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_si222"));
    std::vector<ab_row> rows;
    for (auto const &map : {"ac_pade", "mats_lin", "mats_gmatch", "mode_b"})
      rows.push_back(run_map(mpi_context, mf, ft, map, "qpscf", 10, 1e-8, 12, 1e-5));
    report_and_check(rows);
  }

  /**
   * MODE-B MEASUREMENT MATRIX, CONFIG (iii) -- measurement only, hidden tag, no gate is
   * added and no default is flipped here.
   *
   * The matrix has two configs, both with the STRIP TEST as the diagonal fallback criterion
   * (the earlier min_den-floor / |ReSigma| heuristic triggers no longer exist in the tree):
   *   (ii)  strip test + the DEFAULT W^c fit  (wfit = "tau", wrtol = doctrine)
   *         -> this IS the [modeb] gate case above; no separate case is needed.
   *   (iii) strip test + wfit = "nu", wrtol = 1e-4                 <- THIS CASE
   *
   * Full qp_scf_loop, same niter/conv_tol/THC settings as the ac_pade rows of the
   * corresponding fixture, so the final gaps are directly comparable to the stored
   * ac_pade / mats_lin / mats_gmatch references. One SECTION per fixture: the tau-anchor
   * check inside the driver aborts the process on failure, so the sections are meant to be
   * run one per invocation (catch2 `-c <section>`).
   */
  TEST_CASE("qp_map_modeb_matrix", "[.modeb_matrix]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);

    // config: (ii) default fit  |  (iii) nu route at wrtol 1e-4
    std::string cfg = "ii", wfit = "tau";
    double wrtol = -1.0;
    const char *cenv = std::getenv("COQUI_MODEB_CFG");
    if (cenv != nullptr and std::string(cenv) == "iii") {
      cfg = "iii"; wfit = "nu"; wrtol = 1e-4;
    }
    app_log(1, "\n@@MODEB_CFG {} : strip test + wfit = {}, wrtol = {:.1e}", cfg, wfit, wrtol);

    auto report = [&](std::string const &fix, ab_row const &row) {
      app_log(1, "@@MODEB_ROW cfg={} fixture={} gap={:.6f} eV homo={:.6f} lumo={:.6f} "
                 "taudev={:.4e} recclass={:.4e} ratio={:.3g} antiherm={:.3e} minden={:.4e} "
                 "fallbacks={} nodes={}/{} nJ={} npk={} wfit={}",
              cfg, fix, row.gap_eV(), row.e_homo, row.e_lumo, row.lr.tau_dev,
              row.lr.anchor_expect,
              (row.lr.anchor_expect > 0.0 ? row.lr.tau_dev / row.lr.anchor_expect : 0.0),
              row.lr.anti_herm, row.lr.min_den, row.lr.n_fallback, row.lr.n_support,
              row.lr.np_total, row.lr.nJ, row.lr.npk, row.lr.wfit);
    };

    SECTION("lih222_qpscf") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
      report("qe_lih222/qpscf",
             run_map(mpi_context, mf, ft, "mode_b", "qpscf", 12, 1e-10, 20, 1e-6,
                     wfit, "_lih_q_" + cfg, 0.0, wrtol));
    }
    SECTION("si222_qpscf") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_si222"));
      report("pyscf_si222/qpscf",
             run_map(mpi_context, mf, ft, "mode_b", "qpscf", 10, 1e-8, 12, 1e-5,
                     wfit, "_si_q_" + cfg, 0.0, wrtol));
    }
    SECTION("lih222_evscf") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
      report("qe_lih222/evscf",
             run_map(mpi_context, mf, ft, "mode_b", "evscf", 12, 1e-10, 20, 1e-6,
                     wfit, "_lih_e_" + cfg, 0.0, wrtol));
    }
  }

  /**
   * ==========================================================================================
   * GATE (spec rev 4): THE eta_far = 0 BIT-IDENTITY
   * ==========================================================================================
   * Rev 4 adds the graded-eta far-state evaluation (qp_modea_eta_far). Its DEFAULT is 0, which
   * must reproduce the rev-3.1 mu fallback exactly -- the whole increment is then a no-op on
   * every stored number. THAT is what this case gates, and only that: the four converged gaps
   * of the tree at 690ade1, with eta_far passed EXPLICITLY as 0 so the plumbing (params ->
   * opts -> ctx -> strip_t::zeval) is exercised rather than bypassed.
   *
   *   fixture        map      gap (eV) at 690ade1
   *   qe_lih222      mode_a   11.856870
   *   qe_lih222      mode_b   11.853654
   *   pyscf_si222    mode_a    9.257495
   *   pyscf_si222    mode_b    9.261511
   *
   * Tolerance 1e-5 eV: the references are quoted to 1e-6 eV, and the failure this guards
   * against (an out-of-strip state silently evaluated at eps instead of mu) moves the gap by
   * O(0.1-1 eV) or diverges the loop. The eta_far > 0 cells are a MEASUREMENT, not a gate --
   * they live in [.etafar_scan] below.
   */
  TEST_CASE("qp_map_etafar_identity", "[methods][qpgw][qp_map_ab][etafar]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    constexpr double tol = 1e-5;   // eV

    auto check = [&](std::string const &fix, std::string const &map, double gap, double ref) {
      app_log(1, "@@ETAFAR0 [{}/{}] gap = {:.6f} eV, reference (690ade1) = {:.6f} eV, "
                 "d = {:+.2e} eV", fix, map, gap, ref, gap - ref);
      REQUIRE(std::abs(gap - ref) < tol);
    };
    {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
      auto ra = run_map(mpi_context, mf, ft, "mode_a", "qpscf", 12, 1e-10, 20, 1e-6,
                        "tau", "_ef0a", 0.0, -1.0, 1e-10, 0, 0.0);
      check("qe_lih222", "mode_a", ra.gap_eV(), 11.856870);
      REQUIRE(ra.lr.n_eta == 0);
      auto rb = run_map(mpi_context, mf, ft, "mode_b", "qpscf", 12, 1e-10, 20, 1e-6,
                        "tau", "_ef0b", 0.0, -1.0, 1e-10, 0, 0.0);
      check("qe_lih222", "mode_b", rb.gap_eV(), 11.853654);
    }
    {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_si222"));
      auto ra = run_map(mpi_context, mf, ft, "mode_a", "qpscf", 10, 1e-8, 12, 1e-5,
                        "tau", "_ef0a", 0.0, -1.0, 1e-10, 0, 0.0);
      check("pyscf_si222", "mode_a", ra.gap_eV(), 9.257495);
      REQUIRE(ra.lr.n_eta == 0);
      auto rb = run_map(mpi_context, mf, ft, "mode_b", "qpscf", 10, 1e-8, 12, 1e-5,
                        "tau", "_ef0b", 0.0, -1.0, 1e-10, 0, 0.0);
      check("pyscf_si222", "mode_b", rb.gap_eV(), 9.261511);
    }
  }

  /**
   * ==========================================================================================
   * MEASUREMENT (hidden, run explicitly): THE eta_far SCAN of spec rev 4
   * ==========================================================================================
   * eta_far in {0, 1.8e-3, 3.7e-3, 7.3e-3} a.u. (= 0, 0.05, 0.1, 0.2 eV), full loops, both
   * maps. Reports convergence, the final gap and its drift vs eta_far, the tau anchor, the
   * IN-STRIP anti-Hermitian residual, max|Im Sigma| off strip, the measured pole spacing and
   * the out-of-strip census. On these fixtures the gaps are huge and few states sit near the
   * strip boundary, so little movement is EXPECTED -- the real target is the kp222 judge
   * rerun. One section per (fixture, map); run them one at a time:
   *
   *     KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
   *       <build>/tests/bin/test_methods_qp_map_ab "qp_map_etafar_scan" -c lih222_mode_a
   */
  TEST_CASE("qp_map_etafar_scan", "[.etafar_scan]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    const std::vector<double> etas = {0.0, 1.8e-3, 3.7e-3, 7.3e-3};

    auto scan = [&](std::string const &fixture, std::string const &map, int pref, double thc_tol,
                    int niter, double conv_tol) {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, fixture));
      std::vector<ab_row> rows;
      for (size_t c = 0; c < etas.size(); ++c) {
        app_log(1, "\n@@ETAFAR cell {}: {} / {} / eta_far = {:.4e} a.u. ({:.4g} eV)",
                c, fixture, map, etas[c], etas[c] * HA2EV);
        rows.push_back(run_map(mpi_context, mf, ft, map, "qpscf", pref, thc_tol, niter,
                               conv_tol, "tau", "_ef" + std::to_string(c), 0.0, -1.0, 1e-10,
                               0, etas[c], false));
      }
      app_log(1, "\n== rev-4 eta_far scan: {} / {} ==", fixture, map);
      app_log(1, "{:>12} {:>10} {:>12} {:>12} {:>10} {:>10} {:>7} {:>11} {:>11} {:>11} {:>11}",
              "eta_far (eV)", "gap (eV)", "d vs eta=0", "outer conv", "inner dmax", "taudev",
              "n_eta", "antiherm", "im_off", "spacing", "eta/spacing");
      for (size_t c = 0; c < rows.size(); ++c) {
        auto const &L = rows[c].lr;
        app_log(1, "{:>12.4g} {:>10.6f} {:>12.2e} {:>12} {:>10.2e} {:>10.2e} {:>7} {:>11.3e} "
                   "{:>11.3e} {:>11.3e} {:>11.3g}",
                etas[c] * HA2EV, rows[c].gap_eV(), rows[c].gap_eV() - rows[0].gap_eV(),
                (rows[c].final_iter < rows[c].niter ? "yes" : "NO (cap)"), L.dmax, L.tau_dev,
                L.n_eta, L.anti_herm, L.im_off, L.spacing,
                (L.spacing > 0.0 ? etas[c] / L.spacing : 0.0));
      }
      app_log(1, "  [out-of-strip census of the last iteration: {} of {} evaluations, per-k "
                 "HOMO in {} blocks, LUMO in {}]", rows.back().lr.n_clamp,
              rows.back().lr.n_eval, rows.back().lr.n_homo_clamp, rows.back().lr.n_lumo_clamp);
    };

    SECTION("lih222_mode_a") { scan("qe_lih222", "mode_a", 12, 1e-10, 20, 1e-6); }
    SECTION("lih222_mode_b") { scan("qe_lih222", "mode_b", 12, 1e-10, 20, 1e-6); }
    SECTION("si222_mode_a")  { scan("pyscf_si222", "mode_a", 10, 1e-8, 12, 1e-5); }
    SECTION("si222_mode_b")  { scan("pyscf_si222", "mode_b", 10, 1e-8, 12, 1e-5); }
  }

  /**
   * KNOB MATRIX HARNESS (measurement only -- no gate, no default is adopted here).
   *
   * ONE CELL PER PROCESS, selected by the environment variable COQUI_MODEA_CELL:
   * a cell whose W^c fit is too coarse trips the anchor check in qp_scf_common.cpp and
   * MPI_Aborts, which is exactly the behaviour the failing default-cell gate must KEEP, so
   * the sweep is driven from outside instead of weakening the check. Every number the sweep
   * needs is on the "@@MODEA_CELL" line, which the driver emits before any abort can happen.
   *
   * cell = iw*9 + ir*3 + ie  over  wfit {tau, nu} x wrtol {1e-8, 1e-6, 1e-4}
   *                                x eta {0, 3.7e-3, 1.1e-2} a.u. (0 / ~0.1 / ~0.3 eV).
   */
  TEST_CASE("qp_map_modea_knob_matrix", "[.modea_hold]") {
    using namespace qp_map_ab_detail;
    const char *cenv = std::getenv("COQUI_MODEA_CELL");
    if (cenv == nullptr) {
      app_log(1, "qp_map_modea_knob_matrix: COQUI_MODEA_CELL unset -- nothing to measure.");
      return;
    }
    const int cell = std::atoi(cenv);
    const char *wfits[2] = {"tau", "nu"};
    const double rtols[3] = {1e-8, 1e-6, 1e-4};
    const double etas[3] = {0.0, 3.7e-3, 1.1e-2};
    REQUIRE(cell >= 0);
    REQUIRE(cell < 18);
    const int iw = cell / 9, ir = (cell / 3) % 3, ie = cell % 3;

    // optional: raise the DLR precision of the WHOLE calculation, which sharpens the
    // reference Sigma(i w_n) as well as the mode-A fit. Discriminates a fit-limited anchor
    // from a representation-limited one.
    const char *penv = std::getenv("COQUI_MODEA_PREC");
    const std::string prec = (penv != nullptr) ? penv : "medium";
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis, prec);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    app_log(1, "\n@@CELL {} : prec = {}, wfit = {}, wrtol = {:.0e}, eta = {:.4e} a.u. "
               "({:.4g} eV)", cell, prec, wfits[iw], rtols[ir], etas[ie], etas[ie] * HA2EV);
    auto row = run_map(mpi_context, mf, ft, "mode_a", "qpscf", 12, 1e-10, 1, 1e-6,
                       wfits[iw], "_cell" + std::to_string(cell), etas[ie], rtols[ir]);
    app_log(1, "@@CELLDONE {} gap = {:.6f} eV", cell, row.gap_eV());
  }

  /**
   * ==========================================================================================
   * MEASUREMENT (on hold, run explicitly): THE W^c SLAB LOW-RANK SCAN
   * ==========================================================================================
   * The mode-A context build is dominated by the (Np,Np)x(Np,nbnd) sandwich, whose cost is
   * nqpts*nbnd*npk*8*Np*nbnd*(Np+nbnd) flops per owned (s,k) -- 1.2e14 at the production
   * (nbnd 60, Np 2918, nq 8, npk 60), i.e. the measured ~45 min/iteration. Factoring each
   * residue slab as W^(p) = V S V^dag and contracting through V makes that r/Np of the dense
   * cost (wc_band_elements.hpp, stage 1b + the header's flop model).
   *
   * This case measures the two things that decision rests on, on a fixture small enough that
   * the EXACT eigendecomposition is affordable:
   *   (i)   the eigenvalue decay of the slabs -- the "rank ladder" line of every context
   *         build reports max/mean rank at |lambda| >= {1e-2 ... 1e-10} * max|lambda|;
   *   (ii)  the gap moved by the truncation, against the dense reference path (wrank <= 0),
   *         and the agreement of the RANDOMIZED backend (the production one, which the
   *         gates cannot reach because they run below detail::wslab_dense_max) with the
   *         exact one at the same tolerance.
   *
   *     KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
   *       <build>/tests/bin/test_methods_qp_map_ab "qp_map_modea_wrank_scan"
   */
  TEST_CASE("qp_map_modea_wrank_scan", "[.modea_hold]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));

    // cell 1 isolates the two accuracy effects of the factored path: it truncates nothing
    // (r ~ Np) but still replaces each slab by its Hermitian part, so cell1 - cell0 is the
    // HERMITIZATION alone and cell2 - cell1 is the truncation alone.
    struct cell { const char *name; double wrank; long wsketch; };
    const std::vector<cell> cells = {
        {"dense reference (wrank = 0)", 0.0, 0},
        {"hermitize only (wrank = 1e-14)", 1e-14, -1},
        {"exact heev,  wrank = 1e-10", 1e-10, -1},
        {"exact heev,  wrank = 1e-8", 1e-8, -1},
        {"sketch 32 -> heev fallback", 1e-10, 32},
    };
    // NOT a cell: wrank = 1e-4 ABORTS (measured) -- the tau anchor of qp_approx fires at
    // 3.6e-1 against its 10 x 4.4e-3 gate. That is the intended interlock: an over-aggressive
    // slab truncation is a CONTRACTION error to the anchor, and it stops the run rather than
    // quietly shifting the spectrum. The knob cannot silently buy speed with accuracy.
    std::vector<ab_row> rows;
    for (size_t c = 0; c < cells.size(); ++c) {
      app_log(1, "\n@@WRANK cell {}: {}", c, cells[c].name);
      rows.push_back(run_map(mpi_context, mf, ft, "mode_a", "qpscf", 12, 1e-10, 20, 1e-6,
                             "tau", "_wrank" + std::to_string(c), 0.0, -1.0,
                             cells[c].wrank, cells[c].wsketch));
    }
    app_log(1, "\n  W^c slab low-rank scan (qe_lih222 / mode_a / qpscf), Np = {}:",
            rows[0].lr.Np);
    app_log(1, "  {:<30} {:>10} {:>12} {:>8} {:>9} {:>10} {:>9} {:>9}",
            "cell", "gap (eV)", "d vs dense", "max r", "mean r", "worst res", "t_fac", "t_sand");
    for (size_t c = 0; c < cells.size(); ++c)
      app_log(1, "  {:<30} {:>10.6f} {:>12.2e} {:>8} {:>9.2f} {:>10.2e} {:>8.2f}s {:>8.2f}s",
              cells[c].name, rows[c].gap_eV(), rows[c].gap_eV() - rows[0].gap_eV(),
              rows[c].lr.wrank_max, rows[c].lr.wrank_mean, rows[c].lr.wtrunc,
              rows[c].lr.t_fac, rows[c].lr.t_sand);
    // the DEFAULT cut (cell 2) must not move the gap at the resolution the QM3 gates compare at
    REQUIRE(std::abs(rows[2].gap_eV() - rows[0].gap_eV()) < 1e-6);
    // ... and neither may the Hermitization on its own (cell 1)
    REQUIRE(std::abs(rows[1].gap_eV() - rows[0].gap_eV()) < 1e-6);
  }

  /**
   * ==========================================================================================
   * MEASUREMENT (on hold, run explicitly): THE RANDOMIZED BACKEND AGAINST THE EXACT ONE
   * ==========================================================================================
   * detail::wslab_factorize takes LAPACK heev up to Np = 600 and the randomized Nystrom
   * sketch above it, so the PRODUCTION backend is the one no fixture reaches by default.
   * Forcing the sketch on a fixture is only meaningful where it can actually resolve the
   * tail: the sketch doubles and then gives up to heev once 2l > Np, and at wrank = 1e-10
   * the fixture rank (~135 of 192) is above that ceiling, so the cut is loosened here until
   * the retained rank (~72 mean / 86 max of 192) is inside it -- the backend then reports
   * "mixed", the slabs that fit being sketched and the rest falling back. 1e-6 is as loose as
   * this can go: at 1e-4 the tau anchor of qp_approx aborts the run (measured). One outer
   * iteration -- this compares two factorizations of the SAME W, not converged solutions.
   *
   *     KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
   *       <build>/tests/bin/test_methods_qp_map_ab "qp_map_modea_sketch_check"
   */
  TEST_CASE("qp_map_modea_sketch_check", "[.modea_hold]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    auto ex = run_map(mpi_context, mf, ft, "mode_a", "qpscf", 12, 1e-10, 1, 1e-6,
                      "tau", "_skx", 0.0, -1.0, 1e-6, -1);
    auto rd = run_map(mpi_context, mf, ft, "mode_a", "qpscf", 12, 1e-10, 1, 1e-6,
                      "tau", "_skr", 0.0, -1.0, 1e-6, 32);
    app_log(1, "\n  W^c slab factorization backends at wrank = 1e-6 (qe_lih222, 1 iteration):");
    app_log(1, "  {:<14} {:>12} {:>8} {:>10} {:>12}", "backend", "gap (eV)", "max r",
            "mean r", "worst res");
    app_log(1, "  {:<14} {:>12.6f} {:>8} {:>10.2f} {:>12.2e}", "exact heev", ex.gap_eV(),
            ex.lr.wrank_max, ex.lr.wrank_mean, ex.lr.wtrunc);
    app_log(1, "  {:<14} {:>12.6f} {:>8} {:>10.2f} {:>12.2e}", "randomized", rd.gap_eV(),
            rd.lr.wrank_max, rd.lr.wrank_mean, rd.lr.wtrunc);
    app_log(1, "  gap difference between backends = {:.3e} eV", rd.gap_eV() - ex.gap_eV());
    // the sketch must find the same dominant subspace, well inside the cut it was given
    REQUIRE(std::abs(rd.gap_eV() - ex.gap_eV()) < 1e-3);
    REQUIRE(rd.lr.wrank_max > 0);
  }

  /**
   * ==========================================================================================
   * MEASUREMENT (on hold, run explicitly): DOES THE SLAB RANK SATURATE WITH Np?
   * ==========================================================================================
   * THE question for the production sizes. The low-rank sandwich costs r/Np of the dense one,
   * so the compression that matters is not r/Np at the fixture's Np -- it is whether r is set
   * by the PHYSICS (a fixed number of screening modes per pole, so r/Np falls as 1/Np and the
   * Np = 2918 production case wins by ~Np/r) or by the BASIS (r proportional to Np, so the
   * ratio is frozen at whatever the fixture shows and kp444 needs a different idea).
   *
   * Same cell, same everything, ONE outer iteration, THC prefactor swept: Np = 8/12/18/24
   * times nbnd. Only the auxiliary basis changes, so any growth of r with Np is basis-driven.
   *
   *     KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
   *       <build>/tests/bin/test_methods_qp_map_ab "qp_map_modea_np_scan"
   */
  TEST_CASE("qp_map_modea_np_scan", "[.modea_hold]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));

    const std::vector<int> prefs = {6, 12, 18, 24};
    std::vector<ab_row> rows;
    for (size_t c = 0; c < prefs.size(); ++c) {
      app_log(1, "\n@@NPSCAN cell {}: thc prefactor {} (Np = {} x nbnd)", c, prefs[c], prefs[c]);
      rows.push_back(run_map(mpi_context, mf, ft, "mode_a", "qpscf", prefs[c], 1e-10, 1, 1e-6,
                             "tau", "_np" + std::to_string(c), 0.0, -1.0, 1e-8, -1));
    }
    app_log(1, "\n  W^c slab rank vs THC basis size (qe_lih222 / mode_a, 1 outer iteration).");
    app_log(1, "  mean retained rank over the (q,p) slabs, and r/Np, at each tolerance:");
    app_log(1, "  {:>6} {:>26} {:>26} {:>26} {:>26}", "Np", "1e-4", "1e-6", "1e-8", "1e-10");
    for (size_t c = 0; c < prefs.size(); ++c) {
      std::string line = std::format("  {:>6}", rows[c].lr.Np);
      for (int t = 1; t < 5; ++t)
        line += std::format(" {:>12.1f} (r/Np {:>6.3f})", rows[c].lr.lad_mean[t],
                            rows[c].lr.lad_mean[t] / double(std::max(rows[c].lr.Np, 1L)));
      app_log(1, "{}", line);
    }
    app_log(1, "  [if the mean rank is FLAT down the columns the compression scales as 1/Np "
               "and the production Np = 2918 wins by ~Np/r; if r/Np is flat it does not.]");
  }

  /**
   * WORK-SHARING GATE (stage 2 distribution): qe_lih222_sym has THREE (s,k) blocks and eight
   * (isym, q-in-star) pairs, so ANY run with more than three ranks makes the other ranks
   * helpers on somebody's block. That is the only configuration in this suite which exercises
   * the pair split and the per-pair group reduction of stage 2 -- lih222/si222 have 8 blocks
   * and lih223_sym 6, i.e. they need 9 / 7 ranks, which the ISDF setup of these fixtures does
   * not survive on a laptop. One outer iteration; the gap and the anchor are rank-count
   * invariants, so the SAME numbers must come out at -np 1 and -np 4:
   *
   *     KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 mpirun -np 4 \
   *       <build>/tests/bin/test_methods_qp_map_ab "qp_map_modea_worksharing"
   *
   * [measured, 2026-08-13: gap 12.008579 eV and tau anchor 5.0378e-04 at both -np 1 and -np 2
   *  (the anchor read 7.1816e-04 under the per-element normalization retired the same day --
   *  see "THE GATE'S NORMALIZATION" in wc_band_elements.hpp; the gap is unchanged by it);
   *  >= 4 ranks is not reachable on this laptop -- the ISDF setup of every mode_a fixture
   *  aborts there (thc_reader_t::build, "create_plan_many: howmany=0"), which predates this
   *  increment. A run-time tripwire in wc_band_elements.hpp checks the pair census of every
   *  group instead, so a mis-split aborts rather than silently dropping a q.]
   *
   * The same case is ALSO the union-subspace agreement gate: qe_lih222_sym is the only fixture
   * in the suite with both D-matrix rotations and time-reversed q, so it is the only one that
   * exercises the wconj branch of the stage-1c contraction (the two unreduced fixtures never
   * enter it). At wunion = wrank the basis carries every direction the slab cut kept, to the
   * same tolerance, so the two paths must agree at the resolution the QM3 gates compare at.
   */
  TEST_CASE("qp_map_modea_worksharing", "[methods][qpgw][qp_map_ab][modea2]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_sym"));
    REQUIRE(mf->nkpts_ibz() < mf->nkpts());
    // union ON at wunion = wrank (the restructure is OFF by default -- see the scan below --
    // so it has to be asked for explicitly, and this case is where it is gated)
    auto row = run_map(mpi_context, mf, ft, "mode_a", "qpscf", 12, 1e-10, 1, 1e-6, "tau", "_ws",
                       0.0, -1.0, 1e-10, 0, 0.0, true, 0.0);
    app_log(1, "@@WORKSHARE ranks = {}, (s,k) blocks = {}, (isym,q) pairs = {}: gap = {:.6f} eV,"
               " TAU ANCHOR = {:.4e} (class {:.4e}), inner max|d eps| = {:.3e}",
            mpi_context->comm.size(), mf->nkpts_ibz(), mf->nqpts(), row.gap_eV(),
            row.lr.tau_dev, row.lr.anchor_expect, row.lr.dmax);
    REQUIRE(row.lr.tau_dev < qp_modea::modea_tau_anchor_mult * row.lr.anchor_expect);
    REQUIRE(std::isfinite(row.gap_eV()));
    REQUIRE(row.gap_eV() > 0.0);

    auto ref = run_map(mpi_context, mf, ft, "mode_a", "qpscf", 12, 1e-10, 1, 1e-6, "tau",
                       "_wsref");                                          // union OFF (default)
    app_log(1, "@@UNIONID union ON (R = {} of Np = {}, worst projection residual {:.3e}) vs the "
               "per-slab path: gap {:.6f} vs {:.6f} eV, d = {:+.2e}; anchor {:.4e} vs {:.4e}",
            row.lr.union_R_max, row.lr.Np, row.lr.union_tail, row.gap_eV(), ref.gap_eV(),
            row.gap_eV() - ref.gap_eV(), row.lr.tau_dev, ref.lr.tau_dev);
    REQUIRE(std::abs(row.gap_eV() - ref.gap_eV()) < 1e-6);
  }

  /**
   * ==========================================================================================
   * MEASUREMENT (on hold, run explicitly): THE UNION-SUBSPACE CUT -- WHAT R BUYS, AND AT WHAT
   * ==========================================================================================
   * Stage 1c replaces the npk per-slab bases of one q by ONE basis of R_q vectors, so the Np
   * axis of the sandwich is contracted R times per (k,q,n) instead of sum_p r_p times. The
   * whole restructure is worth R/Np -- and R is a function OF THE CUT, not of the cell: the
   * np_scan probe measures the stack rank saturating at 89 (1e-6) and 143 (1e-8) of Np = 384
   * while at the default 1e-10 it is Np itself (the retained tails of different poles are
   * mutually orthogonal there). So this case is the one that decides the default: each cell
   * runs the SAME full loop with a different qp_modea_wunion at fixed wrank = 1e-10, and
   * reports the gap against the union-OFF reference together with R, the projection residual
   * and the tau anchor.
   *
   * The reference cell is wunion < 0 = the per-slab stage-1b path of 18d35a3, so cell k minus
   * cell 0 IS the union restructure's accuracy cost, isolated from everything else.
   *
   *     KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
   *       <build>/tests/bin/test_methods_qp_map_ab "qp_map_modea_wunion_scan"
   */
  TEST_CASE("qp_map_modea_wunion_scan", "[.modea_hold]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));

    // 1e-4 is NOT a cell: the tau anchor of qp_approx aborts the process there (measured for
    // wrank; the union cut carries the same error class), and an abort loses the table.
    struct cell { const char *name; double wunion; };
    const std::vector<cell> cells = {
        {"union OFF (per-slab, 18d35a3)", -1.0},
        {"wunion = wrank = 1e-10", 0.0},
        {"wunion = 1e-8", 1e-8},
        {"wunion = 1e-6", 1e-6},
    };
    std::vector<ab_row> rows;
    for (size_t c = 0; c < cells.size(); ++c) {
      app_log(1, "\n@@WUNION cell {}: {}", c, cells[c].name);
      rows.push_back(run_map(mpi_context, mf, ft, "mode_a", "qpscf", 12, 1e-10, 20, 1e-6,
                             "tau", "_wu" + std::to_string(c), 0.0, -1.0, 1e-10, 0, 0.0,
                             true, cells[c].wunion));
    }
    app_log(1, "\n  W^c union-subspace scan (qe_lih222 / mode_a / qpscf), Np = {}, "
               "wrank = 1e-10:", rows[0].lr.Np);
    app_log(1, "  {:<32} {:>10} {:>11} {:>7} {:>8} {:>11} {:>10} {:>9} {:>9}",
            "cell", "gap (eV)", "d vs OFF", "max R", "R/Np", "proj resid", "anchor",
            "t_1c", "t_sand");
    for (size_t c = 0; c < cells.size(); ++c)
      app_log(1, "  {:<32} {:>10.6f} {:>11.2e} {:>7} {:>8.3f} {:>11.2e} {:>10.2e} {:>8.2f}s "
                 "{:>8.2f}s",
              cells[c].name, rows[c].gap_eV(), rows[c].gap_eV() - rows[0].gap_eV(),
              rows[c].lr.union_R_max,
              rows[c].lr.union_R_max / double(std::max(rows[c].lr.Np, 1L)),
              rows[c].lr.union_tail, rows[c].lr.tau_dev, rows[c].lr.t_union,
              rows[c].lr.t_sand);
    app_log(1, "  [the DEFAULT is the loosest cell whose gap still sits inside the 1e-5 eV "
               "gate band of qp_map_etafar_identity AND whose tau anchor is inside its own "
               "10 x reconstruction-class interlock.]");
    // the exact cell (wunion = wrank) must reproduce the per-slab path: at that cut the union
    // basis spans every retained direction, so the restructure is algebra, not truncation
    REQUIRE(std::abs(rows[1].gap_eV() - rows[0].gap_eV()) < 1e-6);
  }

  /**
   * SYMMETRY SMOKE GATE (sanctioned extension): the ANCHOR on qe_lih223_sym, one outer
   * iteration, mode_a default cell. Both qp_map_ab fixtures are unreduced meshes (8 k-points,
   * 8 in the IBZ), so they never enter the isym loop, the D-matrix external rotation
   * XCe = X(ks) D C(k), or the time-reversal conj branches of the contraction. This case is
   * the first thing that does; without it the kp222 judge would be.
   */
  TEST_CASE("qp_map_modeb_sym_anchor", "[methods][qpgw][qp_map_ab][modeb]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih223_sym"));
    app_log(1, "\n== mode_a symmetry smoke: nkpts = {} ({} in the IBZ), nqpts = {} ({} in the "
               "IBZ), q-symmetries = {}, trev k-pairs = {} ==",
            mf->nkpts(), mf->nkpts_ibz(), mf->nqpts(), mf->nqpts_ibz(),
            mf->qsymms().size(), mf->nkpts_trev_pairs());
    REQUIRE(mf->nkpts_ibz() < mf->nkpts());   // the point of the fixture
    auto row = run_map(mpi_context, mf, ft, "mode_b", "qpscf", 12, 1e-10, 1, 1e-6);
    app_log(1, "mode_b symmetry smoke: TAU ANCHOR = {:.4e} (gate {:.1g} x class), W-fit "
               "reconstruction class = {:.4e}, ratio = {:.3g}", row.lr.tau_dev,
            qp_modea::modea_tau_anchor_mult,
            row.lr.anchor_expect,
            (row.lr.anchor_expect > 0.0 ? row.lr.tau_dev / row.lr.anchor_expect : 0.0));
    REQUIRE(row.lr.tau_dev < qp_modea::modea_tau_anchor_mult * row.lr.anchor_expect);
  }

  /**
   * THE GAMMA-HEAD GATE (2026-08-13). Until today the head augmentation of stage 1
   * (wc_band_elements.hpp, "Gamma head") was the ONE code path of the map that no gate
   * touched: every QM3 fixture and the QM3-c judge run div_treatment = ignore_g0, where the
   * head is absent on BOTH sides by construction. It is also the only place where this map
   * and the reference build the same physics by DIFFERENT routes -- the map adds
   * W^head_PQ(tau) = nk madelung eps_inv_head(tau) conj(chi_P) chi_Q into the q = Gamma slab
   * and lets it ride through the ordinary contraction, while gw_t::Sigma_div_correction
   * (thc_gw.icc:444-531) forms -madelung eps_inv_head(tau) T G T^dag directly at each IBZ k.
   * The tau anchor is exactly the comparison that closes that loop, so it is now run with the
   * head ON. Same fixture as the qp_map_ab gates, no symmetry (12 k, 12 in the IBZ).
   *
   * [measured 2026-08-13, block-normalized semantics: anchor 3.4578e-04, ratio 0.158 of the
   *  gate, from an absolute deviation of 7.11e-05 a.u. on a block scale of 2.057e-01 -- against
   *  4.1824e-04 from 6.77e-05 a.u. on 1.620e-01 with the head OFF. The head adds ~27% to
   *  |Sigma| and moves the absolute deviation by < 5%.]
   */
  TEST_CASE("qp_map_modeb_head_anchor", "[methods][qpgw][qp_map_ab][modeb]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih223"));
    auto row = run_map(mpi_context, mf, ft, "mode_b", "qpscf", 12, 1e-10, 1, 1e-6, "tau",
                       "_head", 0.0, -1.0, 1e-10, 0, 0.0, true, -1.0, "gygi");
    app_log(1, "mode_b GAMMA HEAD gate (div_treatment = gygi): TAU ANCHOR = {:.4e}, W-fit "
               "reconstruction class = {:.4e}, ratio = {:.3g}", row.lr.tau_dev,
            row.lr.anchor_expect,
            (row.lr.anchor_expect > 0.0 ? row.lr.tau_dev / row.lr.anchor_expect : 0.0));
    REQUIRE(row.lr.tau_dev < qp_modea::modea_tau_anchor_mult * row.lr.anchor_expect);
  }

  /**
   * THE SYMMETRY LADDER (post-mortem of the kp444 tau-anchor abort, 2026-08-13). Three
   * fixtures of the SAME cell and the SAME 2x2x3 mesh, differing only in how the mesh is
   * reduced, so the W-fit class is (nearly) common and the tau deviation is attributable:
   *
   *    qe_lih223       12 k, 12 IBZ, no reduction        -> isym loop and trev branches DEAD
   *    qe_lih223_inv   12 k,  8 IBZ, time reversal only  -> trev branches LIVE, no D
   *    qe_lih223_sym   12 k,  6 IBZ, 2 q-symmetries      -> D-matrix external rotation LIVE
   *
   * [measured 2026-08-13, block-normalized semantics: 4.1824e-04 / 4.1824e-04 / 4.1827e-04
   *  at a common W-fit class of 3.8581e-03 (ratio 0.108). The sym row is not vacuous -- its
   *  census reads "D-rotation exercised on 4 of 6 (isym > 0, k) pairs, worst max|D - 1| = 2.0".
   *  The same rows read 6.3697 / 6.3697 / 6.3703e-04 under the per-element normalization
   *  retired the same day.]
   *
   * REPORTS only (hidden case, run explicitly by name).
   */
  TEST_CASE("qp_map_modea_sym_ladder", "[.nsym]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    const std::vector<std::string> names = {"qe_lih223", "qe_lih223_inv", "qe_lih223_sym"};
    struct lrow { std::string name; long nk, nkibz, nsym, ntrev; double dev, cls; };
    std::vector<lrow> rows;
    for (size_t i = 0; i < names.size(); ++i) {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, names[i]));
      app_log(1, "\n== ladder {}: nkpts = {} ({} IBZ), nqpts = {} ({} IBZ), q-symmetries = {}, "
                 "trev k-pairs = {} ==", names[i], mf->nkpts(), mf->nkpts_ibz(), mf->nqpts(),
              mf->nqpts_ibz(), mf->qsymms().size(), mf->nkpts_trev_pairs());
      auto row = run_map(mpi_context, mf, ft, "mode_b", "qpscf", 12, 1e-10, 1, 1e-6, "tau",
                         "_lad" + std::to_string(i), 0.0, -1.0, 1e-10, 0, 0.0, false);
      rows.push_back({names[i], mf->nkpts(), mf->nkpts_ibz(), long(mf->qsymms().size()),
                      mf->nkpts_trev_pairs(), row.lr.tau_dev, row.lr.anchor_expect});
    }
    app_log(1, "\n== SYMMETRY LADDER (lih223, mode_b, one outer iteration) ==");
    app_log(1, "  {:<16} {:>4} {:>6} {:>6} {:>6} {:>12} {:>12} {:>9}", "fixture", "nk",
            "nk_ibz", "nqsym", "ntrev", "tau dev", "W-fit class", "ratio");
    for (auto const &r : rows)
      app_log(1, "  {:<16} {:>4} {:>6} {:>6} {:>6} {:>12.4e} {:>12.4e} {:>9.3g}", r.name,
              r.nk, r.nkibz, r.nsym, r.ntrev, r.dev, r.cls, (r.cls > 0.0 ? r.dev / r.cls : 0.0));
    REQUIRE(rows.size() == names.size());
  }

  /**
   * Gate QM3-b(v) -- the two MEASUREMENTS that binding requirement 2 of the QM3 spec asks for,
   * on lih222/qpscf:
   *   (a) the DLR-precision notch: the fixture grid ("medium") and one notch higher ("high").
   *       Accuracy of the route-B evaluation is auxiliary-node COVERAGE of the retained
   *       support, so this is the knob that moves it -- not the DLR eps by itself.
   *   (b) the two qp_modea_wfit routes at the fixture grid: "tau" (the QM2-b tested chain,
   *       the default) and "nu" (the nu-space support-constrained LS, 3x better in the QM2
   *       toy probe at equal grid).
   * Each row REPORTS the triple (anchor, worst delta_i/class_i, gap). The DEFAULT IS NOT
   * FLIPPED here -- that is a reported decision, per the spec.
   */
  TEST_CASE("qp_map_modea_measurements", "[.modea_hold]") {
    using namespace qp_map_ab_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));

    struct meas { std::string name; ab_row row; };
    std::vector<meas> ms;
    {
      imag_axes_ft::IAFT ft_med(1000.0, 1.2, imag_axes_ft::dlr_basis, "medium");
      ms.push_back({"prec=medium (fixture), wfit=tau",
                    run_map(mpi_context, mf, ft_med, "mode_a", "qpscf", 12, 1e-10, 20, 1e-6,
                            "tau", "_m_tau")});
      ms.push_back({"prec=medium (fixture), wfit=nu ",
                    run_map(mpi_context, mf, ft_med, "mode_a", "qpscf", 12, 1e-10, 20, 1e-6,
                            "nu", "_m_nu")});
    }
    {
      imag_axes_ft::IAFT ft_hi(1000.0, 1.2, imag_axes_ft::dlr_basis, "high");
      ms.push_back({"prec=high   (one notch), wfit=tau",
                    run_map(mpi_context, mf, ft_hi, "mode_a", "qpscf", 12, 1e-10, 20, 1e-6,
                            "tau", "_h_tau")});
    }

    app_log(1, "\n== QM3-b(v) mode_a measurement table (lih222 / qpscf) ==");
    app_log(1, "{:<34} {:>11} {:>11} {:>8} {:>10} {:>12} {:>10}", "variant", "anchor",
            "rec class", "ratio", "d_i/cls", "gap (eV)", "nodes");
    for (auto const &m : ms) {
      auto const &L = m.row.lr;
      app_log(1, "{:<34} {:>11.4e} {:>11.4e} {:>8.3g} {:>10.3g} {:>12.4f} {:>4}/{:<5}",
              m.name, L.anchor, L.anchor_expect,
              (L.anchor_expect > 0.0 ? L.anchor / L.anchor_expect : 0.0), L.ratio_worst,
              m.row.gap_eV(), L.n_support, L.np_total);
    }
    app_log(1, "-- shifts relative to the fixture row (prec=medium, wfit=tau) --");
    for (size_t i = 1; i < ms.size(); ++i)
      app_log(1, "{:<34} d(gap) = {:+.4f} eV,  anchor x{:.3g},  reconstruction class x{:.3g}",
              ms[i].name, ms[i].row.gap_eV() - ms[0].row.gap_eV(),
              (ms[0].row.lr.anchor > 0.0 ? ms[i].row.lr.anchor / ms[0].row.lr.anchor : 0.0),
              (ms[0].row.lr.anchor_expect > 0.0
                   ? ms[i].row.lr.anchor_expect / ms[0].row.lr.anchor_expect : 0.0));

    for (auto const &m : ms) {
      REQUIRE(m.row.lr.anchor < qp_modea::modea_anchor_gate);
      REQUIRE(std::isfinite(m.row.gap_eV()));
      REQUIRE(m.row.gap_eV() > 0.0);
    }
  }

} // bdft_tests
