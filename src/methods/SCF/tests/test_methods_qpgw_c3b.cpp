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
#include "methods/mb_state/mb_state.hpp"
#include "methods/embedding/dc_utilities.hpp"

/**
 * Project 2 increment Q4-C3b (notes/q4_c3b_orbital_ladder_dc_spec.md): the ORBITAL /
 * chi-convention local part of the lattice ladder -- the eq-7 bosonic double counting's
 * ladder half PROPER, as mandated by the R-Q4-2 AMENDMENT (the C3 THC-adjoint object is a
 * diagnostic that carries the upfold's ||B||^2 gain, not a DC contribution).
 *
 * Gates (spec section 3), measure-first-then-gate:
 *   G2  THE LEG PIN. The E-leg ONE-RUNG output of the kernel against a brute-force 4-leg
 *       contraction built from the chi0/K/E DEFINITIONS with explicit loops (no shared code
 *       with the kernel beyond its inputs). SYNTHETIC random U, fixed seed, 2 orbitals.
 *   G3  THE chi-CONVENTION PIN. The E-leg bubble against the same bubble taken in the
 *       G-space (U G U^dag) association -- eval_Pi_rpa_dc's index order, spin factor and
 *       conjugation from first principles -- and against its PH-sym tau route (the nu grid
 *       P_dc actually lives on).
 *   G4  SCALE SANITY. ||P^lad_loc,orb||_max vs ||bubble[G_loc]||_max on the PRODUCTION path
 *       (real projector, q-average, MBState + checkpoint). The amendment predicts O(1);
 *       the THC-adjoint object measured 3.5e5 on this fixture.
 * G1 (no-disturbance) is the commit-point suites; G5 (the consumer) is test_methods_embed.
 *
 * ==========================================================================================
 * HOW TO RUN (Catch2 v2 traps) -- MEASURED, do not "improve" the command
 * ==========================================================================================
 *
 *     KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 <build>/tests/bin/test_methods_qpgw_c3b
 *
 * i.e. THE BARE BINARY, no filter: a "~[tag]" spec does NOT exclude hidden cases, it RUNS
 * them, and two positional test names are silently concatenated into one (unmatched) name.
 */

namespace bdft_tests {

  using namespace methods;

  TEST_CASE("qpgw_c3b_orbital_ladder_dc_lih222", "[methods][qpgw][edmft][c3b]") {
#ifndef ENABLE_DLR
    SUCCEED("qpgw_c3b_orbital_ladder_dc_lih222 skipped: build has ENABLE_DLR=OFF.");
#else
    decltype(nda::range::all) all;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    auto [outdir, prefix] = utils::utest_filename("qe_lih222");
    const std::string wannier_file = outdir + "/lih_wan.h5";
    const std::string output = "qpgw_c3b";
    const std::string div = "ignore_g0";
    // THE LADDER WINDOW IS [0, 3), NOT the Q3/Q4 suites' [1, 3): the shipped lih_wan.h5
    // projector's band window is [0, 2), and the orbital ladder DC is only defined when the
    // projector's window lies INSIDE the ladder's (the producer skips with a warning
    // otherwise -- see accumulate_pi_lad_loc_orb). [1, 3) does NOT contain [0, 2).
    const nda::range window(0, 3);

    // ---- one short qpGW+BSE run with a bosonic projector and the ladder ON -------------
    // (the run_q4 recipe of test_methods_qpgw_edmft, trimmed to 3 iterations: every gate
    // below is an identity or a scale statement, none of them needs a converged loop)
    solvers::hf_t hf;
    solvers::gw_t gw(&ft, div, output);
    solvers::scr_coulomb_t scr_eri(&ft, "gw_edmft", div);
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*12, "", "incore", "",
                                               output, 1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);
    qp_params_t qp_params("sc", "pade", 18, 0.0001, 1e-8, "qpscf");
    qp_params.qp_map = "ac_pade";
    solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd(), div);
    vtx.set_pol_vertex("ladder", "w0_prev", window, -1, 1e-8, -1.0, -1.0, -1.0,
                       "ladder_n2");
    scr_eri.set_vertex(&vtx);
    iter_scf::iter_scf_t iter_sol(iter_scf::damp_t(0.7));
    MBState mb_state(ft, output, mf, wannier_file, true);
    REQUIRE(mb_state.proj_boson.has_value());
    qp_scf_loop(mb_state, eri, ft, qp_params,
                solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol, 3, false, 1e-6);
    mpi_context->comm.barrier();

    // the extra screening step (run_q4's recipe): puts G, W0bar and the ladder DC objects
    // on the SAME state, which is what the gates below read
    {
      using math::shm::make_shared_array;
      double mu = update_mu(0.0, *mf, mb_state.sE_ska.value(), ft.beta());
      mb_state.sG_tskij.emplace(make_shared_array<Array_view_5D_t>(
          *mpi_context, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
      update_G(mb_state.sG_tskij.value(), mb_state.sMO_skia.value(),
               mb_state.sE_ska.value(), mu, ft);
      scr_eri.update_w(mb_state, thc, -1);
    }
    auto *pv = scr_eri.pol_vertex_instance();
    REQUIRE(pv != nullptr);

    // ====================================================================================
    // G2 / G3 -- the leg pin and the chi-convention pin, on SYNTHETIC legs
    // ====================================================================================
    // A synthetic U (fixed-seed LCG, 2 impurity orbitals) is the sharper probe: it has no
    // structure the kernel and the reference could share, and it is nowhere near unitary,
    // so a transposed / mis-conjugated leg cannot cancel.
    const long ns = mf->nspin(), nk = mf->nkpts(), nc = window.size();
    const long norb = 2;
    nda::array<ComplexType, 4> U_syn(ns, nk, norb, nc);
    {
      unsigned long s = 20260814ul;
      auto rnd = [&s]() {
        s = s * 6364136223846793005ul + 1442695040888963407ul;
        return double((s >> 33) % 100000ul) / 100000.0 - 0.5;
      };
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nk; ++ik)
          for (long m = 0; m < norb; ++m)
            for (long a = 0; a < nc; ++a) U_syn(is, ik, m, a) = ComplexType(rnd(), rnd());
    }
    auto dl = pv->ladder_loc_gate(mb_state, thc, U_syn);
    app_log(1, "@@C3B G2 one-rung (E legs) vs the brute-force 4-leg contraction: rel = "
               "{:.3e} at scale {:.3e} ({} nu nodes, {} orbitals)",
            dl.onerung_resid, dl.onerung_scale, dl.nnu_checked, dl.norb);
    app_log(1, "@@C3B G3 E-leg bubble vs the G-space bubble: rel = {:.3e} at scale {:.3e}; "
               "vs the PH-sym tau route (eval_Pi_rpa_dc's own grid): raw = {:.3e}, after "
               "symmetrizing = {:.3e} (the bubble's tau asymmetry = {:.3e}); "
               "P^lad_loc nu-mirror resid = {:.3e}",
            dl.bub_resid_w, dl.bub_scale, dl.bub_resid_phsym, dl.bub_resid_phsym_sym,
            dl.bub_tau_asym, dl.loc_ph_sym);
    REQUIRE_FALSE(dl.sym_active);
    // non-vacuity first: the references are not zero
    REQUIRE(dl.onerung_scale > 0.0);
    REQUIRE(dl.bub_scale > 0.0);
    REQUIRE(dl.lad_loc_max > 0.0);
    // G2: an algebraic rearrangement of the same sums => machine class, no fitted scale.
    // MEASURED (2026-08-14, qe_lih222, C = [0,3), 3 qpGW iterations, synthetic U):
    // 5.3e-15 relative at scale 7.87e-03 (production legs: 1.2e-14). The gate is the
    // spec's own machine-class bound.
    REQUIRE(dl.onerung_resid < 1e-12);
    // G3: likewise for the bubble in the eval_Pi_rpa_dc pair pack abcd = (m, n, m', n').
    // MEASURED 2.8e-15 at scale 3.71e-01 (production legs: 1.2e-15 at 1.10e-03, which is
    // also the scale of bubble[G_loc] measured below -- they differ only by the off-site
    // terms the spec flags).
    REQUIRE(dl.bub_resid_w < 1e-12);
    // ... and the same object on the PH-sym half grid P_dc lives on. MEASURED: the raw
    // comparison does NOT close (see the @@C3B G3 line) and the reason is exact and
    // characterized, not a bug: tau -> beta - tau maps Pi_{(mn),(m'n')} onto a transposed
    // element, so the BARE local bubble is not PH-symmetric element by element, and a
    // half-grid transform can only ever see its symmetric part. The pin is therefore on
    // mirror-EXTENDED object (IAFT::tau_to_w_PHsym folds Twt(iw,it) + Twt(iw,nt-1-it) onto
    // the first tau half) plus the asymmetry meter that makes the statement non-vacuous.
    // MEASURED: raw 2.24e-01, mirror-extended reference 3.8e-16, tau asymmetry 3.36e-01
    // (production legs: 2.0e-16 and 9.58e-01).
    REQUIRE(dl.bub_tau_asym > 1e-6);            // non-vacuous: there IS an asymmetric part
    REQUIRE(dl.bub_resid_phsym_sym < 1e-12);    // ... and it is exactly what differs

    // ---- the same two pins on the PRODUCTION legs --------------------------------------
    // (the real projector, extracted exactly as accumulate_pi_lad_loc_orb does: C_skIai on
    // the ladder window's columns, zero on the bands the projector does not span). This
    // also measures the nu-mirror residual of the object that actually reaches P_dc.
    nda::array<ComplexType, 4> U_prod(ns, nk, mb_state.proj_boson.value().nImpOrbs(), nc);
    {
      auto &pb = mb_state.proj_boson.value();
      auto C = pb.proj_fermi().C_skIai();
      auto const &W_rng = pb.W_rng()[0];
      const long off = W_rng.first() - window.first();
      U_prod() = ComplexType(0.0);
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nk; ++ik)
          for (long m = 0; m < pb.nImpOrbs(); ++m)
            for (long j = 0; j < W_rng.size(); ++j) U_prod(is, ik, m, off + j) = C(is, ik, 0, m, j);
    }
    auto dp = pv->ladder_loc_gate(mb_state, thc, U_prod);
    app_log(1, "@@C3B PROD legs: G2 = {:.3e} (scale {:.3e}), G3 = {:.3e} (scale {:.3e}), "
               "PH-sym tau route after symmetrizing = {:.3e} (raw {:.3e}, bubble tau "
               "asymmetry {:.3e}); P^lad_loc nu-mirror resid = {:.3e}",
            dp.onerung_resid, dp.onerung_scale, dp.bub_resid_w, dp.bub_scale,
            dp.bub_resid_phsym_sym, dp.bub_resid_phsym, dp.bub_tau_asym, dp.loc_ph_sym);
    REQUIRE(dp.onerung_scale > 0.0);
    REQUIRE(dp.bub_scale > 0.0);
    REQUIRE(dp.onerung_resid < 1e-12);
    REQUIRE(dp.bub_resid_w < 1e-12);
    REQUIRE(dp.bub_resid_phsym_sym < 1e-12);
    // REPORTED, NOT GATED: loc_ph_sym is the transform licence of the object P_dc receives.
    // Unlike the D-leg ladder (ph_sym_resid 4.28e-09, ladder_whalf_gate) the E-leg one need
    // not be nu-mirror symmetric element by element -- same reason the bare local bubble is
    // not -- and P_dc's own bubble half is built on the tau HALF grid, i.e. it carries only
    // the mirror-extended part. Both halves of P_dc therefore sit on the same nu nodes but
    // treat the asymmetric content differently. Flagged for the spec author; nothing here
    // depends on it, and no symmetrization is applied to the shipped object.
    REQUIRE(dp.loc_ph_sym >= 0.0);

    // ====================================================================================
    // G4 -- the scale statement of the R-Q4-2 AMENDMENT, on the PRODUCTION path
    // ====================================================================================
    REQUIRE(mb_state.sPi_lad_loc_orb_wabcd.has_value());     // the producer ran
    REQUIRE(mb_state.sPi_lad_loc_wabcd.has_value());         // ... next to the diagnostic
    const double orb_max = scr_eri.pol_lad_loc_orb_max();
    const double thc_max = scr_eri.pol_lad_loc_max();
    REQUIRE(orb_max > 0.0);

    // bubble[G_loc] through the Q4 machinery: the DC bubble the ladder half is added to
    double bub_max = 0.0;
    {
      auto &pb = mb_state.proj_boson.value();
      auto G_tsIab = pb.proj_fermi().downfold_loc<false>(mb_state.sG_tskij.value(),
                                                         "Gloc for the C3b scale gate");
      auto sPi = eval_Pi_rpa_dc<true>(*mpi_context, G_tsIab, ft, false);
      for (auto const &v : sPi.local()) bub_max = std::max(bub_max, std::abs(v));
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove("pi_rpa_loc_debug.h5");   // the eval_Pi_rpa_dc wart
    }
    const double ratio = orb_max / std::max(bub_max, 1e-300);
    app_log(1, "@@C3B G4 scale: ||P^lad_loc,orb||_max = {:.6e} vs ||bubble[Gloc]||_max = "
               "{:.6e} (ratio {:.4e}); the C3 THC-adjoint object on the same state = "
               "{:.6e} (ratio {:.4e})",
            orb_max, bub_max, ratio, thc_max, thc_max / std::max(bub_max, 1e-300));
    REQUIRE(bub_max > 0.0);
    // MEASURED (2026-08-14): ||P^lad_loc,orb||_max = 1.192e-06 vs ||bubble[Gloc]||_max =
    // 1.094e-03, ratio 1.09e-03 -- an O(1)-class object (a small correction), where the
    // C3 THC-adjoint object on the SAME state is 7.28e+06, i.e. a ratio of 6.65e+09.
    // The amendment's failure mode was 9-10 ORDERS; this is the coarse but decisive guard.
    REQUIRE(ratio < 10.0);
    // ... and the diagnostic object is still the OLD, rejected scale (non-vacuity of the
    // whole increment: the two objects are genuinely different, by many orders)
    REQUIRE(thc_max / std::max(bub_max, 1e-300) > 1e3);

    // the DC-ready object must also reach the checkpoint (how a separate-process
    // downfold_2e and python's DC assembly see it at all)
    long orb_h5_iters = 0, final_iter = -1;
    {
      h5::file file(output + ".mbpt.h5", 'r');
      auto scf_grp = h5::group(file).open_group("scf");
      h5::h5_read(scf_grp, "final_iter", final_iter);
      for (long it = 0; it <= final_iter; ++it) {
        std::string gn = "iter" + std::to_string(it);
        if (scf_grp.has_subgroup(gn) and
            scf_grp.open_group(gn).has_dataset("pi_lad_loc_orb_wabcd"))
          ++orb_h5_iters;
      }
    }
    mpi_context->comm.barrier();
    app_log(1, "@@C3B checkpoint: scf/iterN groups carrying pi_lad_loc_orb_wabcd = {} "
               "(final_iter = {})", orb_h5_iters, final_iter);
    REQUIRE(orb_h5_iters > 0);

    mpi_context->comm.barrier();
    if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
    mpi_context->comm.barrier();
#endif
  }

} // bdft_tests
