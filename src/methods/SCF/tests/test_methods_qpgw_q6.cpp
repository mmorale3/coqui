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

#include <filesystem>

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
 * Project 2 increment Q6 (notes/q6_diagnostics_closeout_spec.md) -- the diagnostics
 * closeout, C++ half. Three deliverables are gated here:
 *
 *   §1.2  THE COMPRESSIBILITY SEAM (PDF §8.4) -- dn/dmu by finite difference against the
 *         q -> 0, nu -> 0 charge response, tier by tier.
 *   §1.3  THE LINESHAPE METER (PDF §9) -- how much of Sigma^c the static map discards, read
 *         at the map stage where the MO-basis Sigma(iw) gather and the assembled V^xc
 *         coexist (qp_scf_common.cpp qp_approx; scf_common.hpp q6_lineshape()).
 *   §1.4  METER PERSISTENCE + the consolidated summary line -- the Q3 injection meters as
 *         h5 scalars under scf/iterN (fixing the Q5 trail's -1 sentinel) and one
 *         "[Q6] qpgw iteration summary" line per qp iteration.
 *
 * ==========================================================================================
 * SCOPE REDUCTION OF §1.2 -- the const_mu reachability survey, resolved FIRST (spec §2)
 * ==========================================================================================
 * VERDICT: const_mu DOES NOT REACH THE QP PATH.
 *
 *   * the knob is parsed in MBPT_drivers.cpp:299 (and :860) and forwarded ONLY to
 *     `scf_loop` -- the Dyson SCF (scf_driver.hpp:49, scf_driver.cpp:43 -> :192
 *     `update_G(..., mu, const_mu)`);
 *   * `qp_scf_loop` (scf_driver.hpp:68-72) HAS NO const_mu PARAMETER, and none of its four
 *     driver call sites (MBPT_drivers.cpp:626, 722, 832, 1210) passes one;
 *   * inside it, `update_mu` runs UNCONDITIONALLY every iteration
 *     (scf_driver.cpp:329 at init and :440 in the loop body).
 *
 * So there is no const-mu path in the qp loop, and building one is explicitly out of scope
 * (spec §2). Per the spec's own fallback the thermodynamic side is therefore NOT two
 * self-consistent const-mu runs. What it IS here, and the protocol for what it would be:
 *
 *   IMPLEMENTED (test level, no loop change): dn/dmu of the CONVERGED QP HAMILTONIAN, by
 *   central difference on the loop's own particle-number functional,
 *       N(mu) = compute_Nelec(mu, mf, E_ska, beta)      (qp_scf_common.cpp:1708-1727)
 *   which is the exact function `update_mu` inverts. This is the FIXED-Sigma dn/dmu, and it
 *   is the correct thermodynamic partner of an RPA-level (bubble) charge response: both are
 *   built from the same qp spectrum with no self-consistent relaxation.
 *
 *   RECORDED PROTOCOL for the full self-consistent number (NOT run -- it needs a const-mu qp
 *   path that does not exist):
 *     1. converge a qpgw run, read mu* from scf/iter{final}/mu;
 *     2. re-run twice at FIXED mu = mu* +- delta (delta ~ 1e-3 Ha) -- this is the step that
 *        needs the missing knob: qp_scf_loop would have to skip both update_mu calls and
 *        carry the injected mu through update_Dm / update_G / the map;
 *     3. Dm from each run's scf/iter{final}/Dm_skij, N = Tr[Dm S] summed with k_weight;
 *     4. dn/dmu = [N(mu*+delta) - N(mu*-delta)] / (2 delta), against the same response side.
 *   Implementing it is a qp-loop feature (a const_mu parameter threaded to both update_mu
 *   call sites), which the spec rules out for this increment.
 *
 * THE RESPONSE SIDE reuses the Q3/L2 readout machinery WITHOUT touching it: the readout
 * already reports eps_M(q_min, inu = 0) for BOTH tiers (scr_coulomb_t.cpp:811-930, exposed
 * by pol_eps_readout()), and the head charge response follows from it by the readout's own
 * convention
 *      eps^-1_head - 1 = v(q) chi(q, inu = 0)   =>   v.chi = 1/eps_M - 1,
 * i.e. exactly the "v(q).P00-class number" the spec asks for, in the same normalisation as
 * the logged CVV T-d meter (scr_coulomb_t.cpp:319-329). chi itself follows by dividing out
 * v_head(q) = 4 pi / (|q|^2 V), the readout's own `factor` (scr_coulomb_t.cpp:875).
 *
 * q -> 0 CAVEAT (spec §1.2, stated in the log below): lih222's q_min is FINITE. This is a
 * SEAM test -- the two sides must be finite, correctly signed and mutually consistent in
 * scale -- not a converged q -> 0 limit measurement.
 *
 * ==========================================================================================
 * HOW TO RUN (Catch2 v2 traps) -- MEASURED, do not "improve" the command
 * ==========================================================================================
 *
 *     KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 <build>/tests/bin/test_methods_qpgw_q6
 *
 * i.e. THE BARE BINARY, no filter: a "~[tag]" spec does NOT exclude hidden cases, it RUNS
 * them, and two positional test names are silently concatenated into one (unmatched) name.
 */

namespace bdft_tests {

  using namespace methods;

  namespace qpgw_q6_detail {

    constexpr double HA2EV = 27.211386245988;

    struct q6_row {
      std::string tag;
      double e_hf = 0.0;
      double e_homo = 0.0, e_lumo = 0.0;
      long final_iter = -1;
      double mu = 0.0;
      nda::array<ComplexType, 3> E_ska;
      // the L2/Q3 readout at q_min, inu = 0 (RPA tier, +ladder tier); -1 if never read
      double eps_rpa = -1.0, eps_lad = -1.0;
      // the Q3 injection meters as the accessors report them after the LAST update_w
      double lam_nu0 = -1.0, lam_max = -1.0, r_rt = -1.0, lad_ratio = -1.0;
      // the same four as READ BACK from scf/iter{final_iter} (gate Q6-d)
      double h5_lam_nu0 = -2.0, h5_lam_max = -2.0, h5_r_rt = -2.0, h5_lad_ratio = -2.0;
      bool h5_meters_present = false;
      // the Q6 lineshape meter of the LAST qp iteration (scf_common.hpp q6_lineshape())
      q6_lineshape_t ls;
      double gap_eV() const { return (e_lumo - e_homo) * HA2EV; }
    };

    /**
     * One qp_scf_loop on qe_lih222 with the Q3 ladder knobs -- the [qpgw] driver's wiring,
     * copied verbatim from the Q5 suite (test_methods_qpgw_q5.cpp:119-175) so the two suites
     * screen with the same objects. Everything Q6 adds is READ AFTER the loop returns.
     */
    inline q6_row run_qp(auto &mpi_context, std::shared_ptr<mf::MF> &mf,
                         imag_axes_ft::IAFT &ft, std::string const &tag,
                         std::string const &output,
                         std::string const &pol_mode, std::string const &inject,
                         nda::range window, int niter, double conv_tol,
                         int thc_prefactor = 12, double thc_tol = 1e-10,
                         const std::string &div = "ignore_g0") {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, div, output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", div);

      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*thc_prefactor, "", "incore", "",
                                                 output, thc_tol, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      qp_params_t qp_params("sc", "pade", 18, 0.0001, 1e-8, "qpscf");
      qp_params.qp_map = "ac_pade";

      solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd(), div);
      vtx.set_pol_vertex(pol_mode, "w0_prev", window, -1, 1e-8, -1.0, -1.0, -1.0, inject);
      if (vtx.pol_vertex_enabled()) scr_eri.set_vertex(&vtx);

      iter_scf::iter_scf_t iter_sol(iter_scf::damp_t(0.7));
      MBState mb_state(mpi_context, ft, output);
      q6_row row;
      row.tag = tag;
      row.e_hf = qp_scf_loop(mb_state, eri, ft, qp_params,
                             solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                             niter, false, conv_tol);
      mpi_context->comm.barrier();

      // the in-process meters, read from the objects that produced them
      auto [e_rpa, e_lad] = scr_eri.pol_eps_readout();
      row.eps_rpa = e_rpa;
      row.eps_lad = e_lad;
      row.lam_nu0 = scr_eri.pol_lambda_nu0();
      row.lam_max = scr_eri.pol_lambda_max();
      row.r_rt = scr_eri.pol_round_trip();
      row.lad_ratio = scr_eri.pol_ladder_ratio();
      row.ls = q6_lineshape();

      {
        h5::file file(output + ".mbpt.h5", 'r');
        auto scf_grp = h5::group(file).open_group("scf");
        h5::h5_read(scf_grp, "final_iter", row.final_iter);
        auto iter_grp = scf_grp.open_group("iter" + std::to_string(row.final_iter));
        nda::h5_read(iter_grp, "E_ska", row.E_ska);
        h5::h5_read(iter_grp, "mu", row.mu);
        // Q6 §1.4(a): the persisted meters (absent unless the ladder was injected)
        row.h5_meters_present = iter_grp.has_dataset("lambda_nu0");
        if (row.h5_meters_present) {
          h5::h5_read(iter_grp, "lambda_nu0", row.h5_lam_nu0);
          h5::h5_read(iter_grp, "lambda_max", row.h5_lam_max);
          h5::h5_read(iter_grp, "r_rt", row.h5_r_rt);
          h5::h5_read(iter_grp, "lad_ratio", row.h5_lad_ratio);
        }
      }
      mpi_context->comm.barrier();

      const long nkpts = row.E_ska.shape(1);
      const int homo = int(mf->nelec()/2 - 1);
      const int lumo = int(mf->nelec()/2);
      row.e_homo = -1e9;
      row.e_lumo = 1e9;
      for (long ik = 0; ik < nkpts; ++ik) {
        row.e_homo = std::max(row.e_homo, row.E_ska(0, ik, homo).real());
        row.e_lumo = std::min(row.e_lumo, row.E_ska(0, ik, lumo).real());
      }
      app_log(1, "qpgw_q6 [{}]: final iter {}, e_hf = {}, gap = {:.9f} eV, mu = {:.12f}",
              tag, row.final_iter, row.e_hf, row.gap_eV(), row.mu);
      return row;
    }

    /**
     * dN/dmu of the CONVERGED QP HAMILTONIAN by central difference on the loop's own
     * particle-number functional (see the scope-reduction note at the top of this file).
     * `E_ska` is the checkpointed qp spectrum; the shared-array copy is what compute_Nelec
     * takes.
     */
    inline double dn_dmu(auto &mpi_context, std::shared_ptr<mf::MF> &mf,
                         nda::array<ComplexType, 3> const &E_ska, double mu, double beta,
                         double delta) {
      using math::shm::make_shared_array;
      auto sE = make_shared_array<Array_view_3D_t>(
          *mpi_context, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd()});
      sE.win().fence();
      if (mpi_context->node_comm.root()) sE.local() = E_ska;
      sE.win().fence();
      mpi_context->comm.barrier();
      const double np = compute_Nelec(mu + delta, *mf, sE, beta);
      const double nm = compute_Nelec(mu - delta, *mf, sE, beta);
      return (np - nm) / (2.0 * delta);
    }

    /** |q_min|^2 over the IBZ transfers, with the readout's own rule (skip |q|^2 < 1e-12). */
    inline double q_min_abs2(std::shared_ptr<mf::MF> &mf) {
      double best = 1e300;
      for (long iq = 0; iq < mf->nqpts_ibz(); ++iq) {
        auto q = mf->Qpts_ibz(iq);
        const double a2 = q(0)*q(0) + q(1)*q(1) + q(2)*q(2);
        if (a2 < 1e-12) continue;
        best = std::min(best, a2);
      }
      return best;
    }

    inline void drop_chkpt(auto &mpi_context, std::string const &prefix) {
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove((prefix + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
    }

  } // qpgw_q6_detail

  /**
   * ==========================================================================================
   * GATE Q6-b -- THE COMPRESSIBILITY SEAM (spec §1.2, PDF §8.4), scope-reduced
   * ==========================================================================================
   * Two tiers on qe_lih222, 2 qp iterations each, ac_pade:
   *   RPA          : pol_vertex = "ladder", inject = "none"        (readout only)
   *   RPA + ladder : pol_vertex = "ladder", inject = "ladder_n2"   (the ladder is IN P)
   *
   * Per tier, both sides of the seam:
   *   (a) thermodynamic: dN/dmu of that tier's converged qp Hamiltonian (central difference,
   *       delta = 1e-3 Ha) -- see the scope-reduction note at the top of this file;
   *   (b) response: chi(q_min, inu = 0) = (1/eps_M - 1) / v_head, with eps_M the tier's own
   *       readout head and v_head = 4 pi / (|q_min|^2 V).
   *
   * The SIGN convention is fixed by the readout: eps^-1 - 1 = v.chi, so a SCREENING medium
   * (eps_M > 1) has chi < 0 while dN/dmu >= 0 always. The compressibility sum rule in this
   * convention is chi(q->0, 0) = -dn/dmu, so the two sides must carry OPPOSITE signs, and the
   * comparable pair is (-chi) against dN/dmu. Sign-gating "both the same sign" would gate a
   * convention error INTO the suite.
   */
  TEST_CASE("qpgw_q6_compressibility_seam_lih222", "[methods][qpgw][q6]") {
#ifndef ENABLE_DLR
    SUCCEED("qpgw_q6_compressibility_seam_lih222 skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace qpgw_q6_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));

    const auto win = nda::range(1, 3);
    const double delta = 1e-3;                 // spec §1.2: delta ~ 1e-3 Ha
    const std::string p_rpa = "qpgw_q6_b_rpa", p_lad = "qpgw_q6_b_lad";

    auto rpa = run_qp(mpi_context, mf, ft, "b_RPA", p_rpa, "ladder", "none", win, 2, 1e-6);
    auto lad = run_qp(mpi_context, mf, ft, "b_LAD", p_lad, "ladder", "ladder_n2", win, 2, 1e-6);

    const double q2 = q_min_abs2(mf);
    const double v_head = 4.0 * M_PI / (q2 * mf->volume());

    // (a) the thermodynamic side, per tier
    const double dndmu_rpa = dn_dmu(mpi_context, mf, rpa.E_ska, rpa.mu, ft.beta(), delta);
    const double dndmu_lad = dn_dmu(mpi_context, mf, lad.E_ska, lad.mu, ft.beta(), delta);

    // (b) the response side, per tier (each tier's OWN eps_M head)
    const double vchi_rpa = 1.0 / rpa.eps_rpa - 1.0;
    const double vchi_lad = 1.0 / lad.eps_lad - 1.0;
    const double chi_rpa = vchi_rpa / v_head;
    const double chi_lad = vchi_lad / v_head;

    // NON-VACUOUS CONTROL for the thermodynamic probe: the same central difference with mu
    // moved to the conduction-band edge, where states DO cross. If that is also zero the
    // probe is broken, and a zero at mu would mean nothing.
    const double dndmu_edge = dn_dmu(mpi_context, mf, rpa.E_ska, rpa.e_lumo, ft.beta(), delta);

    auto ratio = [](double num, double den) {
      return (std::abs(den) > 0.0) ? num / den : std::numeric_limits<double>::infinity();
    };
    app_log(1, "@@Q6B q_min^2 = {:.6e} a.u., V = {:.6f}, v_head = {:.6e} a.u. "
               "(SEAM test: q_min is FINITE, this is not a converged q -> 0 limit)",
            q2, mf->volume(), v_head);
    app_log(1, "@@Q6B tier RPA        : dN/dmu = {:.6e} 1/Ha, eps_M(q_min) = {:.6f}, "
               "v.chi = {:.6e}, chi = {:.6e} 1/Ha, ratio (-chi)/(dN/dmu) = {:.6e}",
            dndmu_rpa, rpa.eps_rpa, vchi_rpa, chi_rpa, ratio(-chi_rpa, dndmu_rpa));
    app_log(1, "@@Q6B tier RPA+ladder : dN/dmu = {:.6e} 1/Ha, eps_M(q_min) = {:.6f}, "
               "v.chi = {:.6e}, chi = {:.6e} 1/Ha, ratio (-chi)/(dN/dmu) = {:.6e}",
            dndmu_lad, lad.eps_lad, vchi_lad, chi_lad, ratio(-chi_lad, dndmu_lad));
    app_log(1, "@@Q6B control: dN/dmu at the conduction edge (mu = {:.6f}) = {:.6e} 1/Ha; "
               "gap = {:.6f} / {:.6f} eV (RPA / +ladder)",
            rpa.e_lumo, dndmu_edge, rpa.gap_eV(), lad.gap_eV());
    // The verdict is BRANCHED, not hard-coded: on a metal (svo on a TRIQS host) the ratio
    // becomes measurable and this line says so instead, so the log never asserts degeneracy
    // it did not measure.
    if (dndmu_rpa <= 1e-6 * dndmu_edge)
      app_log(1, "@@Q6B ratio verdict: DEGENERATE on this fixture -- the thermodynamic side "
                 "is 0 to machine precision for an insulator of gap {:.3f} eV at kT = "
                 "{:.4f} eV (exp(-gap/2kT) ~ e^-217). The O(1)-ratio gate of spec §1.2 is "
                 "therefore not installed; see the escape clause in §1.2 and the gate block "
                 "below.", rpa.gap_eV(), HA2EV / ft.beta());
    else
      app_log(1, "@@Q6B ratio verdict: MEASURABLE -- dN/dmu is {:.6e} 1/Ha at gap {:.3f} eV, "
                 "so (-chi)/(dN/dmu) = {:.6e} (RPA) / {:.6e} (+ladder) is a real number and "
                 "the O(1) expectation of spec §1.2 can be judged against it.",
              dndmu_rpa, rpa.gap_eV(), ratio(-chi_rpa, dndmu_rpa), ratio(-chi_lad, dndmu_lad));

    // MEASURED 2026-08-14 (qe_lih222, ac_pade, 2 qp iterations, DLR beta = 1000, wmax = 1.2,
    // C = [1,3), delta = 1e-3 Ha):
    //   q_min^2 = 4.973527e-01 a.u., V = 114.835126, v_head = 2.200243e-01 a.u.
    //   tier RPA        : dN/dmu = 0.000000e+00 1/Ha, eps_M(q_min) = 1.438859,
    //                     v.chi = -3.050051e-01, chi = -1.386234e+00 1/Ha,  ratio = inf
    //   tier RPA+ladder : dN/dmu = 0.000000e+00 1/Ha, eps_M(q_min) = 1.452027,
    //                     v.chi = -3.113075e-01, chi = -1.414878e+00 1/Ha,  ratio = inf
    //   control dN/dmu at the conduction edge = 1.732815e+02 1/Ha
    //   gap = 11.782989 (RPA) / 11.785136 (+ladder) eV
    //
    // READ THIS BEFORE THE GATES. The thermodynamic side is EXACTLY ZERO, and that is the
    // physics, not a broken probe: lih222 is an 11.78 eV insulator sampled at beta = 1000
    // a.u. (T = 27.2 meV), so exp(-gap / 2kT) ~ e^-217 -- dN/dmu underflows to 0.0 in double
    // precision. This is precisely the case the spec anticipates ("if the lih222 gap makes
    // both sides tiny, report and gate finiteness only -- an insulator's dn/dmu ~ 0 is itself
    // the physics"), so the O(1)-ratio gate of §1.2 is NOT REACHABLE on this fixture and is
    // NOT installed: the ratio is 0/0-degenerate (reported as inf) for both tiers. Reaching
    // it needs a metal (svo on a TRIQS host), the same environment the Q5-a leg is blocked on.
    //
    // What IS gated, in order of sharpness:
    //   (i)   the probe is LIVE -- the same central difference at the conduction-band edge is
    //         strictly positive and O(100)/Ha, so the zero at mu is the gap and nothing else;
    //   (ii)  the thermodynamic side is a legitimate dn/dmu: non-negative, and >= 6 orders
    //         below the control;
    //   (iii) the response side is finite, NEGATIVE (eps_M > 1 = a screening medium; the
    //         readout convention eps^-1 - 1 = v.chi makes chi < 0 the stable sign) and
    //         O(1)/Ha in absolute scale -- gated at 10x / 0.1x the measured numbers.
    REQUIRE(std::isfinite(dndmu_rpa));
    REQUIRE(std::isfinite(dndmu_lad));
    REQUIRE(std::isfinite(chi_rpa));
    REQUIRE(std::isfinite(chi_lad));
    REQUIRE(dndmu_edge > 1.0e2);                       // (i)   MEASURED 1.733e+02
    REQUIRE(dndmu_rpa >= 0.0);                         // (ii)
    REQUIRE(dndmu_lad >= 0.0);
    REQUIRE(dndmu_rpa <= 1e-6 * dndmu_edge);           //       MEASURED exactly 0.0
    REQUIRE(dndmu_lad <= 1e-6 * dndmu_edge);
    REQUIRE(rpa.eps_rpa > 1.0);                        // (iii) MEASURED 1.438859
    REQUIRE(lad.eps_lad > 1.0);                        //       MEASURED 1.452027
    REQUIRE(chi_rpa < 0.0);
    REQUIRE(chi_lad < 0.0);
    REQUIRE(std::abs(chi_rpa) > 0.13);                 //       MEASURED 1.386234
    REQUIRE(std::abs(chi_rpa) < 13.9);
    REQUIRE(std::abs(chi_lad) > 0.14);                 //       MEASURED 1.414878
    REQUIRE(std::abs(chi_lad) < 14.2);
    // the seam must not be measuring a degenerate q: q_min is finite and so is v_head
    REQUIRE(q2 > 1e-12);
    REQUIRE(std::isfinite(v_head));

    drop_chkpt(mpi_context, p_rpa);
    drop_chkpt(mpi_context, p_lad);
#endif
  }

  /**
   * ==========================================================================================
   * GATES Q6-c / Q6-d -- THE LINESHAPE METER (§1.3) and METER PERSISTENCE (§1.4)
   * ==========================================================================================
   * One qpgw run on qe_lih222 with the ladder INJECTED (ac_pade, mode-independent: the meter
   * is populated by qp_approx for every map).
   */
  TEST_CASE("qpgw_q6_lineshape_and_meter_persistence_lih222", "[methods][qpgw][q6]") {
#ifndef ENABLE_DLR
    SUCCEED("qpgw_q6_lineshape_and_meter_persistence_lih222 skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace qpgw_q6_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));

    const auto win = nda::range(1, 3);
    const std::string pfx = "qpgw_q6_cd";
    auto r = run_qp(mpi_context, mf, ft, "cd", pfx, "ladder", "ladder_n2", win, 2, 1e-6);

    app_log(1, "@@Q6C lineshape ({} gap-window diagonals, iw_0 = {:.6f}, iw_top = {:.6f} a.u.): "
               "ratio max/mean at iw_0 = {:.6e}/{:.6e}, at iw_top = {:.6e}/{:.6e}; "
               "ABS discard max/mean at iw_0 = {:.6e}/{:.6e}, at iw_top = {:.6e}/{:.6e} a.u.",
            r.ls.n_states, r.ls.w0, r.ls.wtop,
            r.ls.frac_w0_max, r.ls.frac_w0_mean, r.ls.frac_top_max, r.ls.frac_top_mean,
            r.ls.abs_w0_max, r.ls.abs_w0_mean, r.ls.abs_top_max, r.ls.abs_top_mean);
    app_log(1, "@@Q6D meters in-process lam_nu0/lam_max/r_rt/lad_ratio = "
               "{:.9f}/{:.9f}/{:.6e}/{:.9e}; from scf/iter{} (present = {}) = "
               "{:.9f}/{:.9f}/{:.6e}/{:.9e}",
            r.lam_nu0, r.lam_max, r.r_rt, r.lad_ratio, r.final_iter, r.h5_meters_present,
            r.h5_lam_nu0, r.h5_lam_max, r.h5_r_rt, r.h5_lad_ratio);

    // ========================================================================================
    // GATE Q6-c (§1.3) -- MEASURED 2026-08-14 (qe_lih222, ac_pade, ladder injected, 2 qp
    // iterations, DLR beta = 1000 / wmax = 1.2, C = [1,3)):
    //   32 gap-window diagonals (1 spin x 8 k x 4 states), iw_0 = 0.003142 a.u. (= pi/beta),
    //   iw_top = 5.978451 a.u.
    //   RATIO |Sigma^c - V^xc| / |Sigma^c|  max/mean:  iw_0   1.332471e+00 / 6.275748e-01
    //                                                  iw_top 3.416867e+00 / 1.696602e+00
    //   ABSOLUTE |Sigma^c - V^xc| (a.u.)    max/mean:  iw_0   2.910481e-02 / 1.479561e-02
    //                                                  iw_top 4.924333e-02 / 2.747048e-02
    //
    // *** FLAG -- THE SPEC'S EXPECTED ORDERING IS VIOLATED, AND THE GATE IS NOT INVERTED ***
    // Spec §1.3 predicts the discard fraction to be LARGER at iw_0 than at the top node
    // ("dynamics lives at low omega"). It is MEASURED SMALLER, in BOTH normalisations (ratio
    // 1.33 vs 3.42; absolute 2.91e-02 vs 4.92e-02 a.u.). This is not a defect of the meter --
    // it is what a static map anchored near omega = 0 must do:
    //   * V^xc from ac_pade is Sigma^c evaluated AT the qp energy eps_a - mu, which for
    //     gap-window states is close to zero, i.e. close to iw_0 = pi/beta = 3.1e-03 a.u.
    //     The map is therefore accurate BY CONSTRUCTION at the first node, and the residual
    //     there (2.9e-02 a.u. worst) is the genuine dynamical discard;
    //   * at iw_top = 5.98 a.u. the true Sigma^c(iw) has decayed toward zero while V^xc is a
    //     frequency-independent constant, so |Sigma^c - V^xc| -> |V^xc| and the RATIO
    //     additionally blows up on a vanishing denominator.
    // So the ordering the spec expected would hold for a map anchored in the TAIL, not for
    // one anchored at the qp energy.
    //
    // IT IS MAP-DEPENDENT, which is the confirmation of that reading. The SAME meter under
    // mode_a (the [qpgw] BSE suite's lih222 legs, test_methods_qpgw_bse, iteration 11 of the
    // ladder run) reports
    //     ratio max iw_0/iw_top = 1.331140e+01 / 2.797977e+00   <-- the SPEC's ordering
    //     ratio mean            = 2.269610e+00 / 2.355798e+00   <-- still marginally inverted
    //     ABS discard max       = 1.752900e-02 / 2.785449e-02 a.u.
    // with 21 of 64 evaluation energies clamped out of strip: mode_a evaluates Sigma^c on the
    // REAL axis at the qp energy, so its worst gap-window state is NOT the one ac_pade's
    // near-zero anchor makes accurate, and the low-frequency discard jumps by an order of
    // magnitude. The meter is therefore doing exactly its job -- separating maps by how much
    // dynamics they throw away at low omega -- and the spec's ordering is a property of the
    // map, not of the meter.
    //
    // Per spec §1.3 ("measure first, flag if violated rather than gate-invert") the ordering
    // is NOT gated in either direction; what is gated is that the numbers exist, are finite,
    // are non-vacuous, and sit at the measured scale for THIS (ac_pade) leg.
    REQUIRE(r.ls.n_states == 32);
    REQUIRE(std::abs(r.ls.w0 - M_PI / ft.beta()) < 1e-9);      // the FIRST fermionic node
    REQUIRE(r.ls.wtop > r.ls.w0);
    for (double v : {r.ls.frac_w0_max, r.ls.frac_w0_mean, r.ls.frac_top_max,
                     r.ls.frac_top_mean, r.ls.abs_w0_max, r.ls.abs_w0_mean,
                     r.ls.abs_top_max, r.ls.abs_top_mean}) {
      REQUIRE(std::isfinite(v));
      REQUIRE(v > 0.0);                                        // non-vacuous: never a zero fill
    }
    REQUIRE(r.ls.frac_w0_max >= r.ls.frac_w0_mean);            // max/mean consistency
    REQUIRE(r.ls.frac_top_max >= r.ls.frac_top_mean);
    REQUIRE(r.ls.abs_w0_max >= r.ls.abs_w0_mean);
    REQUIRE(r.ls.abs_top_max >= r.ls.abs_top_mean);
    // scale, at 10x / 0.1x the measured numbers (measure, then gate)
    REQUIRE(r.ls.frac_w0_max > 0.133);                         // MEASURED 1.332471
    REQUIRE(r.ls.frac_w0_max < 13.4);
    REQUIRE(r.ls.abs_w0_max > 2.9e-03);                        // MEASURED 2.910481e-02
    REQUIRE(r.ls.abs_w0_max < 2.92e-01);
    REQUIRE(r.ls.abs_top_max > 4.9e-03);                       // MEASURED 4.924333e-02
    REQUIRE(r.ls.abs_top_max < 4.93e-01);

    // ========================================================================================
    // GATE Q6-d (§1.4) -- meter persistence + the consolidated summary line.
    // MEASURED 2026-08-14 (same run):
    //   in-process  lam_nu0 = 0.114774757, lam_max = 0.114774757, r_rt = 7.825404e-08,
    //               lad_ratio = 3.239043027e-04
    //   scf/iter2   IDENTICAL to all printed digits (the h5 scalars are the same doubles).
    // Before §1.4(a) these datasets did not exist and python's Q5-b trail carried the
    // MISSING = -1 sentinel for lambda_nu0 forever. The round trip is gated BITWISE: the
    // write is a read of the accessors' own storage, so anything else is a bug.
    //
    // §1.4(b), "the fields parse + finite": the summary line is assembled from exactly three
    // sources, all checked here in-process -- q6_lineshape() (above), the scr_coulomb_t
    // accessors (below) and qp_modea::last_run(). Its mode_a-only fields (dmax(map inner),
    // inner-consist iters, the strip census) are inert under ac_pade and are exercised by
    // test_qp_map_ab, which runs full mode_a qp-scf loops through the SAME qp_scf_loop.
    REQUIRE(r.h5_meters_present);
    REQUIRE(r.h5_lam_nu0 == r.lam_nu0);                        // BITWISE round trip
    REQUIRE(r.h5_lam_max == r.lam_max);
    REQUIRE(r.h5_r_rt == r.r_rt);
    REQUIRE(r.h5_lad_ratio == r.lad_ratio);
    for (double v : {r.lam_nu0, r.lam_max, r.r_rt, r.lad_ratio}) {
      REQUIRE(std::isfinite(v));
      REQUIRE(v > 0.0);                                        // never the -1 "never measured"
    }
    REQUIRE(r.lam_max < 1.0);        // the eq-6 watchdog: rho(chi0 Xi) < 1 (MEASURED 0.1148)
    REQUIRE(r.r_rt < 1e-5);          // the transform class (MEASURED 7.83e-08)
    REQUIRE(r.lad_ratio < 1.0);      // the ladder is a CORRECTION (MEASURED 3.24e-04)

    drop_chkpt(mpi_context, pfx);
#endif
  }

} // bdft_tests
