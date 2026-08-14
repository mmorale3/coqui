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
 * Project 2 increment Q5 (notes/q5_option2_outer_loop_spec.md): the Option-2 outer loop,
 * C++ half -- the re-QP-ization step. The qpGW driver gained
 * `greens_func_source` / `greens_func_iteration`; when set, ITERATION 1 of qp_scf_loop
 * consumes the EXTERNAL G of that checkpoint group (its density matrix drives the HF stage,
 * eq 3's Sigma^H[rho_latt]; update_w and the Sigma^GW build screen with the same G) and
 * iterations >= 2 revert to the loop's own analytic QP G.
 *
 * Gates (spec §3), measure-first-then-gate throughout:
 *   Q5-g1  SOURCE-SWAP IDENTITY -- the falsifier for the whole injection. Feed the loop the
 *          very G it would have built itself and require the iteration to come out
 *          unchanged. Any deviation means the injection path differs from the analytic path
 *          (mu handling, Dm, or a stale field).
 *   Q5-g2  RESTART COMPOSITION -- one niter = 6 run == 3 x (niter = 2, restart), ladder ON.
 *          This IS the C = empty-set Option-1-vs-Option-2 insensitivity statement (PDF
 *          §5.1 + §7: with zero impurity corrections the outer loop is continued qp
 *          iteration), and it is what licenses running the qpGW stage inside every cycle.
 *
 * ==========================================================================================
 * HOW TO RUN (Catch2 v2 traps) -- MEASURED, do not "improve" the command
 * ==========================================================================================
 *
 *     KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 <build>/tests/bin/test_methods_qpgw_q5
 *
 * i.e. THE BARE BINARY, no filter: a "~[tag]" spec does NOT exclude hidden cases, it RUNS
 * them, and two positional test names are silently concatenated into one (unmatched) name.
 *
 * ==========================================================================================
 * ENVIRONMENT-BLOCKED LEG (spec §5, recorded -- NOT gated here)
 * ==========================================================================================
 * Q5-a (sensitivity near the transition) needs a correlated metal + QMC, i.e. a TRIQS host.
 * Protocol for rusty, svo fixtures, CT-SEG mode (a):
 *
 *   1. Build the GW checkpoint:   coqui.run_gw(..., prefix="svo", niter=...).
 *   2. Option-1 trail:  run_gw_edmft(params | {"lattice_solver": "qpgw",
 *                                              "outer_loop": "option1", "niter": 8}).
 *   3. Option-2 trail:  the same params with "outer_loop": "option2"
 *                       (+ "outer_qpgw_niter": 1, iter_alg.mixing 0.3 -- PDF §7).
 *   4. Compare the per-cycle H_eff trails (gap(H_eff) of the Q5-b log block):
 *        - metallic side  => the two trails agree = the insensitivity statement;
 *        - near the transition => they separate = the restored gap -> Drude-loss ->
 *          U-growth feedback of PDF §5.3.
 *   The full C = empty-set option2 end-to-end run command is recorded in
 *   src/python/dmft/tests/test_q5_outer_loop.py.
 */

namespace bdft_tests {

  using namespace methods;

  namespace qpgw_q5_detail {

    constexpr double HA2EV = 27.211386245988;

    struct q5_row {
      std::string tag;
      double e_hf = 0.0;                 // = e_1e + e_hf, the qp_scf_loop return value
      double e_homo = 0.0, e_lumo = 0.0;
      long final_iter = -1;
      nda::array<ComplexType, 3> E_ska;
      nda::array<ComplexType, 4> Dm_skij;
      double mu = 0.0;
      double gap_eV() const { return (e_lumo - e_homo) * HA2EV; }
    };

    /**
     * One qp_scf_loop on qe_lih222, with the Q3 ladder knobs and the Q5 external-G knobs.
     * The THC/solver wiring is the [qpgw] driver's, copied from the Q3 suite's run_qpgw
     * (test_methods_qpgw_bse.cpp) so the two suites screen with the same objects.
     *
     * NOTE ON mu: qp_params_t's default mu_update_alg is "bisection" (qp_params_t.h:49) and
     * update_mu_bisection_impl returns old_mu UNCHANGED when |N(old_mu) - N| < mu_tol
     * (scf_common.hpp:81-84). That is what makes Q5-g1 a sharp gate: the restart-init mu is
     * bitwise the checkpointed mu, so read_greens_function's on-the-fly analytic G (built
     * from the checkpointed MO_skia/E_ska/mu, scf_common.cpp:440-459) is bitwise the G the
     * loop's own update_G would build in iteration 1.
     *
     * The checkpoint is NOT removed here -- Q5-g1/g2 chain restarts onto it. Every TEST_CASE
     * cleans up its own prefixes at the end.
     */
    inline q5_row run_qp(auto &mpi_context, std::shared_ptr<mf::MF> &mf,
                         imag_axes_ft::IAFT &ft, std::string const &tag,
                         std::string const &output,
                         std::string const &pol_mode, std::string const &inject,
                         nda::range window, int niter, bool restart, double conv_tol,
                         std::string const &gf_grp = "", long gf_iter = -1,
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

      // the driver's [qpgw] wiring: a pure knob carrier, attached to scr_eri only. It must
      // outlive qp_scf_loop -- same stack frame.
      solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd(), div);
      vtx.set_pol_vertex(pol_mode, "w0_prev", window, -1, 1e-8, -1.0, -1.0, -1.0, inject);
      if (vtx.pol_vertex_enabled()) scr_eri.set_vertex(&vtx);

      iter_scf::iter_scf_t iter_sol(iter_scf::damp_t(0.7));
      MBState mb_state(mpi_context, ft, output);
      q5_row row;
      row.tag = tag;
      row.e_hf = qp_scf_loop(mb_state, eri, ft, qp_params,
                             solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                             niter, restart, conv_tol, gf_grp, gf_iter);
      mpi_context->comm.barrier();

      {
        h5::file file(output + ".mbpt.h5", 'r');
        auto scf_grp = h5::group(file).open_group("scf");
        h5::h5_read(scf_grp, "final_iter", row.final_iter);
        auto iter_grp = scf_grp.open_group("iter" + std::to_string(row.final_iter));
        nda::h5_read(iter_grp, "E_ska", row.E_ska);
        nda::h5_read(iter_grp, "Dm_skij", row.Dm_skij);
        h5::h5_read(iter_grp, "mu", row.mu);
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
      app_log(1, "qpgw_q5 [{}]: final iter {}, e_hf = {}, gap = {:.9f} eV, mu = {:.12f}",
              tag, row.final_iter, row.e_hf, row.gap_eV(), row.mu);
      return row;
    }

    inline double max_abs_diff(nda::array<ComplexType, 3> const &a,
                               nda::array<ComplexType, 3> const &b) {
      REQUIRE(a.shape() == b.shape());
      double d = 0.0;
      for (long i = 0; i < a.size(); ++i)
        d = std::max(d, std::abs(a.data()[i] - b.data()[i]));
      return d;
    }

    inline double max_abs_diff(nda::array<ComplexType, 4> const &a,
                               nda::array<ComplexType, 4> const &b) {
      REQUIRE(a.shape() == b.shape());
      double d = 0.0;
      for (long i = 0; i < a.size(); ++i)
        d = std::max(d, std::abs(a.data()[i] - b.data()[i]));
      return d;
    }

    struct g1_probe {
      double dm_floor = -1.0;      // |C f C^dag  -  (-G(tau -> beta))|_max
      double g_read_resid = -1.0;  // |read_greens_function  -  update_G|_max
      double e_span = 0.0;         // max |E_ska - mu|, the spectral range the IAFT must span
    };

    /**
     * Decompose the Q5-g1 deviation BEFORE gating it. Two independent probes on the
     * checkpointed qp solution (MO_skia, E_ska, mu) of `output`:
     *
     *  (1) g_read_resid -- does the injected G equal the analytic G bit for bit? The
     *      production read path (read_greens_function, scf_common.cpp:440-459) finds no
     *      G_tskij dataset in a qp checkpoint and rebuilds G from (MO_skia, E_ska, mu) with
     *      the SAME update_G the loop calls. This must be exactly 0.0; if it is not, the
     *      injection is reading a different state than the loop would build.
     *
     *  (2) dm_floor -- the density-matrix CONVENTION difference. The qp loop uses
     *      update_Dm = C f(E) C^dag; an external G supports only the Dyson convention
     *      Dm = -G(tau -> beta) (simple_dyson.cpp:143-145). Analytically the two are the
     *      same matrix (G(beta) = -C f C^dag exactly, from update_G's compute_G0), so their
     *      difference is PURELY the IAFT tau -> beta extrapolation error, i.e. the DLR
     *      leakage of a spectrum whose poles reach e_span >> wmax. That is the floor the
     *      Q5-g1 deviation must sit under.
     */
    inline g1_probe probe_analytic_g(auto &mpi_context, std::shared_ptr<mf::MF> &mf,
                                     imag_axes_ft::IAFT &ft, std::string const &output,
                                     bool with_read_probe = true) {
      using math::shm::make_shared_array;
      g1_probe out;
      auto sMO_skia = make_shared_array<Array_view_4D_t>(
          *mpi_context, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});
      auto sE_ska = make_shared_array<Array_view_3D_t>(
          *mpi_context, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd()});
      auto sDm_a = make_shared_array<Array_view_4D_t>(
          *mpi_context, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});
      auto sDm_b = make_shared_array<Array_view_4D_t>(
          *mpi_context, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});
      auto sG_tskij = make_shared_array<Array_view_5D_t>(
          *mpi_context, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()});
      double mu = 0.0;
      long final_iter = -1;

      sMO_skia.win().fence();
      sE_ska.win().fence();
      if (mpi_context->node_comm.root()) {
        h5::file file(output + ".mbpt.h5", 'r');
        auto scf_grp = h5::group(file).open_group("scf");
        h5::h5_read(scf_grp, "final_iter", final_iter);
        auto iter_grp = scf_grp.open_group("iter" + std::to_string(final_iter));
        auto MO_loc = sMO_skia.local();
        auto E_loc = sE_ska.local();
        nda::h5_read(iter_grp, "MO_skia", MO_loc);
        nda::h5_read(iter_grp, "E_ska", E_loc);
        h5::h5_read(iter_grp, "mu", mu);
      }
      sMO_skia.win().fence();
      sE_ska.win().fence();
      mpi_context->comm.broadcast_n(&mu, 1, 0);
      mpi_context->comm.broadcast_n(&final_iter, 1, 0);
      mpi_context->comm.barrier();

      auto E = sE_ska.local();
      for (long i = 0; i < E.size(); ++i)
        out.e_span = std::max(out.e_span, std::abs(E.data()[i].real() - mu));

      update_G(sG_tskij, sMO_skia, sE_ska, mu, ft);
      update_Dm(sDm_a, sMO_skia, sE_ska, mu, ft.beta());
      sDm_b.win().fence();
      if (mpi_context->node_comm.root()) {
        auto Dm = sDm_b.local();
        ft.tau_to_beta(sG_tskij.local(), Dm);
        Dm *= -1;
      }
      sDm_b.win().fence();
      mpi_context->comm.barrier();

      out.dm_floor = 0.0;
      auto A = sDm_a.local();
      auto B = sDm_b.local();
      for (long i = 0; i < A.size(); ++i)
        out.dm_floor = std::max(out.dm_floor, std::abs(A.data()[i] - B.data()[i]));

      if (with_read_probe) {
        auto sG_read = read_greens_function(*mpi_context, mf.get(), output + ".mbpt.h5",
                                            final_iter, "scf");
        out.g_read_resid = 0.0;
        auto R = sG_read.local();
        auto L = sG_tskij.local();
        for (long i = 0; i < L.size(); ++i)
          out.g_read_resid = std::max(out.g_read_resid, std::abs(L.data()[i] - R.data()[i]));
        mpi_context->comm.barrier();
      }
      return out;
    }

    inline void clone_chkpt(auto &mpi_context, std::string const &from, std::string const &to) {
      if (mpi_context->comm.root()) {
        std::filesystem::remove(to + ".mbpt.h5");
        std::filesystem::copy_file(from + ".mbpt.h5", to + ".mbpt.h5");
      }
      mpi_context->comm.barrier();
    }

    inline void drop_chkpt(auto &mpi_context, std::string const &prefix) {
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove((prefix + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
    }

  } // qpgw_q5_detail

  /**
   * ==========================================================================================
   * GATE Q5-g1 -- THE SOURCE-SWAP IDENTITY (the falsifier for the whole injection)
   * ==========================================================================================
   * Run A: 2 qp iterations, ac_pade, qe_lih222, from scratch -> checkpoint "scf/iter2"
   *        (Dm_skij / Heff_skij / MO_skia / E_ska / mu; chkpt_utils.cpp:108-130 -- a qp
   *        checkpoint carries NO G_tskij, so read_greens_function takes its second branch
   *        and REBUILDS the analytic G from (MO_skia, E_ska, mu), scf_common.cpp:440-459).
   * Run B: restart onto a private copy of that checkpoint WITH greens_func_source = "scf".
   * Run C: restart onto a private copy WITHOUT the knob (the plain continuation).
   *
   * G_ext is by construction the G that iteration 1 would have built itself, so B and C must
   * agree. The one place they can legitimately differ is the DENSITY MATRIX convention:
   * C uses update_Dm = C f(E) C^dag, B uses the Dyson convention Dm = -G(tau -> beta)
   * (simple_dyson.cpp:143-145), the only one a GENERAL external G supports. Analytically
   * the two are the same matrix; numerically they differ by the IAFT tau -> beta
   * extrapolation. probe_analytic_g() measures BOTH halves of that story before anything is
   * gated -- the G path (which must be bitwise) and the Dm floor (which must account for
   * everything that is left) -- so the gate below is a decomposition, not a tolerance.
   */
  TEST_CASE("qpgw_q5_source_swap_lih222", "[methods][qpgw][q5]") {
#ifndef ENABLE_DLR
    SUCCEED("qpgw_q5_source_swap_lih222 skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace qpgw_q5_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));

    const std::string pa = "qpgw_q5_g1_a", pb = "qpgw_q5_g1_b", pc = "qpgw_q5_g1_c";

    // ---- run A: the source checkpoint -------------------------------------------------
    auto rA = run_qp(mpi_context, mf, ft, "g1_A", pa, "none", "none", nda::range(0, 0),
                     2, false, 1e-10);
    REQUIRE(rA.final_iter == 2);

    // ---- decompose the deviation BEFORE gating it (measure first) ----------------------
    auto probe = probe_analytic_g(mpi_context, mf, ft, pa);
    const double dm_floor = probe.dm_floor;
    // ... and the SAME Dm probe on a DLR window wide enough to actually span the spectrum:
    // if the floor is the tau -> beta extrapolation of an out-of-window pole set (and not a
    // code path), widening wmax must collapse it.
    imag_axes_ft::IAFT ft_wide(1000.0, 4.0 * std::ceil(probe.e_span), imag_axes_ft::dlr_basis);
    const double dm_floor_wide = probe_analytic_g(mpi_context, mf, ft_wide, pa, false).dm_floor;
    app_log(1, "@@Q5G1 |read_greens_function - update_G|_max = {:.3e}", probe.g_read_resid);
    app_log(1, "@@Q5G1 Dm convention floor |C f C^dag - (-G(beta))|_max = {:.3e} "
               "(wmax = 1.2, max|E - mu| = {:.3f}); same probe at wmax = {:.1f}: {:.3e}",
            dm_floor, probe.e_span, 4.0 * std::ceil(probe.e_span), dm_floor_wide);

    clone_chkpt(mpi_context, pa, pb);
    clone_chkpt(mpi_context, pa, pc);

    // ---- run B: restart + external G ---------------------------------------------------
    auto rB = run_qp(mpi_context, mf, ft, "g1_B(ext G)", pb, "none", "none", nda::range(0, 0),
                     1, true, 1e-10, "scf", -1);
    // ---- run C: plain restart ----------------------------------------------------------
    auto rC = run_qp(mpi_context, mf, ft, "g1_C(plain)", pc, "none", "none", nda::range(0, 0),
                     1, true, 1e-10);

    REQUIRE(rB.final_iter == 3);
    REQUIRE(rC.final_iter == 3);
    // the restart-init mu must be bitwise the checkpointed one (bisection early-exit) --
    // otherwise the external G is built at a DIFFERENT mu than the loop's and the gate below
    // would be measuring mu drift instead of the injection.
    REQUIRE(rB.mu == rC.mu);

    const double d_e   = std::abs(rB.e_hf - rC.e_hf);
    const double d_E   = max_abs_diff(rB.E_ska, rC.E_ska);
    const double d_Dm  = max_abs_diff(rB.Dm_skij, rC.Dm_skij);
    const double d_gap = std::abs(rB.gap_eV() - rC.gap_eV());
    app_log(1, "@@Q5G1 external-G vs plain restart (iteration 3): d_e_hf = {:.3e}, "
               "|dE_ska|_max = {:.3e}, |dDm|_max = {:.3e}, d_gap = {:.3e} eV "
               "(Dm floor = {:.3e})", d_e, d_E, d_Dm, d_gap, dm_floor);

    // MEASURED 2026-08-14 (qe_lih222, ac_pade, 2+1 iterations, DLR beta = 1000, wmax = 1.2):
    //   |read_greens_function - update_G|_max = 0.000e+00   <-- BITWISE, see below
    //   max |E_ska - mu|                      = 2.220 Ha  (against wmax = 1.2)
    //   Dm convention floor (wmax =  1.2)     = 3.056e-06
    //   Dm convention floor (wmax = 12.0)     = 1.355e-12   <-- 2.3e6 x smaller
    //   d_e_hf       = 3.649e-07
    //   |dE_ska|_max = 5.476e-07
    //   |dDm|_max    = 2.139e-07
    //   d_gap        = 7.287e-06 eV
    //   iteration step |e_hf(B) - e_hf(A)| = 2.044e-03  (668 x the floor: non-vacuous)
    //
    // READ THE DECOMPOSITION, not just the numbers. The G path is EXACT: the injected G is
    // bitwise the G iteration 1 would have built (g_read_resid == 0.0), because the restart
    // mu is bitwise the checkpointed mu (bisection early-exit) and read_greens_function
    // rebuilds with the loop's own update_G. So the ONLY difference between run B and run C
    // is the density-matrix CONVENTION -- C f(E) C^dag versus -G(tau -> beta) -- and the
    // whole 3.06e-06 of that difference is the DLR tau -> beta extrapolation of a spectrum
    // that reaches max|E - mu| = 2.22 Ha against a wmax = 1.2 window (the same leakage the
    // Dyson path carries, simple_dyson.cpp:143-145: for a GENERAL external G that IS the
    // only available convention). The wide-window probe above is the falsifier for that
    // attribution. Every downstream deviation lands UNDER that floor.
    //
    // Gate, in order of sharpness:
    //   (i)   the G path, bitwise -- this is the actual injection;
    //   (ii)  structural: nothing may exceed the Dm-convention floor;
    //   (iii) absolute, at 10x the measured numbers (spec §3: measure, then gate).
    REQUIRE(probe.g_read_resid == 0.0);                       // (i)
    REQUIRE(dm_floor_wide < 1e-2 * dm_floor);                 // the attribution, falsified
    REQUIRE(d_e  <= dm_floor);                                // (ii)
    REQUIRE(d_E  <= dm_floor);
    REQUIRE(d_Dm <= dm_floor);
    REQUIRE(dm_floor < 3.1e-5);                               // (iii)
    REQUIRE(d_e  < 3.7e-6);
    REQUIRE(d_E  < 5.5e-6);
    REQUIRE(d_Dm < 2.2e-6);
    REQUIRE(d_gap < 7.3e-5);
    // ... and the gate must not be vacuous: the iteration actually MOVED off the restart
    // state, so a silently-skipped iteration cannot pass it.
    REQUIRE(rB.e_hf != rA.e_hf);
    REQUIRE(std::abs(rB.e_hf - rA.e_hf) > 1e2 * dm_floor);   // MEASURED ratio 668

    drop_chkpt(mpi_context, pa);
    drop_chkpt(mpi_context, pb);
    drop_chkpt(mpi_context, pc);
#endif
  }

  /**
   * ==========================================================================================
   * GATE Q5-g2 -- RESTART COMPOSITION (Option 1 vs Option 2 at C = empty set)
   * ==========================================================================================
   * One niter = 6 run against 3 x (niter = 2, restart) on qe_lih222, ac_pade, ladder ON
   * (pol_vertex = "ladder", inject = "ladder_n2", C = [1, 3) -- the Q3-b configuration).
   *
   * Why this IS the Option-1/Option-2 statement (PDF §5.1 + §7): with no impurity correction
   * the Option-2 outer cycle does nothing to the lattice stage except CUT IT INTO PIECES and
   * restart it. If the pieces compose, then at C = empty set "H_eff re-derived every cycle"
   * and "H_eff derived once" reach the same fixed point -- which is exactly the
   * insensitivity claim, and what licenses moving the stage inside the cycle at all.
   *
   * The composition is not free: each restart re-reads H_eff from h5, re-canonicalizes, and
   * re-enters iter_alg damping against the checkpointed H_eff. Damping (iter_scf::damp_t)
   * carries no state beyond the checkpoint, so the chain reproduces the trajectory; a
   * history-carrying mixer (DIIS) would NOT, and that is a real Option-2 constraint.
   */
  TEST_CASE("qpgw_q5_restart_composition_lih222", "[methods][qpgw][q5]") {
#ifndef ENABLE_DLR
    SUCCEED("qpgw_q5_restart_composition_lih222 skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace qpgw_q5_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));

    const auto win = nda::range(1, 3);
    const double conv_tol = 1e-6;
    const std::string p1 = "qpgw_q5_g2_one", p3 = "qpgw_q5_g2_chain";

    // trajectory 1: one 6-iteration run
    auto one = run_qp(mpi_context, mf, ft, "g2_one(niter=6)", p1, "ladder", "ladder_n2",
                      win, 6, false, conv_tol);

    // trajectory 2: 3 x (niter = 2, restart) -- the Option-2 outer cycle at C = empty set
    auto c1 = run_qp(mpi_context, mf, ft, "g2_chain(1/3)", p3, "ladder", "ladder_n2",
                     win, 2, false, conv_tol);
    auto c2 = run_qp(mpi_context, mf, ft, "g2_chain(2/3)", p3, "ladder", "ladder_n2",
                     win, 2, true, conv_tol);
    auto c3 = run_qp(mpi_context, mf, ft, "g2_chain(3/3)", p3, "ladder", "ladder_n2",
                     win, 2, true, conv_tol);

    const double d_e   = std::abs(one.e_hf - c3.e_hf);
    const double d_gap = std::abs(one.gap_eV() - c3.gap_eV());
    const double d_E   = max_abs_diff(one.E_ska, c3.E_ska);
    app_log(1, "@@Q5G2 one-shot   : final iter {}, e_hf = {:.12f}, gap = {:.9f} eV",
            one.final_iter, one.e_hf, one.gap_eV());
    app_log(1, "@@Q5G2 3x restart : final iter {}, e_hf = {:.12f}, gap = {:.9f} eV "
               "(legs: {:.9f} -> {:.9f} -> {:.9f} eV)",
            c3.final_iter, c3.e_hf, c3.gap_eV(), c1.gap_eV(), c2.gap_eV(), c3.gap_eV());
    app_log(1, "@@Q5G2 composition: d_e_hf = {:.3e}, d_gap = {:.3e} eV, |dE_ska|_max = {:.3e}",
            d_e, d_gap, d_E);

    // MEASURED 2026-08-14 (qe_lih222, ac_pade, ladder ON, C = [1,3), conv_thr = 1e-6):
    //   one-shot   : final iter 6, e_hf = -4.276441914522236, gap = 11.797325416 eV
    //   3x restart : final iter 6, e_hf = -4.276441914522236, gap = 11.797325416 eV
    //                legs 11.785135844 -> 11.793459924 -> 11.797325416 eV
    //   d_e_hf = 0.000e+00, d_gap = 0.000e+00 eV, |dE_ska|_max = 0.000e+00
    // The chain composes BITWISE: damping is stateless beyond the checkpoint, and the
    // restart-init re-derivation is exact (update_MOs on the checkpointed H_eff reproduces
    // MO/E, and update_mu's bisection early-exit returns the checkpointed mu untouched --
    // MEASURED mu = 0.100000000000 at every leg). The spec's conv_thr class (1e-6) is the
    // ceiling it must beat; the gate is set at the measured class, i.e. equality.
    // The absolute energies above carry ~1e-14 run-to-run noise (FP reduction order); the
    // gate is INTRA-run (both trajectories in one process), so that noise cancels.
    REQUIRE(one.final_iter == c3.final_iter);
    REQUIRE(d_e == 0.0);
    REQUIRE(d_gap == 0.0);
    REQUIRE(d_E == 0.0);
    // non-vacuous: the legs really did move (a frozen loop would compose trivially)
    REQUIRE(c1.gap_eV() != c3.gap_eV());

    drop_chkpt(mpi_context, p1);
    drop_chkpt(mpi_context, p3);
#endif
  }

} // bdft_tests
