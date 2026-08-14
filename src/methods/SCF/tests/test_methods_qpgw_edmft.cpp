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

/**
 * Project 2 increment Q4 (notes/q4_edmft_skeleton_spec.md): the EDMFT skeleton's C++ half.
 * The qpGW+BSE lattice stage wired into the GW+EDMFT embedding machinery, and the ladder
 * half of the eq-7 bosonic double counting.
 *
 * Gates (spec section 3), measure-first-then-gate throughout:
 *   Q4-a    C = empty: a qpgw run with screen_type = "gw_edmft" + a bosonic projector and
 *           NO local polarizabilities is EXACTLY the plain screen_type = "rpa" qpGW+BSE
 *           run -- == on the observables and on every ladder meter, ladder ON and OFF.
 *           (A raw h5 bit-compare is not required: the MBState constructor differs.)
 *   Q4-c3   (i)   additivity/no-leak: the trace of the stored P^lad_loc against the SAME
 *                 contraction taken in the other order and through the primary basis
 *                 (t^dag Pl t, then B^dag . B) -- machine class;
 *           (ii)  empty ladder window => pi_lad_loc ABSENT and everything bitwise;
 *           (iii) ||P^lad_loc|| / ||P_dc,bubble|| logged (the ladder column of the PDF
 *                 section 8.3 cancellation-load meter).
 *
 * On P^lad_loc's convention: meter (iii) is what showed the stored object carries the
 * upfold's ||B||^2 gain, and the R-Q4-2 AMENDMENT consequently rules it NOT a DC
 * contribution (increment Q4-C3b delivers the orbital/chi-convention one). It is retained
 * as the interface diagnostic, its only consumer is opt-in and off by default
 * (downfold_edmft_impl's pi_lad_dc), and the gates below are unchanged: they pin the
 * CONTRACTION (additivity, absence, bitwise), which is convention-independent.
 *
 * ==========================================================================================
 * HOW TO RUN (Catch2 v2 traps) -- MEASURED, do not "improve" the command
 * ==========================================================================================
 *
 *     KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 <build>/tests/bin/test_methods_qpgw_edmft
 *
 * i.e. THE BARE BINARY, no filter: a "~[tag]" spec does NOT exclude hidden cases, it RUNS
 * them, and two positional test names are silently concatenated into one (unmatched) name.
 */

namespace bdft_tests {

  using namespace methods;

  namespace qpgw_edmft_detail {

    constexpr double HA2EV = 27.211386245988;

    struct q4_row {
      std::string tag;
      double e_hf = 0.0;
      double e_homo = 0.0, e_lumo = 0.0;
      long final_iter = -1, niter = -1;
      // the update_w readouts / injection meters of the LAST screening step
      double eps_rpa = -1.0, eps_ladder = -1.0, eps_loop = -1.0;
      double lam_nu0 = -1.0, lam_max = -1.0, r_rt = -1.0, lad_ratio = -1.0;
      // Q4 C3
      bool lad_loc_present = false;
      long lad_loc_h5_iters = -1;   // scf/iterN groups carrying pi_lad_loc_wabcd
      double lad_loc_max = -1.0, lad_loc_ratio = -1.0;
      // gate Q4-c3(i): the two contraction orders and their scale
      double tr_dev = -1.0, tr_scale = -1.0;
      // DIAGNOSTIC (reported, never gated): the upfold metric s(q) = B(q)^dag B(q) and the
      // magnitude of the metric-corrected (least-squares inverse) downfold, for the eq-7
      // scale question recorded with the Q4 report.
      double s_max = -1.0, lad_loc_ls_max = -1.0;
      double gap_eV() const { return (e_lumo - e_homo) * HA2EV; }
    };

    /**
     * One qpGW run. wannier_file empty => the plain (projector-less) MBState of the [qpgw]
     * driver; non-empty => the projector-carrying MBState of the Q4 branch, with NO local
     * polarizabilities set (the C = empty leg: update_w's checkpoint-miss path plus
     * eval_Pi_qdep's "corrections not found" skip).
     *
     * With extra_w, ONE more screening step runs outside the loop -- on the loop's own final
     * spectrum, exactly as the Q3 suite does -- and the meters are re-read from it. Legs
     * whose METERS are compared must set it the SAME way: the extra step re-derives mu with
     * update_mu(0.0, ...) instead of inheriting the loop's, so the two sampling points do
     * not agree bit for bit, and lambda_max in particular is a power iteration with a 1e-3
     * stopping test -- one sweep more or less moves it in the 9th digit.
     * The C3 gates additionally need the extra step (they read the state the loop used).
     */
    inline q4_row run_q4(auto &mpi_context, std::shared_ptr<mf::MF> &mf,
                         imag_axes_ft::IAFT &ft, std::string const &tag,
                         std::string const &screen_type, std::string const &wannier_file,
                         std::string const &pol_mode, std::string const &inject,
                         nda::range window, int thc_prefactor, double thc_tol,
                         int niter, double conv_tol, bool extra_w = false,
                         const std::string &div = "ignore_g0") {
      decltype(nda::range::all) all;
      const std::string output = "qpgw_edmft_" + tag;
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, div, output);
      solvers::scr_coulomb_t scr_eri(&ft, screen_type, div);

      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*thc_prefactor, "", "incore", "",
                                                 output, thc_tol, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      qp_params_t qp_params("sc", "pade", 18, 0.0001, 1e-8, "qpscf");
      qp_params.qp_map = "ac_pade";

      // the driver's [qpgw] wiring (MBPT_drivers.cpp): a pure knob carrier, attached to
      // scr_eri only. It must outlive qp_scf_loop -- same stack frame.
      solvers::vertex_t vtx(&ft, "none", nda::range(0, 0), mf->nbnd(), div);
      vtx.set_pol_vertex(pol_mode, "w0_prev", window, -1, 1e-8, -1.0, -1.0, -1.0, inject);
      if (vtx.pol_vertex_enabled()) scr_eri.set_vertex(&vtx);
      REQUIRE(not scr_eri.has_active_vertex());     // Sigma stays GW-form on this path

      iter_scf::iter_scf_t iter_sol(iter_scf::damp_t(0.7));
      std::optional<MBState> mbs;
      if (wannier_file.empty())
        mbs.emplace(mpi_context, ft, output);
      else
        mbs.emplace(ft, output, mf, wannier_file, true);
      auto &mb_state = mbs.value();
      REQUIRE(mb_state.proj_boson.has_value() == (not wannier_file.empty()));

      q4_row row;
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

      nda::array<ComplexType, 3> E_ska;
      {
        h5::file file(output + ".mbpt.h5", 'r');
        auto scf_grp = h5::group(file).open_group("scf");
        h5::h5_read(scf_grp, "final_iter", row.final_iter);
        auto iter_grp = scf_grp.open_group("iter" + std::to_string(row.final_iter));
        nda::h5_read(iter_grp, "E_ska", E_ska);
        // C3: the eq-7 ladder DC has to REACH the checkpoint -- it is how python's DC
        // assembly and a separate-process downfold_2e see it at all.
        row.lad_loc_h5_iters = 0;
        for (long it = 0; it <= row.final_iter; ++it) {
          std::string gn = "iter" + std::to_string(it);
          if (scf_grp.has_subgroup(gn) and
              scf_grp.open_group(gn).has_dataset("pi_lad_loc_wabcd"))
            ++row.lad_loc_h5_iters;
        }
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

      if (extra_w) {
        using math::shm::make_shared_array;
        double mu = update_mu(0.0, *mf, mb_state.sE_ska.value(), ft.beta());
        mb_state.sG_tskij.emplace(make_shared_array<Array_view_5D_t>(
            *mpi_context, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
        update_G(mb_state.sG_tskij.value(), mb_state.sMO_skia.value(),
                 mb_state.sE_ska.value(), mu, ft);
        scr_eri.update_w(mb_state, thc, -1);
        std::tie(row.eps_rpa, row.eps_ladder) = scr_eri.pol_eps_readout();
        row.eps_loop = scr_eri.pol_eps_loop();
        row.lam_nu0 = scr_eri.pol_lambda_nu0();
        row.lam_max = scr_eri.pol_lambda_max();
        row.r_rt = scr_eri.pol_round_trip();
        row.lad_ratio = scr_eri.pol_ladder_ratio();
        row.lad_loc_max = scr_eri.pol_lad_loc_max();
        row.lad_loc_ratio = scr_eri.pol_lad_loc_ratio();
        row.lad_loc_present = mb_state.sPi_lad_loc_wabcd.has_value();

        // ---- gate Q4-c3(i): the additivity / no-leak identity ---------------------------
        // The stored P^lad_loc is assembled with the Y = t(q) B(q) shortcut, one q at a
        // time, accumulating the abcd TENSOR. The reference below takes the SAME sum in
        // the other order and through the PRIMARY basis: form the upfolded ladder
        // M(q) = t(q)^dag Pl(q) t(q) explicitly (Np x Np), then trace against B directly,
        // accumulating a SCALAR. Equal contraction, different association and a different
        // intermediate basis, so a wrong q weight, a wrong star map, a missed time-reversal
        // conjugation or a transposed pair index all show up here.
        if (row.lad_loc_present) {
          auto *pv = scr_eri.pol_vertex_instance();
          REQUIRE(pv != nullptr);
          auto Pl = pv->eval_pol_ladder_whalf(mb_state, thc);   // (nw_h, nq_ibz, Nm, Nm)
          auto const &tmap = pv->secondary_transfer();          // (nq_ibz, Nm, Np)
          auto &pb = mb_state.proj_boson.value();
          auto sB_qIPab = (mf->nqpts_ibz() == mf->nqpts()) ?
                          pb.calc_bosonic_projector(thc) :
                          pb.calc_bosonic_projector_symm(thc);
          auto B = sB_qIPab.local();
          const long nw_h = Pl.shape(0), Nm = Pl.shape(2), Np = tmap.shape(2);
          const long nq_full = mf->nqpts(), n = pb.nImpOrbs();

          // DIAGNOSTIC, reported and never gated (see the Q4 report): the upfold's metric
          // s(q) = B(q)^dag B(q) on the local pair index. upfold_pi_local maps a local Pi
          // to B Pi B^dag, so its LEAST-SQUARES INVERSE is s^-1 B^dag . B s^-1, not the
          // plain adjoint B^dag . B that C3 stores. ||s|| is the gain between the two, and
          // it decides whether the stored object sits on the scale of bubble[G_loc].
          const long nab = n * n;
          nda::matrix<ComplexType> sinv_q(nab, nab), Dls(nab, nab);
          nda::array<ComplexType, 3> Dls_acc(nw_h, nab, nab);
          Dls_acc() = ComplexType(0.0);
          double s_max_loc = 0.0;

          ComplexType tr_q(0.0);
          nda::matrix<ComplexType> tq(Nm, Np), tmp(Nm, Np), M(Np, Np);
          nda::matrix<ComplexType> Bq(Np, nab), MB(Np, nab), BMB(nab, nab), sq(nab, nab);
          nda::vector<ComplexType> bv(Np), Mb(Np);
          for (long iqf = 0; iqf < nq_full; ++iqf) {
            const long iqi = mf->qp_to_ibz(iqf);
            const bool trev = mf->qp_trev(iqf);
            tq = tmap(iqi, all, all);
            for (long P = 0; P < Np; ++P)
              for (long a = 0; a < n; ++a)
                for (long b = 0; b < n; ++b) Bq(P, a*n + b) = B(iqf, 0, P, a, b);
            nda::blas::gemm(nda::dagger(Bq), Bq, sq);
            for (auto const &v : sq) s_max_loc = std::max(s_max_loc, std::abs(v));
            sinv_q = nda::inverse(sq);
            for (long j = 0; j < nw_h; ++j) {
              nda::blas::gemm(nda::matrix<ComplexType>(Pl(j, iqi, all, all)), tq, tmp);
              nda::blas::gemm(nda::dagger(tq), tmp, M);
              if (trev) M = nda::conj(M);
              for (long a = 0; a < n; ++a)
                for (long b = 0; b < n; ++b) {
                  for (long P = 0; P < Np; ++P) bv(P) = B(iqf, 0, P, a, b);
                  nda::blas::gemv(ComplexType(1.0), M, bv, ComplexType(0.0), Mb);
                  for (long P = 0; P < Np; ++P) tr_q += std::conj(bv(P)) * Mb(P);
                }
              // the metric-corrected object at this (q, nu), accumulated as the q-mean
              nda::blas::gemm(M, Bq, MB);
              nda::blas::gemm(nda::dagger(Bq), MB, BMB);
              nda::blas::gemm(sinv_q, BMB, Dls);
              nda::blas::gemm(ComplexType(1.0)/double(nq_full), Dls, sinv_q,
                              ComplexType(0.0), BMB);
              Dls_acc(j, all, all) += BMB;
            }
          }
          tr_q /= double(nq_full);
          row.s_max = s_max_loc;
          row.lad_loc_ls_max = 0.0;
          for (auto const &v : Dls_acc)
            row.lad_loc_ls_max = std::max(row.lad_loc_ls_max, std::abs(v));

          ComplexType tr_loc(0.0);
          double scale = 0.0;
          {
            auto D = mb_state.sPi_lad_loc_wabcd.value().local();
            for (long j = 0; j < nw_h; ++j)
              for (long a = 0; a < n; ++a)
                for (long b = 0; b < n; ++b) {
                  tr_loc += D(j, a, b, a, b);
                  scale += std::abs(D(j, a, b, a, b));
                }
          }
          row.tr_dev = std::abs(tr_loc - tr_q);
          row.tr_scale = scale;
          app_log(1, "@@Q4C3(i) [{}] Tr P^lad_loc: tensor route = {:.12e} {:+.12e}i, "
                     "primary-basis route = {:.12e} {:+.12e}i; |d| = {:.3e}, "
                     "sum |Tr| = {:.3e} (relative {:.3e})", tag,
                  tr_loc.real(), tr_loc.imag(), tr_q.real(), tr_q.imag(),
                  row.tr_dev, scale, row.tr_dev / std::max(scale, 1e-300));
          app_log(1, "@@Q4SCALE [{}] upfold metric ||B^dag B||_max = {:.4e}; stored "
                     "(adjoint) ||P^lad_loc||_max = {:.4e}; least-squares-inverse "
                     "||s^-1 B^dag . B s^-1||_max = {:.4e} (ratio adjoint/LS = {:.3e})",
                  tag, row.s_max, row.lad_loc_max, row.lad_loc_ls_max,
                  row.lad_loc_max / std::max(row.lad_loc_ls_max, 1e-300));
        }
      }

      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      app_log(1, "qpgw_edmft [{}]: screen = {}, projector = {}, final iter {} of {}, "
                 "e_hf = {}, gap = {:.9f} eV (homo {:.9f}, lumo {:.9f} Ha)",
              tag, screen_type, wannier_file.empty() ? "no" : "yes", row.final_iter,
              niter, row.e_hf, row.gap_eV(), row.e_homo, row.e_lumo);
      return row;
    }

  } // qpgw_edmft_detail

  /**
   * ==========================================================================================
   * GATE Q4-a -- C = empty reproduces qpGW+BSE EXACTLY (and Q4-c3)
   * ==========================================================================================
   * screen_type = "gw_edmft" with a bosonic projector but no impurity/DC polarizability is
   * arithmetically the RPA lattice problem: eval_Pi_qdep logs the "corrections not found"
   * note and adds nothing. The Q4 seam refactor moved the ladder kernel build AND the
   * injection into eval_Pi_qdep, and the edmft branch sits BETWEEN them, so this leg is what
   * proves the ladder still lands on the same Pi in the embedding screen mode.
   *
   * The C3 accumulation runs in the projector legs only (it needs B(q)); it writes no
   * lattice quantity, so the observables must be untouched by its presence.
   */
  TEST_CASE("qpgw_edmft_cempty_lih222", "[methods][qpgw][edmft]") {
#ifndef ENABLE_DLR
    SUCCEED("qpgw_edmft_cempty_lih222 skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace qpgw_edmft_detail;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    auto [outdir, prefix] = utils::utest_filename("qe_lih222");
    const std::string wannier_file = outdir + "/lih_wan.h5";

    // ---- ladder OFF: the structural no-op leg (Q4-c3(ii)) ------------------------------
    auto off_rpa = run_q4(mpi_context, mf, ft, "off_rpa", "rpa", "", "none", "none",
                          nda::range(0, 0), 12, 1e-10, 20, 1e-6);
    auto off_edmft = run_q4(mpi_context, mf, ft, "off_edmft", "gw_edmft", wannier_file,
                            "none", "none", nda::range(0, 0), 12, 1e-10, 20, 1e-6, true);
    app_log(1, "@@Q4A ladder OFF: e_hf rpa = {}, gw_edmft = {} (d = {:.3e}); gap "
               "{:.9f} vs {:.9f} eV (d = {:.3e})", off_rpa.e_hf, off_edmft.e_hf,
            std::abs(off_edmft.e_hf - off_rpa.e_hf), off_rpa.gap_eV(),
            off_edmft.gap_eV(), std::abs(off_edmft.gap_eV() - off_rpa.gap_eV()));
    REQUIRE(off_edmft.e_hf == off_rpa.e_hf);
    REQUIRE(off_edmft.e_homo == off_rpa.e_homo);
    REQUIRE(off_edmft.e_lumo == off_rpa.e_lumo);
    REQUIRE(off_edmft.final_iter == off_rpa.final_iter);
    // no ladder was ever built, in either leg
    REQUIRE(off_rpa.lam_nu0 == -1.0);
    REQUIRE(off_edmft.lam_nu0 == -1.0);
    // Q4-c3(ii): no window => no P^lad_loc object at all, in MBState or the checkpoint
    REQUIRE(not off_edmft.lad_loc_present);
    REQUIRE(off_edmft.lad_loc_max == -1.0);
    REQUIRE(off_edmft.lad_loc_h5_iters == 0);
    REQUIRE(off_rpa.lad_loc_h5_iters == 0);

    // ---- ladder ON, C = [1, 3): the live leg -------------------------------------------
    // BOTH legs take the extra screening step, so their meters are sampled at the SAME
    // point (see run_q4: the loop's last update_w and the extra one do not agree bit for
    // bit, and lambda_max is a power iteration with a loose stopping test).
    auto on_rpa = run_q4(mpi_context, mf, ft, "on_rpa", "rpa", "", "ladder", "ladder_n2",
                         nda::range(1, 3), 12, 1e-10, 20, 1e-6, true);
    auto on_edmft = run_q4(mpi_context, mf, ft, "on_edmft", "gw_edmft", wannier_file,
                           "ladder", "ladder_n2", nda::range(1, 3), 12, 1e-10, 20, 1e-6,
                           true);
    app_log(1, "@@Q4A ladder ON: e_hf rpa = {}, gw_edmft = {} (d = {:.3e}); gap {:.9f} vs "
               "{:.9f} eV (d = {:.3e}); meters lam_nu0 {:.9f}/{:.9f}, r_rt {:.3e}/{:.3e}, "
               "||P^lad||/||P^RPA|| {:.6e}/{:.6e}", on_rpa.e_hf, on_edmft.e_hf,
            std::abs(on_edmft.e_hf - on_rpa.e_hf), on_rpa.gap_eV(), on_edmft.gap_eV(),
            std::abs(on_edmft.gap_eV() - on_rpa.gap_eV()), on_rpa.lam_nu0,
            on_edmft.lam_nu0, on_rpa.r_rt, on_edmft.r_rt, on_rpa.lad_ratio,
            on_edmft.lad_ratio);
    REQUIRE(on_edmft.e_hf == on_rpa.e_hf);
    REQUIRE(on_edmft.e_homo == on_rpa.e_homo);
    REQUIRE(on_edmft.e_lumo == on_rpa.e_lumo);
    REQUIRE(on_edmft.final_iter == on_rpa.final_iter);
    // NON-VACUITY: the ladder really was injected, and it really moved the loop
    REQUIRE(on_rpa.lam_nu0 > 0.0);
    REQUIRE(on_rpa.lad_ratio > 0.0);
    REQUIRE(on_rpa.e_hf != off_rpa.e_hf);
    // ... and every injection meter of the two legs agrees, not just the observables
    REQUIRE(on_edmft.lam_nu0 == on_rpa.lam_nu0);
    REQUIRE(on_edmft.lam_max == on_rpa.lam_max);
    REQUIRE(on_edmft.r_rt == on_rpa.r_rt);
    REQUIRE(on_edmft.lad_ratio == on_rpa.lad_ratio);

    // ---- Q4-c3(i)/(iii): the ladder DC object ------------------------------------------
    app_log(1, "@@Q4C3(iii) ||P^lad_loc||_max = {:.6e} (no bubble P_dc in the C = empty "
               "leg, ratio = {:.3e}); additivity |d| = {:.3e} at scale {:.3e}",
            on_edmft.lad_loc_max, on_edmft.lad_loc_ratio, on_edmft.tr_dev,
            on_edmft.tr_scale);
    app_log(1, "@@Q4C3 checkpoint: scf/iterN groups carrying pi_lad_loc_wabcd = {} "
               "(gw_edmft+ladder), {} (rpa+ladder, no projector), {} (gw_edmft, no ladder)",
            on_edmft.lad_loc_h5_iters, on_rpa.lad_loc_h5_iters,
            off_edmft.lad_loc_h5_iters);
    REQUIRE(on_edmft.lad_loc_present);
    REQUIRE(on_edmft.lad_loc_max > 0.0);        // non-vacuous: something was downfolded
    // ... and it reached the checkpoint (every iteration of the loop dumps it)
    REQUIRE(on_edmft.lad_loc_h5_iters > 0);
    // no projector => nothing to downfold onto, so nothing is written
    REQUIRE(on_rpa.lad_loc_h5_iters == 0);
    REQUIRE(on_edmft.tr_scale > 0.0);
    // the two contraction orders must agree at machine class relative to the traced scale.
    // MEASURED: see the @@Q4C3(i) line; the gate is 10x the measured relative deviation,
    // floored at the double-precision accumulation class of an Np^2 sum.
    REQUIRE(on_edmft.tr_dev < 1e-12 * on_edmft.tr_scale);
#endif
  }

} // bdft_tests
