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

/**
 * GATE TC-2-a (spec M3) and GATE TC-2-c of notes/tc_coqui_impl_spec.md.
 *
 * Harness pattern: the RW-1 Lehmann gate
 * (methods/GW_real_axis/tests/test_real_axis_w_lehmann.cpp) -- the SAME QP-pole
 * G on both sides, the same THC factorization (common mode, cancels), the same
 * mu, the same beta, div_treatment = ignore_g0 on the imaginary side.
 *
 *   TC-2-a(i)   THE CONTRACTION. The complex-time kernel evaluated at t = -i tau
 *               against the production Pi(q, tau) from scr_coulomb_t. These are
 *               algebraically the same object (p_contour.hpp section 1), so the
 *               only difference allowed is round-off. This isolates the
 *               contraction -- including the two TRANSPOSES -- from the
 *               transform, and it is what would have caught a conj-vs-transpose
 *               slip.
 *
 *   TC-2-a(ii)  THE M3 GATE. P(i nu) evaluated THROUGH the tilted transform
 *               against the production imaginary-axis Pi(q, i nu), RESTRICTED to
 *               nu >= gamma = delta (1 - rho) (results section 5.5: below that a
 *               target on the imaginary axis has less damping than the contour's
 *               own design floor, so the contour is simply too short for it).
 *               A per-fixture budget is DERIVED from the fixture's own
 *               transition support and reported next to the measurement.
 *
 *   TC-2-c      band truncation along the contour on/off: identical to the
 *               truncation tolerance, with the measured band-work speedup.
 */

#undef NDEBUG

#include <cmath>
#include <complex>
#include <limits>
#include <vector>

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"
#include "mean_field/default_MF.hpp"

#include "methods/ERI/thc_reader_t.hpp"
#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/mb_eri_context.h"

#include "nda/nda.hpp"
#include "nda/linalg.hpp"
#include "numerics/distributed_array/nda_utils.hpp"

#include "methods/mb_state/mb_state.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_common.hpp"
#include "methods/SCF/p_contour.hpp"
#include "methods/SCF/wc_line.hpp"
#include "methods/scr_coulomb/scr_coulomb_t.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "hamiltonian/one_body_hamiltonian.hpp"

namespace bdft_tests {

  using namespace methods;
  namespace pc = methods::p_contour;
  using cval_t = std::complex<double>;

  // =====================================================================
  //  The gate body, parameterized by fixture.
  // =====================================================================
  static void run_tc2_gate(char const *mf_src, double gate_ii) {
    auto &mpi_context = utils::make_unit_test_mpi_context();
    if (mpi_context->comm.size() != 1) return;      // single-rank gate

    decltype(nda::range::all) all;

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, mf_src));
    const int nIpts = mf->nbnd() * 2;
    thc_reader_t thc(mf, make_thc_reader_ptree(nIpts, "", "incore", "",
                                               "bdft", 1e-8,
                                               mf->ecutrho(), 1, 1024));

    const long ns     = mf->nspin();
    const long Nk     = mf->nkpts();
    const long Nk_ibz = mf->nkpts_ibz();
    const long Nq_ibz = mf->nqpts_ibz();
    const long nbnd   = mf->nbnd();
    const long Naux   = thc.Np();
    const double beta = 1000.0;

    // The time-reversal census: eq (T) of p_contour.hpp -- the TRANSPOSE fill of the
    // trev-paired FBZ points -- is only exercised when this is non-zero.
    long n_trev = 0;
    {
      auto kt = mf->kp_trev();
      for (long ik = 0; ik < Nk; ++ik) if (kt(ik)) ++n_trev;
    }
    app_log(2, "[TC-2] fixture {}: ns={} Nk={} Nk_ibz={} Nq_ibz={} nbnd={} Naux={}; "
               "time-reversal pairs = {} ({} FBZ points carry kp_trev -> eq (T), the "
               "TRANSPOSE fill, is {})",
            mf_src, ns, Nk, Nk_ibz, Nq_ibz, nbnd, Naux, mf->nkpts_trev_pairs(), n_trev,
            n_trev > 0 ? "EXERCISED" : "NOT exercised on this fixture");

    auto eigval = mf->eigval();

    double e_min = std::numeric_limits<double>::infinity();
    double e_max = -std::numeric_limits<double>::infinity();
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk_ibz; ++k)
        for (long n = 0; n < nbnd; ++n) {
          e_min = std::min(e_min, eigval(s, k, n));
          e_max = std::max(e_max, eigval(s, k, n));
        }
    const double w_max = std::max(std::abs(e_min), std::abs(e_max)) + 2.0;

    // ---------------- the Matsubara branch ------------------------------
    imag_axes_ft::IAFT ft(beta, w_max + 1.0, imag_axes_ft::dlr_basis, "high");
    const std::string prefix = std::string("coqui_tc2_") + mf_src;

    MBState mb_state(mpi_context, ft, prefix);
    simple_dyson dyson(mf.get(), &ft);

    mb_state.sF_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(
        *mpi_context, {ns, Nk_ibz, nbnd, nbnd}));
    mb_state.sDm_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(
        *mpi_context, {ns, Nk_ibz, nbnd, nbnd}));
    mb_state.sG_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(
        *mpi_context, {ft.nt_f(), ns, Nk_ibz, nbnd, nbnd}));
    mb_state.sSigma_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(
        *mpi_context, {ft.nt_f(), ns, Nk_ibz, nbnd, nbnd}));
    auto &sF = mb_state.sF_skij.value();
    auto &sDm = mb_state.sDm_skij.value();
    auto &sG = mb_state.sG_tskij.value();
    auto &sSigma = mb_state.sSigma_tskij.value();

    hamilt::set_fock(*mf, dyson.PSP(), sF, true);
    if (mpi_context->node_comm.root()) sSigma.local() = ComplexType(0.0);
    sSigma.communicator()->barrier();

    double mu = 0.0;
    update_G(dyson, *mf, ft, sDm, sG, sF, sSigma, mu, /*const_mu*/ false);
    app_log(2, "[TC-2] Matsubara mu = {:.10f} Ha", mu);

    // PIN P1 (the RW-1 pin): the static Hamiltonian must be diag(eps) with S = 1,
    // so that "the same E / MO on both sides" is realized with an identity MO.
    {
      auto H0 = dyson.H0();
      auto S = dyson.sS_skij().local();
      auto F = sF.local();
      double off_H = 0.0, dia_H = 0.0, off_S = 0.0, dia_S = 0.0;
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk_ibz; ++k)
          for (long i = 0; i < nbnd; ++i)
            for (long j = 0; j < nbnd; ++j) {
              const cval_t h = H0(s, k, i, j) + F(s, k, i, j);
              if (i == j) {
                dia_H = std::max(dia_H, std::abs(h - cval_t(eigval(s, k, i), 0.0)));
                dia_S = std::max(dia_S, std::abs(S(s, k, i, j) - cval_t(1.0, 0.0)));
              } else {
                off_H = std::max(off_H, std::abs(h));
                off_S = std::max(off_S, std::abs(S(s, k, i, j)));
              }
            }
      app_log(2, "[TC-2] PIN P1: |H0+F-diag(eps)| = {:.2e}/{:.2e} (dia/off); "
                 "|S-1| = {:.2e}/{:.2e}", dia_H, off_H, dia_S, off_S);
      REQUIRE(dia_H < 1e-8);
      REQUIRE(off_H < 1e-8);
      REQUIRE(dia_S < 1e-8);
      REQUIRE(off_S < 1e-8);
    }

    // the production Pi(q, tau) and Pi(q, i nu), from the SAME G
    solvers::scr_coulomb_t scr_im(&ft, "rpa", "ignore_g0");
    // update_w FIRST (the RW-1 gate's order): it fills mb_state.dW_qtPQ, which the
    // TC-3-b(0) leg below reads as the production W^c reference. eval_Pi_qdep is
    // idempotent given sG_tskij, so the Pi it returns is the same bubble.
    scr_im.update_w(mb_state, thc, -1);
    REQUIRE(mb_state.dW_qtPQ.has_value());
    const long nw_half = (ft.nw_b() % 2 == 0) ? ft.nw_b() / 2 : ft.nw_b() / 2 + 1;
    nda::array<ComplexType, 4> Pi_tau, Pi_im_qwPQ;
    long ntp = 0;
    {
      auto dPi = scr_im.eval_Pi_qdep(mb_state, thc);
      auto gsh = dPi.global_shape();
      ntp = gsh[0];
      nda::array<ComplexType, 4> Pt(ntp, gsh[1], gsh[2], gsh[3]);
      Pt() = ComplexType(0.0);
      Pt(dPi.local_range(0), dPi.local_range(1),
         dPi.local_range(2), dPi.local_range(3)) = dPi.local();
      mpi_context->comm.all_reduce_in_place_n(Pt.data(), Pt.size(), std::plus<>{});
      Pi_tau = Pt;
      nda::array<ComplexType, 4> Pw(nw_half, gsh[1], gsh[2], gsh[3]);
      ft.tau_to_w_PHsym(Pt, Pw);
      Pi_im_qwPQ = nda::array<ComplexType, 4>(Nq_ibz, nw_half, Naux, Naux);
      for (long iq = 0; iq < Nq_ibz; ++iq)
        for (long n = 0; n < nw_half; ++n)
          Pi_im_qwPQ(iq, n, all, all) = Pw(n, iq, all, all);
      app_log(2, "[TC-2] production Pi: (nt_half, nq, P, Q) = ({}, {}, {}, {}); "
                 "nw_half = {}", gsh[0], gsh[1], gsh[2], gsh[3], nw_half);
    }

    // the QP spectrum / MOs handed to the contour: identity MO, absolute eps
    auto sE = math::shm::make_shared_array<Array_view_3D_t>(
        *mpi_context, {ns, Nk_ibz, nbnd});
    auto sMO = math::shm::make_shared_array<Array_view_4D_t>(
        *mpi_context, {ns, Nk_ibz, nbnd, nbnd});
    if (mpi_context->node_comm.root()) {
      sE.local() = ComplexType(0.0);
      sMO.local() = ComplexType(0.0);
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk_ibz; ++k)
          for (long n = 0; n < nbnd; ++n) {
            sE.local()(s, k, n) = ComplexType(eigval(s, k, n), 0.0);
            sMO.local()(s, k, n, n) = ComplexType(1.0, 0.0);
          }
    }
    sE.communicator()->barrier();
    sMO.communicator()->barrier();

    // ---- PROVENANCE PROBE: the complex-time legs at t = -i tau must reproduce
    // the Dyson G(tau) itself, band by band. It ALSO calibrates gate a(i): the
    // reference Pi(q,tau) is built from the DLR-REPRESENTED G(tau), whose own
    // reconstruction error is what a(i) can never go below. Pi is bilinear in G,
    // so the floor is ~2 g_rel; the gate is set at 10 g_rel.
    double g_rel = 0.0;
    {
      // IAFT's tau_mesh_* carries the RELATIVE coordinate x in [-1, 1];
      // tau = (x + 1) beta / 2 (qp_scf_common.cpp:1030, the code's own map).
      auto xm = ft.tau_mesh_f();
      const long ntf = ft.nt_f();
      nda::array<double, 1> tau(ntf);
      for (long i = 0; i < ntf; ++i) tau(i) = (double(xm(i)) + 1.0) * beta / 2.0;
      app_log(2, "[TC-2 probe] nt_f = {}, x in [{:.6g}, {:.6g}] -> tau in "
                 "[{:.6g}, {:.6g}], beta = {:.6g}",
              ntf, double(xm(0)), double(xm(ntf - 1)), tau(0), tau(ntf - 1), beta);
      double dg_num = 0.0, dg_den = 0.0, dn_num = 0.0;
      for (long it = 0; it < ntf; ++it) {
        const ComplexType t = ComplexType(0.0, -1.0) * double(tau(it));
        for (long s = 0; s < ns; ++s)
          for (long k = 0; k < Nk_ibz; ++k)
            for (long n = 0; n < nbnd; ++n) {
              const double xi = eigval(s, k, n) - mu;
              ComplexType fp, fn;
              double lgp, lgn;
              pc::detail::leg_weights(xi, beta, t, fp, fn, lgp, lgn);
              const ComplexType gref = sG.local()(it, s, k, n, n);
              dg_num = std::max(dg_num, std::abs(fp - gref));
              dg_den = std::max(dg_den, std::abs(gref));
              const ComplexType gref_n = sG.local()(ntf - 1 - it, s, k, n, n);
              dn_num = std::max(dn_num, std::abs(fn - gref_n));
            }
      }
      g_rel = std::max(dg_num, dn_num) / std::max(dg_den, 1e-300);
      app_log(2, "[TC-2 probe] max|fp(-i tau) - G(tau)| = {:.3e}, "
                 "max|fn(-i tau) - G(beta-tau)| = {:.3e}, over max|G| = {:.3e} "
                 "-> g_rel = {:.3e} (the DLR reconstruction error of the reference "
                 "G, i.e. the floor of gate a(i))", dg_num, dn_num, dg_den, g_rel);
    }

    // =================================================================
    //  TC-2-a(i) -- the contraction, at t = -i tau
    // =================================================================
    {
      auto xm = ft.tau_mesh_f();
      nda::array<ComplexType, 1> tlist(ntp);
      for (long i = 0; i < ntp; ++i)
        tlist(i) = ComplexType(0.0, -1.0) * ((double(xm(i)) + 1.0) * beta / 2.0);
      auto sPi = math::shm::make_shared_array<Array_view_4D_t>(
          *mpi_context, {Nq_ibz, ntp, Naux, Naux});
      pc::ctx_t dg;
      pc::sample_P_at_times(sPi, tlist, thc, sMO, sE, mu, beta, /*trunc*/ -1.0, &dg);

      double num = 0.0, den = 0.0;
      long wq = -1, wt = -1;
      for (long iq = 0; iq < Nq_ibz; ++iq)
        for (long it = 0; it < ntp; ++it)
          for (long P = 0; P < Naux; ++P)
            for (long Q = 0; Q < Naux; ++Q) {
              const double d = std::abs(sPi.local()(iq, it, P, Q)
                                      - Pi_tau(it, iq, P, Q));
              if (d > num) { num = d; wq = iq; wt = it; }
              den = std::max(den, std::abs(Pi_tau(it, iq, P, Q)));
            }
      app_log(2, "[TC-2-a(i)] complex-time kernel at t = -i tau vs production "
                 "Pi(q,tau): max abs dev {:.3e} over max|Pi| {:.3e} = {:.3e} rel "
                 "(worst at q = {}, tau index {} of {}); worst single-leg wrong-branch occupation weight "
                 "{:.2e}; gate = 10 g_rel = {:.2e}",
              num, den, num / den, wq, wt, ntp, dg.thermal_worst, 10.0 * g_rel);
      REQUIRE(num / den < std::max(1e-13, 10.0 * g_rel));
    }

    // =================================================================
    //  the contour
    // =================================================================
    pc::opts_t o;
    o.eps = 1e-6;
    o.rho = 0.65;
    o.profile = "flat";
    o.zeta_max = 10.0 / pc::ha_to_eV;
    o.nx = 2500;
    long nk_lin = 1;
    {
      auto kg = mf->kp_grid();
      nk_lin = std::min({long(kg(0)), long(kg(1)), long(kg(2))});
    }
    auto ctx = pc::build_contour_for_spectrum(sE.local(), mu, nk_lin, beta, o);
    pc::log_contour(ctx, o, 2);
    const long r = ctx.c.rank;

    auto sPc = math::shm::make_shared_array<Array_view_4D_t>(
        *mpi_context, {Nq_ibz, r, Naux, Naux});
    pc::ctx_t dg;
    pc::sample_P_at_times(sPc, ctx.t_node, thc, sMO, sE, mu, beta, -1.0, &dg);
    app_log(2, "[TC-2] contour samples: {} nodes; Pi Hermiticity at node 0 "
               "max|Pi_PQ - conj(Pi_QP)|/max|Pi| = {:.3e} (REPORTED: Pi is NOT "
               "Hermitian at complex t)", r, dg.herm_rel);

    // =================================================================
    //  TC-2-a(ii) -- P(i nu) through the transform
    // =================================================================
    auto wn_b = ft.wn_mesh_b();
    nda::array<double, 1> nu_n(nw_half);
    for (long n = 0; n < nw_half; ++n)
      nu_n(n) = ft.omega(wn_b(ft.nw_b() / 2 + n)).imag();
    REQUIRE(std::abs(nu_n(0)) < 1e-12);

    const double gamma = ctx.c.g.gamma;         // = delta (1 - rho) cos theta
    std::vector<long> keep;
    for (long n = 0; n < nw_half; ++n)
      if (nu_n(n) >= gamma) keep.push_back(n);
    app_log(2, "[TC-2-a(ii)] VALIDITY WINDOW: the imaginary-axis cross-check is "
               "restricted to nu >= gamma = delta cos(theta) (1 - rho) = {:.6g} a.u. "
               "({:.4g} eV) -- results section 5.5. {} of {} bosonic half-mesh nodes "
               "qualify; nu in [{:.4g}, {:.4g}] a.u.",
            gamma, gamma * pc::ha_to_eV, keep.size(), nw_half,
            keep.empty() ? 0.0 : nu_n(keep.front()),
            keep.empty() ? 0.0 : nu_n(keep.back()));
    REQUIRE(keep.size() >= 4);

    nda::array<ComplexType, 1> ztarg(long(keep.size()));
    for (std::size_t i = 0; i < keep.size(); ++i)
      ztarg(long(i)) = ComplexType(0.0, nu_n(keep[i]));
    // z = i nu is its OWN conjugate mirror (-conj(i nu) = i nu), so the mirror
    // rows would be duplicates: build the resonant rows only and close the
    // anti-resonant half with the dagger (p_contour.hpp section 3).
    auto tr = tilted_contour::build_transform(ctx.c, ztarg, /*with_mirror*/ false);
    app_log(2, "[TC-2-a(ii)] transform: {} targets x {} nodes; LS cond = {:.3e}; "
               "worst LS relative residual = {:.3e}",
            tr.F.shape()[0], tr.F.shape()[1], tr.cond, tr.relres_max);

    // ---- the DERIVED per-fixture budget --------------------------------
    // The same transform applied to a SCALAR pole model built on THIS fixture's
    // own transition support (every occupied/empty pair of the QP spectrum,
    // uniform weight), scored against its closed form. That is the accuracy the
    // contour can deliver here; the matrix-valued measurement is reported
    // against it as well as against the spec's absolute gate.
    double budget = 0.0;
    {
      std::vector<double> occ, emp;
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk_ibz; ++k)
          for (long n = 0; n < nbnd; ++n) {
            const double e = eigval(s, k, n);
            (e < mu ? occ : emp).push_back(e);
          }
      std::vector<double> D;
      D.reserve(occ.size() * emp.size());
      for (double a : emp)
        for (double i2 : occ) D.push_back(a - i2);
      const double wgt = 1.0 / double(D.size());
      const ComplexType rot = std::exp(ComplexType(0.0, -ctx.c.g.theta));
      nda::array<ComplexType, 1> Pcont(r);
      for (long j = 0; j < r; ++j) {
        ComplexType acc(0.0, 0.0);
        for (double d : D)
          acc += wgt * std::exp(ComplexType(0.0, -1.0) * d * ctx.c.s(j) * rot);
        Pcont(j) = acc;
      }
      for (long t = 0; t < long(keep.size()); ++t) {
        ComplexType got(0.0, 0.0);
        for (long j = 0; j < r; ++j) got += tr.F(t, j) * Pcont(j);
        ComplexType Rex(0.0, 0.0);
        for (double d : D) Rex += wgt / (ztarg(t) - d);
        const ComplexType Pgot = -(got + std::conj(got));   // eq (SIGN)
        const ComplexType Pex = -(Rex + std::conj(Rex));
        budget = std::max(budget, std::abs(Pgot - Pex) / std::abs(Pex));
      }
    }

    // ---- the measurement ------------------------------------------------
    double frob_worst = 0.0, cell_worst = 0.0;
    long wq = -1, wn = -1;
    nda::matrix<ComplexType> Rz(Naux, Naux), Pz(Naux, Naux);
    for (long iq = 0; iq < Nq_ibz; ++iq) {
      for (long t = 0; t < long(keep.size()); ++t) {
        Rz() = ComplexType(0.0);
        for (long j = 0; j < r; ++j)
          for (long P = 0; P < Naux; ++P)
            for (long Q = 0; Q < Naux; ++Q)
              Rz(P, Q) += tr.F(t, j) * sPc.local()(iq, j, P, Q);
        pc::polarization_from_contour(Rz, Rz, Pz);      // Pi = -[R + R^dag], eq (SIGN)
        double fn = 0.0, fd = 0.0, cn = 0.0, cd = 0.0;
        const long n = keep[std::size_t(t)];
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q) {
            const ComplexType ex = Pi_im_qwPQ(iq, n, P, Q);
            const double d = std::abs(Pz(P, Q) - ex);
            fn += d * d;
            fd += std::norm(ex);
            cn = std::max(cn, d);
            cd = std::max(cd, std::abs(ex));
          }
        const double fr = std::sqrt(fn / std::max(fd, 1e-300));
        if (fr > frob_worst) { frob_worst = fr; wq = iq; wn = n; }
        cell_worst = std::max(cell_worst, cn / std::max(cd, 1e-300));
      }
    }
    app_log(2, "[TC-2-a(ii)] P(i nu) THROUGH the tilted transform vs production "
               "Pi(q, i nu), nu >= gamma: worst Frobenius rel = {:.3e} (q = {}, "
               "nu index {}); worst cell dev / max|Pi_q| = {:.3e}; DERIVED budget "
               "from this fixture's own transition support = {:.3e} "
               "(measured/budget = {:.2f}); gate {:.1e}",
            frob_worst, wq, wn, cell_worst, budget, frob_worst / budget, gate_ii);
    REQUIRE(frob_worst < gate_ii);
    REQUIRE(cell_worst < gate_ii);

    // =================================================================
    //  TC-3-b(0) -- W ON THE LINE, at fixture scale.
    //  The chain the CD assembly actually rides: contour Pi(q, s_j)
    //  -> transform to z = i nu -> eq (SIGN) -> CoQuI's Dyson chain with
    //  Z = thc.Z(q) -> W^c(q, i nu), against the PRODUCTION W^c(q, i nu)
    //  that the imaginary-axis solver stored in dW_qtPQ. This is Fable
    //  review point 1 -- "assembled in ONE consistent convention" -- at
    //  fixture scale, and it needs none of the mode-A band bookkeeping.
    // =================================================================
    {
      // the production W^c(q, i nu) from the SAME update_w that ran above
      const long nt_half = mb_state.dW_qtPQ.value().global_shape()[1];
      nda::array<ComplexType, 4> Wc_im(Nq_ibz, nw_half, Naux, Naux);
      {
        auto const &dW = mb_state.dW_qtPQ.value();
        nda::array<ComplexType, 4> Wt(nt_half, Nq_ibz, Naux, Naux);
        Wt() = ComplexType(0.0);
        for (long iq = 0; iq < Nq_ibz; ++iq)
          for (long it = 0; it < nt_half; ++it)
            if (iq >= dW.local_range(0).first() and iq < dW.local_range(0).last()
                and it >= dW.local_range(1).first() and it < dW.local_range(1).last())
              Wt(it, iq, all, all) =
                  dW.local()(iq - dW.local_range(0).first(),
                             it - dW.local_range(1).first(), all, all);
        mpi_context->comm.all_reduce_in_place_n(Wt.data(), Wt.size(), std::plus<>{});
        nda::array<ComplexType, 4> Ww(nw_half, Nq_ibz, Naux, Naux);
        ft.tau_to_w_PHsym(Wt, Ww);
        for (long iq = 0; iq < Nq_ibz; ++iq)
          for (long n = 0; n < nw_half; ++n)
            Wc_im(iq, n, all, all) = Ww(n, iq, all, all);
      }

      double worst = 0.0, cond_worst = 0.0;
      long wq2 = -1, wn2 = -1;
      nda::array<ComplexType, 2> Pz(Naux, Naux);
      nda::matrix<ComplexType> Wg(Naux, Naux), Rz2(Naux, Naux);
      methods::wc_line::solve_stats_t wst;
      for (long iq = 0; iq < Nq_ibz; ++iq) {
        auto Zq = thc.Z(int(iq));
        for (long t = 0; t < long(keep.size()); ++t) {
          Rz2() = ComplexType(0.0);
          for (long j = 0; j < r; ++j)
            for (long P = 0; P < Naux; ++P)
              for (long Q = 0; Q < Naux; ++Q)
                Rz2(P, Q) += tr.F(t, j) * sPc.local()(iq, j, P, Q);
          pc::polarization_from_contour(Rz2, Rz2, Pz);      // eq (SIGN)
          methods::wc_line::dyson_wc_line(Zq, Pz, Wg, &wst);
          const long n = keep[std::size_t(t)];
          double num = 0.0, den = 0.0;
          for (long P = 0; P < Naux; ++P)
            for (long Q = 0; Q < Naux; ++Q) {
              const ComplexType ex = Wc_im(iq, n, P, Q);
              num += std::norm(Wg(P, Q) - ex);
              den += std::norm(ex);
            }
          const double fr = std::sqrt(num / std::max(den, 1e-300));
          if (fr > worst) { worst = fr; wq2 = iq; wn2 = n; }
        }
      }
      cond_worst = wst.cond_hint;
      app_log(2, "[TC-3-b(0)] W^c(q, i nu) via contour Pi -> eq (SIGN) -> CoQuI Dyson "
                 "chain, vs the PRODUCTION W^c from dW_qtPQ, nu >= gamma: worst "
                 "Frobenius rel = {:.3e} (q = {}, nu index {}) over {} q x {} targets; "
                 "max |[I - Z.Pi]^-1| = {:.3e}",
              worst, wq2, wn2, Nq_ibz, keep.size(), cond_worst);
      REQUIRE(worst < 1e-3);
    }

    // =================================================================
    //  TC-2-c -- band truncation along the contour
    // =================================================================
    {
      auto sPt = math::shm::make_shared_array<Array_view_4D_t>(
          *mpi_context, {Nq_ibz, r, Naux, Naux});
      pc::ctx_t dgt;
      pc::sample_P_at_times(sPt, ctx.t_node, thc, sMO, sE, mu, beta, o.eps, &dgt);
      double num = 0.0, den = 0.0;
      for (long iq = 0; iq < Nq_ibz; ++iq)
        for (long j = 0; j < r; ++j)
          for (long P = 0; P < Naux; ++P)
            for (long Q = 0; Q < Naux; ++Q) {
              num = std::max(num, std::abs(sPt.local()(iq, j, P, Q)
                                         - sPc.local()(iq, j, P, Q)));
              den = std::max(den, std::abs(sPc.local()(iq, j, P, Q)));
            }
      app_log(2, "[TC-2-c] band truncation at rel tol {:.0e}: max dev / max|Pi| = "
                 "{:.3e}; kept bands per (s,k,node) -- greater leg [{}, {}], lesser "
                 "leg [{}, {}] of {}; band-work speedup {:.3f}x (untruncated leg "
                 "counts [{}, {}] / [{}, {}], ratio {:.3f})",
              o.eps, num / std::max(den, 1e-300), dgt.nband_p_min, dgt.nband_p_max,
              dgt.nband_n_min, dgt.nband_n_max, nbnd, dgt.trunc_ratio,
              dg.nband_p_min, dg.nband_p_max, dg.nband_n_min, dg.nband_n_max,
              dg.trunc_ratio);
      REQUIRE(num / std::max(den, 1e-300) < 10.0 * o.eps);
    }
  }

  // =====================================================================
  //  qp_tc_profile = "growing" -- the eq-8 growing-delta profile.
  //  No fixture needed: the contour builder consumes only (E, mu, N_k, beta).
  //  The profile bites only once 0.05|zeta| exceeds the mesh floor
  //  1.2 W_band / N_k somewhere in the target window, i.e. once
  //  N_k >~ 1.4 W_band/eV (results section 4.2) -- so it is exercised HERE at
  //  N_k = 24 and reduces exactly to "flat" at the N_k = 1..2 of the fixtures.
  // =====================================================================
  TEST_CASE("tc_contour_growing_profile", "[methods][tc_contour]") {
    auto &mpi_context = utils::make_unit_test_mpi_context();
    if (mpi_context->comm.size() != 1) return;

    // a synthetic insulator: one 8-state valence cluster of width 0.2 Ha under mu,
    // 40 empty states from mu + 0.1 Ha up to mu + 3 Ha.
    const long nb = 48;
    nda::array<ComplexType, 3> E(1, 1, nb);
    const double mu = 0.0;
    for (long n = 0; n < 8; ++n)
      E(0, 0, n) = ComplexType(-0.2 + 0.2 * double(n) / 7.0, 0.0);
    for (long n = 8; n < nb; ++n)
      E(0, 0, n) = ComplexType(0.1 + 2.9 * double(n - 8) / double(nb - 9), 0.0);

    pc::opts_t o;
    o.eps = 1e-6;
    o.rho = 0.65;
    o.zeta_max = 10.0 / pc::ha_to_eV;
    o.nx = 1200;
    const double beta = 1000.0;

    long rank_flat = 0, rank_grow = 0;
    for (long nk_lin : {2L, 24L}) {
      o.profile = "flat";
      auto cf = pc::build_contour_for_spectrum(E, mu, nk_lin, beta, o);
      o.profile = "growing";
      auto cg = pc::build_contour_for_spectrum(E, mu, nk_lin, beta, o);
      app_log(2, "[TC-2 growing] N_k = {:>2}: delta_mesh = {:.4g} eV; flat  tan(th) = "
                 "{:.4g}, gamma = {:.4g}, S = {:.4g}, rank = {}",
              nk_lin, cf.geom.delta_mesh * pc::ha_to_eV, cf.c.g.tan_theta,
              cf.c.g.gamma, cf.c.g.S, cf.c.rank);
      app_log(2, "[TC-2 growing] N_k = {:>2}: {:>27} growing tan(th) = {:.4g}, gamma = "
                 "{:.4g}, S = {:.4g}, rank = {} -> gain {:.3f}x",
              nk_lin, "", cg.c.g.tan_theta, cg.c.g.gamma, cg.c.g.S, cg.c.rank,
              double(cf.c.rank) / double(cg.c.rank));
      REQUIRE(cg.c.rank > 0);
      REQUIRE(cg.c.g.gamma > 0.0);
      // a(x) must stay positive over the whole adapted grid -- the PSD-Gram premise
      for (long i = 0; i < cg.c.a.size(); ++i) REQUIRE(cg.c.a(i) > 0.0);
      if (nk_lin == 2) {
        // the mesh floor dominates everywhere in the window: growing == flat
        REQUIRE(cg.c.rank == cf.c.rank);
        REQUIRE(std::abs(cg.c.g.tan_theta - cf.c.g.tan_theta) < 1e-12);
      } else {
        rank_flat = cf.c.rank;
        rank_grow = cg.c.rank;
      }
    }
    // at N_k = 24 the profile must actually buy something (results section 4.2
    // measured 1.25-2.79x there); never worse than flat.
    REQUIRE(rank_grow <= rank_flat);
    app_log(2, "[TC-2 growing] VERDICT: at N_k = 24 the growing profile takes the rank "
               "from {} to {} ({:.3f}x); at N_k = 2 it is identical to flat, which is "
               "the mechanism of results section 4.2 (the profile bites only once "
               "0.05|zeta| exceeds the mesh floor inside the target window)",
            rank_flat, rank_grow, double(rank_flat) / double(rank_grow));
  }

  TEST_CASE("tc_contour_p_lih222", "[methods][tc_contour]") {
    run_tc2_gate("qe_lih222", 1e-4);
  }

  // The spec asks for a "qe_si222-class" fixture. The Si fixtures the repo ships
  // are qe_si111 / qe_si211 / qe_si222_so (spin-orbit); qe_si211 is the one the
  // RW-1 Lehmann gate uses for its Si leg, so it is the like-for-like choice.
  // [DEVIATION, flagged in notes/tc12_report.md]
  TEST_CASE("tc_contour_p_si211", "[methods][tc_contour]") {
    run_tc2_gate("qe_si211", 1e-4);
  }

  // THE TIME-REVERSAL LEG. Its purpose is to exercise eq (T) -- the TRANSPOSE fill of
  // the trev-paired FBZ points, the first of the two conj-vs-transpose traps of
  // p_contour.hpp section 1 -- which is a NO-OP on any mesh with kp_trev_pairs = 0.
  // Surveyed across the shipped fixtures: qe_lih222, qe_lih222_sym, qe_lih223,
  // qe_si111, qe_si211, qe_svo222_sym and qe_GaAs222_hf all have 0 trev pairs;
  // qe_lih223_sym (nk 12 -> 6 IBZ) and qe_lih223_inv (12 -> 8) have 4 each.
  // qe_lih223_sym also exercises the symmetry rotation (nk_ibz < nk) at the same time.
  TEST_CASE("tc_contour_p_lih223_sym", "[methods][tc_contour]") {
    run_tc2_gate("qe_lih223_sym", 1e-4);
  }

} // namespace bdft_tests
