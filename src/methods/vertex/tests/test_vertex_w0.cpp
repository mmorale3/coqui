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
 * STATIC VERTEX (B-S / B-L), increment S2 -- the W0[G] rung infrastructure
 * (notes/static_vertex_implementation_plan.md sections 2.2, 4 "S2", 5).
 *
 *   W0(q) = [1 - v P^0_RPA[G]]^{-1} v  at  i.nu = 0  =  Z(q) + dW(q, i.nu = 0)
 *
 * built inside update_w from the SAME-ITERATION RPA polarizability, before any Pi^C is
 * added (decision D2: no iteration lag, no lag cache). This file is the S2 gate:
 *
 *  1. vertex_w0_transform_row -- nu0_transform_row reproduces index 0 of
 *     IAFT::tau_to_w_PHsym EXACTLY (the verified static-slice convention of
 *     gf2_t::get_static_W / dW0, thc_gf2.icc:239-247).
 *  2. vertex_w0_selfslice -- GATE (i): on a plain-GW state (P = RPA only) the built W0
 *     equals the i.nu = 0 row of the FULLY SOLVED W(q, i.nu) to machine precision. This
 *     is Eq. selfslice of the theory notes: in B-S the two coincide exactly (in B-L they
 *     do not, because the run's own W then carries P^{C,L}).
 *  3. vertex_w0_head_policies -- GATE (ii): all three vertex_div_treatment values at
 *     i.nu = 0. "v1_skip" and "ignore_g0" store the same regularized body (and differ
 *     only in the kernel-side Gamma skip flag); the gygi class adds EXACTLY the analytic
 *     rank-1 head Nk*xi_M*[1 + Re eps^-1_head(i.nu=0)]*chi chi^dag at Gamma and nothing
 *     anywhere else.
 *  4. vertex_w0_update_w_seam -- the production seam: a STATIC-rung vertex gets W0/W0bar
 *     built by update_w automatically, the resulting RPA W is BIT-IDENTICAL to the
 *     vertex-free one (no bootstrap, no Pi^C injection -- plan sections 2.1/2.2), and
 *     nothing is retained across the iteration boundary.
 *  5. vertex_w0_fold -- GATE (iv): W0bar == the replicated t(q) W0(q) t(q)^dag, in the
 *     secondary path (Refinement 2) and in the global-aux reference path (t = identity,
 *     N_m == Np).
 *
 * GATE (iii) -- the forced-(P,Q)-split distributed build vs a replicated reference -- is
 * the vertex_w0_row_fold_distributed case of test_vertex_dfold.cpp (per the plan's test
 * table: "vertex_dfold (W0 fold forced-(P,Q)-split case -- S2"), where the (P,Q) grid can
 * be forced independently of the physics.
 */

#undef NDEBUG

#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"

#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "mean_field/default_MF.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include "methods/vertex/vertex_t.h"

namespace bdft_tests {

  using namespace methods;
  static const auto w0_all = nda::range::all;

  // ---- small gather helpers (the arrays are toy-sized; the zero-pad + all_reduce over a
  // PARTITION is an exact gather -- no floating-point reassociation) --------------------
  template<typename dArr3_t, typename comm_t>
  static nda::array<ComplexType, 3> gather3(dArr3_t const& dA, comm_t& comm) {
    auto gs = dA.global_shape();
    nda::array<ComplexType, 3> out(gs[0], gs[1], gs[2]);
    out() = ComplexType(0.0);
    out(dA.local_range(0), dA.local_range(1), dA.local_range(2)) = dA.local();
    comm.all_reduce_in_place_n(out.data(), out.size(), std::plus<>{});
    return out;
  }

  // gather the i0-th slice of the leading axis of a (n0, n1, n2, n3) distributed array
  template<typename dArr4_t, typename comm_t>
  static nda::array<ComplexType, 3> gather4_slice(dArr4_t const& dA, long i0, comm_t& comm) {
    auto gs = dA.global_shape();
    nda::array<ComplexType, 3> out(gs[1], gs[2], gs[3]);
    out() = ComplexType(0.0);
    const long o0 = dA.origin()[0], l0 = dA.local_shape()[0];
    if (i0 >= o0 and i0 < o0 + l0)
      out(dA.local_range(1), dA.local_range(2), dA.local_range(3)) =
          dA.local()(i0 - o0, nda::ellipsis{});
    comm.all_reduce_in_place_n(out.data(), out.size(), std::plus<>{});
    return out;
  }

  static double max_abs3(nda::array<ComplexType, 3> const& a) {
    double m = 0.0;
    for (auto const& v : a) m = std::max(m, std::abs(v));
    return m;
  }
  static double max_diff3(nda::array<ComplexType, 3> const& a,
                          nda::array<ComplexType, 3> const& b) {
    double m = 0.0;
    auto ia = a.begin();
    auto ib = b.begin();
    for (; ia != a.end(); ++ia, ++ib) m = std::max(m, std::abs(*ia - *ib));
    return m;
  }

  // ====================================================================================
  // 1. the i.nu = 0 transform row IS index 0 of tau_to_w_PHsym (pure IAFT algebra)
  TEST_CASE("vertex_w0_transform_row", "[methods][vertex][w0]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_w0_transform_row skipped: build has ENABLE_DLR=OFF.");
#else
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    const long nt_b = ft.nt_b(), nw_b = ft.nw_b();
    const long nt_half = (nt_b % 2 == 0) ? nt_b / 2 : nt_b / 2 + 1;
    const long nw_half = (nw_b % 2 == 0) ? nw_b / 2 : nw_b / 2 + 1;

    auto R = solvers::vertex_w0_detail::nu0_transform_row(ft);
    REQUIRE(R.shape(0) == nt_half);

    // deterministic tau-domain data with a few "columns" so the row is exercised on more
    // than one trailing index.
    const long ncol = 5;
    nda::array<ComplexType, 2> A_ti(nt_half, ncol), A_wi(nw_half, ncol);
    for (long it = 0; it < nt_half; ++it)
      for (long c = 0; c < ncol; ++c)
        A_ti(it, c) = ComplexType(0.37 * std::cos(0.11 * double(it) + 0.7 * double(c)),
                                  0.21 * std::sin(0.29 * double(it) - 0.3 * double(c)));
    ft.tau_to_w_PHsym(A_ti, A_wi);

    double worst = 0.0, scale = 0.0;
    for (long c = 0; c < ncol; ++c) {
      ComplexType s(0.0, 0.0);
      for (long it = 0; it < nt_half; ++it) s += R(it) * A_ti(it, c);
      worst = std::max(worst, std::abs(s - A_wi(0, c)));
      scale = std::max(scale, std::abs(A_wi(0, c)));
    }
    app_log(1, "vertex_w0_transform_row: |R.A(tau) - tau_to_w_PHsym(A)(0)|_max = {:.3e} "
               "(scale {:.3e}); nt_half = {}, nw_half = {}", worst, scale, nt_half, nw_half);
    REQUIRE(scale > 1e-8);
    REQUIRE(worst <= 1e-14 * scale);
#endif
  }

  // ====================================================================================
  // INCREMENT S4: the tau = 0 ROW (plan section 2.4; decision D3 resolved DLR-only).
  //
  // Gate: against an ANALYTIC reference. A pure exponential A(tau) = e^{-E tau} has a
  // delta spectral function at E, so it is DLR-representable to eps for |E| < wmax, and
  // its equal-time value is known exactly: A(tau = 0) = 1 (and sum_p c_p for a
  // superposition). Two independent legs:
  //   (a) the pure tau-space interpolation row at tau = 0 (tau = 0 is NOT a grid node);
  //   (b) the COMPOSED row R_nu, applied to the BOSONIC Matsubara representation --
  //       i.e. the legal evaluation of (1/beta) sum_nu A(i.nu), which is what
  //       Pi^{C,0}(tau = 0) needs. A plain sum over the sampled nu nodes is NOT this
  //       (sparse nodes are fitting nodes, not Fourier points) and is checked to differ.
  TEST_CASE("vertex_w0_tau0_row", "[methods][vertex][w0]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_w0_tau0_row skipped: build has ENABLE_DLR=OFF.");
#else
    const double beta = 20.0, wmax = 6.0;
    // Swept over the DLR precisions: the row is EXACT by construction, so whatever error
    // remains is the basis' own representation error of the analytic probe and must fall
    // as eps tightens. That is the discriminator between "the row is right" and "the row
    // is buggy" -- a construction bug would NOT improve with eps.
    std::string prec = GENERATE(std::string("low"), std::string("medium"),
                                std::string("high"));
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, prec);
    const long nt_b = ft.nt_b(), nw_b = ft.nw_b();

    auto R = solvers::vertex_w0_detail::tau0_transform_row(ft);
    REQUIRE(R.shape(0) == nw_b);

    // Probes must be GENUINE BOSONIC correlators, i.e. beta-PERIODIC. The single-mode
    // bosonic propagator
    //     A_E(tau) = (1 + n_B(E)) e^{-E tau} + n_B(E) e^{+E tau},   A_E(0) = 1 + 2 n_B(E)
    // satisfies A_E(beta) = A_E(0) exactly and has a delta spectral function at +-E, so
    // it is DLR-representable to eps for |E| < wmax. (A bare e^{-E tau} is NOT periodic;
    // its bosonic Matsubara representation has a slowly decaying tail that the sparse
    // bosonic nodes cannot capture, and leg (b) then saturates around 3e-7 while leg (a)
    // keeps converging -- measured, and the reason this probe is the periodic one.)
    const std::vector<double> Es = {0.5, 1.3, 2.1};
    const std::vector<double> cs = {1.0, 0.6, -0.3};
    auto tau_rel = ft.tau_mesh_b();          // [-1, 1] convention
    const long ncol = long(Es.size()) + 1;   // one column per E, plus the superposition
    nda::array<ComplexType, 2> A_ti(nt_b, ncol), A_wi(nw_b, ncol);
    nda::array<ComplexType, 1> exact(ncol);
    exact() = ComplexType(0.0);
    for (long it = 0; it < nt_b; ++it) {
      const double tau = (tau_rel(it) + 1.0) * 0.5 * beta;
      ComplexType sup(0.0);
      for (long e = 0; e < long(Es.size()); ++e) {
        const double E = Es[size_t(e)];
        const double nB = 1.0 / std::expm1(beta * E);
        A_ti(it, e) = ComplexType((1.0 + nB) * std::exp(-E * tau)
                                  + nB * std::exp(E * tau), 0.0);
        sup += cs[size_t(e)] * A_ti(it, e);
      }
      A_ti(it, ncol - 1) = sup;
    }
    for (long e = 0; e < long(Es.size()); ++e) {
      const double nB = 1.0 / std::expm1(beta * Es[size_t(e)]);
      exact(e) = ComplexType(1.0 + 2.0 * nB, 0.0);
      exact(ncol - 1) += cs[size_t(e)] * exact(e);
    }

    // (a) pure tau-space interpolation at tau = 0
    nda::array<double, 1> t0(1);
    t0(0) = -1.0;
    auto R0 = ft.construct_tau_interpolate_matrix(t0);
    double worst_a = 0.0;
    nda::array<ComplexType, 1> val_a(ncol);
    for (long c = 0; c < ncol; ++c) {
      ComplexType acc(0.0);
      for (long it = 0; it < nt_b; ++it) acc += R0(0, it) * A_ti(it, c);
      val_a(c) = acc;
      worst_a = std::max(worst_a, std::abs(acc - exact(c)) / std::abs(exact(c)));
    }

    // (b) the composed row on the BOSONIC Matsubara representation
    ft.tau_to_w(A_ti, A_wi, imag_axes_ft::boson);
    double worst_b = 0.0, worst_naive = 0.0, worst_compose = 0.0;
    for (long c = 0; c < ncol; ++c) {
      ComplexType acc(0.0), naive(0.0);
      for (long n = 0; n < nw_b; ++n) {
        acc   += R(n) * A_wi(n, c);
        naive += A_wi(n, c) / beta;          // the ILLEGAL plain sum over sampled nodes
      }
      worst_b       = std::max(worst_b,     std::abs(acc   - exact(c)) / std::abs(exact(c)));
      worst_naive   = std::max(worst_naive, std::abs(naive - exact(c)) / std::abs(exact(c)));
      worst_compose = std::max(worst_compose,
                               std::abs(acc - val_a(c)) / std::abs(exact(c)));
    }
    app_log(1, "vertex_w0_tau0_row [prec = {}]: nt_b = {}, nw_b = {}; rel err "
               "(a) tau-interp = {:.3e}, (b) composed row on i.nu = {:.3e}; the illegal "
               "plain (1/beta) sum_nu over SAMPLED nodes = {:.3e}; compose = {:.3e}",
            prec, nt_b, nw_b, worst_a, worst_b, worst_naive, worst_compose);
    // (i) the COMPOSITION is exact: the row applied to the Matsubara data reproduces the
    //     pure tau-space interpolation VALUE (not merely a similar error magnitude).
    //     This is the part that is ours, and it must hold at every precision.
    REQUIRE(worst_compose < 1e-10);
    // (ii) the residual is the BASIS error of the analytic probe and must track eps.
    const double tol = (prec == "low") ? 1e-7 : (prec == "medium" ? 1e-9 : 1e-12);
    REQUIRE(worst_a < tol);
    REQUIRE(worst_b < tol);
    // (iii) the plain sum over SPARSE nodes must be visibly wrong -- the trap the row
    //       exists to avoid (sparse nodes are fitting nodes, not Fourier points).
    REQUIRE(worst_naive > 1e-3);
#endif
  }

#ifdef ENABLE_DLR
  // The cases below each stand up their own physical state: one plain scGW iteration on
  // LiH-222, so mb_state carries a consistent G (the same isolation the refinement2 /
  // conservation / wcache tests use).

  // ====================================================================================
  // 2. GATE (i): plain-GW SELF-SLICE identity (Eq. selfslice)
  TEST_CASE("vertex_w0_selfslice", "[methods][vertex][w0]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    // wmax = 6.0: the vertex [A-comp] headroom requirement (pi design section 4b)
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_w0_selfslice";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    solvers::hf_t hf;
    solvers::gw_t gw(&ft, "ignore_g0", output);
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
    simple_dyson dyson(mf.get(), &ft);
    MBState mb_state(mpi_context, ft, output);
    iter_scf::iter_scf_t iter_sol("damping");
    auto [e_hf_0, e_corr_0] = scf_loop(mb_state, dyson, eri, ft,
                                       solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                       1, false, 1e-9, true);
    mpi_context->comm.barrier();
    REQUIRE(std::isfinite(e_hf_0));
    REQUIRE(std::isfinite(e_corr_0));

    const long nqpts_ibz = thc.MF()->nqpts_ibz();
    const long Np = thc.Np();
    app_log(1, "vertex_w0_selfslice: LiH-222 plain scGW iteration e_hf = {:.12f}, "
               "e_corr = {:.12f}; nq_ibz = {}, Np = {}, ranks = {}",
            e_hf_0, e_corr_0, nqpts_ibz, Np, mpi_context->comm.size());

    // ---- Pi_RPA(q, tau) of the current G (no vertex attached => pure RPA) ------------
    auto dPi_rpa = scr_eri.eval_Pi_qdep(mb_state, thc);

    // ---- the object under test: W0 from that SAME Pi_RPA -----------------------------
    // built on a DYNAMIC-rung vertex on purpose: build_w0 is mode-agnostic infrastructure
    // (the mode gate is needs_w0() at the update_w seam, exercised separately below), and
    // the S1 "kernels not implemented" abort of the static modes still stands at the
    // kernel entry points.
    solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(1, 3), mf->nbnd(),
                          "ignore_g0", "global");
    REQUIRE(vtx.active());
    REQUIRE(not vtx.has_w0());
    vtx.build_w0(mb_state, thc, dPi_rpa);
    REQUIRE(vtx.has_w0());
    REQUIRE(not vtx.w0_head_applied());          // ignore_g0: body only
    REQUIRE(not vtx.w0_skip_gamma());
    auto W0 = gather3(vtx.W0_qPQ(), mpi_context->comm);
    REQUIRE(W0.shape(0) == nqpts_ibz);
    REQUIRE(W0.shape(1) == Np);

    // ---- reference: the i.nu = 0 row of the FULLY SOLVED W(q, i.nu) ------------------
    // dyson_W_from_Pi_tau<true> is the production W solve with the Matsubara output kept
    // (tau -> nu on the FULL bosonic half mesh, then the Np x Np Dyson at every nu). Its
    // index-0 row is W(q, i.nu = 0) - Z(q) by the same convention gf2_t::get_static_W
    // pins. reset_input = false keeps Pi_RPA alive (nothing else reads it here, but the
    // ordering assumption -- W0 built from an untouched RPA Pi -- is what matters).
    auto dW_w = scr_eri.dyson_W_from_Pi_tau<true>(dPi_rpa, thc, false);
    auto dW_nu0 = gather4_slice(dW_w, 0, mpi_context->comm);
    nda::array<ComplexType, 3> W_ref(nqpts_ibz, Np, Np);
    for (long iq = 0; iq < nqpts_ibz; ++iq) {
      nda::array<ComplexType, 2> Zq = thc.Z(int(iq));       // collective: all ranks, all q
      W_ref(iq, w0_all, w0_all) = Zq + dW_nu0(iq, w0_all, w0_all);
    }

    const double scale = max_abs3(W_ref);
    const double worst = max_diff3(W0, W_ref);
    app_log(1, "vertex_w0_selfslice: max|W0| = {:.6e}, max|Z + dW(i.nu=0)| = {:.6e}, "
               "max|W0 - ref| = {:.3e}  (rel {:.3e})",
            max_abs3(W0), scale, worst, worst / std::max(scale, 1e-300));
    REQUIRE(scale > 1e-8);
    // MACHINE precision: the two paths run the SAME single-frequency Dyson on the SAME
    // i.nu = 0 polarizability row; they differ only in the order the tau sum and the
    // distributed gemms are accumulated (the W0 build uses the {1, nP, nQ} layout, the
    // reference the W_omega_proc_grid one), i.e. by FP reassociation only.
    REQUIRE(worst <= 1e-11 * scale);

    if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
    mpi_context->comm.barrier();
  }

  // ====================================================================================
  // 3. GATE (ii): the three q->0 head policies at i.nu = 0
  TEST_CASE("vertex_w0_head_policies", "[methods][vertex][w0]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_w0_head";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    solvers::hf_t hf;
    solvers::gw_t gw(&ft, "ignore_g0", output);
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
    simple_dyson dyson(mf.get(), &ft);
    MBState mb_state(mpi_context, ft, output);
    iter_scf::iter_scf_t iter_sol("damping");
    auto [e_hf_0, e_corr_0] = scf_loop(mb_state, dyson, eri, ft,
                                       solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                       1, false, 1e-9, true);
    mpi_context->comm.barrier();
    REQUIRE(std::isfinite(e_corr_0));
    (void)e_hf_0;

    const long nqpts_ibz = thc.MF()->nqpts_ibz();
    const long nkpts = thc.MF()->nkpts();
    const long Np = thc.Np();
    REQUIRE(nqpts_ibz > 1);        // otherwise the gygi policy is legitimately downgraded

    auto dPi_rpa = scr_eri.eval_Pi_qdep(mb_state, thc);

    // q = Gamma (crystal coordinates), and the rank-1 head data
    long iq_gamma = -1;
    {
      auto Qpts = thc.MF()->Qpts();
      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        double d = 0.0;
        for (long i = 0; i < 3; ++i) d += std::abs(Qpts(iq, i) - std::round(Qpts(iq, i)));
        if (d < 1e-8) iq_gamma = iq;
      }
    }
    REQUIRE(iq_gamma >= 0);
    const double xi = thc.MF()->madelung();
    auto chi = thc.basis_head();
    app_log(1, "vertex_w0_head_policies: iq_gamma = {}, madelung = {:.12f}", iq_gamma, xi);
    REQUIRE(std::abs(xi) > 0.0);

    auto build = [&](std::string const& div) {
      solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(1, 3), mf->nbnd(),
                            div, "global");
      REQUIRE(vtx.active());
      vtx.build_w0(mb_state, thc, dPi_rpa);
      auto W0 = gather3(vtx.W0_qPQ(), mpi_context->comm);
      return std::make_tuple(W0, vtx.w0_head_applied(), vtx.w0_eps_inv_head(),
                             vtx.w0_head_c(), vtx.w0_skip_gamma());
    };

    auto [W_skip, head_skip, h_skip, c_skip, gskip_skip] = build("v1_skip");
    auto [W_ig, head_ig, h_ig, c_ig, gskip_ig] = build("ignore_g0");
    auto [W_gygi, head_gygi, h_gygi, c_gygi, gskip_gygi] = build("gygi");
    (void)h_skip; (void)h_ig; (void)c_skip; (void)c_ig;

    // v1_skip and ignore_g0 store the SAME regularized body -- the v1 fallback acts in
    // the kernel (the Gamma cell of the rung transfer is skipped there), not on the
    // stored array. The only difference is the flag the S3+ kernels read off the handle.
    REQUIRE(not head_skip);
    REQUIRE(not head_ig);
    REQUIRE(gskip_skip);
    REQUIRE(not gskip_ig);
    const double d_skip_ig = max_diff3(W_skip, W_ig);
    app_log(1, "vertex_w0_head_policies: max|W0(v1_skip) - W0(ignore_g0)| = {:.3e} "
               "(bitwise expected)", d_skip_ig);
    REQUIRE(d_skip_ig == 0.0);

    // gygi: EXACTLY the analytic rank-1 head at Gamma, nothing elsewhere.
    REQUIRE(head_gygi);
    REQUIRE(gskip_gygi == false);
    const ComplexType c_head = c_gygi;
    app_log(1, "vertex_w0_head_policies: gygi head at i.nu = 0: Nk*madelung = {:.6e}, "
               "Re[eps^-1_head(i.nu=0) - 1] = {:.6e}, epsilon_inf(RPA, W0) = {:.6f}",
            c_head.real(), h_gygi, 1.0 / (1.0 + h_gygi));
    REQUIRE(std::abs(c_head - ComplexType(double(nkpts) * xi)) == 0.0);
    REQUIRE(h_gygi < 0.0);         // eps^-1_head - 1 < 0 for a screening system
    REQUIRE(h_gygi > -1.0);

    double worst_gamma = 0.0, worst_other = 0.0, head_max = 0.0;
    for (long iq = 0; iq < nqpts_ibz; ++iq)
      for (long ip = 0; ip < Np; ++ip)
        for (long jq = 0; jq < Np; ++jq) {
          const ComplexType d = W_gygi(iq, ip, jq) - W_ig(iq, ip, jq);
          if (iq != iq_gamma) {
            worst_other = std::max(worst_other, std::abs(d));
          } else {
            // the two adds build_w0 performs: bare weight 1 and dynamic weight
            // Re[eps^-1_head(i.nu = 0)], each of the pinned rank-1 form
            // H_PQ = Nk*xi*conj(chi_P) chi_Q (vertex_head_detail::build_head_rank1).
            const ComplexType cp = c_head * std::conj(chi(iq_gamma, ip));
            const ComplexType h = cp * chi(iq_gamma, jq);
            const ComplexType ref = ComplexType(1.0) * h + ComplexType(h_gygi) * h;
            worst_gamma = std::max(worst_gamma, std::abs(d - ref));
            head_max = std::max(head_max, std::abs(ref));
          }
        }
    app_log(1, "vertex_w0_head_policies: gygi - ignore_g0: max|delta| off Gamma = {:.3e} "
               "(must be 0), max|delta - rank1 head| at Gamma = {:.3e} (head scale {:.3e})",
            worst_other, worst_gamma, head_max);
    REQUIRE(worst_other == 0.0);
    REQUIRE(head_max > 1e-8);
    REQUIRE(worst_gamma <= 1e-13 * std::max(head_max, 1.0));

    if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
    mpi_context->comm.barrier();
  }

  // ====================================================================================
  // 4. the production update_w seam (plan sections 2.1/2.2/2.3)
  TEST_CASE("vertex_w0_update_w_seam", "[methods][vertex][w0]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_w0_seam";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    solvers::hf_t hf;
    solvers::gw_t gw(&ft, "ignore_g0", output);
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
    simple_dyson dyson(mf.get(), &ft);
    MBState mb_state(mpi_context, ft, output);
    iter_scf::iter_scf_t iter_sol("damping");
    auto [e_hf_0, e_corr_0] = scf_loop(mb_state, dyson, eri, ft,
                                       solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                       1, false, 1e-9, true);
    mpi_context->comm.barrier();
    REQUIRE(std::isfinite(e_corr_0));
    (void)e_hf_0;

    const long nqpts_ibz = thc.MF()->nqpts_ibz();
    const long Np = thc.Np();

    // ---- A: plain update_w, no vertex attached --------------------------------------
    scr_eri.update_w(mb_state, thc, -1);
    REQUIRE(mb_state.dW_qtPQ.has_value());
    const long nt_half = mb_state.dW_qtPQ.value().global_shape()[1];
    nda::array<ComplexType, 4> dW_ref(nqpts_ibz, nt_half, Np, Np);
    {
      auto const& dW = mb_state.dW_qtPQ.value();
      dW_ref() = ComplexType(0.0);
      dW_ref(dW.local_range(0), dW.local_range(1), dW.local_range(2), dW.local_range(3)) =
          dW.local();
      mpi_context->comm.all_reduce_in_place_n(dW_ref.data(), dW_ref.size(), std::plus<>{});
    }

    // ---- B: the same update_w with a STATIC-rung vertex attached ---------------------
    // Mode wiring (plan section 2.1): no bootstrap (a physical static rung exists from
    // iteration 1 by construction), no Pi^C injection (B-S has P = RPA), no W-bar cache
    // and no dW retention. So the screened interaction must come out BIT-IDENTICAL and
    // the only new state is W0 / W0bar.
    solvers::vertex_t vstat(&ft, "2nd_exchange", nda::range(1, 3), mf->nbnd(),
                            "ignore_g0", "global", -1, 1e-8, -1.0, -1.0, "static");
    REQUIRE(vstat.active());
    REQUIRE(vstat.rung() == solvers::static_rung);
    REQUIRE(vstat.needs_w0());
    REQUIRE(not vstat.has_w0());
    scr_eri.set_vertex(&vstat);
    REQUIRE(scr_eri.has_active_vertex());
    // static/linear: the dW-retention exception and the W-bar cache are RETIRED
    REQUIRE(not scr_eri.needs_dw_retention());

    scr_eri.update_w(mb_state, thc, -1);
    REQUIRE(vstat.has_w0());
    REQUIRE(not vstat.has_cached_w());          // the W-bar cache stays retired
    {
      auto const& dW = mb_state.dW_qtPQ.value();
      nda::array<ComplexType, 4> dW_new(nqpts_ibz, nt_half, Np, Np);
      dW_new() = ComplexType(0.0);
      dW_new(dW.local_range(0), dW.local_range(1), dW.local_range(2), dW.local_range(3)) =
          dW.local();
      mpi_context->comm.all_reduce_in_place_n(dW_new.data(), dW_new.size(), std::plus<>{});
      double worst = 0.0, scale = 0.0;
      auto ia = dW_new.begin();
      auto ib = dW_ref.begin();
      for (; ia != dW_new.end(); ++ia, ++ib) {
        worst = std::max(worst, std::abs(*ia - *ib));
        scale = std::max(scale, std::abs(*ib));
      }
      app_log(1, "vertex_w0_update_w_seam: max|dW(static vertex) - dW(no vertex)| = {:.3e} "
                 "(scale {:.3e}; BITWISE expected -- the W0 build must not touch the RPA "
                 "path)", worst, scale);
      REQUIRE(scale > 1e-8);
      REQUIRE(worst == 0.0);
    }

    // W0 built by the seam == W0 built by hand from the same RPA Pi (the seam feeds the
    // RPA-only polarizability, before any correction).
    auto W0_seam = gather3(vstat.W0_qPQ(), mpi_context->comm);
    scr_eri.set_vertex(nullptr);
    auto dPi_rpa = scr_eri.eval_Pi_qdep(mb_state, thc);
    solvers::vertex_t vman(&ft, "2nd_exchange", nda::range(1, 3), mf->nbnd(),
                           "ignore_g0", "global");
    vman.build_w0(mb_state, thc, dPi_rpa);
    auto W0_man = gather3(vman.W0_qPQ(), mpi_context->comm);
    const double d_seam = max_diff3(W0_seam, W0_man);
    app_log(1, "vertex_w0_update_w_seam: max|W0(update_w seam) - W0(by hand)| = {:.3e} "
               "(scale {:.3e})", d_seam, max_abs3(W0_man));
    REQUIRE(d_seam == 0.0);

    // ITERATION-LOCAL lifetime: reset_w0 drops both objects (called automatically at the
    // top of every build, so no stale static rung can ever be read).
    vstat.reset_w0();
    REQUIRE(not vstat.has_w0());

    // a DYNAMIC-rung vertex must NOT build W0 at all (zero new arithmetic on the
    // Formulation-B path).
    solvers::vertex_t vdyn(&ft, "2nd_exchange", nda::range(1, 3), mf->nbnd(),
                           "ignore_g0", "global");
    REQUIRE(not vdyn.needs_w0());

    if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
    mpi_context->comm.barrier();
  }

  // ====================================================================================
  // 5. GATE (iv): W0bar == the replicated t(q) W0(q) t(q)^dag
  TEST_CASE("vertex_w0_fold", "[methods][vertex][w0]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    std::string output = "coqui_vertex_w0_fold";

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    auto eri = mb_eri_t(thc, thc);

    solvers::hf_t hf;
    solvers::gw_t gw(&ft, "ignore_g0", output);
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
    simple_dyson dyson(mf.get(), &ft);
    MBState mb_state(mpi_context, ft, output);
    iter_scf::iter_scf_t iter_sol("damping");
    auto [e_hf_0, e_corr_0] = scf_loop(mb_state, dyson, eri, ft,
                                       solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                       1, false, 1e-9, true);
    mpi_context->comm.barrier();
    REQUIRE(std::isfinite(e_corr_0));
    (void)e_hf_0;

    const long nqpts_ibz = thc.MF()->nqpts_ibz();
    const long Np = thc.Np();
    auto dPi_rpa = scr_eri.eval_Pi_qdep(mb_state, thc);

    // ---- GLOBAL-aux reference path: N_m == Np, t = identity ---------------------------
    {
      solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(1, 3), mf->nbnd(),
                            "gygi", "global");
      vtx.build_w0(mb_state, thc, dPi_rpa);
      auto W0 = gather3(vtx.W0_qPQ(), mpi_context->comm);
      auto const& Wb = vtx.W0bar_qmm();
      REQUIRE(Wb.shape(0) == nqpts_ibz);
      REQUIRE(Wb.shape(1) == Np);
      double worst = 0.0;
      for (long iq = 0; iq < nqpts_ibz; ++iq)
        for (long ip = 0; ip < Np; ++ip)
          for (long jq = 0; jq < Np; ++jq)
            worst = std::max(worst, std::abs(Wb(iq, ip, jq) - W0(iq, ip, jq)));
      app_log(1, "vertex_w0_fold[global]: max|W0bar - W0| = {:.3e} (t = identity; exact "
                 "gather expected)", worst);
      REQUIRE(worst == 0.0);
    }

    // ---- SECONDARY path (Refinement 2): W0bar = t W0 t^dag ---------------------------
    {
      solvers::vertex_t vtx(&ft, "2nd_exchange", nda::range(1, 3), mf->nbnd(),
                            "gygi", "secondary", 32, 1e-8);
      REQUIRE(vtx.secondary());
      vtx.build_w0(mb_state, thc, dPi_rpa);
      const long Nm = vtx.secondary_rank();
      REQUIRE(Nm > 0);
      auto W0 = gather3(vtx.W0_qPQ(), mpi_context->comm);
      auto const& Wb = vtx.W0bar_qmm();
      auto const& t_qmP = vtx.secondary_transfer();
      REQUIRE(Wb.shape(0) == nqpts_ibz);
      REQUIRE(Wb.shape(1) == Nm);
      REQUIRE(t_qmP.shape(0) == nqpts_ibz);
      REQUIRE(t_qmP.shape(1) == Nm);
      REQUIRE(t_qmP.shape(2) == Np);

      // replicated reference: the two-gemm fold on the full Np x Np, exactly what
      // fold_Z_distributed's per-(P,Q)-block partials must sum to.
      double worst = 0.0, scale = 0.0;
      nda::array<ComplexType, 2> tmp(Nm, Np), ref(Nm, Nm);
      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        auto t_q = t_qmP(iq, w0_all, w0_all);
        nda::array<ComplexType, 2> A = W0(iq, w0_all, w0_all);
        nda::blas::gemm(t_q, A, tmp);
        nda::blas::gemm(tmp, nda::dagger(t_q), ref);
        for (long m = 0; m < Nm; ++m)
          for (long n = 0; n < Nm; ++n) {
            worst = std::max(worst, std::abs(Wb(iq, m, n) - ref(m, n)));
            scale = std::max(scale, std::abs(ref(m, n)));
          }
      }
      app_log(1, "vertex_w0_fold[secondary, N_m = {}]: max|W0bar - t W0 t^dag| = {:.3e} "
                 "(scale {:.3e}, rel {:.3e})",
              Nm, worst, scale, worst / std::max(scale, 1e-300));
      REQUIRE(scale > 1e-8);
      REQUIRE(worst <= 1e-12 * std::max(scale, 1.0));
    }

    if (mpi_context->comm.root()) remove((output + ".mbpt.h5").c_str());
    mpi_context->comm.barrier();
  }
#endif  // ENABLE_DLR

} // namespace bdft_tests
