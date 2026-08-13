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

#ifndef COQUI_WC_BAND_ELEMENTS_HPP
#define COQUI_WC_BAND_ELEMENTS_HPP

/**
 * Project 2 increment QM3: the state-resolved W^c band elements and the mode-A evaluator
 * context. Sibling of qp_modea.hpp -- read ITS header first: the momentum/spin/prefactor
 * routing, the fit-linearity argument and the trev rule are derived there, and this file is
 * their implementation.
 *
 * The build runs ONCE per outer iteration, inside the only window where W is live
 * (qp_scf_common.cpp: between update_w and the dW reset), and produces
 *
 *     M(a, b, J*npk + p)     per external (s,k) held by this rank,   J = (q', n)
 *
 * the residues of Sigma^c_ab in the pole representation of spec section 1, with the +1/nk
 * prefactor folded in. Two stages:
 *
 *  STAGE 1 (collective, per IBZ q): gather the distributed dW(q, tau) into a NODE-SHARED
 *    buffer, augment the Gamma head if the divergence treatment asks for it, transform to
 *    the bosonic Matsubara mesh, and run the SUPPORT-CONSTRAINED pole fit
 *    (imag_axes_ft::masked_pole_fit -- the promoted QM2-b chain) with the auxiliary-node
 *    columns of |eps_p| < gap_edge removed. Result: node-shared residue slabs
 *    W^(p)_PQ(q), p over the retained nodes. The Np^2 right-hand sides are partitioned over
 *    the node's ranks; the fit is elementwise in that batch axis.
 *
 *  STAGE 2 (rank-local): for each owned external (s,k), loop the qsymms/star structure of
 *    thc_gw verbatim and accumulate the two-gemm sandwich
 *        B(P,a) = conj(XCe(P,a)) u(P,n),    M^(n,p) += (1/nk) B^T W^(p) conj(B).
 *
 * Nothing here re-materializes W at (a,n,b,nu); the working set is one q's residue slab
 * (node-shared) plus this rank's (a,b,J,p) slabs.
 */

#include <chrono>
#include <format>

#include "itertools/itertools.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/sparse/sparse.hpp"
#include "numerics/shared_array/nda.hpp"
#include "methods/SCF/qp_modea.hpp"
#include "methods/mb_state/mb_state.hpp"

namespace methods {
namespace qp_modea {

  namespace detail {

    /** max|A - A^dag| / max|A| over a (nt, Np, Np) tau slab -- the trev-rule tripwire. */
    inline void herm_probe(nda::MemoryArrayOfRank<3> auto const &W_tPQ,
                           double &num, double &den) {
      auto [nt, NP, NQ] = W_tPQ.shape();
      if (NP != NQ) return;
      for (long t = 0; t < nt; ++t)
        for (long P = 0; P < NP; ++P)
          for (long Q = 0; Q < NP; ++Q) {
            num = std::max(num, std::abs(W_tPQ(t, P, Q) - std::conj(W_tPQ(t, Q, P))));
            den = std::max(den, std::abs(W_tPQ(t, P, Q)));
          }
    }

  } // detail

  /**
   * Build the mode-A evaluator context. `thc` must be a THC-ERI (the pair vectors are the
   * collocation columns); `sE_ska` / `sMO_skia` are the CURRENT outer-iteration QP spectrum
   * and MO coefficients, and both are frozen for the whole inner-consistency loop.
   *
   * @param need_diag  also build the replicated diagonal residues (the evGW leg needs to
   *                   evaluate Sigma_ii on a DIFFERENT processor grid than the block owner).
   */
  template<typename thc_t>
  void build_modea_context(modea_ctx &ctx, MBState &mb_state, thc_t &thc,
                           const sArray_t<Array_view_4D_t> &sMO_skia,
                           const sArray_t<Array_view_3D_t> &sE_ska,
                           double mu, const imag_axes_ft::IAFT &FT,
                           modea_opts const &opts, std::string const &div_treatment,
                           bool need_diag) {
    using math::shm::make_shared_array;
    decltype(nda::range::all) all;
    const auto t_start = std::chrono::steady_clock::now();

    auto mpi = thc.mpi();
    auto MF = thc.MF();
    const long NP = thc.Np();
    const long nbnd = sE_ska.shape()[2];
    const long ns = sE_ska.shape()[0];
    const long nk_ibz = sE_ska.shape()[1];
    const long nkpts = MF->nkpts();
    const long nqpts = MF->nqpts();
    const long nsym = MF->qsymms().size();
    const int lvl = opts.level;

    ctx = modea_ctx{};
    ctx.opts = opts;
    ctx.beta = FT.beta();
    ctx.mu = mu;
    ctx.eta = opts.eta;
    ctx.ns = ns;
    ctx.nk = nk_ibz;
    ctx.nbnd = nbnd;
    ctx.nkpts_full = nkpts;

    utils::check(FT.basis() == imag_axes_ft::dlr_basis,
                 "qp_modea: qp_map = \"mode_a\" requires the DLR imaginary-axis backend "
                 "(the support-constrained auxiliary pole fit lives there). Rerun with "
                 "iaft basis = \"dlr\".");
    utils::check(mb_state.dW_qtPQ.has_value(),
                 "qp_modea: mb_state.dW_qtPQ is empty -- the mode-A context must be built "
                 "inside the live W window (between update_w and the dW reset).");
    utils::check(thc.X(0, 0, 0).shape(1) == nbnd,
                 "qp_modea: the THC collocation carries {} orbitals but the QP block is {} "
                 "wide; mode_a needs the full band window.", thc.X(0, 0, 0).shape(1), nbnd);

    // ---------------- support constraint + the promoted fit ---------------------------
    imag_axes_ft::dlr_pole_fit pf(FT);
    double max_abs_node = 0.0;
    for (long p = 0; p < pf.np; ++p) max_abs_node = std::max(max_abs_node, std::abs(pf.epsl(p)));
    const double gap_edge =
        resolve_gap_edge_clamped(opts.wsupp, sE_ska.local(), mu, max_abs_node, lvl);

    // The stored dW lives on the PH-symmetric HALF tau grid, so tau_to_w_PHsym returns the
    // POSITIVE half of the bosonic Matsubara mesh (its own index convention:
    // full index nw_b/2 + n). W^c(i nu) is EVEN in nu -- that is exactly what the PH-sym
    // transform pair encodes -- so the full mesh is recovered by mirroring. Both fit routes
    // then see the FULL bosonic mesh, which is what the QM2-b chain measured and what lets
    // the fit reproduce an even function on a NONSYM auxiliary node set.
    // [verified: W(tau) = W(beta - tau) and e^{i nu beta} = 1 give W(-i nu) = W(i nu);
    //  IAFT.icc:48-72 is built on the same identity]
    const long nwb_full = FT.nw_b();
    const long nw_half = (nwb_full % 2 == 0) ? nwb_full / 2 : nwb_full / 2 + 1;
    nda::array<long, 1> half_of(nwb_full);
    nda::array<ComplexType, 1> zb(nwb_full);
    {
      auto wnb = FT.wn_mesh_b();
      for (long iw = 0; iw < nwb_full; ++iw) {
        zb(iw) = FT.omega(wnb(iw));
        const long target = std::abs(wnb(iw));
        long found = -1;
        for (long n = 0; n < nw_half; ++n)
          if (wnb(nwb_full / 2 + n) == target) { found = n; break; }
        utils::check(found >= 0, "qp_modea: bosonic node {} (n = {}) has no partner on the "
                                 "PH-symmetric positive half mesh.", iw, wnb(iw));
        half_of(iw) = found;
      }
    }

    auto mpf = (opts.wfit == "nu")
                   ? imag_axes_ft::masked_pole_fit::from_matsubara(pf, zb, gap_edge, opts.wrtol)
                   : imag_axes_ft::masked_pole_fit::from_tau(pf, gap_edge, opts.wrtol);
    const long npk = mpf.nkeep;
    ctx.npk = npk;
    ctx.om = nda::array<double, 1>(npk);
    ctx.nB = nda::array<double, 1>(npk);
    for (long p = 0; p < npk; ++p) {
      ctx.om(p) = mpf.om(p);
      ctx.nB(p) = sigma_route_b::stable_nB(ctx.beta, ctx.om(p));
    }
    ctx.diag.gap_edge = gap_edge;
    {   // global QP band edges of the CURRENT spectrum -- the strip test needs them
      double lo = -1e300, hi = 1e300;
      auto E = sE_ska.local();
      for (long is2 = 0; is2 < ns; ++is2)
        for (long ik2 = 0; ik2 < nk_ibz; ++ik2)
          for (long a = 0; a < nbnd; ++a) {
            const double e = E(is2, ik2, a).real();
            if (e < mu) lo = std::max(lo, e); else hi = std::min(hi, e);
          }
      ctx.vbm = lo; ctx.cbm = hi;
    }
    ctx.diag.n_support = npk;
    ctx.diag.np_total = pf.np;

    // reality of the tau <-> nu kernels: if Ttw_bb is real then Hermiticity of the STORED
    // W(tau) in (P,Q) transfers exactly to W(i nu), which is what the trev rule needs.
    {
      auto T = FT.Ttw_bb();
      double im = 0.0;
      for (auto const &v : T) im = std::max(im, std::abs(std::imag(ComplexType(v))));
      ctx.diag.ttw_imag = im;
    }

    app_log(lvl, "  - W^c support constraint:      |eps_p| >= {:.6g} a.u. ({:.4g} eV) -- "
                 "{} of {} auxiliary nodes retained, {} singular directions",
            gap_edge, gap_edge * 27.211386245988, npk, pf.np, mpf.n_kept);
    app_log(lvl, "  - W^c pole-fit route:          {} ({} rows), SVD cut rel_tol = {:.2g}",
            opts.wfit, mpf.nrow, mpf.rel_tol);

    // ---------------- stage 1: per-q residue slabs ------------------------------------
    auto &dW = mb_state.dW_qtPQ.value();
    auto [nq_ibz, nt_half, NPg, NQg] = dW.global_shape();
    utils::check(NPg == NP and NQg == NP,
                 "qp_modea: dW has (P,Q) = ({},{}) but thc.Np() = {}.", NPg, NQg, NP);
    auto q_rng = dW.local_range(0);
    auto t_rng = dW.local_range(1);
    auto P_rng = dW.local_range(2);
    auto Q_rng = dW.local_range(3);
    auto W_loc = dW.local();

    const long nwb = nwb_full, ntf = FT.nt_f();
    const long ncols = NP * NP;
    auto [c0, c1] = itertools::chunk_range(0, ncols, mpi->node_comm.size(),
                                           mpi->node_comm.rank());
    const long nc = c1 - c0;

    // the Gamma head (spec section 3). ignore_g0 -> absent on BOTH sides by construction,
    // which is the convention of every QM3 gate and of the QM3-c judge protocol.
    bool head_on = (div_treatment.find("gygi") != std::string::npos or div_treatment == "cvv");
    if (head_on and MF->nqpts_ibz() == 1) {
      app_log(lvl, "  - Gamma head:                  nqpts_ibz == 1 with div_treatment = {} "
                   "-> taking ignore_g0 (same downgrade as gw_t::Sigma_div_correction).",
              div_treatment);
      head_on = false;
    }
    if (head_on and not mb_state.eps_inv_head.has_value()) {
      app_warning("qp_modea: div_treatment = {} but mb_state.eps_inv_head is absent; the "
                  "mode-A Sigma^c is built WITHOUT the long-wavelength head while the "
                  "reference GW Sigma has it. The anchor gate will show the difference.",
                  div_treatment);
      head_on = false;
    }
    nda::array<ComplexType, 1> Hcol(head_on ? nc : 0);
    if (head_on) {
      // W^head_PQ(tau) = nk * madelung * eps_inv_head(tau) * conj(chi_P) * chi_Q.
      // [verified: algebraically identical to gw_t::Sigma_div_correction (thc_gw.icc:461-527),
      //  whose Delta_ij = -madelung * eps_inv_head(tau) * sum_PQ conj(X_Pi) X_Qj G_PQ(k)
      //  conj(chi_P) chi_Q is exactly the q = Gamma term of the main sum with this W added --
      //  the -1/nk prefactor cancels the nk here.]
      // [assumed -- gate: NONE in QM3; both QM3-b fixtures and the QM3-c judge run
      //  div_treatment = ignore_g0, so this branch is UNEXERCISED by any gate.]
      auto chi = thc.basis_head()(0, all);
      const double mad = MF->madelung();
      for (long j = 0; j < nc; ++j) {
        const long P = (c0 + j) / NP, Q = (c0 + j) % NP;
        Hcol(j) = double(nkpts) * mad * std::conj(chi(P)) * chi(Q);
      }
      app_log(lvl, "  - Gamma head:                  ON (div_treatment = {}, madelung = "
                   "{:.6g}) -- UNGATED code path, see wc_band_elements.hpp", div_treatment, mad);
    } else {
      app_log(lvl, "  - Gamma head:                  OFF (div_treatment = {})", div_treatment);
    }

    auto sWres = make_shared_array<nda::array_view<ComplexType, 4>>(
        *mpi, {nq_ibz, npk, NP, NP});
    auto sWt = make_shared_array<nda::array_view<ComplexType, 3>>(*mpi, {nt_half, NP, NP});
    sWres.set_zero();
    sWres.win().fence();

    nda::array<ComplexType, 2> Wt(nt_half, nc), Whalf(nw_half, nc), Ww(nwb, nc);
    nda::array<ComplexType, 2> Wf((opts.wfit == "nu") ? 0 : ntf, (opts.wfit == "nu") ? 0 : nc);
    double herm_num = 0.0, herm_den = 0.0, rec_worst = 0.0, fit_worst = 0.0, ratio_worst = 0.0;

    for (long iq = 0; iq < nq_ibz; ++iq) {
      sWt.set_zero();
      sWt.win().fence();
      if (iq >= q_rng.first() and iq < q_rng.last())
        sWt.local()(t_rng, P_rng, Q_rng) = W_loc(iq - q_rng.first(), all, all, all);
      sWt.win().fence();
      sWt.all_reduce();
      detail::herm_probe(sWt.local(), herm_num, herm_den);

      auto Wall = nda::reshape(sWt.local(), std::array<long, 2>{nt_half, ncols});
      for (long t = 0; t < nt_half; ++t)
        for (long j = 0; j < nc; ++j) Wt(t, j) = Wall(t, c0 + j);
      // q = 0 is the Gamma transfer (the convention of div_utils / embed_eri_t's head).
      if (head_on and iq == 0) {
        auto const &eih = mb_state.eps_inv_head.value();
        utils::check(eih.shape(0) == nt_half,
                     "qp_modea: eps_inv_head has {} nodes, dW(tau) has {}.",
                     eih.shape(0), nt_half);
        for (long t = 0; t < nt_half; ++t)
          for (long j = 0; j < nc; ++j) Wt(t, j) += eih(t).real() * Hcol(j);
      }
      if (nc > 0) {
        FT.tau_to_w_PHsym(Wt, Whalf);
        for (long iw = 0; iw < nwb; ++iw)
          for (long j = 0; j < nc; ++j) Ww(iw, j) = Whalf(half_of(iw), j);
      }

      nda::array<ComplexType, 2> cfit;
      if (opts.wfit == "nu") {
        cfit = mpf.coeffs(Ww);
        if (nc > 0) {
          fit_worst = std::max(fit_worst, mpf.fit_error(Ww, cfit));
          ratio_worst = std::max(ratio_worst, mpf.residue_ratio(Ww, cfit));
        }
      } else {
        if (nc > 0) nda::blas::gemm(FT.Ttw_bb(), Ww, Wf);
        cfit = mpf.coeffs(Wf);
        if (nc > 0) {
          fit_worst = std::max(fit_worst, mpf.fit_error(Wf, cfit));
          ratio_worst = std::max(ratio_worst, mpf.residue_ratio(Wf, cfit));
        }
      }
      for (long p = 0; p < npk; ++p)
        for (long j = 0; j < nc; ++j) cfit(p, j) *= mpf.residue_scale(p);

      // QUALITY METRIC (binding requirement 3): the bosonic-mesh reconstruction of the
      // FITTED representation. NEVER the tau-space fit residual.
      {
        double num = 0.0, den = 0.0;
        for (long m = 0; m < nwb; ++m) {
          const ComplexType z = zb(m);
          for (long j = 0; j < nc; ++j) {
            ComplexType rec(0.0);
            for (long p = 0; p < npk; ++p) rec += cfit(p, j) / (z - ctx.om(p));
            num = std::max(num, std::abs(rec - Ww(m, j)));
            den = std::max(den, std::abs(Ww(m, j)));
          }
        }
        if (den > 0.0) rec_worst = std::max(rec_worst, num / den);
      }

      for (long p = 0; p < npk; ++p)
        for (long j = 0; j < nc; ++j)
          sWres.local()(iq, p, (c0 + j) / NP, (c0 + j) % NP) = cfit(p, j);

      // ---- ONE-OFF fit-conditioning survey on the first q (cheap; diagnostics only) -----
      // Route B at REAL z is destroyed by NEAR-CANCELLING residues, not by the imaginary-axis
      // fit error (dlr_pole_fit::residue_ratio documents exactly this failure mode). Report
      // the (reconstruction, residue-ratio) pair for the unconstrained fit and for the
      // support-constrained fit at a few FIXED SVD cuts, so the choice is measured.
      if (iq == 0 and nc > 0) {
        auto probe = [&](std::string const &name, imag_axes_ft::masked_pole_fit const &f,
                         nda::array<ComplexType, 2> const &data) {
          auto cc = f.coeffs(data);
          const double own = f.fit_error(data, cc);
          const double rr = f.residue_ratio(data, cc);
          double num = 0.0, den = 0.0;
          for (long m = 0; m < nwb; ++m)
            for (long j = 0; j < nc; ++j) {
              ComplexType rec(0.0);
              for (long q2 = 0; q2 < f.nkeep; ++q2)
                rec += f.residue_scale(q2) * cc(q2, j) / (zb(m) - f.om(q2));
              num = std::max(num, std::abs(rec - Ww(m, j)));
              den = std::max(den, std::abs(Ww(m, j)));
            }
          app_log(lvl + 1, "    [fit survey q=0] {:<34} nodes {:>3}/{:<3} rank {:>3}  bosonic-mesh "
                       "rec = {:.3e}  own-grid = {:.3e}  residue ratio = {:.3e}",
                  name, f.nkeep, f.np_all, f.n_kept, (den > 0.0 ? num / den : 0.0), own, rr);
        };
        auto const &data = (opts.wfit == "nu") ? Ww : Wf;
        for (double rt : {1e-8, 1e-6, 1e-4, 1e-2}) {
          auto f0 = (opts.wfit == "nu")
                        ? imag_axes_ft::masked_pole_fit::from_matsubara(pf, zb, 0.0, rt)
                        : imag_axes_ft::masked_pole_fit::from_tau(pf, 0.0, rt);
          probe(std::format("plain,      rel_tol = {:.0e}", rt), f0, data);
          auto f1 = (opts.wfit == "nu")
                        ? imag_axes_ft::masked_pole_fit::from_matsubara(pf, zb, gap_edge, rt)
                        : imag_axes_ft::masked_pole_fit::from_tau(pf, gap_edge, rt);
          probe(std::format("support-constrained, rel_tol = {:.0e}", rt), f1, data);
        }
      }
      mpi->node_comm.barrier();
    }
    sWres.win().fence();
    ctx.diag.w_herm_rel = (herm_den > 0.0) ? herm_num / herm_den : 0.0;
    ctx.diag.w_herm_rel = mpi->comm.all_reduce_value(ctx.diag.w_herm_rel, boost::mpi3::max<>{});
    ctx.diag.rec_rel_worst = mpi->comm.all_reduce_value(rec_worst, boost::mpi3::max<>{});
    ctx.diag.fit_err_worst = mpi->comm.all_reduce_value(fit_worst, boost::mpi3::max<>{});
    ctx.diag.res_ratio_worst = mpi->comm.all_reduce_value(ratio_worst, boost::mpi3::max<>{});
    Wt = nda::array<ComplexType, 2>();
    Whalf = nda::array<ComplexType, 2>();
    Ww = nda::array<ComplexType, 2>();
    Wf = nda::array<ComplexType, 2>();

    // ---------------- MO collocation columns ------------------------------------------
    // internal leg: XCi(s, k_full) = X(k_full) . C(kp_to_ibz(k_full))   [derivation 1]
    nda::array<ComplexType, 3> XCi(ns * nkpts, NP, nbnd);
    {
      auto kp_to_ibz = MF->kp_to_ibz();
      for (long sk = 0; sk < ns * nkpts; ++sk) {
        const long is = sk / nkpts, ik = sk % nkpts;
        nda::blas::gemm(thc.X(is, 0, ik),
                        sMO_skia.local()(is, kp_to_ibz(ik), all, all),
                        XCi(sk, all, all));
      }
    }

    // ---------------- stage 2: the sandwiches -----------------------------------------
    const long nJ = nqpts * nbnd;
    ctx.nJ = nJ;
    ctx.diag.nJ = nJ;
    ctx.diag.npk = npk;
    ctx.epsJ = nda::array<double, 1>(ns * nk_ibz * nJ);
    ctx.fJ = nda::array<double, 1>(ns * nk_ibz * nJ);
    ctx.epsJ() = 0.0;
    ctx.fJ() = 0.0;
    ctx.owner = nda::array<long, 1>(ns * nk_ibz);
    ctx.owner() = -1;

    const long nP_flat = nJ * npk;
    if (need_diag) {
      ctx.Mdiag = nda::array<ComplexType, 4>(ns, nk_ibz, nbnd, nP_flat);
      ctx.Mdiag() = ComplexType(0.0);
      ctx.have_diag = true;
    }

    auto kp_to_ibz = MF->kp_to_ibz();
    auto kp_trev = MF->kp_trev();
    auto kp_trev_pair = MF->kp_trev_pair();
    auto qp_trev = MF->qp_trev();
    auto qminus = MF->qminus();
    const double pref = 1.0 / double(nkpts);

    nda::array<ComplexType, 2> XCe(NP, nbnd), DC(nbnd, nbnd);
    nda::array<ComplexType, 2> B(NP, nbnd), Bc(NP, nbnd), T(NP, nbnd), Msand(nbnd, nbnd);
    nda::array<ComplexType, 2> Wp(NP, NP);

    for (long sk = 0; sk < ns * nk_ibz; ++sk) {
      ctx.owner(sk) = sk % mpi->comm.size();
      if (ctx.owner(sk) != mpi->comm.rank()) continue;
      const long is = sk / nk_ibz, ik = sk % nk_ibz;
      sk_block blk;
      blk.is = is;
      blk.ik = ik;
      blk.M = nda::array<ComplexType, 3>(nbnd, nbnd, nP_flat);
      blk.M() = ComplexType(0.0);

      for (long isym = 0; isym < nsym; ++isym) {
        const long ks = MF->ks_to_k(isym, ik);
        if (isym == 0) {
          XCe = XCi(is * nkpts + ks, all, all);
        } else {
          auto [cjg, D] = MF->symmetry_rotation(isym, ik);
          utils::check(not cjg, "qp_modea: symmetry_rotation(isym = {}, k = {}) reports the "
                                "conjugation flag, which the GW assembly this map reproduces "
                                "does not handle either (thc_gw.icc:311).", isym, ik);
          math::sparse::csrmm(ComplexType(1.0), *D,
                              nda::make_regular(sMO_skia.local()(is, ik, all, all)),
                              ComplexType(0.0), DC);
          nda::blas::gemm(thc.X(is, 0, ks), DC, XCe);
        }

        for (long iq = 0; iq < MF->nq_per_s(isym); ++iq) {
          const long qp = MF->Qs(isym, iq);
          const long qs = MF->qp_to_ibz(qp);
          const bool wconj = qp_trev(qp);
          const long kk = wconj ? MF->qk_to_k2(qminus(qs), ks) : MF->qk_to_k2(qs, ks);
          const bool gconj = kp_trev(kk);
          const long kg = gconj ? kp_trev_pair(kk) : kk;
          const long kg_ibz = kp_to_ibz(kg);
          auto U = XCi(is * nkpts + kg, all, all);

          for (long n = 0; n < nbnd; ++n) {
            const long J = qp * nbnd + n;
            const double e = sE_ska.local()(is, kg_ibz, n).real();
            ctx.epsJ((is * nk_ibz + ik) * nJ + J) = e;
            ctx.fJ((is * nk_ibz + ik) * nJ + J) = sigma_route_b::stable_nF(ctx.beta, e - mu);
            for (long P = 0; P < NP; ++P) {
              const ComplexType u = gconj ? std::conj(U(P, n)) : U(P, n);
              for (long a = 0; a < nbnd; ++a) {
                B(P, a) = std::conj(XCe(P, a)) * u;
                Bc(P, a) = std::conj(B(P, a));
              }
            }
            for (long p = 0; p < npk; ++p) {
              if (wconj) Wp = nda::conj(sWres.local()(qs, p, all, all));
              else       Wp = sWres.local()(qs, p, all, all);
              nda::blas::gemm(Wp, Bc, T);                           // (NP, nbnd)
              nda::blas::gemm(nda::transpose(B), T, Msand);         // (nbnd, nbnd)
              const long Pf = J * npk + p;
              for (long a = 0; a < nbnd; ++a)
                for (long b = 0; b < nbnd; ++b) blk.M(a, b, Pf) += pref * Msand(a, b);
            }
          }
        }
      }
      if (need_diag)
        for (long a = 0; a < nbnd; ++a)
          for (long Pf = 0; Pf < nP_flat; ++Pf) ctx.Mdiag(is, ik, a, Pf) = blk.M(a, a, Pf);
      ctx.blocks.push_back(std::move(blk));
    }

    mpi->comm.all_reduce_in_place_n(ctx.epsJ.data(), ctx.epsJ.size(), std::plus<>{});
    mpi->comm.all_reduce_in_place_n(ctx.fJ.data(), ctx.fJ.size(), std::plus<>{});
    if (need_diag)
      mpi->comm.all_reduce_in_place_n(ctx.Mdiag.data(), ctx.Mdiag.size(), std::plus<>{});

    ctx.active = true;
    ctx.have_cd = true;
    const double mb = 1.0 / (1024.0 * 1024.0);
    double mem = double(ctx.blocks.size()) * double(nbnd) * double(nbnd) * double(nP_flat)
                 * sizeof(ComplexType) * mb
                 + double(nq_ibz) * double(npk) * double(NP) * double(NP)
                       * sizeof(ComplexType) * mb
                 + (need_diag ? double(ns) * nk_ibz * nbnd * nP_flat * sizeof(ComplexType) * mb
                              : 0.0);
    ctx.diag.mem_mb = mpi->comm.all_reduce_value(mem, boost::mpi3::max<>{});
    ctx.diag.wall_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    ctx.diag.wall_s = mpi->comm.all_reduce_value(ctx.diag.wall_s, boost::mpi3::max<>{});

    app_log(lvl, "  - internal pole structure:     nJ = {} (nq_full {} x nbnd {}) x npk = {} "
                 "-> {} poles per (a,b)", nJ, nqpts, nbnd, npk, nP_flat);
    app_log(lvl, "  - stored W Hermiticity:        max|W - W^dag|/max|W| = {:.3e} "
                 "(max|Im Ttw_bb| = {:.3e}); the trev-q rule implemented is conj(W_PQ), "
                 "matching thc_gw.icc:381-393", ctx.diag.w_herm_rel, ctx.diag.ttw_imag);
    app_log(lvl, "  - W^c fit quality:             bosonic-mesh reconstruction rel err = "
                 "{:.3e} (worst q)   [own-grid residual {:.3e}, residue ratio {:.3g} -- "
                 "reported, NOT the quality number]",
            ctx.diag.rec_rel_worst, ctx.diag.fit_err_worst, ctx.diag.res_ratio_worst);
    app_log(lvl, "  - context build:               {:.2f} s wall, {:.1f} MB extra per rank "
                 "(peak)", ctx.diag.wall_s, ctx.diag.mem_mb);

    auto &LR = last_run();
    LR.gap_edge = ctx.diag.gap_edge;
    LR.rec_rel = ctx.diag.rec_rel_worst;
    LR.wall_s = ctx.diag.wall_s;
    LR.mem_mb = ctx.diag.mem_mb;
    LR.n_support = ctx.diag.n_support;
    LR.np_total = ctx.diag.np_total;
    LR.nJ = nJ;
    LR.npk = npk;
    LR.wfit = opts.wfit;
    LR.res_ratio = ctx.diag.res_ratio_worst;
    LR.wrtol = mpf.rel_tol;
  }

} // qp_modea
} // methods

#endif // COQUI_WC_BAND_ELEMENTS_HPP
