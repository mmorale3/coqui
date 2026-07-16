/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
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


#include <cmath>
#include <unordered_set>

#include "utilities/check.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "vertex_t.h"
#include "vertex_pi.icc"
#include "vertex_sigma.icc"  // ISDF-Vertex Phase 1c: fused G^3 W^2 Sigma^C kernel

namespace methods {
namespace solvers {

  namespace vertex_head_detail {

    /**
     * The analytic q->0 head of the rung at the Gamma cell, in the STORED-ARRAY
     * convention (notes/q0_head_treatment.md, (H1) and section 1.5):
     *
     *   H_PQ = N_k * madelung * conj(chi_P(Gamma)) * chi_Q(Gamma)
     *
     * with chi = thc.basis_head() (the G = 0 plane-wave components of the aux basis)
     * and madelung = MF->madelung() (the Gygi-Baldereschi / probe-charge-Ewald
     * constant, PRB 80, 085114 (2009)). Pinned against existing code: consuming
     * H_PQ through the GW Hadamard reproduces Sigma_div_correction
     * (thc_gw.icc:444-525) exactly (dynamic piece, weight Re[eps_inv_head(tau)]),
     * and through the exchange reproduces HF_K_correction (hf_t.cpp:64-100)
     * (bare piece, weight 1).
     *
     * Returns false (H untouched) when the head data are unusable: madelung == 0
     * (model systems) or chi_head not populated (some ERI read paths,
     * thc_reader_t.hpp:380-381). The caller logs and proceeds without insertion.
     */
    template<THC_ERI thc_t>
    bool build_head_rank1(thc_t const& thc, long iq_gamma, long nkpts,
                          nda::array<ComplexType, 2>& H_PQ) {
      auto MF = thc.MF();
      const double xi = MF->madelung();
      auto chi = thc.basis_head();   // (nqpts_ibz, Np)
      const long Np = H_PQ.shape(0);
      utils::check(chi.shape(0) > iq_gamma and chi.shape(1) == Np,
                   "vertex_head_detail::build_head_rank1: basis_head shape mismatch "
                   "(({}, {}) vs iq_gamma = {}, Np = {}).",
                   chi.shape(0), chi.shape(1), iq_gamma, Np);
      double chi_max = 0.0;
      for (long P = 0; P < Np; ++P) chi_max = std::max(chi_max, std::abs(chi(iq_gamma, P)));
      if (xi == 0.0 or chi_max == 0.0) return false;
      for (long P = 0; P < Np; ++P)
        for (long Q = 0; Q < Np; ++Q)
          H_PQ(P, Q) = double(nkpts) * xi * std::conj(chi(iq_gamma, P)) * chi(iq_gamma, Q);
      return true;
    }

  } // vertex_head_detail

  void vertex_t::set_div_treatment(std::string div) {
    const std::unordered_set<std::string> exact = {"ignore_g0", "v1_skip"};
    utils::check(exact.count(div) > 0 or div.find("gygi") != std::string::npos,
                 "vertex_t: unknown vertex div_treatment: {}. Valid options are "
                 "\"ignore_g0\" (v2 default), \"gygi\"-class, and \"v1_skip\".", div);
    _div_treatment = std::move(div);
  }

  vertex_t::vertex_t(const imag_axes_ft::IAFT *ft,
                     std::string vertex_type,
                     nda::range band_window,
                     long nbnd,
                     std::string div_treatment):
    _ft(ft), _vertex_type(std::move(vertex_type)), _band_window(band_window) {

    const std::unordered_set<std::string> valid_vertex_types = {"none", "2nd_exchange"};
    utils::check(valid_vertex_types.find(_vertex_type) != valid_vertex_types.end(),
                 "vertex_t: unknown vertex_type: {}. Valid options are \"none\" and \"2nd_exchange\".",
                 _vertex_type);
    set_div_treatment(std::move(div_treatment));
    if (not enabled()) return;

    utils::check(_ft != nullptr, "vertex_t: IAFT instance is required when the vertex is enabled.");
    utils::check(_band_window.first() >= 0 and _band_window.first() <= _band_window.last(),
                 "vertex_t: invalid vertex_band_window = [{}, {}). Expect 0 <= first <= last.",
                 _band_window.first(), _band_window.last());
    utils::check(_band_window.last() <= nbnd,
                 "vertex_t: invalid vertex_band_window = [{}, {}). "
                 "The window must be within the primary basis: last <= nbnd = {}.",
                 _band_window.first(), _band_window.last(), nbnd);

    if (active()) {
      app_log(1, "\n"
                 "  Second-order exchange vertex correction (ISDF-Vertex)\n"
                 "  ------------------------------------------------------\n"
                 "  Vertex type              = {}\n"
                 "  Subspace C band window   = [{}, {})\n"
                 "  Subspace C size          = {} orbitals (nbnd = {})\n"
                 "  Cuts                     = Sigma^C (G3W2) + Pi^C (G4W), always both\n"
                 "  q->0 rung policy         = {} (notes/q0_head_treatment.md)\n"
                 "  Status                   = Pi^C kernel ACTIVE (Phase 1d); Sigma^C status is\n"
                 "                             reported by eval_Sigma_C at evaluation time\n",
              _vertex_type, _band_window.first(), _band_window.last(),
              _band_window.size(), nbnd, _div_treatment);
    } else {
      app_log(1, "\nvertex_t: vertex_type = \"{}\" with an empty vertex_band_window: "
                 "C = empty set, so the vertex contributes nothing and the "
                 "calculation reduces to plain scGW exactly.\n", _vertex_type);
    }
  }

  void vertex_t::eval_Sigma_C(MBState &mb_state, THC_ERI auto const &thc) {
    utils::check(active(), "vertex_t::eval_Sigma_C: called while the vertex is inactive. "
                           "Callers must guard vertex calls with vertex_t::active().");
    utils::check(mb_state.sG_tskij.has_value(),
                 "vertex_t::eval_Sigma_C: sG_tskij is not initialized in MBState.");
    utils::check(mb_state.sSigma_tskij.has_value(),
                 "vertex_t::eval_Sigma_C: sSigma_tskij is not initialized in MBState.");
    utils::check(mb_state.dW_qtPQ.has_value(),
                 "vertex_t::eval_Sigma_C: dW_qtPQ is not initialized in MBState.");

    decltype(nda::range::all) all;
    auto mpi = thc.mpi();
    auto MF = thc.MF();
    const long nkpts = MF->nkpts();
    const long nqpts = MF->nqpts();
    const long nkpts_ibz = MF->nkpts_ibz();
    const long nqpts_ibz = MF->nqpts_ibz();
    const long Np = thc.Np();
    const long nbnd = MF->nbnd();

    // SYMMETRY POLICY (Phase 1c): symmetry-free meshes only; abort loudly rather than
    // produce silently wrong results (the audit-sec.7 IBZ unfolding is a follow-up).
    utils::check(nqpts == nqpts_ibz and nkpts == nkpts_ibz and nqpts == nkpts,
                 "vertex_t::eval_Sigma_C: vertex + k-space symmetry is not supported yet "
                 "(nqpts = {}, nqpts_ibz = {}, nkpts = {}, nkpts_ibz = {}). "
                 "Rerun with symmetry disabled in the mean-field input.",
                 nqpts, nqpts_ibz, nkpts, nkpts_ibz);
    {
      auto kp_trev = MF->kp_trev();
      for (long ik = 0; ik < nkpts; ++ik)
        utils::check(not kp_trev(ik),
                     "vertex_t::eval_Sigma_C: vertex + time-reversal-reduced meshes are not supported yet.");
    }
    utils::check(MF->npol() == 1, "vertex_t::eval_Sigma_C: npol != 1 is not supported.");
    utils::check(_ft->basis() == imag_axes_ft::dlr_basis,
                 "vertex_t::eval_Sigma_C: the fused G3W2 kernel requires the DLR IAFT backend "
                 "(iaft basis = \"dlr\"); the IR backend is not supported.");

    auto G_tskij = mb_state.sG_tskij.value().local();
    auto& sSigma_tskij = mb_state.sSigma_tskij.value();
    const long nt = G_tskij.shape(0);
    const long ns = G_tskij.shape(1);
    const long nt_half = (nt % 2 == 0) ? nt / 2 : nt / 2 + 1;
    utils::check(nt == _ft->nt_f(), "vertex_t::eval_Sigma_C: G time axis != nt_f.");
    { // the W(beta-tau)=W(tau) unfolding below requires a tau mesh symmetric about beta/2
      auto tau_mesh = _ft->tau_mesh();
      for (long it = 0; it < nt; ++it)
        utils::check(std::abs(std::abs(tau_mesh(it)) - std::abs(tau_mesh(nt - it - 1))) <= 1e-6,
                     "vertex_t::eval_Sigma_C: IAFT tau grid is not particle-hole symmetric.");
    }

    app_log(1, "\n  ISDF-Vertex: evaluating Sigma^C (G^3 W^2, double bosonic convolution)\n"
               "  ---------------------------------------------------------------------\n"
               "  Subspace C band window = [{}, {})  ({} orbitals)\n"
               "  nbnd = {}, Np = {}, nkpts = {}, prefactor = +1 (sign_crossing_report)\n",
            _band_window.first(), _band_window.last(), _band_window.size(),
            nbnd, Np, nkpts);

    // ---- collocation matrices (q-independent X, polarization 0) ----------------------
    nda::array<ComplexType, 4> X_skPa(ns, nkpts, Np, nbnd);
    for (long is = 0; is < ns; ++is)
      for (long ik = 0; ik < nkpts; ++ik)
        X_skPa(is, ik, all, all) = thc.X(is, 0, ik);

    // ---- q = Gamma index (crystal coordinates: all components integer mod G) ----------
    long iq_gamma = -1;
    {
      auto Qpts = MF->Qpts();
      for (long iq = 0; iq < nqpts; ++iq) {
        double d = 0.0;
        for (long i = 0; i < 3; ++i) {
          double x = Qpts(iq, i);
          d += std::abs(x - std::round(x));
        }
        if (d < 1e-8) {
          utils::check(iq_gamma < 0,
                       "vertex_t::eval_Sigma_C: multiple Gamma q-points found ({} and {}).",
                       iq_gamma, iq);
          iq_gamma = iq;
        }
      }
      utils::check(iq_gamma >= 0, "vertex_t::eval_Sigma_C: no Gamma q-point found.");
    }

    // ---- q->0 rung policy (notes/q0_head_treatment.md section 3) ----------------------
    const bool skip_rung_gamma = (_div_treatment == "v1_skip");
    bool head_insertion = (_div_treatment.find("gygi") != std::string::npos);
    if (head_insertion and nqpts_ibz == 1) {
      app_log(1, "  [WARNING] Sigma^C: nqpts_ibz == 1 while vertex div_treatment is "
                 "gygi-class; the q->0\n"
                 "            extrapolation is meaningless on a Gamma-only mesh -- taking "
                 "\"ignore_g0\" instead\n"
                 "            (same downgrade as GW's Sigma_div_correction).");
      head_insertion = false;
    }
    if (skip_rung_gamma)
      app_log(1, "  [NOTE] Sigma^C q->0 policy: v1_skip -- the q = Gamma (iq = {}) cell is "
                 "DROPPED on both rung\n"
                 "         transfers (qx and qy; bare Z and dynamic dW). Fallback mode; "
                 "O(1/N_k) finite-size error.\n", iq_gamma);
    else
      app_log(1, "  [NOTE] Sigma^C q->0 policy: {} -- the q = Gamma (iq = {}) cell of both "
                 "rung transfers is\n"
                 "         INCLUDED with the stored regularized W(Gamma) (v(G=0) zeroed at "
                 "ERI build){}.\n", _div_treatment, iq_gamma,
              head_insertion ? ",\n         PLUS the analytic rank-1 head insertion "
                               "Nk*madelung*[1 | Re eps_inv_head(tau)]*chi chi^dag"
                             : "; no analytic head (GW ignore_g0 analogue)");

    // ---- bare coulomb Z(q): the instantaneous part of the rungs (collective call) -----
    nda::array<ComplexType, 3> Z_qPQ(nqpts, Np, Np);
    for (long iq = 0; iq < nqpts; ++iq)
      Z_qPQ(iq, all, all) = thc.Z(int(iq));

    // head insertion, bare piece (weight 1) into Z(Gamma)
    nda::array<ComplexType, 2> H_PQ(Np, Np);
    bool head_ok = false;
    if (head_insertion) {
      head_ok = vertex_head_detail::build_head_rank1(thc, iq_gamma, nkpts, H_PQ);
      if (head_ok) {
        Z_qPQ(iq_gamma, all, all) += H_PQ;
        double h_max = 0.0;
        for (auto const& v : H_PQ) h_max = std::max(h_max, std::abs(v));
        app_log(1, "  Sigma^C head insertion: madelung = {}, |H|_max = {} (bare piece "
                   "applied to Z(Gamma))", MF->madelung(), h_max);
      } else {
        app_log(1, "  [WARNING] Sigma^C: gygi head insertion requested but head data are "
                   "unusable\n"
                   "            (madelung == 0 or empty basis_head) -- proceeding WITHOUT "
                   "the analytic head\n"
                   "            (equivalent to policy \"ignore_g0\").");
      }
    }

    // ---- dynamic W(tau): replicate and unfold nt_half storage to the full tau mesh ----
    // dW_qtPQ is dynamic-only (bare Z subtracted, scr_coulomb_t.cpp:217); W is
    // PH-symmetric in tau, W(beta-t) = W(t).
    nda::array<ComplexType, 4> Wt_qtPQ(nqpts, nt, Np, Np);
    {
      auto& dW = mb_state.dW_qtPQ.value();
      auto gs = dW.global_shape();
      utils::check(gs[0] == nqpts_ibz and gs[1] == nt_half and gs[2] == Np and gs[3] == Np,
                   "vertex_t::eval_Sigma_C: unexpected dW_qtPQ global shape.");
      nda::array<ComplexType, 4> W_half(nqpts, nt_half, Np, Np);
      W_half() = ComplexType(0.0);
      W_half(dW.local_range(0), dW.local_range(1), dW.local_range(2), dW.local_range(3)) = dW.local();
      mpi->comm.all_reduce_in_place_n(W_half.data(), W_half.size(), std::plus<>{});

      // head insertion, dynamic piece (weight Re[eps_inv_head(tau)]) into dW(Gamma, tau).
      // eps_inv_head = eps^-1_00(q->0, tau) - 1, stored on nt_half by scr_coulomb
      // (scr_coulomb_t.cpp:106-108); same Re[.] convention as Sigma_div_correction.
      if (head_ok) {
        if (mb_state.eps_inv_head.has_value()) {
          auto& eps = mb_state.eps_inv_head.value();
          utils::check(eps.shape(0) == nt_half,
                       "vertex_t::eval_Sigma_C: eps_inv_head size {} != nt_half = {}.",
                       eps.shape(0), nt_half);
          for (long it = 0; it < nt_half; ++it)
            W_half(iq_gamma, it, all, all) += ComplexType(eps(it).real()) * H_PQ;
          app_log(1, "  Sigma^C head insertion: dynamic piece applied to dW(Gamma, tau) "
                     "with eps_inv_head(tau=0) = {}", eps(0).real());
        } else {
          app_log(1, "  [WARNING] Sigma^C: dW is present but eps_inv_head is not in MBState "
                     "-- the DYNAMIC head\n"
                     "            piece is skipped (bare piece applied).");
        }
      }

      for (long it = 0; it < nt; ++it) {
        long ith = std::min(it, nt - it - 1);
        Wt_qtPQ(all, it, all, all) = W_half(all, ith, all, all);
      }
    }

    // ---- momentum maps (symmetry-free mesh) -------------------------------------------
    nda::array<long, 2> kmq(nqpts, nkpts);
    nda::array<long, 1> qmin(nqpts);
    for (long iq = 0; iq < nqpts; ++iq) {
      qmin(iq) = MF->qminus()(iq);
      for (long ik = 0; ik < nkpts; ++ik) kmq(iq, ik) = MF->qk_to_k2(iq, ik);
    }

    // ---- fused kernel (round-robin over (s,k,qx); result all-reduced inside) ----------
    nda::array<ComplexType, 5> Sigma_C(nt, ns, nkpts, nbnd, nbnd);
    vertex_detail::eval_sigma_C_g3w2_nosym(*_ft, mpi->comm, _band_window, G_tskij,
                                           X_skPa, Wt_qtPQ, Z_qPQ, kmq, qmin, iq_gamma,
                                           skip_rung_gamma, Sigma_C);
    {
      double max_abs = 0.0;
      long n_bad = 0;
      for (auto const& v : Sigma_C) {
        double a = std::abs(v);
        if (not std::isfinite(a)) { ++n_bad; continue; }
        max_abs = std::max(max_abs, a);
      }
      utils::check(n_bad == 0,
                   "vertex_t::eval_Sigma_C: Sigma^C contains {} NaN/Inf entries -- aborting.", n_bad);
      app_log(2, "  Sigma^C(tau) max|.| = {}\n", max_abs);
    }

    // accumulate on top of the GW self-energy: Sigma <- Sigma + Sigma^C
    // (Sigma_C is identical on every rank after the kernel's all_reduce; hermitization
    //  stays downstream in scf_driver)
    if (mb_state.mpi->node_comm.root())
      sSigma_tskij.local() += Sigma_C;
    mb_state.mpi->comm.barrier();
  }

  auto vertex_t::eval_Pi_C(MBState &mb_state, THC_ERI auto const &thc,
                           shape_t<4> pi_pgrid, shape_t<4> pi_bsize, shape_t<4> pi_gshape)
  -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>
  {
    decltype(nda::range::all) all;
    utils::check(active(), "vertex_t::eval_Pi_C: called while the vertex is inactive. "
                           "Callers must guard vertex calls with vertex_t::active().");
    utils::check(mb_state.sG_tskij.has_value(),
                 "vertex_t::eval_Pi_C: sG_tskij is not initialized in MBState.");

    auto mpi = thc.mpi();
    auto MF = thc.MF();
    long nkpts = MF->nkpts();
    long nqpts = MF->nqpts();
    long nqpts_ibz = MF->nqpts_ibz();
    long nkpts_ibz = MF->nkpts_ibz();
    long Np = thc.Np();
    long nbnd = MF->nbnd();

    // SYMMETRY POLICY (Phase 1d): only symmetry-free meshes are supported; anything else
    // aborts loudly rather than producing silently wrong results (audit-sec.7 unfolding
    // is a follow-up).
    utils::check(nqpts == nqpts_ibz and nkpts == nkpts_ibz,
                 "vertex_t::eval_Pi_C: vertex + k-space symmetry is not supported yet "
                 "(nqpts = {}, nqpts_ibz = {}, nkpts = {}, nkpts_ibz = {}). "
                 "Rerun with symmetry disabled in the mean-field input.",
                 nqpts, nqpts_ibz, nkpts, nkpts_ibz);
    {
      auto kp_trev = MF->kp_trev();
      for (long ik = 0; ik < nkpts; ++ik)
        utils::check(not kp_trev(ik),
                     "vertex_t::eval_Pi_C: vertex + time-reversal-reduced meshes are not supported yet.");
    }

    auto G_tskij = mb_state.sG_tskij.value().local();
    long nt_f = G_tskij.shape(0);
    long ns = G_tskij.shape(1);
    long nt_half = (nt_f % 2 == 0) ? nt_f / 2 : nt_f / 2 + 1;
    utils::check(pi_gshape[0] == nt_half and pi_gshape[1] == nqpts_ibz and
                 pi_gshape[2] == Np and pi_gshape[3] == Np,
                 "vertex_t::eval_Pi_C: unexpected Pi grid shape ({}, {}, {}, {}); "
                 "expected ({}, {}, {}, {}).",
                 pi_gshape[0], pi_gshape[1], pi_gshape[2], pi_gshape[3],
                 nt_half, nqpts_ibz, Np, Np);

    app_log(1, "\n  ISDF-Vertex: evaluating Pi^C (G^4 W, single rung)\n"
               "  -------------------------------------------------\n"
               "  Subspace C band window = [{}, {})  ({} orbitals)\n"
               "  Grid (nt_half, nq, Np, Np) = ({}, {}, {}, {})\n",
            _band_window.first(), _band_window.last(), _band_window.size(),
            nt_half, nqpts_ibz, Np, Np);
    if (_ft->basis() != imag_axes_ft::dlr_basis)
      app_log(1, "  [NOTE] Pi^C requires off-grid Matsubara interpolation from the imaginary-\n"
                 "         axis backend (IAFT::construct_w_interpolate_matrix). The DLR backend\n"
                 "         provides it; the IR driver does not implement it yet and will abort\n"
                 "         inside the backend if this run proceeds.\n");

    vertex_pi::iaft_tools tools(*_ft);

    // ---- q = Gamma index (crystal coordinates: all components integer mod G) ----------
    long iq_gamma = -1;
    {
      auto Qpts = MF->Qpts();
      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        double d = 0.0;
        for (long i = 0; i < 3; ++i) {
          double x = Qpts(iq, i);
          d += std::abs(x - std::round(x));
        }
        if (d < 1e-8) {
          utils::check(iq_gamma < 0,
                       "vertex_t::eval_Pi_C: multiple Gamma q-points found ({} and {}).",
                       iq_gamma, iq);
          iq_gamma = iq;
        }
      }
      utils::check(iq_gamma >= 0, "vertex_t::eval_Pi_C: no Gamma q-point found.");
    }

    // ---- q->0 rung policy (notes/q0_head_treatment.md section 3) ----------------------
    const bool skip_rung_gamma = (_div_treatment == "v1_skip");
    bool head_insertion = (_div_treatment.find("gygi") != std::string::npos);
    if (head_insertion and nqpts_ibz == 1) {
      app_log(1, "  [WARNING] Pi^C: nqpts_ibz == 1 while vertex div_treatment is "
                 "gygi-class; the q->0\n"
                 "            extrapolation is meaningless on a Gamma-only mesh -- taking "
                 "\"ignore_g0\" instead\n"
                 "            (same downgrade as GW's Sigma_div_correction).");
      head_insertion = false;
    }
    if (skip_rung_gamma)
      app_log(1, "  [NOTE] Pi^C q->0 policy: v1_skip -- the qx = Gamma (iq = {}) cell of "
                 "the internal rung\n"
                 "         transfer is DROPPED (bare Z and dynamic dW). Fallback mode; "
                 "O(1/N_k) finite-size error.\n", iq_gamma);
    else
      app_log(1, "  [NOTE] Pi^C q->0 policy: {} -- the qx = Gamma (iq = {}) cell of the "
                 "internal rung transfer\n"
                 "         is INCLUDED with the stored regularized W(Gamma) (v(G=0) zeroed "
                 "at ERI build){}.\n"
                 "         The external q axis is regular and computed at all q.\n",
              _div_treatment, iq_gamma,
              head_insertion ? ",\n         PLUS the analytic rank-1 head insertion "
                               "Nk*madelung*[1 | Re eps_inv_head(tau)]*chi chi^dag"
                             : "; no analytic head (GW ignore_g0 analogue)");

    // ---- collocation matrices (q-independent X, polarization 0) -----------------------
    nda::array<ComplexType, 4> X_skPa(ns, nkpts, Np, nbnd);
    for (long is = 0; is < ns; ++is)
      for (long ik = 0; ik < nkpts; ++ik)
        X_skPa(is, ik, all, all) = thc.X(is, 0, ik);

    // ---- bare coulomb Z(q) (thc.Z is collective: call uniformly on all ranks) ---------
    nda::array<ComplexType, 3> Z_qPQ(nqpts_ibz, Np, Np);
    for (long iq = 0; iq < nqpts_ibz; ++iq)
      Z_qPQ(iq, all, all) = thc.Z(int(iq));

    // head insertion, bare piece (weight 1) into Z(Gamma) -- (H1) of the memo
    nda::array<ComplexType, 2> H_PQ(Np, Np);
    bool head_ok = false;
    if (head_insertion) {
      head_ok = vertex_head_detail::build_head_rank1(thc, iq_gamma, nkpts, H_PQ);
      if (head_ok) {
        Z_qPQ(iq_gamma, all, all) += H_PQ;
        double h_max = 0.0;
        for (auto const& v : H_PQ) h_max = std::max(h_max, std::abs(v));
        app_log(1, "  Pi^C head insertion: madelung = {}, |H|_max = {} (bare piece "
                   "applied to Z(Gamma))", MF->madelung(), h_max);
      } else {
        app_log(1, "  [WARNING] Pi^C: gygi head insertion requested but head data are "
                   "unusable\n"
                   "            (madelung == 0 or empty basis_head) -- proceeding WITHOUT "
                   "the analytic head\n"
                   "            (equivalent to policy \"ignore_g0\").");
      }
    }

    // ---- dynamic W on the full bosonic Matsubara mesh ---------------------------------
    // mb_state.dW_qtPQ is the dynamic-only screened interaction (bare Z subtracted,
    // scr_coulomb_t.cpp:217) on (nq, nt_half, Np, Np). Absent on the very first
    // iteration: the rung then reduces to the bare interaction Z.
    std::optional<nda::array<ComplexType, 4>> Wdyn_qwPQ;
    if (mb_state.dW_qtPQ.has_value()) {
      auto& dW = mb_state.dW_qtPQ.value();
      auto gs = dW.global_shape();
      utils::check(gs[0] == nqpts_ibz and gs[1] == nt_half and gs[2] == Np and gs[3] == Np,
                   "vertex_t::eval_Pi_C: unexpected dW_qtPQ global shape.");
      nda::array<ComplexType, 4> W_qtPQ(nqpts_ibz, nt_half, Np, Np);
      W_qtPQ() = ComplexType(0.0);
      W_qtPQ(dW.local_range(0), dW.local_range(1), dW.local_range(2), dW.local_range(3)) = dW.local();
      mpi->comm.all_reduce_in_place_n(W_qtPQ.data(), W_qtPQ.size(), std::plus<>{});

      // head insertion, dynamic piece (weight Re[eps_inv_head(tau)]) into dW(Gamma, tau),
      // BEFORE the tau -> Matsubara transform (so the bosonic-mesh rung inherits it).
      // Same Re[.] convention as Sigma_div_correction (thc_gw.icc:506).
      if (head_ok) {
        if (mb_state.eps_inv_head.has_value()) {
          auto& eps = mb_state.eps_inv_head.value();
          utils::check(eps.shape(0) == nt_half,
                       "vertex_t::eval_Pi_C: eps_inv_head size {} != nt_half = {}.",
                       eps.shape(0), nt_half);
          for (long it = 0; it < nt_half; ++it)
            W_qtPQ(iq_gamma, it, all, all) += ComplexType(eps(it).real()) * H_PQ;
          app_log(1, "  Pi^C head insertion: dynamic piece applied to dW(Gamma, tau) "
                     "with eps_inv_head(tau=0) = {}", eps(0).real());
        } else {
          app_log(1, "  [WARNING] Pi^C: dW is present but eps_inv_head is not in MBState "
                     "-- the DYNAMIC head\n"
                     "            piece is skipped (bare piece applied).");
        }
      }

      long nw_b = tools.nw_b;
      long nw_half = (nw_b % 2 == 0) ? nw_b / 2 : nw_b / 2 + 1;
      Wdyn_qwPQ.emplace(nda::array<ComplexType, 4>(nqpts_ibz, nw_b, Np, Np));
      nda::array<ComplexType, 3> W_wpos(nw_half, Np, Np);
      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        auto W_t = W_qtPQ(iq, nda::ellipsis{});
        _ft->tau_to_w_PHsym(W_t, W_wpos);
        // unfold to the full mesh assuming W(-nu) = W(nu) (PH-symmetric storage, same
        // assumption as the SOSEX cache folding, thc_sosex.icc:970-976)
        for (long l = 0; l < nw_b; ++l) {
          long lpos = std::max(l, tools.w_mirror_b(l)) - nw_b / 2;
          Wdyn_qwPQ.value()(iq, l, all, all) = W_wpos(lpos, all, all);
        }
      }
    } else {
      app_log(1, "  [NOTE] Pi^C: no dynamic W in MBState (first iteration) -- "
                 "using the bare-interaction rung W = Z only.\n");
    }

    // ---- momentum maps (symmetry-free mesh) -------------------------------------------
    nda::array<long, 2> kmq(nqpts_ibz, nkpts), kpq(nqpts_ibz, nkpts);
    for (long iq = 0; iq < nqpts_ibz; ++iq) {
      for (long ik = 0; ik < nkpts; ++ik) kmq(iq, ik) = MF->qk_to_k2(iq, ik);
      for (long ik = 0; ik < nkpts; ++ik) kpq(iq, kmq(iq, ik)) = ik;   // inverse: (k-q)+q = k
    }

    // ---- kernel: accumulate Pi^C(inu) over this rank's (s,k,qx) tuples ----------------
    // q->0 policy resolved above (skip_rung_gamma / head-augmented inputs).
    nda::array<double, 1> qx_diag(nqpts_ibz);
    nda::array<ComplexType, 4> Pi_wqMN(tools.nw_b, nqpts_ibz, Np, Np);
    Pi_wqMN() = ComplexType(0.0);
    vertex_pi::pi_c_accumulate_w(*_ft, tools, G_tskij, X_skPa, Z_qPQ,
                                 Wdyn_qwPQ.has_value() ? &Wdyn_qwPQ.value() : nullptr,
                                 kmq, kpq, _band_window, Pi_wqMN,
                                 mpi->comm.rank(), mpi->comm.size(),
                                 skip_rung_gamma, &qx_diag);
    mpi->comm.all_reduce_in_place_n(Pi_wqMN.data(), Pi_wqMN.size(), std::plus<>{});
    mpi->comm.all_reduce_in_place_n(qx_diag.data(), qx_diag.size(), std::plus<>{});

    // per-qx rung diagnostics (sum of rank-local maxima -- order-of-magnitude indicator
    // for the q->0 head pathology; Gamma reads 0 when skipped)
    {
      long iqg = -1;
      for (long iq = 0; iq < nqpts_ibz; ++iq) {
        bool isg = true;
        for (long ik = 0; ik < nkpts; ++ik)
          if (kmq(iq, ik) != ik) { isg = false; break; }
        if (isg) { iqg = iq; break; }
      }
      double g_val = (iqg >= 0) ? qx_diag(iqg) : -1.0;
      double other = 0.0;
      for (long iqx = 0; iqx < nqpts_ibz; ++iqx) {
        app_log(3, "  Pi^C rung diagnostics: qx = {}  max|contribution| = {}", iqx, qx_diag(iqx));
        if (iqx != iqg) other = std::max(other, qx_diag(iqx));
      }
      app_log(2, "  Pi^C rung per-qx |contribution|: Gamma(iq={}) = {}, max(other qx) = {}\n",
              iqg, g_val, other);
    }

    // ---- to the code's tau storage convention -----------------------------------------
    nda::array<ComplexType, 4> Pi_tqMN(nt_half, nqpts_ibz, Np, Np);
    vertex_pi::pi_w_to_code_tau(*_ft, tools, Pi_wqMN, Pi_tqMN);

    {
      double max_abs = 0.0;
      long n_bad = 0;
      for (auto const& v : Pi_tqMN) {
        double a = std::abs(v);
        if (not std::isfinite(a)) { ++n_bad; continue; }
        max_abs = std::max(max_abs, a);
      }
      utils::check(n_bad == 0,
                   "vertex_t::eval_Pi_C: Pi^C contains {} NaN/Inf entries -- aborting.", n_bad);
      app_log(2, "  Pi^C(tau) max|.| = {}\n", max_abs);
    }

    // ---- scatter into the caller's distributed layout ---------------------------------
    auto dPi_C_tqPQ = math::nda::make_distributed_array<memory::array<HOST_MEMORY, ComplexType, 4>>(
        mpi->comm, pi_pgrid, pi_gshape, pi_bsize);
    dPi_C_tqPQ.local() = Pi_tqMN(dPi_C_tqPQ.local_range(0), dPi_C_tqPQ.local_range(1),
                                 dPi_C_tqPQ.local_range(2), dPi_C_tqPQ.local_range(3));
    mpi->comm.barrier();

    return dPi_C_tqPQ;
  }

  // template instantiations
  template void vertex_t::eval_Sigma_C(MBState&, const thc_reader_t&);

  template memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>
  vertex_t::eval_Pi_C(MBState&, const thc_reader_t&,
                      std::array<long, 4>, std::array<long, 4>, std::array<long, 4>);

}  // solvers
}  // methods
