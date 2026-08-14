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


#include <unordered_set>
#include <sstream>
#include <iomanip>
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/HF/thc_solver_comm.hpp"
#include "methods/GW/g0_div_utils.hpp"
#include "methods/vertex/vertex_t.h"
#include "hamiltonian/one_body_hamiltonian.hpp"   // scGW-tilde C4: H0 for the CVV velocity
#include "hamiltonian/pseudo/pseudopot.h"
#include "cvv_head.hpp"
#include "scr_coulomb_t.h"
#include "rpa_pi.icc"
#include "edmft_pi.icc"

namespace methods {
namespace solvers {

  bool scr_coulomb_t::has_active_vertex() const {
    return _vertex != nullptr and _vertex->active();
  }

  bool scr_coulomb_t::needs_dw_retention() const {
    // GLOBAL-basis vertex: next iteration's eval_Pi_C consumes the retained dW.
    // SECONDARY-basis vertex with the W-bar cache enabled: the downfolded rung is
    // cached at the update_w tail (vertex_t::cache_w, notes/wbar_cache.md); dW can
    // be freed unconditionally. The disabled-cache switch restores the legacy
    // retained-dW semantics (machine-identity A/B reference).
    // STATIC/LINEAR rung modes (increment S2, plan section 2.1): the dW-retention
    // exception AND the W-bar cache are RETIRED -- every rung those theories consume
    // (W0[G] and, in B-L, the same-iteration dW) is produced inside the iteration that
    // consumes it, so nothing crosses the boundary and the memory profile is plain GW's.
    return _vertex != nullptr and _vertex->active() and
           _vertex->rung() == dynamic_rung and
           (not _vertex->secondary() or not _vertex->w_cache_enabled());
  }

  bool scr_coulomb_t::vertex_has_rung(MBState const &mb_state) const {
    if (_vertex == nullptr or not _vertex->active()) return false;
    if (_vertex->secondary() and _vertex->w_cache_enabled())
      return _vertex->has_cached_w();
    return mb_state.dW_qtPQ.has_value();
  }

  scr_coulomb_t::scr_coulomb_t(const imag_axes_ft::IAFT *ft,
                               std::string screen_type,
                               std::string div):
    _ft(ft), _screen_type(screen_type),
    _div_treatment(div), _Timer() {

    const std::unordered_set<std::string> valid_pi_scheme = {
        "rpa", "rpa_r", "rpa_k",
        "crpa", "crpa_ks", "crpa_vasp",
        "gw_edmft", "gw_edmft_density",
        "gw_edmft_rpa", "gw_edmft_rpa_density",
        "gw_edmft_zero_pi_imp", "gw_edmft_zero_pi_imp_density",
        "crpa_edmft", "crpa_edmft_density"
    };
    utils::check(valid_pi_scheme.find(_screen_type)!=valid_pi_scheme.end(),
                 "scr_coulomb_t: unknown type of polarizability.");

    // Check if tau_mesh is symmetric w.r.t. beta/2
    auto tau_mesh = _ft->tau_mesh();
    long nts = tau_mesh.shape(0);
    for (size_t it = 0; it < nts; ++it) {
      size_t imt = nts - it - 1;
      double diff = std::abs(tau_mesh(it)) - std::abs(tau_mesh(imt));
      utils::check(diff <= 1e-6, "scr_coulomb_t: IAFT grid is not compatible with particle-hole symmetry. {}, {}",
                   tau_mesh(it), tau_mesh(imt));
    }
  }

  void scr_coulomb_t::update_w(MBState &mb_state, THC_ERI auto &thc, long h5_iter) {
    using math::nda::make_distributed_array;
    using math::shm::make_shared_array;

    // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20screened%20coulomb
    app_log(1, "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌─┐┌─┐┬─┐┌─┐┌─┐┌┐┌┌─┐┌┬┐  ┌─┐┌─┐┬ ┬┬  ┌─┐┌┬┐┌┐ \n"
               "║  ║ ║║═╬╗║ ║║  └─┐│  ├┬┘├┤ ├┤ │││├┤  ││  │  │ ││ ││  │ ││││├┴┐\n"
               "╚═╝╚═╝╚═╝╚╚═╝╩  └─┘└─┘┴└─└─┘└─┘┘└┘└─┘─┴┘  └─┘└─┘└─┘┴─┘└─┘┴ ┴└─┘\n");
    app_log(1, "  Screening type                = {}\n"
               "  Number of bands               = {}\n"
               "  Number of THC auxiliary basis = {}\n"
               "  K-points                      = {} total, {} in the IBZ\n"
               "  Divergent treatment at q->0   = {}\n",
            _screen_type, thc.MF()->nbnd(), thc.Np(),
            thc.MF()->nkpts(), thc.MF()->nkpts_ibz(),
          _div_treatment);
    _ft->metadata_log();

    utils::check(thc.mpi() == mb_state.mpi,
                 "scr_coulomb_t::update_w: THC_ERI and MBState should have the same MPI context.");

    // ---- ISDF-Vertex BOOTSTRAP: a SCREENED rung for Pi^C on the first update ---------
    // Pi^C = -2 dPhi_2^C/dW is a functional of the SCREENED interaction, but on the very
    // first update of a run no W exists yet -- neither a retained dW (global path) nor a
    // folded W-bar cache (secondary path) -- and eval_Pi_C falls back to the BARE rung
    // W = Z. That is not a benign startup detail: on Si kp444 with C = [0, 8) it drives
    // iteration 1 to epsilon_inf = 19.6 against a converged RPA value of 5.35, and the
    // resulting grossly over-screened W feeds Sigma^GW, Sigma^C and hence G for every
    // subsequent iteration -- the observed trajectory never recovers.
    // FIX: on that first update only, solve the RPA problem FIRST with the vertex
    // detached and publish its W, then redo the update with the vertex attached so Pi^C
    // starts from a physically screened rung. The self-consistent FIXED POINT is
    // unchanged (there Pi^C already consumes the converged W); only the starting point
    // of the iteration moves. Cost: one extra RPA polarization + Dyson solve, once.
    // The bootstrap is a DYNAMIC-rung device only. In the static modes (increment S2,
    // plan section 2.2 "Bonus") the rung is W0[G], built below from THIS iteration's RPA
    // polarizability BEFORE any vertex piece runs -- so a physically screened rung exists
    // from iteration 1 by construction, the bare-rung basin the bootstrap was added to
    // escape is structurally absent, and the extra RPA + Dyson solve would be pure waste.
    // The seam itself stays (it is what the dynamic theory needs, and B-L's mixed terms
    // still consume the same-iteration dW).
    if (has_active_vertex() and _vertex->rung() == dynamic_rung
        and not vertex_has_rung(mb_state)) {
      app_log(1, "  [ISDF-Vertex] bootstrap: no screened W is available yet, so Pi^C would "
                 "use the BARE\n"
                 "                rung W = Z. Solving the RPA problem first and re-running "
                 "this update\n"
                 "                with the vertex attached (one extra RPA + Dyson solve, "
                 "this iteration only).\n");
      auto *vtx = _vertex;
      _vertex = nullptr;                      // RPA-only pass (no Pi^C, no cache_w)
      update_w(mb_state, thc, -1);            // publishes dW + eps_inv_head
      _vertex = vtx;
      // secondary path: fold the freshly published RPA W into the N_m x N_m rung cache,
      // exactly as the update_w tail would have done had the vertex been attached.
      if (_vertex->secondary() and _vertex->w_cache_enabled())
        _vertex->cache_w(mb_state, thc);
    }

    if (_screen_type.find("edmft") != std::string::npos) {
      if (!mb_state.sPi_imp_wabcd or !mb_state.sPi_dc_wabcd) {
        if (mb_state.read_local_polarizabilities()) {
          app_log(1, "scr_coulomb_t::update_w: "
                     "No local polarizabilities found in MBState \n"
                     "-> reading from checkpoint file.\n");
        } else {
          app_log(1, "scr_coulomb_t::update_w: "
                     "No local polarizabilities found in MBState or checkpoint file \n"
                     "-> Setting to zero.\n");
        }
      }
    }
    // qpGW Q4 (notes/q4_edmft_skeleton_spec.md, ruling R-Q4-3): the Q3 BSE tier -- the
    // ladder kernel build AND the injection -- now runs INSIDE eval_Pi_qdep, at the pinned
    // points (pure-RPA kernel; injection last, still before the Dyson). What stays here is
    // the eps_M READOUT, which needs the post-Dyson head: it consumes the inu = 0 RPA row
    // stashed at that pure-RPA point (_pol_pi0_qPQ) instead of gathering it here.
    const bool pol_readout = (_vertex != nullptr and _vertex->pol_vertex_active()
                              and not _vertex->active());
    auto dPi_tqPQ = eval_Pi_qdep(mb_state, thc);

    // evaluate screened interaction (dW_tqPQ) and reset polarizability (dPi_tqPQ)
    // a) dPi_tqPQ is reset during dyson_W_from_Pi_tau()
    // b) pgrid and bsize of dW_tqPQ are forced to be the same as in dPi_tqPQ
    auto dW_tqPQ = dyson_W_from_Pi_tau<false>(dPi_tqPQ, thc, true);
    // scGW-tilde C4 (div_treatment = "cvv"): the q -> 0 HEAD comes from the
    // covariant-velocity subtracted head (eval_cvv_eps_inv_head) INSTEAD of the
    // stored/gygi extrapolation; the q-RESOLVED eps_inv (diagnostics + dump) is
    // div-treatment-independent, so eps_inv_head_t runs with "ignore_g0" (its head
    // slot -- the smallest-q value -- is then replaced). Every consumer reads the
    // same mb_state.eps_inv_head (single-sourcing; vertex_t.h coupling warning).
    const bool cvv = (_div_treatment == "cvv");
    auto [eps_inv_head_q, eps_inv_head] =
        div_utils::eps_inv_head_t(dW_tqPQ, thc, *thc.MF(), _ft,
                                  cvv ? "ignore_g0" : _div_treatment);
    if (cvv) eps_inv_head = eval_cvv_eps_inv_head(mb_state, thc);
    mb_state.eps_inv_head = eps_inv_head;

    // ISDF-Vertex: report the static macroscopic dielectric constant
    //   epsilon_inf = 1 / Re[ eps^{-1}_head(q->0, i.nu = 0) ],
    // with eps_inv_head the q->0-extrapolated head of the inverse dielectric (in tau).
    // The vertex correction P^C enters automatically through Pi -> W -> eps_inv_head, so with
    // an active vertex this is the vertex-corrected epsilon_inf; without it, the RPA value.
    // Logged every iteration and written to the h5 checkpoint by dump_eps_inv_head below.
    {
      // NB: eps_inv_head stores (eps^{-1}_head - 1)  (see g0_div_utils::eval_eps_inv_q,
      // "Computes eps^{-1}_{G=0,G'=0} - 1"; the metal branch sets it to -1 so eps^{-1}=0).
      // The PHYSICAL inverse-dielectric head is therefore 1 + eps_inv_head, and
      //   epsilon_inf = 1 / Re[ eps^{-1}_head(inu=0) ] = 1 / (1 + Re[eps_inv_head_w(inu=0)]).
      long nw_half = (_ft->nw_b() % 2 == 0) ? _ft->nw_b() / 2 : _ft->nw_b() / 2 + 1;
      nda::array<ComplexType, 2> eih_w(nw_half, 1);
      auto eih_t = nda::reshape(eps_inv_head, shape_t<2>{eps_inv_head.shape(0), 1});
      _ft->tau_to_w_PHsym(eih_t, eih_w);   // inu=0 (static) node = index 0 of the PH-sym half grid
      ComplexType eps_inv_static = ComplexType(1.0) + eih_w(0, 0);   // physical eps^{-1}_head(inu=0)
      double eps_inf = 1.0 / eps_inv_static.real();
      app_log(1, "  Macroscopic dielectric constant (static, q->0):\n"
                 "    epsilon_inf = {:.6f}   [eps^-1_head(inu=0) = {:.6e} {:+.6e}i]\n",
              eps_inf, eps_inv_static.real(), eps_inv_static.imag());
    }

    // scGW-tilde L2: the ladder eps_M readout (report-only; see pol_ladder_eps_readout).
    // Q3: eps_inv_head_q carries the loop's OWN q-resolved head, so the readout also
    // reports the loop-side eps_M(q_min) -- the second route of gate Q3-b(i).
    if (pol_readout)
      pol_ladder_eps_readout(mb_state, thc, _pol_pi0_qPQ, std::addressof(eps_inv_head_q));

    // make routine to transposed distributed arrays over any 2 indices, so should
    // be easy to template to an array type and to indexes, and replace repeated code
    auto t_pgrid = dW_tqPQ.grid();
    auto t_bsize = dW_tqPQ.block_size();
    auto gshape = dW_tqPQ.global_shape();
    mb_state.dW_qtPQ.emplace(make_distributed_array<nda::array<ComplexType, 4>> (
                             thc.mpi()->comm, {t_pgrid[1], t_pgrid[0], t_pgrid[2], t_pgrid[3]},
                             {gshape[1], gshape[0], gshape[2], gshape[3]},
                             {t_bsize[1], t_bsize[0], t_bsize[2], t_bsize[3]}));
    auto W_tqPQ = dW_tqPQ.local();
    auto W_qtPQ = mb_state.dW_qtPQ.value().local();
    long nt_loc = dW_tqPQ.local_shape()[0];
    long nq_loc = dW_tqPQ.local_shape()[1];
    for (size_t qt = 0; qt < nq_loc * nt_loc; ++qt) {
      size_t iq = qt / nt_loc;
      size_t it = qt % nt_loc;
      W_qtPQ(iq, it, nda::ellipsis{}) = W_tqPQ(it, iq, nda::ellipsis{});
    }
    dW_tqPQ.reset();

    mb_state.screen_type = _screen_type;

    if (h5_iter>=0) {
      dump_eps_inv_head(eps_inv_head_q, eps_inv_head,
                        mb_state.coqui_prefix, h5_iter,
                        thc.mpi()->comm, *thc.MF());
      // Q4 C3: publish the ladder half of the eq-7 bosonic DC next to the other scf/iter
      // outputs so BOTH consumers can read it -- python's DC assembly (weiss.py) and the
      // C++ bosonic closure (downfold_edmft_impl). Written only when THIS update_w
      // injected (a stale MBState copy must not be re-published), and never as a separate
      // file (the eval_Pi_rpa_dc "pi_rpa_loc_debug.h5" wart is not copied).
      // Q6 §1.4(a) widened the ENCLOSING condition from "... and sPi_lad_loc_wabcd" to
      // "the ladder was injected": the scalar meters below exist on every injecting run,
      // including the ones with no bosonic projector (where P^lad_loc is never built).
      // The pi_lad_loc datasets keep their ORIGINAL condition, one level in.
      if (pol_readout and _vertex->pol_vertex_inject_enabled() and thc.mpi()->comm.root()) {
        h5::file file(mb_state.coqui_prefix + ".mbpt.h5", 'a');
        h5::group grp(file);
        auto scf_grp = (grp.has_subgroup("scf")) ? grp.open_group("scf")
                                                 : grp.create_group("scf");
        std::string grp_name = "iter" + std::to_string(h5_iter);
        auto iter_grp = (scf_grp.has_subgroup(grp_name)) ? scf_grp.open_group(grp_name)
                                                         : scf_grp.create_group(grp_name);
        if (mb_state.sPi_lad_loc_wabcd) {
          nda::h5_write(iter_grp, "pi_lad_loc_wabcd",
                        mb_state.sPi_lad_loc_wabcd.value().local(), false);
          // Q4-C3b: the DC-ready orbital/chi-convention object rides in the same group
          // (dataset name distinct -- the consumers select with pi_lad_dc).
          if (mb_state.sPi_lad_loc_orb_wabcd)
            nda::h5_write(iter_grp, "pi_lad_loc_orb_wabcd",
                          mb_state.sPi_lad_loc_orb_wabcd.value().local(), false);
        }
        // Q6 §1.4(a): PERSIST the Q3 injection meters next to the object they describe.
        // Before this, python's Q5-b trail had no source for lambda_nu0 and carried the
        // MISSING = -1 sentinel forever (outer_loop.py:62-66). These are the SAME numbers
        // the pol_lambda_nu0()/pol_lambda_max()/pol_round_trip()/pol_ladder_ratio()
        // accessors return for THIS update_w -- read, not recomputed.
        h5::h5_write(iter_grp, "lambda_nu0", _pol_lam_nu0);
        h5::h5_write(iter_grp, "lambda_max", _pol_lam_max);
        h5::h5_write(iter_grp, "r_rt", _pol_r_rt);
        h5::h5_write(iter_grp, "lad_ratio", _pol_lad_ratio);
      }
      thc.mpi()->comm.barrier();
    }

    // ISDF-Vertex Refinement 2, W-bar iteration cache (notes/wbar_cache.md): with an
    // active SECONDARY-basis vertex, fold the freshest W into the N_m x N_m cache now
    // -- dW is alive here and mb_state.eps_inv_head is the SAME-iteration head (both
    // stored above), so the gygi Gamma augmentation is captured consistently. The
    // cache is consumed by the NEXT iteration's eval_Pi_C (identical one-iteration
    // lag as the retained-dW path); the scf driver then frees dW unconditionally in
    // this mode (needs_dw_retention() == false -- plain-GW memory profile).
    // (dynamic rung only: the static modes retired the cache -- see needs_dw_retention)
    if (_vertex != nullptr and _vertex->active() and _vertex->rung() == dynamic_rung
        and _vertex->secondary() and _vertex->w_cache_enabled())
      _vertex->cache_w(mb_state, thc);
  }

  // scGW-tilde C4: see the declaration in scr_coulomb_t.h for the contract. The
  // returned array matches div_utils::eps_inv_head_t's head slot exactly: the PH-sym
  // tau half grid storing (eps^{-1}_head - 1)(tau).
  nda::array<ComplexType, 1> scr_coulomb_t::eval_cvv_eps_inv_head(MBState &mb_state,
                                                                  THC_ERI auto &thc) {
    mf::MF &mf = *thc.MF();
    utils::check(mb_state.sF_skij.has_value() and mb_state.sG_tskij.has_value(),
                 "eval_cvv_eps_inv_head: mb_state must carry F and G at update_w time.");
    if (not _sH0_cvv.has_value()) {
      _sH0_cvv = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(
          *thc.mpi(), {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()});
      auto psp = hamilt::make_pseudopot(mf);
      hamilt::set_H0(mf, psp.get(), _sH0_cvv.value());
    }

    solvers::cvv_head_t cvv(_ft, _cvv_rspace_tol);
    if (mb_state.sSigma_tskij.has_value())
      cvv.build(mf, _sH0_cvv.value().local(), mb_state.sF_skij.value().local(),
                mb_state.sSigma_tskij.value().local());
    else {
      nda::array<ComplexType, 5> sig_empty(0, 0, 0, 0, 0);
      cvv.build(mf, _sH0_cvv.value().local(), mb_state.sF_skij.value().local(), sig_empty);
    }
    auto head = cvv.eval_head_tensor(mf, mb_state.sG_tskij.value().local());

    // eps^{-1}_head(inu) - 1 on the PH-sym half grid (full-grid index i0 + j):
    // scalar head Dyson per cartesian direction, angular average of eps^{-1}
    const long nwb = _ft->nw_b(), i0 = nwb / 2;
    const long nw_half = (nwb % 2 == 0) ? nwb / 2 : nwb / 2 + 1;
    nda::array<ComplexType, 2> Ew(nw_half, 1);
    for (long j = 0; j < nw_half; ++j) {
      ComplexType acc(0.0);
      for (int a = 0; a < 3; ++a)
        acc += 1.0 / (1.0 - 4.0 * M_PI * head.Phead_wab(i0 + j, a, a));
      Ew(j, 0) = acc / 3.0 - 1.0;
    }
    // T-d meter (PDF G-c): v(q).P00 at the head; the pre-fix runs showed it climbing
    // toward 1 (dielectric collapse carries the J > 1 feedback)
    double td = 0.0;
    for (int a = 0; a < 3; ++a)
      td = std::max(td, std::abs(4.0 * M_PI * head.Phead_wab(i0, a, a)));
    app_log(1, "  [CVV] head (div_treatment = cvv): eps_inf(x, y, z) = "
               "({:.6f}, {:.6f}, {:.6f}); T-d meter v.P00 = {:.4f}{}",
            1.0 - 4.0 * M_PI * head.Phead_wab(i0, 0, 0).real(),
            1.0 - 4.0 * M_PI * head.Phead_wab(i0, 1, 1).real(),
            1.0 - 4.0 * M_PI * head.Phead_wab(i0, 2, 2).real(), td,
            (td > 0.9) ? "  [WARNING: v.P00 approaching 1 -- dielectric collapse]" : "");

    nda::array<ComplexType, 1> Et(_ft->nt_b() % 2 == 0 ? _ft->nt_b() / 2
                                                       : _ft->nt_b() / 2 + 1);
    auto Et_2D = nda::reshape(Et, shape_t<2>{Et.shape(0), 1});
    _ft->w_to_tau_PHsym(Ew, Et_2D);
    return Et;
  }

  // ---- scGW-tilde L2 helpers (pol_vertex = "ladder" readout; stance i) ---------------

  void scr_coulomb_t::ensure_pol_vertex(THC_ERI auto &thc) {
    if (_pol_vtx) return;
    utils::check(_vertex != nullptr, "ensure_pol_vertex: no knob carrier attached.");
    auto w = _vertex->pol_band_window();
    _pol_vtx = std::make_shared<vertex_t>(
        _ft, "2nd_exchange", w, thc.MF()->nbnd(), _div_treatment, "secondary",
        _vertex->pol_isdf_rank(), _vertex->pol_isdf_svd_tol(),
        _vertex->pol_isdf_thresh(), _vertex->pol_isdf_cond_max(), "static");
    _pol_vtx->set_isdf_distr_tol(_vertex->pol_isdf_distr_tol());
    app_log(1, "  [scGW-tilde L2] ladder readout instance: C window = [{}, {}), "
               "secondary rank knob = {}, div_treatment = {} (kernel head follows "
               "build_w0's policy; W0bar is SAME-iteration -- coincides with "
               "pol_vertex_kernel = \"w0_prev\" at a fixed point, R4 note).",
            w.first(), w.last(), _vertex->pol_isdf_rank(), _div_treatment);
  }

  template<nda::MemoryArrayOfRank<4> Array_t, typename communicator_t>
  nda::array<ComplexType, 3>
  scr_coulomb_t::gather_nu0_row(memory::darray_t<Array_t, communicator_t> &dPi_tqPQ) {
    auto [nt_h, nq, Np, Nq2] = dPi_tqPQ.global_shape();
    auto R = solvers::vertex_w0_detail::nu0_transform_row(*_ft);
    utils::check(R.shape(0) == nt_h,
                 "gather_nu0_row: PH-sym tau half grid mismatch ({} vs {}).",
                 R.shape(0), nt_h);
    auto t_rng = dPi_tqPQ.local_range(0);
    auto q_rng = dPi_tqPQ.local_range(1);
    auto P_rng = dPi_tqPQ.local_range(2);
    auto Q_rng = dPi_tqPQ.local_range(3);
    auto Pi_loc = dPi_tqPQ.local();
    nda::array<ComplexType, 3> out(nq, Np, Nq2);
    out() = ComplexType(0.0);
    for (long it = 0; it < long(t_rng.size()); ++it) {
      const ComplexType r = R(t_rng.first() + it);
      for (long iq = 0; iq < long(q_rng.size()); ++iq)
        for (long iP = 0; iP < long(P_rng.size()); ++iP)
          for (long iQ = 0; iQ < long(Q_rng.size()); ++iQ)
            out(q_rng.first() + iq, P_rng.first() + iP, Q_rng.first() + iQ) +=
                r * Pi_loc(it, iq, iP, iQ);
    }
    dPi_tqPQ.communicator()->all_reduce_in_place_n(out.data(), out.size(), std::plus<>{});
    return out;
  }

  /**
   * qpGW Q4 increment C3 (notes/q4_edmft_skeleton_spec.md, ruling R-Q4-2): the LADDER half
   * of the eq-7 bosonic double counting,
   *
   *   P^lad_loc(i.nu)_abcd = (1/N_q) sum_q [ B(q)^dag (t(q)^dag Pl(i.nu, q) t(q)) B(q) ]_abcd,
   *
   * i.e. the exact downfold ADJOINT of the upfold chain the EDMFT correction uses
   * (upfold_pi_local, edmft_pi.icc:61-79: Pi_up(P,Q) = B(P;ab) Pi_ab,cd conj(B(Q;cd));
   * its adjoint is D_ab,cd = conj(B(P;ab)) X_PQ B(Q;cd)). Unlike the bubble part of P_dc
   * there is no impurity-side counterpart object, so eq 7's "what the lattice already
   * contains" is the definition (R-Q4-2).
   *
   * ⚠ CONVENTION CAVEAT -- NOT A DC-READY OBJECT (R-Q4-2 AMENDMENT,
   * notes/q4_edmft_skeleton_spec.md): the ADJOINT of the upfold is not its INVERSE --
   * upfold_pi_local has gain ||B||^2 (a local Pi of O(1) upfolds onto the O(10^4) scale the
   * THC-basis Pi lives on), so the adjoint carries that same gain instead of its reciprocal
   * and this object lands ~10 orders above bubble[G_loc]; the amendment rules that neither
   * the THC adjoint nor the s^-1 = (B^dag B)^-1 dual belongs in the bosonic double counting
   * (the DC-ready object is the orbital/chi-convention 4-leg MLWF U-leg projection of the
   * pair-space ladder, deferred to increment Q4-C3b). What is kept here is the INTERFACE
   * DIAGNOSTIC: the ratio meter logged below is the cancellation-load column that exposed
   * the convention, and the only consumer is opt-in and off by default
   * (downfold_edmft_impl's pi_lad_dc = "thc_adjoint_diag").
   *
   * q-WEIGHTS: B carries the FULL BZ q axis (and its own 1/N_q from
   * calc_bosonic_projector), Pl and t carry the IBZ axis. The star multiplicities are
   * therefore taken exactly as downfold_W does (embed_eri_t.cpp:2124-2146): loop the FULL
   * q mesh, map each q to its IBZ parent, conjugate the parent's matrix when qp_trev, and
   * divide by nqpts once at the end.
   *
   * COST: the Np x Np upfold is never formed. With Y(q) = t(q) B(q) (N_m x nImpOrbs^2, one
   * gemm per q) the whole object is Y^dag Pl Y -- exactly the same number by associativity,
   * at readout scale. Under time reversal the same expression holds with t -> conj(t) and
   * Pl -> conj(Pl) (both conjugations follow from conj(t^dag Pl t)).
   */
  void scr_coulomb_t::accumulate_pi_lad_loc(MBState &mb_state, THC_ERI auto &thc,
                                            nda::array<ComplexType, 4> const &Pl,
                                            nda::array<ComplexType, 3> const &tmap) {
    decltype(nda::range::all) all;
    auto &proj_boson = mb_state.proj_boson.value();
    utils::check(proj_boson.nImps() == 1,
                 "scr_coulomb_t::accumulate_pi_lad_loc: P^lad_loc is implemented for a "
                 "SINGLE impurity only (nImps = {}); the upfold it is the adjoint of is "
                 "single-impurity too (edmft_pi.icc:74/78).", proj_boson.nImps());
    mf::MF &mf = *thc.MF();
    const long nw_h = Pl.shape(0), Nm = Pl.shape(2);
    const long nImpOrbs = proj_boson.nImpOrbs(), nab = nImpOrbs * nImpOrbs;
    const long nqpts = mf.nqpts(), Np = tmap.shape(2);
    const long nw_half = (_ft->nw_b() % 2 == 0) ? _ft->nw_b() / 2 : _ft->nw_b() / 2 + 1;
    utils::check(nw_h == nw_half,
                 "scr_coulomb_t::accumulate_pi_lad_loc: the ladder's PH-sym half grid ({}) "
                 "does not match the local-polarizability one ({}).", nw_h, nw_half);

    auto sB_qIPab = (mf.nqpts_ibz() == mf.nqpts()) ?
                    proj_boson.calc_bosonic_projector(thc) :
                    proj_boson.calc_bosonic_projector_symm(thc);
    utils::check(sB_qIPab.shape()[0] == nqpts and sB_qIPab.shape()[2] == Np,
                 "scr_coulomb_t::accumulate_pi_lad_loc: bosonic projector shape mismatch "
                 "({} x {} vs nqpts = {}, Np = {}).",
                 sB_qIPab.shape()[0], sB_qIPab.shape()[2], nqpts, Np);

    if (not mb_state.sPi_lad_loc_wabcd or
        mb_state.sPi_lad_loc_wabcd.value().shape()[0] != nw_h)
      mb_state.sPi_lad_loc_wabcd.emplace(
          math::shm::make_shared_array<Array_view_5D_t>(
              *thc.mpi(), {nw_h, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs}));

    auto B_loc = nda::reshape(sB_qIPab.local(), shape_t<3>{nqpts, Np, nab});
    double lmax = 0.0;
    mb_state.sPi_lad_loc_wabcd.value().win().fence();
    if (thc.mpi()->node_comm.root()) {
      auto D = nda::reshape(mb_state.sPi_lad_loc_wabcd.value().local(),
                            shape_t<3>{nw_h, nab, nab});
      D() = ComplexType(0.0);
      nda::matrix<ComplexType> T(Nm, Np), Bq(Np, nab), Y(Nm, nab), core(Nm, Nm);
      nda::matrix<ComplexType> CY(Nm, nab), acc(nab, nab);
      for (long iq = 0; iq < nqpts; ++iq) {
        const long iq_ibz = mf.qp_to_ibz(iq);
        const bool trev = mf.qp_trev(iq);
        T = trev ? nda::matrix<ComplexType>(nda::conj(tmap(iq_ibz, all, all)))
                 : nda::matrix<ComplexType>(tmap(iq_ibz, all, all));
        Bq = B_loc(iq, all, all);
        nda::blas::gemm(T, Bq, Y);                        // Y = t(q) B(q)
        for (long j = 0; j < nw_h; ++j) {
          core = trev ? nda::matrix<ComplexType>(nda::conj(Pl(j, iq_ibz, all, all)))
                      : nda::matrix<ComplexType>(Pl(j, iq_ibz, all, all));
          nda::blas::gemm(core, Y, CY);
          nda::blas::gemm(ComplexType(1.0), nda::dagger(Y), CY, ComplexType(0.0), acc);
          D(j, all, all) += acc;
        }
      }
      D() /= double(nqpts);
      for (auto const &v : D) lmax = std::max(lmax, std::abs(v));
    }
    mb_state.sPi_lad_loc_wabcd.value().win().fence();
    thc.mpi()->comm.barrier();
    _pol_lad_loc_max = thc.mpi()->comm.all_reduce_value(lmax, boost::mpi3::max<>{});

    // gate Q4-c3(iii): the cancellation-load meter of PDF section 8.3 gets its ladder
    // column -- ||P^lad_loc|| against the bubble part of P_dc when the latter is present
    // (it is absent in the C = empty leg, where there is nothing to compare against).
    _pol_lad_loc_ratio = -1.0;
    if (mb_state.sPi_dc_wabcd) {
      double dmax = 0.0;
      for (auto const &v : mb_state.sPi_dc_wabcd.value().local())
        dmax = std::max(dmax, std::abs(v));
      _pol_dc_bubble_max = thc.mpi()->comm.all_reduce_value(dmax, boost::mpi3::max<>{});
      if (_pol_dc_bubble_max > 0.0)
        _pol_lad_loc_ratio = _pol_lad_loc_max / _pol_dc_bubble_max;
      app_log(1, "  [qpGW Q4] ||P^lad_loc||_max = {:.4e} vs ||P_dc,bubble||_max = {:.4e} "
                 "(ratio {:.3e})", _pol_lad_loc_max, _pol_dc_bubble_max, _pol_lad_loc_ratio);
    } else {
      _pol_dc_bubble_max = -1.0;
      app_log(1, "  [qpGW Q4] ||P^lad_loc||_max = {:.4e} (no bubble P_dc present to "
                 "compare against)", _pol_lad_loc_max);
    }
  }

  /**
   * qpGW Q4 increment C3b (notes/q4_c3b_orbital_ladder_dc_spec.md; the R-Q4-2 AMENDMENT):
   * the eq-7 bosonic DC's ladder half PROPER,
   *
   *   P^lad_loc,orb(i.nu)_abcd = (1/N_q) sum_q [ E(q)^dag ((1-XK)^-1 XKX)(q, i.nu) E(q) ],
   *   E(k; q)_{(a nc + b),(m norb + n)} = U_{m a}(k) conj(U_{n b}(k + q)),
   *
   * i.e. the SAME pair-space sandwich the lattice ladder is, contracted with the MLWF pair
   * legs instead of the THC density collapse -- an ORBITAL/chi-convention object on the
   * scale of bubble[G_loc], with no metric inverse anywhere (that is the whole point of the
   * amendment: the C3 THC adjoint carries the upfold's ||B||^2 gain, this does not). Leg
   * derivation + the chi-convention statement: vertex_ladder.icc header; pinned by
   * vertex_t::ladder_loc_gate (gates G2/G3).
   *
   * THE LEGS: the ONE fermionic projector, proj_boson.proj_fermi()'s C_skIai on the FULL BZ
   * k axis, placed on the LADDER window's columns. The projector's band window must lie
   * INSIDE the ladder window -- otherwise the projector has weight on bands the ladder
   * never resummed and the object would silently drop it. Both windows are logged.
   *
   * ⚠ WHERE THAT REQUIREMENT ABORTS: this accumulator is OPPORTUNISTIC (it runs on every
   * injection that has a bosonic projector, whether or not anyone asked for the ladder DC),
   * so an incompatible pair of windows here is a SKIP with a loud warning, not a fatal --
   * the shipped lih222 fixture is itself incompatible (projector [0, 2) vs the Q3/Q4 ladder
   * window [1, 3)) and those runs must keep working. The FATAL lives where the object is
   * actually demanded: downfold_edmft_impl's pi_lad_dc = "orbital" aborts when no
   * P^lad_loc,orb is found rather than silently downgrading the DC.
   *
   * q-WEIGHTS: identical to accumulate_pi_lad_loc -- loop the FULL q mesh, map to the IBZ
   * parent, conjugate on qp_trev, divide by nqpts once. (P^orb(-q) = conj(P^orb(q)) follows
   * from the trev relations of U and G, exactly as conj(t^dag Pl t) does there. Both fixture
   * meshes of this increment are unsymmetrized, so the trev branch is UNEXERCISED -- kept,
   * flagged, not faked.)
   */
  void scr_coulomb_t::accumulate_pi_lad_loc_orb(MBState &mb_state, THC_ERI auto &thc) {
    decltype(nda::range::all) all;
    auto &proj_boson = mb_state.proj_boson.value();
    utils::check(proj_boson.nImps() == 1,
                 "scr_coulomb_t::accumulate_pi_lad_loc_orb: implemented for a SINGLE "
                 "impurity only (nImps = {}).", proj_boson.nImps());
    utils::check(_pol_vtx != nullptr,
                 "scr_coulomb_t::accumulate_pi_lad_loc_orb: the ladder instance is absent.");
    mf::MF &mf = *thc.MF();
    auto C_skIai = proj_boson.proj_fermi().C_skIai();      // (ns, nk, nImps, norb, nOrbs_W)
    auto const &W_rng = proj_boson.W_rng()[0];
    auto bw = _pol_vtx->band_window();
    if (W_rng.first() < bw.first() or W_rng.last() > bw.last()) {
      app_log(1, "\n  [WARNING] Q4-C3b: the eq-7 ladder DC (orbital convention) was NOT "
                 "produced.\n            The bosonic projector's band window [{}, {}) is "
                 "not contained in the\n            ladder (C) window [{}, {}), so the "
                 "local projection would drop bands the\n            ladder never "
                 "resummed. Widen pol_vertex's window (or narrow the\n            "
                 "projector) if you intend to consume pi_lad_dc = \"orbital\".",
              W_rng.first(), W_rng.last(), bw.first(), bw.last());
      mb_state.sPi_lad_loc_orb_wabcd.reset();
      _pol_lad_loc_orb_max = -1.0;
      _pol_lad_loc_orb_ratio = -1.0;
      return;
    }

    const long ns = C_skIai.shape(0), nk = C_skIai.shape(1);
    const long norb = proj_boson.nImpOrbs(), nab = norb * norb, nc = bw.size();
    utils::check(nk == mf.nkpts() and ns == mf.nspin(),
                 "scr_coulomb_t::accumulate_pi_lad_loc_orb: projector axes ({} spins, {} "
                 "k-points) do not match the mean field ({}, {}).", ns, nk, mf.nspin(),
                 mf.nkpts());
    utils::check(C_skIai.shape(4) == W_rng.size(),
                 "scr_coulomb_t::accumulate_pi_lad_loc_orb: projector column count {} != "
                 "band window size {}.", C_skIai.shape(4), W_rng.size());

    // U on the LADDER window's columns: zero on the bands the projector does not span
    nda::array<ComplexType, 4> U_skia(ns, nk, norb, nc);
    U_skia() = ComplexType(0.0);
    const long off = W_rng.first() - bw.first();
    for (long is = 0; is < ns; ++is)
      for (long ik = 0; ik < nk; ++ik)
        for (long m = 0; m < norb; ++m)
          for (long j = 0; j < W_rng.size(); ++j)
            U_skia(is, ik, m, off + j) = C_skIai(is, ik, 0, m, j);
    app_log(2, "  [qpGW Q4-C3b] MLWF ladder legs: projector window [{}, {}) inside the "
               "ladder window [{}, {}) (offset {}), {} impurity orbitals.",
            W_rng.first(), W_rng.last(), bw.first(), bw.last(), off, norb);

    auto Ploc = _pol_vtx->eval_pol_ladder_loc_whalf(mb_state, thc, U_skia);
    const long nw_h = Ploc.shape(0), nq_ibz = Ploc.shape(1), nqpts = mf.nqpts();
    const long nw_half = (_ft->nw_b() % 2 == 0) ? _ft->nw_b() / 2 : _ft->nw_b() / 2 + 1;
    utils::check(nw_h == nw_half and Ploc.shape(2) == nab,
                 "scr_coulomb_t::accumulate_pi_lad_loc_orb: evaluator returned {} x {} x "
                 "{} rows, expected {} x {} x {}.", nw_h, nq_ibz, Ploc.shape(2), nw_half,
                 mf.nqpts_ibz(), nab);

    if (not mb_state.sPi_lad_loc_orb_wabcd or
        mb_state.sPi_lad_loc_orb_wabcd.value().shape()[0] != nw_h)
      mb_state.sPi_lad_loc_orb_wabcd.emplace(
          math::shm::make_shared_array<Array_view_5D_t>(
              *thc.mpi(), {nw_h, norb, norb, norb, norb}));

    double lmax = 0.0;
    mb_state.sPi_lad_loc_orb_wabcd.value().win().fence();
    if (thc.mpi()->node_comm.root()) {
      auto D = nda::reshape(mb_state.sPi_lad_loc_orb_wabcd.value().local(),
                            shape_t<3>{nw_h, nab, nab});
      D() = ComplexType(0.0);
      for (long iq = 0; iq < nqpts; ++iq) {
        const long iq_ibz = mf.qp_to_ibz(iq);
        const bool trev = mf.qp_trev(iq);
        for (long j = 0; j < nw_h; ++j) {
          auto Pq = Ploc(j, iq_ibz, all, all);
          if (trev)
            D(j, all, all) += nda::conj(Pq);
          else
            D(j, all, all) += Pq;
        }
      }
      D() /= double(nqpts);
      for (auto const &v : D) lmax = std::max(lmax, std::abs(v));
    }
    mb_state.sPi_lad_loc_orb_wabcd.value().win().fence();
    thc.mpi()->comm.barrier();
    _pol_lad_loc_orb_max = thc.mpi()->comm.all_reduce_value(lmax, boost::mpi3::max<>{});

    // gate G4: the scale statement of the amendment -- this object must sit ON the scale of
    // bubble[G_loc] (the THC adjoint sat ~10 orders above it).
    _pol_lad_loc_orb_ratio = -1.0;
    if (mb_state.sPi_dc_wabcd and _pol_dc_bubble_max > 0.0) {
      _pol_lad_loc_orb_ratio = _pol_lad_loc_orb_max / _pol_dc_bubble_max;
      app_log(1, "  [qpGW Q4-C3b] ||P^lad_loc,orb||_max = {:.4e} vs ||P_dc,bubble||_max = "
                 "{:.4e} (ratio {:.3e})", _pol_lad_loc_orb_max, _pol_dc_bubble_max,
              _pol_lad_loc_orb_ratio);
    } else {
      app_log(1, "  [qpGW Q4-C3b] ||P^lad_loc,orb||_max = {:.4e} (no bubble P_dc present "
                 "to compare against)", _pol_lad_loc_orb_max);
    }
  }

  /**
   * qpGW Q3 increment I2 (notes/q3_bse_tier_spec.md section 4): the BSE tier injection.
   *
   *   P_latt(q, i.nu) = P^RPA(q, i.nu) + P^lad(q, i.nu),   P^lad = eq 6's [.]_{n >= 2}
   *
   * The implemented pair-space ladder resums rungs >= 1, i.e. chi0-FACTOR counts >= 2, so
   * it IS eq 6's object as-is: only the bare bubble is excluded, and it never enters the
   * kernel. NO subtraction is performed anywhere (spec section 1; plan section 5 item 1).
   *
   * MEMORY: the ladder lives in the secondary aux basis, (n_nu_half, nq, N_m, N_m)
   * replicated. It is upfolded to the primary basis and transformed to tau ON THE LOCAL
   * (P, Q) BLOCK ONLY -- the replicated (nu | t, q, Np, Np) object is O(10 TB) at
   * production scale and is never formed. Per-rank cost is the class of the local dPi
   * block itself.
   *
   * The three logged meters are the acceptance instruments, not decoration:
   *   - ||P^lad||_max / ||P^RPA||_max (+ the per-q breakdown, the add_vertex_Pi_C
   *     precedent): a correction comparable to what it corrects means the tier is out of
   *     its regime, and eps = I - Z.P is then at risk of losing positivity;
   *   - lambda_max = rho(Xh Kt) (I3, PDF section 8.2): the resolvent's margin to the
   *     particle-hole instability at 1;
   *   - r_rt: the nu -> tau -> nu round trip of P^lad through the IAFT PH-sym pair. The
   *     injection ASSUMES P^lad is PH-symmetric and representable on this grid; r_rt is
   *     that assumption's meter, measured every update_w (lesson QM2-b: never silently
   *     trust a transform at a new evaluation point). It is a NU-space round trip, not a
   *     tau-space fit error (which must never be gated on).
   */
  template<nda::MemoryArrayOfRank<4> Array_t, typename communicator_t>
  void scr_coulomb_t::inject_pol_ladder(MBState &mb_state, THC_ERI auto &thc,
                                        memory::darray_t<Array_t, communicator_t> &dPi_tqPQ) {
    decltype(nda::range::all) all;
    utils::check(_pol_vtx != nullptr,
                 "inject_pol_ladder: the ladder instance is absent (ensure_pol_vertex).");
    utils::check(_vertex != nullptr and not _vertex->active(),
                 "inject_pol_ladder: an ACTIVE vertex_type injects its own Pi^C -- the "
                 "combination double counts (ruling R-Q3-3).");
    auto [nt_h, nq_g, Np, NQ_g] = dPi_tqPQ.global_shape();

    nda::array<double, 1> lam;
    auto Pl = _pol_vtx->eval_pol_ladder_whalf(mb_state, thc, &lam);  // (nw_h, nq, Nm, Nm)
    auto const &tmap = _pol_vtx->secondary_transfer();               // (nq, Nm, Np)
    const long nw_h = Pl.shape(0), Nm = Pl.shape(2);
    utils::check(Pl.shape(1) == nq_g and tmap.shape(0) == nq_g and tmap.shape(1) == Nm
                 and tmap.shape(2) == Np,
                 "inject_pol_ladder: shape mismatch (ladder {}x{}x{}, t {}x{}x{}, Pi q = "
                 "{}, Np = {}).", Pl.shape(0), Pl.shape(1), Nm, tmap.shape(0),
                 tmap.shape(1), tmap.shape(2), nq_g, Np);

    auto t_rng = dPi_tqPQ.local_range(0);
    auto q_rng = dPi_tqPQ.local_range(1);
    auto P_rng = dPi_tqPQ.local_range(2);
    auto Q_rng = dPi_tqPQ.local_range(3);
    const long ntl = long(t_rng.size()), nql = long(q_rng.size());
    const long nPl = long(P_rng.size()), nQl = long(Q_rng.size());
    auto Pi_loc = dPi_tqPQ.local();

    // ||P^RPA||_max BEFORE the +=, on the same grid (the comparison the ratio reports)
    double nR = 0.0;
    for (auto const &v : Pi_loc) nR = std::max(nR, std::abs(v));

    nda::array<ComplexType, 3> A(nw_h, nPl, nQl), B(nt_h, nPl, nQl);
    nda::array<ComplexType, 2> tq_Q(Nm, nQl), td_P(nPl, Nm), tmp(Nm, nQl);
    double nC = 0.0;
    std::vector<double> qmax(size_t(nq_g), 0.0);
    // a rank with an empty local block still joins the reductions below
    const long nq_own = (nPl > 0 and nQl > 0 and ntl > 0) ? nql : 0;
    for (long iql = 0; iql < nq_own; ++iql) {
      const long iq = q_rng.first() + iql;
      auto tq = tmap(iq, all, all);
      for (long m = 0; m < Nm; ++m) {
        for (long j = 0; j < nQl; ++j) tq_Q(m, j) = tq(m, Q_rng.first() + j);
        for (long i = 0; i < nPl; ++i) td_P(i, m) = std::conj(tq(m, P_rng.first() + i));
      }
      // upfold ONLY the local block: dP(P, Q) = sum_MN conj(t_MP) Pl_MN t_NQ (the
      // adjoint/no-leak map -- two thin gemms, never the full Np x Np)
      for (long j = 0; j < nw_h; ++j) {
        nda::blas::gemm(Pl(j, iq, all, all), tq_Q, tmp);
        nda::blas::gemm(td_P, tmp, A(j, all, all));
      }
      // nu -> tau on the flattened local block through the PUBLIC PH-sym transform
      auto A2 = nda::reshape(A, shape_t<2>{nw_h, nPl * nQl});
      auto B2 = nda::reshape(B, shape_t<2>{nt_h, nPl * nQl});
      _ft->w_to_tau_PHsym(A2, B2);
      for (long it = 0; it < ntl; ++it)
        for (long i = 0; i < nPl; ++i)
          for (long j = 0; j < nQl; ++j) {
            const ComplexType v = B(t_rng.first() + it, i, j);
            const double a = std::abs(v);
            nC = std::max(nC, a);
            qmax[size_t(iq)] = std::max(qmax[size_t(iq)], a);
            Pi_loc(it, iql, i, j) += v;
          }
    }
    auto &comm = *dPi_tqPQ.communicator();
    nC = comm.all_reduce_value(nC, boost::mpi3::max<>{});
    nR = comm.all_reduce_value(nR, boost::mpi3::max<>{});
    comm.all_reduce_in_place_n(qmax.data(), qmax.size(), boost::mpi3::max<>{});
    _pol_lad_ratio = nC / std::max(nR, 1e-300);

    // r_rt on a sampled q (Gamma + the last IBZ transfer): replicated N_m-class algebra
    {
      nda::array<ComplexType, 3> R(nw_h, Nm, Nm), S(nt_h, Nm, Nm);
      double num = 0.0, den = 0.0;
      for (long iq : {0l, nq_g - 1}) {
        R = Pl(all, iq, all, all);
        auto R2 = nda::reshape(R, shape_t<2>{nw_h, Nm * Nm});
        auto S2 = nda::reshape(S, shape_t<2>{nt_h, Nm * Nm});
        _ft->w_to_tau_PHsym(R2, S2);
        _ft->tau_to_w_PHsym(S2, R2);
        for (long j = 0; j < nw_h; ++j)
          for (long M = 0; M < Nm; ++M)
            for (long N = 0; N < Nm; ++N) {
              num += std::norm(R(j, M, N) - Pl(j, iq, M, N));
              den += std::norm(Pl(j, iq, M, N));
            }
        if (nq_g == 1) break;
      }
      _pol_r_rt = std::sqrt(num / std::max(den, 1e-300));
    }

    _pol_lam_nu0 = lam(0);
    _pol_lam_max = 0.0;
    for (long j = 0; j < nw_h; ++j) _pol_lam_max = std::max(_pol_lam_max, lam(j));

    app_log(1, "  [qpGW Q3] ladder injected into P ({} PH-sym nu nodes, N_m = {}): "
               "||P^lad||_max = {:.4e} vs ||P^RPA||_max = {:.4e} (ratio {:.3e}); "
               "transform round trip r_rt = {:.3e}",
            nw_h, Nm, nC, nR, _pol_lad_ratio, _pol_r_rt);
    {
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(2);
      for (long q = 0; q < nq_g; ++q) oss << (q ? " " : "") << qmax[size_t(q)];
      app_log(1, "  [qpGW Q3] ||P^lad||_max by transfer q (q=0 is Gamma): {}", oss.str());
    }
    app_log(1, "  [qpGW Q3] ladder resolvent margin: lambda_max(inu = 0) = {:.6f}, "
               "max over nu = {:.6f}{}", _pol_lam_nu0, _pol_lam_max,
            (_pol_lam_max > 0.9) ? "   [WARNING: approaching the particle-hole "
                                   "instability -- the resolvent is losing margin]" : "");
    utils::check(std::isfinite(_pol_lam_max) and _pol_lam_max < 1.0,
                 "inject_pol_ladder: the ladder kernel's spectral radius rho(Xh Kt) = {} "
                 "has reached 1 -- eq 6's resolvent (1 - chi0 Xi)^-1 is singular "
                 "(particle-hole instability). The BSE tier is outside its regime here; "
                 "reduce the ladder C window.", _pol_lam_max);
    if (nC > nR)
      app_log(1, "  [WARNING] the ladder polarization EXCEEDS the RPA polarization it "
                 "corrects.\n"
                 "            Expect eps = I - Z.P to lose conditioning (see the "
                 "dielectric conditioning below).");

    // Q4 C3: with a bosonic projector attached, the SAME ladder is also downfolded to the
    // impurity's local product basis -- eq 7's P_dc gains P^lad_loc (R-Q4-2). Nothing here
    // feeds back into the lattice P above; the local object is a DC ingredient only.
    if (mb_state.proj_boson.has_value()) {
      accumulate_pi_lad_loc(mb_state, thc, Pl, tmap);
      // Q4-C3b: ... and the DC-READY object next to it (the diagnostic above stays as-is).
      // COST NOTE: this re-runs the pair-space kernel with the E legs -- the K blocks are
      // rebuilt, so the ladder cost of an injection with a bosonic projector roughly
      // doubles. The kernel already accepts both RHS blocks in ONE pass (pair_space_ladder
      // takes Pi_ladder and Pi_lad_loc together); fusing the two calls is a pure
      // performance follow-up and is deliberately NOT done here, so the injection path
      // stays bit-identical to Q3 (gate G1).
      accumulate_pi_lad_loc_orb(mb_state, thc);
    }
  }

  /**
   * scGW-tilde L2, the ladder eps_M readout (stance i -- report-only, PDF section 4.2
   * placement (i)): per q at inu = 0,
   *   dP_ladder(q) = t(q)^dag Pi_ladder(q) t(q)          (upfold, adjoint-t/no-leak),
   *   dW[P](q)     = ([I - Z(q) P(q)]^{-1} - I) Z(q)     (single-frequency THC Dyson),
   *   eps^-1(q)-1  = (q^2 V / 4 pi) chi_bar(q) . dW(q) . chi_bar(q)*   (div_utils
   *                  eval_eps_inv_q convention),
   * evaluated for P = Pi0_RPA and P = Pi0_RPA + dP_ladder; eps_M(q) = 1/(1 + Re[.])
   * reported at the smallest nonzero |q| (gate L2-b measures the DIRECTION of the
   * ladder correction). Replicated Np x Np algebra: readout-scale only.
   */
  void scr_coulomb_t::pol_ladder_eps_readout(MBState &mb_state, THC_ERI auto &thc,
                                             nda::array<ComplexType, 3> const &Pi0_qPQ,
                                             nda::array<ComplexType, 2> const *eps_inv_head_q) {
    decltype(nda::range::all) all;
    auto MF = thc.MF();
    const long nq = Pi0_qPQ.shape(0), Np = Pi0_qPQ.shape(1);
    if (MF->nqpts_ibz() == 1) {
      app_log(1, "  [scGW-tilde L2] ladder readout skipped: nqpts_ibz == 1 (no finite "
                 "q for the eps_M head).");
      return;
    }

    // the ladder at inu = 0 in the readout vertex's secondary basis + upfold
    auto Pl_qmm = _pol_vtx->eval_pol_ladder_nu0(mb_state, thc);   // (nq, Nm, Nm)
    auto const &tmap = _pol_vtx->secondary_transfer();            // (nq, Nm, Np)
    const long Nm = Pl_qmm.shape(1);
    utils::check(tmap.shape(0) == nq and tmap.shape(1) == Nm and tmap.shape(2) == Np,
                 "pol_ladder_eps_readout: transfer map shape mismatch.");

    // replicated Z(q, P, Q) (same gather pattern as gather_nu0_row)
    nda::array<ComplexType, 3> Z_qPQ(nq, Np, Np);
    {
      const long np_ranks = thc.mpi()->comm.size();
      std::array<long, 3> zp = {1, 1, 1};
      zp[1] = utils::find_proc_grid_min_diff(np_ranks, Np, Np);
      zp[2] = np_ranks / zp[1];
      std::array<long, 3> zb = {1, 1, 1};
      zb[1] = std::min({static_cast<long>(1024), std::max(1l, Np / zp[1]),
                        std::max(1l, Np / zp[2])});
      zb[2] = zb[1];
      auto dZ = thc.dZ(zp, zb);
      auto q_rng = dZ.local_range(0);
      auto P_rng = dZ.local_range(1);
      auto Q_rng = dZ.local_range(2);
      auto Z_loc = dZ.local();
      Z_qPQ() = ComplexType(0.0);
      for (long iq = 0; iq < long(q_rng.size()); ++iq)
        for (long iP = 0; iP < long(P_rng.size()); ++iP)
          for (long iQ = 0; iQ < long(Q_rng.size()); ++iQ)
            Z_qPQ(q_rng.first() + iq, P_rng.first() + iP, Q_rng.first() + iQ) =
                Z_loc(iq, iP, iQ);
      thc.mpi()->comm.all_reduce_in_place_n(Z_qPQ.data(), Z_qPQ.size(), std::plus<>{});
    }

    // per q: upfold, two single-frequency Dysons, the chi_bar head contraction
    auto Chi_bar = thc.basis_bar_head();                          // (nq, Np)
    const double fpi = 4.0 * 3.14159265358979323846;
    nda::array<ComplexType, 2> dP(Np, Np), tmpM(Nm, Np), A(Np, Np);
    nda::matrix<ComplexType> Am(Np, Np);
    nda::array<ComplexType, 1> chi_c(Np), buf(Np);
    double eps_rpa_qmin = -1.0, eps_lad_qmin = -1.0, qmin_abs2 = 1e300;
    long iq_min = -1;
    for (long iq = 0; iq < nq; ++iq) {
      auto qpts = MF->Qpts_ibz(iq);
      const double q_abs2 = qpts(0) * qpts(0) + qpts(1) * qpts(1) + qpts(2) * qpts(2);
      if (q_abs2 < 1e-12) continue;                               // Gamma: no head here
      // upfold: dP = t^dag Pl t
      auto tq = tmap(iq, all, all);
      nda::blas::gemm(Pl_qmm(iq, all, all), tq, tmpM);            // Pl . t   (Nm x Np)
      nda::array<ComplexType, 2> td(Np, Nm);
      for (long m = 0; m < Nm; ++m)
        for (long P = 0; P < Np; ++P) td(P, m) = std::conj(tq(m, P));
      nda::blas::gemm(td, tmpM, dP);                              // t^dag Pl t
      const double factor = (q_abs2 / fpi) * MF->volume();
      chi_c = nda::conj(Chi_bar(iq, all));
      auto eps_of = [&](bool with_ladder) {
        // A = I - Z (P0 [+ dP]);  dW = (A^{-1} - I) Z;  head contraction
        A() = Pi0_qPQ(iq, all, all);
        if (with_ladder) A += dP;
        nda::array<ComplexType, 2> ZP(Np, Np);
        nda::blas::gemm(Z_qPQ(iq, all, all), A, ZP);
        Am() = ZP;
        Am() *= ComplexType(-1.0);
        for (long P = 0; P < Np; ++P) Am(P, P) += ComplexType(1.0);
        nda::inverse_in_place(Am);
        for (long P = 0; P < Np; ++P) Am(P, P) -= ComplexType(1.0);
        nda::blas::gemm(Am, Z_qPQ(iq, all, all), ZP);
        nda::blas::gemv(ZP, chi_c, buf);
        const ComplexType eih = factor * nda::blas::dot(Chi_bar(iq, all), buf);
        return 1.0 / (1.0 + eih.real());
      };
      const double e_rpa = eps_of(false);
      const double e_lad = eps_of(true);
      app_log(2, "  [scGW-tilde L2]   q {} (|q|^2 = {:.4e}): eps_M RPA = {:.6f}, "
                 "+ladder = {:.6f}", iq, q_abs2, e_rpa, e_lad);
      if (q_abs2 < qmin_abs2) {
        qmin_abs2 = q_abs2;
        eps_rpa_qmin = e_rpa;
        eps_lad_qmin = e_lad;
        iq_min = iq;
      }
    }
    if (iq_min >= 0) {
      app_log(1, "  [scGW-tilde L2] ladder eps_M readout (inu = 0, q_min = {}): "
                 "RPA = {:.6f}, +ladder = {:.6f} (Delta = {:+.6f}; gate L2-b watches "
                 "the DIRECTION)", iq_min, eps_rpa_qmin, eps_lad_qmin,
              eps_lad_qmin - eps_rpa_qmin);
      _pol_eps_rpa = eps_rpa_qmin;
      _pol_eps_ladder = eps_lad_qmin;
    }
    // Q3-b(i): the SAME q_min read off the loop's own screening. Same G, same kernel, two
    // evaluation routes -- the tau-space Dyson of the (injected) Pi against the readout's
    // single-frequency inu = 0 Dyson above. With the injection ON the two must agree to
    // the transform class (r_rt); with it OFF this is the RPA leg of the same identity.
    if (eps_inv_head_q != nullptr and iq_min >= 0) {
      const long nt_h = eps_inv_head_q->shape(0);
      utils::check(eps_inv_head_q->shape(1) == nq,
                   "pol_ladder_eps_readout: eps_inv_head_q has {} q rows, expected {}.",
                   eps_inv_head_q->shape(1), nq);
      long nw_half = (_ft->nw_b() % 2 == 0) ? _ft->nw_b() / 2 : _ft->nw_b() / 2 + 1;
      nda::array<ComplexType, 2> et(nt_h, 1), ew(nw_half, 1);
      for (long it = 0; it < nt_h; ++it) et(it, 0) = (*eps_inv_head_q)(it, iq_min);
      _ft->tau_to_w_PHsym(et, ew);            // inu = 0 = index 0 of the PH-sym half grid
      _pol_eps_loop = 1.0 / (1.0 + ew(0, 0).real());
      app_log(1, "  [qpGW Q3] loop-side eps_M(q_min = {}, inu = 0) from the tau Dyson = "
                 "{:.9f}; readout route (+ladder) = {:.9f} (deviation = {:.3e})",
              iq_min, _pol_eps_loop, eps_lad_qmin,
              std::abs(_pol_eps_loop - eps_lad_qmin));
    }
  }

  template<bool w_out, nda::MemoryArrayOfRank<4> local_Array_t, typename communicator_t>
  auto scr_coulomb_t::dyson_W_from_Pi_tau(
      memory::darray_t<local_Array_t, communicator_t> &dPi_tqPQ_pos,
      THC_ERI auto &thc, bool reset_input,
      std::array<long, 4> w_pgrid, std::array<long, 4> w_bsize)
  -> memory::darray_t<local_Array_t, mpi3::communicator>
  {
    if (w_pgrid[0]*w_pgrid[1]*w_pgrid[2]*w_pgrid[3] <= 0 or w_bsize[0]*w_bsize[1]*w_bsize[2]*w_bsize[3] <= 0) {
      std::tie(w_pgrid, w_bsize) = scr_coulomb_t::W_omega_proc_grid(
          thc.mpi()->comm.size(), thc.MF()->nqpts_ibz(), _ft->nw_b(), thc.Np());
    }

    auto t_pgrid = dPi_tqPQ_pos.grid();
    auto t_bsize = dPi_tqPQ_pos.block_size();
    auto dPi_wqPQ = tau_to_w(dPi_tqPQ_pos, w_pgrid, w_bsize, reset_input);
    dyson_W_in_place(dPi_wqPQ, thc);
    if constexpr (w_out) {
      return dPi_wqPQ;
    } else {
      return w_to_tau(dPi_wqPQ, t_pgrid, t_bsize, true);
    }
  }

  template<nda::MemoryArray Array_4D_t, typename communicator_t>
  void scr_coulomb_t::dyson_W_in_place(
      memory::darray_t<Array_4D_t, communicator_t> &dPi_wqPQ,
      THC_ERI auto &thc) {

    _Timer.start("EVALUATE_W");
    auto [nw, nqpts, NP, NQ] = dPi_wqPQ.global_shape();
    auto [nw_loc, nq_loc, NP_loc, NQ_loc] = dPi_wqPQ.local_shape();
    auto [w_origin, q_origin, P_origin, Q_origin] = dPi_wqPQ.origin();
    long nq_loc_max = nq_loc;
    dPi_wqPQ.communicator()->broadcast_n(&nq_loc_max, 1, 0);

    auto P_rng = dPi_wqPQ.local_range(2);
    auto Q_rng = dPi_wqPQ.local_range(3);
    auto pgrid = dPi_wqPQ.grid();
    auto block_size = dPi_wqPQ.block_size();
    long qpool_id = (nq_loc==nq_loc_max)? q_origin/nq_loc : (q_origin-nqpts%pgrid[1])/nq_loc;

    app_log(2, "  Evaluation of the screened interaction:");
    app_log(2, "    - processor grid for Pi/W: (w, q, P, Q) = ({}, {}, {}, {})", pgrid[0], pgrid[1], pgrid[2], pgrid[3]);
    app_log(2, "    - block size: (w, q, P, Q) = ({}, {}, {}, {})\n", block_size[0], block_size[1], block_size[2], block_size[3]);

    // Setup wq_intra_comm
    mpi3::communicator wq_intra_comm = thc.mpi()->comm.split(w_origin*nqpts + q_origin, thc.mpi()->comm.rank());
    utils::check(wq_intra_comm.size() == pgrid[2]*pgrid[3], "wq_intra_comm.size() != pgrid[2]*pgrid[3]");
    // Setup q_intra_comm
    mpi3::communicator q_intra_comm = thc.mpi()->comm.split(q_origin, thc.mpi()->comm.rank());
    utils::check(q_intra_comm.size() == pgrid[0]*pgrid[2]*pgrid[3], "q_intra_comm.size() != pgrid[0]*pgrid[2]*pgrid[3]");

    using Array_2D_t = memory::array<HOST_MEMORY, ComplexType, 2>;
    using math::nda::make_distributed_array;
    auto dPi_PQ = make_distributed_array<Array_2D_t>(wq_intra_comm, {pgrid[2], pgrid[3]}, {NP, NQ}, {block_size[2], block_size[3]}, true);
    auto dZ_PQ  = make_distributed_array<Array_2D_t>(wq_intra_comm, {pgrid[2], pgrid[3]}, {NP, NQ}, {block_size[2], block_size[3]}, true);
    auto dA_PQ  = make_distributed_array<Array_2D_t>(wq_intra_comm, {pgrid[2], pgrid[3]}, {NP, NQ}, {block_size[2], block_size[3]}, true);
    utils::check(dPi_PQ.local_range(0) == P_rng, "Error: local range mismatches!" );
    utils::check(dPi_PQ.local_range(1) == Q_rng, "Error: local range mismatches!");
    utils::check(dPi_PQ.local_shape()[0] == NP_loc and dPi_PQ.local_shape()[1] == NQ_loc, "Error: local shape mismatched!");

    std::vector<std::pair<long,long> > diag_idx;
    for (long iP = 0; iP < NP_loc; ++iP) {
      long P = iP + P_origin;
      for (long iQ = 0; iQ < NQ_loc; ++iQ) {
        long Q = iQ + Q_origin;
        if (P == Q) diag_idx.push_back({iP, iQ});
      }
    }

    auto Pi_wqPQ = dPi_wqPQ.local();
    auto Pi_PQ = dPi_PQ.local();
    auto Z_PQ = dZ_PQ.local();
    auto A_PQ = dA_PQ.local();
    // ---- DIELECTRIC POSITIVITY MONITOR (ISDF-Vertex) --------------------------------
    // On the imaginary axis the RPA polarization is negative semi-definite, so
    // eps = I - Z.Pi is positive definite and ||eps^{-1}|| = O(1). A vertex correction
    // P^C = -2 dPhi_2^C/dW carries NO such sign guarantee: once Pi is large enough that
    // eps loses positivity at some (q, i.nu), the inverse below silently returns garbage
    // and W acquires a spurious pole. That is exactly how the scGW+vertex runs fail --
    // several smoothly converging iterations (max |d.Sigma| shrinking 0.19 -> 0.05) and
    // then a single step with max |d.Sigma| ~ 6.5e4. Track ||eps^{-1}||_max over all
    // (q, i.nu) so the failure is DIAGNOSED instead of silently propagating.
    double epsinv_max = 0.0;
    long epsinv_q = -1, epsinv_w = -1;
    for (size_t iq_loc = 0; iq_loc < nq_loc; ++iq_loc) {
      long iq = q_origin + iq_loc;
      Z_PQ = thc.Z(iq, P_rng, Q_rng, qpool_id, pgrid[1], q_intra_comm);

      // W(w) = [ I - Z * Pi(w)]^{-1} * Z - Z
      for (size_t n = 0; n < nw_loc; ++n) {
        Pi_PQ = Pi_wqPQ(n, iq_loc, nda::ellipsis{});

        // A = Z * Pi(w)
        math::nda::slate_ops::multiply(dZ_PQ, dPi_PQ, dA_PQ);
        // A = I - Z * Pi(w)
        for (auto idx: diag_idx) {
          A_PQ(idx.first, idx.second) -= ComplexType(1.0);
        }
        A_PQ *= -1.0;

        // A = [I - Z*Pi(w)]^{-1}
        math::nda::slate_ops::inverse(dA_PQ);
        for (auto const &v : A_PQ)
          if (std::abs(v) > epsinv_max) {
            epsinv_max = std::abs(v);
            epsinv_q = iq;                      // WHERE it is worst: transfer q ...
            epsinv_w = long(n) + long(w_origin);// ... and bosonic Matsubara index
          }

        // A = [I - Z*Pi(w)]^{-1} - I
        for (auto idx: diag_idx) {
          A_PQ(idx.first, idx.second) -= ComplexType(1.0);
        }

        // W = ([I - Z*Pi(w)]^{-1} - I) * Z
        math::nda::slate_ops::multiply(dA_PQ, dZ_PQ, dPi_PQ);
        Pi_wqPQ(n, iq_loc, nda::ellipsis{}) = Pi_PQ;
      }
    }
    // prevent dead block in thc.Z() in case nq_loc is not the same for all processors
    for (long iq_loc = nq_loc; iq_loc < nq_loc_max; ++iq_loc)
      Z_PQ = thc.Z(0, P_rng, Q_rng, qpool_id, pgrid[1], q_intra_comm);

    {
      // reduce the VALUE and carry its (q, i.nu) location along, so the worst cell is
      // identifiable: the Gamma head, a specific transfer, or a high-frequency tail all
      // mean different things about where a vertex-corrected Pi goes wrong.
      double gmax = dPi_wqPQ.communicator()->all_reduce_value(epsinv_max,
                                                              boost::mpi3::max<>{});
      long q_of_max = (epsinv_max == gmax) ? epsinv_q : -1;
      long w_of_max = (epsinv_max == gmax) ? epsinv_w : -1;
      q_of_max = dPi_wqPQ.communicator()->all_reduce_value(q_of_max, boost::mpi3::max<>{});
      w_of_max = dPi_wqPQ.communicator()->all_reduce_value(w_of_max, boost::mpi3::max<>{});
      epsinv_max = gmax;
      app_log(1, "    - dielectric conditioning: max_(q, i.nu) || [I - Z.Pi]^-1 ||_max = "
                 "{:.4e}   (worst cell: q = {}, i.nu index = {} of {})",
              epsinv_max, q_of_max, w_of_max, nw);
    }
    if (epsinv_max > 1e3)
      app_log(1, "    [WARNING] I - Z.Pi is close to SINGULAR. On the imaginary axis the RPA\n"
                 "              polarization is negative semi-definite, so this cannot happen "
                 "for plain\n"
                 "              RPA/scGW; a vertex-corrected Pi carries no such sign guarantee. "
                 "The\n"
                 "              inverse above is then numerically meaningless and W acquires a "
                 "spurious\n"
                 "              pole -- the self-energy will blow up on the NEXT iteration. "
                 "Reduce the\n"
                 "              correlated window C, or ramp the vertex in more slowly.");

    _Timer.stop("EVALUATE_W");

  }

  // CNY: This will be deprecated soon.
  auto scr_coulomb_t::eval_Pi_qdep(const nda::MemoryArrayOfRank<5> auto &G_tskij, THC_ERI auto &thc,
                                   const projector_boson_t* proj,
                                   const nda::array_view<ComplexType, 5> *pi_imp,
                                   const nda::array_view<ComplexType, 5> *pi_dc)
  -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>
  {

    if (_screen_type.find("edmft") == std::string::npos and (pi_imp!= nullptr or pi_dc != nullptr)) {
      app_log(2, "scr_coulomb_t::eval_Pi_qdep: pi_imp and pi_dc are only used in edmft mode. "
                 "Ignoring them in {} mode.", _screen_type);
    }

    if (_screen_type == "rpa_k")
      return eval_Pi_rpa_kspace(G_tskij, thc);

    if (_screen_type.find("gw_edmft_rpa")!=std::string::npos)
      return eval_Pi_rpa_Rspace(G_tskij, thc);

    // RPA polarizability
    auto dPi_tqPQ = eval_Pi_rpa_Rspace(G_tskij, thc);

    // cRPA corrections: Pi_cRPA = Pi_RPA - Pi_active
    if (_screen_type.find("crpa") != std::string::npos) {

      utils::check(proj != nullptr, "scr_coulomb_t::eval_Pi_qdep: projector is missing in the crpa mode.");
      int crpa_scheme = (_screen_type.find("crpa_vasp")!=std::string::npos)? 2 :
                        (_screen_type.find("crpa_ks")!=std::string::npos)? 1 : 0;
      // Pi_dc and Pi are distributed in the same way among the processors since "eval_Pi_rpa_active" call "eval_Pi_qdep" under the hood.
      auto dPi_tqPQ_dc = eval_Pi_rpa_active(G_tskij, thc, proj->proj_fermi(), crpa_scheme);
      dPi_tqPQ.local() -= dPi_tqPQ_dc.local();

    }

    // EDMFT corrections: Pi_edmft = Pi_RPA + (Pi_imp - Pi_dc)
    if (_screen_type.find("edmft") != std::string::npos) {

      utils::check(proj != nullptr, "scr_coulomb_t::eval_Pi_qdep: projector is missing in edmft mode.");
      utils::check(pi_imp != nullptr and pi_dc != nullptr,
                   "scr_coulomb_t::eval_Pi_qdep: "
                   "pi_imp or pi_dc must be provided in edmft mode.");

      auto sPi_correction = math::shm::make_shared_array<Array_view_5D_t>(*thc.mpi(), pi_imp->shape());
      if (thc.mpi()->node_comm.root()) {
        sPi_correction.local() = *pi_imp - *pi_dc;
      }
      thc.mpi()->comm.barrier();
      auto dPi_tqPQ_correction = upfold_pi_local(sPi_correction.local(), thc, *proj, dPi_tqPQ.grid(), dPi_tqPQ.block_size());
      dPi_tqPQ.local() += dPi_tqPQ_correction.local();
      thc.mpi()->comm.barrier();
    }

    return dPi_tqPQ;
  }

  auto scr_coulomb_t::eval_Pi_qdep(MBState &mb_state, THC_ERI auto &thc)
  -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>
  {

    if (_screen_type.find("edmft") == std::string::npos
        and (mb_state.sPi_imp_wabcd or mb_state.sPi_dc_wabcd)) {
      app_log(1, "");
      app_log(1, "╔══════════════════════════════════════════════════════════╗");
      app_log(1, "║ [ NOTE ]                                                 ║");
      app_log(1, "║ Screening type is set to \"non-edmft\" type, but local     ║");
      app_log(1, "║ polarization corrections were found or provided.         ║");
      app_log(1, "║ CoQui will ignore the corrections.                       ║");
      app_log(1, "╚══════════════════════════════════════════════════════════╝\n");
    }
    utils::check(mb_state.sG_tskij.has_value(),
                 "scr_coulomb_t::eval_Pi_qdep: G_tskij is not set in MBState.");

    auto G_tskij = mb_state.sG_tskij.value().local();

    // ISDF-Vertex INCREMENT S2 (notes/static_vertex_implementation_plan.md section 2.2,
    // decision D2): the STATIC rung W0[G] = [1 - v P^0_RPA[G]]^{-1} v at i.nu = 0 is a
    // functional of the RPA polarizability ONLY, so it must be built at exactly this
    // point -- right after Pi_RPA(q, tau) is assembled and BEFORE any vertex/cRPA/EDMFT
    // correction is added (ordering, plan section 2.3). Called immediately after every
    // eval_Pi_rpa_* below and NOWHERE else, so no other Pi contribution can leak into it.
    // No-op unless the attached vertex is active AND its rung mode is static/linear:
    // the dynamic theory (Formulation B) has no W0, and this path then executes zero new
    // arithmetic and allocates nothing.
    auto build_vertex_W0 = [&](auto &dPi_rpa) {
      if (_vertex != nullptr and _vertex->needs_w0())
        _vertex->build_w0(mb_state, thc, dPi_rpa);
    };

    // ISDF-Vertex: additive second-order-exchange polarization cut Pi^C on the
    // same distributed grid as the RPA polarizability (EDMFT "+=" precedent below).
    // When no active vertex is attached this is a strict no-op -- no allocation,
    // no arithmetic -- so the disabled path is bit-identical to plain RPA/scGW.
    auto add_vertex_Pi_C = [&](auto &dPi) {
      // B-S (vertex_rung = "static") has NO polarization injection at all: P = RPA by
      // construction (plan section 2.1), and the ONE vertex_t drives every cut of the
      // selected mode, so the forbidden hybrid "static Sigma^C with a Pi^C injection"
      // must be unrepresentable rather than merely discouraged. B-L keeps the seam (its
      // P^{C,L} is injected here, increment S7).
      if (_vertex != nullptr and _vertex->rung() == static_rung) return;
      if (_vertex != nullptr and _vertex->active()) {
        auto dPi_C_tqPQ = _vertex->eval_Pi_C(mb_state, thc, dPi.grid(),
                                             dPi.block_size(), dPi.global_shape());
        // SIZE OF THE CORRECTION. Pi^C is meant to be a correction to Pi_RPA; if it is
        // comparable to or larger than what it corrects, the second-order-exchange
        // truncation is outside its regime and eps = I - Z.Pi is at risk of losing
        // positivity (notes/vertex_divergence_diagnosis.md section 2). Report the ratio
        // so "large but controlled" is distinguishable from "runaway" at a glance.
        double nC = 0.0, nR = 0.0;
        for (auto const &v : dPi_C_tqPQ.local()) nC = std::max(nC, std::abs(v));
        for (auto const &v : dPi.local()) nR = std::max(nR, std::abs(v));
        nC = thc.mpi()->comm.all_reduce_value(nC, boost::mpi3::max<>{});
        nR = thc.mpi()->comm.all_reduce_value(nR, boost::mpi3::max<>{});
        app_log(1, "  [ISDF-Vertex] ||Pi^C||_max = {:.4e} vs ||Pi_RPA||_max = {:.4e} "
                   "(ratio {:.3e})", nC, nR, nC / std::max(nR, 1e-300));
        // PER-q BREAKDOWN. The analytic q->0 head is inserted ONLY at Gamma, yet the worst
        // dielectric cell in the diverging Si kp444 C=[0,4) run was a NON-Gamma transfer
        // (q = 4), never q = 0. That is not a contradiction: Pi^C's INTERNAL rung sum runs
        // over all transfers, so the Gamma rung -- the one carrying the head -- feeds EVERY
        // external q. Resolving ||Pi^C||_max by external transfer says whether a head-on run
        // deviates from head-off uniformly in q (the head entering through the internal sum,
        // as the construction intends) or concentrates on particular cells (which would point
        // at the Gamma insertion itself). Cheap: nq_ibz values, once per update_w.
        {
          auto Pi_C_loc = dPi_C_tqPQ.local();
          auto ls = Pi_C_loc.shape();
          long nq_g = dPi_C_tqPQ.global_shape()[1];
          std::vector<double> qmax(size_t(nq_g), 0.0);
          long iq_loc = 0;
          for (auto gq : dPi_C_tqPQ.local_range(1)) {
            double m = 0.0;
            for (long it = 0; it < ls[0]; ++it)
              for (long P = 0; P < ls[2]; ++P)
                for (long Q = 0; Q < ls[3]; ++Q)
                  m = std::max(m, std::abs(Pi_C_loc(it, iq_loc, P, Q)));
            qmax[size_t(gq)] = std::max(qmax[size_t(gq)], m);
            ++iq_loc;
          }
          thc.mpi()->comm.all_reduce_in_place_n(qmax.data(), qmax.size(),
                                                boost::mpi3::max<>{});
          std::ostringstream oss;
          oss << std::scientific << std::setprecision(2);
          for (long q = 0; q < nq_g; ++q) oss << (q ? " " : "") << qmax[size_t(q)];
          app_log(1, "  [ISDF-Vertex] ||Pi^C||_max by transfer q (q=0 is Gamma): {}",
                  oss.str());
        }
        if (nC > nR)
          app_log(1, "  [WARNING] the vertex polarization EXCEEDS the RPA polarization it "
                     "corrects.\n"
                     "            The second-order-exchange truncation is outside its "
                     "regime here; expect\n"
                     "            eps = I - Z.Pi to lose positivity (see the dielectric "
                     "conditioning below).");
        dPi.local() += dPi_C_tqPQ.local();
        thc.mpi()->comm.barrier();
      }
    };

    // qpGW Q4 (notes/q4_edmft_skeleton_spec.md, ruling R-Q4-3): the Q3 BSE tier lives HERE,
    // not in update_w. Two structural reasons: (i) in edmft mode update_w would hand the
    // kernel builder the IMPURITY-CORRECTED Pi, violating R-Q3-1 ("the kernel sees the pure
    // RPA Pi"); (ii) the bosonic closure reaches eval_Pi_qdep + its own Dyson directly and
    // never through update_w, so W_loc must get its ladder from this seam. The ORDER is the
    // contract: RPA Pi -> build_vertex_W0 -> build_pol_ladder_kernel (the PURE-RPA point)
    // -> crpa/edmft corrections -> add_vertex_Pi_C -> inject_pol_ladder (last), and every
    // return path below runs the same two hooks -- including the "rpa"/gw_edmft_rpa early
    // return, which is Q3's production mode and must keep injecting.
    // ARITHMETIC IDENTITY with the pre-Q4 update_w placement (gate Q4-s1): the readout
    // requires an INACTIVE _vertex, and add_vertex_Pi_C is a strict no-op for an inactive
    // vertex, so the kernel build moving across it changes no executed operation.
    const bool pol_readout = (_vertex != nullptr and _vertex->pol_vertex_active()
                              and not _vertex->active());
    auto build_pol_ladder_kernel = [&](auto &dPi_rpa) {
      if (not pol_readout) return;
      ensure_pol_vertex(thc);
      _pol_vtx->build_w0(mb_state, thc, dPi_rpa);   // build_w0 only READS dPi
      _pol_pi0_qPQ = gather_nu0_row(dPi_rpa);       // the readout's RPA baseline
    };
    auto inject_pol_tier = [&](auto &dPi) {
      if (pol_readout and _vertex->pol_vertex_inject_enabled())
        inject_pol_ladder(mb_state, thc, dPi);
    };

    if (_screen_type == "rpa_k") {
      auto dPi_tqPQ = eval_Pi_rpa_kspace(G_tskij, thc);
      build_vertex_W0(dPi_tqPQ);            // RPA-only Pi: before ANY correction
      build_pol_ladder_kernel(dPi_tqPQ);
      add_vertex_Pi_C(dPi_tqPQ);
      inject_pol_tier(dPi_tqPQ);
      return dPi_tqPQ;
    }

    // RPA polarizability
    auto dPi_tqPQ = eval_Pi_rpa_Rspace(G_tskij, thc);
    build_vertex_W0(dPi_tqPQ);              // RPA-only Pi: before ANY correction
    build_pol_ladder_kernel(dPi_tqPQ);
    if (_screen_type.find("gw_edmft_rpa")!=std::string::npos or _screen_type=="rpa") {
      add_vertex_Pi_C(dPi_tqPQ);
      inject_pol_tier(dPi_tqPQ);
      return dPi_tqPQ;
    }

    // cRPA corrections: Pi_cRPA = Pi_RPA - Pi_active
    if (_screen_type.find("crpa") != std::string::npos) {

      utils::check(mb_state.proj_boson.has_value(),
                   "scr_coulomb_t::eval_Pi_qdep: projector is missing in the crpa mode.");
      int crpa_scheme = (_screen_type.find("crpa_vasp")!=std::string::npos)? 2 :
                        (_screen_type.find("crpa_ks")!=std::string::npos)? 1 : 0;
      // Pi_dc and Pi are distributed in the same way among the processors since "eval_Pi_rpa_active" call "eval_Pi_qdep" under the hood.
      auto& proj_boson = mb_state.proj_boson.value();
      auto dPi_tqPQ_dc = eval_Pi_rpa_active(G_tskij, thc, proj_boson.proj_fermi(), crpa_scheme);
      dPi_tqPQ.local() -= dPi_tqPQ_dc.local();

    }

    // EDMFT corrections: Pi_edmft = Pi_RPA + (Pi_imp - Pi_dc)
    if (_screen_type.find("edmft") != std::string::npos) {

      utils::check(mb_state.proj_boson.has_value(), "scr_coulomb_t::eval_Pi_qdep: projector is missing in edmft mode.");

      if (!mb_state.sPi_imp_wabcd or !mb_state.sPi_dc_wabcd) {
        app_log(1, "");
        app_log(1, "╔══════════════════════════════════════════════════════╗");
        app_log(1, "║ [ NOTE ]                                             ║");
        app_log(1, "║ Screening type is set to \"edmft\", but local        ║");
        app_log(1, "║ polarization corrections were not found or provided. ║");
        app_log(1, "║ CoQui will proceed assuming zero correction.         ║");
        app_log(1, "╚══════════════════════════════════════════════════════╝\n");

      } else {
        auto &proj_boson = mb_state.proj_boson.value();
        auto nImpOrbs = proj_boson.nImpOrbs();
        auto Pi_imp_iw = mb_state.sPi_imp_wabcd.value().local();
        auto Pi_dc_iw = mb_state.sPi_dc_wabcd.value().local();
        auto sPi_t_correction = math::shm::make_shared_array<Array_view_5D_t>(
            *thc.mpi(), {dPi_tqPQ.global_shape()[0], nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs});
        if (thc.mpi()->node_comm.root()) {
          _ft->w_to_tau_PHsym(Pi_imp_iw, sPi_t_correction.local());

          nda::array<ComplexType, 5> pi_t_buffer(sPi_t_correction.shape());
          _ft->w_to_tau_PHsym(Pi_dc_iw, pi_t_buffer);
          sPi_t_correction.local() -= pi_t_buffer;
        }
        thc.mpi()->comm.barrier();

        auto dPi_tqPQ_correction = upfold_pi_local(sPi_t_correction.local(), thc, proj_boson,
                                                   dPi_tqPQ.grid(), dPi_tqPQ.block_size());
        dPi_tqPQ.local() += dPi_tqPQ_correction.local();
        thc.mpi()->comm.barrier();
      }
    }

    // ISDF-Vertex: Pi = Pi_RPA (+ corrections) + Pi^C
    add_vertex_Pi_C(dPi_tqPQ);
    inject_pol_tier(dPi_tqPQ);

    return dPi_tqPQ;
  }

  template<nda::MemoryArrayOfRank<4> local_Array_t, typename communicator_t>
  auto scr_coulomb_t::tau_to_w(
      memory::darray_t<local_Array_t, communicator_t> &dPi_tqPQ_pos,
      std::array<long, 4> w_pgrid_out, std::array<long, 4> w_bsize_out,
      bool reset_input)
  -> memory::darray_t<local_Array_t, mpi3::communicator>
  {
    using math::nda::make_distributed_array;

    _Timer.start("IMAG_FT_TtoW");
    auto comm = dPi_tqPQ_pos.communicator();
    long npts = dPi_tqPQ_pos.global_shape()[1];
    long Np = dPi_tqPQ_pos.global_shape()[3];
    long nw_half = (_ft->nw_b()%2==0)? _ft->nw_b()/2 : _ft->nw_b()/2 + 1;
    std::array<long, 4> w_gshape = {nw_half, npts, Np, Np};
    std::array<long, 4> t_gshape = dPi_tqPQ_pos.global_shape();

    if (dPi_tqPQ_pos.communicator()->size() == 1) {
      _ft->check_leakage(dPi_tqPQ_pos, imag_axes_ft::boson, "polarizability", true);
      auto dPi_wqPQ = make_distributed_array<local_Array_t>(
          *comm, {1, 1, 1, 1}, w_gshape, dPi_tqPQ_pos.block_size());
      // local arrays cover all tau and w points
      auto Pi_ti_loc = dPi_tqPQ_pos.local();
      auto Pi_wi_loc = dPi_wqPQ.local();
      _ft->tau_to_w_PHsym(Pi_ti_loc, Pi_wi_loc);
      if (reset_input) dPi_tqPQ_pos.reset();
      _Timer.stop("IMAG_FT_TtoW");
      return dPi_wqPQ;
    }
    // redistribute to cover (tau, w)-axes locally -> FT locally -> redistribute back
    std::array<long, 4> b_pgrid = {1, 1, 1, 1}; // pgrid for buffer
    {
      int np = comm->size();
      if (t_gshape[2] * t_gshape[3] >= np) {
        b_pgrid[2] = utils::find_proc_grid_min_diff(np, t_gshape[2], t_gshape[3]);
        b_pgrid[3] = np / b_pgrid[2];
      } else {
        APP_ABORT("scr_coulomb_t::tau_to_w: Error finding proper pgrid: gshape[2]*gshape[3] < np.");
      }
    }
    auto buffer_ti  = make_distributed_array<local_Array_t>(
        *comm, b_pgrid, t_gshape, dPi_tqPQ_pos.block_size());
    _Timer.start("FT_REDISTRIBUTE");
    math::nda::redistribute(dPi_tqPQ_pos, buffer_ti);
    _Timer.stop("FT_REDISTRIBUTE");
    if (reset_input) dPi_tqPQ_pos.reset();
    _ft->check_leakage(buffer_ti, imag_axes_ft::boson, "polarizability", true);
    buffer_ti.communicator()->barrier();

    auto buffer_wi  = make_distributed_array<local_Array_t>(
        *comm, b_pgrid, w_gshape, buffer_ti.block_size());
    {
      auto buf_ti_loc = buffer_ti.local();
      auto buf_wi_loc = buffer_wi.local();
      _ft->tau_to_w_PHsym(buf_ti_loc, buf_wi_loc);
    }
    buffer_ti.reset();
    buffer_wi.communicator()->barrier();

    auto dPi_wqPQ = make_distributed_array<local_Array_t>(
        *comm, w_pgrid_out, w_gshape, w_bsize_out);

    _Timer.start("FT_REDISTRIBUTE");
    math::nda::redistribute(buffer_wi, dPi_wqPQ);
    _Timer.stop("FT_REDISTRIBUTE");
    buffer_wi.reset();
    dPi_wqPQ.communicator()->barrier();

    _Timer.stop("IMAG_FT_TtoW");
    return dPi_wqPQ;
  }

  template<nda::MemoryArrayOfRank<4> local_Array_t, typename communicator_t>
  auto scr_coulomb_t::w_to_tau(
      memory::darray_t<local_Array_t, communicator_t> &dW_wqPQ_pos,
      std::array<long, 4> t_pgrid_out, std::array<long, 4> t_bsize_out,
      bool reset_input)
  -> memory::darray_t<local_Array_t, mpi3::communicator>
  {
    using math::nda::make_distributed_array;

    _Timer.start("IMAG_FT_WtoT");
    auto comm = dW_wqPQ_pos.communicator();
    long npts = dW_wqPQ_pos.global_shape()[1];
    long Np = dW_wqPQ_pos.global_shape()[3];
    auto w_gshape = dW_wqPQ_pos.global_shape();
    size_t nt_half = (_ft->nt_b()%2==0)? _ft->nt_b() / 2 : _ft->nt_b() / 2 + 1;
    std::array<long, 4> t_gshape = {nt_half, npts, Np, Np};

    if (dW_wqPQ_pos.communicator()->size() == 1) {
      auto dW_tqPQ = make_distributed_array<local_Array_t>(
          *comm, {1, 1, 1, 1}, t_gshape, {1, 1, 1, 1});
      // local arrays cover all tau and w points
      auto W_wi_loc = dW_wqPQ_pos.local();
      auto W_ti_loc = dW_tqPQ.local();
      _ft->w_to_tau_PHsym(W_wi_loc, W_ti_loc);
      if (reset_input) dW_wqPQ_pos.reset();
      _ft->check_leakage(dW_tqPQ, imag_axes_ft::boson, "screened interation", true);
      _Timer.stop("IMAG_FT_WtoT");
      return dW_tqPQ;
    }

    // redistribute to cover (tau, w)-axes locally -> FT locally -> redistribute back
    std::array<long, 4> b_pgrid = {1, 1, 1, 1}; // pgrid for buffer
    {
      int np = comm->size();
      if (t_gshape[2] * t_gshape[3] >= np) {
        b_pgrid[2] = utils::find_proc_grid_min_diff(np, t_gshape[2], t_gshape[3]);
        b_pgrid[3] = np / b_pgrid[2];
      } else {
        APP_ABORT("scr_coulomb_t::W_w_to_tau: Error finding proper pgrid: gshape[2]*gshape[3] < np.");
      }
    }
    auto buffer_wi  = make_distributed_array<local_Array_t>(
        *comm, b_pgrid, w_gshape, dW_wqPQ_pos.block_size());
    _Timer.start("FT_REDISTRIBUTE");
    math::nda::redistribute(dW_wqPQ_pos, buffer_wi);
    _Timer.stop("FT_REDISTRIBUTE");
    if (reset_input) dW_wqPQ_pos.reset();

    auto buffer_ti  = make_distributed_array<local_Array_t>(
        *comm, b_pgrid, t_gshape, buffer_wi.block_size());
    {
      auto buf_ti_loc = buffer_ti.local();
      auto buf_wi_loc = buffer_wi.local();
      _ft->w_to_tau_PHsym(buf_wi_loc, buf_ti_loc);
    }
    buffer_wi.reset();
    _ft->check_leakage(buffer_ti, imag_axes_ft::boson, "screened interaction", true);

    auto dW_tqPQ = make_distributed_array<local_Array_t>(
        *comm, t_pgrid_out, t_gshape, t_bsize_out);

    _Timer.start("FT_REDISTRIBUTE");
    math::nda::redistribute(buffer_ti, dW_tqPQ);
    _Timer.stop("FT_REDISTRIBUTE");
    buffer_ti.reset();

    _Timer.stop("IMAG_FT_WtoT");
    return dW_tqPQ;
  }

  template<typename comm_t>
  void scr_coulomb_t::dump_eps_inv_head(const nda::ArrayOfRank<2> auto &eps_inv_head_tq,
                                        const nda::ArrayOfRank<1> auto &eps_inv_head_t,
                                        std::string coqui_h5_prefix, long iter,
                                        comm_t &comm, mf::MF &mf) {
    if (comm.root()) {
      long nw_half = (_ft->nw_b() % 2 == 0) ? _ft->nw_b() / 2 : _ft->nw_b() / 2 + 1;
      nda::array<ComplexType, 2> eps_inv_head_wq(nw_half, mf.nqpts_ibz());
      nda::array<ComplexType, 1> eps_inv_head_w(nw_half);
      auto eps_inv_w_2D = nda::reshape(eps_inv_head_w, shape_t<2>{nw_half, 1});
      auto eps_inv_t_2D = nda::reshape(eps_inv_head_t, shape_t<2>{eps_inv_head_t.shape(0), 1});

      _ft->tau_to_w_PHsym(eps_inv_head_tq, eps_inv_head_wq);
      _ft->tau_to_w_PHsym(eps_inv_t_2D, eps_inv_w_2D);

      std::string filename = coqui_h5_prefix + ".mbpt.h5";
      std::string grp_name = "iter" + std::to_string(iter);
      h5::file file(filename, 'a');
      h5::group grp(file);
      auto scf_grp = (grp.has_subgroup("scf")) ? grp.open_group("scf") : grp.create_group("scf");
      auto iter_grp = (scf_grp.has_subgroup(grp_name)) ?
                      scf_grp.open_group(grp_name) : scf_grp.create_group(grp_name);

      nda::h5_write(iter_grp, "eps_inv_head_wq", eps_inv_head_wq, false);
      nda::h5_write(iter_grp, "eps_inv_head_tq", eps_inv_head_tq, false);
      nda::h5_write(iter_grp, "eps_inv_head_w", eps_inv_head_w, false);
      nda::h5_write(iter_grp, "eps_inv_head_t", eps_inv_head_t, false);

      // ISDF-Vertex: macroscopic dielectric head eps_head(inu) = 1/eps^{-1}_head(q->0, inu)
      // and the static macroscopic dielectric constant epsilon_inf.
      // eps_inv_head_w stores (eps^{-1}_head - 1) on the bosonic Matsubara half-grid
      // (index 0 = inu=0), so the physical eps^{-1}_head = 1 + eps_inv_head_w. With an
      // active vertex this is P^C-corrected.
      nda::array<ComplexType, 1> eps_head_w(nw_half);
      for (long iw = 0; iw < nw_half; ++iw)
        eps_head_w(iw) = ComplexType(1.0) / (ComplexType(1.0) + eps_inv_head_w(iw));
      double epsilon_inf = 1.0 / (1.0 + eps_inv_head_w(0).real());
      nda::h5_write(iter_grp, "eps_head_w", eps_head_w, false);
      h5::h5_write(iter_grp, "epsilon_inf", epsilon_inf);
    }
    comm.barrier();
  }


  // template instantiations
  using Arr4D = nda::array<ComplexType, 4>;
  using Arr = nda::array<ComplexType, 5>;
  using Arrv = nda::array_view<ComplexType, 5>;
  using Arrv2 = nda::array_view<ComplexType, 5, nda::C_layout>;

  template void scr_coulomb_t::update_w(MBState&, thc_reader_t&, long);

  template memory::darray_t<Arr4D, mpi3::communicator>
  scr_coulomb_t::dyson_W_from_Pi_tau<true>(memory::darray_t<Arr4D, mpi3::communicator> &, thc_reader_t&, bool,
                                  std::array<long, 4>, std::array<long, 4>);
  template memory::darray_t<Arr4D, mpi3::communicator>
  scr_coulomb_t::dyson_W_from_Pi_tau<false>(memory::darray_t<Arr4D, mpi3::communicator> &, thc_reader_t&, bool,
                                   std::array<long, 4>, std::array<long, 4>);

  template memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>
  scr_coulomb_t::eval_Pi_qdep(const Arr&, thc_reader_t&, const projector_boson_t*,
                              const Arrv*, const Arrv*);
  template memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>
  scr_coulomb_t::eval_Pi_qdep(const Arrv&, thc_reader_t&, const projector_boson_t*,
                              const Arrv*, const Arrv*);
  template memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>
  scr_coulomb_t::eval_Pi_qdep(const Arrv2&, thc_reader_t&, const projector_boson_t*,
                              const Arrv*, const Arrv*);

  template memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>
  scr_coulomb_t::eval_Pi_qdep(MBState&, thc_reader_t&g);


  template memory::darray_t<Arr4D, mpi3::communicator>
  scr_coulomb_t::w_to_tau(memory::darray_t<Arr4D, mpi3::communicator> &,
                 std::array<long, 4>, std::array<long, 4>, bool);

  template memory::darray_t<Arr4D, mpi3::communicator>
  scr_coulomb_t::tau_to_w(memory::darray_t<Arr4D, mpi3::communicator> &,
                 std::array<long, 4>, std::array<long, 4>, bool);

  // instantiate templates
  template void scr_coulomb_t::dump_eps_inv_head(
      const nda::array<ComplexType,2> &, const nda::array<ComplexType,1> &,
      std::string, long, mpi3::communicator &, mf::MF &);


}  // solvers
}  // methods
