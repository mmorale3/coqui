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


#include <filesystem>
#include <unordered_set>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sys/resource.h>
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/HF/thc_solver_comm.hpp"
#include "methods/GW/g0_div_utils.hpp"
#include "methods/vertex/vertex_t.h"
#include "methods/vertex/vertex_sigma_interp.hpp"
#include "methods/vertex/nu_sampling.hpp"
#include "methods/vertex/vertex_secondary_fold.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "nda/linalg/eigenelements.hpp"
#include "hamiltonian/one_body_hamiltonian.hpp"   // scGW-tilde C4: H0 for the CVV velocity
#include "hamiltonian/pseudo/pseudopot.h"
#include "cvv_head.hpp"
#include "scr_coulomb_t.h"
#include "rpa_pi.icc"
#include "edmft_pi.icc"

namespace methods {
namespace solvers {

  namespace ladder_meter {
    // Injection-side profiling meters (notes/ladder_profiling_spec.md section 2b/2c).
    // PURE OBSERVERS: nothing here is read by the physics.
    inline double rss_gb() {
      struct rusage ru;
      if (getrusage(RUSAGE_SELF, &ru) != 0) return 0.0;
#if defined(__APPLE__)
      return double(ru.ru_maxrss) / (1024.0 * 1024.0 * 1024.0);
#else
      return double(ru.ru_maxrss) / (1024.0 * 1024.0);
#endif
    }
    struct watch {
      std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
      double lap() {
        auto t1 = std::chrono::steady_clock::now();
        double d = std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
        return d;
      }
    };
    // cumulative ladder wall across outer iterations (the section-2c iteration line)
    inline double cum_eval = 0.0, cum_inject = 0.0;
    inline long ncalls = 0;
  }

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

    // Pre-register every timer print_timers() reads: TimerManager::elapsed() aborts on an
    // unregistered name, and these phases are conditional (EVALUATE_W only runs on the THC
    // path, the IMAG_FT pair only when a transform is needed).
    // T-3b (timers only, notes/coqui_threading_t3a.md section 4.5): the THC-RPA sub-timers
    // below were being COLLECTED by rpa_pi.icc but never printed on the scr_coulomb path, so
    // PI_HADPROD_R / PI_PRIM_TO_AUX / PI_FT_R were invisible on the qpGW path and only the
    // aggregate EVALUATE_PI could be read. Purely additive: measurement, not threading --
    // row 9 (the Pi Hadamard) stays UNTOUCHED per RULING R-T3-1 item 4.
    for (auto const &v : {"EVALUATE_PI", "DYSON_W", "EVALUATE_W",
                          "IMAG_FT_TtoW", "IMAG_FT_WtoT", "FT_REDISTRIBUTE",
                          "EVALUATE_PI_R", "PI_ALLOC_R", "PI_HADPROD_R",
                          "EVALUATE_PI_K", "PI_ALLOC_K", "PI_PRIM_TO_AUX", "PI_FT_R"})
      _Timer.add(v);
  }

  void scr_coulomb_t::print_timers() {
    app_log(2, "\n  SCREENED-COULOMB timers");
    app_log(2, "  -----------------------");
    app_log(2, "    Evaluate Pi (RPA + vertex): {0:.3f} sec", _Timer.elapsed("EVALUATE_PI"));
    // T-3b (t3a section 4.5): the RPA Pi internals, previously collected but never printed.
    // Zero rows mean that flavour of the RPA kernel did not run on this path.
    app_log(2, "      - RPA Pi, R space:        {0:.3f} sec", _Timer.elapsed("EVALUATE_PI_R"));
    app_log(2, "      - RPA Pi, k space:        {0:.3f} sec", _Timer.elapsed("EVALUATE_PI_K"));
    app_log(2, "        - Gij->Guv:             {0:.3f} sec", _Timer.elapsed("PI_PRIM_TO_AUX"));
    app_log(2, "        - Hadamard product:     {0:.3f} sec", _Timer.elapsed("PI_HADPROD_R"));
    app_log(2, "        - FT k<->R:             {0:.3f} sec", _Timer.elapsed("PI_FT_R"));
    app_log(2, "        - alloc/zero (R + k):   {0:.3f} sec",
            _Timer.elapsed("PI_ALLOC_R") + _Timer.elapsed("PI_ALLOC_K"));
    app_log(2, "    Dyson W:                    {0:.3f} sec", _Timer.elapsed("DYSON_W"));
    app_log(2, "      - solve (1-Z.Pi)^-1:      {0:.3f} sec", _Timer.elapsed("EVALUATE_W"));
    app_log(2, "    Imaginary FT tau->w:        {0:.3f} sec", _Timer.elapsed("IMAG_FT_TtoW"));
    app_log(2, "    Imaginary FT w->tau:        {0:.3f} sec", _Timer.elapsed("IMAG_FT_WtoT"));
    app_log(2, "      - FT_REDISTRIBUTE:        {0:.3f} sec\n", _Timer.elapsed("FT_REDISTRIBUTE"));
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
    // P18 (vertex_perf_plan.md): the in-process vertex chain. From the second dynamic readout on, the injected all-nu
    // object is the dump THIS run wrote at the previous update (the scripted chains' "restart + inject the previous run's
    // dump", without the restart); the user's pol_vertex_interp_file seeds the first update. Every consumer reads the
    // file name from the vertex objects, so the substitution is made here, once per update.
    if (h5_iter >= 0 and _vertex != nullptr and _vertex->pol_chain() and _pol_dyn_calls > 0) {
      utils::check(_pol_vtx != nullptr and _pol_vtx->ladder_dyn_all_nu(),
                   "scr_coulomb_t::update_w: pol_vertex_chain needs the all-nu dump (pol_vertex_dyn_all_nu = true).");
      const std::string prev = mb_state.coqui_prefix + ".pol_wh_dyn.g" + std::to_string(_pol_dyn_calls) + ".h5";
      utils::check(std::filesystem::exists(prev), "scr_coulomb_t::update_w: pol_vertex_chain: the previous update's dump {} is missing.", prev);
      _vertex->set_pol_interp(prev, _vertex->pol_interp_col());
      _pol_vtx->set_pol_interp(prev, _pol_vtx->pol_interp_col());
      app_log(1, "  [vertex chain] update {}: injecting this run's previous all-nu dump {} (generation {})", h5_iter, prev, _pol_dyn_calls);
    }

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
    // A.1: only the injection that runs inside THIS eval_Pi_qdep may hand its inu = 0 row
    // to the readout below (the bosonic closure calls eval_Pi_qdep outside update_w, and
    // that row must not survive into a later iteration's readout).
    _pol_nu0_row.reset();
    // T-1 item 2 (notes/coqui_threading_spec.md rev 2): this stage was the largest UNTIMED
    // block in the code. T-0 could only bound it by subtraction -- 59% of wall on the LiF
    // kp444 fixture, threading at ~1.42x, with no timer block of its own
    // (notes/coqui_threading_t0.md section 2.1). Without these meters T-2's "no phase
    // regresses at t=1" gate cannot be evaluated on the biggest phase in the run.
    _Timer.start("EVALUATE_PI");
    auto dPi_tqPQ = eval_Pi_qdep(mb_state, thc);
    _Timer.stop("EVALUATE_PI");

    // evaluate screened interaction (dW_tqPQ) and reset polarizability (dPi_tqPQ)
    // a) dPi_tqPQ is reset during dyson_W_from_Pi_tau()
    // b) pgrid and bsize of dW_tqPQ are forced to be the same as in dPi_tqPQ
    _Timer.start("DYSON_W");
    auto dW_tqPQ = dyson_W_from_Pi_tau<false>(dPi_tqPQ, thc, true);
    _Timer.stop("DYSON_W");
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
    // P25 / G32 (notes/vertex_perf_plan.md; [gw] eps_inf_fit = true, default off): epsilon_inf from the
    // SMALL-q FIT eps_M(q) = eps_inf + A |q|^2 (+ B |q|^4) of the loop's OWN static dielectric function
    // on the smallest nonzero |q| of the IBZ mesh -- eps_M(q) = 1 / (1 + Re[eps^-1_{00}(q, i nu = 0) - 1])
    // from eps_inv_head_q (the q-resolved head of THIS dW, div_treatment-independent), reported next to
    // the stored head above as the check of the div_treatment's q -> 0 recipe. Report-only, all ranks
    // evaluate the same replicated numbers; off = bitwise fallthrough of every existing line and dataset.
    std::optional<eps_fit::eps_inf_fit_t> eps_inf_fit_res;
    if (_eps_inf_fit) {
      eps_inf_fit_res = eval_eps_inf_fit(eps_inv_head_q, *thc.MF());
      auto const &f = eps_inf_fit_res.value();
      // the stored head's epsilon_inf (the line above), recomputed here so that block stays untouched
      double eps_inf_head = 0.0;
      {
        long nw_half = (_ft->nw_b() % 2 == 0) ? _ft->nw_b() / 2 : _ft->nw_b() / 2 + 1;
        nda::array<ComplexType, 2> eih_w(nw_half, 1);
        auto eih_t = nda::reshape(eps_inv_head, shape_t<2>{eps_inv_head.shape(0), 1});
        _ft->tau_to_w_PHsym(eih_t, eih_w);
        eps_inf_head = 1.0 / (1.0 + eih_w(0, 0).real());
      }
      if (f.ok) {
        std::string qs, bs;
        for (size_t i = 0; i < f.q_used.size(); ++i) {
          char buf[48];
          std::snprintf(buf, sizeof(buf), "%s%.6f", (i == 0) ? "" : ", ", f.q_used[i]);
          qs += buf;
        }
        if (f.degree >= 2) {
          char buf[48];
          std::snprintf(buf, sizeof(buf), ", B = %+.6e", f.coeffs[2]);
          bs = buf;
        }
        app_log(1, "    eps_inf (small-q fit, {} points |q| = {}; degree {} in |q|^2): {:.6f}, "
                   "A = {:+.6e}{}, residual = {:.3e}   [stored head - fit = {:+.6e}]\n",
                f.q_used.size(), qs, f.degree, f.eps_inf, f.coeffs[1], bs, f.residual,
                eps_inf_head - f.eps_inf);
      } else {
        app_log(1, "    eps_inf (small-q fit): skipped -- fewer than 2 distinct nonzero |q| on the IBZ mesh.\n");
      }
    }

    // scGW-tilde L2: the ladder eps_M readout (report-only; see pol_ladder_eps_readout).
    // Q3: eps_inv_head_q carries the loop's OWN q-resolved head, so the readout also
    // reports the loop-side eps_M(q_min) -- the second route of gate Q3-b(i).
    // Tier 2 (D3): the DYNAMIC-rung readout needs THIS iteration's dW folded into the readout
    // instance's W-bar cache, so it runs after the dW publication below; the static readout
    // keeps its historic place (bitwise).
    const bool pol_dyn_readout = (pol_readout and _pol_vtx != nullptr and _pol_vtx->ladder_dynamic_rung());
    if (pol_readout and not pol_dyn_readout)
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

    if (pol_dyn_readout) {
      _pol_vtx->cache_w(mb_state, thc);
      pol_ladder_eps_readout(mb_state, thc, _pol_pi0_qPQ, std::addressof(eps_inv_head_q));
    }

    if (h5_iter>=0) {
      dump_eps_inv_head(eps_inv_head_q, eps_inv_head,
                        mb_state.coqui_prefix, h5_iter,
                        thc.mpi()->comm, *thc.MF(),
                        eps_inf_fit_res.has_value() ? std::addressof(eps_inf_fit_res.value()) : nullptr);
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

    // LFF-Sigma (Route 1): the vertex correction of W for Sigma ONLY, from THIS iteration's W -- after every other
    // consumer of dW (the kernel cache above included), so W, W-bar and the readout are exactly those without the knob.
    if (_vertex != nullptr and _vertex->sigma_lff_enabled()) build_sigma_lff(mb_state, thc, t_pgrid, t_bsize);
    // LFF-Sigma Route 2 (L-6): the pair-resolved static-ladder vertex in Sigma, on the readout instance (same placement)
    if (_vertex != nullptr and _vertex->sigma_pair_enabled()) build_sigma_pair(mb_state, thc);

    print_timers();
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
    // W-int-0: if the user vertex is Wannierized, the private readout instance inherits the MLWF state
    // so the pol-vertex/dynbse runs in the mesh-independent Wannier-pair frame (coarse->fine interpolation).
    if (_vertex->wannier()) {
      _pol_vtx->adopt_wannier(*_vertex);
      app_log(1, "  [scGW-tilde L2] readout instance inherits the Wannier projector (M = {}): the vertex "
                 "polarization is produced in the MLWF-pair frame.", _vertex->subspace_rank());
    }
    // increment B: the ladder solve-grid knobs live on the knob carrier; the READOUT
    // instance is the one that actually runs eval_pol_ladder_whalf, so they travel here.
    _pol_vtx->set_ladder_solve(_vertex->ladder_solve_grid(),
                               _vertex->ladder_solve_budget_gb());
    // DA Phase 2 (notes/qsgwhat_discrepancy_spec.md): the D-1/D-4/D-7 knobs travel to the
    // READOUT instance for the same reason the solve-grid knobs do -- this is the vertex
    // that actually builds W0 (the ladder rung W-bar_0) and runs the pair-space ladder.
    // Finding F-DA-1 was precisely a knob that did NOT travel here.
    _pol_vtx->set_ladder_da(_vertex->ladder_tda(), _vertex->ladder_head_scale(),
                            _vertex->ladder_qnu_meter());
    // W-int-3 (notes/wannier_coarse_vertex_plan.md, the q -> 0 head): the knob carrier's B-L head scale travels too.
    // It multiplies the madelung weight of EVERY analytic q -> 0 head the vertex kernel inserts (build_w0's static
    // rung W-bar_0 AND cache_w's dynamic W-bar(q, i nu)); 0 = a fully HEAD-FREE (body-only) coarse vertex, the
    // theory's interpolable object; the loop's own RPA W and its div_treatment are untouched. Default 1 = bitwise.
    _pol_vtx->set_bl_head_scale(_vertex->bl_head_scale());
    // Tier 1.5 (notes/tier15_ward_legs_plan.md): the leg vertex travels the same way.
    _pol_vtx->set_ladder_legs(_vertex->ladder_legs());
    // Tier 2 full frequency (notes/dynbse_plan.md D3): the rung and its solve knobs too.
    _pol_vtx->set_ladder_rung(_vertex->ladder_rung(), _vertex->ladder_dyn_tol(), _vertex->ladder_dyn_maxit(),
                              _vertex->ladder_dyn_gmres(), _vertex->ladder_dyn_sign());
    _pol_vtx->set_ladder_dyn_rhs_block(_vertex->ladder_dyn_rhs_block());
    _pol_vtx->set_ladder_dyn_dump(_vertex->ladder_dyn_dump());
    _pol_vtx->set_wcache_mode(_vertex->wcache_mode());
    _pol_vtx->set_ladder_dyn_dense(_vertex->ladder_dyn_dense());
    _pol_vtx->set_ladder_dyn_resolvent(_vertex->ladder_dyn_resolvent());
    _pol_vtx->set_ladder_dyn_union_stride(_vertex->ladder_dyn_union_stride());
    _pol_vtx->set_ladder_dyn_table_mode(_vertex->ladder_dyn_table_mode());
    _pol_vtx->set_ladder_dyn_schedule(_vertex->ladder_dyn_schedule());
    _pol_vtx->set_ladder_dyn_iaft_prec(_vertex->ladder_dyn_iaft_prec());
    _pol_vtx->set_ladder_dyn_tfold(_vertex->ladder_dyn_tfold());
    _pol_vtx->set_ladder_dyn_vmask(_vertex->ladder_dyn_vmask_lo(), _vertex->ladder_dyn_vmask_hi());
    _pol_vtx->set_ladder_dyn_gamma1_only(_vertex->ladder_dyn_gamma1_only());
    _pol_vtx->set_ladder_dyn_all_nu(_vertex->ladder_dyn_all_nu());
    _pol_vtx->set_ladder_dyn_cut_r1(_vertex->ladder_dyn_cut_r1());
    _pol_vtx->set_ladder_dyn_bubble_only(_vertex->ladder_dyn_bubble_only());
    _pol_vtx->set_ladder_dyn_all_nu_nodes(_vertex->ladder_dyn_all_nu_nodes());
    _pol_vtx->set_ladder_dyn_fit(_vertex->ladder_dyn_fit_file(), _vertex->ladder_dyn_fit_rank(), _vertex->ladder_dyn_fit_mode(), _vertex->ladder_dyn_fit_auto_nodes());
    _pol_vtx->set_ladder_dyn_resum_mu_file(_vertex->ladder_dyn_resum_mu_file());
    // W-int-1b: the coarse->fine interpolation knobs travel to the readout instance too
    _pol_vtx->set_isdf_points(_vertex->isdf_points_file(), _vertex->isdf_points_dump());
    _pol_vtx->set_wannier_frame(_vertex->wannier_frame());
    _pol_vtx->set_pol_interp(_vertex->pol_interp_file(), _vertex->pol_interp_col());
    _pol_vtx->set_pol_chain(_vertex->pol_chain());
    _pol_vtx->set_sigma_share(_vertex->sigma_share());
    app_log(1, "  [scGW-tilde L2] ladder readout instance: C window = [{}, {}), "
               "secondary rank knob = {}, div_treatment = {} (kernel head follows "
               "build_w0's policy; W0bar is SAME-iteration -- coincides with "
               "pol_vertex_kernel = \"w0_prev\" at a fixed point, R4 note).",
            w.first(), w.last(), _vertex->pol_isdf_rank(), _div_treatment);
    app_log(1, "  [scGW-tilde L2] DA Phase-2 knobs on the readout instance: ladder_tda = "
               "{}, ladder_head_scale = {:.6g}, ladder_qnu_meter = {}.",
            _vertex->ladder_tda() ? "true" : "false", _vertex->ladder_head_scale(),
            _vertex->ladder_qnu_meter() ? "true" : "false");
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
   * eps(q_i, i nu) cuts (2026-09-11): select the transfers once (q_min plus evenly spaced
   * ranks of |q| among the non-Gamma IBZ transfers) and gather the RPA Pi rows at those
   * transfers on EVERY PH-sym bosonic half node j (the same folded transform rows
   * tau_to_w_PHsym applies, iw = nw_b/2 + j), reduced to rank 0 only (the per-(q, nu)
   * single-frequency Dysons of the cut report are rank-0 work; nothing is replicated).
   */
  template<nda::MemoryArrayOfRank<4> Array_t, typename communicator_t>
  void scr_coulomb_t::gather_cut_rows(THC_ERI auto &thc,
                                      memory::darray_t<Array_t, communicator_t> &dPi_tqPQ) {
    auto [nt_h, nq, Np, Nq2] = dPi_tqPQ.global_shape();
    auto MF = thc.MF();
    const long nsel_req = _vertex->eps_cut_nq();
    if (_pol_cut_q.empty()) {
      std::vector<std::pair<double, long>> qs;
      for (long iq = 0; iq < nq; ++iq) {
        auto qp = MF->Qpts_ibz(iq);
        const double q2 = qp(0) * qp(0) + qp(1) * qp(1) + qp(2) * qp(2);
        if (q2 > 1e-12) qs.emplace_back(q2, iq);
      }
      std::sort(qs.begin(), qs.end());
      const long navail = long(qs.size());
      const long nsel = std::min(nsel_req, navail);
      for (long i = 0; i < nsel; ++i) {
        const long r = (nsel == 1) ? 0 : std::lround(double(i) * double(navail - 1) / double(nsel - 1));
        _pol_cut_q.push_back(qs[size_t(r)].second);
      }
      std::string sel;
      for (long i = 0; i < nsel; ++i) {
        auto qp = MF->Qpts_ibz(_pol_cut_q[size_t(i)]);
        char buf[96];
        std::snprintf(buf, sizeof(buf), " iq %ld (|q| %.6f)", _pol_cut_q[size_t(i)],
                      std::sqrt(qp(0) * qp(0) + qp(1) * qp(1) + qp(2) * qp(2)));
        sel += buf;
      }
      app_log(1, "  [eps-cut] transfers selected ({} of {} non-Gamma IBZ q, by |q| rank):{}", nsel, navail, sel);
    }
    const long nsel = long(_pol_cut_q.size());
    if (nsel == 0) { _pol_pi_cut.reset(); return; }
    const long nt_b = _ft->nt_b(), nw_b = _ft->nw_b();
    const long nt_half = (nt_b % 2 == 0) ? nt_b / 2 : nt_b / 2 + 1;
    const long nw_half = (nw_b % 2 == 0) ? nw_b / 2 : nw_b / 2 + 1;
    utils::check(nt_h == nt_half, "gather_cut_rows: PH-sym tau half grid mismatch ({} vs {}).", nt_h, nt_half);
    auto Twt = _ft->Twt_bb();                                   // (nw_b, nt_b)
    nda::array<ComplexType, 2> Tpos(nw_half, nt_half);
    for (long n = 0; n < nw_half; ++n) {
      const long iw = nw_b / 2 + n;
      for (long it = 0; it < nt_half; ++it) {
        const long imt = nt_b - it - 1;
        Tpos(n, it) = (it == imt) ? Twt(iw, it) : Twt(iw, it) + Twt(iw, imt);
      }
    }
    std::vector<long> sel_of(size_t(nq), -1);
    for (long s = 0; s < nsel; ++s) sel_of[size_t(_pol_cut_q[size_t(s)])] = s;
    auto t_rng = dPi_tqPQ.local_range(0);
    auto q_rng = dPi_tqPQ.local_range(1);
    auto P_rng = dPi_tqPQ.local_range(2);
    auto Q_rng = dPi_tqPQ.local_range(3);
    auto Pi_loc = dPi_tqPQ.local();
    nda::array<ComplexType, 4> out(nsel, nw_half, Np, Nq2);
    out() = ComplexType(0.0);
    for (long iq = 0; iq < long(q_rng.size()); ++iq) {
      const long s = sel_of[size_t(q_rng.first() + iq)];
      if (s < 0) continue;
      for (long it = 0; it < long(t_rng.size()); ++it)
        for (long n = 0; n < nw_half; ++n) {
          const ComplexType r = Tpos(n, t_rng.first() + it);
          for (long iP = 0; iP < long(P_rng.size()); ++iP)
            for (long iQ = 0; iQ < long(Q_rng.size()); ++iQ)
              out(s, n, P_rng.first() + iP, Q_rng.first() + iQ) += r * Pi_loc(it, iq, iP, iQ);
        }
    }
    dPi_tqPQ.communicator()->reduce_in_place_n(out.data(), out.size(), std::plus<>{}, 0);
    if (dPi_tqPQ.communicator()->rank() == 0) {
      app_log(2, "  [eps-cut] RPA rows gathered on rank 0: {} q x {} nu nodes x {}^2 ({:.3f} GB)", nsel, nw_half, Np,
              double(out.size()) * 16.0 / 1.0e9);
      _pol_pi_cut.emplace(std::move(out));
    } else {
      _pol_pi_cut.reset();
    }
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

    ladder_meter::watch lwatch, ltot;
    const double rss_in = ladder_meter::rss_gb();
    nda::array<double, 1> lam;
    // Tier 1.5: with legs = "ward" the whalf pass also returns Delta P^Lambda (its inu = 0
    // row feeds the readout's "+DeltaLambda" column); Pl then already contains it (eq 27).
    const bool ward_legs = _pol_vtx->ladder_ward_legs();
    nda::array<ComplexType, 4> Pd;
    nda::array<ComplexType, 4> Pl;
    ++_pol_inj_calls;
    auto MF = thc.MF();
    auto lat = MF->lattv();
    auto Q = MF->Qpts();
    auto q_crys = [&](long iq, double *qc) {
      for (long i = 0; i < 3; ++i) { double v = 0.0; for (long j = 0; j < 3; ++j) v += lat(i, j) * Q(iq, j); qc[i] = v / (2.0 * M_PI); }
    };
    const long nw_h_ft = (_ft->nw_b() % 2 == 0) ? _ft->nw_b() / 2 : _ft->nw_b() / 2 + 1;
    nda::array<long, 1> nu_half(nw_h_ft);
    { auto wb = _ft->wn_mesh_b(); for (long j = 0; j < nw_h_ft; ++j) nu_half(j) = long(wb(_ft->nw_b() / 2 + j)); }
    const bool from_file = not _pol_vtx->pol_interp_file().empty();
    if (from_file) {
      // W-int-4f (notes/wannier_coarse_vertex_plan.md): the FULL-FREQUENCY consumer. P^{C,L}(q, i nu_j) on this mesh's
      // q list at ALL PH-sym half nodes comes from the file (a coarse run's <prefix>.pol_wh.g<n>.h5, or the offline
      // Route-B interpolant in the same layout) in the frozen-point secondary frame, instead of the ladder solve; the
      // upfold + nu -> tau + the += into the RPA polarization below run unchanged, so the W-Dyson, W and Sigma of the
      // fine loop all see the interpolated vertex. Requires the same imaginary-axis grid (checked on the node list).
      utils::check(not ward_legs, "inject_pol_ladder: pol_vertex_interp_file with legs = \"ward\" is not supported.");
      utils::check(not _pol_vtx->isdf_points_file().empty(),
                   "inject_pol_ladder: pol_vertex_interp_file needs pol_vertex_isdf_points_file (the file's secondary "
                   "frame is the coarse run's point set).");
      const std::string col = "Pi_" + _pol_vtx->pol_interp_col();
      Pl = read_pol_interp_column(_pol_vtx->pol_interp_col(), nq_g, nw_h_ft, nu_half, thc);   // the shared W-int-4f read
      lam = nda::array<double, 1>(nw_h_ft); lam() = 0.0;
      app_log(1, "  [W-int-4f] W-Dyson injection consumes {} from {} ({} half nodes x {} q matched; frozen points, N_m = {}).",
              col, _pol_vtx->pol_interp_file(), nw_h_ft, nq_g, _pol_vtx->secondary_rank());
    } else {
      Pl = _pol_vtx->eval_pol_ladder_whalf(mb_state, thc, &lam, ward_legs ? std::addressof(Pd) : nullptr);
    }
    const double t_eval = lwatch.lap();
    const double rss_eval = ladder_meter::rss_gb();
    auto const &tmap = _pol_vtx->secondary_transfer();               // (nq, Nm, Np)
    const long nw_h = Pl.shape(0), Nm = Pl.shape(2);
    if (_pol_vtx->ladder_dyn_dump() and not from_file and thc.mpi()->comm.root()) {
      // W-int-4f: the coarse run's FULL-FREQUENCY injection object (the resummed static-rung ladder at every PH-sym
      // half node, the L3 object) in the frozen-able secondary frame -- the offline Route-B tool's input, and the fine
      // consumer's format (pol_vertex_interp_file, col "ladder").
      const std::string fn = mb_state.coqui_prefix + ".pol_wh.g" + std::to_string(_pol_inj_calls) + ".h5";
      nda::array<double, 2> qc(nq_g, 3);
      for (long iq = 0; iq < nq_g; ++iq) { double v[3]; q_crys(iq, v); for (long i = 0; i < 3; ++i) qc(iq, i) = v[i]; }
      h5::file f(fn, 'w');
      h5::group g(f);
      nda::h5_write(g, "q", qc);
      nda::h5_write(g, "nu_half", nu_half);
      h5::h5_write(g, "beta", _ft->beta());
      h5::h5_write(g, "nw_b", long(_ft->nw_b()));
      nda::h5_write(g, "Pi_ladder", Pl);
      h5::h5_write(g, "nout", Nm);
      h5::h5_write(g, "frame", std::string("aux"));
      h5::h5_write(g, "wannier", long(_pol_vtx->wannier() ? 1 : 0));
      h5::h5_write(g, "nm", Nm);
      h5::h5_write(g, "window_first", long(_pol_vtx->pol_band_window().first()));
      h5::h5_write(g, "window_size", long(_pol_vtx->pol_band_window().size()));
      if (_pol_vtx->secondary_points().size() > 0) nda::h5_write(g, "ipts", _pol_vtx->secondary_points());
      app_log(1, "  [W-int-4f] full-frequency ladder injection object written to {} ({} half nodes x {} q x {} x {}).",
              fn, nw_h, nq_g, Nm, Nm);
    }
    utils::check(Pl.shape(1) == nq_g and tmap.shape(0) == nq_g and tmap.shape(1) == Nm
                 and tmap.shape(2) == Np,
                 "inject_pol_ladder: shape mismatch (ladder {}x{}x{}, t {}x{}x{}, Pi q = "
                 "{}, Np = {}).", Pl.shape(0), Pl.shape(1), Nm, tmap.shape(0),
                 tmap.shape(1), tmap.shape(2), nq_g, Np);

    // A.1 (notes/ladder_opt_spec.md): stash the inu = 0 row for the eps_M readout of THIS
    // update_w. Half node 0 IS the nw_b/2 node eval_pol_ladder_nu0 pins (ladder_whalf_gate
    // node_map_resid), and the whalf return is already all_reduced/replicated, so the
    // readout consumes it verbatim -- bit-identical to the second, independent pair-space
    // ladder pass it replaces, minus that pass's entire wall (measured 20% of the combined
    // ladder wall, at 70% idle: profiling results section 1.2).
    _pol_nu0_row.emplace(nda::array<ComplexType, 3>(Pl(0, nda::ellipsis{})));
    if (ward_legs) _pol_nu0_dlam.emplace(nda::array<ComplexType, 3>(Pd(0, nda::ellipsis{})));
    else _pol_nu0_dlam.reset();

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
    // ---- DA D-7: the (q, nu) decomposition meter (notes/qsgwhat_discrepancy_spec.md) ----
    // PURE OBSERVER. Accumulates, in the GLOBAL THC aux basis (i.e. AFTER the upfold, which
    // is what the Dyson actually sees), the squared Frobenius norm of the injected P^lad
    // per (q, i.nu) and -- alongside the RPA polarization it corrects -- per (q, tau). Every
    // physics write below is untouched, and with the knob off no accumulator is even sized.
    const bool qnu_meter = (_vertex != nullptr and _vertex->ladder_qnu_meter());
    std::vector<double> lad_qw, lad_qt, rpa_qt;
    if (qnu_meter) {
      lad_qw.assign(size_t(nq_g * nw_h), 0.0);
      lad_qt.assign(size_t(nq_g * nt_h), 0.0);
      rpa_qt.assign(size_t(nq_g * nt_h), 0.0);
    }
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
      if (qnu_meter)
        for (long j = 0; j < nw_h; ++j) {
          double s = 0.0;
          for (long i = 0; i < nPl; ++i)
            for (long jj = 0; jj < nQl; ++jj) s += std::norm(A(j, i, jj));
          lad_qw[size_t(iq * nw_h + j)] += s;
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
            if (qnu_meter) {
              const size_t o = size_t(iq * nt_h + t_rng.first() + it);
              lad_qt[o] += a * a;
              rpa_qt[o] += std::norm(Pi_loc(it, iql, i, j));   // BEFORE the +=
            }
            Pi_loc(it, iql, i, j) += v;
          }
    }
    const double t_upfold = lwatch.lap();
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

    // --- section 2b/2c meters: the ladder wall of THIS outer iteration + the RSS trace.
    // t_meters is charged to the r_rt / q-max diagnostics between the two laps above.
    {
      const double t_diag = lwatch.lap();
      const double t_tot = ltot.lap();
      const double rss_out = ladder_meter::rss_gb();
      ladder_meter::cum_eval += t_eval;
      ladder_meter::cum_inject += t_tot;
      ++ladder_meter::ncalls;
      double rss_mx = comm.all_reduce_value(rss_out, boost::mpi3::max<>{});
      double rss_ev = comm.all_reduce_value(rss_eval, boost::mpi3::max<>{});
      double rss_e0 = comm.all_reduce_value(rss_in, boost::mpi3::max<>{});
      app_log(1, "  [ladder-prof inject] iteration {}: ladder eval = {:.2f} s of {:.2f} s "
                 "injection ({:.1f}%); upfold+nu->tau = {:.2f} s, meters = {:.2f} s; "
                 "cumulative ladder eval = {:.2f} s over {} iterations",
              ladder_meter::ncalls, t_eval, t_tot,
              100.0 * t_eval / std::max(t_tot, 1e-300), t_upfold, t_diag,
              ladder_meter::cum_eval, ladder_meter::ncalls);
      app_log(1, "  [ladder-prof inject] MaxRSS GB (max over ranks): entry {:.2f} -> "
                 "after ladder eval {:.2f} -> exit {:.2f}", rss_e0, rss_ev, rss_mx);
    }
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
    // ---- DA D-7 report: WHERE IN (q, nu) THE INJECTED CORRECTION LIVES -----------------
    if (qnu_meter) {
      comm.all_reduce_in_place_n(lad_qw.data(), lad_qw.size(), std::plus<>{});
      comm.all_reduce_in_place_n(lad_qt.data(), lad_qt.size(), std::plus<>{});
      comm.all_reduce_in_place_n(rpa_qt.data(), rpa_qt.size(), std::plus<>{});
      auto MF = thc.MF();
      // star multiplicity of each IBZ transfer (the weight its cell carries in any q sum)
      std::vector<long> mult(size_t(nq_g), 0);
      {
        auto q2i = MF->qp_to_ibz();
        for (long i = 0; i < q2i.shape(0); ++i) {
          const long ib = q2i(i);
          if (ib >= 0 and ib < nq_g) ++mult[size_t(ib)];
        }
      }
      double tot_w = 0.0;
      for (auto v : lad_qw) tot_w += v;
      app_log(1, "\n  [DA D-7] (q, nu) DECOMPOSITION of the injected P^lad "
                 "(global THC aux basis, Frobenius norms).");
      app_log(1, "  [DA D-7]   q-marginal (sum over the {} PH-sym nu nodes), and the "
                 "tau-summed strength relative to P^RPA:", nw_h);
      app_log(1, "  [DA D-7]   {:>4} {:>12} {:>6} {:>14} {:>10} {:>14} {:>14} {:>10}",
              "iq", "|q|^2", "mult", "||P^lad(q)||", "share", "||P^lad(q)||_t",
              "||P^RPA(q)||_t", "ratio");
      for (long q = 0; q < nq_g; ++q) {
        double sw = 0.0, st = 0.0, sr = 0.0;
        for (long j = 0; j < nw_h; ++j) sw += lad_qw[size_t(q * nw_h + j)];
        for (long it = 0; it < nt_h; ++it) {
          st += lad_qt[size_t(q * nt_h + it)];
          sr += rpa_qt[size_t(q * nt_h + it)];
        }
        auto qp = MF->Qpts_ibz(q);
        const double q2 = qp(0) * qp(0) + qp(1) * qp(1) + qp(2) * qp(2);
        app_log(1, "  [DA D-7]   {:>4} {:>12.5e} {:>6} {:>14.6e} {:>10.4f} {:>14.6e} "
                   "{:>14.6e} {:>10.4e}",
                q, q2, mult[size_t(q)], std::sqrt(sw),
                (tot_w > 0.0 ? sw / tot_w : 0.0), std::sqrt(st), std::sqrt(sr),
                std::sqrt(st) / std::max(std::sqrt(sr), 1e-300));
      }
      {
        std::ostringstream o2;
        o2 << std::scientific << std::setprecision(3);
        for (long j = 0; j < nw_h; ++j) {
          double s = 0.0;
          for (long q = 0; q < nq_g; ++q) s += lad_qw[size_t(q * nw_h + j)];
          o2 << (j ? " " : "") << std::sqrt(s);
        }
        app_log(1, "  [DA D-7]   nu-marginal ||P^lad(nu)|| (PH-sym half grid, node 0 = "
                   "i.nu = 0): {}", o2.str());
      }
      for (long q = 0; q < nq_g; ++q) {
        std::ostringstream o3;
        o3 << std::scientific << std::setprecision(3);
        for (long j = 0; j < nw_h; ++j)
          o3 << (j ? " " : "") << std::sqrt(lad_qw[size_t(q * nw_h + j)]);
        app_log(1, "  [DA D-7]   ||P^lad(q = {}, nu)||: {}", q, o3.str());
      }
      app_log(1, "");
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


  /** the pair-resolved Sigma vertex's options from the user's vertex object (build_sigma_pair and the P3 arming) */
  static vertex_t::sigma_pair_opts sigma_pair_opts_of(vertex_t const &v, MBState const &mb_state) {
    vertex_t::sigma_pair_opts o;
    o.col = v.sigma_pair_col();
    o.outer = v.sigma_pair_outer();
    o.scale = v.sigma_pair_scale();
    o.hermitize = v.sigma_pair_herm();
    o.nu_diag = v.sigma_pair_diag();
    o.sign_ks = v.ladder_dyn_sign();
    o.side = v.sigma_pair_side();
    o.ibz = v.sigma_pair_ibz();
    o.dyn_ckpt_minutes = v.sigma_dyn_ckpt_minutes();   // P20
    o.dyn_refit = v.sigma_dyn_refit(); o.dyn_refit_rtol = v.sigma_dyn_refit_rtol();   // P12
    o.dyn_acc = v.sigma_dyn_acc();   // P4-C14
    o.dyn_dump = v.sigma_dyn_dump(); o.dyn_nodes = v.sigma_dyn_nodes();
    o.dyn_fit_file = v.sigma_dyn_fit_file(); o.dyn_fit_rank = v.sigma_dyn_fit_rank();
    o.dump_prefix = mb_state.coqui_prefix;
    return o;
  }

  // W-int-4f coarse side: the dynamic-rung ladder on ALL transfers x ALL PH-sym half nodes, written as the full-frequency
  // vertex object <prefix>.pol_wh_dyn.g<n>.h5 (four columns; the fine W-Dyson feed reads one of them by name).
  void scr_coulomb_t::dump_pol_dyn_all_nu(MBState &mb_state, THC_ERI auto &thc, long gen) {
    decltype(nda::range::all) all;
    auto MF = thc.MF();
    const long nw_b = _ft->nw_b();
    const long nw_h = (nw_b % 2 == 0) ? nw_b / 2 : nw_b / 2 + 1;
    const long nq = MF->nqpts_ibz();
    std::vector<long> nodes_all(static_cast<size_t>(nw_h));
    std::vector<long> qs(static_cast<size_t>(nq));
    for (long j = 0; j < nw_h; ++j) nodes_all[size_t(j)] = j;
    for (long iq = 0; iq < nq; ++iq) qs[size_t(iq)] = iq;
    // LFF-aux L-0 (notes/lff_aux_plan.md): the dynamic columns on the SAMPLED half nodes (pol_vertex_dyn_all_nu_nodes;
    // all nodes when empty), the window bubble Pi_bub on ALL nodes; bubble_only skips the dynamic columns altogether.
    const bool bub_only = _pol_vtx->ladder_dyn_bubble_only();
    std::vector<long> nodes = _pol_vtx->ladder_dyn_all_nu_nodes();
    // P14: pol_vertex_dyn_fit_auto_nodes = K -> the K sampled half nodes chosen from the fit file's Gamma_1 (else static) column:
    // nu = 0 and the highest node forced, the rest the row pivots of the top-K nu-modes (nu_sampling.hpp)
    if (nodes.empty() and _pol_vtx->ladder_dyn_fit_auto_nodes() > 0 and not _pol_vtx->ladder_dyn_fit_file().empty()) {
      const long Kn = _pol_vtx->ladder_dyn_fit_auto_nodes();
      std::vector<long> chosen;
      if (thc.mpi()->comm.root()) {
        h5::file ff(_pol_vtx->ladder_dyn_fit_file(), 'r');
        h5::group gf(ff);
        const char *pref[3] = {"Pi_gam1", "Pi_dyn", "Pi_static"};
        std::string col;
        for (auto *c : pref) if (gf.has_dataset(c)) { col = c; break; }
        utils::check(not col.empty(), "dump_pol_dyn_all_nu: the fit file {} has no Pi_gam1 / Pi_dyn / Pi_static column for the automatic node choice.",
                     _pol_vtx->ladder_dyn_fit_file());
        nda::array<ComplexType, 4> T;
        nda::h5_read(gf, col, T);
        utils::check(T.shape(0) == nw_h, "dump_pol_dyn_all_nu: the fit file's {} has {} half nodes, this run {}.", col, T.shape(0), nw_h);
        auto T2 = nda::reshape(T, std::array<long, 2>{nw_h, long(T.size()) / nw_h});
        auto G = nusamp::gram(T2);
        chosen = nusamp::pivot_nodes(G, Kn, std::vector<long>{0L, nw_h - 1});
        std::string lst;
        for (long n : chosen) lst += std::to_string(n) + " ";
        app_log(1, "  [LFF L-3] automatic sampled nodes ({} of {}, from the {} column of {}): {}", chosen.size(), nw_h, col,
                _pol_vtx->ladder_dyn_fit_file(), lst);
      }
      long nch = long(chosen.size());
      thc.mpi()->comm.broadcast_value(nch, 0);
      chosen.resize(size_t(nch));
      thc.mpi()->comm.broadcast_n(chosen.data(), nch, 0);
      nodes = chosen;
    }
    if (nodes.empty()) nodes = nodes_all;
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    for (long j : nodes)
      utils::check(j >= 0 and j < nw_h, "dump_pol_dyn_all_nu: pol_vertex_dyn_all_nu_nodes entry {} outside [0, {}).", j, nw_h);
    const bool subset = (long(nodes.size()) != nw_h);
    app_log(1, "  [W-int-4f] all-nu dynamic-rung dump: {} of {} half nodes x {} transfers (Gamma_1-only = {}, bubble_only = {}).",
            nodes.size(), nw_h, nq, _pol_vtx->ladder_dyn_gamma1_only(), bub_only);
    std::optional<vertex_t::dynbse_cut_result> rd;   // the dynamic columns on the sampled nodes
    if (not bub_only) {
      // P3 (vertex_perf_plan.md): with pol_vertex_sigma_share the Sigma hook of the dynamic pair vertex rides THIS solve for
      // the P nodes that belong to the Sigma node set (the same units on the same W-bar cache); build_sigma_pair, later in
      // this update, solves only the remaining nodes
      if (_vertex != nullptr and _vertex->sigma_share() and _vertex->sigma_pair_enabled() and _vertex->sigma_pair_dynamic()
          and long(qs.size()) == nq and not _pol_vtx->wannier()) {
        const long m0 = _ft->nw_b() / 2;
        std::vector<long> p_full;
        for (long j : nodes) p_full.push_back(m0 + j);
        _pol_vtx->arm_shared_sigma_hook(mb_state, thc, sigma_pair_opts_of(*_vertex, mb_state), p_full);
      }
      rd.emplace(_pol_vtx->eval_pol_dynbse_cut(mb_state, thc, nodes, qs, gen, false));
      _pol_dyn_ritz = std::max(_pol_dyn_ritz, rd->ritz_max);
    }
    // the bubble at ALL nodes: the dynamic call's own column when it covered every node, else a bubble-only pass
    nda::array<ComplexType, 4> Pb;
    if (rd.has_value() and not subset) {
      Pb = std::move(rd->Pi_bub);
    } else {
      auto rb = _pol_vtx->eval_pol_dynbse_cut(mb_state, thc, nodes_all, qs, gen, true);
      Pb = std::move(rb.Pi_bub);
    }
    if (thc.mpi()->comm.root()) {
      auto lat = MF->lattv();
      auto Q = MF->Qpts();
      nda::array<double, 2> qc(nq, 3);
      for (long iq = 0; iq < nq; ++iq)
        for (long i = 0; i < 3; ++i) { double v = 0.0; for (long j = 0; j < 3; ++j) v += lat(i, j) * Q(iq, j); qc(iq, i) = v / (2.0 * M_PI); }
      nda::array<long, 1> nu_half(nw_h);
      { auto wb = _ft->wn_mesh_b(); for (long j = 0; j < nw_h; ++j) nu_half(j) = long(wb(nw_b / 2 + j)); }
      nda::array<long, 1> nu_sampled(long(nodes.size()));
      for (long j = 0; j < long(nodes.size()); ++j) nu_sampled(j) = nodes[size_t(j)];
      const long Nm = Pb.shape(2);
      const std::string fn = mb_state.coqui_prefix + ".pol_wh_dyn.g" + std::to_string(gen) + ".h5";
      h5::file f(fn, 'w');
      h5::group g(f);
      nda::h5_write(g, "q", qc);
      nda::h5_write(g, "nu_half", nu_half);
      nda::h5_write(g, "nu_sampled", nu_sampled);
      h5::h5_write(g, "beta", _ft->beta());
      h5::h5_write(g, "nw_b", nw_b);
      if (rd.has_value()) {
        const char *names[4] = {"Pi_static", "Pi_dyn1", "Pi_gam1", "Pi_dyn"};
        // LFF-aux L-3: the on-demand fit. With a sampled-node dump and pol_vertex_dyn_fit_file, each column is refit at ALL
        // nodes in the nu-basis learned from the file's matching column (its top-K nu-modes; K = fit_rank or the number of
        // sampled nodes), by least squares on the sampled rows; the fit is exact at the sampled nodes when K = |nodes|.
        const std::string fit_file = _pol_vtx->ladder_dyn_fit_file();
        const bool do_fit = subset and not fit_file.empty();
        const long Kfit = (_pol_vtx->ladder_dyn_fit_rank() > 0) ? std::min<long>(_pol_vtx->ladder_dyn_fit_rank(), long(nodes.size()))
                                                               : long(nodes.size());
        std::optional<h5::file> ff;
        if (do_fit) {
          ff.emplace(fit_file, 'r');
          h5::group gf(ff.value());
          nda::array<long, 1> nu_f;
          nda::h5_read(gf, "nu_half", nu_f);
          utils::check(nu_f.size() == nw_h, "dump_pol_dyn_all_nu: the fit file {} has {} half nodes, this run {}.", fit_file, nu_f.size(), nw_h);
          for (long j = 0; j < nw_h; ++j)
            utils::check(nu_f(j) == nu_half(j), "dump_pol_dyn_all_nu: the fit file's half node {} is Matsubara index {}, {} here.", j, nu_f(j), nu_half(j));
          app_log(1, "  [LFF L-3] on-demand fit: {} sampled nodes -> all {} nodes in the nu-basis of {} ({} modes per column).",
                  nodes.size(), nw_h, fit_file, Kfit);
        }
        const long NS = long(nodes.size()), ncol = nq * Nm * Nm;
        // LFF: the Gamma_1 -> resummed factor mu(nu_j) (pol_vertex_dyn_resum_mu_file): Pi_dyn = mu Pi_gam1
        std::vector<double> mu_tab;
        const std::string mu_file = _pol_vtx->ladder_dyn_resum_mu_file();
        if (not mu_file.empty()) {
          std::ifstream fin(mu_file);
          utils::check(fin.good(), "dump_pol_dyn_all_nu: cannot open pol_vertex_dyn_resum_mu_file {}.", mu_file);
          std::string line;
          while (std::getline(fin, line)) {
            const auto h = line.find('#');
            if (h != std::string::npos) line = line.substr(0, h);
            std::istringstream ls(line);
            std::vector<double> v; double x;
            while (ls >> x) v.push_back(x);
            if (v.empty()) continue;
            mu_tab.push_back(v.back());   // "mu" or "j nu mu": the last number of the line
          }
          utils::check(long(mu_tab.size()) == nw_h, "dump_pol_dyn_all_nu: {} has {} mu values, this run has {} half nodes.",
                       mu_file, mu_tab.size(), nw_h);
          app_log(1, "  [LFF mu] resummation factor from {}: Pi_dyn = mu(nu) Pi_gam1 with mu(0) = {:.4f}, mu(nu_max) = {:.4f}.",
                  mu_file, mu_tab.front(), mu_tab.back());
        }
        nda::array<ComplexType, 4> P_gam1;   // the written Gamma_1 column (kept for the mu-scaled Pi_dyn)
        for (int c = 0; c < 4; ++c) {
          nda::array<ComplexType, 4> P(nw_h, nq, Nm, Nm);
          P() = ComplexType(0.0);
          for (long j = 0; j < NS; ++j) P(nodes[size_t(j)], all, all, all) = rd->Pi(c, j, all, all, all);
          bool fitted = false;
          if (do_fit) {
            h5::group gf(ff.value());
            if (gf.has_dataset(names[c])) {
              nda::array<ComplexType, 4> T;
              nda::h5_read(gf, names[c], T);
              utils::check(T.shape(0) == nw_h and T.shape(1) == nq and T.shape(2) == Nm and T.shape(3) == Nm,
                           "dump_pol_dyn_all_nu: the fit file's {} is {} x {} x {} x {}, expected {} x {} x {} x {}.", names[c],
                           T.shape(0), T.shape(1), T.shape(2), T.shape(3), nw_h, nq, Nm, Nm);
              auto T2 = nda::reshape(T, std::array<long, 2>{nw_h, ncol});
              // the nu-modes of the basis column (its Gram over the nodes) and the reconstruction R (nw_h x NS) of all nodes
              // from the sampled ones (nu_sampling.hpp): "modes" = the L-3 least squares on the top-K modes (exact at the
              // sampled nodes when K = NS), "regression" = G(:,S) G(S,S)^-1_K (P14)
              const std::string fmode = _pol_vtx->ladder_dyn_fit_mode();
              auto G = nusamp::gram(T2);
              double lkept = 0.0;
              const long nto = nusamp::modes_to(G, 1e-4, Kfit, &lkept);
              (void)nto;
              auto R = nusamp::reconstruction(G, nodes, Kfit, fmode);
              nda::matrix<ComplexType> PS(NS, ncol), Pfull(nw_h, ncol);
              for (long j = 0; j < NS; ++j) { auto row = nda::reshape(rd->Pi(c, j, all, all, all), std::array<long, 1>{ncol}); PS(j, all) = row; }
              nda::blas::gemm(R, PS, Pfull);
              // the residual at the sampled rows (exact when Kfit = NS) and the Hermitized full object
              double rs = 0.0, ns = 0.0;
              for (long j = 0; j < NS; ++j)
                for (long i = 0; i < ncol; ++i) {
                  const ComplexType f = Pfull(nodes[size_t(j)], i);
                  rs += std::norm(f - PS(j, i)); ns += std::norm(PS(j, i));
                }
              for (long j = 0; j < nw_h; ++j)
                for (long iq = 0; iq < nq; ++iq)
                  for (long M = 0; M < Nm; ++M)
                    for (long Nn = 0; Nn < Nm; ++Nn)
                      P(j, iq, M, Nn) = 0.5 * (Pfull(j, (iq * Nm + M) * Nm + Nn) + std::conj(Pfull(j, (iq * Nm + Nn) * Nm + M)));
              fitted = true;
              double pmax = 0.0;
              for (auto const &v : P) pmax = std::max(pmax, std::abs(v));
              app_log(1, "  [LFF L-3]   {}: {} modes carry {:.6f} of the basis column's |.|^2 ({} form); residual at the sampled nodes {:.2e}; "
                         "max |P| {:.3e}", names[c], Kfit, lkept, fmode, (ns > 0.0) ? std::sqrt(rs / ns) : 0.0, pmax);
            } else {
              app_log(1, "  [LFF L-3]   {}: not in the fit file -- written at the sampled nodes only.", names[c]);
            }
          }
          if (c == 2 and not mu_tab.empty()) P_gam1 = P;
          if (c == 3 and not mu_tab.empty()) {
            for (long j = 0; j < nw_h; ++j) P(j, all, all, all) = ComplexType(mu_tab[size_t(j)]) * P_gam1(j, all, all, all);
            app_log(1, "  [LFF mu]   Pi_dyn written as mu(nu) x Pi_gam1 ({} nodes).", nw_h);
          }
          nda::h5_write(g, names[c], P);
          (void)fitted;
        }
        if (do_fit) { h5::h5_write(g, "lff_fit_file", fit_file); h5::h5_write(g, "lff_fit_rank", Kfit); }
        if (not mu_tab.empty()) {
          nda::array<double, 1> mu_a(nw_h);
          for (long j = 0; j < nw_h; ++j) mu_a(j) = mu_tab[size_t(j)];
          nda::h5_write(g, "lff_mu", mu_a);
        }
      }
      nda::h5_write(g, "Pi_bub", Pb);
      h5::h5_write(g, "nout", Nm);
      h5::h5_write(g, "frame", std::string("aux"));
      h5::h5_write(g, "wannier", long(_pol_vtx->wannier() ? 1 : 0));
      h5::h5_write(g, "nm", Nm);
      h5::h5_write(g, "window_first", long(_pol_vtx->pol_band_window().first()));
      h5::h5_write(g, "window_size", long(_pol_vtx->pol_band_window().size()));
      if (_pol_vtx->secondary_points().size() > 0) nda::h5_write(g, "ipts", _pol_vtx->secondary_points());
      if (rd.has_value())
        app_log(1, "  [W-int-4f] all-nu dynamic-rung columns written to {} ({} half nodes ({} sampled) x {} q x {} x {} + the window "
                   "bubble Pi_bub; watchdog max |Ritz| {:.3f}, all converged {}).", fn, nw_h, nodes.size(), nq, Nm, Nm,
                rd->ritz_max, rd->all_converged);
      else
        app_log(1, "  [W-int-4f] bubble_only: the window bubble column Pi_bub written to {} ({} half nodes x {} q x {} x {}; no "
                   "dynamic columns).", fn, nw_h, nq, Nm, Nm);
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

    // W-int-0: the WANNIER-vertex DUMP path. In Wannier mode the vertex polarization is the MLWF-pair
    // Pi_loc (mesh-independent, for coarse->fine interpolation), which the aux eps readout below cannot
    // consume (and eval_pol_ladder_nu0 is window-only). So run ONLY the dynamic vertex, which dumps
    // Pi_loc(q) when pol_vertex_dyn_dump is set, and return -- bypassing the aux static ladder + eps.
    if (_pol_vtx->wannier() and _pol_vtx->ladder_dynamic_rung()) {
      ++_pol_dyn_calls;
      auto dres = _pol_vtx->eval_pol_dynbse_nu0(mb_state, thc, _pol_dyn_calls);
      _pol_dyn_ritz = dres.ritz_max;
      if (_pol_vtx->ladder_dyn_all_nu()) dump_pol_dyn_all_nu(mb_state, thc, _pol_dyn_calls);
      app_log(1, "  [scGW-tilde T2] Wannier-vertex DUMP: Pi_loc(q) produced in the MLWF-pair frame "
                 "({} q x {} x {}) -- dumped for coarse->fine interpolation; the aux eps readout is bypassed.",
              dres.Pi_gam1.shape(0), dres.Pi_gam1.shape(1), dres.Pi_gam1.shape(2));
      return;
    }

    // the ladder at inu = 0 in the readout vertex's secondary basis + upfold.
    // A.1: when the injection already ran in THIS update_w it produced exactly this row as
    // half node 0 of its whalf pass, so consume that instead of re-solving the pair-space
    // ladder from scratch (the injection's row is the SAME iteration's G and W0bar -- the
    // readout's invariant). Consume-once: reset so no later call can read a stale row.
    // With the injection disabled (or any standalone caller) the row is absent and the
    // historic self-contained evaluation runs unchanged.
    nda::array<ComplexType, 3> Pl_qmm;
    // Tier 1.5: the "+DeltaLambda" column = RPA + the zero-rung Lambda term alone
    const bool ward_legs = _pol_vtx->ladder_ward_legs();
    nda::array<ComplexType, 3> Pd_qmm;
    if (_pol_nu0_row.has_value()) {
      Pl_qmm = std::move(_pol_nu0_row.value());
      _pol_nu0_row.reset();
      if (ward_legs) {
        utils::check(_pol_nu0_dlam.has_value(),
                     "pol_ladder_eps_readout: legs = \"ward\" but the injection cached no "
                     "Delta P^Lambda row.");
        Pd_qmm = std::move(_pol_nu0_dlam.value());
        _pol_nu0_dlam.reset();
      }
      utils::check(Pl_qmm.shape(0) == nq,
                   "pol_ladder_eps_readout: cached inu = 0 ladder row has {} q rows, "
                   "expected {}.", Pl_qmm.shape(0), nq);
      app_log(2, "  [scGW-tilde L2] eps_M readout: reusing the injection's inu = 0 ladder "
                 "row (no second pair-space ladder pass).");
    } else if (not _pol_vtx->pol_interp_file().empty()) {
      // W-int-4 (notes/wannier_coarse_vertex_plan.md): the fine-mesh CONSUMER. Pi(q)_{MN} on THIS mesh's q list
      // in the frozen-point secondary frame comes from the file (a coarse run's <prefix>.pol_nu0.g<n>.h5 -- the
      // V0 gate -- or the offline Route-B interpolant written in the same layout) instead of the ladder solve;
      // the upfold + eps readout below run unchanged. Requires pol_vertex_isdf_points_file = the coarse points.
      utils::check(not ward_legs, "pol_ladder_eps_readout: pol_vertex_interp_file with legs = \"ward\" is not supported.");
      utils::check(not _pol_vtx->isdf_points_file().empty(),
                   "pol_ladder_eps_readout: pol_vertex_interp_file needs pol_vertex_isdf_points_file (the secondary "
                   "frame of the file is the coarse run's point set).");
      const std::string col = "Pi_" + _pol_vtx->pol_interp_col();
      nda::array<double, 2> qf;
      nda::array<ComplexType, 3> Pf;
      {
        h5::file f(_pol_vtx->pol_interp_file(), 'r');
        h5::group g(f);
        nda::h5_read(g, "q", qf);
        if (g.has_dataset("nu_half")) {   // an all-nu file: its half node 0 IS the inu = 0 node
          nda::array<ComplexType, 4> P4; nda::h5_read(g, col, P4);
          Pf = nda::array<ComplexType, 3>(P4.shape(1), P4.shape(2), P4.shape(3)); Pf = P4(0, all, all, all);
        } else {
          nda::h5_read(g, col, Pf);
        }
      }
      const long Nm_v = _pol_vtx->secondary_rank();
      utils::check(Pf.shape(1) == Nm_v and Pf.shape(2) == Nm_v and Pf.shape(0) == qf.shape(0),
                   "pol_ladder_eps_readout: {} in {} is {} x {} x {}, expected nq x {} x {} (the frozen-point rank).",
                   col, _pol_vtx->pol_interp_file(), Pf.shape(0), Pf.shape(1), Pf.shape(2), Nm_v, Nm_v);
      auto lat = MF->lattv();
      auto Q = MF->Qpts();
      Pl_qmm = nda::array<ComplexType, 3>(nq, Nm_v, Nm_v);
      for (long iq = 0; iq < nq; ++iq) {
        double qc[3];
        for (long i = 0; i < 3; ++i) { double v = 0.0; for (long j = 0; j < 3; ++j) v += lat(i, j) * Q(iq, j); qc[i] = v / (2.0 * M_PI); }
        long hit = -1;
        for (long jq = 0; jq < qf.shape(0) and hit < 0; ++jq) {
          bool same = true;
          for (long i = 0; i < 3 and same; ++i) { const double d = qf(jq, i) - qc[i]; same = std::abs(d - std::round(d)) < 1e-5; }
          if (same) hit = jq;
        }
        utils::check(hit >= 0, "pol_ladder_eps_readout: q = ({:.6f}, {:.6f}, {:.6f}) (crystal) of this mesh is absent "
                               "from {} (interpolate onto the full fine mesh).", qc[0], qc[1], qc[2], _pol_vtx->pol_interp_file());
        Pl_qmm(iq, all, all) = Pf(hit, all, all);
      }
      app_log(1, "  [W-int-4] eps readout consumes {} from {} ({} q matched on this mesh's q list; frozen points, "
                 "N_m = {}).", col, _pol_vtx->pol_interp_file(), nq, Nm_v);
    } else {
      Pl_qmm = _pol_vtx->eval_pol_ladder_nu0(mb_state, thc,
                                             ward_legs ? std::addressof(Pd_qmm) : nullptr);
    }
    auto const &tmap = _pol_vtx->secondary_transfer();            // (nq, Nm, Np)
    const long Nm = Pl_qmm.shape(1);
    utils::check(tmap.shape(0) == nq and tmap.shape(1) == Nm and tmap.shape(2) == Np,
                 "pol_ladder_eps_readout: transfer map shape mismatch.");
    // Tier 2 (D3): the dynamic-rung columns at inu = 0
    const bool dyn_rung = _pol_vtx->ladder_dynamic_rung();
    std::optional<vertex_t::dynbse_nu0_result> dres;
    if (dyn_rung) {
      ++_pol_dyn_calls;
      dres.emplace(_pol_vtx->eval_pol_dynbse_nu0(mb_state, thc, _pol_dyn_calls));
      if (_pol_vtx->ladder_dyn_all_nu()) dump_pol_dyn_all_nu(mb_state, thc, _pol_dyn_calls);
      // W-int-0: in Wannier mode eval_pol_dynbse_nu0 dumped Pi_loc(q) in the MLWF-pair basis (nab != Nm);
      // consumed OFFLINE for coarse->fine interpolation, not by this aux eps readout -- skip the aux shape
      // check + the dynamic-column upfold below. The static ladder eps columns are unaffected.
      if (not _pol_vtx->wannier())
        utils::check(dres->Pi_dyn.shape(0) == nq and dres->Pi_dyn.shape(1) == Nm,
                     "pol_ladder_eps_readout: dynamic-rung block shape mismatch.");
      else
        app_log(1, "  [scGW-tilde T2] Wannier mode: dynamic-rung eps columns SKIPPED (Pi_loc dumped for interpolation).");
      _pol_dyn_ritz = dres->ritz_max;
    }

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
    nda::array<ComplexType, 2> dPd(ward_legs ? Np : 0, ward_legs ? Np : 0);
    const long Nd = dyn_rung ? Np : 0;
    nda::array<ComplexType, 2> dPs(Nd, Nd), dP1(Nd, Nd), dPg(Nd, Nd), dPy(Nd, Nd);
    double eps_ds_qmin = -1.0, eps_dp_qmin = -1.0, eps_dg_qmin = -1.0, eps_dy_qmin = -1.0;
    nda::matrix<ComplexType> Am(Np, Np);
    nda::array<ComplexType, 1> chi_c(Np), buf(Np);
    // eps(q_i, i nu) cut state (2026-09-11): the per-transfer captures of its eps lambda
    long iq_cut = -1;
    double factor_cut = 0.0;
    nda::array<ComplexType, 2> Z_qPQ_cut;
    double eps_rpa_qmin = -1.0, eps_lad_qmin = -1.0, eps_dlm_qmin = -1.0, qmin_abs2 = 1e300;
    long iq_min = -1;
    // ---- DA D-7: the per-q Dyson-W change driven by P^lad ------------------------------
    // PURE OBSERVER, and the direct answer to "does the ladder act at small q?": for every
    // IBZ transfer (INCLUDING Gamma, which the eps_M head loop must skip) compare the
    // single-frequency i.nu = 0 dW built from P^RPA against the one built from
    // P^RPA + P^lad. ||dW_lad(q)||_F / ||dW_RPA(q)||_F is the relative screening change
    // that q's cell contributes; Delta eps_M(q) is the same statement in the head channel.
    const bool qnu_meter = (_vertex != nullptr and _vertex->ladder_qnu_meter());
    std::vector<double> dw_rel(size_t(nq), 0.0), dw_abs(size_t(nq), 0.0);
    std::vector<double> deps(size_t(nq), 0.0), eps_r(size_t(nq), -1.0);
    // hoisted out of the q loop: at production Np these are ~100 MB each per rank, and the
    // readout is already replicated-Np^2-heavy (Z_qPQ)
    nda::array<ComplexType, 2> W_rpa(qnu_meter ? Np : 0, qnu_meter ? Np : 0);
    nda::array<ComplexType, 2> W_lad(qnu_meter ? Np : 0, qnu_meter ? Np : 0);
    for (long iq = 0; iq < nq; ++iq) {
      auto qpts = MF->Qpts_ibz(iq);
      const double q_abs2 = qpts(0) * qpts(0) + qpts(1) * qpts(1) + qpts(2) * qpts(2);
      const bool is_gamma = (q_abs2 < 1e-12);
      if (is_gamma and not qnu_meter) continue;                   // Gamma: no head here
      // upfold: dP = t^dag Pl t
      auto tq = tmap(iq, all, all);
      nda::blas::gemm(Pl_qmm(iq, all, all), tq, tmpM);            // Pl . t   (Nm x Np)
      nda::array<ComplexType, 2> td(Np, Nm);
      for (long m = 0; m < Nm; ++m)
        for (long P = 0; P < Np; ++P) td(P, m) = std::conj(tq(m, P));
      nda::blas::gemm(td, tmpM, dP);                              // t^dag Pl t
      if (ward_legs) {                                            // t^dag Pd t
        nda::blas::gemm(Pd_qmm(iq, all, all), tq, tmpM);
        nda::blas::gemm(td, tmpM, dPd);
      }
      if (dyn_rung and not _pol_vtx->wannier()) {                // the four dynamic-rung blocks (aux; skipped in Wannier mode)
        auto up = [&](nda::array<ComplexType, 3> const &B, nda::array<ComplexType, 2> &out) {
          nda::blas::gemm(B(iq, all, all), tq, tmpM);
          nda::blas::gemm(td, tmpM, out);
        };
        up(dres->Pi_static, dPs);
        up(dres->Pi_dyn1, dP1);
        up(dres->Pi_gam1, dPg);
        up(dres->Pi_dyn, dPy);
        dP1 += dPs;                                               // static + one dynamic rung
      }
      const double factor = (q_abs2 / fpi) * MF->volume();
      chi_c = nda::conj(Chi_bar(iq, all));
      auto eps_of = [&](nda::array<ComplexType, 2> const *add, nda::array<ComplexType, 2> *dW_out) {
        // A = I - Z (P0 [+ add]);  dW = (A^{-1} - I) Z;  head contraction
        A() = Pi0_qPQ(iq, all, all);
        if (add != nullptr) A += *add;
        nda::array<ComplexType, 2> ZP(Np, Np);
        nda::blas::gemm(Z_qPQ(iq, all, all), A, ZP);
        Am() = ZP;
        Am() *= ComplexType(-1.0);
        for (long P = 0; P < Np; ++P) Am(P, P) += ComplexType(1.0);
        nda::inverse_in_place(Am);
        for (long P = 0; P < Np; ++P) Am(P, P) -= ComplexType(1.0);
        nda::blas::gemm(Am, Z_qPQ(iq, all, all), ZP);
        if (dW_out != nullptr) (*dW_out)() = ZP;
        nda::blas::gemv(ZP, chi_c, buf);
        const ComplexType eih = factor * nda::blas::dot(Chi_bar(iq, all), buf);
        return 1.0 / (1.0 + eih.real());
      };
      const double e_rpa = eps_of(nullptr, qnu_meter ? std::addressof(W_rpa) : nullptr);
      // The +ladder leg is evaluated by the SAME call in both modes -- the meter only asks
      // it to also hand back dW, so eps_lad (a physics readout consumed by the Q3 gates)
      // is bitwise what the historic path produced.
      const double e_lad = eps_of(std::addressof(dP), qnu_meter ? std::addressof(W_lad) : nullptr);
      // Tier 1.5: chi0_Lambda alone (the zero-rung Lambda term on top of RPA)
      const double e_dlm = ward_legs ? eps_of(std::addressof(dPd), nullptr) : -1.0;
      // Tier 2: the dynamic-rung columns (all rungs >= 1 on top of RPA)
      const double e_ds = dyn_rung ? eps_of(std::addressof(dPs), nullptr) : -1.0;
      const double e_dp = dyn_rung ? eps_of(std::addressof(dP1), nullptr) : -1.0;
      const double e_dg = dyn_rung ? eps_of(std::addressof(dPg), nullptr) : -1.0;
      const double e_dy = dyn_rung ? eps_of(std::addressof(dPy), nullptr) : -1.0;
      if (qnu_meter) {
        double dn = 0.0, rn = 0.0;
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            dn += std::norm(W_lad(P, Q) - W_rpa(P, Q));
            rn += std::norm(W_rpa(P, Q));
          }
        dw_abs[size_t(iq)] = std::sqrt(dn);
        dw_rel[size_t(iq)] = std::sqrt(dn) / std::max(std::sqrt(rn), 1e-300);
        deps[size_t(iq)] = is_gamma ? 0.0 : (e_lad - e_rpa);
        eps_r[size_t(iq)] = is_gamma ? -1.0 : e_rpa;
        if (is_gamma) continue;                                   // no head at Gamma
      }
      if (ward_legs)
        app_log(2, "  [scGW-tilde L2]   q {} (|q|^2 = {:.4e}): eps_M RPA = {:.6f}, "
                   "+DeltaLambda = {:.6f}, +ladder(Lambda legs) = {:.6f}", iq, q_abs2, e_rpa,
                e_dlm, e_lad);
      else if (dyn_rung)
        app_log(2, "  [scGW-tilde L2]   q {} (|q|^2 = {:.4e}): eps_M RPA = {:.6f}, +ladder(L2) = {:.6f}, "
                   "+static(dynbse driver) = {:.6f}, +static+Pi^C_dyn = {:.6f}, +Gamma1 = {:.6f}, +resummed = {:.6f}",
                iq, q_abs2, e_rpa, e_lad, e_ds, e_dp, e_dg, e_dy);
      else
        app_log(2, "  [scGW-tilde L2]   q {} (|q|^2 = {:.4e}): eps_M RPA = {:.6f}, "
                   "+ladder = {:.6f}", iq, q_abs2, e_rpa, e_lad);
      if (q_abs2 < qmin_abs2) {
        qmin_abs2 = q_abs2;
        eps_rpa_qmin = e_rpa;
        eps_lad_qmin = e_lad;
        eps_dlm_qmin = e_dlm;
        eps_ds_qmin = e_ds; eps_dp_qmin = e_dp; eps_dg_qmin = e_dg; eps_dy_qmin = e_dy;
        iq_min = iq;
      }
    }
    if (qnu_meter) {
      app_log(1, "\n  [DA D-7] per-q DYSON-W CHANGE driven by P^lad (i.nu = 0). "
                 "dW = (eps^-1 - I) Z; the\n"
                 "  [DA D-7] Gamma row carries no eps_M head by construction (marked -).");
      app_log(1, "  [DA D-7]   {:>4} {:>12} {:>14} {:>12} {:>12} {:>12}",
              "iq", "|q|^2", "||dW_lad||_F", "rel to dW", "eps_M(RPA)", "D eps_M");
      for (long iq = 0; iq < nq; ++iq) {
        auto qp2 = MF->Qpts_ibz(iq);
        const double q2 = qp2(0) * qp2(0) + qp2(1) * qp2(1) + qp2(2) * qp2(2);
        if (eps_r[size_t(iq)] < 0.0)
          app_log(1, "  [DA D-7]   {:>4} {:>12.5e} {:>14.6e} {:>12.5e} {:>12} {:>12}",
                  iq, q2, dw_abs[size_t(iq)], dw_rel[size_t(iq)], "-", "-");
        else
          app_log(1, "  [DA D-7]   {:>4} {:>12.5e} {:>14.6e} {:>12.5e} {:>12.6f} "
                     "{:>+12.6f}", iq, q2, dw_abs[size_t(iq)], dw_rel[size_t(iq)],
                  eps_r[size_t(iq)], deps[size_t(iq)]);
      }
      app_log(1, "");
    }
    if (iq_min >= 0) {
      app_log(1, "  [scGW-tilde L2] ladder eps_M readout (inu = 0, q_min = {}): "
                 "RPA = {:.6f}, +ladder = {:.6f} (Delta = {:+.6f}; gate L2-b watches "
                 "the DIRECTION)", iq_min, eps_rpa_qmin, eps_lad_qmin,
              eps_lad_qmin - eps_rpa_qmin);
      _pol_eps_rpa = eps_rpa_qmin;
      _pol_eps_ladder = eps_lad_qmin;
      _pol_eps_dlam = eps_dlm_qmin;
      if (ward_legs)
        app_log(1, "  [scGW-tilde T1.5] Tier-1.5 eps_M readout (inu = 0, q_min = {}): RPA = "
                   "{:.6f}, +DeltaLambda (chi0_Lambda alone) = {:.6f}, +ladder on Lambda legs "
                   "(eq 27 composite) = {:.6f}  [G-j targets at Si 4^3: ~4.8 / 5.4-5.6]",
                iq_min, eps_rpa_qmin, eps_dlm_qmin, eps_lad_qmin);
      if (dyn_rung) {
        _pol_eps_dyn_static = eps_ds_qmin; _pol_eps_dyn_pc = eps_dp_qmin;
        _pol_eps_dyn_gam1 = eps_dg_qmin; _pol_eps_dyn = eps_dy_qmin;
        app_log(1, "  [scGW-tilde T2] dynamic-rung eps_M readout (inu = 0, q_min = {}): RPA = {:.6f}, "
                   "+ladder(L2, resolvent 1 + Xh Kt) = {:.6f}, +static ladder (dynbse driver; must equal L2) = {:.6f}, "
                   "+static+Pi^C_dyn (one dynamic rung) = {:.6f}, +Gamma_1 (static-dressed one dynamic rung) "
                   "= {:.6f}, +RESUMMED dynamic-rung ladder = {:.6f}  [references at Si 4^3 q_min, C = [0,8): RPA 4.028, "
                   "static 4.699 (former 1 - Xh Kt resolvent 4.443), resummed 5.399, G0W0-class 5.747]; watchdog max "
                   "|Ritz(K_d L_s)| = {:.3f}, converged {}",
                iq_min, eps_rpa_qmin, eps_lad_qmin, eps_ds_qmin, eps_dp_qmin, eps_dg_qmin, eps_dy_qmin,
                dres->ritz_max, dres->all_converged);
      }
    }
    // ---- eps(q_i, i nu) cuts (2026-09-11, pol_eps_cut > 0; report-only) -------------------
    // Every column the readout evaluates, on EVERY PH-sym bosonic half node, at the selected
    // transfers: RPA (the gathered rows), +ladder (eval_pol_ladder_whalf: the same pair-space
    // ladder at all nodes), +DeltaLambda (legs = ward), and the loop's OWN eps^-1 head
    // (eps_inv_head_q -> i nu), which is the in-loop framework's eps_M (RPA scGW, or L3 with
    // the injection on). Node j <-> iw = nw_b/2 + j, nu_j = wn_b(iw) pi / beta (>= 0).
    if (_vertex != nullptr and _vertex->eps_cut_nq() > 0 and not _pol_cut_q.empty()) {
      ++_pol_cut_calls;
      const long nsel = long(_pol_cut_q.size());
      nda::array<ComplexType, 4> Pd_all;
      auto Pl_all = _pol_vtx->eval_pol_ladder_whalf(mb_state, thc, nullptr,
                                                    ward_legs ? std::addressof(Pd_all) : nullptr);
      const long nw_half = Pl_all.shape(0);
      utils::check(Pl_all.shape(1) == nq and Pl_all.shape(2) == Nm, "eps-cut: whalf ladder shape mismatch.");
      const bool have_loop = (eps_inv_head_q != nullptr);
      nda::array<ComplexType, 2> eloop_w(nw_half, std::max(nsel, 1l));
      if (have_loop) {
        const long nt_h = eps_inv_head_q->shape(0);
        nda::array<ComplexType, 2> et(nt_h, nsel);
        for (long s = 0; s < nsel; ++s)
          for (long it = 0; it < nt_h; ++it) et(it, s) = (*eps_inv_head_q)(it, _pol_cut_q[size_t(s)]);
        _ft->tau_to_w_PHsym(et, eloop_w);
      }
      const double beta = _ft->beta();
      auto wn_b = _ft->wn_mesh_b();
      const long nw_b = _ft->nw_b();
      // the dynamic-rung columns on every half node at the selected transfers (collective)
      std::optional<vertex_t::dynbse_cut_result> dcut;
      long n_dyn_nodes = 0;
      if (dyn_rung) {
        const long nn = (_vertex->eps_cut_dyn_nnu() > 0) ? std::min(_vertex->eps_cut_dyn_nnu(), nw_half) : nw_half;
        std::vector<long> hn(static_cast<size_t>(nn), 0l);
        for (long j = 0; j < nn; ++j) hn[size_t(j)] = j;
        dcut.emplace(_pol_vtx->eval_pol_dynbse_cut(mb_state, thc, hn, _pol_cut_q, _pol_cut_calls));
        n_dyn_nodes = nn;
      }
      if (thc.mpi()->comm.rank() == 0) {
        utils::check(_pol_pi_cut.has_value(), "eps-cut: rank 0 holds no gathered RPA rows.");
        auto const &Pc = _pol_pi_cut.value();
        utils::check(Pc.shape(0) == nsel and Pc.shape(1) == nw_half and Pc.shape(2) == Np,
                     "eps-cut: gathered rows shape mismatch.");
        _pol_eps_cut_qmin = nda::array<double, 2>(nw_half, 8);
        _pol_eps_cut_qmin() = -1.0;
        nda::array<ComplexType, 2> dPc[4];
        for (auto &A : dPc) A = nda::array<ComplexType, 2>(dyn_rung ? Np : 0, dyn_rung ? Np : 0);
        nda::array<ComplexType, 2> Arow(Np, Np), ZPc(Np, Np);
        auto eps_row = [&](nda::array<ComplexType, 2> const &Pi0, nda::array<ComplexType, 2> const *add) {
          Arow() = Pi0;
          if (add != nullptr) Arow += *add;
          nda::blas::gemm(Z_qPQ_cut, Arow, ZPc);
          Am() = ZPc;
          Am() *= ComplexType(-1.0);
          for (long P = 0; P < Np; ++P) Am(P, P) += ComplexType(1.0);
          nda::inverse_in_place(Am);
          for (long P = 0; P < Np; ++P) Am(P, P) -= ComplexType(1.0);
          nda::blas::gemm(Am, Z_qPQ_cut, ZPc);
          nda::blas::gemv(ZPc, chi_c, buf);
          const ComplexType eih = factor_cut * nda::blas::dot(Chi_bar(iq_cut, all), buf);
          return 1.0 / (1.0 + eih.real());
        };
        app_log(1, "\n  [eps-cut] call {}: eps_M(q_i, i nu_j) on the PH-sym bosonic half grid; columns: RPA, "
                   "+ladder (static L2, resolvent 1 + Xh Kt){}, loop-side (the loop's own eps^-1 head{}){}",
                _pol_cut_calls, ward_legs ? ", +DeltaLambda (chi0_Lambda alone)" : "",
                have_loop ? "" : ": absent",
                dyn_rung ? ", then the dynamic-rung columns: +static (dynbse driver), +static+Pi^C_dyn, +Gamma_1, +resummed" : "");
        app_log(1, "  [eps-cut]   {:>4} {:>4} {:>10} {:>3} {:>6} {:>12} {:>12} {:>12}{} {:>12}{}", "call", "iq", "|q|",
                "j", "wn", "nu(Ha)", "eps_rpa", "eps_lad", ward_legs ? "      eps_dlm" : "", "eps_loop",
                dyn_rung ? "      eps_dst      eps_dp1      eps_dg1      eps_dyn" : "");
        for (long s = 0; s < nsel; ++s) {
          iq_cut = _pol_cut_q[size_t(s)];
          auto qpc = MF->Qpts_ibz(iq_cut);
          const double q2c = qpc(0) * qpc(0) + qpc(1) * qpc(1) + qpc(2) * qpc(2);
          factor_cut = (q2c / fpi) * MF->volume();
          chi_c = nda::conj(Chi_bar(iq_cut, all));
          Z_qPQ_cut = Z_qPQ(iq_cut, all, all);
          auto tqc = tmap(iq_cut, all, all);
          nda::array<ComplexType, 2> tdc(Np, Nm);
          for (long m = 0; m < Nm; ++m)
            for (long P = 0; P < Np; ++P) tdc(P, m) = std::conj(tqc(m, P));
          for (long j = 0; j < nw_half; ++j) {
            nda::blas::gemm(Pl_all(j, iq_cut, all, all), tqc, tmpM);
            nda::blas::gemm(tdc, tmpM, dP);
            if (ward_legs) {
              nda::blas::gemm(Pd_all(j, iq_cut, all, all), tqc, tmpM);
              nda::blas::gemm(tdc, tmpM, dPd);
            }
            auto Pi0 = nda::array<ComplexType, 2>(Pc(s, j, all, all));
            const double e_rpa = eps_row(Pi0, nullptr);
            const double e_lad = eps_row(Pi0, std::addressof(dP));
            const double e_dlm = ward_legs ? eps_row(Pi0, std::addressof(dPd)) : -1.0;
            const double e_loop = have_loop ? 1.0 / (1.0 + eloop_w(j, s).real()) : -1.0;
            double e_d[4] = {-1.0, -1.0, -1.0, -1.0};
            if (dyn_rung and j < n_dyn_nodes) {
              for (int c = 0; c < 4; ++c) {
                nda::blas::gemm(dcut->Pi(c, j, iq_cut, all, all), tqc, tmpM);
                nda::blas::gemm(tdc, tmpM, dPc[c]);
                e_d[c] = eps_row(Pi0, std::addressof(dPc[c]));
              }
            }
            if (s == 0) {
              _pol_eps_cut_qmin(j, 0) = e_rpa; _pol_eps_cut_qmin(j, 1) = e_lad;
              _pol_eps_cut_qmin(j, 2) = e_dlm; _pol_eps_cut_qmin(j, 3) = e_loop;
              for (int c = 0; c < 4; ++c) _pol_eps_cut_qmin(j, 4 + c) = e_d[c];
            }
            const long wn = wn_b(nw_b / 2 + j);
            std::string dyn_cols;
            if (dyn_rung)
            {
              char buf[96];
              std::snprintf(buf, sizeof(buf), " %12.6f %12.6f %12.6f %12.6f", e_d[0], e_d[1], e_d[2], e_d[3]);
              dyn_cols = buf;
            }
            if (ward_legs)
              app_log(1, "  [eps-cut]   {:>4} {:>4} {:>10.6f} {:>3} {:>6} {:>12.6e} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f}{}",
                      _pol_cut_calls, iq_cut, std::sqrt(q2c), j, wn, double(wn) * M_PI / beta, e_rpa, e_lad, e_dlm, e_loop,
                      dyn_cols);
            else
              app_log(1, "  [eps-cut]   {:>4} {:>4} {:>10.6f} {:>3} {:>6} {:>12.6e} {:>12.6f} {:>12.6f} {:>12.6f}{}",
                      _pol_cut_calls, iq_cut, std::sqrt(q2c), j, wn, double(wn) * M_PI / beta, e_rpa, e_lad, e_loop,
                      dyn_cols);
          }
        }
        app_log(1, "");
      }
      _pol_pi_cut.reset();
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
      if (_vertex != nullptr and _vertex->active() and _vertex->skip_pi_c()) {
        app_log(1, "  [ISDF-Vertex] TEST switch set_skip_pi_c: the Pi^C cut is OMITTED (P = RPA); Sigma^C alone (a gate reference, not a theory).");
        return;
      }
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
      // LFF-Sigma, pol_vertex_sigma_bub = "full": the vertex's Pi_0 = THIS RPA Pi folded to the frozen frame
      if (_vertex->sigma_lff_enabled() and _vertex->sigma_lff_bub() == "full") fold_rpa_pi_secondary(dPi_rpa, thc);
      if (_vertex->eps_cut_nq() > 0) gather_cut_rows(thc, dPi_rpa);   // eps(q_i, i nu) cuts (report-only)
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
                                        comm_t &comm, mf::MF &mf,
                                        eps_fit::eps_inf_fit_t const *fit) {
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
      // P25 / G32 (eps_inf_fit = true): the small-q fit next to the stored head -- the constant term,
      // the coefficients c_k of eps_M(q) = sum_k c_k |q|^{2k}, the |q| and eps_M(q) used, the RMS misfit.
      if (fit != nullptr and fit->ok) {
        const long ncf = static_cast<long>(fit->coeffs.size()), nqa = static_cast<long>(fit->q_used.size());
        nda::array<double, 1> cf(ncf), qa(nqa), eu(nqa);
        for (long i = 0; i < cf.shape(0); ++i) cf(i) = fit->coeffs[size_t(i)];
        for (long i = 0; i < qa.shape(0); ++i) qa(i) = fit->q_used[size_t(i)];
        for (long i = 0; i < eu.shape(0); ++i) eu(i) = fit->eps_used[size_t(i)];
        h5::h5_write(iter_grp, "epsilon_inf_fit", fit->eps_inf);
        h5::h5_write(iter_grp, "epsilon_inf_fit_residual", fit->residual);
        nda::h5_write(iter_grp, "epsilon_inf_fit_coeffs", cf, false);
        nda::h5_write(iter_grp, "epsilon_inf_fit_qabs", qa, false);
        nda::h5_write(iter_grp, "epsilon_inf_fit_eps", eu, false);
      }
    }
    comm.barrier();
  }

  // P25 / G32: eps_M(q) = 1 / (1 + Re[eps^-1_{00}(q, i nu = 0) - 1]) at every IBZ transfer from the
  // q-resolved tau head, then eps_fit::fit_eps_inf on the smallest nonzero Cartesian |q|. The |q|-only
  // fit is exact for a cubic cell; for a lower symmetry eps(q -> 0) is direction-dependent and the fit
  // averages over the directions the smallest IBZ transfers happen to sample (stated in the plan).
  eps_fit::eps_inf_fit_t scr_coulomb_t::eval_eps_inf_fit(const nda::ArrayOfRank<2> auto &eps_inv_head_tq,
                                                          mf::MF &mf) const {
    const long nq = eps_inv_head_tq.shape(1);
    utils::check(nq == mf.nqpts_ibz(), "eval_eps_inf_fit: eps_inv_head_tq has {} q rows, expected nqpts_ibz = {}.",
                 nq, mf.nqpts_ibz());
    long nw_half = (_ft->nw_b() % 2 == 0) ? _ft->nw_b() / 2 : _ft->nw_b() / 2 + 1;
    nda::array<ComplexType, 2> eih_wq(nw_half, nq);
    _ft->tau_to_w_PHsym(eps_inv_head_tq, eih_wq);   // i nu = 0 = index 0 of the PH-sym bosonic half grid
    std::vector<double> q_abs(static_cast<size_t>(nq)), eps_q(static_cast<size_t>(nq));
    for (long iq = 0; iq < nq; ++iq) {
      auto qp = mf.Qpts_ibz(iq);                     // Cartesian, bohr^-1, 2 pi included (bz_symmetry.hpp)
      q_abs[size_t(iq)] = std::sqrt(qp(0) * qp(0) + qp(1) * qp(1) + qp(2) * qp(2));
      eps_q[size_t(iq)] = 1.0 / (1.0 + eih_wq(0, iq).real());
    }
    return eps_fit::fit_eps_inf(q_abs, eps_q, _eps_inf_fit_npts);
  }


  // template instantiations
  using Arr4D = nda::array<ComplexType, 4>;
  using Arr = nda::array<ComplexType, 5>;
  using Arrv = nda::array_view<ComplexType, 5>;
  using Arrv2 = nda::array_view<ComplexType, 5, nda::C_layout>;

  // W-int-4f / LFF-Sigma: the interp file's column "Pi_<col>" on this mesh's q list at the PH-sym half nodes, in the
  // frozen-point secondary frame -- ONE read shared by the W-Dyson injection (inject_pol_ladder) and the Sigma vertex
  // (build_sigma_lff); every consistency check of the consumer (node list, beta, N_m, q matching) lives here.
  nda::array<ComplexType, 4> scr_coulomb_t::read_pol_interp_column(std::string const &colname, long nq_g, long nw_h_ft,
                                                                    nda::array<long, 1> const &nu_half, THC_ERI auto &thc) {
    decltype(nda::range::all) all;
    utils::check(_pol_vtx != nullptr and not _pol_vtx->pol_interp_file().empty(),
                 "read_pol_interp_column: no pol_vertex_interp_file.");
    auto MF = thc.MF();
    auto lat = MF->lattv();
    auto Q = MF->Qpts();
    auto q_crys = [&](long iq, double *qc) {
      for (long i = 0; i < 3; ++i) { double v = 0.0; for (long j = 0; j < 3; ++j) v += lat(i, j) * Q(iq, j); qc[i] = v / (2.0 * M_PI); }
    };
    const std::string col = "Pi_" + colname;
    nda::array<double, 2> qf;
    nda::array<ComplexType, 4> Pf;
    nda::array<long, 1> nu_f;
    double beta_f = -1.0;
    {
      h5::file f(_pol_vtx->pol_interp_file(), 'r');
      h5::group g(f);
      utils::check(g.has_dataset("nu_half"),
                   "inject_pol_ladder: {} carries no nu_half axis -- an inu = 0-only dump cannot feed the W-Dyson "
                   "(dump the coarse run's whalf / all-nu columns).", _pol_vtx->pol_interp_file());
      nda::h5_read(g, "q", qf);
      nda::h5_read(g, col, Pf);
      nda::h5_read(g, "nu_half", nu_f);
      if (g.has_dataset("beta")) h5::h5_read(g, "beta", beta_f);
    }
    utils::check(long(nu_f.size()) == nw_h_ft, "inject_pol_ladder: {} has {} half nodes, this run's IAFT has {} -- the "
                 "coarse and fine runs must share beta / basis / precision.", _pol_vtx->pol_interp_file(), nu_f.size(), nw_h_ft);
    for (long j = 0; j < nw_h_ft; ++j)
      utils::check(nu_f(j) == nu_half(j), "inject_pol_ladder: half node {} is Matsubara index {} in the file, {} here.",
                   j, nu_f(j), nu_half(j));
    if (beta_f > 0.0) utils::check(std::abs(beta_f - _ft->beta()) < 1e-8, "inject_pol_ladder: beta {} (file) != {} (run).", beta_f, _ft->beta());
    const long Nm_v = _pol_vtx->secondary_rank();
    utils::check(Pf.shape(0) == nw_h_ft and Pf.shape(2) == Nm_v and Pf.shape(3) == Nm_v and Pf.shape(1) == qf.shape(0),
                 "inject_pol_ladder: {} in {} is {} x {} x {} x {}, expected {} x nq x {} x {}.", col, _pol_vtx->pol_interp_file(),
                 Pf.shape(0), Pf.shape(1), Pf.shape(2), Pf.shape(3), nw_h_ft, Nm_v, Nm_v);
    nda::array<ComplexType, 4> Pl(nw_h_ft, nq_g, Nm_v, Nm_v);
    for (long iq = 0; iq < nq_g; ++iq) {
      double qc[3]; q_crys(iq, qc);
      long hit = -1;
      for (long jq = 0; jq < qf.shape(0) and hit < 0; ++jq) {
        bool same = true;
        for (long i = 0; i < 3 and same; ++i) { const double d = qf(jq, i) - qc[i]; same = std::abs(d - std::round(d)) < 1e-5; }
        if (same) hit = jq;
      }
      utils::check(hit >= 0, "inject_pol_ladder: q = ({:.6f}, {:.6f}, {:.6f}) (crystal) of this mesh is absent from {}.",
                   qc[0], qc[1], qc[2], _pol_vtx->pol_interp_file());
      Pl(all, iq, all, all) = Pf(all, hit, all, all);
    }
    return Pl;
  }

  namespace sigma_lff_detail {
    // the PH mirror of the bosonic mesh (wn(mirror(l)) = -wn(l)), the map fold_dW_distributed unfolds the half mesh with
    inline nda::array<long, 1> bosonic_mirror(const imag_axes_ft::IAFT &ft) {
      const long nw_b = ft.nw_b();
      auto wb = ft.wn_mesh_b();
      nda::array<long, 1> mirror(nw_b);
      for (long l = 0; l < nw_b; ++l) {
        mirror(l) = -1;
        for (long m = 0; m < nw_b; ++m) if (wb(m) == -wb(l)) { mirror(l) = m; break; }
        utils::check(mirror(l) >= 0, "sigma_lff: the bosonic Matsubara mesh is not PH-symmetric (no mirror of node {}).", l);
      }
      return mirror;
    }
  }

  // LFF-Sigma, pol_vertex_sigma_bub = "full": fold the loop's RPA Pi(tau) (the (t, q, P, Q) darray of eval_Pi_qdep,
  // BEFORE any injection) to the frozen secondary frame at the PH-sym half nodes, with the same distributed fold the
  // dynamic W-bar cache uses (a (q, t, P, Q) copy first: that routine's layout). Replicated (nq, nw_half, N_m, N_m).
  template<nda::MemoryArrayOfRank<4> Array_t, typename communicator_t>
  void scr_coulomb_t::fold_rpa_pi_secondary(memory::darray_t<Array_t, communicator_t> &dPi_tqPQ, THC_ERI auto &thc) {
    decltype(nda::range::all) all;
    using math::nda::make_distributed_array;
    auto mpi = thc.mpi();
    auto MF = thc.MF();
    const long nq = MF->nqpts_ibz(), Np = thc.Np(), Nm = _pol_vtx->secondary_rank();
    const long nw_b = _ft->nw_b(), nw_h = (nw_b % 2 == 0) ? nw_b / 2 : nw_b / 2 + 1;
    auto gs = dPi_tqPQ.global_shape();
    const long nt_h = gs[0];
    utils::check(gs[1] == nq and gs[2] == Np and gs[3] == Np,
                 "fold_rpa_pi_secondary: unexpected Pi global shape ({}, {}, {}, {}).", gs[0], gs[1], gs[2], gs[3]);
    auto pg = dPi_tqPQ.grid();
    auto bs = dPi_tqPQ.block_size();
    auto dPi_qtPQ = make_distributed_array<nda::array<ComplexType, 4>>(
        mpi->comm, {pg[1], pg[0], pg[2], pg[3]}, {gs[1], gs[0], gs[2], gs[3]}, {bs[1], bs[0], bs[2], bs[3]});
    {
      auto A = dPi_tqPQ.local();
      auto B = dPi_qtPQ.local();
      const long ntl = dPi_tqPQ.local_shape()[0], nql = dPi_tqPQ.local_shape()[1];
      for (long iq = 0; iq < nql; ++iq)
        for (long it = 0; it < ntl; ++it) B(iq, it, nda::ellipsis{}) = A(it, iq, nda::ellipsis{});
    }
    auto mirror = sigma_lff_detail::bosonic_mirror(*_ft);
    nda::array<ComplexType, 4> Pb(nq, nw_b, Nm, Nm);
    auto no_head = [](auto &&, long, nda::range const &, nda::range const &) {};
    auto xform = [&](nda::MemoryArrayOfRank<3> auto &&Xt, nda::MemoryArrayOfRank<3> auto &&Xw) { _ft->tau_to_w_PHsym(Xt, Xw); };
    vertex_secondary_detail::fold_dW_distributed(dPi_qtPQ, _pol_vtx->secondary_transfer(), nq, nt_h, Np, Nm, nw_b, nw_h,
                                                 mirror, 0, false, no_head, xform, Pb, mpi->comm);
    _sig_pi0_qwmm.emplace(nda::array<ComplexType, 4>(nq, nw_h, Nm, Nm));
    for (long iq = 0; iq < nq; ++iq)
      for (long j = 0; j < nw_h; ++j) _sig_pi0_qwmm.value()(iq, j, all, all) = Pb(iq, nw_b / 2 + j, all, all);
    app_log(1, "  [LFF-Sigma] pol_vertex_sigma_bub = \"full\": the loop's RPA Pi folded to the frozen secondary frame, "
               "(nq, nw_half, N_m) = ({}, {}, {}).", nq, nw_h, Nm);
  }

  // ---------------------------------------------------------------------------------------------------------------
  // LFF-Sigma (Route 1, notes/lff_aux_plan.md 2026-09-19): the local-field-factor vertex in the SELF-ENERGY.
  // With a vertex that acts on the aux (density) index only, Hedin's Sigma = i G W Gamma collapses to Sigma = G W~ with
  // W~ = W Gamma_eff (Del Sole, Reining, Godby 1994); here Gamma_eff = Pi_0^-1 (Pi_0 + dPi) in the frozen secondary
  // frame, so that P = Pi_0 Gamma_eff reproduces the injected polarization by construction:
  //     Gamma_eff - 1 = Pi_0^-1 dPi,       dW~(q, i nu) = scale x Herm[ W-bar(q, i nu) (Gamma_eff - 1) ],
  // W-bar = t (Z + dW) t^dag = THIS iteration's full W folded to the frame at every PH-sym half node (no Gamma-head
  // insertion: the loop's dW(Gamma) is head-free and the correction's own q -> 0 head is extracted below exactly as the
  // loop extracts eps_inv_head from dW). The Hermitization is a CONVENTION (W Gamma is not Hermitian, Sigma must be;
  // the Gamma^1/2 W Gamma^1/2 split is the alternative -- neither is derived from the ladder's leg structure yet).
  // dPi = the interp file's column (the injected one by default); Pi_0 = the file's window bubble ("Pi_bub",
  // pol_vertex_sigma_bub = "window") or the loop's RPA Pi folded to the frame ("full"). The correction is upfolded to
  // the THC frame with the injection's adjoint map, transformed nu -> tau, its head extracted with the loop's own
  // extractor, and published as (mb_state.dWsig_qtPQ, eps_inv_head_sig) for a SECOND GW contraction in gw_t::evaluate,
  // on top of Sigma^GW. W, the kernel caches, the readout and every other consumer of dW are untouched; scale = 0 is
  // bit-identical to the run without the knob. Cost: N_m-class algebra per (q, nu) + one extra Sigma contraction.
  // ---- LFF-Sigma Route 2 (L-6, notes/lff_aux_plan.md): the pair-resolved static-ladder vertex in Sigma -----------
  // The readout instance (_pol_vtx) carries the frozen secondary frame, W-bar_0 (build_w0 at the pure-RPA point of this
  // update) and, on demand, the W-bar cache; the knobs live on the knob carrier (_vertex). The result is the C-window
  // block dSigma(tau) in band labels, published for gw_t::evaluate (added to Sigma after Sigma^GW).
  void scr_coulomb_t::build_sigma_pair(MBState &mb_state, THC_ERI auto &thc) {
    utils::check(_vertex != nullptr and _vertex->pol_vertex_active() and not _vertex->active(),
                 "build_sigma_pair: pol_vertex_sigma = \"pair\" needs the ladder machinery (pol_vertex = \"ladder\", a non-empty "
                 "window) and an INACTIVE vertex_type.");
    ensure_pol_vertex(thc);
    utils::check(_pol_vtx != nullptr, "build_sigma_pair: no readout instance.");
    vertex_t::sigma_pair_opts o = sigma_pair_opts_of(*_vertex, mb_state);
    const bool dyn = _vertex->sigma_pair_dynamic();
    nda::array<ComplexType, 4> Pchk;
    if (_vertex->sigma_pair_diag() and not dyn) {
      // the P side's own object from the same amplitudes (gate G1): dumped with the run prefix for the test / offline check
      const long nq = thc.MF()->nqpts_ibz(), Nm = _pol_vtx->secondary_rank(), nw_b = _ft->nw_b();
      Pchk = nda::array<ComplexType, 4>(nw_b, nq, Nm, Nm);
      o.Pi_check = std::addressof(Pchk);
    }
    nda::array<ComplexType, 5> dS;
    vertex_t::sigma_pair_meter met;
    const long w0 = _pol_vtx->band_window().first(), ncw = long(_pol_vtx->band_window().size());
    nda::array<double, 1> tau_f(_ft->tau_mesh_f());
    if (not _vertex->sigma_interp_file().empty()) {
      // P16: the fine consumer -- the coarse run's Wannier-frame dSigma interpolated onto this mesh's window (no solve)
      ladder_meter::watch iw;
      utils::check(not _vertex->sigma_interp_projector().empty(), "build_sigma_pair: pol_vertex_sigma_interp_file needs pol_vertex_sigma_interp_projector (this mesh's Wannier file).");
      projector_t proj_f(*thc.MF(), _vertex->sigma_interp_projector());
      dS = vertex_sigma_interp::interpolate_sigma_pair(*thc.MF(), proj_f, _vertex->sigma_interp_file(), tau_f, _ft->beta(), w0, ncw, thc.mpi()->comm);
      for (auto const &v : dS) met.dsig_max = std::max(met.dsig_max, std::abs(v));
      met.dsig_herm = 0.0; met.ks_herm = 0.0; met.t_total = iw.lap();
    } else if (dyn) _pol_vtx->eval_sigma_pair_dyn(mb_state, thc, o, dS, &met);   // L-7: the dynamic-rung ladder in Sigma
    else _pol_vtx->eval_sigma_pair(mb_state, thc, o, dS, &met);
    if (not _vertex->sigma_interp_dump().empty() and _vertex->sigma_interp_file().empty()) {
      // P16: the coarse producer -- dSigma downfolded to the Wannier frame with its k list and R grid, for a finer mesh
      projector_t proj_c(*thc.MF(), _vertex->sigma_interp_dump());
      const std::string fn = mb_state.coqui_prefix + ".sigpair_wan.h5";
      vertex_sigma_interp::dump_sigma_pair_wannier(*thc.MF(), proj_c, dS, w0, tau_f, _ft->beta(), thc.mpi()->comm, fn, o.col, o.outer);
    }
    if (_vertex->sigma_pair_diag() and thc.mpi()->comm.root()) {
      const std::string fn = mb_state.coqui_prefix + ".sigpair.h5";
      h5::file f(fn, 'w');
      h5::group g(f);
      if (not dyn) {
        nda::h5_write(g, "Pi_check", Pchk);
        nda::h5_write(g, "nu_spec", met.nu_spec);
      }
      nda::h5_write(g, "dSigma_tskab", dS);
      h5::h5_write(g, "side", o.side);
      h5::h5_write(g, "col", o.col);
      h5::h5_write(g, "outer", o.outer);
      app_log(1, "  [LFF-Sigma pair] diagnostics written to {} (Pi_check (nw_b, nq, N_m, N_m), nu_spec, dSigma_tskab)", fn);
    }
    mb_state.dSigma_pair_tskab.emplace(std::move(dS));
    mb_state.sigma_pair_window = {_pol_vtx->band_window().first(), long(_pol_vtx->band_window().size())};
    _sig_pair_meter = {met.dsig_max, met.dsig_herm, met.ks_herm, met.t_total};
  }

  void scr_coulomb_t::build_sigma_lff(MBState &mb_state, THC_ERI auto &thc, std::array<long, 4> t_pgrid, std::array<long, 4> t_bsize) {
    decltype(nda::range::all) all;
    using math::nda::make_distributed_array;
    utils::check(_vertex != nullptr and _vertex->pol_vertex_active() and not _vertex->active(),
                 "build_sigma_lff: pol_vertex_sigma needs the ladder machinery (pol_vertex = \"ladder\") and an INACTIVE vertex_type.");
    ensure_pol_vertex(thc);
    utils::check(not _pol_vtx->pol_interp_file().empty() and not _pol_vtx->isdf_points_file().empty(),
                 "build_sigma_lff: pol_vertex_sigma = \"lff\" needs pol_vertex_interp_file (dPi and the window bubble) and "
                 "pol_vertex_isdf_points_file (its frozen frame).");
    utils::check(mb_state.dW_qtPQ.has_value(), "build_sigma_lff: dW_qtPQ is not stored (must run at the update_w tail).");
    ladder_meter::watch sw;
    auto mpi = thc.mpi();
    auto MF = thc.MF();
    const long nq = MF->nqpts_ibz(), Np = thc.Np(), Nm = _pol_vtx->secondary_rank();
    const long nw_b = _ft->nw_b(), nw_h = (nw_b % 2 == 0) ? nw_b / 2 : nw_b / 2 + 1;
    const long nt_h = long(mb_state.dW_qtPQ.value().global_shape()[1]);
    nda::array<long, 1> nu_half(nw_h);
    { auto wb = _ft->wn_mesh_b(); for (long j = 0; j < nw_h; ++j) nu_half(j) = long(wb(nw_b / 2 + j)); }
    const std::string col = _vertex->sigma_lff_col().empty() ? _pol_vtx->pol_interp_col() : _vertex->sigma_lff_col();
    const bool full_bub = (_vertex->sigma_lff_bub() == "full");
    const double scale = _vertex->sigma_lff_scale(), tol = _vertex->sigma_lff_pinv_tol(), hscale = _vertex->sigma_lff_head_scale();
    auto const &tmap = _pol_vtx->secondary_transfer();               // (nq, Nm, Np)
    utils::check(tmap.shape(0) == nq and tmap.shape(1) == Nm and tmap.shape(2) == Np,
                 "build_sigma_lff: secondary transfer is {}x{}x{}, expected {}x{}x{}.", tmap.shape(0), tmap.shape(1), tmap.shape(2), nq, Nm, Np);

    // ---- 1. dPi and Pi_0 in the frozen frame at the half nodes: (nw_h, nq, Nm, Nm), replicated ---------------------
    auto dP = read_pol_interp_column(col, nq, nw_h, nu_half, thc);
    nda::array<ComplexType, 4> P0;
    if (full_bub) {
      utils::check(_sig_pi0_qwmm.has_value(), "build_sigma_lff: the folded RPA bubble is missing (eval_Pi_qdep stash).");
      P0 = nda::array<ComplexType, 4>(nw_h, nq, Nm, Nm);
      for (long j = 0; j < nw_h; ++j)
        for (long iq = 0; iq < nq; ++iq) P0(j, iq, all, all) = _sig_pi0_qwmm.value()(iq, j, all, all);
    }
    auto Bd = read_pol_interp_column("bub", nq, nw_h, nu_half, thc);   // the dump's window bubble (LFF L-0: always written)
    if (not full_bub) P0 = Bd;
    // normalization / sign diagnostics of the local vertex at (q_1, nu_0) and (Gamma, nu_0): the strong-subspace scalar
    // tr(P dPi P) / tr(P B P) (the Si analysis: +0.50 at q_min, nu = 0) and, for "full", tr(Pib) / tr(B_dump) (Pib must
    // be >= the window bubble in magnitude: more transitions; a normalization mismatch shows up here first)
    double v_q1 = 0.0, v_q0 = 0.0, pb_q1 = 0.0, pb_q0 = 0.0, v_tail = 0.0;
    {
      for (long iq : {std::min(1l, nq - 1), 0l, -1l}) {
        const long jn = (iq < 0) ? nw_h - 1 : 0;            // -1: the tail node at q_1
        if (iq < 0) iq = std::min(1l, nq - 1);
        nda::matrix<ComplexType> Bm(Nm, Nm);
        for (long a = 0; a < Nm; ++a)
          for (long b = 0; b < Nm; ++b) Bm(a, b) = 0.5 * (P0(jn, iq, a, b) + std::conj(P0(jn, iq, b, a)));
        auto [lam, V] = nda::linalg::eigenelements(Bm);
        double lmax = 0.0;
        for (long a = 0; a < Nm; ++a) lmax = std::max(lmax, std::abs(lam(a)));
        double num = 0.0, den = 0.0, trb = 0.0, trd = 0.0;
        for (long b = 0; b < Nm; ++b) {
          if (std::abs(lam(b)) <= tol * lmax) continue;
          // <v_b| dPi |v_b> and lam_b on the strong subspace
          ComplexType d = 0.0;
          for (long a = 0; a < Nm; ++a)
            for (long c = 0; c < Nm; ++c) d += std::conj(V(a, b)) * dP(jn, iq, a, c) * V(c, b);
          num += d.real(); den += lam(b);
        }
        for (long a = 0; a < Nm; ++a) { trb += Bd(jn, iq, a, a).real(); trd += P0(jn, iq, a, a).real(); }
        const double v = num / (std::abs(den) > 1e-300 ? den : 1e-300), pb = trd / (std::abs(trb) > 1e-300 ? trb : 1e-300);
        if (jn > 0) v_tail = v; else if (iq == 0) { v_q0 = v; pb_q0 = pb; } else { v_q1 = v; pb_q1 = pb; }
      }
    }
    const double t_read = sw.lap();

    // ---- 2. the frame metric M(q) = t t^dag (t is NOT an isometry), its inverse, and Z t^dag (replicated per q) -----------
    // DERIVED (2026-09-19): with the code's maps -- kernels fold as W-bar = t W t^dag, polarizations upfold as
    // Pi = t^dag Pi_S t -- the primary-frame vertex operator that reproduces P = Pi_0 + t^dag dPi_S t on the window
    // subspace is Gamma - 1 = t^dag G1 t with G1 = M^-1 B^-1 dPi_S (window bubble B = the dump's Pi_bub, primary
    // t^dag B t) or G1 = Pib^-1 M dPi_S (full bubble: Pib = t Pi_0 t^dag, the projector t^dag M^-1 t on Pi_0). Folding
    // W to W-bar and upfolding the product would insert a spurious t^dag t on the left of W (measured: 50x on the head),
    // so W stays in the PRIMARY frame: dW~ = scale x Herm[ (Z + dW) t^dag G1 t ], built block-wise below.
    nda::array<ComplexType, 3> Mq(nq, Nm, Nm), Minv(nq, Nm, Nm), Zt(nq, Np, Nm);
    double mcond_max = 0.0;
    {
      nda::matrix<ComplexType> Mm(Nm, Nm), VL(Nm, Nm);
      for (long iq = 0; iq < nq; ++iq) {
        auto t_q = tmap(iq, all, all);                                   // (Nm x Np)
        nda::blas::gemm(t_q, nda::dagger(t_q), Mm);                     // M = t t^dag
        Mq(iq, all, all) = Mm;
        auto [lam, V] = nda::linalg::eigenelements(Mm);
        double lmax = 0.0, lmin = 1e300;
        for (long a = 0; a < Nm; ++a) { lmax = std::max(lmax, std::abs(lam(a))); lmin = std::min(lmin, std::abs(lam(a))); }
        mcond_max = std::max(mcond_max, lmax / std::max(lmin, 1e-300));
        for (long b = 0; b < Nm; ++b)
          for (long a = 0; a < Nm; ++a) VL(a, b) = (std::abs(lam(b)) > 1e-12 * lmax) ? V(a, b) / lam(b) : ComplexType(0.0);
        nda::blas::gemm(VL, nda::dagger(V), Mm);
        Minv(iq, all, all) = Mm;
        nda::array<ComplexType, 2> Zq = thc.Z(int(iq));                 // collective: every rank, every q
        nda::blas::gemm(Zq, nda::dagger(t_q), Zt(iq, all, all));        // (Np x Np)(Np x Nm)
      }
    }
    const double t_fold = sw.lap();

    // ---- 3. G1(q, nu_j) = M^-1 B^-1 dPi (window) | Pib^-1 M dPi (full), B / Pib regularized (strong modes) -----------
    nda::array<ComplexType, 4> G1(nw_h, nq, Nm, Nm);
    G1() = ComplexType(0.0);
    double g_max = 0.0, g_q1 = 0.0, g_q0 = 0.0;
    long ncut_max = 0;
    {
      nda::matrix<ComplexType> Bm(Nm, Nm), VL(Nm, Nm), Binv(Nm, Nm), dPm(Nm, Nm), T1(Nm, Nm), T2(Nm, Nm);
      for (long iq = 0; iq < nq; ++iq) {
        if (iq % mpi->comm.size() != mpi->comm.rank()) continue;
        for (long j = 0; j < nw_h; ++j) {
          for (long a = 0; a < Nm; ++a)
            for (long b = 0; b < Nm; ++b) Bm(a, b) = 0.5 * (P0(j, iq, a, b) + std::conj(P0(j, iq, b, a)));
          auto [lam, V] = nda::linalg::eigenelements(Bm);
          double lmax = 0.0;
          for (long a = 0; a < Nm; ++a) lmax = std::max(lmax, std::abs(lam(a)));
          long ncut = 0;
          for (long b = 0; b < Nm; ++b) {
            const bool keep = std::abs(lam(b)) > tol * lmax;
            if (not keep) ++ncut;
            for (long a = 0; a < Nm; ++a) VL(a, b) = keep ? V(a, b) / lam(b) : ComplexType(0.0);
          }
          ncut_max = std::max(ncut_max, ncut);
          nda::blas::gemm(VL, nda::dagger(V), Binv);                     // Pi_0^-1 on the strong subspace
          dPm = dP(j, iq, all, all);
          if (full_bub) {
            nda::blas::gemm(Mq(iq, all, all), dPm, T1);                  // M dPi
            nda::blas::gemm(Binv, T1, T2);                               // Pib^-1 M dPi
          } else {
            nda::blas::gemm(Binv, dPm, T1);                              // B^-1 dPi
            nda::blas::gemm(Minv(iq, all, all), T1, T2);                 // M^-1 B^-1 dPi
          }
          double nG = 0.0;
          for (long a = 0; a < Nm; ++a)
            for (long b = 0; b < Nm; ++b) { G1(j, iq, a, b) = T2(a, b); nG += std::norm(T2(a, b)); }
          const double g = std::sqrt(nG / double(Nm));
          g_max = std::max(g_max, g);
          if (j == 0 and iq == std::min(1l, nq - 1)) g_q1 = g;
          if (j == 0 and iq == 0) g_q0 = g;
        }
      }
    }
    mpi->comm.all_reduce_in_place_n(G1.data(), G1.size(), std::plus<>{});
    g_max = mpi->comm.all_reduce_value(g_max, boost::mpi3::max<>{});
    g_q1 = mpi->comm.all_reduce_value(g_q1, boost::mpi3::max<>{});
    g_q0 = mpi->comm.all_reduce_value(g_q0, boost::mpi3::max<>{});
    ncut_max = mpi->comm.all_reduce_value(ncut_max, boost::mpi3::max<>{});
    const double t_alg = sw.lap();

    // ---- 4. per q: Y(nu_j) = (Z + dW(i nu_j)) t^dag (Np x Nm, replicated), my (P, Q) block of dW~, nu -> tau -------------
    auto dWs_tqPQ = make_distributed_array<nda::array<ComplexType, 4>>(mpi->comm, t_pgrid, {nt_h, nq, Np, Np}, t_bsize);
    double r_num = 0.0, r_den = 0.0, r1_num = 0.0, r1_den = 0.0;
    double sinf_num = 0.0, sinf_den = 0.0;   // |dW~_inf|_F^2 and |Z|_F^2 over my HF block
    {
      auto const &dW = mb_state.dW_qtPQ.value();                          // (q, t, P, Q), q NOT split
      auto org = dW.origin();
      auto lsh = dW.local_shape();
      auto grd = dW.grid();
      utils::check(grd[0] == 1, "build_sigma_lff: the q axis of dW_qtPQ must not be split (grid[0] = {}).", grd[0]);
      const long t_org = org[1], P_org = org[2], Q_org = org[3];
      const long t_bs = lsh[1], P_bs = lsh[2], Q_bs = lsh[3], np_Q = grd[3];
      nda::range Ps(P_org, P_org + P_bs), Qs(Q_org, Q_org + Q_bs);
      auto t_rng = dWs_tqPQ.local_range(0);
      auto P_rng = dWs_tqPQ.local_range(2);
      auto Q_rng = dWs_tqPQ.local_range(3);
      utils::check(P_rng.first() == P_org and long(P_rng.size()) == P_bs and Q_rng.first() == Q_org and long(Q_rng.size()) == Q_bs
                   and long(dWs_tqPQ.local_range(1).size()) == nq,
                   "build_sigma_lff: the (t, q, P, Q) target layout does not match the (P, Q) partition of dW_qtPQ.");
      const long ntl = long(t_rng.size());
      boost::mpi3::communicator t_pool = mpi->comm.split(int(P_org * np_Q + Q_org), mpi->comm.rank());
      auto W_loc = dW.local();
      auto loc = dWs_tqPQ.local();
      loc() = ComplexType(0.0);
      nda::array<ComplexType, 3> W_bt(nt_h, P_bs, Q_bs), W_bw(nw_h, P_bs, Q_bs), Y(nw_h, Np, Nm), A(nw_h, P_bs, Q_bs), Bt(nt_h, P_bs, Q_bs);
      nda::array<ComplexType, 2> tmpPm(P_bs, Nm), tmpPQ(P_bs, Q_bs), Sinf(P_bs, Q_bs);
      // the INSTANTANEOUS part dW~_inf = Herm[Z t^dag G1(nu_last) t] (the vertex's nu -> inf limit times the bare Coulomb):
      // a delta(tau) the bosonic tau machinery cannot carry -> subtracted from every node here (the remainder decays) and
      // published on the HF exchange grid for the static self-energy (gw_t::evaluate), or dropped (pol_vertex_sigma_static)
      const long np_hf_P = long(utils::find_proc_grid_min_diff(long(mpi->comm.size()), 1, 1)), np_hf_Q = long(mpi->comm.size()) / np_hf_P;
      auto dWinf = make_distributed_array<nda::array<ComplexType, 3>>(mpi->comm, {1, np_hf_P, np_hf_Q}, {nq, Np, Np});
      auto inf_loc = dWinf.local();
      auto inf_P = dWinf.local_range(1);
      auto inf_Q = dWinf.local_range(2);
      inf_loc() = ComplexType(0.0);
      nda::array<ComplexType, 2> Sfull(Np, Np), tmpNm(Np, Nm);
      for (long iq = 0; iq < nq; ++iq) {
        auto t_q = tmap(iq, all, all);
        // (0) the instantaneous kernel of this q, replicated: Sfull = scale/2 [ (Z t^dag) G1(inf) t + h.c. ]
        nda::blas::gemm(Zt(iq, all, all), G1(nw_h - 1, iq, all, all), tmpNm);      // (Np x Nm)(Nm x Nm)
        nda::blas::gemm(tmpNm, t_q, Sfull);                                          // (Np x Nm)(Nm x Np)
        for (long i = 0; i < Np; ++i)
          for (long k = i; k < Np; ++k) {
            const ComplexType v = 0.5 * scale * (Sfull(i, k) + std::conj(Sfull(k, i)));
            Sfull(i, k) = v; Sfull(k, i) = std::conj(v);
          }
        inf_loc(iq, all, all) = Sfull(inf_P, inf_Q);
        Sinf() = Sfull(Ps, Qs);
        for (long i = 0; i < P_bs; ++i)
          for (long k = 0; k < Q_bs; ++k) { sinf_num += std::norm(Sinf(i, k)); }
        // (a) Y_dW = dW(q, i nu) t^dag: my block's full-t slab (t-pool all_reduce over the disjoint t partition), tau -> nu,
        //     times t^dag on my Q range; the comm all_reduce sums the disjoint (P, Q) block partials -> replicated Y(q)
        W_bt() = ComplexType(0.0);
        if (t_bs > 0) W_bt(nda::range(t_org, t_org + t_bs), all, all) = W_loc(iq, all, all, all);
        t_pool.all_reduce_in_place_n(W_bt.data(), W_bt.size(), std::plus<>{});
        Y() = ComplexType(0.0);
        _ft->tau_to_w_PHsym(W_bt, W_bw);                                   // every rank of the t-pool (its own t-range is kept below)
        if (t_pool.rank() == 0)
          for (long j = 0; j < nw_h; ++j) {
            nda::blas::gemm(W_bw(j, all, all), nda::dagger(t_q(all, Qs)), tmpPm);   // (P_bs x Q_bs)(Q_bs x Nm)
            for (long i = 0; i < P_bs; ++i)
              for (long m = 0; m < Nm; ++m) Y(j, P_org + i, m) += tmpPm(i, m);
          }
        mpi->comm.all_reduce_in_place_n(Y.data(), Y.size(), std::plus<>{});
        for (long j = 0; j < nw_h; ++j)                                      // (b) + Z t^dag
          for (long i = 0; i < Np; ++i)
            for (long m = 0; m < Nm; ++m) Y(j, i, m) += Zt(iq, i, m);
        // (c) my block of dW~ = scale/2 [ Y G1 t + t^dag G1^dag Y^dag ]
        double nW = 0.0, nD = 0.0;
        for (long j = 0; j < nw_h; ++j) {
          nda::blas::gemm(Y(j, Ps, all), G1(j, iq, all, all), tmpPm);        // (P_bs x Nm)(Nm x Nm)
          nda::blas::gemm(tmpPm, t_q(all, Qs), tmpPQ);                        // (P_bs x Nm)(Nm x Q_bs)
          A(j, all, all) = tmpPQ;
          nda::blas::gemm(nda::dagger(t_q(all, Ps)), nda::dagger(G1(j, iq, all, all)), tmpPm);   // t_P^dag G1^dag
          nda::blas::gemm(tmpPm, nda::dagger(Y(j, Qs, all)), tmpPQ);                             // (P_bs x Nm)(Nm x Q_bs)
          for (long i = 0; i < P_bs; ++i)
            for (long k = 0; k < Q_bs; ++k) {
              A(j, i, k) = 0.5 * scale * (A(j, i, k) + tmpPQ(i, k)) - Sinf(i, k);   // the dynamic remainder (-> 0 as nu -> inf)
              nD += std::norm(A(j, i, k)); nW += std::norm(W_bw(j, i, k));
            }
          if (j == 0 and iq == std::min(1l, nq - 1) and t_pool.rank() == 0) {
            for (long i = 0; i < P_bs; ++i)
              for (long k = 0; k < Q_bs; ++k) { r1_num += std::norm(A(j, i, k)); r1_den += std::norm(W_bw(j, i, k)); }
          }
        }
        if (t_pool.rank() == 0) { r_num += nD; r_den += nW; }
        // (d) nu -> tau on my block; keep my t-range
        auto A2 = nda::reshape(A, shape_t<2>{nw_h, P_bs * Q_bs});
        auto B2 = nda::reshape(Bt, shape_t<2>{nt_h, P_bs * Q_bs});
        _ft->w_to_tau_PHsym(A2, B2);
        for (long it = 0; it < ntl; ++it) loc(it, iq, all, all) = Bt(t_rng.first() + it, all, all);
      }
      // |dW~_inf|_F over my HF block vs |Z|_F (the size of the instantaneous piece relative to the bare Coulomb)
      for (long iq = 0; iq < nq; ++iq) {
        nda::array<ComplexType, 2> Zq = thc.Z(int(iq));                    // collective
        for (long i : inf_P) for (long k : inf_Q) sinf_den += std::norm(Zq(i, k));
      }
      sinf_num = 0.0;
      for (long iq = 0; iq < nq; ++iq)
        for (long i = 0; i < long(inf_P.size()); ++i)
          for (long k = 0; k < long(inf_Q.size()); ++k) sinf_num += std::norm(inf_loc(iq, i, k));
      if (_vertex->sigma_lff_static()) mb_state.dWsig_inf_qPQ.emplace(std::move(dWinf));
      else mb_state.dWsig_inf_qPQ.reset();
    }
    sinf_num = mpi->comm.all_reduce_value(sinf_num, std::plus<>{}); sinf_den = mpi->comm.all_reduce_value(sinf_den, std::plus<>{});
    const double r_inf = std::sqrt(sinf_num / std::max(sinf_den, 1e-300));
    r_num = mpi->comm.all_reduce_value(r_num, std::plus<>{}); r_den = mpi->comm.all_reduce_value(r_den, std::plus<>{});
    r1_num = mpi->comm.all_reduce_value(r1_num, std::plus<>{}); r1_den = mpi->comm.all_reduce_value(r1_den, std::plus<>{});
    const double r_max = std::sqrt(r_num / std::max(r_den, 1e-300)), r_q1 = std::sqrt(r1_num / std::max(r1_den, 1e-300));
    // ---- 5. the q -> 0 head of the correction, with the loop's own extractor (eps_inv_head convention) --------------
    const bool cvv = (_div_treatment == "cvv");
    auto [eih_q, eih] = div_utils::eps_inv_head_t(dWs_tqPQ, thc, *MF, _ft, cvv ? "ignore_g0" : _div_treatment);
    for (auto &v : eih) v *= hscale;
    // ---- 6. publish in the layout gw_t consumes (the same transposition update_w applies to dW) --------------------
    {
      auto gsh = dWs_tqPQ.global_shape();
      mb_state.dWsig_qtPQ.emplace(make_distributed_array<nda::array<ComplexType, 4>>(
          mpi->comm, {t_pgrid[1], t_pgrid[0], t_pgrid[2], t_pgrid[3]}, {gsh[1], gsh[0], gsh[2], gsh[3]},
          {t_bsize[1], t_bsize[0], t_bsize[2], t_bsize[3]}));
      auto A = dWs_tqPQ.local();
      auto B = mb_state.dWsig_qtPQ.value().local();
      const long ntl = dWs_tqPQ.local_shape()[0], nql = dWs_tqPQ.local_shape()[1];
      for (long iq = 0; iq < nql; ++iq)
        for (long it = 0; it < ntl; ++it) B(iq, it, nda::ellipsis{}) = A(it, iq, nda::ellipsis{});
    }
    mb_state.eps_inv_head_sig = eih;
    const double t_up = sw.lap();
    // the q -> 0 heads at i nu = 0 (the static limit; tau_0 is the Matsubara sum, tail-dominated): the correction vs the loop's
    double h0_sig = 0.0, h0 = 0.0;
    {
      auto head_nu0 = [&](nda::array<ComplexType, 1> const &ht) {
        nda::array<ComplexType, 1> et(ht.shape(0)), ew(nw_h);
        et() = ht;
        auto a2 = nda::reshape(et, shape_t<2>{long(et.shape(0)), 1});
        auto b2 = nda::reshape(ew, shape_t<2>{nw_h, 1});
        _ft->tau_to_w_PHsym(a2, b2);
        return ew(0).real();
      };
      h0_sig = head_nu0(eih);
      if (mb_state.eps_inv_head.has_value() and long(mb_state.eps_inv_head.value().shape(0)) == nt_h) h0 = head_nu0(mb_state.eps_inv_head.value());
    }
    _sig_lff_meter = {v_q1, r_max, h0_sig, h0};
    app_log(1, "  [LFF-Sigma] instantaneous part |dW~_inf|_F / |Z|_F = {:.3e} (the local vertex at the tail node: {:+.4f}; {}); "
               "heads at i nu = 0: eps^-1 - 1 = {:+.5f} (loop), correction {:+.5f} (expected sign = that of the vertex at q_1: {:+.4f})",
            r_inf, v_tail, _vertex->sigma_lff_static() ? "-> the static self-energy" : "DROPPED (pol_vertex_sigma_static = false)",
            h0, h0_sig, v_q1);
    app_log(1, "  [LFF-Sigma] local vertex scalar tr(P dPi P)/tr(P Pi_0 P) at nu_0: q_1 {:+.4f}, Gamma {:+.4f}; tr(Pi_0)/tr(B_dump) at nu_0: q_1 {:.4f}, Gamma {:.4f}"
               " (1 by definition for the window bubble)", v_q1, v_q0, pb_q1, pb_q0);
    _sig_lff_meter[0] = v_q1;
    app_log(1, "  [LFF-Sigma] dW~ = {} x Herm[(Z + dW) t^dag G1 t], G1 = {} (column {}), N_m = {}, {} half nodes x {} q; frame metric cond(t t^dag) {:.2e}:\n"
               "              |G1| rms max {:.3e} (q_1, nu_0: {:.3e}; Gamma, nu_0: {:.3e}); |dW~|_F / |dW|_F all (q, nu) {:.3e} (q_1, nu_0: {:.3e});\n"
               "              Pi_0^-1 cut {} of {} modes at most (tol {:.1e}); heads at i nu = 0: correction {:+.5f} vs the loop's {:+.5f} "
               "(ratio {:.3f}, head scale {}); wall read {:.1f} s, metric+Z {:.1f} s, G1 {:.1f} s, blocks+head {:.1f} s",
            scale, full_bub ? "Pib^-1 M dPi (full RPA bubble folded)" : "M^-1 B^-1 dPi (window bubble)", col, Nm, nw_h, nq, mcond_max,
            g_max, g_q1, g_q0, r_max, r_q1, ncut_max, Nm, tol, h0_sig, h0,
            (std::abs(h0) > 1e-300 ? h0_sig / h0 : 0.0), hscale, t_read, t_fold, t_alg, t_up);
  }

  template void scr_coulomb_t::update_w(MBState&, thc_reader_t&, long);
  template void scr_coulomb_t::build_sigma_lff(MBState&, thc_reader_t&, std::array<long, 4>, std::array<long, 4>);
  template void scr_coulomb_t::build_sigma_pair(MBState&, thc_reader_t&);
  template nda::array<ComplexType, 4> scr_coulomb_t::read_pol_interp_column(std::string const&, long, long,
                                                                            nda::array<long, 1> const&, thc_reader_t&);
  template void scr_coulomb_t::fold_rpa_pi_secondary(
      memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>&, thc_reader_t&);

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
      std::string, long, mpi3::communicator &, mf::MF &, eps_fit::eps_inf_fit_t const *);


}  // solvers
}  // methods
