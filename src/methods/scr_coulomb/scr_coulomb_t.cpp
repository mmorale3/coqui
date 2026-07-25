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
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/HF/thc_solver_comm.hpp"
#include "methods/GW/g0_div_utils.hpp"
#include "methods/vertex/vertex_t.h"
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
    return _vertex != nullptr and _vertex->active() and
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
    if (has_active_vertex() and not vertex_has_rung(mb_state)) {
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
    auto dPi_tqPQ = eval_Pi_qdep(mb_state, thc);

    // evaluate screened interaction (dW_tqPQ) and reset polarizability (dPi_tqPQ)
    // a) dPi_tqPQ is reset during dyson_W_from_Pi_tau()
    // b) pgrid and bsize of dW_tqPQ are forced to be the same as in dPi_tqPQ
    auto dW_tqPQ = dyson_W_from_Pi_tau<false>(dPi_tqPQ, thc, true);
    auto [eps_inv_head_q, eps_inv_head] =
        div_utils::eps_inv_head_t(dW_tqPQ, thc, *thc.MF(), _ft, _div_treatment);
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
    }

    // ISDF-Vertex Refinement 2, W-bar iteration cache (notes/wbar_cache.md): with an
    // active SECONDARY-basis vertex, fold the freshest W into the N_m x N_m cache now
    // -- dW is alive here and mb_state.eps_inv_head is the SAME-iteration head (both
    // stored above), so the gygi Gamma augmentation is captured consistently. The
    // cache is consumed by the NEXT iteration's eval_Pi_C (identical one-iteration
    // lag as the retained-dW path); the scf driver then frees dW unconditionally in
    // this mode (needs_dw_retention() == false -- plain-GW memory profile).
    if (_vertex != nullptr and _vertex->active() and _vertex->secondary()
        and _vertex->w_cache_enabled())
      _vertex->cache_w(mb_state, thc);
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

    // ISDF-Vertex: additive second-order-exchange polarization cut Pi^C on the
    // same distributed grid as the RPA polarizability (EDMFT "+=" precedent below).
    // When no active vertex is attached this is a strict no-op -- no allocation,
    // no arithmetic -- so the disabled path is bit-identical to plain RPA/scGW.
    auto add_vertex_Pi_C = [&](auto &dPi) {
      if (_vertex != nullptr and _vertex->active()) {
        auto dPi_C_tqPQ = _vertex->eval_Pi_C(mb_state, thc, dPi.grid(),
                                             dPi.block_size(), dPi.global_shape());
        dPi.local() += dPi_C_tqPQ.local();
        thc.mpi()->comm.barrier();
      }
    };

    if (_screen_type == "rpa_k") {
      auto dPi_tqPQ = eval_Pi_rpa_kspace(G_tskij, thc);
      add_vertex_Pi_C(dPi_tqPQ);
      return dPi_tqPQ;
    }

    // RPA polarizability
    auto dPi_tqPQ = eval_Pi_rpa_Rspace(G_tskij, thc);
    if (_screen_type.find("gw_edmft_rpa")!=std::string::npos or _screen_type=="rpa") {
      add_vertex_Pi_C(dPi_tqPQ);
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
