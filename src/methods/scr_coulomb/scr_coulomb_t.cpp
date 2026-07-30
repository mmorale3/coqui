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


#include <unordered_set>
#include "utilities/h5_background_writer.hpp"
#include "utilities/freemem.h"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/HF/thc_solver_comm.hpp"
#include "methods/GW/g0_div_utils.hpp"
#include "scr_coulomb_t.h"
#include "rpa_pi.icc"
#include "edmft_pi.icc"

namespace methods {
namespace solvers {

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

  template<MEMORY_SPACE MEM>
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
    // (TEMP) per-phase GPU timers for update_w. Strip together with
    // the other TEMP_* timers once the perf hot-spots are identified.
    // Pre-register every TEMP_* timer referenced in the dump below:
    // elapsed()/number_of_calls() abort on unregistered names, and some
    // timers only ever start on the DEVICE path (e.g. the dev-alloc ones),
    // which crashed HOST runs at the first dump.
    for (auto const& name : {"TEMP_UW_TOTAL", "TEMP_UW_eval_Pi_qdep",
         "TEMP_UW_dyson_W_from_Pi", "TEMP_DWFP_tau_to_w",
         "TEMP_DWFP_dyson_W_in_place", "TEMP_DWFP_w_to_tau",
         "TEMP_DWiP_alloc", "TEMP_DWiP_Z_load", "TEMP_DWiP_Pi_load",
         "TEMP_DWiP_gemm_ZPi", "TEMP_DWiP_diag_I_minus_A",
         "TEMP_DWiP_inverse", "TEMP_DWiP_diag_A_minus_I",
         "TEMP_DWiP_gemm_AZ", "TEMP_DWiP_writeback", "TEMP_UW_eps_inv_head",
         "TEMP_UW_dW_qtPQ_alloc", "TEMP_UW_dW_transpose_and_mirror",
         "TEMP_UW_dW_qtPQ_dev_alloc", "TEMP_UW_dW_dev_transpose",
         "TEMP_UW_dW_d2h_mirror", "TEMP_UW_dump_eps_inv_head",
         "TEMP_FT_alloc", "TEMP_FT_redist", "TEMP_FT_leakage",
         "TEMP_FT_gemm", "TEMP_FT_reset", "TEMP_FT_barrier"})
      _Timer.add(name);
    _Timer.start("TEMP_UW_TOTAL");
    _Timer.start("TEMP_UW_eval_Pi_qdep");
    auto dPi_tqPQ = eval_Pi_qdep<MEM>(mb_state, thc);
    utils::device_sync();
    _Timer.stop("TEMP_UW_eval_Pi_qdep");

    // evaluate screened interaction (dW_tqPQ) and reset polarizability (dPi_tqPQ)
    // a) dPi_tqPQ is reset during dyson_W_from_Pi_tau()
    // b) pgrid and bsize of dW_tqPQ are forced to be the same as in dPi_tqPQ
    _Timer.start("TEMP_UW_dyson_W_from_Pi");
    auto dW_tqPQ = dyson_W_from_Pi_tau<false>(dPi_tqPQ, thc, true);
    utils::device_sync();
    _Timer.stop("TEMP_UW_dyson_W_from_Pi");

    // div_utils::eval_eps_inv_q (called inside eps_inv_head_t) is now
    // MEM-aware; it runs the per-(ix,iq) gemv/dot on whatever MEM the
    // dW darray lives in. No full host copy of dW needed.
    _Timer.start("TEMP_UW_eps_inv_head");
    auto [eps_inv_head_q, eps_inv_head] =
        div_utils::eps_inv_head_t(dW_tqPQ, thc, *thc.MF(), _ft, _div_treatment);
    mb_state.eps_inv_head = eps_inv_head;
    utils::device_sync();
    _Timer.stop("TEMP_UW_eps_inv_head");

    // Stage W into MBState. The on-device pipeline keeps a device-typed
    // dW_qtPQ_dev so gw_t::evaluate<DEVICE> can read it without a
    // host->device round-trip. The host mirror dW_qtPQ is also written
    // for cross-method consumers (GF2, h5 dump, EDMFT) that still run
    // on host.
    auto t_pgrid = dW_tqPQ.grid();
    auto t_bsize = dW_tqPQ.block_size();
    auto gshape = dW_tqPQ.global_shape();
    std::array<long,4> q_pgrid = {t_pgrid[1], t_pgrid[0], t_pgrid[2], t_pgrid[3]};
    std::array<long,4> q_bsize = {t_bsize[1], t_bsize[0], t_bsize[2], t_bsize[3]};
    std::array<long,4> q_gshape = {gshape[1],  gshape[0],  gshape[2],  gshape[3]};

    long nt_loc = dW_tqPQ.local_shape()[0];
    long nq_loc = dW_tqPQ.local_shape()[1];

    // On the device path the host mirror is only needed by consumers that run
    // on the host (GF2, EDMFT, the optional W h5 dump), and a pure scGW run has
    // none of them: gw_t::evaluate<DEVICE> reads dW_qtPQ_dev. Materializing it
    // anyway cost 4.2 s to allocate and zero 18.5 GB plus 9.2 s to copy it back
    // per iteration, for data nothing read. Skip it here; MBState builds it on
    // demand from the device array (see MBState::W_host()).
    const bool need_host_W = (MEM == HOST_MEMORY) or mb_state.keep_host_W;
    if (need_host_W) {
      _Timer.start("TEMP_UW_dW_qtPQ_alloc");
      mb_state.dW_qtPQ.emplace(make_distributed_array<nda::array<ComplexType, 4>> (
                               thc.mpi()->comm, q_pgrid, q_gshape, q_bsize));
      utils::device_sync();
      _Timer.stop("TEMP_UW_dW_qtPQ_alloc");
    }

    _Timer.start("TEMP_UW_dW_transpose_and_mirror");
    if constexpr (MEM == HOST_MEMORY) {
      auto W_qtPQ_h = mb_state.dW_qtPQ.value().local();
      auto W_tqPQ = dW_tqPQ.local();
      for (size_t qt = 0; qt < nq_loc * nt_loc; ++qt) {
        size_t iq = qt / nt_loc;
        size_t it = qt % nt_loc;
        W_qtPQ_h(iq, it, nda::ellipsis{}) = W_tqPQ(it, iq, nda::ellipsis{});
      }
    }
#if defined(ENABLE_DEVICE)
    else {
      // Device path:
      //  (1) allocate a device-typed (q,t,P,Q) darray and transpose on device,
      //  (2) mirror to the host darray as a single bulk copy (one device->host
      //      transfer per SCF iter — what host-side consumers still need).
      using dev_local_t = memory::array<DEVICE_MEMORY, ComplexType, 4>;
      _Timer.start("TEMP_UW_dW_qtPQ_dev_alloc");
      mb_state.dW_qtPQ_dev.emplace(make_distributed_array<dev_local_t>(
                                   thc.mpi()->comm, q_pgrid, q_gshape, q_bsize));
      utils::device_sync();
      _Timer.stop("TEMP_UW_dW_qtPQ_dev_alloc");
      auto W_qtPQ_d = mb_state.dW_qtPQ_dev.value().local();
      auto W_tqPQ_d = dW_tqPQ.local();
      // On-device (t,q) -> (q,t) transpose, P,Q axes preserved.
      _Timer.start("TEMP_UW_dW_dev_transpose");
      for (size_t qt = 0; qt < nq_loc * nt_loc; ++qt) {
        size_t iq = qt / nt_loc;
        size_t it = qt % nt_loc;
        W_qtPQ_d(iq, it, nda::ellipsis{}) = W_tqPQ_d(it, iq, nda::ellipsis{});
      }
      utils::device_sync();
      _Timer.stop("TEMP_UW_dW_dev_transpose");
      // Only when a host consumer asked for it; see need_host_W above.
      if (need_host_W) {
        _Timer.start("TEMP_UW_dW_d2h_mirror");
        mb_state.dW_qtPQ.value().local() = nda::to_host(W_qtPQ_d);
        utils::device_sync();
        _Timer.stop("TEMP_UW_dW_d2h_mirror");
      }
    }
#endif
    dW_tqPQ.reset();
    utils::device_sync();
    _Timer.stop("TEMP_UW_dW_transpose_and_mirror");

    mb_state.screen_type = _screen_type;

    _Timer.start("TEMP_UW_dump_eps_inv_head");
    if (h5_iter>=0) {
      // This writes the same file, the same scf/iter<N> group and from the same
      // rank as the checkpoint, so it is the one place a background checkpoint
      // write would collide with certainty. Joining here is also what makes the
      // overlap worth having: it sits at the end of update_w, ~50 s into a
      // 141 s iteration, which is long enough to hide a 30 s write.
      utils::h5_quiesce();
      dump_eps_inv_head(eps_inv_head_q, eps_inv_head,
                        mb_state.coqui_prefix, h5_iter,
                        thc.mpi()->comm, *thc.MF());
    }
    _Timer.stop("TEMP_UW_dump_eps_inv_head");
    utils::device_sync();
    _Timer.stop("TEMP_UW_TOTAL");

    // (TEMP) Phase-by-phase update_w timers. Accumulated across SCF
    // iterations; divide by iter count or use number_of_calls() to
    // get per-iter values. Strip when perf hot-spots are nailed.
    app_log(2, "\n  (TEMP) update_w<{}> phase timers (accumulated):",
            (MEM == HOST_MEMORY) ? "HOST" : "DEVICE");
    app_log(2, "    Total:                            {0:.3f} sec  ({1} calls)",
            _Timer.elapsed("TEMP_UW_TOTAL"), _Timer.number_of_calls("TEMP_UW_TOTAL"));
    app_log(2, "      eval_Pi_qdep:                   {0:.3f} sec",
            _Timer.elapsed("TEMP_UW_eval_Pi_qdep"));
    app_log(2, "      dyson_W_from_Pi_tau:            {0:.3f} sec",
            _Timer.elapsed("TEMP_UW_dyson_W_from_Pi"));
    if (_Timer.elapsed("TEMP_DWFP_tau_to_w") > 0.0) {
      app_log(2, "        - tau_to_w:                   {0:.3f} sec",
              _Timer.elapsed("TEMP_DWFP_tau_to_w"));
      app_log(2, "        - dyson_W_in_place:           {0:.3f} sec",
              _Timer.elapsed("TEMP_DWFP_dyson_W_in_place"));
      app_log(2, "        - w_to_tau:                   {0:.3f} sec",
              _Timer.elapsed("TEMP_DWFP_w_to_tau"));
      if (_Timer.elapsed("TEMP_FT_alloc") > 0.0) {
        // tau_to_w + w_to_tau inner sections, both directions together.
        app_log(2, "          FT inner (both directions, accumulated):");
        app_log(2, "            darray alloc+zero:      {0:.3f} sec  ({1} calls)",
                _Timer.elapsed("TEMP_FT_alloc"), _Timer.number_of_calls("TEMP_FT_alloc"));
        app_log(2, "            redistribute:           {0:.3f} sec  ({1} calls)",
                _Timer.elapsed("TEMP_FT_redist"), _Timer.number_of_calls("TEMP_FT_redist"));
        app_log(2, "            FT gemm (PHsym):        {0:.3f} sec",
                _Timer.elapsed("TEMP_FT_gemm"));
        app_log(2, "            check_leakage:          {0:.3f} sec",
                _Timer.elapsed("TEMP_FT_leakage"));
        app_log(2, "            darray reset (free):    {0:.3f} sec  ({1} calls)",
                _Timer.elapsed("TEMP_FT_reset"), _Timer.number_of_calls("TEMP_FT_reset"));
        app_log(2, "            barriers:               {0:.3f} sec",
                _Timer.elapsed("TEMP_FT_barrier"));
      }
      if (_Timer.elapsed("TEMP_DWiP_inverse") > 0.0) {
        app_log(2, "          dyson_W_in_place inner (accumulated):");
        app_log(2, "            alloc dPi/dZ/dA:        {0:.3f} sec",
                _Timer.elapsed("TEMP_DWiP_alloc"));
        app_log(2, "            Z load (per-q):         {0:.3f} sec  ({1} calls)",
                _Timer.elapsed("TEMP_DWiP_Z_load"),
                _Timer.number_of_calls("TEMP_DWiP_Z_load"));
        app_log(2, "            Pi load (per-w):        {0:.3f} sec  ({1} calls)",
                _Timer.elapsed("TEMP_DWiP_Pi_load"),
                _Timer.number_of_calls("TEMP_DWiP_Pi_load"));
        app_log(2, "            gemm Z*Pi -> A:         {0:.3f} sec",
                _Timer.elapsed("TEMP_DWiP_gemm_ZPi"));
        app_log(2, "            A = I - A:              {0:.3f} sec",
                _Timer.elapsed("TEMP_DWiP_diag_I_minus_A"));
        app_log(2, "            inverse(A) [SLATE LU]:  {0:.3f} sec",
                _Timer.elapsed("TEMP_DWiP_inverse"));
        app_log(2, "            A = A - I:              {0:.3f} sec",
                _Timer.elapsed("TEMP_DWiP_diag_A_minus_I"));
        app_log(2, "            gemm A*Z -> W:          {0:.3f} sec",
                _Timer.elapsed("TEMP_DWiP_gemm_AZ"));
        app_log(2, "            writeback to Pi_wqPQ:   {0:.3f} sec",
                _Timer.elapsed("TEMP_DWiP_writeback"));
      }
    }
    app_log(2, "      eps_inv_head_t:                 {0:.3f} sec",
            _Timer.elapsed("TEMP_UW_eps_inv_head"));
    app_log(2, "      dW_qtPQ_alloc:                  {0:.3f} sec",
            _Timer.elapsed("TEMP_UW_dW_qtPQ_alloc"));
    app_log(2, "      dW_transpose_and_mirror:        {0:.3f} sec",
            _Timer.elapsed("TEMP_UW_dW_transpose_and_mirror"));
    if (_Timer.elapsed("TEMP_UW_dW_qtPQ_dev_alloc") > 0.0) {
      app_log(2, "        - dW_qtPQ_dev alloc:          {0:.3f} sec",
              _Timer.elapsed("TEMP_UW_dW_qtPQ_dev_alloc"));
      app_log(2, "        - on-device (t,q)->(q,t):     {0:.3f} sec",
              _Timer.elapsed("TEMP_UW_dW_dev_transpose"));
      app_log(2, "        - bulk D->H mirror:           {0:.3f} sec",
              _Timer.elapsed("TEMP_UW_dW_d2h_mirror"));
    }
    app_log(2, "      dump_eps_inv_head:              {0:.3f} sec\n",
            _Timer.elapsed("TEMP_UW_dump_eps_inv_head"));
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

    // (TEMP) split dyson_W_from_Pi_tau into its three phases. This is
    // currently the dominant cost of update_w (~80% per the TEMP_UW_*
    // breakdown). Strip when perf hot-spots are localized.
    _Timer.start("TEMP_DWFP_tau_to_w");
    auto dPi_wqPQ = tau_to_w(dPi_tqPQ_pos, w_pgrid, w_bsize, reset_input);
    utils::device_sync();
    _Timer.stop("TEMP_DWFP_tau_to_w");

    _Timer.start("TEMP_DWFP_dyson_W_in_place");
    dyson_W_in_place(dPi_wqPQ, thc);
    utils::device_sync();
    _Timer.stop("TEMP_DWFP_dyson_W_in_place");

    if constexpr (w_out) {
      return dPi_wqPQ;
    } else {
      _Timer.start("TEMP_DWFP_w_to_tau");
      auto dPi_tqPQ = w_to_tau(dPi_wqPQ, t_pgrid, t_bsize, true);
      utils::device_sync();
      _Timer.stop("TEMP_DWFP_w_to_tau");
      return dPi_tqPQ;
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

    // MEM is inferred from the input darray; the local slate work
    // arrays (dPi_PQ / dZ_PQ / dA_PQ) live in the same memory space.
    constexpr MEMORY_SPACE MEM =
        ::nda::mem::on_host<Array_4D_t> ? HOST_MEMORY : DEVICE_MEMORY;
    using Array_2D_t = memory::array<MEM, ComplexType, 2>;
    using math::nda::make_distributed_array;
    // (TEMP) inner-loop perf breakdown of dyson_W_in_place
    _Timer.start("TEMP_DWiP_alloc");
    auto dPi_PQ = make_distributed_array<Array_2D_t>(wq_intra_comm, {pgrid[2], pgrid[3]}, {NP, NQ}, {block_size[2], block_size[3]}, true);
    auto dZ_PQ  = make_distributed_array<Array_2D_t>(wq_intra_comm, {pgrid[2], pgrid[3]}, {NP, NQ}, {block_size[2], block_size[3]}, true);
    auto dA_PQ  = make_distributed_array<Array_2D_t>(wq_intra_comm, {pgrid[2], pgrid[3]}, {NP, NQ}, {block_size[2], block_size[3]}, true);
    utils::device_sync();
    _Timer.stop("TEMP_DWiP_alloc");
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

    // Precompute the local-block identity matrix once, in MEM, so the
    // per-element diagonal updates become a single tensor::elementwise
    // call instead of a host loop (lazy on-device assignment is illegal).
    [[maybe_unused]] memory::array<MEM, ComplexType, 2> Id_PQ_mem;
    if constexpr (MEM != HOST_MEMORY) {
      nda::array<ComplexType, 2> Id_PQ_host(NP_loc, NQ_loc);
      Id_PQ_host() = ComplexType(0.0);
      for (auto idx: diag_idx) Id_PQ_host(idx.first, idx.second) = ComplexType(1.0);
      Id_PQ_mem = memory::to_memory_space<MEM>(Id_PQ_host);
    }

    auto Pi_wqPQ = dPi_wqPQ.local();
    auto Pi_PQ = dPi_PQ.local();
    auto Z_PQ = dZ_PQ.local();
    auto A_PQ = dA_PQ.local();
    for (size_t iq_loc = 0; iq_loc < nq_loc; ++iq_loc) {
      long iq = q_origin + iq_loc;
      _Timer.start("TEMP_DWiP_Z_load");
      Z_PQ = thc.template Z<MEM>(iq, P_rng, Q_rng, qpool_id, pgrid[1], q_intra_comm);
      utils::device_sync();
      _Timer.stop("TEMP_DWiP_Z_load");

      // W(w) = [ I - Z * Pi(w)]^{-1} * Z - Z
      for (size_t n = 0; n < nw_loc; ++n) {
        _Timer.start("TEMP_DWiP_Pi_load");
        Pi_PQ = Pi_wqPQ(n, iq_loc, nda::ellipsis{});
        utils::device_sync();
        _Timer.stop("TEMP_DWiP_Pi_load");

        // A = Z * Pi(w)
        _Timer.start("TEMP_DWiP_gemm_ZPi");
        math::nda::slate_ops::multiply(dZ_PQ, dPi_PQ, dA_PQ);
        utils::device_sync();
        _Timer.stop("TEMP_DWiP_gemm_ZPi");
        // A = I - Z * Pi(w) (i.e. A := -A; A_diag += 1)
        _Timer.start("TEMP_DWiP_diag_I_minus_A");
        if constexpr (MEM == HOST_MEMORY) {
          for (auto idx: diag_idx) {
            A_PQ(idx.first, idx.second) -= ComplexType(1.0);
          }
          A_PQ *= -1.0;
        } else {
          nda::tensor::scale(ComplexType(-1.0), A_PQ);
          nda::tensor::elementwise(ComplexType(1.0), Id_PQ_mem, "PQ",
                                   ComplexType(1.0), A_PQ,     "PQ",
                                   nda::tensor::op::SUM);
        }
        utils::device_sync();
        _Timer.stop("TEMP_DWiP_diag_I_minus_A");

        // A = [I - Z*Pi(w)]^{-1}
        _Timer.start("TEMP_DWiP_inverse");
        math::nda::slate_ops::inverse(dA_PQ);
        utils::device_sync();
        _Timer.stop("TEMP_DWiP_inverse");

        // A = [I - Z*Pi(w)]^{-1} - I (diag -= 1)
        _Timer.start("TEMP_DWiP_diag_A_minus_I");
        if constexpr (MEM == HOST_MEMORY) {
          for (auto idx: diag_idx) {
            A_PQ(idx.first, idx.second) -= ComplexType(1.0);
          }
        } else {
          nda::tensor::elementwise(ComplexType(-1.0), Id_PQ_mem, "PQ",
                                   ComplexType(1.0), A_PQ,     "PQ",
                                   nda::tensor::op::SUM);
        }
        utils::device_sync();
        _Timer.stop("TEMP_DWiP_diag_A_minus_I");

        // W = ([I - Z*Pi(w)]^{-1} - I) * Z
        _Timer.start("TEMP_DWiP_gemm_AZ");
        math::nda::slate_ops::multiply(dA_PQ, dZ_PQ, dPi_PQ);
        utils::device_sync();
        _Timer.stop("TEMP_DWiP_gemm_AZ");
        _Timer.start("TEMP_DWiP_writeback");
        Pi_wqPQ(n, iq_loc, nda::ellipsis{}) = Pi_PQ;
        utils::device_sync();
        _Timer.stop("TEMP_DWiP_writeback");
      }
    }
    // prevent dead block in thc.Z() in case nq_loc is not the same for all processors
    for (long iq_loc = nq_loc; iq_loc < nq_loc_max; ++iq_loc)
      Z_PQ = thc.template Z<MEM>(0, P_rng, Q_rng, qpool_id, pgrid[1], q_intra_comm);

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

  template<MEMORY_SPACE MEM>
  auto scr_coulomb_t::eval_Pi_qdep(MBState &mb_state, THC_ERI auto &thc)
  -> memory::darray_t<memory::array<MEM, ComplexType, 4>, mpi3::communicator>
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
    if constexpr (MEM != HOST_MEMORY) {
      utils::check(_screen_type != "rpa_k" and
                   _screen_type.find("crpa") == std::string::npos and
                   _screen_type.find("edmft") == std::string::npos,
                   "scr_coulomb_t::eval_Pi_qdep: device path only supports the "
                   "pure RPA (R-space) algorithm; got screen_type='{}'.",
                   _screen_type);
    }

    auto G_tskij = mb_state.sG_tskij.value().local();

    if constexpr (MEM == HOST_MEMORY) {
      // k-space algorithm is HOST-only (eval_Pi_rpa_kspace has a
      // static_assert against DEVICE_MEMORY); only instantiate the
      // call here on the host path.
      if (_screen_type == "rpa_k")
        return eval_Pi_rpa_kspace<MEM>(G_tskij, thc);
    }

    // RPA polarizability (R-space; works for both HOST and DEVICE).
    auto dPi_tqPQ = eval_Pi_rpa_Rspace<MEM>(G_tskij, thc);
    if (_screen_type.find("gw_edmft_rpa")!=std::string::npos or _screen_type=="rpa")
      return dPi_tqPQ;

    if constexpr (MEM == HOST_MEMORY) {
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
          app_log(1, "╚══════════════════════════════════════════════════╝\n");

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
    }

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
    // (TEMP) The redistributes are only part of this phase: each call also
    // creates three tensor-sized distributed arrays and frees two. Time every
    // section so the cost is attributed rather than assumed.
    _Timer.start("TEMP_FT_alloc");
    auto buffer_ti  = make_distributed_array<local_Array_t>(
        *comm, b_pgrid, t_gshape, dPi_tqPQ_pos.block_size());
    utils::device_sync();
    _Timer.stop("TEMP_FT_alloc");
    _Timer.start("FT_REDISTRIBUTE");
    _Timer.start("TEMP_FT_redist");
    math::nda::redistribute(dPi_tqPQ_pos, buffer_ti);
    _Timer.stop("TEMP_FT_redist");
    _Timer.stop("FT_REDISTRIBUTE");
    _Timer.start("TEMP_FT_reset");
    if (reset_input) dPi_tqPQ_pos.reset();
    _Timer.stop("TEMP_FT_reset");
    _Timer.start("TEMP_FT_leakage");
    _ft->check_leakage(buffer_ti, imag_axes_ft::boson, "polarizability", true);
    _Timer.stop("TEMP_FT_leakage");
    _Timer.start("TEMP_FT_barrier");
    buffer_ti.communicator()->barrier();
    _Timer.stop("TEMP_FT_barrier");

    _Timer.start("TEMP_FT_alloc");
    auto buffer_wi  = make_distributed_array<local_Array_t>(
        *comm, b_pgrid, w_gshape, buffer_ti.block_size());
    utils::device_sync();
    _Timer.stop("TEMP_FT_alloc");
    _Timer.start("TEMP_FT_gemm");
    {
      auto buf_ti_loc = buffer_ti.local();
      auto buf_wi_loc = buffer_wi.local();
      _ft->tau_to_w_PHsym(buf_ti_loc, buf_wi_loc);
    }
    utils::device_sync();
    _Timer.stop("TEMP_FT_gemm");
    _Timer.start("TEMP_FT_reset");
    buffer_ti.reset();
    _Timer.stop("TEMP_FT_reset");
    _Timer.start("TEMP_FT_barrier");
    buffer_wi.communicator()->barrier();
    _Timer.stop("TEMP_FT_barrier");

    _Timer.start("TEMP_FT_alloc");
    auto dPi_wqPQ = make_distributed_array<local_Array_t>(
        *comm, w_pgrid_out, w_gshape, w_bsize_out);
    utils::device_sync();
    _Timer.stop("TEMP_FT_alloc");

    _Timer.start("FT_REDISTRIBUTE");
    _Timer.start("TEMP_FT_redist");
    math::nda::redistribute(buffer_wi, dPi_wqPQ);
    _Timer.stop("TEMP_FT_redist");
    _Timer.stop("FT_REDISTRIBUTE");
    _Timer.start("TEMP_FT_reset");
    buffer_wi.reset();
    _Timer.stop("TEMP_FT_reset");
    _Timer.start("TEMP_FT_barrier");
    dPi_wqPQ.communicator()->barrier();
    _Timer.stop("TEMP_FT_barrier");

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
    _Timer.start("TEMP_FT_alloc");
    auto buffer_wi  = make_distributed_array<local_Array_t>(
        *comm, b_pgrid, w_gshape, dW_wqPQ_pos.block_size());
    utils::device_sync();
    _Timer.stop("TEMP_FT_alloc");
    _Timer.start("FT_REDISTRIBUTE");
    _Timer.start("TEMP_FT_redist");
    math::nda::redistribute(dW_wqPQ_pos, buffer_wi);
    _Timer.stop("TEMP_FT_redist");
    _Timer.stop("FT_REDISTRIBUTE");
    _Timer.start("TEMP_FT_reset");
    if (reset_input) dW_wqPQ_pos.reset();
    _Timer.stop("TEMP_FT_reset");

    _Timer.start("TEMP_FT_alloc");
    auto buffer_ti  = make_distributed_array<local_Array_t>(
        *comm, b_pgrid, t_gshape, buffer_wi.block_size());
    utils::device_sync();
    _Timer.stop("TEMP_FT_alloc");
    _Timer.start("TEMP_FT_gemm");
    {
      auto buf_ti_loc = buffer_ti.local();
      auto buf_wi_loc = buffer_wi.local();
      _ft->w_to_tau_PHsym(buf_wi_loc, buf_ti_loc);
    }
    utils::device_sync();
    _Timer.stop("TEMP_FT_gemm");
    _Timer.start("TEMP_FT_reset");
    buffer_wi.reset();
    _Timer.stop("TEMP_FT_reset");
    _Timer.start("TEMP_FT_leakage");
    _ft->check_leakage(buffer_ti, imag_axes_ft::boson, "screened interaction", true);
    _Timer.stop("TEMP_FT_leakage");

    _Timer.start("TEMP_FT_alloc");
    auto dW_tqPQ = make_distributed_array<local_Array_t>(
        *comm, t_pgrid_out, t_gshape, t_bsize_out);
    utils::device_sync();
    _Timer.stop("TEMP_FT_alloc");

    _Timer.start("FT_REDISTRIBUTE");
    _Timer.start("TEMP_FT_redist");
    math::nda::redistribute(buffer_ti, dW_tqPQ);
    _Timer.stop("TEMP_FT_redist");
    _Timer.stop("FT_REDISTRIBUTE");
    _Timer.start("TEMP_FT_reset");
    buffer_ti.reset();
    _Timer.stop("TEMP_FT_reset");

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
      utils::h5_quiesce();  // see h5_background_writer.hpp
      h5::file file(filename, 'a');
      h5::group grp(file);
      auto scf_grp = (grp.has_subgroup("scf")) ? grp.open_group("scf") : grp.create_group("scf");
      auto iter_grp = (scf_grp.has_subgroup(grp_name)) ?
                      scf_grp.open_group(grp_name) : scf_grp.create_group(grp_name);

      nda::h5_write(iter_grp, "eps_inv_head_wq", eps_inv_head_wq, false);
      nda::h5_write(iter_grp, "eps_inv_head_tq", eps_inv_head_tq, false);
      nda::h5_write(iter_grp, "eps_inv_head_w", eps_inv_head_w, false);
      nda::h5_write(iter_grp, "eps_inv_head_t", eps_inv_head_t, false);
    }
    comm.barrier();
  }


  // template instantiations
  using Arr4D = nda::array<ComplexType, 4>;
  using Arr = nda::array<ComplexType, 5>;
  using Arrv = nda::array_view<ComplexType, 5>;
  using Arrv2 = nda::array_view<ComplexType, 5, nda::C_layout>;

  template void scr_coulomb_t::update_w<HOST_MEMORY>(MBState&, thc_reader_t&, long);
#if defined(ENABLE_DEVICE)
  template void scr_coulomb_t::update_w<DEVICE_MEMORY>(MBState&, thc_reader_t&, long);
#endif

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
  scr_coulomb_t::eval_Pi_qdep<HOST_MEMORY>(MBState&, thc_reader_t&);
#if defined(ENABLE_DEVICE)
  template memory::darray_t<memory::array<DEVICE_MEMORY, ComplexType, 4>, mpi3::communicator>
  scr_coulomb_t::eval_Pi_qdep<DEVICE_MEMORY>(MBState&, thc_reader_t&);
#endif


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
