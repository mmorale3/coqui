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

#ifndef COQUI_VERTEX_SECONDARY_FOLD_HPP
#define COQUI_VERTEX_SECONDARY_FOLD_HPP

/**
 * Impl 2 -- DISTRIBUTED downfold of the RPA (t, P, Q)-distributed dynamic W to the
 * secondary aux basis (N_m x N_m), WITHOUT ever gathering a full Np x Np block per rank
 * (notes/vertex_parallelization_v2_plan.md section "6b. Impl 2 distributed-downfold
 * DESIGN").
 *
 * The dynamic screened interaction mb_state.dW_qtPQ lives on the RPA proc grid
 * {1(q), nt_procs, np_P, np_Q}: the q axis is NOT split, the t/P/Q axes are. The legacy
 * secondary path (eval_Pi_C Step 1b) GATHERED a full (nt_half, Np, Np) tau slab per q
 * (6.4 GB per (t,q) block at production Np = 20000) and folded it with fold_core. This
 * routine instead assembles, PER (P,Q) BLOCK, only that block's full-t slab (a t-pool
 * all_reduce over the disjoint t-partition -- exact), folds it in place with
 * fold_core_block, and sums the (P,Q)-block partials with one final all_reduce. No rank
 * ever holds more than its own (P,Q) block.
 *
 * Correctness (proved in section 6b): at test scale the RPA puts every rank on t
 * (np_P = np_Q = 1) so there is ONE (P,Q) block == the whole array; the t_pool is the
 * whole comm; the block-assembly all_reduce reproduces the exact gathered slab (disjoint
 * partition sum), fold_core_block with the full range == fold_core, and the final comm
 * all_reduce sums one non-zero rank's result with zeros => BIT-IDENTICAL to the
 * replicated fold. At production the (P,Q) split makes the final sum a disjoint-block
 * reduction whose reassociation is O(1e-11) (refinement2 tol). The physics tests never
 * exercise np_P,np_Q > 1 -- test_vertex_dfold does, with a forced 1x1x2x2 grid.
 */

#include <array>
#include <functional>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/nda_functions.hpp"   // nda::blas::gemm(a, b, c) 3-arg wrapper (as fold_core)

#include "mpi3/communicator.hpp"

namespace methods {
namespace solvers {
namespace vertex_secondary_detail {

/**
 * Off-diagonal block fold of the downfold core (theoryB Eq. 36 restricted to a (P,Q)
 * block of the global aux index):
 *   out(m, n) = [ t_qP  A_PQ  t_qQ^dag ](m, n),
 * where t_qP = t(q)(:, P_range) is (N_m x P_bs), t_qQ = t(q)(:, Q_range) is (N_m x Q_bs),
 * and A_PQ is the block (P_bs x Q_bs). tmp_mQ is scratch (N_m x Q_bs). On a DIAGONAL block
 * with t_qP == t_qQ this is exactly fold_core (same two gemms, same values).
 */
inline void fold_core_block(nda::MemoryArrayOfRank<2> auto const& t_qP,
                            nda::MemoryArrayOfRank<2> auto const& t_qQ,
                            nda::MemoryArrayOfRank<2> auto const& A_PQ,
                            nda::array<ComplexType, 2>& tmp_mQ,
                            nda::MemoryArrayOfRank<2> auto&& out_mn) {
  nda::blas::gemm(t_qP, A_PQ, tmp_mQ);              // (N_m x P_bs)(P_bs x Q_bs) -> (N_m x Q_bs)
  nda::blas::gemm(tmp_mQ, nda::dagger(t_qQ), out_mn); // (N_m x Q_bs)(Q_bs x N_m) -> (N_m x N_m)
}

/**
 * Distributed downfold of the dynamic W (section 6b algorithm). Fills Wbar_qwmm
 * (nq, nw_b, N_m, N_m) with sum over the (P,Q) blocks of t Wdyn(q,nu) t^dag, computed
 * without materializing any full Np x Np slab. Template parameters keep the frequency
 * transform (tau -> nu PH-symmetric) and the Gamma-head insertion out of this header, so
 * the unit test can drive the SAME (P,Q)-split assembly + block-fold with deterministic
 * stand-in transforms and compare bit-exactly to a replicated reference.
 *
 *   dW           : distributed_array, global (nq, nt_half, Np, Np), grid {1,nt_procs,np_P,np_Q}
 *   t_qmP        : replicated (nq, N_m, Np) downfold maps
 *   nw_b/nw_half : full/positive bosonic mesh sizes; w_mirror_b: PH-unfold map (nw_b longs)
 *   iq_gamma     : Gamma q index; head_at_gamma: whether to add the head there
 *   head_add     : head_add(W_bt_block, it, P_range, Q_range) -- adds the tau-it head into
 *                  the assembled block slab in place (P_range/Q_range are this rank's block
 *                  ranges into the GLOBAL P,Q; called for it in [0, nt_half) at iq_gamma only)
 *   xform        : xform(W_bt_block (nt_half,P_bs,Q_bs), W_bw_block (nw_half,P_bs,Q_bs)) --
 *                  the tau->nu PH-symmetric transform on the assembled block
 *   Wbar_qwmm    : OUTPUT (nq, nw_b, N_m, N_m); zeroed and filled here
 *   comm         : the WHOLE communicator dW is distributed over
 *
 * The output is FULLY REPLICATED on comm (the kernel consumes it round-robin).
 */
template<typename dArray_t, typename TArr, typename HeadFn, typename XForm, typename OutArr>
void fold_dW_distributed(dArray_t const& dW,
                         TArr const& t_qmP,
                         long nqpts_ibz, long nt_half, long Np, long Nm,
                         long nw_b, long nw_half,
                         nda::array<long, 1> const& w_mirror_b,
                         long iq_gamma, bool head_at_gamma,
                         HeadFn&& head_add,
                         XForm&& xform,
                         OutArr&& Wbar_qwmm,
                         boost::mpi3::communicator& comm) {
  decltype(nda::range::all) all;
  auto gs = dW.global_shape();
  auto grd = dW.grid();
  utils::check(gs[0] == nqpts_ibz and gs[1] == nt_half and gs[2] == Np and gs[3] == Np,
               "vertex_secondary_detail::fold_dW_distributed: unexpected dW global shape "
               "({}, {}, {}, {}); expected ({}, {}, {}, {}).",
               gs[0], gs[1], gs[2], gs[3], nqpts_ibz, nt_half, Np, Np);
  utils::check(grd[0] == 1,
               "vertex_secondary_detail::fold_dW_distributed: the q axis of dW must NOT be "
               "split (grid[0] = {} != 1).", grd[0]);

  const auto org = dW.origin();          // {q_org(0), t_org, P_org, Q_org}
  const long t_org = org[1], P_org = org[2], Q_org = org[3];
  const auto lsh = dW.local_shape();     // {nq, t_bs, P_bs, Q_bs}
  const long t_bs = lsh[1], P_bs = lsh[2], Q_bs = lsh[3];
  const long np_Q = grd[3];
  nda::range P_range(P_org, P_org + P_bs);
  nda::range Q_range(Q_org, Q_org + Q_bs);

  // t_pool: ranks that share MY (P,Q) block (same color) but differ in t. On this pool the
  // t axis is a disjoint partition of [0, nt_half), so the zero-pad + all_reduce below is an
  // EXACT gather of the full-t slab for my block (no FP reassociation).
  boost::mpi3::communicator t_pool = comm.split(int(P_org * np_Q + Q_org), comm.rank());

  auto W_loc = dW.local();               // (nq, t_bs, P_bs, Q_bs)

  Wbar_qwmm = ComplexType(0.0);

  nda::array<ComplexType, 3> W_bt(nt_half, P_bs, Q_bs);   // my block, ALL t
  nda::array<ComplexType, 3> W_bw(nw_half, P_bs, Q_bs);   // my block, positive nu
  nda::array<ComplexType, 2> tmp_mQ(Nm, Q_bs);            // fold scratch

  for (long iq = 0; iq < nqpts_ibz; ++iq) {
    // assemble my (P,Q) block over ALL t: zero, drop in my t-range, reduce over t_pool.
    W_bt() = ComplexType(0.0);
    if (t_bs > 0)
      W_bt(nda::range(t_org, t_org + t_bs), all, all) = W_loc(iq, all, all, all);
    t_pool.all_reduce_in_place_n(W_bt.data(), W_bt.size(), std::plus<>{});

    // ONE rank per (P,Q) block folds (t_pool root); the others contribute zero to the
    // block partial, which the final comm all_reduce then sums.
    if (t_pool.rank() == 0) {
      // Gamma head into the assembled block BEFORE the tau -> nu transform (same order as
      // the legacy add_head_tau; block-sliced to my P_range/Q_range).
      if (head_at_gamma and iq == iq_gamma)
        for (long it = 0; it < nt_half; ++it)
          head_add(W_bt(it, all, all), it, P_range, Q_range);

      // tau -> nu (PH-symmetric, positive half) on the assembled block.
      xform(W_bt(), W_bw());

      auto t_q = t_qmP(iq, all, all);                 // (N_m x Np)
      auto t_qP = t_q(all, P_range);                  // (N_m x P_bs)
      auto t_qQ = t_q(all, Q_range);                  // (N_m x Q_bs)
      // PH-unfold the positive half to the full bosonic mesh (W(-nu) = W(nu)) and fold each l.
      for (long l = 0; l < nw_b; ++l) {
        const long lpos = std::max(l, w_mirror_b(l)) - nw_b / 2;
        fold_core_block(t_qP, t_qQ, W_bw(lpos, all, all), tmp_mQ,
                        Wbar_qwmm(iq, l, all, all));
      }
    }
  }

  // sum the (P,Q)-block partials -> the fully replicated downfolded rung.
  comm.all_reduce_in_place_n(Wbar_qwmm.data(), Wbar_qwmm.size(), std::plus<>{});
}

} // vertex_secondary_detail
} // solvers
} // methods

#endif // COQUI_VERTEX_SECONDARY_FOLD_HPP
