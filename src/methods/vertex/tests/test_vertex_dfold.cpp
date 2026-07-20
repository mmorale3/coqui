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
 * Impl 2 -- DISTRIBUTED downfold unit test (notes/vertex_parallelization_v2_plan.md
 * section "6b. Impl 2 distributed-downfold DESIGN").
 *
 * fold_dW_distributed folds the RPA (t,P,Q)-distributed dynamic W to the secondary aux
 * (N_m x N_m) WITHOUT gathering any full Np x Np block. The physics tests never exercise
 * the (P,Q) split -- at Si scale the RPA puts every rank on t (np_P = np_Q = 1). THIS test
 * FORCES a np_P,np_Q > 1 grid (4 ranks = 1 x 1 x 2 x 2) so the (P,Q)-block assembly and the
 * off-diagonal fold_core_block path are actually driven, and compares the distributed
 * result to a fully-REPLICATED reference (gather dW, tau->nu transform, PH-unfold, fold
 * each (q,l) with fold_core on the full Np x Np). The two must agree to <= 1e-12.
 *
 * At 1 rank the grid is 1x1x1x1 (one (P,Q) block == the whole array) -- the distributed
 * path reduces to the replicated fold and must be bit-identical. At 4 ranks the forced
 * 1x1x2x2 grid genuinely splits P and Q; each rank owns a P_bs x Q_bs block and the
 * off-diagonal blocks use fold_core_block(t(:,P_range), t(:,Q_range), ...).
 */

#undef NDEBUG

#include <vector>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/proc_grid_partition.hpp"

#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "utilities/test_common.hpp"

#include "methods/vertex/vertex_secondary_fold.hpp"

namespace bdft_tests {

using namespace math::nda;
template <int Rank> using shape_t = std::array<long, Rank>;

// deterministic, distribution-independent global tau-domain value dW(q,t,P,Q).
static ComplexType dwref(long q, long t, long P, long Q, long Np) {
  double re = 0.31 * double(q) + 0.17 * double(t)
            - 0.11 * double((P * Np + Q) % 13) + 0.07 * double((Q * 3 + P) % 11);
  double im = 0.13 * double((q * 5 + t) % 7) - 0.19 * double((P + 2 * Q) % 9)
            + 0.05 * double(t);
  return ComplexType(re, im);
}

// deterministic downfold map t(q)(m, P).
static ComplexType tref(long q, long m, long P) {
  double re = 0.4 * std::cos(0.37 * double(m) + 0.11 * double(P) - 0.23 * double(q));
  double im = 0.4 * std::sin(0.29 * double(P) - 0.19 * double(m) + 0.31 * double(q));
  return ComplexType(re, im);
}

// deterministic rank-1-free head H(P,Q) (added at q = iq_gamma, weight w_head(t)).
static ComplexType href(long P, long Q, long Np) {
  double re = 0.09 * double((P * 7 + Q) % 5) - 0.03 * double((Q + Np) % 4);
  double im = 0.06 * double((P + 3 * Q) % 6);
  return ComplexType(re, im);
}
static double whead(long t) { return 0.5 - 0.02 * double(t); }

// deterministic tau -> nu PH-symmetric transform matrix Twt(nw_half, nt_half).
static ComplexType twtref(long w, long t) {
  double re = 0.2 * std::cos(0.51 * double(w) - 0.13 * double(t));
  double im = 0.2 * std::sin(0.23 * double(t) + 0.17 * double(w));
  return ComplexType(re, im);
}

// replicated reference fold: out(m,n) = [t A t^dag](m,n) (the two-gemm fold_core the
// distributed path must reproduce; kept local so the test does not depend on the .cpp-only
// fold_core symbol -- only the header's fold_core_block / fold_dW_distributed are tested).
static void ref_fold(nda::array_view<ComplexType, 2> t_mP,
                     nda::array_view<ComplexType, 2> A_PQ,
                     nda::array_view<ComplexType, 2> out_mn) {
  const long Nm = t_mP.shape(0), Np = t_mP.shape(1);
  nda::array<ComplexType, 2> tmp(Nm, Np);
  nda::blas::gemm(t_mP, A_PQ, tmp);
  nda::blas::gemm(tmp, nda::dagger(t_mP), out_mn);
}

TEST_CASE("vertex_dfold_distributed", "[methods][vertex][dfold]") {
  using methods::solvers::vertex_secondary_detail::fold_dW_distributed;
  using methods::solvers::vertex_secondary_detail::fold_core_block;
  auto world = boost::mpi3::environment::get_world_instance();
  const long P = world.size();
  decltype(nda::range::all) all;

  // small dims (the point is the (P,Q) split, not scale). Np >= 2 so 1x1x2x2 is legal.
  const long nq = 3, nt_half = 4, nw_half = 4, nw_b = 2 * nw_half, Np = 6, Nm = 3;
  const long iq_gamma = 1;
  const bool head_at_gamma = true;

  // PH-unfold map: index layout [0, nw_half) = negative nu (mirror of the upper half),
  // [nw_half, nw_b) = non-negative nu, so nw_b/2 == nw_half and lpos = max(l, mirror(l)) -
  // nw_b/2 lands in [0, nw_half) for both halves (mirror(l) = nw_b - 1 - l).
  nda::array<long, 1> w_mirror_b(nw_b);
  for (long l = 0; l < nw_b; ++l) w_mirror_b(l) = nw_b - 1 - l;

  // replicated downfold maps t(q)(m, P).
  nda::array<ComplexType, 3> t_qmP(nq, Nm, Np);
  for (long q = 0; q < nq; ++q)
    for (long m = 0; m < Nm; ++m)
      for (long jp = 0; jp < Np; ++jp) t_qmP(q, m, jp) = tref(q, m, jp);

  // head + transform closures (identical arithmetic to the reference below). Shared by
  // every grid variant -- the distributed result must not depend on the layout.
  auto head_add = [&](nda::MemoryArrayOfRank<2> auto&& W_bt_block, long it,
                      nda::range const& P_rng, nda::range const& Q_rng) {
    for (long ip = P_rng.first(); ip < P_rng.last(); ++ip)
      for (long jq = Q_rng.first(); jq < Q_rng.last(); ++jq)
        W_bt_block(ip - P_rng.first(), jq - Q_rng.first()) +=
            ComplexType(whead(it)) * href(ip, jq, Np);
  };
  auto xform = [&](nda::MemoryArrayOfRank<3> auto&& W_bt_block,
                   nda::MemoryArrayOfRank<3> auto&& W_bw_block) {
    // W_bw(w,P,Q) = sum_t Twt(w,t) W_bt(t,P,Q); deterministic linear tau->nu stand-in.
    const long pbs = W_bt_block.shape(1), qbs = W_bt_block.shape(2);
    W_bw_block() = ComplexType(0.0);
    for (long w = 0; w < nw_half; ++w)
      for (long tt = 0; tt < nt_half; ++tt) {
        ComplexType c = twtref(w, tt);
        for (long ip = 0; ip < pbs; ++ip)
          for (long jq = 0; jq < qbs; ++jq)
            W_bw_block(w, ip, jq) += c * W_bt_block(tt, ip, jq);
      }
  };

  // --- REPLICATED reference: full Np x Np gather + head + xform + PH-unfold + fold ------
  nda::array<ComplexType, 4> Wbar_ref(nq, nw_b, Nm, Nm);
  Wbar_ref() = ComplexType(0.0);
  {
    // full tau-domain dW, replicated (small; reference only).
    nda::array<ComplexType, 4> W_qtPQ(nq, nt_half, Np, Np);
    for (long q = 0; q < nq; ++q)
      for (long t = 0; t < nt_half; ++t)
        for (long ip = 0; ip < Np; ++ip)
          for (long jq = 0; jq < Np; ++jq)
            W_qtPQ(q, t, ip, jq) = dwref(q, t, ip, jq, Np);
    // head at iq_gamma over all t (full Np x Np).
    if (head_at_gamma)
      for (long t = 0; t < nt_half; ++t)
        for (long ip = 0; ip < Np; ++ip)
          for (long jq = 0; jq < Np; ++jq)
            W_qtPQ(iq_gamma, t, ip, jq) += ComplexType(whead(t)) * href(ip, jq, Np);

    nda::array<ComplexType, 3> W_wpos(nw_half, Np, Np);
    for (long q = 0; q < nq; ++q) {
      // tau -> nu (same Twt as xform).
      W_wpos() = ComplexType(0.0);
      for (long w = 0; w < nw_half; ++w)
        for (long tt = 0; tt < nt_half; ++tt) {
          ComplexType c = twtref(w, tt);
          for (long ip = 0; ip < Np; ++ip)
            for (long jq = 0; jq < Np; ++jq)
              W_wpos(w, ip, jq) += c * W_qtPQ(q, tt, ip, jq);
        }
      auto t_q = t_qmP(q, all, all);
      for (long l = 0; l < nw_b; ++l) {
        const long lpos = std::max(l, w_mirror_b(l)) - nw_b / 2;
        ref_fold(t_q, W_wpos(lpos, all, all), Wbar_ref(q, l, all, all));
      }
    }
  }

  // --- run the DISTRIBUTED downfold on several forced grids, compare to the reference ---
  // The q axis is never split (grid[0] = 1). We exercise, for the current rank count P:
  //   - the mandated 1x1xnp_Pxnp_Q "(P,Q)-split" grid (the whole point of Impl 2), and
  //   - grids that ALSO split t (grid[1] > 1), driving the t-pool block-assembly all_reduce
  //     together with the (P,Q) fold.
  // Each grid axis must divide P and be <= its global extent (t <= nt_half, P/Q <= Np).
  using larray = nda::array<ComplexType, 4>;
  auto run_grid = [&](shape_t<4> grid) {
    auto dW = make_distributed_array<larray>(world, grid, {nq, nt_half, Np, Np}, {1, 1, 1, 1});
    auto loc = dW.local();
    auto [q0, t0, P0, Q0] = dW.origin();
    auto ls = dW.local_shape();
    for (long iq = 0; iq < ls[0]; ++iq)
      for (long it = 0; it < ls[1]; ++it)
        for (long ip = 0; ip < ls[2]; ++ip)
          for (long jq = 0; jq < ls[3]; ++jq)
            loc(iq, it, ip, jq) = dwref(q0 + iq, t0 + it, P0 + ip, Q0 + jq, Np);

    nda::array<ComplexType, 4> Wbar_dist(nq, nw_b, Nm, Nm);
    fold_dW_distributed(dW, t_qmP, nq, nt_half, Np, Nm, nw_b, nw_half, w_mirror_b,
                        iq_gamma, head_at_gamma, head_add, xform, Wbar_dist, world);

    double worst = 0.0;
    for (long q = 0; q < nq; ++q)
      for (long l = 0; l < nw_b; ++l)
        for (long m = 0; m < Nm; ++m)
          for (long n = 0; n < Nm; ++n)
            worst = std::max(worst,
                             std::abs(Wbar_dist(q, l, m, n) - Wbar_ref(q, l, m, n)));
    world.all_reduce_in_place_n(&worst, 1, boost::mpi3::max<>{});
    if (world.root())
      app_log(1, "vertex_dfold_distributed[P={} grid=(q,t,P,Q)=({},{},{},{})]: distributed "
                 "downfold == replicated reference (max|diff| = {:.3e}).",
              P, grid[0], grid[1], grid[2], grid[3], worst);
    return worst;
  };

  // enumerate legal grids {1, nt, nP, nQ} with nt*nP*nQ == P, nt <= nt_half, nP,nQ <= Np.
  std::vector<shape_t<4>> grids;
  for (long nt = 1; nt <= P; ++nt) {
    if (P % nt != 0 or nt > nt_half) continue;
    long rem = P / nt;
    for (long nP = 1; nP <= rem; ++nP) {
      if (rem % nP != 0) continue;
      long nQ = rem / nP;
      if (nP > Np or nQ > Np) continue;
      grids.push_back({1, nt, nP, nQ});
    }
  }
  REQUIRE(not grids.empty());   // at least the fully-P-split grid is always legal here
  // the mandated (P,Q)-split grid (t unsplit) must be present for P a product of <= Np^2.
  bool worst_ok = true;
  for (auto const& g : grids) worst_ok = worst_ok and (run_grid(g) <= 1e-12);
  REQUIRE(worst_ok);

  // fold_core_block sanity: on a DIAGONAL block (t_qP == t_qQ) it must equal the plain
  // two-gemm fold (same gemms, same order -> bit-identical).
  SECTION("fold_core_block_diag_equals_fold") {
    nda::array<ComplexType, 2> A(Np, Np);
    for (long ip = 0; ip < Np; ++ip)
      for (long jq = 0; jq < Np; ++jq) A(ip, jq) = dwref(0, 0, ip, jq, Np);
    auto t_q = t_qmP(0, all, all);
    nda::array<ComplexType, 2> tmpA(Nm, Np), o1(Nm, Nm), o2(Nm, Nm);
    ref_fold(t_q, A, o1);
    fold_core_block(t_q, t_q, A, tmpA, o2);
    double d = 0.0;
    for (long m = 0; m < Nm; ++m)
      for (long n = 0; n < Nm; ++n) d = std::max(d, std::abs(o1(m, n) - o2(m, n)));
    REQUIRE(d == 0.0);
  }
}

// deterministic, distribution-independent global bare-core value Z(q,P,Q). Reuses dwref at a
// fixed tau slice so the Z fold shares the harness's value generator (Z has no t axis).
static ComplexType zref(long q, long P, long Q, long Np) { return dwref(q, 0, P, Q, Np); }

/**
 * Impl 2b -- DISTRIBUTED downfold of the BARE core Z (fold_Z_distributed) unit test.
 * fold_Z_distributed folds the (q,P,Q)-distributed Z to the secondary aux (N_m x N_m) WITHOUT
 * gathering any full Np x Np block. The physics tests never exercise the (P,Q) split (at Si
 * scale the dZ grid is {1,1,1}). THIS test FORCES {1, nP, nQ} grids that split P and Q (4
 * ranks: 1x2x2, 1x1x4, 1x4x1) so the (P,Q)-block off-diagonal fold_core_block path is driven,
 * and compares to a fully-REPLICATED reference (gather Z, fold_core each q on the full Np x Np).
 * The two must agree to <= 1e-12. At 1 rank the grid is 1x1x1 (one (P,Q) block == the whole
 * array) and the distributed path reduces to the replicated fold BIT-IDENTICALLY.
 */
TEST_CASE("vertex_zfold_distributed", "[methods][vertex][dfold]") {
  using methods::solvers::vertex_secondary_detail::fold_Z_distributed;
  auto world = boost::mpi3::environment::get_world_instance();
  const long P = world.size();
  decltype(nda::range::all) all;

  const long nq = 3, Np = 6, Nm = 3;
  const long iq_gamma = 1;
  const bool head_at_gamma = true;

  // replicated downfold maps t(q)(m, P).
  nda::array<ComplexType, 3> t_qmP(nq, Nm, Np);
  for (long q = 0; q < nq; ++q)
    for (long m = 0; m < Nm; ++m)
      for (long jp = 0; jp < Np; ++jp) t_qmP(q, m, jp) = tref(q, m, jp);

  // head_add closure: adds the head BLOCK H(P,Q) into the gamma block in place (bare piece,
  // weight 1 -- mirrors the production Z(iq_gamma) += H_PQ). Layout-independent.
  auto head_add = [&](nda::MemoryArrayOfRank<2> auto&& A_PQ_block,
                      nda::range const& P_rng, nda::range const& Q_rng) {
    for (long ip = P_rng.first(); ip < P_rng.last(); ++ip)
      for (long jq = Q_rng.first(); jq < Q_rng.last(); ++jq)
        A_PQ_block(ip - P_rng.first(), jq - Q_rng.first()) += href(ip, jq, Np);
  };

  // --- REPLICATED reference: full Np x Np gather + head + fold_core each q -----------------
  nda::array<ComplexType, 3> Zbar_ref(nq, Nm, Nm);
  Zbar_ref() = ComplexType(0.0);
  {
    nda::array<ComplexType, 2> A(Np, Np);
    for (long q = 0; q < nq; ++q) {
      for (long ip = 0; ip < Np; ++ip)
        for (long jq = 0; jq < Np; ++jq) A(ip, jq) = zref(q, ip, jq, Np);
      if (head_at_gamma and q == iq_gamma)
        for (long ip = 0; ip < Np; ++ip)
          for (long jq = 0; jq < Np; ++jq) A(ip, jq) += href(ip, jq, Np);
      ref_fold(t_qmP(q, all, all), A, Zbar_ref(q, all, all));
    }
  }

  // --- run the DISTRIBUTED downfold on several forced {1, nP, nQ} grids (q NEVER split) ----
  using larray = nda::array<ComplexType, 3>;
  auto run_grid = [&](shape_t<3> grid) {
    auto dZ = make_distributed_array<larray>(world, grid, {nq, Np, Np}, {1, 1, 1});
    auto loc = dZ.local();
    auto [q0, P0, Q0] = dZ.origin();
    auto ls = dZ.local_shape();
    for (long iq = 0; iq < ls[0]; ++iq)
      for (long ip = 0; ip < ls[1]; ++ip)
        for (long jq = 0; jq < ls[2]; ++jq)
          loc(iq, ip, jq) = zref(q0 + iq, P0 + ip, Q0 + jq, Np);

    nda::array<ComplexType, 3> Zbar_dist(nq, Nm, Nm);
    fold_Z_distributed(dZ, t_qmP, nq, Np, Nm, iq_gamma, head_at_gamma, head_add,
                       Zbar_dist, world);

    double worst = 0.0;
    for (long q = 0; q < nq; ++q)
      for (long m = 0; m < Nm; ++m)
        for (long n = 0; n < Nm; ++n)
          worst = std::max(worst, std::abs(Zbar_dist(q, m, n) - Zbar_ref(q, m, n)));
    world.all_reduce_in_place_n(&worst, 1, boost::mpi3::max<>{});
    if (world.root())
      app_log(1, "vertex_zfold_distributed[P={} grid=(q,P,Q)=({},{},{})]: distributed "
                 "downfold == replicated reference (max|diff| = {:.3e}).",
              P, grid[0], grid[1], grid[2], worst);
    return worst;
  };

  // enumerate legal grids {1, nP, nQ} with nP*nQ == P, nP,nQ <= Np. At 4 ranks this is
  // exactly {1,2,2}, {1,1,4}, {1,4,1} -- the mandated (P,Q)-split coverage.
  std::vector<shape_t<3>> grids;
  for (long nP = 1; nP <= P; ++nP) {
    if (P % nP != 0) continue;
    long nQ = P / nP;
    if (nP > Np or nQ > Np) continue;
    grids.push_back({1, nP, nQ});
  }
  REQUIRE(not grids.empty());
  bool worst_ok = true;
  for (auto const& g : grids) worst_ok = worst_ok and (run_grid(g) <= 1e-12);
  REQUIRE(worst_ok);
}

} // namespace bdft_tests
