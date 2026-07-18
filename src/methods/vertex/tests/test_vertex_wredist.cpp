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
 * Vertex parallelization MILESTONE 1, change-list item #1 -- the W-redistribution
 * round-trip test (notes/vertex_parallelization_analysis.md section 4.2 #4;
 * notes/vertex_parallelization_M1.md).
 *
 * The dynamic screened interaction dW lives on the RPA (t,q,P,Q) proc grid
 * (MBState::dW_qtPQ; the vertex slab has axis order (q,t,P,Q)). M2 (change-list items
 * #3/#4) will consume dW as a q-OWNED layout via the proven math::nda::redistribute
 * (nda_utils.hpp:866 -> redistribute_alltoallv). This test pins that the redistribution
 * is PURE DATA MOVEMENT and therefore BIT-IDENTICAL:
 *
 *   dW on (q,t,P,Q)  --redistribute-->  q-owned  --redistribute-->  back
 *
 * must reproduce the original array element-for-element (exact ==), on any rank count.
 * This is the infrastructure guarantee the M2 kernel re-parallelization rests on.
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/proc_grid_partition.hpp"

#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "utilities/test_common.hpp"

namespace bdft_tests {

using namespace math::nda;
template <int Rank> using shape_t = std::array<long, Rank>;

// deterministic, rank-independent global value for element (q,t,P,Q): the round-trip
// must return exactly this regardless of how the array was distributed.
static ComplexType wref(long q, long t, long P, long Q, long Np) {
  double re = 0.5 * double(q) + 0.25 * double(t) - 0.125 * double(P) + 0.0625 * double(Q);
  double im = 0.1 * double((q * Np + P) % 7) - 0.3 * double((t * Np + Q) % 5);
  return ComplexType(re, im);
}

TEST_CASE("vertex_wredist_roundtrip", "[methods][vertex][wredist]") {
  auto world = boost::mpi3::environment::get_world_instance();
  const long P = world.size();

  // Mimic the vertex dW slab (q, nt_half, Np, Np). Keep every axis >= P so any single
  // axis can absorb all ranks -- the grids below are then always legal.
  const long nq = std::max(8L, P), nt_half = std::max(8L, P), Np = std::max(6L, P);

  using larray = nda::array<ComplexType, 4>;

  auto make = [&](shape_t<4> grid) {
    return make_distributed_array<larray>(world, grid, {nq, nt_half, Np, Np},
                                          {1, 1, 1, 1});
  };
  auto fill = [&](auto& A) {
    auto loc = A.local();
    auto [q0, t0, P0, Q0] = A.origin();
    auto ls = A.local_shape();
    for (long iq = 0; iq < ls[0]; ++iq)
      for (long it = 0; it < ls[1]; ++it)
        for (long ip = 0; ip < ls[2]; ++ip)
          for (long jq = 0; jq < ls[3]; ++jq)
            loc(iq, it, ip, jq) = wref(q0 + iq, t0 + it, P0 + ip, Q0 + jq, Np);
  };
  auto check_owned = [&](auto& A) {
    auto loc = A.local();
    auto [q0, t0, P0, Q0] = A.origin();
    auto ls = A.local_shape();
    long bad = 0;
    for (long iq = 0; iq < ls[0]; ++iq)
      for (long it = 0; it < ls[1]; ++it)
        for (long ip = 0; ip < ls[2]; ++ip)
          for (long jq = 0; jq < ls[3]; ++jq)
            if (loc(iq, it, ip, jq) != wref(q0 + iq, t0 + it, P0 + ip, Q0 + jq, Np)) ++bad;
    return bad;
  };

  // --- source: the RPA Pi/W grid. t pooled (axis 1), q unsplit, PxQ split (axes 2,3).
  long np_P = utils::find_proc_grid_min_diff(P, 1, 1);
  long np_Q = P / np_P;
  shape_t<4> src_grid{1, 1, np_P, np_Q};   // (q,t,P,Q): mirrors rpa_pi.icc PxQ split
  REQUIRE(src_grid[0] * src_grid[1] * src_grid[2] * src_grid[3] == P);
  INFO("P=" << P << " src grid (q,t,P,Q)=(1,1," << np_P << "," << np_Q << ")");

  auto A = make(src_grid);
  fill(A);

  SECTION("q_owned_roundtrip") {
    // q-owned layout: split ONLY the q axis across all ranks (nq >= P).
    shape_t<4> qgrid{P, 1, 1, 1};
    auto B = make(qgrid);
    B.local() = ComplexType(0.0);
    redistribute(A, B);                       // A (RPA grid) -> B (q-owned)
    REQUIRE(check_owned(B) == 0);             // B holds exactly wref on its q rows

    auto C = make(src_grid);
    C.local() = ComplexType(0.0);
    redistribute(B, C);                       // B (q-owned) -> C (== A layout)

    auto la = A.local();
    auto lc = C.local();
    REQUIRE(la.shape() == lc.shape());
    long ndiff = 0;
    for (long i = 0; i < la.size(); ++i)
      if (la.data()[i] != lc.data()[i]) ++ndiff;   // EXACT bitwise equality
    world.all_reduce_in_place_n(&ndiff, 1, std::plus<>{});
    REQUIRE(ndiff == 0);
    if (world.root())
      app_log(1, "vertex_wredist_roundtrip[q_owned]: RPA-grid dW -> q-owned -> back "
                 "exact on {} ranks (0 differing elements).", P);
  }

  SECTION("t_pooled_roundtrip") {
    // an alternative M2 target -- t-pooled (axis 1). Also pure data movement.
    shape_t<4> tgrid{1, P, 1, 1};
    auto B = make(tgrid);
    B.local() = ComplexType(0.0);
    redistribute(A, B);
    REQUIRE(check_owned(B) == 0);

    auto C = make(src_grid);
    C.local() = ComplexType(0.0);
    redistribute(B, C);
    auto la = A.local();
    auto lc = C.local();
    long ndiff = 0;
    for (long i = 0; i < la.size(); ++i)
      if (la.data()[i] != lc.data()[i]) ++ndiff;
    world.all_reduce_in_place_n(&ndiff, 1, std::plus<>{});
    REQUIRE(ndiff == 0);
  }

  // ---- M3 item #8: reduce-scatter of a PARTIAL into the RPA grid ---------------------
  // The Pi^C kernel produces a PARTIAL replicated array on every rank (its round-robin
  // tuple/q_ext contribution to the WHOLE (t,q,P,Q) array). reduce_scatter_into sums the
  // partials and lands each output block ONLY on its owner. This test replicates that
  // exact composition (per-owner MPI_Reduce over gathered block boxes) and pins it against
  // the serial reference: sum over all ranks of the partials == all_reduce of the partial.
  // At P=1 it is a bit-identical copy; at P>1 the reduce-scatter must reproduce the
  // all_reduce result on the owned block (a pure integer/exact-sum data movement here).
  SECTION("reduce_scatter_into_rpa_grid") {
    // rank-dependent PARTIAL: part_r(q,t,P,Q) = wref(...) * (r+1). The true sum over all
    // ranks is wref * sum_{r=0}^{P-1}(r+1) = wref * P*(P+1)/2 -- an exact reference.
    const long r = world.rank();
    nda::array<ComplexType, 4> part(nq, nt_half, Np, Np);
    for (long iq = 0; iq < nq; ++iq)
      for (long it = 0; it < nt_half; ++it)
        for (long ip = 0; ip < Np; ++ip)
          for (long jq = 0; jq < Np; ++jq)
            part(iq, it, ip, jq) =
                wref(iq, it, ip, jq, Np) * ComplexType(double(r + 1));
    const double sum_scale = double(P) * double(P + 1) / 2.0;   // sum_{r=1}^{P}(r)

    // target on the RPA grid (t pooled, q unsplit, PxQ split) -- exactly eval_Pi_C's dPi_C.
    auto dPi = make(src_grid);
    dPi.local() = ComplexType(0.0);

    // === the reduce_scatter_into composition (mirrors vertex_redist_detail) ===
    std::array<long, 8> mine{};
    for (int d = 0; d < 4; ++d) { mine[d] = dPi.origin()[d]; mine[4 + d] = dPi.local_shape()[d]; }
    nda::array<long, 2> boxes(P, 8);
    world.all_gather_n(mine.data(), 8, boxes.data());
    for (long rr = 0; rr < P; ++rr) {
      nda::range b0(boxes(rr, 0), boxes(rr, 0) + boxes(rr, 4));
      nda::range b1(boxes(rr, 1), boxes(rr, 1) + boxes(rr, 5));
      nda::range b2(boxes(rr, 2), boxes(rr, 2) + boxes(rr, 6));
      nda::range b3(boxes(rr, 3), boxes(rr, 3) + boxes(rr, 7));
      nda::array<ComplexType, 4> buf = part(b0, b1, b2, b3);
      world.reduce_in_place_n(buf.data(), buf.size(), std::plus<>{}, int(rr));
      if (rr == r) dPi.local() = buf;
    }

    // verify the owned block == wref * sum_scale (exact reduce-scatter result).
    auto loc = dPi.local();
    auto [q0, t0, P0, Q0] = dPi.origin();
    auto ls = dPi.local_shape();
    long bad = 0;
    double worst = 0.0;
    for (long iq = 0; iq < ls[0]; ++iq)
      for (long it = 0; it < ls[1]; ++it)
        for (long ip = 0; ip < ls[2]; ++ip)
          for (long jq = 0; jq < ls[3]; ++jq) {
            ComplexType ref = wref(q0 + iq, t0 + it, P0 + ip, Q0 + jq, Np) *
                              ComplexType(sum_scale);
            double d = std::abs(loc(iq, it, ip, jq) - ref);
            worst = std::max(worst, d);
            if (d > 1e-11 * std::max(std::abs(ref), 1e-30)) ++bad;
          }
    world.all_reduce_in_place_n(&bad, 1, std::plus<>{});
    world.all_reduce_in_place_n(&worst, 1, boost::mpi3::max<>{});
    REQUIRE(bad == 0);
    if (world.root())
      app_log(1, "vertex_reduce_scatter[P={}]: partial (rank-scaled) reduce-scattered into "
                 "the RPA grid == serial sum on every owned block (max|diff| = {}).",
              P, worst);
  }
}

} // namespace bdft_tests
