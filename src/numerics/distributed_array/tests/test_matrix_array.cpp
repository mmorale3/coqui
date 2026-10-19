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

#undef NDEBUG

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "utilities/proc_grid_partition.hpp"

#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/matrix_array.hpp"

/*
 * distributed_matrix_array is address-space generic: the block-cyclic distribution is a
 * better layout for SLATE on the HOST as well as on the device, and only the tile-stride
 * requirement is device-specific. So every test body below is templated on MEMORY_SPACE and
 * instantiated for both HOST_MEMORY and DEVICE_MEMORY.
 *
 * Element access always goes through a host staging tile which is assigned to/from the
 * container's tile view; nda turns that into a cross-address-space copy when needed. Direct
 * host element access into a device tile would be illegal, so the staging keeps one test body
 * valid in both spaces.
 */

namespace bdft_tests
{

using namespace math::nda;
using boost::mpi3::communicator;

using host_tile_t = nda::array<double, 2, nda::F_layout>;

// value stored at global position (batch..., I, J); distinct for every element
template<int BR>
double ref_value(std::array<long,BR> const& b, long I, long J, long N)
{
  double s = 0.0;
  for (int d = 0; d < BR; ++d) s = s*1000.0 + double(b[d] + 1);
  return s*1.0e6 + double(I)*double(N) + double(J) + 1.0;
}

// factor np into (p,q) as squarely as possible
inline std::pair<long,long> pq_of(long np) {
  long p = 1;
  for (long c = long(std::sqrt(double(np))); c >= 1; --c)
    if (np % c == 0) { p = c; break; }
  return {p, np/p};
}

/*
 * Ownership must agree with slate's own tileIsLocal for every tile, and the tile counts must
 * use slate's ceil convention. Shapes deliberately include M%mb != 0 and N%nb != 0, which is
 * where slate_aux.hpp's mt = M/mb (floor) goes wrong.
 */
template<MEMORY_SPACE MEM>
void test_ownership()
{
  auto world = boost::mpi3::environment::get_world_instance();
  using darr_t = memory::dmatrix_array_t<MEM, double, 0, communicator>;

  struct Case { long M, N, mb, nb; };
  std::vector<Case> cases = {
    {64, 64, 16, 16},   // divisible
    {70, 64, 16, 16},   // M % mb != 0
    {64, 70, 16, 16},   // N % nb != 0
    {70, 66, 16, 8},    // both, different mb/nb
    {17, 13,  4,  3},   // small, many partial tiles
  };

  // exercise several (p,q) factorizations, not just the squarest one: p==1 or q==1 hides
  // grid-order bugs, so a p=q=2 style split must be covered when the rank count allows it
  std::vector<std::pair<long,long>> grids;
  for (long p = 1; p <= world.size(); ++p)
    if (world.size() % p == 0) grids.push_back({p, world.size()/p});

  for (auto [p, q] : grids) {
    for (auto c : cases) {
      if (c.M < p or c.N < q) continue;
      darr_t A(world, {p, q}, {c.M, c.N}, {c.mb, c.nb});

      REQUIRE(A.mt() == (c.M + c.mb - 1)/c.mb);   // ceil, not floor
      REQUIRE(A.nt() == (c.N + c.nb - 1)/c.nb);
#if defined(ENABLE_SLATE)
      REQUIRE(A.ownership_matches_slate());
#endif
      // every tile owned by exactly one rank
      long ntot = A.n_local_tile_rows()*A.n_local_tile_cols();
      world.all_reduce_in_place_n(&ntot, 1, std::plus<>{});
      REQUIRE(ntot == A.mt()*A.nt());
    }
  }
}

/*
 * Tile-major layout: writing through tile() and reading back must round-trip, every local
 * tile must be contiguous with ld == its own number of rows (that is what makes slate's
 * batched path accept these tiles), and the local tiles must together cover the global
 * matrix exactly once.
 */
template<MEMORY_SPACE MEM>
void test_layout()
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto [p, q] = pq_of(world.size());

  using darr_t = memory::dmatrix_array_t<MEM, double, 0, communicator>;
  const long M = 70, N = 66, mb = 16, nb = 8;
  if (M < p or N < q) return;

  darr_t A(world, {p, q}, {M, N}, {mb, nb});
  std::array<long,0> b0{};

  // fill: build each tile on the host, then assign into the container's tile
  for (long a = 0; a < A.n_local_tile_rows(); ++a) {
    for (long bb = 0; bb < A.n_local_tile_cols(); ++bb) {
      auto t = A.tile(0, a, bb);
      long i0 = A.local_tile_row(a)*mb, j0 = A.local_tile_col(bb)*nb;

      // tile must be contiguous column-major with ld == number of rows
      REQUIRE(t.indexmap().strides()[0] == 1);
      REQUIRE(t.indexmap().strides()[1] == t.extent(0));

      host_tile_t h(t.extent(0), t.extent(1));
      for (long jj = 0; jj < h.extent(1); ++jj)
        for (long ii = 0; ii < h.extent(0); ++ii)
          h(ii,jj) = ref_value<0>(b0, i0+ii, j0+jj, N);
      t = h;
    }
  }

  // read back
  long checked = 0;
  for (long a = 0; a < A.n_local_tile_rows(); ++a) {
    for (long bb = 0; bb < A.n_local_tile_cols(); ++bb) {
      auto t = A.tile(0, a, bb);
      long i0 = A.local_tile_row(a)*mb, j0 = A.local_tile_col(bb)*nb;
      host_tile_t h(t.extent(0), t.extent(1));
      h = t;
      for (long jj = 0; jj < h.extent(1); ++jj)
        for (long ii = 0; ii < h.extent(0); ++ii) {
          REQUIRE(h(ii,jj) == ref_value<0>(b0, i0+ii, j0+jj, N));
          ++checked;
        }
    }
  }
  world.all_reduce_in_place_n(&checked, 1, std::plus<>{});
  REQUIRE(checked == M*N);

  // tiles must not overlap
  long used = 0;
  for (long a = 0; a < A.n_local_tile_rows(); ++a)
    for (long bb = 0; bb < A.n_local_tile_cols(); ++bb)
      used += A.tile(0,a,bb).size();
  REQUIRE(A.buffer_size() >= used);

  // set() must reach the whole buffer in either address space
  A.set(-7.0);
  {
    auto t = A.tile(0, 0, 0);
    host_tile_t h(t.extent(0), t.extent(1));
    h = t;
    REQUIRE(h(0,0) == -7.0);
    REQUIRE(h(h.extent(0)-1, h.extent(1)-1) == -7.0);
  }
}

/*
 * Batch dimensions: each batch element keeps its own data, and the local batch ranges
 * partition the global batch extent.
 */
template<MEMORY_SPACE MEM>
void test_batch()
{
  auto world = boost::mpi3::environment::get_world_instance();
  const long M = 40, N = 36, mb = 8, nb = 8;

  long nb_dim = (world.size() % 2 == 0 ? 2 : 1);
  long nmat = world.size()/nb_dim;
  auto [p, q] = pq_of(nmat);

  using darr_t = memory::dmatrix_array_t<MEM, double, 1, communicator>;
  const long NB = 2*nb_dim;   // batch extent, divisible by its grid
  darr_t A(world, {nb_dim, p, q}, {NB, M, N}, {mb, nb});

  REQUIRE(A.n_local_batch() == NB/nb_dim);

  for (long ib = 0; ib < A.n_local_batch(); ++ib) {
    auto bidx = A.local_batch_index(ib);
    for (long a = 0; a < A.n_local_tile_rows(); ++a)
      for (long bb = 0; bb < A.n_local_tile_cols(); ++bb) {
        auto t = A.tile(ib, a, bb);
        long i0 = A.local_tile_row(a)*mb, j0 = A.local_tile_col(bb)*nb;
        host_tile_t h(t.extent(0), t.extent(1));
        for (long jj = 0; jj < h.extent(1); ++jj)
          for (long ii = 0; ii < h.extent(0); ++ii)
            h(ii,jj) = ref_value<1>(bidx, i0+ii, j0+jj, N);
        t = h;
      }
  }
  for (long ib = 0; ib < A.n_local_batch(); ++ib) {
    auto bidx = A.local_batch_index(ib);
    for (long a = 0; a < A.n_local_tile_rows(); ++a)
      for (long bb = 0; bb < A.n_local_tile_cols(); ++bb) {
        auto t = A.tile(ib, a, bb);
        long i0 = A.local_tile_row(a)*mb, j0 = A.local_tile_col(bb)*nb;
        host_tile_t h(t.extent(0), t.extent(1));
        h = t;
        for (long jj = 0; jj < h.extent(1); ++jj)
          for (long ii = 0; ii < h.extent(0); ++ii)
            REQUIRE(h(ii,jj) == ref_value<1>(bidx, i0+ii, j0+jj, N));
      }
  }

  // Every rank in a batch cell holds the SAME batch range (the matrix grid subdivides the
  // matrix, not the batch), so a global sum counts each batch element p*q times.
  long lo = A.local_range(0).first(), n = A.local_range(0).size();
  REQUIRE(lo == A.origin()[0]);
  long total = n;
  world.all_reduce_in_place_n(&total, 1, std::plus<>{});
  REQUIRE(total == NB * p * q);

  // counted once per batch cell it must be exactly NB
  long once = (A.matrix_communicator()->rank() == 0 ? n : 0);
  world.all_reduce_in_place_n(&once, 1, std::plus<>{});
  REQUIRE(once == NB);
}

TEST_CASE("matrix_array_ownership_host", "[math][matrix_array]") { test_ownership<HOST_MEMORY>(); }
TEST_CASE("matrix_array_layout_host",    "[math][matrix_array]") { test_layout<HOST_MEMORY>(); }
TEST_CASE("matrix_array_batch_host",     "[math][matrix_array]") { test_batch<HOST_MEMORY>(); }

TEST_CASE("matrix_array_ownership_dev",  "[math][matrix_array]") { test_ownership<DEVICE_MEMORY>(); }
TEST_CASE("matrix_array_layout_dev",     "[math][matrix_array]") { test_layout<DEVICE_MEMORY>(); }
TEST_CASE("matrix_array_batch_dev",      "[math][matrix_array]") { test_batch<DEVICE_MEMORY>(); }

} // namespace bdft_tests
