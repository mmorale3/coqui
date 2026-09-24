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

#include <iostream>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"

#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/lapack.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/matrix_array.hpp"
#include "numerics/distributed_array/slate_ops_matrix_array.hpp"
#include "numerics/distributed_array/matrix_array_redistribute.hpp"
#include "numerics/distributed_array/slate_ops.hpp"
#include "numerics/distributed_array/ops.hpp"

/*
 * Linear-algebra operations on distributed_matrix_array, checked against a serial reference
 * computed redundantly on every rank. Each test body is templated on MEMORY_SPACE and run for
 * HOST_MEMORY and DEVICE_MEMORY: the block-cyclic layout benefits CPU runs too, so both are
 * first-class.
 */

namespace bdft_tests
{

using namespace math::nda;
using boost::mpi3::communicator;

using cvalue  = ComplexType;
using hmat_t  = nda::array<cvalue, 2, nda::F_layout>;   // serial reference, column-major

// deterministic pseudo-random, identical on every rank
inline cvalue gen(long i, long j, long seed)
{
  auto h = [](long x) { x = (x ^ 61) ^ (x >> 16); x *= 9; x ^= x >> 4; x *= 0x27d4eb2d;
                        x ^= x >> 15; return x; };
  double a = double(h(i*7919 + j*104729 + seed*1299709) % 2000 - 1000)/1000.0;
  double b = double(h(i*104729 + j*7919 + seed*15485863) % 2000 - 1000)/1000.0;
  return cvalue(a, b);
}

inline std::pair<long,long> pq_of(long np) {
  long p = 1;
  for (long c = long(std::sqrt(double(np))); c >= 1; --c)
    if (np % c == 0) { p = c; break; }
  return {p, np/p};
}

// ---- helpers moving whole matrices between the container and a serial reference -----------

template<typename DA>
void scatter_into(DA& A, hmat_t const& src, long ib = 0)
{
  long mb = A.mb(), nb = A.nb();
  for (long a = 0; a < A.n_local_tile_rows(); ++a)
    for (long b = 0; b < A.n_local_tile_cols(); ++b) {
      auto t = A.tile(ib, a, b);
      long i0 = A.local_tile_row(a)*mb, j0 = A.local_tile_col(b)*nb;
      hmat_t h(t.extent(0), t.extent(1));
      for (long jj = 0; jj < h.extent(1); ++jj)
        for (long ii = 0; ii < h.extent(0); ++ii) h(ii,jj) = src(i0+ii, j0+jj);
      t = h;
    }
}

/// gather the distributed matrix onto every rank (reference comparison only)
template<typename DA>
hmat_t gather_all(DA& A, communicator& comm, long ib = 0)
{
  long M = A.global_shape()[std::decay_t<DA>::rank-2];
  long N = A.global_shape()[std::decay_t<DA>::rank-1];
  hmat_t out(M, N);
  out() = cvalue(0.0);
  long mb = A.mb(), nb = A.nb();
  for (long a = 0; a < A.n_local_tile_rows(); ++a)
    for (long b = 0; b < A.n_local_tile_cols(); ++b) {
      auto t = A.tile(ib, a, b);
      long i0 = A.local_tile_row(a)*mb, j0 = A.local_tile_col(b)*nb;
      hmat_t h(t.extent(0), t.extent(1));
      h = t;
      for (long jj = 0; jj < h.extent(1); ++jj)
        for (long ii = 0; ii < h.extent(0); ++ii) out(i0+ii, j0+jj) = h(ii,jj);
    }
  // each element is written by exactly one rank, so a sum reproduces the whole matrix.
  // Batch cells hold the same batch elements, so restrict the sum to one cell.
  if (A.matrix_communicator()->size() != comm.size()) {
    A.matrix_communicator()->all_reduce_in_place_n(out.data(), out.size(), std::plus<>{});
  } else {
    comm.all_reduce_in_place_n(out.data(), out.size(), std::plus<>{});
  }
  return out;
}

inline double max_abs_diff(hmat_t const& X, hmat_t const& Y)
{
  double d = 0.0;
  for (long j = 0; j < X.extent(1); ++j)
    for (long i = 0; i < X.extent(0); ++i)
      d = std::max(d, std::abs(X(i,j) - Y(i,j)));
  return d;
}

/***************************************************************************/
/*                                 gemm                                    */
/***************************************************************************/
template<MEMORY_SPACE MEM>
void test_multiply()
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto [p, q] = pq_of(world.size());
  using darr_t = memory::dmatrix_array_t<MEM, cvalue, 0, communicator>;

  // non-square, non-divisible on purpose
  const long M = 40, K = 28, N = 36, mb = 8, nb = 8;
  if (M < p or N < q or K < p or K < q) return;

  hmat_t Aref(M,K), Bref(K,N), ATref(K,M), BTref(N,K);
  for (long j = 0; j < K; ++j) for (long i = 0; i < M; ++i) Aref(i,j) = gen(i,j,1);
  for (long j = 0; j < N; ++j) for (long i = 0; i < K; ++i) Bref(i,j) = gen(i,j,2);
  for (long j = 0; j < M; ++j) for (long i = 0; i < K; ++i) ATref(i,j) = gen(i,j,3);
  for (long j = 0; j < K; ++j) for (long i = 0; i < N; ++i) BTref(i,j) = gen(i,j,4);

  auto ref_gemm = [](hmat_t const& X, hmat_t const& Y) {
    hmat_t Z(X.extent(0), Y.extent(1));
    Z() = cvalue(0.0);
    nda::blas::gemm(cvalue(1.0), X, Y, cvalue(0.0), Z);
    return Z;
  };
  auto conj_t = [](hmat_t const& X) {
    hmat_t Y(X.extent(1), X.extent(0));
    for (long j = 0; j < X.extent(1); ++j)
      for (long i = 0; i < X.extent(0); ++i) Y(j,i) = std::conj(X(i,j));
    return Y;
  };
  auto trans = [](hmat_t const& X) {
    hmat_t Y(X.extent(1), X.extent(0));
    for (long j = 0; j < X.extent(1); ++j)
      for (long i = 0; i < X.extent(0); ++i) Y(j,i) = X(i,j);
    return Y;
  };

  // ---- N/N : C(MxN) = A(MxK) * B(KxN)
  {
    darr_t dA(world, {p,q}, {M,K}, {mb,nb});
    darr_t dB(world, {p,q}, {K,N}, {mb,nb});
    darr_t dC(world, {p,q}, {M,N}, {mb,nb});
    scatter_into(dA, Aref); scatter_into(dB, Bref);
    slate_ops::multiply(dA, dB, dC);
    auto got = gather_all(dC, world);
    REQUIRE(max_abs_diff(got, ref_gemm(Aref,Bref)) < 1e-11);
  }
  // ---- C/N : C(MxN) = A^H(MxK) * B(KxN), with A stored KxM
  {
    darr_t dA(world, {p,q}, {K,M}, {mb,nb});
    darr_t dB(world, {p,q}, {K,N}, {mb,nb});
    darr_t dC(world, {p,q}, {M,N}, {mb,nb});
    scatter_into(dA, ATref); scatter_into(dB, Bref);
    slate_ops::multiply(dagger(dA), dB, dC);
    auto got = gather_all(dC, world);
    REQUIRE(max_abs_diff(got, ref_gemm(conj_t(ATref), Bref)) < 1e-11);
  }
  // ---- N/C : C(MxN) = A(MxK) * B^H(KxN), with B stored NxK
  {
    darr_t dA(world, {p,q}, {M,K}, {mb,nb});
    darr_t dB(world, {p,q}, {N,K}, {mb,nb});
    darr_t dC(world, {p,q}, {M,N}, {mb,nb});
    scatter_into(dA, Aref); scatter_into(dB, BTref);
    slate_ops::multiply(dA, dagger(dB), dC);
    auto got = gather_all(dC, world);
    REQUIRE(max_abs_diff(got, ref_gemm(Aref, conj_t(BTref))) < 1e-11);
  }
  // ---- T/N : C(MxN) = A^T(MxK) * B(KxN)
  {
    darr_t dA(world, {p,q}, {K,M}, {mb,nb});
    darr_t dB(world, {p,q}, {K,N}, {mb,nb});
    darr_t dC(world, {p,q}, {M,N}, {mb,nb});
    scatter_into(dA, ATref); scatter_into(dB, Bref);
    slate_ops::multiply(transpose(dA), dB, dC);
    auto got = gather_all(dC, world);
    REQUIRE(max_abs_diff(got, ref_gemm(trans(ATref), Bref)) < 1e-11);
  }
  // ---- alpha/beta accumulation
  {
    darr_t dA(world, {p,q}, {M,K}, {mb,nb});
    darr_t dB(world, {p,q}, {K,N}, {mb,nb});
    darr_t dC(world, {p,q}, {M,N}, {mb,nb});
    hmat_t Cref(M,N);
    for (long j = 0; j < N; ++j) for (long i = 0; i < M; ++i) Cref(i,j) = gen(i,j,5);
    scatter_into(dA, Aref); scatter_into(dB, Bref); scatter_into(dC, Cref);
    slate_ops::multiply(cvalue(2.0), dA, dB, cvalue(-3.0), dC);
    auto got = gather_all(dC, world);
    auto AB = ref_gemm(Aref, Bref);
    hmat_t want(M,N);
    for (long j = 0; j < N; ++j) for (long i = 0; i < M; ++i)
      want(i,j) = cvalue(2.0)*AB(i,j) + cvalue(-3.0)*Cref(i,j);
    REQUIRE(max_abs_diff(got, want) < 1e-11);
  }
}

/***************************************************************************/
/*                        batched gemm over batch dims                     */
/***************************************************************************/
template<MEMORY_SPACE MEM>
void test_multiply_batched()
{
  auto world = boost::mpi3::environment::get_world_instance();
  long nbd = (world.size() % 2 == 0 ? 2 : 1);
  auto [p, q] = pq_of(world.size()/nbd);
  using darr_t = memory::dmatrix_array_t<MEM, cvalue, 1, communicator>;

  const long NB = 2*nbd, M = 24, N = 24, mb = 8, nb = 8;
  darr_t dA(world, {nbd,p,q}, {NB,M,N}, {mb,nb});
  darr_t dB(world, {nbd,p,q}, {NB,M,N}, {mb,nb});
  darr_t dC(world, {nbd,p,q}, {NB,M,N}, {mb,nb});

  // each batch element gets its own data, keyed on the global batch index
  for (long ib = 0; ib < dA.n_local_batch(); ++ib) {
    long g = dA.local_batch_index(ib)[0];
    hmat_t a(M,N), b(M,N);
    for (long j = 0; j < N; ++j) for (long i = 0; i < M; ++i) {
      a(i,j) = gen(i,j,10+g); b(i,j) = gen(i,j,20+g);
    }
    scatter_into(dA, a, ib); scatter_into(dB, b, ib);
  }
  slate_ops::multiply(dA, dB, dC);

  for (long ib = 0; ib < dC.n_local_batch(); ++ib) {
    long g = dC.local_batch_index(ib)[0];
    hmat_t a(M,N), b(M,N), want(M,N);
    for (long j = 0; j < N; ++j) for (long i = 0; i < M; ++i) {
      a(i,j) = gen(i,j,10+g); b(i,j) = gen(i,j,20+g);
    }
    want() = cvalue(0.0);
    nda::blas::gemm(cvalue(1.0), a, b, cvalue(0.0), want);
    auto got = gather_all(dC, world, ib);
    REQUIRE(max_abs_diff(got, want) < 1e-11);
  }
}

/***************************************************************************/
/*                          lu_solve / inverse                             */
/***************************************************************************/
template<MEMORY_SPACE MEM>
void test_solves()
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto [p, q] = pq_of(world.size());
  using darr_t = memory::dmatrix_array_t<MEM, cvalue, 0, communicator>;

  const long M = 32, NRHS = 12, mb = 8, nb = 8;
  if (M < p or M < q) return;

  // diagonally dominant => well conditioned, no pivoting surprises
  hmat_t Aref(M,M), Bref(M,NRHS);
  for (long j = 0; j < M; ++j)
    for (long i = 0; i < M; ++i) Aref(i,j) = gen(i,j,7) + (i==j ? cvalue(4.0*M,0.0) : cvalue(0.0));
  for (long j = 0; j < NRHS; ++j) for (long i = 0; i < M; ++i) Bref(i,j) = gen(i,j,8);

  // ---- lu_solve: A X = B, then check A*X == B
  {
    darr_t dA(world, {p,q}, {M,M}, {mb,nb});
    darr_t dB(world, {p,q}, {M,NRHS}, {mb,nb});
    scatter_into(dA, Aref); scatter_into(dB, Bref);
    long info = slate_ops::lu_solve(dA, dB);
    REQUIRE(info == 0);
    auto X = gather_all(dB, world);
    hmat_t AX(M,NRHS);
    AX() = cvalue(0.0);
    nda::blas::gemm(cvalue(1.0), Aref, X, cvalue(0.0), AX);
    REQUIRE(max_abs_diff(AX, Bref) < 1e-9);
  }

  // ---- inverse: A A^{-1} == I
  {
    darr_t dA(world, {p,q}, {M,M}, {mb,nb});
    scatter_into(dA, Aref);
    slate_ops::inverse(dA);
    auto Ainv = gather_all(dA, world);
    hmat_t prod(M,M);
    prod() = cvalue(0.0);
    nda::blas::gemm(cvalue(1.0), Aref, Ainv, cvalue(0.0), prod);
    hmat_t Id(M,M);
    Id() = cvalue(0.0);
    for (long i = 0; i < M; ++i) Id(i,i) = cvalue(1.0);
    REQUIRE(max_abs_diff(prod, Id) < 1e-9);
  }

  // ---- least_squares_solve on a square well-conditioned system: same answer as lu_solve
  {
    darr_t dA(world, {p,q}, {M,M}, {mb,nb});
    darr_t dB(world, {p,q}, {M,NRHS}, {mb,nb});
    scatter_into(dA, Aref); scatter_into(dB, Bref);
    slate_ops::least_squares_solve(dA, dB);
    auto X = gather_all(dB, world);
    hmat_t AX(M,NRHS);
    AX() = cvalue(0.0);
    nda::blas::gemm(cvalue(1.0), Aref, X, cvalue(0.0), AX);
    REQUIRE(max_abs_diff(AX, Bref) < 1e-8);
  }
}

TEST_CASE("matrix_array_multiply_host",  "[math][matrix_array_ops]") { test_multiply<HOST_MEMORY>(); }
TEST_CASE("matrix_array_multiply_dev",   "[math][matrix_array_ops]") { test_multiply<DEVICE_MEMORY>(); }
TEST_CASE("matrix_array_batched_host",   "[math][matrix_array_ops]") { test_multiply_batched<HOST_MEMORY>(); }
TEST_CASE("matrix_array_batched_dev",    "[math][matrix_array_ops]") { test_multiply_batched<DEVICE_MEMORY>(); }
TEST_CASE("matrix_array_solves_host",    "[math][matrix_array_ops]") { test_solves<HOST_MEMORY>(); }
TEST_CASE("matrix_array_solves_dev",     "[math][matrix_array_ops]") { test_solves<DEVICE_MEMORY>(); }

/***************************************************************************/
/*                       redistribute round trip                           */
/***************************************************************************/
/*
 * distributed_array -> distributed_matrix_array -> distributed_array must reproduce the
 * original exactly. This is the conversion that makes the copy into SLATE's layout free at
 * the points where the code already redistributes (thc_aux.icc:1716-1717).
 */
template<MEMORY_SPACE MEM, int BR>
void test_roundtrip(std::array<long,BR+2> gshape, std::array<long,2> tile,
                    std::array<long,BR+2> dgrid, std::array<long,BR+2> mgrid)
{
  auto world = boost::mpi3::environment::get_world_instance();
  constexpr int R = BR + 2;
  using local_t = memory::array<MEM, cvalue, R>;
  using darr_t  = memory::dmatrix_array_t<MEM, cvalue, BR, communicator>;

  long np = 1; for (auto v : dgrid) np *= v;
  long nq = 1; for (auto v : mgrid) nq *= v;
  if (np != world.size() or nq != world.size()) return;
  for (int d = 0; d < R; ++d) if (gshape[d] < dgrid[d] or gshape[d] < mgrid[d]) return;

  auto A = make_distributed_array<local_t>(world, dgrid, gshape);
  auto C = make_distributed_array<local_t>(world, dgrid, gshape);

  // fill A from a global function of the global index
  {
    auto Ah = nda::to_host(A.local());
    auto st = Ah.indexmap().strides();
    long tot = Ah.size();
    for (long f = 0; f < tot; ++f) {
      // decompose f over Ah's shape (row-major)
      std::array<long,R> li{}; long rem = f;
      for (int d = R-1; d >= 0; --d) { li[d] = rem % Ah.extent(d); rem /= Ah.extent(d); }
      long off = 0; for (int d = 0; d < R; ++d) off += li[d]*st[d];
      long gi = li[R-2] + A.origin()[R-2], gj = li[R-1] + A.origin()[R-1];
      long seed = 1; for (int d = 0; d < BR; ++d) seed = seed*97 + (li[d] + A.origin()[d]);
      Ah.data()[off] = gen(gi, gj, seed);
    }
    A.local() = Ah;
  }

  darr_t B(world, mgrid, gshape, tile);
  redistribute_to_matrix_array(A, B);
  redistribute_from_matrix_array(B, C);

  // C must equal A exactly (same distribution, pure permutation of data)
  auto Ah = nda::to_host(A.local());
  auto Ch = nda::to_host(C.local());
  double worst = 0.0;
  for (long f = 0; f < Ah.size(); ++f)
    worst = std::max(worst, std::abs(Ah.data()[f] - Ch.data()[f]));
  REQUIRE(worst == 0.0);
}

template<MEMORY_SPACE MEM>
void test_roundtrip_all()
{
  auto world = boost::mpi3::environment::get_world_instance();
  long n = world.size();
  // rank 2, several matrix grids
  for (long p = 1; p <= n; ++p) {
    if (n % p != 0) continue;
    long q = n/p;
    test_roundtrip<MEM,0>({40,36}, {8,8}, {n,1}, {p,q});
    test_roundtrip<MEM,0>({37,29}, {8,8}, {n,1}, {p,q});   // non-divisible
    test_roundtrip<MEM,0>({40,36}, {8,8}, {1,n}, {p,q});
  }
  // rank 3 with a batch dimension
  if (n % 2 == 0) {
    test_roundtrip<MEM,1>({4,24,20}, {8,8}, {2,n/2,1}, {2,1,n/2});
    test_roundtrip<MEM,1>({4,24,20}, {8,8}, {2,1,n/2}, {2,n/2,1});
  }
  // The shape intvec_impl actually uses: all ranks on the batch dimension, trivial matrix
  // grid, and source == destination grid. This was missing and is where production broke.
  test_roundtrip<MEM,1>({8,24,20},  {8,8}, {n,1,1}, {n,1,1});
  test_roundtrip<MEM,1>({8,37,29},  {8,8}, {n,1,1}, {n,1,1});   // non-divisible
  test_roundtrip<MEM,1>({8,24,100}, {8,16},{n,1,1}, {n,1,1});   // wide RHS, mb != nb
  if (n % 2 == 0) {   // batch grid smaller than the communicator, matrix grid non-trivial
    test_roundtrip<MEM,1>({8,24,20}, {8,8}, {2,n/2,1}, {2,n/2,1});
  }
}

TEST_CASE("matrix_array_roundtrip_host", "[math][matrix_array_ops]") { test_roundtrip_all<HOST_MEMORY>(); }
TEST_CASE("matrix_array_roundtrip_dev",  "[math][matrix_array_ops]") { test_roundtrip_all<DEVICE_MEMORY>(); }

/***************************************************************************/
/*      legacy single-block path vs block-cyclic path, same input          */
/***************************************************************************/
/*
 * The production A/B (job 6719571) showed the two paths disagreeing on the thc ISDF solve.
 * This pins the comparison down with no production noise: identical hermitian A and RHS B,
 * solved once through the legacy C-order view path (slate_ops::lu_solve<true> on a
 * distributed_array) and once through the block-cyclic path, with the RESIDUAL ||A X - B||
 * as ground truth for both. A round-trip test cannot catch a transpose that both directions
 * apply consistently; this can.
 */
template<MEMORY_SPACE MEM>
void test_legacy_vs_block_cyclic_geom(long M, long NRHS, long mb, long nb, long p, long q);

template<MEMORY_SPACE MEM>
void test_legacy_vs_block_cyclic()
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto [p, q] = pq_of(world.size());
  // Geometries to cover. Production (thc ISDF at 4 ranks) has mb == M -- a SINGLE tile row --
  // and a wide RHS with nb != mb, e.g. block sizes (931, 1024) at Np=931. Every earlier
  // comparison here used M=32/mb=8 (four square tiles), so that regime was untested.
  struct Geom { long M, NRHS, mb, nb; };
  for (auto g : std::vector<Geom>{ {32, 20,  8,  8},   // square tiles, divisible
                                   {32, 20, 32, 20},   // single tile row+col
                                   {32, 96, 32, 24},   // single tile row, wide RHS, nb != mb
                                   {31, 90, 31, 24},   // same, non-divisible RHS
                                   {40, 96, 16, 24} }) // mb != nb, several tiles
    test_legacy_vs_block_cyclic_geom<MEM>(g.M, g.NRHS, g.mb, g.nb, p, q);
}

template<MEMORY_SPACE MEM>
void test_legacy_vs_block_cyclic_geom(long M, long NRHS, long mb, long nb, long p, long q)
{
  auto world = boost::mpi3::environment::get_world_instance();
  if (M < p or M < q or NRHS < q) return;

  // hermitian, diagonally dominant  (this is what C_quv is: an overlap matrix)
  hmat_t Aref(M,M), Bref(M,NRHS);
  for (long j = 0; j < M; ++j)
    for (long i = 0; i <= j; ++i) {
      auto v = gen(i,j,31);
      Aref(i,j) = v;
      Aref(j,i) = std::conj(v);
    }
  for (long i = 0; i < M; ++i) Aref(i,i) = cvalue(4.0*M, 0.0);
  for (long j = 0; j < NRHS; ++j) for (long i = 0; i < M; ++i) Bref(i,j) = gen(i,j,32);

  // hermiticity of the reference, so a failure below cannot be blamed on the input
  double herm = 0.0;
  for (long j = 0; j < M; ++j)
    for (long i = 0; i < M; ++i) herm = std::max(herm, std::abs(Aref(i,j)-std::conj(Aref(j,i))));
  REQUIRE(herm == 0.0);

  auto residual = [&](hmat_t const& X) {
    hmat_t AX(M,NRHS);
    AX() = cvalue(0.0);
    nda::blas::gemm(cvalue(1.0), Aref, X, cvalue(0.0), AX);
    return max_abs_diff(AX, Bref);
  };

  // make_distributed_array (nda_utils.hpp) silently clamps every block size to the per-rank
  // chunk:  bsize[n] = min(max(1,bsize[n]), shape[n]/grid[n]).  For the SQUARE matrix A that
  // makes the row and column block sizes differ whenever p != q, so make_slate builds
  // mt = M/min(mb,M/p) != nt = M/min(mb,M/q) and slate's gesv rejects it with
  // "A.mt() == A.nt()".  Concretely at 2 ranks (p,q)=(1,2) with M=32, mb=32: bsize becomes
  // {32,16} -> mt=2, nt=1.  It survives 4 ranks only because pq_of(4)=(2,2) clamps both dims
  // equally, and 1 rank only because that takes the serial nda path.
  // This is a limitation of the LEGACY single-block container; the block-cyclic container
  // tiles A squarely by construction, so only the legacy leg is skipped here.
  const bool legacy_supported = (std::min(mb, M/p) == std::min(mb, M/q));

  // ---- legacy path: C-order distributed_array + lu_solve<true> ----
  using cblock_t = memory::array<MEM, cvalue, 2>;
  hmat_t Xlegacy(M,NRHS);
  if (legacy_supported) {
    auto dA = make_distributed_array<cblock_t>(world, {p,q}, {M,M},    {mb,mb});
    auto dB = make_distributed_array<cblock_t>(world, {p,q}, {M,NRHS}, {mb,nb});
    auto fill = [&](auto& D, hmat_t const& src) {
      auto h = nda::to_host(D.local());
      for (long jj = 0; jj < h.extent(1); ++jj)
        for (long ii = 0; ii < h.extent(0); ++ii)
          h(ii,jj) = src(ii + D.origin()[0], jj + D.origin()[1]);
      D.local() = h;
    };
    fill(dA, Aref); fill(dB, Bref);
    long info = math::nda::slate_ops::lu_solve<true>(dA, dB);
    REQUIRE(info == 0);
    Xlegacy() = cvalue(0.0);
    {
      auto h = nda::to_host(dB.local());
      for (long jj = 0; jj < h.extent(1); ++jj)
        for (long ii = 0; ii < h.extent(0); ++ii)
          Xlegacy(ii + dB.origin()[0], jj + dB.origin()[1]) = h(ii,jj);
      world.all_reduce_in_place_n(Xlegacy.data(), Xlegacy.size(), std::plus<>{});
    }
  }

  // ---- block-cyclic path ----
  using ma_t = memory::dmatrix_array_t<MEM, cvalue, 0, communicator>;
  hmat_t Xbc(M,NRHS);
  {
    ma_t Cma(world, {p,q}, {M,M},    {mb,mb});
    ma_t Zma(world, {p,q}, {M,NRHS}, {mb,nb});
    scatter_into(Cma, Aref); scatter_into(Zma, Bref);
    long info = math::nda::slate_ops::lu_solve(Cma, Zma);
    REQUIRE(info == 0);
    Xbc = gather_all(Zma, world);
  }

  double r_bc = residual(Xbc);
  // the block-cyclic solve must hold at every geometry
  REQUIRE(r_bc < 1e-9);

  if (not legacy_supported) {
    if (world.root())
      std::cout << "  [M="<<M<<" NRHS="<<NRHS<<" mb="<<mb<<" nb="<<nb<<" p="<<p<<" q="<<q
                << "] block-cyclic residual = " << r_bc
                << "   (legacy leg skipped: block-size clamp makes A tile-non-square)"
                << std::endl;
    return;
  }

  double r_legacy = residual(Xlegacy);
  if (world.root()) {
    std::cout << "  [M="<<M<<" NRHS="<<NRHS<<" mb="<<mb<<" nb="<<nb<<"] legacy residual = " << r_legacy
              << "   block-cyclic residual = " << r_bc
              << "   ||Xlegacy - Xbc|| = " << max_abs_diff(Xlegacy, Xbc) << std::endl;
  }
  // both must actually solve the system ...
  REQUIRE(r_legacy < 1e-9);
  // ... and therefore agree (the solve is unique: A is nonsingular)
  REQUIRE(max_abs_diff(Xlegacy, Xbc) < 1e-9);
}

TEST_CASE("matrix_array_vs_legacy_host", "[math][matrix_array_ops]") { test_legacy_vs_block_cyclic<HOST_MEMORY>(); }
TEST_CASE("matrix_array_vs_legacy_dev",  "[math][matrix_array_ops]") { test_legacy_vs_block_cyclic<DEVICE_MEMORY>(); }

/*
 * Same comparison but on an ILL-CONDITIONED hermitian matrix, which is what the ISDF overlap
 * actually is: it comes from a pivoted Cholesky truncated at thresh (1e-5 by default), so it is
 * hermitian and formally nonsingular but badly conditioned. Two factorizations that order
 * operations differently can then land on solutions that differ substantially while both having
 * tiny residuals. This is the test that says whether the production ERI difference is a bug or
 * conditioning: the assertion is on the RESIDUALS, not on X agreeing.
 */
template<MEMORY_SPACE MEM>
void test_legacy_vs_bc_illcond()
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto [p, q] = pq_of(world.size());
  const long M = 32, NRHS = 16, mb = 8, nb = 8;
  if (M < p or M < q or NRHS < q) return;

  // A = U D U^H with D spanning 1 .. 1e-8  =>  hermitian, cond ~ 1e8
  hmat_t U(M,M), Aref(M,M), Bref(M,NRHS);
  for (long j = 0; j < M; ++j) for (long i = 0; i < M; ++i) U(i,j) = gen(i,j,41);
  // orthonormalize U by modified Gram-Schmidt so that A's spectrum is exactly D
  for (long j = 0; j < M; ++j) {
    for (long k = 0; k < j; ++k) {
      cvalue ip(0.0,0.0);
      for (long i = 0; i < M; ++i) ip += std::conj(U(i,k))*U(i,j);
      for (long i = 0; i < M; ++i) U(i,j) -= ip*U(i,k);
    }
    double nrm = 0.0;
    for (long i = 0; i < M; ++i) nrm += std::norm(U(i,j));
    nrm = std::sqrt(nrm);
    for (long i = 0; i < M; ++i) U(i,j) /= nrm;
  }
  std::vector<double> D(static_cast<std::size_t>(M), 0.0);  // 2 args: not a function decl
  for (long i = 0; i < M; ++i) D[std::size_t(i)] = std::pow(10.0, -8.0*double(i)/double(M-1));
  for (long j = 0; j < M; ++j)
    for (long i = 0; i < M; ++i) {
      cvalue v(0.0,0.0);
      for (long k = 0; k < M; ++k) v += U(i,k)*D[std::size_t(k)]*std::conj(U(j,k));
      Aref(i,j) = v;
    }
  for (long j = 0; j < NRHS; ++j) for (long i = 0; i < M; ++i) Bref(i,j) = gen(i,j,42);

  auto residual = [&](hmat_t const& X) {
    hmat_t AX(M,NRHS);
    AX() = cvalue(0.0);
    nda::blas::gemm(cvalue(1.0), Aref, X, cvalue(0.0), AX);
    return max_abs_diff(AX, Bref);
  };
  double bnorm = 0.0;
  for (long j = 0; j < NRHS; ++j) for (long i = 0; i < M; ++i) bnorm = std::max(bnorm, std::abs(Bref(i,j)));

  using cblock_t = memory::array<MEM, cvalue, 2>;
  hmat_t Xlegacy(M,NRHS), Xbc(M,NRHS);
  {
    auto dA = make_distributed_array<cblock_t>(world, {p,q}, {M,M},    {mb,mb});
    auto dB = make_distributed_array<cblock_t>(world, {p,q}, {M,NRHS}, {mb,nb});
    auto fill = [&](auto& Dd, hmat_t const& src) {
      auto h = nda::to_host(Dd.local());
      for (long jj = 0; jj < h.extent(1); ++jj)
        for (long ii = 0; ii < h.extent(0); ++ii)
          h(ii,jj) = src(ii + Dd.origin()[0], jj + Dd.origin()[1]);
      Dd.local() = h;
    };
    fill(dA, Aref); fill(dB, Bref);
    math::nda::slate_ops::lu_solve<true>(dA, dB);
    Xlegacy() = cvalue(0.0);
    auto h = nda::to_host(dB.local());
    for (long jj = 0; jj < h.extent(1); ++jj)
      for (long ii = 0; ii < h.extent(0); ++ii)
        Xlegacy(ii + dB.origin()[0], jj + dB.origin()[1]) = h(ii,jj);
    world.all_reduce_in_place_n(Xlegacy.data(), Xlegacy.size(), std::plus<>{});
  }
  {
    using ma_t = memory::dmatrix_array_t<MEM, cvalue, 0, communicator>;
    ma_t Cma(world, {p,q}, {M,M},    {mb,mb});
    ma_t Zma(world, {p,q}, {M,NRHS}, {mb,nb});
    scatter_into(Cma, Aref); scatter_into(Zma, Bref);
    math::nda::slate_ops::lu_solve(Cma, Zma);
    Xbc = gather_all(Zma, world);
  }

  double r_legacy = residual(Xlegacy), r_bc = residual(Xbc);
  double dx = max_abs_diff(Xlegacy, Xbc);
  double xmax = 0.0;
  for (long j = 0; j < NRHS; ++j) for (long i = 0; i < M; ++i) xmax = std::max(xmax, std::abs(Xlegacy(i,j)));
  if (world.root())
    std::cout << "  [illcond cond~1e8] legacy res = " << r_legacy << "  bc res = " << r_bc
              << "  ||dX|| = " << dx << "  ||X|| = " << xmax
              << "  rel dX = " << dx/xmax << std::endl;

  // both must solve the system to the accuracy the conditioning allows
  REQUIRE(r_legacy < 1e-6*bnorm*1e2);
  REQUIRE(r_bc     < 1e-6*bnorm*1e2);
}

TEST_CASE("matrix_array_illcond_host", "[math][matrix_array_ops]") { test_legacy_vs_bc_illcond<HOST_MEMORY>(); }
TEST_CASE("matrix_array_illcond_dev",  "[math][matrix_array_ops]") { test_legacy_vs_bc_illcond<DEVICE_MEMORY>(); }

/***************************************************************************/
/*      the solve orientation of each path, pinned on a NON-hermitian A    */
/***************************************************************************/
/*
 * The two paths above agree because A is hermitian, and that is not an accident of the test --
 * it is the *only* reason they agree, so it is the thing to pin. The legacy path stores A in C
 * order, so lu_solve<true> conjugates A in place and hands slate the transposed view: slate
 * receives conj(A)^T = A^H and solves A^H X = B. The block-cyclic container is natively column
 * major and hands slate A itself, solving A X = B. For hermitian A those are one system; for
 * anything else they are two.
 *
 * Pinning it on a non-hermitian A makes the convention explicit and gives the tests teeth: if
 * either path's orientation is ever changed, the residual it is checked against here stops being
 * satisfied. The hermitian agreement tests alone cannot see such a change, because A^H == A hides
 * it -- which is how the wrong 2026-07-31 conclusion (that the paths disagreed on orientation, when
 * in fact the input C was singular) survived as long as it did. Production call sites all pass a
 * hermitian A; this test exists to keep the ground under that statement.
 */
template<MEMORY_SPACE MEM>
void test_solve_orientation()
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto [p, q] = pq_of(world.size());
  const long M = 32, NRHS = 16, mb = 8, nb = 8;
  if (M < p or M < q or NRHS < q) return;

  // Deliberately NOT hermitian, and diagonally dominant so both systems are well conditioned
  // and the residuals below are meaningful.
  hmat_t Aref(M,M), Bref(M,NRHS);
  for (long j = 0; j < M; ++j)
    for (long i = 0; i < M; ++i) Aref(i,j) = gen(i,j,51);
  for (long i = 0; i < M; ++i) Aref(i,i) = cvalue(4.0*M, 0.0);
  for (long j = 0; j < NRHS; ++j) for (long i = 0; i < M; ++i) Bref(i,j) = gen(i,j,52);

  // it must really be non-hermitian, or this test proves nothing
  double herm = 0.0;
  for (long j = 0; j < M; ++j)
    for (long i = 0; i < M; ++i) herm = std::max(herm, std::abs(Aref(i,j)-std::conj(Aref(j,i))));
  REQUIRE(herm > 0.1);

  // residual of X against the system  op(A) X = B
  auto residual_against = [&](hmat_t const& op, hmat_t const& X) {
    hmat_t AX(M,NRHS);
    AX() = cvalue(0.0);
    nda::blas::gemm(cvalue(1.0), op, X, cvalue(0.0), AX);
    return max_abs_diff(AX, Bref);
  };
  hmat_t Ah(M,M);                       // A^H
  for (long j = 0; j < M; ++j)
    for (long i = 0; i < M; ++i) Ah(i,j) = std::conj(Aref(j,i));

  // ---- legacy path: C-order view + lu_solve<true>  =>  solves A^H X = B ----
  using cblock_t = memory::array<MEM, cvalue, 2>;
  hmat_t Xlegacy(M,NRHS);
  {
    auto dA = make_distributed_array<cblock_t>(world, {p,q}, {M,M},    {mb,mb});
    auto dB = make_distributed_array<cblock_t>(world, {p,q}, {M,NRHS}, {mb,nb});
    auto fill = [&](auto& Dd, hmat_t const& src) {
      auto h = nda::to_host(Dd.local());
      for (long jj = 0; jj < h.extent(1); ++jj)
        for (long ii = 0; ii < h.extent(0); ++ii)
          h(ii,jj) = src(ii + Dd.origin()[0], jj + Dd.origin()[1]);
      Dd.local() = h;
    };
    fill(dA, Aref); fill(dB, Bref);
    long info = math::nda::slate_ops::lu_solve<true>(dA, dB);
    REQUIRE(info == 0);
    Xlegacy() = cvalue(0.0);
    auto h = nda::to_host(dB.local());
    for (long jj = 0; jj < h.extent(1); ++jj)
      for (long ii = 0; ii < h.extent(0); ++ii)
        Xlegacy(ii + dB.origin()[0], jj + dB.origin()[1]) = h(ii,jj);
    world.all_reduce_in_place_n(Xlegacy.data(), Xlegacy.size(), std::plus<>{});
  }

  // ---- block-cyclic path: column major, no conjugation  =>  solves A X = B ----
  hmat_t Xbc(M,NRHS);
  {
    using ma_t = memory::dmatrix_array_t<MEM, cvalue, 0, communicator>;
    ma_t Cma(world, {p,q}, {M,M},    {mb,mb});
    ma_t Zma(world, {p,q}, {M,NRHS}, {mb,nb});
    scatter_into(Cma, Aref); scatter_into(Zma, Bref);
    long info = math::nda::slate_ops::lu_solve(Cma, Zma);
    REQUIRE(info == 0);
    Xbc = gather_all(Zma, world);
  }

  double leg_vs_Ah = residual_against(Ah,   Xlegacy);
  double leg_vs_A  = residual_against(Aref, Xlegacy);
  double bc_vs_A   = residual_against(Aref, Xbc);
  double bc_vs_Ah  = residual_against(Ah,   Xbc);
  if (world.root())
    std::cout << "  [orientation, non-hermitian A] legacy: ||A^H X - B|| = " << leg_vs_Ah
              << " (vs ||A X - B|| = " << leg_vs_A << ")   block-cyclic: ||A X - B|| = "
              << bc_vs_A << " (vs ||A^H X - B|| = " << bc_vs_Ah << ")" << std::endl;

  // each path solves its own system ...
  REQUIRE(leg_vs_Ah < 1e-9);
  REQUIRE(bc_vs_A   < 1e-9);
  // ... and demonstrably not the other one, so a change of orientation cannot pass unnoticed
  REQUIRE(leg_vs_A  > 1e-6);
  REQUIRE(bc_vs_Ah  > 1e-6);
}

TEST_CASE("matrix_array_hermitian_solve_orientation", "[math][matrix_array_ops]") { test_solve_orientation<HOST_MEMORY>(); }
TEST_CASE("matrix_array_hermitian_solve_orientation_dev", "[math][matrix_array_ops]") { test_solve_orientation<DEVICE_MEMORY>(); }

} // namespace bdft_tests



