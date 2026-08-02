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

#ifndef NUMERICS_DISTRIBUTED_ARRAY_MATRIX_ARRAY_REDISTRIBUTE_HPP
#define NUMERICS_DISTRIBUTED_ARRAY_MATRIX_ARRAY_REDISTRIBUTE_HPP

/*
 * Conversion between math::nda::distributed_array (one contiguous block per rank, the layout
 * the local tensor work wants) and math::nda::distributed_matrix_array (2D block cyclic,
 * tile-major, the layout SLATE wants).
 *
 * Structure of the exchange, in both directions:
 *
 *   - the source's local data is a rectangle in the global index space (batch ranges x
 *     [i0,i1) x [j0,j1));
 *   - the destination's local data is a set of tiles, each also a rectangle;
 *   - so the traffic between a given (src, dst) pair is the set of rectangle intersections.
 *     Enumerating destination tiles is cheap (mt*nt is small), which is why this does not
 *     need to touch the global index space element by element.
 *
 * Sender and receiver enumerate the same (batch, tile) sequence in the same canonical order
 * (batch outer, then tile column, then tile row), so the receiver can unpack without any
 * metadata travelling with the payload. Each rank's source rectangle is all-gathered up front,
 * following the same approach as redistribute_standard.
 *
 * PERFORMANCE NOTE: pack/unpack currently stage through host memory, so a device-to-device
 * conversion costs one D2H and one H2D of the local block. That is deliberate for a first
 * correct version -- it removes any dependence on CUDA-aware MPI. The follow-up is to reuse
 * redistribute_alltoallv's device-direct pairwise exchange, which already exists.
 */

#include <array>
#include <numeric>
#include <vector>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "itertools/itertools.hpp"
#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/matrix_array.hpp"
#include "numerics/distributed_array/slate_ops_matrix_array.hpp"

namespace math::nda
{

namespace detail
{

/// intersection of [a0,a1) and [b0,b1); empty if lo >= hi
inline std::pair<long,long> isect(long a0, long a1, long b0, long b1)
{ return {std::max(a0,b0), std::min(a1,b1)}; }

/*
 * Walk the (batch multi-index, tile) pairs exchanged between a source rectangle and the tiles
 * of one destination rank, in canonical order, calling
 *     f(batch_global_index_array, i0, i1, j0, j1, tile_it, tile_jt)
 * for each non-empty intersection. `i0..j1` are GLOBAL element ranges.
 *
 * Canonical order: batch (row-major over the shared batch range) outer, then tile column,
 * then tile row. Both sides of the exchange use this function, so their orders agree by
 * construction.
 */
template<typename MA, typename F>
void for_each_exchange(MA const& B, long dest_cell, long dest_mat_rank,
                       std::array<long,std::decay_t<MA>::rank> const& src_origin,
                       std::array<long,std::decay_t<MA>::rank> const& src_lshape,
                       F&& f)
{
  constexpr int R  = std::decay_t<MA>::rank;
  constexpr int BR = R - 2;
  long p = B.matrix_grid_rows(), q = B.matrix_grid_cols();
  long pr = dest_mat_rank / q, qr = dest_mat_rank % q;

  // shared batch range per batch dimension
  std::array<long,BR> blo{}, bhi{};
  for (int d = 0; d < BR; ++d) {
    auto r = B.batch_range_of_cell(dest_cell, d);
    auto [lo, hi] = isect(src_origin[d], src_origin[d] + src_lshape[d], r.first(), r.last());
    if (lo >= hi) return;   // nothing shared
    blo[d] = lo; bhi[d] = hi;
  }

  long si0 = src_origin[R-2], si1 = si0 + src_lshape[R-2];
  long sj0 = src_origin[R-1], sj1 = sj0 + src_lshape[R-1];
  long mb = B.mb(), nb = B.nb();

  // iterate the shared batch box row-major
  long nbatch = 1;
  for (int d = 0; d < BR; ++d) nbatch *= (bhi[d] - blo[d]);
  for (long ib = 0; ib < nbatch; ++ib) {
    std::array<long,BR> bidx{};
    long rem = ib;
    for (int d = BR - 1; d >= 0; --d) { bidx[d] = blo[d] + rem % (bhi[d]-blo[d]);
                                       rem /= (bhi[d]-blo[d]); }
    for (long jt = qr; jt < B.nt(); jt += q) {
      long tj0 = jt*nb, tj1 = tj0 + B.tileNb(jt);
      auto [j0, j1] = isect(sj0, sj1, tj0, tj1);
      if (j0 >= j1) continue;
      for (long it = pr; it < B.mt(); it += p) {
        long ti0 = it*mb, ti1 = ti0 + B.tileMb(it);
        auto [i0, i1] = isect(si0, si1, ti0, ti1);
        if (i0 >= i1) continue;
        f(bidx, i0, i1, j0, j1, it, jt);
      }
    }
  }
}

/// element access by std::array index (nda exposes only variadic operator())
template<int R, typename Arr>
auto& at_index(Arr&& A, std::array<long,R> const& idx)
{
  long off = 0;
  auto st = A.indexmap().strides();
  for (int d = 0; d < R; ++d) off += idx[d]*st[d];
  return A.data()[off];
}

/// local batch offset of a global batch multi-index within a container's local batch box
template<int BR>
long batch_local_offset(std::array<long,BR> const& g, std::array<long,BR> const& origin,
                        std::array<long,BR> const& extent)
{
  long off = 0;
  for (int d = 0; d < BR; ++d) off = off*extent[d] + (g[d] - origin[d]);
  return off;
}

} // namespace detail

/***************************************************************************/
/*        distributed_array  ->  distributed_matrix_array                  */
/***************************************************************************/
template<DistributedArray Src_t, DistributedMatrixArray Dst_t>
void redistribute_to_matrix_array(Src_t const& A, Dst_t& B)
{
  constexpr int R  = std::decay_t<Dst_t>::rank;
  constexpr int BR = R - 2;
  static_assert(get_rank<Src_t> == R, "redistribute_to_matrix_array: rank mismatch.");
  using value_type = typename std::decay_t<Dst_t>::value_type;

  utils::check(A.global_shape() == B.global_shape(),
      "redistribute_to_matrix_array: global shape mismatch.");
  auto& comm = *A.communicator();
  utils::check(comm.size() == B.communicator()->size(),
      "redistribute_to_matrix_array: communicator size mismatch.");
  long np = comm.size();

  // ---- everyone needs every rank's source rectangle -----------------------------------
  ::nda::array<long,2> boxes(np, 2*R);
  boxes() = 0;
  for (int d = 0; d < R; ++d) {
    boxes(comm.rank(), d)     = A.origin()[d];
    boxes(comm.rank(), R + d) = A.local_shape()[d];
  }
  comm.all_reduce_in_place_n(boxes.data(), boxes.size(), std::plus<>{});

  auto row_of = [&](long r) {
    std::array<long,R> o{}, l{};
    for (int d = 0; d < R; ++d) { o[d] = boxes(r,d); l[d] = boxes(r,R+d); }
    return std::make_pair(o,l);
  };

  // ---- send counts ---------------------------------------------------------------------
  long nmat = B.matrix_grid_rows()*B.matrix_grid_cols();
  std::vector<long> scount(np, 0), rcount(np, 0);
  auto [my_o, my_l] = row_of(comm.rank());
  for (long d = 0; d < np; ++d) {
    detail::for_each_exchange(B, d/nmat, d%nmat, my_o, my_l,
      [&](auto const&, long i0, long i1, long j0, long j1, long, long) {
        scount[std::size_t(d)] += (i1-i0)*(j1-j0);
      });
  }
  for (long s = 0; s < np; ++s) {
    auto [so, sl] = row_of(s);
    detail::for_each_exchange(B, B.my_batch_cell(),
                              comm.rank() % nmat, so, sl,
      [&](auto const&, long i0, long i1, long j0, long j1, long, long) {
        rcount[std::size_t(s)] += (i1-i0)*(j1-j0);
      });
  }

  // ---- pack ---------------------------------------------------------------------------
  // host copy of the source local block (one D2H when the source is on device)
  auto Ah = ::nda::to_host(A.local());

  std::vector<long> sdisp(np+1,0), rdisp(np+1,0);
  for (long r = 0; r < np; ++r) sdisp[std::size_t(r+1)] = sdisp[std::size_t(r)] + scount[std::size_t(r)];
  for (long r = 0; r < np; ++r) rdisp[std::size_t(r+1)] = rdisp[std::size_t(r)] + rcount[std::size_t(r)];

  std::vector<value_type> sbuf(std::size_t(sdisp[np])), rbuf(std::size_t(rdisp[np]));
  {
    std::vector<long> pos(sdisp.begin(), sdisp.end());
    for (long d = 0; d < np; ++d) {
      detail::for_each_exchange(B, d/nmat, d%nmat, my_o, my_l,
        [&](std::array<long,BR> const& bg, long i0, long i1, long j0, long j1, long, long) {
          // column-major within the intersection, matching the unpack side
          for (long j = j0; j < j1; ++j)
            for (long i = i0; i < i1; ++i) {
              std::array<long,R> idx{};
              for (int k = 0; k < BR; ++k) idx[k] = bg[k] - my_o[k];
              idx[R-2] = i - my_o[R-2];
              idx[R-1] = j - my_o[R-1];
              sbuf[std::size_t(pos[std::size_t(d)]++)] = detail::at_index<R>(Ah, idx);
            }
        });
    }
  }

  // ---- exchange -----------------------------------------------------------------------
  {
    // byte counts keep this free of datatype plumbing.
    // TODO: chunk when a single peer message would exceed INT_MAX bytes.
    constexpr long vs = long(sizeof(value_type));
    std::vector<int> sc(np), rc(np), sd(np), rd(np);
    for (long r = 0; r < np; ++r) {
      utils::check(scount[std::size_t(r)]*vs < 2147483647l and
                   rcount[std::size_t(r)]*vs < 2147483647l,
          "matrix_array redistribute: per-peer message exceeds INT_MAX bytes; needs chunking.");
      sc[std::size_t(r)] = int(scount[std::size_t(r)]*vs);
      rc[std::size_t(r)] = int(rcount[std::size_t(r)]*vs);
      sd[std::size_t(r)] = int(sdisp[std::size_t(r)]*vs);
      rd[std::size_t(r)] = int(rdisp[std::size_t(r)]*vs);
    }
    MPI_Alltoallv(sbuf.data(), sc.data(), sd.data(), MPI_BYTE,
                  rbuf.data(), rc.data(), rd.data(), MPI_BYTE, comm.get());
  }

  // ---- unpack -------------------------------------------------------------------------
  using htile_t = ::nda::array<value_type, 2, ::nda::F_layout>;
  auto lbatch = B.local_batch_extents();
  std::array<long,BR> borigin{};
  for (int d = 0; d < BR; ++d) borigin[d] = B.origin()[d];

  // stage each of my tiles on the host, fill from the recv buffer, then push to the tile
  long nti = B.n_local_tile_rows(), ntj = B.n_local_tile_cols();
  std::vector<htile_t> stage(std::size_t(std::max(1l, B.n_local_batch()*nti*ntj)));
  for (long ib = 0; ib < B.n_local_batch(); ++ib)
    for (long b = 0; b < ntj; ++b)
      for (long a = 0; a < nti; ++a) {
        auto t = B.tile(ib, a, b);
        auto& h = stage[std::size_t((ib*ntj + b)*nti + a)];
        h = htile_t(t.extent(0), t.extent(1));
        h = t;   // preserve entries nobody sends (B may be partially overwritten)
      }

  auto local_tile_slot = [&](long it, long jt) {
    long a = -1, b = -1;
    for (long x = 0; x < nti; ++x) if (B.local_tile_row(x) == it) { a = x; break; }
    for (long y = 0; y < ntj; ++y) if (B.local_tile_col(y) == jt) { b = y; break; }
    return std::make_pair(a,b);
  };

  {
    std::vector<long> pos(rdisp.begin(), rdisp.end());
    for (long s = 0; s < np; ++s) {
      auto [so, sl] = row_of(s);
      detail::for_each_exchange(B, B.my_batch_cell(), comm.rank() % nmat, so, sl,
        [&](std::array<long,BR> const& bg, long i0, long i1, long j0, long j1,
            long it, long jt) {
          long ib = detail::batch_local_offset<BR>(bg, borigin, lbatch);
          auto [a, b] = local_tile_slot(it, jt);
          utils::check(a >= 0 and b >= 0, "redistribute_to_matrix_array: tile not local.");
          auto& h = stage[std::size_t((ib*ntj + b)*nti + a)];
          for (long j = j0; j < j1; ++j)
            for (long i = i0; i < i1; ++i)
              h(i - it*B.mb(), j - jt*B.nb()) = rbuf[std::size_t(pos[std::size_t(s)]++)];
        });
    }
  }

  for (long ib = 0; ib < B.n_local_batch(); ++ib)
    for (long b = 0; b < ntj; ++b)
      for (long a = 0; a < nti; ++a) {
        auto t = B.tile(ib, a, b);
        t = stage[std::size_t((ib*ntj + b)*nti + a)];
      }
}

/***************************************************************************/
/*        distributed_matrix_array  ->  distributed_array                  */
/***************************************************************************/
template<DistributedMatrixArray Src_t, DistributedArray Dst_t>
void redistribute_from_matrix_array(Src_t& A, Dst_t& B)
{
  constexpr int R  = std::decay_t<Src_t>::rank;
  constexpr int BR = R - 2;
  static_assert(get_rank<Dst_t> == R, "redistribute_from_matrix_array: rank mismatch.");
  using value_type = typename std::decay_t<Src_t>::value_type;

  utils::check(A.global_shape() == B.global_shape(),
      "redistribute_from_matrix_array: global shape mismatch.");
  auto& comm = *B.communicator();
  long np = comm.size();
  long nmat = A.matrix_grid_rows()*A.matrix_grid_cols();

  // destination rectangles
  ::nda::array<long,2> boxes(np, 2*R);
  boxes() = 0;
  for (int d = 0; d < R; ++d) {
    boxes(comm.rank(), d)     = B.origin()[d];
    boxes(comm.rank(), R + d) = B.local_shape()[d];
  }
  comm.all_reduce_in_place_n(boxes.data(), boxes.size(), std::plus<>{});
  auto row_of = [&](long r) {
    std::array<long,R> o{}, l{};
    for (int d = 0; d < R; ++d) { o[d] = boxes(r,d); l[d] = boxes(r,R+d); }
    return std::make_pair(o,l);
  };

  // I send from my tiles to whoever owns those elements in B
  std::vector<long> scount(np,0), rcount(np,0);
  for (long d = 0; d < np; ++d) {
    auto [dobox, dlbox] = row_of(d);
    detail::for_each_exchange(A, A.my_batch_cell(), comm.rank() % nmat, dobox, dlbox,
      [&](auto const&, long i0, long i1, long j0, long j1, long, long) {
        scount[std::size_t(d)] += (i1-i0)*(j1-j0);
      });
  }
  auto [my_o, my_l] = row_of(comm.rank());
  for (long s = 0; s < np; ++s) {
    detail::for_each_exchange(A, s/nmat, s%nmat, my_o, my_l,
      [&](auto const&, long i0, long i1, long j0, long j1, long, long) {
        rcount[std::size_t(s)] += (i1-i0)*(j1-j0);
      });
  }

  std::vector<long> sdisp(np+1,0), rdisp(np+1,0);
  for (long r = 0; r < np; ++r) sdisp[std::size_t(r+1)] = sdisp[std::size_t(r)] + scount[std::size_t(r)];
  for (long r = 0; r < np; ++r) rdisp[std::size_t(r+1)] = rdisp[std::size_t(r)] + rcount[std::size_t(r)];
  std::vector<value_type> sbuf(std::size_t(sdisp[np])), rbuf(std::size_t(rdisp[np]));

  // host copies of my tiles
  using htile_t = ::nda::array<value_type, 2, ::nda::F_layout>;
  long nti = A.n_local_tile_rows(), ntj = A.n_local_tile_cols();
  std::vector<htile_t> stage(std::size_t(std::max(1l, A.n_local_batch()*nti*ntj)));
  for (long ib = 0; ib < A.n_local_batch(); ++ib)
    for (long b = 0; b < ntj; ++b)
      for (long a = 0; a < nti; ++a) {
        auto t = A.tile(ib, a, b);
        auto& h = stage[std::size_t((ib*ntj + b)*nti + a)];
        h = htile_t(t.extent(0), t.extent(1));
        h = t;
      }
  auto slot = [&](long it, long jt) {
    long a = -1, b = -1;
    for (long x = 0; x < nti; ++x) if (A.local_tile_row(x) == it) { a = x; break; }
    for (long y = 0; y < ntj; ++y) if (A.local_tile_col(y) == jt) { b = y; break; }
    return std::make_pair(a,b);
  };
  auto lbatch = A.local_batch_extents();
  std::array<long,BR> borigin{};
  for (int d = 0; d < BR; ++d) borigin[d] = A.origin()[d];

  {
    std::vector<long> pos(sdisp.begin(), sdisp.end());
    for (long d = 0; d < np; ++d) {
      auto [dobox, dlbox] = row_of(d);
      detail::for_each_exchange(A, A.my_batch_cell(), comm.rank() % nmat, dobox, dlbox,
        [&](std::array<long,BR> const& bg, long i0, long i1, long j0, long j1,
            long it, long jt) {
          long ib = detail::batch_local_offset<BR>(bg, borigin, lbatch);
          auto [a, b] = slot(it, jt);
          auto const& h = stage[std::size_t((ib*ntj + b)*nti + a)];
          for (long j = j0; j < j1; ++j)
            for (long i = i0; i < i1; ++i)
              sbuf[std::size_t(pos[std::size_t(d)]++)] = h(i - it*A.mb(), j - jt*A.nb());
        });
    }
  }

  {
    // byte counts keep this free of datatype plumbing.
    // TODO: chunk when a single peer message would exceed INT_MAX bytes.
    constexpr long vs = long(sizeof(value_type));
    std::vector<int> sc(np), rc(np), sd(np), rd(np);
    for (long r = 0; r < np; ++r) {
      utils::check(scount[std::size_t(r)]*vs < 2147483647l and
                   rcount[std::size_t(r)]*vs < 2147483647l,
          "matrix_array redistribute: per-peer message exceeds INT_MAX bytes; needs chunking.");
      sc[std::size_t(r)] = int(scount[std::size_t(r)]*vs);
      rc[std::size_t(r)] = int(rcount[std::size_t(r)]*vs);
      sd[std::size_t(r)] = int(sdisp[std::size_t(r)]*vs);
      rd[std::size_t(r)] = int(rdisp[std::size_t(r)]*vs);
    }
    MPI_Alltoallv(sbuf.data(), sc.data(), sd.data(), MPI_BYTE,
                  rbuf.data(), rc.data(), rd.data(), MPI_BYTE, comm.get());
  }

  // unpack into a host image of B's local block, then push
  auto Bh = ::nda::to_host(B.local());
  {
    std::vector<long> pos(rdisp.begin(), rdisp.end());
    for (long s = 0; s < np; ++s) {
      detail::for_each_exchange(A, s/nmat, s%nmat, my_o, my_l,
        [&](std::array<long,BR> const& bg, long i0, long i1, long j0, long j1, long, long) {
          for (long j = j0; j < j1; ++j)
            for (long i = i0; i < i1; ++i) {
              std::array<long,R> idx{};
              for (int k = 0; k < BR; ++k) idx[k] = bg[k] - my_o[k];
              idx[R-2] = i - my_o[R-2];
              idx[R-1] = j - my_o[R-1];
              detail::at_index<R>(Bh, idx) = rbuf[std::size_t(pos[std::size_t(s)]++)];
            }
        });
    }
  }
  B.local() = Bh;
}

} // namespace math::nda

#endif
