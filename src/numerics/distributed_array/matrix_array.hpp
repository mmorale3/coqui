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

#ifndef NUMERICS_DISTRIBUTED_ARRAY_MATRIX_ARRAY_HPP
#define NUMERICS_DISTRIBUTED_ARRAY_MATRIX_ARRAY_HPP

/*
 * distributed_matrix_array: a 2D block-cyclic "array of matrices", laid out the way SLATE
 * wants, for use ONLY as an operand of SLATE's distributed linear algebra.
 *
 * See perf_report/slate_array_design.md. Summary of why this exists:
 *
 *  - math::nda::distributed_array gives each rank one large contiguous local block, which is
 *    the right layout for the local tensor work done through .local(). It is NOT block
 *    cyclic, so it is a poor distribution for distributed factorizations.
 *  - Worse, handing SLATE tiles that are views into such a block gives them
 *    stride == block lld, while SLATE's own workspace copies of remote tiles have
 *    stride == tileMb(i). SLATE's batched device path groups tiles by (mb,nb) only and then
 *    requires one leading dimension per operand
 *    (assert(group.ld[m] == Mij.stride()), slate/src/internal/internal_batch.hh). Mixing the
 *    two aborts.
 *
 * This container stores its local data tile-major, so every tile has stride == tileMb(i),
 * matching SLATE's workspace tiles exactly. Uniform by construction.
 *
 * Dimensionality: block cyclic is 2D but the code's arrays are tensors. The trailing two
 * dimensions form the matrix; all leading dimensions are batch, distributed as contiguous
 * blocks. (s,k,P,Q) becomes (s,k)(P,Q). Every SLATE call in the codebase is a 2D operation
 * batched over leading indices, so this matches the actual use.
 */

#include <array>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "itertools/itertools.hpp"
#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "numerics/distributed_array/detail/concepts.hpp"
#if defined(ENABLE_SLATE)
#include "slate/slate.hh"
#endif
#if defined(ENABLE_CUDA)
#include "cuda_runtime.h"
#endif

namespace math::nda
{

/**
 * Whether call sites should route slate's distributed linear algebra through
 * distributed_matrix_array (block cyclic) instead of the legacy single-block views.
 *
 * Default **OFF** until the solve-orientation question below is settled.
 * `COQUI_SLATE_BLOCK_CYCLIC=1` opts in.
 *
 * KNOWN OPEN ISSUE (2026-07-31), do not enable by default before fixing:
 * the legacy path reaches slate through a C-order view, so with `hermitian=true` it conjugates
 * A and uses the transposed view, i.e. **slate sees M^H** where M is the stored matrix. The
 * block-cyclic container is natively column-major and hands slate **M**. Those agree only if M
 * is exactly hermitian. In the thc ISDF solve they do not agree: at 4 ranks the resulting ERI
 * differs by ~6e1 and the total energy by ~8e-5 (job 6719571), while the A/A control is
 * bit-identical, so the discrepancy is the solve orientation and not the redistribute (the
 * production-shaped redistribute round-trip is bit-exact in the unit tests).
 * Resolve by determining which orientation the ISDF fit intends, then either pass
 * conj_transpose on the new path or drop the conjugation on the legacy one -- and add a unit
 * test that pins the convention.
 */
inline bool use_block_cyclic_slate()
{
  static const bool v = []() {
    char const* e = std::getenv("COQUI_SLATE_BLOCK_CYCLIC");
    return (e == nullptr) ? false : (std::string(e) == "1");
  }();
  return v;
}

/**
 * 2D block-cyclic array of matrices.
 *
 * @tparam Buffer_t    rank-1 nda memory array holding the local tiles (its value_type and
 *                     address space define the container's).
 * @tparam batch_rank  number of leading (batch) dimensions; 0 means a single matrix.
 */
template<typename Buffer_t, int batch_rank, typename communicator_t>
class distributed_matrix_array
{
  static_assert(batch_rank >= 0, "batch_rank must be >= 0");
  static_assert(::nda::get_rank<Buffer_t> == 1, "Buffer_t must be rank 1.");

  public:

  static constexpr int rank = batch_rank + 2;
  static constexpr bool is_view = false;
  using value_type = typename std::decay_t<Buffer_t>::value_type;
  using buffer_type = Buffer_t;

  static constexpr auto addr_space = ::nda::mem::get_addr_space<Buffer_t>;
  static constexpr bool on_host = (addr_space == ::nda::mem::Host);

  // 2D view of a single tile: column-major (Fortran), contiguous, ld == tileMb(i)
  using tile_view_t = ::nda::basic_array_view<value_type, 2, ::nda::F_layout, 'A',
                                             ::nda::default_accessor,
                                             ::nda::borrowed<addr_space>>;
  using const_tile_view_t = ::nda::basic_array_view<const value_type, 2, ::nda::F_layout, 'A',
                                                   ::nda::default_accessor,
                                                   ::nda::borrowed<addr_space>>;

  private:

  // ---- distribution ------------------------------------------------------------------
  communicator_t* _comm = nullptr;
  // communicator holding the p*q ranks that share this rank's batch cell. Split eagerly in
  // the constructor: split() is collective, so it must not hide behind a lazy accessor.
  std::optional<communicator_t> _mat_comm;

  std::array<long, rank> _grid{};      // (batch..., p, q)
  std::array<long, rank> _gshape{};    // (batch..., M, N)
  std::array<long, rank> _origin{};    // batch origins; matrix dims are 0
  std::array<long, rank> _bsize{};     // batch block sizes; (mb, nb) for the matrix dims

  std::array<long, batch_rank> _lbatch{};   // local extent of each batch dim
  long _nlocal_batch = 0;                   // product of _lbatch

  long _p = 1, _q = 1;                 // matrix process grid
  long _pr = 0, _qr = 0;               // this rank's coordinates in it
  long _M = 0, _N = 0, _mb = 1, _nb = 1;
  long _mt = 0, _nt = 0;               // global tile counts (SLATE's ceil convention)

  // local tiles, in column-major tile order
  std::vector<long> _local_it;         // global tile-row indices owned here
  std::vector<long> _local_jt;         // global tile-col indices owned here
  // offset of tile (ib, a, b) within the buffer, a indexes _local_it, b indexes _local_jt
  std::vector<long> _offset;           // size _nlocal_batch * _local_it.size() * _local_jt.size()
  long _tiles_per_batch = 0;           // elements per batch element

  Buffer_t _buf;

  [[maybe_unused]] int _dev = 0;       // cuda device holding _buf (0 when on host)

  static long ceildiv(long a, long b) { return (a + b - 1) / b; }

  long tileMb_(long i) const { return std::min(_mb, _M - i*_mb); }
  long tileNb_(long j) const { return std::min(_nb, _N - j*_nb); }

  // Block-cyclic ownership. The rank index returned is the rank WITHIN the matrix
  // communicator, so its grid order must match how that communicator is ordered.
  //
  // make_distributed_array maps ranks row-major over the proc grid for C-layout arrays
  // (nda_utils.hpp: it walks n from rank-1 down to 0 taking ip%grid[n]), i.e. the LAST
  // dimension varies fastest. Splitting by batch cell preserves relative order, so within
  // the matrix communicator rank = pr*q + qr. Hence row-major tile ranking, which is
  // slate's GridOrder::Row.
  long tile_rank_(long i, long j) const { return (i % _p) * _q + (j % _q); }

  /// this rank's index within the matrix communicator; must use the same grid order as
  /// tile_rank_ above (row-major). Keeping these in one place: having two spellings of it
  /// agreed only when p==1 or q==1 and silently broke at p=q=2.
  long my_matrix_rank_() const { return _pr * _q + _qr; }

  public:

  distributed_matrix_array() = default;

  distributed_matrix_array(communicator_t& comm,
                           std::array<long, rank> grid,
                           std::array<long, rank> gshape,
                           std::array<long, 2> tile_size)
    : _comm(std::addressof(comm)), _grid(grid), _gshape(gshape)
  {
    long np = std::accumulate(grid.begin(), grid.end(), 1l, std::multiplies<>{});
    utils::check(comm.size() == np,
        "distributed_matrix_array: comm.size():{} != prod(grid):{}", comm.size(), np);

    _M  = gshape[rank-2];  _N  = gshape[rank-1];
    _p  = grid[rank-2];    _q  = grid[rank-1];
    _mb = std::min(tile_size[0], _M);
    _nb = std::min(tile_size[1], _N);
    utils::check(_mb > 0 and _nb > 0, "distributed_matrix_array: non-positive tile size.");
    utils::check(_M >= _p and _N >= _q,
        "distributed_matrix_array: matrix ({},{}) too small for grid ({},{}).", _M,_N,_p,_q);

    // SLATE's tile counts use ceil. NOTE: slate_aux.hpp:90 uses floor, which mis-assigns
    // the last partial tile row whenever M % mb != 0. Do not copy that.
    _mt = ceildiv(_M, _mb);
    _nt = ceildiv(_N, _nb);
    _bsize[rank-2] = _mb;
    _bsize[rank-1] = _nb;

    // ---- decompose the rank into (batch cell, matrix coordinates) ---------------------
    // Row-major over the proc grid (last dimension fastest), matching
    // make_distributed_array's C-layout branch so that batch cells line up with the source
    // arrays we redistribute from -- otherwise the exchange would cross batch cells.
    long r = comm.rank();
    _qr = r % _q;             // last dim (N) fastest
    _pr = (r / _q) % _p;      // then M
    long batch_id = r / (_p * _q);

    // batch dims: contiguous chunk of each dimension, last batch dim varying fastest
    {
      long rem = batch_id;
      std::array<long, batch_rank> coord{};
      for (int d = batch_rank - 1; d >= 0; --d) {
        coord[d] = rem % grid[d];
        rem /= grid[d];
      }
      for (int d = 0; d < batch_rank; ++d) {
        utils::check(gshape[d] >= grid[d],
            "distributed_matrix_array: batch dim {} ({}) smaller than its grid ({}).",
            d, gshape[d], grid[d]);
        auto [a, b] = itertools::chunk_range(0, gshape[d], grid[d], coord[d]);
        _origin[d] = a;
        _lbatch[d] = b - a;
        _bsize[d]  = 1;
      }
    }
    _origin[rank-2] = 0;
    _origin[rank-1] = 0;
    _nlocal_batch = 1;
    for (int d = 0; d < batch_rank; ++d) _nlocal_batch *= _lbatch[d];

    // ---- matrix communicator (collective; must be unconditional) ---------------------
    _mat_comm = comm.split(int(batch_id), comm.rank());
    utils::check(_mat_comm->size() == _p * _q,
        "distributed_matrix_array: matrix communicator size {} != p*q = {}",
        _mat_comm->size(), _p * _q);
    utils::check(_mat_comm->rank() == my_matrix_rank_(),
        "distributed_matrix_array: matrix comm rank {} != pr*q+qr = {}",
        _mat_comm->rank(), my_matrix_rank_());

    // ---- local tiles -----------------------------------------------------------------
    for (long i = 0; i < _mt; ++i) if (i % _p == _pr) _local_it.push_back(i);
    for (long j = 0; j < _nt; ++j) if (j % _q == _qr) _local_jt.push_back(j);

    // ---- offsets: per batch element, column-major over local tiles --------------------
    long nti = long(_local_it.size()), ntj = long(_local_jt.size());
    _offset.resize(std::size_t(std::max(1l, _nlocal_batch) * nti * ntj), 0);
    long off = 0;
    for (long ib = 0; ib < _nlocal_batch; ++ib) {
      for (long b = 0; b < ntj; ++b) {
        for (long a = 0; a < nti; ++a) {
          _offset[std::size_t((ib*ntj + b)*nti + a)] = off;
          off += tileMb_(_local_it[a]) * tileNb_(_local_jt[b]);
        }
      }
    }
    _tiles_per_batch = (_nlocal_batch > 0 ? off / _nlocal_batch : 0);

#if defined(ENABLE_CUDA)
    if constexpr (not on_host) cudaGetDevice(std::addressof(_dev));
#endif

    _buf = Buffer_t(std::max(1l, off));
    set(value_type(0));
  }

  // Copying would duplicate a (possibly large) device buffer and a communicator; require an
  // explicit decision instead. Moves are fine.
  distributed_matrix_array(distributed_matrix_array const&) = delete;
  distributed_matrix_array& operator=(distributed_matrix_array const&) = delete;
  distributed_matrix_array(distributed_matrix_array&&) = default;
  distributed_matrix_array& operator=(distributed_matrix_array&&) = default;

  // ---- bookkeeping --------------------------------------------------------------------
  communicator_t* communicator() const { return _comm; }
  std::array<long, rank> const& grid()         const { return _grid; }
  std::array<long, rank> const& global_shape() const { return _gshape; }
  std::array<long, rank> const& origin()       const { return _origin; }
  std::array<long, rank> const& block_size()   const { return _bsize; }

  long mb() const { return _mb; }
  long nb() const { return _nb; }
  long mt() const { return _mt; }
  long nt() const { return _nt; }
  long rows() const { return _M; }
  long cols() const { return _N; }

  static constexpr bool is_stride_order_Fortran() noexcept { return true; }
  static constexpr bool is_stride_order_C() noexcept { return false; }

  ::nda::range local_range(int d) const {
    utils::check(d >= 0 and d < batch_rank,
        "distributed_matrix_array::local_range: batch dim out of range: {}", d);
    return ::nda::range(_origin[d], _origin[d] + _lbatch[d]);
  }

  long n_local_batch() const { return _nlocal_batch; }
  std::array<long, batch_rank> local_batch_extents() const { return _lbatch; }

  // ---- decomposition, exposed so converters do not re-derive it -----------------------
  // (re-deriving the grid order in a second place is exactly what broke this class once)
  long matrix_grid_rows() const { return _p; }
  long matrix_grid_cols() const { return _q; }
  long n_batch_cells()    const { return _comm->size() / (_p * _q); }
  long my_batch_cell()    const { return _comm->rank() / (_p * _q); }
  long tileMb(long i) const { return tileMb_(i); }
  long tileNb(long j) const { return tileNb_(j); }

  /// matrix-grid rank (index within the matrix communicator) owning tile (i,j)
  long tile_owner(long i, long j) const { return tile_rank_(i,j); }

  /// global rank owning tile (i,j) of batch cell `cell`
  long global_owner(long cell, long i, long j) const { return cell*(_p*_q) + tile_rank_(i,j); }

  /// batch range along dimension d for an arbitrary batch cell
  ::nda::range batch_range_of_cell(long cell, int d) const {
    utils::check(d >= 0 and d < batch_rank, "batch_range_of_cell: bad dim {}", d);
    long rem = cell;
    std::array<long, batch_rank> coord{};
    for (int k = batch_rank - 1; k >= 0; --k) { coord[k] = rem % _grid[k]; rem /= _grid[k]; }
    auto [a, b] = itertools::chunk_range(0, _gshape[d], _grid[d], coord[d]);
    return ::nda::range(a, b);
  }

  /// global multi-index of local batch element ib
  std::array<long, batch_rank> local_batch_index(long ib) const {
    utils::check(ib >= 0 and ib < _nlocal_batch, "local_batch_index out of range");
    std::array<long, batch_rank> c{};
    long rem = ib;
    for (int d = batch_rank - 1; d >= 0; --d) { c[d] = rem % _lbatch[d]; rem /= _lbatch[d]; }
    for (int d = 0; d < batch_rank; ++d) c[d] += _origin[d];
    return c;
  }

  /// the p*q ranks sharing this rank's batch cell
  communicator_t* matrix_communicator() {
    utils::check(_mat_comm.has_value(), "distributed_matrix_array: not initialized.");
    return std::addressof(*_mat_comm);
  }

  // ---- data access --------------------------------------------------------------------
  Buffer_t&       buffer()       { return _buf; }
  Buffer_t const& buffer() const { return _buf; }
  long buffer_size() const { return _buf.extent(0); }

  long n_local_tile_rows() const { return long(_local_it.size()); }
  long n_local_tile_cols() const { return long(_local_jt.size()); }
  long local_tile_row(long a) const { return _local_it[std::size_t(a)]; }
  long local_tile_col(long b) const { return _local_jt[std::size_t(b)]; }

  bool tile_is_local(long i, long j) const { return tile_rank_(i,j) == my_matrix_rank_(); }

  /// 2D column-major view of local tile (a,b) of local batch element ib.
  /// a, b index the local tile lists, not global tile indices.
  tile_view_t tile(long ib, long a, long b) {
    long nti = long(_local_it.size()), ntj = long(_local_jt.size());
    utils::check(ib >= 0 and ib < _nlocal_batch and a >= 0 and a < nti and b >= 0 and b < ntj,
        "distributed_matrix_array::tile: index out of range ({},{},{})", ib, a, b);
    long m = tileMb_(_local_it[std::size_t(a)]);
    long n = tileNb_(_local_jt[std::size_t(b)]);
    long o = _offset[std::size_t((ib*ntj + b)*nti + a)];
    using lay_t = typename tile_view_t::layout_t;
    return tile_view_t(lay_t{std::array<long,2>{m,n}}, _buf.data() + o);
  }

  // ---- bulk ops -----------------------------------------------------------------------
  void set(value_type v) {
    if (_buf.size() == 0) return;
    if constexpr (on_host) _buf() = v;
    else ::nda::tensor::set(v, _buf);
  }
  void scale(value_type v) {
    if (_buf.size() == 0) return;
    ::nda::tensor::scale(v, _buf);
  }

  /// Set every batch element to the identity matrix.
  /// Only tiles straddling the global diagonal are touched, and each is staged on the host
  /// first: writing single elements through operator() is a host store, which is invalid on
  /// device memory.
  void set_identity() {
    set(value_type(0));
    if (_buf.size() == 0) return;
    for (long a = 0; a < long(_local_it.size()); ++a) {
      long i0 = _local_it[std::size_t(a)] * _mb;
      long m  = tileMb_(_local_it[std::size_t(a)]);
      for (long b = 0; b < long(_local_jt.size()); ++b) {
        long j0 = _local_jt[std::size_t(b)] * _nb;
        long n  = tileNb_(_local_jt[std::size_t(b)]);
        if (i0 + m <= j0 or j0 + n <= i0) continue;   // tile misses the diagonal
        ::nda::array<value_type, 2, ::nda::F_layout> h(m, n);
        h() = value_type(0);
        for (long r = 0; r < m; ++r) {
          long c = i0 + r - j0;
          if (c >= 0 and c < n) h(r, c) = value_type(1);
        }
        for (long ib = 0; ib < _nlocal_batch; ++ib) tile(ib, a, b) = h;
      }
    }
  }

#if defined(ENABLE_SLATE)
  /// slate::Matrix over local batch element ib. The returned matrix references this
  /// container's buffer; it must not outlive it.
  slate::Matrix<value_type> slate_matrix(long ib) {
    using ij_tuple = std::tuple<int64_t, int64_t>;
    utils::check(ib >= 0 and ib < _nlocal_batch,
        "distributed_matrix_array::slate_matrix: batch index out of range");

    long M = _M, N = _N, mb = _mb, nb = _nb, p = _p, q = _q;
    std::function<int64_t (int64_t)> tileMb =
        [M, mb](int64_t i) { return std::min<int64_t>(mb, M - i*mb); };
    std::function<int64_t (int64_t)> tileNb =
        [N, nb](int64_t j) { return std::min<int64_t>(nb, N - j*nb); };
    std::function<int (ij_tuple)> tileRank = [p, q](ij_tuple ij) {
      // row-major, MUST match tile_rank_() / my_matrix_rank_()
      return int((std::get<0>(ij) % p) * q + (std::get<1>(ij) % q));
    };
    int dev = _dev;
    std::function<int (ij_tuple)> tileDevice = [dev](ij_tuple) { return dev; };

    [[maybe_unused]] int dev_before = 0;
#if defined(ENABLE_CUDA)
    cudaGetDevice(std::addressof(dev_before));
#endif
    slate::Matrix<value_type> R(M, N, tileMb, tileNb, tileRank, tileDevice,
                                matrix_communicator()->get());
#if defined(ENABLE_CUDA)
    cudaSetDevice(dev_before);  // slate may switch devices while creating queues
#endif

    utils::check(R.mt() == _mt and R.nt() == _nt,
        "distributed_matrix_array: slate tile count mismatch ({},{}) vs ({},{})",
        R.mt(), R.nt(), _mt, _nt);

    long nti = long(_local_it.size()), ntj = long(_local_jt.size());
    for (long b = 0; b < ntj; ++b) {
      for (long a = 0; a < nti; ++a) {
        long i = _local_it[std::size_t(a)], j = _local_jt[std::size_t(b)];
        utils::check(R.tileIsLocal(i,j),
            "distributed_matrix_array: ownership disagrees with slate at tile ({},{})", i, j);
        long ld = tileMb_(i);
        auto* ptr = _buf.data() + _offset[std::size_t((ib*ntj + b)*nti + a)];
        if constexpr (on_host) R.tileInsert(i, j, ptr, ld);
        else                   R.tileInsert(i, j, dev, ptr, ld);
      }
    }
    return R;
  }

  /// Debug helper: verify our ownership map agrees with slate's for every tile.
  bool ownership_matches_slate() {
    if (_nlocal_batch == 0) return true;
    auto R = slate_matrix(0);
    long me = my_matrix_rank_();
    for (long j = 0; j < _nt; ++j)
      for (long i = 0; i < _mt; ++i)
        if (R.tileIsLocal(i,j) != (tile_rank_(i,j) == me)) return false;
    return true;
  }
#endif
};

} // namespace math::nda

namespace memory
{
/// convenience alias: dmatrix_array_t<MEM, T, batch_rank, comm>
template<MEMORY_SPACE MEM, typename T, int batch_rank, typename comm_t>
using dmatrix_array_t =
    ::math::nda::distributed_matrix_array<memory::array<MEM, T, 1>, batch_rank, comm_t>;
}

#endif
