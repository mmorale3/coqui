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

#ifndef COQUI_METHODS_GW_LINE_K_DIST_HPP
#define COQUI_METHODS_GW_LINE_K_DIST_HPP

/**
 * k distribution of the per-k host objects of the line GW (S7e): Sigma(k, zeta) at the fermionic nodes is owned by ONE
 * rank, owner(k) = k mod np, local row l = k / np. This is the round-robin ownership of the closure and the spectra
 * (closure.hpp, spectra.hpp), so the self-energy's reduce-scatter delivers every Sigma(k) exactly where the closure uses
 * it; nothing of size N_k N_zeta nb^2 is replicated over the ranks (before S7e: Sig_p, Sig_h, Sp_new, Sh_new on every
 * rank, 4 N_k N_zeta nb^2 x 16 B, e.g. 3.5 GB per rank for Si 4x4x4 nb 60 with 240 nodes).
 *
 * Buffers in "owner order": the rows of rank 0 (k = 0, np, 2np, ...), then those of rank 1, ...; a row is `row` elements.
 * reduce_scatter / gather_root / scatter_root move such buffers with MPI_Reduce_scatter / MPI_Gatherv / MPI_Scatterv.
 */

#include <climits>
#include <complex>
#include <vector>

#include <mpi.h>

#include "configuration.hpp"
#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"

namespace methods::gw_line {

struct k_dist_t {
  long nk = 0, np = 1, rank = 0;

  k_dist_t() = default;
  k_dist_t(long nk_, long np_, long rank_) : nk(nk_), np(np_), rank(rank_) {
    utils::check(nk > 0 and np > 0 and rank >= 0 and rank < np, "gw_line::k_dist_t: invalid nk={} np={} rank={}", nk, np, rank);
  }
  k_dist_t(long nk_, boost::mpi3::communicator &comm) : k_dist_t(nk_, long(comm.size()), long(comm.rank())) {}

  long owner(long k) const { return k % np; }
  long local(long k) const { return k / np; }
  long global(long l, long r) const { return r + l * np; }
  /// number of k owned by rank r
  long nloc(long r) const { return r < nk ? (nk - 1 - r) / np + 1 : 0; }
  long nloc() const { return nloc(rank); }
  bool owns(long k) const { return owner(k) == rank; }
  /// first row of rank r in an owner-ordered buffer
  long offset(long r) const {
    long o = 0;
    for (long i = 0; i < r; ++i) o += nloc(i);
    return o;
  }
};

namespace detail {

inline int mpi_count(long n) {
  utils::check(n >= 0 and n <= long(INT_MAX), "gw_line::k_dist: MPI count {} beyond the int range", n);
  return int(n);
}

/// counts / displacements (elements) of an owner-ordered buffer with `row` elements per k
inline void kd_counts(k_dist_t const &kd, long row, std::vector<int> &cnt, std::vector<int> &dsp) {
  cnt.assign(kd.np, 0);
  dsp.assign(kd.np, 0);
  long o = 0;
  for (long r = 0; r < kd.np; ++r) {
    cnt[r] = mpi_count(kd.nloc(r) * row);
    dsp[r] = mpi_count(o);
    o += kd.nloc(r) * row;
  }
}

} // namespace detail

/**
 * Sum over the ranks of `send` (owner order, N_k rows of `row` elements, identical layout on all ranks) and deliver the
 * rows of k owned by this rank into `recv` (nloc rows). MPI_Reduce_scatter: half the volume of an all_reduce and no
 * replicated result.
 */
inline void kd_reduce_scatter(boost::mpi3::communicator &comm, k_dist_t const &kd, ComplexType const *send, ComplexType *recv,
                              long row) {
  if (kd.np == 1) {
    std::copy_n(send, kd.nk * row, recv);
    return;
  }
  std::vector<int> cnt, dsp;
  detail::kd_counts(kd, row, cnt, dsp);
  MPI_Reduce_scatter(send, recv, cnt.data(), MPI_C_DOUBLE_COMPLEX, MPI_SUM, comm.get());
}

/// rows of all ranks -> root, owner order (root: `all` holds N_k rows; elsewhere unused)
inline void kd_gather_root(boost::mpi3::communicator &comm, k_dist_t const &kd, ComplexType const *mine, ComplexType *all,
                           long row) {
  if (kd.np == 1) {
    std::copy_n(mine, kd.nk * row, all);
    return;
  }
  std::vector<int> cnt, dsp;
  detail::kd_counts(kd, row, cnt, dsp);
  MPI_Gatherv(mine, cnt[kd.rank], MPI_C_DOUBLE_COMPLEX, all, cnt.data(), dsp.data(), MPI_C_DOUBLE_COMPLEX, 0, comm.get());
}

/// root's owner-ordered rows -> every rank's own rows
inline void kd_scatter_root(boost::mpi3::communicator &comm, k_dist_t const &kd, ComplexType const *all, ComplexType *mine,
                            long row) {
  if (kd.np == 1) {
    std::copy_n(all, kd.nk * row, mine);
    return;
  }
  std::vector<int> cnt, dsp;
  detail::kd_counts(kd, row, cnt, dsp);
  MPI_Scatterv(all, cnt.data(), dsp.data(), MPI_C_DOUBLE_COMPLEX, mine, cnt[kd.rank], MPI_C_DOUBLE_COMPLEX, 0, comm.get());
}

/// k-ordered (N_k, ...) <-> owner-ordered (N_k, ...) row permutations of 4D host arrays (first index k)
inline void kd_to_owner_order(k_dist_t const &kd, nda::array<ComplexType, 4> const &kordered, nda::array<ComplexType, 4> &owner) {
  auto all = nda::range::all;
  owner.resize(kordered.shape());
  long o = 0;
  for (long r = 0; r < kd.np; ++r)
    for (long l = 0; l < kd.nloc(r); ++l, ++o) owner(o, all, all, all) = kordered(kd.global(l, r), all, all, all);
}
inline void kd_from_owner_order(k_dist_t const &kd, nda::array<ComplexType, 4> const &owner, nda::array<ComplexType, 4> &kordered) {
  auto all = nda::range::all;
  kordered.resize(owner.shape());
  long o = 0;
  for (long r = 0; r < kd.np; ++r)
    for (long l = 0; l < kd.nloc(r); ++l, ++o) kordered(kd.global(l, r), all, all, all) = owner(o, all, all, all);
}

/// local rows (nloc, a, b, c) of every rank -> the full k-ordered array on the root (empty elsewhere)
inline nda::array<ComplexType, 4> kd_gather_full(boost::mpi3::communicator &comm, k_dist_t const &kd,
                                                 nda::array<ComplexType, 4> const &loc) {
  std::array<long, 3> sh = {loc.extent(1), loc.extent(2), loc.extent(3)};
  comm.broadcast_n(sh.data(), 3, 0);
  const long row = sh[0] * sh[1] * sh[2];
  utils::check(loc.extent(0) == kd.nloc() and (kd.nloc() == 0 or loc.size() == kd.nloc() * row),
               "gw_line::kd_gather_full: local rows ({}) vs owned k ({})", loc.extent(0), kd.nloc());
  nda::array<ComplexType, 4> own(comm.root() ? kd.nk : 0, sh[0], sh[1], sh[2]), full;
  nda::array<ComplexType, 4> mine(loc);   // contiguous
  kd_gather_root(comm, kd, mine.data(), own.data(), row);
  if (comm.root()) kd_from_owner_order(kd, own, full);
  return full;
}

/// the full k-ordered array on the root -> every rank's local rows (nloc, a, b, c)
inline nda::array<ComplexType, 4> kd_scatter_full(boost::mpi3::communicator &comm, k_dist_t const &kd,
                                                  nda::array<ComplexType, 4> const &full) {
  std::array<long, 3> sh{};
  if (comm.root()) sh = {full.extent(1), full.extent(2), full.extent(3)};
  comm.broadcast_n(sh.data(), 3, 0);
  const long row = sh[0] * sh[1] * sh[2];
  nda::array<ComplexType, 4> own, loc(kd.nloc(), sh[0], sh[1], sh[2]);
  if (comm.root()) {
    utils::check(full.extent(0) == kd.nk, "gw_line::kd_scatter_full: {} rows vs N_k = {}", full.extent(0), kd.nk);
    kd_to_owner_order(kd, full, own);
  }
  kd_scatter_root(comm, kd, own.data(), loc.data(), row);
  return loc;
}

} // namespace methods::gw_line

#endif
