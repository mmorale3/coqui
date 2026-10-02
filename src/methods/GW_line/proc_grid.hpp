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

#ifndef COQUI_METHODS_GW_LINE_PROC_GRID_HPP
#define COQUI_METHODS_GW_LINE_PROC_GRID_HPP

/**
 * The ONE 2D process grid of the line GW (notes/line_gw_cpp_plan.md section 6.2): `comm` -> np_P x np_Q over the
 * auxiliary (THC) indices. Every Np x Np object (G~(k,t), Pi(q,t), Pi(q,zeta), W, residues, Sigma~(k,t)) is held as the
 * local block (P_rng, Q_rng) for ALL k, q, t, zeta; k, q and the time nodes are local loops.
 *
 * Rank -> coordinates and the per-dimension chunks follow math::nda::make_distributed_array for a C-layout darray with
 * grid {np_P, np_Q} and block size 1 (row major over the grid: ip_Q = rank % np_Q, ip_P = rank / np_Q;
 * itertools::chunk_range(0, Np, np, ip)), so that the S4 redistributes of {.., np_P, np_Q} darrays line up with these
 * blocks without any reshuffling.
 */

#include <array>
#include <string>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "itertools/itertools.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "utilities/proc_grid_partition.hpp"

namespace methods::gw_line {

struct aux_grid_t {
  long Np   = 0;   ///< global auxiliary dimension
  long np   = 1;   ///< communicator size
  long rank = 0;   ///< rank in the communicator
  long np_P = 1, np_Q = 1;   ///< process grid
  long ip_P = 0, ip_Q = 0;   ///< coordinates of this rank
  long P0 = 0, nP = 0;       ///< rows    P_rng = [P0, P0 + nP)
  long Q0 = 0, nQ = 0;       ///< columns Q_rng = [Q0, Q0 + nQ)

  aux_grid_t() = default;

  /// Grid of `np_` ranks over an Np x Np matrix, seen from rank `rank_`.
  aux_grid_t(long np_, long rank_, long Np_) : Np(Np_), np(np_), rank(rank_) {
    utils::check(np > 0 and rank >= 0 and rank < np, "aux_grid_t: invalid np={} rank={}", np, rank);
    np_P = utils::find_proc_grid_min_diff(np, 1, 1);   // the larger factor (convention of eval_Pi_rpa_Rspace)
    np_Q = np / np_P;
    utils::check(np_P * np_Q == np, "aux_grid_t: np_P*np_Q != np");
    utils::check(np_P <= Np and np_Q <= Np, "aux_grid_t: too many processors ({} x {}) for Np = {}", np_P, np_Q, Np);
    ip_Q = rank % np_Q;
    ip_P = rank / np_Q;
    auto [p0, p1] = itertools::chunk_range(0, Np, np_P, ip_P);
    auto [q0, q1] = itertools::chunk_range(0, Np, np_Q, ip_Q);
    P0 = p0; nP = p1 - p0;
    Q0 = q0; nQ = q1 - q0;
  }

  template <typename comm_t>
  aux_grid_t(utils::mpi_context_t<comm_t> const &mpi, long Np_) : aux_grid_t(long(mpi.comm.size()), long(mpi.comm.rank()), Np_) {}

  nda::range P_rng() const { return nda::range(P0, P0 + nP); }
  nda::range Q_rng() const { return nda::range(Q0, Q0 + nQ); }
  std::array<long, 2> block() const { return {nP, nQ}; }
  long block_size() const { return nP * nQ; }
  /// largest local block over the grid (chunk_range gives the first ranks the extra row/column)
  long max_block_size() const { return ((Np + np_P - 1) / np_P) * ((Np + np_Q - 1) / np_Q); }

  /**
   * Per-rank memory model of plan section 6.7 (bytes; root rank, largest block). Resident: Z(q) blocks, W residues
   * w_j(q), the Pi(q, zeta) group, the X slices (four variants, see propagator_t); transient per time chunk: G~ blocks
   * (2 sectors x 2 variants x N_k), acc (max(N_q, N_k) blocks per time), W(q, t). The S3 polarization alone needs
   * Pi (N_q N_zeta blocks) + 2 N_k t_chunk blocks (A, B of one sector) + 2 t_chunk blocks (acc, tmp).
   */
  void log(long nk, long nq, long nzeta, long r_b, long t_chunk, long nb, long g = -1) const {
    if (g < 0) g = nq;
    const double GB = 1024.0 * 1024.0 * 1024.0;
    const double blk = double(max_block_size()) * 16.0;
    const double xsl = double(nk) * double(2 * ((Np + np_P - 1) / np_P + (Np + np_Q - 1) / np_Q)) * double(nb) * 16.0;
    const double z = nq * blk, w = double(nq) * r_b * blk, pig = double(g) * nzeta * blk;
    const double gt = 4.0 * nk * t_chunk * blk, acc = double(std::max(nq, nk)) * t_chunk * blk, wt = t_chunk * blk;
    const double pi_s3 = double(nq) * nzeta * blk + 2.0 * nk * t_chunk * blk + 2.0 * t_chunk * blk;
    app_log(2, "  gw_line aux grid: {} ranks -> (P,Q) = ({} x {}), Np = {}, block <= {} x {} ({:.3f} MB)", np, np_P, np_Q,
            Np, (Np + np_P - 1) / np_P, (Np + np_Q - 1) / np_Q, blk / 1024.0 / 1024.0);
    app_log(2, "    memory model per rank (N_k={}, N_q={}, N_zeta={}, r_b={}, t_chunk={}, g={}), GB:", nk, nq, nzeta, r_b,
            t_chunk, g);
    app_log(2, "      resident : Z {:.4f}  w {:.4f}  Pi-group {:.4f}  X slices {:.4f}  -> {:.4f}", z / GB, w / GB, pig / GB,
            xsl / GB, (z + w + pig + xsl) / GB);
    app_log(2, "      transient: G~ {:.4f}  acc {:.4f}  W(q,t) {:.4f}  -> {:.4f}", gt / GB, acc / GB, wt / GB,
            (gt + acc + wt) / GB);
    app_log(2, "      S3 polarization alone (Pi + A,B of one sector + acc,tmp + X): {:.4f}", (pi_s3 + xsl) / GB);
  }
};

} // namespace methods::gw_line

#endif
