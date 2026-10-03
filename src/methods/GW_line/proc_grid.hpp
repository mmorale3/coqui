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

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "itertools/itertools.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "utilities/proc_grid_partition.hpp"
#include "utilities/freemem.h"

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
   * Per-rank memory model of plan section 6.7, as the S7d code holds it (bytes; largest block; returned: the predicted
   * high-water = resident + the largest stage transient). blk = block bytes.
   *   resident : Z(q) blocks (N_q blk), residues w (N_q r_b blk), the Pi(q, zeta) group (g N_zeta blk), X slices.
   *   Pi stage : A, B of ONE sector for all k (2 N_k t_chunk blk) + acc (device fused: N_q t_chunk blk, all q of a chunk;
   *              host / cuTENSOR: t_chunk blk).
   *   W stage  : w_plan_t::transient_bytes (the W^T rows of one q sub-step, the block and whole-matrix zeta sub-slabs, the
   *              Dyson scratch, the fit buffer, the redistribute staging); Pi itself is reused for W.
   *   Sigma    : G~ and acc of all k (2 N_k t_chunk blk) + W(q, chunk) (device fused: all q, N_q t_chunk blk; else one)
   *              + the device Sigma accumulator N_k N_zeta_f nb^2 (not counted: N_zeta_f is not known here).
   * t_chunk: the chunk of the kernels (device automatic: the kernels pick it at call time from the free memory).
   */
  double log(long nk, long nq, long nzeta, long r_b, long t_chunk, long nb, long g = -1, bool device_fused = false) const;
};

/**
 * Whole-matrix layout of the Dyson step: 4D darray {g, N_zeta, Np, Np} with grid {np_q, np_z, 1, 1}. np_q is the
 * largest number of q pools (<= g, dividing np, load imbalance <= 20%: find_proc_grid_max_npools), so that each rank's
 * full-Z footprint (its q subset x Np^2) is minimal; np_z = np / np_q. Rank -> coordinates and the chunks follow
 * math::nda::make_distributed_array for a C-layout darray with block size 1 (ip_z = rank % np_z, ip_q = rank / np_z).
 * q indices are relative to the group (q0 of screened_interaction). (Moved here from screened.hpp in S7d: the memory model
 * below needs it.)
 */
struct dyson_layout_t {
  long np = 1, rank = 0, g = 0, nz = 0, Np = 0;
  long np_q = 1, np_z = 1, ip_q = 0, ip_z = 0;
  long q_first = 0, nq_loc = 0;   ///< local q slab [q_first, q_first + nq_loc) (group-relative)
  long z_first = 0, nz_loc = 0;   ///< local zeta slab

  dyson_layout_t() = default;
  dyson_layout_t(long np_, long rank_, long g_, long nz_, long Np_) : np(np_), rank(rank_), g(g_), nz(nz_), Np(Np_) {
    utils::check(np > 0 and rank >= 0 and rank < np and g > 0 and nz > 0 and Np > 0,
                 "dyson_layout_t: invalid np={} rank={} g={} nz={} Np={}", np, rank, g, nz, Np);
    np_q = utils::find_proc_grid_max_npools(np, g, 0.2);
    np_z = np / np_q;
    utils::check(np_q * np_z == np, "dyson_layout_t: np_q*np_z != np");
    utils::check(np_q <= g and np_z <= nz, "dyson_layout_t: too many ranks ({} x {}) for g={} nz={}", np_q, np_z, g, nz);
    ip_z = rank % np_z;
    ip_q = rank / np_z;
    auto [q0, q1] = itertools::chunk_range(0, g, np_q, ip_q);
    auto [z0, z1] = itertools::chunk_range(0, nz, np_z, ip_z);
    q_first = q0; nq_loc = q1 - q0;
    z_first = z0; nz_loc = z1 - z0;
  }
  template <typename comm_t>
  dyson_layout_t(utils::mpi_context_t<comm_t> const &mpi, long g_, long nz_, long Np_)
     : dyson_layout_t(long(mpi.comm.size()), long(mpi.comm.rank()), g_, nz_, Np_) {}

  std::array<long, 4> pgrid() const { return {np_q, np_z, 1, 1}; }
  nda::range q_rng() const { return nda::range(q_first, q_first + nq_loc); }
  nda::range z_rng() const { return nda::range(z_first, z_first + nz_loc); }
  /// largest slab over the grid (elements)
  long max_slab() const { return ((g + np_q - 1) / np_q) * ((nz + np_z - 1) / np_z) * Np * Np; }

  void log() const {
    app_log(2, "  gw_line Dyson layout: {} ranks -> (q, zeta) pools = ({} x {}) over (g, N_zeta) = ({}, {}); slab <= {} "
               "matrices ({:.3f} GB), full Z for <= {} q per rank",
            np, np_q, np_z, g, nz, max_slab() / (Np * Np), double(max_slab()) * 16.0 / 1024.0 / 1024.0 / 1024.0,
            (g + np_q - 1) / np_q);
  }
};

/**
 * Sub-steps of the W stage (S7d, screened_interaction): the Dyson of a q group runs in sub-steps so that only a fraction
 * of the Pi group is ever held twice (before S7d: Pi, the whole-matrix slab and the W^T slab coexisted, 3 Pi-group sizes).
 * Sub-step (s, zeta range):
 *   q rows : local q index s of every q pool of dyson_layout_t (pool p solves q = q_first(p) + s; n_act(s) pools take
 *            part), so every rank solves the q's of its full-group Dyson slab, whose full Z(q) coulomb_blocks_t keeps;
 *   zeta   : [z_a, z_a + nzs), split over the zeta pools with chunk_range as in dyson_layout_t.
 * Per sub-step: Pi rows -> block buffer Tb (n_act, nzs, nP, nQ) -> whole-matrix buffer D (Dyson IN PLACE, giving W^T) ->
 * Tb (W^T blocks) -> WT rows; D transposed in place (W) -> Tb -> Pi rows (W blocks). After the zeta loop, the fit of the
 * n_act q's reads W (Pi rows) and W^T (WT). Held besides the resident set: WT (n_act N_zeta blk), Tb (n_act nzs blk),
 * D (ceil(nzs / np_z) Np^2), the Dyson scratch, the fit buffer Y (k blk) and the redistribute staging.
 * nzs: the largest value that keeps Tb + D within max(S/8, S - WT) (S = g N_zeta blk, the Pi group), >= np_z; the env
 * COQUI_GWLINE_W_ZSUB forces it, nzs_cap (device: from the free memory) caps it. Must be identical on all ranks (callers
 * reduce nzs_cap with a min over the communicator).
 */
struct w_plan_t {
  dyson_layout_t lay;
  long blk = 0;                                 ///< largest block (elements)
  long nsub_q = 0, base = 0, extra = 0, nzs = 0;

  w_plan_t() = default;
  w_plan_t(dyson_layout_t const &lay_, long max_block, long nzs_cap = -1) : lay(lay_), blk(max_block) {
    base   = lay.g / lay.np_q;
    extra  = lay.g % lay.np_q;
    nsub_q = base + (extra > 0 ? 1 : 0);
    utils::check(lay.nq_loc == base + (lay.ip_q < extra ? 1 : 0), "gw_line::w_plan_t: unexpected q pool size {} (g {}, np_q {})",
                 lay.nq_loc, lay.g, lay.np_q);
    const double S = double(lay.g) * lay.nz * blk, WT = double(lay.np_q) * lay.nz * blk;
    const double avail = std::max(S / 8.0, S - WT);
    // Tb + D per zeta node of the sub-slab: np_q blk + Np^2 / np_z (= 2 np_q blk on a square grid)
    const double per_z = double(lay.np_q) * blk + double(lay.Np) * lay.Np / double(lay.np_z);
    nzs = long(avail / per_z);
    if (nzs_cap > 0) nzs = std::min(nzs, nzs_cap);
    if (char const *v = std::getenv("COQUI_GWLINE_W_ZSUB"); v != nullptr and *v != '\0') nzs = std::strtol(v, nullptr, 10);
    nzs = std::clamp(nzs, std::min(lay.np_z, lay.nz), lay.nz);
  }
  long n_act(long s) const { return s < base ? lay.np_q : extra; }
  /// group-relative q solved by pool p in sub-step s
  long q_row(long p, long s) const {
    auto [q0, q1] = itertools::chunk_range(0, lay.g, lay.np_q, p);
    return q0 + s;
  }
  /// this rank's zeta chunk [first, first + n) of a sub-slab of n_z nodes
  std::array<long, 2> z_chunk(long n_z) const {
    auto [z0, z1] = itertools::chunk_range(0, n_z, lay.np_z, lay.ip_z);
    return {long(z0), long(z1 - z0)};
  }
  long n_zsub() const { return (lay.nz + nzs - 1) / nzs; }
  /// per-rank bytes held by the W stage besides the resident set (k_fit: rows of the fit buffer, nbat: Dyson scratch
  /// matrices, staging: redistribute staging bytes)
  double transient_bytes(long k_fit, long nbat, double staging) const {
    const double N2 = double(lay.Np) * lay.Np, nzl = double((nzs + lay.np_z - 1) / lay.np_z);
    return 16.0 * (double(lay.np_q) * lay.nz * blk + double(lay.np_q) * nzs * blk + nzl * N2 + double(nbat) * N2 +
                   double(k_fit) * blk) + staging;
  }
  void log() const {
    app_log(3, "  gw_line W sub-steps: {} q sub-steps (<= {} q per step, one per q pool) x {} zeta sub-slabs of <= {} nodes",
            nsub_q, lay.np_q, n_zsub(), nzs);
  }
};

inline double aux_grid_t::log(long nk, long nq, long nzeta, long r_b, long t_chunk, long nb, long g, bool device_fused) const {
  if (g < 0) g = nq;
  const double GB  = 1024.0 * 1024.0 * 1024.0;
  const long mb    = max_block_size();
  const double blk = double(mb) * 16.0;
  const double xsl = double(nk) * double(2 * ((Np + np_P - 1) / np_P + (Np + np_Q - 1) / np_Q)) * double(nb) * 16.0;
  const double z = nq * blk, w = double(nq) * r_b * blk, pig = double(g) * nzeta * blk;
  const double nacc = device_fused ? double(nq) : 1.0;
  const double pi_t = (2.0 * nk + nacc) * t_chunk * blk;
  const double sg_t = (2.0 * nk + nacc) * t_chunk * blk + double(nk) * t_chunk * nb * nb * 16.0;
  dyson_layout_t lay(np, rank, g, nzeta, Np);
  w_plan_t plan(lay, mb);
  const long nbat     = device_fused ? std::min((plan.nzs + lay.np_z - 1) / lay.np_z, 256L) : 1L;
  const double stg    = std::min(2.0 * GB, 2.0 * 16.0 * double(lay.np_q) * plan.nzs * mb);
  const double w_t    = plan.transient_bytes(nzeta, nbat, np > 1 ? stg : 0.0);
  const double res    = z + w + pig + xsl;
  const double peak   = res + std::max({pi_t, w_t, sg_t});
  app_log(2, "  gw_line aux grid: {} ranks -> (P,Q) = ({} x {}), Np = {}, block <= {} x {} ({:.3f} MB)", np, np_P, np_Q, Np,
          (Np + np_P - 1) / np_P, (Np + np_Q - 1) / np_Q, blk / 1024.0 / 1024.0);
  app_log(2, "    memory model per rank (N_k={}, N_q={}, N_zeta={}, r_b={}, t_chunk={}, g={}, {}), GB:", nk, nq, nzeta, r_b,
          t_chunk, g, device_fused ? "device fused" : "host");
  app_log(2, "      resident : Z {:.4f}  w {:.4f}  Pi-group {:.4f}  X slices {:.4f}  -> {:.4f}", z / GB, w / GB, pig / GB,
          xsl / GB, res / GB);
  app_log(2, "      transient: Pi stage {:.4f} (A, B {} + acc {} chunks)  W stage {:.4f} ({} x {} sub-steps)  Sigma stage {:.4f}",
          pi_t / GB, 2 * nk, long(nacc), w_t / GB, plan.nsub_q, plan.n_zsub(), sg_t / GB);
  app_log(2, "      predicted high-water: {:.4f}", peak / GB);
  return peak;
}

/**
 * Device memory high-water probe (bring-up / benchmarks): device_mem_reset() records the free device memory as the
 * baseline, device_mem_probe() (called by the kernels at their allocation peaks, DEVICE instantiations only) keeps the
 * largest drop below it. Includes everything allocated on the device by this process (cudaMemGetInfo), not only GW_line.
 */
namespace detail {
inline double free_device_bytes() { return double(utils::freemem_device()) * 1048576.0; }   // freemem_device(): MB
inline double &dev_mem_base() { static double v = 0.0; return v; }
inline double &dev_mem_hw() { static double v = 0.0; return v; }
} // namespace detail
inline void device_mem_reset() {
  detail::dev_mem_base() = detail::free_device_bytes();
  detail::dev_mem_hw()   = 0.0;
}
inline void device_mem_probe() {
  if (detail::dev_mem_base() > 0.0)
    detail::dev_mem_hw() = std::max(detail::dev_mem_hw(), detail::dev_mem_base() - detail::free_device_bytes());
}
inline double device_high_water_bytes() { return detail::dev_mem_hw(); }
inline double device_free_bytes() { return detail::free_device_bytes(); }

} // namespace methods::gw_line

#endif
