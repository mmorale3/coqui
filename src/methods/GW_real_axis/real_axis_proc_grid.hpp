/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 *
 * Process-grid helpers for distributed real-axis arrays.
 *
 * Convention for any tensor with a (..., Naux, Naux, ...) auxiliary block:
 * the proc grid distributes ONLY the two Naux axes, never the spectral
 * (q, Omega, w) or BZ (s, k) axes -- the kernels need full slices along
 * those.
 */

#ifndef COQUI_REAL_AXIS_PROC_GRID_HPP
#define COQUI_REAL_AXIS_PROC_GRID_HPP

#include <array>
#include <cmath>
#include <limits>
#include "utilities/check.hpp"
#include "utilities/proc_grid_partition.hpp"

namespace methods {
namespace real_axis {

/**
 * Factor `nproc` as gridP * gridQ with both <= max_dim and minimum |gridP-gridQ|.
 */
inline std::array<long, 2> square_factor_capped(long nproc, long max_dim) {
  long best_P = -1;
  long best_diff = std::numeric_limits<long>::max();
  for (long P = 1; P <= max_dim; ++P) {
    if (nproc % P != 0) continue;
    long Q = nproc / P;
    if (Q > max_dim) continue;
    long diff = std::abs(P - Q);
    if (diff < best_diff) {
      best_diff = diff;
      best_P = P;
    }
  }
  utils::check(best_P > 0,
               "real_axis::square_factor_capped: cannot factor nproc ({}) "
               "as P*Q with both <= max_dim ({}). nproc > max_dim^2 is unsupported.",
               nproc, max_dim);
  long P = best_P;
  long Q = nproc / P;
  // Convention: gridP <= gridQ.
  return {std::min(P, Q), std::max(P, Q)};
}

/**
 * 4D proc grid for bosonic objects of shape (Nq, Naux, Naux, N_Omega).
 * Distributes only over (P, Q): grid = (1, gridP, gridQ, 1) with
 * gridP * gridQ = nproc, both <= Naux, and gridP, gridQ as close as possible.
 */
inline std::array<long, 4> bosonic_proc_grid(long nproc, long Naux) {
  auto [gridP, gridQ] = square_factor_capped(nproc, Naux);
  return {1, gridP, gridQ, 1};
}

/**
 * 4D block size for bosonic dArrays. Picks the largest square block that
 * still leaves at least one full block per rank along each Naux axis.
 */
inline std::array<long, 4> bosonic_block_size(std::array<long, 4> grid,
                                              long Nq, long Naux, long N_O,
                                              long max_block = 1024) {
  long bP = std::min({max_block, Naux / grid[1], Naux / grid[2]});
  long bQ = bP;
  return {1, std::max(1l, bP), std::max(1l, bQ), 1};
  (void) Nq;
  (void) N_O;
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_PROC_GRID_HPP
