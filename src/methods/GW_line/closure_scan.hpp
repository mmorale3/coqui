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

#ifndef COQUI_METHODS_GW_LINE_CLOSURE_SCAN_HPP
#define COQUI_METHODS_GW_LINE_CLOSURE_SCAN_HPP

/**
 * perf 7.1c: the terminal-phase scan of the closure (numerics::line_dlr::upfold_block, n_free > 0) distributed over MPI
 * ranks. Before 7.1c the owner of a k with a free block ran 8 coarse + 2 + 30 golden-section + 1 realizations (one
 * Cayley eigensolve of the Nr x Nr unitary U each, Nr ~ 1300 on Si 4x4x4: 41 x ~3 s on one core) while every other rank
 * waited. Here, after every owner has prepared its k (Gram, SVD: upfold_prepare), the k with a free block ("deferred")
 * are scanned together:
 *   - the owner broadcasts the problem (A1, A0, R, C^(K+1)) to the scan ranks of its k: ranks (owner + t) mod np,
 *     t < max(nphi, number of row blocks of R);
 *   - coarse scan: the nphi realizations on nphi ranks (the same eigen realizations and the same decision,
 *     cayley::coarse_decide, as the serial scan);
 *   - golden section (30 steps, the same bracket and final tolerance): each error is the eigensolve-free held-out error
 *     ||R U^{K+1} R^dag - C^(K+1)|| / (1 + ||C^(K+1)||) (cayley::heldout_rows; equal to the realization's error in exact
 *     arithmetic), the fixed row blocks of R on different ranks, partial sums combined by an exact all_reduce (one
 *     contributor per slot) and summed in block order -- bitwise the serial cayley::heldout_mfree for equal BLAS threads;
 *     all deferred k advance in lockstep (one all_reduce per step);
 *   - the owner realizes the final phase (one eigensolve) and finishes its k (callback).
 * Ranks without a scan task lend their cores to the scan ranks of their host (closure_cores_t, Linux + one core per rank)
 * and sleep. The deferred k are processed in batches whose broadcast problems fit in budget_bytes per rank.
 * The result depends on the BLAS thread counts at the roundoff floor only (as the per-k closure, perf 7.1 (f)).
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <vector>

#include <mpi.h>

#include "configuration.hpp"
#include "mpi3/communicator.hpp"
#include "numerics/line_dlr/cayley.hpp"
#include "utilities/check.hpp"
#include "utilities/Timer.hpp"
#include "methods/GW_line/blas_scope.hpp"
#include "methods/GW_line/closure_cores.hpp"

namespace methods::gw_line {

/// result of the distributed scan of one k (identical on the scan ranks of that k)
struct phase_scan_t {
  double phi = 0.0, phi_tie = 0.0;
  long phi_index = -1, n_rejected = 0;
  bool phi_kept = false;
  // the coarse realizations (sum / max over the nphi tasks)
  long n_eig = 0, ueig_nflag = 0, ueig_retry = 0, ueig_fallback = 0, ueig_reason = 0;
  double ueig_res = 0.0, t_eig = 0.0, t_rr = 0.0;
  long n_mfree = 0;                                  ///< eigensolve-free held-out evaluations (each split in row blocks)
  double t_bcast = 0.0, t_coarse = 0.0, t_refine = 0.0;   ///< wall seconds of the steps on the owner
  long ranks = 0, threads = 0;                       ///< scan ranks of the batch, BLAS threads of the owner in the scan
};

namespace detail {
inline void bcast_doubles(double *x, long n, int root, MPI_Comm c) {
  const long chunk = 1L << 27;   // MPI int counts
  for (long o = 0; o < n; o += chunk) MPI_Bcast(x + o, int(std::min(chunk, n - o)), MPI_DOUBLE, root, c);
}
/// A is only read on the root (MPI_Bcast semantics)
inline void bcast_matrix(numerics::line_dlr::cmatrix_F const &A, int root, MPI_Comm c) {
  bcast_doubles(const_cast<double *>(reinterpret_cast<double const *>(A.data())), 2 * long(A.size()), root, c);
}
} // namespace detail

/**
 * Collective over comm. dk: the deferred k in increasing order (identical on every rank), owner(k) = k mod np;
 * prob[j]: the problem of dk[j] on its owner (ignored elsewhere); opts(k): the upfold options of k (identical on every
 * rank; nphi, scan_rows, reject_unity, phase continuity); threads: BLAS threads of a scan rank without borrowed cores
 * (<= 0: untouched); max_threads: cap of the borrowed cores per scan rank (<= 0: none). On the owner of dk[j],
 * finish(j, scan) is called after the scan of its batch, still with the borrowed cores. Returns the seconds this rank
 * spent waiting for the other ranks at the end of the batches (also accumulated in the timer "closure_wait" of T).
 */
template <typename Opts, typename Finish>
double distributed_phase_scan(boost::mpi3::communicator &comm, std::vector<long> const &dk,
                              std::vector<numerics::line_dlr::upfold_problem_t const *> const &prob, Opts &&opts, long threads,
                              long max_threads, double budget_bytes, Finish &&finish, utils::TimerManager *T = nullptr) {
  namespace ldlr = numerics::line_dlr;
  using clk      = std::chrono::steady_clock;
  auto secs      = [](clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); };
  const long np = comm.size(), rank = comm.rank(), D = long(dk.size());
  utils::check(long(prob.size()) == D, "gw_line::distributed_phase_scan: prob size mismatch");
  double t_wait = 0.0;
  if (D == 0) return t_wait;
  auto owner = [&](long j) { return dk[j] % np; };
  // problem dimensions (n, Nr) of every deferred k, from the owners (exact all_reduce)
  std::vector<double> dims(2 * D, 0.0);
  for (long j = 0; j < D; ++j)
    if (owner(j) == rank) {
      utils::check(prob[j] != nullptr, "gw_line::distributed_phase_scan: missing problem of k {} on its owner", dk[j]);
      dims[2 * j]     = double(prob[j]->n);
      dims[2 * j + 1] = double(prob[j]->Nr);
    }
  if (np > 1) comm.all_reduce_in_place_n(dims.data(), 2 * D, std::plus<>{});
  auto bytes_of = [&](long j) {
    const double n = dims[2 * j], Nr = dims[2 * j + 1];
    return 16.0 * (2.0 * Nr * Nr + n * Nr + n * n);
  };

  for (long j0 = 0; j0 < D;) {
    long j1      = j0 + 1;
    double bytes = bytes_of(j0);
    while (j1 < D and bytes + bytes_of(j1) <= budget_bytes) bytes += bytes_of(j1++);
    const long B = j1 - j0;
    std::vector<ldlr::upfold_opts_t> ob(B);
    std::vector<long> nphi(B), rows(B), nblk(B), ntask(B);
    for (long j = 0; j < B; ++j) {
      ob[j]    = opts(dk[j0 + j]);
      nphi[j]  = ob[j].nphi;
      rows[j]  = ob[j].scan_rows;
      nblk[j]  = (long(std::llround(dims[2 * (j0 + j)])) + rows[j] - 1) / rows[j];
      ntask[j] = std::max(nphi[j], nblk[j]);
    }
    auto task_rank = [&](long j, long t) { return (owner(j0 + j) + t) % np; };
    bool in_S = false;
    for (long j = 0; j < B; ++j)
      in_S = in_S or ((rank - owner(j0 + j) + np) % np < ntask[j]);
    MPI_Comm sc = MPI_COMM_NULL;
    MPI_Comm_split(comm.get(), in_S ? 0 : MPI_UNDEFINED, int(rank), &sc);
    closure_cores_t cores(comm, in_S, max_threads);   // the idle ranks lend their cores to the scan ranks of their host
    if (in_S) {
      const long bt = cores.blas_threads() > 0 ? cores.blas_threads() : threads;
      blas_threads_scope_t bscope(bt);
      int ns = 0;
      MPI_Comm_size(sc, &ns);
      std::vector<int> world(ns);
      const int me = int(rank);
      MPI_Allgather(&me, 1, MPI_INT, world.data(), 1, MPI_INT, sc);
      auto sc_rank = [&](long r) { return int(std::find(world.begin(), world.end(), int(r)) - world.begin()); };
      // 1. the problems: owner -> scan ranks of the batch
      const auto tb = clk::now();
      std::vector<ldlr::upfold_problem_t> loc(B);
      std::vector<ldlr::upfold_problem_t const *> P(B);
      for (long j = 0; j < B; ++j) {
        const bool own = (owner(j0 + j) == rank);
        const int root = sc_rank(owner(j0 + j));
        double hdr[6]  = {0, 0, 0, 0, 0, 0};
        if (own) {
          auto const &q = *prob[j0 + j];
          hdr[0] = double(q.n); hdr[1] = double(q.Nr); hdr[2] = double(q.K); hdr[3] = double(q.n_free);
          hdr[4] = q.wp; hdr[5] = q.nheld;
        }
        MPI_Bcast(hdr, 6, MPI_DOUBLE, root, sc);
        if (own) {
          P[j] = prob[j0 + j];
          detail::bcast_matrix(P[j]->A1, root, sc);
          detail::bcast_matrix(P[j]->A0, root, sc);
          detail::bcast_matrix(P[j]->R, root, sc);
          detail::bcast_matrix(P[j]->Cheld, root, sc);
        } else {
          auto &q  = loc[j];
          q.n      = std::llround(hdr[0]);
          q.Nr     = std::llround(hdr[1]);
          q.K      = std::llround(hdr[2]);
          q.n_free = std::llround(hdr[3]);
          q.wp     = hdr[4];
          q.nheld  = hdr[5];
          q.A1     = ldlr::cmatrix_F(q.Nr, q.Nr);
          q.A0     = ldlr::cmatrix_F(q.Nr, q.Nr);
          q.R      = ldlr::cmatrix_F(q.n, q.Nr);
          q.Cheld  = ldlr::cmatrix_F(q.n, q.n);
          detail::bcast_matrix(q.A1, root, sc);
          detail::bcast_matrix(q.A0, root, sc);
          detail::bcast_matrix(q.R, root, sc);
          detail::bcast_matrix(q.Cheld, root, sc);
          P[j] = &q;
        }
      }
      const double t_bcast = secs(tb);

      // 2. coarse scan: task (j, ip) on rank (owner + ip) mod np
      const auto tc = clk::now();
      constexpr long NF = 9;   // err, umin, nflag, retry, fallback, residual, reason, t_eig, t_rr
      std::vector<long> coff(B + 1, 0);
      for (long j = 0; j < B; ++j) coff[j + 1] = coff[j] + nphi[j] * NF;
      std::vector<double> cv(coff[B], 0.0);
      for (long j = 0; j < B; ++j)
        for (long ip = 0; ip < nphi[j]; ++ip) {
          if (task_rank(j, ip) != rank) continue;
          ldlr::upfold_result_t rt;
          auto rz   = ldlr::realize(*P[j], ldlr::coarse_phase(ip, nphi[j]), ob[j], rt);
          double *x = cv.data() + coff[j] + ip * NF;
          x[0] = rz.err; x[1] = ldlr::unity_distance(rz.u); x[2] = double(rt.ueig_nflag); x[3] = double(rt.ueig_retry);
          x[4] = double(rt.ueig_fallback); x[5] = rt.ueig_res; x[6] = double(rt.ueig_reason); x[7] = rt.t_eig; x[8] = rt.t_rr;
        }
      MPI_Allreduce(MPI_IN_PLACE, cv.data(), int(cv.size()), MPI_DOUBLE, MPI_SUM, sc);   // one contributor per slot
      std::vector<phase_scan_t> res(B);
      std::vector<ldlr::golden_t> g(B);
      for (long j = 0; j < B; ++j) {
        std::vector<double> errs(nphi[j]), umin(nphi[j]);
        auto &s = res[j];
        for (long ip = 0; ip < nphi[j]; ++ip) {
          double const *x = cv.data() + coff[j] + ip * NF;
          errs[ip] = x[0];
          umin[ip] = x[1];
          s.ueig_nflag = std::max(s.ueig_nflag, long(std::llround(x[2])));
          s.ueig_retry += std::llround(x[3]);
          s.ueig_fallback += std::llround(x[4]);
          s.ueig_res    = std::max(s.ueig_res, x[5]);
          s.ueig_reason = std::max(s.ueig_reason, long(std::llround(x[6])));
          s.t_eig += x[7];
          s.t_rr += x[8];
        }
        s.n_eig = nphi[j];
        ldlr::upfold_result_t rd;
        g[j]         = ldlr::golden_t(ldlr::coarse_decide(errs, umin, ob[j], rd), nphi[j]);
        s.phi_index  = rd.phi_index;
        s.n_rejected = rd.n_rejected;
        s.phi_kept   = rd.phi_kept;
        s.phi_tie    = rd.phi_tie;
      }
      const double t_coarse = secs(tc);

      // 3. golden section in lockstep: round 0 evaluates c and d, rounds 1..30 one proposed point per k
      const auto tr = clk::now();
      std::vector<long> roff(B + 1, 0);
      for (long j = 0; j < B; ++j) roff[j + 1] = roff[j] + 2 * nblk[j];
      std::vector<double> rv(roff[B]);
      std::vector<std::array<double, 2>> pts(B);
      for (long round = 0; round <= ldlr::golden_t::nsteps; ++round) {
        const long npt = (round == 0) ? 2 : 1;
        for (long j = 0; j < B; ++j) {
          if (round == 0) pts[j] = {g[j].c, g[j].d};
          else pts[j][0] = g[j].propose();
        }
        std::fill(rv.begin(), rv.end(), 0.0);
        for (long j = 0; j < B; ++j)
          for (long q = 0; q < npt; ++q) {
            ldlr::cmatrix_F U;
            for (long b = 0; b < nblk[j]; ++b) {
              if (task_rank(j, b) != rank) continue;
              if (U.size() == 0) U = ldlr::detail::form_u(*P[j], pts[j][q]);
              rv[roff[j] + q * nblk[j] + b] =
                  ldlr::heldout_rows(*P[j], U, b * rows[j], std::min(P[j]->n, (b + 1) * rows[j]));
            }
          }
        MPI_Allreduce(MPI_IN_PLACE, rv.data(), int(rv.size()), MPI_DOUBLE, MPI_SUM, sc);   // one contributor per slot
        for (long j = 0; j < B; ++j)
          for (long q = 0; q < npt; ++q) {
            double s = 0.0;
            for (long b = 0; b < nblk[j]; ++b) s += rv[roff[j] + q * nblk[j] + b];
            const double err = std::sqrt(s) / (1.0 + P[j]->nheld);
            if (round == 0) (q == 0 ? g[j].fc : g[j].fd) = err;
            else g[j].accept(err);
            ++res[j].n_mfree;
          }
      }
      const double t_refine = secs(tr);
      MPI_Comm_free(&sc);
      // 4. the owners finish their k with the borrowed cores
      for (long j = 0; j < B; ++j) {
        if (owner(j0 + j) != rank) continue;
        auto &s    = res[j];
        s.phi      = g[j].result();
        s.t_bcast  = t_bcast;
        s.t_coarse = t_coarse;
        s.t_refine = t_refine;
        s.ranks    = ns;
        s.threads  = bt;
        finish(j0 + j, s);
      }
    }
    const auto tw = clk::now();
    if (T) T->start("closure_wait");
    cores.wait();
    if (T) T->stop("closure_wait");
    t_wait += secs(tw);
    j0 = j1;
  }
  return t_wait;
}

} // namespace methods::gw_line

#endif
