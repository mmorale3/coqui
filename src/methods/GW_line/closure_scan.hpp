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
 * perf 7.1c: the terminal-phase scan of the closure (numerics::line_dlr::upfold_block, n_free > 0) over MPI ranks.
 * Before 7.1c the owner of a k with a free block ran 8 coarse + 2 + 30 golden-section + 1 realizations (one Cayley
 * eigensolve of the Nr x Nr unitary U each; Si 4x4x4, Nr 1302: 41 x ~3 s) while every other rank waited. Here, after
 * every owner has prepared its k up to the fast SVD (upfold_prepare with defer_ref), the k with a free block ("deferred")
 * are finished together:
 *   - scan ranks of k: its owner and the nphi ranks (owner + (1 + ip) stride) mod np, stride = np / (batch (nphi + 1)). Three phases, each with the cores of the
 *     ranks idle in that phase lent to the busy ranks of their host (closure_cores_t; the lenders sleep):
 *   - (A) the owner redoes the SVD with the reference driver (upfold_prepare_ref, the pre-7.1c arithmetic);
 *   - (B) the owner broadcasts the problem (A1, A0, R, C^(K+1)) to the scan ranks;
 *   - coarse scan: the nphi eigen realizations on the nphi coarse ranks (cayley::realize, the same as the serial scan;
 *     one exact all_reduce), while the owner builds the eigensolve-free held-out error (cayley::heldout_poly_t);
 *   - (C) the owner: the same decision (cayley::coarse_decide), the 30-step golden section on the eigensolve-free error
 *     (exact in exact arithmetic; microseconds per phase), the final realization, then finish(j) (Lehmann).
 * The deferred k are processed in batches whose broadcast problems fit in budget_bytes per rank. Each coarse realization
 * is the same code on the same data as in the serial scan: the result depends on the BLAS thread counts / MKL code paths
 * at the roundoff floor only (as the per-k closure, perf 7.1 (f)).
 */

#include <algorithm>
#include <chrono>
#include <cmath>
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
 * prob[j], res[j]: the problem / upfold result of dk[j] on its owner (ignored elsewhere; on return res[j] is the finished
 * upfolding); opts(k): the upfold options of k (identical on every rank); threads: BLAS threads of a scan rank without
 * borrowed cores (<= 0: untouched); max_threads: cap of borrowed cores per busy rank (<= 0: none). On the owner of
 * dk[j], finish(j) is called after the upfolding, still with the borrowed cores. The waits at the end of the batches are accumulated in the timer "closure_wait" of T.
 */
template <typename Opts, typename Finish>
void distributed_phase_scan(boost::mpi3::communicator &comm, std::vector<long> const &dk,
                            std::vector<numerics::line_dlr::upfold_problem_t *> const &prob,
                            std::vector<numerics::line_dlr::upfold_result_t *> const &res, Opts &&opts, long threads,
                            long max_threads, double budget_bytes, Finish &&finish,
                            utils::TimerManager *T = nullptr) {
  namespace ldlr = numerics::line_dlr;
  using clk      = std::chrono::steady_clock;
  auto secs      = [](clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); };
  const long np = comm.size(), rank = comm.rank(), D = long(dk.size());
  utils::check(long(prob.size()) == D and long(res.size()) == D, "gw_line::distributed_phase_scan: size mismatch");
  if (D == 0) return;
  auto owner = [&](long j) { return dk[j] % np; };
  // problem sizes (Nr, n) of every deferred k, from the owners (exact all_reduce): batches by memory
  std::vector<double> dims(2 * D, 0.0);
  for (long j = 0; j < D; ++j)
    if (owner(j) == rank) {
      utils::check(prob[j] != nullptr and res[j] != nullptr, "gw_line::distributed_phase_scan: missing problem of k {}", dk[j]);
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
    std::vector<long> nphi(B);
    for (long j = 0; j < B; ++j) {
      ob[j]   = opts(dk[j0 + j]);
      nphi[j] = ob[j].nphi;
    }
    // coarse task ip of k -> rank (owner + (1 + ip) stride) mod np: spread over the ranks (memory bandwidth of all sockets /
    // nodes; the lent cores are the nearest idle ones)
    long nphi_max = 1;
    for (long j = 0; j < B; ++j) nphi_max = std::max(nphi_max, nphi[j]);
    const long stride = std::max(1L, np / (B * (nphi_max + 1)));
    auto task_rank    = [&](long j, long ip) { return (owner(j0 + j) + (1 + ip) * stride) % np; };
    bool own_any = false, own_ref = false, in_S = false;
    for (long j = 0; j < B; ++j) {
      if (owner(j0 + j) == rank) {
        own_any = true;
        own_ref = own_ref or prob[j0 + j]->need_ref;
        in_S    = true;
      }
      for (long ip = 0; ip < nphi[j]; ++ip) in_S = in_S or (task_rank(j, ip) == rank);
    }
    MPI_Comm sc = MPI_COMM_NULL;
    MPI_Comm_split(comm.get(), in_S ? 0 : MPI_UNDEFINED, int(rank), &sc);
    auto wait = [&](closure_cores_t &c) {
      if (T) T->start("closure_wait");
      c.wait();
      if (T) T->stop("closure_wait");
    };
    auto bthreads = [&](closure_cores_t const &c) { return c.blas_threads() > 0 ? c.blas_threads() : threads; };
    // phase A: the owners redo the SVD with the reference driver, with the cores of every other rank of their host
    {
      closure_cores_t cores(comm, own_ref ? 1L : 0L, max_threads);
      if (own_ref) {
        blas_threads_scope_t bscope(bthreads(cores));
        for (long j = 0; j < B; ++j)
          if (owner(j0 + j) == rank) ldlr::upfold_prepare_ref(*prob[j0 + j], ob[j], *res[j0 + j]);
      }
      wait(cores);
    }
    // phase B: problems owner -> scan ranks; coarse scan on the coarse ranks, eigensolve-free error built by the owners
    int ns = 0;
    std::vector<double> tb(B, 0.0), tpoly(B, 0.0), cv;
    std::vector<ldlr::upfold_problem_t> loc(B);
    std::vector<ldlr::upfold_problem_t const *> P(B);
    std::vector<ldlr::heldout_poly_t> hp(B);
    constexpr long NF = 9;   // err, umin, nflag, retry, fallback, residual, reason, t_eig, t_rr
    std::vector<long> coff(B + 1, 0);
    for (long j = 0; j < B; ++j) coff[j + 1] = coff[j] + nphi[j] * NF;
    double t_coarse = 0.0;
    long bt_coarse  = 0;
    {
      closure_cores_t cores(comm, in_S ? 1L : 0L, max_threads);
      if (in_S) {
        bt_coarse = bthreads(cores);
        blas_threads_scope_t bscope(bt_coarse);
        MPI_Comm_size(sc, &ns);
        std::vector<int> world(ns);
        const int me = int(rank);
        MPI_Allgather(&me, 1, MPI_INT, world.data(), 1, MPI_INT, sc);
        auto sc_rank = [&](long r) { return int(std::find(world.begin(), world.end(), int(r)) - world.begin()); };
        for (long j = 0; j < B; ++j) {
          const auto t0  = clk::now();
          const bool own = (owner(j0 + j) == rank);
          const int root = sc_rank(owner(j0 + j));
          double hdr[6]  = {0, 0, 0, 0, 0, 0};
          if (own) {
            auto const &q = *prob[j0 + j];
            hdr[0] = double(q.n); hdr[1] = double(q.Nr); hdr[2] = double(q.K); hdr[3] = double(q.n_free);
            hdr[4] = q.wp; hdr[5] = q.nheld;
          }
          MPI_Bcast(hdr, 6, MPI_DOUBLE, root, sc);
          const bool scan = std::llround(hdr[3]) > 0;   // the reference SVD may have closed the free block
          if (own) {
            P[j] = prob[j0 + j];
            if (scan)
              for (auto const *A : {&P[j]->A1, &P[j]->A0, &P[j]->R, &P[j]->Cheld}) detail::bcast_matrix(*A, root, sc);
          } else {
            auto &q  = loc[j];
            q.n      = std::llround(hdr[0]);
            q.Nr     = std::llround(hdr[1]);
            q.K      = std::llround(hdr[2]);
            q.n_free = std::llround(hdr[3]);
            q.wp     = hdr[4];
            q.nheld  = hdr[5];
            if (scan) {
              q.A1    = ldlr::cmatrix_F(q.Nr, q.Nr);
              q.A0    = ldlr::cmatrix_F(q.Nr, q.Nr);
              q.R     = ldlr::cmatrix_F(q.n, q.Nr);
              q.Cheld = ldlr::cmatrix_F(q.n, q.n);
              for (auto *A : {&q.A1, &q.A0, &q.R, &q.Cheld}) detail::bcast_matrix(*A, root, sc);
            }
            P[j] = &q;
          }
          tb[j] = secs(t0);
        }
        const auto tc = clk::now();
        cv.assign(coff[B], 0.0);
        for (long j = 0; j < B; ++j)
          if (owner(j0 + j) == rank and P[j]->n_free > 0 and ob[j].scan_err == "poly") {
            const auto t0 = clk::now();
            hp[j]         = ldlr::heldout_poly(*P[j]);
            tpoly[j]      = secs(t0);
          }
        for (long j = 0; j < B; ++j) {
          if (P[j]->n_free == 0) continue;
          for (long ip = 0; ip < nphi[j]; ++ip) {
            if (task_rank(j, ip) != rank) continue;
            ldlr::upfold_result_t rt;
            auto rz   = ldlr::realize(*P[j], ldlr::coarse_phase(ip, nphi[j]), ob[j], rt);
            double *x = cv.data() + coff[j] + ip * NF;
            x[0] = rz.err; x[1] = ldlr::unity_distance(rz.u); x[2] = double(rt.ueig_nflag); x[3] = double(rt.ueig_retry);
            x[4] = double(rt.ueig_fallback); x[5] = rt.ueig_res; x[6] = double(rt.ueig_reason); x[7] = rt.t_eig; x[8] = rt.t_rr;
          }
        }
        MPI_Allreduce(MPI_IN_PLACE, cv.data(), int(cv.size()), MPI_DOUBLE, MPI_SUM, sc);   // one contributor per slot
        t_coarse = secs(tc);
        for (long j = 0; j < B; ++j)   // the non-owners' copies are no longer needed
          if (owner(j0 + j) != rank) loc[j] = ldlr::upfold_problem_t{};
        MPI_Comm_free(&sc);
      }
      wait(cores);
    }
    // phase C: the owners decide, refine, realize the final phase and finish, with the cores of their host
    {
      closure_cores_t cores(comm, own_any ? 1L : 0L, max_threads);
      if (own_any) {
        const long bt = bthreads(cores);
        blas_threads_scope_t bscope(bt);
        for (long j = 0; j < B; ++j) {
          if (owner(j0 + j) != rank) continue;
          auto &pr = *prob[j0 + j];
          auto &up = *res[j0 + j];
          up.t_bcast      = tb[j];
          up.scan_ranks   = ns;
          up.scan_threads = bt_coarse * 1000 + bt;   // coarse-rank / owner BLAS threads (log: x / 1000, x % 1000)
          if (pr.n_free == 0) {
            ldlr::upfold_complete(pr, ob[j], up);
          } else {
            std::vector<double> errs(nphi[j]), umin(nphi[j]);
            for (long ip = 0; ip < nphi[j]; ++ip) {
              double const *x = cv.data() + coff[j] + ip * NF;
              errs[ip]        = x[0];
              umin[ip]        = x[1];
              up.ueig_nflag   = std::max(up.ueig_nflag, long(std::llround(x[2])));
              up.ueig_retry += std::llround(x[3]);
              up.ueig_fallback += std::llround(x[4]);
              up.ueig_res = std::max(up.ueig_res, x[5]);
              if (x[6] > 0.5) up.ueig_reason = int(std::llround(x[6]));
              up.t_eig += x[7];
              up.t_rr += x[8];
            }
            up.n_eig += nphi[j];
            up.n_realize += nphi[j];
            ldlr::golden_t g(ldlr::coarse_decide(errs, umin, ob[j], up), nphi[j]);
            up.t_coarse   = t_coarse;
            up.t_poly     = tpoly[j];
            const auto tr = clk::now();
            auto f        = [&](double p) {
              if (ob[j].scan_err != "poly") return ldlr::realize(pr, p, ob[j], up).err;
              ++up.n_mfree;
              return hp[j](p);
            };
            ldlr::golden_refine(g, f);
            up.t_refine = secs(tr);
            ldlr::upfold_final(pr, g.result(), ob[j], up);
            up.t_ueig = tb[j] + up.t_coarse + up.t_refine + up.t_final;
          }
          finish(j0 + j);
        }
      }
      wait(cores);
    }
    j0 = j1;
  }
}

} // namespace methods::gw_line

#endif
