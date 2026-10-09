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

#ifndef COQUI_METHODS_GW_LINE_CLOSURE_CORES_HPP
#define COQUI_METHODS_GW_LINE_CLOSURE_CORES_HPP

/**
 * perf 7.1 (f): the closure on more ranks than k points. The closure distributes whole k (owner = k mod np); with
 * np > N_k the ranks >= N_k own nothing and, bound to one core each, idle in the gather. closure_cores_t lends their cores
 * to the owners for the duration of the k loop:
 *   - every rank publishes (host, bound core); a helper h >= N_k on the same host as its owner h mod N_k gives its core;
 *   - the owner widens its own affinity mask to its core + the helpers' cores (sched_setaffinity of the calling thread;
 *     the GNU OpenMP threads MKL creates for the BLAS-thread scope inherit it) and runs the k loop with that many BLAS
 *     threads (blas_threads() > 0; S7g's MKL_Set_Num_Threads_Local scope), then restores the mask;
 *   - wait() replaces the busy-polling MPI wait of the helpers by a sleeping one (MPI_Ibarrier + MPI_Test + 200 us sleeps),
 *     so the lent cores are free; owners pass it after their k loop.
 * Results depend on the BLAS thread count at the closure's roundoff floor only (S7g: 1.1e-6 in Sigma between 1 and 2 MKL
 * threads, no decision flips). Linux only (no-op elsewhere), and only when every rank is bound to exactly one core (else
 * the ranks already float). Env COQUI_GWLINE_CLOSURE_BORROW = 0 disables it.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include <dlfcn.h>
#include <mpi.h>
#include <unistd.h>
#if defined(__linux__)
#include <sched.h>
#endif

#include "mpi3/communicator.hpp"

namespace methods::gw_line {

class closure_cores_t {
 public:
  /**
   * The k loop of the closure (collective over comm; nk: number of k, owner of k = k mod np). np > N_k: the ranks >= N_k
   * lend their cores to the owners of their host. Placement (env COQUI_GWLINE_CLOSURE_PLACE):
   *   "block" (perf 7.5a, default): the cores of the host are split per socket (physical_package_id) into contiguous blocks,
   *            the owners of the host are spread over the sockets in proportion to their cores, one block per owner (owner
   *            i of the host -> block i, blocks in core order); the owner's master thread moves to its block (its own core
   *            may belong to another owner's block). Every BLAS thread of an owner runs on its own core of one socket.
   *   "interleave" (perf 7.1 (f)): owner r keeps its core and takes the cores of the ranks r + N_k, r + 2 N_k, ... (on
   *            rome 1 node: 64 owners / helpers on socket 0, the rest on socket 1 -> chains across both sockets).
   * kcost (optional, size nk, identical on every rank; "block" only): expected cost of every k (e.g. Nr^3 of the previous
   * closure): the owners are assigned to the sockets greedily by cost (largest first, to the socket with the least cost per
   * core so far, at most its share of owners) and every socket's cores are split in proportion to its owners' costs (>= 1).
   */
  closure_cores_t(boost::mpi3::communicator &comm, long nk, std::vector<double> const *kcost = nullptr) : comm_(&comm) {
    char const *v = std::getenv("COQUI_GWLINE_CLOSURE_BORROW");
    const bool enable = (v == nullptr or *v == '\0' or std::strtol(v, nullptr, 10) != 0);
    const long np = comm.size(), rank = comm.rank();
    if (not enable or np <= nk or nk < 1) return;
#if defined(__linux__)
    std::vector<long> all;
    long mycpu = -1;
    if (not gather(comm, 0, all, mycpu)) return;
    active_ = true;
    const long host = all[NF * rank];
    std::vector<long> own, hr;   // owners / all ranks of my host, rank order
    for (long r = 0; r < np; ++r)
      if (all[NF * r] == host) {
        hr.push_back(r);
        if (r < nk) own.push_back(r);
      }
    if (rank >= nk) {
      helper_ = not own.empty();
      return;
    }
    std::vector<int> cores;
    if (place_mode() == "interleave") {
      cores.push_back(int(mycpu));
      for (long h = rank + nk; h < np; h += nk)
        if (all[NF * h] == host) cores.push_back(int(all[NF * h + 1]));
    } else {
      // sockets of the host, their cores in increasing order
      std::vector<long> sock;
      for (long r : hr)
        if (std::find(sock.begin(), sock.end(), all[NF * r + 2]) == sock.end()) sock.push_back(all[NF * r + 2]);
      std::sort(sock.begin(), sock.end());
      const long ns = long(sock.size()), no = long(own.size());
      std::vector<std::vector<long>> sc(ns);
      for (long r : hr) sc[std::find(sock.begin(), sock.end(), all[NF * r + 2]) - sock.begin()].push_back(all[NF * r + 1]);
      for (auto &x : sc) std::sort(x.begin(), x.end());
      // owners per socket in proportion to the cores (largest remainder), at least as many cores as owners per socket
      std::vector<long> nos(ns, 0);
      {
        long ncores = 0, given = 0;
        for (auto &x : sc) ncores += long(x.size());
        std::vector<std::pair<double, long>> rem;
        for (long s = 0; s < ns; ++s) {
          const double e = double(no) * double(sc[s].size()) / double(ncores);
          nos[s] = std::min(long(sc[s].size()), long(std::floor(e)));
          given += nos[s];
          rem.push_back({e - double(nos[s]), s});
        }
        std::stable_sort(rem.begin(), rem.end(), [](auto const &a, auto const &b) { return a.first > b.first; });
        for (long i = 0; given < no; i = (i + 1) % ns)
          if (nos[rem[i].second] < long(sc[rem[i].second].size())) {
            ++nos[rem[i].second];
            ++given;
          }
      }
      const long me = long(std::find(own.begin(), own.end(), rank) - own.begin());
      const bool weighted = kcost != nullptr and long(kcost->size()) == nk;
      if (not weighted) {
        long base = 0;
        for (long s = 0; s < ns; ++s) {
          if (me < base + nos[s]) {
            const long j = me - base, c = long(sc[s].size());
            for (long i = j * c / nos[s]; i < (j + 1) * c / nos[s]; ++i) cores.push_back(int(sc[s][i]));
            break;
          }
          base += nos[s];
        }
      } else {
        // owners -> sockets: largest cost first, to the socket with the least cost per core (count cap nos[s])
        std::vector<double> cst(no);
        for (long i = 0; i < no; ++i) cst[i] = std::max(1e-300, (*kcost)[own[i]]);
        std::vector<long> ord(no);
        for (long i = 0; i < no; ++i) ord[i] = i;
        std::stable_sort(ord.begin(), ord.end(), [&](long x, long y) { return cst[x] > cst[y]; });
        std::vector<long> sk(no, 0), cnt(ns, 0);
        std::vector<double> load(ns, 0.0);
        for (long i : ord) {
          long best = -1;
          for (long s2 = 0; s2 < ns; ++s2) {
            if (cnt[s2] >= nos[s2]) continue;
            const double x = (load[s2] + cst[i]) / double(sc[s2].size());
            if (best < 0 or x < (load[best] + cst[i]) / double(sc[best].size())) best = s2;
          }
          sk[i] = best;
          ++cnt[best];
          load[best] += cst[i];
        }
        // per socket: its owners in owner order, cores in proportion to the costs (largest remainder, >= 1 each)
        const long s0 = sk[me];
        std::vector<long> mem;
        for (long i = 0; i < no; ++i)
          if (sk[i] == s0) mem.push_back(i);
        const long c = long(sc[s0].size()), nm = long(mem.size());
        std::vector<long> nc(nm, 1);
        long given = nm;
        std::vector<std::pair<double, long>> rem;
        for (long j = 0; j < nm; ++j) {
          const double e = double(c - nm) * cst[mem[j]] / load[s0];
          const long f   = long(std::floor(e));
          nc[j] += f;
          given += f;
          rem.push_back({e - double(f), j});
        }
        std::stable_sort(rem.begin(), rem.end(), [](auto const &x, auto const &y) { return x.first > y.first; });
        for (long t = 0; given < c and t < nm; ++t, ++given) ++nc[rem[t].second];
        long off = 0;
        for (long j = 0; j < nm; ++j) {
          if (mem[j] == me)
            for (long i = off; i < off + nc[j]; ++i) cores.push_back(int(sc[s0][i]));
          off += nc[j];
        }
      }
    }
    widen(cores, mycpu);
#else
    (void)rank;
#endif
  }
  /**
   * perf 7.1c: the ranks with weight > 0 ("busy") borrow the cores of the other ranks of their host. Local lending: in
   * rounds, every busy rank (rank order) takes `weight` of the free idle cores nearest to its own core (by core index:
   * same CCX / socket first; spreading the threads over both sockets made the eigensolvers slower than 2 local threads),
   * at most max_threads cores per busy rank (its own included; <= 0: no cap). Same conditions as above (Linux, every
   * rank bound to one core, COQUI_GWLINE_CLOSURE_BORROW != 0). Collective over comm; every rank must call wait() afterwards.
   */
  closure_cores_t(boost::mpi3::communicator &comm, long weight, long max_threads) : comm_(&comm) {
    const bool busy = weight > 0;
    char const *v = std::getenv("COQUI_GWLINE_CLOSURE_BORROW");
    const bool enable = (v == nullptr or *v == '\0' or std::strtol(v, nullptr, 10) != 0);
    const long np = comm.size(), rank = comm.rank();
    if (not enable or np < 2) return;
#if defined(__linux__)
    std::vector<long> all;
    long mycpu = -1;
    if (not gather(comm, std::max(0L, weight), all, mycpu)) return;
    active_ = true;
    const long host = all[NF * rank];
    std::vector<long> hb, hi;   // busy / idle ranks of my host, rank order
    for (long r = 0; r < np; ++r)
      if (all[NF * r] == host) (all[NF * r + 3] ? hb : hi).push_back(r);
    if (not busy or hb.empty()) {
      helper_ = not hb.empty();
      return;
    }
    std::vector<char> taken(hi.size(), 0);
    std::vector<long> got(hb.size(), 1);   // cores per busy rank (own included)
    std::vector<int> cores{int(mycpu)};
    for (bool more = true; more;) {
      more = false;
      for (size_t ib = 0; ib < hb.size(); ++ib) {
        const long b = hb[ib], cb = all[NF * b + 1];
        for (long w = 0; w < all[NF * b + 3]; ++w) {
          if (max_threads > 0 and got[ib] >= max_threads) break;
          long best = -1;
          for (size_t i = 0; i < hi.size(); ++i) {
            if (taken[i]) continue;
            const long d = std::abs(all[NF * hi[i] + 1] - cb);
            if (best < 0 or d < std::abs(all[NF * hi[best] + 1] - cb)) best = long(i);
          }
          if (best < 0) break;
          taken[best] = 1;
          ++got[ib];
          more = true;
          if (b == rank) cores.push_back(int(all[NF * hi[best] + 1]));
        }
      }
    }
    widen(cores, mycpu);
#else
    (void)busy;
    (void)max_threads;
    (void)rank;
    (void)weight;
#endif
  }
  ~closure_cores_t() { restore(); }
  closure_cores_t(closure_cores_t const &)            = delete;
  closure_cores_t &operator=(closure_cores_t const &) = delete;

  /// BLAS threads for the owner's k loop (0: unchanged)
  long blas_threads() const { return widened_ ? ncpu_ : 0; }
  long cores() const { return ncpu_; }
  bool active() const { return active_; }
  /// perf 7.5a diagnostics of the widened owner: first / last core of its set, sockets spanned (bit mask of package ids),
  /// distinct CPUs the BLAS team ran on right after the pinning (sched_getcpu of every team thread)
  long core_first() const { return core_lo_; }
  long core_last() const { return core_hi_; }
  long socket_mask() const { return sock_mask_; }
  long cpus_seen() const { return seen_; }
  /// distinct CPUs of the team now (one parallel region of blas_threads() threads; 0 if not widened)
  long probe_cpus() const {
#if defined(__linux__)
    if (widened_) return team_cpus(ncpu_);
#endif
    return 0;
  }

  /// the owner's mask back to its own core (idempotent)
  void restore() {
#if defined(__linux__)
    if (widened_) {
      sched_setaffinity(0, sizeof(saved_), &saved_);
      if (pin_mode() != "none") repin_pool_mask(&saved_, ncpu_);
      if (mkl_dyn_ >= 0)
        if (auto f = reinterpret_cast<void (*)(int)>(dlsym(RTLD_DEFAULT, "MKL_Set_Dynamic"))) f(mkl_dyn_);
      mkl_dyn_ = -1;
    }
#endif
    widened_ = false;
  }

  /// collective: restore, then a barrier that sleeps instead of polling (helpers wait here during the owners' k loop)
  void wait() {
    restore();
    if (not active_) return;
    MPI_Request req;
    MPI_Ibarrier(comm_->get(), &req);
    int done = 0;
    while (true) {
      MPI_Test(&req, &done, MPI_STATUS_IGNORE);
      if (done) break;
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  }

  /// env COQUI_GWLINE_CLOSURE_PLACE: "block" (default) | "interleave"
  static std::string place_mode() {
    char const *v = std::getenv("COQUI_GWLINE_CLOSURE_PLACE");
    return (v == nullptr or *v == '\0') ? std::string("block") : std::string(v);
  }
  /**
   * env COQUI_GWLINE_CLOSURE_PIN: how the BLAS team (the GNU OpenMP pool MKL reuses) is placed on the owner's cores:
   *   "core" (perf 7.5a, default): thread t of the team bound to core t of the set (the master to the first);
   *   "mask" (perf 7.1c): every team thread gets the whole set (re-pinned from the previous lending's mask); after the
   *          restore() of the previous closure all of them sit on the owner's single core and stay there (the scheduler
   *          does not move a running thread whose CPU is still allowed): the regression of the merge (measured);
   *   "none" (perf 7.1 (f)): only the master's mask is widened; pool threads keep the mask they were created with.
   */
  static std::string pin_mode() {
    char const *v = std::getenv("COQUI_GWLINE_CLOSURE_PIN");
    return (v == nullptr or *v == '\0') ? std::string("core") : std::string(v);
  }

 private:
  static constexpr long NF = 4;   // host, core, socket, weight
#if defined(__linux__)
  /// all ranks: (host hash, bound core, socket of the core, weight); false (collectively) if some rank floats
  bool gather(boost::mpi3::communicator &comm, long weight, std::vector<long> &all, long &mycpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    mycpu = -1;
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0 and CPU_COUNT(&mask) == 1)
      for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &mask)) mycpu = c;
    saved_ = mask;
    char host[256] = {0};
    gethostname(host, sizeof(host) - 1);
    std::array<long, NF> me = {long(std::hash<std::string>{}(std::string(host)) & 0x7fffffffffffL), mycpu, socket_of(mycpu),
                               weight};
    const long np = comm.size();
    all.assign(NF * np, 0);
    comm.all_gather_n(me.data(), NF, all.data(), NF);
    for (long r = 0; r < np; ++r)
      if (all[NF * r + 1] < 0) return false;   // some rank is not bound to one core: nothing to lend (collectively)
    return true;
  }
  static long socket_of(long cpu) {
    if (cpu < 0) return 0;
    std::string f = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/physical_package_id";
    long s = 0;
    if (FILE *fp = std::fopen(f.c_str(), "r")) {
      if (std::fscanf(fp, "%ld", &s) != 1) s = 0;
      std::fclose(fp);
    }
    return s;
  }
  using gomp_parallel_t = void (*)(void (*)(void *), void *, unsigned, unsigned);
  static gomp_parallel_t gomp_parallel() { return reinterpret_cast<gomp_parallel_t>(dlsym(RTLD_DEFAULT, "GOMP_parallel")); }
  static int thread_num() {
    using f_t = int (*)();
    static f_t f = reinterpret_cast<f_t>(dlsym(RTLD_DEFAULT, "omp_get_thread_num"));
    return f ? f() : 0;
  }
  /**
   * perf 7.1c: the GNU OpenMP pool threads that MKL reuses keep the affinity mask they were created with (a previous
   * lending, possibly of cores now lent to another rank): run one parallel region of n threads (GOMP_parallel, resolved
   * at run time; libmkl_gnu_thread) in which every pool thread takes the mask m.
   */
  static void repin_one(void *m) { sched_setaffinity(0, sizeof(cpu_set_t), static_cast<cpu_set_t *>(m)); }
  static void repin_pool_mask(cpu_set_t *m, long n) {
    if (n > 1)
      if (auto g = gomp_parallel()) g(repin_one, m, unsigned(n), 0u);
  }
  /// perf 7.5a: thread t of an n-thread region -> core c[t] (one core each)
  struct pin_ctx_t {
    std::vector<int> const *c;
  };
  static void pin_one(void *x) {
    auto const &c = *static_cast<pin_ctx_t *>(x)->c;
    const int t   = thread_num();
    if (t < 0 or t >= int(c.size())) return;
    cpu_set_t m;
    CPU_ZERO(&m);
    CPU_SET(c[t], &m);
    sched_setaffinity(0, sizeof(m), &m);
  }
  struct seen_ctx_t {
    std::vector<int> *cpu;
  };
  static void see_one(void *x) {
    auto &v     = *static_cast<seen_ctx_t *>(x)->cpu;
    const int t = thread_num();
    if (t >= 0 and t < int(v.size())) v[t] = sched_getcpu();
  }
  static long team_cpus(long n) {
    std::vector<int> v(std::max(1L, n), -1);
    seen_ctx_t ctx{&v};
    if (auto g = gomp_parallel(); g and n > 1) g(see_one, &ctx, unsigned(n), 0u);
    else v[0] = sched_getcpu();
    std::sort(v.begin(), v.end());
    return long(std::unique(v.begin(), v.end()) - v.begin()) - (v[0] < 0 ? 1 : 0);
  }
  /// the owner's thread team on the cores c (c[0] = the master's core): pinning per pin_mode()
  void widen(std::vector<int> cores, long mycpu) {
    const long n = long(cores.size());
    if (n <= 1 and (n == 0 or cores[0] == mycpu)) return;
    const std::string pm = pin_mode();
    cpu_set_t wide;
    CPU_ZERO(&wide);
    for (int c : cores) CPU_SET(c, &wide);
    if (pm == "core") {
      cpu_set_t m0;
      CPU_ZERO(&m0);
      CPU_SET(cores[0], &m0);
      if (sched_setaffinity(0, sizeof(m0), &m0) != 0) return;
      if (n > 1) {
        if (auto g = gomp_parallel()) {
          pin_ctx_t ctx{&cores};
          g(pin_one, &ctx, unsigned(n), 0u);
        } else if (sched_setaffinity(0, sizeof(wide), &wide) != 0) {
          return;
        }
      }
    } else {
      if (sched_setaffinity(0, sizeof(wide), &wide) != 0) return;
      if (pm == "mask") repin_pool_mask(&wide, n);
    }
    widened_ = true;
    ncpu_    = n;
    core_lo_ = *std::min_element(cores.begin(), cores.end());
    core_hi_ = *std::max_element(cores.begin(), cores.end());
    sock_mask_ = 0;
    for (int c : cores) sock_mask_ |= (1L << std::min(62L, socket_of(c)));
    // MKL caps its thread count by the cores it found at initialization (the 1-core mask) unless dynamic adjustment
    // is off: switch it off for the k loop (restored in restore())
    using get_t = int (*)();
    using set_t = void (*)(int);
    if (auto gd = reinterpret_cast<get_t>(dlsym(RTLD_DEFAULT, "MKL_Get_Dynamic")))
      if (auto f = reinterpret_cast<set_t>(dlsym(RTLD_DEFAULT, "MKL_Set_Dynamic"))) {
        mkl_dyn_ = gd();
        f(0);
      }
    seen_ = team_cpus(n);
  }
#endif
  boost::mpi3::communicator *comm_ = nullptr;
  bool active_ = false, helper_ = false, widened_ = false;
  long ncpu_ = 1, core_lo_ = -1, core_hi_ = -1, sock_mask_ = 0, seen_ = 0;
  int mkl_dyn_ = -1;
#if defined(__linux__)
  cpu_set_t saved_{};
#endif
};

} // namespace methods::gw_line

#endif
