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

#include <array>
#include <chrono>
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
  /// collective over comm; nk: number of k of the closure (owner of k = k mod np)
  closure_cores_t(boost::mpi3::communicator &comm, long nk) : comm_(&comm) {
    char const *v = std::getenv("COQUI_GWLINE_CLOSURE_BORROW");
    const bool enable = (v == nullptr or *v == '\0' or std::strtol(v, nullptr, 10) != 0);
    const long np = comm.size(), rank = comm.rank();
    if (not enable or np <= nk or nk < 1) return;
#if defined(__linux__)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    long mycpu = -1;
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0 and CPU_COUNT(&mask) == 1)
      for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &mask)) mycpu = c;
    char host[256] = {0};
    gethostname(host, sizeof(host) - 1);
    std::array<long, 2> me = {long(std::hash<std::string>{}(std::string(host)) & 0x7fffffffffffL), mycpu};
    std::vector<long> all(2 * np);
    comm.all_gather_n(me.data(), 2, all.data(), 2);
    for (long r = 0; r < np; ++r)
      if (all[2 * r + 1] < 0) return;   // some rank is not bound to one core: nothing to lend (collectively)
    active_ = true;
    if (rank >= nk) {
      helper_ = (all[2 * (rank % nk)] == me[0]);
      return;
    }
    saved_   = mask;
    cpu_set_t wide = mask;
    long n   = 1;
    for (long h = rank + nk; h < np; h += nk)
      if (all[2 * h] == me[0]) {
        CPU_SET(int(all[2 * h + 1]), &wide);
        ++n;
      }
    if (n > 1 and sched_setaffinity(0, sizeof(wide), &wide) == 0) {
      widened_ = true;
      ncpu_    = n;
      // MKL caps its thread count by the cores it found at initialization (the 1-core mask) unless dynamic adjustment
      // is off: switch it off for the k loop (restored in restore())
      using get_t = int (*)();
      using set_t = void (*)(int);
      if (auto g = reinterpret_cast<get_t>(dlsym(RTLD_DEFAULT, "MKL_Get_Dynamic")))
        if (auto f = reinterpret_cast<set_t>(dlsym(RTLD_DEFAULT, "MKL_Set_Dynamic"))) {
          mkl_dyn_ = g();
          f(0);
        }
    }
#else
    (void)rank;
#endif
  }
  ~closure_cores_t() { restore(); }
  closure_cores_t(closure_cores_t const &)            = delete;
  closure_cores_t &operator=(closure_cores_t const &) = delete;

  /// BLAS threads for the owner's k loop (0: unchanged)
  long blas_threads() const { return widened_ ? ncpu_ : 0; }
  long cores() const { return ncpu_; }
  bool active() const { return active_; }

  /// the owner's mask back to its own core (idempotent)
  void restore() {
#if defined(__linux__)
    if (widened_) {
      sched_setaffinity(0, sizeof(saved_), &saved_);
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

 private:
  boost::mpi3::communicator *comm_ = nullptr;
  bool active_ = false, helper_ = false, widened_ = false;
  long ncpu_ = 1;
  int mkl_dyn_ = -1;
#if defined(__linux__)
  cpu_set_t saved_{};
#endif
};

} // namespace methods::gw_line

#endif
