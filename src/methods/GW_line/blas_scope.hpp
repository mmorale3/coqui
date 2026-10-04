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

#ifndef COQUI_METHODS_GW_LINE_BLAS_SCOPE_HPP
#define COQUI_METHODS_GW_LINE_BLAS_SCOPE_HPP

/**
 * S7g: BLAS/LAPACK threads scoped to one section (the host closure of GPU runs, where a rank owns several idle cores).
 *
 * Threading rules of this code base (notes/line_gw/coqui_infra_for_gw_line.md section 3, utilities/blas_threads.hpp):
 * only the BLAS layer is threaded, never global OpenMP (OMP_NUM_THREADS stays 1, so SLATE's task layer stays serial).
 * MKL's own OpenMP workers make no MPI calls, so this is safe at MPI_THREAD_SINGLE. The setter is resolved at run time
 * (dlsym) as in utilities/blas_threads.hpp: MKL_Set_Num_Threads_Local (thread-local, returns the previous value; restored
 * on exit) or openblas_set_num_threads / openblas_get_num_threads. Without a known setter the scope is a no-op.
 */

#include <algorithm>
#include <cstdlib>
#include <string>
#include <thread>
#include <dlfcn.h>
#if defined(__linux__)
#include <sched.h>
#endif

namespace methods::gw_line {

/// CPU cores in this process's affinity mask (Linux; elsewhere std::thread::hardware_concurrency()).
inline long affinity_cores() {
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) return std::max(1L, long(CPU_COUNT(&set)));
#endif
  return std::max(1L, long(std::thread::hardware_concurrency()));
}

/**
 * Cores a rank may use for its host section: SLURM_CPUS_PER_TASK if set (capped by the affinity mask), otherwise the
 * affinity mask shared by the `node_ranks` ranks of this node (srun --cpu-bind=none gives every rank the whole step mask).
 */
inline long cores_per_rank(long node_ranks) {
  const long aff = affinity_cores();
  if (auto *e = std::getenv("SLURM_CPUS_PER_TASK")) {
    const long c = std::atol(e);
    if (c > 0) return std::min(c, aff);
  }
  return std::max(1L, aff / std::max(1L, node_ranks));
}

/// RAII: BLAS threads = n inside the scope (n <= 0: untouched), previous value restored on exit.
class blas_threads_scope_t {
  using set_local_t = int (*)(int);
  using set_t       = void (*)(int);
  using get_t       = int (*)();

 public:
  explicit blas_threads_scope_t(long n) {
    if (n <= 0) return;
    if (auto f = reinterpret_cast<set_local_t>(dlsym(RTLD_DEFAULT, "MKL_Set_Num_Threads_Local"))) {
      prev_   = f(int(n));
      mode_   = 1;
      active_ = true;
      return;
    }
    auto s = reinterpret_cast<set_t>(dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
    auto g = reinterpret_cast<get_t>(dlsym(RTLD_DEFAULT, "openblas_get_num_threads"));
    if (s and g) {
      prev_   = g();
      s(int(n));
      mode_   = 2;
      active_ = true;
    }
  }
  ~blas_threads_scope_t() {
    if (not active_) return;
    if (mode_ == 1) reinterpret_cast<set_local_t>(dlsym(RTLD_DEFAULT, "MKL_Set_Num_Threads_Local"))(prev_);
    else reinterpret_cast<set_t>(dlsym(RTLD_DEFAULT, "openblas_set_num_threads"))(prev_);
  }
  blas_threads_scope_t(blas_threads_scope_t const &)            = delete;
  blas_threads_scope_t &operator=(blas_threads_scope_t const &) = delete;
  bool active() const { return active_; }
  std::string backend() const { return mode_ == 1 ? "MKL_Set_Num_Threads_Local" : (mode_ == 2 ? "openblas" : "none"); }

 private:
  bool active_ = false;
  int mode_    = 0;
  int prev_    = 0;
};

} // namespace methods::gw_line

#endif
