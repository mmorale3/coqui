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


#ifndef COQUI_UTILITIES_DEVICE_POOL_H
#define COQUI_UTILITIES_DEVICE_POOL_H

#include <cstddef>
#include <string>
#include "configuration.hpp"

namespace utils {

/**
 * Scoped reservation of the shared device memory pool backing
 * memory::pooled_array.
 *
 * The pool is a process-wide static that starts at nda's default capacity
 * (a few KB, i.e. effectively inert: every request falls through to the raw
 * allocator). A guard sizes it on construction and releases it on
 * destruction, so memory is only withheld from the raw allocator inside the
 * guarded region. This matters because ERI/THC construction sizes its own
 * blocking from the free device memory it observes
 * (thc_aux.icc: nI_per_blk / nI_per_node) and allocates its large buffers
 * through the raw allocator; a pool held for the lifetime of the process
 * would shrink what those phases see. Wrap only the region that benefits
 * (the SCF loop), and phases outside it are unaffected.
 *
 * Two properties make this safe rather than merely convenient:
 *   - Allocations above detail::pool_max_block_size never touch the pool, so
 *     its capacity does not have to grow with the per-iteration tensors.
 *   - The allocator falls back to the raw allocator when the pool cannot
 *     serve a request, so an undersized pool costs performance, never
 *     correctness.
 *
 * Release requires that nothing allocated from the pool is still live
 * (nda's dynamic_bucket refuses to resize while segments are outstanding).
 * The destructor reports rather than throws if that is violated, since
 * throwing from a destructor during stack unwinding would terminate.
 */
class device_pool_guard {
public:
  /**
   * @param bytes  Pool capacity to reserve. 0 leaves the pool inert, which
   *               makes the guard a no-op and is the default behaviour until
   *               a size is chosen explicitly.
   * @param name   Label used in the log messages.
   */
  explicit device_pool_guard(std::size_t bytes, std::string name = "scf");
  ~device_pool_guard();

  device_pool_guard(device_pool_guard const&)            = delete;
  device_pool_guard& operator=(device_pool_guard const&) = delete;
  device_pool_guard(device_pool_guard&&)                 = delete;
  device_pool_guard& operator=(device_pool_guard&&)      = delete;

  /// True if this guard actually reserved a pool.
  bool active() const { return _active; }

private:
  bool _active = false;
  std::size_t _bytes = 0;
  std::string _name;
};

/// Capacity of the device pool in bytes (0 when inert).
std::size_t device_pool_capacity();

/// Bytes currently served from the device pool.
std::size_t device_pool_live();

/// Allocation requests served by the pool / that fell back to the raw allocator.
std::size_t device_pool_hits();
std::size_t device_pool_misses();

/**
 * Pool capacity read from the COQUI_DEVICE_POOL_GB environment variable, in
 * bytes; 0 (inert) when unset or unparsable. A deliberately low-ceremony
 * knob so the pool can be sized and measured before committing to an input
 * -file option and a sizing policy.
 */
std::size_t device_pool_size_from_env();

/**
 * Free device memory in MB, counting the unused part of the pool as
 * available. Consumers that size work from free memory should use this, so a
 * live pool does not make them under-estimate: the unused capacity is
 * reclaimable in principle, and allocations too large for the pool bypass it
 * entirely. Equals freemem_device() when the pool is inert.
 */
std::size_t freemem_device_effective();

/// Log pool capacity, live bytes, high-water mark and hit/miss counts.
void device_pool_report(int io_lvl = 2, std::string message = {});

}

#endif
