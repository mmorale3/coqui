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


#include <cstdlib>
#include <new>
#include <string>
#include "utilities/device_pool.h"
#include "utilities/freemem.h"
#include "IO/app_loggers.h"

namespace utils {

namespace {

  // The pool serving memory::pooled_array<DEVICE_MEMORY,...>. On a CPU-only
  // build the device address space cannot even be named (nda's
  // check_adr_sp_valid static_asserts), and pooled_array<DEVICE_MEMORY>
  // falls back to the host handle, so the guard manages the host pool there.
#if defined(ENABLE_DEVICE)
  using pool_alloc_t = memory::detail::static_allocator_t<DEVICE_MEMORY>;
#else
  using pool_alloc_t = memory::detail::static_allocator_t<HOST_MEMORY>;
#endif

  constexpr double to_mb = 1.0 / (1024.0 * 1024.0);
  constexpr double to_gb = 1.0 / (1024.0 * 1024.0 * 1024.0);

  // Capacity we last reserved, and the authority on whether the pool is
  // inert. dynamic_bucket::size() cannot be used for this: it never reports
  // zero, since resize(s) always keeps an alignment-sized backing block.
  std::size_t g_capacity = 0;

}

std::size_t device_pool_capacity() { return g_capacity; }
std::size_t device_pool_live()     { return pool_alloc_t::bytes_live(); }
std::size_t device_pool_hits()     { return pool_alloc_t::pool_hits(); }
std::size_t device_pool_misses()   { return pool_alloc_t::pool_misses(); }

std::size_t device_pool_size_from_env()
{
  const char* v = std::getenv("COQUI_DEVICE_POOL_GB");
  if (v == nullptr) return 0;
  char* end = nullptr;
  double gb = std::strtod(v, &end);
  if (end == v or gb <= 0.0) {
    app_warning("COQUI_DEVICE_POOL_GB=\"{}\" is not a positive number; device pool left disabled.", v);
    return 0;
  }
  return std::size_t(gb * 1024.0 * 1024.0 * 1024.0);
}

std::size_t freemem_device_effective()
{
  std::size_t raw = freemem_device();
  std::size_t cap = device_pool_capacity();
  std::size_t live = device_pool_live();
  std::size_t unused = (cap > live) ? (cap - live) : 0;
  return raw + std::size_t(double(unused) * to_mb);
}

void device_pool_report(int io_lvl, std::string message)
{
  std::size_t cap = device_pool_capacity();
  if (cap == 0) {
    app_log(io_lvl, "  device pool{}: inert (all allocations via the raw allocator)",
            message.size() > 0 ? " (" + message + ")" : "");
    return;
  }
  app_log(io_lvl, "  device pool{}: capacity {:.2f} GB, live {:.2f} GB, "
                  "high-water {:.2f} GB, served {} / fell back {}",
          message.size() > 0 ? " (" + message + ")" : "",
          double(cap) * to_gb, double(device_pool_live()) * to_gb,
          double(pool_alloc_t::pool().maximum_memory()) * to_gb,
          device_pool_hits(), device_pool_misses());
  app_log_flush();
}

device_pool_guard::device_pool_guard(std::size_t bytes, std::string name):
  _bytes(bytes), _name(std::move(name))
{
  if (bytes == 0) return;

  std::size_t live = device_pool_live();
  if (live > 0) {
    app_warning("device_pool_guard({}): {:.2f} GB already served from the pool; "
                "leaving the existing pool untouched.", _name, double(live) * to_gb);
    return;
  }

  try {
    pool_alloc_t::pool().resize(bytes);
  } catch (std::bad_alloc const&) {
    // dynamic_bucket::resize throws if segments are outstanding; it can also
    // fail if the device cannot back the request. Either way the pool stays
    // as it was and every allocation continues through the raw allocator.
    app_warning("device_pool_guard({}): could not reserve {:.2f} GB for the device pool; "
                "continuing without it.", _name, double(bytes) * to_gb);
    return;
  }
  pool_alloc_t::reset_counters();
  g_capacity = bytes;
  _active = true;
  app_log(2, "  device pool ({}): reserved {:.2f} GB; blocks larger than {:.2f} GB "
             "bypass it. Free device memory now {} MB.",
          _name, double(bytes) * to_gb,
          double(memory::detail::pool_max_block_size) * to_gb, freemem_device());
}

device_pool_guard::~device_pool_guard()
{
  if (not _active) return;

  device_pool_report(2, _name);

  std::size_t live = device_pool_live();
  if (live > 0) {
    // Never throw from a destructor. Leaking the reservation is the safe
    // outcome: the memory stays usable through the pool, it is simply not
    // returned to the raw allocator for later phases.
    app_warning("device_pool_guard({}): {:.2f} GB still allocated from the pool at scope exit; "
                "keeping the reservation. This withholds memory from the raw allocator "
                "for the rest of the run.", _name, double(live) * to_gb);
    return;
  }

  try {
    pool_alloc_t::pool().resize(0);
    g_capacity = 0;
    app_log(2, "  device pool ({}): released. Free device memory now {} MB.",
            _name, freemem_device());
  } catch (std::bad_alloc const&) {
    app_warning("device_pool_guard({}): release failed; keeping the reservation.", _name);
  }
}

}
