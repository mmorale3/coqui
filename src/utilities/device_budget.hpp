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


#ifndef COQUI_UTILITIES_DEVICE_BUDGET_HPP
#define COQUI_UTILITIES_DEVICE_BUDGET_HPP

#include <cstdlib>
#include <functional>
#include <string>
#include "utilities/freemem.h"
#include "utilities/device_pool.h"

namespace utils {

/**
 * Device memory (bytes) kept out of every plan made with device_budget_bytes: library workspaces (cuBLAS, cuTENSOR,
 * SLATE), the CUDA context's growth and allocator fragmentation. Environment variable COQUI_DEVICE_RESERVE_GB,
 * default 2 GB.
 */
inline double device_reserve_bytes()
{
  double gb = 2.0;
  if (const char *v = std::getenv("COQUI_DEVICE_RESERVE_GB"); v != nullptr) {
    char *end = nullptr;
    double x = std::strtod(v, &end);
    if (end != v and x >= 0.0) gb = x;
  }
  return gb * 1024.0 * 1024.0 * 1024.0;
}

/**
 * Number of ranks of the node communicator that drive the same device as this rank (1 without a device build).
 * Collective over node_comm.
 */
template<typename comm_t>
long ranks_on_device(comm_t &node_comm)
{
  const std::string id = device_id();
  const int color = int(std::hash<std::string>{}(id) & 0x3fffffff);
  auto sub = node_comm.split(color, node_comm.rank());
  return long(sub.size());
}

/**
 * Device memory (bytes) this rank may plan with: the free memory of its device (the unused part of a reserved pool
 * counted as free) less device_reserve_bytes(), split evenly over the node's ranks that drive the same device.
 * Collective over node_comm. Negative when the device is already fuller than the reserve.
 */
template<typename comm_t>
double device_budget_bytes(comm_t &node_comm)
{
  const double free_b = double(freemem_device_effective()) * 1024.0 * 1024.0;
  return (free_b - device_reserve_bytes()) / double(ranks_on_device(node_comm));
}

}

#endif
