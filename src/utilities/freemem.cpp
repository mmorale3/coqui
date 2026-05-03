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


#if defined(__linux__)
  #include <sys/sysinfo.h>
  #include <sys/resource.h>
#elif defined(__APPLE__)
  #include <mach/mach.h>
  #include <TargetConditionals.h>
#endif
#if defined(ENABLE_CUDA)
  #include "cuda_runtime.h"
#endif
#include "IO/app_loggers.h"

namespace utils {

std::size_t freemem()
{
#if defined(__linux__)
  struct sysinfo si;
  sysinfo(&si);
  si.freeram += si.bufferram;
  return si.freeram >> 20;
#elif defined(__APPLE__)
  vm_statistics64_data_t vm_stats;
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  vm_size_t page_size;

  if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm_stats, &count) == KERN_SUCCESS) {
    host_page_size(mach_host_self(), &page_size);
      
    // On macOS, Free + Inactive + Speculative is the "available" pool
    uint64_t available_pages = (uint64_t)vm_stats.free_count + vm_stats.inactive_count + vm_stats.speculative_count;
    return (long)((available_pages * page_size) >> 20);
  } else {
    return 0;
  }
#else
  return 0;
#endif
}

std::size_t freemem_device()
{
#if defined(ENABLE_CUDA)
  std::size_t free_, tot_;
  cudaMemGetInfo(&free_, &tot_);
  return std::size_t( double(free_) / 1024.0 / 1024.0 );  
#elif defined(ENABLE_HIP)
  std::size_t free_, tot_;
  hipMemGetInfo(&free_, &tot_);
  return std::size_t( double(free_) / 1024.0 / 1024.0 );  
#else
  return 0;
#endif
}

void memory_report(int io_lvl, std::string message)
{
  if(message.size() > 0) app_log(io_lvl,"  memory report: " + message); 
  app_log(io_lvl, "  --> CPU Memory Available: {} \n ", freemem());
#if defined(ENABLE_CUDA)
  std::size_t free_, tot_;
  cudaMemGetInfo(&free_, &tot_);
  app_log(io_lvl, "  --> GPU Memory Available: {},  Total in MB: {} ",
                  free_ / 1024.0 / 1024.0, tot_ / 1024.0 / 1024.0 );
#elif defined(ENABLE_HIP)
  std::size_t free_, tot_;
  hipMemGetInfo(&free_, &tot_);
  app_log(io_lvl, "  --> GPU Memory Available: {},  Total in MB: {} ",
                  free_ / 1024.0 / 1024.0, tot_ / 1024.0 / 1024.0 );
#endif
  app_log_flush();
}

}

