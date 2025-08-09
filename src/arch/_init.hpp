////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifndef ARCH_INIT_HPP
#define ARCH_INIT_HPP

#include <cstdlib>
#include "config.h"

#include "IO/app_loggers.h"

#if defined(ENABLE_CUDA)
#include "CUDA/cuda_init.hpp"
#elif defined(ENABLE_ROCM)
//#include "ROCm/rocm_arch.h"
#endif

namespace arch 
{

void init(bool active_log, int output_level=2, int debug_level=2)
{
  // setup loggers, can always be changed later
  setup_loggers(active_log, output_level, debug_level);

#if defined(ENABLE_CUDA)
  cuda::init();
#endif

 // setup shared memory, memory buffers, etc, etc
}

}

#endif
