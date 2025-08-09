////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifndef ARCH_ARCH_H
#define ARCH_ARCH_H

#if defined(ENABLE_CUDA)
#include "CUDA/cuda_init.h"
#include "CUDA/cuda_sync.h"
#endif

#include "config.h"

namespace arch
{
  void init(bool active_log, int output_level, int debug_level);
  bool get_device_synchronization();
  void set_device_synchronization(bool);
  void synchronize_if_set();
  void synchronize();
}


#endif
