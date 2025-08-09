#ifndef ARCH_CUDA_CUDA_SYNC_H
#define ARCH_CUDA_CUDA_SYNC_H

namespace cuda 
{
  bool get_device_synchronization();
  void set_device_synchronization(bool);
  void synchronize_if_set();
  void synchronize();
}

#endif
