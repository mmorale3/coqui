
#include "cuda_init.h"
#include "cuda_sync.h"

namespace cuda 
{
  static bool __sync__ = true;
  void synchronize() 
  {
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());    
  }
  bool get_device_synchronization() {return __sync__;}
  void set_device_synchronization(bool s) 
  {
    synchronize();
    __sync__ = s;
  }
  void synchronize_if_set()
  {
    if(__sync__) synchronize(); 
  }
}

