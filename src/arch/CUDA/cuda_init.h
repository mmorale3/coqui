////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifndef CUDA_ARCH_H
#define CUDA_ARCH_H

#include <string>
#include "cuda_runtime.h"

namespace cuda
{

void cuda_check(bool, ::std::string message = "");
void cuda_check(cudaError_t sucess, ::std::string message = "");
void init(); 

}

#endif
