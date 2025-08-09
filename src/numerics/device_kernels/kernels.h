#ifndef NUMERICS_DEVICE_KERNELS_H
#define NUMERICS_DEVICE_KERNELS_H


#if defined(ENABLE_CUDA)

#include "numerics/device_kernels/cuda/argmax_min.cuh"
#include "numerics/device_kernels/cuda/copy_select.cuh"
#include "numerics/device_kernels/cuda/complex_tools.cuh"
#include "numerics/device_kernels/cuda/symmetry_tools.cuh"
#include "numerics/device_kernels/cuda/kpoint_tools.cuh"
#include "numerics/device_kernels/cuda/potentials.cuh"

#endif

#endif
