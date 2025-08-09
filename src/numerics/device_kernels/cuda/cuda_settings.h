#ifndef CUDA_KERNELS_SETTINGS_HPP
#define CUDA_KERNELS_SETTINGS_HPP

#define BOOST_NO_AUTO_PTR

static constexpr size_t MAXIMUM_GRID_DIM_YZ = 65535;
static constexpr size_t SM_SIZE_KB          = 64; // should probably poke the system
static constexpr size_t DEFAULT_BLOCK_SIZE  = 32;
static constexpr size_t DOT_BLOCK_SIZE      = 32;
static constexpr size_t REDUCE_BLOCK_SIZE   = 32;
static constexpr size_t MAX_THREADS_PER_DIM = 1024;

#endif
