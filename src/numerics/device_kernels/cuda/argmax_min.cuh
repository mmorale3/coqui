//////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifndef ARGMAX_KERNELS_HPP
#define ARGMAX_KERNELS_HPP

#include <complex>
#include <tuple>
#include "numerics/device_kernels/cuda/cuda_aux.hpp"

namespace kernels::device
{

template<typename T>
std::tuple<long,T> argmax(T const*, long N);
template<typename T>
std::tuple<long,T> argmin(T const*, long N);

} // namespace kernels::device

#endif
