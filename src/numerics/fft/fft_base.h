#ifndef NUMERICS_FFT_FFT_BASE_H
#define NUMERICS_FFT_FFT_BASE_H

#include "configuration.hpp"
#include "numerics/fft/fft_define.hpp"

// CPU dispatch
#if defined(ENABLE_FFTW)
#include "numerics/fft/fftw.h"
namespace math::fft::impl::host{
  static const bool __normalize__ = true;
}
#else
#error FFTW currently required.
#endif  

// device dispatch
#if defined(ENABLE_DEVICE)

#if defined(ENABLE_CUDA)
#include "numerics/fft/cufft.h"
namespace math::fft::impl::dev{
  static const bool __normalize__ = true;
}
#elif defined(ENABLE_ROCM)
#error ROCM fft backend currently unavailable 
#endif  

#else   // no device backend, set dev to host

namespace math::fft::impl{
  // forces identical interfaces, if this is a problem then move #if defined(ENABLE_CUDA) to nda.hpp
  namespace dev = host;
}

#endif

#endif
