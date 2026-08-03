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


#ifndef CUDA_KERNELS_AUX_HPP
#define CUDA_KERNELS_AUX_HPP

#include <type_traits>
#include <complex>
#include "nda/nda.hpp"
#include <thrust/complex.h>
#include <cuda/std/complex>
#include <cuda/std/mdspan>
#include <string>
#include <cuda_runtime.h>
#include "arch/arch.h"
#include "IO/AppAbort.hpp"

namespace kernels::device
{

  /***************    launch status checking   ***************/

  /**
   * Abort with a useful message if a CUB/thrust launch failed.
   *
   * WHY THIS EXISTS, and why `cuda_check(cudaGetLastError())` is not enough.
   * Every cub::DeviceFor / thrust::for_each call in this directory used to discard the
   * cudaError_t it returns. That is not the harmless omission it looks like: CUB *consumes* the
   * sticky error to build its return value, so by the time `arch::synchronize_if_set()` runs
   * `cudaGetLastError()` the error is already gone and the check passes. A failed launch therefore
   * left every output array holding whatever it was initialised to -- silently, with no error
   * anywhere -- and the run continued on that data.
   *
   * That is exactly how defect B6 presented: build/gpu90 was configured with
   * -DCMAKE_CUDA_ARCHITECTURES=90 but the top-level CMakeLists silently forced 80, so the binary
   * held sm_80 cubins with no PTX. On an H100 every launch failed with
   * cudaErrorNoKernelImageForDevice, `copy_select` returned its pre-zeroed buffer untouched, and
   * the ISDF point selection aborted 250 lines later with "0 cholesky vectors" -- pointing at
   * cuTENSOR, which was innocent. Hours went into the wrong suspect because the launch failure was
   * invisible.
   *
   * So: pass CUB's return value here. `what` should name the kernel for the message.
   */
  // Note: deliberately built with std::string rather than app_error/app_log. Those are fmt-based
  // variadic templates, and instantiating them inside a .cu leaves libcuda_kernels.a with
  // undefined references to fmt::v7 symbols it is not linked against (it fails at the final link
  // of coqui, not here). APP_ABORT takes a std::string and is already used in this header.
  inline void check_launch(cudaError_t status, char const* what)
  {
    if (status == cudaSuccess) return;
    std::string msg = std::string(" Device kernel launch failed in ") + what + ": "
                    + cudaGetErrorName(status) + " (" + cudaGetErrorString(status) + ")";
    if (status == cudaErrorNoKernelImageForDevice) {
      int dev = -1;
      cudaDeviceProp p{};
      if (cudaGetDevice(&dev) == cudaSuccess and cudaGetDeviceProperties(&p, dev) == cudaSuccess)
        msg += "\n This GPU is sm_" + std::to_string(p.major) + std::to_string(p.minor)
             + ", and the binary contains no matching kernel image.";
      msg += "\n Rebuild for this architecture, e.g."
             " cmake -DCMAKE_CUDA_ARCHITECTURES=90 ..."
             "\n Verify with: cuobjdump bin/coqui | grep 'arch ='";
    }
    APP_ABORT(msg);
  }

  /***************    is_complex   ***************/

  template<class T>
  struct is_complex : std::false_type {};

  template <typename T>
  struct is_complex<std::complex<T>> : std::true_type {}; 
  template <typename T>
  struct is_complex<thrust::complex<T>> : std::true_type {}; 
  template <typename T>
  struct is_complex<cuda::std::complex<T>> : std::true_type {};

  template <typename T>
  inline constexpr bool is_complex_v = is_complex<T>::value;

  /***************    complex_ptr_cast   ***************/

  template<typename T> auto complex_ptr_cast( T* x ) 
  { return x; } 
  template<typename T> auto complex_ptr_cast( std::complex<T>* x ) 
  { return reinterpret_cast<thrust::complex<T>*>(x); }
  template<typename T> auto complex_ptr_cast( std::complex<T> const* x ) 
  { return reinterpret_cast<thrust::complex<T> const*>(x); }

  /***************    complex_val_cast   ***************/

  template<typename T> auto complex_val_cast( T x )
  { return x; }
  template<typename T> auto complex_val_cast( std::complex<T> x )
  { return static_cast<thrust::complex<T>>(x); }
  //template<typename T> auto complex_val_cast( std::complex<T> const x )
  //{ return static_cast<thrust::complex<T> const>(x); }

  /***************    remove_thrust_complex   ***************/

  template <typename T>
  struct remove_thrust_complex {typedef T type;};
  template <typename T>
  struct remove_thrust_complex<thrust::complex<T> > {typedef T type;};

  template<typename T>
  using remove_thrust_complex_t = typename remove_thrust_complex<T>::type;

  /***************    add_thrust_complex   ***************/

  template <typename T>
  struct add_thrust_complex {typedef thrust::complex<T> type;};
  template <typename T>
  struct add_thrust_complex<thrust::complex<T> > {typedef thrust::complex<T> type;};

  template<typename T>
  using add_thrust_complex_t = typename add_thrust_complex<T>::type;

  /***************    cuda_std_ptr_cast   ***************/

  template<typename T> auto cuda_std_ptr_cast( T* x )
  { return x; }
  template<typename T> auto cuda_std_ptr_cast( std::complex<T>* x )
  { return reinterpret_cast<cuda::std::complex<T>*>(x); }
  template<typename T> auto cuda_std_ptr_cast( std::complex<T> const* x )
  { return reinterpret_cast<cuda::std::complex<T> const*>(x); }

  /***************    cuda_std_value_cast   ***************/

  template<typename T> auto cuda_std_value_cast( T x )
  { return x; }
  template<typename T> auto cuda_std_value_cast( std::complex<T> x )
  { return static_cast<cuda::std::complex<T>>(x); }

  /***************    remove_cuda_complex   ***************/

  template <typename T>
  struct remove_cuda_complex {typedef T type;};
  template <typename T>
  struct remove_cuda_complex<cuda::std::complex<T> > {typedef T type;};

  template<typename T>
  using remove_cuda_complex_t = typename remove_cuda_complex<T>::type;

  /***************    add_cuda_complex   ***************/

  template <typename T>
  struct add_cuda_complex {typedef cuda::std::complex<T> type;};
  template <typename T>
  struct add_cuda_complex<cuda::std::complex<T> > {typedef cuda::std::complex<T> type;};

  template<typename T>
  using add_cuda_complex_t = typename add_cuda_complex<T>::type;

  /***************    to_cuda_std_mdspan   ***************/
 
  template<typename Arr>
  auto to_cuda_std_mdspan(Arr& A)
  {
    constexpr auto RANK = ::nda::get_rank<Arr>;
    using value_t = typename std::pointer_traits<decltype(cuda_std_ptr_cast(A.data()))>::element_type;
    using cuda::std::mdspan;
    using dext = cuda::std::dextents<long,RANK>;
    using cuda::std::layout_stride;
    using cuda_array = cuda::std::array<long,RANK>;
    cuda_array extents, strides;
    std::copy_n(A.shape().begin(),RANK,extents.begin());
    std::copy_n(A.strides().begin(),RANK,strides.begin());
    // Create a layout_stride mapping
    layout_stride::mapping<dext> mapping(extents,strides);
    return mdspan<value_t,dext,layout_stride>(cuda_std_ptr_cast(A.data()),mapping);
  }

  /***************    to_cuda_std_array   ***************/
  
  // this should be limited to static_arrays, but I don't know how to get the static extents right now
  template<int rank, typename Arr>
  auto to_cuda_std_array(Arr const& A)
  {
    if(rank != A.size())
      APP_ABORT("Error in to_cuda_std_array: rank mismatch"); 
    cuda::std::array<nda::get_value_t<Arr>, rank> cuA;
    std::copy_n(A.data(),rank,cuA.data());
    return cuA;
  }

  template<int rank, typename Arr>
  auto to_std_array(Arr const& A)
  {
    if(rank != A.size())
      APP_ABORT("Error in to_cuda_std_array: rank mismatch");
    std::array<nda::get_value_t<Arr>, rank> stdA;
    std::copy_n(A.data(),rank,stdA.data());
    return stdA;
  }

} // namespace kernels::device

#endif
