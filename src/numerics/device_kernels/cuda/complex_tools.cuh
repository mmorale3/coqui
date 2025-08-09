//////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifndef COMPLEX_TOOLS_CUDA_KERNELS_HPP
#define COMPLEX_TOOLS_CUDA_KERNELS_HPP

#include <complex>
#include "nda/nda.hpp"
#include "numerics/device_kernels/cuda/nda_aux.hpp"

namespace kernels::device
{

namespace detail
{

template<typename Arr>
void zero_imag_impl(Arr & A);

}

template<nda::MemoryArray Arr> 
void zero_imag(Arr && A)
{
  static_assert(nda::mem::have_device_compatible_addr_space<Arr>, "Memory space mismatch");
  if( A.is_contiguous() ) {
    auto Ac = nda::reshape(A,std::array<long,1>{A.size()});
    auto A_b = to_basic_layout(Ac());
    detail::zero_imag_impl(A_b);
  } else {
    utils::check(nda::get_rank<Arr> <= 3, "Rank mismatch");
    auto A_b = to_basic_layout(A());
    detail::zero_imag_impl(A_b);
  }
}

} // kernels::device

#endif
