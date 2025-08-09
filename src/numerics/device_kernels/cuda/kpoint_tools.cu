//////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#include <complex>
#include <algorithm>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/type_traits.hpp"
#include "numerics/device_kernels/cuda/cuda_aux.hpp"
#include "numerics/device_kernels/cuda/cuda_settings.h"
#include "nda/nda.hpp"
#include "utilities/details/kpoint_impl.hpp"
#include <cuda/std/mdspan>
#include <cub/device/device_for.cuh>
#include <thrust/iterator/counting_iterator.h>

namespace kernels::device::detail
{

template<typename V1>
void rspace_phase_factor(nda::stack_array<double,3,3> const& lattv,
                         nda::stack_array<double,3> const& G,
                         nda::stack_array<long,3> const& mesh,
                         nda::range rng, V1& f)
{
  long N = rng.size(), N0 = rng.first();
  if(N==0) return;
  auto f_d = to_cuda_std_mdspan(f);
  auto F = utils::detail::rspace_phase_factor_mesh<decltype(f_d)>{N0,
     to_cuda_std_array<3>(mesh),to_cuda_std_array<3>(G),
     to_cuda_std_array<9>(lattv),f_d}; 
  cub::DeviceFor::Bulk(N,F);
  arch::synchronize_if_set();
}

template<typename V1, typename V2>
void rspace_phase_factor(nda::stack_array<double,3> const& G,
                         nda::stack_array<long,3> const& mesh,
                         V1 const& rp, V2& f)
{
  long N = f.extent(0); 
  if(N==0) return;
  auto f_d = to_cuda_std_mdspan(f);
  auto rp_d = to_cuda_std_mdspan(rp);
  auto F = utils::detail::rspace_phase_factor_crystal<decltype(rp_d),decltype(f_d)>{
     to_cuda_std_array<3>(mesh), to_cuda_std_array<3>(G),rp_d,f_d};
  cub::DeviceFor::Bulk(N,F);
  arch::synchronize_if_set();
}


using memory::device_array_view;
using memory::unified_array_view;
using std::complex;

template<int Rank>
using basic_layout_t = typename nda::basic_layout<0, nda::C_stride_order<Rank>, nda::layout_prop_e::none>;

#define _impl_(T,Arr)  \
template void rspace_phase_factor(nda::stack_array<double,3,3> const&,nda::stack_array<double,3> const&, \
  nda::stack_array<long,3> const&, nda::range, Arr<complex<T>,1,basic_layout_t<1>>&); \
template void rspace_phase_factor(nda::stack_array<double,3> const&, nda::stack_array<long,3> const&, \
  Arr<long const,1,basic_layout_t<1>> const&, Arr<complex<T>,1,basic_layout_t<1>>&); 

_impl_(double,device_array_view)
_impl_(double,unified_array_view)


} // kernels::device::detail

