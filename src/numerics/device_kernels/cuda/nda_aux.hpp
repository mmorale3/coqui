#ifndef NDA_KERNELS_AUX_HPP
#define NDA_KERNELS_AUX_HPP

#include "configuration.hpp"
#include "nda/nda.hpp"

namespace kernels::device 
{ 

// limiting to C_stride for now
template<nda::MemoryArray Arr>
auto to_basic_layout(Arr && A)
requires( Arr::is_stride_order_C() and 
          Arr::layout_t::static_extents_encoded == 0 and
          nda::mem::have_device_compatible_addr_space<Arr> )
{
  constexpr int R = nda::get_rank<Arr>;
  using T = typename std::pointer_traits<decltype(A.data())>::element_type;
  using basic_layout_t = typename nda::basic_layout<0, nda::C_stride_order<R>, nda::layout_prop_e::none>;
  if constexpr (nda::mem::on_device<Arr>) {
    nda::basic_array_view<T,R,basic_layout_t,'A',nda::default_accessor, nda::borrowed<nda::mem::Device>> Ab = A();
    return Ab;
  } else if constexpr (nda::mem::on_unified<Arr>) {
    nda::basic_array_view<T,R,basic_layout_t,'A',nda::default_accessor, nda::borrowed<nda::mem::Unified>> Ab = A();
    return Ab;
  } else {
    static_assert(nda::mem::on_host<Arr>,"Only device/unified arrays allowed.");
    return std::forward<Arr>(A);
  }
}

} //kernels::device

#endif
