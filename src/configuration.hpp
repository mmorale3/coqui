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


#ifndef COQUI_TOP_CONFIGURATION_HPP
#define COQUI_TOP_CONFIGURATION_HPP

#include<complex>
#include<cstddef>
#include<cstdlib>
#include<memory>
#include "config.h"

#include "nda/nda.hpp"

template<typename T>
T* raw_pointer_cast(T* p) { return p; }

using RealType = double;
using SPRealType = float;
using ComplexType = std::complex<RealType>;
using SPComplexType = std::complex<RealType>;

enum MEMORY_SPACE { HOST_MEMORY, DEVICE_MEMORY, UNIFIED_MEMORY, DEFAULT_MEMORY };

#if defined(ENABLE_UNIFIED_MEMORY)
// compile time check for ENABLE_DEVICE done in cmake!
static constexpr  MEMORY_SPACE DEFAULT_MEMORY_SPACE = UNIFIED_MEMORY;
#elif defined(ENABLE_DEVICE)
static constexpr  MEMORY_SPACE DEFAULT_MEMORY_SPACE = DEVICE_MEMORY;
#else
static constexpr  MEMORY_SPACE DEFAULT_MEMORY_SPACE = HOST_MEMORY;
#endif

inline static constexpr nda::mem::AddressSpace to_nda_address_space(MEMORY_SPACE m)
{
  if(m == HOST_MEMORY)
    return nda::mem::Host; 
  else if(m == DEVICE_MEMORY)
    return nda::mem::Device;  
  else if(m == UNIFIED_MEMORY)
    return nda::mem::Unified;  
  else if(m == DEFAULT_MEMORY)
#if defined(ENABLE_UNIFIED_MEMORY)
    return nda::mem::Unified;
#elif defined(ENABLE_DEVICE)
    return nda::mem::Device;  
#else
    return nda::mem::Host; 
#endif
  return nda::mem::None; 
}

inline auto memory_space_to_string(MEMORY_SPACE m)
{
  if(m == HOST_MEMORY)
    return std::string("host");
  else if(m == DEVICE_MEMORY)
    return std::string("device");
  else if(m == UNIFIED_MEMORY)
    return std::string("unified");
  else
    return std::string("unknown"); 
  return std::string("host");
}

namespace memory 
{

template<nda::Array a_t>
constexpr MEMORY_SPACE get_memory_space()
{
  static_assert(nda::mem::on_host<a_t> or nda::mem::on_device<a_t> or nda::mem::on_unified<a_t>, "Unknown memory space");
  if constexpr (nda::mem::on_host<a_t>)
    return HOST_MEMORY;
  else if constexpr (nda::mem::on_device<a_t>)
    return DEVICE_MEMORY;
  else if constexpr (nda::mem::on_unified<a_t>)
    return UNIFIED_MEMORY;
  return HOST_MEMORY; 
}

template<typename T, int N, typename Layout = nda::C_layout>
using host_array = nda::array<T,N,Layout>;
template<typename T, int N, typename Layout = nda::C_stride_layout>
using host_array_view = nda::array_view<T,N,Layout>;

#if defined(ENABLE_DEVICE)
template<typename T, int N, typename Layout = nda::C_layout>
using device_array = nda::cuarray<T,N,Layout>;
template<typename T, int N, typename Layout = nda::C_stride_layout>
using device_array_view = nda::cuarray_view<T,N,Layout>;
#else
template<typename T, int N, typename Layout = nda::C_layout>
using device_array = nda::array<T,N,Layout>;
template<typename T, int N, typename Layout = nda::C_stride_layout>
using device_array_view = nda::array_view<T,N,Layout>;
#endif

#if defined(ENABLE_DEVICE)
template<typename T, int N, typename Layout = nda::C_layout>
using unified_array = nda::basic_array<T, N, Layout, 'A', nda::heap<nda::mem::Unified>>;
template<typename T, int N, typename Layout = nda::C_stride_layout>
using unified_array_view = nda::basic_array_view<T, N, Layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::Unified>>; 
#else
template<typename T, int N, typename Layout = nda::C_layout>
using unified_array = nda::array<T,N,Layout>;
template<typename T, int N, typename Layout = nda::C_stride_layout>
using unified_array_view = nda::array_view<T,N,Layout>;
#endif

#if defined(ENABLE_DEVICE)
template<typename T, int N, typename Layout = nda::C_layout>
using default_array = nda::cuarray<T,N,Layout>;
template<typename T, int N, typename Layout = nda::C_stride_layout>
using default_array_view = nda::cuarray_view<T,N,Layout>;
#else
template<typename T, int N, typename Layout = nda::C_layout>
using default_array = nda::array<T,N,Layout>;
template<typename T, int N, typename Layout = nda::C_stride_layout>
using default_array_view = nda::array_view<T,N,Layout>;
#endif

template<MEMORY_SPACE MEM, typename T, int N, typename Layout = nda::C_layout>
using array = std::conditional_t<MEM==HOST_MEMORY, host_array<T,N,Layout>,
              std::conditional_t<MEM==DEVICE_MEMORY, device_array<T,N,Layout>,
              std::conditional_t<MEM==UNIFIED_MEMORY, unified_array<T,N,Layout>,
						        default_array<T,N,Layout>>>>;

template<MEMORY_SPACE MEM, typename T, int N, typename Layout = nda::C_stride_layout>
using array_view = std::conditional_t<MEM==HOST_MEMORY, host_array_view<T,N,Layout>,
                   std::conditional_t<MEM==DEVICE_MEMORY, device_array_view<T,N,Layout>,
                   std::conditional_t<MEM==UNIFIED_MEMORY, unified_array_view<T,N,Layout>,
                                                           default_array_view<T,N,Layout>>>>;

template<MEMORY_SPACE MEM>
auto to_memory_space(auto &&A)
{
  if constexpr (MEM==HOST_MEMORY) {
    return nda::to_host(A);
  } else if constexpr (MEM==DEVICE_MEMORY) {
    return nda::to_device(A);
  } else if constexpr (MEM==UNIFIED_MEMORY) {
    return nda::to_unified(A);
  } else {
    return to_memory_space<DEFAULT_MEMORY_SPACE>(A);
  }
}

// =====================================================================
// Buffered arrays — backed by a static-fallback dynamic-bucket allocator
// for cheap reuse across hot-loop scratch allocations.
// =====================================================================
namespace detail
{

  /**
   * TEMPORARY local replacement for nda::mem::static_fallback.
   *
   * nda's version leaks the primary allocator's release accounting on the
   * fallback path: dynamic_bucket::allocate() adds the request to
   * _total_requested *before* discovering it cannot serve it (it returns
   * {nullptr,0} after the counter is already bumped), but nda's
   * static_fallback::deallocate() sends non-owned blocks straight to the
   * secondary allocator without ever telling the primary. _total_released
   * therefore misses every fallback block, and the pool's high-water mark
   * (maximum_memory(), i.e. _total_requested - _total_released) drifts
   * upward without bound as soon as any allocation spills past the pool.
   * That statistic is exactly what we need to size the pool, so it has to
   * be trustworthy.
   *
   * Fix: hand the block to the primary on the fallback path too.
   * dynamic_bucket::deallocate() recognizes a pointer outside its pool and
   * only accounts the release. The primary must see the block *before* the
   * secondary frees it, since it inspects b.ptr to classify it.
   *
   * Remove this class and revert static_allocator_t to
   * nda::mem::static_fallback once the upstream nda fix lands.
   */
  template<typename Primary>
  class corrected_static_fallback
  {
    inline static Primary alloc = {};
    using Secondary = nda::mem::mallocator<Primary::address_space>;

  public:
    static constexpr auto address_space = Primary::address_space;

    corrected_static_fallback()                                            = default;
    corrected_static_fallback(corrected_static_fallback const&)            = delete;
    corrected_static_fallback(corrected_static_fallback&&)                 = default;
    corrected_static_fallback& operator=(corrected_static_fallback const&) = delete;
    corrected_static_fallback& operator=(corrected_static_fallback&&)      = default;

    auto get_primary()       { return std::addressof(alloc); }
    auto get_primary() const { return std::addressof(alloc); }

    // The pool is a shared static, so it is reachable without an instance.
    // Used to size/release it (see utils::device_pool_guard) and to report
    // how much of it is in use.
    static Primary& pool() noexcept { return alloc; }
    static std::size_t bytes_live()   noexcept { return _live; }
    static std::size_t pool_hits()    noexcept { return _hits; }
    static std::size_t pool_misses()  noexcept { return _misses; }
    static void reset_counters()      noexcept { _hits = 0; _misses = 0; }

    nda::mem::blk_t allocate(std::size_t s) noexcept
    {
      nda::mem::blk_t b = alloc.allocate(s);
      if (b.ptr) { _live += b.s; ++_hits; return b; }
      ++_misses;
      return Secondary::allocate(s);
    }

    nda::mem::blk_t allocate_zero(std::size_t s) noexcept
    {
      nda::mem::blk_t b = this->allocate(s);
      if (b.ptr and b.s > 0) nda::mem::memset<address_space>(b.ptr, 0, b.s);
      return b;
    }

    void deallocate(nda::mem::blk_t b) noexcept
    {
      if (alloc.owns(b)) {
        _live -= std::min(_live, b.s);
        alloc.deallocate(b);
      } else {
        // account the release in the primary's counters, then free via the secondary
        alloc.deallocate(b);
        Secondary::deallocate(b);
      }
    }

    private:
    // Bytes currently served from the pool, and pool hit/miss counts.
    // Approximate to within the pool's internal alignment rounding, which
    // is not visible through dynamic_bucket's interface. Not atomic: the
    // allocator itself is not thread-safe (one rank per device today).
    inline static std::size_t _live   = 0;
    inline static std::size_t _hits   = 0;
    inline static std::size_t _misses = 0;
  };

  template<MEMORY_SPACE MEM>
  using static_allocator_t = corrected_static_fallback<nda::mem::dynamic_bucket<to_nda_address_space(MEM)>>;

  template<MEMORY_SPACE MEM>
  using buffered_handle_t = nda::heap_basic<static_allocator_t<MEM>>;

  // Largest allocation the pool is allowed to serve. Anything above this
  // goes straight to the raw allocator, so the pool never has to be sized
  // for the multi-GB per-iteration tensors (Pi/W/G/Sigma and the imaginary
  // -axis FT buffers are 18-23 GB each at Si 2x2x2/500b). Those want reuse
  // across iterations, not pooling; pooling them would mean reserving tens
  // of GB and starving phases that allocate through the raw allocator
  // (ERI/THC construction in particular). The recurring scratch this is
  // meant to capture sits at 0.27-2.6 GB.
  inline static constexpr std::size_t pool_max_block_size =
#if defined(COQUI_DEVICE_POOL_MAX_BLOCK)
      COQUI_DEVICE_POOL_MAX_BLOCK;
#else
      std::size_t(3) << 30; // 3 GiB
#endif

  template<MEMORY_SPACE MEM>
  using pooled_allocator_t = nda::mem::segregator<pool_max_block_size,
                                                  static_allocator_t<MEM>,
                                                  nda::mem::mallocator<to_nda_address_space(MEM)>>;

  template<MEMORY_SPACE MEM>
  using pooled_handle_t = nda::heap_basic<pooled_allocator_t<MEM>>;

}

  // Arrays whose allocations are served from the shared pool when the request
  // is at or below detail::pool_max_block_size, and by the raw allocator
  // otherwise. The pool is inert unless a utils::device_pool_guard is in
  // scope, so these behave exactly like the plain arrays outside a guarded
  // region. Structured like the buffered_array family above: the DEVICE and
  // UNIFIED handles must not be named at all on a CPU-only build, since
  // instantiating them trips nda's check_adr_sp_valid static_assert.
  template<typename T, int N, typename Layout = nda::C_layout>
  using host_pooled_array = nda::array<T,N,Layout,detail::pooled_handle_t<HOST_MEMORY>>;

#if defined(ENABLE_DEVICE)

  template<typename T, int N, typename Layout = nda::C_layout>
  using device_pooled_array = nda::array<T,N,Layout,detail::pooled_handle_t<DEVICE_MEMORY>>;

  template<typename T, int N, typename Layout = nda::C_layout>
  using unified_pooled_array = nda::array<T,N,Layout,detail::pooled_handle_t<UNIFIED_MEMORY>>;

#else

  template<typename T, int N, typename Layout = nda::C_layout>
  using device_pooled_array = host_pooled_array<T,N,Layout>;

  template<typename T, int N, typename Layout = nda::C_layout>
  using unified_pooled_array = host_pooled_array<T,N,Layout>;

#endif

  template<MEMORY_SPACE MEM, typename T, int N, typename Layout = nda::C_layout>
  using pooled_array = std::conditional_t<MEM==HOST_MEMORY,    host_pooled_array<T,N,Layout>,
                       std::conditional_t<MEM==DEVICE_MEMORY,  device_pooled_array<T,N,Layout>,
                       std::conditional_t<MEM==UNIFIED_MEMORY, unified_pooled_array<T,N,Layout>,
                                                               device_pooled_array<T,N,Layout>>>>;

  template<typename T, int N, typename Layout = nda::C_layout>
  using host_buffered_array = nda::array<T,N,Layout,detail::buffered_handle_t<HOST_MEMORY>>;

#if defined(ENABLE_DEVICE)

  template<typename T, int N, typename Layout = nda::C_layout>
  using device_buffered_array = nda::array<T,N,Layout,detail::buffered_handle_t<DEVICE_MEMORY>>;

  template<typename T, int N, typename Layout = nda::C_layout>
  using unified_buffered_array = nda::array<T,N,Layout,detail::buffered_handle_t<UNIFIED_MEMORY>>;

  template<typename T, int N, typename Layout = nda::C_layout>
  using default_buffered_array = device_buffered_array<T,N,Layout>;

#else

  template<typename T, int N, typename Layout = nda::C_layout>
  using device_buffered_array = host_buffered_array<T,N,Layout>;

  template<typename T, int N, typename Layout = nda::C_layout>
  using unified_buffered_array = host_buffered_array<T,N,Layout>;

  template<typename T, int N, typename Layout = nda::C_layout>
  using default_buffered_array = host_buffered_array<T,N,Layout>;

#endif

  template<MEMORY_SPACE MEM, typename T, int N, typename Layout = nda::C_layout>
  using buffered_array = std::conditional_t<MEM==HOST_MEMORY,    host_buffered_array<T,N,Layout>,
                         std::conditional_t<MEM==DEVICE_MEMORY,  device_buffered_array<T,N,Layout>,
                         std::conditional_t<MEM==UNIFIED_MEMORY, unified_buffered_array<T,N,Layout>,
                                                                 default_buffered_array<T,N,Layout>>>>;

// =====================================================================
// to_real_view: reinterpret a complex nda array as a (rank+1)-d real
// array view, with the new innermost dimension = 2 (real, imag).
// Trivial pass-through for real-valued inputs. Required by the cusparse
// interface for complex×real mixed CSR multiplies.
// =====================================================================
template<nda::MemoryArray A_t>
auto to_real_view(A_t && a) {
  using A = std::decay_t<A_t>;
  using value_type = typename A::value_type;
  using real_t = nda::remove_complex_t<value_type>;
  constexpr int rank = nda::get_rank<A>;
  constexpr MEMORY_SPACE MEM = get_memory_space<A>();

  if constexpr (nda::is_complex_v<value_type>) {
    static_assert(A::is_stride_order_C() or A::is_stride_order_Fortran(),
                  "to_real_view: stride order must be C or Fortran");
    if (a.indexmap().min_stride() != 1) std::abort();  // to_real_view requires unit min_stride
    if constexpr (A::is_stride_order_C()) {
      std::array<long,rank+1> shape;
      std::copy_n(a.shape().begin(), rank, shape.begin());
      shape[rank] = 2;
      std::array<long,rank+1> str;
      std::transform(a.strides().begin(), a.strides().end(), str.begin(),
                     [](auto const& x) { return 2*x; });
      str[rank] = 1;
      nda::idx_map<rank+1, 0, nda::C_stride_order<rank+1>, nda::layout_prop_e::none> idxm(shape, str);
      if constexpr (std::is_const_v<std::remove_pointer_t<decltype(a.data())>>)
        return memory::array_view<MEM, const real_t, rank+1>(idxm, reinterpret_cast<real_t const*>(a.data()));
      else
        return memory::array_view<MEM, real_t, rank+1>(idxm, reinterpret_cast<real_t*>(a.data()));
    } else {
      std::array<long,rank+1> shape;
      std::copy_n(a.shape().begin(), rank, shape.begin()+1);
      shape[0] = 2;
      std::array<long,rank+1> str;
      std::transform(a.strides().begin(), a.strides().end(), str.begin()+1,
                     [](auto const& x) { return 2*x; });
      str[0] = 1;
      nda::idx_map<rank+1, 0, nda::Fortran_stride_order<rank+1>, nda::layout_prop_e::none> idxm(shape, str);
      if constexpr (std::is_const_v<std::remove_pointer_t<decltype(a.data())>>)
        return memory::array_view<MEM, const real_t, rank+1, nda::F_stride_layout>(idxm, reinterpret_cast<real_t const*>(a.data()));
      else
        return memory::array_view<MEM, real_t, rank+1, nda::F_stride_layout>(idxm, reinterpret_cast<real_t*>(a.data()));
    }
  } else {
    return a();
  }
}

}

#endif
