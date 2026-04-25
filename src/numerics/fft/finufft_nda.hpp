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

#ifndef NUMERICS_FFT_FINUFFT_NDA_HPP
#define NUMERICS_FFT_FINUFFT_NDA_HPP

/*
 * nda-level interface for the FINUFFT (non-uniform FFT) backend.
 *
 * Mirrors the structure of nda.hpp but for NUFFTs.  Only host (CPU)
 * arrays are supported; the nonuniform coordinate arrays and the
 * strength / mode arrays must all be contiguous nda arrays allocated on host.
 *
 * Transform conventions (matching finufft):
 *
 *   Type 1 — nonuniform → uniform  ("forward"):
 *     f[k] = sum_{j=0}^{M-1} c[j] exp(+i * iflag * k . x[j])
 *
 *   Type 2 — uniform → nonuniform  ("adjoint / inverse"):
 *     c[j] = sum_k f[k] exp(-i * iflag * k . x[j])
 *
 * Rank conventions:
 *   - The Fourier-mode array F has shape (N1) / (N1,N2) / (N1,N2,N3)
 *     for 1-D / 2-D / 3-D transforms.
 *   - For batched ("many") transforms F has an extra "slowest" dimension
 *     ntrans, giving shape (ntrans, N1[, N2[, N3]]) in C_layout
 *     and (N1[, N2[, N3]], ntrans) in F_layout.
 *   - The nonuniform coordinate arrays x, y, z are always rank-1
 *     with length M (number of nonuniform points).
 *   - The strength array C has shape (M) for a single transform,
 *     or (ntrans, M) for C_layout [ and (M, ntrans) for F_layout ] 
 *     for a batched transform.
 *
 * Notes:
 *   - finufft does NOT apply a 1/N normalisation; __normalize__ = false.
 *   - The coordinate arrays must remain unmodified between create_plan /
 *     setpts and the execute call.
 *   - For C_layout Fourier-mode arrays F, the order of the modes in create_plan
 *     and the order of nonuniform coordinate arrays must be transposed.
 *     For example, for a 3-d transform with dimensions {N1,N2,N3}, the correct
 *     calls would be:
 *     
 *       nda::array<T,3> F(N1,N2,N3);
 *       auto p = create_plan(std::array<int,3>{N3,N2,N1}, ...)
 *       setpts(p,z,y,x);
 *       fwdnufft(C,F,...);
 */

#include "configuration.hpp"
#include "numerics/fft/finufft_define.hpp"
#include "numerics/fft/finufft.h"
#include "utilities/check.hpp"
#include "nda/nda.hpp"

namespace math::nufft
{

namespace detail
{

template<::nda::MemoryArray CMat, ::nda::MemoryArray FMat>
void check_dimensions(nuplan_t &p, CMat &&C, FMat &&F)
{
  using C_t = std::decay_t<CMat>;
  using F_t = std::decay_t<FMat>;
  utils::check(F.is_contiguous() and C.is_contiguous(), "Only contiguous arrays allowed.");
  static_assert(F_t::layout_t::is_stride_order_Fortran(), "Layout mismatch");
  static_assert(::nda::mem::on_host<CMat>, "Only host arrays supported.");
  static_assert(::nda::mem::on_host<FMat>, "Only host arrays supported.");

  constexpr int rank = ::nda::get_rank<C_t>;
  static_assert(rank == 1 or rank == 2, "Rank mismatch.");
  
  if constexpr (rank==2) {
    constexpr int C_td = ( C_t::layout_t::is_stride_order_C() ? 0 : 1 );
    utils::check(int(C.shape()[C_td]) == p.ntrans,
                 "nufft: C.shape[0]={} != p.ntrans={}",
                 C.shape()[C_td], p.ntrans);
    utils::check(int64_t(C.shape()[1-C_td]) == p.npts,
                 "nufft: C.shape[1]={} != p.npts={}",
                 C.shape()[1-C_td], p.npts);
    utils::check(::nda::get_rank<F_t> == p.rank+1, "Rank mismatch");
    utils::check(F.shape()[(F_t::layout_t::is_stride_order_C() ? 0 : p.rank)] == p.ntrans,
               "nufft: Shape mismatch"); 
  } else {     
    utils::check(int64_t(C.shape()[0]) == p.npts,
                 "nufft: C.shape[0]={} != p.npts={}",
                 C.shape()[0], p.npts);
    utils::check(::nda::get_rank<F_t> == p.rank, "Rank mismatch");
  } 
  // F order: {x, y, z, n}
  for( int i=0; i<p.rank; ++i)
    utils::check(F.shape()[i] == p.nmodes[i], "nufft: Shape mismatch");
}

}

// finufft does not normalise — disable the normalisation step.
namespace impl::host { static constexpr bool __normalize__ = false; }

// =========================================================================
// Precision helper
//
// Selects double or float eps based on the value_type of the nda array,
// analogous to the way fftw.h dispatches on ComplexType vs RealType.
// =========================================================================
namespace impl
{
template<typename VT>
struct eps_type { using type = double; };
template<>
struct eps_type<std::complex<float>> { using type = float; };
template<>
struct eps_type<float> { using type = float; };

template<typename VT>
using eps_t = typename eps_type<VT>::type;

// Default precision: 1e-6 (double) or 1e-6f (float).
template<typename VT>
constexpr eps_t<VT> default_eps()
{
  if constexpr (std::is_same_v<eps_t<VT>, float>) return 1e-6f;
  else return 1e-6;
}
} // namespace impl

// =========================================================================
// create_plan
//
// Single-transform plan.
//
// nmodes: 1-D array with the number of modes in each dimension. 
//         Its shape implicitly defines the rank of the transformation.
// npts:   Number of non-uniform points on the grid.           
// ntrans: Number of transformations.
//
// Usage:
//   auto p = math::nufft::create_plan(nmodes, npts, eps, iflag);
// =========================================================================

template<typename Itype = int64_t, std::size_t rank = 1, typename value_type = double>
nuplan_t create_plan(std::array<Itype,rank> nmodes, int64_t npts,
                     int ntrans = 1,
                     value_type eps = impl::default_eps<value_type>(), 
                     int iflag = NUFFT_FORWARD)
{
//  int rank = nmodes.size(); 
  static_assert(rank >= 1 && rank <= 3, "Fourier-mode rank must be 1, 2, or 3.");

  // nmodes: the shape of F gives N1[,N2[,N3]]
  std::array<int64_t,rank> nm;
  for (int d = 0; d < rank; ++d) nm[d] = int64_t(nmodes[d]);

  return impl::host::create_plan(rank,nm.data(),npts,ntrans,eps,iflag);
}

// =========================================================================
// setpts — attach nonuniform coordinate arrays to a plan.
//
// For arrays with fortran ordering:
// 1-D:  setpts(p, x)
// 2-D:  setpts(p, x, y)
// 3-D:  setpts(p, x, y, z)
//
// For arrays with C ordering:
// 1-D:  setpts(p, x)
// 2-D:  setpts(p, y, x)
// 3-D:  setpts(p, z, y, x)
//
// The coordinate arrays must remain valid and unmodified until after the
// execute (fwdnufft / invnufft) call.
// =========================================================================

template<::nda::MemoryArrayOfRank<1> CoordMat>
void setpts(nuplan_t &p, CoordMat &&x)
{
  using X_t = std::decay_t<CoordMat>;
  static_assert(::nda::mem::on_host<CoordMat>, "Only host arrays supported.");
  utils::check(p.rank == 1, "setpts: plan rank={} but only x supplied.", p.rank);
  utils::check(int64_t(x.shape()[0]) == p.npts,
               "setpts: x.size={} != p.npts={}", x.shape()[0], p.npts);

  using val_t = typename X_t::value_type;
  impl::host::setpts(p, const_cast<val_t*>(x.data()), nullptr, nullptr);
}

template<::nda::MemoryArrayOfRank<1> CoordMat>
void setpts(nuplan_t &p, CoordMat &&x, CoordMat &&y)
{
  using X_t = std::decay_t<CoordMat>;
  static_assert(::nda::mem::on_host<CoordMat>, "Only host arrays supported.");
  utils::check(p.rank == 2, "setpts: plan rank={} but x,y supplied.", p.rank);
  utils::check(int64_t(x.shape()[0]) == p.npts &&
               int64_t(y.shape()[0]) == p.npts,
               "setpts: coordinate array sizes don't match p.npts={}.", p.npts);

  using val_t = typename X_t::value_type;
  impl::host::setpts(p,
                     const_cast<val_t*>(x.data()),
                     const_cast<val_t*>(y.data()),
                     nullptr);
}

template<::nda::MemoryArrayOfRank<1> CoordMat>
void setpts(nuplan_t &p, CoordMat &&x, CoordMat &&y, CoordMat &&z)
{
  using X_t = std::decay_t<CoordMat>;
  static_assert(::nda::mem::on_host<CoordMat>, "Only host arrays supported.");
  utils::check(p.rank == 3, "setpts: plan rank={} but x,y,z supplied.", p.rank);
  utils::check(int64_t(x.shape()[0]) == p.npts &&
               int64_t(y.shape()[0]) == p.npts &&
               int64_t(z.shape()[0]) == p.npts,
               "setpts: coordinate array sizes don't match p.npts={}.", p.npts);

  using val_t = typename X_t::value_type;
  impl::host::setpts(p,
                     const_cast<val_t*>(x.data()),
                     const_cast<val_t*>(y.data()),
                     const_cast<val_t*>(z.data()));
}

// =========================================================================
// destroy_plan
// =========================================================================
inline void destroy_plan(nuplan_t &p)
{
  impl::host::destroy_plan(p);
}

// =========================================================================
// fwdnufft  — type 1 : nonuniform → uniform
//
// Single transform:
//   fwdnufft(p, C, F)    C: rank-1 (M),         F: rank-D (N1[,N2[,N3]])
//
// Batched transform:
//   fwdnufft(p, C, F)    C: rank-2 (ntrans,M),  F: rank-D+1 (ntrans,N1,...) C_layout
//                                               F: rank-D+1 (N1,...,ntrans) F_layout
//
// The template selects single vs batched automatically from rank(C).
// =========================================================================

template<::nda::MemoryArray CMat, ::nda::MemoryArray FMat>
void fwdnufft(nuplan_t &p, CMat &&C, FMat &&F)
{
  using F_t = std::decay_t<FMat>;

  // Change layouts if needed
  if constexpr (::nda::get_rank<F_t> > 1 and F_t::layout_t::is_stride_order_C()) {
    fwdnufft(p,std::forward<CMat>(C),::nda::transpose(F));
    return;
  } else {
    detail::check_dimensions(p,C,F);
  }

  impl::host::fwdnufft(p, C.data(), F.data());
}

// =========================================================================
// invnufft  — type 2 : uniform → nonuniform
//
// Single transform:
//   invnufft(p, F, C)    F: rank-D,     C: rank-1
//
// Batched transform:
//   invnufft(p, F, C)    F: rank-D+1,   C: rank-2
// =========================================================================

template<::nda::MemoryArray FMat, ::nda::MemoryArray CMat>
void invnufft(nuplan_t &p, FMat &&F, CMat &&C)
{
  using F_t = std::decay_t<FMat>;

  // Change layouts if needed
  if constexpr (::nda::get_rank<F_t> > 1 and F_t::layout_t::is_stride_order_C()) {
    invnufft(p,::nda::transpose(F),std::forward<CMat>(C));
    return;
  } else {
    detail::check_dimensions(p,C,F);
  }

  impl::host::invnufft(p, F.data(), C.data());
}

// =========================================================================
// Plan-less convenience wrappers
//
// These mirror the plan-less fwdfft / invfft in nda.hpp.
// They create a plan, set the nonuniform points, execute, and destroy.
// =========================================================================

/// 1-D, single transform
template<::nda::MemoryArray CMat, ::nda::MemoryArray FMat,
         ::nda::MemoryArrayOfRank<1> CoordMat>
void fwdnufft(CMat &&C, FMat &&F, CoordMat &&x,
              impl::eps_t<typename std::decay_t<FMat>::value_type> eps
                = impl::default_eps<typename std::decay_t<FMat>::value_type>(),
              int iflag = NUFFT_FORWARD)
{
  using C_t = std::decay_t<CMat>;
  using F_t = std::decay_t<FMat>;
  constexpr int rank = ::nda::get_rank<C_t>;
  utils::check(::nda::get_rank<F_t> == rank, "Rank mismatch");
  
  // Change layouts if needed
  if constexpr (::nda::get_rank<F_t> > 1 and F_t::layout_t::is_stride_order_C()) {
    fwdnufft(std::forward<CMat>(C),::nda::transpose(F),std::forward<CoordMat>(x),eps,iflag);
    return;
  } else {
    if constexpr (rank==2) {
      constexpr int C_td = ( C_t::layout_t::is_stride_order_C() ? 0 : 1 );
      constexpr int F_td = ( F_t::layout_t::is_stride_order_C() ? 0 : 1 );
      std::array<int64_t,1> nm = { F.extent(1-F_td) }; 
      auto p = create_plan(nm, C.extent(1-C_td), C.extent(C_td), eps, iflag);
      setpts(p, x);
      fwdnufft(p, C, F);
      destroy_plan(p);
    } else {
      auto p = create_plan(F.shape(), C.extent(0), 1, eps, iflag);
      setpts(p, x);
      fwdnufft(p, C, F);
      destroy_plan(p);
    }
  }
}

/// 2-D, single transform
template<::nda::MemoryArray CMat, ::nda::MemoryArray FMat,
         ::nda::MemoryArray CoordMat>
void fwdnufft(CMat &&C, FMat &&F, CoordMat &&x, CoordMat &&y,
              impl::eps_t<typename std::decay_t<FMat>::value_type> eps
                = impl::default_eps<typename std::decay_t<FMat>::value_type>(),
              int iflag = NUFFT_FORWARD)
{
  using C_t = std::decay_t<CMat>;
  using F_t = std::decay_t<FMat>;
  constexpr int rank = ::nda::get_rank<C_t>; 
  utils::check(::nda::get_rank<F_t> == rank+1, "Rank mismatch");
  
  // Change layouts if needed
  if constexpr (F_t::layout_t::is_stride_order_C()) {
    fwdnufft(std::forward<CMat>(C),::nda::transpose(F),std::forward<CoordMat>(x),std::forward<CoordMat>(y),eps,iflag);
    return;
  } else {
    if constexpr (rank==2) { 
      constexpr int C_td = ( C_t::layout_t::is_stride_order_C() ? 0 : 1 );
      constexpr int F_td = ( F_t::layout_t::is_stride_order_C() ? 0 : 1 );
      std::array<int64_t,2> nm = { F.extent(1-F_td), F.extent(2-F_td) }; 
      auto p = create_plan(nm, C.extent(1-C_td), C.extent(C_td), eps, iflag);
      setpts(p, x, y);
      fwdnufft(p, C, F);
      destroy_plan(p);
    } else { 
      auto p = create_plan(F.shape(), C.extent(0), 1, eps, iflag);
      setpts(p, x, y);
      fwdnufft(p, C, F);
      destroy_plan(p);
    }
  }
}

/// 3-D, single transform
template<::nda::MemoryArray CMat, ::nda::MemoryArray FMat,
         ::nda::MemoryArray CoordMat>
void fwdnufft(CMat &&C, FMat &&F, CoordMat &&x, CoordMat &&y, CoordMat &&z,
              impl::eps_t<typename std::decay_t<FMat>::value_type> eps
                = impl::default_eps<typename std::decay_t<FMat>::value_type>(),
              int iflag = NUFFT_FORWARD)
{
  using C_t = std::decay_t<CMat>;
  using F_t = std::decay_t<FMat>;
  constexpr int rank = ::nda::get_rank<C_t>; 
  utils::check(::nda::get_rank<F_t> == rank+2, "Rank mismatch");
  
  // Change layouts if needed
  if constexpr (F_t::layout_t::is_stride_order_C()) {
    fwdnufft(std::forward<CMat>(C),::nda::transpose(F),std::forward<CoordMat>(x),std::forward<CoordMat>(y),std::forward<CoordMat>(z),eps,iflag);
    return;
  } else {
    if constexpr (rank==2) { 
      constexpr int C_td = ( C_t::layout_t::is_stride_order_C() ? 0 : 1 );
      constexpr int F_td = ( F_t::layout_t::is_stride_order_C() ? 0 : 1 );
      std::array<int64_t,3> nm = { F.extent(1-F_td), F.extent(2-F_td), F.extent(3-F_td) }; 
      auto p = create_plan(nm, C.extent(1-C_td), C.extent(C_td), eps, iflag);
      setpts(p, x, y, z);
      fwdnufft(p, C, F);
      destroy_plan(p);
    } else { 
      auto p = create_plan(F.shape(), C.extent(0), 1, eps, iflag);
      setpts(p, x, y, z);
      fwdnufft(p, C, F);
      destroy_plan(p);
    }
  }
}

/// 1-D, adjoint (type 2)
template<::nda::MemoryArray FMat, ::nda::MemoryArray CMat,
         ::nda::MemoryArrayOfRank<1> CoordMat>
void invnufft(FMat &&F, CMat &&C, CoordMat &&x,
              impl::eps_t<typename std::decay_t<FMat>::value_type> eps
                = impl::default_eps<typename std::decay_t<FMat>::value_type>(),
              int iflag = NUFFT_FORWARD)
{
  using C_t = std::decay_t<CMat>;
  using F_t = std::decay_t<FMat>;
  constexpr int rank = ::nda::get_rank<C_t>; 
  utils::check(::nda::get_rank<F_t> == rank, "Rank mismatch");
  
  // Change layouts if needed
  if constexpr (::nda::get_rank<F_t> > 1 and F_t::layout_t::is_stride_order_C()) {
    invnufft(::nda::transpose(F),std::forward<CMat>(C),std::forward<CoordMat>(x),eps,iflag);
    return;
  } else {
    if constexpr (rank==2) { 
      constexpr int C_td = ( C_t::layout_t::is_stride_order_C() ? 0 : 1 );
      constexpr int F_td = ( F_t::layout_t::is_stride_order_C() ? 0 : 1 );
      std::array<int64_t,1> nm = { F.extent(1-F_td) }; 
      auto p = create_plan(nm, C.extent(1-C_td), C.extent(C_td), eps, iflag);
      setpts(p, x);
      invnufft(p, F, C);
      destroy_plan(p);
    } else { 
      auto p = create_plan(F.shape(), C.extent(0), 1, eps, iflag);
      setpts(p, x);
      invnufft(p, F, C);
      destroy_plan(p);
    }
  }
}

/// 2-D, adjoint
template<::nda::MemoryArray FMat, ::nda::MemoryArray CMat,
         ::nda::MemoryArray CoordMat>
void invnufft(FMat &&F, CMat &&C, CoordMat &&x, CoordMat &&y,
              impl::eps_t<typename std::decay_t<FMat>::value_type> eps
                = impl::default_eps<typename std::decay_t<FMat>::value_type>(),
              int iflag = NUFFT_FORWARD)
{
  using C_t = std::decay_t<CMat>;
  using F_t = std::decay_t<FMat>;
  constexpr int rank = ::nda::get_rank<C_t>;
  utils::check(::nda::get_rank<F_t> == rank+1, "Rank mismatch");

  // Change layouts if needed
  if constexpr (F_t::layout_t::is_stride_order_C()) {
    invnufft(::nda::transpose(F),std::forward<CMat>(C),std::forward<CoordMat>(x),std::forward<CoordMat>(y),eps,iflag);
    return;
  } else {
    if constexpr (rank==2) {
      constexpr int C_td = ( C_t::layout_t::is_stride_order_C() ? 0 : 1 );
      constexpr int F_td = ( F_t::layout_t::is_stride_order_C() ? 0 : 1 );
      std::array<int64_t,2> nm = { F.extent(1-F_td), F.extent(2-F_td) };
      auto p = create_plan(nm, C.extent(1-C_td), C.extent(C_td), eps, iflag);
      setpts(p, x, y);
      invnufft(p, F, C);
      destroy_plan(p);
    } else {
      auto p = create_plan(F.shape(), C.extent(0), 1, eps, iflag);
      setpts(p, x, y);
      invnufft(p, F, C);
      destroy_plan(p);
    }
  }
}

/// 3-D, adjoint
template<::nda::MemoryArray FMat, ::nda::MemoryArray CMat,
         ::nda::MemoryArray CoordMat>
void invnufft(FMat &&F, CMat &&C, CoordMat &&x, CoordMat &&y, CoordMat &&z,
              impl::eps_t<typename std::decay_t<FMat>::value_type> eps
                = impl::default_eps<typename std::decay_t<FMat>::value_type>(),
              int iflag = NUFFT_FORWARD)
{
  using C_t = std::decay_t<CMat>;
  using F_t = std::decay_t<FMat>;
  constexpr int rank = ::nda::get_rank<C_t>;
  utils::check(::nda::get_rank<F_t> == rank+2, "Rank mismatch");
  
  // Change layouts if needed
  if constexpr (F_t::layout_t::is_stride_order_C()) {
    invnufft(::nda::transpose(F),std::forward<CMat>(C),std::forward<CoordMat>(x),std::forward<CoordMat>(y),std::forward<CoordMat>(z),eps,iflag);
    return;
  } else {
    if constexpr (rank==2) { 
      constexpr int C_td = ( C_t::layout_t::is_stride_order_C() ? 0 : 1 );
      constexpr int F_td = ( F_t::layout_t::is_stride_order_C() ? 0 : 1 );
      std::array<int64_t,3> nm = { F.extent(1-F_td), F.extent(2-F_td), F.extent(3-F_td) };
      auto p = create_plan(nm, C.extent(1-C_td), C.extent(C_td), eps, iflag);
      setpts(p, x, y, z);
      invnufft(p, F, C);
      destroy_plan(p);
    } else { 
      auto p = create_plan(F.shape(), C.extent(0), 1, eps, iflag);
      setpts(p, x, y, z);
      invnufft(p, F, C);
      destroy_plan(p);
    }
  }
}

// =========================================================================
// RAII wrapper class  math::nda::nufft
//
// Mirrors math::nda::fft from nda.hpp.
// Usage:
//   math::nda::nufft nft(F, C, x, eps, iflag);
//   nft.setpts(x);
//   nft.forward(C, F);   // type-1
//   nft.backward(F, C);  // type-2
// =========================================================================

} // namespace math::nufft

namespace math::nda
{

class nufft
{
public:

  /// Construct and plan (does NOT call setpts — call it separately).
  template<typename Itype = int64_t, std::size_t rank = 1, typename value_type = double>
  nufft(std::array<Itype,rank> const& nmodes, int64_t npts, int ntrans = 1, 
        value_type eps = math::nufft::impl::default_eps<value_type>(),
        int iflag = math::nufft::NUFFT_FORWARD)
    : plan(math::nufft::create_plan(nmodes,npts,ntrans,eps,iflag))
  {}

  ~nufft() { math::nufft::destroy_plan(plan); }

  nufft(nufft const &) = delete;
  nufft(nufft &&other) : plan(other.plan)
  {
    other.plan = math::nufft::nuplan_t{};
  }
  nufft &operator=(nufft const &) = delete;
  nufft &operator=(nufft &&other)
  {
    math::nufft::destroy_plan(plan);
    plan = other.plan;
    other.plan = math::nufft::nuplan_t{};
    return *this;
  }

  /// Attach nonuniform points (1-D, 2-D, or 3-D).
  void setpts(::nda::MemoryArrayOfRank<1> auto &&x)
  { math::nufft::setpts(plan, x); }

  void setpts(::nda::MemoryArrayOfRank<1> auto &&x, ::nda::MemoryArrayOfRank<1> auto &&y)
  { math::nufft::setpts(plan, x, y); }

  void setpts(::nda::MemoryArrayOfRank<1> auto &&x, ::nda::MemoryArrayOfRank<1> auto &&y,
              ::nda::MemoryArrayOfRank<1> auto &&z)
  { math::nufft::setpts(plan, x, y, z); }

  /// Type-1: nonuniform strengths C → uniform modes F.
  void forward(::nda::MemoryArray auto &&C, ::nda::MemoryArray auto &&F)
  { math::nufft::fwdnufft(plan, C, F); }

  /// Type-2 (adjoint): uniform modes F → nonuniform values C.
  void backward(::nda::MemoryArray auto &&F, ::nda::MemoryArray auto &&C)
  { math::nufft::invnufft(plan, F, C); }

private:
  math::nufft::nuplan_t plan;
};

} // namespace math::nda

#endif // NUMERICS_FFT_FINUFFT_NDA_HPP
