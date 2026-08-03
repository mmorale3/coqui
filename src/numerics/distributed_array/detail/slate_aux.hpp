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


#ifndef NUMERICS_DISTRIBUTED_ARRAY_SLATE_AUX_HPP
#define NUMERICS_DISTRIBUTED_ARRAY_SLATE_AUX_HPP

/*
 * Auxiliary functions 
 */ 

#include <utility>
#include <type_traits>
#include "utilities/check.hpp"
#include "numerics/distributed_array/detail/concepts.hpp"
#include "numerics/distributed_array/detail/ops_aux.hpp"
#if defined(ENABLE_SLATE)
#include "slate/slate.hh"
#endif 
#if defined(ENABLE_CUDA)
#include "cuda_runtime.h"
#endif

namespace math::nda::slate_ops::detail
{

/***************************************************************************/
/*  				Utils	  				   */
/***************************************************************************/

#if defined(ENABLE_SLATE)
// MAM: Hardcoded to produce Fortran-ordered slate matrices.
template<bool transpose_layout, typename DMat, bool view>
auto make_slate(DMat& A_)
{
  // check that A has partitioning compatible with slate and has fortran ordering
  using Array_t = typename std::decay_t<DMat>::Array_t;
  using value_type = typename Array_t::value_type;
  using ij_tuple    = std::tuple<int64_t, int64_t>;

  // for now abort if transpose_layout is not consistent with DMat's layout, 
  // since Slate only allows fortran layout right now...
  static_assert( (transpose_layout and Array_t::layout_t::is_stride_order_C()) or
		 (not transpose_layout and Array_t::layout_t::is_stride_order_Fortran()),
		"Layout mismatch.");  
  // view onto a slate tile; only used by the owning (non-view) path, which is host-only
  using arr_t = ::nda::array_view<value_type, 2, ::nda::F_stride_layout>;
  using lay_t = typename arr_t::layout_t;

  int row_index = (transpose_layout?1:0);
  int col_index = (transpose_layout?0:1);

  auto&& A = math::detail::arg(A_);

  // MAM: can add a check for utils::check( A.is_slate_compatible(), "Slate incompatible matrix");
  int64_t p = A.grid()[row_index];
  int64_t q = A.grid()[col_index];
  int64_t m = A.global_shape()[row_index];
  int64_t n = A.global_shape()[col_index];  
  int64_t mb, nb;
  if constexpr (transpose_layout) {
    mb = A.block_size()[1]; // block size along rows
    nb = A.block_size()[0]; // block size along cols
  } else {
    mb = A.block_size()[0]; // block size along rows
    nb = A.block_size()[1]; // block size along cols
  }

  // tile assignment lambdas
  std::function<int64_t (int64_t i)> tileMb = [m, mb](int64_t i) { return (i + 1)*mb > m ? m%mb : mb; };
  std::function<int64_t (int64_t i)> tileNb = [n, nb](int64_t i) { return (i + 1)*nb > n ? n%nb : nb; };

  int64_t mt = (m/mb); //# of full blocks along rows
  int64_t nt = (n/nb); //# of full blocks along cols
  int64_t mx = mt/p;//number of blocks for last rank in row
  int64_t nx = nt/q;//number of blocks for last rank in col  
  int64_t mr = mt%p;//number of ranks with an extra block
  int64_t nr = nt%q;//number of ranks with an extra block 
  //std::function<int (ij_tuple ij)> tileRank = [p,q,mt,nt,mx,nx,mr,nr](ij_tuple ij) {
  std::function<int (ij_tuple ij)> tileRank = [p,q,mx,nx,mr,nr](ij_tuple ij) {
    int64_t i = std::get<0>( ij );
    int64_t j = std::get<1>( ij );    
    int64_t i_ = std::min(p-1, ( (i < mr*(mx+1)) ?  i/(mx+1): mr+(i-mr*(mx+1))/(mx) ));   
    int64_t j_ = std::min(q-1, ( (j < nr*(nx+1)) ?  j/(nx+1): nr+(j-nr*(nx+1))/(nx) ));   
    return int(j_*p+i_);  // column-major 
  };
  int dev = 0;
  [[maybe_unused]] int dev_ = 0;
#if defined(ENABLE_CUDA)
  cudaGetDevice(&dev_);
  if constexpr (not ::nda::mem::on_host<Array_t>) dev = dev_;
#endif
  std::function<int (ij_tuple ij)> tileDevice = [dev]([[maybe_unused]] ij_tuple ij) { return int{dev}; };

  // Slate needs to be told which address space a user-owned tile lives in: it keeps a
  // per-device MOSI state for every tile and stages copies (device workspace for compute,
  // host workspace for MPI and for the host panels of getrf/getri) off that. Tagging a
  // device pointer as a host tile makes slate skip those copies and hand the raw device
  // pointer to CPU BLAS/LAPACK and to MPI -- which "works" only with unified memory,
  // where the host can dereference it anyway. Only ::nda::mem::Device needs the tag;
  // Unified is host-dereferenceable, so leaving it on the host is both valid and what
  // the working unified path has always done.
  constexpr bool tiles_on_device = ::nda::mem::on_device<Array_t>;

  slate::Matrix<value_type> R(A.global_shape()[row_index], A.global_shape()[col_index],
			      tileMb, tileNb, tileRank, tileDevice,
			      A.communicator()->get());
#if defined(ENABLE_CUDA)
  cudaSetDevice(dev_);  // in case slate changes the active device (e.g. initialization of quues, etc)
#endif
  if constexpr (tiles_on_device)
    // tileDevice above hands slate ONE constant device for every tile of this rank, while slate
    // iterates device = 0 .. num_devices()-1 and picks tiles by `device == tileDevice(i,j)`. That
    // is only self-consistent when the process sees exactly one GPU. With several visible, each
    // rank's cudaSetDevice(local_rank % num_devices) gives a different `dev` while every rank
    // reports the same num_devices(), and slate's per-device workspace and the tiles disagree:
    // measured as a cudaErrorIllegalAddress at 12 ranks x 3 nodes with --gpus-per-node=4, and as
    // a clean run the moment --gpu-bind=single:1 collapses it to one visible device per rank.
    // `dev < num_devices()` was the old test and it passes in exactly that broken case, so it
    // never fired. Demand the assumption the code actually makes.
    utils::check(R.num_devices() == 1,
                 "make_slate: this rank sees {} GPUs (using device {}), but slate's tile->device "
                 "map here assumes exactly one GPU per rank. Bind one GPU per rank at launch "
                 "(srun --gpu-bind=single:1, or set CUDA_VISIBLE_DEVICES=$SLURM_LOCALID) and "
                 "rerun.", R.num_devices(), dev);

  if constexpr (not view) {
    // to_slate() copies into tiles slate owns. Only the host case is implemented: the
    // copy below goes through an nda host view of the tile. to_slate_view() is what
    // every caller uses and it handles device memory, so this was never finished.
    static_assert(::nda::mem::on_host<Array_t>,
                  "to_slate(): owning copy is only implemented for host memory, "
                  "use to_slate_view().");
    R.insertLocalTiles();
  }
  // copy data to R 
  auto Aloc = A.local();
  auto lld = Aloc.indexmap().strides()[col_index];
  for ( int64_t j=0; j < R.nt(); ++j )
    for ( int64_t i=0; i < R.mt(); ++i )
      if ( R.tileIsLocal(i,j) ) {
        auto x = i*mb - A.origin()[row_index]; 
        auto y = j*nb - A.origin()[col_index]; 
        utils::check(x>=0 and x <= A.local_shape()[row_index], 
		"Out of range: x:{}, shape:{}",x,A.local_shape()[row_index]);
        utils::check(y>=0 and y <= A.local_shape()[col_index], 
		"Out of range: y:{}, shape:{}, j:{}, nb:{}, org:{}",
		y,A.local_shape()[row_index],j,nb,A.origin()[col_index]);
      
        if constexpr (view) {
          // no dereference of Aloc when it is not on the host, hence the explicit offset
          auto* ptr = [&]() {
            if constexpr (transpose_layout) {
              if constexpr (::nda::mem::on_host<Array_t>) return std::addressof(Aloc(y,x));
              else                                        return Aloc.data() + (y*lld+x);
            } else {
              if constexpr (::nda::mem::on_host<Array_t>) return std::addressof(Aloc(x,y));
              else                                        return Aloc.data() + (x*lld+y);
            }
          }();
          if constexpr (tiles_on_device) {
            R.tileInsert(i,j,dev,ptr,lld);
          } else {
            R.tileInsert(i,j,ptr,lld);
          }
        } else {
          auto tile = R(i,j);
          auto Rloc = arr_t(lay_t{{tile.mb(),tile.nb()},{1,tile.stride()}},tile.data());
          if constexpr (transpose_layout) {
            Rloc = Aloc(::nda::range(y,y+tile.nb()),::nda::range(x,x+tile.mb()));
          } else {
            Rloc = Aloc(::nda::range(x,x+tile.mb()),::nda::range(y,y+tile.nb()));
          }
        }
      }

  if constexpr (math::detail::op_tag<std::decay_t<DMat>>::value == 'T') {
    return slate::transpose(R); 
  } else if constexpr (math::detail::op_tag<std::decay_t<DMat>>::value == 'C') {
    return slate::conj_transpose(R); 
  } else {
    return R; 
  }
}

template<bool transpose_layout, typename DMat>
auto to_slate_view(DMat&& A)
{
  return make_slate<transpose_layout, DMat,true>(A);
}

template<bool transpose_layout, typename DMat>
auto to_slate(DMat&& A)
{
  return make_slate<transpose_layout, DMat,false>(std::forward<DMat>(A));
}
#endif

} // math::nda::detail

#endif
