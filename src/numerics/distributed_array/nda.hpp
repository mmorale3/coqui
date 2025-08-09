#ifndef NUMERICS_DISTRIBUTED_ARRAY_NDA_HPP
#define NUMERICS_DISTRIBUTED_ARRAY_NDA_HPP

#define SYNCHRONIZE_DISTRIBUTED_ARRAY 

#include "numerics/distributed_array/nda_matrix.hpp"
#include "numerics/distributed_array/nda_utils.hpp"
#include "numerics/distributed_array/ops.hpp"
#include "numerics/distributed_array/slate_ops.hpp"

namespace memory
{
template<::nda::MemoryArray local_Array_t,class comm>
using darray_t = math::nda::distributed_array<local_Array_t,comm>;

template<::nda::MemoryArray local_Array_t,class comm>
using darray_view_t = math::nda::distributed_array_view<local_Array_t,comm>;
}

#endif

