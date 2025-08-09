//////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifndef SYMMETRY_TOOLS_CUDA_KERNELS_HPP
#define SYMMETRY_TOOLS_CUDA_KERNELS_HPP

#include <complex>
#include "nda/nda.hpp"

namespace kernels::device
{

template<typename V1> 
void transform_k2g(bool trev, nda::stack_array<double, 3, 3> const& Rinv, 
                   nda::stack_array<double, 3> const& Gs, 
                   nda::stack_array<long, 3> const& mesh,
                   V1 &&k2g);


} // namespace kernels::device

#endif
