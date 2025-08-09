#ifndef NUMERICS_DISTRIBUTED_ARRAY_OPS_HPP
#define NUMERICS_DISTRIBUTED_ARRAY_OPS_HPP

/*
 * Utilities for use of SLATE with math::nda::distributed_matrix
 */

#include <functional>

#include "utilities/check.hpp"
#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "numerics/distributed_array/detail/ops_aux.hpp"

namespace math::nda
{

/***************************************************************************/
/*                              blas/lapack tags                           */
/***************************************************************************/

//DistributedMatrix
template<DistributedArray MA>
auto normal(MA&& arg)
{ return math::detail::normal_tag<MA>(std::forward<MA>(arg)); }

template<DistributedArray MA>
auto transpose(MA&& arg)
{ return math::detail::transpose_tag<MA>(std::forward<MA>(arg)); }

template<DistributedArray MA>
auto dagger(MA&& arg)
{ return math::detail::conjugate_transpose_tag<MA>(std::forward<MA>(arg)); }

template<DistributedArray MA>
auto N(MA&& arg)
{ return math::detail::normal_tag<MA>(std::forward<MA>(arg)); }

template<DistributedArray MA>
auto T(MA&& arg)
{ return math::detail::transpose_tag<MA>(std::forward<MA>(arg)); }

template<DistributedArray MA>
auto H(MA&& arg)
{ return math::detail::conjugate_transpose_tag<MA>(std::forward<MA>(arg)); }

// put custom (non slate) distributed operations here, e.g. custom matrix multiplication

} // math::nda

#endif

