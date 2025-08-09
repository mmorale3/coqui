#ifndef NUMERICS_SPARSE_DETAIL_CONCEPTS_HPP
#define NUMERICS_SPARSE_DETAIL_CONCEPTS_HPP

#include <concepts>
#include "nda/nda.hpp"

namespace math::sparse
{

/*
 * Sparse vector concept
 */
template <typename A>
concept CSRVector = requires(A const& a) {
  { a.nnz() };
  { a.values() };
  { a.columns() };
  { a.size() }; 
  { a.shape() } -> ::nda::StdArrayOfLong;
  { std::decay_t<A>::sparse == true };
  { std::decay_t<A>::rank == 1 };
  { std::decay_t<A>::sorted == true };
};

/*
 * Sparse matrix concept
 */
template <typename A>
concept CSRMatrix = requires(A const& a) {
  { a.nnz() };
  { a.values() };
  { a.columns() };
  { a.row_begin() };
  { a.row_end() };
  { a.shape() } -> ::nda::StdArrayOfLong;
  { std::decay_t<A>::sparse == true };
  { std::decay_t<A>::rank == 2 };
  { std::decay_t<A>::sorted == true };
}; 

}

namespace nda::mem 
{

  template <math::sparse::CSRMatrix A>
  static constexpr AddressSpace get_addr_space<A> = to_nda_address_space(A::mem_type);

  template <math::sparse::CSRVector A>
  static constexpr AddressSpace get_addr_space<A> = to_nda_address_space(A::mem_type);

}

#endif
