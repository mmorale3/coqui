#ifndef NUMERICS_SHARED_ARRAY_DETAIL_CONCEPTS_HPP
#define NUMERICS_SHARED_ARRAY_DETAIL_CONCEPTS_HPP

#include <concepts>

namespace math::shm
{

/*
 * Some concepts
 */
template<typename A>
concept SharedArray = requires(A& a, int src_node) {
  { a.all_reduce() };
  { a.broadcast_to_nodes(src_node) };
  { a.set_zero() };
  { a.win() };
  { a.communicator() };
  { a.node_comm() };
  { a.internode_comm() };
  { a.local() };
  { a.shape() } -> ::nda::StdArrayOfLong;
};

}

#endif
