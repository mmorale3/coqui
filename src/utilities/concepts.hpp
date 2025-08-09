#ifndef UTILITIES_CONCEPTS_HPP
#define UTILITIES_CONCEPTS_HPP

#include <concepts>

namespace utils
{

/*
 * Some concepts
 */
template <typename A, typename T=double, typename I=long>
concept Communicator = requires(A& a, T* ptr, I ) {
  { a.rank() };
  { a.size() };
  { a.split() };
/*
  { a.split_shared() };
  { a.template broadcast_n<T,I>() };
  { a.template send_n<T,I>() };
  { a.template receive_n<T,I>() };
  { a.template gather_n<T,I,T>() };
*/
/// and many more...
};

}

#endif
