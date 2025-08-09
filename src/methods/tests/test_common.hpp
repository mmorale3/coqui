#ifndef METHODS_TESTS_TEST_COMMON_HPP
#define METHODS_TESTS_TEST_COMMON_HPP

#include "configuration.hpp"

#include "nda/nda.hpp"
#include "numerics/shared_array/nda.hpp"

namespace methods {
  using utils::VALUE_EQUAL;
  using utils::ARRAY_EQUAL;

  namespace mpi3 = boost::mpi3;

  template<nda::Array Array_base_t>
  using sArray_t = math::shm::shared_array<Array_base_t>;
  template<int N>
  using shape_t = std::array<long,N>;
} // methods

#endif // METHODS_TESTS_TEST_COMMON_HPP
