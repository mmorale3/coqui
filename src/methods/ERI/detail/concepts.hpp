#ifndef COQUI_CONCEPTS_HPP
#define COQUI_CONCEPTS_HPP

#include <concepts>

namespace methods {
  /* Concepts for different types of ERIs */
  template<typename A>
  concept THC_ERI = requires(const A &a) {
    { a.thc_X_type() };
  };

  template<typename A>
  concept Cholesky_ERI = requires(const A &a) {
    { a.chol_read_type() };
  };
} // methods

#endif //COQUI_CONCEPTS_HPP
