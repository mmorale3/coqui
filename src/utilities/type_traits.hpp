#ifndef UTILS_TYPE_TRAITS_HPP
#define UTILS_TYPE_TRAITS_HPP

#include <complex>

namespace utils
{

  template <typename T>
  struct remove_complex {typedef T type;};
  template <typename T>
  struct remove_complex<std::complex<T> > {typedef T type;};
    
  template<typename T>
  using remove_complex_t = typename remove_complex<T>::type;

  template <typename T>
  struct add_complex {typedef std::complex<T> type;};
  template <typename T>
  struct add_complex<std::complex<T> > {typedef std::complex<T> type;};
    
  template<typename T>
  using add_complex_t = typename add_complex<T>::type;

} 

#endif
