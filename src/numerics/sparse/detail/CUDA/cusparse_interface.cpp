
#include "cuda_runtime.h"
#include "cusparse.h"

#include "numerics/sparse/detail/CUDA/cusparse_interface.hpp"

namespace math::sparse::device {

  cusparseHandle_t &get_cusparse_handle_ptr() {
    struct handle_t {
      handle_t() {
        CUSPARSE_CHECK(cusparseCreate, &h);
      }
      ~handle_t() {
        CUSPARSE_CHECK(cusparseDestroy, h);
      }

      cusparseHandle_t h = {};
    };
    static handle_t h = {};
    return h.h;
  }

} // math::sparse::device

