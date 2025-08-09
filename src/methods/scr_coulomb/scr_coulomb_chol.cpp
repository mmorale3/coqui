#include "methods/ERI/chol_reader_t.hpp"
#include "methods/GW/g0_div_utils.hpp"
#include "scr_coulomb_t.h"

namespace methods {
namespace solvers {

  void scr_coulomb_t::update_w(MBState &mb_state, Cholesky_ERI auto &chol, long h5_iter) {
    (void) mb_state; (void) chol; (void) h5_iter;
    app_log(2, "\nscr_coulomb_t::update_w: the new screen interaction interface is not implemented for Choleskky-ERI. "
               "CoQui will do nothing here, and instead the screened interactions will be computed using the gw solver.\n");
  }


  // template instantiations
  template void scr_coulomb_t::update_w(MBState&, chol_reader_t&, long);

}  // solvers
}  // methods
