#ifndef COQUI_AC_CONTEXT_H
#define COQUI_AC_CONTEXT_H

#include "numerics/imag_axes_ft/iaft_enum_e.hpp"

namespace analyt_cont {

struct ac_context_t {
  std::string ac_alg = "pade";
  imag_axes_ft::stats_e stats = imag_axes_ft::fermi;
  int Nfit = -1;
  double eta = 0.0001;
  // params for real w mesh
  double w_min = -10.0;
  double w_max = 10.0;
  long Nw = 5000;
};

} // analyt_cont

#endif //COQUI_AC_CONTEXT_H
