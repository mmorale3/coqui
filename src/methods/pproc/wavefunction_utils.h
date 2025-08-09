#ifndef AIMBES_METHODS_PPROC_WAVEFUNCTION_UTILS_HPP
#define AIMBES_METHODS_PPROC_WAVEFUNCTION_UTILS_HPP

#include "configuration.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "h5/h5.hpp"
#include "mean_field/MF.hpp"

namespace methods
{

  void add_wavefunction(h5::group & grp, mf::MF &mf, ptree const& pt);

}

#endif
