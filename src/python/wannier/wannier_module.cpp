#include <c2py/c2py.hpp>
#include "IO/app_loggers.h"

#include "wannier/wan90.h"

#include "python/mean_field/mf_module.hpp"
#include "python/mean_field/mf_module.wrap.hxx"

namespace coqui_py::wannier_interface {
 
  void wannier90_library_mode(const Mf &mf, const std::string &params) {
#if defined(ENABLE_WANNIER90)
    auto parser = InputParser(params);
    auto pt = parser.get_root();
    auto& coqui_mf = *mf.get_mf();
    wannier::wannier90_library_mode(coqui_mf,pt);
#else
    APP_ABORT("Error: wannier90.library_mode without wannier90 support. Recompile with ENABLE_WANNIER90=ON.");
#endif
  }

  void coqui2wannier90(const Mf &mf, const std::string &params) {
    auto parser = InputParser(params);
    auto pt = parser.get_root();
    auto& coqui_mf = *mf.get_mf();
    wannier::to_wannier90(coqui_mf,pt);
  }

  void wannier90_append_win(const Mf &mf, const std::string &params) {
    auto parser = InputParser(params);
    auto pt = parser.get_root();
    auto& coqui_mf = *mf.get_mf();
    wannier::append_wannier90_win(coqui_mf,pt);
  }

} // coqui_py::wannier_interface


#include "wannier_module.wrap.cxx"
