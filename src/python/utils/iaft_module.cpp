#include <c2py/c2py.hpp>

#include "numerics/imag_axes_ft/iaft_utils.hpp"


// ==========  Module declaration ==========
namespace c2py_module {
  // Name of the package if any. Default is ""
  auto package_name = ""; // the module will be Package.MyModule

  // The documentation string of the module. Default = ""
  auto documentation = "IAFT module for CoQui ";

  // -------- Automatic selection of function, classes, enums -----------
  auto match_names = "imag_axes_ft::(ir::IR|IAFT|read_iaft)";
  // FIXME CNY: How do we explicit exclude one of the constructor in IAFT?
  //       This allows us to hide ir::IR at the Python level
  //auto reject_names = ".*";

} // namespace c2py_module