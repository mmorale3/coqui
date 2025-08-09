#include <c2py/c2py.hpp>
#include "IO/app_loggers.h"
#include "mpi_handler.hpp"
#include "mpi_handler.wrap.hxx"

#include "utilities/test_input_paths.hpp"

namespace coqui_py {

  /**
   * Set verbosity levels for CoQui logging output.
   * @param mpi_handler  - [INPUT] MPI handler to ensure logging is MPI-aware, i.e. only root prints
   * @param output_level - [INPUT] Level of output verbosity (default: 2)
   * @param debug_level  - [INPUT] Level of debug verbosity (default: 0)
   */
  void set_verbosity(MpiHandler &mpi_handler, int output_level = 2, int debug_level = 0) {
    setup_loggers(mpi_handler.root(), output_level, debug_level);
    mpi_handler.get_mpi()->comm.barrier();
  }

  std::tuple<std::string,std::string> utest_filename(std::string src) {
    return utils::utest_filename(src);
  }

} // coqui_py

#include "utils_module.wrap.cxx"
