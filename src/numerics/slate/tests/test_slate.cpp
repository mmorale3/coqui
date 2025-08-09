#undef NDEBUG

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/proc_grid_partition.hpp"

#include "utilities/test_common.hpp"

#include "slate/slate.hh"

namespace bdft_tests
{


TEST_CASE("slate_basic", "[math]")
{
  auto world = boost::mpi3::environment::get_world_instance();

  if(world.size()==4) {
    int n=1024, nb=8;
    slate::Matrix<double> A22_8(n,n,nb,2,2,MPI_COMM_WORLD);
  }
}

} // bdft_tests
