#undef NDEBUG

#include <complex>

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "utilities/test_common.hpp"
#include "utilities/Timer.hpp"
#include "utilities/interpolation_utils.hpp"

namespace bdft_tests
{

using utils::VALUE_EQUAL;

TEST_CASE("interpolation", "[utilities]")
{
  // not really checking anything right now, just making sure they run without errors
  {
    std::vector<nda::array<double,2>> kp(4);
    kp[0] = nda::array<double,2>{ {0.0,0.0,0.0}, {0.5,0.0,0.0} };
    kp[1] = nda::array<double,2>{ {0.5,0.0,0.0}, {0.5,0.5,0.0} };
    kp[2] = nda::array<double,2>{ {0.5,0.5,0.0}, {0.5,0.5,0.5} };
    kp[3] = nda::array<double,2>{ {0.5,0.5,0.5}, {0.0,0.0,0.0} };
    nda::array<double,2> recv = {{0.0,6.0,8.0},{4.0,0.0,8.0},{4.0,6.0,0.0}};
    std::vector<std::string> id = {"K","G","G","M","M","R","R","G"};
    auto [kpath,idx] = utils::generate_kpath(recv,kp,id,10);
  }

  {
    nda::array<double,2> recv = {{0.0,6.0,8.0},{4.0,0.0,8.0},{4.0,6.0,0.0}};
    nda::array<long,1> mesh = {4,4,4};
    auto ws = utils::WS_rgrid(recv,mesh);
  }
}


}

