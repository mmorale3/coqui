#undef NDEBUG

#include <complex>

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "utilities/test_common.hpp"
#include "utilities/Timer.hpp"
#include "utilities/harmonics.h"
#if defined(ENABLE_SPHERICART)
#include "sphericart.hpp" 
#endif

namespace bdft_tests
{

using utils::VALUE_EQUAL;

template<typename T>
void test_harmonics([[maybe_unused]] T m, [[maybe_unused]] T eps)
{

  utils::harmonics<T> ylm;
#if defined(ENABLE_SPHERICART)
  int Lmax = 3;
  int Lmax2 = (Lmax+1)*(Lmax+1);
  auto ylm_sp = sphericart::SphericalHarmonics<T>(Lmax);

  nda::array<T,1> y1(Lmax2), y2(Lmax2);

  for(int i=0; i<10; i++) {
    // random point
    auto r = nda::rand<T>(3);
    r() -= T(0.5);
 
    ylm.spherical_harmonics(Lmax,r,y1);
    ylm_sp.compute_sample(r.data(),3,y2.data(),Lmax2);
    
    utils::ARRAY_EQUAL(y1,y2,m,eps);
  }

#else
  app_log(0,"Compiled without sphericart support, skipping test of harmonincs.");
#endif

}


TEST_CASE("harmonics", "[utilities]")
{
  test_harmonics<double>(1e-8,1e-8);
  test_harmonics<float>(float(1e-6),float(1e-6));
}

}
