/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */


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
#include "utilities/proc_grid_partition.hpp"

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


// the bounded (k, band) processor grid of the orbital readers
TEST_CASE("proc_grid_kb", "[utilities]")
{
  // the gcd grid is kept whenever it fits (bit-for-bit identical to find_proc_grid_max_rows)
  for (long size : {1L, 2L, 4L, 8L, 13L, 26L, 52L, 56L})
    for (long nk : {1L, 3L, 8L, 13L, 64L})
      for (long nb : {8L, 16L, 60L, 128L}) {
        const long pk_old = utils::find_proc_grid_max_rows(size, nk);
        auto kb = utils::find_proc_grid_kb(size, nk, nb);
        if (size / pk_old <= nb) { REQUIRE(kb[0] == pk_old); REQUIRE(kb[1] == size / pk_old); }
      }
  // coprime size and nkpts (64 ranks, 13 IBZ k, 60 bands: gcd = 1 -> 64 bands per rank with the gcd grid) fits
  { auto kb = utils::find_proc_grid_kb(64, 13, 60); REQUIRE(kb[0] == 8); REQUIRE(kb[1] == 8); }
  { auto kb = utils::find_proc_grid_kb(256, 13, 60); REQUIRE(kb[0] == 8); REQUIRE(kb[1] == 32); }
  { auto kb = utils::find_proc_grid_kb(56, 13, 60); REQUIRE(kb[0] == 1); REQUIRE(kb[1] == 56); }   // the gcd grid, unchanged
  // no grid at all: every layout puts more than nbnd ranks on the band axis
  { auto kb = utils::find_proc_grid_kb(512, 13, 60); REQUIRE(kb[0] == 0); REQUIRE(kb[1] == 0); }
  // every returned grid is a valid, bounded factorization
  for (long size = 1; size <= 1024; ++size)
    for (long nk : {1L, 6L, 13L, 27L})
      for (long nb : {4L, 16L, 60L}) {
        auto kb = utils::find_proc_grid_kb(size, nk, nb);
        if (kb[0] == 0) { REQUIRE(kb[1] == 0); continue; }
        REQUIRE(kb[0] * kb[1] == size);
        REQUIRE(kb[0] <= nk);
        REQUIRE(kb[1] <= nb);
      }
}
