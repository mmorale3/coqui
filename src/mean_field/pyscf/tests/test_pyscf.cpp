#undef NDEBUG

#include "catch2/catch.hpp"
#include "configuration.hpp"
#include "nda/nda.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"

#include "utilities/test_common.hpp"
#include "mean_field/pyscf/pyscf_readonly.hpp"

namespace bdft_tests {

  using utils::ARRAY_EQUAL;
  template<int N>
  using shape_t = std::array<long,N>;

  TEST_CASE("pyscf_h5", "[mean_field_pyscf]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto [outdir,prefix] = utils::utest_filename(mf::pyscf_source);
    mf::pyscf::pyscf_readonly mf(mpi_context, outdir,prefix);

    // unit cell info
    REQUIRE(mf.get_sys().nelec > 0);
    auto a = mf.lattice();
    auto b = mf.recv();
    REQUIRE( a.shape() == shape_t<2>{3,3});
    REQUIRE( b.shape() == shape_t<2>{3,3});
    REQUIRE(mf.get_sys().madelung > 0);

    // BZ info
    auto nkpts = mf.nkpts();
    REQUIRE(nkpts > 0);
    REQUIRE( mf.kpts().shape() == shape_t<2>{nkpts, 3} );
    REQUIRE( mf.get_sys().bz().Qpts.shape() == shape_t<2>{nkpts, 3} );
    REQUIRE( mf.get_sys().bz().qk_to_k2.shape() == shape_t<2>{nkpts, nkpts} );
    REQUIRE(mf.get_sys().bz().qminus.shape()[0] == nkpts);

    // basis info
    auto nbnd = mf.get_sys().nbnd;
    auto fft_mesh = mf.fft_grid_dim();
    REQUIRE(nbnd > 0);
    REQUIRE(mf.fft_grid_size() == fft_mesh(0)*fft_mesh(1)*fft_mesh(2) );

/*
    // mean-field info
    auto nspin = mf.get_sys().nspin;
    {
      auto S  = mf.get_sys().S;
      auto H0 = mf.get_sys().H0;
      auto F  = mf.get_sys().F;
      auto dm = mf.get_sys().dm;
      REQUIRE( S.shape() == shape_t<4>{nspin, nkpts, nbnd, nbnd} );
      REQUIRE( H0.shape() == shape_t<4>{nspin, nkpts, nbnd, nbnd} );
      REQUIRE( F.shape() == shape_t<4>{nspin, nkpts, nbnd, nbnd} );
      REQUIRE( dm.shape() == shape_t<4>{nspin, nkpts, nbnd, nbnd} );
    }
*/
  }
} // bdft_tests
