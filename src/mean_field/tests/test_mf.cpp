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

#include "catch2/catch.hpp"
#include "configuration.hpp"
#include <complex>
#include "stdio.h"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"

#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/test_common.hpp"
#include "utilities/symmetry.hpp"

#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/fft/nda.hpp"
#include "numerics/distributed_array/nda.hpp"

#include "grids/r_grids.hpp"
#include "grids/g_grids.hpp"
#include "mean_field/distributed_orbital_readers.hpp"
#include "mean_field/MF.hpp"
#include "mean_field/default_MF.hpp"
#include "mean_field/mf_utils.hpp"
#include "mean_field/properties.hpp"

#include "mean_field/symmetry/bz_symmetry.hpp"
//#include "hamiltonian/one_body_hamiltonian.hpp"

namespace bdft_tests
{

using utils::VALUE_EQUAL;
using utils::ARRAY_EQUAL;
template<int N>
using shape_t = std::array<long,N>;

TEST_CASE("mf_xml", "[mean_field]") {
  auto& mpi = utils::make_unit_test_mpi_context(); 
  auto [outdir,prefix] = utils::utest_filename(mf::qe_source);
  {
    mf::qe::qe_readonly qe(mpi,outdir, prefix);
    mf::MF mfobj(qe);
  }
  auto mfobj = mf::make_MF(mpi, mf::qe_source, outdir, prefix);

  REQUIRE(mfobj.nelec() > 0);
  { auto b = mfobj.nspin(); REQUIRE(b>=0); REQUIRE(b<2);} 
  { [[maybe_unused]] auto b = mfobj.noncolin(); }
  { [[maybe_unused]] auto b = mfobj.spinorbit(); }

  REQUIRE(mfobj.fft_grid_size() > 0); 
  REQUIRE(mfobj.fft_grid_dim()(0) > 0);

  REQUIRE(mfobj.lattv().shape()[0] == 3);
  REQUIRE(mfobj.recv().shape()[0] == 3);
  REQUIRE(mfobj.ecutrho() > 0.0);

  REQUIRE(mfobj.nkpts() > 0);
  REQUIRE(mfobj.kpts().shape()[0] == mfobj.nkpts());
  REQUIRE(mfobj.k_weight().shape()[0] == mfobj.nkpts());
  REQUIRE(mfobj.Qpts().shape()[0] == mfobj.nkpts());
  REQUIRE(mfobj.qk_to_k2().shape()[0] == mfobj.nkpts());
  REQUIRE(mfobj.qminus().shape()[0] == mfobj.nkpts());

  REQUIRE(mfobj.nbnd() > 0);
  REQUIRE(mfobj.occ().shape()[0] > 0);
  REQUIRE(mfobj.eigval().shape()[0] > 0);

  {
    auto wfc_g = mfobj.wfc_truncated_grid();
    long ngm = wfc_g->size();
    nda::array<ComplexType, 1> Orb(ngm);
    mfobj.get_orbital('w',0,0,0,Orb(nda::range::all));
    ComplexType ov = 0.0;
    for( auto& v: Orb )
      ov += v * std::conj(v);
    VALUE_EQUAL(ov, 1.0);
  }

  {
    auto wfc_g = mfobj.wfc_truncated_grid();
    long ngm = wfc_g->size();
    nda::array<ComplexType, 3> Og(1,2,mfobj.fft_grid_size());
    nda::array<ComplexType, 3> Ow(1,2,ngm);
    mfobj.get_orbital_set('g',0,{0,1},{0,2},Og);
    mfobj.get_orbital_set('w',0,{0,1},{0,2},Ow);

    nda::array<ComplexType, 2> OwC(2,ngm);
    OwC = nda::dagger(Ow(0,nda::ellipsis{}));
    nda::array<ComplexType, 2> ov(2,2);
    nda::blas::gemm(1.0, Ow(0,nda::ellipsis{}),
                         OwC, 0.0, ov);

    for(int i=0; i<2; ++i)
      for(int j=0; j<2; ++j)
        if(i==j)
          VALUE_EQUAL(ov(i,j),1.0);
        else
          VALUE_EQUAL(ov(i,j),0.0);

    nda::array<ComplexType, 2> OgC(2,mfobj.fft_grid_size());
    OgC = nda::dagger(Og(0,nda::ellipsis{}));
    nda::blas::gemm(1.0, Og(0,nda::ellipsis{}),
                         OgC, 0.0, ov);

    for(int i=0; i<2; ++i)
      for(int j=0; j<2; ++j)
        if(i==j)
          VALUE_EQUAL(ov(i,j),1.0);
        else
          VALUE_EQUAL(ov(i,j),0.0);
  }
  mfobj.close();

}

TEST_CASE("qe_orbs", "[mean_field]") {
  auto& mpi = utils::make_unit_test_mpi_context(); 
  auto [outdir,prefix] = utils::utest_filename(mf::qe_source);
  auto mfobj = mf::make_MF(mpi, mf::qe_source, outdir, prefix);

  using local_Array_t = memory::array<HOST_MEMORY, ComplexType, 4>;
  auto dPsi_k = mf::read_distributed_orbital_set<local_Array_t>(mfobj, mpi->comm, 'w');
  auto dPsi_g = mf::read_distributed_orbital_set<local_Array_t>(mfobj, mpi->comm, 'g');
  auto dPsi_r = mf::read_distributed_orbital_set<local_Array_t>(mfobj, mpi->comm, 'r');

  //h5::file file("Orb_qe.h5", 'w');
  //h5::group grp(file);
  //nda::h5_write(grp, "Orb_k", dPsi_k.local(), false);
  //nda::h5_write(grp, "Orb_g", dPsi_g.local(), false);
  //nda::h5_write(grp, "Orb_r", dPsi_r.local(), false);
}

TEST_CASE("qe_rho", "[mean_field]") {
  auto& mpi = utils::make_unit_test_mpi_context(); 
  auto [outdir,prefix] = utils::utest_filename(mf::qe_source);
  auto mfobj = mf::make_MF(mpi, mf::qe_source, outdir, prefix);

  grids::distributed_r_grid g(mfobj.fft_grid_dim(), mfobj.lattv(), mpi->comm,{mpi->comm.size(),1,1});
  // charge density
  auto rho = mf::distributed_charge_density(mfobj,g);
}

TEST_CASE("pyscf_h5", "[mean_field]") {
  auto& mpi = utils::make_unit_test_mpi_context(); 
  auto [outdir,prefix] = utils::utest_filename(mf::pyscf_source);
  {
    mf::pyscf::pyscf_readonly pyscf(mpi, outdir, prefix);
    mf::MF mfobj(pyscf);
  }
  auto mfobj = mf::make_MF(mpi, mf::pyscf_source, outdir, prefix);
  // unit cell info
  REQUIRE(mfobj.nelec() > 0);
  auto a = mfobj.lattv();
  auto b = mfobj.recv();
  REQUIRE( a.shape() == shape_t<2>{3,3} );
  REQUIRE( b.shape() == shape_t<2>{3,3} );
  REQUIRE(mfobj.volume() > 0);
  REQUIRE(mfobj.madelung() > 0);

  // BZ info
  auto nkpts = mfobj.nkpts();
  REQUIRE(nkpts > 0);
  REQUIRE( mfobj.kpts().shape() == shape_t<2>{nkpts, 3} );
  REQUIRE( mfobj.Qpts().shape() == shape_t<2>{nkpts, 3} );
  REQUIRE( mfobj.qk_to_k2().shape() == shape_t<2>{nkpts, nkpts} );
  REQUIRE(mfobj.qminus().shape(0) == nkpts);

  // basis info
  auto nbnd = mfobj.nbnd();
  REQUIRE(nbnd > 0);
  auto fft_mesh = mfobj.fft_grid_dim();
  REQUIRE(mfobj.fft_grid_size() == fft_mesh(0)*fft_mesh(1)*fft_mesh(2) );
}

TEST_CASE("pyscf_orbs", "[mean_field]") {
  auto& mpi = utils::make_unit_test_mpi_context(); 

  auto [outdir,prefix] = utils::utest_filename(mf::pyscf_source);
  auto mfobj = mf::make_MF(mpi, mf::pyscf_source, outdir, prefix);

  using local_Array_t = memory::array<HOST_MEMORY, ComplexType, 4>;
  auto dPsi_g = mf::read_distributed_orbital_set<local_Array_t>(mfobj, mpi->comm, 'g');
  auto dPsi_r = mf::read_distributed_orbital_set<local_Array_t>(mfobj, mpi->comm, 'r');
}

TEST_CASE("pyscf_rho", "[mean_field]") {
  auto& mpi = utils::make_unit_test_mpi_context(); 
  auto [outdir,prefix] = utils::utest_filename(mf::pyscf_source);
  auto mfobj = mf::make_MF(mpi, mf::pyscf_source, outdir, prefix);

  grids::distributed_r_grid g(mfobj.fft_grid_dim(), mfobj.lattv(), mpi->comm,{mpi->comm.size(),1,1});
  // charge density
  auto rho = mf::distributed_charge_density(mfobj,g);

  //h5::file file("rho_pyscf.h5", 'w');
  //h5::group grp(file);
  //nda::h5_write(grp, "rho", rho.local(), false);
}

TEST_CASE("bz_symm", "[mean_field]") 
{
  auto world = boost::mpi3::environment::get_world_instance();
  if(world.root()) {
    h5::file file("dummy_symm.h5", 'w');
    h5::group grp(file);
    h5::group sgrp = grp.create_group("System");
    mf::bz_symm::gamma_point_h5(sgrp);
  }
  world.barrier();
  {
    h5::file file("dummy_symm.h5", 'r');
    h5::group grp(file);
    h5::group sgrp = grp.open_group("System");
    REQUIRE( mf::bz_symm::can_init_from_h5(sgrp) );
  }
  world.barrier();
  mf::bz_symm s("dummy_symm.h5"); 
  REQUIRE( s.nkpts == 1 );
  REQUIRE( s.nqpts == 1 );
  REQUIRE( s.kpts == nda::array<double,2>{{0,0,0}} );
  REQUIRE( s.Qpts == nda::array<double,2>{{0,0,0}} );
  REQUIRE( s.symm_list.size() == 1 );
  world.barrier();
  if(world.root()) {
    remove("dummy_symm.h5");
  }
  world.barrier();
 
  auto s1 = mf::bz_symm::gamma_point_instance();
  REQUIRE( s1.nkpts == 1 );
  REQUIRE( s1.nqpts == 1 );
  REQUIRE( s1.kpts == nda::array<double,2>{{0,0,0}} );
  REQUIRE( s1.Qpts == nda::array<double,2>{{0,0,0}} );
  REQUIRE( s1.symm_list.size() == 1 );
  world.barrier();
}

/*
 * The minimal /System/BZ layout (MP grid + IBZ kpoints + symmetry operations)
 * must rebuild exactly the maps that the full layout stores verbatim.  This is
 * what lets abinit2coqui emit a symmetry-reduced checkpoint without
 * reimplementing bz_symm's map construction converter-side.
 *
 * Rather than ship a second fixture, this reads the full block from the
 * symmetry-reduced LiH checkpoint and feeds its own (kp_grid, IBZ kpoints,
 * symm_list) back through the constructor, then compares every map.
 */
TEST_CASE("bz_symm_minimal_rebuild", "[mean_field]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto [outdir, prefix] = utils::utest_filename("bdft_lih222_sym");
  std::string fn = outdir + "/" + prefix + ".h5";
  if(not std::filesystem::exists(fn)) return;

  nda::array<double, 2> latt(3,3), recv(3,3);
  mf::bz_symm full;
  {
    h5::file file(fn, 'r');
    h5::group grp(file);
    h5::group sgrp = grp.open_group("System");
    REQUIRE( mf::bz_symm::can_init_from_h5(sgrp) );
    nda::h5_read(grp, "System/lattice_vectors", latt);
    nda::h5_read(grp, "System/reciprocal_vectors", recv);
    full.initialize_from_h5(sgrp);
  }
  // a genuinely symmetry-reduced checkpoint, else the test proves nothing
  REQUIRE( full.nkpts_ibz < full.nkpts );
  REQUIRE( full.symm_list.size() > 1 );

  // IBZ kpoints are the leading nkpts_ibz entries of the stored (cartesian) list
  nda::array<double, 2> kpts_ibz(full.nkpts_ibz, 3);
  kpts_ibz() = full.kpts(nda::range(full.nkpts_ibz), nda::range(3));

  bool no_q_sym = (full.nqpts_ibz == full.nqpts);
  bool use_trev = (full.nkpts_trev_pairs > 0);
  mf::bz_symm rebuilt(world, no_q_sym, latt, recv, full.kp_grid, kpts_ibz,
                      full.symm_list, use_trev);

  REQUIRE( rebuilt.nkpts     == full.nkpts );
  REQUIRE( rebuilt.nkpts_ibz == full.nkpts_ibz );
  REQUIRE( rebuilt.nqpts     == full.nqpts );
  REQUIRE( rebuilt.nqpts_ibz == full.nqpts_ibz );
  REQUIRE( rebuilt.nkpts_trev_pairs == full.nkpts_trev_pairs );

  REQUIRE( rebuilt.kp_symm      == full.kp_symm );
  REQUIRE( rebuilt.kp_to_ibz    == full.kp_to_ibz );
  REQUIRE( rebuilt.kp_trev_pair == full.kp_trev_pair );
  REQUIRE( rebuilt.qk_to_k2     == full.qk_to_k2 );
  REQUIRE( rebuilt.qminus       == full.qminus );
  REQUIRE( rebuilt.qp_symm      == full.qp_symm );
  REQUIRE( rebuilt.qp_to_ibz    == full.qp_to_ibz );
  REQUIRE( rebuilt.qsymms       == full.qsymms );
  REQUIRE( rebuilt.nq_per_s     == full.nq_per_s );
  REQUIRE( rebuilt.ks_to_k      == full.ks_to_k );
  REQUIRE( rebuilt.Qs           == full.Qs );
  REQUIRE( rebuilt.qs_to_q      == full.qs_to_q );

  auto max_dev = [](auto const& a, auto const& b) {
    double d = 0.0;
    for(long i=0; i<a.size(); ++i)
      d = std::max(d, std::abs(a.data()[i] - b.data()[i]));
    return d;
  };
  REQUIRE( max_dev(rebuilt.kpts, full.kpts) < 1e-10 );
  REQUIRE( max_dev(rebuilt.kpts_crys, full.kpts_crys) < 1e-10 );
  REQUIRE( max_dev(rebuilt.Qpts, full.Qpts) < 1e-10 );
  world.barrier();
}


TEST_CASE("model_mf", "[mean_field]")
{
  auto& mpi = utils::make_unit_test_mpi_context(); 

  std::string prefix = "dummy.model";
  int ns = 1;
  int nk = 1;
  int nb = 4;
  double nel = 2.0;
  { // construct through h5
    if(mpi->comm.root()) {

      h5::file file("dummy_symm.h5", 'w');
      h5::group grp(file);
      h5::group sgrp = grp.create_group("System");
      mf::bz_symm::gamma_point_h5(sgrp);
      mf::bz_symm symm("dummy_symm.h5");

      auto h = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});
      auto s = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});
      auto d = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});
      auto f = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});

      mf::model::model_system m(mpi,"./",prefix,symm,ns,1,nel,h,s,d,f);
      m.save(prefix+".h5");
      remove("dummy_symm.h5");
    }
    mpi->comm.barrier();

    mf::model::model_readonly m(mpi,"./",prefix,-1);
    mf::MF mf(m);

    REQUIRE(mf.nspin() == ns);
    REQUIRE(mf.nbnd() == nb);
    REQUIRE(mf.nelec() == nel);
    mpi->comm.barrier();

    if(mpi->comm.root())
      remove("dummy.model.h5");
    mpi->comm.barrier();
  }

  { // construct directly
    auto symm = mf::bz_symm::gamma_point_instance();

    auto h = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});
    auto s = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});
    auto d = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});
    auto f = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});

    mf::model::model_system m_sys(mpi,"./",prefix,symm,ns,1,nel,h,s,d,f);
    mf::MF mf(mf::model::model_readonly{m_sys});

    REQUIRE(mf.nspin() == ns);
    REQUIRE(mf.nbnd() == nb);
    REQUIRE(mf.nelec() == nel);
    mpi->comm.barrier();
  }

}

TEST_CASE("symmetry_tools", "[mean_field]")
{
  auto& mpi = utils::make_unit_test_mpi_context(); 

  SECTION("transform_k2g")
  {
    auto mf = mf::default_MF(mpi, "qe_lih223_sym", mf::h5_input_type);
   
    auto wfc_g = mf.wfc_truncated_grid();
    auto mesh = mf.fft_grid_dim(); 
    long ngm = wfc_g->size();
    nda::array<long,1> k2g(ngm,0);
    grids::map_truncated_grid_to_fft_grid(*wfc_g, mesh, k2g);
    auto symm_list = mf.symm_list();
    nda::array<ComplexType,1> *Xft = nullptr;
    nda::stack_array<double, 3> Gs;
    Gs() = 0; // Gs = ??? , these should be stored in sys! 

    nda::array<long,1> k2g_h(k2g);
    utils::transform_k2g(false,symm_list[1],Gs,mesh,mf.kpts(0),k2g_h,Xft);

#if defined(ENABLE_DEVICE)
    auto k2g_d = memory::to_memory_space<DEVICE_MEMORY>(k2g);
    utils::transform_k2g(false,symm_list[1],Gs,mesh,mf.kpts(0),k2g_d,Xft);
    auto k2g_d_h = nda::to_host(k2g_d); 
    ARRAY_EQUAL(k2g_d_h,k2g_h); 
#endif
  }
}

} // bdft_tests
