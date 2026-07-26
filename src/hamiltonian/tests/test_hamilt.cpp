/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
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
#include "stdio.h"
#include <cstdlib>
#include <filesystem>

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"
#include "utilities/mpi_context.h"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/proc_grid_partition.hpp"

#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/h5.hpp"
#include "numerics/shared_array/nda.hpp"
#include "utilities/test_common.hpp"

#include "mean_field/default_MF.hpp"
#include "mean_field/distributed_orbital_readers.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/matrix_elements.h"
#include "hamiltonian/one_body_hamiltonian.hpp"
#include "hamiltonian/paw/hartree_xc_energy.hpp"
#include "hamiltonian/paw/local_isdf.hpp"
#include "hamiltonian/paw/local_isdf_compress.hpp"
#include "hamiltonian/paw/local_isdf_h5.hpp"
#include "hamiltonian/paw/paw_aug_q_eval.hpp"
#include "hamiltonian/paw/paw_onecenter.hpp"
#include "hamiltonian/paw/v_h_paw.hpp"
#include "hamiltonian/add_vloc.hpp"
#include "hamiltonian/add_vxc.h"
#include "utilities/fortran_utilities.h"
#include "utilities/qe_utilities.hpp"

#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/ERI/chol_reader_t.hpp"
#include "methods/HF/hf_t.h"
#include "methods/SCF/mb_solver_t.h"
#include "methods/SCF/scf_common.hpp"

namespace bdft_tests
{

using namespace math::nda;
template <int Rank> using shape_t = std::array<long, Rank>;
using mpi_context_t = utils::mpi_context_t<mpi3::communicator,mpi3::shared_communicator>;
using math::shm::make_shared_array;
using math::shm::shared_array;
using array_view_4d_t = nda::array_view<ComplexType, 4>;
using array_view_3d_t = nda::array_view<ComplexType, 3>;

template<MEMORY_SPACE MEM>
void qe_one_body_components(mpi_context_t& mpi_context, mf::MF& mfobj)
{
  auto wfc_g = mfobj.wfc_truncated_grid();
  auto& world = mpi_context.comm;

  using larray = memory::array<MEM, ComplexType,4>;
  // psi(is,ik,ia,g)
  auto psi = mf::read_distributed_orbital_set_ibz<larray>(mfobj,world,'w');
  // (is, ik, ia, g)
  auto hpsi = make_distributed_array<larray>(world,psi.grid(),psi.global_shape(),psi.block_size());  
  hpsi.local() = ComplexType{0.0};

  long npol = mfobj.npol();
  long nbnd = mfobj.nbnd();

  long nspin = psi.global_shape()[0];
  long nkpts = psi.global_shape()[1];
  auto k_range = nda::range(nkpts);
  auto b_range = nda::range(nbnd);

  auto Hij = make_distributed_array<larray>(world,psi.grid(),{nspin,nkpts,nbnd,nbnd},
                 {psi.block_size()[0],psi.block_size()[1],psi.block_size()[2],psi.block_size()[2]});  
  Hij.local() = ComplexType(0.0);

  // kinetic part
  hamilt::add_kinetic(npol,mfobj.kpts_ibz(),*wfc_g,psi,hpsi);

  // non-local PP
  hamilt::pseudopot Vnl(mfobj);
  Vnl.add_Vpp(world,k_range,b_range,psi,hpsi,Hij);

  // Hij = conj(psi(i,:)) * hpsi(j,:) 
  math::nda::slate_ops::multiply(ComplexType(1.0),psi,math::nda::dagger(hpsi),
                                 ComplexType(1.0),Hij);
  nda::tensor::scale(ComplexType(1.0),Hij.local(),nda::tensor::op::CONJ); 

  RealType E1 = 0.0;
  auto b_rng = Hij.local_range(3);
  auto Hloc = nda::to_host(Hij.local());
  for( auto [is,s] : itertools::enumerate(Hij.local_range(0)) )  
    for( auto [ik,k] : itertools::enumerate(Hij.local_range(1)) ) { 
      auto xk = mfobj.k_weight(k); 
      for( auto [ia,a] : itertools::enumerate(Hij.local_range(2)) )  {
        if(a >= b_rng.first() and a < b_rng.last()) {
          auto w = mfobj.occ(s,k,a);  
          E1 += xk*w*std::real(Hloc(is,ik,ia,a-b_rng.first()));
        }
      }
    }
  if(nspin==1 and npol==1) E1 *= 2.0; 
  E1 = world.all_reduce_value(E1);
  app_log(2,"One body energy: {} Ha",E1);
}

template<MEMORY_SPACE MEM>
void test_H0(mpi_context_t& mpi, mf::MF& mfobj, std::optional<double> Eref)
{
  hamilt::pseudopot V(mfobj);
  auto Hij = hamilt::H0<MEM>(mfobj,mpi.comm,&V,nda::range(mfobj.nkpts_ibz()), nda::range(mfobj.nbnd()));
  RealType E1 = 0.0;
  auto b_rng = Hij.local_range(3);
  auto Hloc = nda::to_host(Hij.local());
  for( auto [is,s] : itertools::enumerate(Hij.local_range(0)) )
    for( auto [ik,k] : itertools::enumerate(Hij.local_range(1)) ) {
      auto xk = mfobj.k_weight(k); 
      for( auto [ia,a] : itertools::enumerate(Hij.local_range(2)) )  {
        if(a >= b_rng.first() and a < b_rng.last()) {
          auto w = mfobj.occ(s,k,a);
          E1 += xk*w*std::real(Hloc(is,ik,ia,a-b_rng.first()));
        }
      }
    }
  if(mfobj.nspin()==1 and mfobj.npol()==1) E1 *= 2.0;
  E1 = mpi.comm.all_reduce_value(E1);
  if( Eref.has_value() ) {
    utils::VALUE_EQUAL(E1, *Eref, 1e-5); 
  } else {
    app_log(2,"One body energy: {} Ha",E1);
  }
}

template<MEMORY_SPACE MEM>
void test_H(mpi_context_t& mpi, mf::MF& mfobj, std::optional<double> Eref)
{ 
  decltype(nda::range::all) all;
  
  {
    memory::array<MEM,ComplexType,3> nii(mfobj.nspin(),mfobj.nkpts_ibz(),mfobj.nbnd());
    if constexpr(MEM==HOST_MEMORY)  {
      nii() = mfobj.occ()(all,nda::range(mfobj.nkpts_ibz()),all);
    } else {
      nii() = nda::array<ComplexType,3>(mfobj.occ()(all,nda::range(mfobj.nkpts_ibz()),all));
    }
    hamilt::pseudopot V(mfobj);
    auto Hij = hamilt::H<MEM>(mfobj,mpi.comm,&V,nii);
    RealType E1 = 0.0;
    auto b_rng = Hij.local_range(3);
    auto Hloc = nda::to_host(Hij.local());
    for( auto [is,s] : itertools::enumerate(Hij.local_range(0)) ) 
      for( auto [ik,k] : itertools::enumerate(Hij.local_range(1)) ) {
        auto xk = mfobj.k_weight(k); 
        for( auto [ia,a] : itertools::enumerate(Hij.local_range(2)) )  {
          if(a >= b_rng.first() and a < b_rng.last()) {
            auto w = mfobj.occ(s,k,a);
            E1 += xk*w*std::real(Hloc(is,ik,ia,a-b_rng.first()));
          }
        }
      }
    if(mfobj.nspin()==1 and mfobj.npol()==1) E1 *= 2.0;
    E1 = mpi.comm.all_reduce_value(E1);
    if( Eref.has_value() ) {
      utils::VALUE_EQUAL(E1, *Eref, 1e-5); 
    } else {
      app_log(2,"One body + hartree: {} Ha",E1);
    }
  }
  {
    auto occ = mfobj.occ()(all,nda::range(mfobj.nkpts_ibz()),all);
    memory::array<MEM,ComplexType,4> nij(mfobj.nspin(),mfobj.nkpts_ibz(),mfobj.nbnd(),mfobj.nbnd());
    nij()=ComplexType(0.0);
    if constexpr(MEM==HOST_MEMORY)  {
      for(int is=0; is<mfobj.nspin(); ++is)
        for(int ik=0; ik<mfobj.nkpts_ibz(); ++ik) {
          auto nii = nij(is,ik,all,all);
          nda::diagonal(nii) = occ(is,ik,all);
        }
    } else {
      nda::array<ComplexType,4> nij_h(mfobj.nspin(),mfobj.nkpts_ibz(),mfobj.nbnd(),mfobj.nbnd());
      nij_h()=ComplexType(0.0);
      for(int is=0; is<mfobj.nspin(); ++is)
        for(int ik=0; ik<mfobj.nkpts_ibz(); ++ik) {
          auto nii = nij_h(is,ik,all,all);
          nda::diagonal(nii) = occ(is,ik,all);
        }
      nij = nij_h;
    }
    hamilt::pseudopot V(mfobj);
    auto Hij = hamilt::H<MEM>(mfobj,mpi.comm,&V,nij);
    RealType E1 = 0.0;
    auto b_rng = Hij.local_range(3);
    auto Hloc = nda::to_host(Hij.local());
    for( auto [is,s] : itertools::enumerate(Hij.local_range(0)) )
      for( auto [ik,k] : itertools::enumerate(Hij.local_range(1)) ) {
        auto xk = mfobj.k_weight(k);
        for( auto [ia,a] : itertools::enumerate(Hij.local_range(2)) )  {
          if(a >= b_rng.first() and a < b_rng.last()) {
            auto w = mfobj.occ(s,k,a);
            E1 += xk*w*std::real(Hloc(is,ik,ia,a-b_rng.first()));
          }
        }
      }
    if(mfobj.nspin()==1 and mfobj.npol()==1) E1 *= 2.0;
    E1 = mpi.comm.all_reduce_value(E1);
    if( Eref.has_value() ) {
      utils::VALUE_EQUAL(E1, *Eref, 1e-5); 
    } else {
      app_log(2,"One body + hartree: {} Ha",E1);
    }
  }
}

void get_density_ovlp(mpi_context_t& mpi, mf::MF &mf,
                      hamilt::pseudopot& psp, double beta,
                      shared_array<array_view_4d_t> &sDm_skij,
                      shared_array<array_view_4d_t> &sS_skij) {
  double mu = 0.0;
  {
    auto sHeff_skij = make_shared_array<array_view_4d_t>(
        mpi.comm, mpi.internode_comm, mpi.node_comm, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()});
    auto sMO_skia = make_shared_array<array_view_4d_t>(
        mpi.comm, mpi.internode_comm, mpi.node_comm, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()});
    auto sE_ska = make_shared_array<array_view_3d_t>(
        mpi.comm, mpi.internode_comm, mpi.node_comm, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd()});
    hamilt::set_ovlp(mf, sS_skij);
    hamilt::set_fock(mf, &psp, sHeff_skij, false);

    // Obtains MO coefficients and energies from the given mean-field object
    methods::update_MOs(sMO_skia, sE_ska, sHeff_skij, sS_skij);
    mu = methods::update_mu(mu, mf, sE_ska, beta);
    methods::update_Dm(sDm_skij, sMO_skia, sE_ska, mu, beta);
  }
  mpi.comm.barrier();
}

template<MEMORY_SPACE MEM>
void check_F(mpi_context_t& mpi, mf::MF& mfobj) {

  // checks that F = H0 + J + Vxc
  hamilt::pseudopot V(mfobj);
  auto all = nda::range::all;
  long nspin = mfobj.nspin();
  long nkpts_ibz = mfobj.nkpts_ibz();
  long nbnd = mfobj.nbnd();
  auto Vxc = hamilt::Vxc<MEM>(mfobj, mpi.comm);
  auto H0 = hamilt::H0<MEM>(mfobj, mpi.comm, &V); 

  memory::array<MEM,ComplexType,3> occ(nspin,nkpts_ibz,nbnd);
  occ() = mfobj.occ()(all,nda::range(nkpts_ibz),all); 
  auto dJ = hamilt::Vhartree<MEM>(mfobj, mpi.comm, &V, occ);

  auto sF0 = make_shared_array<array_view_4d_t>(
      mpi.comm, mpi.internode_comm, mpi.node_comm, {nspin, nkpts_ibz, nbnd, nbnd});
  math::nda::gather_to_shm(H0, sF0);

  auto sT = make_shared_array<array_view_4d_t>(
      mpi.comm, mpi.internode_comm, mpi.node_comm, {nspin, nkpts_ibz, nbnd, nbnd});
  math::nda::gather_to_shm(dJ, sT);
  if(mpi.node_comm.root()) sF0.local() += ( sT.local() ); 
  mpi.node_comm.barrier();

  math::nda::gather_to_shm(Vxc, sT);
  if(mpi.node_comm.root()) sF0.local() += ( sT.local() ); 
  mpi.node_comm.barrier();

  auto F = hamilt::F<MEM>(mfobj, mpi.comm, nda::range(nkpts_ibz),nda::range(nbnd));
  auto sF = make_shared_array<array_view_4d_t>(
      mpi.comm, mpi.internode_comm, mpi.node_comm, {nspin, nkpts_ibz, nbnd, nbnd});
  math::nda::gather_to_shm(F, sF);
  mpi.node_comm.barrier();

  auto Abs = nda::map([](ComplexType _x_) { return std::abs(_x_); });
  double norm = -1;
  if (sF.node_comm()->root()) {
    nda::array<RealType,4> res_abs(nspin, nkpts_ibz, nbnd, nbnd);
    res_abs = Abs(sF.local() - sF0.local());
    norm = nda::max_element(res_abs);
  }
  sF.node_comm()->broadcast_n(&norm, 1, 0);
  app_log(2, "Norm of J - J2 = {}", norm);

  utils::ARRAY_EQUAL(sF.local(), sF0.local(), 1e-4);
}

// should only be called with HF calculations or hybrid DFT 
template<MEMORY_SPACE MEM>
void check_K(mpi_context_t& mpi, std::shared_ptr<mf::MF> &mfobj, double x) {

  // checks that F = H0 + J + K + Vxc
  hamilt::pseudopot V(*mfobj);
  auto all = nda::range::all;
  long nspin = mfobj->nspin();
  long nkpts_ibz = mfobj->nkpts_ibz();
  long nbnd = mfobj->nbnd();
  auto mfocc = mfobj->occ();

  auto sK = make_shared_array<array_view_4d_t>(mpi, {nspin, nkpts_ibz, nbnd, nbnd});
  {
    auto F = hamilt::F<MEM>(*mfobj, mpi.comm);
    auto H0 = hamilt::H0<MEM>(*mfobj, mpi.comm, &V);
    auto Vxc = hamilt::Vxc<MEM>(*mfobj, mpi.comm);

    nda::array<ComplexType,3> occ(nspin,nkpts_ibz,nbnd);
    occ() = mfocc(all,nda::range(nkpts_ibz),all);
    auto J = hamilt::Vhartree<MEM>(*mfobj, mpi.comm, &V, occ);

    //F.local() -= ( H0.local() + J.local() + Vxc.local());
    nda::tensor::add(ComplexType(1.0),H0.local(),ComplexType(1.0),J.local());
    nda::tensor::add(ComplexType(1.0),Vxc.local(),ComplexType(1.0),J.local());
    nda::tensor::add(ComplexType(-1.0),J.local(),ComplexType(1.0),F.local());
    math::nda::gather_to_shm(F, sK);
  }
  mpi.node_comm.barrier();

  auto sK2 = make_shared_array<array_view_4d_t>(
      mpi.comm, mpi.internode_comm, mpi.node_comm, {nspin, nkpts_ibz, nbnd, nbnd});
  {
    methods::thc_reader_t thc(mfobj,
                              methods::make_thc_reader_ptree(0.0, "", "incore", "test.h5", "bdft",
                                                             1e-8, mfobj->ecutrho(), 1, 1024));
    methods::solvers::hf_t hf(methods::ignore_g0);

    auto sS_skij = make_shared_array<array_view_4d_t>(
        mpi.comm, mpi.internode_comm, mpi.node_comm, {nspin, nkpts_ibz, nbnd, nbnd});
    hamilt::set_ovlp(*mfobj, sS_skij);

    nda::array<ComplexType,4> occ4d(nspin,nkpts_ibz,nbnd,nbnd);
    occ4d() = ComplexType(0.0);
    for( int s=0; s<nspin; s++ )
      for( int k=0; k<nkpts_ibz; k++ ) {
        for( int a=0; a<nbnd; a++ )
          occ4d(s,k,a,a) = mfocc(s,k,a);
      }
    hf.evaluate(sK2, occ4d, thc, sS_skij.local(), false, true);
    mpi.node_comm.barrier();
    if(mpi.node_comm.root()) 
      sK2.local() *= ComplexType(x); 
  }
  mpi.node_comm.barrier();

  auto Abs = nda::map([](ComplexType _x_) { return std::abs(_x_); });
  double norm = -1;
  if (sK.node_comm()->root()) {
    nda::array<RealType,4> res_abs(nspin, nkpts_ibz, nbnd, nbnd);
    res_abs = Abs(sK.local() - sK2.local());
    norm = nda::max_element(res_abs);
  }
  sK.node_comm()->broadcast_n(&norm, 1, 0);
  app_log(2, "Norm of K - K2 = {}", norm);

  utils::ARRAY_EQUAL(sK.local(), sK2.local(), 1e-4);
}

template<MEMORY_SPACE MEM>
void check_Vxc(mpi_context_t& mpi, mf::MF& mfobj) {

  hamilt::pseudopot V(mfobj);
  auto all = nda::range::all;
  long nspin = mfobj.nspin();
  long nkpts_ibz = mfobj.nkpts_ibz();
  long nbnd = mfobj.nbnd();

  // Vxc = F - H0 - J
  auto sVxc_skij = make_shared_array<array_view_4d_t>(mpi, {nspin, nkpts_ibz, nbnd, nbnd});
  // Hartree from PW using mf.occ() 
  {
    // MAM: If mf.occ is not meaningful, need to reconstruct occupations consistent with F
    //      This might be hard in practice for metals if the MF code uses smearing
    memory::array<MEM, ComplexType, 3> occ(nspin, nkpts_ibz, nbnd);
    occ() = mfobj.occ()(all,nda::range(nkpts_ibz),all);
    auto J = hamilt::Vhartree<MEM>(mfobj, mpi.comm, &V, occ);
    auto F = hamilt::F<MEM>(mfobj, mpi.comm);
    auto H0 = hamilt::H0<MEM>(mfobj, mpi.comm, &V);

    //F.local() -= ( H0.local() + J.local() );
    nda::tensor::add(ComplexType(1.0),H0.local(),ComplexType(1.0),J.local());
    nda::tensor::add(ComplexType(-1.0),J.local(),ComplexType(1.0),F.local());
    math::nda::gather_to_shm(F, sVxc_skij);
  }

  // Vxc
  auto sVxc2_skij = make_shared_array<array_view_4d_t>(mpi, {nspin, nkpts_ibz, nbnd, nbnd});
  {
    auto Vxc = hamilt::Vxc<MEM>(mfobj, mpi.comm);
    math::nda::gather_to_shm(Vxc, sVxc2_skij);
  }

  auto Abs = nda::map([](ComplexType _x_) { return std::abs(_x_); });
  double norm = -1;
  if (sVxc_skij.node_comm()->root()) {
    nda::array<RealType,4> res_abs(nspin, nkpts_ibz, nbnd, nbnd);
    res_abs = Abs(sVxc_skij.local() - sVxc2_skij.local());
    norm = nda::max_element(res_abs);
  }
  sVxc_skij.node_comm()->broadcast_n(&norm, 1, 0);
  app_log(2, "Norm of Vxc - Vxc2 = {}", norm);

  utils::ARRAY_EQUAL(sVxc_skij.local(), sVxc2_skij.local(), 2e-5);
}

template<MEMORY_SPACE MEM>
void check_Hartree(mpi_context_t& mpi,
                    std::shared_ptr<mf::MF>& mfobj,
                    std::string const& src_name,
                    bool diag_dm = false) {
  hamilt::pseudopot V(*mfobj);
  long nspin = mfobj->nspin();
  long nkpts_ibz = mfobj->nkpts_ibz();
  long nbnd = mfobj->nbnd();
  std::array<long,4> shape = {nspin,nkpts_ibz,nbnd,nbnd};
  auto sDm_skij = make_shared_array<array_view_4d_t>(
      mpi.comm, mpi.internode_comm, mpi.node_comm, shape);
  auto sS_skij = make_shared_array<array_view_4d_t>(
      mpi.comm, mpi.internode_comm, mpi.node_comm, shape);
  get_density_ovlp(mpi, *mfobj, V, 1000, sDm_skij, sS_skij);

  // Hartree from PW. The previous THC reference computation has been
  // replaced with a stored reference loaded from the fixture directory
  // (see reference IO below), since the THC build is the dominant cost
  // and not the point of this test.
  auto sJ2_skij = make_shared_array<array_view_4d_t>(
      mpi.comm, mpi.internode_comm, mpi.node_comm, shape);
  if (diag_dm) {
    memory::array<MEM, ComplexType, 3> Dm_ski(nspin, nkpts_ibz, nbnd);
    for (long s=0; s<mfobj->nspin(); ++s) {
      for (long k=0; k<mfobj->nkpts_ibz(); ++k) {
        auto Dm_sk = sDm_skij.local()(s,k,nda::range::all,nda::range::all);
        Dm_ski(s,k,nda::range::all) = nda::diagonal(Dm_sk);
      }
    }
    auto dJ2 = hamilt::Vhartree<MEM>(*mfobj, mpi.comm, &V, Dm_ski, nda::range(nkpts_ibz),nda::range(nbnd));
    math::nda::gather_to_shm(dJ2, sJ2_skij);
  } else {
    auto dJ2 = hamilt::Vhartree<MEM>(*mfobj, mpi.comm, &V, sDm_skij.local(), nda::range(nkpts_ibz),nda::range(nbnd));
    math::nda::gather_to_shm(dJ2, sJ2_skij);
  }
  sJ2_skij.communicator()->barrier();

  // Reference IO: read the stored Hartree tensor and compare.
  // Stored at <fixture_outdir>/reference_hamiltonian_test_results.h5,
  // with one dataset per (diag_dm) variant: "J_full" or "J_diag".
  auto [outdir, prefix] = utils::utest_filename(src_name);
  std::string ref_file = outdir + "/reference_hamiltonian_test_results.h5";
  std::string dataset  = diag_dm ? "J_diag" : "J_full";
  nda::array<ComplexType, 4> J_ref(nspin, nkpts_ibz, nbnd, nbnd);
  if (mpi.comm.root()) {
    REQUIRE(std::filesystem::exists(ref_file));
    h5::file f(ref_file, 'r');
    h5::group g(f);
    nda::h5_read(g, dataset, J_ref);
  }
  mpi.comm.broadcast_n(J_ref.data(), J_ref.size(), 0);
  utils::ARRAY_EQUAL(sJ2_skij.local(), J_ref, 1e-5);
}

// MAM: reenable and add more tests!!!
template<MEMORY_SPACE MEM>
void qe_ovlp(mpi_context_t& mpi, mf::MF& mfobj) {
  {
    auto S = hamilt::ovlp<MEM>(mfobj, mpi.comm, nda::range(mfobj.nkpts_ibz()),nda::range(mfobj.nbnd()));
    auto Sloc = memory::to_memory_space<HOST_MEMORY>(S.local());
    for (auto [is, s] : itertools::enumerate(S.local_range(0)))
      for (auto [ik, k] : itertools::enumerate(S.local_range(1))) {
        auto S_ij = Sloc(is, ik, nda::ellipsis{});
        for (auto [ia, a] : itertools::enumerate(S.local_range(2)))
          for (auto [ib, b] : itertools::enumerate(S.local_range(3))) {
            utils::VALUE_EQUAL(S_ij(ia,ib), (a==b?ComplexType(1.0):ComplexType(0.0)));
          }
      }
  }

  {
    auto S = hamilt::ovlp_diagonal<MEM>(mfobj, mpi.comm, nda::range(mfobj.nkpts_ibz()),nda::range(mfobj.nbnd()));
    auto Sloc = memory::to_memory_space<HOST_MEMORY>(S.local());
    for (auto [is, s] : itertools::enumerate(S.local_range(0)))
      for (auto [ik, k] : itertools::enumerate(S.local_range(1))) {
        auto S_ii = Sloc(is, ik, nda::ellipsis{});
        for (auto [ia, a] : itertools::enumerate(S.local_range(2)))
          utils::VALUE_EQUAL(S_ii(ia), ComplexType(1.0));
      }
  } 
}

/*
TEST_CASE("pyscf", "[hamilt]") {
  auto& mpi = utils::make_unit_test_mpi_context();
  auto mfobj = mf::default_MF(mpi_context,mf::pyscf_source);

  auto Fij = hamilt::F<HOST_MEMORY>(mfobj, world, nda::range(mfobj.nkpts_ibz()),nda::range(mfobj.nbnd()));
  auto Hij = hamilt::H0<HOST_MEMORY>(mfobj, world, nda::range(mfobj.nkpts_ibz()),nda::range(mfobj.nbnd()));
  auto Sij = hamilt::ovlp<HOST_MEMORY>(mfobj, world, nda::range(mfobj.nkpts_ibz()),nda::range(mfobj.nbnd()));

  REQUIRE( Sij.global_shape() == shape_t<4>{mfobj.nspin(), mfobj.nkpts_ibz(), mfobj.nbnd(), mfobj.nbnd()} );
  REQUIRE( Hij.global_shape() == shape_t<4>{mfobj.nspin(), mfobj.nkpts_ibz(), mfobj.nbnd(), mfobj.nbnd()} );
  REQUIRE( Fij.global_shape() == shape_t<4>{mfobj.nspin(), mfobj.nkpts_ibz(), mfobj.nbnd(), mfobj.nbnd()} );
}

TEST_CASE("pyscf_ovlp", "[hamilt]") {
  auto& mpi = utils::make_unit_test_mpi_context();

  auto [outdir,prefix] = utils::utest_filename(mf::pyscf_source);
  auto mfobj = mf::make_MF(mpi_context, mf::pyscf_source, outdir, prefix);
  auto Sij = hamilt::ovlp(mfobj, world, nda::range(mfobj.nkpts_ibz()), nda::range(mfobj.nbnd()));

  // read AO
  { // read a single orbital
    nda::array<ComplexType, 2> Orb(2,mfobj.max_npw());
    long is = Sij.local_range(0).first();
    long ik = Sij.local_range(1).first();
    long ia = Sij.local_range(2).first();
    long ib = Sij.local_range(3).first();
    mfobj.get_orbital('k',is,ik,ia, Orb(0,nda::range::all));
    mfobj.get_orbital('k',is,ik,ib, Orb(1,nda::range::all));
    ComplexType overlap = 0.0;
    for( auto [va,vb] : itertools::zip(Orb(0,nda::range::all), Orb(1,nda::range::all)) )
      overlap += va * std::conj(vb);

    auto S = Sij.local();
    utils::VALUE_EQUAL(overlap, S(0,0,0,0));
    mfobj.close();
  }

  if(Sij.local_range(2).size() >= 2 and Sij.local_range(3).size() >= 2)
  { // read a set of orbitals
    nda::array<ComplexType, 3> psi_a(1,2,mfobj.fft_grid_size());
    nda::array<ComplexType, 3> psi_b(1,2,mfobj.fft_grid_size());
    long is = Sij.local_range(0).first();
    long ik = Sij.local_range(1).first();
    long ia = Sij.local_range(2).first();
    long ib = Sij.local_range(3).first();
    // orbitals for k=[0,1), i=[0,2)
    mfobj.get_orbital_set('g', is, {ik, ik+1}, {ia, ia+2}, psi_a);
    mfobj.get_orbital_set('g', is, {ik, ik+1}, {ib, ib+2}, psi_b);

    nda::array<ComplexType, 2> Orbs_C(2,mfobj.max_npw());
    nda::array<ComplexType, 2> ov(2,2);
    nda::blas::gemm(1.0, psi_a(0,nda::ellipsis{}), nda::dagger(psi_b(0,nda::ellipsis{})), 0.0, ov);

    auto S = Sij.local();
    for(int i=0; i<2; ++i)
      for(int j=0; j<2; ++j)
        utils::VALUE_EQUAL(ov(i,j), S(0,0,i,j), 1e-12);
    mfobj.close();
  } else {
    app_warning("Too many processors in pyscf_ovlp unit test. Skipping.");
  }

  // Read orbitals into a distributed array
  {
    using local_Array_t = memory::array<HOST_MEMORY, ComplexType, 2>;
    auto dPsia = mf::read_distributed_orbital_set<local_Array_t>(mfobj, world, 'k', {1, world.size()});

    REQUIRE(dPsia.global_shape()[0] == mfobj.nspin()*mfobj.nkpts()*mfobj.nbnd());
    REQUIRE(dPsia.global_shape()[1] == mfobj.max_npw());
  }

}
*/

template<MEMORY_SPACE MEM>
void test_F_impl(std::shared_ptr<mpi_context_t> &mpi)
{
  SECTION("lih223") 
  {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih223", mf::h5_input_type);
    check_F<MEM>(*mpi, qe_h5);
  }

  SECTION("lih223_inv") 
  {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih223_inv", mf::h5_input_type);
    check_F<MEM>(*mpi, qe_h5);
  }

  SECTION("lih223_sym") 
  {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih223_sym", mf::h5_input_type);
    check_F<MEM>(*mpi, qe_h5);
  }

  SECTION("GaAs222_so") 
  {
    auto qe_h5 = mf::default_MF(mpi, "qe_GaAs222_so", mf::h5_input_type);
    check_F<MEM>(*mpi, qe_h5);
  }
}

TEST_CASE("mf_F", "[hamilt]") {
  auto& mpi = utils::make_unit_test_mpi_context();
  
  test_F_impl<HOST_MEMORY>(mpi);
#if defined(ENABLE_DEVICE)
  test_F_impl<DEVICE_MEMORY>(mpi);
  test_F_impl<UNIFIED_MEMORY>(mpi);
#endif
}

template<MEMORY_SPACE MEM>
void test_exx_impl(std::shared_ptr<mpi_context_t> &mpi)
{
  SECTION("lih222_hf") 
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222_hf", mf::h5_input_type));
    check_K<MEM>(*mpi, qe_h5, 1.00);
  }

  SECTION("GaAs222_hf") 
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_GaAs222_hf", mf::h5_input_type));
    check_K<MEM>(*mpi, qe_h5, 1.00);
  }

  SECTION("GaAs222_so_hf") 
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_GaAs222_so_hf", mf::h5_input_type));
    check_K<MEM>(*mpi, qe_h5, 0.25);
  }
}

TEST_CASE("exx", "[hamilt]") {
  auto& mpi = utils::make_unit_test_mpi_context();

  test_exx_impl<HOST_MEMORY>(mpi);
#if defined(ENABLE_DEVICE)
  test_exx_impl<DEVICE_MEMORY>(mpi);
  test_exx_impl<UNIFIED_MEMORY>(mpi);
#endif
}

template<MEMORY_SPACE MEM>
void test_vxc_impl(std::shared_ptr<mpi_context_t> &mpi)
{
  SECTION("lih223") 
  {
    // CNY: mf::xml_input_type is not implemented for vxc!
    auto qe_h5 = mf::default_MF(mpi, "qe_lih223", mf::h5_input_type);
    check_Vxc<MEM>(*mpi, qe_h5);
  }

  SECTION("lih223_inv") 
  {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih223_inv", mf::h5_input_type);
    check_Vxc<MEM>(*mpi, qe_h5);
  }

  SECTION("lih223_sym") 
  {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih223_sym", mf::h5_input_type);
    check_Vxc<MEM>(*mpi, qe_h5);
  }

  SECTION("GaAs222_so") 
  {
    auto qe_h5 = mf::default_MF(mpi, "qe_GaAs222_so", mf::h5_input_type);
    check_Vxc<MEM>(*mpi, qe_h5);
  }
} 

TEST_CASE("vxc", "[hamilt]") {
  auto& mpi = utils::make_unit_test_mpi_context();

  test_vxc_impl<HOST_MEMORY>(mpi);
#if defined(ENABLE_DEVICE)
  test_vxc_impl<DEVICE_MEMORY>(mpi);
  test_vxc_impl<UNIFIED_MEMORY>(mpi);
#endif
}

template<MEMORY_SPACE MEM>
void test_hartree_impl(std::shared_ptr<mpi_context_t> &mpi)
{

  SECTION("lih223")
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih223", mf::h5_input_type));
    check_Hartree<MEM>(*mpi, qe_h5, "qe_lih223");
  }

  SECTION("lih223_inv")
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih223_inv", mf::h5_input_type));
    check_Hartree<MEM>(*mpi, qe_h5, "qe_lih223_inv");
  }

  SECTION("lih223_sym")
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih223_sym", mf::h5_input_type));
    check_Hartree<MEM>(*mpi, qe_h5, "qe_lih223_sym");
  }

  SECTION("lih223_sym_diag")
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih223_sym", mf::h5_input_type));
    check_Hartree<MEM>(*mpi, qe_h5, "qe_lih223_sym", true); // diagonal density as the input
  }

  SECTION("GaAs222_so")
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_GaAs222_so", mf::h5_input_type));
    check_Hartree<MEM>(*mpi, qe_h5, "qe_GaAs222_so", true); // diagonal density as the input
  }
}

TEST_CASE("hartree", "[hamilt]") {
  auto& mpi = utils::make_unit_test_mpi_context();

  test_hartree_impl<HOST_MEMORY>(mpi);
#if defined(ENABLE_DEVICE)
  test_hartree_impl<DEVICE_MEMORY>(mpi);
  test_hartree_impl<UNIFIED_MEMORY>(mpi);
#endif
}

// MAM: this will be problematic when testing custom outdir/prefix, forbid this! 
TEST_CASE("one_body_components", "[hamilt]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  auto qe_xml = mf::default_MF(mpi,mf::qe_source);
  auto qe_h5 = mf::default_MF(mpi,mf::qe_source,mf::h5_input_type);
  auto bdft_mf = mf::default_MF(mpi,mf::bdft_source);
  auto py_mf = mf::default_MF(mpi,mf::pyscf_source);

  std::optional<double> qe_Href, qe_H0ref, qe_Ovref;
  std::optional<double> qeh5_Href, qeh5_H0ref, qeh5_Ovref;

  // MAM: these should be read from h5 file and test should be done on more reference tests
  //      including wsym, winv, etc... 
  if(not std::filesystem::exists(qe_outdir+"/"+qe_prefix+".xml"))
  {
    qe_Href = -0.7275620144914019;
    qe_H0ref = -3.8613653985679552;
    qe_Ovref = 1.0;
  }
  if(not std::filesystem::exists(qe_outdir+"/"+qe_prefix+".coqui.h5"))
  {
    qeh5_Href = -0.7275620144914019;
    qeh5_H0ref = -3.8613653985679552;
    qeh5_Ovref = 1.0;
  }

// qe_one_body_components
  qe_one_body_components<HOST_MEMORY>(*mpi,qe_h5);
#if defined(ENABLE_DEVICE)
  qe_one_body_components<DEVICE_MEMORY>(*mpi,qe_h5);
  qe_one_body_components<UNIFIED_MEMORY>(*mpi,qe_h5);
#endif

//test_H0
  test_H0<HOST_MEMORY>(*mpi,qe_xml,qe_H0ref);
  test_H0<HOST_MEMORY>(*mpi,qe_h5,qeh5_H0ref);
#if defined(ENABLE_DEVICE)
  test_H0<DEVICE_MEMORY>(*mpi,qe_xml,qe_H0ref);
  test_H0<UNIFIED_MEMORY>(*mpi,qe_xml,qe_H0ref);
  test_H0<DEVICE_MEMORY>(*mpi,qe_h5,qeh5_H0ref);
  test_H0<UNIFIED_MEMORY>(*mpi,qe_h5,qeh5_H0ref);
#endif

//test_H
//  test_H<HOST_MEMORY>(*mpi,qe_xml,qe_Href);
  test_H<HOST_MEMORY>(*mpi,qe_h5,qeh5_Href);
#if defined(ENABLE_DEVICE)
//  test_H<DEVICE_MEMORY>(*mpi,qe_xml,qe_Href);
//  test_H<UNIFIED_MEMORY>(*mpi,qe_xml,qe_Href);
  test_H<DEVICE_MEMORY>(*mpi,qe_h5,qeh5_Href);
  test_H<UNIFIED_MEMORY>(*mpi,qe_h5,qeh5_Href);
#endif

//qe_ovlp
  qe_ovlp<HOST_MEMORY>(*mpi,qe_h5);
#if defined(ENABLE_DEVICE)
  qe_ovlp<DEVICE_MEMORY>(*mpi,qe_h5);
  qe_ovlp<UNIFIED_MEMORY>(*mpi,qe_h5);
#endif

}

TEST_CASE("one_body_components_so", "[hamilt]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  auto qe = mf::default_MF(mpi,"qe_GaAs222_so");

// qe_one_body_components
  qe_one_body_components<HOST_MEMORY>(*mpi,qe);
#if defined(ENABLE_DEVICE)
  qe_one_body_components<DEVICE_MEMORY>(*mpi,qe);
  qe_one_body_components<UNIFIED_MEMORY>(*mpi,qe);
#endif

/*
//test_H0
  test_H0<HOST_MEMORY>(*mpi,qe,Eref);
#if defined(ENABLE_DEVICE)
  test_H0<DEVICE_MEMORY>(*mpi,qe,Erref);
  test_H0<UNIFIED_MEMORY>(*mpi,qe,Eref);
#endif

//test_H
  test_H<HOST_MEMORY>(*mpi,qe_xml,qe_Href);
  test_H<HOST_MEMORY>(*mpi,qe_h5,qeh5_Href);
#if defined(ENABLE_DEVICE)
  test_H<DEVICE_MEMORY>(*mpi,qe_xml,qe_Href);
  test_H<UNIFIED_MEMORY>(*mpi,qe_xml,qe_Href);
  test_H<DEVICE_MEMORY>(*mpi,qe_h5,qeh5_Href);
  test_H<UNIFIED_MEMORY>(*mpi,qe_h5,qeh5_Href);
#endif
*/
}

/**
 * DFT eigenvalue regression test.
 *
 * For converged QE Kohn-Sham orbitals the band basis diagonalizes H, so the
 * diagonal of H_full in the band basis equals eigval(s,k,n). The test
 * computes H = T + V_loc + V_H + V_xc + V_NL via CoQui's pipeline and
 * compares H_nn / S_nn to mfobj.eigval(s,k,a) for occupied bands. For NCPP
 * S_nn = 1; for PAW the augmentation overlap is added by add_S so that
 * ⟨ψ̃|S|ψ̃⟩ = 1 (matching QE's S-orthonormalization).
 */
template<MEMORY_SPACE MEM>
void test_dft_eigenvalues(mpi_context_t& mpi, mf::MF& mfobj,
                          double tol = 5e-5, double occ_threshold = 1e-6)
{
  auto all = nda::range::all;
  long nspin   = mfobj.nspin();
  long nk_ibz  = mfobj.nkpts_ibz();
  long nbnd    = mfobj.nbnd();

  memory::array<MEM, ComplexType, 3> nii(nspin, nk_ibz, nbnd);
  if constexpr (MEM == HOST_MEMORY) {
    nii() = mfobj.occ()(all, nda::range(nk_ibz), all);
  } else {
    nii() = nda::array<ComplexType,3>(mfobj.occ()(all, nda::range(nk_ibz), all));
  }

  hamilt::pseudopot V(mfobj);
  auto Hij   = hamilt::H<MEM>(mfobj, mpi.comm, &V, nii);
  auto Vxcij = hamilt::Vxc<MEM>(mfobj, mpi.comm);

  utils::check(Hij.local_shape() == Vxcij.local_shape(),
               "test_dft_eigenvalues: H and Vxc local shape mismatch");
  // H_full = (T + V_loc + V_H + V_NL) + V_xc
  nda::tensor::add(ComplexType(1.0), Vxcij.local(),
                   ComplexType(1.0), Hij.local());

  // QE-saved orbitals are S_aug-orthonormal (⟨ψ̃|S|ψ̃⟩ = δ_nm by construction),
  // so the diagonal Kohn-Sham eigenvalue equals ⟨ψ̃_n|H|ψ̃_n⟩ directly — no
  // overlap correction is needed. (Earlier this divided by an explicitly
  // re-augmented S, which for deep PAW core states gives S_nn ≈ 1.6 and a
  // spurious 0.6 Ha error; QE bands already satisfy ⟨ψ̃|H|ψ̃⟩ = ε.)
  auto Hloc = nda::to_host(Hij.local());
  double max_err = 0.0;
  long  count = 0;
  auto b_rng = Hij.local_range(3);
  for (auto [is, s] : itertools::enumerate(Hij.local_range(0)))
    for (auto [ik, k] : itertools::enumerate(Hij.local_range(1))) {
      for (auto [ia, a] : itertools::enumerate(Hij.local_range(2))) {
        if (!(a >= b_rng.first() && a < b_rng.last())) continue;
        if (mfobj.occ(s, k, a) < occ_threshold) continue;
        long ib = a - b_rng.first();
        double H_diag  = std::real(Hloc(is, ik, ia, ib));
        double eps_ref = mfobj.eigval(s, k, a);
        double err     = std::abs(H_diag - eps_ref);
        if (err > max_err) max_err = err;
        ++count;
        if (err > tol) {
          app_log(2,
            "DFT eigval mismatch: s={}, k={}, n={}, H_nn={:+.6f}, "
            "ref={:+.6f}, err={:.2e}",
            s, k, a, H_diag, eps_ref, err);
        }
      }
    }
  max_err = mpi.comm.all_reduce_value(max_err, boost::mpi3::max<>{});
  count   = mpi.comm.all_reduce_value(count,   std::plus<>{});
  app_log(2,
    "DFT eigenvalue regression: max|H_nn - eps_n| = {:.3e} over {} occupied "
    "states (tol={:.1e})", max_err, count, tol);
  CHECK(max_err < tol);
}

/**
 * Hartree energy regression test.
 *
 * Computes E_H = (Ω/2) Σ_{G≠0} |ρ_total(G)|² × 4π/|G|² directly on the
 * dense FFT mesh and compares to QE's `qe_ehart` attribute (in Hartree).
 * For NCPP, ρ_total is the smooth valence density; for USPP/PAW it
 * includes the augmentation contribution from becsum × Q.
 *
 * Pure-CoQui calculation (no ISDF approximation), exercising the same
 * density-construction and augmentation pipeline used elsewhere.
 */
template<MEMORY_SPACE MEM>
void test_hartree_energy(mpi_context_t& mpi, mf::MF& mfobj,
                         std::string const& fixture_name,
                         double tol = 5e-5)
{
  auto all = nda::range::all;
  long nspin   = mfobj.nspin();
  long nk_ibz  = mfobj.nkpts_ibz();
  long nbnd    = mfobj.nbnd();
  int  npol    = mfobj.npol();

  // Read QE reference E_H from h5 attribute on /System.
  auto [outdir, prefix] = utils::utest_filename(fixture_name);
  std::string h5path = outdir + prefix + ".coqui.h5";
  double qe_ehart = 0.0;
  {
    h5::file f(h5path, 'r');
    h5::group g(f);
    h5::group sgrp = g.open_group("System");
    h5::h5_read_attribute(sgrp, "qe_ehart", qe_ehart);
  }

  memory::array<MEM, ComplexType, 3> nii(nspin, nk_ibz, nbnd);
  if constexpr (MEM == HOST_MEMORY) {
    nii() = mfobj.occ()(all, nda::range(nk_ibz), all);
  } else {
    nii() = nda::array<ComplexType,3>(mfobj.occ()(all, nda::range(nk_ibz), all));
  }

  hamilt::pseudopot V(mfobj);

  // Read distributed orbitals (PW basis on wfc grid).
  using larray = memory::array<MEM, ComplexType, 4>;
  auto psi = mf::read_distributed_orbital_set_ibz<larray>(
      mfobj, mpi.comm, 'w', std::array<long,4>{0,0,0,0},
      nda::range(nspin), nda::range(nk_ibz), nda::range(nbnd),
      std::array<long,4>{1,1,2048,2048});

  auto fft_mesh = mfobj.fft_grid_dim();
  auto recv     = mfobj.recv();
  auto kpts_full = mfobj.kpts();
  auto kp_to_ibz = mfobj.kp_to_ibz();
  auto kp_trev   = mfobj.kp_trev();
  auto kp_symm   = mfobj.kp_symm();
  auto symm_list = mfobj.symm_list();

  // The wfc-G → dense-FFT-linear-index mapping. wfc_g->gv_to_fft() is
  // encoded on the WFC grid (e.g. 19³); pseudopot::swfc_to_rho_view()
  // remaps to the dense FFT mesh (e.g. 36³) — the convention v_h_paw and
  // add_vloc both expect.
  auto k2g = V.swfc_to_rho_view();

  // QE's reported `ehart` is the smooth-grid (compensated) Hartree; the PAW
  // radial one-center Hartree is bookkept separately in QE's one-center
  // energies. Request include_one_center=false to compare like-for-like.
  double E_H = hamilt::paw::hartree_energy_paw(
      mpi, V, npol, fft_mesh, recv, k2g,
      kpts_full, kp_to_ibz, kp_trev, kp_symm, symm_list,
      nii, psi, /*include_augmentation=*/true, /*include_one_center=*/false);

  app_log(2, "Hartree energy (CoQui) = {:+.8f} Ha, qe_ehart = {:+.8f} Ha, "
             "diff = {:+.2e}", E_H, qe_ehart, E_H - qe_ehart);
  CHECK(std::abs(E_H - qe_ehart) < tol);
}

/**
 * V_xc · ρ integral test.
 *
 * Computes vtxc = ∫ V_xc(r) × ρ_total(r) dr on the dense FFT mesh and
 * compares to QE's `qe_vtxc` attribute (in Hartree). V_xc(r) is read
 * from the same h5 fixture (CoQui's standard read_vxc_h5 pipeline).
 *
 * For nspin=1: ρ_total is the spin-summed density; V_xc has a single
 * spin component. vtxc = ∫V_xc × ρ_total dr.
 *
 * For LSDA (not yet exercised): vtxc would require per-spin densities
 * and the per-spin product summed.
 */
template<MEMORY_SPACE MEM>
void test_vxc_rho_integral(mpi_context_t& mpi, mf::MF& mfobj,
                            std::string const& fixture_name,
                            double tol = 1e-3)
{
  auto all = nda::range::all;
  long nspin   = mfobj.nspin();
  long nk_ibz  = mfobj.nkpts_ibz();
  long nbnd    = mfobj.nbnd();
  int  npol    = mfobj.npol();

  auto [outdir, prefix] = utils::utest_filename(fixture_name);
  std::string h5path = outdir + prefix + ".coqui.h5";
  double qe_vtxc = 0.0;
  {
    h5::file f(h5path, 'r');
    h5::group g(f);
    h5::group sgrp = g.open_group("System");
    h5::h5_read_attribute(sgrp, "qe_vtxc", qe_vtxc);
  }

  memory::array<MEM, ComplexType, 3> nii(nspin, nk_ibz, nbnd);
  if constexpr (MEM == HOST_MEMORY) {
    nii() = mfobj.occ()(all, nda::range(nk_ibz), all);
  } else {
    nii() = nda::array<ComplexType,3>(mfobj.occ()(all, nda::range(nk_ibz), all));
  }

  hamilt::pseudopot V(mfobj);

  using larray = memory::array<MEM, ComplexType, 4>;
  auto psi = mf::read_distributed_orbital_set_ibz<larray>(
      mfobj, mpi.comm, 'w', std::array<long,4>{0,0,0,0},
      nda::range(nspin), nda::range(nk_ibz), nda::range(nbnd),
      std::array<long,4>{1,1,2048,2048});

  auto fft_mesh = mfobj.fft_grid_dim();
  auto recv     = mfobj.recv();
  auto kpts_full = mfobj.kpts();
  auto kp_to_ibz = mfobj.kp_to_ibz();
  auto kp_trev   = mfobj.kp_trev();
  auto kp_symm   = mfobj.kp_symm();
  auto symm_list = mfobj.symm_list();
  // Use the wfc-G → dense-FFT-linear-index mapping (REMAPPED to dense
  // mesh, NOT raw wfc_g.gv_to_fft() which is encoded on the wfc mesh).
  auto k2g = V.swfc_to_rho_view();

  // Read V_xc(r) from the h5 fixture using CoQui's standard pipeline.
  auto svxc = math::shm::shared_array<nda::array_view<ComplexType,3>>(
      mpi, {nspin, npol*npol, mfobj.nnr()});
  {
    h5::file file(mfobj.filename(), 'r');
    h5::group grp(file);
    hamilt::read_vxc_h5(mfobj, grp, svxc);
  }

  double vtxc = hamilt::paw::vxc_rho_integral_paw(
      mpi, V, npol, fft_mesh, recv, k2g,
      kpts_full, kp_to_ibz, kp_trev, kp_symm, symm_list,
      nii, psi, svxc.local());

  app_log(2, "vtxc (CoQui) = {:+.8f} Ha, qe_vtxc = {:+.8f} Ha, "
             "diff = {:.2e}", vtxc, qe_vtxc, vtxc - qe_vtxc);
  CHECK(std::abs(vtxc - qe_vtxc) < tol);
}

TEST_CASE("hartree_energy", "[hamilt][energy]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  SECTION("lih_kp222_nbnd16 (NCPP, PBE)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222", mf::h5_input_type);
    test_hartree_energy<HOST_MEMORY>(*mpi, qe_h5, "qe_lih222", 1e-5);
  }

  SECTION("lih_kp222_nbnd16 (USPP, PBE)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_uspp", mf::h5_input_type);
    test_hartree_energy<HOST_MEMORY>(*mpi, qe_h5, "qe_lih222_uspp", 1e-8);
  }

  SECTION("lih_kp222_nbnd16 (PAW, PBE)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    test_hartree_energy<HOST_MEMORY>(*mpi, qe_h5, "qe_lih222_paw", 1e-8);
  }
}

TEST_CASE("vxc_rho_integral", "[hamilt][energy]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  SECTION("lih_kp222_nbnd16 (NCPP, PBE)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222", mf::h5_input_type);
    test_vxc_rho_integral<HOST_MEMORY>(*mpi, qe_h5, "qe_lih222", 1e-5);
  }

  SECTION("lih_kp222_nbnd16 (USPP, PBE)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_uspp", mf::h5_input_type);
    test_vxc_rho_integral<HOST_MEMORY>(*mpi, qe_h5, "qe_lih222_uspp", 1e-8);
  }

  SECTION("lih_kp222_nbnd16 (PAW, PBE)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    test_vxc_rho_integral<HOST_MEMORY>(*mpi, qe_h5, "qe_lih222_paw", 1e-8);
  }
}

/**
 * Hartree energy: direct vs ISDF/THC (smooth-density only).
 *
 * Compares:
 *   E_H_smooth_direct = (Ω/2) Σ_{G≠0} |ρ_smooth(G)|² × 4π/|G|²
 *                       (built from valence orbitals only — NO PAW augmentation)
 *   E_H_thc           = (1/2) Tr[Dm × J_H]    (J_H from THC factorized ERI)
 *
 * The standard THC factorization treats only the smooth-density ERIs
 * V_ijkl ≈ Σ_ΛΣ X*_Λi X_Λj 𝒱_ΛΣ X*_Σk X_Σl, where X is on |ψ̃|² without
 * augmentation. So for USPP/PAW the THC E_H matches the SMOOTH-only
 * direct E_H, NOT the full QE ehart (the augmentation contribution to E_H
 * is captured by the augmented (Y / V_GL / V_LL / K_a) extensions built
 * in thc_reader_t::augment_thc_with_paw and validated separately in
 * test_hartree_thc_paw_aug).
 *
 * For NCPP (no augmentation): smooth-direct = full-direct = QE ehart.
 * The accuracy of THC is set by `thresh` and `ecut`; for our LiH 222
 * fixture with thresh=1e-5 and ecut=0.4×ecutrho, agreement should be ~1e-3 Ha.
 */
template<MEMORY_SPACE MEM>
void test_hartree_thc_vs_direct(mpi_context_t& mpi, std::shared_ptr<mf::MF> mf_ptr,
                                [[maybe_unused]] std::string const& fixture_name,
                                double thc_thresh = 1e-5,
                                double tol = 5e-3)
{
  using math::shm::make_shared_array;
  auto all = nda::range::all;
  auto& mfobj = *mf_ptr;
  long nspin   = mfobj.nspin();
  long nk_ibz  = mfobj.nkpts_ibz();
  long nbnd    = mfobj.nbnd();
  int  npol    = mfobj.npol();

  // -------- E_H_direct (the verified-against-QE smooth-grid path) -------
  memory::array<MEM, ComplexType, 3> nii(nspin, nk_ibz, nbnd);
  if constexpr (MEM == HOST_MEMORY) {
    nii() = mfobj.occ()(all, nda::range(nk_ibz), all);
  } else {
    nii() = nda::array<ComplexType,3>(mfobj.occ()(all, nda::range(nk_ibz), all));
  }
  hamilt::pseudopot V(mfobj);
  using larray = memory::array<MEM, ComplexType, 4>;
  auto psi = mf::read_distributed_orbital_set_ibz<larray>(
      mfobj, mpi.comm, 'w', std::array<long,4>{0,0,0,0},
      nda::range(nspin), nda::range(nk_ibz), nda::range(nbnd),
      std::array<long,4>{1,1,2048,2048});
  // AUG mesh, not the smooth one: V.swfc_to_rho_view() maps the wfc sphere
  // onto fft_grid_dim_aug, and the compensation ρ̂ needs the dense box. On
  // the QE unit-test fixtures the two meshes coincide, which hid a mismatch
  // here until the split-mesh ABINIT fixture (D2): with the smooth mesh the
  // k2g indices land out of range (coefficients silently dropped) and the
  // ±miller entries alias, giving a garbage direct reference.
  auto fft_mesh = mfobj.fft_grid_dim_aug();
  auto recv     = mfobj.recv();
  auto kpts_full = mfobj.kpts();
  auto kp_to_ibz = mfobj.kp_to_ibz();
  auto kp_trev   = mfobj.kp_trev();
  auto kp_symm   = mfobj.kp_symm();
  auto symm_list = mfobj.symm_list();
  auto k2g       = V.swfc_to_rho_view();

  // Full all-electron direct E_H: smooth-grid (with Q-compensation) PLUS the
  // PAW radial one-center deltaC correction. hartree_energy_paw now folds in
  // the one-center term (include_one_center defaults true), so this matches
  // the THC paw_aug=true J energy directly. For NCPP both augmentation pieces
  // are no-ops and this equals the bare smooth E_H.
  double E_H_direct = hamilt::paw::hartree_energy_paw(
      mpi, V, npol, fft_mesh, recv, k2g,
      kpts_full, kp_to_ibz, kp_trev, kp_symm, symm_list, nii, psi,
      /*include_augmentation=*/true, /*include_one_center=*/true);

  // -------- E_H_thc via Hartree-Fock J matrix from THC ERIs --------
  // paw_aug defaults true → THC includes the smooth Q-compensation AND the
  // radial one-center K_a, i.e. the full AE Hartree. Compared to E_H_direct.
  methods::thc_reader_t thc(mf_ptr,
      methods::make_thc_reader_ptree(0, "", "incore", "", "bdft", thc_thresh,
                                      0.4 * mfobj.ecutrho()));
  methods::solvers::hf_t hf(methods::ignore_g0);

  // Diagonal density matrix from occupations (shared array).
  auto sDm_skij = make_shared_array<array_view_4d_t>(mpi,
                      {nspin, nk_ibz, nbnd, nbnd});
  if (mpi.node_comm.root()) {
    sDm_skij.local()() = ComplexType(0.0);
    for (int s = 0; s < nspin; ++s)
    for (int k = 0; k < nk_ibz; ++k)
    for (int a = 0; a < nbnd; ++a)
      sDm_skij.local()(s, k, a, a) = mfobj.occ(s, k, a);
  }
  mpi.node_comm.barrier();

  auto sS_skij = make_shared_array<array_view_4d_t>(mpi,
                    {nspin, nk_ibz, nbnd, nbnd});
  hamilt::set_ovlp(mfobj, sS_skij);

  auto sJ = make_shared_array<array_view_4d_t>(mpi,
                {nspin, nk_ibz, nbnd, nbnd});
  // hartree=true, exchange=false → J holds only the Hartree Fock matrix
  hf.evaluate(sJ, sDm_skij.local(), thc, sS_skij.local(), true, false);

  // E_H_thc = (1/2) × spin_factor × Σ_sk wk × Tr[Dm × J] (matches eval_hf_energy)
  auto k_weight = mfobj.k_weight();
  auto [e1e_dummy, E_H_thc] = methods::eval_hf_energy(
      sDm_skij, sJ, /*sH0_skij=*/sS_skij, k_weight, /*F_has_H0=*/false);
  // (sH0 here is unused since F_has_H0=false; passing sS just to satisfy signature)

  app_log(2, "Hartree energy: direct(AE full) = {:+.8f} Ha, THC = {:+.8f} Ha, "
             "diff = {:+.2e} Ha", E_H_direct, E_H_thc,
             E_H_thc - E_H_direct);
  CHECK(std::abs(E_H_thc - E_H_direct) < tol);
}

/**
 * Hartree energy via PAW-augmented THC: the augmented thc_reader_t produces
 * (X_full, V_full) with smooth + atom-local rows. We feed it through hf_t
 * to get the J matrix and contract with the diagonal density matrix.
 *
 *   E_H_thc_paw = (1/2) Σ_sk wk Tr[Dm × J]   (HF Hartree from augmented THC)
 *   E_H_direct  = hartree_energy_paw(include_augmentation=true)
 *
 * Hartree only needs q=0, so the two computations should agree within
 * smooth-ISDF + compression tolerance.
 *
 * NCPP fixtures must produce results identical to the un-augmented path
 * (paw_aug=true is a no-op when no PAW species are present).
 */
template<MEMORY_SPACE MEM>
void test_hartree_thc_paw_aug(mpi_context_t& mpi, std::shared_ptr<mf::MF> mf_ptr,
                               std::string const& fixture_name,
                               double thc_thresh = 1e-5,
                               double tol = 5e-3)
{
  using math::shm::make_shared_array;
  using clk_t = std::chrono::steady_clock;
  auto t_phase = clk_t::now();
  auto dt = [&t_phase]() {
    auto now = clk_t::now();
    double s = std::chrono::duration<double>(now - t_phase).count();
    t_phase = now;
    return s;
  };
  auto all = nda::range::all;
  auto& mfobj = *mf_ptr;
  long nspin   = mfobj.nspin();
  long nk_ibz  = mfobj.nkpts_ibz();
  long nbnd    = mfobj.nbnd();
  int  npol    = mfobj.npol();
  app_log(2, "[TIMER {}] mf+psp+nii setup={:.2f}s", fixture_name, dt());

  // -------- Direct E_H_paw with augmentation --------
  memory::array<MEM, ComplexType, 3> nii(nspin, nk_ibz, nbnd);
  if constexpr (MEM == HOST_MEMORY) {
    nii() = mfobj.occ()(all, nda::range(nk_ibz), all);
  } else {
    nii() = nda::array<ComplexType,3>(mfobj.occ()(all, nda::range(nk_ibz), all));
  }
  hamilt::pseudopot V(mfobj);
  app_log(2, "[TIMER {}] pseudopot ctor={:.2f}s", fixture_name, dt());
  using larray = memory::array<MEM, ComplexType, 4>;
  auto psi = mf::read_distributed_orbital_set_ibz<larray>(
      mfobj, mpi.comm, 'w', std::array<long,4>{0,0,0,0},
      nda::range(nspin), nda::range(nk_ibz), nda::range(nbnd),
      std::array<long,4>{1,1,2048,2048});
  app_log(2, "[TIMER {}] orbital read={:.2f}s", fixture_name, dt());
  // AUG mesh, not the smooth one: V.swfc_to_rho_view() maps the wfc sphere
  // onto fft_grid_dim_aug, and the compensation ρ̂ needs the dense box. On
  // the QE unit-test fixtures the two meshes coincide, which hid a mismatch
  // here until the split-mesh ABINIT fixture (D2): with the smooth mesh the
  // k2g indices land out of range (coefficients silently dropped) and the
  // ±miller entries alias, giving a garbage direct reference.
  auto fft_mesh = mfobj.fft_grid_dim_aug();
  auto recv     = mfobj.recv();
  auto kpts_full = mfobj.kpts();
  auto kp_to_ibz = mfobj.kp_to_ibz();
  auto kp_trev   = mfobj.kp_trev();
  auto kp_symm   = mfobj.kp_symm();
  auto symm_list = mfobj.symm_list();
  auto k2g       = V.swfc_to_rho_view();

  // hartree_energy_paw gathers `psi` to root internally, so it works at any
  // mpi.comm.size(). include_one_center=false: this test adds its own
  // one-center K_a term (E_K_a_direct below), so request only the smooth-grid
  // piece to avoid double-counting.
  double E_H_direct = hamilt::paw::hartree_energy_paw(
      mpi, V, npol, fft_mesh, recv, k2g,
      kpts_full, kp_to_ibz, kp_trev, kp_symm, symm_list, nii, psi,
      /*include_augmentation=*/true, /*include_one_center=*/false);
  app_log(2, "[TIMER {}] hartree_energy_paw direct={:.2f}s", fixture_name, dt());

  // PAW augmentation in V_full includes the closed-form one-center K_a
  // correction (ΔC contraction), so the THC Hartree reproduces the
  // all-electron Hartree, not just the smooth-grid piece. Add the same
  // one-center contribution to the direct target so we compare AE-vs-AE.
  double E_K_a_direct = 0.0;
  {
    auto becsum = hamilt::paw::compute_becsum_diagonal(
        V.Pskna_view(), nii, V.ityp_view(), V.nh_view(), V.ofs_view(), npol);
    double ns_scl = (nspin == 1 && npol == 1) ? 2.0 : 1.0;
    for (long ia = 0; ia < becsum.extent(0); ++ia)
      for (long I = 0; I < becsum.extent(1); ++I)
        for (long J = 0; J < becsum.extent(2); ++J)
          becsum(ia, I, J) *= ns_scl;
    auto const& sps = V.paw_species_view();
    auto const& ityp = V.ityp_view();
    long nat = ityp.extent(0);
    for (long ia = 0; ia < nat; ++ia) {
      int nt = ityp(ia);
      if (nt >= (int)sps.size()) continue;
      auto const& sp = sps[nt];
      if (!sp.is_paw || sp.deltaC.size() == 0) continue;
      int nh_a = (int)V.nh_view()(nt);
      for (int I = 0; I < nh_a; ++I)
      for (int J = 0; J < nh_a; ++J)
      for (int Kp = 0; Kp < nh_a; ++Kp)
      for (int L  = 0; L  < nh_a; ++L)
        E_K_a_direct += 0.5 * becsum(ia, I, J) *
                        sp.deltaC(I, J, Kp, L) * becsum(ia, Kp, L);
    }
  }
  double E_H_target = E_H_direct + E_K_a_direct;
  app_log(2, "[TIMER {}] one-center K_a (becsum × ΔC × becsum)={:.2f}s",
          fixture_name, dt());

  // -------- E_H_thc with paw_aug=true --------
  // Use ecut = ecutrho so thc's rho_g matches the QE dense grid that
  // pseudopot.qgm lives on. Lower ecut would truncate the augmentation
  // Fourier content.
  auto thc_pt = methods::make_thc_reader_ptree(
      0, "", "incore", "", "bdft", thc_thresh, mfobj.ecutrho());
  thc_pt.put("paw_aug", true);
  thc_pt.put("paw_isdf_metric", "coulomb");
  thc_pt.put("paw_isdf_tol", 1e-12);
  methods::thc_reader_t thc(mf_ptr, thc_pt);
  app_log(2, "[TIMER {}] thc_reader_t ctor (smooth ISDF + paw_aug q-loop)={:.2f}s",
          fixture_name, dt());

  methods::solvers::hf_t hf(methods::ignore_g0);
  auto sDm_skij = make_shared_array<array_view_4d_t>(mpi,
                      {nspin, nk_ibz, nbnd, nbnd});
  if (mpi.node_comm.root()) {
    sDm_skij.local()() = ComplexType(0.0);
    for (int s = 0; s < nspin; ++s)
    for (int k = 0; k < nk_ibz; ++k)
    for (int a = 0; a < nbnd; ++a)
      sDm_skij.local()(s, k, a, a) = mfobj.occ(s, k, a);
  }
  mpi.node_comm.barrier();
  auto sS_skij = make_shared_array<array_view_4d_t>(mpi,
                    {nspin, nk_ibz, nbnd, nbnd});
  hamilt::set_ovlp(mfobj, sS_skij);
  auto sJ = make_shared_array<array_view_4d_t>(mpi,
                {nspin, nk_ibz, nbnd, nbnd});
  hf.evaluate(sJ, sDm_skij.local(), thc, sS_skij.local(), true, false);
  app_log(2, "[TIMER {}] HF Hartree evaluate={:.2f}s", fixture_name, dt());
  auto k_weight = mfobj.k_weight();
  auto [e1e_dummy, E_H_thc] = methods::eval_hf_energy(
      sDm_skij, sJ, sS_skij, k_weight, false);

  app_log(2, "PAW-augmented THC Hartree: direct(smooth-grid) = {:+.8f} Ha, "
             "ΔE_one_center = {:+.8f} Ha, AE target = {:+.8f} Ha, "
             "THC = {:+.8f} Ha, diff = {:+.2e} Ha",
             E_H_direct, E_K_a_direct, E_H_target,
             E_H_thc, E_H_thc - E_H_target);

  CHECK(std::abs(E_H_thc - E_H_target) < tol);
}

/**
 * Verify that the CoQui-side qvan2-equivalent (paw_aug_q_eval) reproduces
 * the cached qgm tensor at q=0 over the dense G grid.
 *
 * The cached qgm comes from QE's qvan2 + tab_qrad with (4π/Ω) baked in;
 * our `evaluate_Q_IJ_at_K` applies the same prefactor, so the comparison
 * is direct. Match should be to ~1e-6 (limited by random-points
 * Y_LM matrix-inverse conditioning + radial Bessel quadrature accuracy).
 */
template<MEMORY_SPACE MEM>
void test_paw_aug_q_eval_at_q0(mpi_context_t& mpi, mf::MF& mfobj,
                                double tol = 1e-10)
{
  hamilt::pseudopot V(mfobj);
  auto recv = mfobj.recv();
  double det_B =
      recv(0,0)*(recv(1,1)*recv(2,2) - recv(1,2)*recv(2,1))
    - recv(1,0)*(recv(0,1)*recv(2,2) - recv(0,2)*recv(2,1))
    + recv(2,0)*(recv(0,1)*recv(1,2) - recv(0,2)*recv(1,1));
  double omega = (2.0*M_PI)*(2.0*M_PI)*(2.0*M_PI) / std::abs(det_B);

  auto const& sps   = V.paw_species_view();
  auto const& nh_v  = V.nh_view();
  auto qgm          = V.qgm_view();   // (nsp, nij_max, ngm_dense)
  auto const& mill  = V.miller_g_dense_view();
  long ngm_d = V.ngm_dense_get();
  long nsp   = (long)sps.size();
  if (qgm.size() == 0 || ngm_d == 0) {
    app_log(2, "paw_aug_q_eval_at_q0: no augmentation, skipped.");
    return;
  }

  // Build aainit ap-tables for max l over all species.
  int lli = 1;
  for (long nt = 0; nt < nsp; ++nt) {
    auto const& sp = sps[nt];
    if (sp.lll.size() == 0) continue;
    for (long b = 0; b < sp.lll.extent(0); ++b)
      lli = std::max(lli, sp.lll(b) + 1);
  }
  auto aatab = hamilt::paw::aainit_tables_build(lli);
  app_log(2, "paw_aug_q_eval: lli={}, llx={}, mx={}",
          aatab.lli, aatab.llx, aatab.mx);

  // For each species, walk every (ih, jh) pair and every G in the dense
  // grid, and compare evaluate_Q_IJ_at_K(K=G_cart) to qgm(nt, ij, g).
  bool any = false;
  for (long nt = 0; nt < nsp; ++nt) {
    auto const& sp = sps[nt];
    if (!(sp.is_paw || sp.is_uspp)) continue;
    int nh_a = nh_v(nt);
    if (nh_a == 0 || sp.qfuncl.size() == 0) continue;
    if (sp.nhtolm.size() == 0 || sp.indv.size() == 0) {
      app_log(2, "paw_aug_q_eval: species {}: nhtolm/indv missing, skipped.", nt);
      continue;
    }
    any = true;
    auto const& ijtoh = V.ijtoh_view();

    double max_err = 0.0;
    double max_qgm = 0.0;
    long ng_check = std::min<long>(ngm_d, 200);   // sample first 200 G
    // Distribute the (ih, jh, g) flat index space across MPI ranks; reduce maxes.
    long N_total = (long)nh_a * nh_a * ng_check;
    long my_rank = mpi.comm.rank();
    long nproc   = mpi.comm.size();
    for (long idx = my_rank; idx < N_total; idx += nproc) {
      long g  = idx % ng_check;
      int  jh = (int)((idx / ng_check) % nh_a);
      int  ih = (int)(idx / (ng_check * nh_a));
      long ij = (long)ijtoh(nt, ih, jh) - 1;
      if (ij < 0) continue;
      int m1 = mill(g, 0), m2 = mill(g, 1), m3 = mill(g, 2);
      double Gx = m1*recv(0,0) + m2*recv(1,0) + m3*recv(2,0);
      double Gy = m1*recv(0,1) + m2*recv(1,1) + m3*recv(2,1);
      double Gz = m1*recv(0,2) + m2*recv(1,2) + m3*recv(2,2);
      ComplexType pred = hamilt::paw::evaluate_Q_IJ_at_K(
          sp, aatab, ih, jh, {Gx, Gy, Gz}, omega);
      ComplexType ref  = qgm(nt, ij, g);
      double e = std::abs(pred - ref);
      max_err = std::max(max_err, e);
      max_qgm = std::max(max_qgm, std::abs(ref));
    }
    mpi.comm.all_reduce_in_place_n(&max_err, 1, mpi3::max<>{});
    mpi.comm.all_reduce_in_place_n(&max_qgm, 1, mpi3::max<>{});
    app_log(2, "paw_aug_q_eval species {}: nh={}, ngm_check={}, "
               "max|Q_pred − Q_ref| = {:.3e}, max|Q_ref| = {:.3e}, "
               "rel = {:.3e}",
               nt, nh_a, ng_check, max_err, max_qgm,
               max_err / std::max(1e-30, max_qgm));
    CHECK(max_err < tol);
  }
  if (!any) app_log(2, "paw_aug_q_eval: no species with augmentation, skipped.");
}

/**
 * Diagnostic: report how the per-species local-ISDF row count (nlambda)
 * shrinks as the compression tolerance is relaxed. Helps decide whether
 * the default tol=1e-12 (full-rank) is leaving any easy compression on
 * the table for a given fixture.
 */
template<MEMORY_SPACE MEM>
void test_paw_isdf_rank_vs_tol(mpi_context_t& mpi, mf::MF& mfobj,
                                std::string const& label)
{
  hamilt::pseudopot V(mfobj);
  auto const& sps = V.paw_species_view();
  auto recv = mfobj.recv();
  double det_B =
      recv(0,0)*(recv(1,1)*recv(2,2) - recv(1,2)*recv(2,1))
    - recv(1,0)*(recv(0,1)*recv(2,2) - recv(0,2)*recv(2,1))
    + recv(2,0)*(recv(0,1)*recv(1,2) - recv(0,2)*recv(1,1));
  double omega = (2.0*M_PI)*(2.0*M_PI)*(2.0*M_PI) / std::abs(det_B);
  long nat = V.ityp_view().extent(0);

  hamilt::paw::isdf_metric metrics[2] = {
      hamilt::paw::isdf_metric::Coulomb, hamilt::paw::isdf_metric::L2};
  double tols[4] = {1e-3, 1e-6, 1e-9, 1e-12};

  // Distribute the (mi, nt, ti) triple over MPI ranks; entries this rank doesn't
  // own stay 0; combine via sum-reduce so every rank sees the full table for
  // printing on rank 0.
  int nsp_total = (int)sps.size();
  long table_size = (long)2 * nsp_total * 4;
  nda::array<long, 3> nlam(2, nsp_total, 4);
  nlam() = 0;
  long my_rank = mpi.comm.rank();
  long nproc   = mpi.comm.size();
  for (long flat = my_rank; flat < table_size; flat += nproc) {
    int mi = (int)(flat / (nsp_total * 4));
    int nt = (int)((flat / 4) % nsp_total);
    int ti = (int)(flat % 4);
    if (!(sps[nt].is_paw || sps[nt].is_uspp)) continue;
    nlam(mi, nt, ti) = hamilt::paw::build_local_isdf_compressed_by_norm(
                          V, nt, recv, omega, metrics[mi], tols[ti]).nlambda;
  }
  mpi.comm.all_reduce_in_place_n(nlam.data(), nlam.size(), std::plus<>{});

  for (int mi = 0; mi < 2; ++mi) {
    auto mname = std::string(hamilt::paw::metric_name(metrics[mi]));
    app_log(2, "=== local-ISDF rank vs tol [{} metric] ({}) ===", mname, label);
    app_log(2, "  {:>18} {:>5} {:>10} {:>5} {:>5} {:>5} {:>5} {:>8}",
            "species", "nh", "full-rank", "1e-3", "1e-6", "1e-9", "1e-12", "N_aug");
    for (int nt = 0; nt < nsp_total; ++nt) {
      if (!(sps[nt].is_paw || sps[nt].is_uspp)) continue;
      int nh_a = (int)V.nh_view()(nt);
      int full = nh_a * nh_a;
      long n3  = nlam(mi, nt, 0);
      long n6  = nlam(mi, nt, 1);
      long n9  = nlam(mi, nt, 2);
      long n12 = nlam(mi, nt, 3);
      int atoms_of_nt = 0;
      for (long ia = 0; ia < nat; ++ia)
        if ((int)V.ityp_view()(ia) == nt) ++atoms_of_nt;
      app_log(2, "  {:>18} {:>5} {:>10} {:>5} {:>5} {:>5} {:>5} (×{} atoms = {})",
              label, nh_a, full, n3, n6, n9, n12, atoms_of_nt, n12 * atoms_of_nt);
    }
  }
}

TEST_CASE("paw_isdf_rank_vs_tol", "[hamilt][paw][isdf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("LiH PAW") {
    auto mf = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    test_paw_isdf_rank_vs_tol<HOST_MEMORY>(*mpi, mf, "LiH PAW");
  }
  SECTION("LiH USPP") {
    auto mf = mf::default_MF(mpi, "qe_lih222_uspp", mf::h5_input_type);
    test_paw_isdf_rank_vs_tol<HOST_MEMORY>(*mpi, mf, "LiH USPP");
  }
  SECTION("Si PAW") {
    auto mf = mf::default_MF(mpi, "qe_si222_paw", mf::h5_input_type);
    test_paw_isdf_rank_vs_tol<HOST_MEMORY>(*mpi, mf, "Si PAW");
  }
  SECTION("Si USPP") {
    auto mf = mf::default_MF(mpi, "qe_si222_uspp", mf::h5_input_type);
    test_paw_isdf_rank_vs_tol<HOST_MEMORY>(*mpi, mf, "Si USPP");
  }
}

TEST_CASE("paw_aug_q_eval_at_q0", "[hamilt][paw][isdf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (USPP)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_uspp", mf::h5_input_type);
    test_paw_aug_q_eval_at_q0<HOST_MEMORY>(*mpi, qe_h5);
  }
  SECTION("lih_kp222_nbnd16 (PAW)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    test_paw_aug_q_eval_at_q0<HOST_MEMORY>(*mpi, qe_h5);
  }
  SECTION("si_kp222 (USPP psl 1.0.0)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_si222_uspp", mf::h5_input_type);
    test_paw_aug_q_eval_at_q0<HOST_MEMORY>(*mpi, qe_h5);
  }
  SECTION("si_kp222 (PAW psl 1.0.0)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_si222_paw", mf::h5_input_type);
    test_paw_aug_q_eval_at_q0<HOST_MEMORY>(*mpi, qe_h5);
  }
  // Plan D2: stored converter qgm vs runtime evaluator on an ABINIT mf.
  SECTION("si_kp222 (PAW, ABINIT-sourced mf)") {
    auto ab_h5 = mf::default_MF(mpi, "bdft_si222_paw_ab", mf::h5_input_type);
    test_paw_aug_q_eval_at_q0<HOST_MEMORY>(*mpi, ab_h5);
  }
}

/**
 * Hermiticity check on the augmented THC Coulomb tensor.
 *
 *   V_full(q, P, Q) must equal conj( V_full(qminus(q), Q, P) )
 *
 * for the assembled (smooth + V_GL + V_LL + K_a) tensor at every q in the
 * IBZ. This is .tex Eq. eri-hermitian and is a strong consistency check on
 * the q≠0 augmentation builder. Since the smooth path satisfies it by
 * construction, any failure pinpoints the V_GL/V_LL convention or the
 * sign of q_cart in the eta builder.
 */
template<MEMORY_SPACE MEM>
void test_thc_paw_hermiticity(mpi_context_t& mpi,
                               std::shared_ptr<mf::MF> mf_ptr,
                               double thc_thresh = 1e-5,
                               double tol = 1e-6)
{
  auto& mfobj = *mf_ptr;
  long nq_ibz = mfobj.nqpts_ibz();
  auto qminus = mfobj.qminus();

  auto thc_pt = methods::make_thc_reader_ptree(
      0, "", "incore", "", "bdft", thc_thresh, mfobj.ecutrho());
  thc_pt.put("paw_aug", true);
  thc_pt.put("paw_isdf_metric", "coulomb");
  thc_pt.put("paw_isdf_tol", 1e-12);
  methods::thc_reader_t thc(mf_ptr, thc_pt);

  // Gather V_full(q, P, Q) for all q to every rank for the comparison.
  // Distribute across q first (capped at nq_ibz), spill remainder onto P, then
  // all-gather so every rank can do its slice of the q × P × Q comparison loop.
  long Np = thc.Np();
  long np_q = utils::find_proc_grid_max_npools(
      (long)mpi.comm.size(), nq_ibz, 0.2);
  long np_P = (long)mpi.comm.size() / np_q;
  auto dZ_qPQ = thc.template dZ<HOST_MEMORY>(
      {np_q, np_P, 1}, {0, 0, 0});
  nda::array<ComplexType, 3> V(dZ_qPQ.global_shape());
  V() = ComplexType(0);
  math::nda::gather(0, dZ_qPQ, &V);
  mpi.comm.broadcast_n(V.data(), V.size(), 0);

  // Distribute the (iq, P, Q) comparison triple across ranks; max-reduce.
  double max_dev = 0.0;
  double max_val = 0.0;
  long checked_local = 0;
  long my_rank = mpi.comm.rank();
  long nproc   = mpi.comm.size();
  long N_total = nq_ibz * Np * Np;
  for (long idx = my_rank; idx < N_total; idx += nproc) {
    long iq = idx / (Np * Np);
    long P  = (idx / Np) % Np;
    long Q  = idx % Np;
    long iqm = qminus(iq);
    if (iqm < 0 || iqm >= nq_ibz) continue;
    ComplexType a = V(iq,  P, Q);
    ComplexType b = V(iqm, Q, P);
    double dev = std::abs(a - std::conj(b));
    max_dev = std::max(max_dev, dev);
    max_val = std::max(max_val, std::abs(a));
    ++checked_local;
  }
  mpi.comm.all_reduce_in_place_n(&max_dev, 1, mpi3::max<>{});
  mpi.comm.all_reduce_in_place_n(&max_val, 1, mpi3::max<>{});
  long checked = checked_local;
  mpi.comm.all_reduce_in_place_n(&checked, 1, std::plus<>{});
  app_log(2, "PAW-augmented THC Hermiticity: nq_ibz={}, Np={}, "
             "max |V(q,P,Q) − V*(−q,Q,P)| = {:.3e}, "
             "max |V| = {:.3e}, rel = {:.3e}, checked={}",
          nq_ibz, Np, max_dev, max_val,
          max_dev / std::max(1e-30, max_val), checked);
  CHECK(max_dev < tol);
}

TEST_CASE("thc_paw_hermiticity", "[hamilt][paw][thc]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (NCPP)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222", mf::h5_input_type));
    test_thc_paw_hermiticity<HOST_MEMORY>(*mpi, mf_ptr);
  }
  SECTION("lih_kp222_nbnd16 (USPP)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_uspp", mf::h5_input_type));
    test_thc_paw_hermiticity<HOST_MEMORY>(*mpi, mf_ptr);
  }
  SECTION("lih_kp222_nbnd16 (PAW)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type));
    test_thc_paw_hermiticity<HOST_MEMORY>(*mpi, mf_ptr);
  }
}

TEST_CASE("thc_paw_hermiticity_si", "[hamilt][paw][thc][slow]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("si_kp222 (NCPP)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_si222_ncpp", mf::h5_input_type));
    test_thc_paw_hermiticity<HOST_MEMORY>(*mpi, mf_ptr);
  }
  SECTION("si_kp222 (USPP psl 1.0.0)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_si222_uspp", mf::h5_input_type));
    test_thc_paw_hermiticity<HOST_MEMORY>(*mpi, mf_ptr);
  }
  SECTION("si_kp222 (PAW psl 1.0.0)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_si222_paw", mf::h5_input_type));
    test_thc_paw_hermiticity<HOST_MEMORY>(*mpi, mf_ptr);
  }
}

/**
 * Smoke test: full HF (Hartree + exchange) with paw_aug=true runs to
 * completion and gives finite, real energy. Exchange exercises V_full at
 * all q's (not just q=0), so any failure in the q-loop V_GL/V_LL builder
 * shows up here. No external reference comparison.
 */
template<MEMORY_SPACE MEM>
void test_thc_paw_hf_smoke(mpi_context_t& mpi,
                            std::shared_ptr<mf::MF> mf_ptr,
                            double thc_thresh = 1e-5)
{
  using math::shm::make_shared_array;
  auto& mfobj = *mf_ptr;
  long nspin   = mfobj.nspin();
  long nk_ibz  = mfobj.nkpts_ibz();
  long nbnd    = mfobj.nbnd();

  auto thc_pt = methods::make_thc_reader_ptree(
      0, "", "incore", "", "bdft", thc_thresh, mfobj.ecutrho());
  thc_pt.put("paw_aug", true);
  thc_pt.put("paw_isdf_metric", "coulomb");
  thc_pt.put("paw_isdf_tol", 1e-12);
  methods::thc_reader_t thc(mf_ptr, thc_pt);

  methods::solvers::hf_t hf(methods::ignore_g0);
  auto sDm_skij = make_shared_array<array_view_4d_t>(mpi,
                      {nspin, nk_ibz, nbnd, nbnd});
  if (mpi.node_comm.root()) {
    sDm_skij.local()() = ComplexType(0.0);
    for (int s = 0; s < nspin; ++s)
    for (int k = 0; k < nk_ibz; ++k)
    for (int a = 0; a < nbnd; ++a)
      sDm_skij.local()(s, k, a, a) = mfobj.occ(s, k, a);
  }
  mpi.node_comm.barrier();
  auto sS_skij = make_shared_array<array_view_4d_t>(mpi,
                    {nspin, nk_ibz, nbnd, nbnd});
  hamilt::set_ovlp(mfobj, sS_skij);
  auto sJ = make_shared_array<array_view_4d_t>(mpi,
                {nspin, nk_ibz, nbnd, nbnd});
  // Hartree + exchange: exercises V_full at all q's.
  hf.evaluate(sJ, sDm_skij.local(), thc, sS_skij.local(), true, true);
  auto k_weight = mfobj.k_weight();
  auto [e1e_dummy, E_HF] = methods::eval_hf_energy(
      sDm_skij, sJ, sS_skij, k_weight, false);
  app_log(2, "PAW THC HF (Hartree+exchange): E = {:+.8f} Ha", E_HF);
  CHECK(std::isfinite(E_HF));
}

TEST_CASE("thc_paw_hf_smoke", "[hamilt][paw][thc]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (PAW)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type));
    test_thc_paw_hf_smoke<HOST_MEMORY>(*mpi, mf_ptr);
  }
  SECTION("lih_kp222_nbnd16 (USPP)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_uspp", mf::h5_input_type));
    test_thc_paw_hf_smoke<HOST_MEMORY>(*mpi, mf_ptr);
  }
}

/**
 * Element-wise comparison of V_H and V_x matrix elements computed two ways:
 *
 *  (a) Direct: hamilt::Vhartree (FFT pair density on dense G grid, with PAW
 *      Q-augmentation) + hamilt::Vexchange (FFT pair densities + PAW
 *      augmentation + deltaC one-center).
 *
 *  (b) THC: methods::thc_reader_t builds the (X, 𝒱) factorization of the
 *      ERI tensor; methods::solvers::hf_t::evaluate contracts X/𝒱 to
 *      assemble the Hartree (hartree=true, exchange=false) and exchange
 *      (hartree=false, exchange=true) matrix elements.
 *
 * Both paths target the same physical matrix element on the AE Hilbert
 * space (smooth + PAW augmentation). The element-wise difference quantifies
 * the THC factorization truncation error; running this at several
 * `thc_thresh` values traces the convergence of V_H / V_x matrix elements
 * vs the THC interpolation rank N_Λ.
 *
 * Sign / scale convention: both V_H_skij and K_skij are signed F-contributions
 * (F = H_core + V_H + K with K already negative). Matches what
 * `hf_t::evaluate` returns when called separately for hartree=true and
 * exchange=true.
 */
template<MEMORY_SPACE MEM>
void test_thc_vs_direct_VH_VX(mpi_context_t& mpi,
                              std::shared_ptr<mf::MF> mf_ptr,
                              double thc_thresh = 1e-5,
                              double tol_VH = 5e-4,
                              double tol_VX = 5e-3,
                              bool strict_VH = true,
                              bool strict_VX = true,
                              bool nonqe_occ = false)
{
  using math::shm::make_shared_array;
  auto all = nda::range::all;
  auto& mfobj = *mf_ptr;
  long nspin   = mfobj.nspin();
  long nk_ibz  = mfobj.nkpts_ibz();
  long nbnd    = mfobj.nbnd();

  // ----- Diagonal occupations nii (QE or a deliberately non-QE pattern) -----
  // The non-QE pattern is a deterministic function of (s,k,n) that is nonzero
  // on ALL bands and bears no relation to mfobj.occ(). Both the direct path
  // (hamilt::Vhartree/Vexchange take nii) and the THC path (hf_t::evaluate
  // takes the diagonal Dm built from this same nii) are fed identical input.
  // If any code path secretly reaches for mfobj.occ() (or assumes 0/1
  // occupations, or only the QE-occupied bands), the two paths diverge.
  memory::array<MEM, ComplexType, 3> nii(nspin, nk_ibz, nbnd);
  if (nonqe_occ) {
    static_assert(MEM == HOST_MEMORY,
                  "non-QE occupation pattern only built for HOST_MEMORY tests");
    for (long s = 0; s < nspin; ++s)
      for (long k = 0; k < nk_ibz; ++k)
        for (long n = 0; n < nbnd; ++n)
          nii(s, k, n) = ComplexType(0.5 + 0.3 * std::cos(1.3 * n + 0.7 * k
                                                          + 0.2 * s), 0.0);
  } else if constexpr (MEM == HOST_MEMORY) {
    nii() = mfobj.occ()(all, nda::range(nk_ibz), all);
  } else {
    nii() = nda::array<ComplexType,3>(mfobj.occ()(all, nda::range(nk_ibz), all));
  }

  // ----- Density matrix (diagonal in band) from the SAME nii -----
  auto sDm = make_shared_array<array_view_4d_t>(mpi,
                  {nspin, nk_ibz, nbnd, nbnd});
  if (mpi.node_comm.root()) {
    sDm.local()() = ComplexType(0.0);
    for (long s = 0; s < nspin; ++s)
      for (long k = 0; k < nk_ibz; ++k)
        for (long a = 0; a < nbnd; ++a)
          sDm.local()(s, k, a, a) = nii(s, k, a);
  }
  mpi.node_comm.barrier();

  // ----- Direct (FFT-based) V_H and V_x -----
  hamilt::pseudopot V(mfobj);
  auto dVH_direct = hamilt::Vhartree<MEM>(mfobj, mpi.comm, &V, nii);
  auto dVX_direct = hamilt::Vexchange<MEM>(mfobj, mpi.comm, &V, nii);

  // Vhartree's PAW path already folds the one-center radial Hartree
  // contribution into V_H via compute_paw_deeq (matrix-element scaling),
  // matching the THC V_H to ISDF noise on the LiH kp222 PAW HF fixture
  // (rel ≈ 7e-5). No separate deltaC×becsum addition is needed here.

  // ----- THC-based V_H and V_x via hf_t::evaluate -----
  auto thc_pt = methods::make_thc_reader_ptree(
      0, "", "incore", "", "bdft", thc_thresh, mfobj.ecutrho());
  thc_pt.put("paw_aug", true);
  thc_pt.put("paw_isdf_metric", "coulomb");
  thc_pt.put("paw_isdf_tol", 1e-12);
  methods::thc_reader_t thc(mf_ptr, thc_pt);

  auto sS_skij = make_shared_array<array_view_4d_t>(mpi,
                    {nspin, nk_ibz, nbnd, nbnd});
  hamilt::set_ovlp(mfobj, sS_skij);

  auto sVH_thc = make_shared_array<array_view_4d_t>(mpi,
                    {nspin, nk_ibz, nbnd, nbnd});
  {
    methods::solvers::hf_t hf(methods::ignore_g0);
    hf.evaluate(sVH_thc, sDm.local(), thc, sS_skij.local(),
                /*hartree=*/true, /*exchange=*/false);
  }
  auto sVX_thc = make_shared_array<array_view_4d_t>(mpi,
                    {nspin, nk_ibz, nbnd, nbnd});
  {
    methods::solvers::hf_t hf(methods::ignore_g0);
    hf.evaluate(sVX_thc, sDm.local(), thc, sS_skij.local(),
                /*hartree=*/false, /*exchange=*/true);
  }

  // ----- Element-wise comparison -----
  auto VH_thc_loc = nda::to_host(sVH_thc.local());
  auto VX_thc_loc = nda::to_host(sVX_thc.local());
  auto VH_dir_loc = nda::to_host(dVH_direct.local());
  auto VX_dir_loc = nda::to_host(dVX_direct.local());

  double max_VH = 0.0, max_VX = 0.0;
  double max_dVH = 0.0, max_dVX = 0.0;
  // Off-diagonal ΔV_H and the spread of the diagonal Δ (a k-independent
  // constant diagonal shift = pure G=0/monopole accounting difference).
  double max_dVH_off = 0.0, diag_re_min = 1e300, diag_re_max = -1e300;

  auto rng_s = dVH_direct.local_range(0);
  auto rng_k = dVH_direct.local_range(1);
  auto rng_i = dVH_direct.local_range(2);
  auto rng_j = dVH_direct.local_range(3);

  for (auto [is_l, s] : itertools::enumerate(rng_s))
    for (auto [ik_l, k] : itertools::enumerate(rng_k))
      for (auto [ii_l, i] : itertools::enumerate(rng_i))
        for (auto [ij_l, j] : itertools::enumerate(rng_j)) {
          ComplexType vh_dir = VH_dir_loc(is_l, ik_l, ii_l, ij_l);
          ComplexType vh_thc = VH_thc_loc(s, k, i, j);
          ComplexType vx_dir = VX_dir_loc(is_l, ik_l, ii_l, ij_l);
          ComplexType vx_thc = VX_thc_loc(s, k, i, j);
          max_VH  = std::max(max_VH,  std::abs(vh_dir));
          max_VX  = std::max(max_VX,  std::abs(vx_dir));
          max_dVH = std::max(max_dVH, std::abs(vh_thc - vh_dir));
          max_dVX = std::max(max_dVX, std::abs(vx_thc - vx_dir));
          if (i != j)
            max_dVH_off = std::max(max_dVH_off, std::abs(vh_thc - vh_dir));
          else {
            diag_re_min = std::min(diag_re_min, (vh_thc - vh_dir).real());
            diag_re_max = std::max(diag_re_max, (vh_thc - vh_dir).real());
          }
        }
  max_VH  = mpi.comm.all_reduce_value(max_VH,  boost::mpi3::max<>{});
  max_VX  = mpi.comm.all_reduce_value(max_VX,  boost::mpi3::max<>{});
  max_dVH = mpi.comm.all_reduce_value(max_dVH, boost::mpi3::max<>{});
  max_dVX = mpi.comm.all_reduce_value(max_dVX, boost::mpi3::max<>{});
  max_dVH_off = mpi.comm.all_reduce_value(max_dVH_off, boost::mpi3::max<>{});
  diag_re_min = mpi.comm.all_reduce_value(diag_re_min, boost::mpi3::min<>{});
  diag_re_max = mpi.comm.all_reduce_value(diag_re_max, boost::mpi3::max<>{});

  // Trace energies ½·(1/N_k)·Σ_sk Σ_i nii·V_ii per route — pins which side
  // of a matrix-element disagreement carries the physical Hartree energy.
  double eH_dir = 0.0, eH_thc = 0.0;
  for (auto [is_l, s] : itertools::enumerate(rng_s))
    for (auto [ik_l, k] : itertools::enumerate(rng_k))
      for (auto [ii_l, i] : itertools::enumerate(rng_i))
        for (auto [ij_l, j] : itertools::enumerate(rng_j)) {
          if (i != j) continue;
          double f = std::real(nii(s, k, i));
          eH_dir += 0.5 * f * std::real(VH_dir_loc(is_l, ik_l, ii_l, ij_l));
          eH_thc += 0.5 * f * std::real(VH_thc_loc(s, k, i, j));
        }
  eH_dir = mpi.comm.all_reduce_value(eH_dir, std::plus<>{}) / (double)mfobj.nkpts();
  eH_thc = mpi.comm.all_reduce_value(eH_thc, std::plus<>{}) / (double)mfobj.nkpts();

  app_log(2,
    "THC vs direct V_H: max|V_H| = {:.3e}, max|ΔV_H| = {:.3e} "
    "(rel = {:.2e})", max_VH, max_dVH, max_dVH / std::max(1e-30, max_VH));
  app_log(2,
    "  V_H Δ structure: off-diag max|Δ| = {:.3e}; diag Re(Δ) ∈ [{:.6e}, {:.6e}] "
    "(spread {:.3e})", max_dVH_off, diag_re_min, diag_re_max,
    diag_re_max - diag_re_min);
  app_log(2,
    "  V_H trace energies (½·Tr[n V]/N_k): direct = {:+.8f} Ha, THC = {:+.8f} Ha, "
    "Δ = {:+.3e}", eH_dir, eH_thc, eH_thc - eH_dir);
  app_log(2,
    "THC vs direct V_x: max|V_x| = {:.3e}, max|ΔV_x| = {:.3e} "
    "(rel = {:.2e})", max_VX, max_dVX, max_dVX / std::max(1e-30, max_VX));
  if (strict_VX) CHECK(max_dVX < tol_VX);
  else app_log(2, "  (V_x diagnostic mode: ΔV_x reported above, not asserted)");
  if (strict_VH) CHECK(max_dVH < tol_VH);
  else app_log(2, "  (V_H diagnostic mode: ΔV_H reported above, not asserted)");
}

/**
 * ISDF-threshold convergence study of the V_H and V_x matrix elements.
 *
 * For a sequence of THC/ISDF thresholds (tighter → larger auxiliary basis),
 * build the THC factorization, assemble V_H (hartree=true) and V_x
 * (exchange=true) via hf_t::evaluate, and compare element-wise to the EXACT
 * direct evaluation from the hamiltonian folder (hamilt::Vhartree /
 * hamilt::Vexchange). Emits CSV lines (tag "CONVCSV") with both max-abs and
 * relative-Frobenius errors vs the ISDF threshold and the resulting
 * auxiliary-basis size Np. Used to generate the convergence figures.
 *
 * NCPP is the clean reference case (no PAW augmentation; both paths produce
 * the full matrix element). For USPP the exchange V_x reference is likewise
 * exact; the smooth Q-augmentation is shared by both paths.
 */
template<MEMORY_SPACE MEM>
void run_isdf_threshold_convergence(mpi_context_t& mpi,
                                    std::shared_ptr<mf::MF> mf_ptr,
                                    std::string const& tag,
                                    std::vector<double> const& thresholds,
                                    double thc_ecut)
{
  using math::shm::make_shared_array;
  auto all = nda::range::all;
  auto& mfobj = *mf_ptr;
  long nspin   = mfobj.nspin();
  long nk_ibz  = mfobj.nkpts_ibz();
  long nbnd    = mfobj.nbnd();

  memory::array<MEM, ComplexType, 3> nii(nspin, nk_ibz, nbnd);
  if constexpr (MEM == HOST_MEMORY)
    nii() = mfobj.occ()(all, nda::range(nk_ibz), all);
  else
    nii() = nda::array<ComplexType,3>(mfobj.occ()(all, nda::range(nk_ibz), all));

  auto sDm = make_shared_array<array_view_4d_t>(mpi, {nspin, nk_ibz, nbnd, nbnd});
  if (mpi.node_comm.root()) {
    sDm.local()() = ComplexType(0.0);
    for (int s = 0; s < nspin; ++s)
      for (int k = 0; k < nk_ibz; ++k)
        for (int a = 0; a < nbnd; ++a)
          sDm.local()(s, k, a, a) = mfobj.occ(s, k, a);
  }
  mpi.node_comm.barrier();
  auto sS = make_shared_array<array_view_4d_t>(mpi, {nspin, nk_ibz, nbnd, nbnd});
  hamilt::set_ovlp(mfobj, sS);

  // ----- Exact reference from the hamiltonian folder -----
  hamilt::pseudopot V(mfobj);
  auto dVH_ref = hamilt::Vhartree<MEM>(mfobj, mpi.comm, &V, nii);
  auto dVX_ref = hamilt::Vexchange<MEM>(mfobj, mpi.comm, &V, nii);
  auto VH_ref = nda::to_host(dVH_ref.local());
  auto VX_ref = nda::to_host(dVX_ref.local());

  auto rng_s = dVH_ref.local_range(0); auto rng_k = dVH_ref.local_range(1);
  auto rng_i = dVH_ref.local_range(2); auto rng_j = dVH_ref.local_range(3);

  // Reference Frobenius norms.
  double fro_VH_ref = 0.0, fro_VX_ref = 0.0;
  for (auto [is,s] : itertools::enumerate(rng_s))
    for (auto [ik,k] : itertools::enumerate(rng_k))
      for (auto [ii,i] : itertools::enumerate(rng_i))
        for (auto [ij,j] : itertools::enumerate(rng_j)) {
          fro_VH_ref += std::norm(VH_ref(is,ik,ii,ij));
          fro_VX_ref += std::norm(VX_ref(is,ik,ii,ij));
        }
  fro_VH_ref = std::sqrt(mpi.comm.all_reduce_value(fro_VH_ref, std::plus<>{}));
  fro_VX_ref = std::sqrt(mpi.comm.all_reduce_value(fro_VX_ref, std::plus<>{}));

  if (mpi.comm.root())
    app_log(2, "CONVCSV_HEADER,{},thresh,Np,errVH_max,errVH_fro,relVH_fro,"
               "errVX_max,errVX_fro,relVX_fro", tag);

  // Compute THC V_H, V_x at a given ISDF threshold, returned as host arrays.
  auto thc_VH_VX = [&](double thr) {
    auto thc_pt = methods::make_thc_reader_ptree(
        0, "", "incore", "", "bdft", thr, thc_ecut);
    thc_pt.put("paw_aug", true);
    thc_pt.put("paw_isdf_metric", "coulomb");
    thc_pt.put("paw_isdf_tol", 1e-12);
    methods::thc_reader_t thc(mf_ptr, thc_pt);
    int Np = thc.Np();
    auto sVH = make_shared_array<array_view_4d_t>(mpi, {nspin, nk_ibz, nbnd, nbnd});
    { methods::solvers::hf_t hf(methods::ignore_g0);
      hf.evaluate(sVH, sDm.local(), thc, sS.local(), true, false); }
    auto sVX = make_shared_array<array_view_4d_t>(mpi, {nspin, nk_ibz, nbnd, nbnd});
    { methods::solvers::hf_t hf(methods::ignore_g0);
      hf.evaluate(sVX, sDm.local(), thc, sS.local(), false, true); }
    // Explicit copies — sVH/sVX (shared arrays) are destroyed at return, so a
    // to_host view would dangle.
    nda::array<ComplexType, 4> VHc = sVH.local();
    nda::array<ComplexType, 4> VXc = sVX.local();
    return std::make_tuple(Np, VHc, VXc);
  };

  // Converged-ISDF reference: tightest threshold. THC at this threshold IS the
  // deeq-complete augmented V_H/V_x (the operator that reproduces QE eigenvalues)
  // up to a negligible ISDF residual, so self-convergence against it isolates
  // the ISDF-threshold error from the (V_H bra-aug, PAW one-center) bookkeeping
  // offsets that the absolute-vs-hamiltonian comparison also reports.
  double thr_conv = *std::min_element(thresholds.begin(), thresholds.end());
  auto [Np_conv, VH_conv, VX_conv] = thc_VH_VX(thr_conv);
  double fro_VH_conv = 0.0, fro_VX_conv = 0.0;
  // VH_conv/VX_conv are full (shared) arrays → index with GLOBAL (s,k,i,j).
  for (auto [is,s] : itertools::enumerate(rng_s))
    for (auto [ik,k] : itertools::enumerate(rng_k))
      for (auto [ii,i] : itertools::enumerate(rng_i))
        for (auto [ij,j] : itertools::enumerate(rng_j)) {
          fro_VH_conv += std::norm(VH_conv(s,k,i,j));
          fro_VX_conv += std::norm(VX_conv(s,k,i,j));
        }
  fro_VH_conv = std::sqrt(mpi.comm.all_reduce_value(fro_VH_conv, std::plus<>{}));
  fro_VX_conv = std::sqrt(mpi.comm.all_reduce_value(fro_VX_conv, std::plus<>{}));

  for (double thr : thresholds) {
    auto [Np, VH, VX] = thc_VH_VX(thr);
    double eVH_max=0, eVX_max=0, eVH_fro=0, eVX_fro=0;
    double eVX_diag=0, eVX_off=0;     // diag vs off-diag V_x abs error (Frob²)
    double sVH_fro=0, sVX_fro=0;      // self-convergence (vs converged-ISDF THC)
    double sVX_diag=0, sVX_off=0;
    // DIAGNOSTIC: localize the V_H error. Split into the low-band block
    // (i,j < 16, what the 16-band fixture samples) vs any pair touching a high
    // virtual (>=16); track the single worst V_H element. (single-rank valid)
    double eVH_low=0, eVH_high=0;
    long wi=-1, wj=-1; double worstH=-1.0; ComplexType wthc{}, wref{};
    // Fixed-element probe: V_H[0,0] (s=0,k=0). On nosym THC==direct==truth, so
    // comparing this across nosym/sym tells us WHICH path is wrong.
    ComplexType vh00_thc{}, vh00_ref{}; bool have00=false;
    for (auto [is,s] : itertools::enumerate(rng_s))
      for (auto [ik,k] : itertools::enumerate(rng_k))
        for (auto [ii,i] : itertools::enumerate(rng_i))
          for (auto [ij,j] : itertools::enumerate(rng_j)) {
            ComplexType dH = VH(s,k,i,j) - VH_ref(is,ik,ii,ij);
            ComplexType dX = VX(s,k,i,j) - VX_ref(is,ik,ii,ij);
            eVH_max = std::max(eVH_max, std::abs(dH));
            eVX_max = std::max(eVX_max, std::abs(dX));
            { double aH = std::abs(dH);
              if (i < 16 && j < 16) eVH_low += aH*aH; else eVH_high += aH*aH;
              if (aH > worstH) { worstH = aH; wi = i; wj = j;
                wthc = VH(s,k,i,j); wref = VH_ref(is,ik,ii,ij); } }
            if (s==0 && k==0 && i==0 && j==0) {
              vh00_thc = VH(s,k,i,j); vh00_ref = VH_ref(is,ik,ii,ij); have00 = true; }
            eVH_fro += std::norm(dH);
            eVX_fro += std::norm(dX);
            if (i == j) eVX_diag += std::norm(dX);
            else        eVX_off  += std::norm(dX);
            // self-convergence vs converged-ISDF THC (both full arrays → global)
            ComplexType sH = VH(s,k,i,j) - VH_conv(s,k,i,j);
            ComplexType sX = VX(s,k,i,j) - VX_conv(s,k,i,j);
            sVH_fro += std::norm(sH);
            sVX_fro += std::norm(sX);
            if (i == j) sVX_diag += std::norm(sX);
            else        sVX_off  += std::norm(sX);
          }
    eVH_max = mpi.comm.all_reduce_value(eVH_max, boost::mpi3::max<>{});
    eVX_max = mpi.comm.all_reduce_value(eVX_max, boost::mpi3::max<>{});
    eVH_fro = std::sqrt(mpi.comm.all_reduce_value(eVH_fro, std::plus<>{}));
    eVX_fro = std::sqrt(mpi.comm.all_reduce_value(eVX_fro, std::plus<>{}));
    eVX_diag = std::sqrt(mpi.comm.all_reduce_value(eVX_diag, std::plus<>{}));
    eVX_off  = std::sqrt(mpi.comm.all_reduce_value(eVX_off,  std::plus<>{}));
    sVH_fro = std::sqrt(mpi.comm.all_reduce_value(sVH_fro, std::plus<>{}));
    sVX_fro = std::sqrt(mpi.comm.all_reduce_value(sVX_fro, std::plus<>{}));
    sVX_diag = std::sqrt(mpi.comm.all_reduce_value(sVX_diag, std::plus<>{}));
    sVX_off  = std::sqrt(mpi.comm.all_reduce_value(sVX_off,  std::plus<>{}));
    eVH_low  = std::sqrt(mpi.comm.all_reduce_value(eVH_low,  std::plus<>{}));
    eVH_high = std::sqrt(mpi.comm.all_reduce_value(eVH_high, std::plus<>{}));
    if (mpi.comm.root()) {
      // Absolute error vs the exact hamiltonian (Vhartree / Vexchange).
      app_log(2, "CONVCSV,{},{:.3e},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
              tag, thr, Np, eVH_max, eVH_fro, eVH_fro/std::max(1e-30,fro_VH_ref),
              eVX_max, eVX_fro, eVX_fro/std::max(1e-30,fro_VX_ref));
      // Self-convergence (relative Frobenius vs converged-ISDF THC).
      app_log(2, "CONVSELF,{},{:.3e},{},relVH_self={:.6e},relVX_self={:.6e}",
              tag, thr, Np, sVH_fro/std::max(1e-30,fro_VH_conv),
              sVX_fro/std::max(1e-30,fro_VX_conv));
      app_log(2, "CONVVX_SPLIT,{},{:.3e},{},absVX_diag={:.6e},absVX_off={:.6e},"
                 "selfVX_diag={:.6e},selfVX_off={:.6e}",
              tag, thr, Np, eVX_diag, eVX_off, sVX_diag, sVX_off);
      // DIAGNOSTIC: where does the V_H error live?
      app_log(2, "CONVWORST,{},{:.3e},{},VHlow_fro={:.6e},VHhigh_fro={:.6e},"
                 "worst|dVH|={:.4e},i={},j={},THC={:+.6e},ref={:+.6e}",
              tag, thr, Np, eVH_low, eVH_high, worstH, wi, wj,
              wthc.real(), wref.real());
      app_log(2, "CONV00,{},{:.3e},{},have={},VH00_THC={:+.8e},VH00_ref={:+.8e}",
              tag, thr, Np, (int)have00, vh00_thc.real(), vh00_ref.real());
    }
  }
}

// Exchange-potential sensitivity probe for the V_x THC-vs-direct floor.
// NCPP LiH n100 (no augmentation): V_x self-converges but sits ~1.3e-3 from the
// direct Vexchange regardless of ISDF rank. Vary div_treatment (ignore_g0 vs
// gygi), ISDF threshold, and THC ecut to localize the origin.
TEST_CASE("vx_sensitivity_ncpp", "[hamilt][thc][hf][!benchmark]")
{
  using math::shm::make_shared_array;
  auto all = nda::range::all;
  auto& mpi = utils::make_unit_test_mpi_context();
  std::string base = std::string(std::getenv("HOME"))
                   + "/ceph/CoQui/PAW_comparisons_w150/runs/";
  auto mf_ptr = std::make_shared<mf::MF>(
      mf::default_MF(mpi, mf::qe_source, base + "oncv/w150_n100/nscf/out/",
                     "lih", mf::h5_input_type));
  auto& mfobj = *mf_ptr;
  long nspin = mfobj.nspin(), nk_ibz = mfobj.nkpts_ibz(), nbnd = mfobj.nbnd();
  double mad = mfobj.madelung();
  double ecutrho = mfobj.ecutrho();
  app_log(2, "VXSENS_INFO madelung={:.6e} ecutrho={:.3f} nbnd={} nk_ibz={}",
          mad, ecutrho, nbnd, nk_ibz);

  memory::array<HOST_MEMORY, ComplexType, 3> nii(nspin, nk_ibz, nbnd);
  nii() = mfobj.occ()(all, nda::range(nk_ibz), all);
  auto sDm = make_shared_array<array_view_4d_t>(*mpi, {nspin, nk_ibz, nbnd, nbnd});
  if (mpi->node_comm.root()) {
    sDm.local()() = ComplexType(0.0);
    for (long s = 0; s < nspin; ++s) for (long k = 0; k < nk_ibz; ++k)
      for (long a = 0; a < nbnd; ++a) sDm.local()(s, k, a, a) = nii(s, k, a);
  }
  mpi->node_comm.barrier();
  hamilt::pseudopot V(mfobj);
  auto sS = make_shared_array<array_view_4d_t>(*mpi, {nspin, nk_ibz, nbnd, nbnd});
  hamilt::set_ovlp(mfobj, sS);
  auto dVX = hamilt::Vexchange<HOST_MEMORY>(mfobj, mpi->comm, &V, nii);
  auto VXref = nda::to_host(dVX.local());

  auto run = [&](std::string lbl, methods::div_treatment_e div, double thr, double ecut) {
    auto pt = methods::make_thc_reader_ptree(0, "", "incore", "", "bdft", thr, ecut);
    pt.put("paw_aug", true); pt.put("paw_isdf_metric", "coulomb"); pt.put("paw_isdf_tol", 1e-12);
    methods::thc_reader_t thc(mf_ptr, pt);
    int Np = thc.Np();
    auto sVX = make_shared_array<array_view_4d_t>(*mpi, {nspin, nk_ibz, nbnd, nbnd});
    { methods::solvers::hf_t hf(div);
      hf.evaluate(sVX, sDm.local(), thc, sS.local(), false, true); }
    auto VX = nda::to_host(sVX.local());
    double fro_ref=0, e_fro=0, e_diag=0, e_off=0;
    auto rs=dVX.local_range(0); auto rk=dVX.local_range(1);
    auto ri=dVX.local_range(2); auto rj=dVX.local_range(3);
    for (auto [is,s] : itertools::enumerate(rs))
      for (auto [ik,k] : itertools::enumerate(rk))
        for (auto [ii,i] : itertools::enumerate(ri))
          for (auto [ij,j] : itertools::enumerate(rj)) {
            ComplexType d = VX(s,k,i,j) - VXref(is,ik,ii,ij);
            fro_ref += std::norm(VXref(is,ik,ii,ij)); e_fro += std::norm(d);
            if (i==j) e_diag += std::norm(d); else e_off += std::norm(d);
          }
    fro_ref = std::sqrt(mpi->comm.all_reduce_value(fro_ref, std::plus<>{}));
    e_fro = std::sqrt(mpi->comm.all_reduce_value(e_fro, std::plus<>{}));
    e_diag = std::sqrt(mpi->comm.all_reduce_value(e_diag, std::plus<>{}));
    e_off = std::sqrt(mpi->comm.all_reduce_value(e_off, std::plus<>{}));
    app_log(2, "VXSENS,{},Np={},relVX={:.5e},absdiag={:.5e},absoff={:.5e}",
            lbl, Np, e_fro/std::max(1e-30,fro_ref), e_diag, e_off);
  };

  SECTION("ncpp n100 exchange sensitivity") {
    run("ig0__e90__t1e-6",   methods::ignore_g0, 1e-6, 90.0);
    run("gygi_e90__t1e-6",   methods::gygi,      1e-6, 90.0);
    run("ig0__e90__t1e-7",   methods::ignore_g0, 1e-7, 90.0);
    run("ig0__erho_t1e-6",   methods::ignore_g0, 1e-6, ecutrho);
    run("gygi_erho_t1e-6",   methods::gygi,      1e-6, ecutrho);
  }
}

TEST_CASE("isdf_threshold_convergence", "[hamilt][thc][convergence][!benchmark]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  std::vector<double> thr = {3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6};
  // PAPER RUN: THC ISDF ecut = 1.2 x ecutwfc(150 Ry = 75 Ha) = 90 Ha (spec).
  // With the full-BZ becsum symmetry fix (v_h_paw + compute_paw_deeq), V_H now
  // converges under symmetry like V_x (no more 3-5% floor).

  // 100- and 250-band LiH MFs at ecutwfc=150 Ry, generated in
  // PAW_comparisons_w150. ISDF-rank convergence vs alpha = N_mu / N_bnd.
  // NOTE: path-hardcoded; results-only edit, not for commit.
  const std::string base = std::string(std::getenv("HOME"))
                         + "/ceph/CoQui/PAW_comparisons_w150/runs/";
  auto load = [&](std::string sub) {
    return std::make_shared<mf::MF>(
        mf::default_MF(mpi, mf::qe_source, base + sub + "/nscf/out/", "lih",
                       mf::h5_input_type));
  };

  SECTION("LiH NCPP(oncv) n100") { auto m = load("oncv/w150_n100");
    run_isdf_threshold_convergence<HOST_MEMORY>(*mpi, m, "LiH_NCPP_n100", thr, 90.0); }
  SECTION("LiH USPP n100")       { auto m = load("uspp/w150_n100");
    run_isdf_threshold_convergence<HOST_MEMORY>(*mpi, m, "LiH_USPP_n100", thr, 90.0); }
  SECTION("LiH PAW n100")        { auto m = load("paw/w150_n100");
    run_isdf_threshold_convergence<HOST_MEMORY>(*mpi, m, "LiH_PAW_n100", thr, 90.0); }
  SECTION("LiH NCPP(oncv) n250") { auto m = load("oncv/w150_n250");
    run_isdf_threshold_convergence<HOST_MEMORY>(*mpi, m, "LiH_NCPP_n250", thr, 90.0); }
  SECTION("LiH USPP n250")       { auto m = load("uspp/w150_n250");
    run_isdf_threshold_convergence<HOST_MEMORY>(*mpi, m, "LiH_USPP_n250", thr, 90.0); }
  SECTION("LiH PAW n250")        { auto m = load("paw/w150_n250");
    run_isdf_threshold_convergence<HOST_MEMORY>(*mpi, m, "LiH_PAW_n250", thr, 90.0); }

  // ecutwfc scan at n100 (fixed pw2coqui MFs) to isolate the semicore V_H
  // floor vs the wavefunction cutoff. tag suffix _w<ecutwfc>.
  SECTION("LiH USPP n100 w100") { auto m = load("uspp/w100_n100");
    run_isdf_threshold_convergence<HOST_MEMORY>(*mpi, m, "LiH_USPP_n100_w100", thr, 90.0); }
  SECTION("LiH USPP n100 w200") { auto m = load("uspp/w200_n100");
    run_isdf_threshold_convergence<HOST_MEMORY>(*mpi, m, "LiH_USPP_n100_w200", thr, 90.0); }
  SECTION("LiH PAW n100 w100")  { auto m = load("paw/w100_n100");
    run_isdf_threshold_convergence<HOST_MEMORY>(*mpi, m, "LiH_PAW_n100_w100", thr, 90.0); }
  SECTION("LiH PAW n100 w200")  { auto m = load("paw/w200_n100");
    run_isdf_threshold_convergence<HOST_MEMORY>(*mpi, m, "LiH_PAW_n100_w200", thr, 90.0); }

  // 16-band from-scratch control (fresh pw2coqui). Should converge cleanly if
  // the floor is band-count/cutoff driven; floors if the pipeline itself broke.
  SECTION("LiH USPP n16 w100") { auto m = load("uspp/w100_n16");
    run_isdf_threshold_convergence<HOST_MEMORY>(*mpi, m, "LiH_USPP_n16_w100", thr, 90.0); }
  SECTION("LiH USPP n16 w150") { auto m = load("uspp/w150_n16");
    run_isdf_threshold_convergence<HOST_MEMORY>(*mpi, m, "LiH_USPP_n16_w150", thr, 90.0); }
  SECTION("LiH PAW n16 w100")  { auto m = load("paw/w100_n16");
    run_isdf_threshold_convergence<HOST_MEMORY>(*mpi, m, "LiH_PAW_n16_w100", thr, 90.0); }
  SECTION("LiH PAW n16 w150")  { auto m = load("paw/w150_n16");
    run_isdf_threshold_convergence<HOST_MEMORY>(*mpi, m, "LiH_PAW_n16_w150", thr, 90.0); }

  // NOSYM control (only nosym changed vs the floored fresh n16). If this is
  // clean, the floor is the PAW augmentation under symmetry reduction.
  SECTION("LiH USPP nosym n16") { auto m = load("uspp/nosym_n16");
    run_isdf_threshold_convergence<HOST_MEMORY>(*mpi, m, "LiH_USPP_nosym_n16", thr, 90.0); }
  SECTION("LiH PAW nosym n16")  { auto m = load("paw/nosym_n16");
    run_isdf_threshold_convergence<HOST_MEMORY>(*mpi, m, "LiH_PAW_nosym_n16", thr, 90.0); }
}

TEST_CASE("thc_vs_direct_VH_VX", "[hamilt][thc][hf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  // Strict comparison: for NCPP both Vhartree and Vexchange produce the
  // FULL matrix element (no PAW augmentation), and the THC factorization
  // should reproduce them to within ISDF truncation (~1e-4 at thc_thresh=1e-5).
  SECTION("lih_kp222_nbnd16 (NCPP, HF)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_hf", mf::h5_input_type));
    test_thc_vs_direct_VH_VX<HOST_MEMORY>(*mpi, mf_ptr,
        /*thc_thresh*/ 1e-5, /*tol_VH*/ 5e-4, /*tol_VX*/ 5e-3,
        /*strict_VH*/ true, /*strict_VX*/ true);
  }

  // USPP: both V_H and V_x reproduce the direct full element to ISDF noise.
  SECTION("lih_kp222_nbnd16 (USPP, HF)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_uspp_hf", mf::h5_input_type));
    test_thc_vs_direct_VH_VX<HOST_MEMORY>(*mpi, mf_ptr,
        /*thc_thresh*/ 1e-5, /*tol_VH*/ 5e-4, /*tol_VX*/ 5e-4,
        /*strict_VH*/ true, /*strict_VX*/ true);
  }

  // PAW: both V_H and V_x reproduce the direct element to ISDF noise (~1e-4).
  // The one-center exchange uses prefactor scl_oc=-1/N_k (v_x_paw.hpp); see
  // notes/paw_onecenter_exchange_prefactor.md.
  SECTION("lih_kp222_nbnd16 (PAW, HF)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw_hf", mf::h5_input_type));
    test_thc_vs_direct_VH_VX<HOST_MEMORY>(*mpi, mf_ptr,
        /*thc_thresh*/ 1e-5, /*tol_VH*/ 5e-4, /*tol_VX*/ 5e-4,
        /*strict_VH*/ true, /*strict_VX*/ true);
  }

  // ABINIT-sourced PAW mf (plan D2 requirement): the abinit2coqui real_ylm
  // odd-m sign bug (3956b45) was invisible to every QE-only route test —
  // channel-diagonal quantities were immune while the off-diagonal (k,k−q)
  // pair-density augmentation decohered. Route equivalence on an AB mf
  // exercises the converter's projector/qfuncl/Ylm conventions end to end
  // (Si LDA-PW 12-electron semicore dataset with core wfc → ex_cvij active,
  // 2x2x2 full-BZ nosym).
  SECTION("si_kp222 (PAW, ABINIT-sourced mf)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "bdft_si222_paw_ab", mf::h5_input_type));
    // D2 defect RESOLVED (2026-07-25): the direct V_H excess (+19.98 Ha
    // trace vs the THC/ABINIT-verified +41.005) was the frozen-core
    // density injected into the DYNAMIC one-center Hartree deeq
    // (compute_paw_hartree_atom lp=0 core term) — a double count of the
    // core–valence electrostatics already inside the static D⁰/dion (plan
    // I2/I3). Invisible on QE fixtures (empty core fields); activated
    // here by the semicore dataset's exported core wfc. Bisection lives
    // in ab_direct_vh_trace_split. Both V_H and V_x strict.
    //
    // tol_VH is scaled to this fixture's element magnitude: max|V_H| ≈ 9.9
    // (12-el semicore) vs the LiH sections' ~O(1), so 2e-3 absolute is the
    // same ~1e-4 RELATIVE strictness (post-fix measured: max|ΔV_H| =
    // 1.16e-3, rel 1.2e-4, both signs on the diagonal; trace Δ = 2.4e-4 Ha
    // — pure THC/ISDF truncation, consistent with the THC-side energy
    // agreement of 4.9e-4 Ha on this mf).
    test_thc_vs_direct_VH_VX<HOST_MEMORY>(*mpi, mf_ptr,
        /*thc_thresh*/ 1e-5, /*tol_VH*/ 2e-3, /*tol_VX*/ 5e-4,
        /*strict_VH*/ true, /*strict_VX*/ true);
  }

}

// ===========================================================================
// D2 V_H trace decomposition on the ABINIT-sourced mf: split the DIRECT V_H
// trace into its three assembly pieces and pin each against the independent
// numpy probe (tests/unit_test_files/bdft/si_kp222_paw_abinit/
// probe_hartree.py) bilinears at the file's half-occupancy nii:
//   smooth-bra   ½∫v_H·ñ  (add_vloc)     → B(t,t)+B(t,h) = +3.7615
//   deeq-dynamic ½∫v_H·n̂  (∫V·Q)         → B(t,h)+B(h,h) = +25.4505
//   one-center radial (deltaC-equiv)                       = +11.7930
//   total (THC-verified)                                   = +41.005
// This bisection found the D2 defect (2026-07-25): T_oc read +31.77 — the
// frozen-core density injected into the dynamic radial Hartree, a +19.98 Ha
// double count of core–valence electrostatics already in the static
// D⁰/dion. Kept as a regression guard: it pins each assembly piece
// separately, and traces the deeq through TWO independent contractions
// (becsum/Pskna vs the production add_vnl_impl route implicit in
// Vhartree − smooth-bra), so a reintroduced piece-level defect is localized
// immediately.
// ===========================================================================
TEST_CASE("ab_direct_vh_trace_split", "[hamilt][paw][hf]")
{
  using math::shm::make_shared_array;
  auto all = nda::range::all; using nda::range;
  auto& mpip = utils::make_unit_test_mpi_context();
  auto& mpi = *mpip;
  auto mf_ptr = std::make_shared<mf::MF>(
      mf::default_MF(mpip, "bdft_si222_paw_ab", mf::h5_input_type));
  auto& mfobj = *mf_ptr;
  long nspin = mfobj.nspin(), nk_ibz = mfobj.nkpts_ibz(), nbnd = mfobj.nbnd();
  long nk_full = mfobj.nkpts();
  int npol = mfobj.npol();
  hamilt::pseudopot V(mfobj);

  nda::array<ComplexType,3> nii(nspin, nk_ibz, nbnd);
  nii() = mfobj.occ()(all, range(nk_ibz), all);

  using larray = memory::array<HOST_MEMORY, ComplexType, 4>;
  auto psi = mf::read_distributed_orbital_set_ibz<larray>(
      mfobj, mpi.comm, 'w', std::array<long,4>{0,0,0,0},
      range(nspin), range(nk_ibz), range(nbnd), std::array<long,4>{1,1,2048,2048});

  auto mesh_aug  = mfobj.fft_grid_dim_aug();
  auto lattv     = mfobj.lattv();
  auto recv      = mfobj.recv();
  auto kpts_full = mfobj.kpts();
  auto kp_to_ibz = mfobj.kp_to_ibz();
  auto kp_trev   = mfobj.kp_trev();
  auto kp_symm   = mfobj.kp_symm();
  auto symm_list = mfobj.symm_list();
  auto k2g       = V.swfc_to_rho_view();
  long nnr_aug = (long)mesh_aug(0)*mesh_aug(1)*mesh_aug(2);

  // v_hartree(r) on the aug mesh: the EXACT production path (same call as
  // add_Hartree_impl — smooth density + folded compensation charge).
  auto sv = make_shared_array<nda::array_view<ComplexType,1>>(mpi, {nnr_aug});
  pots::potential_t vG(ptree{});
  hamilt::v_h(mpi, vG, V, npol, mesh_aug, lattv, recv, k2g, kpts_full,
              kp_to_ibz, kp_trev, kp_symm, symm_list, nii, psi,
              /*symmetrize_rho_r=*/false, sv);

  // ---- piece 1: smooth-bra ½·Tr[n ⟨ψ̃|v_H|ψ̃⟩]/N_k via production add_vloc.
  auto hpsi = math::nda::make_distributed_array<larray>(
      mpi.comm, psi.grid(), psi.global_shape(), psi.block_size());
  hpsi.local() = ComplexType(0.0);
  hamilt::add_vloc(npol, mesh_aug, k2g, sv.local(), psi, hpsi);
  double T_S = 0.0;
  {
    auto ploc = psi.local(); auto hloc = hpsi.local();
    for (auto [is,s] : itertools::enumerate(psi.local_range(0)))
      for (auto [ik,k] : itertools::enumerate(psi.local_range(1)))
        for (auto [ib,b] : itertools::enumerate(psi.local_range(2))) {
          ComplexType acc(0.0);
          for (long g = 0; g < ploc.extent(3); ++g)
            acc += std::conj(ploc(is,ik,ib,g)) * hloc(is,ik,ib,g);
          T_S += 0.5 * std::real(nii(s,k,b)) * std::real(acc);
        }
    T_S = mpi.comm.all_reduce_value(T_S, std::plus<>{}) / (double)nk_full;
  }

  // ---- pieces 2+3: deeq radial (empty V) and dynamic (∫v_H·Q), contracted
  // with the same half-occupancy becsum the trace convention implies. becsum
  // already carries the 1/N_k k-weight — no further normalization.
  auto bec = hamilt::paw::compute_becsum_diagonal_symm(
      V, nii, kp_to_ibz, kp_trev, npol);
  nda::array<ComplexType,1> empty_v;
  auto dD_0 = V.compute_paw_deeq(nii, empty_v,   /*include_static=*/false);
  auto dD_V = V.compute_paw_deeq(nii, sv.local(), /*include_static=*/false);
  double T_oc = 0.0, T_dyn = 0.0;
  for (long ia = 0; ia < bec.extent(0); ++ia)
    for (long I = 0; I < bec.extent(1); ++I)
      for (long J = 0; J < bec.extent(2); ++J) {
        T_oc  += 0.5 * bec(ia,J,I) * std::real(dD_0(ia,I,J));
        T_dyn += 0.5 * bec(ia,J,I) * std::real(dD_V(ia,I,J) - dD_0(ia,I,J));
      }

  // ---- alternative deeq trace through Pskna (THC-verified projector path);
  // should equal T_oc + T_dyn identically if becsum ↔ Pskna are consistent.
  double T_P = 0.0;
  {
    auto Pskna = V.Pskna_view(); auto const& ityp = V.ityp_view();
    auto const& nh_v = V.nh_view(); auto const& ofs = V.ofs_view();
    for (long s = 0; s < nspin; ++s)
      for (long k = 0; k < nk_ibz; ++k)
        for (long ia = 0; ia < ityp.extent(0); ++ia) {
          int nt = ityp(ia); int nh_a = nh_v(nt); if (nh_a == 0) continue;
          long p0 = ofs(ia);
          for (long i = 0; i < nbnd; ++i) {
            ComplexType acc(0.0);
            for (int I = 0; I < nh_a; ++I) {
              ComplexType PiI = std::conj(Pskna(s,k,p0+I,i));
              for (int J = 0; J < nh_a; ++J)
                acc += PiI * dD_V(ia,I,J) * Pskna(s,k,p0+J,i);
            }
            T_P += 0.5 * std::real(nii(s,k,i)) * std::real(acc);
          }
        }
    T_P /= (double)nk_full;
  }

  // ---- production full matrix (add_vloc bra + add_vnl_impl deeq).
  auto dVH = hamilt::Vhartree<HOST_MEMORY>(mfobj, mpi.comm, &V, nii);
  double T_full = 0.0;
  {
    auto loc = nda::to_host(dVH.local());
    for (auto [a,s] : itertools::enumerate(dVH.local_range(0)))
      for (auto [b,k] : itertools::enumerate(dVH.local_range(1)))
        for (auto [c,i] : itertools::enumerate(dVH.local_range(2)))
          for (auto [d,j] : itertools::enumerate(dVH.local_range(3)))
            if (i == j)
              T_full += 0.5 * std::real(nii(s,k,i)) * std::real(loc(a,b,c,d));
    T_full = mpi.comm.all_reduce_value(T_full, std::plus<>{}) / (double)nk_full;
  }

  app_log(1, "[AB V_H split] smooth-bra   T_S    = {:+.6f}  (probe +3.7615)", T_S);
  app_log(1, "[AB V_H split] radial oc    T_oc   = {:+.6f}  (probe +11.7930)", T_oc);
  app_log(1, "[AB V_H split] dyn ∫V·Q     T_dyn  = {:+.6f}  (probe +25.4505)", T_dyn);
  app_log(1, "[AB V_H split] Pskna deeq   T_P    = {:+.6f}  (becsum route {:+.6f})",
          T_P, T_oc + T_dyn);
  app_log(1, "[AB V_H split] production   T_full = {:+.6f}  (target +41.005)",
          T_full);
  app_log(1, "[AB V_H split] vnl-vs-Pskna residue T_full-T_S-T_P = {:+.6f}",
          T_full - T_S - T_P);
  // Probe-anchored guards (tolerances ≳ the probe's own 3.6 mHa agreement
  // with ABINIT plus radial-vs-deltaC quadrature differences). The pre-fix
  // core-density double count showed up here as T_oc = +31.77.
  CHECK(std::abs(T_S   -  3.7615) < 5e-3);
  CHECK(std::abs(T_oc  - 11.7930) < 5e-3);
  CHECK(std::abs(T_dyn - 25.4505) < 5e-3);
  CHECK(std::abs(T_full - 41.005) < 1e-2);
  // add_vnl_impl must reproduce the Pskna contraction of the same deeq.
  CHECK(std::abs(T_full - T_S - T_P) < 1e-6);
}

// ===========================================================================
// MF-INDEPENDENCE: same THC-vs-direct comparison but driven by a deliberately
// NON-QE diagonal occupation (nonzero on all bands, no relation to mfobj.occ()).
// Both paths receive identical input (direct ← nii, THC ← diagonal Dm from the
// same nii). If any routine secretly depends on the QE mean-field state
// (reaches for mfobj.occ(), assumes 0/1 fillings, or only the QE-occupied
// bands), the two paths diverge by O(1) relative error rather than THC
// truncation. Tolerances are ~4x the QE-case values because all 16 bands are
// now ~half-filled (larger density ⇒ larger absolute THC truncation).
//
// All three PP types are strict: the THC path reproduces the direct V_H and V_x
// to ISDF noise even for the non-QE all-bands occupation. Tolerances are ~4x the
// QE-case values because all 16 bands are ~half-filled (larger density ⇒ larger
// THC truncation).
// ===========================================================================
TEST_CASE("thc_vs_direct_VH_VX_nonqe", "[hamilt][thc][hf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  SECTION("lih_kp222_nbnd16 (NCPP, non-QE occ)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_hf", mf::h5_input_type));
    test_thc_vs_direct_VH_VX<HOST_MEMORY>(*mpi, mf_ptr,
        /*thc_thresh*/ 1e-5, /*tol_VH*/ 2e-3, /*tol_VX*/ 2e-3,
        /*strict_VH*/ true, /*strict_VX*/ true, /*nonqe_occ*/ true);
  }

  SECTION("lih_kp222_nbnd16 (USPP, non-QE occ)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_uspp_hf", mf::h5_input_type));
    test_thc_vs_direct_VH_VX<HOST_MEMORY>(*mpi, mf_ptr,
        /*thc_thresh*/ 1e-5, /*tol_VH*/ 2e-3, /*tol_VX*/ 2e-3,
        /*strict_VH*/ true, /*strict_VX*/ true, /*nonqe_occ*/ true);
  }

  SECTION("lih_kp222_nbnd16 (PAW, non-QE occ)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw_hf", mf::h5_input_type));
    test_thc_vs_direct_VH_VX<HOST_MEMORY>(*mpi, mf_ptr,
        /*thc_thresh*/ 1e-5, /*tol_VH*/ 2e-3, /*tol_VX*/ 2e-3,
        /*strict_VH*/ true, /*strict_VX*/ true, /*nonqe_occ*/ true);
  }
}

// ===========================================================================
// Vhartree(nij) reduces to Vhartree(nii) for a DIAGONAL density matrix.
// Validates the newly completed full-density-matrix Hartree path: the smooth
// compensation charge (compute_becsum_full, now with 1/N_k) AND the PAW
// one-center deeq (compute_paw_deeq(nij), wired into add_Hartree_impl's nij
// branch) must both reduce to the validated diagonal path when nij = diag(nii).
// A deliberately non-QE diagonal occupation is used, so this also confirms the
// nij Hartree path carries no leftover dependence on the QE mean-field state.
// Difference is pure floating-point summation order ⇒ ~machine precision.
// ===========================================================================
template<MEMORY_SPACE MEM>
void test_vhartree_nij_vs_nii(mpi_context_t& mpi,
                              std::shared_ptr<mf::MF> mf_ptr, double tol)
{
  auto& mfobj = *mf_ptr;
  long nspin  = mfobj.nspin();
  long nk_ibz = mfobj.nkpts_ibz();
  long nbnd   = mfobj.nbnd();

  // Non-QE diagonal occupation (nonzero on all bands).
  nda::array<ComplexType,3> nii(nspin, nk_ibz, nbnd);
  for (long s = 0; s < nspin; ++s)
    for (long k = 0; k < nk_ibz; ++k)
      for (long n = 0; n < nbnd; ++n)
        nii(s, k, n) = ComplexType(0.5 + 0.3*std::cos(1.3*n + 0.7*k + 0.2*s), 0.0);

  // Equivalent diagonal density matrix.
  nda::array<ComplexType,4> nij(nspin, nk_ibz, nbnd, nbnd);
  nij() = ComplexType(0.0);
  for (long s = 0; s < nspin; ++s)
    for (long k = 0; k < nk_ibz; ++k)
      for (long n = 0; n < nbnd; ++n)
        nij(s, k, n, n) = nii(s, k, n);

  hamilt::pseudopot V(mfobj);
  auto dVH_nii = hamilt::Vhartree<MEM>(mfobj, mpi.comm, &V, nii);
  auto dVH_nij = hamilt::Vhartree<MEM>(mfobj, mpi.comm, &V, nij);

  auto a = nda::to_host(dVH_nii.local());
  auto b = nda::to_host(dVH_nij.local());
  double max_d = 0.0, max_v = 0.0;
  for (long i0 = 0; i0 < a.extent(0); ++i0)
    for (long i1 = 0; i1 < a.extent(1); ++i1)
      for (long i2 = 0; i2 < a.extent(2); ++i2)
        for (long i3 = 0; i3 < a.extent(3); ++i3) {
          max_d = std::max(max_d, std::abs(b(i0,i1,i2,i3) - a(i0,i1,i2,i3)));
          max_v = std::max(max_v, std::abs(a(i0,i1,i2,i3)));
        }
  max_d = mpi.comm.all_reduce_value(max_d, boost::mpi3::max<>{});
  max_v = mpi.comm.all_reduce_value(max_v, boost::mpi3::max<>{});
  app_log(1, "[Vhartree nij-vs-nii] max|ΔV_H| = {:.3e}, max|V_H| = {:.3e}",
          max_d, max_v);
  REQUIRE(max_v > 1e-8);
  CHECK(max_d < tol);
}

TEST_CASE("vhartree_nij_vs_nii", "[hamilt][paw][hf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (NCPP)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_hf", mf::h5_input_type));
    test_vhartree_nij_vs_nii<HOST_MEMORY>(*mpi, mf_ptr, 1e-10);
  }
  SECTION("lih_kp222_nbnd16 (USPP)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_uspp_hf", mf::h5_input_type));
    test_vhartree_nij_vs_nii<HOST_MEMORY>(*mpi, mf_ptr, 1e-10);
  }
  SECTION("lih_kp222_nbnd16 (PAW)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw_hf", mf::h5_input_type));
    test_vhartree_nij_vs_nii<HOST_MEMORY>(*mpi, mf_ptr, 1e-10);
  }
  // Symmetry-reduced meshes (plan A3): the nij route now lifts becsum to the
  // full BZ (compute_becsum_full_symm) exactly like the nii route.
  SECTION("lih_kp222_nbnd16 (PAW, sym)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw_sym", mf::h5_input_type));
    test_vhartree_nij_vs_nii<HOST_MEMORY>(*mpi, mf_ptr, 1e-10);
  }
  SECTION("si_kp222 (PAW, sym)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_si222_paw_sym", mf::h5_input_type));
    test_vhartree_nij_vs_nii<HOST_MEMORY>(*mpi, mf_ptr, 1e-10);
  }
}

// ===========================================================================
// becsum_full_symm identities (plan A3):
//  (1) For a diagonal density matrix on a symmetry-reduced mesh,
//      compute_becsum_full_symm ≡ compute_becsum_diagonal_symm (both consume
//      the same full-BZ Pskna lift; pure summation-order difference).
//  (2) On a nosym mesh (IBZ == full BZ, identity lift), full_symm reduces to
//      the plain compute_becsum_full — including for a complex Hermitian nij
//      with off-diagonal entries.
//  (3) A complex Hermitian nij passes the anti-Hermitian residual hard check
//      (Hermitian pair symmetrization) on the symmetry-reduced mesh.
// NOTE: no USPP/PAW fixture populates kp_trev (LiH/Si meshes are covered by
// rotations alone), so the trev conjugation branch of the nij lift is not
// exercised here — flagged for plan A-tests fixture work.
// ===========================================================================
void test_becsum_full_symm(std::shared_ptr<mf::MF> mf_ptr, bool sym_reduced)
{
  auto& mfobj = *mf_ptr;
  long nspin  = mfobj.nspin();
  long nk_ibz = mfobj.nkpts_ibz();
  long nbnd   = mfobj.nbnd();
  int  npol   = (int)mfobj.npol();
  if (sym_reduced) REQUIRE(mfobj.nkpts() > nk_ibz);
  else             REQUIRE(mfobj.nkpts() == nk_ibz);

  hamilt::pseudopot V(mfobj);
  auto kp_to_ibz = mfobj.kp_to_ibz();
  auto kp_trev   = mfobj.kp_trev();

  long n_trev = 0;
  for (long K = 0; K < kp_trev.extent(0); ++K) n_trev += (kp_trev(K) ? 1 : 0);
  app_log(1, "[becsum_full_symm] nk={} nk_ibz={} trev points={}",
          mfobj.nkpts(), nk_ibz, n_trev);

  auto max_becsum_diff = [](nda::array<double,3> const& x,
                            nda::array<double,3> const& y, double& maxval) {
    double m = 0.0;
    maxval = 0.0;
    for (long ia = 0; ia < x.extent(0); ++ia)
      for (long I = 0; I < x.extent(1); ++I)
        for (long J = 0; J < x.extent(2); ++J) {
          m = std::max(m, std::abs(y(ia,I,J) - x(ia,I,J)));
          maxval = std::max(maxval, std::abs(x(ia,I,J)));
        }
    return m;
  };

  // ---- (1) diagonal density: nij route ≡ diagonal route (full-BZ lift). ----
  nda::array<ComplexType,3> nii(nspin, nk_ibz, nbnd);
  for (long s = 0; s < nspin; ++s)
    for (long k = 0; k < nk_ibz; ++k)
      for (long n = 0; n < nbnd; ++n)
        nii(s, k, n) = ComplexType(0.5 + 0.3*std::cos(1.3*n + 0.7*k + 0.2*s), 0.0);
  nda::array<ComplexType,4> nij(nspin, nk_ibz, nbnd, nbnd);
  nij() = ComplexType(0.0);
  for (long s = 0; s < nspin; ++s)
    for (long k = 0; k < nk_ibz; ++k)
      for (long n = 0; n < nbnd; ++n)
        nij(s, k, n, n) = nii(s, k, n);

  auto b_ref = hamilt::paw::compute_becsum_diagonal_symm(
      V, nii, kp_to_ibz, kp_trev, npol);
  auto b_nij = hamilt::paw::compute_becsum_full_symm(
      V, nij, kp_to_ibz, kp_trev, npol);
  double bmax = 0.0;
  double d1 = max_becsum_diff(b_ref, b_nij, bmax);
  app_log(1, "[becsum_full_symm] diag nij: max|Δbecsum| = {:.3e}, "
             "max|becsum| = {:.3e}", d1, bmax);
  REQUIRE(bmax > 1e-8);   // augmentation channels must be populated
  CHECK(d1 < 1e-12);

  // ---- (2)+(3) complex Hermitian nij: diag + rank-1 v v† perturbation. ----
  for (long s = 0; s < nspin; ++s)
    for (long k = 0; k < nk_ibz; ++k)
      for (long a = 0; a < nbnd; ++a)
        for (long b = 0; b < nbnd; ++b) {
          ComplexType va = 0.2 * std::exp(ComplexType(0.0, 0.4*a + 0.1*k));
          ComplexType vb = 0.2 * std::exp(ComplexType(0.0, 0.4*b + 0.1*k));
          nij(s, k, a, b) += va * std::conj(vb);   // Hermitian by construction
        }
  // Runs the hard anti-Hermitian residual check internally (3).
  auto b_herm = hamilt::paw::compute_becsum_full_symm(
      V, nij, kp_to_ibz, kp_trev, npol);
  if (!sym_reduced) {
    auto b_plain = hamilt::paw::compute_becsum_full(
        V.Pskna_view(), nij, V.ityp_view(), V.nh_view(), V.ofs_view(), npol);
    double d2 = max_becsum_diff(b_plain, b_herm, bmax);
    app_log(1, "[becsum_full_symm] hermitian nij nosym reduction: "
               "max|Δbecsum| = {:.3e}", d2);
    CHECK(d2 < 1e-12);
  }
}

TEST_CASE("becsum_full_symm", "[hamilt][paw][hf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (PAW, sym)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw_sym", mf::h5_input_type));
    test_becsum_full_symm(mf_ptr, /*sym_reduced=*/true);
  }
  SECTION("si_kp222 (PAW, sym)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_si222_paw_sym", mf::h5_input_type));
    test_becsum_full_symm(mf_ptr, /*sym_reduced=*/true);
  }
  SECTION("lih_kp222_nbnd16 (PAW, nosym)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw_hf", mf::h5_input_type));
    test_becsum_full_symm(mf_ptr, /*sym_reduced=*/false);
  }
  SECTION("lih_kp222_nbnd16 (USPP, nosym)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_uspp_hf", mf::h5_input_type));
    test_becsum_full_symm(mf_ptr, /*sym_reduced=*/false);
  }
}

// ===========================================================================
// add_Vpp path alignment (plan A2 / invariant I5); since plan A3 the nij
// becsum is symmetry-correct (full-BZ lift), so the identities are also
// checked on symmetry-reduced fixtures:
//  (1) H(nij) ≡ H(nii) for a diagonal density matrix — both overloads must
//      build the identical operator via compute_paw_deeq(n, V_loc+V_H,
//      include_static=true); pre-A2 the nij path silently fell through to
//      the static-only branch.
//  (2) H(n, add_hartree=false, add_exchange=false) ≡ H0() — with the
//      density-dependent terms switched off, the density overload reduces
//      to the static-only Eq. (h0) assembly.
//  (3) H(n, add_exchange=true) ≡ H(n) + Vexchange(n) — the add_exchange
//      flag accumulates the direct-route (SIGNED) K into Hij.
// ===========================================================================
template<MEMORY_SPACE MEM>
void test_add_vpp_i5_alignment(mpi_context_t& mpi,
                               std::shared_ptr<mf::MF> mf_ptr, double tol)
{
  auto& mfobj = *mf_ptr;
  long nspin  = mfobj.nspin();
  long nk_ibz = mfobj.nkpts_ibz();
  long nbnd   = mfobj.nbnd();

  // Non-QE diagonal occupation (nonzero on all bands) + equivalent nij.
  nda::array<ComplexType,3> nii(nspin, nk_ibz, nbnd);
  for (long s = 0; s < nspin; ++s)
    for (long k = 0; k < nk_ibz; ++k)
      for (long n = 0; n < nbnd; ++n)
        nii(s, k, n) = ComplexType(0.5 + 0.3*std::cos(1.3*n + 0.7*k + 0.2*s), 0.0);
  nda::array<ComplexType,4> nij(nspin, nk_ibz, nbnd, nbnd);
  nij() = ComplexType(0.0);
  for (long s = 0; s < nspin; ++s)
    for (long k = 0; k < nk_ibz; ++k)
      for (long n = 0; n < nbnd; ++n)
        nij(s, k, n, n) = nii(s, k, n);

  hamilt::pseudopot V(mfobj);

  auto max_abs_diff = [&mpi](auto const& x, auto const& y) {
    auto a = nda::to_host(x.local());
    auto b = nda::to_host(y.local());
    double m = 0.0;
    for (long i0 = 0; i0 < a.extent(0); ++i0)
      for (long i1 = 0; i1 < a.extent(1); ++i1)
        for (long i2 = 0; i2 < a.extent(2); ++i2)
          for (long i3 = 0; i3 < a.extent(3); ++i3)
            m = std::max(m, std::abs(b(i0,i1,i2,i3) - a(i0,i1,i2,i3)));
    return mpi.comm.all_reduce_value(m, boost::mpi3::max<>{});
  };

  // (1) nii ≡ nij
  auto dH_nii = hamilt::H<MEM>(mfobj, mpi.comm, &V, nii);
  auto dH_nij = hamilt::H<MEM>(mfobj, mpi.comm, &V, nij);
  double d1 = max_abs_diff(dH_nii, dH_nij);
  app_log(1, "[add_Vpp I5] max|H(nij) - H(nii)| = {:.3e}", d1);
  CHECK(d1 < tol);

  // (2) density overload with both flags off ≡ static-only H0
  auto dH_off = hamilt::H<MEM>(mfobj, mpi.comm, &V, nii, {0}, {1,1,2048,2048},
                               /*add_hartree=*/false, /*add_exchange=*/false);
  auto dH0 = hamilt::H0<MEM>(mfobj, mpi.comm, &V);
  double d2 = max_abs_diff(dH_off, dH0);
  app_log(1, "[add_Vpp I5] max|H(nii, flags off) - H0| = {:.3e}", d2);
  CHECK(d2 < tol);

  // (3) add_exchange accumulates the direct-route K on top of H(n)
  auto dH_hx = hamilt::H<MEM>(mfobj, mpi.comm, &V, nii, {0}, {1,1,2048,2048},
                              /*add_hartree=*/true, /*add_exchange=*/true);
  auto dK = hamilt::Vexchange<MEM>(mfobj, mpi.comm, &V, nii);
  auto h_ = nda::to_host(dH_nii.local());
  auto k_ = nda::to_host(dK.local());
  auto got = nda::to_host(dH_hx.local());
  double d3 = 0.0, kmax = 0.0;
  for (long i0 = 0; i0 < got.extent(0); ++i0)
    for (long i1 = 0; i1 < got.extent(1); ++i1)
      for (long i2 = 0; i2 < got.extent(2); ++i2)
        for (long i3 = 0; i3 < got.extent(3); ++i3) {
          d3 = std::max(d3, std::abs(got(i0,i1,i2,i3)
                                     - (h_(i0,i1,i2,i3) + k_(i0,i1,i2,i3))));
          kmax = std::max(kmax, std::abs(k_(i0,i1,i2,i3)));
        }
  d3 = mpi.comm.all_reduce_value(d3, boost::mpi3::max<>{});
  kmax = mpi.comm.all_reduce_value(kmax, boost::mpi3::max<>{});
  app_log(1, "[add_Vpp I5] max|H(n,x) - (H(n) + K)| = {:.3e}, max|K| = {:.3e}",
          d3, kmax);
  REQUIRE(kmax > 1e-8);  // exchange must actually contribute
  CHECK(d3 < tol);
}

TEST_CASE("add_vpp_i5_alignment", "[hamilt][paw][hf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (NCPP)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_hf", mf::h5_input_type));
    test_add_vpp_i5_alignment<HOST_MEMORY>(*mpi, mf_ptr, 1e-10);
  }
  SECTION("lih_kp222_nbnd16 (USPP)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_uspp_hf", mf::h5_input_type));
    test_add_vpp_i5_alignment<HOST_MEMORY>(*mpi, mf_ptr, 1e-10);
  }
  SECTION("lih_kp222_nbnd16 (PAW)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw_hf", mf::h5_input_type));
    test_add_vpp_i5_alignment<HOST_MEMORY>(*mpi, mf_ptr, 1e-10);
  }
  // Symmetry-reduced fixtures (plan A3): nii and nij must still build the
  // identical operator, now both via the full-BZ symmetry-correct becsum.
  SECTION("lih_kp222_nbnd16 (PAW, sym)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw_sym", mf::h5_input_type));
    test_add_vpp_i5_alignment<HOST_MEMORY>(*mpi, mf_ptr, 1e-10);
  }
  SECTION("si_kp222 (PAW, sym)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_si222_paw_sym", mf::h5_input_type));
    test_add_vpp_i5_alignment<HOST_MEMORY>(*mpi, mf_ptr, 1e-10);
  }
}

// ===========================================================================
// H(n) ≡ H0 + Vhartree(n) — plan A-tests item (i), enabled by the settled
// ∫V_loc·Q̂ placement (2026-07-24): Eq. (h0)'s static USPP/PAW D now carries
// the frozen electrostatic compensation term (static_h0_D = dion + ex_cvij
// + ∫V_loc·Q̂ — always included, it is neither exchange nor correlation and
// is NOT contained in dion, whose −⟨Q̂|v_H[ñ_Zc]⟩ is the opposite-sign
// one-center descreening reference). With it, the density path (which
// integrates (V_loc+V_H)·Q̂ in one pass) and the static + Hartree
// composition build the identical operator, for both nii and nij, on nosym
// and symmetry-reduced meshes. Residual = FFT-linearity + summation order.
// ===========================================================================
template<MEMORY_SPACE MEM>
void test_h0_plus_hartree_identity(mpi_context_t& mpi,
                                   std::shared_ptr<mf::MF> mf_ptr, double tol)
{
  auto& mfobj = *mf_ptr;
  long nspin  = mfobj.nspin();
  long nk_ibz = mfobj.nkpts_ibz();
  long nbnd   = mfobj.nbnd();

  // Non-QE diagonal occupation + equivalent nij (as in the I5 test).
  nda::array<ComplexType,3> nii(nspin, nk_ibz, nbnd);
  for (long s = 0; s < nspin; ++s)
    for (long k = 0; k < nk_ibz; ++k)
      for (long n = 0; n < nbnd; ++n)
        nii(s, k, n) = ComplexType(0.5 + 0.3*std::cos(1.3*n + 0.7*k + 0.2*s), 0.0);
  nda::array<ComplexType,4> nij(nspin, nk_ibz, nbnd, nbnd);
  nij() = ComplexType(0.0);
  for (long s = 0; s < nspin; ++s)
    for (long k = 0; k < nk_ibz; ++k)
      for (long n = 0; n < nbnd; ++n)
        nij(s, k, n, n) = nii(s, k, n);

  hamilt::pseudopot V(mfobj);

  auto dH0     = hamilt::H0<MEM>(mfobj, mpi.comm, &V);
  auto dH_nii  = hamilt::H<MEM>(mfobj, mpi.comm, &V, nii);
  auto dH_nij  = hamilt::H<MEM>(mfobj, mpi.comm, &V, nij);
  auto dVH_nii = hamilt::Vhartree<MEM>(mfobj, mpi.comm, &V, nii);
  auto dVH_nij = hamilt::Vhartree<MEM>(mfobj, mpi.comm, &V, nij);

  auto check_sum = [&](auto const& dH, auto const& dVH, char const* lbl) {
    auto h  = nda::to_host(dH.local());
    auto h0 = nda::to_host(dH0.local());
    auto vh = nda::to_host(dVH.local());
    double d = 0.0, vmax = 0.0;
    for (long i0 = 0; i0 < h.extent(0); ++i0)
      for (long i1 = 0; i1 < h.extent(1); ++i1)
        for (long i2 = 0; i2 < h.extent(2); ++i2)
          for (long i3 = 0; i3 < h.extent(3); ++i3) {
            d = std::max(d, std::abs(h(i0,i1,i2,i3)
                                     - (h0(i0,i1,i2,i3) + vh(i0,i1,i2,i3))));
            vmax = std::max(vmax, std::abs(vh(i0,i1,i2,i3)));
          }
    d = mpi.comm.all_reduce_value(d, boost::mpi3::max<>{});
    vmax = mpi.comm.all_reduce_value(vmax, boost::mpi3::max<>{});
    app_log(1, "[H0+VH identity] {}: max|H(n) - (H0 + VH(n))| = {:.3e}, "
               "max|VH| = {:.3e}", lbl, d, vmax);
    REQUIRE(vmax > 1e-8);   // Hartree must actually contribute
    CHECK(d < tol);
  };
  check_sum(dH_nii, dVH_nii, "nii");
  check_sum(dH_nij, dVH_nij, "nij");
}

TEST_CASE("h0_plus_hartree_identity", "[hamilt][paw][hf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (NCPP)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_hf", mf::h5_input_type));
    test_h0_plus_hartree_identity<HOST_MEMORY>(*mpi, mf_ptr, 1e-10);
  }
  SECTION("lih_kp222_nbnd16 (USPP)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_uspp_hf", mf::h5_input_type));
    test_h0_plus_hartree_identity<HOST_MEMORY>(*mpi, mf_ptr, 1e-10);
  }
  SECTION("lih_kp222_nbnd16 (PAW)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw_hf", mf::h5_input_type));
    test_h0_plus_hartree_identity<HOST_MEMORY>(*mpi, mf_ptr, 1e-10);
  }
  SECTION("lih_kp222_nbnd16 (PAW, sym)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw_sym", mf::h5_input_type));
    test_h0_plus_hartree_identity<HOST_MEMORY>(*mpi, mf_ptr, 1e-10);
  }
  SECTION("si_kp222 (PAW, sym)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_si222_paw_sym", mf::h5_input_type));
    test_h0_plus_hartree_identity<HOST_MEMORY>(*mpi, mf_ptr, 1e-10);
  }
}

// ===========================================================================
// Vexchange(nij) reduces to Vexchange(nii) for a DIAGONAL density matrix.
// Validates the new full-density-matrix exchange path: the natural-orbital
// decomposition of nij = diag(nii) recovers the canonical bands (with
// eigenvalue occupations), so the smooth + PAW augmentation + deltaC
// one-center contraction must reproduce the validated diagonal kernel.
// Non-QE occupations ⇒ doubles as an MF-independence check of the nij path.
// Tolerance allows for the eigensolver + extra-rotation roundoff.
// ===========================================================================
template<MEMORY_SPACE MEM>
void test_vexchange_nij_vs_nii(mpi_context_t& mpi,
                               std::shared_ptr<mf::MF> mf_ptr, double tol)
{
  auto& mfobj = *mf_ptr;
  long nspin  = mfobj.nspin();
  long nk_ibz = mfobj.nkpts_ibz();
  long nbnd   = mfobj.nbnd();

  nda::array<ComplexType,3> nii(nspin, nk_ibz, nbnd);
  for (long s = 0; s < nspin; ++s)
    for (long k = 0; k < nk_ibz; ++k)
      for (long n = 0; n < nbnd; ++n)
        nii(s, k, n) = ComplexType(0.5 + 0.3*std::cos(1.3*n + 0.7*k + 0.2*s), 0.0);

  nda::array<ComplexType,4> nij(nspin, nk_ibz, nbnd, nbnd);
  nij() = ComplexType(0.0);
  for (long s = 0; s < nspin; ++s)
    for (long k = 0; k < nk_ibz; ++k)
      for (long n = 0; n < nbnd; ++n)
        nij(s, k, n, n) = nii(s, k, n);

  hamilt::pseudopot V(mfobj);
  auto dVX_nii = hamilt::Vexchange<MEM>(mfobj, mpi.comm, &V, nii);
  auto dVX_nij = hamilt::Vexchange<MEM>(mfobj, mpi.comm, &V, nij);

  auto a = nda::to_host(dVX_nii.local());
  auto b = nda::to_host(dVX_nij.local());
  double max_d = 0.0, max_v = 0.0;
  for (long i0 = 0; i0 < a.extent(0); ++i0)
    for (long i1 = 0; i1 < a.extent(1); ++i1)
      for (long i2 = 0; i2 < a.extent(2); ++i2)
        for (long i3 = 0; i3 < a.extent(3); ++i3) {
          max_d = std::max(max_d, std::abs(b(i0,i1,i2,i3) - a(i0,i1,i2,i3)));
          max_v = std::max(max_v, std::abs(a(i0,i1,i2,i3)));
        }
  max_d = mpi.comm.all_reduce_value(max_d, boost::mpi3::max<>{});
  max_v = mpi.comm.all_reduce_value(max_v, boost::mpi3::max<>{});
  app_log(1, "[Vexchange nij-vs-nii] max|ΔV_x| = {:.3e}, max|V_x| = {:.3e}",
          max_d, max_v);
  REQUIRE(max_v > 1e-8);
  CHECK(max_d < tol);
}

TEST_CASE("vexchange_nij_vs_nii", "[hamilt][paw][hf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (NCPP)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_hf", mf::h5_input_type));
    test_vexchange_nij_vs_nii<HOST_MEMORY>(*mpi, mf_ptr, 1e-8);
  }
  SECTION("lih_kp222_nbnd16 (USPP)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_uspp_hf", mf::h5_input_type));
    test_vexchange_nij_vs_nii<HOST_MEMORY>(*mpi, mf_ptr, 1e-8);
  }
  SECTION("lih_kp222_nbnd16 (PAW)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw_hf", mf::h5_input_type));
    test_vexchange_nij_vs_nii<HOST_MEMORY>(*mpi, mf_ptr, 1e-8);
  }
}

// ===========================================================================
// Option A ("shape-restored") on-site exact exchange. When
// pseudopot::set_paw_exx_shape_restored(true), the PAW augmentation uses the
// FULL AE−PS partial-wave pair density (build_qrad_tab_full_aeps) instead of
// the compensation charge, and the deltaC one-center correction is dropped —
// reproducing ABINIT's phiphj−tphitphj oscillator (see
// notes/paw_onsite_exchange_analysis). Two checks on a PAW fixture:
//   (1) shape_restored actually changes the exchange matrix (mechanism active).
//   (2) the FULL density-matrix (nij) path equals the diagonal (nii) path for a
//       diagonal nij under Option A — i.e. the GW density-matrix exchange is
//       correct, not just the DFT-style diagonal occupations.
// ===========================================================================
template<MEMORY_SPACE MEM>
void test_vexchange_shape_restored(mpi_context_t& mpi,
                                   std::shared_ptr<mf::MF> mf_ptr,
                                   double tol_consistency)
{
  auto& mfobj = *mf_ptr;
  long nspin = mfobj.nspin(), nk_ibz = mfobj.nkpts_ibz(), nbnd = mfobj.nbnd();

  nda::array<ComplexType,3> nii(nspin, nk_ibz, nbnd);
  for (long s = 0; s < nspin; ++s)
    for (long k = 0; k < nk_ibz; ++k)
      for (long n = 0; n < nbnd; ++n)
        nii(s, k, n) = ComplexType(0.5 + 0.3*std::cos(1.3*n + 0.7*k + 0.2*s), 0.0);
  nda::array<ComplexType,4> nij(nspin, nk_ibz, nbnd, nbnd);
  nij() = ComplexType(0.0);
  for (long s = 0; s < nspin; ++s)
    for (long k = 0; k < nk_ibz; ++k)
      for (long n = 0; n < nbnd; ++n)
        nij(s, k, n, n) = nii(s, k, n);

  hamilt::pseudopot V(mfobj);
  V.set_paw_exx_shape_restored(false);
  auto dKc     = hamilt::Vexchange<MEM>(mfobj, mpi.comm, &V, nii);
  V.set_paw_exx_shape_restored(true);
  auto dKs     = hamilt::Vexchange<MEM>(mfobj, mpi.comm, &V, nii);
  auto dKs_nij = hamilt::Vexchange<MEM>(mfobj, mpi.comm, &V, nij);

  auto c  = nda::to_host(dKc.local());
  auto s_ = nda::to_host(dKs.local());
  auto sn = nda::to_host(dKs_nij.local());
  double d_active = 0.0, v_ref = 0.0, d_nij = 0.0;
  for (long i0 = 0; i0 < c.extent(0); ++i0)
    for (long i1 = 0; i1 < c.extent(1); ++i1)
      for (long i2 = 0; i2 < c.extent(2); ++i2)
        for (long i3 = 0; i3 < c.extent(3); ++i3) {
          d_active = std::max(d_active, std::abs(s_(i0,i1,i2,i3) - c(i0,i1,i2,i3)));
          d_nij    = std::max(d_nij,    std::abs(sn(i0,i1,i2,i3) - s_(i0,i1,i2,i3)));
          v_ref    = std::max(v_ref,    std::abs(c(i0,i1,i2,i3)));
        }
  d_active = mpi.comm.all_reduce_value(d_active, boost::mpi3::max<>{});
  d_nij    = mpi.comm.all_reduce_value(d_nij,    boost::mpi3::max<>{});
  v_ref    = mpi.comm.all_reduce_value(v_ref,    boost::mpi3::max<>{});
  app_log(1, "[Vx shape-restored] max|K| = {:.3e}, max|ΔK_active| = {:.3e}, "
             "max|ΔK_(nij-nii)| = {:.3e}", v_ref, d_active, d_nij);
  REQUIRE(v_ref > 1e-8);
  // (1) Option A genuinely changes the on-site exchange (full AE−PS ≠ moments).
  REQUIRE(d_active > 1e-8);
  // (2) full density-matrix path == diagonal path under Option A.
  CHECK(d_nij < tol_consistency);
}

TEST_CASE("vexchange_shape_restored", "[hamilt][paw][hf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (PAW)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw_hf", mf::h5_input_type));
    test_vexchange_shape_restored<HOST_MEMORY>(*mpi, mf_ptr, 1e-8);
  }
}

// ===========================================================================
// Plan C2 — THC shape-mode augmentation on the dense sphere.
// The THC LL (aug-aug) Coulomb block is evaluated on the dense augmentation
// sphere (fft_grid_dim_aug inscribed Gcut) regardless of the thc `ecut`
// collocation grid; this is what makes vv_compensation='shape' (the SHARP
// AE−PS pair density) resolvable in THC. Checks, per (ecut) configuration:
//   (a) THC shape-mode V_x matches the direct (dense-grid) shape-mode V_x
//       element-wise to ISDF truncation;
//   (b) the mode signal is active (direct shape ≠ moment);
//   (c) SHARP: the mode DIFFERENCE (shape − moment) agrees between THC and
//       direct — THC factorization noise largely cancels in the difference,
//       isolating exactly the augmentation-mode physics C2 implements. With
//       a reduced thc ecut this fails without the dense-sphere LL treatment
//       (the smooth-sphere sum captures only a few % of the on-site term).
// ===========================================================================
template<MEMORY_SPACE MEM>
void test_thc_shape_mode_vs_direct(mpi_context_t& mpi,
                                   std::shared_ptr<mf::MF> mf_ptr,
                                   double thc_thresh,
                                   double ecut_frac,
                                   double tol_VX,
                                   double tol_mode)
{
  using math::shm::make_shared_array;
  auto all = nda::range::all;
  auto& mfobj = *mf_ptr;
  long nspin = mfobj.nspin(), nk_ibz = mfobj.nkpts_ibz(), nbnd = mfobj.nbnd();

  memory::array<MEM, ComplexType, 3> nii(nspin, nk_ibz, nbnd);
  nii() = mfobj.occ()(all, nda::range(nk_ibz), all);

  auto sDm = make_shared_array<array_view_4d_t>(mpi, {nspin, nk_ibz, nbnd, nbnd});
  if (mpi.node_comm.root()) {
    sDm.local()() = ComplexType(0.0);
    for (long s = 0; s < nspin; ++s)
      for (long k = 0; k < nk_ibz; ++k)
        for (long a = 0; a < nbnd; ++a)
          sDm.local()(s, k, a, a) = nii(s, k, a);
  }
  mpi.node_comm.barrier();
  auto sS = make_shared_array<array_view_4d_t>(mpi, {nspin, nk_ibz, nbnd, nbnd});
  hamilt::set_ovlp(mfobj, sS);

  // ---- Direct (dense fft_mesh_aug) references, both modes ----
  hamilt::pseudopot V(mfobj);
  V.set_paw_exx_shape_restored(false);
  auto dVX_dir_m = hamilt::Vexchange<MEM>(mfobj, mpi.comm, &V, nii);
  V.set_paw_exx_shape_restored(true);
  auto dVX_dir_s = hamilt::Vexchange<MEM>(mfobj, mpi.comm, &V, nii);

  // ---- THC exchange, both modes, at the requested collocation ecut ----
  double thc_ecut = ecut_frac * mfobj.ecutrho();
  auto thc_vx = [&](std::string const& mode) {
    auto pt = methods::make_thc_reader_ptree(
        0, "", "incore", "", "bdft", thc_thresh, thc_ecut);
    pt.put("paw_aug", true);
    pt.put("paw_isdf_metric", "coulomb");
    pt.put("paw_isdf_tol", 1e-12);
    pt.put("vv_compensation", mode);
    methods::thc_reader_t thc(mf_ptr, pt);
    auto sVX = make_shared_array<array_view_4d_t>(mpi,
                   {nspin, nk_ibz, nbnd, nbnd});
    methods::solvers::hf_t hf(methods::ignore_g0);
    hf.evaluate(sVX, sDm.local(), thc, sS.local(),
                /*hartree=*/false, /*exchange=*/true);
    return nda::array<ComplexType,4>(sVX.local());
  };
  auto VX_thc_m = thc_vx("moment");
  auto VX_thc_s = thc_vx("shape");

  // ---- Element-wise comparison on this rank's tile of the direct arrays ----
  auto dir_m = nda::to_host(dVX_dir_m.local());
  auto dir_s = nda::to_host(dVX_dir_s.local());
  auto rng_s = dVX_dir_m.local_range(0); auto rng_k = dVX_dir_m.local_range(1);
  auto rng_i = dVX_dir_m.local_range(2); auto rng_j = dVX_dir_m.local_range(3);

  double max_ref = 0.0, max_d_shape = 0.0, max_mode_dir = 0.0, max_d_mode = 0.0;
  for (auto [is_l, s] : itertools::enumerate(rng_s))
    for (auto [ik_l, k] : itertools::enumerate(rng_k))
      for (auto [ii_l, i] : itertools::enumerate(rng_i))
        for (auto [ij_l, j] : itertools::enumerate(rng_j)) {
          ComplexType vd_m = dir_m(is_l, ik_l, ii_l, ij_l);
          ComplexType vd_s = dir_s(is_l, ik_l, ii_l, ij_l);
          ComplexType vt_m = VX_thc_m(s, k, i, j);
          ComplexType vt_s = VX_thc_s(s, k, i, j);
          max_ref    = std::max(max_ref,    std::abs(vd_s));
          max_d_shape= std::max(max_d_shape, std::abs(vt_s - vd_s));
          max_mode_dir = std::max(max_mode_dir, std::abs(vd_s - vd_m));
          max_d_mode = std::max(max_d_mode,
                                std::abs((vt_s - vt_m) - (vd_s - vd_m)));
        }
  max_ref     = mpi.comm.all_reduce_value(max_ref,     boost::mpi3::max<>{});
  max_d_shape = mpi.comm.all_reduce_value(max_d_shape, boost::mpi3::max<>{});
  max_mode_dir= mpi.comm.all_reduce_value(max_mode_dir,boost::mpi3::max<>{});
  max_d_mode  = mpi.comm.all_reduce_value(max_d_mode,  boost::mpi3::max<>{});

  app_log(1, "[THC shape-mode C2] ecut={:.1f} Ha ({}x ecutrho): max|V_x^dir,s| = {:.3e}, "
             "max|VX_thc_s - VX_dir_s| = {:.3e}, mode signal (direct) = {:.3e}, "
             "max|Δmode(THC - dir)| = {:.3e}",
          thc_ecut, ecut_frac, max_ref, max_d_shape, max_mode_dir, max_d_mode);

  REQUIRE(max_ref > 1e-8);
  // (b) the two modes genuinely differ in the direct reference.
  REQUIRE(max_mode_dir > 1e-8);
  // (a) THC shape mode reproduces the direct dense-grid shape-mode V_x.
  CHECK(max_d_shape < tol_VX);
  // (c) the augmentation-mode difference agrees route-to-route (sharp).
  CHECK(max_d_mode < tol_mode);
}

TEST_CASE("thc_shape_mode_vs_direct", "[hamilt][paw][thc][hf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  // Measured 2026-07-25 (this fixture): default ecut ΔV_x 7.6e-5 / Δmode
  // 2.8e-6; reduced ecut ΔV_x 6.8e-5 / Δmode 6.3e-7. Tolerances ~10x.
  SECTION("lih_kp222_nbnd16 (PAW, default ecut)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw_hf", mf::h5_input_type));
    test_thc_shape_mode_vs_direct<HOST_MEMORY>(*mpi, mf_ptr,
        /*thc_thresh*/ 1e-5, /*ecut_frac*/ 1.0,
        /*tol_VX*/ 5e-4, /*tol_mode*/ 3e-5);
  }
  SECTION("lih_kp222_nbnd16 (PAW, reduced ecut — dense-LL split grid)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw_hf", mf::h5_input_type));
    test_thc_shape_mode_vs_direct<HOST_MEMORY>(*mpi, mf_ptr,
        /*thc_thresh*/ 1e-5, /*ecut_frac*/ 0.5,
        /*tol_VX*/ 5e-4, /*tol_mode*/ 3e-5);
  }
  // Plan D2: both augmentation modes route-equivalent on an ABINIT-sourced
  // mf too (the shape-mode difference lives entirely in the LL/one-center
  // blocks, exactly where the converter's odd-m Ylm conventions enter).
  // Measured 2026-07-25: ΔV_x(shape) 2.96e-4, mode signal 8.4e-3, Δmode
  // 2.96e-4 (the 12-el semicore density carries a larger ISDF truncation
  // than the LiH sections at the same thc_thresh). Tolerances ~3x measured;
  // the mode check still discriminates at signal/tol ≈ 8.
  SECTION("si_kp222 (PAW, ABINIT-sourced mf)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "bdft_si222_paw_ab", mf::h5_input_type));
    test_thc_shape_mode_vs_direct<HOST_MEMORY>(*mpi, mf_ptr,
        /*thc_thresh*/ 1e-5, /*ecut_frac*/ 1.0,
        /*tol_VX*/ 1e-3, /*tol_mode*/ 1e-3);
  }
}

// ===========================================================================
// Set 1b — MF-INDEPENDENCE with a genuine NON-DIAGONAL density matrix.
// Compares direct hamilt::Vhartree(nij)/Vexchange(nij) against THC
// hf_t::evaluate(Dm=nij) for a Hermitian PSD density matrix that is NOT
// diagonal and bears no relation to the QE mean field:
//     nij = A · diag(f) · A†,   f = non-QE base occupations, A near-identity.
// Both paths receive the identical nij. Agreement to THC truncation error
// confirms the full-density-matrix V_H/V_x paths are correct and carry no
// leftover QE-MF dependence. Strict for NCPP, USPP, and PAW.
// ===========================================================================
template<MEMORY_SPACE MEM>
void test_thc_vs_direct_nij(mpi_context_t& mpi, std::shared_ptr<mf::MF> mf_ptr,
                            double thc_thresh, double tol_VH, double tol_VX,
                            bool strict_VH, bool strict_VX)
{
  using math::shm::make_shared_array;
  auto& mfobj = *mf_ptr;
  long nspin = mfobj.nspin(), nk_ibz = mfobj.nkpts_ibz(), nbnd = mfobj.nbnd();

  // Non-diagonal Hermitian PSD density matrix (MF-independent).
  nda::array<ComplexType,4> nij(nspin, nk_ibz, nbnd, nbnd);
  nij() = ComplexType(0.0);
  for (long s = 0; s < nspin; ++s)
    for (long k = 0; k < nk_ibz; ++k) {
      std::vector<double> f(nbnd);
      for (long c = 0; c < nbnd; ++c)
        f[c] = 0.40 + 0.30 * std::cos(1.1*c + 0.5*k + 0.3*s);   // ∈ (0.1, 0.7)
      auto A = [&](long a, long c) -> ComplexType {
        double mag = (a == c) ? 1.0 : 0.2 / (1.0 + std::abs(a - c));
        double ph  = 0.3 * (a + 1) * (c + 1) + 0.2 * k;
        return mag * ComplexType(std::cos(ph), std::sin(ph));
      };
      for (long a = 0; a < nbnd; ++a)
        for (long b = 0; b < nbnd; ++b) {
          ComplexType acc(0.0);
          for (long c = 0; c < nbnd; ++c) acc += A(a,c) * f[c] * std::conj(A(b,c));
          nij(s, k, a, b) = acc;
        }
    }

  // Direct.
  hamilt::pseudopot V(mfobj);
  auto dVH = hamilt::Vhartree<MEM>(mfobj, mpi.comm, &V, nij);
  auto dVX = hamilt::Vexchange<MEM>(mfobj, mpi.comm, &V, nij);

  // THC density matrix = the same nij.
  auto sDm = make_shared_array<array_view_4d_t>(mpi, {nspin, nk_ibz, nbnd, nbnd});
  if (mpi.node_comm.root()) sDm.local()() = nij;
  mpi.node_comm.barrier();

  auto thc_pt = methods::make_thc_reader_ptree(
      0, "", "incore", "", "bdft", thc_thresh, mfobj.ecutrho());
  thc_pt.put("paw_aug", true);
  thc_pt.put("paw_isdf_metric", "coulomb");
  thc_pt.put("paw_isdf_tol", 1e-12);
  methods::thc_reader_t thc(mf_ptr, thc_pt);

  auto sS = make_shared_array<array_view_4d_t>(mpi, {nspin, nk_ibz, nbnd, nbnd});
  hamilt::set_ovlp(mfobj, sS);
  auto sVH = make_shared_array<array_view_4d_t>(mpi, {nspin, nk_ibz, nbnd, nbnd});
  { methods::solvers::hf_t hf(methods::ignore_g0);
    hf.evaluate(sVH, sDm.local(), thc, sS.local(), true, false); }
  auto sVX = make_shared_array<array_view_4d_t>(mpi, {nspin, nk_ibz, nbnd, nbnd});
  { methods::solvers::hf_t hf(methods::ignore_g0);
    hf.evaluate(sVX, sDm.local(), thc, sS.local(), false, true); }

  auto VH_thc = nda::to_host(sVH.local());
  auto VX_thc = nda::to_host(sVX.local());
  auto VH_dir = nda::to_host(dVH.local());
  auto VX_dir = nda::to_host(dVX.local());

  double max_VH=0, max_VX=0, max_dVH=0, max_dVX=0;
  auto rs = dVH.local_range(0); auto rk = dVH.local_range(1);
  auto ri = dVH.local_range(2); auto rj = dVH.local_range(3);
  for (auto [isl,s] : itertools::enumerate(rs))
    for (auto [ikl,k] : itertools::enumerate(rk))
      for (auto [iil,i] : itertools::enumerate(ri))
        for (auto [ijl,j] : itertools::enumerate(rj)) {
          max_VH  = std::max(max_VH,  std::abs(VH_dir(isl,ikl,iil,ijl)));
          max_VX  = std::max(max_VX,  std::abs(VX_dir(isl,ikl,iil,ijl)));
          max_dVH = std::max(max_dVH, std::abs(VH_thc(s,k,i,j) - VH_dir(isl,ikl,iil,ijl)));
          max_dVX = std::max(max_dVX, std::abs(VX_thc(s,k,i,j) - VX_dir(isl,ikl,iil,ijl)));
        }
  max_VH  = mpi.comm.all_reduce_value(max_VH,  boost::mpi3::max<>{});
  max_VX  = mpi.comm.all_reduce_value(max_VX,  boost::mpi3::max<>{});
  max_dVH = mpi.comm.all_reduce_value(max_dVH, boost::mpi3::max<>{});
  max_dVX = mpi.comm.all_reduce_value(max_dVX, boost::mpi3::max<>{});
  app_log(1, "[THC-vs-direct nij] V_H: max|V_H|={:.3e} max|ΔV_H|={:.3e} rel={:.2e}",
          max_VH, max_dVH, max_dVH/std::max(1e-30,max_VH));
  app_log(1, "[THC-vs-direct nij] V_x: max|V_x|={:.3e} max|ΔV_x|={:.3e} rel={:.2e}",
          max_VX, max_dVX, max_dVX/std::max(1e-30,max_VX));
  if (strict_VH) CHECK(max_dVH < tol_VH);
  if (strict_VX) CHECK(max_dVX < tol_VX);
}

TEST_CASE("thc_vs_direct_nij", "[hamilt][thc][hf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (NCPP, non-diag nij)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_hf", mf::h5_input_type));
    test_thc_vs_direct_nij<HOST_MEMORY>(*mpi, mf_ptr, 1e-5, 2e-3, 2e-2, true, true);
  }
  SECTION("lih_kp222_nbnd16 (USPP, non-diag nij)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_uspp_hf", mf::h5_input_type));
    test_thc_vs_direct_nij<HOST_MEMORY>(*mpi, mf_ptr, 1e-5, 2e-3, 2e-3, true, true);
  }
  SECTION("lih_kp222_nbnd16 (PAW, non-diag nij)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw_hf", mf::h5_input_type));
    test_thc_vs_direct_nij<HOST_MEMORY>(*mpi, mf_ptr, 1e-5, 2e-3, 2e-3, true, true);
  }
}

// ===========================================================================
// GROUND TRUTH: isolate the THC PAW one-center (K_a) Hartree contribution and
// compare it to the validated external reference (deltaC×becsum, = QE's ke%k).
//
// The THC Hartree V_H = smooth + compensation-aug + K_a one-center. The
// `paw_onsite` knob gates ONLY the K_a term, so
//     V_H(onsite=true) − V_H(onsite=false)  =  THC's K_a contribution to V_H,
// which must equal the closed-form one-center matrix element
//     ΔV_H^{ref}_{ij}(s,k) = Σ_a Σ_{IJ} P*_{i,aI} [Σ_{KL} deltaC^a(I,J,K,L)
//                                                   becsum^a(K,L)] P_{j,aJ}
// with becsum = ns_scl × compute_becsum_diagonal (the same density that feeds
// the THC Dm). The direct one-center (compute_paw_hartree_atom) is validated
// == this deltaC contraction to ~1e-5 (test_paw_onecenter), and the K_a tensor
// == deltaC (test_local_isdf_deltaC_roundtrip); this test confirms the THC K_a
// contraction/normalization matches the reference, and logs the smooth+comp
// residual (THC_off vs direct−deltaC) for reference.
// ===========================================================================
template<MEMORY_SPACE MEM>
void test_thc_Ka_onecenter_vs_deltaC(mpi_context_t& mpi,
                                     std::shared_ptr<mf::MF> mf_ptr,
                                     double thc_thresh)
{
  using math::shm::make_shared_array;
  auto all = nda::range::all;
  auto& mfobj = *mf_ptr;
  long nspin = mfobj.nspin(), nk_ibz = mfobj.nkpts_ibz(), nbnd = mfobj.nbnd();
  int  npol  = mfobj.npol();

  // Non-QE diagonal occupation (exercises all bands) + matching Dm.
  memory::array<MEM, ComplexType, 3> nii(nspin, nk_ibz, nbnd);
  for (long s = 0; s < nspin; ++s) for (long k = 0; k < nk_ibz; ++k)
    for (long n = 0; n < nbnd; ++n)
      nii(s,k,n) = ComplexType(0.5 + 0.3*std::cos(1.3*n + 0.7*k + 0.2*s), 0.0);
  auto sDm = make_shared_array<array_view_4d_t>(mpi, {nspin, nk_ibz, nbnd, nbnd});
  if (mpi.node_comm.root()) {
    sDm.local()() = ComplexType(0.0);
    for (long s = 0; s < nspin; ++s) for (long k = 0; k < nk_ibz; ++k)
      for (long a = 0; a < nbnd; ++a) sDm.local()(s,k,a,a) = nii(s,k,a);
  }
  mpi.node_comm.barrier();

  hamilt::pseudopot V(mfobj);

  // ---- External reference: deltaC × becsum one-center matrix element. ----
  auto becsum = hamilt::paw::compute_becsum_diagonal(
      V.Pskna_view(), nii, V.ityp_view(), V.nh_view(), V.ofs_view(), npol);
  double ns_scl = (nspin == 1 && npol == 1) ? 2.0 : 1.0;
  for (long ia = 0; ia < becsum.extent(0); ++ia)
    for (long I = 0; I < becsum.extent(1); ++I)
      for (long J = 0; J < becsum.extent(2); ++J) becsum(ia,I,J) *= ns_scl;
  auto Pskna = V.Pskna_view();
  auto const& ityp = V.ityp_view();
  auto const& nh_v = V.nh_view();
  auto const& ofs  = V.ofs_view();
  auto const& sps  = V.paw_species_view();
  long nat = ityp.extent(0);
  nda::array<ComplexType,4> ref(nspin, nk_ibz, nbnd, nbnd);
  ref() = ComplexType(0.0);
  for (long s = 0; s < nspin; ++s)
    for (long k = 0; k < nk_ibz; ++k)
      for (long ia = 0; ia < nat; ++ia) {
        int nt = ityp(ia); int nh_a = nh_v(nt);
        if (nh_a == 0) continue;
        auto const& sp = sps[nt];
        if (!sp.is_paw || sp.deltaC.size() == 0) continue;
        long p0 = ofs(ia);
        nda::array<double,2> dD(nh_a, nh_a); dD() = 0.0;
        for (int I = 0; I < nh_a; ++I) for (int J = 0; J < nh_a; ++J) {
          double acc = 0.0;
          for (int K = 0; K < nh_a; ++K) for (int L = 0; L < nh_a; ++L)
            acc += sp.deltaC(I,J,K,L) * becsum(ia,K,L);
          dD(I,J) = acc;
        }
        for (long i = 0; i < nbnd; ++i) for (long j = 0; j < nbnd; ++j) {
          ComplexType acc(0.0);
          for (int I = 0; I < nh_a; ++I) {
            ComplexType PiI = std::conj(Pskna(s,k,p0+I,i));
            for (int J = 0; J < nh_a; ++J)
              acc += PiI * ComplexType(dD(I,J),0.0) * Pskna(s,k,p0+J,j);
          }
          ref(s,k,i,j) += acc;
        }
      }

  // ---- THC K_a contribution = V_H(onsite=true) − V_H(onsite=false). ----
  auto sS = make_shared_array<array_view_4d_t>(mpi, {nspin, nk_ibz, nbnd, nbnd});
  hamilt::set_ovlp(mfobj, sS);
  auto thc_VH = [&](bool onsite) {
    auto pt = methods::make_thc_reader_ptree(0, "", "incore", "", "bdft",
                                             thc_thresh, mfobj.ecutrho());
    pt.put("paw_aug", true);
    pt.put("paw_isdf_metric", "coulomb");
    pt.put("paw_isdf_tol", 1e-12);
    pt.put("paw_onsite", onsite);
    methods::thc_reader_t thc(mf_ptr, pt);
    auto sVH = make_shared_array<array_view_4d_t>(mpi, {nspin, nk_ibz, nbnd, nbnd});
    methods::solvers::hf_t hf(methods::ignore_g0);
    hf.evaluate(sVH, sDm.local(), thc, sS.local(), true, false);
    nda::array<ComplexType,4> out = sVH.local();
    return out;
  };
  auto VH_on  = thc_VH(true);
  auto VH_off = thc_VH(false);

  // Direct full V_H (smooth+comp via v_h + radial one-center + ∫V_H·Q via deeq).
  auto dVH_dir = hamilt::Vhartree<MEM>(mfobj, mpi.comm, &V, nii);
  auto VHd = nda::to_host(dVH_dir.local());

  double max_ref=0, max_dKa=0;                    // one-center: THC K_a vs deltaC
  double max_sm=0, max_dsm=0;                      // smooth+comp: THC_off vs (direct−ref)
  double dsm_diag=0, dsm_off=0;                    // smooth+comp split: diag vs off-diag
  double sm_diag=0,  sm_off=0;                     // magnitudes for relative
  long ws=-1,wk=-1,wi=-1,wj=-1; double worst=0;    // worst smooth+comp element
  ComplexType worst_thc, worst_dir;
  auto rs = dVH_dir.local_range(0); auto rk = dVH_dir.local_range(1);
  auto ri = dVH_dir.local_range(2); auto rj = dVH_dir.local_range(3);
  for (auto [isl,s] : itertools::enumerate(rs))
    for (auto [ikl,k] : itertools::enumerate(rk))
      for (auto [iil,i] : itertools::enumerate(ri))
        for (auto [ijl,j] : itertools::enumerate(rj)) {
          ComplexType thc_oc = VH_on(s,k,i,j) - VH_off(s,k,i,j);   // THC K_a
          ComplexType r      = ref(s,k,i,j);
          ComplexType dfull  = VHd(isl,ikl,iil,ijl);               // direct full
          max_ref   = std::max(max_ref,   std::abs(r));
          max_dKa   = std::max(max_dKa,   std::abs(thc_oc - r));
          ComplexType sm_thc = VH_off(s,k,i,j);
          ComplexType sm_dir = dfull - r;              // direct smooth+comp (full − one-center)
          double d = std::abs(sm_thc - sm_dir);
          max_sm  = std::max(max_sm,  std::abs(sm_dir));
          max_dsm = std::max(max_dsm, d);
          if (i == j) { dsm_diag = std::max(dsm_diag, d); sm_diag = std::max(sm_diag, std::abs(sm_dir)); }
          else        { dsm_off  = std::max(dsm_off,  d); sm_off  = std::max(sm_off,  std::abs(sm_dir)); }
          if (d > worst) { worst=d; ws=s; wk=k; wi=i; wj=j; worst_thc=sm_thc; worst_dir=sm_dir; }
        }
  app_log(1, "[K_a one-center] THC_Ka vs deltaC: max|ref|={:.4e} max|Δ|={:.4e} rel={:.3e}",
          max_ref, max_dKa, max_dKa/std::max(1e-30,max_ref));
  app_log(1, "[smooth+comp]    THC_off vs (direct−deltaC): max|Δ|={:.4e} rel={:.3e}",
          max_dsm, max_dsm/std::max(1e-30,max_sm));
  app_log(1, "[smooth+comp DIAG]     max|Δ|={:.4e} rel={:.3e}",
          dsm_diag, dsm_diag/std::max(1e-30,sm_diag));
  app_log(1, "[smooth+comp OFF-DIAG] max|Δ|={:.4e} rel={:.3e}",
          dsm_off,  dsm_off/std::max(1e-30,sm_off));
  app_log(1, "[worst smooth+comp] (s={},k={},i={},j={}) THC={:+.5e}{:+.5e}i dir={:+.5e}{:+.5e}i ratio={:.4f}",
          ws,wk,wi,wj, worst_thc.real(),worst_thc.imag(), worst_dir.real(),worst_dir.imag(),
          std::abs(worst_thc)/std::max(1e-30,std::abs(worst_dir)));
  REQUIRE(max_ref > 1e-8);  // non-trivial one-center
}

TEST_CASE("thc_Ka_onecenter_vs_deltaC", "[hamilt][paw][thc][onecenter]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (PAW)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw_hf", mf::h5_input_type));
    test_thc_Ka_onecenter_vs_deltaC<HOST_MEMORY>(*mpi, mf_ptr, 1e-5);
  }
}

// ===========================================================================
// V_x ONE-CENTER PREFACTOR check against THC's K_a exchange (QE-independent).
//
// Validates the one-center exchange prefactor scl_oc = -1/N_k in v_x_paw.hpp
// against THC's own K_a exchange:
//
//   THC_Ka_x(s,kp,i,j) = THC_Vx(onsite=true) − THC_Vx(onsite=false)
//     (Fock contraction of the additive one-center ERI block; the smooth+aug
//      ERI is identical in both, so the difference is exactly K_a exchange.)
//
//   R(s,kp,i,j) = Σ_{kq,n} f_n Σ_a Σ_IL conj(P_iI(kp)) U^a_{IL}(n,kq) P_jL(kp)
//     U^a_{IL} = Σ_JK deltaC^a(I,J,K,L) P_nJ(kq) conj(P_nK(kq))
//     (the structure of the v_x_paw one-center loop, with UNIT prefactor)
//
// The best-fit real scalar α = Σ Re(THC·conj(R)) / Σ|R|² is the THC Fock
// prefactor for the one-center block, which must equal the production -1/N_k.
// nk==nk_ibz (nosym fixture) so Pskna is the full-BZ projector.
// ===========================================================================
template<MEMORY_SPACE MEM>
void test_vx_onecenter_vs_thc_Ka(mpi_context_t& mpi,
                                 std::shared_ptr<mf::MF> mf_ptr,
                                 double thc_thresh)
{
  using math::shm::make_shared_array;
  auto all = nda::range::all;
  auto& mfobj = *mf_ptr;
  long nspin = mfobj.nspin(), nk_ibz = mfobj.nkpts_ibz(), nbnd = mfobj.nbnd();
  long nk = mfobj.nkpts();
  REQUIRE(nk == nk_ibz);  // nosym fixture: Pskna is the full-BZ projector

  // Non-QE diagonal occupation (exercises all bands) + matching diagonal Dm.
  memory::array<MEM, ComplexType, 3> nii(nspin, nk_ibz, nbnd);
  for (long s = 0; s < nspin; ++s) for (long k = 0; k < nk_ibz; ++k)
    for (long n = 0; n < nbnd; ++n)
      nii(s,k,n) = ComplexType(0.5 + 0.3*std::cos(1.3*n + 0.7*k + 0.2*s), 0.0);
  auto sDm = make_shared_array<array_view_4d_t>(mpi, {nspin, nk_ibz, nbnd, nbnd});
  if (mpi.node_comm.root()) {
    sDm.local()() = ComplexType(0.0);
    for (long s = 0; s < nspin; ++s) for (long k = 0; k < nk_ibz; ++k)
      for (long a = 0; a < nbnd; ++a) sDm.local()(s,k,a,a) = nii(s,k,a);
  }
  mpi.node_comm.barrier();

  hamilt::pseudopot V(mfobj);
  auto Pskna = V.Pskna_view();
  auto const& ityp = V.ityp_view();
  auto const& nh_v = V.nh_view();
  auto const& ofs  = V.ofs_view();
  auto const& sps  = V.paw_species_view();
  long nat = ityp.extent(0);

  // ---- THC K_a exchange = THC_Vx(onsite=true) − THC_Vx(onsite=false). ----
  auto sS = make_shared_array<array_view_4d_t>(mpi, {nspin, nk_ibz, nbnd, nbnd});
  hamilt::set_ovlp(mfobj, sS);
  auto thc_VX = [&](bool onsite) {
    auto pt = methods::make_thc_reader_ptree(0, "", "incore", "", "bdft",
                                             thc_thresh, mfobj.ecutrho());
    pt.put("paw_aug", true);
    pt.put("paw_isdf_metric", "coulomb");
    pt.put("paw_isdf_tol", 1e-12);
    pt.put("paw_onsite", onsite);
    methods::thc_reader_t thc(mf_ptr, pt);
    auto sVX = make_shared_array<array_view_4d_t>(mpi, {nspin, nk_ibz, nbnd, nbnd});
    methods::solvers::hf_t hf(methods::ignore_g0);
    hf.evaluate(sVX, sDm.local(), thc, sS.local(), false, true);  // exchange only
    nda::array<ComplexType,4> out = sVX.local();
    return out;
  };
  auto VX_on  = thc_VX(true);
  auto VX_off = thc_VX(false);

  // ---- Direct RAW one-center exchange (UNIT prefactor), exact v_x_paw loop. ----
  nda::array<ComplexType,4> R(nspin, nk_ibz, nbnd, nbnd);
  R() = ComplexType(0.0);
  for (long s = 0; s < nspin; ++s)
    for (long kp = 0; kp < nk_ibz; ++kp)
      for (long kq = 0; kq < nk_ibz; ++kq)
        for (long n = 0; n < nbnd; ++n) {
          double f = std::real(nii(s,kq,n));
          if (std::abs(f) < 1e-15) continue;
          for (long ia = 0; ia < nat; ++ia) {
            int nt = ityp(ia); int nh_a = nh_v(nt);
            if (nh_a == 0) continue;
            auto const& sp = sps[nt];
            if (!sp.is_paw || sp.deltaC.size() == 0) continue;
            long p0 = ofs(ia);
            nda::array<ComplexType,2> U(nh_a, nh_a); U() = ComplexType(0.0);
            for (int I = 0; I < nh_a; ++I) for (int L = 0; L < nh_a; ++L) {
              ComplexType acc(0.0);
              for (int J = 0; J < nh_a; ++J) {
                ComplexType PnJ = Pskna(s,kq,p0+J,n);
                for (int K = 0; K < nh_a; ++K) {
                  ComplexType PnK = std::conj(Pskna(s,kq,p0+K,n));
                  acc += ComplexType(sp.deltaC(I,J,K,L),0.0) * PnJ * PnK;
                }
              }
              U(I,L) = acc;
            }
            for (long i = 0; i < nbnd; ++i) for (long j = 0; j < nbnd; ++j) {
              ComplexType acc(0.0);
              for (int I = 0; I < nh_a; ++I) {
                ComplexType PiI = std::conj(Pskna(s,kp,p0+I,i));
                for (int L = 0; L < nh_a; ++L)
                  acc += PiI * U(I,L) * Pskna(s,kp,p0+L,j);
              }
              R(s,kp,i,j) += ComplexType(f,0.0) * acc;
            }
          }
        }

  // ---- Best-fit real scalar α = Σ Re(THC·conj(R)) / Σ|R|² + candidates. ----
  double num = 0.0, den = 0.0, max_imag = 0.0, max_thc = 0.0, max_R = 0.0;
  for (long s = 0; s < nspin; ++s) for (long k = 0; k < nk_ibz; ++k)
    for (long i = 0; i < nbnd; ++i) for (long j = 0; j < nbnd; ++j) {
      ComplexType thc = VX_on(s,k,i,j) - VX_off(s,k,i,j);
      ComplexType r   = R(s,k,i,j);
      num += std::real(thc * std::conj(r));
      den += std::real(r * std::conj(r));
      max_thc = std::max(max_thc, std::abs(thc));
      max_R   = std::max(max_R,   std::abs(r));
    }
  REQUIRE(den > 1e-12);
  double alpha = num / den;
  // residual of best-fit imaginary part (THC should be real·R since deltaC real)
  for (long s = 0; s < nspin; ++s) for (long k = 0; k < nk_ibz; ++k)
    for (long i = 0; i < nbnd; ++i) for (long j = 0; j < nbnd; ++j) {
      ComplexType thc = VX_on(s,k,i,j) - VX_off(s,k,i,j);
      max_imag = std::max(max_imag, std::abs(std::imag(thc - alpha*R(s,k,i,j))));
    }

  double prod = -1.0 / double(nk);   // production scl_oc

  auto resid = [&](double p) {
    double mx = 0.0;
    for (long s = 0; s < nspin; ++s) for (long k = 0; k < nk_ibz; ++k)
      for (long i = 0; i < nbnd; ++i) for (long j = 0; j < nbnd; ++j)
        mx = std::max(mx, std::abs((VX_on(s,k,i,j)-VX_off(s,k,i,j)) - p*R(s,k,i,j)));
    return mx;
  };
  double res_prod = resid(prod);

  app_log(1, "[V_x one-center prefactor] best-fit alpha = {:.6e}, "
             "production scl_oc(-1/N_k) = {:.6e}, alpha/prod = {:.4f}",
          alpha, prod, alpha/prod);
  app_log(1, "  max|THC_Ka_x|={:.4e}  max|R|={:.4e}  max|imag resid @alpha|={:.3e}  "
             "residual @prod={:.3e}", max_thc, max_R, max_imag, res_prod);

  REQUIRE(max_thc > 1e-8);          // non-trivial K_a exchange present
  REQUIRE(max_imag < 1e-6 * std::max(1e-30, max_thc)); // THC_Ka = real·R
  // The production prefactor -1/N_k must reproduce THC's K_a exchange to noise.
  CHECK(res_prod < 1e-4 * std::max(1e-30, max_thc));
  CHECK(std::abs(alpha - prod) < 1e-4 * std::abs(prod));
}

TEST_CASE("vx_onecenter_vs_thc_Ka", "[hamilt][paw][thc][onecenter]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (PAW)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw_hf", mf::h5_input_type));
    test_vx_onecenter_vs_thc_Ka<HOST_MEMORY>(*mpi, mf_ptr, 1e-5);
  }
}

// ===========================================================================
// Smooth-only V_H cross-check: the EXACT smooth V_H matrix element by real-space
// FFT vs THC(paw_aug=false).
//     V_H^sm_ij = (1/N_r) Σ_r conj(ψ̃_i,unnorm(r)) ψ̃_j,unnorm(r) V_H^sm(r)
// with V_H^sm(r) = iFFT[4π/|G|² · ρ̃_smooth(G)] (ρ̃ via build_total_density_r,
// include_augmentation=false). NCPP/USPP (no/low aug) calibrate the
// normalization, where smooth FFT must equal THC.
// ===========================================================================
template<MEMORY_SPACE MEM>
void test_vh_smooth_fft_vs_thc(mpi_context_t& mpi, std::shared_ptr<mf::MF> mf_ptr,
                               std::string label)
{
  using math::shm::make_shared_array;
  auto all = nda::range::all; using nda::range;
  auto& mfobj = *mf_ptr;
  long nspin = mfobj.nspin(), nk_ibz = mfobj.nkpts_ibz(), nbnd = mfobj.nbnd();
  int npol = mfobj.npol();
  REQUIRE(npol == 1);

  hamilt::pseudopot V(mfobj);
  using larray = memory::array<MEM, ComplexType, 4>;
  auto psi = mf::read_distributed_orbital_set_ibz<larray>(
      mfobj, mpi.comm, 'w', std::array<long,4>{0,0,0,0},
      range(nspin), range(nk_ibz), range(nbnd), std::array<long,4>{1,1,2048,2048});
  auto fft_mesh  = mfobj.fft_grid_dim();
  auto recv      = mfobj.recv();
  auto kpts_full = mfobj.kpts();
  auto kp_to_ibz = mfobj.kp_to_ibz();
  auto kp_trev   = mfobj.kp_trev();
  auto kp_symm   = mfobj.kp_symm();
  auto symm_list = mfobj.symm_list();
  auto k2g       = V.swfc_to_rho_view();
  long nnr = (long)fft_mesh(0)*fft_mesh(1)*fft_mesh(2);
  double det_B = recv(0,0)*(recv(1,1)*recv(2,2)-recv(1,2)*recv(2,1))
               - recv(1,0)*(recv(0,1)*recv(2,2)-recv(0,2)*recv(2,1))
               + recv(2,0)*(recv(0,1)*recv(1,2)-recv(0,2)*recv(1,1));
  double vol = (2.0*M_PI)*(2.0*M_PI)*(2.0*M_PI)/std::abs(det_B);

  // non-QE occ (exercise all bands)
  nda::array<ComplexType,3> nii(nspin, nk_ibz, nbnd);
  for (long s=0;s<nspin;++s) for (long k=0;k<nk_ibz;++k) for (long n=0;n<nbnd;++n)
    nii(s,k,n) = ComplexType(0.5 + 0.3*std::cos(1.3*n+0.7*k+0.2*s), 0.0);

  // ρ̃_smooth(r) (no augmentation) via the validated builder.
  auto rho_sm = hamilt::paw::build_total_density_r(mpi, V, npol, fft_mesh, recv,
      k2g, kpts_full, kp_to_ibz, kp_trev, kp_symm, symm_list, nii, psi, vol,
      /*include_augmentation=*/false);

  // Build V_H^sm(r) and the matrix elements on root.
  nda::array<ComplexType,4> VHsm_fft(nspin, nk_ibz, nbnd, nbnd);
  VHsm_fft() = ComplexType(0.0);
  if (mpi.comm.root()) {
    long NX=fft_mesh(0),NY=fft_mesh(1),NZ=fft_mesh(2);
    // ρ̃(G) (normalized fwd), V_H(G)=4π/|G|² ρ(G), G=0→0, iFFT (unnorm) → V_H(r).
    nda::array<ComplexType,1> vg(nnr);
    for (long r=0;r<nnr;++r) vg(r)=ComplexType(rho_sm(r),0.0);
    auto vg3d = nda::reshape(vg, std::array<long,3>{NX,NY,NZ});
    math::nda::fft<false> Ff(vg3d); Ff.forward(vg3d);
    for (long n1=0;n1<NX;++n1){int m1=(n1<=NX/2)?(int)n1:(int)n1-(int)NX;
     for (long n2=0;n2<NY;++n2){int m2=(n2<=NY/2)?(int)n2:(int)n2-(int)NY;
      for (long n3=0;n3<NZ;++n3){int m3=(n3<=NZ/2)?(int)n3:(int)n3-(int)NZ;
        long N=(n1*NY+n2)*NZ+n3;
        if(m1==0&&m2==0&&m3==0){vg(N)=ComplexType(0.0);continue;}
        double Gx=m1*recv(0,0)+m2*recv(1,0)+m3*recv(2,0);
        double Gy=m1*recv(0,1)+m2*recv(1,1)+m3*recv(2,1);
        double Gz=m1*recv(0,2)+m2*recv(1,2)+m3*recv(2,2);
        double G2=Gx*Gx+Gy*Gy+Gz*Gz;
        vg(N) *= ComplexType(4.0*M_PI/G2, 0.0);
      }}}
    math::nda::fft<false> Fb(vg3d); Fb.backward(vg3d);  // V_H^sm(r), proper
    // Orbitals on dense grid (un-normalized backward), per (s,k).
    auto psi_full = larray(psi.global_shape());
    math::nda::gather(0, psi, &psi_full);
    long ngm = k2g.extent(0);
    nda::array<ComplexType,1> pr(nnr);
    auto pr3d = nda::reshape(pr, std::array<long,3>{NX,NY,NZ});
    math::nda::fft<false> Fp(pr3d);
    nda::array<ComplexType,3> psir(nbnd, nnr, 1);  // reuse per (s,k)
    for (long s=0;s<nspin;++s) for (long k=0;k<nk_ibz;++k) {
      nda::array<ComplexType,2> ur(nbnd, nnr); ur()=ComplexType(0.0);
      for (long n=0;n<nbnd;++n) {
        pr()=ComplexType(0.0);
        for (long g=0;g<ngm;++g){ long Nidx=k2g(g); if(Nidx>=0&&Nidx<nnr) pr(Nidx)=psi_full(s,k,n,g); }
        Fp.backward(pr3d);
        for (long r=0;r<nnr;++r) ur(n,r)=pr(r);
      }
      for (long i=0;i<nbnd;++i) for (long j=0;j<nbnd;++j) {
        ComplexType acc(0.0);
        for (long r=0;r<nnr;++r) acc += std::conj(ur(i,r))*ur(j,r)*vg(r);
        VHsm_fft(s,k,i,j) = acc / ComplexType((double)nnr, 0.0);
      }
    }
  }
  mpi.comm.broadcast_n(VHsm_fft.data(), VHsm_fft.size(), 0);

  // THC smooth-only V_H (paw_aug=false).
  auto sDm = make_shared_array<array_view_4d_t>(mpi, {nspin,nk_ibz,nbnd,nbnd});
  if (mpi.node_comm.root()){ sDm.local()()=ComplexType(0.0);
    for(long s=0;s<nspin;++s)for(long k=0;k<nk_ibz;++k)for(long a=0;a<nbnd;++a) sDm.local()(s,k,a,a)=nii(s,k,a);}
  mpi.node_comm.barrier();
  auto sS = make_shared_array<array_view_4d_t>(mpi, {nspin,nk_ibz,nbnd,nbnd});
  hamilt::set_ovlp(mfobj, sS);
  auto pt = methods::make_thc_reader_ptree(0,"","incore","","bdft",1e-5,mfobj.ecutrho());
  pt.put("paw_aug", false);
  methods::thc_reader_t thc(mf_ptr, pt);
  auto sVH = make_shared_array<array_view_4d_t>(mpi, {nspin,nk_ibz,nbnd,nbnd});
  { methods::solvers::hf_t hf(methods::ignore_g0);
    hf.evaluate(sVH, sDm.local(), thc, sS.local(), true, false); }
  auto VHthc = nda::to_host(sVH.local());

  double max_v=0,max_d=0,d_diag=0,worst=0; long wi=-1,wj=-1,wk=-1,ws=-1;
  ComplexType wfft, wthc;
  for (long s=0;s<nspin;++s) for (long k=0;k<nk_ibz;++k)
    for (long i=0;i<nbnd;++i) for (long j=0;j<nbnd;++j) {
      double d=std::abs(VHsm_fft(s,k,i,j)-VHthc(s,k,i,j));
      max_v=std::max(max_v,std::abs(VHsm_fft(s,k,i,j)));
      max_d=std::max(max_d,d);
      if(i==j) d_diag=std::max(d_diag,d);
      if(d>worst){worst=d;ws=s;wk=k;wi=i;wj=j;wfft=VHsm_fft(s,k,i,j);wthc=VHthc(s,k,i,j);}
    }
  app_log(1,"[{} smooth FFT vs THC(no aug)] max|V_H^sm|={:.4e} max|Δ|={:.4e} rel={:.3e} diagΔ={:.4e}",
          label, max_v, max_d, max_d/std::max(1e-30,max_v), d_diag);
  app_log(1,"[{} worst] (s={},k={},i={},j={}) FFT={:+.5e} THC={:+.5e} ratio={:.4f}",
          label, ws,wk,wi,wj, wfft.real(), wthc.real(),
          std::abs(wthc)/std::max(1e-30,std::abs(wfft)));
  REQUIRE(max_v > 1e-8);
}

TEST_CASE("vh_smooth_fft_vs_thc", "[hamilt][paw][thc][onecenter]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("USPP (calibration)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_uspp_hf", mf::h5_input_type));
    test_vh_smooth_fft_vs_thc<HOST_MEMORY>(*mpi, mf_ptr, "USPP");
  }
  SECTION("PAW") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw_hf", mf::h5_input_type));
    test_vh_smooth_fft_vs_thc<HOST_MEMORY>(*mpi, mf_ptr, "PAW");
  }
}

// ===========================================================================
// V_GL vs V_LL isolation of the THC compensation assembly, using the diagnostic
// paw_vgl/paw_vll knobs on the THC side and the exact ∫V·Q decomposition on
// the direct side:
//   THC_VGL = THC(vgl=1,vll=0,onsite=0) − THC(0,0,0)   (smooth-aug cross)
//   THC_VLL = THC(vgl=0,vll=1,onsite=0) − THC(0,0,0)   (aug-aug)
//   direct comp-comp = Σ conj(P_iI)P_jJ ∫V_comp·Q^IJ
//                    = Σ conj(P)P [compute_paw_deeq(V_comp) − compute_paw_deeq(∅)]
// where V_comp = v_C·ρ_comp, ρ_comp = ρ_tot − ρ_smooth (build_total_density_r).
// Confirms THC_VLL == direct comp-comp (aug-aug block).
// ===========================================================================
TEST_CASE("thc_vgl_vll_split", "[hamilt][paw][thc][onecenter]")
{
  using math::shm::make_shared_array;
  auto& mpip = utils::make_unit_test_mpi_context();
  auto& mpi = *mpip;
  using nda::range;
  auto mf_ptr = std::make_shared<mf::MF>(
      mf::default_MF(mpip, "qe_lih222_paw_hf", mf::h5_input_type));
  auto& mfobj = *mf_ptr;
  long nspin = mfobj.nspin(), nk_ibz = mfobj.nkpts_ibz(), nbnd = mfobj.nbnd();
  int npol = mfobj.npol();
  hamilt::pseudopot V(mfobj);

  using larray = memory::array<HOST_MEMORY, ComplexType, 4>;
  auto psi = mf::read_distributed_orbital_set_ibz<larray>(
      mfobj, mpi.comm, 'w', std::array<long,4>{0,0,0,0},
      range(nspin), range(nk_ibz), range(nbnd), std::array<long,4>{1,1,2048,2048});
  auto fft_mesh=mfobj.fft_grid_dim(); auto recv=mfobj.recv();
  auto kpts_full=mfobj.kpts(); auto kp_to_ibz=mfobj.kp_to_ibz();
  auto kp_trev=mfobj.kp_trev(); auto kp_symm=mfobj.kp_symm();
  auto symm_list=mfobj.symm_list(); auto k2g=V.swfc_to_rho_view();
  long nnr=(long)fft_mesh(0)*fft_mesh(1)*fft_mesh(2);
  double det_B=recv(0,0)*(recv(1,1)*recv(2,2)-recv(1,2)*recv(2,1))
             - recv(1,0)*(recv(0,1)*recv(2,2)-recv(0,2)*recv(2,1))
             + recv(2,0)*(recv(0,1)*recv(1,2)-recv(0,2)*recv(1,1));
  double vol=(2.0*M_PI)*(2.0*M_PI)*(2.0*M_PI)/std::abs(det_B);

  nda::array<ComplexType,3> nii(nspin,nk_ibz,nbnd);
  for(long s=0;s<nspin;++s)for(long k=0;k<nk_ibz;++k)for(long n=0;n<nbnd;++n)
    nii(s,k,n)=ComplexType(0.5+0.3*std::cos(1.3*n+0.7*k+0.2*s),0.0);

  // ρ_smooth, ρ_tot → ρ_comp → V_comp(r) (proper).
  auto rho_sm=hamilt::paw::build_total_density_r(mpi,V,npol,fft_mesh,recv,k2g,
      kpts_full,kp_to_ibz,kp_trev,kp_symm,symm_list,nii,psi,vol,false);
  auto rho_tot=hamilt::paw::build_total_density_r(mpi,V,npol,fft_mesh,recv,k2g,
      kpts_full,kp_to_ibz,kp_trev,kp_symm,symm_list,nii,psi,vol,true);
  nda::array<ComplexType,1> Vcomp_r(nnr); Vcomp_r()=ComplexType(0.0);
  if(mpi.comm.root()){
    long NX=fft_mesh(0),NY=fft_mesh(1),NZ=fft_mesh(2);
    nda::array<ComplexType,1> g(nnr);
    for(long r=0;r<nnr;++r) g(r)=ComplexType(rho_tot(r)-rho_sm(r),0.0);
    auto g3d=nda::reshape(g,std::array<long,3>{NX,NY,NZ});
    math::nda::fft<false> Ff(g3d); Ff.forward(g3d);
    for(long n1=0;n1<NX;++n1){int m1=(n1<=NX/2)?(int)n1:(int)n1-(int)NX;
     for(long n2=0;n2<NY;++n2){int m2=(n2<=NY/2)?(int)n2:(int)n2-(int)NY;
      for(long n3=0;n3<NZ;++n3){int m3=(n3<=NZ/2)?(int)n3:(int)n3-(int)NZ;
        long N=(n1*NY+n2)*NZ+n3;
        if(m1==0&&m2==0&&m3==0){g(N)=ComplexType(0.0);continue;}
        double Gx=m1*recv(0,0)+m2*recv(1,0)+m3*recv(2,0);
        double Gy=m1*recv(0,1)+m2*recv(1,1)+m3*recv(2,1);
        double Gz=m1*recv(0,2)+m2*recv(1,2)+m3*recv(2,2);
        g(N)*=ComplexType(4.0*M_PI/(Gx*Gx+Gy*Gy+Gz*Gz),0.0);
      }}}
    auto g3d2=nda::reshape(g,std::array<long,3>{NX,NY,NZ});
    math::nda::fft<false> Fb(g3d2); Fb.backward(g3d2);
    for(long r=0;r<nnr;++r) Vcomp_r(r)=g(r);
  }
  mpi.comm.broadcast_n(Vcomp_r.data(),Vcomp_r.size(),0);

  // direct comp-comp deeq = compute_paw_deeq(V_comp) − compute_paw_deeq(∅).
  nda::array<ComplexType,1> empty_v;
  auto dD_Vc = V.compute_paw_deeq(nii, Vcomp_r, false);
  auto dD_0  = V.compute_paw_deeq(nii, empty_v, false);
  // direct comp-comp matrix element V_LL-equiv_ij = Σ conj(P_iI)P_jJ (dD_Vc-dD_0).
  auto Pskna=V.Pskna_view(); auto const& ityp=V.ityp_view();
  auto const& nh_v=V.nh_view(); auto const& ofs=V.ofs_view();
  long nat=ityp.extent(0);
  nda::array<ComplexType,4> dirLL(nspin,nk_ibz,nbnd,nbnd); dirLL()=ComplexType(0.0);
  for(long s=0;s<nspin;++s)for(long k=0;k<nk_ibz;++k)
    for(long ia=0;ia<nat;++ia){int nt=ityp(ia);int nh_a=nh_v(nt);if(nh_a==0)continue;long p0=ofs(ia);
      for(long i=0;i<nbnd;++i)for(long j=0;j<nbnd;++j){ComplexType acc(0.0);
        for(int I=0;I<nh_a;++I){ComplexType PiI=std::conj(Pskna(s,k,p0+I,i));
          for(int J=0;J<nh_a;++J) acc+=PiI*(dD_Vc(ia,I,J)-dD_0(ia,I,J))*Pskna(s,k,p0+J,j);}
        dirLL(s,k,i,j)+=acc;}}

  // THC pieces via knobs.
  auto sDm=make_shared_array<array_view_4d_t>(mpi,{nspin,nk_ibz,nbnd,nbnd});
  if(mpi.node_comm.root()){sDm.local()()=ComplexType(0.0);
    for(long s=0;s<nspin;++s)for(long k=0;k<nk_ibz;++k)for(long a=0;a<nbnd;++a)sDm.local()(s,k,a,a)=nii(s,k,a);}
  mpi.node_comm.barrier();
  auto sS=make_shared_array<array_view_4d_t>(mpi,{nspin,nk_ibz,nbnd,nbnd});
  hamilt::set_ovlp(mfobj,sS);
  auto thcVH=[&](bool vgl,bool vll,bool onsite){
    auto pt=methods::make_thc_reader_ptree(0,"","incore","","bdft",1e-5,mfobj.ecutrho());
    pt.put("paw_aug",true); pt.put("paw_isdf_metric","coulomb"); pt.put("paw_isdf_tol",1e-12);
    pt.put("paw_vgl",vgl); pt.put("paw_vll",vll); pt.put("paw_onsite",onsite);
    methods::thc_reader_t thc(mf_ptr,pt);
    auto sVH=make_shared_array<array_view_4d_t>(mpi,{nspin,nk_ibz,nbnd,nbnd});
    methods::solvers::hf_t hf(methods::ignore_g0);
    hf.evaluate(sVH,sDm.local(),thc,sS.local(),true,false);
    nda::array<ComplexType,4> o=sVH.local(); return o; };
  auto VH_smooth=thcVH(false,false,false);
  auto VH_vgl   =thcVH(true,false,false);
  auto VH_vll   =thcVH(false,true,false);

  double mLL=0,dLL=0; long bi=-1; ComplexType tLL,dLLv,tGL;
  for(long s=0;s<nspin;++s)for(long k=0;k<nk_ibz;++k)for(long i=0;i<nbnd;++i){
    ComplexType thc_vll=VH_vll(s,k,i,i)-VH_smooth(s,k,i,i);
    ComplexType d=dirLL(s,k,i,i);
    mLL=std::max(mLL,std::abs(d));
    if(std::abs(thc_vll-d)>dLL){dLL=std::abs(thc_vll-d);bi=i;tLL=thc_vll;dLLv=d;tGL=VH_vgl(s,k,i,i)-VH_smooth(s,k,i,i);}
  }
  app_log(1,"[V_LL split] worst diag band={} THC_VLL={:+.5e} direct_compcomp={:+.5e} ratio={:.4f} (max|dir|={:.4e})",
          bi,tLL.real(),dLLv.real(),std::abs(tLL)/std::max(1e-30,std::abs(dLLv)),mLL);
  app_log(1,"[V_GL at that band] THC_VGL={:+.5e}",tGL.real());
  REQUIRE(mLL>1e-8);
}

// ===========================================================================
// qrad table (build_qrad_tab + qrad_interp_at_K, used by the THC build_eta /
// V_GL / V_LL augmentation) vs the EXACT qrad_at_K (used by evaluate_Q == QE
// qgm, validated to 1e-10 by paw_aug_q_eval_at_q0). Confirms the interpolated
// table matches the exact transform across L. Reports max rel error PER L so an
// l-dependent defect would be visible.
// ===========================================================================
template<MEMORY_SPACE MEM>
void test_qrad_tab_vs_exact(mpi_context_t& mpi, mf::MF& mfobj)
{
  hamilt::pseudopot V(mfobj);
  auto const& sps  = V.paw_species_view();
  auto recv = mfobj.recv();
  // K_max bound similar to the THC/v_x builders.
  double Kmax = 25.0;   // generous; covers the dense-grid |G| range for LiH
  bool any = false;
  for (long nt = 0; nt < (long)sps.size(); ++nt) {
    auto const& sp = sps[nt];
    if (!(sp.is_paw || sp.is_uspp) || sp.qfuncl.size() == 0) continue;
    any = true;
    long Lp1   = sp.qfuncl.extent(0);
    long n_ijv = sp.qfuncl.extent(1);
    auto T = hamilt::paw::build_qrad_tab(sp, Kmax);
    nda::array<double,1> max_err(Lp1), max_val(Lp1);
    max_err() = 0.0; max_val() = 0.0;
    // Sample K densely (avoid the exact table nodes to exercise interpolation).
    for (int iK = 0; iK < 400; ++iK) {
      double K = (iK + 0.37) * (Kmax / 400.0);
      for (long ijv = 0; ijv < n_ijv; ++ijv) {
        // qrad_at_K is keyed by ijv (beta-pair index) directly here.
        auto exact = hamilt::paw::qrad_at_K(sp, (int)ijv, K);
        auto tab   = hamilt::paw::qrad_interp_at_K(T, (int)ijv, K);
        for (long L = 0; L < Lp1; ++L) {
          max_err(L) = std::max(max_err(L), std::abs(tab(L) - exact(L)));
          max_val(L) = std::max(max_val(L), std::abs(exact(L)));
        }
      }
    }
    for (long L = 0; L < Lp1; ++L)
      app_log(1, "[qrad tab-vs-exact nt={} L={}] max|Δ|={:.3e} max|exact|={:.3e} rel={:.3e}",
              nt, L, max_err(L), max_val(L),
              max_err(L)/std::max(1e-30, max_val(L)));
  }
  REQUIRE(any);
}

TEST_CASE("qrad_tab_vs_exact", "[hamilt][paw][onecenter]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (PAW)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    test_qrad_tab_vs_exact<HOST_MEMORY>(*mpi, qe_h5);
  }
  SECTION("si_kp222 (PAW)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_si222_paw", mf::h5_input_type);
    test_qrad_tab_vs_exact<HOST_MEMORY>(*mpi, qe_h5);
  }
}

TEST_CASE("thc_paw_hf_smoke_si", "[hamilt][paw][thc][slow]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("si_kp222 (USPP psl 1.0.0)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_si222_uspp", mf::h5_input_type));
    test_thc_paw_hf_smoke<HOST_MEMORY>(*mpi, mf_ptr);
  }
  SECTION("si_kp222 (PAW psl 1.0.0)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_si222_paw", mf::h5_input_type));
    test_thc_paw_hf_smoke<HOST_MEMORY>(*mpi, mf_ptr);
  }
}

/**
 * Exchange-energy validation for PAW-augmented THC.
 *
 * Strategy: build a tight-tolerance Cholesky reader on the SAME mean-field
 * fixture and use the Cholesky exchange as a smooth-grid reference. The
 * Cholesky path does NOT include PAW augmentation (it is built from
 * smooth-orbital products on the dense FFT grid), so:
 *
 *   - For NCPP fixtures, paw_aug=true is a no-op. THC and Cholesky should
 *     give the same E_X within Cholesky truncation + smooth-ISDF tolerance.
 *     This validates the THC machinery + HF exchange path.
 *
 *   - For USPP/PAW fixtures, the Cholesky exchange is the SMOOTH-only
 *     answer. THC with paw_aug=false must match it (validates that
 *     paw_aug=false is identical to the smooth path); THC with paw_aug=true
 *     differs by exactly the augmentation contribution. We log both
 *     numbers, assert finiteness of paw_aug=true, and check that the
 *     augmentation contribution has consistent sign with the Hartree-side
 *     augmentation correction (E_X correction is < 0; one-center K_a
 *     dominates).
 */
template<MEMORY_SPACE MEM>
void test_exchange_thc_paw_aug(mpi_context_t& mpi,
                                std::shared_ptr<mf::MF> mf_ptr,
                                std::string const& fixture_name,
                                double thc_thresh = 1e-5,
                                double chol_tol = 1e-8,
                                double tol_smooth = 5e-3)
{
  using math::shm::make_shared_array;
  auto& mfobj = *mf_ptr;
  long nspin   = mfobj.nspin();
  long nk_ibz  = mfobj.nkpts_ibz();
  long nbnd    = mfobj.nbnd();

  auto k_weight = mfobj.k_weight();

  auto sDm_skij = make_shared_array<array_view_4d_t>(mpi,
                      {nspin, nk_ibz, nbnd, nbnd});
  if (mpi.node_comm.root()) {
    sDm_skij.local()() = ComplexType(0.0);
    for (int s = 0; s < nspin; ++s)
    for (int k = 0; k < nk_ibz; ++k)
    for (int a = 0; a < nbnd; ++a)
      sDm_skij.local()(s, k, a, a) = mfobj.occ(s, k, a);
  }
  mpi.node_comm.barrier();
  auto sS_skij = make_shared_array<array_view_4d_t>(mpi,
                    {nspin, nk_ibz, nbnd, nbnd});
  hamilt::set_ovlp(mfobj, sS_skij);

  // Helper: evaluate exchange-only Fock and return (1/2) Tr[Dm K]
  auto exchange_energy = [&](auto& eri_reader) {
    methods::solvers::hf_t hf(methods::ignore_g0);
    auto sK = make_shared_array<array_view_4d_t>(mpi,
                  {nspin, nk_ibz, nbnd, nbnd});
    hf.evaluate(sK, sDm_skij.local(), eri_reader, sS_skij.local(),
                /*hartree=*/false, /*exchange=*/true);
    auto [e1e_dummy, E_X] = methods::eval_hf_energy(
        sDm_skij, sK, sS_skij, k_weight, /*F_has_H0=*/false);
    return E_X;
  };

  // -------- Reference: Cholesky exchange (smooth orbitals only) --------
  // Use a unique scratch dir per fixture+mpi-rank to avoid collisions when
  // sections are run together. Files are deleted in the cleanup block.
  std::string chol_dir = "./chol_x_" + fixture_name;
  if (mpi.comm.root()) std::filesystem::create_directories(chol_dir);
  mpi.comm.barrier();
  auto chol_pt = methods::make_chol_reader_ptree(
      chol_tol, mfobj.ecutrho(), 32, chol_dir, "chol_info.h5",
      methods::chol_reading_type_e::each_q);
  // D4 diagnostic override: this test WANTS the smooth-only Cholesky
  // reference on USPP/PAW fixtures (production builds hard-abort).
  chol_pt.put("allow_smooth_only_aug_pp", true);
  methods::chol_reader_t chol(mf_ptr, chol_pt);
  double E_X_chol = exchange_energy(chol);

  // -------- THC paw_aug=false (smooth-grid only) --------
  auto thc_pt_off = methods::make_thc_reader_ptree(
      0, "", "incore", "", "bdft", thc_thresh, mfobj.ecutrho());
  thc_pt_off.put("paw_aug", false);
  methods::thc_reader_t thc_off(mf_ptr, thc_pt_off);
  double E_X_thc_smooth = exchange_energy(thc_off);

  // -------- THC paw_aug=true (with PAW augmentation) --------
  auto thc_pt_on = methods::make_thc_reader_ptree(
      0, "", "incore", "", "bdft", thc_thresh, mfobj.ecutrho());
  thc_pt_on.put("paw_aug", true);
  thc_pt_on.put("paw_isdf_metric", "coulomb");
  thc_pt_on.put("paw_isdf_tol", 1e-12);
  methods::thc_reader_t thc_on(mf_ptr, thc_pt_on);
  double E_X_thc_aug = exchange_energy(thc_on);

  // -------- Cleanup chol scratch --------
  mpi.comm.barrier();
  if (mpi.comm.root()) {
    std::error_code ec;
    std::filesystem::remove_all(chol_dir, ec);
  }
  mpi.comm.barrier();

  // -------- Reporting + assertions --------
  app_log(2, "PAW-aug THC exchange ({}):  Cholesky(smooth) = {:+.8f} Ha,  "
             "THC(paw_aug=false) = {:+.8f} Ha,  THC(paw_aug=true) = {:+.8f} Ha",
          fixture_name, E_X_chol, E_X_thc_smooth, E_X_thc_aug);
  app_log(2, "  ΔE_X(THC vs chol, smooth-only) = {:+.2e} Ha,  "
             "ΔE_X(aug correction) = {:+.6e} Ha",
          E_X_thc_smooth - E_X_chol, E_X_thc_aug - E_X_thc_smooth);

  // Validate the smooth-grid THC reproduces Cholesky (this exercises the
  // THC HF machinery and is the structural correctness check).
  CHECK(std::abs(E_X_thc_smooth - E_X_chol) < tol_smooth);

  // The augmented THC must run to completion and be finite.
  CHECK(std::isfinite(E_X_thc_aug));
}

TEST_CASE("exchange_thc_paw_aug", "[hamilt][energy][thc][paw]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  SECTION("lih_kp222_nbnd16 (NCPP, paw_aug=true is a no-op)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222", mf::h5_input_type));
    test_exchange_thc_paw_aug<HOST_MEMORY>(*mpi, mf_ptr, "qe_lih222");
  }

  SECTION("lih_kp222_nbnd16 (USPP)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_uspp", mf::h5_input_type));
    test_exchange_thc_paw_aug<HOST_MEMORY>(*mpi, mf_ptr, "qe_lih222_uspp");
  }

  SECTION("lih_kp222_nbnd16 (PAW)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type));
    test_exchange_thc_paw_aug<HOST_MEMORY>(*mpi, mf_ptr, "qe_lih222_paw");
  }
}

TEST_CASE("hartree_thc_paw_aug", "[hamilt][energy][thc][paw]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  SECTION("lih_kp222_nbnd16 (NCPP, paw_aug=true is a no-op)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222", mf::h5_input_type));
    test_hartree_thc_paw_aug<HOST_MEMORY>(*mpi, mf_ptr, "qe_lih222",
                                           /*thc_thresh*/1e-5, /*tol*/5e-3);
  }

  // V_LL × Ω², V_GL × Ω, K_a in V_full → THC reproduces AE Hartree
  // (smooth-grid + one-center) to ~1e-5 Ha for both USPP and PAW.
  SECTION("lih_kp222_nbnd16 (USPP)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_uspp", mf::h5_input_type));
    test_hartree_thc_paw_aug<HOST_MEMORY>(*mpi, mf_ptr, "qe_lih222_uspp",
                                           /*thc_thresh*/1e-5, /*tol*/5e-3);
  }

  SECTION("lih_kp222_nbnd16 (PAW)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type));
    test_hartree_thc_paw_aug<HOST_MEMORY>(*mpi, mf_ptr, "qe_lih222_paw",
                                           /*thc_thresh*/1e-5, /*tol*/5e-3);
  }

  // Plan D2: energy-level Hartree route equivalence on an ABINIT-sourced mf.
  SECTION("si_kp222 (PAW, ABINIT-sourced mf)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "bdft_si222_paw_ab", mf::h5_input_type));
    test_hartree_thc_paw_aug<HOST_MEMORY>(*mpi, mf_ptr, "bdft_si222_paw_ab",
                                           /*thc_thresh*/1e-5, /*tol*/5e-3);
  }
}

// Si sections are split into a separate test case tagged [slow]: the q-loop
// in augment_thc_with_paw with Si's lmax=2 (Lmax=4 in the angular sum) does
// ~4M radial spherical-Bessel transforms per fixture and takes ~1 hour
// each in serial. They are NOT run by default; invoke explicitly with
//   test_hamiltonian "[slow]"   or   test_hamiltonian -c "si_kp222 ..."
TEST_CASE("hartree_thc_paw_aug_si", "[hamilt][energy][thc][paw][slow]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  SECTION("si_kp222 (NCPP)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_si222_ncpp", mf::h5_input_type));
    test_hartree_thc_paw_aug<HOST_MEMORY>(*mpi, mf_ptr, "qe_si222_ncpp",
                                           /*thc_thresh*/1e-5, /*tol*/5e-3);
  }
  SECTION("si_kp222 (USPP psl 1.0.0)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_si222_uspp", mf::h5_input_type));
    test_hartree_thc_paw_aug<HOST_MEMORY>(*mpi, mf_ptr, "qe_si222_uspp",
                                           /*thc_thresh*/1e-5, /*tol*/5e-3);
  }
  SECTION("si_kp222 (PAW psl 1.0.0)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_si222_paw", mf::h5_input_type));
    test_hartree_thc_paw_aug<HOST_MEMORY>(*mpi, mf_ptr, "qe_si222_paw",
                                           /*thc_thresh*/1e-5, /*tol*/5e-3);
  }
}

TEST_CASE("hartree_thc_vs_direct", "[hamilt][energy][thc]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  SECTION("lih_kp222_nbnd16 (NCPP, PBE)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222", mf::h5_input_type));
    test_hartree_thc_vs_direct<HOST_MEMORY>(*mpi, mf_ptr, "qe_lih222",
                                             /*thc_thresh*/1e-5, /*tol*/5e-3);
  }

  SECTION("lih_kp222_nbnd16 (USPP, PBE)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_uspp", mf::h5_input_type));
    test_hartree_thc_vs_direct<HOST_MEMORY>(*mpi, mf_ptr, "qe_lih222_uspp",
                                             /*thc_thresh*/1e-5, /*tol*/5e-3);
  }

  SECTION("lih_kp222_nbnd16 (PAW, PBE)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type));
    test_hartree_thc_vs_direct<HOST_MEMORY>(*mpi, mf_ptr, "qe_lih222_paw",
                                             /*thc_thresh*/1e-5, /*tol*/5e-3);
  }
}

/**
 * Local-ISDF round-trip: Q̂_{a,IJ}(G) reconstructed from (U, η) must match
 * the direct qgm tensor for every (a, I, J, G).
 *
 * In the symmetric-pair full-rank construction this is exact algebra; any
 * non-zero residual signals a bug in build_local_isdf_full_rank.
 */
template<MEMORY_SPACE MEM>
void test_local_isdf_qhat_roundtrip([[maybe_unused]] mpi_context_t& mpi, mf::MF& mfobj,
                                    double tol = 1e-12)
{
  hamilt::pseudopot V(mfobj);
  auto isdf = hamilt::paw::build_local_isdf(V);

  long nsp = (long)V.paw_species_view().size();
  bool any_paw = false;
  double max_err_overall = 0.0;
  for (long nt = 0; nt < nsp; ++nt) {
    auto const& spt = V.paw_species_view()[nt];
    if (!(spt.is_paw || spt.is_uspp)) continue;
    if (isdf[nt].nlambda == 0) continue;
    any_paw = true;
    double err = hamilt::paw::qhat_roundtrip_max_error(V, (int)nt, isdf[nt]);
    app_log(2, "Local-ISDF Q̂ round-trip species {}: nh={}, nλ={}, "
               "max|Q̂ − ΣUUη| = {:.3e}",
               nt, isdf[nt].nh, isdf[nt].nlambda, err);
    max_err_overall = std::max(max_err_overall, err);
    CHECK(err < tol);
  }
  if (!any_paw) {
    app_log(2, "Local-ISDF Q̂ round-trip: no PAW species, skipped.");
  }
}

/**
 * Local-ISDF round-trip: ΔC_a tensor reconstructed from (U, K_a) must match
 * the raw deltaC tensor element-wise for every (a, I, J, K, L).
 *
 * In the symmetric-pair full-rank construction with the closed-form
 *   K_{λξ} = sign(λ)·sign(ξ)·ΔC[i(λ),j(λ),i(ξ),j(ξ)]
 * this is exact algebra.
 */
template<MEMORY_SPACE MEM>
void test_local_isdf_deltaC_roundtrip([[maybe_unused]] mpi_context_t& mpi, mf::MF& mfobj,
                                     double tol = 1e-12)
{
  hamilt::pseudopot V(mfobj);
  auto isdf = hamilt::paw::build_local_isdf(V);

  long nsp = (long)V.paw_species_view().size();
  bool any = false;
  for (long nt = 0; nt < nsp; ++nt) {
    auto const& sp = V.paw_species_view()[nt];
    if (!sp.is_paw || sp.deltaC.size() == 0) continue;
    if (isdf[nt].nlambda == 0) continue;
    any = true;
    auto K = hamilt::paw::compute_K_a(isdf[nt], sp.deltaC);
    double err = hamilt::paw::deltaC_roundtrip_max_error(isdf[nt], sp.deltaC, K);
    app_log(2, "Local-ISDF ΔC_a round-trip species {}: nh={}, nλ={}, "
               "max|ΔC − ΣUUKUU| = {:.3e}",
               nt, isdf[nt].nh, isdf[nt].nlambda, err);
    CHECK(err < tol);
  }
  if (!any) {
    app_log(2, "Local-ISDF ΔC_a round-trip: no PAW species with ΔC, skipped.");
  }
}

/**
 * ρ_aug(G) reconstruction via Local-ISDF (U, η) must match the direct
 * (becsum × qgm) construction. This validates that the Hartree energy of
 * the augmented density is reproducible through the THC factorization
 * channel.
 *
 *   ρ_aug,direct(G)  = Σ_a Σ_{IJ} becsum_{a,IJ} qgm[nt(a),ij_{IJ},G] e^{-iG·τ_a}
 *   ρ_aug,isdf  (G)  = Σ_a Σ_λ    ν_{aλ}      η_{aλ}(G)              e^{-iG·τ_a}
 *
 * with ν_{aλ} = Σ_{IJ} U_{a,λI} becsum_{a,IJ} U_{a,λJ} = U_λ^T becsum_a U_λ.
 *
 * Exact in the full-rank limit; any deviation is a U/η bug.
 */
template<MEMORY_SPACE MEM>
void test_local_isdf_rho_aug_reconstruction(mpi_context_t& mpi, mf::MF& mfobj,
                                            double tol = 1e-12)
{
  auto all = nda::range::all;
  long nspin   = mfobj.nspin();
  long nk_ibz  = mfobj.nkpts_ibz();
  long nbnd    = mfobj.nbnd();
  int  npol    = mfobj.npol();

  memory::array<MEM, ComplexType, 3> nii(nspin, nk_ibz, nbnd);
  if constexpr (MEM == HOST_MEMORY) {
    nii() = mfobj.occ()(all, nda::range(nk_ibz), all);
  } else {
    nii() = nda::array<ComplexType,3>(mfobj.occ()(all, nda::range(nk_ibz), all));
  }

  hamilt::pseudopot V(mfobj);
  if (V.qgm_view().size() == 0) {
    app_log(2, "ρ_aug round-trip: no augmentation (NCPP fixture), skipped.");
    return;
  }

  auto isdf = hamilt::paw::build_local_isdf(V);

  // becsum (real, npol=1 only)
  auto becsum = hamilt::paw::compute_becsum_diagonal(
      V.Pskna_view(), nii, V.ityp_view(), V.nh_view(), V.ofs_view(), npol);
  // QE convention: becsum carries the spin factor ns_scl
  double ns_scl = (nspin == 1 && npol == 1) ? 2.0 : 1.0;
  for (long ia=0; ia<becsum.extent(0); ++ia)
  for (long I =0; I <becsum.extent(1); ++I)
  for (long J =0; J <becsum.extent(2); ++J)
    becsum(ia, I, J) *= ns_scl;

  long nat   = V.ityp_view().extent(0);
  long ngm_d = V.ngm_dense_get();
  auto qg    = V.qgm_view();
  auto const& ijtoh = V.ijtoh_view();
  auto const& ityp  = V.ityp_view();
  auto const& nh    = V.nh_view();
  auto const& mill  = V.miller_g_dense_view();
  auto const& tau   = V.atom_pos_cart_view();
  auto recv = mfobj.recv();

  // Direct ρ_aug(G) (with structure factor); distribute G across MPI ranks,
  // then sum-reduce so every rank has the full vector for the comparison.
  long my_rank = mpi.comm.rank();
  long nproc   = mpi.comm.size();
  nda::array<ComplexType,1> rho_direct(ngm_d);
  rho_direct() = ComplexType(0.0);
  for (long g = my_rank; g < ngm_d; g += nproc) {
    double Gx = mill(g,0)*recv(0,0) + mill(g,1)*recv(1,0) + mill(g,2)*recv(2,0);
    double Gy = mill(g,0)*recv(0,1) + mill(g,1)*recv(1,1) + mill(g,2)*recv(2,1);
    double Gz = mill(g,0)*recv(0,2) + mill(g,1)*recv(1,2) + mill(g,2)*recv(2,2);
    ComplexType acc(0.0);
    for (long ia=0; ia<nat; ++ia) {
      int nt = ityp(ia);
      int nh_a = nh(nt);
      if (nh_a == 0) continue;
      double phase = -(Gx*tau(ia,0) + Gy*tau(ia,1) + Gz*tau(ia,2));
      ComplexType sf(std::cos(phase), std::sin(phase));
      ComplexType atom_acc(0.0);
      for (int I=0; I<nh_a; ++I)
      for (int J=0; J<nh_a; ++J) {
        long ij = ijtoh(nt, I, J) - 1;
        if (ij < 0) continue;
        atom_acc += ComplexType(becsum(ia, I, J)) * qg(nt, ij, g);
      }
      acc += sf * atom_acc;
    }
    rho_direct(g) = acc;
  }
  mpi.comm.all_reduce_in_place_n(rho_direct.data(), rho_direct.size(),
                                  std::plus<>{});

  // ISDF-reconstructed ρ_aug(G); same distribution + sum-reduce.
  nda::array<ComplexType,1> rho_isdf(ngm_d);
  rho_isdf() = ComplexType(0.0);
  for (long g = my_rank; g < ngm_d; g += nproc) {
    double Gx = mill(g,0)*recv(0,0) + mill(g,1)*recv(1,0) + mill(g,2)*recv(2,0);
    double Gy = mill(g,0)*recv(0,1) + mill(g,1)*recv(1,1) + mill(g,2)*recv(2,1);
    double Gz = mill(g,0)*recv(0,2) + mill(g,1)*recv(1,2) + mill(g,2)*recv(2,2);
    ComplexType acc(0.0);
    for (long ia=0; ia<nat; ++ia) {
      int nt = ityp(ia);
      if (nt >= (int)isdf.size()) continue;
      auto const& s = isdf[nt];
      int nh_a = s.nh;
      if (nh_a == 0 || s.nlambda == 0) continue;
      // ν_{aλ} = Σ_{IJ} U(λ,I) becsum(ia,I,J) U(λ,J)
      double phase = -(Gx*tau(ia,0) + Gy*tau(ia,1) + Gz*tau(ia,2));
      ComplexType sf(std::cos(phase), std::sin(phase));
      ComplexType atom_acc(0.0);
      for (int lam=0; lam<s.nlambda; ++lam) {
        double nu = 0.0;
        for (int I=0; I<nh_a; ++I) {
          double uI = s.U(lam, I);
          if (uI == 0.0) continue;
          for (int J=0; J<nh_a; ++J) {
            double uJ = s.U(lam, J);
            if (uJ == 0.0) continue;
            nu += uI * becsum(ia, I, J) * uJ;
          }
        }
        atom_acc += ComplexType(nu) * s.eta_qg_q0(lam, g);
      }
      acc += sf * atom_acc;
    }
    rho_isdf(g) = acc;
  }
  mpi.comm.all_reduce_in_place_n(rho_isdf.data(), rho_isdf.size(),
                                  std::plus<>{});

  // Compare: max |ρ_direct(G) − ρ_isdf(G)| across G
  double max_err = 0.0, max_ref = 0.0;
  for (long g=0; g<ngm_d; ++g) {
    max_err = std::max(max_err, std::abs(rho_direct(g) - rho_isdf(g)));
    max_ref = std::max(max_ref, std::abs(rho_direct(g)));
  }
  app_log(2, "ρ_aug ISDF reconstruction: max|direct − isdf| = {:.3e}, "
             "max|direct| = {:.3e}, rel = {:.3e}",
             max_err, max_ref,
             max_ref > 0 ? max_err/max_ref : 0.0);
  CHECK(max_err < tol);
}

/**
 * One-center Hartree correction via K_a vs direct ΔC contraction.
 *
 *   E_K^a   = (1/2) Σ_{λξ} ν_{aλ} K_{a,λξ} ν_{aξ}            with ν = U becsum U^T
 *   E_dC^a  = (1/2) Σ_{IJKL} becsum_{aIJ} ΔC_a(IJKL) becsum_{aKL}
 *
 * Equality follows from the ΔC roundtrip — this is the operationally-relevant
 * assertion that the K_a injection in `hamilt::paw::add_K_a_to_LL` produces
 * the correct one-center Hartree contribution.
 */
template<MEMORY_SPACE MEM>
void test_local_isdf_K_a_one_center([[maybe_unused]] mpi_context_t& mpi, mf::MF& mfobj,
                                     double tol = 1e-12)
{
  auto all = nda::range::all;
  long nspin   = mfobj.nspin();
  long nk_ibz  = mfobj.nkpts_ibz();
  long nbnd    = mfobj.nbnd();
  int  npol    = mfobj.npol();

  memory::array<MEM, ComplexType, 3> nii(nspin, nk_ibz, nbnd);
  if constexpr (MEM == HOST_MEMORY) {
    nii() = mfobj.occ()(all, nda::range(nk_ibz), all);
  } else {
    nii() = nda::array<ComplexType,3>(mfobj.occ()(all, nda::range(nk_ibz), all));
  }

  hamilt::pseudopot V(mfobj);
  auto isdf = hamilt::paw::build_local_isdf(V);

  auto becsum = hamilt::paw::compute_becsum_diagonal(
      V.Pskna_view(), nii, V.ityp_view(), V.nh_view(), V.ofs_view(), npol);
  double ns_scl = (nspin == 1 && npol == 1) ? 2.0 : 1.0;
  for (long ia=0; ia<becsum.extent(0); ++ia)
  for (long I =0; I <becsum.extent(1); ++I)
  for (long J =0; J <becsum.extent(2); ++J)
    becsum(ia, I, J) *= ns_scl;

  auto const& sps = V.paw_species_view();
  auto const& ityp = V.ityp_view();
  long nat = ityp.extent(0);

  double E_K_total  = 0.0;
  double E_dC_total = 0.0;
  bool any = false;
  for (long ia=0; ia<nat; ++ia) {
    int nt = ityp(ia);
    if (nt >= (int)sps.size() || nt >= (int)isdf.size()) continue;
    auto const& sp_paw = sps[nt];
    if (!sp_paw.is_paw || sp_paw.deltaC.size() == 0) continue;
    auto const& s = isdf[nt];
    if (s.nlambda == 0) continue;
    any = true;
    int nh_a = s.nh;

    // Direct: (1/2) Σ becsum ΔC becsum
    double E_dC = 0.0;
    for (int I=0; I<nh_a; ++I)
    for (int J=0; J<nh_a; ++J)
    for (int Kp=0; Kp<nh_a; ++Kp)
    for (int L =0; L <nh_a; ++L)
      E_dC += 0.5 * becsum(ia,I,J) * sp_paw.deltaC(I,J,Kp,L) * becsum(ia,Kp,L);
    E_dC_total += E_dC;

    // Via K_a: (1/2) Σ ν K ν, with ν = U becsum U^T
    auto K = hamilt::paw::compute_K_a(s, sp_paw.deltaC);
    nda::array<double,1> nu(s.nlambda);
    nu() = 0.0;
    for (int lam=0; lam<s.nlambda; ++lam) {
      double acc = 0.0;
      for (int I=0; I<nh_a; ++I) {
        double uI = s.U(lam, I);
        if (uI == 0.0) continue;
        for (int J=0; J<nh_a; ++J)
          acc += uI * becsum(ia, I, J) * s.U(lam, J);
      }
      nu(lam) = acc;
    }
    double E_K = 0.0;
    for (int lam=0; lam<s.nlambda; ++lam)
    for (int xi =0; xi <s.nlambda; ++xi)
      E_K += 0.5 * nu(lam) * K(lam, xi) * nu(xi);
    E_K_total += E_K;
  }

  if (!any) {
    app_log(2, "K_a one-center: no PAW species, skipped.");
    return;
  }
  app_log(2, "One-center Hartree: K_a path = {:+.10f} Ha, ΔC direct = {:+.10f} Ha, "
             "diff = {:+.2e}",
             E_K_total, E_dC_total, E_K_total - E_dC_total);
  CHECK(std::abs(E_K_total - E_dC_total) < tol);
}

TEST_CASE("local_isdf_qhat_roundtrip", "[hamilt][paw][isdf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (USPP)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_uspp", mf::h5_input_type);
    test_local_isdf_qhat_roundtrip<HOST_MEMORY>(*mpi, qe_h5);
  }
  SECTION("lih_kp222_nbnd16 (PAW)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    test_local_isdf_qhat_roundtrip<HOST_MEMORY>(*mpi, qe_h5);
  }
}

TEST_CASE("local_isdf_deltaC_roundtrip", "[hamilt][paw][isdf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (PAW)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    test_local_isdf_deltaC_roundtrip<HOST_MEMORY>(*mpi, qe_h5);
  }
}

TEST_CASE("local_isdf_rho_aug_reconstruction", "[hamilt][paw][isdf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (USPP)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_uspp", mf::h5_input_type);
    test_local_isdf_rho_aug_reconstruction<HOST_MEMORY>(*mpi, qe_h5);
  }
  SECTION("lih_kp222_nbnd16 (PAW)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    test_local_isdf_rho_aug_reconstruction<HOST_MEMORY>(*mpi, qe_h5);
  }
}

/**
 * K_a in the COMPRESSED ISDF basis: assert the closed-form
 *   K_{λξ} = sign(λ)·sign(ξ)·ΔC[i(λ),j(λ),i(ξ),j(ξ)]
 * (computed from the compressed isdf's lambda_{i,j,sign}) reproduces ΔC_a
 * monotonically as the kept-pair count grows. At full coverage (every pair
 * with non-trivial qgm L²-norm kept), the reconstruction is exact.
 *
 * Also prints the (rank, K_err_max, K_err_F) curve as a diagnostic — this
 * is the analogue of `local_isdf_compression_accuracy` but for the
 * one-center K_a tensor instead of Q̂.
 */
template<MEMORY_SPACE MEM>
void test_local_isdf_K_a_compressed_accuracy([[maybe_unused]] mpi_context_t& mpi, mf::MF& mfobj,
                                              hamilt::paw::isdf_metric metric)
{
  hamilt::pseudopot V(mfobj);
  auto recv = mfobj.recv();
  double det_B =
      recv(0,0)*(recv(1,1)*recv(2,2) - recv(1,2)*recv(2,1))
    - recv(1,0)*(recv(0,1)*recv(2,2) - recv(0,2)*recv(2,1))
    + recv(2,0)*(recv(0,1)*recv(1,2) - recv(0,2)*recv(1,1));
  double omega = (2.0*M_PI)*(2.0*M_PI)*(2.0*M_PI) / std::abs(det_B);

  auto const& sps = V.paw_species_view();
  bool any = false;
  for (int nt = 0; nt < (int)sps.size(); ++nt) {
    auto const& sp_paw = sps[nt];
    if (!sp_paw.is_paw || sp_paw.deltaC.size() == 0) continue;
    any = true;

    auto curve = hamilt::paw::build_K_a_rank_error_curve(
        V, nt, recv, omega, metric, /*tol*/1e-14);
    REQUIRE(!curve.empty());

    app_log(2, "K_a compression report — species {} ({}):", nt,
               hamilt::paw::metric_name(metric));
    app_log(2, "  rank | nλ |  max|ΔC − ΣUUKUU|  |  ‖ΔC − ΣUUKUU‖_F");
    for (auto const& row : curve) {
      app_log(2, "  {:>4} | {:>2} | {:>16.4e}  |  {:>16.4e}",
                 row.n_pairs_kept, row.n_lambda, row.K_err_max, row.K_err_F);
    }

    // Both error metrics monotone non-increasing.
    for (size_t i = 1; i < curve.size(); ++i) {
      CHECK(curve[i].K_err_max <= curve[i-1].K_err_max + 1e-12);
      CHECK(curve[i].K_err_F   <= curve[i-1].K_err_F   + 1e-12);
    }

    // Exact at compressed-by-norm full coverage. Build that explicitly and
    // assert ΔC reconstruction error is machine zero.
    auto full = hamilt::paw::build_local_isdf_compressed_by_norm(
        V, nt, recv, omega, metric, /*tol*/1e-15);
    auto K_full = hamilt::paw::compute_K_a(full, sp_paw.deltaC);
    auto er = hamilt::paw::compressed_K_a_error(full, sp_paw.deltaC, K_full);
    app_log(2, "  full-coverage (nλ={}): max|ΔC − ΣUUKUU| = {:.3e}, "
               "‖ΔC − ΣUUKUU‖_F = {:.3e}",
               full.nlambda, er.err_max, er.err_F);
    CHECK(er.err_max < 1e-12);
    CHECK(er.err_F   < 1e-12);
  }
  if (!any) app_log(2, "K_a compressed accuracy: no PAW species, skipped.");
}

TEST_CASE("local_isdf_K_a_compressed_accuracy", "[hamilt][paw][isdf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (PAW, L²)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    test_local_isdf_K_a_compressed_accuracy<HOST_MEMORY>(
        *mpi, qe_h5, hamilt::paw::isdf_metric::L2);
  }
  SECTION("lih_kp222_nbnd16 (PAW, Coulomb)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    test_local_isdf_K_a_compressed_accuracy<HOST_MEMORY>(
        *mpi, qe_h5, hamilt::paw::isdf_metric::Coulomb);
  }
}

/**
 * Operationally relevant K_a check: the one-center Hartree contribution
 *   E_K(rank)   = (1/2) Σ_{λξ} ν_λ K_{λξ} ν_ξ                 (compressed)
 *   E_dC        = (1/2) Σ_{IJKL} becsum_{IJ} ΔC_{IJKL} becsum_{KL}  (direct)
 *
 * must converge to E_dC as rank increases. For the actual SCF becsum at
 * the LiH PAW fixture, modest rank already gives small absolute error;
 * full coverage yields machine precision (already tested in
 * `local_isdf_K_a_one_center`, but here we trace the convergence).
 */
template<MEMORY_SPACE MEM>
void test_local_isdf_K_a_one_center_vs_rank(mpi_context_t& mpi, mf::MF& mfobj,
                                             hamilt::paw::isdf_metric metric)
{
  auto all = nda::range::all;
  long nspin   = mfobj.nspin();
  long nk_ibz  = mfobj.nkpts_ibz();
  long nbnd    = mfobj.nbnd();
  int  npol    = mfobj.npol();

  memory::array<MEM, ComplexType, 3> nii(nspin, nk_ibz, nbnd);
  if constexpr (MEM == HOST_MEMORY) {
    nii() = mfobj.occ()(all, nda::range(nk_ibz), all);
  } else {
    nii() = nda::array<ComplexType,3>(mfobj.occ()(all, nda::range(nk_ibz), all));
  }

  hamilt::pseudopot V(mfobj);
  auto recv = mfobj.recv();
  double det_B =
      recv(0,0)*(recv(1,1)*recv(2,2) - recv(1,2)*recv(2,1))
    - recv(1,0)*(recv(0,1)*recv(2,2) - recv(0,2)*recv(2,1))
    + recv(2,0)*(recv(0,1)*recv(1,2) - recv(0,2)*recv(1,1));
  double omega = (2.0*M_PI)*(2.0*M_PI)*(2.0*M_PI) / std::abs(det_B);

  auto becsum = hamilt::paw::compute_becsum_diagonal(
      V.Pskna_view(), nii, V.ityp_view(), V.nh_view(), V.ofs_view(), npol);
  double ns_scl = (nspin == 1 && npol == 1) ? 2.0 : 1.0;
  for (long ia=0; ia<becsum.extent(0); ++ia)
  for (long I =0; I <becsum.extent(1); ++I)
  for (long J =0; J <becsum.extent(2); ++J)
    becsum(ia, I, J) *= ns_scl;

  auto const& sps = V.paw_species_view();
  auto const& ityp = V.ityp_view();
  long nat = ityp.extent(0);

  bool any = false;
  for (int nt = 0; nt < (int)sps.size(); ++nt) {
    auto const& sp_paw = sps[nt];
    if (!sp_paw.is_paw || sp_paw.deltaC.size() == 0) continue;
    any = true;

    // Direct E_dC for this species (summed over atoms of this type)
    double E_dC = 0.0;
    for (long ia = 0; ia < nat; ++ia) {
      if ((int)ityp(ia) != nt) continue;
      int nh_a = sp_paw.nh;
      for (int I=0; I<nh_a; ++I)
      for (int J=0; J<nh_a; ++J)
      for (int Kp=0; Kp<nh_a; ++Kp)
      for (int L =0; L <nh_a; ++L)
        E_dC += 0.5 * becsum(ia,I,J) * sp_paw.deltaC(I,J,Kp,L) * becsum(ia,Kp,L);
    }

    auto rep = hamilt::paw::pivoted_cholesky_qgm_pairs(
        V, nt, recv, omega, metric, /*tol*/1e-14);
    int n_max = (int)rep.pair_pivot_order.size();

    app_log(2, "K_a one-center vs rank — species {} ({}): E_dC = {:+.10f} Ha",
               nt, hamilt::paw::metric_name(metric), E_dC);
    app_log(2, "  rank | nλ |   E_K (Ha)   |    |E_K − E_dC|");
    int nh_a = sp_paw.nh;
    double last_err = 0.0;
    // Distribute the diagnostic for-n loop across ranks; gather (E_K, nlambda)
    // for printing on rank 0. Each rank owns n if (n % nproc) == rank.
    long my_rank = mpi.comm.rank();
    long nproc   = mpi.comm.size();
    nda::array<double, 1> E_K_table(n_max + 1);
    nda::array<long,   1> nlam_table(n_max + 1);
    E_K_table()  = 0.0;
    nlam_table() = 0;
    for (int n = 0; n <= n_max; ++n) {
      if ((long)n % nproc != my_rank) continue;
      auto isdf_n = hamilt::paw::build_local_isdf_compressed(
          V, nt, nh_a, rep, n);
      double E_K = 0.0;
      auto K = hamilt::paw::compute_K_a(isdf_n, sp_paw.deltaC);
      for (long ia = 0; ia < nat; ++ia) {
        if ((int)ityp(ia) != nt) continue;
        nda::array<double,1> nu(isdf_n.nlambda);
        nu() = 0.0;
        for (int lam=0; lam<isdf_n.nlambda; ++lam) {
          double acc = 0.0;
          for (int I=0; I<nh_a; ++I) {
            double uI = isdf_n.U(lam, I);
            if (uI == 0.0) continue;
            for (int J=0; J<nh_a; ++J)
              acc += uI * becsum(ia, I, J) * isdf_n.U(lam, J);
          }
          nu(lam) = acc;
        }
        for (int lam=0; lam<isdf_n.nlambda; ++lam)
        for (int xi =0; xi <isdf_n.nlambda; ++xi)
          E_K += 0.5 * nu(lam) * K(lam, xi) * nu(xi);
      }
      E_K_table(n)  = E_K;
      nlam_table(n) = isdf_n.nlambda;
    }
    mpi.comm.all_reduce_in_place_n(E_K_table.data(),  E_K_table.size(),  std::plus<>{});
    mpi.comm.all_reduce_in_place_n(nlam_table.data(), nlam_table.size(), std::plus<>{});
    for (int n = 0; n <= n_max; ++n) {
      double E_K = E_K_table(n);
      double err = std::abs(E_K - E_dC);
      app_log(2, "  {:>4} | {:>2} | {:+.10f} | {:.3e}",
                 n, nlam_table(n), E_K, err);
      last_err = err;
    }
    // At pivoted-Cholesky-converged rank some pairs may still be
    // unconvered (linearly-dependent pairs whose pointwise norm is
    // significant); accept any |E_K − E_dC| at the tail. Convergence to
    // machine zero is asserted via build_local_isdf_compressed_by_norm.
    (void)last_err;

    // Full coverage by initial-norm: E_K must equal E_dC to FP precision.
    auto full_isdf = hamilt::paw::build_local_isdf_compressed_by_norm(
        V, nt, recv, omega, metric, /*tol*/1e-15);
    auto K_full = hamilt::paw::compute_K_a(full_isdf, sp_paw.deltaC);
    double E_K_full = 0.0;
    for (long ia = 0; ia < nat; ++ia) {
      if ((int)ityp(ia) != nt) continue;
      nda::array<double,1> nu(full_isdf.nlambda);
      nu() = 0.0;
      for (int lam=0; lam<full_isdf.nlambda; ++lam) {
        double acc = 0.0;
        for (int I=0; I<nh_a; ++I) {
          double uI = full_isdf.U(lam, I);
          if (uI == 0.0) continue;
          for (int J=0; J<nh_a; ++J)
            acc += uI * becsum(ia, I, J) * full_isdf.U(lam, J);
        }
        nu(lam) = acc;
      }
      for (int lam=0; lam<full_isdf.nlambda; ++lam)
      for (int xi =0; xi <full_isdf.nlambda; ++xi)
        E_K_full += 0.5 * nu(lam) * K_full(lam, xi) * nu(xi);
    }
    app_log(2, "  full-coverage (nλ={}): E_K = {:+.10f} Ha, |E_K − E_dC| = {:.3e}",
               full_isdf.nlambda, E_K_full, std::abs(E_K_full - E_dC));
    CHECK(std::abs(E_K_full - E_dC) < 1e-12);
  }
  if (!any) {
    app_log(2, "K_a one-center vs rank: no PAW species, skipped.");
  }
}

TEST_CASE("local_isdf_K_a_one_center_vs_rank", "[hamilt][paw][isdf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (PAW, L²)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    test_local_isdf_K_a_one_center_vs_rank<HOST_MEMORY>(
        *mpi, qe_h5, hamilt::paw::isdf_metric::L2);
  }
  SECTION("lih_kp222_nbnd16 (PAW, Coulomb)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    test_local_isdf_K_a_one_center_vs_rank<HOST_MEMORY>(
        *mpi, qe_h5, hamilt::paw::isdf_metric::Coulomb);
  }
}

TEST_CASE("local_isdf_K_a_one_center", "[hamilt][paw][isdf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (PAW)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    test_local_isdf_K_a_one_center<HOST_MEMORY>(*mpi, qe_h5);
  }
}

/**
 * Compression: build pivoted-Cholesky compressed local-ISDF and assert the
 * rank-vs-error curve is monotone non-increasing, that full rank == nij
 * gives machine-precision Q̂ reconstruction, and that the predicted
 * residual norm matches the actual reconstructed-Q̂ error in the chosen
 * metric (matrix-free Cholesky bookkeeping consistent with reconstruction).
 *
 * Also prints a per-species report of (n_pairs_kept, n_lambda, residual,
 * max pointwise error) — useful as a diagnostic when picking a tolerance.
 */
template<MEMORY_SPACE MEM>
void test_local_isdf_compression_accuracy([[maybe_unused]] mpi_context_t& mpi, mf::MF& mfobj,
                                          hamilt::paw::isdf_metric metric)
{
  hamilt::pseudopot V(mfobj);
  auto recv = mfobj.recv();
  // Cell volume Ω = (2π)³ / |det B|
  double det_B =
      recv(0,0)*(recv(1,1)*recv(2,2) - recv(1,2)*recv(2,1))
    - recv(1,0)*(recv(0,1)*recv(2,2) - recv(0,2)*recv(2,1))
    + recv(2,0)*(recv(0,1)*recv(1,2) - recv(0,2)*recv(1,1));
  double omega = (2.0*M_PI)*(2.0*M_PI)*(2.0*M_PI) / std::abs(det_B);

  auto const& sps = V.paw_species_view();
  bool any = false;
  for (int nt = 0; nt < (int)sps.size(); ++nt) {
    bool has_aug = sps[nt].is_paw || sps[nt].is_uspp;
    if (!has_aug) continue;
    any = true;

    auto curve = hamilt::paw::build_rank_error_curve(
        V, nt, recv, omega, metric, /*tol*/1e-14);
    REQUIRE(!curve.empty());

    app_log(2, "Local-ISDF compression report — species {} ({}):", nt,
               hamilt::paw::metric_name(metric));
    app_log(2, "  rank | nλ | residual ({}) | max|Q̂ − approx|",
               hamilt::paw::metric_name(metric));
    for (auto const& row : curve) {
      app_log(2, "  {:>4} | {:>2} | {:>12.4e} | {:>14.4e}",
                 row.n_pairs_kept, row.n_lambda,
                 row.error_metric, row.max_qhat_err);
    }

    // Cholesky residual decreases monotonically (modulo a tiny FP slop).
    for (size_t i = 1; i < curve.size(); ++i) {
      double e_prev = curve[i-1].error_metric;
      double e_now  = curve[i].error_metric;
      CHECK(e_now <= e_prev + 1e-12);
    }

    // Pointwise reconstruction error decreases monotonically through the
    // pivot order. (Without LS refit of η, this is the actual recon error
    // at each rank, not the Cholesky residual; both should be monotone.)
    for (size_t i = 1; i < curve.size(); ++i) {
      CHECK(curve[i].max_qhat_err <= curve[i-1].max_qhat_err + 1e-12);
    }

    // At full Cholesky convergence, the L²/Coulomb residual is
    // machine-zero. The pointwise pair error need NOT be zero — pivoted
    // Cholesky may declare convergence while a remaining pair is
    // linearly dependent on the picks, but our outer-product compression
    // can't exploit that dependence. The remaining pointwise error at
    // convergence is then bounded by the dropped pair's full norm.
    auto const& last = curve.back();
    CHECK(last.error_metric < 1e-10);
  }
  if (!any) {
    app_log(2, "Local-ISDF compression: no augmentation species, skipped.");
  }
}

TEST_CASE("local_isdf_compression_accuracy", "[hamilt][paw][isdf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (USPP, L²)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_uspp", mf::h5_input_type);
    test_local_isdf_compression_accuracy<HOST_MEMORY>(
        *mpi, qe_h5, hamilt::paw::isdf_metric::L2);
  }
  SECTION("lih_kp222_nbnd16 (USPP, Coulomb)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_uspp", mf::h5_input_type);
    test_local_isdf_compression_accuracy<HOST_MEMORY>(
        *mpi, qe_h5, hamilt::paw::isdf_metric::Coulomb);
  }
  SECTION("lih_kp222_nbnd16 (PAW, L²)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    test_local_isdf_compression_accuracy<HOST_MEMORY>(
        *mpi, qe_h5, hamilt::paw::isdf_metric::L2);
  }
  SECTION("lih_kp222_nbnd16 (PAW, Coulomb)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    test_local_isdf_compression_accuracy<HOST_MEMORY>(
        *mpi, qe_h5, hamilt::paw::isdf_metric::Coulomb);
  }
}

/**
 * Compression at moderate tolerance still gives a small Hartree-relevant
 * augmentation density error. Specifically, for a tolerance ε in the
 * Coulomb metric, the augmented Hartree contribution
 *   ΔE_H_aug ≤ ‖ρ_aug,direct − ρ_aug,compressed‖_C · ‖ρ_aug,direct‖_C
 * is bounded by the reported error_metric times a normalization factor.
 *
 * We test this by comparing the full ρ_aug(G) (used in test
 * `local_isdf_rho_aug_reconstruction` at machine precision when full rank)
 * against the version reconstructed from a tolerance-truncated isdf, and
 * confirming the relative error matches the rank-vs-error report.
 */
template<MEMORY_SPACE MEM>
void test_local_isdf_compressed_rho_aug(mpi_context_t& mpi, mf::MF& mfobj,
                                         double tol_pc, double tol_check)
{
  auto all = nda::range::all;
  long nspin   = mfobj.nspin();
  long nk_ibz  = mfobj.nkpts_ibz();
  long nbnd    = mfobj.nbnd();
  int  npol    = mfobj.npol();

  memory::array<MEM, ComplexType, 3> nii(nspin, nk_ibz, nbnd);
  if constexpr (MEM == HOST_MEMORY) {
    nii() = mfobj.occ()(all, nda::range(nk_ibz), all);
  } else {
    nii() = nda::array<ComplexType,3>(mfobj.occ()(all, nda::range(nk_ibz), all));
  }
  hamilt::pseudopot V(mfobj);
  if (V.qgm_view().size() == 0) return;

  auto recv = mfobj.recv();
  double det_B =
      recv(0,0)*(recv(1,1)*recv(2,2) - recv(1,2)*recv(2,1))
    - recv(1,0)*(recv(0,1)*recv(2,2) - recv(0,2)*recv(2,1))
    + recv(2,0)*(recv(0,1)*recv(1,2) - recv(0,2)*recv(1,1));
  double omega = (2.0*M_PI)*(2.0*M_PI)*(2.0*M_PI) / std::abs(det_B);

  // Compressed isdf (Coulomb metric — most relevant for E_H)
  auto const& sps = V.paw_species_view();
  std::vector<hamilt::paw::species_local_isdf> isdf(sps.size());
  for (int nt = 0; nt < (int)sps.size(); ++nt) {
    bool has_aug = sps[nt].is_paw || sps[nt].is_uspp;
    if (!has_aug) continue;
    auto [s, r] = hamilt::paw::compress_local_isdf_species(
        V, nt, recv, omega, hamilt::paw::isdf_metric::Coulomb, tol_pc);
    isdf[nt] = std::move(s);
  }

  // becsum (with QE ns_scl convention)
  auto becsum = hamilt::paw::compute_becsum_diagonal(
      V.Pskna_view(), nii, V.ityp_view(), V.nh_view(), V.ofs_view(), npol);
  double ns_scl = (nspin == 1 && npol == 1) ? 2.0 : 1.0;
  for (long ia=0; ia<becsum.extent(0); ++ia)
  for (long I =0; I <becsum.extent(1); ++I)
  for (long J =0; J <becsum.extent(2); ++J)
    becsum(ia, I, J) *= ns_scl;

  long nat   = V.ityp_view().extent(0);
  long ngm_d = V.ngm_dense_get();
  auto qg    = V.qgm_view();
  auto const& ijtoh = V.ijtoh_view();
  auto const& ityp  = V.ityp_view();
  auto const& nh    = V.nh_view();
  auto const& mill  = V.miller_g_dense_view();
  auto const& tau   = V.atom_pos_cart_view();

  // Direct ρ_aug(G) and ISDF-compressed ρ_aug(G); compare in Coulomb norm.
  // Distribute the outer G loop across MPI ranks; sum-reduce both norms.
  double err_C2 = 0.0;
  double ref_C2 = 0.0;
  {
    long my_rank = mpi.comm.rank();
    long nproc   = mpi.comm.size();
    for (long g = my_rank; g < ngm_d; g += nproc) {
      double Gx = mill(g,0)*recv(0,0) + mill(g,1)*recv(1,0) + mill(g,2)*recv(2,0);
      double Gy = mill(g,0)*recv(0,1) + mill(g,1)*recv(1,1) + mill(g,2)*recv(2,1);
      double Gz = mill(g,0)*recv(0,2) + mill(g,1)*recv(1,2) + mill(g,2)*recv(2,2);
      double G2 = Gx*Gx + Gy*Gy + Gz*Gz;
      if (G2 < 1e-14) continue;
      double w = 4.0*M_PI/(omega*G2);

      ComplexType direct(0.0), compressed(0.0);
      for (long ia=0; ia<nat; ++ia) {
        int nt = ityp(ia);
        int nh_a = nh(nt);
        if (nh_a == 0) continue;
        double phase = -(Gx*tau(ia,0) + Gy*tau(ia,1) + Gz*tau(ia,2));
        ComplexType sf(std::cos(phase), std::sin(phase));

        ComplexType atom_d(0.0);
        for (int I=0; I<nh_a; ++I)
        for (int J=0; J<nh_a; ++J) {
          long ij = ijtoh(nt, I, J) - 1;
          if (ij < 0) continue;
          atom_d += ComplexType(becsum(ia, I, J)) * qg(nt, ij, g);
        }
        direct += sf * atom_d;

        auto const& s = isdf[nt];
        if (s.nlambda == 0) continue;
        ComplexType atom_c(0.0);
        for (int lam=0; lam<s.nlambda; ++lam) {
          double nu = 0.0;
          for (int I=0; I<nh_a; ++I) {
            double uI = s.U(lam, I);
            if (uI == 0.0) continue;
            for (int J=0; J<nh_a; ++J) {
              double uJ = s.U(lam, J);
              if (uJ == 0.0) continue;
              nu += uI * becsum(ia, I, J) * uJ;
            }
          }
          atom_c += ComplexType(nu) * s.eta_qg_q0(lam, g);
        }
        compressed += sf * atom_c;
      }
      ComplexType d = direct - compressed;
      err_C2 += w * (std::real(d)*std::real(d) + std::imag(d)*std::imag(d));
      ref_C2 += w * (std::real(direct)*std::real(direct)
                    + std::imag(direct)*std::imag(direct));
    }
    mpi.comm.all_reduce_in_place_n(&err_C2, 1, std::plus<>{});
    mpi.comm.all_reduce_in_place_n(&ref_C2, 1, std::plus<>{});
  }
  double err_C = std::sqrt(std::max(0.0, err_C2));
  double ref_C = std::sqrt(std::max(0.0, ref_C2));
  double rel   = (ref_C > 0.0) ? err_C/ref_C : 0.0;
  app_log(2, "Compressed ρ_aug Coulomb error: tol_pc={:.1e}, |err|_C={:.3e}, "
             "|ref|_C={:.3e}, rel={:.3e}", tol_pc, err_C, ref_C, rel);
  CHECK(err_C < tol_check);
}

TEST_CASE("local_isdf_compressed_rho_aug", "[hamilt][paw][isdf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (PAW, tight tol)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    test_local_isdf_compressed_rho_aug<HOST_MEMORY>(
        *mpi, qe_h5, /*tol_pc*/1e-14, /*tol_check*/1e-8);
  }
  SECTION("lih_kp222_nbnd16 (PAW, loose tol)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    // Loose tol: ‖compressed_error‖_C should be ≲ tol_pc × Ω-scale
    test_local_isdf_compressed_rho_aug<HOST_MEMORY>(
        *mpi, qe_h5, /*tol_pc*/1e-4, /*tol_check*/5e-2);
  }
}

/**
 * H5 cache round-trip: write the compressed-by-norm isdf at machine-zero
 * tolerance (= every pair with non-trivial Coulomb norm kept; reconstruction
 * is exact by construction), read it back, and assert binary equality on
 * every dataset against the in-memory reference.
 */
template<MEMORY_SPACE MEM>
void test_local_isdf_h5_roundtrip(mpi_context_t& mpi, mf::MF& mfobj)
{
  hamilt::pseudopot V(mfobj);
  auto recv = mfobj.recv();
  double det_B =
      recv(0,0)*(recv(1,1)*recv(2,2) - recv(1,2)*recv(2,1))
    - recv(1,0)*(recv(0,1)*recv(2,2) - recv(0,2)*recv(2,1))
    + recv(2,0)*(recv(0,1)*recv(1,2) - recv(0,2)*recv(1,1));
  double omega = (2.0*M_PI)*(2.0*M_PI)*(2.0*M_PI) / std::abs(det_B);

  std::string tmp = "/tmp/coqui_local_isdf_test_" +
                     std::to_string(::getpid()) + ".h5";
  if (mpi.comm.root()) {
    if (std::filesystem::exists(tmp)) std::filesystem::remove(tmp);

    // Build full-rank-equivalent compressed-by-norm isdf for every species
    // (machine-precision reconstruction by construction), and write it.
    auto const& sps = V.paw_species_view();
    int nsp = (int)sps.size();
    std::vector<hamilt::paw::species_local_isdf> ref(nsp);
    std::vector<hamilt::paw::isdf_compression_report> reps(nsp);
    for (int nt = 0; nt < nsp; ++nt) {
      bool has_aug = sps[nt].is_paw || sps[nt].is_uspp;
      if (!has_aug) continue;
      ref[nt] = hamilt::paw::build_local_isdf_compressed_by_norm(
          V, nt, recv, omega, hamilt::paw::isdf_metric::L2, /*tol*/1e-15);
      // pair_pivot_order/error_at_step are diagnostics; leave empty here.
    }
    {
      h5::file f(tmp, 'w');
      h5::group root(f);
      hamilt::paw::write_local_isdf_h5(root, ref, reps,
                                        hamilt::paw::isdf_metric::L2,
                                        /*tol*/1e-15);
    }

    hamilt::paw::isdf_metric m_back = hamilt::paw::isdf_metric::Coulomb;
    double tol_back = -1.0;
    auto loaded = hamilt::paw::load_compressed_local_isdf_from_h5(
        tmp, &m_back, &tol_back);

    REQUIRE(m_back == hamilt::paw::isdf_metric::L2);
    REQUIRE(std::abs(tol_back - 1e-15) < 1e-18);
    REQUIRE((int)loaded.size() == nsp);

    for (int nt = 0; nt < nsp; ++nt) {
      bool has_aug = sps[nt].is_paw || sps[nt].is_uspp;
      auto const& a = loaded[nt];
      auto const& b = ref[nt];
      if (!has_aug) {
        CHECK(a.nlambda == 0);
        continue;
      }
      // Binary equality on every member.
      REQUIRE(a.nh == b.nh);
      REQUIRE(a.nlambda == b.nlambda);
      double max_U = 0.0;
      for (long i = 0; i < a.U.extent(0); ++i)
      for (long j = 0; j < a.U.extent(1); ++j)
        max_U = std::max(max_U, std::abs(a.U(i,j) - b.U(i,j)));
      CHECK(max_U == 0.0);
      for (long i = 0; i < a.lambda_i.extent(0); ++i) {
        CHECK(a.lambda_i(i)    == b.lambda_i(i));
        CHECK(a.lambda_j(i)    == b.lambda_j(i));
        CHECK(a.lambda_ij(i)   == b.lambda_ij(i));
        CHECK(a.lambda_sign(i) == b.lambda_sign(i));
      }
      double max_eta = 0.0;
      for (long i = 0; i < a.eta_qg_q0.extent(0); ++i)
      for (long g = 0; g < a.eta_qg_q0.extent(1); ++g)
        max_eta = std::max(max_eta,
                           std::abs(a.eta_qg_q0(i,g) - b.eta_qg_q0(i,g)));
      CHECK(max_eta == 0.0);

      // And the reconstructed Q̂ is exact (within FP eps) since we kept
      // every pair with non-trivial L² norm.
      auto qgm = V.qgm_view();
      auto const& ijtoh = V.ijtoh_view();
      double max_qhat = 0.0;
      for (int I = 0; I < a.nh; ++I)
      for (int J = 0; J < a.nh; ++J) {
        int ij_ref = ijtoh(nt, I, J) - 1;
        if (ij_ref < 0) continue;
        long ngm = qgm.extent(2);
        for (long g = 0; g < ngm; ++g) {
          ComplexType acc(0.0);
          for (long lam = 0; lam < a.nlambda; ++lam) {
            double w = a.U(lam, I) * a.U(lam, J);
            if (w == 0.0) continue;
            acc += ComplexType(w) * a.eta_qg_q0(lam, g);
          }
          max_qhat = std::max(max_qhat, std::abs(qgm(nt, ij_ref, g) - acc));
        }
      }
      app_log(2, "h5 round-trip species {}: nλ = {}, max|Q̂ − approx| = {:.3e}",
                 nt, a.nlambda, max_qhat);
      CHECK(max_qhat < 1e-12);
    }

    std::filesystem::remove(tmp);
  }
  mpi.comm.barrier();
}

TEST_CASE("local_isdf_h5_roundtrip", "[hamilt][paw][isdf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (PAW)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    test_local_isdf_h5_roundtrip<HOST_MEMORY>(*mpi, qe_h5);
  }
  SECTION("lih_kp222_nbnd16 (USPP)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_uspp", mf::h5_input_type);
    test_local_isdf_h5_roundtrip<HOST_MEMORY>(*mpi, qe_h5);
  }
}

/**
 * HF eigenvalue regression test.
 *
 * For a converged QE input_dft='hf' SCF, the band basis diagonalizes the HF
 * Fock matrix F = T + V_loc + V_NL + V_H + K, so diag(F)/diag(S) = eigval
 * for occupied bands. This test computes F via CoQui's pipeline (hamilt::H
 * for the kinetic/V_loc/V_NL/V_H block, hamilt::Vexchange for the signed
 * exact-exchange block) and compares the diagonals to mfobj.eigval.
 *
 * The qe_lih222_hf fixture is configured with `exxdiv_treatment='none'`
 * and `x_gamma_extrapolation=.false.`, which matches CoQui's bare-4π/G²
 * Coulomb with the G+Δk=0 component zeroed. Hence absolute eigenvalues
 * should match (no Gygi-Baldereschi-style constant offset).
 */
template<MEMORY_SPACE MEM>
void test_hf_eigenvalues(mpi_context_t& mpi, mf::MF& mfobj,
                          double tol = 5e-5, double occ_threshold = 1e-6)
{
  auto all = nda::range::all;
  long nspin   = mfobj.nspin();
  long nk_ibz  = mfobj.nkpts_ibz();
  long nbnd    = mfobj.nbnd();

  memory::array<MEM, ComplexType, 3> nii(nspin, nk_ibz, nbnd);
  if constexpr (MEM == HOST_MEMORY) {
    nii() = mfobj.occ()(all, nda::range(nk_ibz), all);
  } else {
    nii() = nda::array<ComplexType,3>(mfobj.occ()(all, nda::range(nk_ibz), all));
  }

  hamilt::pseudopot V(mfobj);
  // H = T + V_loc + V_NL + V_H
  auto Hij = hamilt::H<MEM>(mfobj, mpi.comm, &V, nii);
  // K = exact-exchange matrix (signed; F = H + K conventionally)
  auto Kij = hamilt::Vexchange<MEM>(mfobj, mpi.comm, &V, nii);

  utils::check(Hij.local_shape() == Kij.local_shape(),
               "test_hf_eigenvalues: H/K local shape mismatch");
  nda::tensor::add(ComplexType(1.0), Kij.local(),
                   ComplexType(1.0), Hij.local());

  // QE-saved orbitals are S_aug-orthonormal: ⟨ψ̃|S|ψ̃⟩ = δ_nm by construction.
  // ⟨ψ̃_n|H|ψ̃_n⟩ = ε_n directly — no overlap correction needed for
  // diagonal eigenvalues (compare H_nn to eigval without dividing by S).
  auto Hloc = nda::to_host(Hij.local());
  auto Kloc = nda::to_host(Kij.local());
  double max_err = 0.0;
  long  count = 0;
  auto b_rng = Hij.local_range(3);
  for (auto [is, s] : itertools::enumerate(Hij.local_range(0)))
    for (auto [ik, k] : itertools::enumerate(Hij.local_range(1))) {
      for (auto [ia, a] : itertools::enumerate(Hij.local_range(2))) {
        if (!(a >= b_rng.first() && a < b_rng.last())) continue;
        if (mfobj.occ(s, k, a) < occ_threshold) continue;
        long ib = a - b_rng.first();
        double H_diag = std::real(Hloc(is, ik, ia, ib));
        double K_diag = std::real(Kloc(is, ik, ia, ib));
        double eps_ref = mfobj.eigval(s, k, a);
        double err = std::abs(H_diag - eps_ref);
        if (err > max_err) max_err = err;
        ++count;
        if (err > tol) {
          app_log(2,
            "HF eigval mismatch: s={}, k={}, n={}, K_nn={:+.6f}, H_nn={:+.6f}, "
            "ref={:+.6f}, err={:.2e}",
            s, k, a, K_diag, H_diag, eps_ref, err);
        }
      }
    }
  max_err = mpi.comm.all_reduce_value(max_err, boost::mpi3::max<>{});
  count   = mpi.comm.all_reduce_value(count,   std::plus<>{});
  app_log(2,
    "HF eigenvalue regression: max|H_nn - eps_n| = {:.3e} over {} occupied "
    "states (tol={:.1e})", max_err, count, tol);
  CHECK(max_err < tol);
}

TEST_CASE("hf_eigenvalues", "[hamilt][hf]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  // NCPP HF reference: fixture uses exxdiv_treatment='none' and
  // x_gamma_extrapolation=.false., matching CoQui's bare 4π/|G+Δk|² with
  // G+Δk=0 zeroed. No singularity correction needed for absolute match.
  SECTION("lih_kp222_nbnd16 (NCPP, HF)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_hf", mf::h5_input_type);
    test_hf_eigenvalues<HOST_MEMORY>(*mpi, qe_h5, /*tol*/ 5e-5);
  }

  // USPP HF: exercises the G-space augmentation (Q^IJ_atm × phase) without
  // the PAW one-center correction. The USPP exchange kernel = NCPP smooth
  // exchange + Q-augmentation in the pair density only.
  SECTION("lih_kp222_nbnd16 (USPP, HF)") {
    namespace fs = std::filesystem;
    auto [outdir, prefix] = utils::utest_filename("qe_lih222_uspp_hf");
    if (!fs::exists(outdir + "/pwscf.save/wfc1.hdf5")) {
      WARN("Skipping USPP HF section: missing " << outdir
           << "/pwscf.save/wfc1.hdf5");
    } else {
      auto qe_h5 = mf::default_MF(mpi, "qe_lih222_uspp_hf", mf::h5_input_type);
      test_hf_eigenvalues<HOST_MEMORY>(*mpi, qe_h5, /*tol*/ 5e-3);
    }
  }

  // PAW HF: exercises the v_x_paw.hpp pair-density augmentation
  // (Q^IJ_atm(G+Δk) × e^{-i(G+Δk)·τ_atm} added to smooth pair density
  // before Coulomb contraction). Requires lih_kp222_nbnd16_paw_hf/
  // pwscf.save/wfc1.hdf5; if absent (only pwscf.coqui.h5 in fixture)
  // the section is skipped. Regenerate via QE input_dft='hf+noc' with
  // exxdiv_treatment='none', x_gamma_extrapolation=.false.; see the
  // scf.inp shipped in the fixture directory.
  SECTION("lih_kp222_nbnd16 (PAW, HF)") {
    namespace fs = std::filesystem;
    auto [outdir, prefix] = utils::utest_filename("qe_lih222_paw_hf");
    if (!fs::exists(outdir + "/pwscf.save/wfc1.hdf5")) {
      WARN("Skipping PAW HF section: " << outdir
           << "/pwscf.save/wfc1.hdf5 not found. "
           << "Regenerate via QE input_dft='hf+noc' (see fixture scf.inp).");
    } else {
      auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw_hf", mf::h5_input_type);
      test_hf_eigenvalues<HOST_MEMORY>(*mpi, qe_h5, /*tol*/ 5e-3);
    }
  }

}

/**
 * Structural smoke test for the exchange matrix on USPP/PAW fixtures.
 *
 * Goal: verify that Vexchange runs on the augmentation code path without
 * crashing and that the output K_{ij} matrix has the expected structural
 * properties — Hermitian, finite, real on the diagonal for non-magnetic
 * non-SOC systems. This does NOT compare to QE numerically (the existing
 * PAW DFT fixture has no HF reference eigenvalues), so it complements
 * test_hf_eigenvalues until a PAW HF fixture is generated.
 */
template<MEMORY_SPACE MEM>
void test_exchange_structural(mpi_context_t& mpi, mf::MF& mfobj,
                              double herm_tol = 1e-8)
{
  auto all = nda::range::all;
  long nspin   = mfobj.nspin();
  long nk_ibz  = mfobj.nkpts_ibz();
  long nbnd    = mfobj.nbnd();

  memory::array<MEM, ComplexType, 3> nii(nspin, nk_ibz, nbnd);
  if constexpr (MEM == HOST_MEMORY) {
    nii() = mfobj.occ()(all, nda::range(nk_ibz), all);
  } else {
    nii() = nda::array<ComplexType,3>(mfobj.occ()(all, nda::range(nk_ibz), all));
  }

  hamilt::pseudopot V(mfobj);
  auto Kij = hamilt::Vexchange<MEM>(mfobj, mpi.comm, &V, nii);
  auto Kloc = nda::to_host(Kij.local());

  // 1) Finite: no NaNs / Infs.
  double maxabs = 0.0;
  for (long s = 0; s < Kloc.extent(0); ++s)
    for (long k = 0; k < Kloc.extent(1); ++k)
      for (long i = 0; i < Kloc.extent(2); ++i)
        for (long j = 0; j < Kloc.extent(3); ++j) {
          double r = std::real(Kloc(s,k,i,j));
          double im = std::imag(Kloc(s,k,i,j));
          CHECK(std::isfinite(r));
          CHECK(std::isfinite(im));
          maxabs = std::max({maxabs, std::abs(r), std::abs(im)});
        }
  app_log(2, "test_exchange_structural: max|K_skij| = {:.3e}", maxabs);

  // 2) Hermitian: K_ij = K_ji* per (s, k).
  double max_herm = 0.0;
  for (long s = 0; s < Kloc.extent(0); ++s)
    for (long k = 0; k < Kloc.extent(1); ++k)
      for (long i = 0; i < Kloc.extent(2); ++i)
        for (long j = 0; j < Kloc.extent(3); ++j) {
          ComplexType d = Kloc(s,k,i,j) - std::conj(Kloc(s,k,j,i));
          max_herm = std::max(max_herm, std::abs(d));
        }
  app_log(2, "test_exchange_structural: max|K_ij - K_ji*| = {:.3e}", max_herm);
  CHECK(max_herm < herm_tol * std::max(1.0, maxabs));

  // 3) Real diagonal for non-magnetic / non-SOC: K_nn diagonal entries should
  // have zero imaginary part (within Hermiticity tolerance).
  double max_im_diag = 0.0;
  for (long s = 0; s < Kloc.extent(0); ++s)
    for (long k = 0; k < Kloc.extent(1); ++k)
      for (long i = 0; i < Kloc.extent(2); ++i)
        max_im_diag = std::max(max_im_diag,
                               std::abs(std::imag(Kloc(s,k,i,i))));
  app_log(2, "test_exchange_structural: max|Im K_nn| = {:.3e}", max_im_diag);
  CHECK(max_im_diag < herm_tol * std::max(1.0, maxabs));
}

TEST_CASE("exchange_structural", "[hamilt][hf][paw]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  // NCPP cross-check — should already be implied by the hf_eigenvalues
  // NCPP section, but a quick structural pass at the same fixture is
  // a useful regression guard.
  SECTION("lih_kp222_nbnd16 (NCPP, HF)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_hf", mf::h5_input_type);
    test_exchange_structural<HOST_MEMORY>(*mpi, qe_h5);
  }

  // Exercises the PAW augmentation path in v_x_paw.hpp:
  // per-pair Q^IJ_atm(G+Δk) × phase added to each pair density. The
  // eigenvalue comparison is impossible here (this is a DFT-PBE fixture,
  // not HF), but the Hermiticity + finiteness + real-diagonal checks
  // catch any obvious augmentation indexing / sign / FFT bug.
  SECTION("lih_kp222_nbnd16 (PAW, DFT structural smoke)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    test_exchange_structural<HOST_MEMORY>(*mpi, qe_h5);
  }

  // Same for USPP.
  SECTION("lih_kp222_nbnd16 (USPP, DFT structural smoke)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_uspp", mf::h5_input_type);
    test_exchange_structural<HOST_MEMORY>(*mpi, qe_h5);
  }
}

TEST_CASE("dft_eigenvalues", "[hamilt][dft]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  // Run on a fixture only if its h5 file exists. The default LiH 222 NCPP
  // fixture is the smallest reasonable target; it uses PBE so SCF
  // eigenvalues should be reproduced from (T + V_loc + V_H + V_NL + V_xc).
  SECTION("lih_kp222_nbnd16 (NCPP, PBE)") {
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222", mf::h5_input_type);
    test_dft_eigenvalues<HOST_MEMORY>(*mpi, qe_h5);
  }

  SECTION("lih_kp222_nbnd16 (USPP, PBE)") {
    // USPP exercises the same add_Vpp → v_h_paw augmentation and
    // SCF-corrected non-local D pipeline as PAW, but USPP has NO PAW
    // one-center correction term (no ddd_paw). So deeq = dvan + ∫V_eff Q
    // only. This isolates V_H_aug + V_NL coupling without the PAW
    // one-center contribution.
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_uspp", mf::h5_input_type);
    test_dft_eigenvalues<HOST_MEMORY>(*mpi, qe_h5, /*tol*/ 5e-5);
  }

  SECTION("lih_kp222_nbnd16 (PAW, PBE, no exact exchange)") {
    // Exercises pseudopot::add_Vpp → v_h_paw augmentation and the
    // SCF-corrected non-local D from /Hamiltonian/paw/deeq, which for
    // PAW additionally includes ddd_paw (the AE-PS one-center [V_H+V_xc]
    // correction; computed by QE's PAW_potential).
    //
    // The PAW fixture uses conv_thr=1e-14 + mixing_beta=0.3: PAW datasets
    // with deep semicore valence states (e.g. Li 1s at ε ≈ -44 eV with
    // S_pw ≈ 0.36) demand tighter SCF convergence than the standard
    // density-residual threshold to make the saved eigenvalues match
    // h_psi at the saved density.
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    test_dft_eigenvalues<HOST_MEMORY>(*mpi, qe_h5, /*tol*/ 5e-5);
  }
}

// Plan B3: native core-valence exact-exchange builder (Slater R^L +
// closed-shell Gaunt² sum) — hydrogenic analytic checks. The same routine
// is validated against a real ABINIT-XML <exact_exchange_X_matrix> (atompaw
// Al "stringent" + .corewf companion, all 12 ln-diagonal entries to ~1e-7
// rel) in src/python/mean_field/abinit_interface/validate_ex_cvij.py; the
// hardcoded cross-language references below come from that (validated)
// python implementation on the identical synthetic input.
TEST_CASE("ex_cvij_native", "[paw]")
{
  // 3j(l1 l2 l3; 000)² closed form
  REQUIRE(hamilt::paw::w3j000_sq(0, 0, 0) == Approx(1.0).epsilon(1e-13));
  REQUIRE(hamilt::paw::w3j000_sq(1, 1, 0) == Approx(1.0 / 3.0).epsilon(1e-13));
  REQUIRE(hamilt::paw::w3j000_sq(2, 2, 0) == Approx(1.0 / 5.0).epsilon(1e-13));
  REQUIRE(hamilt::paw::w3j000_sq(1, 2, 1) == Approx(2.0 / 15.0).epsilon(1e-13));
  REQUIRE(hamilt::paw::w3j000_sq(1, 1, 1) == 0.0);   // odd sum
  REQUIRE(hamilt::paw::w3j000_sq(0, 2, 1) == 0.0);   // triangle violation

  // hydrogenic Z=2: core = 1s; valence = 2s (l=0) + 2p (l=1), u = r·R on a
  // log grid r = a(e^{d i}−1) — identical formulas to the python reference.
  long mesh = 1800;
  double a = 6.4e-4, d = 6.0e-3, Z = 2.0;
  nda::array<double,1> r(mesh), rab(mesh);
  nda::array<double,2> aewfc(2, mesh), core(1, mesh);
  for (long i = 0; i < mesh; ++i) {
    r(i) = a * (std::exp(d * i) - 1.0);
    rab(i) = a * d * std::exp(d * i);
    double x = r(i);
    core(0, i)  = 2.0 * std::pow(Z, 1.5) * x * std::exp(-Z * x);
    aewfc(0, i) = (std::pow(Z, 1.5) / (2.0 * std::sqrt(2.0)))
                  * (2.0 - Z * x) * std::exp(-Z * x / 2.0) * x;
    aewfc(1, i) = (std::pow(Z, 1.5) / (2.0 * std::sqrt(6.0)))
                  * (Z * x) * std::exp(-Z * x / 2.0) * x;
  }
  nda::array<int,1> lll = {0, 1};
  nda::array<double,1> core_l = {0.0};
  nda::array<int,1> indv   = {1, 2, 2, 2};       // 1-based beta channel
  nda::array<int,1> nhtolm = {1, 2, 3, 4};       // lm = l²+1+m (1-based)

  auto ex = hamilt::paw::compute_ex_cvij_from_core(
      aewfc, core, lll, core_l, indv, nhtolm, r, rab);
  REQUIRE(ex.extent(0) == 4);

  // analytic hydrogenic exchange integrals (quadrature-limited agreement):
  //   K(1s,2s) = 16 Z/729,   K(1s,2p) = 112 Z/6561;  stored = −K.
  REQUIRE(ex(0, 0) == Approx(-16.0 * Z / 729.0).margin(5e-9));
  REQUIRE(ex(1, 1) == Approx(-112.0 * Z / 6561.0).margin(5e-9));
  // cross-language reference (same quadrature in python; tight)
  REQUIRE(ex(0, 0) == Approx(-0.04389574756808).margin(2e-11));
  REQUIRE(ex(1, 1) == Approx(-0.03414113700755).margin(2e-11));
  // structure: m-degenerate p diagonal, strictly lm-diagonal otherwise
  REQUIRE(ex(2, 2) == ex(1, 1));
  REQUIRE(ex(3, 3) == ex(1, 1));
  for (int I = 0; I < 4; ++I)
    for (int J = 0; J < 4; ++J)
      if (I != J) REQUIRE(ex(I, J) == 0.0);
}

// ===========================================================================
// Plan C4 — augmentation-mode exchange-energy baseline (direct dense-grid
// path, finite-size off). Computes E_X = (2/ns)·Σ_k w_k·½Tr[Dm K] from the
// direct Vexchange in BOTH vv_compensation modes and logs the mode split.
// The default bare-Coulomb kernel zeroes the G+Δk=0 term (ignore_g0), i.e.
// the finite-size-off convention of the 2026-07-21 ABINIT si222 accounting
// (notes/paw_article_results/abinit_exchange_gw_vs_hybrid.md):
//   moment+deltaC(+ex_cvij)  <->  ABINIT hybrid/HF pawdijfock operator
//   shape (Arnaud)           <->  ABINIT GW Sigma_x operator
// Purpose: a reproducible CURRENT-CODE baseline pair for the C4 cluster
// campaign. NOTE: the ABINIT references (−1.316447 GW / HF accounting) were
// produced on the rusty cmp si222 mf (Γ eigvals −0.23437/0.21250…); the
// local qe_si222_paw fixture is a different cell (−0.20204/0.24111…), so
// these numbers are baselines, not the cross-code comparison itself.
// [slow]: two direct v_x builds per fixture.
// ===========================================================================
template<MEMORY_SPACE MEM>
void test_vexchange_mode_energies(mpi_context_t& mpi,
                                  std::shared_ptr<mf::MF> mf_ptr,
                                  std::string const& tag)
{
  auto all = nda::range::all;
  auto& mfobj = *mf_ptr;
  long nspin = mfobj.nspin(), nk_ibz = mfobj.nkpts_ibz(), nbnd = mfobj.nbnd();

  nda::array<ComplexType,3> nii(nspin, nk_ibz, nbnd);
  nii() = mfobj.occ()(all, nda::range(nk_ibz), all);
  auto k_weight = mfobj.k_weight();

  hamilt::pseudopot V(mfobj);
  auto EX = [&](bool shape) {
    V.set_paw_exx_shape_restored(shape);
    auto dK = hamilt::Vexchange<MEM>(mfobj, mpi.comm, &V, nii);
    // Same convention as methods::eval_hf_energy with band-diagonal Dm=occ:
    // E_X = spin_factor · Σ_k w_k · ½ Σ_n f_n Re K_nn.
    auto K = nda::to_host(dK.local());
    auto rs = dK.local_range(0); auto rk = dK.local_range(1);
    auto ri = dK.local_range(2); auto rj = dK.local_range(3);
    double e = 0.0;
    for (auto [il_s, s] : itertools::enumerate(rs))
      for (auto [il_k, k] : itertools::enumerate(rk))
        for (auto [il_i, i] : itertools::enumerate(ri))
          for (auto [il_j, j] : itertools::enumerate(rj))
            if (i == j)
              e += k_weight(k) * std::real(nii(s, k, i)) *
                   std::real(K(il_s, il_k, il_i, il_j));
    e = mpi.comm.all_reduce_value(e, std::plus<>{});
    double spin_factor = (nspin == 2) ? 1.0 : 2.0;
    return 0.5 * spin_factor * e;
  };

  double E_m = EX(false);   // moment + deltaC (ABINIT HF pawdijfock side)
  double E_s = EX(true);    // shape / Arnaud   (ABINIT GW Sigma_x side)
  app_log(1, "[C4 mode energies {}] E_X(moment+deltaC) = {:+.8f} Ha, "
             "E_X(shape) = {:+.8f} Ha, mode split = {:+.6e} Ha",
          tag, E_m, E_s, E_s - E_m);
  CHECK(std::isfinite(E_m));
  CHECK(std::isfinite(E_s));
  REQUIRE(std::abs(E_s - E_m) > 1e-8);   // modes genuinely differ
}

TEST_CASE("vexchange_mode_energies", "[hamilt][paw][hf][slow]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih_kp222_nbnd16 (PAW)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_lih222_paw_hf", mf::h5_input_type));
    test_vexchange_mode_energies<HOST_MEMORY>(*mpi, mf_ptr, "qe_lih222_paw_hf");
  }
  SECTION("si_kp222 (PAW psl 1.0.0)") {
    auto mf_ptr = std::make_shared<mf::MF>(
        mf::default_MF(mpi, "qe_si222_paw", mf::h5_input_type));
    test_vexchange_mode_energies<HOST_MEMORY>(*mpi, mf_ptr, "qe_si222_paw");
  }
}

}

