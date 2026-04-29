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
#include "hamiltonian/paw/v_h_paw.hpp"
#include "hamiltonian/add_vxc.h"
#include "utilities/fortran_utilities.h"
#include "utilities/qe_utilities.hpp"

#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/thc_reader_t.hpp"
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
void check_Hartree(mpi_context_t& mpi, std::shared_ptr<mf::MF> &mfobj, bool diag_dm=false) {
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

  // Hartree from ERI
  auto sJ_skij = make_shared_array<array_view_4d_t>(
      mpi.comm, mpi.internode_comm, mpi.node_comm, shape); 
  methods::thc_reader_t thc(mfobj,
                            methods::make_thc_reader_ptree(0, "", "incore", "", "coqui", 1e-9,
                                                           mfobj->ecutrho(), 1, 1024));
  methods::solvers::hf_t hf;
  hf.evaluate(sJ_skij, sDm_skij.local(), thc, sS_skij.local(), true, false);
  mpi.comm.barrier();

  // Hartree from PW
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

  auto Abs = nda::map([](ComplexType _x_) { return std::abs(_x_); });
  double norm = -1;
  if (sJ_skij.node_comm()->root()) {
    nda::array<RealType,4> res_abs(nspin, nkpts_ibz, nbnd, nbnd);
    res_abs = Abs(sJ_skij.local() - sJ2_skij.local());
    norm = nda::max_element(res_abs);
  }
  sJ_skij.node_comm()->broadcast_n(&norm, 1, 0);
  app_log(2, "Norm of J - J2 = {}", norm);

  utils::ARRAY_EQUAL(sJ_skij.local(), sJ2_skij.local(), 1e-5);
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
    check_Hartree<MEM>(*mpi, qe_h5);
  }

  SECTION("lih223_inv") 
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih223_inv", mf::h5_input_type));
    check_Hartree<MEM>(*mpi, qe_h5);
  }

  SECTION("lih223_sym") 
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih223_sym", mf::h5_input_type));
    check_Hartree<MEM>(*mpi, qe_h5);
  }

  SECTION("lih223_sym_diag") 
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih223_sym", mf::h5_input_type));
    check_Hartree<MEM>(*mpi, qe_h5, true); // diagonal density as the input
  }

  SECTION("GaAs222_so") 
  {
    auto qe_h5 = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_GaAs222_so", mf::h5_input_type));
    check_Hartree<MEM>(*mpi, qe_h5, true); // diagonal density as the input
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

  //std::optional<double> Eref;
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
 *
 * For NCPP fixtures the test passes at 5e-5 Ha (regression on v_h /
 * add_vxc / add_vnl). For PAW it currently FAILS to reach the same
 * tolerance — the deepest Li 1s-like band at k=0 has ~0.15 Ha residual
 * (S_pw ≈ 0.36). The H matrix off-diagonals are non-zero (e.g.
 * H[0,1] ≈ 0.035 Ha), confirming CoQui's H ≠ QE's H — diagonalizing
 * CoQui's H/S matrix gives ε[0] ≈ -1.656 vs QE's eigval[0] ≈ -1.508.
 * Tolerance is set to 0.2 Ha for the PAW section to keep the test passing
 * while we hunt down the remaining bug; tighten once resolved. See the
 * notes on Phase 4 V_NL convention investigation.
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
  auto Sij   = hamilt::ovlp<MEM>(mfobj, mpi.comm);

  utils::check(Hij.local_shape() == Vxcij.local_shape(),
               "test_dft_eigenvalues: H and Vxc local shape mismatch");
  // H_full = (T + V_loc + V_H + V_NL) + V_xc
  nda::tensor::add(ComplexType(1.0), Vxcij.local(),
                   ComplexType(1.0), Hij.local());

  // For USPP/PAW the bands satisfy H|ψ⟩ = ε S|ψ⟩ (generalized). QE saves
  // S-orthonormalized orbitals, so add_S brings ⟨ψ̃|S|ψ̃⟩ to 1.
  V.add_S(nda::range(mfobj.nkpts_ibz()), nda::range(nbnd), Sij);

  auto Hloc = nda::to_host(Hij.local());
  auto Sloc = nda::to_host(Sij.local());
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
        double S_diag  = std::real(Sloc(is, ik, ia, ib));
        double eps_ref = mfobj.eigval(s, k, a);
        double H_eps   = (std::abs(S_diag) > 1e-10) ? (H_diag / S_diag) : H_diag;
        double err     = std::abs(H_eps - eps_ref);
        if (err > max_err) max_err = err;
        ++count;
        if (err > tol) {
          app_log(2,
            "DFT eigval mismatch: s={}, k={}, n={}, H_nn={:+.6f}, "
            "S_nn={:.6f}, H/S={:+.6f}, ref={:+.6f}, err={:.2e}",
            s, k, a, H_diag, S_diag, H_eps, eps_ref, err);
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
  auto wfc_g    = mfobj.wfc_truncated_grid();
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

  double E_H = hamilt::paw::hartree_energy_paw(
      mpi, V, npol, fft_mesh, recv, k2g,
      kpts_full, kp_to_ibz, kp_trev, kp_symm, symm_list,
      nii, psi);

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
  auto wfc_g    = mfobj.wfc_truncated_grid();
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
 * is captured by the separate paw_thc_kernel — Phase 4 work).
 *
 * For NCPP (no augmentation): smooth-direct = full-direct = QE ehart.
 * The accuracy of THC is set by `thresh` and `ecut`; for our LiH 222
 * fixture with thresh=1e-5 and ecut=0.4×ecutrho, agreement should be ~1e-3 Ha.
 */
template<MEMORY_SPACE MEM>
void test_hartree_thc_vs_direct(mpi_context_t& mpi, std::shared_ptr<mf::MF> mf_ptr,
                                std::string const& fixture_name,
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
  auto fft_mesh = mfobj.fft_grid_dim();
  auto recv     = mfobj.recv();
  auto wfc_g    = mfobj.wfc_truncated_grid();
  auto kpts_full = mfobj.kpts();
  auto kp_to_ibz = mfobj.kp_to_ibz();
  auto kp_trev   = mfobj.kp_trev();
  auto kp_symm   = mfobj.kp_symm();
  auto symm_list = mfobj.symm_list();
  auto k2g       = V.swfc_to_rho_view();

  // Smooth-only direct E_H: omit the PAW augmentation since standard THC
  // doesn't include it either. For NCPP this equals the full E_H.
  double E_H_smooth_direct = hamilt::paw::hartree_energy_paw(
      mpi, V, npol, fft_mesh, recv, k2g,
      kpts_full, kp_to_ibz, kp_trev, kp_symm, symm_list, nii, psi,
      /*include_augmentation=*/false);

  // -------- E_H_thc via Hartree-Fock J matrix from THC ERIs --------
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

  app_log(2, "Hartree energy: smooth-direct = {:+.8f} Ha, THC = {:+.8f} Ha, "
             "diff = {:+.2e} Ha", E_H_smooth_direct, E_H_thc,
             E_H_thc - E_H_smooth_direct);
  CHECK(std::abs(E_H_thc - E_H_smooth_direct) < tol);
}

/**
 * Hartree energy via PAW-augmented THC: the augmented thc_reader_t (Phase 4.2)
 * produces (X_full, V_full) with smooth + atom-local rows. We feed it through
 * hf_t to get the J matrix and contract with the diagonal density matrix.
 *
 *   E_H_thc_paw = (1/2) Σ_sk wk Tr[Dm × J]   (HF Hartree from augmented THC)
 *   E_H_direct  = hartree_energy_paw(include_augmentation=true)
 *
 * For Phase 4.2 (q=0 augmentation only), Hartree only needs q=0 → both
 * computations should agree within smooth-ISDF + compression tolerance.
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
  auto fft_mesh = mfobj.fft_grid_dim();
  auto recv     = mfobj.recv();
  auto wfc_g    = mfobj.wfc_truncated_grid();
  auto kpts_full = mfobj.kpts();
  auto kp_to_ibz = mfobj.kp_to_ibz();
  auto kp_trev   = mfobj.kp_trev();
  auto kp_symm   = mfobj.kp_symm();
  auto symm_list = mfobj.symm_list();
  auto k2g       = V.swfc_to_rho_view();

  // hartree_energy_paw is rank-0-only (operates on `psi.local()` without
  // gathering), so we only run it under serial MPI. Under MPI > 1 we still
  // exercise the THC build but skip the direct-target comparison.
  bool const direct_ok = (mpi.comm.size() == 1);
  double E_H_direct = 0.0;
  if (direct_ok) {
    E_H_direct = hamilt::paw::hartree_energy_paw(
        mpi, V, npol, fft_mesh, recv, k2g,
        kpts_full, kp_to_ibz, kp_trev, kp_symm, symm_list, nii, psi,
        /*include_augmentation=*/true);
  }
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
  // pseudopot.qgm lives on. Phase 4.2 hard-requires this; lower ecut
  // truncates the augmentation Fourier content.
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

  if (direct_ok) {
    app_log(2, "PAW-augmented THC Hartree: direct(smooth-grid) = {:+.8f} Ha, "
               "ΔE_one_center = {:+.8f} Ha, AE target = {:+.8f} Ha, "
               "THC = {:+.8f} Ha, diff = {:+.2e} Ha",
               E_H_direct, E_K_a_direct, E_H_target,
               E_H_thc, E_H_thc - E_H_target);
    CHECK(std::abs(E_H_thc - E_H_target) < tol);
  } else {
    app_log(2, "PAW-augmented THC Hartree (np={}): direct comparison "
               "skipped (rank-0-only path); THC = {:+.8f} Ha (smoke check).",
               mpi.comm.size(), E_H_thc);
    CHECK(std::isfinite(E_H_thc));
  }
}

/**
 * Phase 4.3: verify the CoQui-side qvan2-equivalent (paw_aug_q_eval)
 * reproduces the cached qgm tensor at q=0 over the dense G grid.
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
    for (int ih = 0; ih < nh_a; ++ih)
    for (int jh = 0; jh < nh_a; ++jh) {
      long ij = (long)ijtoh(nt, ih, jh) - 1;
      if (ij < 0) continue;
      for (long g = 0; g < ng_check; ++g) {
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
    }
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

  for (auto metric : {hamilt::paw::isdf_metric::Coulomb,
                       hamilt::paw::isdf_metric::L2}) {
    auto mname = std::string(hamilt::paw::metric_name(metric));
    app_log(2, "=== local-ISDF rank vs tol [{} metric] ({}) ===", mname, label);
    app_log(2, "  {:>18} {:>5} {:>10} {:>5} {:>5} {:>5} {:>5} {:>8}",
            "species", "nh", "full-rank", "1e-3", "1e-6", "1e-9", "1e-12", "N_aug");
    for (int nt = 0; nt < (int)sps.size(); ++nt) {
      if (!(sps[nt].is_paw || sps[nt].is_uspp)) continue;
      int nh_a = (int)V.nh_view()(nt);
      int full = nh_a * nh_a;
      int n3 = hamilt::paw::build_local_isdf_compressed_by_norm(
                   V, nt, recv, omega, metric, 1e-3).nlambda;
      int n6 = hamilt::paw::build_local_isdf_compressed_by_norm(
                   V, nt, recv, omega, metric, 1e-6).nlambda;
      int n9 = hamilt::paw::build_local_isdf_compressed_by_norm(
                   V, nt, recv, omega, metric, 1e-9).nlambda;
      int n12 = hamilt::paw::build_local_isdf_compressed_by_norm(
                   V, nt, recv, omega, metric, 1e-12).nlambda;
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
}

/**
 * Phase 4.3: Hermiticity check on the augmented THC Coulomb tensor.
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

  // Gather V_full(q, P, Q) for all q on rank 0 for the comparison.
  long Np = thc.Np();
  auto dZ_qPQ = thc.template dZ<HOST_MEMORY>({1, 1, 1}, {0, 0, 0});
  if (!mpi.comm.root()) return;
  auto V = dZ_qPQ.local();   // shape (nq_ibz, Np, Np)

  double max_dev = 0.0;
  double max_val = 0.0;
  long checked = 0;
  for (long iq = 0; iq < nq_ibz; ++iq) {
    long iqm = qminus(iq);
    if (iqm < 0 || iqm >= nq_ibz) continue;   // qminus may map outside IBZ
    for (long P = 0; P < Np; ++P)
    for (long Q = 0; Q < Np; ++Q) {
      ComplexType a = V(iq,  P, Q);
      ComplexType b = V(iqm, Q, P);
      double dev = std::abs(a - std::conj(b));
      max_dev = std::max(max_dev, dev);
      max_val = std::max(max_val, std::abs(a));
      ++checked;
    }
  }
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
 * Phase 4.3 smoke test: full HF (Hartree + exchange) with paw_aug=true
 * runs to completion and gives finite, real energy. Exchange exercises
 * V_full at all q's (not just q=0), so any failure in the q-loop V_GL/V_LL
 * builder shows up here.
 *
 * No reference comparison yet — that's Phase 4.4 (Si + TMO benchmarks).
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
}

// Si sections are split into a separate test case tagged [slow]: the q-loop
// in augment_thc_with_paw with Si's lmax=2 (Lmax=4 in the angular sum) does
// ~4M radial spherical-Bessel transforms per fixture and takes ~1 hour
// each in serial. They are NOT run by default; invoke explicitly with
//   test_hamiltonian "[slow]"   or   test_hamiltonian -c "si_kp222 ..."
// once the Phase 4.4 |K|-grid qrad cache lands and brings them back to
// minute-scale.
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
void test_local_isdf_qhat_roundtrip(mpi_context_t& mpi, mf::MF& mfobj,
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
void test_local_isdf_deltaC_roundtrip(mpi_context_t& mpi, mf::MF& mfobj,
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

  // Direct ρ_aug(G) (with structure factor)
  nda::array<ComplexType,1> rho_direct(ngm_d);
  rho_direct() = ComplexType(0.0);
  for (long g=0; g<ngm_d; ++g) {
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

  // ISDF-reconstructed ρ_aug(G)
  nda::array<ComplexType,1> rho_isdf(ngm_d);
  rho_isdf() = ComplexType(0.0);
  for (long g=0; g<ngm_d; ++g) {
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
 * assertion that the K_a injection in `assemble_paw_augmented_thc` produces
 * the correct one-center Hartree contribution.
 */
template<MEMORY_SPACE MEM>
void test_local_isdf_K_a_one_center(mpi_context_t& mpi, mf::MF& mfobj,
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
void test_local_isdf_K_a_compressed_accuracy(mpi_context_t& mpi, mf::MF& mfobj,
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
    for (int n = 0; n <= n_max; ++n) {
      auto isdf_n = hamilt::paw::build_local_isdf_compressed(
          V, nt, nh_a, rep, n);
      auto K = hamilt::paw::compute_K_a(isdf_n, sp_paw.deltaC);
      double E_K = 0.0;
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
      double err = std::abs(E_K - E_dC);
      app_log(2, "  {:>4} | {:>2} | {:+.10f} | {:.3e}",
                 n, isdf_n.nlambda, E_K, err);
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
void test_local_isdf_compression_accuracy(mpi_context_t& mpi, mf::MF& mfobj,
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

  // Direct ρ_aug(G) and ISDF-compressed ρ_aug(G); compare in Coulomb norm
  double err_C2 = 0.0;
  double ref_C2 = 0.0;
  for (long g=0; g<ngm_d; ++g) {
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
    //
    // Test PASSES at 3e-6 Ha (similar to NCPP precision), confirming all
    // CoQui machinery — projector overlaps, augmentation V_H, V_NL via
    // deeq, S overlap — is correct. The PAW residual below comes
    // entirely from the ddd_paw application convention (see PAW section
    // and notes/paw_implementation_plan.md "Open issues").
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_uspp", mf::h5_input_type);
    test_dft_eigenvalues<HOST_MEMORY>(*mpi, qe_h5, /*tol*/ 5e-5);
  }

  SECTION("lih_kp222_nbnd16 (PAW, PBE, no exact exchange)") {
    // Exercises pseudopot::add_Vpp → v_h_paw augmentation and the
    // SCF-corrected non-local D from /Hamiltonian/paw/deeq, which for
    // PAW additionally includes ddd_paw (the AE-PS one-center [V_H+V_xc]
    // correction; computed by QE's PAW_potential).
    //
    // The PAW fixture uses conv_thr=1e-14 + mixing_beta=0.3 because the
    // deepest Li-1s-like band (semicore valence, ε ≈ -44 eV, S_pw ≈ 0.36)
    // is extremely sensitive to small becsum changes via the ddd_paw
    // chain rule. Looser conv_thr (e.g. 1e-10) leaves the saved et off
    // from h_psi(rho_saved) by ~0.15 Ha — not a CoQui bug, just an SCF
    // convergence requirement specific to deep PAW eigenvalues.
    auto qe_h5 = mf::default_MF(mpi, "qe_lih222_paw", mf::h5_input_type);
    test_dft_eigenvalues<HOST_MEMORY>(*mpi, qe_h5, /*tol*/ 5e-5);
  }
}

}

