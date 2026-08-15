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


#pragma once

#include <filesystem>
#include <algorithm>
#include <vector>
#include "configuration.hpp"
#include <nda/nda.hpp>
#include <nda/blas.hpp>
#include "utilities/concepts.hpp"
#include "utilities/Timer.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/sparse/csr_utils.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/pseudo/pseudopot_query.h"

namespace utils
{

/*
 * MAM: You only need d(R,k) for k_IBZ and for R in the little group of k_IBZ.
 *      Use the fact that any rotation in the left coset can be written in terms of
 *      the chosen symmetry at that kpoint and some element of the little group at k_IBZ.
 *      This is the same idea used to calculate orbital orverlaps for wannier90 from IBZ data only.
 *
 * Generates rotation matrices between symmetry related kpoints in the BZ, 
 * defined as:
 *     d(R, k)(a,b) = int_r conj(psi(kS^{-1},a,S*r)) * psi(k,b,r). 
 * Notice that, for k outside the IBZ, d( R, k ) = d( R*R0, k_ibz), 
 * where k_IBZ*R0^{-1} = k. Hence, we only need to calculate 'd' explicitly for k in IBZ
 * and for all {R, R0(k)}. If k involves trev symmetry (trev(k) == true), then:
 * d( R, k ) = conj(d( R*R0, k_ibz)).
 * The routine returns:
 *   - nda::array<int,2>(s,k) -> n: location of the rotation matrix in the array 
 *                                  for a given {s,k} pair, k in the full BZ.
 *   - (if Sparse==true:)  std::vector<sparse_mat> with the list of rotation matrices
 *                         (indexed by mapping) in sparse format
 *   - (if Sparse==false:) shared_array<C,3>(n,a,b) with the rotation matrices
 *                         (indexed by mapping) in dense tensor format
*/
// MAM: Need to be able to allocate sparse matrix in shared memory!
//      If the matrix is fairly dense, then this will eat up a lot of memory!
template<bool Sparse = true, typename MF_t>
auto generate_dmatrix(MF_t &mf,
                      std::vector<utils::symm_op> const& symms, 
                      nda::ArrayOfRank<1> auto const& slist,
                      hamilt::pseudopot *ps = nullptr, 
                      bool assume_irreducible=true,
                      bool normalize=true)
{
  constexpr auto MEM = HOST_MEMORY;
  using Array_view_3D_t = memory::array_view<MEM,ComplexType,3>;
  using sp_mat = typename math::sparse::csr_matrix<ComplexType,MEM,int,int>;
  using math::shm::make_shared_array;
  using nda::range;
  auto all = range::all;

  // checks!
  utils::check(mf.has_wfc_grid(), "Error in generate_dmat: has_wfc_grid==false");
  utils::check(mf.nspin() == 1, "Error in generate_dmat: Implement nspin>1");
  utils::check(mf.npol() == 1, "Error in generate_dmat: Implement npol>1");

  auto mpi = mf.mpi();
  auto comm = mpi->comm;
  auto node_comm = mpi->node_comm;
  // vector of sparse matrices, used only if Sparse==true
  std::vector<sp_mat> dmat_s;
  // shared memory array, used only if Sparse==false. Will resize later if needed
  auto shm_dmat = make_shared_array<Array_view_3D_t>(*mpi, {0,mf.nbnd(),mf.nbnd()});
  if(slist.extent(0) == 0) {
    nda::array<int,2> sk_to_n(0,0); 
    if constexpr (Sparse) {
      return std::make_tuple(sk_to_n,dmat_s);
    } else {
      return std::make_tuple(sk_to_n,shm_dmat);
    }
  }
  
  auto const & eigv = mf.get_sys().eigval();
  bool irred = (assume_irreducible and 
                std::all_of(eigv.begin(),eigv.end(),
                            [&](auto const& a) {return a!=0;}));
  auto const& bz = mf.bz();
  long nkpts = bz.nkpts;
  long nkpts_ibz = bz.nkpts_ibz;
  long nkpts_trev_pairs = bz.nkpts_trev_pairs;
  auto kpts = mf.kpts_crystal();
  auto const & kp_to_ibz = bz.kp_to_ibz;
  auto const & kp_symm = bz.kp_symm;
  long nbnd = mf.nbnd();
  long ns = slist.extent(0);
  long nstot = symms.size();
  auto wfc_g = mf.wfc_truncated_grid();
  long ngm = wfc_g->size();
  auto mesh = wfc_g->mesh();
  // index of G vectors in fft grid (G vectors inside the cutoff)
  auto gv_to_fft = wfc_g->gv_to_fft();   
  // inverse mapping to gv_to_fft
  auto fft_to_gv = wfc_g->fft_to_gv();   
  nda::array<ComplexType,1> *Xft = nullptr;
  nda::stack_array<double,3> kp;
  nda::stack_array<double,3> Gs;
  nda::array<double,2> RR0(3,3);
  memory::unified_array<long,1> Gout(ngm);
  // 
  bool needs_aug = hamilt::mf_requires_augmentation(mf);
  utils::check(not needs_aug or ps!=nullptr, "Error in generate_dmatrix: Nullptr pseudopotential pointer with PAW/USPP.");

  if(irred) {
    // a hardcoded threshold for sorted eigenvalues.
    double thresh = 1e-6;
    for (int ik=0; ik<nkpts; ++ik) {
      auto iter = std::adjacent_find(eigv(0,ik,all).begin(), eigv(0,ik,all).end(),
                                     [&thresh](double a, double b) { return a-b>thresh; });
      auto idx = std::distance(eigv(0,ik,all).begin(), iter);
      // don't use utils::check, since it will evaluate the parameters (e.g. in debug mode) which will
      // cause a seg fault when the check is fine 
      if( iter!=eigv(0,ik,all).end() )
        APP_ABORT("Error in generate_dmat: assume_irreducible with unsorted eigenvalues at ik={}: \n"
                   "Error in generate_dmat: assume_irreducible with unsorted eigenvalues at ik={}: \n"
                   "eigv(i={})={}, eigv(i+1={})={}, threshold = {}.",
                   ik, idx, eigv(0,ik,idx), idx+1, eigv(0,ik,idx+1), thresh);
    }
  }

  // cheating a bit, since it is hardwired to compare to kp! Careful!
  auto comp = [&kp](nda::ArrayOfRank<1> auto&& a) {
    // doing this by hand, not sure what's a better way
    double di = std::abs(a(0)-kp(0));
    double dj = std::abs(a(1)-kp(1));
    double dk = std::abs(a(2)-kp(2));
    di -= std::floor(di); if( std::abs(di-1.0) < 1e-4 ) di = 0.0;
    dj -= std::floor(dj); if( std::abs(dj-1.0) < 1e-4 ) dj = 0.0;
    dk -= std::floor(dk); if( std::abs(dk-1.0) < 1e-4 ) dk = 0.0;
    return di + dj + dk < 1e-12;
  };

  auto comp_R = [](nda::ArrayOfRank<2> auto&& R1, nda::ArrayOfRank<2> auto&& R2) {
      return nda::frobenius_norm(R1-R2) < 1e-10;
  };

  auto find_k2 = [&kpts,&comp] () {
    for(int ik=0; ik<kpts.extent(0); ik++) {
      if(comp(kpts(ik,nda::range::all))) 
        return ik; 
    }
    utils::check(false, "Could not find k2.");
    return -1;
  };

  // list of {symm,kp} pairs needed 
  std::vector<std::pair<int,int>> sk; 
  sk.reserve(ns*nkpts_ibz); 
  // first term is reserve for the identity
  sk.emplace_back(std::make_pair(-1,-1));  

  // returned mapping, from {s,k} to dmat index (e.g. some element in sk list) 
  nda::array<int,2> sk_to_n(ns,nkpts-nkpts_trev_pairs);
  sk_to_n() = -1; // safeguard 

  // array to keep track of existing symmetries in sk list 
  nda::array<int,1> sv(nstot);

  // all kpoints outside IBZ, excluding trev ones
  range k_rng(nkpts_ibz,nkpts-nkpts_trev_pairs); 

  // construct sk and sk_to_n arrays
  // find a way to parallelize this process
  for( auto ik : range(nkpts_ibz) ) {
    sv() = -1;
    // first push all from slist
    for( auto [is,s] : itertools::enumerate(slist) ) {
      // by convention, d( kp_symm(k), kp_to_ibz(k) ) = delta(i,j)
      auto s_ = s;  // clang bypass: capturing a structured binding is not yet supported in OpenMP
      if( std::any_of(k_rng.begin(), k_rng.end(), 
             [&](auto const&& a) {return ((kp_to_ibz(a)==ik) and (kp_symm(a)==s_));}) ) {
        sk_to_n(is,ik) = 0;  // default position of identity!
        sv(s) = 0; 
      } else {
        sk_to_n(is,ik) = sk.size();
        sv(s) = sk.size(); 
        sk.emplace_back(std::make_pair(int(s),int(ik)));
      }
    }
    // now add all k-points from outside IBZ that are symmetry related to ik 
    for( auto kr : k_rng ) {
      if(bz.kp_to_ibz(kr) != ik) continue;
      for( auto [is,s] : itertools::enumerate(slist) ) {
        // by convention, d( kp_symm(k)^{-1}, k ) = delta(i,j)
        if( comp_R(symms[kp_symm(kr)].Rinv,symms[s].R) ) {
          sk_to_n(is,kr) = 0;
        } else {
          // RR0
          nda::blas::gemm(1.0,symms[s].R,symms[kp_symm(kr)].R,0.0,RR0);
          // look for RR0 in symms
          int ss0 = -1;
          for(int ss=0; ss<nstot; ++ss) 
            if(comp_R(symms[ss].R,RR0)) { 
              ss0 = ss; 
              break; 
            }
          utils::check(ss0 >= 0 and ss0 < nstot, "Error in generate_dmat: Could not find compound symmetry");
          if( sv(ss0) >= 0 ) {
            sk_to_n(is,kr) = sv(ss0);
          } else {
            sk_to_n(is,kr) = sk.size();
            sv(ss0) = sk.size(); 
            sk.emplace_back(std::make_pair(ss0,int(ik)));
          }
        }
      }
    }
  }
  utils::check( std::all_of( range(nkpts-nkpts_trev_pairs).begin(), range(nkpts-nkpts_trev_pairs).end(),
             [&](auto const&& a) {return a>=0;}), "Error in generate_dmat: Unassigned (s,k) pair");

  // resize and setup temporary objects
  sp_mat sp_d({1,1},1);
  nda::array<ComplexType,2> Temp(0,0);
  if constexpr (Sparse) {
    int nz = (irred?long(8):nbnd);
    sp_d = sp_mat({nbnd,nbnd},nz);  
    Temp = nda::array<ComplexType,2>(nz,nz);
  } else {
    shm_dmat = make_shared_array<Array_view_3D_t>(*mpi, {sk.size(),nbnd,nbnd});
    shm_dmat.set_zero();
  }

// MAM: This assumes an orthonormal basis, the general formula is:
//      F = d * S, where F is the overlap evaluated below (the current dmat) and S is the overlap matrix 
// for states at k*Rinv. FIX!!!

  // The mesh needs to be compatible with all the relevant symmetry rotations.
  // If this is not the case, generate a new mesh, a new mapping from the old mesh to the new,
  // and apply rotations/contractions on the new mesh.
  // This does not have any implications outside this routine, only used as an intermediate space with appropriate
  // dimensions to allow symmetry rotations and matrix element calculations.
  // MAM: Currently forcing the mesh to be compatible from the start, but this can still be done if needed

  int kold = -1;
  auto [isk0, isk1] = itertools::chunk_range(0, sk.size(), comm.size() ,comm.rank());
  long bad_norm_count = 0;
  
  if(needs_aug) {

    // can't use shared arrays
    utils::check(MEM==HOST_MEMORY, "Finish PAW/USPP generate_dmatrix on GPU!");

    // MAM: Duplicating code to avoid constucting pseudopotential object unnecesarily.
    //      Can be cleaned up later...
    // for now, reading projectors from h5 here. Hide this later on
    nda::array<ComplexType,3> psi(3,nbnd,ngm);
    auto filename = ps->get_input_file_name();
    auto filetype = ps->get_input_file_type();
    // on site overlap matrix
    auto Qij = ps->Qij_view();
    // matrix elements between projectors and pseudo-orbitals
    auto Pskna = ps->Pskna_view();
    auto ityp = ps->ityp_view();
    auto nh = ps->nh_view();
    auto ofs = ps->ofs_view();
    long nkb = Pskna.extent(2);  // in principle this is nkb*npol
    memory::array<MEM,ComplexType,2> Fmaj(nbnd,nkb);
    memory::array<MEM,ComplexType,2> vkb(nkb,ngm);
    h5::file file;
    utils::check(filetype == mf::h5_input_type, "Error in generate_dmatrix: Only h5 input files allowed when using PAW/USPP and symmetries."); 
    utils::check(std::filesystem::exists(filename), "Error: Missing file: {}",filename);
    try {
      file = h5::file(filename, 'r');
    } catch(...) {
      utils::check(false,"Failed to open h5 file: {}, mode:r",filename);
    }
    h5::group grp0(file);
    h5::group grp1 = grp0.open_group("Hamiltonian");
    std::string type("");
    h5::h5_read_attribute(grp1, "pp_type", type);
    h5::group grp = grp1.open_group(type); 

    // copied from hamiltonian/pseudo/pseudopot.cpp, separate routine?
    // mapping from indexing in h5 and wfc mapping
    int npwx;
    h5::h5_read_attribute(grp,"max_npw",npwx);
    math::shm::shared_array<nda::array_view<long,2>> sk2g(*mpi,{nkpts_ibz,npwx});
    nda::array<ComplexType,2> buff(1,npwx);
    auto k2g = sk2g.local();
    auto fft2gv = wfc_g->fft_to_gv();
    long wfc_nnr = wfc_g->nnr();
    int rank = mpi->comm.rank();
    int np = mpi->comm.size();
    nda::array<int,1> npw(nkpts_ibz);
    nda::h5_read(grp,"npw",npw);

    // setup index mappings
    {
      long NX = wfc_g->mesh(0), NY = wfc_g->mesh(1), NZ = wfc_g->mesh(2);
      for( int ik=0; ik<nkpts_ibz; ik++ ) {
        if(  ik%np != rank ) continue;
        nda::array<int,2> mill(npw(ik),3);
        nda::h5_read(grp,"miller_k"+std::to_string(ik),mill);
        // map miller index to wfc_g truncated grid
        for( int i=0; i<npw(ik); i++ ) {
          long n1 = mill(i,0); if(n1<0) n1 += NX;
          long n2 = mill(i,1); if(n2<0) n2 += NY;
          long n3 = mill(i,2); if(n3<0) n3 += NZ;
          utils::check(n1 < NX, "read_vnl_h5: Index out of range. i:{}, n:{}, NX:{}",i,n1,NX);
          utils::check(n2 < NY, "read_vnl_h5: Index out of range. i:{}, n:{}, NY:{}",i,n2,NY);
          utils::check(n3 < NZ, "read_vnl_h5: Index out of range. i:{}, n:{}, NZ:{}",i,n3,NZ);
          long N = (n1*NY + n2)*NZ + n3;
          utils::check( N >= 0 and N < wfc_nnr, "read_vnl_h5: Index out of range. N:{}, nnr:{}",N,wfc_nnr);
          k2g(ik,i) = fft2gv(N);
          utils::check( k2g(ik,i) >= 0 and k2g(ik,i) < ngm, "read_vnl_h5: Index not mapped in truncated grid. ");
        }
      }
      mpi->comm.barrier();
      if(mpi->node_comm.root())
        mpi->internode_comm.all_reduce_in_place_n(k2g.data(),k2g.size(),std::plus<>{});
      mpi->comm.barrier();
    }

    int ispin = 0;

    for( long isk=isk0; isk<isk1; ++isk ) { 

      if(isk == 0) { // first term is always the identity
        if constexpr (Sparse) {
          dmat_s.emplace_back(math::sparse::identity<ComplexType,MEM>(nbnd));
        } else {
          auto dsk = shm_dmat.local()(isk,nda::ellipsis{});
          nda::diagonal(dsk) = ComplexType(1.0);
        }
        continue;
      }

      int is = sk[isk].first;
      int ik = sk[isk].second;

      // read orbital if necessary, add augmentation
      if(kold != ik) {

        kold = ik;
        mf.get_orbital_set('w',ispin,ik,{0,nbnd},psi(0,all,all)); 

        // keep a copy of pseudo orbitals
        psi(2,all,all) = psi(0,all,all);

        vkb() = ComplexType(0.0);
        // read projector from h5 and map to wfc grid
        for( int ib=0; ib<nkb; ++ib ) {
          auto b_k = buff(all,range(npw(ik)));
          auto tpl = std::tuple{range(ib,ib+1),range(npw(ik))};
          nda::h5_read(grp,"projector_k"+std::to_string(ik),b_k,tpl);
          for( auto [in,n] : itertools::enumerate(k2g(ik,range(npw(ik)))) )
            vkb(ib,n) = buff(0,in);
            //vkb(ib,n) = std::conj(buff(0,in));
        }

        // add augmentation terms: 
        //     psi(0,m,G) += sum_ai_aj Pskna(s,k,ai,m) Qij(ai,aj) conj(v(s,k,aj,G))
        // MAM: FIX for npol>1!!!
        for (auto [ia,nt] : itertools::enumerate(ityp)) {
          if (nh(nt) == 0) continue;
          nda::blas::gemm(ComplexType(1.0),
            nda::transpose(Pskna(ispin, ik,
                  range(ofs(ia), ofs(ia)+nh(nt)), range{0,nbnd})),
            Qij(nt, range(nh(nt)), range(nh(nt))),
            ComplexType(0.0),
            Fmaj(all, range(ofs(ia), ofs(ia)+nh(nt))));
        }
        nda::blas::gemm(ComplexType(1.0), Fmaj, vkb(all, all),
                        ComplexType(1.0), psi(0, all, all));

      }

      // kRinv = k2 - kp, kp = k2 - kRinv
      nda::blas::gemv(1.0,nda::transpose(symms[is].Rinv),kpts(ik,all),0.0,kp); 
      int k2 = find_k2();
      kp = kpts(k2,all) - kp;

      // RR0 = Rinv * R0
      nda::blas::gemm(1.0,symms[is].Rinv,symms[kp_symm(k2)].R,0.0,RR0);

      // Gs = kp * R0
      nda::blas::gemv(1.0,nda::transpose(symms[kp_symm(k2)].R),kp,0.0,Gs); 

      // rotate indexes
      // G -> (-1)^{trev} * ( G * Rinv * R0 - kp * R0 ) 
      Gout() = gv_to_fft();  // list of indexes of G vectors in truncated wfc grid
      utils::transform_k2g(bz.kp_trev(k2),RR0,Gs,mesh,kpts(ik,all),Gout,Xft);

      // rotate psi(k2)
      psi(1,all,all) = ComplexType(0.0);
      for( auto [i,n]: itertools::enumerate(Gout)) {
        if(fft_to_gv(n) >= 0) {
          if(bz.kp_trev(k2)) {
            psi(1,all,i) = psi(2,all,fft_to_gv(n));
          } else {
            psi(1,all,i) = nda::conj(psi(2,all,fft_to_gv(n)));
          }
        }
      }
      if(irred) {
        // assumes ordered eigenvalues, can be fixed otherwise
        if constexpr (Sparse) 
          sp_d.clear();
        int ib=0;
        while(ib < nbnd) {
          int nb=1;
          while( ib+nb<nbnd and std::abs(eigv(0,ik,ib)-eigv(0,ik,ib+nb)) < 1e-4 ) nb++;
          range b_rng(ib,ib+nb);
          if constexpr (Sparse) {
            nda::blas::gemm(ComplexType(1.0),psi(1,b_rng,all),
                                             nda::transpose(psi(0,b_rng,all)),
                            ComplexType(0.0),Temp(range(nb),range(nb)));
            // dump into sp_d
            for( int a=0; a<nb; ++a )
              for( int b=0; b<nb; ++b )
                if( std::abs(Temp(a,b)) > 1e-8 )
                  sp_d[ib+a][ib+b] = Temp(a,b);  
          } else {
            nda::blas::gemm(ComplexType(1.0),psi(1,b_rng,all),
                                             nda::transpose(psi(0,b_rng,all)),
                            ComplexType(0.0),shm_dmat.local()(isk,b_rng,b_rng));
          }
          ib+=nb;
        };
        if constexpr (Sparse)
          dmat_s.emplace_back(math::sparse::to_compact(sp_d));
      } else {
        if constexpr (Sparse) {
          nda::blas::gemm(ComplexType(1.0),psi(1,all,all),nda::transpose(psi(0,all,all)),
                          ComplexType(0.0),Temp);
          dmat_s.emplace_back(math::sparse::to_csr<MEM,int,int>(Temp,1e-8));
        } else {
          nda::blas::gemm(ComplexType(1.0),psi(1,all,all),nda::transpose(psi(0,all,all)),
                          ComplexType(0.0),shm_dmat.local()(isk,all,all));
        }
      }
    
      // check and normalize along rows, since breaking degenerate sets can cause leakage
      if(normalize) {
        for(int r=0; r<nbnd; r++) {
          double e = 0.0;
          if constexpr (Sparse) {
            auto vals = dmat_s.back()[r].values();
            e=std::sqrt(std::abs(nda::blas::dotc(vals,vals)));
            vals() /= e;
          } else {
            e=std::sqrt(std::abs(nda::blas::dotc(shm_dmat.local()(isk,r,all),shm_dmat.local()(isk,r,all))));
            shm_dmat.local()(isk,r,all) /= e;
          }
          if(std::abs(e-1.0) > 1e-3) bad_norm_count += 1;
        }
      }
    }

  } else {

    // no augmentation
    nda::array<ComplexType,3> psi(2,nbnd,wfc_g->size());
    for( long isk=isk0; isk<isk1; ++isk ) { 

      if(isk == 0) { // first term is always the identity
        if constexpr (Sparse) {
          dmat_s.emplace_back(math::sparse::identity<ComplexType,MEM>(nbnd));
        } else {
          auto dsk = shm_dmat.local()(isk,nda::ellipsis{});
          nda::diagonal(dsk) = ComplexType(1.0);
        }
        continue;
      }

      int is = sk[isk].first;
      int ik = sk[isk].second;

      // read orbital if necessary
      if(kold != ik) {
        kold = ik;
        mf.get_orbital_set('w',0,ik,{0,nbnd},psi(0,all,all)); 
      }

      // kRinv = k2 - kp, kp = k2 - kRinv
      nda::blas::gemv(1.0,nda::transpose(symms[is].Rinv),kpts(ik,all),0.0,kp); 
      int k2 = find_k2();
      kp = kpts(k2,all) - kp;

      // RR0 = Rinv * R0
      nda::blas::gemm(1.0,symms[is].Rinv,symms[kp_symm(k2)].R,0.0,RR0);

      // Gs = kp * R0
      nda::blas::gemv(1.0,nda::transpose(symms[kp_symm(k2)].R),kp,0.0,Gs); 

      // rotate indexes
      // G -> (-1)^{trev} * ( G * Rinv * R0 - kp * R0 ) 
      Gout() = gv_to_fft();  // list of indexes of G vectors in truncated wfc grid
      utils::transform_k2g(bz.kp_trev(k2),RR0,Gs,mesh,kpts(ik,all),Gout,Xft);

      // rotate psi(k2)
      psi(1,all,all) = ComplexType(0.0);
      for( auto [i,n]: itertools::enumerate(Gout)) {
        if(fft_to_gv(n) >= 0) {
          if(bz.kp_trev(k2)) {
            psi(1,all,i) = psi(0,all,fft_to_gv(n));
          } else {
            psi(1,all,i) = nda::conj(psi(0,all,fft_to_gv(n)));
          }
        }
      }
      if(irred) {
        // assumes ordered eigenvalues, can be fixed otherwise
        if constexpr (Sparse) 
          sp_d.clear();
        int ib=0;
        while(ib < nbnd) {
          int nb=1;
          while( ib+nb<nbnd and std::abs(eigv(0,ik,ib)-eigv(0,ik,ib+nb)) < 1e-4 ) nb++;
          range b_rng(ib,ib+nb);
          if constexpr (Sparse) {
            nda::blas::gemm(ComplexType(1.0),psi(1,b_rng,all),
                                             nda::transpose(psi(0,b_rng,all)),
                            ComplexType(0.0),Temp(range(nb),range(nb)));
            // dump into sp_d
            for( int a=0; a<nb; ++a )
              for( int b=0; b<nb; ++b )
                if( std::abs(Temp(a,b)) > 1e-8 )
                  sp_d[ib+a][ib+b] = Temp(a,b);  
          } else {
            nda::blas::gemm(ComplexType(1.0),psi(1,b_rng,all),
                                             nda::transpose(psi(0,b_rng,all)),
                            ComplexType(0.0),shm_dmat.local()(isk,b_rng,b_rng));
          }
          ib+=nb;
        };
        if constexpr (Sparse)
          dmat_s.emplace_back(math::sparse::to_compact(sp_d));
      } else {
        if constexpr (Sparse) {
          nda::blas::gemm(ComplexType(1.0),psi(1,all,all),nda::transpose(psi(0,all,all)),
                          ComplexType(0.0),Temp);
          dmat_s.emplace_back(math::sparse::to_csr<MEM,int,int>(Temp,1e-8));
        } else {
          nda::blas::gemm(ComplexType(1.0),psi(1,all,all),nda::transpose(psi(0,all,all)),
                          ComplexType(0.0),shm_dmat.local()(isk,all,all));
        }
      }
    
      // check and normalize along rows, since breaking degenerate sets can cause leakage
      if(normalize) {
        for(int r=0; r<nbnd; r++) {
          double e = 0.0;
          if constexpr (Sparse) {
            auto vals = dmat_s.back()[r].values();
            e=std::sqrt(std::abs(nda::blas::dotc(vals,vals)));
            vals() /= e;
          } else {
            e=std::sqrt(std::abs(nda::blas::dotc(shm_dmat.local()(isk,r,all),shm_dmat.local()(isk,r,all))));
            shm_dmat.local()(isk,r,all) /= e;
          }
          if(std::abs(e-1.0) > 1e-3) bad_norm_count += 1;
        }
      }
    }

  } // needs_aug

  // check # of Bloch orbitals with a incomplete degenerate set
  bad_norm_count = comm.all_reduce_value(bad_norm_count, std::plus<>{});
  if (bad_norm_count > 0)
    app_log(2, "  [WARNING] {} of Bloch orbitals in a reducible representation are not normalized.\n"
               "            This is because high-lying virtual bands in a reducible representation are not \n"
               "            included fully. Make sure degenerate sets of states are fully included. Otherwise, \n"
               "            this error is typically negligible as more and more virtual bands are included.\n",
               bad_norm_count);

  if constexpr (Sparse) {
    std::vector<sp_mat> dmat;
    dmat.reserve(sk.size());
    if( comm.size() == 1 ) {
      for(int i=0; i<dmat_s.size(); i++)
        dmat.emplace_back(dmat_s[i]);
    } else {
      // loop over ranks and bcast sparse arrays in compact format
      nda::array<long,1> sz(sk.size(),0);
      nda::array<char,1> buff(0);
      for(long i=0; i<dmat_s.size(); ++i)
        sz(isk0+i) = dmat_s[i].size_of_serialized_in_bytes(true); 
      comm.all_reduce_in_place_n(sz.data(),sk.size(),std::plus<>{});
      for(int r=0; r<comm.size(); r++) {
        auto [p0, p1] = itertools::chunk_range(0, sk.size(), comm.size(), r);
        long nr = std::accumulate(sz.data()+p0, sz.data()+p1, long(0));
        buff.resize(nr);
        if(comm.rank()==r) {
          for(long i=0, cnt=0; i<dmat_s.size(); ++i) {
            utils::check(isk0==p0, "Partition mismatch");
            dmat_s[i].serialize(buff.data()+cnt,sz(i+isk0),true);
            cnt+=sz(i+isk0);
          }
          dmat_s = std::vector<sp_mat>(0);
        }
        comm.broadcast_n(buff.data(),nr,r);
        for(long p=p0, cnt=0; p<p1; ++p) {
          dmat.emplace_back(sp_mat());
          dmat.back().deserialize(buff(range(cnt,cnt+sz(p))));
          cnt+=sz(p);
        }
      }
    }
    return std::make_tuple(std::move(sk_to_n),std::move(dmat));
  } else {
    node_comm.barrier();
    shm_dmat.all_reduce();
    node_comm.barrier();
    return std::make_tuple(std::move(sk_to_n),std::move(shm_dmat));
  }
}

} // namespace utils

