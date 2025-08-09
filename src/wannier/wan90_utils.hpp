#ifndef COQUI_WAN90_UTILS_HPP
#define COQUI_WAN90_UTILS_HPP

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>

#include "configuration.hpp"

#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "nda/linalg.hpp"
#include "h5/h5.hpp"

#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/mpi_context.h"
#include "utilities/check.hpp"
#include "utilities/kpoint_utils.hpp"
#include "utilities/basis_set_utilities.hpp"
#include "utilities/integration.hpp"
#include "utilities/parser.h"
#include "utilities/stl_utils.hpp"
#include "utilities/fortran_utilities.h"
#include "numerics/nda_functions.hpp"
#include "numerics/shared_array/nda.hpp"
#include "mean_field/MF.hpp"
#include "utilities/harmonics.h"
#if defined(ENABLE_SPHERICART)
#include "sphericart.hpp"
#endif

namespace wannier::detail {

auto wann90_lm_to_Ylm()
{
  int l=3, l2=(l+1)*(l+1);
  nda::array<int,1> lm(l2,0);
  // l=0      //  Ylm  Wann90
  lm(0) = 0;  //  1     1
  // l=1 
  lm(1) = 2;  //  y     z
  lm(2) = 3;  //  z     x
  lm(3) = 1;  //  x     y
  // l=2
  lm(4) = 6;  //  xy     z2
  lm(5) = 7;  //  yz     xz
  lm(6) = 5;  //  z2     yz
  lm(7) = 8;  //  xz     x2-y2
  lm(8) = 4;  //  x2-y2  xy
  // l=3
  lm(9) = 12;   //  y(3x2-y2) z3
  lm(10) = 13;  //  xyz       xz2
  lm(11) = 11;  //  yz2       yz2
  lm(12) = 14;   //  z3        z(x2-y2) 
  lm(13) = 10;  //  xz2       xyz 
  lm(14) = 15;  //  z(x2-y2)  x(x2-3y2)
  lm(15) = 9;  //  x(x2-3y2) y(3x2-y2) 
  return lm;
}

template<bool single>
std::string get_token(std::string const& line, std::string const& key)
{
  std::istringstream iss(line);
  std::string token;
  // make sure first element is key or key= or key:
  if(iss >> token) {
    io::tolower(token);
    utils::check((token == key) or (key+"=" == token) or (key+":" == token), 
                 "get_token: Expect key:{}, found:{} in line:{}",key,token,line);
    if(token == key) {
      // remove = or : from beginning if found
      // skip blank spaces
      while(iss.peek() == ' ') iss.get();
      char d = iss.peek();
      if( (d=='=') or (d==':') ) iss.get(); 
    } 
  } else 
    utils::check(false,"Error in parsing line in wannier::get_token: {}",line); 
  
  if constexpr (single) {
    if(iss >> token) {
      return token;
    } else
      utils::check(false,"Error in parsing line in wannier::get_token: {}",line);  
  } else {
    // return everything
    if(std::getline(iss,token)) {
      return token;
    } else
      utils::check(false,"Error in parsing line in wannier::get_token: {}",line);  
  }
  return token;
}

template<bool must_exist, typename T>
T read_key(std::string const& file_data, std::string const& key, T def = T{}) {
  T val = def;
  std::istringstream iss(file_data);
  std::string line;
  std::string key_lower = io::tolower_copy(key);

  while (std::getline(iss, line)) {
    // ignore anythong past # or !
    if( (line.find("#") != std::string::npos) or (line.find("!") != std::string::npos) ) {
      // is this correct?
      auto pos = std::min(line.find("#"),line.find("!"));
      line = line.substr(0l,long(pos)); 
    } 
    if(io::tolower_copy(line).find(key_lower) != std::string::npos) {
      auto token = get_token<true>(line,key); 
      // right now only bool, integer, double/float
      if constexpr (std::is_same_v<T,bool>) {
        if(token == "T" or token == "true" or token == ".true.") return true; 
        else if(token == "F" or token == "false" or token == ".false.") return false; 
        else
          utils::check(false,"Invalid boolean: {} = {}",key,token);
      } else if constexpr (std::is_same_v<T,std::string>) {
        return token; 
      } else { 
        std::istringstream iss2(token);
        if(iss2 >> val) {
          return val;
        } else
          utils::check(false,"Error parsing line: {}",line);
      }
    }
  }
  if constexpr (must_exist) { 
    utils::check(false, "read_key: Error reading required input key: {}",key);
  }
  return val;
};

template<bool must_exist>
std::vector<int> read_range(std::string const& file_data, std::string const& key, std::string const delim = " ") {
  std::vector<int> val;
  std::istringstream iss(file_data);
  std::string line;
  std::string key_lower = io::tolower_copy(key);

  while (std::getline(iss, line)) {
    // ignore anythong past # or !
    if( (line.find("#") != std::string::npos) or (line.find("!") != std::string::npos) ) {
      // is this correct?
      auto pos = std::min(line.find("#"),line.find("!"));
      line = line.substr(0l,long(pos));
    }
    if(io::tolower_copy(line).find(key_lower) != std::string::npos) {
      auto str = get_token<false>(line,key);
      // interpret token  
      auto tokens = utils::split(str,delim); 
      for( auto& v : tokens ) {
        // no negative numbers allowed!!!
        if(v.find('-') != std::string::npos) {
          auto p = v.find('-');
          int a=0,b=0;
          try {
            a = std::stoi(v.substr(0,p));
            b = std::stoi(v.substr(p+1,v.length()-p-1));
          } catch (...) {
            utils::check(false,"Problems parsing range: {}",line);
          }
          utils::check(a>0 and b>=a, "Error in read_range: Invalid range: {}",line);
          for(int i=a; i<=b; ++i ) val.emplace_back(i); 
        } else {
          try {
            val.emplace_back(std::stoi(v));
          } catch (...) {
            utils::check(false,"Problems parsing range: {}",line);
          }
        } 
      }
      return val;
    }
  }
  if constexpr (must_exist) {
    utils::check(false, "read_key: Error reading required input key: {}",key);
  }
  return val;
};

template<bool must_exist, typename F_t>
auto read_block(std::string const& file_data, std::string const& open_key,
                     std::string const& close_key, F_t && F) {
  std::string open_key_lower = io::tolower_copy(open_key);
  std::string close_key_lower = io::tolower_copy(close_key);
  std::istringstream iss(file_data);
  std::string line, line2;
  while (std::getline(iss, line)) {
    if( (line.find("#") != std::string::npos) or (line.find("!") != std::string::npos) ) {
      // is this correct?
      auto pos = std::min(line.find("#"),line.find("!"));
      line = line.substr(0l,long(pos));
    }
    if(io::tolower_copy(line).find(open_key_lower) != std::string::npos) {
      F(iss); // process string with lambda function
      if(close_key.length() > 0) {
        iss.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::getline(iss, line2);
        utils::check(io::tolower_copy(line2).find(close_key_lower) != std::string::npos,
                     "read_block: Error reading input block, missing close key: {}",close_key);
      }
      return true;
    }
  }
  if constexpr (must_exist)
    utils::check(false, "read_block: Error reading input block: {}",open_key);
  return false; 
};

void write_mmn_file(std::string prefix, nda::MemoryArrayOfRank<3> auto&& nnkpts, 
                    nda::MemoryArrayOfRank<4> auto&& M, bool transposed)
{
  long nkpts = M.extent(0);
  long nnb   = M.extent(1);
  long nband = M.extent(2);
  std::ofstream out(prefix+".mmn");
  time_t timestamp;
  time(&timestamp);
  out<<" Coqui generated mmn file: " <<ctime(&timestamp);
  out<<std::setw(12) <<nband <<" "
     <<std::setw(12) <<nkpts <<" "
     <<std::setw(12) <<nnb <<"\n";
  for(long ik=0l; ik<nkpts; ++ik) {
    for(long in=0l; in<nnb; ++in) {
      auto M_ = M(ik,in,::nda::ellipsis{});
      out<<std::fixed <<std::setw(10) <<ik+1 <<" "
                    <<std::setw(10) <<nnkpts(ik,in,0) <<" "
                    <<std::setw(10) <<nnkpts(ik,in,1) <<" "
                    <<std::setw(10) <<nnkpts(ik,in,2) <<" "
                    <<std::setw(10) <<nnkpts(ik,in,3) <<"\n";
      // column-major ordering on output
      if(transposed) {
        for(long m=0; m<nband; ++m)
          for(long n=0; n<nband; ++n)
            out<<std::fixed <<std::setw(18) <<std::setprecision(12) <<std::real(M_(m,n)) <<" "
                            <<std::setw(18) <<std::setprecision(12) <<std::imag(M_(m,n)) <<"\n";
      } else {
        for(long n=0; n<nband; ++n)
          for(long m=0; m<nband; ++m)
            out<<std::fixed <<std::setw(18) <<std::setprecision(12) <<std::real(M_(m,n)) <<" "
                            <<std::setw(18) <<std::setprecision(12) <<std::imag(M_(m,n)) <<"\n";
      }
    }
  }
  out.close();
}

void write_amn_file(std::string prefix, nda::MemoryArrayOfRank<3> auto&& A, bool transposed)
{
  long nkpts = A.extent(0);
  long nproj = (transposed ? A.extent(1) : A.extent(2));
  long nband = (transposed ? A.extent(2) : A.extent(1));
  std::ofstream out(prefix+".amn");
  time_t timestamp;
  time(&timestamp);
  out<<" Coqui generated amn file: " <<ctime(&timestamp);
  out<<std::setw(12) <<nband <<" "
     <<std::setw(12) <<nkpts <<" "
     <<std::setw(12) <<nproj <<"\n";
  for(long ik=0; ik<nkpts; ++ik) {
    if(transposed) {
      for(long m=0; m<nproj; ++m)
        for(long n=0; n<nband; ++n)
          out<<std::fixed <<std::setw(10) <<n+1 <<" "
                    <<std::setw(10) <<m+1 <<" "
                    <<std::setw(10) <<ik+1 <<" "
                    <<std::setw(18) <<std::setprecision(12) <<std::real(A(ik,m,n)) <<" "
                    <<std::setw(18) <<std::setprecision(12) <<std::imag(A(ik,m,n)) <<"\n";
    } else {
      for(long m=0; m<nproj; ++m)
        for(long n=0; n<nband; ++n)
          out<<std::fixed <<std::setw(10) <<n+1 <<" "
                    <<std::setw(10) <<m+1 <<" "
                    <<std::setw(10) <<ik+1 <<" "
                    <<std::setw(18) <<std::setprecision(12) <<std::real(A(ik,n,m)) <<" "
                    <<std::setw(18) <<std::setprecision(12) <<std::imag(A(ik,n,m)) <<"\n";
    }
  }
  out.close();
}

auto read_amn_file(std::string fname, bool transposed)
{
  std::ifstream file(fname);
  std::string line;
  std::getline(file, line);
  int nkpts,nproj,nband,k_,n_,m_;
  file>>nband >>nkpts >>nproj;
  std::array<long,3> shape = {nkpts,nband,nproj};
  if(transposed) shape = {nkpts,nproj,nband};
  nda::array<ComplexType,3> A(shape);
  file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  double r_,c_;
  for(long ik=0l; ik<nkpts; ++ik) { 
    for(long m=0; m<nproj; ++m) {
      for(long n=0; n<nband; ++n) { 
        file>>n_ >>m_ >>k_ >>r_ >>c_;
        if(transposed) {
          A(k_-1,m_-1,n_-1) = ComplexType(r_,c_);
        } else {
          A(k_-1,n_-1,m_-1) = ComplexType(r_,c_);
        }
      }
    }
  }
  file.close();
  return A;
}

void write_modest_h5(mf::MF &mf, ptree pt, nda::array<int,1> const& band_list, 
                     nda::MemoryArrayOfRank<3> auto const& eigv, nda::Array auto && Pkwa) {

  auto all = nda::range::all;
  app_log(2, "\n");
  app_log(2," Writing ModEST h5 file.");

  auto [nspin,nkpts,nwann,nband] = Pkwa.shape();

  auto prefix = io::get_value<std::string>(pt,"prefix");
  auto dset_name = io::get_value_with_default<std::string>(pt,"dset_name","dft_input");
  auto h5_fname = io::get_value_with_default<std::string>(pt,"h5_filename",prefix+".modest.h5");
 
  // read shell information: 'atom', 'sort', 'l', 'dim', 'SO', 'irep'
  nda::array<long,2> shells = { {0,0,0,long(nwann),0,0} };
  auto shell_atoms = io::get_array_with_default<long>(pt,"shells.atoms",std::vector<long>{});
  auto shell_sort = io::get_array_with_default<long>(pt,"shells.sort",std::vector<long>{});
  auto shell_l = io::get_array_with_default<long>(pt,"shells.l",std::vector<long>{});
  auto shell_dim = io::get_array_with_default<long>(pt,"shells.dim",std::vector<long>{});
  auto shell_SO = io::get_array_with_default<long>(pt,"shells.SO",std::vector<long>{});
  auto shell_irrep = io::get_array_with_default<long>(pt,"shells.irrep",std::vector<long>{});
  // if any of them is found, they all need to have the same size
  long nshell = std::max({shell_atoms.size(),shell_sort.size(),shell_l.size(),
                         shell_dim.size(),shell_SO.size(),shell_irrep.size()}); 
  auto check_size = [&](std::vector<long> &A, std::string const& str) {
        utils::check(A.size()==0 or A.size()==nshell, 
          "write_modest_h5: Incompatible size in shell definition of {}: nshell:{}, size:{}",
          str,nshell,A.size());  
      };
  check_size(shell_atoms,"shells.atoms"); 
  check_size(shell_sort,"shells.sort"); 
  check_size(shell_l,"shells.l"); 
  check_size(shell_SO,"shells.SO"); 
  check_size(shell_irrep,"shells.irrep"); 
  // shells.dim is the exception, if nshell>1, then shells.dim must be defined
  if(nshell == 1) {
    check_size(shell_dim,"shells.irrep"); 
    if(shell_dim.size() == 0)
      shell_dim.emplace_back(long(nwann));
  } else {
    utils::check(shell_dim.size() == nshell, 
                 "write_modest_h5: shells.dim is required if any shell information is provided."); 
  }
  long tot_norbs = std::accumulate(shell_dim.begin(),shell_dim.end(),0);
  utils::check(tot_norbs <= nwann,
               "write_modest_h5: Too many orbitals specified in shells: sum(shells.dim):{}, nwann:{}",
               tot_norbs,nwann); 
  if(nshell > 0) {
    shells = nda::array<long,2>{nshell,6};
    shells() = 0;
    for(int i=0; i<nshell; ++i) {
      if(shell_atoms.size() > 0) shells(i,0) = shell_atoms[i];
      if(shell_sort.size() > 0) shells(i,1) = shell_sort[i];
      if(shell_l.size() > 0) shells(i,2) = shell_l[i];
      if(shell_dim.size() > 0) shells(i,3) = shell_dim[i];
      if(shell_SO.size() > 0) shells(i,4) = shell_SO[i];
      if(shell_irrep.size() > 0) shells(i,5) = shell_irrep[i];
    }
    app_log(2,"  Shell information found.");
    app_log(2," Number of shells: {}",shells.extent(0));
    app_log(2," Number of orbitals (summed over all shells): {}",tot_norbs);
    for(int i=0; i<shells.extent(0); ++i)
      app_log(2,"  shell:{} - atom:{} sort:{} l:{} dim:{} SO:{} irrep:{}",
              i,shells(i,0),shells(i,1),shells(i,2),shells(i,3),shells(i,4),shells(i,5));
  } else {
    app_log(2, "  Shell information not found on input. Using a single shell with all wannier orbitals");
    tot_norbs = nwann;
    nshell = 1;
    shells = nda::array<long,2>{nshell,6};
    shells() = 0;
    shells(0,3) = nwann;
    shell_dim.emplace_back(nwann);
  }

  if(tot_norbs < nwann) {
    app_log(2, " Number of specified orbitals:{} is smaller than the number of wannier orbitals:{}",
            tot_norbs,nwann);
    app_log(2, " Choosing the first {} wannier orbitals",tot_norbs); 
  }

  h5::file fh5(h5_fname,'a');
  h5::group grp(fh5);

  h5::group dgrp = (grp.has_subgroup(dset_name) ?
                    grp.open_group(dset_name)    :
                    grp.create_group(dset_name));

  // SO/SP
  h5::h5_write(dgrp, "SP", nspin-1);
  h5::h5_write(dgrp, "SO", long(mf.npol()-1));

  // bz_weight: should sum to 1? or to nspin?
  {
    nda::array<double,1> wgt(nkpts,1.0/double(nkpts));
    nda::h5_write(dgrp, "bz_weights", wgt);
  }

  // n_orbitals, need to partition over impurities
  {
    // in principle this can be customized, right now assume most general case
    nda::array<long,2> norbs(nkpts,nspin);
    norbs() = nband;
    nda::h5_write(dgrp, "n_orbitals", norbs);
  }

  // dft_code: QE?, w90, or coqui?
  h5::h5_write(dgrp, "dft_code", "w90");

  // corr_shell
  {
    h5::group cgrp1 = dgrp.create_group("corr_shells"); 
    h5::group cgrp2 = dgrp.create_group("shells"); 
    for(int i=0; i<nshell; ++i) {
      {
        h5::group sgrp = cgrp1.create_group(std::to_string(i)); 
        h5::h5_write(sgrp,"atom",shells(i,0));
        h5::h5_write(sgrp,"sort",shells(i,1));
        h5::h5_write(sgrp,"l",shells(i,2));
        h5::h5_write(sgrp,"dim",shells(i,3));
        h5::h5_write(sgrp,"SO",shells(i,4));
        h5::h5_write(sgrp,"irrep",shells(i,5));
      }
      {  // mimicking dfttools wannier90 converter, no real reason to do this in principle
        h5::group sgrp = cgrp2.create_group(std::to_string(i));
        h5::h5_write(sgrp,"atom",shells(i,0));
        h5::h5_write(sgrp,"sort",shells(i,1));
        h5::h5_write(sgrp,"l",shells(i,2));
        h5::h5_write(sgrp,"dim",shells(i,3));
      }
    }
  }

  // proj_mat: need to partition over impurities
  if(nshell == 1 and tot_norbs == nwann and nspin == 1) {
    auto proj_mat = nda::reshape(Pkwa,std::array<long,5>{nkpts,nspin,1,nwann,nband});
    proj_mat() = nda::conj(proj_mat());
    nda::h5_write(dgrp,"proj_mat",proj_mat,false);
    proj_mat() = nda::conj(proj_mat());
  } else {
    // assuming the first orbitals are the correlated ones, in principle this is not necessary
    long nwmax = *std::max_element(shell_dim.begin(),shell_dim.end()); 
    auto proj_mat = nda::array<ComplexType,5>(nkpts,nspin,nshell,nwmax,nband); 
    proj_mat() = ComplexType(0.0);
    for(int i=0, w0=0, w1=0; i<nshell; i++) {
      w1 += shells(i,3);
      for(int is=0; is<nspin; is++) 
        proj_mat(all,is,i,nda::range(shells(i,3)),all) = nda::conj(Pkwa(is,all,nda::range(w0,w1),all)); 
      w0 += shells(i,3);
    } 
    nda::h5_write(dgrp,"proj_mat",proj_mat,false);
  }

  // misc
  {
    // MAM: default value of density_required makes no sense to me
    auto occ = mf.occ();
    double rho_sum = 0.0;  
    for( auto [i,n]: itertools::enumerate(band_list) )
      rho_sum += nda::sum(occ(all,all,n))/double(nkpts);
    if(mf.nspin() == 1 and mf.npol() == 1) rho_sum*=2.0;
    auto charge_below = io::get_value_with_default<long>(pt,"charge_below",0l);
    auto density_required = io::get_value_with_default<double>(pt,"density_required",rho_sum);
    h5::h5_write(dgrp,"charge_below",charge_below);
    h5::h5_write(dgrp,"energy_unit",1.0);
    h5::h5_write(dgrp,"symm_op",0l);
    h5::h5_write(dgrp,"density_required",density_required);
  }

  // hopping: assuming bloch_basis = T
  {
    auto bloch_basis = io::get_value_with_default<bool>(pt,"bloch_basis",true);
    utils::check(bloch_basis, "write_modest_h5: bloch_basis = false not implemented yet.");
    nda::array<ComplexType,4> hopping(nkpts,nspin,nband,nband);
    hopping() = ComplexType(0.0);
    ComplexType ef(mf.efermi()/3.674932540e-2);
    for( int ik=0; ik<nkpts; ik++ )
      for( int is=0; is<nspin; is++ )
        for( int ib=0; ib<nband; ib++ )
          hopping(ik,is,ib,ib) = (eigv(is,ik,ib)-ef);
    nda::h5_write(dgrp,"hopping",hopping,false);
  } 

  // rot_mat
  {
    h5::h5_write(dgrp,"use_rotations",0l);
    h5::group cgrp = dgrp.create_group("rot_mat"); 
    for(int i=0; i<nshell; i++) {
      nda::array<ComplexType,2> rot_mat(shell_dim[i],shell_dim[i]);
      rot_mat() = ComplexType(0.0); 
      nda::diagonal(rot_mat()) = ComplexType(1.0); 
      nda::h5_write(cgrp,std::to_string(i),rot_mat);
    }
  }
  {
    h5::group cgrp = dgrp.create_group("rot_mat_time_inv");
    for(int i=0; i<nshell; i++) {
      h5::h5_write(cgrp,std::to_string(i),0l);
    }
  }

}

} // wannier::detail 

#endif
