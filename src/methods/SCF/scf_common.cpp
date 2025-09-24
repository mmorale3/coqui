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


#include "scf_common.hpp"
#include "dca_dyson.h"
#include "hamiltonian/one_body_hamiltonian.hpp"
#include "mean_field/MF.hpp"
#include "utilities/mpi_context.h"
#include "methods/tools/chkpt_utils.h"
#include "simple_dyson.h"

namespace methods {
double compute_Nelec(double mu, const nda::array<ComplexType, 4> &spectra,
                     const mf::MF &mf, const imag_axes_ft::IAFT &FT) {
  auto [_nw, _ns, nkpts, _nbnd] = spectra.shape();
  nda::array<ComplexType, 2> Xw(_nw, _ns);
  nda::array<ComplexType, 2> Xt(FT.nt_f(), _ns);
  nda::array<ComplexType, 1> nelecs(_ns);
  auto k_weight = mf.k_weight();
  double scl = (_ns == 1 and mf.npol() == 1 ? 2.0 : 1.0); 

  for (size_t n = 0; n < _nw; ++n) {
    long wn = FT.wn_mesh()(n);
    ComplexType omega_mu = FT.omega(wn) + mu;
    for (size_t is = 0; is < _ns; ++is) {
      for (size_t ik = 0; ik < nkpts; ++ik) {
        for (size_t ib = 0; ib < _nbnd; ++ib) {
          Xw(n, is) += k_weight(ik) / (omega_mu - spectra(n, is, ik, ib));
        }
      }
    }
  }

  FT.w_to_tau(Xw, Xt, imag_axes_ft::fermi);
  FT.tau_to_beta(Xt, nelecs);

  ComplexType nelec = scl*std::accumulate(nelecs.begin(),nelecs.end(),ComplexType(0.0));
  nelec *= -1.0;
  if (nelec.imag() / mf.nelec() >= 1e-10) {
    app_log(1, "[WARNING] nelec.imag()/nelec_target = {}",
            nelec.imag() / mf.nelec());
  }

  return nelec.real();
}

template<typename X_t, nda::ArrayOfRank<1> Array1D>
auto eval_hf_energy(const X_t &sDm_skij, const X_t &sF_skij, const X_t &sH0_skij,
                    Array1D &k_weight, bool F_has_H0)
  -> std::tuple<double, double> {
  auto [ns, nkpts, nbnd, nbnd2] = sDm_skij.shape();
  // HF energy = Tr[Dm*H0] + 0.5*Tr[Dm*F] + e_nuc
  nda::matrix<ComplexType> buffer(nbnd, nbnd);
  ComplexType e_1e(0.0, 0.0);
  ComplexType e_hf(0.0, 0.0);
  for (size_t is = 0; is < ns; ++is) {
    for (size_t ik = 0; ik < nkpts; ++ik) {
      nda::matrix_const_view<ComplexType> Dm_ij =
          sDm_skij.local()(is, ik, nda::ellipsis{});
      nda::matrix_const_view<ComplexType> F_ij =
          sF_skij.local()(is, ik, nda::ellipsis{});
      nda::matrix_const_view<ComplexType> H0_ij =
          sH0_skij.local()(is, ik, nda::ellipsis{});

      buffer = Dm_ij * H0_ij;
      auto diag_H0 = nda::diagonal(buffer);
      e_1e += k_weight(ik) * nda::sum(diag_H0);

      buffer = (F_has_H0)? 0.5 * Dm_ij * (F_ij - H0_ij) : 0.5 * Dm_ij * F_ij;
      auto diag_F = nda::diagonal(buffer);
      e_hf += k_weight(ik) * nda::sum(diag_F);
    }
  }
  // MAM: need to know npol here, scale only when npol==1 and ns==1
  RealType spin_factor = (ns == 2) ? 1.0 : 2.0;
  e_1e *= spin_factor;
  e_hf *= spin_factor;
  // TODO CNY: _MF->e_nuc() is missing
  if (e_1e.imag() / e_1e.real() >= 1e-10) {
    app_log(1, "[WARNING] e_1e.imag()/e_1e.real() = {}, e_1e.imag() = {}, e_1e.real() = {}",
            e_1e.imag()/e_1e.real(), e_1e.imag(), e_1e.real());
  }
  if (e_hf.imag() / e_hf.real() >= 1e-10) {
    app_log(1, "[WARNING] e_hf.imag()/e_hf.real() = {}, e_hf.imag() = {}, e_hf.real() = {}",
            e_hf.imag()/e_hf.real(), e_hf.imag(), e_hf.real());
  }
  return std::make_tuple(e_1e.real(), e_hf.real());
}

template<typename comm_t, typename X_t, nda::ArrayOfRank<1> Array1D>
double eval_corr_energy(comm_t& comm, const imag_axes_ft::IAFT &FT,
                        const X_t & G_shm, const X_t & Sigma_shm,
                        Array1D &k_weight) {
  decltype(nda::range::all) all;
  int nw = FT.nw_f();
  auto [nts, ns, nkpts, nbnd, nbnd2] = G_shm.shape();
  nda::array<ComplexType, 2> SigmaG_ws(nw, ns);
  nda::array<ComplexType, 4> Sigma_tski(nts, ns, nkpts, nbnd);
  nda::array<ComplexType, 4> G_tski(nts, ns, nkpts, nbnd);
  nda::array<ComplexType, 4> Sigma_wski(nw, ns, nkpts, nbnd);
  nda::array<ComplexType, 4> G_wski(nw, ns, nkpts, nbnd);
  auto SigmaG_ws_1D =
      nda::reshape(SigmaG_ws, std::array<long, 1>{nw * ns});
  auto Sigma_w_3D = nda::reshape(
      Sigma_wski, std::array<long, 3>{nw * ns, nkpts, nbnd});
  auto G_w_3D = nda::reshape(
      G_wski, std::array<long, 3>{nw * ns, nkpts, nbnd});

  int size = comm.size();
  int rank = comm.rank();
  comm.barrier();
  for (size_t i = rank; i < nbnd; i += size) {
    Sigma_tski = Sigma_shm.local()(all, all, all, i, all);
    G_tski = G_shm.local()(all, all, all, all, i);
    FT.tau_to_w(Sigma_tski, Sigma_wski, imag_axes_ft::fermi);
    FT.tau_to_w(G_tski, G_wski, imag_axes_ft::fermi);
    for (size_t ws = 0; ws < nw * ns; ++ws) {
      for (size_t ik = 0; ik < nkpts; ++ik ) {
        SigmaG_ws_1D(ws) += k_weight(ik) * nda::blas::dot(Sigma_w_3D(ws, ik, all), G_w_3D(ws, ik, all));
      }
    }
  }
  comm.all_reduce_in_place_n(SigmaG_ws.data(), SigmaG_ws.size(),
                             std::plus<>{});

  nda::array<ComplexType, 2> SigmaG_ts(nts, ns);
  nda::array<ComplexType, 1> SigmaG_beta_s(ns);
  FT.w_to_tau(SigmaG_ws, SigmaG_ts, imag_axes_ft::fermi);
  FT.tau_to_beta(SigmaG_ts, SigmaG_beta_s);

  // MAM: need to know npol here, scale only when npol==1 and ns==1
  RealType spin_factor = (ns == 2) ? 1.0 : 2.0;
  ComplexType e_corr = (-0.5 * spin_factor) * nda::sum(SigmaG_beta_s);
  if (e_corr.imag() / e_corr.real() >= 1e-8) {
    app_log(1, "[WARNING] e_corr.imag()/e_corr.real() = {}, e_corr.imag() = {}, e_corr.real() = {}",
            e_corr.imag()/e_corr.real(), e_corr.imag(), e_corr.real());
  }
  return e_corr.real();
}

template<typename dyson_type, typename X_t, typename Xt_t>
void update_G(dyson_type &dyson, const mf::MF &mf, const imag_axes_ft::IAFT &FT, X_t & Dm, Xt_t &G,
              const X_t & F, const Xt_t &Sigma, double &mu, bool const_mu) {
  app_log(2, "* Solving Green's function:");
  if(!const_mu)
    mu = update_mu(mu, dyson, mf, FT, F, G, Sigma);
  dyson.solve_dyson(Dm, G, F, Sigma, mu);
}

template<typename dyson_type, typename X_t, typename Xt_t>
double update_mu(double old_mu, dyson_type& dyson, const mf::MF &mf, const imag_axes_ft::IAFT &FT,
                 const X_t&F, const Xt_t&G, const Xt_t&Sigma) {
  double nel, mu1, mu2, mu_mid;
  double mu = old_mu;
  double nel_target = mf.nelec();
  double delta = 0.2;
  nda::array<ComplexType, 4> FpSigma_spectra(FT.nw_f(), mf.nspin(), mf.nkpts_ibz(), mf.nbnd());
  dyson.compute_eigenspectra(mu, F, G, Sigma, FpSigma_spectra);
  nel = compute_Nelec(old_mu, FpSigma_spectra, mf, FT);
  app_log(2, "Initial chemical potential (mu) = {}, nelec = {}", old_mu, nel);

  if (std::abs(nel - nel_target) < dyson.mu_tol()) {
    app_log(1, "Chemical potential found (mu) = {} a.u.", mu);
    app_log(1, "Number of electrons per unit cell = {}", nel);
    return mu;
  }

  if (nel >= nel_target) {
    mu2 = old_mu;
    mu1 = old_mu - delta;
    double nel1 = compute_Nelec(mu1, FpSigma_spectra, mf, FT);
    while (nel1 > nel_target) {
      mu1 -= delta;
      nel1 = compute_Nelec(mu1, FpSigma_spectra, mf, FT);
    }
    app_log(4, "mu = {}, nelec = {}", mu1, nel1);
  } else {
    mu1 = old_mu;
    mu2 = old_mu + delta;
    double nel2 = compute_Nelec(mu2, FpSigma_spectra, mf, FT);
    while (nel2 < nel_target) {
      mu2 += delta;
      nel2 = compute_Nelec(mu2, FpSigma_spectra, mf, FT);
    }
    app_log(4, "mu = {}, nelec = {}", mu2, nel2);
  }
  mu_mid = (mu1 + mu2) * 0.5;
  nel = compute_Nelec(mu_mid, FpSigma_spectra, mf, FT);
  app_log(4, "mu = {}, nelec = {}", mu_mid, nel);

  while (std::abs(nel - nel_target) >= dyson.mu_tol()) {
    if (nel >= nel_target) {
      mu2 = mu_mid;
    } else {
      mu1 = mu_mid;
    }
    mu_mid = (mu1 + mu2) * 0.5;
    nel = compute_Nelec(mu_mid, FpSigma_spectra, mf, FT);
    app_log(4, "mu = {}, nelec = {}", mu_mid, nel);
  }
  mu = mu_mid;
  app_log(1, "Chemical potential found (mu) = {} a.u.", mu);
  app_log(1, "Number of electrons per unit cell = {}", nel);
  return mu;
}

template<typename comm_t, typename X_t, typename Xt_t>
auto init_solver(comm_t &context, iter_scf::iter_scf_t& iter_solver,
                 long it, std::string output,
                 X_t &sF_skij, Xt_t &sSigma_tskij, const imag_axes_ft::IAFT *FT){
  if(iter_solver.iter_alg() == iter_scf::DIIS and context.comm.root()) { // Initialize the iterative solver
    std::string filename = output + ".mbpt.h5";
    h5::file file(filename, 'r');
    h5::group grp(file);
    utils::check(grp.has_subgroup("scf"), "Simulation HDF5 file does not have an scf group");
    auto scf_grp = grp.open_group("scf");
    auto sys_grp = grp.open_group("system");
    nda::array<ComplexType, 4> H0 = sF_skij.local();
    nda::array<ComplexType, 4> S = sF_skij.local();
    nda::h5_read(sys_grp, "H0_skij", H0);
    nda::h5_read(sys_grp, "S_skij", S);
    double mu = 0;
    if (scf_grp.has_subgroup("iter" + std::to_string(it-1))) {
      auto mf_grp = scf_grp.open_group("iter" + std::to_string(it-1));
      h5::h5_read(mf_grp, "mu", mu);
    }
    iter_solver.initialize(sF_skij.local(), sSigma_tskij.local(), mu, S, H0, FT, output);
  }
  context.comm.barrier();
}

template<typename MPI_Context_t, typename X_t, typename Xt_t>
auto damping_impl(MPI_Context_t &context, iter_scf::iter_scf_t& iter_solver,
                  long it, std::string h5_prefix,
                  X_t &sF_skij, Xt_t &sSigma_tskij,
                  std::array<std::string,3> datasets)
  -> std::tuple<double, double> {
  double conv_F = 0;
  double conv_Sigma = 0;
  if (it == 1) {
    utils::check(false, "damping_impl: it = 1 is not allowed.");
  } else {
    iter_solver.metadata_log();
    if (context.node_comm.root()) {
      std::string filename = h5_prefix + ".mbpt.h5";
      h5::file file(filename, 'r');
      h5::group grp(file);

      std::string grp_name = datasets[0]+"/iter"+std::to_string(it-1);
      utils::check(grp.has_subgroup(grp_name), "damping_impl: {} does not exist in {}.",
                   grp_name, filename);
      auto scf_grp = grp.open_group(datasets[0]);
      conv_F = iter_solver.solve(sF_skij.local(), datasets[1], scf_grp, it);
      conv_Sigma = iter_solver.solve(sSigma_tskij.local(), datasets[2], scf_grp, it);
    }
    context.node_comm.broadcast_n(&conv_F, 1, 0);
    context.node_comm.broadcast_n(&conv_Sigma, 1, 0);
  }
  context.comm.barrier();
  return std::make_tuple(conv_F, conv_Sigma);
}

template<typename MPI_Context_t, typename X_t, typename Xt_t>
auto diis_impl(MPI_Context_t &context, iter_scf::iter_scf_t& iter_solver,
               long it, std::string h5_prefix, X_t &sF_skij, Xt_t &sSigma_tskij,
               const imag_axes_ft::IAFT *FT, bool restart,
               std::array<std::string,3> datasets)
  -> std::tuple<double, double> {
  double conv_F = 0;
  double conv_Sigma = 0;
  if (it == 1) {
    utils::check(false, "diis_impl: it = 1 is not allowed.");
  } else {
    if (restart) { // restart DIIS
      init_solver(context, iter_solver, it, h5_prefix, sF_skij, sSigma_tskij, FT);
    }
    iter_solver.metadata_log();
    int internode_proc_holding_extrap = 0;
    if (context.comm.root()) { // A global communicator here is needed for DIIS
      std::string filename = h5_prefix + ".mbpt.h5";
      h5::file file(filename, 'r');
      h5::group grp(file);

      std::string grp_name = datasets[0]+"/iter"+std::to_string(it-1);
      utils::check(grp.has_subgroup(grp_name), "diis_impl: {} does not exist in {}.",
                   grp_name, filename);

      auto scf_grp = grp.open_group(datasets[0]);
      auto [conv_F_,conv_Sigma_] = iter_solver.solve(sF_skij.local(), datasets[1],
                                                     sSigma_tskij.local(), datasets[2], scf_grp, it);
      conv_F = conv_F_;
      conv_Sigma = conv_Sigma_;
      internode_proc_holding_extrap = context.internode_comm.rank();
    }
    context.comm.broadcast_n(&conv_F, 1, 0);
    context.comm.broadcast_n(&conv_Sigma, 1, 0);
    // internode_proc_holding_extrap should be 0 everywhere, but if not,
    // the broadcast below ensures that all procs get it
    context.comm.broadcast_n(&internode_proc_holding_extrap, 1, 0);
    // Send extrapolated F and Sigma to all nodes
    sF_skij.broadcast_to_nodes(internode_proc_holding_extrap);
    sSigma_tskij.broadcast_to_nodes(internode_proc_holding_extrap);
  }
  context.comm.barrier();
  return std::make_tuple(conv_F, conv_Sigma);
}

template<typename comm_t, typename X_t, typename Xt_t>
auto solve_iterative(utils::mpi_context_t<comm_t> &context, iter_scf::iter_scf_t& iter_solver,
                     long it, std::string h5_prefix,
                     X_t &sF_skij, Xt_t &sSigma_tskij, const imag_axes_ft::IAFT *FT, bool restart,
                     std::array<std::string,3> datasets)
  -> std::tuple<double, double> {
  double conv_F = 0;
  double conv_Sigma = 0;
  if (it == 1) {
    // Just check changes w.r.t. mf
    if (context.node_comm.root()) {
      auto F_mf = nda::make_regular(sF_skij.local());
      std::string filename = h5_prefix + ".mbpt.h5";
      h5::file file(filename, 'r');
      h5::group grp(file);
      if (grp.has_subgroup("scf/iter0")) {
        auto mf_grp = grp.open_group("scf/iter0");
        if (mf_grp.has_dataset("F_skij")) {
          nda::h5_read(mf_grp, "F_skij", F_mf);
        } else if (mf_grp.has_dataset("Heff_skij")) {
          // checkpoint from a qp scf
          nda::h5_read(mf_grp, "Heff_skij", F_mf);
          nda::array<ComplexType, 4> H0(F_mf.shape());
          auto sys_grp = grp.open_group("system");
          nda::h5_read(sys_grp, "H0_skij", H0);
          F_mf -= H0;
        }
      }
      F_mf -= sF_skij.local();
      auto Fmax_iter = max_element(F_mf.data(), F_mf.data()+F_mf.size(),
                                   [](auto a, auto b) { return std::abs(a) < std::abs(b); });
      conv_F =  std::abs((*Fmax_iter));
    }
    context.node_comm.broadcast_n(&conv_F, 1, 0);
    auto Sigma_max_iter = max_element(sSigma_tskij.local().data(), sSigma_tskij.local().data()+sSigma_tskij.local().size(),
                                      [](auto a, auto b) { return std::abs(a) < std::abs(b); });
    conv_Sigma =  std::abs((*Sigma_max_iter));
    init_solver(context, iter_solver, it, h5_prefix, sF_skij, sSigma_tskij, FT);
  } else {
    if (iter_solver.iter_alg() == iter_scf::damping) {
      std::tie(conv_F, conv_Sigma) = damping_impl(context, iter_solver, it, h5_prefix,
                                                  sF_skij, sSigma_tskij, datasets);
    } else if (iter_solver.iter_alg() == iter_scf::DIIS) {
      std::tie(conv_F, conv_Sigma) = diis_impl(context, iter_solver, it, h5_prefix,
                                               sF_skij, sSigma_tskij, FT, restart, datasets);
    } else {
      utils::check(false, "scf_common::solve_iterative: unknown type of iterative algorithm.");
    }
  }
  return std::make_tuple(conv_F, conv_Sigma);
}

template<typename dyson_type>
void write_mf_data(mf::MF &mf,
                   const imag_axes_ft::IAFT &ft, dyson_type &dyson,
                   std::string output) {
  auto mpi = mf.mpi();
  sArray_t<Array_view_4D_t> sF_skij(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  sArray_t<Array_view_4D_t> sDm_skij(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  sArray_t<Array_view_5D_t> G_shm(math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {ft.nt_f(), mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  sArray_t<Array_view_5D_t> Sigma_shm(math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {ft.nt_f(), mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  hamilt::set_fock(mf, dyson.PSP(), sF_skij, true);
  double mu = 0.0;

  // init Green's function. By default, we update mu as well.
  update_G(dyson, mf, ft, sDm_skij, G_shm, sF_skij, Sigma_shm, mu, false);

  chkpt::write_metadata(mpi->comm, mf, ft, dyson.sH0_skij(), dyson.sS_skij(), output);
  chkpt::dump_scf(mpi->comm, 0, sDm_skij, G_shm, sF_skij, Sigma_shm, mu, output);
}

template<typename MPI_Context_t>
auto read_greens_function(MPI_Context_t &context, mf::MF *mf,
                          std::string filename, long scf_iter, std::string scf_grp)
-> sArray_t<Array_view_5D_t> {
  using math::shm::make_shared_array;

  h5::file file(filename, 'r');
  h5::group grp(file);

  nda::array<double, 1> tau_mesh;
  auto iaft_grp = h5::group(file).open_group("imaginary_fourier_transform");
  auto tau_grp = iaft_grp.open_group("tau_mesh");
  nda::h5_read(tau_grp, "fermion", tau_mesh);
  int nts = tau_mesh.shape(0);
  int ns = mf->nspin();
  int nkpts_ibz = mf->nkpts_ibz();
  int nbnd = mf->nbnd();

  auto sG_tskij = make_shared_array<Array_view_5D_t>(context.comm, context.internode_comm,
                                                     context.node_comm, {nts, ns, nkpts_ibz, nbnd, nbnd});

  auto iter_grp = h5::group(file).open_group(scf_grp+"/iter"+std::to_string(scf_iter));
  if (iter_grp.has_dataset("G_tskij")) {
    // it's a Dyson type calculation -> read Green's function
    sG_tskij.win().fence();
    if (context.node_comm.root()) {
      auto Gloc = sG_tskij.local();
      nda::h5_read(iter_grp, "G_tskij", Gloc);
    }
    sG_tskij.win().fence();
  } else {
    // it's a qp type calculation -> construct the Green's function on-the-fly
    auto ft = imag_axes_ft::read_iaft(filename, false);
    auto sMO_skia = make_shared_array<Array_view_4D_t>(
        context.comm, context.internode_comm, context.node_comm, {ns, nkpts_ibz, nbnd, nbnd});
    auto sE_ska = make_shared_array<Array_view_3D_t>(
        context.comm, context.internode_comm, context.node_comm, {ns, nkpts_ibz, nbnd});
    double mu;

    sMO_skia.win().fence();
    if (context.node_comm.root()) {
      auto MO_loc = sMO_skia.local();
      auto E_loc = sE_ska.local();
      nda::h5_read(iter_grp, "MO_skia", MO_loc);
      nda::h5_read(iter_grp, "E_ska", E_loc);
    }
    sMO_skia.win().fence();
    h5::h5_read(iter_grp, "mu", mu);

    update_G(sG_tskij, sMO_skia, sE_ska, mu, ft);
  }
  context.comm.barrier();
  return sG_tskij;
}


template auto eval_hf_energy(const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_4D_t>&,
                             nda::array_contiguous_const_view<double, 1>&, bool)
    -> std::tuple<double, double>;

template double eval_corr_energy(mpi3::communicator& comm, const imag_axes_ft::IAFT &,
                                 const sArray_t<Array_view_5D_t> &, const sArray_t<Array_view_5D_t> &,
                                 nda::array_contiguous_const_view<double, 1>&);

template void update_G(simple_dyson &, const mf::MF &, const imag_axes_ft::IAFT &,
                       sArray_t<Array_view_4D_t> & Dm, sArray_t<Array_view_5D_t> &G,
                       const sArray_t<Array_view_4D_t> & F, const sArray_t<Array_view_5D_t> &Sigma, double&,
                       bool);
template void update_G(dca_dyson &, const mf::MF &, const imag_axes_ft::IAFT &,
                       sArray_t<Array_view_4D_t> & Dm, sArray_t<Array_view_5D_t> &G,
                       const sArray_t<Array_view_4D_t> & F, const sArray_t<Array_view_5D_t> &Sigma, double&,
                       bool);

template double update_mu(double, simple_dyson&, const mf::MF &, const imag_axes_ft::IAFT &,
                          const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_5D_t>&,
                          const sArray_t<Array_view_5D_t>&);
template double update_mu(double, dca_dyson &, const mf::MF &, const imag_axes_ft::IAFT &,
                          const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_5D_t>&,
                          const sArray_t<Array_view_5D_t>&);

template auto solve_iterative(utils::mpi_context_t<mpi3::communicator>&, iter_scf::iter_scf_t&, long, std::string,
                              sArray_t<Array_view_4D_t>&, sArray_t<Array_view_5D_t>&, const imag_axes_ft::IAFT*, bool,
                              std::array<std::string,3>)
         -> std::tuple<double, double>;

template void write_mf_data(mf::MF&, const imag_axes_ft::IAFT&, simple_dyson&,
                            std::string);
template auto read_greens_function(utils::mpi_context_t<>&, mf::MF*, std::string, long, std::string)
    -> sArray_t<Array_view_5D_t>;
} // methods
