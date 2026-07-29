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


#include <cmath>

#include <cstdlib>
#include <map>
#include <thread>
#include <vector>

#include "methods/SCF/simple_dyson.h"
#include "nda/h5.hpp"
#include "nda/linalg.hpp"
#include "nda/nda.hpp"

namespace methods {

namespace {

  /**
   * Threads for the eigenvalue loop in compute_eigenspectra.
   *
   * The eigensolves are independent and they are the only thing running at that
   * point, while a GPU rank typically owns many more cores than it uses: with
   * one rank per GPU the job holds 16 cores per rank and 15 sit idle. Measured
   * on an A100 node (n=500, 137 problems = one rank's share at 8 ranks):
   * 27.5 s on one core, 1.9 s on sixteen. The device is not the answer here --
   * cuSOLVER's Xgeev is 1.6x one core and 4.4 s across four streams, MAGMA's
   * hybrid zgeev 1.15x -- because non-symmetric eigensolves do not map to GPUs
   * and the CUDA stack has no batched form of them at this size.
   *
   * std::thread rather than OpenMP: the build sets ENABLE_OPENMP=OFF, so a
   * pragma would silently do nothing. LAPACK inside each thread must stay
   * sequential, which OMP_NUM_THREADS=1 (what every run script sets) ensures;
   * the count is divided by OMP_NUM_THREADS otherwise, so a threaded BLAS
   * cannot oversubscribe.
   */
  int eigenspectra_threads()
  {
    static const int nthreads = []() {
      auto env_int = [](const char* name) -> long {
        const char* v = std::getenv(name);
        if (v == nullptr) return 0;
        long n = std::strtol(v, nullptr, 10);
        return (n > 0) ? n : 0;
      };
      long n = env_int("COQUI_EIG_THREADS");
      if (n > 0) return int(n);
      long budget = env_int("SLURM_CPUS_PER_TASK");
      if (budget <= 0) return 1;
      long per_blas = env_int("OMP_NUM_THREADS");
      if (per_blas > 1) budget /= per_blas;
      return int((budget > 0) ? budget : 1);
    }();
    return nthreads;
  }

}

  template<typename G_t, typename F_t, typename Sigma_t>
  void simple_dyson::solve_dyson(G_t &_G_shm,
                                 const F_t &_sF_skij, const Sigma_t &_Sigma_shm, double mu) {
    _Timer.start("DYSON");
    using math::nda::make_distributed_array;
    using Array_5D_t = nda::array<ComplexType, 5>;

    // processor grid for Dyson equation
    std::array<long, 5> w_pgrid;
    std::array<long, 5> w_bsize;
    {
      int np = _context->comm.size();
      int nwpools = utils::find_proc_grid_max_npools(np, _nw, 0.4);
      np /= nwpools;
      int nkpools = utils::find_proc_grid_max_npools(np, _nkpts_ibz, 0.4);
      np /= nkpools;
      int np_i = utils::find_proc_grid_min_diff(np, 1, 1);
      int np_j = np / np_i;

      w_pgrid = {nwpools, 1, nkpools, np_i, np_j};
      long ibsize = std::min({1024, _nbnd/np_i, _nbnd/np_j});
      w_bsize = {1, 1, 1, ibsize, ibsize};

      utils::check(nwpools*nkpools*np_i*np_j == _context->comm.size(), "solve_dyson: pgrid mismatches!");

      app_log(2, "Dyson equation for Green's function:");
      app_log(2, "  - processor grid for G/Self-energy: (w, k, i, j) = ({}, {}, {}, {})", nwpools, nkpools, np_i, np_j);
      app_log(2, "  - block size: (w, k, i, j) = ({}, {}, {}, {})", 1, 1, ibsize, ibsize);
    }

    _Timer.start("SIGMA_TAU_TO_W");
    auto dSigma_wskij = distributed_tau_to_w(_context->comm, _Sigma_shm, *_FT, w_pgrid, w_bsize);
    _Timer.stop("SIGMA_TAU_TO_W");
    auto dG_wskij = make_distributed_array<Array_5D_t>(_context->comm, w_pgrid,
                                                       {_nw, _ns, _nkpts_ibz, _nbnd, _nbnd}, w_bsize);
    auto [nw_loc, ns_loc, nk_loc, ni_loc, nj_loc] = dSigma_wskij.local_shape();
    auto [w_org, s_org, k_org, i_org, j_org] = dSigma_wskij.origin();
    auto i_rng = dSigma_wskij.local_range(3);
    auto j_rng = dSigma_wskij.local_range(4);

    // Setup wk_intra_comm
    int color = w_org*_nkpts_ibz + k_org;
    int key = _context->comm.rank();
    mpi3::communicator wk_intra_comm = _context->comm.split(color, key);
    utils::check(wk_intra_comm.size() == w_pgrid[3]*w_pgrid[4], "wk_intra_comm.size() != pgrid[3]*pgrid[4]");
    auto dX = make_distributed_array<nda::array<ComplexType, 2>>(wk_intra_comm, {w_pgrid[3],w_pgrid[4]},
        {_nbnd,_nbnd}, {w_bsize[3],w_bsize[4]});

    auto S  = _sS_skij.local();
    auto H0 = _sH0_skij.local();
    auto F  = _sF_skij.local();
    auto Sigma_w_loc = dSigma_wskij.local();
    auto G_w_loc = dG_wskij.local();
    auto X_loc = dX.local();

    // Dyson on w-axis
    _Timer.start("DYSON_LOOP");
    for (long nsk = 0; nsk < nw_loc*ns_loc*nk_loc; ++nsk) {
      long n = nsk / (ns_loc*nk_loc); // nsk = n*ns_loc*nk_loc + s*nk_loc + k
      long s = (nsk / nk_loc) % ns_loc;
      long k = nsk % nk_loc;

      long wn = _FT->wn_mesh()(n+w_org);
      ComplexType omega_mu = _FT->omega(wn) + mu;
      X_loc = omega_mu * S(s+s_org,k+k_org,i_rng,j_rng) - H0(s+s_org,k+k_org,i_rng,j_rng)
              - F(s+s_org,k+k_org,i_rng,j_rng) - Sigma_w_loc(n,s,k,nda::ellipsis{});
      math::nda::slate_ops::inverse(dX);
      G_w_loc(n,s,k,nda::ellipsis{}) = X_loc;
    }
    _Timer.stop("DYSON_LOOP");
    dSigma_wskij.reset();
    _context->comm.barrier();

    // G(w) -> G(tau)
    {
      int np = _context->comm.size();
      long nkpools = utils::find_proc_grid_max_npools(np, _nkpts_ibz, 0.2);
      np /= nkpools;
      long np_i = utils::find_proc_grid_min_diff(np, 1, 1);
      long np_j = np / np_i;

      auto dG_wskij_tmp = make_distributed_array<Array_5D_t>(_context->comm, {1, 1, nkpools, np_i, np_j},
                                                             {_nw, _ns, _nkpts_ibz, _nbnd, _nbnd});
      _Timer.start("REDISTRIBUTE");
      math::nda::redistribute(dG_wskij, dG_wskij_tmp);
      _Timer.stop("REDISTRIBUTE");
      dG_wskij.reset(); 

      auto dG_tskij = make_distributed_array<Array_5D_t>(_context->comm, {1, 1, nkpools, np_i, np_j},
                                                         {_nts, _ns, _nkpts_ibz, _nbnd, _nbnd});
      auto Gt_loc = dG_tskij.local();
      auto Gw_loc = dG_wskij_tmp.local();
      _FT->w_to_tau(Gw_loc, Gt_loc, imag_axes_ft::fermi);
      dG_wskij_tmp.reset();

      _FT->check_leakage(dG_tskij, imag_axes_ft::fermi, "Green's function");

      // Gather to shared memory
      _Timer.start("DYSON_GATHER");
      math::nda::gather_to_shm(dG_tskij, _G_shm);
      _Timer.stop("DYSON_GATHER");
    }
    _context->comm.barrier();
    _Timer.stop("DYSON");
    print_timers();
  }

  template<typename Dm_t, typename G_t, typename F_t, typename Sigma_t>
  void simple_dyson::solve_dyson(Dm_t &_sDm_skij, G_t &_G_shm, const F_t &_sF_skij,
                                 const Sigma_t &_Sigma_shm, double mu) {
    solve_dyson(_G_shm, _sF_skij, _Sigma_shm, mu);
    if (_context->node_comm.root()) {
      auto Dm = _sDm_skij.local();
      _FT->tau_to_beta(_G_shm.local(), Dm);
      Dm *= -1;
    }
    _context->comm.barrier();
  }

  template<typename X_t, typename Xt_t>
  void simple_dyson::compute_eigenspectra([[maybe_unused]] double mu, const X_t&_sF_skij, [[maybe_unused]] const Xt_t &_G_shm, const Xt_t &_Sigma_shm, nda::array<ComplexType, 4> &spectra){
    utils::check(spectra.shape() == std::array<long, 4>{_nw, _ns, _nkpts_ibz, _nbnd},
                 "simple_dyson::compute_eigenspectra: Incorrect dimension for spectra.");
    using math::shm::make_shared_array;
    _Timer.start("EIGSPEC");
    spectra() = 0.0;
    auto sS_inv = make_shared_array<Array_view_4D_t>(*_context, {_ns, _nkpts_ibz, _nbnd, _nbnd});
    auto S  = _sS_skij.local();
    auto H0 = _sH0_skij.local();
    auto F  = _sF_skij.local();
    auto S_inv = sS_inv.local();

    int node_rank = _context->node_comm.rank();
    int node_size = _context->node_comm.size();
    _Timer.start("EIGSPEC_SINV");
    sS_inv.win().fence();
    for (size_t sk = node_rank; sk < _ns*_nkpts_ibz; sk+=node_size) {
      size_t is = sk / _nkpts_ibz;
      size_t ik = sk % _nkpts_ibz;
      nda::matrix_const_view<ComplexType> S_ij = S(is, ik, nda::ellipsis{});
      S_inv(is, ik, nda::ellipsis{}) = nda::inverse(S_ij);
    }
    sS_inv.win().fence();
    _Timer.stop("EIGSPEC_SINV");

    // Sigma(tau) is Hermitian -- scf_loop hermitizes it every iteration -- and
    // the fermionic Matsubara mesh is symmetric, so Sigma(-iw) = Sigma(iw)^H.
    // With S, H0 and F Hermitian, A(-iw) = S^-1 (H0+F+Sigma(iw)^H) is similar
    // to A(iw)^H (S^-1 B and B S^-1 are similar), so its eigenvalues are the
    // conjugates of A(iw)'s. Diagonalizing the positive half of the mesh and
    // conjugating therefore halves the work exactly. Verified against the full
    // loop before this was switched on; if a grid ever turns up that is not
    // symmetric (or contains w=0) we fall back to every frequency.
    nda::array<long, 1> mirror(_nw);
    bool symmetric_mesh = true;
    {
      auto wn = _FT->wn_mesh();
      std::map<long, long> pos_of;
      for (long n = 0; n < long(_nw); ++n) pos_of[wn(n)] = n;
      for (long n = 0; n < long(_nw); ++n) {
        auto it = pos_of.find(-wn(n));
        if (wn(n) == 0 or it == pos_of.end()) { symmetric_mesh = false; break; }
        mirror(n) = it->second;
      }
    }

    // Frequencies this rank owns: the positive half, strided over ranks. Each
    // (n,is,ik) is written by exactly one rank, so the all-reduce below still
    // just sums disjoint contributions.
    std::vector<long> local_w;
    {
      auto wn = _FT->wn_mesh();
      long j = 0;
      for (long n = 0; n < long(_nw); ++n) {
        if (symmetric_mesh and wn(n) < 0) continue;   // covered by its mirror
        if (j % _context->comm.size() == _context->comm.rank()) local_w.push_back(n);
        ++j;
      }
    }
    const long nwl = long(local_w.size());

    // Transform only this rank's frequencies. tau_to_w is one gemm against the
    // (nw x nt) transform matrix, so a row subset gives a partial transform:
    // the buffer drops from nw to nwl (4.4 GB -> ~0.3 GB per rank at 8 ranks,
    // which is also most of B5) and the gemm from nw to nwl rows. Every rank
    // used to build the entire Sigma(w) and then use 1/n_ranks of it.
    _Timer.start("EIGSPEC_SIGMA_FT");
    nda::array<ComplexType, 5> Sigmaw_wskij(std::max(nwl, 1l), _ns, _nkpts_ibz, _nbnd, _nbnd);
    if (nwl > 0) {
      auto Twt = _FT->Twt_ff();
      nda::array<ComplexType, 2> Twt_loc(nwl, Twt.shape(1));
      for (long j = 0; j < nwl; ++j)
        Twt_loc(j, nda::range::all) = Twt(local_w[j], nda::range::all);
      long dim1 = long(_ns)*_nkpts_ibz*_nbnd*_nbnd;
      auto S_ti_2D = nda::reshape(_Sigma_shm.local(), std::array<long,2>{long(_FT->nt_f()), dim1});
      auto S_wi_2D = nda::reshape(Sigmaw_wskij, std::array<long,2>{nwl, dim1});
      nda::blas::gemm(Twt_loc, S_ti_2D, S_wi_2D);
    }
    _Timer.stop("EIGSPEC_SIGMA_FT");

    _Timer.start("EIGSPEC_GEEV");
    {
      const long ntask = nwl*long(_ns*_nkpts_ibz);
      const int nthreads = std::max(1, std::min<int>(eigenspectra_threads(),
                                                     int(std::min<long>(ntask, 1024))));
      auto worker = [&](int tid) {
        // Per-thread scratch: nda's geigenvalues copies into its own workspace,
        // so the only sharing is the read-only S_inv/H0/F and disjoint writes
        // into spectra.
        nda::matrix<ComplexType> FpSigma_t(_nbnd, _nbnd), SFS_t(_nbnd, _nbnd);
        for (long task = tid; task < ntask; task += nthreads) {
          long j = task / long(_ns*_nkpts_ibz);
          long i = task % long(_ns*_nkpts_ibz);
          long n = local_w[j];
          size_t is = size_t(i) / _nkpts_ibz;  // i = is * _nkpts_ibz + ik
          size_t ik = size_t(i) % _nkpts_ibz;
          auto Sigma_ij = Sigmaw_wskij(j, is, ik, nda::range::all, nda::range::all);
          FpSigma_t = H0(is, ik, nda::ellipsis{}) + F(is, ik, nda::ellipsis{}) + Sigma_ij;
          nda::blas::gemm(ComplexType(1.0), S_inv(is, ik, nda::range::all, nda::range::all),
                          FpSigma_t, ComplexType(0.0), SFS_t);
          // Matsubara quantities are not Hermitian: Sigma(iw)^H = Sigma(-iw),
          // and S^-1 (H0+F+Sigma) is not Hermitian even where the sum is, so
          // this is a general eigenvalue problem and heev/hegv do not apply.
          auto eigvals = nda::linalg::geigenvalues(SFS_t);
          spectra(n, is, ik, nda::range::all) = eigvals;
          if (symmetric_mesh) {
            long nm = mirror(n);
            for (long b = 0; b < long(_nbnd); ++b)
              spectra(nm, is, ik, b) = std::conj(eigvals(b));
          }
        }
      };
      if (nthreads <= 1) {
        worker(0);
      } else {
        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (int tid = 0; tid < nthreads; ++tid) pool.emplace_back(worker, tid);
        for (auto& th : pool) th.join();
      }
    }
    _Timer.stop("EIGSPEC_GEEV");
    _Timer.start("EIGSPEC_REDUCE");
    _context->comm.all_reduce_in_place_n(spectra.data(), spectra.size(), std::plus<>{});
    _Timer.stop("EIGSPEC_REDUCE");
    _Timer.stop("EIGSPEC");
  }




  /** Instantiation of public template **/
  template void simple_dyson::solve_dyson(sArray_t<Array_view_5D_t>&,
      const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_5D_t> &, double);
  template void simple_dyson::solve_dyson(sArray_t<Array_view_4D_t>&, sArray_t<Array_view_5D_t>&,
      const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_5D_t> &, double);

  template void simple_dyson::compute_eigenspectra(double,const sArray_t<Array_view_4D_t>&,
      const sArray_t<Array_view_5D_t> &, const sArray_t<Array_view_5D_t> &, nda::array<ComplexType, 4> &);

} // methods
