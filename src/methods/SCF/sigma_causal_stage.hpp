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


#ifndef COQUI_SIGMA_CAUSAL_STAGE_HPP
#define COQUI_SIGMA_CAUSAL_STAGE_HPP

#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "h5/h5.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "utilities/omp_threads.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/SCF/mb_solver_t.h"
#include "methods/SCF/sigma_causal_projection.hpp"
#include "methods/scr_coulomb/scr_coulomb_t.h"
#include "methods/vertex/vertex_debug.hpp"

namespace methods {

/**
 * The causal projection of the fresh Sigma_c as a stage of the SCF loops (scf_loop and the
 * quasiparticle loops of qp_scf_loop): -Sigma_c(tau, s, k) is replaced by its best fit with PSD
 * spectral weights (sigma_causal_projection.hpp). ON by default when a vertex adds to Sigma
 * (scr_coulomb_t::sigma_vertex_active), OFF for plain GW; vertex_debug scf_causal_proj = 0 / 1
 * overrides either way. The projector (real-frequency nodes, ADMM operators) is built once per tau
 * mesh; the ADMM state of every (s, k) this rank projects is kept across iterations as the warm
 * start, so one instance lives for the whole loop. The (s, k) blocks are split over ALL ranks; each
 * node's shared array receives its own ranks' blocks and the node roots sum the rest in (every
 * block is owned by one rank).
 */
class sigma_causal_stage {
public:
  template<typename corr_solver_t>
  static bool enabled(solvers::mb_solver_t<corr_solver_t> const& mb_solver) {
    return mb_solver.corr != nullptr and
        (vertex_debug::get("scf_causal_proj") ? vertex_debug::flag("scf_causal_proj")      // vertex_debug: scf_causal_proj
                                              : (mb_solver.scr_eri != nullptr and mb_solver.scr_eri->sigma_vertex_active()));
  }

  /**
   * Projects sSigma_tskij in place.
   * @param sSigma_tskij - [INPUT/OUTPUT] node-shared Sigma_c(tau, s, k, i, j)
   * @param mpi          - [INPUT] MPI context (pointer-like)
   * @param FT           - [INPUT] imaginary-axis transform of Sigma_c
   * @param output_iter  - [INPUT] iteration label of the log and of the diagnostic dump
   * @param caller       - [INPUT] name of the calling loop, for the messages
   * MEM is the memory space of the loop: the device projection is the default for a device loop.
   */
  template<MEMORY_SPACE MEM, typename shared_array_t, typename mpi_ptr_t>
  void apply(shared_array_t &sSigma_tskij, mpi_ptr_t const& mpi, const imag_axes_ft::IAFT &FT,
             long output_iter, std::string const& caller) {
    // Device or host projection: the device path (sigma_causal_projection.hpp::project_device, causal_proj_cuda.cu) is the
    // default on a CUDA build when the run computes on the device (--compute gpu / unified); vertex_debug
    // scf_causal_proj_device = 0 selects the host path, = 1 the device path also under --compute cpu. Asking for the device
    // path on a build without CUDA aborts; a device path that cannot run aborts too (no silent host fallback).
    const bool causal_dev_auto = causal_proj::projector::device_built() and MEM != HOST_MEMORY;
    const bool causal_dev_set = bool(vertex_debug::get("scf_causal_proj_device"));
    const bool causal_dev = causal_dev_set ? vertex_debug::flag("scf_causal_proj_device")    // vertex_debug: scf_causal_proj_device
                                           : causal_dev_auto;
    utils::check(not causal_dev or causal_proj::projector::device_built(),
                 "{}: vertex_debug scf_causal_proj_device = 1 asks for the device causal projection, but this build has no "
                 "CUDA (ENABLE_CUDA off). Remove the key, or set it to 0 for the host path.", caller);
    const auto t_proj0 = std::chrono::steady_clock::now();
    if (not causal_P) {
      causal_proj::options o;
      o.maxit = long(vertex_debug::number("scf_causal_proj_maxit", double(o.maxit)));   // vertex_debug: scf_causal_proj_maxit
      o.tol = vertex_debug::number("scf_causal_proj_tol", o.tol);                         // vertex_debug: scf_causal_proj_tol
      o.rho = vertex_debug::number("scf_causal_proj_rho", o.rho);                         // vertex_debug: scf_causal_proj_rho
      // the nodes stay inside the imaginary-axis window: spectral weight placed beyond FT.wmax() is not carried by the tau
      // basis through the Dyson step's tau -> i w transform (a causal Sigma_c(tau) would then give a non-causal G)
      o.wmax = vertex_debug::number("scf_causal_proj_wmax", FT.wmax());                   // vertex_debug: scf_causal_proj_wmax
      o.wmin = vertex_debug::number("scf_causal_proj_wmin", o.wmin);                      // vertex_debug: scf_causal_proj_wmin
      o.nw = long(vertex_debug::number("scf_causal_proj_nw", double(o.nw)));             // vertex_debug: scf_causal_proj_nw
      o.select = vertex_debug::number("scf_causal_proj_select", o.select);                // vertex_debug: scf_causal_proj_select
      causal_P.emplace(FT.beta(), FT.tau_mesh(), o);
      double wlo = 1e300, whi = 0.0;
      for (auto const &v : causal_P->nodes()) { wlo = std::min(wlo, std::abs(v)); whi = std::max(whi, std::abs(v)); }
      app_log(1, "  [causal projection] Sigma_c is projected onto PSD spectral weights every iteration ({}): {} real-frequency "
                 "nodes (|w| from {:.2e} to {:.2e} Ha), ADMM maxit {}, tol {:.1e}, rho {:.1e} s_max^2, Anderson memory {}; (s, k) blocks over {} ranks; "
                 "{} path ({})",
              vertex_debug::get("scf_causal_proj") ? "vertex_debug scf_causal_proj" : "default: a vertex adds to Sigma",
              causal_P->n_nodes(), wlo, whi, o.maxit, o.tol, o.rho, o.aa_mem, mpi->comm.size(),
              causal_dev ? "DEVICE" : "HOST",
              causal_dev_set ? "vertex_debug scf_causal_proj_device"
                             : (causal_dev ? "default: CUDA build, the run computes on the device"
                                           : (causal_proj::projector::device_built() ? "default: --compute cpu" : "default: no CUDA in this build")));
    }
    auto S = sSigma_tskij.local();
    const long ns = S.shape(1), nk = S.shape(2);
    const long nr = mpi->comm.size(), me = mpi->comm.rank();
    const long n_nodes = mpi->comm.size() / mpi->node_comm.size();
    // diagnostic dump (vertex_debug scf_causal_proj_dump = <file prefix>): the projection's input (the fresh Sigma_c) and
    // output of this iteration, with beta and the tau mesh, for offline solver studies; written by the global root
    const auto cp_dump = vertex_debug::get("scf_causal_proj_dump");                     // vertex_debug: scf_causal_proj_dump = <file prefix>
    auto cp_dump_write = [&](std::string const &what) {
      if (not cp_dump or not mpi->comm.root()) return;
      const std::string fn = *cp_dump + ".cproj_dump.h5";
      h5::file f(fn, std::filesystem::exists(fn) ? 'a' : 'w');
      h5::group g(f);
      if (not g.has_dataset("beta")) {
        h5::h5_write(g, "beta", FT.beta());
        nda::array<double, 1> x(FT.tau_mesh());
        nda::h5_write(g, "tau_mesh", x);
      }
      auto gi = g.has_subgroup("iter" + std::to_string(output_iter)) ? g.open_group("iter" + std::to_string(output_iter))
                                                                   : g.create_group("iter" + std::to_string(output_iter));
      nda::array<ComplexType, 5> A(S);
      nda::h5_write(gi, what, A);
    };
    if (cp_dump) {
      sSigma_tskij.win().fence();
      cp_dump_write("Sigma_in");
      mpi->comm.barrier();
    }
    double y2 = 0.0, r2 = 0.0, rel_max = 0.0, lb = 1e300, la = 1e300;
    double fit_max = 0.0;
    long it_max = 0, n_conv = 0, n_blk = 0, aa_rst = 0;
    sSigma_tskij.win().fence();
    // several nodes: every node zeroes the blocks its ranks do not own, so the node roots' sum assembles the projection
    if (n_nodes > 1) {
      std::vector<char> mine(size_t(ns * nk), 0);
      for (long p = 0; p < ns * nk; ++p) {
        const long owner = p % nr;
        mine[size_t(p)] = char(mpi->node_comm.all_reduce_value(long(owner == me), std::plus<>{}) > 0);
      }
      if (mpi->node_comm.root())
        for (long p = 0; p < ns * nk; ++p)
          if (not mine[size_t(p)]) S(nda::range::all, p / nk, p % nk, nda::range::all, nda::range::all) = ComplexType(0.0);
      sSigma_tskij.win().fence();
    }
    std::vector<long> my_p;
    for (long p = me; p < ns * nk; p += nr) my_p.push_back(p);
    std::vector<causal_proj::stats> my_st(my_p.size());
    double dev_info[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};   // sweeps (warm / cold), clips, kept, batches, loop wall, mean sweeps
    if (causal_dev) {
#if defined(ENABLE_CUDA)
      // every block this rank owns in one call (batched on the device; the warm-start states stay on the host)
      std::vector<nda::array<ComplexType, 3>> blks;
      blks.reserve(my_p.size());
      std::vector<nda::array<ComplexType, 3> *> pb;
      std::vector<nda::matrix<ComplexType> *> pc, pu;
      for (long p : my_p) {
        blks.emplace_back(S(nda::range::all, p / nk, p % nk, nda::range::all, nda::range::all));
        auto &cs = causal_state[p];
        pc.push_back(&cs.first);
        pu.push_back(&cs.second);
      }
      for (auto &b : blks) pb.push_back(&b);
      causal_proj_cuda::call_info ci;
      my_st = causal_P->project_device(pb, pc, pu, &ci);
      for (size_t i = 0; i < my_p.size(); ++i)
        S(nda::range::all, my_p[i] / nk, my_p[i] % nk, nda::range::all, nda::range::all) = blks[i];
      dev_info[0] = double(ci.sweeps_max); dev_info[1] = double(ci.sweeps_max_cold); dev_info[2] = ci.psd_calls;
      dev_info[3] = ci.psd_kept; dev_info[4] = double(ci.nbatch); dev_info[5] = ci.t_loop; dev_info[6] = ci.sweeps_mean;
#endif
    } else {
      for (size_t i = 0; i < my_p.size(); ++i) {
        const long p = my_p[i], is = p / nk, ik = p % nk;
        nda::array<ComplexType, 3> blk = S(nda::range::all, is, ik, nda::range::all, nda::range::all);
        auto &cs = causal_state[p];
        my_st[i] = causal_P->project(blk, cs.first, cs.second);
        S(nda::range::all, is, ik, nda::range::all, nda::range::all) = blk;
      }
    }
    for (auto const &st : my_st) {
      y2 += st.y_norm * st.y_norm; r2 += st.removed * st.removed;
      rel_max = std::max(rel_max, st.removed / std::max(st.y_norm, 1e-300));
      lb = std::min(lb, st.lam_before); la = std::min(la, st.lam_after);
      it_max = std::max(it_max, st.iters);
      fit_max = std::max(fit_max, st.fit_res);
      n_conv += st.converged ? 1 : 0; ++n_blk; aa_rst += st.aa_restarts;
    }
    sSigma_tskij.win().fence();
    if (n_nodes > 1) {
      if (mpi->node_comm.root()) mpi->internode_comm.all_reduce_in_place_n(S.data(), S.size(), std::plus<>{});
      mpi->comm.barrier();
      sSigma_tskij.win().fence();
    }
    y2 = mpi->comm.all_reduce_value(y2, std::plus<>{});
    r2 = mpi->comm.all_reduce_value(r2, std::plus<>{});
    rel_max = mpi->comm.all_reduce_value(rel_max, boost::mpi3::max<>{});
    lb = -mpi->comm.all_reduce_value(-lb, boost::mpi3::max<>{});
    la = -mpi->comm.all_reduce_value(-la, boost::mpi3::max<>{});
    it_max = mpi->comm.all_reduce_value(it_max, boost::mpi3::max<>{});
    fit_max = mpi->comm.all_reduce_value(fit_max, boost::mpi3::max<>{});
    n_conv = mpi->comm.all_reduce_value(n_conv, std::plus<>{});
    n_blk = mpi->comm.all_reduce_value(n_blk, std::plus<>{});
    aa_rst = mpi->comm.all_reduce_value(aa_rst, std::plus<>{});
    for (int i : {0, 1, 4, 5, 6}) dev_info[i] = mpi->comm.all_reduce_value(dev_info[i], boost::mpi3::max<>{});
    for (int i : {2, 3}) dev_info[i] = mpi->comm.all_reduce_value(dev_info[i], std::plus<>{});
    mpi->comm.barrier();
    if (cp_dump) {
      sSigma_tskij.win().fence();
      cp_dump_write("Sigma_out");
      mpi->comm.barrier();
    }
    const double t_proj = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_proj0).count();
    app_log(1, "  [causal projection] iteration {}: removed |dSigma_c| = {:.3e} ({:.2e} of |Sigma_c|, worst (s,k) {:.2e}); "
               "lambda_min(-Sigma_c) {:.3e} -> {:.3e}; ADMM iterations <= {}, converged {} of {} (s, k) blocks, max fit residual {:.2e} "
               "of |Sigma_c|, Anderson restarts {}; {:.1f} s ({})", output_iter, std::sqrt(r2),
            std::sqrt(r2 / std::max(y2, 1e-300)), rel_max, lb, la, it_max, n_conv, n_blk, fit_max, aa_rst,
            t_proj,
            causal_dev ? "device" : std::format("host, omp_threads {}", utils::omp_threads()));
    if (causal_dev)
      app_log(2, "  [causal projection] device: ADMM loop {:.1f} s on the slowest rank, batches <= {}; Jacobi sweeps <= {} (mean <= {:.2f}) per "
                 "warm-started PSD clip ({:.0f} clips, {:.1f} % kept by the semidefinite test), <= {} per cold eigenvalue solve",
              dev_info[5], long(dev_info[4]), long(dev_info[0]), dev_info[6], dev_info[2], 100.0 * dev_info[3] / std::max(dev_info[2], 1.0),
              long(dev_info[1]));
    // a projection that removes a sizeable part of Sigma_c changes the physics, not just a non-causal seed: say so loudly
    const double warn_rel = vertex_debug::number("scf_causal_proj_warn", 1e-2);            // vertex_debug: scf_causal_proj_warn
    if (rel_max > warn_rel)
      app_log(1, "  [WARNING] causal projection, iteration {}: it removed {:.2e} of Sigma_c in the worst (s, k) (> {:.0e}). The "
                 "self-energy is strongly non-causal (lambda_min(-Sigma_c) = {:.3e}); the run continues with its causal part, "
                 "which is NOT the requested Sigma. Check the vertex (scale, window, imaginary-axis window) before trusting it.",
              output_iter, rel_max, warn_rel, lb);
  }

private:
  std::optional<causal_proj::projector> causal_P;
  std::map<long, std::pair<nda::matrix<ComplexType>, nda::matrix<ComplexType>>> causal_state;
};

} // methods

#endif // COQUI_SIGMA_CAUSAL_STAGE_HPP
