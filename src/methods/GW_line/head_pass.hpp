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

#ifndef COQUI_METHODS_GW_LINE_HEAD_PASS_HPP
#define COQUI_METHODS_GW_LINE_HEAD_PASS_HPP

/**
 * S9b: ONE pass Pi -> W -> head h(q, zeta_i) = eps^-1_00(q, zeta_i) - 1 from a given pole set (the converged G) on a bosonic
 * line at an arbitrary angle theta_b (notes section "Optics from the line", flatter final line):
 *   - own bosonic basis bosonic_basis_t(theta_b, lam_b, eps, bos_gap, nline, npole) (rank ~ 1/theta_b; at 5 deg the
 *     defaults 800 / 1200 saturate: use npole 2400, nline 3000, notes/bosonic_closure_design.md 7.3),
 *   - own Pi time nodes on the rays theta_t = theta_b / 2: the time-node ID (time_id_t) of the Pi particle / hole products
 *     for the current pole ranges (as line_time_grids_t; only the two Pi grids are built), or the generic GL rays,
 *   - polarization -> screened_interaction (W at the nodes, own Dyson layout / Coulomb blocks) -> head_nodes_partial,
 *     in pair-closed q groups sized from a host-memory budget (the Pi group is g N_zeta block; at 5 deg N_zeta ~ 4x).
 * Used (a) by the optics after convergence for a flatter line (optics.theta_deg), (b) to recompute the head at the SCF angle
 * when the checkpoint predates the head group (S9a), (c) by the kernel validation at 10 / 5 deg ([V6][flat]).
 * Collective over mpi.comm (Coulomb blocks: lockstep thc.Z; redistributions; one all_reduce of the heads).
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "nda/nda.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/check.hpp"
#include "utilities/Timer.hpp"
#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "numerics/line_dlr/bosonic_basis.hpp"
#include "numerics/line_dlr/time_ray.hpp"
#include "numerics/line_dlr/time_id.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/polarization.hpp"
#include "methods/GW_line/screened.hpp"
#include "methods/GW_line/time_grids.hpp"
#include "methods/GW_line/head.hpp"

namespace methods::gw_line {

struct head_pass_params_t {
  double theta_deg = 20.0;          ///< bosonic line angle; time rays at theta_deg / 2
  double lam_b = 12.0, eps = 1e-10, bos_gap = 0.02;
  long nline = -1, npole = -1;      ///< basis selection grids; < 0: 1200 / 800 above 7.5 deg, 3000 / 2400 below
  std::string time_grid = "id";     ///< "id" | "gl"
  double time_eps = 1e-10, time_pad = 1.25, time_oversample = 1.0, ray_decades = 36.0;
  long t_chunk = 0;                 ///< 0: kernel default
  long qgroup = 0;                  ///< q-group size of the Pi -> W stage; 0: from mem_gb
  double mem_gb = 2.0;              ///< host budget per rank for the Pi group (1.8 x g N_zeta block x 16 B)

  // time-grid resolution on flat rays (S9b, [V6][flat], [.flat_scan]): the ID LS energy grid / candidate s grid and the GL
  // panels per e-fold; < 0: auto = the SCF defaults (120 / 40 per e-fold, GL 3 per e-fold) with the candidate s density
  // x sqrt(flat_scale()) and the GL panels x flat_scale() (flat_scale = the ray-angle factor)
  double id_nE_per_efold = -1.0, id_ns_per_efold = -1.0, gl_per_efold = -1.0, id_smax_fac = -1.0;
  long gl_nn = 16;

  /// max(1, sin(10 deg) / sin(theta_t)): 1 at the SCF angle (theta_t = 10 deg), 2 at theta_t = 5 deg, 4 at 2.5 deg
  double flat_scale() const {
    const double st = std::sin(0.5 * theta_deg * std::numbers::pi / 180.0), s10 = std::sin(10.0 * std::numbers::pi / 180.0);
    return std::max(1.0, s10 / st);
  }
  long nline_eff() const { return nline > 0 ? nline : (theta_deg >= 7.5 ? 1200 : 3000); }
  long npole_eff() const { return npole > 0 ? npole : (theta_deg >= 7.5 ? 800 : 2400); }
};

/// bosonic basis + Pi time nodes at one angle
struct head_pass_grid_t {
  double theta = 0.0, theta_t = 0.0;
  std::optional<numerics::line_dlr::bosonic_basis_t> bos;
  std::optional<numerics::line_dlr::time_nodes_t> pi_p, pi_h;
  std::string kind;
  double Emin = 0.0, Emax = 0.0;   ///< Pi summed-energy range of the poles
  double t_basis = 0.0, t_grid = 0.0;
};

/// Collective (the ID grids are built on ranks 0 / 1 and broadcast, as line_time_grids_t).
inline head_pass_grid_t head_pass_grid(pole_data_t const &poles, head_pass_params_t const &p, boost::mpi3::communicator &comm) {
  using numerics::line_dlr::sector_t;
  head_pass_grid_t g;
  g.theta   = p.theta_deg * std::numbers::pi / 180.0;
  g.theta_t = 0.5 * g.theta;
  g.kind    = p.time_grid;
  auto t0   = std::chrono::steady_clock::now();
  // the pivoted-QR selection (5 deg: 6000 x 2400, ~350 s) on the root only, poles and nodes broadcast (the other ranks
  // construct a trivial basis and take the root's members; the kernels use theta, lam, eps, gap, rank, nu, zeta_nodes)
  if (comm.rank() == 0 or comm.size() == 1) g.bos.emplace(g.theta, p.lam_b, p.eps, p.bos_gap, -1.0, -1.0, p.nline_eff(), p.npole_eff());
  else g.bos.emplace(g.theta, p.lam_b, p.eps, p.bos_gap, -1.0, -1.0, 4L, 4L);
  if (comm.size() > 1) {
    long r = g.bos->rank;
    comm.broadcast_n(&r, 1, 0);
    g.bos->rank = r;
    detail::bcast_nda(comm, g.bos->nu, 0);
    detail::bcast_nda(comm, g.bos->zeta_nodes, 0);
    detail::bcast_nda(comm, g.bos->zeta_dense, 0);
  }
  g.t_basis = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  t0        = std::chrono::steady_clock::now();
  auto pr   = pole_ranges_t::from(poles);
  g.Emin    = pr.p_min + pr.h_min;
  g.Emax    = pr.p_max + pr.h_max;
  if (p.time_grid == "gl") {
    const double emin = poles.emin();
    const double pe = p.gl_per_efold > 0.0 ? p.gl_per_efold : 3.0 * p.flat_scale();
    g.pi_p.emplace(numerics::line_dlr::time_ray_t::for_spectrum(g.theta_t, emin, p.ray_decades, 1e-5, pe, p.gl_nn, sector_t::particle));
    g.pi_h.emplace(numerics::line_dlr::time_ray_t::for_spectrum(g.theta_t, emin, p.ray_decades, 1e-5, pe, p.gl_nn, sector_t::hole));
  } else {
    utils::check(p.time_grid == "id", "gw_line head_pass: time_grid must be \"id\" or \"gl\" (got \"{}\")", p.time_grid);
    numerics::line_dlr::time_id_opts_t opts;
    opts.pad        = p.time_pad;
    opts.oversample = p.time_oversample;
    // [.flat_scan] (lih222, KS G, Pi vs exact): the candidate-s density is what matters on flat rays (5 deg: ns 40 -> 1.2e-5,
    // 80 -> 2.2e-10, 160 -> 9e-14 at oversample 1.25; the E-grid density 120 -> 480 changes nothing): ns x sqrt(flat_scale)
    opts.nE_per_efold = p.id_nE_per_efold > 0.0 ? p.id_nE_per_efold : opts.nE_per_efold;
    opts.ns_per_efold = p.id_ns_per_efold > 0.0 ? p.id_ns_per_efold : opts.ns_per_efold * std::sqrt(p.flat_scale());
    if (p.id_smax_fac > 0.0) opts.smax_fac = p.id_smax_fac;
    numerics::line_dlr::time_id_t idp, idh;
    if (comm.rank() == 0) idp = numerics::line_dlr::time_id_t(g.theta_t, sector_t::particle, g.Emin, g.Emax, p.time_eps, opts);
    if (comm.rank() == 1 % comm.size())
      idh = numerics::line_dlr::time_id_t(g.theta_t, sector_t::hole, g.Emin, g.Emax, p.time_eps, opts);
    if (comm.rank() != 0) idp.opts = opts;
    if (comm.rank() != 1 % comm.size()) idh.opts = opts;
    detail::bcast_time_id(comm, idp, 0);
    detail::bcast_time_id(comm, idh, 1 % comm.size());
    g.pi_p.emplace(idp);
    g.pi_h.emplace(idh);
  }
  g.t_grid = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return g;
}

struct head_pass_out_t {
  double theta_deg = 0.0;
  long rank = 0, nz = 0, nt_p = 0, nt_h = 0, ngroups = 0, group_size = 0;
  nda::array<ComplexType, 1> zeta;     ///< (nz) bosonic nodes
  nda::array<double, 1> nu;            ///< (rank) bosonic poles
  nda::array<ComplexType, 2> h_nodes;  ///< (nq, nz) eps^-1_00(q, zeta_i) - 1 (Gamma: 0)
  double t_basis = 0.0, t_grid = 0.0, t_Z = 0.0, t_Pi = 0.0, t_W = 0.0, t_head = 0.0, t_total = 0.0;
  std::string time_grid;
};

/**
 * The pass (see the file header). prop: the run's propagator (its poles are set by polarization); hb: head vectors of the
 * run's aux grid. Collective.
 */
template <MEMORY_SPACE MEM>
head_pass_out_t head_pass(methods::thc_reader_t &thc, mf::MF &mf, utils::mpi_context_t<boost::mpi3::communicator> &mpi,
                          aux_grid_t const &grid, propagator_t<MEM> &prop, pole_data_t const &poles, head_basis_t const &hb,
                          head_pass_params_t const &p) {
  using arr4_t   = memory::array<MEM, ComplexType, 4>;
  auto &comm     = mpi.comm;
  const auto t00 = std::chrono::steady_clock::now();
  auto secs      = [](auto a) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count(); };
  head_pass_out_t out;
  out.theta_deg = p.theta_deg;
  out.time_grid = p.time_grid;
  auto g        = head_pass_grid(poles, p, comm);
  auto const &bos = *g.bos;
  out.rank = bos.rank;
  out.nz   = bos.zeta_nodes.size();
  out.zeta = bos.zeta_nodes;
  out.nu   = bos.nu;
  out.nt_p = g.pi_p->size();
  out.nt_h = g.pi_h->size();
  out.t_basis = g.t_basis;
  out.t_grid  = g.t_grid;
  const long nq = mf.nqpts(), Np = thc.Np(), nz = out.nz;
  // q groups from the budget: Pi group + ~0.8 of it in the W stage
  long gs = p.qgroup;
  if (gs <= 0) {
    const double per_q = 1.8 * double(nz) * 16.0 * double(grid.max_block_size());
    gs = std::clamp(long(p.mem_gb * 1073741824.0 / per_q), 1L, nq);
    gs = comm.all_reduce_value(gs, boost::mpi3::min<>{});
  }
  const q_groups_t qg(nq, gs, qminus_list(mf));
  out.ngroups    = qg.n;
  out.group_size = qg.max_size();
  app_log(2, "  head pass theta {} deg: bosonic rank {} ({} nodes, nu [{:.4f}, {:.3f}] Ha; basis {:.2f} s), Pi time nodes ({}) {} + {} "
             "(|E| [{:.4f}, {:.3f}] Ha; {:.2f} s), {} q group(s) of <= {}",
          p.theta_deg, out.rank, nz, bos.nu(0), bos.nu(out.rank - 1), g.t_basis, p.time_grid, out.nt_p, out.nt_h, g.Emin, g.Emax,
          g.t_grid, qg.n, qg.max_size());
  utils::TimerManager Timer;
  auto t0 = std::chrono::steady_clock::now();
  coulomb_blocks_t<MEM> Zb(thc, grid, qg.dyson_q_list(comm.size(), comm.rank(), nz, Np), Timer);
  out.t_Z = secs(t0);
  nda::array<ComplexType, 2> Hn(nq, nz);
  Hn() = ComplexType(0.0);
  arr4_t Pi, w, Wn;
  for (long G = 0; G < qg.n; ++G) {
    t0 = std::chrono::steady_clock::now();
    polarization<MEM>(prop, poles, mf, grid, bos.zeta_nodes, *g.pi_p, *g.pi_h, p.t_chunk, Pi, Timer,
                      numerics::line_dlr::sector_t::both, qg.rows(G));
    out.t_Pi += secs(t0);
    t0 = std::chrono::steady_clock::now();
    screened_interaction<MEM>(Pi, Zb, bos, grid, mpi, w, Timer, &Wn, qg.rows(G), false);
    out.t_W += secs(t0);
    t0 = std::chrono::steady_clock::now();
    head_nodes_partial<MEM>(Wn, qg.rows(G), hb, Hn);
    Wn = arr4_t{};
    w  = arr4_t{};
    out.t_head += secs(t0);
  }
  t0 = std::chrono::steady_clock::now();
  head_reduce(comm, {&Hn});
  out.t_head += secs(t0);
  out.h_nodes = std::move(Hn);
  out.t_total = secs(t00);
  app_log(2, "  head pass theta {} deg: {:.2f} s (basis {:.2f}, time grid {:.2f}, Z {:.2f}, Pi {:.2f}, W {:.2f}, head {:.2f})",
          p.theta_deg, out.t_total, out.t_basis, out.t_grid, out.t_Z, out.t_Pi, out.t_W, out.t_head);
  return out;
}

} // namespace methods::gw_line

#endif
