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

#ifndef COQUI_METHODS_GW_LINE_TIME_GRIDS_HPP
#define COQUI_METHODS_GW_LINE_TIME_GRIDS_HPP

/**
 * The four compressed time grids of one line-GW iteration (plan S7b; notes section 4.3; numerics/line_dlr/time_id.hpp).
 *
 * The kernels (polarization, self_energy) take the type-erased view `time_nodes_t`; this file builds, per iteration, the
 * time-node ID of every ray product from the CURRENT poles. Each product is a sum of exponentials e^{-i E t} whose summed
 * energies E are bounded by the pole ranges (|e| per sector over all k, see pole_ranges_t):
 *   Pi particle ray  conj(G~^<(k, conj t)) o G~^>(k-q, t):  E = e_a + |e_i|   in [e^>_min + |e|^<_min, e^>_max + |e|^<_max]
 *   Pi hole ray      conj(G~^>(k, conj t)) o G~^<(k-q, t):  E = -(e_a + |e_i|), same |E| range
 *   Sigma particle   G~^>(k-q, t) o W^>(q, t):              E = e_m + nu_j     in [e^>_min + nu_min, e^>_max + nu_max]
 *   Sigma hole       G~^<(k-q, t) o W^<(q, t):              E = -(|e_m| + nu_j) in [|e|^<_min + nu_min, |e|^<_max + nu_max]
 * with nu_j the poles of the CURRENT bosonic basis. time_id_t constrains its least-squares transform on the PADDED range
 * [Emin / pad, pad Emax] (opts.pad; the transform is unconstrained outside it, S7a), so pad > 1 is the safety margin for
 * poles that move between the construction and the use (none within an iteration: the grids are rebuilt every iteration).
 *
 * S7e: grid i (0..3) is built by ONE rank, i mod np, and broadcast from it (before: all four on every rank, then the
 * root's broadcast; the four constructions now run concurrently on four ranks, ~4x less wall time). All ranks hold
 * bitwise identical nodes and LS factors (the node COUNT enters the collective of self_energy).
 */

#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>

#include "configuration.hpp"
#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "numerics/line_dlr/time_id.hpp"
#include "methods/GW_line/line_state.hpp"

namespace methods::gw_line {

/// |e| ranges of the poles per sector over all k (mu-relative energies).
struct pole_ranges_t {
  double p_min = 0.0, p_max = 0.0;   ///< particle poles e > 0
  double h_min = 0.0, h_max = 0.0;   ///< hole poles |e|, e < 0

  static pole_ranges_t from(pole_data_t const &pd) {
    pole_ranges_t r;
    r.p_min = r.h_min = std::numeric_limits<double>::max();
    r.p_max = r.h_max = 0.0;
    for (long ik = 0; ik < pd.nk; ++ik) {
      for (long m = 0; m < pd.part[ik].size(); ++m) {
        r.p_min = std::min(r.p_min, pd.part[ik].e(m));
        r.p_max = std::max(r.p_max, pd.part[ik].e(m));
      }
      for (long m = 0; m < pd.hole[ik].size(); ++m) {
        r.h_min = std::min(r.h_min, -pd.hole[ik].e(m));
        r.h_max = std::max(r.h_max, -pd.hole[ik].e(m));
      }
    }
    utils::check(r.p_max > 0.0 and r.h_max > 0.0, "pole_ranges_t: a sector is empty at every k");
    return r;
  }
};

/// Diagnostics of one grid.
struct time_grid_info_t {
  std::string name;
  double Emin = 0.0, Emax = 0.0;   ///< nominal |E| range (before the pad)
  long rank = 0, size = 0;         ///< eps-rank and number of nodes
  double ls_residual = 0.0;        ///< max relative weighted LS residual over the target points
  double maxF = 0.0;               ///< max |F(zeta, t)| over the target points
  double time = 0.0;               ///< construction + diagnostics (s)
};

namespace detail {

template <typename T, int R> void bcast_nda(boost::mpi3::communicator &comm, nda::array<T, R> &A, int root = 0) {
  std::array<long, R> shp{};
  if (comm.rank() == root) shp = A.shape();
  comm.broadcast_n(shp.data(), R, root);
  if (comm.rank() != root and A.shape() != shp) A.resize(shp);
  if (A.size() > 0) comm.broadcast_n(A.data(), A.size(), root);
}

/// the builder's (root's) time_id_t -> all ranks: every member (scalars, nodes, LS factors); opts are input (identical)
inline void bcast_time_id(boost::mpi3::communicator &comm, numerics::line_dlr::time_id_t &g, int root = 0) {
  if (comm.size() == 1) return;
  std::array<long, 4> n = {g.rank, g.ls_rank, g.n_cand, g.sector == numerics::line_dlr::sector_t::particle ? 0L : 1L};
  comm.broadcast_n(n.data(), 4, root);
  g.rank    = n[0];
  g.ls_rank = n[1];
  g.n_cand  = n[2];
  g.sector  = n[3] == 0 ? numerics::line_dlr::sector_t::particle : numerics::line_dlr::sector_t::hole;
  std::array<double, 6> x = {g.theta_t, g.Emin, g.Emax, g.eps, g.phase.real(), g.phase.imag()};
  comm.broadcast_n(x.data(), 6, root);
  g.theta_t = x[0];
  g.Emin    = x[1];
  g.Emax    = x[2];
  g.eps     = x[3];
  g.phase   = ComplexType(x[4], x[5]);
  bcast_nda(comm, g.s, root);
  bcast_nda(comm, g.t, root);
  bcast_nda(comm, g.rdiag, root);
  bcast_nda(comm, g.E, root);
  bcast_nda(comm, g.w, root);
  bcast_nda(comm, g.Uc, root);
  bcast_nda(comm, g.Vs, root);
  bcast_nda(comm, g.sv, root);
}

} // namespace detail

/// The four ID grids of one iteration (Pi particle/hole, Sigma particle/hole).
struct line_time_grids_t {
  numerics::line_dlr::time_id_t pi_p, pi_h, sig_p, sig_h;
  std::array<time_grid_info_t, 4> info;
  pole_ranges_t pr;
  double nu_min = 0.0, nu_max = 0.0;

  line_time_grids_t() = default;

  /**
   * poles: current poles; nu: bosonic basis poles (> 0); theta_t: ray angle; eps, opts: time_id_t tolerance and knobs
   * (opts.pad = safety margin); zeta_b / zeta_f: target points of the Pi / Sigma transforms (only for the diagnostics).
   * Collective over comm.
   */
  line_time_grids_t(pole_data_t const &poles, nda::array<double, 1> const &nu, double theta_t, double eps,
                    numerics::line_dlr::time_id_opts_t const &opts, nda::array<ComplexType, 1> const &zeta_b,
                    nda::array<ComplexType, 1> const &zeta_f, boost::mpi3::communicator &comm) {
    using numerics::line_dlr::time_id_t;
    using numerics::line_dlr::sector_t;
    utils::check(nu.size() > 0, "line_time_grids_t: empty bosonic basis");
    pr     = pole_ranges_t::from(poles);
    nu_min = nu(0);
    nu_max = nu(0);
    for (long j = 0; j < nu.size(); ++j) {
      nu_min = std::min(nu_min, nu(j));
      nu_max = std::max(nu_max, nu(j));
    }
    utils::check(nu_min > 0.0, "line_time_grids_t: bosonic poles must be > 0 (nu_min = {})", nu_min);
    const double pi_lo = pr.p_min + pr.h_min, pi_hi = pr.p_max + pr.h_max;
    const double sp_lo = pr.p_min + nu_min, sp_hi = pr.p_max + nu_max;
    const double sh_lo = pr.h_min + nu_min, sh_hi = pr.h_max + nu_max;
    utils::check(pi_lo > 0.0 and sp_lo > 0.0 and sh_lo > 0.0,
                 "line_time_grids_t: summed-energy ranges must start above 0 (Pi {}, Sigma^> {}, Sigma^< {})", pi_lo, sp_lo,
                 sh_lo);

    struct spec_t {
      char const *nm;
      sector_t s;
      double lo, hi;
      nda::array<ComplexType, 1> const *z;
      time_id_t *g;
    };
    const spec_t spec[4] = {{"Pi^>", sector_t::particle, pi_lo, pi_hi, &zeta_b, &pi_p},
                            {"Pi^<", sector_t::hole, pi_lo, pi_hi, &zeta_b, &pi_h},
                            {"Sigma^>", sector_t::particle, sp_lo, sp_hi, &zeta_f, &sig_p},
                            {"Sigma^<", sector_t::hole, sh_lo, sh_hi, &zeta_f, &sig_h}};
    // 1. construction + diagnostics on the builder rank i mod np (concurrent on >= 4 ranks)
    std::array<std::array<double, 3>, 4> diag{};   // LS residual, max|F|, build time
    for (int i = 0; i < 4; ++i) {
      if (comm.rank() != i % comm.size()) continue;
      const auto t0 = std::chrono::steady_clock::now();
      *spec[i].g    = time_id_t(theta_t, spec[i].s, spec[i].lo, spec[i].hi, eps, opts);
      double res = 0.0, fmax = 0.0;
      auto F = spec[i].g->transform_matrix(*spec[i].z, &res);
      for (auto const &v : F) fmax = std::max(fmax, std::abs(v));
      diag[i] = {res, fmax, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()};
    }
    // 2. builder -> all ranks
    for (int i = 0; i < 4; ++i) {
      const int root = int(i % comm.size());
      if (comm.rank() != root) spec[i].g->opts = opts;
      detail::bcast_time_id(comm, *spec[i].g, root);
      if (comm.size() > 1) comm.broadcast_n(diag[i].data(), 3, root);
      info[i] = time_grid_info_t{spec[i].nm, spec[i].lo, spec[i].hi, spec[i].g->rank, spec[i].g->size(), diag[i][0], diag[i][1],
                                 diag[i][2]};
    }
  }

  void log(int level = 2) const {
    app_log(level, "  time grids (ID, eps {:.1e}, pad {}, oversample {}): poles e^> [{:.4f}, {:.4f}], |e^<| [{:.4f}, {:.4f}], "
                   "nu [{:.4f}, {:.4f}] Ha",
            pi_p.eps, pi_p.opts.pad, pi_p.opts.oversample, pr.p_min, pr.p_max, pr.h_min, pr.h_max, nu_min, nu_max);
    for (auto const &g : info)
      app_log(level, "    {:<8s} |E| in [{:.4f}, {:.4f}] Ha: rank {:4d}, nodes {:4d}, LS residual {:.1e}, max|F| {:7.1f}, {:.2f} s",
              g.name, g.Emin, g.Emax, g.rank, g.size, g.ls_residual, g.maxF, g.time);
  }
};

} // namespace methods::gw_line

#endif
