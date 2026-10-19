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

#ifndef COQUI_VERTEX_VIEW_HPP
#define COQUI_VERTEX_VIEW_HPP

/**
 * Restricted k views of the ladder inputs.
 *
 * The pair-space ladder (vertex_dynbse.icc, pair_space_ladder_dyn) sums its internal momenta over the points it is
 * given. A k_view selects a SUBSET of the mean field's k mesh as those internal points -- a coarse sublattice of a
 * fine mesh -- together with a list of external transfers that need not belong to the sublattice:
 *
 *   internal points   k_int      (nk_int points of the full mesh, closed under their own differences)
 *   external q        q_ext      (any transfers of the full mesh)
 *   partner points    k + q_ext  (appended after the internal points when they are not internal themselves)
 *
 * Every rung transfer k - k' of two internal points is a transfer of the sublattice, so the rungs read the screened
 * interaction at sublattice transfers only: the W tables of a view hold just those rows.
 *
 * view_inputs holds the gathered arrays in the layout the driver expects (dyn_opts::qx_of / nk_int).
 */

#include <array>
#include <cmath>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"
#include "numerics/nda_functions.hpp"
#include "methods/vertex/vertex_sym.hpp"

namespace methods {
namespace solvers {
namespace vertex_view {

  struct k_view {
    long nk_int = 0;                 // internal points = the first nk_int entries of kpts
    nda::array<long, 1> kpts;        // (nk_arr) full-mesh k index of every point of the view
    std::vector<long> q_ext;         // full-mesh transfer index of every external transfer
    nda::array<long, 2> kpq;         // (n_ext, nk_int): view index of k + q_ext
    nda::array<long, 2> qx_full;     // (nk_int, nk_int): full-mesh transfer index of k - k'
    std::vector<long> wrows;         // row of the full W tables (IBZ transfer index) of every row of the view's W tables
    nda::array<long, 1> wrow_of;     // (rows of the full W tables) -> row of the view's W tables, or -1

    long nk_arr() const { return kpts.shape(0); }
    long n_ext() const { return long(q_ext.size()); }
    long n_wrows() const { return long(wrows.size()); }
  };

  /**
   * The points of the full mesh that belong to the Gamma-centred sublattice with n_sub = (n1, n2, n3) points per
   * reciprocal axis: k (crystal coordinates) with k_a n_a integer. Checked to hold exactly n1 n2 n3 points
   * (the mesh must be commensurate with the sublattice). kcrys: (nk, 3) crystal coordinates of the full mesh.
   */
  inline std::vector<long> sublattice(nda::MemoryArrayOfRank<2> auto const &kcrys, std::array<long, 3> const &n_sub) {
    std::vector<long> ks;
    for (long k = 0; k < kcrys.shape(0); ++k) {
      bool on = true;
      for (int a = 0; a < 3; ++a) {
        const double x = kcrys(k, a) * double(n_sub[size_t(a)]);
        on = on and (std::abs(x - std::round(x)) < 1e-6);
      }
      if (on) ks.push_back(k);
    }
    utils::check(long(ks.size()) == n_sub[0] * n_sub[1] * n_sub[2],
                 "vertex_view::sublattice: the k mesh holds {} points of the {} x {} x {} sublattice (expected {}): the mesh is not "
                 "commensurate with it.", ks.size(), n_sub[0], n_sub[1], n_sub[2], n_sub[0] * n_sub[1] * n_sub[2]);
    return ks;
  }

  /**
   * Build a view. kmq_of(q, k) / kpq_of(q, k) return the full-mesh index of k - q / k + q (q a full-mesh transfer index,
   * nq_full of them); wrow_full(q) returns the row of the full W tables that stores transfer q (the IBZ representative
   * on a symmetric mesh, q itself otherwise), nw_full the number of such rows.
   */
  template<class KMQ, class KPQ, class WROW>
  inline k_view make_view(long nk_full, long nq_full, long nw_full, std::vector<long> const &k_int, std::vector<long> const &q_ext,
                          KMQ &&kmq_of, KPQ &&kpq_of, WROW &&wrow_full) {
    k_view v;
    v.nk_int = long(k_int.size());
    v.q_ext = q_ext;
    utils::check(v.nk_int > 0 and not q_ext.empty(), "vertex_view::make_view: empty internal point or external transfer list.");
    std::vector<long> pos(static_cast<size_t>(nk_full), -1l);
    std::vector<long> pts;
    for (long k : k_int) {
      utils::check(k >= 0 and k < nk_full and pos[size_t(k)] < 0, "vertex_view::make_view: invalid or repeated internal point {}.", k);
      pos[size_t(k)] = long(pts.size());
      pts.push_back(k);
    }
    // partners k + q_ext
    v.kpq = nda::array<long, 2>(v.n_ext(), v.nk_int);
    for (long e = 0; e < v.n_ext(); ++e) {
      utils::check(q_ext[size_t(e)] >= 0 and q_ext[size_t(e)] < nq_full, "vertex_view::make_view: external transfer {} out of range.",
                   q_ext[size_t(e)]);
      for (long i = 0; i < v.nk_int; ++i) {
        const long kq = kpq_of(q_ext[size_t(e)], k_int[size_t(i)]);
        if (pos[size_t(kq)] < 0) { pos[size_t(kq)] = long(pts.size()); pts.push_back(kq); }
        v.kpq(e, i) = pos[size_t(kq)];
      }
    }
    v.kpts = nda::array<long, 1>(long(pts.size()));
    for (long i = 0; i < long(pts.size()); ++i) v.kpts(i) = pts[size_t(i)];
    // rung transfers of the internal pairs
    v.qx_full = nda::array<long, 2>(v.nk_int, v.nk_int);
    v.qx_full() = -1;
    for (long q = 0; q < nq_full; ++q)
      for (long i = 0; i < v.nk_int; ++i) {
        const long kp = kmq_of(q, k_int[size_t(i)]);
        const long j = pos[size_t(kp)];
        if (j >= 0 and j < v.nk_int) v.qx_full(i, j) = q;
      }
    v.wrow_of = nda::array<long, 1>(nw_full);
    v.wrow_of() = -1;
    for (long i = 0; i < v.nk_int; ++i)
      for (long j = 0; j < v.nk_int; ++j) {
        utils::check(v.qx_full(i, j) >= 0, "vertex_view::make_view: the internal points {} and {} are not connected by a transfer of "
                     "the mesh.", k_int[size_t(i)], k_int[size_t(j)]);
        const long w = wrow_full(v.qx_full(i, j));
        utils::check(w >= 0 and w < nw_full, "vertex_view::make_view: transfer {} has no row in the W tables.", v.qx_full(i, j));
        if (v.wrow_of(w) < 0) { v.wrow_of(w) = long(v.wrows.size()); v.wrows.push_back(w); }
      }
    return v;
  }

  /** A view on a mean field's mesh: the internal points k_int and the external transfers q_ext (full-mesh indices). */
  inline k_view make_view(auto const &MF, std::vector<long> const &k_int, std::vector<long> const &q_ext) {
    auto qmin = MF.qminus();
    return make_view(MF.nkpts(), MF.nqpts(), MF.nqpts_ibz(), k_int, q_ext,
                     [&](long q, long k) { return long(MF.qk_to_k2(int(q), int(k))); },
                     [&](long q, long k) { return long(MF.qk_to_k2(int(qmin(q)), int(k))); },
                     [&](long q) { return long(MF.qp_to_ibz(int(q))); });
  }

  /** The gathered ladder inputs of a view, in the driver's restricted-view layout. */
  struct view_inputs {
    nda::array<ComplexType, 5> G_CC;     // (nt, ns, nk_arr, nc, nc)
    nda::array<ComplexType, 4> Xb;       // (ns, nk_arr, Nm, nc)
    nda::array<ComplexType, 3> W0b;      // (n_wrows, Nm, Nm)
    nda::array<ComplexType, 4> Wbh;      // (n_wrows, nw_half, Nm, Nm)
    nda::array<long, 2> qx;              // (nk_int, nk_int): the driver's transfer table (see below)
    bool sym = false;
    vertex_sym::sym_ctx symc;            // symmetric mesh: the view's symmetry context (k axis = the view's points)
    nda::array<ComplexType, 5> Xhat;     // storage of symc.Xhat: (ns, nsym, nk_arr, Nm, nc)
    vertex_sym::sym_ctx const *symc_ptr() const { return sym ? &symc : nullptr; }
  };

  /**
   * Gather the inputs of a view from the full-mesh ones. Without symmetry the driver's transfer table names rows of the
   * view's W tables directly. With symmetry it keeps the full-mesh transfer indices and the view's symmetry context maps
   * them (q_isym, q_trev as on the full mesh; q_star to the rows of the view's W tables).
   * head_scale multiplies NOTHING here: the Gamma-row head weight of a view is set by the caller on the W tables.
   */
  inline void gather(k_view const &v, nda::MemoryArrayOfRank<5> auto const &G_CC, nda::MemoryArrayOfRank<4> auto const &Xb,
                     nda::MemoryArrayOfRank<3> auto const &W0b, nda::MemoryArrayOfRank<4> auto const &Wbh,
                     vertex_sym::sym_ctx const *symc, view_inputs &o, nda::array<long, 1> const *cache_row_of = nullptr) {
    decltype(nda::range::all) all;
    const long nka = v.nk_arr(), nw = v.n_wrows();
    const long nt = G_CC.shape(0), ns = G_CC.shape(1), nc = G_CC.shape(3), Nm = Xb.shape(2);
    // cache_row_of: the dynamic table Wbh holds a subset of the rows (full row -> its row, or -1); null = every row
    utils::check(v.wrow_of.shape(0) == W0b.shape(0) and (cache_row_of != nullptr or Wbh.shape(0) == W0b.shape(0)),
                 "vertex_view::gather: the view was built for {} W rows, the tables carry {} / {}.", v.wrow_of.shape(0), W0b.shape(0),
                 Wbh.shape(0));
    o.G_CC = nda::array<ComplexType, 5>(nt, ns, nka, nc, nc);
    o.Xb = nda::array<ComplexType, 4>(ns, nka, Nm, nc);
    for (long i = 0; i < nka; ++i) {
      o.G_CC(all, all, i, all, all) = G_CC(all, all, v.kpts(i), all, all);
      o.Xb(all, i, all, all) = Xb(all, v.kpts(i), all, all);
    }
    o.W0b = nda::array<ComplexType, 3>(nw, Nm, Nm);
    o.Wbh = nda::array<ComplexType, 4>(nw, Wbh.shape(1), Nm, Nm);
    for (long r = 0; r < nw; ++r) {
      o.W0b(r, all, all) = W0b(v.wrows[size_t(r)], all, all);
      const long cr = (cache_row_of != nullptr) ? (*cache_row_of)(v.wrows[size_t(r)]) : v.wrows[size_t(r)];
      utils::check(cr >= 0 and cr < Wbh.shape(0), "vertex_view::gather: transfer row {} is not held by the W-bar cache.",
                   v.wrows[size_t(r)]);
      o.Wbh(r, all, all, all) = Wbh(cr, all, all, all);
    }
    o.sym = (symc != nullptr and symc->active);
    o.qx = nda::array<long, 2>(v.nk_int, v.nk_int);
    if (not o.sym) {
      for (long i = 0; i < v.nk_int; ++i)
        for (long j = 0; j < v.nk_int; ++j) o.qx(i, j) = v.wrow_of(v.qx_full(i, j));
      return;
    }
    o.qx() = v.qx_full;
    auto &c = o.symc;
    c.active = true;
    c.ns = symc->ns; c.nsym = symc->nsym; c.nc = symc->nc; c.naux = symc->naux;
    c.nk_full = nka; c.nk_ibz = symc->nk_ibz;
    c.nq_full = symc->nq_full; c.nq_ibz = nw;
    c.q_isym = symc->q_isym;
    c.q_trev = symc->q_trev;
    c.q_star = nda::array<long, 1>(symc->nq_full);
    for (long q = 0; q < symc->nq_full; ++q) c.q_star(q) = v.wrow_of(symc->q_star(q));
    c.leak_max = symc->leak_max; c.leak_mean = symc->leak_mean; c.d_unitarity_max = symc->d_unitarity_max;
    o.Xhat = nda::array<ComplexType, 5>(symc->ns, symc->nsym, nka, symc->naux, symc->nc);
    for (long i = 0; i < nka; ++i) o.Xhat(all, all, i, all, all) = symc->Xhat(all, all, v.kpts(i), all, all);
    c.Xhat.rebind(o.Xhat());
  }

} // vertex_view
} // solvers
} // methods

#endif // COQUI_VERTEX_VIEW_HPP
