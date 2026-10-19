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

#ifndef COQUI_VERTEX_SC_HPP
#define COQUI_VERTEX_SC_HPP

/**
 * In-run self-consistency of the polarization vertex on a coarse sublattice of the k mesh.
 *
 * The scGW loop runs on the (fine) mesh of the mean field. Every `period` iterations the vertex polarization
 * Pi-bar(q, i nu) in the secondary frame is recomputed from the CURRENT G and W with the internal momentum sums of
 * the ladder restricted to a Gamma-centred sublattice (vertex_view.hpp), mixed with the previous one, and injected
 * in the W Dyson equation of the following iterations. Two evaluations of the external transfers:
 *
 *   "interp" : the ladder is solved at the transfers of the sublattice and Fourier-interpolated to the transfers of
 *              the mesh (real-space route: Pi(R) on the Wigner-Seitz images of the sublattice's supercell).
 *   "direct" : the ladder is solved at every (IBZ) transfer q of the mesh, pairs (k, k + q) with k on the sublattice;
 *              one view per class of q modulo the sublattice's reciprocal vectors.
 *
 * This header holds the mesh bookkeeping, the interpolation kernel and the store; the ladder solve is supplied by the
 * caller as a function of a view.
 */

#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <map>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "h5/h5.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "numerics/shared_array/nda.hpp"
#include "methods/vertex/vertex_view.hpp"

namespace methods {
namespace solvers {
namespace vertex_sc {

  struct config {
    std::string mode = "none";            // "none" | "interp" | "direct" | "hybrid" (the input default is "hybrid" when a
                                          // sublattice is given; on the identity sublattice it runs as "direct")
    std::array<long, 3> mesh = {0, 0, 0}; // the sublattice (points per reciprocal axis)
    long period = 1;                      // refresh every `period` updates ...
    long start = 1;                       // ... from update `start` on (1 = the first)
    double mix = 0.5;                     // Pi <- (1 - mix) Pi + mix Pi_new (the first refresh takes Pi_new)
    std::string head = "coarse";          // weight of the q -> 0 head of the rung on the sublattice:
                                          //   "coarse" = the Madelung weight of the sublattice's supercell,
                                          //   "fine" = the mesh's own weight (unchanged), "none" = no head
    std::string col = "gam1";             // the ladder column that is injected ("gam1" | "dyn" | "static")
    std::string restart_file = "";        // a <prefix>.pol_sc.h5 of a previous run to start from
    long batch = 8;                       // external transfers per ladder pass (bounds the memory of a pass)
    bool trev = true;                     // "interp": solve one of every (q, -q) pair of sublattice transfers
    std::string direct_select = "radius"; // "hybrid": which fine IBZ transfers are solved directly --
    double direct_radius = 1.0;           //   "radius": |q| <= direct_radius x (the coarse reciprocal spacing, min_i |b_i| / n_i);
    long direct_shells = 1;               //   "shells": Gamma and the direct_shells smallest non-zero |q| shells of the fine IBZ set;
                                          //   "count":  the direct_count fine IBZ transfers of smallest |q| (whole shells, so the
    long direct_count = 0;                //             set may be a little larger). |q| = the minimum-image Cartesian norm.
    std::string sigma = "follow";         // the Sigma side's scheme: "follow" (interp / hybrid -> S1, direct -> S2) | "interp" | "direct"
    bool sigma_direct() const { return sigma == "direct" or (sigma == "follow" and mode == "direct"); }
    std::string unfold = "fit";           // "interp" on a symmetric mesh: "fit" = solve the IBZ sublattice transfers only and
                                          // unfold their star images in the aux frame (vertex_t::unfold_fit_matrix);
                                          // "solve" = solve every sublattice transfer (time reversal only)
    // nu-SAMPLED refreshes: nu_nodes = K > 0 solves the ladder on K half nodes chosen from the nu-modes of the STORED Pi-bar
    // (nu = 0 and the highest node forced, the rest the row pivots of its top-K modes) and reconstructs every node in that
    // basis (rank nu_rank, 0 = K; nu_mode "modes" | "regression", nu_sampling.hpp). Refreshes with no stored Pi-bar and every
    // nu_ref_period-th refresh (0 = never again) solve all nodes, which renews the basis.
    long nu_nodes = 0;
    long nu_rank = 0;
    std::string nu_mode = "modes";
    long nu_ref_period = 0;
    bool enabled() const { return mode != "none"; }
  };

  /** crystal coordinates of a transfer folded to [-1/2, 1/2) */
  inline double fold(double x) { x -= std::floor(x); return (x >= 0.5 - 1e-9) ? x - 1.0 : x; }

  /** the reciprocal lattice vectors b_i (rows, 1/bohr) from the lattice vectors (rows, bohr): b = 2 pi (a^-1)^T */
  inline std::array<std::array<double, 3>, 3> reciprocal_vectors(nda::MemoryArrayOfRank<2> auto const &lattv) {
    double A[3][3];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) A[i][j] = lattv(i, j);
    const double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                     + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    utils::check(std::abs(det) > 1e-14, "vertex_sc::reciprocal_vectors: singular lattice.");
    std::array<std::array<double, 3>, 3> b{};
    for (int i = 0; i < 3; ++i)                       // (A^-1)(j, i) = C_ij / det (adjugate); b_i(j) = 2 pi (A^-1)(j, i)
      for (int j = 0; j < 3; ++j) {
        const int i1 = (i + 1) % 3, i2 = (i + 2) % 3, j1 = (j + 1) % 3, j2 = (j + 2) % 3;
        b[size_t(i)][size_t(j)] = 2.0 * M_PI * (A[i1][j1] * A[i2][j2] - A[i1][j2] * A[i2][j1]) / det;
      }
    return b;
  }

  /** the minimum-image Cartesian norm of a transfer given in crystal coordinates (the shortest |q + G| over the 27 neighbours
   *  of the folded point; exact for every cell shape that is not extremely skewed) */
  inline double min_image_norm(std::array<std::array<double, 3>, 3> const &b, double q0, double q1, double q2) {
    const double f[3] = {fold(q0), fold(q1), fold(q2)};
    double best = 1e300;
    for (int g0 = -1; g0 <= 1; ++g0)
      for (int g1 = -1; g1 <= 1; ++g1)
        for (int g2 = -1; g2 <= 1; ++g2) {
          const double c[3] = {f[0] + g0, f[1] + g1, f[2] + g2};
          double n2 = 0.0;
          for (int j = 0; j < 3; ++j) {
            double v = 0.0;
            for (int i = 0; i < 3; ++i) v += c[i] * b[size_t(i)][size_t(j)];
            n2 += v * v;
          }
          best = std::min(best, n2);
        }
    return std::sqrt(best);
  }

  /**
   * "hybrid": the fine IBZ transfers that are solved directly (the others are interpolated from the sublattice set).
   * cfg.direct_select = "radius" (|q| <= direct_radius x the coarse spacing min_i |b_i| / n_i), "shells" (Gamma and the
   * direct_shells smallest non-zero |q| shells) or "count" (the direct_count smallest |q|, completed to whole shells).
   * Returns the flags; logs the choice.
   */
  inline std::vector<char> direct_set(config const &cfg, nda::MemoryArrayOfRank<2> auto const &lattv,
                                      nda::MemoryArrayOfRank<2> auto const &qcrys, long nq_ibz) {
    const auto b = reciprocal_vectors(lattv);
    double dq_c = 1e300;
    for (int i = 0; i < 3; ++i) {
      double n2 = 0.0;
      for (int j = 0; j < 3; ++j) n2 += b[size_t(i)][size_t(j)] * b[size_t(i)][size_t(j)];
      dq_c = std::min(dq_c, std::sqrt(n2) / double(cfg.mesh[size_t(i)]));
    }
    std::vector<double> qn(static_cast<size_t>(nq_ibz));
    for (long q = 0; q < nq_ibz; ++q) qn[size_t(q)] = min_image_norm(b, qcrys(q, 0), qcrys(q, 1), qcrys(q, 2));
    // the distinct |q| shells (ascending; two norms within 1e-8 x dq_c are one shell)
    std::vector<double> shells(qn);
    std::sort(shells.begin(), shells.end());
    shells.erase(std::unique(shells.begin(), shells.end(), [&](double x, double y) { return std::abs(x - y) <= 1e-8 * dq_c; }),
                 shells.end());
    auto shell_of = [&](double x) {
      for (size_t s = 0; s < shells.size(); ++s) if (std::abs(x - shells[s]) <= 1e-8 * dq_c) return long(s);
      utils::check(false, "vertex_sc::direct_set: no shell for |q| = {}.", x);
      return -1l;
    };
    double rmax = -1.0;
    if (cfg.direct_select == "radius") {
      rmax = cfg.direct_radius * dq_c;
    } else if (cfg.direct_select == "shells") {
      // shell 0 = Gamma (|q| = 0) when the IBZ set has it; the direct_shells shells after it
      const long s0 = (shells.empty() or shells[0] > 1e-8 * dq_c) ? -1 : 0;
      const long last = std::min(long(shells.size()) - 1, s0 + cfg.direct_shells);
      rmax = (last >= 0) ? shells[size_t(last)] : -1.0;
    } else if (cfg.direct_select == "count") {
      std::vector<double> srt(qn);
      std::sort(srt.begin(), srt.end());
      const long n = std::min(cfg.direct_count, nq_ibz);
      rmax = (n > 0) ? srt[size_t(n - 1)] : -1.0;    // completed to the whole shell by the <= test below
    } else {
      utils::check(false, "vertex_sc::direct_set: pol_vertex_sc_direct_select = \"{}\" is not valid.", cfg.direct_select);
    }
    std::vector<char> flags(static_cast<size_t>(nq_ibz), 0);
    long nd = 0, ns = 0;
    for (long q = 0; q < nq_ibz; ++q)
      if (rmax >= 0.0 and qn[size_t(q)] <= rmax * (1.0 + 1e-9) + 1e-12) { flags[size_t(q)] = 1; ++nd; ns = std::max(ns, shell_of(qn[size_t(q)]) + 1); }
    app_log(1, "  [vertex self-consistency] hybrid ({}): {} of the {} fine IBZ transfers ({} |q| shell(s), |q| <= {:.4f} 1/bohr = {:.3f} coarse "
               "spacing(s) of {:.4f} 1/bohr) are solved directly; the others are interpolated from the sublattice.",
            cfg.direct_select, nd, nq_ibz, ns, std::max(rmax, 0.0), std::max(rmax, 0.0) / dq_c, dq_c);
    return flags;
  }

  /**
   * The real-space interpolation kernel K (nf x nc):  Pi(q_f) = sum_c K(f, c) Pi(q_c),
   *   K(f, c) = (1 / N_c) sum_n sum_{R in images(n)} (1 / deg_n) exp(2 pi i (q_f - q_c) . R),
   * n over the N_c lattice vectors of the sublattice's supercell and images(n) the shortest of its periodic images
   * n + N m (the Wigner-Seitz cell of the supercell; degenerate images share the weight). K(f, c) = delta when q_f is a
   * transfer of the sublattice. lattv: rows = lattice vectors; q in crystal coordinates.
   */
  inline nda::array<ComplexType, 2> interpolation_kernel(nda::MemoryArrayOfRank<2> auto const &lattv, std::array<long, 3> const &mesh,
                                                         nda::MemoryArrayOfRank<2> auto const &qc, nda::MemoryArrayOfRank<2> auto const &qf) {
    const long Nc = mesh[0] * mesh[1] * mesh[2];
    utils::check(qc.shape(0) == Nc, "vertex_sc::interpolation_kernel: {} sublattice transfers for a {} x {} x {} sublattice.",
                 qc.shape(0), mesh[0], mesh[1], mesh[2]);
    std::vector<std::array<double, 4>> R;   // (R1, R2, R3, weight)
    for (long n0 = 0; n0 < mesh[0]; ++n0)
      for (long n1 = 0; n1 < mesh[1]; ++n1)
        for (long n2 = 0; n2 < mesh[2]; ++n2) {
          double dmin = 1e300;
          std::vector<std::array<long, 3>> best;
          for (long m0 = -2; m0 <= 2; ++m0)
            for (long m1 = -2; m1 <= 2; ++m1)
              for (long m2 = -2; m2 <= 2; ++m2) {
                const long r[3] = {n0 + m0 * mesh[0], n1 + m1 * mesh[1], n2 + m2 * mesh[2]};
                double d2 = 0.0;
                for (int c = 0; c < 3; ++c) {
                  double x = 0.0;
                  for (int a = 0; a < 3; ++a) x += double(r[a]) * lattv(a, c);
                  d2 += x * x;
                }
                if (d2 < dmin * (1.0 - 1e-8)) { dmin = d2; best.clear(); }
                if (d2 <= dmin * (1.0 + 1e-8)) best.push_back({r[0], r[1], r[2]});
              }
          for (auto const &b : best) R.push_back({double(b[0]), double(b[1]), double(b[2]), 1.0 / double(best.size())});
        }
    const long nf = qf.shape(0);
    nda::array<ComplexType, 2> K(nf, Nc);
    for (long f = 0; f < nf; ++f)
      for (long c = 0; c < Nc; ++c) {
        ComplexType s(0.0);
        for (auto const &r : R) {
          const double ph = 2.0 * M_PI * ((qf(f, 0) - qc(c, 0)) * r[0] + (qf(f, 1) - qc(c, 1)) * r[1] + (qf(f, 2) - qc(c, 2)) * r[2]);
          s += r[3] * ComplexType(std::cos(ph), std::sin(ph));
        }
        K(f, c) = s / double(Nc);
      }
    return K;
  }

  /** the transfers of the full mesh (rows of qcrys) that belong to the sublattice */
  inline std::vector<long> sublattice_transfers(nda::MemoryArrayOfRank<2> auto const &qcrys, std::array<long, 3> const &mesh) {
    return vertex_view::sublattice(qcrys, mesh);
  }

  /** the classes of the transfers `qs` (rows of qcrys) modulo the reciprocal vectors of the sublattice */
  inline std::vector<std::vector<long>> transfer_classes(nda::MemoryArrayOfRank<2> auto const &qcrys, std::vector<long> const &qs,
                                                         std::array<long, 3> const &mesh) {
    std::map<std::array<long, 3>, std::vector<long>> cls;
    for (long q : qs) {
      std::array<long, 3> key;
      for (int a = 0; a < 3; ++a) {
        const double x = qcrys(q, a) * double(mesh[size_t(a)]);
        double fr = x - std::floor(x + 1e-7);
        if (fr < 0.0) fr = 0.0;
        key[size_t(a)] = std::llround(fr * 1.0e6);
      }
      cls[key].push_back(q);
    }
    std::vector<std::vector<long>> out;
    for (auto &kv : cls) out.push_back(std::move(kv.second));
    return out;
  }

  /**
   * The store: the current (mixed) Pi-bar(nu_half, q, m, n) at the IBZ transfers of the mesh, ONE COPY PER NODE (a
   * node-shared window; every rank reads it through view()). A refresh accumulates the new solution in a second
   * node-shared array (begin / add / commit) and mixes it into the current one.
   */
  class store {
    using mpi_t = utils::mpi_context_t<mpi3::communicator, mpi3::shared_communicator>;
    using shm_t = math::shm::shared_array<nda::array_view<ComplexType, 4>>;
  public:
    config cfg;
    long n_refresh = 0;
    double last_change = -1.0;            // |Pi_new - Pi_old|_F / |Pi_new|_F of the last refresh
    double last_norm = 0.0;               // |Pi_new|_F of the last refresh
    double t_last = 0.0, t_total = 0.0;   // wall time of the refreshes (s)

    bool has() const { return n_refresh > 0 and _Pi != nullptr; }
    bool due(long update) const { return cfg.enabled() and update >= cfg.start and (update - cfg.start) % std::max(1l, cfg.period) == 0; }
    /** the current Pi-bar (nw_half, nq_ibz, Nm, Nm) */
    nda::array_view<ComplexType, 4> view() const {
      utils::check(_Pi != nullptr, "vertex_sc::store::view: no vertex polarization stored.");
      return _Pi->local();
    }
    double bytes() const { return _Pi ? 16.0 * double(_Pi->local().size()) : 0.0; }

    /** start a refresh: the accumulator of the new solution, zeroed */
    void begin(mpi_t &mpi, long nw, long nq, long Nm) {
      _New = std::make_shared<shm_t>(math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(mpi, std::array<long, 4>{nw, nq, Nm, Nm}));
      _New->win().fence();
      if (mpi.node_comm.root()) _New->local()() = ComplexType(0.0);
      _New->win().fence();
      _root = mpi.node_comm.root();
    }
    /** Pi_new(:, q, :, :) += w * P (or w * conj(P)); P (nw_half, Nm, Nm) is the same on every rank: the node root writes */
    void add(long q, ComplexType w, bool cj, nda::MemoryArrayOfRank<3> auto const &P) {
      if (not _root) return;
      auto N = _New->local();
      const long nw = N.shape(0), n2 = N.shape(2) * N.shape(3);
      utils::check(P.shape(0) == nw and P.shape(1) == N.shape(2) and P.shape(2) == N.shape(3), "vertex_sc::store::add: shape mismatch.");
      for (long j = 0; j < nw; ++j) {
        ComplexType *d = std::addressof(N(j, q, 0, 0));
        ComplexType const *sp = std::addressof(P(j, 0, 0));
        if (cj) for (long i = 0; i < n2; ++i) d[i] += w * std::conj(sp[i]);
        else    for (long i = 0; i < n2; ++i) d[i] += w * sp[i];
      }
    }
    /** end a refresh: mix the accumulated solution into the stored one; returns the relative change (-1 at the first) */
    double commit(mpi_t &mpi) {
      _New->win().fence();
      double num = 0.0, den = 0.0;
      const bool first = (_Pi == nullptr);
      if (first) {
        if (_root) for (auto const &x : _New->local()) den += std::norm(x);
        _Pi = std::move(_New);
      } else {
        utils::check(_Pi->local().shape() == _New->local().shape(), "vertex_sc::store::commit: the shape of the vertex polarization changed.");
        _Pi->win().fence();
        if (_root) {
          const double a = cfg.mix;
          auto O = _Pi->local();
          auto N = _New->local();
          ComplexType *po = O.data();
          ComplexType const *pn = N.data();
          for (long i = 0; i < long(O.size()); ++i) {
            num += std::norm(pn[i] - po[i]);
            den += std::norm(pn[i]);
            po[i] = (1.0 - a) * po[i] + a * pn[i];
          }
        }
        _Pi->win().fence();
        _New.reset();
      }
      mpi.node_comm.broadcast_n(&num, 1, 0);
      mpi.node_comm.broadcast_n(&den, 1, 0);
      last_norm = std::sqrt(den);
      last_change = first ? -1.0 : std::sqrt(num / std::max(den, 1e-300));
      ++n_refresh;
      return last_change;
    }

    /** checkpoint (called on the global root): everything needed to continue the mixing in a restarted run */
    void write(std::string const &fn, nda::MemoryArrayOfRank<2> auto const &q_ibz, nda::MemoryArrayOfRank<1> auto const &nu_half,
               double beta) const {
      const std::string tmp = fn + ".tmp";
      {
        h5::file f(tmp, 'w');
        h5::group g(f);
        h5::h5_write(g, "mode", cfg.mode);
        h5::h5_write(g, "col", cfg.col);
        h5::h5_write(g, "head", cfg.head);
        nda::h5_write(g, "mesh", nda::array<long, 1>{cfg.mesh[0], cfg.mesh[1], cfg.mesh[2]});
        h5::h5_write(g, "n_refresh", n_refresh);
        h5::h5_write(g, "beta", beta);
        nda::h5_write(g, "q", q_ibz);
        nda::h5_write(g, "nu_half", nu_half);
        nda::h5_write(g, "Pi", nda::array<ComplexType, 4>(_Pi->local()), false);
      }
      std::filesystem::rename(tmp, fn);
    }

    /** restart (called on every rank; the node roots read): a checkpoint written with the same mode, sublattice, column,
     *  head policy, frequency grid, transfers and secondary points */
    void read(mpi_t &mpi, std::string const &fn, nda::MemoryArrayOfRank<2> auto const &q_ibz, nda::MemoryArrayOfRank<1> auto const &nu_half,
              double beta, long Nm) {
      begin(mpi, nu_half.size(), q_ibz.shape(0), Nm);
      long nref = 0;
      if (_root) {
        h5::file f(fn, 'r');
        h5::group g(f);
        std::string mode, col, head;
        nda::array<long, 1> mesh, nu_in;
        nda::array<double, 2> q_in;
        double beta_in = 0.0;
        h5::h5_read(g, "mode", mode);
        h5::h5_read(g, "col", col);
        h5::h5_read(g, "head", head);
        nda::h5_read(g, "mesh", mesh);
        h5::h5_read(g, "beta", beta_in);
        nda::h5_read(g, "q", q_in);
        nda::h5_read(g, "nu_half", nu_in);
        utils::check(mode == cfg.mode and col == cfg.col and head == cfg.head and mesh(0) == cfg.mesh[0] and mesh(1) == cfg.mesh[1] and
                     mesh(2) == cfg.mesh[2],
                     "vertex_sc::store::read: {} was written with mode {}, column {}, head {}, sublattice {} x {} x {}; this run uses "
                     "{}, {}, {}, {} x {} x {}.", fn, mode, col, head, mesh(0), mesh(1), mesh(2), cfg.mode, cfg.col, cfg.head,
                     cfg.mesh[0], cfg.mesh[1], cfg.mesh[2]);
        utils::check(std::abs(beta_in - beta) < 1e-10 * std::abs(beta) and nu_in.size() == nu_half.size() and
                     q_in.shape(0) == q_ibz.shape(0),
                     "vertex_sc::store::read: {} has beta {}, {} half nodes, {} transfers; this run {}, {}, {}.", fn, beta_in,
                     nu_in.size(), q_in.shape(0), beta, nu_half.size(), q_ibz.shape(0));
        for (long j = 0; j < nu_half.size(); ++j)
          utils::check(nu_in(j) == nu_half(j), "vertex_sc::store::read: half node {} of {} is Matsubara index {}, {} here.", j, fn,
                       nu_in(j), nu_half(j));
        for (long q = 0; q < q_ibz.shape(0); ++q)
          for (int a = 0; a < 3; ++a)
            utils::check(std::abs(fold(q_in(q, a) - q_ibz(q, a))) < 1e-6, "vertex_sc::store::read: transfer {} of {} differs from this "
                         "mesh's.", q, fn);
        nda::array<ComplexType, 4> P;
        nda::h5_read(g, "Pi", P);
        utils::check(P.shape(0) == nu_half.size() and P.shape(1) == q_ibz.shape(0) and P.shape(2) == Nm and P.shape(3) == Nm,
                     "vertex_sc::store::read: Pi of {} is {} x {} x {} x {}; expected {} x {} x {} x {} (the secondary points must be "
                     "the same: freeze them with pol_vertex_isdf_points_file).", fn, P.shape(0), P.shape(1), P.shape(2), P.shape(3),
                     nu_half.size(), q_ibz.shape(0), Nm, Nm);
        _New->local() = P;
        h5::h5_read(g, "n_refresh", nref);
      }
      _New->win().fence();
      _Pi = std::move(_New);
      mpi.node_comm.broadcast_n(&nref, 1, 0);
      n_refresh = nref;
    }

  private:
    std::shared_ptr<shm_t> _Pi, _New;
    bool _root = false;
  };

  /**
   * One refresh. make(k_int, q_ext) builds a view on the mesh; solve(view) returns the ladder column
   * (nw_half, view.n_ext(), Nm, Nm) on the view, the same on every rank; add(q, w, conj, P) accumulates w * P (or its
   * complex conjugate) into the new solution at the IBZ transfer q. kcrys / qcrys: crystal coordinates of the full k mesh /
   * of the full transfer mesh, whose first nq_ibz rows are the IBZ transfers. The sublattice transfers are solved in
   * batches of cfg.batch, so only one batch of solutions is held at a time; with cfg.trev one of every pair (q, -q) is
   * solved and the other follows from time reversal, Pi-bar(-q) = conj(Pi-bar(q)).
   * Returns the number of ladder passes.
   */
  template<class MAKE, class SOLVE, class ADD, class STAR, class UNFOLD>
  inline long evaluate(config const &cfg, nda::MemoryArrayOfRank<2> auto const &lattv, nda::MemoryArrayOfRank<2> auto const &kcrys,
                       nda::MemoryArrayOfRank<2> auto const &qcrys, long nq_ibz, MAKE &&make, SOLVE &&solve, ADD &&add,
                       STAR &&star, UNFOLD &&unfold) {
    // star(q_full) -> (q_ibz_full, js, trev) of transfer q_full on a symmetric mesh (S_js q_full = +-q_ibz; q_ibz = -1 on a
    // nosym mesh); unfold(q_img, js, trev, k_int, P) -> Pi-bar at the image q_img from P = Pi-bar(q_ibz)
    // (vertex_t::unfold_fit_matrix / unfold_pol), used by "interp" with cfg.unfold = "fit"
    decltype(nda::range::all) all;
    const auto k_int = vertex_view::sublattice(kcrys, cfg.mesh);
    const long nb = std::max(1l, cfg.batch);
    long npass = 0;
    const bool hybrid = (cfg.mode == "hybrid");
    // "hybrid": the fine IBZ transfers close to Gamma (within direct_radius coarse spacings) are solved directly, the others
    // interpolated from the sublattice set; the sublattice transfers themselves are exact either way (K = delta)
    std::vector<char> q_direct(static_cast<size_t>(nq_ibz), 0);
    if (hybrid) q_direct = direct_set(cfg, lattv, qcrys, nq_ibz);
    if (cfg.mode == "interp" or hybrid) {
      const auto q_sub = sublattice_transfers(qcrys, cfg.mesh);
      const long Nc = long(q_sub.size());
      nda::array<double, 2> qc(Nc, 3), qf(nq_ibz, 3);
      for (long c = 0; c < Nc; ++c) qc(c, all) = qcrys(q_sub[size_t(c)], all);
      for (long q = 0; q < nq_ibz; ++q) qf(q, all) = qcrys(q, all);
      auto K = interpolation_kernel(lattv, cfg.mesh, qc, qf);
      if (hybrid)   // the directly solved fine transfers take no interpolated contribution
        for (long q = 0; q < nq_ibz; ++q) if (q_direct[size_t(q)]) for (long c = 0; c < Nc; ++c) K(q, c) = ComplexType(0.0);
      // -q of every sublattice transfer; with time reversal only the first of each pair is solved
      std::vector<long> minus(static_cast<size_t>(Nc), -1l), todo;
      for (long c = 0; c < Nc; ++c)
        for (long d = 0; d < Nc and minus[size_t(c)] < 0; ++d) {
          bool same = true;
          for (int a = 0; a < 3; ++a) same = same and std::abs(fold(qc(c, a) + qc(d, a))) < 1e-6;
          if (same) minus[size_t(c)] = d;
        }
      for (long c = 0; c < Nc; ++c) utils::check(minus[size_t(c)] >= 0, "vertex_sc::evaluate: the sublattice has no -q for transfer {}.", c);
      // the IBZ symmetry (cfg.unfold = "fit"): every sublattice transfer is the image S q (or -S q) of an IBZ sublattice
      // transfer q; only those are solved and the images are unfolded in the aux frame
      std::vector<long> rep(static_cast<size_t>(Nc), -1l), rep_js(static_cast<size_t>(Nc), 0l);
      std::vector<char> rep_trev(static_cast<size_t>(Nc), 0);
      bool fit = (cfg.unfold == "fit");
      if (fit) {
        std::map<long, long> c_of_full;
        for (long c = 0; c < Nc; ++c) c_of_full[q_sub[size_t(c)]] = c;
        for (long c = 0; c < Nc; ++c) {
          long q_ibz = -1, js = 0; bool tr = false;
          star(q_sub[size_t(c)], q_ibz, js, tr);
          if (q_ibz < 0) { fit = false; break; }                       // nosym: nothing to unfold
          auto it = c_of_full.find(q_ibz);
          utils::check(it != c_of_full.end(), "vertex_sc::evaluate: the IBZ representative {} of the sublattice transfer {} is not a sublattice "
                       "transfer (the sublattice is not closed under the symmetry group).", q_ibz, q_sub[size_t(c)]);
          rep[size_t(c)] = it->second; rep_js[size_t(c)] = js; rep_trev[size_t(c)] = tr ? 1 : 0;
        }
      }
      if (fit) {
        for (long c = 0; c < Nc; ++c) if (rep[size_t(c)] == c) todo.push_back(c);
      } else {
        for (long c = 0; c < Nc; ++c) if (not cfg.trev or minus[size_t(c)] >= c) todo.push_back(c);
      }
      const bool check = fit and vertex_debug::flag("sc_unfold_check");   // vertex_debug: sc_unfold_check -- solve the images too and compare
      app_log(1, "  [vertex self-consistency] interpolation: {} of the {} sublattice transfers are solved ({}), in batches of {}; "
                 "{} IBZ transfers of the mesh are interpolated.", todo.size(), Nc,
              fit ? "the IBZ ones; the star images unfolded in the aux frame" : cfg.trev ? "the others by time reversal" : "time reversal not used",
              nb, nq_ibz);
      std::map<long, nda::array<ComplexType, 3>> solved;                // "fit": the IBZ solutions (the images are unfolded after)
      for (long i0 = 0; i0 < long(todo.size()); i0 += nb) {
        std::vector<long> cs(todo.begin() + i0, todo.begin() + std::min<long>(i0 + nb, long(todo.size())));
        std::vector<long> qe;
        for (long c : cs) qe.push_back(q_sub[size_t(c)]);
        auto view = make(k_int, qe);
        utils::check(view.nk_arr() == view.nk_int, "vertex_sc::evaluate: the sublattice is not closed under its own transfers.");
        auto P = solve(view);
        ++npass;
        for (long e = 0; e < long(cs.size()); ++e) {
          const long c = cs[size_t(e)], cm = minus[size_t(c)];
          for (long q = 0; q < nq_ibz; ++q)
            if (std::abs(K(q, c)) > 1e-14) add(q, K(q, c), false, P(all, e, all, all));
          if (fit) solved.emplace(c, nda::array<ComplexType, 3>(P(all, e, all, all)));
          else if (cfg.trev and cm != c)
            for (long q = 0; q < nq_ibz; ++q)
              if (std::abs(K(q, cm)) > 1e-14) add(q, K(q, cm), true, P(all, e, all, all));
        }
      }
      if (fit) {
        double res_max = 0.0, err_max = 0.0;
        long n_img = 0;
        std::vector<long> imgs;
        for (long c = 0; c < Nc; ++c) if (rep[size_t(c)] != c) imgs.push_back(c);
        std::map<long, nda::array<ComplexType, 3>> ref;
        if (check) {   // the images solved directly, for the error meter (DIAGNOSTIC: doubles the cost)
          for (long i0 = 0; i0 < long(imgs.size()); i0 += nb) {
            std::vector<long> cs(imgs.begin() + i0, imgs.begin() + std::min<long>(i0 + nb, long(imgs.size())));
            std::vector<long> qe;
            for (long c : cs) qe.push_back(q_sub[size_t(c)]);
            auto view = make(k_int, qe);
            auto P = solve(view);
            ++npass;
            for (long e = 0; e < long(cs.size()); ++e) ref.emplace(cs[size_t(e)], nda::array<ComplexType, 3>(P(all, e, all, all)));
          }
        }
        for (long c : imgs) {
          const long r = rep[size_t(c)];
          double resid = 0.0;
          auto Pi = unfold(q_sub[size_t(c)], rep_js[size_t(c)], rep_trev[size_t(c)] != 0, k_int, solved.at(r), &resid);
          res_max = std::max(res_max, resid);
          ++n_img;
          if (check) {
            auto const &R = ref.at(c);
            double num = 0.0, den = 0.0;
            for (long i = 0; i < long(R.size()); ++i) { num += std::norm(Pi.data()[i] - R.data()[i]); den += std::norm(R.data()[i]); }
            const double err = std::sqrt(num / std::max(den, 1e-300));
            err_max = std::max(err_max, err);
            app_log(1, "    [sc_unfold_check] sublattice transfer {} (full {}) = image of {} (js {}, trev {}): fit residual {:.3e}, "
                       "|unfolded - solved| / |solved| = {:.3e}", c, q_sub[size_t(c)], r, rep_js[size_t(c)], int(rep_trev[size_t(c)]), resid, err);
          }
          for (long q = 0; q < nq_ibz; ++q)
            if (std::abs(K(q, c)) > 1e-14) add(q, K(q, c), false, Pi);
        }
        app_log(1, "  [vertex self-consistency] interpolation: {} star images unfolded in the aux frame (max fit residual {:.3e}{})",
                n_img, res_max, check ? std::string(", max |unfolded - solved| / |solved| ") + [&] { char b[32]; std::snprintf(b, 32, "%.3e", err_max); return std::string(b); }() : std::string(""));
      }
    }
    if (cfg.mode == "direct" or hybrid) {
      std::vector<long> qs;
      for (long q = 0; q < nq_ibz; ++q) if (not hybrid or q_direct[size_t(q)]) qs.push_back(q);
      const auto classes = transfer_classes(qcrys, qs, cfg.mesh);
      long icl = 0;
      for (auto const &cl : classes) {
        ++icl;
        for (long i0 = 0; i0 < long(cl.size()); i0 += nb) {
          std::vector<long> qe(cl.begin() + i0, cl.begin() + std::min<long>(i0 + nb, long(cl.size())));
          auto view = make(k_int, qe);
          app_log(1, "  [vertex self-consistency] direct evaluation: class {} of {} of the transfers modulo the sublattice, "
                     "{} of its {} transfers ({} k points gathered)", icl, classes.size(), qe.size(), cl.size(), view.nk_arr());
          auto P = solve(view);
          ++npass;
          for (long e = 0; e < long(qe.size()); ++e) add(qe[size_t(e)], ComplexType(1.0), false, P(all, e, all, all));
        }
      }
    }
    utils::check(cfg.mode == "interp" or cfg.mode == "direct" or hybrid, "vertex_sc::evaluate: unknown mode \"{}\" (interp | direct | hybrid).", cfg.mode);
    return npass;
  }

} // vertex_sc
} // solvers
} // methods

#endif // COQUI_VERTEX_SC_HPP
