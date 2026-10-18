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

#ifndef COQUI_METHODS_GW_LINE_WARM_START_HPP
#define COQUI_METHODS_GW_LINE_WARM_START_HPP

/**
 * Quasiparticle warm starts of the line scGW (plan 7.2; driver key `start`):
 *
 *   qp_diag_energies: the diagonal QP equation in the KS band basis, per k and band,
 *       E_n = h_nn + Sigma_nn(E_n),   h = H0 + F - mu (KS basis), energies mu-relative,
 *     with Sigma_c(z) = W (z - d)^{-1} W^dagger the upfolded (Cayley-moment) representation of the closure of this k
 *     (closure_k: sector fits -> moments -> upfolding; real poles d, positive weights g_j = |W_nj|^2, the representation the
 *     spectra use; the raw sector-fit residues are not usable on the real axis: they are determined on the line only).
 *     D2: solved EXACTLY at eta = 0 by qp_secular (python oracle cayley.line.closure.qp_secular): f(E) = h - E + sum_j g_j /
 *     (E - d_j) falls from +inf to -inf between consecutive poles, so there is one root per interval (the eigenvalues of the
 *     arrowhead matrix [[h, sqrt g^T], [sqrt g, diag d]]), with weight Z = 1 / (1 + sum_j g_j / (E - d_j)^2); the QP energy is
 *     the root of LARGEST Z. (The perf-7.2 Newton / linearized solve at eta = start_eta linearized near poles of Sigma_nn and
 *     sent 157 (k, band) of Si 4^3 100 Ry by up to 162 eV -> "no gap" abort; start_eta is unused since D2.)
 *   read_qp_energies: QP energies (absolute, Ha, band order) from an h5 file (e.g. a CoQui Matsubara mbpt.h5).
 */

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/check.hpp"
#include "methods/GW_line/closure.hpp"

namespace methods::gw_line {

struct qp_diag_out_t {
  nda::array<double, 2> e;   ///< (nk, nb) mu-relative QP energies
  nda::array<double, 2> z;   ///< (nk, nb) their weights Z
  long n_newton = 0, n_lin = 0;   ///< (k, band) solved / without poles (E = h_nn)
  double max_shift = 0.0;    ///< max |E - e_ref| (Ha)
};

/// Largest-Z root of E = h + sum_j g_j / (E - d_j) (eta = 0, g_j >= 0), python closure.qp_secular: poles with g <= gtol_rel sum g
/// dropped, poles closer than merge_tol merged, one root per interval by bisection to the double resolution. Returns {E, Z}.
inline std::pair<double, double> qp_secular(double h, std::vector<std::pair<double, double>> dg, double gtol_rel = 1e-14,
                                            double merge_tol = 1e-12) {
  double G = 0.0;
  for (auto const &p : dg) G += p.second;
  if (dg.empty() or G <= 0.0) return {h, 1.0};
  std::vector<double> d, g;
  std::stable_sort(dg.begin(), dg.end(), [](auto const &a, auto const &b) { return a.first < b.first; });
  for (auto const &p : dg) {
    if (not(p.second > gtol_rel * G)) continue;
    if (not d.empty() and p.first - d.back() < merge_tol) g.back() += p.second;
    else {
      d.push_back(p.first);
      g.push_back(p.second);
    }
  }
  const long r = d.size();
  double sg    = 0.0;
  for (double x : g) sg += x;
  const double s = std::sqrt(sg) + 1e-3;
  auto f = [&](double x) {
    double v = h - x;
    for (long j = 0; j < r; ++j) v += g[j] / (x - d[j]);
    return v;
  };
  auto solve = [&](long j) -> std::pair<double, double> {   // root of interval j and its Z
    double lo = (j == 0) ? std::min(h, d[0]) - s : d[j - 1];
    double hi = (j == r) ? std::max(h, d[r - 1]) + s : d[j];
    for (int it = 0; it < 200; ++it) {
      const double x = 0.5 * (lo + hi);
      if (x == lo or x == hi) break;
      if (f(x) > 0.0) lo = x;
      else hi = x;
    }
    const double x = 0.5 * (lo + hi);
    double q       = 1.0;
    for (long i = 0; i < r; ++i) q += g[i] / ((x - d[i]) * (x - d[i]));
    return {x, 1.0 / q};
  };
  // candidates (python qp_secular): the outer intervals and the one containing h, then every interior interval whose rigorous
  // upper bound of Z beats them (two-neighbour minimum + the next nn poles on each side at the farthest end); exact pruning,
  // ties -> the lowest energy (intervals in index order, strict >)
  const long jh = long(std::lower_bound(d.begin(), d.end(), h) - d.begin());
  std::vector<long> js0 = {0, jh, r};
  std::sort(js0.begin(), js0.end());
  js0.erase(std::unique(js0.begin(), js0.end()), js0.end());
  double z0 = -1.0;
  std::vector<std::pair<double, double>> sol(r + 1, {0.0, -1.0});
  std::vector<char> done(r + 1, 0);
  for (long j : js0) {
    sol[j]  = solve(j);
    done[j] = 1;
    z0      = std::max(z0, sol[j].second);
  }
  constexpr long nn = 16;
  const bool certified = (z0 > 0.5);   // sum Z = 1: no other root can beat it
  for (long jl = 0; jl + 1 < r and not certified; ++jl) {   // interval jl + 1 = (d[jl], d[jl + 1])
    if (done[jl + 1]) continue;
    const double L = d[jl + 1] - d[jl], c = std::cbrt(g[jl]) + std::cbrt(g[jl + 1]);
    double S       = c * c * c / (L * L);
    for (long m = 1; m <= nn; ++m) {
      const long il = jl - m, ir = jl + 1 + m;
      if (il >= 0) S += g[il] / ((d[jl + 1] - d[il]) * (d[jl + 1] - d[il]));
      if (ir < r) S += g[ir] / ((d[ir] - d[jl]) * (d[ir] - d[jl]));
    }
    if (1.0 / (1.0 + S) > z0) {
      sol[jl + 1]  = solve(jl + 1);
      done[jl + 1] = 1;
    }
  }
  double bestE = h, bestZ = -1.0;
  for (long j = 0; j <= r; ++j)
    if (done[j] and sol[j].second > bestZ) {
      bestZ = sol[j].second;
      bestE = sol[j].first;
    }
  return {bestE, bestZ};
}

/**
 * eks (nk, nb): reference energies mu-relative (max_shift only); Hrel (nk, nb, nb) = H0 + F - mu; Sig_p / Sig_h: the rows of
 * this rank (k_rows = their global k; replicated rows: every rank computes all k, no reduction needed). Collective over comm.
 */
inline qp_diag_out_t qp_diag_energies(boost::mpi3::communicator &comm, nda::array<double, 2> const &eks,
                                      nda::array<ComplexType, 3> const &Hrel, nda::array<ComplexType, 4> const &Sig_p,
                                      nda::array<ComplexType, 4> const &Sig_h, std::vector<long> const &k_rows, bool distributed,
                                      nda::array<ComplexType, 1> const &zeta, line_basis_t const &bp, line_basis_t const &bh,
                                      closure_params_t const &cp) {
  auto all      = nda::range::all;
  const long nk = eks.extent(0), nb = eks.extent(1);
  qp_diag_out_t out;
  out.e = nda::array<double, 2>(nk, nb);
  out.z = nda::array<double, 2>(nk, nb);
  out.e() = 0.0;
  out.z() = 0.0;
  nda::array<double, 1> cnt(2);
  cnt() = 0.0;
  for (long l = 0; l < long(k_rows.size()); ++l) {
    const long k = k_rows[l];
    if (not distributed and k % comm.size() != comm.rank()) continue;   // replicated: round-robin, then all_reduce
    nda::array<ComplexType, 3> Sp(Sig_p(l, all, all, all)), Sh(Sig_h(l, all, all, all));
    auto sp      = fit_sigma_sectors(bp, bh, zeta, Sp, Sh);
    nda::array<ComplexType, 2> Hk(Hrel(k, all, all));
    auto ck      = closure_k(Hk, sp, cp);
    const long r = ck.d.size();
    for (long n = 0; n < nb; ++n) {
      std::vector<std::pair<double, double>> dg(r);
      for (long j = 0; j < r; ++j) dg[j] = {ck.d(j), std::norm(ck.W(n, j))};
      auto [E, Z] = qp_secular(std::real(Hrel(k, n, n)), std::move(dg));
      out.e(k, n) = E;
      out.z(k, n) = Z;
      cnt(r > 0 ? 0 : 1) += 1.0;
    }
  }
  if (comm.size() > 1) {
    comm.all_reduce_in_place_n(out.e.data(), out.e.size(), std::plus<>{});
    comm.all_reduce_in_place_n(out.z.data(), out.z.size(), std::plus<>{});
    comm.all_reduce_in_place_n(cnt.data(), cnt.size(), std::plus<>{});
  }
  out.n_newton = long(std::llround(cnt(0)));
  out.n_lin    = long(std::llround(cnt(1)));
  for (long k = 0; k < nk; ++k)
    for (long n = 0; n < nb; ++n) out.max_shift = std::max(out.max_shift, std::abs(out.e(k, n) - eks(k, n)));
  return out;
}

/// QP energies (nk, nb) absolute (Ha) from `file`: dataset `ds`, or (ds empty) scf/iter<final_iter>/qp_approx/E_ska, else
/// E_ska / qp_energies at the root. Shapes (nk, nb') or (1, nk, nb') with nb' >= nb (the first nb bands are used).
/// Root reads, broadcast.
inline nda::array<double, 2> read_qp_energies(boost::mpi3::communicator &comm, std::string const &file, std::string ds, long nk,
                                              long nb) {
  nda::array<double, 2> E(nk, nb);
  std::string used;
  if (comm.root()) {
    utils::check(not file.empty(), "gw_line: start = \"qp_file\" needs start_file");
    h5::file f(file, 'r');
    h5::group g(f);
    if (ds.empty()) {
      if (g.has_subgroup("scf") and g.open_group("scf").has_dataset("final_iter")) {
        long fi = 0;
        h5::h5_read(g.open_group("scf"), "final_iter", fi);
        ds = "scf/iter" + std::to_string(fi) + "/qp_approx/E_ska";
      } else if (g.has_dataset("E_ska")) {
        ds = "E_ska";
      } else {
        ds = "qp_energies";
      }
    }
    used = ds;
    nda::array<double, 3> e3;
    nda::array<double, 2> e2;
    utils::check(g.has_dataset(ds), "gw_line: {} has no dataset {}", file, ds);
    const int rk = h5::array_interface::get_dataset_info(g, ds).rank();
    utils::check(rk == 2 or rk == 3, "gw_line: {}:{} has rank {}, expected (nk, nb) or (1, nk, nb)", file, ds, rk);
    if (rk == 3) {
      nda::h5_read(g, ds, e3);
      utils::check(e3.extent(0) == 1 and e3.extent(1) == nk and e3.extent(2) >= nb,
                   "gw_line: {}:{} has shape ({}, {}, {}), expected (1, {}, >= {})", file, ds, e3.extent(0), e3.extent(1),
                   e3.extent(2), nk, nb);
      for (long k = 0; k < nk; ++k)
        for (long n = 0; n < nb; ++n) E(k, n) = e3(0, k, n);
    } else {
      nda::h5_read(g, ds, e2);
      utils::check(e2.extent(0) == nk and e2.extent(1) >= nb, "gw_line: {}:{} has shape ({}, {}), expected ({}, >= {})", file, ds,
                   e2.extent(0), e2.extent(1), nk, nb);
      for (long k = 0; k < nk; ++k)
        for (long n = 0; n < nb; ++n) E(k, n) = e2(k, n);
    }
    app_log(1, "  start: QP energies read from {}:{} ({} k x {} bands used)", file, ds, nk, nb);
  }
  comm.broadcast_n(E.data(), E.size(), 0);
  return E;
}

} // namespace methods::gw_line

#endif
