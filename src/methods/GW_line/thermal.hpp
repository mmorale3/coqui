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

#ifndef COQUI_METHODS_GW_LINE_THERMAL_HPP
#define COQUI_METHODS_GW_LINE_THERMAL_HPP

/**
 * Finite temperature on the line (S8b; notes section 11; python coqui/cayley/cayley/line/thc_gw.py, set_poles(beta=...),
 * bos_data, pi_tau_leg, pi_tau_dpi, and line/closure.py). Energies are mu-relative, f(e) = 1 / (e^{beta e} + 1),
 * n(nu) = 1 / (e^{beta nu} - 1), c_T = ln(1 / thermal_tol), E_T = c_T / beta.
 *
 *  - thermal_lists (Eq. fT_sectors): per k, the particle list = {e > E_T, v} U {|e| <= E_T, sqrt(1 - f) v}, the hole list =
 *    {e < -E_T, v} U {|e| <= E_T, sqrt(f) v} (a window pole is in BOTH lists). They are ordinary pole_data_t (factorized):
 *    the propagator, the Pi / Sigma kernels, the IBZ sharing and the device code are unchanged; D(k) = the hole list's
 *    density (static_part unchanged). Thermal mode is active iff some window is non-empty (window_active); otherwise every
 *    kernel runs the T = 0 path (bitwise).
 *  - bos_data (notes section 11.5(a)): the bosonic data set D = unmasked line nodes (rho beta |zeta| >= c_zeta) of the
 *    gapless bosonic LINE basis U the wedge band (band_heights x band_x points, heights v = y cos(theta_t) - |x| sin(theta_t)
 *    log-spaced in [d0, vtop], d0 = band_c sin(theta_t) / beta, vtop = band_top zeta_T sin(theta)) U {i nu_n, n = 1..N_M},
 *    N_M = ceil(mats_factor zeta_T beta / 2 pi) U {nu_0 = 0}. Mirror layout (default, when the line nodes are mirror-symmetric):
 *    [half, -conj(half)] with half = the ray-1 line nodes, the band points with x >= 0, the i nu_n and 0 (the self-mirror
 *    points appear twice, once per half): the perf-7.1 mirror paths of polarization / screened stay active (mirror_half).
 *    kind: 0 line, 1 band, 2 Matsubara, 3 nu_0.
 *  - tau leg (notes section 11.5 "Prototype (S8b.1b)"): Pi(q, i nu_n) in the Matsubara convention by the tau quadrature of
 *    Pi(q, tau)_PQ = -(2/N_k) sum_k [X (sum_m f_m e^{e_m tau} c_m) X^dag]_PQ [X (sum_m (1 - f_m) e^{-e_m tau} c_m) X^dag](k-q)_QP
 *    with ALL poles (not the thermal lists: a far particle dropped from the hole list is O(1) near tau = beta). Implemented with
 *    the polarization kernel on tau "rays" at theta_t = pi / 2 over [0, beta / 2] only (time_ray_t::tau_half or the
 *    finite-interval ID): the particle leg (t = -i tau, lists tau_lists: particle = all poles x sqrt(1 - f), hole = all
 *    poles x sqrt(f)) gives int_0^{beta/2} Pi(q, tau) e^{i nu tau}, the hole leg (t = +i tau) gives int_0^{beta/2}
 *    Pi(-q, tau)^T e^{-i nu tau} = int_{beta/2}^beta Pi(q, tau) e^{i nu tau} (bosonic KMS). On [0, beta / 2] every factor is
 *    bounded by e^{-beta |e| / 2} beyond the window and computed as (scaled residue) x (exponential) without overflow
 *    (poles whose weight underflows to 0 are dropped: their products are < e^{-370}). The dynamic Pi(q, 0) of the data set
 *    = Pi^Mats(q, i nu_0) - dPi, dPi = the degenerate pairs (|e_n(k) - e_m(k-q)| < deg_tol) with their exact tau integral
 *    f_n (1 - f_m) int_0^beta e^{(e_n - e_m) tau} (pi_tau_dpi).
 *  - Sigma: the W(t) of Eq. fT_W through the Bose-augmented basis (bosonic_basis_t::with_bose; the fit of screened.hpp writes
 *    the extra rows w_j(-q)^T), the self-energy kernels unchanged. Head term (Eq. headsig at finite T): head_sigma_thermal.
 *  - chemical potential: mu_rule "gap" | "number" | "auto" (notes section 11.6, Eq. fT_nth: the widest admissible gap
 *    midpoint mu_g iff |dN| <= mu_dn_max and |dN| > mu_th_factor n_th(mu_g), dN = N_T(mu_g) - N_el; else N(mu) = N_el);
 *    with an EMPTY window at mu_g the run is T = 0 there and the midpoint is kept (bitwise the T = 0 path, gate T4).
 */

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "utilities/Timer.hpp"
#include "mean_field/MF.hpp"
#include "numerics/line_dlr/bosonic_basis.hpp"
#include "numerics/line_dlr/cayley.hpp"
#include "numerics/line_dlr/time_ray.hpp"
#include "numerics/line_dlr/time_id.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/polarization.hpp"
#include "methods/GW_line/ibz.hpp"
#include "methods/GW_line/head.hpp"
#include "methods/GW_line/thermal_mu.hpp"
#include "methods/GW_line/time_grids.hpp"

namespace methods::gw_line {

/// number of poles with |e| <= E_T per k (mu-relative)
inline std::vector<long> window_counts(pole_data_t const &p, double E_T) {
  std::vector<long> c(p.nk, 0);
  for (long k = 0; k < p.nk; ++k)
    for (auto const *ps : {&p.part[k], &p.hole[k]})
      for (long m = 0; m < ps->size(); ++m)
        if (std::abs(ps->e(m)) <= E_T) ++c[k];
  return c;
}
inline bool window_active(pole_data_t const &p, double E_T) {
  for (long c : window_counts(p, E_T))
    if (c > 0) return true;
  return false;
}

namespace detail {
/// all poles of k (hole sector then particle sector: ascending for Lehmann poles), factorized
inline void merged_poles(pole_data_t const &p, long k, std::vector<double> &e, std::vector<long> &src, std::vector<long> &col) {
  e.clear(); src.clear(); col.clear();
  for (int s = 0; s < 2; ++s) {
    auto const &ps = s == 0 ? p.hole[k] : p.part[k];
    utils::check(ps.size() == 0 or ps.is_factorized(), "gw_line::thermal: factorized (Lehmann) poles required (g_repr = \"lehmann\")");
    for (long m = 0; m < ps.size(); ++m) {
      e.push_back(ps.e(m));
      src.push_back(s);
      col.push_back(m);
    }
  }
}
/// log(1 + e^y), overflow-free
inline double log1pexp(double y) { return std::max(0.0, y) + std::log1p(std::exp(-std::abs(y))); }
} // namespace detail

/**
 * Eq. fT_sectors: the thermal sector lists of the (factorized, mu-relative) poles; weights sqrt(1 - f) / sqrt(f) on the
 * window |e| <= E_T, far poles unscaled (exact copies).
 */
inline pole_data_t thermal_lists(pole_data_t const &p, double beta, double E_T) {
  pole_data_t out;
  out.nk = p.nk;
  out.nb = p.nb;
  out.part.resize(p.nk);
  out.hole.resize(p.nk);
  std::vector<double> e;
  std::vector<long> src, col;
  for (long k = 0; k < p.nk; ++k) {
    detail::merged_poles(p, k, e, src, col);
    for (int s = 0; s < 2; ++s) {   // 0: particle list, 1: hole list
      std::vector<long> idx;
      for (long m = 0; m < long(e.size()); ++m) {
        const bool win = std::abs(e[m]) <= E_T;
        if (win or (s == 0 ? e[m] > E_T : e[m] < -E_T)) idx.push_back(m);
      }
      nda::array<double, 1> es(long(idx.size()));
      nda::array<ComplexType, 2> vs(p.nb, long(idx.size()));
      for (long j = 0; j < long(idx.size()); ++j) {
        const long m   = idx[j];
        auto const &ps = src[m] == 0 ? p.hole[k] : p.part[k];
        es(j)          = e[m];
        if (std::abs(e[m]) <= E_T) {
          const double f = numerics::line_dlr::fermi(e[m], beta);
          const double c = std::sqrt(s == 0 ? numerics::line_dlr::fermi(-e[m], beta) : f);   // 1 - f = f(-e)
          for (long i = 0; i < p.nb; ++i) vs(i, j) = c * ps.v(i, col[m]);
        } else
          for (long i = 0; i < p.nb; ++i) vs(i, j) = ps.v(i, col[m]);
      }
      (s == 0 ? out.part[k] : out.hole[k]) = pole_sector_t::factorized_form(std::move(es), std::move(vs));
    }
  }
  return out;
}

/// tau leg lists (file header): particle = all poles x sqrt(1 - f), hole = all poles x sqrt(f) (weights that underflow to 0 dropped)
inline pole_data_t tau_lists(pole_data_t const &p, double beta) {
  pole_data_t out;
  out.nk = p.nk;
  out.nb = p.nb;
  out.part.resize(p.nk);
  out.hole.resize(p.nk);
  std::vector<double> e;
  std::vector<long> src, col;
  for (long k = 0; k < p.nk; ++k) {
    detail::merged_poles(p, k, e, src, col);
    for (int s = 0; s < 2; ++s) {
      std::vector<long> idx;
      std::vector<double> wt;
      for (long m = 0; m < long(e.size()); ++m) {
        const double w = std::exp(-detail::log1pexp(s == 0 ? -beta * e[m] : beta * e[m]));   // 1 - f  /  f
        if (w > 0.0) {
          idx.push_back(m);
          wt.push_back(w);
        }
      }
      nda::array<double, 1> es(long(idx.size()));
      nda::array<ComplexType, 2> vs(p.nb, long(idx.size()));
      for (long j = 0; j < long(idx.size()); ++j) {
        const long m   = idx[j];
        auto const &ps = src[m] == 0 ? p.hole[k] : p.part[k];
        es(j)          = e[m];
        const double c = std::sqrt(wt[j]);
        for (long i = 0; i < p.nb; ++i) vs(i, j) = c * ps.v(i, col[m]);
      }
      (s == 0 ? out.part[k] : out.hole[k]) = pole_sector_t::factorized_form(std::move(es), std::move(vs));
    }
  }
  return out;
}

/// The bosonic data set D (file header). Returns the points and kinds; mirror layout when requested and possible.
struct bos_data_t {
  nda::array<ComplexType, 1> z;
  std::vector<int> kind;   ///< 0 line, 1 band, 2 i nu_n, 3 nu_0
  long n_mats = 0, n_line = 0, n_band = 0;
  double d0 = 0.0, vtop = 0.0;
  bool mirror = false;
  long count(int kd) const { return std::count(kind.begin(), kind.end(), kd); }
};

inline bos_data_t make_bos_data(nda::array<ComplexType, 1> const &line_nodes, thermal_params_t const &tp) {
  const double beta = tp.beta, tht = tp.theta_t, rho = tp.rho(), zT = tp.zeta_T();
  const double cb = tp.band_c > 0.0 ? tp.band_c : tp.c_zeta;
  bos_data_t D;
  D.d0   = cb * std::sin(tht) / beta;
  D.vtop = tp.band_top * zT * std::sin(tp.theta);
  utils::check(D.vtop > D.d0, "gw_line::make_bos_data: empty wedge band (d0 {} >= vtop {})", D.d0, D.vtop);
  const long nl = line_nodes.size();
  const long n1 = numerics::line_dlr::mirror_half(line_nodes);
  D.mirror = tp.mirror_D and n1 > 0;
  std::vector<ComplexType> z;
  std::vector<int> kd;
  auto keep = [&](ComplexType x) { return rho * beta * std::abs(x) >= tp.c_zeta; };
  // line nodes
  for (long i = 0; i < (D.mirror ? n1 : nl); ++i)
    if (keep(line_nodes(i))) { z.push_back(line_nodes(i)); kd.push_back(0); }
  // band
  auto hv = numerics::line_dlr::detail::logspace(D.d0, D.vtop, tp.band_heights);
  for (long h = 0; h < tp.band_heights; ++h) {
    const double v = hv(h), xm = std::abs(D.vtop * std::cos(tht) - v) / std::sin(tht);
    auto xs        = numerics::line_dlr::detail::linspace(-xm, xm, tp.band_x);
    const long nx  = tp.band_x;
    std::vector<double> xx(nx);
    for (long i = 0; i < nx; ++i) xx[i] = 0.5 * (xs(i) - xs(nx - 1 - i));   // exactly symmetric (python)
    for (long i = 0; i < nx; ++i) {
      if (D.mirror and xx[i] < 0.0) continue;
      z.push_back(ComplexType(xx[i], (v + std::abs(xx[i]) * std::sin(tht)) / std::cos(tht)));
      kd.push_back(1);
    }
  }
  // Matsubara
  D.n_mats = long(std::ceil(tp.mats_factor * zT * beta / (2.0 * std::numbers::pi)));
  for (long n = 1; n <= D.n_mats; ++n) {
    z.push_back(ComplexType(0.0, 2.0 * std::numbers::pi * double(n) / beta));
    kd.push_back(2);
  }
  z.push_back(ComplexType(0.0, 0.0));
  kd.push_back(3);
  const long nh = long(z.size());
  if (D.mirror) {
    for (long i = 0; i < nh; ++i) {
      z.push_back(-std::conj(z[i]));
      kd.push_back(kd[i]);
    }
    // the negated zero is (-0, 0): store +0 exactly (mirror_half compares |z_{n1+i} + conj z_i|)
    for (long i = nh; i < 2 * nh; ++i)
      if (z[i] == ComplexType(0.0, 0.0)) z[i] = ComplexType(0.0, 0.0);
  }
  D.z = nda::array<ComplexType, 1>(long(z.size()));
  for (long i = 0; i < long(z.size()); ++i) D.z(i) = z[i];
  D.kind   = kd;
  D.n_line = D.count(0);
  D.n_band = D.count(1);
  if (D.mirror) utils::check(numerics::line_dlr::mirror_half(D.z) == nh, "gw_line::make_bos_data: mirror layout broken");
  return D;
}

/**
 * Degenerate-pair term of the tau leg at nu_0 (python LineGW.pi_tau_dpi), (g, nP, nQ) block layout for the q list qs:
 *   dPi(q)_PQ = -(2/N_k) sum_k sum_{n at k, m at k-q, |e_n - e_m| < deg_tol} f_n (1 - f_m) phi_nm a_P conj(a_Q) b_Q conj(b_P),
 *   a = X(k) v_n, b = X(k-q) v_m, phi = int_0^beta e^{(e_n - e_m) tau} dtau.
 * pf: the poles of ALL full-BZ k (factorized). Xp (nk, nP, nb) = X(k)[P_rng, :], Xq (nk, nQ, nb) = X(k)[Q_rng, :] (host).
 */
inline nda::array<ComplexType, 3> pi_tau_dpi(pole_data_t const &pf, nda::array<ComplexType, 3> const &Xp,
                                             nda::array<ComplexType, 3> const &Xq, mf::MF const &mf, std::vector<long> const &qs,
                                             double beta, double deg_tol, long *npairs = nullptr) {
  const long nk = pf.nk, nb = pf.nb, nP = Xp.extent(1), nQ = Xq.extent(1), g = qs.size();
  auto qk = mf.qk_to_k2();
  nda::array<ComplexType, 3> out(g, nP, nQ);
  out() = ComplexType(0.0);
  std::vector<std::vector<double>> E(nk);
  std::vector<std::vector<long>> S(nk), C(nk), order(nk);
  for (long k = 0; k < nk; ++k) {
    detail::merged_poles(pf, k, E[k], S[k], C[k]);
    order[k].resize(E[k].size());
    std::iota(order[k].begin(), order[k].end(), 0L);
    std::stable_sort(order[k].begin(), order[k].end(), [&](long a, long b) { return E[k][a] < E[k][b]; });
  }
  auto vcol = [&](long k, long m) {
    auto const &ps = S[k][m] == 0 ? pf.hole[k] : pf.part[k];
    return ps.v(nda::range::all, C[k][m]);
  };
  long np_ = 0;
  nda::array<ComplexType, 1> aP(nP), aQ(nQ), bP(nP), bQ(nQ);
  for (long iq = 0; iq < g; ++iq) {
    for (long k = 0; k < nk; ++k) {
      const long kq = qk(qs[iq], k);
      std::vector<double> es(order[kq].size());
      for (size_t i = 0; i < es.size(); ++i) es[i] = E[kq][order[kq][i]];
      for (long n = 0; n < long(E[k].size()); ++n) {
        const double en = E[k][n];
        auto lo = std::lower_bound(es.begin(), es.end(), en - deg_tol);
        for (auto it = lo; it != es.end() and *it < en + deg_tol; ++it) {
          const long m   = order[kq][it - es.begin()];
          const double x = en - E[kq][m];
          if (not(std::abs(x) < deg_tol)) continue;
          const double phi = (x == 0.0) ? beta : std::expm1(beta * x) / x;
          const double w   = std::exp(-detail::log1pexp(beta * en) - detail::log1pexp(-beta * E[kq][m])) * phi;
          if (w == 0.0) continue;
          ++np_;
          auto vn = vcol(k, n);
          auto vm = vcol(kq, m);
          for (long P = 0; P < nP; ++P) {
            ComplexType a = 0.0, b = 0.0;
            for (long i = 0; i < nb; ++i) {
              a += Xp(k, P, i) * vn(i);
              b += Xp(kq, P, i) * vm(i);
            }
            aP(P) = a;
            bP(P) = b;
          }
          for (long Q = 0; Q < nQ; ++Q) {
            ComplexType a = 0.0, b = 0.0;
            for (long i = 0; i < nb; ++i) {
              a += Xq(k, Q, i) * vn(i);
              b += Xq(kq, Q, i) * vm(i);
            }
            aQ(Q) = a;
            bQ(Q) = b;
          }
          const double c = -2.0 / double(nk) * w;
          for (long P = 0; P < nP; ++P) {
            const ComplexType u = c * aP(P) * std::conj(bP(P));
            for (long Q = 0; Q < nQ; ++Q) out(iq, P, Q) += u * std::conj(aQ(Q)) * bQ(Q);
          }
        }
      }
    }
  }
  if (npairs) *npairs = np_;
  return out;
}

/// host X slices of the propagator: Xp (nk, nP, nb), Xq (nk, nQ, nb) = X(k)[Q_rng, :]
template <MEMORY_SPACE MEM> inline std::pair<nda::array<ComplexType, 3>, nda::array<ComplexType, 3>> host_x_slices(propagator_t<MEM> const &prop) {
  nda::array<ComplexType, 3> Xp(memory::to_memory_space<HOST_MEMORY>(prop.Xp));
  nda::array<ComplexType, 3> XqT(memory::to_memory_space<HOST_MEMORY>(prop.XqT));   // (nk, nb, nQ) = X[Q, :]^T
  nda::array<ComplexType, 3> Xq(XqT.extent(0), XqT.extent(2), XqT.extent(1));
  for (long k = 0; k < Xq.extent(0); ++k)
    for (long Q = 0; Q < Xq.extent(1); ++Q)
      for (long i = 0; i < Xq.extent(2); ++i) Xq(k, Q, i) = XqT(k, i, Q);
  return {std::move(Xp), std::move(Xq)};
}

/// the tau "rays" (particle t = -i tau, hole = conjugate) of the tau leg on [0, beta / 2] for the poles of emax
struct tau_nodes_t {
  std::optional<numerics::line_dlr::time_nodes_t> p, h;
  long size() const { return p ? p->size() : 0; }
  std::string kind;
  long rank = 0;
};

/// comm (optional): the tau ID is built on its rank 0 and broadcast (bitwise identical nodes on every rank, as the ray grids)
inline tau_nodes_t make_tau_nodes(thermal_params_t const &tp, double emax, double espread,
                                  boost::mpi3::communicator *comm = nullptr) {
  tau_nodes_t T;
  using numerics::line_dlr::sector_t;
  if (tp.tau_grid == "id") {
    numerics::line_dlr::time_id_opts_t o;
    o.pad          = 1.25;
    const double En = 2.0 * std::log(1.0 / tp.tau_eps) / tp.beta;   // KMS: E < -En is below tau_eps on [0, beta / 2]
    numerics::line_dlr::time_id_t g;
    if (comm == nullptr or comm->rank() == 0)
      g = numerics::line_dlr::time_id_t::finite_interval(std::numbers::pi / 2.0, sector_t::particle, En, std::max(espread, 2.0 * En),
                                                         0.5 * tp.beta, tp.tau_eps, o);
    else
      g.opts = o;
    if (comm != nullptr) detail::bcast_time_id(*comm, g, 0);
    T.rank         = g.rank;
    auto gh        = g;
    gh.sector      = sector_t::hole;
    gh.phase       = std::conj(g.phase);
    for (auto &x : gh.t) x = std::conj(x);
    for (auto &x : gh.E) x = -x;
    for (auto &x : gh.Uc) x = std::conj(x);
    for (auto &x : gh.Vs) x = std::conj(x);
    T.p.emplace(g);
    T.h.emplace(gh);
    T.kind = "id";
  } else {
    utils::check(tp.tau_grid == "gl", "gw_line: tau_grid must be \"gl\" or \"id\" (got \"{}\")", tp.tau_grid);
    auto rp = numerics::line_dlr::time_ray_t::tau_half(tp.beta, emax, tp.tau_nn, tp.tau_per_efold, tp.tau_x0, sector_t::particle);
    auto rh = numerics::line_dlr::time_ray_t::from_nodes(std::numbers::pi / 2.0, sector_t::hole, rp.s, rp.ws);
    T.p.emplace(rp);
    T.h.emplace(rh);
    T.kind = "gl";
  }
  return T;
}

/**
 * The tau leg for the q list qs (file header): Pi_tau (g, nz, nP, nQ) = Pi(q, z) at z = zeta_tau (zeta_tau(0) = 0 = nu_0, then
 * optional i nu_n), Matsubara convention, MINUS the degenerate-pair term at z = 0 (the dynamic Pi(q, 0) of the data set).
 * poles: the (IBZ) Lehmann poles, mu-relative. Collective: none beyond polarization's (none).
 */
template <MEMORY_SPACE MEM>
void pi_tau_leg(propagator_t<MEM> &prop, pole_data_t const &poles, mf::MF const &mf, ibz_t const &ibz, aux_grid_t const &grid,
                thermal_params_t const &tp, tau_nodes_t const &tn, nda::array<ComplexType, 1> const &zeta_tau, long t_chunk,
                memory::array<MEM, ComplexType, 4> &Pi_tau, utils::TimerManager &Timer, std::vector<long> const &qs,
                long *ndeg = nullptr) {
  utils::check(zeta_tau.size() >= 1 and zeta_tau(0) == ComplexType(0.0), "gw_line::pi_tau_leg: zeta_tau(0) must be 0");
  Timer.add("Pi_tau_leg");
  Timer.start("Pi_tau_leg");
  auto tl = tau_lists(poles, tp.beta);
  polarization<MEM>(prop, tl, mf, grid, zeta_tau, *tn.p, *tn.h, t_chunk, Pi_tau, Timer, sector_t::both, qs);
  // degenerate pairs (host, full BZ)
  auto pf        = unfold_poles(poles, ibz);
  auto [Xp, Xq]  = host_x_slices(prop);
  auto dPi       = pi_tau_dpi(pf, Xp, Xq, mf, qs, tp.beta, tp.deg_tol, ndeg);
  auto dPiM      = memory::to_memory_space<MEM>(dPi);
  const long g = qs.size();
  for (long i = 0; i < g; ++i) {
    auto row = Pi_tau(i, 0, nda::range::all, nda::range::all);
    if constexpr (MEM == HOST_MEMORY) row -= dPiM(i, nda::range::all, nda::range::all);
    else nda::tensor::add(ComplexType(-1.0), dPiM(i, nda::range::all, nda::range::all), ComplexType(1.0), row);
  }
  Timer.stop("Pi_tau_leg");
}

/**
 * Finite-T head term of Sigma_c (notes section 11.5 "Screened interaction and self-energy", last sentence): weights
 * (1 - f_m + n_j) on the particle residues hp0_j and (f_m + n_j) on the hole residues hh0_j, summed over ALL poles m of k.
 * = the T = 0 correction on the thermal lists (gives (1 - f) hp0 / f hh0, far poles weight 1) + the Bose part on all poles
 * (both residue sets times n_j). nfit: the fitted poles (n_j = 0 beyond the window).
 */
inline void head_sigma_thermal(pole_data_t const &lists, pole_data_t const &poles, nda::array<ComplexType, 3> const &T,
                               nda::array<double, 1> const &nu, nda::array<ComplexType, 1> const &hp0,
                               nda::array<ComplexType, 1> const &hh0, double madelung, double beta, double E_T,
                               nda::array<ComplexType, 1> const &zeta, std::vector<long> const &krows, nda::array<ComplexType, 4> &Sp,
                               nda::array<ComplexType, 4> &Sh) {
  head_sigma_correction(lists, T, nu, hp0, hh0, madelung, zeta, krows, Sp, Sh);
  const long r = nu.size();
  nda::array<ComplexType, 1> np0(r), nh0(r);
  bool any = false;
  for (long j = 0; j < r; ++j) {
    const double x = beta * nu(j);
    const double n = (nu(j) <= E_T and x <= 700.0) ? 1.0 / std::expm1(x) : 0.0;
    np0(j) = n * hp0(j);
    nh0(j) = n * hh0(j);
    any    = any or n > 0.0;
  }
  if (not any) return;
  pole_data_t all;
  all.nk = poles.nk;
  all.nb = poles.nb;
  all.part.resize(poles.nk);
  all.hole.resize(poles.nk);
  std::vector<double> e;
  std::vector<long> src, col;
  for (long k = 0; k < poles.nk; ++k) {
    detail::merged_poles(poles, k, e, src, col);
    nda::array<double, 1> es(long(e.size()));
    nda::array<ComplexType, 2> vs(poles.nb, long(e.size()));
    for (long m = 0; m < long(e.size()); ++m) {
      es(m)          = e[m];
      auto const &ps = src[m] == 0 ? poles.hole[k] : poles.part[k];
      vs(nda::range::all, m) = ps.v(nda::range::all, col[m]);
    }
    all.part[k] = pole_sector_t::factorized_form(es, vs);
    all.hole[k] = pole_sector_t::factorized_form(es, vs);
  }
  head_sigma_correction(all, T, nu, np0, nh0, madelung, zeta, krows, Sp, Sh);
}

} // namespace methods::gw_line

#endif
