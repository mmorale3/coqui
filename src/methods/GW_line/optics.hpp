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

#ifndef COQUI_METHODS_GW_LINE_OPTICS_HPP
#define COQUI_METHODS_GW_LINE_OPTICS_HPP

/**
 * S9b: real-axis optics from the line (notes section "Optics from the line"; design notes/bosonic_closure_design.md 7).
 *
 * Input per q (finite mesh q != Gamma and the q -> 0 extrapolation h0 = sum_q c_q h(q)): the head
 * h(q, zeta_i) = eps^-1_00(q, zeta_i) - 1 at the bosonic nodes (head.hpp / head_pass.hpp) and the bosonic poles nu_j.
 * Two channels, both odd with a POSITIVE particle measure (response_closure.hpp convention):
 *   loss channel (R1):        h = eps^-1 - 1                     -> loss L = -Im h,     eps_R1 = 1 / (1 + h)
 *   dielectric channel (R2):  m = 1 - eps_M = h / (1 + h)       -> eps = 1 - m (eps_1, eps_2, n, kappa, alpha, R, sigma)
 *   (m is formed POINTWISE at the nodes; eps_M - 1 = -m has the negative measure, we close 1 - eps as the study does).
 * Per channel: odd fit (real residues) -> delta = its relative residual at the nodes (floor 1e-10) -> K rule and tol_gram
 * rule at the line angle -> MB closure (primary) and the NNLS positive fit G (second estimate) -> evaluation at
 * z = omega + i eta(omega) for every broadening (constant eta and eta = c omega).
 * Stored next to every curve: the error bar |Q(MB) - Q(G)| (the same quantity from the NNLS estimate of the same channel;
 * it tracks the true error of MB within 2x in 85% of the study's cases) and the cross-channel consistency meter
 * |eps_R1 - eps_R2| (eps from the loss channel vs the dielectric channel) and |L_R1 - L_R2|.
 * Statics and sum rules from the line FITS (not from the closures): eps_inf = 1 / (1 + h_fit(0)) and 1 - m_fit(0), and from
 * the NNLS measure; f-sum omega_p^2 = sum_j 2 r_j nu_j of both channels (int_0^inf omega eps_2 = (pi/2) omega_p^2), vs the
 * valence plasma frequency omega_p^2 = 4 pi N_e / Omega; the closures' own f-sum (diagnostic).
 *
 * PHYSICS SCOPE (also in the h5 attributes): RPA polarization built from the self-consistent G of the line (no vertex
 * corrections: no excitons, the absorption onset is the direct QP gap); head eps^-1_00 including local fields through the
 * Dyson inversion of W; coarse k meshes (eps_2 on 8 or 64 k is a sum of few broadened transitions); q -> 0 from the
 * extrapolation of the finite-q heads (variant-dependent).
 *
 * Distribution: q (and q0) round-robin over the ranks, each owner broadcasts its packed results; the root writes.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numbers>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/check.hpp"
#include "numerics/line_dlr/response_closure.hpp"

namespace methods::gw_line {

namespace rc = numerics::line_dlr::response;

struct optics_params_t {
  bool enable = false;
  double wmin = 0.0, wmax = 1.5;   ///< Ha
  long nw = 1501;
  std::vector<double> eta     = {0.01};   ///< constant broadenings (Ha)
  std::vector<double> eta_rel = {0.05};   ///< energy-proportional broadenings eta = c omega
  std::vector<double> scales;             ///< MB scales (Ha); empty: auto
  long nscales = 4;
  long K = -1;                            ///< < 0: the K rule
  bool q0 = true, finite_q = true;
  std::vector<double> theta_deg;          ///< flatter final line(s) (deg, < the SCF angle); empty: none
  /// other q -> 0 extrapolation variants evaluated for q0 as well (sensitivity; groups q0_<variant>)
  std::vector<std::string> q0_variants = {"gygi_perdir", "gygi_smallest_q", "gygi_average"};
  long nnls_n = 1500;
  // head pass of the flatter line (head_pass_params_t)
  long nline = -1, npole = -1;
  std::string time_grid = "id";
  double mem_gb = 2.0;
  double delta_floor = 1e-10;

  static std::vector<double> opt_list(ptree const &pt, std::string const &id, std::vector<double> def, bool &is_auto) {
    is_auto = false;
    auto node = pt.get_child_optional(id);
    if (not node) return def;
    if (node->empty()) {
      auto s = node->get_value<std::string>();
      io::tolower(s);
      if (s == "auto" or s.empty()) {
        is_auto = true;
        return {};
      }
      return {std::stod(s)};
    }
    return io::get_array_with_default<double>(pt, id, def);
  }

  static optics_params_t from_ptree(ptree const &pt) {
    optics_params_t p;
    p.enable   = io::get_value_with_default<bool>(pt, "optics.enable", pt.get_child_optional("optics").has_value());
    p.wmin     = io::get_value_with_default<double>(pt, "optics.wmin", p.wmin);
    p.wmax     = io::get_value_with_default<double>(pt, "optics.wmax", p.wmax);
    p.nw       = io::get_value_with_default<long>(pt, "optics.nw", p.nw);
    p.eta      = io::get_array_with_default<double>(pt, "optics.eta", p.eta);
    p.eta_rel  = io::get_array_with_default<double>(pt, "optics.eta_rel", p.eta_rel);
    bool a     = false;
    p.scales   = opt_list(pt, "optics.scales", {}, a);
    p.nscales  = io::get_value_with_default<long>(pt, "optics.nscales", p.nscales);
    {
      auto k = opt_list(pt, "optics.K", {}, a);
      p.K    = (a or k.empty()) ? -1 : long(std::llround(k[0]));
    }
    p.q0        = io::get_value_with_default<bool>(pt, "optics.q0", p.q0);
    p.finite_q  = io::get_value_with_default<bool>(pt, "optics.finite_q", p.finite_q);
    p.theta_deg = opt_list(pt, "optics.theta_deg", {}, a);
    p.q0_variants = io::get_array_with_default<std::string>(pt, "optics.q0_variants", p.q0_variants);
    p.nnls_n    = io::get_value_with_default<long>(pt, "optics.nnls_n", p.nnls_n);
    p.nline     = io::get_value_with_default<long>(pt, "optics.nline", p.nline);
    p.npole     = io::get_value_with_default<long>(pt, "optics.npole", p.npole);
    p.time_grid = io::get_value_with_default<std::string>(pt, "optics.time_grid", p.time_grid);
    io::tolower(p.time_grid);
    p.mem_gb      = io::get_value_with_default<double>(pt, "optics.mem_gb", p.mem_gb);
    p.delta_floor = io::get_value_with_default<double>(pt, "optics.delta_floor", p.delta_floor);
    utils::check(p.nw >= 2 and p.wmax > p.wmin and p.wmin >= 0.0, "gw_line optics: need 0 <= wmin < wmax, nw >= 2");
    for (double e : p.eta) utils::check(e > 0.0, "gw_line optics: eta must be > 0");
    for (double e : p.eta_rel) utils::check(e > 0.0, "gw_line optics: eta_rel must be > 0");
    utils::check(not p.eta.empty() or not p.eta_rel.empty(), "gw_line optics: no broadening given");
    for (size_t i = 1; i < p.scales.size(); ++i) utils::check(p.scales[i] > p.scales[i - 1], "gw_line optics: scales must increase");
    for (double t : p.theta_deg) utils::check(t > 0.0 and t < 90.0, "gw_line optics: theta_deg in (0, 90)");
    utils::check(p.time_grid == "id" or p.time_grid == "gl", "gw_line optics: time_grid must be \"id\" or \"gl\"");
    return p;
  }

  long nbroad() const { return long(eta.size() + eta_rel.size()); }
  /// broadening b at omega: constant eta first, then the energy-proportional ones
  double eta_at(long b, double w) const { return b < long(eta.size()) ? eta[b] : eta_rel[b - eta.size()] * w; }
  nda::array<double, 1> omega() const {
    nda::array<double, 1> w(nw);
    for (long i = 0; i < nw; ++i) w(i) = wmin + (wmax - wmin) * double(i) / double(nw - 1);
    return w;
  }
  void log() const {
    std::string e, er, sc, th;
    for (auto x : eta) e += std::to_string(x) + " ";
    for (auto x : eta_rel) er += std::to_string(x) + " ";
    for (auto x : scales) sc += std::to_string(x) + " ";
    for (auto x : theta_deg) th += std::to_string(x) + " ";
    app_log(1, "    optics: omega [{}, {}] Ha ({} points), eta = [ {}] Ha, eta_rel = [ {}], scales {}, K {}, q0 {}, finite q {}, "
               "flatter line(s) [ {}] deg ({} time grid), NNLS {} poles",
            wmin, wmax, nw, e, er, scales.empty() ? std::string("auto") : "[ " + sc + "]",
            K < 0 ? std::string("rule") : std::to_string(K), q0, finite_q, th, time_grid, nnls_n);
  }
};

/// MB + G estimates of one scalar odd response from its line data
struct response_fit_t {
  double delta = 0.0, theta = 0.0, tol_gram = 0.0;
  double delta_c = 0.0;      ///< residual of the complex-residue odd fit (diagnostic: delta >> delta_c = non-real data)
  long K = 0;
  std::vector<double> scales;
  rc::odd_measure_t fit;     ///< (nu, odd-fit residues)
  rc::mb_closure_t mb;
  rc::nnls_fit_t G;
  double t_fit = 0.0, t_mb = 0.0, t_G = 0.0;
};

inline response_fit_t close_response(nda::array<ComplexType, 1> const &zeta, nda::array<ComplexType, 1> const &data,
                                     nda::array<double, 1> const &nu, double theta_rad, optics_params_t const &p) {
  response_fit_t r;
  r.theta = theta_rad;
  auto t0 = std::chrono::steady_clock::now();
  auto f  = rc::odd_fit(zeta, data, nu);
  r.delta = std::max(p.delta_floor, rc::fit_residual(zeta, data, nu, f));
  r.fit   = rc::odd_measure_t(nu, f);
  r.delta_c = rc::odd_fit_residual_complex(zeta, data, nu);
  r.t_fit = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  t0      = std::chrono::steady_clock::now();
  double lo = nu(0), hi = nu(0);
  for (long j = 0; j < nu.size(); ++j) {
    lo = std::min(lo, nu(j));
    hi = std::max(hi, nu(j));
  }
  r.G   = rc::nnls_odd_fit(zeta, data, rc::nnls_grid(lo, hi, p.nnls_n));
  r.t_G = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  t0    = std::chrono::steady_clock::now();
  r.K        = p.K > 0 ? p.K : rc::k_rule(r.delta, theta_rad);
  r.tol_gram = rc::tol_gram_rule(r.delta);
  r.scales   = p.scales.empty() ? rc::auto_scales(r.G.measure.size() > 0 ? r.G.measure : r.fit, p.nscales) : p.scales;
  r.mb       = rc::mb_close(nu, f, r.scales, r.K, r.tol_gram);
  r.t_mb     = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return r;
}

/// all optics of one q (or q0), packed for the broadcast
struct optics_q_t {
  long iq = -1;   ///< -1: q -> 0; <= -2: q -> 0 with the extrapolation variant -2 - iq of optics_line_t
  long nb = 0, nw = 0;
  // (nbroad, nw) curves: primary (MB) and error bars (|MB - G|), and the consistency meters
  std::vector<std::string> qn = {"eps1", "eps2", "n", "kappa", "alpha", "R", "loss", "sigma1", "sigma2"};
  std::vector<nda::array<double, 2>> val, err;
  nda::array<double, 2> mis_eps, mis_loss, eps_R1_re, eps_R1_im;
  // scalars
  double eps_inf_h = 0, eps_inf_m = 0, eps_inf_Gh = 0, eps_inf_Gm = 0, fsum_h = 0, fsum_m = 0, fsum_Gh = 0, fsum_Gm = 0,
         fsum_mbh = 0, fsum_mbm = 0, delta_h = 0, delta_m = 0, tol_gram_h = 0, tol_gram_m = 0, wneg_h = 0, wneg_m = 0,
         G_resid_h = 0, G_resid_m = 0, time = 0, h_asym = 0, delta_c_h = 0, delta_c_m = 0;
  long K_h = 0, K_m = 0, nG_h = 0, nG_m = 0, nmb_h = 0, nmb_m = 0;
  std::vector<double> scales_h, scales_m;
  // pole sets: per scale (d, a) of both channels, NNLS (W, r) of both channels, odd fit residues of both channels
  std::vector<nda::array<double, 1>> mb_h_d, mb_h_a, mb_m_d, mb_m_a;
  nda::array<double, 1> G_h_W, G_h_r, G_m_W, G_m_r, fit_h, fit_m;
  nda::array<ComplexType, 1> h_nodes;

  // ---- packing (owner -> all) ----
  std::vector<double> pack() const {
    std::vector<double> v;
    auto put  = [&](double x) { v.push_back(x); };
    auto put1 = [&](nda::array<double, 1> const &a) {
      put(double(a.size()));
      for (auto x : a) put(x);
    };
    auto put2 = [&](nda::array<double, 2> const &a) {
      put(double(a.extent(0)));
      put(double(a.extent(1)));
      for (long i = 0; i < a.extent(0); ++i)
        for (long j = 0; j < a.extent(1); ++j) put(a(i, j));
    };
    put(double(iq)); put(double(nb)); put(double(nw));
    for (auto const &a : val) put2(a);
    for (auto const &a : err) put2(a);
    put2(mis_eps); put2(mis_loss); put2(eps_R1_re); put2(eps_R1_im);
    for (double x : {eps_inf_h, eps_inf_m, eps_inf_Gh, eps_inf_Gm, fsum_h, fsum_m, fsum_Gh, fsum_Gm, fsum_mbh, fsum_mbm, delta_h,
                     delta_m, tol_gram_h, tol_gram_m, wneg_h, wneg_m, G_resid_h, G_resid_m, time, h_asym, delta_c_h, delta_c_m})
      put(x);
    for (long x : {K_h, K_m, nG_h, nG_m, nmb_h, nmb_m}) put(double(x));
    for (auto const *s : {&scales_h, &scales_m}) {
      put(double(s->size()));
      for (double x : *s) put(x);
    }
    for (auto const *L : {&mb_h_d, &mb_h_a, &mb_m_d, &mb_m_a}) {
      put(double(L->size()));
      for (auto const &a : *L) put1(a);
    }
    for (auto const *a : {&G_h_W, &G_h_r, &G_m_W, &G_m_r, &fit_h, &fit_m}) put1(*a);
    put(double(h_nodes.size()));
    for (auto x : h_nodes) { put(x.real()); put(x.imag()); }
    return v;
  }
  void unpack(std::vector<double> const &v) {
    size_t o  = 0;
    auto get  = [&]() { return v[o++]; };
    auto getl = [&]() { return long(std::llround(v[o++])); };
    auto get1 = [&]() {
      nda::array<double, 1> a(getl());
      for (auto &x : a) x = get();
      return a;
    };
    auto get2 = [&]() {
      const long n0 = getl(), n1 = getl();
      nda::array<double, 2> a(n0, n1);
      for (long i = 0; i < n0; ++i)
        for (long j = 0; j < n1; ++j) a(i, j) = get();
      return a;
    };
    iq = getl(); nb = getl(); nw = getl();
    val.clear(); err.clear();
    for (size_t i = 0; i < qn.size(); ++i) val.push_back(get2());
    for (size_t i = 0; i < qn.size(); ++i) err.push_back(get2());
    mis_eps = get2(); mis_loss = get2(); eps_R1_re = get2(); eps_R1_im = get2();
    for (double *x : {&eps_inf_h, &eps_inf_m, &eps_inf_Gh, &eps_inf_Gm, &fsum_h, &fsum_m, &fsum_Gh, &fsum_Gm, &fsum_mbh, &fsum_mbm,
                      &delta_h, &delta_m, &tol_gram_h, &tol_gram_m, &wneg_h, &wneg_m, &G_resid_h, &G_resid_m, &time, &h_asym, &delta_c_h,
                      &delta_c_m})
      *x = get();
    for (long *x : {&K_h, &K_m, &nG_h, &nG_m, &nmb_h, &nmb_m}) *x = getl();
    for (auto *s : {&scales_h, &scales_m}) {
      s->resize(getl());
      for (auto &x : *s) x = get();
    }
    for (auto *L : {&mb_h_d, &mb_h_a, &mb_m_d, &mb_m_a}) {
      L->resize(getl());
      for (auto &a : *L) a = get1();
    }
    for (auto *a : {&G_h_W, &G_h_r, &G_m_W, &G_m_r, &fit_h, &fit_m}) *a = get1();
    h_nodes = nda::array<ComplexType, 1>(getl());
    for (auto &x : h_nodes) {
      const double re = get();
      x = ComplexType(re, get());
    }
  }
  long index(std::string const &q) const {
    for (size_t i = 0; i < qn.size(); ++i)
      if (qn[i] == q) return long(i);
    utils::check(false, "optics_q_t: unknown quantity {}", q);
    return -1;
  }
};

/**
 * The optics of one head h at the nodes: both channels closed (MB + G), observables on the grid for every broadening.
 * h_partner: h(-q) at the nodes for the asymmetry diagnostic (may be empty).
 */
inline optics_q_t optics_one(long iq, nda::array<ComplexType, 1> const &zeta, nda::array<double, 1> const &nu,
                             nda::array<ComplexType, 1> const &h, double theta_rad, optics_params_t const &p,
                             nda::array<ComplexType, 1> const *h_partner = nullptr) {
  const auto t0 = std::chrono::steady_clock::now();
  optics_q_t o;
  o.iq      = iq;
  o.h_nodes = h;
  const long nz = zeta.size();
  nda::array<ComplexType, 1> m(nz);
  for (long i = 0; i < nz; ++i) m(i) = h(i) / (1.0 + h(i));
  if (h_partner != nullptr and h_partner->size() == nz) {
    double d = 0.0, s = 0.0;
    for (long i = 0; i < nz; ++i) {
      d = std::max(d, std::abs(h(i) - (*h_partner)(i)));
      s = std::max(s, std::abs(h(i)));
    }
    o.h_asym = s > 0.0 ? d / s : d;
  }
  auto H = close_response(zeta, h, nu, theta_rad, p);
  auto M = close_response(zeta, m, nu, theta_rad, p);
  const auto w = p.omega();
  const long nw = w.size(), nb = p.nbroad();
  o.nb = nb;
  o.nw = nw;
  for (size_t i = 0; i < o.qn.size(); ++i) {
    o.val.emplace_back(nb, nw);
    o.err.emplace_back(nb, nw);
  }
  o.mis_eps   = nda::array<double, 2>(nb, nw);
  o.mis_loss  = nda::array<double, 2>(nb, nw);
  o.eps_R1_re = nda::array<double, 2>(nb, nw);
  o.eps_R1_im = nda::array<double, 2>(nb, nw);
  for (long b = 0; b < nb; ++b) {
    nda::array<ComplexType, 1> z(nw);
    for (long i = 0; i < nw; ++i) z(i) = ComplexType(w(i), p.eta_at(b, w(i)));
    auto hmb = H.mb(z), hG = H.G.measure(z), mmb = M.mb(z), mG = M.G.measure(z);
    nda::array<ComplexType, 1> e2(nw), e2G(nw), e1(nw);
    for (long i = 0; i < nw; ++i) {
      e2(i)  = 1.0 - mmb(i);
      e2G(i) = 1.0 - mG(i);
      e1(i)  = 1.0 / (1.0 + hmb(i));
    }
    auto Q  = rc::optical_quantities(e2, w);
    auto QG = rc::optical_quantities(e2G, w);
    for (size_t k = 0; k < o.qn.size(); ++k) {
      auto const &a = Q.get(o.qn[k]);
      auto const &g = QG.get(o.qn[k]);
      for (long i = 0; i < nw; ++i) {
        o.val[k](b, i) = a(i);
        o.err[k](b, i) = std::abs(a(i) - g(i));
      }
    }
    // the loss from the loss channel (R1): primary loss = -Im h_MB, error bar |Im (h_MB - h_G)|
    const long kl = o.index("loss");
    for (long i = 0; i < nw; ++i) {
      const double lr1 = -hmb(i).imag(), lr2 = Q.loss(i);
      o.val[kl](b, i)  = lr1;
      o.err[kl](b, i)  = std::abs(hmb(i).imag() - hG(i).imag());
      o.mis_eps(b, i)  = std::abs(e1(i) - e2(i));
      o.mis_loss(b, i) = std::abs(lr1 - lr2);
      o.eps_R1_re(b, i) = e1(i).real();
      o.eps_R1_im(b, i) = e1(i).imag();
    }
  }
  o.eps_inf_h  = 1.0 / (1.0 + H.fit.static_value());
  o.eps_inf_m  = 1.0 - M.fit.static_value();
  o.eps_inf_Gh = 1.0 / (1.0 + H.G.measure.static_value());
  o.eps_inf_Gm = 1.0 - M.G.measure.static_value();
  o.fsum_h     = H.fit.fsum();
  o.fsum_m     = M.fit.fsum();
  o.fsum_Gh    = H.G.measure.fsum();
  o.fsum_Gm    = M.G.measure.fsum();
  for (auto const &pp : H.mb.parts) o.fsum_mbh += pp.fsum() / double(H.mb.parts.size());
  for (auto const &pp : M.mb.parts) o.fsum_mbm += pp.fsum() / double(M.mb.parts.size());
  o.delta_h = H.delta; o.delta_m = M.delta;
  o.delta_c_h = H.delta_c; o.delta_c_m = M.delta_c;
  o.K_h = H.K; o.K_m = M.K;
  o.tol_gram_h = H.tol_gram; o.tol_gram_m = M.tol_gram;
  o.wneg_h = H.mb.max_wneg(); o.wneg_m = M.mb.max_wneg();
  o.nG_h = H.G.measure.size(); o.nG_m = M.G.measure.size();
  o.nmb_h = H.mb.npoles(); o.nmb_m = M.mb.npoles();
  o.G_resid_h = H.G.resid_rel; o.G_resid_m = M.G.resid_rel;
  o.scales_h = H.scales; o.scales_m = M.scales;
  for (auto const &pp : H.mb.parts) { o.mb_h_d.push_back(pp.d); o.mb_h_a.push_back(pp.a); }
  for (auto const &pp : M.mb.parts) { o.mb_m_d.push_back(pp.d); o.mb_m_a.push_back(pp.a); }
  o.G_h_W = H.G.measure.W; o.G_h_r = H.G.measure.r; o.G_m_W = M.G.measure.W; o.G_m_r = M.G.measure.r;
  o.fit_h = H.fit.r; o.fit_m = M.fit.r;
  o.time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return o;
}

/// the heads of one line (nodes, poles, per-q heads, q -> 0 weights) and where they came from
struct optics_line_t {
  double theta_deg = 0.0;
  std::string source;                  ///< "checkpoint iter<N>" | "last iteration" | "recomputed" | "flat pass"
  std::string time_grid;
  nda::array<ComplexType, 1> zeta;
  nda::array<double, 1> nu;
  nda::array<ComplexType, 2> h_nodes;  ///< (nq, nz)
  nda::array<double, 1> q_weights;     ///< (nq) q -> 0 extrapolation weights
  nda::array<double, 2> qpts;          ///< (nq, 3) Cartesian (bohr^-1)
  nda::array<double, 2> lattv;         ///< (3, 3) lattice vectors (bohr; rows), for the q directions
  std::vector<double> qfac;            ///< (nq) f(q) = Omega |q|^2 / 4 pi (0: Gamma)
  std::vector<long> qminus;
  std::vector<std::string> variant_names;            ///< other q -> 0 variants (optics_params_t::q0_variants)
  std::vector<nda::array<double, 1>> variant_weights;
  double pass_time = 0.0;              ///< cost of the head pass (0: from the checkpoint)
  long nt_p = 0, nt_h = 0;
};

/// h5 group / log label of a job: q0, q0_<variant>, iq<n>
inline std::string optics_label(optics_line_t const &L, long iq) {
  if (iq == -1) return "q0";
  if (iq <= -2) return "q0_" + L.variant_names[-2 - iq];
  return "iq" + std::to_string(iq);
}

/// valence plasma frequency squared 4 pi N / Omega (a.u.)
inline double wp2_valence(double nelec, double volume) { return 4.0 * std::numbers::pi * nelec / volume; }

/**
 * Optics of one line for q0 and the finite q (q distributed round-robin, owner broadcasts), root writes
 * <file>:/optics/<tag>/{q0 (the q -> 0 extrapolation), q0_<variant> (other extrapolation variants), iq<n> (mesh q n)}. Collective.
 */
inline std::vector<optics_q_t> optics_line(boost::mpi3::communicator &comm, optics_line_t const &L, optics_params_t const &p) {
  const long nq = L.h_nodes.extent(0), nz = L.zeta.size();
  const double th = L.theta_deg * std::numbers::pi / 180.0;
  std::vector<long> jobs;   // -1 = q0, <= -2: q0 of the variant -2 - job
  if (p.q0) jobs.push_back(-1);
  if (p.q0)
    for (long v = 0; v < long(L.variant_names.size()); ++v) jobs.push_back(-2 - v);
  if (p.finite_q)
    for (long q = 0; q < nq; ++q)
      if (L.qfac[q] > 0.0) jobs.push_back(q);
  nda::array<ComplexType, 1> h0(nz);
  h0() = ComplexType(0.0);
  for (long q = 0; q < nq; ++q)
    if (L.q_weights(q) != 0.0) h0 += L.q_weights(q) * L.h_nodes(q, nda::range::all);
  std::vector<optics_q_t> out(jobs.size());
  // 1. every rank computes its own jobs (j mod np), 2. the owners broadcast (computing inside the broadcast loop would
  //    serialize the jobs: si444 S9b runs, 67 curves in 150-720 s instead of the max per-curve time)
  for (size_t j = 0; j < jobs.size(); ++j) {
    if (comm.rank() != int(j % comm.size())) continue;
    {
      const long q = jobs[j];
      if (q == -1) out[j] = optics_one(-1, L.zeta, L.nu, h0, th, p);
      else if (q <= -2) {
        auto const &c = L.variant_weights[-2 - q];
        nda::array<ComplexType, 1> hv(nz);
        hv() = ComplexType(0.0);
        for (long qq = 0; qq < nq; ++qq)
          if (c(qq) != 0.0) hv += c(qq) * L.h_nodes(qq, nda::range::all);
        out[j] = optics_one(q, L.zeta, L.nu, hv, th, p);
      } else {
        nda::array<ComplexType, 1> hq(L.h_nodes(q, nda::range::all)), hm(L.h_nodes(L.qminus[q], nda::range::all));
        out[j] = optics_one(q, L.zeta, L.nu, hq, th, p, &hm);
      }
    }
  }
  for (size_t j = 0; j < jobs.size(); ++j) {
    const int owner = int(j % comm.size());
    std::vector<double> buf;
    if (comm.rank() == owner) buf = out[j].pack();
    long n = long(buf.size());
    comm.broadcast_n(&n, 1, owner);
    buf.resize(n);
    comm.broadcast_n(buf.data(), n, owner);
    if (comm.rank() != owner) out[j].unpack(buf);
  }
  return out;
}

/// root writes /optics/<tag>/ (+ the common grid at /optics/omega etc. if absent)
inline void write_optics(boost::mpi3::communicator &comm, std::string const &file, std::string const &tag, optics_line_t const &L,
                         std::vector<optics_q_t> const &R, optics_params_t const &p, double nelec, double volume,
                         std::string const &extra_scope = "") {
  if (comm.root()) {
    h5::file f(file, 'a');
    h5::group g(f);
    auto og = g.has_subgroup("optics") ? g.open_group("optics") : g.create_group("optics");
    if (not og.has_dataset("omega")) {
      auto w = p.omega();
      nda::h5_write(og, "omega", w, false);
      nda::array<double, 1> wev(w.size());
      for (long i = 0; i < w.size(); ++i) wev(i) = w(i) * rc::HA_EV;
      nda::h5_write(og, "omega_eV", wev, false);
      nda::array<double, 1> e(p.eta.size()), er(p.eta_rel.size());
      for (size_t i = 0; i < p.eta.size(); ++i) e(i) = p.eta[i];
      for (size_t i = 0; i < p.eta_rel.size(); ++i) er(i) = p.eta_rel[i];
      nda::h5_write(og, "eta", e, false);
      nda::h5_write(og, "eta_rel", er, false);
      h5::h5_write_attribute(og, "broadening_order", std::string("rows of every (nbroad, nw) curve: the constant eta first, then eta = "
                                                                 "eta_rel x omega"));
      h5::h5_write_attribute(og, "units",
                             std::string("omega, eta in Ha (omega_eV in eV); eps1, eps2, n, kappa, R, loss dimensionless; alpha in "
                                         "1/bohr (alpha_cm in cm^-1 = alpha x 1.8897261e8); sigma1, sigma2 in atomic units "
                                         "e^2/(hbar a0) (x 4.599848e4 = S/cm); c = 137.035999"));
      h5::h5_write_attribute(og, "physics_scope",
                             std::string("RPA polarization built from the self-consistent G of the line (no vertex corrections: no "
                                         "excitons, absorption onset = direct QP gap); head eps^-1_00 with local fields from the "
                                         "Dyson inversion of W; coarse k mesh: eps2 is a sum of few broadened transitions; q -> 0 "
                                         "from the extrapolation of the finite-q heads (variant dependent). ") +
                                 extra_scope);
      h5::h5_write_attribute(og, "method",
                             std::string("loss from the closure of h = eps^-1 - 1 (R1), eps and derived quantities from the closure "
                                         "of m = 1 - eps (R2); primary = MB (multi-scale Cayley closure centred at the line "
                                         "crossing, cos^2 blend), error bars *_err = |Q(MB) - Q(NNLS)| of the same channel; "
                                         "mismatch_eps = |eps_R1 - eps_R2|, mismatch_loss = |L_R1 - L_R2|; eps_inf and f-sums from "
                                         "the line fits"));
      h5::h5_write(og, "wp2_valence", wp2_valence(nelec, volume));
      h5::h5_write(og, "nelec", nelec);
      h5::h5_write(og, "volume", volume);
    }
    if (og.has_subgroup(tag)) og.unlink(tag);
    auto tg = og.create_group(tag);
    h5::h5_write(tg, "theta_deg", L.theta_deg);
    h5::h5_write(tg, "source", L.source);
    h5::h5_write(tg, "time_grid", L.time_grid);
    h5::h5_write(tg, "pass_time", L.pass_time);
    h5::h5_write(tg, "nt_pi_p", L.nt_p);
    h5::h5_write(tg, "nt_pi_h", L.nt_h);
    nda::h5_write(tg, "zeta", L.zeta, false);
    nda::h5_write(tg, "nu", L.nu, false);
    nda::h5_write(tg, "h_nodes", L.h_nodes, false);
    nda::h5_write(tg, "q_weights", L.q_weights, false);
    nda::h5_write(tg, "qpts", L.qpts, false);
    if (L.lattv.size() > 0) nda::h5_write(tg, "lattv", L.lattv, false);
    for (auto const &o : R) {
      auto qg = tg.create_group(optics_label(L, o.iq));
      h5::h5_write(qg, "iq", o.iq);
      for (size_t k = 0; k < o.qn.size(); ++k) {
        nda::h5_write(qg, o.qn[k], o.val[k], false);
        nda::h5_write(qg, o.qn[k] + "_err", o.err[k], false);
      }
      {
        nda::array<double, 2> acm(o.val[o.index("alpha")]), acme(o.err[o.index("alpha")]);
        acm *= rc::ALPHA_AU_TO_CM;
        acme *= rc::ALPHA_AU_TO_CM;
        nda::h5_write(qg, "alpha_cm", acm, false);
        nda::h5_write(qg, "alpha_cm_err", acme, false);
      }
      nda::h5_write(qg, "mismatch_eps", o.mis_eps, false);
      nda::h5_write(qg, "mismatch_loss", o.mis_loss, false);
      nda::h5_write(qg, "eps1_R1", o.eps_R1_re, false);
      nda::h5_write(qg, "eps2_R1", o.eps_R1_im, false);
      nda::h5_write(qg, "h_nodes", o.h_nodes, false);
      h5::h5_write(qg, "eps_inf", o.eps_inf_h);
      h5::h5_write(qg, "eps_inf_m", o.eps_inf_m);
      h5::h5_write(qg, "eps_inf_nnls_h", o.eps_inf_Gh);
      h5::h5_write(qg, "eps_inf_nnls_m", o.eps_inf_Gm);
      h5::h5_write(qg, "fsum_h", o.fsum_h);
      h5::h5_write(qg, "fsum_m", o.fsum_m);
      h5::h5_write(qg, "fsum_nnls_h", o.fsum_Gh);
      h5::h5_write(qg, "fsum_nnls_m", o.fsum_Gm);
      h5::h5_write(qg, "fsum_mb_h", o.fsum_mbh);
      h5::h5_write(qg, "fsum_mb_m", o.fsum_mbm);
      h5::h5_write(qg, "delta_h", o.delta_h);
      h5::h5_write(qg, "delta_m", o.delta_m);
      h5::h5_write(qg, "delta_complex_fit_h", o.delta_c_h);
      h5::h5_write(qg, "delta_complex_fit_m", o.delta_c_m);
      h5::h5_write(qg, "K_h", o.K_h);
      h5::h5_write(qg, "K_m", o.K_m);
      h5::h5_write(qg, "tol_gram_h", o.tol_gram_h);
      h5::h5_write(qg, "tol_gram_m", o.tol_gram_m);
      h5::h5_write(qg, "wneg_h", o.wneg_h);
      h5::h5_write(qg, "wneg_m", o.wneg_m);
      h5::h5_write(qg, "nnls_resid_h", o.G_resid_h);
      h5::h5_write(qg, "nnls_resid_m", o.G_resid_m);
      h5::h5_write(qg, "h_asymmetry", o.h_asym);
      h5::h5_write(qg, "time", o.time);
      auto sv = [](std::vector<double> const &v) {
        nda::array<double, 1> a(v.size());
        for (size_t i = 0; i < v.size(); ++i) a(i) = v[i];
        return a;
      };
      nda::h5_write(qg, "scales_h", sv(o.scales_h), false);
      nda::h5_write(qg, "scales_m", sv(o.scales_m), false);
      auto pg = qg.create_group("poles");
      for (size_t i = 0; i < o.mb_h_d.size(); ++i) {
        nda::h5_write(pg, "mb_h_d" + std::to_string(i), o.mb_h_d[i], false);
        nda::h5_write(pg, "mb_h_a" + std::to_string(i), o.mb_h_a[i], false);
      }
      for (size_t i = 0; i < o.mb_m_d.size(); ++i) {
        nda::h5_write(pg, "mb_m_d" + std::to_string(i), o.mb_m_d[i], false);
        nda::h5_write(pg, "mb_m_a" + std::to_string(i), o.mb_m_a[i], false);
      }
      nda::h5_write(pg, "nnls_h_W", o.G_h_W, false);
      nda::h5_write(pg, "nnls_h_r", o.G_h_r, false);
      nda::h5_write(pg, "nnls_m_W", o.G_m_W, false);
      nda::h5_write(pg, "nnls_m_r", o.G_m_r, false);
      nda::h5_write(pg, "fit_h", o.fit_h, false);
      nda::h5_write(pg, "fit_m", o.fit_m, false);
    }
  }
  comm.barrier();
}

/// one log line per q (level 1 for q0, 2 for the finite q)
inline void log_optics(optics_line_t const &L, std::vector<optics_q_t> const &R, double wp2) {
  for (auto const &o : R) {
    app_log(o.iq == -1 ? 1 : 2,
            "  optics {} deg {}: eps_inf {:.6f} (m channel {:.6f}, NNLS {:.6f}/{:.6f}); f-sum h {:.5f} m {:.5f} (NNLS {:.5f}, MB "
            "{:.5f}; valence 4 pi n {:.5f}); delta h {:.1e} m {:.1e} (complex fit {:.1e}) -> K {}/{}; NNLS {}/{} poles (resid {:.1e}/{:.1e}); MB {}/{} "
            "poles (w(d<=0) {:.1e}/{:.1e}); |h(q)-h(-q)| {:.1e}; {:.2f} s",
            L.theta_deg, optics_label(L, o.iq), o.eps_inf_h, o.eps_inf_m, o.eps_inf_Gh,
            o.eps_inf_Gm, o.fsum_h, o.fsum_m, o.fsum_Gh, o.fsum_mbh, wp2, o.delta_h, o.delta_m, o.delta_c_h, o.K_h, o.K_m, o.nG_h, o.nG_m,
            o.G_resid_h, o.G_resid_m, o.nmb_h, o.nmb_m, o.wneg_h, o.wneg_m, o.h_asym, o.time);
  }
}

} // namespace methods::gw_line

#endif
