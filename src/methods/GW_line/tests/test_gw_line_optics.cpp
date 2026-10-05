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

/**
 * S9b: the kernels on FLATTER bosonic lines and the optics from the line, on the THC fixtures with KS G (setup of
 * [V2]/[V6]: nIpts = 8 nbnd, mu mid-gap, lam_b = max(4, 1.2 x largest transition), bos_gap = KS gap / 2, eps = 1e-10),
 * exact references from the Casida solution (gw_line_casida_ref.hpp).
 *
 * [V6][flat] for theta_b = 20, 10, 5 deg (time rays at theta_b / 2) and both time grids (GL rays, time-node ID):
 *   Pi(q, zeta_i) at the bosonic nodes vs the exact transition sum, W(q, zeta_i) vs the Casida W_dyn, the head
 *   h(q, zeta_i) vs the exact head, for one (pair-closed) group of q per fixture (memory: N_zeta ~ 1200 at 5 deg);
 *   head_pass (all q, its own q groups) vs the explicit chain. Gates: <= 1e-9 (relative to max) at every angle that passes;
 *   the bosonic rank, node and time-node counts and the kernel times are printed.
 * [V6][optics] see run_optics below (20 / 10 deg; [.optics5]: 5 deg); gates in optics_gates.
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <format>
#include <map>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include "mpi3/communicator.hpp"
#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "nda/linalg.hpp"

#include "mean_field/default_MF.hpp"
#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/thc_reader_t.hpp"

#include "numerics/line_dlr/time_ray.hpp"
#include "numerics/line_dlr/bosonic_basis.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/polarization.hpp"
#include "methods/GW_line/screened.hpp"
#include "methods/GW_line/head.hpp"
#include "methods/GW_line/head_pass.hpp"
#include "methods/GW_line/optics.hpp"
#include "gw_line_casida_ref.hpp"

namespace {

using namespace methods::gw_line;
using numerics::line_dlr::sector_t;
using numerics::line_dlr::bosonic_basis_t;
using namespace gw_line_test;

constexpr double deg = std::numbers::pi / 180.0;

struct osetup_t {
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;
  std::shared_ptr<mf::MF> mf;
  std::unique_ptr<methods::thc_reader_t> thc;
  long nk = 0, nq = 0, nb = 0, Np = 0;
  nda::array<double, 2> eig, e_rel;
  double mu = 0, ks_gap = 0, etr_max = 0;
  pole_data_t poles;

  explicit osetup_t(std::string const &fixture) {
    mpi = utils::make_unit_test_mpi_context();
    mf  = std::make_shared<mf::MF>(mf::default_MF(mpi, fixture));
    utils::check(mf->nkpts() == mf->nkpts_ibz() and mf->nqpts() == mf->nqpts_ibz(), "{}: nosym fixture required", fixture);
    thc = std::make_unique<methods::thc_reader_t>(
        mf, methods::make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft", 1e-10, mf->ecutrho(), 1, 1024));
    nk = mf->nkpts(); nq = mf->nqpts(); nb = thc->nbnd(); Np = thc->Np();
    eig = nda::array<double, 2>(nk, nb);
    double omax = 0.0;
    for (long ik = 0; ik < nk; ++ik)
      for (long n = 0; n < nb; ++n) {
        eig(ik, n) = mf->eigval()(0, ik, n);
        omax       = std::max(omax, double(mf->occ()(0, ik, n)));
      }
    double homo = -1e300, lumo = 1e300, emin_occ = 1e300, emax_vir = -1e300;
    for (long ik = 0; ik < nk; ++ik)
      for (long n = 0; n < nb; ++n) {
        if (mf->occ()(0, ik, n) > 0.5 * omax) { homo = std::max(homo, eig(ik, n)); emin_occ = std::min(emin_occ, eig(ik, n)); }
        else { lumo = std::min(lumo, eig(ik, n)); emax_vir = std::max(emax_vir, eig(ik, n)); }
      }
    mu = 0.5 * (homo + lumo); ks_gap = lumo - homo; etr_max = emax_vir - emin_occ;
    poles = pole_data_t::from_ks(eig, mu);
    e_rel = nda::array<double, 2>(nk, nb);
    for (long ik = 0; ik < nk; ++ik)
      for (long n = 0; n < nb; ++n) e_rel(ik, n) = eig(ik, n) - mu;
  }
};

/// exact Pi(q, zeta) block (both sectors) for one q: sum_t s_t S_t S_t^dagger / (zeta - E_t)   (casida_t transitions)
nda::array<ComplexType, 3> pi_exact_q(casida_t const &c, aux_grid_t const &g, nda::array<ComplexType, 1> const &z) {
  const long nz = z.size(), T = c.T;
  nda::array<ComplexType, 3> out(nz, g.nP, g.nQ);
  cmat A(g.nP, T);
  cmat B(c.S(g.Q_rng(), nda::range::all));
  for (long iz = 0; iz < nz; ++iz) {
    for (long t = 0; t < T; ++t) {
      const ComplexType f = c.sg[t] / (z(iz) - c.E[t]);
      for (long P = 0; P < g.nP; ++P) A(P, t) = c.S(g.P0 + P, t) * f;
    }
    nda::matrix_view<ComplexType> o(out(iz, nda::range::all, nda::range::all));
    nda::blas::gemm(ComplexType(1.0), A, nda::dagger(B), ComplexType(0.0), o);
  }
  return out;
}

/// exact head of q at z: f(q) sum_s lam_s |cb^T v_s|^2 / (z - lam_s)
nda::array<ComplexType, 1> head_exact_q(casida_t const &c, methods::thc_reader_t &thc, double fac, long iq,
                                        nda::array<ComplexType, 1> const &z) {
  auto cb = thc.basis_bar_head();
  const long Np = c.V.extent(0);
  nda::array<double, 1> g2(c.T);
  for (long s = 0; s < c.T; ++s) {
    ComplexType a(0.0);
    for (long P = 0; P < Np; ++P) a += cb(iq, P) * c.V(P, s);
    g2(s) = std::norm(a);
  }
  nda::array<ComplexType, 1> o(z.size());
  for (long i = 0; i < z.size(); ++i) {
    ComplexType a(0.0);
    for (long s = 0; s < c.T; ++s) a += c.lam(s) * g2(s) / (z(i) - c.lam(s));
    o(i) = fac * a;
  }
  return o;
}

double maxabs(auto const &a) {
  double m = 0.0;
  for (auto const &v : a) m = std::max(m, std::abs(v));
  return m;
}

/// one fixture: the kernels at theta_b in thetas with both time grids; returns nothing, gates inside
void run_flat(std::string const &fixture, std::vector<double> const &thetas, std::vector<std::string> const &grids, bool check_pass) {
  osetup_t su(fixture);
  auto &mpi  = *su.mpi;
  auto &comm = mpi.comm;
  auto &thc  = *su.thc;
  auto &mf   = *su.mf;
  auto all   = nda::range::all;
  const long nq = su.nq, Np = su.Np;
  const double lam_b = std::max(4.0, 1.2 * su.etr_max), gap_b = 0.5 * su.ks_gap;
  aux_grid_t grid(mpi, Np);
  head_basis_t hb(thc, mf, grid);
  propagator_t<HOST_MEMORY> prop(thc, grid);
  // the q group of the explicit chain: the first pair-closed group of size <= 2 with a q != Gamma
  const q_groups_t qg2(nq, 2, qminus_list(mf));
  long Gsel = 0;
  for (long G = 0; G < qg2.n; ++G) {
    bool nonzero = false;
    for (long q : qg2.rows(G)) nonzero = nonzero or (hb.fac(q) > 0.0);
    if (nonzero) { Gsel = G; break; }
  }
  auto const &qsel = qg2.rows(Gsel);
  app_log(2, "\n[V6][flat] {} ({} ranks): nk={} nq={} nb={} Np={}, KS gap {:.4f} Ha, lam_b {:.3f}, bos_gap {:.4f}; explicit chain on q = "
             "{}{}",
          fixture, comm.size(), su.nk, nq, su.nb, Np, su.ks_gap, lam_b, gap_b, qsel[0], qsel.size() > 1 ? ", " + std::to_string(qsel[1]) : "");
  // Casida of the selected q (lockstep thc.Z over the group on every rank)
  std::vector<casida_t> cas;
  for (long q : qsel) {
    cmat Zq(thc.Z(int(q)));
    cas.push_back(casida_q(thc, mf, su.e_rel, q, Zq));
  }
  for (double th : thetas) {
    for (std::string const &tg : grids) {
      head_pass_params_t hp;
      hp.theta_deg = th;
      hp.lam_b     = lam_b;
      hp.eps       = 1e-10;
      hp.bos_gap   = gap_b;
      hp.time_grid = tg;
      hp.t_chunk   = 8;
      hp.qgroup    = 2;
      auto g = head_pass_grid(su.poles, hp, comm);
      auto const &bos = *g.bos;
      auto const &zb  = bos.zeta_nodes;
      const long nz   = zb.size();
      utils::TimerManager T;
      memory::array<HOST_MEMORY, ComplexType, 4> Pi, w, Wn;
      auto t0 = std::chrono::steady_clock::now();
      polarization<HOST_MEMORY>(prop, su.poles, mf, grid, zb, *g.pi_p, *g.pi_h, 8, Pi, T, sector_t::both, qsel);
      const double tPi = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      // Pi vs exact
      double ep = 0.0, sp = 0.0;
      for (long i = 0; i < long(qsel.size()); ++i) {
        auto Pex = pi_exact_q(cas[i], grid, zb);
        nda::array<ComplexType, 3> Pl(Pi(i, all, all, all));
        ep = std::max(ep, max_diff3(Pl, Pex));
        sp = std::max(sp, max_abs3(Pex));
      }
      ep = comm.all_reduce_value(ep, mpi3::max<>{});
      sp = comm.all_reduce_value(sp, mpi3::max<>{});
      // W at the nodes
      const q_groups_t qg_all(nq, 2, qminus_list(mf));
      utils::TimerManager Tw;
      coulomb_blocks_t<HOST_MEMORY> Zb(thc, grid, qg_all.dyson_q_list(comm.size(), comm.rank(), nz, Np), Tw);
      t0 = std::chrono::steady_clock::now();
      screened_interaction<HOST_MEMORY>(Pi, Zb, bos, grid, mpi, w, Tw, &Wn, qsel, false);
      const double tW = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      double ew = 0.0, sw = 0.0;
      for (long i = 0; i < long(qsel.size()); ++i) {
        auto Wex = casida_eval(cas[i], grid, zb, false);
        nda::array<ComplexType, 3> Wl(Wn(i, all, all, all));
        ew = std::max(ew, max_diff3(Wl, Wex));
        sw = std::max(sw, max_abs3(Wex));
      }
      ew = comm.all_reduce_value(ew, mpi3::max<>{});
      sw = comm.all_reduce_value(sw, mpi3::max<>{});
      // head
      nda::array<ComplexType, 2> Hn;
      head_nodes_partial<HOST_MEMORY>(Wn, qsel, hb, Hn);
      head_reduce(comm, {&Hn});
      double eh = 0.0, sh = 0.0;
      for (long i = 0; i < long(qsel.size()); ++i) {
        const long q = qsel[i];
        if (hb.fac(q) == 0.0) continue;
        auto hx = head_exact_q(cas[i], thc, hb.fac(q), q, zb);
        eh = std::max(eh, maxabs(nda::array<ComplexType, 1>(Hn(q, all) - hx)));
        sh = std::max(sh, maxabs(hx));
      }
      // head_pass (all q) vs the explicit chain on the selected q
      head_pass_out_t hpo;
      double ehp = 0.0;
      if (check_pass) {
        hpo = head_pass<HOST_MEMORY>(thc, mf, mpi, grid, prop, su.poles, hb, hp);
        for (long q : qsel) ehp = std::max(ehp, maxabs(nda::array<ComplexType, 1>(hpo.h_nodes(q, all) - Hn(q, all))));
      }
      app_log(2, "  [V6][flat] {} theta_b {:4.1f} deg, {} time grid: bosonic rank {} ({} nodes), Pi time nodes {} + {}", fixture, th,
              tg, bos.rank, nz, g.pi_p->size(), g.pi_h->size());
      app_log(2, "      Pi vs exact {:.2e}, W vs Casida {:.2e}, head vs exact {:.2e} (max|h| {:.3e}); head_pass (all {} q, {} groups) vs "
                 "chain {:.1e}",
              ep / sp, ew / sw, eh / sh, sh, nq, hpo.ngroups, ehp / sh);
      app_log(2, "      cost ({} q, rank 0): Pi {:.2f} s, W {:.2f} s; head_pass all q {:.2f} s (grid {:.2f}, Z {:.2f}, Pi {:.2f}, W {:.2f})",
              qsel.size(), tPi, tW, hpo.t_total, hpo.t_grid, hpo.t_Z, hpo.t_Pi, hpo.t_W);
      if (tg == "id" or th >= 20.0) {   // the GL rays need the flat-scaled panels (head_pass auto); gated: ID everywhere, GL at 20
        REQUIRE(ep / sp <= 1e-9);
        REQUIRE(ew / sw <= 1e-9);
        REQUIRE(eh / sh <= 1e-9);
      }
      REQUIRE(ehp / sh <= 1e-12);
    }
  }
}

/// window errors of a real curve vs the exact one: max |a - x| / max |x| per eV window (0-5, 5-10, 10-20, 20-40)
std::array<double, 4> werr(nda::array<double, 1> const &w, nda::array<double, 1> const &a, nda::array<double, 1> const &x) {
  const double W[4][2] = {{0, 5}, {5, 10}, {10, 20}, {20, 40}};
  std::array<double, 4> e{};
  for (int k = 0; k < 4; ++k) {
    double d = 0.0, s = 0.0;
    for (long i = 0; i < w.size(); ++i) {
      const double ev = w(i) * rc::HA_EV;
      if (ev <= W[k][0] or ev > W[k][1]) continue;
      d = std::max(d, std::abs(a(i) - x(i)));
      s = std::max(s, std::abs(x(i)));
    }
    e[k] = s > 0.0 ? d / s : 0.0;
  }
  return e;
}

/// exact odd measure of the head of q (fac lam |cb^T v|^2 at lam > 0 and the pairing with lam < 0 for odd functions)
struct exact_head_t {
  std::vector<double> lam, g2;   // all Casida poles (both signs) and fac |cb^T v|^2
  ComplexType operator()(ComplexType z) const {
    ComplexType a(0.0);
    for (size_t s = 0; s < lam.size(); ++s) a += lam[s] * g2[s] / (z - lam[s]);
    return a;
  }
  double h0() const {
    double a = 0.0;
    for (size_t s = 0; s < lam.size(); ++s) a -= g2[s];
    return a;
  }
  double fsum() const {
    double a = 0.0;
    for (size_t s = 0; s < lam.size(); ++s) a += lam[s] * lam[s] * g2[s];
    return a;
  }
};

struct optics_stats_t {
  // [method 0 MB / 1 NNLS][quantity 0 loss / 1 eps1 / 2 eps2][broadening 0 eta 0.01 / 1 eta 0.05 w][window] -> errors over q
  std::vector<double> e[2][3][2][4];
  long n_cov = 0, n_cov_ok = 0, n_cov_3 = 0;
  std::vector<double> d_einf, d_fsum;
};

/**
 * [V6][optics] one fixture: for each theta_b, h(q, zeta_i) at the nodes from head_pass (ID time grid), the optics of every
 * finite q and of q0 (the extrapolation weights applied to the line heads and to the exact heads alike) vs the exact
 * eps^-1_00 = 1 + h_exact from Casida: loss (R1), eps1 and eps2 (R2) per window, MB and NNLS, eta = 0.01 Ha and 0.05 omega;
 * error-bar coverage (stored |MB - NNLS| >= true MB error, per window); eps_inf(q) and the f-sum from the fits vs exact.
 */
std::map<double, optics_stats_t> run_optics(std::string const &fixture, std::vector<double> const &thetas) {
  osetup_t su(fixture);
  auto &mpi  = *su.mpi;
  auto &comm = mpi.comm;
  auto &thc  = *su.thc;
  auto &mf   = *su.mf;
  auto all   = nda::range::all;
  const long nq = su.nq, Np = su.Np;
  const double lam_b = std::max(4.0, 1.2 * su.etr_max), gap_b = 0.5 * su.ks_gap;
  aux_grid_t grid(mpi, Np);
  head_basis_t hb(thc, mf, grid);
  propagator_t<HOST_MEMORY> prop(thc, grid);
  head_extrapolation_t hx(mf, "gygi");
  // exact heads of every q (lockstep Z)
  std::vector<exact_head_t> ex(nq);
  {
    auto cb = thc.basis_bar_head();
    for (long q = 0; q < nq; ++q) {
      cmat Zq(thc.Z(int(q)));
      if (hb.fac(q) == 0.0) continue;
      auto c = casida_q(thc, mf, su.e_rel, q, Zq);
      for (long s = 0; s < c.T; ++s) {
        ComplexType a(0.0);
        for (long P = 0; P < Np; ++P) a += cb(q, P) * c.V(P, s);
        ex[q].lam.push_back(c.lam(s));
        ex[q].g2.push_back(hb.fac(q) * std::norm(a));
      }
    }
  }
  optics_params_t op;
  op.enable  = true;
  op.wmin    = 0.0;
  op.wmax    = 40.0 / rc::HA_EV;
  op.nw      = 2001;
  op.eta     = {0.01};
  op.eta_rel = {0.05};
  const auto w = op.omega();
  std::map<double, optics_stats_t> stats;
  for (double th : thetas) {
    head_pass_params_t hp;
    hp.theta_deg = th;
    hp.lam_b     = lam_b;
    hp.eps       = 1e-10;
    hp.bos_gap   = gap_b;
    hp.time_grid = "id";
    hp.t_chunk   = 8;
    auto hpo = head_pass<HOST_MEMORY>(thc, mf, mpi, grid, prop, su.poles, hb, hp);
    auto &st = stats[th];
    double t_opt = 0.0;
    std::vector<long> jobs = {-1};
    for (long q = 0; q < nq; ++q)
      if (hb.fac(q) > 0.0) jobs.push_back(q);
    for (long q : jobs) {
      // line data and exact function (q0: the same extrapolation weights)
      nda::array<ComplexType, 1> h(hpo.nz);
      std::vector<std::pair<double, long>> wq;   // (weight, q)
      if (q < 0) {
        h = hx.apply(hpo.h_nodes);
        for (long p = 0; p < nq; ++p)
          if (hx.c(p) != 0.0 and hb.fac(p) > 0.0) wq.push_back({hx.c(p), p});
      } else {
        h = hpo.h_nodes(q, all);
        wq.push_back({1.0, q});
      }
      auto hex = [&](ComplexType z) {
        ComplexType a(0.0);
        for (auto [c, p] : wq) a += c * ex[p](z);
        return a;
      };
      double h0x = 0.0, fsx = 0.0;
      for (auto [c, p] : wq) {
        h0x += c * ex[p].h0();
        fsx += c * ex[p].fsum();
      }
      const auto t0 = std::chrono::steady_clock::now();
      auto o = optics_one(q, hpo.zeta, hpo.nu, h, th * deg, op);
      nda::array<ComplexType, 1> m(hpo.nz);
      for (long i = 0; i < hpo.nz; ++i) m(i) = h(i) / (1.0 + h(i));
      auto H = close_response(hpo.zeta, h, hpo.nu, th * deg, op);
      auto M = close_response(hpo.zeta, m, hpo.nu, th * deg, op);
      t_opt += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      for (long b = 0; b < 2; ++b) {
        nda::array<ComplexType, 1> z(w.size());
        for (long i = 0; i < w.size(); ++i) z(i) = ComplexType(w(i), op.eta_at(b, w(i)));
        auto hmb = H.mb(z), hG = H.G.measure(z), mmb = M.mb(z), mG = M.G.measure(z);
        nda::array<double, 1> Lx(w.size()), e1x(w.size()), e2x(w.size());
        nda::array<double, 1> L[2], e1[2], e2[2];
        for (int k = 0; k < 2; ++k) L[k] = e1[k] = e2[k] = nda::array<double, 1>(w.size());
        for (long i = 0; i < w.size(); ++i) {
          const ComplexType hz = hex(z(i)), epx = 1.0 / (1.0 + hz);
          Lx(i)  = -hz.imag();
          e1x(i) = epx.real();
          e2x(i) = epx.imag();
          L[0](i) = -hmb(i).imag();
          L[1](i) = -hG(i).imag();
          e1[0](i) = 1.0 - mmb(i).real();
          e2[0](i) = -mmb(i).imag();
          e1[1](i) = 1.0 - mG(i).real();
          e2[1](i) = -mG(i).imag();
        }
        for (int k = 0; k < 2; ++k) {
          std::array<double, 4> er[3] = {werr(w, L[k], Lx), werr(w, e1[k], e1x), werr(w, e2[k], e2x)};
          for (int qq = 0; qq < 3; ++qq)
            for (int win = 0; win < 4; ++win) st.e[k][qq][b][win].push_back(er[qq][win]);
        }
        // coverage of the stored error bar (optics_one: |MB - NNLS|) per window
        const long kq[3] = {o.index("loss"), o.index("eps1"), o.index("eps2")};
        nda::array<double, 1> const *xs[3] = {&Lx, &e1x, &e2x};
        for (int qq = 0; qq < 3; ++qq) {
          nda::array<double, 1> v(o.val[kq[qq]](b, all)), eb(o.err[kq[qq]](b, all));
          const double W[4][2] = {{0, 5}, {5, 10}, {10, 20}, {20, 40}};
          for (int win = 0; win < 4; ++win) {
            double bar = 0.0, tru = 0.0;
            for (long i = 0; i < w.size(); ++i) {
              const double ev = w(i) * rc::HA_EV;
              if (ev <= W[win][0] or ev > W[win][1]) continue;
              bar = std::max(bar, eb(i));
              tru = std::max(tru, std::abs(v(i) - (*xs[qq])(i)));
            }
            ++st.n_cov;
            st.n_cov_ok += (bar >= tru);
            st.n_cov_3 += (bar >= tru / 3.0 and bar <= 3.0 * tru);
          }
        }
      }
      const double einf_x = 1.0 / (1.0 + h0x);
      st.d_einf.push_back(std::abs(o.eps_inf_h - einf_x) / einf_x);
      st.d_fsum.push_back(std::abs(o.fsum_h - fsx) / fsx);
      if (q < 0)
        app_log(2, "  [V6][optics] {} theta {} q0: eps_inf {:.8f} (m {:.8f}) exact {:.8f}; f-sum h {:.6e} m {:.6e} exact {:.6e}; K {}/{} "
                   "delta {:.1e}/{:.1e}",
                fixture, th, o.eps_inf_h, o.eps_inf_m, einf_x, o.fsum_h, o.fsum_m, fsx, o.K_h, o.K_m, o.delta_h, o.delta_m);
    }
    app_log(2, "  [V6][optics] {} theta {:4.1f}: bosonic rank {} ({} nodes), Pi time nodes {}+{}, head pass {:.2f} s, optics {} curves "
               "{:.2f} s",
            fixture, th, hpo.rank, hpo.nz, hpo.nt_p, hpo.nt_h, hpo.t_total, jobs.size(), t_opt);
  }
  return stats;
}

double med(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
double vmax(std::vector<double> const &v) { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); }

/**
 * Gates of [V6][optics] (MB): max over the finite q and q0 and over loss / eps1 / eps2 of the window errors, per fixture, angle,
 * broadening (0: eta = 0.01 Ha, 1: eta = 0.05 omega) and window (0-5 / 5-10 / 10-20 / 20-40 eV) = 3 x the Mac measurement
 * (2026-10-05, 1 rank, Accelerate; rounded up), floor 1e-4. Error-bar coverage (stored |MB - NNLS| >= true MB error per
 * window) >= 0.4 (measured 0.54-0.92); eps_inf(q) <= 1e-9 (LiH; si211 1e-6: its head is not exactly odd, the PH asymmetry of
 * the fixture, r_apex 2.7e-5), f-sum <= 1e-7 (LiH; si211 2e-3).
 */
struct optics_gate_t {
  std::string fixture;
  double theta;
  int eta;
  std::array<double, 4> g;
};
const std::vector<optics_gate_t> optics_gates = {
    {"qe_lih222", 5.0, 0, {1e-02, 3e-03, 6e-04, 4e-02}},   // measured 3.3e-03 8.6e-04 1.7e-04 1.1e-02
    {"qe_lih222", 5.0, 1, {2e-02, 3e-03, 4e-04, 4e-04}},   // measured 4.0e-03 1.0e-03 1.3e-04 1.3e-04
    {"qe_lih222", 10.0, 0, {2e-03, 9e-04, 6e-02, 2e-01}},   // measured 4.0e-04 2.7e-04 2.0e-02 6.3e-02
    {"qe_lih222", 10.0, 1, {2e-03, 8e-04, 8e-04, 8e-04}},   // measured 4.1e-04 2.5e-04 2.4e-04 2.4e-04
    {"qe_lih222", 20.0, 0, {7e-04, 2e-04, 2e+00, 4e+00}},   // measured 2.2e-04 4.7e-05 4.2e-01 1.1e+00
    {"qe_lih222", 20.0, 1, {7e-04, 2e-04, 8e-02, 2e-01}},   // measured 2.2e-04 4.8e-05 2.5e-02 4.4e-02
    {"qe_lih223", 5.0, 0, {2e-03, 2e-03, 3e-01, 5e-01}},   // measured 4.4e-04 4.2e-04 7.0e-02 1.6e-01
    {"qe_lih223", 5.0, 1, {2e-03, 2e-03, 2e-04, 2e-04}},   // measured 5.2e-04 4.0e-04 6.6e-05 3.4e-05
    {"qe_lih223", 10.0, 0, {6e-04, 3e-02, 2e+00, 3e+00}},   // measured 1.8e-04 6.7e-03 3.7e-01 6.9e-01
    {"qe_lih223", 10.0, 1, {9e-04, 6e-03, 2e-02, 3e-02}},   // measured 3.0e-04 2.0e-03 5.1e-03 7.1e-03
    {"qe_lih223", 20.0, 0, {2e-03, 5e-01, 3e+00, 4e+00}},   // measured 4.8e-04 1.4e-01 7.3e-01 1.2e+00
    {"qe_lih223", 20.0, 1, {2e-03, 2e-01, 4e-01, 3e-01}},   // measured 6.1e-04 3.6e-02 1.1e-01 8.7e-02
    {"qe_si211", 5.0, 0, {6e-04, 1e-03, 5e-03, 4e-02}},   // measured 1.8e-04 3.3e-04 1.4e-03 1.1e-02
    {"qe_si211", 5.0, 1, {6e-04, 1e-03, 2e-03, 8e-03}},   // measured 1.8e-04 3.2e-04 5.7e-04 2.5e-03
    {"qe_si211", 10.0, 0, {5e-03, 4e-03, 6e-02, 3e-01}},   // measured 1.6e-03 1.2e-03 1.9e-02 7.4e-02
    {"qe_si211", 10.0, 1, {6e-03, 3e-03, 8e-03, 2e-02}},   // measured 1.8e-03 9.1e-04 2.4e-03 6.5e-03
    {"qe_si211", 20.0, 0, {3e-03, 5e-03, 4e+00, 3e+00}},   // measured 9.7e-04 1.4e-03 1.1e+00 6.7e-01
    {"qe_si211", 20.0, 1, {5e-03, 2e-03, 6e-01, 2e-01}},   // measured 1.6e-03 6.3e-04 1.8e-01 5.8e-02
};

void report_optics(std::string const &fixture, std::map<double, optics_stats_t> const &S) {
  const char *qn[3] = {"loss", "eps1", "eps2"}, *bn[2] = {"eta 0.01", "eta 0.05w"}, *mn[2] = {"MB", "NNLS"};
  for (auto const &[th, st] : S) {
    for (int b = 0; b < 2; ++b)
      for (int qq = 0; qq < 3; ++qq) {
        std::string s;
        for (int k = 0; k < 2; ++k) {
          s += std::string("  ") + mn[k] + " ";
          for (int win = 0; win < 4; ++win) s += std::format(" {:.1e}/{:.1e}", med(st.e[k][qq][b][win]), vmax(st.e[k][qq][b][win]));
        }
        app_log(2, "  [V6][optics] {} {:4.1f} deg {:<9s} {:<4s} (median/max over q; 0-5 / 5-10 / 10-20 / 20-40 eV):{}", fixture, th, bn[b],
                qn[qq], s);
      }
    app_log(2, "  [V6][optics] {} {:4.1f} deg: error bar >= true MB error in {} / {} windows ({:.2f}), within x3 {:.2f}; eps_inf(q) vs "
               "exact max {:.1e}, f-sum max {:.1e}",
            fixture, th, st.n_cov_ok, st.n_cov, double(st.n_cov_ok) / st.n_cov, double(st.n_cov_3) / st.n_cov, vmax(st.d_einf),
            vmax(st.d_fsum));
    // gates
    for (auto const &gt : optics_gates) {
      if (gt.fixture != fixture or gt.theta != th) continue;
      std::string s;
      for (int win = 0; win < 4; ++win) {
        double m = 0.0;
        for (int qq = 0; qq < 3; ++qq) m = std::max(m, vmax(st.e[0][qq][gt.eta][win]));
        s += std::format(" {:.1e} <= {:.0e}", m, gt.g[win]);
        INFO(fixture << " " << th << " deg eta " << gt.eta << " window " << win << ": " << m << " gate " << gt.g[win]);
        CHECK(m <= gt.g[win]);
      }
      app_log(2, "  [V6][optics] gate {} {:4.1f} deg {}: MB max error per window (measured <= gate):{}", fixture, th,
              gt.eta == 0 ? "eta 0.01 " : "eta 0.05w", s);
    }
    const bool si = fixture == "qe_si211";
    CHECK(double(st.n_cov_ok) / st.n_cov >= 0.4);
    CHECK(vmax(st.d_einf) <= (si ? 1e-6 : 1e-9));
    CHECK(vmax(st.d_fsum) <= (si ? 2e-3 : 1e-7));
  }
}

} // namespace

// regular: 20 and 10 deg (~5 min on the Mac, 1 rank); [.optics5]: 5 deg (~12 min: the head passes of all q at 5 deg)
TEST_CASE("gw_line_optics_lih222", "[gw_line][V6][optics]") { report_optics("qe_lih222", run_optics("qe_lih222", {20.0, 10.0})); }
TEST_CASE("gw_line_optics_si211", "[gw_line][V6][optics]") { report_optics("qe_si211", run_optics("qe_si211", {20.0, 10.0})); }
TEST_CASE("gw_line_optics_lih223", "[gw_line][V6][optics]") { report_optics("qe_lih223", run_optics("qe_lih223", {20.0, 10.0})); }
TEST_CASE("gw_line_optics5_lih222", "[.optics5]") { report_optics("qe_lih222", run_optics("qe_lih222", {5.0})); }
TEST_CASE("gw_line_optics5_si211", "[.optics5]") { report_optics("qe_si211", run_optics("qe_si211", {5.0})); }
TEST_CASE("gw_line_optics5_lih223", "[.optics5]") { report_optics("qe_lih223", run_optics("qe_lih223", {5.0})); }

/// [.flat_scan] diagnostic: Pi error vs the time-grid knobs on flat lines (lih222, one q group)
TEST_CASE("gw_line_flat_scan", "[.flat_scan]") {
  osetup_t su("qe_lih222");
  auto &mpi  = *su.mpi;
  auto &comm = mpi.comm;
  auto &thc  = *su.thc;
  auto &mf   = *su.mf;
  auto all   = nda::range::all;
  aux_grid_t grid(mpi, su.Np);
  head_basis_t hb(thc, mf, grid);
  propagator_t<HOST_MEMORY> prop(thc, grid);
  const q_groups_t qg2(su.nq, 2, qminus_list(mf));
  auto const &qsel = qg2.rows(0);
  std::vector<casida_t> cas;
  for (long q : qsel) {
    cmat Zq(thc.Z(int(q)));
    cas.push_back(casida_q(thc, mf, su.e_rel, q, Zq));
  }
  for (double th : {10.0, 5.0}) {
    struct v_t { std::string tg; double nE, ns, gl, os, smax; };
    std::vector<v_t> V = {{"id", 120, 40, -1, 1.0, -1}, {"id", 240, 40, -1, 1.0, -1}, {"id", 480, 40, -1, 1.0, -1},
                          {"id", 120, 80, -1, 1.0, -1}, {"id", 120, 40, -1, 1.25, -1}, {"id", 480, 160, -1, 1.25, -1},
                          {"id", 240, 80, -1, 1.0, -1}, {"id", 480, 80, -1, 1.0, 3.0}, {"gl", 0, 0, 3, 1, -1},
                          {"gl", 0, 0, 6, 1, -1}, {"gl", 0, 0, 12, 1, -1}};
    for (auto const &v : V) {
      head_pass_params_t hp;
      hp.theta_deg = th;
      hp.lam_b     = std::max(4.0, 1.2 * su.etr_max);
      hp.bos_gap   = 0.5 * su.ks_gap;
      hp.time_grid = v.tg;
      hp.id_nE_per_efold = v.nE;
      hp.id_ns_per_efold = v.ns;
      hp.gl_per_efold    = v.gl;
      hp.time_oversample = v.os;
      hp.id_smax_fac     = v.smax;
      auto g = head_pass_grid(su.poles, hp, comm);
      utils::TimerManager T;
      memory::array<HOST_MEMORY, ComplexType, 4> Pi;
      auto t0 = std::chrono::steady_clock::now();
      polarization<HOST_MEMORY>(prop, su.poles, mf, grid, g.bos->zeta_nodes, *g.pi_p, *g.pi_h, 8, Pi, T, sector_t::both, qsel);
      const double tPi = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      double ep = 0.0, sp = 0.0;
      for (long i = 0; i < long(qsel.size()); ++i) {
        auto Pex = pi_exact_q(cas[i], grid, g.bos->zeta_nodes);
        nda::array<ComplexType, 3> Pl(Pi(i, all, all, all));
        ep = std::max(ep, max_diff3(Pl, Pex));
        sp = std::max(sp, max_abs3(Pex));
      }
      app_log(2, "  [flat_scan] theta {:4.1f} {} nE/efold {} ns/efold {} GL/efold {} oversample {} smax_fac {}: nodes {}+{}, Pi err {:.2e} "
                 "({:.2f} s, grid {:.2f} s)",
              th, v.tg, v.nE, v.ns, v.gl, v.os, v.smax, g.pi_p->size(), g.pi_h->size(), ep / sp, tPi, g.t_grid);
    }
  }
}

// regular: the time-node ID grids at 10 and 5 deg on one q group (the 20 deg kernels are [V1]/[V2]/[V6][head])
TEST_CASE("gw_line_flat_lih222", "[gw_line][V6][flat]") { run_flat("qe_lih222", {10.0, 5.0}, {"id"}, false); }
TEST_CASE("gw_line_flat_si211", "[gw_line][V6][flat]") { run_flat("qe_si211", {10.0, 5.0}, {"id"}, false); }
TEST_CASE("gw_line_flat_lih223", "[gw_line][V6][flat]") { run_flat("qe_lih223", {10.0, 5.0}, {"id"}, false); }
// full: 20 / 10 / 5 deg, GL and ID, head_pass of all q vs the explicit chain (lih222 ~15 min on the Mac)
TEST_CASE("gw_line_flat_full_lih222", "[.flat_full]") { run_flat("qe_lih222", {20.0, 10.0, 5.0}, {"gl", "id"}, true); }
TEST_CASE("gw_line_flat_full_si211", "[.flat_full]") { run_flat("qe_si211", {20.0, 10.0, 5.0}, {"gl", "id"}, true); }
TEST_CASE("gw_line_flat_full_lih223", "[.flat_full]") { run_flat("qe_lih223", {20.0, 10.0, 5.0}, {"gl", "id"}, true); }
