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
 * S8b of notes/line_gw_cpp_plan.md: finite temperature on the line (notes section 11), kernel level. References: the python
 * prototype's oracles tests/unit_test_files/gw_line/{lih222,lih223}_finiteT_ref.h5 (coqui/cayley/scripts/gen_finiteT_ref.py;
 * layout in its header) and mu_rule_ref.h5 (gen_mu_rule_ref.py). KS poles at the reference mu0 of each beta, thermal_tol
 * 1e-12 (= the generator's ver_thermal_tol: at 1e-8 the far poles just outside E_T carry weight 1 instead of 1 - f, a floor
 * of 1e-9..1e-11 by design), theta 20 deg, theta_t 10 deg, c_zeta = c_f = 30.
 *
 * [finiteT][numerics] guarded GL ray / finite-interval ID / tau leg transforms of single exponentials vs the exact T_S
 *   (Eq. fT_TS) and the exact tau integrals (report + gates).
 * [finiteT][mu_rule] (T5 b, unit case) mu_rule "auto" / "gap" / "number" on the lih222 / svo222 KS spectra with the injected
 *   weight errors of dev/s8b_mu_rule.py vs mu_rule_ref.h5: same rule, mu to 1e-12.
 * [finiteT][T1] Pi on the guarded rays (thermal lists) and the tau leg (nu_0, dynamic) vs the finite-T transition sum computed
 *   here, at the data set D of this code (mirror layout) and at the generator's D (vs its Pi_D_probe): GL and ID, <= 1e-9
 *   (10 eps) relative to the max over the set and q; the tau leg's dynamic Pi(q, 0) <= 1e-12; lih222 / lih223, beta 50 / 200.
 * [finiteT][T2] W: Dyson at D, the D-selected basis (the generator's nu_b injected: pivoted-QR choices differ between LAPACKs)
 *   and the split pair fit; fitted W at nu_0 / band / i nu_n / line nodes / needed points vs the finite-T Casida W <= 1e-10;
 *   Sigma_c (thermal lists, Bose-augmented basis) at the fermionic test points rho beta |zeta| >= c_f and i w_n >= zeta_T vs
 *   Eq. fT_sigma <= 1e-8 (k = 0, 3 / 0, 9).
 * [finiteT][T3] iteration-1 G0W0 Sigma_c(i w_n) vs CoQui gw_t at the same beta (IAFT of that beta, mu by update_G) after the
 *   predicted nu_0 term of the reference (the Matsubara convention's degenerate pairs): <= 1e-7; KS mu vs CoQui's <= 1e-10.
 * The SCF-level cases (T4, T5) are in this file too (see further down).
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <numbers>
#include <string>
#include <tuple>
#include <vector>

#include "mpi3/communicator.hpp"
#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "nda/linalg.hpp"
#include "h5/h5.hpp"
#include "nda/h5.hpp"

#include "mean_field/default_MF.hpp"
#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/mb_state/mb_state.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_common.hpp"
#include "methods/SCF/scf_driver.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "numerics/iter_scf/iter_scf_utils.hpp"
#include "methods/HF/hf_t.h"
#include "methods/GW/gw_t.h"
#include "methods/scr_coulomb/scr_coulomb_t.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "hamiltonian/one_body_hamiltonian.hpp"

#include "numerics/line_dlr/time_ray.hpp"
#include "numerics/line_dlr/time_id.hpp"
#include "numerics/line_dlr/line_basis.hpp"
#include "numerics/line_dlr/bosonic_basis.hpp"
#include "numerics/line_dlr/cayley.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/polarization.hpp"
#include "methods/GW_line/screened.hpp"
#include "methods/GW_line/self_energy.hpp"
#include "methods/GW_line/static_part.hpp"
#include "methods/GW_line/time_grids.hpp"
#include "methods/GW_line/thermal.hpp"
#include "methods/GW_line/driver.hpp"

namespace {

using namespace methods::gw_line;
using numerics::line_dlr::time_ray_t;
using numerics::line_dlr::time_id_t;
using numerics::line_dlr::sector_t;
using numerics::line_dlr::bosonic_basis_t;
namespace mpi3 = boost::mpi3;

std::string ft_dir() { return std::string(PROJECT_SOURCE_DIR) + "/tests/unit_test_files/gw_line/"; }

template <int R> nda::array<ComplexType, R> read_c(h5::group &g, std::string const &nm) {
  nda::array<double, R> re, im;
  nda::h5_read(g, nm + "_re", re);
  nda::h5_read(g, nm + "_im", im);
  nda::array<ComplexType, R> out(re.shape());
  for (long a = 0; a < re.size(); ++a) out.data()[a] = ComplexType(re.data()[a], im.data()[a]);
  return out;
}
template <typename T> T read_attr(h5::group &g, std::string const &nm) {
  T x{};
  h5::h5_read_attribute(g, nm, x);
  return x;
}

bool gate(char const *what, double v, double g) {
  const bool ok = (v <= g);
  app_log(1, "    {:<52s} {:.3e}  (gate {:.1e}) {}", what, v, g, ok ? "ok" : "FAIL");
  return ok;
}

struct fx_t {
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;
  std::shared_ptr<mf::MF> mf;
  std::unique_ptr<methods::thc_reader_t> thc;
  std::string name;
  explicit fx_t(std::string const &nm) : name(nm) {
    mpi = utils::make_unit_test_mpi_context();
    if (nm == "svo222n") {   // S8c: the nosym twin of svo222 ($GW_LINE_SVO222N: out/svo.coqui.h5 + svo222n_thc/, rusty data/svo222n;
                             // test_gw_line_scf [.gw_line_dump_svo222n]); the full-BZ path of the driver
      char const *e = std::getenv("GW_LINE_SVO222N");
      utils::check(e != nullptr, "set GW_LINE_SVO222N to the svo222n data directory");
      const std::string D = std::string(e) + "/";
      mf  = std::make_shared<mf::MF>(mf::default_MF(mpi, mf::qe_source, D + "out/", "svo", mf::h5_input_type));
      thc = std::make_unique<methods::thc_reader_t>(mf, "incore", D + "svo222n_thc/thc.eri.h5");
      return;
    }
    mf  = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_" + nm));
    thc = std::make_unique<methods::thc_reader_t>(mf, "incore", ft_dir() + nm + "_thc/thc.eri.h5");
  }
};

/// U^H M U of the blocks M (n, nP, nQ) (all_reduced): (n, np, np)
template <typename A>
nda::array<ComplexType, 3> probe_blocks(A const &M, nda::array<ComplexType, 2> const &U, aux_grid_t const &g,
                                        mpi3::communicator &comm) {
  const long n = M.extent(0), np = U.extent(1);
  nda::array<ComplexType, 3> out(n, np, np);
  out() = 0.0;
  for (long i = 0; i < n; ++i)
    for (long a = 0; a < np; ++a)
      for (long b = 0; b < np; ++b) {
        ComplexType s = 0.0;
        for (long P = 0; P < g.nP; ++P) {
          ComplexType t = 0.0;
          for (long Q = 0; Q < g.nQ; ++Q) t += M(i, P, Q) * U(g.Q0 + Q, b);
          s += std::conj(U(g.P0 + P, a)) * t;
        }
        out(i, a, b) = s;
      }
  comm.all_reduce_in_place_n(out.data(), out.size(), std::plus<>{});
  return out;
}

/// finite-T transition sum (Eq. fT_pi; all (n, m) pairs, weight f(e_n) - f(e_m), |E| >= deg_tol) in probe space: (nq, nz, np, np)
nda::array<ComplexType, 4> pi_transition_probe(methods::thc_reader_t &thc, mf::MF &mf, nda::array<double, 2> const &e_rel, double beta,
                                               nda::array<ComplexType, 1> const &z, nda::array<ComplexType, 2> const &U,
                                               double deg_tol = 1e-8) {
  const long nk = mf.nkpts(), nq = mf.nqpts(), nb = e_rel.extent(1), nz = z.size(), np = U.extent(1), Np = U.extent(0);
  auto qk = mf.qk_to_k2();
  nda::array<ComplexType, 4> out(nq, nz, np, np);
  out() = 0.0;
  const double nrm = std::sqrt(2.0 / double(nk));
  for (long iq = 0; iq < nq; ++iq)
    for (long ik = 0; ik < nk; ++ik) {
      const long kq = qk(iq, ik);
      auto Xk = thc.X(0, 0, ik);
      auto Xq = thc.X(0, 0, kq);
      for (long n = 0; n < nb; ++n)
        for (long m = 0; m < nb; ++m) {
          const double E = e_rel(kq, m) - e_rel(ik, n);
          if (std::abs(E) < deg_tol) continue;
          const double w = numerics::line_dlr::fermi(e_rel(ik, n), beta) - numerics::line_dlr::fermi(e_rel(kq, m), beta);
          if (w == 0.0) continue;
          std::vector<ComplexType> s(np, 0.0);
          for (long P = 0; P < Np; ++P) {
            const ComplexType S = nrm * Xk(P, n) * std::conj(Xq(P, m));
            for (long a = 0; a < np; ++a) s[a] += std::conj(U(P, a)) * S;
          }
          for (long iz = 0; iz < nz; ++iz) {
            const ComplexType c = w / (z(iz) - E);
            for (long a = 0; a < np; ++a)
              for (long b = 0; b < np; ++b) out(iq, iz, a, b) += c * s[a] * std::conj(s[b]);
          }
        }
    }
  return out;
}

/// max over (q, points in sel) of |A - B|, and max |B| (relative error = first / second)
std::pair<double, double> cmp4(nda::array<ComplexType, 4> const &A, nda::array<ComplexType, 4> const &B, std::vector<long> const &selA,
                               std::vector<long> const &selB) {
  double d = 0.0, m = 0.0;
  for (long q = 0; q < A.extent(0); ++q)
    for (size_t i = 0; i < selA.size(); ++i)
      for (long a = 0; a < A.extent(2); ++a)
        for (long b = 0; b < A.extent(3); ++b) {
          d = std::max(d, std::abs(A(q, selA[i], a, b) - B(q, selB[i], a, b)));
          m = std::max(m, std::abs(B(q, selB[i], a, b)));
        }
  return {d, m};
}

} // namespace

// ============================================================================================================== numerics
TEST_CASE("gw_line_finiteT_numerics", "[gw_line][finiteT][numerics]") {
  const double deg = std::numbers::pi / 180.0, theta = 20.0 * deg, tht = 10.0 * deg;
  bool ok = true;
  for (double beta : {50.0, 200.0}) {
    const double cT = std::log(1e12), ET = cT / beta, ST = beta / std::sin(tht);
    auto rp = time_ray_t::guarded(tht, beta, ET, 1e-5, 3.0, 16, sector_t::particle);
    numerics::line_dlr::time_id_opts_t o;
    o.pad = 1.25;
    auto gid = time_id_t::finite_interval(tht, sector_t::particle, 2.0 * ET, 8.0, ST, 1e-10, o);
    // targets: wedge points (band-like), i nu_n, line nodes above the floor
    std::vector<ComplexType> zs;
    const double d0 = 30.0 * std::sin(tht) / beta;
    for (double v : {d0, 3 * d0, 10 * d0})
      for (double x : {-0.5, -0.05, 0.0, 0.2}) zs.push_back(ComplexType(x, (v + std::abs(x) * std::sin(tht)) / std::cos(tht)));
    for (long n : {1, 3, 10}) zs.push_back(ComplexType(0.0, 2.0 * std::numbers::pi * n / beta));
    for (double r : {30.0 / beta, 1.0, 20.0}) zs.push_back(r * std::exp(ComplexType(0.0, theta)));
    nda::array<ComplexType, 1> z(long(zs.size()));
    for (long i = 0; i < z.size(); ++i) z(i) = zs[i];
    auto Fg = rp.transform_matrix(z);
    double res = 0.0;
    auto Fi = gid.transform_matrix(z, &res);
    double eg = 0.0, ei = 0.0;
    for (double E : {-2.0 * ET, -ET, -1e-3, 0.0, 1e-4, 0.03, 0.5, 3.0, 7.9}) {
      for (long i = 0; i < z.size(); ++i) {
        const ComplexType ex = gid.target_finite(z(i), E);
        // bounded products: the pair weight e^{-beta |E|} (E < 0) times the transform (KMS); normalize by max(1, |T_S|)
        const double sc = (E < 0 ? std::exp(-beta * std::abs(E)) : 1.0) / std::max(1.0, std::abs(ex));
        ComplexType sg = 0.0, si = 0.0;
        for (long m = 0; m < rp.size(); ++m) sg += Fg(i, m) * std::exp(ComplexType(0.0, -E) * rp.t(m));
        for (long m = 0; m < gid.size(); ++m) si += Fi(i, m) * std::exp(ComplexType(0.0, -E) * gid.t(m));
        eg = std::max(eg, std::abs(sg - ex) * sc);
        ei = std::max(ei, std::abs(si - ex) * sc);
      }
    }
    app_log(1, "\n[finiteT][numerics] beta {}: GL guarded ray {} nodes; finite-interval ID rank {} nodes {} (candidates {}, nE {}, LS "
               "residual {:.1e}); max weighted error vs T_S: GL {:.2e}, ID {:.2e}",
            beta, rp.size(), gid.rank, gid.size(), gid.n_cand, gid.nE(), res, eg, ei);
    ok = gate("GL guarded ray vs T_S", eg, 1e-12) and ok;
    ok = gate("finite-interval ID vs T_S (10 eps)", ei, 1e-9) and ok;
    // tau leg nodes: -sum w e^{i nu tau} e^{-E tau} = -int_0^{beta/2} e^{(i nu - E) tau}
    for (std::string kind : {"gl", "id"}) {
      thermal_params_t tp;
      tp.beta = beta;
      tp.tau_grid = kind;
      tp.tau_eps = 1e-12;
      auto tn = make_tau_nodes(tp, 1.7, 3.4);
      nda::array<ComplexType, 1> zt(3);
      zt(0) = 0.0;
      zt(1) = ComplexType(0.0, 2 * std::numbers::pi / beta);
      zt(2) = ComplexType(0.0, 14 * std::numbers::pi / beta);
      auto F = tn.p->transform_matrix(zt);
      double et = 0.0;
      for (double E : {-2.0 * cT / beta, -0.05, 0.0, 0.01, 0.3, 1.7, 3.4}) {
        for (long i = 0; i < 3; ++i) {
          const ComplexType a = zt(i) - E;   // int_0^{b/2} e^{a tau}
          const ComplexType ex = (std::abs(a) < 1e-14) ? ComplexType(0.5 * beta) : (std::exp(a * 0.5 * beta) - 1.0) / a;
          ComplexType s = 0.0;
          for (long m = 0; m < tn.size(); ++m) s += F(i, m) * std::exp(ComplexType(0.0, -E) * tn.p->t(m));
          const double sc = (E < 0 ? std::exp(-0.5 * beta * std::abs(E)) : 1.0) / std::max(1.0, std::abs(ex));
          et = std::max(et, std::abs(s + ex) * sc);
        }
      }
      app_log(1, "  tau leg ({}): {} nodes on [0, beta/2], max weighted error {:.2e}", kind, tn.size(), et);
      ok = gate("tau nodes vs exact tau integrals (10 tau_eps)", et, 10.0 * tp.tau_eps) and ok;
    }
  }
  REQUIRE(ok);
}

// ============================================================================================================== mu rule
TEST_CASE("gw_line_finiteT_mu_rule", "[gw_line][finiteT][mu_rule]") {
  const std::string ref = ft_dir() + "mu_rule_ref.h5";
  REQUIRE(std::filesystem::exists(ref));
  h5::file f(ref, 'r');
  h5::group g(f);
  const long nc = read_attr<long>(g, "ncases");
  bool ok = true;
  double dmax = 0.0;
  app_log(1, "\n[finiteT][mu_rule] {} cases (mu_rule_ref.h5)", nc);
  for (long c = 0; c < nc; ++c) {
    auto gc        = g.open_group("c" + std::to_string(c));
    auto fx        = read_attr<std::string>(gc, "fixture");
    const double beta = read_attr<double>(gc, "beta"), err = read_attr<double>(gc, "err"), mu_ref = read_attr<double>(gc, "mu_ref");
    const double nelec = read_attr<double>(gc, "nelec");
    nda::array<double, 2> eig;
    {
      h5::file fs(ft_dir() + fx + "_thc/system.h5", 'r');
      h5::group gs(fs);
      auto sg = gs.open_group("system");
      nda::h5_read(sg, "eigval", eig);
    }
    const long nk = eig.extent(0), nb = eig.extent(1);
    std::vector<nda::array<double, 1>> e(nk);
    std::vector<nda::array<ComplexType, 2>> v(nk);
    for (long k = 0; k < nk; ++k) {
      e[k] = eig(k, nda::range::all);
      v[k] = nda::array<ComplexType, 2>(nb, nb);
      v[k]() = 0.0;
      long nocc = 0;
      for (long n = 0; n < nb; ++n) nocc += eig(k, n) < mu_ref ? 1 : 0;
      const double sc = std::sqrt(1.0 + err * 2.0 * double(nk) / (2.0 * double(nocc ? nocc : 1)));
      for (long n = 0; n < nb; ++n) v[k](n, n) = (eig(k, n) < mu_ref) ? sc : 1.0;
    }
    thermal_params_t tp;
    tp.beta = beta;
    tp.thermal_tol = 1e-8;
    double mus[4];
    std::string rules[4];
    const char *names[4] = {"auto", "gap", "number", "neutral"};   // S8b.4 (e): "neutral" (the continuous rule, default)
    for (int r = 0; r < 4; ++r) {
      tp.mu_rule = names[r];
      auto o     = mu_rule_apply(e, v, nelec, {}, tp);
      mus[r]     = o.mu;
      rules[r]   = o.rule;
    }
    const auto rule_ref = read_attr<std::string>(gc, "rule_auto");
    const double d = std::max({std::abs(mus[0] - read_attr<double>(gc, "mu_auto")), std::abs(mus[1] - read_attr<double>(gc, "mu_gap")),
                               std::abs(mus[2] - read_attr<double>(gc, "mu_number")),
                               std::abs(mus[3] - read_attr<double>(gc, "mu_neutral"))});
    dmax = std::max(dmax, d);
    const bool same = (rules[0] == rule_ref) and (rules[3] == read_attr<std::string>(gc, "rule_neutral"));
    app_log(1, "  {} beta {:4.0f} err {:+.0e}: auto -> {:7s} (python {:6s}) mu {:+.12f}, neutral -> {:7s} mu {:+.12f}, max |dmu| (auto, "
               "gap, number, neutral) {:.1e}",
            fx, beta, err, rules[0], rule_ref, mus[0], rules[3], mus[3], d);
    ok = ok and same and d <= 1e-12;
  }
  ok = gate("max |dmu| over cases and rules", dmax, 1e-12) and ok;
  REQUIRE(ok);
}

// ============================================================================================================== T1 / T2
namespace {

/// one fixture / beta: T1 (Pi) and T2 (W, Sigma). time_grid: "gl" | "id"
bool run_t12(fx_t &F, double beta, std::string const &time_grid, bool do_t2, double tt_frac = 0.5) {
  auto &comm    = F.mpi->comm;
  auto &mf      = *F.mf;
  auto &thc     = *F.thc;
  const long nk = mf.nkpts(), nq = mf.nqpts(), nb = thc.nbnd(), Np = thc.Np();
  const std::string ref = ft_dir() + F.name + "_finiteT_ref.h5";
  REQUIRE(std::filesystem::exists(ref));
  h5::file hf(ref, 'r');
  h5::group root(hf);
  auto gb = root.open_group("beta_" + std::to_string(long(beta)));
  const double mu0 = read_attr<double>(gb, "mu0");
  auto U           = read_c<2>(root, "probe");
  const double deg = std::numbers::pi / 180.0;
  thermal_params_t tp;
  tp.beta = beta;
  tp.thermal_tol = 1e-12;
  tp.theta = 20.0 * deg;
  tp.theta_t = tt_frac * 20.0 * deg;   // theta_t_frac (0.5: the references' 10 deg; 0.25: the metals' theta / 4)
  tp.tau_grid = "gl";   // the composite GL tau grid here (python's); the tau ID is checked against it below
  const double ET = tp.E_T();
  utils::TimerManager Timer;
  bool ok = true;

  nda::array<double, 2> eig(nk, nb), e_rel(nk, nb);
  for (long k = 0; k < nk; ++k)
    for (long n = 0; n < nb; ++n) {
      eig(k, n)   = mf.eigval()(0, k, n);
      e_rel(k, n) = eig(k, n) - mu0;
    }
  auto ks = pole_data_t::from_ks(eig, mu0);
  auto tl = thermal_lists(ks, beta, ET);
  {
    auto wc  = window_counts(ks, std::log(1e8) / beta);
    nda::array<long, 1> wr;
    nda::h5_read(gb, "window_counts", wr);
    long dw = 0;
    for (long k = 0; k < nk; ++k) dw += std::abs(wc[k] - wr(k));
    app_log(1, "\n[finiteT][T1{}] {} beta {} ({} grid, theta_t = {} theta, rho {:.3f}): mu0 {:.12f}, E_T {:.4f} (thermal_tol 1e-12), window counts at 1e-8 vs "
               "reference: {} differences",
            do_t2 ? "/T2" : "", F.name, beta, time_grid, tt_frac, tp.rho(), mu0, ET, dw);
    ok = (dw == 0) and ok;
  }
  aux_grid_t grid(*F.mpi, Np);
  propagator_t<HOST_MEMORY> prop(thc, grid);
  ibz_t ibz(mf, nb, false);

  // the data sets: this code's D (mirror layout, gapless line basis lam_b 4, eps 1e-10) and the generator's D
  numerics::line_dlr::bosonic_basis_t bline(tp.theta, 4.0, 1e-10, 0.0);
  auto Dm = make_bos_data(bline.zeta_nodes, tp);
  nda::array<ComplexType, 1> Dr = read_c<1>(gb, "D_zeta");
  nda::array<long, 1> Dk;
  {
    nda::array<signed char, 1> k8;
    nda::h5_read(gb, "D_kind", k8);
    Dk = nda::array<long, 1>(k8.size());
    for (long i = 0; i < k8.size(); ++i) Dk(i) = k8(i);
  }
  app_log(1, "  D (this code): {} points (line {}, band {}, Matsubara {}, nu_0 x{}), mirror {}; generator's D {} points", Dm.z.size(),
          Dm.n_line, Dm.n_band, Dm.n_mats, Dm.count(3), Dm.mirror, Dr.size());

  // time nodes
  std::optional<numerics::line_dlr::time_nodes_t> tpn, thn;
  double emax = 0.0;
  for (long k = 0; k < nk; ++k)
    for (long n = 0; n < nb; ++n) emax = std::max(emax, std::abs(e_rel(k, n)));
  double numax = 4.0;
  if (time_grid == "gl") {
    tpn.emplace(time_ray_t::guarded(tp.theta_t, beta, ET, 1e-5, 3.0, 16, sector_t::particle));
    thn.emplace(time_ray_t::guarded(tp.theta_t, beta, ET, 1e-5, 3.0, 16, sector_t::hole));
  } else {
    numerics::line_dlr::time_id_opts_t o;
    o.pad = 1.25;
    auto T = line_time_grids_t::thermal(tp.theta_t, 2.0 * ET, 2.0 * emax + numax, tp.S_T(), 1e-10, o, Dm.z, Dm.z, comm);
    T.log(1);
    tpn.emplace(T.pi_p);
    thn.emplace(T.pi_h);
  }
  auto tn = make_tau_nodes(tp, emax, 2.0 * emax);
  std::vector<long> qall(nq);
  std::iota(qall.begin(), qall.end(), 0L);

  // Pi at a point set (rays) with the nu_0 rows from the tau leg
  auto pi_at = [&](nda::array<ComplexType, 1> const &z, memory::array<HOST_MEMORY, ComplexType, 4> &Pi) {
    memory::array<HOST_MEMORY, ComplexType, 4> Pt;
    nda::array<ComplexType, 1> z0(1);
    z0(0) = 0.0;
    long nd = 0;
    pi_tau_leg<HOST_MEMORY>(prop, ks, mf, ibz, grid, tp, tn, z0, 8, Pt, Timer, qall, &nd);
    polarization<HOST_MEMORY>(prop, tl, mf, grid, z, *tpn, *thn, 8, Pi, Timer, sector_t::both, qall);
    for (long i = 0; i < z.size(); ++i)
      if (z(i) == ComplexType(0.0))
        for (long q = 0; q < nq; ++q) Pi(q, i, nda::range::all, nda::range::all) = Pt(q, 0, nda::range::all, nda::range::all);
    return nd;
  };

  // ---- T1 at this code's D vs the transition sum
  {
    memory::array<HOST_MEMORY, ComplexType, 4> Pi;
    auto t0      = std::chrono::steady_clock::now();
    const long nd = pi_at(Dm.z, Pi);
    const double tsec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    nda::array<ComplexType, 4> P(nq, Dm.z.size(), U.extent(1), U.extent(1));
    for (long q = 0; q < nq; ++q) P(q, nda::ellipsis{}) = probe_blocks(Pi(q, nda::ellipsis{}), U, grid, comm);
    auto X = pi_transition_probe(thc, mf, e_rel, beta, Dm.z, U);
    for (int kd = 0; kd < 4; ++kd) {
      std::vector<long> sel;
      for (long i = 0; i < Dm.z.size(); ++i)
        if (Dm.kind[i] == kd) sel.push_back(i);
      auto [d, m] = cmp4(P, X, sel, sel);
      char const *nm[4] = {"T1 Pi line nodes (this D) vs transition sum", "T1 Pi wedge band vs transition sum",
                           "T1 Pi i nu_n vs transition sum", "T1 Pi(q, 0) tau leg (dynamic) vs transition sum"};
      ok = gate(nm[kd], d / m, kd == 3 ? 1e-12 : 1e-9) and ok;
    }
    app_log(1, "  Pi on this D: {:.1f} s, ray nodes {} + {}, tau nodes {} (x2 legs), degenerate pairs {}", tsec, tpn->size(), thn->size(),
            tn.size(), nd);
    // the tau ID (finite-interval ID at theta_t = pi / 2 on [0, beta / 2]) vs the composite GL tau grid
    for (double teps : {1e-12, 1e-13}) {
      auto tp2     = tp;
      tp2.tau_grid = "id";
      tp2.tau_eps  = teps;
      auto tn2     = make_tau_nodes(tp2, emax, 2.0 * emax);
      memory::array<HOST_MEMORY, ComplexType, 4> Pt2;
      nda::array<ComplexType, 1> z0(1);
      z0(0) = 0.0;
      pi_tau_leg<HOST_MEMORY>(prop, ks, mf, ibz, grid, tp2, tn2, z0, 8, Pt2, Timer, qall);
      nda::array<ComplexType, 4> P2(nq, 1, U.extent(1), U.extent(1));
      for (long q = 0; q < nq; ++q) P2(q, nda::ellipsis{}) = probe_blocks(Pt2(q, nda::ellipsis{}), U, grid, comm);
      auto X0 = pi_transition_probe(thc, mf, e_rel, beta, z0, U);
      auto [d, m] = cmp4(P2, X0, {0}, {0});
      app_log(1, "  tau ID (tau_eps {:.0e}): {} nodes (rank {}) vs GL {}: Pi(q, 0) error {:.2e}", teps, tn2.size(), tn2.rank, tn.size(), d / m);
      ok = gate("T1 Pi(q, 0) tau ID vs transition sum", d / m, 10.0 * teps) and ok;
    }
  }
  // ---- T1 at the generator's D vs its oracle (kinds 1-3)
  memory::array<HOST_MEMORY, ComplexType, 4> PiR;
  pi_at(Dr, PiR);
  {
    nda::array<ComplexType, 4> P(nq, Dr.size(), U.extent(1), U.extent(1));
    for (long q = 0; q < nq; ++q) P(q, nda::ellipsis{}) = probe_blocks(PiR(q, nda::ellipsis{}), U, grid, comm);
    auto PD = read_c<4>(gb, "Pi_D_probe");
    std::vector<long> sa, sb;
    long j = 0;
    for (long i = 0; i < Dr.size(); ++i)
      if (Dk(i) != 0) {
        sa.push_back(i);
        sb.push_back(j++);
      }
    auto [d, m] = cmp4(P, PD, sa, sb);
    ok = gate("T1 Pi at the generator's D (kinds 1-3) vs its oracle", d / m, 1e-9) and ok;
  }
  if (not do_t2) return ok;

  // ---- T2: W on the generator's D with its nu_b, split fit; all rows augmented (w(q), w(-q)^T of every pole)
  nda::array<double, 1> nub;
  nda::h5_read(gb, "nu_b", nub);
  auto B = bosonic_basis_t::with_poles(tp.theta, Dr, 4.0, nub);
  const long r = B.rank;
  auto B2      = B;
  B2.n_bose    = r;
  B2.rank      = 2 * r;
  B2.bose_src.resize(r);
  std::iota(B2.bose_src.begin(), B2.bose_src.end(), 0L);
  B2.nu = nda::array<double, 1>(2 * r);
  B2.nu(nda::range(r)) = nub;
  B2.nu(nda::range(r, 2 * r)) = nub;
  coulomb_blocks_t<HOST_MEMORY> Zb(thc, grid, qall, Timer);
  memory::array<HOST_MEMORY, ComplexType, 4> w, Wn;
  screened_interaction<HOST_MEMORY>(PiR, Zb, B2, grid, *F.mpi, w, Timer, &Wn, qall, false);
  auto WD = read_c<4>(gb, "W_D_probe");
  {
    nda::array<ComplexType, 4> P(nq, Dr.size(), U.extent(1), U.extent(1));
    for (long q = 0; q < nq; ++q) P(q, nda::ellipsis{}) = probe_blocks(Wn(q, nda::ellipsis{}), U, grid, comm);
    std::vector<long> s(Dr.size());
    std::iota(s.begin(), s.end(), 0L);
    auto [d, m] = cmp4(P, WD, s, s);
    ok = gate("T2 W (Dyson of the line Pi) at D vs Casida", d / m, 1e-8) and ok;
  }
  // fitted W at a point set: sum_j w_j / (z - nu_j) - wB_j / (z + nu_j)
  auto wfit = [&](nda::array<ComplexType, 1> const &z) {
    nda::array<ComplexType, 4> P(nq, z.size(), U.extent(1), U.extent(1));
    for (long q = 0; q < nq; ++q) {
      nda::array<ComplexType, 3> M(z.size(), grid.nP, grid.nQ);
      M() = 0.0;
      for (long i = 0; i < z.size(); ++i)
        for (long j = 0; j < r; ++j) {
          const ComplexType a = 1.0 / (z(i) - nub(j)), b = -1.0 / (z(i) + nub(j));
          M(i, nda::range::all, nda::range::all) += a * w(q, j, nda::range::all, nda::range::all) + b * w(q, r + j, nda::range::all, nda::range::all);
        }
      P(q, nda::ellipsis{}) = probe_blocks(M, U, grid, comm);
    }
    return P;
  };
  {
    auto P = wfit(Dr);
    char const *nm[4] = {"T2 fitted W at the line nodes of D", "T2 fitted W on the wedge band", "T2 fitted W at i nu_n",
                         "T2 fitted W at nu_0"};
    for (int kd = 0; kd < 4; ++kd) {
      std::vector<long> sel;
      for (long i = 0; i < Dr.size(); ++i)
        if (Dk(i) == kd) sel.push_back(i);
      auto [d, m] = cmp4(P, WD, sel, sel);
      ok = gate(nm[kd], d / m, 1e-10) and ok;
    }
    auto zn = read_c<1>(gb, "need_zeta");
    auto Pn = wfit(zn);
    auto WN = read_c<4>(gb, "W_need_probe");
    std::vector<long> s(zn.size());
    std::iota(s.begin(), s.end(), 0L);
    auto [d, m] = cmp4(Pn, WN, s, s);
    ok = gate("T2 fitted W at the needed points zeta_f - e_m", d / m, 1e-10) and ok;
  }
  // Sigma: the Bose-augmented basis of the fitted poles, its rows taken from the all-augmented w
  {
    auto Bs = B.with_bose(beta, ET);
    memory::array<HOST_MEMORY, ComplexType, 4> ws(nq, Bs.rank, grid.nP, grid.nQ);
    for (long q = 0; q < nq; ++q) {
      for (long j = 0; j < r; ++j) ws(q, j, nda::range::all, nda::range::all) = w(q, j, nda::range::all, nda::range::all);
      for (long b = 0; b < Bs.n_bose; ++b)
        ws(q, r + b, nda::range::all, nda::range::all) = w(q, r + Bs.bose_src[b], nda::range::all, nda::range::all);
    }
    auto zf = read_c<1>(root, "ferm_zeta");
    nda::array<long, 1> wn;
    nda::h5_read(gb, "w_n", wn);
    nda::array<ComplexType, 1> zz(zf.size() + wn.size());
    for (long i = 0; i < zf.size(); ++i) zz(i) = zf(i);
    for (long i = 0; i < wn.size(); ++i) zz(zf.size() + i) = ComplexType(0.0, (2.0 * wn(i) + 1.0) * std::numbers::pi / beta);
    nda::array<signed char, 1> fm, im;
    nda::h5_read(gb, "ferm_mask", fm);
    nda::h5_read(gb, "iw_mask", im);
    // Sigma time nodes: the GL guarded rays or a finite-interval ID on the Sigma range
    std::optional<numerics::line_dlr::time_nodes_t> sp_, sh_;
    if (time_grid == "gl") {
      sp_ = tpn;
      sh_ = thn;
    } else {
      numerics::line_dlr::time_id_opts_t o;
      o.pad = 1.25;
      auto T = line_time_grids_t::thermal(tp.theta_t, 2.0 * ET, emax + 4.0, tp.S_T(), 1e-10, o, zz, zz, comm);
      sp_.emplace(T.sig_p);
      sh_.emplace(T.sig_h);
    }
    nda::array<ComplexType, 4> Sp, Sh;
    self_energy<HOST_MEMORY>(prop, tl, ws, Bs, mf, grid, *F.mpi, zz, *sp_, *sh_, 8, Sp, Timer, sector_t::both, false, nullptr, 0, &Sh);
    auto SZ = read_c<4>(gb, "Sigma_zeta");
    auto SI = read_c<4>(gb, "Sigma_iw");
    // attr sigma_k of the generator: k = 0 and the gap k (lih222: 3, lih223: 9, svo222: 1)
    nda::array<long, 1> sk(2);
    sk(0) = 0;
    sk(1) = (F.name == "lih222") ? 3 : (F.name == "lih223" ? 9 : 1);
    double dz = 0.0, mz = 0.0, di = 0.0, mi = 0.0;
    for (long a = 0; a < sk.size(); ++a) {
      const long k = sk(a);
      for (long i = 0; i < zf.size(); ++i) {
        if (not fm(i)) continue;
        for (long x = 0; x < nb; ++x)
          for (long y = 0; y < nb; ++y) {
            dz = std::max(dz, std::abs(Sp(k, i, x, y) + Sh(k, i, x, y) - SZ(a, i, x, y)));
            mz = std::max(mz, std::abs(SZ(a, i, x, y)));
          }
      }
      for (long i = 0; i < wn.size(); ++i) {
        if (not im(i)) continue;
        const long iz = zf.size() + i;
        for (long x = 0; x < nb; ++x)
          for (long y = 0; y < nb; ++y) {
            di = std::max(di, std::abs(Sp(k, iz, x, y) + Sh(k, iz, x, y) - SI(a, i, x, y)));
            mi = std::max(mi, std::abs(SI(a, i, x, y)));
          }
      }
    }
    app_log(1, "  Sigma basis: {} fitted poles + {} Bose rows (n_j > 0, nu_j <= E_T), Sigma nodes {} + {}", r, Bs.n_bose, sp_->size(),
            sh_->size());
    ok = gate("T2 Sigma_c at the fermionic nodes rho beta |z| >= c_f", dz / mz, 1e-8) and ok;
    ok = gate("T2 Sigma_c at i w_n >= zeta_T", di / mi, 1e-8) and ok;
  }
  return ok;
}

} // namespace

TEST_CASE("gw_line_finiteT_T12_lih222", "[gw_line][finiteT][T1][T2]") {
  fx_t F("lih222");
  bool ok = true;
  for (double beta : {200.0, 50.0}) ok = run_t12(F, beta, "gl", true) and ok;
  ok = run_t12(F, 200.0, "id", true) and ok;
  // theta_t = theta / 4 (rho 2.97; the metals' setting, plan S8c): the references' points stay inside the wider wedge
  ok = run_t12(F, 50.0, "id", true, 0.25) and ok;
  REQUIRE(ok);
}

TEST_CASE("gw_line_finiteT_T12_lih223", "[gw_line][finiteT][T1][T2]") {
  fx_t F("lih223");
  bool ok = true;
  for (double beta : {200.0, 50.0}) ok = run_t12(F, beta, "gl", true) and ok;
  ok = run_t12(F, 50.0, "id", false) and ok;
  REQUIRE(ok);
}


// ============================================================================================================== T3
namespace {

/// full Np x Np matrix of one block row (all_reduce of the zero-padded blocks)
nda::array<ComplexType, 2> full_of(nda::array<ComplexType, 2> const &blk, aux_grid_t const &g, mpi3::communicator &comm) {
  nda::array<ComplexType, 2> M(g.Np, g.Np);
  M() = 0.0;
  for (long P = 0; P < g.nP; ++P)
    for (long Q = 0; Q < g.nQ; ++Q) M(g.P0 + P, g.Q0 + Q) = blk(P, Q);
  comm.all_reduce_in_place_n(M.data(), M.size(), std::plus<>{});
  return M;
}

/// the Matsubara nu_0 term -(1/(beta N_k)) sum_q X(k)^dag [G~(k - q, z) o dW(q)] X(k) (KS G at the poles e_rel)
nda::array<ComplexType, 3> nu0_term(methods::thc_reader_t &thc, mf::MF &mf, nda::array<double, 2> const &e_rel,
                                    std::vector<nda::array<ComplexType, 2>> const &dW, long ik, nda::array<ComplexType, 1> const &z,
                                    double beta) {
  const long nk = mf.nkpts(), nq = mf.nqpts(), nb = e_rel.extent(1), Np = thc.Np();
  auto qk = mf.qk_to_k2();
  nda::array<ComplexType, 3> out(z.size(), nb, nb);
  out() = 0.0;
  nda::matrix<ComplexType> Xk(thc.X(0, 0, ik));
  for (long iq = 0; iq < nq; ++iq) {
    const long kq = qk(iq, ik);
    nda::matrix<ComplexType> Xq(thc.X(0, 0, kq));
    for (long iz = 0; iz < z.size(); ++iz) {
      nda::matrix<ComplexType> XG(Np, nb), Gt(Np, Np);
      for (long P = 0; P < Np; ++P)
        for (long m = 0; m < nb; ++m) XG(P, m) = Xq(P, m) / (z(iz) - e_rel(kq, m));
      Gt = XG * nda::dagger(Xq);
      for (long P = 0; P < Np; ++P)
        for (long Q = 0; Q < Np; ++Q) Gt(P, Q) *= dW[iq](P, Q);
      nda::matrix<ComplexType> S = nda::dagger(Xk) * Gt * Xk;
      out(iz, nda::range::all, nda::range::all) += (-1.0 / (beta * double(nk))) * S;
    }
  }
  return out;
}

bool run_t3(double beta, std::string const &time_grid) {
  using namespace methods;
  fx_t F("lih222");
  auto &comm    = F.mpi->comm;
  auto &mpi     = *F.mpi;
  auto &mf      = *F.mf;
  auto &thc     = *F.thc;
  const long nk = mf.nkpts(), nq = mf.nqpts(), nb = thc.nbnd(), Np = thc.Np();
  bool ok       = true;
  utils::TimerManager Timer;
  // ---- CoQui: G of the KS mean field at beta (mu by particle number, update_G), RPA W, G0W0 Sigma(tau) -> Sigma(i w_n)
  nda::array<double, 2> eig(nk, nb);
  double elo = 1e300, ehi = -1e300;
  for (long k = 0; k < nk; ++k)
    for (long n = 0; n < nb; ++n) {
      eig(k, n) = mf.eigval()(0, k, n);
      elo       = std::min(elo, eig(k, n));
      ehi       = std::max(ehi, eig(k, n));
    }
  const double w_max = std::max(std::abs(elo), std::abs(ehi)) + 2.0;
  imag_axes_ft::IAFT ft(beta, w_max + 1.0, imag_axes_ft::dlr_basis, "high");
  MBState mb_state(F.mpi, ft, "coqui_gw_line_ft_t3");
  simple_dyson dyson(F.mf.get(), &ft);
  const long ns = 1;
  mb_state.sF_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(mpi, {ns, nk, nb, nb}));
  mb_state.sDm_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(mpi, {ns, nk, nb, nb}));
  mb_state.sG_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(mpi, {ft.nt_f(), ns, nk, nb, nb}));
  mb_state.sSigma_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(mpi, {ft.nt_f(), ns, nk, nb, nb}));
  auto &sF     = mb_state.sF_skij.value();
  auto &sDm    = mb_state.sDm_skij.value();
  auto &sG     = mb_state.sG_tskij.value();
  auto &sSigma = mb_state.sSigma_tskij.value();
  hamilt::set_fock(mf, dyson.PSP(), sF, true);
  if (mpi.node_comm.root()) sSigma.local() = ComplexType(0.0);
  sSigma.communicator()->barrier();
  double mu_C = 0.0;
  update_G(dyson, mf, ft, sDm, sG, sF, sSigma, mu_C, false);
  solvers::scr_coulomb_t scr_im(&ft, "rpa", "ignore_g0");
  scr_im.update_w(mb_state, thc, -1);
  solvers::gw_t gw(&ft, "ignore_g0", "coqui_gw_line_ft_t3");
  gw.evaluate<HOST_MEMORY>(mb_state, thc);
  const long nwf = ft.nw_f();
  nda::array<ComplexType, 5> Sig_w(nwf, ns, nk, nb, nb);
  {
    nda::array<ComplexType, 5> Sig_t(sSigma.local());
    ft.tau_to_w(Sig_t, Sig_w, imag_axes_ft::fermion);
  }
  auto wn_f = ft.wn_mesh_f();
  // ---- the KS mu of this code (rule "number") and of the reference vs CoQui's
  thermal_params_t tp;
  const double deg = std::numbers::pi / 180.0;
  tp.beta = beta; tp.thermal_tol = 1e-12; tp.theta = 20.0 * deg; tp.theta_t = 10.0 * deg; tp.mu_rule = "number";
  double mu_ref = 0.0;
  nda::array<ComplexType, 4> nu0_ref;
  nda::array<long, 1> wn_ref;
  {
    h5::file hf(ft_dir() + "lih222_finiteT_ref.h5", 'r');
    h5::group root(hf);
    auto gb = root.open_group("beta_" + std::to_string(long(beta)));
    mu_ref  = read_attr<double>(gb, "mu0");
    nu0_ref = read_c<4>(gb, "nu0_term");
    nda::h5_read(gb, "w_n", wn_ref);
  }
  auto ks0 = pole_data_t::from_ks(eig, 0.0);
  std::vector<nda::array<double, 1>> le(nk);
  std::vector<nda::array<ComplexType, 2>> lv(nk);
  for (long k = 0; k < nk; ++k) {
    le[k] = eig(k, nda::range::all);
    lv[k] = nda::array<ComplexType, 2>(nb, nb);
    lv[k]() = 0.0;
    for (long n = 0; n < nb; ++n) lv[k](n, n) = 1.0;
  }
  const double mu_line = mu_rule_apply(le, lv, double(mf.nelec()), {}, tp).mu;
  // CoQui's update_G stops at a particle-number tolerance: in the gap dN/dmu is tiny (0.009 at beta 200), so its mu is
  // compared through the count: N_T(mu_CoQui) = N_el
  const double N_C = numerics::line_dlr::electron_count_T(le, lv, beta, mu_C);
  app_log(1, "\n[finiteT][T3] lih222 beta {} ({} grid): CoQui KS mu {:.12f}, this code (rule number) {:.12f} (diff {:.2e}), reference "
             "{:.12f}; IAFT {} fermionic / {} bosonic points",
          beta, time_grid, mu_C, mu_line, mu_line - mu_C, mu_ref, nwf, ft.nw_b());
  ok = gate("|mu_0 (this code) - mu_0 (reference)| (Ha)", std::abs(mu_line - mu_ref), 1e-10) and ok;
  ok = gate("|N_T(mu CoQui) - N_el| (CoQui's mu tolerance)", std::abs(N_C - double(mf.nelec())), 1e-8) and ok;
  // ---- the line at mu_C: thermal lists, D, the D-selected basis, tau leg, W, Sigma at the Matsubara points (upper images)
  auto ks = pole_data_t::from_ks(eig, mu_C);
  const double ET = tp.E_T();
  auto tl = thermal_lists(ks, beta, ET);
  nda::array<double, 2> e_rel(nk, nb);
  double emax = 0.0;
  for (long k = 0; k < nk; ++k)
    for (long n = 0; n < nb; ++n) {
      e_rel(k, n) = eig(k, n) - mu_C;
      emax        = std::max(emax, std::abs(e_rel(k, n)));
    }
  aux_grid_t grid(mpi, Np);
  propagator_t<HOST_MEMORY> prop(thc, grid);
  ibz_t ibz(mf, nb, false);
  bosonic_basis_t bline(tp.theta, 4.0, 1e-10, 0.0);
  auto Dm = make_bos_data(bline.zeta_nodes, tp);
  auto B  = bosonic_basis_t::from_data(tp.theta, Dm.z, 4.0, 1e-12, 800, -1.0, tp.cut_odd, tp.cut_even).with_bose(beta, ET);   // S8b.4: cut_odd 1e-11
  nda::array<ComplexType, 1> zi(nwf), ziu(nwf);
  for (long n = 0; n < nwf; ++n) {
    zi(n)  = (mu_C - mu_C) + ft.omega(wn_f(n));
    ziu(n) = zi(n).imag() > 0.0 ? zi(n) : std::conj(zi(n));
  }
  std::optional<numerics::line_dlr::time_nodes_t> rp, rh;
  if (time_grid == "gl") {
    rp.emplace(time_ray_t::guarded(tp.theta_t, beta, ET, 1e-5, 3.0, 16, sector_t::particle));
    rh.emplace(time_ray_t::guarded(tp.theta_t, beta, ET, 1e-5, 3.0, 16, sector_t::hole));
  } else {
    numerics::line_dlr::time_id_opts_t o;
    o.pad  = 1.25;
    auto T = line_time_grids_t::thermal(tp.theta_t, 2.0 * ET, 2.0 * emax + nda::max_element(B.nu), tp.S_T(), 1e-10, o, Dm.z, ziu, comm);
    rp.emplace(T.pi_p);
    rh.emplace(T.pi_h);
  }
  auto tn = make_tau_nodes(tp, emax, 2.0 * emax);
  std::vector<long> qall(nq);
  std::iota(qall.begin(), qall.end(), 0L);
  memory::array<HOST_MEMORY, ComplexType, 4> Pt, Pi, w, Wn;
  nda::array<ComplexType, 1> z0(1);
  z0(0) = 0.0;
  pi_tau_leg<HOST_MEMORY>(prop, ks, mf, ibz, grid, tp, tn, z0, 8, Pt, Timer, qall);
  polarization<HOST_MEMORY>(prop, tl, mf, grid, Dm.z, *rp, *rh, 8, Pi, Timer, sector_t::both, qall);
  for (long i = 0; i < Dm.z.size(); ++i)
    if (Dm.z(i) == ComplexType(0.0))
      for (long q = 0; q < nq; ++q) Pi(q, i, nda::range::all, nda::range::all) = Pt(q, 0, nda::range::all, nda::range::all);
  // dW(q) = W[Pi^Mats(q, 0)] - W[Pi^an(q, 0)] (full matrices), Pi^Mats = Pi^an + dPi, for the KS poles at mu
  coulomb_blocks_t<HOST_MEMORY> Zb(thc, grid, qall, Timer);
  auto dW_at = [&](double mu, memory::array<HOST_MEMORY, ComplexType, 4> const *Pt_in) {
    std::vector<nda::array<ComplexType, 2>> dW(nq);
    auto ksm = pole_data_t::from_ks(eig, mu);
    memory::array<HOST_MEMORY, ComplexType, 4> Ptm;
    if (Pt_in == nullptr) pi_tau_leg<HOST_MEMORY>(prop, ksm, mf, ibz, grid, tp, tn, z0, 8, Ptm, Timer, qall);
    auto const &P0 = Pt_in ? *Pt_in : Ptm;
    auto pf       = unfold_poles(ksm, ibz);
    auto [Xp, Xq] = host_x_slices(prop);
    auto dPi      = pi_tau_dpi(pf, Xp, Xq, mf, qall, beta, 1e-8);
    for (long q = 0; q < nq; ++q) {
      nda::array<ComplexType, 2> pa(P0(q, 0, nda::range::all, nda::range::all)), pd(dPi(q, nda::range::all, nda::range::all));
      auto Pa = full_of(pa, grid, comm), Pd = full_of(pd, grid, comm);
      nda::matrix<ComplexType> Z(Zb.full(q)), I(Np, Np), Wa, Wm;
      I = 0.0;
      for (long P = 0; P < Np; ++P) I(P, P) = 1.0;
      nda::matrix<ComplexType> Ma = nda::matrix<ComplexType>(Pa), Mm = nda::matrix<ComplexType>(Pa + Pd);
      Wa    = nda::inverse(I - Z * Ma) * Z;
      Wm    = nda::inverse(I - Z * Mm) * Z;
      dW[q] = nda::array<ComplexType, 2>(Wm - Wa);
    }
    return dW;
  };
  auto dW = dW_at(mu_C, &Pt);
  screened_interaction<HOST_MEMORY>(Pi, Zb, B, grid, mpi, w, Timer, &Wn, qall, false);
  std::optional<numerics::line_dlr::time_nodes_t> sp, sh;
  if (time_grid == "gl") {
    sp = rp;
    sh = rh;
  } else {
    numerics::line_dlr::time_id_opts_t o;
    o.pad  = 1.25;
    auto T = line_time_grids_t::thermal(tp.theta_t, 2.0 * ET, emax + nda::max_element(B.nu), tp.S_T(), 1e-10, o, Dm.z, ziu, comm);
    sp.emplace(T.sig_p);
    sh.emplace(T.sig_h);
  }
  nda::array<ComplexType, 4> Sp, Sh;
  self_energy<HOST_MEMORY>(prop, tl, w, B, mf, grid, mpi, ziu, *sp, *sh, 8, Sp, Timer, sector_t::both, false, nullptr, 0, &Sh);
  // ---- the nu_0 term: this code's formula vs the reference's (at its w_n, k = 0, 3), then at the IAFT points
  {
    nda::array<ComplexType, 1> zr(wn_ref.size());
    for (long i = 0; i < zr.size(); ++i) zr(i) = ComplexType(0.0, (2.0 * wn_ref(i) + 1.0) * std::numbers::pi / beta);
    double d = 0.0, m = 0.0;
    const long ksel[2] = {0, 3};
    auto dWr = dW_at(mu_ref, nullptr);   // at the reference's mu_0 (beta |dmu| ~ 1e-6 relative otherwise)
    nda::array<double, 2> e_ref(nk, nb);
    for (long k = 0; k < nk; ++k)
      for (long n = 0; n < nb; ++n) e_ref(k, n) = eig(k, n) - mu_ref;
    for (int a = 0; a < 2; ++a) {
      auto T = nu0_term(thc, mf, e_ref, dWr, ksel[a], zr, beta);
      for (long i = 0; i < zr.size(); ++i)
        for (long x = 0; x < nb; ++x)
          for (long y = 0; y < nb; ++y) {
            d = std::max(d, std::abs(T(i, x, y) - nu0_ref(a, i, x, y)));
            m = std::max(m, std::abs(nu0_ref(a, i, x, y)));
          }
    }
    ok = gate("nu_0 term (this code) vs the reference's", d / m, 1e-8) and ok;
  }
  double d = 0.0, m = 0.0, mt = 0.0;
  long npts = 0;
  for (long k = 0; k < nk; ++k) {
    auto T = nu0_term(thc, mf, e_rel, dW, k, zi, beta);
    for (long n = 0; n < nwf; ++n) {
      if (std::abs(zi(n).imag()) < tp.zeta_T()) continue;
      ++npts;
      nda::matrix<ComplexType> Su(Sp(k, n, nda::range::all, nda::range::all) + Sh(k, n, nda::range::all, nda::range::all));
      nda::matrix<ComplexType> Sl = zi(n).imag() > 0.0 ? Su : nda::matrix<ComplexType>(nda::dagger(Su));
      for (long x = 0; x < nb; ++x)
        for (long y = 0; y < nb; ++y) {
          d  = std::max(d, std::abs(Sig_w(n, 0, k, x, y) - T(n, x, y) - Sl(x, y)));
          m  = std::max(m, std::abs(Sig_w(n, 0, k, x, y)));
          mt = std::max(mt, std::abs(T(n, x, y)));
        }
    }
  }
  app_log(1, "  {} Matsubara points |w_n| >= zeta_T x {} k; max|Sigma_c| {:.3e}, max|nu_0 term| {:.3e}", npts / nk, nk, m, mt);
  ok = gate("T3 CoQui Sigma_c(i w_n) - nu_0 term vs line (rel)", d / m, 1e-7) and ok;
  return ok;
}

} // namespace

TEST_CASE("gw_line_finiteT_T3_coqui", "[gw_line][finiteT][T3]") {
  bool ok = run_t3(200.0, "id");
  ok      = run_t3(50.0, "id") and ok;
  REQUIRE(ok);
}

namespace {
void bcast_nda_(mpi3::communicator &comm, nda::array<double, 1> &a) {
  long n = a.size();
  comm.broadcast_n(&n, 1, 0);
  if (not comm.root()) a.resize(n);
  if (n > 0) comm.broadcast_n(a.data(), n, 0);
}
} // namespace

// ============================================================================================================== driver (T4, T5)
namespace {

std::string lih222_thc() { return ft_dir() + "lih222_thc/thc.eri.h5"; }

/// the settings of the finite-T SCF reference (gen_finiteT_scf_ref.py attrs) / the T4 pair
ptree ft_params(std::string const &out, long niter, double beta, std::string const &time_grid, std::string const &mu_rule,
                bool restart = false, bool parity = false) {
  ptree pt;
  pt.put("time_grid", time_grid);
  pt.put("g_repr", "lehmann");
  if (parity) {   // the prototype keeps every Lehmann pole, cuts the Gram matrix at tol_gram only, and its tau grid is GL
    pt.put("tau_grid", "gl");
    pt.put("g_emax", 1e4);
    pt.put("g_wtol", 0.0);
    pt.put("g_emin_frac", 0.0);
    pt.put("tol_gram_eps", 0.0);
  }
  pt.put("theta_deg", 20.0);
  pt.put("eps", 1e-8);
  pt.put("lam", 6.0);
  pt.put("lam_b", 12.0);
  pt.put("sigma_gap", 0.02);
  pt.put("bos_gap", 0.02);
  pt.put("g_gap", 0.0);
  pt.put("nodes_per_ray", 120);
  pt.put("node_tmin", 1e-3);
  pt.put("node_tmax", 60.0);
  pt.put("wp", 0.11);
  pt.put("K", 8);
  pt.put("tol_gram", 1e-10);
  pt.put("nphi", 8);
  pt.put("niter", niter);
  pt.put("mixing", 0.5);
  pt.put("damp_below", 0.0);
  pt.put("conv_thr", 1e-14);
  pt.put("t_chunk", 8);
  pt.put("restart", restart);
  pt.put("output", out);
  pt.put("checkpoint_sigma", "all");
  pt.put("spectra.enable", false);
  pt.put("beta", beta);
  pt.put("thermal_tol", 1e-8);
  pt.put("thermal_floor", 30.0);
  pt.put("thermal_floor_f", 30.0);
  pt.put("wp_floor", 15.0);
  pt.put("mu_rule", mu_rule);
  pt.put("bos_line_eps", 1e-10);
  pt.put("bos_eps_T", 1e-12);
  return pt;
}

struct lih_ft_t {
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;
  std::shared_ptr<mf::MF> mf;
  std::unique_ptr<methods::thc_reader_t> thc;
  lih_ft_t() {
    mpi = utils::make_unit_test_mpi_context();
    mf  = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222"));
    thc = std::make_unique<methods::thc_reader_t>(mf, "incore", lih222_thc());
  }
};

/// total Sigma (Sigma_p + Sigma_h) of iteration it at all k and nodes (root reads, broadcast)
nda::array<ComplexType, 4> sigma_total_of(mpi3::communicator &comm, std::string const &file, long it) {
  nda::array<ComplexType, 4> Sp, Sh;
  if (comm.root()) {
    h5::file f(file, 'r');
    h5::group g(f);
    auto gi = g.open_group("scf_line/iter" + std::to_string(it));
    nda::h5_read(gi, "Sigma_p", Sp);
    nda::h5_read(gi, "Sigma_h", Sh);
    Sp += Sh;
  }
  std::array<long, 4> shp{};
  if (comm.root()) shp = Sp.shape();
  comm.broadcast_n(shp.data(), 4, 0);
  if (not comm.root()) Sp.resize(shp);
  comm.broadcast_n(Sp.data(), Sp.size(), 0);
  return Sp;
}

void rm_ckpt(mpi3::communicator &comm, std::string const &stem) {
  comm.barrier();
  if (comm.root() and std::getenv("GW_LINE_KEEP_CKPT") == nullptr)   // D4: keep the checkpoints for post-processing checks
    for (auto sfx : {".gw_line.h5", ".gw_line.sigma.h5"}) std::filesystem::remove(stem + sfx);
  comm.barrier();
}

/// bitwise comparison of two runs (mu, poles, F, Sigma at the nodes per iteration)
bool same_runs(mpi3::communicator &comm, gw_line_result_t const &A, gw_line_result_t const &B, std::string const &fa,
               std::string const &fb, long niter) {
  bool ok = A.history.size() == B.history.size();
  for (size_t i = 0; ok and i < A.history.size(); ++i) ok = (A.history[i].mu == B.history[i].mu) and (A.history[i].gap == B.history[i].gap);
  double dF = 0.0;
  for (long a = 0; a < A.F.size(); ++a) dF = std::max(dF, std::abs(A.F.data()[a] - B.F.data()[a]));
  double dP = 0.0;
  for (long k = 0; k < A.poles.nk; ++k)
    for (int s = 0; s < 2; ++s) {
      auto const &pa = s ? A.poles.part[k] : A.poles.hole[k];
      auto const &pb = s ? B.poles.part[k] : B.poles.hole[k];
      if (pa.size() != pb.size()) { dP = 1e300; continue; }
      for (long m = 0; m < pa.size(); ++m) dP = std::max(dP, std::abs(pa.e(m) - pb.e(m)));
      for (long a = 0; a < pa.v.size(); ++a) dP = std::max(dP, std::abs(pa.v.data()[a] - pb.v.data()[a]));
    }
  double dS = 0.0;
  for (long it = 1; it <= niter; ++it) {
    auto Sa = sigma_total_of(comm, fa + ".gw_line.h5", it), Sb = sigma_total_of(comm, fb + ".gw_line.h5", it);
    for (long a = 0; a < Sa.size(); ++a) dS = std::max(dS, std::abs(Sa.data()[a] - Sb.data()[a]));
  }
  app_log(1, "    bitwise check: history mu/gap {}, max|dF| {:.1e}, max|d poles| {:.1e}, max|dSigma| {:.1e}", ok ? "identical" : "DIFFER", dF,
          dP, dS);
  return ok and dF == 0.0 and dP == 0.0 and dS == 0.0;
}

} // namespace

// T4: beta = 1e4 (no window at any iteration) == the T = 0 run, bitwise (same binary, thermal mode off)
TEST_CASE("gw_line_finiteT_T4_bitwise", "[gw_line][finiteT][T4]") {
  lih_ft_t L;
  auto &comm = L.mpi->comm;
  bool ok    = true;
  for (std::string tg : {"id", "gl"}) {
    const std::string fa = "gw_line_ft_t4_T0_" + tg, fb = "gw_line_ft_t4_b1e4_" + tg;
    auto A = gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, ft_params(fa, 2, 0.0, tg, "auto"));
    auto B = gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, ft_params(fb, 2, 1e4, tg, "auto"));
    app_log(1, "\n[finiteT][T4] time grid {}: T = 0 mu {:.12f} / {:.12f}, beta 1e4 mu {:.12f} / {:.12f}", tg, A.history[0].mu,
            A.history[1].mu, B.history[0].mu, B.history[1].mu);
    ok = same_runs(comm, A, B, fa, fb, 2) and ok;
    rm_ckpt(comm, fa);
    rm_ckpt(comm, fb);
  }
  REQUIRE(ok);
}

// T5 (a): 3 iterations vs the prototype's thermal SCF (lih222_finiteT_scf_ref.h5), bases / D / nu_b injected, mu_rule "auto"
// (the reference's rule per iteration: gap / number / number). Gates = 10 x the measured floor.
// S8b.4 (free_block "orbital", cut_odd 1e-11; Mac 2026-10-17): parity |dmu| 4.1e-3 / 1.3e-2 / 0.12 meV, Sigma (masked nodes)
// 5.2e-14 / 1.3e-7 / 4.2e-7 at iterations 1 / 2 / 3; noise meter (GW_LINE_FT_NOISE = 1e-13 / 2e-13 on the iteration-1 Sigma vs
// none): |dmu| 4.0e-3 / 2.7e-3 / 4.4e-3 meV, Sigma 6.6e-13 / 7.3e-9 / 5.9e-7 (iteration 1: the injected noise itself). The floors
// below round these up (x 2-3, room for the MKL / rank spreads). Pre-S8b.4 floors (SVD pairing of the free terminal block, n_free
// 16 at iteration 1): 152 / 789 / 825 meV and 1e-12 / 1.7e-3 / 2.8e-2 -- the closure floor was the gauge, not the data.
constexpr double FT_MU_FLOOR[3]  = {0.01, 0.05, 0.3};       // meV
constexpr double FT_SIG_FLOOR[3] = {1e-12, 3e-7, 1.5e-6};   // relative, masked nodes
TEST_CASE("gw_line_finiteT_parity", "[gw_line][finiteT][parity]") {
  const std::string ref = ft_dir() + "lih222_finiteT_scf_ref.h5";
  REQUIRE(std::filesystem::exists(ref));
  lih_ft_t L;
  auto &comm = L.mpi->comm;
  nda::array<long, 1> idx;
  nda::array<signed char, 1> fmask;
  std::vector<double> mu_r, N_r, gap_r;
  std::vector<std::string> rule_r;
  std::vector<nda::array<ComplexType, 3>> S0_r, S3_r;
  long niter = 0;
  double cut_odd_r = 1e-13;
  std::string free_block_r = "svd";
  nda::array<ComplexType, 2> F_r;
  {
    h5::file f(ref, 'r');
    h5::group g(f);
    nda::h5_read(g, "node_index", idx);
    nda::h5_read(g, "fmask", fmask);
    niter = read_attr<long>(g, "niter");
    cut_odd_r = read_attr<double>(g, "wstep_cut_odd");
    if (g.has_dataset("mixing_F_re")) {   // S8b.4 references: the closure's free_block rule and its fixed unitary F
      free_block_r = read_attr<std::string>(g, "free_block");
      F_r          = read_c<2>(g, "mixing_F");
    }
    for (long it = 1; it <= niter; ++it) {
      auto gi = g.open_group("iter" + std::to_string(it));
      mu_r.push_back(read_attr<double>(gi, "mu"));
      N_r.push_back(read_attr<double>(gi, "N_mu"));
      gap_r.push_back(read_attr<double>(gi, "gap"));
      rule_r.push_back(read_attr<std::string>(gi, "mu_rule_used"));
      S0_r.push_back(read_c<3>(gi, "Sigma_k0"));
      S3_r.push_back(read_c<3>(gi, "Sigma_k3"));
    }
  }
  std::string fo = "gw_line_ft_parity";
  if (char const *e = std::getenv("GW_LINE_FT_PARITY_NIT")) niter = std::min(niter, long(std::atol(e)));   // diagnostics
  if (char const *e = std::getenv("GW_LINE_FT_TAG")) fo += std::string("_") + e;
  // S8b.4: with the continuous closure the "auto" rule of the reference is reproduced (before: "number" pinned, the rule
  // decision flipped with the closure noise); the rule per iteration is gated equal to the prototype's
  auto pt = ft_params(fo, niter, 200.0, "gl", "auto", false, true);
  if (char const *e = std::getenv("GW_LINE_FT_NOISE")) {   // noise-floor meter: relative noise on Sigma of iteration 1
    pt.put("debug_noise_sigma", std::atof(e));
    pt.put("debug_noise_iter", 1);
  }
  if (std::getenv("GW_LINE_FT_CLOSURE_PY")) {   // diagnostics: the pre-S7g closure drivers (python's gesvd / Schur)
    pt.put("closure_svd", "gesvd");
    pt.put("closure_ueig", "schur");
  }
  pt.put("thermal_bases_file", ref);
  pt.put("cut_odd", cut_odd_r);                 // S8b.4: the reference's split-fit cut (the C++ default 1e-11)
  pt.put("closure_free_block", free_block_r);   // S8b.4: the reference's free-block rule ("orbital")
  bool ok = true;
  if (F_r.size() > 0) {   // the closed-form F of the orbital rule = the prototype's
    auto const &F = numerics::line_dlr::mixing_unitary(F_r.extent(0));
    double dF = 0.0;
    for (long i = 0; i < F_r.extent(0); ++i)
      for (long j = 0; j < F_r.extent(1); ++j) dF = std::max(dF, std::abs(F(i, j) - F_r(i, j)));
    app_log(1, "\n[finiteT][parity] reference: free_block {}, cut_odd {:.0e}", free_block_r, cut_odd_r);
    ok = gate("mixing unitary F (nb x nb) vs the reference's", dF, 1e-12) and ok;
  }
  auto t0 = std::chrono::steady_clock::now();
  auto R  = gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pt);
  const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  REQUIRE(long(R.history.size()) == niter);
  app_log(1, "\n[finiteT][parity] {} iterations in {:.1f} s ({} ranks)", niter, dt, comm.size());
  app_log(1, "  iter |   mu C++ (Ha)       mu py (Ha)      d(meV)  |  N(mu) C++        N py     | rule | Sigma(k=0,3) rel");
  for (long it = 0; it < niter; ++it) {
    auto S = sigma_total_of(comm, fo + ".gw_line.h5", it + 1);
    double d = 0.0, m = 0.0;
    for (long n = 0; n < idx.size(); ++n)
      if (fmask(idx(n)))
      for (long a = 0; a < S.extent(2); ++a)
        for (long b = 0; b < S.extent(3); ++b) {
          d = std::max({d, std::abs(S(0, idx(n), a, b) - S0_r[it](n, a, b)), std::abs(S(3, idx(n), a, b) - S3_r[it](n, a, b))});
          m = std::max({m, std::abs(S0_r[it](n, a, b)), std::abs(S3_r[it](n, a, b))});
        }
    auto const &h = R.history[it];
    const double dmu = (h.mu - mu_r[it]) * 27.211386e3;
    app_log(1, "  {:4d} | {:.12f}  {:.12f}  {:+.2e} | {:.12f}  {:.12f} | {} | {:.2e}", it + 1, h.mu, mu_r[it], dmu, h.N_mu, N_r[it],
            h.mu_rule, d / m);
    ok = gate("|dmu| (meV) [10 x floor]", std::abs(dmu), 10.0 * FT_MU_FLOOR[std::min(it, 2L)]) and ok;
    if (h.mu_rule != rule_r[it]) {
      app_log(1, "    mu rule {} vs the reference's {}: FAIL", h.mu_rule, rule_r[it]);
      ok = false;
    }
    if (h.mu_rule == "number") ok = gate("|N(mu) - N_el| (rule number)", std::abs(h.N_mu - 4.0), 1e-10) and ok;
    ok = gate("Sigma(k = 0, 3) rel, masked nodes [10 x floor]", d / m, 10.0 * FT_SIG_FLOOR[std::min(it, 2L)]) and ok;
  }
  if (not std::getenv("GW_LINE_FT_KEEP")) rm_ckpt(comm, fo);
  REQUIRE(ok);
}

// T5 (b): 5 iterations, mu_rule "neutral" (the default since S8b.4 (e); ID grids): the rule per iteration, mu in the central half
// of the admissible gap (with "auto" the rule flipped gap -> number at iteration 2 and mu sat at 0.275 of the gap at iteration 3:
// the closure's count error dN ~ 4e-3 ~ n_th compensated by moving mu towards an edge; "neutral" removes the error first),
// N(mu) of the Lehmann G, restart bitwise (3 + restart vs 5); 1 vs 2 ranks: mu per iteration written to
// gw_line_ft_scf_np<np>.txt and compared with the other rank count's file when present (run 1 then 2 ranks)
TEST_CASE("gw_line_finiteT_scf", "[gw_line][finiteT][scf]") {
  lih_ft_t L;
  auto &comm = L.mpi->comm;
  const long NIT = 5, NR = 3;
  const std::string fa = "gw_line_ft_scf_a", fb = "gw_line_ft_scf_b";
  auto t0 = std::chrono::steady_clock::now();
  auto A  = gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, ft_params(fa, NIT, 200.0, "id", "neutral"));
  const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  bool ok = long(A.history.size()) == NIT;
  app_log(1, "\n[finiteT][scf] {} iterations in {:.1f} s ({} ranks), beta 200, mu_rule neutral, time grid id", NIT, dt, comm.size());
  for (auto const &h : A.history) {
    const double D = h.gap, off = std::abs(0.5 * (h.e_homo + h.e_lumo));
    app_log(1, "  iter {}: mu {:.12f} rule {:8s} dN {:+.3e} n_th {:.2e} N(mu) {:.12f} gap {:.4f} eV, |mu - mid gap| / gap {:.3f}, "
               "thermal {}, wp {:.3f}, |D| {}, rank_b {}, tau nodes {}, {:.1f} s",
            h.iter, h.mu, h.mu_rule, h.dN, h.n_th, h.N_mu, D * 27.211386, off / D, h.thermal, h.wp_used, h.nD, h.rank_b, h.ntau, h.time);
    // plan T5(b): mu in the central half of the gap (<= 0.25); rule "neutral" at every iteration (|eps0| and n_th <= 0.1)
    ok = gate("|mu - gap midpoint| / gap (central half: <= 0.25)", off / D, 0.25) and ok;
    ok = (h.mu_rule == "neutral") and ok;
    ok = gate("|dN| (closure count error at the midpoint)", std::abs(h.dN), 0.1) and ok;
  }
  {   // restart: NR iterations, then resume to NIT
    gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, ft_params(fb, NR, 200.0, "id", "neutral"));
    auto B = gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, ft_params(fb, NIT, 200.0, "id", "neutral", true));
    app_log(1, "  restart {} + {} vs {} straight:", NR, NIT - NR, NIT);
    ok = same_runs(comm, A, B, fa, fb, NIT) and ok;
  }
  {   // 1 vs 2 ranks: the iteration-1 Sigma (kernels on the KS lists: reduction order only) to 1e-12; mu per iteration within
      // 10 x the closure floor of these settings (FT_MU_FLOOR, S8b.4: 1e-13 in Sigma moves mu by 4e-3 meV at iteration 1)
    const std::string mine = "gw_line_ft_scf_np" + std::to_string(comm.size()) + ".h5";
    const std::string other = "gw_line_ft_scf_np" + std::to_string(comm.size() == 1 ? 2 : 1) + ".h5";
    nda::array<double, 1> mus(long(A.history.size()));
    for (long i = 0; i < mus.size(); ++i) mus(i) = A.history[i].mu;
    auto S1 = sigma_total_of(comm, fa + ".gw_line.h5", 1);
    if (comm.root()) {
      h5::file f(mine, 'w');
      h5::group g(f);
      nda::h5_write(g, "mu", mus, false);
      nda::h5_write(g, "Sigma1", S1, false);
    }
    comm.barrier();
    if (std::filesystem::exists(other)) {
      nda::array<double, 1> mo;
      nda::array<ComplexType, 4> So;
      {
        h5::file f(other, 'r');
        h5::group g(f);
        nda::h5_read(g, "mu", mo);
        nda::h5_read(g, "Sigma1", So);
      }
      // S8b.4: the masked nodes rho beta |zeta| >= c_f only (the reference's mask: same fermionic nodes); sub-floor reported
      nda::array<signed char, 1> fm;
      {
        h5::file f(ft_dir() + "lih222_finiteT_scf_ref.h5", 'r');
        h5::group g(f);
        nda::h5_read(g, "fmask", fm);
      }
      REQUIRE(fm.size() == S1.extent(1));
      double ds = 0.0, ms = 0.0, du = 0.0;
      for (long k = 0; k < S1.extent(0); ++k)
        for (long n = 0; n < S1.extent(1); ++n)
          for (long i = 0; i < S1.extent(2); ++i)
            for (long j = 0; j < S1.extent(3); ++j) {
              const double d = std::abs(S1(k, n, i, j) - So(k, n, i, j));
              if (fm(n)) {
                ds = std::max(ds, d);
                ms = std::max(ms, std::abs(S1(k, n, i, j)));
              } else
                du = std::max(du, d);
            }
      app_log(1, "  {} vs {}: iteration-1 Sigma below the floor (report) max|d| {:.3e}", mine, other, du);
      ok = gate("1 vs 2 ranks: iteration-1 Sigma rel (masked nodes)", ds / ms, 1e-12) and ok;
      for (long i = 0; i < std::min(mo.size(), mus.size()); ++i)
        ok = gate(("1 vs 2 ranks: |dmu| (meV) iteration " + std::to_string(i + 1) + " [10 x floor]").c_str(),
                  std::abs(mo(i) - mus(i)) * 27.211386e3, 10.0 * FT_MU_FLOOR[std::min(i, 2L)]) and ok;
    } else
      app_log(1, "  {} written; run with the other rank count to compare", mine);
  }
  rm_ckpt(comm, fa);
  rm_ckpt(comm, fb);
  REQUIRE(ok);
}

// ======================================================================================== W-stage diagnostics (hidden)
namespace {
struct wdbg_out_t {
  nda::array<ComplexType, 4> Pi, Wn, w, S;
  std::vector<char> fmask;   ///< S8b.4: fermionic node kept (rho beta |zeta| >= c_f): Sigma_c is an output only there
};
/// S8b.4 (notes sec:fT_wcond): max |a - b| / max |a| of Sigma (k, node, nb, nb) over the nodes with mask == want (all ranks)
template <typename C> double rel_nodes(C &comm, nda::array<ComplexType, 4> const &a, nda::array<ComplexType, 4> const &b,
                                       std::vector<char> const &mask, bool want) {
  double d = 0.0, m = 0.0;
  for (long k = 0; k < a.extent(0); ++k)
    for (long n = 0; n < a.extent(1); ++n) {
      if ((mask[n] != 0) != want) continue;
      for (long i = 0; i < a.extent(2); ++i)
        for (long j = 0; j < a.extent(3); ++j) {
          d = std::max(d, std::abs(a(k, n, i, j) - b(k, n, i, j)));
          m = std::max(m, std::abs(a(k, n, i, j)));
        }
    }
  d = comm.all_reduce_value(d, mpi3::max<>{});
  m = comm.all_reduce_value(m, mpi3::max<>{});
  return d / std::max(m, 1e-300);
}
/// the thermal pass of [finiteT][device] in MEM with injection points: Pi_in (host, the block layout of this rank) replaces
/// the computed Pi, noise > 0 multiplies it by (1 + noise x uniform(-1, 1)), w_in (host) replaces the W stage
template <MEMORY_SPACE MEM>
wdbg_out_t wdbg_run(fx_t &F, double beta, double mu0, nda::array<ComplexType, 4> const *Pi_in, double noise,
                    nda::array<ComplexType, 4> const *w_in) {
  auto &mf = *F.mf;
  auto &thc = *F.thc;
  const long nk = mf.nkpts(), nq = mf.nqpts(), nb = thc.nbnd(), Np = thc.Np();
  const double deg = std::numbers::pi / 180.0;
  thermal_params_t tp;
  tp.beta = beta; tp.thermal_tol = 1e-12; tp.theta = 20.0 * deg; tp.theta_t = 10.0 * deg;
  const double ET = tp.E_T();
  utils::TimerManager Timer;
  nda::array<double, 2> eig(nk, nb);
  double emax = 0.0;
  for (long k = 0; k < nk; ++k)
    for (long n = 0; n < nb; ++n) {
      eig(k, n) = mf.eigval()(0, k, n);
      emax      = std::max(emax, std::abs(eig(k, n) - mu0));
    }
  auto ks = pole_data_t::from_ks(eig, mu0);
  auto tl = thermal_lists(ks, beta, ET);
  aux_grid_t grid(*F.mpi, Np);
  propagator_t<MEM> prop(thc, grid);
  bosonic_basis_t bline(tp.theta, 4.0, 1e-10, 0.0);
  auto Dm = make_bos_data(bline.zeta_nodes, tp);
  auto B  = bosonic_basis_t::from_data(tp.theta, Dm.z, 4.0, 1e-12).with_bose(beta, ET);
  auto zf = numerics::line_dlr::dense_nodes(tp.theta, 1e-3, 60.0, 40);
  numerics::line_dlr::time_id_opts_t o;
  o.pad  = 1.25;
  auto T = line_time_grids_t::thermal(tp.theta_t, 2.0 * ET, 2.0 * emax + nda::max_element(B.nu), tp.S_T(), 1e-10, o, Dm.z, zf,
                                      F.mpi->comm);
  std::vector<long> qall(nq);
  std::iota(qall.begin(), qall.end(), 0L);
  wdbg_out_t out;
  memory::array<MEM, ComplexType, 4> Pi, w, Wn;
  if (Pi_in == nullptr) {
    ibz_t ibz(mf, nb, false);
    auto tn = make_tau_nodes(tp, emax, 2.0 * emax);
    memory::array<MEM, ComplexType, 4> Pt;
    nda::array<ComplexType, 1> z0(1);
    z0(0) = 0.0;
    pi_tau_leg<MEM>(prop, ks, mf, ibz, grid, tp, tn, z0, 0, Pt, Timer, qall);
    polarization<MEM>(prop, tl, mf, grid, Dm.z, T.pi_p, T.pi_h, 0, Pi, Timer, sector_t::both, qall);
    for (long i = 0; i < Dm.z.size(); ++i)
      if (Dm.z(i) == ComplexType(0.0))
        for (long q = 0; q < nq; ++q) Pi(q, i, nda::range::all, nda::range::all) = Pt(q, 0, nda::range::all, nda::range::all);
    out.Pi = nda::array<ComplexType, 4>(memory::to_memory_space<HOST_MEMORY>(Pi));
  } else {
    out.Pi = *Pi_in;
  }
  if (noise > 0.0) {
    std::srand(12345 + F.mpi->comm.rank());
    for (long a = 0; a < out.Pi.size(); ++a)
      out.Pi.data()[a] *= 1.0 + noise * (2.0 * double(std::rand()) / double(RAND_MAX) - 1.0);
  }
  Pi = memory::to_memory_space<MEM>(out.Pi);
  if (w_in == nullptr) {
    coulomb_blocks_t<MEM> Zb(thc, grid, qall, Timer);
    screened_interaction<MEM>(Pi, Zb, B, grid, *F.mpi, w, Timer, &Wn, qall, false);
    out.Wn = nda::array<ComplexType, 4>(memory::to_memory_space<HOST_MEMORY>(Wn));
    out.w  = nda::array<ComplexType, 4>(memory::to_memory_space<HOST_MEMORY>(w));
  } else {
    out.w = *w_in;
    w     = memory::to_memory_space<MEM>(out.w);
  }
  nda::array<ComplexType, 4> Sp, Sh;
  self_energy<MEM>(prop, tl, w, B, mf, grid, *F.mpi, zf, T.sig_p, T.sig_h, 0, Sp, Timer, sector_t::both, false, nullptr, 0, &Sh);
  out.S = Sp + Sh;
  out.fmask.assign(zf.size(), 0);
  long nkeep = 0;
  for (long i = 0; i < zf.size(); ++i) nkeep += (out.fmask[i] = (tp.rho() * beta * std::abs(zf(i)) >= tp.c_f) ? 1 : 0);
  app_log(1, "  wdbg: basis rank {} (fit {}, Bose rows {}), {} nodes, mirror_half {}, cut_odd {:.0e}; fermionic nodes {} of which {} "
             "above the floor rho beta |zeta| >= {}",
          B.rank, B.rank_fit(), B.n_bose, B.zeta_nodes.size(), numerics::line_dlr::mirror_half(B.zeta_nodes), tp.cut_odd, zf.size(), nkeep,
          tp.c_f);
  return out;
}
} // namespace

// [.][finiteT][wdbg]: where host and device (or two host W paths) diverge in the thermal pass of [finiteT][device]
TEST_CASE("gw_line_finiteT_wdbg", "[.][finiteT][wdbg]") {
  fx_t F("lih222");
  const double beta = 50.0;
  double mu0 = 0.0;
  {
    h5::file hf(ft_dir() + "lih222_finiteT_ref.h5", 'r');
    h5::group root(hf);
    auto gb = root.open_group("beta_50");
    mu0     = read_attr<double>(gb, "mu0");
  }
  auto &comm = F.mpi->comm;
  auto rel   = [&](auto const &a, auto const &b) {
    double d = 0.0, m = 0.0;
    for (long i = 0; i < a.size(); ++i) {
      d = std::max(d, std::abs(a.data()[i] - b.data()[i]));
      m = std::max(m, std::abs(a.data()[i]));
    }
    d = comm.all_reduce_value(d, mpi3::max<>{});
    m = comm.all_reduce_value(m, mpi3::max<>{});
    return d / m;
  };
  auto report = [&](char const *what, wdbg_out_t const &a, wdbg_out_t const &b) {
    app_log(1, "  {:<44s} Pi {:.3e}  W(nodes) {:.3e}  w {:.3e}  Sigma masked {:.3e} (sub-floor {:.3e})", what, rel(a.Pi, b.Pi),
            (a.Wn.size() == b.Wn.size() and a.Wn.size() > 0) ? rel(a.Wn, b.Wn) : -1.0, rel(a.w, b.w), rel_nodes(comm, a.S, b.S, a.fmask, true),
            rel_nodes(comm, a.S, b.S, a.fmask, false));
  };
  auto H0 = wdbg_run<HOST_MEMORY>(F, beta, mu0, nullptr, 0.0, nullptr);
  {
    auto H = wdbg_run<HOST_MEMORY>(F, beta, mu0, &H0.Pi, 0.0, nullptr);
    report("host node path, again", H0, H);
  }
  {
    setenv("COQUI_GWLINE_W_REDIST", "old", 1);
    auto H = wdbg_run<HOST_MEMORY>(F, beta, mu0, &H0.Pi, 0.0, nullptr);
    unsetenv("COQUI_GWLINE_W_REDIST");
    report("host mirror redistribute path (old)", H0, H);
  }
  {
    setenv("COQUI_GWLINE_W_MIRROR", "0", 1);
    auto H = wdbg_run<HOST_MEMORY>(F, beta, mu0, &H0.Pi, 0.0, nullptr);
    unsetenv("COQUI_GWLINE_W_MIRROR");
    report("host full-node path (mirror off)", H0, H);
  }
  for (double nz : {1e-16, 1e-15, 1e-14}) {
    auto H = wdbg_run<HOST_MEMORY>(F, beta, mu0, &H0.Pi, nz, nullptr);
    char lab[64];
    std::snprintf(lab, sizeof(lab), "host, Pi x (1 + %.0e noise)", nz);
    report(lab, H0, H);
  }
#if defined(ENABLE_DEVICE)
  {
    auto D = wdbg_run<DEVICE_MEMORY>(F, beta, mu0, nullptr, 0.0, nullptr);
    report("device, own Pi", H0, D);
  }
  {
    auto D = wdbg_run<DEVICE_MEMORY>(F, beta, mu0, &H0.Pi, 0.0, nullptr);
    report("device W + Sigma from the host Pi", H0, D);
  }
  {
    setenv("COQUI_GWLINE_DYSON_BATCHED", "0", 1);
    auto D = wdbg_run<DEVICE_MEMORY>(F, beta, mu0, &H0.Pi, 0.0, nullptr);
    unsetenv("COQUI_GWLINE_DYSON_BATCHED");
    report("device (per-matrix Dyson) from the host Pi", H0, D);
  }
  {
    setenv("COQUI_GWLINE_W_MIRROR", "0", 1);
    auto D = wdbg_run<DEVICE_MEMORY>(F, beta, mu0, &H0.Pi, 0.0, nullptr);
    unsetenv("COQUI_GWLINE_W_MIRROR");
    report("device full-node path from the host Pi", H0, D);
  }
  {
    auto D = wdbg_run<DEVICE_MEMORY>(F, beta, mu0, &H0.Pi, 0.0, &H0.w);
    report("device Sigma from the host w", H0, D);
  }
#endif
}

// ============================================================================================================== device A/B
#if defined(ENABLE_DEVICE)
/**
 * [finiteT][device] lih222 beta 50: the thermal pass (thermal lists, tau leg, D, split fit, Bose rows) on the device vs the
 * host, stage by stage with identical inputs:
 *   Pi (rays + tau leg)                      <= 1e-12
 *   W at the nodes (device Dyson, own Pi)    <= 1e-12
 *   Sigma_c from the HOST residues           <= 1e-11   (the device Sigma path alone)   } at the masked nodes
 *   Sigma_c of the device chain (own w)      <= 1e-11   (full chain)                    } rho beta |zeta| >= c_f only
 * S8b.4 (notes sec:fT_wcond): Sigma_c below the floor (|zeta| <~ 0.02 Ha here) is not an output of the thermal W step (wrong by
 * design, 8e-2 vs Eq. fT_sigma) and moves by 1e10 x the Pi noise (the pre-S8b.4 1e-4 gate of the full chain measured those
 * nodes: 1.5e-5); it is reported, not gated. The residues w are a gauge of the split fit (near-null directions of the kernel
 * on D), compared only through W at the nodes and the masked Sigma_c (their own difference is reported).
 */
TEST_CASE("gw_line_finiteT_device", "[gw_line][finiteT][device]") {
  fx_t F("lih222");
  const double beta = 50.0;
  double mu0 = 0.0;
  {
    h5::file hf(ft_dir() + "lih222_finiteT_ref.h5", 'r');
    h5::group root(hf);
    auto gb = root.open_group("beta_50");
    mu0     = read_attr<double>(gb, "mu0");
  }
  auto &comm = F.mpi->comm;
  auto rel   = [&](auto const &a, auto const &b) {
    double d = 0.0, m = 0.0;
    for (long i = 0; i < a.size(); ++i) {
      d = std::max(d, std::abs(a.data()[i] - b.data()[i]));
      m = std::max(m, std::abs(a.data()[i]));
    }
    d = comm.all_reduce_value(d, mpi3::max<>{});
    m = comm.all_reduce_value(m, mpi3::max<>{});
    return d / m;
  };
  auto H  = wdbg_run<HOST_MEMORY>(F, beta, mu0, nullptr, 0.0, nullptr);
  auto D  = wdbg_run<DEVICE_MEMORY>(F, beta, mu0, nullptr, 0.0, nullptr);
  auto Dh = wdbg_run<DEVICE_MEMORY>(F, beta, mu0, &H.Pi, 0.0, &H.w);
  app_log(1, "\n[finiteT][device] lih222 beta {}: host vs device (thermal lists, tau leg, D, split fit, Bose rows)", beta);
  bool ok = gate("Pi on D (rays + tau leg)", rel(H.Pi, D.Pi), 1e-12);
  ok      = gate("W at the nodes (device Dyson, own Pi)", rel(H.Wn, D.Wn), 1e-12) and ok;
  ok      = gate("Sigma_c masked nodes, from the host residues", rel_nodes(comm, H.S, Dh.S, H.fmask, true), 1e-11) and ok;
  ok      = gate("Sigma_c masked nodes, device chain (own Pi, W, w)", rel_nodes(comm, H.S, D.S, H.fmask, true), 1e-11) and ok;
  app_log(1, "    Sigma_c sub-floor nodes (report: not an output)      {:.3e} (host residues) / {:.3e} (device chain)",
          rel_nodes(comm, H.S, Dh.S, H.fmask, false), rel_nodes(comm, H.S, D.S, H.fmask, false));
  app_log(1, "    residues w (report only: a gauge)                    {:.3e}", rel(H.w, D.w));
  REQUIRE(ok);
}
#endif

// ============================================================================================================== S8b.3 hybrid
#include "methods/GW_line/hybrid.hpp"
namespace {
/// tau-leg Sigma_c(i w_n) with the line W (generator's D and nu_b, all poles Bose-augmented) vs Eq. fT_sigma, and the
/// Matsubara density vs the hybrid reference (lih222_finiteT_hybrid_ref.h5)
bool run_hybrid_unit(fx_t &F, double beta, [[maybe_unused]] bool device_ab = false) {
  auto &comm = F.mpi->comm;
  auto &mf   = *F.mf;
  auto &thc  = *F.thc;
  const long nk = mf.nkpts(), nq = mf.nqpts(), nb = thc.nbnd(), Np = thc.Np();
  const std::string gname = "beta_" + std::to_string(long(beta));
  h5::file hf(ft_dir() + "lih222_finiteT_ref.h5", 'r');
  h5::group root(hf);
  auto gb = root.open_group(gname);
  h5::file hh(ft_dir() + "lih222_finiteT_hybrid_ref.h5", 'r');
  h5::group hroot(hh);
  auto hb = hroot.open_group(gname);
  const double mu0 = read_attr<double>(gb, "mu0"), wmax = read_attr<double>(hroot, "wmax");
  const double deg = std::numbers::pi / 180.0;
  thermal_params_t tp;
  tp.beta = beta; tp.thermal_tol = 1e-12; tp.theta = 20.0 * deg; tp.theta_t = 10.0 * deg; tp.tau_grid = "gl";
  utils::TimerManager Timer;
  bool ok = true;
  nda::array<double, 2> eig(nk, nb);
  double emax = 0.0;
  for (long k = 0; k < nk; ++k)
    for (long n = 0; n < nb; ++n) {
      eig(k, n) = mf.eigval()(0, k, n);
      emax      = std::max(emax, std::abs(eig(k, n) - mu0));
    }
  auto ks = pole_data_t::from_ks(eig, mu0);
  aux_grid_t grid(*F.mpi, Np);
  propagator_t<HOST_MEMORY> prop(thc, grid);
  ibz_t ibz(mf, nb, false);
  std::vector<long> qall(nq);
  std::iota(qall.begin(), qall.end(), 0L);
  // the W of T2: generator's D, nu_b; Pi on D (GL rays + GL tau leg); split fit with w(q) and w(-q)^T rows of every pole
  nda::array<ComplexType, 1> Dr = read_c<1>(gb, "D_zeta");
  nda::array<double, 1> nub;
  nda::h5_read(gb, "nu_b", nub);
  const long r = nub.size();
  auto tl = thermal_lists(ks, beta, tp.E_T());
  auto rp = time_ray_t::guarded(tp.theta_t, beta, tp.E_T(), 1e-5, 3.0, 16, sector_t::particle);
  auto rh = time_ray_t::guarded(tp.theta_t, beta, tp.E_T(), 1e-5, 3.0, 16, sector_t::hole);
  auto tn = make_tau_nodes(tp, emax, 2.0 * emax);
  memory::array<HOST_MEMORY, ComplexType, 4> Pt, Pi, w, Wn;
  nda::array<ComplexType, 1> z0(1);
  z0(0) = 0.0;
  pi_tau_leg<HOST_MEMORY>(prop, ks, mf, ibz, grid, tp, tn, z0, 8, Pt, Timer, qall);
  polarization<HOST_MEMORY>(prop, tl, mf, grid, Dr, numerics::line_dlr::time_nodes_t(rp), numerics::line_dlr::time_nodes_t(rh), 8, Pi,
                            Timer, sector_t::both, qall);
  for (long i = 0; i < Dr.size(); ++i)
    if (Dr(i) == ComplexType(0.0))
      for (long q = 0; q < nq; ++q) Pi(q, i, nda::range::all, nda::range::all) = Pt(q, 0, nda::range::all, nda::range::all);
  auto B  = bosonic_basis_t::with_poles(tp.theta, Dr, 4.0, nub);
  auto B2 = B;
  B2.n_bose = r;
  B2.rank   = 2 * r;
  B2.bose_src.resize(r);
  std::iota(B2.bose_src.begin(), B2.bose_src.end(), 0L);
  B2.nu = nda::array<double, 1>(2 * r);
  B2.nu(nda::range(r)) = nub;
  B2.nu(nda::range(r, 2 * r)) = nub;
  coulomb_blocks_t<HOST_MEMORY> Zb(thc, grid, qall, Timer);
  screened_interaction<HOST_MEMORY>(Pi, Zb, B2, grid, *F.mpi, w, Timer, &Wn, qall, false);
  B2.beta = beta;
  B2.tw   = nda::array<double, 1>(2 * r);
  B2.tw() = 1.0;
  auto Bx = B2.with_exact_bose();
  // the tau leg
  auto HN = make_hybrid_nodes(beta, emax + nda::max_element(nub), 1e-13, comm);
  nda::array<ComplexType, 4> Sp, Sh;
  auto tq = tau_lists(ks, beta);
  auto t0 = std::chrono::steady_clock::now();
  self_energy<HOST_MEMORY>(prop, tq, w, Bx, mf, grid, *F.mpi, HN.zdummy, *HN.kp, *HN.kh, 8, Sp, Timer, sector_t::both, false, nullptr, 0,
                           &Sh);
  const double tsec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
#if defined(ENABLE_DEVICE)
  if (device_ab) {   // [hybrid][device]: the tau-leg Sigma kernels on the device from the same residues (exact up to BLAS order)
    propagator_t<DEVICE_MEMORY> prop_d(thc, grid);
    memory::array<DEVICE_MEMORY, ComplexType, 4> w_d = memory::to_memory_space<DEVICE_MEMORY>(w);
    nda::array<ComplexType, 4> Sp_d, Sh_d;
    self_energy<DEVICE_MEMORY>(prop_d, tq, w_d, Bx, mf, grid, *F.mpi, HN.zdummy, *HN.kp, *HN.kh, 8, Sp_d, Timer, sector_t::both, false,
                               nullptr, 0, &Sh_d);
    double dd = 0.0, mm = 0.0;
    for (long a = 0; a < Sp.size(); ++a) {
      dd = std::max({dd, std::abs(Sp.data()[a] - Sp_d.data()[a]), std::abs(Sh.data()[a] - Sh_d.data()[a])});
      mm = std::max({mm, std::abs(Sp.data()[a]), std::abs(Sh.data()[a])});
    }
    dd = comm.all_reduce_value(dd, mpi3::max<>{});
    mm = comm.all_reduce_value(mm, mpi3::max<>{});
    app_log(1, "\n[finiteT][hybrid][device] lih222 beta {}: tau-leg Sigma_c ({} nodes per leg, Bose rows {}) host vs device, same w",
            beta, HN.ntau, r);
    return gate("tau-leg nodal Sigma_c^>, Sigma_c^< host vs device", dd / mm, 1e-11);
  }
#endif
  nda::array<long, 1> np_;
  nda::h5_read(hb, "n_probe", np_);
  nda::array<ComplexType, 1> izp(np_.size());
  for (long i = 0; i < np_.size(); ++i) izp(i) = ComplexType(0.0, std::numbers::pi * double(2 * np_(i) + 1) / beta);
  auto [Fp, Fh] = HN.fourier(izp);
  auto SE = read_c<4>(hb, "Sigma_probe_exact");
  double d = 0.0, m = 0.0;
  const long ksel[2] = {0, 3};
  for (int a = 0; a < 2; ++a)
    for (long i = 0; i < izp.size(); ++i) {
      nda::matrix<ComplexType> S(nb, nb);
      S() = 0.0;
      for (long j = 0; j < HN.ntau; ++j)
        S += Fp(i, j) * Sp(ksel[a], j, nda::range::all, nda::range::all) + Fh(i, j) * Sh(ksel[a], j, nda::range::all, nda::range::all);
      for (long x = 0; x < nb; ++x)
        for (long y = 0; y < nb; ++y) {
          d = std::max(d, std::abs(S(x, y) - SE(a, i, x, y)));
          m = std::max(m, std::abs(SE(a, i, x, y)));
        }
    }
  app_log(1, "\n[finiteT][hybrid] lih222 beta {}: Sigma tau ID {} nodes per leg (rank {}, E in [{:.3f}, {:.3f}]), Bose rows {}, tau leg "
             "{:.1f} s; probes n up to {}",
          beta, HN.ntau, HN.idp.rank, -HN.Eneg, HN.Emax, r, tsec, np_(np_.size() - 1));
  ok = gate("tau-leg Sigma_c(i w_n) (line W) vs Eq. fT_sigma", d / m, 1e-10) and ok;
  {   // moments vs exact S1, S2
    auto S1e = read_c<3>(hb, "S1_exact");
    auto S2e = read_c<3>(hb, "S2_exact");
    double d1 = 0.0, m1 = 0.0, d2 = 0.0, m2 = 0.0;
    for (long k = 0; k < nk; ++k) {
      nda::array<ComplexType, 3> sp(Sp(k, nda::ellipsis{})), sh(Sh(k, nda::ellipsis{}));
      auto [S1, S2] = hybrid_moments(HN, sp, sh);
      for (long x = 0; x < nb; ++x)
        for (long y = 0; y < nb; ++y) {
          d1 = std::max(d1, std::abs(S1(x, y) - S1e(k, x, y)));
          m1 = std::max(m1, std::abs(S1e(k, x, y)));
          d2 = std::max(d2, std::abs(S2(x, y) - S2e(k, x, y)));
          m2 = std::max(m2, std::abs(S2e(k, x, y)));
        }
    }
    ok = gate("S1 (end points) vs exact", d1 / m1, 1e-10) and ok;
    ok = gate("S2 (third-order differences) vs exact", d2 / m2, 1e-5) and ok;
  }
  // the Matsubara density with H = diag(eig - mu0) (the reference's), k rows of this rank
  nda::array<ComplexType, 3> H(nk, nb, nb);
  H() = 0.0;
  for (long k = 0; k < nk; ++k)
    for (long n = 0; n < nb; ++n) H(k, n, n) = eig(k, n) - mu0;
  std::vector<long> krows;
  for (long k = comm.rank(); k < nk; k += comm.size()) krows.push_back(k);
  nda::array<ComplexType, 4> Spl(long(krows.size()), HN.size(), nb, nb), Shl(long(krows.size()), HN.size(), nb, nb);
  for (long l = 0; l < long(krows.size()); ++l) {
    Spl(l, nda::ellipsis{}) = Sp(krows[l], nda::ellipsis{});
    Shl(l, nda::ellipsis{}) = Sh(krows[l], nda::ellipsis{});
  }
  t0 = std::chrono::steady_clock::now();
  auto hd = matsubara_density(comm, H, Spl, Shl, krows, {}, HN, beta, wmax, 4.0);
  const double tden = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  auto Dl = read_c<3>(hb, "D_lineW");
  auto De = read_c<3>(hb, "D_exact");
  double dl = 0.0, de = 0.0;
  for (long x = 0; x < Dl.size(); ++x) {
    dl = std::max(dl, std::abs(hd.D.data()[x] - Dl.data()[x]));
    de = std::max(de, std::abs(hd.D.data()[x] - De.data()[x]));
  }
  const double dmu_l = read_attr<double>(hb, "dmu_lineW"), dmu_e = read_attr<double>(hb, "dmu_exact");
  app_log(1, "  Matsubara density: N = {} frequencies, {} bisection steps, {:.1f} s; dmu {:.15f} (reference line W {:.15f}, exact {:.15f})",
          hd.nfreq, hd.nbisect, tden, hd.dmu, dmu_l, dmu_e);
  ok = gate("D vs the reference's line-W hybrid D", dl, 1e-9) and ok;
  ok = gate("D vs the exact-G Matsubara sum", de, 1e-9) and ok;
  ok = gate("|dmu - dmu_exact| (Ha)", std::abs(hd.dmu - dmu_e), 1e-9) and ok;
  ok = gate("|N(mu) - N_el| (inversion)", std::abs(hd.N - 4.0), 1e-9) and ok;
  ok = gate("|N(mu) - N_el| (trace)", std::abs(hd.N_trace - 4.0), 1e-12) and ok;
  return ok;
}
} // namespace

TEST_CASE("gw_line_finiteT_hybrid_unit", "[gw_line][finiteT][hybrid]") {
  fx_t F("lih222");
  bool ok = run_hybrid_unit(F, 200.0);
  ok      = run_hybrid_unit(F, 50.0) and ok;
  REQUIRE(ok);
}

#if defined(ENABLE_DEVICE)
TEST_CASE("gw_line_finiteT_hybrid_device", "[gw_line][finiteT][hybrid][device]") {
  fx_t F("lih222");
  REQUIRE(run_hybrid_unit(F, 50.0, true));
}
#endif

// S8b.3 hybrid SCF: lih222 beta 200, theta_t = theta / 4 (the prototype's dev/s8b3_scf.py settings), scf_density = "matsubara":
// N(mu) = N_el every iteration, mu vs the prototype (iteration 1: kernels + the exact density only), restart bitwise,
// 1 vs 2 ranks (iteration 1 sharp; later iterations carry the closure's noise through the next poles: gated at 1 mHa)
TEST_CASE("gw_line_finiteT_hybrid_scf", "[gw_line][finiteT][hybrid][scf]") {
  lih_ft_t L;
  auto &comm = L.mpi->comm;
  const long NIT = 3, NR = 2;
  const double mu_py[5] = {0.20608474063937182, 0.2141961519644476, 0.2150988899866979, 0.21540045798415544, 0.21554059947920648};
  auto params = [&](std::string const &out, long niter, bool restart) {
    auto pt = ft_params(out, niter, 200.0, "id", "auto", restart);
    pt.put("theta_t_frac", 0.25);
    pt.put("scf_density", "matsubara");
    return pt;
  };
  const std::string fa = "gw_line_ft_hyb_a", fb = "gw_line_ft_hyb_b";
  auto t0 = std::chrono::steady_clock::now();
  auto A  = gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, params(fa, NIT, false));
  const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  bool ok = long(A.history.size()) == NIT;
  app_log(1, "\n[finiteT][hybrid][scf] {} iterations in {:.1f} s ({} ranks), beta 200, theta_t = theta / 4, scf_density matsubara", NIT, dt,
          comm.size());
  for (auto const &h : A.history) {
    app_log(1, "  iter {}: mu {:.12f} (prototype {:.12f}, diff {:+.2e} Ha) N(mu) {:.15f} | closure N {:.8f} max|D_cl - D| {:.1e} own mu "
               "{:+.2e} ({}) | gap {:.4f} eV, {} frequencies, {:.1f} s",
            h.iter, h.mu, mu_py[h.iter - 1], h.mu - mu_py[h.iter - 1], h.N_mu, h.N_closure, h.D_cl_err, h.dmu_closure, h.rule_closure,
            h.gap * 27.211386, h.nfreq, h.time);
    ok = gate("|N(mu) - N_el|", std::abs(h.N_mu - 4.0), 1e-12) and ok;
    if (h.iter == 1) ok = gate("iteration-1 |mu - mu_prototype| (Ha)", std::abs(h.mu - mu_py[0]), 1e-6) and ok;
  }
  {   // restart
    gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, params(fb, NR, false));
    auto B = gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, params(fb, NIT, true));
    app_log(1, "  restart {} + {} vs {} straight:", NR, NIT - NR, NIT);
    ok = same_runs(comm, A, B, fa, fb, NIT) and ok;
  }
  {   // 1 vs 2 ranks
    const std::string mine = "gw_line_ft_hyb_np" + std::to_string(comm.size()) + ".h5";
    const std::string other = "gw_line_ft_hyb_np" + std::to_string(comm.size() == 1 ? 2 : 1) + ".h5";
    nda::array<double, 1> mus(long(A.history.size()));
    for (long i = 0; i < mus.size(); ++i) mus(i) = A.history[i].mu;
    if (comm.root()) {
      h5::file f(mine, 'w');
      h5::group g(f);
      nda::h5_write(g, "mu", mus, false);
    }
    comm.barrier();
    if (std::filesystem::exists(other)) {
      nda::array<double, 1> mo;
      {
        h5::file f(other, 'r');
        h5::group g(f);
        nda::h5_read(g, "mu", mo);
      }
      for (long i = 0; i < std::min(mo.size(), mus.size()); ++i)
        // S8b.4 (c): with the covariant free block the hybrid SCF is rank invariant (Mac 2026-10-18: 2.4e-15 / 1.6e-14 / 1.2e-11 Ha);
        // gate = the plan's rank-invariance gate 1e-8 Ha (was 1e-3 beyond iteration 1)
        ok = gate(("1 vs 2 ranks: |dmu| (Ha) iteration " + std::to_string(i + 1)).c_str(), std::abs(mo(i) - mus(i)),
                  i == 0 ? 1e-10 : 1e-8) and ok;
    } else
      app_log(1, "  {} written; run with the other rank count to compare", mine);
  }
  rm_ckpt(comm, fa);
  rm_ckpt(comm, fb);
  REQUIRE(ok);
}

// D4 (metals, charge sloshing): linear F mixing (key mixing_F). The F of the iteration-2 closure must be
// mixing_F F[D_1] + (1 - mixing_F) F[D_KS] exactly (F replicated, rank independent); iteration 1 is unmixed (= prototype mu).
// lih222 beta 200 hybrid theta / 4 as [hybrid][scf]; 1 iteration, then a restart to 2 (the restart path carries F and F_closure).
TEST_CASE("gw_line_finiteT_mixF", "[gw_line][finiteT][hybrid][mixF]") {
  lih_ft_t L;
  auto &comm     = L.mpi->comm;
  const double a = 0.4, mu_py0 = 0.20608474063937182;
  auto params    = [&](std::string const &out, long niter, bool restart) {
    auto pt = ft_params(out, niter, 200.0, "id", "auto", restart);
    pt.put("theta_t_frac", 0.25);
    pt.put("scf_density", "matsubara");
    pt.put("mixing_F", a);
    return pt;
  };
  const std::string fa = "gw_line_ft_mixF";
  auto A  = gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, params(fa, 1, false));   // A.F = F[D_1], A.F_closure = F[D_KS]
  auto B  = gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, params(fa, 2, true));    // B.F_closure = the mixed F of iteration 2
  bool ok = long(B.history.size()) == 2 and B.history.back().mix == "linear";
  double d = 0.0, m = 0.0, dF = 0.0;
  for (long x = 0; x < A.F.size(); ++x) {
    const ComplexType e = a * A.F.data()[x] + (1.0 - a) * A.F_closure.data()[x];
    d  = std::max(d, std::abs(B.F_closure.data()[x] - e));
    m  = std::max(m, std::abs(e));
    dF = std::max(dF, std::abs(A.F.data()[x] - A.F_closure.data()[x]));
  }
  app_log(1, "\n[finiteT][mixF] mixing_F {}: max|F[D_1] - F[D_KS]| {:.3e}, iteration-2 F_closure vs mixed {:.2e} (rel {:.2e}); mu {:.12f} / "
             "{:.12f}, residF {:.3e}",
          a, dF, d, d / m, B.history[0].mu, B.history[1].mu, B.history[1].residF);
  ok = gate("F[D_1] differs from F[D_KS] (the check is not empty)", dF > 1e-6 ? 0.0 : 1.0, 0.5) and ok;
  ok = gate("iteration-2 F_closure = mixing_F F[D_1] + (1 - mixing_F) F[D_KS] (rel)", d / m, 1e-14) and ok;
  ok = gate("residF of iteration 2 = max|F[D_1] - F[D_KS]|", std::abs(B.history[1].residF - dF), 1e-14) and ok;
  ok = gate("iteration-1 |mu - mu_prototype| (Ha)", std::abs(B.history[0].mu - mu_py0), 1e-6) and ok;
  for (auto const &h : B.history) ok = gate("|N(mu) - N_el|", std::abs(h.N_mu - 4.0), 1e-12) and ok;
  rm_ckpt(comm, fa);
  REQUIRE(ok);
}

// D4 closure follow-up: the frame of the closure in hybrid runs (key hyb_frame). "new" computes the hybrid mu first and closes the
// line Sigma at H - mu_new (the G of the hybrid density), "old" (default, bitwise) closes at the previous mu and re-centres the
// poles. lih222 beta 200 hybrid theta / 4 ([hybrid][scf] settings), 2 iterations per frame; reference gen_frame_ref.py (prototype
// LineSCGW(hyb_frame)): the iteration-1 hybrid mu does not depend on the frame (bitwise here); mu of iteration 2 and the closure
// count N_closure of iterations 1 and 2 against the prototype of the same frame (closure parity C++ vs python as [hybrid][scf]).
TEST_CASE("gw_line_finiteT_frame", "[gw_line][finiteT][hybrid][frame]") {
  lih_ft_t L;
  auto &comm = L.mpi->comm;
  std::map<std::string, std::array<nda::array<double, 1>, 2>> ref;   // frame -> (mu, N_closure)
  {
    h5::file f(ft_dir() + "lih222_frame_ref.h5", 'r');
    h5::group g(f);
    for (std::string fr : {"old", "new", "count"}) {
      auto gg = g.open_group(fr);
      nda::h5_read(gg, "mu", ref[fr][0]);
      nda::h5_read(gg, "N_closure", ref[fr][1]);
    }
  }
  auto run = [&](std::string const &fr) {
    auto pt = ft_params("gw_line_ft_frame_" + fr, 2, 200.0, "id", "auto", false);
    pt.put("theta_t_frac", 0.25);
    pt.put("scf_density", "matsubara");
    pt.put("hyb_frame", fr);
    auto R = gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pt);
    rm_ckpt(comm, "gw_line_ft_frame_" + fr);
    return R;
  };
  auto O  = run("old");
  auto N  = run("new");
  auto C  = run("count");
  bool ok = O.history.size() == 2 and N.history.size() == 2 and C.history.size() == 2;
  app_log(1, "\n[finiteT][hybrid][frame] lih222 beta 200, 2 hybrid iterations per frame:");
  for (auto const &[fr, R] : std::vector<std::pair<std::string, decltype(O) const *>>{{"old", &O}, {"new", &N}, {"count", &C}})
    for (auto const &h : R->history)
      app_log(1, "  {} iter {}: mu {:.12f} (prototype {:.12f}, {:+.2e}) | closure N {:.8f} (prototype {:.8f}, {:+.2e}), max|D_cl - D| "
                 "{:.2e}, own mu {:+.2e} Ha",
              fr, h.iter, h.mu, ref[fr][0](h.iter - 1), h.mu - ref[fr][0](h.iter - 1), h.N_closure, ref[fr][1](h.iter - 1),
              h.N_closure - ref[fr][1](h.iter - 1), h.D_cl_err, h.dmu_closure);
  ok = gate("iteration-1 hybrid mu independent of the frame (Ha)", std::abs(O.history[0].mu - N.history[0].mu), 1e-14) and ok;
  ok = gate("iteration-1 |mu - mu_prototype| (Ha)", std::abs(N.history[0].mu - ref["new"][0](0)), 1e-6) and ok;
  // iteration 2: the C++ / python closure parity of the hybrid (time grids, pruned Lehmann G; Mac 2026-10-19: mu 4.7e-7 Ha in
  // both frames, N_closure 1.9e-6); iteration 1: the same closure of the same Sigma (9e-10)
  ok = gate("frame new: iteration-2 |mu - mu_prototype| (Ha)", std::abs(N.history[1].mu - ref["new"][0](1)), 5e-6) and ok;
  ok = gate("frame new: |N_closure - prototype| iteration 1", std::abs(N.history[0].N_closure - ref["new"][1](0)), 1e-8) and ok;
  ok = gate("frame new: |N_closure - prototype| iteration 2", std::abs(N.history[1].N_closure - ref["new"][1](1)), 2e-5) and ok;
  // the point of the frame: the closure G of iteration 1 carries N_el (old: 3.963, the lag of the hybrid mu step of 3.45 eV)
  ok = gate("frame new: |N_closure - N_el| iteration 1", std::abs(N.history[0].N_closure - 4.0), 1e-5) and ok;
  // "count": the closure's poles keep its own mu (rule "number" here): N_closure = N_el; iteration 1 = the poles of "new" shifted
  ok = gate("frame count: iteration-2 |mu - mu_prototype| (Ha)", std::abs(C.history[1].mu - ref["count"][0](1)), 5e-6) and ok;
  // (iteration 1 the closure's rule is "number": N_el exactly; iteration 2 lih222 is gapped at the new mu and the rule keeps the gap)
  ok = gate("frame count: |N_closure - N_el| iteration 1", std::abs(C.history[0].N_closure - 4.0), 1e-8) and ok;
  ok = gate("frame count: |N_closure - prototype| iteration 2", std::abs(C.history[1].N_closure - ref["count"][1](1)), 2e-5) and ok;
  for (auto const *R : {&N, &C})
    for (auto const &h : R->history) ok = gate("|N(mu) - N_el|", std::abs(h.N_mu - 4.0), 1e-12) and ok;
  REQUIRE(ok);
}

// D4: hybrid energies (E_H0, E_HF, E_corr = sum_k w_k (1/beta) sum_n Tr[Sigma_c G]) and Sigma_c(i w_n < hyb_sigma_out) in the checkpoint.
// The S1 / (i w)^2 tail of E_corr: one hybrid iteration (lih222 beta 200, G0W0@KS + Matsubara Dyson) with hyb_wmax 100 (N 3184, the
// default) and 200 Ha (N 6367): E_corr equal to 1e-7 (S1 tail + w^-4 elimination; with the S1 tail only: 1.1e-6, at 25 Ha 8e-5); the
// D-energies E_H0 / E_HF agree likewise (D's own tail); Sigma_c(i w_n) written (n < 8) identical in both (the same tau-leg nodes).
TEST_CASE("gw_line_finiteT_hybrid_energy", "[gw_line][finiteT][hybrid][energy]") {
  lih_ft_t L;
  auto &comm = L.mpi->comm;
  auto run   = [&](std::string const &out, double wmax) {
    auto pt = ft_params(out, 1, 200.0, "id", "auto", false);
    pt.put("theta_t_frac", 0.25);
    pt.put("scf_density", "matsubara");
    pt.put("hyb_wmax", wmax);
    pt.put("hyb_sigma_out", 8);
    gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pt);
    std::array<double, 4> E{};
    nda::array<ComplexType, 4> S;
    {
      h5::file f(out + ".gw_line.h5", 'r');
      h5::group g(f);
      auto it = g.open_group("scf_line").open_group("iter1");
      auto eg = it.open_group("energy");
      h5::h5_read(eg, "E_H0", E[0]);
      h5::h5_read(eg, "E_HF", E[1]);
      h5::h5_read(eg, "E_corr", E[2]);
      h5::h5_read(it, "mu", E[3]);
      nda::h5_read(it, "Sigma_c_iw", S);
    }
    comm.barrier();
    rm_ckpt(comm, out);
    return std::make_pair(E, S);
  };
  auto [Ea, Sa] = run("gw_line_ft_en_a", 100.0);
  auto [Eb, Sb] = run("gw_line_ft_en_b", 200.0);
  app_log(1, "\n[finiteT][hybrid][energy] lih222 beta 200 it 1: E_H0 {:.10f} / {:.10f}, E_HF {:.10f} / {:.10f}, E_corr {:.10f} / {:.10f} "
             "(w_max 100 / 200 Ha), mu {:.12f} / {:.12f}",
          Ea[0], Eb[0], Ea[1], Eb[1], Ea[2], Eb[2], Ea[3], Eb[3]);
  bool ok = Sa.shape() == Sb.shape() and Sa.extent(1) == 8;
  double dS = 0.0;
  for (long x = 0; x < Sa.size(); ++x) dS = std::max(dS, std::abs(Sa.data()[x] - Sb.data()[x]));
  ok = gate("E_corr(w_max 100) - E_corr(w_max 200) (Ha)", std::abs(Ea[2] - Eb[2]), 1e-7) and ok;
  ok = gate("E_H0 + E_HF (w_max 100 vs 200) (Ha)", std::abs(Ea[0] + Ea[1] - Eb[0] - Eb[1]), 1e-7) and ok;
  ok = gate("Sigma_c(i w_n < 8) written, w_max 100 vs 200", dS, 1e-13) and ok;
  ok = gate("E_corr < 0 and O(0.01-1) Ha", (Ea[2] < 0.0 and Ea[2] > -1.0) ? 0.0 : 1.0, 0.5) and ok;
  REQUIRE(ok);
}

// D4 nu0_intraband (nu0.hpp; notes section 12 "The nu_0 term"): one hybrid iteration (G0W0@KS, lih222 beta 50, theta / 4, mu rule
// "number") with the option on and off. Same G and W, so on - off = the Matsubara nu_0 term of the exactly degenerate pairs exactly:
// at i w_n (Sigma_c_iw, the tau-leg nodes + their transform) and at the line nodes (Sigma_p + Sigma_h), against the prototype oracle
// lih222_nu0_ref.h5 (gen_nu0_ref.py: finite_t transition sums, dW = W[Pi^an(0) + dPi] - W[Pi^an(0)], pole residues R_qm(k)).
TEST_CASE("gw_line_finiteT_nu0", "[gw_line][finiteT][hybrid][nu0]") {
  lih_ft_t L;
  auto &comm = L.mpi->comm;
  double beta = 0.0, mu_ref = 0.0;
  nda::array<ComplexType, 4> term_ref, pR;
  nda::array<double, 2> pe;
  {
    h5::file f(ft_dir() + "lih222_nu0_ref.h5", 'r');
    h5::group g(f);
    beta     = read_attr<double>(g, "beta");
    mu_ref   = read_attr<double>(g, "mu0");
    term_ref = read_c<4>(g, "term_iw");
    pR       = read_c<4>(g, "pole_R");
    nda::h5_read(g, "pole_e", pe);
  }
  auto run = [&](std::string const &out, bool on) {
    auto pt = ft_params(out, 1, beta, "id", "number", false);
    pt.put("theta_t_frac", 0.25);
    pt.put("scf_density", "matsubara");
    pt.put("hyb_sigma_out", 8);
    if (on) pt.put("nu0_intraband", true);
    auto R = gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pt);
    nda::array<ComplexType, 4> S;
    nda::array<ComplexType, 1> z;
    if (comm.root()) {
      h5::file f(out + ".gw_line.h5", 'r');
      h5::group g(f);
      auto it = g.open_group("scf_line").open_group("iter1");
      nda::h5_read(it, "Sigma_c_iw", S);
      auto ig = g.open_group("input");
      nda::h5_read(ig, "fermionic_nodes", z);
    }
    auto Sn = sigma_total_of(comm, out + ".gw_line.h5", 1);
    rm_ckpt(comm, out);
    return std::make_tuple(R, S, Sn, z);
  };
  auto [Ra, Sa, Na, za] = run("gw_line_ft_nu0_off", false);
  auto [Rb, Sb, Nb, zb] = run("gw_line_ft_nu0_on", true);
  bool ok = true;
  if (comm.root()) {
    const long nk = term_ref.extent(0), nw = term_ref.extent(1), nb = term_ref.extent(2), M = pe.extent(1);
    double d = 0.0, m = 0.0, dn = 0.0, mn = 0.0, ds = 0.0;
    for (long k = 0; k < nk; ++k)
      for (long n = 0; n < nw; ++n)
        for (long a = 0; a < nb; ++a)
          for (long b = 0; b < nb; ++b) {
            d = std::max(d, std::abs(Sb(k, n, a, b) - Sa(k, n, a, b) - term_ref(k, n, a, b)));
            m = std::max(m, std::abs(term_ref(k, n, a, b)));
            ds = std::max(ds, std::abs(Sa(k, n, a, b)));
          }
    for (long k = 0; k < nk; ++k)
      for (long iz = 0; iz < za.size(); ++iz)
        for (long a = 0; a < nb; ++a)
          for (long b = 0; b < nb; ++b) {
            ComplexType t = 0.0;
            for (long p = 0; p < M; ++p) t += pR(k, p, a, b) / (za(iz) - pe(k, p));
            dn = std::max(dn, std::abs(Nb(k, iz, a, b) - Na(k, iz, a, b) - t));
            mn = std::max(mn, std::abs(t));
          }
    app_log(1, "\n[finiteT][hybrid][nu0] lih222 beta {}: iteration-1 mu (off / on) {:.12f} / {:.12f}; max|Sigma_c(i w_n)| {:.3e}, max|nu_0 "
               "term| {:.3e} (i w_n), {:.3e} (line nodes)",
            beta, Ra.history[0].mu, Rb.history[0].mu, ds, m, mn);
    ok = gate("the term is not empty (max|term| / max|Sigma_c| > 1e-3)", m / ds > 1e-3 ? 0.0 : 1.0, 0.5) and ok;
    ok = gate("Sigma_c(i w_n): on - off vs the oracle term (rel)", d / m, 1e-8) and ok;
    ok = gate("line nodes: on - off vs the oracle term (rel)", dn / mn, 1e-8) and ok;
    ok = gate("iteration-1 mu (on) moves (|dmu| > 1e-9 Ha)", std::abs(Rb.history[0].mu - Ra.history[0].mu) > 1e-9 ? 0.0 : 1.0, 0.5) and ok;
    (void)mu_ref;
  }
  int okc = ok ? 1 : 0;
  comm.broadcast_n(&okc, 1, 0);
  REQUIRE(okc == 1);
}

// S8b thermal closure perf: the "lazy" terminal-phase scan (default in thermal iterations) vs the S8b.2 distributed eigen scan
// (COQUI_GWLINE_THERMAL_SCAN = parallel): one thermal iteration (lih222 beta 200, the [scf] settings; n_free 16 at every k)
// -> the same closure decisions (terminal phases), mu and poles within the roundoff of the two held-out error evaluations.
// S8b.3 lazy-scan fix (2026-10-17): for n_free > 1 the family U(phi) = A1 + e^{i phi} P0 Q0^dag is fixed by the roundoff of the
// reference SVD (the pairing of the null singular vectors), so both scans must do that SVD in the same place with the same
// BLAS threads. The second pass makes the k loop and the scan differ in BLAS threads (closure_threads 4 over 4 k workers:
// 1 thread per k in the k loop, 4 in the scan), the situation of Si 4x4x4 on rusty (k loop 9-11 borrowed cores, scan
// phase A 16) where the lazy scan with its SVD in the k loop chose other phases (iteration-1 mu 0.279015 vs 0.278035 Ha).
TEST_CASE("gw_line_finiteT_lazy_scan", "[gw_line][finiteT][lazy]") {
  lih_ft_t L;
  auto &comm = L.mpi->comm;
  auto run = [&](std::string const &mode, std::string const &out, long threads) {
    setenv("COQUI_GWLINE_THERMAL_SCAN", mode.c_str(), 1);
    auto t0 = std::chrono::steady_clock::now();
    auto pt = ft_params(out, 1, 200.0, "id", "number");
    if (threads > 0) {
      pt.put("closure_threads", threads);
      pt.put("closure_k_workers", threads);
    }
    auto R = gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pt);
    unsetenv("COQUI_GWLINE_THERMAL_SCAN");
    nda::array<double, 1> phi;
    if (comm.root()) {
      h5::file f(out + ".gw_line.h5", 'r');
      h5::group g(f);
      auto gi = g.open_group("scf_line/iter1");
      nda::h5_read(gi, "closure_phi", phi);
    }
    bcast_nda_(comm, phi);
    return std::make_tuple(R, phi, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  };
  bool ok = true;
  for (long threads : {0L, 4L}) {
    auto [A, pa, ta] = run("lazy", "gw_line_ft_lazy_a", threads);
    auto [B, pb, tb] = run("parallel", "gw_line_ft_lazy_b", threads);
    double dphi = 0.0, dP = 0.0, emax = 0.0;
    for (long k = 0; k < pa.size(); ++k) dphi = std::max(dphi, std::abs(pa(k) - pb(k)));
    for (long k = 0; k < A.poles.nk; ++k)
      for (int s = 0; s < 2; ++s) {
        auto const &x = s ? A.poles.part[k] : A.poles.hole[k];
        auto const &y = s ? B.poles.part[k] : B.poles.hole[k];
        if (x.size() != y.size()) { dP = 1e300; continue; }
        for (long m = 0; m < x.size(); ++m) {
          dP   = std::max(dP, std::abs(x.e(m) - y.e(m)));
          emax = std::max(emax, std::abs(x.e(m)));
        }
      }
    app_log(1, "\n[finiteT][lazy] one thermal iteration (closure threads {}): lazy {:.1f} s, parallel {:.1f} s; max|d phi| {:.2e}, "
               "|d mu| {:.2e} Ha, max|d e_m| {:.2e} (max|e| {:.1f}), phases {}",
            threads, ta, tb, dphi, std::abs(A.history[0].mu - B.history[0].mu), dP, emax, pa.size());
    const std::string tag = " (closure threads " + std::to_string(threads) + ")";
    ok = gate(("terminal phases: lazy vs parallel scan" + tag).c_str(), dphi, 1e-6) and ok;
    ok = gate(("mu: lazy vs parallel scan (Ha)" + tag).c_str(), std::abs(A.history[0].mu - B.history[0].mu), 1e-8) and ok;
    ok = gate(("Lehmann poles: lazy vs parallel scan (Ha)" + tag).c_str(), dP, 1e-6) and ok;
    rm_ckpt(comm, "gw_line_ft_lazy_a");
    rm_ckpt(comm, "gw_line_ft_lazy_b");
  }
  REQUIRE(ok);
}

// ============================================================================================================== T4c (hidden)
// VALIDATION (plan S8b T4c, not a gate): the 5-iteration scGW on lih222 at beta 200 vs CoQui's imaginary-axis scGW (DLR,
// damping 0.5, mu by the particle number), for scf_density = "closure" and "matsubara" (theta_t = theta/2 and theta/4).
// Compared through G(i w_n) at CoQui's Matsubara points: X(i w_n) = i w_n + mu - G^{-1}(i w_n) = F + Sigma_c(i w_n) (both
// codes, H0 included), so dX = the total self-energy difference; reported: mu, N(mu), dN, max|dX| relative to max|X|, the
// dynamic part (dX - dX(i w_max)) relative to max|Sigma_c^CoQui| and the static offset dX(i w_max).
namespace {
/// G(i w_n) (nw, nk, nb, nb) of a pole set (e relative to its mu) at z = i w_n
nda::array<ComplexType, 4> g_of_poles(pole_data_t const &pd, nda::array<ComplexType, 1> const &z) {
  const long nk = pd.nk, nb = pd.nb, nw = z.size();
  nda::array<ComplexType, 4> G(nw, nk, nb, nb);
  G() = 0.0;
  for (long k = 0; k < nk; ++k)
    for (auto s : {sector_t::particle, sector_t::hole}) {
      auto const &ps = pd(k, s);
      for (long m = 0; m < ps.size(); ++m) {
        nda::matrix<ComplexType> C(nb, nb);
        if (ps.is_factorized())
          for (long i = 0; i < nb; ++i)
            for (long j = 0; j < nb; ++j) C(i, j) = ps.v(i, m) * std::conj(ps.v(j, m));
        else
          C = ps.coef(m, nda::range::all, nda::range::all);
        for (long n = 0; n < nw; ++n) {
          const ComplexType r = 1.0 / (z(n) - ps.e(m));
          for (long i = 0; i < nb; ++i)
            for (long j = 0; j < nb; ++j) G(n, k, i, j) += r * C(i, j);
        }
      }
    }
  return G;
}
/// X = z + mu - G^{-1}(z) per (n, k)
nda::array<ComplexType, 4> x_of_g(nda::array<ComplexType, 4> const &G, nda::array<ComplexType, 1> const &z, double mu) {
  nda::array<ComplexType, 4> X(G.shape());
  const long nb = G.extent(2);
  for (long n = 0; n < G.extent(0); ++n)
    for (long k = 0; k < G.extent(1); ++k) {
      nda::matrix<ComplexType> Gi = nda::inverse(nda::matrix<ComplexType>(G(n, k, nda::range::all, nda::range::all)));
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) X(n, k, i, j) = (i == j ? z(n) + mu : ComplexType(0.0)) - Gi(i, j);
    }
  return X;
}
} // namespace

TEST_CASE("gw_line_finiteT_T4c", "[.][gw_line][finiteT][T4c]") {
  using namespace methods;
  lih_ft_t L;
  auto &comm = L.mpi->comm;
  auto &mf   = *L.mf;
  const long NIT = 5, nk = mf.nkpts(), nb = L.thc->nbnd();
  const double beta = 200.0;
  double w_max = 0.0;
  for (long k = 0; k < nk; ++k)
    for (long n = 0; n < nb; ++n) w_max = std::max(w_max, std::abs(mf.eigval()(0, k, n)));
  w_max += 2.0;
  // ---- CoQui scGW
  imag_axes_ft::IAFT ft(beta, w_max + 1.0, imag_axes_ft::dlr_basis, "high");
  const std::string cq = "coqui_gw_line_ft_t4c";
  auto t0 = std::chrono::steady_clock::now();
  {
    simple_dyson dyson(&mf, &ft);
    solvers::hf_t hf("ignore_g0");   // the line's hf_div_treatment (CoQui's default is gygi)
    solvers::gw_t gw(&ft, "ignore_g0", cq);
    solvers::scr_coulomb_t scr(&ft, "rpa", "ignore_g0");
    auto eri = mb_eri_t(*L.thc, *L.thc);
    MBState mb_state(L.mpi, ft, cq);
    iter_scf::iter_scf_t damp(iter_scf::damp_t(0.5));
    scf_loop(mb_state, dyson, eri, ft, solvers::mb_solver_t(&hf, &gw, &scr), &damp, NIT, false, 1e-14, false);
  }
  comm.barrier();
  const double tC = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  auto wn = ft.wn_mesh_f();
  const long nw = ft.nw_f();
  nda::array<ComplexType, 1> z(nw);
  for (long n = 0; n < nw; ++n) z(n) = ft.omega(wn(n));
  double mu_C = 0.0;
  long itC    = 0;
  std::vector<double> muC_it;
  nda::array<ComplexType, 4> XC, FC, GC;
  {
    nda::array<ComplexType, 5> Gt, Gw(nw, 1, nk, nb, nb);
    nda::array<ComplexType, 4> F;
    {
      h5::file f(cq + ".mbpt.h5", 'r');
      h5::group g(f);
      auto sg = g.open_group("scf");
      h5::h5_read(sg, "final_iter", itC);
      auto gi = sg.open_group("iter" + std::to_string(itC));
      h5::h5_read(gi, "mu", mu_C);
      for (long i = 1; i <= itC; ++i) {
        double m = 0.0;
        h5::h5_read(sg.open_group("iter" + std::to_string(i)), "mu", m);
        muC_it.push_back(m);
      }
      nda::h5_read(gi, "G_tskij", Gt);
      nda::h5_read(gi, "F_skij", F);
    }
    ft.tau_to_w(Gt, Gw, imag_axes_ft::fermion);
    nda::array<ComplexType, 4> G(nw, nk, nb, nb);
    for (long n = 0; n < nw; ++n) G(n, nda::ellipsis{}) = Gw(n, 0, nda::ellipsis{});
    XC = x_of_g(G, z, mu_C);
    GC = G;
    FC = F;
  }
  long nmax = 0;
  for (long n = 0; n < nw; ++n)
    if (std::abs(z(n).imag()) > std::abs(z(nmax).imag())) nmax = n;
  app_log(1, "\n[finiteT][T4c] lih222 beta {}: CoQui scGW {} iterations ({:.1f} s, {} ranks): mu {:.10f}, {} Matsubara points "
             "(max |w_n| {:.1f} Ha)",
          beta, itC, tC, comm.size(), mu_C, nw, std::abs(z(nmax).imag()));
  struct case_t {
    std::string name, density;
    double tfrac;
  };
  const std::vector<case_t> cases = {{"closure", "closure", 0.0}, {"matsubara", "matsubara", 0.0}, {"matsubara_t4", "matsubara", 0.25}};
  for (auto const &c : cases) {
    auto pt = ft_params("gw_line_ft_t4c_" + c.name, NIT, beta, "id", "auto");
    pt.put("scf_density", c.density);
    if (c.tfrac > 0.0) pt.put("theta_t_frac", c.tfrac);
    auto t1 = std::chrono::steady_clock::now();
    auto R  = gw_line_scf<HOST_MEMORY>(*L.thc, mf, pt);
    const double tl = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    app_log(1, "  -- gw_line scf_density {} theta_t_frac {} ({:.1f} s):", c.density,
            c.tfrac > 0.0 ? std::to_string(c.tfrac) : std::string("default"), tl);
    for (auto const &h : R.history) {
      char ex[128] = "";
      if (c.density == "matsubara") std::snprintf(ex, sizeof(ex), " | closure N %.6f max|D_cl - D| %.1e", h.N_closure, h.D_cl_err);
      app_log(1, "     iter {}: mu {:.10f} (- CoQui same iteration {:+.3e}) rule {} dN {:+.2e} N(mu) {:.12f} gap {:.4f} eV{}", h.iter, h.mu,
              h.mu - (h.iter <= long(muC_it.size()) ? muC_it[h.iter - 1] : mu_C), h.mu_rule, h.dN, h.N_mu, h.gap * 27.211386, std::string(ex));
    }
    utils::check(R.poles.nk == nk and R.H0.extent(0) == nk, "T4c: the line's poles are not on the full k mesh");
    auto GL = g_of_poles(R.poles, z);
    auto XL = x_of_g(GL, z, R.mu);
    // the closure's Lehmann G keeps a total weight W0 = sum of residues != 1 (dropped / pruned poles): X grows like
    // i w (1 - 1/W0) at large |w_n|, so Sigma is compared on |w_n| <= 2 Ha (the spectral range) and G on all points
    double dg = 0.0, gm = 0.0, wdef = 0.0;
    for (long k = 0; k < nk; ++k) {
      for (long n = 0; n < nw; ++n)
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j) {
            dg = std::max(dg, std::abs(GL(n, k, i, j) - GC(n, k, i, j)));
            gm = std::max(gm, std::abs(GC(n, k, i, j)));
          }
      double w0 = 0.0;
      for (auto sct : {sector_t::particle, sector_t::hole}) {
        auto const &ps = R.poles(k, sct);
        for (long m = 0; m < ps.size(); ++m)
          for (long i = 0; i < nb; ++i) w0 += ps.is_factorized() ? std::norm(ps.v(i, m)) : ps.coef(m, i, i).real();
      }
      wdef = std::max(wdef, std::abs(w0 / double(nb) - 1.0));
    }
    // total Sigma = X - H0 (both codes), Sigma_c = X - H0 - F (CoQui F_skij = V_H + Sigma_x; line F_closure, the F of the poles)
    double dx = 0.0, xm = 0.0, ds = 0.0, sm = 0.0, df = 0.0, fm = 0.0;
    for (long k = 0; k < nk; ++k)
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) {
          df = std::max(df, std::abs(R.F_closure(k, i, j) - FC(0, k, i, j)));
          fm = std::max(fm, std::abs(FC(0, k, i, j)));
          for (long n = 0; n < nw; ++n) {
            if (std::abs(z(n).imag()) > 2.0) continue;
            const ComplexType aC = XC(n, k, i, j) - R.H0(k, i, j), aL = XL(n, k, i, j) - R.H0(k, i, j);
            const ComplexType sC = aC - FC(0, k, i, j), sL = aL - R.F_closure(k, i, j);
            dx = std::max(dx, std::abs(aL - aC));
            xm = std::max(xm, std::abs(aC));
            ds = std::max(ds, std::abs(sL - sC));
            sm = std::max(sm, std::abs(sC));
          }
        }
    app_log(1, "     vs CoQui (final iteration): |dmu| {:.3e} Ha; G(i w_n) max|d| / max {:.3e} (all {} points; line weight deficit "
               "max_k |Tr W0 / nb - 1| {:.1e}); |w_n| <= 2 Ha: total Sigma max|d| / max {:.3e} (max {:.3e} Ha), Sigma_c {:.3e} (max "
               "{:.3e} Ha); F {:.3e} Ha (max {:.3e})",
            std::abs(R.mu - mu_C), dg / gm, nw, wdef, dx / xm, xm, ds / sm, sm, df, fm);
  }
  REQUIRE(true);
}

// ============================================================================================================== S8b.4 gauge
// [finiteT][gauge] (plan S8b.4 task 4a, notes sec:fT_gauge): the free terminal block A0 = P0 V Q0^dag of the closure when
// n_free > 1. Problems: a synthetic 3 x 3 measure (40 poles, K 6: n_free 3) and the Si 2x2x2 Gamma moments of
// si222_moments_k0.h5 at K 4 (Nr 290, r1 232, n_free 58 = n; the Gram rank exceeds K r there already at the stored omega_p).
//   (i)   A0 invariant under random unitary mixing of the null bases P0 -> P0 V_L, Q0 -> Q0 V_R ("orbital", "canonical"; "svd"
//         reported: O(1));
//   (ii)  the moments 0..K reproduced by every rule;
//   (iii) BLAS threads 1 vs 4: Sigma_c(z) and the poles of "orbital" identical to 1e-8 (MKL measured 5.2e-10 = the continuity of (iv); "svd": O(1) with MKL;
//         OpenBLAS on the Mac is thread-invariant here);
//   (iv)  continuity: 1e-15 relative noise on the moments moves Sigma_c(z) of "orbital" continuously ("svd" reported);
//   (v)   n_free <= 1: every rule bitwise the pre-S8b.4 realization ("svd").
#include "methods/GW_line/blas_scope.hpp"
#include <random>
namespace {
namespace ldlr = numerics::line_dlr;

/// Cayley moments 0..K+1 of npol poles in [-emax, emax] with seeded rank-1 residues v v^dag
nda::array<ComplexType, 3> gauge_measure(long n, long npol, double emax, double wp, long K, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  nda::array<double, 1> w(npol);
  nda::array<ComplexType, 3> g(npol, n, n);
  for (long l = 0; l < npol; ++l) {
    w(l) = emax * U(rng);
    std::vector<ComplexType> v(n);
    for (auto &x : v) x = ComplexType(U(rng), U(rng)) / std::sqrt(double(npol));
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) g(l, i, j) = v[i] * std::conj(v[j]);
  }
  return ldlr::moments_from_poles(w, g, wp, K + 1);
}

ldlr::cmatrix_F random_unitary(long n, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> N(0.0, 1.0);
  ldlr::cmatrix_F M(n, n);
  for (long j = 0; j < n; ++j)
    for (long i = 0; i < n; ++i) M(i, j) = ComplexType(N(rng), N(rng));
  nda::array<double, 1> s;
  return ldlr::detail::polar_unitary(M, s);
}

/// Sigma_c(z) = sum_l W_l W_l^dag / (z - d_l) at 25 points x + 0.05 i, x in [-3, 3] Ha (flattened)
std::vector<ComplexType> gauge_sigma(ldlr::upfold_result_t const &r) {
  const long n = r.W.extent(0), np = r.d.size();
  std::vector<ComplexType> out;
  for (long p = 0; p < 25; ++p) {
    const ComplexType z(-3.0 + 0.25 * double(p), 0.05);
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) {
        ComplexType s(0.0);
        for (long l = 0; l < np; ++l) s += r.W(i, l) * std::conj(r.W(j, l)) / (z - r.d(l));
        out.push_back(s);
      }
  }
  return out;
}
double rel_vec(std::vector<ComplexType> const &a, std::vector<ComplexType> const &b) {
  double d = 0.0, m = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    d = std::max(d, std::abs(a[i] - b[i]));
    m = std::max(m, std::abs(a[i]));
  }
  return d / std::max(m, 1e-300);
}
/// max_m<=K ||W diag(u^m) W^dag - C^(m)|| / ||C^(0)|| (and the held-out m = K + 1)
std::pair<double, double> gauge_moment_err(ldlr::upfold_result_t const &r, nda::array<ComplexType, 3> const &C, long K) {
  const long n = r.W.extent(0), np = r.d.size();
  double c0 = 0.0, emax = 0.0, eheld = 0.0;
  for (long i = 0; i < n; ++i)
    for (long j = 0; j < n; ++j) c0 += std::norm(C(0, i, j));
  c0 = std::sqrt(c0);
  for (long m = 0; m <= K + 1; ++m) {
    double e = 0.0;
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) {
        ComplexType s(0.0);
        for (long l = 0; l < np; ++l) s += r.W(i, l) * std::pow(r.u(l), double(m)) * std::conj(r.W(j, l));
        e += std::norm(s - C(m, i, j));
      }
    e = std::sqrt(e) / c0;
    if (m <= K) emax = std::max(emax, e);
    else eheld = e;
  }
  return {emax, eheld};
}

ldlr::upfold_opts_t gauge_opts(std::string const &rule, double tol_gram) {
  ldlr::upfold_opts_t o;
  o.tol_gram   = tol_gram;
  o.nphi       = 8;
  o.svd_driver = "gesvd";
  o.scan_err   = "poly";
  o.free_block = rule;
  return o;
}

/// one problem: the checks (i)-(iv) of the test-case comment
bool gauge_checks(std::string const &name, nda::array<ComplexType, 3> const &C, long K, double wp, double tol_gram) {
  bool ok = true;
  const long nC = C.extent(1);
  // (i) invariance of A0 under the null-basis ambiguity (the reference SVD of phase A, then mixed)
  {
    auto o = gauge_opts("orbital", tol_gram);
    o.svd_driver = "gesdd";
    ldlr::upfold_result_t res;
    auto pr = ldlr::upfold_prepare(C, K, wp, o, res, true);
    REQUIRE(pr.need_ref);
    const long Nr = pr.Nr;
    ldlr::cmatrix_F M = pr.Mref, P(Nr, Nr), Qh(Nr, Nr);
    nda::array<double, 1> sv(Nr);
    nda::lapack::gesvd(M, sv, P, Qh);
    const long r1 = ldlr::detail::svd_rank(sv, o, res), f = Nr - r1;
    REQUIRE(f > 1);
    auto VL = random_unitary(f, 11), VR = random_unitary(f, 12);
    ldlr::cmatrix_F P2(P), Qh2(Qh);
    {
      ldlr::cmatrix_F P0 = P(nda::range::all, nda::range(r1, Nr)), Q0h = Qh(nda::range(r1, Nr), nda::range::all);
      P2(nda::range::all, nda::range(r1, Nr))  = ldlr::detail::mm(P0, VL);
      Qh2(nda::range(r1, Nr), nda::range::all) = ldlr::detail::mm(VR, Q0h, 'C', 'N');   // Q0 V_R
    }
    app_log(1, "\n[finiteT][gauge] {}: n {} K {} wp {} | Nr {} r1 {} n_free {}", name, nC, K, wp, Nr, r1, f);
    for (std::string rule : {"orbital", "canonical", "svd"}) {
      auto oo = gauge_opts(rule, tol_gram);
      auto qa = pr, qb = pr;
      ldlr::upfold_result_t ra, rb;
      ldlr::detail::set_free_block(qa, P, Qh, r1, ra, oo);
      ldlr::detail::set_free_block(qb, P2, Qh2, r1, rb, oo);
      double d = 0.0, m = 0.0;
      for (long x = 0; x < qa.A0.size(); ++x) {
        d += std::norm(qa.A0.data()[x] - qb.A0.data()[x]);
        m += std::norm(qa.A0.data()[x]);
      }
      const double rel = std::sqrt(d / m);
      if (rule == "svd")
        app_log(1, "    {:<52s} {:.3e}  (report: the SVD pairing)", "(i) svd: |A0(mixed) - A0| / |A0|", rel);
      else
        ok = gate(("(i) " + rule + ": |A0(mixed) - A0| / |A0| (sigma ratio " + std::to_string(ra.free_sig).substr(0, 8) + ")").c_str(), rel,
                  1e-13) and ok;
    }
  }
  // (ii) moments, (iii) threads, (iv) continuity
  nda::array<ComplexType, 3> Cn(C);
  {
    std::mt19937 rng(4242);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (long a = 0; a < Cn.size(); ++a) Cn.data()[a] *= 1.0 + 1e-15 * ComplexType(U(rng), U(rng));
  }
  for (std::string rule : {"orbital", "canonical", "svd"}) {
    auto o = gauge_opts(rule, tol_gram);
    ldlr::upfold_result_t r1t, r4t, rn;
    {
      blas_threads_scope_t b(1);
      r1t = ldlr::upfold_block(C, K, wp, o);
    }
    {
      blas_threads_scope_t b(4);
      r4t = ldlr::upfold_block(C, K, wp, o);
    }
    {
      blas_threads_scope_t b(1);
      rn = ldlr::upfold_block(Cn, K, wp, o);
    }
    auto [em, eh] = gauge_moment_err(r1t, C, K);
    auto s1 = gauge_sigma(r1t), s4 = gauge_sigma(r4t), sn = gauge_sigma(rn);
    const double dthr = rel_vec(s1, s4), dnz = rel_vec(s1, sn);
    app_log(1, "    {:<9s}: n_free {} free-block sigma ratio {:.2e} phi {:.6f} | moments <= K {:.2e} held-out {:.2e} | threads 1 vs 4 "
               "Sigma {:.2e} | 1e-15 noise Sigma {:.2e}",
            rule, r1t.n_free, r1t.free_sig, r1t.phi, em, eh, dthr, dnz);
    ok = gate(("(ii) " + rule + ": moments 0..K (rel. ||C0||)").c_str(), em, 1e-12) and ok;
    if (rule == "orbital") {
      ok = gate("(iii) orbital: BLAS threads 1 vs 4, Sigma_c(z)", dthr, 1e-8) and ok;
      ok = gate("(iv) orbital: 1e-15 moment noise, Sigma_c(z) [continuity]", dnz, 1e-7) and ok;
    }
  }
  return ok;
}
} // namespace

TEST_CASE("gw_line_finiteT_gauge", "[gw_line][finiteT][gauge]") {
  bool ok = true;
  {   // synthetic 3 x 3 measure
    const long K = 6;
    auto C = gauge_measure(3, 40, 3.0, 1.0, K, 7);
    ok = gauge_checks("synthetic 3x3 (40 poles)", C, K, 1.0, 1e-12) and ok;
  }
  {   // Si 2x2x2 Gamma moments (T = 0 G0W0, stored omega_p 0.11)
    h5::file f(ft_dir() + "si222_moments_k0.h5", 'r');
    h5::group g(f);
    auto gs = g.open_group("si_k0");
    const double wp = read_attr<double>(gs, "wp"), tg = read_attr<double>(gs, "tol_gram");
    nda::array<double, 3> Cre, Cim;
    nda::h5_read(gs, "C_re", Cre);
    nda::h5_read(gs, "C_im", Cim);
    const long K = 4;
    nda::array<ComplexType, 3> C(K + 2, Cre.extent(1), Cre.extent(2));
    for (long m = 0; m < K + 2; ++m)
      for (long i = 0; i < C.extent(1); ++i)
        for (long j = 0; j < C.extent(2); ++j) C(m, i, j) = ComplexType(Cre(m, i, j), Cim(m, i, j));
    ok = gauge_checks("si222 Gamma moments", C, K, wp, tg) and ok;
    // vs the prototype (si222_gauge_ref.h5, gen_gauge_ref.py): the closed-form F, phi, held-out error, Sigma_c(z)
    h5::file fr(ft_dir() + "si222_gauge_ref.h5", 'r');
    h5::group gr(fr);
    REQUIRE(read_attr<long>(gr, "K") == K);
    auto Fpy = read_c<2>(gr, "mixing_F");
    auto const &F = ldlr::mixing_unitary(C.extent(1));
    double dF = 0.0;
    for (long i = 0; i < F.extent(0); ++i)
      for (long j = 0; j < F.extent(1); ++j) dF = std::max(dF, std::abs(F(i, j) - Fpy(i, j)));
    app_log(1, "\n[finiteT][gauge] si222 K {} vs the prototype (free_block rules of cayley/upfold.py):", K);
    ok = gate("closed-form mixing unitary F (58 x 58) vs python", dF, 1e-12) and ok;
    for (std::string rule : {"orbital", "canonical"}) {
      auto o = gauge_opts(rule, tg);
      o.nphi = read_attr<long>(gr, "nphi");
      auto r  = ldlr::upfold_block(C, K, wp, o);
      auto gg = gr.open_group(rule);
      auto Spy = read_c<3>(gg, "sigma");
      std::vector<ComplexType> sp(Spy.data(), Spy.data() + Spy.size());
      const double dphi = std::abs(r.phi - read_attr<double>(gg, "phi"));
      const double dres = std::abs(r.residual / read_attr<double>(gg, "heldout_err") - 1.0);
      app_log(1, "    {:<9s}: n_free {} ({}) phi {:.10f} ({:.10f}) held-out {:.6e} ({:.6e}) sigma ratio {:.6e} ({:.6e})", rule, r.n_free,
              read_attr<long>(gg, "n_free"), r.phi, read_attr<double>(gg, "phi"), r.residual, read_attr<double>(gg, "heldout_err"),
              r.free_sig, read_attr<double>(gg, "sig_min"));
      ok = ok and r.n_free == read_attr<long>(gg, "n_free");
      ok = gate((rule + ": |phi - phi_py| (golden-section roundoff)").c_str(), dphi, 1e-5) and ok;
      ok = gate((rule + ": held-out error rel. to python").c_str(), dres, 1e-6) and ok;
      app_log(1, "    {:<52s} {:.3e}  (report: phi of the golden section differs by {:.1e})", (rule + ": Sigma_c(z) vs python, own phi").c_str(),
              rel_vec(sp, gauge_sigma(r)), dphi);
      o.force_phi = read_attr<double>(gg, "phi");   // the same terminal phase: the realization of the same free block
      auto rf = ldlr::upfold_block(C, K, wp, o);
      ok = gate((rule + ": Sigma_c(z) vs python at phi_py (25 points)").c_str(), rel_vec(sp, gauge_sigma(rf)), 1e-8) and ok;
      if (rule == "orbital")
        ok = gate("orbital: sigma_min / sigma_max of L^dag F Rm^dag rel. to python",
                  std::abs(r.free_sig / read_attr<double>(gg, "sig_min") - 1.0), 1e-8) and ok;
    }
  }
  {   // (v) n_free <= 1: every rule bitwise "svd" (synthetic 3 x 3 measure with 8 poles: Nr 8 <= K r)
    const long K = 6;
    auto C = gauge_measure(3, 8, 3.0, 1.0, K, 9);
    auto a = ldlr::upfold_block(C, K, 1.0, gauge_opts("svd", 1e-12));
    bool same = true;
    for (std::string rule : {"orbital", "canonical"}) {
      auto b = ldlr::upfold_block(C, K, 1.0, gauge_opts(rule, 1e-12));
      same = same and a.d.size() == b.d.size() and a.phi == b.phi;
      for (long x = 0; same and x < a.d.size(); ++x) same = (a.d(x) == b.d(x));
      for (long x = 0; same and x < a.W.size(); ++x) same = (a.W.data()[x] == b.W.data()[x]);
    }
    app_log(1, "\n[finiteT][gauge] n_free {} (<= 1): orbital / canonical bitwise svd: {}", a.n_free, same ? "yes" : "NO");
    ok = ok and same and a.n_free <= 1;
  }
  REQUIRE(ok);
}

// ============================================================================================================== [sharp]
// S8c task 3 (notes section 12.6): the sharper final pass (lih222 and svo222n at beta 200: zeta_T' 0.0505, omega_p' 0.758 Ha). G0W0@KS (niter = 0 + final_pass: the pass IS the first iteration,
// from the KS poles at theta_t' = theta / 4, K 32, omega_p' = 15 zeta_T', node data at eps 1e-10, thermal_tol 1e-12 in the pass):
//  (i)  Sigma_c of the pass at the nodes vs the exact finite-T Casida Sigma_c (sharp_ref.h5, the run's 240 nodes) on the theta / 2 mask
//       (rho beta |z| >= c_f at theta / 2) and on the pass's own mask (rho' beta |z| >= c_f, rho' = 2.97): <= 1e-8;
//  (ii) A(k, w) of the pass's closure in |w - mu0| <= 0.3 Ha vs the exact A from the exact Sigma_c(w + i eta) (sharp_ref.h5,
//       cayley/scripts/gen_sharp_ref.py) with the run's own static part H0 + F[D_KS]: max|dA| / max A <= 1e-2 at eta 0.02 and
//       <= 1e-3 at eta 0.05 (S8c.0 assessment, k = 1 of svo222: 2.9e-3 / 1.5e-4); the SCF-angle (theta / 2) G0W0 closure is
//       reported next to it (no gate);
//  (i') the SCF-angle (theta / 2) G0W0 Sigma_c at the nodes vs exact on the theta / 2 mask (a T2-type kernel check through the
//       driver; svo222n replaces a stand-alone svo222 T1/T2 case);
//  (iii) lih222: the pass from a restart (niter = 1 done, restart, final_pass) == the in-run pass (niter = 1 + final_pass), bitwise.
namespace {
ptree sharp_params(std::string const &out, double beta, long niter, bool restart, double mu0, bool pass) {
  auto pt = ft_params(out, niter, beta, "id", "neutral", restart);
  pt.put("eps", 1e-10);   // node data at 1e-10 (the S8c.0 assessment's level)
  pt.put("nphi", 72);
  pt.put("K", 32);
  pt.put("spectra.enable", true);
  ptree eta, v1, v2;
  v1.put("", 0.02);
  v2.put("", 0.05);
  eta.push_back({"", v1});
  eta.push_back({"", v2});
  pt.put_child("spectra.eta", eta);
  pt.put("spectra.wmin", -0.3);
  pt.put("spectra.wmax", 0.3);
  pt.put("spectra.nw", 151);
  pt.put("spectra.origin", mu0);   // the reference's frame: w relative to the KS number-rule mu0
  pt.put("final_pass.enable", pass);
  pt.put("final_pass.theta_t_frac", 0.25);
  pt.put("final_pass.K", 32);
  pt.put("final_pass.thermal_tol", 1e-12);
  return pt;
}

/// exact A(k, w) (trace) from the exact Sigma_c(w + i eta) of sharp_ref.h5 and the static part H (absolute) of the run
nda::array<double, 3> exact_A(nda::array<ComplexType, 5> const &Sw, nda::array<double, 1> const &om, nda::array<double, 1> const &eta,
                              double mu0, std::vector<nda::array<ComplexType, 2>> const &H) {
  const long nks = Sw.extent(0), ne = Sw.extent(1), nw = Sw.extent(2), nb = Sw.extent(3);
  nda::array<double, 3> A(nks, ne, nw);
  for (long a = 0; a < nks; ++a)
    for (long ie = 0; ie < ne; ++ie)
      for (long iw = 0; iw < nw; ++iw) {
        nda::matrix<ComplexType> M(nb, nb);
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j)
            M(i, j) = (i == j ? ComplexType(om(iw) + mu0, eta(ie)) : ComplexType(0.0)) - H[a](i, j) - Sw(a, ie, iw, i, j);
        auto G   = nda::inverse(M);
        double t = 0.0;
        for (long i = 0; i < nb; ++i) t += -std::imag(G(i, i)) / std::numbers::pi;
        A(a, ie, iw) = t;
      }
  return A;
}

bool run_sharp(std::string const &fxn, double beta, bool check_restart) {
  fx_t F(fxn);
  auto &comm = F.mpi->comm;
  auto &mf   = *F.mf;
  const long nb = F.thc->nbnd();
  bool ok = true;
  auto sp = [&](std::string const &out, double bt, long niter, bool restart, double mu, bool pass) {
    auto pt = sharp_params(out, bt, niter, restart, mu, pass);
    return pt;
  };
  // references
  // svo222n: the reference lives with its data ($GW_LINE_SVO222N/sharp_ref_svo222n.h5, gen_sharp_ref.py on rusty)
  const std::string sref = (fxn == "svo222n") ? std::string(std::getenv("GW_LINE_SVO222N")) + "/sharp_ref_svo222n.h5" : ft_dir() + "sharp_ref.h5";
  h5::file hs(sref, 'r');
  h5::group rs(hs);
  auto gs          = rs.open_group(fxn);
  const double mu0 = read_attr<double>(gs, "mu0");
  nda::array<double, 1> om, eta;
  nda::h5_read(gs, "om", om);
  nda::h5_read(gs, "eta", eta);
  nda::array<float, 5> swr, swi;
  nda::h5_read(gs, "Sw_re", swr);
  nda::h5_read(gs, "Sw_im", swi);
  nda::array<ComplexType, 5> Sw(swr.shape());
  for (long a = 0; a < Sw.size(); ++a) Sw.data()[a] = ComplexType(swr.data()[a], swi.data()[a]);
  const std::vector<long> ks = (fxn == "lih222") ? std::vector<long>{0, 3} : std::vector<long>{1};   // svo222n: k 1
  REQUIRE(long(ks.size()) == Sw.extent(0));
  nda::array<double, 1> zr, zi;
  nda::array<double, 4> szr, szi;
  nda::h5_read(gs, "zeta_re", zr);
  nda::h5_read(gs, "zeta_im", zi);
  nda::h5_read(gs, "Sz_re", szr);
  nda::h5_read(gs, "Sz_im", szi);
  nda::array<ComplexType, 1> zf(zr.size());
  for (long i = 0; i < zf.size(); ++i) zf(i) = ComplexType(zr(i), zi(i));
  nda::array<ComplexType, 4> SZ(szr.shape());   // exact Sigma_c at the run's line nodes (n_k, 240, nb, nb)
  for (long a = 0; a < SZ.size(); ++a) SZ.data()[a] = ComplexType(szr.data()[a], szi.data()[a]);
  const std::vector<long> &kz = ks;

  // (i) + (ii): the G0W0@KS pass
  const std::string fo = "gw_line_sharp_" + fxn;
  auto t0  = std::chrono::steady_clock::now();
  auto R   = gw_line_scf<HOST_MEMORY>(*F.thc, mf, sp(fo, beta, 0, false, mu0, true));
  const double tp = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  REQUIRE(R.fp_done);
  REQUIRE(R.fp_spectra.has_value());
  app_log(1, "\n[finiteT][sharp] {} beta {}: pass theta_t' {:.3f} deg, zeta_T' {:.4f} Ha, omega_p' {:.4f} Ha, K {}, mu {:.12f} (mu0 {:.12f}), {:.1f} s",
          fxn, beta, R.fp_theta_t * 180.0 / std::numbers::pi, R.fp_zeta_T, R.fp_wp, R.fp_K, R.fp_mu, mu0, tp);
  // the reference's fermionic nodes are a subset of the run's line nodes (same rays / node set): map them
  std::vector<long> zmap(zf.size(), -1);
  {
    double dzn = 0.0;
    for (long i = 0; i < zf.size(); ++i) {
      double best = 1e300;
      for (long j = 0; j < R.zeta.size(); ++j)
        if (std::abs(R.zeta(j) - zf(i)) < best) best = std::abs(R.zeta(j) - zf(i)), zmap[i] = j;
      dzn = std::max(dzn, best);
    }
    ok = gate("reference nodes found among the run's nodes", dzn, 1e-12) and ok;
  }
  {
    auto S = sigma_total_of(comm, R.fp_file, 1);
    const double deg = std::numbers::pi / 180.0, rho4 = std::sin(20.0 * deg - 5.0 * deg) / std::sin(5.0 * deg);
    double d1 = 0.0, d2 = 0.0, m = 0.0;
    long n2 = 0;
    for (size_t a = 0; a < kz.size(); ++a)
      for (long i = 0; i < zf.size(); ++i) {
        const bool own = rho4 * beta * std::abs(zf(i)) >= 30.0, half = beta * std::abs(zf(i)) >= 30.0;   // rho(theta / 2) = 1
        const bool fmi = half;
        if (own and not fmi) ++n2;
        if (not own) continue;
        for (long x = 0; x < nb; ++x)
          for (long y = 0; y < nb; ++y) {
            const double d = std::abs(S(kz[a], zmap[i], x, y) - SZ(a, i, x, y));
            m = std::max(m, std::abs(SZ(a, i, x, y)));
            if (fmi) d1 = std::max(d1, d);
            else d2 = std::max(d2, d);
          }
      }
    ok = gate("(i) pass Sigma_c vs exact, reference mask (theta / 2 floor)", d1 / m, 1e-8) and ok;
    ok = gate(("(i) pass Sigma_c vs exact, the pass's extra nodes (" + std::to_string(n2 / long(kz.size())) + ", theta / 4 floor)").c_str(),
              d2 / m, 1e-8) and ok;
  }
  {
    std::vector<nda::array<ComplexType, 2>> H;
    for (long k : ks) H.emplace_back(R.fp_H0(k, nda::range::all, nda::range::all) + R.fp_F_closure(k, nda::range::all, nda::range::all));
    auto Ae = exact_A(Sw, om, eta, mu0, H);
    auto const &sp = *R.fp_spectra;
    REQUIRE(sp.omega.size() == om.size());
    double dom = 0.0;
    for (long i = 0; i < om.size(); ++i) dom = std::max(dom, std::abs(sp.omega(i) - om(i)));
    ok = gate("spectra grid == reference grid (origin mu0)", dom, 1e-12) and ok;
    const double gates[2] = {1e-2, 1e-3};
    for (long ie = 0; ie < eta.size(); ++ie)
      for (size_t a = 0; a < ks.size(); ++a) {
        double d = 0.0, m = 0.0;
        for (long iw = 0; iw < om.size(); ++iw) {
          d = std::max(d, std::abs(sp.A_trace(ie, ks[a], iw) - Ae(a, ie, iw)));
          m = std::max(m, Ae(a, ie, iw));
        }
        ok = gate(("(ii) pass A(k = " + std::to_string(ks[a]) + ", w) vs exact, eta " + std::to_string(eta(ie)).substr(0, 4) + " (max A " +
                   std::to_string(m).substr(0, 6) + ")").c_str(),
                  d / m, gates[std::min(ie, 1L)]) and ok;
      }
  }
  rm_ckpt(comm, fo);
  rm_ckpt(comm, fo + ".final_pass");
  // the SCF-angle (theta / 2) G0W0@KS closure, reported
  {
    auto pt = sp(fo + "_scf", beta, 1, false, mu0, false);
    pt.put("thermal_tol", 1e-12);
    auto Q  = gw_line_scf<HOST_MEMORY>(*F.thc, mf, pt);
    {   // the SCF-angle G0W0 Sigma_c at the nodes vs exact on the theta / 2 mask (the T2-type kernel check through the driver;
        // svo222n: the metal at beta 200)
      auto S = sigma_total_of(comm, fo + "_scf.gw_line.h5", 1);
      double d = 0.0, m = 0.0;
      for (size_t a = 0; a < kz.size(); ++a)
        for (long i = 0; i < zf.size(); ++i) {
          if (beta * std::abs(zf(i)) < 30.0) continue;
          for (long x = 0; x < nb; ++x)
            for (long y = 0; y < nb; ++y) {
              d = std::max(d, std::abs(S(kz[a], zmap[i], x, y) - SZ(a, i, x, y)));
              m = std::max(m, std::abs(SZ(a, i, x, y)));
            }
        }
      ok = gate("(i') SCF-angle (theta / 2) G0W0 Sigma_c vs exact, theta / 2 mask", d / m, 1e-8) and ok;
    }
    std::vector<nda::array<ComplexType, 2>> H;
    for (long k : ks) H.emplace_back(Q.H0(k, nda::range::all, nda::range::all) + Q.F_closure(k, nda::range::all, nda::range::all));
    auto Ae = exact_A(Sw, om, eta, mu0, H);
    for (long ie = 0; ie < eta.size(); ++ie)
      for (size_t a = 0; a < ks.size(); ++a) {
        double d = 0.0, m = 0.0;
        for (long iw = 0; iw < om.size(); ++iw) {
          d = std::max(d, std::abs(Q.spectra->A_trace(ie, ks[a], iw) - Ae(a, ie, iw)));
          m = std::max(m, Ae(a, ie, iw));
        }
        app_log(1, "    SCF-angle (theta / 2, omega_p {:.4f}) G0W0 closure: A(k = {}, w) vs exact, eta {}: {:.3e} (report)",
                Q.history.back().wp_used, ks[a], eta(ie), d / m);
      }
    rm_ckpt(comm, fo + "_scf");
  }
  // (iii) restart pass == in-run pass
  if (check_restart) {
    const std::string fa = "gw_line_sharp_a", fb = "gw_line_sharp_b";
    auto A = gw_line_scf<HOST_MEMORY>(*F.thc, mf, sp(fa, beta, 1, false, mu0, true));
    gw_line_scf<HOST_MEMORY>(*F.thc, mf, sp(fb, beta, 1, false, mu0, false));
    auto B = gw_line_scf<HOST_MEMORY>(*F.thc, mf, sp(fb, beta, 1, true, mu0, true));
    REQUIRE((A.fp_done and B.fp_done and A.fp_spectra and B.fp_spectra));
    double dA = 0.0;
    for (long a = 0; a < A.fp_spectra->A_trace.size(); ++a)
      dA = std::max(dA, std::abs(A.fp_spectra->A_trace.data()[a] - B.fp_spectra->A_trace.data()[a]));
    auto Sa = sigma_total_of(comm, A.fp_file, 2), Sb = sigma_total_of(comm, B.fp_file, 2);
    double dS = 0.0;
    for (long a = 0; a < Sa.size(); ++a) dS = std::max(dS, std::abs(Sa.data()[a] - Sb.data()[a]));
    app_log(1, "  (iii) restart pass vs in-run pass: mu {:.15f} / {:.15f}, max|dA| {:.1e}, max|dSigma| {:.1e}", A.fp_mu, B.fp_mu, dA, dS);
    ok = (A.fp_mu == B.fp_mu and dA == 0.0 and dS == 0.0) and ok;
    for (auto const &s : {fa, fb}) {
      rm_ckpt(comm, s);
      rm_ckpt(comm, s + ".final_pass");
    }
  }
  return ok;
}
} // namespace

TEST_CASE("gw_line_finiteT_sharp_lih222", "[gw_line][finiteT][sharp]") { REQUIRE(run_sharp("lih222", 200.0, true)); }

// S8c task 2 / 3 on the metal: the svo222 driver cases run on the nosym twin svo222n (the original svo222_thc fixture sits on the
// xml-route mean field -- IBZ k, full q, no pseudopotential data -- that the driver cannot run). Data outside the repository:
// GW_LINE_SVO222N=<rusty data/svo222n> (out/svo.coqui.h5, svo222n_thc/, sharp_ref_svo222n.h5). Hidden (cost, data): run
// "[.metal_sharp]" and "[.metal_scf]" explicitly.
TEST_CASE("gw_line_metal_sharp_svo222n", "[.metal_sharp]") { REQUIRE(run_sharp("svo222n", 200.0, false)); }

// S8c task 2 (metals): svo222 thermal SCF at beta 200 with the metals' settings (option A of the S8c.0 assessment: theta_t = theta / 4,
// scf_density "matsubara" = hybrid, K 24, omega_p = 15 zeta_T = 0.758 Ha, mu_rule default): N(mu) = 41 to 1e-10 at every iteration,
// restart (NIT / 2 + rest) bitwise = straight, 1 vs 2 ranks mu <= 1e-8 Ha per iteration (file gw_line_metal_scf_np<np>.h5; run 1 then 2
// ranks). The mu vs CoQui scGW at beta 200 (<= 5 meV) is the demo comparison (S8c task 4, runs/s8c_mats), not gated here.
// GW_LINE_METAL_NIT (default 6) for a local smoke run. Heavy (rusty): "[metal][scf]".
TEST_CASE("gw_line_metal_scf_svo222n", "[.metal_scf]") {
  fx_t F("svo222n");
  auto &comm = F.mpi->comm;
  long NIT   = 6;
  if (char const *e = std::getenv("GW_LINE_METAL_NIT")) NIT = std::max(2L, std::strtol(e, nullptr, 10));
  const long NR = NIT / 2;
  const double nel = double(F.mf->nelec());
  auto params = [&](std::string const &out, long niter, bool restart) {
    auto pt = ft_params(out, niter, 200.0, "id", "neutral", restart);
    pt.put("theta_t_frac", 0.25);
    pt.put("scf_density", "matsubara");
    pt.put("K", 24);
    pt.put("nphi", 72);
    return pt;
  };
  const std::string fa = "gw_line_metal_scf_a", fb = "gw_line_metal_scf_b";
  auto t0 = std::chrono::steady_clock::now();
  auto A  = gw_line_scf<HOST_MEMORY>(*F.thc, *F.mf, params(fa, NIT, false));
  const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  bool ok = long(A.history.size()) == NIT;
  app_log(1, "\n[metal][scf] svo222: {} iterations in {:.1f} s ({} ranks), beta 200, theta_t = theta / 4, hybrid, K 24", NIT, dt, comm.size());
  for (auto const &h : A.history) {
    app_log(1, "  iter {}: mu {:.12f} Ha N(mu) {:.15f} dSigma {:.2e} | closure N {:.8f} own mu {:+.2e} ({}) | wp {:.4f}, {} frequencies, "
               "{:.1f} s",
            h.iter, h.mu, h.N_mu, h.dSigma, h.N_closure, h.dmu_closure, h.rule_closure, h.wp_used, h.nfreq, h.time);
    ok = gate("|N(mu) - N_el|", std::abs(h.N_mu - nel), 1e-10) and ok;
  }
  {   // restart
    gw_line_scf<HOST_MEMORY>(*F.thc, *F.mf, params(fb, NR, false));
    auto B = gw_line_scf<HOST_MEMORY>(*F.thc, *F.mf, params(fb, NIT, true));
    app_log(1, "  restart {} + {} vs {} straight:", NR, NIT - NR, NIT);
    ok = same_runs(comm, A, B, fa, fb, NIT) and ok;
  }
  {   // 1 vs 2 ranks
    const std::string mine = "gw_line_metal_scf_np" + std::to_string(comm.size()) + ".h5";
    const std::string other = "gw_line_metal_scf_np" + std::to_string(comm.size() == 1 ? 2 : 1) + ".h5";
    nda::array<double, 1> mus(long(A.history.size()));
    for (long i = 0; i < mus.size(); ++i) mus(i) = A.history[i].mu;
    if (comm.root()) {
      h5::file f(mine, 'w');
      h5::group g(f);
      nda::h5_write(g, "mu", mus, false);
    }
    comm.barrier();
    if (std::filesystem::exists(other)) {
      nda::array<double, 1> mo;
      {
        h5::file f(other, 'r');
        h5::group g(f);
        nda::h5_read(g, "mu", mo);
      }
      for (long i = 0; i < std::min(mo.size(), mus.size()); ++i)
        ok = gate(("1 vs 2 ranks: |dmu| (Ha) iteration " + std::to_string(i + 1)).c_str(), std::abs(mo(i) - mus(i)), 1e-8) and ok;
    } else
      app_log(1, "  {} written; run with the other rank count to compare", mine);
  }
  rm_ckpt(comm, fa);
  rm_ckpt(comm, fb);
  REQUIRE(ok);
}

// D4 step 0 (mixing study, hidden): the svo222n hybrid SCF of [.metal_scf] with the mixing from the environment:
// GW_LINE_MIX (Sigma, default 0.5), GW_LINE_MIX_F (F, default 1), GW_LINE_DAMP_BELOW (default 0), GW_LINE_METAL_NIT (default 12),
// GW_LINE_CONV (conv_thr, default 1e-5). Prints the history (mu, dSigma, residual, F residual, closure count); no gate beyond N(mu).
TEST_CASE("gw_line_metal_mix_svo222n", "[.metal_mix]") {
  fx_t F("svo222n");
  auto env_d = [](char const *n, double d) { char const *e = std::getenv(n); return e ? std::strtod(e, nullptr) : d; };
  const long NIT = long(env_d("GW_LINE_METAL_NIT", 12));
  const double mix = env_d("GW_LINE_MIX", 0.5), mixF = env_d("GW_LINE_MIX_F", 1.0), dbelow = env_d("GW_LINE_DAMP_BELOW", 0.0);
  const double conv = env_d("GW_LINE_CONV", 1e-5);
  auto pt = ft_params("gw_line_metal_mix", NIT, 200.0, "id", "neutral", false);
  pt.put("theta_t_frac", 0.25);
  pt.put("scf_density", "matsubara");
  pt.put("K", 24);
  pt.put("nphi", 72);
  pt.put("mixing", mix);
  pt.put("mixing_F", mixF);
  pt.put("damp_below", dbelow);
  pt.put("conv_thr", conv);
  auto t0 = std::chrono::steady_clock::now();
  auto A  = gw_line_scf<HOST_MEMORY>(*F.thc, *F.mf, pt);
  const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  app_log(1, "\n[metal][mix] svo222n beta 200 theta / 4 hybrid K 24: mixing {} mixing_F {} damp_below {} conv_thr {:.0e}: {} iterations "
             "in {:.0f} s, converged {}",
          mix, mixF, dbelow, conv, A.history.size(), dt, A.converged);
  bool ok = true;
  for (auto const &h : A.history) {
    app_log(1, "  MIX it {:2d} mu {:.8f} dmu {:+.3e} dSigma {:.3e} resid {:.3e} residF {:.3e} | closure N {:.6f} max|D_cl - D| {:.2e} | {}",
            h.iter, h.mu, h.dmu, h.dSigma, h.resid, h.residF, h.N_closure, h.D_cl_err, h.mix);
    ok = gate("|N(mu) - N_el|", std::abs(h.N_mu - 41.0), 1e-10) and ok;
  }
  rm_ckpt(F.mpi->comm, "gw_line_metal_mix");
  REQUIRE(ok);
}
