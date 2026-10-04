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
 * S9a: the Coulomb head on the line (head.hpp), setup of [V2]/[V3] (THC nIpts = 8 nbnd, KS poles, mu mid-gap, theta = 20 deg,
 * theta_t = 10 deg, bosonic basis lam_b = max(4, 1.2 x largest transition), gap = KS gap / 2, eps = 1e-10).
 *
 * [V6] (V6a of plan S9) h(q, zeta) = eps^{-1}_00(q, zeta) - 1 for every finite q:
 *   (a) at the bosonic nodes from the block-layout W(q, zeta_i) (head_nodes_partial) vs the same projection of the exact Casida
 *       W_dyn (f sum_s lam_s |cb^T v_s|^2 / (z - lam_s)):                                          <= 1e-10 (rel. to max|h|)
 *   (b) from the scalar residues hp, hh (head_residues_partial, q <-> -q pairing of the hole part) at the nodes, 12 ray points
 *       and the bosonic Matsubara nodes vs the exact Casida head:                                   <= 1e-8
 *   (c) at i nu_n vs CoQui's div_utils::eval_eps_inv_q applied to CoQui's own W^c(q, i nu_n) (scr_coulomb_t("rpa", "gygi"),
 *       KS G, beta 1000; dW_qtPQ -> tau_to_w_PHsym, wrapped in a darray [nw, nq, Np, Np]):   <= 1e-8 + 2 max|Pi - Pi^T|/max|Pi|
 *       (the gate of V2(c): CoQui's PH-symmetric tau grid is exact only for (P,Q)-symmetric Pi)
 *   info: max|hp - hh| / max|hp|, max|Im hp| / max|hp| (exactly 0), min Re hp (>= 0 up to the fit noise).
 * [V6][extrap] q -> 0: the weights of head_extrapolation_t applied to the line heads at i nu_n vs CoQui's
 *   div_utils::extrapolate_eps_inv_q0 on the same data (the copy, <= 1e-13) and vs CoQui's extrapolated head
 *   mb_state.eps_inv_head (tau -> i nu, PH-symmetric) (<= 1e-8 + 2 r_apex); eps_inf line vs CoQui; info: the variants
 *   gygi / gygi_perdir / gygi_smallest_q / gygi_average.
 * [V0][gygi] F = V_H + Sigma_x - madelung D (exchange_head_correction) vs hf_t("gygi").evaluate(...) (S = 1):      <= 1e-7 Ha
 * [V3][gygi] Sigma_c + dSigma_head on the line, per-sector fits evaluated on the Matsubara axis (V3b), vs CoQui's iteration-1
 *   Sigma_c(i omega_n) with gw_t("gygi") + scr_coulomb_t("rpa", "gygi"):                              <= 1e-6 + 2 r_apex
 *   (dSigma_head enters the fits at the dense fermionic nodes, as in the driver; the sector gaps of the fits cover the head
 *   poles e_m +- nu_j). Head term alone: dSigma_line(i omega_n) vs Sigma_CoQui(gygi) - Sigma_CoQui(ignore_g0) (info + gate).
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include <cmath>
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
#include "methods/mb_state/mb_state.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_common.hpp"
#include "methods/HF/hf_t.h"
#include "methods/GW/gw_t.h"
#include "methods/GW/g0_div_utils.hpp"
#include "methods/scr_coulomb/scr_coulomb_t.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "hamiltonian/one_body_hamiltonian.hpp"

#include "numerics/line_dlr/time_ray.hpp"
#include "numerics/line_dlr/line_basis.hpp"
#include "numerics/line_dlr/bosonic_basis.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/polarization.hpp"
#include "methods/GW_line/screened.hpp"
#include "methods/GW_line/self_energy.hpp"
#include "methods/GW_line/static_part.hpp"
#include "methods/GW_line/head.hpp"
#include "gw_line_casida_ref.hpp"

namespace {

using namespace methods::gw_line;
using numerics::line_dlr::time_ray_t;
using numerics::line_dlr::sector_t;
using numerics::line_dlr::bosonic_basis_t;
using numerics::line_dlr::line_basis_t;
using namespace gw_line_test;

constexpr double deg = std::numbers::pi / 180.0;

struct hsetup_t {
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;
  std::shared_ptr<mf::MF> mf;
  std::unique_ptr<methods::thc_reader_t> thc;
  long nk = 0, nq = 0, nb = 0, Np = 0;
  nda::array<double, 2> eig, e_rel;
  double mu = 0, ks_gap = 0, etr_max = 0, e_lo = 0, e_hi = 0;
  pole_data_t poles;

  explicit hsetup_t(std::string const &fixture) {
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
    e_lo = 1e300; e_hi = -1e300;
    for (long ik = 0; ik < nk; ++ik)
      for (long n = 0; n < nb; ++n) {
        e_lo = std::min(e_lo, eig(ik, n)); e_hi = std::max(e_hi, eig(ik, n));
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

double maxabs(auto const &a) {
  double m = 0.0;
  for (auto const &v : a) m = std::max(m, std::abs(v));
  return m;
}

void run_head(std::string const &fixture, double v3_tol) {
  using namespace methods;
  hsetup_t su(fixture);
  auto &mpi  = *su.mpi;
  auto &comm = mpi.comm;
  auto all   = nda::range::all;
  auto &thc  = *su.thc;
  auto &mf   = *su.mf;
  const long nk = su.nk, nq = su.nq, nb = su.nb, Np = su.Np;
  const double madelung = mf.madelung();

  const double theta = 20.0 * deg, theta_t = 10.0 * deg;
  bosonic_basis_t basis(theta, std::max(4.0, 1.2 * su.etr_max), 1e-10, 0.5 * su.ks_gap);
  auto const &zeta = basis.zeta_nodes;
  const long nz = zeta.size(), r = basis.rank;
  aux_grid_t grid(mpi, Np);
  const long t_chunk = 8;
  auto ray_p = time_ray_t::for_spectrum(theta_t, su.poles.emin(), 36.0, 1e-5, 3.0, 16, sector_t::particle);
  auto ray_h = time_ray_t::for_spectrum(theta_t, su.poles.emin(), 36.0, 1e-5, 3.0, 16, sector_t::hole);
  auto fz    = numerics::line_dlr::dense_nodes(theta, 1e-3, 60.0, 120);
  const long nzf = fz.size();
  app_log(2, "\n[V6 head] {} ({} ranks, aux grid {}x{}): nk={} nq={} nb={} Np={} mu={:.6f}, bosonic rank {} ({} nodes), madelung {:.6f}",
          fixture, comm.size(), grid.np_P, grid.np_Q, nk, nq, nb, Np, su.mu, r, nz, madelung);

  // ---- line: Pi -> W (nodes + residues) -> heads
  utils::TimerManager Timer;
  propagator_t<HOST_MEMORY> prop(thc, grid);
  memory::array<HOST_MEMORY, ComplexType, 4> Pi, w, Wn;
  polarization<HOST_MEMORY>(prop, su.poles, mf, grid, zeta, ray_p, ray_h, t_chunk, Pi, Timer);
  dyson_layout_t lay(mpi, nq, nz, Np);
  coulomb_blocks_t<HOST_MEMORY> Zb(thc, grid, lay.q_rng(), Timer);
  screened_interaction<HOST_MEMORY>(Pi, Zb, basis, grid, mpi, w, Timer, &Wn);
  head_basis_t hb(thc, mf, grid);
  std::vector<long> qall(nq);
  for (long q = 0; q < nq; ++q) qall[q] = q;
  nda::array<ComplexType, 2> Hn, hp, hh;
  head_nodes_partial<HOST_MEMORY>(Wn, qall, hb, Hn);
  head_residues_partial<HOST_MEMORY>(w, qall, hb, hp, hh);
  head_reduce(comm, {&Hn, &hp, &hh});
  Wn = memory::array<HOST_MEMORY, ComplexType, 4>{};

  // ---- CoQui imaginary axis with gygi: W^c, eps_inv_head (extrapolated), Sigma_c (gygi and ignore_g0), HF (gygi)
  const double beta  = 1000.0;
  const double w_max = std::max(std::abs(su.e_lo), std::abs(su.e_hi)) + 2.0;
  imag_axes_ft::IAFT ft(beta, w_max + 1.0, imag_axes_ft::dlr_basis, "high");
  MBState mb_state(su.mpi, ft, std::string("coqui_gw_line_head_") + fixture);
  simple_dyson dyson(su.mf.get(), &ft);
  const long ns = 1;
  mb_state.sF_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(mpi, {ns, nk, nb, nb}));
  mb_state.sDm_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(mpi, {ns, nk, nb, nb}));
  mb_state.sG_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(mpi, {ft.nt_f(), ns, nk, nb, nb}));
  mb_state.sSigma_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(mpi, {ft.nt_f(), ns, nk, nb, nb}));
  auto &sF = mb_state.sF_skij.value();
  auto &sDm = mb_state.sDm_skij.value();
  auto &sG = mb_state.sG_tskij.value();
  auto &sSigma = mb_state.sSigma_tskij.value();
  hamilt::set_fock(mf, dyson.PSP(), sF, true);
  if (mpi.node_comm.root()) sSigma.local() = ComplexType(0.0);
  sSigma.communicator()->barrier();
  double mu_T = 0.0;
  update_G(dyson, mf, ft, sDm, sG, sF, sSigma, mu_T, false);
  {
    auto S = dyson.sS_skij().local();
    double dS = 0.0;
    for (long k = 0; k < nk; ++k)
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) dS = std::max(dS, std::abs(S(0, k, i, j) - ComplexType(i == j ? 1.0 : 0.0)));
    app_log(2, "  CoQui overlap of the KS band basis: max|S - 1| = {:.1e} (the exchange correction uses S = 1)", dS);
    REQUIRE(dS < 1e-8);
  }
  solvers::scr_coulomb_t scr_im(&ft, "rpa", "gygi");
  scr_im.update_w(mb_state, thc, -1);
  REQUIRE(mb_state.dW_qtPQ.has_value());
  REQUIRE(mb_state.eps_inv_head.has_value());
  nda::array<ComplexType, 1> eih_t(mb_state.eps_inv_head.value());
  const long nw_half = (ft.nw_b() % 2 == 0) ? ft.nw_b() / 2 : ft.nw_b() / 2 + 1;
  auto wn_b = ft.wn_mesh_b();
  nda::array<ComplexType, 1> zb(nw_half);
  for (long n = 0; n < nw_half; ++n) zb(n) = ComplexType(0.0, ft.omega(wn_b(ft.nw_b() / 2 + n)).imag());
  // CoQui's extrapolated head on the bosonic Matsubara half axis
  nda::array<ComplexType, 1> h0_coqui(nw_half);
  {
    nda::array<ComplexType, 2> t2(eih_t.size(), 1), w2(nw_half, 1);
    t2(all, 0) = eih_t;
    ft.tau_to_w_PHsym(t2, w2);
    h0_coqui = w2(all, 0);
  }
  // CoQui's W^c(q, i nu_n), gathered, wrapped in a darray [nw, nq, Np, Np] -> div_utils::eval_eps_inv_q
  nda::array<ComplexType, 2> hq_coqui;   // (nw_half, nq)
  {
    auto const &dW = mb_state.dW_qtPQ.value();
    const long nt_half = dW.global_shape()[1];
    nda::array<ComplexType, 4> Wq(nw_half, nq, Np, Np);
    for (long iq = 0; iq < nq; ++iq) {
      nda::array<ComplexType, 4> Wt(nt_half, 1, Np, Np), Ww(nw_half, 1, Np, Np);
      Wt() = 0.0;
      if (iq >= dW.local_range(0).first() and iq < dW.local_range(0).last())
        Wt(dW.local_range(1), 0, dW.local_range(2), dW.local_range(3)) = dW.local()(iq - dW.local_range(0).first(), all, all, all);
      comm.all_reduce_in_place_n(Wt.data(), Wt.size(), std::plus<>{});
      ft.tau_to_w_PHsym(Wt, Ww);
      Wq(all, iq, all, all) = Ww(all, 0, all, all);
    }
    const long npx = comm.size();
    utils::check(nw_half >= npx, "head test: too few bosonic nodes for the darray grid");
    auto dWq = math::nda::make_distributed_array<memory::array<HOST_MEMORY, ComplexType, 4>>(comm, {npx, 1, 1, 1},
                                                                                             {nw_half, nq, Np, Np});
    dWq.local() = Wq(dWq.local_range(0), all, all, all);
    hq_coqui = solvers::div_utils::eval_eps_inv_q(dWq, thc, mf);
  }

  // ---- V6 (a), (b): exact Casida heads; (c) vs CoQui
  auto rr = numerics::line_dlr::detail::logspace(0.05, 5.0, 6);
  nda::array<ComplexType, 1> z12(12);
  for (long i = 0; i < 6; ++i) {
    z12(i)     = rr(i) * std::exp(ComplexType(0.0, theta));
    z12(6 + i) = rr(i) * std::exp(ComplexType(0.0, std::numbers::pi - theta));
  }
  double ea = 0, sa = 0, eb = 0, sb = 0, ec = 0, sc = 0, a_pex = 0, s_pex = 0, dph = 0, sph = 0, imh = 0, minre = 1e300;
  nda::array<ComplexType, 2> hl_w(nw_half, nq);   // line heads at i nu_n
  for (long iq = 0; iq < nq; ++iq) {   // LOCKSTEP: thc.Z is collective
    cmat Zq(thc.Z(int(iq)));
    auto cas = casida_q(thc, mf, su.e_rel, iq, Zq);
    nda::array<ComplexType, 1> hpq(hp(iq, all)), hhq(hh(iq, all));
    hl_w(all, iq) = head_eval(hpq, hhq, basis.nu, zb);
    // exact Pi asymmetry (gate of V2(c))
    for (long n = 0; n < nw_half; n += std::max(1L, nw_half / 8)) {
      cmat SP(cas.S);
      for (long t = 0; t < cas.T; ++t) {
        const ComplexType f = cas.sg[t] / (zb(n) - cas.E[t]);
        for (long P = 0; P < Np; ++P) SP(P, t) *= f;
      }
      cmat Pex = SP * nda::dagger(cas.S);
      for (long P = 0; P < Np; ++P)
        for (long Q = 0; Q < Np; ++Q) {
          a_pex = std::max(a_pex, std::abs(Pex(P, Q) - Pex(Q, P)));
          s_pex = std::max(s_pex, std::abs(Pex(P, Q)));
        }
    }
    if (hb.fac(iq) == 0.0) continue;   // Gamma
    // exact head: f sum_s lam_s |cb^T v_s|^2 / (z - lam_s)
    nda::array<double, 1> g2(cas.T);
    auto cb = thc.basis_bar_head();
    for (long s = 0; s < cas.T; ++s) {
      ComplexType a(0.0);
      for (long P = 0; P < Np; ++P) a += cb(iq, P) * cas.V(P, s);
      g2(s) = std::norm(a);
    }
    auto hex = [&](nda::array<ComplexType, 1> const &z) {
      nda::array<ComplexType, 1> o(z.size());
      for (long i = 0; i < z.size(); ++i) {
        ComplexType a(0.0);
        for (long s = 0; s < cas.T; ++s) a += cas.lam(s) * g2(s) / (z(i) - cas.lam(s));
        o(i) = hb.fac(iq) * a;
      }
      return o;
    };
    auto hz = hex(zeta);
    ea = std::max(ea, maxabs(nda::array<ComplexType, 1>(Hn(iq, all) - hz)));
    sa = std::max(sa, maxabs(hz));
    for (auto const *zz : std::array<nda::array<ComplexType, 1> const *, 3>{&zeta, &z12, &zb}) {
      auto he = hex(*zz);
      auto hl = head_eval(hpq, hhq, basis.nu, *zz);
      eb = std::max(eb, maxabs(nda::array<ComplexType, 1>(hl - he)));
      sb = std::max(sb, maxabs(he));
    }
    ec = std::max(ec, maxabs(nda::array<ComplexType, 1>(hl_w(all, iq) - hq_coqui(all, iq))));
    sc = std::max(sc, maxabs(nda::array<ComplexType, 1>(hq_coqui(all, iq))));
    dph = std::max(dph, maxabs(nda::array<ComplexType, 1>(hpq - hhq)));
    sph = std::max(sph, maxabs(hpq));
    for (long j = 0; j < r; ++j) {
      imh   = std::max(imh, std::abs(hpq(j).imag()));
      minre = std::min(minre, hpq(j).real());
    }
  }
  a_pex = comm.all_reduce_value(a_pex, mpi3::max<>{});
  s_pex = comm.all_reduce_value(s_pex, mpi3::max<>{});
  const double r_apex = a_pex / s_pex;
  const double ra = ea / sa, rb = eb / sb, rc = ec / sc;
  app_log(2, "  [V6] (a) h(q, zeta_i) at the {} nodes (block-layout W) vs exact Casida head: rel {:.2e} (max|h| {:.4e})", nz, ra, sa);
  app_log(2, "  [V6] (b) h from the scalar residues hp, hh at nodes / 12 ray points / i nu_n vs exact: rel {:.2e}", rb);
  app_log(2, "  [V6] (c) line h(q, i nu_n) vs CoQui eval_eps_inv_q on its W^c(i nu_n), {} nodes: rel {:.2e} (max|h| {:.4e}); "
             "max|Pi - Pi^T|/max|Pi| {:.2e}",
          nw_half, rc, sc, r_apex);
  app_log(2, "  [V6] info: residues max|hp - hh|/max|hp| {:.2e}, max|Im hp|/max|hp| {:.2e}, min Re hp / max|hp| {:.2e} (rank {})",
          dph / sph, imh / sph, minre / sph, r);
  REQUIRE(ra <= 1e-10);
  REQUIRE(rb <= 1e-8);
  REQUIRE(rc <= 1e-8 + 2.0 * r_apex);

  // ---- extrapolation: the copy vs CoQui's function on the same data; vs CoQui's extrapolated head; variants
  head_extrapolation_t ex(mf, "gygi");
  ex.log(2);
  nda::array<ComplexType, 1> hp0 = ex.apply(hp), hh0 = ex.apply(hh);
  auto h0_line = head_eval(hp0, hh0, basis.nu, zb);
  auto h0_copy = solvers::div_utils::extrapolate_eps_inv_q0(hl_w, mf, "gygi");
  const double e_copy = maxabs(nda::array<ComplexType, 1>(h0_line - h0_copy)) / maxabs(h0_copy);
  const double e_cq   = maxabs(nda::array<ComplexType, 1>(h0_line - h0_coqui)) / maxabs(h0_coqui);
  const double eps_line = head_eps_inf(hp0, hh0, basis.nu), eps_coqui = 1.0 / (1.0 + h0_coqui(0).real());
  app_log(2, "  [V6][extrap] gygi: line weights vs CoQui extrapolate_eps_inv_q0 on the line heads {:.2e}; line h0(i nu_n) vs CoQui "
             "eps_inv_head (tau -> i nu) {:.2e}; eps_inf line {:.8f} CoQui {:.8f} (diff {:.1e})",
          e_copy, e_cq, eps_line, eps_coqui, eps_line - eps_coqui);
  for (std::string v : {"gygi_perdir", "gygi_smallest_q", "gygi_average", "gygi_order_1"}) {
    head_extrapolation_t ev(mf, v);
    auto h0v = head_eval(ev.apply(hp), ev.apply(hh), basis.nu, zb);
    auto h0c = solvers::div_utils::extrapolate_eps_inv_q0(hl_w, mf, v);
    app_log(2, "    variant {:<16s}: eps_inf {:.8f}, max|h0 - h0(gygi)| {:.2e}, copy vs CoQui {:.1e}", v,
            head_eps_inf(ev.apply(hp), ev.apply(hh), basis.nu), maxabs(nda::array<ComplexType, 1>(h0v - h0_line)),
            maxabs(nda::array<ComplexType, 1>(h0v - h0c)) / maxabs(h0c));
    REQUIRE(maxabs(nda::array<ComplexType, 1>(h0v - h0c)) <= 1e-13 * maxabs(h0c));
  }
  REQUIRE(e_copy <= 1e-13);
  REQUIRE(e_cq <= 1e-8 + 2.0 * r_apex);

  // ---- V0-gygi: exchange with the Madelung term vs hf_t("gygi")
  {
    auto D = density_matrix(su.poles);
    nda::array<ComplexType, 3> F;
    hartree_exchange<HOST_MEMORY>(prop, Zb, D, mf, grid, mpi, F, Timer);
    exchange_head_correction(F, D, madelung);
    solvers::hf_t hf("gygi");
    auto sFh = math::shm::make_shared_array<Array_view_4D_t>(mpi, {1, nk, nb, nb});
    nda::array<ComplexType, 4> Dm(1, nk, nb, nb), S(1, nk, nb, nb);
    Dm(0, all, all, all) = D;
    S() = ComplexType(0.0);
    for (long ik = 0; ik < nk; ++ik)
      for (long i = 0; i < nb; ++i) S(0, ik, i, i) = 1.0;
    hf.evaluate(sFh, Dm, thc, S, true, true);
    nda::array<ComplexType, 3> Fc(sFh.local()(0, all, all, all));
    const double e0 = max_diff3(F, Fc);
    app_log(2, "  [V0][gygi] F = V_H + Sigma_x - madelung D vs CoQui hf_t(gygi): max|diff| {:.2e} Ha (max|F| {:.4f}, madelung term "
             "{:.4f} Ha)",
            e0, max_abs3(Fc), madelung);
    REQUIRE(e0 <= 1e-7);
  }

  // ---- V3-gygi: Sigma_c + head term vs CoQui gw_t("gygi")
  auto coqui_sigma_w = [&](std::string const &div) {
    solvers::gw_t gw(&ft, div, std::string("coqui_gw_line_head_") + fixture + div);
    gw.evaluate<HOST_MEMORY>(mb_state, thc);
    const long nwf = ft.nw_f();
    nda::array<ComplexType, 5> Sig_w(nwf, ns, nk, nb, nb);
    nda::array<ComplexType, 5> Sig_t(sSigma.local());
    ft.tau_to_w(Sig_t, Sig_w, imag_axes_ft::fermion);
    return Sig_w;
  };
  auto Sw_gygi = coqui_sigma_w("gygi");
  auto Sw_ign  = coqui_sigma_w("ignore_g0");
  const long nwf = ft.nw_f();
  auto wn_f = ft.wn_mesh_f();
  nda::array<ComplexType, 1> zi(nwf), ziu(nwf);
  for (long n = 0; n < nwf; ++n) {
    zi(n)  = (mu_T - su.mu) + ft.omega(wn_f(n));
    ziu(n) = (zi(n).imag() > 0.0) ? zi(n) : std::conj(zi(n));
  }
  nda::array<ComplexType, 4> Sp, Sh;
  self_energy<HOST_MEMORY>(prop, su.poles, w, basis, mf, grid, mpi, fz, ray_p, ray_h, t_chunk, Sp, Timer, sector_t::particle);
  self_energy<HOST_MEMORY>(prop, su.poles, w, basis, mf, grid, mpi, fz, ray_p, ray_h, t_chunk, Sh, Timer, sector_t::hole);
  auto T = head_overlap_T(thc, gamma_index(mf));
  {
    double dT = 0.0;
    for (long k = 0; k < nk; ++k)
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) dT = std::max(dT, std::abs(T(k, i, j) - ComplexType(i == j ? 1.0 : 0.0)));
    app_log(2, "  [V3][gygi] T = X^dagger diag(conj chi_head(Gamma)) X: max|T - 1| = {:.2e} (THC overlap of the band basis)", dT);
  }
  std::vector<long> krows(nk);
  for (long k = 0; k < nk; ++k) krows[k] = k;
  nda::array<ComplexType, 4> dP(nk, nzf, nb, nb), dH(nk, nzf, nb, nb), dPm(nk, nwf, nb, nb), dHm(nk, nwf, nb, nb);
  dP() = 0; dH() = 0; dPm() = 0; dHm() = 0;
  head_sigma_correction(su.poles, T, basis.nu, hp0, hh0, madelung, fz, krows, dP, dH);
  head_sigma_correction(su.poles, T, basis.nu, hp0, hh0, madelung, zi, krows, dPm, dHm);
  nda::array<ComplexType, 4> Spt = Sp + dP, Sht = Sh + dH;
  // sector edges of Sigma_c + head (mu-relative): Casida edges are not needed; the fits need a gap below every pole:
  // Sigma_c: e_n(k-q) +- |lam_s| >= the smallest transition beyond the KS gap; head: e_m +- nu_j with h_j above the noise
  double hp_edge = 1e300, hh_edge = 1e300;
  const double hmax = std::max(maxabs(hp0), maxabs(hh0));
  for (long k = 0; k < nk; ++k)
    for (long n = 0; n < nb; ++n)
      for (long j = 0; j < r; ++j) {
        const double e = su.e_rel(k, n);
        if (e > 0 and std::abs(hp0(j)) > 1e-8 * hmax) hp_edge = std::min(hp_edge, e + basis.nu(j));
        if (e < 0 and std::abs(hh0(j)) > 1e-8 * hmax) hh_edge = std::min(hh_edge, -e + basis.nu(j));
      }
  double lam_min = 1e300;   // smallest RPA pole over q (Casida): the Sigma_c sector edges are |e| + lam_min at least
  for (long iq = 0; iq < nq; ++iq) {
    cmat Zq(thc.Z(int(iq)));
    auto cas = casida_q(thc, mf, su.e_rel, iq, Zq);
    for (long s = 0; s < cas.T; ++s) lam_min = std::min(lam_min, std::abs(cas.lam(s)));
  }
  double ep_min = 1e300, eh_min = 1e300;
  for (long k = 0; k < nk; ++k)
    for (long n = 0; n < nb; ++n) {
      if (su.e_rel(k, n) > 0) ep_min = std::min(ep_min, su.e_rel(k, n) + lam_min);
      else eh_min = std::min(eh_min, -su.e_rel(k, n) + lam_min);
    }
  const double gap_p = 0.8 * std::min(ep_min, hp_edge), gap_h = 0.8 * std::min(eh_min, hh_edge);
  line_basis_t bp(theta, 6.0, 1e-10, 6.0, gap_p, -1.0, 60.0), bh(theta, 6.0, 1e-10, gap_h, 6.0, -1.0, 60.0);
  nda::array<ComplexType, 4> Sl(nk, nwf, nb, nb), Sl0(nk, nwf, nb, nb), Sx(nk, nwf, nb, nb), Sc(nk, nwf, nb, nb),
      dSc(nk, nwf, nb, nb), dSl(nk, nwf, nb, nb);
  auto fit_eval = [&](nda::array<ComplexType, 3> const &sp, nda::array<ComplexType, 3> const &sh) {
    auto cp = bp.fit(fz, sp);
    auto ch = bh.fit(fz, sh);
    nda::array<ComplexType, 3> Su = bp.eval(cp, ziu) + bh.eval(ch, ziu), o(nwf, nb, nb);
    for (long n = 0; n < nwf; ++n)
      o(n, all, all) = (zi(n).imag() > 0.0) ? nda::array<ComplexType, 2>(Su(n, all, all))
                                            : nda::array<ComplexType, 2>(nda::dagger(Su(n, all, all)));
    return o;
  };
  for (long ik = 0; ik < nk; ++ik) {
    Sl(ik, all, all, all)  = fit_eval(nda::array<ComplexType, 3>(Spt(ik, all, all, all)), nda::array<ComplexType, 3>(Sht(ik, all, all, all)));
    Sl0(ik, all, all, all) = fit_eval(nda::array<ComplexType, 3>(Sp(ik, all, all, all)), nda::array<ComplexType, 3>(Sh(ik, all, all, all)));
    Sx(ik, all, all, all)  = Sl0(ik, all, all, all) + dPm(ik, all, all, all) + dHm(ik, all, all, all);
    Sc(ik, all, all, all)  = Sw_gygi(all, 0, ik, all, all);
    dSc(ik, all, all, all) = Sw_gygi(all, 0, ik, all, all) - Sw_ign(all, 0, ik, all, all);
    dSl(ik, all, all, all) = dPm(ik, all, all, all) + dHm(ik, all, all, all);
  }
  const double r_tot  = max_diff3(Sl, Sc) / max_abs3(Sc);
  const double r_totx = max_diff3(Sx, Sc) / max_abs3(Sc);
  const double r_head = max_diff3(dSl, dSc) / max_abs3(dSc);
  const double r_hrel = max_diff3(dSl, dSc) / max_abs3(Sc);
  app_log(2, "  [V3][gygi] {} fermionic Matsubara nodes; fit gaps {:.4f} / {:.4f} Ha (Sigma_c edges {:.4f} / {:.4f}, head edges "
             "{:.4f} / {:.4f}); h0 residues max|hp0 - hh0|/max {:.1e}, max|Im| {:.1e}",
          nwf, gap_p, gap_h, ep_min, eh_min, hp_edge, hh_edge, maxabs(nda::array<ComplexType, 1>(hp0 - hh0)) / hmax,
          std::max(maxabs(nda::array<double, 1>(nda::imag(hp0))), maxabs(nda::array<double, 1>(nda::imag(hh0)))) / hmax);
  app_log(2, "    Sigma_c + head (line, through the fits) vs CoQui gw_t(gygi): rel {:.2e} (max|Sigma_c| {:.4e}); head term exact at "
             "i w_n: {:.2e}",
          r_tot, max_abs3(Sc), r_totx);
  app_log(2, "    head term alone: line dSigma(i w_n) vs CoQui Sigma(gygi) - Sigma(ignore_g0): rel {:.2e} of max|dSigma| {:.4e} "
             "({:.2e} of max|Sigma_c|)",
          r_head, max_abs3(dSc), r_hrel);
  REQUIRE(r_tot <= v3_tol + 2.0 * r_apex);
  REQUIRE(r_totx <= v3_tol + 2.0 * r_apex);
  REQUIRE(r_hrel <= v3_tol + 2.0 * r_apex);
}

} // namespace

TEST_CASE("gw_line_head_lih222", "[gw_line][V6][head][gygi]") { run_head("qe_lih222", 1e-6); }

TEST_CASE("gw_line_head_si211", "[gw_line][V6][head][gygi]") { run_head("qe_si211", 1e-6); }

TEST_CASE("gw_line_head_lih223", "[gw_line][V6][head][gygi]") { run_head("qe_lih223", 1e-6); }
