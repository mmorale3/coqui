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
 * S5 of notes/line_gw_cpp_plan.md: self-energy and static part on the line, [V0] and [V3]
 * (python oracles: coqui/cayley/scripts/si222c_v012.py (V0), si222c_v3.py + cayley/casida_g0w0.py (V3)).
 *
 * Setup as [V2]: THC nIpts = 8 nbnd, KS poles, mu = KS gap midpoint, theta = 20 deg, theta_t = 10 deg, rays
 * for_spectrum(theta_t, emin, 36), bosonic basis lam_b = max(4, 1.2 x largest transition), gap = KS gap / 2, eps = 1e-10.
 *
 * [V0] F = V_H + Sigma_x (static_part.hpp, Eq. hf) from the KS density matrix (occupied projector in the band basis) vs
 *      CoQui's hf_t("ignore_g0").evaluate(sF, Dm, thc, S = 1, hartree, exchange) on the same thc_reader with the same Dm.
 *      hf_t::evaluate zeroes sF and writes ONLY the two-body part V_H + Sigma_x (no H0; hamilt::set_fock writes the
 *      mean-field V_H + V_xc and is not used here), so the two-body parts are compared: <= 1e-7 Ha max abs.
 * [V3a] Sigma_c(k, zeta) per sector at the dense fermionic nodes dense_nodes(20 deg, 1e-3, 60, 120) vs the EXACT pole sum
 *      from the Casida solution of [V2] (python casida_g0w0.py at T = 0: c_T = -1 particle, +1 hole, mixed terms 0):
 *        Sigma^>_ab(k,z) = +(1/N_k) sum_q sum_{n: e_n(k-q) > 0} sum_{s: lam_s > 0} lam_s A_as conj(A_bs) / (z - e_n(k-q) - lam_s)
 *        Sigma^<_ab(k,z) = -(1/N_k) sum_q sum_{n: e_n(k-q) < 0} sum_{s: lam_s < 0} lam_s A_as conj(A_bs) / (z - e_n(k-q) - lam_s)
 *      with A_as = sum_P conj(X_Pa(k)) X_Pn(k-q) v_Ps(q), i.e. [X(k)^dagger (R~_n(k-q) o lam_s v_s v_s^dagger) X(k)]_ab with
 *      R~_n = X(k-q) e_n e_n^T X(k-q)^dagger (KS residues):  <= 1e-6 relative (max norm), per sector and total.
 * [V3b] per-sector real-pole fits on one-sided fermionic bases (line_basis_t(20 deg, lam 6, eps 1e-10), particle gap
 *      (6, gap_p), hole gap (gap_h, 6), tmax 60, as driver.py; gap_p/h = 0.8 x the exact sector edges of Sigma_c, as
 *      si222c_v3.py) evaluated at zeta = (mu_coqui - mu_line) + i omega_n on all
 *      fermionic DLR nodes (Im zeta < 0 by Schwarz reflection Sigma(conj z) = Sigma(z)^dagger) vs CoQui's iteration-1 Sigma_c(i omega_n): gw_t("ignore_g0").evaluate after
 *      scr_coulomb_t("rpa", "ignore_g0").update_w with the KS G (beta = 1000, DLR "high"), Sigma(tau) -> tau_to_w(fermion).
 *      CoQui's sSigma_tskij is the correlation part only (-G W^c with W^c = (1 - Z Pi)^{-1} Z - Z; ignore_g0: no head term).
 *      Gate: <= 1e-6 + 2 max|Pi - Pi^T| / max|Pi| (exact Pi at the bosonic Matsubara nodes; CoQui's PH-symmetric half tau
 *      grid is exact only for (P,Q)-symmetric Pi, see the S4 log); decomposition gate: line fit vs the exact Casida
 *      Sigma_c at the same points <= 1e-6.
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
#include "methods/scr_coulomb/scr_coulomb_t.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
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
#include "gw_line_casida_ref.hpp"

namespace {

using namespace methods::gw_line;
using numerics::line_dlr::time_ray_t;
using numerics::line_dlr::sector_t;
using numerics::line_dlr::bosonic_basis_t;
using numerics::line_dlr::line_basis_t;
using namespace gw_line_test;

/// fixture + THC + KS spectrum (as [V2])
struct setup_t {
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;
  std::shared_ptr<mf::MF> mf;
  std::unique_ptr<methods::thc_reader_t> thc;
  long nk = 0, nq = 0, nb = 0, Np = 0;
  nda::array<double, 2> eig, e_rel;
  double mu = 0, ks_gap = 0, etr_max = 0, e_lo = 0, e_hi = 0;
  pole_data_t poles;

  setup_t(std::string const &fixture, long nI_factor) {
    mpi = utils::make_unit_test_mpi_context();
    mf  = std::make_shared<mf::MF>(mf::default_MF(mpi, fixture));
    utils::check(mf->nkpts() == mf->nkpts_ibz() and mf->nqpts() == mf->nqpts_ibz(), "{}: nosym fixture required", fixture);
    utils::check(mf->nspin() == 1 and mf->npol() == 1, "{}: spin-restricted collinear fixture required", fixture);
    thc = std::make_unique<methods::thc_reader_t>(
        mf, methods::make_thc_reader_ptree(mf->nbnd() * nI_factor, "", "incore", "", "bdft", 1e-10, mf->ecutrho(), 1, 1024));
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
    utils::check(lumo > homo, "{}: no KS gap", fixture);
    mu = 0.5 * (homo + lumo); ks_gap = lumo - homo; etr_max = emax_vir - emin_occ;
    poles = pole_data_t::from_ks(eig, mu);
    e_rel = nda::array<double, 2>(nk, nb);
    for (long ik = 0; ik < nk; ++ik)
      for (long n = 0; n < nb; ++n) e_rel(ik, n) = eig(ik, n) - mu;
  }
};

constexpr double deg = std::numbers::pi / 180.0;

// ------------------------------------------------------------------------------------------------------------------ V0
void run_v0(std::string const &fixture) {
  using namespace methods;
  setup_t su(fixture, 8);
  auto &mpi = *su.mpi;
  auto &comm = mpi.comm;
  auto all = nda::range::all;
  const long nk = su.nk, nb = su.nb, Np = su.Np;

  aux_grid_t grid(mpi, Np);
  utils::TimerManager Timer;
  propagator_t<HOST_MEMORY> prop(*su.thc, grid);
  coulomb_blocks_t<HOST_MEMORY> Zb(*su.thc, grid, nda::range(0, 0), Timer);
  auto D = density_matrix(su.poles);
  nda::array<ComplexType, 3> F;
  Timer.add("line_HX");
  Timer.start("line_HX");
  hartree_exchange<HOST_MEMORY>(prop, Zb, D, *su.mf, grid, mpi, F, Timer);
  Timer.stop("line_HX");

  // D is the occupied projector of the band basis
  double dD = 0.0;
  for (long ik = 0; ik < nk; ++ik)
    for (long i = 0; i < nb; ++i)
      for (long j = 0; j < nb; ++j)
        dD = std::max(dD, std::abs(D(ik, i, j) - ComplexType((i == j and su.e_rel(ik, i) < 0.0) ? 1.0 : 0.0)));
  REQUIRE(dD == 0.0);

  // CoQui: two-body Fock matrix V_H + Sigma_x of the same D (ignore_g0, S = 1)
  solvers::hf_t hf("ignore_g0");
  auto sF = math::shm::make_shared_array<Array_view_4D_t>(mpi, {1, nk, nb, nb});
  nda::array<ComplexType, 4> Dm(1, nk, nb, nb), S(1, nk, nb, nb);
  Dm(0, all, all, all) = D;
  S() = ComplexType(0.0);
  for (long ik = 0; ik < nk; ++ik)
    for (long i = 0; i < nb; ++i) S(0, ik, i, i) = 1.0;
  Timer.add("coqui_HF");
  Timer.start("coqui_HF");
  hf.evaluate(sF, Dm, *su.thc, S, true, true);
  Timer.stop("coqui_HF");
  nda::array<ComplexType, 3> Fc(sF.local()(0, all, all, all));
  // also the Hartree part alone (diagnostic)
  hf.evaluate(sF, Dm, *su.thc, S, true, false);
  nda::array<ComplexType, 3> Hc(sF.local()(0, all, all, all));

  double err = 0.0, mx = 0.0, herm = 0.0, mxh = 0.0;
  for (long ik = 0; ik < nk; ++ik)
    for (long i = 0; i < nb; ++i)
      for (long j = 0; j < nb; ++j) {
        err  = std::max(err, std::abs(F(ik, i, j) - Fc(ik, i, j)));
        mx   = std::max(mx, std::abs(Fc(ik, i, j)));
        mxh  = std::max(mxh, std::abs(Hc(ik, i, j)));
        herm = std::max(herm, std::abs(F(ik, i, j) - std::conj(F(ik, j, i))));
      }
  app_log(2, "\n[V0] {} ({} ranks, aux grid {}x{}): nk={} nb={} Np={} mu={:.6f}", fixture, comm.size(), grid.np_P, grid.np_Q, nk,
          nb, Np, su.mu);
  app_log(2, "  [V0] F = V_H + Sigma_x (KS density) vs CoQui hf_t(ignore_g0) two-body F: max|diff| {:.2e} Ha (rel {:.1e}); "
             "max|F| {:.4f}, max|V_H| {:.4f} Ha; max|F - F^dagger| {:.1e}",
          err, err / mx, mx, mxh, herm);
  app_log(2, "  [V0] timers (rank 0): line V_H + Sigma_x {:.3f} s (density {:.3f} hartree {:.3f} exchange {:.3f} allreduce {:.3f}); "
             "CoQui HF {:.3f} s",
          Timer.elapsed("line_HX"), Timer.elapsed("HX_density"), Timer.elapsed("HX_hartree"), Timer.elapsed("HX_exchange"),
          Timer.elapsed("HX_allreduce"), Timer.elapsed("coqui_HF"));
  REQUIRE(err <= 1e-7);
  REQUIRE(herm <= 1e-10);

  // block offsets in both P and Q (1 rank): the four blocks of a virtual 2x2 grid, each on a size-1 communicator, summed
  if (comm.size() == 1) {
    nda::array<ComplexType, 3> Fs(nk, nb, nb);
    Fs() = 0.0;
    for (long vr = 0; vr < 4; ++vr) {
      aux_grid_t gv(4, vr, Np);
      propagator_t<HOST_MEMORY> pv(*su.thc, gv);
      utils::TimerManager Tv;
      coulomb_blocks_t<HOST_MEMORY> Zv(*su.thc, gv, nda::range(0, 0), Tv);
      nda::array<ComplexType, 3> Fv;
      hartree_exchange<HOST_MEMORY>(pv, Zv, D, *su.mf, gv, comm, Fv, Tv);
      Fs += Fv;
    }
    const double ev = max_diff3(Fs, F);
    app_log(2, "  [V0] virtual 2x2 grid (sum of the four block contributions) vs 1x1: max|diff| {:.2e} Ha", ev);
    REQUIRE(ev <= 1e-13);
  }
}

// ------------------------------------------------------------------------------------------------------------------ V3
void run_v3(std::string const &fixture, long nodes_per_ray, bool virtual_grid) {
  using namespace methods;
  setup_t su(fixture, 8);
  auto &mpi  = *su.mpi;
  auto &comm = mpi.comm;
  auto all   = nda::range::all;
  auto &thc  = *su.thc;
  auto &mf   = *su.mf;
  const long nk = su.nk, nq = su.nq, nb = su.nb, Np = su.Np;
  auto qk = mf.qk_to_k2();

  const double theta = 20.0 * deg, theta_t = 10.0 * deg;
  const double lam_b = std::max(4.0, 1.2 * su.etr_max), gap_b = 0.5 * su.ks_gap;
  bosonic_basis_t basis(theta, lam_b, 1e-10, gap_b);
  const long nzb = basis.zeta_nodes.size(), r = basis.rank;
  auto fz       = numerics::line_dlr::dense_nodes(theta, 1e-3, 60.0, nodes_per_ray);
  const long nz = fz.size();
  aux_grid_t grid(mpi, Np);
  const long t_chunk = 8;
  app_log(2, "\n[V3] {} ({} ranks): nk={} nq={} nb={} Np={} (nIpts = 8 nbnd), mu={:.6f} Ha, KS gap {:.4f} Ha; bosonic rank {} "
             "({} nodes); fermionic nodes {} ({} per ray, 1e-3..60 Ha)",
          fixture, comm.size(), nk, nq, nb, Np, su.mu, su.ks_gap, r, nzb, nz, nodes_per_ray);
  grid.log(nk, nq, nzb, r, t_chunk, nb);

  // ---- line: Pi -> W -> Sigma per sector
  auto ray_p = time_ray_t::for_spectrum(theta_t, su.poles.emin(), 36.0, 1e-5, 3.0, 16, sector_t::particle);
  auto ray_h = time_ray_t::for_spectrum(theta_t, su.poles.emin(), 36.0, 1e-5, 3.0, 16, sector_t::hole);
  utils::TimerManager Timer;
  for (auto nm : {"line_Pi", "line_W", "line_Sigma"}) Timer.add(nm);
  propagator_t<HOST_MEMORY> prop(thc, grid);
  memory::array<HOST_MEMORY, ComplexType, 4> Pi, w;
  Timer.start("line_Pi");
  polarization<HOST_MEMORY>(prop, su.poles, mf, grid, basis.zeta_nodes, ray_p, ray_h, t_chunk, Pi, Timer);
  Timer.stop("line_Pi");
  Timer.start("line_W");
  dyson_layout_t lay(mpi, nq, nzb, Np);
  coulomb_blocks_t<HOST_MEMORY> Zb(thc, grid, lay.q_rng(), Timer);
  screened_interaction<HOST_MEMORY>(Pi, Zb, basis, grid, mpi, w, Timer);
  Timer.stop("line_W");
  nda::array<ComplexType, 4> Sp, Sh;
  Timer.start("line_Sigma");
  self_energy<HOST_MEMORY>(prop, su.poles, w, basis, mf, grid, mpi, fz, ray_p, ray_h, t_chunk, Sp, Timer, sector_t::particle);
  self_energy<HOST_MEMORY>(prop, su.poles, w, basis, mf, grid, mpi, fz, ray_p, ray_h, t_chunk, Sh, Timer, sector_t::hole);
  Timer.stop("line_Sigma");

  // ---- CoQui iteration-1 Sigma_c(i omega_n) (harness of SCF/tests/test_methods_tc_contour.cpp, as [V2])
  const double beta  = 1000.0;
  const double w_max = std::max(std::abs(su.e_lo), std::abs(su.e_hi)) + 2.0;
  imag_axes_ft::IAFT ft(beta, w_max + 1.0, imag_axes_ft::dlr_basis, "high");
  MBState mb_state(su.mpi, ft, std::string("coqui_gw_line_v3_") + fixture);
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
  Timer.add("coqui_Sigma");
  Timer.start("coqui_Sigma");
  update_G(dyson, mf, ft, sDm, sG, sF, sSigma, mu_T, false);
  solvers::scr_coulomb_t scr_im(&ft, "rpa", "ignore_g0");
  scr_im.update_w(mb_state, thc, -1);
  solvers::gw_t gw(&ft, "ignore_g0", std::string("coqui_gw_line_v3_") + fixture);
  gw.evaluate<HOST_MEMORY>(mb_state, thc);
  Timer.stop("coqui_Sigma");
  const long nwf = ft.nw_f();
  nda::array<ComplexType, 5> Sig_w(nwf, ns, nk, nb, nb);
  {
    nda::array<ComplexType, 5> Sig_t(sSigma.local());
    ft.tau_to_w(Sig_t, Sig_w, imag_axes_ft::fermion);
  }
  auto wn_f = ft.wn_mesh_f();
  nda::array<ComplexType, 1> zi(nwf);
  for (long n = 0; n < nwf; ++n) zi(n) = (mu_T - su.mu) + ft.omega(wn_f(n));
  // bosonic Matsubara nodes (Pi asymmetry gate)
  const long nw_half = (ft.nw_b() % 2 == 0) ? ft.nw_b() / 2 : ft.nw_b() / 2 + 1;
  auto wn_b = ft.wn_mesh_b();
  nda::array<ComplexType, 1> zb(nw_half);
  for (long n = 0; n < nw_half; ++n) zb(n) = ComplexType(0.0, ft.omega(wn_b(ft.nw_b() / 2 + n)).imag());

  // ---- exact Casida Sigma_c per sector at the dense nodes and the Matsubara points (k distributed over ranks)
  const long nzr = nz + nwf;
  nda::array<ComplexType, 1> zr(nzr);
  zr(nda::range(0, nz)) = fz;
  zr(nda::range(nz, nzr)) = zi;
  nda::array<ComplexType, 4> Rp(nk, nzr, nb, nb), Rh(nk, nzr, nb, nb);
  Rp() = 0.0;
  Rh() = 0.0;
  double a_pex = 0.0, s_pex = 0.0;
  double Ep_min = 1e300, Eh_max = -1e300;   // edges of the exact Sigma_c sectors (mu-relative)
  long Tmax = 0;
  Timer.add("reference");
  Timer.start("reference");
  for (long iq = 0; iq < nq; ++iq) {   // LOCKSTEP: thc.Z is collective
    cmat Zq(thc.Z(int(iq)));
    auto cas = casida_q(thc, mf, su.e_rel, iq, Zq);
    Tmax = std::max(Tmax, cas.T);
    // exact Pi(q, i nu_n) asymmetry (only the rank owning q's turn computes; cheap on the fixtures)
    if (iq % comm.size() == comm.rank()) {
      for (long n = 0; n < nw_half; ++n) {
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
    }
    for (long ik = comm.rank(); ik < nk; ik += comm.size()) {
      const long ikmq = qk(iq, ik);
      auto Xk = thc.X(0, 0, ik);
      auto Xm = thc.X(0, 0, ikmq);
      cmat XkH = nda::dagger(cmat(Xk));
      for (int sec = 0; sec < 2; ++sec) {   // 0: particle, 1: hole
        std::vector<long> ns_, ss_;
        for (long n = 0; n < nb; ++n)
          if ((sec == 0) == (su.e_rel(ikmq, n) > 0.0)) ns_.push_back(n);
        for (long s = 0; s < cas.T; ++s)
          if ((sec == 0) == (cas.lam(s) > 0.0)) ss_.push_back(s);
        const long Nn = ns_.size(), Ns = ss_.size(), npl = Nn * Ns;
        if (npl == 0) continue;
        nda::array<ComplexType, 2> K(nzr, npl), B(npl, nb * nb);
        cmat Y(Np, Ns), A(nb, Ns);
        const double sgn = (sec == 0) ? 1.0 : -1.0;
        for (long in = 0; in < Nn; ++in) {
          const long n = ns_[in];
          for (long P = 0; P < Np; ++P)
            for (long js = 0; js < Ns; ++js) Y(P, js) = Xm(P, n) * cas.V(P, ss_[js]);
          A = XkH * Y;
          for (long js = 0; js < Ns; ++js) {
            const long p = in * Ns + js;
            const double lam = cas.lam(ss_[js]), E = su.e_rel(ikmq, n) + lam, wgt = sgn * lam / double(nk);
            if (sec == 0) Ep_min = std::min(Ep_min, E);
            else Eh_max = std::max(Eh_max, E);
            for (long a = 0; a < nb; ++a)
              for (long b = 0; b < nb; ++b) B(p, a * nb + b) = wgt * A(a, js) * std::conj(A(b, js));
            for (long z = 0; z < nzr; ++z) K(z, p) = 1.0 / (zr(z) - E);
          }
        }
        auto R2 = nda::reshape((sec == 0 ? Rp : Rh)(ik, all, all, all), std::array<long, 2>{nzr, nb * nb});
        nda::blas::gemm(ComplexType(1.0), K, B, ComplexType(1.0), R2);
      }
    }
  }
  comm.all_reduce_in_place_n(Rp.data(), Rp.size(), std::plus<>{});
  comm.all_reduce_in_place_n(Rh.data(), Rh.size(), std::plus<>{});
  a_pex = comm.all_reduce_value(a_pex, mpi3::max<>{});
  s_pex = comm.all_reduce_value(s_pex, mpi3::max<>{});
  Ep_min = comm.all_reduce_value(Ep_min, mpi3::min<>{});
  Eh_max = comm.all_reduce_value(Eh_max, mpi3::max<>{});
  REQUIRE(Ep_min > 0.0);
  REQUIRE(Eh_max < 0.0);
  Timer.stop("reference");
  const double r_apex = a_pex / s_pex;

  // ---- V3a: line Sigma at the dense nodes vs exact, per sector and total
  auto nodes = nda::range(0, nz), mats = nda::range(nz, nzr);
  nda::array<ComplexType, 4> Rp_n(Rp(all, nodes, all, all)), Rh_n(Rh(all, nodes, all, all));
  nda::array<ComplexType, 4> St = Sp + Sh, Rt_n = Rp_n + Rh_n;
  const double rp = max_diff3(Sp, Rp_n) / max_abs3(Rp_n), rh = max_diff3(Sh, Rh_n) / max_abs3(Rh_n),
               rt = max_diff3(St, Rt_n) / max_abs3(Rt_n);

  // ---- V3b: per-sector fits on the one-sided bases, evaluated on the Matsubara axis. One-sided gaps at 0.8 x the sector
  //      edges of Sigma_c (python si222c_v3.py): the axis points (mu_coqui - mu_line) + i omega_n sit inside the gap at
  //      Im = pi/beta, so candidate poles must not reach into it (with the driver's fixed 0.02 Ha the extrapolation to
  //      Re zeta = 0.021, Im zeta = 0.003 picks up 1e-4 from spurious near-threshold coefficients on lih222).
  const double gap_p = 0.8 * Ep_min, gap_h = -0.8 * Eh_max;
  utils::check(std::abs(mu_T - su.mu) < 0.5 * std::min(gap_p, gap_h), "[V3] CoQui mu outside the Sigma_c gap window");
  line_basis_t bp(theta, 6.0, 1e-10, 6.0, gap_p, -1.0, 60.0), bh(theta, 6.0, 1e-10, gap_h, 6.0, -1.0, 60.0);
  nda::array<ComplexType, 4> Sl(nk, nwf, nb, nb), Sc(nk, nwf, nb, nb), Rt_m(nk, nwf, nb, nb);
  double res_fit = 0.0;
  nda::array<ComplexType, 1> ziu(nwf);   // the upper-half-plane images of the Matsubara points
  for (long n = 0; n < nwf; ++n) ziu(n) = (zi(n).imag() > 0.0) ? zi(n) : std::conj(zi(n));
  for (long ik = 0; ik < nk; ++ik) {
    nda::array<ComplexType, 3> sp(Sp(ik, all, all, all)), sh(Sh(ik, all, all, all));
    auto cp = bp.fit(fz, sp);
    auto ch = bh.fit(fz, sh);
    nda::array<ComplexType, 3> refit = bp.eval(cp, fz) + bh.eval(ch, fz);
    nda::array<ComplexType, 3> stk(St(ik, all, all, all));
    res_fit = std::max(res_fit, max_diff3(refit, stk) / max_abs3(St));
    // the line representation lives on the UPPER half plane (both rays); the lower half by Schwarz reflection,
    // Sigma(conj z) = Sigma(z)^dagger (notes section 2). Direct evaluation of the pole sum at Im z < 0 is NOT used: the fit
    // only constrains the upper half plane, and its non-unique near-threshold coefficients give ~3e-7 there (lih222).
    nda::array<ComplexType, 3> Su = bp.eval(cp, ziu) + bh.eval(ch, ziu);
    for (long n = 0; n < nwf; ++n)
      Sl(ik, n, all, all) = (zi(n).imag() > 0.0) ? nda::array<ComplexType, 2>(Su(n, all, all))
                                                 : nda::array<ComplexType, 2>(nda::dagger(Su(n, all, all)));
    Sc(ik, all, all, all) = Sig_w(all, 0, ik, all, all);
    Rt_m(ik, all, all, all) = Rp(ik, mats, all, all) + Rh(ik, mats, all, all);
  }
  const double r_lc = max_diff3(Sl, Sc) / max_abs3(Sc);       // line vs CoQui (the V3b number)
  const double r_le = max_diff3(Sl, Rt_m) / max_abs3(Rt_m);   // line vs exact on the axis
  const double r_ce = max_diff3(Sc, Rt_m) / max_abs3(Rt_m);   // CoQui vs exact on the axis

  app_log(2, "  [V3] {} ({} ranks, aux grid {}x{}, Dyson pools {}x{}), rays {} + {} nodes, Casida dim <= {}:", fixture,
          comm.size(), grid.np_P, grid.np_Q, lay.np_q, lay.np_z, ray_p.size(), ray_h.size(), Tmax);
  app_log(2, "    (a) Sigma_c at the {} dense nodes vs exact Casida: particle {:.2e} (max|S^>| {:.3e}), hole {:.2e} (max|S^<| {:.3e}), "
             "total {:.2e}",
          nz, rp, max_abs3(Rp_n), rh, max_abs3(Rh_n), rt);
  app_log(2, "    (b) fits: Sigma_c sector edges {:.4f} / {:.4f} Ha -> one-sided gaps {:.4f} / {:.4f}; particle basis rank {}, hole "
             "basis rank {}; refit residual at the nodes {:.2e}; mu_coqui - mu_line = {:.3e} Ha",
          Ep_min, Eh_max, gap_p, gap_h, bp.rank, bh.rank, res_fit, mu_T - su.mu);
  app_log(2, "        on {} fermionic Matsubara nodes (beta {}, w_max {:.2e}): line vs CoQui Sigma_c {:.2e} (max|Sigma_c| {:.3e}); "
             "line vs exact {:.2e}; CoQui vs exact {:.2e}; max|Pi - Pi^T|/max|Pi| (exact, bosonic nodes) {:.2e}",
          nwf, beta, zi(nwf - 1).imag(), r_lc, max_abs3(Sc), r_le, r_ce, r_apex);
  app_log(2, "  [V3] {} timers (rank 0): Pi {:.2f} s (G_tilde {:.2f} hadamard {:.2f} transform {:.2f}) | W {:.2f} s | Sigma {:.2f} s:", fixture,
          Timer.elapsed("line_Pi"), Timer.elapsed("G_tilde"), Timer.elapsed("Pi_hadamard"), Timer.elapsed("Pi_transform"),
          Timer.elapsed("line_W"), Timer.elapsed("line_Sigma"));
  app_log(2, "       Sigma_G_tilde {:.3f}  Sigma_W_time {:.3f}  Sigma_hadamard {:.3f}  Sigma_contract {:.3f}  Sigma_allreduce {:.3f}  "
             "Sigma_transform {:.3f} | CoQui (G, W, Sigma) {:.2f} s | Casida reference {:.2f} s",
          Timer.elapsed("Sigma_G_tilde"), Timer.elapsed("Sigma_W_time"), Timer.elapsed("Sigma_hadamard"),
          Timer.elapsed("Sigma_contract"), Timer.elapsed("Sigma_allreduce"), Timer.elapsed("Sigma_transform"),
          Timer.elapsed("coqui_Sigma"), Timer.elapsed("reference"));
  // block offsets in both P and Q (1 rank, si211): the four blocks of a virtual 2x2 grid on a size-1 communicator, with
  // the residues sliced from the 1x1 run, summed, vs the 1x1 result (both sectors in one call)
  if (comm.size() == 1 and virtual_grid) {
    nda::array<ComplexType, 4> Ss(nk, nz, nb, nb);
    Ss() = 0.0;
    for (long vr = 0; vr < 4; ++vr) {
      aux_grid_t gv(4, vr, Np);
      propagator_t<HOST_MEMORY> pv(thc, gv);
      memory::array<HOST_MEMORY, ComplexType, 4> wv(w(all, all, gv.P_rng(), gv.Q_rng()));
      utils::TimerManager Tv;
      nda::array<ComplexType, 4> Sv;
      self_energy<HOST_MEMORY>(pv, su.poles, wv, basis, mf, gv, comm, fz, ray_p, ray_h, t_chunk, Sv, Tv);
      Ss += Sv;
    }
    const double ev = max_diff3(Ss, St) / max_abs3(St);
    app_log(2, "  [V3] {}: virtual 2x2 grid (sum of the four blocks, both sectors) vs 1x1: rel {:.2e}", fixture, ev);
    REQUIRE(ev <= 1e-13);
  }
  REQUIRE(rp <= 1e-6);
  REQUIRE(rh <= 1e-6);
  REQUIRE(rt <= 1e-6);
  REQUIRE(r_le <= 1e-6);
  REQUIRE(r_lc <= 1e-6 + 2.0 * r_apex);
}

} // namespace

TEST_CASE("gw_line_V0_lih222", "[gw_line][V0]") { run_v0("qe_lih222"); }

TEST_CASE("gw_line_V0_si211", "[gw_line][V0]") { run_v0("qe_si211"); }

TEST_CASE("gw_line_V3_lih222", "[gw_line][V3]") { run_v3("qe_lih222", 120, false); }

TEST_CASE("gw_line_V3_si211", "[gw_line][V3]") { run_v3("qe_si211", 120, true); }
