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
 * S4 of notes/line_gw_cpp_plan.md: screened interaction on the line, [V2] (python oracle: coqui/cayley/scripts/si222c_v012.py).
 *
 * KS poles (mu mid-gap) -> Pi(q, zeta_i) at the bosonic line nodes (S3 polarization) -> screened_interaction
 * (redistribute to whole matrices, Dyson Eq. dyson_w, redistribute back, symmetric fit Eq. bfit). Checks, all q:
 *   (a) W at the nodes vs the Dyson of the gathered full Pi done here with nda::inverse:            <= 1e-12
 *   (b) residues vs bosonic_basis_t::fit (gelss) of the gathered full W at the nodes, compared as pole sums on the nodes,
 *       the 12 ray points and the Matsubara nodes (the raw residues are not unique at ~1e-5, see below):  <= 1e-10
 *       (q != -q: the paired fit / evaluation fit(zeta, W(q), W(-q)), eval(w(q), w(-q), z), notes section 3.3)
 *   (c) the fit evaluated at zeta = i nu_n vs CoQui's own W^c(q, i nu_n) (scr_coulomb_t "rpa", "ignore_g0", KS G at
 *       beta = 1000, DLR IAFT; dW_qtPQ -> tau_to_w_PHsym) on all bosonic Matsubara nodes of the IAFT grid: <= 1e-8 plus
 *       2 max|Pi - Pi^T| / max|Pi| of the exact Pi(q, i nu) (CoQui's PH-symmetric half tau grid treats Pi(beta - tau) =
 *       Pi(tau) elementwise, exact only for (P,Q)-symmetric Pi; si211: 2.7e-5, lih222: 7e-9). Decomposition gates: line vs
 *       exact Dyson(Pi(i nu_n)) <= 1e-8; CoQui W^c = Dyson(CoQui Pi) <= 1e-10; symmetric parts of CoQui Pi vs exact <= 1e-8
 *   (d) W^>(zeta) = sum_j w_j/(zeta - nu_j) vs the positive-pole part of the exact Casida W_dyn at 12 mu-relative points
 *       on the upper rays:                                                                           <= 1e-8
 *   (e) W^<(q, zeta)^T = -sum_j w_j(-q)/(zeta + nu_j) (eval_poles hole, transposed: the residues of the PARTNER -q) vs the
 *       transposed negative-pole part of the exact Casida W_dyn(q) at the 12 points:                <= 1e-8
 *   [V2-sym] exact Casida W_dyn(q, -z) = W_dyn(-q, z)^T and Z(-q) = Z(q)^T (info: vs the pre-fix W(q, -z) = W(q, z)^T)
 * plus (info + loose gate) full W at the nodes vs Casida W_dyn and the time-domain convention of w_time
 * (ray transform of W^>(t) and W^<(t)^T reproduces eval_poles).
 *
 * Casida (si_pipeline.py::casida_q, Hermitian form): S_Pt = sqrt(2/N_k) X_Pn(k) conj(X_Pm(k-q)), E_t = e_m(k-q) - e_n(k),
 * s_t = +1 (n occupied) / -1, K = S^dagger Z S, H = diag(E) + diag(s) K = eta M with M = diag(|E|) + K > 0, eta = diag(s).
 * With C = M^{1/2} eta M^{1/2} = U diag(lambda) U^dagger:  (z - H)^{-1} eta = M^{-1/2} U diag(lambda/(z - lambda)) U^dagger M^{-1/2},
 * so W_dyn(z) = Z Pi_RPA(z) Z = sum_s lambda_s v_s v_s^dagger / (z - lambda_s), v = Z S M^{-1/2} U   (= python's alpha bet).
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include <cmath>
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
#include "nda/linalg/eigenelements.hpp"

#include "mean_field/default_MF.hpp"
#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/mb_state/mb_state.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_common.hpp"
#include "methods/scr_coulomb/scr_coulomb_t.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "hamiltonian/one_body_hamiltonian.hpp"

#include "numerics/line_dlr/time_ray.hpp"
#include "numerics/line_dlr/bosonic_basis.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/polarization.hpp"
#include "methods/GW_line/screened.hpp"
#include "gw_line_casida_ref.hpp"

namespace {

using namespace methods::gw_line;
using numerics::line_dlr::time_ray_t;
using numerics::line_dlr::sector_t;
using numerics::line_dlr::bosonic_basis_t;
namespace mpi3 = boost::mpi3;
using cmat = nda::matrix<ComplexType>;
using namespace gw_line_test;

void run_v2(std::string const &fixture, long nI_factor) {
  using namespace methods;
  auto &mpi  = utils::make_unit_test_mpi_context();
  auto &comm = mpi->comm;
  auto all   = nda::range::all;
  auto mf    = std::make_shared<mf::MF>(mf::default_MF(mpi, fixture));
  utils::check(mf->nkpts() == mf->nkpts_ibz() and mf->nqpts() == mf->nqpts_ibz(), "[V2] {}: nosym fixture required", fixture);
  utils::check(mf->nspin() == 1 and mf->npol() == 1, "[V2] {}: spin-restricted collinear fixture required", fixture);
  thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * nI_factor, "", "incore", "", "bdft", 1e-10, mf->ecutrho(), 1, 1024));
  const long nk = mf->nkpts(), nq = mf->nqpts(), nb = thc.nbnd(), Np = thc.Np();

  // KS spectrum, mu mid-gap, the largest transition energy
  nda::array<double, 2> eig(nk, nb);
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
  utils::check(lumo > homo, "[V2] {}: no KS gap", fixture);
  const double mu = 0.5 * (homo + lumo), ks_gap = lumo - homo, etr_max = emax_vir - emin_occ;
  auto poles = pole_data_t::from_ks(eig, mu);
  nda::array<double, 2> e_rel(nk, nb);
  for (long ik = 0; ik < nk; ++ik)
    for (long n = 0; n < nb; ++n) e_rel(ik, n) = eig(ik, n) - mu;

  // bosonic basis: theta = 20 deg, lam_b = max(4, 1.2 x largest transition), gap = KS gap / 2, eps = 1e-10
  const double deg = std::numbers::pi / 180.0, theta = 20.0 * deg, theta_t = 10.0 * deg;
  const double lam_b = std::max(4.0, 1.2 * etr_max), gap_b = 0.5 * ks_gap;
  bosonic_basis_t basis(theta, lam_b, 1e-10, gap_b);
  auto const &zeta = basis.zeta_nodes;
  const long nz = zeta.size(), r = basis.rank;
  app_log(2, "\n[V2] {}: nk={} nq={} nb={} Np={} (nIpts = {} nbnd), mu={:.6f} Ha, KS gap {:.4f} Ha, largest transition {:.4f} Ha",
          fixture, nk, nq, nb, Np, nI_factor, mu, ks_gap, etr_max);
  app_log(2, "  bosonic basis: theta 20 deg, lam_b {:.4f} Ha, gap {:.4f} Ha, eps 1e-10 -> rank {}, {} nodes, nu in [{:.4f}, {:.4f}]",
          lam_b, gap_b, r, nz, basis.nu(0), basis.nu(r - 1));

  // Pi at the bosonic nodes (S3)
  aux_grid_t grid(*mpi, Np);
  const long t_chunk = 8;
  grid.log(nk, nq, nz, r, t_chunk, nb);
  auto ray_p = time_ray_t::for_spectrum(theta_t, poles.emin(), 36.0, 1e-5, 3.0, 16, sector_t::particle);
  auto ray_h = time_ray_t::for_spectrum(theta_t, poles.emin(), 36.0, 1e-5, 3.0, 16, sector_t::hole);
  utils::TimerManager Timer;
  propagator_t<HOST_MEMORY> prop(thc, grid);
  memory::array<HOST_MEMORY, ComplexType, 4> Pi;
  Timer.add("Pi");
  Timer.start("Pi");
  polarization<HOST_MEMORY>(prop, poles, *mf, grid, zeta, ray_p, ray_h, t_chunk, Pi, Timer);
  Timer.stop("Pi");
  nda::array<ComplexType, 4> Pi0 = Pi;   // the call below consumes Pi

  // W
  dyson_layout_t lay(*mpi, nq, nz, Np);
  lay.log();
  coulomb_blocks_t<HOST_MEMORY> Zb(thc, grid, lay.q_rng(), Timer);
  memory::array<HOST_MEMORY, ComplexType, 4> w, Wn;
  screened_interaction<HOST_MEMORY>(Pi, Zb, basis, grid, *mpi, w, Timer, &Wn);
  REQUIRE(Pi.size() == 0);
  REQUIRE(w.extent(0) == nq);
  REQUIRE(w.extent(1) == r);

  // ---- CoQui's imaginary-axis W^c(q, i nu) (harness of SCF/tests/test_methods_tc_contour.cpp)
  const double beta = 1000.0;
  double e_lo = 1e300, e_hi = -1e300;
  for (long ik = 0; ik < nk; ++ik)
    for (long n = 0; n < nb; ++n) { e_lo = std::min(e_lo, eig(ik, n)); e_hi = std::max(e_hi, eig(ik, n)); }
  const double w_max = std::max(std::abs(e_lo), std::abs(e_hi)) + 2.0;
  imag_axes_ft::IAFT ft(beta, w_max + 1.0, imag_axes_ft::dlr_basis, "high");
  MBState mb_state(mpi, ft, std::string("coqui_gw_line_v2_") + fixture);
  simple_dyson dyson(mf.get(), &ft);
  const long ns = 1;
  mb_state.sF_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(*mpi, {ns, nk, nb, nb}));
  mb_state.sDm_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(*mpi, {ns, nk, nb, nb}));
  mb_state.sG_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(*mpi, {ft.nt_f(), ns, nk, nb, nb}));
  mb_state.sSigma_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(*mpi, {ft.nt_f(), ns, nk, nb, nb}));
  auto &sF = mb_state.sF_skij.value();
  auto &sDm = mb_state.sDm_skij.value();
  auto &sG = mb_state.sG_tskij.value();
  auto &sSigma = mb_state.sSigma_tskij.value();
  hamilt::set_fock(*mf, dyson.PSP(), sF, true);
  if (mpi->node_comm.root()) sSigma.local() = ComplexType(0.0);
  sSigma.communicator()->barrier();
  double mu_T = 0.0;
  Timer.add("coqui_W");
  Timer.start("coqui_W");
  update_G(dyson, *mf, ft, sDm, sG, sF, sSigma, mu_T, false);
  {   // P1 pin of the harness: H0 + F = diag(eps) in the KS basis, S = 1
    auto H0 = dyson.H0();
    auto S  = dyson.sS_skij().local();
    auto F  = sF.local();
    double dH = 0.0, dS = 0.0;
    for (long k = 0; k < nk; ++k)
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) {
          dH = std::max(dH, std::abs(H0(0, k, i, j) + F(0, k, i, j) - ComplexType(i == j ? eig(k, i) : 0.0)));
          dS = std::max(dS, std::abs(S(0, k, i, j) - ComplexType(i == j ? 1.0 : 0.0)));
        }
    app_log(2, "  CoQui reference: beta {} wmax {:.3f}, mu(T) {:.8f} Ha (line mu {:.8f}); |H0+F-diag(eps)| {:.1e}, |S-1| {:.1e}",
            beta, w_max + 1.0, mu_T, mu, dH, dS);
    REQUIRE(dH < 1e-8);
    REQUIRE(dS < 1e-8);
  }
  solvers::scr_coulomb_t scr_im(&ft, "rpa", "ignore_g0");
  scr_im.update_w(mb_state, thc, -1);
  REQUIRE(mb_state.dW_qtPQ.has_value());
  Timer.stop("coqui_W");
  nda::array<ComplexType, 4> Pi_c_wq;   // CoQui's Pi(i nu_n), (nw_half, nq, Np, Np)
  {
    auto dPi = scr_im.eval_Pi_qdep(mb_state, thc);
    auto gsh = dPi.global_shape();
    nda::array<ComplexType, 4> Pt(gsh[0], gsh[1], gsh[2], gsh[3]);
    Pt() = ComplexType(0.0);
    Pt(dPi.local_range(0), dPi.local_range(1), dPi.local_range(2), dPi.local_range(3)) = dPi.local();
    comm.all_reduce_in_place_n(Pt.data(), Pt.size(), std::plus<>{});
    const long nwh = (ft.nw_b() % 2 == 0) ? ft.nw_b() / 2 : ft.nw_b() / 2 + 1;
    Pi_c_wq = nda::array<ComplexType, 4>(nwh, gsh[1], gsh[2], gsh[3]);
    ft.tau_to_w_PHsym(Pt, Pi_c_wq);
  }
  auto const &dW = mb_state.dW_qtPQ.value();
  const long nt_half = dW.global_shape()[1];
  const long nw_half = (ft.nw_b() % 2 == 0) ? ft.nw_b() / 2 : ft.nw_b() / 2 + 1;
  auto wn_b = ft.wn_mesh_b();
  nda::array<ComplexType, 1> zi(nw_half);
  for (long n = 0; n < nw_half; ++n) zi(n) = ComplexType(0.0, ft.omega(wn_b(ft.nw_b() / 2 + n)).imag());
  REQUIRE(std::abs(zi(0)) < 1e-12);

  // 12 mu-relative points on the two upper rays (as V1)
  auto rr = numerics::line_dlr::detail::logspace(0.05, 5.0, 6);
  nda::array<ComplexType, 1> z12(12);
  for (long i = 0; i < 6; ++i) {
    z12(i)     = rr(i) * std::exp(ComplexType(0.0, theta));
    z12(6 + i) = rr(i) * std::exp(ComplexType(0.0, std::numbers::pi - theta));
  }

  // ---- per q (lockstep: thc.Z is collective)
  double a_pex = 0, e_psym = 0, e_wcq = 0, e_pex = 0, s_pc = 0, e_lex = 0, ewb = 0, ea = 0, sa = 0, eb = 0, sb = 0, ec = 0, sc = 0, ed = 0, sd = 0, ecn = 0, scn = 0, res_fit = 0, et = 0, st = 0;
  double ee = 0, se = 0, e_wsym = 0, e_wold = 0, s_wsym = 0, e_zsym = 0, e_zold = 0, s_z = 0, e_mir = 0;
  auto qm = mf->qminus();
  long n_self = 0;
  for (long iq = 0; iq < nq; ++iq) n_self += (qm(iq) == iq) ? 1 : 0;
  // [V2-sym] the exact Casida solutions of all q (W_dyn(-q) is needed with q; small on the fixtures)
  std::vector<casida_t> cas_all;
  std::vector<cmat> Z_all;
  for (long iq = 0; iq < nq; ++iq) {   // LOCKSTEP: thc.Z is collective
    Z_all.emplace_back(thc.Z(int(iq)));
    cas_all.push_back(casida_q(thc, *mf, e_rel, iq, Z_all.back()));
  }
  {
    aux_grid_t g1(1, 0, Np);
    nda::array<ComplexType, 1> mz(12);
    for (long i = 0; i < 12; ++i) mz(i) = -z12(i);
    for (long iq = 0; iq < nq; ++iq) {
      const long jq = qm(iq);
      auto Wm = casida_eval(cas_all[iq], g1, mz, false);    // W_dyn(q, -z)
      auto Wp = casida_eval(cas_all[jq], g1, z12, false);   // W_dyn(-q, z)
      auto Wq = casida_eval(cas_all[iq], g1, z12, false);   // W_dyn(q, z)
      for (long i = 0; i < 12; ++i)
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            s_wsym = std::max(s_wsym, std::abs(Wq(i, P, Q)));
            e_wsym = std::max(e_wsym, std::abs(Wm(i, P, Q) - Wp(i, Q, P)));
            e_wold = std::max(e_wold, std::abs(Wm(i, P, Q) - Wq(i, Q, P)));
          }
      for (long P = 0; P < Np; ++P)
        for (long Q = 0; Q < Np; ++Q) {
          s_z    = std::max(s_z, std::abs(Z_all[iq](P, Q)));
          e_zsym = std::max(e_zsym, std::abs(Z_all[jq](P, Q) - Z_all[iq](Q, P)));
          e_zold = std::max(e_zold, std::abs(Z_all[jq](P, Q) - Z_all[iq](P, Q)));
        }
    }
  }
  long Tmax = 0, npos = 0;
  double minM = 1e300, lmax = 0.0, lmin = 1e300;
  Timer.add("reference");
  for (long iq = 0; iq < nq; ++iq) {
    Timer.start("reference");
    const long jq     = qm(iq);   // -q
    const bool self_q = (jq == iq);
    cmat const &Zq = Z_all[iq];
    auto Pf = gather_full(comm, grid, Pi0(iq, all, all, all));
    auto Wf = gather_full(comm, grid, Wn(iq, all, all, all));
    auto Wfm = self_q ? Wf : gather_full(comm, grid, Wn(jq, all, all, all));   // W(-q) at the nodes
    cmat Id = nda::eye<ComplexType>(Np);
    // (a) Dyson by nda::inverse on the full matrices. Mirror-symmetric nodes (perf 7.1 (b)): the code solves the ray-1
    //     nodes and sets W(q, -conj z_i) = conj W(-q, z_i); the reference does the same (Dyson of -q with Z(-q)); the
    //     direct Dyson of q at the ray-2 nodes differs by the THC asymmetry of Z (info e_mir)
    const long n1m = numerics::line_dlr::mirror_half(zeta);
    auto Pfm       = (n1m > 0 and not self_q) ? gather_full(comm, grid, Pi0(jq, all, all, all)) : Pf;
    nda::array<ComplexType, 3> Wr(nz, Np, Np);
    for (long iz = 0; iz < nz; ++iz) {
      cmat P(Pf(iz, all, all));
      cmat A = Id - Zq * P;
      cmat Ai = nda::inverse(A);
      Wr(iz, all, all) = Ai * Zq - Zq;
      if (n1m > 0 and iz >= n1m) {
        cmat const &Zm = Z_all[jq];
        cmat Pm(Pfm(iz - n1m, all, all));
        cmat Wm = nda::inverse(cmat(Id - Zm * Pm)) * Zm - Zm;
        for (long P_ = 0; P_ < Np; ++P_)
          for (long Q_ = 0; Q_ < Np; ++Q_) {
            e_mir = std::max(e_mir, std::abs(std::conj(Wm(P_, Q_)) - Wr(iz, P_, Q_)));
            Wr(iz, P_, Q_) = std::conj(Wm(P_, Q_));
          }
      }
    }
    ea = std::max(ea, max_diff3(Wf, Wr));
    sa = std::max(sa, max_abs3(Wr));
    // (b) the residues vs bosonic_basis_t::fit (gelss) of the gathered W, compared as pole functions on the nodes, the
    //     12 ray points and the Matsubara nodes (the residues themselves are not unique below ~1e-5: the stacked kernel has
    //     cond ~1e13 and both solvers resolve the near-threshold singular directions from rounding noise; info ewb);
    //     refit residual at the nodes (info)
    //     q != -q: the paired fit of q (data W(q), W(-q)^T) and the paired pole sums eval(w(q), w(-q), z)
    auto wr  = self_q ? basis.fit(zeta, Wf) : basis.fit(zeta, Wf, Wfm);
    auto wrm = self_q ? wr : basis.fit(zeta, Wfm, Wf);
    auto wf  = gather_full(comm, grid, w(iq, all, all, all));
    auto wfm = self_q ? wf : gather_full(comm, grid, w(jq, all, all, all));
    auto evp = [&](auto const &a, auto const &am, auto const &zz) { return self_q ? basis.eval(a, zz) : basis.eval(a, am, zz); };
    ewb = std::max(ewb, max_diff3(wf, wr) / max_abs3(wr));
    for (auto const *zz : std::array<nda::array<ComplexType, 1> const *, 3>{&zeta, &z12, &zi}) {
      auto Fr = evp(wr, wrm, *zz);
      eb      = std::max(eb, max_diff3(evp(wf, wfm, *zz), Fr));
      sb      = std::max(sb, max_abs3(Fr));
    }
    res_fit = std::max(res_fit, max_diff3(evp(wf, wfm, zeta), Wf) / max_abs3(Wf));
    auto const &cas = cas_all[iq];
    // (c) the fit on the Matsubara axis vs CoQui W^c(q, i nu_n). Decomposed with the exact transition-sum Pi(i nu_n):
    //     line vs W_ex = Dyson(exact Pi); CoQui W^c vs Dyson(CoQui's Pi); CoQui's Pi vs the exact Pi, in full and for the
    //     (P,Q)-symmetric part. CoQui keeps Pi(q, tau) on the PH-symmetric half tau grid (tau_to_w_PHsym assumes
    //     Pi(beta - tau) = Pi(tau) ELEMENTWISE; exactly Pi(beta - tau) = Pi(tau)^T), so only the symmetric part of its Pi is
    //     exact; its antisymmetric part is off by the size of the exact one (max|Pi - Pi^T|: si211 2.7e-5 of max|Pi|,
    //     lih222 7e-9). The line builds both sectors explicitly and is not affected.
    {
      nda::array<ComplexType, 4> Wt(nt_half, 1, Np, Np), Ww(nw_half, 1, Np, Np);
      Wt() = 0.0;
      if (iq >= dW.local_range(0).first() and iq < dW.local_range(0).last())
        Wt(dW.local_range(1), 0, dW.local_range(2), dW.local_range(3)) =
            dW.local()(iq - dW.local_range(0).first(), all, all, all);
      comm.all_reduce_in_place_n(Wt.data(), Wt.size(), std::plus<>{});
      ft.tau_to_w_PHsym(Wt, Ww);
      auto Wl = evp(wf, wfm, zi);
      nda::array<ComplexType, 3> Wc(Ww(all, 0, all, all));
      ec = std::max(ec, max_diff3(Wl, Wc));
      for (long n = 0; n < nw_half; ++n) {
        cmat SP(cas.S);
        for (long t = 0; t < cas.T; ++t) {
          const ComplexType f = cas.sg[t] / (zi(n) - cas.E[t]);
          for (long P = 0; P < Np; ++P) SP(P, t) *= f;
        }
        cmat Pex = SP * nda::dagger(cas.S);   // exact transition sum, both sectors
        cmat Pcq(Pi_c_wq(n, iq, all, all));     // CoQui's own Pi
        cmat Wex = nda::inverse(cmat(Id - Zq * Pex)) * Zq - Zq;
        cmat Wcq = nda::inverse(cmat(Id - Zq * Pcq)) * Zq - Zq;
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            e_lex  = std::max(e_lex, std::abs(Wl(n, P, Q) - Wex(P, Q)));
            e_wcq  = std::max(e_wcq, std::abs(Wc(n, P, Q) - Wcq(P, Q)));
            e_pex  = std::max(e_pex, std::abs(Pcq(P, Q) - Pex(P, Q)));
            e_psym = std::max(e_psym, std::abs(0.5 * (Pcq(P, Q) + Pcq(Q, P) - Pex(P, Q) - Pex(Q, P))));
            s_pc   = std::max(s_pc, std::abs(Pex(P, Q)));
            a_pex  = std::max(a_pex, std::abs(Pex(P, Q) - Pex(Q, P)));
          }
      }
      sc = std::max(sc, max_abs3(Wc));
    }
    // (d) Casida: particle part at 12 points; full W_dyn at the nodes (info)
    {
      Tmax = std::max(Tmax, cas.T);
      npos = cas.npos;
      minM = std::min(minM, cas.min_M);
      for (long s = 0; s < cas.T; ++s) { lmax = std::max(lmax, std::abs(cas.lam(s))); lmin = std::min(lmin, std::abs(cas.lam(s))); }
      auto Cp = casida_eval(cas, grid, z12, true);
      memory::array<HOST_MEMORY, ComplexType, 3> Lp(12, grid.nP, grid.nQ);
      eval_poles<HOST_MEMORY>(w, basis, iq, jq, z12, sector_t::particle, false, Lp());
      ed = std::max(ed, comm.all_reduce_value(max_diff3(Lp, Cp), mpi3::max<>{}));
      sd = std::max(sd, comm.all_reduce_value(max_abs3(Cp), mpi3::max<>{}));
      // (e) hole part, transposed: block (P_rng, Q_rng) of W^<(q, z)^T vs the Casida negative poles of q, transposed
      {
        aux_grid_t g1(1, 0, Np);
        auto Ct = casida_eval(cas, g1, z12, false);
        auto Cq = casida_eval(cas, g1, z12, true);
        memory::array<HOST_MEMORY, ComplexType, 3> Lh(12, grid.nP, grid.nQ);
        eval_poles<HOST_MEMORY>(w, basis, iq, jq, z12, sector_t::hole, true, Lh());
        nda::array<ComplexType, 3> Ch(12, grid.nP, grid.nQ);
        for (long i = 0; i < 12; ++i)
          for (long P = 0; P < grid.nP; ++P)
            for (long Q = 0; Q < grid.nQ; ++Q)
              Ch(i, P, Q) = Ct(i, grid.Q0 + Q, grid.P0 + P) - Cq(i, grid.Q0 + Q, grid.P0 + P);
        ee = std::max(ee, comm.all_reduce_value(max_diff3(Lh, Ch), mpi3::max<>{}));
        se = std::max(se, comm.all_reduce_value(max_abs3(Ch), mpi3::max<>{}));
      }
      auto Cn = casida_eval(cas, grid, zeta, false);
      nda::array<ComplexType, 3> Wnb(Wn(iq, all, all, all));
      ecn = std::max(ecn, comm.all_reduce_value(max_diff3(Wnb, Cn), mpi3::max<>{}));
      scn = std::max(scn, comm.all_reduce_value(max_abs3(Cn), mpi3::max<>{}));
    }
    // time-domain convention (q = 0): F_ray W^>(t) = W^>(zeta), F_ray W^<(t)^T = W^<(zeta)^T at 4 nodes
    if (iq == 0) {
      nda::array<ComplexType, 1> z4(4);
      for (long i = 0; i < 4; ++i) z4(i) = zeta((i * (nz - 1)) / 3);
      for (auto s : {sector_t::particle, sector_t::hole}) {
        auto ray = time_ray_t::for_spectrum(theta_t, basis.nu(0), 36.0, 1e-5, 4.0, 20, s);
        const bool tr = (s == sector_t::hole);
        memory::array<HOST_MEMORY, ComplexType, 3> Wt(ray.size(), grid.nP, grid.nQ), Wz(4, grid.nP, grid.nQ);
        w_time<HOST_MEMORY>(w, basis, iq, jq, ray.t, s, tr, Wt());
        eval_poles<HOST_MEMORY>(w, basis, iq, jq, z4, s, tr, Wz());
        auto F = ray.transform_matrix(z4);
        nda::array<ComplexType, 2> R(4, grid.nP * grid.nQ);
        nda::blas::gemm(ComplexType(1.0), F, nda::reshape(Wt, std::array<long, 2>{ray.size(), grid.nP * grid.nQ}),
                        ComplexType(0.0), R);
        auto Wz2 = nda::reshape(Wz, std::array<long, 2>{4, grid.nP * grid.nQ});
        et = std::max(et, comm.all_reduce_value(max_diff3(R, Wz2), mpi3::max<>{}));
        st = std::max(st, comm.all_reduce_value(max_abs3(Wz2), mpi3::max<>{}));
      }
    }
    Timer.stop("reference");
  }
  const double r_lex = e_lex / sc, r_pex = e_pex / s_pc, r_psym = e_psym / s_pc, r_apex = a_pex / s_pc;
  const double ra = ea / sa, rb = eb / sb, rc = ec / sc, rd = ed / sd, rcn = ecn / scn, rt = et / st, re = ee / se;
  app_log(2, "  [V2] {} ({} ranks, aux grid {}x{}, Dyson pools {}x{}):", fixture, comm.size(), grid.np_P, grid.np_Q, lay.np_q,
          lay.np_z);
  app_log(2, "    (a) W at the {} nodes vs Dyson (nda::inverse) of the gathered Pi: rel {:.2e} (max|W| {:.3e}); mirror nodes (ray 2 = "
             "conj W(-q) at ray 1, {} per ray) vs the direct Dyson of q there: {:.2e} (THC Z asymmetry)",
          nz, ra, sa, numerics::line_dlr::mirror_half(zeta), e_mir / sa);
  app_log(2, "    (b) fit vs bosonic_basis_t::fit of the gathered W, as pole sums on nodes/ray points/i nu_n: rel {:.2e} (max|W| {:.3e}); "
             "raw residues differ by {:.1e} (non-unique, info); refit residual at the nodes {:.2e}",
          rb, sb, ewb, res_fit);
  app_log(2, "    (c) fit at i nu_n vs CoQui W^c(q, i nu_n), {} bosonic nodes, nu_max {:.3e}: rel {:.2e} (max|W^c| {:.3e}); beta {}, "
             "KS gap {:.4f} Ha (e^(-beta gap/2) = {:.1e})",
          nw_half, zi(nw_half - 1).imag(), rc, sc, beta, ks_gap, std::exp(-beta * ks_gap / 2));
  app_log(2, "        decomposition: line fit vs exact Dyson(Pi(i nu_n)) {:.2e}; CoQui W^c vs Dyson(CoQui Pi) {:.2e}; CoQui Pi vs exact Pi "
             "{:.2e}, (P,Q)-symmetric parts {:.2e}; max|Pi - Pi^T| of the exact Pi {:.2e} (max|Pi| {:.3e})",
          r_lex, e_wcq / sc, r_pex, r_psym, r_apex, s_pc);
  app_log(2, "    (d) W^> at 12 ray points vs Casida positive poles:               rel {:.2e} (max|W^>| {:.3e}); Casida dim {} "
             "({} positive), min eig(M) {:.3e}, |RPA poles| in [{:.4f}, {:.4f}] Ha",
          rd, sd, Tmax, npos, minM, lmin, lmax);
  app_log(2, "    (e) W^<(q)^T at 12 ray points (residues of -q) vs Casida negative poles of q, transposed: rel {:.2e} (max|W^<| {:.3e})",
          re, se);
  app_log(2, "    info: W at the nodes vs Casida W_dyn {:.2e}; ray transform of w_time vs eval_poles (q=0) {:.2e}", rcn, rt);
  app_log(2, "    [V2-sym] {} of {} q self-inverse; exact Casida: max|W(q,-z) - W(-q,z)^T| / max|W| = {:.2e}, pre-fix form "
             "max|W(q,-z) - W(q,z)^T| / max|W| = {:.2e}; THC Coulomb: max|Z(-q) - Z(q)^T| / max|Z| = {:.2e}, "
             "max|Z(-q) - Z(q)| / max|Z| = {:.2e}",
          n_self, nq, e_wsym / s_wsym, e_wold / s_wsym, e_zsym / s_z, e_zold / s_z);
  app_log(2, "  [V2] {} timers (rank 0): G_tilde {:.2f} Pi_hadamard {:.2f} Pi_transform {:.2f} | Z_gather {:.3f} W_redistribute {:.3f} "
             "W_dyson {:.3f} W_fit {:.3f} | CoQui W {:.2f} references {:.2f} s",
          fixture, Timer.elapsed("G_tilde"), Timer.elapsed("Pi_hadamard"), Timer.elapsed("Pi_transform"),
          Timer.elapsed("Z_gather"), Timer.elapsed("W_redistribute"), Timer.elapsed("W_dyson"), Timer.elapsed("W_fit"),
          Timer.elapsed("coqui_W"), Timer.elapsed("reference"));
  REQUIRE(ra <= 1e-12);
  REQUIRE(rb <= 1e-10);
  REQUIRE(r_lex <= 1e-8);                 // the line fit on the Matsubara axis vs the exact W(i nu_n)
  REQUIRE(e_wcq / sc <= 1e-10);           // CoQui's W^c is the Dyson of CoQui's Pi (harness, conventions, Z, head)
  REQUIRE(r_psym <= 1e-8);                // CoQui's Pi = exact Pi on the (P,Q)-symmetric part (sign, factor 2, mu, FT)
  REQUIRE(r_pex <= 1e-8 + 2.0 * r_apex);  // ... and off only at the size of the antisymmetric part (PH-symmetric tau grid)
  REQUIRE(rc <= 1e-8 + 2.0 * r_apex);     // line vs CoQui W^c: 1e-8 up to CoQui's PH-symmetric approximation
  REQUIRE(rd <= 1e-8);
  REQUIRE(re <= 1e-8);
  // Z(-q) = Z(q)^T holds to the THC fit noise (lih222 1.3e-12, lih223 3.5e-12, si211 1.8e-9 even at q = -q, where it is
  // the imaginary part of Z); the W relation inherits it
  REQUIRE(e_zsym <= 1e-8 * s_z);
  REQUIRE(e_wsym / s_wsym <= 1e-10 + 10.0 * e_zsym / s_z);
  REQUIRE(rcn <= 1e-10 + 10.0 * e_zsym / s_z);   // ray-2 nodes from W(-q) with Z(-q): the THC asymmetry of Z (si211 1.7e-9)
  REQUIRE(e_mir / sa <= 1e-11 + 10.0 * e_zsym / s_z);
  REQUIRE(rt <= 1e-9);
}

} // namespace

TEST_CASE("gw_line_V2_lih222", "[gw_line][V2]") { run_v2("qe_lih222", 8); }

TEST_CASE("gw_line_V2_si211", "[gw_line][V2]") { run_v2("qe_si211", 8); }

// q != -q mesh (2x2x3, 8 of 12 q not self-inverse): the W pairing fix (notes section 3.3)
TEST_CASE("gw_line_V2_lih223", "[gw_line][V2]") { run_v2("qe_lih223", 8); }
