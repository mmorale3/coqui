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
 * S7b of notes/line_gw_cpp_plan.md: the GW_line kernels on the compressed time grids (time-node ID, time_id.hpp, built by
 * time_grids.hpp from the current poles exactly as the driver does) vs the generic Gauss-Legendre rays (the oracle).
 *
 * Setup as [V2]/[V3]: THC nIpts = 8 nbnd, KS poles, mu = KS gap midpoint, theta = 20 deg, theta_t = 10 deg, GL rays
 * for_spectrum(theta_t, emin, 36), bosonic basis lam_b = max(4, 1.2 x largest transition), gap = KS gap / 2, eps 1e-10,
 * dense fermionic nodes dense_nodes(20 deg, 1e-3, 60, 120). ID grids: eps_t in {1e-10, 1e-8}, pad 1.25, oversample 1.0.
 *  (a) Pi(q, zeta) at the bosonic nodes per sector, ID vs GL: <= 10 eps_t relative (max norm over q, zeta, P, Q);
 *      both vs the exact Casida transition sum (V1) for the record.
 *  (b) Sigma(k, zeta) at the dense nodes per sector, ID vs GL with the SAME W residues (from the GL Pi): <= 10 eps_t;
 *  (c) the full ID chain (Pi_ID -> W -> Sigma_ID) and the GL chain vs the exact Casida Sigma_c (V3a) for the record.
 * Kernel timers ID vs GL are logged (the ray products scale with the node count).
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

#include "numerics/line_dlr/time_ray.hpp"
#include "numerics/line_dlr/time_id.hpp"
#include "numerics/line_dlr/line_basis.hpp"
#include "numerics/line_dlr/bosonic_basis.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/polarization.hpp"
#include "methods/GW_line/screened.hpp"
#include "methods/GW_line/self_energy.hpp"
#include "methods/GW_line/time_grids.hpp"
#include "gw_line_casida_ref.hpp"

namespace {

using namespace methods::gw_line;
using numerics::line_dlr::time_ray_t;
using numerics::line_dlr::time_id_opts_t;
using numerics::line_dlr::sector_t;
using numerics::line_dlr::bosonic_basis_t;
using namespace gw_line_test;

constexpr double deg = std::numbers::pi / 180.0;

/// Casida transition sum of Pi for the (P_rng, Q_rng) block, all q, one sector (as [V1]).
nda::array<ComplexType, 4> pi_exact_block(methods::thc_reader_t &thc, mf::MF &mf, nda::array<double, 2> const &e_rel,
                                          aux_grid_t const &g, nda::array<ComplexType, 1> const &zeta, sector_t sec) {
  const long nk = mf.nkpts(), nq = mf.nqpts(), nb = e_rel.extent(1), nz = zeta.size();
  auto qk       = mf.qk_to_k2();
  nda::array<ComplexType, 4> out(nq, nz, g.nP, g.nQ);
  out() = 0.0;
  const double nrm = std::sqrt(2.0 / double(nk));
  for (long iq = 0; iq < nq; ++iq) {
    std::vector<long> tk, tn, tm;
    std::vector<double> E, sg;
    for (long ik = 0; ik < nk; ++ik) {
      const long ikmq = qk(iq, ik);
      for (long n = 0; n < nb; ++n)
        for (long m = 0; m < nb; ++m) {
          const bool occ_n = e_rel(ik, n) < 0.0, occ_m = e_rel(ikmq, m) < 0.0;
          if (occ_n == occ_m) continue;
          if ((sec == sector_t::particle) != occ_n) continue;
          tk.push_back(ik); tn.push_back(n); tm.push_back(m);
          E.push_back(e_rel(ikmq, m) - e_rel(ik, n));
          sg.push_back(occ_n ? 1.0 : -1.0);
        }
    }
    const long T = E.size();
    if (T == 0) continue;
    cmat SP(g.nP, T), SQ(g.nQ, T), W(g.nP, T);
    for (long it = 0; it < T; ++it) {
      auto Xk   = thc.X(0, 0, tk[it]);
      auto Xkmq = thc.X(0, 0, qk(iq, tk[it]));
      for (long P = 0; P < g.nP; ++P) SP(P, it) = nrm * Xk(g.P0 + P, tn[it]) * std::conj(Xkmq(g.P0 + P, tm[it]));
      for (long Q = 0; Q < g.nQ; ++Q) SQ(Q, it) = nrm * Xk(g.Q0 + Q, tn[it]) * std::conj(Xkmq(g.Q0 + Q, tm[it]));
    }
    for (long iz = 0; iz < nz; ++iz) {
      for (long it = 0; it < T; ++it) {
        const ComplexType f = sg[it] / (zeta(iz) - E[it]);
        for (long P = 0; P < g.nP; ++P) W(P, it) = SP(P, it) * f;
      }
      nda::matrix_view<ComplexType> o(out(iq, iz, nda::range::all, nda::range::all));
      nda::blas::gemm(ComplexType(1.0), W, nda::dagger(SQ), ComplexType(0.0), o);
    }
  }
  return out;
}

struct tid_setup_t {
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;
  std::shared_ptr<mf::MF> mf;
  std::unique_ptr<methods::thc_reader_t> thc;
  long nk = 0, nq = 0, nb = 0, Np = 0;
  nda::array<double, 2> eig, e_rel;
  double mu = 0, ks_gap = 0, etr_max = 0;
  pole_data_t poles;

  explicit tid_setup_t(std::string const &fixture) {
    mpi = utils::make_unit_test_mpi_context();
    mf  = std::make_shared<mf::MF>(mf::default_MF(mpi, fixture));
    utils::check(mf->nkpts() == mf->nkpts_ibz() and mf->nqpts() == mf->nqpts_ibz(), "{}: nosym fixture required", fixture);
    utils::check(mf->nspin() == 1 and mf->npol() == 1, "{}: spin-restricted collinear fixture required", fixture);
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
    utils::check(lumo > homo, "{}: no KS gap", fixture);
    mu = 0.5 * (homo + lumo); ks_gap = lumo - homo; etr_max = emax_vir - emin_occ;
    poles = pole_data_t::from_ks(eig, mu);
    e_rel = nda::array<double, 2>(nk, nb);
    for (long ik = 0; ik < nk; ++ik)
      for (long n = 0; n < nb; ++n) e_rel(ik, n) = eig(ik, n) - mu;
  }
};

/// exact Casida Sigma_c per sector at the points z (as [V3a]); k distributed over ranks, all_reduced.
void sigma_exact(tid_setup_t &su, nda::array<ComplexType, 1> const &z, nda::array<ComplexType, 4> &Rp,
                 nda::array<ComplexType, 4> &Rh) {
  auto &comm = su.mpi->comm;
  auto &thc  = *su.thc;
  auto &mf   = *su.mf;
  auto all   = nda::range::all;
  const long nk = su.nk, nq = su.nq, nb = su.nb, Np = su.Np, nz = z.size();
  auto qk = mf.qk_to_k2();
  Rp = nda::array<ComplexType, 4>(nk, nz, nb, nb);
  Rh = nda::array<ComplexType, 4>(nk, nz, nb, nb);
  Rp() = 0.0;
  Rh() = 0.0;
  for (long iq = 0; iq < nq; ++iq) {   // LOCKSTEP: thc.Z is collective
    cmat Zq(thc.Z(int(iq)));
    auto cas = casida_q(thc, mf, su.e_rel, iq, Zq);
    for (long ik = comm.rank(); ik < nk; ik += comm.size()) {
      const long ikmq = qk(iq, ik);
      auto Xk  = thc.X(0, 0, ik);
      auto Xm  = thc.X(0, 0, ikmq);
      cmat XkH = nda::dagger(cmat(Xk));
      for (int sec = 0; sec < 2; ++sec) {   // 0: particle, 1: hole
        std::vector<long> ns_, ss_;
        for (long n = 0; n < nb; ++n)
          if ((sec == 0) == (su.e_rel(ikmq, n) > 0.0)) ns_.push_back(n);
        for (long s = 0; s < cas.T; ++s)
          if ((sec == 0) == (cas.lam(s) > 0.0)) ss_.push_back(s);
        const long Nn = ns_.size(), Ns = ss_.size(), npl = Nn * Ns;
        if (npl == 0) continue;
        nda::array<ComplexType, 2> K(nz, npl), B(npl, nb * nb);
        cmat Y(Np, Ns), A(nb, Ns);
        const double sgn = (sec == 0) ? 1.0 : -1.0;
        for (long in = 0; in < Nn; ++in) {
          const long n = ns_[in];
          for (long P = 0; P < Np; ++P)
            for (long js = 0; js < Ns; ++js) Y(P, js) = Xm(P, n) * cas.V(P, ss_[js]);
          A = XkH * Y;
          for (long js = 0; js < Ns; ++js) {
            const long p     = in * Ns + js;
            const double lam = cas.lam(ss_[js]), E = su.e_rel(ikmq, n) + lam, wgt = sgn * lam / double(nk);
            for (long a = 0; a < nb; ++a)
              for (long b = 0; b < nb; ++b) B(p, a * nb + b) = wgt * A(a, js) * std::conj(A(b, js));
            for (long iz = 0; iz < nz; ++iz) K(iz, p) = 1.0 / (z(iz) - E);
          }
        }
        auto R2 = nda::reshape((sec == 0 ? Rp : Rh)(ik, all, all, all), std::array<long, 2>{nz, nb * nb});
        nda::blas::gemm(ComplexType(1.0), K, B, ComplexType(1.0), R2);
      }
    }
  }
  comm.all_reduce_in_place_n(Rp.data(), Rp.size(), std::plus<>{});
  comm.all_reduce_in_place_n(Rh.data(), Rh.size(), std::plus<>{});
}

double pi_time(utils::TimerManager &T) { return T.elapsed("G_tilde") + T.elapsed("Pi_hadamard") + T.elapsed("Pi_transform"); }
double sigma_time(utils::TimerManager &T) {
  double s = 0.0;
  for (auto nm : {"Sigma_G_tilde", "Sigma_W_time", "Sigma_hadamard", "Sigma_contract", "Sigma_allreduce", "Sigma_transform"})
    s += T.elapsed(nm);
  return s;
}

void run_time_id(std::string const &fixture) {
  tid_setup_t su(fixture);
  auto &mpi  = *su.mpi;
  auto &comm = mpi.comm;
  auto &thc  = *su.thc;
  auto &mf   = *su.mf;
  const long nq = su.nq, Np = su.Np;
  const long t_chunk = 8;

  // relative max-norm difference of distributed blocks (A vs reference B)
  auto rel = [&](auto const &A, auto const &B) {
    const double d = comm.all_reduce_value(max_diff3(A, B), mpi3::max<>{});
    const double m = comm.all_reduce_value(max_abs3(B), mpi3::max<>{});
    return d / m;
  };

  const double theta = 20.0 * deg, theta_t = 10.0 * deg;
  const double lam_b = std::max(4.0, 1.2 * su.etr_max), gap_b = 0.5 * su.ks_gap;
  bosonic_basis_t basis(theta, lam_b, 1e-10, gap_b);
  auto const &zb = basis.zeta_nodes;
  const long nzb = zb.size();
  auto fz        = numerics::line_dlr::dense_nodes(theta, 1e-3, 60.0, 120);
  aux_grid_t grid(mpi, Np);
  dyson_layout_t lay(mpi, nq, nzb, Np);
  utils::TimerManager Tw;
  coulomb_blocks_t<HOST_MEMORY> Zb(thc, grid, lay.q_rng(), Tw);
  propagator_t<HOST_MEMORY> prop(thc, grid);

  auto ray_p = time_ray_t::for_spectrum(theta_t, su.poles.emin(), 36.0, 1e-5, 3.0, 16, sector_t::particle);
  auto ray_h = time_ray_t::for_spectrum(theta_t, su.poles.emin(), 36.0, 1e-5, 3.0, 16, sector_t::hole);
  app_log(2, "\n[time_id] {} ({} ranks, aux grid {}x{}): nk={} nq={} nb={} Np={}, KS gap {:.4f} Ha, bosonic rank {} ({} nodes, "
             "nu [{:.4f}, {:.4f}]), {} fermionic nodes; GL rays {} + {} nodes",
          fixture, comm.size(), grid.np_P, grid.np_Q, su.nk, nq, su.nb, Np, su.ks_gap, basis.rank, nzb, basis.nu(0),
          basis.nu(basis.rank - 1), fz.size(), ray_p.size(), ray_h.size());

  // ---- GL oracle: Pi per sector, W (from the GL Pi), Sigma per sector
  memory::array<HOST_MEMORY, ComplexType, 4> Pp_gl, Ph_gl, w_gl;
  nda::array<ComplexType, 4> Sp_gl, Sh_gl;
  utils::TimerManager Tgl;
  polarization<HOST_MEMORY>(prop, su.poles, mf, grid, zb, ray_p, ray_h, t_chunk, Pp_gl, Tgl, sector_t::particle);
  polarization<HOST_MEMORY>(prop, su.poles, mf, grid, zb, ray_p, ray_h, t_chunk, Ph_gl, Tgl, sector_t::hole);
  const double tPi_gl = pi_time(Tgl);
  {
    memory::array<HOST_MEMORY, ComplexType, 4> Pi = Pp_gl;
    Pi += Ph_gl;
    screened_interaction<HOST_MEMORY>(Pi, Zb, basis, grid, mpi, w_gl, Tw);
  }
  self_energy<HOST_MEMORY>(prop, su.poles, w_gl, basis, mf, grid, mpi, fz, ray_p, ray_h, t_chunk, Sp_gl, Tgl, sector_t::particle);
  self_energy<HOST_MEMORY>(prop, su.poles, w_gl, basis, mf, grid, mpi, fz, ray_p, ray_h, t_chunk, Sh_gl, Tgl, sector_t::hole);
  const double tS_gl = sigma_time(Tgl);

  // ---- exact references
  auto Cp = pi_exact_block(thc, mf, su.e_rel, grid, zb, sector_t::particle);
  auto Ch = pi_exact_block(thc, mf, su.e_rel, grid, zb, sector_t::hole);
  nda::array<ComplexType, 4> Rp, Rh;
  sigma_exact(su, fz, Rp, Rh);
  const double pi_gl_ex_p = rel(Pp_gl, Cp), pi_gl_ex_h = rel(Ph_gl, Ch);
  const double s_gl_ex_p = max_diff3(Sp_gl, Rp) / max_abs3(Rp), s_gl_ex_h = max_diff3(Sh_gl, Rh) / max_abs3(Rh);
  app_log(2, "  GL: Pi vs exact {:.2e} / {:.2e} (particle / hole), Sigma (GL chain) vs exact {:.2e} / {:.2e}; kernels Pi {:.2f} s, "
             "Sigma {:.2f} s (rank 0)",
          pi_gl_ex_p, pi_gl_ex_h, s_gl_ex_p, s_gl_ex_h, tPi_gl, tS_gl);

  for (double eps_t : {1e-10, 1e-8}) {
    time_id_opts_t opts;
    opts.pad        = 1.25;
    opts.oversample = 1.0;
    line_time_grids_t G(su.poles, basis.nu, theta_t, eps_t, opts, zb, fz, comm);
    G.log(2);
    // every rank holds the same nodes
    for (auto const *g : {&G.pi_p, &G.pi_h, &G.sig_p, &G.sig_h}) {
      const long n = g->size();
      REQUIRE(comm.all_reduce_value(n, mpi3::max<>{}) == n);
      REQUIRE(comm.all_reduce_value(n, mpi3::min<>{}) == n);
    }

    memory::array<HOST_MEMORY, ComplexType, 4> Pp, Ph, w_id;
    nda::array<ComplexType, 4> Sp, Sh, Sp_c, Sh_c;
    utils::TimerManager Tid;
    polarization<HOST_MEMORY>(prop, su.poles, mf, grid, zb, G.pi_p, G.pi_h, t_chunk, Pp, Tid, sector_t::particle);
    polarization<HOST_MEMORY>(prop, su.poles, mf, grid, zb, G.pi_p, G.pi_h, t_chunk, Ph, Tid, sector_t::hole);
    const double tPi = pi_time(Tid);
    // Sigma with the GL W residues: isolates the Sigma time grid
    self_energy<HOST_MEMORY>(prop, su.poles, w_gl, basis, mf, grid, mpi, fz, G.sig_p, G.sig_h, t_chunk, Sp, Tid,
                             sector_t::particle);
    self_energy<HOST_MEMORY>(prop, su.poles, w_gl, basis, mf, grid, mpi, fz, G.sig_p, G.sig_h, t_chunk, Sh, Tid,
                             sector_t::hole);
    const double tS = sigma_time(Tid);
    // the full ID chain: Pi_ID -> W -> Sigma_ID
    {
      memory::array<HOST_MEMORY, ComplexType, 4> Pi = Pp;
      Pi += Ph;
      screened_interaction<HOST_MEMORY>(Pi, Zb, basis, grid, mpi, w_id, Tw);
    }
    utils::TimerManager Tc;
    self_energy<HOST_MEMORY>(prop, su.poles, w_id, basis, mf, grid, mpi, fz, G.sig_p, G.sig_h, t_chunk, Sp_c, Tc,
                             sector_t::particle);
    self_energy<HOST_MEMORY>(prop, su.poles, w_id, basis, mf, grid, mpi, fz, G.sig_p, G.sig_h, t_chunk, Sh_c, Tc,
                             sector_t::hole);

    const double pp = rel(Pp, Pp_gl), ph = rel(Ph, Ph_gl);
    const double sp = max_diff3(Sp, Sp_gl) / max_abs3(Sp_gl), sh = max_diff3(Sh, Sh_gl) / max_abs3(Sh_gl);
    const double pxp = rel(Pp, Cp), pxh = rel(Ph, Ch);
    const double sxp = max_diff3(Sp_c, Rp) / max_abs3(Rp), sxh = max_diff3(Sh_c, Rh) / max_abs3(Rh);
    app_log(2, "  [time_id] {} eps_t {:.0e} ({} ranks): nodes Pi {} + {}, Sigma {} + {} (GL {} + {})", fixture, eps_t,
            comm.size(), G.pi_p.size(), G.pi_h.size(), G.sig_p.size(), G.sig_h.size(), ray_p.size(), ray_h.size());
    app_log(2, "    ID vs GL: Pi particle {:.2e}, hole {:.2e} | Sigma (same W) particle {:.2e}, hole {:.2e}   (gate 10 eps_t = {:.0e})",
            pp, ph, sp, sh, 10.0 * eps_t);
    app_log(2, "    ID vs exact: Pi particle {:.2e}, hole {:.2e} | Sigma (ID chain) particle {:.2e}, hole {:.2e}", pxp, pxh, sxp,
            sxh);
    app_log(2, "    kernel timers (rank 0) ID vs GL: Pi {:.3f} vs {:.3f} s (x{:.1f}; G_tilde {:.3f}/{:.3f} hadamard {:.3f}/{:.3f} "
               "transform {:.3f}/{:.3f}), Sigma {:.3f} vs {:.3f} s (x{:.1f}; W_time {:.3f}/{:.3f} hadamard {:.3f}/{:.3f} contract "
               "{:.3f}/{:.3f})",
            tPi, tPi_gl, tPi_gl / tPi, Tid.elapsed("G_tilde"), Tgl.elapsed("G_tilde"), Tid.elapsed("Pi_hadamard"),
            Tgl.elapsed("Pi_hadamard"), Tid.elapsed("Pi_transform"), Tgl.elapsed("Pi_transform"), tS, tS_gl, tS_gl / tS,
            Tid.elapsed("Sigma_W_time"), Tgl.elapsed("Sigma_W_time"), Tid.elapsed("Sigma_hadamard"),
            Tgl.elapsed("Sigma_hadamard"), Tid.elapsed("Sigma_contract"), Tgl.elapsed("Sigma_contract"));
    REQUIRE(pp <= 10.0 * eps_t);
    REQUIRE(ph <= 10.0 * eps_t);
    REQUIRE(sp <= 10.0 * eps_t);
    REQUIRE(sh <= 10.0 * eps_t);
    // the ID chain is as accurate as the GL chain w.r.t. the exact answer (up to the ID error)
    REQUIRE(sxp <= s_gl_ex_p + 100.0 * eps_t);
    REQUIRE(sxh <= s_gl_ex_h + 100.0 * eps_t);
  }
}

} // namespace

TEST_CASE("gw_line_time_id_lih222", "[gw_line][time_id]") { run_time_id("qe_lih222"); }

TEST_CASE("gw_line_time_id_si211", "[gw_line][time_id]") { run_time_id("qe_si211"); }
