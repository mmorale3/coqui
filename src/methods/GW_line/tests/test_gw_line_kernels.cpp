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
 * S3 of notes/line_gw_cpp_plan.md: GW_line kernels on the THC fixtures.
 *
 * [V1] Pi(q, zeta) at 12 mu-relative line points from KS poles (time-ray products + transform, Eqs. pigtr/pilss) vs the
 *      Casida transition sum of the same quantity (python oracle coqui/cayley/scripts/si222c_v012.py::pi_casida):
 *        Pi(q,z)_PQ = sum_t s_t S_Pt conj(S_Qt) / (z - E_t),  S_Pt = sqrt(2/N_k) X_Pn(k) conj(X_Pm(k-q)),  E_t = e_m(k-q) - e_n(k),
 *      over pairs with different occupation, s_t = +1 for n occupied at k (particle sector, E_t > 0), -1 otherwise (hole).
 *      Each rank checks its own (P,Q) block; with >1 rank the gathered (q=0, zeta_0) matrix is also compared with the
 *      1x1-grid result computed on every rank.
 * [V1-sym] (q != -q fixtures, lih223: 2x2x3 mesh, 8 of 12 q with q != -q) the symmetry of the bosonic propagator that the
 *      W fit uses (screened.hpp, notes section 3.3), on the exact transition sum (full matrices):
 *        Pi(q, -z) = Pi(-q, z)^T  (<= 1e-13)     and NOT  Pi(q, -z) = Pi(q, z)^T  (the pre-fix assumption; O(1) here),
 *      per sector: Pi^<(q, -z) = Pi^>(-q, z)^T, i.e. the hole residues of q are the transposed particle residues of -q.
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include <cmath>
#include <numbers>
#include <string>

#include "mpi3/communicator.hpp"
#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"

#include "mean_field/default_MF.hpp"
#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/thc_reader_t.hpp"

#include "numerics/line_dlr/time_ray.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/polarization.hpp"

namespace {

using namespace methods::gw_line;
using numerics::line_dlr::time_ray_t;
using numerics::line_dlr::sector_t;
namespace mpi3 = boost::mpi3;

/// Casida transition sum for the (P_rng, Q_rng) block, all q, restricted to one sector (python pi_casida).
nda::array<ComplexType, 4> pi_casida_block(methods::thc_reader_t &thc, mf::MF &mf, nda::array<double, 2> const &e_rel,
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
          if (sec == sector_t::particle and not occ_n) continue;
          if (sec == sector_t::hole and occ_n) continue;
          tk.push_back(ik); tn.push_back(n); tm.push_back(m);
          E.push_back(e_rel(ikmq, m) - e_rel(ik, n));
          sg.push_back(occ_n ? 1.0 : -1.0);
        }
    }
    const long T = E.size();
    if (T == 0) continue;
    nda::matrix<ComplexType> SP(g.nP, T), SQ(g.nQ, T), W(g.nP, T);
    for (long it = 0; it < T; ++it) {
      auto Xk = thc.X(0, 0, tk[it]);
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

template <typename A, typename B>
double max_abs_diff(A const &a, B const &b) {
  double d = 0.0;
  auto a1 = nda::reshape(a, std::array<long, 1>{a.size()});
  auto b1 = nda::reshape(b, std::array<long, 1>{b.size()});
  for (long i = 0; i < a1.size(); ++i) d = std::max(d, std::abs(a1(i) - b1(i)));
  return d;
}
template <typename A>
double max_abs(A const &a) {
  double d = 0.0;
  auto a1 = nda::reshape(a, std::array<long, 1>{a.size()});
  for (long i = 0; i < a1.size(); ++i) d = std::max(d, std::abs(a1(i)));
  return d;
}

void run_v1(std::string const &fixture, bool virtual_grid) {
  auto &mpi = utils::make_unit_test_mpi_context();
  auto &comm = mpi->comm;
  auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, fixture));
  utils::check(mf->nkpts() == mf->nkpts_ibz() and mf->nqpts() == mf->nqpts_ibz(),
               "[V1] {}: fixture must be a mesh without symmetry reduction (nkpts {} nkpts_ibz {})", fixture, mf->nkpts(),
               mf->nkpts_ibz());
  utils::check(mf->nspin() == 1 and mf->npol() == 1, "[V1] {}: spin-restricted collinear fixture required", fixture);
  methods::thc_reader_t thc(mf, methods::make_thc_reader_ptree(mf->nbnd() * 20, "", "incore", "", "bdft", 1e-10,
                                                               mf->ecutrho(), 1, 1024));
  const long nk = mf->nkpts(), nq = mf->nqpts(), nb = thc.nbnd(), Np = thc.Np();

  // KS poles; mu = midpoint of the KS gap from the mean-field occupations
  nda::array<double, 2> eig(nk, nb);
  double omax = 0.0;
  for (long ik = 0; ik < nk; ++ik)
    for (long n = 0; n < nb; ++n) {
      eig(ik, n) = mf->eigval()(0, ik, n);
      omax       = std::max(omax, double(mf->occ()(0, ik, n)));
    }
  double homo = -1e300, lumo = 1e300;
  for (long ik = 0; ik < nk; ++ik)
    for (long n = 0; n < nb; ++n) {
      if (mf->occ()(0, ik, n) > 0.5 * omax) homo = std::max(homo, eig(ik, n));
      else lumo = std::min(lumo, eig(ik, n));
    }
  utils::check(lumo > homo, "[V1] {}: no KS gap (homo {} lumo {})", fixture, homo, lumo);
  const double mu = 0.5 * (homo + lumo);
  auto poles      = pole_data_t::from_ks(eig, mu);
  nda::array<double, 2> e_rel(nk, nb);
  for (long ik = 0; ik < nk; ++ik)
    for (long n = 0; n < nb; ++n) e_rel(ik, n) = eig(ik, n) - mu;

  // rays (theta_t = 10 deg, 36 decades) and 12 mu-relative points on the two upper rays of the theta = 20 deg line
  const double deg = std::numbers::pi / 180.0, theta = 20.0 * deg, theta_t = 10.0 * deg;
  const double emin = poles.emin();
  auto ray_p = time_ray_t::for_spectrum(theta_t, emin, 36.0, 1e-5, 3.0, 16, sector_t::particle);
  auto ray_h = time_ray_t::for_spectrum(theta_t, emin, 36.0, 1e-5, 3.0, 16, sector_t::hole);
  auto r     = numerics::line_dlr::detail::logspace(0.05, 5.0, 6);
  nda::array<ComplexType, 1> zeta(12);
  for (long i = 0; i < 6; ++i) {
    zeta(i)     = r(i) * std::exp(ComplexType(0.0, theta));
    zeta(6 + i) = r(i) * std::exp(ComplexType(0.0, std::numbers::pi - theta));
  }
  const long t_chunk = 8;

  app_log(2, "\n[V1] {}: nk={} nq={} nb={} Np={} mu={:.6f} Ha, KS gap {:.4f} Ha, emin {:.4f} Ha, rays {} + {} nodes",
          fixture, nk, nq, nb, Np, mu, lumo - homo, emin, ray_p.size(), ray_h.size());
  aux_grid_t grid(*mpi, Np);
  grid.log(nk, nq, zeta.size(), 0, t_chunk, nb);

  // [Eq. gtilde] plain propagator block vs the direct sum X(k) diag(e^{-i e t}) X(k)^dagger (KS poles)
  {
    propagator_t<HOST_MEMORY> prop(thc, grid);
    prop.set_poles(poles);
    nda::array<ComplexType, 1> t(ray_p.t(nda::range(100, 103)));
    for (long ik : {0L, nk - 1}) {
      memory::array<HOST_MEMORY, ComplexType, 3> G(t.size(), grid.nP, grid.nQ);
      prop.gtilde(ik, t, sector_t::particle, G());
      auto X = thc.X(0, 0, ik);
      nda::array<ComplexType, 3> R(t.size(), grid.nP, grid.nQ);
      R() = 0.0;
      for (long it = 0; it < t.size(); ++it)
        for (long m = 0; m < nb; ++m) {
          if (e_rel(ik, m) <= 0.0) continue;
          const ComplexType ph = std::exp(ComplexType(0.0, -e_rel(ik, m)) * t(it));
          for (long P = 0; P < grid.nP; ++P)
            for (long Q = 0; Q < grid.nQ; ++Q) R(it, P, Q) += X(grid.P0 + P, m) * ph * std::conj(X(grid.Q0 + Q, m));
        }
      const double err = comm.all_reduce_value(max_abs_diff(G, R), mpi3::max<>{});
      const double ref = comm.all_reduce_value(max_abs(R), mpi3::max<>{});
      app_log(2, "  [gtilde] k={}: max|G~ - direct| = {:.2e} (rel {:.2e})", ik, err, err / ref);
      REQUIRE(err <= 1e-12 * ref);
    }
  }

  // Pi by sector on the line, and the Casida transition sums of the same blocks
  utils::TimerManager Timer;
  propagator_t<HOST_MEMORY> prop(thc, grid);
  memory::array<HOST_MEMORY, ComplexType, 4> Pp, Ph;
  polarization<HOST_MEMORY>(prop, poles, *mf, grid, zeta, ray_p, ray_h, t_chunk, Pp, Timer, sector_t::particle);
  polarization<HOST_MEMORY>(prop, poles, *mf, grid, zeta, ray_p, ray_h, t_chunk, Ph, Timer, sector_t::hole);
  utils::TimerManager Tc;
  Tc.start("casida");
  auto Cp = pi_casida_block(thc, *mf, e_rel, grid, zeta, sector_t::particle);
  auto Ch = pi_casida_block(thc, *mf, e_rel, grid, zeta, sector_t::hole);
  Tc.stop("casida");
  nda::array<ComplexType, 4> Pt = Pp + Ph, Ct = Cp + Ch;

  auto rel = [&](auto const &A, auto const &B) {
    const double d = comm.all_reduce_value(max_abs_diff(A, B), mpi3::max<>{});
    const double m = comm.all_reduce_value(max_abs(B), mpi3::max<>{});
    return std::array<double, 2>{d / m, m};
  };
  auto [rp, mp] = rel(Pp, Cp);
  auto [rh, mh] = rel(Ph, Ch);
  auto [rt, mt] = rel(Pt, Ct);
  app_log(2, "  [V1] {}: max|Pi - Casida| / max|Pi| over {} q x {} zeta, block {}x{} of {}x{} grid:", fixture, nq,
          zeta.size(), grid.nP, grid.nQ, grid.np_P, grid.np_Q);
  app_log(2, "       particle sector {:.2e} (max|Pi^>| {:.3e}),  hole sector {:.2e} (max|Pi^<| {:.3e}),  total {:.2e}", rp,
          mp, rh, mh, rt);
  app_log(2, "  [V1] {} timers (rank 0): G_tilde {:.3f} s, Pi_hadamard {:.3f} s, Pi_transform {:.3f} s; Casida reference {:.3f} s",
          fixture, Timer.elapsed("G_tilde"), Timer.elapsed("Pi_hadamard"), Timer.elapsed("Pi_transform"),
          Tc.elapsed("casida"));
  REQUIRE(rp <= 1e-10);
  REQUIRE(rh <= 1e-10);
  REQUIRE(rt <= 1e-10);

  // Block offsets in both P and Q: every block of a virtual 2x2 grid (all on this rank) vs its Casida block
  if (virtual_grid) {
    double worst = 0.0;
    for (long vr = 0; vr < 4; ++vr) {
      aux_grid_t gv(4, vr, Np);
      propagator_t<HOST_MEMORY> pv(thc, gv);
      memory::array<HOST_MEMORY, ComplexType, 4> Pv;
      utils::TimerManager Tv;
      polarization<HOST_MEMORY>(pv, poles, *mf, gv, zeta, ray_p, ray_h, t_chunk, Pv, Tv);
      nda::array<ComplexType, 4> Cv = pi_casida_block(thc, *mf, e_rel, gv, zeta, sector_t::particle) +
                                      pi_casida_block(thc, *mf, e_rel, gv, zeta, sector_t::hole);
      worst = std::max(worst, max_abs_diff(Pv, Cv) / mt);
    }
    app_log(2, "  [V1] {}: virtual 2x2 grid, all four blocks (P,Q offsets): max rel error {:.2e}", fixture, worst);
    REQUIRE(worst <= 1e-10);
  }

  // MPI consistency: gathered (q=0, zeta_0) of the distributed run vs the 1x1-grid run (computed on every rank)
  if (comm.size() > 1) {
    aux_grid_t g1(1, 0, Np);
    propagator_t<HOST_MEMORY> prop1(thc, g1);
    memory::array<HOST_MEMORY, ComplexType, 4> P1;
    utils::TimerManager T1;
    polarization<HOST_MEMORY>(prop1, poles, *mf, g1, zeta, ray_p, ray_h, t_chunk, P1, T1);
    nda::array<ComplexType, 2> full(Np, Np);
    full() = 0.0;
    full(grid.P_rng(), grid.Q_rng()) = Pt(0, 0, nda::range::all, nda::range::all);
    comm.all_reduce_in_place_n(full.data(), full.size(), std::plus<>{});
    nda::array<ComplexType, 2> one(P1(0, 0, nda::range::all, nda::range::all));
    const double d = max_abs_diff(full, one), m = max_abs(one);
    app_log(2, "  [V1] {}: {} ranks gathered vs 1x1 grid at (q=0, zeta_0): max|diff| {:.2e} (rel {:.2e})", fixture,
            comm.size(), d, d / m);
    REQUIRE(d <= 1e-14 * m);
  }

  // [V1-sym] symmetry of Pi under zeta -> -zeta, exact transition sums on the full matrices (1x1 grid)
  {
    aux_grid_t g1(1, 0, Np);
    nda::array<ComplexType, 1> mz(zeta.size());
    for (long i = 0; i < zeta.size(); ++i) mz(i) = -zeta(i);
    auto Pp_z  = pi_casida_block(thc, *mf, e_rel, g1, zeta, sector_t::particle);
    auto Ph_mz = pi_casida_block(thc, *mf, e_rel, g1, mz, sector_t::hole);
    nda::array<ComplexType, 4> P_z = Pp_z + pi_casida_block(thc, *mf, e_rel, g1, zeta, sector_t::hole);
    nda::array<ComplexType, 4> P_mz = pi_casida_block(thc, *mf, e_rel, g1, mz, sector_t::particle) + Ph_mz;
    auto qm = mf->qminus();
    double e_pair = 0.0, e_old = 0.0, e_sec = 0.0, m = 0.0;
    long n_self = 0;
    for (long iq = 0; iq < nq; ++iq) {
      n_self += (qm(iq) == iq) ? 1 : 0;
      for (long iz = 0; iz < zeta.size(); ++iz)
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            m      = std::max(m, std::abs(P_z(iq, iz, P, Q)));
            e_pair = std::max(e_pair, std::abs(P_mz(iq, iz, P, Q) - P_z(qm(iq), iz, Q, P)));
            e_old  = std::max(e_old, std::abs(P_mz(iq, iz, P, Q) - P_z(iq, iz, Q, P)));
            e_sec  = std::max(e_sec, std::abs(Ph_mz(iq, iz, P, Q) - Pp_z(qm(iq), iz, Q, P)));
          }
    }
    app_log(2, "  [V1-sym] {}: {} of {} q self-inverse; exact Pi: max|Pi(q,-z) - Pi(-q,z)^T| / max|Pi| = {:.2e} (sector form "
               "Pi^<(q,-z) vs Pi^>(-q,z)^T {:.2e});  max|Pi(q,-z) - Pi(q,z)^T| / max|Pi| = {:.2e} (the pre-fix per-q assumption)",
            fixture, n_self, nq, e_pair / m, e_sec / m, e_old / m);
    REQUIRE(e_pair <= 1e-13 * m);
    REQUIRE(e_sec <= 1e-13 * m);
    if (n_self == nq) REQUIRE(e_old <= 1e-10 * m);   // q = -q meshes: both forms agree (time reversal of the KS states)
  }
}

} // namespace

TEST_CASE("gw_line_V1_lih222", "[gw_line][V1]") { run_v1("qe_lih222", false); }

TEST_CASE("gw_line_V1_si211", "[gw_line][V1]") { run_v1("qe_si211", true); }

// q != -q mesh (2x2x3, 8 of 12 q not self-inverse): the W pairing fix (notes section 3.3)
TEST_CASE("gw_line_V1_lih223", "[gw_line][V1]") { run_v1("qe_lih223", false); }
