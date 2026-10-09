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
 * Performance campaign 7.1 (notes/line_gw_cpp_plan.md section 7.1): the exact symmetry relations the fast kernels use,
 * verified on the exact references (transition sums, Casida) of the THC fixtures lih222 / si211 (q = -q for every q) and
 * lih223 (2x2x3, 8 of 12 q with q != -q), before any of them is used:
 *
 *  [.perf71_rel] (hidden, report + gates)
 *   (a)  Pi^<(q, zeta)_PQ = conj(Pi^>(-q, -conj zeta)_PQ)      exact transition sums at 12 points on both rays, and the
 *        line kernel (explicit hole sector vs the conjugated particle sector at the mirror points)
 *   (b)  W(-q, -conj zeta) = conj(W(q, zeta))                   exact Casida W_dyn; Z(-q) vs conj Z(q) and Z(q)^T, Z = Z^dagger
 *   (c)  W(-q, -conj zeta)^T = W(q, zeta)^dagger                 the mirror block of the pair fit (Casida)
 *   (t)  the time transforms: F_hole(zeta) = -conj(F_particle(-conj zeta)) for the conjugated ray (GL and ID nodes)
 *   (R)  k-mesh convolutions by the Fourier matrices of kmesh_ft_t vs the direct k sums with qk_to_k2 / qminus (random
 *        blocks): sum_k A(k) B(k-q), sum_q G(k-q) W(q), sum_q' G(k+q') V(q')
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <numbers>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "mpi3/communicator.hpp"
#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"

#include "mean_field/default_MF.hpp"
#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/thc_reader_t.hpp"

#include "numerics/line_dlr/time_ray.hpp"
#include "numerics/line_dlr/time_id.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/polarization.hpp"
#include "methods/GW_line/kmesh_ft.hpp"
#include "methods/GW_line/kmesh_fft.hpp"
#include "methods/GW_line/screened.hpp"
#include "methods/GW_line/self_energy.hpp"
#include "numerics/line_dlr/bosonic_basis.hpp"
#include "gw_line_casida_ref.hpp"

namespace {

using namespace methods::gw_line;
using numerics::line_dlr::time_ray_t;
using numerics::line_dlr::time_id_t;
using numerics::line_dlr::sector_t;
namespace mpi3 = boost::mpi3;
using cmat = nda::matrix<ComplexType>;
using namespace gw_line_test;

/// exact transition sum of one sector, full matrices, all q (as pi_casida_block of test_gw_line_kernels.cpp)
nda::array<ComplexType, 4> pi_exact(methods::thc_reader_t &thc, mf::MF &mf, nda::array<double, 2> const &e_rel,
                                    nda::array<ComplexType, 1> const &zeta, sector_t sec) {
  const long nk = mf.nkpts(), nq = mf.nqpts(), nb = e_rel.extent(1), nz = zeta.size(), Np = thc.Np();
  auto qk       = mf.qk_to_k2();
  nda::array<ComplexType, 4> out(nq, nz, Np, Np);
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
    cmat S(Np, T), W(Np, T);
    for (long it = 0; it < T; ++it) {
      auto Xk   = thc.X(0, 0, tk[it]);
      auto Xkmq = thc.X(0, 0, qk(iq, tk[it]));
      for (long P = 0; P < Np; ++P) S(P, it) = nrm * Xk(P, tn[it]) * std::conj(Xkmq(P, tm[it]));
    }
    for (long iz = 0; iz < nz; ++iz) {
      for (long it = 0; it < T; ++it) {
        const ComplexType f = sg[it] / (zeta(iz) - E[it]);
        for (long P = 0; P < Np; ++P) W(P, it) = S(P, it) * f;
      }
      nda::matrix_view<ComplexType> o(out(iq, iz, nda::range::all, nda::range::all));
      nda::blas::gemm(ComplexType(1.0), W, nda::dagger(S), ComplexType(0.0), o);
    }
  }
  return out;
}

struct fixture_t {
  std::shared_ptr<mf::MF> mf;
  std::unique_ptr<methods::thc_reader_t> thc;
  long nk = 0, nq = 0, nb = 0, Np = 0;
  double mu = 0.0, ks_gap = 0.0;
  pole_data_t poles;
  nda::array<double, 2> e_rel;
};

fixture_t make_fixture(std::string const &name, long nI_factor) {
  auto &mpi = utils::make_unit_test_mpi_context();
  fixture_t f;
  f.mf = std::make_shared<mf::MF>(mf::default_MF(mpi, name));
  f.thc = std::make_unique<methods::thc_reader_t>(
      f.mf, methods::make_thc_reader_ptree(f.mf->nbnd() * nI_factor, "", "incore", "", "bdft", 1e-10, f.mf->ecutrho(), 1, 1024));
  f.nk = f.mf->nkpts();
  f.nq = f.mf->nqpts();
  f.nb = f.thc->nbnd();
  f.Np = f.thc->Np();
  nda::array<double, 2> eig(f.nk, f.nb);
  double omax = 0.0;
  for (long ik = 0; ik < f.nk; ++ik)
    for (long n = 0; n < f.nb; ++n) {
      eig(ik, n) = f.mf->eigval()(0, ik, n);
      omax       = std::max(omax, double(f.mf->occ()(0, ik, n)));
    }
  double homo = -1e300, lumo = 1e300;
  for (long ik = 0; ik < f.nk; ++ik)
    for (long n = 0; n < f.nb; ++n) {
      if (f.mf->occ()(0, ik, n) > 0.5 * omax) homo = std::max(homo, eig(ik, n));
      else lumo = std::min(lumo, eig(ik, n));
    }
  f.mu     = 0.5 * (homo + lumo);
  f.ks_gap = lumo - homo;
  f.poles  = pole_data_t::from_ks(eig, f.mu);
  f.e_rel  = nda::array<double, 2>(f.nk, f.nb);
  for (long ik = 0; ik < f.nk; ++ik)
    for (long n = 0; n < f.nb; ++n) f.e_rel(ik, n) = eig(ik, n) - f.mu;
  return f;
}

/// 6 points per upper ray at the same radii (the ray-2 point i + 6 is the mirror -conj of the ray-1 point i)
nda::array<ComplexType, 1> twelve_points(double theta) {
  auto r = numerics::line_dlr::detail::logspace(0.05, 5.0, 6);
  nda::array<ComplexType, 1> z(12);
  for (long i = 0; i < 6; ++i) {
    z(i)     = r(i) * std::exp(ComplexType(0.0, theta));
    z(6 + i) = r(i) * std::exp(ComplexType(0.0, std::numbers::pi - theta));
  }
  return z;
}

void run_relations(std::string const &name, long nI_factor) {
  auto &mpi  = utils::make_unit_test_mpi_context();
  auto &comm = mpi->comm;
  auto f     = make_fixture(name, nI_factor);
  auto &mf   = *f.mf;
  auto &thc  = *f.thc;
  const long nq = f.nq, Np = f.Np;
  auto qm       = mf.qminus();
  long n_self   = 0;
  for (long q = 0; q < nq; ++q) n_self += (qm(q) == q) ? 1 : 0;
  const double deg = std::numbers::pi / 180.0, theta = 20.0 * deg, theta_t = 10.0 * deg;
  auto z  = twelve_points(theta);
  const long nz = z.size();
  nda::array<ComplexType, 1> mz(nz);   // -conj z
  for (long i = 0; i < nz; ++i) mz(i) = -std::conj(z(i));
  app_log(2, "\n[perf71 relations] {}: nk {} nq {} ({} self-inverse) nb {} Np {} (nIpts {} nbnd), mu {:.6f}, KS gap {:.4f} Ha",
          name, f.nk, nq, n_self, f.nb, Np, nI_factor, f.mu, f.ks_gap);

  // (a) exact: Pi^<(q, z) vs conj(Pi^>(-q, -conj z))
  {
    auto Pp_m = pi_exact(thc, mf, f.e_rel, mz, sector_t::particle);
    auto Ph   = pi_exact(thc, mf, f.e_rel, z, sector_t::hole);
    double e = 0.0, s = 0.0, e_old = 0.0;
    for (long q = 0; q < nq; ++q)
      for (long i = 0; i < nz; ++i)
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            s     = std::max(s, std::abs(Ph(q, i, P, Q)));
            e     = std::max(e, std::abs(Ph(q, i, P, Q) - std::conj(Pp_m(qm(q), i, P, Q))));
            e_old = std::max(e_old, std::abs(Ph(q, i, P, Q) - std::conj(Pp_m(q, i, P, Q))));
          }
    app_log(2, "  (a) exact transition sums: max|Pi^<(q,z) - conj Pi^>(-q,-conj z)| / max|Pi^<| = {:.2e}  (same q: {:.2e})", e / s,
            e_old / s);
    REQUIRE(e <= 1e-13 * s);
  }
  // (a) line kernel: explicit hole sector vs the conjugated particle sector at the mirror points (GL rays, 1x1 grid)
  {
    aux_grid_t g1(1, 0, Np);
    auto ray_p = time_ray_t::for_spectrum(theta_t, f.poles.emin(), 36.0, 1e-5, 3.0, 16, sector_t::particle);
    auto ray_h = time_ray_t::for_spectrum(theta_t, f.poles.emin(), 36.0, 1e-5, 3.0, 16, sector_t::hole);
    propagator_t<HOST_MEMORY> prop(thc, g1);
    utils::TimerManager T;
    memory::array<HOST_MEMORY, ComplexType, 4> Pp_m, Ph;
    ::setenv("COQUI_GWLINE_PI_MIRROR", "0", 1);   // the explicit two-sector path
    polarization<HOST_MEMORY>(prop, f.poles, mf, g1, mz, ray_p, ray_h, 8, Pp_m, T, sector_t::particle);
    polarization<HOST_MEMORY>(prop, f.poles, mf, g1, z, ray_p, ray_h, 8, Ph, T, sector_t::hole);
    ::unsetenv("COQUI_GWLINE_PI_MIRROR");
    double e = 0.0, s = 0.0;
    for (long q = 0; q < nq; ++q)
      for (long i = 0; i < nz; ++i)
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            s = std::max(s, std::abs(Ph(q, i, P, Q)));
            e = std::max(e, std::abs(Ph(q, i, P, Q) - std::conj(Pp_m(qm(q), i, P, Q))));
          }
    app_log(2, "  (a) line kernel (GL rays): max|Pi^<_line(q,z) - conj Pi^>_line(-q,-conj z)| / max = {:.2e}", e / s);
    REQUIRE(e <= 1e-12 * s);
  }

  // (b), (c): Casida W_dyn of all q (lockstep: thc.Z is collective)
  {
    std::vector<casida_t> cas;
    std::vector<cmat> Z;
    for (long q = 0; q < nq; ++q) {
      Z.emplace_back(thc.Z(int(q)));
      cas.push_back(casida_q(thc, mf, f.e_rel, q, Z.back()));
    }
    aux_grid_t g1(1, 0, Np);
    double eb = 0.0, ec = 0.0, sw = 0.0, ez_c = 0.0, ez_t = 0.0, ez_h = 0.0, sz = 0.0, ec_same = 0.0;
    for (long q = 0; q < nq; ++q) {
      const long jq = qm(q);
      auto Wq  = casida_eval(cas[q], g1, z, false);    // W(q, z)
      auto Wmm = casida_eval(cas[jq], g1, mz, false);  // W(-q, -conj z)
      auto Wsm = casida_eval(cas[q], g1, mz, false);   // W(q, -conj z) (the per-q form, info)
      for (long i = 0; i < nz; ++i)
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            sw      = std::max(sw, std::abs(Wq(i, P, Q)));
            eb      = std::max(eb, std::abs(Wmm(i, P, Q) - std::conj(Wq(i, P, Q))));
            ec      = std::max(ec, std::abs(Wmm(i, Q, P) - std::conj(Wq(i, Q, P))));   // W(-q,-conj z)^T vs W(q,z)^dagger
            ec_same = std::max(ec_same, std::abs(Wsm(i, P, Q) - std::conj(Wq(i, P, Q))));
          }
      for (long P = 0; P < Np; ++P)
        for (long Q = 0; Q < Np; ++Q) {
          sz   = std::max(sz, std::abs(Z[q](P, Q)));
          ez_c = std::max(ez_c, std::abs(Z[jq](P, Q) - std::conj(Z[q](P, Q))));
          ez_t = std::max(ez_t, std::abs(Z[jq](P, Q) - Z[q](Q, P)));
          ez_h = std::max(ez_h, std::abs(Z[q](P, Q) - std::conj(Z[q](Q, P))));
        }
    }
    app_log(2, "  (b) Casida: max|W(-q,-conj z) - conj W(q,z)| / max|W| = {:.2e}  (same q, the q=-q form: {:.2e})", eb / sw,
            ec_same / sw);
    app_log(2, "  (c) Casida: max|W(-q,-conj z)^T - W(q,z)^dagger| / max|W| = {:.2e}", ec / sw);
    app_log(2, "      Z: max|Z(-q) - conj Z(q)| / max|Z| = {:.2e}, |Z(-q) - Z(q)^T| {:.2e}, |Z - Z^dagger| {:.2e}", ez_c / sz,
            ez_t / sz, ez_h / sz);
    // the relation holds to the THC asymmetry of Z (Z(-q) vs conj Z(q): 1e-12 lih, 1.7e-9 si211)
    REQUIRE(eb <= (1e-11 + 2.0 * ez_c / sz) * sw);
    REQUIRE(ec <= (1e-11 + 2.0 * ez_c / sz) * sw);
  }

  // (t) the conjugated ray: F_hole(z) = -conj(F_particle(-conj z)) (GL rays; ID nodes built independently per sector)
  {
    auto ray_p = time_ray_t::for_spectrum(theta_t, f.poles.emin(), 36.0, 1e-5, 3.0, 16, sector_t::particle);
    auto ray_h = time_ray_t::for_spectrum(theta_t, f.poles.emin(), 36.0, 1e-5, 3.0, 16, sector_t::hole);
    auto Fp = ray_p.transform_matrix(mz), Fh = ray_h.transform_matrix(z);
    double e = 0.0, s = 0.0;
    for (long i = 0; i < Fh.extent(0); ++i)
      for (long m = 0; m < Fh.extent(1); ++m) {
        s = std::max(s, std::abs(Fh(i, m)));
        e = std::max(e, std::abs(Fh(i, m) + std::conj(Fp(i, m))));
      }
    numerics::line_dlr::time_id_opts_t o;
    o.pad = 1.25;
    time_id_t idp(theta_t, sector_t::particle, 0.05, 4.0, 1e-10, o), idh(theta_t, sector_t::hole, 0.05, 4.0, 1e-10, o);
    double et = 0.0, ds = 0.0, ei = 0.0, si = 0.0;
    utils::check(idp.size() == idh.size(), "(t) particle / hole ID node counts differ ({} vs {})", idp.size(), idh.size());
    for (long j = 0; j < idp.size(); ++j) {
      ds = std::max(ds, std::abs(idp.s(j) - idh.s(j)));
      et = std::max(et, std::abs(idh.t(j) - std::conj(idp.t(j))));
    }
    auto Ip = idp.transform_matrix(mz), Ih = idh.transform_matrix(z);
    for (long i = 0; i < Ih.extent(0); ++i)
      for (long m = 0; m < Ih.extent(1); ++m) {
        si = std::max(si, std::abs(Ih(i, m)));
        ei = std::max(ei, std::abs(Ih(i, m) + std::conj(Ip(i, m))));
      }
    app_log(2, "  (t) GL: max|F_h(z) + conj F_p(-conj z)| / max = {:.2e};  ID ({} nodes): |s_h - s_p| {:.1e}, |t_h - conj t_p| {:.1e}, "
               "|F_h(z) + conj F_p(-conj z)| / max {:.2e}",
            e / s, idp.size(), ds, et, ei / si);
    REQUIRE(e <= 1e-14 * s);
  }

  // (R) k-mesh convolutions in real space vs the direct sums (random blocks, 1x1 block of size 3 x 2)
  {
    kmesh_ft_t ft(mf);
    REQUIRE(ft.ok);
    const long nk = f.nk, n = 6;
    std::mt19937_64 gen(1234);
    std::normal_distribution<double> N01;
    nda::array<ComplexType, 2> A(nk, n), B(nk, n);
    for (auto &v : A) v = ComplexType(N01(gen), N01(gen));
    for (auto &v : B) v = ComplexType(N01(gen), N01(gen));
    auto qk = mf.qk_to_k2();
    // direct sums
    nda::array<ComplexType, 2> P1(nq, n), S1(nk, n), H1(nk, n);
    P1() = 0.0; S1() = 0.0; H1() = 0.0;
    for (long q = 0; q < nq; ++q)
      for (long k = 0; k < nk; ++k)
        for (long e = 0; e < n; ++e) {
          P1(q, e) += A(k, e) * B(qk(q, k), e);              // sum_k A(k) B(k-q)
          S1(k, e) += A(qk(q, k), e) * B(q, e);              // sum_q G(k-q) W(q)   (G = A, W = B)
          H1(k, e) += A(qk(qm(q), k), e) * B(q, e);          // sum_q' G(k+q') V(q')
        }
    // real space
    nda::array<ComplexType, 2> Ah(nk, n), Bh(nk, n), P2(nq, n), S2(nk, n), H2(nk, n), X(nk, n);
    nda::blas::gemm(ComplexType(1.0), ft.Fp, A, ComplexType(0.0), Ah);   // sum_k e^{+ikR} A(k)
    nda::blas::gemm(ComplexType(1.0), ft.Fm, B, ComplexType(0.0), Bh);   // sum_k e^{-ikR} B(k)
    X = Ah * Bh;
    nda::blas::gemm(ComplexType(1.0 / nk), ft.Gm, X, ComplexType(0.0), P2);   // sum_R e^{-iqR} / N
    nda::array<ComplexType, 2> Gh(nk, n), Wh(nk, n), Vh(nk, n);
    nda::blas::gemm(ComplexType(1.0), ft.Fm, A, ComplexType(0.0), Gh);     // sum_k e^{-ikR} G(k)
    nda::blas::gemm(ComplexType(1.0), ft.Hm, B, ComplexType(0.0), Wh);     // sum_q e^{-iqR} W(q)
    X = Gh * Wh;
    nda::blas::gemm(ComplexType(1.0 / nk), ft.Bp, X, ComplexType(0.0), S2);   // sum_R e^{+ikR} / N
    // V^(R) = sum_q e^{+iqR} V(q) = W^(-R) of the e^{-iqR} transform
    for (long R = 0; R < nk; ++R) Vh(R, nda::range::all) = Wh(ft.minusR[R], nda::range::all);
    X = Gh * Vh;
    nda::blas::gemm(ComplexType(1.0 / nk), ft.Bp, X, ComplexType(0.0), H2);
    auto rel = [](auto const &a, auto const &b) { return max_diff3(a, b) / max_abs3(b); };
    app_log(2, "  (R) k-mesh {}x{}x{}: real-space vs direct: Pi-type {:.1e}, Sigma^> type {:.1e}, Sigma^< type {:.1e}; "
               "mesh closure {:.1e}, |e^{{iq(-R)}} - conj e^{{iqR}}| {:.1e}",
            ft.mesh[0], ft.mesh[1], ft.mesh[2], rel(P2, P1), rel(S2, S1), rel(H2, H1), ft.closure_err, ft.minusR_err);
    REQUIRE(rel(P2, P1) <= 1e-13);
    REQUIRE(rel(S2, S1) <= 1e-13);
    REQUIRE(rel(H2, H1) <= 1e-13);
  }
  comm.barrier();
}

/**
 * [.perf71_nodes] node oversampling of the pair fit (perf 7.1 (c)): exact Casida W_dyn(q) at the bosonic nodes (both rays),
 * paired fit (bosonic_basis_t::fit(zeta, W(q), W(-q))), pole sums vs Casida at 12 ray points, 40 imaginary-axis points and
 * 60 points on a 5x denser line grid; for the python QR nodes (2 r, asymmetric) and mirror-symmetric sets of nz = f r nodes.
 * Basis as [V2]: theta 20 deg, lam_b = max(4, 1.2 x largest KS transition), gap = KS gap / 2, eps 1e-10.
 */
void run_nodes(std::string const &name, long nI_factor) {
  auto &mpi = utils::make_unit_test_mpi_context();
  auto f    = make_fixture(name, nI_factor);
  auto &mf  = *f.mf;
  auto &thc = *f.thc;
  const long nq = f.nq, Np = f.Np;
  auto qm = mf.qminus();
  double emin_occ = 1e300, emax_vir = -1e300;
  for (long ik = 0; ik < f.nk; ++ik)
    for (long n = 0; n < f.nb; ++n) {
      if (f.e_rel(ik, n) < 0) emin_occ = std::min(emin_occ, f.e_rel(ik, n));
      else emax_vir = std::max(emax_vir, f.e_rel(ik, n));
    }
  const double deg = std::numbers::pi / 180.0, theta = 20.0 * deg;
  const double lam_b = std::max(4.0, 1.2 * (emax_vir - emin_occ)), gap_b = 0.5 * f.ks_gap;
  std::vector<casida_t> cas;
  for (long q = 0; q < nq; ++q) {
    cmat Z(thc.Z(int(q)));
    cas.push_back(casida_q(thc, mf, f.e_rel, q, Z));
  }
  aux_grid_t g1(1, 0, Np);
  auto z12 = twelve_points(theta);
  nda::array<ComplexType, 1> zi(40);
  {
    auto r = numerics::line_dlr::detail::logspace(1e-3, 50.0, 40);
    for (long i = 0; i < 40; ++i) zi(i) = ComplexType(0.0, r(i));
  }
  auto zd = numerics::line_dlr::dense_nodes(theta, 1e-3, 30.0, 30);
  app_log(2, "\n[perf71 nodes] {}: nq {} Np {}, lam_b {:.3f}, gap {:.4f}", name, nq, Np, lam_b, gap_b);
  for (double nf : {0.0, 1.0, 1.1, 1.25, 1.5, 2.0}) {
    numerics::line_dlr::bosonic_basis_t b(theta, lam_b, 1e-10, gap_b, -1.0, -1.0, 1200, 800, nf);
    auto const &zeta = b.zeta_nodes;
    double e12 = 0.0, s12 = 0.0, ei = 0.0, si = 0.0, ed = 0.0, sd = 0.0, cond = 0.0;
    {
      methods::gw_line::bosonic_fit_t bf(b, zeta);
      cond = double(bf.k);
    }
    for (long q = 0; q < nq; ++q) {
      const long jq = qm(q);
      auto W  = casida_eval(cas[q], g1, zeta, false);
      auto Wm = casida_eval(cas[jq], g1, zeta, false);
      auto w  = b.fit(zeta, W, Wm);
      auto wm = b.fit(zeta, Wm, W);
      for (auto [zz, e, sc] : {std::tuple{&z12, &e12, &s12}, std::tuple{&zi, &ei, &si}, std::tuple{&zd, &ed, &sd}}) {
        auto F = b.eval(w, wm, *zz);
        auto C = casida_eval(cas[q], g1, *zz, false);
        *e = std::max(*e, max_diff3(F, C));
        *sc = std::max(*sc, max_abs3(C));
      }
    }
    app_log(2, "  {} nodes ({}, nz/r = {:.2f}; n1 {}): rank {}, kept singular values {}: max rel error vs Casida W: 12 ray points "
               "{:.2e}, imaginary axis {:.2e}, dense line {:.2e}",
            zeta.size(), nf > 0 ? "mirror-symmetric" : "python QR", double(zeta.size()) / b.rank, b.n_mirror, b.rank, long(cond),
            e12 / s12, ei / si, ed / sd);
  }
}

/// scoped environment switch (restored on exit)
struct env_scope_t {
  std::string nm, old;
  bool had = false;
  env_scope_t(char const *n, char const *v) : nm(n) {
    if (char const *o = std::getenv(n)) { had = true; old = o; }
    ::setenv(n, v, 1);
  }
  ~env_scope_t() { if (had) ::setenv(nm.c_str(), old.c_str(), 1); else ::unsetenv(nm.c_str()); }
};

/**
 * [gw_line][perf71] A/B gates of the perf 7.1 paths against the previous ones on the same inputs (KS poles, GL rays of
 * ~250 nodes, the mirror-symmetric bosonic nodes, the process grid of the test):
 *   Pi : mirror (a) + real space (e)  vs  explicit hole leg + k-space Hadamards           <= 1e-12 (relative, max norm)
 *   W  : mirror stage (b)(c)          vs  the full-node stage (Dyson at every node, pair pass), as W at the nodes and as
 *        pole sums at 12 ray points   <= 1e-11 + 10 x the THC asymmetry of Z (ray-2 values use Z(-q))
 *   Sigma (both sectors, one call): real space vs k-space Hadamards (same residues)        <= 1e-12
 */
void run_ab(std::string const &name, long nI_factor) {
  auto &mpi  = utils::make_unit_test_mpi_context();
  auto &comm = mpi->comm;
  auto f     = make_fixture(name, nI_factor);
  auto &mf   = *f.mf;
  auto &thc  = *f.thc;
  const long nq = f.nq, Np = f.Np;
  const double deg = std::numbers::pi / 180.0, theta = 20.0 * deg, theta_t = 10.0 * deg;
  double emin_occ = 1e300, emax_vir = -1e300;
  for (long ik = 0; ik < f.nk; ++ik)
    for (long n = 0; n < f.nb; ++n) {
      if (f.e_rel(ik, n) < 0) emin_occ = std::min(emin_occ, f.e_rel(ik, n));
      else emax_vir = std::max(emax_vir, f.e_rel(ik, n));
    }
  numerics::line_dlr::bosonic_basis_t basis(theta, std::max(4.0, 1.2 * (emax_vir - emin_occ)), 1e-10, 0.5 * f.ks_gap);
  REQUIRE(numerics::line_dlr::mirror_half(basis.zeta_nodes) > 0);
  const double smax = 30.0 / (f.poles.emin() * std::sin(theta_t));
  time_ray_t ray_p(theta_t, smax, 1e-5, 1.5, 12, sector_t::particle), ray_h(theta_t, smax, 1e-5, 1.5, 12, sector_t::hole);
  aux_grid_t grid(*mpi, Np);
  propagator_t<HOST_MEMORY> prop(thc, grid);
  auto rel = [&](auto const &a, auto const &b) {
    const double d = comm.all_reduce_value(max_diff3(a, b), mpi3::max<>{});
    const double m = comm.all_reduce_value(max_abs3(b), mpi3::max<>{});
    return d / m;
  };
  utils::TimerManager T;
  // Pi
  memory::array<HOST_MEMORY, ComplexType, 4> Pn, Po;
  polarization<HOST_MEMORY>(prop, f.poles, mf, grid, basis.zeta_nodes, ray_p, ray_h, 8, Pn, T);
  {
    env_scope_t e1("COQUI_GWLINE_PI_MIRROR", "0"), e2("COQUI_GWLINE_RSPACE", "0");
    polarization<HOST_MEMORY>(prop, f.poles, mf, grid, basis.zeta_nodes, ray_p, ray_h, 8, Po, T);
  }
  const double e_pi = rel(Pn, Po);
  // W
  dyson_layout_t lay(*mpi, nq, basis.zeta_nodes.size(), Np);
  coulomb_blocks_t<HOST_MEMORY> Zb(thc, grid, lay.q_rng(), T);
  double ez = 0.0, sz = 0.0;
  {
    auto qm = mf.qminus();
    for (long q = 0; q < nq; ++q)
      for (long P = 0; P < grid.nP; ++P)
        for (long Q = 0; Q < grid.nQ; ++Q) {
          sz = std::max(sz, std::abs(Zb.Z(q, P, Q)));
          ez = std::max(ez, std::abs(Zb.Z(qm(q), P, Q) - std::conj(Zb.Z(q, P, Q))));
        }
    ez = comm.all_reduce_value(ez, mpi3::max<>{}) / comm.all_reduce_value(sz, mpi3::max<>{});
  }
  memory::array<HOST_MEMORY, ComplexType, 4> Pc = Pn, wn, wo, Wnn, Wno;
  screened_interaction<HOST_MEMORY>(Pc, Zb, basis, grid, *mpi, wn, T, &Wnn);
  Pc = Pn;
  {
    env_scope_t e1("COQUI_GWLINE_W_MIRROR", "0");
    screened_interaction<HOST_MEMORY>(Pc, Zb, basis, grid, *mpi, wo, T, &Wno);
  }
  const double e_wn = rel(Wnn, Wno);
  auto z12 = twelve_points(theta);
  double e_w12 = 0.0;
  {
    auto qm = mf.qminus();
    for (long q = 0; q < nq; ++q)
      for (auto s : {sector_t::particle, sector_t::hole}) {
        memory::array<HOST_MEMORY, ComplexType, 3> A(12, grid.nP, grid.nQ), B(12, grid.nP, grid.nQ);
        eval_poles<HOST_MEMORY>(wn, basis, q, qm(q), z12, s, s == sector_t::hole, A());
        eval_poles<HOST_MEMORY>(wo, basis, q, qm(q), z12, s, s == sector_t::hole, B());
        e_w12 = std::max(e_w12, rel(A, B));
      }
  }
  // Sigma, both sectors in one call (real space) vs the k-space Hadamards
  auto fz = numerics::line_dlr::dense_nodes(theta, 1e-3, 60.0, 30);
  nda::array<ComplexType, 4> Sp, Sh, Sp0, Sh0;
  self_energy<HOST_MEMORY>(prop, f.poles, wn, basis, mf, grid, *mpi, fz, ray_p, ray_h, 8, Sp, T, sector_t::both, false, nullptr, 0,
                           &Sh);
  {
    env_scope_t e1("COQUI_GWLINE_RSPACE", "0");
    self_energy<HOST_MEMORY>(prop, f.poles, wn, basis, mf, grid, *mpi, fz, ray_p, ray_h, 8, Sp0, T, sector_t::particle);
    self_energy<HOST_MEMORY>(prop, f.poles, wn, basis, mf, grid, *mpi, fz, ray_p, ray_h, 8, Sh0, T, sector_t::hole);
  }
  // the Sigma^< leg above used the A^ cache of the polarization call (G^ = conj A^ on the conjugated ray); without it:
  const bool cache_used = (prop.ahat_key >= 0.0 and prop.ahat_key == prop.pole_key);
  nda::array<ComplexType, 4> Sp1, Sh1;
  {
    env_scope_t e1("COQUI_GWLINE_GT_CACHE", "0");
    self_energy<HOST_MEMORY>(prop, f.poles, wn, basis, mf, grid, *mpi, fz, ray_p, ray_h, 8, Sp1, T, sector_t::both, false, nullptr,
                             0, &Sh1);
  }
  const double e_cache = max_diff3(Sh, Sh1) / max_abs3(Sh1);
  app_log(2, "[perf71 A/B] {}: Sigma^< with the cached A^ of Pi vs rebuilt G~: {:.2e} (cache valid: {})", name, e_cache, cache_used);
  REQUIRE(cache_used);
  REQUIRE(e_cache <= 1e-14);
  const double e_sp = max_diff3(Sp, Sp0) / max_abs3(Sp0), e_sh = max_diff3(Sh, Sh0) / max_abs3(Sh0);
  app_log(2, "\n[perf71 A/B] {} ({} ranks, grid {}x{}, {} nodes = 2 x {} mirror, {} time nodes): Pi new vs old {:.2e}; W at the nodes "
             "{:.2e}, pole sums at 12 points {:.2e} (THC asymmetry of Z {:.1e}); Sigma^> {:.2e}, Sigma^< {:.2e}",
          name, comm.size(), grid.np_P, grid.np_Q, basis.zeta_nodes.size(), basis.zeta_nodes.size() / 2, ray_p.size(), e_pi, e_wn,
          e_w12, ez, e_sp, e_sh);
  REQUIRE(e_pi <= 1e-12);
  REQUIRE(e_wn <= 1e-11 + 10.0 * ez);
  REQUIRE(e_w12 <= 1e-10 + 10.0 * ez);
  REQUIRE(e_sp <= 1e-12);
  REQUIRE(e_sh <= 1e-12);
}

/**
 * [gw_line][kft] perf 7.5c: the k-mesh transforms as 3-D FFTs (kmesh_fft.hpp) vs the dense Fourier gemms of perf 7.1, same
 * inputs (KS poles, GL rays, mirror nodes, process grid of the test):
 *   map    the FFT phases vs kmesh_ft_t's Fp / conj Hm (checked at construction, <= 1e-12) and the five transforms of
 *          kmesh_ft.hpp on random blocks through the host engine (placement, sign, scale) vs the gemms     <= 1e-14
 *   Pi     (mirror, real space) fft vs gemm                                                                  <= 1e-13
 *   Sigma  both legs in one call, same residues w: with the A^ cache of each mode's Pi, with rebuilt G~ (GT_CACHE = 0)
 *          and with the residues transformed in place (w_consume)                                            <= 1e-13 (si211 2e-13)
 * Device builds: the same on the device (cuFFT) vs the host gemm path.
 */
void run_kft(std::string const &name, long nI_factor) {
  auto &mpi  = utils::make_unit_test_mpi_context();
  auto &comm = mpi->comm;
  auto f     = make_fixture(name, nI_factor);
  auto &mf   = *f.mf;
  auto &thc  = *f.thc;
  const long nq = f.nq, Np = f.Np;
  // map and the raw transforms
  const kmesh_ft_t kft(mf);
  const kmesh_map_t km(kft, mf);
  REQUIRE(kft.ok);
  REQUIRE(km.ok);
  double e_tr = 0.0;
  {
    const long N = km.N, nc = 37;
    std::mt19937_64 gen(11);
    std::normal_distribution<double> N01;
    nda::array<ComplexType, 2> X(N, nc), Y(N, nc), Z(N, nc);
    for (long i = 0; i < N; ++i)
      for (long c = 0; c < nc; ++c) X(i, c) = ComplexType(N01(gen), N01(gen));
    kmesh_fft_host_t eng(km, 1, 16);   // width 16: blocks 16, 16, 5
    eng.prepare(nc, {{0, +1}, {0, -1}});
    auto via_fft = [&](std::vector<long> const &place, int sign, std::vector<long> const &take) {   // rows take[i] of FFT[X at place]
      nda::array<ComplexType, 2> O(long(take.size()), nc);
      for (long c0 = 0; c0 < nc; c0 += eng.cb) {
        const long w = std::min(eng.cb, nc - c0);
        detail::block_place(eng.buf(0), w, X.data() + c0, nc, N, place.data());
        eng.fft(0, w, sign);
        detail::block_out(O.data() + c0, nc, long(take.size()), w, eng.buf(0), take, 0, ComplexType(1.0));
      }
      return O;
    };
    std::vector<long> idR(N);
    for (long r = 0; r < N; ++r) idR[r] = r;
    auto chk = [&](nda::array<ComplexType, 2> const &a, nda::array<ComplexType, 2> const &b) {
      double d = 0.0, m = 0.0;
      for (long i = 0; i < a.size(); ++i) {
        d = std::max(d, std::abs(a.data()[i] - b.data()[i]));
        m = std::max(m, std::abs(b.data()[i]));
      }
      e_tr = std::max(e_tr, d / m);
    };
    nda::array<ComplexType, 2> G1 = nda::matmul(kft.Fp, X);   // sum_k e^{ikR} X(k)
    chk(via_fft(km.kpos, +1, idR), G1);
    nda::array<ComplexType, 2> G2 = nda::matmul(kft.Fm, X);   // sum_k e^{-ikR} X(k)
    chk(via_fft(km.kpos, -1, idR), G2);
    nda::array<ComplexType, 2> G3 = nda::matmul(kft.Hm, X);   // sum_q e^{-iQR} X(q)
    chk(via_fft(km.qpos, -1, idR), G3);
    nda::array<ComplexType, 2> G4 = nda::matmul(kft.Gm, X);   // (q) sum_R e^{-iQR} X(R)
    chk(via_fft(idR, -1, km.qpos), G4);
    nda::array<ComplexType, 2> G5 = nda::matmul(kft.Bp, X);   // (k) sum_R e^{ikR} X(R)
    chk(via_fft(idR, +1, km.kpos), G5);
  }
  const double deg = std::numbers::pi / 180.0, theta = 20.0 * deg, theta_t = 10.0 * deg;
  double emin_occ = 1e300, emax_vir = -1e300;
  for (long ik = 0; ik < f.nk; ++ik)
    for (long n = 0; n < f.nb; ++n) {
      if (f.e_rel(ik, n) < 0) emin_occ = std::min(emin_occ, f.e_rel(ik, n));
      else emax_vir = std::max(emax_vir, f.e_rel(ik, n));
    }
  numerics::line_dlr::bosonic_basis_t basis(theta, std::max(4.0, 1.2 * (emax_vir - emin_occ)), 1e-10, 0.5 * f.ks_gap);
  const double smax = 30.0 / (f.poles.emin() * std::sin(theta_t));
  time_ray_t ray_p(theta_t, smax, 1e-5, 1.5, 12, sector_t::particle), ray_h(theta_t, smax, 1e-5, 1.5, 12, sector_t::hole);
  aux_grid_t grid(*mpi, Np);
  propagator_t<HOST_MEMORY> prop(thc, grid);
  auto rel = [&](auto const &a, auto const &b) {
    const double d = comm.all_reduce_value(max_diff3(a, b), mpi3::max<>{});
    const double m = comm.all_reduce_value(max_abs3(b), mpi3::max<>{});
    return d / m;
  };
  utils::TimerManager T;
  auto fz = numerics::line_dlr::dense_nodes(theta, 1e-3, 60.0, 30);
  struct res_t {
    memory::array<HOST_MEMORY, ComplexType, 4> Pi;
    nda::array<ComplexType, 4> Sp, Sh, Sp1, Sh1, Sp2, Sh2;
  };
  memory::array<HOST_MEMORY, ComplexType, 4> wn;
  auto pass = [&](char const *mode, bool make_w) {
    env_scope_t e("COQUI_GWLINE_KFT", mode);
    res_t o;
    polarization<HOST_MEMORY>(prop, f.poles, mf, grid, basis.zeta_nodes, ray_p, ray_h, 8, o.Pi, T);
    if (make_w) {
      dyson_layout_t lay(*mpi, nq, basis.zeta_nodes.size(), Np);
      coulomb_blocks_t<HOST_MEMORY> Zb(thc, grid, lay.q_rng(), T);
      memory::array<HOST_MEMORY, ComplexType, 4> Pc = o.Pi;
      screened_interaction<HOST_MEMORY>(Pc, Zb, basis, grid, *mpi, wn, T);
    }
    self_energy<HOST_MEMORY>(prop, f.poles, wn, basis, mf, grid, *mpi, fz, ray_p, ray_h, 8, o.Sp, T, sector_t::both, false, nullptr,
                             0, &o.Sh);
    REQUIRE((prop.ahat_key >= 0.0 and prop.ahat_key == prop.pole_key));   // the hole leg used this mode's A^ cache
    {
      env_scope_t e1("COQUI_GWLINE_GT_CACHE", "0");
      self_energy<HOST_MEMORY>(prop, f.poles, wn, basis, mf, grid, *mpi, fz, ray_p, ray_h, 8, o.Sp1, T, sector_t::both, false,
                               nullptr, 0, &o.Sh1);
    }
    {   // residues transformed in place (the q plan's relief level 1)
      memory::array<HOST_MEMORY, ComplexType, 4> wc = wn;
      self_energy<HOST_MEMORY>(prop, f.poles, wc, basis, mf, grid, *mpi, fz, ray_p, ray_h, 8, o.Sp2, T, sector_t::both, false,
                               nullptr, 0, &o.Sh2, &wc);
    }
    return o;
  };
  auto g  = pass("gemm", true);
  auto ff = pass("fft", false);
  const double e_pi = rel(ff.Pi, g.Pi);
  const double e_s  = std::max(rel(ff.Sp, g.Sp), rel(ff.Sh, g.Sh));
  const double e_s1 = std::max(rel(ff.Sp1, g.Sp1), rel(ff.Sh1, g.Sh1));
  const double e_s2 = std::max(rel(ff.Sp2, g.Sp2), rel(ff.Sh2, g.Sh2));
  const double e_gg = std::max(rel(g.Sp2, g.Sp), rel(g.Sh1, g.Sh));   // gemm self-consistency (in place / no cache)
  app_log(1, "  [kft] {} ({} ranks, mesh {}x{}x{}, k order {}, q order {}): phases k {:.1e} Q {:.1e}, transforms {:.2e}; fft vs gemm: "
             "Pi {:.2e}, Sigma {:.2e} (rebuilt G~ {:.2e}, w^(R) in place {:.2e}; gemm variants among themselves {:.2e})",
          name, comm.size(), km.n[0], km.n[1], km.n[2], km.kid ? "mesh" : "permuted", km.qid ? "mesh" : "permuted", km.err_k, km.err_q,
          e_tr, e_pi, e_s, e_s1, e_s2, e_gg);
  REQUIRE(e_tr <= 1e-14);
  REQUIRE(e_pi <= 1e-13);
  const double gs = (name == "qe_si211") ? 1e-12 : 1e-13;   // si211: residue cancellations (perf 7.1: 2e-13 - 1e-12)
  REQUIRE(e_s <= gs);
  REQUIRE(e_s1 <= gs);
  REQUIRE(e_s2 <= gs);
#if defined(ENABLE_DEVICE)
  {   // device (cuFFT) vs the host gemm results on the same residues
    env_scope_t e("COQUI_GWLINE_KFT", "fft");
    propagator_t<DEVICE_MEMORY> pd(thc, grid);
    memory::array<DEVICE_MEMORY, ComplexType, 4> Pd;
    polarization<DEVICE_MEMORY>(pd, f.poles, mf, grid, basis.zeta_nodes, ray_p, ray_h, 8, Pd, T);
    nda::array<ComplexType, 4> Pdh = memory::to_memory_space<HOST_MEMORY>(Pd);
    memory::array<DEVICE_MEMORY, ComplexType, 4> wd = memory::to_memory_space<DEVICE_MEMORY>(wn);
    nda::array<ComplexType, 4> Sp, Sh, Sp2, Sh2;
    self_energy<DEVICE_MEMORY>(pd, f.poles, wd, basis, mf, grid, *mpi, fz, ray_p, ray_h, 8, Sp, T, sector_t::both, false, nullptr, 0,
                               &Sh);
    self_energy<DEVICE_MEMORY>(pd, f.poles, wd, basis, mf, grid, *mpi, fz, ray_p, ray_h, 8, Sp2, T, sector_t::both, false, nullptr, 0,
                               &Sh2, &wd);
    const double dpi = rel(Pdh, g.Pi), ds = std::max(rel(Sp, g.Sp), rel(Sh, g.Sh)), ds2 = std::max(rel(Sp2, g.Sp), rel(Sh2, g.Sh));
    app_log(1, "  [kft][device] {}: device fft vs host gemm: Pi {:.2e}, Sigma {:.2e}, w^(R) in place {:.2e}", name, dpi, ds, ds2);
    REQUIRE(dpi <= 1e-13);
    REQUIRE(ds <= gs);
    REQUIRE(ds2 <= gs);
  }
#endif
}

/// a kmesh_map_t of an n1 x n2 x n3 mesh with random k and q orders (the fixtures are all mesh-ordered) and its gemm matrices
struct synth_mesh_t {
  kmesh_map_t m;
  nda::array<ComplexType, 2> Fp, Hm;   // (R, k) e^{+ikR}, (R, q) e^{-iQR}
  synth_mesh_t(long n1, long n2, long n3, bool permute, unsigned seed) {
    m.n  = {n1, n2, n3};
    m.N  = n1 * n2 * n3;
    m.ok = true;
    m.kpos.resize(m.N);
    m.qpos.resize(m.N);
    for (long i = 0; i < m.N; ++i) m.kpos[i] = m.qpos[i] = i;
    std::mt19937_64 gen(seed);
    if (permute) {
      std::shuffle(m.kpos.begin(), m.kpos.end(), gen);
      std::shuffle(m.qpos.begin(), m.qpos.end(), gen);
    }
    m.kid = not permute;
    m.qid = not permute;
    Fp = nda::array<ComplexType, 2>(m.N, m.N);
    Hm = nda::array<ComplexType, 2>(m.N, m.N);
    auto ph = [&](long pos, long r) {
      const long a[3] = {r / (n2 * n3), (r / n3) % n2, r % n3}, j[3] = {pos / (n2 * n3), (pos / n3) % n2, pos % n3};
      double x = double((j[0] * a[0]) % n1) / n1 + double((j[1] * a[1]) % n2) / n2 + double((j[2] * a[2]) % n3) / n3;
      x -= std::floor(x);
      return std::exp(ComplexType(0.0, 2.0 * std::numbers::pi * x));
    };
    for (long r = 0; r < m.N; ++r)
      for (long k = 0; k < m.N; ++k) {
        Fp(r, k) = ph(m.kpos[k], r);
        Hm(r, k) = std::conj(ph(m.qpos[k], r));
      }
  }
};

} // namespace

/// [gw_line][kft] the host engine and the row helpers on meshes with permuted k / q orders (incl. odd and mixed sizes)
TEST_CASE("gw_line_kft_engine", "[gw_line][kft]") {
  double e = 0.0;
  for (auto [n1, n2, n3] : std::vector<std::tuple<long, long, long>>{{3, 4, 5}, {4, 4, 4}, {2, 3, 1}, {6, 6, 6}}) {
    synth_mesh_t sm(n1, n2, n3, true, 3);
    auto const &m = sm.m;
    const long N = m.N, nc = 41;
    std::mt19937_64 gen(5);
    std::normal_distribution<double> N01;
    nda::array<ComplexType, 2> X(N, nc);
    for (long i = 0; i < N; ++i)
      for (long c = 0; c < nc; ++c) X(i, c) = ComplexType(N01(gen), N01(gen));
    kmesh_fft_host_t eng(m, 1, 16);
    eng.prepare(nc, {{0, +1}, {0, -1}});
    std::vector<long> id(N);
    for (long r = 0; r < N; ++r) id[r] = r;
    auto via = [&](std::vector<long> const &place, int sign, std::vector<long> const &take) {
      nda::array<ComplexType, 2> O(N, nc);
      for (long c0 = 0; c0 < nc; c0 += eng.cb) {
        const long w = std::min(eng.cb, nc - c0);
        detail::block_place(eng.buf(0), w, X.data() + c0, nc, N, place.data());
        eng.fft(0, w, sign);
        detail::block_out(O.data() + c0, nc, N, w, eng.buf(0), take, 0, ComplexType(1.0));
      }
      return O;
    };
    auto rel = [](auto const &a, auto const &b) {
      double d = 0.0, mx = 0.0;
      for (long i = 0; i < a.size(); ++i) {
        d  = std::max(d, std::abs(a.data()[i] - b.data()[i]));
        mx = std::max(mx, std::abs(b.data()[i]));
      }
      return d / mx;
    };
    nda::array<ComplexType, 2> Fm = nda::conj(sm.Fp), Bp = nda::transpose(sm.Fp), Gm = nda::transpose(sm.Hm);
    e = std::max(e, rel(via(m.kpos, +1, id), nda::matmul(sm.Fp, X)));
    e = std::max(e, rel(via(m.kpos, -1, id), nda::matmul(Fm, X)));
    e = std::max(e, rel(via(m.qpos, -1, id), nda::matmul(sm.Hm, X)));
    e = std::max(e, rel(via(id, -1, m.qpos), nda::matmul(Gm, X)));
    e = std::max(e, rel(via(id, +1, m.kpos), nda::matmul(Bp, X)));
    // permute_rows (the device in-place residue placement) on the host
    nda::array<ComplexType, 2> Y = X, Z(N, nc);
    detail::permute_rows<HOST_MEMORY>(Y.data(), N, nc, m.qpos);
    for (long r = 0; r < N; ++r) Z(m.qpos[r], nda::range::all) = X(r, nda::range::all);
    e = std::max(e, rel(Y, Z));
  }
  app_log(1, "  [kft] engine on permuted meshes (3x4x5, 4x4x4, 2x3x1, 6x6x6): max rel {:.2e}", e);
  REQUIRE(e <= 1e-14);
}

/**
 * [.kft_bench] (hidden) per-column cost of the Pi-type convolution (two forward transforms, product, back transform to nrows
 * rows) with the dense gemms vs the blocked FFT engine, on meshes 4^3, 5^3, 6^3, 8^3; ncols columns (env
 * COQUI_GWLINE_BENCH_NCOLS, default 32 x 4300 = one si444 chunk on 128 ranks), nrows = N / 3 (IBZ rows). Run it on every core
 * of a node at once (mpirun -np <cores>) to include the memory-bandwidth contention of the production runs; rank 0 prints
 * the max over ranks.
 */
TEST_CASE("gw_line_kft_bench", "[.kft_bench]") {
  auto &mpi  = utils::make_unit_test_mpi_context();
  auto &comm = mpi->comm;
  const long ncols0 = detail::env_long("COQUI_GWLINE_BENCH_NCOLS", 32 * 4300);
  for (long n : {4L, 5L, 6L, 8L}) {
    synth_mesh_t sm(n, n, n, false, 1);
    auto const &m = sm.m;
    const long N = m.N, nr = std::max(1L, N / 3);
    const long ncols = std::max(1024L, ncols0 * 64 / N);   // same memory per array for every mesh
    nda::array<ComplexType, 2> A(N, ncols), B(N, ncols), X(N, ncols), C(nr, ncols);
    for (long i = 0; i < A.size(); ++i) {
      A.data()[i] = ComplexType(1e-3 * (i % 7), 1.0);
      B.data()[i] = ComplexType(1.0, 1e-3 * (i % 5));
    }
    nda::array<ComplexType, 2> Fm = nda::conj(sm.Fp), Gq(nr, N);
    std::vector<long> rows(nr);
    for (long i = 0; i < nr; ++i) {
      rows[i] = (7 * i) % N;
      for (long R = 0; R < N; ++R) Gq(i, R) = sm.Hm(R, rows[i]);
    }
    comm.barrier();
    auto t0 = std::chrono::steady_clock::now();
    nda::blas::gemm(ComplexType(1.0), sm.Fp, A, ComplexType(0.0), X);
    nda::blas::gemm(ComplexType(1.0), Fm, B, ComplexType(0.0), A);
    for (long i = 0; i < A.size(); ++i) A.data()[i] *= X.data()[i];
    nda::blas::gemm(ComplexType(1.0), Gq, A, ComplexType(0.0), C);
    const double tg = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    comm.barrier();
    for (long i = 0; i < A.size(); ++i) A.data()[i] = ComplexType(1e-3 * (i % 7), 1.0);
    kmesh_fft_host_t eng(m, 2);
    eng.prepare(ncols, {{0, +1}, {1, -1}, {0, -1}});
    std::vector<long> qrow(nr);
    for (long i = 0; i < nr; ++i) qrow[i] = m.qpos[rows[i]];
    comm.barrier();
    t0 = std::chrono::steady_clock::now();
    for (long c0 = 0; c0 < ncols; c0 += eng.cb) {
      const long w = std::min(eng.cb, ncols - c0);
      detail::block_in(eng.buf(0), N, w, A.data() + c0, ncols);
      eng.fft(0, w, +1);
      detail::block_in(eng.buf(1), N, w, B.data() + c0, ncols);
      eng.fft(1, w, -1);
      detail::block_mul(eng.buf(0), N, w, eng.buf(1), w);
      eng.fft(0, w, -1);
      detail::block_out(C.data() + c0, ncols, nr, w, eng.buf(0), qrow, 0, ComplexType(1.0));
    }
    const double tf = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double mg = comm.all_reduce_value(tg, mpi3::max<>{}), mf = comm.all_reduce_value(tf, mpi3::max<>{});
    app_log(1, "  [kft bench] mesh {}^3 (N {}), {} columns, {} ranks, cb {}: gemm {:.3f} s ({:.1f} ns/col), fft {:.3f} s ({:.1f} ns/col): x{:.2f}",
            n, N, ncols, comm.size(), eng.cb, mg, 1e9 * mg / ncols, mf, 1e9 * mf / ncols, mg / mf);
  }
}

TEST_CASE("gw_line_kft_lih222", "[gw_line][kft]") { run_kft("qe_lih222", 8); }
TEST_CASE("gw_line_kft_si211", "[gw_line][kft]") { run_kft("qe_si211", 8); }
TEST_CASE("gw_line_kft_lih223", "[gw_line][kft]") { run_kft("qe_lih223", 8); }

TEST_CASE("gw_line_perf71_ab_lih222", "[gw_line][perf71]") { run_ab("qe_lih222", 8); }
TEST_CASE("gw_line_perf71_ab_si211", "[gw_line][perf71]") { run_ab("qe_si211", 8); }
TEST_CASE("gw_line_perf71_ab_lih223", "[gw_line][perf71]") { run_ab("qe_lih223", 8); }

TEST_CASE("gw_line_perf71_nodes_lih222", "[.perf71_nodes]") { run_nodes("qe_lih222", 8); }
TEST_CASE("gw_line_perf71_nodes_si211", "[.perf71_nodes]") { run_nodes("qe_si211", 8); }
TEST_CASE("gw_line_perf71_nodes_lih223", "[.perf71_nodes]") { run_nodes("qe_lih223", 8); }

TEST_CASE("gw_line_perf71_rel_lih222", "[.perf71_rel]") { run_relations("qe_lih222", 8); }
TEST_CASE("gw_line_perf71_rel_si211", "[.perf71_rel]") { run_relations("qe_si211", 8); }
TEST_CASE("gw_line_perf71_rel_lih223", "[.perf71_rel]") { run_relations("qe_lih223", 8); }
