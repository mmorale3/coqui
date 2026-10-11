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

#ifndef COQUI_METHODS_GW_LINE_HYBRID_HPP
#define COQUI_METHODS_GW_LINE_HYBRID_HPP

/**
 * S8b.3 hybrid finite-T density (notes section 11.6 "Hybrid"; python LineGW.sigma_tau_leg, finite_t.density_matsubara):
 * scf_density = "matsubara".
 *
 * 1. The tau leg of Sigma. Sigma(k, tau) = -(1/N_k) sum_q X^dag [G~_p(k-q, tau) o Wb(q, tau)] X on [0, beta] with ALL poles,
 *    G~_p = X [sum_m (1 - f_m) e^{-e_m tau} c_m] X^dag, Wb = sum_j [(1 + n_j) e^{-nu_j tau} w_j(q) + n_j e^{+nu_j tau} w_j(-q)^T]
 *    (python's Stau). It is the self-energy KERNEL itself (self_energy.hpp / self_energy_ibz.hpp, unchanged) on tau "rays" at
 *    theta_t = pi / 2 over [0, beta / 2] with the tau lists of thermal.hpp (all poles x sqrt(1 - f) / sqrt(f)) and the
 *    Bose-augmented basis with EXACT weights (bosonic_basis_t::with_exact_bose): the particle leg (t = -i tau) gives
 *    Sp(tau) = -Stau(tau), the hole leg (t = +i tau') gives Sh(tau') = -Stau(beta - tau') (fermionic: G~_p(tau) = G~_h(beta - tau),
 *    and the bosonic weights swap (1 + n) e^{-nu tau} <-> n e^{nu (beta - tau)}), every factor bounded on [0, beta / 2].
 *    The kernel runs with an IDENTITY transform on the node set (it returns the nodal values; hybrid_nodes_t), the
 *    Matsubara transform is applied per k afterwards with the tau ID's own transform matrices (any i w_n, exact for the
 *    family {e^{-E tau}} to the ID tolerance): Sigma_c(i w_n) = F_p(i w_n) Sp + F_h(i w_n) Sh. The node set is the tau ID of
 *    the run's fixed range (E_max = g_emax + lam_b, E_neg = 2 ln(1/tau_eps)/beta), so the nodal values are mixed and
 *    checkpointed like the node Sigma. Four extra nodes tau = 0, d, 2d, 3d per leg give the high-frequency moments
 *    S1 = Sp(0) + Sh(0), S2 = -Sp'(0) + Sh'(0) (one-sided third-order differences, d = 1e-3 / E_max).
 * 2. The Matsubara density (density_matsubara): dense truncated set n = 0..N-1 (w_{N-1} >= wmax), H = H0 + F - mu (Hermitian),
 *    G(i w; dmu) = [i w + dmu - H - Sigma_c(i w)]^{-1}, D = f(H - dmu) + (1/beta) sum_n [dG + dG^dag] + T4 B (B = S2 + H' S1 +
 *    S1 H', T4 = (2/beta)(beta/pi)^4 zeta(4, N + 1/2)/16), the w^-6 remainder eliminated from the partial sums at N/2 and N;
 *    N(dmu) = N_el by bisection to the root with Sigma_c fixed (Tr G from the eigenvalues of H + Sigma_c(i w_n), once); D at the
 *    root by inversion. k-distributed (the rows of Sigma owned by the rank; each count / density element has one owner).
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/linalg.hpp"
#include "nda/linalg/eigenelements.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/check.hpp"
#include "numerics/line_dlr/cayley.hpp"
#include "numerics/line_dlr/time_id.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/thermal_mu.hpp"
#include "methods/GW_line/time_grids.hpp"

namespace methods::gw_line {

/// Hurwitz zeta(s, a) for s > 1, a > 0 (Euler-Maclaurin after 16 direct terms; relative accuracy ~1e-16)
inline double hurwitz_zeta(double s, double a) {
  const long M = 16;
  double z = 0.0;
  for (long k = 0; k < M; ++k) z += std::pow(a + double(k), -s);
  const double x = a + double(M);
  z += std::pow(x, 1.0 - s) / (s - 1.0) + 0.5 * std::pow(x, -s);
  const double B[5] = {1.0 / 6.0, -1.0 / 30.0, 1.0 / 42.0, -1.0 / 30.0, 5.0 / 66.0};
  double fact = 1.0, rise = s;   // (2j)!, s (s + 1) ... (s + 2j - 2)
  double xp = std::pow(x, -s - 1.0);
  for (int j = 1; j <= 5; ++j) {
    fact *= double(2 * j - 1) * double(2 * j);
    z += B[j - 1] / fact * rise * xp;
    rise *= (s + double(2 * j - 1)) * (s + double(2 * j));
    xp /= x * x;
  }
  return z;
}

/// the dense truncated fermionic set: N = ceil((wmax beta / pi - 1) / 2) + 1, i w_n = i pi (2n + 1) / beta, n < N
inline nda::array<ComplexType, 1> matsubara_set(double beta, double wmax) {
  const long N = long(std::ceil((wmax * beta / std::numbers::pi - 1.0) / 2.0)) + 1;
  nda::array<ComplexType, 1> iw(N);
  for (long n = 0; n < N; ++n) iw(n) = ComplexType(0.0, std::numbers::pi * double(2 * n + 1) / beta);
  return iw;
}

/// identity "transform" on a node set (the kernel then returns the nodal values; zeta must have one dummy point per node)
struct identity_nodes_t {
  double theta_t = std::numbers::pi / 2.0;
  numerics::line_dlr::sector_t sector = numerics::line_dlr::sector_t::particle;
  ComplexType phase = ComplexType(1.0, 0.0);
  nda::array<double, 1> s;
  nda::array<ComplexType, 1> t;
  nda::array<ComplexType, 2> transform_matrix(nda::array<ComplexType, 1> const &zeta) const {
    utils::check(zeta.size() == s.size(), "gw_line::identity_nodes_t: {} points for {} nodes", zeta.size(), s.size());
    nda::array<ComplexType, 2> F(zeta.size(), s.size());
    F() = ComplexType(0.0);
    for (long i = 0; i < s.size(); ++i) F(i, i) = ComplexType(1.0);
    return F;
  }
};

/// the tau nodes of the Sigma leg (file header, 1.)
struct hybrid_nodes_t {
  numerics::line_dlr::time_id_t idp, idh;   ///< tau ID on [0, beta / 2] (particle t = -i tau, hole = conjugate)
  long ntau = 0;                            ///< ID nodes; the kernel node sets have ntau + 4 (tau = 0, d, 2d, 3d)
  double delta = 0.0, Emax = 0.0, Eneg = 0.0;
  std::optional<numerics::line_dlr::time_nodes_t> kp, kh;   ///< kernel node sets (identity transform)
  nda::array<ComplexType, 1> zdummy;        ///< ntau + 4 dummy points
  long size() const { return ntau + 4; }
  /// F_p (nw, ntau), F_h (nw, ntau): Sigma_c(i w) = F_p Sp + F_h Sh
  std::pair<nda::array<ComplexType, 2>, nda::array<ComplexType, 2>> fourier(nda::array<ComplexType, 1> const &iw) const {
    return {idp.transform_matrix(iw), idh.transform_matrix(iw)};
  }
};

/// built on rank 0 of comm and broadcast; Emax = the run's fixed decay scale (g_emax + lam_b)
inline hybrid_nodes_t make_hybrid_nodes(double beta, double Emax, double tau_eps, boost::mpi3::communicator &comm) {
  using numerics::line_dlr::sector_t;
  hybrid_nodes_t H;
  H.Emax = Emax;
  H.Eneg = 2.0 * std::log(1.0 / tau_eps) / beta;   // E < -Eneg: below tau_eps on [0, beta / 2] (KMS)
  numerics::line_dlr::time_id_opts_t o;
  o.pad = 1.25;
  if (comm.rank() == 0)
    H.idp = numerics::line_dlr::time_id_t::finite_interval(std::numbers::pi / 2.0, sector_t::particle, H.Eneg, std::max(Emax, 2.0 * H.Eneg),
                                                           0.5 * beta, tau_eps, o);
  else
    H.idp.opts = o;
  detail::bcast_time_id(comm, H.idp, 0);
  H.idh   = detail::conjugate_grid(H.idp);
  H.ntau  = H.idp.size();
  H.delta = 1e-3 / Emax;
  for (int leg = 0; leg < 2; ++leg) {
    identity_nodes_t n;
    n.sector = leg == 0 ? sector_t::particle : sector_t::hole;
    n.phase  = leg == 0 ? ComplexType(0.0, -1.0) : ComplexType(0.0, 1.0);
    n.s      = nda::array<double, 1>(H.ntau + 4);
    for (long j = 0; j < H.ntau; ++j) n.s(j) = H.idp.s(j);
    for (long j = 0; j < 4; ++j) n.s(H.ntau + j) = double(j) * H.delta;
    n.t = nda::array<ComplexType, 1>(n.s.size());
    for (long j = 0; j < n.s.size(); ++j) n.t(j) = n.s(j) * n.phase;
    (leg == 0 ? H.kp : H.kh).emplace(n);
  }
  H.zdummy = nda::array<ComplexType, 1>(H.ntau + 4);
  for (long j = 0; j < H.zdummy.size(); ++j) H.zdummy(j) = ComplexType(0.0, double(j + 1));
  return H;
}

/// S1, S2 of one k from the nodal values (rows of Sp / Sh: (ntau + 4, nb, nb))
inline std::pair<nda::matrix<ComplexType>, nda::matrix<ComplexType>> hybrid_moments(hybrid_nodes_t const &H,
                                                                                  nda::array<ComplexType, 3> const &Sp,
                                                                                  nda::array<ComplexType, 3> const &Sh) {
  const long n0 = H.ntau, nb = Sp.extent(1);
  nda::matrix<ComplexType> S1(nb, nb), S2(nb, nb);
  auto all = nda::range::all;
  auto der = [&](nda::array<ComplexType, 3> const &S) {
    nda::matrix<ComplexType> d(nb, nb);
    d = (-11.0 * S(n0, all, all) + 18.0 * S(n0 + 1, all, all) - 9.0 * S(n0 + 2, all, all) + 2.0 * S(n0 + 3, all, all)) / (6.0 * H.delta);
    return d;
  };
  S1 = Sp(n0, all, all) + Sh(n0, all, all);
  S2 = der(Sh) - der(Sp);
  return {S1, S2};
}

/// eigenvalues of a general complex matrix (LAPACK zgees, no Schur vectors)
inline nda::array<ComplexType, 1> general_eigenvalues(nda::matrix<ComplexType> const &A) {
  const int n = int(A.extent(0));
  nda::matrix<ComplexType, nda::F_layout> M(A);
  nda::array<ComplexType, 1> w(n);
  int sdim = 0, info = 0, lwork = -1, ldvs = 1;
  ComplexType wq, dummy;
  std::vector<double> rwork(n);
  numerics::line_dlr::detail::f77::zgees_("N", "N", nullptr, &n, M.data(), &n, &sdim, w.data(), &dummy, &ldvs, &wq, &lwork, rwork.data(),
                                         nullptr, &info);
  lwork = std::max(1, int(wq.real()));
  std::vector<ComplexType> work(lwork);
  numerics::line_dlr::detail::f77::zgees_("N", "N", nullptr, &n, M.data(), &n, &sdim, w.data(), &dummy, &ldvs, work.data(), &lwork,
                                         rwork.data(), nullptr, &info);
  utils::check(info == 0, "gw_line::general_eigenvalues: zgees info {}", info);
  return w;
}

struct hybrid_density_t {
  double dmu = 0.0, N = 0.0, N_trace = 0.0, tail_max = 0.0;
  long nfreq = 0, nbisect = 0;
  nda::array<ComplexType, 3> D;   ///< (nk, nb, nb), every rank
};

/**
 * File header, 2. Hrel (nk, nb, nb) Hermitian (H0 + F - mu, mu-relative); Sp / Sh (nloc, ntau + 4, nb, nb): the mixed nodal
 * values of the k in krows (this rank); kw: k weights (empty: uniform), nelec. Collective over comm.
 */
inline hybrid_density_t matsubara_density(boost::mpi3::communicator &comm, nda::array<ComplexType, 3> const &Hrel,
                                          nda::array<ComplexType, 4> const &Sp, nda::array<ComplexType, 4> const &Sh,
                                          std::vector<long> const &krows, std::vector<double> const &kw_in, hybrid_nodes_t const &HN,
                                          double beta, double wmax, double nelec, long maxit = 200) {
  using numerics::line_dlr::fermi;
  auto all = nda::range::all;
  const long nk = Hrel.extent(0), nb = Hrel.extent(1), nloc = long(krows.size()), nt = HN.ntau;
  auto iw  = matsubara_set(beta, wmax);
  const long Nf = iw.size(), Nh = Nf / 2;
  std::vector<double> kw(nk, 1.0 / double(nk));
  if (not kw_in.empty()) {
    double s = 0.0;
    for (double x : kw_in) s += x;
    for (long k = 0; k < nk; ++k) kw[k] = kw_in[k] / s;
  }
  const double T4 = 2.0 / beta * std::pow(beta / std::numbers::pi, 4) * hurwitz_zeta(4.0, double(Nf) + 0.5) / 16.0;
  const double T4h = 2.0 / beta * std::pow(beta / std::numbers::pi, 4) * hurwitz_zeta(4.0, double(Nh) + 0.5) / 16.0;
  const double z6 = hurwitz_zeta(6.0, double(Nf) + 0.5), z6h = hurwitz_zeta(6.0, double(Nh) + 0.5);
  auto extrap = [&](auto const &xN, auto const &xh) { return xN + (xN - xh) / (z6h - z6) * z6; };
  auto [Fp, Fh] = HN.fourier(iw);
  // per owned k: H eigen pairs, the moments, the eigenvalues of H + Sigma_c(i w_n)
  struct kdata_t {
    nda::array<double, 1> h;
    nda::matrix<ComplexType> V, H, S1, S2;
    nda::array<ComplexType, 2> lam;   ///< (Nf, nb)
    double trS1 = 0.0, trS2 = 0.0, trHS1 = 0.0;
  };
  std::vector<kdata_t> kd(nloc);
  auto sigma_iw = [&](long l, long n) {   // Sigma_c(k, i w_n), (nb, nb)
    nda::matrix<ComplexType> S(nb, nb);
    S() = 0.0;
    for (long j = 0; j < nt; ++j) S += Fp(n, j) * Sp(l, j, all, all) + Fh(n, j) * Sh(l, j, all, all);
    return S;
  };
  for (long l = 0; l < nloc; ++l) {
    auto &d = kd[l];
    const long k = krows[l];
    d.H = nda::matrix<ComplexType>(Hrel(k, all, all));
    auto [ev, V] = nda::linalg::eigenelements(d.H);
    d.h = ev;
    d.V = V;
    nda::array<ComplexType, 3> sp(Sp(l, all, all, all)), sh(Sh(l, all, all, all));
    auto [S1, S2] = hybrid_moments(HN, sp, sh);
    d.S1 = S1;
    d.S2 = S2;
    for (long i = 0; i < nb; ++i) {
      d.trS1 += S1(i, i).real();
      d.trS2 += S2(i, i).real();
      for (long j = 0; j < nb; ++j) d.trHS1 += (d.H(i, j) * S1(j, i)).real();
    }
    d.lam = nda::array<ComplexType, 2>(Nf, nb);
    for (long n = 0; n < Nf; ++n) {
      nda::matrix<ComplexType> M = d.H + sigma_iw(l, n);
      d.lam(n, all) = general_eigenvalues(M);
    }
  }
  auto ntrace = [&](double dm) {   // N(dm), the k sum in k order (rank-count independent)
    std::vector<double> nk_(nk, 0.0);
    for (long l = 0; l < nloc; ++l) {
      auto const &d = kd[l];
      double s = 0.0, sh_ = 0.0;
      for (long n = 0; n < Nf; ++n) {
        const ComplexType z = iw(n) + dm;
        ComplexType a = 0.0;
        for (long b = 0; b < nb; ++b) a += 1.0 / (z - d.lam(n, b)) - 1.0 / (z - d.h(b));
        s += a.real();
        if (n < Nh) sh_ += a.real();
      }
      double f0 = 0.0;
      for (long b = 0; b < nb; ++b) f0 += fermi(d.h(b) - dm, beta);
      const double tB = d.trS2 + 2.0 * d.trHS1 - 2.0 * dm * d.trS1;
      nk_[krows[l]] = 2.0 * kw[krows[l]] * extrap(f0 + 2.0 * s / beta + T4 * tB, f0 + 2.0 * sh_ / beta + T4h * tB);
    }
    if (comm.size() > 1) comm.all_reduce_in_place_n(nk_.data(), nk, std::plus<>{});
    double N = 0.0;
    for (long k = 0; k < nk; ++k) N += nk_[k];
    return N;
  };
  hybrid_density_t out;
  out.nfreq = Nf;
  {   // bracket from all eigenvalues
    double lo = 1e300, hi = -1e300;
    for (auto const &d : kd) {
      for (long b = 0; b < nb; ++b) lo = std::min(lo, d.h(b)), hi = std::max(hi, d.h(b));
      for (auto const &x : d.lam) lo = std::min(lo, x.real()), hi = std::max(hi, x.real());
    }
    lo = comm.all_reduce_value(lo, boost::mpi3::min<>{}) - 1.0 - 50.0 / beta;
    hi = comm.all_reduce_value(hi, boost::mpi3::max<>{}) + 1.0 + 50.0 / beta;
    double Nlo = ntrace(lo) - nelec, mid = lo, Nm = Nlo;
    for (long it = 0; it < maxit; ++it) {
      mid = 0.5 * (lo + hi);
      Nm  = ntrace(mid) - nelec;
      ++out.nbisect;
      if (Nm == 0.0 or hi - lo < 4e-16 * std::max(1.0, std::abs(mid))) break;
      if ((Nm < 0) == (Nlo < 0)) {
        lo  = mid;
        Nlo = Nm;
      } else
        hi = mid;
    }
    out.dmu     = mid;
    out.N_trace = Nm + nelec;
  }
  // D at the root
  out.D = nda::array<ComplexType, 3>(nk, nb, nb);
  out.D() = 0.0;
  const double dm = out.dmu;
  for (long l = 0; l < nloc; ++l) {
    auto const &d = kd[l];
    nda::matrix<ComplexType> dG(nb, nb), dGh(nb, nb), I(nb, nb);
    dG()  = 0.0;
    dGh() = 0.0;
    I()   = 0.0;
    for (long b = 0; b < nb; ++b) I(b, b) = 1.0;
    for (long n = 0; n < Nf; ++n) {
      const ComplexType z = iw(n) + dm;
      nda::matrix<ComplexType> G = nda::inverse(nda::matrix<ComplexType>(z * I - d.H - sigma_iw(l, n)));
      nda::matrix<ComplexType> Vz(nb, nb);
      for (long a = 0; a < nb; ++a)
        for (long b = 0; b < nb; ++b) Vz(a, b) = d.V(a, b) / (z - d.h(b));
      G -= Vz * nda::dagger(d.V);
      dG += G;
      if (n < Nh) dGh += G;
    }
    nda::matrix<ComplexType> Hd = d.H - dm * I;
    nda::matrix<ComplexType> B  = d.S2 + Hd * d.S1 + d.S1 * Hd;
    nda::matrix<ComplexType> Vf(nb, nb);
    for (long a = 0; a < nb; ++a)
      for (long b = 0; b < nb; ++b) Vf(a, b) = d.V(a, b) * fermi(d.h(b) - dm, beta);
    nda::matrix<ComplexType> D0 = Vf * nda::dagger(d.V);
    nda::matrix<ComplexType> xN = D0 + (dG + nda::dagger(dG)) / beta + T4 * B;
    nda::matrix<ComplexType> xh = D0 + (dGh + nda::dagger(dGh)) / beta + T4h * B;
    out.D(krows[l], all, all) = extrap(xN, xh);
    for (auto const &x : B) out.tail_max = std::max(out.tail_max, T4 * std::abs(x));
  }
  if (comm.size() > 1) {
    comm.all_reduce_in_place_n(out.D.data(), out.D.size(), std::plus<>{});
    out.tail_max = comm.all_reduce_value(out.tail_max, boost::mpi3::max<>{});
  }
  out.N = 0.0;
  for (long k = 0; k < nk; ++k) {
    double tr = 0.0;
    for (long b = 0; b < nb; ++b) tr += out.D(k, b, b).real();
    out.N += 2.0 * kw[k] * tr;
  }
  return out;
}

/// the exact Fermi density sum_m f(e_m) v_m v_m^dag of factorized poles (both sectors), (nk, nb, nb)
inline nda::array<ComplexType, 3> density_fermi(pole_data_t const &p, double beta) {
  nda::array<ComplexType, 3> D(p.nk, p.nb, p.nb);
  D() = 0.0;
  for (long k = 0; k < p.nk; ++k)
    for (auto const *ps : {&p.hole[k], &p.part[k]}) {
      utils::check(ps->size() == 0 or ps->is_factorized(), "gw_line::density_fermi: factorized poles required");
      for (long m = 0; m < ps->size(); ++m) {
        const double f = numerics::line_dlr::fermi(ps->e(m), beta);
        for (long a = 0; a < p.nb; ++a)
          for (long b = 0; b < p.nb; ++b) D(k, a, b) += f * ps->v(a, m) * std::conj(ps->v(b, m));
      }
    }
  return D;
}

} // namespace methods::gw_line

#endif
