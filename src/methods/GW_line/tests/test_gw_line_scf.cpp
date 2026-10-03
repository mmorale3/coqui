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
 * S6 of notes/line_gw_cpp_plan.md: closure, driver, checkpoint/restart, python parity.
 *
 * [closure] toy (2 k, nb = 3, H + an exact 8-pole Sigma_c with rank-1 positive residues per k): Sigma^{>/<} sampled at the
 *   dense fermionic nodes, closure() (sector fits -> moments -> upfold -> Lehmann -> mu -> gapless compression) vs the exact
 *   G of the finite upfolded matrix [[H, B], [B^dag, diag E]]: the Lehmann G and the compressed G on the imaginary axis
 *   about the new centre (<= 1e-4 relative), N of the compressed density matrix vs the exact count at the new mu
 *   (<= 1e-3), the QP gap vs the exact one; with > 1 rank the k-distributed closure vs a single-rank closure (bitwise).
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include <chrono>
#include <cmath>
#include <complex>
#include <random>
#include <string>
#include <vector>

#include "mpi3/communicator.hpp"
#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "nda/linalg.hpp"

#include "numerics/line_dlr/line_basis.hpp"
#include "numerics/line_dlr/cayley.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/closure.hpp"

namespace {

using namespace methods::gw_line;
using numerics::line_dlr::line_basis_t;

/// exact G(z) = [(z - Ht)^{-1}]_{nb x nb} with Ht = [[H, B], [B^dag, diag E]]
nda::matrix<ComplexType> exact_G(nda::matrix<ComplexType> const &Ht, long nb, ComplexType z) {
  const long n = Ht.extent(0);
  nda::matrix<ComplexType> M(n, n);
  for (long i = 0; i < n; ++i)
    for (long j = 0; j < n; ++j) M(i, j) = (i == j ? z : ComplexType(0.0)) - Ht(i, j);
  M = nda::inverse(M);
  return nda::matrix<ComplexType>(M(nda::range(nb), nda::range(nb)));
}

} // namespace

namespace {
/// npk = 8: 4 + 4 fixed poles (exactly representable by K = 24 moments); npk > 8: random poles in +-[0.6, 4] (the moment
/// truncation is then the dominant error and the compression is checked against the Lehmann G of the closure).
void closure_toy(long npk) {
  auto &mpi  = utils::make_unit_test_mpi_context();
  auto &comm = mpi->comm;
  const long nk = 2, nb = 3;
  const bool exact_model = (npk == 8);
  const double theta = 20.0 * std::numbers::pi / 180.0, lam = 6.0, eps = 1e-10, nelec = 4.0;
  auto t0 = std::chrono::steady_clock::now();

  // model: H(k) = diag(eps_k) + small Hermitian noise; Sigma_c(k) = sum_l b_l b_l^dag / (z - E_l), 4 particle + 4 hole poles
  std::mt19937 gen(12345);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  const double Ep[4] = {0.7, 1.1, 1.8, 3.0}, Eh[4] = {-0.7, -1.2, -2.0, -3.5};
  nda::array<ComplexType, 3> H(nk, nb, nb);
  std::vector<nda::matrix<ComplexType>> Ht(nk);
  std::vector<nda::array<double, 1>> Epole(nk);
  std::vector<nda::array<ComplexType, 2>> Bk(nk);
  for (long ik = 0; ik < nk; ++ik) {
    const double e0[3] = {-0.5 + 0.03 * ik, -0.3 - 0.02 * ik, 0.5 + 0.04 * ik};
    for (long i = 0; i < nb; ++i)
      for (long j = 0; j <= i; ++j) {
        ComplexType x = (i == j) ? ComplexType(e0[i] + 0.02 * U(gen), 0.0) : 0.02 * ComplexType(U(gen), U(gen));
        H(ik, i, j) = x;
        H(ik, j, i) = std::conj(x);
      }
    Epole[ik] = nda::array<double, 1>(npk);
    Bk[ik]    = nda::array<ComplexType, 2>(nb, npk);
    for (long l = 0; l < npk; ++l) {
      if (exact_model) Epole[ik](l) = (l < 4 ? Ep[l] : Eh[l - 4]) * (1.0 + 0.05 * ik);
      else Epole[ik](l) = (l % 2 == 0 ? 1.0 : -1.0) * (0.6 + 3.4 * 0.5 * (1.0 + U(gen)));
      const double amp = exact_model ? 0.15 : 0.15 * std::sqrt(8.0 / double(npk));
      for (long i = 0; i < nb; ++i) Bk[ik](i, l) = amp * ComplexType(U(gen), U(gen));
    }
    Ht[ik] = nda::matrix<ComplexType>(nb + npk, nb + npk);
    Ht[ik]() = ComplexType(0.0);
    for (long i = 0; i < nb; ++i)
      for (long j = 0; j < nb; ++j) Ht[ik](i, j) = H(ik, i, j);
    for (long i = 0; i < nb; ++i)
      for (long l = 0; l < npk; ++l) {
        Ht[ik](i, nb + l) = Bk[ik](i, l);
        Ht[ik](nb + l, i) = std::conj(Bk[ik](i, l));
      }
    for (long l = 0; l < npk; ++l) Ht[ik](nb + l, nb + l) = Epole[ik](l);
  }

  // Sigma per sector at the dense nodes (centre 0)
  auto zeta = numerics::line_dlr::dense_nodes(theta, 1e-3, 60.0, 120);
  const long nz = zeta.size();
  nda::array<ComplexType, 4> Sp(nk, nz, nb, nb), Sh(nk, nz, nb, nb);
  Sp() = ComplexType(0.0);
  Sh() = ComplexType(0.0);
  for (long ik = 0; ik < nk; ++ik)
    for (long iz = 0; iz < nz; ++iz)
      for (long l = 0; l < npk; ++l) {
        auto &S = (Epole[ik](l) > 0.0) ? Sp : Sh;
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j)
            S(ik, iz, i, j) += Bk[ik](i, l) * std::conj(Bk[ik](j, l)) / (zeta(iz) - Epole[ik](l));
      }

  line_basis_t bp(theta, lam, eps, lam, 0.02, -1.0, 60.0), bh(theta, lam, eps, 0.02, lam, -1.0, 60.0);
  line_basis_t gp(theta, lam, eps, lam, 0.0, -1.0, 60.0), gh(theta, lam, eps, 0.0, lam, -1.0, 60.0);
  closure_params_t p;   // wp 0.11, K 24, tol_gram 1e-10, nphi 8
  utils::TimerManager Timer;
  auto out = closure(comm, H, Sp, Sh, zeta, bp, bh, gp, gh, p, nelec, Timer);

  // exact Lehmann of the finite matrices, QP gap and electron count at the new centre
  std::vector<nda::array<double, 1>> ee(nk);
  std::vector<nda::array<ComplexType, 2>> vv(nk);
  for (long ik = 0; ik < nk; ++ik) {
    auto [ev, V] = nda::linalg::eigenelements(Ht[ik]);
    ee[ik]       = ev;
    vv[ik]       = nda::array<ComplexType, 2>(V(nda::range(nb), nda::range::all));
  }
  auto cp_ex = numerics::line_dlr::chemical_potential(ee, vv, nelec);
  double N_ex = 0.0;
  for (long ik = 0; ik < nk; ++ik)
    for (long m = 0; m < ee[ik].size(); ++m)
      if (ee[ik](m) < out.dmu)
        for (long i = 0; i < nb; ++i) N_ex += 2.0 / nk * std::norm(vv[ik](i, m));

  // G on the imaginary axis about the new centre: Lehmann, compressed, exact
  double errL = 0.0, errC = 0.0, errCL = 0.0, gmax = 0.0;
  for (long iw = 0; iw < 60; ++iw) {
    const double w = 1e-2 * std::pow(10.0, 3.0 * iw / 59.0);
    const ComplexType z(0.0, w);
    for (long ik = 0; ik < nk; ++ik) {
      auto Gx = exact_G(Ht[ik], nb, out.dmu + z);
      nda::matrix<ComplexType> GL(nb, nb), GC(nb, nb);
      GL() = ComplexType(0.0);
      GC() = ComplexType(0.0);
      for (long m = 0; m < out.leh.e[ik].size(); ++m)
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j)
            GL(i, j) += out.leh.v[ik](i, m) * std::conj(out.leh.v[ik](j, m)) / (z - out.leh.e[ik](m));
      for (auto const *ps : {&out.poles.part[ik], &out.poles.hole[ik]})
        for (long m = 0; m < ps->size(); ++m)
          for (long i = 0; i < nb; ++i)
            for (long j = 0; j < nb; ++j) GC(i, j) += ps->coef(m, i, j) / (z - ps->e(m));
      gmax = std::max(gmax, nda::max_element(nda::abs(Gx)));
      errL = std::max(errL, nda::max_element(nda::abs(GL - Gx)));
      errC = std::max(errC, nda::max_element(nda::abs(GC - Gx)));
      errCL = std::max(errCL, nda::max_element(nda::abs(GC - GL)));
    }
  }
  const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  app_log(1, "[closure toy, {} Sigma poles per k] ranks {} | Sigma bases {}+{}, G bases {}+{} | upfolded poles {} {} | held-out {:.1e} {:.1e}", npk, comm.size(),
          bp.rank, bh.rank, gp.rank, gh.rank, out.npoles[0], out.npoles[1], out.heldout[0], out.heldout[1]);
  app_log(1, "  dmu {:.6f} (exact-model finder {:.6f}) | gap {:.6f} (exact {:.6f}) | N_mu {:.6f} N_lehmann {:.6f} N_compressed {:.6f} "
             "N_exact {:.6f} | dropped {:.1e}",
          out.dmu, cp_ex.mu, out.e_lumo - out.e_homo, cp_ex.gap, out.N_mu, out.nel_lehmann, out.nel_compressed, N_ex, out.dropped);
  app_log(1, "  G(i w) rel. error vs exact: Lehmann {:.2e}  compressed {:.2e}; compressed vs Lehmann {:.2e} (max|G| {:.3f}) | {:.1f} s",
          errL / gmax, errC / gmax, errCL / gmax, gmax, dt);
  REQUIRE(errCL / gmax < 1e-4);
  REQUIRE(std::abs(out.nel_compressed - out.nel_lehmann) < 1e-3);
  if (exact_model) {
    REQUIRE(errL / gmax < 1e-4);
    REQUIRE(errC / gmax < 1e-4);
    REQUIRE(std::abs(out.nel_compressed - N_ex) < 1e-3);
    REQUIRE(std::abs((out.e_lumo - out.e_homo) - cp_ex.gap) < 1e-4);
  }

  // rank-count independence: the same closure on a single-rank communicator (every rank alone)
  if (comm.size() > 1) {
    auto self = comm.split(comm.rank(), 0);
    utils::TimerManager T2;
    auto o1 = closure(self, H, Sp, Sh, zeta, bp, bh, gp, gh, p, nelec, T2);
    double d = std::abs(o1.dmu - out.dmu);
    for (long ik = 0; ik < nk; ++ik) {
      REQUIRE(o1.leh.e[ik].size() == out.leh.e[ik].size());
      d = std::max(d, nda::max_element(nda::abs(o1.leh.e[ik] - out.leh.e[ik])));
      d = std::max(d, nda::max_element(nda::abs(o1.poles.part[ik].coef - out.poles.part[ik].coef)));
      d = std::max(d, nda::max_element(nda::abs(o1.poles.hole[ik].coef - out.poles.hole[ik].coef)));
    }
    app_log(1, "  {} ranks vs 1 rank: max |diff| {:.1e}", comm.size(), d);
    REQUIRE(d <= 1e-12);
  }
}
} // namespace

TEST_CASE("gw_line_closure_toy", "[gw_line][scf][closure]") {
  SECTION("exact 8-pole Sigma") { closure_toy(8); }
  SECTION("120-pole Sigma") { closure_toy(120); }
}
