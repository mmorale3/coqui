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
 *   Second section: 120 random poles per k (moment truncation active): compressed vs Lehmann G and N.
 *
 * lih222 (qe_lih222, THC loaded from tests/unit_test_files/gw_line/lih222_thc/thc.eri.h5 = nIpts 8 nbnd, written once by
 * the hidden test case [.gw_line_dump] together with system.h5 = H0, KS eigenvalues, qk_to_k2, mu0, nelec for python).
 * Small settings (scf_params): theta 20, eps 1e-8, lam 6, lam_b 4, sigma_gap = bos_gap = 0.02, g_gap 0, 120 nodes per ray,
 * wp 0.11, K 8, tol_gram 1e-10, nphi 8, mixing 0.5, t_chunk 8, ray_decades 36.
 * [restart] 2 iterations + restart + 1 vs 3 uninterrupted iterations: bitwise identical mu, poles, F, Sigma at the nodes;
 *   spectra written by the restarted run and again by a restart with nothing left to iterate (niter reached): identical.
 *   Both time grids (S7b): time_grid = "id" (time-node ID, the default) and "gl" (Gauss-Legendre rays).
 * [parity] (S7f) 3 iterations vs the python driver on the same THC/H0/KS data AND the same real-pole bases on every
 *   platform (bases_file = lih222_thc/bases_lamb12.h5; coqui/cayley/scripts/gen_lih222_scf_ref.py --lam-b 12 ->
 *   tests/unit_test_files/gw_line/lih222_scf_ref.h5), lam_b 12, compressed, time_grid "gl" (python's GL rays): mu, QP gap
 *   and Sigma at the stored nodes of k = 0 per iteration within gates = 10 x the measured noise floor (parity_floor).
 * [id_vs_gl] (S7b, S7f) 2 iterations, lehmann, lam_b auto, time_grid = "id" (time_eps 1e-8 and 1e-10) vs "gl": |dmu|,
 *   |dgap| and max|dSigma| / max|Sigma| at all nodes and k; gates = 5 x the measured noise floor (idgl_gate).
 * [gygi] (S9a) div_treatment = "gygi" (hf_div_treatment follows: "gygi"), lehmann, id, lam_b auto: 2 iterations vs 1 + restart + 1
 *   (bitwise mu, poles, F, Sigma); the head group scf_line/iter<N>/head/ (shapes, eps_inf, h0 at the nodes = the
 *   extrapolated residue function at the nodes); F(iter0) gygi - F(iter0) ignore_g0 = -madelung D_KS; iteration-1 Sigma
 *   gygi - ignore_g0 = the head term (positive: its anti-Hermitian part at the nodes has a definite sign per sector).
 * [.time_id_poles] (hidden diagnostic) kernels on the poles of a checkpoint iteration (GW_LINE_DIAG_FILE, GW_LINE_DIAG_ITER):
 *   default GL rays and ID grids (time_eps 1e-8/1e-10/1e-12, pad 1.25/2) vs a refined GL reference.
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <complex>
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
#include "nda/linalg.hpp"

#include <filesystem>

#include "h5/h5.hpp"
#include "nda/h5.hpp"
#include "mean_field/default_MF.hpp"
#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/GW_line/driver.hpp"
#include "numerics/line_dlr/line_basis.hpp"
#include "numerics/line_dlr/bosonic_basis.hpp"
#include "numerics/line_dlr/cayley.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/closure.hpp"
#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/polarization.hpp"
#include "methods/GW_line/screened.hpp"
#include "methods/GW_line/self_energy.hpp"
#include "methods/GW_line/time_grids.hpp"
#include "methods/GW_line/static_part.hpp"
#include "methods/GW_line/head.hpp"

namespace {

using namespace methods::gw_line;
using numerics::line_dlr::line_basis_t;
using numerics::line_dlr::sector_t;

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

  // S7c: the Lehmann representation (pruning only): the factorized poles reproduce the Lehmann G up to the pruned poles,
  // N = 2 sum_k w_k Tr D of the retained poles
  utils::TimerManager TL;
  g_repr_params_t grl;
  grl.repr = "lehmann";
  auto outL = closure(comm, H, Sp, Sh, zeta, bp, bh, gp, gh, p, nelec, TL, grl);
  REQUIRE(outL.repr == "lehmann");
  REQUIRE(outL.poles.is_factorized());
  REQUIRE(outL.dmu == out.dmu);
  double errF = 0.0, nF = 0.0;
  for (long iw = 0; iw < 60; ++iw) {
    const ComplexType z(0.0, 1e-2 * std::pow(10.0, 3.0 * iw / 59.0));
    for (long ik = 0; ik < nk; ++ik) {
      nda::matrix<ComplexType> GL(nb, nb), GF(nb, nb);
      GL() = ComplexType(0.0);
      GF() = ComplexType(0.0);
      for (long m = 0; m < out.leh.e[ik].size(); ++m)
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j)
            GL(i, j) += out.leh.v[ik](i, m) * std::conj(out.leh.v[ik](j, m)) / (z - out.leh.e[ik](m));
      for (auto const *ps : {&outL.poles.part[ik], &outL.poles.hole[ik]})
        for (long m = 0; m < ps->size(); ++m)
          for (long i = 0; i < nb; ++i)
            for (long j = 0; j < nb; ++j) GF(i, j) += ps->v(i, m) * std::conj(ps->v(j, m)) / (z - ps->e(m));
      errF = std::max(errF, nda::max_element(nda::abs(GF - GL)));
    }
  }
  {
    auto D = density_matrix(outL.poles);
    for (long ik = 0; ik < nk; ++ik)
      for (long i = 0; i < nb; ++i) nF += 2.0 / double(nk) * std::real(D(ik, i, i));
  }
  app_log(1, "  lehmann representation: poles {}+{} / {}+{} (k = 0 / 1), G vs Lehmann {:.2e} (rel), N {:.8f} (Tr D {:.8f}, "
             "Lehmann {:.8f}), dropped {:.1e}, pruned weight {} ({:.1e})",
          outL.poles.part[0].size(), outL.poles.hole[0].size(), outL.poles.part[1].size(), outL.poles.hole[1].size(), errF / gmax,
          outL.nel_compressed, nF, outL.nel_lehmann, outL.dropped, outL.pruned_w, outL.pruned_w_weight);
  REQUIRE(errF / gmax < 1e-6);
  REQUIRE(std::abs(nF - outL.nel_compressed) < 1e-12);
  REQUIRE(std::abs(outL.nel_compressed - outL.nel_lehmann) < 1e-3);
  if (exact_model) REQUIRE(std::abs(outL.nel_compressed - N_ex) < 1e-3);

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
    auto oL1 = closure(self, H, Sp, Sh, zeta, bp, bh, gp, gh, p, nelec, T2, grl);
    for (long ik = 0; ik < nk; ++ik)
      for (auto s : {sector_t::particle, sector_t::hole}) {
        REQUIRE(oL1.poles(ik, s).size() == outL.poles(ik, s).size());
        if (outL.poles(ik, s).size() == 0) continue;
        d = std::max(d, nda::max_element(nda::abs(oL1.poles(ik, s).e - outL.poles(ik, s).e)));
        d = std::max(d, nda::max_element(nda::abs(oL1.poles(ik, s).v - outL.poles(ik, s).v)));
      }
    app_log(1, "  {} ranks vs 1 rank: max |diff| {:.1e}", comm.size(), d);
    REQUIRE(d <= 1e-12);
  }

  // S7g: concurrent k workers (single-rank communicator: this rank owns every k) give the same closure; the fast drivers
  // (gesdd SVD, Hermitian U-eigen) agree with the python path to roundoff-level effects
  {
    auto self = comm.split(comm.rank(), 0);
    auto diff = [&](closure_out_t const &a) {
      double d = std::abs(a.dmu - out.dmu);
      for (long ik = 0; ik < nk; ++ik) {
        REQUIRE(a.leh.e[ik].size() == out.leh.e[ik].size());
        d = std::max(d, nda::max_element(nda::abs(a.leh.e[ik] - out.leh.e[ik])));
        d = std::max(d, nda::max_element(nda::abs(a.poles.part[ik].coef - out.poles.part[ik].coef)));
        d = std::max(d, nda::max_element(nda::abs(a.poles.hole[ik].coef - out.poles.hole[ik].coef)));
      }
      return d;
    };
    utils::TimerManager T3;
    closure_params_t pw = p;
    pw.k_workers        = 2;
    const double dw     = diff(closure(self, H, Sp, Sh, zeta, bp, bh, gp, gh, pw, nelec, T3));
    closure_params_t pf = p;
    pf.svd_driver       = "gesdd";
    pf.ueig             = "cayley";
    auto of             = closure(self, H, Sp, Sh, zeta, bp, bh, gp, gh, pf, nelec, T3);
    // Lehmann G on the imaginary axis (relative), dmu, QP edges: the closure amplifies roundoff ~1e5 x at these settings
    double dG = 0.0, gm = 0.0;
    for (long iw = 0; iw < 60; ++iw) {
      const ComplexType z(0.0, 1e-2 * std::pow(10.0, 3.0 * iw / 59.0));
      for (long ik = 0; ik < nk; ++ik) {
        nda::matrix<ComplexType> G0(nb, nb), G1(nb, nb);
        G0() = ComplexType(0.0);
        G1() = ComplexType(0.0);
        for (long m = 0; m < out.leh.e[ik].size(); ++m)
          for (long i = 0; i < nb; ++i)
            for (long j = 0; j < nb; ++j) G0(i, j) += out.leh.v[ik](i, m) * std::conj(out.leh.v[ik](j, m)) / (z - out.leh.e[ik](m));
        for (long m = 0; m < of.leh.e[ik].size(); ++m)
          for (long i = 0; i < nb; ++i)
            for (long j = 0; j < nb; ++j) G1(i, j) += of.leh.v[ik](i, m) * std::conj(of.leh.v[ik](j, m)) / (z - of.leh.e[ik](m));
        dG = std::max(dG, nda::max_element(nda::abs(G1 - G0)));
        gm = std::max(gm, nda::max_element(nda::abs(G0)));
      }
    }
    const double dmu = std::abs(of.dmu - out.dmu), dedge = std::max(std::abs(of.e_homo - out.e_homo), std::abs(of.e_lumo - out.e_lumo));
    app_log(1, "  S7g: k_workers 2 vs 1: max |diff| {:.1e}; gesdd + cayley vs gesvd + schur: G(i w) {:.1e} (rel), dmu {:.1e}, "
               "QP edges {:.1e} Ha (the refit coefficients are ill-conditioned and not compared)", dw, dG / gm, dmu, dedge);
    REQUIRE(dw <= 1e-12);
    REQUIRE(dG / gm <= 1e-8);
    REQUIRE(dmu <= 1e-8);
    REQUIRE(dedge <= 1e-8);
  }
}
} // namespace

TEST_CASE("gw_line_closure_toy", "[gw_line][scf][closure]") {
  SECTION("exact 8-pole Sigma") { closure_toy(8); }
  SECTION("120-pole Sigma") { closure_toy(120); }
}

// ======================================================================================================================
// lih222: driver, restart, python parity
// ======================================================================================================================
namespace {

std::string gw_line_dir() { return std::string(PROJECT_SOURCE_DIR) + "/tests/unit_test_files/gw_line/"; }
double env_or(char const *nm, double d) { return std::getenv(nm) ? std::atof(std::getenv(nm)) : d; }
std::string lih_thc_file() { return gw_line_dir() + "lih222_thc/thc.eri.h5"; }

struct lih_t {
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;
  std::shared_ptr<mf::MF> mf;
  std::unique_ptr<methods::thc_reader_t> thc;
  std::string fixture;
  /// qe_lih222 (stored THC if present) or another fixture (THC built, nIpts = 8 nbnd), e.g. qe_lih223 (q != -q)
  explicit lih_t(std::string const &fx = "qe_lih222") : fixture(fx) {
    mpi = utils::make_unit_test_mpi_context();
    mf  = std::make_shared<mf::MF>(mf::default_MF(mpi, fixture));
    if (fixture == "qe_lih222" and std::filesystem::exists(lih_thc_file())) {
      thc = std::make_unique<methods::thc_reader_t>(mf, "incore", lih_thc_file());
    } else {
      app_log(1, "  [gw_line test] {} not found: building the THC (nIpts = 8 nbnd); run [.gw_line_dump] to store it",
              lih_thc_file());
      thc = std::make_unique<methods::thc_reader_t>(
          mf, methods::make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", "", "bdft", 1e-10, mf->ecutrho(), 1, 1024));
    }
  }
};

/// small settings shared by the restart and parity tests (and by gen_lih222_scf_ref.py)
ptree scf_params(std::string const &output, long niter, bool restart, std::string const &time_grid = "id",
                 std::string const &g_repr = "compressed") {
  ptree pt;
  pt.put("time_grid", time_grid);
  pt.put("g_repr", g_repr);
  pt.put("theta_deg", 20.0);
  pt.put("eps", 1e-8);
  pt.put("lam", 6.0);
  pt.put("lam_b", 4.0);
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
  pt.put("conv_thr", 1e-14);
  pt.put("t_chunk", 8);
  pt.put("ray_decades", 36.0);
  pt.put("restart", restart);
  pt.put("output", output);
  pt.put("spectra.enable", false);
  pt.put("checkpoint_sigma", "all");   // S7e: these tests read Sigma of every iteration from the checkpoint
  return pt;
}

void enable_spectra(ptree &pt, long nw) {
  pt.put("spectra.enable", true);
  ptree eta, v1, v2;
  v1.put("", 0.004);
  v2.put("", 0.01);
  eta.push_back({"", v1});
  eta.push_back({"", v2});
  pt.put_child("spectra.eta", eta);
  pt.put("spectra.wmin", -0.3);
  pt.put("spectra.wmax", 0.3);
  pt.put("spectra.nw", nw);
}

double maxdiff_poles(pole_data_t const &a, pole_data_t const &b) {
  double d = 0.0;
  for (long ik = 0; ik < a.nk; ++ik) {
    REQUIRE(a.part[ik].size() == b.part[ik].size());
    REQUIRE(a.hole[ik].size() == b.hole[ik].size());
    d = std::max(d, nda::max_element(nda::abs(a.part[ik].e - b.part[ik].e)));
    d = std::max(d, nda::max_element(nda::abs(a.hole[ik].e - b.hole[ik].e)));
    for (auto s : {sector_t::particle, sector_t::hole}) {
      auto const &x = a(ik, s);
      auto const &y = b(ik, s);
      REQUIRE(x.is_factorized() == y.is_factorized());
      if (x.size() == 0) continue;
      if (x.is_factorized()) d = std::max(d, nda::max_element(nda::abs(x.v - y.v)));
      else d = std::max(d, nda::max_element(nda::abs(x.coef - y.coef)));
    }
  }
  return d;
}

/// checkpoints are removed unless GW_LINE_TEST_KEEP is set (diagnostics against python)
void remove_file(boost::mpi3::communicator &comm, std::string const &f) {
  comm.barrier();
  if (comm.root() and std::getenv("GW_LINE_TEST_KEEP") == nullptr) {
    if (std::filesystem::exists(f)) std::filesystem::remove(f);
    const std::string sf = f.substr(0, f.size() - 3) + ".sigma.h5";   // S7e checkpoint_sigma = "last"
    if (f.size() > 3 and std::filesystem::exists(sf)) std::filesystem::remove(sf);
  }
  comm.barrier();
}

} // namespace

/// Writes the THC of the lih222 fixture (nIpts = 8 nbnd) and system.h5 (H0, KS eigenvalues, qk_to_k2, mu0, nelec) for
/// coqui/cayley/scripts/gen_lih222_scf_ref.py. Run once on 1 rank: test_gw_line_scf "[.gw_line_dump]".
TEST_CASE("gw_line_dump_lih222", "[.gw_line_dump]") {
  auto mpi = utils::make_unit_test_mpi_context();
  auto mf  = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222"));
  if (mpi->comm.root()) std::filesystem::create_directories(gw_line_dir() + "lih222_thc");
  mpi->comm.barrier();
  methods::thc_reader_t thc(
      mf, methods::make_thc_reader_ptree(mf->nbnd() * 8, "", "incore", lih_thc_file(), "bdft", 1e-10, mf->ecutrho(), 1, 1024));
  auto H0 = one_body_h0(*mf);
  const long nk = mf->nkpts(), nb = mf->nbnd(), nocc = long(std::llround(double(mf->nelec()) / 2.0));
  double homo = -1e300, lumo = 1e300;
  for (long ik = 0; ik < nk; ++ik)
    for (long n = 0; n < nb; ++n) (n < nocc ? homo : lumo) = (n < nocc) ? std::max(homo, mf->eigval()(0, ik, n)) : std::min(lumo, mf->eigval()(0, ik, n));
  write_system_h5(mpi->comm, gw_line_dir() + "lih222_thc/system.h5", *mf, thc.Np(), H0, 0.5 * (homo + lumo), true);
  // the driver's bases for the parity settings (scf_params): the pivoted-QR pole selection of the hole Sigma basis and of
  // the bosonic basis differs between this LAPACK and numpy/scipy's (near-tie pivots; both valid eps-bases), so the python
  // reference is run on THESE bases (gen_lih222_scf_ref.py overrides its own) to compare the SCF numerics like for like.
  if (mpi->comm.root()) {
    const double th = 20.0 * std::numbers::pi / 180.0, eps = 1e-8;
    line_basis_t bp(th, 6.0, eps, 6.0, 0.02, -1.0, 60.0), bh(th, 6.0, eps, 0.02, 6.0, -1.0, 60.0);
    line_basis_t gp(th, 6.0, eps, 6.0, 0.0, -1.0, 60.0), gh(th, 6.0, eps, 0.0, 6.0, -1.0, 60.0);
    numerics::line_dlr::bosonic_basis_t bos(th, 4.0, eps, 0.02);
    h5::file f(gw_line_dir() + "lih222_thc/bases.h5", 'w');
    h5::group g(f);
    nda::h5_write(g, "sigma_particle_w", bp.w, false);
    nda::h5_write(g, "sigma_hole_w", bh.w, false);
    nda::h5_write(g, "g_particle_w", gp.w, false);
    nda::h5_write(g, "g_hole_w", gh.w, false);
    nda::h5_write(g, "bos_nu", bos.nu, false);
    nda::h5_write(g, "bos_zeta_nodes", bos.zeta_nodes, false);
  }
  mpi->comm.barrier();
  app_log(1, "wrote {} (Np {}) and system.h5 (mu0 {:.8f}, nelec {})", lih_thc_file(), thc.Np(), 0.5 * (homo + lumo), mf->nelec());
}

/// S7f: the driver's real-pole bases for the parity settings with the bosonic range lam_b = GW_LINE_DUMP_LAMB (default 12)
/// -> lih222_thc/bases_lamb<L>.h5, read by BOTH the python reference (gen_lih222_scf_ref.py --bases) and the C++ parity
/// run (driver key bases_file): the column-pivoted QR pole selection differs between LAPACKs (near-tie pivots: Mac
/// Accelerate/OpenBLAS bosonic rank 104, Sigma 66+66, G 102+102 vs MKL 103, 65+66, 101+102 at lam_b 4), so a portable
/// parity test must not build them. Run once on 1 rank: test_gw_line_scf "[.gw_line_dump_bases]".
TEST_CASE("gw_line_dump_bases", "[.gw_line_dump_bases]") {
  auto mpi = utils::make_unit_test_mpi_context();
  const double lamb = env_or("GW_LINE_DUMP_LAMB", 12.0);
  if (mpi->comm.root()) {
    const double th = 20.0 * std::numbers::pi / 180.0, eps = 1e-8;
    line_basis_t bp(th, 6.0, eps, 6.0, 0.02, -1.0, 60.0), bh(th, 6.0, eps, 0.02, 6.0, -1.0, 60.0);
    line_basis_t gp(th, 6.0, eps, 6.0, 0.0, -1.0, 60.0), gh(th, 6.0, eps, 0.0, 6.0, -1.0, 60.0);
    numerics::line_dlr::bosonic_basis_t bos(th, lamb, eps, 0.02);
    const std::string fn = gw_line_dir() + "lih222_thc/bases_lamb" + std::to_string(long(std::llround(lamb))) + ".h5";
    h5::file f(fn, 'w');
    h5::group g(f);
    nda::h5_write(g, "sigma_particle_w", bp.w, false);
    nda::h5_write(g, "sigma_hole_w", bh.w, false);
    nda::h5_write(g, "g_particle_w", gp.w, false);
    nda::h5_write(g, "g_hole_w", gh.w, false);
    nda::h5_write(g, "bos_nu", bos.nu, false);
    nda::h5_write(g, "bos_zeta_nodes", bos.zeta_nodes, false);
    h5::h5_write(g, "lam_b", lamb);
    app_log(1, "wrote {}: Sigma {}+{}, G {}+{}, bosonic {} ({} nodes)", fn, bp.rank, bh.rank, gp.rank, gh.rank, bos.rank,
            bos.zeta_nodes.size());
  }
  mpi->comm.barrier();
}

namespace {
void restart_test(std::string const &tg, std::string const &gr = "compressed") {
  lih_t L;
  auto &comm = L.mpi->comm;
  const std::string fa = "gw_line_rsA_" + tg + gr, fb = "gw_line_rsB_" + tg + gr;
  auto t0 = std::chrono::steady_clock::now();
  auto A  = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, scf_params(fa, 3, false, tg, gr));
  auto t1 = std::chrono::steady_clock::now();
  auto B2 = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, scf_params(fb, 2, false, tg, gr));
  auto pb = scf_params(fb, 3, true, tg, gr);
  enable_spectra(pb, 41);
  auto B3 = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pb);
  auto t2 = std::chrono::steady_clock::now();
  REQUIRE(A.history.size() == 3);
  REQUIRE(B3.history.size() == 3);
  REQUIRE(B3.history[1].mu == B2.history[1].mu);
  const double dmu = std::abs(A.mu - B3.mu), dpo = maxdiff_poles(A.poles, B3.poles);
  const double dF  = nda::max_element(nda::abs(A.F - B3.F));
  const double dSp = nda::max_element(nda::abs(A.Sig_p - B3.Sig_p)), dSh = nda::max_element(nda::abs(A.Sig_h - B3.Sig_h));
  REQUIRE(A.poles.is_factorized() == (gr == "lehmann"));
  REQUIRE(B3.poles.is_factorized() == (gr == "lehmann"));
  app_log(1, "[restart] time_grid {}, g_repr {}, ranks {}: 3 iterations ({:.1f} s) vs 2 + restart + 1 ({:.1f} s): |dmu| {:.1e}, "
             "poles {:.1e}, F {:.1e}, Sigma_p {:.1e}, Sigma_h {:.1e}",
          tg, gr, comm.size(), std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count(), dmu, dpo,
          dF, dSp, dSh);
  for (long i = 0; i < 3; ++i) {
    app_log(1, "  iter {}: mu {:.10f} / {:.10f}  gap {:.8f} / {:.8f}  dSigma {:.3e} / {:.3e}  t-nodes ({}) {}+{} {}+{}", i + 1,
            A.history[i].mu, B3.history[i].mu, A.history[i].gap, B3.history[i].gap, A.history[i].dSigma, B3.history[i].dSigma,
            B3.history[i].time_grid, B3.history[i].nt_pi_p, B3.history[i].nt_pi_h, B3.history[i].nt_sig_p,
            B3.history[i].nt_sig_h);
    // the restored history carries the time grid and the node counts
    REQUIRE(B3.history[i].time_grid == tg);
    REQUIRE(B3.history[i].g_repr == gr);
    REQUIRE(B3.history[i].ng_max == A.history[i].ng_max);
    REQUIRE(B3.history[i].g_emin == A.history[i].g_emin);
    REQUIRE(B3.history[i].nt_pi_p == A.history[i].nt_pi_p);
    REQUIRE(B3.history[i].nt_sig_h == A.history[i].nt_sig_h);
    REQUIRE(A.history[i].nt_pi_p > 0);
  }
  REQUIRE(dmu == 0.0);
  REQUIRE(dpo == 0.0);
  REQUIRE(dF == 0.0);
  REQUIRE(dSp == 0.0);
  REQUIRE(dSh == 0.0);

  // spectra: written by the restarted run; a restart with niter already reached only recomputes them
  REQUIRE(B3.spectra.has_value());
  auto B4 = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pb);
  REQUIRE(B4.history.size() == 3);
  REQUIRE(B4.spectra.has_value());
  const double dA = nda::max_element(nda::abs(B4.spectra->A_diag - B3.spectra->A_diag));
  double sum_tr = 0.0;
  for (long ik = 0; ik < B3.spectra->A_trace.extent(1); ++ik)
    for (long iw = 0; iw < B3.spectra->A_trace.extent(2); ++iw) sum_tr += B3.spectra->A_trace(1, ik, iw);
  app_log(1, "  spectra: {} x {} x {} x {}, QP edges {:.6f} {:.6f} (rel. mu), niter-reached restart vs restart: {:.1e}, "
             "sum Tr A (eta 0.01) {:.4f}",
          B3.spectra->A_diag.extent(0), B3.spectra->A_diag.extent(1), B3.spectra->A_diag.extent(2), B3.spectra->A_diag.extent(3),
          B3.spectra->e_homo, B3.spectra->e_lumo, dA, sum_tr);
  REQUIRE(dA == 0.0);
  if (comm.root()) {
    h5::file f(fb + ".gw_line.h5", 'r');
    h5::group g(f);
    nda::array<double, 4> Ad;
    nda::h5_read(g, "spectra/A_k_w_diag", Ad);
    REQUIRE(Ad.shape() == B3.spectra->A_diag.shape());
    long fi = 0;
    h5::h5_read(g, "scf_line/final_iter", fi);
    REQUIRE(fi == 3);
  }
  remove_file(comm, fa + ".gw_line.h5");
  remove_file(comm, fb + ".gw_line.h5");
}
} // namespace

namespace {
/// S9a: the gygi driver: restart, head checkpoint, Madelung terms
void gygi_test() {
  lih_t L;
  auto &comm = L.mpi->comm;
  auto &mf   = *L.mf;
  const std::string fa = "gw_line_gyA", fb = "gw_line_gyB", fi = "gw_line_gyI";
  auto par = [&](std::string const &f, long n, bool rs, std::string const &div) {
    auto pt = scf_params(f, n, rs, "id", "lehmann");
    pt.put("lam_b", -1.0);   // auto (S7c): the bosonic basis covers the Pi transitions of the Lehmann poles
    pt.put("div_treatment", div);
    return pt;
  };
  auto A  = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, mf, par(fa, 2, false, "gygi"));
  auto B1 = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, mf, par(fb, 1, false, "gygi"));
  auto B2 = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, mf, par(fb, 2, true, "gygi"));
  auto I  = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, mf, par(fi, 1, false, "ignore_g0"));
  REQUIRE(A.history.size() == 2);
  REQUIRE(B2.history.size() == 2);
  REQUIRE(A.eps_inf.size() == 2);
  REQUIRE(B2.eps_inf.size() == 1);
  const double dmu = std::abs(A.mu - B2.mu), dpo = maxdiff_poles(A.poles, B2.poles);
  const double dF  = nda::max_element(nda::abs(A.F - B2.F));
  const double dSp = nda::max_element(nda::abs(A.Sig_p - B2.Sig_p)), dSh = nda::max_element(nda::abs(A.Sig_h - B2.Sig_h));
  app_log(1, "[gygi] ranks {}: eps_inf it 1 / 2 = {:.10f} / {:.10f} (restart: {:.10f}); mu it 1 / 2 = {:.10f} / {:.10f} "
             "(ignore_g0 it 1 {:.10f}); gap it 1 / 2 = {:.6f} / {:.6f} eV (ignore_g0 it 1 {:.6f} eV)",
          comm.size(), A.eps_inf[0], A.eps_inf[1], B2.eps_inf[0], A.history[0].mu, A.history[1].mu, I.history[0].mu,
          A.history[0].gap * 27.211386, A.history[1].gap * 27.211386, I.history[0].gap * 27.211386);
  app_log(1, "  restart 1 + 1 vs 2: |dmu| {:.1e}, poles {:.1e}, F {:.1e}, Sigma_p {:.1e}, Sigma_h {:.1e}; eps_inf it 2 {:.1e}", dmu,
          dpo, dF, dSp, dSh, std::abs(A.eps_inf[1] - B2.eps_inf[0]));
  REQUIRE(dmu == 0.0);
  REQUIRE(dpo == 0.0);
  REQUIRE(dF == 0.0);
  REQUIRE(dSp == 0.0);
  REQUIRE(dSh == 0.0);
  REQUIRE(A.eps_inf[1] == B2.eps_inf[0]);
  REQUIRE(A.eps_inf[0] > 1.0);
  if (comm.root()) {
    h5::file f(fa + ".gw_line.h5", 'r');
    h5::group g(f);
    std::string hfd;
    h5::h5_read(g, "input/hf_div_treatment", hfd);
    REQUIRE(hfd == "gygi");
    const long nq = mf.nqpts(), nk = mf.nkpts(), nb = mf.nbnd();
    for (long it = 1; it <= 2; ++it) {
      auto hg = g.open_group("scf_line/iter" + std::to_string(it) + "/head");
      nda::array<ComplexType, 2> hn, hr, hh;
      nda::array<ComplexType, 1> h0n, h0r, h0h, z;
      nda::array<double, 1> nu, cw;
      double eps = 0.0, mad = 0.0;
      nda::h5_read(hg, "h_nodes", hn);
      nda::h5_read(hg, "h_res", hr);
      nda::h5_read(hg, "h_res_hole", hh);
      nda::h5_read(hg, "h0_nodes", h0n);
      nda::h5_read(hg, "h0_res", h0r);
      nda::h5_read(hg, "h0_res_hole", h0h);
      nda::h5_read(hg, "zeta", z);
      nda::h5_read(hg, "nu", nu);
      nda::h5_read(hg, "q_weights", cw);
      h5::h5_read(hg, "eps_inf", eps);
      h5::h5_read(hg, "madelung", mad);
      REQUIRE(hn.extent(0) == nq);
      REQUIRE(hn.extent(1) == z.size());
      REQUIRE(hr.extent(0) == nq);
      REQUIRE(hr.extent(1) == nu.size());
      REQUIRE(hh.shape() == hr.shape());
      REQUIRE(cw.size() == nq);
      REQUIRE(eps == A.eps_inf[it - 1]);
      REQUIRE(mad == mf.madelung());
      auto h0e = methods::gw_line::head_eval(h0r, h0h, nu, z);
      const double e = nda::max_element(nda::abs(h0e - h0n)) / nda::max_element(nda::abs(h0n));
      app_log(1, "  head checkpoint it {}: h_nodes {} x {}, h_res {} x {}, eps_inf {:.8f}, h0 residues vs nodes {:.1e}", it,
              hn.extent(0), hn.extent(1), hr.extent(0), hr.extent(1), eps, e);
      REQUIRE(e <= 1e-8);
    }
    // F(iter 0): gygi - ignore_g0 = -madelung D_KS
    h5::file fI(fi + ".gw_line.h5", 'r');
    h5::group gI(fI);
    nda::array<ComplexType, 3> Fg, Fi;
    nda::h5_read(g, "scf_line/iter0/F", Fg);
    nda::h5_read(gI, "scf_line/iter0/F", Fi);
    const long nocc = long(std::llround(double(mf.nelec()) / 2.0));
    double dFk = 0.0;
    for (long ik = 0; ik < nk; ++ik)
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j)
          dFk = std::max(dFk, std::abs(Fg(ik, i, j) - Fi(ik, i, j) + ((i == j and i < nocc) ? mf.madelung() : 0.0)));
    app_log(1, "  F(iter0) gygi - ignore_g0 + madelung D_KS: {:.1e}", dFk);
    REQUIRE(dFk <= 1e-14);
  }
  for (auto const &f : {fa, fb, fi}) remove_file(comm, f + ".gw_line.h5");
  for (auto const &f : {fa, fb, fi}) remove_file(comm, f + ".gw_line.sigma.h5");
}
} // namespace

TEST_CASE("gw_line_scf_gygi", "[gw_line][scf][gygi]") { gygi_test(); }

namespace {
ptree arr_child(std::vector<double> const &v) {
  ptree a;
  for (double x : v) {
    ptree c;
    c.put("", x);
    a.push_back({"", c});
  }
  return a;
}
/// S9b: optics in the driver: after the loop, from a restart with nothing to iterate, from a checkpoint WITHOUT head groups
/// (recomputed from the final poles, = the head of one more iteration), the flatter-line pass; the h5 layout. perf 7.2: the runs
/// are not converged, so the SCF-angle line is always recomputed from the final poles (optics.poles = "final")
void optics_driver_test() {
  lih_t L;
  auto &comm = L.mpi->comm;
  auto &mf   = *L.mf;
  const std::string fa = "gw_line_opA", fc = "gw_line_opC", fd = "gw_line_opD";
  auto par = [&](std::string const &f, long n, bool rs, bool optics) {
    auto pt = scf_params(f, n, rs, "id", "lehmann");
    pt.put("lam_b", -1.0);
    pt.put("div_treatment", "gygi");
    pt.put("eps", 1e-10);
    if (optics) {
      pt.put("optics.enable", true);
      pt.put("optics.wmin", 0.0);
      pt.put("optics.wmax", 1.0);
      pt.put("optics.nw", 201);
      pt.add_child("optics.eta", arr_child({0.01}));
      pt.add_child("optics.eta_rel", arr_child({0.05}));
      pt.put("optics.theta_deg", 10.0);
    }
    return pt;
  };
  auto A = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, mf, par(fa, 2, false, true));
  REQUIRE(A.optics_theta.size() == 2);
  REQUIRE(A.optics_q0.size() == 2);
  if (comm.root()) {
    std::filesystem::copy_file(fa + ".gw_line.h5", fc + ".gw_line.h5", std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(fa + ".gw_line.h5", fd + ".gw_line.h5", std::filesystem::copy_options::overwrite_existing);
    h5::file f(fc + ".gw_line.h5", 'a');
    h5::group g(f);
    for (long it = 1; it <= 2; ++it) g.open_group("scf_line/iter" + std::to_string(it)).unlink("head");
    g.unlink("optics");
  }
  comm.barrier();
  // B: restart with nothing to iterate (head group of the checkpoint) -> identical optics
  auto B = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, mf, par(fa, 2, true, true));
  REQUIRE(B.history.size() == 2);
  REQUIRE(B.optics_q0.size() == 2);
  double dAB = 0.0;
  for (long l = 0; l < 2; ++l)
    for (size_t k = 0; k < A.optics_q0[l].val.size(); ++k)
      dAB = std::max(dAB, double(nda::max_element(nda::abs(A.optics_q0[l].val[k] - B.optics_q0[l].val[k]))));
  // C: no head groups -> recomputed; D: one more iteration (its head = the same final poles of iteration 2)
  auto C = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, mf, par(fc, 2, true, true));
  auto D = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, mf, par(fd, 3, true, false));
  REQUIRE(C.optics_q0.size() == 2);
  double dCD = 0.0, sCD = 0.0;
  if (comm.root()) {
    nda::array<ComplexType, 2> hc, hd;
    h5::file f1(fc + ".gw_line.h5", 'r'), f2(fd + ".gw_line.h5", 'r');
    h5::group g1(f1), g2(f2);
    nda::h5_read(g1, "optics/theta20.0/h_nodes", hc);
    nda::h5_read(g2, "scf_line/iter3/head/h_nodes", hd);
    REQUIRE(hc.shape() == hd.shape());
    dCD = nda::max_element(nda::abs(hc - hd));
    sCD = nda::max_element(nda::abs(hd));
    std::string src;
    h5::h5_read(g1, "optics/theta20.0/source", src);
    REQUIRE(src.find("recomputed") != std::string::npos);
    // layout
    for (std::string const &t : {"theta20.0", "theta10.0"}) {
      auto tg = g1.open_group("optics/" + t);
      REQUIRE(tg.has_subgroup("q0"));
      long nfin = 0;
      for (long q = 0; q < mf.nqpts(); ++q) nfin += tg.has_subgroup("iq" + std::to_string(q));
      REQUIRE(nfin == mf.nqpts() - 1);
      auto q0 = tg.open_group("q0");
      nda::array<double, 2> e2, e2e, lo;
      nda::h5_read(q0, "eps2", e2);
      nda::h5_read(q0, "eps2_err", e2e);
      nda::h5_read(q0, "loss", lo);
      REQUIRE(e2.extent(0) == 2);
      REQUIRE(e2.extent(1) == 201);
      REQUIRE(e2e.shape() == e2.shape());
      REQUIRE(nda::min_element(lo) >= -1e-8 * nda::max_element(lo));
      REQUIRE(nda::min_element(e2) >= -1e-8 * nda::max_element(e2));
    }
  }
  comm.broadcast_n(&dCD, 1, 0);
  comm.broadcast_n(&sCD, 1, 0);
  auto const &o = A.optics_q0[0];
  app_log(1, "[optics driver] ranks {}: q0 eps_inf {:.8f} (driver eps_inf it 2 {:.8f}), f-sum {:.5f} / {:.5f}, K {} / {}; restart (checkpoint "
             "head) vs in-run optics {:.1e}; recomputed head vs iteration-3 head {:.1e} (rel); flat pass 10 deg eps_inf {:.8f}",
          comm.size(), o.eps_inf_h, A.eps_inf.back(), o.fsum_h, o.fsum_m, o.K_h, o.K_m, dAB, dCD / sCD, A.optics_q0[1].eps_inf_h);
  REQUIRE(dAB == 0.0);
  REQUIRE(dCD <= 1e-10 * sCD);
  // perf 7.2: A is not converged (2 iterations): its SCF-angle optics come from a pass on the FINAL poles (= the head of iteration 3)
  REQUIRE(std::abs(o.eps_inf_h - D.eps_inf.back()) <= 1e-6 * D.eps_inf.back());
  REQUIRE(std::abs(A.optics_q0[1].eps_inf_h - D.eps_inf.back()) <= 1e-6 * D.eps_inf.back());
  for (auto const &f : {fa, fc, fd}) remove_file(comm, f + ".gw_line.h5");
  for (auto const &f : {fa, fc, fd}) remove_file(comm, f + ".gw_line.sigma.h5");
}
} // namespace

TEST_CASE("gw_line_scf_optics", "[gw_line][scf][optics]") { optics_driver_test(); }

TEST_CASE("gw_line_scf_restart", "[gw_line][scf][restart]") {
  SECTION("time_grid id") { restart_test("id"); }
  SECTION("time_grid gl") { restart_test("gl"); }
  SECTION("time_grid id, g_repr lehmann") { restart_test("id", "lehmann"); }
}

namespace {
nda::array<ComplexType, 4> read_sigma_total(boost::mpi3::communicator &comm, std::string const &file, long it);

/// S7f: a measured-vs-gate line; returns pass/fail
bool gate_line(std::string const &what, double value, double gate) {
  const bool ok = std::abs(value) <= gate;
  app_log(1, "    {:<34s} {:10.3e} <= gate {:10.3e}  {}", what, std::abs(value), gate, ok ? "ok" : "FAIL");
  return ok;
}
} // namespace

/**
 * S7f gates. The noise floor of the method at the parity settings (compressed, gl, lam_b 12, K 8, eps 1e-8, test settings):
 * [.scf_noise] GW_LINE_NOISE_SETTINGS=test LAMB=auto REPR=cmp TG=gl, relative noise 1e-13 on H0, 4 runs, spread of mu and
 * of the QP gap (meV) and max|dSigma|/max|Sigma| per iteration; the values below are the MAX over the Mac (Accelerate /
 * OpenBLAS, 2 ranks) and rusty (MKL, gcc, 2 ranks). Gate of iteration it > 1 = max(strict, PARITY_K x floor(it)); iteration 1
 * (KS poles, identical bases): strict 1e-3 meV and 1e-10 in Sigma. The yardstick is relative noise 2e-14 on Sigma of iteration 1
 * (GW_LINE_NOISE_MODE=sigma GW_LINE_NOISE_AMP=2e-14) = the measured python-vs-C++ difference of Sigma at iteration 1 (1.9e-14):
 * noise on H0 (1e-13) is far too weak a probe (it does not pass through the ill-conditioned sector fits: spread <= 3e-4 meV),
 * noise on Sigma at 1e-12 already saturates (meV at iteration 3). 3 iterations: the 2e-14 floor is 0.49 meV at iteration 4
 * and 7.5 meV at 5 (the K 8 attractor), so later iterations cannot be gated meaningfully. Measured (Mac, 1 / 2 ranks):
 * iteration 2 |dmu| 2.2e-5 / 5.3e-5 meV, Sigma 7.7e-8 / 2.2e-7; iteration 3 8.4e-4 meV, 1.1e-6.
 */
constexpr double PARITY_K = 10.0;
constexpr long PARITY_NIT = 3;
// Mac it2 {1e-4, 2e-4, 4.7e-7}, it3 {1.2e-2, 3.2e-3, 6.7e-6}; rusty it2 {1.2e-3, 2.0e-3, 2.9e-7}, it3 {1.8e-3, 3.0e-3, 1.2e-5}
constexpr std::array<std::array<double, 3>, PARITY_NIT> parity_floor = {{{0, 0, 0}, {1.2e-3, 2.0e-3, 4.71e-7}, {1.22e-2, 3.2e-3, 1.2e-5}}};

TEST_CASE("gw_line_scf_parity", "[gw_line][scf][parity]") {
  const std::string ref = gw_line_dir() + "lih222_scf_ref.h5";
  if (not std::filesystem::exists(ref)) {
    app_log(1, "[parity] {} not found: run coqui/cayley/scripts/gen_lih222_scf_ref.py", ref);
    FAIL("missing python reference");
  }
  lih_t L;
  auto &comm = L.mpi->comm;
  // reference
  nda::array<double, 1> mu_r, gap_r, nel_r, dS_r;
  nda::array<long, 1> idx;
  nda::array<ComplexType, 4> Sig_r;   // (niter, n_sel, nb, nb), total Sigma at k = 0
  long niter = 0;
  double lam_b = 4.0;
  std::string bases;
  {
    h5::file f(ref, 'r');
    h5::group g(f);
    nda::h5_read(g, "mu", mu_r);
    nda::h5_read(g, "gap", gap_r);
    nda::h5_read(g, "nelec", nel_r);
    nda::h5_read(g, "dSigma", dS_r);
    nda::h5_read(g, "node_index", idx);
    nda::array<double, 4> re, im;
    nda::h5_read(g, "Sigma_k0_re", re);
    nda::h5_read(g, "Sigma_k0_im", im);
    Sig_r = nda::array<ComplexType, 4>(re.shape());
    for (long a = 0; a < re.size(); ++a) Sig_r.data()[a] = ComplexType(re.data()[a], im.data()[a]);
    niter = mu_r.size();
    h5::h5_read_attribute(g, "lam_b", lam_b);
    h5::h5_read_attribute(g, "bases", bases);
  }
  REQUIRE(lam_b == 12.0);          // S7f: the well-conditioned bosonic range (lam_b 4 amplifies roundoff ~1e5 x, S7c)
  REQUIRE(not bases.empty());      // python ran on the dumped C++ bases
  REQUIRE(niter >= PARITY_NIT);
  const std::string bfile = gw_line_dir() + "lih222_thc/" + bases;
  // platform report: the bases this LAPACK would build vs the dumped ones (pivoted-QR near ties differ between LAPACKs)
  {
    const double th = 20.0 * std::numbers::pi / 180.0, eps = 1e-8;
    line_basis_t bh(th, 6.0, eps, 0.02, 6.0, -1.0, 60.0);
    numerics::line_dlr::bosonic_basis_t bos(th, lam_b, eps, 0.02);
    nda::array<double, 1> wh, nu;
    h5::file f(bfile, 'r');
    h5::group g(f);
    nda::h5_read(g, "sigma_hole_w", wh);
    nda::h5_read(g, "bos_nu", nu);
    app_log(1, "[parity] bases of this platform vs {}: Sigma hole rank {} / {} (max|dw| {:.1e}), bosonic rank {} / {} (max|dnu| {:.1e}); "
               "the run uses the file",
            bases, bh.rank, wh.size(), bh.rank == wh.size() ? double(nda::max_element(nda::abs(bh.w - wh))) : -1.0, bos.rank, nu.size(),
            bos.rank == nu.size() ? double(nda::max_element(nda::abs(bos.nu - nu))) : -1.0);
  }
  const std::string fo = "gw_line_parity";
  auto pt = scf_params(fo, PARITY_NIT, false, "gl", "compressed");
  pt.put("lam_b", lam_b);
  pt.put("bases_file", bfile);
  pt.put("tol_gram_eps", 0.0);   // python's closure: the Gram cut is tol_gram only
  auto t0 = std::chrono::steady_clock::now();
  auto R  = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pt);
  const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  REQUIRE(long(R.history.size()) == PARITY_NIT);
  const long nb = R.F.extent(1);
  app_log(1, "[parity] ranks {}, {} of {} reference iterations in {:.1f} s ({} stored nodes of k = 0), lam_b {}, bases {}", comm.size(),
          PARITY_NIT, niter, dt, idx.size(), lam_b, bases);
  app_log(1, "  iter |   mu C++ (Ha)    mu py (Ha)   d(meV) |  gap C++ (eV)  gap py (eV)  d(meV) |  N C++      N py     |  "
             "dSigma C++  dSigma py | Sigma rel");
  bool ok = true;
  for (long it = 0; it < PARITY_NIT; ++it) {
    auto S = read_sigma_total(comm, fo + ".gw_line.h5", it + 1);
    double dmax = 0.0, smax = 0.0;
    for (long n = 0; n < idx.size(); ++n)
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) {
          dmax = std::max(dmax, std::abs(S(0, idx(n), i, j) - Sig_r(it, n, i, j)));
          smax = std::max(smax, std::abs(Sig_r(it, n, i, j)));
        }
    auto const &h     = R.history[it];
    const double dmu  = (h.mu - mu_r(it)) * 27.211386e3, dgap = (h.gap - gap_r(it)) * 27.211386e3;
    app_log(1, "  {:4d} | {:.8f}  {:.8f}  {:+7.3f} | {:.6f}     {:.6f}    {:+7.3f} | {:.6f}  {:.6f} | {:.3e}  {:.3e} | {:.2e}",
            it + 1, h.mu, mu_r(it), dmu, h.gap * 27.211386, gap_r(it) * 27.211386, dgap, h.nelec, nel_r(it), h.dSigma, dS_r(it),
            dmax / smax);
    auto const &fl = parity_floor[it];
    const double gmu = (it == 0) ? 1e-3 : std::max(1e-3, PARITY_K * fl[0]);
    const double gga = (it == 0) ? 1e-3 : std::max(1e-3, PARITY_K * fl[1]);
    const double gsi = (it == 0) ? 1e-10 : std::max(1e-10, PARITY_K * fl[2]);
    ok = gate_line("|dmu| (meV)", dmu, gmu) and ok;
    ok = gate_line("|dgap| (meV)", dgap, gga) and ok;
    ok = gate_line("Sigma(k=0) rel", dmax / smax, gsi) and ok;
  }
  REQUIRE(ok);
  remove_file(comm, fo + ".gw_line.h5");
}

namespace {
/// total Sigma at all k and nodes of iteration it from a checkpoint (root reads, broadcast)
nda::array<ComplexType, 4> read_sigma_total(boost::mpi3::communicator &comm, std::string const &file, long it) {
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
} // namespace

/**
 * S7f gates of [id_vs_gl] and [lehmann]: the id-vs-gl difference after iteration 1 is the SCF's response to the iteration-1
 * kernel difference (the ID error, ~0.3 time_eps relative in Sigma), which the closure amplifies like any noise (S7f study:
 * no hard decision flips, continuous amplification ~1e5 of moment noise into G at K 8). Yardstick = [.scf_noise]
 * GW_LINE_NOISE_MODE=sigma at the amplitude of that difference (3e-9 for time_eps 1e-8, 3e-11 for 1e-10), test settings,
 * lam_b auto, time_grid id, default closure (tol_gram_eps = 1): spread of mu / gap (meV) and max|dSigma|/max|Sigma| at
 * iteration 2, max over the Mac and rusty. Gate of iteration 2 = max(strict, IDGL_K x floor); iteration 1 (KS poles):
 * dSigma <= 10 time_eps, |dmu|, |dgap| <= 1e-3 meV. Only 2 iterations are compared: from iteration 3 on the spread grows
 * to the size of the K = 8 attractor (3e-11: 0.3 meV at it 3, 1-8 meV at it 6; 3e-9: 1-9 meV), where no gate is meaningful.
 * The time_eps 1e-8 gate (3e-9: 1.4 meV) is loose by nature; the tight checks are iteration 1 and time_eps 1e-10 at iteration 2.
 */
constexpr double IDGL_K = 5.0;
using floor_tab_t = std::vector<std::array<double, 3>>;
// default closure (tol_gram_eps = 1 -> Gram cut 1e-8 at eps 1e-8); time_eps 1e-8 -> 3e-9, 1e-10 -> 3e-11 (lehmann; compressed
// identical within 1.3x)
// Mac 3e-9 {1.39, 0.97, 2.9e-3}, 3e-11 {4.4e-3, 5.7e-3, 3.1e-5}; rusty 3e-9 {0.84, 0.26, 2.8e-3}, 3e-11 {4.3e-3, 4.4e-3, 3.2e-5}
const floor_tab_t floor_3e9  = {{0, 0, 0}, {1.39, 0.97, 2.91e-3}};
const floor_tab_t floor_3e11 = {{0, 0, 0}, {4.35e-3, 5.69e-3, 3.23e-5}};

namespace {
/// gates (mu meV, gap meV, Sigma rel) of iteration it (0-based) from a floor table
std::array<double, 3> idgl_gate(floor_tab_t const &fl, long it, double teps) {
  if (it == 0) return {1e-3, 1e-3, 10.0 * teps};
  utils::check(it < long(fl.size()), "idgl_gate: no floor for iteration {}", it + 1);
  return {std::max(1e-3, IDGL_K * fl[it][0]), std::max(1e-3, IDGL_K * fl[it][1]), std::max(10.0 * teps, IDGL_K * fl[it][2])};
}
} // namespace

TEST_CASE("gw_line_scf_id_vs_gl", "[gw_line][scf][id_vs_gl]") {
  lih_t L;
  auto &comm       = L.mpi->comm;
  const long niter = 2;   // S7f: from iteration 3 on the spread saturates at ~10 meV (the K 8 attractor), see idgl_gate
  const double meV = 27.211386e3;
  struct run_t {
    std::string name, file;
    methods::gw_line::gw_line_result_t R;
    double time = 0.0;
  };
  std::vector<run_t> runs;
  for (auto const &[nm, tg, teps] : std::vector<std::tuple<std::string, std::string, double>>{
           {"gl", "gl", 0.0}, {"id(1e-8)", "id", 1e-8}, {"id(1e-10)", "id", 1e-10}}) {
    run_t r;
    r.name = nm;
    r.file = "gw_line_idgl_" + std::to_string(runs.size());
    // S7f: the production representation and bosonic range (lehmann, lam_b auto = 12) at the test settings
    auto pt = scf_params(r.file, niter, false, tg, "lehmann");
    pt.put("lam_b", -1.0);
    if (teps > 0.0) pt.put("time_eps", teps);
    auto t0 = std::chrono::steady_clock::now();
    r.R     = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pt);
    r.time  = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    REQUIRE(long(r.R.history.size()) == niter);
    runs.push_back(std::move(r));
  }
  auto const &G = runs[0];
  app_log(1, "[id_vs_gl] lih222, test settings (eps 1e-8, K 8, lehmann, lam_b auto), ranks {}: {} iterations, wall gl {:.1f} s, "
             "id(1e-8) {:.1f} s, id(1e-10) {:.1f} s",
          comm.size(), niter, runs[0].time, runs[1].time, runs[2].time);
  app_log(1, "  run        iter | t-nodes Pi    Sigma   |   mu (Ha)       dmu vs gl (meV) |  gap (eV)   dgap vs gl (meV) | "
             "max|dSigma|/max|Sigma| vs gl");
  bool ok = true;
  for (long ir = 0; ir < long(runs.size()); ++ir) {
    auto const &r = runs[ir];
    for (long it = 0; it < niter; ++it) {
      auto const &h = r.R.history[it];
      auto Sg       = read_sigma_total(comm, G.file + ".gw_line.h5", it + 1);
      auto Sr       = read_sigma_total(comm, r.file + ".gw_line.h5", it + 1);
      const double ds = nda::max_element(nda::abs(Sr - Sg)) / nda::max_element(nda::abs(Sg));
      const double dmu = (h.mu - G.R.history[it].mu) * meV, dgap = (h.gap - G.R.history[it].gap) * meV;
      app_log(1, "  {:<10s} {:4d} | {:4d}+{:<4d} {:4d}+{:<4d} | {:.8f}  {:+10.4f}     | {:.6f}   {:+10.4f}      | {:.2e}", r.name,
              it + 1, h.nt_pi_p, h.nt_pi_h, h.nt_sig_p, h.nt_sig_h, h.mu, dmu, h.gap * 27.211386, dgap, ds);
      if (ir == 0) continue;
      REQUIRE(h.time_grid == "id");
      const bool fine   = (r.name == "id(1e-10)");
      const double teps = fine ? 1e-10 : 1e-8;
      auto g            = idgl_gate(fine ? floor_3e11 : floor_3e9, it, teps);
      ok = gate_line("|dmu| (meV)", dmu, g[0]) and ok;
      ok = gate_line("|dgap| (meV)", dgap, g[1]) and ok;
      ok = gate_line("max|dSigma|/max|Sigma|", ds, g[2]) and ok;
    }
  }
  REQUIRE(ok);
  for (auto const &r : runs) remove_file(comm, r.file + ".gw_line.h5");
}

/// Diagnostic (hidden): kernels on the poles of a checkpoint iteration (GW_LINE_DIAG_FILE, GW_LINE_DIAG_ITER), default GL
/// rays and ID grids vs a refined GL reference (ray_decades 60, smin 1e-7, 6 panels/e-fold, 24 nodes/panel).
TEST_CASE("gw_line_time_id_poles", "[.time_id_poles]") {
  using namespace methods::gw_line;
  using numerics::line_dlr::time_ray_t;
  using numerics::line_dlr::sector_t;
  const char *fenv = std::getenv("GW_LINE_DIAG_FILE");
  if (fenv == nullptr) return;
  const long iter = std::getenv("GW_LINE_DIAG_ITER") ? std::atol(std::getenv("GW_LINE_DIAG_ITER")) : 1;
  lih_t L;
  auto &mpi  = *L.mpi;
  auto &comm = mpi.comm;
  auto &mf   = *L.mf;
  const long nk = mf.nkpts(), nb = mf.nbnd(), nq = mf.nqpts(), Np = L.thc->Np();
  pole_data_t pd;
  pd.nk = nk; pd.nb = nb; pd.part.resize(nk); pd.hole.resize(nk);
  {
    h5::file f(fenv, 'r');
    h5::group g(f);
    auto pg = g.open_group("scf_line/iter" + std::to_string(iter) + "/poles");
    for (auto s : {sector_t::particle, sector_t::hole}) {
      const std::string nm = (s == sector_t::particle) ? "particle" : "hole";
      nda::array<long, 1> cnt; nda::array<double, 1> e; nda::array<ComplexType, 3> c; nda::array<ComplexType, 2> v;
      const bool fact = pg.has_dataset(nm + "_v");
      nda::h5_read(pg, nm + "_counts", cnt); nda::h5_read(pg, nm + "_e", e);
      if (fact) nda::h5_read(pg, nm + "_v", v);
      else nda::h5_read(pg, nm + "_coef", c);
      long o = 0;
      for (long ik = 0; ik < nk; ++ik) {
        auto &ps = (s == sector_t::particle) ? pd.part[ik] : pd.hole[ik];
        nda::array<double, 1> es(e(nda::range(o, o + cnt(ik))));
        if (fact) {
          nda::array<ComplexType, 2> vs(nb, cnt(ik));
          for (long m = 0; m < cnt(ik); ++m)
            for (long i = 0; i < nb; ++i) vs(i, m) = v(o + m, i);
          ps = pole_sector_t::factorized_form(std::move(es), std::move(vs));
        } else {
          ps = pole_sector_t(std::move(es), nda::array<ComplexType, 3>(c(nda::range(o, o + cnt(ik)), nda::range::all, nda::range::all)));
        }
        o += cnt(ik);
      }
    }
  }
  // optional filters of factorized poles (experiments): GW_LINE_DIAG_EMAX (drop |e| > emax), GW_LINE_DIAG_WMIN (drop weight <)
  if (std::getenv("GW_LINE_DIAG_EMAX") or std::getenv("GW_LINE_DIAG_WMIN")) {
    const double fe = std::getenv("GW_LINE_DIAG_EMAX") ? std::atof(std::getenv("GW_LINE_DIAG_EMAX")) : 1e300;
    const double fw = std::getenv("GW_LINE_DIAG_WMIN") ? std::atof(std::getenv("GW_LINE_DIAG_WMIN")) : 0.0;
    long nd = 0;
    for (long ik = 0; ik < nk; ++ik)
      for (auto *ps : {&pd.part[ik], &pd.hole[ik]}) {
        REQUIRE(ps->is_factorized());
        std::vector<long> keep;
        for (long m = 0; m < ps->size(); ++m)
          if (std::abs(ps->e(m)) <= fe and ps->weight(m) >= fw) keep.push_back(m);
        nd += ps->size() - long(keep.size());
        nda::array<double, 1> e(long(keep.size()));
        nda::array<ComplexType, 2> v(nb, long(keep.size()));
        for (long j = 0; j < long(keep.size()); ++j) {
          e(j) = ps->e(keep[j]);
          for (long i = 0; i < nb; ++i) v(i, j) = ps->v(i, keep[j]);
        }
        *ps = pole_sector_t::factorized_form(std::move(e), std::move(v));
      }
    app_log(1, "[diag] filter |e| <= {:.3e}, weight >= {:.1e}: {} poles dropped", fe, fw, nd);
  }
  double cmax = 0.0;
  for (long ik = 0; ik < nk; ++ik) {
    for (auto const *ps : {&pd.part[ik], &pd.hole[ik]})
      for (long m = 0; m < ps->size(); ++m) {
        double nrm = 0.0;
        for (long i = 0; i < nb; ++i)
          nrm = std::max(nrm, ps->is_factorized() ? std::norm(ps->v(i, m)) : std::abs(ps->coef(m, i, i)));
        cmax = std::max(cmax, nrm);
      }
  }
  // run parameters from the checkpoint (S6/S7b checkpoints: test settings)
  double th_deg = 20.0, eps_r = 1e-8, lam = 6.0, lam_b = 4.0, bgap = 0.02, sgap = 0.02, ntmin = 1e-3, ntmax = 60.0, wp = 0.11;
  double tolg = 1e-10, mixing = 0.5, mu_n = 0.0, eh_n = 0.0, el_n = 0.0;
  long npr = 120, Kc = 8, nphi = 8;
  nda::array<ComplexType, 3> H0, Fn;
  nda::array<ComplexType, 4> Sig_np, Sig_nh;
  bool have_sig = false;
  {
    h5::file f(fenv, 'r');
    h5::group g(f);
    auto ig = g.open_group("input");
    h5::h5_read(ig, "theta_deg", th_deg); h5::h5_read(ig, "eps", eps_r); h5::h5_read(ig, "lam", lam);
    h5::h5_read(ig, "lam_b", lam_b); h5::h5_read(ig, "bos_gap", bgap); h5::h5_read(ig, "sigma_gap", sgap);
    h5::h5_read(ig, "nodes_per_ray", npr); h5::h5_read(ig, "node_tmin", ntmin); h5::h5_read(ig, "node_tmax", ntmax);
    h5::h5_read(ig, "wp", wp); h5::h5_read(ig, "K", Kc); h5::h5_read(ig, "tol_gram", tolg); h5::h5_read(ig, "nphi", nphi);
    h5::h5_read(ig, "mixing", mixing);
    nda::h5_read(g, "system/H0", H0);
    auto it = g.open_group("scf_line/iter" + std::to_string(iter));
    h5::h5_read(it, "mu", mu_n); h5::h5_read(it, "e_homo", eh_n); h5::h5_read(it, "e_lumo", el_n);
    nda::h5_read(it, "F", Fn);
    have_sig = it.has_dataset("Sigma_p");
    if (have_sig) { nda::h5_read(it, "Sigma_p", Sig_np); nda::h5_read(it, "Sigma_h", Sig_nh); }
  }
  if (bgap < 0.0) bgap = 0.5 * (el_n - eh_n);
  if (std::getenv("GW_LINE_DIAG_LAMB")) lam_b = std::atof(std::getenv("GW_LINE_DIAG_LAMB"));   // experiment: bosonic range
  const double theta = th_deg * std::numbers::pi / 180.0, theta_t = 0.5 * theta;
  numerics::line_dlr::bosonic_basis_t bos(theta, lam_b, eps_r, bgap);
  auto fz = numerics::line_dlr::dense_nodes(theta, ntmin, ntmax, npr);
  auto pr = pole_ranges_t::from(pd);
  app_log(1, "[diag] run settings: eps {:.0e}, K {}, bosonic rank {} (gap {:.4f}), {} fermionic nodes, mixing {}", eps_r, Kc,
          bos.rank, bgap, fz.size(), mixing);
  app_log(1, "[diag] {} iter {}: poles e^> [{:.2e}, {:.3f}], |e^<| [{:.2e}, {:.3f}], max |coef_ii| {:.2e}, emin {:.2e}", fenv, iter,
          pr.p_min, pr.p_max, pr.h_min, pr.h_max, cmax, pd.emin());
  aux_grid_t grid(mpi, Np);
  dyson_layout_t lay(comm.size(), comm.rank(), nq, bos.zeta_nodes.size(), Np);
  utils::TimerManager T;
  coulomb_blocks_t<HOST_MEMORY> Zb(*L.thc, grid, lay.q_rng(), T);
  propagator_t<HOST_MEMORY> prop(*L.thc, grid);
  const double emin = pd.emin();
  time_ray_t rp(theta_t, 60.0 / (emin * std::sin(theta_t)), 1e-7, 6.0, 24, sector_t::particle);
  time_ray_t rh(theta_t, 60.0 / (emin * std::sin(theta_t)), 1e-7, 6.0, 24, sector_t::hole);
  auto gp = time_ray_t::for_spectrum(theta_t, emin, 36.0, 1e-5, 3.0, 16, sector_t::particle);
  auto gh = time_ray_t::for_spectrum(theta_t, emin, 36.0, 1e-5, 3.0, 16, sector_t::hole);
  auto rel = [&](auto const &A, auto const &B) {
    double d = 0.0, m = 0.0;
    for (long i = 0; i < A.size(); ++i) { d = std::max(d, std::abs(A.data()[i] - B.data()[i])); m = std::max(m, std::abs(B.data()[i])); }
    d = comm.all_reduce_value(d, mpi3::max<>{});
    m = comm.all_reduce_value(m, mpi3::max<>{});
    return std::array<double, 2>{d / m, m};
  };
  memory::array<HOST_MEMORY, ComplexType, 4> Pr_p, Pr_h, w;
  nda::array<ComplexType, 4> Sr_p, Sr_h;
  polarization<HOST_MEMORY>(prop, pd, mf, grid, bos.zeta_nodes, rp, rh, 8, Pr_p, T, sector_t::particle);
  polarization<HOST_MEMORY>(prop, pd, mf, grid, bos.zeta_nodes, rp, rh, 8, Pr_h, T, sector_t::hole);
  {
    memory::array<HOST_MEMORY, ComplexType, 4> Pi = Pr_p;
    Pi += Pr_h;
    screened_interaction<HOST_MEMORY>(Pi, Zb, bos, grid, mpi, w, T);
  }
  {
    // size of the W residues vs W at the bosonic nodes (cancellation in the bosonic real-pole fit)
    double wmax = 0.0, wsum = 0.0;
    for (long a = 0; a < w.size(); ++a) wmax = std::max(wmax, std::abs(w.data()[a]));
    const long r = bos.rank, nqw = w.extent(0);
    for (long iq = 0; iq < nqw; ++iq)
      for (long P = 0; P < w.extent(2); ++P)
        for (long Q = 0; Q < w.extent(3); ++Q) {
          double sm = 0.0;
          for (long j = 0; j < r; ++j) sm += std::abs(w(iq, j, P, Q));
          wsum = std::max(wsum, sm);
        }
    wmax = comm.all_reduce_value(wmax, mpi3::max<>{});
    wsum = comm.all_reduce_value(wsum, mpi3::max<>{});
    app_log(1, "[diag] W residues: max |w_j| {:.3e}, max_PQ sum_j |w_j| {:.3e} (w extents {} x {} x {} x {})", wmax, wsum,
            w.extent(0), w.extent(1), w.extent(2), w.extent(3));
  }
  self_energy<HOST_MEMORY>(prop, pd, w, bos, mf, grid, mpi, fz, rp, rh, 8, Sr_p, T, sector_t::particle);
  self_energy<HOST_MEMORY>(prop, pd, w, bos, mf, grid, mpi, fz, rp, rh, 8, Sr_h, T, sector_t::hole);
  app_log(1, "  reference GL rays {} nodes; default GL {} nodes", rp.size(), gp.size());
  std::vector<std::tuple<std::string, nda::array<ComplexType, 4>, nda::array<ComplexType, 4>>> sig_runs;
  sig_runs.emplace_back("ref", Sr_p, Sr_h);
  auto run = [&](std::string const &nm, numerics::line_dlr::time_nodes_t const &pp, numerics::line_dlr::time_nodes_t const &ph,
                 numerics::line_dlr::time_nodes_t const &sp, numerics::line_dlr::time_nodes_t const &sh) {
    memory::array<HOST_MEMORY, ComplexType, 4> Pp, Ph;
    nda::array<ComplexType, 4> Sp, Sh;
    polarization<HOST_MEMORY>(prop, pd, mf, grid, bos.zeta_nodes, pp, ph, 8, Pp, T, sector_t::particle);
    polarization<HOST_MEMORY>(prop, pd, mf, grid, bos.zeta_nodes, pp, ph, 8, Ph, T, sector_t::hole);
    self_energy<HOST_MEMORY>(prop, pd, w, bos, mf, grid, mpi, fz, sp, sh, 8, Sp, T, sector_t::particle);
    self_energy<HOST_MEMORY>(prop, pd, w, bos, mf, grid, mpi, fz, sp, sh, 8, Sh, T, sector_t::hole);
    auto a = rel(Pp, Pr_p), b = rel(Ph, Pr_h), c = rel(Sp, Sr_p), d = rel(Sh, Sr_h);
    app_log(1, "  {:<10s} nodes Pi {}+{} Sigma {}+{}: vs ref Pi^> {:.2e} (max {:.2e}) Pi^< {:.2e} | Sigma^> {:.2e} (max {:.2e}) "
               "Sigma^< {:.2e} (max {:.2e})",
            nm, pp.size(), ph.size(), sp.size(), sh.size(), a[0], a[1], b[0], c[0], c[1], d[0], d[1]);
    sig_runs.emplace_back(nm, Sp, Sh);
  };
  run("GL", gp, gh, gp, gh);
  for (double te : {1e-8, 1e-10, 1e-12})
    for (double pad : {1.25, 2.0}) {
      numerics::line_dlr::time_id_opts_t o;
      o.pad = pad;
      line_time_grids_t tg(pd, bos.nu, theta_t, te, o, bos.zeta_nodes, fz, comm);
      tg.log(1);
      char nm[32];
      std::snprintf(nm, sizeof(nm), "ID %.0e p%.2f", te, pad);
      run(nm, tg.pi_p, tg.pi_h, tg.sig_p, tg.sig_h);
      if (pad == 1.25 and te >= 1e-10) {
        // the full ID chain as the driver runs it: Pi_ID -> W -> Sigma_ID, vs the reference chain
        memory::array<HOST_MEMORY, ComplexType, 4> Pp, Ph, wc;
        nda::array<ComplexType, 4> Sp, Sh;
        polarization<HOST_MEMORY>(prop, pd, mf, grid, bos.zeta_nodes, tg.pi_p, tg.pi_h, 8, Pp, T, sector_t::particle);
        polarization<HOST_MEMORY>(prop, pd, mf, grid, bos.zeta_nodes, tg.pi_p, tg.pi_h, 8, Ph, T, sector_t::hole);
        Pp += Ph;
        screened_interaction<HOST_MEMORY>(Pp, Zb, bos, grid, mpi, wc, T);
        self_energy<HOST_MEMORY>(prop, pd, wc, bos, mf, grid, mpi, fz, tg.sig_p, tg.sig_h, 8, Sp, T, sector_t::particle);
        self_energy<HOST_MEMORY>(prop, pd, wc, bos, mf, grid, mpi, fz, tg.sig_p, tg.sig_h, 8, Sh, T, sector_t::hole);
        auto c = rel(Sp, Sr_p), d = rel(Sh, Sr_h);
        app_log(1, "  {:<10s} FULL CHAIN (Pi_ID -> W -> Sigma_ID) vs reference chain: Sigma^> {:.2e} Sigma^< {:.2e}", nm, c[0], d[0]);
        sig_runs.emplace_back(std::string(nm) + " chain", Sp, Sh);
      }
    }

  // closure response (S7c): the closure of the NEXT iteration (mixing with the stored Sigma of this iteration, H0 + F - mu)
  // on the Sigma of each time grid vs on the reference Sigma, and on the reference Sigma + random noise of relative size
  // 1e-10 / 1e-8: dmu, dgap and the Lehmann G on the imaginary axis (max |dG| / max |G|)
  if (std::getenv("GW_LINE_DIAG_CLOSURE") == nullptr) return;
  auto all = nda::range::all;
  nda::array<ComplexType, 3> Hrel(nk, nb, nb);
  for (long ik = 0; ik < nk; ++ik)
    for (long i = 0; i < nb; ++i)
      for (long j = 0; j < nb; ++j) Hrel(ik, i, j) = H0(ik, i, j) + Fn(ik, i, j) - (i == j ? mu_n : 0.0);
  double sg_p = sgap, sg_h = sgap;
  if (sgap < 0.0) { sg_p = 0.8 * (el_n + bos.gap); sg_h = 0.8 * (std::abs(eh_n) + bos.gap); }
  line_basis_t bp(theta, lam, eps_r, lam, sg_p, -1.0, ntmax), bh(theta, lam, eps_r, sg_h, lam, -1.0, ntmax);
  line_basis_t gb_p(theta, lam, eps_r, lam, 0.0, -1.0, ntmax), gb_h(theta, lam, eps_r, 0.0, lam, -1.0, ntmax);
  closure_params_t cp{wp, Kc, tolg, nphi};
  g_repr_params_t gr;
  gr.repr = "lehmann";
  const double nelec = double(mf.nelec()), meV = 27.211386e3;
  auto mixed = [&](nda::array<ComplexType, 4> const &Sn, nda::array<ComplexType, 4> const &So) {
    if (not have_sig) return nda::array<ComplexType, 4>(Sn);
    nda::array<ComplexType, 4> M(Sn.shape());
    for (long a = 0; a < M.size(); ++a) M.data()[a] = mixing * Sn.data()[a] + (1.0 - mixing) * So.data()[a];
    return M;
  };
  auto do_closure = [&](nda::array<ComplexType, 4> const &Sp, nda::array<ComplexType, 4> const &Sh) {
    utils::TimerManager Tc;
    return closure(comm, Hrel, mixed(Sp, Sig_np), mixed(Sh, Sig_nh), fz, bp, bh, gb_p, gb_h, cp, nelec, Tc, gr);
  };
  auto G_axis = [&](closure_out_t const &o) {
    nda::array<ComplexType, 4> G(nk, 40, nb, nb);
    G() = 0.0;
    for (long ik = 0; ik < nk; ++ik)
      for (long iw = 0; iw < 40; ++iw) {
        const ComplexType z(0.0, 1e-3 * std::pow(10.0, 4.0 * iw / 39.0));
        for (long m = 0; m < o.leh.e[ik].size(); ++m) {
          const ComplexType f = 1.0 / (z - o.leh.e[ik](m));
          for (long i = 0; i < nb; ++i)
            for (long j = 0; j < nb; ++j) G(ik, iw, i, j) += o.leh.v[ik](i, m) * std::conj(o.leh.v[ik](j, m)) * f;
        }
      }
    return G;
  };
  auto o_ref = do_closure(Sr_p, Sr_h);
  auto G_ref = G_axis(o_ref);
  app_log(1, "  closure response (next iteration, {} on every rank): reference mu {:.8f}, gap {:.6f} eV, upfolded poles {}-{}, "
             "held-out {:.1e}",
          "lehmann", mu_n + o_ref.dmu, (o_ref.e_lumo - o_ref.e_homo) * 27.211386,
          *std::min_element(o_ref.npoles.begin(), o_ref.npoles.end()), *std::max_element(o_ref.npoles.begin(), o_ref.npoles.end()),
          *std::max_element(o_ref.heldout.begin(), o_ref.heldout.end()));
  auto report = [&](std::string const &nm, nda::array<ComplexType, 4> const &Sp, nda::array<ComplexType, 4> const &Sh) {
    auto o  = do_closure(Sp, Sh);
    auto Gx = G_axis(o);
    nda::array<ComplexType, 4> St = Sp + Sh, Sr = Sr_p + Sr_h;
    long dnp = 0;
    for (long ik = 0; ik < nk; ++ik) dnp = std::max(dnp, std::abs(o.npoles[ik] - o_ref.npoles[ik]));
    app_log(1, "    {:<14s} dSigma_new {:.2e} -> dmu {:+.4f} meV, dgap {:+.4f} meV, Lehmann G(i w) {:.2e}, max |d upfold rank| {}", nm,
            nda::max_element(nda::abs(St - Sr)) / nda::max_element(nda::abs(Sr)), (o.dmu - o_ref.dmu) * meV,
            ((o.e_lumo - o.e_homo) - (o_ref.e_lumo - o_ref.e_homo)) * meV,
            nda::max_element(nda::abs(Gx - G_ref)) / nda::max_element(nda::abs(G_ref)), dnp);
  };
  for (auto const &[nm, Sp, Sh] : sig_runs)
    if (nm != "ref") report(nm, Sp, Sh);
  std::mt19937 gen(7);
  std::normal_distribution<double> N01;
  const double smax = nda::max_element(nda::abs(Sr_p + Sr_h));
  for (double amp : {1e-12, 1e-10, 1e-8}) {
    nda::array<ComplexType, 4> Np = Sr_p, Nh = Sr_h;
    for (long a = 0; a < Np.size(); ++a) {
      Np.data()[a] += amp * smax * ComplexType(N01(gen), N01(gen));
      Nh.data()[a] += amp * smax * ComplexType(N01(gen), N01(gen));
    }
    char nm[32];
    std::snprintf(nm, sizeof(nm), "noise %.0e", amp);
    report(nm, Np, Nh);
  }
  (void)all;
}

// ======================================================================================================================
// S7c: G representation study (lehmann vs compressed, gl vs id)
// ======================================================================================================================
namespace {

/// production settings of runs/lih222_gw_line/lih222_gw_line.toml (eps 1e-10, K 24), conv_thr off
ptree prod_params(std::string const &output, long niter, std::string const &time_grid, std::string const &g_repr) {
  auto pt = scf_params(output, niter, false, time_grid, g_repr);
  pt.put("eps", 1e-10);
  pt.put("K", 24);
  return pt;
}

struct variant_t {
  std::string name, repr;
  double emin_frac = 0.0, wsmall = 0.0;
};

struct study_run_t {
  variant_t var;
  std::string tg, file;
  methods::gw_line::gw_line_result_t R;
  double time = 0.0;
};

/**
 * Runs every variant with time_grid "gl" and "id" for niter iterations (settings from mk), prints the per-iteration table
 * and the id-vs-gl differences per variant; returns the runs (checkpoints removed unless GW_LINE_TEST_KEEP is set).
 * idgl[v][it] = {|dmu| meV, |dgap| meV, max|dSigma|/max|Sigma|}.
 */
std::vector<study_run_t> repr_study(lih_t &L, std::string const &tag, std::vector<variant_t> const &vars, long niter,
                                    std::function<ptree(std::string const &, std::string const &, std::string const &)> const &mk,
                                    std::vector<std::vector<std::array<double, 3>>> &idgl) {
  auto &comm       = L.mpi->comm;
  const double meV = 27.211386e3;
  std::vector<study_run_t> runs;
  for (auto const &v : vars)
    for (std::string tg : {"gl", "id"}) {
      study_run_t r;
      r.var  = v;
      r.tg   = tg;
      r.file = "gw_line_repr_" + tag + "_" + v.name + "_" + tg;
      auto pt = mk(r.file, tg, v.repr);
      pt.put("g_emin_frac", v.emin_frac);
      pt.put("g_wsmall", v.wsmall);
      if (std::getenv("GW_LINE_LAMB")) pt.put("lam_b", std::atof(std::getenv("GW_LINE_LAMB")));
      auto t0 = std::chrono::steady_clock::now();
      r.R     = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pt);
      r.time  = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      REQUIRE(long(r.R.history.size()) == niter);
      runs.push_back(std::move(r));
    }
  app_log(1, "\n[repr {}] {}, ranks {}, {} iterations per run, lam_b {}", tag, L.fixture, comm.size(), niter,
          runs.empty() ? 0.0 : mk("x", "gl", "lehmann").get<double>("lam_b"));
  if (std::getenv("GW_LINE_LAMB")) app_log(1, "  (lam_b overridden by GW_LINE_LAMB = {})", std::getenv("GW_LINE_LAMB"));
  app_log(1, "  run          it |   mu (Ha)     gap (eV)  N(mu)     Tr D      dSigma   | G poles/k&s  min|e| (Ha) | t-nodes Pi  "
             "Sigma   | pruned near-mu (w)   | dropped | time (s)");
  for (auto const &r : runs)
    for (long it = 0; it < niter; ++it) {
      auto const &h = r.R.history[it];
      app_log(1, "  {:<5s} {:<3s} {:4d} | {:.7f}  {:.5f}  {:.6f}  {:.6f}  {:.2e} | {:4d}-{:<4d}   {:.2e}    | {:4d}+{:<4d} {:4d}+{:<4d} | "
                 "{:4d} ({:.1e})        | {:.1e} | {:.1f}",
              r.var.name, r.tg, it + 1, h.mu, h.gap * 27.211386, h.N_mu, h.nelec, h.dSigma, h.ng_min, h.ng_max, h.g_emin, h.nt_pi_p,
              h.nt_pi_h, h.nt_sig_p, h.nt_sig_h, h.pruned_near, h.pruned_near_weight, h.dropped, h.time);
    }
  app_log(1, "  id vs gl per representation:   run    it |  dmu (meV)   dgap (meV) | max|dSigma|/max|Sigma|");
  idgl.assign(vars.size(), {});
  for (long iv = 0; iv < long(vars.size()); ++iv) {
    auto const &G = runs[2 * iv], &I = runs[2 * iv + 1];
    for (long it = 0; it < niter; ++it) {
      auto Sg = read_sigma_total(comm, G.file + ".gw_line.h5", it + 1);
      auto Si = read_sigma_total(comm, I.file + ".gw_line.h5", it + 1);
      const double ds   = nda::max_element(nda::abs(Si - Sg)) / nda::max_element(nda::abs(Sg));
      const double dmu  = (I.R.history[it].mu - G.R.history[it].mu) * meV;
      const double dgap = (I.R.history[it].gap - G.R.history[it].gap) * meV;
      idgl[iv].push_back({std::abs(dmu), std::abs(dgap), ds});
      app_log(1, "                                 {:<5s} {:3d} | {:+10.4f}  {:+10.4f} | {:.2e}", vars[iv].name, it + 1, dmu, dgap, ds);
    }
  }
  if (vars.size() > 1) {
    app_log(1, "  representation vs {} (same time grid):  run     it |  dmu (meV)   dgap (meV) | max|dSigma|/max|Sigma|", vars[0].name);
    for (long iv = 1; iv < long(vars.size()); ++iv)
      for (long ig = 0; ig < 2; ++ig) {
        auto const &A = runs[ig], &B = runs[2 * iv + ig];
        for (long it = 0; it < niter; ++it) {
          auto Sa = read_sigma_total(comm, A.file + ".gw_line.h5", it + 1);
          auto Sb = read_sigma_total(comm, B.file + ".gw_line.h5", it + 1);
          app_log(1, "                                       {:<5s} {:<3s} {:3d} | {:+10.4f}  {:+10.4f} | {:.2e}", vars[iv].name,
                  B.tg, it + 1, (B.R.history[it].mu - A.R.history[it].mu) * meV, (B.R.history[it].gap - A.R.history[it].gap) * meV,
                  nda::max_element(nda::abs(Sb - Sa)) / nda::max_element(nda::abs(Sa)));
        }
      }
  }
  for (auto const &r : runs) app_log(1, "  wall {:<5s} {:<3s}: {:.1f} s", r.var.name, r.tg, r.time);
  for (auto const &r : runs) remove_file(comm, r.file + ".gw_line.h5");
  return runs;
}

} // namespace

/**
 * (S7c) test settings (eps 1e-8, K 8) with the automatic bosonic range (lam_b = 2 x 6 Ha), 2 iterations (S7f; was 4):
 * lehmann vs compressed, gl vs id, gated by idgl_gate (S7f). S7c measurements (2 ranks, lam_b 12): iteration 1 (KS poles,
 * identical for both representations)
 * id vs gl dSigma 2.1e-9 (= the ID error); iteration 2: dSigma 7.8e-5, |dmu|, |dgap| <= 0.07 meV for BOTH representations:
 * not the kernels (fixed-pole ID error ~ 10 eps_t once lam_b covers Pi's spectrum, [.time_id_poles]) but the closure's
 * response to the 2e-9 difference of iteration 1 (K = 8: x 1e4); iterations 3-4: closure noise floor of K = 8, 1-2 meV.
 * lehmann vs compressed on the same grid: iteration 2 6.5e-9 (the gapless refit reproduces the Lehmann G), then the same
 * closure noise.
 */
TEST_CASE("gw_line_scf_lehmann", "[gw_line][scf][lehmann]") {
  lih_t L;
  const long niter = 2;   // S7f: see idgl_gate
  std::vector<variant_t> vars = {{"cmp", "compressed"}, {"leh", "lehmann", env_or("GW_LINE_EMIN_FRAC", 0.5), env_or("GW_LINE_WSMALL", 1e-4)}};
  std::vector<std::vector<std::array<double, 3>>> idgl;
  auto runs = repr_study(L, "test", vars, niter,
                         [&](std::string const &f, std::string const &tg, std::string const &gr) {
                           auto pt = scf_params(f, niter, false, tg, gr);
                           pt.put("lam_b", -1.0);   // auto: 2 x 6 Ha
                           pt.put("time_eps", 1e-10);   // S7f: the it-1 ID difference 3e-11 keeps the it-2 gate tight
                           return pt;
                         },
                         idgl);
  for (auto const &r : runs) {
    REQUIRE(r.R.poles.is_factorized() == (r.var.repr == "lehmann"));
    for (auto const &h : r.R.history) {
      REQUIRE(h.g_repr == r.var.repr);
      if (r.var.repr == "lehmann" and h.iter > 1) REQUIRE(h.g_emin > 0.25 * 0.5 * h.gap);   // no in-gap poles left
    }
  }
  // S7f: gates from the measured noise floor of each representation (see idgl_gate)
  bool ok = true;
  for (long iv = 0; iv < 2; ++iv)
    for (long it = 0; it < niter; ++it) {
      auto const &d = idgl[iv][it];
      auto g        = idgl_gate(floor_3e11, it, 1e-10);
      app_log(1, "  [lehmann] id vs gl, {} iteration {}:", vars[iv].name, it + 1);
      ok = gate_line("|dmu| (meV)", d[0], g[0]) and ok;
      ok = gate_line("|dgap| (meV)", d[1], g[1]) and ok;
      ok = gate_line("max|dSigma|/max|Sigma|", d[2], g[2]) and ok;
    }
  REQUIRE(ok);
}

/**
 * W pairing fix (2026-10-04, notes section 3.3): the driver on a q != -q mesh (qe_lih223, 2x2x3: 8 of 12 q not
 * self-inverse; THC nIpts 8 nbnd built here), the settings of gw_line_scf_lehmann, 2 iterations, compressed and lehmann, gl
 * and id. Sanity per iteration: held-out moment error (Si 4x4x4 before the fix: 0.13 at iteration 1), N(mu) = nelec,
 * a QP gap near the KS gap, lehmann: a QP pole in every k and sector (largest pole weight >= 0.5; before the fix the 4x4x4
 * run had only combs of weight 0.05-0.2), and id vs gl / lehmann vs compressed agreeing as on lih222
 * (idgl_gate of the lih222 noise floor).
 */
TEST_CASE("gw_line_scf_qpair_lih223", "[gw_line][scf][qpair]") {
  lih_t L("qe_lih223");
  auto &comm = L.mpi->comm;
  const long niter = 2;
  std::vector<variant_t> vars = {{"cmp", "compressed"}, {"leh", "lehmann", env_or("GW_LINE_EMIN_FRAC", 0.5), env_or("GW_LINE_WSMALL", 1e-4)}};
  std::vector<std::vector<std::array<double, 3>>> idgl;
  auto runs = repr_study(L, "qpair", vars, niter,
                         [&](std::string const &f, std::string const &tg, std::string const &gr) {
                           auto pt = scf_params(f, niter, false, tg, gr);
                           pt.put("lam_b", -1.0);
                           pt.put("time_eps", 1e-10);
                           return pt;
                         },
                         idgl);
  // KS gap of the fixture
  const long nk = L.mf->nkpts(), nb = L.mf->nbnd(), nocc = long(std::llround(double(L.mf->nelec()) / 2.0));
  double homo = -1e300, lumo = 1e300;
  for (long ik = 0; ik < nk; ++ik)
    for (long n = 0; n < nb; ++n) {
      if (n < nocc) homo = std::max(homo, L.mf->eigval()(0, ik, n));
      else lumo = std::min(lumo, L.mf->eigval()(0, ik, n));
    }
  const double ks_gap = lumo - homo, nel = double(L.mf->nelec());
  bool ok = true;
  for (auto const &r : runs) {
    for (auto const &h : r.R.history) {
      app_log(1, "  [qpair] {} {} it {}: held-out {:.2e}, N(mu) {:.6f} (nelec {}), gap {:.4f} eV (KS {:.4f} eV), poles {}-{}",
              r.var.name, r.tg, h.iter, h.heldout_max, h.N_mu, nel, h.gap * 27.211386, ks_gap * 27.211386, h.npoles_min,
              h.npoles_max);
      ok = gate_line("held-out moment error", h.heldout_max, 1e-4) and ok;
      ok = gate_line("|N(mu) / nelec - 1|", std::abs(h.N_mu / nel - 1.0), 2e-2) and ok;
      ok = gate_line("|gap / KS gap - 1|", std::abs(h.gap / ks_gap - 1.0), 1.0) and ok;
    }
    if (r.var.repr == "lehmann") {
      double wmin = 1e300;
      for (long ik = 0; ik < nk; ++ik)
        for (auto s : {sector_t::particle, sector_t::hole}) {
          auto const &ps = r.R.poles(ik, s);
          REQUIRE(ps.size() > 0);
          double wmax = 0.0;
          for (long m = 0; m < ps.size(); ++m) wmax = std::max(wmax, ps.weight(m));
          wmin = std::min(wmin, wmax);
        }
      app_log(1, "  [qpair] {} {}: min over k and sectors of the largest pole weight (QP) {:.3f}", r.var.name, r.tg, wmin);
      ok = gate_line("1 - QP weight", 1.0 - wmin, 0.5) and ok;
    }
  }
  for (long iv = 0; iv < 2; ++iv)
    for (long it = 0; it < niter; ++it) {
      auto const &d = idgl[iv][it];
      auto g        = idgl_gate(floor_3e11, it, 1e-10);
      app_log(1, "  [qpair] id vs gl, {} iteration {}:", vars[iv].name, it + 1);
      ok = gate_line("|dmu| (meV)", d[0], g[0]) and ok;
      ok = gate_line("|dgap| (meV)", d[1], g[1]) and ok;
      ok = gate_line("max|dSigma|/max|Sigma|", d[2], g[2]) and ok;
    }
  (void)comm;
  REQUIRE(ok);
}

/// (S7c, hidden) production settings (eps 1e-10, K 24), GW_LINE_STUDY_NITER iterations (default 4): compressed, lehmann
/// with the near-mu rule (GW_LINE_EMIN_FRAC / GW_LINE_WSMALL, default 0.5 / 1e-4) and lehmann without it.
TEST_CASE("gw_line_scf_lehmann_prod", "[.lehmann_prod]") {
  lih_t L;
  const long niter = long(env_or("GW_LINE_STUDY_NITER", 4));
  std::vector<variant_t> vars = {{"cmp", "compressed"},
                                 {"leh", "lehmann", env_or("GW_LINE_EMIN_FRAC", 0.5), env_or("GW_LINE_WSMALL", 1e-4)},
                                 {"leh0", "lehmann", 0.0, 0.0}};
  std::vector<std::vector<std::array<double, 3>>> idgl;
  repr_study(L, "prod", vars, niter,
             [&](std::string const &f, std::string const &tg, std::string const &gr) { return prod_params(f, niter, tg, gr); }, idgl);
}

// S7e: k-distributed Sigma (reduce-scatter, sigma_kdist = true, the default) + the separate last-iteration Sigma file
// (checkpoint_sigma = "last") vs the pre-S7e replicated Sigma written into every iteration (false / "all"): 2 iterations,
// mu, gap, dSigma per iteration and the owned rows of Sigma; plus a restart from the "last" layout.
TEST_CASE("gw_line_scf_kdist", "[gw_line][scf][s7e]") {
  lih_t L;
  auto &comm = L.mpi->comm;
  auto pa    = scf_params("gw_line_kdA", 2, false, "id", "lehmann");
  auto pb    = scf_params("gw_line_kdB", 2, false, "id", "lehmann");
  pa.put("checkpoint_sigma", "last");   // the S7e defaults (sigma_kdist = true)
  pb.put("sigma_kdist", false);
  pb.put("checkpoint_sigma", "all");
  auto A = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pa);
  auto B = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pb);
  const methods::gw_line::k_dist_t kd(L.mf->nkpts(), comm);
  REQUIRE(A.Sig_p.extent(0) == kd.nloc());
  REQUIRE(B.Sig_p.extent(0) == L.mf->nkpts());
  double dS = 0.0, mS = nda::max_element(nda::abs(B.Sig_p));
  for (long l = 0; l < kd.nloc(); ++l) {
    const long k = kd.global(l, kd.rank);
    dS = std::max(dS, double(nda::max_element(nda::abs(A.Sig_p(l, nda::ellipsis{}) - B.Sig_p(k, nda::ellipsis{})))));
    dS = std::max(dS, double(nda::max_element(nda::abs(A.Sig_h(l, nda::ellipsis{}) - B.Sig_h(k, nda::ellipsis{})))));
  }
  dS = comm.all_reduce_value(dS, mpi3::max<>{}) / mS;
  double dmu = 0.0, dgap = 0.0, ddS = 0.0;
  for (long i = 0; i < 2; ++i) {
    dmu  = std::max(dmu, std::abs(A.history[i].mu - B.history[i].mu));
    dgap = std::max(dgap, std::abs(A.history[i].gap - B.history[i].gap));
    if (i > 0) ddS = std::max(ddS, std::abs(A.history[i].dSigma - B.history[i].dSigma) / B.history[i].dSigma);
  }
  app_log(1, "[s7e] k-distributed vs replicated Sigma, {} ranks, 2 iterations: |dmu| {:.1e} Ha, |dgap| {:.1e} Ha, dSigma rel {:.1e}, "
             "Sigma (owned rows) {:.1e}",
          comm.size(), dmu, dgap, ddS, dS);
  REQUIRE(dmu <= 1e-12);
  REQUIRE(dgap <= 1e-12);
  REQUIRE(ddS <= 1e-10);
  REQUIRE(dS <= 1e-13);
  // q groups of the Pi -> W stage (COQUI_GWLINE_QGROUP = 3: groups 3, 3, 2) vs all q at once
  {
    setenv("COQUI_GWLINE_QGROUP", "3", 1);
    auto Q = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, scf_params("gw_line_kdQ", 2, false, "id", "lehmann"));
    unsetenv("COQUI_GWLINE_QGROUP");
    double dq = 0.0;
    for (long i = 0; i < 2; ++i) dq = std::max(dq, std::abs(Q.history[i].mu - A.history[i].mu));
    const double dSq = comm.all_reduce_value(double(nda::max_element(nda::abs(Q.Sig_p - A.Sig_p))), mpi3::max<>{}) / mS;
    app_log(1, "[s7e] q groups of 3 vs all q: |dmu| {:.1e}, Sigma_p {:.1e}", dq, dSq);
    REQUIRE(dq <= 1e-12);
    REQUIRE(dSq <= 1e-12);
    // host-resident residues streamed by Sigma in q groups of 3 (the device fallback, here on the host). The q sum of
    // Sigma is then split over groups (another summation order): compared after ONE iteration (from the second on, the
    // closure amplifies 1e-16 differences to ~1e-9, S7c), the kernel-level comparison is in [s7e] of the kernels suite
    auto A1 = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, scf_params("gw_line_kdQ", 1, false, "id", "lehmann"));
    setenv("COQUI_GWLINE_QGROUP", "3", 1);
    setenv("COQUI_GWLINE_W_HOST", "1", 1);
    setenv("COQUI_GWLINE_SIGMA_QGROUP", "3", 1);
    auto H = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, scf_params("gw_line_kdQ", 1, false, "id", "lehmann"));
    unsetenv("COQUI_GWLINE_QGROUP");
    unsetenv("COQUI_GWLINE_W_HOST");
    unsetenv("COQUI_GWLINE_SIGMA_QGROUP");
    const double dh  = std::abs(H.history[0].mu - A1.history[0].mu);
    const double mS1 = comm.all_reduce_value(double(nda::max_element(nda::abs(A1.Sig_p))), mpi3::max<>{});
    const double dSh = comm.all_reduce_value(double(nda::max_element(nda::abs(H.Sig_p - A1.Sig_p))), mpi3::max<>{}) / mS1;
    app_log(1, "[s7e] host-resident residues + Sigma q groups of 3 vs all q resident, 1 iteration: |dmu| {:.1e}, Sigma_p {:.1e}", dh,
            dSh);
    REQUIRE(dh <= 1e-12);
    REQUIRE(dSh <= 1e-13);
    if (comm.root()) {
      std::filesystem::remove("gw_line_kdQ.gw_line.h5");
      std::filesystem::remove("gw_line_kdQ.gw_line.sigma.h5");
    }
  }
  // restart from the "last" layout (Sigma in gw_line_kdA.gw_line.sigma.h5) continues bitwise like 3 straight iterations
  auto pc = scf_params("gw_line_kdC", 3, false, "id", "lehmann");
  pc.put("checkpoint_sigma", "last");
  auto C  = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pc);
  auto pr = scf_params("gw_line_kdA", 3, true, "id", "lehmann");
  pr.put("checkpoint_sigma", "last");
  auto R  = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pr);
  REQUIRE(R.history.size() == 3);
  const double dmu3 = std::abs(R.mu - C.mu);
  const double dS3  = comm.all_reduce_value(double(nda::max_element(nda::abs(R.Sig_p - C.Sig_p))), mpi3::max<>{});
  app_log(1, "[s7e] restart from checkpoint_sigma = \"last\": |dmu| {:.1e}, Sigma_p {:.1e} after 2 + restart + 1 vs 3", dmu3, dS3);
  REQUIRE(dmu3 == 0.0);
  REQUIRE(dS3 == 0.0);
  if (comm.root()) {
    REQUIRE(std::filesystem::exists("gw_line_kdA.gw_line.sigma.h5"));
    for (auto f : {"gw_line_kdA.gw_line.h5", "gw_line_kdA.gw_line.sigma.h5", "gw_line_kdB.gw_line.h5", "gw_line_kdC.gw_line.h5",
                   "gw_line_kdC.gw_line.sigma.h5"})
      std::filesystem::remove(f);
  }
  comm.barrier();
}

// ======================================================================================================================
// S7f: closure noise floor and sensitivity
// ======================================================================================================================
namespace {

std::vector<std::string> env_list(char const *nm, std::string const &dflt) {
  std::string s = std::getenv(nm) ? std::getenv(nm) : dflt;
  std::vector<std::string> out;
  size_t a = 0;
  while (a <= s.size()) {
    size_t b = s.find(',', a);
    if (b == std::string::npos) b = s.size();
    if (b > a) out.push_back(s.substr(a, b - a));
    a = b + 1;
  }
  return out;
}

/// closure options from the environment (GW_LINE_CUT, GW_LINE_SVDCUT, GW_LINE_CUTWIN, GW_LINE_PHASE, GW_LINE_TOLSVD,
/// GW_LINE_TOLGRAM, GW_LINE_TSNAP = time_snap, GW_LINE_TOLGEPS = tol_gram_eps) into a driver ptree: lets the noise meter run the S7f remedies through the whole SCF
void closure_opts_from_env(ptree &pt) {
  if (auto *c = std::getenv("GW_LINE_CUT")) pt.put("closure_cut", std::string(c));
  if (auto *c = std::getenv("GW_LINE_SVDCUT")) pt.put("closure_svd_cut", std::string(c));
  if (auto *c = std::getenv("GW_LINE_CUTWIN")) pt.put("closure_cut_window", std::atof(c));
  if (auto *c = std::getenv("GW_LINE_PHASE")) pt.put("phase_keep", std::atof(c));
  if (auto *c = std::getenv("GW_LINE_TOLSVD")) pt.put("tol_svd", std::atof(c));
  if (auto *c = std::getenv("GW_LINE_TOLGRAM")) pt.put("tol_gram", std::atof(c));
  if (auto *c = std::getenv("GW_LINE_TSNAP")) pt.put("time_snap", std::atof(c));
  if (auto *c = std::getenv("GW_LINE_TOLGEPS")) pt.put("tol_gram_eps", std::atof(c));
}

/// Lehmann G(i w) of a closure output about the OLD centre (the energies shifted back by dmu), [nk, nw, nb, nb]:
/// the upfolding's response without the mu choice. pruned = the representation handed to the next iteration.
nda::array<ComplexType, 4> closure_G_axis(closure_out_t const &o, long nb, bool pruned, long nw = 40) {
  const long nk = long(o.leh.e.size());
  nda::array<ComplexType, 4> G(nk, nw, nb, nb);
  G() = 0.0;
  for (long ik = 0; ik < nk; ++ik)
    for (long iw = 0; iw < nw; ++iw) {
      const ComplexType z(0.0, 1e-3 * std::pow(10.0, 4.0 * iw / double(nw - 1)));
      auto add = [&](double e, auto const &res) {
        const ComplexType f = 1.0 / (z - (e + o.dmu));
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j) G(ik, iw, i, j) += res(i, j) * f;
      };
      if (not pruned) {
        for (long m = 0; m < o.leh.e[ik].size(); ++m)
          add(o.leh.e[ik](m), [&](long i, long j) { return o.leh.v[ik](i, m) * std::conj(o.leh.v[ik](j, m)); });
      } else {
        for (auto const *ps : {&o.poles.part[ik], &o.poles.hole[ik]})
          for (long m = 0; m < ps->size(); ++m) {
            if (ps->is_factorized()) add(ps->e(m), [&](long i, long j) { return ps->v(i, m) * std::conj(ps->v(j, m)); });
            else add(ps->e(m), [&](long i, long j) { return ps->coef(m, i, j); });
          }
      }
    }
  return G;
}

} // namespace

/**
 * (S7f, hidden) NOISE-FLOOR METER: the lih222 SCF run N times (GW_LINE_NOISE_N, default 4) from H0 perturbed by relative
 * Hermitian noise GW_LINE_NOISE_AMP (default 1e-13; run 0 unperturbed, run r seed r; GW_LINE_NOISE_MODE = sigma: the noise
 * is put on Sigma at the nodes of iteration 1 instead, relative to max|Sigma|), GW_LINE_NOISE_NITER iterations
 * (default 6); per iteration the spread (max - min over the runs) of mu and of the QP gap and max_r max|Sigma_r - Sigma_0| /
 * max|Sigma_0| at all k and nodes. Settings (comma lists): GW_LINE_NOISE_SETTINGS = test (K 8, eps 1e-8), prod (K 24,
 * eps 1e-10); GW_LINE_NOISE_LAMB = 4, auto; GW_LINE_NOISE_REPR = cmp, leh; GW_LINE_NOISE_TG = gl, id (default: all).
 * Closure options from GW_LINE_CUT / SVDCUT / CUTWIN / PHASE / TOLSVD / TOLGRAM (closure_opts_from_env).
 * This is the intrinsic reproducibility of the method at each setting and the yardstick of the [parity], [id_vs_gl] and
 * [lehmann] gates; greppable lines "[scf_noise] <setting> it <i> ...".
 */
TEST_CASE("gw_line_scf_noise", "[.scf_noise]") {
  lih_t L;
  auto &comm       = L.mpi->comm;
  const long N     = long(env_or("GW_LINE_NOISE_N", 4));
  const long niter = long(env_or("GW_LINE_NOISE_NITER", 6));
  const double amp = env_or("GW_LINE_NOISE_AMP", 1e-13);
  const double meV = 27.211386e3;
  // GW_LINE_NOISE_MODE = h0 (default) | sigma: the noise on Sigma at the nodes of iteration 1 (debug_noise_sigma), the
  // entry point of a kernel-level difference (e.g. the time-grid error of [id_vs_gl])
  const bool sig_noise = std::getenv("GW_LINE_NOISE_MODE") and std::string(std::getenv("GW_LINE_NOISE_MODE")) == "sigma";
  for (auto const &set : env_list("GW_LINE_NOISE_SETTINGS", "test,prod"))
    for (auto const &lb : env_list("GW_LINE_NOISE_LAMB", "4,auto"))
      for (auto const &rp : env_list("GW_LINE_NOISE_REPR", "cmp,leh"))
        for (auto const &tg : env_list("GW_LINE_NOISE_TG", "gl,id")) {
          const std::string cfg = set + "/lamb" + lb + "/" + rp + "/" + tg;
          std::vector<std::vector<double>> mu(N), gap(N);
          std::vector<nda::array<ComplexType, 4>> S0(niter);
          std::vector<double> dS(niter, 0.0);
          double wall = 0.0;
          for (long r = 0; r < N; ++r) {
            const std::string f = "gw_line_noise_" + std::to_string(r);
            auto pt = (set == "prod") ? prod_params(f, niter, tg, rp == "cmp" ? "compressed" : "lehmann")
                                      : scf_params(f, niter, false, tg, rp == "cmp" ? "compressed" : "lehmann");
            pt.put("lam_b", lb == "auto" ? -1.0 : std::atof(lb.c_str()));
            pt.put(sig_noise ? "debug_noise_sigma" : "debug_noise_h0", r == 0 ? 0.0 : amp);
            pt.put("debug_noise_seed", r);
            closure_opts_from_env(pt);
            auto t0 = std::chrono::steady_clock::now();
            auto R  = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pt);
            wall += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            REQUIRE(long(R.history.size()) == niter);
            for (long it = 0; it < niter; ++it) {
              mu[r].push_back(R.history[it].mu);
              gap[r].push_back(R.history[it].gap);
              auto S = read_sigma_total(comm, f + ".gw_line.h5", it + 1);
              if (r == 0) S0[it] = S;
              else dS[it] = std::max(dS[it], double(nda::max_element(nda::abs(S - S0[it])) / nda::max_element(nda::abs(S0[it]))));
            }
            remove_file(comm, f + ".gw_line.h5");
          }
          app_log(1, "\n[scf_noise] {} : {} runs (noise {:.0e} on {}), {} iterations, ranks {}, wall {:.0f} s", cfg, N, amp,
                  sig_noise ? "Sigma of iteration 1" : "H0", niter, comm.size(), wall);
          for (long it = 0; it < niter; ++it) {
            double mn = 1e300, mx = -1e300, gn = 1e300, gx = -1e300;
            for (long r = 0; r < N; ++r) {
              mn = std::min(mn, mu[r][it]); mx = std::max(mx, mu[r][it]);
              gn = std::min(gn, gap[r][it]); gx = std::max(gx, gap[r][it]);
            }
            app_log(1, "[scf_noise] {} it {} | mu {:.8f} gap {:.6f} eV | spread mu {:.2e} meV gap {:.2e} meV | max dSigma {:.2e}", cfg,
                    it + 1, mu[0][it], gap[0][it] * 27.211386, (mx - mn) * meV, (gx - gn) * meV, dS[it]);
          }
        }
}

/**
 * (S7f, hidden) CLOSURE SENSITIVITY at a fixed Sigma: the lih222 SCF (GW_LINE_CN_SET = test | prod, GW_LINE_CN_REPR =
 * leh | cmp, lam_b auto, time_grid id) is run for GW_LINE_CN_ITER iterations (default 2); the closure of the last iteration
 * (its mixed Sigma, H0 + F - mu of the iteration before) is repeated with the Cayley moments perturbed by relative noise
 * GW_LINE_CN_AMP (default 1e-12, GW_LINE_CN_SEEDS seeds, default 6). Response: Lehmann G(i w) about the old centre (max
 * over k, w of |dG| / max|G|; full Lehmann and the pruned representation), dmu, d(e_homo), d(e_lumo); amplification =
 * response / noise. Attribution: the decisions that flipped (Gram rank, r1, coarse phase basin, QP edges) and the response
 * with the hard decisions frozen to the unperturbed ones (all / all but one). Remedies: gap / smooth Gram cut, gap SVD
 * cut, phase continuity (phi_prev = the unperturbed phi*), tol_svd and tol_gram variants; accuracy: held-out moment error,
 * N, QP gap of each variant vs the python algorithm ("hard").
 */
TEST_CASE("gw_line_closure_noise", "[.closure_noise]") {
  lih_t L;
  auto &comm = L.mpi->comm;
  auto &mf   = *L.mf;
  const std::string set = std::getenv("GW_LINE_CN_SET") ? std::getenv("GW_LINE_CN_SET") : "test";
  const std::string rp  = std::getenv("GW_LINE_CN_REPR") ? std::getenv("GW_LINE_CN_REPR") : "leh";
  const long it_c = long(env_or("GW_LINE_CN_ITER", 2));
  const double amp = env_or("GW_LINE_CN_AMP", 1e-12);
  const long nseed = long(env_or("GW_LINE_CN_SEEDS", 6));
  const double meV = 27.211386e3;
  const std::string f = "gw_line_cn";
  const std::string repr = (rp == "cmp") ? "compressed" : "lehmann";
  auto pt = (set == "prod") ? prod_params(f, it_c, "id", repr) : scf_params(f, it_c, false, "id", repr);
  pt.put("lam_b", -1.0);
  auto R = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pt);
  auto prm = methods::gw_line::gw_line_params_t::from_ptree(pt);
  const long nk = mf.nkpts(), nb = mf.nbnd();
  nda::array<ComplexType, 3> F;
  nda::array<ComplexType, 4> Sp, Sh;
  double mu = 0.0;
  if (comm.root()) {
    h5::file fh(f + ".gw_line.h5", 'r');
    h5::group g(fh);
    auto gp = g.open_group("scf_line/iter" + std::to_string(it_c - 1));
    nda::h5_read(gp, "F", F);
    h5::h5_read(gp, "mu", mu);
    auto gi = g.open_group("scf_line/iter" + std::to_string(it_c));
    nda::h5_read(gi, "Sigma_p", Sp);
    nda::h5_read(gi, "Sigma_h", Sh);
  }
  {
    std::array<long, 4> shp{};
    if (comm.root()) shp = Sp.shape();
    comm.broadcast_n(shp.data(), 4, 0);
    if (not comm.root()) { Sp.resize(shp); Sh.resize(shp); F.resize(std::array<long, 3>{nk, nb, nb}); }
    comm.broadcast_n(Sp.data(), Sp.size(), 0);
    comm.broadcast_n(Sh.data(), Sh.size(), 0);
    comm.broadcast_n(F.data(), F.size(), 0);
    comm.broadcast_value(mu, 0);
  }
  remove_file(comm, f + ".gw_line.h5");
  nda::array<ComplexType, 3> Hrel(nk, nb, nb);
  for (long ik = 0; ik < nk; ++ik)
    for (long i = 0; i < nb; ++i)
      for (long j = 0; j < nb; ++j) Hrel(ik, i, j) = R.H0(ik, i, j) + F(ik, i, j) - (i == j ? mu : 0.0);
  const double theta = prm.theta_deg * std::numbers::pi / 180.0;
  auto zeta = numerics::line_dlr::dense_nodes(theta, prm.node_tmin, prm.node_tmax, prm.nodes_per_ray);
  line_basis_t bp(theta, prm.lam, prm.eps, prm.lam, prm.sigma_gap, -1.0, prm.node_tmax);
  line_basis_t bh(theta, prm.lam, prm.eps, prm.sigma_gap, prm.lam, -1.0, prm.node_tmax);
  line_basis_t gp(theta, prm.lam, prm.eps, prm.lam, prm.g_gap, -1.0, prm.node_tmax);
  line_basis_t gh(theta, prm.lam, prm.eps, prm.g_gap, prm.lam, -1.0, prm.node_tmax);
  g_repr_params_t grepr{prm.g_repr, prm.g_emax, prm.g_wtol, prm.g_emin_frac, prm.g_wsmall};
  const double nelec = double(mf.nelec());
  app_log(1, "\n[closure_noise] lih222 {} ({}), closure of iteration {} (K {}, eps {:.0e}, tol_gram {:.0e}), moment noise {:.0e}, "
             "{} seeds, ranks {}{}",
          set, repr, it_c, prm.K, prm.eps, prm.tol_gram, amp, nseed, comm.size(),
          (std::getenv("GW_LINE_CN_MODE") and std::string(std::getenv("GW_LINE_CN_MODE")) == "sigma") ? " -- NOISE ON SIGMA AT THE NODES" : "");

  // GW_LINE_CN_MODE = moments (default): relative noise on the Cayley moments (closure_params_t::moment_noise);
  // sigma: relative noise amp x max|Sigma| on Sigma^{>/<} at the nodes (the input of the sector fits)
  const bool sig_mode = std::getenv("GW_LINE_CN_MODE") and std::string(std::getenv("GW_LINE_CN_MODE")) == "sigma";
  const double smax   = nda::max_element(nda::abs(Sp + Sh));
  auto run = [&](closure_params_t const &cp0) {
    utils::TimerManager Tc;
    if (sig_mode and cp0.moment_noise > 0.0) {
      auto cp = cp0;
      cp.moment_noise = 0.0;
      std::mt19937_64 gen(cp0.noise_seed);
      std::normal_distribution<double> N01;
      nda::array<ComplexType, 4> Np = Sp, Nh = Sh;
      for (auto &x : Np) x += cp0.moment_noise * smax * ComplexType(N01(gen), N01(gen));
      for (auto &x : Nh) x += cp0.moment_noise * smax * ComplexType(N01(gen), N01(gen));
      return closure(comm, Hrel, Np, Nh, zeta, bp, bh, gp, gh, cp, nelec, Tc, grepr);
    }
    return closure(comm, Hrel, Sp, Sh, zeta, bp, bh, gp, gh, cp0, nelec, Tc, grepr);
  };
  auto base = [&]() {
    closure_params_t cp{prm.wp, prm.K, prm.tol_gram, prm.nphi};
    return cp;
  };
  struct resp_t { double dG = 0, dGp = 0, dmu = 0, dh = 0, dl = 0; long fl_g = 0, fl_s = 0, fl_p = 0, fl_qp = 0; };
  auto compare = [&](closure_out_t const &a, closure_out_t const &b) {
    resp_t x;
    auto Ga = closure_G_axis(a, nb, false), Gb = closure_G_axis(b, nb, false);
    auto Pa = closure_G_axis(a, nb, true), Pb = closure_G_axis(b, nb, true);
    x.dG  = nda::max_element(nda::abs(Gb - Ga)) / nda::max_element(nda::abs(Ga));
    x.dGp = nda::max_element(nda::abs(Pb - Pa)) / nda::max_element(nda::abs(Pa));
    x.dmu = std::abs(b.dmu - a.dmu) * meV;
    x.dh  = std::abs((b.e_homo + b.dmu) - (a.e_homo + a.dmu)) * meV;
    x.dl  = std::abs((b.e_lumo + b.dmu) - (a.e_lumo + a.dmu)) * meV;
    for (long ik = 0; ik < nk; ++ik) {
      x.fl_g += (a.diag[ik].r_gram != b.diag[ik].r_gram);
      x.fl_s += (a.diag[ik].r1 != b.diag[ik].r1);
      x.fl_p += (a.diag[ik].phi_index != b.diag[ik].phi_index);
    }
    return x;
  };
  auto summary = [&](std::string const &nm, closure_params_t cp, bool print_diag) {
    auto ref = run(cp);
    double hmax = *std::max_element(ref.heldout.begin(), ref.heldout.end());
    long gn = 0, sn = 0, np = 0;
    double gm = 1e300, sm = 1e300, tie = 1e300;
    for (auto const &d : ref.diag) {
      gn += d.gram_near; sn += d.svd_near; np += d.r_gram;
      gm = std::min(gm, d.gram_margin); sm = std::min(sm, d.svd_margin); tie = std::min(tie, d.phi_tie);
    }
    if (print_diag) {
      app_log(1, "  [{}] unperturbed: per k r_gram / r1 / n_free / phi idx / phi tie / Gram near (margin dec) / svd near (margin dec)", nm);
      for (long ik = 0; ik < nk; ++ik) {
        auto const &d = ref.diag[ik];
        app_log(1, "    k {}: {:4d} {:4d} {:3d} {:2d} {:8.3f} | {:2d} ({:.2e}) ratio at cut {:.2e} | {:2d} ({:.2e})", ik, d.r_gram, d.r1,
                d.r_gram - d.r1, d.phi_index, d.phi_tie, d.gram_near, d.gram_margin, d.gram_ratio, d.svd_near, d.svd_margin);
      }
    }
    std::vector<resp_t> rs;
    std::vector<closure_out_t> noisy;
    for (long s = 1; s <= nseed; ++s) {
      auto c = cp;
      c.moment_noise = amp;
      c.noise_seed   = unsigned(s);
      noisy.push_back(run(c));
      rs.push_back(compare(ref, noisy.back()));
    }
    resp_t mx;
    double med = 0.0;
    std::vector<double> dgs;
    for (auto const &x : rs) {
      mx.dG = std::max(mx.dG, x.dG); mx.dGp = std::max(mx.dGp, x.dGp); mx.dmu = std::max(mx.dmu, x.dmu);
      mx.dh = std::max(mx.dh, x.dh); mx.dl = std::max(mx.dl, x.dl);
      mx.fl_g += x.fl_g; mx.fl_s += x.fl_s; mx.fl_p += x.fl_p;
      dgs.push_back(x.dG);
    }
    std::sort(dgs.begin(), dgs.end());
    med = dgs[dgs.size() / 2];
    app_log(1, "[closure_noise] {:<22s} | amplification G max {:.1e} median {:.1e}, pruned G {:.1e} | dmu {:.2e} dVBM {:.2e} dCBM {:.2e} meV "
               "| flips/{} k-runs: Gram {} r1 {} phase {} | poles {} held-out {:.1e} N {:.6f} gap {:.6f} eV | near: Gram {} (min {:.1e} "
               "dec) svd {} (min {:.1e} dec) phase tie {:.3f}",
            nm, mx.dG / amp, med / amp, mx.dGp / amp, mx.dmu, mx.dh, mx.dl, nseed * nk, mx.fl_g, mx.fl_s, mx.fl_p, np, hmax, ref.N_mu,
            (ref.e_lumo - ref.e_homo) * 27.211386, gn, gm, sn, sm, std::min(tie, 999.0));
    return std::make_tuple(ref, noisy, rs);
  };

  // 1. python algorithm ("hard"): response and attribution by freezing decisions to the unperturbed values
  auto [ref, noisy, rs] = summary("hard (python)", base(), true);
  {
    std::vector<long> rg(nk), r1(nk);
    std::vector<double> ph(nk);
    for (long ik = 0; ik < nk; ++ik) { rg[ik] = ref.diag[ik].r_gram; r1[ik] = ref.diag[ik].r1; ph[ik] = ref.diag[ik].phi; }
    struct fz_t { std::string nm; bool g, s, p; };
    for (auto const &fz : std::vector<fz_t>{{"freeze all", true, true, true}, {"free Gram only", false, true, true},
                                             {"free r1 only", true, false, true}, {"free phase only", true, true, false}}) {
      double dG = 0, dmu = 0, de = 0;
      for (long s = 1; s <= nseed; ++s) {
        auto c = base();
        c.moment_noise = amp;
        c.noise_seed   = unsigned(s);
        if (fz.g) c.force_rgram = rg;
        if (fz.s) c.force_r1 = r1;
        if (fz.p) c.force_phi = ph;
        auto o = run(c);
        auto x = compare(ref, o);
        dG = std::max(dG, x.dG); dmu = std::max(dmu, x.dmu); de = std::max({de, x.dh, x.dl});
      }
      app_log(1, "[closure_noise]   attribution {:<16s}: amplification G {:.1e}, dmu {:.2e} meV, QP edges {:.2e} meV", fz.nm, dG / amp, dmu,
              de);
    }
    // QP selection: QP-like poles (weight > 0.1) near the edges and poles close to the 0.1 threshold
    long near_w = 0;
    for (long ik = 0; ik < nk; ++ik)
      for (long m = 0; m < ref.leh.e[ik].size(); ++m) {
        double w = 0.0;
        for (long i = 0; i < nb; ++i) w += std::norm(ref.leh.v[ik](i, m));
        if (w > 0.05 and w < 0.2 and std::abs(ref.leh.e[ik](m)) < 2.0 * (ref.e_lumo - ref.e_homo)) ++near_w;
      }
    app_log(1, "[closure_noise]   QP selection: poles with weight in (0.05, 0.2) within 2 gaps of mu: {}; edges VBM {:.6f} CBM {:.6f} Ha "
               "(rel. new mu)",
            near_w, ref.e_homo, ref.e_lumo);
  }

  // 2. remedies
  auto variant = [&](std::string const &nm, std::function<void(closure_params_t &)> const &mod) {
    auto cp = base();
    mod(cp);
    auto [r, nz, rr] = summary(nm, cp, std::getenv("GW_LINE_CN_DIAG") != nullptr);
    auto x = compare(ref, r);
    app_log(1, "[closure_noise]   {:<22s} vs hard (accuracy): G {:.2e}, pruned G {:.2e}, dmu {:.3f} meV, dVBM {:.3f} dCBM {:.3f} meV", nm, x.dG,
            x.dGp, x.dmu, x.dh, x.dl);
  };
  variant("gap cut (Gram)", [](closure_params_t &c) { c.gram_cut = "gap"; });
  variant("smooth cut (Gram)", [](closure_params_t &c) { c.gram_cut = "smooth"; });
  variant("gap cut (Gram + SVD)", [](closure_params_t &c) { c.gram_cut = "gap"; c.svd_cut = "gap"; });
  variant("smooth Gram + gap SVD", [](closure_params_t &c) { c.gram_cut = "smooth"; c.svd_cut = "gap"; });
  {
    std::vector<double> ph(nk);
    for (long ik = 0; ik < nk; ++ik) ph[ik] = ref.diag[ik].phi;
    variant("phase continuity", [&](closure_params_t &c) { c.phase_keep = 10.0; c.phi_prev = ph; });
  }
  for (double ts : {1e-10, 1e-8})
    variant("tol_svd " + std::to_string(ts).substr(0, 0) + (ts == 1e-10 ? "1e-10" : "1e-8"), [=](closure_params_t &c) { c.tol_svd = ts; });
  for (double tg : {1e-9, 1e-8})
    variant(std::string("tol_gram ") + (tg == 1e-9 ? "1e-9" : "1e-8"), [=](closure_params_t &c) { c.tol_gram = tg; });
}

/// (hidden) regression dump for code changes that must leave the q = -q fixtures bitwise unchanged (the q <-> -q pairing
/// fix of 2026-10-04): on the STORED lih222 THC (deterministic input), KS poles -> Pi -> W residues -> Sigma (both sectors,
/// GL rays) and a 2-iteration driver run (compressed, gl); writes w, Sigma, the driver's mu / Sigma to $GW_LINE_REGRESSION_H5.
/// Run with the old and the new binary on 1 rank and compare the files.
TEST_CASE("gw_line_regression_dump", "[.gw_line_regression_dump]") {
  lih_t L;
  using numerics::line_dlr::time_ray_t;
  auto &mpi = *L.mpi;
  auto &mf  = *L.mf;
  char const *fn = std::getenv("GW_LINE_REGRESSION_H5");
  REQUIRE(fn != nullptr);
  const long nk = mf.nkpts(), nb = mf.nbnd(), nocc = long(std::llround(double(mf.nelec()) / 2.0));
  nda::array<double, 2> eig(nk, nb);
  double homo = -1e300, lumo = 1e300;
  for (long ik = 0; ik < nk; ++ik)
    for (long n = 0; n < nb; ++n) {
      eig(ik, n) = mf.eigval()(0, ik, n);
      if (n < nocc) homo = std::max(homo, eig(ik, n));
      else lumo = std::min(lumo, eig(ik, n));
    }
  auto pd = pole_data_t::from_ks(eig, 0.5 * (homo + lumo));
  const double th = 20.0 * std::numbers::pi / 180.0;
  numerics::line_dlr::bosonic_basis_t bos(th, 4.0, 1e-10, 0.5 * (lumo - homo));
  auto fz = numerics::line_dlr::dense_nodes(th, 1e-3, 60.0, 120);
  aux_grid_t grid(mpi, L.thc->Np());
  dyson_layout_t lay(mpi, mf.nqpts(), bos.zeta_nodes.size(), L.thc->Np());
  utils::TimerManager T;
  coulomb_blocks_t<HOST_MEMORY> Zb(*L.thc, grid, lay.q_rng(), T);
  propagator_t<HOST_MEMORY> prop(*L.thc, grid);
  auto rp = time_ray_t::for_spectrum(0.5 * th, pd.emin(), 36.0, 1e-5, 3.0, 16, sector_t::particle);
  auto rh = time_ray_t::for_spectrum(0.5 * th, pd.emin(), 36.0, 1e-5, 3.0, 16, sector_t::hole);
  memory::array<HOST_MEMORY, ComplexType, 4> Pi, w;
  polarization<HOST_MEMORY>(prop, pd, mf, grid, bos.zeta_nodes, rp, rh, 8, Pi, T);
  screened_interaction<HOST_MEMORY>(Pi, Zb, bos, grid, mpi, w, T);
  nda::array<ComplexType, 4> Sp, Sh;
  self_energy<HOST_MEMORY>(prop, pd, w, bos, mf, grid, mpi, fz, rp, rh, 8, Sp, T, sector_t::particle);
  self_energy<HOST_MEMORY>(prop, pd, w, bos, mf, grid, mpi, fz, rp, rh, 8, Sh, T, sector_t::hole);
  auto pt = scf_params("gw_line_regdump", 2, false, "gl", "compressed");
  auto R  = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, mf, pt);
  if (mpi.comm.root()) {
    h5::file f(fn, 'w');
    h5::group g(f);
    nda::array<ComplexType, 4> wh(w);
    nda::h5_write(g, "w", wh, false);
    nda::h5_write(g, "Sigma_p", Sp, false);
    nda::h5_write(g, "Sigma_h", Sh, false);
    nda::h5_write(g, "scf_Sig_p", R.Sig_p, false);
    nda::h5_write(g, "scf_Sig_h", R.Sig_h, false);
    h5::h5_write(g, "scf_mu", R.mu);
  }
  remove_file(mpi.comm, "gw_line_regdump.gw_line.h5");
}

// ======================================================================================================================
// perf 7.2: mixing (DIIS), spectra/optics consistency, warm starts, multilevel schedule (lih222, small settings)
// ======================================================================================================================
namespace {
ptree p72_params(std::string const &f, long niter, bool restart = false) {
  auto pt = scf_params(f, niter, restart, "id", "lehmann");
  pt.put("lam_b", -1.0);   // auto (S7c)
  return pt;
}
template <typename T, int R> void bcast_test(boost::mpi3::communicator &comm, nda::array<T, R> &A) {
  std::array<long, R> shp{};
  if (comm.root()) shp = A.shape();
  comm.broadcast_n(shp.data(), R, 0);
  if (not comm.root()) A.resize(shp);
  if (A.size() > 0) comm.broadcast_n(A.data(), A.size(), 0);
}
void p72_table(std::string const &tag, gw_line_result_t const &R) {
  for (auto const &h : R.history)
    app_log(1, "  [{}] iter {}: level {} mix {:<6s} m {} resid {:.3e} dSigma {:.3e} residF {:.3e} mu {:.8f} gap {:.6f} eV", tag, h.iter,
            h.level, h.mix, h.ndiis, h.resid, h.dSigma, h.residF, h.mu, h.gap * 27.211386);
}
} // namespace

/// [mixing] linear vs DIIS (4 iterations each): the residual max|Sigma[G] - Sigma_in| of the linear run is dSigma / mixing
/// (definition); the DIIS run extrapolates from iteration 2 and its residual after 4 iterations is below the linear run's;
/// damp_below: the damped tail; restart of a DIIS run: the history is not checkpointed, the first iteration after the restart is a 1-entry DIIS step
/// (x + beta r), the restored history carries the mixing records.
TEST_CASE("gw_line_scf_mixing", "[gw_line][scf][mixing]") {
  lih_t L;
  auto &comm = L.mpi->comm;
  const long n = 4;
  auto pl = p72_params("gw_line_mixL", n);
  auto Lr = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pl);
  auto pd = p72_params("gw_line_mixD", n);
  pd.put("mixing_alg", "diis");
  auto Dr = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pd);
  p72_table("linear", Lr);
  p72_table("diis", Dr);
  REQUIRE(long(Lr.history.size()) == n);
  REQUIRE(long(Dr.history.size()) == n);
  REQUIRE(Lr.history[0].mix == "none");
  REQUIRE(Dr.history[0].mix == "none");
  for (long i = 1; i < n; ++i) {
    REQUIRE(Lr.history[i].mix == "linear");
    REQUIRE(std::abs(Lr.history[i].dSigma - 0.5 * Lr.history[i].resid) <= 1e-12 * Lr.history[i].resid);
    REQUIRE(Dr.history[i].mix == "diis");
    REQUIRE(Dr.history[i].ndiis == std::min(i, 6L));
    REQUIRE(std::isfinite(Dr.history[i].resid));
  }
  // iteration 2 is identical up to the mixing step (same G_1); DIIS with 1 entry and beta 1 = mixing 1
  REQUIRE(Dr.history[1].resid == Lr.history[1].resid);
  REQUIRE(Dr.history[n - 1].resid < Lr.history[n - 1].resid);
  // damped tail: mixing 1 with damp_below above every residual = linear 0.5 from iteration 2 on (bitwise)
  auto pt = p72_params("gw_line_mixT", n);
  pt.put("mixing", 1.0);
  pt.put("damp_below", 1.0);
  pt.put("damp_mixing", 0.5);
  auto Tr = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pt);
  for (long i = 1; i < n; ++i) {
    REQUIRE(Tr.history[i].mix == "damped");
    REQUIRE(Tr.history[i].resid == Lr.history[i].resid);
    REQUIRE(Tr.history[i].mu == Lr.history[i].mu);
  }
  // restart: 2 + restart + 2
  auto p2 = p72_params("gw_line_mixR", 2);
  p2.put("mixing_alg", "diis");
  auto R2 = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, p2);
  auto p4 = p72_params("gw_line_mixR", n, true);
  p4.put("mixing_alg", "diis");
  auto R4 = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, p4);
  p72_table("diis 2 + restart + 2", R4);
  REQUIRE(long(R4.history.size()) == n);
  REQUIRE(R4.history[1].mix == "diis");
  REQUIRE(R4.history[1].resid == Dr.history[1].resid);   // restored record
  REQUIRE(R4.history[2].mix == "diis");
  REQUIRE(R4.history[2].ndiis == 1);                     // rebuilt history
  REQUIRE(R4.history[3].ndiis == 2);
  REQUIRE(R4.history[3].resid < 2.0 * Lr.history[3].resid);
  app_log(1, "[mixing] ranks {}: residual after {} iterations: linear {:.3e}, diis {:.3e}, diis with a restart after 2 {:.3e}", comm.size(),
          n, Lr.history[n - 1].resid, Dr.history[n - 1].resid, R4.history[n - 1].resid);
  for (auto f : {"gw_line_mixL", "gw_line_mixD", "gw_line_mixR", "gw_line_mixT"}) remove_file(comm, std::string(f) + ".gw_line.h5");
}

/// [spectra_F] the spectra are those of the stored G: niter = 1 from the KS start: F_closure = F[D_KS] (iteration 0 F,
/// bitwise), the QP edges of the spectra = those of the closure of iteration 1 (same upfolded G; before 7.2 the spectra used
/// F[D_1]); restart with nothing to iterate: F_closure restored, identical spectra.
TEST_CASE("gw_line_scf_spectra_F", "[gw_line][scf][spectra_F]") {
  lih_t L;
  auto &comm = L.mpi->comm;
  auto pt    = p72_params("gw_line_spF", 1);
  enable_spectra(pt, 41);
  auto R = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pt);
  REQUIRE(R.spectra.has_value());
  nda::array<ComplexType, 3> F0;
  if (comm.root()) {
    h5::file f("gw_line_spF.gw_line.h5", 'r');
    h5::group g(f);
    nda::h5_read(g, "scf_line/iter0/F", F0);
  }
  bcast_test(comm, F0);
  const double dF0 = nda::max_element(nda::abs(R.F_closure - F0));
  const double dF1 = nda::max_element(nda::abs(R.F - R.F_closure));
  auto const &h    = R.history[0];
  const double dh = std::abs(R.spectra->e_homo - h.e_homo), dl = std::abs(R.spectra->e_lumo - h.e_lumo);
  app_log(1, "[spectra_F] ranks {}: F_closure - F[D_KS] {:.1e}, F[D_1] - F_closure {:.3e} Ha; spectra edges vs closure of iteration 1: "
             "homo {:.2e} lumo {:.2e} Ha (gap {:.6f} vs {:.6f} eV)",
          comm.size(), dF0, dF1, dh, dl, (R.spectra->e_lumo - R.spectra->e_homo) * 27.211386, h.gap * 27.211386);
  REQUIRE(dF0 == 0.0);
  REQUIRE(dF1 > 1e-6);
  REQUIRE(dh < 1e-6);
  REQUIRE(dl < 1e-6);
  auto pr = p72_params("gw_line_spF", 1, true);
  enable_spectra(pr, 41);
  auto R2 = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pr);
  REQUIRE(R2.spectra.has_value());
  REQUIRE(nda::max_element(nda::abs(R2.F_closure - R.F_closure)) == 0.0);
  REQUIRE(nda::max_element(nda::abs(R2.spectra->A_diag - R.spectra->A_diag)) == 0.0);
  remove_file(comm, "gw_line_spF.gw_line.h5");
}

/// [qp_start] start = "qp_diag": one Pi -> W -> Sigma pass on the KS poles (not an iteration), KS vectors + diagonal QP
/// energies (iteration 0 poles: unit vectors, mu = QP mid-gap, F = F[D_KS]); start = "qp_file": the energies of a file
/// (KS + scissor) become the iteration-0 poles exactly.
TEST_CASE("gw_line_scf_qp_start", "[gw_line][scf][qp_start]") {
  lih_t L;
  auto &comm = L.mpi->comm;
  auto &mf   = *L.mf;
  const long nk = mf.nkpts(), nb = mf.nbnd(), nocc = long(std::llround(mf.nelec() / 2.0));
  auto pq    = p72_params("gw_line_qpd", 2);
  pq.put("start", "qp_diag");
  auto Q = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, mf, pq);
  p72_table("qp_diag", Q);
  REQUIRE(Q.history.size() == 2);
  REQUIRE(Q.start_time > 0.0);
  REQUIRE(Q.eps_inf.size() == 2);
  double ks_gap = 0.0, qp_gap = 0.0, mu0 = 0.0;
  {
    double h = -1e300, l = 1e300;
    for (long k = 0; k < nk; ++k)
      for (long b = 0; b < nb; ++b) (b < nocc ? h : l) = (b < nocc ? std::max(h, mf.eigval()(0, k, b)) : std::min(l, mf.eigval()(0, k, b)));
    ks_gap = l - h;
  }
  if (comm.root()) {
    h5::file f("gw_line_qpd.gw_line.h5", 'r');
    h5::group g(f);
    auto it0 = g.open_group("scf_line/iter0");
    double eh = 0, el = 0;
    h5::h5_read(it0, "e_homo", eh);
    h5::h5_read(it0, "e_lumo", el);
    h5::h5_read(it0, "mu", mu0);
    qp_gap = el - eh;
    long sd = 0;
    h5::h5_read(g.open_group("scf_line"), "start_done", sd);
    REQUIRE(sd == 1);
    REQUIRE(std::abs(eh + el) < 1e-12);   // mu = QP mid-gap
  }
  comm.broadcast_n(&qp_gap, 1, 0);
  app_log(1, "[qp_start] qp_diag: KS gap {:.4f} eV -> diagonal G0W0 QP gap {:.4f} eV; first iteration gap {:.4f} eV (start pass {:.1f} s)",
          ks_gap * 27.211386, qp_gap * 27.211386, Q.history[0].gap * 27.211386, Q.start_time);
  REQUIRE(qp_gap > ks_gap);
  REQUIRE(Q.history[0].mix == "none");
  REQUIRE(Q.history[1].mix == "linear");
  // qp_file: KS + 0.05 Ha scissor on the conduction bands
  nda::array<double, 2> E(nk, nb);
  for (long k = 0; k < nk; ++k)
    for (long b = 0; b < nb; ++b) E(k, b) = mf.eigval()(0, k, b) + (b < nocc ? 0.0 : 0.05);
  if (comm.root()) {
    h5::file f("gw_line_qpf_energies.h5", 'w');
    h5::group g(f);
    nda::h5_write(g, "qp_energies", E, false);
  }
  comm.barrier();
  auto pf = p72_params("gw_line_qpf", 1);
  pf.put("start", "qp_file");
  pf.put("start_file", "gw_line_qpf_energies.h5");
  auto Fr = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, mf, pf);
  REQUIRE(Fr.history.size() == 1);
  if (comm.root()) {
    h5::file f("gw_line_qpf.gw_line.h5", 'r');
    h5::group g(f);
    auto it0 = g.open_group("scf_line/iter0");
    double mu = 0.0;
    h5::h5_read(it0, "mu", mu);
    nda::array<double, 1> ep, eh;
    nda::array<long, 1> cp, ch;
    nda::h5_read(it0, "poles/particle_e", ep);
    nda::h5_read(it0, "poles/hole_e", eh);
    nda::h5_read(it0, "poles/particle_counts", cp);
    nda::h5_read(it0, "poles/hole_counts", ch);
    double d = 0.0;
    long op = 0, oh = 0;
    for (long k = 0; k < nk; ++k) {
      REQUIRE(ch(k) == nocc);
      REQUIRE(cp(k) == nb - nocc);
      for (long b = 0; b < nocc; ++b) d = std::max(d, std::abs(eh(oh++) - (E(k, b) - mu)));
      for (long b = nocc; b < nb; ++b) d = std::max(d, std::abs(ep(op++) - (E(k, b) - mu)));
    }
    app_log(1, "[qp_start] qp_file: iteration-0 poles vs file energies - mu: {:.1e}", d);
    REQUIRE(d == 0.0);
  }
  for (auto f : {"gw_line_qpd", "gw_line_qpf"}) remove_file(comm, std::string(f) + ".gw_line.h5");
  remove_file(comm, "gw_line_qpf_energies.h5");
}

/// [multilevel] coarse = { niter 1, eps 1e-6, K 6, nodes_per_ray 60, time_eps 1e-6 } then production: levels 1, 0, 0; the
/// coarse Sigma is dropped at the switch (iteration 2 unmixed), the production iterations equal those of a run started from
/// the coarse iteration's poles (same Sigma bases and nodes as a cold production run).
TEST_CASE("gw_line_scf_multilevel", "[gw_line][scf][multilevel]") {
  lih_t L;
  auto &comm = L.mpi->comm;
  auto pm    = p72_params("gw_line_ml", 3);
  pm.put("coarse.niter", 1);
  pm.put("coarse.eps", 1e-6);
  pm.put("coarse.K", 6);
  pm.put("coarse.nodes_per_ray", 60);
  pm.put("coarse.time_eps", 1e-6);
  auto M = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pm);
  p72_table("multilevel", M);
  auto C = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, p72_params("gw_line_mlc", 3));
  p72_table("cold", C);
  REQUIRE(M.history.size() == 3);
  REQUIRE(M.history[0].level == 1);
  REQUIRE(M.history[1].level == 0);
  REQUIRE(M.history[2].level == 0);
  REQUIRE(M.history[1].mix == "none");
  REQUIRE(M.history[2].mix == "linear");
  REQUIRE(M.Sig_p.extent(1) == C.Sig_p.extent(1));   // production nodes at the end
  // restart after the coarse iteration (M's checkpoint truncated to iteration 1) = M's production iterations (bitwise)
  auto pr = pm;
  pr.put("output", "gw_line_mlr");
  pr.put("restart", true);
  if (comm.root()) {
    std::filesystem::copy_file("gw_line_ml.gw_line.h5", "gw_line_mlr.gw_line.h5", std::filesystem::copy_options::overwrite_existing);
    h5::file f("gw_line_mlr.gw_line.h5", 'a');
    h5::group g(f);
    auto sg = g.open_group("scf_line");
    h5::h5_write(sg, "final_iter", long(1));
    sg.unlink("iter2");
    sg.unlink("iter3");
  }
  comm.barrier();
  auto R = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pr);
  const double dmu = std::abs(R.mu - M.mu), dpo = maxdiff_poles(R.poles, M.poles);
  app_log(1, "[multilevel] ranks {}: mu {:.8f} (cold {:.8f}), gap {:.6f} eV (cold {:.6f}); restart after the coarse iteration: |dmu| {:.1e} "
             "poles {:.1e}; time per iteration coarse {:.1f} s, production {:.1f} s",
          comm.size(), M.mu, C.mu, M.history[2].gap * 27.211386, C.history[2].gap * 27.211386, dmu, dpo, M.history[0].time,
          M.history[2].time);
  REQUIRE(dmu == 0.0);
  REQUIRE(dpo == 0.0);
  for (auto f : {"gw_line_ml", "gw_line_mlc", "gw_line_mlr"}) remove_file(comm, std::string(f) + ".gw_line.h5");
}

/// [optics_initial] optics.poles = "initial" in a niter = 1 run (G0W0 next to RPA@KS): every optics line from the KS poles. SCF
/// angle: the head of iteration 1 (bitwise the checkpoint's iter1/head); vs a niter = 0 run (final poles = KS, the heads from
/// passes): SCF-angle heads equal to the kernel accuracy, flatter line (both passes on the KS poles) bitwise.
TEST_CASE("gw_line_scf_optics_initial", "[gw_line][scf][optics_initial]") {
  lih_t L;
  auto &comm = L.mpi->comm;
  auto par   = [&](std::string const &f, long n, std::string const &poles) {
    auto pt = p72_params(f, n);
    pt.put("div_treatment", "gygi");
    pt.put("eps", 1e-10);
    pt.put("optics.enable", true);
    pt.put("optics.wmin", 0.0);
    pt.put("optics.wmax", 1.0);
    pt.put("optics.nw", 101);
    pt.add_child("optics.eta_rel", arr_child({0.05}));
    pt.put("optics.theta_deg", 10.0);
    pt.put("optics.poles", poles);
    return pt;
  };
  auto A = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, par("gw_line_oiA", 1, "initial"));
  auto B = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, par("gw_line_oiB", 0, "final"));
  REQUIRE(A.optics_q0.size() == 2);
  REQUIRE(B.optics_q0.size() == 2);
  double d20 = 0.0, s20 = 0.0, d10 = 0.0, dit = 0.0;
  if (comm.root()) {
    h5::file fa("gw_line_oiA.gw_line.h5", 'r'), fb("gw_line_oiB.gw_line.h5", 'r');
    h5::group ga(fa), gb(fb);
    nda::array<ComplexType, 2> a20, b20, a10, b10, h1;
    nda::h5_read(ga, "optics/theta20.0/h_nodes", a20);
    nda::h5_read(gb, "optics/theta20.0/h_nodes", b20);
    nda::h5_read(ga, "optics/theta10.0/h_nodes", a10);
    nda::h5_read(gb, "optics/theta10.0/h_nodes", b10);
    nda::h5_read(ga, "scf_line/iter1/head/h_nodes", h1);
    std::string sa, sb;
    h5::h5_read(ga, "optics/theta20.0/source", sa);
    h5::h5_read(gb, "optics/theta20.0/source", sb);
    app_log(1, "  sources: initial run: \"{}\"; niter 0 run: \"{}\"", sa, sb);
    REQUIRE(sa.find("iteration 1") != std::string::npos);
    REQUIRE(a20.shape() == b20.shape());
    REQUIRE(a10.shape() == b10.shape());
    d20 = nda::max_element(nda::abs(a20 - b20));
    s20 = nda::max_element(nda::abs(b20));
    d10 = nda::max_element(nda::abs(a10 - b10));
    dit = nda::max_element(nda::abs(a20 - h1));
  }
  comm.broadcast_n(&d20, 1, 0);
  comm.broadcast_n(&s20, 1, 0);
  comm.broadcast_n(&d10, 1, 0);
  comm.broadcast_n(&dit, 1, 0);
  app_log(1, "[optics_initial] ranks {}: SCF-angle head (iteration 1) vs the niter = 0 pass {:.1e} (rel), vs iter1/head {:.1e}; 10-deg "
             "line initial (niter 1) vs niter 0: {:.1e}; eps_inf {:.8f} / {:.8f}",
          comm.size(), d20 / s20, dit, d10, A.optics_q0[0].eps_inf_h, B.optics_q0[0].eps_inf_h);
  REQUIRE(dit == 0.0);
  REQUIRE(d20 <= 1e-9 * s20);
  REQUIRE(d10 == 0.0);
  for (auto f : {"gw_line_oiA", "gw_line_oiB"}) {
    remove_file(comm, std::string(f) + ".gw_line.h5");
    remove_file(comm, std::string(f) + ".gw_line.sigma.h5");
  }
}
