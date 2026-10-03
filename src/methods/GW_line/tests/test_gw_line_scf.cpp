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
 * [parity] 6 iterations vs the python driver on the same THC/H0/KS data (coqui/cayley/scripts/gen_lih222_scf_ref.py ->
 *   tests/unit_test_files/gw_line/lih222_scf_ref.h5): mu and QP gap per iteration within 1 meV, Sigma at the stored nodes
 *   of k = 0 within 1e-5 relative (max norm over the stored nodes), per iteration.
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include <chrono>
#include <cstdlib>
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

// ======================================================================================================================
// lih222: driver, restart, python parity
// ======================================================================================================================
namespace {

std::string gw_line_dir() { return std::string(PROJECT_SOURCE_DIR) + "/tests/unit_test_files/gw_line/"; }
std::string lih_thc_file() { return gw_line_dir() + "lih222_thc/thc.eri.h5"; }

struct lih_t {
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;
  std::shared_ptr<mf::MF> mf;
  std::unique_ptr<methods::thc_reader_t> thc;
  lih_t() {
    mpi = utils::make_unit_test_mpi_context();
    mf  = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222"));
    if (std::filesystem::exists(lih_thc_file())) {
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
ptree scf_params(std::string const &output, long niter, bool restart) {
  ptree pt;
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
    d = std::max(d, nda::max_element(nda::abs(a.part[ik].coef - b.part[ik].coef)));
    d = std::max(d, nda::max_element(nda::abs(a.hole[ik].coef - b.hole[ik].coef)));
  }
  return d;
}

/// checkpoints are removed unless GW_LINE_TEST_KEEP is set (diagnostics against python)
void remove_file(boost::mpi3::communicator &comm, std::string const &f) {
  comm.barrier();
  if (comm.root() and std::getenv("GW_LINE_TEST_KEEP") == nullptr and std::filesystem::exists(f)) std::filesystem::remove(f);
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

TEST_CASE("gw_line_scf_restart", "[gw_line][scf][restart]") {
  lih_t L;
  auto &comm = L.mpi->comm;
  const std::string fa = "gw_line_rsA", fb = "gw_line_rsB";
  auto t0 = std::chrono::steady_clock::now();
  auto A  = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, scf_params(fa, 3, false));
  auto t1 = std::chrono::steady_clock::now();
  auto B2 = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, scf_params(fb, 2, false));
  auto pb = scf_params(fb, 3, true);
  enable_spectra(pb, 41);
  auto B3 = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, pb);
  auto t2 = std::chrono::steady_clock::now();
  REQUIRE(A.history.size() == 3);
  REQUIRE(B3.history.size() == 3);
  REQUIRE(B3.history[1].mu == B2.history[1].mu);
  const double dmu = std::abs(A.mu - B3.mu), dpo = maxdiff_poles(A.poles, B3.poles);
  const double dF  = nda::max_element(nda::abs(A.F - B3.F));
  const double dSp = nda::max_element(nda::abs(A.Sig_p - B3.Sig_p)), dSh = nda::max_element(nda::abs(A.Sig_h - B3.Sig_h));
  app_log(1, "[restart] ranks {}: 3 iterations ({:.1f} s) vs 2 + restart + 1 ({:.1f} s): |dmu| {:.1e}, poles {:.1e}, F {:.1e}, "
             "Sigma_p {:.1e}, Sigma_h {:.1e}",
          comm.size(), std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count(), dmu, dpo,
          dF, dSp, dSh);
  for (long i = 0; i < 3; ++i)
    app_log(1, "  iter {}: mu {:.10f} / {:.10f}  gap {:.8f} / {:.8f}  dSigma {:.3e} / {:.3e}", i + 1, A.history[i].mu,
            B3.history[i].mu, A.history[i].gap, B3.history[i].gap, A.history[i].dSigma, B3.history[i].dSigma);
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
  }
  const std::string fo = "gw_line_parity";
  auto t0 = std::chrono::steady_clock::now();
  auto R  = methods::gw_line::gw_line_scf<HOST_MEMORY>(*L.thc, *L.mf, scf_params(fo, niter, false));
  const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  REQUIRE(long(R.history.size()) == niter);
  const long nb = R.F.extent(1);
  app_log(1, "[parity] ranks {}, {} iterations in {:.1f} s ({} stored nodes of k = 0)", comm.size(), niter, dt, idx.size());
  app_log(1, "  iter |   mu C++ (Ha)    mu py (Ha)   d(meV) |  gap C++ (eV)  gap py (eV)  d(meV) |  N C++      N py     |  "
             "dSigma C++  dSigma py | Sigma rel");
  bool ok = true;
  for (long it = 0; it < niter; ++it) {
    nda::array<ComplexType, 4> Sp, Sh;
    if (comm.root()) {
      h5::file f(fo + ".gw_line.h5", 'r');
      h5::group g(f);
      auto gi = g.open_group("scf_line/iter" + std::to_string(it + 1));
      nda::h5_read(gi, "Sigma_p", Sp);
      nda::h5_read(gi, "Sigma_h", Sh);
    }
    std::array<long, 4> shp{};
    if (comm.root()) shp = Sp.shape();
    comm.broadcast_n(shp.data(), 4, 0);
    if (not comm.root()) { Sp.resize(shp); Sh.resize(shp); }
    comm.broadcast_n(Sp.data(), Sp.size(), 0);
    comm.broadcast_n(Sh.data(), Sh.size(), 0);
    double dmax = 0.0, smax = 0.0;
    for (long n = 0; n < idx.size(); ++n)
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) {
          const ComplexType s = Sp(0, idx(n), i, j) + Sh(0, idx(n), i, j);
          dmax = std::max(dmax, std::abs(s - Sig_r(it, n, i, j)));
          smax = std::max(smax, std::abs(Sig_r(it, n, i, j)));
        }
    auto const &h     = R.history[it];
    const double dmu  = (h.mu - mu_r(it)) * 27.211386e3, dgap = (h.gap - gap_r(it)) * 27.211386e3;
    app_log(1, "  {:4d} | {:.8f}  {:.8f}  {:+7.3f} | {:.6f}     {:.6f}    {:+7.3f} | {:.6f}  {:.6f} | {:.3e}  {:.3e} | {:.2e}",
            it + 1, h.mu, mu_r(it), dmu, h.gap * 27.211386, gap_r(it) * 27.211386, dgap, h.nelec, nel_r(it), h.dSigma, dS_r(it),
            dmax / smax);
    ok = ok and std::abs(dmu) < 1.0 and std::abs(dgap) < 1.0 and dmax / smax < 1e-5;
  }
  REQUIRE(ok);
  remove_file(comm, fo + ".gw_line.h5");
}

