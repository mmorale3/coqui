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
 * test_cayley (plan notes/line_gw_cpp_plan.md, session S2): the closure numerics of numerics/line_dlr/cayley.hpp.
 *   1. exact recovery of a 2-orbital 6-pole model from its Cayley moments (poles, residues, Sigma_c off the axis);
 *   2. Lehmann G from the upfolded Hamiltonian (completeness, G vs Dyson with the exact Sigma_c);
 *   3. Si 2x2x2 G0W0 Gamma demo from exact moments (tests/unit_test_files/gw_line/si222_moments_k0.h5, written by
 *      coqui/cayley/scripts/gen_cayley_ref.py): Tr A window error at K = 8, 16 vs python, nphi = 72 vs 8;
 *   4. chemical potential (widest admissible QP gap) on the toy of the same reference file.
 * Every measured number is printed.
 */

#undef NDEBUG

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <cstdlib>
#include <string>
#include <vector>

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "h5/h5.hpp"
#include "nda/h5.hpp"
#include "nda/nda.hpp"
#include "numerics/line_dlr/cayley.hpp"
#include "numerics/line_dlr/tests/closure_bench.hpp"

namespace bdft_tests {

namespace ldlr = numerics::line_dlr;
using dcomplex = std::complex<double>;

namespace {

const std::string ref_file = std::string(PROJECT_SOURCE_DIR) + "/tests/unit_test_files/gw_line/si222_moments_k0.h5";

template <typename T> T read_attr(h5::group &g, std::string const &name) {
  T v{};
  h5::h5_read_attribute(g, name, v);
  return v;
}

/// portable uniform deviate on [a, b) from the raw mt19937 stream (the <random> distributions are implementation-defined)
double uniform(std::mt19937 &gen, double a, double b) { return a + (b - a) * (double(gen()) / 4294967296.0); }

/// the 2-orbital model: 6 real poles spread over [-3, 3] Ha with random rank-1 PSD residues R_l = v_l v_l^dag
struct pole_model_t {
  nda::array<double, 1> E;
  nda::array<dcomplex, 2> V;        // [n, npoles]: W of the exact model
  nda::array<dcomplex, 3> R;        // [npoles, n, n]
};

pole_model_t make_model(unsigned seed) {
  pole_model_t m{nda::array<double, 1>{-2.6, -1.2, -0.3, 0.15, 0.9, 2.8}, nda::array<dcomplex, 2>(2, 6),
                 nda::array<dcomplex, 3>(6, 2, 2)};
  std::mt19937 gen(seed);
  for (long l = 0; l < 6; ++l)
    for (long i = 0; i < 2; ++i) m.V(i, l) = 0.5 * dcomplex(uniform(gen, -1.0, 1.0), uniform(gen, -1.0, 1.0));
  for (long l = 0; l < 6; ++l)
    for (long i = 0; i < 2; ++i)
      for (long j = 0; j < 2; ++j) m.R(l, i, j) = m.V(i, l) * std::conj(m.V(j, l));
  return m;
}

double max_abs(nda::array<dcomplex, 2> const &A) {
  double x = 0.0;
  for (long i = 0; i < A.extent(0); ++i)
    for (long j = 0; j < A.extent(1); ++j) x = std::max(x, std::abs(A(i, j)));
  return x;
}

std::vector<dcomplex> test_points(long nz, unsigned seed) {
  std::mt19937 gen(seed);
  std::vector<dcomplex> z(nz);
  for (long i = 0; i < nz; ++i) {
    const double x = uniform(gen, -3.5, 3.5), y = uniform(gen, 0.02, 0.5);
    z[i] = dcomplex(x, (i % 2 ? -1.0 : 1.0) * y);
  }
  return z;
}

} // namespace

// =======================================================================
//  1. exact recovery of a pole model
// =======================================================================
TEST_CASE("cayley_exact_recovery", "[numerics][cayley]") {
  const double wp = 0.11;
  std::cout << std::scientific << std::setprecision(2);
  std::cout << "\n[cayley] exact recovery: n=2, 6 poles in [-2.6, 2.8] Ha, rank-1 residues, wp=0.11, tol_gram=1e-13\n"
            << "[cayley] seed  K  npoles r_gram r1  max|dd|  max|dd|/(1+(d^2+wp^2)/(2wp))  max|dR|  #extra(max|W|^2)  "
               "Sigma rel err (20 pts)  held-out\n";
  for (unsigned seed : {1u, 2u, 3u}) {
    auto m  = make_model(seed);
    auto C  = ldlr::moments_from_poles(m.E, m.R, wp, 13);
    double bc = std::abs(ldlr::moment_bound_violation(C));
    std::cout << "[cayley] seed " << seed << ": |bound_check - 1| = " << bc << "\n";
    CHECK(bc <= 1e-13);
    for (long K : {5L, 6L, 8L}) {
      auto res = ldlr::upfold_block(C, K, wp, 1e-12, 1e-13, 1e-12, 72);
      const long np = res.d.size();
      double dd = 0.0, dd_sc = 0.0, wextra = 0.0;
      long nextra = 0;
      nda::array<dcomplex, 3> Rrec(6, 2, 2);
      Rrec() = 0.0;
      std::vector<long> nmatch(6, 0);
      for (long l = 0; l < np; ++l) {
        long best = 0;
        for (long q = 1; q < 6; ++q)
          if (std::abs(res.d(l) - m.E(q)) < std::abs(res.d(l) - m.E(best))) best = q;
        const double dl = std::abs(res.d(l) - m.E(best));
        double w2 = std::norm(res.W(0, l)) + std::norm(res.W(1, l));
        if (dl > 1e-8) { ++nextra; wextra = std::max(wextra, w2); continue; }
        ++nmatch[best];
        dd    = std::max(dd, dl);
        dd_sc = std::max(dd_sc, dl / (1.0 + (m.E(best) * m.E(best) + wp * wp) / (2.0 * wp)));
        for (long i = 0; i < 2; ++i)
          for (long j = 0; j < 2; ++j) Rrec(best, i, j) += res.W(i, l) * std::conj(res.W(j, l));
      }
      double dR = 0.0;
      for (long q = 0; q < 6; ++q)
        for (long i = 0; i < 2; ++i)
          for (long j = 0; j < 2; ++j) dR = std::max(dR, std::abs(Rrec(q, i, j) - m.R(q, i, j)));
      double serr = 0.0, smax = 0.0;
      for (auto z : test_points(20, 7)) {
        auto Sx = ldlr::sigma_from_poles(m.E, m.V, z);
        auto Sr = ldlr::sigma_from_poles(res.d, res.W, z);
        serr    = std::max(serr, max_abs(nda::array<dcomplex, 2>(Sr - Sx)));
        smax    = std::max(smax, max_abs(Sx));
      }
      std::cout << "[cayley]  " << seed << "    " << K << "  " << np << "  " << res.r_gram << "  " << res.r1 << "  " << dd
                << "  " << dd_sc << "  " << dR << "  " << nextra << "(" << wextra << ")  " << serr / smax << "  "
                << res.residual << "\n";
      for (long q = 0; q < 6; ++q) CHECK(nmatch[q] >= 1);
      CHECK(dd_sc <= 1e-12);                     // pole error in the Cayley angle (scale-free), all K >= 5
      if (K >= 6) CHECK(dd <= 1e-12);            // absolute pole error (K = 5 is the minimal order, see report)
      CHECK((nextra == 0 or wextra < 1e-12));
      CHECK(dR <= 1e-10);
      CHECK(serr / smax <= 1e-11);
    }
  }
}

// =======================================================================
//  2. Lehmann G from the upfolded Hamiltonian
// =======================================================================
TEST_CASE("cayley_lehmann", "[numerics][cayley]") {
  const double wp = 0.11;
  auto m          = make_model(1);
  auto C          = ldlr::moments_from_poles(m.E, m.R, wp, 13);
  auto res        = ldlr::upfold_block(C, 8, wp, 1e-12, 1e-13, 1e-12, 72);
  nda::array<dcomplex, 2> H{{0.2, 0.1}, {0.1, -0.4}};
  auto L = ldlr::lehmann(H, res.d, res.W);
  const long M = L.e.size();
  // completeness: sum_m v_m v_m^dag = 1
  nda::array<dcomplex, 2> S(2, 2);
  S() = 0.0;
  for (long m_ = 0; m_ < M; ++m_)
    for (long i = 0; i < 2; ++i)
      for (long j = 0; j < 2; ++j) S(i, j) += L.v(i, m_) * std::conj(L.v(j, m_));
  for (long i = 0; i < 2; ++i) S(i, i) -= 1.0;
  const double compl_err = max_abs(S);
  bool sorted = std::is_sorted(L.e.begin(), L.e.end());
  // G(z) = sum_m v v^dag/(z - e_m) vs [z - H - Sigma_exact(z)]^{-1}
  double gerr = 0.0, gmax = 0.0;
  for (auto z : test_points(10, 11)) {
    auto Gx = ldlr::greens_function(H, m.E, m.V, z);
    nda::array<dcomplex, 2> Gl(2, 2);
    Gl() = 0.0;
    for (long m_ = 0; m_ < M; ++m_)
      for (long i = 0; i < 2; ++i)
        for (long j = 0; j < 2; ++j) Gl(i, j) += L.v(i, m_) * std::conj(L.v(j, m_)) / (z - L.e(m_));
    gerr = std::max(gerr, max_abs(nda::array<dcomplex, 2>(Gl - Gx)));
    gmax = std::max(gmax, max_abs(Gx));
  }
  std::cout << std::scientific << std::setprecision(2) << "\n[cayley] Lehmann (K=8): M = " << M
            << " poles, sorted " << sorted << ", completeness |sum v v^dag - 1| = " << compl_err
            << ", max|G_Lehmann - G_Dyson(exact Sigma)| = " << gerr << " (max|G| " << gmax << ")\n";
  CHECK(M == 2 + res.d.size());
  CHECK(sorted);
  CHECK(compl_err <= 1e-12);
  CHECK(gerr <= 1e-10);
}

// =======================================================================
//  3. Si 2x2x2 G0W0 Gamma: spectra from exact moments
// =======================================================================
TEST_CASE("cayley_si_gamma", "[numerics][cayley]") {
  h5::file file(ref_file, 'r');
  h5::group root(file);
  auto g = root.open_group("si_k0");
  const double mu = read_attr<double>(g, "mu"), wp = read_attr<double>(g, "wp"), tol_gram = read_attr<double>(g, "tol_gram");
  const double eta = read_attr<double>(g, "eta"), hw = read_attr<double>(g, "win_halfwidth");
  nda::array<double, 3> Cre, Cim;
  nda::array<double, 2> Hre, Him;
  nda::array<double, 1> om, Aex;
  nda::h5_read(g, "C_re", Cre);
  nda::h5_read(g, "C_im", Cim);
  nda::h5_read(g, "H_re", Hre);
  nda::h5_read(g, "H_im", Him);
  nda::h5_read(g, "om", om);
  nda::h5_read(g, "Aex_tr_eta0.01", Aex);
  const long nm = Cre.extent(0), n = Cre.extent(1), nw = om.size();
  nda::array<dcomplex, 3> C(nm, n, n);
  for (long a = 0; a < nm; ++a)
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) C(a, i, j) = dcomplex(Cre(a, i, j), Cim(a, i, j));
  nda::array<dcomplex, 2> Hrel(n, n);                                // mu-relative
  for (long i = 0; i < n; ++i)
    for (long j = 0; j < n; ++j) Hrel(i, j) = dcomplex(Hre(i, j), Him(i, j)) - (i == j ? mu : 0.0);
  nda::array<double, 1> omr(nw);
  for (long i = 0; i < nw; ++i) omr(i) = om(i) - mu;
  auto win_err = [&](nda::array<double, 1> const &trA) {          // metric of si222_g0w0_casida.py
    double num = 0.0, den = 0.0;
    for (long i = 0; i < nw; ++i)
      if (std::abs(om(i) - mu) < hw) {
        num = std::max(num, std::abs(trA(i) - Aex(i)));
        den = std::max(den, Aex(i));
      }
    return num / den;
  };
  auto maxdiff = [&](nda::array<double, 1> const &a, nda::array<double, 1> const &b, bool window) {
    double x = 0.0;
    for (long i = 0; i < nw; ++i)
      if (not window or std::abs(om(i) - mu) < hw) x = std::max(x, std::abs(a(i) - b(i)));
    return x;
  };
  double bc = ldlr::bound_check(C);
  std::cout << std::scientific << std::setprecision(3) << "\n[cayley] Si222 G0W0 k=0, nb=" << n << ", wp=" << wp
            << ", mu=" << mu << ", tol_gram=" << tol_gram << ", eta=" << eta << ", window |w-mu|<" << hw
            << "; bound_check(C^(0..17)) = " << bc << "\n"
            << "[cayley] K nphi | npoles(py) r_gram r1 n_free | phi(C++) phi(py) | residual(C++) residual(py) | "
               "win err C++  py  npz | max|A-A_py| win/all  max|A-A_npz| win/all | time\n";
  for (long K : {8L, 16L}) {
    nda::array<double, 1> Anpz;
    nda::h5_read(g, "A_total_K" + std::to_string(K) + "_eta0.01", Anpz);
    const double e_npz = read_attr<double>(g, "K" + std::to_string(K) + "_win_err_stored");
    for (long nphi : {72L, 8L}) {
      const std::string p = "K" + std::to_string(K) + "_nphi" + std::to_string(nphi) + "_";
      nda::array<double, 1> Apy;
      nda::h5_read(g, "A_py_" + p.substr(0, p.size() - 1) + "_eta0.01", Apy);
      auto t0  = std::chrono::steady_clock::now();
      auto res = ldlr::upfold_block(C, K, wp, 1e-12, tol_gram, 1e-12, nphi);
      auto trA = ldlr::spectral_trace(Hrel, res.d, res.W, omr, eta);
      const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      const double e = win_err(trA), e_py = read_attr<double>(g, p + "win_err");
      std::cout << "[cayley] " << K << " " << nphi << " | " << res.d.size() << "(" << read_attr<long>(g, p + "npoles") << ") "
                << res.r_gram << " " << res.r1 << " " << res.n_free << " | " << std::setprecision(8) << res.phi << " "
                << read_attr<double>(g, p + "phi") << std::setprecision(3) << " | " << res.residual << " "
                << read_attr<double>(g, p + "residual") << " | " << e << " " << e_py << " " << e_npz << " | "
                << maxdiff(trA, Apy, true) << "/" << maxdiff(trA, Apy, false) << "  " << maxdiff(trA, Anpz, true) << "/"
                << maxdiff(trA, Anpz, false) << " | " << std::setprecision(1) << std::fixed << secs << "s"
                << std::scientific << std::setprecision(3) << "\n";
      CHECK(res.d.size() == read_attr<long>(g, p + "npoles"));
      CHECK(e <= 1.2 * e_py);
      CHECK(e <= 1.2 * (K == 8 ? 1.7e-2 : 4.0e-4));   // published python numbers (notes section 6)
    }
  }
}

// =======================================================================
//  3b. (S7f, hidden) the same demo with the alternative cuts of upfold_opts_t (gap / smooth Gram cut, gap SVD cut,
//      tol_svd 1e-8, tol_gram 1e-10 / 1e-8 instead of the file's 1e-13): window error and held-out residual vs the python
//      algorithm ("hard"), nphi 8 (report only)
// =======================================================================
TEST_CASE("cayley_si_gamma_cuts", "[.cayley_cuts]") {
  h5::file file(ref_file, 'r');
  h5::group root(file);
  auto g = root.open_group("si_k0");
  const double mu = read_attr<double>(g, "mu"), wp = read_attr<double>(g, "wp"), tol_gram = read_attr<double>(g, "tol_gram");
  const double eta = read_attr<double>(g, "eta"), hw = read_attr<double>(g, "win_halfwidth");
  nda::array<double, 3> Cre, Cim;
  nda::array<double, 2> Hre, Him;
  nda::array<double, 1> om, Aex;
  nda::h5_read(g, "C_re", Cre);
  nda::h5_read(g, "C_im", Cim);
  nda::h5_read(g, "H_re", Hre);
  nda::h5_read(g, "H_im", Him);
  nda::h5_read(g, "om", om);
  nda::h5_read(g, "Aex_tr_eta0.01", Aex);
  const long nm = Cre.extent(0), n = Cre.extent(1), nw = om.size();
  nda::array<dcomplex, 3> C(nm, n, n);
  for (long a = 0; a < nm; ++a)
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) C(a, i, j) = dcomplex(Cre(a, i, j), Cim(a, i, j));
  nda::array<dcomplex, 2> Hrel(n, n);
  for (long i = 0; i < n; ++i)
    for (long j = 0; j < n; ++j) Hrel(i, j) = dcomplex(Hre(i, j), Him(i, j)) - (i == j ? mu : 0.0);
  nda::array<double, 1> omr(nw);
  for (long i = 0; i < nw; ++i) omr(i) = om(i) - mu;
  auto win_err = [&](nda::array<double, 1> const &trA) {
    double num = 0.0, den = 0.0;
    for (long i = 0; i < nw; ++i)
      if (std::abs(om(i) - mu) < hw) {
        num = std::max(num, std::abs(trA(i) - Aex(i)));
        den = std::max(den, Aex(i));
      }
    return num / den;
  };
  std::cout << std::scientific << std::setprecision(3) << "\n[cayley_cuts] Si222 G0W0 k=0, tol_gram=" << tol_gram << ", nphi 8\n"
            << "[cayley_cuts] K cut           | r_gram r1 | residual | win err\n";
  for (long K : {8L, 16L}) {
    double e_hard = 0.0;
    for (std::string v : {"hard", "gap", "smooth", "gap+svdgap", "tol_svd 1e-8", "tol_gram 1e-10", "tol_gram 1e-8"}) {
      ldlr::upfold_opts_t o;
      o.tol_gram = tol_gram;
      o.nphi     = 8;
      if (v == "gap" or v == "gap+svdgap") o.gram_cut = "gap";
      if (v == "smooth") o.gram_cut = "smooth";
      if (v == "gap+svdgap") o.svd_cut = "gap";
      if (v == "tol_svd 1e-8") o.tol_svd = 1e-8;
      if (v == "tol_gram 1e-10") o.tol_gram = 1e-10;
      if (v == "tol_gram 1e-8") o.tol_gram = 1e-8;
      auto res = ldlr::upfold_block(C, K, wp, o);
      auto trA = ldlr::spectral_trace(Hrel, res.d, res.W, omr, eta);
      const double e = win_err(trA);
      if (v == "hard") e_hard = e;
      std::cout << "[cayley_cuts] " << K << " " << std::setw(14) << v << " | " << res.r_gram << " " << res.r1 << " | " << res.residual
                << " | " << e << " (" << std::fixed << std::setprecision(2) << e / e_hard << " x hard)" << std::scientific
                << std::setprecision(3) << "\n";
    }
  }
}

// =======================================================================
//  4. chemical potential: widest admissible QP gap
// =======================================================================
TEST_CASE("cayley_chemical_potential", "[numerics][cayley]") {
  h5::file file(ref_file, 'r');
  h5::group root(file);
  auto g = root.open_group("mu_toy");
  std::vector<nda::array<double, 1>> e_k(2);
  std::vector<nda::array<dcomplex, 2>> v_k(2);
  for (int k = 0; k < 2; ++k) {
    nda::array<double, 2> vr;
    nda::h5_read(g, "e_k" + std::to_string(k), e_k[k]);
    nda::h5_read(g, "v_k" + std::to_string(k) + "_re", vr);
    v_k[k] = nda::array<dcomplex, 2>(vr);
  }
  nda::array<double, 1> kw;
  nda::h5_read(g, "k_weight", kw);
  std::vector<double> kwv(kw.begin(), kw.end());
  auto r = ldlr::chemical_potential(e_k, v_k, read_attr<double>(g, "nelec"), kwv, read_attr<double>(g, "qp_weight"),
                                    read_attr<double>(g, "ntol"));
  const double mu_py = read_attr<double>(g, "mu"), eh = read_attr<double>(g, "e_homo"), el = read_attr<double>(g, "e_lumo"),
               N_py = read_attr<double>(g, "N");
  std::cout << std::setprecision(10) << std::fixed << "\n[cayley] mu toy: C++ mu " << r.mu << " homo " << r.e_homo << " lumo "
            << r.e_lumo << " gap " << r.gap << " N " << r.N << " | python mu " << mu_py << " homo " << eh << " lumo " << el
            << " N " << N_py << std::scientific << "\n";
  CHECK(std::abs(r.mu - mu_py) <= 1e-14);
  CHECK(std::abs(r.e_homo - eh) <= 1e-14);
  CHECK(std::abs(r.e_lumo - el) <= 1e-14);
  CHECK(std::abs(r.N - N_py) <= 1e-12);
  CHECK(r.gap > 0.4);                                   // the wide gap, not the 1.5e-5 Ha noise split (which has better N)
}

// =======================================================================
//  5. (S7g, hidden) closure profile at production size (closure_bench.hpp); host LAPACK only here, the device variants
//     run from test_gw_line_device "[.closure_bench_dev]"
// =======================================================================
TEST_CASE("cayley_closure_bench", "[.closure_bench]") { closure_bench::run(nullptr, nullptr); }

// S7g: the Hermitian ("cayley") eigenvectors of U and the divide-and-conquer SVD vs the python path (schur + gesvd)
TEST_CASE("cayley_ueig_ab", "[numerics][cayley]") {
  CHECK(closure_bench::ab_small(nullptr, "gesvd") <= 1e-11);
  CHECK(closure_bench::ab_small(nullptr, "gesdd") <= 1e-11);
  // n_free > 0 (terminal-phase scan): the fast SVD driver hands over to zgesvd (driver-dependent free block)
  CHECK(closure_bench::ab_small(nullptr, "gesdd", 160) <= 1e-9);
}

} // namespace bdft_tests
