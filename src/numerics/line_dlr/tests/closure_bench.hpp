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

#ifndef COQUI_NUMERICS_LINE_DLR_TESTS_CLOSURE_BENCH_HPP
#define COQUI_NUMERICS_LINE_DLR_TESTS_CLOSURE_BENCH_HPP

/**
 * S7g: profile of one closure (upfold_block + lehmann) at production size, shared by test_cayley "[.closure_bench]" (host)
 * and test_gw_line_device "[.closure_bench_dev]" (cuSOLVER hooks). Synthetic real-pole measure: n = 58 orbitals, P poles
 * log-spaced in +-[0.02, 6] Ha with rank-1 residues, K = 24 (block Toeplitz dimension 1450), wp 0.11, tol_gram 1e-10.
 * Env: CAYLEY_BENCH_THREADS (BLAS threads, list, default "1"), CAYLEY_BENCH_P (1250), CAYLEY_BENCH_N (58), CAYLEY_BENCH_K
 * (24), CAYLEY_BENCH_VARIANTS (list of "<ueig>/<svd>[/dev|/devh]", ueig in {schur, cayley}, svd in {gesvd, gesdd,
 * gesvdp}; "/dev" = the device hooks (gesvdp only there), "/devh" = the device hooks except in the Cayley U-eigen path). Accuracy columns are relative to the FIRST variant at the same thread
 * count (default the python path schur/gesvd): max |dSigma| / max |Sigma| and max |dG| / max |G| (Lehmann G with a random
 * Hermitian H) at 40 points z = x + i y, x in [-1, 1], y in [0.005, 0.1] Ha. CAYLEY_BENCH_KCONC = m > 1: also m copies of
 * each closure on m concurrent threads with threads / m BLAS threads each (wall and wall / m per k; MKL only).
 */

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <dlfcn.h>

#include "catch2/catch.hpp"
#include "nda/nda.hpp"
#include "numerics/line_dlr/cayley.hpp"
#include "utilities/blas_threads.hpp"

namespace closure_bench {

namespace ldlr = numerics::line_dlr;
using dcomplex = std::complex<double>;

inline double unif(std::mt19937 &gen, double a, double b) { return a + (b - a) * (double(gen()) / 4294967296.0); }

inline double max_abs2(nda::array<dcomplex, 2> const &A) {
  double x = 0.0;
  for (long i = 0; i < A.extent(0); ++i)
    for (long j = 0; j < A.extent(1); ++j) x = std::max(x, std::abs(A(i, j)));
  return x;
}

inline std::vector<std::string> split(std::string t) {
  for (auto &c : t)
    if (c == ',') c = ' ';
  std::istringstream is(t);
  std::vector<std::string> out;
  std::string v;
  while (is >> v) out.push_back(v);
  return out;
}

/// hooks_svd: device hooks with the QR-iteration SVD; hooks_svdp: with the polar-decomposition SVD (both may be null)
inline void run(ldlr::lapack_hooks_t const *hooks_svd, ldlr::lapack_hooks_t const *hooks_svdp) {
  auto env_l   = [](char const *nm, long d) { auto *e = std::getenv(nm); return e ? std::atol(e) : d; };
  const long n = env_l("CAYLEY_BENCH_N", 58), P = env_l("CAYLEY_BENCH_P", 1250), K = env_l("CAYLEY_BENCH_K", 24);
  std::vector<long> threads;
  for (auto const &t : split(std::getenv("CAYLEY_BENCH_THREADS") ? std::getenv("CAYLEY_BENCH_THREADS") : "1"))
    threads.push_back(std::atol(t.c_str()));
  std::vector<std::string> variants{"schur/gesvd", "schur/gesdd", "cayley/gesvd", "cayley/gesdd"};
  if (hooks_svd)
    variants = {"schur/gesvd", "cayley/gesdd", "cayley/gesvd/dev", "cayley/gesvdp/dev", "cayley/gesvd/devh", "cayley/gesvdp/devh"};
  if (auto *e = std::getenv("CAYLEY_BENCH_VARIANTS")) variants = split(e);
  const double wp = 0.11;
  std::mt19937 gen(12345);
  nda::array<double, 1> w(P);
  nda::array<dcomplex, 3> g(P, n, n);
  for (long l = 0; l < P; ++l) {
    const double a = 0.02 * std::pow(6.0 / 0.02, unif(gen, 0.0, 1.0));
    w(l)           = (l % 2 ? a : -a);
    std::vector<dcomplex> v(n);
    for (auto &x : v) x = dcomplex(unif(gen, -1.0, 1.0), unif(gen, -1.0, 1.0)) / std::sqrt(double(P));
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) g(l, i, j) = v[i] * std::conj(v[j]);
  }
  nda::array<dcomplex, 2> H(n, n);
  for (long i = 0; i < n; ++i)
    for (long j = 0; j <= i; ++j) {
      H(i, j) = (i == j) ? dcomplex(unif(gen, -0.5, 0.5), 0.0) : 0.02 * dcomplex(unif(gen, -1.0, 1.0), unif(gen, -1.0, 1.0));
      H(j, i) = std::conj(H(i, j));
    }
  std::vector<dcomplex> zs(40);
  for (auto &z : zs) z = dcomplex(unif(gen, -1.0, 1.0), unif(gen, 0.005, 0.1));
  auto t0 = std::chrono::steady_clock::now();
  auto C  = ldlr::moments_from_poles(w, g, wp, K + 1);
  const double t_mom = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::cout << std::scientific << std::setprecision(2) << "\n[closure_bench] n " << n << ", P " << P << " poles, K " << K
            << ", block Toeplitz dimension " << (K + 1) * n << ", moments " << t_mom << " s\n"
            << "[closure_bench] thr variant           | C0    Gram   SVD    U-eig  Lehm   total (s) | r_gram r1 n_free nreal | "
               "nflag fallback res | dSigma dG (vs the first variant, same threads) held-out | dSigma dG vs the first run\n";
  std::vector<nda::array<dcomplex, 2>> Sg, Gg;   // the first variant at the first thread count
  for (long nt : threads) {
    utils::apply_blas_threads(nt);
    std::vector<nda::array<dcomplex, 2>> S0, G0;
    for (auto const &v : variants) {
      auto parts = split([&] { std::string t = v; for (auto &c : t) if (c == '/') c = ' '; return t; }());
      REQUIRE(parts.size() >= 2);
      const bool dev = (parts.size() > 2 and (parts[2] == "dev" or parts[2] == "devh"));
      if (dev and not hooks_svd) continue;
      ldlr::lapack_hooks_t hloc;   // "devh": device hooks except in the U-eigen (Cayley) path
      if (dev and parts[2] == "devh") {
        hloc         = *(parts[1] == "gesvdp" ? hooks_svdp : hooks_svd);
        hloc.in_ueig = false;
      }
      ldlr::upfold_opts_t o;
      o.tol_gram   = 1e-10;
      o.ueig       = parts[0];
      o.svd_driver = (parts[1] == "gesvdp") ? "gesvd" : parts[1];
      o.hooks      = dev ? (parts[2] == "devh" ? &hloc : (parts[1] == "gesvdp" ? hooks_svdp : hooks_svd)) : nullptr;
      auto ta      = std::chrono::steady_clock::now();
      auto up      = ldlr::upfold_block(C, K, wp, o);
      auto tb      = std::chrono::steady_clock::now();
      auto L       = ldlr::lehmann(H, up.d, up.W, o.hooks);
      auto tc      = std::chrono::steady_clock::now();
      const double t_up = std::chrono::duration<double>(tb - ta).count(), t_l = std::chrono::duration<double>(tc - tb).count();
      std::vector<nda::array<dcomplex, 2>> S, G;
      for (auto z : zs) {
        S.push_back(ldlr::sigma_from_poles(up.d, up.W, z));
        nda::array<dcomplex, 2> Gz(n, n);
        Gz() = 0.0;
        for (long m = 0; m < L.e.size(); ++m) {
          const dcomplex k = 1.0 / (z - L.e(m));
          for (long i = 0; i < n; ++i)
            for (long j = 0; j < n; ++j) Gz(i, j) += L.v(i, m) * k * std::conj(L.v(j, m));
        }
        G.push_back(Gz);
      }
      double dSg = 0.0, dGg = 0.0;
      if (Sg.empty()) {
        Sg = S;
        Gg = G;
      } else {
        double sm = 0.0, gm = 0.0;
        for (size_t q = 0; q < zs.size(); ++q) {
          dSg = std::max(dSg, max_abs2(nda::array<dcomplex, 2>(S[q] - Sg[q])));
          dGg = std::max(dGg, max_abs2(nda::array<dcomplex, 2>(G[q] - Gg[q])));
          sm  = std::max(sm, max_abs2(Sg[q]));
          gm  = std::max(gm, max_abs2(Gg[q]));
        }
        dSg /= sm;
        dGg /= gm;
      }
      double dS = 0.0, dG = 0.0;
      if (S0.empty()) {
        S0 = S;
        G0 = G;
      } else {
        double sm = 0.0, gm = 0.0;
        for (size_t q = 0; q < zs.size(); ++q) {
          dS = std::max(dS, max_abs2(nda::array<dcomplex, 2>(S[q] - S0[q])));
          dG = std::max(dG, max_abs2(nda::array<dcomplex, 2>(G[q] - G0[q])));
          sm = std::max(sm, max_abs2(S0[q]));
          gm = std::max(gm, max_abs2(G0[q]));
        }
        dS /= sm;
        dG /= gm;
      }
      std::cout << std::fixed << std::setprecision(3) << "[closure_bench] " << std::setw(3) << nt << " " << std::setw(17) << v
                << " | " << up.t_c0 << " " << up.t_gram << " " << up.t_svd << " " << up.t_ueig << " " << t_l << " "
                << t_up + t_l << " | " << up.r_gram << " " << up.r1 << " " << up.n_free << " " << up.n_realize << " | "
                << up.ueig_nflag << " " << up.ueig_fallback << " " << std::scientific << std::setprecision(1) << up.ueig_res
                << " | " << dS << " " << dG << " " << up.residual << " | vs first: " << dSg << " " << dGg << std::endl;
      CHECK(up.ueig_fallback == 0);
      // concurrent k (closure_k_workers): nkc copies of this closure on nkc std::threads with nt / nkc BLAS threads each
      const long nkc = env_l("CAYLEY_BENCH_KCONC", 0);
      if (nkc > 1) {
        const int bt = int(std::max(1L, nt / nkc));
        auto tw0     = std::chrono::steady_clock::now();
        std::vector<std::thread> th;
        for (long w = 0; w < nkc; ++w)
          th.emplace_back([&]() {
            using set_local_t = int (*)(int);
            if (auto f = reinterpret_cast<set_local_t>(dlsym(RTLD_DEFAULT, "MKL_Set_Num_Threads_Local"))) f(bt);
            auto u2 = ldlr::upfold_block(C, K, wp, o);
            auto L2 = ldlr::lehmann(H, u2.d, u2.W, o.hooks);
          });
        for (auto &t : th) t.join();
        const double tw = std::chrono::duration<double>(std::chrono::steady_clock::now() - tw0).count();
        std::cout << std::fixed << std::setprecision(3) << "[closure_bench] " << std::setw(3) << nt << " " << std::setw(17) << v
                  << " | " << nkc << " concurrent k x " << bt << " BLAS threads: wall " << tw << " s, " << tw / double(nkc)
                  << " s per k" << std::scientific << std::endl;
      }
    }
  }
  utils::apply_blas_threads(1);
}

/**
 * Accuracy of the individual hooks on random matrices of dimension n (backward errors, relative): heevd
 * |A V - V diag(w)| / |A| and |V^dag V - 1|, gesvd |A - P diag(s) Qh| / |A| and |P^dag P - 1|, lu_solve |A X - B| / (|A||X|);
 * the same for the host drivers. Returns the max over the hooks of the backward errors.
 */
inline double hooks_accuracy(ldlr::lapack_hooks_t const *h, long n) {
  using ldlr::cmatrix_F;
  std::mt19937 gen(4242);
  cmatrix_F A(n, n), Hm(n, n), B(n, 3);
  for (long j = 0; j < n; ++j)
    for (long i = 0; i < n; ++i) A(i, j) = dcomplex(unif(gen, -1.0, 1.0), unif(gen, -1.0, 1.0));
  for (long j = 0; j < n; ++j)
    for (long i = 0; i < n; ++i) Hm(i, j) = 0.5 * (A(i, j) + std::conj(A(j, i)));
  for (long j = 0; j < 3; ++j)
    for (long i = 0; i < n; ++i) B(i, j) = dcomplex(unif(gen, -1.0, 1.0), 0.0);
  auto fro = [](cmatrix_F const &M) { double x = 0.0; for (auto const &v : M) x += std::norm(v); return std::sqrt(x); };
  auto ortho = [&](cmatrix_F const &V) {
    auto G = ldlr::detail::mm(V, V, 'C', 'N');
    for (long i = 0; i < G.extent(0); ++i) G(i, i) -= 1.0;
    return fro(G);
  };
  double worst = 0.0;
  for (int pass = 0; pass < 2; ++pass) {   // 0: host, 1: hooks
    if (pass == 1 and not h) break;
    // heevd
    cmatrix_F V(Hm);
    nda::array<double, 1> w(n);
    bool ok = true;
    if (pass == 0) w = ldlr::detail::herm_eig(V);
    else ok = h->heevd(V, w);
    cmatrix_F R = ldlr::detail::mm(Hm, V);
    for (long j = 0; j < n; ++j)
      for (long i = 0; i < n; ++i) R(i, j) -= V(i, j) * w(j);
    const double e_he = fro(R) / fro(Hm), o_he = ortho(V);
    // gesvd
    cmatrix_F M(A), P, Qh;
    nda::array<double, 1> sv;
    if (pass == 0) ldlr::detail::svd_dc(M, sv, P, Qh);
    else ok = ok and h->gesvd(M, sv, P, Qh);
    cmatrix_F PS(P);
    for (long j = 0; j < n; ++j)
      for (long i = 0; i < n; ++i) PS(i, j) *= sv(j);
    cmatrix_F R2 = ldlr::detail::mm(PS, Qh);
    R2 -= A;
    const double e_sv = fro(R2) / fro(A), o_sv = std::max(ortho(P), ortho(Qh));
    // lu_solve
    cmatrix_F A2(A), X(B);
    double e_lu = 0.0;
    if (pass == 1) {
      ok = ok and h->lu_solve(A2, X);
      cmatrix_F R3 = ldlr::detail::mm(A, X);
      R3 -= B;
      e_lu = fro(R3) / (fro(A) * fro(X));
    }
    std::cout << std::scientific << std::setprecision(2) << "[hooks_accuracy] " << (pass ? "device" : "host  ") << " n " << n
              << " ok " << ok << " | heevd res " << e_he << " ortho " << o_he << " | gesvd res " << e_sv << " ortho " << o_sv
              << " | lu res " << e_lu << std::endl;
    if (pass == 1) worst = std::max({worst, e_he, o_he, e_sv, o_sv, e_lu, ok ? 0.0 : 1.0});
  }
  return worst;
}

/**
 * Small A/B (not hidden): n 8, P poles (default 80), K 12 -- the "cayley" U-eigen path and the given hooks (min_dim lowered to 0, so
 * every dense problem goes through them) against the python path (schur + gesvd, host). Returns max(dSigma, dG) relative.
 */
inline double ab_small(ldlr::lapack_hooks_t const *hooks, std::string const &svd = "gesvd", long P = 80) {
  const long n = 8, K = 12;   // P < (K+1) n: exact moments, n_free = 0 (the production case); P > 104: n_free > 0 (phase scan)
  const double wp = 0.11;
  std::mt19937 gen(777);
  nda::array<double, 1> w(P);
  nda::array<dcomplex, 3> g(P, n, n);
  for (long l = 0; l < P; ++l) {
    const double a = 0.02 * std::pow(6.0 / 0.02, unif(gen, 0.0, 1.0));
    w(l)           = (l % 2 ? a : -a);
    std::vector<dcomplex> v(n);
    for (auto &x : v) x = dcomplex(unif(gen, -1.0, 1.0), unif(gen, -1.0, 1.0)) / std::sqrt(double(P));
    for (long i = 0; i < n; ++i)
      for (long j = 0; j < n; ++j) g(l, i, j) = v[i] * std::conj(v[j]);
  }
  nda::array<dcomplex, 2> H(n, n);
  H() = 0.0;
  for (long i = 0; i < n; ++i) H(i, i) = unif(gen, -0.5, 0.5);
  auto C = ldlr::moments_from_poles(w, g, wp, K + 1);
  ldlr::lapack_hooks_t hk;
  if (hooks) {
    hk         = *hooks;
    hk.min_dim = 0;
  }
  ldlr::upfold_opts_t o0, o1;
  o0.tol_gram = o1.tol_gram = 1e-10;
  o1.ueig       = "cayley";
  o1.svd_driver = svd;
  o1.hooks      = hooks ? &hk : nullptr;
  auto u0 = ldlr::upfold_block(C, K, wp, o0);
  auto u1 = ldlr::upfold_block(C, K, wp, o1);
  auto L0 = ldlr::lehmann(H, u0.d, u0.W);
  auto L1 = ldlr::lehmann(H, u1.d, u1.W, o1.hooks);
  double dS = 0.0, sm = 0.0, dG = 0.0, gm = 0.0;
  for (int q = 0; q < 20; ++q) {
    const dcomplex z(unif(gen, -1.0, 1.0), unif(gen, 0.005, 0.1));
    auto S0 = ldlr::sigma_from_poles(u0.d, u0.W, z), S1 = ldlr::sigma_from_poles(u1.d, u1.W, z);
    dS      = std::max(dS, max_abs2(nda::array<dcomplex, 2>(S1 - S0)));
    sm      = std::max(sm, max_abs2(S0));
    auto lg = [&](ldlr::lehmann_t const &L) {
      nda::array<dcomplex, 2> G(n, n);
      G() = 0.0;
      for (long m = 0; m < L.e.size(); ++m)
        for (long i = 0; i < n; ++i)
          for (long j = 0; j < n; ++j) G(i, j) += L.v(i, m) * std::conj(L.v(j, m)) / (z - L.e(m));
      return G;
    };
    auto G0 = lg(L0), G1 = lg(L1);
    dG      = std::max(dG, max_abs2(nda::array<dcomplex, 2>(G1 - G0)));
    gm      = std::max(gm, max_abs2(G0));
  }
  std::cout << std::scientific << std::setprecision(2) << "[closure_ab] n " << n << " P " << P << " K " << K << " r_gram "
            << u0.r_gram << "/" << u1.r_gram << " n_free " << u0.n_free << " fallback " << u1.ueig_fallback << " nflag "
            << u1.ueig_nflag << " | dSigma " << dS / sm << " dG " << dG / gm << " | held-out " << u0.residual << " / "
            << u1.residual << (hooks ? " (hooks)" : " (host)") << std::endl;
  CHECK(u1.ueig_fallback == 0);
  CHECK(u0.r_gram == u1.r_gram);
  return std::max(dS / sm, dG / gm);
}

} // namespace closure_bench

#endif
