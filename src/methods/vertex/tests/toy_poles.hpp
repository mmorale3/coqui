/**
 * Shared exact-toy helpers for the scGW-tilde vertex tests (Tier 1.5 and the dynamic BSE):
 *   dense_fsum / richardson_fsum -- dense fermionic Matsubara sums with Richardson tail elimination;
 *   toy_k / make_toy              -- an EXACT rational G and Sigma_c from a Hermitian Schur embedding
 *                                    H_big = [[h, B_p], [B_p^dag, e_p 1]]: G(z) = P (z - H_big)^-1 P^dag
 *                                    is a pole sum on the eigenvalues with PSD residues, Sigma_c(z) =
 *                                    sum_p B_p B_p^dag/(z - e_p), and G^-1 = z - h - Sigma_c EXACTLY;
 *   toy_set / assemble            -- several toys (one per k) on the union node set.
 * Header-only, test namespace; not part of the library.
 */
#ifndef COQUI_VERTEX_TESTS_TOY_POLES_HPP
#define COQUI_VERTEX_TESTS_TOY_POLES_HPP

#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include "catch2/catch.hpp"
#include "nda/nda.hpp"
#include "nda/linalg/eigenelements.hpp"
#include "numerics/nda_functions.hpp"
#include "configuration.hpp"

namespace bdft_tests {

  using cplx = ComplexType;
  static const cplx I_(0.0, 1.0);
  static auto const all_ = nda::range::all;

  // ---------------- dense fermionic Matsubara oracle ----------------------------------
  // (1/beta) sum_{n=-M}^{M-1} h(i(2n+1)pi/beta); the tail is a power series in 1/M, removed
  // by Richardson: `levels` = 3 leaves O(1/M^3), 4 leaves O(1/M^4).
  template<class F>
  cplx dense_fsum(double beta, long M, F &&h) {
    cplx s(0.0);
    for (long n = -M; n < M; ++n) s += h(I_ * cplx((2.0 * double(n) + 1.0) * M_PI / beta));
    return s / beta;
  }

  template<class F>
  cplx richardson_fsum(double beta, long M, int levels, F &&h) {
    std::vector<cplx> s(static_cast<size_t>(levels));
    for (int k = 0; k < levels; ++k) s[size_t(k)] = dense_fsum(beta, M << k, h);
    // Richardson table: kill 1/M, then 1/M^2, then 1/M^3, ...
    for (int order = 1; order < levels; ++order) {
      const double f = std::pow(2.0, order);
      for (int k = 0; k + order < levels; ++k)
        s[size_t(k)] = (f * s[size_t(k + 1)] - s[size_t(k)]) / (f - 1.0);
    }
    return s[0];
  }

  // ---------------- the exact rational toy ---------------------------------------------
  struct toy_k {
    long nb = 0, np = 0, ng = 0;
    nda::array<cplx, 2> h;         // (nb, nb)
    nda::array<double, 1> epsS;    // (np)
    nda::array<cplx, 3> R;         // (np, nb, nb)  PSD
    nda::array<double, 1> lam;     // (ng)          poles of G
    nda::array<cplx, 3> g;         // (ng, nb, nb)  PSD residues, sum = 1

    nda::array<cplx, 2> Sigma(cplx z) const {
      nda::array<cplx, 2> S(nb, nb);
      S() = cplx(0.0);
      for (long p = 0; p < np; ++p) S += R(p, all_, all_) / (z - epsS(p));
      return S;
    }
    nda::array<cplx, 2> G(cplx z) const {
      nda::array<cplx, 2> Gz(nb, nb);
      Gz() = cplx(0.0);
      for (long j = 0; j < ng; ++j) Gz += g(j, all_, all_) / (z - lam(j));
      return Gz;
    }
    // Lambda0(z; inu) - 1 = sum_p R_p / ((z + inu - e_p)(z - e_p))   (eq 28; double pole at inu = 0)
    nda::array<cplx, 2> Lam1(cplx z, cplx inu) const {
      nda::array<cplx, 2> L(nb, nb);
      L() = cplx(0.0);
      for (long p = 0; p < np; ++p) L += R(p, all_, all_) / ((z + inu - epsS(p)) * (z - epsS(p)));
      return L;
    }
  };

  toy_k make_toy(long nb, std::vector<double> const &epsS, std::vector<double> const &rscale,
                 unsigned seed) {
    toy_k t;
    t.nb = nb;
    t.np = long(epsS.size());
    t.ng = nb * (t.np + 1);
    std::mt19937 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    auto rnd = [&]() { return cplx(nd(gen), nd(gen)); };

    t.h = nda::array<cplx, 2>(nb, nb);
    {
      nda::array<cplx, 2> A(nb, nb);
      for (auto &v : A) v = 0.5 * rnd();
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) t.h(i, j) = 0.5 * (A(i, j) + std::conj(A(j, i)));
    }
    t.epsS = nda::array<double, 1>(t.np);
    for (long p = 0; p < t.np; ++p) t.epsS(p) = epsS[size_t(p)];

    const long N = t.ng;
    nda::matrix<cplx> Hb(N, N);
    Hb() = cplx(0.0);
    for (long i = 0; i < nb; ++i)
      for (long j = 0; j < nb; ++j) Hb(i, j) = t.h(i, j);
    t.R = nda::array<cplx, 3>(t.np, nb, nb);
    for (long p = 0; p < t.np; ++p) {
      nda::array<cplx, 2> B(nb, nb);
      for (auto &v : B) v = rscale[size_t(p)] * rnd();
      const long o = nb * (p + 1);
      for (long i = 0; i < nb; ++i) {
        for (long j = 0; j < nb; ++j) {
          Hb(i, o + j) = B(i, j);
          Hb(o + j, i) = std::conj(B(i, j));
        }
        Hb(o + i, o + i) = cplx(epsS[size_t(p)]);
      }
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) {
          cplx s(0.0);
          for (long m = 0; m < nb; ++m) s += B(i, m) * std::conj(B(j, m));
          t.R(p, i, j) = s;
        }
    }
    auto [ev, U] = nda::linalg::eigenelements(Hb);
    // defensive: the eigenvectors are the COLUMNS of U (nda doc); verify on the toy itself
    {
      double resid = 0.0;
      for (long j = 0; j < N; ++j)
        for (long i = 0; i < N; ++i) {
          cplx s(0.0);
          for (long m = 0; m < N; ++m) s += Hb(i, m) * U(m, j);
          resid = std::max(resid, std::abs(s - ev(j) * U(i, j)));
        }
      REQUIRE(resid < 1e-12);
    }
    t.lam = nda::array<double, 1>(N);
    t.g = nda::array<cplx, 3>(N, nb, nb);
    for (long j = 0; j < N; ++j) {
      t.lam(j) = ev(j);
      for (long i = 0; i < nb; ++i)
        for (long k = 0; k < nb; ++k) t.g(j, i, k) = U(i, j) * std::conj(U(k, j));
    }
    // completeness and the Schur identity G(z) (z - h - Sigma(z)) = 1
    {
      nda::array<cplx, 2> s1(nb, nb);
      s1() = cplx(0.0);
      for (long j = 0; j < N; ++j) s1 += t.g(j, all_, all_);
      double d = 0.0;
      for (long i = 0; i < nb; ++i)
        for (long k = 0; k < nb; ++k) d = std::max(d, std::abs(s1(i, k) - ((i == k) ? 1.0 : 0.0)));
      REQUIRE(d < 1e-12);
      for (double w : {0.31, 2.7, 40.0}) {
        const cplx z = I_ * w;
        auto Gz = t.G(z);
        auto Sz = t.Sigma(z);
        nda::array<cplx, 2> M(nb, nb);
        for (long i = 0; i < nb; ++i)
          for (long k = 0; k < nb; ++k) M(i, k) = ((i == k) ? z : cplx(0.0)) - t.h(i, k) - Sz(i, k);
        nda::array<cplx, 2> P(nb, nb);
        nda::blas::gemm(Gz, M, P);
        double e = 0.0;
        for (long i = 0; i < nb; ++i)
          for (long k = 0; k < nb; ++k) e = std::max(e, std::abs(P(i, k) - ((i == k) ? 1.0 : 0.0)));
        REQUIRE(e < 1e-11);
      }
    }
    return t;
  }

  // residues of a list of toys (one per k) into the (ns=1, ng_total, nk, nb, nb) layout with
  // the union node set: toy ik occupies the nodes [off_k, off_k + ng_k).
  struct toy_set {
    std::vector<toy_k> toys;
    nda::array<double, 1> epsS, epsG;
    nda::array<cplx, 5> g, R;      // (1, ng_tot, nk, nb, nb), (1, np, nk, nb, nb)
    std::vector<long> off;
  };

  toy_set assemble(std::vector<toy_k> toys) {
    toy_set s;
    const long nk = long(toys.size()), nb = toys[0].nb, np = toys[0].np;
    long ng = 0;
    for (auto const &t : toys) { s.off.push_back(ng); ng += t.ng; }
    s.epsS = toys[0].epsS;
    s.epsG = nda::array<double, 1>(ng);
    s.g = nda::array<cplx, 5>(1, ng, nk, nb, nb);
    s.R = nda::array<cplx, 5>(1, np, nk, nb, nb);
    s.g() = cplx(0.0);
    for (long ik = 0; ik < nk; ++ik) {
      auto const &t = toys[size_t(ik)];
      for (long j = 0; j < t.ng; ++j) {
        s.epsG(s.off[size_t(ik)] + j) = t.lam(j);
        s.g(0, s.off[size_t(ik)] + j, ik, all_, all_) = t.g(j, all_, all_);
      }
      s.R(0, all_, ik, all_, all_) = t.R;
    }
    s.toys = std::move(toys);
    return s;
  }


  inline double max_abs(nda::MemoryArrayOfRank<2> auto const &A) {
    double m = 0.0;
    for (auto const &v : A) m = std::max(m, std::abs(v));
    return m;
  }

} // namespace bdft_tests

#endif
