/**
 * scGW-tilde Tier 1.5, increment T15-a (notes/tier15_ward_legs_plan.md sections 5-6): the
 * discrete-Ward leg vertex ALGEBRA (methods/vertex/ward_legs.hpp) on exact rational data.
 *
 * Pure toy, no MF, no scGW, seconds to run. Two cases:
 *
 *  ward_legs_conventions -- gate P0. The two-pole family T/T1/T2/T12 and the four-pole
 *    tables S_{plm}(nu), every confluence branch, nu = 0 and nu != 0, at beta = 10 and at
 *    the production beta = 1000, against DENSE fermionic Matsubara sums with Richardson
 *    elimination of the 1/M tail. Distinct G/Sigma node sets and the shared (by-index
 *    confluent) case.
 *
 *  ward_legs_toy_gates -- gates P1-P4. G and Sigma_c are built as EXACT pole sums from a
 *    Hermitian embedding H_big = [[h, B_p], [B_p^dag, e_p 1]]: G(z) = P (z - H_big)^-1 P^dag is
 *    a pole sum on the eigenvalues with PSD residues, Sigma_c(z) = sum_p B_p B_p^dag/(z - e_p),
 *    and G^-1 = z - h - Sigma_c EXACTLY (Schur complement), so the telescoping identity of
 *    proposal eq 22 holds to roundoff and any p-channel bug shows per nu.
 *      (b) the pole bare bubble and the Lambda correction at k != k+q against dense
 *          matrix-valued Matsubara oracles (pins S, the assembly, and the two insertion
 *          terms with distinguishable k / k+q data);
 *      P1 (G-g) q = 0, full window: the vertex-traced chi0_Lambda vanishes for every nu != 0;
 *          at nu = 0 it equals sum_j g_j f'(lambda_j) (the static thermal intraband term);
 *      P2 (G-h) Tr[g_j Lambda0(lambda_j; 0)] = 1 for every pole (the exact matrix Z-link);
 *          Herm[Lambda0(i pi/beta; 0) - 1] PSD at beta = 1000;
 *      P3 (G-i) R = 0 leaves the pair block bitwise untouched;
 *      P4 window: with C a proper subset the traced residual equals MINUS the a-outside-C
 *          complement of the full-window object (the exact bookkeeping identity);
 *      conditioning: a near-mu Sigma node (beta |e| ~ 2) at beta = 1000 -- the regime where
 *          S ~ beta^3 -- reports the telescoping residual (asserted at 1e-8, logged).
 */

#undef NDEBUG

#include <array>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "nda/linalg/eigenelements.hpp"
#include "numerics/nda_functions.hpp"
#include "methods/vertex/ward_legs.hpp"

namespace bdft_tests {

  namespace wl = methods::solvers::ward_legs;
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

  // ======================================================================================
  TEST_CASE("ward_legs_conventions", "[methods][vertex][scgwt][tier15]") {
    auto &mpi = utils::make_unit_test_mpi_context();
    (void)mpi;

    auto run = [&](double beta, long M, int levels, bool shared, double tol, long max_triples) {
      std::vector<double> eS = {-1.5, -0.3, 0.2, 0.9, 2.5};
      std::vector<double> eG = shared ? eS : std::vector<double>{-2.1, -0.7, 0.05, 0.6, 1.8, 3.3};
      nda::array<double, 1> epsS(long(eS.size())), epsG(long(eG.size()));
      for (long p = 0; p < epsS.shape(0); ++p) epsS(p) = eS[size_t(p)];
      for (long l = 0; l < epsG.shape(0); ++l) epsG(l) = eG[size_t(l)];
      nda::array<cplx, 1> inu(3);
      inu(0) = cplx(0.0);
      inu(1) = I_ * (2.0 * M_PI / beta);
      inu(2) = I_ * (2.0 * M_PI * 3.0 / beta);
      auto tab = wl::build_s_tables(beta, epsS, epsG, shared, inu);

      double errT = 0.0, errS = 0.0, scaleS = 0.0;
      long ntr = 0;
      for (long jn = 0; jn < 3; ++jn) {
        const cplx nu = inu(jn);
        // the bare kernel T(e_l, e_m)
        for (long l = 0; l < tab.ng; ++l)
          for (long m = 0; m < tab.ng; ++m) {
            const double a = epsG(l), b = epsG(m);
            const cplx ref = richardson_fsum(beta, M, levels,
                                             [&](cplx z) { return 1.0 / ((z - a) * (z + nu - b)); });
            errT = std::max(errT, std::abs(tab.Tb(jn, l, m) - ref) / std::max(1.0, std::abs(ref)));
          }
        // S_{plm}: all confluent triples + a bounded sample of the generic ones
        for (long p = 0; p < tab.np; ++p)
          for (long l = 0; l < tab.ng; ++l)
            for (long m = 0; m < tab.ng; ++m) {
              const bool conf = shared and (l == p or m == p);
              if (not conf) {
                if (ntr >= max_triples) continue;
                ++ntr;
              }
              const double ep = epsS(p), el = epsG(l), em = epsG(m);
              const cplx ref = richardson_fsum(beta, M, levels, [&](cplx z) {
                return 1.0 / ((z - el) * (z - ep) * (z + nu - ep) * (z + nu - em));
              });
              const cplx val = tab.Sm(jn, m, p, l);
              REQUIRE(tab.Sl(jn, l, p, m) == val);
              errS = std::max(errS, std::abs(val - ref) / std::max(1.0, std::abs(ref)));
              scaleS = std::max(scaleS, std::abs(ref));
            }
      }
      app_log(1, "ward_legs_conventions: beta = {} shared = {} M = {} levels = {}: "
                 "max rel err T = {:.3e}, S = {:.3e} (max |S| = {:.3e}, {} generic triples)",
              beta, shared, M, levels, errT, errS, scaleS, ntr);
      REQUIRE(errT < tol);
      REQUIRE(errS < tol);
    };
    // beta = 10: everything, 3 Richardson levels at M = 2e4 (tail ~ (beta/pi)^4 / M^3)
    run(10.0, 20000, 3, false, 1e-9, 1000000);
    run(10.0, 20000, 3, true, 1e-9, 1000000);
    // production beta = 1000: the pole scale is beta|e|/pi ~ 800 nodes, so 4 levels at
    // M = 4e5 (tail ~ (beta/pi)^5 / M^4 ~ 1e-11); confluent triples all, generic sampled
    run(1000.0, 400000, 4, true, 1e-8, 12);
    run(1000.0, 400000, 4, false, 1e-8, 12);
  }

  // ======================================================================================
  TEST_CASE("ward_legs_toy_gates", "[methods][vertex][scgwt][tier15]") {
    auto &mpi = utils::make_unit_test_mpi_context();
    (void)mpi;

    const long nb = 3;
    const std::vector<double> eS = {-1.5, -0.3, 0.2, 0.9, 2.5};
    const std::vector<double> rs = {0.35, 0.25, 0.3, 0.25, 0.4};
    auto set = assemble({make_toy(nb, eS, rs, 11u), make_toy(nb, eS, rs, 23u)});
    const long ng = set.epsG.shape(0), np = set.epsS.shape(0);
    auto const &t0 = set.toys[0];
    auto const &t1 = set.toys[1];

    const double beta = 10.0;
    nda::array<cplx, 1> inu(4);
    inu(0) = cplx(0.0);
    inu(1) = I_ * (2.0 * M_PI / beta);
    inu(2) = I_ * (2.0 * M_PI * 2.0 / beta);
    inu(3) = I_ * (2.0 * M_PI * 7.0 / beta);
    const long nnu = inu.shape(0);

    auto tab = wl::build_s_tables(beta, set.epsS, set.epsG, false, inu);
    auto blk = wl::slice_blocks(set.g, set.R, nda::range(0, nb));
    auto ctx = wl::build_ward_ctx(tab, blk.gC, blk.gA, blk.gB, blk.RA, blk.RB);
    REQUIRE(ctx.nc == nb);
    REQUIRE(ctx.ng == ng);
    REQUIRE(ctx.np == np);
    const long nc2 = nb * nb;

    // ---- (b) pins at k = 0, k+q = 1 against dense matrix-valued oracles ------------------
    {
      const long M = 20000;
      // one Matsubara pass per M level accumulates every pair-block entry of both objects
      auto dense_pair = [&](long Mm, cplx nu, nda::array<cplx, 2> &Sb, nda::array<cplx, 2> &Sl) {
        Sb() = cplx(0.0);
        Sl() = cplx(0.0);
        nda::array<cplx, 2> LG0(nb, nb), GL1(nb, nb);
        for (long n = -Mm; n < Mm; ++n) {
          const cplx z = I_ * cplx((2.0 * double(n) + 1.0) * M_PI / beta);
          auto G0 = t0.G(z);
          auto G1 = t1.G(z + nu);
          auto L0 = t0.Lam1(z, nu);
          auto L1 = t1.Lam1(z, nu);
          nda::blas::gemm(L0, G0, LG0);   // [(Lambda0 - 1) G](k, z; nu)
          nda::blas::gemm(G1, L1, GL1);   // [G (Lambda0 - 1)](k+q, z + nu; nu)
          for (long p1p = 0; p1p < nb; ++p1p)
            for (long p3 = 0; p3 < nb; ++p3)
              for (long a = 0; a < nb; ++a)
                for (long b = 0; b < nb; ++b) {
                  const long r = p1p * nb + p3, c = a * nb + b;
                  Sb(r, c) += G0(a, p1p) * G1(p3, b);
                  Sl(r, c) += 0.5 * (LG0(a, p1p) * G1(p3, b) + G0(a, p1p) * GL1(p3, b));
                }
        }
        Sb /= beta;
        Sl /= beta;
      };
      auto richardson_pair = [&](cplx nu, nda::array<cplx, 2> &Rb, nda::array<cplx, 2> &Rl) {
        std::array<nda::array<cplx, 2>, 3> sb, sl;
        for (int k = 0; k < 3; ++k) {
          sb[size_t(k)] = nda::array<cplx, 2>(nc2, nc2);
          sl[size_t(k)] = nda::array<cplx, 2>(nc2, nc2);
          dense_pair(M << k, nu, sb[size_t(k)], sl[size_t(k)]);
        }
        for (int order = 1; order < 3; ++order) {
          const double f = std::pow(2.0, order);
          for (int k = 0; k + order < 3; ++k) {
            sb[size_t(k)] = (f * sb[size_t(k + 1)] - sb[size_t(k)]) / (f - 1.0);
            sl[size_t(k)] = (f * sl[size_t(k + 1)] - sl[size_t(k)]) / (f - 1.0);
          }
        }
        Rb = sb[0];
        Rl = sl[0];
      };
      double err_bare = 0.0, err_lam = 0.0, sc_bare = 0.0, sc_lam = 0.0;
      for (long jn = 0; jn < nnu; ++jn) {
        nda::array<cplx, 2> Xb(nc2, nc2), Xl(nc2, nc2), Rb, Rl;
        wl::pole_bare_bubble(ctx, 0, jn, 0, 1, Xb);
        Xl() = cplx(0.0);
        wl::add_pair_correction(ctx, 0, jn, 0, 1, Xl);
        richardson_pair(inu(jn), Rb, Rl);
        for (long r = 0; r < nc2; ++r)
          for (long c = 0; c < nc2; ++c) {
            err_bare = std::max(err_bare, std::abs(Xb(r, c) - Rb(r, c)));
            err_lam = std::max(err_lam, std::abs(Xl(r, c) - Rl(r, c)));
            sc_bare = std::max(sc_bare, std::abs(Rb(r, c)));
            sc_lam = std::max(sc_lam, std::abs(Rl(r, c)));
          }
      }
      app_log(1, "ward_legs_toy_gates (b): k != k+q pins vs dense oracles: bare max err {:.3e} "
                 "(scale {:.3e}), Lambda max err {:.3e} (scale {:.3e})",
              err_bare, sc_bare, err_lam, sc_lam);
      REQUIRE(err_bare < 1e-9 * sc_bare);
      REQUIRE(err_lam < 1e-9 * sc_lam);
      REQUIRE(sc_lam > 1e-3 * sc_bare);   // the correction is not trivially small on this toy
    }

    // ---- P1 (G-g): q = 0 matrix telescoping at k = 0, full window ------------------------
    auto traced = [&](wl::ward_ctx const &c, long jn, long ik, nda::array<cplx, 2> &tr_bare,
                      nda::array<cplx, 2> &tr_full) {
      const long nc = c.nc, n2 = nc * nc;
      nda::array<cplx, 2> Xb(n2, n2), X(n2, n2);
      wl::pole_bare_bubble(c, 0, jn, ik, ik, Xb);
      X() = Xb;
      wl::add_pair_correction(c, 0, jn, ik, ik, X);
      tr_bare = nda::array<cplx, 2>(nc, nc);
      tr_full = nda::array<cplx, 2>(nc, nc);
      tr_bare() = cplx(0.0);
      tr_full() = cplx(0.0);
      for (long p1p = 0; p1p < nc; ++p1p)
        for (long p3 = 0; p3 < nc; ++p3)
          for (long a = 0; a < nc; ++a) {
            tr_bare(p3, p1p) += Xb(p1p * nc + p3, a * nc + a);
            tr_full(p3, p1p) += X(p1p * nc + p3, a * nc + a);
          }
    };
    {
      double worst = 0.0;
      for (long jn = 1; jn < nnu; ++jn) {
        nda::array<cplx, 2> trb, trf;
        traced(ctx, jn, 0, trb, trf);
        const double sb = max_abs(trb), sf = max_abs(trf);
        app_log(1, "ward_legs_toy_gates P1 (G-g): nu node {}: |traced bare| = {:.3e}, "
                   "|traced chi0_Lambda| = {:.3e} (ratio {:.3e})", jn, sb, sf, sf / sb);
        worst = std::max(worst, sf / sb);
        REQUIRE(sb > 1e-3);              // the bare bubble violates C1 at O(1)
      }
      REQUIRE(worst < 1e-12);
      // nu = 0: the traced object equals sum_j g_j f'(lambda_j)
      nda::array<cplx, 2> trb, trf;
      traced(ctx, 0, 0, trb, trf);
      nda::array<cplx, 2> ref(nb, nb);
      ref() = cplx(0.0);
      for (long j = 0; j < t0.ng; ++j) ref += t0.g(j, all_, all_) * wl::fermi_all(beta, t0.lam(j)).f1;
      double e = 0.0;
      for (long i = 0; i < nb; ++i)
        for (long k = 0; k < nb; ++k) e = std::max(e, std::abs(trf(i, k) - ref(i, k)));
      app_log(1, "ward_legs_toy_gates P1 (G-g) nu = 0: traced chi0_Lambda vs sum_j g_j f'(lambda_j): "
                 "max err {:.3e} (scale {:.3e}, bare {:.3e})", e, max_abs(ref), max_abs(trb));
      REQUIRE(e < 1e-12 * std::max(1.0, max_abs(ref)));
    }

    // ---- P2 (G-h): the Z-link and positivity ----------------------------------------------
    {
      double worst = 0.0;
      for (auto const &t : set.toys)
        for (long j = 0; j < t.ng; ++j) {
          auto L = t.Lam1(cplx(t.lam(j)), cplx(0.0));   // Lambda0(lambda_j; 0) - 1 (real z: exact here)
          cplx tr(0.0);
          for (long i = 0; i < nb; ++i)
            for (long k = 0; k < nb; ++k) tr += t.g(j, i, k) * (L(k, i) + ((i == k) ? 1.0 : 0.0));
          worst = std::max(worst, std::abs(tr - 1.0));
        }
      app_log(1, "ward_legs_toy_gates P2 (G-h): max |Tr[g_j Lambda0(lambda_j; 0)] - 1| = {:.3e}", worst);
      REQUIRE(worst < 1e-10);
      // positivity of Herm[Lambda0(i pi/beta; 0) - 1] at beta = 1000 (T -> 0 corner)
      const double b1 = 1000.0;
      auto L = t0.Lam1(I_ * (M_PI / b1), cplx(0.0));
      nda::matrix<cplx> Hm(nb, nb);
      for (long i = 0; i < nb; ++i)
        for (long k = 0; k < nb; ++k) Hm(i, k) = 0.5 * (L(i, k) + std::conj(L(k, i)));
      auto ev = nda::linalg::eigenvalues(Hm);
      double emin = 1e300, emax = -1e300;
      for (long i = 0; i < nb; ++i) { emin = std::min(emin, ev(i)); emax = std::max(emax, ev(i)); }
      app_log(1, "ward_legs_toy_gates P2 (G-h) positivity at beta = 1000: eig[Herm(Lambda0 - 1)] in "
                 "[{:.4e}, {:.4e}]", emin, emax);
      REQUIRE(emin > -1e-12 * emax);
      REQUIRE(emax > 0.0);
    }

    // ---- P3 (G-i): R = 0 leaves the block bitwise untouched ---------------------------------
    {
      nda::array<cplx, 5> R0(set.R.shape());
      R0() = cplx(0.0);
      auto b0 = wl::slice_blocks(set.g, R0, nda::range(0, nb));
      auto c0 = wl::build_ward_ctx(tab, b0.gC, b0.gA, b0.gB, b0.RA, b0.RB);
      std::mt19937 gen(7u);
      std::normal_distribution<double> nd(0.0, 1.0);
      nda::array<cplx, 2> Cb(nc2, nc2), Cref(nc2, nc2);
      for (auto &v : Cb) v = cplx(nd(gen), nd(gen));
      Cref() = Cb;
      for (long jn = 0; jn < nnu; ++jn) wl::add_pair_correction(c0, 0, jn, 0, 1, Cb);
      bool same = true;
      for (long r = 0; r < nc2; ++r)
        for (long c = 0; c < nc2; ++c) same = same and (Cb(r, c) == Cref(r, c));
      app_log(1, "ward_legs_toy_gates P3 (G-i): R = 0 leaves the pair block bitwise: {}", same);
      REQUIRE(same);
    }

    // ---- P4 window: C = [0, 2) residual = minus the a = 2 complement of the full object -----
    {
      const long ncw = 2;
      auto bw = wl::slice_blocks(set.g, set.R, nda::range(0, ncw));
      auto cw = wl::build_ward_ctx(tab, bw.gC, bw.gA, bw.gB, bw.RA, bw.RB);
      double worst = 0.0, scale = 0.0;
      for (long jn = 1; jn < nnu; ++jn) {
        nda::array<cplx, 2> trb_w, trf_w;
        traced(cw, jn, 0, trb_w, trf_w);          // (ncw, ncw): sum_{a < 2} X_w
        nda::array<cplx, 2> Xb(nc2, nc2), X(nc2, nc2);
        wl::pole_bare_bubble(ctx, 0, jn, 0, 0, Xb);
        X() = Xb;
        wl::add_pair_correction(ctx, 0, jn, 0, 0, X);
        for (long p1p = 0; p1p < ncw; ++p1p)
          for (long p3 = 0; p3 < ncw; ++p3) {
            const cplx comp = X(p1p * nb + p3, 2 * nb + 2);   // the a = b = 2 term of the full object
            worst = std::max(worst, std::abs(trf_w(p3, p1p) + comp));
            scale = std::max(scale, std::abs(comp));
          }
      }
      app_log(1, "ward_legs_toy_gates P4 (window C = [0,2) of 3): traced residual + complement: "
                 "max {:.3e} (complement scale {:.3e})", worst, scale);
      REQUIRE(scale > 1e-6);
      REQUIRE(worst < 1e-12 * std::max(1.0, scale));
    }

    // ---- conditioning: near-mu Sigma node at beta = 1000 --------------------------------------
    {
      const double b1 = 1000.0;
      const std::vector<double> eS2 = {-1.5, -0.3, 0.002, 0.2, 0.9, 2.5};
      const std::vector<double> rs2 = {0.35, 0.25, 0.05, 0.3, 0.25, 0.4};
      auto set2 = assemble({make_toy(nb, eS2, rs2, 31u)});
      nda::array<cplx, 1> inu2(3);
      inu2(0) = cplx(0.0);
      inu2(1) = I_ * (2.0 * M_PI / b1);
      inu2(2) = I_ * (2.0 * M_PI * 5.0 / b1);
      auto tab2 = wl::build_s_tables(b1, set2.epsS, set2.epsG, false, inu2);
      auto b2 = wl::slice_blocks(set2.g, set2.R, nda::range(0, nb));
      auto c2 = wl::build_ward_ctx(tab2, b2.gC, b2.gA, b2.gB, b2.RA, b2.RB);
      double smax = 0.0;
      for (auto const &v : tab2.Sm) smax = std::max(smax, std::abs(v));
      double worst = 0.0;
      for (long jn = 1; jn < 3; ++jn) {
        nda::array<cplx, 2> trb, trf;
        traced(c2, jn, 0, trb, trf);
        worst = std::max(worst, max_abs(trf) / max_abs(trb));
      }
      app_log(1, "ward_legs_toy_gates conditioning (beta = 1000, Sigma node at 0.002): max |S| = "
                 "{:.3e}, telescoping residual / bare = {:.3e}", smax, worst);
      REQUIRE(worst < 1e-8);
    }
  }

} // namespace bdft_tests
