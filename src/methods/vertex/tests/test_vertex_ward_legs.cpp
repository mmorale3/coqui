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
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/imag_axes_ft/dlr_pole_fit.hpp"
#include "toy_poles.hpp"

namespace bdft_tests {

  namespace wl = methods::solvers::ward_legs;

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

  // ======================================================================================
  // PHYSICS PROBE (T15-b): the SIGN and size of the eq-21 correction in the interband
  // channel on the imaginary axis. Scalar toys: a valence-like pole at k (h < 0) and a
  // conduction-like pole at k+q (h > 0), each dressed by satellite-like Sigma poles on its
  // own side. The bare pair element (1/beta) sum G_v G_c is a screening-like (negative)
  // number; the proposal's coherent-pole bookkeeping (eq 26) predicts Delta of the SAME
  // sign at ~ (1/Z - 1) relative size. Reported, not asserted -- this is what the LiH
  // fixture readout (T15-b) contradicted (anti-screening ~3x the RPA head at |q|^2 ~ 0.5).
  TEST_CASE("ward_legs_interband_sign", "[methods][vertex][scgwt][tier15]") {
    auto &mpi = utils::make_unit_test_mpi_context();
    (void)mpi;
    for (double beta : {10.0, 1000.0}) {
      for (double rs : {0.10, 0.15, 0.20}) {
        // k: valence at -0.5 with its main Sigma weight below; k+q: conduction at +0.5 with
        // its main weight above (a small weight on the other side keeps every G pole off
        // the Sigma nodes -- exact coincidences would divide by zero in the toy's tables)
        auto tv = make_toy(1, {-2.0, -1.5, 1.5, 2.0}, {rs, rs, 0.03, 0.03}, 5u);
        auto tc = make_toy(1, {-2.0, -1.5, 1.5, 2.0}, {0.03, 0.03, rs, rs}, 6u);
        tv.h(0, 0) = cplx(-0.5);
        tc.h(0, 0) = cplx(0.5);
        // rebuild the exact pole data for the shifted h (make_toy drew a random h)
        auto rebuild = [&](toy_k &t) {
          const long N = t.ng;
          nda::matrix<cplx> Hb(N, N);
          Hb() = cplx(0.0);
          Hb(0, 0) = t.h(0, 0);
          for (long p = 0; p < t.np; ++p) {
            const long o = 1 + p;
            const cplx b = std::sqrt(t.R(p, 0, 0));   // R = b b^*, b real >= 0
            Hb(0, o) = b;
            Hb(o, 0) = b;
            Hb(o, o) = cplx(t.epsS(p));
          }
          auto [ev, U] = nda::linalg::eigenelements(Hb);
          for (long j = 0; j < N; ++j) {
            t.lam(j) = ev(j);
            t.g(j, 0, 0) = U(0, j) * std::conj(U(0, j));
          }
        };
        rebuild(tv);
        rebuild(tc);
        // Z of the valence QP pole (the pole nearest -0.3) and its Lambda0(E;0)
        long jv = 0;
        for (long j = 0; j < tv.ng; ++j)
          if (std::abs(tv.lam(j) + 0.5) < std::abs(tv.lam(jv) + 0.5)) jv = j;
        const double Zv = tv.g(jv, 0, 0).real();
        auto set = assemble({tv, tc});
        nda::array<cplx, 1> inu(3);
        inu(0) = cplx(0.0);
        inu(1) = I_ * (2.0 * M_PI / beta);
        inu(2) = I_ * (2.0 * M_PI * 4.0 / beta);
        auto tab = wl::build_s_tables(beta, set.epsS, set.epsG, false, inu);
        auto blk = wl::slice_blocks(set.g, set.R, nda::range(0, 1));
        auto ctx = wl::build_ward_ctx(tab, blk.gC, blk.gA, blk.gB, blk.RA, blk.RB);
        for (long jn = 0; jn < 3; ++jn) {
          nda::array<cplx, 2> Xb(1, 1), Xl(1, 1);
          wl::pole_bare_bubble(ctx, 0, jn, 0, 1, Xb);
          Xl() = cplx(0.0);
          wl::add_pair_correction(ctx, 0, jn, 0, 1, Xl);
          app_log(1, "ward_legs_interband_sign: beta {} rs {} Z_v {:.4f} lam_v {:+.4f} node {}: "
                     "bare (v,c) = {:+.6e}{:+.3e}i, Delta = {:+.6e}{:+.3e}i, Delta/bare = {:+.4f}",
                  beta, rs, Zv, tv.lam(jv), jn, Xb(0, 0).real(), Xb(0, 0).imag(),
                  Xl(0, 0).real(), Xl(0, 0).imag(), (Xl(0, 0) / Xb(0, 0)).real());
        }
      }
    }
  }

  // PHYSICS PROBE 2 (T15-b): the SAME-BAND (intraband) channel of a FILLED band at finite q.
  // Both poles occupied (valence at k and at k+q, dispersion E_k != E_{k+q}); the bare pair
  // element carries the Pauli factor f(E_k) - f(E_{k+q}) = 0 (plus the tiny QP -> satellite
  // piece). With a q-INDEPENDENT Lambda0(k) the residue bracket becomes
  // Lambda0(E_k) - Lambda0(E_{k+q}) -- O(dispersion x dLambda/dE) over the same denominator,
  // i.e. O(1) at any finite q: a Pauli-blocking violation of the ansatz in the channel whose
  // density vertex is O(1). Reported, not asserted.
  TEST_CASE("ward_legs_intraband_probe", "[methods][vertex][scgwt][tier15]") {
    auto &mpi = utils::make_unit_test_mpi_context();
    (void)mpi;
    const double beta = 1000.0;
    for (double disp : {0.0, 0.05, 0.15, 0.30}) {
      for (double rs : {0.10, 0.20}) {
        auto ta = make_toy(1, {-2.0, -1.5, 1.5, 2.0}, {rs, rs, 0.03, 0.03}, 5u);
        auto tb = make_toy(1, {-2.0, -1.5, 1.5, 2.0}, {rs, rs, 0.03, 0.03}, 7u);
        ta.h(0, 0) = cplx(-0.5);
        tb.h(0, 0) = cplx(-0.5 + disp);
        auto rebuild = [&](toy_k &t) {
          const long N = t.ng;
          nda::matrix<cplx> Hb(N, N);
          Hb() = cplx(0.0);
          Hb(0, 0) = t.h(0, 0);
          for (long p = 0; p < t.np; ++p) {
            const long o = 1 + p;
            const cplx b = std::sqrt(t.R(p, 0, 0));
            Hb(0, o) = b;
            Hb(o, 0) = b;
            Hb(o, o) = cplx(t.epsS(p));
          }
          auto [ev, U] = nda::linalg::eigenelements(Hb);
          for (long j = 0; j < N; ++j) {
            t.lam(j) = ev(j);
            t.g(j, 0, 0) = U(0, j) * std::conj(U(0, j));
          }
        };
        rebuild(ta);
        rebuild(tb);
        auto set = assemble({ta, tb});
        nda::array<cplx, 1> inu(2);
        inu(0) = cplx(0.0);
        inu(1) = I_ * (2.0 * M_PI / beta);
        auto tab = wl::build_s_tables(beta, set.epsS, set.epsG, false, inu);
        auto blk = wl::slice_blocks(set.g, set.R, nda::range(0, 1));
        auto ctx = wl::build_ward_ctx(tab, blk.gC, blk.gA, blk.gB, blk.RA, blk.RB);
        for (long jn = 0; jn < 2; ++jn) {
          nda::array<cplx, 2> Xb(1, 1), Xl(1, 1);
          wl::pole_bare_bubble(ctx, 0, jn, 0, 1, Xb);
          Xl() = cplx(0.0);
          wl::add_pair_correction(ctx, 0, jn, 0, 1, Xl);
          app_log(1, "ward_legs_intraband_probe: disp {:.2f} rs {} node {}: bare (occ,occ) = "
                     "{:+.6e}{:+.3e}i, Delta = {:+.6e}{:+.3e}i", disp, rs, jn, Xb(0, 0).real(),
                  Xb(0, 0).imag(), Xl(0, 0).real(), Xl(0, 0).imag());
        }
      }
    }
  }

  // ======================================================================================
  // FITTED-RESIDUE PROBE (T15-b): the production pathway on exact data. The exact toy's
  // G(tau) and Sigma_c(tau) are sampled on the fixture's DLR grid (beta 1000, wmax 6, prec
  // low), pole-fitted with dlr_pole_fit (the aux NONSYM grid, shared node set), and the
  // ward algebra is run on the FITTED residues (shared = true) -- exactly what
  // build_ward_legs does. Compared against the exact-residue result at q = 0: the traced
  // pair propagator (bare and Lambda) at nu = 0 and at the first two nu != 0 nodes.
  // This isolates the "bilinear in the residues" hazard at the nu = 0 derivative branches.
  TEST_CASE("ward_legs_fitted_residues", "[methods][vertex][scgwt][tier15]") {
#ifndef ENABLE_DLR
    SUCCEED("ward_legs_fitted_residues skipped: build has ENABLE_DLR=OFF.");
#else
    auto &mpi = utils::make_unit_test_mpi_context();
    (void)mpi;
    imag_axes_ft::IAFT ft(1000, 6.0, imag_axes_ft::dlr_basis, "low");
    const double beta = ft.beta();
    imag_axes_ft::dlr_pole_fit pf(ft);
    const long nt = pf.nt, np = pf.np, nb = 3;
    // an insulator-like toy: h with eigenvalues straddling mu = 0, Sigma poles away from mu
    auto t = make_toy(nb, {-2.2, -1.4, -0.9, 1.1, 1.6, 2.4}, {0.3, 0.25, 0.2, 0.2, 0.25, 0.3}, 17u);
    for (long i = 0; i < nb; ++i) t.h(i, i) += cplx((i - 1) * 0.6);   // spread ~[-0.6, 0.6]
    {
      const long N = t.ng;
      nda::matrix<cplx> Hb(N, N);
      Hb() = cplx(0.0);
      for (long i = 0; i < nb; ++i)
        for (long j = 0; j < nb; ++j) Hb(i, j) = t.h(i, j);
      std::mt19937 gen(17u);
      std::normal_distribution<double> nd(0.0, 1.0);
      for (long p = 0; p < t.np; ++p) {
        // regenerate a B_p consistent with R_p = B B^dag: Cholesky-free route -- use the
        // eigen-decomposition of R_p (PSD) to get B = V sqrt(D)
        nda::matrix<cplx> Rm(nb, nb);
        for (long i = 0; i < nb; ++i)
          for (long j = 0; j < nb; ++j) Rm(i, j) = t.R(p, i, j);
        auto [ev, V] = nda::linalg::eigenelements(Rm);
        const long o = nb * (p + 1);
        for (long i = 0; i < nb; ++i) {
          for (long j = 0; j < nb; ++j) {
            const cplx b = V(i, j) * std::sqrt(std::max(ev(j), 0.0));
            Hb(i, o + j) = b;
            Hb(o + j, i) = std::conj(b);
          }
          Hb(o + i, o + i) = cplx(t.epsS(p));
        }
      }
      auto [ev, U] = nda::linalg::eigenelements(Hb);
      for (long j = 0; j < N; ++j) {
        t.lam(j) = ev(j);
        for (long i = 0; i < nb; ++i)
          for (long k = 0; k < nb; ++k) t.g(j, i, k) = U(i, j) * std::conj(U(k, j));
      }
      double gap_lo = -1e300, gap_hi = 1e300;
      for (long j = 0; j < N; ++j) {
        if (t.lam(j) < 0.0) gap_lo = std::max(gap_lo, t.lam(j));
        else gap_hi = std::min(gap_hi, t.lam(j));
      }
      app_log(1, "ward_legs_fitted_residues: toy poles in [{:.3f}, {:.3f}], gap ({:.3f}, {:.3f}); "
                 "aux grid np = {} nodes, min |eps| = {:.3e} Ha (beta*eps = {:.3e})",
              t.lam(0), t.lam(N - 1), gap_lo, gap_hi, np, pf.min_abs_node / beta, pf.min_abs_node);
    }
    // tau samples on the backend grid: F(tau) = sum_j c_j K_F(tau, e_j)
    auto sample = [&](nda::array<double, 1> const &e, nda::array<cplx, 3> const &c,
                      long ncoef) {
      nda::array<cplx, 2> F(nt, nb * nb);
      F() = cplx(0.0);
      for (long i = 0; i < nt; ++i)
        for (long j = 0; j < ncoef; ++j) {
          const double K = imag_axes_ft::dlr_kF(beta, pf.s_phys(i), e(j));
          for (long a = 0; a < nb; ++a)
            for (long b = 0; b < nb; ++b) F(i, a * nb + b) += c(j, a, b) * K;
        }
      return F;
    };
    auto FG = sample(t.lam, t.g, t.ng);
    auto FS = sample(t.epsS, t.R, t.np);
    auto cG = pf.coeffs(FG);
    auto cS = pf.coeffs(FS);
    app_log(1, "ward_legs_fitted_residues: fit_error G {:.3e}, Sigma {:.3e}; residue ratio G "
               "{:.3g}, Sigma {:.3g}", pf.fit_error(FG, cG), pf.fit_error(FS, cS),
            pf.residue_ratio(FG, cG), pf.residue_ratio(FS, cS));
    // fitted residues into the (1, np, 1, nb, nb) layout
    nda::array<cplx, 5> gF(1, np, 1, nb, nb), RF(1, np, 1, nb, nb);
    for (long p = 0; p < np; ++p)
      for (long a = 0; a < nb; ++a)
        for (long b = 0; b < nb; ++b) {
          gF(0, p, 0, a, b) = cG(p, a * nb + b);
          RF(0, p, 0, a, b) = cS(p, a * nb + b);
        }
    nda::array<cplx, 1> inu(3);
    inu(0) = cplx(0.0);
    inu(1) = I_ * (2.0 * M_PI / beta);
    inu(2) = I_ * (2.0 * M_PI * 2.0 / beta);
    // fitted route: shared aux nodes -- the nu = 0 row as the exact derivative branches
    // (K = 0) and as the nu -> 0 limit from K = 2, 3, 4 Matsubara nodes
    for (long K : {0l, 2l, 3l, 4l}) {
    auto tabF = wl::build_s_tables(beta, pf.epsl, pf.epsl, true, inu, K);
    auto blkF = wl::slice_blocks(gF, RF, nda::range(0, nb));
    auto ctxF = wl::build_ward_ctx(tabF, blkF.gC, blkF.gA, blkF.gB, blkF.RA, blkF.RB);
    // exact route: the toy's own poles
    auto set = assemble({t});
    auto tabE = wl::build_s_tables(beta, set.epsS, set.epsG, false, inu);
    auto blkE = wl::slice_blocks(set.g, set.R, nda::range(0, nb));
    auto ctxE = wl::build_ward_ctx(tabE, blkE.gC, blkE.gA, blkE.gB, blkE.RA, blkE.RB);
    const long nc2 = nb * nb;
    for (long jn = 0; jn < 3; ++jn) {
      nda::array<cplx, 2> XbF(nc2, nc2), XlF(nc2, nc2), XbE(nc2, nc2), XlE(nc2, nc2);
      wl::pole_bare_bubble(ctxF, 0, jn, 0, 0, XbF);
      wl::pole_bare_bubble(ctxE, 0, jn, 0, 0, XbE);
      XlF() = cplx(0.0);
      XlE() = cplx(0.0);
      wl::add_pair_correction(ctxF, 0, jn, 0, 0, XlF);
      wl::add_pair_correction(ctxE, 0, jn, 0, 0, XlE);
      double db = 0.0, dl = 0.0, sb = 0.0, sl = 0.0;
      nda::array<cplx, 2> trE(nb, nb), trF(nb, nb), trbE(nb, nb), trbF(nb, nb);
      trE() = cplx(0.0); trF() = cplx(0.0); trbE() = cplx(0.0); trbF() = cplx(0.0);
      for (long r = 0; r < nc2; ++r)
        for (long c = 0; c < nc2; ++c) {
          db = std::max(db, std::abs(XbF(r, c) - XbE(r, c)));
          dl = std::max(dl, std::abs(XlF(r, c) - XlE(r, c)));
          sb = std::max(sb, std::abs(XbE(r, c)));
          sl = std::max(sl, std::abs(XlE(r, c)));
        }
      for (long p1p = 0; p1p < nb; ++p1p)
        for (long p3 = 0; p3 < nb; ++p3)
          for (long a = 0; a < nb; ++a) {
            const long r = p1p * nb + p3, c = a * nb + a;
            trbE(p3, p1p) += XbE(r, c);
            trbF(p3, p1p) += XbF(r, c);
            trE(p3, p1p) += XbE(r, c) + XlE(r, c);
            trF(p3, p1p) += XbF(r, c) + XlF(r, c);
          }
      app_log(1, "ward_legs_fitted_residues [nu0_extrap {}]: node {}: |bare_fit - bare_exact| "
                 "{:.3e} (scale {:.3e}); |Delta_fit - Delta_exact| {:.3e} (scale {:.3e}); traced "
                 "q=0: bare exact {:.3e} fit {:.3e}; Lambda-corrected exact {:.3e} fit {:.3e}",
              K, jn, db, sb, dl, sl, max_abs(trbE), max_abs(trbF), max_abs(trE), max_abs(trF));
      REQUIRE(db < 1e-4 * sb);                          // the bare bubble: fit class at every node
      if (jn > 0) REQUIRE(dl < 1e-3 * sl);              // nu != 0: fit class always
      if (jn == 0 and K == 3) REQUIRE(dl < 1e-2 * sl);  // nu = 0 via the limit: extrapolation class
    }
    }
#endif
  }

} // namespace bdft_tests
