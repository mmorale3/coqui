/**
 * P2.1 / eq:pibardynfact -- gate for the FACTORIZED equal-time dynamic-rung polarization
 * (vertex_pi::pi_dyn_factorized).
 *
 * WHAT IS BEING PROVED. B-L's response middle factor needs only
 *
 *     pi^dyn(q) = (1/beta) sum_nu Pi^{C,dyn}(q, i nu) = Pi^{C,dyn}(q, tau = 0),
 *
 * and the production route obtained it by running the FULL dynamic-rung Pi^C kernel over
 * all nw_b frequencies -- twisted pairs, DLR pole algebra and all -- and then keeping only
 * the tau = 0 row. That slot measured 98.9 % of B-L's vertex time
 * (notes/vertex_optimization_plan.md P2.1). Theory eq:pibardynfact says the external
 * frequency sum closes the (12) and (34) G-pairs and leaves ONE bosonic pairing of two
 * ordinary bubbles against W -- no pole algebra at all (open item O7: "nothing in section
 * BL needs pole algebra").
 *
 * This file makes the switch-over a PROVABLE REFACTOR rather than a physics change: the new
 * primitive is compared against R0-contracted output of the existing, separately-pinned
 * kernel (pinned by dense_cross_check / pin_rpa_bubble in test_vertex_pi.cpp) on IDENTICAL
 * inputs. The two implementations route indices completely differently -- (12)/(34) at the
 * rung frequency versus (14)/(23) at the external frequency -- so agreement is a real
 * cross-check and not a tautology.
 *
 * The routing itself is pinned to machine precision independently, on a cyclic Matsubara
 * model where the identity is pure algebra: notes/pins/pin_pibardynfact.py reports
 * rel 1.5e-15 at three transfers and REJECTS all five plausible mis-readings (swapped
 * bubbles, either bubble transposed, the (14)/(23) grouping, bubbles at q instead of qx)
 * at O(1). What this file adds is that the C++ implementation -- packing, prefactor,
 * external fold, trev/PQ handling -- realizes that pinned routing.
 *
 * SECTIONS
 *  - static_rung            the POLE-FREE gate: a frequency-independent rung, so neither
 *                           side touches the aux pole basis. Any disagreement here is
 *                           packing/prefactor/fold, isolated from every pole question.
 *  - static_rung_nonhermitian  the same with a rung carrying NO pair symmetry. A Hermitian
 *                           or symmetric rung makes several distinct transpose readings
 *                           coincide; this is the case that separates them, and it is the
 *                           O7 risk class.
 *  - dynamic_rung           the real refactor gate: the reference runs phase 2's
 *                           twisted-pair pole algebra; the new primitive does not.
 *  - dynamic_rung_nonhermitian  both at once.
 *  - rung_linearity         pi^dyn is linear in the rung => (Z, Wdyn) must equal
 *                           (Z, none) + (0, Wdyn). Pins the Wl = Z + Wdyn assembly.
 *  - rank_split             the (tuple x q_ext) partials sum to the serial result.
 *  - floor_tracks_dlr_eps   the residual ~1e-10 above is a REPRESENTABILITY floor, not a
 *                           routing bug: it must fall with the basis tolerance. Measured
 *                           ~30*eps at this grid.
 *  - production_grid_attribution  the same at LiH-222's grid parameters (beta = 1000,
 *                           wmax = 6), where the floor is ~2000*eps and the production
 *                           check-mode gate reads 3.6e-3. Attributes it: NOT the pole
 *                           algebra (excess factor 0.80), and splits each route's own
 *                           convergence error. See the conclusions block in that section --
 *                           it carries a retraction and the honest accuracy trade-off.
 *  - no_pole_basis          the primitive leaves the pole machinery UNINITIALIZED, i.e. the
 *                           removal of B-L's only contact with the conditioning defect is
 *                           structural and not merely incidental.
 */

#include <cmath>
#include <random>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"

#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/vertex/vertex_pi.icc"
#include "methods/vertex/vertex_t.h"

namespace bdft_tests {

  using namespace methods;
  namespace vertex_pi = methods::solvers::vertex_pi;
  using vertex_pi::iaft_tools;
  using cplx = ComplexType;
  static auto const t_all = nda::range::all;

  namespace pbdf {

    constexpr double beta = 20.0;
    constexpr double wmax = 8.0;
    constexpr double Omega = 0.61;              // toy W_dyn pole
    constexpr long nk = 3, nbnd = 3, Np = 4, ns = 1;
    inline nda::range C() { return nda::range(0, 2); }

    struct rng_t {
      std::mt19937 gen;
      std::uniform_real_distribution<double> dist{-0.5, 0.5};
      explicit rng_t(unsigned seed) : gen(seed) {}
      cplx operator()() { return cplx(dist(gen), dist(gen)); }
    };

    inline nda::array<cplx, 2> unitary(long n, rng_t& rng) {
      auto refl = [&](nda::array<cplx, 1> const& v) {
        nda::array<cplx, 2> H(n, n);
        double nv = 0;
        for (long i = 0; i < n; ++i) nv += std::norm(v(i));
        for (long i = 0; i < n; ++i)
          for (long j = 0; j < n; ++j)
            H(i, j) = ((i == j) ? cplx(1.0) : cplx(0.0)) - 2.0 * v(i) * std::conj(v(j)) / nv;
        return H;
      };
      nda::array<cplx, 1> v1(n), v2(n);
      for (long i = 0; i < n; ++i) { v1(i) = rng(); v2(i) = rng(); }
      nda::array<cplx, 2> U(n, n);
      nda::blas::gemm(refl(v1), refl(v2), U);
      return U;
    }

    // stable -e^{-eps*tau} (1 - nF(eps)) for tau in [0, bta]
    inline double g_tau(double eps, double tau, double bta) {
      if (eps >= 0.0) return -std::exp(-eps * tau) / (1.0 + std::exp(-bta * eps));
      return -std::exp(eps * (bta - tau)) / (1.0 + std::exp(bta * eps));
    }

    /**
     * Same construction as toy::model_t in test_vertex_pi.cpp (a Lehmann G from random
     * unitaries, a positive-definite core, a single-pole dynamic rung), plus a rung with
     * NO pair symmetry for the transpose-discriminating sections.
     */
    struct model_t {
      nda::array<double, 2> eps;       // (nk, nbnd)
      nda::array<cplx, 4> Pr;          // (nk, nbnd[r], nbnd, nbnd)
      nda::array<cplx, 4> X_skPa;      // (ns, nk, Np, nbnd)
      nda::array<cplx, 3> Z_qPQ;       // (nk, Np, Np)  Hermitian
      nda::array<cplx, 3> Zns_qPQ;     // (nk, Np, Np)  NO symmetry at all
      nda::array<cplx, 3> M_qPQ;       // (nk, Np, Np)  W_dyn(q, inu) = M_q * s(nu)
      nda::array<long, 2> kmq, kpq;    // (nq, nk)

      model_t() : eps(nk, nbnd), Pr(nk, nbnd, nbnd, nbnd),
                  X_skPa(ns, nk, Np, nbnd), Z_qPQ(nk, Np, Np), Zns_qPQ(nk, Np, Np),
                  M_qPQ(nk, Np, Np), kmq(nk, nk), kpq(nk, nk) {
        rng_t rng(17);
        const double base[3] = {-0.71, -0.13, 0.47};
        for (long k = 0; k < nk; ++k) {
          for (long r = 0; r < nbnd; ++r) eps(k, r) = base[r] + 0.07 * double(k);
          auto U = unitary(nbnd, rng);
          for (long r = 0; r < nbnd; ++r)
            for (long i = 0; i < nbnd; ++i)
              for (long j = 0; j < nbnd; ++j)
                Pr(k, r, i, j) = U(i, r) * std::conj(U(j, r));
        }
        for (long k = 0; k < nk; ++k)
          for (long P = 0; P < Np; ++P)
            for (long a = 0; a < nbnd; ++a) X_skPa(0, k, P, a) = rng();
        for (long q = 0; q < nk; ++q) {
          nda::array<cplx, 2> Y(Np, Np), V(Np, 2);
          for (auto& y : Y) y = rng();
          for (auto& v : V) v = rng();
          for (long P = 0; P < Np; ++P)
            for (long Q = 0; Q < Np; ++Q) {
              cplx zy = 0, zv = 0;
              for (long r = 0; r < Np; ++r) zy += Y(P, r) * std::conj(Y(Q, r));
              for (long r = 0; r < 2; ++r) zv += V(P, r) * std::conj(V(Q, r));
              Z_qPQ(q, P, Q) = zy / double(Np) + ((P == Q) ? cplx(0.3) : cplx(0.0));
              M_qPQ(q, P, Q) = zv / double(Np);
            }
        }
        // deliberately asymmetric: Zns(P,Q) != Zns(Q,P) and != conj(Zns(Q,P)) for all P<Q,
        // so a transposed / side-swapped reading of either rung pair cannot survive.
        rng_t rng2(20260730);
        for (long q = 0; q < nk; ++q)
          for (long P = 0; P < Np; ++P)
            for (long Q = 0; Q < Np; ++Q)
              Zns_qPQ(q, P, Q) = rng2() + ((P == Q) ? cplx(0.4) : cplx(0.0));
        for (long q = 0; q < nk; ++q)
          for (long k = 0; k < nk; ++k) {
            kmq(q, k) = (k - q + nk) % nk;
            kpq(q, k) = (k + q) % nk;
          }
      }

      nda::array<cplx, 5> G_tau(imag_axes_ft::IAFT const& ft) const {
        long nt = ft.nt_f();
        const double bta = ft.beta();
        auto xm = ft.tau_mesh();
        nda::array<cplx, 5> G(nt, ns, nk, nbnd, nbnd);
        G() = cplx(0.0);
        for (long it = 0; it < nt; ++it) {
          double tau = (xm(it) + 1.0) * 0.5 * bta;
          for (long k = 0; k < nk; ++k)
            for (long r = 0; r < nbnd; ++r) {
              double g = g_tau(eps(k, r), tau, bta);
              for (long i = 0; i < nbnd; ++i)
                for (long j = 0; j < nbnd; ++j)
                  G(it, 0, k, i, j) += g * Pr(k, r, i, j);
            }
        }
        return G;
      }

      nda::array<cplx, 4> Wdyn(iaft_tools const& tools) const {
        nda::array<cplx, 4> W(nk, tools.nw_b, Np, Np);
        for (long q = 0; q < nk; ++q)
          for (long l = 0; l < tools.nw_b; ++l) {
            double nu = double(tools.wn_b(l)) * M_PI / tools.beta;
            double s = 2.0 * Omega / (Omega * Omega + nu * nu);
            for (long P = 0; P < Np; ++P)
              for (long Q = 0; Q < Np; ++Q) W(q, l, P, Q) = M_qPQ(q, P, Q) * s;
          }
        return W;
      }
    };

  } // namespace pbdf

  TEST_CASE("vertex_pibardynfact", "[methods][vertex][pi_c][pibardynfact]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_pibardynfact skipped: build has ENABLE_DLR=OFF. (The factorized "
            "primitive itself needs no pole basis, but the dynamic-rung REFERENCE does.)");
#else
    using namespace pbdf;
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, 1e-11);
    model_t mdl;
    auto G = mdl.G_tau(ft);
    const long nw_b = ft.nw_b();
    auto R0 = solvers::vertex_w0_detail::tau0_transform_row(ft);
    REQUIRE(R0.shape(0) == nw_b);

    // ---- the reference: the existing kernel on the frequency mesh, then the tau = 0 row.
    // This is EXACTLY what vertex_t.cpp does today (pi_c_accumulate_w followed by tau0_of).
    auto reference = [&](nda::array<cplx, 3> const& Zc,
                        nda::array<cplx, 4> const* Wd) {
      iaft_tools tools(ft);                    // fresh: the reference may build poles
      nda::array<cplx, 4> Pi_wq(nw_b, nk, Np, Np);
      Pi_wq() = cplx(0.0);
      vertex_pi::pi_c_accumulate_w(ft, tools, G, mdl.X_skPa, Zc, Wd,
                                   mdl.kmq, mdl.kpq, C(), Pi_wq, 0, 1);
      nda::array<cplx, 3> out(nk, Np, Np);
      for (long iq = 0; iq < nk; ++iq)
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            cplx acc(0.0);
            for (long m = 0; m < nw_b; ++m) acc += R0(m) * Pi_wq(m, iq, P, Q);
            out(iq, P, Q) = acc;
          }
      return out;
    };

    // ---- the new primitive
    auto factorized = [&](nda::array<cplx, 3> const& Zc,
                          nda::array<cplx, 4> const* Wd,
                          long rank = 0, long nproc = 1) {
      iaft_tools tools(ft);
      nda::array<cplx, 3> out(nk, Np, Np);
      out() = cplx(0.0);
      vertex_pi::pi_dyn_factorized(tools, G, mdl.X_skPa, Zc, Wd,
                                   mdl.kmq, mdl.kpq, C(), R0, out, rank, nproc);
      return out;
    };

    auto rel_dev = [&](nda::array<cplx, 3> const& A, nda::array<cplx, 3> const& B,
                       double& num, double& den) {
      num = 0.0; den = 0.0;
      for (long iq = 0; iq < nk; ++iq)
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            num = std::max(num, std::abs(A(iq, P, Q) - B(iq, P, Q)));
            den = std::max(den, std::abs(B(iq, P, Q)));
          }
      return (den > 0.0) ? num / den : num;
    };

    // Both sides are exact Matsubara sums of DIFFERENT integrands -- Pi^C(i nu_ext) on one
    // side, b34(i nu_x) wbar b12(i nu_x) on the other -- each evaluated through the same
    // tau = 0 transform row. So the agreement floor is the bosonic REPRESENTABILITY of the
    // two integrands on the sparse grid (ft eps = 1e-11 here), not machine epsilon. Every
    // mis-reading the routing pin rejects is O(1), so this separates them by >= 8 orders.
    // MEASURED on this toy (2026-07-30): 2.2e-10 / 1.4e-10 (static, Hermitian / no pair
    // symmetry) and 2.9e-10 / 1.9e-10 (dynamic), i.e. absolute deviations of ~1e-12 on
    // objects of size ~4e-3. tol is set ~30x above the worst measurement.
    const double tol = 1e-8;

    SECTION("static_rung") {
      // POLE-FREE on both sides: a frequency-independent rung makes the reference's phase 2
      // identically zero (pinned by test_vertex_pi's static_rung_W0), so a failure here is
      // packing / prefactor / external fold, with every pole question excluded.
      auto ref = reference(mdl.Z_qPQ, nullptr);
      auto got = factorized(mdl.Z_qPQ, nullptr);
      double num, den;
      double rel = rel_dev(got, ref, num, den);
      app_log(1, "pibardynfact static_rung: max|fact - kernel@tau0| = {:.6e}, "
                 "max|kernel@tau0| = {:.6e}, rel = {:.3e}", num, den, rel);
      REQUIRE(den > 1e-10);
      REQUIRE(rel <= tol);
    }

    SECTION("static_rung_nonhermitian") {
      // The discriminating case: with a rung that has no pair symmetry, a transposed or
      // side-swapped reading of either rung pair changes the answer at O(1).
      auto ref = reference(mdl.Zns_qPQ, nullptr);
      auto got = factorized(mdl.Zns_qPQ, nullptr);
      double num, den;
      double rel = rel_dev(got, ref, num, den);
      app_log(1, "pibardynfact static_rung_nonhermitian: max|fact - kernel@tau0| = {:.6e}, "
                 "max|kernel@tau0| = {:.6e}, rel = {:.3e}", num, den, rel);
      REQUIRE(den > 1e-10);
      REQUIRE(rel <= tol);
      // positive control: the deliberately asymmetric rung really is asymmetric, i.e. this
      // section is not silently repeating the Hermitian one.
      double asym = 0.0;
      for (long q = 0; q < nk; ++q)
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q)
            asym = std::max(asym, std::abs(mdl.Zns_qPQ(q, P, Q) -
                                           std::conj(mdl.Zns_qPQ(q, Q, P))));
      REQUIRE(asym > 1e-2);
    }

    SECTION("dynamic_rung") {
      // THE REFACTOR GATE. The reference runs the twisted-pair DLR pole algebra over all
      // nw_b frequencies and throws away everything but tau = 0; the new primitive never
      // enters that algebra.
      auto Wd = mdl.Wdyn(iaft_tools(ft));
      auto ref = reference(mdl.Z_qPQ, &Wd);
      auto got = factorized(mdl.Z_qPQ, &Wd);
      double num, den;
      double rel = rel_dev(got, ref, num, den);
      app_log(1, "pibardynfact dynamic_rung: max|fact - kernel@tau0| = {:.6e}, "
                 "max|kernel@tau0| = {:.6e}, rel = {:.3e}", num, den, rel);
      REQUIRE(den > 1e-10);
      REQUIRE(rel <= tol);
      // the dynamic rung must actually MOVE the answer -- otherwise this section would
      // silently degenerate into static_rung.
      auto stat = factorized(mdl.Z_qPQ, nullptr);
      double n2, d2;
      double moved = rel_dev(got, stat, n2, d2);
      app_log(1, "pibardynfact dynamic_rung: |pi^dyn(Z + Wdyn) - pi^dyn(Z)| / |pi^dyn(Z)| "
                 "= {:.3e} (must be O(1) for the section to have teeth)", moved);
      REQUIRE(moved > 1e-3);
    }

    SECTION("dynamic_rung_nonhermitian") {
      auto Wd = mdl.Wdyn(iaft_tools(ft));
      auto ref = reference(mdl.Zns_qPQ, &Wd);
      auto got = factorized(mdl.Zns_qPQ, &Wd);
      double num, den;
      double rel = rel_dev(got, ref, num, den);
      app_log(1, "pibardynfact dynamic_rung_nonhermitian: max|fact - kernel@tau0| = {:.6e}, "
                 "max|kernel@tau0| = {:.6e}, rel = {:.3e}", num, den, rel);
      REQUIRE(den > 1e-10);
      REQUIRE(rel <= tol);
    }

    SECTION("rung_linearity") {
      // pi^dyn is linear in the rung, so the Wl = Z + Wdyn(l) assembly must satisfy
      //   pi^dyn(Z, Wdyn) = pi^dyn(Z, none) + pi^dyn(0, Wdyn).
      // This is internal to the new primitive (no kernel involved) and pins the one place
      // where the instantaneous core and the dynamic part are combined.
      auto Wd = mdl.Wdyn(iaft_tools(ft));
      nda::array<cplx, 3> Zzero(nk, Np, Np);
      Zzero() = cplx(0.0);
      auto full = factorized(mdl.Z_qPQ, &Wd);
      auto zpart = factorized(mdl.Z_qPQ, nullptr);
      auto wpart = factorized(Zzero, &Wd);
      double num = 0.0, den = 0.0;
      for (long iq = 0; iq < nk; ++iq)
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            num = std::max(num, std::abs(full(iq, P, Q) -
                                         (zpart(iq, P, Q) + wpart(iq, P, Q))));
            den = std::max(den, std::abs(full(iq, P, Q)));
          }
      app_log(1, "pibardynfact rung_linearity: max|full - (Z part + W part)| = {:.6e}, "
                 "max|full| = {:.6e}", num, den);
      REQUIRE(den > 1e-10);
      REQUIRE(num <= 1e-12 * den);
    }

    SECTION("rank_split") {
      // The (tuple x q_ext) round-robin partials must sum to the serial result. Only the
      // reduction order changes, so this is the FP-reassociation floor.
      auto Wd = mdl.Wdyn(iaft_tools(ft));
      auto serial = factorized(mdl.Z_qPQ, &Wd, 0, 1);
      for (long nproc : {2L, 3L, 4L, 6L}) {
        nda::array<cplx, 3> acc(nk, Np, Np);
        acc() = cplx(0.0);
        for (long r = 0; r < nproc; ++r) {
          auto part = factorized(mdl.Z_qPQ, &Wd, r, nproc);
          acc() += part();
        }
        double num, den;
        double rel = rel_dev(acc, serial, num, den);
        app_log(1, "pibardynfact rank_split (nproc = {}): rel = {:.3e}", nproc, rel);
        REQUIRE(den > 1e-10);
        REQUIRE(rel <= 1e-12);
      }
    }

    SECTION("floor_tracks_dlr_eps") {
      // WHY THIS SECTION EXISTS. Every gate above lands at ~1e-10, not at machine epsilon,
      // and the claim is that this is the bosonic REPRESENTABILITY of two different
      // integrands on the sparse grid. That claim is load-bearing: it is what makes the
      // residual difference benign, and vertex_t's "check"-mode abort message asks the
      // operator to decide "representability failure or routing bug" -- so answer it here
      // rather than asserting it in prose. A routing bug is eps-INDEPENDENT; a
      // representability floor tracks eps. Measure the scaling.
      //
      // This also CALIBRATES the production numbers: physics runs use prec = "low"
      // (eps = 1e-6), where the floor is correspondingly ~1e-5 of pi^dyn -- which is what
      // shows up as a ~3e-5 relative movement of B-L's vertex shift on LiH-222.
      // Uses the NO-pair-symmetry rung so the scan is run on the discriminating case.
      auto dev_at = [&](double eps) {
        imag_axes_ft::IAFT f(beta, wmax, imag_axes_ft::dlr_basis, eps);
        auto Ge = mdl.G_tau(f);
        auto Re = solvers::vertex_w0_detail::tau0_transform_row(f);
        const long nwb = f.nw_b();
        iaft_tools tk(f), tf(f);
        nda::array<cplx, 4> Pi_wq(nwb, nk, Np, Np);
        Pi_wq() = cplx(0.0);
        vertex_pi::pi_c_accumulate_w(f, tk, Ge, mdl.X_skPa, mdl.Zns_qPQ, static_cast<nda::array<ComplexType, 4> const*>(nullptr),
                                     mdl.kmq, mdl.kpq, C(), Pi_wq, 0, 1);
        nda::array<cplx, 3> ref(nk, Np, Np), got(nk, Np, Np);
        for (long iq = 0; iq < nk; ++iq)
          for (long P = 0; P < Np; ++P)
            for (long Q = 0; Q < Np; ++Q) {
              cplx acc(0.0);
              for (long m = 0; m < nwb; ++m) acc += Re(m) * Pi_wq(m, iq, P, Q);
              ref(iq, P, Q) = acc;
            }
        got() = cplx(0.0);
        vertex_pi::pi_dyn_factorized(tf, Ge, mdl.X_skPa, mdl.Zns_qPQ, static_cast<nda::array<ComplexType, 4> const*>(nullptr),
                                     mdl.kmq, mdl.kpq, C(), Re, got, 0, 1);
        double n, d;
        double r = rel_dev(got, ref, n, d);
        app_log(1, "pibardynfact floor_tracks_dlr_eps: eps = {:.0e} -> nw_b = {}, "
                   "rel dev = {:.3e}", eps, nwb, r);
        return r;
      };
      const double d6 = dev_at(1e-6);
      const double d9 = dev_at(1e-9);
      const double d12 = dev_at(1e-12);
      app_log(1, "pibardynfact floor_tracks_dlr_eps: 1e-6 -> {:.3e}, 1e-9 -> {:.3e}, "
                 "1e-12 -> {:.3e}; ratio(1e-6 / 1e-12) = {:.3e}", d6, d9, d12, d6 / d12);
      REQUIRE(d6 > d9);
      REQUIRE(d9 > d12);
      // falls by orders as the basis tightens => representability, not routing.
      REQUIRE(d6 / d12 > 1e3);
    }

    SECTION("production_grid_attribution") {
      // WHY. On LiH-222 -- beta = 1000, wmax = 6, prec = "low" (eps = 1e-6) -- the
      // production check-mode gate measured |factorized - kernel| / |kernel| = 3.6e-3 on
      // pi^dyn. That is 120x above the ~30*eps floor the beta = 20 scan above shows, so
      // "it's representability" does not explain it, and vertex_t's own abort message
      // forbids raising a tolerance without deciding WHICH route is at fault. Decide it.
      //
      // THE DISCRIMINATOR is the STATIC rung at the SAME grid. With a frequency-independent
      // rung the reference route is phase 1 only, and phase 1 is pole-free (phase 2 is
      // pinned identically zero for a constant rung by test_vertex_pi/static_rung_W0). So
      //   static deviation  = the floor SHARED by the two routings (two different exact
      //                       Matsubara sums, each read through the same tau = 0 row)
      //   dynamic deviation = that floor PLUS whatever the kernel's twisted-pair pole
      //                       algebra adds -- the map whose worst-case residue
      //                       amplification this grid reports as ~1e7.
      // dynamic >> static at one grid therefore attributes the excess to the route being
      // REMOVED, not to the primitive being added.
      const double bta = 1000.0, wmx = 6.0;
      // returns {factorized, kernel@tau0} for one (eps, rung) -- both are approximations to
      // the SAME grid-independent object pi^dyn(q)_{MN}, so results from different eps are
      // directly comparable.
      auto run_both = [&](double eps, bool dynamic) {
        imag_axes_ft::IAFT f(bta, wmx, imag_axes_ft::dlr_basis, eps);
        auto Ge = mdl.G_tau(f);
        auto Re = solvers::vertex_w0_detail::tau0_transform_row(f);
        const long nwb = f.nw_b();
        iaft_tools tk(f), tf(f);
        nda::array<cplx, 4> Wd = mdl.Wdyn(tk);
        nda::array<cplx, 4> const* wp = dynamic ? &Wd : nullptr;
        nda::array<cplx, 4> Pi_wq(nwb, nk, Np, Np);
        Pi_wq() = cplx(0.0);
        vertex_pi::pi_c_accumulate_w(f, tk, Ge, mdl.X_skPa, mdl.Zns_qPQ, wp,
                                     mdl.kmq, mdl.kpq, C(), Pi_wq, 0, 1);
        nda::array<cplx, 3> kern(nk, Np, Np), fact(nk, Np, Np);
        for (long iq = 0; iq < nk; ++iq)
          for (long P = 0; P < Np; ++P)
            for (long Q = 0; Q < Np; ++Q) {
              cplx acc(0.0);
              for (long m = 0; m < nwb; ++m) acc += Re(m) * Pi_wq(m, iq, P, Q);
              kern(iq, P, Q) = acc;
            }
        fact() = cplx(0.0);
        vertex_pi::pi_dyn_factorized(tf, Ge, mdl.X_skPa, mdl.Zns_qPQ, wp,
                                     mdl.kmq, mdl.kpq, C(), Re, fact, 0, 1);
        double n, d;
        double r = rel_dev(fact, kern, n, d);
        app_log(1, "pibardynfact production_grid_attribution: beta = {}, wmax = {}, "
                   "eps = {:.0e}, rung = {:8s} -> nw_b = {}, abs = {:.3e}, |ref| = {:.3e}, "
                   "rel = {:.3e}", bta, wmx, eps, dynamic ? "dynamic" : "static", nwb, n, d, r);
        return std::make_pair(fact, kern);
      };
      auto [f6, k6] = run_both(1e-6, false);   // pole-free BOTH sides
      auto [fd6, kd6] = run_both(1e-6, true);  // reference goes through the pole algebra
      auto [f9, k9] = run_both(1e-9, false);
      auto [f12, k12] = run_both(1e-12, false);
      double n_, d_;
      const double s6 = rel_dev(f6, k6, n_, d_);
      const double d6 = rel_dev(fd6, kd6, n_, d_);
      const double s9 = rel_dev(f9, k9, n_, d_);
      const double s12 = rel_dev(f12, k12, n_, d_);
      app_log(1, "pibardynfact production_grid_attribution: eps = 1e-6 static (pole-free) = "
                 "{:.3e} vs dynamic (kernel uses poles) = {:.3e} -> pole excess factor "
                 "{:.2f}", s6, d6, (s6 > 0.0 ? d6 / s6 : 0.0));
      app_log(1, "pibardynfact production_grid_attribution: pole-free floor vs eps at "
                 "beta = 1000: 1e-6 -> {:.3e}, 1e-9 -> {:.3e}, 1e-12 -> {:.3e}",
              s6, s9, s12);

      // ---- WHICH ROUTE IS THE MORE ACCURATE ONE -------------------------------------------
      // The eps = 1e-12 pair agree to ~5e-9, so they share a common limit and either can
      // serve as the converged reference. Measuring EACH route's own drift from that limit
      // splits the disagreement above and settles which one carries it -- something no
      // route-vs-route comparison can do.
      auto drift = [&](nda::array<cplx, 3> const& A) {
        double n2, d2; return rel_dev(A, f12, n2, d2);
      };
      const double ef6 = drift(f6), ek6 = drift(k6);
      const double ef9 = drift(f9), ek9 = drift(k9);
      app_log(1, "pibardynfact production_grid_attribution: own error vs the eps = 1e-12 "
                 "limit -- factorized {:.3e} (1e-6) / {:.3e} (1e-9); kernel {:.3e} (1e-6) / "
                 "{:.3e} (1e-9)", ef6, ef9, ek6, ek9);
      // ---- MEASURED 2026-07-30 (beta = 1000, wmax = 6, static rung) -----------------------
      //   route-vs-route          eps = 1e-6  2.04e-3   1e-9  3.96e-5   1e-12  4.64e-9
      //   own error vs 1e-12      factorized  2.21e-3         4.08e-5
      //                           kernel      1.07e-3         1.97e-5
      //   pole excess factor (dynamic / static route-vs-route) = 0.80
      //
      // THREE CONCLUSIONS, all evidence-based:
      //
      // 1. The disagreement is a REPRESENTABILITY FLOOR, not a routing bug: it falls by
      //    ~5 orders as eps tightens by 6, and both routes converge to one common value.
      //    Its prefactor grows with beta*wmax (~30*eps at beta*wmax = 160, ~2000*eps at
      //    6000), which is why the beta = 20 toy showed 2.5e-5 and LiH-222 shows ~2e-3.
      //
      // 2. RETRACTION of the natural first guess. "The kernel's pole fit -- worst-case
      //    residue amplification ~1e6-1e7 on this grid -- must be what LiH's 3.6e-3 is made
      //    of" is WRONG. The excess factor is 0.80: the dynamic comparison is if anything
      //    slightly BETTER than the pole-free one. The twisted-pair pole algebra contributes
      //    essentially nothing here.
      //
      // 3. THE HONEST TRADE-OFF. The factorized route carries about TWICE the tau = 0
      //    discretization error of the kernel route at a given DLR tolerance (2.2e-3 vs
      //    1.1e-3 at eps = 1e-6). Both scale linearly in eps. So eq:pibardynfact buys ~300x
      //    on the dominant stage at the price of a factor ~2 in this quantity's grid error --
      //    and the pre-existing consequence, which is what actually matters, is that at
      //    prec = "low" pi^dyn is only good to ~1e-3 BY EITHER ROUTE. If B-L needs better,
      //    the lever is prec, not the route. Physics impact on LiH-222 is measured
      //    separately in test_vertex_static_e2e: the vertex shift moves ~3e-5 relative.
      REQUIRE(std::isfinite(s6));
      REQUIRE(std::isfinite(d6));
      REQUIRE(std::isfinite(ef6));
      REQUIRE(std::isfinite(ek6));
      // The floor must FALL with the basis tolerance -- that is what makes it a
      // representability floor rather than a routing bug (a bug is eps-independent).
      REQUIRE(s6 > s9);
      REQUIRE(s9 > s12);
      REQUIRE(s6 / s12 > 1e3);
      // at a converged basis the two routings agree to the level the beta = 20 scan reaches
      REQUIRE(s12 < 1e-7);
      // each route's own convergence, and the factor-2 trade-off. If the factorized route
      // ever drifts to >10x the kernel's error, conclusion 3 has changed and the DEFAULT
      // route needs revisiting -- that is what this assertion guards.
      REQUIRE(ef6 < 1e-2);
      REQUIRE(ek6 < 1e-2);
      REQUIRE(ef9 < 1e-3);
      REQUIRE(ek9 < 1e-3);
      REQUIRE(ef6 < 10.0 * ek6);
      REQUIRE(ef9 < 10.0 * ek9);
    }

    SECTION("no_pole_basis") {
      // O7's structural claim, made mechanical: the factorized primitive must never
      // initialize the auxiliary DLR pole basis. That is what removes B-L's only contact
      // with the parent project's open pole-conditioning defect (amplification 5.52e+06),
      // so it deserves an assertion rather than a comment.
      auto Wd = mdl.Wdyn(iaft_tools(ft));
      iaft_tools tools(ft);
      REQUIRE(tools.poles_ready == false);
      nda::array<cplx, 3> out(nk, Np, Np);
      out() = cplx(0.0);
      vertex_pi::pi_dyn_factorized(tools, G, mdl.X_skPa, mdl.Z_qPQ, &Wd,
                                   mdl.kmq, mdl.kpq, C(), R0, out, 0, 1);
      REQUIRE(tools.poles_ready == false);
      REQUIRE(tools.np == 0);
      double mx = 0.0;
      for (auto const& v : out) mx = std::max(mx, std::abs(v));
      REQUIRE(mx > 1e-10);
      // and the reference DOES build it -- so the assertion above is discriminating.
      iaft_tools tref(ft);
      nda::array<cplx, 4> Pi_wq(nw_b, nk, Np, Np);
      Pi_wq() = cplx(0.0);
      vertex_pi::pi_c_accumulate_w(ft, tref, G, mdl.X_skPa, mdl.Z_qPQ, &Wd,
                                   mdl.kmq, mdl.kpq, C(), Pi_wq, 0, 1);
      REQUIRE(tref.poles_ready == true);
      REQUIRE(tref.np > 0);
    }
#endif
  }

} // namespace bdft_tests
