/**
 * INCREMENT S5 -- the FUNCTIONAL-DERIVATIVE ORACLE for Formulation B-S.
 *
 * The parent theory's conservation test verified "two cuts of one Phi". B-S needs a
 * different statement, because its W-cut vanishes identically and ALL of the vertex
 * physics sits in the self-energy: what must be verified is "ONE TOTAL DERIVATIVE of one
 * Phi". For an arbitrary perturbation dG,
 *
 *   d/dl Phi_2^{C,0}[G + l dG]|_0 = T[ Sigma^{C,x}, P_C dG P_C ] + T[ Sigma^{C,r}, dG ]
 *
 * with T the pinned SAME-INDEX pairing. Sigma^{C,x} pairs on the C block (all eight
 * labels of Phi are in C, so the kernel's full-range extension is not dPhi/dG outside
 * it); Sigma^{C,r} pairs full-space (it is born in W0's RPA bubble, which carries no
 * projector). Phi itself is obtained from the Sigma^{C,x} kernel through the Euler
 * identity T[Sigma^{C,x}, G] = 4 Phi -- so no separate Phi evaluator is needed, and the
 * two sides of the oracle share one normalization by construction.
 *
 * This is the test with teeth for the S0 routing corrections
 * (verification/static_vertex_routing_report.md section 3): the notes' UNTRANSPOSED
 * W0 . Pi . W0 sandwich fails it by ~20 %, and dropping or sign-flipping Sigma^{C,r}
 * fails it by the response share.
 *
 * TOY DATA -- deliberately REALITY-SYMMETRIC (X(-k) = conj X(k), G(-k) = G(k)^T,
 * Z(-q) = Z(q)^T Hermitian), i.e. the symmetries a real crystal has. Rationale, recorded
 * so it is not silently re-litigated (plan, "S5 test-data ruling"):
 *   * a FULLY ASYMMETRIC toy would separate ^T from conj in the sandwich, but there the
 *     single-Eq.-(10)-pattern Sigma^x that CoQui implements is itself off by 17 %, so
 *     the oracle would fail for a reason unrelated to Sigma^{C,r};
 *   * production data IS symmetric, so ^T vs conj cannot change any physical result --
 *     that distinction is settled by the Python arbiter (verify_static_cuts.py) and
 *     needs no C++ re-derivation;
 *   * the correction that DOES change physics -- the untransposed sandwich -- is wrong
 *     by 20 % even on symmetric data, so this toy keeps full teeth against it.
 * dG is non-Hermitian (the parent's convention-discriminating control) while still
 * respecting the k-reality symmetry, which is what keeps Sigma^x exact.
 *
 * ns = 2 throughout: with one stored spin channel the RPA bubble and the Pi^C kernel
 * both carry an implicit degeneracy factor 2 and the chain rule would need matching
 * bookkeeping. Two explicit channels make every spin sum manifest.
 */

#undef NDEBUG

#include <complex>
#include <cmath>
#include <vector>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"

#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/vertex/vertex_t.h"
#include "methods/vertex/vertex_pi.icc"
#include "methods/vertex/vertex_sigma.icc"
#include "methods/vertex/vertex_sigma_r.icc"

namespace bdft_tests {

  using namespace methods;
  namespace vertex_pi = methods::solvers::vertex_pi;
  using vertex_pi::iaft_tools;
  using cplx = ComplexType;
  static auto const f_all = nda::range::all;

  namespace fdo {
    constexpr long nk = 3, nbnd = 3, ncw = 2, Np = 4, ns = 2;
    constexpr double beta = 20.0, wmax = 6.0;
    inline nda::range C() { return nda::range(0, ncw); }

    struct rng_t {
      unsigned long s;
      explicit rng_t(unsigned long seed) : s(seed) {}
      double u() {                       // deterministic LCG in (-1, 1)
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return 2.0 * (double((s >> 11) & ((1ULL << 53) - 1)) / double(1ULL << 53)) - 1.0;
      }
      cplx z() { return cplx(u(), u()); }
    };

    inline double g_tau(double eps, double tau) {   // -e^{-eps tau}(1 - nF(eps)), stable
      if (eps >= 0.0) return -std::exp(-eps * tau) / (1.0 + std::exp(-beta * eps));
      return -std::exp(eps * (beta - tau)) / (1.0 + std::exp(beta * eps));
    }

    inline long kminus(long k) { return (nk - k) % nk; }

    /**
     * Toy THC model carrying the REALITY symmetry. Built on a k half-set and mirrored:
     *   eps(-k) = eps(k),  Pr(-k) = conj Pr(k),  X(-k) = conj X(k),  Z(-q) = conj Z(q)
     * (self-inverse points made real). With a Hermitian per-k spectral projector this
     * gives G(-k, tau) = conj G(k, tau) = G(k, tau)^T exactly, as a real-space-real
     * Hamiltonian does.
     */
    struct model_t {
      nda::array<double, 3> eps;     // (ns, nk, nbnd)
      nda::array<cplx, 5> Pr;        // (ns, nk, nbnd[r], nbnd, nbnd)
      nda::array<cplx, 4> X_skPa;    // (ns, nk, Np, nbnd)
      nda::array<cplx, 3> Z_qPQ;     // (nq, Np, Np) Hermitian
      nda::array<long, 2> kmq, kpq;  // (nq, nk)
      nda::array<long, 1> qmin;      // (nq)

      model_t() : eps(ns, nk, nbnd), Pr(ns, nk, nbnd, nbnd, nbnd),
                  X_skPa(ns, nk, Np, nbnd), Z_qPQ(nk, Np, Np),
                  kmq(nk, nk), kpq(nk, nk), qmin(nk) {
        rng_t rng(4711);
        const double base[3] = {-0.61, -0.09, 0.47};
        // --- spectral data on the k half-set, then mirrored -------------------------
        for (long s = 0; s < ns; ++s)
          for (long k = 0; k < nk; ++k) {
            if (kminus(k) < k) continue;
            for (long r = 0; r < nbnd; ++r)
              eps(s, k, r) = base[r] + 0.05 * double(k) + 0.03 * double(s);
            // Hermitian rank-1 projectors from a random orthonormal-ish set
            nda::array<cplx, 2> U(nbnd, nbnd);
            for (long i = 0; i < nbnd; ++i)
              for (long j = 0; j < nbnd; ++j)
                U(i, j) = (kminus(k) == k) ? cplx(rng.u(), 0.0) : rng.z();
            // Gram-Schmidt
            for (long r = 0; r < nbnd; ++r) {
              for (long t = 0; t < r; ++t) {
                cplx ov(0.0);
                for (long i = 0; i < nbnd; ++i) ov += std::conj(U(i, t)) * U(i, r);
                for (long i = 0; i < nbnd; ++i) U(i, r) -= ov * U(i, t);
              }
              double nn = 0.0;
              for (long i = 0; i < nbnd; ++i) nn += std::norm(U(i, r));
              nn = std::sqrt(nn);
              for (long i = 0; i < nbnd; ++i) U(i, r) /= nn;
            }
            for (long r = 0; r < nbnd; ++r)
              for (long i = 0; i < nbnd; ++i)
                for (long j = 0; j < nbnd; ++j)
                  Pr(s, k, r, i, j) = U(i, r) * std::conj(U(j, r));
            const long km = kminus(k);
            if (km != k) {
              for (long r = 0; r < nbnd; ++r) {
                eps(s, km, r) = eps(s, k, r);
                for (long i = 0; i < nbnd; ++i)
                  for (long j = 0; j < nbnd; ++j)
                    Pr(s, km, r, i, j) = std::conj(Pr(s, k, r, i, j));
              }
            }
          }
        // --- collocation: X(-k) = conj X(k), X(self-inverse) real -------------------
        for (long k = 0; k < nk; ++k) {
          if (kminus(k) < k) continue;
          const long km = kminus(k);
          for (long P = 0; P < Np; ++P)
            for (long a = 0; a < nbnd; ++a) {
              cplx x = (km == k) ? cplx(rng.u(), 0.0) : rng.z();
              for (long s = 0; s < ns; ++s) {
                X_skPa(s, k, P, a) = x;
                if (km != k) X_skPa(s, km, P, a) = std::conj(x);
              }
            }
        }
        // --- bare core: Hermitian, Z(-q) = conj Z(q) --------------------------------
        for (long q = 0; q < nk; ++q) {
          if (kminus(q) < q) continue;
          const long qm = kminus(q);
          nda::array<cplx, 2> Y(Np, Np);
          for (long P = 0; P < Np; ++P)
            for (long r = 0; r < Np; ++r) Y(P, r) = (qm == q) ? cplx(rng.u(), 0.0) : rng.z();
          for (long P = 0; P < Np; ++P)
            for (long Q = 0; Q < Np; ++Q) {
              cplx zy(0.0);
              for (long r = 0; r < Np; ++r) zy += Y(P, r) * std::conj(Y(Q, r));
              cplx v = zy / double(Np) + ((P == Q) ? cplx(0.30) : cplx(0.0));
              Z_qPQ(q, P, Q) = v;
              if (qm != q) Z_qPQ(qm, P, Q) = std::conj(v);
            }
        }
        for (long q = 0; q < nk; ++q) {
          qmin(q) = kminus(q);
          for (long k = 0; k < nk; ++k) {
            kmq(q, k) = (k - q + nk) % nk;
            kpq(q, k) = (k + q) % nk;
          }
        }
      }

      nda::array<cplx, 5> G_tau(imag_axes_ft::IAFT const &ft) const {
        const long nt = ft.nt_f();
        auto xm = ft.tau_mesh();
        nda::array<cplx, 5> G(nt, ns, nk, nbnd, nbnd);
        G() = cplx(0.0);
        for (long it = 0; it < nt; ++it) {
          const double tau = (xm(it) + 1.0) * 0.5 * beta;
          for (long s = 0; s < ns; ++s)
            for (long k = 0; k < nk; ++k)
              for (long r = 0; r < nbnd; ++r) {
                const double g = g_tau(eps(s, k, r), tau);
                for (long i = 0; i < nbnd; ++i)
                  for (long j = 0; j < nbnd; ++j)
                    G(it, s, k, i, j) += g * Pr(s, k, r, i, j);
              }
        }
        return G;
      }
    };
  } // fdo

  // ======================================================================================
  TEST_CASE("vertex_fdoracle_bs", "[methods][vertex][fdoracle]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_fdoracle_bs skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace fdo;
    auto &mpi_context = utils::make_unit_test_mpi_context();
    auto comm = mpi_context->comm;

    // The oracle's tolerance class is BASIS-eps (plan, S5 gate): Phi is a product of
    // three/four tau-objects, so the residual is set by how well the basis represents
    // those products, not by machine precision. Swept so that dependence is on record.
    std::string prec = GENERATE(std::string("medium"), std::string("high"));
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, prec);
    iaft_tools tools(ft);
    const long nt = ft.nt_f(), nw_b = ft.nw_b();
    model_t mdl;
    auto G0 = mdl.G_tau(ft);

    // sanity: the toy really carries the reality symmetry G(-k) = G(k)^T
    {
      double d = 0.0, sc = 0.0;
      for (long it = 0; it < nt; ++it)
        for (long s = 0; s < ns; ++s)
          for (long k = 0; k < nk; ++k)
            for (long a = 0; a < nbnd; ++a)
              for (long b = 0; b < nbnd; ++b) {
                d = std::max(d, std::abs(G0(it, s, kminus(k), a, b) - G0(it, s, k, b, a)));
                sc = std::max(sc, std::abs(G0(it, s, k, a, b)));
              }
      REQUIRE(sc > 1e-8);
      REQUIRE(d < 1e-12 * sc);
    }

    // --- integral over tau: int_0^beta dtau f(tau) = f(i.nu = 0) ------------------------
    auto tau_integral = [&](nda::array<cplx, 1> const &f_t) {
      cplx acc(0.0);
      for (long it = 0; it < nt; ++it) acc += tools.Twt_bb(tools.m0, it) * f_t(it);
      return acc;
    };

    // --- aux-dressed FULL-space propagator ----------------------------------------------
    auto dress_all = [&](nda::array<cplx, 5> const &G) {
      nda::array<cplx, 5> Gt(nt, ns, nk, Np, Np);
      Gt() = cplx(0.0);
      for (long it = 0; it < nt; ++it)
        for (long s = 0; s < ns; ++s)
          for (long k = 0; k < nk; ++k)
            for (long P = 0; P < Np; ++P)
              for (long Q = 0; Q < Np; ++Q) {
                cplx acc(0.0);
                for (long a = 0; a < nbnd; ++a)
                  for (long b = 0; b < nbnd; ++b)
                    acc += mdl.X_skPa(s, k, P, a) * G(it, s, k, a, b)
                           * std::conj(mdl.X_skPa(s, k, Q, b));
                Gt(it, s, k, P, Q) = acc;
              }
      return Gt;
    };

    // --- W0[G] = [1 - Z P0]^{-1} Z at i.nu = 0 -------------------------------------------
    //  P0_PQ(q) = sum_s (1/(Nk beta)) sum_{k,w} Gt_PQ(k+q,iw) Gt_QP(k,iw)
    //           = -(1/Nk) sum_{s,k} int dtau Gt_PQ(k+q,tau) Gt_QP(k, beta-tau)
    //  (the pairing identity -- NOT a sum over the sampled Matsubara nodes).
    nda::array<cplx, 3> P0_keep(nk, Np, Np);
    auto W0_from_G = [&](nda::array<cplx, 5> const &G) {
      auto Gt = dress_all(G);
      nda::array<cplx, 3> P0(nk, Np, Np), W0(nk, Np, Np);
      P0() = cplx(0.0);
      nda::array<cplx, 1> f_t(nt);
      for (long q = 0; q < nk; ++q)
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            f_t() = cplx(0.0);
            for (long s = 0; s < ns; ++s)
              for (long k = 0; k < nk; ++k) {
                const long kpq = mdl.kpq(q, k);
                for (long it = 0; it < nt; ++it)
                  f_t(it) += Gt(it, s, kpq, P, Q) * Gt(tools.t_mirror(it), s, k, Q, P);
              }
            P0(q, P, Q) = -tau_integral(f_t) / double(nk);
          }
      // one-frequency Dyson per q
      for (long q = 0; q < nk; ++q) {
        nda::matrix<cplx> A(Np, Np), B(Np, Np);
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            cplx zp(0.0);
            for (long r = 0; r < Np; ++r) zp += mdl.Z_qPQ(q, P, r) * P0(q, r, Q);
            A(P, Q) = ((P == Q) ? cplx(1.0) : cplx(0.0)) - zp;
            B(P, Q) = mdl.Z_qPQ(q, P, Q);
          }
        auto Ainv = nda::inverse(A);
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            cplx acc(0.0);
            for (long r = 0; r < Np; ++r) acc += Ainv(P, r) * B(r, Q);
            W0(q, P, Q) = acc;
          }
      }
      P0_keep = P0;
      return W0;
    };

    // --- Sigma^{C,x}: the static-rung path of the pinned kernel --------------------------
    auto sigma_x = [&](nda::array<cplx, 5> const &G, nda::array<cplx, 3> const &W0) {
      nda::array<cplx, 4> Wstub(nk, 0, Np, Np);
      nda::array<cplx, 5> S(nt, ns, nk, nbnd, nbnd);
      solvers::vertex_detail::eval_sigma_C_g3w2(ft, comm, C(), G, mdl.X_skPa, Wstub, W0,
                                                mdl.kmq, mdl.qmin, /*iq_gamma*/ 0,
                                                /*skip_rung_gamma*/ false,
                                                /*rung_mode*/ 1, nullptr, nullptr, S);
      return S;
    };

    // --- Pi^{C,0}(q, tau = 0): the Z-phase with the rung W0, then the tau=0 row ----------
    auto piC0_tau0 = [&](nda::array<cplx, 5> const &G, nda::array<cplx, 3> const &W0) {
      // The Pi kernel CONTRACTS its external orbital legs into the aux indices, so the
      // range of those legs is part of the object. All eight labels of Phi are in C, so
      // the cut's externals are in C too: the kernel must be fed the C-C block of G and
      // the C columns of X -- exactly what vertex_t::eval_Pi_C does (it passes G_CC of
      // shape (nt, ns, nk, nc, nc)). Feeding the FULL G computes the kernel's EXTENDED
      // object, which is NOT the rung derivative of Phi.
      // (Contrast Sigma^x, whose externals stay FREE: there the full-range extension is
      //  harmless because we simply pair on the C block afterwards.)
      nda::array<cplx, 5> G_CC(nt, ns, nk, ncw, ncw);
      nda::array<cplx, 4> X_C(ns, nk, Np, ncw);
      for (long it = 0; it < nt; ++it)
        for (long s = 0; s < ns; ++s)
          for (long k = 0; k < nk; ++k)
            for (long a = 0; a < ncw; ++a)
              for (long b = 0; b < ncw; ++b) G_CC(it, s, k, a, b) = G(it, s, k, a, b);
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < nk; ++k)
          for (long P = 0; P < Np; ++P)
            for (long a = 0; a < ncw; ++a) X_C(s, k, P, a) = mdl.X_skPa(s, k, P, a);

      nda::array<cplx, 4> Pi_wq(nw_b, nk, Np, Np);
      Pi_wq() = cplx(0.0);
      vertex_pi::pi_c_accumulate_w(ft, tools, G_CC, X_C, W0, /*Wdyn*/ nullptr,
                                   mdl.kmq, mdl.kpq, nda::range(0, ncw), Pi_wq,
                                   comm.rank(), comm.size());
      comm.all_reduce_in_place_n(Pi_wq.data(), Pi_wq.size(), std::plus<>{});
      auto R = solvers::vertex_w0_detail::tau0_transform_row(ft);
      nda::array<cplx, 3> Pi0(nk, Np, Np);
      Pi0() = cplx(0.0);
      for (long q = 0; q < nk; ++q)
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            cplx acc(0.0);
            for (long m = 0; m < nw_b; ++m) acc += R(m) * Pi_wq(m, q, P, Q);
            Pi0(q, P, Q) = acc;
          }
      return Pi0;
    };

    // --- the pinned SAME-INDEX pairing, evaluated in tau ---------------------------------
    //   T[A,B] = (1/(Nk beta)) sum_{s,k,w,ab} A_ab B_ab
    //          = -(1/Nk) sum_{s,k,ab} int dtau A_ab(tau) B_ab(beta - tau)
    auto pairing = [&](nda::array<cplx, 5> const &A, nda::array<cplx, 5> const &B,
                       long nrow, long ncol) {
      cplx tot(0.0);
      nda::array<cplx, 1> f_t(nt);
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < nk; ++k)
          for (long a = 0; a < nrow; ++a)
            for (long b = 0; b < ncol; ++b) {
              for (long it = 0; it < nt; ++it)
                f_t(it) = A(it, s, k, a, b) * B(tools.t_mirror(it), s, k, a, b);
              tot += -tau_integral(f_t) / double(nk);
            }
      return tot;
    };

    // Phi via the Euler identity T[Sigma^x, G] = 4 Phi (pairing on the C block)
    auto phi_of = [&](nda::array<cplx, 5> const &G) {
      auto W0 = W0_from_G(G);
      auto Sx = sigma_x(G, W0);
      return 0.25 * pairing(Sx, G, ncw, ncw);
    };

    // --- the perturbation: NON-Hermitian, but respecting the k-reality symmetry ----------
    // dG must be a GENUINE tau-FUNCTION, not per-node noise. Random values on the sparse
    // nodes are not representable in the basis at all -- they are mesh-dependent, so both
    // Phi and its derivative would change when the DLR precision changes (measured: the
    // residual got WORSE from medium to high, and dPhi/dl itself moved by 60 %). Build it
    // instead from the same pole machinery as G, with DIFFERENT poles and arbitrary
    // (NON-Hermitian) amplitudes -- representable to eps and mesh-independent.
    nda::array<cplx, 5> dG(nt, ns, nk, nbnd, nbnd);
    {
      rng_t rg(90210);
      const double de[3] = {-0.33, 0.19, 0.71};
      nda::array<cplx, 5> Amp(ns, nk, nbnd, nbnd, nbnd);   // (s,k,r,a,b), non-Hermitian
      Amp() = cplx(0.0);
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < nk; ++k) {
          if (kminus(k) < k) continue;
          const long km = kminus(k);
          for (long r = 0; r < nbnd; ++r)
            for (long a = 0; a < nbnd; ++a)
              for (long b = 0; b < nbnd; ++b) {
                if (km == k) {                 // self-inverse k: SYMMETRIC (not Hermitian)
                  if (b < a) continue;
                  cplx v = rg.z();
                  Amp(s, k, r, a, b) = v;
                  Amp(s, k, r, b, a) = v;
                } else {
                  cplx v = rg.z();
                  Amp(s, k, r, a, b) = v;
                  Amp(s, km, r, b, a) = v;     // dG(-k) = dG(k)^T
                }
              }
        }
      auto xm = ft.tau_mesh();
      dG() = cplx(0.0);
      for (long it = 0; it < nt; ++it) {
        const double tau = (xm(it) + 1.0) * 0.5 * beta;
        for (long s = 0; s < ns; ++s)
          for (long k = 0; k < nk; ++k)
            for (long r = 0; r < nbnd; ++r) {
              const double g = g_tau(de[r], tau);
              for (long a = 0; a < nbnd; ++a)
                for (long b = 0; b < nbnd; ++b)
                  dG(it, s, k, a, b) += g * Amp(s, k, r, a, b);
            }
      }
      // confirm it really is non-Hermitian (the convention-discriminating control)
      double herm = 0.0, sc = 0.0;
      for (long it = 0; it < nt; ++it)
        for (long s = 0; s < ns; ++s)
          for (long k = 0; k < nk; ++k)
            for (long a = 0; a < nbnd; ++a)
              for (long b = 0; b < nbnd; ++b) {
                herm = std::max(herm, std::abs(dG(it, s, k, a, b)
                                               - std::conj(dG(it, s, k, b, a))));
                sc = std::max(sc, std::abs(dG(it, s, k, a, b)));
              }
      REQUIRE(herm > 0.1 * sc);      // genuinely non-Hermitian
    }

    auto shifted = [&](double lam) {
      nda::array<cplx, 5> G(nt, ns, nk, nbnd, nbnd);
      for (long i = 0; i < G.size(); ++i) G.data()[i] = G0.data()[i] + lam * dG.data()[i];
      return G;
    };

    // --- the oracle ----------------------------------------------------------------------
    // h sweep: a centered difference converges as h^2, so if the residual falls by ~100x
    // per decade it is TRUNCATION (dG is not small compared with G's tau decay), not a
    // routing error. Reported before the gate so the distinction is on the record.
    const double h = 1e-6;
    const cplx dphi_fd = (phi_of(shifted(h)) - phi_of(shifted(-h))) / (2.0 * h);

    auto W0 = W0_from_G(G0);
    auto Sx = sigma_x(G0, W0);
    auto Pi0 = piC0_tau0(G0, W0);

    nda::array<cplx, 3> Dw(nk, Np, Np);
    solvers::vertex_detail::build_delta_w(W0, Pi0, mdl.qmin, Dw);
    nda::array<cplx, 5> Sr(nt, ns, nk, nbnd, nbnd);
    solvers::vertex_detail::eval_sigma_C_response(comm, G0, mdl.X_skPa, Dw, mdl.kmq,
                                                  mdl.qmin, Sr);

    const cplx t_x = pairing(Sx, dG, ncw, ncw);      // C-block pairing (P_C dG P_C)
    const cplx t_r = pairing(Sr, dG, nbnd, nbnd);    // full-space pairing
    const cplx pred = t_x + t_r;

    // ---- DIAGNOSTIC SPLIT: freeze W0 so only the explicit lines vary. Its derivative
    //      must be exactly T[Sigma^x, P_C dG P_C]; the remainder must be the response.
    auto phi_frozen = [&](nda::array<cplx, 5> const &G) {
      auto Sxf = sigma_x(G, W0);
      return 0.25 * pairing(Sxf, G, ncw, ncw);
    };
    const cplx dphi_expl = (phi_frozen(shifted(h)) - phi_frozen(shifted(-h))) / (2.0 * h);
    // symmetry audit of the objects Sigma^x's exactness depends on
    double sw = 0.0, swsc = 0.0, sp = 0.0, spsc = 0.0;
    for (long q = 0; q < nk; ++q)
      for (long P = 0; P < Np; ++P)
        for (long Q = 0; Q < Np; ++Q) {
          sw = std::max(sw, std::abs(W0(q, P, Q) - W0(mdl.qmin(q), Q, P)));
          swsc = std::max(swsc, std::abs(W0(q, P, Q)));
          sp = std::max(sp, std::abs(Pi0(q, P, Q) - Pi0(mdl.qmin(q), Q, P)));
          spsc = std::max(spsc, std::abs(Pi0(q, P, Q)));
        }
    app_log(1, "vertex_fdoracle_bs SPLIT: T[Sx,dG_C] = {:.10g}{:+.10g}i, "
               "dPhi_frozenW0/dl = {:.10g}{:+.10g}i  (rel {:.3e});  "
               "T[Sr,dG] = {:.10g}{:+.10g}i, remainder = {:.10g}{:+.10g}i  (ratio {:.6f})",
            t_x.real(), t_x.imag(), dphi_expl.real(), dphi_expl.imag(),
            std::abs(t_x - dphi_expl) / std::abs(dphi_expl),
            t_r.real(), t_r.imag(), (dphi_fd - dphi_expl).real(),
            (dphi_fd - dphi_expl).imag(),
            std::abs(t_r) / std::abs(dphi_fd - dphi_expl));
    app_log(1, "vertex_fdoracle_bs SYM: |W0(q)-W0(-q)^T| = {:.3e} (scale {:.3e}), "
               "|Pi0(q)-Pi0(-q)^T| = {:.3e} (scale {:.3e})", sw, swsc, sp, spsc);

    const double rel = std::abs(dphi_fd - pred) / std::abs(dphi_fd);
    app_log(1, "vertex_fdoracle_bs [prec = {}]: dPhi/dl (FD) = {:.10g}{:+.10g}i, "
               "Sigma^x.dG + Sigma^r.dG = {:.10g}{:+.10g}i, rel = {:.3e}; "
               "response share |Sigma^r.dG|/|Sigma^x.dG| = {:.4f}",
            prec, dphi_fd.real(), dphi_fd.imag(), pred.real(), pred.imag(), rel,
            std::abs(t_r) / std::abs(t_x));
    REQUIRE(std::abs(dphi_fd) > 1e-10);
    REQUIRE(std::abs(t_r) > 1e-3 * std::abs(t_x));   // the response term must MATTER
    REQUIRE(rel < 1e-7);                             // ** THE TEST WITH TEETH **
    // The SPLIT is asserted too, so a failure says WHICH cut is wrong:
    //   the explicit lines alone (W0 frozen) must reproduce T[Sigma^x, P_C dG P_C], and
    //   the remainder must be exactly T[Sigma^{C,r}, dG].
    REQUIRE(std::abs(t_x - dphi_expl) / std::abs(dphi_expl) < 1e-7);
    REQUIRE(std::abs(t_r - (dphi_fd - dphi_expl)) / std::abs(t_r) < 1e-7);
    // mesh-independence: dPhi/dl is a property of the FUNCTIONAL, so it must not move
    // with the DLR precision. (It does if dG is per-node noise instead of a representable
    // tau-function -- which is exactly how the first draft of this test was wrong.)

    // --- the FOLDED single-transfer form must be IDENTICAL -------------------------------
    // sum_q [Dw(q)_AB Gt(k-q)_BA + Dw(q)_BA Gt(k+q)_BA] = sum_q [Dw(q)+Dw(-q)^T]_AB Gt(k-q)_BA
    // This is what turns Sigma^{C,r} into a Sigma^GW-shaped kernel (ONE two-body object,
    // read at IBZ q only; ONE Gt leg) and hence what makes the symmetry-adapted path a
    // near-copy of gw_t::eval_Sigma_all_k_impl.
    {
      nda::array<cplx, 3> Dw_eff(nk, Np, Np);
      solvers::vertex_detail::fold_delta_w(Dw, mdl.qmin, Dw_eff);
      nda::array<cplx, 5> Sr_f(nt, ns, nk, nbnd, nbnd);
      solvers::vertex_detail::eval_sigma_C_response_folded(comm, G0, mdl.X_skPa, Dw_eff,
                                                           mdl.kmq, Sr_f);
      double num = 0.0, den = 0.0;
      for (long it = 0; it < nt; ++it)
        for (long s = 0; s < ns; ++s)
          for (long k = 0; k < nk; ++k)
            for (long a = 0; a < nbnd; ++a)
              for (long b = 0; b < nbnd; ++b) {
                num = std::max(num, std::abs(Sr_f(it, s, k, a, b) - Sr(it, s, k, a, b)));
                den = std::max(den, std::abs(Sr(it, s, k, a, b)));
              }
      app_log(1, "vertex_fdoracle_bs: folded single-transfer Sigma^(C,r) vs the +-q pair: "
                 "max|d| = {:.3e} (scale {:.3e}, rel {:.3e})", num, den, num / den);
      REQUIRE(den > 1e-12);
      REQUIRE(num < 1e-13 * den);
    }

    // --- EULER IDENTITIES (i)-(iii)  [O6] ------------------------------------------------
    // These tie the THREE kernels together (Sigma^x, Pi^{C,0}, Sigma^{C,r}) through one
    // Phi, so they are cross-kernel checks, not restatements of the oracle. Trace
    // DIRECTIONS are the corrected ones of the S0 report section 3.3.
    const cplx phi_hat = 0.25 * pairing(Sx, G0, ncw, ncw);
    {
      // (i) Phi is degree-4 HOMOGENEOUS in the explicit G's (the content of euler1; the
      //     identity T[Sigma^x, G] = 4 Phi is definitional once Phi is read off it).
      const double c = 1.37;
      nda::array<cplx, 5> Gs(nt, ns, nk, nbnd, nbnd);
      for (long i = 0; i < G0.size(); ++i) Gs.data()[i] = c * G0.data()[i];
      const cplx phi_c = 0.25 * pairing(sigma_x(Gs, W0), Gs, ncw, ncw);   // W0 FROZEN
      const double rel1 = std::abs(phi_c - std::pow(c, 4) * phi_hat) / std::abs(phi_c);
      app_log(1, "vertex_fdoracle_bs euler(i): Phi[cG]/Phi[G] = {:.10f} vs c^4 = {:.10f} "
                 "(rel {:.3e})", std::abs(phi_c / phi_hat), std::pow(c, 4), rel1);
      REQUIRE(rel1 < 1e-9);
    }
    {
      // (ii) rung Euler:  (1/Nk) sum_q sum_IJ Pi^{C,0}(q,tau=0)_IJ Wt_IJ(q) = -4 Phi,
      //      Wt(q) = 1/2 [W0(q)^T + W0(-q)]   (the beta of PiBar = beta Pi(tau=0) cancels)
      cplx e2(0.0);
      for (long q = 0; q < nk; ++q)
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            const cplx Wt = 0.5 * (W0(q, Q, P) + W0(mdl.qmin(q), P, Q));
            e2 += Pi0(q, P, Q) * Wt;
          }
      e2 /= double(nk);
      const double rel2 = std::abs(e2 + 4.0 * phi_hat) / std::abs(4.0 * phi_hat);
      // the notes' UNCORRECTED direction (plain Pi_IJ W0_IJ) must NOT satisfy it
      cplx e2n(0.0);
      for (long q = 0; q < nk; ++q)
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) e2n += Pi0(q, P, Q) * W0(q, P, Q);
      e2n /= double(nk);
      app_log(1, "vertex_fdoracle_bs euler(ii): sum Pi.Wt = {:.10g}{:+.10g}i vs -4 Phi = "
                 "{:.10g}{:+.10g}i (rel {:.3e}); notes' plain Pi_IJ W0_IJ direction: "
                 "rel {:.3e}", e2.real(), e2.imag(), (-4.0 * phi_hat).real(),
              (-4.0 * phi_hat).imag(), rel2,
              std::abs(e2n + 4.0 * phi_hat) / std::abs(4.0 * phi_hat));
      REQUIRE(rel2 < 1e-9);
    }
    {
      // (iii) response trace:  T[Sigma^r, G] = -(1/Nk) sum_q sum_IJ Psym_IJ [W0 P0 W0]_IJ
      cplx e3(0.0);
      for (long q = 0; q < nk; ++q) {
        nda::array<cplx, 2> WPW(Np, Np), tmp(Np, Np);
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            cplx acc(0.0);
            for (long r = 0; r < Np; ++r) acc += W0(q, P, r) * P0_keep(q, r, Q);
            tmp(P, Q) = acc;
          }
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            cplx acc(0.0);
            for (long r = 0; r < Np; ++r) acc += tmp(P, r) * W0(q, r, Q);
            WPW(P, Q) = acc;
          }
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            const cplx Psym = 0.5 * (Pi0(q, Q, P) + Pi0(mdl.qmin(q), P, Q));
            e3 += Psym * WPW(P, Q);
          }
      }
      e3 = -e3 / double(nk);
      const cplx lhs3 = pairing(Sr, G0, nbnd, nbnd);
      const double rel3 = std::abs(lhs3 - e3) / std::abs(lhs3);
      app_log(1, "vertex_fdoracle_bs euler(iii): T[Sr,G] = {:.10g}{:+.10g}i vs "
                 "-(1/Nk) sum Psym.[W0 P0 W0] = {:.10g}{:+.10g}i (rel {:.3e})",
              lhs3.real(), lhs3.imag(), e3.real(), e3.imag(), rel3);
      REQUIRE(rel3 < 1e-9);
    }

    // --- positive controls ---------------------------------------------------------------
    // (a) drop Sigma^r
    REQUIRE(std::abs(dphi_fd - t_x) / std::abs(dphi_fd) > 1e-3);
    // (b) sign-flip Sigma^r -> breaks by twice the response share
    REQUIRE(std::abs(dphi_fd - (t_x - t_r)) / std::abs(dphi_fd) > 1e-3);
    // (c) the NOTES' sandwich: untransposed W0 . Pi(q) . W0, unsymmetrized. This is the
    //     S0 correction; it must fail even on this symmetric toy.
    {
      nda::array<cplx, 3> Dw_notes(nk, Np, Np);
      for (long q = 0; q < nk; ++q) {
        nda::array<cplx, 2> tmp(Np, Np);
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            cplx acc(0.0);
            for (long r = 0; r < Np; ++r) acc += W0(q, P, r) * Pi0(q, r, Q);
            tmp(P, Q) = acc;
          }
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            cplx acc(0.0);
            for (long r = 0; r < Np; ++r) acc += tmp(P, r) * W0(q, r, Q);
            Dw_notes(q, P, Q) = acc;
          }
      }
      nda::array<cplx, 5> Sr_n(nt, ns, nk, nbnd, nbnd);
      solvers::vertex_detail::eval_sigma_C_response(comm, G0, mdl.X_skPa, Dw_notes,
                                                    mdl.kmq, mdl.qmin, Sr_n);
      const cplx pred_n = t_x + pairing(Sr_n, dG, nbnd, nbnd);
      const double rel_n = std::abs(dphi_fd - pred_n) / std::abs(dphi_fd);
      app_log(1, "vertex_fdoracle_bs control (c): the notes' untransposed/unsymmetrized "
                 "W0.Pi.W0 sandwich gives rel = {:.3e}", rel_n);
      REQUIRE(rel_n > 1e-3);
    }
#endif
  }

} // bdft_tests
