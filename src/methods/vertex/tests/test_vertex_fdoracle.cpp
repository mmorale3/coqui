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
     * A q-resolved rung matrix carrying the symmetry EVERY rung object of this kernel must
     * have: Hermitian per q, AND
     *
     *     M_{PQ}(q) = M_{QP}(-q).
     *
     * That second condition BINDS at a self-inverse transfer (q = -q: Gamma, and the
     * zone-boundary points of an even mesh). There it forces M(q) to be REAL SYMMETRIC,
     * not merely Hermitian. Writing the +q and -q blocks in a single pass silently
     * violates it exactly there -- both writes land on the same element and the last one
     * wins -- which is how the B-L oracles came to be fed an ILLEGAL rung.
     *
     * Why it matters, and why nothing else notices: Sigma^C is the Eq.-(10) pattern, i.e.
     * ONE of the four ways to cut a G line out of Phi_2. It equals dPhi/dG only because
     * the diagram's C4 rotation makes all four cuts equal -- and that rotation TRANSPOSES
     * a rung's (row pair, col pair). So the four cuts are equal iff the rung obeys the
     * relation above. Feed a rung that does not and the G-side oracle breaks by O(10%)
     * with no other symptom: the W-side oracle, the Euler identities and the
     * Sigma-vs-reference pins are all blind to it.
     *
     * The model's own Z_qPQ has always been built correctly (it uses a REAL draw at the
     * self-inverse q); only the B-L perturbations did not. Use rung_sym_err() to assert it.
     */
    inline nda::array<cplx, 3> make_rung(unsigned long seed, double scale) {
      rng_t rg(seed);
      nda::array<cplx, 3> M(nk, Np, Np);
      M() = cplx(0.0);
      for (long q = 0; q < nk; ++q) {
        if (kminus(q) < q) continue;
        const long qm = kminus(q);
        const bool self = (qm == q);
        for (long P = 0; P < Np; ++P)
          for (long Q = P; Q < Np; ++Q) {
            // Hermitian diagonal is real; a self-inverse q is real SYMMETRIC throughout
            cplx v = scale * ((self or P == Q) ? cplx(rg.u(), 0.0) : rg.z());
            M(q, P, Q) = v;   M(q, Q, P) = std::conj(v);
            if (not self) { M(qm, Q, P) = v;  M(qm, P, Q) = std::conj(v); }
          }
      }
      return M;
    }

    /** max |M_{PQ}(q) - M_{QP}(-q)|, relative to max|M|. Must be ~0 for a legal rung. */
    inline double rung_sym_err(nda::array<cplx, 3> const &M) {
      double d = 0.0, sc = 0.0;
      for (long q = 0; q < nk; ++q)
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            d = std::max(d, std::abs(M(q, P, Q) - M(kminus(q), Q, P)));
            sc = std::max(sc, std::abs(M(q, P, Q)));
          }
      return (sc > 0.0) ? d / sc : 0.0;
    }

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
                                                /*rung_mode*/ 1, static_cast<nda::array<ComplexType, 4> const*>(nullptr), nullptr, S);
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
      vertex_pi::pi_c_accumulate_w(ft, tools, G_CC, X_C, W0, static_cast<nda::array<ComplexType, 4> const*>(nullptr),
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

    // --- T-4: THE ORACLE UNDER A COMPLEX WANNIER GAUGE (plan note, erratum E-S2) --------
    // Feed the SAME functional through the production Wannier substitutions with a
    // FULL-RANK complex unitary V on the window (range(P) unchanged, so Phi is the same
    // functional): Xbar = X(:,C) V, Gbar = V^dag G_CC V, kernel at strict C-C. The
    // conserving pairing is stated in the KERNEL'S OWN (Wannier) labels,
    // T[Sbar^x, V^dag dG_CC V]; with the CHAIN-RULE injection this equals the band-label
    // pairing exactly (associativity), so the oracle gate carries over unchanged. The
    // OPERATOR-sandwich reading (V Sbar V^dag paired against dG_CC) is the documented
    // trap (wannier_projector_theory section 2.7) and must break at O(1).
    {
      const long M = ncw;
      nda::array<cplx, 2> V(M, M);
      {
        rng_t rg(777);
        for (long a = 0; a < M; ++a)
          for (long b = 0; b < M; ++b) V(a, b) = rg.z();
        for (long b = 0; b < M; ++b) {                 // Gram-Schmidt -> unitary
          for (long c = 0; c < b; ++c) {
            cplx ip(0.0);
            for (long a = 0; a < M; ++a) ip += std::conj(V(a, c)) * V(a, b);
            for (long a = 0; a < M; ++a) V(a, b) -= ip * V(a, c);
          }
          double nrm = 0.0;
          for (long a = 0; a < M; ++a) nrm += std::norm(V(a, b));
          nrm = std::sqrt(nrm);
          for (long a = 0; a < M; ++a) V(a, b) /= nrm;
        }
      }
      auto rot_G = [&](nda::array<cplx, 5> const& G) {   // Gbar = V^dag G_CC V
        nda::array<cplx, 5> Gb(nt, ns, nk, M, M);
        Gb() = cplx(0.0);
        for (long it = 0; it < nt; ++it)
          for (long s = 0; s < ns; ++s)
            for (long k = 0; k < nk; ++k)
              for (long a = 0; a < M; ++a)
                for (long b = 0; b < M; ++b) {
                  cplx acc(0.0);
                  for (long i = 0; i < M; ++i)
                    for (long j = 0; j < M; ++j)
                      acc += std::conj(V(i, a)) * G(it, s, k, i, j) * V(j, b);
                  Gb(it, s, k, a, b) = acc;
                }
        return Gb;
      };
      nda::array<cplx, 4> Xbar(ns, nk, Np, M);
      Xbar() = cplx(0.0);
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < nk; ++k)
          for (long P = 0; P < Np; ++P)
            for (long a = 0; a < M; ++a)
              for (long j = 0; j < M; ++j)
                Xbar(s, k, P, a) += mdl.X_skPa(s, k, P, j) * V(j, a);
      nda::array<cplx, 4> Wstub_w(nk, 0, Np, Np);
      nda::array<cplx, 5> Sxb(nt, ns, nk, M, M);
      solvers::vertex_detail::eval_sigma_C_g3w2(ft, comm, nda::range(0, M), rot_G(G0),
                                                Xbar, Wstub_w, W0, mdl.kmq, mdl.qmin,
                                                /*iq_gamma*/ 0, /*skip*/ false,
                                                /*rung_mode*/ 1,
                                                static_cast<nda::array<ComplexType, 4> const*>(nullptr),
                                                nullptr, Sxb);
      // (1) Phi evaluated in the kernel's own Wannier labels == the band-label Phi
      const cplx phi_w = 0.25 * pairing(Sxb, rot_G(G0), M, M);
      // (2) the WANNIER-label explicit pairing reproduces the band one exactly, and the
      //     full oracle gate carries over
      const cplx t_x_w = pairing(Sxb, rot_G(dG), M, M);
      const double rel_w = std::abs(dphi_fd - (t_x_w + t_r)) / std::abs(dphi_fd);
      // (3) POSITIVE CONTROL: the operator-sandwich band injection V Sbar V^dag
      nda::array<cplx, 5> S_bad(nt, ns, nk, M, M);
      S_bad() = cplx(0.0);
      for (long it = 0; it < nt; ++it)
        for (long s = 0; s < ns; ++s)
          for (long k = 0; k < nk; ++k)
            for (long i = 0; i < M; ++i)
              for (long j = 0; j < M; ++j) {
                cplx acc(0.0);
                for (long a = 0; a < M; ++a)
                  for (long b = 0; b < M; ++b)
                    acc += V(i, a) * Sxb(it, s, k, a, b) * std::conj(V(j, b));
                S_bad(it, s, k, i, j) = acc;
              }
      const cplx t_x_bad = pairing(S_bad, dG, M, M);
      const double rel_bad = std::abs(t_x_bad - dphi_expl) / std::abs(dphi_expl);
      app_log(1, "vertex_fdoracle_bs wannier-gauge [prec = {}]: |Phi_w - Phi| rel = "
                 "{:.3e}; |T_w[Sbar,dG] - T[Sx,dG]| rel = {:.3e}; oracle rel = {:.3e}; "
                 "operator-sandwich control rel = {:.3e} (must be O(1)-broken)",
              prec, std::abs(phi_w - phi_hat) / std::abs(phi_hat),
              std::abs(t_x_w - t_x) / std::abs(t_x), rel_w, rel_bad);
      REQUIRE(std::abs(phi_w - phi_hat) < 1e-9 * std::abs(phi_hat));
      REQUIRE(std::abs(t_x_w - t_x) < 1e-9 * std::abs(t_x));
      REQUIRE(rel_w < 1e-7);                     // the oracle, in Wannier labels (E-S2)
      REQUIRE(rel_bad > 1e-2);                   // the section-2.7 trap, made a control
    }
#endif
  }

  /**
   * INCREMENT S9 -- the W-SIDE FUNCTIONAL-DERIVATIVE ORACLE for Formulation B-L.
   *
   * WHY THIS EXISTS (and why its absence mattered). S9 shipped WITHOUT its mandated gate:
   * the file above covers B-S only (its W-cut vanishes identically), so until now NOTHING
   * verified that B-L's two cuts are consistent cuts of one functional. Every other B-L
   * test in the suite is loose (finiteness / magnitude bands), relative (sym-vs-nosym) or
   * trivial (C = empty). A wrong WEIGHT in P^{C,L} -- the one term that distinguishes B-L
   * from B-S -- passes all of them, and shows up only as bad physics in production. It did:
   * on Si kp222 B-L departs from B-S by 7.5x with the opposite sign and goes unstable at
   * iteration 8, while B-S (which is gated) converges cleanly.
   *
   * THE IDENTITY (theoryB_static eq:Woracle):
   *
   *   d/dl Phi_2^{C,L}[G, W + l dW]|_0
   *       = -1/2 * (1/(Nk beta)) sum_{q,nu} sum_{IJ} P^{C,L}_{IJ}(q,inu) dW_{JI}(q,inu)
   *
   * It "pins the FULL weight of P^{C,L} -- the discriminator against the half-weight trap
   * (the naive Variant-F functional fails it by exactly a factor 2)".
   *
   * WHAT MAKES IT SHARP. The two sides come from DIFFERENT kernels: the left from the
   * Sigma kernel (eval_sigma_C_g3w2, rung_mode = 2) through the Euler identity
   * eq:eulerBL1, T[Sigma^{C,L}, G] = 4 Phi; the right from the Pi kernel
   * (pi_c_accumulate_w with the static rung W0 and NO dynamic rung, which is exactly what
   * vertex_t::eval_Pi_C injects for B-L). Nothing forces them to agree unless the relative
   * normalization of the two cuts is right.
   *
   * TWO STRUCTURAL SIMPLIFICATIONS, both exact:
   *  (1) Phi_2^{C,L} is AFFINE in W (degree 1 + degree 0, eq:hierarchy), so the identity
   *      holds with NO finite-difference truncation at all: Phi(l) - Phi(0) = l * dPhi/dl
   *      exactly, for any l. The residual is pure basis-eps, not h^2. Affinity is itself
   *      asserted below -- if it ever fails, the W-dependence is not what the theory says.
   *  (2) The W-derivative holds W0[G] FIXED (W0 is a functional of G, not of W), so the
   *      identity must hold for ANY fixed kernel. The toy therefore uses the model's own
   *      Hermitian Z as the static kernel instead of reconstructing W0[G] -- fewer moving
   *      parts, and it keeps this case independent of the B-S helpers above.
   *
   * The perturbation dW is built as a bosonic Lorentzian 2*Om/(Om^2+nu^2) times a
   * q-resolved Hermitian matrix obeying dW(-q) = dW(q)^T -- i.e. a genuine representable
   * tau-function with the reality symmetry the kernel assumes, for the same reason the
   * B-S case builds dG from pole data rather than from per-node noise.
   */
  TEST_CASE("vertex_fdoracle_bl_wside", "[methods][vertex][fdoracle][bl]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_fdoracle_bl_wside skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace fdo;
    auto &mpi_context = utils::make_unit_test_mpi_context();
    auto comm = mpi_context->comm;

    std::string prec = GENERATE(std::string("medium"), std::string("high"));
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, prec);
    iaft_tools tools(ft);
    const long nt = ft.nt_f(), nw_b = ft.nw_b();
    model_t mdl;
    auto G0 = mdl.G_tau(ft);

    // int_0^beta dtau f(tau) = f(i.nu = 0), via the bosonic transform row
    auto tau_integral = [&](nda::array<cplx, 1> const &f_t) {
      cplx acc(0.0);
      for (long it = 0; it < nt; ++it) acc += tools.Twt_bb(tools.m0, it) * f_t(it);
      return acc;
    };

    // the pinned SAME-INDEX FERMIONIC pairing (as above):
    //   T[A,B] = (1/(Nk beta)) sum_{s,k,w,ab} A_ab B_ab
    //          = -(1/Nk) sum_{s,k,ab} int dtau A_ab(tau) B_ab(beta - tau)
    // (the minus is fermionic antiperiodicity; the BOSONIC pairing below has none)
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

    // the STATIC KERNEL. Any fixed Hermitian rung with Z(-q) = Z(q)^T is legal here (see
    // simplification (2) above); the model's Z already carries exactly that symmetry.
    auto const &W0 = mdl.Z_qPQ;

    // ---- Sigma^{C,L}: the THREE explicit terms of eq:sigmaBL --------------------------
    //   W_x W_y -> W0_x W_y + W_x W0_y - W0_x W0_y,
    // which is what rung_mode = 2 computes from (Z-slot = W0, dynamic rung = dW = W - W0):
    // its S3/S1/S2 reductions are W0_x W0_y, W0_x dW_y, dW_x W0_y. Externals stay FREE
    // (as for Sigma^x); the C-block pairing is applied afterwards.
    auto sigma_BL = [&](nda::array<cplx, 5> const &G,
                        nda::array<cplx, 4> const &dW_qw) {
      nda::array<cplx, 4> Wstub(nk, 0, Np, Np);
      nda::array<cplx, 5> S(nt, ns, nk, nbnd, nbnd);
      solvers::vertex_detail::eval_sigma_C_g3w2(ft, comm, C(), G, mdl.X_skPa, Wstub, W0,
                                                mdl.kmq, mdl.qmin, /*iq_gamma*/ 0,
                                                /*skip_rung_gamma*/ false,
                                                /*rung_mode*/ 2, &dW_qw, nullptr, S);
      return S;
    };

    // Phi_2^{C,L} from the Euler identity eq:eulerBL1: T[Sigma^{C,L}, G] = 4 Phi.
    // Uses the SAME kernel the production path uses, so the two sides of the oracle share
    // one normalization by construction -- as in the B-S case.
    auto phi_BL = [&](nda::array<cplx, 4> const &dW_qw) {
      auto S = sigma_BL(G0, dW_qw);
      return 0.25 * pairing(S, G0, ncw, ncw);
    };

    // ---- P^{C,L}(q, i.nu) = the static-rung Pi^C at FULL weight (eq:PCL) --------------
    // EXACTLY the call vertex_t::eval_Pi_C makes for vertex_rung = "linear": rung W0bar,
    // Wdyn = nullptr. Fed the C-C block, because the Pi kernel contracts its external
    // orbital legs into the aux indices (all eight labels of Phi are in C).
    nda::array<cplx, 4> PCL_w(nw_b, nk, Np, Np);
    {
      nda::array<cplx, 5> G_CC(nt, ns, nk, ncw, ncw);
      nda::array<cplx, 4> X_C(ns, nk, Np, ncw);
      for (long it = 0; it < nt; ++it)
        for (long s = 0; s < ns; ++s)
          for (long k = 0; k < nk; ++k)
            for (long a = 0; a < ncw; ++a)
              for (long b = 0; b < ncw; ++b) G_CC(it, s, k, a, b) = G0(it, s, k, a, b);
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < nk; ++k)
          for (long P = 0; P < Np; ++P)
            for (long a = 0; a < ncw; ++a) X_C(s, k, P, a) = mdl.X_skPa(s, k, P, a);
      PCL_w() = cplx(0.0);
      vertex_pi::pi_c_accumulate_w(ft, tools, G_CC, X_C, W0, static_cast<nda::array<ComplexType, 4> const*>(nullptr),
                                   mdl.kmq, mdl.kpq, nda::range(0, ncw), PCL_w,
                                   comm.rank(), comm.size());
      comm.all_reduce_in_place_n(PCL_w.data(), PCL_w.size(), std::plus<>{});
    }

    // ---- the perturbation dW: a representable bosonic tau-function -------------------
    // Lorentzian 2*Om/(Om^2 + nu^2) (a single bosonic pole, hence exactly representable)
    // times a q-resolved Hermitian matrix with dW(-q) = dW(q)^T, matching the symmetry
    // the rung objects carry. Built in BOTH layouts: (nq, nw_b, .) for the Sigma kernel's
    // Ww_override, (nw_b, nq, .) for w_to_tau.
    const double Om = 0.83;
    nda::array<cplx, 4> dW_qw(nk, nw_b, Np, Np), dW_wq(nw_b, nk, Np, Np);
    {
      auto M = make_rung(4242, 1.0);
      REQUIRE(rung_sym_err(M) < 1e-14);   // the rung symmetry the Sigma pattern assumes
      for (long q = 0; q < nk; ++q)
        for (long l = 0; l < nw_b; ++l) {
          const double nu = double(tools.wn_b(l)) * M_PI / beta;
          const double s = 2.0 * Om / (Om * Om + nu * nu);
          for (long P = 0; P < Np; ++P)
            for (long Q = 0; Q < Np; ++Q) {
              dW_qw(q, l, P, Q) = M(q, P, Q) * s;
              dW_wq(l, q, P, Q) = M(q, P, Q) * s;
            }
        }
      // the perturbation must be nontrivial, else the oracle is vacuous
      double mx = 0.0;
      for (auto const &v : dW_qw) mx = std::max(mx, std::abs(v));
      REQUIRE(mx > 1e-3);
    }

    // ---- the BOSONIC pairing of the RHS ----------------------------------------------
    //   (1/(Nk beta)) sum_{q,nu} sum_{IJ} P_{IJ}(q,inu) dW_{JI}(q,inu)
    // = (1/Nk) sum_q sum_{IJ} int_0^beta dtau P_{IJ}(q,tau) dW_{JI}(q, beta - tau).
    // NO minus: for two BOSONIC functions e^{-i nu beta} = +1 (contrast the fermionic
    // pairing above). And NOT a sum over the sampled nodes -- DLR nodes are fitting
    // nodes, not Fourier points.
    auto w_pairing = [&](nda::array<cplx, 4> const &P_w) {
      nda::array<cplx, 4> P_t(nt, nk, Np, Np), dW_t(nt, nk, Np, Np);
      ft.w_to_tau(P_w, P_t, imag_axes_ft::boson);
      ft.w_to_tau(dW_wq, dW_t, imag_axes_ft::boson);
      cplx tot(0.0);
      nda::array<cplx, 1> f_t(nt);
      for (long q = 0; q < nk; ++q)
        for (long I = 0; I < Np; ++I)
          for (long J = 0; J < Np; ++J) {
            for (long it = 0; it < nt; ++it)
              f_t(it) = P_t(it, q, I, J) * dW_t(tools.t_mirror(it), q, J, I);
            tot += tau_integral(f_t) / double(nk);
          }
      return tot;
    };

    // ---- THE ORACLE ------------------------------------------------------------------
    nda::array<cplx, 4> dW_base(nk, nw_b, Np, Np);
    {   // a nonzero base point, so the test is not secretly evaluated at dW = 0 (the B-S
        // limit) where the mixed terms vanish and the W-dependence would be untested
      auto B = make_rung(1717, 0.4);
      REQUIRE(rung_sym_err(B) < 1e-14);
      const double Ob = 1.31;
      for (long q = 0; q < nk; ++q)
        for (long l = 0; l < nw_b; ++l) {
          const double nu = double(tools.wn_b(l)) * M_PI / beta;
          const double s = 2.0 * Ob / (Ob * Ob + nu * nu);
          for (long P = 0; P < Np; ++P)
            for (long Q = 0; Q < Np; ++Q) dW_base(q, l, P, Q) = B(q, P, Q) * s;
        }
    }
    auto shifted_W = [&](double lam) {
      nda::array<cplx, 4> A(nk, nw_b, Np, Np);
      for (long i = 0; i < A.size(); ++i)
        A.data()[i] = dW_base.data()[i] + lam * dW_qw.data()[i];
      return A;
    };

    const cplx phi0 = phi_BL(dW_base);
    const cplx rhs = -0.5 * w_pairing(PCL_w);

    // AFFINITY: Phi must be exactly linear in lambda. Checked FIRST -- if it fails, the
    // W-dependence is not degree-1 and the oracle statement itself does not apply.
    const cplx lhs1 = phi_BL(shifted_W(1.0)) - phi0;
    const cplx lhs2 = phi_BL(shifted_W(2.0)) - phi0;
    const double affine = std::abs(lhs2 - 2.0 * lhs1) / std::max(std::abs(lhs1), 1e-30);
    app_log(1, "fdoracle_bl_wside [{}]: affinity |Phi(2l)-Phi(0) - 2(Phi(l)-Phi(0))| / "
               "|Phi(l)-Phi(0)| = {:.3e}", prec, affine);
    REQUIRE(affine < 1e-8);

    const double rel = std::abs(lhs1 - rhs) / std::max(std::abs(rhs), 1e-30);
    app_log(1, "fdoracle_bl_wside [{}]: dPhi/dl = {:.12e} {:+.12e}i   "
               "-1/2 T_W[P^(C,L), dW] = {:.12e} {:+.12e}i   rel = {:.3e}",
            prec, lhs1.real(), lhs1.imag(), rhs.real(), rhs.imag(), rel);
    REQUIRE(std::abs(rhs) > 1e-10);          // the RHS must not be vacuously zero
    REQUIRE(rel < 1e-6);

    // ---- POSITIVE CONTROLS (theoryB_static section BLconservation) -------------------
    // "replace P^{C,L} -> 1/2 P^{C,L} -> the W-oracle breaks by 2". If this control does
    // NOT break, the test is blind to the very trap it exists to catch.
    {
      nda::array<cplx, 4> Phalf(PCL_w);
      Phalf() *= 0.5;
      const cplx rhs_half = -0.5 * w_pairing(Phalf);
      const double r_half = std::abs(lhs1 / rhs_half);
      app_log(1, "fdoracle_bl_wside [{}]: CONTROL half-weight P^(C,L) -> ratio "
                 "dPhi / RHS = {:.6f} (must be 2)", prec, r_half);
      REQUIRE(std::abs(r_half - 2.0) < 1e-5);
    }
    // sign flip must also break it
    {
      nda::array<cplx, 4> Pneg(PCL_w);
      Pneg() *= -1.0;
      const cplx rhs_neg = -0.5 * w_pairing(Pneg);
      const double r_neg = std::abs(lhs1 - rhs_neg) / std::max(std::abs(rhs_neg), 1e-30);
      app_log(1, "fdoracle_bl_wside [{}]: CONTROL sign-flipped P^(C,L) -> rel = {:.3e} "
                 "(must be O(1))", prec, r_neg);
      REQUIRE(r_neg > 1.0);
    }
#endif
  }

  /**
   * DIAGNOSTIC (temporary): pin the MIXED reductions S1/S2 against the ALREADY-PINNED
   * doubly-instantaneous reduction S3.
   *
   * S1/S2 are BILINEAR in (x-rung, y-rung) exactly as S3 is, and the kernel is LINEAR in
   * the Z slot separately for each rung. So if the "dynamic" rung handed to rung_mode = 2
   * is a CONSTANT in i.nu, call it D, the mixed terms must be reproducible through the S3
   * route alone by expanding the bilinear form:
   *
   *   S3[W0 + D, W0 + D] = S3[W0,W0] + S3[W0,D] + S3[D,W0] + S3[D,D]
   *
   * hence   S3[W0,D] + S3[D,W0] = static(W0+D) - static(W0) - static(D),
   * while   S1[W0,D] + S2[D,W0] = linear(Z = W0, dW = D) - static(W0).
   *
   * The two right-hand sides use DISJOINT code paths (S3 only vs S1/S2 only) but must be
   * the same object. A mismatch convicts the ORBITAL/MOMENTUM ROUTING of S1/S2 (the
   * frequency algebra is trivial here -- D is a constant); a match exonerates the routing
   * and moves the suspicion to the frequency handling of a genuinely dynamic rung.
   */
  TEST_CASE("vertex_mixpin", "[methods][vertex][fdoracle][bl]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_mixpin skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace fdo;
    auto &mpi_context = utils::make_unit_test_mpi_context();
    auto comm = mpi_context->comm;

    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, "medium");
    iaft_tools tools(ft);
    const long nt = ft.nt_f(), nw_b = ft.nw_b();
    model_t mdl;
    auto G0 = mdl.G_tau(ft);

    auto const &W0 = mdl.Z_qPQ;

    // a SECOND fixed rung D, same symmetry class as W0 (Hermitian, D(-q) = D(q)^T)
    auto D = make_rung(20260730, 0.37);
    REQUIRE(rung_sym_err(D) < 1e-14);
    nda::array<cplx, 3> WpD(nk, Np, Np);
    for (long i = 0; i < WpD.size(); ++i) WpD.data()[i] = W0.data()[i] + D.data()[i];

    nda::array<cplx, 4> Wstub(nk, 0, Np, Np);
    auto sigma_static = [&](nda::array<cplx, 3> const &Zin) {
      nda::array<cplx, 5> S(nt, ns, nk, nbnd, nbnd);
      solvers::vertex_detail::eval_sigma_C_g3w2(ft, comm, C(), G0, mdl.X_skPa, Wstub, Zin,
                                                mdl.kmq, mdl.qmin, 0, false,
                                                /*rung_mode*/ 1, static_cast<nda::array<ComplexType, 4> const*>(nullptr), nullptr, S);
      return S;
    };
    // rung_mode = 2 with a nu-INDEPENDENT "dW"
    nda::array<cplx, 4> Dw_qw(nk, nw_b, Np, Np);
    for (long q = 0; q < nk; ++q)
      for (long l = 0; l < nw_b; ++l)
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) Dw_qw(q, l, P, Q) = D(q, P, Q);

    nda::array<cplx, 5> S_lin(nt, ns, nk, nbnd, nbnd);
    solvers::vertex_detail::eval_sigma_C_g3w2(ft, comm, C(), G0, mdl.X_skPa, Wstub, W0,
                                              mdl.kmq, mdl.qmin, 0, false,
                                              /*rung_mode*/ 2, &Dw_qw, nullptr, S_lin);

    auto S_W0  = sigma_static(W0);
    auto S_D   = sigma_static(D);
    auto S_WpD = sigma_static(WpD);

    double num = 0.0, den = 0.0, den_tot = 0.0;
    for (long i = 0; i < S_lin.size(); ++i) {
      const cplx mix_S1S2 = S_lin.data()[i] - S_W0.data()[i];
      const cplx mix_S3   = S_WpD.data()[i] - S_W0.data()[i] - S_D.data()[i];
      num = std::max(num, std::abs(mix_S1S2 - mix_S3));
      den = std::max(den, std::abs(mix_S3));
      den_tot = std::max(den_tot, std::abs(S_lin.data()[i]));
    }
    app_log(1, "vertex_mixpin: max|(S1+S2) - (S3 route)| = {:.4e}, scale |mixed| = {:.4e} "
               "(rel {:.4e}), scale |Sigma^(C,L)| = {:.4e}",
            num, den, num / den, den_tot);
    REQUIRE(den > 1e-10);
    REQUIRE(num < 1e-10 * den);
#endif
  }

  /**
   * SLOT-RESOLVED cut symmetry -- the sharpest form of "Sigma^C = dPhi/dG".
   *
   * Write Phi = (1/4) F(G,G,G,G) with the multilinear form
   *     F(g_A0, g_B, g_C, g_D) := T[ Sigma^C(g_B, g_C, g_D), g_A0 ],
   * the four slots being the four G lines of Phi_2 (A0 = the line the kernel cuts open;
   * B, C, D = its three internal lines). Then dPhi/dl = (1/4) sum_i F(dG in slot i),
   * whereas the kernel asserts T[Sigma, dG] = F(dG in slot A0). Those agree IFF F is
   * invariant under the diagram's C4 rotation of its slots -- so this test measures all
   * four cuts SEPARATELY (via sigma_C_slot_probe) instead of only their average.
   *
   * WHY IT EXISTS. The G-side oracle sees only the average, so when it fails it cannot say
   * WHICH cut moved. Run slot-resolved, the failure fingerprint is unmistakable: the
   * profile came out in arithmetic progression along the 4-cycle, invariant under exactly
   * the reflection that transposes the LEGAL rung and broken under the one that transposes
   * the illegal one. That identified the culprit -- rungs violating W_PQ(q) = W_QP(-q) at
   * the self-inverse transfer -- in one measurement, after the aggregate oracle had been
   * misread as convicting the mixed Sigma terms (eq:mixgw), which are in fact exact.
   *
   * The test covers B-S (S3) and both B-L mixed reductions (S1, S2), with constant and
   * nu-dependent rungs, and carries the illegal-rung positive control at the end.
   */
  TEST_CASE("vertex_slotprobe", "[methods][vertex][fdoracle][bl]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_slotprobe skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace fdo;
    auto &mpi_context = utils::make_unit_test_mpi_context();
    auto comm = mpi_context->comm;

    std::string prec = GENERATE(std::string("medium"), std::string("high"));
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, prec);
    iaft_tools tools(ft);
    const long nt = ft.nt_f(), nw_b = ft.nw_b();
    model_t mdl;
    auto G0 = mdl.G_tau(ft);
    auto const &W0 = mdl.Z_qPQ;

    auto tau_integral = [&](nda::array<cplx, 1> const &f_t) {
      cplx acc(0.0);
      for (long it = 0; it < nt; ++it) acc += tools.Twt_bb(tools.m0, it) * f_t(it);
      return acc;
    };
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

    // dG: same construction as the oracles (representable, k-reality-symmetric, non-Herm.)
    nda::array<cplx, 5> dG(nt, ns, nk, nbnd, nbnd);
    {
      rng_t rg(90210);
      const double de[3] = {-0.33, 0.19, 0.71};
      nda::array<cplx, 5> Amp(ns, nk, nbnd, nbnd, nbnd);
      Amp() = cplx(0.0);
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < nk; ++k) {
          if (kminus(k) < k) continue;
          const long km = kminus(k);
          for (long r = 0; r < nbnd; ++r)
            for (long a = 0; a < nbnd; ++a)
              for (long b = 0; b < nbnd; ++b) {
                if (km == k) {
                  if (b < a) continue;
                  cplx v = rg.z();
                  Amp(s, k, r, a, b) = v;  Amp(s, k, r, b, a) = v;
                } else {
                  cplx v = rg.z();
                  Amp(s, k, r, a, b) = v;  Amp(s, km, r, b, a) = v;
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
    }

    // a nu-DEPENDENT dW (Lorentzian, no constant piece) with dW(-q) = dW(q)^T Hermitian
    const double Om = 1.07;
    nda::array<cplx, 4> dW_qw(nk, nw_b, Np, Np);
    {
      auto M = make_rung(31337, 0.5);
      REQUIRE(rung_sym_err(M) < 1e-14);
      for (long q = 0; q < nk; ++q)
        for (long l = 0; l < nw_b; ++l) {
          const double nu = double(tools.wn_b(l)) * M_PI / beta;
          const double s = 2.0 * Om / (Om * Om + nu * nu);
          for (long P = 0; P < Np; ++P)
            for (long Q = 0; Q < Np; ++Q) dW_qw(q, l, P, Q) = M(q, P, Q) * s;
        }
    }

    // a nu-INDEPENDENT control rung, same symmetry class
    nda::array<cplx, 4> Dc_qw(nk, nw_b, Np, Np);
    {
      auto Dm = make_rung(20260730, 0.37);
      for (long q = 0; q < nk; ++q)
        for (long l = 0; l < nw_b; ++l)
          for (long P = 0; P < Np; ++P)
            for (long Q = 0; Q < Np; ++Q) Dc_qw(q, l, P, Q) = Dm(q, P, Q);
    }

    nda::array<cplx, 3> Dconst(nk, Np, Np), WpD(nk, Np, Np);
    for (long q = 0; q < nk; ++q)
      for (long P = 0; P < Np; ++P)
        for (long Q = 0; Q < Np; ++Q) {
          Dconst(q, P, Q) = Dc_qw(q, 0, P, Q);
          WpD(q, P, Q) = W0(q, P, Q) + Dconst(q, P, Q);
        }

    nda::array<cplx, 4> Wstub(nk, 0, Np, Np);
    // Sigma with slot overrides + term selection in place
    auto sigma = [&](int mode, int only, nda::array<cplx, 3> const &Zin,
                     nda::array<cplx, 4> const *rung,
                     nda::array<cplx, 5> const *pB, nda::array<cplx, 5> const *pC,
                     nda::array<cplx, 5> const *pD) {
      solvers::vertex_detail::sigma_C_slot_probe.B = pB;
      solvers::vertex_detail::sigma_C_slot_probe.C = pC;
      solvers::vertex_detail::sigma_C_slot_probe.D = pD;
      solvers::vertex_detail::sigma_C_slot_probe.only_term = only;
      nda::array<cplx, 5> S(nt, ns, nk, nbnd, nbnd);
      solvers::vertex_detail::eval_sigma_C_g3w2(
          ft, comm, C(), G0, mdl.X_skPa, Wstub, Zin, mdl.kmq, mdl.qmin, 0, false,
          mode, rung, nullptr, S);
      solvers::vertex_detail::sigma_C_slot_probe.clear();
      return S;
    };

    auto report = [&](const char *tag, std::array<cplx, 4> const &f) {
      const cplx avg = 0.25 * (f[0] + f[1] + f[2] + f[3]);
      const double flat = std::abs(f[0] - avg) / std::abs(avg);
      app_log(1, "vertex_slotprobe [{}] {}:\n"
                 "    A0 = {:.10e}{:+.10e}i   B  = {:.10e}{:+.10e}i\n"
                 "    C  = {:.10e}{:+.10e}i   D  = {:.10e}{:+.10e}i\n"
                 "    |A0-avg|/|avg| = {:.4e}   |A0-C|/|A0| = {:.4e}   "
                 "|B-D|/|B| = {:.4e}",
              prec, tag, f[0].real(), f[0].imag(), f[1].real(), f[1].imag(),
              f[2].real(), f[2].imag(), f[3].real(), f[3].imag(), flat,
              std::abs(f[0] - f[2]) / std::abs(f[0]),
              std::abs(f[1] - f[3]) / std::abs(f[1]));
      return flat;
    };

    std::array<cplx, 4> last{};
    auto profile = [&](const char *tag, int only, nda::array<cplx, 3> const &Zin,
                       nda::array<cplx, 4> const *rung) {
      std::array<cplx, 4> f{};
      for (int slot = 0; slot < 4; ++slot) {
        nda::array<cplx, 5> const *pB = (slot == 1) ? &dG : nullptr;
        nda::array<cplx, 5> const *pC = (slot == 2) ? &dG : nullptr;
        nda::array<cplx, 5> const *pD = (slot == 3) ? &dG : nullptr;
        auto S = sigma(rung ? 2 : 1, only, Zin, rung, pB, pC, pD);
        f[slot] = (slot == 0) ? pairing(S, dG, ncw, ncw) : pairing(S, G0, ncw, ncw);
      }
      last = f;
      return report(tag, f);
    };

    // ---- every LEGAL rung must give a FLAT profile: that IS "Sigma = dPhi/dG" ---------
    const double gate = 1e-9;
    REQUIRE(profile("S3, Z = W0", 3, W0, nullptr) < gate);
    auto pW0 = last;
    REQUIRE(profile("S3, Z = D", 3, Dconst, nullptr) < gate);
    auto pD_ = last;
    REQUIRE(profile("S3, Z = W0 + D", 3, WpD, nullptr) < gate);
    auto pWpD = last;
    {   // the mixed part reached through the PINNED S3 route only
      std::array<cplx, 4> f{};
      for (int i = 0; i < 4; ++i) f[i] = pWpD[i] - pW0[i] - pD_[i];
      REQUIRE(report("MIXED via the S3 route (S3[W0,D]+S3[D,W0])", f) < gate);
    }
    REQUIRE(profile("S1 only, CONSTANT dW", 1, W0, &Dc_qw) < gate);
    auto s1c = last;
    REQUIRE(profile("S2 only, CONSTANT dW", 2, W0, &Dc_qw) < gate);
    auto s2c = last;
    {   // S1 + S2 with a constant rung must BE the S3 route, slot by slot
      std::array<cplx, 4> f{};
      for (int i = 0; i < 4; ++i) f[i] = s1c[i] + s2c[i];
      REQUIRE(report("MIXED via S1+S2, CONSTANT dW", f) < gate);
    }
    // the ones that actually matter for B-L: a genuinely nu-DEPENDENT rung
    REQUIRE(profile("S1 only, Lorentzian dW", 1, W0, &dW_qw) < gate);
    REQUIRE(profile("S2 only, Lorentzian dW", 2, W0, &dW_qw) < gate);

    // ---- POSITIVE CONTROL: an ILLEGAL rung MUST break the flatness -------------------
    // Hermitian per q and dW(-q) = dW(q)^T away from the zone centre, but NOT symmetric
    // at the self-inverse q -- i.e. exactly what a single-pass +q/-q write produces, and
    // exactly the defect that made the B-L G-side oracle read 1.118e-01. If this control
    // stops firing, the probe has gone blind to the thing it exists to catch.
    {
      nda::array<cplx, 3> Bad(Dconst);
      for (long q = 0; q < nk; ++q) {
        if (kminus(q) != q) continue;                 // self-inverse transfers only
        for (long P = 0; P < Np; ++P)
          for (long Q = P + 1; Q < Np; ++Q) {         // Hermitian, deliberately non-real
            const cplx v(Bad(q, P, Q).real(), 0.21);
            Bad(q, P, Q) = v;  Bad(q, Q, P) = std::conj(v);
          }
      }
      REQUIRE(rung_sym_err(Bad) > 1e-3);              // it really is illegal
      const double broke = profile("S3, ILLEGAL Z (CONTROL, must NOT be flat)", 3, Bad,
                                   nullptr);
      REQUIRE(broke > 1e-3);
    }
#endif
  }

  /**
   * INCREMENT S9 -- the G-SIDE FUNCTIONAL-DERIVATIVE ORACLE for Formulation B-L.
   *
   * The companion to the W-side oracle above. That one pins P^{C,L}'s weight against the
   * explicit Sigma terms; this one pins the REMAINING B-L object, the response self-energy
   *
   *   Delta w^L(q) := W0(q) [ pi^dyn(q) - Pi^C(q, tau=0) ] W0(q)     (eq:deltawL)
   *
   * which the W-side oracle cannot see at all (it is a G-side object, born from the chain
   * rule through W0[G]). theoryB_static eq:eulerBL1 + the G-side oracle:
   *
   *   d/dl Phi_2^{C,L}[G + l dG, W]|_0
   *       = T[ Sigma^{C,L}, P_C dG P_C ]  +  T[ Sigma^{L,r}, dG ]
   *
   * with Sigma^{C,L} the THREE explicit terms of eq:sigmaBL and Sigma^{L,r} the response.
   *
   * THE ESSENTIAL SUBTLETY, and the reason this test has teeth: W (the physical screened
   * interaction) is held FIXED while G varies, but the KERNEL W0[G] is a functional of G,
   * so the fluctuation dW = W - W0[G] VARIES WITH G. Every place W0 appears -- the two
   * mixed rungs, the -W0 W0 term, and dW itself -- contributes to the chain rule, and
   * X^L (eq:XL) is exactly the sum of those rung derivatives. If Delta w^L had the wrong
   * sign, the wrong factor, or the wrong Pi combination, the residual here is O(1).
   *
   * This is also the ONLY test that exercises pi^dyn inside a conservation statement
   * rather than against another implementation of itself.
   */
  TEST_CASE("vertex_fdoracle_bl_gside", "[methods][vertex][fdoracle][bl]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_fdoracle_bl_gside skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace fdo;
    auto &mpi_context = utils::make_unit_test_mpi_context();
    auto comm = mpi_context->comm;

    std::string prec = GENERATE(std::string("medium"), std::string("high"));
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, prec);
    iaft_tools tools(ft);
    const long nt = ft.nt_f(), nw_b = ft.nw_b();
    model_t mdl;
    auto G0 = mdl.G_tau(ft);
    auto R0row = solvers::vertex_w0_detail::tau0_transform_row(ft);

    auto tau_integral = [&](nda::array<cplx, 1> const &f_t) {
      cplx acc(0.0);
      for (long it = 0; it < nt; ++it) acc += tools.Twt_bb(tools.m0, it) * f_t(it);
      return acc;
    };
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
    // W0[G] = [1 - Z P0]^{-1} Z at i.nu = 0 (identical to the B-S case)
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
      return W0;
    };

    // C-C block extractors for the Pi kernels
    auto GCC_of = [&](nda::array<cplx, 5> const &G) {
      nda::array<cplx, 5> G_CC(nt, ns, nk, ncw, ncw);
      for (long it = 0; it < nt; ++it)
        for (long s = 0; s < ns; ++s)
          for (long k = 0; k < nk; ++k)
            for (long a = 0; a < ncw; ++a)
              for (long b = 0; b < ncw; ++b) G_CC(it, s, k, a, b) = G(it, s, k, a, b);
      return G_CC;
    };
    nda::array<cplx, 4> X_C(ns, nk, Np, ncw);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < nk; ++k)
        for (long P = 0; P < Np; ++P)
          for (long a = 0; a < ncw; ++a) X_C(s, k, P, a) = mdl.X_skPa(s, k, P, a);

    // ---- the FIXED physical W(q, i.nu). Held constant as G varies; a bosonic Lorentzian
    //      around the bare core, with W(-q) = W(q)^T Hermitian, so it is representable and
    //      carries the reality symmetry the kernels assume.
    const double Om = 1.07;
    nda::array<cplx, 4> W_qw(nk, nw_b, Np, Np);
    {
      auto M = make_rung(31337, 0.5);
      REQUIRE(rung_sym_err(M) < 1e-14);
      for (long q = 0; q < nk; ++q)
        for (long l = 0; l < nw_b; ++l) {
          const double nu = double(tools.wn_b(l)) * M_PI / beta;
          const double s = 2.0 * Om / (Om * Om + nu * nu);
          for (long P = 0; P < Np; ++P)
            for (long Q = 0; Q < Np; ++Q)
              W_qw(q, l, P, Q) = mdl.Z_qPQ(q, P, Q) + M(q, P, Q) * s;
        }
    }

    // dW(G) = W - W0[G]: the fluctuation VARIES with G through the kernel
    auto dW_of = [&](nda::array<cplx, 3> const &W0) {
      nda::array<cplx, 4> dW(nk, nw_b, Np, Np);
      for (long q = 0; q < nk; ++q)
        for (long l = 0; l < nw_b; ++l)
          for (long P = 0; P < Np; ++P)
            for (long Q = 0; Q < Np; ++Q)
              dW(q, l, P, Q) = W_qw(q, l, P, Q) - W0(q, P, Q);
      return dW;
    };

    auto sigma_BL = [&](nda::array<cplx, 5> const &G, nda::array<cplx, 3> const &W0,
                        nda::array<cplx, 4> const &dW) {
      nda::array<cplx, 4> Wstub(nk, 0, Np, Np);
      nda::array<cplx, 5> S(nt, ns, nk, nbnd, nbnd);
      solvers::vertex_detail::eval_sigma_C_g3w2(ft, comm, C(), G, mdl.X_skPa, Wstub, W0,
                                                mdl.kmq, mdl.qmin, 0, false,
                                                /*rung_mode*/ 2, &dW, nullptr, S);
      return S;
    };

    // Phi_2^{C,L}[G, W] via Euler (eq:eulerBL1), with W0 and dW BOTH following G
    auto phi_BL_of = [&](nda::array<cplx, 5> const &G) {
      auto W0 = W0_from_G(G);
      auto S = sigma_BL(G, W0, dW_of(W0));
      return 0.25 * pairing(S, G, ncw, ncw);
    };

    // ---- Pi^{C,0}(q, tau=0): static rung W0, and pi^dyn(q): full dynamic rung W -------
    auto piC0_tau0 = [&](nda::array<cplx, 5> const &G, nda::array<cplx, 3> const &W0) {
      auto G_CC = GCC_of(G);
      nda::array<cplx, 4> Pi_wq(nw_b, nk, Np, Np);
      Pi_wq() = cplx(0.0);
      vertex_pi::pi_c_accumulate_w(ft, tools, G_CC, X_C, W0, static_cast<nda::array<ComplexType, 4> const*>(nullptr),
                                   mdl.kmq, mdl.kpq, nda::range(0, ncw), Pi_wq,
                                   comm.rank(), comm.size());
      comm.all_reduce_in_place_n(Pi_wq.data(), Pi_wq.size(), std::plus<>{});
      nda::array<cplx, 3> Pi0(nk, Np, Np);
      for (long q = 0; q < nk; ++q)
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            cplx acc(0.0);
            for (long m = 0; m < nw_b; ++m) acc += R0row(m) * Pi_wq(m, q, P, Q);
            Pi0(q, P, Q) = acc;
          }
      return Pi0;
    };
    auto pidyn = [&](nda::array<cplx, 5> const &G) {
      auto G_CC = GCC_of(G);
      nda::array<cplx, 3> Zzero(nk, Np, Np);      // rung = Z + Wdyn(l) = W(i.nu_l)
      Zzero() = cplx(0.0);
      nda::array<cplx, 3> out(nk, Np, Np);
      out() = cplx(0.0);
      vertex_pi::pi_dyn_factorized(tools, G_CC, X_C, Zzero, &W_qw,
                                   mdl.kmq, mdl.kpq, nda::range(0, ncw), R0row, out,
                                   comm.rank(), comm.size());
      comm.all_reduce_in_place_n(out.data(), out.size(), std::plus<>{});
      return out;
    };

    // ---- the perturbation dG: non-Hermitian, k-reality-symmetric (as in the B-S case)
    nda::array<cplx, 5> dG(nt, ns, nk, nbnd, nbnd);
    {
      rng_t rg(90210);
      const double de[3] = {-0.33, 0.19, 0.71};
      nda::array<cplx, 5> Amp(ns, nk, nbnd, nbnd, nbnd);
      Amp() = cplx(0.0);
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < nk; ++k) {
          if (kminus(k) < k) continue;
          const long km = kminus(k);
          for (long r = 0; r < nbnd; ++r)
            for (long a = 0; a < nbnd; ++a)
              for (long b = 0; b < nbnd; ++b) {
                if (km == k) {
                  if (b < a) continue;
                  cplx v = rg.z();
                  Amp(s, k, r, a, b) = v;  Amp(s, k, r, b, a) = v;
                } else {
                  cplx v = rg.z();
                  Amp(s, k, r, a, b) = v;  Amp(s, km, r, b, a) = v;
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
      REQUIRE(herm > 0.1 * sc);
    }
    auto shifted = [&](double lam) {
      nda::array<cplx, 5> G(nt, ns, nk, nbnd, nbnd);
      for (long i = 0; i < G.size(); ++i) G.data()[i] = G0.data()[i] + lam * dG.data()[i];
      return G;
    };

    // ---- THE ORACLE ------------------------------------------------------------------
    const double h = 1e-6;
    const cplx dphi_fd = (phi_BL_of(shifted(h)) - phi_BL_of(shifted(-h))) / (2.0 * h);

    auto W0 = W0_from_G(G0);
    auto SL = sigma_BL(G0, W0, dW_of(W0));
    auto Pi0 = piC0_tau0(G0, W0);
    auto Pdyn = pidyn(G0);

    // Pi^L = pi^dyn - Pi^{C,0}(tau=0), then Delta w^L = W0 Pi^L W0 (eq:deltawL)
    nda::array<cplx, 3> PiL(nk, Np, Np);
    for (long q = 0; q < nk; ++q)
      for (long P = 0; P < Np; ++P)
        for (long Q = 0; Q < Np; ++Q) PiL(q, P, Q) = Pdyn(q, P, Q) - Pi0(q, P, Q);
    {
      double xl = 0.0, p0 = 0.0;
      for (auto const &v : PiL) xl = std::max(xl, std::abs(v));
      for (auto const &v : Pi0) p0 = std::max(p0, std::abs(v));
      app_log(1, "fdoracle_bl_gside [{}]: X^L meter max|pi^dyn - Pi^0| = {:.4e}, "
                 "max|Pi^0| = {:.4e}, ratio {:.4f}", prec, xl, p0, xl / p0);
      REQUIRE(xl > 1e-10);     // the response must be non-trivial, else the test is blind
    }
    nda::array<cplx, 3> DwL(nk, Np, Np);
    solvers::vertex_detail::build_delta_w(W0, PiL, mdl.qmin, DwL);
    nda::array<cplx, 5> SLr(nt, ns, nk, nbnd, nbnd);
    solvers::vertex_detail::eval_sigma_C_response(comm, G0, mdl.X_skPa, DwL, mdl.kmq,
                                                  mdl.qmin, SLr);

    const cplx t_expl = pairing(SL, dG, ncw, ncw);     // C-block: P_C dG P_C
    const cplx t_resp = pairing(SLr, dG, nbnd, nbnd);  // full space
    const cplx pred = t_expl + t_resp;

    // ---- SPLIT (the B-S diagnostic, applied to B-L): freeze the KERNEL so only the
    //      explicit lines vary. Its derivative must be exactly T[Sigma^{C,L}, P_C dG P_C];
    //      whatever is left over is the response, and must equal T[Sigma^{L,r}, dG].
    //      This localizes any discrepancy to ONE of the two cuts.
    auto phi_frozen = [&](nda::array<cplx, 5> const &G) {
      auto S = sigma_BL(G, W0, dW_of(W0));      // W0 AND dW held at their G0 values
      return 0.25 * pairing(S, G, ncw, ncw);
    };
    const cplx dphi_expl = (phi_frozen(shifted(h)) - phi_frozen(shifted(-h))) / (2.0 * h);
    const cplx remainder = dphi_fd - dphi_expl;
    app_log(1, "fdoracle_bl_gside [{}] SPLIT: T[Sigma^(C,L),dG_C] = {:.10e}{:+.10e}i, "
               "dPhi_frozenW0/dl = {:.10e}{:+.10e}i (rel {:.3e});  "
               "T[Sigma^(L,r),dG] = {:.10e}{:+.10e}i, remainder = {:.10e}{:+.10e}i "
               "(ratio {:.6f})", prec,
            t_expl.real(), t_expl.imag(), dphi_expl.real(), dphi_expl.imag(),
            std::abs(t_expl - dphi_expl) / std::max(std::abs(dphi_expl), 1e-30),
            t_resp.real(), t_resp.imag(), remainder.real(), remainder.imag(),
            std::abs(t_resp) / std::max(std::abs(remainder), 1e-30));
    // ---- ISOLATION: with dW == 0 only S3 = Sigma^x survives, so the explicit split MUST
    //      fall back to the B-S result (which passes at ~1e-10). If the zero-dW split is
    //      clean while the full one is not, the discrepancy is entirely in the MIXED terms
    //      S1/S2 -- i.e. eq:mixgw, which open item O7 flags as hand-derived, O1 risk class.
    double rel_S3 = 0.0;
    {
      nda::array<cplx, 4> dWz(nk, nw_b, Np, Np);
      dWz() = cplx(0.0);
      auto phi_S3 = [&](nda::array<cplx, 5> const &G) {
        auto S = sigma_BL(G, W0, dWz);
        return 0.25 * pairing(S, G, ncw, ncw);
      };
      auto S3 = sigma_BL(G0, W0, dWz);
      const cplx t3 = pairing(S3, dG, ncw, ncw);
      const cplx d3 = (phi_S3(shifted(h)) - phi_S3(shifted(-h))) / (2.0 * h);
      rel_S3 = std::abs(t3 - d3) / std::max(std::abs(d3), 1e-30);
      app_log(1, "fdoracle_bl_gside [{}] ISOLATION: dW == 0 (S3 only) -> explicit split "
                 "rel = {:.3e}   [vs {:.3e} with the mixed terms on]", prec, rel_S3,
              std::abs(t_expl - dphi_expl) / std::max(std::abs(dphi_expl), 1e-30));
      // S3 alone is B-S's Sigma^x, whose gradient property is pinned by vertex_fdoracle_bs
      REQUIRE(rel_S3 < 1e-6);
    }

    const double rel = std::abs(dphi_fd - pred) / std::max(std::abs(dphi_fd), 1e-30);
    app_log(1, "fdoracle_bl_gside [{}]: dPhi/dl(FD) = {:.10e} {:+.10e}i   "
               "T[Sigma^(C,L),dG]_C = {:.10e}   T[Sigma^(L,r),dG] = {:.10e}   rel = {:.3e}",
            prec, dphi_fd.real(), dphi_fd.imag(), t_expl.real(), t_resp.real(), rel);
    REQUIRE(std::abs(dphi_fd) > 1e-8);
    REQUIRE(rel < 1e-4);

    // ---- THE q -> 0 HEAD-CHANNEL PROJECTION IS NOT A FUNCTIONAL DERIVATIVE ------------
    // vertex_t::eval_Sigma_C (vertex_t.cpp, the _bl_head_projection block) deletes the
    // rank-1 head component chi chi^dag from the response middle factor at q = Gamma
    // BEFORE build_delta_w. On Si that removes 66 % of max|Pi(Gamma)| and changes
    // Sigma^(L,r) by 9.7x, so it is not a round-off cleanup -- it is a modification of the
    // theory. It is also applied to the SIGMA CUT ONLY: eval_Pi_C's P^{C,L}, which feeds
    // the Dyson equation, keeps its head channel.
    //
    // This block asks the only question that settles whether that is legal: does the
    // PROJECTED Sigma^(L,r) still satisfy the B-L G-side identity? The identity above
    // holds for the unprojected middle factor at ~1e-11. Pi^L is what the chain rule
    // through W0[G] produces, so deleting any part of it must show up as a residual --
    // unless the deleted part is annihilated downstream, which is exactly what a
    // "harmless projection" would mean.
    //
    // The toy model has no basis_head, so chi here is SYNTHETIC. That is sufficient: the
    // claim under test is about deleting a rank-1 channel from Pi^L, not about which
    // direction chi points. chi is given a non-trivial complex phase (Si's head is not
    // real; LiH's is) and H is built exactly as vertex_head_detail::build_head_rank1 does,
    // H_PQ = c * Re[conj(chi_P) chi_Q], including the Re[.].
    {
      nda::array<cplx, 1> chi(Np);
      rng_t rg(20260731);
      for (long P = 0; P < Np; ++P) chi(P) = cplx(rg.u(), 0.37 * rg.u());
      const double c_head = 1.7;                 // stands in for N_k * madelung
      nda::array<cplx, 2> H(Np, Np);
      for (long P = 0; P < Np; ++P)
        for (long Q = 0; Q < Np; ++Q)
          H(P, Q) = c_head * std::real(std::conj(chi(P)) * chi(Q));

      // the production projection, verbatim, at the same iq_gamma = 0 the rest of this
      // test uses for the Gamma transfer
      const long iq_gamma = 0;
      cplx hp(0.0);
      double hn = 0.0, pmax = 0.0, dmax = 0.0;
      for (long P = 0; P < Np; ++P)
        for (long Q = 0; Q < Np; ++Q) {
          hp += std::conj(H(P, Q)) * PiL(iq_gamma, P, Q);
          hn += std::norm(H(P, Q));
          pmax = std::max(pmax, std::abs(PiL(iq_gamma, P, Q)));
        }
      REQUIRE(hn > 0.0);
      const cplx cproj = hp / cplx(hn);
      nda::array<cplx, 3> PiP(PiL);
      for (long P = 0; P < Np; ++P)
        for (long Q = 0; Q < Np; ++Q) {
          const cplx d = cproj * H(P, Q);
          dmax = std::max(dmax, std::abs(d));
          PiP(iq_gamma, P, Q) -= d;
        }
      // the projection must actually remove something, or this control is blind
      REQUIRE(dmax > 0.05 * pmax);

      nda::array<cplx, 3> DwP(nk, Np, Np);
      solvers::vertex_detail::build_delta_w(W0, PiP, mdl.qmin, DwP);
      nda::array<cplx, 5> SLrP(nt, ns, nk, nbnd, nbnd);
      solvers::vertex_detail::eval_sigma_C_response(comm, G0, mdl.X_skPa, DwP, mdl.kmq,
                                                    mdl.qmin, SLrP);
      const cplx t_resp_p = pairing(SLrP, dG, nbnd, nbnd);
      const double rel_p =
          std::abs(dphi_fd - (t_expl + t_resp_p)) / std::max(std::abs(dphi_fd), 1e-30);
      app_log(1, "fdoracle_bl_gside [{}]: HEAD-PROJECTION control -> removed "
                 "max|c H| = {:.4e} of max|Pi^L(Gamma)| = {:.4e} ({:.2f} %); "
                 "Sigma^(L,r) pairing {:.6e} -> {:.6e} ({:.4f}x); "
                 "identity rel {:.3e} -> {:.3e}",
              prec, dmax, pmax, (pmax > 0.0 ? 100.0 * dmax / pmax : 0.0),
              t_resp.real(), t_resp_p.real(),
              (std::abs(t_resp) > 0.0 ? std::abs(t_resp_p) / std::abs(t_resp) : 0.0),
              rel, rel_p);
      // THE ASSERTION: projecting the head channel out of Pi^L BREAKS the B-L G-side
      // functional-derivative identity. If this ever stops firing, the projection has
      // become a no-op (or the oracle has gone blind) and that must be understood before
      // the projection can be defended as a legal approximation.
      REQUIRE(rel_p > 1e-3);
    }

    // ---- POSITIVE CONTROLS -----------------------------------------------------------
    // "drop the -Sx term of eq:sigmaBL -> the G-oracle breaks" and the response controls.
    {
      const double r_drop = std::abs(dphi_fd - t_expl) / std::abs(dphi_fd);
      app_log(1, "fdoracle_bl_gside [{}]: CONTROL drop Sigma^(L,r) -> rel = {:.3e} "
                 "(response share {:.4f})", prec, r_drop,
              std::abs(t_resp) / std::abs(t_expl));
      REQUIRE(r_drop > 1e-3);
    }
    {
      const cplx pred_flip = t_expl - t_resp;
      const double r_flip = std::abs(dphi_fd - pred_flip) / std::abs(dphi_fd);
      app_log(1, "fdoracle_bl_gside [{}]: CONTROL sign-flip Sigma^(L,r) -> rel = {:.3e}",
              prec, r_flip);
      REQUIRE(r_flip > 1e-3);
    }
    {   // the notes' UNTRANSPOSED sandwich: Delta w^L built from Pi^L without the
        // transpose/symmetrization -- the S0 routing correction, re-checked for B-L
      nda::array<cplx, 3> DwNT(nk, Np, Np);
      DwNT() = cplx(0.0);
      for (long q = 0; q < nk; ++q)
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q) {
            cplx acc(0.0);
            for (long r = 0; r < Np; ++r)
              for (long t = 0; t < Np; ++t)
                acc += W0(q, P, r) * PiL(q, r, t) * W0(q, t, Q);
            DwNT(q, P, Q) = acc;
          }
      nda::array<cplx, 5> SNT(nt, ns, nk, nbnd, nbnd);
      solvers::vertex_detail::eval_sigma_C_response(comm, G0, mdl.X_skPa, DwNT, mdl.kmq,
                                                    mdl.qmin, SNT);
      const cplx pred_nt = t_expl + pairing(SNT, dG, nbnd, nbnd);
      const double r_nt = std::abs(dphi_fd - pred_nt) / std::abs(dphi_fd);
      app_log(1, "fdoracle_bl_gside [{}]: CONTROL untransposed W0.Pi^L.W0 sandwich -> "
                 "rel = {:.3e}", prec, r_nt);
    }
#endif
  }

} // bdft_tests
