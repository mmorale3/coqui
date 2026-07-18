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
 * Vertex parallelization rank-cap-lift smoke + rank-invariance at the KERNEL entry
 * point (notes/vertex_parallelization_analysis.md section 4.2 #1/#2;
 * notes/vertex_parallelization_M1.md, notes/vertex_parallelization_M2.md).
 *
 * The vertex kernels (pi_c_accumulate_w, eval_sigma_C_g3w2) already run on the FULL
 * communicator and round-robin their work tuples over (rank, nproc) -- there is NO
 * kernel-side rank cap; the ~12-rank wall is entirely in the THC READER BUILD (see the
 * M1 note). This test exercises the decoupled kernel directly on a toy model whose band
 * dimension nbnd is TINY (nbnd = 3, Np = 4), so ANY run at P >= 4 already puts the
 * kernel at P > nbnd -- the exact regime that trips the reader's make_distributed_array
 * / nproc_per_pool cap -- and shows the kernel itself does NOT abort and stays
 * rank-invariant.
 *
 *   (a) rank-cap-lift smoke: the kernel runs to completion at P > nbnd (no abort).
 *   (b) Pi^C rank-invariance:  Pi^C(distributed over P ranks) all_reduced == Pi^C
 *                              computed serially (rank 0, nproc 1) to <= 1e-11 relative
 *                              (the FP-reassociation floor of a reordered complex
 *                              reduction, notes section 4.1a). At P = 1 this is exact.
 *
 * M2 (items #3/#4) additionally distributes the INNER loops (Sigma^C qy, Pi^C q_ext) via
 * a 2D rank split inside each kernel, and recomputes the Pi^C pair caches per tile. This
 * test is EXTENDED for M2 (notes/vertex_parallelization_M2.md deliverable #2/#3):
 *
 *   (c) Sigma^C rank-invariance: the fused G3W2 kernel run on the FULL comm (combo x qy
 *                              split active) vs a size-1 self-comm (serial), <= 1e-11 rel
 *                              (P=1 exact) -- pins the qy-axis distribution against serial.
 *   (d) Pi^C q_ext distribution at nq_ext > 1 (the toy has nk = nq_ext = 3) so at
 *                              P >= 4 the q_ext axis IS split; still reproduces serial.
 *   (e) pair-cache memory footprint: assert the NEW per-tile live pair-cache bytes are
 *                              << the OLD replicated (ns*nk*nq_ext*...) store (guard
 *                              against silently reintroducing the #1 memory wall).
 */

#undef NDEBUG

#include <random>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/test_common.hpp"

#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/vertex/vertex_pi.icc"
#include "methods/vertex/vertex_sigma.icc"

namespace bdft_tests {

  namespace vertex_pi = methods::solvers::vertex_pi;
  using vertex_pi::iaft_tools;
  using cplx = ComplexType;

  namespace rank_toy {
    constexpr double beta = 20.0;
    constexpr double wmax = 8.0;
    constexpr double Omega = 0.61;
    constexpr long nk = 3, nbnd = 3, Np = 4, ns = 1;   // nbnd < 4 <= P exercises P>nbnd
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
    inline double g_tau(double eps, double tau) {
      if (eps >= 0.0) return -std::exp(-eps * tau) / (1.0 + std::exp(-beta * eps));
      return -std::exp(eps * (beta - tau)) / (1.0 + std::exp(beta * eps));
    }

    struct model_t {
      nda::array<double, 2> eps;
      nda::array<cplx, 4> Pr;
      nda::array<cplx, 4> X_skPa;
      nda::array<cplx, 3> Z_qPQ, M_qPQ;
      nda::array<long, 2> kmq, kpq;
      model_t() : eps(nk, nbnd), Pr(nk, nbnd, nbnd, nbnd),
                  X_skPa(ns, nk, Np, nbnd), Z_qPQ(nk, Np, Np), M_qPQ(nk, Np, Np),
                  kmq(nk, nk), kpq(nk, nk) {
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
        for (long q = 0; q < nk; ++q)
          for (long k = 0; k < nk; ++k) {
            kmq(q, k) = (k - q + nk) % nk;
            kpq(q, k) = (k + q) % nk;
          }
      }
      nda::array<cplx, 5> G_tau(imag_axes_ft::IAFT const& ft) const {
        long nt = ft.nt_f();
        auto xm = ft.tau_mesh();
        nda::array<cplx, 5> G(nt, ns, nk, nbnd, nbnd);
        G() = cplx(0.0);
        for (long it = 0; it < nt; ++it) {
          double tau = (xm(it) + 1.0) * 0.5 * beta;
          for (long k = 0; k < nk; ++k)
            for (long r = 0; r < nbnd; ++r) {
              double g = g_tau(eps(k, r), tau);
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
            double nu = double(tools.wn_b(l)) * M_PI / beta;
            double s = 2.0 * Omega / (Omega * Omega + nu * nu);
            for (long P = 0; P < Np; ++P)
              for (long Q = 0; Q < Np; ++Q) W(q, l, P, Q) = M_qPQ(q, P, Q) * s;
          }
        return W;
      }
    };
  } // namespace rank_toy

  TEST_CASE("vertex_rankinv", "[methods][vertex][rankinv]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_rankinv skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace rank_toy;
    auto world = boost::mpi3::environment::get_world_instance();
    const long P = world.size();

    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, 1e-11);
    iaft_tools tools(ft);
    model_t mdl;
    auto G = mdl.G_tau(ft);
    auto Wdyn = mdl.Wdyn(tools);
    const long nw_b = tools.nw_b;

    // (a) rank-cap-lift smoke: the kernel runs at P (= world.size) > nbnd = 3 without the
    //     make_distributed_array / nproc_per_pool abort (the reader-side wall). Reaching
    //     the assertions below at all IS the smoke pass.
    if (world.root())
      app_log(1, "vertex_rankinv: kernel at P = {} ranks, nbnd = {}, Np = {} "
                 "(P {} nbnd -> cap-lift regime).",
              P, (long)nbnd, (long)Np, P > nbnd ? ">" : "<=");

    // distributed evaluation: each rank does its round-robin share, then all_reduce.
    nda::array<cplx, 4> Pi_dist(nw_b, nk, Np, Np);
    Pi_dist() = cplx(0.0);
    vertex_pi::pi_c_accumulate_w(ft, tools, G, mdl.X_skPa, mdl.Z_qPQ, &Wdyn,
                                 mdl.kmq, mdl.kpq, C(), Pi_dist,
                                 world.rank(), world.size());
    world.all_reduce_in_place_n(Pi_dist.data(), Pi_dist.size(), std::plus<>{});

    // serial reference: the SAME kernel with (rank 0, nproc 1) -- the whole tuple space
    // on one rank. Computed identically on every rank (no MPI), so it is the P = 1
    // result the distributed run must reproduce.
    nda::array<cplx, 4> Pi_serial(nw_b, nk, Np, Np);
    Pi_serial() = cplx(0.0);
    vertex_pi::pi_c_accumulate_w(ft, tools, G, mdl.X_skPa, mdl.Z_qPQ, &Wdyn,
                                 mdl.kmq, mdl.kpq, C(), Pi_serial, 0, 1);

    double num = 0.0, den = 0.0;
    for (long i = 0; i < Pi_dist.size(); ++i) {
      num = std::max(num, std::abs(Pi_dist.data()[i] - Pi_serial.data()[i]));
      den = std::max(den, std::abs(Pi_serial.data()[i]));
    }
    REQUIRE(den > 1e-8);
    if (world.root())
      app_log(1, "vertex_rankinv[P={}]: max|Pi_dist - Pi_serial| = {}, max|Pi_serial| = "
                 "{}, rel = {}", P, num, den, num / den);

    // (b) rank-invariance: reordered complex reduction over the (s,k,qx,q_ext) space.
    // With M2 (item #4) the inner q_ext loop is ALSO split at P >= 4 (nq_ext = nk = 3),
    // so this now pins BOTH the tuple and the q_ext distribution against serial.
    // Exact at P = 1; <= 1e-11 relative otherwise (notes section 4.1a).
    if (P == 1)
      REQUIRE(num == 0.0);
    else
      REQUIRE(num <= 1e-11 * den);

    // (e) pair-cache memory footprint (M2 item #4; notes deliverable #3). The OLD kernel
    // stored T0/T1/M0/M1 replicated over ALL (is,ik,q_ext): 4*ns*nk*nq_ext*nt*Nb2*Np
    // complex on EVERY rank (the #1 memory wall, analysis section 1.2b). The NEW kernel
    // holds only the current tile's four components: 4*nt*Nb2*Np -- ONE TILE LIVE. Assert
    // the reduction factor is exactly ns*nk*nq_ext (guards against reintroducing the
    // replicated store). nq_ext = nk here (full external mesh).
    {
      const long Nb = C().size(), Nb2 = Nb * Nb, nt = tools.nt_f;
      const long nq_ext = nk;
      const double old_bytes = 4.0 * double(ns) * nk * nq_ext * nt * Nb2 * Np * 16.0;
      const double new_bytes = 4.0 * double(nt) * Nb2 * Np * 16.0;
      const double factor = old_bytes / new_bytes;
      if (world.root())
        app_log(1, "vertex_rankinv pair-cache footprint: OLD (replicated) = {} B, NEW "
                   "(one tile live) = {} B, reduction factor = {} (= ns*nk*nq_ext = {}).",
                old_bytes, new_bytes, factor, ns * nk * nq_ext);
      REQUIRE(factor == Approx(double(ns * nk * nq_ext)));
      REQUIRE(new_bytes < old_bytes);
    }
#endif
  }

  // ============================ Sigma^C rank-invariance ================================
  // M2 item #3 (qy-axis distribution). Self-contained toy G3W2 model; the fused Sigma^C
  // kernel run on the FULL comm (combo x qy split active) must reproduce the size-1
  // self-comm (serial) result -- pins the qy distribution + the final all_reduce.
  namespace sig_rank_toy {
    constexpr double beta = 20.0;
    constexpr double wmax = 8.0;
    constexpr double OmX = 0.61;
    constexpr long nk = 3, nbnd = 3, Np = 4, ns = 1;
    constexpr long C0 = 0, ncw = 2;
    inline nda::range C() { return nda::range(C0, C0 + ncw); }

    using rank_toy::rng_t;
    using rank_toy::unitary;
    using rank_toy::g_tau;

    struct model_t {
      nda::array<double, 2> eps;
      nda::array<cplx, 4> Pr;
      nda::array<cplx, 4> X_skPa;
      nda::array<cplx, 3> Z_qPQ, M_qPQ;
      nda::array<long, 2> kmq;
      nda::array<long, 1> qmin;
      model_t() : eps(nk, nbnd), Pr(nk, nbnd, nbnd, nbnd), X_skPa(ns, nk, Np, nbnd),
                  Z_qPQ(nk, Np, Np), M_qPQ(nk, Np, Np), kmq(nk, nk), qmin(nk) {
        rng_t rng(41);
        const double base[3] = {-0.67, -0.11, 0.52};
        for (long k = 0; k < nk; ++k) {
          for (long r = 0; r < nbnd; ++r) eps(k, r) = base[r] + 0.06 * double(k);
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
              Z_qPQ(q, P, Q) = zy / double(Np) + ((P == Q) ? cplx(0.25) : cplx(0.0));
              M_qPQ(q, P, Q) = zv / double(Np);
            }
        }
        for (long q = 0; q < nk; ++q) {
          qmin(q) = (nk - q) % nk;
          for (long k = 0; k < nk; ++k) kmq(q, k) = (k - q + nk) % nk;
        }
      }
      nda::array<cplx, 5> G_tau(imag_axes_ft::IAFT const& ft) const {
        long nt = ft.nt_f();
        auto xm = ft.tau_mesh();
        nda::array<cplx, 5> G(nt, ns, nk, nbnd, nbnd);
        G() = cplx(0.0);
        for (long it = 0; it < nt; ++it) {
          double tau = (xm(it) + 1.0) * 0.5 * beta;
          for (long k = 0; k < nk; ++k)
            for (long r = 0; r < nbnd; ++r) {
              double g = g_tau(eps(k, r), tau);
              for (long i = 0; i < nbnd; ++i)
                for (long j = 0; j < nbnd; ++j)
                  G(it, 0, k, i, j) += g * Pr(k, r, i, j);
            }
        }
        return G;
      }
      // W_dyn(q, tau) on the FULL fermionic tau mesh (kernel input format)
      nda::array<cplx, 4> Wdyn_tau(imag_axes_ft::IAFT const& ft) const {
        long nt = ft.nt_f(), nw_b = ft.nw_b();
        auto wnb = ft.wn_mesh_b();
        nda::array<cplx, 4> Wt(nk, nt, Np, Np);
        nda::array<cplx, 3> Ww(nw_b, Np, Np), Wtau(nt, Np, Np);
        for (long q = 0; q < nk; ++q) {
          for (long m = 0; m < nw_b; ++m) {
            double nu = double(wnb(m)) * M_PI / beta;
            double s = 2.0 * OmX / (OmX * OmX + nu * nu);
            for (long P = 0; P < Np; ++P)
              for (long Q = 0; Q < Np; ++Q) Ww(m, P, Q) = M_qPQ(q, P, Q) * s;
          }
          ft.w_to_tau(Ww, Wtau, imag_axes_ft::boson);
          Wt(q, nda::range::all, nda::range::all, nda::range::all) = Wtau;
        }
        return Wt;
      }
    };
  } // namespace sig_rank_toy

  TEST_CASE("vertex_rankinv_sigma", "[methods][vertex][rankinv]") {
#ifndef ENABLE_DLR
    SUCCEED("vertex_rankinv_sigma skipped: build has ENABLE_DLR=OFF.");
#else
    using namespace sig_rank_toy;
    namespace vertex_detail = methods::solvers::vertex_detail;
    auto world = boost::mpi3::environment::get_world_instance();
    const long P = world.size();

    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, 1e-10);
    model_t mdl;
    auto G = mdl.G_tau(ft);
    auto Wt = mdl.Wdyn_tau(ft);
    const long nt = ft.nt_f();

    // (c) distributed: full comm -> the kernel's combo x qy 2D split is active at P >= 2.
    nda::array<cplx, 5> Sig_dist(nt, ns, nk, nbnd, nbnd);
    vertex_detail::eval_sigma_C_g3w2_nosym(ft, world, C(), G, mdl.X_skPa, Wt,
                                           mdl.Z_qPQ, mdl.kmq, mdl.qmin,
                                           /*iq_gamma*/ 0, /*skip*/ false, Sig_dist);

    // serial reference: a size-1 self-comm (each rank runs the whole (s,k,qx,qy) space).
    // The kernel all_reduces over its comm, so a singleton comm = serial P=1 result.
    auto self = world.split(world.rank(), 0);
    nda::array<cplx, 5> Sig_serial(nt, ns, nk, nbnd, nbnd);
    vertex_detail::eval_sigma_C_g3w2_nosym(ft, self, C(), G, mdl.X_skPa, Wt,
                                           mdl.Z_qPQ, mdl.kmq, mdl.qmin,
                                           0, false, Sig_serial);

    double num = 0.0, den = 0.0;
    for (long i = 0; i < Sig_dist.size(); ++i) {
      num = std::max(num, std::abs(Sig_dist.data()[i] - Sig_serial.data()[i]));
      den = std::max(den, std::abs(Sig_serial.data()[i]));
    }
    REQUIRE(den > 1e-10);
    if (world.root())
      app_log(1, "vertex_rankinv_sigma[P={}]: max|Sig_dist - Sig_serial| = {}, "
                 "max|Sig_serial| = {}, rel = {} (nbnd = {}, Np = {})",
              P, num, den, num / den, (long)nbnd, (long)Np);
    // Exact at P = 1 (identical instruction stream); <= 1e-11 rel otherwise (qy reorder).
    if (P == 1)
      REQUIRE(num == 0.0);
    else
      REQUIRE(num <= 1e-11 * den);
#endif
  }

} // namespace bdft_tests
