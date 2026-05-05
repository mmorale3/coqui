/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_GW_DRIVER_HPP
#define COQUI_REAL_AXIS_GW_DRIVER_HPP

#include <chrono>
#include <complex>
#include <string>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/check.hpp"

#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_conv.hpp"
#include "methods/GW_real_axis/real_axis_mb_state.hpp"
#include "methods/GW_real_axis/real_axis_thc_project.hpp"
#include "methods/GW_real_axis/real_axis_pi.hpp"
#include "methods/GW_real_axis/real_axis_dyson.hpp"
#include "methods/GW_real_axis/real_axis_sigma.hpp"

namespace methods {
namespace real_axis {

/**
 * Serial single-node one-shot G0W0 driver in the THC auxiliary basis.
 *
 * Layout convention: every auxiliary-basis quantity stores its frequency
 * axis innermost,
 *
 *     A_aux_skPQw[s, k, P, Q, iw],   ImPi_qPQO[q, P, Q, iO],   B_qPQO[q, P, Q, iO],
 *
 * matching what the FINUFFT-batched polarization / self-energy kernels and the
 * Hilbert-transform engine want. With this layout each per-(k, q) call into
 * a kernel is a contiguous slice of the storage array, so the driver carries
 * no transpose copies in its hot loops.
 *
 * Inputs (all dense nda arrays on the local rank):
 *
 *   A_skwij    : (ns, nkpts, N_w, nbnd, nbnd) fermionic spectral function
 *                A_{munu}(k, w) = -(1/pi) Im G^R_{munu}(k, w).
 *   X_skPmu    : (ns, nkpts, Naux, nbnd)      THC factor X(k)[P, mu].
 *   V_qPQ      : (nqpts, Naux, Naux)          auxiliary-basis Coulomb V(q).
 *   kpq_to_kp  : (nkpts, nqpts)               kp = k + q (BZ index).
 *   kmq_to_kp  : (nkpts, nqpts)               kp = k - q (BZ index).
 *   q_weights  : (nqpts,)                     BZ weight of each q.
 *
 * Outputs (allocated by caller, accumulated into):
 *
 *   ImSigma_c_skwij : (ns, nkpts, N_w, nbnd, nbnd) Im Sigma^c (orbital basis)
 *   ReSigma_c_skwij : (ns, nkpts, N_w, nbnd, nbnd) Re Sigma^c (orbital basis)
 *
 * Workflow (Algorithm 1 of v2 notes, single iteration):
 *   1. For each (s, k), project A in orbital basis to A_{PQ}(k, w).
 *   2. For each (q, k), accumulate Im Pi_{PQ}(q, Omega) via spectral
 *      cross-correlation kernel.
 *   3. Recover Re Pi via batched Hilbert transform.
 *   4. Solve Dyson: W_{PQ}(q, Omega) = [I - V(q) Pi(q, Omega)]^{-1} V(q).
 *   5. Form bosonic spectral function B = -1/pi Im W.
 *   6. For each (s, k, q), accumulate Im Sigma^c_{PQ}(k, w) (NUFFT kernel).
 *   7. Recover Re Sigma^c via batched Hilbert transform.
 *   8. For each (s, k), back-project Sigma_{PQ}(k, w) -> Sigma_{munu}(k, w).
 *
 * NOTE: Step-1 of MPI distribution: the comm is plumbed through the API but
 * the body still does the full computation redundantly on every rank. Loop
 * partitioning + reductions land in subsequent steps. Multi-rank runs
 * therefore produce identical results on every rank.
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
inline void evaluate_serial(boost::mpi3::communicator& comm,
                            real_freq_grid_t const& grid,
                            memory::array<MEM, ComplexType, 5> const& A_skwij,
                            memory::array<MEM, ComplexType, 4> const& X_skPmu,
                            memory::array<MEM, ComplexType, 3> const& V_qPQ,
                            nda::array<long, 2>                const& kpq_to_kp,
                            nda::array<long, 2>                const& kmq_to_kp,
                            nda::array<double, 1>              const& q_weights,
                            memory::array<MEM, ComplexType, 5>       & ImSigma_c_skwij,
                            memory::array<MEM, ComplexType, 5>       & ReSigma_c_skwij,
                            double eps_nufft = 1e-10,
                            long iq_gamma = -1,
                            bool verbose = false,
                            memory::array<MEM, ComplexType, 2> const& f_Rk = memory::array<MEM, ComplexType, 2>{},
                            memory::array<MEM, ComplexType, 2> const& f_qR = memory::array<MEM, ComplexType, 2>{},
                            memory::array<MEM, ComplexType, 2> const& f_Rq = memory::array<MEM, ComplexType, 2>{},
                            memory::array<MEM, ComplexType, 2> const& f_kR = memory::array<MEM, ComplexType, 2>{})
{
  // The lower-level array-API driver runs on host arrays internally and
  // is reachable from MEM-templated callers. Treat MEM as a template
  // marker; the body uses memory::array<MEM, ...> for I/O but stages
  // on-host scratch tensors.
  if constexpr (MEM != HOST_MEMORY) {
    auto A_h    = nda::to_host(A_skwij);
    auto X_h    = nda::to_host(X_skPmu);
    auto V_h    = nda::to_host(V_qPQ);
    nda::array<ComplexType, 5> ImSigma_h(ImSigma_c_skwij.shape());
    nda::array<ComplexType, 5> ReSigma_h(ReSigma_c_skwij.shape());
    ImSigma_h() = ComplexType(0.0, 0.0);
    ReSigma_h() = ComplexType(0.0, 0.0);
    nda::array<ComplexType, 2> f_Rk_h, f_qR_h, f_Rq_h, f_kR_h;
    if (f_Rk.size() > 0) f_Rk_h = nda::to_host(f_Rk);
    if (f_qR.size() > 0) f_qR_h = nda::to_host(f_qR);
    if (f_Rq.size() > 0) f_Rq_h = nda::to_host(f_Rq);
    if (f_kR.size() > 0) f_kR_h = nda::to_host(f_kR);
    evaluate_serial<HOST_MEMORY>(comm, grid, A_h, X_h, V_h,
                                 kpq_to_kp, kmq_to_kp, q_weights,
                                 ImSigma_h, ReSigma_h,
                                 eps_nufft, iq_gamma, verbose,
                                 f_Rk_h, f_qR_h, f_Rq_h, f_kR_h);
    ImSigma_c_skwij = ImSigma_h;
    ReSigma_c_skwij = ReSigma_h;
    return;
  }
  // Phase-2 R-space options.
  //  - Step 2 (Im Pi) runs in R-space when f_Rk (NR, Nk) and f_qR (Nq, NR)
  //    are supplied. Identity: with the kernel's (P,Q)<->(Q,P) swap on the
  //    second leg, Pi(q, P, Q, Omega) = sum_R exp(+i q.R) K[A(R, P, Q),
  //    A(R, Q, P)](Omega).
  //  - Step 6 (Im Sigma_c) runs in R-space when, additionally, f_Rq (R, q)
  //    and f_kR (k, R) are supplied. Identity: with the kernel linear in
  //    both legs at fixed (P, Q),
  //    Im Sigma(k, P, Q, w) = sum_R exp(+i k.R) K[A(R, P, Q), B(R, P, Q)](w).
  //  - Otherwise the existing k-space distributions are used.
  const bool use_rspace_pi    = (f_Rk.size() > 0 and f_qR.size() > 0);
  const bool use_rspace_sigma = use_rspace_pi
                                and (f_Rq.size() > 0 and f_kR.size() > 0);
  // Shape sanity.
  const long ns    = A_skwij.shape()[0];
  const long Nk    = A_skwij.shape()[1];
  const long N_w   = A_skwij.shape()[2];
  const long nbnd  = A_skwij.shape()[3];
  const long Nq    = V_qPQ.shape()[0];
  const long Naux  = V_qPQ.shape()[1];
  const long N_O   = grid.N_Omega();

  utils::check(A_skwij.shape()[4] == nbnd, "evaluate_serial: A nbnd mismatch");
  utils::check(X_skPmu.shape()[0] == ns,   "evaluate_serial: X ns mismatch");
  utils::check(X_skPmu.shape()[1] == Nk,   "evaluate_serial: X Nk mismatch");
  utils::check(X_skPmu.shape()[2] == Naux, "evaluate_serial: X Naux mismatch");
  utils::check(X_skPmu.shape()[3] == nbnd, "evaluate_serial: X nbnd mismatch");
  utils::check(V_qPQ.shape()[2] == Naux,   "evaluate_serial: V not square in (P,Q)");
  utils::check(kpq_to_kp.shape()[0] == Nk and kpq_to_kp.shape()[1] == Nq,
               "evaluate_serial: kpq_to_kp shape mismatch");
  utils::check(kmq_to_kp.shape()[0] == Nk and kmq_to_kp.shape()[1] == Nq,
               "evaluate_serial: kmq_to_kp shape mismatch");
  utils::check(q_weights.shape()[0] == Nq, "evaluate_serial: q_weights size mismatch");
  utils::check(N_w == grid.N_w(),
               "evaluate_serial: N_w mismatch with grid");
  utils::check(ImSigma_c_skwij.shape() == A_skwij.shape() and
               ReSigma_c_skwij.shape() == A_skwij.shape(),
               "evaluate_serial: Sigma output shapes must match A");

  using nda::range;
  const auto _ = range::all;
  using clock_t = std::chrono::steady_clock;
  auto t_now = []{ return clock_t::now(); };
  auto sec_since = [](clock_t::time_point t0) {
    return std::chrono::duration<double>(clock_t::now() - t0).count();
  };
  const auto t_total = t_now();

  // Engine sized to handle the largest batched cross-correlation (Naux*Naux).
  const auto t_conv0 = t_now();
  real_axis_conv_t conv(grid, /*ntrans*/ Naux*Naux, eps_nufft);
  const double dt_conv = sec_since(t_conv0);

  // --------------------------------------------------------------------
  // Step 1: Project A(k) -> A_{PQ}(k, w) for all (s, k).
  // --------------------------------------------------------------------
  // Storage A_aux_skPQw with iw innermost; slices into projection routines.
  const auto t1 = t_now();
  nda::array<ComplexType, 5> A_aux_skPQw(ns, Nk, Naux, Naux, N_w);
  for (long s = 0; s < ns; ++s)
    for (long ik = 0; ik < Nk; ++ik) {
      auto X_view     = X_skPmu (s, ik, _, _);    // (Naux, nbnd) view
      auto A_view     = A_skwij (s, ik, _, _, _); // (N_w, nbnd, nbnd) view
      auto A_aux_view = A_aux_skPQw(s, ik, _, _, _); // (Naux, Naux, N_w) view
      primary_to_aux_one_k(X_view, A_view, A_aux_view);
    }
  const double dt1 = sec_since(t1);

  // --------------------------------------------------------------------
  // Step 2: Im Pi_{PQ}(q, Omega) via spectral cross-correlation.
  // Two implementations:
  //  - k-space: distributed over (iq, s, ik), allreduced.
  //  - R-space: FT A k->R (gemm), per-R cross-correlation distributed
  //    over R, FT result R->q (gemm). Trades one k-loop for two BLAS-3
  //    FTs and reduces NUFFT calls from ns*Nk*Nq to ns*NR.
  // --------------------------------------------------------------------
  const auto t2 = t_now();
  nda::array<ComplexType, 4> ImPi_qPQO(Nq, Naux, Naux, N_O);
  ImPi_qPQO = ComplexType(0.0, 0.0);

  if (use_rspace_pi) {
    const long NR = f_Rk.shape()[0];
    utils::check(f_Rk.shape()[1] == Nk,
                 "evaluate_serial: f_Rk shape ({}, {}) does not match Nk={}",
                 f_Rk.shape()[0], f_Rk.shape()[1], Nk);
    utils::check(f_qR.shape()[0] == Nq and f_qR.shape()[1] == NR,
                 "evaluate_serial: f_qR shape ({}, {}) does not match (Nq, NR)=({}, {})",
                 f_qR.shape()[0], f_qR.shape()[1], Nq, NR);

    // FT A_aux from k-space to R-space, per s.
    nda::array<ComplexType, 5> A_aux_sRPQw(ns, NR, Naux, Naux, N_w);
    for (long s = 0; s < ns; ++s) {
      auto A_in_2D  = nda::reshape(A_aux_skPQw(s, _, _, _, _),
                                   std::array<long,2>{Nk, Naux*Naux*N_w});
      auto A_out_2D = nda::reshape(A_aux_sRPQw(s, _, _, _, _),
                                   std::array<long,2>{NR, Naux*Naux*N_w});
      nda::blas::gemm(ComplexType(1.0, 0.0), f_Rk, A_in_2D,
                      ComplexType(0.0, 0.0), A_out_2D);
    }

    // Per-R cross-correlation, distributed over R.
    nda::array<ComplexType, 4> ImPi_RPQO(NR, Naux, Naux, N_O);
    ImPi_RPQO = ComplexType(0.0, 0.0);
    {
      const int rank = comm.rank();
      const int size = comm.size();
      for (long iR = rank; iR < NR; iR += size) {
        for (long s = 0; s < ns; ++s) {
          auto A_view      = A_aux_sRPQw(s, iR, _, _, _);
          auto ImPi_R_view = ImPi_RPQO(iR, _, _, _);
          // k_weight = 1: the 1/Nk normalization is absorbed into f_Rk.
          accumulate_ImPi_one_kq(conv, A_view, A_view, ImPi_R_view, 1.0);
        }
      }
      if (size > 1)
        comm.all_reduce_in_place_n(ImPi_RPQO.data(), ImPi_RPQO.size(), std::plus<>{});
    }

    // FT ImPi from R-space to q-space.
    {
      auto ImPi_R_2D = nda::reshape(ImPi_RPQO, std::array<long,2>{NR, Naux*Naux*N_O});
      auto ImPi_q_2D = nda::reshape(ImPi_qPQO, std::array<long,2>{Nq, Naux*Naux*N_O});
      nda::blas::gemm(ComplexType(1.0, 0.0), f_qR, ImPi_R_2D,
                      ComplexType(0.0, 0.0), ImPi_q_2D);
    }

    // ignore_g0: the k-space path zeros out Pi at iq_gamma; reproduce here.
    if (iq_gamma >= 0) {
      for (long P = 0; P < Naux; ++P)
        for (long Q = 0; Q < Naux; ++Q)
          for (long iO = 0; iO < N_O; ++iO)
            ImPi_qPQO(iq_gamma, P, Q, iO) = ComplexType(0.0, 0.0);
    }
  } else {
    const double k_weight = 1.0 / static_cast<double>(Nk);
    const long total_qsk = Nq * ns * Nk;
    const int rank = comm.rank();
    const int size = comm.size();
    for (long idx = rank; idx < total_qsk; idx += size) {
      const long iq  = idx / (ns * Nk);
      const long rem = idx % (ns * Nk);
      const long s   = rem / Nk;
      const long ik  = rem % Nk;
      if (iq == iq_gamma) continue;     // ignore_g0 — already zero.
      const long ikq = kpq_to_kp(ik, iq);
      auto Ak_view   = A_aux_skPQw(s, ik,  _, _, _);
      auto Akq_view  = A_aux_skPQw(s, ikq, _, _, _);
      auto ImPi_view = ImPi_qPQO(iq, _, _, _);
      accumulate_ImPi_one_kq(conv, Ak_view, Akq_view, ImPi_view, k_weight);
    }
    if (size > 1)
      comm.all_reduce_in_place_n(ImPi_qPQO.data(), ImPi_qPQO.size(), std::plus<>{});
  }

  const double dt2 = sec_since(t2);

  // --------------------------------------------------------------------
  // Step 3: Re Pi via Hilbert transform on the bosonic grid. Distributed
  // over iq by rank; allreduced.
  // --------------------------------------------------------------------
  const auto t3 = t_now();
  nda::array<ComplexType, 4> RePi_qPQO(Nq, Naux, Naux, N_O);
  RePi_qPQO = ComplexType(0.0, 0.0);
  {
    const int rank = comm.rank();
    const int size = comm.size();
    nda::array<double, 3> ImPi_PQ_O(Naux, Naux, N_O);
    nda::array<double, 3> RePi_PQ_O(Naux, Naux, N_O);
    for (long iq = rank; iq < Nq; iq += size) {
      if (iq == iq_gamma) continue;
      auto ImPi_view = ImPi_qPQO(iq, _, _, _);
      for (long P = 0; P < Naux; ++P)
        for (long Q = 0; Q < Naux; ++Q)
          for (long iO = 0; iO < N_O; ++iO)
            ImPi_PQ_O(P, Q, iO) = ImPi_view(P, Q, iO).real();
      RePi_from_ImPi(conv, ImPi_PQ_O, RePi_PQ_O);
      auto RePi_view = RePi_qPQO(iq, _, _, _);
      for (long P = 0; P < Naux; ++P)
        for (long Q = 0; Q < Naux; ++Q)
          for (long iO = 0; iO < N_O; ++iO)
            RePi_view(P, Q, iO) = ComplexType(RePi_PQ_O(P, Q, iO), 0.0);
    }
    if (size > 1)
      comm.all_reduce_in_place_n(RePi_qPQO.data(), RePi_qPQO.size(), std::plus<>{});
  }

  const double dt3 = sec_since(t3);

  // --------------------------------------------------------------------
  // Step 4: Solve Dyson for W per (q, Omega). Distributed over iq.
  // --------------------------------------------------------------------
  const auto t4 = t_now();
  nda::array<ComplexType, 4> ImW_qPQO(Nq, Naux, Naux, N_O);
  nda::array<ComplexType, 4> ReW_qPQO(Nq, Naux, Naux, N_O);
  ImW_qPQO = ComplexType(0.0, 0.0);
  ReW_qPQO = ComplexType(0.0, 0.0);
  {
    const int rank = comm.rank();
    const int size = comm.size();
    nda::array<ComplexType, 2> Vmat(Naux, Naux);
    nda::array<ComplexType, 2> Pi(Naux, Naux);
    nda::array<ComplexType, 2> W(Naux, Naux);
    for (long iq = rank; iq < Nq; iq += size) {
      for (long P = 0; P < Naux; ++P)
        for (long Q = 0; Q < Naux; ++Q)
          Vmat(P, Q) = V_qPQ(iq, P, Q);
      for (long iO = 0; iO < N_O; ++iO) {
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q)
            Pi(P, Q) = ComplexType(RePi_qPQO(iq, P, Q, iO).real(),
                                   ImPi_qPQO(iq, P, Q, iO).real());
        solve_dyson_W_aux(Vmat, Pi, W);
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q) {
            ReW_qPQO(iq, P, Q, iO) = ComplexType(W(P, Q).real(), 0.0);
            ImW_qPQO(iq, P, Q, iO) = ComplexType(W(P, Q).imag(), 0.0);
          }
      }
    }
    if (size > 1) {
      comm.all_reduce_in_place_n(ReW_qPQO.data(), ReW_qPQO.size(), std::plus<>{});
      comm.all_reduce_in_place_n(ImW_qPQO.data(), ImW_qPQO.size(), std::plus<>{});
    }
  }

  const double dt4 = sec_since(t4);

  // --------------------------------------------------------------------
  // Step 5: Bosonic spectral function B = -1/pi Im W.
  // --------------------------------------------------------------------
  const auto t5 = t_now();
  nda::array<ComplexType, 4> B_qPQO(Nq, Naux, Naux, N_O);
  // Step 5: B = -Im W / pi. MEM-agnostic via nda::map (host/device).
  B_qPQO = nda::map([](ComplexType w) {
    return ComplexType(-w.real() / M_PI, 0.0);
  })(ImW_qPQO);

  const double dt5 = sec_since(t5);

  // --------------------------------------------------------------------
  // Step 6: Im Sigma_{PQ}(k, w) accumulated over q (NUFFT kernel).
  // Two implementations:
  //  - k-space: distributed over (s, ik, iq), allreduced.
  //  - R-space: FT A k->R and B q->R via gemm, per-R convolution
  //    distributed over R, FT result R->k via gemm. Trades the (k, q)
  //    coupling for two BLAS-3 FTs and reduces NUFFT calls from
  //    ns*Nk*Nq to ns*NR.
  // --------------------------------------------------------------------
  const auto t6 = t_now();
  nda::array<ComplexType, 5> ImSigma_aux_skPQw(ns, Nk, Naux, Naux, N_w);
  ImSigma_aux_skPQw = ComplexType(0.0, 0.0);

  if (use_rspace_sigma) {
    const long NR = f_Rk.shape()[0];
    utils::check(f_Rq.shape()[0] == NR and f_Rq.shape()[1] == Nq,
                 "evaluate_serial: f_Rq shape ({}, {}) does not match (NR, Nq)=({}, {})",
                 f_Rq.shape()[0], f_Rq.shape()[1], NR, Nq);
    utils::check(f_kR.shape()[0] == Nk and f_kR.shape()[1] == NR,
                 "evaluate_serial: f_kR shape ({}, {}) does not match (Nk, NR)=({}, {})",
                 f_kR.shape()[0], f_kR.shape()[1], Nk, NR);

    // ignore_g0: zero out B at iq_gamma so it doesn't contribute.
    if (iq_gamma >= 0) {
      for (long P = 0; P < Naux; ++P)
        for (long Q = 0; Q < Naux; ++Q)
          for (long iO = 0; iO < N_O; ++iO)
            B_qPQO(iq_gamma, P, Q, iO) = ComplexType(0.0, 0.0);
    }

    // FT B from q-space to R-space.
    nda::array<ComplexType, 4> B_RPQO(NR, Naux, Naux, N_O);
    {
      auto B_q_2D = nda::reshape(B_qPQO, std::array<long,2>{Nq, Naux*Naux*N_O});
      auto B_R_2D = nda::reshape(B_RPQO, std::array<long,2>{NR, Naux*Naux*N_O});
      nda::blas::gemm(ComplexType(1.0, 0.0), f_Rq, B_q_2D,
                      ComplexType(0.0, 0.0), B_R_2D);
    }

    // Per-R convolution (re-uses the already-built A_aux_sRPQw if available
    // -- but we don't keep it past Step 2, so rebuild here to keep the
    // implementations independent. Cheap: one gemm per s.)
    nda::array<ComplexType, 5> A_aux_sRPQw(ns, NR, Naux, Naux, N_w);
    for (long s = 0; s < ns; ++s) {
      auto A_in_2D  = nda::reshape(A_aux_skPQw(s, _, _, _, _),
                                   std::array<long,2>{Nk, Naux*Naux*N_w});
      auto A_out_2D = nda::reshape(A_aux_sRPQw(s, _, _, _, _),
                                   std::array<long,2>{NR, Naux*Naux*N_w});
      nda::blas::gemm(ComplexType(1.0, 0.0), f_Rk, A_in_2D,
                      ComplexType(0.0, 0.0), A_out_2D);
    }

    nda::array<ComplexType, 5> ImSigma_aux_sRPQw(ns, NR, Naux, Naux, N_w);
    ImSigma_aux_sRPQw = ComplexType(0.0, 0.0);
    {
      const int rank = comm.rank();
      const int size = comm.size();
      for (long iR = rank; iR < NR; iR += size) {
        for (long s = 0; s < ns; ++s) {
          auto A_view   = A_aux_sRPQw(s, iR, _, _, _);
          auto B_view   = B_RPQO(iR, _, _, _);
          auto Sig_view = ImSigma_aux_sRPQw(s, iR, _, _, _);
          // q_weight = 1: the 1/Nq normalization is absorbed into f_Rq.
          accumulate_ImSigma_one_kq_nufft(conv, A_view, B_view, Sig_view, 1.0);
        }
      }
      if (size > 1)
        comm.all_reduce_in_place_n(ImSigma_aux_sRPQw.data(),
                                    ImSigma_aux_sRPQw.size(), std::plus<>{});
    }

    // FT ImSigma from R-space to k-space, per s.
    for (long s = 0; s < ns; ++s) {
      auto Sig_R_2D = nda::reshape(ImSigma_aux_sRPQw(s, _, _, _, _),
                                   std::array<long,2>{NR, Naux*Naux*N_w});
      auto Sig_k_2D = nda::reshape(ImSigma_aux_skPQw(s, _, _, _, _),
                                   std::array<long,2>{Nk, Naux*Naux*N_w});
      nda::blas::gemm(ComplexType(1.0, 0.0), f_kR, Sig_R_2D,
                      ComplexType(0.0, 0.0), Sig_k_2D);
    }
  } else {
    const long total_skq = ns * Nk * Nq;
    const int rank = comm.rank();
    const int size = comm.size();
    for (long idx = rank; idx < total_skq; idx += size) {
      const long s   = idx / (Nk * Nq);
      const long rem = idx % (Nk * Nq);
      const long ik  = rem / Nq;
      const long iq  = rem % Nq;
      if (iq == iq_gamma) continue;
      const long ikmq = kmq_to_kp(ik, iq);
      auto A_view   = A_aux_skPQw(s, ikmq, _, _, _);
      auto B_view   = B_qPQO(iq, _, _, _);
      auto Sig_view = ImSigma_aux_skPQw(s, ik, _, _, _);
      accumulate_ImSigma_one_kq_nufft(conv, A_view, B_view,
                                      Sig_view, q_weights(iq));
    }
    if (size > 1)
      comm.all_reduce_in_place_n(ImSigma_aux_skPQw.data(),
                                  ImSigma_aux_skPQw.size(), std::plus<>{});
  }

  const double dt6 = sec_since(t6);

  // --------------------------------------------------------------------
  // Step 7: Re Sigma^c via Hilbert transform on the fermionic grid.
  // --------------------------------------------------------------------
  const auto t7 = t_now();
  nda::array<ComplexType, 5> ReSigma_aux_skPQw(ns, Nk, Naux, Naux, N_w);
  for (long s = 0; s < ns; ++s)
    for (long ik = 0; ik < Nk; ++ik) {
      auto Im_view = ImSigma_aux_skPQw(s, ik, _, _, _);
      auto Re_view = ReSigma_aux_skPQw(s, ik, _, _, _);
      ReSigma_from_ImSigma_aux(conv, Im_view, Re_view);
    }

  const double dt7 = sec_since(t7);

  // --------------------------------------------------------------------
  // Step 8: Back-project Sigma_{PQ}(k, w) -> Sigma_{munu}(k, w) using X(k).
  // --------------------------------------------------------------------
  const auto t8 = t_now();
  ImSigma_c_skwij = ComplexType(0.0, 0.0);
  ReSigma_c_skwij = ComplexType(0.0, 0.0);
  for (long s = 0; s < ns; ++s)
    for (long ik = 0; ik < Nk; ++ik) {
      auto X_view  = X_skPmu(s, ik, _, _);
      auto Im_view = ImSigma_aux_skPQw(s, ik, _, _, _);
      auto Re_view = ReSigma_aux_skPQw(s, ik, _, _, _);
      auto ImOut   = ImSigma_c_skwij(s, ik, _, _, _);
      auto ReOut   = ReSigma_c_skwij(s, ik, _, _, _);
      aux_to_primary_one_k(X_view, Im_view, ImOut);
      aux_to_primary_one_k(X_view, Re_view, ReOut);
    }
  const double dt8 = sec_since(t8);

  if (verbose and comm.root()) {
    const double dt_total = sec_since(t_total);
    app_log(2, "[evaluate_serial] timings (seconds), Naux={}, N_w={}, N_O={}, "
                "Nk={}, Nq={}, ns={}, nbnd={}",
            Naux, N_w, N_O, Nk, Nq, ns, nbnd);
    app_log(2, "[evaluate_serial]   conv_t setup       : {0:8.3f}", dt_conv);
    app_log(2, "[evaluate_serial]   step 1 project     : {0:8.3f}", dt1);
    app_log(2, "[evaluate_serial]   step 2 Im Pi       : {0:8.3f}", dt2);
    app_log(2, "[evaluate_serial]   step 3 Re Pi (Hilb): {0:8.3f}", dt3);
    app_log(2, "[evaluate_serial]   step 4 Dyson W     : {0:8.3f}", dt4);
    app_log(2, "[evaluate_serial]   step 5 B = -ImW/pi : {0:8.3f}", dt5);
    app_log(2, "[evaluate_serial]   step 6 Im Sigma^c  : {0:8.3f}", dt6);
    app_log(2, "[evaluate_serial]   step 7 Re Sig (Hil): {0:8.3f}", dt7);
    app_log(2, "[evaluate_serial]   step 8 back-project: {0:8.3f}", dt8);
    app_log(2, "[evaluate_serial]   TOTAL              : {0:8.3f}", dt_total);
  }
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_GW_DRIVER_HPP
