/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_GW_THC_HPP
#define COQUI_REAL_AXIS_GW_THC_HPP

#include <complex>
#include <memory>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/check.hpp"

#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "utilities/kpoint_utils.hpp"
#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_mb_state.hpp"
#include "methods/GW_real_axis/real_axis_gw_driver.hpp"

namespace methods {
namespace real_axis {

/**
 * High-level driver: real-axis G0W0 in the THC auxiliary basis, using a
 * concrete THC ERI object (e.g. methods::thc_reader_t).
 *
 * Pulls from `thc`:
 *   - X(s, p=0, k)    -> auxiliary projection at each (s, k)
 *   - Z(iq)           -> auxiliary Coulomb V_PQ at each q (full-BZ, see note)
 *   - Np()            -> auxiliary rank
 *   - MF()            -> mean-field handle for k/q points and BZ closure
 *
 * Pulls from `MF`:
 *   - nspin(), nbnd(), nkpts(), nqpts()
 *   - qk_to_k2(iq, ik)   -> k - q index in the BZ
 *   - qminus(iq)         -> -q index in the BZ (used to build k+q map)
 *   - k_weight(ik)       -> Brillouin-zone weight of k (sums to 1)
 *
 * @param state       Real-axis many-body state. Reads `state.A_wskij`
 *                    (allocated by caller; layout (N_w, ns, nkpts, nbnd, nbnd)).
 *                    Allocates and writes `state.ImSigma_wskij` and
 *                    `state.ReSigma_wskij`.
 * @param thc         THC ERI object (must satisfy CoQui's reader interface).
 * @param eps_nufft   FINUFFT accuracy tolerance.
 *
 * Restrictions in this version:
 *   - npol == 1 assumed (no SOC / non-collinear support yet).
 *   - state.A_wskij stored on the FULL BZ (not just IBZ); symmetry expansion
 *     of the input is the caller's responsibility for now.
 *   - Step-1 of MPI distribution: the comm is plumbed through the API but
 *     the body still does the full computation redundantly on every rank.
 *     thc.X / thc.Z accesses are already MPI-aware (collectives over the
 *     thc communicator), so calling on every rank is consistent.
 *
 * The function is templated on the THC ERI type and constrained by the
 * THC_ERI concept.
 */
template<methods::THC_ERI THC_t>
void evaluate_thc_serial(boost::mpi3::communicator& comm,
                         real_axis_mb_state_t & state,
                         THC_t const& thc,
                         double eps_nufft = 1e-10,
                         std::string div_treatment = "ignore_g0",
                         bool verbose = false,
                         bool use_rspace = false)
{
  utils::check(state.A_wskij.has_value(),
               "evaluate_thc_serial: state.A_wskij not allocated");
  utils::check(state.grid != nullptr,
               "evaluate_thc_serial: state.grid not bound");

  auto const& grid = *state.grid;
  auto const& MF   = *thc.MF();

  const long ns    = MF.nspin();
  const long nbnd  = MF.nbnd();
  const long Nk    = MF.nkpts();
  const long Nq    = MF.nqpts();
  const long Naux  = thc.Np();
  const long N_w   = grid.N_w();

  utils::check(MF.npol() == 1,
               "evaluate_thc_serial: npol={} not supported (need 1)", MF.npol());

  // Repack input A from (N_w, ns, nkpts, nbnd, nbnd) to driver layout
  // (ns, nkpts, N_w, nbnd, nbnd).
  auto const& A_in = *state.A_wskij;
  utils::check(A_in.shape()[0] == N_w and A_in.shape()[1] == ns and
               A_in.shape()[2] == Nk and A_in.shape()[3] == nbnd and
               A_in.shape()[4] == nbnd,
               "evaluate_thc_serial: state.A_wskij shape mismatch");

  nda::array<ComplexType, 5> A(ns, Nk, N_w, nbnd, nbnd);
  for (long s = 0; s < ns; ++s)
    for (long k = 0; k < Nk; ++k)
      for (long iw = 0; iw < N_w; ++iw)
        for (long mu = 0; mu < nbnd; ++mu)
          for (long nu = 0; nu < nbnd; ++nu)
            A(s, k, iw, mu, nu) = A_in(iw, s, k, mu, nu);

  // X tensor: copy from THC reader's view to a local owning array.
  // thc.X(is, ip, ik) returns a (Naux, nbnd) view (q-independent).
  nda::array<ComplexType, 4> X(ns, Nk, Naux, nbnd);
  for (long s = 0; s < ns; ++s)
    for (long k = 0; k < Nk; ++k) {
      auto Xsk = thc.X(static_cast<int>(s), /*ip*/ 0, static_cast<int>(k));
      for (long P = 0; P < Naux; ++P)
        for (long mu = 0; mu < nbnd; ++mu)
          X(s, k, P, mu) = Xsk(P, mu);
    }

  // V tensor: V(iq, P, Q) = thc.Z(iq).
  nda::array<ComplexType, 3> V(Nq, Naux, Naux);
  for (long iq = 0; iq < Nq; ++iq) {
    auto Zq = thc.Z(static_cast<int>(iq));
    for (long P = 0; P < Naux; ++P)
      for (long Q = 0; Q < Naux; ++Q)
        V(iq, P, Q) = Zq(P, Q);
  }

  // BZ closure maps. CoQui's MF provides:
  //   qk_to_k2(iq, ik)   = ik - iq   (index of k - q in BZ)
  //   qminus(iq)         = index of -q in BZ
  // Hence:
  //   kpq_to_kp(ik, iq) = qk_to_k2(qminus(iq), ik)   (gives k + q)
  //   kmq_to_kp(ik, iq) = qk_to_k2(iq, ik)           (gives k - q)
  nda::array<long, 2> kpq(Nk, Nq), kmq(Nk, Nq);
  auto const& qk_to_k2 = MF.qk_to_k2();   // (nqpts, nkpts)
  auto const& qm       = MF.qminus();     // (nqpts,)
  for (long iq = 0; iq < Nq; ++iq)
    for (long ik = 0; ik < Nk; ++ik) {
      kmq(ik, iq) = qk_to_k2(iq, ik);
      kpq(ik, iq) = qk_to_k2(qm(iq), ik);
    }

  // q-mesh weights. With CoQui's BZ convention the q-weight equals the
  // k-weight of the same index for a uniform mesh; however the precise
  // mapping is system-dependent. For a serial single-rank pass we use
  // q_weights[iq] = 1/Nq (uniform) which is exact for a uniformly sampled
  // BZ. Replace with MF->q_weight(iq) once that accessor is added.
  nda::array<double, 1> qw(Nq);
  for (long iq = 0; iq < Nq; ++iq)
    qw(iq) = 1.0 / static_cast<double>(Nq);

  // Allocate outputs sized to driver layout, then transpose back at the end.
  nda::array<ComplexType, 5> ImSigma(ns, Nk, N_w, nbnd, nbnd);
  nda::array<ComplexType, 5> ReSigma(ns, Nk, N_w, nbnd, nbnd);
  ImSigma = ComplexType(0.0, 0.0);
  ReSigma = ComplexType(0.0, 0.0);

  // Identify the Gamma q-point for divergence handling.
  // CoQui convention: iq=0 typically corresponds to q=Gamma. Verify by
  // checking the L1 norm of Qpts at iq=0; if small, treat as Gamma.
  long iq_gamma = -1;
  if (div_treatment == "ignore_g0") {
    auto Qp = MF.Qpts();
    if (Qp.shape()[0] >= 1) {
      double norm0 = 0.0;
      for (long c = 0; c < Qp.shape()[1]; ++c) norm0 += std::abs(Qp(0, c));
      if (norm0 < 1e-10) iq_gamma = 0;
    }
  }

  // Optional R-space FT matrices. Pi needs (f_Rk, f_qR); Sigma additionally
  // needs (f_Rq, f_kR). Built only when use_rspace=true AND the system has
  // more than one k-point (the trivial Nk=1 case has no FT to do).
  nda::array<ComplexType, 2> f_Rk, f_qR, f_Rq, f_kR;
  if (use_rspace and Nk > 1) {
    auto kp_grid = MF.kp_grid();
    auto lattv   = MF.lattv();
    const long nx = kp_grid(0);
    const long ny = kp_grid(1);
    const long nz = kp_grid(2);
    const long NR = nx * ny * nz;
    utils::check(NR == Nk,
                 "evaluate_thc_serial: R-space path expects NR ({}) == Nk ({})",
                 NR, Nk);

    nda::array<long, 2> Rpts_idx(NR, 3);
    for (long p = 0; p < NR; ++p) {
      long a = p / (ny * nz);
      long b = (p / nz) % ny;
      long c = p % nz;
      if (a > nx / 2) a -= nx;
      if (b > ny / 2) b -= ny;
      if (c > nz / 2) c -= nz;
      Rpts_idx(p, 0) = a;
      Rpts_idx(p, 1) = b;
      Rpts_idx(p, 2) = c;
    }
    nda::array<long, 1> Rpts_weights(NR);
    Rpts_weights() = 1;

    f_Rk = nda::array<ComplexType, 2>(NR, Nk);
    f_qR = nda::array<ComplexType, 2>(Nq, NR);
    f_Rq = nda::array<ComplexType, 2>(NR, Nq);
    f_kR = nda::array<ComplexType, 2>(Nk, NR);
    utils::k_to_R_coefficients(Rpts_idx, MF.kpts(), lattv, f_Rk);
    utils::R_to_k_coefficients(Rpts_idx, Rpts_weights, MF.Qpts(), lattv, f_qR);
    // For Sigma: B is on q-grid, FT to R; Sigma on R, FT to k-grid.
    utils::k_to_R_coefficients(Rpts_idx, MF.Qpts(), lattv, f_Rq);
    utils::R_to_k_coefficients(Rpts_idx, Rpts_weights, MF.kpts(), lattv, f_kR);
  }

  evaluate_serial(comm, grid, A, X, V, kpq, kmq, qw, ImSigma, ReSigma, eps_nufft,
                  iq_gamma, verbose, f_Rk, f_qR, f_Rq, f_kR);

  // Repack outputs into state.{Im,Re}Sigma_wskij with (N_w, ns, nkpts, nbnd, nbnd) layout.
  state.ImSigma_wskij = nda::array<ComplexType, 5>(N_w, ns, Nk, nbnd, nbnd);
  state.ReSigma_wskij = nda::array<ComplexType, 5>(N_w, ns, Nk, nbnd, nbnd);
  auto & ImOut = *state.ImSigma_wskij;
  auto & ReOut = *state.ReSigma_wskij;
  for (long s = 0; s < ns; ++s)
    for (long k = 0; k < Nk; ++k)
      for (long iw = 0; iw < N_w; ++iw)
        for (long mu = 0; mu < nbnd; ++mu)
          for (long nu = 0; nu < nbnd; ++nu) {
            ImOut(iw, s, k, mu, nu) = ImSigma(s, k, iw, mu, nu);
            ReOut(iw, s, k, mu, nu) = ReSigma(s, k, iw, mu, nu);
          }
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_GW_THC_HPP
