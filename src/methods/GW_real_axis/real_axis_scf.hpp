/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_SCF_HPP
#define COQUI_REAL_AXIS_SCF_HPP

#include <complex>
#include <iostream>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/check.hpp"

#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_gw_driver.hpp"
#include "methods/GW_real_axis/real_axis_sigma_x.hpp"
#include "methods/GW_real_axis/real_axis_dyson_G.hpp"
#include "methods/GW_real_axis/real_axis_diis.hpp"

namespace methods {
namespace real_axis {

/// Mixing strategy for the SCF iteration.
enum class scgw_mix_kind {
  linear,   ///< A_next = (1 - alpha) A_old + alpha A_full
  diis,     ///< Pulay DIIS with sliding window of size diis_window
};

/**
 * Configuration for the real-axis G0W0 / scGW driver.
 *
 *   max_iter:     1 -> G_0W_0; >1 -> scGW with mixing.
 *   alpha_mix:    mixing coefficient (linear) or DIIS damping on R^(i).
 *   tol:          stop when || dA ||_F < tol.
 *   eta:          Lorentzian broadening for the Dyson G update.
 *   eps_nufft:    FINUFFT accuracy tolerance.
 *   update_mu:    if true, update mu_chem each iteration to enforce N_elec.
 *   verbose:      print one line per iteration with diff and mu.
 *   iq_gamma:     q-index to drop (q=0 divergence). -1 = none.
 *   mix_kind:     linear or DIIS.
 *   diis_window:  number of history entries kept by DIIS (typical: 6-10).
 */
struct scgw_config {
  long          max_iter    = 1;
  double        alpha_mix   = 0.5;
  double        tol         = 1e-6;
  double        eta         = 1e-2;
  double        eps_nufft   = 1e-10;
  bool          update_mu   = true;
  bool          verbose     = false;
  long          iq_gamma    = -1;
  scgw_mix_kind mix_kind    = scgw_mix_kind::linear;
  long          diis_window = 8;
};

/**
 * Result of one scGW run.
 */
struct scgw_result {
  long   iter_used  = 0;
  double final_diff = 0.0;
  double final_mu   = 0.0;
  bool   converged  = false;
};

/**
 * Real-axis self-consistent GW driver.
 *
 * Step-1 of MPI distribution: the comm is plumbed through the API but the
 * body still does the full computation redundantly on every rank. Multi-rank
 * runs therefore produce identical results on every rank.
 *
 * Inputs:
 *   grid:           finite-T real-frequency grid; carries beta and INITIAL mu_chem.
 *   H_MF_skij:      (ns, Nk, nbnd, nbnd) one-body mean-field Hamiltonian.
 *   X_skPmu:        (ns, Nk, Naux, nbnd) THC factor.
 *   V_qPQ:          (Nq, Naux, Naux) auxiliary Coulomb.
 *   kpq_to_kp:      (Nk, Nq) BZ index of k+q.
 *   kmq_to_kp:      (Nk, Nq) BZ index of k-q.
 *   q_weights:      (Nq,) BZ weights of q (sum = 1).
 *   k_weights:      (Nk,) BZ weights of k (sum = 1).
 *   N_elec:         target electron count for mu update (per spin if ns=1
 *                   the convention is total N over all bands and k).
 *   cfg:            scgw_config.
 *
 * Outputs (allocated/written by the routine):
 *   A_wskij:        final spectral function (real part = -1/pi Im G^R).
 *   Sigma_x_skij:   static exchange self-energy.
 *   ImSigma_c_skwij, ReSigma_c_skwij: correlation self-energy.
 *
 * Returns: scgw_result with iteration counter, final residual, final mu,
 * and convergence flag.
 *
 * The driver does NOT itself construct an initial A (it expects A_wskij to
 * already contain a sensible starting point — typically a Lorentzian-broadened
 * mean-field spectral function); however if A_wskij is empty/zero, the
 * routine builds a Lorentzian initial A from H_MF and grid.eta.
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
inline scgw_result run_scgw_serial(
    boost::mpi3::communicator        & comm,
    real_freq_grid_t            const& grid_in,
    memory::array<MEM, ComplexType, 4> const& H_MF_skij,
    memory::array<MEM, ComplexType, 4> const& X_skPmu,
    memory::array<MEM, ComplexType, 3> const& V_qPQ,
    nda::array<long, 2>                const& kpq_to_kp,
    nda::array<long, 2>                const& kmq_to_kp,
    nda::array<double, 1>              const& q_weights,
    nda::array<double, 1>              const& k_weights,
    double                                    N_elec,
    scgw_config                        const& cfg,
    memory::array<MEM, ComplexType, 5>       & A_wskij,
    memory::array<MEM, ComplexType, 4>       & Sigma_x_skij,
    memory::array<MEM, ComplexType, 5>       & ImSigma_c_skwij,
    memory::array<MEM, ComplexType, 5>       & ReSigma_c_skwij,
    memory::array<MEM, ComplexType, 2> const& f_Rk = memory::array<MEM, ComplexType, 2>{},
    memory::array<MEM, ComplexType, 2> const& f_qR = memory::array<MEM, ComplexType, 2>{},
    memory::array<MEM, ComplexType, 2> const& f_Rq = memory::array<MEM, ComplexType, 2>{},
    memory::array<MEM, ComplexType, 2> const& f_kR = memory::array<MEM, ComplexType, 2>{})
{
  // Stage device callers through host: pull arrays once, run the SCF on
  // host scratch, push final outputs back. The legacy free-function driver
  // is mostly used by the array-API tests; the class-API SCF
  // (real_axis_scf_loop) is the canonical path.
  if constexpr (MEM != HOST_MEMORY) {
    auto H_h = nda::to_host(H_MF_skij);
    auto X_h = nda::to_host(X_skPmu);
    auto V_h = nda::to_host(V_qPQ);
    nda::array<ComplexType, 5> A_h(A_wskij.shape());
    nda::array<ComplexType, 4> Sx_h(Sigma_x_skij.shape());
    nda::array<ComplexType, 5> ImSc_h(ImSigma_c_skwij.shape());
    nda::array<ComplexType, 5> ReSc_h(ReSigma_c_skwij.shape());
    A_h    = nda::to_host(A_wskij);
    Sx_h   = nda::to_host(Sigma_x_skij);
    ImSc_h = nda::to_host(ImSigma_c_skwij);
    ReSc_h = nda::to_host(ReSigma_c_skwij);
    nda::array<ComplexType, 2> f_Rk_h, f_qR_h, f_Rq_h, f_kR_h;
    if (f_Rk.size() > 0) f_Rk_h = nda::to_host(f_Rk);
    if (f_qR.size() > 0) f_qR_h = nda::to_host(f_qR);
    if (f_Rq.size() > 0) f_Rq_h = nda::to_host(f_Rq);
    if (f_kR.size() > 0) f_kR_h = nda::to_host(f_kR);
    auto res = run_scgw_serial<HOST_MEMORY>(comm, grid_in, H_h, X_h, V_h,
                                            kpq_to_kp, kmq_to_kp, q_weights,
                                            k_weights, N_elec, cfg,
                                            A_h, Sx_h, ImSc_h, ReSc_h,
                                            f_Rk_h, f_qR_h, f_Rq_h, f_kR_h);
    A_wskij        = A_h;
    Sigma_x_skij   = Sx_h;
    ImSigma_c_skwij = ImSc_h;
    ReSigma_c_skwij = ReSc_h;
    return res;
  }
  const long ns   = H_MF_skij.shape()[0];
  const long Nk   = H_MF_skij.shape()[1];
  const long nbnd = H_MF_skij.shape()[2];
  const long N_w  = grid_in.N_w();

  utils::check(H_MF_skij.shape()[3] == nbnd,
               "run_scgw_serial: H_MF not square in (i,j)");
  utils::check(A_wskij.shape()[0] == N_w and A_wskij.shape()[1] == ns and
               A_wskij.shape()[2] == Nk and A_wskij.shape()[3] == nbnd and
               A_wskij.shape()[4] == nbnd,
               "run_scgw_serial: A_wskij shape mismatch");
  utils::check(k_weights.shape()[0] == Nk,
               "run_scgw_serial: k_weights size mismatch");

  // Build the input A in driver layout (ns, Nk, N_w, nbnd, nbnd).
  // Initial A: if A_wskij is identically zero, build a Lorentzian-broadened
  // mean-field A from H_MF.
  bool A_is_zero = true;
  {
    auto const* d = A_wskij.data();
    for (long i = 0; i < A_wskij.size(); ++i)
      if (std::abs(d[i]) > 0.0) { A_is_zero = false; break; }
  }
  if (A_is_zero) {
    // Diagonalize H_MF(s, k) per (s, k); place a Lorentzian per eigenvalue.
    // For simplicity assume H_MF is already diagonal in the input basis;
    // place A diagonally using H_MF diagonal entries as eigenvalues.
    const double eta_init = std::max(cfg.eta, 1e-2);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k)
        for (long m = 0; m < nbnd; ++m) {
          const double e = H_MF_skij(s, k, m, m).real();
          for (long iw = 0; iw < N_w; ++iw) {
            const double wl = grid_in.w()(iw) + grid_in.mu_chem();
            const double v = (1.0 / M_PI) * eta_init / ((wl - e)*(wl - e) + eta_init*eta_init);
            A_wskij(iw, s, k, m, m) = ComplexType(v, 0.0);
          }
        }
  }

  // Working grid (mu may change each iteration).
  double mu_cur = grid_in.mu_chem();

  // Iteration buffers.
  ImSigma_c_skwij = ComplexType(0.0, 0.0);
  ReSigma_c_skwij = ComplexType(0.0, 0.0);
  Sigma_x_skij    = ComplexType(0.0, 0.0);

  scgw_result res;

  // Legacy run_scgw_serial supports linear mixing only -- the new
  // diis_mixer_t requires shared-memory storage (sArray) which the
  // free-function array-API does not provide. The class-API SCF
  // (real_axis_scf_loop) supports DIIS.
  utils::check(cfg.mix_kind == scgw_mix_kind::linear,
               "run_scgw_serial: only linear mixing is supported in the "
               "legacy free-function API. Use real_axis_scf_loop for DIIS.");

  for (long it = 0; it < cfg.max_iter; ++it) {

    // Build the grid for this iteration with the current mu.
    auto grid = real_freq_grid_t(grid_in.beta(), mu_cur,
                                 nda::array<double,1>(grid_in.w()),
                                 nda::array<double,1>(grid_in.Omega()),
                                 grid_in.N_t(), grid_in.T_window());

    // Repack A into driver layout (ns, Nk, N_w, nbnd, nbnd) and apply
    // matrix-hermitian symmetrization. state.A_wskij stores -(i/pi) G^R
    // componentwise; the matrix-valued physical spectral function is
    //   A_phys_{ij} = 0.5 * (A_wskij_{ij} + conj(A_wskij_{ji}))
    // (See scr_coulomb_t::update_w for the long-form comment and
    //  test_real_axis_hermiticity for validation.) The Pi cross-correlation
    // kernel relies on aux-hermiticity to apply the second-leg swap as
    // conj on the local block, so the input A must be matrix-hermitian.
    nda::array<ComplexType, 5> A_drv(ns, Nk, N_w, nbnd, nbnd);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k)
        for (long iw = 0; iw < N_w; ++iw)
          for (long mu = 0; mu < nbnd; ++mu)
            for (long nu = 0; nu < nbnd; ++nu)
              A_drv(s, k, iw, mu, nu) =
                  ComplexType(0.5, 0.0) *
                  (A_wskij(iw, s, k, mu, nu)
                   + std::conj(A_wskij(iw, s, k, nu, mu)));

    // ---- Sigma^c ----
    nda::array<ComplexType, 5> ImSc_drv(ns, Nk, N_w, nbnd, nbnd);
    nda::array<ComplexType, 5> ReSc_drv(ns, Nk, N_w, nbnd, nbnd);
    evaluate_serial(comm, grid, A_drv, X_skPmu, V_qPQ, kpq_to_kp, kmq_to_kp, q_weights,
                    ImSc_drv, ReSc_drv, cfg.eps_nufft, cfg.iq_gamma, /*verbose*/ false,
                    f_Rk, f_qR, f_Rq, f_kR);

    // Repack Sigma^c into (ns, Nk, N_w, nbnd, nbnd) layout (already that way).
    // Apply causality projection.
    nda::array<ComplexType, 5> ImSc_layout(ns, Nk, N_w, nbnd, nbnd);
    nda::array<ComplexType, 5> ReSc_layout(ns, Nk, N_w, nbnd, nbnd);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k)
        for (long iw = 0; iw < N_w; ++iw)
          for (long mu = 0; mu < nbnd; ++mu)
            for (long nu = 0; nu < nbnd; ++nu) {
              ImSc_layout(s, k, iw, mu, nu) = ImSc_drv(s, k, iw, mu, nu);
              ReSc_layout(s, k, iw, mu, nu) = ReSc_drv(s, k, iw, mu, nu);
            }
    project_causality_ImSigma(ImSc_layout);

    // ---- Sigma^x ----
    nda::array<ComplexType, 4> Sx_new(ns, Nk, nbnd, nbnd);
    evaluate_Sigma_x_serial(comm, grid, A_drv, X_skPmu, V_qPQ, kmq_to_kp, Sx_new,
                            cfg.iq_gamma);

    // ---- Store latest Sigma (no Sigma-side mixing; we mix on A below). ----
    ImSigma_c_skwij = ImSc_layout;
    ReSigma_c_skwij = ReSc_layout;
    Sigma_x_skij    = Sx_new;

    // ---- Dyson update for A ----
    nda::array<ComplexType, 5> A_full_wskij(N_w, ns, Nk, nbnd, nbnd);
    dyson_update_A(grid, H_MF_skij, Sigma_x_skij,
                   ReSigma_c_skwij, ImSigma_c_skwij, cfg.eta, A_full_wskij);

    // ---- Linear mix: A_next = (1 - alpha) * A_old + alpha * A_full.
    // MEM-agnostic via nda::map.
    const double diff = frobenius_diff(A_wskij, A_full_wskij);
    const double a = cfg.alpha_mix;
    A_wskij = nda::map([a](ComplexType old_v, ComplexType new_v) {
      return (1.0 - a) * old_v + a * new_v;
    })(A_wskij, A_full_wskij);

    // ---- mu update ----
    if (cfg.update_mu) {
      const double mu_new = find_mu_chem(grid, A_wskij, k_weights, N_elec,
                                         /*tol*/ 1e-7, /*max_iter*/ 100);
      mu_cur = mu_new;
    }

    if (cfg.verbose and comm.root()) {
      std::cout << "[scgw] iter=" << (it + 1)
                << "  ||dA||=" << diff
                << "  mu=" << mu_cur << std::endl;
    }

    res.iter_used  = it + 1;
    res.final_diff = diff;
    res.final_mu   = mu_cur;
    if (diff < cfg.tol) { res.converged = true; break; }
  }

  return res;
}

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_SCF_HPP
