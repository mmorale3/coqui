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

#ifndef COQUI_CVV_HEAD_HPP
#define COQUI_CVV_HEAD_HPP

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "utilities/check.hpp"
#include "utilities/kpoint_utils.hpp"
#include "utilities/interpolation_utils.hpp"
#include "IO/app_loggers.h"
#include "mean_field/MF.hpp"

namespace methods {
namespace solvers {

  /**
   * scGW-tilde CVV head, increment C1: the R-space engine + covariant velocity
   * (notes/scgwt_implementation_plan.md; theory notes/scgw_screening_fix_proposal.pdf
   * section 4.1).
   *
   *   v~_a(k, iw) = d_k_a [ H0 + F + Sigma(k, iw) ]
   *
   * evaluated analytically through the Wigner-Seitz R-space store: the band-basis
   * h(k) = H0 + F (+ Sigma(tau)) is unfolded IBZ -> full BZ (copy / conj-on-trev, the
   * SAME convention as the total-Sigma unfolding in mean_field/symmetry/unfold_bz.cpp --
   * for hermitian slices conj == the transpose branch of
   * vertex_wannier_detail::build_Gbar_fullbz), Fourier-transformed k -> R by one gemm
   * against utils::k_to_R_coefficients on the WS grid, R-shell-truncated at
   * cvv_rspace_tol (dropped shells are ZEROED in the store; shells are |R|-symmetric so
   * hermiticity v~(k)^dag = v~(k) is preserved exactly), and tau -> iw transformed ONCE
   * on the (small, node-shared) R-object. Velocities are then assembled ON THE FLY per
   * (s, k):
   *
   *   v~_a(k, iw) = sum_R i R_a e^{i k.R} / w_R  [ h_stat(R) + Sigma(R, iw) ]
   *
   * (one gemm; never stored globally -- ground rule 4). The k.R phase convention is the
   * exact inverse pair of utils::k_to_R_coefficients / utils::R_to_k_coefficients, so
   * the interpolant reproduces h(k) on the mesh exactly and the velocity is the exact
   * analytic derivative of that interpolant (gates C1-a/C1-c).
   */
  namespace cvv_detail {

    // Cartesian R vectors from lattice-unit indices: R = a*lattv(0,:) + b*lattv(1,:) + c*lattv(2,:)
    inline nda::array<double, 2> rcart_from_idx(nda::MemoryArrayOfRank<2> auto const &Rpts_idx,
                                                nda::stack_array<double, 3, 3> const &lattv) {
      const long nR = Rpts_idx.shape(0);
      nda::array<double, 2> Rc(nR, 3);
      for (long i = 0; i < nR; ++i)
        for (int a = 0; a < 3; ++a)
          Rc(i, a) = Rpts_idx(i, 0) * lattv(0, a) + Rpts_idx(i, 1) * lattv(1, a) +
                     Rpts_idx(i, 2) * lattv(2, a);
      return Rc;
    }

    /**
     * The per-k contraction rows over the WS R store. Rows 0..2 are the DERIVATIVE rows
     *   P(a, R) = i R_a e^{i k.R} / w_R,
     * row 3 (present iff with_value_row) is the VALUE (interpolation) row
     *   P(3, R) = e^{i k.R} / w_R
     * -- the exact adjoint pair of utils::k_to_R_coefficients' e^{-i k.R}/nk kernel
     * (utils::R_to_k_coefficients convention, WS degeneracy weights w_R). kvec is
     * CARTESIAN (the same convention k_to_R_coefficients applies to mf.kpts()).
     */
    inline nda::array<ComplexType, 2> phase_rows(nda::MemoryArrayOfRank<2> auto const &Rcart,
                                                 nda::MemoryArrayOfRank<1> auto const &wR,
                                                 nda::MemoryArrayOfRank<1> auto const &kvec,
                                                 bool with_value_row = false) {
      const long nR = Rcart.shape(0);
      nda::array<ComplexType, 2> P(with_value_row ? 4 : 3, nR);
      for (long i = 0; i < nR; ++i) {
        const double kR = kvec(0) * Rcart(i, 0) + kvec(1) * Rcart(i, 1) + kvec(2) * Rcart(i, 2);
        const ComplexType ph = std::exp(ComplexType(0.0, kR)) / double(wR(i));
        for (int a = 0; a < 3; ++a) P(a, i) = ComplexType(0.0, Rcart(i, a)) * ph;
        if (with_value_row) P(3, i) = ph;
      }
      return P;
    }

  } // cvv_detail

  class cvv_head_t {
  public:
    template<nda::MemoryArray Array_base_t>
    using sArray_t = math::shm::shared_array<Array_base_t>;

    cvv_head_t(const imag_axes_ft::IAFT *ft, double rspace_tol = 1e-6);

    cvv_head_t(cvv_head_t &&) = default;
    cvv_head_t& operator=(cvv_head_t &&) = default;
    ~cvv_head_t() {}

    // R-shell truncation tolerance of the R-space store ([gw] cvv_rspace_tol)
    double rspace_tol() const { return _rspace_tol; }
    bool built() const { return _built; }
    long nR() const { return _nR; }
    long nR_kept() const { return _nR_kept; }
    long nw() const { return _nw; }

    /**
     * Build the node-shared R-space store from the IBZ-resident state.
     * @param H0_skij    - (ns, nkpts_ibz, nbnd, nbnd) bare one-body Hamiltonian
     * @param F_skij     - (ns, nkpts_ibz, nbnd, nbnd) static self-energy (Fock)
     * @param Sigma_tskij- (nts, ns, nkpts_ibz, nbnd, nbnd) dynamic self-energy on the
     *                     fermionic tau mesh; an EMPTY array (size 0) means Sigma = 0
     *                     (KS/HF control -- the velocity is then iw-independent).
     * All ranks call this (collective over mf.mpi()); node roots compute the identical
     * small store, node_sync publishes it.
     */
    void build(mf::MF &mf,
               nda::MemoryArrayOfRank<4> auto const &H0_skij,
               nda::MemoryArrayOfRank<4> auto const &F_skij,
               nda::MemoryArrayOfRank<5> auto const &Sigma_tskij) {
      decltype(nda::range::all) all;
      auto ctx = mf.mpi();
      const long ns = H0_skij.shape(0), nk_ibz = H0_skij.shape(1), nb = H0_skij.shape(2);
      const long nk = mf.nkpts();
      const bool has_sigma = Sigma_tskij.size() > 0;
      const long nt = has_sigma ? Sigma_tskij.shape(0) : 0;
      utils::check(F_skij.shape() == H0_skij.shape(),
                   "cvv_head_t::build: F/H0 shape mismatch.");
      if (has_sigma)
        utils::check(Sigma_tskij.shape(1) == ns and Sigma_tskij.shape(2) == nk_ibz and
                     Sigma_tskij.shape(3) == nb,
                     "cvv_head_t::build: Sigma shape mismatch.");
      utils::check(nk_ibz == mf.nkpts_ibz(), "cvv_head_t::build: nkpts_ibz mismatch.");
      _ns = ns; _nb = nb; _nw = _ft->nw_f(); _has_sigma = has_sigma;

      // Wigner-Seitz R grid of the k-mesh supercell (degeneracy-weighted boundary)
      auto [rw, rp] = utils::WS_rgrid(mf.lattv(), mf.kp_grid());
      _nR = rp.shape(0);
      _Rcart = cvv_detail::rcart_from_idx(rp, mf.lattv());
      _wR.resize(_nR);
      for (long i = 0; i < _nR; ++i) _wR(i) = double(rw(i));
      _kpts.resize(nk, 3);
      _kpts() = mf.kpts();

      // node-shared stores (ground rule 4): static h(R) and Sigma(R, iw)
      _shstat = sArray_t<nda::array_view<ComplexType, 4>>(
          math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(
              *ctx, {ns, _nR, nb, nb}));
      _shstat.value().set_zero();
      if (has_sigma) {
        _ssig = sArray_t<nda::array_view<ComplexType, 5>>(
            math::shm::make_shared_array<nda::array_view<ComplexType, 5>>(
                *ctx, {ns, _nR, _nw, nb, nb}));
        _ssig.value().set_zero();
      } else {
        _ssig.reset();
      }

      if (ctx->node_comm.root()) {
        // f_Rk = e^{-i k.R}/nk on the full BZ (the plain non-collective kernel; the
        // store is small so each node root computes the identical object)
        nda::array<ComplexType, 2> f_Rk(_nR, nk);
        utils::k_to_R_coefficients(rp, mf.kpts(), mf.lattv(), f_Rk);

        auto kp_to_ibz = mf.kp_to_ibz();
        auto kp_trev = mf.kp_trev();
        const bool sym = (nk != nk_ibz);

        // full-BZ slice gather (copy / conj-on-trev -- unfold_bz.cpp convention) into a
        // (nk, nb*nb) buffer, then ONE gemm k -> R per slice (idiom (a): flattened
        // trailing pair). No global unfolded object is ever materialized.
        nda::array<ComplexType, 2> buf(nk, nb * nb);
        auto gather_full = [&](auto const &A_kij_ibz) {   // (nk_ibz, nb, nb) at fixed pre-axes
          for (long ik = 0; ik < nk; ++ik) {
            const long ksrc = sym ? long(kp_to_ibz(ik)) : ik;
            const bool trev = sym and bool(kp_trev(ik));
            auto row = nda::reshape(buf(ik, all), std::array<long, 2>{nb, nb});
            if (trev) row = nda::conj(A_kij_ibz(ksrc, all, all));
            else      row = A_kij_ibz(ksrc, all, all);
          }
        };

        auto hstat = _shstat.value().local();
        for (long is = 0; is < ns; ++is) {
          gather_full(H0_skij(is, all, all, all));
          nda::array<ComplexType, 2> buf2(buf);
          gather_full(F_skij(is, all, all, all));
          buf += buf2;
          auto h2 = nda::reshape(hstat(is, all, all, all), std::array<long, 2>{_nR, nb * nb});
          nda::blas::gemm(f_Rk, buf, h2);
        }

        // Sigma: k -> R per (t, s) into a tau staging, then tau -> iw ONCE per (s, R)
        if (has_sigma) {
          nda::array<ComplexType, 4> sig_t(ns, _nR, nt, nb * nb);
          nda::array<ComplexType, 2> tmpR(_nR, nb * nb);
          for (long is = 0; is < ns; ++is)
            for (long it = 0; it < nt; ++it) {
              gather_full(Sigma_tskij(it, is, all, all, all));
              nda::blas::gemm(f_Rk, buf, tmpR);
              for (long iR = 0; iR < _nR; ++iR) sig_t(is, iR, it, all) = tmpR(iR, all);
            }
          auto sig_w = _ssig.value().local();
          for (long is = 0; is < ns; ++is)
            for (long iR = 0; iR < _nR; ++iR) {
              auto S_wf = nda::reshape(sig_w(is, iR, all, all, all),
                                       std::array<long, 2>{_nw, nb * nb});
              _ft->tau_to_w(sig_t(is, iR, all, all), S_wf, imag_axes_ft::fermion);
            }
        }

        // R-SHELL truncation at cvv_rspace_tol: group by |R|, drop the largest-|R|
        // suffix whose total norm fraction stays below tol, ZERO the dropped rows in
        // the store (shells are inversion-symmetric sets, so hermiticity of the
        // velocity is preserved exactly). Norms measured on the FULL stored object.
        truncate_shells();
      }
      ctx->comm.barrier();
      _shstat.value().node_sync();
      if (has_sigma) _ssig.value().node_sync();
      ctx->node_comm.broadcast_n(&_nR_kept, 1, 0);
      _built = true;
    }

    /**
     * The covariant velocity at one (spin, k): v~_a(k, iw_n), shape (3, nw_f, nb, nb).
     * One gemm over the kept R shells; never stored globally.
     */
    nda::array<ComplexType, 4> velocity(long is, long ik) const;

    // ---- increment C2: head tensor Pi_ab(inu) (aborts until it lands) ----
    void eval_head_tensor();

  private:
    [[noreturn]] void not_implemented(std::string_view where) const;
    void truncate_shells();

    const imag_axes_ft::IAFT* _ft = nullptr;
    double _rspace_tol = 1e-6;

    bool _built = false;
    bool _has_sigma = false;
    long _ns = 0, _nb = 0, _nw = 0, _nR = 0, _nR_kept = 0;
    nda::array<double, 2> _Rcart;   // (nR, 3) cartesian
    nda::array<double, 1> _wR;      // (nR) WS degeneracy weights
    nda::array<double, 2> _kpts;    // (nk, 3) cartesian (mf.kpts() copy)
    // node-shared R stores; dropped shells are zeroed rows
    std::optional<sArray_t<nda::array_view<ComplexType, 4>>> _shstat;  // (ns, nR, nb, nb)
    std::optional<sArray_t<nda::array_view<ComplexType, 5>>> _ssig;   // (ns, nR, nw, nb, nb)
  };

} // solvers
} // methods

#endif // COQUI_CVV_HEAD_HPP
