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
#include "numerics/imag_axes_ft/dlr_pole_fit.hpp"
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

    /**
     * Generic fermionic pair bubble, accumulated on the FULL bosonic tau grid
     * (increment C2):
     *
     *   Pi(tau_j, a, b) += pref * tr[ A_a(tau_j) . B_b(beta - tau_j) ]
     *
     * A and B arrive at the backend's fermionic iw nodes as flat batches
     * (nw, da*nb*nb) / (nw, db*nb*nb); they are pole-fitted by the REGULARIZED w-side
     * fit (imag_axes_ft::dlr_pole_fit_w -- rule 7: never a square interpolatory solve;
     * fit_error/residue_ratio maxima are reported to the caller, who gates) and
     * evaluated at the bosonic tau nodes through the analytic kernel K_F. A bosonic
     * tau_to_w of the accumulated Pi then yields the Matsubara convolution
     * -pref * (1/beta) sum_iw tr[A_a(iw) B_b(iw + inu)] per (a, b); the overall
     * sign/normalization convention is PINNED by the dense-Matsubara oracle in
     * test_scr_cvv (gate C2-b), not by rederivation.
     */
    inline void bubble_accumulate(imag_axes_ft::dlr_pole_fit_w const &pfw,
                                  nda::MemoryArrayOfRank<2> auto const &A_wd,
                                  nda::MemoryArrayOfRank<2> auto const &B_wd,
                                  nda::MemoryArrayOfRank<2> auto const &Kt,      // (ntb, np): K_F(tau_j, eps_p)
                                  nda::MemoryArrayOfRank<2> auto const &Kt_mir,  // (ntb, np): K_F(beta - tau_j, eps_p)
                                  long nb, double pref,
                                  nda::MemoryArrayOfRank<3> auto &&Pi_tab,       // (ntb, da, db), +=
                                  double &fit_err_max, double &res_ratio_max) {
      const long ntb = Kt.shape(0), nb2 = nb * nb;
      const long da = A_wd.shape(1) / nb2, db = B_wd.shape(1) / nb2;
      utils::check(A_wd.shape(1) == da * nb2 and B_wd.shape(1) == db * nb2,
                   "cvv_detail::bubble_accumulate: batch width not a multiple of nb^2.");
      utils::check(Pi_tab.shape(0) == ntb and Pi_tab.shape(1) == da and
                   Pi_tab.shape(2) == db,
                   "cvv_detail::bubble_accumulate: Pi shape mismatch.");

      auto cA = pfw.coeffs(A_wd);
      auto cB = pfw.coeffs(B_wd);
      fit_err_max = std::max({fit_err_max, pfw.fit_error(A_wd, cA), pfw.fit_error(B_wd, cB)});
      res_ratio_max = std::max({res_ratio_max, pfw.residue_ratio(A_wd, cA),
                                pfw.residue_ratio(B_wd, cB)});

      nda::array<ComplexType, 2> At(ntb, da * nb2), Bt(ntb, db * nb2);
      nda::blas::gemm(Kt, cA, At);
      nda::blas::gemm(Kt_mir, cB, Bt);

      for (long j = 0; j < ntb; ++j)
        for (long a = 0; a < da; ++a) {
          const long oa = a * nb2;
          for (long b = 0; b < db; ++b) {
            const long ob = b * nb2;
            ComplexType tr(0.0);   // tr[A B] = sum_im A(i,m) B(m,i), flat row indexing
            for (long i = 0; i < nb; ++i)
              for (long m = 0; m < nb; ++m)
                tr += At(j, oa + i * nb + m) * Bt(j, ob + m * nb + i);
            Pi_tab(j, a, b) += pref * tr;
          }
        }
    }

    /**
     * The SUBTRACTED head coefficient (the object P00 actually consumes):
     *
     *   Phead_ab(inu) = [ Pi^jj_ab(inu) - Pi^jj_ab(0) ] / (inu)^2,
     *
     * finite at inu = 0 by Richardson extrapolation from the two smallest nonzero
     * bosonic nodes. WHY THE SUBTRACTION [measured, gate C2-c 2026-08-11]: the raw
     * paramagnetic bubble at inu = 0 does NOT give the density head -- on the gapped
     * toy it overshoots by exactly gap^2 (measured 53x at gap ~ 8). Continuity ties the
     * density response to the TOTAL (paramagnetic + diamagnetic) current response,
     * (inu)^2 P00(q, inu) = q^2 [k_para(inu) + k_dia]; for an INSULATOR the Drude
     * weight vanishes, k_dia = -k_para(0), so
     *
     *   P00(q -> 0, inu) = q_a q_b [Pi_ab(inu) - Pi_ab(0)] / (inu)^2
     *
     * at EVERY inu (k_dia is nu-independent), with the nu -> 0 limit at the static
     * point -- no explicit diamagnetic / d2h evaluation needed. The PDF's boxed
     * eq:tier1 inu = 0 line elides this subtraction; gate C2-c pins the corrected form
     * against exact Adler-Wiser on the toy (2-level check: [2D/(D^2+nu^2) - 2/D]/(inu)^2
     * -> 2/D^3 = the chi_rho-rho q^2 coefficient).
     */
    inline nda::array<ComplexType, 3> head_subtract(nda::MemoryArrayOfRank<3> auto const &Pi_wab,
                                                    imag_axes_ft::IAFT const &ft) {
      const long nwb = Pi_wab.shape(0), da = Pi_wab.shape(1), db = Pi_wab.shape(2);
      auto wn_b = ft.wn_mesh_b();
      long i0 = -1;
      for (long m = 0; m < nwb; ++m)
        if (wn_b(m) == 0) i0 = m;
      utils::check(i0 >= 0 and i0 + 2 < nwb,
                   "cvv_detail::head_subtract: no inu = 0 node / bosonic grid too small.");
      nda::array<ComplexType, 3> Ph(nwb, da, db);
      for (long m = 0; m < nwb; ++m) {
        if (m == i0) continue;
        const ComplexType inu2 = ft.omega(wn_b(m)) * ft.omega(wn_b(m));
        for (long a = 0; a < da; ++a)
          for (long b = 0; b < db; ++b)
            Ph(m, a, b) = (Pi_wab(m, a, b) - Pi_wab(i0, a, b)) / inu2;
      }
      // inu = 0: Richardson in nu^2 from the two smallest positive nodes
      const double x1 = double(wn_b(i0 + 1)) * double(wn_b(i0 + 1));
      const double x2 = double(wn_b(i0 + 2)) * double(wn_b(i0 + 2));
      for (long a = 0; a < da; ++a)
        for (long b = 0; b < db; ++b)
          Ph(i0, a, b) = (x2 * Ph(i0 + 1, a, b) - x1 * Ph(i0 + 2, a, b)) / (x2 - x1);
      return Ph;
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

    /** Result of the C2 head tensor evaluation. */
    struct head_result_t {
      // Pi_ab on the FULL bosonic tau grid (bubble integrand incl. prefactor) and its
      // bosonic transform; inu = 0 sits at index nw_b/2 of the full Matsubara grid.
      nda::array<ComplexType, 3> Pi_tab;   // (nt_b, 3, 3) paramagnetic bubble
      nda::array<ComplexType, 3> Pi_wab;   // (nw_b, 3, 3) paramagnetic bubble
      // the SUBTRACTED head coefficient (cvv_detail::head_subtract) -- the object
      // P00(q->0, inu) = q_a q_b Phead_ab(inu) consumes (C3 readout / C4 in-loop)
      nda::array<ComplexType, 3> Phead_wab;  // (nw_b, 3, 3)
      double fit_error_max = 0.0;          // worst w-side pole-fit reconstruction error
      double res_ratio_max = 0.0;          // worst residue amplification (watched)
    };

    /**
     * Increment C2: the covariant-velocity head tensor
     *   Pi_ab(inu) = -(2/(beta Nk V)) sum_{s,k,iw} Tr[ v~_a G v~_b G ]     (PDF eq. tier1;
     * spin factor 2 for ns = 1, plain spin sum for ns = 2). Per (s, k): G(tau) -> iw,
     * M_a = v~_a G at the fermionic nodes, regularized w-side pole fit (rule 7), bubble
     * on the bosonic tau grid (cvv_detail::bubble_accumulate), k round-robin over the
     * global communicator + one tiny all_reduce, one bosonic tau_to_w at the end.
     * G_tskij is the IBZ-resident Green's function on the fermionic tau mesh; the
     * full-BZ gather uses the TRANSPOSE on trev k (the build_Gbar_fullbz G-convention;
     * Sigma-class objects use conj -- see build()).
     */
    head_result_t eval_head_tensor(mf::MF &mf,
                                   nda::MemoryArrayOfRank<5> auto const &G_tskij) {
      decltype(nda::range::all) all;
      utils::check(_built, "cvv_head_t::eval_head_tensor: build() has not run.");
      auto ctx = mf.mpi();
      const long nk = mf.nkpts(), nk_ibz = mf.nkpts_ibz();
      const long nt_f = G_tskij.shape(0), nb = _nb, nb2 = nb * nb;
      utils::check(G_tskij.shape(1) == _ns and G_tskij.shape(2) == nk_ibz and
                   G_tskij.shape(3) == nb,
                   "cvv_head_t::eval_head_tensor: G shape mismatch.");
      utils::check(nt_f == _ft->nt_f(), "cvv_head_t::eval_head_tensor: nt mismatch.");
      ensure_bubble_tables();

      const long ntb = _Kt.shape(0);
      head_result_t res;
      res.Pi_tab = nda::array<ComplexType, 3>(ntb, 3, 3);
      res.Pi_tab() = ComplexType(0.0);
      // eq:tier1 normalization: -(spin_fac / (Nk V)) per k term; the 1/beta lives in the
      // bosonic transform; the overall SIGN is pinned by the C2-b oracle.
      const double pref = -((_ns == 1) ? 2.0 : 1.0) / (double(nk) * mf.volume());

      auto kp_to_ibz = mf.kp_to_ibz();
      auto kp_trev = mf.kp_trev();
      const bool sym = (nk != nk_ibz);
      nda::array<ComplexType, 2> Gk_t(nt_f, nb2), Gk_w(_nw, nb2);
      nda::array<ComplexType, 2> M(_nw, 3 * nb2);
      const long rank = ctx->comm.rank(), size = ctx->comm.size();
      for (long isk = rank; isk < _ns * nk; isk += size) {
        const long is = isk / nk, ik = isk % nk;
        const long ksrc = sym ? long(kp_to_ibz(ik)) : ik;
        const bool trev = sym and bool(kp_trev(ik));
        for (long it = 0; it < nt_f; ++it) {
          auto Gt = nda::reshape(Gk_t(it, all), std::array<long, 2>{nb, nb});
          auto Gsrc = G_tskij(it, is, ksrc, all, all);
          if (trev) { for (long i = 0; i < nb; ++i)
                        for (long j = 0; j < nb; ++j) Gt(i, j) = Gsrc(j, i); }
          else      Gt = Gsrc;
        }
        _ft->tau_to_w(Gk_t, Gk_w, imag_axes_ft::fermion);
        auto v = velocity(is, ik);                       // (3, nw, nb, nb)
        nda::array<ComplexType, 2> Gn(nb, nb), Mn(nb, nb);
        for (long n = 0; n < _nw; ++n) {
          for (long i = 0; i < nb; ++i)
            for (long m = 0; m < nb; ++m) Gn(i, m) = Gk_w(n, i * nb + m);
          for (long a = 0; a < 3; ++a) {
            nda::blas::gemm(v(a, n, all, all), Gn, Mn);
            const long oa = a * nb2;
            for (long i = 0; i < nb; ++i)
              for (long m = 0; m < nb; ++m) M(n, oa + i * nb + m) = Mn(i, m);
          }
        }
        cvv_detail::bubble_accumulate(_pfw.value(), M, M, _Kt, _Kt_mir, nb, pref,
                                      res.Pi_tab, res.fit_error_max, res.res_ratio_max);
      }
      ctx->comm.all_reduce_in_place_n(res.Pi_tab.data(), res.Pi_tab.size(), std::plus<>{});
      res.fit_error_max = ctx->comm.max(res.fit_error_max);
      res.res_ratio_max = ctx->comm.max(res.res_ratio_max);
      imag_axes_ft::dlr_pole_fit_gate(res.fit_error_max, "cvv_head_t::eval_head_tensor",
                                      1e-3, 1e-2, res.res_ratio_max);

      res.Pi_wab = nda::array<ComplexType, 3>(_ft->nw_b(), 3, 3);
      auto Pt2 = nda::reshape(res.Pi_tab, std::array<long, 2>{ntb, 9});
      auto Pw2 = nda::reshape(res.Pi_wab, std::array<long, 2>{long(_ft->nw_b()), 9});
      _ft->tau_to_w(Pt2, Pw2, imag_axes_ft::boson);
      res.Phead_wab = cvv_detail::head_subtract(res.Pi_wab, *_ft);
      const long i0 = _ft->nw_b() / 2;
      app_log(2, "  [CVV] head tensor: diag Phead(inu=0) = ({:.6e}, {:.6e}, {:.6e}) "
                 "[raw para bubble Pi(0) diag = ({:.3e}, {:.3e}, {:.3e})]; "
                 "pole-fit err_max = {:.3e}, residue ratio max = {:.3g}",
              res.Phead_wab(i0, 0, 0).real(), res.Phead_wab(i0, 1, 1).real(),
              res.Phead_wab(i0, 2, 2).real(), res.Pi_wab(i0, 0, 0).real(),
              res.Pi_wab(i0, 1, 1).real(), res.Pi_wab(i0, 2, 2).real(),
              res.fit_error_max, res.res_ratio_max);
      return res;
    }

    /** lazy build of the w-side pole fit + the bosonic-tau kernel tables (public so the
     *  unit tests can drive cvv_detail::bubble_accumulate with the same tables). */
    void ensure_bubble_tables() {
      if (_pfw.has_value()) return;
      _pfw = imag_axes_ft::dlr_pole_fit_w(*_ft);
      auto const &pf = _pfw.value();
      const long ntb = _ft->nt_b();
      const double beta = _ft->beta();
      auto taub = _ft->tau_mesh_b();
      _Kt = nda::array<ComplexType, 2>(ntb, pf.np);
      _Kt_mir = nda::array<ComplexType, 2>(ntb, pf.np);
      for (long j = 0; j < ntb; ++j) {
        const double s = (taub(j) + 1.0) * 0.5 * beta;   // tau_mesh in [-1, 1] convention
        for (long p = 0; p < pf.np; ++p) {
          _Kt(j, p) = ComplexType(imag_axes_ft::dlr_kF(beta, s, pf.epsl(p)));
          _Kt_mir(j, p) = ComplexType(imag_axes_ft::dlr_kF(beta, beta - s, pf.epsl(p)));
        }
      }
    }
    imag_axes_ft::dlr_pole_fit_w const &pole_fit_w() { ensure_bubble_tables(); return _pfw.value(); }
    nda::array<ComplexType, 2> const &Kt() const { return _Kt; }
    nda::array<ComplexType, 2> const &Kt_mir() const { return _Kt_mir; }

  private:
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
    // C2 bubble machinery (lazy): the w-side regularized pole fit + the K_F tables on
    // the bosonic tau grid (tau_j and beta - tau_j)
    std::optional<imag_axes_ft::dlr_pole_fit_w> _pfw;
    nda::array<ComplexType, 2> _Kt, _Kt_mir;
  };

} // solvers
} // methods

#endif // COQUI_CVV_HEAD_HPP
