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

#ifndef COQUI_IAFT_DCONV_HPP
#define COQUI_IAFT_DCONV_HPP

/**
 * Double bosonic Matsubara convolution on the DLR imaginary-axis backend.
 *
 *   S(iw) = (1/beta^2) sum_{vx,vy} [Wx(ivx)+cx] [Wy(ivy)+cy]
 *                                  B(iw+ivy) C(iw-ivx+ivy) D(iw-ivx)
 *
 * with w fermionic, vx/vy bosonic, B/C/D fermionic propagator-like factors,
 * and Wx/Wy bosonic, PH-symmetric (W(-iv) = W(iv)) DYNAMIC (decaying) parts;
 * cx/cy are the exact instantaneous (frequency-constant, batch-varying) parts.
 *
 * Algorithm ("DLR channel algebra", see notes/double_convolution_design.md):
 * the DLR expansion of each input is an exact-to-eps pole representation
 *   fermionic F(z) = sum_l f_l / (z - eps_l),        f_l = DLR time coeffs
 *   bosonic   W(z) = sum_l [w_l tanh(hw_l/2)]/(z-eps_l)
 * (eps_l = dlr_rf(l)/beta). The inner (vy) Matsubara sum is closed by residues
 * into a divided-difference-stabilized 3-channel form; the outer (vx) sum then
 * reduces to O(r^2) standard single convolutions (families I-IV) computed with
 *   (1/beta) sum_v V(iv) H(iw-iv) = int_0^beta dt e^{iwt} V(t) H(t),
 * i.e. Twt_ff * [(Ttw_bb V_w) . (Ttw_ff H_w)], plus the closed-form family V
 * (the true double-pole term of the shared-node diagonal at vx = 0). All
 * analytic continuations sit at complex arguments with |Im| >= pi/beta or at
 * exactly-deflated real arguments, so there are no small-denominator hazards.
 * The pole data lives on cppdlr's NONSYM auxiliary grid (well-conditioned
 * interpolation, moderate coefficients); the backend SYM grid's near-degenerate
 * +/- node pairs would otherwise produce huge cancelling residues.
 *
 * Exactness within the DLR eps requires (flagged assumptions):
 *  (A-in)   the inputs are DLR-representable at the instance's (lambda, eps);
 *  (A-comp) pointwise products / real-shifted composites used in the single
 *           convolutions are representable (support enlarged by <= 2*wmax:
 *           choose wmax with headroom, exactly as for existing GG/WGG code);
 *  (A-node) the DLR real-frequency grid has no zero node (guarded).
 *
 * Batch semantics: ELEMENTWISE in the trailing (batch) axes. The caller
 * pre-aligns all momentum/orbital pairings; B/C/D/S share the tau leading axis
 * (nt_f DLR nodes), Wx/Wy the bosonic Matsubara leading axis (nw_b nodes),
 * cx/cy have batch size (any shape, flattened).
 *
 * The primitive requires the DLR IAFT backend and hard-aborts on IR.
 */

#include <cmath>
#include <complex>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"
#include "nda/nda.hpp"
#include "nda/blas.hpp"

#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/imag_axes_ft/dlr_pole_fit.hpp"

namespace imag_axes_ft {

  namespace dconv_detail {

    template<int N>
    using shape_t = std::array<long, N>;

    /** stable n_F(e) = 1/(exp(beta*e)+1) */
    inline double stable_nF(double beta, double e) {
      if (e >= 0.0) {
        double x = std::exp(-beta * e);
        return x / (1.0 + x);
      }
      return 1.0 / (1.0 + std::exp(beta * e));
    }

    /** stable n_B(e) = 1/(exp(beta*e)-1) (e != 0) */
    inline double stable_nB(double beta, double e) {
      if (e > 0.0) {
        double x = std::exp(-beta * e);
        return x / (1.0 - x);
      }
      return -1.0 / (1.0 - std::exp(beta * e));
    }

    /**
     * Implementation. All arrays flattened to 2D: leading = grid axis, trailing = batch.
     * cx/cy: flattened batch-shaped instantaneous parts; size 0 disables the respective part.
     */
    template<nda::MemoryArray AB, nda::MemoryArray AC, nda::MemoryArray AD,
             nda::MemoryArray AWX, nda::MemoryArray AWY, nda::MemoryArray AS>
    void double_boson_conv_impl(IAFT const &ft,
                                AB &&B_t, AC &&C_t, AD &&D_t,
                                AWX &&Wx_w, AWY &&Wy_w, AS &&S_t,
                                nda::array<ComplexType, 1> const &cx,
                                nda::array<ComplexType, 1> const &cy) {
#ifndef ENABLE_DLR
      (void)ft; (void)B_t; (void)C_t; (void)D_t; (void)Wx_w; (void)Wy_w; (void)S_t; (void)cx; (void)cy;
      APP_ABORT("double_boson_conv: this build has ENABLE_DLR=OFF; the DLR IAFT backend is required.");
#else
      static_assert(nda::get_rank<AB> == nda::get_rank<AC> and nda::get_rank<AB> == nda::get_rank<AD> and
                    nda::get_rank<AB> == nda::get_rank<AS>,
                    "double_boson_conv: B_t, C_t, D_t, S_t must have the same rank.");
      static_assert(nda::get_rank<AWX> == nda::get_rank<AWY> and nda::get_rank<AWX> == nda::get_rank<AB>,
                    "double_boson_conv: Wx_w, Wy_w must have the same rank as B_t (leading axis = nw_b).");

      utils::check(ft.basis() == dlr_basis,
                   "double_boson_conv: requires the DLR IAFT backend; IR is not supported.");

      const long nt   = ft.nt_f();
      const long nw_f = ft.nw_f();
      const long nw_b = ft.nw_b();
      const double beta = ft.beta();

      utils::check(B_t.shape(0) == nt and C_t.shape(0) == nt and D_t.shape(0) == nt and S_t.shape(0) == nt,
                   "double_boson_conv: B_t/C_t/D_t/S_t leading axis must be nt_f = {}.", nt);
      utils::check(Wx_w.shape(0) == nw_b and Wy_w.shape(0) == nw_b,
                   "double_boson_conv: Wx_w/Wy_w leading axis must be nw_b = {}.", nw_b);

      long d = 1;
      for (int i = 1; i < nda::get_rank<AB>; ++i) d *= B_t.shape(i);
      long dW = 1;
      for (int i = 1; i < nda::get_rank<AWX>; ++i) dW *= Wx_w.shape(i);
      utils::check(C_t.size() == nt * d and D_t.size() == nt * d and S_t.size() == nt * d,
                   "double_boson_conv: batch dims of B_t/C_t/D_t/S_t must match.");
      utils::check(dW == d and Wy_w.size() == nw_b * d,
                   "double_boson_conv: batch dims of Wx_w/Wy_w must match B_t.");
      utils::check(cx.size() == 0 or cx.size() == d, "double_boson_conv: cx batch size must be {}.", d);
      utils::check(cy.size() == 0 or cy.size() == d, "double_boson_conv: cy batch size must be {}.", d);

      auto B2  = nda::reshape(B_t,  shape_t<2>{nt, d});
      auto C2  = nda::reshape(C_t,  shape_t<2>{nt, d});
      auto D2  = nda::reshape(D_t,  shape_t<2>{nt, d});
      auto S2  = nda::reshape(S_t,  shape_t<2>{nt, d});
      auto Wx2 = nda::reshape(Wx_w, shape_t<2>{nw_b, d});
      auto Wy2 = nda::reshape(Wy_w, shape_t<2>{nw_b, d});

      auto Ttw_ff = ft.Ttw_ff();  // (nt, nw_f)
      auto Twt_ff = ft.Twt_ff();  // (nw_f, nt)
      auto Ttw_bb = ft.Ttw_bb();  // (nt, nw_b)
      auto Twt_bb = ft.Twt_bb();  // (nw_b, nt)

      // ---- auxiliary DLR pole basis, REGULARIZED fit ----------------------
      // The pole DATA lives on cppdlr's NONSYM grid: the backend's SYM grid contains
      // near-degenerate +/- node pairs whose interpolatory coefficients are huge and
      // cancelling (cond(cf2it) 6.2e10 vs 6.7e6, measured), and this algebra divides by
      // node gaps. What CHANGED (2026-07-28, see dlr_pole_fit.hpp for the measurements):
      // the residues used to come from an interpolation onto the aux tau nodes followed by
      // a square interpolatory solve there. That composite has 2-norm 9.4e6, so O(eps)
      // content the pole basis cannot resolve came back as residues hundreds of times the
      // data -- and this algebra is bilinear in them. They now come from a truncated-SVD
      // least-squares fit against the backend tau grid directly.
      imag_axes_ft::dlr_pole_fit pfit(ft);
      const long np = pfit.np;
      auto const& epsl = pfit.epsl;

      nda::array<double, 1> nF(np), nB(np), th(np);
      for (long l = 0; l < np; ++l) {
        nF(l) = stable_nF(beta, epsl(l));
        nB(l) = stable_nB(beta, epsl(l));
        th(l) = std::tanh(0.5 * pfit.rf(l));
      }
      pfit.log(2, "double_boson_conv");

      // Matsubara frequencies (physical, imaginary values)
      auto wnf = ft.wn_mesh_f();
      auto wnb = ft.wn_mesh_b();
      nda::array<ComplexType, 1> zf(nw_f), zb(nw_b);
      for (long n = 0; n < nw_f; ++n) zf(n) = ft.omega(wnf(n));
      for (long m = 0; m < nw_b; ++m) zb(m) = ft.omega(wnb(m));

      // ---- residues and node values --------------------------------------
      // fermionic residues = aux DLR time coefficients; bosonic residues
      // additionally carry tanh(hw_l/2)
      nda::array<ComplexType, 2> b_res(np, d), c_res(np, d), Y(np, d);
      {
        double fit_err = 0.0;
        b_res = pfit.coeffs(B2);
        fit_err = std::max(fit_err, pfit.fit_error(B2, b_res));
        c_res = pfit.coeffs(C2);
        fit_err = std::max(fit_err, pfit.fit_error(C2, c_res));
        nda::array<ComplexType, 2> Wy_t(nt, d);
        nda::blas::gemm(Ttw_bb, Wy2, Wy_t);
        auto ycoef = pfit.coeffs(Wy_t);
        fit_err = std::max(fit_err, pfit.fit_error(Wy_t, ycoef));
        for (long l = 0; l < np; ++l)
          for (long j = 0; j < d; ++j) Y(l, j) = th(l) * ycoef(l, j);
        // The residue algebra below is bilinear in b_res/c_res/Y, so this error enters the
        // result squared. Past the hard gate the pole representation has failed outright.
        app_log(3, "  double_boson_conv: auxiliary pole-fit reconstruction error = {:.4e}",
                fit_err);
        imag_axes_ft::dlr_pole_fit_gate(fit_err, "imag_axes_ft::double_boson_conv");
      }
      // D on the fermionic frequency nodes; Wx on the tau nodes
      nda::array<ComplexType, 2> D_w(nw_f, d), Wx_t(nt, d);
      nda::blas::gemm(Twt_ff, D2, D_w);
      nda::blas::gemm(Ttw_bb, Wx2, Wx_t);

      // ---- work arrays ----------------------------------------------------
      nda::array<ComplexType, 2> Kf(nw_f, np), Kb(nw_b, np);
      nda::array<ComplexType, 2> Hw(nw_f, d), Ht(nt, d), Vw(nw_b, d), Vt(nt, d);
      nda::array<ComplexType, 2> P(nt, d), conv_w(nw_f, d), pref(nw_f, d);
      nda::array<ComplexType, 2> S_w(nw_f, d);
      S_w() = ComplexType(0.0);

      auto max_abs = [](nda::array<ComplexType, 2> const &A) {
        double m = 0;
        for (auto const &v : A) m = std::max(m, std::abs(v));
        return m;
      };

      // =====================================================================
      // family (I):  sum_u  [-Y_u n_B(e_u) B(iw+e_u)] * T[Wx, D*C(.+e_u)]
      // =====================================================================
      for (long u = 0; u < np; ++u) {
        for (long n = 0; n < nw_f; ++n)
          for (long l = 0; l < np; ++l) Kf(n, l) = 1.0 / (zf(n) + epsl(u) - epsl(l));
        nda::blas::gemm(Kf, c_res, Hw);            // C(zf + e_u)
        for (long n = 0; n < nw_f; ++n)
          for (long j = 0; j < d; ++j) Hw(n, j) *= D_w(n, j);
        nda::blas::gemm(Ttw_ff, Hw, Ht);
        for (long i = 0; i < nt; ++i)
          for (long j = 0; j < d; ++j) P(i, j) = Wx_t(i, j) * Ht(i, j);
        nda::blas::gemm(Twt_ff, P, conv_w);
        nda::blas::gemm(Kf, b_res, pref);          // B(zf + e_u)
        const double fac = -nB(u);
        for (long n = 0; n < nw_f; ++n)
          for (long j = 0; j < d; ++j) S_w(n, j) += fac * Y(u, j) * pref(n, j) * conv_w(n, j);
      }
      app_log(3, "double_boson_conv: max|S_w| after family I   = {}", max_abs(S_w));

      // =====================================================================
      // family (II): sum_p  [b_p n_F(e_p) Wy(e_p-iw)] * T[Wx*C^{(p-defl)}(e_p-i.), D]
      // =====================================================================
      for (long p = 0; p < np; ++p) {
        for (long m = 0; m < nw_b; ++m)
          for (long l = 0; l < np; ++l)
            Kb(m, l) = (l == p) ? ComplexType(0.0) : 1.0 / (epsl(p) - zb(m) - epsl(l));
        nda::blas::gemm(Kb, c_res, Vw);            // C^{(p-defl)}(e_p - iv)
        for (long m = 0; m < nw_b; ++m)
          for (long j = 0; j < d; ++j) Vw(m, j) *= Wx2(m, j);
        nda::blas::gemm(Ttw_bb, Vw, Vt);
        for (long i = 0; i < nt; ++i)
          for (long j = 0; j < d; ++j) P(i, j) = Vt(i, j) * D2(i, j);
        nda::blas::gemm(Twt_ff, P, conv_w);
        for (long n = 0; n < nw_f; ++n)
          for (long l = 0; l < np; ++l) Kf(n, l) = 1.0 / (epsl(p) - zf(n) - epsl(l));
        nda::blas::gemm(Kf, Y, pref);              // Wy(e_p - iw)
        const double fac = nF(p);
        for (long n = 0; n < nw_f; ++n)
          for (long j = 0; j < d; ++j) S_w(n, j) += fac * b_res(p, j) * pref(n, j) * conv_w(n, j);
      }
      app_log(3, "double_boson_conv: max|S_w| after family II  = {}", max_abs(S_w));

      // =====================================================================
      // family (III): sum_q [c_q n_F(e_q)] * T[Wx*B^{(q-defl)}(e_q+i.), D*Wy(e_q-i.)]
      // =====================================================================
      for (long q = 0; q < np; ++q) {
        for (long m = 0; m < nw_b; ++m)
          for (long l = 0; l < np; ++l)
            Kb(m, l) = (l == q) ? ComplexType(0.0) : 1.0 / (epsl(q) + zb(m) - epsl(l));
        nda::blas::gemm(Kb, b_res, Vw);            // B^{(q-defl)}(e_q + iv)
        for (long m = 0; m < nw_b; ++m)
          for (long j = 0; j < d; ++j) Vw(m, j) *= Wx2(m, j);
        nda::blas::gemm(Ttw_bb, Vw, Vt);
        for (long n = 0; n < nw_f; ++n)
          for (long l = 0; l < np; ++l) Kf(n, l) = 1.0 / (epsl(q) - zf(n) - epsl(l));
        nda::blas::gemm(Kf, Y, Hw);                // Wy(e_q - iw')
        for (long n = 0; n < nw_f; ++n)
          for (long j = 0; j < d; ++j) Hw(n, j) *= D_w(n, j);
        nda::blas::gemm(Ttw_ff, Hw, Ht);
        for (long i = 0; i < nt; ++i)
          for (long j = 0; j < d; ++j) P(i, j) = Vt(i, j) * Ht(i, j);
        nda::blas::gemm(Twt_ff, P, conv_w);
        const double fac = nF(q);
        for (long n = 0; n < nw_f; ++n)
          for (long j = 0; j < d; ++j) S_w(n, j) += fac * c_res(q, j) * conv_w(n, j);
      }
      app_log(3, "double_boson_conv: max|S_w| after family III = {}", max_abs(S_w));

      // =====================================================================
      // family (IV): sum_{p,u} [b_p c_p n_F(e_p) Y_u/(e_p-iw-e_u)] * T[Wx, D/(i.-(e_p-e_u))]
      // (the exactly-removable p=q diagonal of the divided-difference form)
      // =====================================================================
      for (long p = 0; p < np; ++p) {
        const double fac = nF(p);
        for (long u = 0; u < np; ++u) {
          const double Delta = epsl(p) - epsl(u);
          for (long n = 0; n < nw_f; ++n) {
            const ComplexType den = 1.0 / (zf(n) - Delta);
            for (long j = 0; j < d; ++j) Hw(n, j) = D_w(n, j) * den;
          }
          nda::blas::gemm(Ttw_ff, Hw, Ht);
          for (long i = 0; i < nt; ++i)
            for (long j = 0; j < d; ++j) P(i, j) = Wx_t(i, j) * Ht(i, j);
          nda::blas::gemm(Twt_ff, P, conv_w);
          for (long n = 0; n < nw_f; ++n) {
            const ComplexType den = fac / (epsl(p) - zf(n) - epsl(u));
            for (long j = 0; j < d; ++j)
              S_w(n, j) += den * b_res(p, j) * c_res(p, j) * Y(u, j) * conv_w(n, j);
          }
        }
      }
      app_log(3, "double_boson_conv: max|S_w| after family IV  = {}", max_abs(S_w));

      // =====================================================================
      // family (V): true double-pole correction at the single lattice point
      // vx = 0 of the p=q diagonal. The divided-difference (families II-IV)
      // represents the two coinciding simple poles by their vx->0 limit; the
      // exact vx=0 residue additionally carries d/dz n_B:
      //   dT(iw; vx=0) = -beta sum_p b_p c_p n_F(e_p)(1-n_F(e_p)) Wy(e_p-iw)
      //   dS(iw) = (1/beta) Wx(0) D(iw) dT
      // =====================================================================
      {
        long m0 = -1;
        for (long m = 0; m < nw_b; ++m)
          if (wnb(m) == 0) { m0 = m; break; }
        utils::check(m0 >= 0, "double_boson_conv: bosonic Matsubara mesh does not contain iv = 0.");
        for (long p = 0; p < np; ++p) {
          const double fac = -nF(p) * (1.0 - nF(p));
          for (long n = 0; n < nw_f; ++n)
            for (long l = 0; l < np; ++l) Kf(n, l) = 1.0 / (epsl(p) - zf(n) - epsl(l));
          nda::blas::gemm(Kf, Y, pref);            // Wy(e_p - iw)
          for (long n = 0; n < nw_f; ++n)
            for (long j = 0; j < d; ++j)
              S_w(n, j) += fac * b_res(p, j) * c_res(p, j) * Wx2(m0, j) * D_w(n, j) * pref(n, j);
        }
      }
      app_log(3, "double_boson_conv: max|S_w| after family V   = {}", max_abs(S_w));

      // dynamic part to tau
      {
        nda::array<ComplexType, 2> Sdyn_t(nt, d);
        nda::blas::gemm(Ttw_ff, S_w, Sdyn_t);
        for (long i = 0; i < nt; ++i)
          for (long j = 0; j < d; ++j) S2(i, j) = Sdyn_t(i, j);
      }

      // =====================================================================
      // instantaneous parts (exact degenerate reductions):
      //   S1(t) = cx * B(t) * V(beta-t),   V = Wy*Q,  Q(t) = -C(t) D(beta-t)
      //   S2(t) = cy * D(t) * V'(t),       V' = Wx*Q', Q'(t) = -B(t) C(beta-t)
      //   S3(t) = -cx*cy * B(t) C(beta-t) D(t)
      // =====================================================================
      const bool has_cx = (cx.size() > 0), has_cy = (cy.size() > 0);
      if (has_cx or has_cy) {
        // exact tau -> beta-tau evaluation through the DLR basis (x -> -x)
        auto tau = ft.tau_mesh();
        nda::array<double, 1> xrefl(nt);
        for (long i = 0; i < nt; ++i) xrefl(i) = -tau(i);
        auto Rarr = ft.construct_tau_interpolate_matrix(xrefl);  // (nt, nt)
        nda::matrix<ComplexType> R(nt, nt);
        for (long i = 0; i < nt; ++i)
          for (long l = 0; l < nt; ++l) R(i, l) = Rarr(i, l);

        nda::array<ComplexType, 2> Crefl(nt, d), Qw(nw_b, d);
        nda::blas::gemm(R, C2, Crefl);

        if (has_cx) {
          nda::array<ComplexType, 2> Drefl(nt, d), Vrefl(nt, d);
          nda::blas::gemm(R, D2, Drefl);
          for (long i = 0; i < nt; ++i)
            for (long j = 0; j < d; ++j) P(i, j) = -C2(i, j) * Drefl(i, j);  // Q(t)
          nda::blas::gemm(Twt_bb, P, Qw);
          for (long m = 0; m < nw_b; ++m)
            for (long j = 0; j < d; ++j) Qw(m, j) *= Wy2(m, j);              // V(iv)
          nda::blas::gemm(Ttw_bb, Qw, Vt);
          nda::blas::gemm(R, Vt, Vrefl);                                     // V(beta-t)
          for (long i = 0; i < nt; ++i)
            for (long j = 0; j < d; ++j) S2(i, j) += cx(j) * B2(i, j) * Vrefl(i, j);
        }
        if (has_cy) {
          for (long i = 0; i < nt; ++i)
            for (long j = 0; j < d; ++j) P(i, j) = -B2(i, j) * Crefl(i, j);  // Q'(t)
          nda::blas::gemm(Twt_bb, P, Qw);
          for (long m = 0; m < nw_b; ++m)
            for (long j = 0; j < d; ++j) Qw(m, j) *= Wx2(m, j);              // V'(iv)
          nda::blas::gemm(Ttw_bb, Qw, Vt);
          for (long i = 0; i < nt; ++i)
            for (long j = 0; j < d; ++j) S2(i, j) += cy(j) * D2(i, j) * Vt(i, j);
        }
        if (has_cx and has_cy) {
          for (long i = 0; i < nt; ++i)
            for (long j = 0; j < d; ++j)
              S2(i, j) += -cx(j) * cy(j) * B2(i, j) * Crefl(i, j) * D2(i, j);
        }
      }
#endif  // ENABLE_DLR
    }

  }  // namespace dconv_detail

  /**
   * Double bosonic Matsubara convolution (dynamic parts only; see file header).
   *
   * @param ft    - [INPUT] IAFT instance; must hold the DLR backend (aborts on IR)
   * @param B_t   - [INPUT] (nt_f, batch...) fermionic factor B on the DLR tau nodes
   * @param C_t   - [INPUT] (nt_f, batch...) fermionic factor C on the DLR tau nodes
   * @param D_t   - [INPUT] (nt_f, batch...) fermionic factor D on the DLR tau nodes
   * @param Wx_w  - [INPUT] (nw_b, batch...) dynamic Wx on the bosonic Matsubara nodes
   * @param Wy_w  - [INPUT] (nw_b, batch...) dynamic Wy on the bosonic Matsubara nodes
   * @param S_t   - [OUTPUT] (nt_f, batch...) S on the DLR tau nodes (overwritten)
   */
  template<nda::MemoryArray AB, nda::MemoryArray AC, nda::MemoryArray AD,
           nda::MemoryArray AWX, nda::MemoryArray AWY, nda::MemoryArray AS>
  void double_boson_conv(IAFT const &ft, AB &&B_t, AC &&C_t, AD &&D_t,
                         AWX &&Wx_w, AWY &&Wy_w, AS &&S_t) {
    nda::array<ComplexType, 1> empty;
    dconv_detail::double_boson_conv_impl(ft, B_t, C_t, D_t, Wx_w, Wy_w, S_t, empty, empty);
  }

  /**
   * Overload with instantaneous (frequency-constant, batch-varying) parts cx/cy:
   * the full interactions are Wx(iv) + cx and Wy(iv) + cy. cx_i/cy_i are
   * batch-shaped arrays (any shape; flattened; size must equal the flattened
   * batch size of B_t). The instantaneous terms are computed by the exact
   * single-convolution reductions (a frequency constant is a delta(tau) in
   * time and is NOT DLR-representable, hence never passed through the
   * dynamic-part machinery).
   */
  template<nda::MemoryArray AB, nda::MemoryArray AC, nda::MemoryArray AD,
           nda::MemoryArray AWX, nda::MemoryArray AWY, nda::MemoryArray AS,
           nda::MemoryArray ACX, nda::MemoryArray ACY>
  void double_boson_conv(IAFT const &ft, AB &&B_t, AC &&C_t, AD &&D_t,
                         AWX &&Wx_w, AWY &&Wy_w, AS &&S_t,
                         ACX &&cx_i, ACY &&cy_i) {
    nda::array<ComplexType, 1> cx(cx_i.size()), cy(cy_i.size());
    {
      auto cxf = nda::reshape(cx_i, dconv_detail::shape_t<1>{long(cx_i.size())});
      auto cyf = nda::reshape(cy_i, dconv_detail::shape_t<1>{long(cy_i.size())});
      for (long j = 0; j < long(cx_i.size()); ++j) cx(j) = cxf(j);
      for (long j = 0; j < long(cy_i.size()); ++j) cy(j) = cyf(j);
    }
    dconv_detail::double_boson_conv_impl(ft, B_t, C_t, D_t, Wx_w, Wy_w, S_t, cx, cy);
  }

}  // namespace imag_axes_ft

#endif  // COQUI_IAFT_DCONV_HPP
