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

// Unit tests for imag_axes_ft::double_boson_conv (DLR double bosonic Matsubara
// convolution). See notes/double_convolution_design.md for the algorithm.

#undef NDEBUG

#include <vector>
#include <cmath>
#include <complex>

#include "catch2/catch.hpp"
#include "configuration.hpp"
#include "nda/nda.hpp"

#include "utilities/test_common.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/imag_axes_ft/iaft_dconv.hpp"

namespace bdft_tests {

  namespace dconv_test {

    using cplx = ComplexType;
    constexpr cplx I{0.0, 1.0};

    using imag_axes_ft::dconv_detail::stable_nF;
    using imag_axes_ft::dconv_detail::stable_nB;

    // ---- analytic pole models --------------------------------------------
    // fermionic: F(z) = sum_k w_k / (z - e_k)
    struct FModel {
      std::vector<double> e, w;
    };
    // bosonic:  W(z) = sum_k g2_k * 2*Om_k / (Om_k^2 - z^2)   [W(iv) = g2*2*Om/(Om^2+v^2)]
    struct BModel {
      std::vector<double> Om, g2;
    };

    inline cplx evalF(FModel const &m, cplx z) {
      cplx r = 0;
      for (size_t k = 0; k < m.e.size(); ++k) r += m.w[k] / (z - m.e[k]);
      return r;
    }

    // stable physical kernel: single pole e -> G(tau) = -exp(-tau*e)/(1+exp(-beta*e))
    inline double ktau_phys(double beta, double tau, double e) {
      if (e >= 0.0) return -std::exp(-tau * e) / (1.0 + std::exp(-beta * e));
      return -std::exp((beta - tau) * e) / (1.0 + std::exp(beta * e));
    }

    inline double evalF_tau(FModel const &m, double beta, double tau) {
      double r = 0;
      for (size_t k = 0; k < m.e.size(); ++k) r += m.w[k] * ktau_phys(beta, tau, m.e[k]);
      return r;
    }

    inline cplx evalW(BModel const &m, cplx z) {
      cplx r = 0;
      for (size_t k = 0; k < m.Om.size(); ++k) r += m.g2[k] * 2.0 * m.Om[k] / (m.Om[k] * m.Om[k] - z * z);
      return r;
    }
    inline cplx evalWp(BModel const &m, cplx z) {  // dW/dz
      cplx r = 0;
      for (size_t k = 0; k < m.Om.size(); ++k) {
        cplx den = m.Om[k] * m.Om[k] - z * z;
        r += m.g2[k] * 2.0 * m.Om[k] * 2.0 * z / (den * den);
      }
      return r;
    }
    // signed pole list of W(z): poles at +/-Om with residues -/+ g2
    inline void bose_poles(BModel const &m, std::vector<double> &rho, std::vector<double> &res) {
      rho.clear(); res.clear();
      for (size_t k = 0; k < m.Om.size(); ++k) {
        rho.push_back(m.Om[k]);  res.push_back(-m.g2[k]);
        rho.push_back(-m.Om[k]); res.push_back(m.g2[k]);
      }
    }

    // ---- closed-form T(iw, ivx) [stage-1 formula, test-local implementation] ----
    // T = -sum_rho v_rho n_B(rho) B(iw+rho) C(iw-ivx+rho)
    //     +sum_{p,q} b_p c_q DD,  DD = [nF(ep) Wy(ep-iw) - nF(eq) Wy(eq+ivx-iw)]/(ep-eq-ivx)
    inline cplx eval_T_closed(double beta, FModel const &B, FModel const &C, BModel const &Wy,
                              cplx iw, cplx ivx) {
      std::vector<double> rho, vres;
      bose_poles(Wy, rho, vres);
      cplx T = 0;
      for (size_t u = 0; u < rho.size(); ++u)
        T += -vres[u] * stable_nB(beta, rho[u]) * evalF(B, iw + rho[u]) * evalF(C, iw - ivx + rho[u]);
      for (size_t p = 0; p < B.e.size(); ++p) {
        for (size_t q = 0; q < C.e.size(); ++q) {
          cplx a = B.e[p];
          cplx b = C.e[q] + ivx;
          cplx dd;
          if (std::abs(a - b) < 1e-12) {
            // true double pole: -d/dz[Wy(z) n_B(z)] at z = e_p - iw picks up the n_B' term
            double nf = stable_nF(beta, B.e[p]);
            dd = nf * evalWp(Wy, a - iw) - beta * nf * (1.0 - nf) * evalW(Wy, a - iw);
          } else {
            dd = (stable_nF(beta, B.e[p]) * evalW(Wy, a - iw) -
                  stable_nF(beta, C.e[q]) * evalW(Wy, b - iw)) / (a - b);
          }
          T += B.w[p] * C.w[q] * dd;
        }
      }
      return T;
    }

    // dense (1/beta) sum_{|m|<=M} Wy(iv_m) B(iw+iv_m) C(iw-ivx+iv_m)
    inline cplx eval_T_dense(double beta, FModel const &B, FModel const &C, BModel const &Wy,
                             cplx iw, cplx ivx, long M) {
      cplx T = 0;
      for (long m = -M; m <= M; ++m) {
        cplx iv = I * (2.0 * M_PI * m / beta);
        T += evalW(Wy, iv) * evalF(B, iw + iv) * evalF(C, iw - ivx + iv);
      }
      return T / beta;
    }

    // fill input arrays on the IAFT grids from the analytic models
    template<typename ArrT>
    void fill_F_tau(imag_axes_ft::IAFT const &ft, FModel const &m, ArrT &&A, long j) {
      auto x = ft.tau_mesh();
      double beta = ft.beta();
      for (long i = 0; i < ft.nt_f(); ++i) {
        double tau = (x(i) + 1.0) * 0.5 * beta;
        A(i, j) = evalF_tau(m, beta, tau);
      }
    }
    template<typename ArrT>
    void fill_W_iv(imag_axes_ft::IAFT const &ft, BModel const &m, ArrT &&A, long j) {
      auto wn = ft.wn_mesh_b();
      double beta = ft.beta();
      for (long i = 0; i < ft.nw_b(); ++i) {
        cplx iv = I * (wn(i) * M_PI / beta);
        A(i, j) = evalW(m, iv);
      }
    }

  }  // namespace dconv_test

  using namespace dconv_test;

  TEST_CASE("dconv_conventions", "[iaft_dconv]") {
    const double beta = 20.0;

    // --- pin the bosonic residue-sum sign convention (kappa = -1):
    //     (1/beta) sum_m h(iv_m) = - sum_j Res[h](z_j) n_B(z_j),  h ~ 1/z^3
    {
      const double a = 0.8, b = -1.3, c = 0.4;
      auto h = [&](cplx z) { return 1.0 / ((z - a) * (z - b) * (z - c)); };
      const long M = 200000;
      cplx dense = 0;
      for (long m = -M; m <= M; ++m) dense += h(I * (2.0 * M_PI * m / beta));
      dense /= beta;
      cplx closed = -(stable_nB(beta, a) / ((a - b) * (a - c)) +
                      stable_nB(beta, b) / ((b - a) * (b - c)) +
                      stable_nB(beta, c) / ((c - a) * (c - b)));
      app_log(1, "dconv_conventions: kappa pin |dense - closed| = {}", std::abs(dense - closed));
      REQUIRE(std::abs(dense - closed) < 1e-8);
    }

    // --- pin the stage-1 closed form T(iw,ivx) against dense vy sums,
    //     including a shared-pole B/C variant at vx=0 (divided-difference limit)
    {
      FModel B{{-1.3, 0.7}, {0.6, 0.4}};
      FModel C1{{-0.45, 1.1}, {0.55, 0.45}};   // disjoint poles
      FModel C2{{-1.3, 0.7}, {0.35, 0.65}};    // SAME pole locations as B
      BModel Wy{{1.5}, {0.8}};

      const long M = 40000;
      double max_err = 0.0;
      for (auto const &C : {C1, C2}) {
        for (long nf : {1L, 5L, 17L}) {          // fermionic iw = i*nf*pi/beta (nf odd)
          for (long mb : {0L, 2L, -6L, 20L}) {   // bosonic ivx = i*mb*pi/beta (mb even)
            cplx iw = I * (nf * M_PI / beta);
            cplx ivx = I * (mb * M_PI / beta);
            cplx Tc = eval_T_closed(beta, B, C, Wy, iw, ivx);
            cplx Td = eval_T_dense(beta, B, C, Wy, iw, ivx, M);
            max_err = std::max(max_err, std::abs(Tc - Td));
          }
        }
      }
      app_log(1, "dconv_conventions: stage-1 T closed-form max err vs dense = {}", max_err);
      REQUIRE(max_err < 1e-9);
    }

    // --- pin the auxiliary (NONSYM) DLR pole-data map used by the primitive:
    //     fermionic residues == aux DLR time coefficients, bosonic residues ==
    //     coeffs * tanh(hw_l/2), poles eps_l = dlr_rf(l)/beta. The coefficient
    //     magnitudes must be moderate (well-conditioned interpolation).
    {
      const double wmax = 12.0;
      imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, "high");
      const long nt = ft.nt_f(), nw_f = ft.nw_f(), nw_b = ft.nw_b();

      auto rf = cppdlr::build_dlr_rf(ft.lambda(), ft.eps());  // NONSYM aux grid
      auto itops_p = cppdlr::imtime_ops(ft.lambda(), rf);
      const long np = itops_p.rank();
      nda::array<double, 1> x_p(np);
      for (long i = 0; i < np; ++i) x_p(i) = 2.0 * cppdlr::rel2abs(itops_p.get_itnodes(i)) - 1.0;
      auto Tmap = ft.construct_tau_interpolate_matrix(x_p);  // (np, nt)

      FModel B{{-1.3, 0.7}, {0.6, 0.4}};
      BModel Wy{{1.5}, {0.8}};

      nda::array<cplx, 2> B_t(nt, 1), Wy_w(nw_b, 1);
      fill_F_tau(ft, B, B_t, 0);
      fill_W_iv(ft, Wy, Wy_w, 0);

      auto to_aux_coefs = [&](nda::array<cplx, 2> const &A_t) {
        nda::array<cplx, 2> An(np, 1);
        for (long i = 0; i < np; ++i) {
          An(i, 0) = 0;
          for (long l = 0; l < nt; ++l) An(i, 0) += Tmap(i, l) * A_t(l, 0);
        }
        return itops_p.vals2coefs(An);
      };

      // fermionic: the pole rep must agree with the true function on the dense
      // Matsubara LATTICE (that is what the double convolution couples to; the
      // DLR model is not required to agree pointwise off the imaginary axis).
      auto b_res = to_aux_coefs(B_t);
      double cmax = 0;
      for (long l = 0; l < np; ++l) cmax = std::max(cmax, std::abs(b_res(l, 0)));
      double errF = 0;
      for (long n : {0L, 1L, 3L, 10L, 50L, 500L, 20000L}) {
        cplx z = I * ((2 * n + 1) * M_PI / beta);  // dense fermionic lattice points
        cplx pole_rep = 0;
        for (long l = 0; l < np; ++l) pole_rep += b_res(l, 0) / (z - rf(l) / beta);
        errF = std::max(errF, std::abs(pole_rep - evalF(B, z)));
      }
      app_log(1, "dconv_conventions: pole-map fermion max lattice err = {} (max|coef| = {})", errF, cmax);
      REQUIRE(errF < 1e-9);
      REQUIRE(cmax < 100.0);

      // bosonic: Matsubara values -> tau values -> aux coefficients
      nda::array<cplx, 2> Wy_t(nt, 1);
      nda::blas::gemm(ft.Ttw_bb(), Wy_w, Wy_t);
      auto ycoef = to_aux_coefs(Wy_t);
      double ymax = 0;
      for (long l = 0; l < np; ++l) ymax = std::max(ymax, std::abs(ycoef(l, 0)));
      double errB = 0;
      for (long n : {1L, 2L, 5L, 20L, 100L, 5000L}) {
        cplx z = I * (2 * n * M_PI / beta);  // dense bosonic lattice points
        cplx pole_rep = 0;
        for (long l = 0; l < np; ++l)
          pole_rep += ycoef(l, 0) * std::tanh(0.5 * rf(l)) / (z - rf(l) / beta);
        errB = std::max(errB, std::abs(pole_rep - evalW(Wy, z)));
      }
      app_log(1, "dconv_conventions: pole-map boson max lattice err = {} (max|coef| = {})", errB, ymax);
      REQUIRE(errB < 1e-9);
      REQUIRE(ymax < 100.0);
    }
  }

  TEST_CASE("dconv_brute_force", "[iaft_dconv]") {
    const double beta = 20.0, wmax = 12.0;
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, "high");
    const long nt = ft.nt_f(), nw_f = ft.nw_f(), nw_b = ft.nw_b();
    app_log(1, "dconv_brute_force: DLR rank = {}, nw_f = {}, nw_b = {}, eps = {}", nt, nw_f, nw_b, ft.eps());

    FModel B{{-1.3, 0.7}, {0.6, 0.4}};
    FModel C{{-0.45, 1.1}, {0.55, 0.45}};
    FModel D{{0.9, -0.8}, {0.7, 0.3}};
    BModel Wx{{2.2}, {0.6}};
    BModel Wy{{1.5}, {0.8}};
    const cplx cx0 = 0.35, cy0 = -0.2;

    // inputs on the DLR grids (rank-2, batch d=1)
    nda::array<cplx, 2> B_t(nt, 1), C_t(nt, 1), D_t(nt, 1), S_t(nt, 1);
    nda::array<cplx, 2> Wx_w(nw_b, 1), Wy_w(nw_b, 1);
    fill_F_tau(ft, B, B_t, 0);
    fill_F_tau(ft, C, C_t, 0);
    fill_F_tau(ft, D, D_t, 0);
    fill_W_iv(ft, Wx, Wx_w, 0);
    fill_W_iv(ft, Wy, Wy_w, 0);
    nda::array<cplx, 1> cx(1), cy(1);
    cx(0) = cx0;
    cy(0) = cy0;

    imag_axes_ft::double_boson_conv(ft, B_t, C_t, D_t, Wx_w, Wy_w, S_t, cx, cy);

    nda::array<cplx, 2> S_w(nw_f, 1);
    ft.tau_to_w(S_t, S_w, imag_axes_ft::fermion);

    // ---- brute-force reference on the fermionic nodes ----------------------
    auto wnf = ft.wn_mesh_f();

    // dyn (x) dyn : dense double Matsubara sum with cutoff M (+ convergence check)
    auto dd_ref = [&](long n, long M) {
      cplx iw = I * (wnf(n) * M_PI / beta);
      // precompute C at iw + i*2*pi*k/beta for k in [-2M, 2M]
      std::vector<cplx> Cs(4 * M + 1);
      for (long k = -2 * M; k <= 2 * M; ++k) Cs[k + 2 * M] = evalF(C, iw + I * (2.0 * M_PI * k / beta));
      std::vector<cplx> WyB(2 * M + 1), WxD(2 * M + 1);
      for (long m = -M; m <= M; ++m) {
        cplx iv = I * (2.0 * M_PI * m / beta);
        WyB[m + M] = evalW(Wy, iv) * evalF(B, iw + iv);
        WxD[m + M] = evalW(Wx, iv) * evalF(D, iw - iv);
      }
      cplx s = 0;
      for (long mx = -M; mx <= M; ++mx) {
        cplx sy = 0;
        for (long my = -M; my <= M; ++my) sy += WyB[my + M] * Cs[my - mx + 2 * M];
        s += WxD[mx + M] * sy;
      }
      return s / (beta * beta);
    };

    {  // cutoff convergence verification on a few nodes
      double conv = 0.0;
      for (long n : {0L, nw_f / 2, nw_f - 1}) conv = std::max(conv, std::abs(dd_ref(n, 1024) - dd_ref(n, 2048)));
      app_log(1, "dconv_brute_force: dd cutoff convergence |S(M=1024)-S(M=2048)| = {}", conv);
      REQUIRE(conv < 1e-9);
    }

    // cx (x) dyn and dyn (x) cy : 1D dense sums with analytic pair bubbles
    // Q (iv) = (1/beta) sum_k C(ik+iv) D(ik)   = sum_{q,r} c_q d_r (nF(e_r)-nF(e_q))/(iv+e_r-e_q)
    // Q'(iv) = (1/beta) sum_k B(ik+iv) C(ik)   = sum_{p,q} b_p c_q (nF(e_q)-nF(e_p))/(iv+e_q-e_p)
    auto Qpair = [&](FModel const &F1, FModel const &F2, cplx iv) {
      cplx r = 0;
      for (size_t a = 0; a < F1.e.size(); ++a)
        for (size_t b = 0; b < F2.e.size(); ++b)
          r += F1.w[a] * F2.w[b] * (stable_nF(beta, F2.e[b]) - stable_nF(beta, F1.e[a])) /
               (iv + F2.e[b] - F1.e[a]);
      return r;
    };
    const long M1 = 200000;
    auto Sx_ref = [&](long n) {  // cx * (1/beta) sum_vy Wy B(iw+ivy) Q(ivy)
      cplx iw = I * (wnf(n) * M_PI / beta), s = 0;
      for (long m = -M1; m <= M1; ++m) {
        cplx iv = I * (2.0 * M_PI * m / beta);
        s += evalW(Wy, iv) * evalF(B, iw + iv) * Qpair(C, D, iv);
      }
      return cx0 * s / beta;
    };
    auto Sy_ref = [&](long n) {  // cy * (1/beta) sum_vx Wx Q'(ivx) D(iw-ivx)
      cplx iw = I * (wnf(n) * M_PI / beta), s = 0;
      for (long m = -M1; m <= M1; ++m) {
        cplx iv = I * (2.0 * M_PI * m / beta);
        s += evalW(Wx, iv) * Qpair(B, C, iv) * evalF(D, iw - iv);
      }
      return cy0 * s / beta;
    };
    // cx (x) cy : fully analytic
    auto Sxy_ref = [&](long n) {
      cplx iw = I * (wnf(n) * M_PI / beta), s = 0;
      for (size_t p = 0; p < B.e.size(); ++p)
        for (size_t q = 0; q < C.e.size(); ++q)
          for (size_t r = 0; r < D.e.size(); ++r) {
            double np = stable_nF(beta, B.e[p]), nq = stable_nF(beta, C.e[q]), nr = stable_nF(beta, D.e[r]);
            double Ebar = B.e[p] - C.e[q] + D.e[r];
            s += B.w[p] * C.w[q] * D.w[r] * (np * (1 - nq) * nr + (1 - np) * nq * (1 - nr)) / (iw - Ebar);
          }
      return -cx0 * cy0 * s;
    };

    double max_abs_err = 0.0, max_val = 0.0;
    for (long n = 0; n < nw_f; ++n) {
      cplx ref = dd_ref(n, 2048) + Sx_ref(n) + Sy_ref(n) + Sxy_ref(n);
      max_abs_err = std::max(max_abs_err, std::abs(S_w(n, 0) - ref));
      max_val = std::max(max_val, std::abs(ref));
    }
    app_log(1, "dconv_brute_force: max |S_prim - S_ref| = {}  (max |S_ref| = {}, rel = {})",
            max_abs_err, max_val, max_abs_err / max_val);
    REQUIRE(max_abs_err / max_val < 1e-7);
  }

  TEST_CASE("dconv_degenerate_reduction", "[iaft_dconv]") {
    const double beta = 20.0, wmax = 12.0;
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, "high");
    const long nt = ft.nt_f(), nw_b = ft.nw_b();

    FModel B{{-1.3, 0.7}, {0.6, 0.4}};
    FModel C{{-0.45, 1.1}, {0.55, 0.45}};
    FModel D{{0.9, -0.8}, {0.7, 0.3}};
    BModel Wx{{2.2}, {0.6}};
    BModel Wy{{1.5}, {0.8}};

    nda::array<cplx, 2> B_t(nt, 1), C_t(nt, 1), D_t(nt, 1), S_t(nt, 1);
    nda::array<cplx, 2> Wx_w(nw_b, 1), Wy_w(nw_b, 1), Wzero(nw_b, 1);
    fill_F_tau(ft, B, B_t, 0);
    fill_F_tau(ft, C, C_t, 0);
    fill_F_tau(ft, D, D_t, 0);
    fill_W_iv(ft, Wx, Wx_w, 0);
    fill_W_iv(ft, Wy, Wy_w, 0);
    Wzero() = cplx(0.0);

    // exact tau -> beta-tau map through the DLR basis
    auto x = ft.tau_mesh();
    nda::array<double, 1> xrefl(nt);
    for (long i = 0; i < nt; ++i) xrefl(i) = -x(i);
    auto R = ft.construct_tau_interpolate_matrix(xrefl);

    auto reflect = [&](nda::array<cplx, 2> const &A) {
      nda::array<cplx, 2> Ar(nt, 1);
      for (long i = 0; i < nt; ++i) {
        Ar(i, 0) = 0;
        for (long l = 0; l < nt; ++l) Ar(i, 0) += R(i, l) * A(l, 0);
      }
      return Ar;
    };

    // (a) instantaneous Wx (cx path): S(t) = cx * B(t) * V(beta-t), V = Wy*Q, Q = -C(t) D(beta-t)
    {
      const cplx cx0 = 0.5;
      nda::array<cplx, 1> cx(1), cy(1);
      cx(0) = cx0;
      cy(0) = 0.0;
      imag_axes_ft::double_boson_conv(ft, B_t, C_t, D_t, Wzero, Wy_w, S_t, cx, cy);

      // independent single-convolution reference via the plain IAFT pattern
      auto Drefl = reflect(D_t);
      nda::array<cplx, 2> Q_t(nt, 1), Q_w(nw_b, 1), V_t(nt, 1);
      for (long i = 0; i < nt; ++i) Q_t(i, 0) = -C_t(i, 0) * Drefl(i, 0);
      ft.tau_to_w(Q_t, Q_w, imag_axes_ft::boson);
      for (long m = 0; m < nw_b; ++m) Q_w(m, 0) *= Wy_w(m, 0);
      ft.w_to_tau(Q_w, V_t, imag_axes_ft::boson);
      auto Vrefl = reflect(V_t);

      double max_err = 0.0, scale = 0.0;
      for (long i = 0; i < nt; ++i) {
        cplx ref = cx0 * B_t(i, 0) * Vrefl(i, 0);
        max_err = std::max(max_err, std::abs(S_t(i, 0) - ref));
        scale = std::max(scale, std::abs(ref));
      }
      app_log(1, "dconv_degenerate (cx path): max err = {} (scale {})", max_err, scale);
      REQUIRE(max_err < 1e-12 * std::max(scale, 1.0));
    }

    // (b) instantaneous Wy (cy path): S(t) = cy * D(t) * V'(t), V' = Wx*Q', Q' = -B(t) C(beta-t)
    {
      const cplx cy0 = -0.7;
      nda::array<cplx, 1> cx(1), cy(1);
      cx(0) = 0.0;
      cy(0) = cy0;
      imag_axes_ft::double_boson_conv(ft, B_t, C_t, D_t, Wx_w, Wzero, S_t, cx, cy);

      auto Crefl = reflect(C_t);
      nda::array<cplx, 2> Q_t(nt, 1), Q_w(nw_b, 1), V_t(nt, 1);
      for (long i = 0; i < nt; ++i) Q_t(i, 0) = -B_t(i, 0) * Crefl(i, 0);
      ft.tau_to_w(Q_t, Q_w, imag_axes_ft::boson);
      for (long m = 0; m < nw_b; ++m) Q_w(m, 0) *= Wx_w(m, 0);
      ft.w_to_tau(Q_w, V_t, imag_axes_ft::boson);

      double max_err = 0.0, scale = 0.0;
      for (long i = 0; i < nt; ++i) {
        cplx ref = cy0 * D_t(i, 0) * V_t(i, 0);
        max_err = std::max(max_err, std::abs(S_t(i, 0) - ref));
        scale = std::max(scale, std::abs(ref));
      }
      app_log(1, "dconv_degenerate (cy path): max err = {} (scale {})", max_err, scale);
      REQUIRE(max_err < 1e-12 * std::max(scale, 1.0));
    }
  }

  TEST_CASE("dconv_symmetry", "[iaft_dconv]") {
    const double beta = 20.0, wmax = 12.0;
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, "high");
    const long nt = ft.nt_f(), nw_b = ft.nw_b();

    // particle-hole symmetric fermionic models: G(beta-tau) = G(tau)
    FModel B{{-1.1, 1.1}, {0.5, 0.5}};
    FModel C{{-0.6, 0.6}, {0.5, 0.5}};
    FModel D{{-0.9, 0.9}, {0.5, 0.5}};
    BModel Wx{{2.2}, {0.6}};
    BModel Wy{{1.5}, {0.8}};

    nda::array<cplx, 2> B_t(nt, 1), C_t(nt, 1), D_t(nt, 1), S_t(nt, 1), S2_t(nt, 1);
    nda::array<cplx, 2> Wx_w(nw_b, 1), Wy_w(nw_b, 1);
    fill_F_tau(ft, B, B_t, 0);
    fill_F_tau(ft, C, C_t, 0);
    fill_F_tau(ft, D, D_t, 0);
    fill_W_iv(ft, Wx, Wx_w, 0);
    fill_W_iv(ft, Wy, Wy_w, 0);
    nda::array<cplx, 1> cx(1), cy(1);
    cx(0) = 0.3;
    cy(0) = 0.3;

    imag_axes_ft::double_boson_conv(ft, B_t, C_t, D_t, Wx_w, Wy_w, S_t, cx, cy);

    // (a) PH symmetry of the output: S(beta-tau) = S(tau)
    {
      auto x = ft.tau_mesh();
      nda::array<double, 1> xrefl(nt);
      for (long i = 0; i < nt; ++i) xrefl(i) = -x(i);
      auto R = ft.construct_tau_interpolate_matrix(xrefl);
      double max_err = 0.0, scale = 0.0;
      for (long i = 0; i < nt; ++i) {
        cplx sr = 0;
        for (long l = 0; l < nt; ++l) sr += R(i, l) * S_t(l, 0);
        max_err = std::max(max_err, std::abs(sr - S_t(i, 0)));
        scale = std::max(scale, std::abs(S_t(i, 0)));
      }
      app_log(1, "dconv_symmetry: max |S(beta-tau) - S(tau)| = {} (scale {})", max_err, scale);
      REQUIRE(max_err < 1e-9 * std::max(scale, 1.0));
    }

    // (b) W(-iv) = W(iv) handling: feeding the nu-mirrored arrays gives identical S
    {
      auto wn = ft.wn_mesh_b();
      nda::array<cplx, 2> Wx_m(nw_b, 1), Wy_m(nw_b, 1);
      for (long m = 0; m < nw_b; ++m) {
        long mm = -1;
        for (long m2 = 0; m2 < nw_b; ++m2)
          if (wn(m2) == -wn(m)) { mm = m2; break; }
        REQUIRE(mm >= 0);  // the bosonic mesh contains the mirrored frequency
        Wx_m(m, 0) = Wx_w(mm, 0);
        Wy_m(m, 0) = Wy_w(mm, 0);
      }
      imag_axes_ft::double_boson_conv(ft, B_t, C_t, D_t, Wx_m, Wy_m, S2_t, cx, cy);
      double max_err = 0.0;
      for (long i = 0; i < nt; ++i) max_err = std::max(max_err, std::abs(S2_t(i, 0) - S_t(i, 0)));
      app_log(1, "dconv_symmetry: mirrored-W max diff = {}", max_err);
      REQUIRE(max_err < 1e-12);
    }
  }

  TEST_CASE("dconv_batch", "[iaft_dconv]") {
    const double beta = 20.0, wmax = 12.0;
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::dlr_basis, "high");
    const long nt = ft.nt_f(), nw_b = ft.nw_b();

    std::vector<FModel> Bv = {{{-1.3, 0.7}, {0.6, 0.4}}, {{-0.5, 1.4}, {0.3, 0.7}}};
    std::vector<FModel> Cv = {{{-0.45, 1.1}, {0.55, 0.45}}, {{-1.0, 0.2}, {0.5, 0.5}}};
    std::vector<FModel> Dv = {{{0.9, -0.8}, {0.7, 0.3}}, {{0.4, -1.2}, {0.45, 0.55}}};
    std::vector<BModel> Wxv = {{{2.2}, {0.6}}, {{1.1}, {0.9}}};
    std::vector<BModel> Wyv = {{{1.5}, {0.8}}, {{2.8}, {0.4}}};
    std::vector<cplx> cxv = {0.35, -0.15}, cyv = {-0.2, 0.6};

    // batched call (batch shape (2,))
    nda::array<cplx, 2> B_t(nt, 2), C_t(nt, 2), D_t(nt, 2), S_t(nt, 2);
    nda::array<cplx, 2> Wx_w(nw_b, 2), Wy_w(nw_b, 2);
    nda::array<cplx, 1> cx(2), cy(2);
    for (long j = 0; j < 2; ++j) {
      fill_F_tau(ft, Bv[j], B_t, j);
      fill_F_tau(ft, Cv[j], C_t, j);
      fill_F_tau(ft, Dv[j], D_t, j);
      fill_W_iv(ft, Wxv[j], Wx_w, j);
      fill_W_iv(ft, Wyv[j], Wy_w, j);
      cx(j) = cxv[j];
      cy(j) = cyv[j];
    }
    imag_axes_ft::double_boson_conv(ft, B_t, C_t, D_t, Wx_w, Wy_w, S_t, cx, cy);

    // per-element calls
    double max_err = 0.0, scale = 0.0;
    for (long j = 0; j < 2; ++j) {
      nda::array<cplx, 2> b(nt, 1), c(nt, 1), dd(nt, 1), s(nt, 1), wx(nw_b, 1), wy(nw_b, 1);
      nda::array<cplx, 1> cx1(1), cy1(1);
      fill_F_tau(ft, Bv[j], b, 0);
      fill_F_tau(ft, Cv[j], c, 0);
      fill_F_tau(ft, Dv[j], dd, 0);
      fill_W_iv(ft, Wxv[j], wx, 0);
      fill_W_iv(ft, Wyv[j], wy, 0);
      cx1(0) = cxv[j];
      cy1(0) = cyv[j];
      imag_axes_ft::double_boson_conv(ft, b, c, dd, wx, wy, s, cx1, cy1);
      for (long i = 0; i < nt; ++i) {
        max_err = std::max(max_err, std::abs(S_t(i, j) - s(i, 0)));
        scale = std::max(scale, std::abs(s(i, 0)));
      }
    }
    app_log(1, "dconv_batch: max |batched - per-element| = {} (scale {})", max_err, scale);
    REQUIRE(max_err < 1e-12 * std::max(scale, 1.0));
  }

}  // namespace bdft_tests
