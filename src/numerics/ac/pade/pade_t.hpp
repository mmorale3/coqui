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


#ifndef COQUI_PADE_HPP
#define COQUI_PADE_HPP

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/omp_threads.hpp"
#include "nda/nda.hpp"

#include <boost/multiprecision/cpp_dec_float.hpp>

namespace analyt_cont {
  enum class pade_impl_e {
    original,
    updated
  };

  template<int prec=100>
  class pade_t {
  public:

    // Customized precision
    using mp_float = boost::multiprecision::number<boost::multiprecision::cpp_dec_float<prec>, boost::multiprecision::et_off>;
    using mp_complex = std::complex<mp_float>;

  public:
    explicit pade_t(pade_impl_e impl = pade_impl_e::original): _impl(impl) {}
    pade_t(pade_t const&) = default;
    pade_t(pade_t &&) = default;
    pade_t& operator=(pade_t const& other) = default;
    pade_t& operator=(pade_t && other) = default;

    ~pade_t() = default;

    /**
     * interpolate A(iw) to the full complex plane using pade interpolation
     * @param iw_mesh - [INPUT] imaginary frequency grid
     * @param A_iw    - [INPUT] function on the Matsuabara axis
     */
    template<nda::ArrayOfRank<1> iw_mesh_t, nda::ArrayOfRank<2> Array_iw_t>
    void init(iw_mesh_t &&iw_mesh, Array_iw_t &&A_iw) {
      _Nfit = A_iw.shape(0);
      _dim1 = A_iw.shape(1);
      utils::check(_Nfit >= 1, "pade_t.hpp::init: Nfit ({}) must be at least 1", _Nfit);

      _iw_fit = nda::array<mp_complex, 1>(_Nfit);
      _coeffs = nda::array<mp_complex, 2>(_dim1, _Nfit);
      _orders = nda::array<long, 1>(_dim1);

      for (long w=0; w<_Nfit; ++w)
        _iw_fit(w) = mp_complex(mp_float(iw_mesh(w).real()), mp_float(iw_mesh(w).imag()));

      // ---------------- T-3b threaded region 1 of 4 ----------------
      // notes/coqui_threading_t3a.md section 2.3 ROW 1 / section 3.1: the per-state Thiele
      // reciprocal-difference fit, 84% of the whole QP-map phase and its single largest
      // parity loss (+11.2 s at 24r x 4t, +25.2 s at 12r x 8t).
      //
      // SAFETY INVENTORY (spec section 7.2 item 2, hazards from t3a section 4.2):
      //   * no MPI, no HDF5, no logging, no utils::check on this path -- init's own checks
      //     and pade_driver::init's app_log are all OUTSIDE this loop;
      //   * `Ad_iw` used to be declared once outside the `d` loop and rewritten every
      //     iteration (t3a section 4.2 item 1) -> now a per-thread buffer declared INSIDE
      //     the parallel region;
      //   * `fit()` used to heap-allocate its (_Nfit,_Nfit) `g` table on every call,
      //     ~43 KB per state (t3a section 4.2 item 2) -> now a per-thread scratch buffer
      //     allocated once per thread and passed in, which removes the malloc/free churn
      //     that would otherwise dominate at 8 threads;
      //   * `_coeffs(d,:)` and `_orders(d)` are disjoint per `d`; `_iw_fit` is read-only;
      //   * cpp_dec_float is a fixed-size backend -- no allocation per arithmetic op.
      //
      // DETERMINISM: iterations are independent and nothing is reassociated, so the result
      // is bitwise identical at any thread count, not merely bit-class identical.
      [[maybe_unused]] const long nthr = utils::omp_threads_for(_dim1);
#ifdef _OPENMP
#pragma omp parallel num_threads(nthr) if (nthr > 1)
#endif
      {
        nda::array<mp_complex, 1> Ad_iw(_Nfit);
        nda::array<mp_complex, 2> g_scr(_Nfit, _Nfit);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (long d=0; d<_dim1; ++d) {
          for (long w=0; w<_Nfit; ++w)
            Ad_iw(w) = mp_complex(mp_float(A_iw(w, d).real()),
                                  mp_float(A_iw(w, d).imag()));
          // the effective order of the pade approx. can be smaller than Nfit due to truncation.
          _orders(d) = fit(Ad_iw, _iw_fit, _coeffs(d,nda::range::all), g_scr);
        }
      }
    }

    // T-3b: the evaluate family is `const`. It provably only reads _orders/_coeffs/_iw_fit
    // (t3a section 4.2 item 4), and concurrent calls on one shared kernel are exactly what
    // the threaded qp_approx / solve_qp_eqn regions do -- so pin that in the type system
    // instead of leaving it to inspection.
    template<nda::ArrayOfRank<1> w_mesh_t>
    auto evaluate(w_mesh_t &&w_mesh) const -> nda::array<ComplexType, 2> {
      long Nw = w_mesh.shape(0);
      nda::array<ComplexType, 2> A_wI(Nw, _dim1);
      for (long d1 = 0; d1 < _dim1; ++d1) {
        for (long w = 0; w < Nw; ++w) {
          mp_complex w_hpc( mp_float(w_mesh(w).real()), mp_float(w_mesh(w).imag()) );
          mp_complex Xw = _evaluate_impl(w_hpc, d1);
          A_wI(w, d1) = ComplexType(Xw.real().template convert_to<RealType>(),
                                    Xw.imag().template convert_to<RealType>());
        }
      }
      return A_wI;
    }

    template<nda::ArrayOfRank<1> w_mesh_t>
    auto evaluate(w_mesh_t &&w_mesh, long d1) const -> nda::array<ComplexType, 1> {
      long Nw = w_mesh.shape(0);
      nda::array<ComplexType, 1> A_w(Nw);
      for (long w = 0; w < Nw; ++w) {
        mp_complex w_hpc( mp_float(w_mesh(w).real()), mp_float(w_mesh(w).imag()) );
        mp_complex Xw = _evaluate_impl(w_hpc, d1);

        A_w(w) = ComplexType(Xw.real().template convert_to<RealType>(),
                             Xw.imag().template convert_to<RealType>());
      }
      return A_w;
    }

    ComplexType evaluate(ComplexType w, long d1) const {
      mp_complex w_hpc( mp_float(w.real()), mp_float(w.imag()) );
      mp_complex Xw = _evaluate_impl(w_hpc, d1);

      return ComplexType(Xw.real().template convert_to<RealType>(),
                         Xw.imag().template convert_to<RealType>());
    }

    void reset() {
      _iw_fit = nda::array<mp_complex, 1>();
      _coeffs = nda::array<mp_complex, 2>();
      _orders = nda::array<long, 1>();
    }

    pade_impl_e implementation() const { return _impl; }

  private:
    /**
     * N-point pade interpolation from Thiele's reciprocal difference method.
     * Ref: J. of Low Temp. Phys. 29, 179–192 (1977)
     * @param Ai
     * @param iw_fit
     * @return
     */
    template<nda::ArrayOfRank<1> Array_t, nda::ArrayOfRank<1> w_mesh_t>
    auto fit(Array_t &&Az, w_mesh_t &&z_fit) -> nda::array<ComplexType, 1> {
      long nfit = z_fit.shape(0);
      // g(i, j) = g_i(z_j)
      nda::array<ComplexType, 2> g(nfit, nfit);

      g(0, nda::range::all) = Az();
      for (long i = 1; i < nfit; ++i) {
        for (long j = i; j < nfit; ++j) {
          // g(i, j) = ( g(i-1,i-1) - g(i-1,j) ) / ( (z_fit(j) - z_fit(i-1)) * g(i-1, j) );
          g(i, j) = _sub_div_mul(g(i-1,i-1), g(i-1,j), z_fit(j), z_fit(i-1), g(i-1,j));
        }
      }
      // coefficients a(i) = g(i, i) = g_i(z_i)
      auto a = nda::make_regular(nda::diagonal(g));
      return a;
    }

    /**
     * T-3b: `g` is a CALLER-OWNED (_Nfit,_Nfit) scratch table rather than a local
     * allocation. It was ~43 KB of malloc/free per state (t3a section 4.2 item 2), which is
     * pure allocator churn once the enclosing `d` loop is threaded. It is fully overwritten
     * on entry (the `g() = 0` below), so reuse across calls is arithmetically identical to
     * a fresh allocation. `const` because the fit mutates no member -- only `coeffs` and
     * the caller's scratch.
     */
    template<nda::ArrayOfRank<1> Az_t, nda::ArrayOfRank<1> w_mesh_t, nda::ArrayOfRank<1> coeff_t,
             nda::ArrayOfRank<2> scratch_t>
    long fit(Az_t &&Az, w_mesh_t &&z_fit, coeff_t &&coeffs, scratch_t &&g) const {
      // g(i, j) = g_i(z_j)
      g() = mp_complex{mp_float(0), mp_float(0)};
      g(0, nda::range::all) = Az();

      long order = 1;

      for (long i=1; i<_Nfit; ++i) {
        if (_impl == pade_impl_e::updated && _norm(g(i-1, i-1)) < _tiny_pivot_tol()) break;

        for (long j=i; j<_Nfit; ++j) {
          // g(i, j) = ( g(i-1,i-1) - g(i-1,j) ) / ( (z_fit(j) - z_fit(i-1)) * g(i-1, j) );
          g(i, j) = _sub_div_mul(g(i-1,i-1), g(i-1,j), z_fit(j), z_fit(i-1), g(i-1,j));
        }
        order = i + 1;
      }
      coeffs() = nda::diagonal(g);
      return order;
    }

    mp_complex _evaluate_impl(const mp_complex& w, long d1) const {
      return (_impl == pade_impl_e::updated)? _evaluate_updated(w, d1) : _evaluate_original(w, d1);
    }

    mp_complex _evaluate_original(const mp_complex& w, long d1) const {
      long order = _orders(d1);
      if (order <= 0) return mp_complex{mp_float(0), mp_float(0)};
      if (order == 1) return _coeffs(d1, 0);

      mp_complex Xw = _mul_sub(_coeffs(d1, order-1), w, _iw_fit(order-2));
      for (long f = 1; f < order-1; ++f) {
        long idx = order - f - 1;
        Xw = _mul_sub_div_add(_coeffs(d1, idx), w, _iw_fit(idx-1), Xw);
      }
      return _div_add(_coeffs(d1, 0), Xw);
    }

    mp_complex _evaluate_updated(const mp_complex& w, long d1) const {
      long order = _orders(d1);
      if (order <= 0) return mp_complex{mp_float(0), mp_float(0)};

      mp_complex A1{mp_float(0), mp_float(0)};
      mp_complex A2 = _coeffs(d1, 0);
      mp_complex B1{mp_float(1), mp_float(0)};

      for (long i = 0; i < order - 1; ++i) {
        mp_complex scaled = _mul_sub(_coeffs(d1, i+1), w, _iw_fit(i));
        mp_complex Anew = _add(A2, _mul(scaled, A1));
        mp_complex Bnew = _add(mp_complex{mp_float(1), mp_float(0)}, _mul(scaled, B1));
        mp_complex inv_Bnew = _inverse(Bnew, "pade_t.hpp::_evaluate_new");
        A1 = _mul(A2, inv_Bnew);
        A2 = _mul(Anew, inv_Bnew);
        B1 = inv_Bnew;
      }

      return A2;
    }

    // helper functions to avoid std::complex<mp_float> arithmetics directly
    // Sadly, boost multiprecision number type is not fully compatible with std::complex<...> arithmetics
    // A * (B - C)
    inline mp_complex _mul_sub(const mp_complex& A, const mp_complex& B, const mp_complex& C) const {
      mp_float br = B.real() - C.real();
      mp_float bi = B.imag() - C.imag();
      mp_float ar = A.real();
      mp_float ai = A.imag();

      return mp_complex{
          ar * br - ai * bi,
          ar * bi + ai * br
      };
    }

    inline mp_complex _mul(const mp_complex& A, const mp_complex& B) const {
      mp_float ar = A.real();
      mp_float ai = A.imag();
      mp_float br = B.real();
      mp_float bi = B.imag();

      return mp_complex{
          ar * br - ai * bi,
          ar * bi + ai * br
      };
    }

    inline mp_complex _add(const mp_complex& A, const mp_complex& B) const {
      return mp_complex{A.real() + B.real(), A.imag() + B.imag()};
    }

    inline mp_float _norm(const mp_complex& A) const {
      return A.real() * A.real() + A.imag() * A.imag();
    }

    inline mp_complex _inverse(const mp_complex& A, const char* func_name) const {
      mp_float ar = A.real();
      mp_float ai = A.imag();
      mp_float denom = ar * ar + ai * ai;
      utils::check(denom != mp_float(0), "{}: division by zero in normalized Pad\u00e9 recurrence", func_name);
      return mp_complex{ar / denom, -ai / denom};
    }

    static mp_float _tiny_pivot_tol() {
      return mp_float("1e-20");
    }
    // A * (B - C) / (1 + D)
    inline mp_complex _mul_sub_div_add(const mp_complex& A, const mp_complex& B, const mp_complex& C, const mp_complex& D) const {
      // Numerator: A * (B - C)
      mp_float br = B.real() - C.real();
      mp_float bi = B.imag() - C.imag();
      mp_float ar = A.real();
      mp_float ai = A.imag();

      mp_float num_r = ar * br - ai * bi;
      mp_float num_i = ar * bi + ai * br;

      // Denominator: 1 + D
      mp_float dr = mp_float(1) + D.real();
      mp_float di = D.imag();
      mp_float denom = dr * dr + di * di;

      return mp_complex{
          (num_r * dr + num_i * di) / denom,
          (num_i * dr - num_r * di) / denom
      };
    }
    // A / (1 + B)
    inline mp_complex _div_add(const mp_complex& A, const mp_complex& B) const {
      // Denominator: 1 + B
      mp_float br = mp_float(1) + B.real();
      mp_float bi = B.imag();
      mp_float denom = br * br + bi * bi;

      mp_float ar = A.real();
      mp_float ai = A.imag();

      return mp_complex{
          (ar * br + ai * bi) / denom,
          (ai * br - ar * bi) / denom
      };
    }

    mp_complex _sub_div_mul(
      const mp_complex& A,
      const mp_complex& B,
      const mp_complex& C,
      const mp_complex& D,
      const mp_complex& E) const {
      // U = A - B
      mp_float ur = A.real() - B.real();
      mp_float ui = A.imag() - B.imag();

      // V = (C - D)
      mp_float vr = C.real() - D.real();
      mp_float vi = C.imag() - D.imag();

      // V * E
      mp_float er = E.real();
      mp_float ei = E.imag();

      // V * E = (vr * er - vi * ei) + i(vr * ei + vi * er)
      mp_float den_r = vr * er - vi * ei;
      mp_float den_i = vr * ei + vi * er;

      // |denominator|^2
      mp_float denom = den_r * den_r + den_i * den_i;

      // (U / (V * E))
      return mp_complex{
          (ur * den_r + ui * den_i) / denom,
          (ui * den_r - ur * den_i) / denom
      };
    }

  private:
    pade_impl_e _impl = pade_impl_e::original;
    long _dim1 = 0;
    long _Nfit = 0;
    nda::array<mp_complex, 1> _iw_fit;
    nda::array<mp_complex, 2> _coeffs;
    nda::array<long, 1> _orders;
  };
} // analyt_cont


#endif //COQUI_PADE_HPP
