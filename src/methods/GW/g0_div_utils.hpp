/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
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


#ifndef COQUI_G0_DIV_UTILS_HPP
#define COQUI_G0_DIV_UTILS_HPP

#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/h5.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/distributed_array/nda.hpp"

#include "IO/app_loggers.h"
#include "utilities/check.hpp"

#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/ERI/div_treatment_e.hpp"
#include "mean_field/MF.hpp"

namespace methods {
  namespace solvers {
    /**
     * Utility functions for GW divergence treatments
     */
    struct div_utils {

      template<nda::ArrayOfRank<2> Array_q_t>
      static auto filter_qpts(Array_q_t &&Qpts, double threshold, int order, bool two_dim=false) {
        std::vector<int> q_indices;
        std::set<double> q_set{};
        for (int i=0; i<Qpts.shape(0); ++i) {
          auto qpt = Qpts(i,nda::range::all);
          double norm = std::sqrt( qpt(0)*qpt(0) + qpt(1)*qpt(1) + qpt(2)*qpt(2) );
          norm = std::pow(norm, order);
          if (norm > 0.0 && norm <= threshold && (!two_dim or qpt(2)==0.0)) {
            auto insert_pair = q_set.insert(norm);
            if (insert_pair.second) {
              q_indices.push_back(i);
            }
          }
        }
        return q_indices;
      }

      /**
       * Estimate the inverse of symmetric dielectric function in plane-wave basis at G=G'=0
       * on the imaginary-time axis
       * @tparam extrapolate - perform extrapolation or not
       * @param dW_wqPQ - [INPUT] screened interaction in the THC auxiliary basis on
       *                          the imaginary-time or Matsubara frequency axis.
       * @param thc - [INPUT] THC-ERI
       * @param mf - [INPUT] mean-field object
       * @return - inverse of symmetric dielectric function at finite q and q = 0
       */
      template<nda::MemoryArrayOfRank<4> Array_w_t, THC_ERI thc_t, typename communicator_t>
      static auto eps_inv_head_t(memory::darray_t<Array_w_t, communicator_t> &dW_tqPQ, thc_t &thc,
                                 mf::MF &mf, const imag_axes_ft::IAFT *ft,
                                 div_treatment_e div_treatment=gygi_extrplt,
                                 bool poly_in_q2=false, double q_max=0.8, int fit_order=2)
      -> std::tuple<nda::array<ComplexType, 2>, nda::array<ComplexType, 1> > {

        auto [eps_inv_t, Q_abs, Q_abs2] = eval_eps_inv_q(dW_tqPQ, thc, mf);
        auto [nts, nqpts_ibz] = eps_inv_t.shape();

        if (thc.MF()->nqpts_ibz() == 1 and div_treatment != ignore_g0) {
          app_log(2, "eps_inv_head_t: nqpts_ibz == 1 while div_treatment != ignore. "
                     "Will take div_treatment = ignore_g0 anyway!");
          div_treatment = ignore_g0;
        }

        nda::array<ComplexType, 1> eps_inv_q0_t(nts);
        if (div_treatment==gygi_extrplt or div_treatment==gygi_extrplt_2d) {
          utils::check(mf.nkpts_ibz() > 1, "eps_inv_head_t: nkpts_ibz <= 1 while div_treatment==gygi_extrplt");
          bool two_dim = (div_treatment==gygi_extrplt_2d)? true : false;
          // extrapolation happens at frequency axis
          long nw = (ft->nw_b()%2==0)? ft->nw_b()/2 : ft->nw_b()/2 + 1;
          nda::array<ComplexType, 2> eps_inv_w(nw, nqpts_ibz);
          ft->tau_to_w_PHsym(eps_inv_t, eps_inv_w);

          auto q_indices = (poly_in_q2)? filter_qpts(mf.Qpts_ibz(), q_max, 2, two_dim)
                           : filter_qpts(mf.Qpts_ibz(), q_max, 1, two_dim);
          nda::array<ComplexType, 1> Q_filtered(q_indices.size());
          nda::array<ComplexType, 2> eps_inv_filtered(eps_inv_w.shape(0), q_indices.size());
          for (size_t i = 0; i < q_indices.size(); ++i) {
            Q_filtered(i) = (poly_in_q2)? Q_abs2( q_indices[i] ) : Q_abs( q_indices[i] );
            eps_inv_filtered(nda::range::all, i) = eps_inv_w(nda::range::all, q_indices[i] );
          }
          if ( q_indices.size() < 3) {
            // Choose the closest point to the gamma instead
            eps_inv_q0_t = eps_inv_t(nda::range::all, q_indices[0]);
          } else {
            if (two_dim)
              app_log(2, "  Extrapolate head of the inverse of the dielectric function from {} q-points on the xy-plane", q_indices.size());
            else
              app_log(2, "  Extrapolate head of the inverse of the dielectric function from {} q-points", q_indices.size());
            nda::array<ComplexType, 1> eps_inv_q0_w(nw);
            for (int n = 0; n < nw; ++n) {
              eps_inv_q0_w(n) = extrapolate_to_q0(Q_filtered, eps_inv_filtered(n, nda::range::all), fit_order,
                                                  (dW_tqPQ.communicator()->root() and n==0)? true : false);
            }
            auto eps_inv_q0_w_2D = nda::reshape(eps_inv_q0_w, std::array<long,2>{nw, 1});
            auto eps_inv_q0_t_2D = nda::reshape(eps_inv_q0_t, std::array<long,2>{nts,1});
            ft->w_to_tau_PHsym(eps_inv_q0_w_2D, eps_inv_q0_t_2D);
          }
        } else {
          // Choose the closest point to the gamma
          int smallest_idx = find_smallest_qabs(mf.Qpts_ibz(), false);
          eps_inv_q0_t = eps_inv_t(nda::range::all, smallest_idx);
        }

        return std::make_tuple(std::move(eps_inv_t), std::move(eps_inv_q0_t));
      }

      /**
       * Estimate the inverse of symmetric dielectric function in plane-wave basis at G=G'=0
       * on the Matsubara frequency axis
       * @tparam extrapolate - perform extrapolation or not
       * @param dW_wqPQ - [INPUT] screened interaction in the THC auxiliary basis on
       *                          the imaginary-time or Matsubara frequency axis.
       * @param thc - [INPUT] THC-ERI
       * @param mf - [INPUT] mean-field object
       * @return - inverse of symmetric dielectric function at finite q and q = 0
       */
      template<nda::MemoryArrayOfRank<4> Array_w_t, THC_ERI thc_t, typename communicator_t>
      static auto eps_inv_head_w(memory::darray_t<Array_w_t, communicator_t> &dW_wqPQ, thc_t &thc,
                                 mf::MF &mf, div_treatment_e div_treatment=gygi_extrplt,
                                 bool poly_in_q2=false, double q_max=0.8, int fit_order=2)
      -> std::tuple<nda::array<ComplexType, 2>, nda::array<ComplexType, 1> > {

        auto [eps_inv_w, Q_abs, Q_abs2] = eval_eps_inv_q(dW_wqPQ, thc, mf);
        auto [nw, nqpts_ibz] = eps_inv_w.shape();

        if (thc.MF()->nqpts_ibz() == 1 and div_treatment != ignore_g0) {
          app_log(2, "eps_inv_head_w: nqpts_ibz == 1 while div_treatment != ignore. "
                     "Will take div_treatment = ignore_g0 anyway!");
          div_treatment = ignore_g0;
        }

        nda::array<ComplexType, 1> eps_inv_q0_w(nw);
        if (div_treatment==gygi_extrplt or div_treatment==gygi_extrplt_2d) {
          utils::check(mf.nkpts_ibz() > 1, "eps_inv_head_w: nkpts_ibz <= 1 while div_treatment==gygi_extrplt");
          bool two_dim = (div_treatment==gygi_extrplt_2d)? true : false;
          // extrapolation happens at frequency axis
          auto q_indices = (poly_in_q2)? filter_qpts(mf.Qpts_ibz(), q_max, 2, two_dim) : filter_qpts(mf.Qpts_ibz(), q_max, 1, two_dim);
          nda::array<ComplexType, 1> Q_filter(q_indices.size());
          nda::array<ComplexType, 2> eps_inv_filter(eps_inv_w.shape(0), q_indices.size());
          for (size_t i = 0; i < q_indices.size(); ++i) {
            Q_filter(i) = (poly_in_q2)? Q_abs2( q_indices[i] ) : Q_abs( q_indices[i] );
            eps_inv_filter(nda::range::all, i) = eps_inv_w(nda::range::all, q_indices[i] );
          }
          if ( q_indices.size() < 3) {
            // Choose the closest point to the gamma instead
            eps_inv_q0_w = eps_inv_w(nda::range::all, q_indices[0]);
          } else {
            if (two_dim)
              app_log(2, "  Extrapolate head of the inverse of the dielectric function from {} q-points on the xy-plane", q_indices.size());
            else
              app_log(2, "  Extrapolate head of the inverse of the dielectric function from {} q-points", q_indices.size());
            for (int n = 0; n < nw; ++n) {
              eps_inv_q0_w(n) = extrapolate_to_q0(Q_filter, eps_inv_filter(n, nda::range::all), fit_order,
                                                  (dW_wqPQ.communicator()->root() and n==0)? true : false);
            }
          }
        } else {
          // Choose the closest point to the gamma
          int smallest_idx = find_smallest_qabs(mf.Qpts_ibz(), false);
          eps_inv_q0_w = eps_inv_w(nda::range::all, smallest_idx);
        }

        return std::make_tuple(std::move(eps_inv_w), std::move(eps_inv_q0_w));
      }

      /**
       * Evaluate \epsilon^{q,-1}_{G=0,G'=0} - 1
       * @param dW_tqPQ - [INPUT] screened interaction in the thc product basis
       * @param thc - [INPUT] thc-eri instance
       * @param mf - [INPUT] mean-field instance
       * @return
       */
      template<nda::MemoryArrayOfRank<4> Array_w_t, THC_ERI thc_t, typename communicator_t>
      static auto eval_eps_inv_q(memory::darray_t<Array_w_t, communicator_t> &dW_tqPQ,
                                 thc_t &thc, mf::MF &mf) {
        auto [nt_loc, nq_loc, NP_loc, NQ_loc] = dW_tqPQ.local_shape();
        auto t_rng = dW_tqPQ.local_range(0);
        auto qpt_rng = dW_tqPQ.local_range(1);
        auto P_rng = dW_tqPQ.local_range(2);
        auto Q_rng = dW_tqPQ.local_range(3);
        auto [nts, nqpts_ibz, NP, NQ] = dW_tqPQ.global_shape();

        nda::array<ComplexType, 1> Q_abs2(nqpts_ibz);
        nda::array<ComplexType, 1> Q_abs(nqpts_ibz);
        for (int iq = 0; iq < nqpts_ibz; ++iq) {
          auto qpt = mf.Qpts_ibz(iq);
          Q_abs2(iq) = ComplexType( qpt(0)*qpt(0) + qpt(1)*qpt(1) + qpt(2)*qpt(2) );
          Q_abs(iq) = ComplexType( std::sqrt( qpt(0)*qpt(0) + qpt(1)*qpt(1) + qpt(2)*qpt(2) ) );
        }

        nda::array<ComplexType, 2> eps_inv_t(nts, nqpts_ibz);
        nda::array<ComplexType, 1> Chi_bar_Q_conj(NQ_loc);
        nda::array<ComplexType, 1> buffer_P(NP_loc);
        const double fpi = 4.0*3.14159265358979323846;
        auto Chi_bar_qu = thc.basis_bar_head();
        auto Wloc = dW_tqPQ.local();
        for (auto [iq, q] : itertools::enumerate(qpt_rng)) {
          Chi_bar_Q_conj = nda::conj(Chi_bar_qu(q, Q_rng));
          double factor = (Q_abs2(q).real() / fpi) * mf.volume();
          for (auto [it, t] : itertools::enumerate(t_rng)) {
            nda::blas::gemv(Wloc(it, iq, nda::ellipsis{}), Chi_bar_Q_conj, buffer_P);
            eps_inv_t(t, q) += factor * nda::blas::dot(Chi_bar_qu(q, P_rng), buffer_P);
          }
        }
        dW_tqPQ.communicator()->all_reduce_in_place_n(eps_inv_t.data(), eps_inv_t.size(), std::plus<>{});

        return std::make_tuple(eps_inv_t, Q_abs, Q_abs2);
      }

      /**
       * Evaluate the head of a matrix, M_{G=0,G'=0}(t,q), for an arbitrary tensor, M(t,q,P,Q), in the thc product basis. 
       * @param dM_tqPQ - [INPUT] input tensor in the thc product basis
       * @param thc - [INPUT] thc-eri instance
       * @param mf - [INPUT] mean-field instance
       * @param bar_basis - [INPUT] bool (default = false). If true, use the "bar" basis, otherwise use the direct basis.
       * @return
       */
      template<nda::MemoryArrayOfRank<4> Array_w_t, THC_ERI thc_t, typename communicator_t>
      static auto head_from_prod_basis(memory::darray_t<Array_w_t, communicator_t> &dM_tqPQ,
                                 thc_t &thc, bool bar_basis = false) {
        auto [nt_loc, nq_loc, NP_loc, NQ_loc] = dM_tqPQ.local_shape();
        auto t_rng = dM_tqPQ.local_range(0);
        auto qpt_rng = dM_tqPQ.local_range(1);
        auto P_rng = dM_tqPQ.local_range(2);
        auto Q_rng = dM_tqPQ.local_range(3);
        auto [nts, nqpts_ibz, NP, NQ] = dM_tqPQ.global_shape();

        nda::array<ComplexType, 2> m_t(nts, nqpts_ibz);
        nda::array<ComplexType, 1> buffer_P(NP_loc);
        auto Mloc = dM_tqPQ.local();
        if(bar_basis) {
          // M(t,q) = sum_PQ B_bar(q,P) * M(t,q,P,Q) conj(B_bar(q,Q))
          auto Chi_bar_qu = thc.basis_bar_head();
          nda::array<ComplexType, 1> Chi_bar_Q_conj(NQ_loc);
          for (auto [iq, q] : itertools::enumerate(qpt_rng)) {
            Chi_bar_Q_conj = nda::conj(Chi_bar_qu(q, Q_rng));
            for (auto [it, t] : itertools::enumerate(t_rng)) {
              nda::blas::gemv(Mloc(it, iq, nda::ellipsis{}), Chi_bar_Q_conj, buffer_P);
              m_t(t, q) += nda::blas::dot(Chi_bar_qu(q, P_rng), buffer_P);
            }
          }
        } else {
          // M(t,q) = sum_PQ conj(B(q,P)) * M(t,q,P,Q) B(q,Q)
          auto Chi_qu = thc.basis_head();
          for (auto [iq, q] : itertools::enumerate(qpt_rng)) {
            for (auto [it, t] : itertools::enumerate(t_rng)) {
              nda::blas::gemv(Mloc(it, iq, nda::ellipsis{}), Chi_qu(q, Q_rng), buffer_P);
              m_t(t, q) += nda::blas::dotc(Chi_qu(q, P_rng), buffer_P);
            }
          }
        }
        dM_tqPQ.communicator()->all_reduce_in_place_n(m_t.data(), m_t.size(), std::plus<>{});

        return m_t;
      }

      template<nda::ArrayOfRank<2> Array_q_t>
      static int find_smallest_qabs(Array_q_t &&Qpts, bool two_dim=false) {
        if (Qpts.shape(0) == 1) return 0;

        double min = -1;
        int idx = -1;
        for (int i=0; i<Qpts.shape(0); ++i) {
          auto qpt = Qpts(i,nda::range::all);
          double norm = std::sqrt( qpt(0)*qpt(0) + qpt(1)*qpt(1) + qpt(2)*qpt(2) );
          if (norm > 0.0 and min==-1 and (!two_dim or qpt(2)!=0.0)) {
            min = norm;
            idx = i;
          }
          else if (norm > 0.0 and norm<min and (!two_dim or qpt(2)==0.0)) {
            min = norm;
            idx = i;
          }
        }
        return idx;
      }

      template<nda::ArrayOfRank<1> Array_q_t, nda::ArrayOfRank<1> Array_eps_t>
      static auto extrapolate_to_q0(const Array_q_t &Qpts_abs2, const Array_eps_t &eps_inv_n, int fit_order,
                                    bool print=false) {

        long num_sample = Qpts_abs2.shape(0);
        nda::matrix<ComplexType> A(num_sample, fit_order+1);
        for (int i = 0; i < num_sample; ++i) {
          A(i, 0) = 1.0;
          for (int j = 1; j <= fit_order; ++j) {
            A(i, j) = std::pow(Qpts_abs2(i), j);
          }
        }

        auto AT = nda::make_regular(nda::transpose(A));
        nda::array<ComplexType, 1> AT_b(fit_order+1);
        nda::blas::gemv(AT, eps_inv_n, AT_b);
        nda::matrix<ComplexType> ATA_inv(fit_order+1, fit_order+1);
        nda::blas::gemm(AT, A, ATA_inv);
        ATA_inv = nda::inverse(ATA_inv);

        nda::array<ComplexType, 1> x(fit_order+1);
        nda::blas::gemv(ATA_inv, AT_b, x);

        if (print) {
          for (size_t n=0; n<=fit_order; ++n)
            app_log(2, "    x({}) = {}", n, x(n));
          app_log(2, "");
        }

        return x(0);
      }

    }; // div_utils
  } // solvers
} // methods



#endif //COQUI_G0_DIV_UTILS_HPP
