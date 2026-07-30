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


#ifndef COQUI_DIIS_T_HPP
#define COQUI_DIIS_T_HPP

#include "configuration.hpp"
#include "utilities/h5_background_writer.hpp"
#include "utilities/check.hpp"

#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "numerics/iter_scf/iter_scf_type_e.hpp"

#include "numerics/iter_scf/diis/vspace.h"
#include "numerics/iter_scf/diis/vspace_fock_sigma.hpp"

#include "numerics/iter_scf/diis/state.h"
#include "numerics/iter_scf/diis/com_diis_residual.h"

#include "numerics/iter_scf/diis/diis_alg.hpp"
#include "numerics/iter_scf/damp/damp_t.hpp"

namespace iter_scf {
  /**
   * Simple class connecting an abstract DIIS with specific interface requirements
   * serving as a driver for DIIS algorithm
   */
  struct diis_t {
    using Array_4D = nda::array<ComplexType,4>;
    using Array_5D = nda::array<ComplexType,5>;

    static constexpr iter_alg_e iter_alg = DIIS;
    static constexpr iter_alg_e get_iter_alg() { return iter_alg; }

  public:
    diis_t() = default;
    diis_t(double mixing_, size_t max_subsp_size_, size_t warmup_iter_):
        mixing(mixing_), max_subsp_size(max_subsp_size_), warmup_iter(warmup_iter_) {};

    diis_t(const diis_t& other) = default;
    diis_t(diis_t&& other) = default;
    diis_t& operator=(const diis_t& other) = default;
    diis_t& operator=(diis_t&& other) = default;

    ~diis_t(){}

    template<nda::MemoryArrayOfRank<4> F_t, nda::MemoryArrayOfRank<5> Sigma_t,
        nda::MemoryArrayOfRank<4> S_t, nda::MemoryArrayOfRank<4> H0_t>
    void initialize(F_t &&F, Sigma_t &&Sigma, double mu, S_t &&S, H0_t &&H0,
                    const imag_axes_ft::IAFT *FT, std::string mbpt_output_) {
        warmup_count = 0; // reset warmup counter
        mbpt_output = mbpt_output_;
        // Initialize the extrapolated state using the current Fock and Sigma
        extrapolated_state.initialize(FockSigma(F, Sigma, mu));
        // initialize the vector space used to extrapolation
        x_vsp.initialize("diis_vectors.h5");
        // initialize the vector space for residuals
        res_vsp.initialize("diis_residuals.h5");
        comFS_residual.initialize(&extrapolated_state, S, H0, FT, mbpt_output);
        // providing non-owning pointers to DIIS kernel as well as the starting state
        d_alg.init(&extrapolated_state, &comFS_residual, &x_vsp, &res_vsp,
                   max_subsp_size, true, FockSigma(F, Sigma, mu));
        initialized = true;
    }

    /**
     * Present only so that iter_scf_t's variant visit compiles for damping's
     * in-memory fast path; DIIS needs the whole iterate history rather than just
     * the previous one, so it always goes through the checkpoint. Never reached:
     * damping_impl only takes that path when the algorithm is simple damping.
     */
    template<nda::MemoryArray Array_H_t, nda::MemoryArray Array_P_t>
    double solve(Array_H_t &&, Array_P_t const&) {
      utils::check(false, "diis_t::solve: mixing against an in-memory previous iterate is "
                          "only implemented for simple damping.");
      return 0.0;
    }

    template<nda::MemoryArray Array_H_t>
    double solve(Array_H_t &&H, std::string dataset, h5::group &grp, long iter) {
        (void) H; (void) dataset; (void) grp; (void) iter;
        APP_ABORT("This use case for DIIS is not ready yet");
        return 0.0; // to suppress compile warnings
    }
   

    template<nda::MemoryArray Array_4D_t, nda::MemoryArray Array_5D_t>
    std::array<double, 2> solve(
        Array_4D_t &&F, std::string dataset_F, Array_5D_t &&Sigma, std::string dataset_Sigma,
        h5::group &scf_grp, long iter) {
        utils::check(initialized, "DIIS must be initialized before solving");
        warmup_count += 1;
        if (x_vsp.size() == 1 || warmup_count <= warmup_iter) {
            app_log(2, "DIIS: Warmup iteration {}/{}. Simple damping will be executed instead.\n",
                    warmup_count, warmup_iter);
            damp_t damp(mixing);

            // grow x_vsp only if x_vsp.size() <= 1, otherwise grow both x_vsp and res_vsp
            d_alg.grow_xvsp_only = (x_vsp.size() <= 1);
            d_alg.extrap = false;
            utils::check(d_alg.next_step(FockSigma(F, Sigma, get_mu()))==0,
                         "DIIS: Unexpected extrapolation while DIIS algorithm is only growing the subspace");
            app_log(4, "DIIS: DIIS vector space size: {}", x_vsp.size());
            app_log(4, "DIIS: DIIS residual space size: {}\n", res_vsp.size());

            // damping instead
            return damp.solve(F, dataset_F, Sigma, dataset_Sigma, scf_grp, iter);

         } else {
            // DO DIIS
            d_alg.extrap = true;
            d_alg.grow_xvsp_only = false;
            FockSigma fs(F, Sigma, get_mu());
            int is_extrapolated = d_alg.next_step(FockSigma(F, Sigma, get_mu()));
            if(is_extrapolated != 0) {
                auto Fdiff = nda::make_regular(F - d_alg.get_extrapolated_state().get_fock());
                auto Sdiff = nda::make_regular(Sigma - d_alg.get_extrapolated_state().get_sigma());
                auto Fmax_iter = max_element(Fdiff.data(), Fdiff.data()+Fdiff.size(),
                                    [](auto a, auto b) { return std::abs(a) < std::abs(b); });
                auto Smax_iter = max_element(Sdiff.data(), Sdiff.data()+Sdiff.size(),
                                  [](auto a, auto b) { return std::abs(a) < std::abs(b); });
                F     = d_alg.get_extrapolated_state().get_fock();
                Sigma = d_alg.get_extrapolated_state().get_sigma();

                return std::array<double, 2>{std::abs(*Fmax_iter), std::abs(*Smax_iter)};

            } else {
                // No DIIS extrapolation has been applied
                app_log(2, "DIIS: Performing simple damping instead.\n");
                damp_t damp(mixing);

                return damp.solve(F, dataset_F, Sigma, dataset_Sigma, scf_grp, iter);
            }
        }
    }

    // TODO: update if other DIIS versions will be plugged in
    void metadata_log() const {
      app_log(2, "\nIterative algorithm for SCF");
      app_log(2, "-----------------------------");
      app_log(2, "  * algorithm: frequency-dependent commutator DIIS\n"
                 "               P. Pokhilko, C.-N. Yeh, D. Zgid. J. Chem. Phys., 2022, 156, 094101\n"
                 "               https://doi.org/10.1063/5.0082586");
      app_log(2, "  * DIIS parameters: ");
      app_log(2, "    mixing            = {}", mixing);
      app_log(2, "    max subspace size = {}", max_subsp_size);
      app_log(2, "    warmup iteration  = {}", warmup_iter);
      app_log(2, "    checkpoint output = {}\n", mbpt_output);
    }

  public:
    double mixing = 0.2;
    size_t max_subsp_size = 5;
    size_t warmup_iter = 5;
    size_t warmup_count = 0;
    bool initialized = false;
    
  private:
    VSpace<FockSigma> x_vsp;                 // vector space of Fock-self-energy vectors
    VSpace<FockSigma> res_vsp;               // vector space of residuals-commutators
    opt_state<FockSigma> extrapolated_state; // extrapolated DIIS state
    com_diis_residual comFS_residual;        // residual kernel

    diis_alg<FockSigma> d_alg;               // DIIS kernel

    std::string mbpt_output;

    // Read mu from the checkpoint file
    double get_mu() {
        long iter_from_file;
        std::string filename = mbpt_output + ".mbpt.h5";
        utils::h5_quiesce();  // see h5_background_writer.hpp
        h5::file file(filename, 'r');
        h5::group grp(file);
        utils::check(grp.has_subgroup("scf"), "Simulation HDF5 file does not have an scf group");
        auto scf_grp = grp.open_group("scf");
        h5::h5_read(scf_grp, "final_iter", iter_from_file);
        auto iter_grp = scf_grp.open_group("iter"+std::to_string(iter_from_file));
        double mu;
        h5::h5_read(iter_grp, "mu", mu);
        return mu;
    }


  };
} // iter_scf

#endif //COQUI_DIIS_T_HPP
