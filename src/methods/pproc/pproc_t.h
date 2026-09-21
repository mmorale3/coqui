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


#ifndef COQUI_PPROC_T_HPP
#define COQUI_PPROC_T_HPP

#include <array>
#include <tuple>
#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "h5/h5.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "IO/app_loggers.h"
#include "utilities/Timer.hpp"
#include "utilities/proc_grid_partition.hpp"

#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "utilities/mpi_context.h"
#include "numerics/ac/AC_t.hpp"
#include "numerics/imag_axes_ft/iaft_utils.hpp"
#include "methods/SCF/qp_params_t.h"
#include "methods/SCF/qp_solvers.hpp"

namespace methods {
  namespace mpi3 = boost::mpi3;

  class projector_t;   // methods/embedding/projector_t.h (the Wannier projector of qp_bands_on_mesh)

  // TODO This should not be a class! Separate these into free functions.
  // TODO Useful features:
  //      1. band-gap estimator
  /**
   * A proxy for different post-processing steps after a mbpt calculation.
   * The mbpt solution is given by reading the bdft h5 output file: outdir/prefix.mbpt.h5
   */
  class pproc_t {
  public:
    pproc_t(utils::mpi_context_t<mpi3::communicator> &context, std::string prefix, std::string outdir):
        _context(context), _scf_output(outdir+"/"+prefix) {

      for (auto& v: {"READ", "WRITE", "AC"}) {
        _Timer.add(v);
      }

    }

    /**
     * Perform analytical continuation
     * @param mf - [INPUT] a mean-field instance for metadata of the system
     * @param ac_context - [INPUT] parameters for ac
     * @param dataset - [INPUT] dataset for ac
     */
    void analyt_cont(mf::MF &mf, analyt_cont::ac_context_t &ac_context, std::string dataset="G_tskij");
    /**
     * Perform Wannier interpolation to the mbpt solutions on the provided k-points
     * @param mf - [INPUT] a mean-field instance for all the metadata of the system
     * @param project_file - [INPUT] a h5 file which stores the projection matrices and the target k-points
     * @param target - [INPUT] type of the mbpt calculation: quasiparticle or dyson
     */
    void wannier_interpolation(mf::MF &mf, ptree const& pt, std::string project_file, std::string target,
                               std::string grp_name="scf", long iter=-1, 
                               bool translate_home_cell=false);
    /**
     * Wannier interpolation plus analytical continuation for spectral functions on the provided k-points.
     * Also computes quasiparticle energies on the IBZ k-mesh from the dynamic self-energy and
     * Wannier-interpolates them along the k-path.
     * (Only for dyson-type calculation)
     * @param mf - [INPUT] a mean-field instance for all the metadata of the system
     * @param project_file - [INPUT] a h5 file which stores the projection matrices and the target k-points
     * @param ac_params - [INPUT] parameters for ac
     * @param qp_params - [INPUT] parameters for quasiparticle equation solver
     */
    void spectral_interpolation(mf::MF &mf, ptree const& pt, std::string project_file,
                                analyt_cont::ac_context_t &ac_params,
                                std::string grp_name="scf", long iter=-1, bool translate_home_cell=false);

    void local_density_of_state(mf::MF &mf, std::string project_file, analyt_cont::ac_context_t &ac_params,
                                std::string grp_name="scf", long iter=-1, bool translate_home_cell=false);
            
    /**
     * Compute quasiparticle energies on the IBZ k-mesh by solving the QP equation
     * E_a = F_aa + Re[Sigma_aa(E_a - mu)] for each (spin, k, band).
     * Writes E_ska to {grp_name}/iter{N}/qp_approx/E_ska in the checkpoint file.
     */
    void compute_qp_on_ibz_kmesh(mf::MF &mf, const qp_params_t &qp_params,
                                std::string grp_name="scf", long iter=-1);

    /**
     * scGW-tilde increment C3 (notes/scgwt_implementation_plan.md; the T-c probe):
     * covariant-velocity dielectric readout on a STORED checkpoint. Loads
     * (F, Sigma(tau), mu) of the requested iteration, rebuilds G by Dyson, builds the
     * CVV R-space store, evaluates the head tensor and the SUBTRACTED head coefficient
     * Phead_ab(inu) = [Pi^jj(inu) - Pi^jj(0)]/(inu)^2 (cvv_detail::head_subtract), and
     * reports eps_inf(q^) = 1 - 4 pi q^ q^ : Phead(inu = 0) per cartesian direction --
     * an explicit O(q^2) coefficient, NO q -> 0 extrapolation, so the
     * stored-vs-quadratic convention split does not arise (gate C3-a / PDF G-b).
     * Results are written to {grp_name}/iter{N}/cvv_eps in the checkpoint.
     */
    void cvv_eps(mf::MF &mf, ptree const& pt, std::string grp_name="scf", long iter=-1);

    /**
     * P24 / G31 (notes/vertex_perf_plan.md): Wannier interpolation of the quasiparticle Hamiltonian
     * of {grp_name}/iter{N} (qp_approx/Heff_skij when present, Heff_skij otherwise) onto a UNIFORM
     * fine k mesh -- all n1*n2*n3 points of the mesh in crystal coordinates, not the IBZ -- with the
     * "quasiparticle" machinery of wannier_interpolation (IBZ -> full-BZ unfold, projector downfold,
     * k -> R on the Wigner-Seitz R grid of the mean-field mesh, R -> k on the fine mesh), followed by
     * a per-k Hermitization and diagonalization (interpolate_qp_bands_on_mesh). Unlike the band-path
     * route, the imaginary part of H(R) is KEPT, so the interpolation is exact on the mean-field mesh:
     * at a mesh k the interpolated bands are the eigenvalues of the downfolded Heff(k) (the identity
     * gate of test_methods_pproc). Consumed by the qp_gaps post-processing (qp_gaps_interp = true).
     * Writes nothing to the checkpoint.
     * @param mf - [INPUT] mean-field instance (mesh, lattice, symmetry maps)
     * @param project_file - [INPUT] h5 file with the Wannier projection matrices (dft_input/proj_mat)
     * @param mesh - [INPUT] the fine mesh (n1, n2, n3)
     * @return (kpts_crys (nk_fine, 3), E_ska (ns, nk_fine, nImpOrbs) in Ha, ascending per k)
     */
    auto qp_bands_on_mesh(mf::MF &mf, std::string project_file, std::array<long, 3> const& mesh,
                          std::string grp_name="scf", long iter=-1, bool translate_home_cell=false)
      -> std::tuple<nda::array<double, 2>, nda::array<RealType, 3>>;

    /**
     * The interpolation kernel of qp_bands_on_mesh, checkpoint-free (the unit-test entry).
     * @param Heff_skij_full - [INPUT] the QP Hamiltonian on the FULL BZ in the primary (DFT band)
     *                         basis, (ns, nkpts, nbnd, nbnd), k ordered as mf.kpts()
     * @param Rpts_idx / Rpts_weights - [INPUT] the R grid (integer lattice coordinates) and its
     *                         Wigner-Seitz degeneracies (utils::WS_rgrid or dft_input/r_vector)
     * @param mesh - [INPUT] the fine mesh (n1, n2, n3); k index = (i*n2 + j)*n3 + l, k = (i/n1, j/n2, l/n3)
     * @return (kpts_crys (nk_fine, 3), E_ska (ns, nk_fine, nImpOrbs) in Ha, ascending per k)
     */
    static auto interpolate_qp_bands_on_mesh(utils::mpi_context_t<mpi3::communicator> &context, mf::MF &mf,
                                             projector_t const& proj,
                                             nda::array_view<ComplexType, 4> Heff_skij_full,
                                             nda::array<long, 2> const& Rpts_idx,
                                             nda::array<long, 1> const& Rpts_weights,
                                             std::array<long, 3> const& mesh)
      -> std::tuple<nda::array<double, 2>, nda::array<RealType, 3>>;

  private:
    template<nda::ArrayOfRank<4> local_Array_4D_t, typename communicator_t>
    void read_scf_dataset(std::string dataset,
                          memory::darray_t<local_Array_4D_t, communicator_t> &A_tski);

    /* Read full dataset without extracting a diagonal
     * from a hdf5 group.
     * The function is supposed to work for "scf" and "system" groups 
     * and with 5D and 4D respectively.
     */
    template<nda::MemoryArray local_Array_t, typename communicator_t>
    void read_scf_dataset_full(std::string dataset, std::string group,
                                 memory::darray_t<local_Array_t, communicator_t> &A);

    template<nda::ArrayOfRank<4> local_Array_4D_t, typename communicator_t>
    void dump_ac_output(nda::array<ComplexType, 1> &w_mesh,
                        memory::darray_t<local_Array_4D_t, communicator_t> &dA_out,
                        nda::array<ComplexType, 1> &iw_mesh,
                        memory::darray_t<local_Array_4D_t, communicator_t> &dA_in,
                        std::string dataset, std::string grp_name="scf", int iter=-1);

    template<nda::MemoryArray local_Array_t>
    void dump_ac_output(nda::array<ComplexType, 1> &w_mesh,
                        local_Array_t &A_out,
                        nda::array<ComplexType, 1> &iw_mesh,
                        local_Array_t& A_in,
                        std::string dataset, std::string grp_name="scf", int iter=-1);

    template<nda::ArrayOfRank<5> local_Array_5D_t, nda::ArrayOfRank<4> local_Array_4D_t, typename communicator_t>
    auto evaluate_GS_diag(memory::darray_t<local_Array_5D_t, communicator_t> & dG_tau_skij,
                          memory::darray_t<local_Array_4D_t, communicator_t> & dS_skij)
      -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>;

  private:
    utils::mpi_context_t<mpi3::communicator> &_context;
    std::string _scf_output;
    utils::TimerManager _Timer;
  };
} // methods


#endif //COQUI_PPROC_T_HPP
