/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_REAL_AXIS_MB_STATE_HPP
#define COQUI_REAL_AXIS_REAL_AXIS_MB_STATE_HPP

#include <optional>
#include <memory>
#include <string>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/mpi_context.h"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/nda_utils.hpp"
#include "numerics/shared_array/nda.hpp"
#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_proc_grid.hpp"

namespace methods {
namespace real_axis {

/**
 * State container for the finite-temperature real-axis ISDF/THC GW solver.
 *
 * Mirrors the pattern of methods::MBState (the imaginary-axis container) but
 * keeps real-axis spectral and Re/Im parts in dedicated fields. The container
 * holds:
 *
 *   - finite-T parameters (beta, mu_chem) -- first-class members, never optional;
 *   - the frequency/time grids (via real_freq_grid_t pointer);
 *   - lattice-resolved spectral, polarization, screened-interaction, and
 *     self-energy fields, each stored as Im and Re separately for direct
 *     visualization and to make the causality-projection step trivial.
 *
 * Shape conventions (C-layout, leading axes first):
 *
 *   A_wskij        : (N_w,  ns, nkpts_ibz, nbnd, nbnd)   spectral function
 *   ImPi_qPQO      : (nqpts_ibz, Naux, Naux, N_Omega)    Im polarization
 *   RePi_qPQO      : (nqpts_ibz, Naux, Naux, N_Omega)    Re polarization
 *   ImW_qPQO       : (nqpts_ibz, Naux, Naux, N_Omega)    Im screened interaction
 *   ReW_qPQO       : (nqpts_ibz, Naux, Naux, N_Omega)    Re screened interaction
 *   ImSigma_wskij  : (N_w,  ns, nkpts_ibz, nbnd, nbnd)   Im correlation self-energy
 *   ReSigma_wskij  : (N_w,  ns, nkpts_ibz, nbnd, nbnd)   Re correlation self-energy
 *   Sigma_x_skij   : (ns, nkpts_ibz, nbnd, nbnd)         static exchange self-energy
 *
 * Bosonic (q,Naux,Naux,N_Omega) layout puts Omega innermost: matches the
 * FINUFFT-batched kernels (cross_correlate / convolve / Hilbert) that operate
 * on contiguous (P,Q,Omega) slices.
 *
 * Bosonic objects live on the Omega>=0 half-grid (see notes section on BZ
 * symmetries: bosonic Omega->-Omega symmetry is exact at any temperature).
 * Fermionic objects span the full window [-w_max, w_max] because at finite T
 * the Fermi factor breaks any reflection symmetry around mu_chem.
 *
 * SCF iteration counter and a tag string for checkpoint files are kept for
 * compatibility with the existing solver-driver patterns.
 *
 * The container is constructed empty (only beta, mu_chem, and the grid set);
 * individual arrays are allocated lazily by the solver as it produces them.
 */
struct real_axis_mb_state_t {

  using mpi_context_t = utils::mpi_context_t<boost::mpi3::communicator>;
  using bosonic_dArray_t = memory::darray_t<
      memory::array<HOST_MEMORY, ComplexType, 4>,
      boost::mpi3::communicator>;
  template<nda::MemoryArray Array_base_t>
  using sArray_t = math::shm::shared_array<Array_base_t>;

  // Finite-temperature parameters. Mandatory, not optional.
  double beta    = 0.0;
  double mu_chem = 0.0;

  // Reference to the frequency/time grid (non-owning).
  real_freq_grid_t const* grid = nullptr;

  // MPI context (mirrors methods::MBState::mpi). Distributed solver classes
  // read state.mpi->comm rather than taking a separate communicator argument.
  std::shared_ptr<mpi_context_t> mpi;

  // SCF metadata.
  std::string coqui_prefix = "coqui_real_axis";
  long mbpt_iter = -1;

  // Fermionic fields. Indexing: (w, s, k, i, j) or (s, k, i, j).
  // Stored as math::shm::shared_array (one copy per node) since these
  // are orbital-basis (nbnd^2 -- moderate per-rank size, but worth
  // sharing across ranks on the same node).
  std::optional<sArray_t<nda::array_view<ComplexType, 5>>> A_wskij;
  std::optional<sArray_t<nda::array_view<ComplexType, 5>>> ImSigma_wskij;
  std::optional<sArray_t<nda::array_view<ComplexType, 5>>> ReSigma_wskij;
  std::optional<sArray_t<nda::array_view<ComplexType, 4>>> Sigma_x_skij;

  // Quasiparticle SCF fields. Allocated only by the QP-SCF driver
  // (real_axis_qp_scf_loop); unused by the standard scGW path. Mirrors
  // methods::MBState::{sF_skij, sMO_skia, sE_ska, sDm_skij} on the imag-axis side.
  //
  //   H_eff_skij  : effective one-body Hamiltonian (the QSGW iteration
  //                 variable) -- (s, k, nbnd, nbnd) complex hermitian.
  //   MO_skia     : MO coefficients in the primary basis -- column a is
  //                 the a-th MO. (s, k, nbnd, nbnd).
  //   E_ska       : MO eigenvalues in absolute energy. (s, k, nbnd).
  //   Dm_skij     : density matrix in the primary basis,
  //                 Dm_ij = sum_n MO(i,n) * f(eps_n - mu) * conj(MO(j,n)).
  std::optional<sArray_t<nda::array_view<ComplexType, 4>>> H_eff_skij;
  std::optional<sArray_t<nda::array_view<ComplexType, 4>>> MO_skia;
  std::optional<sArray_t<nda::array_view<ComplexType, 3>>> E_ska;
  std::optional<sArray_t<nda::array_view<ComplexType, 4>>> Dm_skij;

  // Bosonic fields, auxiliary basis. Indexing: (q, P, Q, Omega).
  // Distributed over (P, Q) with grid = (1, gridP, gridQ, 1). Each rank
  // holds a (Nq, Naux_loc_P, Naux_loc_Q, N_Omega) local slice; the q and
  // Omega axes are not distributed (kernels need full slices along them).
  std::optional<bosonic_dArray_t> ImPi_qPQO;
  std::optional<bosonic_dArray_t> RePi_qPQO;
  std::optional<bosonic_dArray_t> ImW_qPQO;
  std::optional<bosonic_dArray_t> ReW_qPQO;

  // Head channel of the inverse symmetric dielectric function at q->0,
  // shape (N_Omega,), complex. Set by scr_coulomb_t::update_w when
  // div_treatment != "ignore_g0"; consumed by gw_t::evaluate to apply
  // the q=0 GW divergence correction. Same role as MBState::eps_inv_head
  // on the imag-axis side.
  std::optional<nda::array<ComplexType, 1>> eps_inv_head_O;

  // Default constructor leaves everything in a default-initialized state.
  real_axis_mb_state_t() = default;

  /// Convenience constructor that binds to a grid and sets finite-T params.
  real_axis_mb_state_t(real_freq_grid_t const& g)
    : beta(g.beta()), mu_chem(g.mu_chem()), grid(&g) {}

  /// Allocate fermionic sArrays for given (ns, nkpts_ibz, nbnd) shape.
  /// Each is one copy per node (math::shm::shared_array).
  void allocate_fermionic(long ns, long nkpts_ibz, long nbnd) {
    utils::check(this->mpi != nullptr,
                 "real_axis_mb_state_t::allocate_fermionic: mpi context not bound");
    utils::check(this->grid != nullptr,
                 "real_axis_mb_state_t::allocate_fermionic: grid not bound");
    long N_w = grid->N_w();
    A_wskij.emplace(*this->mpi,
        std::array<long, 5>{N_w, ns, nkpts_ibz, nbnd, nbnd});
    ImSigma_wskij.emplace(*this->mpi,
        std::array<long, 5>{N_w, ns, nkpts_ibz, nbnd, nbnd});
    ReSigma_wskij.emplace(*this->mpi,
        std::array<long, 5>{N_w, ns, nkpts_ibz, nbnd, nbnd});
    Sigma_x_skij.emplace(*this->mpi,
        std::array<long, 4>{ns, nkpts_ibz, nbnd, nbnd});
  }

  /// Allocate bosonic dArrays for given (nqpts_ibz, Naux) shape.
  /// Proc grid auto-picked: distributes (P, Q) over the full mpi->comm.
  void allocate_bosonic(long nqpts_ibz, long Naux) {
    utils::check(this->mpi != nullptr,
                 "real_axis_mb_state_t::allocate_bosonic: mpi context not bound");
    utils::check(this->grid != nullptr,
                 "real_axis_mb_state_t::allocate_bosonic: grid not bound");
    long N_O = grid->N_Omega();
    auto pgrid = bosonic_proc_grid(this->mpi->comm.size(), Naux);
    auto bsize = bosonic_block_size(pgrid, nqpts_ibz, Naux, N_O);
    using local_t = memory::array<HOST_MEMORY, ComplexType, 4>;
    std::array<long, 4> shape = {nqpts_ibz, Naux, Naux, N_O};
    ImPi_qPQO.emplace(math::nda::make_distributed_array<local_t>(
        this->mpi->comm, pgrid, shape, bsize));
    RePi_qPQO.emplace(math::nda::make_distributed_array<local_t>(
        this->mpi->comm, pgrid, shape, bsize));
    ImW_qPQO.emplace(math::nda::make_distributed_array<local_t>(
        this->mpi->comm, pgrid, shape, bsize));
    ReW_qPQO.emplace(math::nda::make_distributed_array<local_t>(
        this->mpi->comm, pgrid, shape, bsize));
    ImPi_qPQO->local() = ComplexType(0.0, 0.0);
    RePi_qPQO->local() = ComplexType(0.0, 0.0);
    ImW_qPQO->local()  = ComplexType(0.0, 0.0);
    ReW_qPQO->local()  = ComplexType(0.0, 0.0);
  }
};

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_REAL_AXIS_MB_STATE_HPP
