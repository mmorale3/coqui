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
#include "methods/GW_real_axis/real_freq_grid.hpp"

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
 *   ImPi_qOmegaPQ  : (nqpts_ibz, N_Omega, Naux, Naux)    Im polarization
 *   RePi_qOmegaPQ  : (nqpts_ibz, N_Omega, Naux, Naux)    Re polarization
 *   ImW_qOmegaPQ   : (nqpts_ibz, N_Omega, Naux, Naux)    Im screened interaction
 *   ReW_qOmegaPQ   : (nqpts_ibz, N_Omega, Naux, Naux)    Re screened interaction
 *   ImSigma_wskij  : (N_w,  ns, nkpts_ibz, nbnd, nbnd)   Im correlation self-energy
 *   ReSigma_wskij  : (N_w,  ns, nkpts_ibz, nbnd, nbnd)   Re correlation self-energy
 *   Sigma_x_skij   : (ns, nkpts_ibz, nbnd, nbnd)         static exchange self-energy
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

  // Finite-temperature parameters. Mandatory, not optional.
  double beta    = 0.0;
  double mu_chem = 0.0;

  // Reference to the frequency/time grid (non-owning).
  real_freq_grid_t const* grid = nullptr;

  // Non-owning MPI communicator. Distributed kernels accept a comm by
  // reference; this slot is provided for callers that want to thread the
  // comm through state alongside the grid.
  boost::mpi3::communicator* comm = nullptr;

  // SCF metadata.
  std::string coqui_prefix = "coqui_real_axis";
  long mbpt_iter = -1;

  // Fermionic fields. Indexing: (w, s, k, i, j).
  std::optional<nda::array<ComplexType, 5>> A_wskij;
  std::optional<nda::array<ComplexType, 5>> ImSigma_wskij;
  std::optional<nda::array<ComplexType, 5>> ReSigma_wskij;
  std::optional<nda::array<ComplexType, 4>> Sigma_x_skij;

  // Bosonic fields, auxiliary basis. Indexing: (q, Omega, P, Q).
  std::optional<nda::array<ComplexType, 4>> ImPi_qOmegaPQ;
  std::optional<nda::array<ComplexType, 4>> RePi_qOmegaPQ;
  std::optional<nda::array<ComplexType, 4>> ImW_qOmegaPQ;
  std::optional<nda::array<ComplexType, 4>> ReW_qOmegaPQ;

  // Default constructor leaves everything in a default-initialized state.
  real_axis_mb_state_t() = default;

  /// Convenience constructor that binds to a grid and sets finite-T params.
  real_axis_mb_state_t(real_freq_grid_t const& g)
    : beta(g.beta()), mu_chem(g.mu_chem()), grid(&g) {}

  /// Allocate fermionic arrays for given (ns, nkpts_ibz, nbnd) shape.
  void allocate_fermionic(long ns, long nkpts_ibz, long nbnd) {
    long N_w = grid->N_w();
    A_wskij        = nda::array<ComplexType, 5>(N_w, ns, nkpts_ibz, nbnd, nbnd);
    ImSigma_wskij  = nda::array<ComplexType, 5>(N_w, ns, nkpts_ibz, nbnd, nbnd);
    ReSigma_wskij  = nda::array<ComplexType, 5>(N_w, ns, nkpts_ibz, nbnd, nbnd);
    Sigma_x_skij   = nda::array<ComplexType, 4>(ns, nkpts_ibz, nbnd, nbnd);
  }

  /// Allocate bosonic arrays for given (nqpts_ibz, Naux) shape.
  void allocate_bosonic(long nqpts_ibz, long Naux) {
    long N_O = grid->N_Omega();
    ImPi_qOmegaPQ = nda::array<ComplexType, 4>(nqpts_ibz, N_O, Naux, Naux);
    RePi_qOmegaPQ = nda::array<ComplexType, 4>(nqpts_ibz, N_O, Naux, Naux);
    ImW_qOmegaPQ  = nda::array<ComplexType, 4>(nqpts_ibz, N_O, Naux, Naux);
    ReW_qOmegaPQ  = nda::array<ComplexType, 4>(nqpts_ibz, N_O, Naux, Naux);
  }
};

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_REAL_AXIS_MB_STATE_HPP
