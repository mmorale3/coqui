#ifndef HAMILTONIAN_ADD_KINETIC_H
#define HAMILTONIAN_ADD_KINETIC_H

#include "configuration.hpp"
#include "grids/g_grids.hpp"
#include "nda/concepts.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "mean_field/MF.hpp"

namespace hamilt
{

/**
 * Adds the kinetic contribution to hpsi: hpsi(s,k,a,G) += 0.5*|G+k|^2 * psi(s,k,b,G)
 * Assumes psi and hpsi are in the 'w' grid.
 *
 * @param npol  - [INPUT] # of polarizations
 * @param kpts  - [INPUT] k-points in the Brillouin zones
 * @param wfc_g - [INPUT] Handler for "Wavefunction" plane-wave grid
 * @param psi   - [INPUT] Single-particle basis (s,k,a,g) where g is in the 'w' grid
 * @param hpsi  - [INPUT] \hat{h} * psi where h is an arbitrary local operator
 *                [OUTPUT] (\hat{h} + 0.5*|G+k|^2) * psi
 */
void add_kinetic(int npol,
                 nda::ArrayOfRank<2> auto const& kpts,
                 grids::truncated_g_grid const& wfc_g,
                 math::nda::DistributedArrayOfRank<4> auto const& psi,
                 math::nda::DistributedArrayOfRank<4> auto & hpsi);

}

#endif
