#ifndef COQUI_ADD_VXC_H
#define COQUI_ADD_VXC_H

#include "configuration.hpp"
#include "nda/concepts.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"
#include "mean_field/MF.hpp"

namespace hamilt
{

/**
 * Adds the exchange-correlation potential to hpsi
 *
 * @tparam MF_t   - Type parameter for mean-field handler
 * @param k_range - [INPUT] Index range of k-points
 * @param b_range - [INPUT] Index range of orbitals
 * @param psi     - [INPUT] Single-particle basis (s,k,a,g) where g is in the 'w' grid
 * @param Vxc     - [INPUT] conj(psi(i)) * h * psi(j) 
 *                - [OUTPUT] conj(psi(i)) * ( h + Vxc ) * psi(j)
 */
template<typename MF_t>
void add_vxc(MF_t& mf, nda::range k_range, nda::range b_range,
             math::nda::DistributedArrayOfRank<4> auto const& psi,
             math::nda::DistributedArrayOfRank<4> auto & hpsi);

template<typename MF_t, nda::MemoryArrayOfRank<3> array_t>
void read_vxc_h5(MF_t &mf, h5::group& grp0, math::shm::shared_array<array_t> &svxc);

}

#endif //COQUI_ADD_VXC_H
