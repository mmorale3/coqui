#ifndef ORBITALS_ROTATE_H
#define ORBITALS_ROTATE_H

#include "configuration.hpp"

#include "mpi3/communicator.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "mean_field/MF.hpp"

namespace orbitals
{


template<MEMORY_SPACE MEM = HOST_MEMORY, utils::Communicator comm_t>
void orthonormalize(memory::darray_t<memory::array<MEM, ComplexType, 4>, comm_t>& psi,
                    double cutoff = 1e-8); 

template<MEMORY_SPACE MEM = HOST_MEMORY, utils::Communicator comm_t>
void orthonormalize(nda::array<int,2> const& ranges,
                    memory::darray_t<memory::array<MEM, ComplexType, 4>, comm_t>& psi);
                                        

template<MEMORY_SPACE MEM = HOST_MEMORY, utils::Communicator comm_t>
auto canonicalize_diagonal_basis(mf::MF& mf, 
                    memory::darray_t<memory::array<MEM, ComplexType, 4>, comm_t>& psi)
  -> nda::array<ComplexType, 3>;


//void diagonalize() {};

}

#endif
