#ifndef ORBITAL_GENERATOR_H
#define ORBITAL_GENERATOR_H

#include "configuration.hpp"

#include "mpi3/communicator.hpp"
#include "IO/ptree/ptree_utilities.hpp"

#include "numerics/distributed_array/nda.hpp"
#include "mean_field/MF.hpp"

#include "orbitals/pgto.h"

namespace orbitals
{

template<MEMORY_SPACE MEM = HOST_MEMORY>
mf::MF add_pgto(mf::MF& mf, std::string fn, std::string basis, std::string type,
                int b0 = -1, bool diag_F = false, bool orthonormalize = false,
                double thresh = 1e-8, bool orthonormalize_by_shell = true); 

mf::MF eigenstate_selection(mf::MF& mf, std::string fn,
                            std::string grid_type, long n0, long nblk);

template<MEMORY_SPACE MEM = HOST_MEMORY>
void orbital_factory(mf::MF &mf, ptree const& pt);

}

#endif
