#ifndef HAMILTONIANS_H
#define HAMILTONIANS_H

#include "configuration.hpp"

#include "mpi3/communicator.hpp"
#include "utilities/mpi_context.h"
#include "IO/ptree/ptree_utilities.hpp"

#include "mean_field/MF.hpp"

#include "methods/ERI/thc_reader_t.hpp"
#include "methods/ERI/chol_reader_t.hpp"

namespace methods 
{

template<MEMORY_SPACE MEM = HOST_MEMORY>
void add_core_hamiltonian(mf::MF &mf, ptree const& pt);

template<MEMORY_SPACE MEM = HOST_MEMORY>
void add_thc_hamiltonian_components(mf::MF &mf,
                          thc_reader_t& thc, ptree const& pt);

template<MEMORY_SPACE MEM = HOST_MEMORY>
void add_cholesky_hamiltonian_components(mf::MF &mf,
                          chol_reader_t& chol, ptree const& pt);

}

#endif
