#ifndef COQUI_MODEL_HAMILTONIAN_UTILS_HPP
#define COQUI_MODEL_HAMILTONIAN_UTILS_HPP

#include "configuration.hpp"

#include <h5/h5.hpp>
#include <nda/h5.hpp>

#include "IO/app_loggers.h"
#include "utilities/concepts.hpp"
#include "utilities/mpi_context.h"

#include "mean_field/model_hamiltonian/model_system.hpp"

namespace mf {
namespace model {

// creates a model_readonly object with (empty) matrices at gamma point kpoint
template<utils::Communicator comm_t>
auto make_dummy_model(std::shared_ptr<utils::mpi_context_t<comm_t>> mpi, int nb, double nelec,
                      std::string outdir = "./", std::string prefix = "__dummy_model__")
{

  int ns = 1;
  int nk = 1;
  int npol = 1;

  auto symm = mf::bz_symm::gamma_point_instance();
  auto h = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});
  auto s = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});
  auto d = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});
  auto f = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});

  mf::model::model_system m(std::move(mpi),outdir,prefix,symm,ns,npol,nelec,h,s,d,f);
  return model_readonly(std::move(m));

}

} // model
} // mf
#endif
