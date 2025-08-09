#ifndef COQUI_MBSTATE_H
#define COQUI_MBSTATE_H

#include "IO/app_loggers.h"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/embedding/projector_boson_t.h"

namespace methods {

/**
 * @class MBState
 * @brief Container of a many-body Green's function solution
 *
 * This class encapsulates components of a many-body Green's function solution.
 * It serves as a state holder which can be used as input/output of a
 * many-body electronic structure method.
 *
 * Note that whatever is known before a many-body calculation does not belong to this class.
 * Those components are part of the definition of a `system`.
 */
struct MBState {

public:
  using mpi_context_t = utils::mpi_context_t<>;
  template<nda::MemoryArray Array_base_t>
  using sArray_t = math::shm::shared_array<Array_base_t>;
  template<nda::MemoryArray Array_base_t>
  using dArray_t = memory::darray_t<Array_base_t, mpi3::communicator>;
  template<int N>
  using shape_t = std::array<long,N>;

public:
  // mbpt constructor
  MBState(std::shared_ptr<mpi_context_t> mpi_in, imag_axes_ft::IAFT &ft_in,
          std::string prefix, bool restart_from_checkpoint=false);

  // mbpt embedding constructors
  MBState(imag_axes_ft::IAFT &ft_in, std::string prefix,
          std::shared_ptr<mf::MF> &mf, std::string C_file, bool translate_home_cell=false,
          bool restart_from_checkpoint=false);
  MBState(imag_axes_ft::IAFT &ft_in, std::string prefix,
          std::shared_ptr<mf::MF> &mf, const nda::array<ComplexType, 5> &C_ksIai,
          const nda::array<long, 3> &band_window, const nda::array<RealType, 2> &kpts_crys,
          bool translate_home_cell=false, bool restart_from_checkpoint=false);

  MBState(MBState const&) = default;
  MBState(MBState &&) = default;
  MBState & operator=(const MBState &) = default;
  MBState & operator=(MBState &&) = default;

  ~MBState(){}

  bool read_local_polarizabilities(long weiss_b_iter=-1);
  void set_local_polarizabilities(std::map<std::string, nda::array<ComplexType, 5>> local_polarizabilities);
  void set_local_hf_potentials(std::map<std::string, nda::array<ComplexType, 4>> local_hf_potentials);
  void set_local_selfenergies(std::map<std::string, nda::array<ComplexType, 5>> local_selfenergies);

public:
  std::shared_ptr<mpi_context_t> mpi;
  imag_axes_ft::IAFT* ft = nullptr;
  // do we need to store the mean-field object?

  // checkpoint file info
  std::string coqui_prefix = "coqui";
  long mbpt_iter = -1; // iteration number of the many-body solution
  long df_1e_iter = -1; // iteration number of the downfolded 1e Hamiltonian
  long df_2e_iter = -1; // iteration number of the downfolded 2e Hamiltonian
  long embed_iter = -1; // iteration number of the embedding solution

  /** all components of a many-body state are optional **/
  // lattice quantities
  std::optional<sArray_t<nda::array_view<ComplexType, 5> > > sG_tskij;
  std::optional<sArray_t<nda::array_view<ComplexType, 4> > > sDm_skij;
  std::optional<sArray_t<nda::array_view<ComplexType, 5> > > sSigma_tskij;
  std::optional<sArray_t<nda::array_view<ComplexType, 4> > > sF_skij;
  std::optional<dArray_t<nda::array<ComplexType, 4> > > dW_qtPQ;
  std::optional<nda::array<ComplexType, 1> > eps_inv_head;
  std::string screen_type = "";
  // local quantities for quantum embedding
  std::optional<projector_boson_t> proj_boson; // Since projector is always used with a many-body state, we let MBState own it
  std::optional<nda::array<ComplexType, 5> > Sigma_imp_wsIab;
  std::optional<nda::array<ComplexType, 4> > Vhf_imp_sIab;
  std::optional<nda::array<ComplexType, 4> > Vcorr_dc_sIab;
  std::optional<nda::array<ComplexType, 5> > Sigma_dc_wsIab;
  std::optional<nda::array<ComplexType, 4> > Vhf_dc_sIab;
  std::optional<sArray_t<nda::array_view<ComplexType, 5> > > sPi_imp_wabcd;
  std::optional<sArray_t<nda::array_view<ComplexType, 5> > > sPi_dc_wabcd;

};

} // methods

#endif //COQUI_MBSTATE_H
