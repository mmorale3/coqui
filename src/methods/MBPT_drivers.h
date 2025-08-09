#ifndef METHODS_MBPT_DRIVERS_H
#define METHODS_MBPT_DRIVERS_H

#include <string>

#include "configuration.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "cxxopts.hpp"

#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"

#include "mean_field/MF.hpp"
#include "methods/mb_state/mb_state.hpp"

namespace mpi3 = boost::mpi3;
namespace methods
{

/**
 * @brief Many-body perturbation calculations from a given mean-field and ERI objects with arguments in property tree.
 */
template<typename eri_t>
void mbpt(std::string solver_type, eri_t &eri, ptree const& pt);

template<typename eri_t>
void mbpt(std::string solver_type, eri_t &eri, ptree const& pt,
          nda::array<ComplexType, 5> const& C_ksIai,
          nda::array<long, 3> const& band_window,
          nda::array<RealType, 2> const& kpts_crys,
          std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities=std::nullopt);


/**
 * @brief Downfolding of the local Green's function
 */
auto downfold_gloc_impl(std::shared_ptr<mf::MF> mf,
                        MBState&& mb_state, ptree const& pt)
-> nda::array<ComplexType, 5>;

auto downfold_gloc(std::shared_ptr<mf::MF> mf, ptree const& pt)
-> nda::array<ComplexType, 5>;

auto downfold_gloc(std::shared_ptr<mf::MF> mf, ptree const& pt,
                   nda::array<ComplexType, 5> const& C_ksIai,
                   nda::array<long, 3> const& band_window,
                   nda::array<RealType, 2> const& kpts_crys)
-> nda::array<ComplexType, 5>;


/**
 * @brief Downfolding of one-electron quantities, Green's function, self-energy, and effective potential
 * This is currently used in the GW+EDMFT calculations in the toml input mode.
 * Note: This functions is doing too many things at once, it will be refactored in the future.
 */
void downfolding_1e(std::shared_ptr<mf::MF> mf, ptree const& pt,
                    std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies=std::nullopt,
                    std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials=std::nullopt);

void downfolding_1e(std::shared_ptr<mf::MF> mf, ptree const& pt,
                    nda::array<ComplexType, 5> const& C_ksIai,
                    nda::array<long, 3> const& band_window,
                    nda::array<RealType, 2> const& kpts_crys,
                    std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies=std::nullopt,
                    std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials=std::nullopt);

/**
 * @brief Downfolding of the local Coulomb interactions
 */
template<typename eri_t>
std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_wloc_impl(eri_t &eri, MBState&& mb_state, ptree const& pt,
                   std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities);

template<typename eri_t>
std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_wloc(eri_t &eri, ptree const& pt,
              std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities=std::nullopt);

template<typename eri_t>
std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
downfold_wloc(eri_t &eri, ptree const& pt,
              nda::array<ComplexType, 5> const& C_ksIai,
              nda::array<long, 3> const& band_window,
              nda::array<RealType, 2> const& kpts_crys,
              std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities=std::nullopt);


/**
 * @brief Downfolding of various two-electron quantities, bare Coulomb, screened Coulomb and partially screened Coulomb interactions.
 * This is currently used in the GW+EDMFT calculations in the toml input mode.
 * Note: This functions is doing too many things at once, it will be refactored in the future.
 */
template<bool return_vw=false, typename eri_t>
std::conditional_t<return_vw, std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>, void>
downfolding_2e(eri_t &eri, ptree const& pt,
               std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities=std::nullopt);

template<bool return_vw=false, typename eri_t>
std::conditional_t<return_vw, std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>, void>
downfolding_2e(eri_t &eri, ptree const& pt,
               nda::array<ComplexType, 5> const& C_ksIai,
               nda::array<long, 3> const& band_window,
               nda::array<RealType, 2> const& kpts_crys,
               std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities=std::nullopt);



template<typename eri_t>
void hf_downfold(eri_t &eri, ptree const& pt);

template<typename eri_t>
void gw_downfold(eri_t &eri, ptree const& pt);

/**
 * @brief Embedding (i.e. upfolding) of local DMFT self-energy corrections to a MBPT solution stored in the checkpoint file.
 */
void dmft_embed(std::shared_ptr<mf::MF> mf, ptree const& pt,
                nda::array<ComplexType, 5> const& C_ksIai,
                nda::array<long, 3> const& band_window,
                nda::array<RealType, 2> const& kpts_crys,
                std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials=std::nullopt,
                std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies=std::nullopt);

void dmft_embed(std::shared_ptr<mf::MF> mf, ptree const& pt);

}
#endif
