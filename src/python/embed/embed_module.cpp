#include <c2py/c2py.hpp>
#include "IO/app_loggers.h"
#include "methods/MBPT_drivers.h"

#include "python/mean_field/mf_module.hpp"
#include "python/mean_field/mf_module.wrap.hxx"
#include "python/interaction/eri_module.hpp"
#include "python/interaction/eri_module.wrap.hxx"

namespace coqui_py {

  template<typename eri_handler_t>
  void downfold_2e(eri_handler_t &eri, const std::string &df_params,
              const nda::array<ComplexType, 5> &C_ksIai,
              const nda::array<long, 3> &band_window,
              const nda::array<RealType, 2> &kpts_crys,
              std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
    auto parser = InputParser(df_params);
    methods::downfolding_2e<false>(eri.get_eri(), parser.get_root(),
                                   C_ksIai, band_window, kpts_crys, std::move(local_polarizabilities));
  }

  template<typename eri_handler_t>
  std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>
  downfold_2e_return_vw(eri_handler_t &eri, const std::string &df_params,
              const nda::array<ComplexType, 5> &C_ksIai,
              const nda::array<long, 3> &band_window,
              const nda::array<RealType, 2> &kpts_crys,
              std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
    auto parser = InputParser(df_params);
    return methods::downfolding_2e<true>(eri.get_eri(), parser.get_root(),
                                         C_ksIai, band_window, kpts_crys, std::move(local_polarizabilities));
  }

  template<typename eri_handler_t>
  void downfold_2e(eri_handler_t &eri, const std::string &df_params,
                   std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
    auto parser = InputParser(df_params);
    methods::downfolding_2e<false>(eri.get_eri(), parser.get_root(), std::move(local_polarizabilities));
  }

  template<typename eri_handler_t>
  std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>
  downfold_2e_return_vw(eri_handler_t &eri, const std::string &df_params,
                        std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
    auto parser = InputParser(df_params);
    return methods::downfolding_2e<true>(eri.get_eri(), parser.get_root(), local_polarizabilities);
  }

  template<typename eri_handler_t>
  std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>
  downfold_wloc(eri_handler_t &eri, const std::string &df_params,
                std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
    auto parser = InputParser(df_params);
    return methods::downfold_wloc(eri.get_eri(), parser.get_root(),
                                  std::move(local_polarizabilities));
  }
  template<typename eri_handler_t>
  std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>
  downfold_wloc(eri_handler_t &eri, const std::string &df_params,
                const nda::array<ComplexType, 5> &C_ksIai,
                const nda::array<long, 3> &band_window,
                const nda::array<RealType, 2> &kpts_crys,
                std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
    auto parser = InputParser(df_params);
    return methods::downfold_wloc(eri.get_eri(), parser.get_root(),
                                  C_ksIai, band_window, kpts_crys, std::move(local_polarizabilities));
  }

  auto downfold_gloc(const Mf &mf, const std::string &df_params)
  -> nda::array<ComplexType, 5> {
    auto parser = InputParser(df_params);
    return methods::downfold_gloc(mf.get_mf(), parser.get_root());
  }

  auto downfold_gloc(const Mf &mf, const std::string &df_params,
                     const nda::array<ComplexType, 5> &C_ksIai,
                     const nda::array<long, 3> &band_window,
                     const nda::array<RealType, 2> &kpts_crys)
  -> nda::array<ComplexType, 5> {
    auto parser = InputParser(df_params);
    return methods::downfold_gloc(mf.get_mf(), parser.get_root(), C_ksIai, band_window, kpts_crys);
  }

  void downfold_1e(const Mf &mf, const std::string &df_params,
                   const nda::array<ComplexType, 5> &C_ksIai,
                   const nda::array<long, 3> &band_window,
                   const nda::array<RealType, 2> &kpts_crys,
                   std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies,
                   std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials) {
    auto parser = InputParser(df_params);
    methods::downfolding_1e(mf.get_mf(), parser.get_root(), C_ksIai, band_window, kpts_crys,
                            std::move(local_selfenergies), std::move(local_hf_potentials));
  }

  void downfold_1e(const Mf &mf, const std::string &df_params,
                   std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies,
                   std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials) {
    auto parser = InputParser(df_params);
    methods::downfolding_1e(mf.get_mf(), parser.get_root(),
                            std::move(local_selfenergies), std::move(local_hf_potentials));
  }

  void dmft_embed(const Mf &mf, const std::string &embed_params,
                  const nda::array<ComplexType, 5> &C_ksIai,
                  const nda::array<long, 3> &band_window,
                  const nda::array<RealType, 2> &kpts_crys,
                  std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials,
                  std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies) {
    auto parser = InputParser(embed_params);
    methods::dmft_embed(mf.get_mf(), parser.get_root(), C_ksIai, band_window, kpts_crys,
                        local_hf_potentials, local_selfenergies);
  }

  void dmft_embed(const Mf &mf, const std::string &embed_params) {
    auto parser = InputParser(embed_params);
    methods::dmft_embed(mf.get_mf(), parser.get_root());
  }

  // public template instantiation
  template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>
  downfold_wloc(ThcCoulomb&, const std::string &,
                std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

  template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>
  downfold_wloc(ThcCoulomb&, const std::string &,
                const nda::array<ComplexType, 5> &,
                const nda::array<long, 3> &,
                const nda::array<RealType, 2> &,
                std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);


  template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>
  downfold_2e_return_vw(ThcCoulomb&, const std::string&,
                    const nda::array<ComplexType, 5>&,
                    const nda::array<long, 3>&,
                    const nda::array<RealType, 2>&,
                    std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);
  template void
  downfold_2e(ThcCoulomb&, const std::string&,
              const nda::array<ComplexType, 5>&,
              const nda::array<long, 3>&,
              const nda::array<RealType, 2>&,
              std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

  template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>
  downfold_2e_return_vw(ThcCoulomb&, const std::string&, std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);
  template void downfold_2e(ThcCoulomb&, const std::string&, std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

} // coqui_py

#include "embed_module.wrap.cxx"
