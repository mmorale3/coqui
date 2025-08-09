#ifndef ERI_MODULE_HPP
#define ERI_MODULE_HPP

#include "python/utils/mpi_handler.hpp"
#include "python/utils/mpi_handler.wrap.hxx"
#include "python/mean_field/mf_module.hpp"
#include "python/mean_field/mf_module.wrap.hxx"

#include "IO/ptree/InputParser.hpp"
#include "methods/ERI/eri_utils.hpp"

namespace coqui_py {

  C2PY_IGNORE
  inline decltype(auto) make_thc(const Mf &mf, const std::string &thc_params) {
    auto parser = InputParser(thc_params);
    return methods::make_thc(mf.get_mf(), parser.get_root());
  }

  class ThcCoulomb {
  public:
    ThcCoulomb(const Mf &mf, const std::string &thc_params):
    _thc(make_thc(mf, thc_params)) {}

    ~ThcCoulomb() = default;
    ThcCoulomb(ThcCoulomb const&) = default;
    ThcCoulomb(ThcCoulomb &&) = default;
    ThcCoulomb& operator=(ThcCoulomb const&) = default;
    ThcCoulomb& operator=(ThcCoulomb &&) = default;

    void init() { _thc.init(!_thc.thc_builder_is_null()); }
    auto initialized() const { return _thc.initialized(); }

    auto Np() const { return _thc.Np(); }
    auto nkpts() const { return _thc.nkpts(); }
    auto nkpts_ibz() const { return _thc.nkpts_ibz(); }
    auto nqpts() const { return _thc.nqpts(); }
    auto nqpts_ibz() const { return _thc.nqpts_ibz(); }
    auto nspin() const { return _thc.ns(); }
    auto nspin_in_basis() const { return _thc.ns_in_basis(); }
    auto nbnd() const { return _thc.nbnd(); }

    // create a new MpiHandler from _thc's mpi
    auto mpi() const { return MpiHandler(_thc.mpi()); }
    // create a new Mf from _thc's MF
    auto mf() const { return Mf(_thc.MF()); }

    C2PY_IGNORE
    auto& get_eri() { return _thc; }
    C2PY_IGNORE
    auto get_mpi() const { return _thc.mpi(); }
    C2PY_IGNORE
    auto get_mf() const { return _thc.MF(); }

  private:
    methods::thc_reader_t _thc;

  }; // ThcCoulomb

  C2PY_IGNORE
  inline decltype(auto) make_cholesky(const Mf &mf, const std::string &chol_params) {
    auto parser = InputParser(chol_params);
    return methods::make_cholesky(mf.get_mf(), parser.get_root());
  }

  class CholCoulomb {
  public:
    CholCoulomb(const Mf &mf, const std::string &chol_params):
    _cholesky(make_cholesky(mf, chol_params)) {}

    ~CholCoulomb() = default;
    CholCoulomb(CholCoulomb const&) = default;
    CholCoulomb(CholCoulomb &&) = default;
    CholCoulomb& operator=(CholCoulomb const&) = default;
    CholCoulomb& operator=(CholCoulomb &&) = default;

    //void init() { _thc.init(!_thc.thc_builder_is_null()); }
    //auto initialized() { return _thc.initialized(); }

    // create a new MpiHandler from _thc's mpi
    auto mpi() const { return MpiHandler(_cholesky.mpi()); }
    // create a new Mf from _thc's MF
    auto mf() const { return Mf(_cholesky.MF()); }

    C2PY_IGNORE
    auto& get_eri() { return _cholesky; }
    C2PY_IGNORE
    auto get_mpi() const { return _cholesky.mpi(); }
    C2PY_IGNORE
    auto get_mf() const { return _cholesky.MF(); }

  private:
    methods::chol_reader_t _cholesky;

  }; // CholCoulomb

} // coqui_py

#endif
