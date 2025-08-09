#ifndef COQUI_IAFT_UTILS_HPP
#define COQUI_IAFT_UTILS_HPP

#include "h5/h5.hpp"

#include "numerics/imag_axes_ft/IAFT.hpp"

namespace imag_axes_ft {
  /**
   * Reconstruct IAFT object from the metadata in bdft scf output
   * @return IAFT
   */
  inline decltype(auto) read_iaft(std::string scf_file, bool print_meta_log = true) {
    double beta;
    double lambda;
    std::string prec;
    std::string source;

    h5::file file(scf_file, 'r');
    h5::group grp(file);
    auto iaft_grp = grp.open_group("imaginary_fourier_transform");
    h5::h5_read(iaft_grp, "source", source);
    h5::h5_read(iaft_grp, "prec", prec);
    h5::h5_read(iaft_grp, "beta", beta);
    h5::h5_read(iaft_grp, "lambda", lambda);

    return imag_axes_ft::IAFT(beta, lambda, imag_axes_ft::string_to_source_enum(source), prec, print_meta_log);
  }

} // imag_axes_ft



#endif //COQUI_IAFT_UTILS_HPP
