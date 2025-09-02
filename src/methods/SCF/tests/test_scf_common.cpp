#undef NDEBUG

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"

#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "utilities/mpi_context.h"
#include "mean_field/mf_utils.hpp"
#include "methods/SCF/scf_common.hpp"
#include "utilities/mpi_context.h"
#include "hamiltonian/pseudo/pseudopot.h"

namespace bdft_tests {

  using utils::VALUE_EQUAL;
  using utils::ARRAY_EQUAL;
  namespace mpi3 = boost::mpi3;
  using namespace methods;
  using utils::mpi_context_t;

  TEST_CASE("MO", "[methods_scf]") {
    auto& context = utils::make_unit_test_mpi_context();

    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/pyscf/si_kp222_krhf/";
    auto mf = mf::make_MF(context, mf::pyscf_source, filepath, "pyscf");
    hamilt::pseudopot psp(mf);
    double beta = 1000.0;

    auto [sMO_skij, sE_ski] = get_mf_MOs(*context, mf, psp);

    auto E_ski_ref = mf.eigval();
    ARRAY_EQUAL(sE_ski.local(), E_ski_ref, 1e-8);

    double Nelec = compute_Nelec(0.2, mf, sE_ski, beta);
    VALUE_EQUAL(Nelec, 8.0, 1e-9);
  }

  TEST_CASE("mu", "[methods_scf]") {
    auto& context = utils::make_unit_test_mpi_context();

    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/pyscf/si_kp222_krhf/";
    auto mf = mf::make_MF(context, mf::pyscf_source, filepath, "pyscf");
    hamilt::pseudopot psp(mf);
    double beta = 1000.0;

    auto [sMO_skij, sE_ski] = get_mf_MOs(*context, mf, psp);

    double mu = update_mu(0.0, mf, sE_ski, beta);
    VALUE_EQUAL(mu, 0.175, 1e-9);
  }

  TEST_CASE("qp_Dm_and_G", "[methods_scf]") {
    auto& context = utils::make_unit_test_mpi_context();

    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/pyscf/si_kp222_krhf/";
    auto mf = mf::make_MF(context, mf::pyscf_source, filepath, "pyscf");
    hamilt::pseudopot psp(mf);

    double beta = 1000;
    double wmax = 1.2;
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_source);

    auto [sMO_skij, sE_ski] = get_mf_MOs(*context, mf, psp);
    double mu = 0.175;
    double Nelec = compute_Nelec(mu, mf, sE_ski, ft.beta());
    VALUE_EQUAL(Nelec, 8.0, 1e-8);

    auto sDm_skij = math::shm::make_shared_array<Array_view_4D_t>(
        *context, {mf.nspin(), mf.nkpts(), mf.nbnd(), mf.nbnd()});
    auto sG_tskij = math::shm::make_shared_array<Array_view_5D_t>(
        *context, {ft.nt_f(), mf.nspin(), mf.nkpts(), mf.nbnd(), mf.nbnd()});
    update_Dm(sDm_skij, sMO_skij, sE_ski, mu, ft.beta());
    update_G(sG_tskij, sMO_skij, sE_ski, mu, ft);
    ft.check_leakage(sG_tskij, imag_axes_ft::fermi, "Green's function");

    h5::file file(filepath+"/hf_Gw_Gt_beta1000_wmax1.2_high.h5", 'r');
    h5::group grp(file);

    nda::array<std::complex<double>, 5> G_tskij_ref;
    nda::array<std::complex<double>, 4> Dm_skij_ref;
    nda::h5_read(grp, "Dm", Dm_skij_ref);
    nda::h5_read(grp, "Gt", G_tskij_ref);

    ARRAY_EQUAL(sG_tskij.local(), G_tskij_ref, 1e-8);
    ARRAY_EQUAL(sDm_skij.local(), Dm_skij_ref, 1e-8);
  }


} // bdft_tests
