#undef NDEBUG

#include "catch2/catch.hpp"
#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "utilities/test_common.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"

namespace bdft_tests {

  using utils::ARRAY_EQUAL;
  template<int N>
  using shape_t = std::array<long,N>;

  TEST_CASE("iaft_ir_read", "[iaft_ir_read]") {
    double beta = 1000;
    double lambda = 1.2e4;
    {
      imag_axes_ft::ir::IR myir(beta, lambda);
      imag_axes_ft::IAFT iaft(myir);
    }
    imag_axes_ft::IAFT iaft(beta, lambda, imag_axes_ft::ir_source);

    REQUIRE(iaft.beta() == beta);
    REQUIRE(iaft.nt_f() == 137);
    REQUIRE(iaft.nw_f() == 138);
    REQUIRE(iaft.nt_b() == 137);
    REQUIRE(iaft.nw_b() == 137);

    REQUIRE(iaft.Ttw_ff().shape() == shape_t<2>{iaft.nt_f(), iaft.nw_f()});
    REQUIRE(iaft.Twt_ff().shape() == shape_t<2>{iaft.nw_f(), iaft.nt_f()});
    REQUIRE(iaft.Ttt_bf().shape() == shape_t<2>{iaft.nt_b(), iaft.nt_f()});
    REQUIRE(iaft.Ttw_bb().shape() == shape_t<2>{iaft.nt_b(), iaft.nw_b()});
    REQUIRE(iaft.Twt_bb().shape() == shape_t<2>{iaft.nw_b(), iaft.nt_b()});
    REQUIRE(iaft.Ttt_fb().shape() == shape_t<2>{iaft.nt_f(), iaft.nt_b()});
    REQUIRE(iaft.T_beta_t_ff().shape() == shape_t<1>{iaft.nt_f()});

    auto eye1 = iaft.Ttw_ff() * iaft.Twt_ff();
    auto eye2 = iaft.Ttt_bf() * iaft.Ttt_fb();
    ARRAY_EQUAL(eye1, nda::eye<ComplexType>(iaft.nt_f()), 1e-10);
    ARRAY_EQUAL(eye2, nda::eye<RealType>(iaft.nt_b()), 1e-10);
  }

  TEST_CASE("iaft_ir_ft", "[iaft_ir_ft]") {
    decltype(nda::range::all) all;
    double beta = 1000;
    double lambda = 1.2e4;
    imag_axes_ft::IAFT myft(beta, lambda, imag_axes_ft::ir_source);
    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filename = source_path + "/tests/unit_test_files/pyscf/si_kp222_krhf/Gw_Gt_beta1000_1e5.h5";
    h5::file file(filename, 'r');
    h5::group grp(file);

    nda::array<std::complex<double>, 5> G_tskij_ref;
    nda::array<std::complex<double>, 5> G_wskij_ref;
    nda::array<std::complex<double>, 4> Dm_skij_ref;
    nda::h5_read(grp, "Gt", G_tskij_ref);
    nda::h5_read(grp, "Gw", G_wskij_ref);
    nda::h5_read(grp, "Dm", Dm_skij_ref);

    size_t nts   = G_tskij_ref.shape(0);
    size_t nw    = G_wskij_ref.shape(0);
    size_t ns    = G_tskij_ref.shape(1);
    size_t nkpts = G_tskij_ref.shape(2);
    size_t nbnd  = G_tskij_ref.shape(3);
    // Fourier transform between tau and iwn
    {
      nda::array<std::complex<double>, 5> G_tskij(nts, ns, nkpts, nbnd, nbnd);
      nda::array<std::complex<double>, 5> G_wskij(nw, ns, nkpts, nbnd, nbnd);
      myft.tau_to_w(G_tskij_ref, G_wskij, imag_axes_ft::fermi);
      myft.w_to_tau(G_wskij_ref, G_tskij, imag_axes_ft::fermi);

      ARRAY_EQUAL(G_tskij, G_tskij_ref, 1e-12);
      ARRAY_EQUAL(G_wskij, G_wskij_ref, 1e-12);
    }
    // tau to a specific w
    {
      nda::array<std::complex<double>, 5> G_wskij(nw, ns, nkpts, nbnd, nbnd);
      nda::array<std::complex<double>, 4> G_skij(ns, nkpts, nbnd, nbnd);
      for (size_t n = 0; n < nw; ++n) {
        nda::array_view<std::complex<double>, 4> Gw_skij({ns, nkpts, nbnd, nbnd},
                                                         G_wskij.data() + n*ns*nkpts*nbnd*nbnd);
        myft.tau_to_w(G_tskij_ref, G_skij, imag_axes_ft::fermi, n);
        Gw_skij = G_skij;
      }
      ARRAY_EQUAL(G_wskij, G_wskij_ref, 1e-12);
    }
    // Partial Fourier transform
    {
      nda::array<std::complex<double>, 5> G_tskij(nts, ns, nkpts, nbnd, nbnd);
      for (size_t n = 0; n < nw; ++n) {
        auto Gw_skij = G_wskij_ref(n, all, all, all, all);
        myft.w_to_tau_partial(Gw_skij, G_tskij, imag_axes_ft::fermi, n);
      }
      ARRAY_EQUAL(G_tskij, G_tskij_ref, 1e-12);
    }
    // tau = beta^{-} via the interpolation at sparse sampling nodes
    {
      nda::array<std::complex<double>, 4> Dm_skij(ns, nkpts, nbnd, nbnd);
      myft.tau_to_beta(G_tskij_ref, Dm_skij);
      Dm_skij *= -1.0;
      ARRAY_EQUAL(Dm_skij, Dm_skij_ref, 1e-12);
    }
  }
} // bdft_tests
