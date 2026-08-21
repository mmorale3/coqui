#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

add_library(scf_lib simple_dyson.cpp dca_dyson.cpp dca_dyson.h scf_driver.hpp scf_driver.cpp rpa.cpp scf_common.hpp scf_common.cpp qp_scf_common.cpp mb_solver_t.h qp_params_t.h)
target_link_libraries(scf_lib PUBLIC utils numerics meanfield hamilt ac_lib hf_lib gw_lib gf2_lib methods_tools_lib)

# RW-2 (notes/rw_real_axis_w_spec.md): qp_scf_common.cpp -> wc_band_elements.hpp pulls in the
# RW-1 real-axis W chain when qp_modea_wfit = "spectral" is compiled in. Header-only module,
# but it needs the FINUFFT symbols (via fft_lib) and the headers. With ENABLE_FINUFFT OFF the
# include block is #ifdef'd away and this link is not added -- the flag-off inertness class.
if(ENABLE_FINUFFT)
  target_link_libraries(scf_lib PUBLIC fft_lib FINUFFT::finufft)
endif()
install(TARGETS scf_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if (BUILD_UNIT_TESTS)
    add_subdirectory(tests)
endif()
