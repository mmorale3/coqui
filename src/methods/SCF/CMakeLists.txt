#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

add_library(scf_lib simple_dyson.cpp dca_dyson.cpp dca_dyson.h scf_driver.hpp scf_driver.cpp rpa.cpp scf_common.hpp scf_common.cpp qp_scf_common.cpp mb_solver_t.h qp_params_t.h)
target_link_libraries(scf_lib PUBLIC utils numerics meanfield hamilt ac_lib hf_lib gw_lib gf2_lib methods_tools_lib)

# qp_scf_common.cpp includes wc_band_elements.hpp (the real-axis W_c chain used by
# qp_modea_wfit = "spectral"). The module is header-only but needs the FINUFFT symbols (via
# fft_lib) and headers. With ENABLE_FINUFFT OFF the include block is compiled out and this
# link is not added.
if(ENABLE_FINUFFT)
  target_link_libraries(scf_lib PUBLIC fft_lib FINUFFT::finufft)
endif()
# the device causal projection of Sigma_c (CUDA builds only)
if(ENABLE_CUDA)
  add_subdirectory(cuda)
  target_link_libraries(scf_lib PUBLIC scf_cuda)
endif()
install(TARGETS scf_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if (BUILD_UNIT_TESTS)
    add_subdirectory(tests)
endif()
