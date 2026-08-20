#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

add_subdirectory(fft)
add_subdirectory(device_kernels)
add_subdirectory(ac)
add_subdirectory(imag_axes_ft)

add_library(numerics INTERFACE)
# coqui_blas_threading MUST come first: it carries MKL's GNU OpenMP threading layer, and it
# only wins the MKL symbols if it precedes slate/blaspp/lapackpp, which drag in
# libmkl_sequential (T-2 option (a); see the top-level CMakeLists for the full rationale).
# Empty target -- hence a no-op -- when MKL is not the BLAS in use.
if(ENABLE_SLATE)
target_link_libraries(numerics INTERFACE coqui_blas_threading arch_lib fft_lib nda_c h5_c slate_lapack_api slate device_kernels iaft_utils_lib)
ELSE(ENABLE_SLATE)
target_link_libraries(numerics INTERFACE coqui_blas_threading arch_lib fft_lib nda_c h5_c device_kernels iaft_utils_lib)
ENDIF(ENABLE_SLATE)

if(ENABLE_DLR)
  target_link_libraries(numerics INTERFACE cppdlr::cppdlr_c)
endif()

install(TARGETS numerics
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(ENABLE_TBLIS)
add_subdirectory(sandbox)
endif(ENABLE_TBLIS)

if (BUILD_UNIT_TESTS)
  add_subdirectory(fft/tests)
  add_subdirectory(distributed_array/tests)
  add_subdirectory(shared_array/tests)
  add_subdirectory(sparse/tests)

  if(ENABLE_SLATE)
    add_subdirectory(slate/tests)
  endif(ENABLE_SLATE)

  add_subdirectory(tests)
endif()

