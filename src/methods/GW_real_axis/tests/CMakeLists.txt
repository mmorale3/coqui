
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# Tests of the real-axis W chain: grids, convolutions, Pi, update_w, the hermiticity of the
# aux-projected spectral function, and the real-axis vs imaginary-axis W cross-check.
if(ENABLE_FINUFFT)
  set(TEST_ID methods_gw_real_axis)

  set(TEST_SRCS
      test_real_freq_grid.cpp
      test_real_axis_conv.cpp
      test_real_axis_pi.cpp
      test_real_axis_scr_coulomb.cpp
      test_real_axis_hermiticity.cpp
      test_real_axis_w_lehmann.cpp
     )

  add_executable(test_${TEST_ID} ${TEST_SRCS})

  target_link_libraries(test_${TEST_ID}
    PRIVATE
      catch_main
      utils
      numerics
      fft_lib
      methods_lib
      meanfield
      eri_lib
      scf_lib
      FINUFFT::finufft
  )

  add_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")
  set_tests_properties(test_${TEST_ID}
    PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
endif()
