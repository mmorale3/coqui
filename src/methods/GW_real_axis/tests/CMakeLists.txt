
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# RW-1: only the W-chain subset of the branch's 23 test files is ported (the rest
# exercise Sigma / HF / Dyson-G / the SCF driver / the dispatcher / the QP solver,
# none of which are in the ported slice). See notes/rw1_port_report.md for the
# per-file deviations.
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
