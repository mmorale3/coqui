
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

if(ENABLE_FINUFFT)
  set(TEST_ID methods_gw_real_axis)

  set(TEST_SRCS
      test_real_freq_grid.cpp
      test_real_axis_conv.cpp
      test_real_axis_dyson.cpp
      test_real_axis_pi.cpp
      test_real_axis_sigma.cpp
      test_real_axis_gw.cpp
      test_real_axis_driver.cpp
      test_real_axis_sigma_x.cpp
      test_real_axis_dyson_G.cpp
      test_real_axis_scf.cpp
      test_real_axis_thc.cpp
      test_real_axis_xvalidate.cpp
      test_real_axis_div_utils.cpp
      test_real_axis_scr_coulomb.cpp
      test_real_axis_dyson_t.cpp
      test_real_axis_scf_loop.cpp
     )

  add_executable(test_${TEST_ID} ${TEST_SRCS})

  target_link_libraries(test_${TEST_ID}
    PRIVATE
      catch_main
      utils
      numerics
      fft_lib
      gw_real_axis_lib
      meanfield
      eri_lib
      scf_lib
      FINUFFT::finufft
  )

  add_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")
  set_tests_properties(test_${TEST_ID}
    PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
endif()
