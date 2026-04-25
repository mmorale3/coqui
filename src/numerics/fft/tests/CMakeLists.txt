
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID math_fft_nda)

set( TEST_SRCS 
     test_fft_nda.cpp
   )

add_executable(test_${TEST_ID} ${TEST_SRCS} )
target_link_libraries(test_${TEST_ID} PRIVATE catch_main utils fft_lib)

add_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

if(ENABLE_FINUFFT)

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

  set(TEST_ID math_finufft_nda)

  set(TEST_SRCS
    test_finufft_nda.cpp
  )

  add_executable(test_${TEST_ID} ${TEST_SRCS})

  # Link against the same base libraries as the fft test, plus finufft.
  target_link_libraries(test_${TEST_ID}
    PRIVATE
      catch_main
      utils
      fft_lib          # brings in configuration.hpp, nda, IO helpers, etc.
      FINUFFT::finufft # from cmake/FindFINUFFT.cmake
  )

  add_unit_test(test_${TEST_ID}
    "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")

  set_tests_properties(test_${TEST_ID}
    PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

endif() # ENABLE_FINUFFT
