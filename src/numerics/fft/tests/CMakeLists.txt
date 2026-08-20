
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID math_fft_nda)

set( TEST_SRCS 
     test_fft_nda.cpp
   )

if(USE_DEVICE_COMPILER)
if(ENABLE_CUDA)
  foreach( X ${TEST_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()
elseif(ENABLE_HIP)
  foreach( X ${TEST_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE HIP)
  endforeach()
endif()
endif()

add_executable(test_${TEST_ID} ${TEST_SRCS} )
target_link_libraries(test_${TEST_ID} PRIVATE catch_main utils fft_lib)

add_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

if(ENABLE_FINUFFT)

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

  set(TEST_ID math_finufft_nda)

  # RW-1 deviation: test_finufft_t3.cpp is NOT ported (type-3 has no consumer in the
  # W chain -- survey section 5 "Type-3 is unused by the physics").
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
