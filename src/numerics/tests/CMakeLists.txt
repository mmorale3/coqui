#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID nda_functions)

set( TEST_SRCS
     test_nda_functions.cpp
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
target_link_libraries(test_${TEST_ID} PRIVATE catch_main utils numerics)

add_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# increment TC-1 (notes/tc_coqui_impl_spec.md): the offline tilted-contour builder.
# Pure math (nda + LAPACK), no MPI, no fixture -- it builds and runs in every
# configuration. Gate TC-1-a reads the campaign reference JSON produced by
# tc_validation/tests/export_tc1_reference.py.
set(TEST_ID tilted_contour)

add_executable(test_${TEST_ID} test_tilted_contour.cpp)
target_link_libraries(test_${TEST_ID} PRIVATE catch_main utils numerics)

add_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                     TIMEOUT 3600)

