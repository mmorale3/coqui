
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID methods_eri)

set( TEST_SRCS 
     test_thc.cpp
     test_cholesky.cpp
     test_thc_reader.cpp
     test_chol_reader.cpp
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
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib nda_c h5_c slate)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})


#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

## Add timing routine
#if(USE_DEVICE_COMPILER)
#if(ENABLE_CUDA)
#  set_source_files_properties(time_cholesky.cpp PROPERTIES LANGUAGE CUDA)
#endif()
#endif()

#add_executable(time_cholesky time_cholesky.cpp)
#target_link_libraries(time_cholesky utils numerics meanfield eri_lib nda_c h5_c)

#install(TARGETS time_cholesky
#        RUNTIME DESTINATION bin)
