
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID hamiltonian)

set( TEST_SRCS
     test_hamilt.cpp
     test_pseudo.cpp
     test_paw_energy_check.cpp
     test_paw_radial.cpp
     test_paw_onecenter.cpp
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
target_link_libraries(test_${TEST_ID} PRIVATE catch_main utils_f hamilt methods_lib)
#target_link_libraries(test_${TEST_ID} PRIVATE catch_main utils_f hamilt)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# The shm one-body builder (set_H0) runs its H0 build on
# node-root ranks only — a stray global-communicator collective inside that
# call graph (the pre-fff6d4e compute_int_VQ class) DEADLOCKS at np>1 and is
# invisible at the CTEST_NPROC=1 default. Always exercise the [shm_h0] tag at
# 2 ranks, with a TIMEOUT so a regression fails instead of hanging forever.
if(MPIEXEC_MAX_NUMPROCS GREATER 1)
  add_mpi_unit_test(test_${TEST_ID}_np2_shm "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" 2 "[shm_h0]")
  set_tests_properties(test_${TEST_ID}_np2_shm PROPERTIES
                       WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR} TIMEOUT 900)
endif()
