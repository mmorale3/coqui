# session S3 of notes/line_gw_cpp_plan.md: GW_line kernels (propagators, polarization) on the THC fixtures.
# V1: Pi(q, zeta) on the line vs the Casida transition sum computed in the test (qe_lih222, qe_si211).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID gw_line_kernels)

add_executable(test_${TEST_ID} test_gw_line_kernels.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib gw_line_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
