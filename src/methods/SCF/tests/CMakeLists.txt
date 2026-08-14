
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID methods_scf)

add_executable(test_${TEST_ID} test_simple_dyson.cpp test_scf_common.cpp test_qp_maps_matsubara.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib scf_lib)


add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# Project 2 increment Q2: the qp_map A/B surrogate-spread gate (full qp-scf
# loops on real fixtures -- separate executable so ctest can target it).
set(TEST_ID methods_qp_map_ab)

add_executable(test_${TEST_ID} test_qp_map_ab.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# Project 2 increment Q3 (notes/q3_bse_tier_spec.md): the BSE (ladder) polarization tier
# in the qpGW loop -- the noop legs, the loop-vs-readout eps_M identity, the lambda_max
# watchdog and the s = 1 anchor pin on the qpGW state. Links vertex/scr_coulomb: the
# gates run on the production ladder instance.
set(TEST_ID methods_qpgw_bse)

add_executable(test_${TEST_ID} test_methods_qpgw_bse.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      gw_lib scf_lib vertex_lib scr_coulomb_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT 14400)
