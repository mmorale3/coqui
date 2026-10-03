# sessions S3-S5 of notes/line_gw_cpp_plan.md: GW_line kernels on the THC fixtures (qe_lih222, qe_si211).
# V1: Pi(q, zeta) on the line vs the Casida transition sum computed in the test.
# V2: W at the bosonic nodes, symmetric fit, vs Dyson/fit of gathered matrices, CoQui's W^c(q, i nu) (scr_coulomb_t) and Casida.
# V0: V_H + Sigma_x vs CoQui hf_t; V3: Sigma_c on the line vs the exact Casida Sigma_c and CoQui iteration-1 Sigma_c(i w_n).
# S7b [time_id]: Pi and Sigma on the time-node ID grids vs the GL rays (<= 10 eps) and vs Casida.
# S7c [lehmann]: factorized (Lehmann) residues v v^dagger vs the matrix-coefficient form (propagators, Pi, Sigma).
# device: HOST vs DEVICE A/B of every kernel ([device], device builds only); [.bench]: hidden one-pass benchmark.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID gw_line_kernels)

add_executable(test_${TEST_ID} test_gw_line_kernels.cpp test_gw_line_w.cpp test_gw_line_sigma.cpp
                              test_gw_line_time_id.cpp test_gw_line_device.cpp test_gw_line_lehmann.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib gw_lib scf_lib scr_coulomb_lib
                      gw_line_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# S6: closure (toy), driver checkpoint/restart and python parity on lih222 (gw_line_scf).
set(TEST_ID gw_line_scf)
add_executable(test_${TEST_ID} test_gw_line_scf.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib gw_line_lib)
add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
