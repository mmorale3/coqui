
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID methods_scf)

add_executable(test_${TEST_ID} test_simple_dyson.cpp test_scf_common.cpp test_qp_maps_matsubara.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib scf_lib)


add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# RW-2 (notes/rw_real_axis_w_spec.md): gate RW-2-a, the TOY gate of the spectral-quadrature
# W^c pole representation. Pure math on a closed-form Drude W^c -- no fixture, no FINUFFT --
# so it builds and runs in every configuration.
set(TEST_ID methods_wc_spectral)

add_executable(test_${TEST_ID} test_wc_spectral.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# increment TC-2 (notes/tc_coqui_impl_spec.md, spec M3): P on the tilted contour --
# the complex-time ISDF kernel against the production imaginary-axis Pi, at t = -i tau
# (the contraction) and through the tilted transform at z = i nu (the M3 gate), plus
# the band-truncation on/off check. Separate executable so the existing suites keep
# their exact tallies.
set(TEST_ID methods_tc_contour)

add_executable(test_${TEST_ID} test_methods_tc_contour.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      gw_lib scf_lib scr_coulomb_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT 14400)

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

# Project 2 increment Q4 (notes/q4_edmft_skeleton_spec.md): the EDMFT skeleton -- the
# C = empty identity of the qpGW+BSE lattice stage under screen_type = "gw_edmft" with a
# bosonic projector, and the eq-7 ladder DC (P^lad_loc). Links methods_lib: it needs the
# embedding projector alongside the scf/vertex/scr_coulomb stack.
set(TEST_ID methods_qpgw_edmft)

add_executable(test_${TEST_ID} test_methods_qpgw_edmft.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield methods_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT 14400)

# Project 2 increment Q4-C3b (notes/q4_c3b_orbital_ladder_dc_spec.md): the eq-7 ladder DC
# proper -- the orbital/chi-convention local part of the lattice ladder. Separate executable
# so the Q3/Q4 suites keep their exact tallies (gate G1's no-disturbance discipline).
set(TEST_ID methods_qpgw_c3b)

add_executable(test_${TEST_ID} test_methods_qpgw_c3b.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield methods_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT 14400)

# Project 2 increment Q5 (notes/q5_option2_outer_loop_spec.md): the Option-2 outer loop --
# the source-swap identity of the external-G injection (Q5-g1) and the restart composition
# that IS the C = empty-set Option-1-vs-Option-2 statement (Q5-g2). Separate executable so
# the Q3/Q4 suites keep their exact tallies.
set(TEST_ID methods_qpgw_q5)

add_executable(test_${TEST_ID} test_methods_qpgw_q5.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      gw_lib scf_lib vertex_lib scr_coulomb_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT 14400)

# Project 2 increment Q6 (notes/q6_diagnostics_closeout_spec.md): the diagnostics closeout --
# the compressibility seam (§1.2, scope-reduced: const_mu does not reach the qp path), the
# map-stage lineshape meter (§1.3) and the persisted injection meters + consolidated summary
# line (§1.4). Separate executable so the Q3/Q4/Q5 suites keep their exact tallies.
set(TEST_ID methods_qpgw_q6)

add_executable(test_${TEST_ID} test_methods_qpgw_q6.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      gw_lib scf_lib vertex_lib scr_coulomb_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT 14400)
