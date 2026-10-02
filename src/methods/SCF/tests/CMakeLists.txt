
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID methods_scf)

add_executable(test_${TEST_ID} test_simple_dyson.cpp test_scf_common.cpp test_qp_maps_matsubara.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib scf_lib)


add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# The spectral-quadrature pole representation of W^c, checked against a closed-form
# Lorentzian-broadened W^c. Pure math -- no fixture, no FINUFFT -- so it builds and runs in
# every configuration.
set(TEST_ID methods_wc_spectral)

add_executable(test_${TEST_ID} test_wc_spectral.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# The small-q fit of epsilon_inf (eps_inf_fit = true), the pure function of
# methods/scr_coulomb/eps_inf_fit.hpp alone -- synthetic eps(q) = eps_inf + A q^2 (+ B q^4)
# recovered to round-off, q = 0 excluded, the degenerate inputs refused. Header-only math:
# no fixture, no MPI, builds and runs in every configuration.
set(TEST_ID methods_eps_inf_fit)

add_executable(test_${TEST_ID} test_eps_inf_fit.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# P on the tilted contour: the complex-time ISDF kernel against the production
# imaginary-axis Pi, at t = -i tau (the contraction) and through the tilted transform at
# z = i nu, plus the band-truncation on/off check.
set(TEST_ID methods_tc_contour)

add_executable(test_${TEST_ID} test_methods_tc_contour.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      gw_lib scf_lib scr_coulomb_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT 14400)

# W^c on the tilted line and the contour-deformation Sigma^c assembly: the single-pole
# Lehmann sign check (closed form, machine precision), the Dyson chain against an
# independently generated RPA W^c reference, dense == warm-started GMRES, and the multipole
# assembly against its exact Lehmann pole sum. Pure math + a JSON reference: no fixture,
# no MPI, builds and runs in every configuration.
set(TEST_ID methods_tc_wline)

add_executable(test_${TEST_ID} test_methods_tc_wline.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT 3600)

# The quasiparticle maps of the qpGW loop compared on the same fixtures (full qp-scf
# loops; the band-edge spread between the static maps is reported).
set(TEST_ID methods_qp_map_ab)

add_executable(test_${TEST_ID} test_qp_map_ab.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# The BSE (ladder) polarization in the qpGW loop -- the no-op legs, the loop-vs-readout
# eps_M identity, the lambda_max watchdog and the one-rung anchor check on the qpGW state.
# Links vertex/scr_coulomb: the checks run on the production ladder instance.
set(TEST_ID methods_qpgw_bse)

add_executable(test_${TEST_ID} test_methods_qpgw_bse.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      gw_lib scf_lib vertex_lib scr_coulomb_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT 14400)

# qpGW+BSE as the lattice stage of GW+EDMFT: the identity with the plain RPA-screened run
# when no local polarizabilities are set (screen_type = "gw_edmft" with a bosonic
# projector), and the local ladder polarization P^lad_loc for the bosonic double counting.
# Links methods_lib: it needs the embedding projector alongside the scf/vertex/scr_coulomb
# stack.
set(TEST_ID methods_qpgw_edmft)

add_executable(test_${TEST_ID} test_methods_qpgw_edmft.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield methods_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT 14400)

# The ladder double counting in the orbital (chi-convention) basis: the local part of the
# lattice ladder entering the bosonic double counting.
set(TEST_ID methods_qpgw_c3b)

add_executable(test_${TEST_ID} test_methods_qpgw_c3b.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield methods_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT 14400)

# The external-G injection of the qpGW loop (greens_func_source): the source-swap
# identity, and the restart composition that shows the "option1" and "option2" outer loops
# agree when no impurity corrections are present.
set(TEST_ID methods_qpgw_q5)

add_executable(test_${TEST_ID} test_methods_qpgw_q5.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      gw_lib scf_lib vertex_lib scr_coulomb_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT 14400)

# qpGW diagnostics: the compressibility check (fixed-Sigma dN/dmu against the q -> 0
# charge response), the map-stage lineshape meter, and the ladder injection meters
# persisted in the checkpoint.
set(TEST_ID methods_qpgw_q6)

add_executable(test_${TEST_ID} test_methods_qpgw_q6.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      gw_lib scf_lib vertex_lib scr_coulomb_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT 14400)
