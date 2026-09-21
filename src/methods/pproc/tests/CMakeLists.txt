
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# P24 / G31 (notes/vertex_perf_plan.md): the qp_gaps fine-mesh Wannier interpolation -- the
# coarse-mesh identity gate of pproc_t::interpolate_qp_bands_on_mesh (checkpoint-free, the
# lih222 projector) and the end-to-end qp_gaps driver with qp_gaps_interp = true on a short
# scGW checkpoint. Links methods_lib: the driver needs the scf/gw/eri stack next to pproc_lib.
set(TEST_ID methods_pproc)

add_executable(test_${TEST_ID} test_pproc_qp_gaps.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield methods_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                     TIMEOUT 3600)
