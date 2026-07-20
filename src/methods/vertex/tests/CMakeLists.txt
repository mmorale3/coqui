
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID methods_vertex)

add_executable(test_${TEST_ID} test_vertex_pi.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# ISDF-Vertex Phase 1c: Sigma^C (G3W2) kernel tests
set(TEST_ID methods_vertex_sigma)

add_executable(test_${TEST_ID} test_vertex_sigma.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# ISDF-Vertex Phase 3: conservation validation -- Sigma^C and Pi^C are the two cuts of
# ONE Phi_2^C:  Tr[Sigma^C G] + Tr[Pi^C W] = 0  (notes/conservation_validation.md)
set(TEST_ID methods_vertex_conservation)

add_executable(test_${TEST_ID} test_vertex_conservation.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# ISDF-Vertex Phase 2: C = empty set reproduces plain scGW bit-for-bit (primary gate)
set(TEST_ID methods_vertex_noop)

add_executable(test_${TEST_ID} test_vertex_noop.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# ISDF-Vertex Refinement 2: secondary ISDF basis + Option-A downfold
# (notes/refinement2_optionA.md): transfer algebra, no-leak, secondary-vs-global,
# eta(q, nu) rank scan, conservation identity in the secondary path, end-to-end.
set(TEST_ID methods_vertex_refinement2)

add_executable(test_${TEST_ID} test_vertex_refinement2.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# ISDF-Vertex: IBZ k-point symmetry (notes/vertex_ibz_symmetry.md): leakage
# diagnostic, the sym-vs-nosym gold check, the star-weighted conservation identity
# on the symmetry-reduced mesh, and the C = empty noop on the sym mesh.
set(TEST_ID methods_vertex_ibz)

add_executable(test_${TEST_ID} test_vertex_ibz.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# ISDF-Vertex Refinement 2: W-bar iteration cache (notes/wbar_cache.md): the secondary
# path caches the downfolded rung at update_w time so the scf driver frees dW
# unconditionally; machine-identity vs the legacy retained-dW consumption + memory
# behavior at the retention site.
set(TEST_ID methods_vertex_wcache)

add_executable(test_${TEST_ID} test_vertex_wcache.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# ISDF-Vertex: Wannier-projector generalization of the subspace C
# (notes/wannier_projector_theory.md): degenerate-U bit-identity vs the window path,
# gauge invariance (Phi depends on range(P) only), a genuinely restricting rank-1
# Wannier C + the secondary rotated point selection, and the Loewdin isometry repair.
# Links embed_lib for projector_t (the U source).
set(TEST_ID methods_vertex_wannier)

add_executable(test_${TEST_ID} test_vertex_wannier.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib embed_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# Vertex parallelization MILESTONE 1, item #1: the W-redistribution round-trip
# (notes/vertex_parallelization_M1.md; analysis section 4.2 #4). dW on the RPA (q,t,P,Q)
# grid -> q-owned layout -> back is pure data movement (math::nda::redistribute) and
# must be bit-identical. The infra guarantee the M2 kernel re-parallelization rests on.
set(TEST_ID methods_vertex_wredist)

add_executable(test_${TEST_ID} test_vertex_wredist.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# Vertex parallelization MILESTONE 1, item #2: rank-cap-lift smoke + rank-invariance at
# the kernel entry point (notes/vertex_parallelization_M1.md; analysis section 4.2 #1/#2).
# The Pi^C kernel runs at P > nbnd (toy nbnd = 3) without the reader-side cap abort, and
# the distributed result reproduces the serial result to the FP-reassociation floor.
set(TEST_ID methods_vertex_rankinv)

add_executable(test_${TEST_ID} test_vertex_rankinv.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# Vertex parallelization Impl 2: DISTRIBUTED downfold of dW (no full Np^2 gather)
# (notes/vertex_parallelization_v2_plan.md section 6b). fold_dW_distributed folds the RPA
# (t,P,Q)-distributed dynamic W to the secondary aux (N_m x N_m) via per-(P,Q)-block
# partials + reduce, holding no full Np x Np block. The physics tests never split (P,Q)
# (np_P = np_Q = 1 at Si scale) -- THIS test FORCES a 1x1x2x2 grid (at 4 ranks) and
# compares to the replicated fold_core reference (<= 1e-12; bit-identical at 1 rank).
set(TEST_ID methods_vertex_dfold)

add_executable(test_${TEST_ID} test_vertex_dfold.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
