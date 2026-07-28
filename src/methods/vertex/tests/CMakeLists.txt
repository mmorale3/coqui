# ctest's default 1500 s TIMEOUT is too tight for this directory: test_methods_vertex alone
# takes ~1806 s uncontended on an M-series laptop, so the suite reports a SPURIOUS failure
# with the stock timeout (and worse on anything slower, or when another job shares the box).
# Raise it here rather than per-test so a slow machine does not report phantom regressions.
# 2026-07-28: raised again to 14400 s -- test_methods_vertex_gf2xchk exceeded 5400 s while still
# working (it had already passed its key assertion, lambda = 1.0000000000 with residual 9.7e-13).
# NOTE a per-test TIMEOUT property OVERRIDES ctest --timeout, so raising it here is the only
# thing that actually takes effect.
set(COQUI_VERTEX_TEST_TIMEOUT 14400 CACHE STRING "ctest timeout (s) for the vertex tests")


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID methods_vertex)

add_executable(test_${TEST_ID} test_vertex_pi.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# ISDF-Vertex Phase 1c: Sigma^C (G3W2) kernel tests
set(TEST_ID methods_vertex_sigma)

add_executable(test_${TEST_ID} test_vertex_sigma.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# ISDF-Vertex Phase 3: conservation validation -- Sigma^C and Pi^C are the two cuts of
# ONE Phi_2^C:  Tr[Sigma^C G] + Tr[Pi^C W] = 0  (notes/conservation_validation.md)
set(TEST_ID methods_vertex_conservation)

add_executable(test_${TEST_ID} test_vertex_conservation.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# ISDF-Vertex Phase 2: C = empty set reproduces plain scGW bit-for-bit (primary gate)
set(TEST_ID methods_vertex_noop)

add_executable(test_${TEST_ID} test_vertex_noop.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# ISDF-Vertex Refinement 2: secondary ISDF basis + Option-A downfold
# (notes/refinement2_optionA.md): transfer algebra, no-leak, secondary-vs-global,
# eta(q, nu) rank scan, conservation identity in the secondary path, end-to-end.
set(TEST_ID methods_vertex_refinement2)

add_executable(test_${TEST_ID} test_vertex_refinement2.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# ISDF-Vertex: IBZ k-point symmetry (notes/vertex_ibz_symmetry.md): leakage
# diagnostic, the sym-vs-nosym gold check, the star-weighted conservation identity
# on the symmetry-reduced mesh, and the C = empty noop on the sym mesh.
set(TEST_ID methods_vertex_ibz)

add_executable(test_${TEST_ID} test_vertex_ibz.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# ISDF-Vertex Refinement 2: W-bar iteration cache (notes/wbar_cache.md): the secondary
# path caches the downfolded rung at update_w time so the scf driver frees dW
# unconditionally; machine-identity vs the legacy retained-dW consumption + memory
# behavior at the retention site.
set(TEST_ID methods_vertex_wcache)

add_executable(test_${TEST_ID} test_vertex_wcache.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

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
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Vertex parallelization MILESTONE 1, item #1: the W-redistribution round-trip
# (notes/vertex_parallelization_M1.md; analysis section 4.2 #4). dW on the RPA (q,t,P,Q)
# grid -> q-owned layout -> back is pure data movement (math::nda::redistribute) and
# must be bit-identical. The infra guarantee the M2 kernel re-parallelization rests on.
set(TEST_ID methods_vertex_wredist)

add_executable(test_${TEST_ID} test_vertex_wredist.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Vertex parallelization MILESTONE 1, item #2: rank-cap-lift smoke + rank-invariance at
# the kernel entry point (notes/vertex_parallelization_M1.md; analysis section 4.2 #1/#2).
# The Pi^C kernel runs at P > nbnd (toy nbnd = 3) without the reader-side cap abort, and
# the distributed result reproduces the serial result to the FP-reassociation floor.
set(TEST_ID methods_vertex_rankinv)

add_executable(test_${TEST_ID} test_vertex_rankinv.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

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
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# ISDF-Vertex self-consistency diagnostic (missing controls for the Si M8 divergence):
# the little-group residual of the SELF-CONSISTENT G with the vertex OFF (baseline
# control the IBZ test never measured), the same with the vertex ON over several
# iterations, and a sym-vs-nosym e_corr TRAJECTORY (the gold check only looks at
# 2 iterations with a 25%-of-shift tolerance).
set(TEST_ID methods_vertex_scdiag)

add_executable(test_${TEST_ID} test_vertex_scdiag.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# ISDF-Vertex: ABSOLUTE normalization/sign cross-check of Sigma^C against CoQui's own
# bare second-order exchange (GF2 exchange channel, pyscf-pinned). With C = full band
# range and W -> Z the vertex functional IS the bare second-order exchange, so the two
# self-energies must be identical. This is the check verified-cut-verdicts flagged as
# "recommended before production" and that was never run: it is the only test that
# constrains the ABSOLUTE prefactor (powers of 1/N_k, 1/beta) and the overall sign.
set(TEST_ID methods_vertex_gf2xchk)

add_executable(test_${TEST_ID} test_vertex_gf2xchk.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib gf2_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})
