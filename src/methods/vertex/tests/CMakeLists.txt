# ctest's default 1500 s TIMEOUT is too short for this directory: several vertex tests
# (e.g. test_methods_vertex, test_methods_vertex_gf2xchk) run for tens of minutes to hours,
# so the stock timeout reports SPURIOUS failures (worse on slower or shared machines).
# The limit is raised here for every test in the directory.
# NOTE a per-test TIMEOUT property OVERRIDES ctest --timeout, so setting it here is the only
# setting that actually takes effect.
set(COQUI_VERTEX_TEST_TIMEOUT 14400 CACHE STRING "ctest timeout (s) for the vertex tests")


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID methods_vertex)

add_executable(test_${TEST_ID} test_vertex_pi.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Vertex: Sigma^C (G3W2) kernel tests
set(TEST_ID methods_vertex_sigma)

add_executable(test_${TEST_ID} test_vertex_sigma.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Vertex conservation: Sigma^C and Pi^C are the two cuts of
# ONE Phi_2^C:  Tr[Sigma^C G] + Tr[Pi^C W] = 0
set(TEST_ID methods_vertex_conservation)

add_executable(test_${TEST_ID} test_vertex_conservation.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Vertex: C = empty set reproduces plain scGW bit-for-bit
set(TEST_ID methods_vertex_noop)

add_executable(test_${TEST_ID} test_vertex_noop.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Vertex ladder: the knobs OFF/inert (pol_vertex none-or-empty-window, cvv_rspace_tol set but
# unused without div_treatment = "cvv") reproduce the plain scGW loop bit-for-bit, in both
# production wirings; plus the pair-space ladder diagnostics and eps_M readouts.
set(TEST_ID methods_vertex_scgwt_noop)

add_executable(test_${TEST_ID} test_vertex_scgwt_noop.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib scr_coulomb_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# CVV head of the screened interaction: the R-space engine + covariant velocity (toy
# tight-binding oracle, hermiticity/trev identities, FD-in-k cross-check, LiH-222 build
# smoke) and the head tensor checks.
set(TEST_ID methods_scr_cvv)

add_executable(test_${TEST_ID} test_scr_cvv.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib scr_coulomb_lib pproc_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Vertex: secondary ISDF basis with the downfolded rung: transfer algebra, no-leak,
# secondary-vs-global, eta(q, nu) rank scan, conservation identity in the secondary path,
# end-to-end.
set(TEST_ID methods_vertex_refinement2)

add_executable(test_${TEST_ID} test_vertex_refinement2.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Vertex: IBZ k-point symmetry: leakage diagnostic, the sym-vs-nosym gold check, the
# star-weighted conservation identity on the symmetry-reduced mesh, and the C = empty noop
# on the sym mesh.
set(TEST_ID methods_vertex_ibz)

add_executable(test_${TEST_ID} test_vertex_ibz.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Vertex: W-bar iteration cache: the secondary path caches the downfolded rung at
# update_w time so the scf driver frees dW unconditionally; machine-identity vs the
# retained-dW consumption + memory behavior at the retention site.
set(TEST_ID methods_vertex_wcache)

add_executable(test_${TEST_ID} test_vertex_wcache.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Vertex: Wannier-projector generalization of the subspace C: degenerate-U
# bit-identity vs the window path, gauge invariance (Phi depends on range(P) only), a
# genuinely restricting rank-1 Wannier C + the secondary rotated point selection, and the
# Loewdin isometry repair. Links embed_lib for projector_t (the U source).
set(TEST_ID methods_vertex_wannier)

add_executable(test_${TEST_ID} test_vertex_wannier.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib embed_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Vertex parallel infrastructure: the W-redistribution round trip. dW on the RPA (q,t,P,Q)
# grid -> q-owned layout -> back is pure data movement (math::nda::redistribute) and
# must be bit-identical; plus the reduce-scatter of partial Pi^C contributions.
set(TEST_ID methods_vertex_wredist)

add_executable(test_${TEST_ID} test_vertex_wredist.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Vertex rank-invariance at the kernel entry point: the Pi^C kernel runs at P > nbnd
# (toy nbnd = 3) without the reader-side rank-cap abort, and the distributed Pi^C and
# Sigma^C reproduce the serial results to the FP-reassociation floor.
set(TEST_ID methods_vertex_rankinv)

add_executable(test_${TEST_ID} test_vertex_rankinv.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# DISTRIBUTED downfold of dW (no full Np^2 gather). fold_dW_distributed folds the RPA
# (t,P,Q)-distributed dynamic W to the secondary aux (N_m x N_m) via per-(P,Q)-block
# partials + reduce, holding no full Np x Np block. The physics tests typically do not split
# (P,Q) (np_P = np_Q = 1) -- THIS test FORCES a 1x1x2x2 grid (at 4 ranks) and compares to
# the replicated fold_core reference (<= 1e-12; bit-identical at 1 rank).
set(TEST_ID methods_vertex_dfold)

add_executable(test_${TEST_ID} test_vertex_dfold.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Static vertex (B-S): end-to-end on LiH-222 (nosym) -- the whole loop with
# vertex_rung = "static": W0 built inside update_w, P stays RPA, Sigma carries BOTH cuts.
set(TEST_ID methods_vertex_static_e2e)

add_executable(test_${TEST_ID} test_vertex_static_e2e.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Static vertex (B-S): the FUNCTIONAL-DERIVATIVE ORACLE for Formulation B-S. Verifies
# "one total derivative of one Phi":
#   d/dl Phi[G + l dG] = T[Sigma^{C,x}, P_C dG P_C] + T[Sigma^{C,r}, dG]
# on a reality-symmetric toy with a NON-Hermitian dG, plus the drop / sign-flip /
# sandwich-convention positive controls. Pure toy: no MF, no scGW, seconds to run.
set(TEST_ID methods_vertex_fdoracle)

add_executable(test_${TEST_ID} test_vertex_fdoracle.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Static vertex (B-S/B-L): the W0[G] rung infrastructure. Pins (i) the plain-GW
# SELF-SLICE identity W0 == the i.nu = 0 row of the fully solved W, (ii) the three
# q->0 head policies at i.nu = 0, (iii) the update_w seam (a static-rung vertex gets
# W0 built and leaves the RPA W bit-identical; no bootstrap, no Pi^C injection) and
# (iv) the distributed W0bar fold against a replicated t W0 t^dag reference.
set(TEST_ID methods_vertex_w0)

add_executable(test_${TEST_ID} test_vertex_w0.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# The FACTORIZED equal-time dynamic-rung polarization (vertex_pi::pi_dyn_factorized)
# against the R0-contracted output of the full Pi^C kernel, on identical inputs: pole-free
# and dynamic-rung checks, a rung with no pair symmetry (discriminates a transposed rung),
# rung linearity, rank-split invariance, and the structural assertion that the factorized
# primitive never initializes the aux DLR pole basis. Pure toy: seconds to run.
set(TEST_ID methods_vertex_pibardynfact)

add_executable(test_${TEST_ID} test_vertex_pibardynfact.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Vertex self-consistency diagnostic under k-point symmetry: the little-group residual
# of the SELF-CONSISTENT G with the vertex OFF (baseline control), the same with the vertex
# ON over several iterations, and a sym-vs-nosym e_corr TRAJECTORY over more iterations
# than the 2-iteration gold check of test_vertex_ibz.
set(TEST_ID methods_vertex_scdiag)

add_executable(test_${TEST_ID} test_vertex_scdiag.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# The DISTRIBUTED (SLATE) solve path of the pair-space ladder behind ladder_solve_grid = g,
# compared with the per-rank path on one fixture state -- the whalf evaluator,
# ladder_whalf_gate, the lambda_max watchdog and the E-leg (orbital) family, each at g >= 2
# against g = 1. Needs >= 2 ranks (it SKIPS at np = 1, which is what ctest's default
# CTEST_NPROC = 1 gives); links scr_coulomb_lib for the readout-vertex plumbing that
# carries the knob.
set(TEST_ID methods_vertex_ladder_slate)

add_executable(test_${TEST_ID} test_vertex_ladder_slate.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib scr_coulomb_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Vertex: ABSOLUTE normalization/sign cross-check of Sigma^C against CoQui's own
# bare second-order exchange (GF2 exchange channel, pyscf-pinned). With C = full band
# range and W -> Z the vertex functional IS the bare second-order exchange, so the two
# self-energies must be identical. It is the only test that constrains the ABSOLUTE
# prefactor (powers of 1/N_k, 1/beta) and the overall sign.
set(TEST_ID methods_vertex_gf2xchk)

add_executable(test_${TEST_ID} test_vertex_gf2xchk.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib gf2_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Ward legs: the discrete-Ward leg vertex ALGEBRA (ward_legs.hpp) on exact rational data --
# closed-form pole sums vs dense Matsubara oracles, the q = 0 matrix telescoping, the
# Z-link + positivity, R = 0 bitwise, the window complement identity. Pure toy.
set(TEST_ID methods_vertex_ward_legs)

add_executable(test_${TEST_ID} test_vertex_ward_legs.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Ward legs: the WIRING of the discrete-Ward leg vertex (pol_vertex_legs = "ward") into
# the ladder readout on lih222 -- the bitwise leg at Sigma = 0 (iteration 1), the
# report-only loop invariance, the zero-rung normalization pin against eval_pol_pi0, and
# the fixture telescoping / fit meters.
set(TEST_ID methods_vertex_ward_wiring)

add_executable(test_${TEST_ID} test_vertex_ward_wiring.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib scr_coulomb_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Dynamic-rung BSE: the two-family DLR algebra of the resummed DYNAMIC-rung BSE
# (dynbse.hpp) against a dense Matsubara BSE oracle on an exact rational toy -- the static
# limit, the first iterate and the resummed vertex, at inu = 0 and inu != 0; then the
# fitted-residue (shared aux grid) pathway. Pure toy.
set(TEST_ID methods_vertex_dynbse)

add_executable(test_${TEST_ID} test_vertex_dynbse.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# Dynamic-rung BSE: the production driver's checks on lih222 -- the THC rung
# operator vs Kbig, the static limit vs the static pair-space ladder resolvent, the one
# bare dynamic rung vs pi_c_accumulate_w, GMRES vs Neumann, Hermiticity.
set(TEST_ID methods_vertex_dynbse_gate)

add_executable(test_${TEST_ID} test_vertex_dynbse_gate.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib scr_coulomb_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})

# The vertex on the DEVICE path reproduces the host path. Only meaningful (and only
# compiled) with a device build.
if(ENABLE_DEVICE)
set(TEST_ID methods_vertex_device)

add_executable(test_${TEST_ID} test_vertex_device.cpp)
target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield eri_lib hf_lib
                      vertex_lib gw_lib scf_lib arch_lib)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
                    TIMEOUT ${COQUI_VERTEX_TEST_TIMEOUT})
endif()
