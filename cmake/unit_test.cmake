
# Runs unit tests
FUNCTION( ADD_UNIT_TEST TESTNAME TEST_BINARY )
    MESSAGE( STATUS "Adding test ${TESTNAME}")
    ADD_TEST(NAME ${TESTNAME} COMMAND ${TEST_BINARY} ${ARGN})
    SET_TESTS_PROPERTIES( ${TESTNAME} PROPERTIES ENVIRONMENT OMP_NUM_THREADS=1 )
    SET_PROPERTY(TEST ${TESTNAME} APPEND PROPERTY LABELS "unit")
ENDFUNCTION()

FUNCTION( ADD_MPI_UNIT_TEST TESTNAME TEST_BINARY PROC_COUNT )
    MESSAGE( STATUS "Adding test ${TESTNAME}")
    ADD_TEST(NAME ${TESTNAME} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${PROC_COUNT} ${MPIEXEC_PREFLAGS} ${TEST_BINARY} ${ARGN})
    # OMPI_MCA_hwloc_base_binding_policy=none makes "ctest at MKL_NUM_THREADS=N" a
    # REPRODUCIBLE configuration (notes/coqui_threading_spec.md rev 2 gates; T-3 needs it too).
    #
    # Without it, OpenMPI 4.1 binds a low-rank-count job to a single core -- verified:
    #     mpiexec -n 1 --oversubscribe --report-bindings hostname
    #     -> MCW rank 0 bound to socket 0[core 0[hwt 0]]: [B/././. ...]
    # so a threaded BLAS layer puts N threads on ONE core. On jobs 6895143 / 6895484 that
    # produced a UNIFORM 4-22x slowdown across the whole suite and three spurious ctest
    # Timeouts (test_methods_vertex, _refinement2, qp_map_ab) that looked like correctness
    # failures but were pure oversubscription. The same binary at the same MKL_NUM_THREADS
    # was 1.35x FASTER than serial on the production fixture, where each rank owned 4 real
    # cores -- opposite sign, purely from binding.
    #
    # Set as an environment property rather than an mpiexec flag so it stays portable:
    # non-OpenMPI launchers simply ignore the variable, whereas --bind-to none would be a
    # hard error under MPICH.
    # TIMEOUT 7200: ctest's 1500 s default silently truncated four VALID tests at exactly
    # 1500.0x s in job 6895598 (qp_map_ab, hamiltonian, methods_eri, methods_gw) -- a
    # harness ceiling that reads as a correctness failure. Per-test CMakeLists may still
    # raise it (e.g. qpgw_bse sets 14400); a later set_tests_properties wins.
    SET_TESTS_PROPERTIES( ${TESTNAME} PROPERTIES
                          ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_MCA_hwloc_base_binding_policy=none"
                          TIMEOUT 7200 )
    SET_PROPERTY(TEST ${TESTNAME} APPEND PROPERTY LABELS "unit")
ENDFUNCTION()

