
# Runs unit tests
# Prefix command for the NON-MPI unit tests (they still initialize MPI). Empty by default; under a slurm
# allocation set it to a launcher, e.g. -DCOQUI_UNIT_TEST_LAUNCHER="/path/srun_mpiexec.sh;-n;1", so that
# they run as their own step with a PMI environment instead of directly on the node that runs ctest.
SET( COQUI_UNIT_TEST_LAUNCHER "" CACHE STRING "prefix command (;-separated list) for the non-MPI unit tests" )

FUNCTION( ADD_UNIT_TEST TESTNAME TEST_BINARY )
    MESSAGE( STATUS "Adding test ${TESTNAME}")
    ADD_TEST(NAME ${TESTNAME} COMMAND ${COQUI_UNIT_TEST_LAUNCHER} ${TEST_BINARY} ${ARGN})
    SET_TESTS_PROPERTIES( ${TESTNAME} PROPERTIES ENVIRONMENT OMP_NUM_THREADS=1 )
    SET_PROPERTY(TEST ${TESTNAME} APPEND PROPERTY LABELS "unit")
ENDFUNCTION()

FUNCTION( ADD_MPI_UNIT_TEST TESTNAME TEST_BINARY PROC_COUNT )
    MESSAGE( STATUS "Adding test ${TESTNAME}")
    ADD_TEST(NAME ${TESTNAME} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${PROC_COUNT} ${MPIEXEC_PREFLAGS} ${TEST_BINARY} ${ARGN})
    # OMPI_MCA_hwloc_base_binding_policy=none keeps threaded runs (e.g. ctest at
    # MKL_NUM_THREADS=N) reproducible. Without it, OpenMPI 4.1 binds a low-rank-count job to a
    # single core:
    #     mpiexec -n 1 --oversubscribe --report-bindings hostname
    #     -> MCW rank 0 bound to socket 0[core 0[hwt 0]]: [B/././. ...]
    # so a threaded BLAS layer puts N threads on ONE core, which slows the whole suite down
    # severalfold and can push tests past their timeout.
    #
    # Set as an environment property rather than an mpiexec flag so it stays portable:
    # non-OpenMPI launchers simply ignore the variable, whereas --bind-to none would be a
    # hard error under MPICH.
    # TIMEOUT 7200: ctest's 1500 s default is too short for several of the larger MPI tests.
    # Per-test CMakeLists may still raise it (e.g. qpgw_bse sets 14400); a later
    # set_tests_properties wins.
    SET_TESTS_PROPERTIES( ${TESTNAME} PROPERTIES
                          ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_MCA_hwloc_base_binding_policy=none"
                          TIMEOUT 7200 )
    SET_PROPERTY(TEST ${TESTNAME} APPEND PROPERTY LABELS "unit")
ENDFUNCTION()

