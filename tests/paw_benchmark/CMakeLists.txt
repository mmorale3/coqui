# Phase 4 / Phase 5 PAW external-reference benchmark harness.
#
# Each bench_<name>/ subdirectory becomes one CTest. Tests SKIP at runtime
# (exit 77) when the reference data is missing — that keeps `ctest` honest
# on a fresh clone without pre-failing the suite. Provision references by
# following tests/paw_benchmark/README.md.

set(_harness "${CMAKE_CURRENT_SOURCE_DIR}/run_benchmark.sh")
file(GLOB _bench_dirs RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     "${CMAKE_CURRENT_SOURCE_DIR}/bench_*")

set(_n_registered 0)
foreach(_d IN LISTS _bench_dirs)
    set(_full "${CMAKE_CURRENT_SOURCE_DIR}/${_d}")
    if(IS_DIRECTORY "${_full}")
        set(_test_name "paw_benchmark_${_d}")
        add_test(
            NAME ${_test_name}
            COMMAND bash "${_harness}" "${_full}"
        )
        set_tests_properties(${_test_name} PROPERTIES
            LABELS "paw_benchmark;external"
            SKIP_RETURN_CODE 77
        )
        math(EXPR _n_registered "${_n_registered}+1")
    endif()
endforeach()

if(_n_registered EQUAL 0)
    message(STATUS
        "paw_benchmark: no bench_* directories under "
        "${CMAKE_CURRENT_SOURCE_DIR}; harness installed but no live cases. "
        "See tests/paw_benchmark/README.md to provision references.")
else()
    message(STATUS
        "paw_benchmark: registered ${_n_registered} test(s); "
        "they SKIP at runtime if reference data is absent.")
endif()
