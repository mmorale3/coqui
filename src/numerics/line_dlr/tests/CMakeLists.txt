# session S1 of notes/line_gw_cpp_plan.md: the line-DLR numerics (real-pole bases on the tilted line, time rays).
# Pure math (nda + LAPACK + h5), no MPI, no fixture. Reference: tests/unit_test_files/gw_line/line_dlr_ref.h5,
# written by coqui/cayley/scripts/gen_line_dlr_ref.py.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID line_dlr)

add_executable(test_${TEST_ID} test_line_dlr.cpp)
target_link_libraries(test_${TEST_ID} PRIVATE catch_main utils numerics)

add_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# session S2: Cayley moments, block-Toeplitz upfolding, Lehmann G, chemical potential (numerics/line_dlr/cayley.hpp).
# Reference: tests/unit_test_files/gw_line/si222_moments_k0.h5, written by coqui/cayley/scripts/gen_cayley_ref.py.
set(TEST_ID cayley)

add_executable(test_${TEST_ID} test_cayley.cpp)
target_link_libraries(test_${TEST_ID} PRIVATE catch_main utils numerics)

add_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

# session S9b: the bosonic response closure (numerics/line_dlr/response_closure.hpp: odd fit, NNLS, MB multi-scale closure,
# K rule, optical quantities). Reference: tests/unit_test_files/gw_line/response_closure_ref.h5, written by
# coqui/cayley/scripts/gen_response_closure_ref.py from the python design study (scripts/closure_design/).
set(TEST_ID response_closure)

add_executable(test_${TEST_ID} test_response_closure.cpp)
target_link_libraries(test_${TEST_ID} PRIVATE catch_main utils numerics)

add_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
