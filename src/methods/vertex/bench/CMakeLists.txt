# notes/ladder_opt_spec.md increment B0: the standalone SLATE dense-pipeline bench.
# Its own target, deliberately NOT a unit test -- it is a scan harness (minutes to hours
# per design point) and a regression bench for the distributed ladder path. It carries no
# coqui state, so it links only what the distributed dense algebra needs.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(ladder_slate_bench ladder_slate_bench.cpp)
target_link_libraries(ladder_slate_bench utils numerics)
install(TARGETS ladder_slate_bench RUNTIME DESTINATION bin)
