#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

configure_file(${PROJECT_SOURCE_DIR}/src/config.h.cmake.in
  ${PROJECT_BINARY_DIR}/src/config.h)

include(${PROJECT_SOURCE_DIR}/cmake/unit_test.cmake)

include_directories("${PROJECT_SOURCE_DIR}/extern/cxxopts")
include_directories("${PROJECT_SOURCE_DIR}/extern/mpi_wrapper")

add_subdirectory(IO)
add_subdirectory(arch)
add_subdirectory(utilities)
add_subdirectory(numerics)
add_subdirectory(mean_field)
add_subdirectory(hamiltonian)
add_subdirectory(orbitals)
add_subdirectory(wannier)
add_subdirectory(methods)

add_subdirectory(main)

if(COQUI_PYTHON_SUPPORT)
  add_subdirectory(python)
endif()

