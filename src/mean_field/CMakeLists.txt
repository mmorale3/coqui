#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

SET(MEANFIELD
  qe/qe_interface.cpp
  qe/qe_readonly.cpp
  bdft/bdft_readonly.cpp
  symmetry/unfold_bz.cpp
)
ADD_LIBRARY(meanfield ${MEANFIELD})
target_link_libraries(meanfield PUBLIC utils utils_f numerics nda_c h5_c)
install(TARGETS meanfield
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if (BUILD_UNIT_TESTS)
  add_subdirectory(qe/tests)
  add_subdirectory(bdft/tests)
  add_subdirectory(pyscf/tests)
  add_subdirectory(model_hamiltonian/tests)
  add_subdirectory(tests)
endif()
