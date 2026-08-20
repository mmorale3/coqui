#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

#enable_language(Fortran)
include(FortranCInterface)
FortranCInterface_VERIFY(CXX)
FortranCInterface_HEADER(FC.h MACRO_NAMESPACE "FC_")
if(ENABLE_WANNIER90)
  SET(UTILITIES_F
    fortran_helpers.f90
    wannier90_interface.f90
  )
  ADD_LIBRARY(utils_f ${UTILITIES_F})
  TARGET_LINK_LIBRARIES(utils_f PRIVATE MPI::MPI_Fortran)
  TARGET_LINK_LIBRARIES(utils_f PUBLIC Wannier90::wannier90) 
else()
  SET(UTILITIES_F
    fortran_helpers.f90
  )
  ADD_LIBRARY(utils_f ${UTILITIES_F})
endif(ENABLE_WANNIER90)
install(TARGETS utils_f
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

SET(UTILITIES
  math.cpp
  parser.cpp
  harmonics.cpp
  freemem.cpp
  mpi_context.cpp
)
ADD_LIBRARY(utils ${UTILITIES})
TARGET_LINK_LIBRARIES(utils PUBLIC io_lib arch_lib nda_c h5_c)
# T-3b (notes/coqui_threading_spec.md rev 3 section 7.2 item 5): host OpenMP for CoQui's own
# regions -- the `omp_threads` knob (src/utilities/omp_threads.hpp). PUBLIC on purpose: every
# CoQui library reaches `utils` transitively, so this puts every C++ TU in the same -fopenmp
# state. The threaded regions live in headers instantiated from several targets
# (numerics/ac/pade/pade_t.hpp, methods/GW/thc_gw.icc), and a mixed state there would be an
# ODR violation whose symptom is a region that silently does not thread.
if(COQUI_ENABLE_HOST_OPENMP AND OpenMP_CXX_FOUND)
  TARGET_LINK_LIBRARIES(utils PUBLIC OpenMP::OpenMP_CXX)
endif()
if(ENABLE_SPHERICART)
  TARGET_LINK_LIBRARIES(utils PUBLIC sphericart)
endif(ENABLE_SPHERICART)
install(TARGETS utils
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(BUILD_UNIT_TESTS)
  ADD_LIBRARY(catch_main catch_main.cpp)
  TARGET_LINK_LIBRARIES(catch_main PUBLIC utils Catch2::Catch2)
  add_subdirectory(tests)
endif(BUILD_UNIT_TESTS)
