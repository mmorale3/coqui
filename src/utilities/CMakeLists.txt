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
