# ===========================================================================
# cmake/FindFINUFFT.cmake
#
# Finds the FINUFFT library.  Sets:
#   FINUFFT_FOUND
#   FINUFFT_INCLUDE_DIRS
#   FINUFFT_LIBRARIES
#   FINUFFT::finufft   (imported target)
#
# Hints (set before find_package):
#   FINUFFT_ROOT  or  finufft_ROOT  — prefix of the installation.
# ===========================================================================
cmake_minimum_required(VERSION 3.14)

find_path(FINUFFT_INCLUDE_DIR
  NAMES finufft.h
  HINTS
    ${FINUFFT_ROOT}     ${finufft_ROOT}
    $ENV{FINUFFT_ROOT}  $ENV{finufft_ROOT}
  PATH_SUFFIXES include
)

find_library(FINUFFT_LIBRARY
  NAMES finufft
  HINTS
    ${FINUFFT_ROOT}     ${finufft_ROOT}
    $ENV{FINUFFT_ROOT}  $ENV{finufft_ROOT}
  PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FINUFFT
  REQUIRED_VARS FINUFFT_INCLUDE_DIR FINUFFT_LIBRARY
)

if(FINUFFT_FOUND AND NOT TARGET FINUFFT::finufft)
  add_library(FINUFFT::finufft UNKNOWN IMPORTED)
  set_target_properties(FINUFFT::finufft PROPERTIES
    IMPORTED_LOCATION             "${FINUFFT_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FINUFFT_INCLUDE_DIR}"
  )
  # FINUFFT itself calls FFTW internally; propagate the dependency so
  # the linker can find fftw3 when using a static libfinufft.
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3 QUIET fftw3)
    if(FFTW3_FOUND)
      set_property(TARGET FINUFFT::finufft APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES ${FFTW3_LIBRARIES})
    endif()
  endif()
endif()

mark_as_advanced(FINUFFT_INCLUDE_DIR FINUFFT_LIBRARY)
set(FINUFFT_INCLUDE_DIRS "${FINUFFT_INCLUDE_DIR}")
set(FINUFFT_LIBRARIES    "${FINUFFT_LIBRARY}")


