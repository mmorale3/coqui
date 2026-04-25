# ===========================================================================
# cmake/finufft.cmake
#
# Downloads, configures, and builds FINUFFT via CMake FetchContent.
# Include this file (or add it to CMAKE_MODULE_PATH and include() it)
# from your top-level or src/numerics/CMakeLists.txt when ENABLE_FINUFFT=ON.
#
# After this file is processed, the imported target FINUFFT::finufft is
# available for target_link_libraries() exactly as if find_package had
# found an installed copy.
#
# Key build options forwarded to finufft:
#   FINUFFT_USE_OPENMP  (default: matches your project's USE_OPENMP)
#   FINUFFT_FFTW_SUFFIX (default: empty — links double-precision libfftw3)
#
# Minimum CMake: 3.14 (required by FetchContent_MakeAvailable).
# ===========================================================================

include(FetchContent)

# ---------------------------------------------------------------------------
# Version pin — update the GIT_TAG to move to a newer release.
# Using a tag (not a branch) gives a reproducible build.
# ---------------------------------------------------------------------------
set(FINUFFT_GIT_REPO "https://github.com/flatironinstitute/finufft.git")
set(FINUFFT_GIT_TAG  "v2.4.1")   # <-- change here to upgrade

FetchContent_Declare(
  finufft
  GIT_REPOSITORY ${FINUFFT_GIT_REPO}
  GIT_TAG        ${FINUFFT_GIT_TAG}
  GIT_SHALLOW    TRUE   # only fetch the tagged commit, not full history
)

# ---------------------------------------------------------------------------
# Configure finufft's own CMake options before making it available.
#
# We turn off everything we don't need so the build stays fast and does not
# pull in unexpected dependencies (Fortran, Python, MATLAB, tests, ...).
# ---------------------------------------------------------------------------

# Match OpenMP usage to the rest of the project if the variable is set.
if(DEFINED USE_OPENMP)
  set(FINUFFT_USE_OPENMP ${USE_OPENMP} CACHE BOOL "" FORCE)
else()
  set(FINUFFT_USE_OPENMP ON  CACHE BOOL "" FORCE)
endif()

# finufft calls FFTW internally.  It must find the same fftw3 the rest of
# the project uses.  If your project sets FFTW_ROOT, propagate it.
if(DEFINED FFTW_ROOT)
  set(FFTW_ROOT ${FFTW_ROOT} CACHE PATH "" FORCE)
endif()

# Disable everything we don't want compiled as part of the superbuild.
set(FINUFFT_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(FINUFFT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FINUFFT_SPREAD_ONLY    OFF CACHE BOOL "" FORCE)
set(FINUFFT_BUILD_FORTRAN  OFF CACHE BOOL "" FORCE)
set(FINUFFT_BUILD_MATLAB   OFF CACHE BOOL "" FORCE)
set(FINUFFT_BUILD_PYTHON   OFF CACHE BOOL "" FORCE)
# Build a static library by default to avoid runtime-path headaches.
# Set to ON if you prefer a shared library.
set(BUILD_SHARED_LIBS      OFF CACHE BOOL "" FORCE)

# 
if(DEFINED ENABLE_FFTW)
  if(DEFINED USE_OPENMP)
    set(FINUFFT_FFTW_LIBRARIES fftw3 fftw3_omp CACHE STRING "" FORCE)
    set(FINUFFT_USE_FFTW ON CACHE BOOL "" FORCE)
  else()
    set(FINUFFT_FFTW_LIBRARIES fftw3 CACHE STRING "" FORCE)
    set(FINUFFT_USE_FFTW ON CACHE BOOL "" FORCE)
  endif()
endif()

# ---------------------------------------------------------------------------
# Download and configure.  After this call:
#   finufft_SOURCE_DIR  — path to the cloned source
#   finufft_BINARY_DIR  — path to the build directory
# and the target  finufft::finufft  is defined by finufft's own CMakeLists.
# ---------------------------------------------------------------------------
FetchContent_MakeAvailable(finufft)

# ---------------------------------------------------------------------------
# Alias finufft::finufft → FINUFFT::finufft so the rest of the project
# (and the test CMakeLists) can use the same target name regardless of
# whether the library was fetched or found via find_package.
# ---------------------------------------------------------------------------
if(TARGET finufft AND NOT TARGET FINUFFT::finufft)
  add_library(FINUFFT::finufft ALIAS finufft)
endif()

# ---------------------------------------------------------------------------
# Convenience: expose include path as a variable in case any file needs it
# directly (usually not necessary when using the target).
# ---------------------------------------------------------------------------
set(FINUFFT_INCLUDE_DIRS
  "${finufft_SOURCE_DIR}/include"
  CACHE PATH "FINUFFT include directory (FetchContent)" FORCE)

message(STATUS "FINUFFT: using FetchContent source at ${finufft_SOURCE_DIR}")
message(STATUS "FINUFFT: version tag ${FINUFFT_GIT_TAG}")
