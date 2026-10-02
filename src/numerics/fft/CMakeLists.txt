#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

set(fft_src
   )

if(ENABLE_FFTW)
  set(fft_src 
       ${fft_src}
       fftw.cpp 
     )
endif()

if(ENABLE_FINUFFT)
  set(fft_src
       ${fft_src}
       finufft.cpp
     )
  # cufinufft.cpp ships in every build that has FINUFFT, so the device
  # dispatch in finufft_nda.hpp links against the math::nufft::impl::dev
  # symbols. The actual cuFINUFFT calls are gated on COQUI_HAVE_CUFINUFFT
  # at the source level; without that flag the bodies abort with a clear
  # "compiled without cuFINUFFT" message but compile and link cleanly.
  set(fft_src
       ${fft_src}
       cufinufft.cpp
     )
endif()

if(ENABLE_CUDA)
  set(fft_src
       ${fft_src}
       cufft.cpp
     )
endif()


add_library(fft_lib ${fft_src})
target_link_libraries(fft_lib PUBLIC utils nda_c) 
if(ENABLE_FFTW)
  target_link_libraries(fft_lib PUBLIC FFTW::Double)
  # ------------------------------------------------------------------------------------
  # Link a threaded FFTW so init_threads() in fftw.cpp has something to drive: with only
  # FFTW::Double linked, fftw_init_threads() / fftw_plan_with_nthreads() have no
  # implementation behind them.
  #
  # WHY THE PTHREADS FLAVOUR AND NOT THE OpenMP ONE. FFTW's OpenMP backend takes its TEAM
  # SIZE from the ambient OpenMP thread count, not from fftw_plan_with_nthreads (it calls
  # GOMP_parallel with num_threads = 0): at OMP_NUM_THREADS=1 libfftw3_omp runs serially
  # whatever nthreads is requested. CoQui keeps OMP_NUM_THREADS at 1, because raising it
  # activates SLATE's libgomp task layer, whose tasks issue MPI -- so FFTW::DoubleOpenMP
  # would be a knob that silently does nothing, or an OMP_NUM_THREADS dependence.
  # The pthreads flavour honours fftw_plan_with_nthreads exactly, with zero OpenMP
  # involvement.
  #
  # If the pthreads flavour is absent we link NO threaded flavour and say so: silently
  # falling back to the OpenMP one would produce an inert `omp_threads` knob for FFTs.
  # ------------------------------------------------------------------------------------
  if(TARGET FFTW::DoubleThreads)
    target_link_libraries(fft_lib PUBLIC FFTW::DoubleThreads)
    message(STATUS "FFTW threading: FFTW::DoubleThreads linked (fftw_plan_with_nthreads is "
                   "driven by the omp_threads knob)")
  else()
    message(STATUS "FFTW threading: libfftw3_threads not found -- FFTs stay single-threaded "
                   "(FFTW::DoubleOpenMP is deliberately NOT used: its team size comes from "
                   "OMP_NUM_THREADS, which CoQui pins at 1)")
  endif()
endif()
if(ENABLE_CUDA)
  target_link_libraries(fft_lib PUBLIC CUDA::cufft)
endif()
# FINUFFT (and, optionally, cuFINUFFT) for the non-uniform FFT backend.
if(ENABLE_FINUFFT)
  target_link_libraries(fft_lib PUBLIC FINUFFT::finufft)
  if(ENABLE_CUFINUFFT)
    # The cufinufft target is produced by the FINUFFT FetchContent superbuild
    # when FINUFFT_USE_CUDA=ON (set in cmake/finufft.cmake). The
    # COQUI_HAVE_CUFINUFFT compile flag is set at the project level
    # (see top-level CMakeLists.txt) so cufinufft.cpp uses the real
    # cuFINUFFT calls instead of its runtime-aborting stubs.
    target_link_libraries(fft_lib PUBLIC cufinufft)
  endif()
endif()
install(TARGETS fft_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
