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
  # T-3b item 3.3 (notes/coqui_threading_t3a.md section 2.3 row 8 / section 3.3): actually
  # LINK a threaded FFTW so init_threads() in fftw.cpp has something to drive. The root
  # CMakeLists has been asking find_package for the threaded double library since the FFTW
  # block was written; only FFTW::Double was ever linked, so fftw_init_threads() /
  # fftw_plan_with_nthreads() had no implementation behind them.
  #
  # WHY THE PTHREADS FLAVOUR AND NOT THE OpenMP ONE. Measured on a rusty genoa node with
  # fftw/3.3.10 (job 6912051, 128^3 c2c, 20 executes, OMP_NUM_THREADS=1):
  #
  #     libfftw3_omp      nthreads = 1 / 2 / 4 / 8  ->  0.213 / 0.214 / 0.213 / 0.256 s
  #     libfftw3_threads  nthreads = 1 / 2 / 4 / 8  ->  0.208 / 0.102 / 0.067 / 0.041 s
  #     libfftw3_omp at OMP_NUM_THREADS=8, nthreads = 8 -> 0.041 s
  #
  # i.e. FFTW's OpenMP backend takes its TEAM SIZE from the ambient OpenMP thread count, not
  # from fftw_plan_with_nthreads (confirmed in the disassembly: it calls GOMP_parallel with
  # num_threads = 0). CoQui's threading contract pins OMP_NUM_THREADS at 1 forever, because
  # raising it activates SLATE's libgomp task layer and its tasks issue MPI
  # (notes/coqui_threading_spec.md rev 3 section 7.2 item 1) -- so FFTW::DoubleOpenMP would be
  # a knob that silently does nothing, or an OMP_NUM_THREADS dependence the spec forbids.
  # The pthreads flavour honours fftw_plan_with_nthreads exactly, with zero OpenMP
  # involvement, which is precisely the decoupling this increment is built on.
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
# RW-1: the finufft link block, merged in ALONGSIDE the T-3b FFTW::DoubleThreads block
# above (both live in this file; neither replaces the other).
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
