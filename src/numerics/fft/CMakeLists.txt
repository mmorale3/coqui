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
endif()
if(ENABLE_CUDA)
  target_link_libraries(fft_lib PUBLIC CUDA::cufft)
endif()
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
