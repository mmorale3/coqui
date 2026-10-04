#//////////////////////////////////////////////////////////////////////////////////////
# The device kernels of GW_line (S7d: the fused Hadamard / slab-convolution kernel).
# Built the way methods/vertex/cuda builds its kernels: a small CUDA-language library that gw_line_lib links; only with
# ENABLE_CUDA.
#//////////////////////////////////////////////////////////////////////////////////////
if(ENABLE_CUDA)
  set(GW_LINE_CUDA_SRCS
    gw_line_cuda.cu
    gw_line_lapack.cu
  )
  foreach( X ${GW_LINE_CUDA_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()
  add_library(gw_line_cuda ${GW_LINE_CUDA_SRCS})
  target_include_directories(gw_line_cuda PUBLIC ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(gw_line_cuda PUBLIC io_lib arch_lib CUDA::cudart CUDA::cusolver)
  install(TARGETS gw_line_cuda
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif(ENABLE_CUDA)
