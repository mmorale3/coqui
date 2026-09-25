#//////////////////////////////////////////////////////////////////////////////////////
# The device kernels of the vertex code (gpu port step 5, notes/gpu_port_plan.md section 5).
# Built the way numerics/device_kernels/cuda builds its kernels: a small CUDA-language library that
# vertex_lib links; only with ENABLE_CUDA.
#//////////////////////////////////////////////////////////////////////////////////////
if(ENABLE_CUDA)
  set(VERTEX_CUDA_SRCS
    l0_cuda.cu
  )
  foreach( X ${VERTEX_CUDA_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()
  add_library(vertex_cuda ${VERTEX_CUDA_SRCS})
  target_include_directories(vertex_cuda PUBLIC ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(vertex_cuda PUBLIC io_lib arch_lib CUDA::cudart CUDA::cublas)
  install(TARGETS vertex_cuda
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif(ENABLE_CUDA)
