#//////////////////////////////////////////////////////////////////////////////////////
# The device kernels of the SCF driver (the causal projection of Sigma_c).
# Built the way methods/vertex/cuda builds its kernels: a small CUDA-language library that
# scf_lib links; only with ENABLE_CUDA.
#//////////////////////////////////////////////////////////////////////////////////////
if(ENABLE_CUDA)
  set(SCF_CUDA_SRCS
    causal_proj_cuda.cu
  )
  foreach( X ${SCF_CUDA_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()
  add_library(scf_cuda ${SCF_CUDA_SRCS})
  target_include_directories(scf_cuda PUBLIC ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(scf_cuda PUBLIC io_lib arch_lib CUDA::cudart CUDA::cublas)
  install(TARGETS scf_cuda
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif(ENABLE_CUDA)
