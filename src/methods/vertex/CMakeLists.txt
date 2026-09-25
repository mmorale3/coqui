#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

SET(SRCS
  vertex_t.cpp
  )

if(USE_DEVICE_COMPILER)
if(ENABLE_CUDA)
  foreach( X ${SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()
elseif(ENABLE_HIP)
  foreach( X ${SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE HIP)
  endforeach()
endif()
endif()

add_library(vertex_lib ${SRCS})
target_link_libraries(vertex_lib PUBLIC meanfield eri_lib mb_state_lib)
# gpu port step 5: the device kernels of the vertex code (CUDA builds only)
if(ENABLE_CUDA)
  add_subdirectory(cuda)
  target_link_libraries(vertex_lib PUBLIC vertex_cuda)
endif()
install(TARGETS vertex_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if (BUILD_UNIT_TESTS)
  add_subdirectory(tests)
endif()

# notes/ladder_opt_spec.md B0: the distributed dense-pipeline bench (always built -- it is
# the design/regression harness for the SLATE ladder path, and it is small)
add_subdirectory(bench)
