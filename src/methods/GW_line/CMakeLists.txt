#//////////////////////////////////////////////////////////////////////////////////////
# gw_line_lib: self-consistent GW on the tilted frequency line (notes/line_gw_cpp_plan.md, S3-S7).
# Kernels are MEM-templated headers; gw_line.cpp holds their explicit instantiations.
#//////////////////////////////////////////////////////////////////////////////////////

SET(SRCS
  gw_line.cpp
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

add_library(gw_line_lib ${SRCS})
target_link_libraries(gw_line_lib PUBLIC utils numerics meanfield eri_lib)
install(TARGETS gw_line_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if (BUILD_UNIT_TESTS)
  add_subdirectory(tests)
endif()
