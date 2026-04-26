#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

# Real-axis ISDF/THC GW solver. Header-only for now (foundational utilities);
# implementation .cpp files are added as the solver gains state-coupled methods.

SET(SRCS
  real_axis_gw.cpp
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

add_library(gw_real_axis_lib ${SRCS})
target_link_libraries(gw_real_axis_lib PUBLIC meanfield eri_lib scr_coulomb_lib fft_lib)
if(ENABLE_FINUFFT)
  target_link_libraries(gw_real_axis_lib PUBLIC FINUFFT::finufft)
endif()
install(TARGETS gw_real_axis_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if (BUILD_UNIT_TESTS)
  add_subdirectory(tests)
endif()
