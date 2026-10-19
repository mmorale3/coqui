#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

add_subdirectory(ERI)
add_subdirectory(tools)
add_subdirectory(mb_state)
add_subdirectory(HF)
add_subdirectory(vertex)
add_subdirectory(scr_coulomb)
add_subdirectory(GW)
add_subdirectory(GF2)
add_subdirectory(embedding)
add_subdirectory(SCF)
add_subdirectory(pproc)
# Real-axis P(omega)/W(omega) components. Header-only (no library target); the directory
# is only active when ENABLE_FINUFFT is ON (its CMakeLists returns early otherwise), so
# with the flag OFF (the default) it does not change the build.
add_subdirectory(GW_real_axis)
add_subdirectory(tests)

SET(SRCS
  MBPT_drivers.cpp
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

add_library(methods_lib ${SRCS})
target_link_libraries(methods_lib PUBLIC meanfield methods_tools_lib eri_lib hf_lib vertex_lib scr_coulomb_lib gw_lib gf2_lib scf_lib embed_lib pproc_lib)
install(TARGETS methods_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
