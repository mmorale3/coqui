#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

SET(SRCS
  scr_coulomb_t.cpp
  scr_coulomb_chol.cpp
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

add_library(scr_coulomb_lib ${SRCS})
target_link_libraries(scr_coulomb_lib PUBLIC meanfield eri_lib embed_lib)
install(TARGETS scr_coulomb_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})


#if (BUILD_UNIT_TESTS)
#  add_subdirectory(tests)
#endif()
