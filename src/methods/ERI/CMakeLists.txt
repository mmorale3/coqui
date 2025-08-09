#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

SET(ERI_SRCS
  cholesky.cpp
  thc.cpp
  mb_eri_context.h
  eri_utils.cpp
)

if(USE_DEVICE_COMPILER)
if(ENABLE_CUDA)
  foreach( X ${ERI_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()
elseif(ENABLE_HIP)
  foreach( X ${ERI_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE HIP)
  endforeach()
endif()
endif()

ADD_LIBRARY(eri_lib ${ERI_SRCS})
target_link_libraries(eri_lib PUBLIC utils numerics meanfield nda_c h5_c slate)
install(TARGETS eri_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})


#///////////  code to test/compare ERI factorization approaches /////////////

if(USE_DEVICE_COMPILER)
if(ENABLE_CUDA)
  foreach( X compare_eri.cpp )
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()
elseif(ENABLE_HIP)
  foreach( X compare_eri.cpp )
    set_source_files_properties(${X} PROPERTIES LANGUAGE HIP)
  endforeach()
endif()
endif()

#add_executable(compare_eri compare_eri.cpp)
#target_link_libraries(compare_eri utils numerics meanfield hamilt eri_lib nda_c h5_c slate)

#install(TARGETS compare_eri 
#        RUNTIME DESTINATION bin)

#///////////  code to test/compare ERI factorization approaches /////////////

# MAM: disabling since it needs to be heavily updated. Reenabled if needed
#if(USE_DEVICE_COMPILER)
#if(ENABLE_CUDA)
#  foreach( X check_eri_symmetry.cpp )
#    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
#  endforeach()
#elseif(ENABLE_HIP)
#  foreach( X check_eri_symmetry.cpp )
#    set_source_files_properties(${X} PROPERTIES LANGUAGE HIP)
#  endforeach()
#endif()
#endif()
#
#add_executable(check_eri_symmetry check_eri_symmetry.cpp)
#target_link_libraries(check_eri_symmetry utils numerics meanfield eri_lib nda_c h5_c slate)
#
#install(TARGETS check_eri_symmetry
#        RUNTIME DESTINATION bin)


#////////// Add tests //////////////

if (BUILD_UNIT_TESTS)
  add_subdirectory(tests)
endif()
