#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

SET(HAMIL
  pseudo/pseudopot.cpp
  add_kinetic.cpp
  add_vxc.cpp
)
ADD_LIBRARY(hamilt ${HAMIL})
TARGET_LINK_LIBRARIES(hamilt PUBLIC arch_lib meanfield utils_f)
install(TARGETS hamilt
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if (BUILD_UNIT_TESTS)
  add_subdirectory(tests)
endif()

