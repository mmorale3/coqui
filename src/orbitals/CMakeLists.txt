#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

SET(SRC
  pgto.cpp
  rotate.cpp
  orbital_generator.cpp
)
ADD_LIBRARY(orbit ${SRC})
TARGET_LINK_LIBRARIES(orbit PUBLIC meanfield)
install(TARGETS orbit
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if (BUILD_UNIT_TESTS)
  add_subdirectory(tests)
endif()

