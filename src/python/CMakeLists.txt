set(PYTHON_PACKAGE_PATH ${PROJECT_BINARY_DIR}/python/coqui  CACHE PATH "Python package directory.")
set(PYTHON_PACKAGE_INSTALL_PATH lib/python/site-packages/coqui CACHE PATH "Python package install directory")

add_subdirectory(utils)
add_subdirectory(mean_field)
add_subdirectory(interaction)
add_subdirectory(mbpt)
add_subdirectory(embed)
add_subdirectory(post_proc)
add_subdirectory(wannier)

# Configure the version for CoQui Python
configure_file(${PROJECT_SOURCE_DIR}/src/python/version.py.in
               ${PYTHON_PACKAGE_PATH}/version.py)

# Copy the Python package placeholder into the build directory
file(GLOB_RECURSE python_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py)
foreach(file ${python_sources})
    configure_file(${file} ${PYTHON_PACKAGE_PATH}/${file} COPYONLY)
endforeach()

# Install python files to proper location
install(FILES ${PYTHON_PACKAGE_PATH}/version.py
        DESTINATION ${PYTHON_PACKAGE_INSTALL_PATH})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/python/
        DESTINATION ${PYTHON_PACKAGE_INSTALL_PATH}
        FILES_MATCHING PATTERN "*.py")
