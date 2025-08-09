
Python_add_library(wannier_module MODULE wannier_module.cpp)

# clair is needed only if we want to generate python bindings from scratch
target_link_libraries(wannier_module PRIVATE c2py::c2py nda::nda_py meanfield wannier_lib)
set_target_properties(wannier_module PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PYTHON_PACKAGE_PATH}/_lib")

install(TARGETS wannier_module
        LIBRARY DESTINATION ${PYTHON_PACKAGE_INSTALL_PATH}/_lib)

if (COQUI_UPDATE_PYTHON_BINDINGS)
    add_custom_command(
        COMMAND ${clair-c2py} -p ${PROJECT_BINARY_DIR} wannier_module.cpp  # -p path/to/compile/commands
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}                   # Execute in source directory
        OUTPUT wannier_module.wrap.cxx                                     # Generates the .wrap.cxx file
        DEPENDS wannier_module.cpp                    # which depends on the source (hence will be regenerated if it changes)
    )
    add_custom_target(wannier_bindings_generation DEPENDS wannier_module.wrap.cxx) # We make the module depend on the bindings
    add_dependencies(wannier_module wannier_bindings_generation mf_module)                   # so that the generation happens before the compilation
endif()
