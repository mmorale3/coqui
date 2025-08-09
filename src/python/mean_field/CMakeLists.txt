
Python_add_library(mf_module MODULE mf_module.cpp)

# clair is needed only if we want to generate python bindings from scratch
target_link_libraries(mf_module PRIVATE c2py::c2py nda::nda_py meanfield)
set_target_properties(mf_module PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PYTHON_PACKAGE_PATH}/_lib")

install(TARGETS mf_module
        LIBRARY DESTINATION ${PYTHON_PACKAGE_INSTALL_PATH}/_lib)

if (COQUI_UPDATE_PYTHON_BINDINGS)
    add_custom_command(
        COMMAND ${clair-c2py} -p ${PROJECT_BINARY_DIR} mf_module.cpp  # -p path/to/compile/commands
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}                   # Execute in source directory
        OUTPUT mf_module.wrap.cxx                                     # Generates the .wrap.cxx file
        DEPENDS mf_module.cpp                    # which depends on the source (hence will be regenerated if it changes)
    )
    add_custom_target(mf_bindings_generation DEPENDS mf_module.wrap.cxx) # We make the module depend on the bindings
    add_dependencies(mf_module mf_bindings_generation mpi_handler)                   # so that the generation happens before the compilation
endif()
