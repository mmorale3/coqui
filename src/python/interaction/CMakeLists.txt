
Python_add_library(eri_module MODULE eri_module.cpp)

# clair is needed only if we want to generate python bindings from scratch
target_link_libraries(eri_module PRIVATE c2py::c2py nda::nda_py meanfield eri_lib)
set_target_properties(eri_module PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PYTHON_PACKAGE_PATH}/_lib")

install(TARGETS eri_module
        LIBRARY DESTINATION ${PYTHON_PACKAGE_INSTALL_PATH}/_lib)

if (COQUI_UPDATE_PYTHON_BINDINGS)
    add_custom_command(
        COMMAND ${clair-c2py} -p ${PROJECT_BINARY_DIR} eri_module.cpp  # -p path/to/compile/commands
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}                   # Execute in source directory
        OUTPUT eri_module.wrap.cxx                                     # Generates the .wrap.cxx file
        DEPENDS eri_module.cpp                    # which depends on the source (hence will be regenerated if it changes)
    )
    add_custom_target(eri_bindings_generation DEPENDS eri_module.wrap.cxx) # We make the module depend on the bindings
    add_dependencies(eri_module eri_bindings_generation mf_module)                   # so that the generation happens before the compilation
endif()
