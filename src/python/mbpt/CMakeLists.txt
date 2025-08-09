
Python_add_library(mbpt_module MODULE mbpt_module.cpp)

# clair is needed only if we want to generate python bindings from scratch
target_link_libraries(mbpt_module PRIVATE
        c2py::c2py nda::nda_py methods_lib
)
set_target_properties(mbpt_module PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PYTHON_PACKAGE_PATH}/_lib")

install(TARGETS mbpt_module
        LIBRARY DESTINATION ${PYTHON_PACKAGE_INSTALL_PATH}/_lib)

if (COQUI_UPDATE_PYTHON_BINDINGS)
    add_custom_command(
        COMMAND ${clair-c2py} -p ${PROJECT_BINARY_DIR} mbpt_module.cpp  # -p path/to/compile/commands
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}                   # Execute in source directory
        OUTPUT mbpt_module.wrap.cxx                                     # Generates the .wrap.cxx file
        DEPENDS mbpt_module.cpp                    # which depends on the source (hence will be regenerated if it changes)
    )
    add_custom_target(mbpt_bindings_generation DEPENDS mbpt_module.wrap.cxx) # We make the module depend on the bindings
    add_dependencies(mbpt_module mbpt_bindings_generation eri_module)                   # so that the generation happens before the compilation
endif()
