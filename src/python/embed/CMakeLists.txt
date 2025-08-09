
Python_add_library(embed_module MODULE embed_module.cpp)
target_link_libraries(embed_module PRIVATE
        c2py::c2py nda::nda_py methods_lib
)
set_target_properties(embed_module PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PYTHON_PACKAGE_PATH}/_lib")
install(TARGETS embed_module
        LIBRARY DESTINATION ${PYTHON_PACKAGE_INSTALL_PATH}/_lib)

if (COQUI_UPDATE_PYTHON_BINDINGS)
    add_custom_command(
        COMMAND ${clair-c2py} -p ${PROJECT_BINARY_DIR} embed_module.cpp  # -p path/to/compile/commands
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}                   # Execute in source directory
        OUTPUT embed_module.wrap.cxx                                     # Generates the .wrap.cxx file
        DEPENDS embed_module.cpp                    # which depends on the source (hence will be regenerated if it changes)
    )
    add_custom_target(embed_bindings_generation DEPENDS embed_module.wrap.cxx)     # We make the module depend on the bindings
    add_dependencies(embed_module embed_bindings_generation mf_module eri_module)  # so that the generation happens before the compilation
endif()
