
if(USE_DEVICE_COMPILER)
if(ENABLE_CUDA)
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
endif()
endif()

add_executable(coqui main.cpp)
target_link_libraries(coqui orbit wannier_lib methods_lib)

install(TARGETS coqui
        RUNTIME DESTINATION bin)

