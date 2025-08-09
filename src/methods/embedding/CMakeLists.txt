#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

add_library(embed_lib projector_t.cpp projector_boson_t.cpp dc_utilities.hpp embed_t.cpp downfold_1e.cpp embed_eri_t.cpp)
target_link_libraries(embed_lib PUBLIC utils numerics meanfield methods_tools_lib hf_lib gw_lib scf_lib mb_state_lib)
install(TARGETS embed_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if (BUILD_UNIT_TESTS)
    add_subdirectory(tests)
endif()
