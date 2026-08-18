#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

add_library(embed_lib projector_t.cpp projector_boson_t.cpp dc_utilities.hpp embed_t.cpp downfold_1e.cpp embed_eri_t.cpp)
target_link_libraries(embed_lib PUBLIC utils numerics meanfield methods_tools_lib hf_lib gw_lib scf_lib mb_state_lib)
# embed_lib sits in a static-library dependency CYCLE
#   embed_lib -> gw_lib -> scr_coulomb_lib -> embed_lib
# and, since the scGW-tilde C3 readout made pproc_lib depend on scr_coulomb_lib as well
# (src/methods/pproc/CMakeLists.txt), the cycle is now entered from pproc_lib too. CMake
# breaks such a cycle by repeating its members on the link line LINK_INTERFACE_MULTIPLICITY
# times (default 2), and two rounds are one short here: round 2 is where
# scr_coulomb_t.cpp.o first gets pulled in, and its reference to
# projector_boson_t::calc_bosonic_projector<thc_reader_t> (defined in embed_lib's
# projector_boson_t.cpp.o) then has no later embed_lib occurrence to resolve against. For
# the executables the reference is satisfied by another target's ordering, but a
# Python_add_library MODULE (post_proc_module) links only pproc_lib and silently keeps the
# symbol undefined -- the failure surfaces at `import coqui` as
#   post_proc_module.so: undefined symbol: ..._22calc_bosonic_projector...
# A third round makes projector_boson_t.cpp.o reachable.
set_target_properties(embed_lib PROPERTIES LINK_INTERFACE_MULTIPLICITY 3)
install(TARGETS embed_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if (BUILD_UNIT_TESTS)
    add_subdirectory(tests)
endif()
