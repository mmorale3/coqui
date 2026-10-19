#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

# Real-axis ISDF/THC P(Omega)/W(Omega) chain: real-frequency grids, NUFFT convolutions,
# Im/Re Pi, THC projection, update_w, the real-axis state, the processor grid, the
# eps^-1 head and a serial reference Dyson solver.
#
# The module is header-only (there is no translation unit and no library target);
# consumers include the headers directly.
#
# The whole directory is inert unless ENABLE_FINUFFT is ON.

if(NOT ENABLE_FINUFFT)
  return()
endif()

if (BUILD_UNIT_TESTS)
  add_subdirectory(tests)
endif()
