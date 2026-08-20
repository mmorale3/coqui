#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

# Real-axis ISDF/THC P(Omega)/W(Omega) chain, ported from origin/real_axis
# (increment RW-1, notes/rw_real_axis_w_spec.md; inventory + anchors in
# notes/real_axis_branch_survey.md).
#
# DEVIATION FROM THE BRANCH: only the ~9-header leaf slice of the module is ported
# (grids, NUFFT conv, Im/Re Pi, THC projection, update_w, state, proc grid, eps^-1
# head, serial reference Dyson). The Sigma / HF / Dyson-G / SCF-driver / dispatcher
# files are NOT here, so real_axis_gw.cpp -- the branch's only translation unit and
# the sole reason gw_real_axis_lib exists -- has no content to compile. The slice is
# header-only; consumers include the headers directly (cf. the branch's own
# GW_real_axis_qp/CMakeLists.txt, "currently header-only; no library target needed").
#
# The whole directory is inert unless ENABLE_FINUFFT is ON (gate RW-1-c).

if(NOT ENABLE_FINUFFT)
  return()
endif()

if (BUILD_UNIT_TESTS)
  add_subdirectory(tests)
endif()
