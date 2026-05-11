# ==========================================================================
# Real-axis QP-form QSGW with contour deformation (separate from numerical-A
# real-axis QSGW in src/methods/GW_real_axis/).
#
# The module is currently header-only; no library target needed for the
# math primitives (Pade + iaxis integral) since they're templated and
# included directly by the SCF driver. Tests link against the existing
# fft_lib + utils for the math infrastructure.
# ==========================================================================

if (NOT ENABLE_FINUFFT)
  return()
endif()

add_subdirectory(tests)
