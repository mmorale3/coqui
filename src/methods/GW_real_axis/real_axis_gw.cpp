/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 */

// Translation unit for the real-axis ISDF/THC GW solver. The solver is
// built up from header-only foundational utilities (real_freq_grid,
// real_axis_conv) and progressively higher-level kernels. This file
// hosts non-template entry points as those are added.

#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_conv.hpp"
#include "methods/GW_real_axis/real_axis_mb_state.hpp"
#include "methods/GW_real_axis/real_axis_dyson.hpp"
#include "methods/GW_real_axis/real_axis_pi.hpp"
#include "methods/GW_real_axis/real_axis_sigma.hpp"
#include "methods/GW_real_axis/real_axis_gw_t.h"

namespace methods {
namespace real_axis {

// Anchor symbol so the static library has a non-empty translation unit.
namespace detail {
  int gw_real_axis_translation_unit_anchor() { return 0; }
}

} // namespace real_axis
} // namespace methods
