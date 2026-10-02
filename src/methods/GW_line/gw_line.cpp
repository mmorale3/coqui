/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */

/**
 * gw_line_lib (notes/line_gw_cpp_plan.md, S3): explicit instantiations of the MEM-templated line-GW kernels
 * (propagators, polarization) for HOST_MEMORY and, in device builds, DEVICE_MEMORY.
 */

#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/polarization.hpp"

namespace methods::gw_line {

using numerics::line_dlr::time_ray_t;

template struct propagator_t<HOST_MEMORY>;
template void polarization<HOST_MEMORY>(propagator_t<HOST_MEMORY> &, pole_data_t const &, mf::MF const &,
                                        aux_grid_t const &, nda::array<ComplexType, 1> const &, time_ray_t const &,
                                        time_ray_t const &, long, memory::array<HOST_MEMORY, ComplexType, 4> &,
                                        utils::TimerManager &, sector_t);

#if defined(ENABLE_DEVICE)
template struct propagator_t<DEVICE_MEMORY>;
template void polarization<DEVICE_MEMORY>(propagator_t<DEVICE_MEMORY> &, pole_data_t const &, mf::MF const &,
                                          aux_grid_t const &, nda::array<ComplexType, 1> const &, time_ray_t const &,
                                          time_ray_t const &, long, memory::array<DEVICE_MEMORY, ComplexType, 4> &,
                                          utils::TimerManager &, sector_t);
#endif

} // namespace methods::gw_line
