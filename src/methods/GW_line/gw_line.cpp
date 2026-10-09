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
 * gw_line_lib (notes/line_gw_cpp_plan.md, S3-S5): explicit instantiations of the MEM-templated line-GW kernels
 * (propagators, polarization, screened interaction, self-energy, static part) for HOST_MEMORY and, in device builds,
 * DEVICE_MEMORY.
 */

#include "methods/GW_line/proc_grid.hpp"
#include "methods/GW_line/line_state.hpp"
#include "methods/GW_line/propagators.hpp"
#include "methods/GW_line/polarization.hpp"
#include "methods/GW_line/screened.hpp"
#include "methods/GW_line/self_energy.hpp"
#include "methods/GW_line/static_part.hpp"

namespace methods::gw_line {

using time_ray_t = numerics::line_dlr::time_nodes_t;   // the kernels take the type-erased node view (S7b)

template struct propagator_t<HOST_MEMORY>;
template void polarization<HOST_MEMORY>(propagator_t<HOST_MEMORY> &, pole_data_t const &, mf::MF const &,
                                        aux_grid_t const &, nda::array<ComplexType, 1> const &, time_ray_t const &,
                                        time_ray_t const &, long, memory::array<HOST_MEMORY, ComplexType, 4> &,
                                        utils::TimerManager &, sector_t, std::vector<long> const &);

#if defined(ENABLE_DEVICE)
template struct propagator_t<DEVICE_MEMORY>;
template void polarization<DEVICE_MEMORY>(propagator_t<DEVICE_MEMORY> &, pole_data_t const &, mf::MF const &,
                                          aux_grid_t const &, nda::array<ComplexType, 1> const &, time_ray_t const &,
                                          time_ray_t const &, long, memory::array<DEVICE_MEMORY, ComplexType, 4> &,
                                          utils::TimerManager &, sector_t, std::vector<long> const &);
#endif

#define GW_LINE_SCREENED_INST(MEM)                                                                                       \
  template struct coulomb_blocks_t<MEM>;                                                                                 \
  template void screened_interaction<MEM>(memory::array<MEM, ComplexType, 4> &, coulomb_blocks_t<MEM> const &,            \
                                          bosonic_basis_t const &, aux_grid_t const &,                                    \
                                          utils::mpi_context_t<boost::mpi3::communicator> &,                              \
                                          memory::array<MEM, ComplexType, 4> &, utils::TimerManager &,                    \
                                          memory::array<MEM, ComplexType, 4> *, std::vector<long> const &, bool);         \
  template void w_time<MEM>(memory::array<MEM, ComplexType, 4> const &, bosonic_basis_t const &, long, long,              \
                            nda::array<ComplexType, 1> const &, sector_t, bool, memory::array_view<MEM, ComplexType, 3>); \
  template void eval_poles<MEM>(memory::array<MEM, ComplexType, 4> const &, bosonic_basis_t const &, long, long,          \
                                nda::array<ComplexType, 1> const &, sector_t, bool,                                       \
                                memory::array_view<MEM, ComplexType, 3>);

GW_LINE_SCREENED_INST(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
GW_LINE_SCREENED_INST(DEVICE_MEMORY)
#endif
#undef GW_LINE_SCREENED_INST

#define GW_LINE_SIGMA_INST(MEM)                                                                                          \
  template void self_energy<MEM>(propagator_t<MEM> &, pole_data_t const &, memory::array<MEM, ComplexType, 4> const &,    \
                                 bosonic_basis_t const &, mf::MF const &, aux_grid_t const &,                             \
                                 boost::mpi3::communicator &, nda::array<ComplexType, 1> const &,                         \
                                 time_ray_t const &, time_ray_t const &, long, nda::array<ComplexType, 4> &,              \
                                 utils::TimerManager &, sector_t, bool,                                                   \
                                 memory::array<HOST_MEMORY, ComplexType, 4> const *, long, nda::array<ComplexType, 4> *,  \
                                 memory::array<MEM, ComplexType, 4> *);                                                  \
  template void hartree_exchange<MEM>(propagator_t<MEM> &, coulomb_blocks_t<MEM> const &,                                \
                                      nda::array<ComplexType, 3> const &, mf::MF const &, aux_grid_t const &,             \
                                      boost::mpi3::communicator &, nda::array<ComplexType, 3> &,                          \
                                      utils::TimerManager &);

GW_LINE_SIGMA_INST(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
GW_LINE_SIGMA_INST(DEVICE_MEMORY)
#endif
#undef GW_LINE_SIGMA_INST

} // namespace methods::gw_line
