/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
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


#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "IO/AppAbort.hpp"
#include "utilities/proc_grid_partition.hpp"

#include "mean_field/MF.hpp"
#include "hamiltonian/one_body_hamiltonian.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/ERI/hamilt_eval_t.hpp"
#include "methods/HF/hf_t.h"
#include "methods/HF/hamilt_hf.icc"

namespace methods {
  namespace solvers {

    // instantiate templates (same combos as the thc/cholesky overloads)
    using Arr4D = nda::array<ComplexType, 4>;
    using Arrv4D = nda::array_view<ComplexType, 4>;
    using Arrv4D2 = nda::array_view<ComplexType, 4, nda::C_layout>;
    template void hf_t::evaluate(sArray_t<Arr4D> &,Arr4D const&, hamilt_eval_t&, Arr4D const&, bool, bool);
    template void hf_t::evaluate(sArray_t<Arr4D> &,Arrv4D2 const&, hamilt_eval_t&, Arrv4D2 const&, bool, bool);
    template void hf_t::evaluate(sArray_t<Arrv4D> &,Arr4D const&, hamilt_eval_t&, Arr4D const&, bool, bool);
    template void hf_t::evaluate(sArray_t<Arrv4D> &,Arr4D const&, hamilt_eval_t&, Arrv4D const&, bool, bool);
    template void hf_t::evaluate(sArray_t<Arrv4D> &,Arrv4D const&, hamilt_eval_t&, Arrv4D const&, bool, bool);

  } // solvers
} // methods
