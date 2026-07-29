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


#ifndef COQUI_THC_SOLVER_COMMON_HPP
#define COQUI_THC_SOLVER_COMMON_HPP

#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/shared_array/detail/concepts.hpp"

#include "methods/ERI/detail/concepts.hpp"

namespace methods {
  namespace solvers {
    /**
     * A collection of functions for basis transformation
     * between the primary basis and the auxiliary basis
     * Still in design stage...
     */
    struct thc_solver_comm {
      template<nda::MemoryArray local_Array_t>
      using dArray_t = math::nda::distributed_array<local_Array_t, mpi3::communicator>;
      template<nda::MemoryArray Array_base_t>
      using sArray_t = math::shm::shared_array<Array_base_t>;
      template<int N>
      using shape_t = std::array<long, N>;

      /**
       * Basis transformation from the primary basis to the THC auxiliary basis
       * [array type]: normal array -> normal array
       * @param ip      - [INPUT] polarization index.
       * @param iq      - [INPUT] polarization index.
       * @param O_Iab   - [INPUT] tensor in the primary basis:
       *                  (nts, ns, nkpts_ibz, nbnd, nbnd) or (ns, nkpts_ibz, Np, Np)
       * @param O_IPQ   - [OUTPUT] tensor in the auxiliary basis for polarization (ip,iq):
       *                  (nts, ns, nkpts_ibz, Np, Np) or (ns, nkpts_ibz, Np, Np)
       * @param thc     - [INPUT] THC-ERI instance
       * @param kp_map  - [INPUT] mapping of k-point indices to their symmetry-related ones inside IBZ
       * @param kp_trev - [INPUT] whether time-reversal symmetry is needed in combination
       *                  with space group symmetry
       */
      template<nda::MemoryArray Array_primary_t, nda::MemoryArray Array_aux_t, THC_ERI thc_t>
      static void primary_to_aux(int ip, int iq, const Array_primary_t &O_Iab,
                                 Array_aux_t &O_IPQ,
                                 thc_t &thc,
                                 nda::ArrayOfRank<1> auto const& kp_map,
                                 nda::ArrayOfRank<1> auto const& kp_trev) {
        _primary_to_aux_impl(ip,iq,O_Iab, O_IPQ, thc, kp_map, kp_trev, 1, 0, 1);
      }

      /**
       * Basis transformation from the primary basis to the THC auxiliary basis
       * [array type]: normal array -> shared array
       * @param ip      - [INPUT] polarization index.
       * @param iq      - [INPUT] polarization index
       * @param O_Iab   - [INPUT] tensor in the primary basis:
       *                  (nts, ns, nkpts_ibz, nbnd, nbnd) or (ns, nkpts_ibz, Np, Np)
       * @param O_IPQ   - [OUTPUT] tensor in the auxiliary basis for polarization ip:
       *                  (nts, ns, nkpts_ibz, Np, Np) or (ns, nkpts_ibz, Np, Np)
       * @param thc     - [INPUT] THC-ERI instance
       * @param kp_map  - [INPUT] mapping of k-point indices to their symmetry-related ones inside IBZ
       * @param kp_trev - [INPUT] whether time-reversal symmetry is needed in combination
       *                  with space group symmetry
       * @param nbatch  - [INPUT] batch size of temporary objects
       */
      template<nda::MemoryArray Array_primary_t, nda::MemoryArray Array_aux_t, THC_ERI thc_t>
      static void primary_to_aux(int ip, int iq,  const Array_primary_t &O_Iab,
                                 sArray_t<Array_aux_t> &O_IPQ,
                                 thc_t &thc,
                                 nda::ArrayOfRank<1> auto const& kp_map,
                                 nda::ArrayOfRank<1> auto const& kp_trev,
                                 long nbatch = -1) {
        int rank = O_IPQ.communicator()->rank();
        int comm_size = O_IPQ.communicator()->size();
        O_IPQ.win().fence();
        _primary_to_aux_impl(ip,iq,O_Iab, O_IPQ.local(), thc, kp_map, kp_trev, nbatch,
                             rank, comm_size);
        O_IPQ.win().fence();
        O_IPQ.all_reduce();
      }

      /**
       * Basis transformation from the primary basis to the THC auxiliary basis
       * [array type]: normal array -> distributed array
       * @param ip      - [INPUT] polarization index.
       * @param iq      - [INPUT] polarization index
       * @param O_Iab   - [INPUT] tensor in the primary basis:
       *                  (nts, ns, nkpts_ibz, nbnd, nbnd) or (ns, nkpts_ibz, nbnd, nbnd)
       * @param O_IPQ   - [OUTPUT] tensor in the auxiliary basis for polarization ip:
       *                  (nts, ns, nkpts_ibz, Np, Np) or (ns, nkpts_ibz, Np, Np)
       * @param thc     - [INPUT] THC-ERI instance
       * @param kp_map  - [INPUT] mapping of k-point indices to their symmetry-related ones inside IBZ
       * @param kp_trev - [INPUT] whether time-reversal symmetry is needed in combination
       *                  with space group symmetry
       */
      template<nda::MemoryArray Array_primary_t, nda::MemoryArray Array_aux_t, THC_ERI thc_t, typename communicator_t>
      static void primary_to_aux(int ip, int iq, const Array_primary_t &O_Iab,
                                 memory::darray_t<Array_aux_t, communicator_t> &O_IPQ,
                                 thc_t &thc,
                                 nda::ArrayOfRank<1> auto const& kp_map,
                                 nda::ArrayOfRank<1> auto const& kp_trev) {
        constexpr int rank = ::nda::get_rank<Array_primary_t>;

        if constexpr (rank == 5) {
          // O_Iab = O_tskab, O_IPQ = O_tskPQ
          auto t_loc_rng = O_IPQ.local_range(0);
          auto P_offset = O_IPQ.origin()[3];
          auto Q_offset = O_IPQ.origin()[4];

          auto O_tskab_loc = O_Iab(t_loc_rng, nda::ellipsis{});
          auto O_tskPQ_loc = O_IPQ.local();
          utils::check(O_IPQ.global_shape()[2] == O_IPQ.local_shape()[2], "primary_to_aux: Does not support mpi distributed along k-axis.");

          // execute locally
          _primary_to_aux_impl(ip,iq,O_tskab_loc, O_tskPQ_loc, thc, kp_map, kp_trev, 1,
                               0, 1, P_offset, Q_offset);
        } else if constexpr (rank == 4) {
          // O_Iab = O_skab, O_IPQ = O_skPQ
          auto P_offset = O_IPQ.origin()[2];
          auto Q_offset = O_IPQ.origin()[3];

          auto O_skPQ_loc = O_IPQ.local();
          utils::check(O_IPQ.global_shape()[1] == O_IPQ.local_shape()[1], "primary_to_aux: Does not support mpi distributed along k-axis.");

          // execute locally
          _primary_to_aux_impl(ip,iq,O_Iab, O_skPQ_loc, thc, kp_map, kp_trev, 1,
                               0, 1, P_offset, Q_offset);
        } else {
          static_assert(rank == 4 or rank == 5, "thc_solver_comm::primary_to_aux: Rank != 5 or 4");
        }
      }

      /**
       * Basis transformation from the primary basis to the THC auxiliary basis
       * [array type]: shared array -> distributed array
       * @param ip      - [INPUT] polarization index.
       * @param O_Iab   - [INPUT] tensor in the primary basis:
       *                  (nts, ns, nkpts_ibz, nbnd, nbnd) or (ns, nkpts_ibz, Np, Np)
       * @param O_IPQ   - [OUTPUT] tensor in the auxiliary basis for polarization ip:
       *                  (nts, ns, nkpts_ibz, Np, Np) or (ns, nkpts_ibz, Np, Np)
       * @param thc     - [INPUT] THC-ERI instance
       * @param kp_map  - [INPUT] mapping of k-point indices to their symmetry-related ones inside IBZ
       * @param kp_trev - [INPUT] whether time-reversal symmetry is needed in combination
       *                  with space group symmetry
       */
      template<nda::MemoryArray Array_primary_t, nda::MemoryArray Array_aux_t, THC_ERI thc_t, typename communicator_t>
      static void primary_to_aux(int ip, int iq, const sArray_t<Array_primary_t> &O_Iab,
                                 memory::darray_t<Array_aux_t, communicator_t> &O_IPQ,
                                 thc_t &thc,
                                 nda::ArrayOfRank<1> auto const& kp_map,
                                 nda::ArrayOfRank<1> auto const& kp_trev) {
        primary_to_aux(ip,iq,O_Iab.local(), O_IPQ, thc, kp_map, kp_trev);
      }

      /**
       * Basis transformation from the THC auxiliary basis to the primary basis
       * [array type]: nda array -> nda array
       * @param ip      - [INPUT] polarization index.
       * @param O_IPQ  - [INPUT] tensor in the auxiliary basis:
       *                 (nts, ns, nkpts_ibz, Np, Np) or (ns, nkpts_ibz, Np, Np)
       * @param O_Iab  - [OUTPUT] tensor in the primary basis for polarization ip:
       *                 (nts, ns, nkpts_ibz, nbnd, nbnd) or (ns, nkpts_ibz, Np, Np)
       * @param thc    - [INPUT] THC-ERI instance
       * @param kp_map - [INPUT] mapping of k-points based on a given symmetry operation
       */
      template<nda::MemoryArray Array_aux_t, nda::MemoryArray Array_primary_t, THC_ERI thc_t>
      static void aux_to_primary(int ip, int iq, ComplexType scl, const Array_aux_t &O_IPQ,
                                 Array_primary_t &O_Iab,
                                 thc_t &thc,
                                 nda::ArrayOfRank<1> auto const& kp_map) {
        _aux_to_primary_impl(ip,iq,scl,O_IPQ, O_Iab, thc, kp_map, 1, 0, 1);
      }

      /**
       * Basis transformation from the THC auxiliary basis to the primary basis for a given time index
       * [array type]: distributed array -> shared memory array
       * @param ip      - [INPUT] polarization index.
       * @param iq      - [INPUT] polarization index.
       * @param scl     - [INPUT] Multiplicative scalar. 
       * @param it      - [INPUT] time index
       * @param O_skPQ  - [INPUT] tensor in the auxiliary basis for polarization ip:: (ns, nkpts_ibz, Np, Np)
       * @param O_tskab - [OUTPUT] tensor in the primary basis: (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param thc     - [INPUT] THC-ERI instance
       * @param kp_map  - [INPUT] mapping of k-points based on a given symmetry operation
       */
      template<nda::MemoryArrayOfRank<4> Array_aux_t, nda::MemoryArrayOfRank<5> Array_primary_t,
          THC_ERI thc_t, typename communicator_t>
      static void aux_to_primary(int ip, int iq, long it,
                                 ComplexType scl,
                                 const memory::darray_t<Array_aux_t, communicator_t> &O_skPQ,
                                 sArray_t<Array_primary_t> &O_tskab,
                                 thc_t &thc,
                                 nda::ArrayOfRank<1> auto const& kp_map) {
        using value_type = typename std::decay_t<Array_primary_t>::value_type;
        auto pgrid = O_skPQ.grid();
        auto s_rng = O_skPQ.local_range(0);
        auto k_rng = O_skPQ.local_range(1);

        auto [s_org, k_org, P_org, Q_org] = O_skPQ.origin();
        auto [ns, nkpts, NP, NQ] = O_skPQ.global_shape();
        utils::check(ns == O_skPQ.local_shape()[0], "aux_to_primary: Does not support mpi distributed along spin-axis.");

        auto O_skab_loc = O_tskab.local()(it, s_rng, k_rng, nda::ellipsis{});
        auto O_skPQ_loc = O_skPQ.local();

        // Setup sk_intra_comm
        communicator_t *comm = O_skPQ.communicator();
        int color = s_org * nkpts + k_org;
        int key = comm->rank();
        communicator_t dim0_intra_comm = comm->split(color, key);
        utils::check(dim0_intra_comm.size() == pgrid[2]*pgrid[3], "dim0_intra_comm.size() != pgrid[2]*pgrid[3]");

        O_tskab.win().fence();
        // to compensate for reduction
        O_tskab.communicator()->barrier();
        if(O_tskab.node_comm()->root())
          O_tskab.local() /= value_type(O_tskab.internode_comm()->size());
        O_tskab.node_comm()->barrier();
        
        _aux_to_primary_impl(ip,iq,dim0_intra_comm, scl, O_skPQ_loc, O_skab_loc, thc, kp_map, k_org, P_org, Q_org);
        O_tskab.win().fence();
        O_tskab.all_reduce();
      }

      /**
       * Basis transformation from the THC auxiliary basis to the primary basis
       * [arrary type]: distributed array -> shared memory array
       * @param ip      - [INPUT] polarization index.
       * @param iq      - [INPUT] polarization index.
       * @param scl     - [INPUT] Multiplicative scalar. 
       * @param O_IPQ   - [INPUT] tensor in the auxiliary basis for polarization ip::
       *                 (nts, ns, nkpts_ibz, Np, Np) or (ns, nkpts_ibz, Np, Np)
       * @param O_Iab   - [OUTPUT] tensor in the primary basis:
       *                 (nts, ns, nkpts_ibz, nbnd, nbnd) or (ns, nkpts_ibz, nbnd, nbnd)
       * @param thc    - [INPUT] THC-ERI instance
       * @param kp_map - [INPUT] mapping of k-points based on a given symmetry operation
       */
      template<nda::MemoryArray Array_aux_t, nda::MemoryArray Array_primary_t, THC_ERI thc_t, typename communicator_t>
      static void aux_to_primary(int ip, int iq, 
                                 ComplexType scl,
                                 const memory::darray_t<Array_aux_t, communicator_t> &O_IPQ,
                                 sArray_t<Array_primary_t> &O_Iab,
                                 thc_t &thc,
                                 nda::ArrayOfRank<1> auto const& kp_map) {
        constexpr int rank = nda::get_rank<Array_primary_t>;
        static_assert(nda::get_rank<Array_aux_t> == nda::get_rank<Array_primary_t>,
                      "thc_solver_comm::aux_to_primay: rank mismatches!");
        static_assert(rank==4 or rank==5, "thc_solver_comm::aux_to_primary: Rank != 5 or 4");
        using value_type = typename std::decay_t<Array_primary_t>::value_type;

        // to compensate for reduction
        O_Iab.win().fence();
        O_Iab.communicator()->barrier();
        if(O_Iab.node_comm()->root())
          O_Iab.local() /= value_type(O_Iab.internode_comm()->size());
        O_Iab.node_comm()->barrier();

        if constexpr (rank == 5) {
          auto pgrid = O_IPQ.grid();
          auto t_rng = O_IPQ.local_range(0);
          auto s_rng = O_IPQ.local_range(1);
          auto k_rng = O_IPQ.local_range(2);

          auto [t_org, s_org, k_org, P_org, Q_org] = O_IPQ.origin();
          auto [nt, ns, nkpts, NP, NQ] = O_IPQ.global_shape();
          utils::check(nkpts == O_IPQ.local_shape()[2], "aux_to_primary: Does not support mpi distributed along k-axis.");

          auto O_tskab_loc = O_Iab.local()(t_rng, s_rng, k_rng, nda::ellipsis{});
          auto O_tskPQ_loc = O_IPQ.local();

          // Setup wq_intra_comm
          communicator_t *gcomm = O_IPQ.communicator();
          int color = t_org * ns * nkpts + s_org * nkpts + k_org;
          int key = gcomm->rank();
          communicator_t dim0_intra_comm = gcomm->split(color, key);
          utils::check(dim0_intra_comm.size() == pgrid[3] * pgrid[4], "dim0_intra_comm.size() != pgrid[3]*pgrid[4]");

          _aux_to_primary_impl(ip,iq,dim0_intra_comm, scl, O_tskPQ_loc, O_tskab_loc, thc, kp_map, k_org, P_org, Q_org);
        } else if constexpr (rank == 4) {
          auto pgrid = O_IPQ.grid();
          auto s_rng = O_IPQ.local_range(0);
          auto k_rng = O_IPQ.local_range(1);

          auto [s_org, k_org, P_org, Q_org] = O_IPQ.origin();
          auto [ns, nkpts, NP, NQ] = O_IPQ.global_shape();
          utils::check(nkpts == O_IPQ.local_shape()[1], "aux_to_primary: Does not support mpi distributed along k-axis.");

          auto O_tskab_loc = O_Iab.local()(s_rng, k_rng, nda::ellipsis{});
          auto O_tskPQ_loc = O_IPQ.local();

          // Setup q_intra_comm
          communicator_t *gcomm = O_IPQ.communicator();
          int color = s_org * nkpts + k_org;
          int key = gcomm->rank();
          communicator_t dim0_intra_comm = gcomm->split(color, key);
          utils::check(dim0_intra_comm.size() == pgrid[2]*pgrid[3], "dim0_intra_comm.size() != pgrid[2]*pgrid[3]");

          _aux_to_primary_impl(ip,iq,dim0_intra_comm, scl, O_tskPQ_loc, O_tskab_loc, thc, kp_map,  k_org, P_org, Q_org);
        }
        // reduce
        O_Iab.win().fence();
        O_Iab.all_reduce();
      }

      /**
       * Basis transformation from the THC auxiliary basis to the primary basis
       * [array type]: shared array -> normal array
       * @param ip      - [INPUT] polarization index.
       * @param iq      - [INPUT] polarization index.
       * @param scl     - [INPUT] Multiplicative scalar. 
       * @param O_IPQ   - [INPUT] tensor in the auxiliary basis for polarization ip::
       *                 (nts, ns, nkpts_ibz, Np, Np) or (ns, nkpts_ibz, Np, Np)
       * @param O_Iab   - [OUTPUT] tensor in the primary basis:
       *                 (nts, ns, nkpts_ibz, nbnd, nbnd) or (ns, nkpts_ibz, nbnd, nbnd)
       * @param thc    - [INPUT] THC-ERI instance
       * @param kp_map - [INPUT] mapping of k-points based on a given symmetry operation
       * @param nbatch - [OPTIONAL] batch size for temporary objects
       */
      template<nda::MemoryArray Array_aux_t, nda::MemoryArray Array_primary_t, THC_ERI thc_t>
      static void aux_to_primary(int ip, int iq, 
                                 ComplexType scl,
                                 const sArray_t<Array_aux_t> &O_IPQ,
                                 Array_primary_t &O_Iab,
                                 thc_t &thc,
                                 nda::ArrayOfRank<1> auto const& kp_map,
                                 long nbatch = -1) {
        int rank = O_IPQ.communicator()->rank();
        int comm_size = O_IPQ.communicator()->size();
        using value_type = typename std::decay_t<Array_primary_t>::value_type;
        // to compensate for reduction
        O_Iab() /= value_type(O_IPQ.communicator()->size());
        _aux_to_primary_impl(ip,iq,scl,O_IPQ.local(), O_Iab, thc, kp_map, nbatch,
                            rank, comm_size);
        O_IPQ.communicator()->all_reduce_in_place_n(O_Iab.data(), O_Iab.size(), std::plus<>{});
      }

    private:
      /**
       * Internal implementation of transformation from a primary basis to a THC auxiliary basis:
       *         O_tskPQ += scl * X_skPa * O_tskab * conj(X_skQb), where the t dimension is optional.
       * This function does not take care of MPI reduction, and therefore should never be used externally.
       * Users should provide proper reduction instructions after calling this function.
       *
       * This function assumes
       *   - O_tskab contains k-points only in the IBZ
       *
       * @param ip        - [INPUT] polarization index.
       * @param iq        - [INPUT] polarization index.
       * @param scl       - [INPUT] Multiplicative scalar. 
       * @param O_tskab   - [INPUT] input tensor in the primary basis
       * @param O_tskPQ   - [OUTPUT] output tensor in the THC auxiliary basis
       * @param thc       - [INPUT] THC-ERI instance
       * @param kp_map    - [INPUT] mapping of k-point indices to their symmetry-related ones inside IBZ
       * @param kp_trev   - [INPUT] whether time-reversal symmetry is needed in combination with space group symmetry
       * @param nbatch    - [INPUT] batch number for auxiliary indices
       * @param rank      - [INPUT] local processor rank
       * @param comm_size - [INPUT] size of the mpi communicator
       * @param P_offset  - [INPUT] local offset for P index
       * @param Q_offset  - [INPUT] local offset for Q index
       */
      template<nda::Array Array_primary_t, nda::Array Array_aux_t, THC_ERI thc_t>
      static void _primary_to_aux_impl(int ip, int iq,
                                       const Array_primary_t& O_tskab,
                                       Array_aux_t& O_tskPQ,
                                       thc_t &thc,
                                       nda::ArrayOfRank<1> auto const& kp_map,
                                       nda::ArrayOfRank<1> auto const& kp_trev,
                                       long nbatch,
                                       int rank, int comm_size,
                                       long P_offset = 0, long Q_offset = 0) {
        static_assert(nda::get_rank<Array_primary_t> == nda::get_rank<Array_aux_t>,
                      "thc_solver_comm::primary_to_aux_impl: Rank mismatch");
        static_assert(nda::get_rank<Array_primary_t> >= 4, "thc_solver_comm::primary_to_aux_impl: Rank < 4");

        decltype(nda::range::all) all;
        constexpr int N = nda::get_rank<Array_primary_t>;
        // The aux output may live on device while the primary input
        // is HOST. Both gemms need to share an address space; we run
        // them in HOST scratch on the host path and stage the second
        // result into the device-typed O_ikPQ slice on the device path.
        constexpr bool aux_on_device = ::nda::mem::on_device<Array_aux_t>;
        size_t ns         = O_tskab.shape(N-4);
        size_t nkpts_ibz  = O_tskab.shape(N-3);
        size_t nbnd       = O_tskab.shape(N-2);
        size_t nkpts      = O_tskPQ.shape(N-3);
        size_t NP_loc     = O_tskPQ.shape(N-2);
        size_t NQ_loc     = O_tskPQ.shape(N-1);
        utils::check(NP_loc+P_offset <= thc.Np(), "thc_solver_comm::primary_to_aux_impl: NP_loc+P_offset > thc.Np()");
        utils::check(NQ_loc+Q_offset <= thc.Np(), "thc_solver_comm::primary_to_aux_impl: NQ_loc+Q_offset > thc.Np()");

        // dim_i = (t, s) or (s)
        size_t dim_i = std::accumulate(O_tskPQ.shape().begin(), O_tskPQ.shape().end()-3, (size_t)1, std::multiplies<>{});
        utils::check(dim_i == std::accumulate(O_tskab.shape().begin(), O_tskab.shape().end()-3, (size_t)1, std::multiplies<>{}),
                     "thc_solver_comm::primary_to_aux_impl: dim_i mismatched");

        auto O_ikPQ_4D = nda::reshape(O_tskPQ, shape_t<4>{dim_i, nkpts, NP_loc, NQ_loc});
        auto O_ikab_4D = nda::reshape(O_tskab, shape_t<4>{dim_i, nkpts_ibz, nbnd, nbnd});

        if (nbatch < 0) nbatch = utils::find_min_col(comm_size, dim_i*nkpts);
        utils::check(comm_size % nbatch == 0, "primary_to_aux_impl: comm_size % nbatch != 0");
        auto [origin, end] = itertools::chunk_range(0, NP_loc, nbatch, rank % nbatch);
        int batch_size = end - origin;
        int n_large_batch = NP_loc - nbatch * (NP_loc / nbatch);
        int offset = (rank % nbatch < n_large_batch)? 0 : 0 + n_large_batch;

        nda::array<ComplexType, 2> Ask_Pb(batch_size, nbnd);
#if defined(ENABLE_DEVICE)
        // Device-side scratch for the second gemm output, plus the X-factor
        // cache, all only referenced when ENABLE_DEVICE is on. Wrapping the
        // declarations themselves with #if is necessary because nda's
        // mem::check_adr_sp_valid<Device> static_asserts when GPU support
        // is not compiled in — even `if constexpr` discarded branches
        // still get parsed and trigger the assert on a CPU-only build.
        [[maybe_unused]] memory::array<DEVICE_MEMORY, ComplexType, 2> O_PQ_dev;
        [[maybe_unused]] memory::array<DEVICE_MEMORY, ComplexType, 2> Ask_Pb_dev;
        [[maybe_unused]] memory::array<DEVICE_MEMORY, ComplexType, 4> Xr_cache;
        [[maybe_unused]] memory::array<DEVICE_MEMORY, ComplexType, 4> Xl_cache;
        [[maybe_unused]] memory::array<DEVICE_MEMORY, ComplexType, 2> Oab_dev;
        if constexpr (aux_on_device) {
          O_PQ_dev   = memory::array<DEVICE_MEMORY, ComplexType, 2>(batch_size, NQ_loc);
          Ask_Pb_dev = memory::array<DEVICE_MEMORY, ComplexType, 2>(batch_size, nbnd);
          // Both X factors are staged once per call so that the first gemm can
          // run on the device too. It used to run on the host -- one core per
          // rank at ~100 GF/s against ~14 TF/s device-resident -- and then push
          // its result across PCIe per (t,s,k). Uploading the (nbnd x nbnd)
          // input block instead of the (batch x nbnd) result moves the same
          // order of bytes, so the transform becomes device-resident for free.
          //
          // The two caches are kept separate even when ip == iq and the ranges
          // coincide: sharing the P-sliced factor as the right operand is
          // exactly the bug fixed in 765754c, and 287 MB per cache here
          // (nk*Np*nbnd*16 at Si 2x2x2/500b) is not worth the risk.
          nda::range O_P_rng_all(P_offset, P_offset + NP_loc);
          nda::range O_Q_rng(Q_offset, Q_offset + NQ_loc);
          nda::array<ComplexType, 4> Xr_host(ns, nkpts, NQ_loc, nbnd);
          nda::array<ComplexType, 4> Xl_host(ns, nkpts, NP_loc, nbnd);
          for (size_t s_ = 0; s_ < ns; ++s_) {
            for (size_t k_ = 0; k_ < nkpts; ++k_) {
              auto Xsk_Pa_r = thc.X(s_, iq, k_);
              Xr_host(s_, k_, all, all) = Xsk_Pa_r(O_Q_rng, all);
              auto Xsk_Pa_l = thc.X(s_, ip, k_);
              Xl_host(s_, k_, all, all) = Xsk_Pa_l(O_P_rng_all, all);
            }
          }
          Xr_cache = memory::to_memory_space<DEVICE_MEMORY>(Xr_host);
          Xl_cache = memory::to_memory_space<DEVICE_MEMORY>(Xl_host);
          Oab_dev  = memory::array<DEVICE_MEMORY, ComplexType, 2>(nbnd, nbnd);
        }
#endif

        for (size_t ikP = rank; ikP < dim_i*nkpts*nbatch; ikP += comm_size) {
          // ikP = (i * nkpts + k) * nbatch + PP
          size_t i = ikP / (nkpts*nbatch);
          size_t s = i % ns; // i = it * ns + is
          size_t k = (ikP / nbatch) % nkpts;
          size_t PP = ikP % nbatch;
          nda::range X_P_rng(PP*batch_size + offset + P_offset, (PP+1)*batch_size + offset + P_offset);
          nda::range O_P_rng(PP*batch_size + offset, (PP+1)*batch_size + offset);
          nda::range O_Q_rng(Q_offset, Q_offset + NQ_loc);

          if constexpr (aux_on_device) {
#if defined(ENABLE_DEVICE)
            // Both gemms on the device: stage this (i,k) block of the primary
            // input, contract with the cached left factor, then with the right.
            // The cache rows for X_P_rng are exactly O_P_rng, since the cache
            // holds rows [P_offset, P_offset+NP_loc).
            Oab_dev = O_ikab_4D(i, kp_map(k), all, all);
            auto Xl = Xl_cache(s, k, O_P_rng, all);
            if (kp_trev(k)) {
              nda::blas::gemm(Xl, nda::transpose(Oab_dev), Ask_Pb_dev);
            } else {
              nda::blas::gemm(Xl, Oab_dev, Ask_Pb_dev);
            }
            nda::blas::gemm(Ask_Pb_dev, nda::dagger(Xr_cache(s, k, all, all)), O_PQ_dev);
            O_ikPQ_4D(i, k, O_P_rng, all) = O_PQ_dev;
#endif
          } else {
            // HOST path unchanged: Ask_Pb = Xsk_Pa * Osk_ab, then * conj(Xsk_Qb)
            auto Xsk_Pa_l = thc.X(s, ip, k);
            if(kp_trev(k)) {
              nda::blas::gemm(Xsk_Pa_l(X_P_rng, all), nda::transpose(O_ikab_4D(i, kp_map(k), all, all)), Ask_Pb);
            } else {
              nda::blas::gemm(Xsk_Pa_l(X_P_rng, all), O_ikab_4D(i, kp_map(k), all, all), Ask_Pb);
            }
            auto Xsk_Pa_r = thc.X(s, iq, k);
            nda::blas::gemm(Ask_Pb, nda::dagger(Xsk_Pa_r(O_Q_rng, all)), O_ikPQ_4D(i, k, O_P_rng, all));
          }
        }
      }

      /**
       * Internal implementation of transformation from a THC auxiliary basis to a primary basis:
       *         O_tskab += scl * conj(X_skPa) * O_tskPQ * X_skQb, where the t dimension is optional.
       * This function does not take care of MPI reduction, and therefore should never be used externally.
       * Users should provide proper reduction instructions after calling this function.
       */
      template<nda::Array Array_aux_t, nda::Array Array_primary_t, THC_ERI thc_t>
      static void _aux_to_primary_impl(int ip, int iq,
                                       ComplexType scl,
                                       const Array_aux_t &O_tskPQ,
                                       Array_primary_t &O_tskab,
                                       thc_t &thc,
                                       nda::ArrayOfRank<1> auto const& kp_map,
                                       long nbatch,
                                       int rank,
                                       int comm_size) {
        static_assert(nda::get_rank<Array_primary_t> == nda::get_rank<Array_aux_t>,
                      "thc_solver_comm::aux_to_primary_impl: Rank mismatch");
        static_assert(nda::get_rank<Array_primary_t> >= 4, "thc_solver_comm::aux_to_primary_impl: Rank < 4");

        decltype(nda::range::all) all;

        constexpr int N = nda::get_rank<Array_primary_t>;
        size_t nbnd = O_tskab.shape(N-2);
        size_t nkpts = O_tskab.shape(N-3);
        size_t ns = O_tskab.shape(N-4);
        size_t dim_i = std::accumulate(O_tskPQ.shape().begin(), O_tskPQ.shape().end()-2, (size_t)1, std::multiplies<>{});

        auto O_iPQ_3D = nda::reshape(O_tskPQ, shape_t<3>{dim_i, thc.Np(), thc.Np()});
        auto O_iab_3D = nda::reshape(O_tskab, shape_t<3>{dim_i, nbnd, nbnd});

        if (nbatch < 0) {
          nbatch = utils::find_min_col(comm_size, dim_i);
        }
        utils::check(comm_size % nbatch == 0, "aux_to_primary_impl: comm_size % nbatch != 0");
        long origin, end;
        std::tie(origin, end) = itertools::chunk_range(0, nbnd, nbatch, rank % nbatch);
        int batch_size = end - origin;
        int n_large_batch = nbnd - nbatch * (nbnd / nbatch);
        int offset = (rank % nbatch < n_large_batch)? 0 : n_large_batch;


        nda::array<ComplexType, 2> Ask_aQ(batch_size, thc.Np());
        //nda::matrix<ComplexType> Xsk_Pa_conj(thc.Np(), batch_size);

        for (size_t ia = rank; ia < dim_i*nbatch; ia += comm_size) {
          // ia = i * nbatch + aa
          size_t i  = ia / nbatch;
          size_t aa = ia % nbatch;
          // i = (it * ns + is) * nkpts + ik
          size_t s = (i / nkpts) % ns;
          size_t k = i % nkpts;

          nda::range a_range(aa * batch_size + offset, (aa + 1) * batch_size + offset);

          // Ask_aQ = conj(Xsk_Pa) * Osk_PQ
          auto Xsk_Pa_l = thc.X(s, ip, kp_map(k)); 
          auto Xsk_Pa_r = ( ip==iq ? Xsk_Pa_l : thc.X(s, iq, kp_map(k))); 
          //Xsk_Pa_conj = nda::conj(Xsk_Pa(all, a_range));
          nda::blas::gemm(nda::dagger(Xsk_Pa_l(all, a_range)), O_iPQ_3D(i, all, all), Ask_aQ);

          // Osk_ab = Ask_aQ * Xsk_Qb
          nda::blas::gemm(scl, Ask_aQ, Xsk_Pa_r, 
                          ComplexType(1.0), O_iab_3D(i, a_range, all));
        }
      }

      /**
       * Internal implementation of transformation from a THC auxiliary basis to a primary basis:
       *         O_tskab += scl * conj(X_skPa) * O_tskPQ * X_skQb, where the t dimension is optional.
       * This function does not take care of MPI reduction, and therefore should never be used externally.
       * Users should provide proper reduction instructions after calling this function.
       *
       * @param ip        - [INPUT] polarization index.
       * @param iq        - [INPUT] polarization index.
       * @param scl       - [INPUT] Multiplicative scalar. 
       * @param dim0_comm - [INPUT] communicator for dimension 0: (t,s,k) or (s,k)
       * @param O_tskPQ   - [INPUT] input tensor in the THC auxiliary basis
       * @param O_tskab   - [OUTPUT] output tensor in the primary basis
       * @param thc       - [INPUT] THC-ERI instance
       * @param kp_map    - [INPUT] mapping of k-point indices to their rotated ones based on a symmetry operation
       * @param P_offset  - [INPUT] local offset for P index
       * @param Q_offset  - [INPUT] local offset for Q index
       */
      template<nda::Array Array_aux_t, nda::Array Array_primary_t, THC_ERI thc_t, typename communicator_t>
      static void _aux_to_primary_impl(int ip, int iq,
                                       communicator_t &dim0_comm,
                                       ComplexType scl,
                                       const Array_aux_t &O_tskPQ,
                                       Array_primary_t &O_tskab,
                                       thc_t &thc,
                                       nda::ArrayOfRank<1> auto const& kp_map,
                                       long k_offset, long P_offset, long Q_offset) {
        static_assert(nda::get_rank<Array_primary_t> == nda::get_rank<Array_aux_t>,
                      "thc_solver_comm::aux_to_primary_impl: Rank mismatch");
        static_assert(nda::get_rank<Array_primary_t> >= 4, "thc_solver_comm::aux_to_primary_impl: Rank < 4");

        decltype(nda::range::all) all;

        constexpr int N = nda::get_rank<Array_primary_t>;
        // The auxiliary input may live on device while the primary output
        // (a HOST shared-array slice) lives on the host. The two gemms
        // below must execute in the same address space as O_iPQ; any
        // result needed for the host MPI reduce + accumulate is staged
        // back to host once at the end of each (s, k) iteration.
        constexpr bool aux_on_device = ::nda::mem::on_device<Array_aux_t>;
        constexpr MEMORY_SPACE WORK_MEM = aux_on_device ? DEVICE_MEMORY : HOST_MEMORY;

        size_t nbnd = O_tskab.shape(N-2);
        size_t ns_loc = O_tskPQ.shape(N-4);
        size_t nk_loc = O_tskPQ.shape(N-3);
        size_t NP_loc = O_tskPQ.shape(N-2);
        size_t NQ_loc = O_tskPQ.shape(N-1);
        nda::range P_rng(P_offset, P_offset+NP_loc);
        nda::range Q_rng(Q_offset, Q_offset+NQ_loc);
        utils::check(NP_loc+P_offset <= thc.Np(), "thc_solver_comm::aux_to_primary_impl: NP_loc+P_offset > thc.Np()");
        utils::check(NQ_loc+Q_offset <= thc.Np(), "thc_solver_comm::aux_to_primary_impl: NQ_loc+Q_offset > thc.Np()");

        size_t dim0 = std::accumulate(O_tskPQ.shape().begin(), O_tskPQ.shape().end()-2, (size_t)1, std::multiplies<>{});

        auto O_iPQ_3D = nda::reshape(O_tskPQ, shape_t<3>{dim0, NP_loc, NQ_loc});
        auto O_iab_3D = nda::reshape(O_tskab, shape_t<3>{dim0, nbnd, nbnd});

        memory::array<WORK_MEM, ComplexType, 2> Ask_aQ(nbnd, NQ_loc);

        // X-factor cache: thc.X(s, ip, kp_map(k))(P_rng, all) is constant
        // across the t axis in the i sweep (each unique (s, k) appears
        // dim0 / (ns_loc * nk_loc) = nt times). Materialise it on WORK_MEM
        // once instead of mirroring per i — one bulk H->D copy replaces
        // nt copies per (s, k).
        //
        // The left factor is sliced with P_rng, the right factor with
        // Q_rng. They coincide only when ip == iq AND this rank's local
        // (P, Q) block is diagonal (P_rng == Q_rng); sharing the cache on
        // ip == iq alone silently used X(P_rng) as the right factor on
        // ranks with off-diagonal or rectangular blocks (wrong data, and
        // mismatched gemm dims for NP_loc != NQ_loc).
        const bool same_lr = (ip == iq) and (P_offset == Q_offset) and (NP_loc == NQ_loc);
        memory::array<WORK_MEM, ComplexType, 4> Xl_cache;
        memory::array<WORK_MEM, ComplexType, 4> Xr_cache;
        {
          nda::array<ComplexType, 4> Xl_host(ns_loc, nk_loc, NP_loc, nbnd);
          for (size_t s = 0; s < ns_loc; ++s) {
            for (size_t k = 0; k < nk_loc; ++k) {
              auto Xsk_Pa_l = thc.X(s, ip, kp_map(k + k_offset));
              Xl_host(s, k, all, all) = Xsk_Pa_l(P_rng, all);
            }
          }
          Xl_cache = memory::to_memory_space<WORK_MEM>(Xl_host);
          if (not same_lr) {
            nda::array<ComplexType, 4> Xr_host(ns_loc, nk_loc, NQ_loc, nbnd);
            for (size_t s = 0; s < ns_loc; ++s) {
              for (size_t k = 0; k < nk_loc; ++k) {
                auto Xsk_Pa_r = thc.X(s, iq, kp_map(k + k_offset));
                Xr_host(s, k, all, all) = Xsk_Pa_r(Q_rng, all);
              }
            }
            Xr_cache = memory::to_memory_space<WORK_MEM>(Xr_host);
          }
        }

        // Buffer all Oab outputs in one (dim0, nbnd, nbnd) tensor and do a
        // single bulk D->H copy + MPI reduce at the end. The per-iter
        // to_host + reduce we used to do drains the cuBLAS stream every
        // iter; this version keeps all gemms in-flight on the GPU and
        // batches the reduce.
        memory::array<WORK_MEM, ComplexType, 3> Oab_all(dim0, nbnd, nbnd);
        nda::array<ComplexType, 3> Oab_all_host;
        if constexpr (aux_on_device) {
          Oab_all_host.resize(std::array<long,3>{long(dim0), long(nbnd), long(nbnd)});
        }

        for (size_t i = 0; i < dim0; ++i) {
          // i = (it * ns_loc + is) * nk_loc + ik
          size_t s = (i / nk_loc) % ns_loc;
          size_t k = i % nk_loc;

          auto Xl_view = Xl_cache(s, k, all, all);
          auto Xr_view = same_lr ? Xl_cache(s, k, all, all)
                                 : Xr_cache(s, k, all, all);
          auto Oab_i   = Oab_all(i, all, all);

          nda::blas::gemm(nda::dagger(Xl_view), O_iPQ_3D(i, all, all), Ask_aQ);
          nda::blas::gemm(Ask_aQ, Xr_view, Oab_i);
        } // i

        if constexpr (aux_on_device) {
          // One bulk device -> host transfer (drains cuBLAS stream once),
          // one bulk MPI reduce, then per-i host accumulate on root.
          Oab_all_host = nda::to_host(Oab_all);
          dim0_comm.reduce_in_place_n(Oab_all_host.data(), Oab_all_host.size(),
                                      std::plus<>{}, 0);
          if (dim0_comm.root()) {
            for (size_t i = 0; i < dim0; ++i) {
              O_iab_3D(i, all, all) += scl * Oab_all_host(i, all, all);
            }
          }
        } else {
          // Host: bulk MPI reduce on Oab_all itself (host buffer), then
          // accumulate. Bit-identical to per-iter reduce + accumulate.
          dim0_comm.reduce_in_place_n(Oab_all.data(), Oab_all.size(),
                                      std::plus<>{}, 0);
          if (dim0_comm.root()) {
            for (size_t i = 0; i < dim0; ++i) {
              O_iab_3D(i, all, all) += scl * Oab_all(i, all, all);
            }
          }
        }
      } // _aux_to_primary_impl

    }; // thc_solver_comm
  } // solvers
} // methods



#endif //COQUI_THC_SOLVER_COMMON_HPP
