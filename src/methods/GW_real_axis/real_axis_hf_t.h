/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_HF_T_H
#define COQUI_REAL_AXIS_HF_T_H

#include <string>
#include <utility>

#include "configuration.hpp"
#include "numerics/shared_array/nda.hpp"
#include "nda/nda.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/check.hpp"
#include "numerics/sparse/csr_blas.hpp"

#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_mb_state.hpp"
#include "methods/GW_real_axis/real_axis_sigma_x.hpp"
#include "methods/GW_real_axis/real_axis_thc_project.hpp"
#include "methods/GW_real_axis/real_axis_proc_grid.hpp"

namespace methods {
namespace real_axis {

/**
 * Real-axis Hartree-Fock (static exchange) solver. Wraps
 * `evaluate_Sigma_x_serial` with a state/THC API matching the rest of the
 * real-axis solver classes (`real_axis_scr_coulomb_t`, `real_axis_gw_t`).
 *
 * Mirrors `methods::solvers::hf_t` in role; the signature differs because
 * the real-axis side computes Sigma_x directly from the spectral function
 * A (via the auxiliary-basis density n_aux = X.A.X^dag integrated against
 * f(w)) rather than from a separately-stored density matrix Dm.
 *
 * Reads:  `state.A_wskij`, MF accessors via `thc.MF()`.
 * Writes: `state.Sigma_x_skij`.
 *
 * Configuration:
 *   div_treatment: "ignore_g0" zeroes the q=Gamma contribution.
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
class real_axis_hf_base_t {
public:
  using mpi_communicator_t = boost::mpi3::communicator;

  real_axis_hf_base_t(real_freq_grid_t const* grid,
                      std::string div_treatment = "ignore_g0")
    : _grid(grid),
      _div_treatment(std::move(div_treatment))
  {
    // The exchange evaluation runs on host throughout: state.Sigma_x_skij
    // and state.A_wskij are sArrays (node-shared host memory), and the
    // marshalled X / V / kmq buffers are local nda::arrays. MEM is a
    // template marker so the SCF driver can dispatch uniformly on MEM.
    utils::check(_grid != nullptr,
                 "real_axis_hf_t: grid pointer must not be null");
  }

  ~real_axis_hf_base_t() = default;

  std::string div_treatment() const { return _div_treatment; }

  /**
   * Evaluate the static-exchange self-energy Sigma_x from state.A_wskij.
   *
   * The mu used for the f(w) integration in `evaluate_Sigma_x_serial` is
   * taken from `mu` (NOT grid->mu_chem()), so the SCF loop can pass the
   * current mu without rebuilding the grid. The MPI communicator is read
   * from state.mpi->comm.
   */
  template<methods::THC_ERI THC_t>
  void evaluate(real_axis_mb_state_t& state,
                THC_t const& thc,
                double mu)
  {
    utils::check(state.grid != nullptr,
                 "real_axis_hf_t::evaluate: state.grid not bound");
    utils::check(state.grid == _grid,
                 "real_axis_hf_t::evaluate: state.grid disagrees with the "
                 "grid the solver was constructed with");
    utils::check(state.A_wskij.has_value(),
                 "real_axis_hf_t::evaluate: state.A_wskij not allocated");
    utils::check(state.mpi != nullptr,
                 "real_axis_hf_t::evaluate: state.mpi not bound");
    auto& comm = state.mpi->comm;

    auto const& grid_in = *_grid;
    auto const& MF      = *thc.MF();
    auto A_in           = state.A_wskij->local();

    const long ns      = MF.nspin();
    const long Nk      = MF.nkpts();
    const long Nq      = MF.nqpts();
    const long Nk_ibz  = MF.nkpts_ibz();
    const long Nq_ibz  = MF.nqpts_ibz();
    const long nbnd    = MF.nbnd();
    const long Naux    = thc.Np();
    const long N_w     = grid_in.N_w();

    utils::check(MF.npol() == 1,
                 "real_axis_hf_t::evaluate: npol={} not supported (need 1)",
                 MF.npol());

    // The FBZ-direct-sum Sigma_x kernel below (via evaluate_Sigma_x_serial)
    // works only when IBZ == FBZ. For non-trivial IBZ the symmetry-adapted
    // isym kernel is dispatched via use_isym below. Both produce
    // state.Sigma_x_skij at IBZ k.
    const bool use_isym = (Nk != Nk_ibz);

    auto kp_to_ibz_arr    = MF.kp_to_ibz();
    auto kp_trev_arr      = MF.kp_trev();
    auto kp_trev_pair_arr = MF.kp_trev_pair();

    // Allocate Sigma_x output sArray at IBZ k (one copy per node).
    if (!state.Sigma_x_skij.has_value())
      state.Sigma_x_skij.emplace(*state.mpi,
          std::array<long, 4>{ns, Nk_ibz, nbnd, nbnd});
    if (state.Sigma_x_skij->node_comm()->root())
      state.Sigma_x_skij->local() = ComplexType(0.0, 0.0);
    state.Sigma_x_skij->node_sync();

    // Repack A from IBZ-stored (N_w, ns, Nk_ibz, nbnd, nbnd) to FBZ-k
    // driver layout, with matrix-hermitian symmetrization. The FBZ k
    // expansion uses kp_to_ibz; X(FBZ k) carries the orbital rotation
    // implicitly (symmetry-adapted ISDF).
    nda::array<ComplexType, 5> A_drv(ns, Nk, N_w, nbnd, nbnd);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < Nk; ++k) {
        const long kibz = kp_to_ibz_arr(k);
        for (long iw = 0; iw < N_w; ++iw)
          for (long mu_i = 0; mu_i < nbnd; ++mu_i)
            for (long nu = 0; nu < nbnd; ++nu)
              A_drv(s, k, iw, mu_i, nu) =
                  ComplexType(0.5, 0.0) *
                  (A_in(iw, s, kibz, mu_i, nu)
                   + std::conj(A_in(iw, s, kibz, nu, mu_i)));
      }

    // Marshal X, V, kmq from THC. X is moderately large (Naux x nbnd per
    // s, k); put it in shared memory (one copy per node). V has 2 aux
    // indices; marshal only this rank's local (P_loc, Q_loc) block from
    // each thc.Z(iq) -- evaluate_Sigma_x_serial accepts this directly.
    math::shm::shared_array<nda::array_view<ComplexType, 4>>
        sX_skPmu(*state.mpi, {ns, Nk, Naux, nbnd});
    if (sX_skPmu.node_comm()->root()) {
      auto X_loc = sX_skPmu.local();
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk; ++k) {
          auto Xsk = thc.X(static_cast<int>(s), 0, static_cast<int>(k));
          for (long P = 0; P < Naux; ++P)
            for (long mu_i = 0; mu_i < nbnd; ++mu_i)
              X_loc(s, k, P, mu_i) = Xsk(P, mu_i);
        }
    }
    sX_skPmu.node_sync();

    // Determine this rank's (P_loc, Q_loc) block.
    auto block = real_axis::bosonic_local_block(
        static_cast<long>(comm.size()),
        static_cast<long>(comm.rank()), Naux);
    const long P0_hf     = block[0];
    const long NP_loc_hf = block[1];
    const long Q0_hf     = block[2];
    const long NQ_loc_hf = block[3];
    nda::array<ComplexType, 3> V_qPQ_loc(Nq, NP_loc_hf, NQ_loc_hf);
    for (long iq = 0; iq < Nq; ++iq) {
      auto Zq = thc.Z(static_cast<int>(iq));
      for (long iP = 0; iP < NP_loc_hf; ++iP)
        for (long iQ = 0; iQ < NQ_loc_hf; ++iQ)
          V_qPQ_loc(iq, iP, iQ) = Zq(P0_hf + iP, Q0_hf + iQ);
    }
    auto X_skPmu = sX_skPmu.local();
    math::shm::shared_array<nda::array_view<long, 2>> skmq(*state.mpi, {Nk, Nq});
    {
      if (skmq.node_comm()->root()) {
        auto kmq_loc = skmq.local();
        auto const& qk_to_k2 = MF.qk_to_k2();
        for (long iq = 0; iq < Nq; ++iq)
          for (long ik = 0; ik < Nk; ++ik)
            kmq_loc(ik, iq) = qk_to_k2(iq, ik);
      }
      skmq.node_sync();
    }
    auto kmq_to_kp = skmq.local();

    long iq_gamma = -1;
    if (_div_treatment == "ignore_g0") {
      auto Qp = MF.Qpts();
      if (Qp.shape()[0] >= 1) {
        double norm0 = 0.0;
        for (long c = 0; c < Qp.shape()[1]; ++c) norm0 += std::abs(Qp(0, c));
        if (norm0 < 1e-10) iq_gamma = 0;
      }
    }

    // Build a grid at the requested mu (Sigma_x integrates the Fermi factor).
    auto grid_at_mu = real_freq_grid_t(grid_in.beta(), mu,
                                       nda::array<double,1>(grid_in.w()),
                                       nda::array<double,1>(grid_in.Omega()),
                                       grid_in.N_t(), grid_in.T_window());

    // ================================================================
    // Symmetry-adapted Sigma_x path (use_isym = Nk != Nk_ibz). Outer
    // isym loop, per-isym aux Sigma_x build + back-projection with
    // X(FBZ ks=ks_to_k(isym, ik_ibz)) + orbital rotation D for isym!=0.
    // Pattern mirrors gw_t::evaluate's isym branch and the imag-axis
    // methods/HF/thc_hf.icc reference.
    // ================================================================
    if (use_isym) {
      using nda::range;
      const auto _ = range::all;

      // Step 1: compute n_{munu}(s, k_ibz) = int dw f(w) A_phys(s, k_ibz, w).
      // A_phys is the matrix-hermitian symmetrized spectral function.
      nda::array<ComplexType, 4> n_skij_ibz(ns, Nk_ibz, nbnd, nbnd);
      n_skij_ibz = ComplexType(0.0, 0.0);
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < Nk_ibz; ++k) {
          for (long iw = 0; iw < N_w; ++iw) {
            const double f_w   = grid_at_mu.fermi(grid_at_mu.w()(iw));
            const double w_w   = grid_at_mu.w_weights()(iw);
            const double coeff = f_w * w_w;
            for (long mu_i = 0; mu_i < nbnd; ++mu_i)
              for (long nu = 0; nu < nbnd; ++nu) {
                const ComplexType a_mn = A_in(iw, s, k, mu_i, nu);
                const ComplexType a_nm = A_in(iw, s, k, nu,  mu_i);
                const ComplexType a_sym =
                    ComplexType(0.5, 0.0) * (a_mn + std::conj(a_nm));
                n_skij_ibz(s, k, mu_i, nu) += coeff * a_sym;
              }
          }
        }

      // Step 2: project n to aux at FBZ k via X(FBZ k) and n at IBZ kp_to_ibz(k).
      // For TR-pair k's, fill aux-n by conj-copy from the non-TR partner.
      nda::array<ComplexType, 4> n_aux_skPQ_fbz(ns, Nk, NP_loc_hf, NQ_loc_hf);
      {
        nda::array<ComplexType, 3> n_dummy_munu(1, nbnd, nbnd);
        nda::array<ComplexType, 3> n_dummy_PQ(NP_loc_hf, NQ_loc_hf, 1);
        range Pr(P0_hf, P0_hf + NP_loc_hf);
        range Qr(Q0_hf, Q0_hf + NQ_loc_hf);
        // Pass 1: project for non-TR k's only.
        for (long s = 0; s < ns; ++s)
          for (long k = 0; k < Nk; ++k) {
            if (kp_trev_arr(k) != 0) continue;
            const long kibz = kp_to_ibz_arr(k);
            auto X_P_slice = X_skPmu(s, k, Pr, _);
            auto X_Q_slice = X_skPmu(s, k, Qr, _);
            for (long mu_i = 0; mu_i < nbnd; ++mu_i)
              for (long nu = 0; nu < nbnd; ++nu)
                n_dummy_munu(0, mu_i, nu) = n_skij_ibz(s, kibz, mu_i, nu);
            primary_to_aux_one_k(X_P_slice, X_Q_slice,
                                 n_dummy_munu, n_dummy_PQ);
            for (long iP = 0; iP < NP_loc_hf; ++iP)
              for (long iQ = 0; iQ < NQ_loc_hf; ++iQ)
                n_aux_skPQ_fbz(s, k, iP, iQ) = n_dummy_PQ(iP, iQ, 0);
          }
        // Pass 2: TR-pair fix-up (conj-copy from kp_trev_pair partner).
        for (long s = 0; s < ns; ++s)
          for (long k = 0; k < Nk; ++k) {
            if (kp_trev_arr(k) == 0) continue;
            const long k_partner = kp_trev_pair_arr(k);
            for (long iP = 0; iP < NP_loc_hf; ++iP)
              for (long iQ = 0; iQ < NQ_loc_hf; ++iQ)
                n_aux_skPQ_fbz(s, k, iP, iQ) =
                    std::conj(n_aux_skPQ_fbz(s, k_partner, iP, iQ));
          }
      }

      // Step 3: V at IBZ q (W lives at IBZ q in the symmetry-adapted picture).
      nda::array<ComplexType, 3> V_qPQ_loc_ibz(Nq_ibz, NP_loc_hf, NQ_loc_hf);
      for (long iq = 0; iq < Nq_ibz; ++iq) {
        auto Zq = thc.Z(static_cast<int>(iq));
        for (long iP = 0; iP < NP_loc_hf; ++iP)
          for (long iQ = 0; iQ < NQ_loc_hf; ++iQ)
            V_qPQ_loc_ibz(iq, iP, iQ) = Zq(P0_hf + iP, Q0_hf + iQ);
      }

      // Identify Gamma in IBZ q.
      long iq_gamma_ibz = -1;
      if (_div_treatment == "ignore_g0") {
        auto Qp = MF.Qpts_ibz();
        if (Qp.shape()[0] >= 1) {
          double norm0 = 0.0;
          for (long c = 0; c < Qp.shape()[1]; ++c) norm0 += std::abs(Qp(0, c));
          if (norm0 < 1e-10) iq_gamma_ibz = 0;
        }
      }

      // Step 4: per-isym aux Σ_x partial -> orbital Σ_x at IBZ k, with D rotation.
      nda::array<ComplexType, 4> Sigma_x_orb(ns, Nk_ibz, nbnd, nbnd);
      Sigma_x_orb = ComplexType(0.0, 0.0);

      nda::array<ComplexType, 3> SxA_dummy_PQ(NP_loc_hf, NQ_loc_hf, 1);
      nda::array<ComplexType, 3> SxA_orb_at_ks_3d(1, nbnd, nbnd);
      nda::array<ComplexType, 2> Tm(nbnd, nbnd);

      range Pr(P0_hf, P0_hf + NP_loc_hf);
      range Qr(Q0_hf, Q0_hf + NQ_loc_hf);

      const long Nsymq = static_cast<long>(MF.qsymms().shape()[0]);
      auto qp_to_ibz_arr = MF.qp_to_ibz();
      auto qp_trev_arr_l = MF.qp_trev();
      auto qk_to_k2_l    = MF.qk_to_k2();
      const double inv_Nq = 1.0 / static_cast<double>(Nq);

      for (long isym = 0; isym < Nsymq; ++isym) {
        const long nqs_isym = MF.nq_per_s(isym);
        for (long s = 0; s < ns; ++s) {
          for (long ik_ibz = 0; ik_ibz < Nk_ibz; ++ik_ibz) {
            const long ks = MF.ks_to_k(isym, ik_ibz);

            // Accumulate aux Σ_x partial: -1/Nq * V(qs)_loc * n_aux(ksmqs)_loc.
            SxA_dummy_PQ = ComplexType(0.0, 0.0);
            for (long iq = 0; iq < nqs_isym; ++iq) {
              const long qp = MF.Qs(isym, iq);
              const long qs = qp_to_ibz_arr(qp);
              if (qs == iq_gamma_ibz) continue;
              utils::check(qp_trev_arr_l(qp) == 0,
                           "real_axis_hf_t::evaluate (isym): qp_trev(qp={}) "
                           "!= 0; q-side TR branch not yet implemented.", qp);
              // kp_trev on ksmqs is handled by the n_aux conj-copy above.
              const long ksmqs = qk_to_k2_l(qs, ks);
              for (long iP = 0; iP < NP_loc_hf; ++iP)
                for (long iQ = 0; iQ < NQ_loc_hf; ++iQ)
                  SxA_dummy_PQ(iP, iQ, 0) -=
                      inv_Nq * V_qPQ_loc_ibz(qs, iP, iQ)
                             * n_aux_skPQ_fbz(s, ksmqs, iP, iQ);
            }

            // aux_to_primary at X(FBZ ks).
            auto X_P_slice = X_skPmu(s, ks, Pr, _);
            auto X_Q_slice = X_skPmu(s, ks, Qr, _);
            aux_to_primary_one_k(X_P_slice, X_Q_slice,
                                 SxA_dummy_PQ, SxA_orb_at_ks_3d);

            // Accumulate into orbital Σ_x at IBZ k, with D rotation if isym!=0.
            if (isym == 0) {
              for (long mu_i = 0; mu_i < nbnd; ++mu_i)
                for (long nu = 0; nu < nbnd; ++nu)
                  Sigma_x_orb(s, ik_ibz, mu_i, nu) +=
                      SxA_orb_at_ks_3d(0, mu_i, nu);
            } else {
              auto [cjg, D] = MF.symmetry_rotation(isym, ik_ibz);
              utils::check(not cjg,
                           "real_axis_hf_t::evaluate (isym): "
                           "symmetry_rotation cjg=true branch not supported.");
              using math::sparse::csrmm;
              auto Sx_at_ks_2d = SxA_orb_at_ks_3d(0, _, _);
              csrmm<'H'>(ComplexType(1.0, 0.0), *D, Sx_at_ks_2d,
                         ComplexType(0.0, 0.0), Tm);
              csrmm<'T'>(ComplexType(1.0, 0.0), *D, nda::transpose(Tm),
                         ComplexType(1.0, 0.0),
                         nda::transpose(Sigma_x_orb(s, ik_ibz, _, _)));
            }
          }
        }
      } // for isym

      // Allreduce orbital Σ_x across (P, Q) ranks.
      if (comm.size() > 1)
        comm.all_reduce_in_place_n(Sigma_x_orb.data(), Sigma_x_orb.size(),
                                   std::plus<>{});

      // Write to state.
      if (state.Sigma_x_skij->node_comm()->root())
        state.Sigma_x_skij->local() = Sigma_x_orb;
      state.Sigma_x_skij->node_sync();

      state.mu_chem = mu;
      return;
    }

    // === Non-isym (IBZ == FBZ) path: existing FBZ-direct sum. ===
    // evaluate_Sigma_x_serial allreduces Sigma_x to per-rank-replicated;
    // we then copy into the sArray on node root and sync.
    nda::array<ComplexType, 4> Sigma_x_local(ns, Nk, nbnd, nbnd);
    Sigma_x_local() = ComplexType(0.0, 0.0);
    evaluate_Sigma_x_serial(comm, grid_at_mu, A_drv, X_skPmu, V_qPQ_loc,
                            kmq_to_kp, Sigma_x_local, iq_gamma);
    if (state.Sigma_x_skij->node_comm()->root())
      state.Sigma_x_skij->local() = Sigma_x_local;
    state.Sigma_x_skij->node_sync();

    state.mu_chem = mu;
  }

  real_freq_grid_t const& grid() const noexcept { return *_grid; }

private:
  real_freq_grid_t const* _grid;
  std::string             _div_treatment;
};

using real_axis_hf_t = real_axis_hf_base_t<HOST_MEMORY>;

} // namespace real_axis
} // namespace methods

#endif // COQUI_REAL_AXIS_HF_T_H
