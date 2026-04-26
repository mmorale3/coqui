/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 * ==========================================================================
 */

#ifndef COQUI_REAL_AXIS_GW_T_H
#define COQUI_REAL_AXIS_GW_T_H

#include <string>
#include <memory>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_conv.hpp"
#include "methods/GW_real_axis/real_axis_mb_state.hpp"
#include "methods/GW_real_axis/real_axis_pi.hpp"
#include "methods/GW_real_axis/real_axis_dyson.hpp"
#include "methods/GW_real_axis/real_axis_sigma.hpp"

namespace methods {
namespace solvers {

/**
 * Finite-temperature real-axis ISDF/THC GW solver.
 *
 * Mirrors the public-API shape of methods::solvers::gw_t but operates on
 * methods::real_axis::real_axis_mb_state_t and a real_freq_grid_t (real
 * frequencies + uniform conjugate time grid + finite-T parameters beta,
 * mu_chem). The single SCF iteration follows Algorithm 1 in the
 * accompanying methods notes (notes/isdf_gw_prb_draft_v2.tex).
 *
 * Self-consistency policy:
 *   - max_iter = 1 (default) corresponds to one-shot G_0 W_0;
 *   - max_iter > 1 with mixing performs scGW.
 *
 * Causality projection is applied at the end of each iteration:
 *   - Im Sigma^{c,R}(w) <- min(Im Sigma^{c,R}(w), 0)
 *   - sgn(Omega) Im Pi^R(Omega) <- min(.,0)
 * before recomputing the corresponding real parts via Hilbert transforms.
 *
 * NOTE on integration with the rest of CoQui:
 *   The full self-consistency loop requires the THC ERI (X, Y, V_PQ) and an
 *   MPI-distributed iteration over k and q points. The standalone class here
 *   provides:
 *     - one-iteration structure (compute Pi -> W -> Sigma_c -> update G)
 *     - causality projection
 *     - hooks for Z (THC) tensor contractions
 *   Connecting the THC factor inputs and the BZ/IBZ machinery is the next
 *   integration step (see TODO markers in the body of the iteration). For
 *   testing purposes the kernels (accumulate_ImPi_one_kq,
 *   accumulate_ImSigma_one_kq, solve_dyson_W_aux) are exercised directly via
 *   their dedicated unit tests.
 */
class real_axis_gw_t {
public:

  /**
   * @param grid        real-frequency grid (finite T, mu_chem, time grid)
   * @param max_iter    maximum number of SCF iterations (1 -> G_0W_0)
   * @param mix         linear mixing coefficient for Sigma updates
   * @param eps_nufft   FINUFFT accuracy tolerance
   * @param ntrans_aux  ntrans of the NUFFT plans, sized to the largest
   *                    Naux*Naux batched cross-correlation.
   * @param output      filename prefix for diagnostic outputs
   */
  real_axis_gw_t(real_axis::real_freq_grid_t const& grid,
                 long max_iter   = 1,
                 double mix      = 0.5,
                 double eps_nufft = 1e-10,
                 long ntrans_aux  = 1,
                 std::string output = "coqui_real_axis")
    : _grid(&grid)
    , _max_iter(max_iter)
    , _mix(mix)
    , _output(std::move(output))
    , _conv(std::make_shared<real_axis::real_axis_conv_t>(
              grid, ntrans_aux, eps_nufft))
  {}

  ~real_axis_gw_t() = default;

  long max_iter() const  { return _max_iter; }
  double mix()    const  { return _mix; }
  std::string const& output() const { return _output; }

  /// Apply causality projection to a fermionic Im Sigma array
  /// (any leading dims, last dim = N_w). Modifies in place.
  template<typename ArrayT>
  void apply_causality_fermionic(ArrayT && Im_w) const {
    // Im Sigma^{c,R}(w) must be <= 0 on the real axis. Clip positive noise.
    auto shape = Im_w.shape();
    auto * data = Im_w.data();
    long total = 1;
    for (long d = 0; d < (long)shape.size(); ++d) total *= shape[d];
    for (long i = 0; i < total; ++i) {
      if (data[i].real() > 0.0)
        data[i] = ComplexType(0.0, data[i].imag());
    }
  }

  /// Apply causality projection to a bosonic Im Pi array. The diagonal of
  /// Im Pi must satisfy sgn(Omega) Im Pi(Omega) <= 0 on the diagonal; we
  /// clip violations.
  template<typename ArrayT>
  void apply_causality_bosonic_diag(ArrayT && ImPi_q_O_PQ,
                                    nda::array<double,1> const& Omega) const {
    auto sh = ImPi_q_O_PQ.shape();
    const long Nq  = sh[0];
    const long NO  = sh[1];
    const long Naux = sh[2];
    for (long iq = 0; iq < Nq; ++iq)
      for (long iO = 0; iO < NO; ++iO) {
        const double sO = (Omega(iO) > 0 ? 1.0 : -1.0);
        for (long P = 0; P < Naux; ++P) {
          auto v = ImPi_q_O_PQ(iq, iO, P, P).real();
          if (sO * v > 0.0)
            ImPi_q_O_PQ(iq, iO, P, P) =
                ComplexType(0.0, ImPi_q_O_PQ(iq, iO, P, P).imag());
        }
      }
  }

  /**
   * Solve the bosonic Dyson equation W = (I - V Pi)^{-1} V across all (q, Omega)
   * grid points using state.RePi_qOmegaPQ + i state.ImPi_qOmegaPQ as input
   * and writing state.ReW_qOmegaPQ + i state.ImW_qOmegaPQ as output.
   *
   * @param V_qPQ   (Nq, Naux, Naux) auxiliary Coulomb matrices, complex.
   */
  void solve_W(real_axis::real_axis_mb_state_t & state,
               nda::array<ComplexType, 3> const& V_qPQ) const
  {
    utils::check(state.ImPi_qOmegaPQ.has_value() and state.RePi_qOmegaPQ.has_value(),
                 "real_axis_gw_t::solve_W: state Pi arrays not allocated");
    utils::check(state.ImW_qOmegaPQ.has_value() and state.ReW_qOmegaPQ.has_value(),
                 "real_axis_gw_t::solve_W: state W arrays not allocated");

    auto const& ImPi = *state.ImPi_qOmegaPQ;
    auto const& RePi = *state.RePi_qOmegaPQ;
    auto      & ImW  = *state.ImW_qOmegaPQ;
    auto      & ReW  = *state.ReW_qOmegaPQ;

    const long Nq    = ImPi.shape()[0];
    const long NOm   = ImPi.shape()[1];
    const long Naux  = ImPi.shape()[2];

    nda::array<ComplexType, 2> Vmat(Naux, Naux);
    nda::array<ComplexType, 2> Pi(Naux, Naux);
    nda::array<ComplexType, 2> W(Naux, Naux);

    for (long iq = 0; iq < Nq; ++iq) {
      for (long P = 0; P < Naux; ++P)
        for (long Q = 0; Q < Naux; ++Q)
          Vmat(P, Q) = V_qPQ(iq, P, Q);
      for (long iO = 0; iO < NOm; ++iO) {
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q)
            Pi(P, Q) = ComplexType(RePi(iq, iO, P, Q).real(),
                                   ImPi(iq, iO, P, Q).real());
        real_axis::solve_dyson_W_aux(Vmat, Pi, W);
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q) {
            ReW(iq, iO, P, Q) = ComplexType(W(P, Q).real(), 0.0);
            ImW(iq, iO, P, Q) = ComplexType(W(P, Q).imag(), 0.0);
          }
      }
    }
  }

  real_axis::real_axis_conv_t & conv() { return *_conv; }

  real_axis::real_freq_grid_t const& grid() const { return *_grid; }

private:
  real_axis::real_freq_grid_t const* _grid;
  long _max_iter;
  double _mix;
  std::string _output;
  std::shared_ptr<real_axis::real_axis_conv_t> _conv;
};

} // namespace solvers
} // namespace methods

#endif // COQUI_REAL_AXIS_GW_T_H
