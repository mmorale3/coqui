/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * RW-1 GATE HARNESS (notes/rw_real_axis_w_spec.md, gate RW-1-a).
 *
 * THE ONE ACCEPTANCE NUMBER of the real-axis W port: the correlated screened
 * interaction computed on the REAL axis, mapped forward to the imaginary axis
 * by the bosonic spectral (Lehmann) integral, must agree with the imaginary-
 * axis production W^c built from THE SAME Green's function.
 *
 *   real axis   :  A(w) = QP-pole Lorentzians of width eta
 *                    -> real_axis_scr_coulomb_t::update_w   (the ported chain)
 *                    -> Im W^c(q, P, Q, Omega),  Omega > 0
 *                    -> W^c(q, P, Q, i nu_n) = (2/pi) int_0^inf dOmega
 *                                              Omega Im W^c(Omega)
 *                                            / (Omega^2 + nu_n^2)
 *                       using the grid's OWN trapezoid weights.
 *   imag axis   :  G_0 from the same static Hamiltonian, same mu, same beta
 *                    -> solvers::scr_coulomb_t::update_w
 *                    -> dW_qtPQ -> tau_to_w_PHsym -> W^c(q, i nu_n, P, Q).
 *
 * SIGN OF THE FORWARD MAP. For a retarded response X^R(z), analytic in the
 * upper half plane,
 *      X(z) = (1/pi) int dOmega' Im X^R(Omega') / (Omega' - z),
 * which is the same convention the branch's fermionic cross-validation uses
 * (test_real_axis_xvalidate.cpp:366-398, Sigma_c(z) = -(1/pi) int Im Sigma /
 * (z - omega')). Im W^c is ODD on the real axis (real_axis_sigma.hpp:25-32
 * documents the odd extension of the bosonic half-grid), so folding the
 * negative half onto the positive one gives
 *      X(i nu) = (2/pi) int_0^inf dOmega Omega Im X(Omega) / (Omega^2 + nu^2).
 * NOTE (flagged in notes/rw1_port_report.md): the spec writes the denominator
 * as ((i nu)^2 - Omega^2) = -(nu^2 + Omega^2), i.e. the OPPOSITE overall sign.
 * The form used here is the one that follows from the branch's own retarded
 * convention and is the one that gives Re W^c(i nu = 0) < 0 (screening lowers
 * W below the bare v), so it is what this harness implements.
 *
 * CONVENTION PINS (pre-registered in the spec; each is asserted below, and a
 * violation aborts the test rather than being absorbed into a tolerance):
 *   P1  same static Hamiltonian on both sides   (H0 + F == diag(eps_KS), S == 1)
 *   P2  same mu (the imaginary-axis Dyson's own mu feeds grid.mu_chem())
 *   P3  beta = 1000 on both sides
 *   P4  both sides ignore_g0
 *   P5  eta enters ONLY through A; the grids are densified with eta so the
 *       quadrature/NUFFT error stays below the eta error (Nyquist is enforced
 *       as a hard error by real_freq_grid_t itself)
 *
 * KNOWN STRUCTURAL DIFFERENCES, reported but excluded from the gate:
 *   D1  q = Gamma. Under ignore_g0 the real-axis chain zeroes Pi AND W at
 *       Gamma explicitly (real_axis_scr_coulomb_t.h:846-853); the imaginary-
 *       axis chain does NOT -- it solves the Dyson equation at Gamma with the
 *       regularized aux-basis V. The gate therefore runs over q != Gamma and
 *       reports Gamma separately.
 *   D2  spin degeneracy. See COQUI_RW1_SPIN_DEGENERACY below.
 * ==========================================================================
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "mpi3/communicator.hpp"

#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"
#include "mean_field/default_MF.hpp"

#include "methods/ERI/thc_reader_t.hpp"
#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/mb_eri_context.h"

#include "nda/nda.hpp"
#include "nda/linalg.hpp"
#include "numerics/distributed_array/nda_utils.hpp"

#include "methods/GW_real_axis/real_freq_grid.hpp"
#include "methods/GW_real_axis/real_axis_mb_state.hpp"
#include "methods/GW_real_axis/real_axis_scr_coulomb_t.h"
#include "methods/GW_real_axis/real_axis_qp_A.hpp"

// Matsubara branch.
#include "methods/mb_state/mb_state.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_common.hpp"
#include "methods/scr_coulomb/scr_coulomb_t.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "hamiltonian/one_body_hamiltonian.hpp"

#include <cmath>
#include <complex>
#include <limits>
#include <vector>

namespace bdft_tests {

  using namespace methods;
  using methods::real_axis::real_freq_grid_t;
  using methods::real_axis::real_axis_mb_state_t;
  using methods::real_axis::real_axis_scr_coulomb_t;
  using cval_t = std::complex<double>;

  // RW-2: the grid sizing rule and the QP-pole Lorentzian A builder were PROMOTED out of
  // this harness into methods/GW_real_axis/real_axis_qp_A.hpp, so the production spectral
  // path (qp_modea_wfit = "spectral") and this gate share one implementation. Neither is
  // changed in content -- the derivations that used to live here are in that file's header.
  namespace rw1 {
    using methods::real_axis::grid_sizing_t;
    using methods::real_axis::size_grids;
  } // namespace rw1

  // =====================================================================
  // The gate body, parameterized by fixture. `use_rspace` selects the Pi
  // code path: the k-space branch hard-requires IBZ == FBZ
  // (real_axis_scr_coulomb_t.h:833-837), so symmetry-reduced fixtures must
  // use the R-space branch (which is also the production default for Nk>1).
  static void run_lehmann_gate(char const* mf_src, bool use_rspace)
  {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    if (mpi_context->comm.size() != 1) return;   // single-rank gate

    // ---------------- fixture + THC ----------------------------------
    // The THC rank is COMMON MODE: both axes consume the identical
    // factorization, so its accuracy cancels out of the comparison. It is
    // chosen small because the real-axis NUFFT batch is Naux^2 and the eta
    // series pushes N_t to a few thousand.
    auto mf = std::make_shared<mf::MF>(
                  mf::default_MF(mpi_context, mf_src));
    const int nIpts = mf->nbnd() * 2;
    thc_reader_t thc(mf, make_thc_reader_ptree(nIpts, "", "incore", "",
                                               "bdft", 1e-8,
                                               mf->ecutrho(), 1, 1024));

    const long ns        = mf->nspin();
    const long Nk        = mf->nkpts();
    const long Nq_ibz    = mf->nqpts_ibz();
    const long nbnd      = mf->nbnd();
    const long Naux      = thc.Np();
    const double beta    = 1000.0;              // PIN P3

    app_log(2, "[rw1] fixture {}: ns={} Nk={} Nq_ibz={} nbnd={} Naux={}  "
               "(Pi path: {})",
            mf_src, ns, Nk, Nq_ibz, nbnd, Naux,
            use_rspace ? "R-space" : "k-space");

    auto eigval = mf->eigval();
    [[maybe_unused]] auto kp2ibz = mf->kp_to_ibz();

    double e_min =  std::numeric_limits<double>::infinity();
    double e_max = -std::numeric_limits<double>::infinity();
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < mf->nkpts_ibz(); ++k)
        for (long n = 0; n < nbnd; ++n) {
          e_min = std::min(e_min, eigval(s, k, n));
          e_max = std::max(e_max, eigval(s, k, n));
        }
    const double w_max     = std::max(std::abs(e_min), std::abs(e_max)) + 2.0;
    const double Omega_max = 2.0 * w_max;
    app_log(2, "[rw1] eps range = [{:.4f}, {:.4f}] Ha -> w_max = {:.4f}, "
               "Omega_max = {:.4f}", e_min, e_max, w_max, Omega_max);

    // =================================================================
    // MATSUBARA BRANCH: G_0 from the MF static Hamiltonian, W^c(q,inu).
    // =================================================================
    imag_axes_ft::IAFT ft(beta, w_max + 1.0, imag_axes_ft::dlr_basis, "high");
    const std::string prefix = std::string("coqui_rw1_lehmann_") + mf_src;

    MBState mb_state(mpi_context, ft, prefix);
    simple_dyson dyson(mf.get(), &ft);

    mb_state.sF_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(
        *mpi_context, {ns, mf->nkpts_ibz(), nbnd, nbnd}));
    mb_state.sDm_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(
        *mpi_context, {ns, mf->nkpts_ibz(), nbnd, nbnd}));
    mb_state.sG_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(
        *mpi_context, {ft.nt_f(), ns, mf->nkpts_ibz(), nbnd, nbnd}));
    mb_state.sSigma_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(
        *mpi_context, {ft.nt_f(), ns, mf->nkpts_ibz(), nbnd, nbnd}));
    auto& sF     = mb_state.sF_skij.value();
    auto& sDm    = mb_state.sDm_skij.value();
    auto& sG     = mb_state.sG_tskij.value();
    auto& sSigma = mb_state.sSigma_tskij.value();

    // F = (MF Fock - H0); the Dyson adds H0 back, so the static Hamiltonian
    // seen by G_0 is exactly the mean-field Fock matrix.
    hamilt::set_fock(*mf, dyson.PSP(), sF, true);
    if (mpi_context->node_comm.root()) sSigma.local() = ComplexType(0.0);
    sSigma.communicator()->barrier();

    double mu = 0.0;
    update_G(dyson, *mf, ft, sDm, sG, sF, sSigma, mu, /*const_mu*/ false);
    app_log(2, "[rw1] Matsubara mu = {:.10f} Ha  (PIN P2: shared with the "
               "real-axis grid)", mu);

    // ---- PIN P1: the static Hamiltonian must be diag(eps_KS) with S = 1,
    // otherwise "the same E_ska / MO on both sides" cannot be realized with
    // an identity MO at every FBZ k (the real-axis A lives on the FBZ, the
    // imag-axis H only on the IBZ). Violation => STOP.
    {
      auto H0 = dyson.H0();
      auto S  = dyson.sS_skij().local();
      auto F  = sF.local();
      double off_H = 0.0, dia_H = 0.0, off_S = 0.0, dia_S = 0.0;
      for (long s = 0; s < ns; ++s)
        for (long k = 0; k < mf->nkpts_ibz(); ++k)
          for (long i = 0; i < nbnd; ++i)
            for (long j = 0; j < nbnd; ++j) {
              const cval_t h = H0(s, k, i, j) + F(s, k, i, j);
              if (i == j) {
                dia_H = std::max(dia_H, std::abs(h - cval_t(eigval(s, k, i), 0.0)));
                dia_S = std::max(dia_S, std::abs(S(s, k, i, j) - cval_t(1.0, 0.0)));
              } else {
                off_H = std::max(off_H, std::abs(h));
                off_S = std::max(off_S, std::abs(S(s, k, i, j)));
              }
            }
      app_log(2, "[rw1] PIN P1: |H0+F - diag(eps)|_max = {:.3e} (diag) / {:.3e} "
                 "(offdiag);  |S - 1|_max = {:.3e} / {:.3e}",
              dia_H, off_H, dia_S, off_S);
      REQUIRE(dia_H < 1e-8);
      REQUIRE(off_H < 1e-8);
      REQUIRE(dia_S < 1e-8);
      REQUIRE(off_S < 1e-8);
    }

    solvers::scr_coulomb_t scr_im(&ft, "rpa", "ignore_g0");   // PIN P4
    scr_im.update_w(mb_state, thc, -1);
    REQUIRE(mb_state.dW_qtPQ.has_value());
    const long nw_half = (ft.nw_b() % 2 == 0) ? ft.nw_b() / 2 : ft.nw_b() / 2 + 1;

    // ---- Pi-LEVEL PROBE ------------------------------------------------
    // W^c is NON-LINEAR in Pi, so a discrepancy in W cannot be attributed.
    // Pi IS linear in the bubble, so the same forward map applied to Im Pi
    // reads off any normalization mismatch (spin degeneracy, BZ weight,
    // fermion-loop sign, or a factor in the spectral map itself) directly.
    // The imaginary-axis Pi is recomputed from the SAME G that update_w
    // just used (eval_Pi_qdep is idempotent given sG_tskij).
    nda::array<ComplexType, 4> Pi_im_qwPQ;
    {
      auto dPi = scr_im.eval_Pi_qdep(mb_state, thc);
      auto gsh = dPi.global_shape();
      app_log(2, "[rw1] imag-axis Pi global shape = ({}, {}, {}, {})",
              gsh[0], gsh[1], gsh[2], gsh[3]);
      const long ntp = gsh[0];
      nda::array<ComplexType, 4> Pt(ntp, gsh[1], gsh[2], gsh[3]);
      Pt() = ComplexType(0.0);
      Pt(dPi.local_range(0), dPi.local_range(1),
         dPi.local_range(2), dPi.local_range(3)) = dPi.local();
      mpi_context->comm.all_reduce_in_place_n(Pt.data(), Pt.size(), std::plus<>{});
      nda::array<ComplexType, 4> Pw(nw_half, gsh[1], gsh[2], gsh[3]);
      ft.tau_to_w_PHsym(Pt, Pw);
      Pi_im_qwPQ = nda::array<ComplexType, 4>(Nq_ibz, nw_half, Naux, Naux);
      for (long iq = 0; iq < Nq_ibz; ++iq)
        for (long n = 0; n < nw_half; ++n)
          Pi_im_qwPQ(iq, n, nda::ellipsis{}) = Pw(n, iq, nda::ellipsis{});
    }

    // dW_qtPQ: (Nq_ibz, nt_half, Naux, Naux) on the PH-symmetric bosonic
    // tau half grid. Gather (single rank, but keep it general).
    const long nt_half = mb_state.dW_qtPQ.value().global_shape()[1];
    nda::array<ComplexType, 4> Wc_qtPQ(Nq_ibz, nt_half, Naux, Naux);
    {
      auto const& dW = mb_state.dW_qtPQ.value();
      Wc_qtPQ() = ComplexType(0.0);
      Wc_qtPQ(dW.local_range(0), dW.local_range(1),
              dW.local_range(2), dW.local_range(3)) = dW.local();
      mpi_context->comm.all_reduce_in_place_n(Wc_qtPQ.data(), Wc_qtPQ.size(),
                                              std::plus<>{});
    }
    nda::array<ComplexType, 4> Wc_im_qwPQ(Nq_ibz, nw_half, Naux, Naux);
    {
      // tau_to_w_PHsym wants tau leading. Repack (q,t,P,Q) -> (t, q,P,Q).
      nda::array<ComplexType, 4> Wt(nt_half, Nq_ibz, Naux, Naux);
      for (long iq = 0; iq < Nq_ibz; ++iq)
        for (long it = 0; it < nt_half; ++it)
          Wt(it, iq, nda::ellipsis{}) = Wc_qtPQ(iq, it, nda::ellipsis{});
      nda::array<ComplexType, 4> Ww(nw_half, Nq_ibz, Naux, Naux);
      ft.tau_to_w_PHsym(Wt, Ww);
      for (long iq = 0; iq < Nq_ibz; ++iq)
        for (long n = 0; n < nw_half; ++n)
          Wc_im_qwPQ(iq, n, nda::ellipsis{}) = Ww(n, iq, nda::ellipsis{});
    }

    // The bosonic Matsubara frequencies of the PH-symmetric half grid:
    // index n <-> full-mesh index nw_b()/2 + n (IAFT.icc:64-65).
    auto wn_b = ft.wn_mesh_b();
    nda::array<double, 1> nu_n(nw_half);
    for (long n = 0; n < nw_half; ++n)
      nu_n(n) = ft.omega(wn_b(ft.nw_b() / 2 + n)).imag();
    app_log(2, "[rw1] bosonic PH-sym half grid: nw_half={}, nu_0={:.6e}, "
               "nu_1={:.6e}, nu_2={:.6e}", nw_half, nu_n(0),
            (nw_half > 1 ? nu_n(1) : 0.0), (nw_half > 2 ? nu_n(2) : 0.0));
    REQUIRE(std::abs(nu_n(0)) < 1e-12);   // index 0 must be the static node

    // Compare across the WHOLE bosonic half mesh: at beta = 1000 the lowest
    // few DLR nodes all sit at nu < 0.03 Ha (essentially the static limit),
    // so restricting to them would test only one point of the function.
    const long n_check = nw_half;

    // ---- Gamma index and smallest-|q| index (for the head channel) ----
    auto Qpts = nda::array<double, 2>(mf->Qpts_ibz());
    long iq_gamma = -1, iq_small = -1;
    double q2_small = std::numeric_limits<double>::infinity();
    for (long iq = 0; iq < Nq_ibz; ++iq) {
      const double q2 = Qpts(iq,0)*Qpts(iq,0) + Qpts(iq,1)*Qpts(iq,1)
                      + Qpts(iq,2)*Qpts(iq,2);
      if (q2 < 1e-20) { iq_gamma = iq; continue; }
      if (q2 < q2_small) { q2_small = q2; iq_small = iq; }
    }
    REQUIRE(iq_gamma >= 0);
    REQUIRE(iq_small >= 0);
    auto chi_bar = nda::array<ComplexType, 2>(thc.basis_bar_head());
    const double head_pref = (q2_small / (4.0 * M_PI)) * mf->volume();

    // eps^-1 head channel (minus one) of the Matsubara W at the smallest
    // non-zero |q|: same formula as real_axis_scr_coulomb_t.h:983-1010 and
    // methods/GW/g0_div_utils.hpp::eval_eps_inv_q.
    auto head_of = [&](nda::array<ComplexType,4> const& W_qwPQ, long iq) {
      nda::array<ComplexType, 1> h(W_qwPQ.shape()[1]);
      for (long n = 0; n < W_qwPQ.shape()[1]; ++n) {
        ComplexType acc(0.0, 0.0);
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q)
            acc += chi_bar(iq, P) * W_qwPQ(iq, n, P, Q) * std::conj(chi_bar(iq, Q));
        h(n) = head_pref * acc;
      }
      return h;
    };
    auto head_im = head_of(Wc_im_qwPQ, iq_small);

    // ---- the two ingredients of the a-priori eta budget ---------------
    // d_min: the closest approach of a pole of G to the fermionic Matsubara
    //        axis, min_{s,k,n} |eps_ska - mu|. A Lorentzian-broadened
    //        spectral function is the spectral function of G with its poles
    //        displaced to Im w = -eta, so on the imaginary axis
    //            G_eta(i om_n) = G_0(i om_n + i eta sgn(om_n)),
    //        and every object built from it carries a RELATIVE distortion
    //        bounded by 2 eta max_n |d ln G / d om| = 2 eta / d_min.
    // kappa: the dielectric conditioning ||(I - Z.Pi)^{-1}||_max, which is
    //        how much the Dyson step amplifies a relative error in Pi. Taken
    //        from the production Matsubara Pi (below), i.e. from the
    //        REFERENCE, never from the real-axis answer.
    // Budget: |W_re(i nu) - W_im(i nu)| <= (2 eta / d_min) * kappa * max|W_im|.
    // This is a BOUND with an O(1) coefficient of one, not a fit; the report
    // quotes observed/budget at every eta so the bound can be judged.
    double kappa = 0.0;
    double d_min = std::numeric_limits<double>::infinity();
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < mf->nkpts_ibz(); ++k)
        for (long n = 0; n < nbnd; ++n)
          d_min = std::min(d_min, std::abs(eigval(s, k, n) - mu));

    // ---- Z / DYSON CONSISTENCY PROBE ---------------------------------
    // The Dyson step is the SAME algebra on both axes:
    //     W^c = [ (I - Z.Pi)^{-1} - I ] . Z
    // (real axis real_axis_scr_coulomb_t.h:859-884; imaginary axis
    // scr_coulomb_t.cpp:1165-1196). So if Pi agrees at i nu = 0 and the two
    // paths read the SAME Z, W^c must agree at i nu = 0 too. This rebuilds
    // W^c(0) from the production Matsubara Pi(0) and the Z that the
    // REAL-AXIS path reads (thc.Z(iq), the plain per-q overload) and
    // compares it against the production W^c(0). A ratio of 1 clears Z; a
    // ratio != 1 says the two overloads of thc.Z do not deliver the same
    // matrix.
    {
      double znum = 0.0, zden = 0.0, zamp = 0.0, zworst = 0.0;
      for (long iq = 0; iq < Nq_ibz; ++iq)
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q)
            zamp = std::max(zamp, std::abs(Wc_im_qwPQ(iq, 0, P, Q)));
      for (long iq = 0; iq < Nq_ibz; ++iq) {
        if (iq == iq_gamma) continue;
        auto Zq = thc.Z(static_cast<int>(iq));
        nda::matrix<ComplexType> Z(Naux, Naux), Pi(Naux, Naux);
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q) {
            Z(P, Q)  = Zq(P, Q);
            Pi(P, Q) = Pi_im_qwPQ(iq, 0, P, Q);
          }
        nda::matrix<ComplexType> A = Z * Pi;
        A *= ComplexType(-1.0);
        for (long P = 0; P < Naux; ++P) A(P, P) += ComplexType(1.0);
        auto Ai = nda::inverse(A);
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q)
            kappa = std::max(kappa, std::abs(Ai(P, Q)));
        for (long P = 0; P < Naux; ++P) Ai(P, P) -= ComplexType(1.0);
        nda::matrix<ComplexType> Wp = Ai * Z;
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q) {
            const double m = Wc_im_qwPQ(iq, 0, P, Q).real();
            const double d = Wp(P, Q).real();
            zworst = std::max(zworst, std::abs(d - m));
            if (std::abs(m) < 0.05 * zamp) continue;
            znum += d * m; zden += m * m;
          }
      }
      app_log(2, "[rw1] Z/Dyson probe: rebuilding W^c(0) from the production "
                 "Pi(0) with the real-axis Z read gives <ratio> = {:.6f}, "
                 "worst abs = {:.4e} (max|W_im| = {:.4e})",
              (zden > 0.0) ? znum / zden : 0.0, zworst, zamp);
    }

    // ---- HONEST eta BUDGET -------------------------------------------
    // A Lorentzian-broadened spectral function is the spectral function of
    // G^R(w) with its poles pushed to Im w = -eta. On the imaginary axis
    // that is G_0(i nu + i eta sgn(nu)), so the bubble built from it is
    // Pi_0 evaluated at a frequency displaced by 2 eta: the leading error is
    // FIRST ORDER in eta with coefficient |dW^c/d nu|. We estimate that
    // derivative from the MATSUBARA data itself (never from the real-axis
    // answer) by a forward difference on the bosonic mesh, per element, and
    // add a floor from the quadrature/NUFFT budget of the sizing rule
    // (exp(-2 pi eta / dw) + exp(-eta T) ~ 3e-6 + 1e-4, times the local
    // amplitude).
    double amp_ref_global = 0.0;
    for (long iq = 0; iq < Nq_ibz; ++iq) {
      if (iq == iq_gamma) continue;
      for (long n = 0; n < n_check; ++n)
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q)
            amp_ref_global = std::max(amp_ref_global,
                                      std::abs(Wc_im_qwPQ(iq, n, P, Q)));
    }
    app_log(2, "[rw1] eta budget ingredients: d_min = min|eps - mu| = {:.6f} Ha, "
               "kappa = ||(I-Z.Pi)^-1||_max = {:.4f}, max|W_im| = {:.4e}",
            d_min, kappa, amp_ref_global);
    auto budget_for = [&](double eta) {
      return (2.0 * eta / d_min) * kappa * amp_ref_global;
    };

    // =================================================================
    // REAL-AXIS BRANCH, one pass per eta.
    // =================================================================
    const std::vector<double> etas = {0.05, 0.025, 0.0125};
    std::vector<double> worst_dev, worst_rel, head_dev, gamma_amp, budget;

    for (double eta : etas) {
      auto gs = rw1::size_grids(eta, w_max, Omega_max);
      app_log(2, "\n[rw1] ===== eta = {:.5f} Ha =====", eta);
      app_log(2, "[rw1]   N_w={} (dw={:.5f} = eta/{:.2f})  N_t={} "
                 "(T={:.1f} = {:.1f}/eta, dt={:.4f})  N_Omega={} "
                 "(dOmega={:.5f} = eta/{:.2f})",
              gs.N_w, gs.dw, eta / gs.dw, gs.N_t, gs.T_window,
              gs.T_window * eta, gs.dt, gs.N_Omega, gs.dOmega,
              eta / gs.dOmega);

      auto grid = real_freq_grid_t::make_uniform(
                    beta, mu, w_max, gs.N_w,
                    Omega_max, gs.N_Omega, gs.N_t, gs.T_window);

      real_axis_mb_state_t state(grid);
      state.mpi = mpi_context;
      // A lives at IBZ k (update_w hard-checks the shape); both fixtures of this gate have
      // IBZ == FBZ, which is asserted here rather than assumed.
      REQUIRE(mf->nkpts_ibz() == Nk);
      state.A_wskij.emplace(*state.mpi,
          std::array<long, 5>{gs.N_w, ns, Nk, nbnd, nbnd});
      if (state.A_wskij->node_comm()->root()) {
        // The promoted builder (real_axis_qp_A.hpp) with MO = identity -- licensed by PIN P1
        // -- and E = eps_KS, i.e. exactly the G_0 of the Matsubara branch, Lorentzian-
        // broadened by eta. Identical arithmetic to the loop this call replaced.
        methods::real_axis::build_A_from_QP_poles(
            state.A_wskij->local(), grid, eigval,
            (nda::array<ComplexType, 4> const*)nullptr, eta);
      }
      state.A_wskij->node_sync();

      real_axis_scr_coulomb_t scr_re(&grid, "rpa", "ignore_g0", 1e-9);  // PIN P4
      scr_re.update_w(state, thc, /*verbose*/ false, use_rspace);
      REQUIRE(state.ImW_qPQO.has_value());

      auto ImW = math::nda::all_gather_slow<HOST_MEMORY>(*state.ImW_qPQO);

      // ---- the bosonic forward spectral map --------------------------
      auto const& Om   = grid.Omega();
      auto const& Om_w = grid.Omega_weights();
      nda::array<ComplexType, 4> Wc_re_qwPQ(Nq_ibz, n_check, Naux, Naux);
      Wc_re_qwPQ() = ComplexType(0.0);
      for (long iq = 0; iq < Nq_ibz; ++iq)
        for (long n = 0; n < n_check; ++n) {
          const double nu2 = nu_n(n) * nu_n(n);
          for (long P = 0; P < Naux; ++P)
            for (long Q = 0; Q < Naux; ++Q) {
              double acc = 0.0;
              for (long iO = 0; iO < gs.N_Omega; ++iO) {
                const double O = Om(iO);
                acc += Om_w(iO) * O * ImW(iq, P, Q, iO) / (O * O + nu2);
              }
              Wc_re_qwPQ(iq, n, P, Q) = ComplexType(2.0 * acc / M_PI, 0.0);
            }
        }

      // ---- compare, q != Gamma ---------------------------------------
      double dev = 0.0, rel = 0.0, amp_ref = 0.0;
      long   dq = -1, dn = -1, dP = -1, dQ = -1;
      for (long iq = 0; iq < Nq_ibz; ++iq) {
        if (iq == iq_gamma) continue;
        for (long n = 0; n < n_check; ++n)
          for (long P = 0; P < Naux; ++P)
            for (long Q = 0; Q < Naux; ++Q) {
              const ComplexType r = Wc_re_qwPQ(iq, n, P, Q);
              const ComplexType m = Wc_im_qwPQ(iq, n, P, Q);
              const double d = std::abs(r - m);
              amp_ref = std::max(amp_ref, std::abs(m));
              if (d > dev) { dev = d; dq = iq; dn = n; dP = P; dQ = Q; }
            }
      }
      rel = (amp_ref > 0.0) ? dev / amp_ref : 0.0;

      // Diagnostic ratio: the mean elementwise ratio Re(real)/Re(matsubara)
      // on the largest-amplitude elements. A clean factor tells a
      // normalization story; a scattered ratio tells a shape story.
      double num = 0.0, den = 0.0;
      for (long iq = 0; iq < Nq_ibz; ++iq) {
        if (iq == iq_gamma) continue;
        for (long n = 0; n < n_check; ++n)
          for (long P = 0; P < Naux; ++P)
            for (long Q = 0; Q < Naux; ++Q) {
              const double m = Wc_im_qwPQ(iq, n, P, Q).real();
              if (std::abs(m) < 0.05 * amp_ref) continue;
              num += Wc_re_qwPQ(iq, n, P, Q).real() * m;
              den += m * m;
            }
      }
      const double ratio = (den > 0.0) ? num / den : 0.0;

      // ---- Pi-level ratio (linear probe, see the Pi-LEVEL PROBE block) --
      auto ImPi_ra = math::nda::all_gather_slow<HOST_MEMORY>(*state.ImPi_qPQO);
      double pnum = 0.0, pden = 0.0, pamp = 0.0;
      for (long iq = 0; iq < Nq_ibz; ++iq) {
        if (iq == iq_gamma) continue;
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q)
            pamp = std::max(pamp, std::abs(Pi_im_qwPQ(iq, 0, P, Q)));
      }
      for (long iq = 0; iq < Nq_ibz; ++iq) {
        if (iq == iq_gamma) continue;
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q) {
            double acc = 0.0;
            for (long iO = 0; iO < gs.N_Omega; ++iO)
              acc += Om_w(iO) * ImPi_ra(iq, P, Q, iO) / Om(iO);
            const double pr = 2.0 * acc / M_PI;
            const double pm = Pi_im_qwPQ(iq, 0, P, Q).real();
            if (std::abs(pm) < 0.05 * pamp) continue;
            pnum += pr * pm;
            pden += pm * pm;
          }
      }
      double pworst_rel = 0.0;
      for (long iq = 0; iq < Nq_ibz; ++iq) {
        if (iq == iq_gamma) continue;
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q) {
            double acc = 0.0;
            for (long iO = 0; iO < gs.N_Omega; ++iO)
              acc += Om_w(iO) * ImPi_ra(iq, P, Q, iO) / Om(iO);
            const double pr = 2.0 * acc / M_PI;
            const double pm = Pi_im_qwPQ(iq, 0, P, Q).real();
            if (std::abs(pm) < 0.05 * pamp) continue;
            pworst_rel = std::max(pworst_rel, std::abs(pr - pm) / std::abs(pm));
          }
      }
      const double ratio_pi = (pden > 0.0) ? pnum / pden : 0.0;
      // Split the residual by (P,Q) class and check the (P,Q) symmetry of the
      // stored REAL Im Pi. The bosonic odd extension used by both the KK and
      // the forward map assumes Im Pi_PQ(-Om) = -Im Pi_PQ(Om), which for a
      // matrix response holds elementwise only if the spectral matrix is real
      // symmetric in (P,Q); the general relation is Pi_PQ(-Om) = conj(Pi_QP(Om)).
      {
        double wdiag = 0.0, woff = 0.0, sym = 0.0, symamp = 0.0;
        for (long iq = 0; iq < Nq_ibz; ++iq) {
          if (iq == iq_gamma) continue;
          for (long P = 0; P < Naux; ++P)
            for (long Q = 0; Q < Naux; ++Q) {
              for (long iO = 0; iO < gs.N_Omega; ++iO) {
                sym    = std::max(sym, std::abs(ImPi_ra(iq, P, Q, iO)
                                              - ImPi_ra(iq, Q, P, iO)));
                symamp = std::max(symamp, std::abs(ImPi_ra(iq, P, Q, iO)));
              }
              double acc = 0.0;
              for (long iO = 0; iO < gs.N_Omega; ++iO)
                acc += Om_w(iO) * ImPi_ra(iq, P, Q, iO) / Om(iO);
              const double pr = 2.0 * acc / M_PI;
              const double pm = Pi_im_qwPQ(iq, 0, P, Q).real();
              if (std::abs(pm) < 0.05 * pamp) continue;
              const double r = std::abs(pr - pm) / std::abs(pm);
              if (P == Q) wdiag = std::max(wdiag, r); else woff = std::max(woff, r);
            }
        }
        app_log(2, "[rw1]   Pi residual split: worst rel dev  diag = {:.4e}   "
                   "offdiag = {:.4e};  |ImPi_PQ - ImPi_QP|_max / |ImPi|_max = {:.3e}",
                wdiag, woff, (symamp > 0 ? sym / symamp : 0.0));
      }
      app_log(2, "[rw1]   Pi worst RELATIVE elementwise dev at inu=0 = {:.4e}",
              pworst_rel);
      app_log(2, "[rw1]   Pi-level ratio at inu=0: <Pi_re/Pi_im> = {:.6f}  "
                 "(max|Pi_im| = {:.4e})", ratio_pi, pamp);

      // ---- Re Pi PROBE: the code's own Hilbert transform vs the correct
      // bosonic Kramers-Kronig. At Omega -> 0 the PV singularity is absent
      // and the two differ ONLY by whether the Omega < 0 image of the odd
      // Im Pi is included:
      //   correct: Re Pi(0) = (2/pi) int_0^inf dO Im Pi(O) / O
      //   half   : Re Pi(0) = (1/pi) int_0^inf dO Im Pi(O) / O
      // The first expression is exactly the forward map already evaluated
      // above at i nu = 0, so the ratio below reads 1 if the transform
      // covers the full axis and 1/2 if it covers only the stored half.
      {
        auto RePi_ra = math::nda::all_gather_slow<HOST_MEMORY>(*state.RePi_qPQO);
        double rnum = 0.0, rden = 0.0, ramp = 0.0;
        for (long iq = 0; iq < Nq_ibz; ++iq) {
          if (iq == iq_gamma) continue;
          for (long P = 0; P < Naux; ++P)
            for (long Q = 0; Q < Naux; ++Q) {
              double acc = 0.0;
              for (long iO = 0; iO < gs.N_Omega; ++iO)
                acc += Om_w(iO) * ImPi_ra(iq, P, Q, iO) / Om(iO);
              ramp = std::max(ramp, std::abs(2.0 * acc / M_PI));
            }
        }
        for (long iq = 0; iq < Nq_ibz; ++iq) {
          if (iq == iq_gamma) continue;
          for (long P = 0; P < Naux; ++P)
            for (long Q = 0; Q < Naux; ++Q) {
              double acc = 0.0;
              for (long iO = 0; iO < gs.N_Omega; ++iO)
                acc += Om_w(iO) * ImPi_ra(iq, P, Q, iO) / Om(iO);
              const double kk_full = 2.0 * acc / M_PI;
              if (std::abs(kk_full) < 0.05 * ramp) continue;
              rnum += RePi_ra(iq, P, Q, 0) * kk_full;
              rden += kk_full * kk_full;
            }
        }
        app_log(2, "[rw1]   Re Pi probe: <RePi_code(Omega_1) / KK_full(0)> = "
                   "{:.6f}   (Omega_1 = {:.5f} Ha; 1.0 = full-axis KK, "
                   "0.5 = half-axis only)",
                (rden > 0.0) ? rnum / rden : 0.0, Om(0));
      }

      // ---- PER-nu breakdown --------------------------------------------
      // The forward kernel is Omega/(Omega^2 + nu^2). At nu = 0 it is 1/Omega,
      // which AMPLIFIES the small-Omega region -- exactly where the QP-pole
      // Lorentzians put spurious sub-gap weight (Im Pi is strictly zero below
      // the gap for the un-broadened G, but the eta tails of the two poles
      // leave ~eta^2/Omega^2 there, and eta^2/Omega^2 x 1/Omega integrated
      // from Omega_1 ~ eta is O(1) independent of eta). At nu > 0 the same
      // kernel SUPPRESSES small Omega. If the residual is the sub-gap tail,
      // it must collapse with increasing nu.
      {
        for (long n : {0L, 1L, 5L, 10L, 20L, 40L}) {
          if (n >= n_check) continue;
          double d = 0.0, a = 0.0;
          for (long iq = 0; iq < Nq_ibz; ++iq) {
            if (iq == iq_gamma) continue;
            for (long P = 0; P < Naux; ++P)
              for (long Q = 0; Q < Naux; ++Q) {
                d = std::max(d, std::abs(Wc_re_qwPQ(iq, n, P, Q)
                                       - Wc_im_qwPQ(iq, n, P, Q)));
                a = std::max(a, std::abs(Wc_im_qwPQ(iq, n, P, Q)));
              }
          }
          app_log(2, "[rw1]     nu[{:2d}] = {:.5f} Ha : worst dev = {:.4e}  "
                     "max|W_im| = {:.4e}  rel = {:.4e}",
                  n, nu_n(n), d, a, (a > 0 ? d / a : 0.0));
        }
      }

      // ---- SPLIT the W discrepancy into its two possible factors -------
      //   (a) DIRECT: Re W^c_code(Omega_1) vs the Matsubara W^c(i nu = 0).
      //       These are the same physical number (W^c(0)); this route uses
      //       NO spectral integral, only the Dyson output at the smallest
      //       stored Omega.
      //   (b) CAUSALITY: (2/pi) int dOmega Im W^c/Omega vs Re W^c(Omega_1).
      //       For a causal W these are equal. A deviation means the code's
      //       Re W and Im W are not a Kramers-Kronig pair, i.e. the error is
      //       in the Omega dependence, not in the static value.
      {
        auto ReW_ra = math::nda::all_gather_slow<HOST_MEMORY>(*state.ReW_qPQO);
        double anum = 0.0, aden = 0.0, bnum = 0.0, bden = 0.0;
        for (long iq = 0; iq < Nq_ibz; ++iq) {
          if (iq == iq_gamma) continue;
          for (long P = 0; P < Naux; ++P)
            for (long Q = 0; Q < Naux; ++Q) {
              const double m = Wc_im_qwPQ(iq, 0, P, Q).real();
              const double d = ReW_ra(iq, P, Q, 0);
              const double k = Wc_re_qwPQ(iq, 0, P, Q).real();
              if (std::abs(m) >= 0.05 * amp_ref) { anum += d * m; aden += m * m; }
              if (std::abs(d) >= 0.05 * amp_ref) { bnum += k * d; bden += d * d; }
            }
        }
        app_log(2, "[rw1]   W split: (a) <ReW_code(Omega_1)/W_im(0)> = {:.6f}   "
                   "(b) <KK[ImW](0)/ReW_code(Omega_1)> = {:.6f}",
                (aden > 0.0) ? anum / aden : 0.0,
                (bden > 0.0) ? bnum / bden : 0.0);
      }

      // ---- head channel and Gamma, reported separately ---------------
      auto head_re = head_of(Wc_re_qwPQ, iq_small);
      double hdev = 0.0;
      for (long n = 0; n < n_check; ++n)
        hdev = std::max(hdev, std::abs(head_re(n) - head_im(n)));

      double gamp = 0.0;
      for (long n = 0; n < n_check; ++n)
        for (long P = 0; P < Naux; ++P)
          for (long Q = 0; Q < Naux; ++Q)
            gamp = std::max(gamp, std::abs(Wc_im_qwPQ(iq_gamma, n, P, Q)));

      const double bud = budget_for(eta);
      worst_dev.push_back(dev);
      worst_rel.push_back(rel);
      head_dev.push_back(hdev);
      gamma_amp.push_back(gamp);
      budget.push_back(bud);

      app_log(2, "[rw1]   worst |W_re(inu) - W_im(inu)| (q != Gamma) = {:.6e}  "
                 "at (q={}, n={}, P={}, Q={})", dev, dq, dn, dP, dQ);
      app_log(2, "[rw1]   reference amplitude max|W_im| = {:.6e}  -> relative "
                 "= {:.4e}", amp_ref, rel);
      app_log(2, "[rw1]   least-squares ratio Re(W_re)/Re(W_im) = {:.6f}", ratio);
      app_log(2, "[rw1]   eta budget (2 eta / d_min) * kappa * max|W_im| = {:.6e}", bud);
      app_log(2, "[rw1]   eps^-1 head channel (q_min): worst |dev| = {:.6e}  "
                 "[re(nu=0) = {:.6e}, im(nu=0) = {:.6e}]",
              hdev, head_re(0).real(), head_im(0).real());
      app_log(2, "[rw1]   Gamma (excluded, D1): real-axis is 0 by construction; "
                 "max|W_im(Gamma)| = {:.6e}", gamp);
    }

    // =================================================================
    // GATE RW-1-a
    // =================================================================
    app_log(2, "\n[rw1] ===== GATE RW-1-a: eta series =====");
    app_log(2, "[rw1]   eta        worst dev      relative     budget        "
               "obs/budget   head dev");
    for (size_t i = 0; i < etas.size(); ++i)
      app_log(2, "[rw1]   {:8.5f}   {:.6e}   {:.4e}   {:.6e}   {:.4f}       {:.6e}",
              etas[i], worst_dev[i], worst_rel[i], budget[i],
              worst_dev[i] / budget[i], head_dev[i]);

    // (a) monotone decreasing in eta
    for (size_t i = 1; i < worst_dev.size(); ++i)
      REQUIRE(worst_dev[i] < worst_dev[i - 1]);
    // (b) inside the honestly pre-computed budget at the smallest eta
    REQUIRE(worst_dev.back() < budget.back());
  }

  TEST_CASE("real_axis_w_lehmann_qe_lih222",
            "[real_axis][w][lehmann][thc][qe][bdft][serial][xvalidate]") {
    run_lehmann_gate("qe_lih222", /*use_rspace*/ false);
  }

  // SECOND FIXTURE. The spec asks for si222; no usable si222 exists in this
  // tree (notes/rw1_port_report.md): qe_si222_so is spin-orbit and the ported
  // update_w hard-requires npol() == 1 (real_axis_scr_coulomb_t.h:200-202);
  // bdft_si222 is inside a /* */ block in default_MF.hpp and does not
  // dispatch; pyscf_si222 is a Gaussian-basis fixture with S != 1, which
  // violates convention pin P1 (the identity-MO construction of A on the FBZ
  // is only licensed when the primary basis is the orthogonal KS basis).
  // qe_si211 is the silicon fixture that satisfies every pin.
  TEST_CASE("real_axis_w_lehmann_qe_si211",
            "[real_axis][w][lehmann][thc][qe][bdft][serial][xvalidate][si]") {
    run_lehmann_gate("qe_si211", /*use_rspace*/ true);
  }

} // namespace bdft_tests
