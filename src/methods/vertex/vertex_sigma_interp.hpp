#ifndef COQUI_VERTEX_SIGMA_INTERP_HPP
#define COQUI_VERTEX_SIGMA_INTERP_HPP

/**
 * P16 (notes/vertex_perf_plan.md, 2026-09-21): the coarse -> fine interpolation of the pair-resolved Sigma vertex
 * dSigma(tau, k) (the C-window block of the self-energy correction, vertex_sigma_pair.icc / vertex_sigma_dyn.icc).
 *
 * The band basis at k is gauge-dependent (phases, degenerate mixing), so the smooth object is the WANNIER-frame one,
 *     dS(tau, k)_ab = sum_ij C(k)_{a i} dSigma(tau, k)_{ij} C(k)^*_{b j}      (projector_t::downfold_k: C O C^dag),
 * on the coarse FULL mesh; its lattice transform on the Wigner-Seitz R grid (the Wannier-interpolation route of
 * pproc_t::interpolate_qp_bands_on_mesh, the imaginary part kept: exact on the coarse mesh)
 *     dS(tau, R) = 1/N_c sum_k e^{-i k R} dS(tau, k),   dS(tau, k_f) = sum_R w_R e^{i k_f R} dS(tau, R),
 * and the fine mesh's own projector puts it back on the fine window bands, dSigma_f(tau, k_f)_{ij} = sum_ab
 * C_f(k_f)^*_{a i} dS(tau, k_f)_{ab} C_f(k_f)_{b j}, Hermitized. The coarse run writes the Wannier-frame object with
 * its k list, R grid and tau nodes (dump_sigma_pair_wannier); the fine run consumes it instead of solving
 * (interpolate_sigma_pair). Requirements: a NOSYM coarse mesh (the full BZ is what the R transform needs), the same
 * imaginary-axis grid (beta and the tau nodes are checked), a fine projector on the fine mesh whose window is the
 * fine pair vertex's window (nc_f = |W_rng|), unitary projectors (the round trip downfold -> upfold is the identity
 * on the window only then; the measured unitarity defect is reported), and -- not checkable here -- the SAME Wannier
 * functions on both meshes: the Wannier-frame object is covariant under a k-INDEPENDENT unitary of the orbitals (the
 * upfold undoes it) but not under a k-dependent gauge difference between two separately generated projector files
 * (notes/CLAUDE.md section 8, demand D2: one U per run). Coarse and fine projectors must come from one Wannierization
 * (the fine one by interpolation of the coarse MLWFs, or both from a common set with the same projections and gauge).
 */

#include <string>
#include <vector>
#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "h5/h5.hpp"
#include "utilities/check.hpp"
#include "utilities/kpoint_utils.hpp"
#include "utilities/interpolation_utils.hpp"
#include "IO/app_loggers.h"
#include "mean_field/MF.hpp"
#include "methods/embedding/projector_t.h"

namespace methods {
namespace solvers {
namespace vertex_sigma_interp {

  using cplx = ComplexType;

  /** the coarse run: the Wannier-frame dSigma on the full mesh + everything the fine consumer needs, into the h5 group */
  template<typename comm_t>
  inline void dump_sigma_pair_wannier(mf::MF &mf, projector_t const &proj, nda::array<cplx, 5> const &dSig,   // (nt, ns, nk, nc, nc)
                                      long window_first, nda::array<double, 1> const &tau, double beta, comm_t &comm,
                                      std::string const &fn, std::string const &col, std::string const &outer) {
    decltype(nda::range::all) all;
    const long nt = dSig.shape(0), ns = dSig.shape(1), nk = dSig.shape(2), nc = dSig.shape(3), nbnd = mf.nbnd();
    utils::check(mf.nkpts() == mf.nkpts_ibz() and nk == mf.nkpts(),
                 "dump_sigma_pair_wannier: the coarse run must be on a NOSYM mesh (dSigma on {} k of {}, nkpts_ibz {}).", nk, mf.nkpts(), mf.nkpts_ibz());
    utils::check(proj.nImps() == 1, "dump_sigma_pair_wannier: one impurity block expected.");
    auto const &W = proj.W_rng()[0];
    utils::check(W.first() == window_first and long(W.size()) == nc,
                 "dump_sigma_pair_wannier: the projector's window [{}, {}) is not the pair vertex's [{}, {}).", W.first(), W.last(), window_first, window_first + nc);
    // embed the window block into the band space (the downfold reads the window slice only)
    nda::array<cplx, 5> O(nt, ns, nk, nbnd, nbnd);
    O() = cplx(0.0);
    for (long it = 0; it < nt; ++it)
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nk; ++ik) O(it, is, ik, W, W) = dSig(it, is, ik, all, all);
    auto dS = proj.downfold_k(O, comm);                    // (nt, ns, nk, 1, M, M)
    const long M = dS.shape(4);
    nda::array<cplx, 5> dSw(nt, ns, nk, M, M);
    for (long it = 0; it < nt; ++it)
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nk; ++ik) dSw(it, is, ik, all, all) = dS(it, is, ik, 0, all, all);
    // the R grid of the coarse mesh (Wigner-Seitz copies with 1/degeneracy weights)
    auto [Rw, Ridx] = utils::WS_rgrid(mf.lattv(), mf.kp_grid());
    nda::array<double, 2> kc(mf.kpts());                   // Cartesian, the lattice's own units
    // the projector's unitarity on the window (the round trip is exact only then)
    double dunit = 0.0;
    {
      auto C = proj.C_skIai();
      nda::array<cplx, 2> CC(M, M);
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nk; ++ik) {
          nda::blas::gemm(C(is, ik, 0, all, all), nda::dagger(C(is, ik, 0, all, all)), CC);
          for (long a = 0; a < M; ++a) for (long b = 0; b < M; ++b) dunit = std::max(dunit, std::abs(CC(a, b) - ((a == b) ? cplx(1.0) : cplx(0.0))));
        }
    }
    if (comm.root()) {
      h5::file f(fn, 'w');
      h5::group g(f);
      h5::h5_write(g, "col", col); h5::h5_write(g, "outer", outer);
      nda::h5_write(g, "dSigma_wan_tskab", dSw);
      nda::h5_write(g, "kpts_cart", kc);
      nda::h5_write(g, "Rpts_idx", Ridx);
      nda::h5_write(g, "Rpts_weights", Rw);
      nda::h5_write(g, "tau", tau);
      h5::h5_write(g, "beta", beta);
      h5::h5_write(g, "nwan", M);
      h5::h5_write(g, "window_first", window_first);
      h5::h5_write(g, "window_size", nc);
      h5::h5_write(g, "proj_unitarity_defect", dunit);
    }
    app_log(1, "  [LFF-Sigma pair] P16: the Wannier-frame dSigma ({} tau x {} spins x {} k x {} x {}) written to {} with its R grid ({} vectors); "
               "projector unitarity defect on the window {:.2e}", nt, ns, nk, M, M, fn, Ridx.shape(0), dunit);
  }

  /** the fine run: dSigma(tau, k_ibz) on the fine window from the coarse dump (replaces the solve) */
  template<typename comm_t>
  inline nda::array<cplx, 5> interpolate_sigma_pair(mf::MF &mf, projector_t const &proj, std::string const &file,
                                                     nda::array<double, 1> const &tau, double beta, long window_first, long nc, comm_t &comm) {
    decltype(nda::range::all) all;
    nda::array<cplx, 5> dSw;
    nda::array<double, 2> kc;
    nda::array<long, 2> Ridx;
    nda::array<long, 1> Rw;
    nda::array<double, 1> tau_c;
    double beta_c = 0.0, dunit_c = 0.0;
    long M_c = 0, w0_c = -1, nc_c = -1;
    {
      h5::file f(file, 'r');
      h5::group g(f);
      utils::check(g.has_dataset("dSigma_wan_tskab"), "interpolate_sigma_pair: {} carries no Wannier-frame dSigma (the coarse run needs pol_vertex_sigma_interp_dump).", file);
      nda::h5_read(g, "dSigma_wan_tskab", dSw); nda::h5_read(g, "kpts_cart", kc); nda::h5_read(g, "Rpts_idx", Ridx); nda::h5_read(g, "Rpts_weights", Rw);
      nda::h5_read(g, "tau", tau_c); h5::h5_read(g, "beta", beta_c); h5::h5_read(g, "nwan", M_c);
      h5::h5_read(g, "window_first", w0_c); h5::h5_read(g, "window_size", nc_c); h5::h5_read(g, "proj_unitarity_defect", dunit_c);
    }
    const long nt = dSw.shape(0), ns = dSw.shape(1), nkc = dSw.shape(2), M = dSw.shape(3), nR = Ridx.shape(0);
    utils::check(std::abs(beta_c - beta) < 1e-10 * std::max(1.0, beta) and tau_c.shape(0) == tau.shape(0),
                 "interpolate_sigma_pair: the coarse dump's imaginary-axis grid (beta {}, {} tau nodes) is not this run's (beta {}, {} nodes).",
                 beta_c, tau_c.shape(0), beta, tau.shape(0));
    for (long it = 0; it < nt; ++it)
      utils::check(std::abs(tau_c(it) - tau(it)) < 1e-10 * beta, "interpolate_sigma_pair: tau node {} differs ({} vs {}).", it, tau_c(it), tau(it));
    utils::check(proj.nImps() == 1 and proj.nImpOrbs() == M, "interpolate_sigma_pair: the fine projector has {} Wannier orbitals, the dump {}.", proj.nImpOrbs(), M);
    auto const &W = proj.W_rng()[0];
    utils::check(W.first() == window_first and long(W.size()) == nc,
                 "interpolate_sigma_pair: the fine projector's window [{}, {}) is not the pair vertex's [{}, {}).", W.first(), W.last(), window_first, window_first + nc);
    utils::check(ns == mf.nspin(), "interpolate_sigma_pair: spin count mismatch.");
    // k -> R on the coarse mesh, R -> k on the fine IBZ points
    const long nkf = mf.nkpts_ibz();
    nda::array<cplx, 2> f_Rk(nR, nkc), f_kR(nkf, nR);
    utils::k_to_R_coefficients(Ridx, kc, mf.lattv(), f_Rk);
    nda::array<double, 2> kf(nkf, 3);
    for (long ik = 0; ik < nkf; ++ik) kf(ik, all) = mf.kpts()(ik, all);
    utils::R_to_k_coefficients(Ridx, Rw, kf, mf.lattv(), f_kR);
    const long cols = ns * M * M;
    nda::array<cplx, 2> A(nkc, cols), B(nR, cols), Cf(nkf, cols);
    nda::array<cplx, 5> dSf(nt, ns, nkf, M, M);
    for (long it = 0; it < nt; ++it) {
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nkc; ++ik)
          for (long a = 0; a < M; ++a)
            for (long b = 0; b < M; ++b) A(ik, (is * M + a) * M + b) = dSw(it, is, ik, a, b);
      nda::blas::gemm(f_Rk, A, B);
      nda::blas::gemm(f_kR, B, Cf);
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nkf; ++ik)
          for (long a = 0; a < M; ++a)
            for (long b = 0; b < M; ++b) dSf(it, is, ik, a, b) = Cf(ik, (is * M + a) * M + b);
    }
    // upfold onto the fine window bands: dSigma_ij = sum_ab conj(C_ai) dS_ab C_bj, then Hermitize
    nda::array<cplx, 5> out(nt, ns, nkf, nc, nc);
    auto C = proj.C_skIai();
    nda::array<cplx, 2> T(M, nc), Y(nc, nc);
    double dunit = 0.0;
    for (long is = 0; is < ns; ++is)
      for (long ik = 0; ik < nkf; ++ik) {
        auto Ck = C(is, ik, 0, all, all);                    // (M, nc)
        {
          nda::array<cplx, 2> CC(M, M);
          nda::blas::gemm(Ck, nda::dagger(Ck), CC);
          for (long a = 0; a < M; ++a) for (long b = 0; b < M; ++b) dunit = std::max(dunit, std::abs(CC(a, b) - ((a == b) ? cplx(1.0) : cplx(0.0))));
        }
        for (long it = 0; it < nt; ++it) {
          nda::blas::gemm(dSf(it, is, ik, all, all), Ck, T);              // (M, M)(M, nc)
          nda::blas::gemm(nda::dagger(Ck), T, Y);                         // (nc, M)(M, nc)
          for (long i = 0; i < nc; ++i)
            for (long j = 0; j < nc; ++j) out(it, is, ik, i, j) = 0.5 * (Y(i, j) + std::conj(Y(j, i)));
        }
      }
    app_log(1, "  [LFF-Sigma pair] P16: dSigma interpolated from {} ({} coarse k, {} R vectors, {} Wannier orbitals) onto {} fine IBZ k of the window "
               "[{}, {}); projector unitarity defect coarse {:.2e} fine {:.2e}", file, nkc, nR, M, nkf, window_first, window_first + nc, dunit_c, dunit);
    (void)comm;
    return out;
  }

} // vertex_sigma_interp
} // solvers
} // methods

#endif // COQUI_VERTEX_SIGMA_INTERP_HPP
