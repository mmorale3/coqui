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
 * C_f(k_f)^*_{a i} dS(tau, k_f)_{ab} C_f(k_f)_{b j}, Hermitized when pol_vertex_sigma_pair_herm is on (its anti-Hermitian
 * residual measured first and returned, audit C5). The coarse run writes the Wannier-frame object with its k list, R grid,
 * tau nodes, lattice and the full Sigma-vertex configuration (dump_sigma_pair_wannier; the fine run aborts on any
 * configuration mismatch, audit A12); the fine run consumes it instead of solving
 * (interpolate_sigma_pair). Requirements: a NOSYM coarse mesh (the full BZ is what the R transform needs), the same
 * imaginary-axis grid (beta and the tau nodes are checked), a fine projector on the fine mesh whose window is the
 * fine pair vertex's window (nc_f = |W_rng|), unitary projectors (the round trip downfold -> upfold is the identity
 * on the window only then; the measured unitarity defect is reported), and -- not checkable here -- the SAME Wannier
 * functions on both meshes: the Wannier-frame object is covariant under a k-INDEPENDENT unitary of the orbitals (the
 * upfold undoes it) but not under a k-dependent gauge difference between two separately generated projector files
 * (notes/CLAUDE.md section 8, demand D2: one U per run). Coarse and fine projectors must come from one Wannierization
 * (the fine one by interpolation of the coarse MLWFs, or both from a common set with the same projections and gauge).
 */

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
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

  /**
   * audit A12 (2026-10-01): EVERY option that shapes the stored dSigma. The dump used to carry col / outer only and the
   * fine run compared nothing, so a dump produced with another column, junction, scale, rung sign, Hermitization, Sigma
   * path or band window was consumed as if it were this run's. The coarse run writes the full set (cfg_version 1); the
   * fine run compares it with its own settings and aborts on any mismatch, and a dump without cfg_version (written before
   * this change) is refused with "regenerate the dump". Fill it with cfg_of(sigma_pair_opts, sigma_pair_dynamic()).
   */
  struct interp_cfg {
    std::string col, outer, side;
    double scale = 1.0, sign_ks = -1.0;
    bool hermitize = true;
    bool dyn = false;                 // the DYNAMIC-rung Sigma path (eval_sigma_pair_dyn) vs the static one (eval_sigma_pair)
    // provenance of the dynamic path's sampled mode: written and logged, NOT compared (the fine run does not solve, so it
    // has no reason to carry the coarse run's node list / fit file)
    long dyn_nodes_n = 0, dyn_fit_rank = 0, dyn_auto_nodes = 0;
    std::string dyn_refit;
  };
  /** the cfg of a vertex_t::sigma_pair_opts (templated: this header does not include vertex_t.h) */
  template<typename opts_t>
  inline interp_cfg cfg_of(opts_t const &o, bool dyn) {
    interp_cfg c;
    c.col = o.col; c.outer = o.outer; c.side = o.side;
    c.scale = o.scale; c.sign_ks = o.sign_ks; c.hermitize = o.hermitize; c.dyn = dyn;
    c.dyn_nodes_n = long(o.dyn_nodes.size()); c.dyn_fit_rank = o.dyn_fit_rank; c.dyn_auto_nodes = o.dyn_auto_nodes;
    c.dyn_refit = o.dyn_refit;
    return c;
  }
  inline constexpr long interp_cfg_version = 1;
  /** audit C5: above this the downfold -> upfold round trip is not the identity on the window (WARNING) */
  inline constexpr double proj_unitarity_warn_tol = 1e-8;
  /** audit C5: what the fine consumer measured -- herm_resid is the anti-Hermitian residual of the upfolded dSigma BEFORE
   *  any symmetrization (max |Y - Y^dag| / max |Y|, the eval_sigma_pair dsig_herm convention) */
  struct interp_meter {
    double herm_resid = 0.0, dunit_fine = 0.0, dunit_coarse = 0.0;
  };

  /** the coarse run: the Wannier-frame dSigma on the full mesh + everything the fine consumer needs, into the h5 group */
  template<typename comm_t>
  inline void dump_sigma_pair_wannier(mf::MF &mf, projector_t const &proj, nda::array<cplx, 5> const &dSig,   // (nt, ns, nk, nc, nc)
                                      long window_first, nda::array<double, 1> const &tau, double beta, comm_t &comm,
                                      std::string const &fn, interp_cfg const &cfg) {
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
    // audit A12: the coarse lattice, so the consumer converts the coarse k with the lattice they were generated on
    nda::array<double, 2> lat(3, 3);
    {
      auto L = mf.lattv();
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) lat(i, j) = L(i, j);
    }
    if (comm.root()) {
      h5::file f(fn, 'w');
      h5::group g(f);
      h5::h5_write(g, "col", cfg.col); h5::h5_write(g, "outer", cfg.outer);
      // audit A12: the full configuration (compared by interpolate_sigma_pair)
      h5::h5_write(g, "cfg_version", interp_cfg_version);
      h5::h5_write(g, "side", cfg.side);
      h5::h5_write(g, "scale", cfg.scale);
      h5::h5_write(g, "sign_ks", cfg.sign_ks);
      h5::h5_write(g, "hermitize", long(cfg.hermitize ? 1 : 0));
      h5::h5_write(g, "dyn", long(cfg.dyn ? 1 : 0));
      h5::h5_write(g, "dyn_nodes_n", cfg.dyn_nodes_n);
      h5::h5_write(g, "dyn_fit_rank", cfg.dyn_fit_rank);
      h5::h5_write(g, "dyn_auto_nodes", cfg.dyn_auto_nodes);
      h5::h5_write(g, "dyn_refit", cfg.dyn_refit);
      nda::h5_write(g, "lattv", lat);
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
               "projector unitarity defect on the window {:.2e}; configuration: {} path, col {}, outer {}, junction {}, scale {}, "
               "sign_ks {}, Hermitized {}", nt, ns, nk, M, M, fn, Ridx.shape(0), dunit, cfg.dyn ? "DYNAMIC" : "static", cfg.col,
            cfg.outer, cfg.side, cfg.scale, cfg.sign_ks, cfg.hermitize);
    if (dunit > proj_unitarity_warn_tol)
      app_log(1, "  [WARNING] P16 dump: the coarse projector is not unitary on the window (defect {:.2e} > {:.0e}): the Wannier-frame "
                 "dSigma is not an exact image of the band-frame one, and the fine run's upfold will not reproduce it.", dunit,
              proj_unitarity_warn_tol);
  }


  /** the fine run: dSigma(tau, k_ibz) on the fine window from the coarse dump (replaces the solve). cfg = THIS run's Sigma-vertex
   *  configuration (compared with the dump's; audit A12); met, when given, receives the measured residuals (audit C5). */
  template<typename comm_t>
  inline nda::array<cplx, 5> interpolate_sigma_pair(mf::MF &mf, projector_t const &proj, std::string const &file,
                                                     nda::array<double, 1> const &tau, double beta, long window_first, long nc, comm_t &comm,
                                                     interp_cfg const &cfg, interp_meter *met = nullptr) {
    decltype(nda::range::all) all;
    nda::array<cplx, 5> dSw;
    nda::array<double, 2> kc;
    nda::array<long, 2> Ridx;
    nda::array<long, 1> Rw;
    nda::array<double, 1> tau_c;
    double beta_c = 0.0, dunit_c = 0.0;
    long M_c = 0, w0_c = -1, nc_c = -1;
    interp_cfg cc;                    // audit A12: the coarse run's configuration
    nda::array<double, 2> lat_c;
    {
      h5::file f(file, 'r');
      h5::group g(f);
      utils::check(g.has_dataset("dSigma_wan_tskab"), "interpolate_sigma_pair: {} carries no Wannier-frame dSigma (the coarse run needs pol_vertex_sigma_interp_dump).", file);
      utils::check(g.has_dataset("cfg_version"),
                   "interpolate_sigma_pair: {} was written before the configuration record (audit A12, 2026-10-01): it stores col / outer "
                   "only, so it cannot be checked against this run (junction, scale, rung sign, Hermitization, Sigma path, window, "
                   "lattice). Regenerate the dump with the coarse run (pol_vertex_sigma_interp_dump) on this code.", file);
      long ver = 0;
      h5::h5_read(g, "cfg_version", ver);
      utils::check(ver == interp_cfg_version, "interpolate_sigma_pair: {} has configuration record version {}, this code reads {}. "
                   "Regenerate the dump.", file, ver, interp_cfg_version);
      nda::h5_read(g, "dSigma_wan_tskab", dSw); nda::h5_read(g, "kpts_cart", kc); nda::h5_read(g, "Rpts_idx", Ridx); nda::h5_read(g, "Rpts_weights", Rw);
      nda::h5_read(g, "tau", tau_c); h5::h5_read(g, "beta", beta_c); h5::h5_read(g, "nwan", M_c);
      h5::h5_read(g, "window_first", w0_c); h5::h5_read(g, "window_size", nc_c); h5::h5_read(g, "proj_unitarity_defect", dunit_c);
      long herm_c = 1, dyn_c = 0;
      h5::h5_read(g, "col", cc.col); h5::h5_read(g, "outer", cc.outer); h5::h5_read(g, "side", cc.side);
      h5::h5_read(g, "scale", cc.scale); h5::h5_read(g, "sign_ks", cc.sign_ks);
      h5::h5_read(g, "hermitize", herm_c); h5::h5_read(g, "dyn", dyn_c);
      cc.hermitize = (herm_c != 0); cc.dyn = (dyn_c != 0);
      h5::h5_read(g, "dyn_nodes_n", cc.dyn_nodes_n); h5::h5_read(g, "dyn_fit_rank", cc.dyn_fit_rank);
      h5::h5_read(g, "dyn_auto_nodes", cc.dyn_auto_nodes); h5::h5_read(g, "dyn_refit", cc.dyn_refit);
      nda::h5_read(g, "lattv", lat_c);
    }
    // ---- audit A12: the stored dSigma must be THIS run's object -----------------------------------------------------------
    {
      auto same_str = [&](std::string const &a, std::string const &b, const char *key) {
        utils::check(a == b, "interpolate_sigma_pair: {} was written with {} = \"{}\", this run has \"{}\" -- the stored dSigma is a "
                     "different object. Regenerate the dump with this run's settings, or set the coarse run's value here.",
                     file, key, a, b);
      };
      auto same_dbl = [&](double a, double b, const char *key) {
        utils::check(std::abs(a - b) <= 1e-12 * std::max(1.0, std::abs(a)),
                     "interpolate_sigma_pair: {} was written with {} = {}, this run has {} -- the stored dSigma is a different "
                     "object. Regenerate the dump with this run's settings, or set the coarse run's value here.", file, key, a, b);
      };
      auto same_bool = [&](bool a, bool b, const char *key) {
        utils::check(a == b, "interpolate_sigma_pair: {} was written with {} = {}, this run has {} -- the stored dSigma is a different "
                     "object. Regenerate the dump with this run's settings, or set the coarse run's value here.", file, key, a, b);
      };
      same_bool(cc.dyn, cfg.dyn, "the dynamic-rung Sigma path (pol_vertex_sigma_pair_col in static_dyn | dyn1_bare | dyn1 | dyn)");
      same_str(cc.col, cfg.col, "pol_vertex_sigma_pair_col");
      same_str(cc.outer, cfg.outer, "pol_vertex_sigma_pair_outer");
      same_str(cc.side, cfg.side, "pol_vertex_sigma_pair_side");
      same_dbl(cc.scale, cfg.scale, "pol_vertex_sigma_pair_scale");
      same_dbl(cc.sign_ks, cfg.sign_ks, "the rung sign sign_ks (pol_vertex_dyn_sign)");
      same_bool(cc.hermitize, cfg.hermitize, "pol_vertex_sigma_pair_herm");
      utils::check(w0_c == window_first and nc_c == nc,
                   "interpolate_sigma_pair: {} was computed on the band window [{}, {}), this run's pair vertex window is [{}, {}) -- "
                   "the stored dSigma is a different object. Regenerate the dump, or use the same window.", file, w0_c, w0_c + nc_c,
                   window_first, window_first + nc);
      utils::check(lat_c.shape(0) == 3 and lat_c.shape(1) == 3, "interpolate_sigma_pair: {}: malformed lattv record.", file);
      auto L = mf.lattv();
      double dl = 0.0, sl = 0.0;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) { dl = std::max(dl, std::abs(lat_c(i, j) - L(i, j))); sl = std::max(sl, std::abs(L(i, j))); }
      utils::check(dl <= 1e-8 * std::max(sl, 1.0),
                   "interpolate_sigma_pair: {} was written on a lattice that differs from this run's by {:.3e} (max |lattv_c - lattv_f|): "
                   "the Wigner-Seitz R vectors of the dump (lattice indices) would denote different vectors here. Coarse and fine "
                   "meshes must be the same crystal in the same units.", file, dl);
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
    // audit E: the fine IBZ points are taken as the first nkf points of the full list (kf below, and the caller's dSigma
    // k axis) -- assert that they ARE the identity-mapped IBZ representatives
    if (mf.nkpts() != mf.nkpts_ibz()) {
      auto k2i = mf.kp_to_ibz();
      auto ktr = mf.kp_trev();
      for (long ik = 0; ik < nkf; ++ik)
        utils::check(long(k2i(ik)) == ik and not bool(ktr(ik)),
                     "interpolate_sigma_pair: the fine full k list does not start with the IBZ points (k = {}: kp_to_ibz {}, trev {}).",
                     ik, long(k2i(ik)), int(bool(ktr(ik))));
    }
    nda::array<cplx, 2> f_Rk(nR, nkc), f_kR(nkf, nR);
    // audit A12: the coarse k are converted with the COARSE lattice (the one they were generated on; checked equal to this
    // run's above, so the R vectors mean the same thing on both sides)
    nda::stack_array<double, 3, 3> lat_cs;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) lat_cs(i, j) = lat_c(i, j);
    utils::k_to_R_coefficients(Ridx, kc, lat_cs, f_Rk);
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
    // upfold onto the fine window bands: dSigma_ij = sum_ab conj(C_ai) dS_ab C_bj, then (cfg.hermitize) Hermitize.
    // audit C5: the anti-Hermitian residual is MEASURED before the symmetrization (it used to be hard-wired and reported as 0
    // by the caller), and the projection honours cfg.hermitize (= the coarse run's, checked above) instead of always applying.
    nda::array<cplx, 5> out(nt, ns, nkf, nc, nc);
    auto C = proj.C_skIai();
    nda::array<cplx, 2> T(M, nc), Y(nc, nc);
    double dunit = 0.0, ymax = 0.0, yherm = 0.0;
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
            for (long j = 0; j < nc; ++j) {
              ymax = std::max(ymax, std::abs(Y(i, j)));
              yherm = std::max(yherm, std::abs(Y(i, j) - std::conj(Y(j, i))));
              out(it, is, ik, i, j) = cfg.hermitize ? 0.5 * (Y(i, j) + std::conj(Y(j, i))) : Y(i, j);
            }
        }
      }
    const double herm_resid = (ymax > 0.0) ? yherm / ymax : 0.0;
    app_log(1, "  [LFF-Sigma pair] P16: dSigma interpolated from {} ({} coarse k, {} R vectors, {} Wannier orbitals) onto {} fine IBZ k of the window "
               "[{}, {}); projector unitarity defect coarse {:.2e} fine {:.2e}; anti-Hermitian residual before symmetrization {:.3e} ({}); "
               "configuration matches the dump ({} path, col {}, outer {}, junction {}, scale {}, sign_ks {}{})", file, nkc, nR, M, nkf,
            window_first, window_first + nc, dunit_c, dunit, herm_resid, cfg.hermitize ? "Hermitized" : "NOT Hermitized",
            cfg.dyn ? "DYNAMIC" : "static", cfg.col, cfg.outer, cfg.side, cfg.scale, cfg.sign_ks,
            cfg.dyn ? (", coarse sampled-mode provenance: " + std::to_string(cc.dyn_nodes_n) + " explicit nodes, fit rank " +
                       std::to_string(cc.dyn_fit_rank) + ", auto nodes " + std::to_string(cc.dyn_auto_nodes) + ", refit " + cc.dyn_refit)
                    : std::string());
    for (auto [d, who] : {std::pair<double, const char *>{dunit_c, "coarse"}, std::pair<double, const char *>{dunit, "fine"}})
      if (d > proj_unitarity_warn_tol)
        app_log(1, "  [WARNING] P16 interpolation: the {} projector is not unitary on the window (defect {:.2e} > {:.0e}): the "
                   "downfold -> upfold round trip is not the identity there, so the interpolated dSigma carries a projector error of "
                   "that order (relative).", who, d, proj_unitarity_warn_tol);
    if (met) { met->herm_resid = herm_resid; met->dunit_fine = dunit; met->dunit_coarse = dunit_c; }
    (void)comm;
    return out;
  }


} // vertex_sigma_interp
} // solvers
} // methods

#endif // COQUI_VERTEX_SIGMA_INTERP_HPP
