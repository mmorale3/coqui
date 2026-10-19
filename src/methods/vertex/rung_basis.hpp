#ifndef COQUI_VERTEX_RUNG_BASIS_HPP
#define COQUI_VERTEX_RUNG_BASIS_HPP

/**
 * THE TIME FUNCTIONS OF THE FREQUENCY-FACTORIZED DYNAMIC RUNG (pol_vertex_dyn_rung_rank / pol_vertex_dyn_rung_tol).
 *
 * The dense dynamic rung of the pair-space ladder is one (D x D) matrix per PH-representative tau node,
 * K_d(tau_i) = K[W_d(tau_i)] with K[.] the THC rung contraction (vertex_dynbse.icc, build_kbig), linear in the W-bar data.
 * The factorized form replaces the W-bar data by
 *     W_d(q', tau_i) ~ sum_{r < R} c_r(tau_i) A_r(q'),   A_r(q') = sum_i conj(c_r(tau_i)) W_d(q', tau_i),
 * with R UNIVERSAL (q'-independent) orthonormal time functions c_r, so the rung is sum_r c_r(tau_i) K_r, K_r = K[A_r]:
 * R matrices per unit instead of nt/2. Linearity makes the rung-level error of the approximation
 *     E_K(tau_i) = K[E(tau_i)],  E(tau_i) = W_d(tau_i) - sum_r c_r(tau_i) A_r = ((1 - C C^H) W_d)(tau_i),
 * and its Frobenius norm summed over the internal pairs (k, k') of one external transfer is a quadratic form in the W data,
 *     |K[E]|_F^2 = sum_pairs tr(E(q_kk')^H A_kk' E(q_kk') B_kk'^T),
 *     A_kk' = conj(S_k') o S_k,  B_kk' = conj(S_{k+q}) o S_{k'+q},  S_k = X_k X_k^H (N_m x N_m, the secondary collocation),
 * (o = the elementwise product; both factors are positive semidefinite). Averaging the pair metrics over the pairs, which
 * run over all of the internal k independently, gives the SEPARABLE proxy A ~ B ~ conj(S) o S with S the mean of S_k; the
 * rung-level Gram over the tau nodes is then approximated at the cost of the W-bar data alone,
 *     G(i, j) = sum_q' m(q') tr(What_i(q')^H What_j(q')),  What = L W_d conj(L),  L = (conj(S) o S)^(1/2),
 * m(q') = the number of full-mesh transfers whose rung reads the stored row q' (its star). The time functions are the top-R
 * eigenvectors of G, and the reported truncation error is
 *     e(R) = max_i |E(tau_i)|_G / max_i |W(tau_i)|_G = sqrt( max_i sum_{j >= R} lam_j |V_ij|^2 / max_i G_ii ),
 * read off the eigendecomposition for every R at once (the rank requested by a tolerance is the smallest R with e(R) <= tol).
 * vertex_debug dyn_rung_gram = legs (default) | mult (no leg metric) | incode (every row normalized
 * by its largest tau norm, no multiplicity) selects the Gram for comparisons.
 *
 * PH SYMMETRY. The dense per-tau rung uses ONE matrix per mirror pair, K_d(tau_i) = K[W_d(rep(i))], rep(i) = min(i, mirror(i)),
 * i.e. the PH-symmetrized data. The time functions are built on the same data, in the space of the PH representatives: the
 * Gram is over the ndist representatives with the rows sqrt(n_rep) What(rep) (n_rep = 1 or 2 nodes per representative), its
 * eigenvectors u give c_r(tau_i) = u_r(rep(i)) / sqrt(n_rep(i)) -- EXACTLY mirror-even (c_r(tau_i) == c_r(tau_mirror(i)),
 * bitwise), orthonormal over the nt nodes -- and A_r = sum_i conj(c_r(tau_i)) W_d(rep(i)). R <= ndist, and at R = ndist the
 * factorized rung IS the dense per-tau rung (the projector on the representatives is the identity). e(R) is measured per
 * node (the tail and the norm of a representative divided by its n_rep).
 *
 * The functions are computed ONCE per W update -- at the W-bar cache fill (vertex_t::cache_w), from every row the cache holds
 * (all IBZ transfers, or the sublattice transfers of a row-restricted cache, which are the only rows any ladder of that run
 * reads) -- and every consumer of that cache (the P refresh, the direct P, the Sigma reference pass, the sampled Sigma) uses
 * the same c_r: the truncation acts on the rung, i.e. on Phi, identically for both cuts. The eigenvectors carry a canonical
 * phase (the largest component real positive), are computed on one rank and broadcast. They are kept in
 * <prefix>.rungbasis.h5 with a probe of the W-bar data they came from: a restarted run that recomputes the same W-bar (to the
 * checkpoint tolerance) reloads the stored functions instead, so it reproduces them bitwise.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "nda/nda.hpp"
#include "nda/linalg/eigenelements.hpp"
#include "utilities/check.hpp"

namespace methods::solvers::vertex_rung {

  using cplx = std::complex<double>;

  /** the format of the stored sets: 2 = mirror-even (the PH representatives); sets of another format are never reloaded */
  inline constexpr long basis_format = 2;

  /** the factorized rung's time functions (one per W update; see the file comment) */
  struct rung_basis {
    long R = 0;                       // the number of time functions in use
    long nt = 0;                      // the tau nodes of the vertex grid they live on
    long ndist = 0;                   // its PH representatives (the largest R; R = ndist is the dense per-tau rung)
    nda::array<cplx, 2> c;            // (nt, R): orthonormal columns c_r(tau_i)
    nda::array<double, 1> err;        // (nt + 1): the rung-level error estimate e(R') for R' = 0 .. nt
    double tol = 0.0;                 // the requested tolerance (0 = a fixed rank)
    long rank_req = 0;                // the requested rank / cap (0 = none)
    bool capped = false;              // the tolerance asked for more than the cap allowed
    std::string gram;                 // the Gram construction (legs | mult | incode)
    std::string hash;                 // FNV-1a of (R, c) as 16 hex digits: the same in every consumer's log
    std::string source;               // how this instance came about (computed / reloaded from <file>)
    long gen = 0;                     // the cache fill (cache_w call) of this vertex instance that produced it
    long nrow = 0;                    // the W-bar rows it was computed from
    bool on() const { return R > 0; }
    double err_R() const { return (err.size() > R) ? err(R) : 0.0; }
  };

  /** FNV-1a of the rank and the time functions, 16 hex digits */
  inline std::string basis_hash(long R, nda::array<cplx, 2> const &c) {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](void const *p, size_t nb) {
      auto const *b = static_cast<unsigned char const *>(p);
      for (size_t i = 0; i < nb; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    };
    const long fmt = basis_format;
    mix(&fmt, sizeof(fmt));
    mix(&R, sizeof(R));
    mix(c.data(), size_t(c.size()) * sizeof(cplx));
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    return std::string(buf);
  }

  /** the PH representatives of a tau grid (the dense rung's): rep_of(i) = the representative index of node i, first(r) its
   *  representative node min(i, mirror(i)), cnt(r) = 1 or 2 nodes */
  struct ph_reps {
    std::vector<long> rep_of, node, cnt;
    long size() const { return long(node.size()); }
  };
  inline ph_reps make_ph_reps(nda::array<long, 1> const &t_mirror) {
    ph_reps p;
    const long nt = t_mirror.size();
    p.rep_of.assign(size_t(nt), -1);
    for (long i = 0; i < nt; ++i) {
      const long r0 = std::min(i, long(t_mirror(i)));
      long slot = -1;
      for (long r = 0; r < p.size(); ++r) if (p.node[size_t(r)] == r0) { slot = r; break; }
      if (slot < 0) { slot = p.size(); p.node.push_back(r0); p.cnt.push_back(0); }
      p.rep_of[size_t(i)] = slot;
      ++p.cnt[size_t(slot)];
    }
    return p;
  }

  /** the eigendecomposition of the tau Gram (descending), the canonical phases and e(R) for every R */
  struct gram_eig {
    nda::array<double, 1> lam;        // (nt) descending, clipped at 0
    nda::array<cplx, 2> V;            // (nt, nt) columns = eigenvectors, canonical phase
    nda::array<double, 1> err;        // (nt + 1)
  };
  /** w (optional, nt): the per-row weight of the error measure (1 / n_rep in the representatives' space) */
  inline gram_eig rung_gram_eig(nda::matrix<cplx> G, std::vector<double> const *w = nullptr) {
    const long nt = G.shape(0);
    utils::check(G.shape(1) == nt and nt > 0, "rung_gram_eig: the tau Gram is not square ({} x {}).", G.shape(0), G.shape(1));
    for (long i = 0; i < nt; ++i)
      for (long j = 0; j < i; ++j) { const cplx a = 0.5 * (G(i, j) + std::conj(G(j, i))); G(i, j) = a; G(j, i) = std::conj(a); }
    for (long i = 0; i < nt; ++i) G(i, i) = cplx(G(i, i).real(), 0.0);
    auto [lam_a, Va] = nda::linalg::eigenelements(G);   // ascending
    gram_eig e;
    e.lam = nda::array<double, 1>(nt);
    e.V = nda::array<cplx, 2>(nt, nt);
    for (long r = 0; r < nt; ++r) {
      const long s = nt - 1 - r;
      e.lam(r) = std::max(double(lam_a(s)), 0.0);
      // canonical phase: the largest component (the first within 1e-10 of the maximum) real positive
      double vmax = 0.0;
      for (long i = 0; i < nt; ++i) vmax = std::max(vmax, std::abs(Va(i, s)));
      long imax = 0;
      for (long i = 0; i < nt; ++i)
        if (std::abs(Va(i, s)) >= (1.0 - 1e-10) * vmax) { imax = i; break; }
      const cplx ph = (vmax > 0.0) ? std::conj(Va(imax, s)) / std::abs(Va(imax, s)) : cplx(1.0);
      for (long i = 0; i < nt; ++i) e.V(i, r) = ph * Va(i, s);
    }
    // e(R)^2 = max_i sum_{j >= R} lam_j |V_ij|^2 / max_i sum_j lam_j |V_ij|^2  (tail sums per node, O(nt^2))
    e.err = nda::array<double, 1>(nt + 1);
    std::vector<double> tail(static_cast<size_t>(nt), 0.0);
    double gmax = 0.0;
    for (long i = 0; i < nt; ++i) {
      double s = 0.0;
      for (long j = 0; j < nt; ++j) s += e.lam(j) * std::norm(e.V(i, j));
      gmax = std::max(gmax, s * (w ? (*w)[size_t(i)] : 1.0));
    }
    for (long R = nt; R >= 0; --R) {
      if (R < nt)
        for (long i = 0; i < nt; ++i) tail[size_t(i)] += e.lam(R) * std::norm(e.V(i, R));
      double m = 0.0;
      for (long i = 0; i < nt; ++i) m = std::max(m, tail[size_t(i)] * (w ? (*w)[size_t(i)] : 1.0));
      e.err(R) = (gmax > 0.0) ? std::sqrt(m / gmax) : 0.0;
    }
    return e;
  }

  /** the rank: rank_req alone (> 0), the smallest R with e(R) <= tol, or that R capped at rank_req when both are set */
  inline long rung_select(gram_eig const &e, long rank_req, double tol, bool *capped) {
    const long nt = e.lam.size();
    if (capped) *capped = false;
    utils::check(rank_req >= 0 and rank_req <= nt, "rung_select: pol_vertex_dyn_rung_rank = {} is outside [1, {}] (the PH "
                 "representatives of the vertex tau grid: at {} the factorized rung is the dense per-tau rung). ABORTING.", rank_req, nt, nt);
    if (tol <= 0.0) return rank_req;
    long R = nt;
    for (long r = 1; r <= nt; ++r)
      if (e.err(r) <= tol) { R = r; break; }
    if (rank_req > 0 and R > rank_req) {
      if (capped) *capped = true;
      R = rank_req;
    }
    return R;
  }

} // namespace methods::solvers::vertex_rung

#endif
