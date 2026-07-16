/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * CoQui-side reimplementation of QE's qvan2: evaluate Q^{IJ}(K) for any
 * cartesian K = q + G via radial spherical-Bessel transform of qfuncl
 * combined with the angular-momentum coupling table aainit.
 *
 * Needed because the cached qgm tensor exported by pw2coqui is computed
 * only on the q=0 dense G grid. For HF exchange and post-HF methods we
 * need V_{GL}/V_{LL} at all q's in the IBZ, hence Q^{IJ}(q+G).
 *
 * Mathematical formula (.tex Eq. local-bloch-eta + qvan2 docs):
 *   Q^{IJ}(K) = Σ_{LM} (-i)^L · ap(LM, lm_I, lm_J) · Y_{LM}(K̂) · J_L(|K|)
 * where
 *   J_L(K) = ∫₀^∞ r² qfuncl_L(r) j_L(Kr) dr
 *   ap(LM, lm_I, lm_J) = ∫_{Ω} Y_{LM}(r̂) Y_{lm_I}(r̂) Y_{lm_J}(r̂) dΩ
 * (real spherical harmonics convention, matches CoQui's
 *  utils::spherical_harmonics_l).
 *
 * The (lm_I, lm_J) → (LM) coupling sparsity is encoded in (lpx, lpl) so
 * inner loops only touch the non-zero LM channels.
 * ==========================================================================
 */
#ifndef HAMILTONIAN_PAW_PAW_AUG_Q_EVAL_HPP
#define HAMILTONIAN_PAW_PAW_AUG_Q_EVAL_HPP

#include <cmath>
#include <random>
#include <vector>
#include <array>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/harmonics.h"
#include "utilities/basis_set_utilities.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "nda/clef.hpp"
#include "nda/nda.hpp"

namespace hamilt::paw {

/**
 * Angular-momentum coupling tables computed by aainit_tables().
 *   ap(lp, li, lj)  : real coupling coefficient, lp,li,lj ∈ [0, llx).
 *   lpx(li, lj)     : count of non-zero lp's for the (li,lj) pair.
 *   lpl(li, lj, k)  : k-th non-zero lp for that pair, k ∈ [0, lpx(li,lj)).
 *
 * Index convention (matches QE's lpx/lpl/ap and CoQui's spherical_harmonics):
 *   lp = L*(L+1) + M with M ∈ [0, 2L+1) — 0-based combined LM index.
 *   QE stores nhtolm 1-based; subtract 1 before indexing.
 *   llx = (2*lli - 1)² where lli = lmax + 1 (so the maximum LM dim).
 */
struct aainit_tables {
    int lli  = 0;     // max li (number of l-channels: l=0..lli-1)
    int llx  = 0;     // (2*lli - 1)² — full LM dim including L = 2*(lli-1)
    int mx   = 0;     // upper bound on lpx (= max non-zero lp's per pair)
    nda::array<double, 3> ap;     // (llx, lli², lli²) — built in QE m-order
    nda::array<int, 2>    lpx;    // (lli², lli²)
    nda::array<int, 3>    lpl;    // (lli², lli², mx) — lp values in QE m-order
};

/**
 * Convert a QE-flat lp index (0-based, m-ordering = 0,+1,-1,+2,-2,...
 * within each L) to the corresponding CoQui flat index of
 * `utils::harmonics::spherical_harmonics` (m-ordering = -L,...,0,...,+L).
 */
inline int qe_lp_to_coqui_lp(int lp_qe)
{
    int L = (int)std::floor(std::sqrt((double)lp_qe + 1e-9));
    int qe_within = lp_qe - L*L;     // 0..2L
    int coqui_within;
    if (qe_within == 0) coqui_within = L;       // m=0
    else {
        int k = (qe_within + 1) / 2;            // |m|
        coqui_within = (qe_within % 2 == 1) ? L + k : L - k;
    }
    return L*L + coqui_within;
}

/**
 * QE-compatible ylmr2 for a single point. Matches QE/upflib/ylmr2.f90 exactly
 * including sign conventions (Condon-Shortley P_l^m used directly, no extra
 * (-1)^m flip — different from CoQui's wannier90-style spherical_harmonics).
 *
 * Output `Yflat` is in QE flat-LM order (within each L: m=0, +1, -1, +2, -2, ...).
 *
 * Implementation tracks ylmr2.f90's recursion + post-processing exactly.
 */
inline void qe_real_ylm_flat(int Lmax, std::array<double, 3> const& g,
                              nda::array<double, 1>& Yflat)
{
    int lmax2 = (Lmax + 1) * (Lmax + 1);
    Yflat() = 0.0;
    if (lmax2 < 1) return;
    constexpr double eps = 1e-9;
    constexpr double pi  = M_PI;
    constexpr double fpi = 4.0 * M_PI;

    if (Lmax == 0) {
        Yflat(0) = std::sqrt(1.0 / fpi);
        return;
    }

    double gmod = std::sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);
    double cost = (gmod < eps) ? 0.0 : g[2]/gmod;
    double sent = std::sqrt(std::max(0.0, 1.0 - cost*cost));

    // Q_l^m recursion in QE's storage indexing: Yflat(lm-1) where lm = l²+1+2m
    // (1-based), so 0-based lp = l²+2m for m=0..l-1, and lp = l²+2l for max m.
    // We will assign within-L offsets (0-based within each L's block):
    //   QE's lm=l²+1+2m → 0-based offset = 2m for m=0..l-1 (after later
    //   re-shuffle by the post-processing loop).
    // But the recursion phase fills these offsets pre-shuffle; post-processing
    // (line 117-130) then merges Q_l^m with cos/sin(m·φ).
    //
    // Easiest: just allocate a (Lmax+1)² buffer and run the recursion exactly
    // as QE does (using QE's lm = l²+1+2m offsets, 1-based), then the post-
    // processing pass overwrites in place.

    // Use 1-based indexing internally to match QE; final output indexed 0-based.
    nda::array<double, 1> Y1(lmax2 + 1);   // index 1..lmax2
    Y1() = 0.0;
    Y1(1) = 1.0;
    if (Lmax >= 1) {
        Y1(2) = cost;
        Y1(4) = -sent / std::sqrt(2.0);
    }
    for (int l = 2; l <= Lmax; ++l) {
        for (int m = 0; m <= l - 2; ++m) {
            int lm  = l*l + 1 + 2*m;
            int lm1 = (l-1)*(l-1) + 1 + 2*m;
            int lm2 = (l-2)*(l-2) + 1 + 2*m;
            double a = (double)(2*l - 1) / std::sqrt((double)(l*l - m*m));
            double b = std::sqrt((double)((l-1)*(l-1) - m*m)) /
                       std::sqrt((double)(l*l - m*m));
            Y1(lm) = cost*a*Y1(lm1) - b*Y1(lm2);
        }
        int lm  = l*l + 1 + 2*l;
        int lm1 = l*l + 1 + 2*(l-1);
        int lm2 = (l-1)*(l-1) + 1 + 2*(l-1);
        Y1(lm1) = cost * std::sqrt((double)(2*l - 1)) * Y1(lm2);
        Y1(lm)  = -std::sqrt((double)(2*l - 1)) /
                   std::sqrt((double)(2*l)) * sent * Y1(lm2);
    }
    // Post-processing: cos/sin(m·φ) and 1/sqrt(fpi) factors.
    double phi;
    if (g[0] > eps)        phi = std::atan(g[1] / g[0]);
    else if (g[0] < -eps)  phi = std::atan(g[1] / g[0]) + pi;
    else                   phi = std::copysign(pi/2.0, g[1]);

    Y1(1) = Y1(1) / std::sqrt(fpi);
    int lm = 1;
    for (int l = 1; l <= Lmax; ++l) {
        double c = std::sqrt((double)(2*l + 1) / fpi);
        // Y_l,0
        lm = lm + 1;
        Y1(lm) = c * Y1(lm);
        for (int m = 1; m <= l; ++m) {
            lm = lm + 2;
            // Read Y1(lm) BEFORE overwriting — same as QE.
            double tmp = Y1(lm);
            Y1(lm - 1) = c * std::sqrt(2.0) * tmp * std::cos(m * phi); // Y_l,+m
            Y1(lm)     = c * std::sqrt(2.0) * tmp * std::sin(m * phi); // Y_l,-m
        }
    }
    for (int i = 0; i < lmax2; ++i) Yflat(i) = Y1(i + 1);
}

/**
 * Adapter wrapping a UPF radial mesh (r, rab) into the duck-typed grid
 * interface expected by `utils::sph_bessel_transform_boost` / `simpson_rule_f`
 * (size(), r(i), r_dr(i)). On UPF meshes rab(i) = dr/di so simpson uses
 * rab(i) as the weight.
 */
struct upf_radial_grid {
    nda::array<double, 1> const* rp = nullptr;
    nda::array<double, 1> const* rabp = nullptr;
    upf_radial_grid(nda::array<double, 1> const& r,
                    nda::array<double, 1> const& rab)
        : rp(&r), rabp(&rab) {}
    long size() const {
        long n = (long)rp->extent(0);
        // simpson_rule_f requires odd N; clip to nearest odd if needed.
        return (n % 2 == 1) ? n : n - 1;
    }
    double r(long i) const { return (*rp)(i); }
    double dr(long i) const { return (*rabp)(i); }
    auto r_dr(long i) const { return std::make_tuple((*rp)(i), (*rabp)(i)); }
};

namespace detail {

// Generate llx points roughly uniformly distributed on the unit sphere.
// Algorithm matches QE's gen_rndm_r (uniform deviate on cosθ × azimuthal).
// We use std::mt19937 with a fixed seed for reproducibility — the absolute
// values of ap depend on the points, but two evaluations with the same
// seed give bit-exact results.
inline void gen_rndm_r(int llx,
                       nda::array<double, 2>& r,
                       nda::array<double, 1>& rr)
{
    std::mt19937_64 rng(0xC001C001ULL);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int ir = 0; ir < llx; ++ir) {
        double costheta = 2.0 * u(rng) - 1.0;
        double sintheta = std::sqrt(std::max(0.0, 1.0 - costheta*costheta));
        double phi = 2.0 * M_PI * u(rng);
        r(ir, 0) = sintheta * std::cos(phi);
        r(ir, 1) = sintheta * std::sin(phi);
        r(ir, 2) = costheta;
        rr(ir)   = 1.0;
    }
}

// In-place matrix inverse via nda's getrf+getri or equivalent.
inline void invert_in_place(nda::array<double, 2>& A)
{
    // Use nda::inverse_in_place if available; fall back to LAPACK directly.
    long n = A.extent(0);
    utils::check(A.extent(1) == n, "invert_in_place: not square ({} x {})",
                 A.extent(0), A.extent(1));
    nda::matrix<double> M(n, n);
    M = A;
    M = nda::inverse(M);
    A = M;
}

} // namespace detail

/**
 * Build (ap, lpx, lpl) for spherical harmonics up to L = 2*(lli-1).
 *
 * `lli` should equal max_l + 1 over all projector channels (so lli = lmax+1
 * where lmax = max(lll)). For typical PAW pseudos (s,p,d projectors,
 * lmax = 2), lli = 3 → llx = 25 (Lmax = 4).
 *
 * Cost: O(llx³) for the inverse + O(llx⁴) for filling ap. Trivial.
 */
inline aainit_tables aainit_tables_build(int lli)
{
    aainit_tables T;
    T.lli = lli;
    int Lmax = 2*(lli - 1);
    T.llx = (Lmax + 1) * (Lmax + 1);
    int npairs = lli * lli;
    int llx = T.llx;

    nda::array<double, 2> r(llx, 3);
    nda::array<double, 1> rr(llx);
    nda::array<double, 2> ylm(llx, llx);
    nda::array<double, 2> mly(llx, llx);
    r()   = 0.0; ylm() = 0.0; mly() = 0.0;

    detail::gen_rndm_r(llx, r, rr);

    // ylm(ir, lp_qe) = Y_LM at point r_ir, in QE flat-LM ordering and QE
    // sign convention (matches qvan2's expected ylmk0 input exactly).
    {
        for (int ir = 0; ir < llx; ++ir) {
            nda::array<double, 1> Yflat(llx);
            qe_real_ylm_flat(Lmax, {r(ir, 0), r(ir, 1), r(ir, 2)}, Yflat);
            for (int lp = 0; lp < llx; ++lp) ylm(ir, lp) = Yflat(lp);
        }
    }

    // mly = ylm^{-1}, but we want mly indexed as (lp, ir) (row = lp).
    // QE: mly = invmat(ylm), then ap = sum_ir mly(l, ir) * ylm(ir, li) * ylm(ir, lj)
    // The "matrix" interpretation: rows are evaluation points, columns are
    // basis functions. We invert and treat the resulting matrix as (lp, ir).
    mly = ylm;
    detail::invert_in_place(mly);
    // mly is now mly_(orig)^{-1} = invlmy(lp, ir) where the index roles flip:
    // original ylm(ir, lp) row=ir col=lp, inverse has row=lp col=ir conceptually.

    // ap(lp, li, lj) = Σ_ir mly(lp, ir) × ylm(ir, li) × ylm(ir, lj)
    T.ap = nda::array<double,3>::zeros({(long)llx, (long)npairs, (long)npairs});
    T.lpx = nda::array<int,2>::zeros({(long)npairs, (long)npairs});
    int mx_used = 0;
    // Fill ap, count non-zeros for lpx.
    nda::array<int, 3> lpl_buf(npairs, npairs, llx);
    lpl_buf() = 0;
    for (int li = 0; li < npairs; ++li) {
        for (int lj = 0; lj < npairs; ++lj) {
            int cnt = 0;
            for (int lp = 0; lp < llx; ++lp) {
                double a = 0.0;
                for (int ir = 0; ir < llx; ++ir)
                    a += mly(lp, ir) * ylm(ir, li) * ylm(ir, lj);
                T.ap(lp, li, lj) = a;
                if (std::abs(a) > 1e-3) {
                    lpl_buf(li, lj, cnt) = lp;
                    ++cnt;
                }
            }
            T.lpx(li, lj) = cnt;
            mx_used = std::max(mx_used, cnt);
        }
    }
    T.mx = mx_used;
    T.lpl = nda::array<int,3>::zeros(
        {(long)npairs, (long)npairs, (long)std::max(1, mx_used)});
    for (int li = 0; li < npairs; ++li)
        for (int lj = 0; lj < npairs; ++lj)
            for (int k = 0; k < T.lpx(li, lj); ++k)
                T.lpl(li, lj, k) = lpl_buf(li, lj, k);
    return T;
}

/**
 * Radial Bessel transform of qfuncl per L channel for ONE species, ONE pair
 * ij at a single |K|. Returns a vector indexed by L of length lmax_aug+1.
 *
 *   J_L(K) = ∫₀^∞ r² qfuncl_L_ij(r) j_L(Kr) dr
 *
 * (No 4π or i^L — those are applied by the caller in the qvan2 sum.)
 */
/**
 * Per-species precomputed radial Bessel table — analogue of QE's tab_qrad.
 *
 *   tab(iK, ijv, L) = ∫ qfuncl(L, ijv, r) j_L(K_iK r) dr,    K_iK = iK · dq
 *
 * Filled once per species over `iK ∈ [0, n_K)`. Runtime evaluation is a
 * 4-point cubic interpolation in iK (`qrad_interp_at_K` below). Replaces
 * the per-(q+G) Bessel transform inside `build_eta_on_rho_g_at_q` and
 * brings Si-class fixtures (lmax=2, Lmax_aug=4, ngm~5000) from minutes
 * per q to milliseconds per q. (4π/Ω) prefactor is baked in (matches
 * QE's tab_qrad convention).
 */
struct qrad_tab {
    double dq      = 0.01;     // grid spacing in cartesian a.u.
    long   n_K     = 0;        // number of grid points
    nda::array<double, 3> tab; // (n_K, n_ijv, Lp1) — flattened for fast iK access
};

/**
 * Build a qrad table for one species, sampling K in [0, K_max + safety) on
 * a uniform grid of spacing dq. The Bessel quadrature uses Simpson's rule
 * on the species's UPF radial mesh (no extra r² — qfuncl carries it).
 *
 * Bare radial integral (no 4π/Ω prefactor) — callers (build_eta_on_rho_g_at_q,
 * evaluate_Q_IJ_at_K) apply that factor at the end of their angular sum.
 */
inline qrad_tab build_qrad_tab(
    pseudopot::species_paw_t const& sp,
    double K_max,
    double dq = 0.01)
{
    qrad_tab T;
    T.dq = dq;
    if (sp.qfuncl.size() == 0) return T;
    long Lp1   = sp.qfuncl.extent(0);
    long n_ijv = sp.qfuncl.extent(1);
    long mesh_full = sp.qfuncl.extent(2);
    if (mesh_full == 0) return T;

    // QE pattern: integrate only out to kkbeta (the projector cutoff radius).
    // qfuncl is zero beyond kkbeta by construction, so extending to mesh
    // wastes 5–10× of the radial work for typical UPFs.
    long mesh = (sp.kkbeta > 0 && sp.kkbeta <= (int)mesh_full)
              ? (long)sp.kkbeta : mesh_full;

    // 3 extra grid points to allow 4-point cubic interpolation up through
    // the highest sampled K without indexing past the end.
    T.n_K = (long)std::ceil(K_max / dq) + 4;
    T.tab = nda::array<double, 3>::zeros({T.n_K, n_ijv, Lp1});

    long N = (mesh % 2 == 1) ? mesh : mesh - 1;  // odd for Simpson
    using boost::math::sph_bessel;

    // Precompute Simpson weights × rab (same for all iK, L, ijv).
    nda::array<double, 1> simp(N);
    {
        simp(0)     = sp.rab(0);
        simp(N - 1) = sp.rab(N - 1);
        double fct = 4.0;
        int sg = -1;
        for (long i = 1; i < N - 1; ++i) {
            simp(i) = fct * sp.rab(i);
            fct += sg * 2.0;
            sg = -sg;
        }
    }
    nda::array<double, 1> besr(N);
    for (long iK = 0; iK < T.n_K; ++iK) {
        double K = (double)iK * dq;
        for (long L = 0; L < Lp1; ++L) {
            // Cache j_L(K·r) once for this (iK, L) — independent of ijv.
            for (long i = 0; i < N; ++i) {
                double Kr = K * sp.r(i);
                besr(i) = (std::abs(Kr) < 1e-30) ? (L == 0 ? 1.0 : 0.0)
                                                 : sph_bessel((unsigned int)L, Kr);
            }
            for (long ijv = 0; ijv < n_ijv; ++ijv) {
                double F = 0.0;
                for (long i = 0; i < N; ++i)
                    F += simp(i) * sp.qfuncl(L, ijv, i) * besr(i);
                T.tab(iK, ijv, L) = F / 3.0;
            }
        }
    }
    return T;
}

/**
 * Shape-restored ("Option A") variant of build_qrad_tab.
 *
 * Instead of the compensation charge qfuncl (moment × shape function), this
 * transforms the FULL AE−PS on-site pair density
 *
 *   D_ijv(r) = φ_nb(r) φ_mb(r) − φ̃_nb(r) φ̃_mb(r)
 *            = aewfc(nb,r)·aewfc(mb,r) − pswfc(nb,r)·pswfc(mb,r)
 *
 * (u = r·R form, so it already carries r²; no extra r² in the transform, same
 * as qfuncl). The radial function is L-independent — the L-decomposition is
 * purely angular via the Gaunt coefficients `ap` applied by the caller — so
 * tab(iK, ijv, L) = ∫ D_ijv(r) j_L(K r) dr for every L in [0, Lp1).
 *
 * Fed into `evaluate_Q_IJ_at_K_fast` in place of the compensation-charge table,
 * this reproduces ABINIT's `phiphj - tphitphj` oscillator form factor
 * (m_pawpwij.F90): the resulting augmented pair density is the exact
 * all-electron pair density (up to the FFT-mesh cutoff), so the smooth+aug
 * Coulomb contraction alone gives the exact on-site exchange and the separate
 * `deltaC` one-center correction is dropped. See
 * notes/paw_onsite_exchange_analysis.{md,pdf}, Route A.
 *
 * D → 0 beyond kkbeta (PAW: φ = φ̃ outside r_c), so the kkbeta truncation is
 * exact here too. PAW-only (requires aewfc/pswfc); returns an empty table
 * otherwise so callers can fall back.
 */
inline qrad_tab build_qrad_tab_full_aeps(
    pseudopot::species_paw_t const& sp,
    double K_max,
    double dq = 0.01)
{
    qrad_tab T;
    T.dq = dq;
    if (sp.aewfc.size() == 0 || sp.pswfc.size() == 0) return T;
    long nbeta = sp.nbeta;
    long n_ijv = nbeta * (nbeta + 1) / 2;
    long mesh_full = sp.aewfc.extent(1);
    if (mesh_full == 0 || n_ijv == 0) return T;

    // L-channel count: match the compensation table's Lp1 when present so the
    // angular sum in evaluate_Q_IJ_at_K_fast indexes the table identically;
    // else 2*lmax_beta + 1 (the full AE−PS pair density spans |li−lj|..li+lj).
    long Lp1;
    if (sp.qfuncl.size() > 0) {
        Lp1 = sp.qfuncl.extent(0);
    } else {
        int lmax_beta = 0;
        for (long b = 0; b < nbeta; ++b) lmax_beta = std::max(lmax_beta, (int)sp.lll(b));
        Lp1 = 2 * lmax_beta + 1;
    }

    long mesh = (sp.kkbeta > 0 && sp.kkbeta <= (int)mesh_full)
              ? (long)sp.kkbeta : mesh_full;
    T.n_K = (long)std::ceil(K_max / dq) + 4;
    T.tab = nda::array<double, 3>::zeros({T.n_K, n_ijv, Lp1});

    long N = (mesh % 2 == 1) ? mesh : mesh - 1;  // odd for Simpson
    using boost::math::sph_bessel;

    // Simpson weights × rab (same for all iK, L, ijv).
    nda::array<double, 1> simp(N);
    {
        simp(0)     = sp.rab(0);
        simp(N - 1) = sp.rab(N - 1);
        double fct = 4.0;
        int sg = -1;
        for (long i = 1; i < N - 1; ++i) {
            simp(i) = fct * sp.rab(i);
            fct += sg * 2.0;
            sg = -sg;
        }
    }

    // AE−PS pair density D_ijv(r) for every beta pair (nb ≥ mb), indexed by the
    // same triangular ijv used by qfuncl / evaluate_Q_IJ_at_K.
    nda::array<double, 2> D(n_ijv, N);
    D() = 0.0;
    for (long nb = 0; nb < nbeta; ++nb)
        for (long mb = 0; mb <= nb; ++mb) {
            long ijv = nb * (nb + 1) / 2 + mb;
            for (long i = 0; i < N; ++i)
                D(ijv, i) = sp.aewfc(nb, i) * sp.aewfc(mb, i)
                          - sp.pswfc(nb, i) * sp.pswfc(mb, i);
        }

    nda::array<double, 1> besr(N);
    for (long iK = 0; iK < T.n_K; ++iK) {
        double K = (double)iK * dq;
        for (long L = 0; L < Lp1; ++L) {
            for (long i = 0; i < N; ++i) {
                double Kr = K * sp.r(i);
                besr(i) = (std::abs(Kr) < 1e-30) ? (L == 0 ? 1.0 : 0.0)
                                                 : sph_bessel((unsigned int)L, Kr);
            }
            for (long ijv = 0; ijv < n_ijv; ++ijv) {
                double F = 0.0;
                for (long i = 0; i < N; ++i)
                    F += simp(i) * D(ijv, i) * besr(i);
                T.tab(iK, ijv, L) = F / 3.0;
            }
        }
    }
    return T;
}

/**
 * 4-point cubic interpolation of qrad at arbitrary K, matching the formula
 * in QE/upflib/qvan2.f90 lines 143-157. Returns an array of length Lp1
 * with the per-L radial values for the requested ijv.
 */
inline nda::array<double, 1> qrad_interp_at_K(
    qrad_tab const& T, int ijv, double K)
{
    long Lp1 = T.tab.extent(2);
    nda::array<double, 1> out(Lp1);
    out() = 0.0;
    if (T.n_K == 0) return out;
    double qm = K / T.dq;
    long i0 = (long)std::floor(qm);
    if (i0 < 0) i0 = 0;
    if (i0 + 3 >= T.n_K) i0 = T.n_K - 4;
    double px = qm - (double)i0;
    double ux = 1.0 - px;
    double vx = 2.0 - px;
    double wx = 3.0 - px;
    double uvx = ux * vx * (1.0 / 6.0);
    double pwx = px * wx * 0.5;
    for (long L = 0; L < Lp1; ++L) {
        out(L) = T.tab(i0,     ijv, L) * uvx * wx
               + T.tab(i0 + 1, ijv, L) * pwx * vx
               - T.tab(i0 + 2, ijv, L) * pwx * ux
               + T.tab(i0 + 3, ijv, L) * px * uvx;
    }
    return out;
}

inline nda::array<double, 1> qrad_at_K(
    pseudopot::species_paw_t const& sp, int ij, double K)
{
    long Lp1   = sp.qfuncl.extent(0);
    long mesh  = sp.qfuncl.extent(2);
    nda::array<double, 1> out(Lp1);
    out() = 0.0;
    if (sp.qfuncl.size() == 0 || mesh == 0) return out;

    // QE's qfuncl(L, ij, r) ALREADY has r² baked in (UPF convention), so
    // the Bessel transform is just ∫ qfuncl × j_L(Kr) × rab dr — NO extra
    // r² factor. Integrate via Simpson's rule over the UPF radial mesh
    // (rab(i) = dr/du · du; using rab as weight directly).
    using boost::math::sph_bessel;
    long N = (mesh % 2 == 1) ? mesh : mesh - 1;  // odd for Simpson
    for (long L = 0; L < Lp1; ++L) {
        double F = 0.0;
        // Endpoints with weight 1.
        {
            double q0 = std::abs(K * sp.r(0)) < 1e-30 ? (L == 0 ? 1.0 : 0.0)
                       : sph_bessel((unsigned int)L, K * sp.r(0));
            double qN = sph_bessel((unsigned int)L, K * sp.r(N - 1));
            F += sp.qfuncl(L, ij, 0)     * q0 * sp.rab(0);
            F += sp.qfuncl(L, ij, N - 1) * qN * sp.rab(N - 1);
        }
        double fct = 4.0;
        int sg = -1;
        for (long i = 1; i < N - 1; ++i) {
            double ri = sp.r(i);
            double jL = (std::abs(K * ri) < 1e-30) ? (L == 0 ? 1.0 : 0.0)
                       : sph_bessel((unsigned int)L, K * ri);
            F += fct * sp.qfuncl(L, ij, i) * jL * sp.rab(i);
            fct += sg * 2.0;
            sg = -sg;
        }
        out(L) = F / 3.0;
    }
    return out;
}

/**
 * Full qvan2-equivalent: Q^{IJ}_nt(K) for ONE species, projector-pair (ih,jh)
 * (ih, jh ∈ [0, nh)), and a single K_vec (cartesian).
 *
 * Convention: matches QE's qvan2 (= the qgm tensor exported by pw2coqui),
 *   Q^{IJ}(K) = (4π/Ω) Σ_lp ap(lp, ivl, jvl) × (-i)^L × Y_lp(K̂) × J_L(|K|)
 *
 * The (4π/Ω) prefactor is baked into QE's tab_qrad (line 138 of qrad_mod.f90),
 * so any consumer reading qgm directly inherits it. We apply it here so that
 * `evaluate_Q_IJ_at_K` is drop-in compatible with the cached qgm tensor.
 *
 * `omega_volume` is the unit-cell volume Ω. Caller must NOT include the
 * structure factor e^{-iK·τ_a} — that is layered on at the call site.
 */
inline ComplexType evaluate_Q_IJ_at_K(
    pseudopot::species_paw_t const& sp,
    aainit_tables const& aatab,
    int ih, int jh,
    std::array<double, 3> const& K_vec,
    double omega_volume)
{
    if (sp.nhtolm.size() == 0 || sp.indv.size() == 0 || sp.lll.size() == 0) {
        return ComplexType(0.0);
    }
    int ivl = sp.nhtolm(ih) - 1;  // QE 1-based → 0-based
    int jvl = sp.nhtolm(jh) - 1;
    int nb  = sp.indv(ih) - 1;
    int mb  = sp.indv(jh) - 1;
    int nbeta = sp.nbeta;
    utils::check(nb >= 0 && mb >= 0 && nb < nbeta && mb < nbeta,
                 "evaluate_Q_IJ_at_K: indv out of range");
    int n1 = std::max(nb, mb), n2 = std::min(nb, mb);
    int ijv = (n1 * (n1 + 1)) / 2 + n2;          // 0-based ijv
    utils::check(ijv >= 0 && ijv < (int)sp.qfuncl.extent(1),
                 "evaluate_Q_IJ_at_K: ijv={} out of range [0,{})",
                 ijv, sp.qfuncl.extent(1));

    double K = std::sqrt(K_vec[0]*K_vec[0] + K_vec[1]*K_vec[1] +
                         K_vec[2]*K_vec[2]);

    auto qradL = qrad_at_K(sp, ijv, K);  // (Lp1)

    // Y_lp(K̂) in QE flat-LM order, using QE's exact ylmr2 convention.
    int llx = aatab.llx;
    nda::array<double, 1> Yflat_qe(llx);
    Yflat_qe() = 0.0;
    {
        std::array<double, 3> dir;
        if (K > 1e-14) {
            dir = {K_vec[0]/K, K_vec[1]/K, K_vec[2]/K};
        } else {
            dir = {0.0, 0.0, 1.0};
        }
        int Lmax = 2*(aatab.lli - 1);
        qe_real_ylm_flat(Lmax, dir, Yflat_qe);
        if (K <= 1e-14) {
            // Only L=0 component is well-defined at K=0.
            for (int lp = 1; lp < llx; ++lp) Yflat_qe(lp) = 0.0;
        }
    }

    ComplexType qout(0.0, 0.0);
    int Lp1 = (int)qradL.extent(0);
    int npairs = aatab.lli * aatab.lli;
    utils::check(ivl < npairs && jvl < npairs,
                 "evaluate_Q_IJ_at_K: ivl/jvl out of aatab range");
    int n_lp = aatab.lpx(ivl, jvl);
    for (int k = 0; k < n_lp; ++k) {
        int lp = aatab.lpl(ivl, jvl, k);
        // L from lp = L*(L+1) + M  (0-based, M ∈ [0, 2L+1)):
        int L = (int)std::floor(std::sqrt((double)lp + 1e-9));
        // sanity: L*(L+1) ≤ lp < (L+1)*(L+2)
        utils::check(L >= 0 && L < Lp1,
                     "evaluate_Q_IJ_at_K: L={} out of qfuncl L-range [{},{})",
                     L, 0, Lp1);
        // (-i)^L cycles 1, -i, -1, +i.
        ComplexType iL_factor;
        switch (L % 4) {
            case 0: iL_factor = ComplexType( 1.0,  0.0); break;
            case 1: iL_factor = ComplexType( 0.0, -1.0); break;
            case 2: iL_factor = ComplexType(-1.0,  0.0); break;
            case 3: iL_factor = ComplexType( 0.0,  1.0); break;
        }
        double coeff = aatab.ap(lp, ivl, jvl) * Yflat_qe(lp) * qradL(L);
        qout += iL_factor * ComplexType(coeff, 0.0);
    }
    // (4π/Ω) prefactor — baked into QE's tab_qrad (qrad_mod.f90:138).
    return qout * ComplexType((4.0 * M_PI) / omega_volume, 0.0);
}

} // namespace hamilt::paw

#endif // HAMILTONIAN_PAW_PAW_AUG_Q_EVAL_HPP
