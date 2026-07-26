/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Unit tests for the PAW radial scaffolding (paw_radial.hpp):
 *   - radial_simpson against polynomial/exponential analytic integrals,
 *   - radial_hartree_multipole against the uniform-sphere closed-form
 *     V_H(r) = 4π ρ₀ (R²/2 − r²/6)         inside,
 *            = 4π ρ₀ R³ / (3r)              outside,
 *     for L = 0; plus a check that L ≥ 1 gives 0 for spherically
 *     symmetric ρ.
 *
 * Synthetic test cases — no fixture dependency. Runs in milliseconds.
 * ==========================================================================
 */

#undef NDEBUG

#include "catch2/catch.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <utility>
#include <vector>

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "IO/app_loggers.h"

#include "hamiltonian/paw/paw_radial.hpp"
#include "hamiltonian/paw/paw_aug_q_eval.hpp"

namespace {

// Build a uniform radial mesh of `n` points on [0, r_max], stored as
// (r, rab) where rab(i) = dr/di = constant.
struct uniform_mesh {
    nda::array<double, 1> r;
    nda::array<double, 1> rab;
    double dr;
    uniform_mesh(long n, double r_max) : r(n), rab(n) {
        dr = r_max / (double)(n - 1);
        for (long i = 0; i < n; ++i) {
            r(i)   = (double)i * dr;
            rab(i) = dr;
        }
    }
};

} // namespace

TEST_CASE("paw_radial_simpson_polynomials", "[hamilt][paw][radial]")
{
    // Use a uniform mesh on [0, 1] with 401 points (Simpson exact for
    // cubics and below; tail trap fallback only used for even-n meshes).
    long n = 401;
    double r_max = 1.0;
    uniform_mesh m(n, r_max);

    // ∫₀¹ 1 dr = 1
    {
        nda::array<double, 1> f(n);
        f() = 1.0;
        double s = hamilt::paw::radial_simpson(f, m.rab);
        REQUIRE(std::abs(s - 1.0) < 1e-12);
    }
    // ∫₀¹ r dr = 1/2
    {
        nda::array<double, 1> f(n);
        for (long i = 0; i < n; ++i) f(i) = m.r(i);
        double s = hamilt::paw::radial_simpson(f, m.rab);
        REQUIRE(std::abs(s - 0.5) < 1e-10);
    }
    // ∫₀¹ r² dr = 1/3
    {
        nda::array<double, 1> f(n);
        for (long i = 0; i < n; ++i) f(i) = m.r(i) * m.r(i);
        double s = hamilt::paw::radial_simpson(f, m.rab);
        REQUIRE(std::abs(s - 1.0 / 3.0) < 1e-10);
    }
    // ∫₀¹ r³ dr = 1/4
    {
        nda::array<double, 1> f(n);
        for (long i = 0; i < n; ++i)
            f(i) = m.r(i) * m.r(i) * m.r(i);
        double s = hamilt::paw::radial_simpson(f, m.rab);
        REQUIRE(std::abs(s - 0.25) < 1e-10);
    }
}

TEST_CASE("paw_radial_hartree_uniform_sphere", "[hamilt][paw][radial]")
{
    // Uniform-sphere reference: ρ(r) = ρ₀ Θ(R − r), ρ_00(r) = ρ₀ · √(4π).
    // V_00(r) (= Y_00 component of V_H, NOT the full V_H), so V_H(r) =
    // V_00(r) · Y_00 = V_00(r) / √(4π).
    //
    // Closed forms for V_H(r):
    //   r < R :  V_H(r) = 4π ρ₀ (R²/2 − r²/6)
    //   r > R :  V_H(r) = 4π ρ₀ R³ / (3 r)
    //
    // Hence V_00(r) = √(4π) · V_H(r).
    long n = 1001;
    double r_max = 5.0;
    double R     = 2.0;        // sphere radius
    double rho0  = 0.5;        // bulk density
    uniform_mesh m(n, r_max);

    // Build ρ_00(r) = ρ₀ √(4π) for r < R, 0 otherwise.
    nda::array<double, 1> rho_00(n);
    double sqrt4pi = std::sqrt(4.0 * M_PI);
    for (long i = 0; i < n; ++i) {
        rho_00(i) = (m.r(i) < R) ? rho0 * sqrt4pi : 0.0;
    }

    nda::array<double, 1> V_00(n);
    hamilt::paw::radial_hartree_multipole(rho_00, m.r, m.rab, /*L=*/0, V_00);

    // Convert to V_H(r) = V_00(r) / sqrt(4π) and compare against the
    // analytic uniform-sphere potential. The step-discontinuity in ρ
    // at r=R + the trapezoidal cumulative integrator → an O(dr/R)
    // global bias in V_H proportional to the local charge mass moved
    // by the half-step at the boundary; ~0.3% relative on n=1001. We
    // therefore compare in *relative* error and skip the boundary kink.
    double V_scale = 4.0 * M_PI * rho0 * 0.5 * R * R;   // V_H(0)
    for (long i = 0; i < n; ++i) {
        double V_H = V_00(i) / sqrt4pi;
        double r_i = m.r(i);
        double V_ref = (r_i < R)
            ? 4.0 * M_PI * rho0 * (0.5 * R * R - r_i * r_i / 6.0)
            : 4.0 * M_PI * rho0 * R * R * R / (3.0 * std::max(r_i, 1e-30));
        // Skip a small region around the boundary kink (discontinuity in
        // ρ → 1st-order error in V_H from the trapezoidal cumulative).
        if (std::abs(r_i - R) < 0.05) continue;
        double rel = std::abs(V_H - V_ref) / V_scale;
        if (!(rel < 5e-3)) {
            INFO("r=" << r_i << " V_H=" << V_H
                 << " V_ref=" << V_ref << " rel=" << rel);
            REQUIRE(rel < 5e-3);
        }
    }
}

TEST_CASE("paw_radial_hartree_higher_L_for_spherical_rho", "[hamilt][paw][radial]")
{
    // For a spherically symmetric ρ, only ρ_00 is non-zero. If we feed
    // ρ_00 into the multipole solver but with L ≥ 1, we are effectively
    // asking V_L for the wrong moment of a charge that has no L-th
    // moment. The integrals don't vanish identically (the integrand
    // r'^{L+2} weights any non-zero ρ at large r'), but this regression
    // sanity-check just confirms that V_L(0) = 0 for L ≥ 1 (boundary
    // condition handled correctly).
    long n = 401;
    double r_max = 4.0;
    uniform_mesh m(n, r_max);
    nda::array<double, 1> rho_L(n);
    for (long i = 0; i < n; ++i)
        rho_L(i) = (m.r(i) < 1.5) ? 1.0 : 0.0;

    for (int L : {1, 2, 3}) {
        nda::array<double, 1> V_L(n);
        hamilt::paw::radial_hartree_multipole(rho_L, m.r, m.rab, L, V_L);
        REQUIRE(std::abs(V_L(0)) < 1e-14);
    }
}

namespace {

// Independent real spherical harmonics in the QE ylmr2 convention
// (Condon–Shortley P_l^m, within-L order m = 0, +1, −1, +2, −2, …), built
// on boost::math::spherical_harmonic — implementation-independent of the
// QE-ported recursion in qe_real_ylm_flat:
//   Y_l0   =    Y_l^0,   Y_l,+m = √2·Re Y_l^m,   Y_l,−m = √2·Im Y_l^m .
inline double ref_real_ylm_qe_slot(int l, int slot /*0..2l*/,
                                   double theta, double phi)
{
    using boost::math::spherical_harmonic_i;
    using boost::math::spherical_harmonic_r;
    if (slot == 0) return spherical_harmonic_r(l, 0, theta, phi);
    int m = (slot + 1) / 2;
    if (slot % 2 == 1)
        return std::sqrt(2.0) * spherical_harmonic_r(l, m, theta, phi);
    return std::sqrt(2.0) * spherical_harmonic_i(l, m, theta, phi);
}

// Gauss–Legendre nodes/weights on [−1, 1] (Newton on the recurrence).
inline void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w)
{
    x.assign(n, 0.0);
    w.assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double xi = std::cos(M_PI * (i + 0.75) / (n + 0.5));
        double p1 = 0.0, p0 = 0.0, dp = 0.0;
        for (int it = 0; it < 100; ++it) {
            p0 = 1.0;
            p1 = xi;
            for (int k = 2; k <= n; ++k) {
                double p2 = ((2 * k - 1) * xi * p1 - (k - 1) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            dp = n * (xi * p1 - p0) / (xi * xi - 1.0);
            double dx = p1 / dp;
            xi -= dx;
            if (std::abs(dx) < 1e-15) break;
        }
        x[i] = xi;
        w[i] = 2.0 / ((1.0 - xi * xi) * dp * dp);
    }
}

} // namespace

/**
 * Synthetic l=3 (f-projector) augmentation test — plan D3. The LaNiO3-class
 * path (projector pairs up to L = 2·l = 6) has no fixture coverage on this
 * host; every l-dependent ingredient of the becsum/Q(G) augmentation chain
 * is validated here against implementation-independent references:
 *
 *   A) qe_real_ylm_flat recursion at Lmax=6 vs boost spherical harmonics
 *      (also pins the QE ylmr2 Condon–Shortley convention — the 3956b45
 *      real_ylm bug class);
 *   B) aainit_tables_build(lli=4) Gaunt-like ap coefficients (random-point
 *      collocation + matrix inversion, QE aainit port) vs direct angular
 *      quadrature ∫ Y_lp·Y_li·Y_lj dΩ;
 *   C) build_qrad_tab cubic interpolation vs the exact Bessel transform on
 *      a synthetic species with s and f projectors (L rows 0..6);
 *   D) evaluate_Q_IJ_at_K (full qvan2 chain: nhtolm/indv maps, lp→L, (−i)^L,
 *      Ylm, radial table) vs an independently assembled reference for s⊗s,
 *      s⊗f and f⊗f pairs at ~20 K vectors.
 */
TEST_CASE("paw_q_eval_synthetic_l3", "[hamilt][paw][radial]")
{
    using hamilt::paw::qe_real_ylm_flat;
    constexpr int lli    = 4;                        // s..f projectors
    constexpr int Lmax   = 2 * (lli - 1);            // 6
    constexpr int llx    = (Lmax + 1) * (Lmax + 1);  // 49
    constexpr int npairs = lli * lli;                // 16

    std::mt19937 gen(20260725);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    // ---- A) Ylm recursion at Lmax=6 vs boost reference ----
    double maxdevA = 0.0;
    {
        nda::array<double, 1> Yflat(llx);
        for (int s = 0; s < 200; ++s) {
            double v0 = uni(gen), v1 = uni(gen), v2 = uni(gen);
            double n2 = v0 * v0 + v1 * v1 + v2 * v2;
            if (n2 < 1e-3) continue;
            double nn = std::sqrt(n2);
            std::array<double, 3> dir{v0 / nn, v1 / nn, v2 / nn};
            qe_real_ylm_flat(Lmax, dir, Yflat);
            double theta = std::acos(std::clamp(dir[2], -1.0, 1.0));
            double phi   = std::atan2(dir[1], dir[0]);
            for (int l = 0; l <= Lmax; ++l)
                for (int k = 0; k <= 2 * l; ++k)
                    maxdevA = std::max(maxdevA,
                        std::abs(Yflat(l * l + k)
                                 - ref_real_ylm_qe_slot(l, k, theta, phi)));
        }
        CHECK(maxdevA < 1e-11);
    }

    // ---- B) aainit ap tables vs direct angular quadrature ----
    auto aatab = hamilt::paw::aainit_tables_build(lli);
    REQUIRE(aatab.llx == llx);
    std::vector<double> xg, wg;
    gauss_legendre(24, xg, wg);          // exact for cosθ-degree ≤ 47
    const int Mphi = 64;                 // exact for trig degree < 64
    const long npts = (long)xg.size() * Mphi;
    nda::array<double, 2> Ygrid(npts, llx);
    nda::array<double, 1> wgrid(npts);
    {
        long p = 0;
        for (std::size_t it = 0; it < xg.size(); ++it) {
            double theta = std::acos(xg[it]);
            for (int ip = 0; ip < Mphi; ++ip, ++p) {
                double phi = 2.0 * M_PI * ip / Mphi;
                for (int l = 0; l <= Lmax; ++l)
                    for (int k = 0; k <= 2 * l; ++k)
                        Ygrid(p, l * l + k) =
                            ref_real_ylm_qe_slot(l, k, theta, phi);
                wgrid(p) = wg[it] * (2.0 * M_PI / Mphi);
            }
        }
    }
    double maxdevB = 0.0;
    for (int li = 0; li < npairs; ++li)
        for (int lj = 0; lj < npairs; ++lj)
            for (int lp = 0; lp < llx; ++lp) {
                double ref = 0.0;
                for (long p = 0; p < npts; ++p)
                    ref += wgrid(p) * Ygrid(p, lp) * Ygrid(p, li) * Ygrid(p, lj);
                maxdevB = std::max(maxdevB,
                                   std::abs(aatab.ap(lp, li, lj) - ref));
            }
    CHECK(maxdevB < 1e-9);

    // ---- Synthetic species: s (l=0) + f (l=3) projectors ----
    const long mesh = 2001;              // odd (Simpson-exact path)
    const double rmax = 2.0;
    uniform_mesh M(mesh, rmax);
    nda::array<double, 3> qf_backing(2 * 3 + 1, 3, mesh);
    qf_backing() = 0.0;
    auto fill = [&](int L, int ijv, double alpha, double scl) {
        for (long i = 0; i < mesh; ++i) {
            double r = M.r(i);
            // q^L(r)·r² with q^L ~ r^L at the origin (regularity), compact
            // support numerically enforced by the Gaussian at r_max = 2.
            qf_backing(L, ijv, i) =
                scl * std::pow(r, L + 2) * std::exp(-alpha * r * r);
        }
    };
    fill(0, 0, 8.0, 1.0);                              // s⊗s : L = 0
    fill(3, 1, 6.0, 0.7);                              // s⊗f : L = 3
    fill(0, 2, 5.0, 0.9);                              // f⊗f : L = 0,2,4,6
    fill(2, 2, 5.0, -0.6);
    fill(4, 2, 5.0, 0.8);
    fill(6, 2, 5.0, -0.5);

    hamilt::pseudopot::species_paw_t sp;
    sp.is_paw = true;
    sp.mesh   = (int)mesh;
    sp.nbeta  = 2;
    sp.kkbeta = (int)mesh;
    sp.nh     = 8;
    sp.lmax_aug = 6;
    sp.r   = M.r;
    sp.rab = M.rab;
    sp.qfuncl.rebind(qf_backing());
    sp.lll    = nda::array<int, 1>{0, 3};
    sp.nhtol  = nda::array<int, 1>{0, 3, 3, 3, 3, 3, 3, 3};
    // QE flat lm (1-based): l² + k + 1 with within-L order m=0,+1,−1,…
    sp.nhtolm = nda::array<int, 1>{1, 10, 11, 12, 13, 14, 15, 16};
    sp.indv   = nda::array<int, 1>{1, 2, 2, 2, 2, 2, 2, 2};

    // ---- C) qrad table interpolation vs exact transform (L rows to 6) ----
    const double Kmax = 20.0;
    auto T = hamilt::paw::build_qrad_tab(sp, Kmax);
    double maxdevC = 0.0, maxvalC = 0.0;
    for (int iK = 0; iK < 300; ++iK) {
        double K = (iK + 0.41) * (Kmax / 300.0);
        for (int ijv = 0; ijv < 3; ++ijv) {
            auto ex = hamilt::paw::qrad_at_K(sp, ijv, K);
            auto tb = hamilt::paw::qrad_interp_at_K(T, ijv, K);
            for (long L = 0; L < ex.extent(0); ++L) {
                maxdevC = std::max(maxdevC, std::abs(tb(L) - ex(L)));
                maxvalC = std::max(maxvalC, std::abs(ex(L)));
            }
        }
    }
    CHECK(maxdevC < 1e-7 * std::max(1.0, maxvalC));

    // ---- D) full Q_IJ(K) vs independently assembled reference ----
    const double omega = 100.0;
    // Independent Simpson radial transform (uniform mesh, odd N).
    auto Jref = [&](int L, int ijv, double K) {
        auto jl = [&](double r) {
            double Kr = K * r;
            return (std::abs(Kr) < 1e-30)
                       ? (L == 0 ? 1.0 : 0.0)
                       : boost::math::sph_bessel((unsigned)L, Kr);
        };
        double F = qf_backing(L, ijv, 0) * jl(M.r(0))
                 + qf_backing(L, ijv, mesh - 1) * jl(M.r(mesh - 1));
        for (long i = 1; i < mesh - 1; ++i)
            F += ((i % 2 == 1) ? 4.0 : 2.0) * qf_backing(L, ijv, i) * jl(M.r(i));
        return F * M.dr / 3.0;
    };
    std::vector<std::pair<int, int>> pairs =
        {{0, 0}, {0, 1}, {0, 7}, {1, 1}, {2, 5}, {4, 4}, {7, 7}, {1, 4}};
    double maxdevD = 0.0, maxvalD = 0.0;
    for (int s = 0; s < 20; ++s) {
        double v0 = uni(gen), v1 = uni(gen), v2 = uni(gen);
        double n2 = v0 * v0 + v1 * v1 + v2 * v2;
        if (n2 < 1e-3) continue;
        double nn = std::sqrt(n2);
        double Kmag = 0.2 + 0.9 * s;     // 0.2 .. 17.3 < Kmax
        std::array<double, 3> Kv{Kmag * v0 / nn, Kmag * v1 / nn, Kmag * v2 / nn};
        double theta = std::acos(std::clamp(v2 / nn, -1.0, 1.0));
        double phi   = std::atan2(v1 / nn, v0 / nn);
        for (auto [ih, jh] : pairs) {
            auto Qcode = hamilt::paw::evaluate_Q_IJ_at_K(sp, aatab, ih, jh,
                                                         Kv, omega);
            int ivl = sp.nhtolm(ih) - 1, jvl = sp.nhtolm(jh) - 1;
            int nb = sp.indv(ih) - 1, mb = sp.indv(jh) - 1;
            int n1 = std::max(nb, mb), n2i = std::min(nb, mb);
            int ijv = n1 * (n1 + 1) / 2 + n2i;
            std::complex<double> Qref(0.0, 0.0);
            for (int lp = 0; lp < llx; ++lp) {
                double ap = 0.0;
                for (long p = 0; p < npts; ++p)
                    ap += wgrid(p) * Ygrid(p, lp) * Ygrid(p, ivl) * Ygrid(p, jvl);
                if (std::abs(ap) < 1e-12) continue;
                int L = (int)std::floor(std::sqrt((double)lp + 1e-9));
                std::complex<double> miL(1.0, 0.0);
                for (int t = 0; t < L; ++t) miL *= std::complex<double>(0.0, -1.0);
                double Ylp = ref_real_ylm_qe_slot(L, lp - L * L, theta, phi);
                Qref += miL * (ap * Ylp * Jref(L, ijv, Kmag));
            }
            Qref *= (4.0 * M_PI) / omega;
            maxdevD = std::max(maxdevD,
                std::abs(Qcode - ComplexType(Qref.real(), Qref.imag())));
            maxvalD = std::max(maxvalD, std::abs(Qcode));
        }
    }
    app_log(1, "[synthetic l=3] Ylm dev {:.2e}; ap dev {:.2e}; qrad-interp "
               "dev {:.2e}; Q(K) dev {:.2e} (max|Q| {:.2e})",
            maxdevA, maxdevB, maxdevC, maxdevD, maxvalD);
    REQUIRE(maxvalD > 0.0);
    CHECK(maxdevD < 5e-7 * maxvalD);
}
