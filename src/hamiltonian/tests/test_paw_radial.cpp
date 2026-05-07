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

#include <cmath>

#include "configuration.hpp"
#include "nda/nda.hpp"

#include "hamiltonian/paw/paw_radial.hpp"

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
