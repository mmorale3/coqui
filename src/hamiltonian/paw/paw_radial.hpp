/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Radial-grid scaffolding for the PAW one-center solver:
 *
 *   - `radial_simpson(f, rab)` — composite Simpson's rule on a logarithmic
 *     mesh, accepting QE's `rab` (dr/di) measure as the per-point weight.
 *
 *   - `radial_hartree_multipole(rho_L, r, rab, L, V_L)` — radial Hartree
 *     potential for a single multipole channel L:
 *
 *       V_L(r) = (4π / (2L+1)) · [
 *                  (1 / r^{L+1}) ∫₀^r ρ_L(r') r'^{L+2} dr'
 *                  +  r^L         ∫_r^∞ ρ_L(r') r'^{1-L} dr'
 *                ]
 *
 *     This is the closed-form solution of the radial Poisson equation
 *     after the angular Y_LM decomposition; computed in O(N) via two
 *     cumulative integrals (one outward, one inward).
 *
 *   - `radial_kinetic_matel(...)` — ⟨ψ_I | -½∇² | ψ_J⟩ on the radial mesh
 *     for fixed angular momentum l (uses the centrifugal term l(l+1)/r²).
 *     Used by paw_init_keeq to assemble the kinetic part of the PAW
 *     static D matrix.
 *
 * All routines work on QE's radial mesh: r(:), rab(:), mesh points indexed
 * 0..mesh-1. Conventions match QE's `Modules/radial_grids.f90` and
 * `Modules/simpsn.f90` (composite Simpson over the index variable, with
 * rab(i) folded in as dr/di).
 * ==========================================================================
 */
#ifndef HAMILTONIAN_PAW_PAW_RADIAL_HPP
#define HAMILTONIAN_PAW_PAW_RADIAL_HPP

#include <cmath>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"

namespace hamilt::paw {

/**
 * Composite Simpson's rule over a 1D radial integrand,
 *   ∫ f(r) dr  ≈  Σ_i w_i · f(r_i) · rab(i)
 * where w_i are the Simpson weights {1, 4, 2, 4, 2, ..., 4, 1}/3.
 *
 * If `n = f.size()` is even, the last subinterval is folded in via the
 * trapezoidal rule (Simpson's 3/8 would change the cumulative-integral
 * complexity downstream; the trap fallback at one endpoint is what QE
 * does in `simpson_2`). Accuracy is O(dx^4) on the bulk, O(dx^3) at the
 * tail when n is even — fine for our radial meshes (n is typically odd
 * by convention but this guards either way).
 */
inline double radial_simpson(nda::ArrayOfRank<1> auto const& f,
                             nda::ArrayOfRank<1> auto const& rab)
{
    long n = f.size();
    utils::check(rab.size() == n,
                 "radial_simpson: f.size ({}) != rab.size ({})", n, rab.size());
    if (n == 0) return 0.0;
    if (n == 1) return 0.0;       // single sample — degenerate

    double s = 0.0;
    long n_simp = (n % 2 == 1) ? n : (n - 1);
    // Composite Simpson over [0, n_simp-1]
    s += f(0) * rab(0);
    s += f(n_simp - 1) * rab(n_simp - 1);
    for (long i = 1; i < n_simp - 1; ++i) {
        double w = (i % 2 == 1) ? 4.0 : 2.0;
        s += w * f(i) * rab(i);
    }
    s /= 3.0;
    // If n is even, append trapezoidal contribution on the last interval.
    if (n_simp != n) {
        s += 0.5 * (f(n_simp - 1) * rab(n_simp - 1) + f(n - 1) * rab(n - 1));
    }
    return s;
}

/**
 * Radial Hartree potential V_L(r) for a single multipole channel L,
 *   V_L(r) = (4π / (2L+1)) · [
 *              r^{-(L+1)} I_in(r)   +   r^L · I_out(r)
 *            ]
 * with
 *   I_in(r)  = ∫₀^r  ρ_L(r') r'^{L+2} dr'
 *   I_out(r) = ∫_r^∞ ρ_L(r') r'^{1-L} dr'
 *
 * Implementation: two passes over the mesh, one accumulating I_in
 * outward and one accumulating I_out inward. Trapezoidal sub-step
 * (the Simpson cumulative is messier and the leading error is the
 * same on logarithmic meshes — QE uses trap here too).
 *
 * For r=0: V_L(0) is finite for L=0 (= 4π·∫ ρ₀ r' dr' = sum of charges /
 * radius limit). For L≥1, V_L(0) = 0 by the boundary condition. We set
 * V_L(0) explicitly via the L=0 special case; for L≥1 we rely on
 * `r^L → 0` and the integrand `r'^{1-L}` integrable around 0 for ρ_L
 * regular enough at the origin (which it is for physical pair densities).
 */
inline void radial_hartree_multipole(
    nda::ArrayOfRank<1> auto const& rho_L,
    nda::ArrayOfRank<1> auto const& r,
    nda::ArrayOfRank<1> auto const& rab,
    int L,
    nda::ArrayOfRank<1> auto& V_L)
{
    long n = rho_L.size();
    utils::check(r.size() == n && rab.size() == n && V_L.size() == n,
                 "radial_hartree_multipole: shape mismatch (rho={}, r={}, rab={}, V={})",
                 n, r.size(), rab.size(), V_L.size());
    if (n == 0) return;

    // I_in(r_i) = ∫₀^{r_i} ρ_L r'^{L+2} dr'   (cumulative, trap rule on rab)
    nda::array<double, 1> I_in(n);
    I_in(0) = 0.0;
    for (long i = 1; i < n; ++i) {
        double r_im1 = r(i - 1), r_i = r(i);
        double pim1 = rho_L(i - 1) * std::pow(r_im1, L + 2);
        double pi   = rho_L(i)     * std::pow(r_i,   L + 2);
        // trapezoid in the index variable, weighted by rab average
        I_in(i) = I_in(i - 1) + 0.5 * (pim1 * rab(i - 1) + pi * rab(i));
    }

    // I_out(r_i) = ∫_{r_i}^∞ ρ_L r'^{1-L} dr'
    nda::array<double, 1> I_out(n);
    I_out(n - 1) = 0.0;
    for (long i = n - 2; i >= 0; --i) {
        double r_ip1 = r(i + 1), r_i = r(i);
        // For L=0 the integrand r'^{1-0} = r' is finite. For L≥1,
        // r'^{1-L} is regular for r > 0 (which all radial-mesh points are).
        double pip1 = rho_L(i + 1) * std::pow(r_ip1, 1 - L);
        double pi   = rho_L(i)     * std::pow(r_i,   1 - L);
        I_out(i) = I_out(i + 1) + 0.5 * (pip1 * rab(i + 1) + pi * rab(i));
    }

    double pref = 4.0 * M_PI / (2.0 * L + 1.0);
    for (long i = 0; i < n; ++i) {
        // V_L(r_i). For r=0 with L=0 the inner term vanishes and the
        // outer term limits to 4π · ∫ ρ₀ r' dr'. For L>0, r=0 yields 0.
        if (r(i) <= 0.0) {
            V_L(i) = (L == 0) ? pref * I_out(i) : 0.0;
        } else {
            double inner = I_in(i)  / std::pow(r(i), L + 1);
            double outer = std::pow(r(i), L) * I_out(i);
            V_L(i) = pref * (inner + outer);
        }
    }
}

/**
 * Kinetic-energy matrix element ⟨ψ_I | -½∇² | ψ_J⟩ on the radial mesh
 * for a fixed angular-momentum quantum number `l` (channels share l;
 * for I, J in the same beta-channel the l matches).
 *
 *   ⟨I | T | J⟩ = ½ ∫₀^∞ [ ψ_I'(r) · ψ_J'(r)  +  l(l+1)/r² · ψ_I(r) · ψ_J(r) ] dr
 *
 * Convention: `ψ` here is the QE radial partial wave (`aewfc(I, r)` or
 * `pswfc(I, r)`); QE stores u(r) = r·R(r) so the radial integrand of
 * ⟨I | T | J⟩ is (½) [ u_I' u_J' + l(l+1)/r² u_I u_J ] dr without the
 * extra r² Jacobian. We use the index-variable derivative (ψ' on the
 * grid) — finite difference is sufficient at the precision we need
 * (5-point stencil internally; endpoints handled with one-sided).
 *
 * Returns the integral as a scalar.
 */
inline double radial_kinetic_matel(
    nda::ArrayOfRank<1> auto const& psi_I,
    nda::ArrayOfRank<1> auto const& psi_J,
    nda::ArrayOfRank<1> auto const& r,
    nda::ArrayOfRank<1> auto const& rab,
    int l)
{
    long n = psi_I.size();
    utils::check(psi_J.size() == n && r.size() == n && rab.size() == n,
                 "radial_kinetic_matel: shape mismatch");
    if (n < 5) return 0.0;

    nda::array<double, 1> dpsi_I(n);
    nda::array<double, 1> dpsi_J(n);
    // d/dr ψ(r). On a non-uniform mesh, dψ/dr = (dψ/di) · (di/dr) =
    // (dψ/di) / rab. We compute dψ/di by central differences (5-point at
    // interior, 3-point at endpoints).
    auto deriv_di = [&](auto const& f, long i) {
        if (i == 0)       return (-3.0*f(0) + 4.0*f(1) - f(2)) / 2.0;
        if (i == n - 1)   return ( 3.0*f(n-1) - 4.0*f(n-2) + f(n-3)) / 2.0;
        if (i == 1 || i == n - 2)
                          return (f(i + 1) - f(i - 1)) / 2.0;
        return (-f(i + 2) + 8.0*f(i + 1) - 8.0*f(i - 1) + f(i - 2)) / 12.0;
    };
    for (long i = 0; i < n; ++i) {
        dpsi_I(i) = deriv_di(psi_I, i) / rab(i);
        dpsi_J(i) = deriv_di(psi_J, i) / rab(i);
    }

    // Build the integrand and integrate.
    nda::array<double, 1> integrand(n);
    double cf = (double)l * (double)(l + 1);
    for (long i = 0; i < n; ++i) {
        double cent = (r(i) > 0.0) ? cf / (r(i) * r(i)) * psi_I(i) * psi_J(i)
                                   : 0.0;
        integrand(i) = 0.5 * (dpsi_I(i) * dpsi_J(i) + cent);
    }
    return radial_simpson(integrand, rab);
}

} // namespace hamilt::paw

#endif // HAMILTONIAN_PAW_PAW_RADIAL_HPP
