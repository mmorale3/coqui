/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Radial-to-G transform of the PAW atomic auxiliary functions η_{a,λ}.
 *
 * For a radial function f(r) attached to angular indices (L, M), centered at
 * atom position τ_a, the (q+G) component of its lattice-summed Bloch form is
 *
 *   η_{a,λ}^q(G)
 *     = ∫_cell d³s e^{-i(q+G)·s} Σ_R η_{a,λ}(s - R - τ_a)
 *     = e^{-i(q+G)·τ_a}   ·   4π (-i)^L Y_{LM}(q̂+Ĝ)
 *                             · ∫₀^∞ r² f_λ(r) j_L(|q+G| r) dr
 *
 * (.tex Eq. local-bloch-eta combined with the standard radial Fourier
 * transform of an angular-momentum-resolved function.)
 *
 * This file gives a focused helper that returns the complex value of
 * η_{a,λ}^q(G) for a single (a, λ, G), reusing CoQui's existing
 * sph_bessel_transform_boost and spherical_harmonics_l utilities.
 * ==========================================================================
 */
#ifndef HAMILTONIAN_PAW_ETA_TO_G_HPP
#define HAMILTONIAN_PAW_ETA_TO_G_HPP

#include <cmath>
#include <complex>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/basis_set_utilities.hpp"
#include "utilities/harmonics.h"
#include "nda/nda.hpp"

namespace hamilt::paw {

inline constexpr double kFourPi = 4.0 * M_PI;

/**
 * Radial spherical-Bessel transform restricted to a single (L, q+G):
 *   I_L(K) = ∫₀^∞ r² f_radial(r) j_L(K r) dr
 *
 * Reuses utils::sph_bessel_transform_boost; we ask for a singleton range
 * `l ∈ {L}`, a single output, and read it.
 *
 * Inputs:
 *   L         : angular momentum
 *   K         : magnitude |q+G|
 *   r         : radial mesh values, shape (mesh,)
 *   f_radial  : radial values of η_{a,λ}, shape (mesh,)
 * Returns:
 *   I_L(K), a real scalar.
 */
inline double eta_radial_integral(int L, double K,
    nda::array<double,1> const& r,
    nda::array<double,1> const& f_radial)
{
    long mesh = r.extent(0);
    utils::check(f_radial.extent(0) == mesh,
                 "eta_radial_integral: f / r mesh mismatch");
    auto fun = [&](double ri) {
        // Linear interpolation on the prescribed mesh — sph_bessel_transform_boost
        // calls the lambda at the same r-points it received, so we just
        // dispatch by index. Use a binary search via std::lower_bound to be
        // safe if the mesh isn't strictly uniform.
        // Hot path: for typical PAW radial meshes this is called O(mesh)
        // times per (L, K), and the lookup cost is overshadowed by the
        // sph_bessel evaluation inside simpson_rule_f.
        const double *rd = r.data();
        long lo = 0, hi = mesh - 1;
        if (ri <= rd[0]) return f_radial(0);
        if (ri >= rd[hi]) return f_radial(hi);
        while (hi - lo > 1) {
            long mid = (lo + hi) / 2;
            if (rd[mid] <= ri) lo = mid; else hi = mid;
        }
        double t = (ri - rd[lo]) / (rd[hi] - rd[lo]);
        return (1.0 - t) * f_radial(lo) + t * f_radial(hi);
    };

    nda::array<double,1> F(1);
    std::array<int,1> lrng = {L};
    utils::sph_bessel_transform_boost(lrng, K, r, fun, F);
    return F(0);
}

/**
 * Single-G value of η_{a,λ}^q(G) including angular factor and structure
 * factor. K_vec = q + G (cartesian), tau_a = atom position (cartesian).
 *
 * The (-i)^L factor and 4π normalization come from the standard radial
 * Fourier transform of an angular-momentum-resolved function.
 *
 * For real spherical harmonics Y_{LM} (CoQui's convention), the value
 * returned is complex through the structure factor and (-i)^L only.
 */
inline ComplexType eta_to_G_value(int L, int M,
    std::array<double,3> const& K_vec,
    nda::array<double,1> const& r,
    nda::array<double,1> const& f_radial,
    std::array<double,3> const& tau_a)
{
    double K = std::sqrt(K_vec[0]*K_vec[0] + K_vec[1]*K_vec[1] +
                         K_vec[2]*K_vec[2]);

    // Y_{LM} on the (q+G) direction. For K -> 0, Y_LM is well defined
    // by convention only at L=0 (Y_00 = 1/√(4π)); other L vanish in the
    // physically integrated combination. We special-case K==0.
    double Ylm = 0.0;
    if (K > 1e-14) {
        std::array<double,3> dir = {K_vec[0]/K, K_vec[1]/K, K_vec[2]/K};
        long Ymax = (2*L + 1);
        nda::array<double,1> Yall(Ymax);
        utils::spherical_harmonics_l<double>(L,
            nda::array<double,1>({dir[0], dir[1], dir[2]}), Yall);
        utils::check(M >= 0 && M < Ymax,
            "eta_to_G_value: M={} out of range for L={} (allowed [0,{}))",
            M, L, Ymax);
        Ylm = Yall(M);
    } else if (L == 0) {
        Ylm = 1.0 / std::sqrt(4.0 * M_PI);
    } else {
        return ComplexType(0.0); // higher-L moments vanish at G=0 with q=0
    }

    double radial = eta_radial_integral(L, K, r, f_radial);

    // (-i)^L  : powers cycle 1, -i, -1, +i, ...
    ComplexType iL_factor;
    switch (L % 4) {
        case 0: iL_factor = ComplexType( 1.0,  0.0); break;
        case 1: iL_factor = ComplexType( 0.0, -1.0); break;
        case 2: iL_factor = ComplexType(-1.0,  0.0); break;
        case 3: iL_factor = ComplexType( 0.0,  1.0); break;
    }

    // structure factor e^{-i(q+G)·τ_a}
    double phase = -(K_vec[0]*tau_a[0] + K_vec[1]*tau_a[1] + K_vec[2]*tau_a[2]);
    ComplexType sfac(std::cos(phase), std::sin(phase));

    return sfac * iL_factor * ComplexType(kFourPi * Ylm * radial, 0.0);
}

/**
 * Build a per-atom, per-(αβ,L) table of η values across a (q+G) grid using
 * the radial spherical-Bessel transform. Helper for q≠0 V_{GL}/V_{LL}
 * assembly when the cached q=0 qgm tensor is not directly applicable.
 *
 * The angular pair (L,M) is encoded by the caller — this routine is a thin
 * wrapper around eta_to_G_value over the (a, λ, g) cube with caller-supplied
 * radial functions and (L,M) labels.
 *
 * Inputs:
 *   r_a, f_a   : (mesh_a) radial mesh + radial function f_λ(r) per λ
 *   L_a, M_a   : angular labels per λ
 *   tau_a      : atom position
 *   q_cart     : q-point cartesian
 *   G_cart     : (ngm, 3)
 * Output:
 *   eta_g      : (ngm) — η^q(G)
 */
inline void compute_eta_q_grid_from_radial(
    nda::array<double,1> const& r_mesh,
    nda::array<double,1> const& f_radial,
    int L, int M,
    std::array<double,3> const& tau_cart,
    std::array<double,3> const& q_cart,
    nda::ArrayOfRank<2> auto const& G_cart,
    nda::ArrayOfRank<1> auto       & eta_g)
{
    long ngm = G_cart.extent(0);
    utils::check(G_cart.extent(1) == 3, "G_cart must be (ngm, 3)");
    utils::check(eta_g.extent(0) == ngm, "eta_g g dim mismatch");
    for (long g=0; g<ngm; ++g) {
        std::array<double,3> K = {
            q_cart[0] + G_cart(g, 0),
            q_cart[1] + G_cart(g, 1),
            q_cart[2] + G_cart(g, 2)};
        eta_g(g) = eta_to_G_value(L, M, K, r_mesh, f_radial, tau_cart);
    }
}

} // namespace hamilt::paw

#endif // HAMILTONIAN_PAW_ETA_TO_G_HPP
