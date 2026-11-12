"""Closing the T = 0 loop with the Dyson equation on the line instead of the Cayley closure (g_repr = "line_dyson", gapped only).

Given Sigma_c(k, z) = sum_l g_l/(z - w_l) (the sector fits of the driver, mu-relative) and H = H_stat - mu:
    G(z; dmu) = [z - (H - dmu) - Sigma_c(z)]^-1                 on the two upper rays through 0, G(conj z) = G(z)^dagger.
Frame: as the Matsubara Dyson equation of the imaginary-axis code, Sigma_c stays FIXED RELATIVE TO mu and only H moves with dmu, so
Sigma_c is used only on the line it was computed on. The other frame (Sigma_c fixed in absolute energy, the line moved to the new mu)
evaluates the signed fit near the real axis, where it is not determined. The fixed point is the same (dmu -> 0).

The next GW iteration needs G^>, G^< (one-sided pole sums, each on its own time ray) and D. All three follow from G on the line,
oriented along e^{i ang} through 0, because G is analytic there and G ~ 1/z:
    P_ang(zt) = (1/2 pi i) int_L G(z)/(z - zt) dz    (closing on the side of the hole axis)
    contour ang = theta       : zt on rho_- lies on its left   -> G^>(zt) = P, G^<(zt) = G(zt) - P
    contour ang = pi - theta  : zt on rho_+ lies on its right  -> G^<(zt) = -P, G^>(zt) = G(zt) + P
    D = sum_{e_m < 0} c_m = (1/2 pi i) int_L G(z) dz + 1/2
No poles of G enter; the signed one-sided fits of G^> and G^< at the nodes then play the role of the DLR transform to the time rays.
mu comes from the plateau of the step function N(dmu) = 2 sum_k w_k Tr D_k(dmu) around N_el."""
import numpy as np


def line_quadrature(smin=1e-5, smax=1e9, per_efold=3, nn=16):
    """Composite Gauss-Legendre on [0, smax]: a first panel [0, smin], then log panels. Returns (s, ds)."""
    npan = int(np.ceil(per_efold * np.log(smax / smin)))
    edges = np.concatenate([[0.0], np.exp(np.linspace(np.log(smin), np.log(smax), npan + 1))])
    x, w = np.polynomial.legendre.leggauss(nn)
    a, b = edges[:-1, None], edges[1:, None]
    return ((b - a) / 2 * x + (a + b) / 2).ravel(), ((b - a) / 2 * w).ravel()


def sigma_at(w, g, z):
    """Sigma(z) = sum_l g_l/(z - w_l), (nz, nb, nb)."""
    K = 1.0 / (z[:, None] - w[None, :])
    nb = g.shape[1]
    return (K @ g.reshape(len(w), -1)).reshape(len(z), nb, nb)


def dyson_at(H, Sig, z):
    """G(z) = [z - H - Sig]^-1 for Sig (nz, nb, nb) at z, (nz, nb, nb)."""
    nb = H.shape[0]
    return np.linalg.inv(z[:, None, None] * np.eye(nb) - H - Sig)


def _dag(A):
    return np.conj(np.transpose(A, (0, 2, 1)))


class LineDyson:
    """Dyson G of one k on the line through mu; dmu shifts H only (module docstring). Sigma_c at the quadrature points is cached."""

    def __init__(self, H, w, g, theta, quad=None):
        self.H, self.w, self.g, self.theta = H, w, g, theta
        self.nb = H.shape[0]
        self.s, self.ds = line_quadrature() if quad is None else quad
        self.smax = float(self.s[-1] + self.ds[-1])                      # end of the last panel, to the width of its last node
        self.zp, self.zm = self.s * np.exp(1j * theta), self.s * np.exp(1j * (np.pi - theta))
        self.Sp, self.Sm = sigma_at(w, g, self.zp), sigma_at(w, g, self.zm)

    def _rays(self, dmu):
        """G on rho_+ and rho_- at the quadrature points."""
        H = self.H - dmu * np.eye(self.nb)
        return dyson_at(H, self.Sp, self.zp), dyson_at(H, self.Sm, self.zm)

    def density(self, dmu=0.0):
        """D(dmu): hole weight, line integral along ang = theta (Hermitian part). Lower arm: G(-s e^{i theta}) = G(rho_-)^dagger."""
        Gp, Gm = self._rays(dmu)
        I = np.einsum('z,zij->ij', self.ds, Gp + _dag(Gm)) * np.exp(1j * self.theta) / (2j * np.pi)
        D = I + 0.5 * np.eye(self.nb)
        return 0.5 * (D + D.conj().T)

    def _cauchy(self, zt, up, Gup, lo, Glo, ang):
        """P_ang(zt) for targets zt (nt,) with the arms up / lo, (nt, nb, nb)."""
        dz = np.exp(1j * ang) * self.ds
        Ku = dz[None, :] / (up[None, :] - zt[:, None])
        Kl = dz[None, :] / (lo[None, :] - zt[:, None])
        P = (Ku @ Gup.reshape(len(up), -1) + Kl @ Glo.reshape(len(lo), -1)).reshape(len(zt), self.nb, self.nb) / (2j * np.pi)
        # tail beyond smax on both arms, G ~ 1/z: int dz/z^2 = 2 e^{-i ang}/smax
        return P + np.eye(self.nb) * (2.0 * np.exp(-1j * ang) / self.smax) / (2j * np.pi)

    def sectors(self, zeta, dmu=0.0):
        """(G^>, G^<) at nodes zeta (both upper rays, arg zeta = theta or pi - theta). Poles above / below 0 of H - dmu + Sigma_c."""
        Gp, Gm = self._rays(dmu)
        Gt = dyson_at(self.H - dmu * np.eye(self.nb), sigma_at(self.w, self.g, zeta), zeta)
        right = zeta.real > 0                                             # rho_+: right half of the upper plane
        Gpart, Ghole = np.zeros_like(Gt), np.zeros_like(Gt)
        if right.any():                                                    # ang = pi - theta: up = rho_-, lo = conj rho_+
            P = self._cauchy(zeta[right], self.zm, Gm, np.conj(self.zp), _dag(Gp), np.pi - self.theta)
            Ghole[right] = -P
            Gpart[right] = Gt[right] + P
        if (~right).any():                                                 # ang = theta: up = rho_+, lo = conj rho_-
            P = self._cauchy(zeta[~right], self.zp, Gp, np.conj(self.zm), _dag(Gm), self.theta)
            Gpart[~right] = P
            Ghole[~right] = Gt[~right] - P
        return Gpart, Ghole


class LineDysonFailure(RuntimeError):
    """This iteration cannot be closed by line Dyson; the driver does it by the closure (unless ld_strict)."""


class NoGapError(LineDysonFailure):
    """No plateau of N(dmu) at N_el wider than min_gap: the G of this Sigma_c (aligned with mu) is metallic or nearly so."""


class GFitError(LineDysonFailure):
    """The signed one-sided fits of G^> / G^< at the nodes miss the Cauchy-split values by more than ld_fit_tol. Seen on Mg2Si
    (QP gap 0.29 eV) with an ungapped G basis and Sigma basis gap 0.002 Ha: residual 2.5e-6 -> 2.4e-5 -> 0.2 and max|c| 2e3 -> 5e8 over
    three iterations; Si (1.1 eV) stays at 9e-9."""


def electron_count(lds, dmu, k_weight):
    """N(dmu) = 2 sum_k w_k Tr D_k(dmu)."""
    return float(sum(2.0 * wk * np.trace(ld.density(dmu)).real for ld, wk in zip(lds, k_weight)))


def _bisect_step(lds, target, k_weight, lo, hi, tol):
    """dmu with N(dmu) = target (N increasing in dmu), to width tol."""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if electron_count(lds, mid, k_weight) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def chemical_potential_line(lds, nelec, k_weight, dn=0.05, tol=1e-6, bracket=2.0, min_gap=1e-3):
    """T = 0, gapped: N(dmu) is a step function with a plateau at N_el. The plateau edges are where N crosses N_el -+ dn; dn is
    well below the smallest QP step 2 Z w_k and above the integration error. Returns (mid-plateau dmu, lower edge, upper edge,
    N(mid)). The plateau width is the gap of H + Sigma_c(0), NOT the QP gap (no Z factor): the QP edges need the poles of G. N is
    only nearly flat on the plateau for a fixed Sigma_c (the QP weights change with dmu); at the fixed point N = N_el at dmu = 0.
    Raises if there is no plateau at N_el (a metal or a closed gap). A pole crossing 0 makes N jump within a few tol, which looks
    like a plateau of width ~tol, so the plateau must exceed min_gap (Ha): below ~1e-3 Ha the one-sided G fits fail anyway (edge
    poles closer to mu than the lowest fermionic node; notes/paper app. line Dyson)."""
    lo, hi = -bracket, bracket
    if electron_count(lds, lo, k_weight) >= nelec - dn or electron_count(lds, hi, k_weight) <= nelec + dn:
        raise NoGapError("chemical_potential_line: N_el not bracketed")
    e_h = _bisect_step(lds, nelec - dn, k_weight, lo, hi, tol)
    e_l = _bisect_step(lds, nelec + dn, k_weight, e_h, hi, tol)
    if e_l - e_h <= max(4 * tol, min_gap):
        raise NoGapError(f"chemical_potential_line: no gap at N_el (plateau {e_l - e_h:.2e} Ha < min_gap {min_gap:g}; edges {e_h}, {e_l})")
    mid = 0.5 * (e_h + e_l)
    return mid, e_h, e_l, electron_count(lds, mid, k_weight)
