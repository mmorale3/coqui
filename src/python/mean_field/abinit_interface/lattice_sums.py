"""
lattice_sums.py -- Ewald nuclear energy + CoQui madelung constant.

Two independent lattice sums consumed by abinit2coqui's /System attributes:

  * ewald_energy(rprimd, tau, zion)  ->  the ion-ion electrostatic energy of
    point charges zion at tau in a neutralizing uniform background, in
    HARTREE.  This is what pw2coqui stores as /System@nuclear_energy
    (QE `ewald(...)/e2`).

  * madelung_bdft(lattv, recv, kp_grid, fft_mesh, prec) -> exact python port
    of utils::madelung (src/utilities/madelung_utils.hpp), including its
    alpha/rcut estimation and the fixed nimg=8 real-space box.  The stored
    /System@madelung_constant is  -2 * madelung_bdft(...)  (see
    bdft_system.hpp compute-when-0 guard, which remains as a fallback).

Validation (validate_b2.py):
  ewald_energy vs QE 'ewald contribution' (si scf, -16.79894377 Ry) and vs
  /System@nuclear_energy of the checked-in QE fixtures;
  madelung_bdft vs the simple-cubic point-charge limit -2.8372974794/L
  (stored value +2.8372974794/L for a cubic cell, 1x1x1 mesh).
"""

import numpy as np
from scipy.special import erfc


# ---------------------------------------------------------------------------
# Ewald sum (Ha, e^2 = 1): ions zion at tau in a neutralizing background
# ---------------------------------------------------------------------------
def ewald_energy(rprimd, tau, zion, alpha=None, prec=1e-14):
    """Ion-ion Ewald energy in Hartree.

    rprimd : (3,3) rows a_n (Bohr);  tau : (nat,3) cartesian (Bohr);
    zion   : (nat,) ionic (valence) charges.
    E = E_G + E_R - E_self - E_bg with the standard splitting:
      E_G    = (2pi/vol) sum_{G!=0} exp(-G^2/4a^2)/G^2 |sum_i z_i e^{iG.tau_i}|^2
      E_R    = 1/2 sum'_{ijR} z_i z_j erfc(a|d_ijR|)/|d_ijR|
      E_self = a/sqrt(pi) sum_i z_i^2
      E_bg   = pi/(2 vol a^2) (sum_i z_i)^2      (neutralizing background)
    """
    rprimd = np.asarray(rprimd, float)
    tau = np.atleast_2d(np.asarray(tau, float))
    z = np.asarray(zion, float)
    vol = abs(np.linalg.det(rprimd))
    recv = 2.0 * np.pi * np.linalg.inv(rprimd).T          # rows b_n

    if alpha is None:
        # balance both sums: a few inverse cell lengths
        alpha = 2.0 / vol ** (1.0 / 3.0) * 2.0

    # --- reciprocal sum ---
    gcut2 = -4.0 * alpha * alpha * np.log(prec)           # exp(-G^2/4a^2) >= prec
    # integer bounds per direction from the dual metric
    amat = rprimd
    nmax = [int(np.ceil(np.sqrt(gcut2) * np.linalg.norm(amat[d]) / (2 * np.pi))) + 1
            for d in range(3)]
    m1, m2, m3 = (np.arange(-n, n + 1) for n in nmax)
    M = np.stack(np.meshgrid(m1, m2, m3, indexing="ij"), axis=-1).reshape(-1, 3)
    M = M[np.any(M != 0, axis=1)]
    G = M @ recv
    G2 = np.einsum("ij,ij->i", G, G)
    keep = G2 <= gcut2
    G, G2 = G[keep], G2[keep]
    S = (np.exp(1j * (G @ tau.T)) * z[None, :]).sum(axis=1)   # structure factor
    e_g = (2.0 * np.pi / vol) * np.sum(np.exp(-G2 / (4 * alpha ** 2)) / G2 * np.abs(S) ** 2)

    # --- real-space sum ---
    rcut = np.sqrt(-np.log(prec)) / alpha
    nimg = [int(np.ceil(rcut * np.linalg.norm(np.cross(rprimd[(d + 1) % 3],
                                                       rprimd[(d + 2) % 3])) / vol)) + 1
            for d in range(3)]
    i1, i2, i3 = (np.arange(-n, n + 1) for n in nimg)
    L = np.stack(np.meshgrid(i1, i2, i3, indexing="ij"), axis=-1).reshape(-1, 3) @ rprimd
    nat = tau.shape[0]
    e_r = 0.0
    for i in range(nat):
        d = tau[i][None, None, :] - tau[None, :, :] + L[:, None, :]   # (nL, nat, 3)
        dist = np.linalg.norm(d, axis=-1)
        mask = (dist > 1e-10) & (dist < rcut)
        zz = (z[i] * z[None, :]) * np.ones_like(dist)
        e_r += 0.5 * np.sum(zz[mask] * erfc(alpha * dist[mask]) / dist[mask])

    e_self = alpha / np.sqrt(np.pi) * np.sum(z ** 2)
    e_bg = np.pi / (2.0 * vol * alpha ** 2) * np.sum(z) ** 2
    return e_g + e_r - e_self - e_bg


# ---------------------------------------------------------------------------
# Exact port of utils::madelung (madelung_utils.hpp).  The h5 stores -2x this.
# ---------------------------------------------------------------------------
def _estimate_ewald_params(recv_sc, fft_mesh_sc, prec):
    """Port of utils::estimate_ewald_params (3D branch)."""
    gm = [fft_mesh_sc[d] // 2 * np.linalg.norm(recv_sc[d]) for d in range(3)]
    Gmax = min(gm)
    log_prec = np.log(prec / (4 * np.pi * (Gmax + 1e-100) ** 2))
    alpha = np.sqrt(-Gmax ** 2 / (4 * log_prec)) + 1e-100
    r_cut = np.sqrt(-np.log(prec)) / alpha
    return alpha, r_cut


def madelung_bdft(lattv, recv, kp_grid, fft_mesh, prec=1e-10):
    """utils::madelung(lattv, recv, mp_mesh, fft_mesh, prec), 3D.

    Returns e_sr + e_lr - (vlr_r0 + vsr_k0); CoQui stores
    madelung_constant = -2 * madelung_bdft(...) (Ha).
    """
    lattv = np.asarray(lattv, float)
    recv = np.asarray(recv, float)
    kp = np.asarray(kp_grid, int)
    mesh = np.asarray(fft_mesh, int)

    lattv_sc = lattv * kp[:, None]
    recv_sc = recv / kp[:, None]
    mesh_sc = mesh * kp
    volume = abs(np.linalg.det(lattv_sc))
    alpha, _ = _estimate_ewald_params(recv_sc, mesh_sc, prec)

    vlr_r0 = 0.5 * (2 * alpha) / np.sqrt(np.pi)
    vsr_k0 = 0.5 * np.pi / (volume * alpha * alpha)

    # long range: G over the fft box [-n/2, n/2)
    r1, r2, r3 = (np.arange(-(int(n) // 2), int(n) // 2) for n in mesh_sc)
    M = np.stack(np.meshgrid(r1, r2, r3, indexing="ij"), axis=-1).reshape(-1, 3)
    M = M[np.any(M != 0, axis=1)]
    G = M @ recv_sc
    G2 = np.einsum("ij,ij->i", G, G)
    e_lr = np.sum(0.5 * (4 * np.pi / G2) * np.exp(-G2 / (4 * alpha * alpha))) / volume

    # short range: fixed nimg=8 box, i,j,k in [-8, 8)
    idx = np.arange(-8, 8)
    I = np.stack(np.meshgrid(idx, idx, idx, indexing="ij"), axis=-1).reshape(-1, 3)
    I = I[np.any(I != 0, axis=1)]
    R = I @ lattv_sc
    dist = np.linalg.norm(R, axis=1)
    e_sr = np.sum(0.5 * erfc(alpha * dist) / dist)

    return e_sr + e_lr - (vlr_r0 + vsr_k0)


if __name__ == "__main__":
    # simple-cubic point-charge check: stored madelung = -2*nu = +2.8372974794/L
    L = 7.0
    latt = np.eye(3) * L
    recv = 2 * np.pi * np.linalg.inv(latt).T
    nu = madelung_bdft(latt, recv, [1, 1, 1], [40, 40, 40], 1e-10)
    print("madelung_bdft(SC L=%.1f) = %.10f  (expect %.10f)"
          % (L, nu, -1.4186487397 / L))
    # Ewald: NaCl-structure +/-1 charges, Madelung constant 1.747565
    a = 2.0
    latt = np.eye(3) * (2 * a)
    tau = np.array([[i, j, k] for i in (0, a) for j in (0, a) for k in (0, a)], float)
    z = np.array([(-1.0) ** ((int(round(t[0] / a)) + int(round(t[1] / a))
                              + int(round(t[2] / a))) % 2) for t in tau])
    e = ewald_energy(latt, tau, z)
    print("NaCl Madelung alpha = %.8f (expect 1.74756459)" % (-e * 2 * a / (8 * 1.0)))
