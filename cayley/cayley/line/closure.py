"""Closing the self-consistency loop on the line: Sigma_c at the fermionic nodes -> real-pole fit per sector -> Cayley moments
-> block upfolding -> upfolded Hamiltonian -> G in Lehmann form (positive) -> compressed real-pole fit per sector -> next poles.
Also the chemical potential (T=0 filling) and the Dyson G at the nodes for diagnostics."""
import numpy as np
from ..moments import moments_from_poles
from ..upfold import upfold_block
from ..spectral import upfolded_hamiltonian
from .line_dlr import LineBasis


def fit_sigma_sectors(basis_p, basis_h, zeta, Sig_p, Sig_h):
    """Real-pole fits of the particle and hole parts of Sigma_c sampled at mu-relative nodes zeta:
    Sigma^>(zeta) ~ sum_{l in basis_p} g_l/(zeta - w_l),  Sigma^<(zeta) ~ sum_{l in basis_h} ... ; returns (w_all, g_all)."""
    gp = basis_p.fit(zeta, Sig_p); gh = basis_h.fit(zeta, Sig_h)
    return np.concatenate([basis_h.w, basis_p.w]), np.concatenate([gh, gp], axis=0)


def sigma_moments(w, g, wp, nmax, mu_rel=0.0):
    """Cayley moments of the real-pole Sigma_c about the (mu-relative) centre, total measure."""
    return moments_from_poles(w, g, wp, nmax, mu_rel)


def lehmann_from_sigma(Hstat_rel, w, g, wp, K, tol_gram=1e-10, nphi=72):
    """Upfold the moments of Sigma_c(k) (poles w, matrix coefficients g) and diagonalize [[H, W],[W^dag, d]] (mu-relative):
    returns (e_m, v_m) with G(zeta) = sum_m v_m v_m^dag/(zeta - e_m), plus the upfolding info."""
    C = sigma_moments(w, g, wp, K + 1)
    d, W, info = upfold_block(C, K, wp, 0.0, tol_gram=tol_gram, nphi=nphi, return_info=True)
    e, V = np.linalg.eigh(upfolded_hamiltonian(Hstat_rel, d, W))
    nb = Hstat_rel.shape[0]
    return e, V[:nb, :], dict(npoles=len(d), **info)


def chemical_potential(e, v, nk, nelec, k_weight=None):
    """T=0 filling: order all poles (k, m) by energy, fill weight 2*|v|^2*w_k until nelec; return the mid-gap mu shift."""
    nk_ = len(e)
    wk = np.full(nk_, 1.0 / nk_) if k_weight is None else np.asarray(k_weight) / np.sum(k_weight)
    E = np.concatenate([e[k] for k in range(nk_)])
    Wt = np.concatenate([2.0 * wk[k] * (np.abs(v[k]) ** 2).sum(0) for k in range(nk_)])
    order = np.argsort(E); cum = np.cumsum(Wt[order])
    j = int(np.searchsorted(cum, nelec - 1e-8))
    e_homo, e_lumo = E[order][j], E[order][min(j + 1, len(E) - 1)]
    return 0.5 * (e_homo + e_lumo), e_homo, e_lumo, cum[j] if j < len(cum) else cum[-1]


def compress_sectors(basis_p, basis_h, zeta, e, v):
    """Compressed matrix-coefficient real-pole representation of the Lehmann G per sector by LS fit on the line nodes.
    Returns (w_all, coef_all) with sector by sign of w (hole poles first)."""
    Gp = np.einsum('zm,im,jm->zij', 1.0 / (zeta[:, None] - e[e > 0][None, :]), v[:, e > 0], v[:, e > 0].conj())
    Gh = np.einsum('zm,im,jm->zij', 1.0 / (zeta[:, None] - e[e < 0][None, :]), v[:, e < 0], v[:, e < 0].conj())
    cp = basis_p.fit(zeta, Gp); ch = basis_h.fit(zeta, Gh)
    return np.concatenate([basis_h.w, basis_p.w]), np.concatenate([ch, cp], axis=0)
