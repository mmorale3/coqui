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


def chemical_potential(e, v, nk, nelec, k_weight=None, qp_weight=0.1):
    """T=0 chemical potential for an approximate (moment-truncated) Lehmann G: the spectral weight below the physical gap is
    N_el only up to the moment/fit error, so exact filling would push mu across the gap. Instead: among the gaps between
    consecutive quasiparticle-like poles (total weight > qp_weight), choose the one whose midpoint gives the electron count
    closest to nelec. Returns (mu_shift, e_homo, e_lumo, N(mu))."""
    nk_ = len(e)
    wk = np.full(nk_, 1.0 / nk_) if k_weight is None else np.asarray(k_weight) / np.sum(k_weight)
    E = np.concatenate([e[k] for k in range(nk_)])
    Wt = np.concatenate([2.0 * wk[k] * (np.abs(v[k]) ** 2).sum(0) for k in range(nk_)])        # electron weight per pole
    Wtot = np.concatenate([(np.abs(v[k]) ** 2).sum(0) for k in range(nk_)])                   # weight per pole (sum over orbitals)
    order = np.argsort(E); Es, Ws = E[order], Wt[order]; cum = np.cumsum(Ws)
    qp_idx = np.where(Wtot[order] > qp_weight)[0]
    best = None
    for a, b in zip(qp_idx[:-1], qp_idx[1:]):
        mid = 0.5 * (Es[a] + Es[b]); N = cum[np.searchsorted(Es, mid) - 1]
        score = abs(N - nelec)
        if best is None or score < best[0] - 1e-12 or (abs(score - best[0]) < 1e-12 and Es[b] - Es[a] > best[3] - best[2]):
            best = (score, mid, Es[a], Es[b], N)
    _, mu, e_homo, e_lumo, N = best
    return mu, e_homo, e_lumo, N


def compress_sectors(basis_p, basis_h, zeta, e, v, emax=None):
    """Compressed matrix-coefficient real-pole representation of the Lehmann G per sector by LS fit on the line nodes.
    Poles beyond emax (default: the basis range lam) cannot be represented and are dropped; their weight is returned.
    Returns (w_all, coef_all, dropped_weight) with sector by sign of w (hole poles first)."""
    emax = basis_p.lam if emax is None else emax
    keep = np.abs(e) <= emax
    dropped = float((np.abs(v[:, ~keep]) ** 2).sum())
    def lehmann(mask):
        K = 1.0 / (zeta[:, None] - e[mask][None, :]); vm = v[:, mask]
        return (vm[None, :, :] * K[:, None, :]) @ vm.conj().T                  # (nz, nb, nb)
    Gp = lehmann(keep & (e > 0)); Gh = lehmann(keep & (e < 0))
    cp = basis_p.fit(zeta, Gp); ch = basis_h.fit(zeta, Gh)
    return np.concatenate([basis_h.w, basis_p.w]), np.concatenate([ch, cp], axis=0), dropped
