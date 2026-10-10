"""Closing the self-consistency loop on the line: Sigma_c at the fermionic nodes -> real-pole fit per sector -> Cayley moments
-> block upfolding -> upfolded Hamiltonian -> G in Lehmann form (positive) -> compressed real-pole fit per sector -> next poles.
Also the chemical potential (T=0 filling) and the Dyson G at the nodes for diagnostics.
Finite T (S8b, notes section 11): chemical_potential_T (Eq. fT_mu, bisection), chemical_potential_auto (rule "auto"),
fit_sigma_total (TOTAL Sigma_c on the two-sided gapless basis at the unmasked nodes), wp_thermal (omega_p floor)."""
import numpy as np
from ..moments import moments_from_poles
from ..upfold import upfold_block
from ..spectral import upfolded_hamiltonian
from .line_dlr import LineBasis


def fit_sigma_sectors(basis_p, basis_h, zeta, Sig_p, Sig_h, mask=None):
    """Real-pole fits of the particle and hole parts of Sigma_c sampled at mu-relative nodes zeta:
    Sigma^>(zeta) ~ sum_{l in basis_p} g_l/(zeta - w_l),  Sigma^<(zeta) ~ sum_{l in basis_h} ... ; returns (w_all, g_all).
    mask (finite T): boolean node mask (node floor); None = all nodes (T = 0, unchanged)."""
    if mask is not None:
        zeta, Sig_p, Sig_h = zeta[mask], Sig_p[mask], Sig_h[mask]
    gp = basis_p.fit(zeta, Sig_p); gh = basis_h.fit(zeta, Sig_h)
    return np.concatenate([basis_h.w, basis_p.w]), np.concatenate([gh, gp], axis=0)


def fit_sigma_total(basis, zeta, Sig, mask=None):
    """Finite-T closure input (notes section 11.6): the TOTAL Sigma_c = I^> + I^< at the unmasked nodes (rho beta |zeta| >=
    c_zeta) fitted on the two-sided GAPLESS fermionic basis (LineBasis(..., gap=(0, 0))); returns (w, g) for sigma_moments."""
    if mask is not None:
        zeta, Sig = zeta[mask], Sig[mask]
    return basis.w.copy(), basis.fit(zeta, Sig)


def wp_thermal(wp, zeta_T, wp_floor=15.0):
    """omega_p := max(wp, wp_floor * zeta_T) (E3 on the real lih222 Sigma_c, S8b.1: omega_p >= 15 zeta_T for K <= 24 at 1e-8
    noise; the toy E3 gave 10)."""
    return max(wp, wp_floor * zeta_T)


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


def chemical_potential(e, v, nk, nelec, k_weight=None, qp_weight=0.1, ntol=0.5):
    """T=0 chemical potential for an approximate (moment-truncated) Lehmann G: the spectral weight below the physical gap is
    N_el only up to the moment/fit error, so exact filling would push mu across the gap. Candidate gaps are the intervals between
    consecutive quasiparticle-like poles (total weight > qp_weight) whose midpoint gives an electron count within ntol of nelec;
    among them the WIDEST gap is chosen (a near-degenerate multiplet split by noise must not be mistaken for the gap).
    Returns (mu_shift, e_homo, e_lumo, N(mu))."""
    nk_ = len(e)
    wk = np.full(nk_, 1.0 / nk_) if k_weight is None else np.asarray(k_weight) / np.sum(k_weight)
    E = np.concatenate([e[k] for k in range(nk_)])
    Wt = np.concatenate([2.0 * wk[k] * (np.abs(v[k]) ** 2).sum(0) for k in range(nk_)])        # electron weight per pole
    Wtot = np.concatenate([(np.abs(v[k]) ** 2).sum(0) for k in range(nk_)])                   # weight per pole (sum over orbitals)
    order = np.argsort(E); Es, Ws = E[order], Wt[order]; cum = np.cumsum(Ws)
    qp_idx = np.where(Wtot[order] > qp_weight)[0]
    cands = []
    for a, b in zip(qp_idx[:-1], qp_idx[1:]):
        mid = 0.5 * (Es[a] + Es[b]); N = cum[np.searchsorted(Es, mid) - 1]
        cands.append((Es[b] - Es[a], abs(N - nelec), mid, Es[a], Es[b], N))
    ok = [c for c in cands if c[1] <= ntol]
    if not ok:                                   # fall back to the best electron count
        ok = [min(cands, key=lambda c: c[1])]
    width, dn, mu, e_homo, e_lumo, N = max(ok, key=lambda c: c[0])
    return mu, e_homo, e_lumo, N


def electron_count_T(e, v, beta, mu=0.0, k_weight=None, dropped=0.0):
    """N(mu) = 2 sum_k w_k sum_m f(e_m - mu) |v_m|^2 + dropped (Eq. fT_mu); e (nk, M) mu-relative, v (nk, nb, M)."""
    nk_ = len(e)
    wk = np.full(nk_, 1.0 / nk_) if k_weight is None else np.asarray(k_weight) / np.sum(k_weight)
    N = dropped
    for k in range(nk_):
        f = 0.5 * (1.0 - np.tanh(0.5 * beta * (np.asarray(e[k], float) - mu)))
        N += 2.0 * wk[k] * float((f * (np.abs(v[k]) ** 2).sum(0)).sum())
    return N


def chemical_potential_T(e, v, nk, nelec, beta, k_weight=None, dropped=0.0, tol=1e-12, maxit=200):
    """Finite-T chemical potential (Eq. fT_mu): N(mu) = N_el by bisection (N monotone and continuous in mu); e mu-relative.
    dropped: hole weight beyond Lambda removed by the pruning (counted as occupied). Returns (mu_shift, N(mu_shift)).
    The bisection runs to the root in mu (machine precision); tol is kept for the interface (|N - N_el| ends far below it)."""
    E = np.concatenate([np.asarray(e[k], float) for k in range(len(e))])
    E = E[np.abs(E) < 1e5]                                              # padding poles (driver) are not bracketing candidates
    lo, hi = E.min() - 50.0 / beta - 1.0, E.max() + 50.0 / beta + 1.0
    Nlo = electron_count_T(e, v, beta, lo, k_weight, dropped) - nelec
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        Nm = electron_count_T(e, v, beta, mid, k_weight, dropped) - nelec
        if Nm == 0.0 or hi - lo < 4e-16 * max(1.0, abs(mid)):          # to the root in mu (|N - N_el| <= tol is not enough:
            break                                                        # dN/dmu ~ beta e^{-beta gap/2} in a gap)
        if (Nm < 0) == (Nlo < 0):
            lo, Nlo = mid, Nm
        else:
            hi = mid
    return mid, Nm + nelec


def chemical_potential_auto(e, v, nk, nelec, beta, thermal_tol=1e-8, k_weight=None, dropped=0.0, **kw):
    """Rule mu_rule = "auto" (notes section 11.6): the widest admissible gap of chemical_potential first; if both its edges
    lie outside the window around its midpoint (beta * min(e_lumo - mu, mu - e_homo) > c_T = ln(1/thermal_tol)) the midpoint
    is used (T = 0 equivalent; thermal mode stays off), otherwise N(mu) = N_el (chemical_potential_T).
    Returns (mu_shift, rule, N(mu)) with rule in {"gap", "number"}."""
    mu, e_homo, e_lumo, N = chemical_potential(e, v, nk, nelec, k_weight, **kw)
    if beta is None or beta * min(e_lumo - mu, mu - e_homo) > np.log(1.0 / thermal_tol):
        return mu, 'gap', N
    mu, N = chemical_potential_T(e, v, nk, nelec, beta, k_weight, dropped)
    return mu, 'number', N


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
