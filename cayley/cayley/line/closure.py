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


def thermal_carriers(e, v, beta, mu=0.0, k_weight=None):
    """n_th(mu) of notes Eq. fT_nth: thermal carriers across mu, 2 sum_k w_k [sum_{e > mu} f |v|^2 + sum_{e < mu} (1 - f) |v|^2]."""
    nk_ = len(e)
    wk = np.full(nk_, 1.0 / nk_) if k_weight is None else np.asarray(k_weight) / np.sum(k_weight)
    n = 0.0
    for k in range(nk_):
        ek = np.asarray(e[k], float)
        f = 0.5 * (1.0 - np.tanh(0.5 * beta * (ek - mu)))
        n += 2.0 * wk[k] * float((np.where(ek > mu, f, 1.0 - f) * (np.abs(v[k]) ** 2).sum(0)).sum())
    return n


def chemical_potential_auto(e, v, nk, nelec, beta, thermal_tol=1e-8, k_weight=None, dropped=0.0, mu_dn_max=0.1,
                            mu_th_factor=10.0, return_info=False, **kw):
    """Rule mu_rule = "auto" (notes section 11.6, Eq. fT_nth; revised 2026-10-10): the widest admissible gap of
    chemical_potential first (midpoint mu_g); dN = N_T(mu_g) - N_el (Eq. fT_mu, thermal count) and n_th(mu_g) (thermal carriers
    across the gap). The midpoint is used (rule "gap"; dN = the closure's weight deficit) iff |dN| <= mu_dn_max AND
    |dN| > mu_th_factor n_th(mu_g); otherwise N(mu) = N_el (chemical_potential_T, rule "number"). thermal_tol is unused (kept
    for the interface; the first version's window condition is the n_th = 0 special case). Reference: dev/s8b_mu_rule.py.
    Returns (mu_shift, rule, N(mu)) (+ dict(dN, n_th, mu_g, e_homo, e_lumo) with return_info)."""
    mu_g, e_homo, e_lumo, N = chemical_potential(e, v, nk, nelec, k_weight, **kw)
    info = dict(mu_g=mu_g, e_homo=e_homo, e_lumo=e_lumo, dN=0.0, n_th=0.0)
    if beta is None:
        return (mu_g, 'gap', N, info) if return_info else (mu_g, 'gap', N)
    dN = electron_count_T(e, v, beta, mu_g, k_weight, dropped) - nelec
    nth = thermal_carriers(e, v, beta, mu_g, k_weight)
    info.update(dN=dN, n_th=nth)
    if abs(dN) <= mu_dn_max and abs(dN) > mu_th_factor * nth:
        out = (mu_g, 'gap', dN + nelec)
    else:
        mu, Nn = chemical_potential_T(e, v, nk, nelec, beta, k_weight, dropped)
        out = (mu, 'number', Nn)
    return out + (info,) if return_info else out


def chemical_potential_neutral(e, v, nk, nelec, beta, k_weight=None, dropped=0.0, mu_dn_max=0.1, return_info=False, **kw):
    """Rule mu_rule = "neutral" (S8b.4 (e), notes section 11.6): a continuous version of "auto". The widest admissible gap of
    chemical_potential (midpoint mu_g, T = 0 count N_g = 2 sum_k w_k sum_{e < mu_g} |v|^2) gives the closure's count error
    eps0 = N_g + dropped - N_el (zero for an exact insulator: the weight below the gap is N_el). If |eps0| <= mu_dn_max AND
    n_th(mu_g) <= mu_dn_max (a gap with few thermal carriers), mu solves N_T(mu) = N_el + eps0, i.e. thermal electrons = thermal
    holes (the intrinsic, charge-neutral mu: continuous in the poles, the error eps0 is not compensated by moving mu towards an
    edge); otherwise N_T(mu) = N_el (rule "number"; metals: |eps0| = O(0.5) or many carriers). With an exact closure eps0 = 0 and
    the rule IS "number". Returns (mu_shift, rule, N(mu)) (+ dict(dN = eps0, n_th, mu_g, e_homo, e_lumo) with return_info)."""
    mu_g, e_homo, e_lumo, Ng = chemical_potential(e, v, nk, nelec, k_weight, **kw)
    eps0 = Ng + dropped - nelec
    nth = thermal_carriers(e, v, beta, mu_g, k_weight)
    info = dict(mu_g=mu_g, e_homo=e_homo, e_lumo=e_lumo, dN=eps0, n_th=nth)
    if abs(eps0) <= mu_dn_max and nth <= mu_dn_max:
        mu, Nn = chemical_potential_T(e, v, nk, nelec + eps0, beta, k_weight, dropped)
        out = (mu, 'neutral', Nn)
    else:
        mu, Nn = chemical_potential_T(e, v, nk, nelec, beta, k_weight, dropped)
        out = (mu, 'number', Nn)
    return out + (info,) if return_info else out


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


def upfold_sigma(w, g, wp, K, tol_gram=1e-10, nphi=72):
    """Upfolded real-pole form of Sigma_c(k) from its sector fit (w, g): Sigma_c(z) = W (z - d)^{-1} W^dagger (mu-relative d)."""
    C = sigma_moments(w, g, wp, K + 1)
    d, W, info = upfold_block(C, K, wp, 0.0, tol_gram=tol_gram, nphi=nphi, return_info=True)
    return np.asarray(d, float), np.asarray(W), info


def qp_secular(h, d, g, gtol_rel=1e-14, merge_tol=1e-12, nbisect=200, nn=16):
    """Diagonal QP equation E = h + sum_j g_j / (E - d_j) (eta = 0; d real, g >= 0: the upfolded Sigma_nn) solved EXACTLY:
    f(E) = h - E + sum_j g_j/(E - d_j) is strictly decreasing between consecutive poles, from +inf to -inf, so it has exactly
    one root in each of the r + 1 intervals (the roots = the eigenvalues of the arrowhead matrix [[h, sqrt g^T], [sqrt g,
    diag d]]); the weight of root E is Z = 1/(1 + sum g/(E - d)^2) (sum Z = 1). Returns the root of LARGEST Z (the QP) and Z.
    Poles with g <= gtol_rel * sum g are dropped; poles closer than merge_tol are merged (summed g). The outer brackets:
    root_low >= min(h, d_0) - sqrt(sum g) (there f >= s - G/s >= 0), symmetric above. Bisection to the double resolution.
    Replaces the Newton / linearized solve of the C++ qp_diag start (D2: the linearization near a pole of Sigma_nn sent 157
    (k, band) of Si 4^3 by up to 162 eV)."""
    d = np.asarray(d, float); g = np.asarray(g, float)
    G = g.sum()
    if len(d) == 0 or G <= 0.0:
        return float(h), 1.0
    keep = g > gtol_rel * G
    d, g = d[keep], g[keep]
    o = np.argsort(d, kind='stable'); d, g = d[o], g[o]
    dm, gm = [d[0]], [g[0]]
    for x, y in zip(d[1:], g[1:]):
        if x - dm[-1] < merge_tol: gm[-1] += y
        else: dm.append(x); gm.append(y)
    d, g = np.array(dm), np.array(gm)
    s = np.sqrt(g.sum()) + 1e-3
    r = len(d)
    LO = np.concatenate([[min(h, d[0]) - s], d]); HI = np.concatenate([d, [max(h, d[-1]) + s]])

    @np.errstate(divide='ignore', invalid='ignore')
    def solve(js):                                                           # x == d_j only if the interval collapsed: Z = 0
        lo, hi = LO[js].copy(), HI[js].copy()
        f = lambda x: h - x + (g[None, :] / (x[:, None] - d[None, :])).sum(1)
        for _ in range(nbisect):
            x = 0.5 * (lo + hi)
            if np.all((x == lo) | (x == hi)): break
            pos = f(x) > 0
            lo = np.where(pos, x, lo); hi = np.where(pos, hi, x)
        x = 0.5 * (lo + hi)
        return x, 1.0 / (1.0 + (g[None, :] / (x[:, None] - d[None, :]) ** 2).sum(1))
    # candidates: the two outer intervals and the one containing h; then every interior interval (d_j, d_j+1) whose rigorous
    # bound beats them: for x in it, S(x) = sum_i g_i/(x - d_i)^2 >= (g_j^1/3 + g_j+1^1/3)^3 / L^2 (the minimum of the two
    # neighbour terms) + sum over the next nn poles on each side of g_i / (farthest end - d_i)^2, Z = 1/(1 + S)
    # (pruning is exact: the result is the largest-Z root over ALL intervals; ties -> the lowest energy)
    js0 = np.unique([0, r, int(np.searchsorted(d, h))])
    x0, z0 = solve(js0)
    if z0.max() > 0.5:                                                       # sum Z = 1: no other root can beat it
        j = int(np.argmax(z0)); return float(x0[j]), float(z0[j])
    if r > 1:
        L = d[1:] - d[:-1]
        S = (np.cbrt(g[:-1]) + np.cbrt(g[1:])) ** 3 / L ** 2
        jl = np.arange(r - 1)                                                # interval jl = (d[jl], d[jl + 1])
        for m in range(1, nn + 1):
            il, ir = jl - m, jl + 1 + m
            okl, okr = il >= 0, ir < r
            S[okl] += g[il[okl]] / (d[jl[okl] + 1] - d[il[okl]]) ** 2
            S[okr] += g[ir[okr]] / (d[ir[okr]] - d[jl[okr]]) ** 2
        ub = 1.0 / (1.0 + S)
        js = np.union1d(js0, 1 + np.where(ub > z0.max())[0])
    else:
        js = js0
    x, Z = (x0, z0) if len(js) == len(js0) else solve(js)
    j = int(np.argmax(Z))
    return float(x[j]), float(Z[j])


def qp_diag(Hstat_rel, w, g, wp, K, V=None, tol_gram=1e-10, nphi=72):
    """Diagonal QP energies (mu-relative) and Z of the orbitals V (nb, nb) columns (default: the basis vectors) from the
    sector fit (w, g) of Sigma_c(k): E_n = h_nn + Sigma_nn(E_n), h = V^dag Hstat_rel V, Sigma_nn from the upfolded form."""
    d, W, info = upfold_sigma(w, g, wp, K, tol_gram=tol_gram, nphi=nphi)
    nb = Hstat_rel.shape[0]
    V = np.eye(nb) if V is None else V
    WV = V.conj().T @ W
    h = np.real(np.einsum('in,ij,jn->n', V.conj(), Hstat_rel, V))
    E, Z = np.zeros(nb), np.zeros(nb)
    for n in range(nb):
        E[n], Z[n] = qp_secular(h[n], d, np.abs(WV[n]) ** 2)
    return E, Z, dict(npoles=len(d), **info)
