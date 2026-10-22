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


def chemical_potential(e, v, nk, nelec, k_weight=None, qp_weight=0.1, ntol=0.5, ntol_k=0.5):
    """T=0 chemical potential for an approximate (moment-truncated) Lehmann G: the spectral weight below the physical gap is
    N_el only up to the moment/fit error, so exact filling would push mu across the gap. Candidate gaps are the intervals between
    consecutive quasiparticle-like poles (total weight > qp_weight) whose midpoint gives an electron count within ntol of nelec;
    among them the WIDEST gap is chosen (a near-degenerate multiplet split by noise must not be mistaken for the gap).
    Insulator filter (D2, 2026-10-22): if some candidates fill EVERY k to within ntol_k of nelec (per-k count
    2 sum_{e_m(k) < mid} |v_m|^2, no k weight), the choice is restricted to them. An isolated conduction valley (SiC X: 3 of 64 k,
    k-weighted count 8.07, inside ntol) carries per-k counts N_el + 2Z there and is no longer taken for the gap; a metal has no
    such candidate (per-k counts differ by 2Z across the Fermi surface) and keeps the rule above. ntol_k <= 0 disables the filter.
    Returns (mu_shift, e_homo, e_lumo, N(mu))."""
    nk_ = len(e)
    wk = np.full(nk_, 1.0 / nk_) if k_weight is None else np.asarray(k_weight) / np.sum(k_weight)
    E = np.concatenate([e[k] for k in range(nk_)])
    Wt = np.concatenate([2.0 * wk[k] * (np.abs(v[k]) ** 2).sum(0) for k in range(nk_)])        # electron weight per pole
    Wtot = np.concatenate([(np.abs(v[k]) ** 2).sum(0) for k in range(nk_)])                   # weight per pole (sum over orbitals)
    order = np.argsort(E); Es, Ws = E[order], Wt[order]; cum = np.cumsum(Ws)
    qp_idx = np.where(Wtot[order] > qp_weight)[0]
    ek, ck = [], []                                                                            # per-k sorted poles, counts
    for k in range(nk_):
        o = np.argsort(e[k]); ek.append(np.asarray(e[k])[o])
        ck.append(np.concatenate([[0.0], np.cumsum(2.0 * (np.abs(v[k]) ** 2).sum(0)[o])]))
    cands = []
    for a, b in zip(qp_idx[:-1], qp_idx[1:]):
        mid = 0.5 * (Es[a] + Es[b]); N = cum[np.searchsorted(Es, mid) - 1]
        dnk = max(abs(ck[k][np.searchsorted(ek[k], mid)] - nelec) for k in range(nk_))
        cands.append((Es[b] - Es[a], abs(N - nelec), mid, Es[a], Es[b], N, dnk))
    ok = [c for c in cands if c[1] <= ntol]
    ins = [c for c in ok if c[6] <= ntol_k] if ntol_k > 0 else []
    if ins:
        ok = ins
    if not ok:                                   # fall back to the best electron count
        ok = [min(cands, key=lambda c: c[1])]
    width, dn, mu, e_homo, e_lumo, N, _ = max(ok, key=lambda c: c[0])
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


# ---------------------------------------------------------------------------------------------------------------------------
# D5 (2026-10-19): multi-scale spectra. The resolution of one closure about the centre is (pi/K)[(w - mu)^2 + wp^2]/wp, so the
# small wp of the SCF closure (Si: 0.11 Ha) cannot resolve plasmon satellites 10-30 eV below mu. The spectra are blended from
# closures of the SAME Sigma_c fit at several Cayley scales s_0 = wp < s_1 < ... (each a positive Lehmann G), with the cos^2
# partition of unity chi_i(w) in ln|w - mu| whose node i sits at |w - mu| = s_i (as the optics closure MB): A = sum_i chi_i A_i,
# a convex combination (A >= 0); |w - mu| <= s_0 is the base closure unchanged.
def multiscale_weights(om, scales):
    """chi (ns, nw): cos^2 partition of unity in ln|w| (w mu-relative), node of scale i at |w| = scales[i] (ascending)."""
    om = np.asarray(om, float); L = np.log(np.asarray(scales, float)); ns = len(L)
    chi = np.zeros((ns, len(om)))
    for j, x in enumerate(np.log(np.maximum(np.abs(om), 1e-300))):
        if x <= L[0]: chi[0, j] = 1.0; continue
        if x >= L[-1]: chi[-1, j] = 1.0; continue
        i = int(np.searchsorted(L, x, side='right')) - 1; t = (x - L[i]) / (L[i + 1] - L[i])
        chi[i, j] = np.cos(0.5 * np.pi * t) ** 2; chi[i + 1, j] = 1.0 - chi[i, j]
    return chi


def lorentz_tails(e, a, eta, lo, hi):
    """Weight and principal-value first moment of sum_m a_m L_eta(w - e_m) OUTSIDE [lo, hi] (analytic)."""
    fin = (np.arctan((hi - e) / eta) - np.arctan((lo - e) / eta)) / np.pi
    m0 = (a * (1.0 - fin)).sum()
    m1 = (a * (e * (1.0 - fin) - eta / (2 * np.pi) * np.log(((hi - e) ** 2 + eta ** 2) / ((lo - e) ** 2 + eta ** 2)))).sum()
    return m0, m1


def spectra_multiscale(Hrel, w, g, scales, K, om, etas, K_base=None, tol_gram=1e-10, nphi=72):
    """Blended A(w) of one k from the Sigma_c fit (w, g) and Hrel = H_stat - mu. scales[0] = the base closure scale (order
    K_base, default K), scales[1:] the extra scales (order K). Returns dict(A_diag (ne, nw, nb), A_trace (ne, nw),
    comp_trace (ns, ne, nw), chi (ns, nw), m0_err (ne,), m1_err (ne,)): sum rules of the blended trace on the grid (trapezoid)
    + the analytic Lorentzian tails of the last scale's Lehmann G outside the grid; m0_err = |M0 - nb| / nb,
    m1_err = |M1 - Tr Hrel| / sum_i |Hrel_ii| (first moment: Sigma_c ~ C0/z enters only the second)."""
    from ..spectral import spectral_function
    nb = Hrel.shape[0]; om = np.asarray(om, float); scales = list(scales); ns = len(scales)
    chi = multiscale_weights(om, scales)
    comp = []
    for i, s in enumerate(scales):
        Ki = (K if K_base is None else K_base) if i == 0 else K
        C = sigma_moments(w, g, s, Ki + 1)
        d, W = upfold_block(C, Ki, s, 0.0, tol_gram=tol_gram, nphi=nphi)
        comp.append((np.asarray(d, float), np.asarray(W)))
    Ad = np.zeros((len(etas), len(om), nb)); At = np.zeros((len(etas), len(om))); Ct = np.zeros((ns, len(etas), len(om)))
    m0e = np.zeros(len(etas)); m1e = np.zeros(len(etas))
    for ie, eta in enumerate(etas):
        for i, (d, W) in enumerate(comp):
            S = lambda z, d=d, W=W: np.einsum('zl,il,jl->zij', 1.0 / (np.atleast_1d(z)[:, None] - d[None, :]), W, W.conj())
            A = spectral_function(Hrel, S, om, eta)
            dg = np.real(np.einsum('wii->wi', A))
            Ad[ie] += chi[i][:, None] * dg; Ct[i, ie] = dg.sum(1)
        At[ie] = Ad[ie].sum(1)
        d, W = comp[-1]
        e, V = np.linalg.eigh(upfolded_hamiltonian(Hrel, d, W)); a = (np.abs(V[:nb]) ** 2).sum(0)
        t0, t1 = lorentz_tails(e, a, eta, om[0], om[-1])
        M0 = np.trapezoid(At[ie], om) + t0; M1 = np.trapezoid(om * At[ie], om) + t1
        tr = np.real(np.trace(Hrel))
        m0e[ie] = abs(M0 - nb) / nb; m1e[ie] = abs(M1 - tr) / np.abs(np.real(np.diag(Hrel))).sum()
    return dict(A_diag=Ad, A_trace=At, comp_trace=Ct, chi=chi, m0_err=m0e, m1_err=m1e)
