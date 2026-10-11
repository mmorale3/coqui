"""Finite-temperature oracles (S8b, notes section 11), independent of the line machinery (no rays, no line bases).

Energies are measured from the chemical potential mu0 (e = eig - mu0), f(e) = 1/(e^{beta e}+1), n(nu) = 1/(e^{beta nu}-1).
THC: X (nk, Np, nb) collocation, Z (nq, Np, Np); qk[iq, ik] = index of k - q; KS poles (c_m = unit vectors).

* Transition sum (Eq. fT_pi): Pi(q, zeta) = (2/Nk) sum_k sum_{n in k, m in k-q} [f(e_n(k)) - f(e_m(k-q))] S S^H/(zeta - E),
  E = e_m(k-q) - e_n(k), S_P = X_Pn(k) conj(X_Pm(k-q)) (rank one, Hermitian residue). Exactly degenerate pairs (|E| < deg_tol)
  are excluded: the line's dynamic (retarded) convention; their Matsubara static term is pi_nu0_extra().
* Imaginary-time route (second route for the thermal factors, Matsubara convention): Pi(q, i nu_n) = int_0^beta dtau
  e^{i nu_n tau} (2/Nk) sum_k [G~(k,-tau)]_PQ [G~(k-q,tau)]_QP with G(k,-tau) = +sum_n f_n e^{e_n tau} c_n,
  G(k-q,tau) = -sum_m (1-f_m) e^{-e_m tau} c_m (each factor bounded by 1 on [0, beta]); composite Gauss-Legendre in tau graded
  at both ends. Contains the degenerate pairs (beta f(1-f) R at nu_0).
* Finite-T Casida (make_casida.py --beta): all pairs with E != 0, sg = sign(E), columns scaled by sqrt|F|, F = f_n - f_m;
  H = diag(E) + sg (S^H Z S) -> W_dyn(q, z) = Z Pi Z (I - ...)^{-1} = sum_s alpha_s bet_s/(z - lam_s).
* Eq. fT_sigma (the finite-T GW pole form, positive measure) from the exact Casida poles: positive poles nu_j of q with
  residues w_j(q) = alpha_s bet_s and of -q transposed,
    Sigma_c(k, zeta) = (1/Nk) sum_q sum_{m in k-q} sum_j [(1-f_m+n_j) X^+[c~_m o w_j(q)]X/(zeta - e_m - nu_j)
                                                       + (f_m+n_j) X^+[c~_m o w_j(-q)^T]X/(zeta - e_m + nu_j)].
* S8b.3 hybrid (notes section 11.6 "Hybrid"): matsubara_set (the dense truncated fermionic set n = 0..N-1, N from w_max),
  density_matsubara (Matsubara Dyson G(i w_n; mu) = [i w_n + dmu - H - Sigma_c(i w_n)]^-1, D by the Matsubara sum with the
  free reference f(H - dmu) and the analytic 1/w^4 tail from the moments S1, S2 of Sigma_c, N(mu) = N_el by bisection at
  fixed Sigma_c), sigma_fT_hf (exact S1, S2 of Eq. fT_sigma), density_upfold (continuation-free exact D for a small pole list).
* nu_0 term (test T3): Sigma^Mats(k, i w_n) - Sigma^an(k, i w_n) = -(1/beta)(1/Nk) sum_q X^+[G~(k-q, i w_n) o dW(q)] X,
  dW(q) = W^Mats(q, i nu_0) - W^an(q, 0), Pi^Mats(q, 0) = Pi^an(q, 0) + dPi(q), dPi = -(2/Nk) beta sum_{|E|<deg_tol} f_n(1-f_m) S S^H.
"""
import numpy as np


# ------------------------------------------------------------------------------------------------ thermal functions
def fermi(e, beta):
    return 0.5 * (1.0 - np.tanh(0.5 * beta * np.asarray(e, float)))


def bose(nu, beta):
    x = beta * np.asarray(nu, float)
    with np.errstate(over='ignore'):
        return np.where(x > 700.0, 0.0, 1.0 / np.expm1(np.minimum(x, 700.0)))


def fermi_diff(a, b, beta):
    """f(a) - f(b) without cancellation: [e^{beta b} - e^{beta a}]/[(1+e^{beta a})(1+e^{beta b})] in log form."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    sgn = np.where(a < b, 1.0, -1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        logv = beta * hi + np.log1p(-np.exp(beta * (lo - hi))) - np.logaddexp(0.0, beta * a) - np.logaddexp(0.0, beta * b)
        out = sgn * np.exp(logv)
    return np.where(a == b, 0.0, out)


def fermi_logs(e, beta):
    """(log f(e), log(1 - f(e)))."""
    e = np.asarray(e, float)
    return -np.logaddexp(0.0, beta * e), -np.logaddexp(0.0, -beta * e)


# ------------------------------------------------------------------------------------------------ transitions
def transitions(X, e, qk, iq, beta, deg_tol=1e-8, rel_drop=1e-16):
    """All (k, n, m) pairs of Eq. fT_pi for q = iq: returns dict with S (Np, T) = sqrt(2/Nk) X_n(k) conj(X_m(k-q)),
    E (T,), F (T,) = f(e_n(k)) - f(e_m(k-q)), and the degenerate pairs (|E| < deg_tol): Sd (Np, Td), wd (Td,) = f_n (1 - f_m).
    Pairs with |F| < rel_drop |E| (contribution < rel_drop R everywhere on the line and at zeta = 0) are dropped.
    beta None: T = 0 occupations (F in {0, +-1})."""
    nk, Np, nb = X.shape
    cols, E, F, cd, wd = [], [], [], [], []
    for ik in range(nk):
        ikmq = qk[iq, ik]
        en, em = e[ik][:, None], e[ikmq][None, :]
        Ekk = em - en                                                        # (n, m)
        if beta is None:
            Fkk = (en < 0).astype(float) - (em < 0).astype(float)
        else:
            Fkk = fermi_diff(en, em, beta)
        for n in range(nb):
            for m in range(nb):
                Ev, Fv = Ekk[n, m], Fkk[n, m]
                if abs(Ev) < deg_tol:
                    if beta is not None:
                        w = float(np.exp(fermi_logs(e[ik][n], beta)[0] + fermi_logs(e[ikmq][m], beta)[1]))
                        if w > 0:
                            cd.append(X[ik][:, n] * np.conj(X[ikmq][:, m])); wd.append(w)
                    continue
                if abs(Fv) < rel_drop * abs(Ev) or Fv == 0.0:
                    continue
                cols.append(X[ik][:, n] * np.conj(X[ikmq][:, m])); E.append(Ev); F.append(Fv)
    sc = np.sqrt(2.0 / nk)
    out = dict(S=sc * np.array(cols).T, E=np.array(E), F=np.array(F))
    out['Sd'] = sc * np.array(cd).T if cd else np.zeros((Np, 0), complex)
    out['wd'] = np.array(wd)
    return out


def pi_transition(tr, zeta):
    """Pi(q, zeta) (nz, Np, Np) from transitions(): sum_t F_t S_t S_t^H/(zeta - E_t) at any complex zeta (dynamic convention)."""
    S, E, F = tr['S'], tr['E'], tr['F']
    zeta = np.atleast_1d(np.asarray(zeta, complex))
    return np.array([(S * (F / (z - E))[None, :]) @ S.conj().T for z in zeta])


def pi_nu0_extra(tr, beta):
    """dPi(q) = Pi^Mats(q, i nu_0) - Pi^an(q, 0) = -beta sum_{degenerate} f_n (1 - f_m) S S^H (S already has sqrt(2/Nk))."""
    Sd, wd = tr['Sd'], tr['wd']
    return -beta * (Sd * wd[None, :]) @ Sd.conj().T


# ------------------------------------------------------------------------------------------------ imaginary-time route
def gl_tau_grid(beta, emax, smin=1e-6, per_efold=3, nn=16, hmax=None):
    """Composite Gauss-Legendre nodes/weights on [0, beta], log-graded towards both ends (decay scale 1/emax)."""
    xg, wg = np.polynomial.legendre.leggauss(nn)
    half = 0.5 * beta
    a0 = min(smin, 0.1 / emax)
    edges = np.concatenate([[0.0], np.exp(np.linspace(np.log(a0), np.log(half), int(np.log(half / a0) * per_efold) + 2))])
    if hmax is not None:
        edges = np.concatenate([[0.0]] + [np.linspace(a, b, int(np.ceil((b - a) / hmax)) + 1)[1:] for a, b in zip(edges[:-1], edges[1:])])
    s = ((edges[1:] + edges[:-1]) / 2)[:, None] + (edges[1:] - edges[:-1])[:, None] / 2 * xg[None, :]
    w = (edges[1:] - edges[:-1])[:, None] / 2 * wg[None, :]
    s, w = s.ravel(), w.ravel()
    return np.concatenate([s, beta - s[::-1]]), np.concatenate([w, w[::-1]])


def pi_matsubara_tau(X, e, qk, iq, beta, n_list, tau=None, chunk=64):
    """Pi(q, i nu_n) (len(n_list), Np, Np) by tau quadrature of (2/Nk) sum_k G~(k,-tau) o G~(k-q,tau)^T (Matsubara convention,
    degenerate pairs included). Independent of the transition list and of the Fermi-difference weights."""
    nk, Np, nb = X.shape
    if tau is None:
        emax = np.abs(e).max() * 2
        nmax = max(abs(int(n)) for n in n_list)
        tau, wt = gl_tau_grid(beta, emax, hmax=(2 * np.pi / beta * max(nmax, 1)) ** -1 * 2 * np.pi if nmax > 0 else None)
    else:
        tau, wt = tau
    nus = 2 * np.pi * np.asarray(n_list) / beta
    out = np.zeros((len(nus), Np, Np), complex)
    for i0 in range(0, len(tau), chunk):
        tt = tau[i0:i0 + chunk]
        acc = np.zeros((len(tt), Np, Np), complex)
        for ik in range(nk):
            ikmq = qk[iq, ik]
            lf, _ = fermi_logs(e[ik], beta)
            _, l1f = fermi_logs(e[ikmq], beta)
            a = np.exp(lf[None, :] + e[ik][None, :] * tt[:, None])           # f_n e^{e_n tau} <= 1
            b = -np.exp(l1f[None, :] - e[ikmq][None, :] * tt[:, None])       # -(1-f_m) e^{-e_m tau}
            Xa, Xb = X[ik], X[ikmq]
            A = np.einsum('pn,tn,qn->tpq', Xa, a, Xa.conj(), optimize=True)    # G~(k,-tau)_PQ
            B = np.einsum('qm,tm,pm->tpq', Xb, b, Xb.conj(), optimize=True)    # G~(k-q,tau)_QP stored at [t,p,q]
            acc += A * B
        acc *= 2.0 / nk
        ph = np.exp(1j * nus[:, None] * tt[None, :]) * wt[i0:i0 + chunk][None, :]
        out += (ph @ acc.reshape(len(tt), -1)).reshape(len(nus), Np, Np)
    return out


# ------------------------------------------------------------------------------------------------ finite-T Casida
def casida_from_transitions(tr, Zq):
    """Casida (full RPA) for W_dyn(q, z) = sum_s alpha_s bet_s/(z - lam_s) from transitions() (finite or zero T):
    sg = sign(F), columns sqrt|F| S, H = diag(E) + sg K, K = S^H Z S. Returns (lam, alpha, bet, info)."""
    S, E, F = tr['S'], tr['E'], tr['F']
    sg = np.sign(F)
    Ss = S * np.sqrt(np.abs(F))[None, :]
    ZS = Zq @ Ss
    K = Ss.conj().T @ ZS
    H = np.diag(E) + sg[:, None] * K
    lam, R = np.linalg.eig(H)
    Rinv = np.linalg.inv(R)
    alpha = ZS @ R
    bet = Rinv @ (sg[:, None] * ZS.conj().T)
    return lam.real, alpha, bet, dict(imlam=float(np.max(np.abs(lam.imag))) if len(lam) else 0.0,
                                      condR=float(np.linalg.cond(R)) if len(lam) else 1.0, Nt=len(E))


def casida_w(lam, alpha, bet, z):
    """W_dyn(q, z) (nz, Np, Np) from the Casida poles."""
    z = np.atleast_1d(np.asarray(z, complex))
    return np.array([(alpha / (zz - lam)[None, :]) @ bet for zz in z])


def dyson_w(Zq, Pi):
    """W = ([1 - Z Pi]^-1 - 1) Z per frequency."""
    I = np.eye(Zq.shape[0])
    return np.array([np.linalg.solve(I - Zq @ P, Zq) - Zq for P in Pi])


# ------------------------------------------------------------------------------------------------ Sigma_c, Eq. fT_sigma
def sigma_fT_blocks(X, e, qk, qminus, cas, beta, ik, lam_tol=0.0):
    """Pole blocks of Eq. fT_sigma for k: yields (E (np,), w (np,), A (nb, np), B (np, nb)) with
    Sigma_c(k, zeta) = sum over blocks of A diag(w/(zeta - E)) B; all weights w >= 0 (positive measure).
    cas[iq] = (lam, alpha, bet) Casida poles of q; positive poles of q (residues alpha_s bet_s) with (1 - f_m + n_j)/Nk at
    E = e_m + nu_j, positive poles of -q transposed with (f_m + n_j)/Nk at E = e_m - nu_j. beta None: T = 0."""
    nk, Np, nb = X.shape
    Xk = X[ik]
    for iq in range(nk):
        ikmq = qk[iq, ik]; Xm = X[ikmq]
        lam, alpha, bet = cas[iq]
        lm, am, bm = cas[qminus[iq]]
        p, pm = lam > lam_tol, lm > lam_tol
        nu, nup = lam[p], lm[pm]
        nj = np.zeros_like(nu) if beta is None else bose(nu, beta)
        njm = np.zeros_like(nup) if beta is None else bose(nup, beta)
        f = (e[ikmq] < 0).astype(float) if beta is None else fermi(e[ikmq], beta)
        for m in range(nb):
            x = Xm[:, m]
            A1 = (Xk.conj() * x[:, None]).T @ alpha[:, p]                     # (a, j): sum_P conj(X_Pa) x_P alpha_Pj
            B1 = bet[p] @ (x.conj()[:, None] * Xk)                             # (j, b)
            A2 = (Xk.conj() * x[:, None]).T @ bm[pm].T                         # w(-q)^T_PQ = bet'_jP alpha'_Qj
            B2 = am[:, pm].T @ (x.conj()[:, None] * Xk)
            yield e[ikmq][m] + nu, (1.0 - f[m] + nj) / nk, A1, B1
            yield e[ikmq][m] - nup, (f[m] + njm) / nk, A2, B2


def sigma_fT(X, e, qk, qminus, cas, beta, ik, zeta, lam_tol=0.0):
    """Exact finite-T Sigma_c(k, zeta) (nz, nb, nb) of Eq. fT_sigma (see sigma_fT_blocks)."""
    zeta = np.atleast_1d(np.asarray(zeta, complex))
    out = np.zeros((len(zeta), X.shape[2], X.shape[2]), complex)
    for E, w, A, B in sigma_fT_blocks(X, e, qk, qminus, cas, beta, ik, lam_tol):
        for iz, z in enumerate(zeta):
            out[iz] += (A * (w / (z - E))[None, :]) @ B
    return out


def sigma_fT_moments(X, e, qk, qminus, cas, beta, ik, wp, nmax, lam_tol=0.0):
    """Exact Cayley moments C^(n) = sum_poles R u(E)^n, u = (E + i wp)/(E - i wp), n = 0..nmax, about mu (E mu-relative)."""
    out = np.zeros((nmax + 1, X.shape[2], X.shape[2]), complex)
    for E, w, A, B in sigma_fT_blocks(X, e, qk, qminus, cas, beta, ik, lam_tol):
        u = (E + 1j * wp) / (E - 1j * wp); un = w.astype(complex)
        for n in range(nmax + 1):
            out[n] += (A * un[None, :]) @ B
            un = un * u
    return out


def g_tilde_iw(X, e, ik, z):
    """G~(k, z) = X(k) diag(1/(z - e)) X(k)^dagger (nz, Np, Np) (KS poles)."""
    z = np.atleast_1d(np.asarray(z, complex))
    return np.array([(X[ik] / (zz - e[ik])[None, :]) @ X[ik].conj().T for zz in z])


def sigma_nu0_term(X, e, qk, dW, beta, ik, iw):
    """Predicted Sigma^Mats(k, i w_n) - Sigma^an(k, i w_n) = -(1/beta)(1/Nk) sum_q X^+[G~(k-q, i w_n) o dW(q)] X (nz, nb, nb);
    dW[iq] = W^Mats(q, i nu_0) - W^an(q, 0) (Np, Np). iw: mu-relative i w_n."""
    nk = X.shape[0]
    iw = np.atleast_1d(np.asarray(iw, complex))
    out = np.zeros((len(iw), X.shape[2], X.shape[2]), complex)
    for iq in range(nk):
        if dW[iq] is None or not np.any(dW[iq]): continue
        G = g_tilde_iw(X, e, qk[iq, ik], iw)
        out += np.array([X[ik].conj().T @ (g * dW[iq]) @ X[ik] for g in G])
    return -out / (beta * nk)


def thermal_factor_check(beta, e_list, nu_list, iw, tau_grid=None):
    """Scalar second route for the Sigma thermal factors: int_0^beta e^{i w tau} [-G_m(tau) W_j(tau)] dtau with
    G_m(tau) = -(1-f_m) e^{-e_m tau}, W_j(tau) = -[(1+n_j) e^{-nu_j tau} + n_j e^{+nu_j tau}] (a +nu_j/-nu_j pole pair with
    residues +1/-1), vs the closed form (1-f+n)/(iw - e - nu) + (f+n)/(iw - e + nu). Returns max relative difference."""
    e_list, nu_list = np.asarray(e_list, float), np.asarray(nu_list, float)
    tau, wt = gl_tau_grid(beta, max(np.abs(e_list).max() + nu_list.max(), 1.0), hmax=0.5) if tau_grid is None else tau_grid
    f, n = fermi(e_list, beta), bose(nu_list, beta)
    _, l1f = fermi_logs(e_list, beta)
    err = 0.0
    for w in np.atleast_1d(iw):
        ph = np.exp(1j * w.imag * tau) * wt
        for a, ea in enumerate(e_list):
            Gt = -np.exp(l1f[a] - ea * tau)                                   # (ntau,)
            # n_j e^{+nu tau} computed as e^{nu tau}/(e^{beta nu}-1) = e^{nu (tau - beta)}/(1 - e^{-beta nu})
            Wt = -(np.exp(-np.outer(tau, nu_list)) / (-np.expm1(-beta * nu_list))[None, :]
                   + np.exp(np.outer(tau - beta, nu_list)) / (-np.expm1(-beta * nu_list))[None, :])
            q = ph @ (-(Gt[:, None] * Wt))
            ex = (1 - f[a] + n) / (w - ea - nu_list) + (f[a] + n) / (w - ea + nu_list)
            err = max(err, float(np.max(np.abs(q - ex) / np.abs(ex))))
    return err


# ------------------------------------------------------------------------------------------------ chemical potential (KS)
def electron_number(eig, mu, beta, k_weight=None):
    """N(mu) = 2 sum_k w_k sum_n f(eig - mu) for diagonal (KS) poles; eig (nk, nb) absolute."""
    eig = np.asarray(eig, float)
    wk = np.full(eig.shape[0], 1.0 / eig.shape[0]) if k_weight is None else np.asarray(k_weight) / np.sum(k_weight)
    return float(2.0 * (wk[:, None] * fermi(eig - mu, beta)).sum())


def mu_number(eig, nelec, beta, k_weight=None, tol=1e-13):
    """N(mu) = N_el by bisection (KS poles, absolute energies)."""
    lo, hi = np.min(eig) - 1.0, np.max(eig) + 1.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if electron_number(eig, mid, beta, k_weight) < nelec: lo = mid
        else: hi = mid
        if hi - lo < tol: break
    return 0.5 * (lo + hi)


def mu0_auto(eig, nelec, beta, thermal_tol=1e-8, k_weight=None):
    """Initial-state mu_0 (notes section 11.6, rule "auto" on the KS spectrum): the KS gap midpoint if beta * half-gap >
    c_T = ln(1/thermal_tol) (no pole in the window: T = 0 equivalent), else N(mu_0) = N_el. Returns (mu0, rule)."""
    eig = np.asarray(eig, float)
    nocc = int(round(nelec / 2))
    srt = np.sort(eig.ravel())
    # band-ordered edges (gapped insulator: the nocc lowest bands at every k are occupied)
    homo = np.sort(eig, axis=1)[:, nocc - 1].max(); lumo = np.sort(eig, axis=1)[:, nocc].min()
    mid = 0.5 * (homo + lumo)
    if beta is None or (lumo > homo and beta * 0.5 * (lumo - homo) > np.log(1.0 / thermal_tol)):
        return mid, 'gap'
    return mu_number(eig, nelec, beta, k_weight), 'number'


# ------------------------------------------------------------------------------------------------ S8b.3 hybrid: Matsubara density
def matsubara_set(beta, wmax):
    """Dense truncated fermionic set: (n (N,), i w_n (N,)) for n = 0..N-1 with w_{N-1} >= wmax (negative n by Hermitian
    conjugation, G(-i w_n) = G(i w_n)^dagger). N = ceil((wmax beta/pi - 1)/2) + 1 grows linearly with beta."""
    N = int(np.ceil((wmax * beta / np.pi - 1.0) / 2.0)) + 1
    n = np.arange(N)
    return n, 1j * np.pi * (2 * n + 1) / beta


def matsubara_tail4(beta, N):
    """(1/beta) sum_{|n| >= N, pairs (n, -n-1)} 2/w_n^4 = (2/beta) (beta/pi)^4 zeta(4, N + 1/2)/16 (Hurwitz zeta)."""
    from scipy.special import zeta
    return 2.0 / beta * (beta / np.pi) ** 4 * zeta(4.0, N + 0.5) / 16.0


def sigma_fT_hf(X, e, qk, qminus, cas, beta, ik, lam_tol=0.0):
    """Exact high-frequency moments of Eq. fT_sigma: S1 = sum_p R_p (total weight), S2 = sum_p R_p E_p (nb, nb)."""
    nb = X.shape[2]
    S1 = np.zeros((nb, nb), complex); S2 = np.zeros((nb, nb), complex)
    for E, w, A, B in sigma_fT_blocks(X, e, qk, qminus, cas, beta, ik, lam_tol):
        S1 += (A * w[None, :]) @ B; S2 += (A * (w * E)[None, :]) @ B
    return S1, S2


def density_upfold(H, E, U, beta, dmu=0.0):
    """Continuation-free exact density of G(i w_n) = [i w_n + dmu - H - Sigma(i w_n)]^-1 with Sigma(z) = U diag(1/(z - E))
    U^dagger (a positive pole list, U (nb, P)) held FIXED at the Matsubara points while mu moves (the convention of
    density_matsubara and of the imaginary-axis code): D = [f(H_up)]_{11}, H_up = [[H - dmu, U], [U^dagger, diag(E)]]
    (test oracle of density_matsubara; small P only)."""
    nb, P = U.shape
    Hu = np.zeros((nb + P, nb + P), complex)
    Hu[:nb, :nb] = H - dmu * np.eye(nb); Hu[:nb, nb:] = U; Hu[nb:, :nb] = U.conj().T; Hu[nb:, nb:] = np.diag(E)
    lam, V = np.linalg.eigh(Hu)
    Vt = V[:nb]
    return (Vt * fermi(lam, beta)[None, :]) @ Vt.conj().T


def density_matsubara(H, Siw, iw, beta, S1, S2, nelec=None, k_weight=None, dmu=None, tol=4e-16, maxit=200, tail6=True):
    """S8b.3 hybrid density and chemical potential (notes section 11.6 "Hybrid").
    H (nk, nb, nb): Hermitian mu-relative static Hamiltonian (H0 + F - mu_old); Siw (nk, N, nb, nb): Sigma_c(k, i w_n) at the
    dense set iw = i w_n, n = 0..N-1 (matsubara_set); S1, S2 (nk, nb, nb): Sigma_c(i w) = S1/(i w) + S2/(i w)^2 + ...
    G(i w; dmu) = [i w + dmu - H - Sigma_c(i w)]^-1, G_ref = [i w + dmu - H]^-1 (exact density f(H - dmu)),
      D(dmu) = f(H - dmu) + (1/beta) sum_{n=0}^{N-1} [dG(i w_n) + dG(i w_n)^dagger] + T4 B,
      dG = G - G_ref = S1/(i w)^3 + B/(i w)^4 + O(w^-5), B = S2 + (H - dmu) S1 + S1 (H - dmu); the odd orders cancel in the
      pair sum (n, -n-1) (Hermitian moments), T4 = matsubara_tail4(beta, N); truncation error O(w_N^-5).
      tail6: the w^-6 term of the remainder, c zeta(6, N + 1/2), is eliminated from the partial sums at N/2 and N (same
      inversions; exact shape of the term) -> error O(w_N^-7). Measured on the toy of tests/test_finite_t.py (beta 50):
      w_N 50 / 100 / 200 Ha: 2.6e-9 / 8.3e-11 / 2.6e-12 without, see the test for with.
    dmu None: N(dmu) = 2 sum_k w_k Tr D_k = nelec by bisection to the root in dmu (Sigma_c fixed; Tr G from the eigenvalues of
    H + Sigma_c(i w_n), computed once); else D at the given dmu. D at the root by full inversion.
    Returns (dmu, D (nk, nb, nb), N (full inversion), info dict(N_trace, nfreq, tail_max, wN))."""
    H = np.asarray(H); nk, nb = H.shape[0], H.shape[1]
    wk = np.full(nk, 1.0 / nk) if k_weight is None else np.asarray(k_weight, float) / np.sum(k_weight)
    iw = np.asarray(iw); Nf = len(iw)
    from scipy.special import zeta as hzeta
    Nh = Nf // 2
    T4, T4h = matsubara_tail4(beta, Nf), matsubara_tail4(beta, Nh)
    z6, z6h = hzeta(6.0, Nf + 0.5), hzeta(6.0, Nh + 0.5)
    def extrap(xN, xh):                    # x(N) = x_inf - c zeta(6, N + 1/2): eliminate c
        return xN + (xN - xh) / (z6h - z6) * z6 if tail6 else xN
    h, Vh = np.linalg.eigh(H)                                                 # (nk, nb)
    trS1 = np.einsum('kii->k', S1).real; trS2 = np.einsum('kii->k', S2).real
    trHS1 = np.einsum('kij,kji->k', H, S1).real
    def ntrace(dm, lam):
        """N(dm) from the eigenvalues lam (nk, N, nb) of H + Sigma_c(i w_n)."""
        Nt = 0.0
        for k in range(nk):
            z = iw[:, None] + dm
            d = ((1.0 / (z - lam[k])).sum(1) - (1.0 / (z - h[k][None, :])).sum(1)).real
            f0, tB = fermi(h[k] - dm, beta).sum(), trS2[k] + 2.0 * trHS1[k] - 2.0 * dm * trS1[k]
            trD = extrap(f0 + 2.0 * d.sum() / beta + T4 * tB, f0 + 2.0 * d[:Nh].sum() / beta + T4h * tB)
            Nt += 2.0 * wk[k] * trD
        return Nt
    def dens(dm):
        D = np.zeros((nk, nb, nb), complex); tmax = 0.0
        I = np.eye(nb)
        for k in range(nk):
            z = (iw + dm)[:, None, None]
            G = np.linalg.inv(z * I - H[k][None] - Siw[k])
            G0 = (Vh[k][None] / (z[:, :, 0] - h[k][None, :])[:, None, :]) @ Vh[k].conj().T
            dG = G - G0; dGh = dG[:Nh].sum(0); dG = dG.sum(0)
            Hd = H[k] - dm * I
            B = S2[k] + Hd @ S1[k] + S1[k] @ Hd
            tail = T4 * B
            D0 = (Vh[k] * fermi(h[k] - dm, beta)[None, :]) @ Vh[k].conj().T
            D[k] = extrap(D0 + (dG + dG.conj().T) / beta + tail, D0 + (dGh + dGh.conj().T) / beta + T4h * B)
            tmax = max(tmax, float(np.abs(tail).max()))
        return D, tmax
    info = dict(nfreq=Nf, wN=float(np.abs(iw[-1])))
    if dmu is None:
        lam = np.array([np.linalg.eigvals(H[k][None] + Siw[k]) for k in range(nk)])     # (nk, N, nb)
        allh = np.concatenate([h.ravel(), lam.real.ravel()])
        lo, hi = allh.min() - 1.0 - 50.0 / beta, allh.max() + 1.0 + 50.0 / beta
        Nlo = ntrace(lo, lam) - nelec
        for _ in range(maxit):
            mid = 0.5 * (lo + hi)
            Nm = ntrace(mid, lam) - nelec
            if Nm == 0.0 or hi - lo < tol * max(1.0, abs(mid)):
                break
            if (Nm < 0) == (Nlo < 0):
                lo, Nlo = mid, Nm
            else:
                hi = mid
        dmu = mid; info['N_trace'] = Nm + nelec
    D, tmax = dens(dmu)
    Nfull = float(2.0 * np.sum(wk * np.einsum('kii->k', D).real))
    info['tail_max'] = tmax
    return dmu, D, Nfull, info
