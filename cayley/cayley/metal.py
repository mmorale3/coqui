"""S8c metals helpers (go/no-go study and svo222 references; notes section 12), independent of the line machinery.

* casida_hermitian(tr, Zq): the finite-T Casida of finite_t.casida_from_transitions through the Hermitian form of the
  pseudo-Hermitian RPA problem. H = diag(E) + sg K = sigma A with sigma = diag(sign E) (= sign F: every pair has E F > 0) and
  A = |E| + K Hermitian positive definite (K = S~^H Z S~ PSD, |E| > 0). With A = L L^H, L^H sigma L y = lam y is Hermitian,
  the right eigenvectors are R = L^{-H} Y and R^{-1} = Y^H L^H. Same output (lam, alpha, bet, info) as the general route
  (scipy eigh 'evr' instead of a non-Hermitian eig: ~3x faster, real eigenvalues by construction); falls back to it if the
  Cholesky fails.
* cached(fn, cache_dir): wraps a casida(tr, Zq) function with a disk cache keyed by a hash of (E, F, S, Zq).
* SigmaFT: exact Sigma_c(k, zeta) of Eq. fT_sigma (finite_t.sigma_fT_blocks) in blocked gemm form: for every q and term the
  pole list of k is (T1 (nb, P), E (P), w (P), T2 (P, nb)) with Sigma_c = sum_p T1[:, p] w_p g(E_p) T2[p, :]; values at many
  zeta and the Cayley moments for several omega_p are accumulated in ONE pass over the q's (the pole lists of svo222 have
  ~3e6 poles per k: the per-block python loops of finite_t are too slow there).
* pi_ph_asymmetry(trs, beta, n_list): max_q,n |Pi(q, i nu_n) - Pi(q, i nu_n)^T| / max|Pi| of the transition sum (the PH
  symmetry assumed by CoQui's half tau grid, memory note "imag-axis Pi PH-symmetry transpose bug").
"""
import os, hashlib, time, numpy as np, scipy.linalg as sl
from cayley import finite_t as ft


def casida_hermitian(tr, Zq):
    S, E, F = tr['S'], tr['E'], tr['F']
    if len(E) == 0:
        return ft.casida_from_transitions(tr, Zq)
    sg = np.sign(F)
    if not np.all(sg == np.sign(E)):
        return ft.casida_from_transitions(tr, Zq)
    Ss = S * np.sqrt(np.abs(F))[None, :]
    ZS = Zq @ Ss                                                         # (Np, Nt)
    A = Ss.conj().T @ ZS
    A = 0.5 * (A + A.conj().T)
    A[np.diag_indices_from(A)] += np.abs(E)
    try:
        L = sl.cholesky(A, lower=True, overwrite_a=True, check_finite=False)
    except np.linalg.LinAlgError:
        return ft.casida_from_transitions(tr, Zq)
    del A
    M = L.conj().T @ (sg[:, None] * L)
    M = 0.5 * (M + M.conj().T)
    lam, Y = sl.eigh(M, driver='evr', overwrite_a=True, check_finite=False)
    del M
    P = sl.solve_triangular(L, ZS.conj().T, lower=True, check_finite=False).conj().T   # ZS L^{-H}
    alpha = P @ Y
    bet = Y.conj().T @ (L.conj().T @ (sg[:, None] * ZS.conj().T))
    return lam, alpha, bet, dict(imlam=0.0, condR=float('nan'), Nt=len(E), route='hermitian')


def _key(tr, Zq):
    h = hashlib.sha1()
    for a in (tr['E'], tr['F'], tr['S'], Zq):
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()[:20]


def cached(fn, cache_dir, verbose=True):
    """casida(tr, Zq) with a disk cache (npz per key) in cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)

    def wrapped(tr, Zq):
        key = _key(tr, Zq)
        fnm = os.path.join(cache_dir, f'casida_{key}.npz')
        if os.path.exists(fnm):
            d = np.load(fnm, allow_pickle=True)
            return d['lam'], d['alpha'], d['bet'], d['info'][()]
        t0 = time.time()
        lam, alpha, bet, info = fn(tr, Zq)
        info = dict(info); info['seconds'] = time.time() - t0
        tmp = fnm + f'.tmp{os.getpid()}.npz'
        np.savez(tmp, lam=lam, alpha=alpha, bet=bet, info=np.array(info, dtype=object))
        os.replace(tmp, fnm)
        if verbose:
            print(f"    casida Nt {info.get('Nt')} ({info.get('route', 'eig')}) {info['seconds']:.0f}s -> {fnm}", flush=True)
        return lam, alpha, bet, info
    return wrapped


class SigmaFT:
    """Exact Eq. fT_sigma for one k, evaluated in one pass over the q's.
    evaluate(zeta=..., wps=..., nmax=...) -> dict(zeta: (nz, nb, nb), moments: {wp: (nmax+1, nb, nb)})."""

    def __init__(self, X, e, qk, qminus, cas, beta, ik, lam_tol=0.0):
        self.X, self.e, self.qk, self.qminus, self.cas, self.beta, self.ik, self.lam_tol = X, e, qk, qminus, cas, beta, ik, lam_tol

    def terms(self):
        X, e, qk, cas, beta, ik = self.X, self.e, self.qk, self.cas, self.beta, self.ik
        nk, Np, nb = X.shape
        Xk = X[ik]
        for iq in range(nk):
            ikmq = qk[iq, ik]; Xm = X[ikmq]
            f = (e[ikmq] < 0).astype(float) if beta is None else ft.fermi(e[ikmq], beta)
            Y = (Xk.conj()[:, :, None] * Xm[:, None, :]).reshape(Np, nb * nb)                 # (P, a m)
            Yb = (Xm.conj()[:, :, None] * Xk[:, None, :]).reshape(Np, nb * nb)                # (Q, m b)
            for which in (0, 1):
                lam, alpha, bet = cas[iq] if which == 0 else cas[self.qminus[iq]]
                p = lam > self.lam_tol
                nu = lam[p]
                if nu.size == 0: continue
                nj = np.zeros_like(nu) if beta is None else ft.bose(nu, beta)
                if which == 0:   # w_j(q) = alpha_j bet_j : (1 - f_m + n_j) at e_m + nu_j
                    L, R = alpha[:, p], bet[p]
                    E = e[ikmq][:, None] + nu[None, :]
                    w = (1.0 - f[:, None] + nj[None, :]) / nk
                else:            # w_j(-q)^T = bet'_j^T alpha'_j^T : (f_m + n_j) at e_m - nu_j
                    L, R = bet[p].T, alpha[:, p].T
                    E = e[ikmq][:, None] - nu[None, :]
                    w = (f[:, None] + nj[None, :]) / nk
                npj = nu.size
                T1 = (Y.T @ L).reshape(nb, nb * npj)                                          # [a, (m, j)]
                T2 = (R @ Yb).reshape(npj, nb, nb).transpose(1, 0, 2).reshape(nb * npj, nb)   # [(m, j), b]
                yield T1, E.ravel(), w.ravel(), T2

    def evaluate(self, zeta=None, wps=(), nmax=0, zeta_binned=None, prune=1e-16, blk=6000, bin_delta=1e-3, bin_E0=1e-4):
        """Exact values at zeta and Cayley moments n = 0..nmax for each wp in wps (one gemm per pole block with all rows),
        plus values at zeta_binned from a binned copy of the pole measure (4-point Lagrange weights on the graded grid
        E_i = bin_E0 sinh(bin_delta i): relative error ~ 4 (h/d)^4, h = bin_delta max(|E|, bin_E0), d = distance of zeta from
        the pole: <= 1e-5 for Im zeta >= 0.01 near mu; use it for the dense real-axis grids only). Poles whose
        w |T1_p| |T2_p| is below prune x (term total) are skipped (<= 2e-13 of the measure on svo222)."""
        nb = self.X.shape[2]
        zeta = np.zeros(0, complex) if zeta is None else np.atleast_1d(np.asarray(zeta, complex))
        zb = np.zeros(0, complex) if zeta_binned is None else np.atleast_1d(np.asarray(zeta_binned, complex))
        nz, nm = len(zeta), (nmax + 1) * len(wps)
        acc = np.zeros((nz + nm, nb * nb), complex)
        if len(zb):
            from scipy.sparse import csr_matrix
            imax = int(np.ceil(np.arcsinh(20.0 / bin_E0) / bin_delta)) + 4
            Eg = bin_E0 * np.sinh(bin_delta * np.arange(-imax, imax + 1))
            R = np.zeros((len(Eg), nb * nb), complex)
        for T1, E, w, T2 in self.terms():
            mag = w * np.linalg.norm(T1, axis=0) * np.linalg.norm(T2, axis=1)
            keep = np.nonzero(mag > prune * mag.sum())[0]
            for b0 in range(0, len(keep), blk):
                ib = keep[b0:b0 + blk]
                Eb, wb = E[ib], w[ib]
                Q = (T1[:, ib].T[:, :, None] * T2[ib][:, None, :]).reshape(len(ib), nb * nb)
                G = np.empty((nz + nm, len(ib)), complex)
                if nz: G[:nz] = wb[None, :] / (zeta[:, None] - Eb[None, :])
                r = nz
                for wp in wps:
                    u = (Eb + 1j * wp) / (Eb - 1j * wp); g = wb.astype(complex)
                    for n in range(nmax + 1):
                        G[r] = g; g = g * u; r += 1
                acc += G @ Q
                if len(zb):
                    sidx = np.arcsinh(Eb / bin_E0) / bin_delta + imax
                    i0 = np.clip(np.floor(sidx).astype(int) - 1, 0, len(Eg) - 4)
                    idx = i0[:, None] + np.arange(4)[None, :]
                    Ek = Eg[idx]
                    L = np.ones((len(ib), 4))
                    for j in range(4):
                        for k in range(4):
                            if k != j: L[:, j] *= (Eb - Ek[:, k]) / (Ek[:, j] - Ek[:, k])
                    Wm = csr_matrix(((L * wb[:, None]).ravel(), (idx.ravel(), np.repeat(np.arange(len(ib)), 4))), shape=(len(Eg), len(ib)))
                    R += Wm @ Q
        acc = acc.reshape(nz + nm, nb, nb)
        out = dict(zeta=acc[:nz], moments={wp: acc[nz + i * (nmax + 1): nz + (i + 1) * (nmax + 1)] for i, wp in enumerate(wps)})
        if len(zb):
            out['zeta_binned'] = ((1.0 / (zb[:, None] - Eg[None, :])) @ R).reshape(len(zb), nb, nb)
        return out


def pi_ph_asymmetry(trs, beta, n_list=(0, 1, 2, 5, 20)):
    """max over q and nu_n of |Pi - Pi^T| / max|Pi| (transition sum; n = 0 with the Matsubara dPi of the degenerate pairs)."""
    num = den = 0.0
    for tr in trs:
        nus = 2j * np.pi * np.asarray(n_list) / beta
        P = ft.pi_transition(tr, nus)
        if 0 in n_list:
            P[list(n_list).index(0)] += ft.pi_nu0_extra(tr, beta)
        num = max(num, float(np.abs(P - np.transpose(P, (0, 2, 1))).max()))
        den = max(den, float(np.abs(P).max()))
    return num / den


def upfold_block_fast(C, K, wp, mu=0.0, tol_c0=1e-12, tol_gram=1e-12, tol_svd=1e-12, nphi=72, reject_unity=1e-6, phase_refine=True):
    """cayley.upfold.upfold_block with the same realization and the same terminal-phase objective, evaluated without an
    eigensolve per phase: the held-out moment of U(phi) = U1 + x U0 (x = e^{i phi}) is the degree-(K+1) polynomial
    P(x) = R U(x)^{K+1} R^dagger, whose K+2 coefficients come from K+2 samples on the roots of unity (FFT); the scan and the
    golden-section refinement then cost O(K N^2) per phase, and ONE complex Schur form is taken at the optimum (r = 40,
    K = 32: ~30 s instead of ~100 Schur forms of 1300 x 1300). The admissibility test min|u - 1| >= reject_unity is applied
    at the optimum only; if it fails the original scanning routine is called."""
    from scipy.linalg import schur
    from cayley.upfold import normalize_c0, block_toeplitz, upfold_block
    from cayley.maps import inv_cayley
    C = np.asarray(C)
    B, Bp, Chat = normalize_c0(C, tol_c0)
    r = B.shape[1]
    T = block_toeplitz(Chat, K)
    lam, V = np.linalg.eigh(T)
    keep = lam > tol_gram * lam.max()
    Xg = np.sqrt(lam[keep])[:, None] * V[:, keep].conj().T
    Dm, Dp = Xg[:, :K * r], Xg[:, r:(K + 1) * r]
    P, s, Qh = np.linalg.svd(Dp @ Dm.conj().T)
    r1 = int((s > tol_svd * s[0]).sum())
    U1 = P[:, :r1] @ Qh[:r1]
    U0 = P[:, r1:] @ Qh[r1:]
    nfree = U0.shape[0] - r1 if U0.size else 0
    R = B @ Xg[:, :r].conj().T
    Cheld = C[K + 1]
    nrm = 1 + np.linalg.norm(Cheld)
    if nfree > 0:
        xs = np.exp(2j * np.pi * np.arange(K + 2) / (K + 2))
        Ps = []
        for x in xs:
            M = R
            U = U1 + x * U0
            for _ in range(K + 1): M = M @ U
            Ps.append(M @ R.conj().T)
        coef = np.fft.fft(np.array(Ps), axis=0) / (K + 2)          # P(x) = sum_j c_j x^j, c_j = (1/n) sum_m P(x_m) x_m^{-j}
        pw = np.arange(K + 2)
        f = lambda p: np.linalg.norm(np.tensordot(np.exp(1j * pw * p), coef, axes=1) - Cheld) / nrm
        phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
        errs = np.array([f(p) for p in phis])
        phi0 = phis[int(np.argmin(errs))]
        if phase_refine:
            a, b = phi0 - 2 * np.pi / nphi, phi0 + 2 * np.pi / nphi
            gr = (np.sqrt(5) - 1) / 2
            c, d = b - gr * (b - a), a + gr * (b - a); fc, fd = f(c), f(d)
            for _ in range(30):
                if fc < fd: b, d, fd = d, c, fc; c = b - gr * (b - a); fc = f(c)
                else: a, c, fc = c, d, fd; d = a + gr * (b - a); fd = f(d)
            phi0 = 0.5 * (a + b)
    else:
        phi0 = 0.0
    U = U1 + (np.exp(1j * phi0) * U0 if nfree else 0.0)
    Tu, Zs = schur(U, output='complex')
    u = np.diag(Tu)
    if nfree > 0 and np.min(np.abs(u - 1)) < reject_unity:
        return upfold_block(C, K, wp, mu=mu, tol_c0=tol_c0, tol_gram=tol_gram, tol_svd=tol_svd, nphi=nphi,
                            reject_unity=reject_unity, phase_refine=phase_refine)
    W = R @ Zs
    d = inv_cayley(u, wp, mu)
    order = np.argsort(d)
    return d[order], W[:, order]
