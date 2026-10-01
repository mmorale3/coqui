"""Block Toeplitz-as-Gram unitary realization of Cayley moments (Allen & Booth Sec. IV.0.2-IV.0.3)."""
import numpy as np
from scipy.linalg import schur
from .maps import inv_cayley


def block_toeplitz(C, K):
    """Hermitian block Toeplitz T_K with [T]_ij = C^(j-i) (j >= i), C^(i-j)^dagger (j < i); C: (>=K+1, N, N)."""
    N = C.shape[1]
    T = np.zeros(((K + 1) * N, (K + 1) * N), complex)
    for i in range(K + 1):
        for j in range(K + 1):
            T[i * N:(i + 1) * N, j * N:(j + 1) * N] = C[j - i] if j >= i else C[i - j].conj().T
    return T


def normalize_c0(C, tol=1e-12):
    """C^(0) = B B^dagger (rank-revealing); returns B (N x r), B^+ (r x N), Chat = B^+ C B^+dagger with Chat^(0) = I_r."""
    C0 = 0.5 * (C[0] + C[0].conj().T)
    lam, V = np.linalg.eigh(C0)
    keep = lam > tol * lam.max()
    B = V[:, keep] * np.sqrt(lam[keep])
    Bp = (V[:, keep] / np.sqrt(lam[keep])).conj().T
    Chat = np.array([Bp @ c @ Bp.conj().T for c in C])
    return B, Bp, Chat


def upfold_block(C, K, wp, mu=0.0, tol_c0=1e-12, tol_gram=1e-12, tol_svd=1e-12, nphi=72,
                 reject_unity=1e-6, phase_refine=True, return_info=False):
    """Unitary realization of block moments C[0..K]; C[K+1] is the held-out moment that fixes the terminal phase.

    C: (>= K+2, N, N) Hermitian-sequence moments (C^(-n) = C^(n)^dagger implied).
    tol_gram: eigenvalues of the normalized block-Toeplitz Gram matrix below tol_gram*lam_max are dropped. This is the
              regularization that keeps noisy moments usable (set to ~10x the relative moment error).
    Returns d (poles, real, (Np,)), W (N x Np couplings): Sigma_c(z) = sum_l W[:,l] W[:,l]^dagger/(z - d_l).
    """
    C = np.asarray(C)
    N = C.shape[1]
    B, Bp, Chat = normalize_c0(C, tol_c0)
    r = B.shape[1]
    T = block_toeplitz(Chat, K)
    if not np.all(np.isfinite(T)) or T.size == 0:
        raise ValueError("upfold_block: moments are empty or non-finite (C0 rank %d)" % r)
    lam, V = np.linalg.eigh(T)
    lam_max = lam.max()
    keep = lam > tol_gram * lam_max
    X = np.sqrt(lam[keep])[:, None] * V[:, keep].conj().T            # T = X^dagger X,  X: Nr x (K+1) r
    Nr = X.shape[0]
    Dm, Dp = X[:, :K * r], X[:, r:(K + 1) * r]
    P, s, Qh = np.linalg.svd(Dp @ Dm.conj().T)
    r1 = int((s > tol_svd * s[0]).sum())
    P1, Q1, P0, Q0 = P[:, :r1], Qh[:r1].conj().T, P[:, r1:], Qh[r1:].conj().T
    R = B @ X[:, :r].conj().T                                          # N x Nr
    Cheld = C[K + 1]
    nfree = P0.shape[1]

    def realize(phi):
        U = P1 @ Q1.conj().T + (np.exp(1j * phi) * (P0 @ Q0.conj().T) if nfree else 0.0)
        # U is normal (unitary up to rounding): its complex Schur form is diagonal with a unitary Z
        Tu, Z = schur(U, output='complex')
        u = np.diag(Tu)
        W = R @ Z
        err = np.linalg.norm((W * u ** (K + 1)) @ W.conj().T - Cheld) / (1 + np.linalg.norm(Cheld))
        return err, u, W

    if nfree == 0:
        best = realize(0.0) + (0.0,)
    else:
        phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
        errs = []
        for phi in phis:
            e, u, W = realize(phi)
            errs.append(np.inf if np.min(np.abs(u - 1)) < reject_unity else e)
        errs = np.array(errs)
        if not np.isfinite(errs).any():
            raise RuntimeError("no admissible terminal phase")
        i0 = int(np.argmin(errs)); phi0 = phis[i0]
        if phase_refine:                                                   # golden-section refine around the grid minimum
            a, b = phi0 - 2 * np.pi / nphi, phi0 + 2 * np.pi / nphi
            f = lambda p: realize(p)[0]
            gr = (np.sqrt(5) - 1) / 2
            c, d = b - gr * (b - a), a + gr * (b - a); fc, fd = f(c), f(d)
            for _ in range(30):
                if fc < fd: b, d, fd = d, c, fc; c = b - gr * (b - a); fc = f(c)
                else: a, c, fc = c, d, fd; d = a + gr * (b - a); fd = f(d)
            phi0 = 0.5 * (a + b)
        best = realize(phi0) + (phi0,)
    err, u, W, phi = best
    d = inv_cayley(u, wp, mu)
    order = np.argsort(d)
    d, W, u = d[order], W[:, order], u[order]
    if return_info:
        info = dict(rank_c0=r, rank_gram=Nr, n_free=nfree, phi=phi, heldout_err=err, gram_lam_min=lam.min() / lam_max,
                    moment_err=[np.linalg.norm((W * u ** n) @ W.conj().T - C[n]) / np.linalg.norm(C[0]) for n in range(K + 2)])
        return d, W, info
    return d, W


def upfold_sectors(C_less, C_greater, K, wp, mu=0.0, **kw):
    """Upfold hole and particle sectors separately (paper's default) and concatenate the pole sets."""
    dl, Wl = upfold_block(C_less, K, wp, mu, **kw)
    dg, Wg = upfold_block(C_greater, K, wp, mu, **kw)
    return np.concatenate([dl, dg]), np.concatenate([Wl, Wg], axis=1)
