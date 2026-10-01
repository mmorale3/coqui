# Copied from the user's real_axis_GW/toys/dlr.py (2026-09-28) for use inside the cayley prototype.
"""Discrete Lehmann representation with geometric fine grids (handles Lambda = beta*wmax up to ~1e6).

Fermionic: K(tau,w) = exp(-w tau)/(1+exp(-beta w)),   X(tau) = sum_p c_p K(tau,w_p),  X(iw_n) = sum_p c_p/(w_p - iw_n)
Bosonic:   K(tau,w) = w exp(-w tau)/(1-exp(-beta w)) (w=0 -> 1/beta),  X(i nu_m) = sum_p c_p w_p/(w_p - i nu_m) (w_p=0 -> delta_m0)
so that a spectral density A(w) (fermion) or b(w)=B(w)/w (boson) enters as c = -A(w_p)*dw (X(tau) = -int A K).
"""
import numpy as np
from scipy.linalg import qr

def fermi_kernel(E, t, beta):
    E = np.asarray(E, float)[None, :]; t = np.asarray(t, float)[:, None]
    return np.exp(-E * t - np.logaddexp(0.0, -beta * E))

def bose_kernel_w(nu, t, beta):
    """nu e^{-nu t}/(1-e^{-beta nu}), regular at nu=0 (-> 1/beta)."""
    nu = np.asarray(nu, float)[None, :]; t = np.asarray(t, float)[:, None]
    N = np.broadcast_to(nu, np.broadcast(nu, t).shape); T = np.broadcast_to(t, N.shape)
    out = np.empty(N.shape)
    pos = N > 0; neg = N < 0; zer = N == 0
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        out[pos] = N[pos] * np.exp(-N[pos] * T[pos] - np.log1p(-np.exp(-beta * N[pos])))
        out[neg] = -N[neg] * np.exp(-N[neg] * T[neg] + beta * N[neg] - np.log1p(-np.exp(beta * N[neg])))
    out[zer] = 1.0 / beta
    return out

class DLRg:
    def __init__(self, beta, wmax, eps, kind='fermi', ratio=1.02, nmats_extra=200):
        self.beta, self.wmax, self.eps, self.kind = beta, wmax, eps, kind
        Lam = beta * wmax
        # fine omega grid: 0 and +-geometric from wmax/(10 Lam) to wmax
        n = int(np.ceil(np.log(10 * Lam) / np.log(ratio)))
        g = wmax * np.exp(np.linspace(np.log(1 / (10 * Lam)), 0.0, n))
        w = np.concatenate([-g[::-1], [0.0], g])
        # fine tau grid: geometric from beta/(10 Lam) to beta/2, mirrored
        m = int(np.ceil(np.log(5 * Lam) / np.log(ratio)))
        gt = beta * np.exp(np.linspace(np.log(1 / (10 * Lam)), np.log(0.5), m))
        tau = np.concatenate([[0.0], gt, beta - gt[::-1][1:], [beta]])
        K = self.K_tau(tau, w)
        _, R, piv = qr(K, mode='economic', pivoting=True)
        d = np.abs(np.diag(R)); r = int(np.sum(d > eps * d[0]))
        self.w = np.sort(w[piv[:r]]); self.r = r
        Kw = self.K_tau(tau, self.w)
        _, R2, piv2 = qr(Kw.T, mode='economic', pivoting=True)
        self.tau = np.sort(tau[piv2[:r]]); self.Ktau = self.K_tau(self.tau, self.w)
        nmax = int(Lam / np.pi) + nmats_extra
        nn = np.arange(-nmax, nmax + 1)
        self.wn_all = (2 * nn + (1 if kind == 'fermi' else 0)) * np.pi / beta
        Kiw = self.K_iw(self.wn_all, self.w)
        _, R3, piv3 = qr(Kiw.T, mode='economic', pivoting=True)
        self.wn = np.sort(self.wn_all[piv3[:r]]); self.Kiw = self.K_iw(self.wn, self.w)
    def K_tau(self, tau, w):
        return fermi_kernel(w, tau, self.beta) if self.kind == 'fermi' else bose_kernel_w(w, tau, self.beta)
    def K_iw(self, wn, w):
        z = 1j * np.asarray(wn, float)[:, None]; wp = np.asarray(w, float)[None, :]
        if self.kind == 'fermi': return 1.0 / (wp - z)
        out = np.where(wp != 0, wp / np.where(wp != 0, wp - z, 1.0), 0.0) + 0j
        out[:, np.asarray(w) == 0] = (np.abs(np.asarray(wn)) < 1e-12)[:, None] * 1.0
        return out
    # fits: values may have trailing dims (…): solve along axis 0
    def coefs_from_tau(self, X, tau=None):
        """least-squares fit of DLR coefficients from values at tau points (default: own nodes). X: (ntau, ...)"""
        K = self.Ktau if tau is None else self.K_tau(np.asarray(tau), self.w)
        sh = X.shape; Xm = X.reshape(sh[0], -1)
        c = np.linalg.lstsq(K, Xm, rcond=None)[0] if K.shape[0] != K.shape[1] or tau is not None else np.linalg.solve(K, Xm)
        return c.reshape((self.r,) + sh[1:])
    def coefs_from_iw(self, X, wn=None):
        K = self.Kiw if wn is None else self.K_iw(np.asarray(wn), self.w)
        sh = X.shape; Xm = X.reshape(sh[0], -1)
        c = np.linalg.lstsq(K, Xm, rcond=None)[0] if wn is not None else np.linalg.solve(K, Xm)
        return c.reshape((self.r,) + sh[1:])
    def eval_tau(self, c, tau):
        K = self.K_tau(np.atleast_1d(tau), self.w); sh = c.shape
        return (K @ c.reshape(sh[0], -1)).reshape((K.shape[0],) + sh[1:])
    def eval_iw(self, c, wn):
        K = self.K_iw(np.atleast_1d(wn), self.w); sh = c.shape
        return (K @ c.reshape(sh[0], -1)).reshape((K.shape[0],) + sh[1:])

if __name__ == "__main__":
    import time
    for beta, wmax in ((40., 45.), (1000., 20.), (3000., 20.)):
        t0 = time.time(); F = DLRg(beta, wmax, 1e-12, 'fermi'); B = DLRg(beta, wmax, 1e-12, 'bose')
        # test: fermionic function with poles at random energies (incl. near 0) evaluated at own nodes, fit, evaluate elsewhere
        rng = np.random.default_rng(0); E = rng.uniform(-0.8 * wmax, 0.8 * wmax, 50); E[:5] *= 1e-3; a = rng.uniform(0, 1, 50); a /= a.sum()
        tt = np.linspace(0, beta, 3001)
        ex = lambda t: -(fermi_kernel(E, t, beta) @ a)
        c = F.coefs_from_tau(ex(F.tau)); errt = np.max(np.abs(F.eval_tau(c, tt) - ex(tt)))
        exiw = lambda wn: -np.sum(a[None, :] / (1j * wn[:, None] - E[None, :]), axis=1)
        c2 = F.coefs_from_iw(exiw(F.wn)); erriw = np.max(np.abs(F.eval_tau(c2, tt) - ex(tt)))
        # bosonic: odd spectral function B(nu) = nu b(nu), poles at +-nu_j
        nu = rng.uniform(0.01 * wmax, 0.8 * wmax, 30); bw = rng.uniform(0, 1, 30)
        exb = lambda t: -(bose_kernel_w(nu, t, beta) @ bw + bose_kernel_w(-nu, t, beta) @ bw)
        cb = B.coefs_from_tau(exb(B.tau)); errb = np.max(np.abs(B.eval_tau(cb, tt) - exb(tt)))
        exbiw = lambda wn: -np.sum(bw[None, :] * (nu[None, :] / (nu[None, :] - 1j * wn[:, None]) - nu[None, :] / (-nu[None, :] - 1j * wn[:, None])), axis=1)
        cb2 = B.coefs_from_iw(exbiw(B.wn)); errb2 = np.max(np.abs(B.eval_tau(cb2, tt) - exb(tt)))
        print(f"beta={beta} wmax={wmax} Lambda={beta*wmax:.0f}: r_f={F.r} r_b={B.r}  fermi err(tau-fit)={errt:.1e} err(iw-fit)={erriw:.1e}  bose err(tau-fit)={errb:.1e} err(iw-fit)={errb2:.1e}   [{time.time()-t0:.1f}s]")
