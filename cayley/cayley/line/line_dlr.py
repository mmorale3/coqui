"""Real-pole basis for functions analytic off the real axis, sampled on the tilted line zeta = mu + t e^{i theta}.

Data live on the two upper rays (angles theta and pi - theta, after Schwarz reflection of the lower half line). Poles are
selected by column-pivoted QR of K(zeta_i, w_j) = 1/(zeta_i - w_j) at tolerance eps; line nodes by row-pivoted QR.
Energies are measured from the line centre mu (pass mu-relative data in, get mu-relative poles out)."""
import numpy as np
from scipy.linalg import qr


class LineBasis:
    def __init__(self, theta, lam, eps=1e-8, gap=(0.0, 0.0), tmin=None, tmax=None, nline=1200, npole=1500):
        """theta: tilt angle (rad); lam: pole range |w| <= lam; gap = (D_minus, D_plus): no poles in (-D_minus, D_plus);
        tmin/tmax: |t| range of the dense line grid (defaults: 1e-4*lam .. 20*lam)."""
        self.theta, self.lam, self.eps, self.gap = theta, lam, eps, gap
        tmin = 1e-4 * lam if tmin is None else tmin; tmax = 20 * lam if tmax is None else tmax
        t = np.exp(np.linspace(np.log(tmin), np.log(tmax), nline))
        self.zeta_dense = np.concatenate([t * np.exp(1j * theta), t * np.exp(1j * (np.pi - theta))])
        Dm, Dp = gap
        gp = np.exp(np.linspace(np.log(max(Dp, 1e-4 * lam)), np.log(lam), npole // 2))
        gm = np.exp(np.linspace(np.log(max(Dm, 1e-4 * lam)), np.log(lam), npole // 2))
        w = np.concatenate([-gm[::-1], gp])
        if Dm <= 0 and Dp <= 0: w = np.concatenate([-gm[::-1], [0.0], gp])
        K = 1.0 / (self.zeta_dense[:, None] - w[None, :])
        Kn = K / np.linalg.norm(K, axis=0, keepdims=True)
        _, R, piv = qr(Kn, mode='economic', pivoting=True)
        d = np.abs(np.diag(R)); r = int((d > eps * d[0]).sum())
        self.w = np.sort(w[piv[:r]]); self.r = r
        Kw = 1.0 / (self.zeta_dense[:, None] - self.w[None, :])
        _, R2, piv2 = qr(Kw.T, mode='economic', pivoting=True)
        self.zeta = self.zeta_dense[np.sort(piv2[:r])]             # line nodes (mu-relative)
        self.K = self.kernel(self.zeta)
        self.pos = self.w > 0

    def kernel(self, zeta):
        return 1.0 / (np.asarray(zeta, complex)[:, None] - self.w[None, :])

    def fit(self, zeta, X, rcond=None):
        """LS coefficients c (r, ...) from samples X (nz, ...) at mu-relative points zeta (nz,)."""
        K = self.kernel(zeta); sh = X.shape
        c = np.linalg.lstsq(K, X.reshape(sh[0], -1), rcond=rcond)[0]
        return c.reshape((self.r,) + sh[1:])

    def eval(self, c, zeta):
        """X(zeta) = sum_l c_l/(zeta - w_l) at arbitrary mu-relative complex zeta; (nz, ...)."""
        K = self.kernel(zeta); sh = c.shape
        return (K @ c.reshape(sh[0], -1)).reshape((K.shape[0],) + sh[1:])

    def split(self, c):
        """(c_hole, c_particle): coefficients of poles below / above the centre (sector split)."""
        return c[~self.pos], c[self.pos]

    def __repr__(self):
        return f"LineBasis(theta={np.degrees(self.theta):.1f} deg, lam={self.lam}, eps={self.eps:g}, gap={self.gap}, rank={self.r})"


class BosonicLineBasis:
    """Odd-symmetric real-pole basis for bosonic functions W with B(-nu) = -B(nu)^T, i.e.
         W_PQ(zeta) = sum_j [ w_j,PQ/(zeta - nu_j) - w_j,QP/(zeta + nu_j) ],   nu_j > 0.
    The hole part is tied to the particle part (no sector-split ambiguity). Poles nu_j > 0 selected by pivoted QR of the
    stacked kernel [K-, -K+] on the two upper rays; the fit solves the coupled (PQ, QP) pair system for all pairs at once."""
    def __init__(self, theta, lam, eps=1e-8, gap=0.0, tmin=None, tmax=None, nline=1200, npole=800):
        self.theta, self.lam, self.eps, self.gap = theta, lam, eps, gap
        tmin = 1e-4 * lam if tmin is None else tmin; tmax = 20 * lam if tmax is None else tmax
        t = np.exp(np.linspace(np.log(tmin), np.log(tmax), nline))
        self.zeta_dense = np.concatenate([t * np.exp(1j * theta), t * np.exp(1j * (np.pi - theta))])
        nu = np.exp(np.linspace(np.log(max(gap, 1e-4 * lam)), np.log(lam), npole))
        Kst = np.vstack([1.0 / (self.zeta_dense[:, None] - nu[None, :]), -1.0 / (self.zeta_dense[:, None] + nu[None, :])])
        Kn = Kst / np.linalg.norm(Kst, axis=0, keepdims=True)
        _, R, piv = qr(Kn, mode='economic', pivoting=True)
        d = np.abs(np.diag(R)); r = int((d > eps * d[0]).sum())
        self.nu = np.sort(nu[piv[:r]]); self.r = r
        # line nodes: row-pivoted QR on the odd kernel (diagonal-element structure), 2r nodes to over-determine the pair system
        Ko = 1.0 / (self.zeta_dense[:, None] - self.nu[None, :]) - 1.0 / (self.zeta_dense[:, None] + self.nu[None, :])
        _, R2, piv2 = qr(Ko.T, mode='economic', pivoting=True)
        self.zeta = self.zeta_dense[np.sort(piv2[:min(2 * r, len(self.zeta_dense))])]

    def kernels(self, zeta):
        z = np.asarray(zeta, complex)[:, None]
        return 1.0 / (z - self.nu[None, :]), 1.0 / (z + self.nu[None, :])

    def fit(self, zeta, W, rcond=None):
        """W (nz, N, N) at mu-relative points zeta -> residues w (r, N, N) of the positive poles."""
        Km, Kp = self.kernels(zeta); nz, N = W.shape[0], W.shape[1]
        A = np.block([[Km, -Kp], [-Kp, Km]])                              # unknowns [w_PQ ; w_QP] per pair, data [W_PQ ; W_QP]
        iu = np.triu_indices(N)
        data = np.concatenate([W[:, iu[0], iu[1]], W[:, iu[1], iu[0]]], axis=0)   # (2 nz, npairs)
        sol = np.linalg.lstsq(A, data, rcond=rcond)[0]                       # (2 r, npairs)
        w = np.zeros((self.r, N, N), complex)
        w[:, iu[0], iu[1]] = sol[:self.r]; w[:, iu[1], iu[0]] = sol[self.r:]
        return w

    def eval(self, w, zeta, sector=None):
        """W(zeta) (nz, N, N); sector '>' = positive-pole part only, '<' = negative-pole part only, None = both."""
        Km, Kp = self.kernels(zeta); r = w.shape[0]
        wp = w.reshape(r, -1); wt = np.transpose(w, (0, 2, 1)).reshape(r, -1)
        out = 0
        if sector in (None, '>'): out = out + Km @ wp
        if sector in (None, '<'): out = out - Kp @ wt
        return out.reshape((len(zeta),) + w.shape[1:])

    def time_exponentials(self, t, sector='>'):
        """Residue-weighted exponentials for the time ray: W^>(t) = sum_j w_j e^{-i nu_j t};  W^<(t) = -sum_j w_j^T e^{+i nu_j t}."""
        t = np.asarray(t, complex)
        return np.exp(-1j * self.nu[None, :] * t[:, None]) if sector == '>' else -np.exp(1j * self.nu[None, :] * t[:, None])

    def __repr__(self):
        return f"BosonicLineBasis(theta={np.degrees(self.theta):.1f} deg, lam={self.lam}, eps={self.eps:g}, gap={self.gap}, rank={self.r}, nodes={len(self.zeta)})"
