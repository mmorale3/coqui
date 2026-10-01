"""Exact G0W0 correlation self-energy in Lehmann form from a Casida (full RPA) solution in the THC basis.

Casida cache (from the user's real_axis_GW/scripts/si_pipeline.py::casida_q): per q, poles lam (Nt,) real, alpha (Np, Nt),
bet (Nt, Np) with W_dyn(q, z) = sum_s alpha[:, s] bet[s, :] / (z - lam_s). With KS lines e_n(k) (measured from mu0) and
T=0-like finite-T factors, CoQui's Sigma_ab(k, tau) = -(1/Nk) sum_q sum_n g_n(k-q, tau) sum_s Bt(tau, s) A_{a n s} Bm_{s n b},
  g_n(tau) = -K_F(e_n, tau),  Bt(tau, s) = -K_B(lam_s, tau)/lam_s,  A = (conj(X_k) * X_{k-q}[:, n])^T alpha,  Bm = bet (conj(X_{k-q}[:, n]) * X_k).
Product of the two kernels: g_n Bt = c_T(e_n, lam_s) * [-K_F(E, tau)],  E = e_n + lam_s,
  c_T = -(1 + e^{-beta E}) / [(1 + e^{-beta e_n})(1 - e^{-beta lam_s})]      (-> -1 for particle poles, +1 for hole poles at T=0),
so Sigma_c(z)_ab(k) = sum_{q,n,s} R^{qns}_ab / (z - E_{qns}),  R^{qns}_ab = -(1/Nk) c_T A_{a n s} Bm_{s n b}.
Moments and Sigma(z) are accumulated per (q, n) with GEMMs over s; nothing of size Nt x nb x nb is stored.
"""
import numpy as np
from .maps import cayley


def _logaddexp0(x):      # log(1 + e^x), stable
    return np.logaddexp(0.0, x)


def thermal_factor(e, lam, beta):
    """c_T(e, lam) = -(1 + e^{-beta E})/[(1 + e^{-beta e})(1 - e^{-beta lam})], E = e + lam, computed stably; (len(lam),)."""
    E = e + lam
    num = _logaddexp0(-beta * E)
    den_f = _logaddexp0(-beta * e)
    # log|1 - e^{-beta lam}| and its sign
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        pos = lam > 0
        log_den_b = np.where(pos, np.log1p(-np.exp(-beta * np.abs(lam))),
                             -beta * lam + np.log1p(-np.exp(-beta * np.abs(lam))))   # lam<0: 1 - e^{-beta lam} = -e^{-beta lam}(1 - e^{beta lam})
        sign_b = np.where(pos, 1.0, -1.0)
    return -sign_b * np.exp(num - den_f - log_den_b)


class CasidaG0W0:
    def __init__(self, thc, eig, mu0, beta, qk_to_k2, nk, casida_dir, prefix='casida_q'):
        """thc: THC object (X (ns,nk,Np,nb), Z); eig: (nk, nb) KS energies (Ha); mu0: KS chemical potential."""
        self.X = thc.X[0]; self.Np = thc.Np; self.eig = np.asarray(eig); self.mu0 = mu0; self.beta = beta
        self.qk_to_k2 = qk_to_k2; self.nk = nk; self.nb = self.X.shape[-1]
        self.cdir, self.prefix = casida_dir, prefix
        self._cache = {}

    def casida(self, iq):
        if iq not in self._cache:
            z = np.load(f"{self.cdir}/{self.prefix}{iq}.npz")
            self._cache[iq] = (np.asarray(z['lam'], float), np.asarray(z['alpha']), np.asarray(z['bet']))
        return self._cache[iq]

    def iter_blocks(self, ik):
        """yield (E (Nt,) ABSOLUTE pole energies mu0 + e_n + lam, A (nb, Nt), Bm (Nt, nb), cT (Nt,)) for every (q, n);
        Sigma_c(z) = -(1/Nk) sum A diag(cT/(z-E)) Bm with z absolute."""
        e = self.eig - self.mu0
        Xk = self.X[ik]
        for iq in range(self.nk):
            lam, alpha, bet = self.casida(iq)
            ikmq = self.qk_to_k2[iq, ik]; Xm = self.X[ikmq]
            for n in range(self.nb):
                A = (Xk.conj() * Xm[:, n][:, None]).T @ alpha          # (a, s)
                Bm = bet @ (Xm[:, n].conj()[:, None] * Xk)              # (s, b)
                cT = thermal_factor(e[ikmq, n], lam, self.beta)
                yield self.mu0 + e[ikmq, n] + lam, A, Bm, cT

    def moments(self, ik, wp, nmax, mu, sector=None, return_c0_sectors=False):
        """Exact Cayley moments C^(0..nmax) (nb, nb) of Sigma_c(k) about mu; sector None/'>'/'<' selects poles above/below mu."""
        out = np.zeros((nmax + 1, self.nb, self.nb), complex)
        for E, A, Bm, cT in self.iter_blocks(ik):
            if sector == '>': m = E > mu
            elif sector == '<': m = E < mu
            else: m = slice(None)
            u = cayley(E[m], wp, mu); w = -cT[m] / self.nk
            Am, Bmm = A[:, m], Bm[m]
            un = np.ones(len(u), complex)
            for n in range(nmax + 1):
                out[n] += (Am * (w * un)[None, :]) @ Bmm
                un = un * u
        return out

    def sigma_z(self, ik, z):
        """Exact Sigma_c(k, z) for complex z (nz,) -> (nz, nb, nb) (upper half plane: retarded)."""
        z = np.atleast_1d(np.asarray(z, complex))
        out = np.zeros((len(z), self.nb, self.nb), complex)
        for E, A, Bm, cT in self.iter_blocks(ik):
            w = -cT / self.nk
            for iz, zz in enumerate(z):
                out[iz] += (A * (w / (zz - E))[None, :]) @ Bm
        return out

    def sigma_tau(self, ik, tau):
        """Sigma_c(k, tau) on CoQui's convention (for validation against the checkpoint / SigmaExact files): (ntau, nb, nb)."""
        tau = np.atleast_1d(tau)
        out = np.zeros((len(tau), self.nb, self.nb), complex)
        for E, A, Bm, cT in self.iter_blocks(ik):
            w = -cT / self.nk
            Er = E - self.mu0                                                                   # kernel energies relative to mu0
            KF = np.exp(-Er[None, :] * tau[:, None] - _logaddexp0(-self.beta * Er)[None, :])    # (nt, s)
            for it in range(len(tau)):
                out[it] += (A * (-w * KF[it])[None, :]) @ Bm
        return out

    def pole_summary(self, ik, mu, wtol=1e-8):
        """Edges of the Sigma_c spectral support about mu, defined by cumulative weight: walking away from mu in each sector,
        the edge is the first pole at which the accumulated |tr R| exceeds wtol times the sector total (robust against the
        thermally suppressed combinations that sit inside the gap with weights ~e^{-beta E}). Also the sector weights."""
        Es, ws = [], []
        for E, A, Bm, cT in self.iter_blocks(ik):
            w = -cT / self.nk
            Es.append(E); ws.append((np.einsum('as,sa->s', A, Bm) * w).real)
        E = np.concatenate(Es); w = np.concatenate(ws)
        out = {}
        for name, m, sgn in (('particle', E > mu, 1.0), ('hole', E <= mu, -1.0)):
            Em, wm = E[m], np.abs(w[m]); tot = wm.sum()
            order = np.argsort(sgn * (Em - mu))                    # increasing distance from mu
            cum = np.cumsum(wm[order])
            j = int(np.searchsorted(cum, wtol * tot))
            out[name + '_edge'] = Em[order][min(j, len(order) - 1)]
            out['weight_' + name] = w[m].sum()
        return out
