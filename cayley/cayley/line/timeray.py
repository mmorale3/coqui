"""Complex-time ray t = s e^{-i theta_t} (particle sector; +theta_t for the hole sector) and the Laplace transform to the line:
   X^>(zeta) = -i int_0^inf dt e^{i zeta t} X^>(t),   X^>(t) = sum_p w_p e^{-i E_p t}  (E_p > 0, decays on the ray)
   X^<(zeta) = -i int_0^inf dt e^{i zeta t} X^<(t)   on the mirrored ray (E_p < 0).
All energies mu-relative; zeta on the upper half plane. Graded composite Gauss-Legendre in s with the first panel at 0."""
import numpy as np


class TimeRay:
    def __init__(self, theta_t, smax, smin=1e-5, per_efold=3, nn=16, sector='>'):
        xg, wg = np.polynomial.legendre.leggauss(nn)
        edges = np.concatenate([[0.0], np.exp(np.linspace(np.log(smin), np.log(smax), int(np.log(smax / smin) * per_efold) + 2))])
        s = ((edges[1:] + edges[:-1]) / 2)[:, None] + (edges[1:] - edges[:-1])[:, None] / 2 * xg[None, :]
        w = (edges[1:] - edges[:-1])[:, None] / 2 * wg[None, :]
        self.s, self.ws = s.ravel(), w.ravel()
        self.sector = sector
        self.phase = np.exp(-1j * theta_t) if sector == '>' else np.exp(1j * theta_t)
        self.t = self.s * self.phase                                  # complex times
        self.theta_t = theta_t

    @classmethod
    def for_spectrum(cls, theta_t, emin, decades=40.0, **kw):
        """smax from the smallest |pole energy| emin: e^{-emin smax sin(theta_t)} = e^{-decades}."""
        return cls(theta_t, decades / (emin * np.sin(theta_t)), **kw)

    def exponentials(self, E):
        """e^{-i E t} for pole energies E (np,) -> (nt, np)."""
        return np.exp(-1j * np.asarray(E)[None, :] * self.t[:, None])

    def transform_matrix(self, zeta):
        """F (nz, nt) with X(zeta) = F @ X(t):  F = -i * dt/ds * w_s * e^{i zeta t}."""
        return -1j * self.phase * self.ws[None, :] * np.exp(1j * np.asarray(zeta, complex)[:, None] * self.t[None, :])

    def __len__(self):
        return len(self.s)
