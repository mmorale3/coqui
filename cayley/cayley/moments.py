"""Cayley moments C^(n) = int A_Sigma(w) u(w)^n dw of a (matrix-valued) self-energy."""
import numpy as np
from .maps import cayley, disk_variable


def moments_from_poles(E, R, wp, nmax, mu=0.0, sector=None):
    """Exact moments of Sigma_c(z) = sum_k R_k/(z - E_k) with real poles E (K,) and residues R (K, N, N) or (K,) scalar.
    sector: None (total), '>' (E > mu) or '<' (E < mu). Returns (nmax+1, N, N) (or (nmax+1,) for scalar)."""
    E = np.asarray(E); R = np.asarray(R)
    if sector == '>': m = E > mu
    elif sector == '<': m = E < mu
    else: m = np.ones(E.shape, bool)
    u = cayley(E[m], wp, mu)
    Rm = R[m]
    if Rm.ndim == 1:
        return np.array([(Rm * u ** n).sum() for n in range(nmax + 1)])
    # accumulate with a Vandermonde in u: (nmax+1, K) @ (K, N*N)
    V = u[None, :] ** np.arange(nmax + 1)[:, None]
    return (V @ Rm.reshape(len(u), -1)).reshape(nmax + 1, *Rm.shape[1:])


def moments_from_pole_factors(E, A, B, wp, nmax, mu=0.0, weights=None):
    """Moments when residues factor as R_k = A[:, k] B[k, :] (e.g. Casida: a-index x pole x b-index), without forming R.
    A: (N, K), B: (K, N), weights: optional (K,) multiplying each pole. Returns (nmax+1, N, N)."""
    u = cayley(np.asarray(E), wp, mu)
    w = np.ones(len(u)) if weights is None else np.asarray(weights)
    out = np.empty((nmax + 1,) + (A.shape[0], B.shape[1]), complex)
    un = np.ones(len(u), complex)
    for n in range(nmax + 1):
        out[n] = (A * (w * un)[None, :]) @ B
        un = un * u
    return out


def bound_check(C):
    """max_n ||C^(n)|| / ||C^(0)|| (must be <= 1 for a positive measure)."""
    n0 = np.linalg.norm(C[0])
    return max(np.linalg.norm(c) for c in C) / n0


def lens_moments(Sig, C0, theta, wp, nmax, mu=0.0, s=None, orientation=None):
    """Moments by the Cauchy integral on the lens = tilted line zeta = mu + t e^{i theta} (t in R) and its Schwarz
    reflection, i.e. the rays at angles theta and pi-theta in the upper half plane, with
      Phi(z) = [C0 - (zeta - mu - i wp) Sigma(zeta)]/(1 - z),   C^(n) = (1/2 pi i) oint Phi z^{-n-1} dz.
    Sig(zeta) -> (nz, N, N) (or (nz,) scalar) must be the retarded/upper-half-plane analytic Sigma_c.
    Log grid t = wp e^s. Returns moments 0..nmax+1 (the last one is a held-out moment for the terminal phase).
    Error bound for data error delta on the line: delta * r^-n, r = tan(pi/4 - theta/2)."""
    if s is None: s = np.linspace(-35, 25, 7000)
    t = wp * np.exp(s); ds = s[1] - s[0]
    C0 = np.asarray(C0)
    tot = None
    for ang, sgn in [(theta, 1), (np.pi - theta, -1)]:
        zeta = mu + t * np.exp(1j * ang)
        S = np.asarray(Sig(zeta))
        z = disk_variable(zeta, wp, mu)
        dz = 2j * wp / (zeta - mu + 1j * wp) ** 2 * (zeta - mu)                 # dz/ds
        shp = (slice(None),) + (None,) * (S.ndim - 1)
        Fz = (C0[None] - (zeta - mu - 1j * wp)[shp] * S) / (1 - z)[shp]
        V = (z[None, :] ** (-np.arange(nmax + 2)[:, None] - 1)) * (dz * ds)[None, :]     # (n, nz)
        c = (V @ Fz.reshape(len(z), -1)).reshape((nmax + 2,) + S.shape[1:])
        tot = sgn * c if tot is None else tot + sgn * c
    c = tot / (2j * np.pi)
    # orientation: the two rays traversed t: 0 -> inf at angle theta and inf -> 0 at pi - theta form the lens boundary;
    # the sign convention above gives +C0 at n = 0 for theta in (0, pi/2) (checked against pole models in tests)
    if orientation is None:
        orientation = 1.0 if np.linalg.norm(c[0] - C0) <= np.linalg.norm(c[0] + C0) else -1.0
    return orientation * c
