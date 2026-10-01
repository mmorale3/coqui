"""Moments from IMAGINARY-AXIS data (what a converged Matsubara scGW provides).

For an insulator Sigma_c has a spectral gap (mu - D_h, mu + D_p). The gap-Laplace identity (window-closure notes, Prop. 1)
gives Sigma_c(zeta) for the whole strip -D_h < Re(zeta - mu) < D_p from Sigma_c(tau):
    Sigma(zeta) = int_0^{tau_c} dtau [ e^{(zeta-mu) tau} Sigma(tau) - e^{-(zeta-mu) tau} Sigma(beta - tau) ]   (fermionic, mu-shifted)
with an error ~ delta^{1-r}, r = |Re(zeta-mu)|/D (delta = data error), tau_c(zeta) = ln(1/eps)/(D - |x|), eps = delta^{1-r}.
With a DLR representation Sigma(tau) = sum_l c_l K(tau, w_l), K = -e^{-w tau}/(1+e^{-beta w}), the truncated transform is analytic.
The Cayley moments are the Taylor coefficients of Phi(z) = [C0 - (zeta-mu-i wp) Sigma(zeta)]/(1-z) at z = 0 (zeta = mu + i wp),
obtained by a Cauchy integral on the circle |z| = rho, whose zeta-image has max |Re(zeta-mu)| = wp 2 rho/(1-rho^2) (must be < D).
Error ~ max_circle delta^{1-r(zeta)} rho^-n: this is the quantitative limit of the imaginary-axis route.
"""
import numpy as np
from .maps import zeta_from_disk


def laplace_pole_term(zeta, w, tau_c, beta):
    """Truncated two-branch Laplace transform of one fermionic DLR basis function K(tau, w) = -e^{-w tau}/(1+e^{-beta w}):
       L = int_0^{tau_c} [e^{zeta tau} K(tau, w) - e^{-zeta tau} K(beta - tau, w)] dtau   (zeta measured from mu),
    K(beta - tau, w) = -e^{w tau}/(1+e^{beta w}).  Returns (nzeta, nw)."""
    zeta = np.asarray(zeta, complex)[:, None]; w = np.asarray(w, float)[None, :]; tc = np.asarray(tau_c, float)[:, None]
    nF_m = np.exp(-np.logaddexp(0.0, -beta * w))        # 1/(1+e^{-beta w}) = 1 - f(w)
    nF_p = np.exp(-np.logaddexp(0.0, beta * w))         # 1/(1+e^{beta w})  = f(w)
    a = zeta - w; b = -zeta + w
    with np.errstate(over='ignore', invalid='ignore'):
        I1 = np.where(np.abs(a) > 1e-14, np.expm1(a * tc) / np.where(np.abs(a) > 1e-14, a, 1.0), tc)
        I2 = np.where(np.abs(b) > 1e-14, np.expm1(b * tc) / np.where(np.abs(b) > 1e-14, b, 1.0), tc)
    return -(nF_m * I1) + (nF_p * I2)


def sigma_strip_dlr(zeta_rel, c, w, beta, D_minus, D_plus, delta):
    """Sigma_c(mu + zeta_rel) in the strip from DLR coefficients c (nw, ...) at real poles w (relative to mu), per-target
    optimal tau_c. D_minus/D_plus: spectral gap below/above mu (> 0). delta: relative data error (eps = delta^(1-r))."""
    zeta_rel = np.atleast_1d(np.asarray(zeta_rel, complex))
    x = zeta_rel.real
    D = np.where(x >= 0, D_plus, D_minus)
    r = np.minimum(np.abs(x) / D, 0.97)
    eps = np.maximum(delta, 1e-16) ** (1 - r)
    tau_c = np.minimum(np.log(1 / eps) / (D - np.abs(x)), beta / 2)
    L = laplace_pole_term(zeta_rel, w, tau_c, beta)                              # (nz, nw)
    sh = c.shape
    return (L @ c.reshape(sh[0], -1)).reshape((len(zeta_rel),) + sh[1:])


def sigma_dlr_direct(zeta_rel, c, w):
    """Direct evaluation of the DLR interpolant off the axis: Sigma(zeta) = sum_l c_l /(zeta - w_l) (signed pole sum)."""
    zeta_rel = np.atleast_1d(np.asarray(zeta_rel, complex))
    K = 1.0 / (zeta_rel[:, None] - np.asarray(w)[None, :])
    sh = c.shape
    return (K @ c.reshape(sh[0], -1)).reshape((len(zeta_rel),) + sh[1:])


def circle_moments(sig_fun, C0, wp, rho, nmax, nphi=256):
    """C^(n) = (1/2 pi i) oint_{|z|=rho} Phi(z) z^{-n-1} dz, Phi = [C0 - (zeta-mu-i wp) Sigma]/(1-z); sig_fun(zeta_rel) -> (nz, ...)."""
    phi = 2 * np.pi * (np.arange(nphi) + 0.5) / nphi
    z = rho * np.exp(1j * phi)
    zr = zeta_from_disk(z, wp, 0.0)
    S = np.asarray(sig_fun(zr))
    shp = (slice(None),) + (None,) * (S.ndim - 1)
    Phi = (np.asarray(C0)[None] - (zr - 1j * wp)[shp] * S) / (1 - z)[shp]
    V = z[None, :] ** (-np.arange(nmax + 1)[:, None]) / nphi
    return (V @ Phi.reshape(len(z), -1)).reshape((nmax + 1,) + S.shape[1:])


def max_rho(wp, D, rfrac=0.5):
    """largest circle radius whose zeta-image stays within |Re(zeta-mu)| <= rfrac*D:  wp 2 rho/(1-rho^2) = rfrac D."""
    s = rfrac * D / wp
    return (-1 + np.sqrt(1 + s * s)) / s
