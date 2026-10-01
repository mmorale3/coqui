"""Synthetic checks of the cayley package: exact pole moments vs lens integral, exact upfolding of a small pole model,
spectral-function consistency, and the noisy-moment regularization. Run: python3 tests/test_synthetic.py"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cayley import cayley, inv_cayley, moments_from_poles, lens_moments, bound_check, upfold_block
from cayley.spectral import sigma_from_poles, spectral_function, lehmann_from_upfolded

rng = np.random.default_rng(0)
wp, mu = 3.0, 0.5

# 1. small exact model: N=2, 6 poles with rank-1 PSD residues -> upfolding with K >= 6 must be exact
E = np.array([-9.0, -5.0, -3.2, 3.5, 6.0, 11.0]) + mu
v = rng.standard_normal((6, 2)) + 1j * rng.standard_normal((6, 2))
R = np.einsum('ki,kj->kij', v, v.conj())
C = moments_from_poles(E, R, wp, 12, mu)
assert abs(bound_check(C) - 1) < 1e-12
Sig = lambda z: np.einsum('zk,kij->zij', 1.0 / (np.atleast_1d(z)[:, None] - E[None, :]), R)
Cl = lens_moments(Sig, C[0], np.deg2rad(20), wp, 11, mu)
print("lens vs exact moments (N=2, 6 poles):   %.1e" % (np.abs(Cl[:12] - C[:12]).max() / np.linalg.norm(C[0])))
for K in [3, 5, 6, 8]:
    d, W, info = upfold_block(C, K, wp, mu, tol_gram=1e-13, return_info=True)
    print(f"  K={K}: poles found {len(d)}, moment err {max(info['moment_err'][:K+1]):.1e}, held-out err {info['heldout_err']:.1e}, "
          f"max|d - E| = {max(min(abs(dd - E)) for dd in d):.1e}" if K >= 5 else f"  K={K}: poles found {len(d)}, moment err {max(info['moment_err'][:K+1]):.1e}")
d, W = upfold_block(C, 8, wp, mu, tol_gram=1e-13)
H = np.array([[0.2, 0.1], [0.1, -0.4]]) + mu
om = np.linspace(-12, 12, 2401)
A1 = spectral_function(H, sigma_from_poles(d, W), om, 0.1, trace=True)
A2 = spectral_function(H, Sig, om, 0.1, trace=True)
print("A(w) upfolded vs exact (exact-moment, K=8):  %.1e" % (np.abs(A1 - A2).max() / A2.max()))
e, V = lehmann_from_upfolded(H, d, W)
print("Lehmann weights sum (should be N=2):  %.6f" % (np.abs(V) ** 2).sum())

# 2. continuum model (many poles) with noisy lens moments and Gram truncation (the realistic case)
Ec = np.concatenate([-3 - 20 * rng.random(300) ** 1.5, 3 + 25 * rng.random(300) ** 1.5]) + mu
vc = rng.standard_normal((600, 3)) + 1j * rng.standard_normal((600, 3))
Rc = np.einsum('ki,kj->kij', vc, vc.conj()); Rc *= 60 / np.trace(Rc.sum(0)).real
Sigc = lambda z: np.einsum('zk,kij->zij', 1.0 / (np.atleast_1d(z)[:, None] - Ec[None, :]), Rc)
Cex = moments_from_poles(Ec, Rc, wp, 41, mu)
Hc = np.diag([-1.5, 0.3, 1.2]) + mu
Aex = spectral_function(Hc, Sigc, om, 0.3, trace=True)
noisy = lambda eps: (lambda z: Sigc(z) * (1 + eps * (rng.standard_normal((len(np.atleast_1d(z)), 3, 3)) + 1j * rng.standard_normal((len(np.atleast_1d(z)), 3, 3))) / np.sqrt(2)))
for eps in [0, 1e-8, 1e-6]:
    Cn = Cex if eps == 0 else lens_moments(noisy(eps), Cex[0], np.deg2rad(20), wp, 40, mu)
    row = []
    for K in [8, 16, 24, 32]:
        d, W = upfold_block(Cn, K, wp, mu, tol_gram=max(10 * eps, 1e-13))
        A = spectral_function(Hc, sigma_from_poles(d, W), om, 0.3, trace=True)
        row.append(f"K={K}: {np.abs(A - Aex).max() / Aex.max():.1e}")
    print(f"continuum N=3, line noise {eps:g}, rel. A error in [-12,12] at eta=0.3:  " + "  ".join(row))
print("OK")
