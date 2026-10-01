"""Line basis + time-ray checks on pole models (mu = 0): fit/eval accuracy, sector split, moments, and the
time-ray transform of a product of two pole functions. Run: python3 tests/test_line_basis.py"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cayley.line.line_dlr import LineBasis
from cayley.line.timeray import TimeRay
from cayley import moments_from_poles
rng = np.random.default_rng(1)
th = np.deg2rad(20)
B = LineBasis(th, lam=60.0, eps=1e-10, gap=(3.0, 3.0))
print(B)
# model: poles outside the gap, matrix-valued (2x2) residues
E = np.concatenate([-(3 + 50 * rng.random(80) ** 2), 3 + 50 * rng.random(80) ** 2])
v = rng.standard_normal((160, 2)) + 1j * rng.standard_normal((160, 2)); R = np.einsum('ki,kj->kij', v, v.conj())
X = lambda z: np.einsum('zk,kij->zij', 1.0 / (np.atleast_1d(z)[:, None] - E[None, :]), R)
c = B.fit(B.zeta, X(B.zeta))
zd = B.zeta_dense
print("fit error on the dense line:          %.1e" % (np.abs(B.eval(c, zd) - X(zd)).max() / np.abs(X(zd)).max()))
zi = 1j * np.exp(np.linspace(np.log(1e-2), np.log(1e3), 50))
print("evaluation on the imaginary axis:     %.1e" % (np.abs(B.eval(c, zi) - X(zi)).max() / np.abs(X(zi)).max()))
ch, cp = B.split(c)
Xp = lambda z: np.einsum('zk,kij->zij', 1.0 / (np.atleast_1d(z)[:, None] - E[None, E > 0]), R[E > 0])
print("particle-sector split on the axis:    %.1e" % (np.abs(B.eval(cp, zi[:, None] * 0 + zi)[..., 0, 0] - 0)[:0].sum() if False else np.abs((B.kernel(zi)[:, B.pos] @ cp.reshape(len(cp), -1)).reshape(len(zi), 2, 2) - Xp(zi)).max() / np.abs(Xp(zi)).max()))
Cex = moments_from_poles(E, R, 3.0, 40); Cl = moments_from_poles(B.w, c, 3.0, 40)
print("moments from fitted poles vs exact, n=8/16/24/32/40: " + " ".join("%.0e" % (np.linalg.norm(Cl[n] - Cex[n]) / np.linalg.norm(Cex[0])) for n in (8, 16, 24, 32, 40)))
# time ray: product of two scalar pole functions -> Laplace to the line vs exact pole-pair sum
ea = 0.4 + 60 * rng.random(30) ** 2; ei = -(0.35 + 60 * rng.random(30) ** 2); ra, ri = rng.random(30), rng.random(30)
ray = TimeRay.for_spectrum(th / 2, emin=ea.min() - ei.max() if False else (ea.min() + abs(ei).min()), per_efold=3, nn=16)
Gt = ray.exponentials(ea) @ ra; Ht = ray.exponentials(-ei) @ ri      # G^>(t) and conj-part H^<(t) = sum r_i e^{+i e_i t}
Pt = Gt * Ht
F = ray.transform_matrix(B.zeta)
Pz = F @ Pt
D = (ea[:, None] - ei[None, :]).ravel(); W = (ra[:, None] * ri[None, :]).ravel()
Pex = (W[None, :] / (B.zeta[:, None] - D[None, :])).sum(1)
print("time-ray product -> line vs exact (%d nodes): %.1e" % (len(ray), np.abs(Pz - Pex).max() / np.abs(Pex).max()))
print("OK")
