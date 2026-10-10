"""Finite-T checks of the line prototype on a small random THC model (S8b, notes section 11): oracles cross-checked by two
routes, thermal Pi on the guarded rays vs the transition sum, the line Sigma_c (W known on all nodes) vs Eq. fT_sigma, and
the T = 0 path unchanged when no pole lies in the window. Run: python3 tests/test_finite_t.py"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cayley import finite_t as ft
from cayley.line.thc_gw import LineGW, node_floor_mask
from cayley.line.line_dlr import BosonicLineBasis
from cayley.line.closure import chemical_potential_T, chemical_potential_auto

rng = np.random.default_rng(7)
nk, Np, nb, beta = 2, 6, 4, 50.0
qk = np.array([[0, 1], [1, 0]]); qminus = np.array([0, 1])
X = (rng.standard_normal((nk, Np, nb)) + 1j * rng.standard_normal((nk, Np, nb))) / np.sqrt(Np)
A = rng.standard_normal((nk, Np, Np)); Z = 0.3 * np.einsum("qij,qkj->qik", A, A) / Np + 0j   # real: Z(-q) = conj Z(q) = Z(q) for q = -q
e = np.array([[-1.2, -0.08, 0.05, 1.5], [-0.9, -0.03, 0.11, 2.0]])          # mu-relative, window E_T = 0.37 at beta 50
th, tht = np.deg2rad(20.0), np.deg2rad(10.0)

# 1. oracles: transition sum vs tau route (Matsubara), nu_0 term, Casida vs Dyson, Sigma thermal factors
for iq in range(nk):
    tr = ft.transitions(X, e, qk, iq, beta)
    Pt = ft.pi_transition(tr, 2j * np.pi * np.arange(3) / beta); Pm = ft.pi_matsubara_tau(X, e, qk, iq, beta, [0, 1, 2])
    sc = np.abs(Pt).max()
    assert np.abs(Pm[1:] - Pt[1:]).max() / sc < 1e-12
    assert np.abs(Pm[0] - Pt[0] - ft.pi_nu0_extra(tr, beta)).max() / sc < 1e-12
    lam, al, be, _ = ft.casida_from_transitions(tr, Z[iq])
    zs = [0.1j, 1j, 0.3 + 0.2j]
    assert np.abs(ft.casida_w(lam, al, be, zs) - ft.dyson_w(Z[iq], ft.pi_transition(tr, zs))).max() < 1e-12
print("oracles: tau route, nu_0 term, Casida: OK")
assert ft.thermal_factor_check(beta, e.ravel(), np.array([0.01, 0.2, 1.5]), 1j * np.pi / beta * np.array([1, 7])) < 1e-12

# 2. chemical potential: two implementations
eig = e + 0.3; nel = 4.0
mu_a = ft.mu_number(eig, nel, beta)
dmu, N = chemical_potential_T(eig, np.array([np.eye(nb)] * nk), nk, nel, beta)
assert abs(mu_a - dmu) < 1e-12 and abs(N - nel) < 1e-12
print(f"mu(N = 4) = {mu_a:.12f}: OK")

# 3. line: thermal Pi vs transition sum at unmasked nodes; Sigma with W on all nodes vs Eq. fT_sigma
bos = BosonicLineBasis(th, 4.0, eps=1e-10, gap=0.0)
t = np.exp(np.linspace(np.log(1e-3), np.log(60.0), 30)); fz = np.concatenate([t * np.exp(1j * th), t * np.exp(1j * (np.pi - th))])
gw = LineGW(X, Z, qk, nk, 0.0, th, tht, bos, fz)
v = np.array([np.eye(nb, dtype=complex)] * nk)
gw.set_poles(e, v, beta=beta)
assert gw.thermal
m = node_floor_mask(bos.zeta, beta, th, tht)
cas = []
for iq in range(nk):
    tr = ft.transitions(X, e, qk, iq, beta); cas.append(ft.casida_from_transitions(tr, Z[iq])[:3])
    Pl, Pt = gw.polarization(iq, bos.zeta[m]), ft.pi_transition(tr, bos.zeta[m])
    err = np.abs(Pl - Pt).max() / np.abs(Pt).max()
    assert err < 1e-10, err
    print(f"q {iq}: thermal Pi on guarded rays vs transition sum (unmasked nodes) {err:.1e}")
wres = [bos.fit(bos.zeta, ft.casida_w(*cas[iq], bos.zeta)) for iq in range(nk)]
fm = node_floor_mask(fz, beta, th, tht)
for ik in range(nk):
    Sl, Sx = gw.sigma(ik, wres, fz[fm], qminus=qminus), ft.sigma_fT(X, e, qk, qminus, cas, beta, ik, fz[fm])
    err = np.abs(Sl - Sx).max() / np.abs(Sx).max()
    assert err < 1e-8, err
    print(f"k {ik}: line Sigma_c (thermal lists + Bose W, W on all nodes) vs Eq. fT_sigma {err:.1e}")

# 4. no pole in the window (beta 1e4) and beta None: identical T = 0 path
g0 = LineGW(X, Z, qk, nk, 0.0, th, tht, bos, fz); g0.set_poles(e, v)
g1 = LineGW(X, Z, qk, nk, 0.0, th, tht, bos, fz); g1.set_poles(e, v, beta=1e4)
assert not g1.thermal
assert np.array_equal(g0.polarization(1), g1.polarization(1))
assert np.array_equal(g0.sigma(0, wres), g1.sigma(0, wres)) and np.array_equal(g0.density_matrix(), g1.density_matrix())
print("beta 1e4 (empty window) == T = 0 path, bitwise: OK")
print("OK")
