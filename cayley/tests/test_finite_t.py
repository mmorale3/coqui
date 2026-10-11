"""Finite-T checks of the line prototype on a small random THC model (S8b, notes section 11): oracles cross-checked by two
routes, thermal Pi on the guarded rays vs the transition sum, the line Sigma_c (W known on all nodes) vs Eq. fT_sigma, and
the T = 0 path unchanged when no pole lies in the window; the redesigned W step (S8b.1b, notes section 11.5): data set D, tau leg,
basis on D, joint odd/even fit, line-only Sigma_c, one thermal SCF iteration; the S8b.3 hybrid (notes section 11.6 "Hybrid"):
Filon tau transform, tau-leg Sigma_c(i w_n) vs Eq. fT_sigma (+ nu_0 option), Matsubara density vs the exact upfolded density,
one SCF iteration with scf_density = "matsubara". Run: python3 tests/test_finite_t.py"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cayley import finite_t as ft
from cayley.line.thc_gw import LineGW, node_floor_mask
from cayley.line.line_dlr import BosonicLineBasis
from cayley.line.closure import chemical_potential_T, chemical_potential_auto, electron_count_T
from cayley.line.driver import LineSCGW

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
    Sl, Sx = gw.sigma(ik, wres, fz[fm], qminus=qminus, nu=[bos.nu] * nk), ft.sigma_fT(X, e, qk, qminus, cas, beta, ik, fz[fm])
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

# 5. redesigned W step (line only): D, tau leg, basis on D, joint split fit, Sigma_c vs Eq. fT_sigma
gw.set_poles(e, v, beta=beta, thermal_tol=1e-12)
z, kind = gw.bos_data()
assert np.array_equal(z, gw.zD) and np.array_equal(np.bincount(kind), [np.sum(m), 168, 20, 1]) and z[kind == 3][0] == 0
zb = z[kind == 1].reshape(8, 21); assert np.array_equal(zb, -zb[:, ::-1].conj())      # band rows: z -> -conj(z) exactly
for iq in range(nk):
    tr = ft.transitions(X, e, qk, iq, beta)
    PiM, dPi = gw.pi_tau_leg(iq, [0, 1])
    Pt = ft.pi_transition(tr, [0.0, 2j * np.pi / beta]); sc = np.abs(Pt).max()
    assert np.abs(PiM[0] - dPi - Pt[0]).max() / sc < 1e-13 and np.abs(PiM[1] - Pt[1]).max() / sc < 1e-13
    assert np.abs(dPi - ft.pi_nu0_extra(tr, beta)).max() / sc < 1e-13
    assert np.array_equal(gw.polarization(iq, np.array([0j]))[0], PiM[0] - dPi)
print(f"tau leg (ntau {len(gw.tau)}): dynamic Pi(q, 0), Pi(i nu_1), degenerate term vs oracles: OK")
wres, Wl = gw.w_step(qminus)
for iq in range(nk):
    Wc = ft.casida_w(*cas[iq], z); sc = np.abs(Wc).max()
    assert np.abs(Wl[iq] - Wc).max() / sc < 1e-11
    Wf = gw.bos_w.eval(wres[iq], z, w_minus=wres[qminus[iq]])
    for kd, tol in ((3, 1e-9), (1, 1e-9), (2, 1e-9), (0, 1e-9)):      # toy model (gates of T2 on lih222 / lih223: gen_finiteT_ref.py)
        err = np.abs(Wf[kind == kd] - Wc[kind == kd]).max() / np.abs(Wc[kind == kd]).max()
        assert err < tol, (iq, kd, err)
w1, w1m = gw.bos_w.fit_split(z, Wl[1], W_minus=Wl[1])
assert np.array_equal(w1, wres[1])
assert np.abs(w1m - w1).max() <= 1e-12 * np.abs(w1).max()          # self-inverse q: w(-q) of the joint solve = w(q)
iw = 1j * np.pi * (2 * np.arange(40) + 1) / beta; iw = iw[np.abs(iw) >= gw.zeta_T]
for ik in range(nk):
    zz = np.concatenate([fz[fm], iw])
    Sl, Sx = gw.sigma(ik, wres, zz, qminus=qminus), ft.sigma_fT(X, e, qk, qminus, cas, beta, ik, zz)
    err = np.abs(Sl - Sx).max() / np.abs(Sx).max()
    assert err < 1e-8, err
    print(f"k {ik}: line-only Sigma_c (W step on D, rank {gw.bos_w.r}) vs Eq. fT_sigma {err:.1e}")

# 6. one thermal SCF iteration (line only, Lehmann G, mu by N(mu) = N_el)
H0 = np.array([np.diag(e[k] + 0.3) for k in range(nk)]).astype(complex)
sc = LineSCGW(X, Z, qk, nk, 4.0, H0, 0.3, eps=1e-8, lam=6.0, bos_lam=4.0, K=4, nodes_per_ray=30, beta=beta, mu_rule='number',
              verbose=False, qminus=qminus)
sc.start_from_hamiltonian(H0)
rec = sc.iterate()
assert rec['thermal'] and abs(rec['N_mu'] - 4.0) < 1e-10 and np.isfinite(rec['mu'])
assert abs(electron_count_T(sc.e_leh, sc.v_leh, beta) - 4.0) < 1e-10
print(f"thermal SCF iteration: mu {rec['mu']:.6f}, N(mu) - N_el {rec['N_mu'] - 4.0:.1e}, |D| {rec['nD']}, rank_b {rec['rank_b']}: OK")

# 7. S8b.3 hybrid: tau-leg Sigma_c(i w_n), Matsubara density, scf_density = "matsubara"
from cayley.line.timeray import tau_fourier, tau_panels
pan = tau_panels(beta, 8.0, 20, 1.0, 0.02); iwt = 1j * np.pi * (2 * np.array([0, 1, 7, 70, 700, 7000, 70000]) + 1) / beta
Ft = tau_fourier(beta, 8.0, iwt, 20, 1.0, 0.02, panels=pan)
for E in (-7.9, -0.3, -1e-3, 0.0, 2e-3, 0.4, 7.9):                  # -G(tau) of one pole: transform -1/(i w - E)
    assert np.abs(Ft @ np.exp(-np.logaddexp(0, -beta * E) - E * pan[0]) + 1 / (iwt - E)).max() < 1e-13 * beta / np.pi
n_h, iwh = ft.matsubara_set(beta, 100.0)
wf = [(c[1][:, c[0] > 0], c[2][c[0] > 0]) for c in cas]; nuf = [c[0][c[0] > 0] for c in cas]
gw.set_poles(e, v, beta=beta, thermal_tol=1e-12)
r = gw.sigma_tau_leg(wf, iwh, qminus=qminus, nu=nuf)
Sx = np.array([ft.sigma_fT(X, e, qk, qminus, cas, beta, ik, iwh) for ik in range(nk)])
hf = [ft.sigma_fT_hf(X, e, qk, qminus, cas, beta, ik) for ik in range(nk)]
S1x, S2x = np.array([h[0] for h in hf]), np.array([h[1] for h in hf])
err = max(np.abs(r['sigma'][k] - Sx[k]).max() / np.abs(Sx[k]).max() for k in range(nk))
assert err < 1e-12 and np.abs(r['S1'] - S1x).max() < 1e-13 * np.abs(S1x).max() and np.abs(r['S2'] - S2x).max() < 1e-13 * np.abs(S2x).max(), err
dW = []
for iq in range(nk):
    tr = ft.transitions(X, e, qk, iq, beta); P0 = ft.pi_transition(tr, [0.0])[0]
    dW.append(ft.dyson_w(Z[iq], [P0 + ft.pi_nu0_extra(tr, beta)])[0] - ft.dyson_w(Z[iq], [P0])[0] if len(tr['wd']) else None)
assert dW[0] is not None
r0 = gw.sigma_tau_leg(wf, iwh[:40], qminus=qminus, nu=nuf, dW=dW)
for ik in range(nk):
    ex = Sx[ik][:40] + ft.sigma_nu0_term(X, e, qk, dW, beta, ik, iwh[:40])
    assert np.abs(r0['sigma'][ik] - ex).max() < 1e-12 * np.abs(ex).max()
print(f"tau-leg Sigma_c (exact W, ntau {r['ntau']}, {len(iwh)} i w_n) vs Eq. fT_sigma {err:.1e}; S1/S2 and the nu_0 option: OK")
wres_l, _ = gw.w_step(qminus)                                         # the line W (redesigned W step) in both transforms
rl = gw.sigma_tau_leg(wres_l, iwh, qminus=qminus)
mT = np.abs(iwh) >= gw.zeta_T
for ik in range(nk):
    Sr = gw.sigma(ik, wres_l, iwh[mT][:60], qminus=qminus)
    assert np.abs(rl['sigma'][ik][mT][:60] - Sr).max() < 1e-10 * np.abs(Sr).max()
    assert np.abs(rl['sigma'][ik] - Sx[ik]).max() < 1e-8 * np.abs(Sx[ik]).max()
Hh = np.array([np.diag(e[k]) for k in range(nk)]).astype(complex)
Hh = Hh + 0.05 * (lambda M: M + M.conj().transpose(0, 2, 1))(rng.standard_normal((nk, nb, nb)) + 1j * rng.standard_normal((nk, nb, nb)))
dmu_h, Dh, Nh, hi = ft.density_matsubara(Hh, Sx, iwh, beta, S1x, S2x, nelec=4.0)
assert abs(Nh - 4.0) < 1e-12 and abs(hi['N_trace'] - 4.0) < 1e-12
for ik in range(nk):
    Us, Es = [], []
    for E_, w_, A_, B_ in ft.sigma_fT_blocks(X, e, qk, qminus, cas, beta, ik):       # rank-one PSD residues -> U columns
        c_ = np.einsum('jb,bj->j', B_, A_) / np.einsum('bj,bj->j', A_.conj(), A_)
        Us.append(A_ * np.sqrt(w_ * c_.real)[None, :]); Es.append(E_)
    Dex = ft.density_upfold(Hh[ik], np.concatenate(Es), np.concatenate(Us, 1), beta, dmu_h)
    assert np.abs(Dh[ik] - Dex).max() < 1e-12, np.abs(Dh[ik] - Dex).max()
dmu_t, Dt, Nt, _ = ft.density_matsubara(Hh, r['sigma'], iwh, beta, r['S1'], r['S2'], nelec=4.0)
assert np.abs(Dt - Dh).max() < 1e-12 and abs(dmu_t - dmu_h) < 1e-12
print(f"Matsubara density (N {len(iwh)}, w_N 100 Ha): exact Sigma vs exact upfolded density <= 1e-12, tau-leg Sigma vs exact Sigma "
      f"D {np.abs(Dt - Dh).max():.1e} dmu {abs(dmu_t - dmu_h):.1e}: OK")
sm = LineSCGW(X, Z, qk, nk, 4.0, H0, 0.3, eps=1e-8, lam=6.0, bos_lam=4.0, K=4, nodes_per_ray=30, beta=beta, mu_rule='number',
              verbose=False, qminus=qminus, scf_density='matsubara', hyb_wmax=50.0)
sm.start_from_hamiltonian(H0)
assert abs(ft.electron_number(np.array([np.diag(H0[k]).real for k in range(nk)]), sm.mu, beta) - 4.0) < 1e-12
rm = sm.iterate()
assert rm['mu_rule'] == 'matsubara' and abs(rm['N_mu'] - 4.0) < 1e-10 and abs(rm['nelec'] - 4.0) < 1e-10
print(f"thermal SCF iteration, scf_density matsubara: mu {rm['mu']:.6f} (closure mu {rm['mu'] + rm['dmu_closure']:.6f}), N(mu) - N_el "
      f"{rm['N_mu'] - 4.0:.1e}, closure N - N_el {rm['dN_closure']:+.1e}, max|D_closure - D| {rm['D_cl_err']:.1e}: OK")
print("OK")
