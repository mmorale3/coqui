#!/usr/bin/env python3
"""D4 nu_0 option (C++ key nu0_intraband; notes section 12 "The nu_0 term"): reference of the Matsubara-convention static term of
the exactly degenerate pairs for the C++ test [finiteT][hybrid][nu0], lih222 at beta 50 (KS G at mu0 = N(mu) root, "number").

Oracle (finite_t, independent of the line machinery): for every q the KS transition list (deg_tol 1e-8),
  dPi(q)  = pi_nu0_extra = -beta (2/Nk) sum_{E=0} f (1 - f) S S^H    (= Pi^Mats(q, i nu_0) - Pi^an(q, 0)),
  dW(q)   = W[Pi^an(q, 0) + dPi] - W[Pi^an(q, 0)],  W[P] = ([1 - Z P]^-1 - 1) Z,  Pi^an(q, 0) = the transition sum at zeta = 0,
  Sigma_nu0(k, z) = -(1/(beta Nk)) sum_q X(k)^+ [G~(k-q, z) o dW(q)] X(k) = sum_{q, m} R_{qm}(k) / (z - e_m(k-q)),
  R_{qm}(k) = -(1/(beta Nk)) B^+ dW(q) B, B = diag(conj x_m) X(k), x_m = X(k-q)[:, m]   (positive semidefinite: dW <= 0).
Checks (printed, gated): the pole form vs finite_t.sigma_nu0_term [1e-13]; the prototype line machinery LineGW.nu0_dW (tau-leg
Pi^an(q, 0) and pi_tau_dpi) vs the oracle dW [1e-10]; LineGW.sigma_nu0 vs the oracle at i w_n [1e-10]; one hybrid iteration of
LineSCGW with nu0_intraband on minus off = the oracle term at i w_n [1e-8] (same G, same W: only the term differs).

H5 (complex as <name>_re / _im): attrs beta, mu0 (absolute), nk, nb, Np, deg_tol, generator, date; ndeg (nq,) degenerate pairs;
  dW (nq, Np, Np); pole_e (nk, nq*nb) energies e_m(k-q) - mu0 (q-major); pole_R (nk, nq*nb, nb, nb) residues;
  n (8,) Matsubara indices, term_iw (nk, 8, nb, nb) the term at i w_n = i (2n+1) pi / beta.
Usage (Mac): KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 coqui/cayley/scripts/gen_nu0_ref.py [--beta 50] [--no-scf] [--out F]
"""
import sys, os, time, datetime, functools, subprocess, numpy as np, h5py
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
from cayley.coqui_io import THC, _c
from cayley import finite_t as ft


def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


beta = float(opt('--beta', 50.0))
D = ROOT + '/coqui/tests/unit_test_files/gw_line/'
out = opt('--out', D + 'lih222_nu0_ref.h5')
thc = THC(D + 'lih222_thc/thc.eri.h5')
with h5py.File(D + 'lih222_thc/system.h5', 'r') as f:
    s = f['system']
    H0 = _c(s['H0'][()]); eig = np.array(s['eigval']); qk = np.array(s['qk_to_k2']); nelec = float(s['nelec'][()])
X, Z = thc.X[0], thc.Z
nk, Np, nb = X.shape
mu0 = ft.mu_number(eig, nelec, beta)
e = eig - mu0
DEG = 1e-8
NW = 8
iw = 1j * (2 * np.arange(NW) + 1) * np.pi / beta
t0 = time.time()
dW, ndeg = [], []
for iq in range(nk):
    tr = ft.transitions(X, e, qk, iq, beta, deg_tol=DEG)
    ndeg.append(len(tr['wd']))
    if len(tr['wd']) == 0:
        dW.append(np.zeros((Np, Np), complex)); continue
    Pan = ft.pi_transition(tr, [0.0])
    dW.append(ft.dyson_w(Z[iq], Pan + ft.pi_nu0_extra(tr, beta)[None])[0] - ft.dyson_w(Z[iq], Pan)[0])
dW = np.array(dW)
pe = np.zeros((nk, nk * nb)); pR = np.zeros((nk, nk * nb, nb, nb), complex)
for ik in range(nk):
    for iq in range(nk):
        km = qk[iq, ik]
        for m in range(nb):
            B = np.conj(X[km][:, m])[:, None] * X[ik]
            pe[ik, iq * nb + m] = e[km, m]
            pR[ik, iq * nb + m] = -(B.conj().T @ dW[iq] @ B) / (beta * nk)
term = np.einsum('kpij,kwp->kwij', pR, 1.0 / (iw[None, :, None] - pe[:, None, :]))
chk = max(np.abs(term[ik] - ft.sigma_nu0_term(X, e, qk, list(dW), beta, ik, iw)).max() for ik in range(nk)) / np.abs(term).max()
mineig = min(np.linalg.eigvalsh(0.5 * (R + R.conj().T)).min() for R in pR.reshape(-1, nb, nb))
print(f"lih222 beta {beta:g}: mu0 {mu0:.12f}; degenerate pairs per q {ndeg}; max|dW| {np.abs(dW).max():.3e}; max|term(i w_n)| "
      f"{np.abs(term).max():.3e}; pole form vs sigma_nu0_term {chk:.1e}; min eig of the residues {mineig:.1e} [{time.time() - t0:.1f}s]", flush=True)
gates = {'pole_form': (chk, 1e-13), 'residues_psd': (max(0.0, -mineig) / np.abs(pR).max(), 1e-12)}

# ---- the prototype's line machinery (tau leg) and one hybrid iteration with the option on / off
if '--no-scf' not in sys.argv:
    from cayley.line import driver as drv
    from cayley.line.thc_gw import WSTEP_DEFAULTS
    from cayley.line.closure import lehmann_from_sigma
    drv.lehmann_from_sigma = functools.partial(lehmann_from_sigma, nphi=8)
    P = dict(theta=np.deg2rad(20.0), eps=1e-8, lam=6.0, bos_lam=12.0, bos_gap=0.02, sig_gap=(0.02, 0.02), g_gap=(0.0, 0.0), wp=0.11, K=8,
             tol_gram=1e-10, mixing=0.5, nodes_per_ray=120, node_range=(1e-3, 60.0), beta=beta, thermal_tol=1e-8, thermal_floor=30.0,
             thermal_floor_f=30.0, wp_floor=15.0, mu_rule='number', bos_eps_T=1e-10, wstep=dict(WSTEP_DEFAULTS, cut_odd=1e-11),
             scf_density='matsubara', theta_t_frac=0.25, g_repr='lehmann', verbose=False)
    Siw = {}
    for flag in (False, True):
        sc = drv.LineSCGW(X, Z, qk, nk, nelec, H0, mu0, nu0_intraband=flag, **P)
        sc.start_from_hamiltonian(np.array([np.diag(eig[k]) for k in range(nk)]).astype(complex))
        assert abs(sc.mu - mu0) < 1e-10, (sc.mu, mu0)
        if flag:
            dWl = sc.gw.nu0_dW()
            gates['line_dW'] = (max(np.abs((dWl[q] if dWl[q] is not None else 0) - dW[q]).max() for q in range(nk)) / np.abs(dW).max(), 1e-10)
            Sl = np.array([sc.gw.sigma_nu0(ik, dWl, iw) for ik in range(nk)])
            gates['line_sigma_nu0'] = (np.abs(Sl - term).max() / np.abs(term).max(), 1e-10)
        sc.iterate()
        Siw[flag] = np.array(sc.Siw_prev[0])          # (nk, N, nb, nb): iteration 1 (no mixing at iteration 1)
    dS = Siw[True][:, :NW] - Siw[False][:, :NW]
    gates['scf_on_minus_off'] = (np.abs(dS - term).max() / np.abs(term).max(), 1e-8)
for k, (v, g) in gates.items():
    print(f"  {k:18s} {v:.2e}  (gate {g:.0e})  {'ok' if v <= g else 'FAIL'}")
assert all(v <= g for v, g in gates.values()), "a gate failed: reference not written"

head = subprocess.run(['git', '-C', ROOT + '/coqui', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
with h5py.File(out, 'w') as f:
    f.attrs.update(dict(fixture='lih222', beta=beta, mu0=mu0, nk=nk, nb=nb, Np=Np, deg_tol=DEG, generator='gen_nu0_ref.py',
                        cayley_head=head, date=str(datetime.date.today())))
    for k, (v, g) in gates.items(): f.attrs['ver_' + k] = v
    f['ndeg'] = np.array(ndeg)
    for nm, a in (('dW', dW), ('pole_R', pR), ('term_iw', term)):
        f[nm + '_re'] = a.real; f[nm + '_im'] = a.imag
    f['pole_e'] = pe
    f['n'] = np.arange(NW)
print(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB)")
