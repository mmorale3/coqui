#!/usr/bin/env python3
"""Finite-temperature SCF reference for the C++ [gw_line][finiteT] T5 parity test (plan S8b, notes section 11): a few iterations
of the prototype's thermal loop (cayley/line/driver.py LineSCGW(..., beta=B): redesigned W step on the data set D, total Sigma_c
on the two-sided gapless basis at the unmasked nodes, omega_p floor, Lehmann G, mu by the "auto" rule) on lih222 from the KS
poles. Line only: no oracle input.

Inputs: tests/unit_test_files/gw_line/lih222_thc/{thc.eri.h5, system.h5} (as gen_lih222_scf_ref.py).
Settings (attrs; mirror gen_lih222_scf_ref.py where they exist): theta 20 deg, theta_t = theta/2, eps 1e-8, lam 6 (two-sided
gapless Sigma basis), lam_b 12 (gapless bosonic line basis, eps bos_eps_T 1e-10; its unmasked nodes enter D; candidates of the
D-selected basis [1e-4 lam_b, lam_b]), 120 log nodes per ray in [1e-3, 60] Ha, wp 0.11 -> omega_p = max(wp, wp_floor zeta_T),
K 8, tol_gram 1e-10, nphi 8, mixing 0.5 (linear, total Sigma_c), beta 200, thermal_tol 1e-8, c_zeta = c_f = 30, wp_floor 15,
mu_rule "auto"; W-step parameters = cayley.line.thc_gw.WSTEP_DEFAULTS (root attrs wstep_*).
BASES: the pivoted-QR pole / node choices differ between LAPACK builds (gen_lih222_scf_ref.py docstring); this file therefore
stores every basis the loop used so that the C++ test can inject them (like bases.h5 of the T = 0 parity test):
  sigma_basis_w (r_s)          poles of the two-sided gapless Sigma basis (LineBasis(theta, lam, eps, gap=(0, 0)))
  D_zeta_re/_im (nD), D_kind   the bosonic data set D (kinds: 0 line nodes of the gapless bosonic line basis, unmasked; 1 band;
                               2 i nu_n; 3 nu_0), fixed for the run (depends on beta / c_zeta / the line nodes only)
  nu_b (r_b)                   poles of the D-selected bosonic basis (fixed for the run)
  tau_nodes / tau_weights      the tau grid of the LAST iteration (it follows E_max of the current poles; per-iteration ntau in
                               the iteration groups)
Output /iter<N>/ (N = 1..niter), attrs: mu (absolute, Ha), dmu, mu_rule_used, N_mu (N(mu) of the Lehmann G at the new mu, Eq. fT_mu),
  nelec_D (2 tr D / Nk of the thermal hole list), gap (Ha, widest admissible gap of the new poles), dSigma (max |dSigma_c| at the
  unmasked nodes, after mixing; 0 at N = 1), thermal (1 if the window was active during the iteration), wp_used, nD, rank_b,
  npoles_min/max, ntau; datasets window_counts (nk) (of the poles the iteration STARTED from), Sigma_k<k>_re/_im (nsel, nb, nb) =
  total Sigma_c (after mixing) at k = 0 and k = 3 at the fermionic nodes node_index (every 4th node), e_k0 (M) the Lehmann pole
  energies at k = 0 after the iteration (mu-relative to the NEW mu).
Root: mu0 (KS input), mu_start (after the initial "auto" re-centring of the KS poles), mu_rule_start, fz_re/_im (fermionic nodes),
  node_index, fmask (int8, rho beta |zeta| >= c_f).
Usage: gen_finiteT_scf_ref.py [niter=3] [--beta 200] [--out FILE]
Run (Mac): KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 coqui/cayley/scripts/gen_finiteT_scf_ref.py 3"""
import sys, os, time, functools, datetime, subprocess, numpy as np, h5py
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
import cayley.line.driver as drv
from cayley.line.closure import lehmann_from_sigma
from cayley.line.thc_gw import WSTEP_DEFAULTS
from cayley.coqui_io import THC, _c

def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
vals = {opt(o, None) for o in ('--out', '--beta')} - {None}
args = [a for a in sys.argv[1:] if not a.startswith('--') and a not in vals]
niter = int(args[0]) if args else 3
beta = float(opt('--beta', 200.0))
D = ROOT + '/coqui/tests/unit_test_files/gw_line/'
out = opt('--out', D + 'lih222_finiteT_scf_ref.h5')
thc = THC(D + 'lih222_thc/thc.eri.h5')
with h5py.File(D + 'lih222_thc/system.h5', 'r') as f:
    s = f['system']
    H0 = _c(s['H0'][()]); eig = np.array(s['eigval']); qk = np.array(s['qk_to_k2']); mu0 = float(s['mu0'][()]); nelec = float(s['nelec'][()])
X, Z = thc.X[0], thc.Z
nk, nb = X.shape[0], X.shape[2]
assert np.all(qk[0] == np.arange(nk)), "python hartree_exchange assumes q index 0 = Gamma"
drv.lehmann_from_sigma = functools.partial(lehmann_from_sigma, nphi=8)
P = dict(theta=np.deg2rad(20.0), eps=1e-8, lam=6.0, bos_lam=12.0, bos_gap=0.02, sig_gap=(0.02, 0.02), g_gap=(0.0, 0.0), wp=0.11, K=8,
         tol_gram=1e-10, mixing=0.5, nodes_per_ray=120, node_range=(1e-3, 60.0), beta=beta, thermal_tol=1e-8, thermal_floor=30.0,
         thermal_floor_f=30.0, wp_floor=15.0, mu_rule='auto', bos_eps_T=1e-10)
sc = drv.LineSCGW(X, Z, qk, nk, nelec, H0, mu0, **P)
sc.start_from_hamiltonian(np.array([np.diag(eig[k]) for k in range(nk)]).astype(complex))
mu_start = sc.mu
print(f"start: KS mu0 {mu0:.12f} -> {mu_start:.12f} ({sc.mu_rule_used}); window/k {sc.gw.window_counts()}; |D| {len(sc.gw.zD)}, "
      f"rank_b {sc.gw.bos_w.r}, ntau {len(sc.gw.tau)}", flush=True)
idx = np.arange(0, len(sc.fz), 4)
try:
    head = subprocess.run(['git', '-C', ROOT + '/coqui', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
except Exception:
    head = '?'
recs = []
t0 = time.time()
for it in range(niter):
    win = sc.gw.window_counts().copy()
    rec = sc.iterate()
    recs.append((rec, win, {k: sc.sigma_total(k)[idx] for k in (0, 3)}, sc.e_leh[0][np.abs(sc.e_leh[0]) < 1e5].copy(), len(sc.gw.tau)))
with h5py.File(out, 'w') as f:
    f.attrs.update(fixture='lih222', generator='coqui/cayley/scripts/gen_finiteT_scf_ref.py', cayley_head=head,
                   date=datetime.date.today().isoformat(), niter=niter, mu0=mu0, mu_start=mu_start, mu_rule_start=sc.mu_rule_used,
                   nelec=nelec, theta_deg=20.0, theta_t_deg=10.0, eps=P['eps'], lam=P['lam'], lam_b=P['bos_lam'], bos_eps_T=P['bos_eps_T'],
                   nodes_per_ray=P['nodes_per_ray'], node_tmin=1e-3, node_tmax=60.0, wp=P['wp'], K=P['K'], tol_gram=P['tol_gram'], nphi=8,
                   mixing=P['mixing'], beta=beta, thermal_tol=P['thermal_tol'], thermal_floor=P['thermal_floor'],
                   thermal_floor_f=P['thermal_floor_f'], wp_floor=P['wp_floor'], mu_rule=P['mu_rule'], g_repr='lehmann',
                   **{'wstep_' + k: (np.nan if v is None else v) for k, v in WSTEP_DEFAULTS.items()})
    f['fz_re'] = sc.fz.real; f['fz_im'] = sc.fz.imag; f['node_index'] = idx.astype(np.int64); f['fmask'] = sc.fmask.astype(np.int8)
    f['sigma_basis_w'] = sc.bt.w
    f['D_zeta_re'] = sc.gw.zD.real; f['D_zeta_im'] = sc.gw.zD.imag; f['D_kind'] = sc.gw.kD; f['nu_b'] = sc.gw.bos_w.nu
    f['tau_nodes'] = sc.gw.tau; f['tau_weights'] = sc.gw.wtau
    for it, (rec, win, sig, e0, ntau) in enumerate(recs):
        g = f.create_group(f'iter{it + 1}')
        g.attrs.update(mu=rec['mu'], dmu=rec['dmu'], mu_rule_used=rec['mu_rule'], N_mu=rec['N_mu'], nelec_D=rec['nelec'],
                       gap=rec['gap_eV'] / 27.211386, dSigma=rec['dSigma'], thermal=int(rec['thermal']), wp_used=rec['wp'], nD=rec['nD'],
                       rank_b=rec['rank_b'], npoles_min=min(rec['npoles']), npoles_max=max(rec['npoles']), ntau=ntau)
        g['window_counts'] = np.array(win, np.int64)
        for k, S in sig.items():
            g[f'Sigma_k{k}_re'] = S.real; g[f'Sigma_k{k}_im'] = S.imag
        g['e_k0'] = e0
print(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB): {niter} iterations in {time.time() - t0:.0f} s")
