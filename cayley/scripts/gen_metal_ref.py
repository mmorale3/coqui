#!/usr/bin/env python3
"""Finite-T references of the metallic fixture svo222 (plan S8c task 0, "as S8b.1c") and the Casida precompute of the S8c
go/no-go study. The reference file has EXACTLY the layout of gen_finiteT_ref.py (which is run unchanged, with --no-verify:
the line checks of that script exercise the line W step, under redesign after S8b.1), with two differences:
  * the finite-T Casida is cayley.metal.casida_hermitian (Hermitian form of the pseudo-Hermitian RPA problem; agrees with
    finite_t.casida_from_transitions to 1e-14 on lih222, ~3x faster; svo222 has Nt ~ 9.5e3 / 1.1e4 transitions per q at
    beta 200 / 100), through a disk cache shared with the go/no-go study (dev/s8c_gonogo.py);
  * extra attrs per beta group: ph_asym_pi = max_q,n |Pi(q, i nu_n) - Pi(q, i nu_n)^T| / max|Pi| (n = 0, 1, 2, 5, 20; transition
    sum + the Matsubara dPi at n = 0): the PH asymmetry allowance of test T3 (CoQui's half tau grid assumes Pi = Pi^T), and
    casida_route.
Inputs: tests/unit_test_files/gw_line/svo222_thc/{thc.eri.h5, system.h5} (C++ dump: test_gw_line_scf "[.gw_line_dump_svo222]";
QE xml reader with no_q_sym: Z(q) for all 8 q, X(k) of the full BZ from the rotated IBZ orbitals).

Usage:
  gen_metal_ref.py casida <fx> <beta> <iq> [--cache DIR]      one cached Casida (run the 8 q of a beta in parallel processes)
  gen_metal_ref.py ref <fx> [--betas 200,100] [--cache DIR] [--out F]   the reference h5 (gen_finiteT_ref.py, cached Casida)
  gen_metal_ref.py merge <fx> <a.h5> <b.h5> [--out F]          beta groups of b added to (a copy of) a
Default cache: $CAYLEY_ROOT/data/casida_cache/<fx>. Run (Mac): KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 ...
"""
import sys, os, runpy, numpy as np, h5py
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
from cayley.coqui_io import THC
from cayley import finite_t as ft, metal

mode, fx = sys.argv[1], sys.argv[2]
opt = lambda nm, d: sys.argv[sys.argv.index(nm) + 1] if nm in sys.argv else d
cache = opt('--cache', f'{ROOT}/data/casida_cache/{fx}')
D = ROOT + '/coqui/tests/unit_test_files/gw_line/'
TTOL, DEG = 1e-8, 1e-8                      # as gen_finiteT_ref.py
casida = metal.cached(metal.casida_hermitian, cache)


def load(fx):
    thc = THC(D + f'{fx}_thc/thc.eri.h5')
    with h5py.File(D + f'{fx}_thc/system.h5', 'r') as f:
        s = f['system']; eig = np.array(s['eigval']); qk = np.array(s['qk_to_k2']); nelec = float(s['nelec'][()])
    return thc.X[0], thc.Z, eig, qk, nelec


if mode == 'casida':
    beta, iq = float(sys.argv[3]), int(sys.argv[4])
    X, Z, eig, qk, nelec = load(fx)
    mu0, rule = ft.mu0_auto(eig, nelec, beta, TTOL)
    tr = ft.transitions(X, eig - mu0, qk, iq, beta, deg_tol=DEG)
    print(f"{fx} beta {beta:g} q {iq}: mu0 {mu0:.12f} ({rule}) Nt {len(tr['E'])}", flush=True)
    lam, al, be, info = casida(tr, Z[iq])
    zs = np.concatenate([2j * np.pi * np.array([1, 2, 5]) / beta, [0.1j, 1j, 10j, 0.05 + 0.3j, 1e-300j]])
    Wd = ft.dyson_w(Z[iq], ft.pi_transition(tr, zs)); Wc = ft.casida_w(lam, al, be, zs)
    print(f"  Casida W vs Dyson[transition sum] {np.abs(Wd - Wc).max() / np.abs(Wd).max():.1e}  [{info.get('seconds', 0):.0f}s]", flush=True)
elif mode == 'ref':
    betas = opt('--betas', '200,100')
    ft.casida_from_transitions = casida                    # the generator calls ft.casida_from_transitions(tr, Z[iq])
    out = opt('--out', D + f'{fx}_finiteT_ref.h5')
    sys.argv = ['gen_finiteT_ref.py', fx, '--betas', betas, '--no-verify', '--out', out]
    runpy.run_path(ROOT + '/coqui/cayley/scripts/gen_finiteT_ref.py', run_name='__main__')
    X, Z, eig, qk, nelec = load(fx)
    with h5py.File(out, 'a') as F:
        F.attrs['casida_route'] = 'cayley.metal.casida_hermitian (via scripts/gen_metal_ref.py)'
        F.attrs['generator'] = 'coqui/cayley/scripts/gen_metal_ref.py -> gen_finiteT_ref.py --no-verify'
        for b in [float(x) for x in betas.split(',')]:
            mu0, _ = ft.mu0_auto(eig, nelec, b, TTOL)
            trs = [ft.transitions(X, eig - mu0, qk, iq, b, deg_tol=DEG) for iq in range(eig.shape[0])]
            ph = metal.pi_ph_asymmetry(trs, b)
            F[f'beta_{int(b)}'].attrs['ph_asym_pi'] = ph
            print(f"beta {b:g}: PH asymmetry max|Pi - Pi^T|/max|Pi| = {ph:.2e}", flush=True)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB)")
elif mode == 'merge':                                       # merge <fx> <a.h5> <b.h5>: beta groups of b added to a copy of a
    import shutil
    a, b = sys.argv[3], sys.argv[4]
    out = opt('--out', D + f'{fx}_finiteT_ref.h5')
    if os.path.abspath(a) != os.path.abspath(out): shutil.copy(a, out)
    with h5py.File(out, 'a') as F, h5py.File(b, 'r') as G:
        for g in G:
            if g.startswith('beta_') and g not in F: G.copy(g, F)
        F.attrs['betas'] = np.array(sorted({float(x) for x in F.attrs['betas']} | {float(x) for x in G.attrs['betas']}, reverse=True))
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB): groups {[g for g in h5py.File(out, 'r') if g.startswith('beta_')]}")
else:
    raise SystemExit(__doc__)
