#!/usr/bin/env python3
"""Reference data for the C++ line-DLR numerics (src/numerics/line_dlr, test_line_dlr, plan session S1).

Writes <coqui>/tests/unit_test_files/gw_line/line_dlr_ref.h5 with one group per case:
  f0, f1, f2   fermionic LineBasis   (attrs theta, lam, eps, gap_minus, gap_plus, tmin, tmax, nline, npole, rank;
                                      datasets poles (r,), nodes_re/nodes_im (r,))
  b0, b1       bosonic BosonicLineBasis (attrs theta, lam, eps, gap, tmin, tmax, nline, npole, rank;
                                      datasets poles (r,) = nu, nodes_re/nodes_im (min(2r, 2 nline),))
  ray          TimeRay.for_spectrum   (attrs theta_t, emin, decades, smax, smin, per_efold, nn; datasets s, ws)
All energies in Ha, mu-relative; angles in radians.
Usage: cd coqui/cayley && python3 scripts/gen_line_dlr_ref.py"""
import os, sys
import numpy as np
import h5py
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..'))
from cayley.line.line_dlr import LineBasis, BosonicLineBasis
from cayley.line.timeray import TimeRay

out = os.path.join(here, '..', '..', 'tests', 'unit_test_files', 'gw_line', 'line_dlr_ref.h5')
os.makedirs(os.path.dirname(out), exist_ok=True)
th = np.deg2rad(20.0)
NLINE, NPOLE_F, NPOLE_B = 1200, 1500, 800

fermionic = [dict(lam=6.0, eps=1e-8, gap=(0.02, 0.02)),
             dict(lam=6.0, eps=1e-10, gap=(0.02, 0.02)),
             dict(lam=2.2, eps=1e-8, gap=(0.0, 0.0))]
bosonic = [dict(lam=4.0, eps=1e-8, gap=0.02),
           dict(lam=4.0, eps=1e-10, gap=0.02)]

with h5py.File(out, 'w') as f:
    for i, p in enumerate(fermionic):
        b = LineBasis(th, p['lam'], p['eps'], p['gap'], nline=NLINE, npole=NPOLE_F)
        g = f.create_group(f'f{i}')
        g.attrs.update(theta=th, lam=p['lam'], eps=p['eps'], gap_minus=p['gap'][0], gap_plus=p['gap'][1],
                       tmin=1e-4 * p['lam'], tmax=20 * p['lam'], nline=NLINE, npole=NPOLE_F, rank=b.r)
        g['poles'] = b.w; g['nodes_re'] = b.zeta.real; g['nodes_im'] = b.zeta.imag
        print(f'f{i}:', b, flush=True)
    for i, p in enumerate(bosonic):
        b = BosonicLineBasis(th, p['lam'], p['eps'], p['gap'], nline=NLINE, npole=NPOLE_B)
        g = f.create_group(f'b{i}')
        g.attrs.update(theta=th, lam=p['lam'], eps=p['eps'], gap=p['gap'],
                       tmin=1e-4 * p['lam'], tmax=20 * p['lam'], nline=NLINE, npole=NPOLE_B, rank=b.r)
        g['poles'] = b.nu; g['nodes_re'] = b.zeta.real; g['nodes_im'] = b.zeta.imag
        print(f'b{i}:', b, flush=True)
    th_t, emin, decades = np.deg2rad(10.0), 0.02, 40.0
    r = TimeRay.for_spectrum(th_t, emin, decades=decades, smin=1e-5, per_efold=3.0, nn=16)
    g = f.create_group('ray')
    g.attrs.update(theta_t=th_t, emin=emin, decades=decades, smax=decades / (emin * np.sin(th_t)), smin=1e-5,
                   per_efold=3.0, nn=16)
    g['s'] = r.s; g['ws'] = r.ws
    print(f'ray: {len(r)} nodes, smax={g.attrs["smax"]:.6g}')
print('wrote', os.path.normpath(out), f'({os.path.getsize(out)} bytes)')
