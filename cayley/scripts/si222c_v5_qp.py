#!/usr/bin/env python3
"""V5: quasiparticle energies of the line scGW (Lehmann poles of the upfolded G, weight > wmin) per k vs CoQui's Pade QP energies
(scf/iterN/qp_approx/E_ska of the Matsubara scGW). Part 1 (this script, local): line poles from a state file -> npz.
Usage: si222c_v5_qp.py [state.npz] [K=24] [qp_approx.npz (optional: E_ska (nk,nb) Ha, mu)]"""
import sys, os, numpy as np
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
from cayley.coqui_io import Checkpoint
from cayley.line.line_dlr import LineBasis
from cayley.line.closure import fit_sigma_sectors, lehmann_from_sigma, chemical_potential
HA = 27.211386
state = sys.argv[1] if len(sys.argv) > 1 else ROOT + '/results/si222c_line_scgw_th20_K24_state.npz'
K = int(sys.argv[2]) if len(sys.argv) > 2 else 24
qpf = sys.argv[3] if len(sys.argv) > 3 else ROOT + '/results/si222c_qp_approx.npz'
z = np.load(state, allow_pickle=True); fz, Sp, Sh, F, mu_l = z['fz'], z['Sig_p'], z['Sig_h'], z['F'], float(z['mu'])
cm = Checkpoint(ROOT + '/data/si222c_scgw/si222c.mbpt.h5'); nk, nb = cm.nk, cm.nb
th = np.deg2rad(20); eps = 1e-10; lam = 6.0; wp = 0.11
bp = LineBasis(th, lam=lam, eps=eps, gap=(lam, 0.02), tmax=60.0); bh = LineBasis(th, lam=lam, eps=eps, gap=(0.02, lam), tmax=60.0)
out = ROOT + f'/results/si222c_line_qp_K{K}.npz'
if os.path.exists(out):
    q = np.load(out, allow_pickle=True); E = list(q['E']); Wg = list(q['W']); mu_l = float(q['mu'])
else:
    E, Wg = [], []
    for ik in range(nk):
        w, g = fit_sigma_sectors(bp, bh, fz, Sp[ik], Sh[ik])
        Hrel = cm.H0[0, ik] + F[ik] - mu_l * np.eye(nb)
        e, v, info = lehmann_from_sigma(Hrel, w, g, wp, K, tol_gram=1e-10)
        E.append(e); Wg.append((np.abs(v) ** 2).sum(0)); print(f"k={ik}: {len(e)} poles, held-out {info['heldout_err']:.1e}", flush=True)
    np.savez(out, E=np.array(E, dtype=object), W=np.array(Wg, dtype=object), mu=mu_l)
print(f"line scGW QP-like poles (weight > 0.3) within [-4, +6] eV of mu = {mu_l:.6f} Ha:")
vbm = -np.inf; cbm = np.inf
for ik in range(nk):
    e, w = np.asarray(E[ik], float), np.asarray(Wg[ik], float); m = (w > 0.3) & (e > -4 / HA) & (e < 6 / HA)
    vbm = max(vbm, e[m & (e < 0)].max() if np.any(m & (e < 0)) else -np.inf); cbm = min(cbm, e[m & (e > 0)].min() if np.any(m & (e > 0)) else np.inf)
    print(f"  k={ik}: " + "  ".join(f"{ee*HA:+.3f}({ww:.2f})" for ee, ww in zip(e[m], w[m])))
print(f"line scGW: VBM {vbm*HA:+.3f} eV, CBM {cbm*HA:+.3f} eV (rel. mu) -> fundamental gap {(cbm-vbm)*HA:.3f} eV")
if os.path.exists(qpf):
    q = np.load(qpf); Eqp = q['E_ska']; mu_q = float(q['mu'])
    print(f"\nCoQui Pade QP energies (Matsubara scGW), mu_qp {mu_q:.6f} Ha; comparison per k (line pole with weight>0.3 nearest to each Pade level within 1 eV):")
    for ik in range(nk):
        e, w = np.asarray(E[ik], float), np.asarray(Wg[ik], float); m = w > 0.3
        row = []
        for a in range(nb):
            eq = Eqp[ik, a] - mu_q
            if abs(eq) > 6 / HA: continue
            j = np.argmin(np.abs(e[m] - eq)) if m.any() else None
            if j is not None and abs(e[m][j] - eq) < 1 / HA: row.append(f"{eq*HA:+.3f}->{e[m][j]*HA:+.3f}({(e[m][j]-eq)*HA*1000:+.0f}meV)")
            else: row.append(f"{eq*HA:+.3f}->?")
        print(f"  k={ik}: " + "  ".join(row))
    occ = Eqp < mu_q
    print(f"Pade: VBM {(Eqp[occ].max()-mu_q)*HA:+.3f} eV, CBM {(Eqp[~occ].min()-mu_q)*HA:+.3f} eV -> gap {(Eqp[~occ].min()-Eqp[occ].max())*HA:.3f} eV")
