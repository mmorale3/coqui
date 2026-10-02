#!/usr/bin/env python3
"""Closure diagnostic at production settings, reusing V3's Sigma on the line (results/si222c_v3_k{ik}.npz):
 sector fits -> moments (vs exact) -> upfold (K) -> upfolded Hamiltonian eigen-decomposition (Lehmann G) ->
 spectral function vs the exact G0W0 A(w) (same H, exact Sigma_c(z)); weight beyond the basis range; QP gap at this k;
 compressed per-sector refit of G (line basis) vs the Lehmann G at the nodes and on the imaginary axis.
Usage: si222c_closure_check.py [ik=0] [theta_deg=20] [eps=1e-8] [wp=0.11]"""
import sys, os, numpy as np
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
from cayley.coqui_io import Checkpoint
from cayley.line.line_dlr import LineBasis
from cayley.line.closure import fit_sigma_sectors, sigma_moments, lehmann_from_sigma, compress_sectors
from cayley.spectral import spectral_function, sigma_from_poles
HA = 27.211386
ik = int(sys.argv[1]) if len(sys.argv) > 1 else 0
theta = np.deg2rad(float(sys.argv[2]) if len(sys.argv) > 2 else 20.0)
eps = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-8
wp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.11
z = np.load(ROOT + f'/results/si222c_v3_k{ik}.npz')
fz, Sl, Sx, C_ex, om, Aex, H = z['fz'], z['Sl'], z['Sx'], z['C_ex'], z['om'], z['Aex'], None
ck = Checkpoint(ROOT + '/data/si222c_nb58_thc1e-4/si222c.mbpt.h5'); mu0 = ck.mu[0]
H = ck.H0[0, ik] + ck.F(1)[0, ik]
# the V3 npz stores Sl (total) only; rebuild the sector parts from the exact structure is not possible here -> fit the TOTAL with
# a two-sided basis and ALSO fit sectors from Sl using the knowledge that Sigma^> and Sigma^< are separately available in V3?
# -> V3 saved only the total; refit per sector by using the exact sector split as reference is circular. Use the total fit
#    (two-sided basis with the Sigma gap) for the moments, which is what matters for the closure.
ps_gap = (2.939 / HA, 1.088 / HA)         # Sigma_c support edges about mu0 at k=0 from V3's log (hole, particle), Ha
basis = LineBasis(theta, lam=3.0, eps=eps, gap=(0.8 * ps_gap[0], 0.8 * ps_gap[1]))
g = basis.fit(fz, Sl)
res = np.abs(basis.eval(g, fz) - Sl).max() / np.abs(Sl).max()
mu_c = mu0 + 0.5 * (ps_gap[1] - ps_gap[0])
C = sigma_moments(basis.w, g, wp, 41, mu_c - mu0)
err = [np.linalg.norm(C[n] - C_ex[n]) / np.linalg.norm(C_ex[0]) for n in range(42)]
print(f"{basis}\n total-Sigma fit residual {res:.1e}; moments vs exact n=4/8/16/24/32: " + " ".join(f"{err[n]:.0e}" for n in (4, 8, 16, 24, 32)))
Hrel = H - mu_c * np.eye(H.shape[0])
eta = 0.01; win = np.abs(om - mu_c) < 0.19
for K in [8, 12, 16, 24]:
    for tol in [10 * eps, 1e-6]:
        e, v, info = lehmann_from_sigma(Hrel, basis.w - (mu_c - mu0), g, wp, K, tol_gram=tol)
        wgt = (np.abs(v) ** 2).sum(0)
        far = wgt[np.abs(e) > 3.0].sum(); near = wgt[np.abs(e) < 0.02].sum()
        # spectral function of the Lehmann G (exact for the upfolded Sigma) vs exact
        A = np.einsum('m,wm->w', wgt, (eta / np.pi) / ((om - mu_c)[:, None] - e[None, :]) ** 2 + 0 * 1j).real if False else \
            (wgt[None, :] * (eta / np.pi) / (((om - mu_c)[:, None] - e[None, :]) ** 2 + eta ** 2)).sum(1)
        trAex = Aex
        # filling and gap at this k alone (8 electrons / 8 k-points -> 4 per k per spin ... use weight 4 per spin at this k)
        order = np.argsort(e); cum = np.cumsum(wgt[order]); j = int(np.searchsorted(cum, 4.0 - 1e-8))
        qp = wgt > 0.1; ef = 0.5 * (e[order][j] + e[order][min(j + 1, len(e) - 1)])
        eh = e[qp & (e <= ef)].max() if np.any(qp & (e <= ef)) else np.nan; el = e[qp & (e > ef)].min() if np.any(qp & (e > ef)) else np.nan
        print(f" K={K:2d} tol_gram={tol:g}: poles {len(e):4d}, held-out {info['heldout_err']:.1e}, weight |e|>3 Ha {far:.1e}, |e|<0.02 Ha {near:.1e}, "
              f"QP gap at k {(el-eh)*HA:.3f} eV, weight sum {wgt.sum():.3f}/58, Tr A error |w-mu|<5 eV {np.abs(A-trAex)[win].max()/trAex[win].max():.1e}")
# compression check for the best case
e, v, info = lehmann_from_sigma(Hrel, basis.w - (mu_c - mu0), g, wp, 16, tol_gram=10 * eps)
gp = LineBasis(theta, lam=3.0, eps=eps, gap=(3.0, 0.01)); gh = LineBasis(theta, lam=3.0, eps=eps, gap=(0.01, 3.0))
w_all, c_all, dropped = compress_sectors(gp, gh, fz, e, v)
Gl = np.einsum('zm,im,jm->zij', 1.0 / (fz[:, None] - e[None, :]), v, v.conj())
Gc = np.einsum('zl,lij->zij', 1.0 / (fz[:, None] - w_all[None, :]), c_all)
zi = 1j * np.exp(np.linspace(np.log(1e-2), np.log(10), 20))
Gli = np.einsum('zm,im,jm->zij', 1.0 / (zi[:, None] - e[None, :]), v, v.conj()); Gci = np.einsum('zl,lij->zij', 1.0 / (zi[:, None] - w_all[None, :]), c_all)
Dm_l = v[:, e < 0] @ v[:, e < 0].conj().T; Dm_c = c_all[w_all < 0].sum(0)
print(f"compression (K=16): G at nodes rel {np.abs(Gc-Gl).max()/np.abs(Gl).max():.1e}, on imag axis {np.abs(Gci-Gli).max()/np.abs(Gli).max():.1e}, "
      f"dropped weight {dropped:.1e}, tr Dm Lehmann {np.trace(Dm_l).real:.4f} vs compressed {np.trace(Dm_c).real:.4f}, max|Dm diff| {np.abs(Dm_c-Dm_l).max():.1e}")
