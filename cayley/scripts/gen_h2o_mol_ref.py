#!/usr/bin/env python3
"""Python reference for the C++ line scGW driver on a MOLECULE (test_gw_line_mol "[gw_line][mol]", D1 of notes/demo_plan.md).

Fixture tests/unit_test_files/gw_line/h2o_ccpvdz_mo/ (written by scripts/molecules/pyscf_to_coqui.py, MO basis, RHF):
  pyscf.h5      the pyscf mean field (H0, eigval in the MO basis, S = 1)
  thc.eri.h5    CoQui LS-THC from the GDF Cholesky vectors (nIpts 288 = 12 nbnd, thresh 0), X (1, 1, Np, nb), Z (1, Np, Np)
  bases.h5      the C++ real-pole bases for these settings (test_gw_line_mol "[.gw_line_dump_mol]"); python's bases are
                REPLACED by them (pivoted-QR near ties differ between LAPACKs, see gen_lih222_scf_ref.py).
Settings (= mol_params() of test_gw_line_mol.cpp; the D1 molecular rules): theta 20 deg, eps 1e-8, lam 55 (support rule
  1.2 max(e_max - 2 e_min + mu, 2 e_max - e_min - mu)), node range [1e-3, 110] (node_tmax = 2 lam), 120 nodes per ray,
  lam_b 110 (= C++ auto 2 g_emax), sigma_gap = bos_gap = 0.02, wp 0.35, K 24, tol_gram 1e-10, nphi 8, mixing 0.5,
  g_repr "lehmann" with the C++ pruning defaults (g_emax = lam, g_wtol 1e-12, g_emin_frac 0.5, g_wsmall 1e-4); start from
  the HF eigenvalues at mu0 = (e_HOMO + e_LUMO) / 2.
  (The compressed-G path of the prototype is NOT usable here: with node_tmax = 2 lam the LS fit of G does not pin the total
  weight, the electron count drifts to 10.03 and the loop diverges after ~8 iterations; see LineSCGW g_repr.)
Output tests/unit_test_files/gw_line/h2o_mol_ref.h5: F0 (V_H + Sigma_x of the HF density from the THC, (nb, nb)), eps_hf;
  per iteration mu, gap (Ha), nelec, nelec_lehmann, dSigma, and the total Sigma at every 4th fermionic node
  (Sigma_k0_re/_im (niter, nsel, nb, nb), node_index).
Noise floor (--noise A [--seed S], with --out): the Sigma of iteration 1 at the nodes is multiplied elementwise by
  (1 + A xi), xi ~ N(0, 1), before its closure: the spread of mu / gap / Sigma at iterations 2-3 over seeds at A ~ the iteration-1 C++ - python
  Sigma difference is the floor of the [gw_line][mol] gates (the closure amplifies any Sigma noise, cf. [.scf_noise] on lih222).
Usage: gen_h2o_mol_ref.py [niter=3] [--out FILE] [--noise A] [--seed S]"""
import sys, os, time, functools, datetime, subprocess, numpy as np, h5py
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
import cayley.line.driver as drv
from cayley.line.closure import lehmann_from_sigma
from cayley.coqui_io import THC, _c

def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
vals = {opt(o, None) for o in ('--out', '--noise', '--seed')} - {None}
args = [a for a in sys.argv[1:] if not a.startswith('--') and a not in vals]
noise = float(opt('--noise', 0.0)); rng = np.random.default_rng(int(opt('--seed', 1)))
niter = int(args[0]) if args else 3
D = ROOT + '/coqui/tests/unit_test_files/gw_line/'
F = D + 'h2o_ccpvdz_mo/'

def rd(fn, name):
    with h5py.File(fn, 'r') as f:
        d = f[name]; a = d[()]
        return a[..., 0] + 1j * a[..., 1] if '__complex__' in d.attrs else a

H0 = rd(F + 'pyscf.h5', 'SCF/H0')[0, 0]; eps = np.asarray(rd(F + 'pyscf.h5', 'SCF/eigval')[0, 0]).real
nelec = int(round(float(np.real(rd(F + 'pyscf.h5', 'nelec')))))
H0 = 0.5 * (H0 + H0.conj().T)
thc = THC(F + 'thc.eri.h5'); X = np.asarray(thc.X[0]); Z = np.asarray(thc.Z)                 # (1, Np, nb), (1, Np, Np)
nb = len(eps); ho = nelec // 2 - 1; mu0 = 0.5 * (eps[ho] + eps[ho + 1])
P = dict(theta_deg=20.0, eps=1e-8, lam=55.0, lam_b=110.0, sigma_gap=0.02, bos_gap=0.02, g_gap=0.0, nodes_per_ray=120,
         node_tmin=1e-3, node_tmax=110.0, wp=0.35, K=24, tol_gram=1e-10, nphi=8, mixing=0.5, g_repr='lehmann',
         g_wtol=1e-12, g_emin_frac=0.5, g_wsmall=1e-4)
drv.lehmann_from_sigma = functools.partial(lehmann_from_sigma, nphi=P['nphi'])
sc = drv.LineSCGW(X, Z, np.zeros((1, 1), int), 1, nelec, H0[None], mu0, theta=np.deg2rad(P['theta_deg']), eps=P['eps'],
                  lam=P['lam'], bos_lam=P['lam_b'], bos_gap=P['bos_gap'], sig_gap=(P['sigma_gap'],) * 2, g_gap=(0.0, 0.0),
                  wp=P['wp'], K=P['K'], tol_gram=P['tol_gram'], mixing=P['mixing'], nodes_per_ray=P['nodes_per_ray'],
                  node_range=(P['node_tmin'], P['node_tmax']), g_repr='lehmann', g_wtol=P['g_wtol'],
                  g_emin_frac=P['g_emin_frac'], g_wsmall=P['g_wsmall'])
bases_file = F + 'bases.h5'
print(f"  bases from {bases_file}")
with h5py.File(bases_file, 'r') as f:
    for b, key in ((sc.bp, 'sigma_particle_w'), (sc.bh, 'sigma_hole_w'), (sc.gp, 'g_particle_w'), (sc.gh, 'g_hole_w')):
        w = np.array(f[key]); print(f"  basis {key}: python rank {b.r}, C++ rank {len(w)}, max|dw| "
                                     f"{np.abs(b.w - w).max() if len(w) == b.r else float('nan'):.2e}")
        b.w = w; b.r = len(w); b.pos = w > 0; b.zeta = None; b.K = None
    nu = np.array(f['bos_nu']); print(f"  bosonic: python rank {sc.bos.r}, C++ rank {len(nu)}")
    sc.bos.nu = nu; sc.bos.r = len(nu); sc.bos.zeta = _c(f['bos_zeta_nodes'][()])
sc.start_from_hamiltonian(np.diag(eps)[None].astype(complex))
F0 = np.asarray(sc.F[0])
print(f"  iteration 0: max|H0 + F[D_HF] - diag(eps_HF)| = {np.abs(H0 + F0 - np.diag(eps)).max():.3e} Ha (THC exchange error)")
idx = np.arange(0, len(sc.fz), 4)
if noise > 0.0:                                      # perturb Sigma of iteration 1 BEFORE its closure (C++ - python differ there)
    _ss = sc.sigma_sectors
    def _noisy(ik, wres):
        out = _ss(ik, wres)
        if not sc.history:
            out = [x * (1.0 + noise * rng.standard_normal(x.shape)) for x in out]
        return out
    sc.sigma_sectors = _noisy
recs, Sig = [], []
t0 = time.time()
for it in range(niter):
    rec = sc.iterate(); recs.append(rec); Sig.append(sc.sigma_total(0)[idx])
    print(f"  iter {it + 1}: mu {rec['mu']:.9f} gap {rec['gap_eV']:.6f} eV HOMO {rec['mu'] - 0.5 * rec['gap_eV'] / 27.211386:.8f} "
          f"LUMO {rec['mu'] + 0.5 * rec['gap_eV'] / 27.211386:.8f} Ha nelec {rec['nelec']:.8f} dropped {rec['dropped_weight']:.1e} "
          f"pruned {rec['pruned']}", flush=True)
Sig = np.array(Sig)
out = opt('--out', D + 'h2o_mol_ref.h5')
try:
    head = subprocess.run(['git', '-C', ROOT + '/coqui', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
except Exception:
    head = ''
with h5py.File(out, 'w') as f:
    for k, key in dict(mu='mu', nelec='nelec', nelec_lehmann='nelec_exact', dSigma='dSigma').items():
        f[k] = np.array([r[key] for r in recs], float)
    f['gap'] = np.array([r['gap_eV'] / 27.211386 for r in recs], float)
    f['dropped'] = np.array([r['dropped_weight'] for r in recs], float)
    f['node_index'] = idx.astype(np.int64)
    f['Sigma_k0_re'] = Sig.real; f['Sigma_k0_im'] = Sig.imag
    f['F0_re'] = F0.real; f['F0_im'] = F0.imag; f['eps_hf'] = eps
    f.attrs.update(mu0=mu0, nelec=nelec, fixture='h2o_ccpvdz_mo', generator='coqui/cayley/scripts/gen_h2o_mol_ref.py',
                   noise=noise, cayley_head=head, date=datetime.date.today().isoformat(), bases='bases.h5', **P)
print(f"wrote {out}: {niter} iterations in {time.time() - t0:.0f} s")
