#!/usr/bin/env python3
"""S8b.3 hybrid references (plan S8b task 4, notes section 11.6 "Hybrid"): Sigma_c(i w_n) from the tau leg, the Matsubara
density D, N(mu) and mu at fixed Sigma_c, for the C++ S8b.3 tests; with the prototype gates measured before writing.

Oracles (independent of the tau leg): Eq. fT_sigma at every i w_n of the dense set from the exact finite-T Casida poles
(metal.SigmaFT, direct, no pruning), its exact high-frequency moments S1 = sum_p R_p, S2 = sum_p R_p E_p, and the Matsubara sum
of the exact G = [i w_n + dmu - H - Sigma_c^exact(i w_n)]^-1 (finite_t.density_matsubara: free reference f(H - dmu), analytic
1/w^4 tail from S1/S2, w^-6 term eliminated from the partial sums at N/2 and N). H = diag(eig - mu0) (KS, mu-relative; the
C++ test builds it from /eig of <fx>_finiteT_ref.h5). The truncation of the dense set is checked separately (D at 2 w_max).

Gates (prototype, relative to the max over the set; D absolute (D <= 1)):
  ver_sigma_tau_exactW   tau-leg Sigma_c (exact Casida W, KS G, all poles) vs Eq. fT_sigma, all k, all n < N        [1e-12]
  ver_S1, ver_S2         tau-leg end-point moments vs the exact S1 / S2                                               [1e-12]
  ver_D_exactW, ver_N_exactW, ver_dmu_exactW   hybrid D / N / mu (tau-leg Sigma, exact W) vs the exact-G Matsubara sum [1e-10]
  report_trunc_D         D(w_max) - D(2 w_max) (tau-leg Sigma, exact W): truncation of the dense set
  report_sigma_tau_lineW tau-leg Sigma_c with the LINE-ONLY W (redesigned W step, thermal_tol 1e-12) vs Eq. fT_sigma
  report_D_lineW, report_N_lineW, report_dmu_lineW   hybrid D / N / mu with the line W vs the exact-G Matsubara sum
  report_tau_vs_ray      tau-leg vs ray (LineGW.sigma) Sigma_c with the same line W at i w_n >= zeta_T (two transforms)

H5 layout (complex arrays as <name>_re / <name>_im):
  attrs (root): fixture, generator, cayley_head, date, betas, wmax (Ha), sig_tau_nn / sig_tau_per_efold / sig_tau_x0 (the
        Filon-GL tau grid of the Sigma leg: timeray.tau_panels(beta, E_max(G) + max nu, ...)), sigma_k, H ("diag(eig - mu0)"),
        tail6 (1), finiteT_ref (the <fx>_finiteT_ref.h5 holding /eig, mu0, the THC dir)
  /beta_<int(beta)>/
     attrs: beta, mu0 (absolute, = <fx>_finiteT_ref.h5 beta group), nfreq N (dense set n = 0..N-1, w_n = (2n+1) pi/beta,
            w_{N-1} >= wmax), ntau_sigma (tau-leg nodes, exact-W run), dmu_exact, N_exact (at the root), N_exact_mu0 (at dmu 0),
            dmu_lineW, N_lineW, nfreq_2wmax, the ver_* / report_* numbers above
     n_probe (np) int                  probe Matsubara indices (0..3 and log-spaced to N - 1)
     Sigma_probe_exact (nks, np, nb, nb)   Eq. fT_sigma at i w_{n_probe}, k = sigma_k
     Sigma_probe_tau_exactW (nks, np, nb, nb)   the tau leg with the exact Casida W (= the above to ver_sigma_tau_exactW)
     Sigma_probe_tau_lineW (nks, np, nb, nb)    the tau leg with the prototype's line-only W (what the C++ path computes;
                                       its W poles differ by LAPACK build, compare with Sigma_probe_exact at the report level)
     S1_exact, S2_exact (nk, nb, nb)   exact moments of Sigma_c (all k)
     D_exact_mu0 (nk, nb, nb)          exact-G Matsubara density at dmu = 0 (mu = mu0)
     D_exact (nk, nb, nb)              exact-G Matsubara density at the root dmu_exact of N(mu) = nelec (Sigma_c fixed)
     D_lineW (nk, nb, nb)              hybrid density with the line W at its root dmu_lineW
Usage: gen_finiteT_hybrid_ref.py <lih222|lih223> [--betas 50,200] [--wmax 100] [--out FILE] [--no-line]
       gen_finiteT_hybrid_ref.py <fx> --merge F1,F2 --out FILE   (single-beta runs in parallel, merged)
Run (Mac): KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 coqui/cayley/scripts/gen_finiteT_hybrid_ref.py lih222
"""
import sys, os, time, subprocess, datetime, numpy as np, h5py
ROOT = os.environ.get('CAYLEY_ROOT', '/Users/mmorales/Projects/Cayley_real_axis_scGW')
sys.path.insert(0, ROOT + '/coqui/cayley')
from cayley.coqui_io import THC
from cayley import finite_t as ft, metal
from cayley.line.thc_gw import LineGW, HYBRID_DEFAULTS
from cayley.line.line_dlr import BosonicLineBasis


def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


fx = sys.argv[1]
D = ROOT + '/coqui/tests/unit_test_files/gw_line/'
if '--merge' in sys.argv:
    srcs = opt('--merge', '').split(','); out = opt('--out', None)
    with h5py.File(out, 'w') as F:
        bl = []
        for i, src in enumerate(srcs):
            with h5py.File(src, 'r') as f:
                assert f.attrs['fixture'] == fx
                if i == 0:
                    for k, a in f.attrs.items(): F.attrs[k] = a
                for k in f.keys():
                    f.copy(k, F); bl.append(float(f[k].attrs['beta']))
        F.attrs['betas'] = np.array(bl)
    print(f"merged {srcs} -> {out} ({os.path.getsize(out) / 1e6:.2f} MB)"); sys.exit(0)

betas = [float(b) for b in opt('--betas', '50,200').split(',')]
wmax = float(opt('--wmax', 100.0))
do_line = '--no-line' not in sys.argv
out = opt('--out', D + f'{fx}_finiteT_hybrid_ref.h5')
TH, THT = np.deg2rad(20.0), np.deg2rad(10.0)
VTTOL, CZ = 1e-12, 30.0
GATES = dict(ver_sigma_tau_exactW=1e-12, ver_S1=1e-12, ver_S2=1e-12, ver_D_exactW=1e-10, ver_N_exactW=1e-10, ver_dmu_exactW=1e-10)
thc = THC(D + f'{fx}_thc/thc.eri.h5')
with h5py.File(D + f'{fx}_thc/system.h5', 'r') as f:
    s = f['system']; eig = np.array(s['eigval']); qk = np.array(s['qk_to_k2']); nelec = float(s['nelec'][()])
X, Z = thc.X[0], thc.Z
nk, nb = eig.shape; Np = X.shape[1]; qminus = qk[:, 0].copy()
casida = metal.cached(metal.casida_hermitian, f'{ROOT}/data/casida_cache/{fx}', verbose=False)
fref = D + f'{fx}_finiteT_ref.h5'
with h5py.File(fref, 'r') as f:
    sigma_k = [int(x) for x in f.attrs['sigma_k']]
    mu0s = {float(f[g].attrs['beta']): float(f[g].attrs['mu0']) for g in f if g.startswith('beta_')}
    assert np.array_equal(np.array(f['eig']), eig)
try:
    head = subprocess.run(['git', '-C', ROOT + '/coqui', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
except Exception:
    head = '?'
rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())
v = np.array([np.eye(nb, dtype=complex)] * nk)
print(f"{fx}: nk {nk} nb {nb} Np {Np} nelec {nelec} betas {betas} wmax {wmax} sigma_k {sigma_k} out {out}", flush=True)


def wr(g, name, a):
    a = np.asarray(a)
    if np.iscomplexobj(a):
        g[name + '_re'] = a.real; g[name + '_im'] = a.imag
    else:
        g[name] = a


def exact_hf(SF):
    """S1, S2 of Eq. fT_sigma from metal.SigmaFT.terms (blocked)."""
    S1 = np.zeros((nb, nb), complex); S2 = np.zeros((nb, nb), complex)
    for T1, E, w, T2 in SF.terms():
        S1 += (T1 * w[None, :]) @ T2; S2 += (T1 * (w * E)[None, :]) @ T2
    return S1, S2


groups = {}
for beta in betas:
    T0 = time.time(); res = {}; data = {}
    mu0 = mu0s[beta]
    mu_chk, _ = ft.mu0_auto(eig, nelec, beta, 1e-8)
    assert abs(mu_chk - mu0) < 1e-12, (mu_chk, mu0)
    e = eig - mu0
    H = np.array([np.diag(e[k]) for k in range(nk)]).astype(complex)
    cas = [casida(ft.transitions(X, e, qk, iq, beta, deg_tol=1e-8), Z[iq])[:3] for iq in range(nk)]
    n, iw = ft.matsubara_set(beta, wmax); N = len(iw)
    n2, iw2 = ft.matsubara_set(beta, 2 * wmax)
    npr = np.unique(np.concatenate([np.arange(4), np.round(np.exp(np.linspace(np.log(4), np.log(N - 1), 16))).astype(int)]))
    res.update(beta=beta, mu0=mu0, nfreq=N, nfreq_2wmax=len(iw2))
    print(f"\n[beta {beta:g}] mu0 {mu0:.12f}, dense set N {N} (w_max {wmax:g} Ha; {len(iw2)} at 2 w_max), Casida r per q "
          f"{[int((c[0] > 0).sum()) for c in cas]}  [{time.time() - T0:.0f}s]", flush=True)
    # ---------------------------------------------------------------- oracle: Eq. fT_sigma at every i w_n, exact moments
    Sex = np.zeros((nk, N, nb, nb), complex); S1x = np.zeros((nk, nb, nb), complex); S2x = np.zeros_like(S1x)
    for ik in range(nk):
        SF = metal.SigmaFT(X, e, qk, qminus, cas, beta, ik)
        Sex[ik] = SF.evaluate(zeta=iw, prune=0.0)['zeta']
        S1x[ik], S2x[ik] = exact_hf(SF)
    print(f"  exact Sigma_c at all {N} i w_n and S1, S2 for {nk} k  [{time.time() - T0:.0f}s]", flush=True)
    with h5py.File(fref, 'r') as f:                         # cross-check with the S8b.1b reference (eig-route Casida)
        g = f[f'beta_{int(beta)}']; wn_ref = np.array(g['w_n']); Sref = np.array(g['Sigma_iw_re']) + 1j * np.array(g['Sigma_iw_im'])
    sel = wn_ref < N
    res['ver_sigma_exact_vs_finiteT_ref'] = max(rel(Sex[k][wn_ref[sel]], Sref[j][sel]) for j, k in enumerate(sigma_k))
    # ---------------------------------------------------------------- tau leg with the exact Casida W
    gw = LineGW(X, Z, qk, nk, mu0, TH, THT, BosonicLineBasis(TH, 4.0, eps=1e-10, gap=0.0),
                np.array([1j]), wstep=None)
    gw.set_poles(e, v, beta=beta, thermal_tol=VTTOL, thermal_floor=CZ)
    wf = [(c[1][:, c[0] > 0], c[2][c[0] > 0]) for c in cas]; nuf = [c[0][c[0] > 0] for c in cas]
    iw_all = np.concatenate([iw, iw2[N:]])
    r = gw.sigma_tau_leg(wf, iw_all, qminus=qminus, nu=nuf)
    St = r['sigma'][:, :N]; St2 = r['sigma']
    res.update(ntau_sigma=r['ntau'], ver_sigma_tau_exactW=max(rel(St[k], Sex[k]) for k in range(nk)),
               ver_S1=max(rel(r['S1'][k], S1x[k]) for k in range(nk)), ver_S2=max(rel(r['S2'][k], S2x[k]) for k in range(nk)))
    print(f"  tau leg (exact W, ntau {r['ntau']}): Sigma_c vs Eq. fT_sigma {res['ver_sigma_tau_exactW']:.1e}, S1 {res['ver_S1']:.1e}, "
          f"S2 {res['ver_S2']:.1e}; exact Sigma vs finiteT_ref Sigma_iw {res['ver_sigma_exact_vs_finiteT_ref']:.1e}  [{time.time() - T0:.0f}s]", flush=True)
    # ---------------------------------------------------------------- Matsubara density: exact G vs tau-leg G
    _, Dx0, Nx0, _ = ft.density_matsubara(H, Sex, iw, beta, S1x, S2x, dmu=0.0)
    dmx, Dx, Nx, ix = ft.density_matsubara(H, Sex, iw, beta, S1x, S2x, nelec=nelec)
    _, Dt0, Nt0, _ = ft.density_matsubara(H, St, iw, beta, r['S1'], r['S2'], dmu=0.0)
    dmt, Dt, Nt, it = ft.density_matsubara(H, St, iw, beta, r['S1'], r['S2'], nelec=nelec)
    _, Dt2, _, _ = ft.density_matsubara(H, St2, iw2, beta, r['S1'], r['S2'], dmu=0.0)
    res.update(dmu_exact=dmx, N_exact=Nx, N_exact_mu0=Nx0, N_exact_trace=ix['N_trace'],
               ver_D_exactW=max(float(np.abs(Dt0 - Dx0).max()), float(np.abs(Dt - Dx).max())),
               ver_N_exactW=max(abs(Nt0 - Nx0), abs(Nt - Nx)), ver_dmu_exactW=abs(dmt - dmx),
               report_trunc_D=float(np.abs(Dt0 - Dt2).max()), report_tail_max=ix['tail_max'])
    print(f"  density: exact G at mu0: N {Nx0:.14f}; root dmu {dmx:+.3e} Ha N {Nx:.14f} (trace route {ix['N_trace']:.14f}); "
          f"tau leg vs exact: D {res['ver_D_exactW']:.1e} N {res['ver_N_exactW']:.1e} dmu {res['ver_dmu_exactW']:.1e}; "
          f"truncation D(w_max) - D(2 w_max) {res['report_trunc_D']:.1e}; tail {ix['tail_max']:.1e}  [{time.time() - T0:.0f}s]", flush=True)
    data.update(n_probe=npr, Sigma_probe_exact=Sex[sigma_k][:, npr], Sigma_probe_tau_exactW=St[sigma_k][:, npr],
                S1_exact=S1x, S2_exact=S2x, D_exact_mu0=Dx0, D_exact=Dx)
    del St, St2, r
    # ---------------------------------------------------------------- the line-only W (redesigned W step) in the tau leg
    if do_line:
        wres, _ = gw.w_step(qminus)
        print(f"  line W step: |D| {len(gw.zD)}, rank_b {gw.bos_w.r}  [{time.time() - T0:.0f}s]", flush=True)
        rl = gw.sigma_tau_leg(wres, iw, qminus=qminus)
        Sl = rl['sigma']
        dml, Dl, Nl, _ = ft.density_matsubara(H, Sl, iw, beta, rl['S1'], rl['S2'], nelec=nelec)
        _, Dl0, Nl0, _ = ft.density_matsubara(H, Sl, iw, beta, rl['S1'], rl['S2'], dmu=0.0)
        mT = np.abs(iw) >= gw.zeta_T
        ray = max(rel(gw.sigma(k, wres, iw[mT][:200], qminus=qminus), Sl[k][mT][:200]) for k in sigma_k)
        res.update(dmu_lineW=dml, N_lineW=Nl, report_sigma_tau_lineW=max(rel(Sl[k], Sex[k]) for k in range(nk)),
                   report_D_lineW=max(float(np.abs(Dl0 - Dx0).max()), float(np.abs(Dl - Dx).max())),
                   report_N_lineW=max(abs(Nl0 - Nx0), abs(Nl - Nx)), report_dmu_lineW=abs(dml - dmx), report_tau_vs_ray=ray)
        print(f"  line W in the tau leg: Sigma_c vs Eq. fT_sigma {res['report_sigma_tau_lineW']:.1e}; D {res['report_D_lineW']:.1e} "
              f"N {res['report_N_lineW']:.1e} dmu {res['report_dmu_lineW']:.1e}; tau leg vs ray Sigma (same W, i w_n >= zeta_T) {ray:.1e}  "
              f"[{time.time() - T0:.0f}s]", flush=True)
        data.update(Sigma_probe_tau_lineW=Sl[sigma_k][:, npr], D_lineW=Dl)
        del wres, Sl
    for k, gt in GATES.items():
        if res[k] > gt:
            print(f"  GATE FAILED: {k} = {res[k]:.1e} > {gt:g}", flush=True); sys.exit(2)
    groups[beta] = (res, data)
    del Sex, cas

with h5py.File(out, 'w') as F:
    F.attrs.update(fixture=fx, generator='coqui/cayley/scripts/gen_finiteT_hybrid_ref.py', cayley_head=head,
                   date=datetime.date.today().isoformat(), betas=np.array(betas), wmax=wmax, sigma_k=np.array(sigma_k),
                   H='diag(eig - mu0)', tail6=1, finiteT_ref=f'{fx}_finiteT_ref.h5', nelec=nelec,
                   **{k: v for k, v in HYBRID_DEFAULTS.items()})
    for beta, (res, data) in groups.items():
        g = F.create_group(f'beta_{int(beta)}')
        g.attrs.update(res)
        for k, a in data.items(): wr(g, k, a)
print(f"\nwrote {out} ({os.path.getsize(out) / 1e6:.2f} MB)")
