#!/usr/bin/env python3
"""Reference data for the C++ bosonic response closure (src/numerics/line_dlr/response_closure.hpp, test_response_closure,
plan session S9b; design study notes/bosonic_closure_design.md section 7).

The oracle is the python design study scripts/closure_design/ (cd_lib.py, run_designs.py, optics.py; outside this repo,
under $CAYLEY_ROOT) and its exact cases results/closure_design/cache/cases.npz (built by cases.py / cases_sieps.py).

Writes <coqui>/tests/unit_test_files/gw_line/response_closure_ref.h5:
  basis20/{nu, zeta_re, zeta_im}   BosonicLineBasis(20 deg, lam 12, eps 1e-10, gap 0.02) of the study (rank 153, 306 nodes)
  basis10/{...}                    BosonicLineBasis(10 deg, lam 12, eps 1e-10, gap 0.02, nline 1200, npole 800)
  omega (4500,)                    the study's real grid (0, 45 eV] in Ha; coarse = every 5th point (900)
  grid_G (1500,)                   NNLS candidate poles, log-spaced in [0.02, 12] Ha
  scales (4,)                      the MB scales 3.3 ... 27.2 eV (log-spaced), Ha
  case<i>/ (one per (case, noise, angle)):
     attrs: name, delta, theta, K (K rule), tol_gram, fsum_fit = sum 2 f_j nu_j, f0_fit = f(0) of the odd fit,
            fit_resid (max |K f - data| / max |data|), G_rnorm (NNLS residual 2-norm, column-scaled system), G_npos
     W, r                          the exact odd measure (Omega_s, r_s)
     data_re, data_im (nz)         the (noisy) line data at the basis nodes
     fj (r)                        the odd least-squares fit (real residues, numpy lstsq)
     mb_d<i>, mb_a<i> (i = scale)  the per-scale closure poles / weights (closure_B centre 0, scale w_i, K, tol_gram)
     G_x (1500)                    NNLS residues on grid_G
     curves: {exact, mb, G}_{e01, r05}_{re, im} on the coarse grid (eta = 0.01 Ha and eta = 0.05 omega)
     err_{mb, G}_{e01, r05} (4, 2)  window errors (0-5, 5-10, 10-20, 20-40 eV) of f and of -Im f on the FULL grid (cd_lib.window_errors)
  optics/ (Si-like model 'sieps|h', noise 1e-8, seed 1000, eta = 0.05 omega, routes R1 / R2 with MB and G; optics.py):
     h_re, h_im, m_re, m_im (nz)   line data of h = 1/eps - 1 and m = h/(1+h) = 1 - eps (pointwise); fh, fm their odd fits
     <route>_<method>_<q> (coarse) for q in eps1 eps2 n kappa alpha R loss, route R1 (close h) / R2 (close m),
     method mb / G; exact_<q>; err_<route>_<method> (7 quantities, 4 windows) on the full grid.
Usage: cd coqui/cayley && python3 scripts/gen_response_closure_ref.py   (~2 min)"""
import os, sys, time
import numpy as np
import h5py

here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('CAYLEY_ROOT', os.path.abspath(os.path.join(here, '..', '..', '..')))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'closure_design'))
import cd_lib as L
import run_designs as R
import optics as O
import flatline as FL

HA = L.HA
out = os.path.join(here, '..', '..', 'tests', 'unit_test_files', 'gw_line', 'response_closure_ref.h5')
t0 = time.time()
C = R.load_cases()
om = np.linspace(0, 45 / HA, 4501)[1:]
co = slice(0, None, 5)
grid = np.exp(np.linspace(np.log(0.02), np.log(12.0), 1500))
sc4 = R.scales_log(3.3 / HA, 27.2 / HA, 4)


def k_rule(delta, theta):
    """notes/bosonic_closure_design.md 7.1 item 4: K = min(K_cap, floor(ln(0.1/max(delta,1e-10)) / ln(1/r))),
    K_cap = 48 ln(1/r_20) / ln(1/r)"""
    r = np.tan(np.pi / 4 - np.deg2rad(theta) / 2); r20 = np.tan(np.pi / 4 - np.deg2rad(20.0) / 2)
    kcap = int(48 * np.log(1 / r20) / np.log(1 / r))
    return int(min(kcap, np.floor(np.log(0.1 / max(delta, 1e-10)) / np.log(1 / r))))


def wr_basis(f, name, bos):
    g = f.create_group(name)
    g['nu'] = bos.nu; g['zeta_re'] = bos.zeta.real; g['zeta_im'] = bos.zeta.imag
    g.attrs['theta'] = bos.theta; g.attrs['rank'] = bos.r


def errs(fv, fe):
    e = L.window_errors(om, fv, fe)
    return np.array([e[k] for k in ('0-5', '5-10', '10-20', '20-40')])


with h5py.File(out, 'w') as f:
    b20 = L.bos_basis(12.0); b10 = FL.basis(10.0)
    wr_basis(f, 'basis20', b20); wr_basis(f, 'basis10', b10)
    f['omega'] = om; f['grid_G'] = grid; f['scales'] = np.array(sc4)
    ic = 0
    for (cn, delta, th) in (('si211_sc|q0|const', 0.0, 20), ('si211_sc|q0|const', 1e-8, 20), ('lih222_sc|q1|rand', 1e-8, 20),
                            ('sieps|h', 0.0, 20), ('sieps|h', 1e-8, 20), ('cont', 1e-8, 20), ('sieps|h', 1e-8, 10)):
        ex = C[cn]; bos = b20 if th == 20 else b10
        data = L.add_noise(ex(bos.zeta), delta, np.random.default_rng(1000))
        fj = L.odd_fit(bos, data)
        K = k_rule(delta, th); tg = max(1e-10, 10 * delta)
        mb = R.multiscale_blend(bos.nu, fj, sc4, K, tg)
        G = L.nnls_fit(bos.zeta, data, grid)
        Kz = L.odd_kernel(bos.zeta, bos.nu)
        g = f.create_group(f'case{ic}'); ic += 1
        g.attrs['name'] = cn; g.attrs['delta'] = delta; g.attrs['theta'] = float(th); g.attrs['K'] = K; g.attrs['tol_gram'] = tg
        g.attrs['fsum_fit'] = float((2 * fj * bos.nu).sum()); g.attrs['f0_fit'] = float(-(2 * fj / bos.nu).sum())
        g.attrs['fit_resid'] = float(np.abs(Kz @ fj - data).max() / np.abs(data).max())
        # NNLS residual of the column-scaled real system
        Kg = L.odd_kernel(bos.zeta, grid); A = np.vstack([Kg.real, Kg.imag]); sc = np.linalg.norm(A, axis=0)
        bvec = np.r_[data.real, data.imag]
        g.attrs['G_rnorm'] = float(np.linalg.norm(A @ G.r - bvec)); g.attrs['G_npos'] = int((G.r > 0).sum())
        g['W'] = ex.W; g['r'] = ex.r
        g['data_re'] = data.real; g['data_im'] = data.imag; g['fj'] = fj
        for i, m in enumerate(mb.parts):
            g[f'mb_d{i}'] = m.W; g[f'mb_a{i}'] = m.r
        g['G_x'] = G.r
        for lab, eta in (('e01', 0.01), ('r05', 'r0.05')):
            ev = 0.05 * om if isinstance(eta, str) else np.full(len(om), eta)
            z = om + 1j * ev
            fe, fm, fg = ex(z), mb(z), G(z)
            for nm, v in (('exact', fe), ('mb', fm), ('G', fg)):
                g[f'{nm}_{lab}_re'] = v[co].real; g[f'{nm}_{lab}_im'] = v[co].imag
            g[f'err_mb_{lab}'] = errs(fm, fe); g[f'err_G_{lab}'] = errs(fg, fe)
        print(f"case{ic-1} {cn} delta {delta:g} theta {th}: K {K}, MB err e01 {g[f'err_mb_e01'][:, 0]}, G {g['err_G_e01'][:, 0]} "
              f"({time.time()-t0:.0f} s)", flush=True)
    f.attrs['ncases'] = ic
    # optics (Si-like model), routes R1 / R2
    ex = C['sieps|h']; bos = b20; delta = 1e-8
    cfac = (1.0 / 12.0 - 1.0) / ex(np.array([1e-9j]))[0].real            # = 1 for sieps|h (h(0) = 1/12 - 1 already)
    hfun = lambda z: cfac * ex(z)
    hd = L.add_noise(hfun(bos.zeta), delta, np.random.default_rng(1000)); md = hd / (1 + hd)
    fh = L.odd_fit(bos, hd); fm = L.odd_fit(bos, md)
    K = k_rule(delta, 20.0); tg = max(1e-10, 10 * delta)
    Mh = {'mb': R.multiscale_blend(bos.nu, fh, sc4, K, tg), 'G': L.nnls_fit(bos.zeta, hd, grid)}
    Mm = {'mb': R.multiscale_blend(bos.nu, fm, sc4, K, tg), 'G': L.nnls_fit(bos.zeta, md, grid)}
    z = om * (1 + 0.05j)
    Qx = O.quantities(1.0 / (1.0 + hfun(z)), om)
    g = f.create_group('optics')
    g.attrs['cfac'] = cfac; g.attrs['K'] = K; g.attrs['tol_gram'] = tg; g.attrs['delta'] = delta
    g['h_re'] = hd.real; g['h_im'] = hd.imag; g['m_re'] = md.real; g['m_im'] = md.imag; g['fh'] = fh; g['fm'] = fm
    qn = ('eps1', 'eps2', 'n', 'kappa', 'alpha', 'R', 'loss')
    for q in qn: g[f'exact_{q}'] = Qx[q][co]
    for meth in ('mb', 'G'):
        for route, eps in (('R1', 1.0 / (1.0 + Mh[meth](z))), ('R2', 1.0 - Mm[meth](z))):
            Q = O.quantities(eps, om)
            E = np.zeros((len(qn), 4))
            for iq, q in enumerate(qn):
                g[f'{route}_{meth}_{q}'] = Q[q][co]
                for iw, (a, b) in enumerate(L.WINS):
                    msk = (om * HA > a) & (om * HA <= b)
                    E[iq, iw] = np.abs(Q[q][msk] - Qx[q][msk]).max() / np.abs(Qx[q][msk]).max()
            g[f'err_{route}_{meth}'] = E
            print(f"optics {route} {meth}: eps2 {E[1]}  loss {E[6]}", flush=True)
print(f"wrote {out} ({os.path.getsize(out)/1e6:.2f} MB, {time.time()-t0:.0f} s)")
