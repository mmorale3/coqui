"""Driver start rule of g_repr "line_dyson" (default at T = 0 for scgw / scgw0) on a small random THC model: the first iteration is
a closure warm-up, then line Dyson; an iteration without a plateau is done by the closure and recorded (ld_strict raises).
Run: python3 tests/test_line_dyson_driver.py"""
import sys, os, functools, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import cayley.line.driver as drv
from cayley.line.driver import LineSCGW
from cayley.line.line_dyson import NoGapError, chemical_potential_line

rng = np.random.default_rng(7)
nk, Np, nb = 2, 6, 4
qk = np.array([[0, 1], [1, 0]]); qminus = np.array([0, 1])
X = (rng.standard_normal((nk, Np, nb)) + 1j * rng.standard_normal((nk, Np, nb))) / np.sqrt(Np)
A = rng.standard_normal((nk, Np, Np)); Z = 0.3 * np.einsum("qij,qkj->qik", A, A) / Np + 0j
e = np.array([[-1.2, -0.4, 0.5, 1.5], [-0.9, -0.3, 0.6, 2.0]])                # gapped, mu-relative
H0 = np.array([np.diag(e[k] + 0.3) for k in range(nk)]).astype(complex)


def make(**kw):
    kw = {'nodes_per_ray': 120, **kw}     # 30 nodes/ray under-determine the fits (60 data, 132 Sigma / 204 G functions): G fit res 2e-3, diverges
    sc = LineSCGW(X, Z, qk, nk, 4.0, H0, 0.3, eps=1e-8, lam=6.0, bos_lam=4.0, K=4, verbose=False, qminus=qminus, **kw)
    sc.start_from_hamiltonian(H0)
    return sc


def test_default_and_warmup():
    sc = make()
    assert sc.g_repr == 'line_dyson'
    assert make(scheme='scgw0').g_repr == 'line_dyson' and make(scheme='evgw').g_repr == 'compressed'
    steps = [sc.iterate()['g_step'] for _ in range(5)]
    assert steps == ['closure (warm-up)'] + ['line_dyson'] * 4, steps
    assert abs(sc.history[-1]['nelec'] - 4.0) < 1e-3, sc.history[-1]['nelec']    # N(mid-plateau): exact only at the fixed point
    dS = [r['dSigma'] for r in sc.history[1:]]
    assert all(b < 0.7 * a for a, b in zip(dS, dS[1:])), dS                     # contracts (x0.5 per iteration on this toy)
    assert max(r['G_fit_res'] for r in sc.history[1:]) < sc.ld_fit_tol


def test_no_gap_fallback_and_strict():
    orig = drv.chemical_potential_line
    drv.chemical_potential_line = functools.partial(chemical_potential_line, min_gap=10.0)   # no plateau is wide enough
    try:
        sc = make()
        steps = [sc.iterate()['g_step'] for _ in range(2)]
        assert steps == ['closure (warm-up)', 'closure (line Dyson: no gap)'], steps
        sc = make(ld_strict=True); sc.iterate()
        try:
            sc.iterate()
        except NoGapError:
            pass
        else:
            raise AssertionError('ld_strict did not raise')
    finally:
        drv.chemical_potential_line = orig


def test_g_fit_fallback():
    """A G^>/G^< fit worse than ld_fit_tol: nothing is committed by line Dyson, the closure does the iteration."""
    sc = make(ld_fit_tol=-1.0)                                                    # every fit fails the tolerance
    steps = [sc.iterate()['g_step'] for _ in range(2)]
    assert steps == ['closure (warm-up)', 'closure (line Dyson: G fit)'], steps
    assert abs(sc.history[-1]['nelec'] - 4.0) < 2e-2


def test_small_gap_node_floor():
    """KS gap ~0.6 eV, basis gaps 0.002 Ha: with the lowest node at 1e-3 the G-fit coefficients grow x5, x37, ... (the Mg2Si failure)
    and the guard hands the step to the closure; at 1e-4 (20x below the basis gaps) line Dyson contracts with max|c| < 1e2."""
    e2 = np.array([[-1.2, -0.012, 0.012, 1.5], [-0.9, -0.010, 0.014, 2.0]])
    H2 = np.array([np.diag(e2[k] + 0.3) for k in range(nk)]).astype(complex)
    def run(nr, npr, n):
        sc = LineSCGW(X, Z, qk, nk, 4.0, H2, 0.3, eps=1e-8, lam=6.0, bos_lam=4.0, K=4, verbose=False, qminus=qminus, nodes_per_ray=npr,
                      node_range=nr, bos_gap=0.002, sig_gap=(0.002, 0.002))
        sc.start_from_hamiltonian(H2)
        return sc, [sc.iterate()['g_step'] for _ in range(n)]
    sc, steps = run((1e-4, 60.0), 150, 5)
    assert steps == ['closure (warm-up)'] + ['line_dyson'] * 4, steps
    dS = [r['dSigma'] for r in sc.history[1:]]
    assert all(b < 0.7 * a for a, b in zip(dS, dS[1:])) and max(r['cmax'] for r in sc.history[1:]) < 1e2, (dS, [r['cmax'] for r in sc.history[1:]])
    sc, steps = run((1e-3, 60.0), 120, 4)
    assert steps[-1] == 'closure (line Dyson: G fit)', steps


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f()
            print('PASS', name)
