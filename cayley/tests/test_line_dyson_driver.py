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
    sc = LineSCGW(X, Z, qk, nk, 4.0, H0, 0.3, eps=1e-8, lam=6.0, bos_lam=4.0, K=4, nodes_per_ray=30, verbose=False, qminus=qminus, **kw)
    sc.start_from_hamiltonian(H0)
    return sc


def test_default_and_warmup():
    sc = make()
    assert sc.g_repr == 'line_dyson'
    assert make(scheme='scgw0').g_repr == 'line_dyson' and make(scheme='evgw').g_repr == 'compressed'
    steps = [sc.iterate()['g_step'] for _ in range(3)]
    assert steps == ['closure (warm-up)', 'line_dyson', 'line_dyson'], steps
    assert abs(sc.history[-1]['nelec'] - 4.0) < 1e-2, sc.history[-1]['nelec']    # N(mid-plateau): exact only at the fixed point


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


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f()
            print('PASS', name)
