"""line_dyson.py against an exact matrix model: Sigma_c with PSD residues, G from the diagonalized upfolded Hamiltonian."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cayley.line.line_dyson import LineDyson, chemical_potential_line, electron_count, _bisect_step, line_quadrature
from cayley.line.line_dlr import LineBasis

THETA = np.deg2rad(20.0)


def model(nb=4, npole=60, seed=3, shift=0.0):
    """H, Sigma poles w, residues g, and the coupling W of Sigma = W (z - w)^-1 W^dagger."""
    r = np.random.default_rng(seed)
    w = np.concatenate([-(0.25 + r.gamma(2.0, 0.4, npole)), 0.3 + r.gamma(2.0, 0.4, npole)])
    W = (r.normal(size=(nb, 2 * npole)) + 1j * r.normal(size=(nb, 2 * npole))) * np.sqrt(0.3 / (2 * npole))
    A = r.normal(size=(nb, nb)) + 1j * r.normal(size=(nb, nb))
    H = (A + A.conj().T) * 0.15 + np.diag(np.linspace(-0.5, 0.5, nb)) + shift * np.eye(nb)
    return H, w, np.einsum('il,jl->lij', W, W.conj()), W


def lehmann(H, w, W):
    e, V = np.linalg.eigh(np.block([[H, W], [W.conj().T, np.diag(w)]]))
    return e, V[:H.shape[0], :]


def exact(e, v, z, side):
    m = (e > 0) if side == '>' else (e < 0)
    K = 1.0 / (z[:, None] - e[m][None, :])
    return np.einsum('zm,im,jm->zij', K, v[:, m], v[:, m].conj())


def nodes():
    t = np.exp(np.linspace(np.log(1e-3), np.log(60.0), 120))
    return np.concatenate([t * np.exp(1j * THETA), t * np.exp(1j * (np.pi - THETA))])


def test_sum_of_sectors_is_dyson():
    H, w, g, W = model()
    e, v = lehmann(H, w, W)
    z = nodes()
    Gp, Gh = LineDyson(H, w, g, THETA).sectors(z)
    assert np.abs(Gp + Gh - exact(e, v, z, '>') - exact(e, v, z, '<')).max() < 1e-11


def test_sector_split_and_density():
    z = nodes()
    for shift in (0.0, 0.2, -0.2):
        H, w, g, W = model(shift=shift)
        e, v = lehmann(H, w, W)
        assert np.abs(e).min() > 1e-3
        ld = LineDyson(H, w, g, THETA)
        Gp, Gh = ld.sectors(z)
        assert np.abs(Gp - exact(e, v, z, '>')).max() < 1e-11, shift
        assert np.abs(Gh - exact(e, v, z, '<')).max() < 1e-11, shift
        assert np.abs(ld.density() - v[:, e < 0] @ v[:, e < 0].conj().T).max() < 1e-10, shift


def test_dmu_moves_H_only():
    """dmu = the exact poles of H - dmu with Sigma_c fixed relative to mu, in G, its sectors and D."""
    H, w, g, W = model()
    z = nodes()
    ld = LineDyson(H, w, g, THETA)
    for dmu in (0.05, -0.1, 0.25):
        e, v = lehmann(H - dmu * np.eye(H.shape[0]), w, W)
        Gp, Gh = ld.sectors(z, dmu)
        scale = 1.0 / max(np.abs(e).min(), 1.0 / np.abs(z).max())       # |G| grows as a pole approaches the vertex (dmu 0.25: |e| 0.006)
        assert np.abs(Gp - exact(e, v, z, '>')).max() < 1e-12 * scale, dmu
        assert np.abs(Gh - exact(e, v, z, '<')).max() < 1e-12 * scale, dmu
        assert np.abs(ld.density(dmu) - v[:, e < 0] @ v[:, e < 0].conj().T).max() < 1e-10, dmu


def test_chemical_potential_plateau():
    """The plateau of N(dmu) contains dmu = 0 for the model's own filling. N is not exactly flat on it (Sigma_c fixed, H moves: the
    QP weights change continuously), so N(mid) is compared with the exact count at mid, and with the target only roughly."""
    H, w, g, W = model()
    e, v = lehmann(H, w, W)
    target = float(2 * (np.abs(v[:, e < 0]) ** 2).sum())
    mid, eh, el, N = chemical_potential_line([LineDyson(H, w, g, THETA)], target, np.array([1.0]))
    assert eh < 0.0 < el, (eh, el)
    assert abs(N - target) < 1e-2, (N, target)
    e2, v2 = lehmann(H - mid * np.eye(H.shape[0]), w, W)
    assert abs(float(2 * (np.abs(v2[:, e2 < 0]) ** 2).sum()) - N) < 1e-9


def test_chemical_potential_rejects_a_step():
    """A target inside one QP step (a pole crossing mu, as in the Mg2Si iteration 1 with too-gapped bases) must raise, not return.
    Two k copies with the crossing pole 2e-5 Ha apart: the 'plateau' between their steps is wider than a few tol (the old guard
    passed it, as on Mg2Si with 8 k) but far below min_gap."""
    H, w, g, W = model()
    e, v = lehmann(H, w, W)
    lds = [LineDyson(H, w, g, THETA), LineDyson(H + 2e-5 * np.eye(H.shape[0]), w, g, THETA)]
    n0 = float(2 * (np.abs(v[:, e < 0]) ** 2).sum())
    _, _, el, _ = chemical_potential_line(lds, n0, np.array([0.5, 0.5]))
    dl = _bisect_step(lds[:1], n0 + 0.5, np.array([1.0]), el, 1.0, 1e-9)  # copy 1 LUMO step (N jumps by 2Z ~ 1)
    target = electron_count(lds, dl + 1e-5, np.array([0.5, 0.5]))        # between the two copies' LUMO steps
    try:
        chemical_potential_line(lds, target, np.array([0.5, 0.5]))
    except RuntimeError as ex:
        assert 'no gap' in str(ex), ex
    else:
        raise AssertionError('a step was accepted as a plateau')


def test_signed_fits_reach_the_imaginary_axis():
    """The signed one-sided fits of the split G reproduce the exact sector functions off the line."""
    H, w, g, W = model()
    e, v = lehmann(H, w, W)
    z = nodes()
    Gp, Gh = LineDyson(H, w, g, THETA).sectors(z)
    bp = LineBasis(THETA, lam=6.0, eps=1e-10, gap=(6.0, 0.0), tmax=60.0)
    bh = LineBasis(THETA, lam=6.0, eps=1e-10, gap=(0.0, 6.0), tmax=60.0)
    cp, ch = bp.fit(z, Gp), bh.fit(z, Gh)
    zi = 1j * np.array([0.01, 0.1, 1.0, 10.0])
    assert np.abs(bp.eval(cp, z) - Gp).max() < 1e-7
    assert np.abs(bp.eval(cp, zi) - exact(e, v, zi, '>')).max() < 1e-6
    assert np.abs(bh.eval(ch, zi) - exact(e, v, zi, '<')).max() < 1e-6


def test_quadrature_integrates_a_pole_near_the_vertex():
    s, ds = line_quadrature()
    for a in (1e-3, 1e-2, 1.0):
        assert abs((ds / (s + a) ** 2).sum() * a - 1.0) < 1e-8, a


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f()
            print('PASS', name)
