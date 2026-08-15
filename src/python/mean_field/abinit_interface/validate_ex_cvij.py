"""
validate_ex_cvij.py -- pin the native core-valence exact-exchange formula
against an ABINIT/atompaw ground truth.

Formula (derived; Condon-Shortley closed-shell exchange, real or complex
harmonics give the same shell sum):

  ex_cvij(i,j) = -delta_{l_i l_j} sum_c sum_L (2 l_c + 1) [3j(l_i L l_c;000)]^2
                                             R^L(u_i u_c ; u_j u_c)      [Ha]

with u = r*R radial functions, c running over core SPATIAL shells (n_c,l_c),
and the Slater integral
  R^L(f;g) = INT g(r) [ r^-(L+1) INT_0^r f s^L ds + r^L INT_r^inf f s^-(L+1) ds ] dr.

The minus sign is the ABINIT convention: m_pawdij.F90 pawdijfock copies
pawtab%ex_cvij into dijfock_cv with factor 1, so the stored matrix IS the
(attractive) exchange contribution to Dij.

Ground truth: tests/Pspdir/Al.GGA-PBE-paw-stringent{,.corewf}.xml from an
ABINIT source tree (same atompaw generation & radial grid): valence partial
waves + <exact_exchange_X_matrix> in one file, <ae_core_wavefunction> blocks
in the other.

Usage:  python validate_ex_cvij.py <valence.xml> <corewf.xml>
"""
import sys
import numpy as np
from math import comb

from abinit_pawxml import parse_pawxml, parse_corewf


def w3j000_sq(l1, l2, l3):
    """[3j(l1 l2 l3; 0 0 0)]^2 (Racah closed form; 0 unless triangle + even sum)."""
    J = l1 + l2 + l3
    if J % 2 or l3 < abs(l1 - l2) or l3 > l1 + l2:
        return 0.0
    g = J // 2
    # 3j000 = (-1)^g g!/((g-l1)!(g-l2)!(g-l3)!) * sqrt((J-2l1)!(J-2l2)!(J-2l3)!/(J+1)!)
    from math import factorial as fa
    pref = fa(g) / (fa(g - l1) * fa(g - l2) * fa(g - l3))
    rad = fa(J - 2 * l1) * fa(J - 2 * l2) * fa(J - 2 * l3) / fa(J + 1)
    return pref * pref * rad


def _cumsimp0(F):
    """Cumulative Simpson of an index-space integrand F(i) (uniform spacing 1),
    leading 0; odd intermediate points via the parabola through the triple.
    Same scheme the C++ port uses (paw_onecenter cumulative Simpson)."""
    n = F.size
    out = np.zeros(n)
    for k in range(1, n):
        if k == 1:
            out[1] = 0.5 * (F[0] + F[1])      # first segment: trapezoid
        elif k % 2 == 0:
            out[k] = out[k - 2] + (F[k - 2] + 4.0 * F[k - 1] + F[k]) / 3.0
        else:
            # [k-1, k] segment of the parabola through (k-2, k-1, k)
            out[k] = out[k - 1] + (-F[k - 2] + 8.0 * F[k - 1] + 5.0 * F[k]) / 12.0
    return out


def slater_RL(f, g, r, rab, L):
    """R^L(f; g) as defined above; f,g vanish ~r^2 at the origin (u-forms).
    Simpson in the radial-grid index (weight rab = dr/di)."""
    rs = np.where(r < 1e-30, 1.0, r)
    inner = _cumsimp0(f * rs ** L * rab)                     # INT_0^r f s^L ds
    fol = f / rs ** (L + 1)
    fol[r < 1e-30] = 0.0
    tail_all = _cumsimp0(fol * rab)
    tail = tail_all[-1] - tail_all                           # INT_r^inf
    V = inner / rs ** (L + 1) + rs ** L * tail
    V[r < 1e-30] = 0.0
    return _cumsimp0(g * V * rab)[-1]


def build_ex_cvij_ln(r, rab, u_val, l_val, u_core, l_core):
    """ln-basis (nval x nval) native ex_cvij; -K, l-diagonal."""
    nv = len(l_val)
    X = np.zeros((nv, nv))
    for i in range(nv):
        for j in range(nv):
            if l_val[i] != l_val[j]:
                continue
            acc = 0.0
            for c in range(len(l_core)):
                lc = l_core[c]
                for L in range(abs(l_val[i] - lc), l_val[i] + lc + 1):
                    w = w3j000_sq(l_val[i], L, lc)
                    if w == 0.0:
                        continue
                    RL = slater_RL(u_val[i] * u_core[c], u_val[j] * u_core[c],
                                   r, rab, L)
                    acc += (2 * lc + 1) * w * RL
            X[i, j] = -acc
    return X


def main(val_xml, core_xml):
    p = parse_pawxml(val_xml)
    c = parse_corewf(core_xml)
    assert p["r"].size == c["r"].size and np.allclose(p["r"], c["r"]), \
        "valence / corewf radial grids differ"
    r = p["r"]
    rab = p["a"] * p["d"] * np.exp(p["d"] * np.arange(p["nr"]))   # dr/di
    u_val = p["phi_ae"] * r
    l_val = [s["l"] for s in p["states"]]
    u_core = c["wfc"] * r
    l_core = [s["l"] for s in c["states"]]
    # core wf normalization sanity: INT u^2 dr = 1 per orbital
    norms = [_cumsimp0(u_core[k] ** 2 * rab)[-1] for k in range(len(l_core))]
    print("core u-norms:", " ".join("%.9f" % n for n in norms))

    X = build_ex_cvij_ln(r, rab, u_val, l_val, u_core, l_core)
    ref = p["exx_X"]
    assert ref is not None, "valence XML lacks exact_exchange_X_matrix"
    scale = np.abs(ref).max()
    err = np.abs(X - ref).max()
    print("max|native - XML| = %.3e  (max|XML| = %.3e, rel %.3e)"
          % (err, scale, err / scale))
    for i in range(len(l_val)):
        print("  diag %d (l=%d): native %+.10f ref %+.10f"
              % (i, l_val[i], X[i, i], ref[i, i]))
    assert err / scale < 1e-5, "native ex_cvij does not reproduce the XML"
    print("PASS ex_cvij native builder vs ABINIT-XML")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
