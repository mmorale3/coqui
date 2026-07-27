#!/usr/bin/env python3
"""Score an atompaw-generated PAW dataset against the two failure modes that make
RPA/GW energies diverge with the number of bands (Klimes/Kaltak/Kresse, PRB 90, 075125).

Metric 1 -- PARTIAL-WAVE NORM VIOLATION
    q_a = int_0^rc (phi_a^2 - phit_a^2) r^2 dr
  KKK Eq. (12)/(24)-(27): for a plane-wave-like high-energy state, <a|p_a> -> 0, the
  augmentation terms drop out, and the overlap density collapses to the PSEUDO density.
  The RPA correlation energy then still decays like 1/N_pw but to a WRONG LIMIT set by
  the norm defect of rho-tilde.  The occupancy-weighted sum
    Q_occ = sum_a f_a q_a
  is the leading (l=0) error in that limit, and is the number to drive to zero.

Metric 2 -- HIGH-ENERGY SCATTERING FIDELITY
  Max/RMS deviation between the AE and PAW arctan-unwrapped log-derivative phases over
  a chosen energy window.  KKK place a third partial wave ~20 Ry above vacuum to keep
  this small out to ~30 Ry; beyond that only norm conservation helps.

Usage:  paw_diag.py <dataset_dir> [<dataset_dir> ...]
"""

import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

RY = 0.5  # 1 Ry in Hartree; PAW-XML energies are Hartree, atompaw logderiv files are Ry
LNAME = {0: "s", 1: "p", 2: "d", 3: "f"}


def _floats(text):
    return np.array([float(x) for x in text.split()])


def read_pawxml(path):
    """Return (r, a, d, states) where states carry AE/PS partial waves on the log grid."""
    root = ET.parse(path).getroot()

    grid_el = root.find("radial_grid")
    a = float(grid_el.get("a"))
    d = float(grid_el.get("d"))
    r = _floats(grid_el.find("values").text)

    states = {}
    order = []
    for st in root.find("valence_states"):
        sid = st.get("id").strip()
        order.append(sid)
        states[sid] = dict(
            l=int(st.get("l")),
            n=(int(st.get("n")) if st.get("n") else None),
            f=(float(st.get("f")) if st.get("f") else 0.0),
            e_ha=float(st.get("e")),
            rc=float(st.get("rc")),
        )

    for tag, key in (("ae_partial_wave", "phi"), ("pseudo_partial_wave", "phit")):
        for el in root.findall(tag):
            states[el.get("state").strip()][key] = _floats(el.text)

    return r, a, d, [states[s] for s in order]


def log_integrate(f, r, a, d, i_max):
    """int_0^r(i_max) f(r) dr on the atompaw log grid r = a*(exp(d*i)-1).

    dr = d*(r+a) di, and the index grid is uniform, so Simpson in index space is exact
    to the same order as it would be on a uniform radial mesh.
    """
    g = f[: i_max + 1] * d * (r[: i_max + 1] + a)
    n = len(g)
    if n < 3:
        return float(np.trapezoid(g))
    # Simpson needs an odd number of points; handle the tail with trapezoid.
    if n % 2 == 0:
        head = g[: n - 1]
        tail = 0.5 * (g[n - 2] + g[n - 1])
    else:
        head = g
        tail = 0.0
    m = len(head)
    s = head[0] + head[-1] + 4.0 * head[1:m - 1:2].sum() + 2.0 * head[2:m - 2:2].sum()
    return float(s / 3.0 + tail)


def norm_violations(path):
    r, a, d, states = read_pawxml(path)
    rows = []
    for st in states:
        i_rc = int(np.searchsorted(r, st["rc"]))
        w = r ** 2
        n_ae = log_integrate(st["phi"] ** 2 * w, r, a, d, i_rc)
        n_ps = log_integrate(st["phit"] ** 2 * w, r, a, d, i_rc)
        rows.append(dict(
            label=(f"{st['n']}{LNAME[st['l']]}" if st["n"]
                   else f"{LNAME[st['l']]}@{st['e_ha'] / RY:.3g}Ry"),
            l=st["l"], f=st["f"], e_ry=st["e_ha"] / RY, rc=st["rc"],
            n_ae=n_ae, n_ps=n_ps, q=n_ae - n_ps,
            rel=(n_ae - n_ps) / n_ae if n_ae else float("nan"),
        ))
    return rows


def qij_matrix(dirpath):
    """PP_Q (nbeta x nbeta augmentation charges) from the UPF in `dirpath`, or None.

    max|q_ij| is the practical conditioning gate: an inflated partial wave (see
    `ae_poles` below) shows up as a huge diagonal entry, and QE's density then goes
    negative.  Reference scale: the library kjpaw dataset has all |q_ij| < 0.08.

    It is a strong but NOT sufficient predictor, so report it together with the PS/AE
    interior-norm ratio: 0.48 (N9) and 0.53 (P1) are healthy, 3.47 (P4) still converges
    with a trace of negative rho, but 0.79 at an inflation ratio of 2.0 (the 2-wave
    VANDERBILT dataset) diverges to -759 Ry.  The ratio is what separates that last case.
    """
    upfs = [f for f in os.listdir(dirpath) if f.endswith(".UPF")]
    if not upfs:
        return None
    txt = open(os.path.join(dirpath, upfs[0]), errors="replace").read()
    m = re.search(r"<PP_Q type[^>]*>(.*?)</PP_Q>", txt, re.S)
    if not m:
        return None
    v = _floats(m.group(1))
    n = int(round(len(v) ** 0.5))
    return v.reshape(n, n)


def ae_poles(dirpath, emax=None):
    """{l: [E_Ry, ...]} — energies where the AE log-derivative at r_c diverges.

    A pole is `psi(r_c) = 0`.  atompaw normalizes each partial wave to unit amplitude
    at r_c, so a reference energy near a pole inflates the interior of the wave without
    bound (D3's p@20Ry has AE norm 52.7 with the l=1 pole at 19.3 Ry).  Zeros of the
    log-derivative -- `psi'(r_c) = 0` -- are harmless and are excluded: only the
    -inf -> +inf crossings count, which is what the |L| > 2 guard on both sides selects.
    """
    out = {}
    for fn in sorted(os.listdir(dirpath)):
        m = re.fullmatch(r"logderiv\.(\d+)", fn)
        if not m:
            continue
        dat = np.loadtxt(os.path.join(dirpath, fn))
        e, L = dat[:, 0], dat[:, 1]
        p = [0.5 * (e[i] + e[i + 1]) for i in range(len(e) - 1)
             if L[i] < -2.0 and L[i + 1] > 2.0
             and (emax is None or e[i] <= emax)]
        out[int(m.group(1))] = p
    return out


def safe_windows(poles, emin=0.0, emax=30.0):
    """Intervals between consecutive poles, with their midpoints — the off-resonance
    reference energies available in [emin, emax] for one channel."""
    edges = [emin] + [p for p in poles if emin < p < emax] + [emax]
    return [(lo, hi, 0.5 * (lo + hi)) for lo, hi in zip(edges[:-1], edges[1:])]


def logderiv_error(dirpath, emin=0.0, emax=30.0):
    """Max/RMS scattering-phase mismatch per l over [emin, emax] Ry.

    Uses the RAW log-derivatives L = r*psi'/psi at r_c (columns 2,3) rather than
    atompaw's pre-unwrapped phase columns, whose branch counters jump spuriously (the
    AE column can shift by pi where the two log-derivatives in fact agree to 1e-2).
    Two log-derivatives describe the same scattering state iff arctan(L) agree modulo
    pi, so the well-defined mismatch is |wrap_pi(arctan(L_ae) - arctan(L_ps))|.
    """
    out = {}
    for fn in sorted(os.listdir(dirpath)):
        m = re.fullmatch(r"logderiv\.(\d+)", fn)
        if not m:
            continue
        dat = np.loadtxt(os.path.join(dirpath, fn))
        e, ae, ps = dat[:, 0], np.arctan(dat[:, 1]), np.arctan(dat[:, 2])
        sel = (e >= emin) & (e <= emax)
        if not sel.any():
            continue
        diff = ae[sel] - ps[sel]
        dev = np.abs((diff + np.pi / 2) % np.pi - np.pi / 2)
        out[int(m.group(1))] = (float(dev.max()), float(np.sqrt((dev ** 2).mean())))
    return out


def report(dirpath):
    xmls = [f for f in os.listdir(dirpath) if f.endswith(".xml") and "corewf" not in f]
    if not xmls:
        print(f"{os.path.basename(dirpath)}: no PAW-XML found (generation failed?)")
        return None
    rows = norm_violations(os.path.join(dirpath, xmls[0]))

    name = os.path.basename(dirpath.rstrip("/"))
    print(f"\n=== {name} ===")
    print(f"{'wave':>10} {'l':>2} {'occ':>5} {'E(Ry)':>9} {'rc':>6} "
          f"{'|phi|^2':>10} {'|phit|^2':>10} {'q_aa':>10} {'rel':>8}")
    for w in rows:
        print(f"{w['label']:>10} {w['l']:>2} {w['f']:>5.1f} {w['e_ry']:>9.3f} "
              f"{w['rc']:>6.3f} {w['n_ae']:>10.5f} {w['n_ps']:>10.5f} "
              f"{w['q']:>10.5f} {w['rel']:>7.1%}")

    q_occ = sum(w["f"] * w["q"] for w in rows)
    q_val = sum(abs(w["q"]) for w in rows if w["f"] > 0)
    print(f"  Q_occ = sum_a f_a q_a = {q_occ:+.6f} e   "
          f"(occupancy-weighted norm defect -> wrong RPA limit)")
    print(f"  sum |q_aa| over occupied waves = {q_val:.6f}")

    q = qij_matrix(dirpath)
    if q is not None:
        infl = max(w["n_ps"] / w["n_ae"] for w in rows)
        print(f"  max |q_ij| = {np.abs(q).max():.4f}   max PS/AE interior norm = {infl:.2f}"
              f"   (kjpaw library reference: |q_ij| < 0.08)")

    pol = ae_poles(dirpath)
    if pol:
        print("  AE log-derivative poles / off-resonance reference energies (Ry):")
        for l in sorted(pol):
            wins = safe_windows(pol[l])
            print(f"    l={l} ({LNAME.get(l, l)}): poles "
                  + " ".join(f"{p:.2f}" for p in pol[l])
                  + "   | safe mid-points " + " ".join(f"{m:.1f}" for (_lo, _hi, m) in wins))

    # Energy-resolved scattering fidelity: the KKK failure is a *high-energy* one, so a
    # single [0,30] Ry number hides exactly the trend we are trying to expose.
    bins = [(-12.0, 0.0), (0.0, 5.0), (5.0, 15.0), (15.0, 30.0)]
    binned = [logderiv_error(dirpath, lo, hi) for (lo, hi) in bins]
    if binned[0]:
        print("  logderiv RMS phase error (rad) by window:  "
              + "  ".join(f"[{lo:g},{hi:g}]" for (lo, hi) in bins))
        for l in sorted(binned[0]):
            cells = "  ".join(f"{b.get(l, (float('nan'),) * 2)[1]:>8.4f}" for b in binned)
            print(f"    l={l} ({LNAME.get(l, l)}):  {cells}")

    return dict(name=name, rows=rows, q_occ=q_occ, binned=binned)


if __name__ == "__main__":
    targets = sys.argv[1:] or ["."]
    for t in targets:
        report(t)
