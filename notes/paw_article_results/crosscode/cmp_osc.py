#!/usr/bin/env python3
"""Compare CoQui's PAW oscillator rho_vc(q+G) against ABINIT's, element by element.

This is the first EXTERNAL check of CoQui's augmented oscillators. Every prior
test in this campaign compared CoQui against a reference that shared its own
PAW machinery (Pskna, build_eta_on_rho_g_at_q_chunk, the lambda split), so a
wrong Q_ij(q+G) or projector contraction moved both sides identically.

Method, and why it is not circular:

  The SMOOTH oscillator is computed by both codes from the same WFK with no
  PAW involvement at all -- ABINIT's rho_tw_g, CoQui's FFT of conj(u_v) u_c.
  It is therefore a genuine common reference, and any difference between the
  codes in it can only be convention (overall normalization, a global phase,
  or a conjugation). So: fit ONE complex scale alpha on the smooth part,
  check the residual is small (that validates the fit), then apply the SAME
  alpha to the AE part. If smooth matches and AE does not, the augmentation
  differs -- and that is a physics result, not a convention artifact.

Inputs are matched by MILLER INDEX, never by position: the two codes have no
reason to order their G-spheres the same way, and ABINIT's sphere (ecuteps)
is a small subset of CoQui's rho_g.
"""
import sys
import numpy as np


def read_abinit(path):
    """OSC header then per-G: gx gy gz Re(sm) Im(sm) Re(ae) Im(ae)."""
    d = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("OSC"):
                continue
            t = line.split()
            if len(t) < 7:
                continue
            g = (int(t[0]), int(t[1]), int(t[2]))
            d[g] = (complex(float(t[3]), float(t[4])),
                    complex(float(t[5]), float(t[6])))
    return d


def read_coqui(path):
    """gx gy gz Re(sm) Im(sm) Re(ae) Im(ae)."""
    d = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            t = line.split()
            if len(t) < 7:
                continue
            g = (int(t[0]), int(t[1]), int(t[2]))
            d[g] = (complex(float(t[3]), float(t[4])),
                    complex(float(t[5]), float(t[6])))
    return d


def fit_scale(a, b):
    """Least-squares complex alpha minimising |b - alpha*a|."""
    num = np.vdot(a, b)          # sum conj(a)*b
    den = np.vdot(a, a).real
    return num / den if den > 0 else 0.0


def report(tag, a, b, alpha):
    resid = b - alpha * a
    denom = np.linalg.norm(b)
    print(f"  {tag:9s}  ||resid|| / ||CoQui|| = {np.linalg.norm(resid)/denom:.6e}"
          f"   max|resid| = {np.abs(resid).max():.6e}"
          f"   max|CoQui| = {np.abs(b).max():.6e}")


def main():
    abi = read_abinit(sys.argv[1])
    cq = read_coqui(sys.argv[2])
    common = sorted(set(abi) & set(cq))
    print(f"ABINIT G: {len(abi)}   CoQui G: {len(cq)}   matched by Miller index: {len(common)}")
    if not common:
        print("NO OVERLAP -- Miller conventions differ; nothing can be concluded.")
        return

    a_sm = np.array([abi[g][0] for g in common])
    a_ae = np.array([abi[g][1] for g in common])
    c_sm = np.array([cq[g][0] for g in common])
    c_ae = np.array([cq[g][1] for g in common])

    # Try both conjugations; the codes may differ by rho <-> conj(rho).
    for name, aa_sm, aa_ae in (("as-is", a_sm, a_ae),
                               ("conj", np.conj(a_sm), np.conj(a_ae))):
        alpha = fit_scale(aa_sm, c_sm)
        r = np.linalg.norm(c_sm - alpha*aa_sm) / max(np.linalg.norm(c_sm), 1e-300)
        print(f"\n[{name}] alpha fitted on SMOOTH = {alpha:.8g}  |alpha| = {abs(alpha):.8g}"
              f"   smooth residual = {r:.6e}")
        if r > 0.1:
            print("        smooth does not match under this conjugation; skipping AE")
            continue
        print("        -> smooth agrees, so alpha is a genuine convention factor.")
        report("SMOOTH", aa_sm, c_sm, alpha)
        report("AE", aa_ae, c_ae, alpha)
        # The augmentation alone, which is what is actually in question.
        report("AUG only", aa_ae - aa_sm, c_ae - c_sm, alpha)
        print(f"        max|AUG| ABINIT*alpha = {np.abs(alpha*(aa_ae-aa_sm)).max():.6e}"
              f"   CoQui = {np.abs(c_ae-c_sm).max():.6e}")


if __name__ == "__main__":
    main()
