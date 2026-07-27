#!/usr/bin/env python3
"""Generate atompaw inputs for the Si PAW causal ladder (KKK / PRB 90, 075125 recipe).

Each rung changes exactly ONE ingredient relative to the previous one, so that the
RPA(nbnd) behaviour can be attributed to a specific dataset deficiency:

  D0  sp only, frozen 2s2p, 2 partial waves/l, low-energy 2nd reference   (standard-PAW baseline)
  D1  D0 + d channel                                                      ("missing d" hypothesis)
  D2  D1 + 3rd partial wave per l at +20 Ry                               ("missing high-E projectors")
  D3  D2 + r_c tuned per partial wave to match the AE norm                (KKK norm conservation)
  D4  D3 + f channel                                                      (l completeness)
  D5  D4 + 2s2p promoted to valence                                       (frozen-core; full KKK clone)

Energies in the atompaw input are RYDBERG.  AE PBE scalar-relativistic Si eigenvalues
(this build, loggrid 2001):
    1s -131.2640   2s -10.25310   3s -0.7947293   2p -7.023471   3p -0.2999630
"""

import os

# Si all-electron reference eigenvalues (Ry), PBE scalar-relativistic.
AE_EIG = {(1, 0): -131.2640, (2, 0): -10.25310, (3, 0): -0.7947293,
          (2, 1): -7.023471, (3, 1): -0.2999630}

# atompaw enumerates AE orbitals grouped by l, ascending n.
ORBITAL_ORDER = [(1, 0), (2, 0), (3, 0), (2, 1), (3, 1)]

LMAX_NAME = {0: "s", 1: "p", 2: "d", 3: "f"}


def basis_layout(spec):
    """Ordered list of (l, label, energy_or_None, rc) for every partial wave.

    atompaw builds, for each l in 0..lmax: first the *bound valence* states of that l
    (ascending n), then the user-supplied extra reference energies in the order given.
    The trailing r_c block must follow exactly this order.
    """
    out = []
    for l in range(spec["lmax"] + 1):
        bound = sorted(n for (n, ll) in spec["valence"] if ll == l)
        for n in bound:
            out.append((l, f"{n}{LMAX_NAME[l]}", None, spec["rc"][l].pop(0)))
        for e in spec["extras"].get(l, []):
            out.append((l, f"{LMAX_NAME[l]}@{e:g}Ry", e, spec["rc"][l].pop(0)))
    return out


def write_input(spec, path):
    """Emit one atompaw input file."""
    spec = dict(spec)
    spec["rc"] = {l: list(v) for l, v in spec["rc"].items()}  # basis_layout consumes these
    layout = basis_layout(spec)

    L = []
    L.append("Si 14")
    # LOGDERIVRANGE must precede the grid spec: the parser bounds its argument list by
    # the position of LOGGRID, and the grid reader would otherwise swallow the numbers.
    L.append(f"GGA-PBE SCALARRELATIVISTIC "
             f"LOGDERIVRANGE {spec['ld_min']} {spec['ld_max']} {spec['ld_n']} LOGGRID 2001")
    L.append("3 3 0 0 0")          # max n per l for the AE configuration
    L.append("3 1 2")              # Si ground state: 3p occupancy 2 (not the default full shell)
    L.append("0 0 0")              # end of occupancy edits
    for (n, l) in ORBITAL_ORDER:   # core / valence flags, atompaw ordering
        L.append("v" if (n, l) in spec["valence"] else "c")
    L.append(str(spec["lmax"]))
    L.append(f"{spec['rc_max']:.4f} {spec['rc_shape']:.4f} "
             f"{spec['rc_vloc']:.4f} {spec['rc_core']:.4f}")

    # per-l "add another basis function?" dialogue
    for l in range(spec["lmax"] + 1):
        for e in spec["extras"].get(l, []):
            L.append("y")
            L.append(f"{e:g}")
        L.append("n")

    L.append(f"{spec.get('scheme', 'MODRRKJ')} "
             f"{spec.get('ortho', 'VANDERBILTORTHO')} BESSELSHAPE")
    L.append(f"{spec['vloc_l']} {spec['vloc_e']:g} VPSMATCHNC")
    for (_l, _lab, _e, rc) in layout:
        L.append(f"{rc:.4f}")

    # Emit BOTH ABINIT PAW-XML (-> abinit2coqui) and UPF (-> QE -> pw2coqui) so the two
    # converter paths can be cross-checked against one identical dataset.
    # PRTCOREWF is required for the core-valence exchange (ex_cvij) route.
    L.append("XMLOUT")
    L.append("PRTCOREWF")
    L.append("PWSCFOUT")
    L.append("UPFDX 0.0125d0 UPFXMIN -7.d0 UPFZMESH 14.d0")
    L.append("END")

    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return layout


# --------------------------------------------------------------------------------------
# Ladder definitions.  r_c values are STARTING points; rung D3 onward are re-tuned by
# scan_norm.py so that each pseudo partial wave matches the AE norm inside r_c.
# --------------------------------------------------------------------------------------

def rung(name):
    common = dict(ld_min=-12.0, ld_max=30.0, ld_n=1000,
                  rc_shape=1.60, rc_vloc=1.50, rc_core=1.30)

    # RESONANCE RULE (learned the hard way, see notes/si_gw_paw_dataset_plan.md §6):
    # a reference energy must NOT sit on a log-derivative pole.  At a resonance the AE
    # solution has psi(r_c) -> 0, so normalizing it to unit amplitude at r_c inflates the
    # interior enormously: E_s=2.5 Ry (on Si's s resonance) gave int|phi|^2 = 1053 and a
    # PP_Q entry of 168, versus |q_ij| < 0.08 in the kjpaw library dataset.  QE then drives
    # the density negative and the SCF diverges.  E_s=12 / E_p=10 are off-resonance and
    # give max|q_ij| = 0.240 with a converged QE total energy.
    if name == "D0":   # standard-PAW baseline: sp, frozen core, 2 partial waves per l
        return dict(common, valence=[(3, 0), (3, 1)], lmax=1,
                    extras={0: [12.0], 1: [10.0]},
                    rc={0: [1.90, 1.90], 1: [1.90, 1.90]},
                    rc_max=1.90, vloc_l=2, vloc_e=0.0)

    # Channels with no bound state (d, f in Si) scatter weakly: two reference energies
    # closer than ~10 Ry give partial waves that are nearly proportional inside r_c and
    # atompaw aborts with a non-positive-definite overlap operator.  Each such channel
    # therefore gets ONE mid-energy reference plus the +20 Ry one (added at rung D2).
    if name == "D1":   # + d channel, low energy only ("missing d" hypothesis in isolation)
        return dict(common, valence=[(3, 0), (3, 1)], lmax=2,
                    extras={0: [2.5], 1: [3.0], 2: [1.0]},
                    rc={0: [1.90, 1.90], 1: [1.90, 1.90], 2: [1.90]},
                    rc_max=1.90, vloc_l=3, vloc_e=0.0)

    if name == "D2":   # + a +20 Ry partial wave in every channel (KKK high-E scattering)
        return dict(common, valence=[(3, 0), (3, 1)], lmax=2,
                    extras={0: [2.5, 20.0], 1: [3.0, 20.0], 2: [1.0, 20.0]},
                    rc={0: [1.90] * 3, 1: [1.90] * 3, 2: [1.90] * 2},
                    rc_max=1.90, vloc_l=3, vloc_e=0.0)

    # D3 onward: r_c of each occupied wave is tuned so that q_aa -> 0 (KKK option (i),
    # "set the core radius such that the pseudo partial wave possesses the same norm as
    # the all-electron partial wave").  Roots found by sweep.py: r_c(3s)=1.555 (q=-8e-4),
    # r_c(3p)=1.796 (q=+2e-4); Q_occ falls from -0.1624 to -0.0012 e.
    #
    # Two construction constraints are forced by the smaller radii and are NOT free
    # choices:
    #  * E_s must move 2.5 -> 14 Ry.  The (r_c, E_ref) region where atompaw's overlap
    #    operator stays positive definite is a narrow diagonal band; at r_c(s)=1.56 only
    #    E_s ~ 14 Ry survives (the atompaw manual's own Si example also uses 14 Ry).
    #  * the augmentation radius must equal the LARGEST channel radius, not stay at 1.90.
    #    Leaving r_aug > max(r_c) leaves a shell in which no projector has support, and
    #    the p scattering degrades badly (RMS phase error 0.688 vs 0.053 when matched).
    if name == "D3":   # + norm-matched r_c
        return dict(common, valence=[(3, 0), (3, 1)], lmax=2,
                    extras={0: [14.0, 20.0], 1: [3.0, 20.0], 2: [1.0, 20.0]},
                    rc={0: [1.555] * 3, 1: [1.796] * 3, 2: [1.796] * 2},
                    rc_max=1.796, rc_shape=1.55, rc_vloc=1.50, rc_core=1.30,
                    vloc_l=3, vloc_e=0.0)

    if name == "D4":   # + f channel
        return dict(common, valence=[(3, 0), (3, 1)], lmax=3,
                    extras={0: [14.0, 20.0], 1: [3.0, 20.0],
                            2: [1.0, 20.0], 3: [1.0, 20.0]},
                    rc={0: [1.555] * 3, 1: [1.796] * 3,
                        2: [1.796] * 2, 3: [1.796] * 2},
                    rc_max=1.796, rc_shape=1.55, rc_vloc=1.50, rc_core=1.30,
                    vloc_l=4, vloc_e=0.0)

    # D5: full KKK clone.  The shallow core in valence is what KKK actually did for Si
    # ("we have included the 2p electrons in the valence", PRB 90, 075125 p. 11) and it
    # also makes the basis naturally well conditioned: each of l=0,1 then carries two
    # genuinely BOUND, widely separated references (2s -10.25 / 3s -0.79 Ry) instead of
    # one bound plus an artificial unbound one.  The semicore waves need their own much
    # smaller matching radius -- at r_c=1.96 the pseudo 2p retains only 4% of the AE
    # norm, which is what produced Q_occ = +7.7 e in the first attempt.
    if name == "D5":
        return dict(common, valence=[(2, 0), (3, 0), (2, 1), (3, 1)], lmax=3,
                    extras={0: [20.0], 1: [20.0], 2: [1.0, 20.0], 3: [1.0, 20.0]},
                    rc={0: [1.10, 1.555, 1.555], 1: [1.10, 1.796, 1.796],
                        2: [1.796] * 2, 3: [1.796] * 2},
                    rc_max=1.796, rc_shape=1.55, rc_vloc=1.50, rc_core=1.10,
                    vloc_l=4, vloc_e=0.0)

    raise KeyError(name)


# --------------------------------------------------------------------------------------
# THE NORM SERIES (N0..N9) -- the production datasets.
#
# The six-rung D-ladder answered the attribution question (adding d channels and +20 Ry
# projectors leaves Q_occ bit-identical), but only D0 survived contact with QE.  What is
# actually needed to test the KKK mechanism is a family of datasets that differ ONLY in
# how badly the occupied partial waves violate the norm, all of them well conditioned and
# QE-validated.  Shrinking r_c walks q_aa through zero, so the series below spans a 137x
# range in Q_occ while keeping every dataset usable in a solid.
#
# E_s must be re-tuned at each radius: it has to stay off the s resonance (else q_ij
# explodes and QE's density goes negative) AND inside atompaw's narrow positive-definite
# band, and both move with r_c.  E_s=14 at r_c=1.555 is near-resonant (max|q|=3.19);
# E_s=18 is clean (0.477); E_s=20 breaks QE again (-4666 Ry).  All entries below were
# verified end-to-end: atompaw -> UPF -> QE SCF, PBE, a=10.26 bohr, 4x4x4 k, 60 Ry.
#
#   name  rc_s   rc_p   E_s   Q_occ(e)    reduction   E_QE(Ry)      negative rho
NORM_SERIES = [
    ("N0", 1.900, 1.900, 12.0, -0.162399, "1.0x",  -93.441373),
    ("N1", 1.850, 1.900, 12.0, -0.133559, "1.2x",  -93.474136),
    ("N2", 1.800, 1.850, 12.0, -0.107830, "1.5x",  -93.457336),
    ("N3", 1.750, 1.850, 12.0, -0.082139, "2.0x",  -93.507778),
    ("N4", 1.700, 1.800, 16.0, -0.055340, "2.9x",  -93.524997),
    ("N5", 1.650, 1.800, 16.0, -0.034071, "4.8x",  -93.563716),
    ("N6", 1.600, 1.796, 16.0, -0.014425, "11.3x", -93.590701),
    ("N7", 1.580, 1.796, 18.0, -0.007600, "21.4x", -93.707718),
    ("N8", 1.570, 1.780, 18.0, -0.003514, "46.2x", -93.663905),
    ("N9", 1.555, 1.796, 18.0, -0.001184, "137x",  -93.729401),
]


def norm_series_spec(rc_s, rc_p, e_s, e_p=10.0):
    """One member of the norm series: D0's 2-waves-per-l structure at tuned radii."""
    common = dict(ld_min=-12.0, ld_max=30.0, ld_n=1000)
    m = max(rc_s, rc_p)
    return dict(common, valence=[(3, 0), (3, 1)], lmax=1,
                extras={0: [e_s], 1: [e_p]},
                rc={0: [rc_s] * 2, 1: [rc_p] * 2},
                rc_max=m, rc_shape=min(1.60, m * 0.86),
                rc_vloc=min(1.50, m * 0.84), rc_core=1.30,
                vloc_l=2, vloc_e=0.0)


if __name__ == "__main__":
    import sys
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    root = positional[0] if positional else os.path.expanduser("~/Projects/PAW_GW/si_gw_paw")
    if "--series" in sys.argv:
        for (name, rc_s, rc_p, e_s, q, ratio, _e) in NORM_SERIES:
            d = os.path.join(root, name)
            os.makedirs(d, exist_ok=True)
            write_input(norm_series_spec(rc_s, rc_p, e_s), os.path.join(d, "Si.input"))
            print(f"{name}: rc_s={rc_s:.3f} rc_p={rc_p:.3f} E_s={e_s:g}  "
                  f"Q_occ={q:+.6f} ({ratio})")
        raise SystemExit(0)
    for name in ["D0", "D1", "D2", "D3", "D4", "D5"]:
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        layout = write_input(rung(name), os.path.join(d, "Si.input"))
        print(f"{name}: {len(layout)} partial waves  "
              + ", ".join(f"{lab}(rc={rc:.2f})" for (_l, lab, _e, rc) in layout))
