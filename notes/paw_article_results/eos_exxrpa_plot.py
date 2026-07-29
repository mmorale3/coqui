#!/usr/bin/env python3
"""
Si EXX+RPA equation of state: CoQui vs ABINIT, before and after the V_LL fix.

Two figures:
  eos_exxrpa_total.pdf  -- total energy vs lattice constant
  eos_exxrpa_ec.pdf     -- RPA correlation energy alone

On the ABINIT total-energy curve. ABINIT's ACFD-RPA run reports E_c only; its
`etotal` is the PBE total, a DIFFERENT quantity from CoQui's RPA@PBE total, and
plotting the two on one axis would be a category error. Constructing a genuine
ABINIT RPA@PBE total would need E_x^EXX on PBE orbitals from ABINIT, which is a
separate calculation and carries the known Arnaud-plane-wave vs pawdijfock trap
(ABINIT's GW Sigma_x and hybrid Fock are different operators).

What IS legitimate: both codes consume the BIT-IDENTICAL 500-band WFK, so E_1e
and E_HF are common to them and E_c is the ONLY difference. So the curve

    E_total[ABINIT E_c] = (CoQui total - CoQui E_c) + ABINIT E_c

is CoQui's own one-body + exchange with ABINIT's correlation swapped in. It
answers exactly the question the figure is for -- what the EOS would be if the
RPA correlation came from ABINIT -- and it is labelled as such rather than
passed off as an independent ABINIT total.
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HA2MHA = 1e3

EWALD_HA = {
    10.05: -8.57599688619593, 10.15: -8.49150430603639,
    10.25: -8.40866036158725, 10.35: -8.32741726630622,
    10.45: -8.24772906280087, 10.55: -8.16955153613926,
}

# ABINIT ACFD-RPA (gwrpacorr 1, gwcalctyp 1, gw_icutcoul 7), n=500, same WFK.
AB_EC = {
    10.05: -0.43112, 10.15: -0.43012, 10.25: -0.42926,
    10.35: -0.42855, 10.45: -0.42794, 10.55: -0.42755,
}

# CoQui BEFORE the V_LL fix (eos_conv500_coqui: thresh 1e-4, paw_isdf_tol 5e-5).
# Kept at its own settings deliberately -- this is the published-state curve.
PRE = {
    10.05: (0.2029784671080026,  -0.618628733605779),
    10.15: (0.11802472123308883, -0.6146142980223505),
    10.25: (0.03508031825637192, -0.6111665055015347),
    10.35: (-0.04592027028490664,-0.6082237815569294),
    10.45: (-0.12501515724127477,-0.6057050050588075),
    10.55: (-0.20227050928685641,-0.6035790559552046),
}


ANCHOR = 10.35   # largest lattice constant present in every series


def rel(a_list, E, anchor=ANCHOR):
    """Shift a curve to a COMMON lattice constant, in mHa.

    Deliberately not 'shift each curve to its own minimum': the post-fix series
    has no minimum inside the sampled range, so a per-curve minimum would anchor
    it at an endpoint and invent a visual comparison between an interior
    minimum and an edge. A common anchor compares shapes without that artifact.
    """
    E = np.asarray(E, float)
    i = list(a_list).index(anchor)
    return (E - E[i]) * HA2MHA


def main(path="eos_exxrpa.json"):
    post = {float(k): (v["total"], v["ec"]) for k, v in json.load(open(path)).items()}
    a_post = sorted(post)
    a_pre = sorted(PRE)

    Epost = np.array([post[a][0] + EWALD_HA[a] for a in a_post])
    ecpost = np.array([post[a][1] for a in a_post])
    # CoQui one-body+exchange with ABINIT's correlation substituted.
    Esub = np.array([post[a][0] - post[a][1] + AB_EC[a] + EWALD_HA[a] for a in a_post])
    Epre = np.array([PRE[a][0] + EWALD_HA[a] for a in a_pre])
    ecpre = np.array([PRE[a][1] for a in a_pre])
    ecab = np.array([AB_EC[a] for a in a_post])

    bracketed = not (np.all(np.diff(Epost) < 0) or np.all(np.diff(Epost) > 0))

    # ---------------- Figure 1: total energy ----------------
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    # Post-fix and the ABINIT-E_c substitution differ only by a ~flat 6.2 mHa,
    # so they overlay. Draw the substitution as a wide pale band UNDER the
    # post-fix line so the coincidence is the visible message rather than one
    # curve silently hiding the other.
    ax.plot(a_post, rel(a_post, Esub), "-", color="#55a868", lw=6.5, alpha=0.35,
            solid_capstyle="round",
            label="CoQui $E_{1e}+E_{HF}$ + ABINIT $E_c$ (same WFK)")
    ax.plot(a_post, rel(a_post, Epost), "o-", color="#4c72b0", lw=2.0, ms=6,
            label="CoQui, post-fix  (thresh 1e-5, tol 1e-8)")
    ax.plot(a_pre, rel(a_pre, Epre), "s--", color="#c44e52", lw=1.6, ms=5,
            mfc="white",
            label="CoQui, pre-fix  (thresh 1e-4, tol 5e-5) — CONFOUNDED")
    ax.axvline(ANCHOR, color="#bbb", lw=0.8, ls=":", zorder=0)
    ax.set_xlabel("lattice constant $a$ (Bohr)")
    ax.set_ylabel(rf"$E_{{\rm total}}(a)-E_{{\rm total}}({ANCHOR})$ (mHa)")
    ax.set_title("Si RPA@PBE equation of state, $n_{\\rm band}=500$")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=8.0, loc="upper right")
    # Caveats live in the footer, not over the data.
    warn = []
    if not bracketed:
        warn.append("post-fix minimum NOT bracketed — $E(a)$ still descending at "
                    f"$a={max(a_post)}$; no $a_0$/$B_0$ is quoted from this.")
    warn.append("pre-fix curve is at DIFFERENT settings: its $E_{1e}+E_{HF}$ alone "
                "differs from post-fix by 6.15 mHa across\n0.30 Bohr (monotone), so "
                "pre-vs-post here is NOT a clean measure of the V$_{LL}$ fix — "
                "use Fig. 2 ($E_c$).")
    warn.append("The pre-fix curve is nearly FLAT because its over-steep $E_c$ "
                "(4$\\times$ ABINIT's volume dependence) happened to\ncancel the "
                "$E_{1e}+E_{HF}$ slope. Its $a_0\\approx$10.23 was error "
                "cancellation, not agreement.")
    fig.text(0.012, 0.115, "\n".join(warn), fontsize=7.2, color="#8c1a11",
             va="bottom", ha="left")
    fig.text(0.012, 0.012,
             "Total = CoQui($E_{1e}+E_{HF}+E_{RPA}$) + $E_{Ewald}$(ABINIT, Ha). "
             "ABINIT reports no RPA total in this convention; the green band\n"
             "substitutes its $E_c$ into CoQui's decomposition on the identical "
             f"WFK. All curves anchored at $a={ANCHOR}$ Bohr (dotted line).",
             fontsize=6.2, color="#555")
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    fig.savefig("eos_exxrpa_total.pdf")
    fig.savefig("eos_exxrpa_total.png", dpi=180)

    # ---------------- Figure 2: RPA correlation only ----------------
    # Explicit margins: symlog axes are not tight_layout-compatible, and the
    # earlier automatic pass clipped the y-label and collided the x-label with
    # the footer.
    fig2, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.0, 6.4), sharex=True,
        gridspec_kw=dict(height_ratios=[2.0, 1.15], hspace=0.10))
    fig2.subplots_adjust(left=0.135, right=0.975, top=0.935, bottom=0.325)
    ax1.plot(a_pre, ecpre, "o--", color="#c44e52", lw=1.6, ms=5,
             label="CoQui, pre-fix")
    ax1.plot(a_post, ecpost, "o-", color="#4c72b0", lw=2.0, ms=6,
             label="CoQui, post-fix")
    ax1.plot(a_post, ecab, "s-", color="#55a868", lw=1.8, ms=5,
             label="ABINIT ACFD-RPA")
    ax1.set_ylabel(r"$E_c^{\rm RPA}$ (Ha)")
    ax1.set_title("Si RPA correlation energy, $n_{\\rm band}=500$, identical WFK")
    ax1.grid(alpha=0.25, lw=0.6)
    ax1.legend(frameon=False, fontsize=8.5)

    ax2.axhline(0.0, color="#999", lw=0.8)
    ax2.plot(a_post, (ecpost - ecab) * HA2MHA, "o-", color="#4c72b0", lw=1.8, ms=5,
             label="post-fix $-$ ABINIT")
    ax2.plot(a_pre, (ecpre - np.array([AB_EC[a] for a in a_pre])) * HA2MHA,
             "o--", color="#c44e52", lw=1.4, ms=4, label="pre-fix $-$ ABINIT")
    ax2.set_xlabel("lattice constant $a$ (Bohr)")
    ax2.set_ylabel(r"$\Delta E_c$ (mHa)")
    ax2.set_yscale("symlog", linthresh=10.0)
    ax2.grid(alpha=0.25, lw=0.6)
    ax2.legend(frameon=False, fontsize=8, loc="center right")
    sp = (ecpost - ecab).max() - (ecpost - ecab).min()
    spp = ((ecpre - np.array([AB_EC[a] for a in a_pre])).max()
           - (ecpre - np.array([AB_EC[a] for a in a_pre])).min())
    fig2.text(0.135, 0.225,
              f"post-fix offset varies {sp*HA2MHA:.2f} mHa across the sampled "
              f"range; pre-fix varies {spp*HA2MHA:.1f} mHa.\n"
              "A CONSTANT offset cancels from the EOS; a volume-dependent one "
              "does not — that is why the\npre-fix curvature ($B_0$ = 45 GPa vs "
              "98–101 reference) was destroyed while $a_0$ looked fine.",
              fontsize=7.6, color="#31527d", va="top", ha="left")
    fig2.text(0.012, 0.012,
              "Lower panel on a symlog axis (linear within $\\pm$10 mHa). Both "
              "codes consume the bit-identical 500-band WFK, so $E_c$ is the "
              "only difference.\nCaveat: the pre-fix series also used looser "
              "thresh/tol, so it is not a pure V$_{LL}$ contrast — but the "
              "compression defect it carries is\nitself volume-dependent, which "
              "is the same failure mode.", fontsize=6.4, color="#555")
    fig2.savefig("eos_exxrpa_ec.pdf")
    fig2.savefig("eos_exxrpa_ec.png", dpi=180)

    print(f"volumes plotted (post-fix): {a_post}")
    print(f"minimum bracketed: {bracketed}")
    print(f"post-fix E_c offset vs ABINIT: "
          f"{(ecpost-ecab).min()*HA2MHA:.2f} .. {(ecpost-ecab).max()*HA2MHA:.2f} mHa")
    print("wrote eos_exxrpa_total.{pdf,png}, eos_exxrpa_ec.{pdf,png}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eos_exxrpa.json")
