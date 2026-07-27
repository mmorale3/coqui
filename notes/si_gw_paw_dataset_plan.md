# Si PAW datasets for RPA equations of state (KKK construction)

Goal: a Si PAW dataset whose RPA correlation energy converges to the *right* limit as
the band count grows, so that an RPA EOS is usable. Every previous attempt degraded at
large `nbnd`; the standing hypothesis was "missing d projectors", which the cluster
campaign already showed to be insufficient (`jth_with_d` regen: cv parity 31 uHa but EOS
still unusable at kp4/n500).

## 1. Why this fails — Klimes, Kaltak & Kresse, PRB 90, 075125 (2014)

Their §II B/D derives exactly this symptom. In PAW,

    |i> = |i~> + sum_a (|phi_a> - |phi~_a>) <p_a|i~>

For a high-kinetic-energy state `a` (essentially a plane wave), `<a|p_a> -> 0` because
the projectors have finite Fourier support. The completeness relation is violated, the
augmentation terms drop out of the overlap density, and `<i|G|a><a|-G'|i>` collapses to
the **pseudo** density `rho~(G-G')`. Their Eq. (14): the correlation energy still decays
as `1/N_pw`, but **to a wrong limit** fixed by the norm defect of `rho~`. Eq. (11) makes
the coefficient explicit, `dE ~ (2 m e^4 / 9 pi^2 hbar^2) N_el^2 / N_pw` — density- and
therefore volume-dependent, so the error does *not* cancel in an EOS.

Their fix is **norm-conserving partial waves**, not more channels. Explicitly (p. 9):

> "Restoring just the norm of Q_ab(r) was found to be insufficient for GW calculations."
> "With norm-conserving partial waves the results become even robust and stable for very
> large basis sets."

Secondary ingredient: 3 partial waves per l — one at the uppermost core state, one at the
valence binding energy, one **~20 Ry above vacuum** — which "guarantees excellent high
energy scattering properties up to about 30 Ry". Beyond 30 Ry only norm conservation helps.

How they enforce norm conservation (p. 6, option (i)): *not* a norm-conserving pseudization
scheme, but tuning r_c per partial wave until the pseudo partial wave has the same norm as
the AE one inside the sphere. "Generally small core radii (below 1.6 a.u.) indicate that
option (i) was chosen."

Their Si (Table I): `r_s=1.70 r_p=1.95 r_d=1.70 r_f=2.00`, local 1.4, `E_cut=609 eV`, and
p. 11 confirms **2p is in the valence** ("we have included the 2p electrons in the valence").

## 2. Tooling

`atompaw` (`~/Projects/PAW_GW/build/atompaw-install/bin/atompaw`) — the only generator here
with arbitrary reference energies, per-partial-wave r_c, l up to 3, and *both* PAW-XML
(-> `abinit2coqui`) and UPF (-> QE -> `pw2coqui`) from one run, so both converter paths can
be cross-checked on an identical dataset. `PRTCOREWF` emits the core wavefunctions needed
for `ex_cvij`. `ld1.x` is also built as a fallback. Input energies are **Rydberg**;
PAW-XML energies are **Hartree**.

AE PBE scalar-relativistic Si (this build): 1s -131.264, 2s -10.2531, 2p -7.02347,
3s -0.794729, 3p -0.299963 Ry.

## 3. The causal ladder

One ingredient per rung, so the RPA behaviour can be attributed rather than assumed.
Scripts in `notes/paw_article_results/si_gw_paw/`; runs in `~/Projects/PAW_GW/si_gw_paw/`.

| rung | ingredient added | status |
|------|------------------|--------|
| D0 | sp only, frozen 2s2p, 2 waves/l, low-E 2nd reference | built |
| D1 | + d channel | built |
| D2 | + a +20 Ry wave in every channel | built |
| D3 | + r_c tuned so occupied waves match the AE norm | built |
| D4 | + f channel | **blocked** (see §5) |
| D5 | + 2s2p in valence (full KKK clone) | **blocked** (see §5) |

Metrics (`paw_diag.py`): `Q_occ = sum_a f_a q_a` with
`q_a = int_0^rc (phi_a^2 - phi~_a^2) r^2 dr` — the occupancy-weighted norm defect that sets
the wrong RPA limit; and the AE-vs-PAW scattering phase mismatch
`|wrap_pi(arctan L_AE - arctan L_PAW)|`, binned in energy.

## 4. Result so far — the d channel and the high-energy projectors do NOT touch the norm defect

```
rung    Q_occ (e)     s[15,30]  p[15,30]  d[15,30]     (RMS phase error, rad)
D0      -0.162399      0.823     0.774     0.370
D1      -0.162399      0.782     0.719     0.888     <- +d
D2      -0.162399      0.760     0.187     0.104     <- +20 Ry waves
D3      -0.001184      0.269     0.687     0.145     <- norm-matched r_c
```

* `Q_occ` is **bit-identical** across D0/D1/D2. Adding the d channel and adding the 20 Ry
  partial waves leave the occupied-state norm defect completely unchanged — they cannot,
  even in principle, fix the KKK error. This is the quantitative form of the intuition that
  the missing d is contributing but not exclusive, and it is falsifiable independently of
  any solid-state calculation.
* What those ingredients *do* fix is high-energy scattering: D2 improves `p[15,30]`
  0.774 -> 0.187 and `d[15,30]` 0.370 -> 0.104. Both effects are real but distinct.
* Norm matching (D3) reduces `Q_occ` by **137x**, to -0.0012 e.

Structural fact used throughout: `q_a` depends only on its own partial wave and that
wave's r_c — verified by its invariance to 6 digits under every reference-energy change.
So r_c is the only knob for the norm defect, exactly as KKK describe. Roots:
r_c(3s)=1.555 (q=-8e-4), r_c(3p)=1.796 (q=+2e-4); both below/near 1.6 a.u., consistent
with KKK's "option (i)" remark.

Two constraints discovered while tuning, both forced rather than chosen:
* The (r_c, E_ref) region where atompaw's overlap operator stays positive definite is a
  narrow diagonal band — smaller r_c demands a higher mid reference. At r_c(s)=1.56 only
  E_s ~ 14 Ry survives (the atompaw manual's own Si example also uses 14 Ry).
* **The augmentation radius must equal the largest channel radius.** Leaving r_aug=1.90
  while r_c(p)=1.796 leaves a shell in which no projector has support, and the p scattering
  degrades badly: RMS phase error 0.688 vs 0.053 once matched. Q_occ is unaffected.

Incidental finding: Si does not need an f channel. The l=3 phase error is 0.0002 / 0.0025 /
0.029 rad across [0,5]/[5,15]/[15,30] Ry *with no f projector at all* — one to two orders of
magnitude below the s/p/d errors. KKK carry `r_f=2.00` for Si, but for this element it buys
nothing.

## 5. Open / next

1. **D4, D5 blocked** on atompaw's positive-definiteness check. D4/D5 need `vloc_l=4`
   (l=4 local potential, since f projectors occupy l=3); every f-reference variant tried
   (single at 10 or 20 Ry, pair at 4+20) aborts. The first D5 attempt at KKK Table I radii
   *did* build, but with `Q_occ = +7.67 e`: at r_c=1.96 the pseudo 2p keeps only 4% of the
   AE norm (0.039 vs 0.999). Semicore waves need their own much smaller matching radius;
   scanning r_c(2s,2p) in 0.9-1.4 at lmax=2 also failed, so the vloc/shape radii need
   re-tuning together rather than one at a time. D5 matters (KKK put 2p in valence for Si);
   D4 is low value given the f finding above.
2. Solid-state validation, not yet started: QE SCF with each UPF -> CoQui RPA vs `nbnd`
   (50/100/150/250/500) at 2 volumes, extrapolated in `1/nbnd` against the ONCV/ccECP
   references already in `si_rpa_proj.csv`; then the full EOS for the winner. Cross-check
   the XML path through `abinit2coqui` on the same dataset.
3. The expected correlation to plot for the paper: extrapolated RPA-limit error vs `Q_occ`,
   which should separate D0/D1/D2 (identical `Q_occ`) from D3 regardless of their differing
   scattering quality.
4. Code-side lever, complementary to the dataset: CoQui's `shape` compensation mode is the
   analogue of VASP's `NMAXFOCKAE=2` AE pair-density restoration. KKK state norm restoration
   alone is insufficient for GW, so `shape` + a norm-conserving dataset is the combination
   their production setup corresponds to.
