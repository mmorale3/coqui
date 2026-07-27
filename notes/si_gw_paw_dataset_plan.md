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

## 6. Solid-state validation — the resonance rule

Harness: `run_qe.py` (QE 7.4.1 `pw.x`, Si diamond a=10.26 bohr, PBE, 4x4x4 k, ecutrho=8x).
Control first: the library `Si.pbe-n-kjpaw_psl.1.0.0.UPF` gives **E = -93.439419 Ry** with
zero negative rho, so the harness is sound.

The first atompaw UPFs all **diverged**: SCF fell to 3.6e-4 Ry, then blew up with "negative
rho" growing 1.5e-2 -> 3.7e-2 and IEEE_INVALID from XC. Not a cutoff effect — the negative
rho got *worse* at higher cutoff (tested 40-120 Ry, ecutrho 8-16x, beta 0.2-0.4). Section
structure of the UPF is identical to the library dataset, so the defect is in the values:

    D0   PP_Q contains 168.17          kjpaw   all |q_ij| < 0.08

**Cause: a reference energy sitting on a log-derivative pole.** At a resonance the AE
solution has `psi(r_c) -> 0`; normalizing it to unit amplitude at `r_c` inflates the
interior without bound. `E_s = 2.5 Ry` is essentially on Si's s resonance (the AE s
log-derivative has its pole between 1.45 and 4.82 Ry) and gives `int|phi|^2 = 1053`, hence
the 168 in `PP_Q`. `E_p = 3.0 Ry` — which had the *best* log-derivative fidelity — sits
near the p resonance at 3.6 Ry for the same reason. Choosing reference energies off
resonance fixes it:

| E_s | E_p | max abs q_ij | p[0,5] | QE |
|-----|-----|--------------|--------|-----|
| 2.5 | 3.0 | 168.17 | 0.006 | diverged |
| 12 | 10 | **0.240** | 0.007 | **E = -93.441373 Ry** |

D0 now agrees with the kjpaw reference to **2 mRy** — the baseline rung is validated
end-to-end (atompaw -> UPF -> QE). Reference-energy selection therefore has *two*
competing criteria: log-derivative fidelity AND distance from resonance; the second was
missing from the first pass and is now recorded in `gen_inputs.py`.

**Still blocked: the norm-matched rungs.** At r_c(3s)=1.555 / r_c(3p)=1.796 the dataset
cannot yet be made to work in QE. With 2 waves/l atompaw fails positive-definiteness at
every (vloc_l, r_c_vloc, E) tried — the 20 Ry wave is what keeps the small-r_c basis
conditioned, so it cannot simply be dropped. With 3 waves/l it builds (Q_occ = -0.001184,
as reported in §4) but the high-energy wave is itself badly scaled (max|q_ij| = 3.2 at
E_hi=18, 27.5 at E_hi=30 — here it is the *pseudo* norm that is large, 6.58 vs AE 2.49),
and QE returns garbage total energies (-90338 Ry). E_hi = 15/20/22/25 fail to build at all.
So §4's `Q_occ` ladder numbers stand as dataset diagnostics, but only D0 is so far usable
in a solid.

Next lever to try: the pseudization scheme itself. `gen_inputs.py` now takes a `scheme`
key (MODRRKJ / VANDERBILT / BLOECHL / CUSTOM); an earlier attempt to compare schemes was a
no-op because the scheme was hardcoded, so this is genuinely untested. The prior Ga/Zn
`*_kkk` runs used `vanderbilt besselshape` successfully with 3 waves/l at comparable radii,
which is weak evidence it may behave better at small r_c.

> **Superseded by §8.** The scheme was the right thing to look at but the wrong culprit:
> what blocks a norm-matched dataset is the *number* of partial waves per s/p channel, and
> the reference energies of the unbound ones relative to the AE log-derivative poles.

## 7. The norm series N0-N9 — production datasets

The D-ladder answered the *attribution* question but only D0 survived contact with QE. What
the KKK mechanism actually needs is a family differing ONLY in how badly the occupied waves
violate the norm, every member well conditioned and QE-validated. Shrinking r_c walks
`q_aa` through zero, giving a continuous lever:

| name | r_c(s) | r_c(p) | E_s | Q_occ (e) | reduction | E_QE (Ry) |
|------|--------|--------|-----|-----------|-----------|-----------|
| N0 | 1.900 | 1.900 | 12 | -0.162399 | 1.0x | -93.441373 |
| N1 | 1.850 | 1.900 | 12 | -0.133559 | 1.2x | -93.474136 |
| N2 | 1.800 | 1.850 | 12 | -0.107830 | 1.5x | -93.457336 |
| N3 | 1.750 | 1.850 | 12 | -0.082139 | 2.0x | -93.507778 |
| N4 | 1.700 | 1.800 | 16 | -0.055340 | 2.9x | -93.524997 |
| N5 | 1.650 | 1.800 | 16 | -0.034071 | 4.8x | -93.563716 |
| N6 | 1.600 | 1.796 | 16 | -0.014425 | 11.3x | -93.590701 |
| N7 | 1.580 | 1.796 | 18 | -0.007600 | 21.4x | -93.707718 |
| N8 | 1.570 | 1.780 | 18 | -0.003514 | 46.2x | -93.663905 |
| N9 | 1.555 | 1.796 | 18 | -0.001184 | **137x** | -93.729401 |

**137x range in the norm defect, all ten converging in QE** (N6-N9 with zero negative rho),
with low-energy phase errors of 0.01-0.03 rad. `gen_inputs.py --series` regenerates them;
all ten reproduce their `Q_occ` to <1e-6 on rebuild.

`E_s` must be retuned at every radius — it has to stay off the s resonance *and* inside
atompaw's narrow positive-definite band, and both move with r_c. This is what blocked §6:
at r_c=1.555, E_s=14 is near-resonant (max|q_ij| = 3.19, QE returns -90338 Ry), E_s=18 is
clean (0.477, -93.729401 Ry), and E_s=20 breaks it again (-4666 Ry). Nothing about the
norm-matched radii was wrong; the reference energy was.

Absolute total energies are *not* comparable across the series (different pseudization and
frozen-core treatment); only nbnd-converged correlation energies are.

Note on VANDERBILT: it shifts the norm-matching root itself (Q_occ = +0.0437 at the radii
where MODRRKJ gives -0.0012), so `q_aa` depends on the pseudization scheme as well as r_c,
and its q_ij scale worse. MODRRKJ retained.

> **Corrected in §8.** Judging VANDERBILT at *MODRRKJ's* radii was the wrong test — a
> scheme that moves the root has to be evaluated at its own root. VANDERBILT's root is
> r_c = 1.70/1.90, and there it is norm-conserving to +3e-4 e in a much softer dataset.

## 8. The production series P0-P4 — norm-conserving AND d-complete

The N-series settles the norm question but every member has `lmax=1`: no d channel at all.
Si RPA is known to run away without l=2 completeness (the JTH no-d projector diagnostics,
`si_rpa_proj.csv`), so an lmax=1 dataset cannot be the production answer however good its
norm is. The dataset this campaign actually needs is norm-matched **and** d-complete
**and** stable in a solid, and until now nothing satisfied all three.

### 8.1 Two distinct inflation mechanisms, and the diagnostic for each

`paw_diag.py` gained two metrics that make the failures legible instead of empirical:
`qij_matrix` (max|q_ij| from the UPF's `PP_Q`, the conditioning gate — the kjpaw library
dataset has all |q_ij| < 0.08) and `ae_poles` (the energies where the AE log-derivative at
r_c diverges, read off the `logderiv.l` files; only the -inf -> +inf crossings count, the
zeros of L are harmless). The pole map at the norm-matched radii is:

| channel | r_c | AE poles (Ry) | off-resonance mid-points |
|---------|-----|---------------|--------------------------|
| s | 1.563 | 3.45, 19.68 | 11.6, 24.8 |
| p | 1.797 | 4.38, 19.30 | 11.8, 24.7 |
| d | 1.797 | 4.59, 16.48 | 10.5, 23.2 |
| s | 1.90 | 2.57, 16.69 | 9.6, 23.3 |
| p | 1.90 | 3.53, 16.57 | 10.1, 23.3 |
| d | 1.90 | 4.00, 14.46 | 9.2, 22.2 |

With that map the earlier failures separate into two mechanisms, distinguishable by
*which* norm is large:

* **AE-resonance inflation** — the reference energy sits on a pole, `psi(r_c) -> 0`, and
  unit-amplitude normalization blows up the interior of the AE wave. D3's `p@20Ry` has AE
  norm **52.7** against the l=1 pole at 19.30; `SC2_VANDERBILT_18`'s `s@18Ry` has AE 9.14
  and `p@18Ry` AE 16.34 (poles at 19.68/19.30). This is §6's resonance rule, which had
  been applied to the *mid* reference energies but never to the high-energy third wave.
* **Pseudization inflation** — the AE wave is fine but the pseudo wave is not, because an
  unbound wave nearly parallel to its neighbour inside r_c leaves a tiny residual after
  orthogonalization and is then renormalized. D3's `s@14Ry`: AE 0.63 -> PS 3.81.
  `SC2_VANDERBILT_25`'s `s@25Ry`: AE 1.17 -> PS 12.14. This one is invisible to the pole
  map and is the reason `max|q_ij|` has to be checked independently.

### 8.2 The (r_c, E_ref) band is scheme-dependent, and the two schemes are complementary

Mapping build success over the grid (`sweep.py band`) shows the positive-definite band is
not a fixed property of the element — the two schemes occupy *opposite* corners. At
E_ref = 11 Ry, MODRRKJ builds only for r_c >= 1.80 while VANDERBILT builds only for
r_c <= 1.58. `q_aa` is confirmed invariant to the reference energies (identical to 5
digits across every E in a row) and depends only on its own wave, its r_c and the scheme.

VANDERBILT's norm root is therefore reachable and lies at **r_c(3s) = 1.70** (q = -1e-4)
and **r_c(3p) = 1.90** (q = +2.5e-4), i.e. Q_occ = +0.000306 e — four times better than
N9's -0.001184 e, in a dataset 0.15/0.10 bohr *softer* per channel. And at that root
VANDERBILT builds d-complete datasets in **12 of 12** configurations tried, where MODRRKJ
aborts on positive-definiteness at its own root.

### 8.3 What actually unlocks it: the unbound s/p waves, not the scheme

The decisive test is dropping the unbound s/p partial waves altogether, leaving one
norm-matched bound wave per s/p channel plus a two-wave d channel. The occupied
augmentation charges then nearly vanish, and the construction succeeds under **both**
schemes — including MODRRKJ at exactly the radii where its 2-wave-per-l version aborts.
All QE numbers below: PBE, a = 10.26 bohr, 4x4x4 k, 60 Ry, ecutrho = 8x; the library
`Si.pbe-n-kjpaw_psl.1.0.0.UPF` control is -93.439419 Ry.

| name | scheme | r_c(s)/r_c(p) | s/p waves | d refs | Q_occ (e) | max abs q_ij | E_QE (Ry) | neg. rho | RMS[15,30] l0/l1/l2 |
|------|--------|---------------|-----------|--------|-----------|--------------|-----------|----------|---------------------|
| P0 | MODRRKJ | 1.555/1.796 | 1 each | 2, 10 | -0.001184 | **0.157** | -93.449913 | none | 0.61/0.77/0.57 |
| P1 | VANDERBILT | 1.700/1.900 | 1 each | 2, 9 | +0.000306 | 0.533 | -93.448121 | none | 0.58/0.76/0.09 |
| P2 | VANDERBILT | 1.700/1.900 | 1 each | 2, 22 | +0.000306 | 0.997 | **-93.439028** | none | 0.58/0.76/0.04 |
| P3 | MODRRKJ | 1.555/1.796 | s@18, p@10 | 2, 10 | -0.001184 | 0.477 | -93.724550 | none | 0.63/0.66/0.57 |
| P4 | VANDERBILT | 1.700/1.900 | s@26, p@24 | 2, 22 | +0.000306 | 3.471 | -93.448501 | 1.3e-2 | **0.17/0.19/0.04** |

`gen_inputs.py --prod` regenerates all five; each reproduces its `Q_occ` and `max|q_ij|`
exactly on rebuild. Reading the table:

* **P0-P2** are the minimal form. `infl = 1.00` — not a single pseudo wave exceeds its AE
  norm — and P2 lands **0.4 mRy** from the library kjpaw control. What they give up is
  s/p scattering above 15 Ry (one reference energy per channel).
* **P3 is N9's own s/p construction plus the d channel it was missing.** The three earlier
  attempts at this exact dataset (`NM2_23_*`) aborted only because they used E_s = 12/14
  instead of N9's 18 — the second s wave was the obstruction, never the d channel. This is
  the "full package": norm-matched, d-complete, two waves in s/p, no negative rho.
* **P4** buys the high-energy s/p scattering back (0.63/0.66 -> 0.17/0.19 rad) by putting
  the second wave in the *upper* off-resonance window, at the cost of max|q_ij| = 3.5 and a
  small residual negative rho — usable but the one to watch.

Two rules of thumb fall out, both of which contradict what §6 concluded. `max|q_ij|` is a
strong but not sufficient predictor of QE stability: 0.48 (N9) and 0.53 (P1) are fine,
3.47 (P4) converges with a trace of negative rho, but 0.79 with `infl = 2.0`
(VANDERBILT 2-wave, no d) diverges to -759 Ry — the *inflation ratio* discriminates the
last case, so both are worth reporting. And a scheme comparison is only meaningful at each
scheme's own norm root.

## 5. Open / next

0. **Retry D5 with the §8 findings.** The semicore rung was abandoned against the old
   assumption that positive-definiteness was a hard wall; §8 shows the wall is set by the
   unbound waves and the pole map, neither of which was controlled in the D5 attempts. A
   minimal-form D5 (bound 2s/2p/3s/3p only, plus a two-wave d channel, at each scheme's
   own root) is the version that has not been tried.
1. **D4, D5 blocked** on atompaw's positive-definiteness check. D4/D5 need `vloc_l=4`
   (l=4 local potential, since f projectors occupy l=3); every f-reference variant tried
   (single at 10 or 20 Ry, pair at 4+20) aborts. The first D5 attempt at KKK Table I radii
   *did* build, but with `Q_occ = +7.67 e`: at r_c=1.96 the pseudo 2p keeps only 4% of the
   AE norm (0.039 vs 0.999). Semicore waves need their own much smaller matching radius;
   scanning r_c(2s,2p) in 0.9-1.4 at lmax=2 also failed, so the vloc/shape radii need
   re-tuning together rather than one at a time. D5 matters (KKK put 2p in valence for Si);
   D4 is low value given the f finding above.
2. Solid-state validation: QE SCF is now done for N0-N9 and P0-P4 (§7, §8). Still to run:
   CoQui RPA vs `nbnd` (50/100/150/250/500) at 2 volumes via `run_rpa.py`, extrapolated in
   `1/nbnd` against the ONCV/ccECP references already in `si_rpa_proj.csv`; then the full
   EOS for the winner. Cross-check the XML path through `abinit2coqui` on the same dataset.
   The P-series is the production candidate set — P3 for the all-round dataset, P2 for the
   softest/best-DFT one, P4 if high-energy s/p scattering turns out to matter; N0-N9 stay
   as the norm-defect lever for the mechanism plot.
3. The expected correlation to plot for the paper: extrapolated RPA-limit error vs `Q_occ`,
   which should separate D0/D1/D2 (identical `Q_occ`) from D3 regardless of their differing
   scattering quality.
4. Code-side lever, complementary to the dataset: CoQui's `shape` compensation mode is the
   analogue of VASP's `NMAXFOCKAE=2` AE pair-density restoration. KKK state norm restoration
   alone is insufficient for GW, so `shape` + a norm-conserving dataset is the combination
   their production setup corresponds to.
