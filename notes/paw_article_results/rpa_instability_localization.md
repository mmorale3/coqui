# Si PAW RPA instability: cross-code test and localization (2026-07-27)

System: fcc Si, PAW `jth_with_d/Si.xml`, 4×4×4 Γ-centred k-mesh, ABINIT `ecut 25`
/ `pawecutdg 50`, 500-band NSCF WFK (`abinit/eos_jthd/a*/sio_DS2_WFK.nc`).
Every number below uses the **same** WFK, so the mean field is bit-identical and
the only variable is the RPA implementation.

## 1. The dataset is not at fault

ABINIT's own ACFD-RPA (`gwrpacorr 1`, `gwcalctyp 1`, `gw_icutcoul 7` = Gygi to
match CoQui's `div_treatment="gygi"`) vs CoQui THC-RPA. E_c in Ha:

| a (Bohr) | AB n=100 | AB n=250 | AB n=500 | CQ n=100 | CQ n=250 | CQ n=500 | NC n=500 |
|---|---|---|---|---|---|---|---|
| 10.05 | −0.40338 | −0.42509 | −0.43112 | −0.41621 | −0.44284 | −0.59630 | −0.44687 |
| 10.15 | −0.40224 | −0.42405 | −0.43012 | −0.41494 | −0.44125 | −0.58349 | −0.44576 |
| 10.25 | −0.40124 | −0.42314 | −0.42926 | −0.41377 | −0.43983 | −0.55294 | −0.44479 |
| 10.35 | −0.40037 | −0.42236 | −0.42855 | −0.41280 | −0.43860 | −0.52796 | −0.44396 |
| 10.45 | −0.39963 | −0.42170 | −0.42794 | −0.41189 | −0.43752 | −0.50534 | −0.44328 |
| 10.55 | −0.39902 | −0.42119 | −0.42755 | −0.41119 | −0.43662 | −0.48914 | −0.44272 |

Spread over the volume range (NC reference 4.2 mHa):

| nband | ABINIT | CoQui |
|---|---|---|
| 100 | 4.36 mHa | 5.03 mHa |
| 250 | 3.91 mHa | 6.23 mHa |
| 500 | **3.56 mHa** | **107.16 mHa** |

ABINIT's spread *shrinks* with band count. The codes agree through n=250, where
CoQui sits on the NC reference. The 107 mHa volume-dependent over-binding is
generated entirely in CoQui's 250→500 band window.

`ecuteps` probe (a=10.05, n=500) rules out ABINIT's dielectric-matrix truncation
hiding the effect: E_c = −0.41918 / −0.43112 / −0.43169 Ha at ecuteps = 6 / 12 /
18 Ha — converged to 0.6 mHa by 12 Ha.

## 2. Localization (a=10.05, diagnostic = the n=250→500 increment)

ABINIT's increment on the identical mean field is **−6.0 mHa**.

| variant | E_c(250) | E_c(500) | increment |
|---|---|---|---|
| baseline (all on) | −0.44284 | −0.59630 | −153.5 mHa |
| `paw_onsite=false` | −0.43929 | −0.59025 | −151.0 mHa |
| `thresh` 1e-5 + `paw_isdf_tol` 5e-6 | −0.43322 | −0.58261 | −149.4 mHa |
| `vv_compensation="shape"` | −0.44461 | −0.59577 | −151.2 mHa |
| `paw_vgl=false` (no smooth↔aug) | −0.63064 | −3.04640 | −2415.8 mHa |
| `paw_vll=false` (no aug↔aug) | −0.46138 | −2.71550 | −2254.1 mHa |
| `paw_aug=false` (no augmentation) | −0.53971 | −1.36097 | −821.3 mHa |

**Refuted, each worth ≲4 mHa of the 153:** the one-center term; THC/ISDF
compression tolerance; the compensation mode (so moment-mode EOS numbers are not
invalidated by this); `aug_lmax` (default −1 is already "full 2·lmax, no cap",
`pseudopot.h:86`, so nothing was ever truncated).

**The finding.** Removing *either* augmentation block is far worse than removing
*both*: V_GL and V_LL are each ~2.3–2.4 Ha in band-increment and nearly cancel,
leaving −153.5 mHa where ABINIT leaves −6.0. A ~6 % relative error in either
block reproduces the entire residual. This is a catastrophic-cancellation /
conditioning problem, not a missing or mis-specified physics term — which is why
no toml knob moves it.

`paw_aug=false` is **not** a valid control (it breaks the PAW pair densities;
E_c = −1.36 Ha is unphysical) and is bit-identical to `paw_aug=false` +
`paw_onsite=false`, i.e. `paw_onsite` is a no-op once `paw_aug=false`.

## 3. Explicitly ruled out — do not re-raise

The V_GL/V_LL representation asymmetry is **correct**, not a bug. Smooth orbitals
at ecut 25 Ha give pair densities needing ~100 Ha, and `rho_g` is 225.14 Ha
(ngm 40821), so ζ is fully contained and the V_GL sum truncated at `rho_g` is
lossless. The dense-sphere branch used by V_LL (Gcut 25.99 a.u. ≈ 338 Ha,
ngm 64661) exists because η is genuinely not band-limited to `rho_g`.

Structural fact: `N_smooth` grows with band count (2288 @ n=250 → 4301 @ n=500)
while `N_aug` is fixed at 534 (nh=18, nij_max=171, kept nlambda=267/species).
That is structurally right — band dependence enters through ⟨p_i|ψ_n⟩, not the
channel basis — but the two near-cancelling blocks therefore carry very
differently scaling fidelity as bands grow, and high virtuals carry huge
projector amplitudes (max|P| ≈ 38, see `jth_proj_amplitude.pdf`).

## 3b. Code-level band scan — V_LL is EXONERATED

`thc_vgl_vll_band_scan` (test_hamilt.cpp) compares the THC V_LL block against the
direct, non-factorized comp-comp reference, resolved by band decile, on the real
a=10.05 MF. Two results, both against the hypothesis that motivated it.

**(i) Relative accuracy is FLAT with band index.** `rel@max` is the same in the
lowest and highest decile at both n=250 and n=500 (~0.29 at production
tolerance, 4.8e-5 at tight). The joint smooth+aug basis does not represent high
virtuals any less faithfully than low ones.

**(ii) What grows is the blocks themselves.** Across deciles at n=500:
max|V_LL| 9.0e-3 → 7.2e-1, max|V_GL| 5.9e-2 → 5.0e+0, max abs err 2.6e-3 →
2.1e-1 — all ~80x. Driver is visible in the log: max|Pskna| = 3.62 at nbnd=250,
11.13 at nbnd=500. Projector amplitudes grow with band index (cf.
jth_proj_amplitude.pdf), |V_LL| follows as ~P^4, and a CONSTANT fractional error
on a quantity growing 80x gives an absolute error growing 80x. The mechanism is
a constant-quality representation of a quantity that explodes — not a degrading
representation.

**(iii) DECISIVE: V_LL is not the culprit.** Run at `isdf_tol=5e-6, thresh=1e-5`
— the EXACT tolerances of the `tight` RPA run in §2 — the V_LL block is accurate
to rel 4.8e-5 / abs 3.4e-5 (vs 0.29 / 2.1e-1 at production tolerance, a ~4-order
improvement). Yet the RPA at those identical settings still gave E_c = −0.58261,
increment −149.4 mHa against ABINIT's −6.0. **V_LL is represented accurately
precisely where the blow-up persists undiminished, so its representation error
is not the cause.**

Corollary worth noting separately: at PRODUCTION tolerance the V_LL block does
carry a ~29% relative error (abs 2.1e-1 at the top decile). That is a genuine
accuracy defect in the production settings — it just is not what drives the
RPA instability, since removing it changes nothing.

**Remaining suspect: V_GL.** It is ~7x larger than V_LL at high band index
(5.0 vs 0.72 at the top decile), so a relative error an order of magnitude
smaller than V_LL's would still dominate. No direct reference for it exists yet.

## 3c. Oscillator completeness sum rule — the physics is SOUND (2026-07-28)

`paw_oscillator_sum_rule` (test_hamilt.cpp) uses an exact, reference-free
identity: for a complete set of AE bands at a given k,

    sum_c |rho_vc(G)|^2 = <psi_v| e^{-iGr} (sum_c |psi_c><psi_c|) e^{iGr} |psi_v>
                        = 1     for EVERY G and every v,

so a partial sum can only approach 1 from below. Exceeding it proves the
summed oscillators are not AE oscillators of an orthonormal set. The test
self-validates at G=0, where rho_vc(0) = <psi~_v|S|psi~_c> = delta_vc must hold
exactly; on the QE LiH fixture the gate lands at 4e-12 while the deliberately
reversed conj ordering lands at 0.76, so the conventions are pinned rather than
assumed. On the ABINIT-converted Si MF the gate sits at 8e-7 (residual mismatch
between ABINIT's own S and the q_ij reconstructed from the converted dataset).

Si jth_with_d, a=10.05, nbnd=500, k=0, max over the 4 occupied v:

| \|G\| (a.u.) | G^2/2 (Ha) | smooth N=125 | smooth N=250 | smooth N=500 | **AE N=500** |
|---|---|---|---|---|---|
| 0.000 | 0.00 | 1.1100 | 1.2456 | **2.9805** | **1.0000** |
| 1.083 | 0.59 | 1.0994 | 1.2135 | 2.5943 | 0.9869 |
| 2.074 | 2.15 | 1.0621 | 1.1609 | 1.8022 | 0.9704 |
| 3.063 | 4.69 | 0.5125 | 1.0477 | 1.2749 | 0.9839 |
| 3.954 | 7.82 | 0.1767 | 0.4759 | 1.0951 | 0.9986 |
| 5.002 | 12.51 | 0.0027 | 0.1406 | 0.4932 | 0.4494 |
| >= 8.0 | >= 32 | 0.0000 | 0.0000 | 0.0001 | 0.0003 |

**The exact PAW augmented oscillators are sound at 500 bands** — every value at
or below 1, approached monotonically, at every \|G\|. On-site incompleteness is
therefore NOT the mechanism, and the dataset is exonerated a second time by an
argument that does not go through ABINIT at all.

Two consequences, both eliminating candidate fixes:

- **A high-\|G\| augmentation cutoff cannot be the fix.** Above \|G\| = 8 a.u.
  the augmented oscillators are identically zero, so there is nothing spurious
  up there for CoQui's larger rho_g sphere (\|G\| to 21 a.u.) to pick up that
  ABINIT's `ecuteps` truncation avoids. The `paw_aug_ecut` knob added in this
  session (thc_reader_t, defaults OFF, cuts eta itself so Z stays a PSD Gram
  matrix of {zeta, eta_cut}) is retained as a control but is not the answer.
- **What the augmentation has to do is now quantified.** The SMOOTH oscillators
  carry an excess reaching 2.98 at G=0 that grows steeply with band count
  (1.11 -> 1.25 -> 2.98 for N=125/250/500). The augmentation must cancel a
  factor of three, and the physics supports it exactly.

**The excess tracks the volume dependence of the error.** At a=10.55 the G=0
smooth value is 1.9528 (vs 2.9805 at a=10.05), i.e. an excess to cancel of 0.95
vs 1.98, ratio 2.1 — against a CoQui error ratio of 165/61 = 2.7 at those same
two volumes (§1). The over-binding scales with the SIZE OF THE CANCELLATION,
which is what a fixed missed FRACTION of it would produce.

## 4. Recommended next step (code-level, not another sweep)

Superseded twice. §3b killed the V_LL-representation hypothesis; §3c killed
both the dataset/on-site-completeness hypothesis and the high-\|G\| cutoff fix,
and reframed the question: the exact AE ERI is right, the smooth one is a
factor ~3 too big, and the only open question is what fraction of that
cancellation the ASSEMBLED THC ERI actually realizes. Every probe so far
compared one augmentation BLOCK against a direct reference; none compared the
assembled ERI against the exact answer.

`paw_thc_vs_exact_eri` (test_hamilt.cpp) is that measurement. Per occupied v,
restricted to q=0 so the exact side stays tractable, it compares

    D(v) = sum_c (v c | c v) / (eps_c - eps_v)     [the RPA's own integrand]

in four columns: exact AE, exact smooth, THC AE, THC smooth. The smooth pair is
the CALIBRATION — it fixes the prefactor and the contraction convention
independently, and if it does not read ~1 nothing in the AE columns can be
believed. Two traps it caught while being built, both recorded because they are
the same class of error that produced four plausible-but-meaningless references
earlier in this investigation: (1) `math::nda::fft` normalizes on the FORWARD
transform, so the natural-looking extra 1/nnr is wrong — anchored by requiring
rho~_vv(G=0) = sum_g |C(v,g)|^2, which fails by exactly nnr if you get it
wrong; (2) `hf_t`'s K carries the 1/N_k BZ factor, which showed up as the
calibration reading exactly 0.124971 = 1/8 on the 8-k LiH fixture. With both
fixed it reads calibration 0.999766 / measurement 0.999703 on LiH.

Historical note, superseded: the previous recommendation here was a direct
reference for **V_GL**, on the grounds that it was the only block never checked
and the largest at high band index (5.0 vs V_LL's 0.72 at the top decile).
`paw_thc_vs_exact_eri` subsumes it — it checks the whole assembled ERI,
V_GL included, against the exact answer rather than block by block.

## 5. The q-sign hypothesis (OPEN — test queued)

`paw_thc_vs_exact_eri` at **q = 0**, Si jth_with_d:

| case | calibration (THC sm / exact sm) | **THC AE / exact AE** |
|---|---|---|
| a=10.05, nbnd=250 | 1.000000 | **1.000001** |
| a=10.05, nbnd=500 | 0.999996 | **0.999996** |
| a=10.55, nbnd=500 | 0.999998 | **0.999998** |

Every band decile matches to 5–6 digits. **The assembled THC ERI is exact at
q = 0, including at 500 bands, at both volumes.** V_GL, V_LL and the smooth
ISDF all reproduce the exact AE answer there. That does NOT close the case —
and combined with one older datum it becomes close to a deduction that the
defect lives at q != 0:

- `paw_aug=false` at n=500 gives E_c = −1.36 Ha, ~3x the correct −0.43, which
  is exactly the factor the smooth oscillators are too large by (§3c). So the
  RPA machinery converts oscillator strength into E_c faithfully; it is not
  the solver.
- With augmentation on, E_c = −0.596, i.e. the augmentation removes 83% of
  that excess (§3c).
- But the ERI test says the augmentation is **100%** effective at q = 0.

E_c is dominated by the 63 q != 0 points, so an augmentation that is fully
effective at q = 0 and 83% effective overall must be losing ground at q != 0.
The q = 0 table above also shows WHY it cannot be otherwise: the
Coulomb-weighted q = 0 sum shows only a 7% smooth excess (AE/smooth 0.9293 at
n=500), not the 3x the sum rule shows at G = 0, because `ignore_g0` zeroes
precisely the G -> 0 region where the excess concentrates. At q != 0 that
region is present, non-singular, and carries the largest Coulomb weight in
the problem.

`augment_thc_with_paw` rebuilds the augmentation channels at every q,

    q_cart = { -Qpts_cart(iq, 0), -Qpts_cart(iq, 1), -Qpts_cart(iq, 2) }

and the sign is justified in `build_eta_on_rho_g_at_q`'s header as chosen "to
match the Coulomb kernel convention used in V_GG (thc.icc evaluates the kernel
at \|G - Q_thc\|, so q_cart = -Q_thc(iq) here)". **That criterion cannot
constrain what it is being used to justify.** \|G - Q\| is a magnitude; the
Coulomb kernel is identical for K and -K. eta is not: it carries
Y_lp(K_hat) and the structure factor e^{-iK.tau_a}, so eta^q(G) at K = G - Q
and at K = G + Q are different complex numbers. The sign was fixed against a
quantity blind to it.

A wrong sign here would have exactly the fingerprint that has resisted every
sweep:

- **exact at q=0** (the sign is moot there) — consistent with 1.000001 above;
- **invisible on the LiH 2x2x2 fixture**, where q and -q differ by a
  reciprocal lattice vector, so the two conventions agree identically. The
  q!=0 LiH check reads 1.000012 and is therefore NOT evidence against this;
- **corrupts V_GL at general q** on the 4x4x4 Si mesh (V_LL is more forgiving:
  the phases cancel on its diagonal), which is the block estimated at ~10%
  deficient and the only one never checked;
- **band-count independent as a FRACTION**, matching the constant 84.5% /
  82.2% cancellation at n=250 / n=500;
- **tolerance independent**, matching the immunity to thresh and paw_isdf_tol;
- **absent for NC** (no augmentation at all).

Test: `paw_thc_vs_exact_eri` with COQUI_ERICHK_K2 != 0 on the Si 4x4x4 mesh,
run at BOTH COQUI_ERICHK_QSIGN = +1 and -1. The exact reference is built
independently of the THC for each sign, and the completeness gate
sum_c |rho_{v k0, c kc}(q+G)|^2 <= 1 (exact at any q, evaluated over every G of
rho_g) says which sign is physical. If the THC matches the reference built
with the UNPHYSICAL sign, that is the defect. On LiH the two signs separate
cleanly — gate 0.990652 / calibration 1.000049 for +1 versus gate 1.014059 /
calibration 1.323530 for -1 — so the discriminator works; it simply has no
purchase on a 2x2x2 mesh.

Status: **REFUTED.** Si 4x4x4, nbnd=250, a=10.05:

| k_virt | \|q\| | qsign | gate | calibration | THC/exact |
|---|---|---|---|---|---|
| 1 | 0.271 | **+1** | 1.021258 | 1.000022 | 0.999881 |
| 5 | 0.313 | **+1** | 1.020984 | 1.000016 | 0.999772 |
| 1 | 0.271 | −1 | 1.072055 | 1.007445 | 0.993853 |
| 5 | 0.313 | −1 | 1.244305 | 1.021090 | 0.987228 |

`+1` wins on all three independent measures and the THC matches it, so the sign
in `augment_thc_with_paw` is CORRECT — as the derivation says it should be
(`coulomb_t::evaluate` forms `dk = kp - kq`, `thc.icc` passes `kp=0, kq=Q`, so
the kernel really is at `|G - Q|` and `q_cart = -Q` matches). The violation is
also flat in \|q\| (1.021258 vs 1.020984), where a missing `e^{-iq.tau}` phase
would grow as \|q.tau\|^2, so it is not a phase error either.

## 6. The compression defect (REAL, but worth only ~14 mHa)

The q != 0 gate violation IS the lambda compression. Same q, same everything
else, only `paw_isdf_tol` changed:

    paw_isdf_tol = 5e-5  (production)   gate = 1.021258
    paw_isdf_tol = 1e-12 (full rank)    gate = 0.999054

Mechanism: `build_local_isdf_compressed_by_norm` keeps (ij) pairs by their
Coulomb-metric norm evaluated at **q = 0**. A pair with l_i != l_j has no L=0
component, so Q_ij(G) -> 0 as G -> 0 and its q=0 Coulomb-metric norm is small —
it gets dropped. But at q != 0 the smallest wavevector on the mesh is
\|q\| = 0.271, not 0, and there those channels are NOT small. Hence: exact at
q=0, violated at q!=0, invisible to every q=0 test. This is the physical
characterization of the previously unexplained "V_LL carries ~29% relative
error at production tolerance" (§3b corollary).

It is a genuine defect and production settings should not be used, but it is
**not** the instability: the converged RPA run below uses `paw_isdf_tol = 1e-8`
(essentially full rank) and is unchanged.

## 7. ISDF convergence — EXCLUDED quantitatively

| thresh | paw_isdf_tol | Np | E_c (Ha) | vs ABINIT |
|---|---|---|---|---|
| 1e-4 | 5e-5 | 4301 | −0.596300 | −165.18 mHa |
| 1e-5 | 5e-6 | 4947 | −0.582610 | −151.49 mHa |
| 1e-6 | 1e-8 | 6205 | −0.582622 | −151.50 mHa |

A full decade of `thresh` (Np 4947 -> 6205) moves E_c by **0.01 mHa**. The RPA
is converged in the ISDF basis and in the lambda basis, and remains 151 mHa
from ABINIT. `nIpts` does not help: it does not override the threshold-driven
Cholesky rank (both the `thresh=1e-6` and `nIpts=8000` runs landed on
Np = 6205).

## 8. The ERI is EXACT — and that is the surprise

`paw_thc_vs_exact_eri`, full lambda rank, Si a=10.05:

| q | nbnd | gate | calibration | **THC AE / exact AE** |
|---|---|---|---|---|
| 0 | 250 | (exact by identity) | 1.000000 | **1.000001** |
| 0 | 500 | (exact by identity) | 0.999996 | **0.999996** |
| \|q\|=0.271 | 250 | 0.999054 | 1.000022 | **0.999882** |
| \|q\|=0.271 | 500 | 1.003246 | 0.999995 | **0.999875** |

Every band decile agrees to ~1e-4, including the 250->500 window. So at
converged settings the augmented oscillators satisfy completeness AND the
assembled THC ERI reproduces the exact AE answer — yet E_c is still 151 mHa
wrong. Neither the oscillators nor the ERI assembly is the defect.

**A conditioning explanation was proposed here and is now REFUTED** (see §9).
The argument was: two ratios from the same data,

    polarizability trace   D_smooth / D_AE    = 34.96 / 32.49 = 1.076   (7.6%)
    correlation energy     E_c(sm) / E_c(AE)  = 1.36  / 0.43  = 3.2     (220%)

E_c = Tr(Pi*Z) + ln\|det(I - Pi*Z)\| is a SECOND-ORDER quantity: for small Pi*Z
the two terms cancel to leading order and the sum is -Tr((Pi*Z)^2)/2. A 7.6%
change in Pi should move E_c by ~16%. Producing 220% requires eigenvalues of
Pi*Z approaching 1, where ln det is dominated by ln(1 - lambda) and the
response to Pi is exponential rather than quadratic. That also explains why
bands 250->500 add only 0.9% to the trace while moving E_c by −149 mHa, and
why no tolerance ever helped: **the input is correct; the conditioning is not.**

Instrument added (`thc_rpa.icc`): a `RPA conditioning:` line reporting
`sum/(Tr^2/2)` where `sum = Tr(Pi*Z) + ln|det|`. ~1 is the ordinary
second-order cancellation; >> 1 means the block is running into the
singularity.

## 9. Conditioning — REFUTED by its own diagnostic

`rpa_cond/`, Si a=10.05, thresh 1e-5, paw_isdf_tol 1e-8:

| run | max \|resid\|/(Tr^2/2) | log_det<0 / Tr>0 | E_c (Ha) |
|---|---|---|---|
| PAW n=250 | 2.21e-02 | none | −0.433243 |
| PAW n=500 | 4.65e-01 | none | −0.582623 |

Both well under 1, no physical violations anywhere. **E_c is squarely in the
second-order regime; the RPA is not near the ln(1 − lambda) singularity.** The
"220% from 7.6%" argument in §8 does not hold and is withdrawn.

It was weak on its own terms and should have been flagged as such when written:
the 7.6% is an AGGREGATE over all 40821 G, dominated by large \|q+G\| where the
smooth excess is small, while the sum rule shows the excess reaching 2.98 at
G = 0. The Pi error in the region E_c actually weights may well be ~200%,
which needs no conditioning story at all.

Two corrections to the diagnostic itself:
- `|resid|/(Tr^2/2) ~ sum lambda^2/(sum lambda)^2 = 1/n_eff` is an INVERSE
  PARTICIPATION RATIO (how few modes of Pi*Z carry a block), not a distance to
  the singularity. What it does show, and this is real: the n=500 spectrum is
  dominated by ~2 modes where n=250 spreads over ~45.
- the report conflated two maxima over DIFFERENT (q, w) blocks. At the
  largest-\|resid\| block (iq=0, iw=68) Tr moves only 5.6% and \|resid\| only
  0.2% between n=250 and n=500 while E_c moves 34% — so that block is not
  where the change lives.

## 9b. The "PAW-specific spectral anomaly" — RETRACTED (metric artifact)

Reported and then withdrawn within the same session. With block identification
fixed, the maxima are:

| run | max 1/n_eff | at | Tr there | resid there |
|---|---|---|---|---|
| PAW n=250 | 2.21e-02 | iq=21 iw=68 | −2.64e+01 | −7.70e+00 |
| NC n=500 | 1.99e-02 | iq=3 iw=68 | −2.80e+01 | −7.78e+00 |
| PAW n=500 | 4.65e-01 | iq=21 **iw=135** | **−8.30e-04** | **−1.60e-07** |

`iw=135` of 137 is the top of the frequency range, where Pi -> 0. The ratio is
`|resid|/(Tr^2/2)` with BOTH parts vanishing, so an empty block returns a
garbage O(1) value from noise/noise^2 — 1.6e-7/(8.3e-4^2/2) = 0.465. The
controls happened to peak on real blocks (Tr ~ −27), which made the cross-run
comparison look like a PAW-specific anomaly. It was an artifact of a missing
magnitude floor in my own metric. Floor added (`|Tr| > 1e-2` to be eligible).

## 9c. THE TERM SPLIT — the actual result, and it reframes the accuracy target

| run | Tr(Pi*Z) | ln\|det\| | E_c |
|---|---|---|---|
| PAW n=250 | −7.5289 | +7.0957 | −0.4332 |
| PAW n=500 | −10.1383 | +9.5557 | −0.5826 |
| NC n=500 | −10.2291 | +9.7822 | −0.4469 |

**E_c is a ~1.4% residual of two ~10 Ha terms.** Two consequences:

**(i) The accuracy requirement is ~1e-3 relative on each term, not 1e-2.** A
1.3% error in Tr(Pi*Z) alone reproduces the entire 136 mHa gap to NC.

**(ii) The first term carries NO 1/Delta weight, and that invalidates the
metric used in sec 8.** Tr(Pi*Z) is `(1/2pi) int dw Tr[chi0(iw) v]` with
`chi0(iw) = sum_ia 2*Delta/(Delta^2+w^2)|rho_ia><rho_ia|`, and
`int dw 2*Delta/(Delta^2+w^2) = 2*pi` INDEPENDENT of Delta. So that term is
just `-sum_ia (ia|ai)` — an exchange-like sum weighting high-energy
transitions exactly as much as low-energy ones.

Every ERI number in sec 8 used `D1(v) = sum_c (vc|cv)/(eps_c - eps_v)`, which
de-weights precisely the bands that dominate Tr(Pi*Z). That is why bands
250->500 add only 0.9% to D1 but **+2.61 Ha (35%) to Tr(Pi*Z)** — and it
dissolves the "1% vs 34% contradiction" of sec 10 without needing any anomaly:
the contradiction was an artifact of measuring with the wrong weight.

`paw_thc_vs_exact_eri` now reports BOTH: D1 and `D0(v) = sum_c (vc|cv)` with no
denominator. D0 is the one that matters. Validated on LiH: calibration
0.999853, measurement 0.999790, and the smooth excess is visibly larger under
D0 (1.598x) than D1 (1.367x) as expected.

## 13. RESOLVED — post-fix RPA vs ABINIT

Si jth_with_d, a=10.05, thresh 1e-5, paw_isdf_tol 1e-8:

| nbnd | ABINIT | CoQui pre-fix | **CoQui post-fix** | residual |
|---|---|---|---|---|
| 250 | −0.42509 | −0.433243 | **−0.430864** | 5.77 mHa |
| 500 | −0.43112 | −0.582623 | **−0.437371** | 6.25 mHa |

**The band-count runaway is gone.** The 250->500 increment, which was the
disease:

    ABINIT           -6.03 mHa
    CoQui pre-fix  -149.4  mHa
    CoQui post-fix   -6.51 mHa      <- matches ABINIT to 0.5 mHa

E_c at n=500 went from 151.5 mHa off to 6.3 mHa off, and the residual is now
essentially band-count independent (5.77 at n=250 vs 6.25 at n=500) instead of
exploding. Post-fix PAW (−0.4374) sits between ABINIT-PAW (−0.4311) and
CoQui-NC (−0.4469), as it should.

The term split isolates the mechanism beyond doubt:

| | Tr(Pi*Z) | ln\|det\| | E_c |
|---|---|---|---|
| pre-fix | −10.1383195 | +9.5556973 | −0.582622 |
| post-fix | −10.1380636 | +9.7006923 | −0.437371 |
| change | +0.00026 (0.003%) | +0.14499 (1.5%) | +145 mHa |

Tr(Pi*Z) is built from DIAGONAL ERIs and is untouched. ln|det| is built from
the full matrix, including the off-diagonal elements that were transposed, and
carries the entire 145 mHa. Nothing else in the calculation moved.

### Off-diagonal ERI, before vs after — why n=500 failed and n=250 did not

`paw_thc_vs_exact_eri` OFFDIAG probe, Si a=10.05, paw_isdf_tol 1e-8:

| nbnd | thresh | **pre-fix** | **post-fix** |
|---|---|---|---|
| 250 | 1e-4 | 1.390937 | 0.999942 |
| 250 | 1e-5 | 1.390985 | 0.999990 |
| **500** | **1e-4** | **168.30** | **1.001326** |

**A 16,730% error at n=500**, against 39% at n=250. That is the quantitative
reason the instability was a band-count runaway: at n=500 the exact AE
off-diagonal is 1.368e-03 against a smooth part of 1.841e+00 -- a **1345x
cancellation** -- so the transposed V_LL does not perturb the answer, it swamps
it. At n=250 the same cancellation is only ~2.4x.

Post-fix the residual is ordinary ISDF error that converges with thresh
(5.8e-5 -> 1.0e-5 at n=250), not a systematic term.

### EOS spread — the original symptom, resolved

| a (Bohr), n=500 | ABINIT | CoQui pre-fix | **CoQui post-fix** |
|---|---|---|---|
| 10.05 | −0.43112 | −0.59630 | **−0.437371** |
| 10.55 | −0.42755 | −0.48914 | **−0.433617** |
| **spread** | **3.57 mHa** | **107.16 mHa** | **3.75 mHa** |

NC reference spread is 4.2 mHa. **107.16 -> 3.75 mHa, matching ABINIT to
0.2 mHa.** The residual offset is FLAT in volume (6.25 mHa at a=10.05, 6.07 at
a=10.55), so it cancels out of the EOS entirely.

Residual 6 mHa: within the spread of method differences (ABINIT truncates the
dielectric matrix at ecuteps 12 Ha while CoQui uses the full THC basis; PAW
forces inclvkb=0 there). Not investigated further -- it is flat in band count,
so it does not touch the EOS.

## 12. THE BUG — V_LL conjugates the wrong index of Z

`thc_reader_t.hpp`, both V_LL branches (rho_g ~line 973, dense ~line 1054).

The convention is set by the smooth block, `thc.icc:1608`:

    multiply(Z_quG, dagger(Z_quG))   ->   Z_uv = sum_G zeta_u v conj(zeta_v)

i.e. **conjugate the SECOND (column) index**. V_GL follows it:
`gemm(omega, zP, dagger(ewQ))` -> `Omega sum_g zeta_u conj(eta_lambda) w`.

Both V_LL branches conjugated the **FIRST** index instead:

    rho_g:  ePc = conj(eta_P);        gemm(omega^2, ePc,    transpose(ewQ))
    dense:  etaP_v = conj(etaP_v);    gemm(omega^2, etaP_v, transpose(etaQ_v))

V_LL is Hermitian, so conjugating the wrong index stores exactly its
**TRANSPOSE**.

### Why nine months of testing never saw it

- **Diagonal elements of a Hermitian matrix are real**, so every diagonal ERI
  is unaffected. D1, D0, the head-resolved sweep, `thc_vgl_vll_split`, the band
  scan — all diagonal.
- **V_LL_old is itself Hermitian** (`V_LL_old(l,x)* = V_LL_old(x,l)`), so the
  assembled Z stays Hermitian and the 1e-8 Hermiticity check cannot see it.
- **The pair densities are untouched** — only Z is wrong — so the ABINIT
  oscillator comparison still matches to 1e-5 (sec: crosscode/README.md).
- **V_LL contains no zeta**, so the error is thresh-independent: measured
  1.390937 at thresh 1e-4 and 1.390985 at 1e-5, frozen to four digits.
- **Tr(Pi*Z) uses only the diagonal** and was right (PAW −10.138 vs NC −10.229);
  **ln|det(I-Pi*Z)| uses the whole matrix** and carried the 30%.
- **eta is nearly real on the LiH fixture** (2 atoms, high symmetry) so the
  transpose is nearly a no-op there — the fix moves LiH by <1e-5. On Si
  jth_with_d, with two atoms at (0,0,0) and (1/4,1/4,1/4) and d-channels, eta
  is strongly complex and the error is 39%.

Fix: `dagger` on the second index in both branches, with `ePc`/`etaP_v` left
unconjugated.

## 11. THE DIAGNOSIS — off-diagonal ERIs, and why every earlier test missed them

`Tr(Pi*Z)` and `ln|det(I - Pi*Z)|` depend on DIFFERENT classes of ERI:

    Tr(Pi*Z)        = sum_ia w_ia (ia|ai)      -> DIAGONAL elements only
    ln|det(I-Pi*Z)| -> Pi*Z as a MATRIX        -> general (ia|bj), (i,a)!=(j,b)

and the term split says the trace is right (PAW −10.138 vs NC −10.229, 0.89%)
while E_c is 30% off. **Every ERI number in this campaign — D1, D0, the
head-resolved sweep, even the ABINIT oscillator comparison — constrains only
the diagonal.** The off-diagonal class had never been measured.

New probe (`OFFDIAG` lines in `paw_thc_vs_exact_eri`): a Hermitian NON-diagonal
density matrix on the top band decile, giving `sum_cd Dm_cd (vc|dv)`. On the QE
LiH fixture, THC vs the exact reference:

| thresh | Np | diagonal (D0) | **off-diagonal** |
|---|---|---|---|
| 1e-3 | 175 | 1.50e-3 | 5.62e-3 |
| 1e-4 | 220 | 2.10e-4 | **2.70e-3** |
| 1e-5 | 280 | 9.7e-5 | 4.39e-3 |
| 1e-6 | 375 | 5.0e-6 | 6.26e-4 |
| 1e-7 | 542 | 1.6e-5 | 2.75e-4 |
| 1e-8 | 779 | 2.0e-6 | 1.1e-5 |

The diagonal converges cleanly (300x over three decades). **The off-diagonal is
10-100x worse at the same thresh, non-monotonic, and needs Np = 779 — 3.5x the
production basis — to reach the accuracy the diagonal has at Np = 375.**

That is the missing piece, and it explains the observation that killed every
earlier hypothesis: **tightening `thresh` does not move E_c** (1e-5 -> 1e-6 gave
0.01 mHa) because the error that drives E_c is not the one the diagonal-based
convergence criterion measures.

Mechanism: the probe's exact AE value is −4.573e-03 against a smooth part of
−2.123e-01 — a **46x cancellation**, versus 1.6x for the diagonal. The ISDF
error is relative to the SMOOTH magnitude and the augmentation is added
exactly, so the error lands on the AE quantity amplified by the cancellation
ratio. Diagonal ERIs are norm-like and forgiving; off-diagonal ones are not.

### The fix direction

Out-resolving it is not viable: matching the diagonal's accuracy needs ~3.5x
the basis, i.e. Np ~ 15-20k on Si against a production 4301, and the RPA solve
is O(Np^3).

The amplification has to be removed instead, by an EXACT regrouping that shrinks
what the ISDF must fit:

    current:   rho_AE = rho~                + sum YY.eta       eta   ~ (phi*phi - phit*phit)
    proposed:  rho_AE = [rho~ - sum YY.etat] + sum YY.eta_AE   etat  ~ phit*phit
                                                                eta_AE ~ phi*phi

Identical rho_AE (eta_AE - etat = eta), but the ISDF target becomes
`rho~ - conj(chi_a) chi_b` with `chi_a = sum_i P_ai phit_i` the on-site PS
expansion — which is SMALL inside the spheres by PAW completeness and equals
rho~ outside. Both ingredients exist in the species data (`sp.aewfc`,
`sp.pswfc`); `build_qrad_tab_full_aeps` already forms `phi*phi - phit*phit` and
needs only PS-only / AE-only variants.

The cost is that this changes what the THC builder fits, which is a deeper
change than any knob: the target `conj(psit_a)psit_b - conj(chi_a)chi_b` is a
DIFFERENCE of two products, so it does not fit the single-product collocation
ansatz directly and needs the same +/- polarization split the aug channels
already use.

## 10. THE REMAINING GAP — the ERI is verified in AGGREGATE, not at the HEAD

This is the honest state of the investigation and the next thing to test.

Every ERI number in §8 is `D(v) = sum_c (vc|cv)/(eps_c - eps_v)` summed over
ALL 40821 G of rho_g. That aggregate is dominated by large \|q+G\|, where the
smooth excess is only ~7% (AE/smooth 0.9294) — so "THC = exact to 1e-4"
establishes accuracy where the cancellation is MILD. It does not establish it
at \|q+G\| -> 0, where the sum rule says the cancellation is 3:1 and where
4pi/(Omega \|q+G\|^2) puts essentially all of E_c's weight. The q=0 runs are
blind there by construction (`ignore_g0` removes G=0), and the q!=0 runs
average it away.

Test: `erichk_head.sbatch` sweeps `COQUI_ERICHK_AUGECUT` over
0.05 / 0.2 / 1.0 / 4.0 / off at n=500, q!=0, full lambda rank. Restricting eta
to \|q+G\|^2/2 <= AE on BOTH sides isolates the head's augmentation, so the
THC/exact ratio at small AE is the head accuracy the aggregate never tested.

CAUTION recorded because it already cost one run: the first attempt
(job 6690036) returned byte-identical numbers for every cutoff INCLUDING
"off", because rusty's test binary predated the `COQUI_ERICHK_AUGECUT`
handling. If the sweep does not move `exact AE`, and the `eta truncated at ...`
line is absent, the binary is stale — the result is void, not null.

Constructing it is harder than the V_LL reference, and the asymmetry is the
reason it was not done first. V_LL is a pure one-center object, so
`compute_paw_deeq(V_comp)` contracted with the projectors captures it entirely.
The V_GL cross term has two pieces:
  (a) band i's augmentation charge against V[ρ_smooth] — available the same way,
      as `compute_paw_deeq(nii, V_smooth_r)` contracted with projectors;
  (b) band i's SMOOTH pair density against V[ρ_comp] — needs
      ⟨ψ̃_i|V_comp(r)|ψ̃_i⟩ evaluated on the real-space grid, which the deeq
      machinery does not provide.
Both must be derived against what `hf_t` actually assembles for the
`paw_vgl=true` path (thc_reader_t.hpp:916-960, the V_GL/V_LG GEMMs and their
stitch blocks) before coding — a wrong split would silently produce a
plausible-but-meaningless reference, which is exactly the failure mode this
investigation has already hit three times.

Two traps recorded for whoever does it. (1) `compute_paw_deeq` expects the local
potential on the DENSE aug mesh (`nnr_aug`); `thc_vgl_vll_split` passes the
smooth mesh and is only correct because its QE fixture has both meshes equal —
on an ABINIT MF they are 18³ vs 48³ and the short array yields an all-NaN deeq.
(2) Report ABSOLUTE error as primary: relative error against near-zero elements
produced apparent 0.27–16 discrepancies here that were 1e-8 in absolute terms.

## Reproduction

Generators/harvesters in this directory; runs on rusty under `~/ceph/CoQui/abinit/`:
`rpa_eos_jthd/` (ABINIT), `rpa_eos_jthd_coqui_nb/` (CoQui n=100/250),
`rpa_localize_jthd/` + `rpa_localize2_jthd/` (localization).

Not available: the `pawcross 1` probe (on-site completeness in ABINIT's
oscillator strengths) aborts with heap corruption in this ABINIT 10.6.7 build at
both 64 and 16 ranks.

## 14. Channel selection made q-aware (2026-07-29) — §6 defect FIXED

§6 identified the mechanism but left it in place, on the grounds that the
converged runs use `paw_isdf_tol = 1e-8` where nothing is dropped. That is true
for Si (nh=18 gives `kept nlambda = 324 (full-rank cap = 324)`), but it left a
loaded gun for any system or tolerance where the cap is not saturated. The
criterion is now fixed rather than avoided.

`build_local_isdf_compressed_by_norm` is still used to build the channels, but
at **tol = 0** (full rank). Selection then happens in
`thc_reader_t::select_aug_channels_qaware`, which ranks each (I,J) pair by

    d_IJ = max over q on the FULL mesh of  sum_G w_q(G) |eta_IJ(q+G)|^2
    w_q(G) = 4pi / (Omega |q+G|^2),  the |q+G| -> 0 term dropped

and keeps it when `sqrt(d_IJ / max_IJ d_IJ) > paw_isdf_tol`. The tolerance is
therefore RELATIVE and system-independent. eta is evaluated with the same
interpolated tables (`build_eta_on_rho_g_at_q_chunk`) the augmentation itself
uses, over the same rho_g sphere, so this is the production path and not a
parallel approximation of it. Cost is one eta build per q — O(nq * N_aug * ngm)
— done redundantly on every rank (it is pure local table evaluation, not
collective, and is deterministic so all ranks agree). Not separately timed on
Si; on the LiH fixture it is not resolvable against the 9.6 s THC build. If it
ever shows up on a large mesh, the fix is to rank on a strided subset of q
rather than to go back to q=0.

Max-over-q, not q=0, is the point: the ERI sums over the whole mesh, and a pair
with no L=0 component has Q_IJ(G) -> 0 like G^L as G -> 0. At q=0 it collects
none of the 1/G^2 enhancement and looks negligible; at the smallest wavevector
the mesh actually contains (|q| = 0.271 a.u. for Si 4x4x4) it is not small.

Ranking is over `Qpts()`, the full q-mesh, which is a superset of the
symmetry-distinct set — conservative by construction, so no symmetry argument
is needed for correctness.

Two things that were silent are now logged, because this defect's whole
character is that it is invisible to diagonal tests:
  - how many pairs were dropped, and the largest dropped one as a fraction of
    the largest kept;
  - how many KEPT pairs the old q=0-only ranking would have discarded, and the
    largest factor by which it under-ranked them.

Cache staleness closed at the same time: the local-ISDF h5 now carries a
`selection` attribute (`kISDFSelectionTag = "qaware-maxq-v1"`). A cache written
under a different rule is rejected with a warning and rebuilt — the same `tol`
under a different rule yields a different channel set, so reusing it silently
would reintroduce exactly the bug being fixed. Pre-v1 caches carry no tag and
are treated as stale.

On the tolerance question: for Si at 1e-8 the cap is saturated, so nothing is
dropped and smaller values change nothing. The robust check is not the number
itself but the logged `kept nlambda == full-rank cap`. Since the aug block is a
small fraction of the basis (648 of Np = 5588 for Si, so full rank costs ~2% in
Np and ~4% in Pi memory over a compressed set), the default `paw_isdf_tol =
1e-12` is deliberately conservative: at that relative threshold a pair must be
~1e-24 in squared norm to be dropped, i.e. effectively full rank.

CAUTION for anyone re-deriving the §6 numbers: the 1.021258/0.999054 gate is the
**Si q!=0** gate. The LiH `paw_oscillator_sum_rule` **G=0** gate reads 4.1e-12 at
both tolerances and does NOT reproduce them — it is a different quantity, and
confusing the two will make the defect look nonexistent. On LiH the visible
signature is instead the off-diagonal residual max_{v!=c}|rho_vc(0)|, 8.9e-16 at
full rank vs 4.1e-9 at 5e-5.

## 15. Si EXX+RPA lattice constant — the pre-fix curve, and why a0 hid the bug

Redoing the EOS with the V_LL fix required re-running the whole series, so the
pre-fix series (`~/ceph/CoQui/abinit/eos_conv500_coqui`, n=500, thresh=1e-4,
paw_isdf_tol=5e-5 — i.e. carrying BOTH the transpose and the q=0 selection
defect) was fitted first, as the "before" datum.

    a (Bohr)   CoQui Total      E_Ewald        E_total      E_c(RPA)
      10.05     0.20297847   -8.57599689    -8.37301842   -0.61862873
      10.15     0.11802472   -8.49150431    -8.37347958   -0.61461430
      10.25     0.03508032   -8.40866036    -8.37358004   -0.61116651
      10.35    -0.04592027   -8.32741727    -8.37333754   -0.60822378
      10.45    -0.12501516   -8.24772906    -8.37274422   -0.60570501
      10.55    -0.20227051   -8.16955154    -8.37182205   -0.60357906

Birch-Murnaghan (3rd order, 6 points, max residual 0.002 mHa):

    a0 = 10.2293 Bohr      B0 = 45.0 GPa      B' = 1.63

against the unaffected references — VASP/PAW RPA@PBE (Harl 2010) 10.244 Bohr /
98 GPa, and CoQui's own ONCV RPA@PBE kp4/n500 10.228 Bohr / 101 GPa (ONCV has
no augmentation, so no V_LL block, so the bug cannot touch it).

**a0 was right to 0.015 Bohr while B0 was wrong by a factor of 2.2 and B' by a
factor of 2.5.** That is the whole reason this survived so long. a0 depends on
where the derivative crosses zero; the transpose error is smooth and slowly
varying in volume, so it shifts the curve far more than it tilts it, and the
minimum barely moves. B0 is the curvature, and the curvature is destroyed. Any
future validation of an augmentation change must look at B0/B', not a0 — an
EOS that reproduces the literature lattice constant is NOT evidence that the
two-particle terms are right.

The fit residual being 0.002 mHa is worth noting too: the corrupted data is
beautifully described by a Birch-Murnaghan form. Goodness of fit was never
going to catch this.

Tooling: `eos_exxrpa_harvest.sh` (rusty -> json; skips any run without an
`RPA energy:` line, so a job that died in the Pi stage cannot silently
contribute a truncated `Total energy`) and `eos_exxrpa_fit.py`, both in this
directory. The fitter self-tests against a synthetic BM curve and recovers
a0/B0/B' exactly.

Unit trap, recorded because it is worth ~4 Ha per point: the campaign memory's
recipe "CoQui_Total + 0.5 * E_Ewald" has that 1/2 as a **Rydberg-to-Hartree
conversion**, because that campaign read Ewald from a QE SCF. Ewald taken from
an ABINIT `.abo` is already in Hartree and takes NO factor. CoQui's own
`Total energy` line omits Ewald entirely in either case.

### 14a. `paw_isdf_tol` changed meaning (absolute -> relative)

Recorded separately because it silently reinterprets existing input files.
`build_local_isdf_compressed_by_norm` tested `d > tol^2` on the RAW q=0
Coulomb-metric norm — an ABSOLUTE threshold. `select_aug_channels_qaware` tests
`d / max_IJ d > tol^2` — RELATIVE to the largest pair.

Absolute was the wrong choice on its own terms, independently of the q=0 bias:
d carries the 4pi/Omega of the Coulomb metric, so a fixed numeric tolerance is a
different physical threshold at every cell volume. In an EOS series that means
the aug basis could change size from one volume to the next for no physical
reason — a volume-dependent basis-set artifact sitting directly on top of the
quantity being differentiated to get B0.

A given numeric `paw_isdf_tol` is therefore NOT equivalent between the two
rules, and old inputs should be re-read with that in mind. The default (1e-12)
is safe under either reading: relative-1e-12 requires a pair to be 1e-24 of the
largest in squared norm before it is dropped, i.e. effectively full rank.

## 16. Which pre-86ace47 numbers are actually invalid (regeneration triage)

"Regenerate everything" is the wrong instruction — most of the campaign is
untouched, and knowing why is the same argument that explains the nine-month
blindness.

**Criterion.** Z is Hermitian, so storing Z^T maps any quantity to its complex
conjugate that has the form of a Hermitian quadratic form

    S = sum_PQ A_P Z_PQ conj(A_Q)        (real, hence S -> S: INVARIANT)

Anything built from a single vector contracted against Z on both sides is
therefore unaffected. Quantities that contract Z between DIFFERENT vectors, or
that apply a non-linear matrix function to it, are not.

**INVARIANT — do not regenerate:**
- All diagonal ERIs (vc|cv), hence every D0/D1-style accuracy number.
- `Tr(Pi*Z) = -sum_ia (ia|ai)` — measured drift 0.003%.
- E_Hartree.
- E_X. The exchange integral (ij|ji) = int int rho_ij v conj(rho_ij) is a
  Hermitian quadratic form in rho_ij, real and non-negative, so it is invariant
  even for i != j. This is why the cross-converter E_X parity (40 uHa) and the
  ABINIT smooth-exchange agreement held all along WITH the bug present.
- Everything static: deeq, becsum, V_H/V_x route equivalence, the I1-I8
  acceptance tests, the whole fast PAW suite (bit-identical across the fix).
- The ABINIT oscillator comparison — pair densities were never touched.

**INVALID — must be regenerated:**
- **RPA correlation energies** on any PAW/USPP system. `ln|det(I - Pi*Z)|`
  applies a matrix function to Z and carried the entire 145 mHa.
- **GW**: W = Z(I - Pi*Z)^-1 inherits it, so Sigma_c and every quasiparticle
  energy from a PAW/USPP THC run are affected.
- **Off-diagonal Sigma_x(n,n') for n != n'.** Careful here — the invariance
  argument does NOT extend to it. With A_P = conj(X_nP) X_iP the second factor
  is conj(X_iQ) X_n'Q, which equals conj(A_Q) only when n = n'. So the
  exchange ENERGY (a trace over the diagonal) is invariant while the exchange
  SELF-ENERGY MATRIX is not. This is not hypothetical in CoQui: `qp_approx`
  (scf_common.hpp:153) explicitly forms off-diagonal V_corr_ab in BOTH modes
  ("qp_energy" averages V_corr_ab at e_a/e_b; "fermi" uses V_corr(w=0)), and
  scGW carries the full Sigma_skij in the primary basis. So every PAW/USPP
  quasiparticle and self-consistent-GW result is affected, not just the
  correlation energy — even though E_X itself came out fine.
- Any EOS built from the above — including all six volumes of
  `eos_conv500_coqui` and the a0/B0 fitted from them (§15).

**UNAFFECTED BY CONSTRUCTION:** all NCPP/ONCV results. There is no augmentation
block, hence no V_LL, hence nothing to transpose. That is precisely what makes
the ONCV RPA@PBE 10.228 Bohr / 101 GPa a usable reference for judging the
post-fix PAW EOS rather than a co-contaminated one.

Practical consequence: the regeneration list is "PAW/USPP RPA and GW", not "the
campaign". The static/exchange/converter validation work stands.
