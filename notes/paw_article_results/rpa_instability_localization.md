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

**The resolution is conditioning, not accuracy.** Two ratios from the same
data:

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

Open question, and the next measurement: NC at nbnd=500 has a comparable basis
size and does NOT blow up, so what puts PAW's Pi*Z near the singularity? One
structural candidate is that the augmentation basis is EXACTLY rank-deficient
by construction — the polarization-identity split gives eta_+ = -eta_- for
every off-diagonal (I,J) pair, so Z_LL is singular with a null space of
dimension ~(nlambda - n_pairs) per atom, and Pi's numerical content in those
directions is unprotected. `rpa_cond/` runs PAW n=250, PAW n=500 and NC n=500
through the same solver with the new diagnostic to test it.

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
