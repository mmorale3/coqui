# Cross-code oscillator comparison: CoQui vs ABINIT

**Result (2026-07-28): CoQui's PAW oscillators are CORRECT, verified externally.**

Until this test, every check of CoQui's PAW pair densities compared CoQui
against a reference that shared its own machinery — `Pskna`,
`build_eta_on_rho_g_at_q_chunk`, the λ split. A wrong `Q_ij(q+G)` or a wrong
projector contraction would have moved both sides identically and the ratio
would still have read 0.9999. `paw_thc_vs_exact_eri` validates the THC
**factorization**; it cannot validate the underlying AE matrix elements.

This does. Same WFK, same PAW dataset, same q, matched by Miller index:

| pair | \|AUG\| scale | α (fitted on smooth) | smooth | AE | **AUG only** |
|---|---|---|---|---|---|
| v=0, c=5   | 0.0250 | 1 + 3.7e-09j | 3.3e-08 | 3.25e-06 | **2.14e-05** |
| v=0, c=400 | 0.4571 | 1 + 1.1e-09j | 3.8e-08 | 3.45e-05 | **1.29e-05** |

Residuals are `||CoQui − α·ABINIT|| / ||CoQui||`.

Two things make this conclusive:

- **α = 1 exactly.** The codes agree on normalization *and* phase with no
  fitted factor at all, so they are demonstrably computing the same object
  rather than two things forced into agreement.
- **The high-band case is the one that mattered.** The augmentation is 18×
  larger at c=400 than at c=5 (0.457 vs 0.025) — that is the `|P|²` growth
  the whole instability hypothesis was built on (`max|Pskna|` 3.6 → 11.1 over
  the 250→500 band window). CoQui reproduces it to six significant figures in
  exactly that regime.

This externally eliminates `Q_ij(q+G)`, the `Pskna` contraction, the structure
factor `e^{-i(q+G)·τ}`, the λ decomposition, and the augmented-oscillator
construction as a whole.

## Why the calibration is not circular

The SMOOTH oscillator is computed by both codes from the same WFK with **no
PAW involvement** — ABINIT's `rho_tw_g`, CoQui's FFT of `conj(u_v) u_c`. It is
therefore a genuine common reference, and any code-to-code difference in it can
only be convention. So: fit one complex α on the smooth part, confirm the
residual is ~1e-8 (validating the fit), then apply that **same** α to the AE
part. Smooth-agrees / AE-disagrees would have been a physics result, not a
convention artifact. `cmp_osc.py` also tries the conjugated convention and
reports that it fails on the smooth part (residual ~0.9), which is the control
showing the fit discriminates.

## ABINIT instrumentation

`src/70_gw/m_chi0.F90`, subroutine `cchi0` (the q≠0 path). Inert unless
`COQUI_OSC_DUMP=1`. Three insertions:

1. **Declarations + one-time init** after the `Pwij`/`Pwij_fft` declarations
   and at the top of the routine body: `cq_init/cq_on/cq_ik/cq_b1/cq_b2/cq_unt`
   (all `save`), `cq_env`, `rhotwg_sm(:)`. The init reads `COQUI_OSC_DUMP`,
   `COQUI_OSC_IK`, `COQUI_OSC_B1`, `COQUI_OSC_B2` and opens
   `coqui_osc_dump.dat`.
2. **Before** `call paw_rho_tw_g(...)`: copy `rhotwg` into `rhotwg_sm` —
   the smooth oscillator, before the on-site term is folded in.
3. **After** `paw_rho_tw_g`: write one `OSC` record per (ik_bz, band1, band2)
   carrying `kbz` and `kmq_bz`, then per G: `gx gy gz Re(sm) Im(sm) Re(ae)
   Im(ae)`.

`COQUI_OSC_IK = -1` dumps every k in the BZ. That matters: ABINIT's `ik_bz`
ordering need not match CoQui's, so the matching k is found **by coordinates**
from the record header, never by an assumed index. On the Si 4×4×4 mesh the
pair (band1 at Γ, band2 at k₁) is `ik_bz = 2`, with `kbz = (0.25,0,0)` and
`kmq_bz = (0,0,0)`.

Build: patch the tree, then an incremental `make` (~2 min); the binary was
copied to `bin/abinit_osc` so the working `abinit_isz` is not clobbered.
Verify the instrumentation actually linked with
`strings bin/abinit_osc | grep COQUI_OSC` — an incremental build that silently
skips the file reads as a null result, not a failure.

## CoQui side

`paw_thc_vs_exact_eri` with `COQUI_ERICHK_DUMP=<path>`, `_DUMP_V`, `_DUMP_C`.
Writes the same quantity on the `rho_g` grid with Miller indices. ABINIT band
indices are 1-based, CoQui's 0-based: ABINIT `band1=1, band2=6` is CoQui
`v=0, c=5`.

## Geometry

CoQui `k2=1` gives `q_cart = (−0.15630, 0.15630, 0.15630)`, which is exactly
`0.25 × b₁` — i.e. **q = (0.25, 0, 0) reduced**, selectable in ABINIT with
`nqptdm 1 / qptdm 0.25 0.0 0.0`. ABINIT's `rhotwg = u*_{b1,k−q} u_{b2,k}` with
k = (0.25,0,0), k−q = Γ is the same pair as CoQui's `conj(ψ_{v,Γ}) ψ_{c,k₁}`.

ABINIT's sphere is `ecuteps`-limited (531 G at 12 Ha); CoQui's `rho_g` has
40821. All 531 match by Miller index — the comparison runs on ABINIT's subset,
which covers the low-|q+G| region carrying the Coulomb weight.

## Files

- `cmp_osc.py` — the comparison (matches by Miller index, fits α on smooth,
  reports smooth / AE / augmentation-only residuals, tries both conjugations)
- `osc.abi`, `osc_hi.abi` — minimal single-q screening inputs (nband 24 / 420).
  The frequency mesh is irrelevant: `rhotwg` is built before any frequency work.

## What this does NOT establish

The oscillators and the ERI are now verified; the step from oscillators to Π,
and the RPA energy evaluation, are not. The term split
(`Tr(ΠZ)` vs `ln|det|`, §9c of the localization write-up) points there: PAW and
NC at n=500 have nearly identical `Tr(ΠZ)` (−10.138 vs −10.229, 0.89% apart)
yet E_c differs by 30%, so the discrepancy lives in `Tr[(ΠZ)²]` and higher
moments — which no oscillator-level or trace-level check constrains.
