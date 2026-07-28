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

## 4. Recommended next step (code-level, not another sweep)

Every exposed toml knob has been swept. Compare CoQui's assembled augmented pair
density ρ_mn(G) = ρ̃_mn(G) + Σ_ij Q_ij(G) P*_mi P_nj against the same quantity
built directly (non-factorized) from the same h5, for a few (m,n) at large band
index. The THC form is algebraically equivalent only if the factorization is
exact; the open question is whether the joint smooth+aug basis can represent the
near-cancellation at high band index, where each block is ~400× the residual.

## Reproduction

Generators/harvesters in this directory; runs on rusty under `~/ceph/CoQui/abinit/`:
`rpa_eos_jthd/` (ABINIT), `rpa_eos_jthd_coqui_nb/` (CoQui n=100/250),
`rpa_localize_jthd/` + `rpa_localize2_jthd/` (localization).

Not available: the `pawcross 1` probe (on-site completeness in ABINIT's
oscillator strengths) aborts with heap corruption in this ABINIT 10.6.7 build at
both 64 and 16 ranks.
