# GW Reference Benchmarks for PAW+ISDF-THC Validation

**Purpose:** Reference band gaps from established GW codes to target when validating the CoQui
PAW+ISDF-THC implementation. Systems ordered by difficulty: Si (easy), ZnO (hard).

**Last updated:** 2026-04-30

---

## Table of Contents
1. [Silicon (Si) — diamond structure](#1-silicon-si--diamond-structure)
2. [ZnO — wurtzite](#2-zno--wurtzite)
3. [Secondary benchmarks](#3-secondary-benchmarks-mgo-lif-gaas-tio2)
4. [Cross-code precision benchmarks (2024)](#4-cross-code-precision-benchmarks-2024)
5. [ZnO convergence saga](#5-zno-convergence-saga)
6. [Key references](#6-key-references)

---

## 1. Silicon (Si) — diamond structure

**Structure:** Fd-3m, a = 5.431 Å. Gap is indirect Γ→Δ (≈0.85×X). Experimental indirect gap: **1.17 eV** (0 K, no ZPR).

| Level | Gap (eV) | Code | PP type | k-mesh | N_bands | Freq. method | DOI / source |
|-------|----------|------|---------|--------|---------|--------------|-------------|
| LDA (KS) | 0.51 | VASP | PAW | — | — | — | textbook |
| G0W0@LDA | 0.85 | VASP | PAW (all-e) | 4×4×4 | ~100 | PP (Godby-Needs) | Shishkin & Kresse, PRB 75, 235102 (2007) |
| GW0@LDA | 1.22 | VASP | PAW (all-e) | 4×4×4 | ~100 | PP | Shishkin & Kresse, PRB 75, 235102 (2007) |
| scGW@LDA | 1.41 | VASP | PAW (all-e) | 4×4×4 | ~100 | PP | Shishkin & Kresse, PRB 75, 235102 (2007) |
| G0W0@LDA | 1.094 | VASP | PAW | 10×10×10 | large | CD | VASP wiki (mat3ra tutorial) |
| G0W0@PBEsol | 1.257 | ABINIT | ONCV (NC-PP) | 8×8×8 | 350–2000 | CD+AC | arXiv:2411.19701 (Borlido et al. 2024) |
| G0W0@PBEsol | 1.145 | exciting | LAPW+lo | 8×8×8 | extrap. | 32 imag. freq. | arXiv:2411.19701 (Borlido et al. 2024) |
| G0W0@PBEsol | 1.147 | FHI-aims | NAO (all-e) | 8×8×8 | STO ℓ≤7 | 60 freq. (Padé) | arXiv:2411.19701 (Borlido et al. 2024) |
| G0W0@PBEsol | 1.152 | GPAW | PW (NC-PP) | 8×8×8 | 3-pt extrap. | real-freq. | arXiv:2411.19701 (Borlido et al. 2024) |
| G0W0@PBE (FHI-aims) | 1.09 | FHI-aims | NAO (all-e) | 8×8×8 | intermediate_gw | 60 freq. | FHI-aims club tutorial (lit. ref. 1.12 eV) |
| **Experiment** | **1.17** | — | — | — | — | — | standard reference (0 K) |

**Notes on Si:**
- G0W0@LDA (Shishkin/Kresse 2007) gives 0.85 eV — significantly too small (~0.32 eV below expt).
- GW0@LDA (eigenvalue self-consistency in G only) closes most of the gap: 1.22 eV.
- The 2024 multi-code benchmark (arXiv:2411.19701) with PBEsol starting point gives 1.145–1.257 eV across
  four codes; ABINIT outlier at 1.257 eV suggests residual ONCV/ecuteps sensitivity.
- All-electron codes (FHI-aims, exciting) agree at ~1.15 eV, within 0.10 eV of experiment.
- **Target for CoQui validation:** G0W0@LDA indirect gap ~0.85–1.1 eV (depending on starting point);
  G0W0@PBE/PBEsol ~1.1–1.2 eV. Aim for < 0.1 eV of established codes.

---

## 2. ZnO — wurtzite

**Structure:** P6₃mc, a = 3.250 Å, c = 5.207 Å. Gap is **direct** Γ→Γ. Experimental: **3.44 eV** (optical, 4 K).
Zn 3d at ~−7 eV in experiment. LDA gap ~0.7–0.8 eV.

| Level | Gap (eV) | Code | PP type | Zn 3d | k-mesh | N_bands | Freq. method | DOI / source |
|-------|----------|------|---------|--------|--------|---------|--------------|-------------|
| LDA (KS) | 0.76 | ABINIT | ONCV | valence | 8×8×5 | — | — | arXiv:2411.19701 (2024) |
| LDA (KS) | 0.71 | exciting | LAPW | valence | 8×8×5 | — | — | arXiv:2411.19701 (2024) |
| G0W0@LDA (early) | ~0.8 | VASP | PAW | **core** | 4×4×4 | ~200 | PP | Shishkin & Kresse PRB 75, 235102 (2007) — incomplete convergence |
| GW0@LDA (early) | ~2.97 | VASP | PAW | core | 4×4×4 | ~200 | PP | Shishkin & Kresse PRB 75, 235102 (2007) |
| G0W0@LDA (converged) | ~2.5 | BerkeleyGW | NC-PP | **valence** | 4×4×3 | ~3000+extrap. | PP (GPP) | Shih et al. PRL 105, 146401 (2010) |
| G0W0@LDA (converged) | ~3.0–3.2 | SPEX/FLEUR | FLAPW (all-e) | valence | 4×4×4 | →∞ extrap. | full-freq. | Friedrich, Müller, Blügel PRB 83, 081101(R) (2011); erratum PRB 84, 039906 |
| G0W0@LDA (PP models) | 2.1–2.7 | ABINIT | PAW | valence | 6×6×4 | 800 | 4 PP models | Stankovski et al. PRB 84, 241201(R) (2011) |
| G0W0@LDA (CD, converged) | ~2.4 | ABINIT | PAW | valence | 6×6×4 | 800 | contour deform. | Stankovski et al. PRB 84, 241201(R) (2011) |
| G0W0@LDA+U | ~3.2 | VASP | PAW | valence | 6×6×4 | large | PP | Lany & Zunger, PRB 81, 113201 (2010) |
| G0W0@PBEsol | 2.613 | ABINIT | ONCV | valence | 8×8×5 | 350–2000 | CD+AC | arXiv:2411.19701 (Borlido et al. 2024) |
| G0W0@PBEsol | 2.878 | exciting | LAPW (all-e) | valence | 8×8×5 | →∞ extrap. | 32 imag. freq. | arXiv:2411.19701 (Borlido et al. 2024) |
| G0W0@PBEsol | 2.758 | FHI-aims | NAO (all-e) | valence | 8×8×5 | STO | 60 freq. | arXiv:2411.19701 (Borlido et al. 2024) |
| G0W0@PBEsol | 2.544 | GPAW | PW (NC-PP) | valence | 8×8×5 | 3-pt extrap. | real-freq. | arXiv:2411.19701 (Borlido et al. 2024) |
| **Experiment (optical)** | **3.44** | — | — | — | — | — | — | standard |
| **Experiment (transport)** | **3.37** | — | — | — | — | — | — | standard |

**Notes on ZnO — the convergence saga (see also Section 5):**
- Early VASP/PAW calculations with Zn 3d as **core** and ~200 bands gave G0W0 ~0.8 eV — severe underestimate.
- The fix requires two things simultaneously: (1) treat Zn 3s, 3p, 3d as **valence**; (2) use >> 1000 bands
  and extrapolate (or apply basis-set corrections). Either fix alone is insufficient.
- Shih et al. (PRL 2010, BerkeleyGW) identified "false convergence" from small ecuteps: the Coulomb-hole
  self-energy Σ_COH shows non-uniform convergence — VBM and CBM converge at different rates with N_bands.
  With ~3000 bands + extrapolation + proper semicore: G0W0 ≈ 2.5 eV.
- Friedrich et al. (PRB 2011, SPEX/FLAPW) showed even 3000 bands are not converged in FLAPW; after
  hyperbolical extrapolation to N→∞ and linearization-error correction: ~3.0–3.2 eV. An erratum (PRB 84,
  039906, 2011) corrected the extrapolated value from 2.99 to 2.83 eV.
- Stankovski et al. (PRB 2011, ABINIT) compared 4 plasmon-pole models + contour deformation: PP models
  using f-sum rule are unreliable for ZnO (semicore d resonance causes problems); CD gives ~2.4 eV with
  800 bands.
- The 2024 multi-code benchmark (arXiv:2411.19701) shows the largest spread of any system: 2.54–2.88 eV
  (334 meV spread), consistent with known sensitivity. None reach 3.44 eV expt at G0W0@PBEsol level.
- **Bottom line:** converged G0W0 with semicore-in-valence gives ~2.4–2.9 eV depending on code/basis/starting
  point. Reaching experiment requires vertex corrections, self-consistency (GW0/QSGW), or hybrid starting points.
- **Target for CoQui validation:** G0W0@LDA with Zn 3d valence: aim for 2.4–2.9 eV range. If PAW
  augmentation is incomplete, expect underestimate. Compare against arXiv:2411.19701 code-average ~2.70 eV.

---

## 3. Secondary Benchmarks: MgO, LiF, GaAs, TiO2

### MgO (rock-salt, Fm-3m)
Experimental gap: **7.83 eV** (direct Γ→Γ). LDA ~4.5 eV.

| Level | Gap (eV) | Code | PP/basis | Source |
|-------|----------|------|----------|--------|
| G0W0@LDA | ~6.2–6.5 | VASP | PAW | Shishkin & Kresse PRB 75, 235102 (2007) |
| GW0@LDA | ~7.2–7.5 | VASP | PAW | Shishkin & Kresse PRB 75, 235102 (2007) |
| scGW@LDA | ~7.8 | VASP | PAW | Shishkin & Kresse PRB 75, 235102 (2007) |
| **Experiment** | **7.83** | — | — | — |

### LiF (rock-salt, Fm-3m)
Experimental gap: **14.2 eV** (direct). LDA ~9.2 eV.

| Level | Gap (eV) | Code | PP/basis | Source |
|-------|----------|------|----------|--------|
| G0W0@LDA | ~12.0–12.5 | VASP | PAW | Shishkin & Kresse PRB 75, 235102 (2007) |
| GW0@LDA | ~13.5 | VASP | PAW | Shishkin & Kresse PRB 75, 235102 (2007) |
| scGW@LDA | ~14.8 | VASP | PAW | Shishkin & Kresse PRB 75, 235102 (2007) — overshoots |
| **Experiment** | **14.2** | — | — | — |

### GaAs (zincblende, F-43m)
Experimental direct gap (Γ→Γ): **1.52 eV** (0 K). LDA ~0.2–0.5 eV. Ga 3d at ~−18 eV.

| Level | Gap (eV) | Code | PP/basis | Ga 3d | Source |
|-------|----------|------|----------|--------|--------|
| G0W0@LDA | ~1.1–1.3 | VASP | PAW | core | Shishkin & Kresse PRB 75, 235102 (2007) |
| GW0@LDA | ~1.5 | VASP | PAW | core | Shishkin & Kresse PRB 75, 235102 (2007) |
| G0W0@LDA | ~1.3–1.5 | BerkeleyGW | NC-PP | valence | Rangel et al. CPC 255, 107242 (2020) |
| **Experiment** | **1.52** | — | — | — | — |

**Note:** Ga 3d is deeper than Zn 3d (~−18 vs ~−7 eV) so treating as core is less problematic,
but convergence with bands is still slower than Si.

### TiO2 (rutile, P4₂/mnm)
Experimental optical gap: **3.03 eV** (direct). LDA ~1.8 eV.

| Level | Gap (eV) | Code | PP/basis | Source |
|-------|----------|------|----------|--------|
| G0W0@PBEsol | 3.333 | ABINIT | ONCV | arXiv:2411.19701 (2024) |
| G0W0@PBEsol | 3.321 | exciting | LAPW | arXiv:2411.19701 (2024) |
| G0W0@PBEsol | 3.546 | FHI-aims | NAO | arXiv:2411.19701 (2024) |
| G0W0@PBEsol | 3.281 | GPAW | PW | arXiv:2411.19701 (2024) |
| **Experiment** | **3.03** | — | — | — |

---

## 4. Cross-Code Precision Benchmarks (2024)

**Source:** Borlido et al., "Precision benchmarks for solids: G0W0 calculations with different basis sets,"
arXiv:2411.19701 (2024); published in Comput. Mater. Sci. (2025). DOI: 10.1016/j.commatsci.2024.113495

**DFT starting point:** PBEsol for all codes. All use full-frequency methods (no plasmon-pole).
**Zn valence:** 20 electrons (includes 3d) in ABINIT ONCV; all-electron in exciting/FHI-aims.

| Material | Abinit | exciting | FHI-aims | GPAW | Spread | Expt |
|----------|--------|----------|----------|------|--------|------|
| Si (Γ→Δ, indirect) | 1.257 | 1.145 | 1.147 | 1.152 | 112 meV | 1.17 eV |
| ZnO (Γ→Γ, direct) | 2.613 | 2.878 | 2.758 | 2.544 | 334 meV | 3.44 eV |
| TiO2 rutile (Γ→Γ) | 3.333 | 3.321 | 3.546 | 3.281 | 265 meV | 3.03 eV |

**Key observations:**
- Si: three codes (exciting, FHI-aims, GPAW) cluster at 1.15 eV; ABINIT ~0.11 eV higher — residual
  ecuteps or AC sensitivity.
- ZnO: largest code-to-code spread (334 meV). ABINIT lowest, exciting highest. No code reaches expt.
- TiO2: FHI-aims outlier (+0.2 eV vs others); otherwise codes agree within ~50 meV.
- All-electron codes (exciting, FHI-aims) show < 0.1 eV agreement with each other for Si; larger
  discrepancy for ZnO, confirming the intrinsic difficulty of that system.

**Convergence details (from paper):**
| Code | ecuteps / equiv. | N_bands | Freq. grid | Conv. criterion |
|------|-----------------|---------|-----------|-----------------|
| ABINIT | 2–12 Ha (χ cutoff) | 350–2000 | 50-pt linear + AC | 20 meV |
| exciting | LAPW+lo basis | →∞ extrap. | 32 imag. freq. | extrapolated |
| FHI-aims | STO ℓ≤7 + Padé | NAO complete | 60 freq. (16-param Padé) | NAO basis converged |
| GPAW | 18–20 Ha (3-pt) | →∞ poly extrap. | real-freq. full | polynomial fit |

---

## 5. ZnO Convergence Saga — Chronological

This table documents how reported G0W0 ZnO gaps evolved as methodological issues were resolved.
The "canonical hard benchmark" label comes from this history.

| Year | Gap (eV) | Code | Critical detail | Reference |
|------|----------|------|-----------------|-----------|
| 2007 | ~0.8 | VASP/PAW | Zn 3d as **core**, ~200 bands, PP model | Shishkin & Kresse PRB 75, 235102 |
| 2007 | ~2.97 (GW0) | VASP/PAW | Same setup, eigenvalue self-consistency in G | Shishkin & Kresse PRB 75, 235102 |
| 2010 | ~2.5 | BerkeleyGW/NC-PP | Zn 3d **valence**, ~3000 bands + extrap., GPP | Shih et al. PRL 105, 146401 |
| 2011 | 2.99→2.83 | SPEX/FLAPW (all-e) | All-electron, linearization-error correction, N→∞ extrap. | Friedrich et al. PRB 83, 081101(R); erratum PRB 84, 039906 |
| 2011 | 2.1–2.7 (PP) / ~2.4 (CD) | ABINIT/PAW | Zn 3d valence; f-sum PP unreliable; CD more reliable | Stankovski et al. PRB 84, 241201(R) |
| 2024 | 2.54–2.88 | ABINIT/exciting/FHI-aims/GPAW | PBEsol start, semicore valence, full-freq., 4 codes | arXiv:2411.19701 |

**What drives the spread:**
1. **Semicore treatment:** Zn 3s, 3p, 3d must be valence. Core treatment → ~0.8 eV (wrong by ~2.6 eV).
2. **Number of bands / basis completeness:** Σ_COH converges very slowly. VBM and CBM converge at
   different rates ("nonuniform convergence"), so premature truncation biases the gap. Need ~1000–3000
   bands or basis-set extrapolation.
3. **Kinetic-energy cutoff for χ (ecuteps/ENCUTGW):** Too small → false convergence plateau. Must be
   pushed to ~30–50 Ha (ABINIT) or equivalent.
4. **Frequency integration:** f-sum-rule plasmon-pole models fail for ZnO (shallow d-resonance
   pollutes the f-sum). Contour deformation or full real-frequency integration required.
5. **DFT starting point:** LDA puts Zn 3d too high (~−5 eV vs −7 eV expt), underscreening affects G
   and W. Hybrid or LDA+U starting points improve but are code-dependent.
6. **Norm-conserving vs PAW:** Standard PAW partial waves are incomplete for high-energy unoccupied
   states; Klimes et al. (PRB 90, 075125, 2014) showed NC-PAW partial waves or basis-set corrections
   are needed for predictive results.

---

## 6. Key References

| Citation | DOI / arXiv | Annotation |
|----------|-------------|------------|
| Shishkin & Kresse, PRB **75**, 235102 (2007) | 10.1103/PhysRevB.75.235102 | Canonical VASP PAW G0W0/GW0/scGW benchmark: Si, ZnO, MgO, LiF, GaAs, GaN + others. **First entry point.** |
| Shishkin & Kresse, PRB **74**, 035101 (2006) | 10.1103/PhysRevB.74.035101 | VASP implementation paper: freq.-dependent GW within PAW. |
| Shih, Xue, Zhang, Cohen, Louie, PRL **105**, 146401 (2010) | 10.1103/PhysRevLett.105.146401 | BerkeleyGW; resolves ZnO underestimate: semicore + ~3000 bands + GPP → 2.5 eV. |
| Friedrich, Müller, Blügel, PRB **83**, 081101(R) (2011); erratum PRB **84**, 039906 (2011) | 10.1103/PhysRevB.83.081101 | SPEX/FLAPW all-electron; ZnO as "extreme case"; linearization-error correction; extrapolated gap 2.83 eV. |
| Stankovski et al., PRB **84**, 241201(R) (2011) | 10.1103/PhysRevB.84.241201 | ABINIT; 4 PP models vs contour-deformation for ZnO; f-sum PP unreliable; confirms CD ~2.4 eV. |
| Klimeš, Kaltak, Kresse, PRB **90**, 075125 (2014) | 10.1103/PhysRevB.90.075125 | VASP; shows standard PAW converges to wrong value; NC partial waves + finite-basis correction fix it. Large NC-PAW dataset for semiconductors/insulators. |
| van Setten, Giantomassi, Gonze, Rignanese, Hautier, PRB **96**, 155207 (2017) | 10.1103/PhysRevB.96.155207 | ABINIT high-throughput GW automation; ONCV pseudopotentials; convergence criterion 0.02 eV. |
| Rangel et al., Comput. Phys. Commun. **255**, 107242 (2020) | 10.1016/j.cpc.2020.107242 | BerkeleyGW + ABINIT + Yambo cross-code reproducibility; Si, GaAs, LiF, BN, AlP, ZnS, GaN; ZnO spread > 1 eV traced to Coulomb divergence and freq. scheme differences. |
| Borlido et al., Comput. Mater. Sci. (2025); preprint arXiv:**2411.19701** | 10.1016/j.commatsci.2024.113495 | 4-code precision benchmark (ABINIT/exciting/FHI-aims/GPAW); Si, ZnO, TiO2; PBEsol start; full-frequency; **most current multi-code reference.** |
| Govoni & Galli, JCTC **11**, 2680 (2015) | 10.1021/ct500958p | WEST code; large-scale GW via spectral decomposition; no explicit virtual-state sum; scalable to large systems. |
| Jiang & Blaha, PRB **93**, 115203 (2016) | 10.1103/PhysRevB.93.115203 | GAP2 / exciting LAPW+HLO GW; improves linearization error treatment in FLAPW. |

---

## Notes for CoQui PAW+ISDF-THC Validation

1. **Start with Si** (easiest): Target G0W0@LDA indirect gap ~0.85–1.1 eV. The 2024 benchmark
   (PBEsol start) gives 1.145–1.257 eV across codes. Compare against VASP PAW result 1.094 eV
   (mat3ra/VASP wiki).

2. **ZnO diagnostic:** Run with Zn 3d **in valence** only. With insufficient bands or small ecuteps,
   expect values well below 2.5 eV — this is the known false-convergence trap. Target the 2.54–2.88 eV
   window from arXiv:2411.19701.

3. **PAW augmentation check:** The PAW augmentation (the whole point of CoQui's PAW+THC work) affects
   both the Coulomb matrix elements and the density matrix. For ZnO, any error in the Zn 3d projector
   overlaps will manifest as an anomalous gap shift. The Shih et al. (2010) and Stankovski et al. (2011)
   papers are the most relevant for diagnosing augmentation errors.

4. **Frequency integration:** For ZnO, avoid f-sum-rule plasmon-pole models (Stankovski 2011). Use
   contour deformation or full imaginary-axis integration.

5. **Klimeš 2014 warning:** Standard PAW partial waves converge to the wrong QP energy due to
   incomplete partial-wave basis inside augmentation spheres. The ISDF-THC approach constructs the
   two-electron integrals; verify that the PAW augmentation terms are complete enough for high-energy
   virtual states.
