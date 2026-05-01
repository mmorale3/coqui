# RPA Reference Benchmarks for PAW+ISDF-THC Validation

**Purpose:** Reference values from established codes/papers to compare against when validating the
CoQui PAW+ISDF-THC RPA implementation. Companion to `paw_gw_reference_benchmarks.md`.

**Level of theory:** RPA@PBE (post-PBE RPA correlation + PBE exchange) unless noted.
"RPA" without qualifier in older VASP literature means RPA correlation energy on top of PBE
orbitals; total energy = E_x(EXX@PBE) + E_c(RPA@PBE).

**Cohesive energy sign convention:** positive = bound (energy released on forming crystal from
free atoms). Some papers report atomization energy (same sign); others report negative formation
energy. All values below are converted to eV/atom, positive = bound.

**Last updated:** 2026-04-30

---

## Table of Contents
1. [Silicon (Si)](#1-silicon-si)
2. [Diamond (C)](#2-diamond-c)
3. [MgO](#3-mgo)
4. [LiF](#4-lif)
5. [NaCl](#5-nacl)
6. [ZnO](#6-zno)
7. [hBN and graphite](#7-hbn-and-graphite)
8. [Noble-gas solids](#8-noble-gas-solids)
9. [Metals (alkali, alkaline-earth, transition)](#9-metals)
10. [Lattice constants and bulk moduli summary](#10-lattice-constants-and-bulk-moduli-summary)
11. [Convergence parameters across codes](#11-convergence-parameters-across-codes)
12. [What drives spread across codes](#12-what-drives-spread-across-codes)
13. [Key references](#13-key-references)
14. [Notes for CoQui validation](#14-notes-for-coqui-validation)

---

## 1. Silicon (Si)

**Structure:** diamond cubic, Fd-3m. Experimental cohesive energy: **4.62 eV/atom** (0 K,
zero-point corrected); raw sublimation ~4.68 eV/atom. Experimental lattice constant: **5.431 Å**.
Experimental bulk modulus: **99 GPa**.

### 1a. Cohesive energy

| Method | E_coh (eV/atom) | Code | PP/basis | k-mesh | Notes | Source |
|--------|----------------|------|----------|--------|-------|--------|
| PBE | 4.55 | VASP | PAW | 8×8×8 | reference DFT | Harl et al. PRB 81, 115126 (2010) |
| RPA@PBE | ~4.6 | VASP | PAW | 8×8×8 | 16 imag. freq.; basis extrap. | Harl et al. PRB 81, 115126 (2010) |
| RPA@LDA | 4.32 | GPAW | PAW (PW) | 4×4×4 | cutoff 164 eV + E^{-3/2} extrap. | Olsen & Thygesen, PRB 87, 075111 (2013); GPAW tutorial |
| EXX@PBE | 2.82 | GPAW | PAW (PW) | 4×4×4 | exchange only, no correlation | GPAW tutorial |
| FCIQMC | — | — | PAW (VASP) | Γ-supercell | exact within basis; compared to RPA | Booth et al., Nature 493, 365 (2013) |
| **Experiment** | **4.62** | — | — | — | 0 K, ZPE-corrected | standard |

**Notes:**
- The Harl/Kresse 2010 VASP RPA value is ~4.6 eV/atom (close to experiment), a significant
  improvement over LDA (underbinds) but roughly similar MAE to PBE for covalent solids.
- GPAW RPA@LDA gives 4.32 eV — below experiment by ~0.3 eV. RPA@LDA systematically
  underbinds covalent solids due to delocalization error in the LDA starting point.
- Booth et al. (2013) showed CCSD(T)/FCIQMC substantially improves over RPA for C, BN, LiH,
  AlP; Si was studied but exact RPA-vs-FCIQMC difference not publicly tabulated in abstract.

### 1b. Lattice constant and bulk modulus

| Method | a (Å) | B (GPa) | Code | Source |
|--------|--------|---------|------|--------|
| PBE | 5.468 | 90 | VASP/PAW | Harl et al. PRB 81, 115126 (2010) |
| RPA@PBE | 5.421 | ~98 | VASP/PAW | Harl et al. PRB 81, 115126 (2010) |
| RPA | 5.414 | — | QE/ONCV | Pitts et al., PRB 112, 085137 (2025) |
| PBE | 5.479 | — | QE/ONCV | Pitts et al., PRB 112, 085137 (2025) |
| **Experiment** | **5.431** | **99** | — | standard (0 K) |

**Key result:** RPA@PBE lattice constant (5.421 Å, VASP PAW) agrees with experiment to < 0.01 Å.
PBE overestimates by ~0.04 Å (standard GGA softening).

---

## 2. Diamond (C)

**Structure:** diamond cubic, Fd-3m. Experimental cohesive energy: **7.37 eV/atom** (0 K,
ZPE-corrected); raw ~7.55 eV/atom. Experimental lattice constant: **3.553 Å**. Bulk modulus:
**443 GPa**.

### 2a. Cohesive energy

| Method | E_coh (eV/atom) | Code | PP/basis | Notes | Source |
|--------|----------------|------|----------|-------|--------|
| PBE | ~7.7 | VASP | PAW | slightly overbinds | Harl et al. PRB 81, 115126 (2010) |
| RPA@PBE | ~7.4–7.5 | VASP | PAW | 8×8×8; 16 imag. freq. | Harl et al. PRB 81, 115126 (2010) |
| CCSD(T) | ~7.37 | VASP | PAW (MP2/CC) | Γ-point supercell | Booth et al., Nature 493, 365 (2013) |
| **Experiment** | **7.37** | — | — | ZPE-corrected | standard |

**Notes:**
- Booth et al. (2013) Table 1: CCSD(T) for diamond agrees with experiment to < 0.05 eV/atom.
  RPA (reported in same table) underbinds C relative to experiment more than CCSD(T).
- The RPA underbinding of covalent solids is a known systematic error (~0.1–0.3 eV/atom for
  C and Si when starting from LDA/PBE).

### 2b. Lattice constant and bulk modulus

| Method | a (Å) | B (GPa) | Code | Source |
|--------|--------|---------|------|--------|
| PBE | 3.572 | 420 | VASP/PAW | Harl et al. PRB 81, 115126 (2010) |
| RPA@PBE | 3.553 | ~443 | VASP/PAW | Harl et al. PRB 81, 115126 (2010) |
| RPA | 3.567 | — | QE/ONCV | Pitts et al., PRB 112, 085137 (2025) |
| PBE | 3.567 | — | QE/ONCV | Pitts et al., PRB 112, 085137 (2025) |
| **Experiment** | **3.553** | **443** | — | standard |

**Key result:** RPA@PBE (VASP PAW) hits the experimental C lattice constant almost exactly.
The QE/ONCV RPA (2025) gives 3.567 Å — slightly high, possible ONCV softness or missing
basis correction.

---

## 3. MgO

**Structure:** rock-salt, Fm-3m. Experimental cohesive energy: **5.20 eV/atom** (0 K,
ZPE-corrected; some tabulations give 5.16–5.21). Experimental lattice constant: **4.211 Å**.
Bulk modulus: **160 GPa**.

### 3a. Cohesive energy

| Method | E_coh (eV/atom) | Code | PP/basis | Notes | Source |
|--------|----------------|------|----------|-------|--------|
| PBE | ~5.0 | VASP | PAW | underbinds | Harl et al. PRB 81, 115126 (2010) |
| RPA@PBE | ~5.0–5.1 | VASP | PAW | 8×8×8; 16 imag. freq. | Harl et al. PRB 81, 115126 (2010) |
| RPA@PBE | ~5.1 | FHI-aims | NAO (all-e) | tier-2 basis | Ren et al., J. Mater. Sci. 47, 7447 (2012) |
| CCSD(T) | ~5.2 | VASP | PAW | Γ-supercell | Booth et al., Nature 493, 365 (2013) |
| **Experiment** | **5.20** | — | — | ZPE-corrected | standard |

### 3b. Lattice constant and bulk modulus

| Method | a (Å) | B (GPa) | Code | Source |
|--------|--------|---------|------|--------|
| PBE | 4.258 | 152 | VASP/PAW | Harl et al. PRB 81, 115126 (2010) |
| RPA@PBE | 4.196 | ~162 | VASP/PAW | Harl et al. PRB 81, 115126 (2010) |
| RPA | 4.200 | — | QE/ONCV | Pitts et al., PRB 112, 085137 (2025) |
| PBE | 4.255 | — | QE/ONCV | Pitts et al., PRB 112, 085137 (2025) |
| **Experiment** | **4.211** | **160** | — | standard |

**Notes:**
- RPA@PBE underestimates MgO lattice constant slightly (~0.01 Å below experiment in VASP PAW).
- PBE overestimates by ~0.05 Å — typical GGA behavior for ionic solids.
- MgO cohesive energy is well reproduced by RPA (< 0.1 eV/atom error). Ionic solids fare
  better than covalent solids at the RPA level.

---

## 4. LiF

**Structure:** rock-salt, Fm-3m. Experimental cohesive energy: **4.46 eV/atom** (0 K,
ZPE-corrected). Experimental lattice constant: **3.972 Å**. Bulk modulus: **67 GPa**.

### 4a. Cohesive energy

| Method | E_coh (eV/atom) | Code | PP/basis | Notes | Source |
|--------|----------------|------|----------|-------|--------|
| PBE | ~4.3 | VASP | PAW | underbinds | Harl et al. PRB 81, 115126 (2010) |
| RPA@PBE | ~4.3–4.4 | VASP | PAW | 8×8×8; 16 imag. freq. | Harl et al. PRB 81, 115126 (2010) |
| RPA@PBE | ~4.4 | FHI-aims | NAO (all-e) | tier-2 | Ren et al., J. Mater. Sci. 47, 7447 (2012) |
| **Experiment** | **4.46** | — | — | ZPE-corrected | standard |

### 4b. Lattice constant and bulk modulus

| Method | a (Å) | B (GPa) | Code | Source |
|--------|--------|---------|------|--------|
| PBE | 4.073 | 62 | VASP/PAW | Harl et al. PRB 81, 115126 (2010) |
| RPA@PBE | 3.986 | ~68 | VASP/PAW | Harl et al. PRB 81, 115126 (2010) |
| RPA | 3.996 | — | QE/ONCV | Pitts et al., PRB 112, 085137 (2025) |
| PBE | 4.062 | — | QE/ONCV | Pitts et al., PRB 112, 085137 (2025) |
| **Experiment** | **3.972** | **67** | — | standard |

**Notes:**
- RPA@PBE overshoots LiF lattice constant slightly (3.986 vs 3.972 Å VASP; 3.996 QE/ONCV).
- LiF and MgO both show RPA closer to experiment than PBE for lattice constants.
- LiF is in the Booth et al. (2013) CCSD(T) test set; CCSD(T) essentially exact (< 0.02 eV
  error). RPA error is slightly larger (~0.05–0.10 eV).

---

## 5. NaCl

**Structure:** rock-salt, Fm-3m. Experimental cohesive energy: **3.34 eV/atom** (0 K).
Experimental lattice constant: **5.595 Å**. Bulk modulus: **25 GPa**.

| Method | a (Å) | B (GPa) | E_coh (eV/atom) | Code | Source |
|--------|--------|---------|----------------|------|--------|
| PBE | 5.694 | 22 | ~3.2 | VASP/PAW | Harl et al. PRB 81, 115126 (2010) |
| RPA@PBE | 5.573 | ~26 | ~3.3 | VASP/PAW | Harl et al. PRB 81, 115126 (2010) |
| **Experiment** | **5.595** | **25** | **3.34** | — | standard |

**Notes:** NaCl shows good RPA performance. Lattice constant: RPA underestimates slightly
(5.573 vs 5.595 Å); PBE overestimates (5.694 Å). The RPA compression is a known feature for
softer ionic solids.

---

## 6. ZnO

**Structure:** wurtzite, P6₃mc. Experimental cohesive energy: **3.60 eV/atom** (0 K, per Zn or O
atom in two-atom basis, i.e., ~7.2 eV/formula unit). Experimental lattice constants:
a = 3.250 Å, c = 5.207 Å, u = 0.3820. Bulk modulus: ~143 GPa.

> **Important:** Zn has shallow 3d states (~−7 eV). RPA@PBE for ZnO is significantly harder
> than for Si/MgO because: (1) PBE places Zn 3d too high (~−5 eV), causing overscreening;
> (2) convergence with number of unoccupied bands is very slow (Zn 3d→conduction couplings
> require large virtual space). The same semicore convergence problem that afflicts GW for ZnO
> (see `paw_gw_reference_benchmarks.md`) applies to RPA.

| Method | a (Å) | c (Å) | E_coh (eV/f.u.) | Code | PP | Notes | Source |
|--------|--------|--------|----------------|------|----|-------|--------|
| PBE | 3.283 | 5.304 | ~7.0 | VASP | PAW | Zn 3d valence | Harl et al. PRB 81, 115126 (2010) |
| RPA@PBE | 3.254 | 5.262 | ~7.0 | VASP | PAW | Zn 3d valence; 8×8×5 | Harl et al. PRB 81, 115126 (2010) |
| RPA@PBE | ~3.25 | ~5.21 | — | FHI-aims | NAO (all-e) | Ren et al., J. Mater. Sci. 47, 7447 (2012) |
| **Experiment** | **3.250** | **5.207** | **~7.2** | — | — | | standard |

**Notes:**
- Harl et al. (2010) show that RPA@PBE lattice constants for ZnO agree well with experiment
  (a: 3.254 vs 3.250 Å). This is better than PBE (a: 3.283 Å).
- Cohesive energy for ZnO at RPA@PBE is not prominently tabulated in publicly accessible
  portions of Harl 2010; the lattice constant convergence is the main result.
- Convergence is significantly harder than Si: need > 400 bands and large ENCUTGW in VASP,
  or all-electron basis. With PAW and insufficient augmentation, expect errors.
- No reliable open-access ZnO RPA@PBE cohesive energy table found; treat the ~7.0 eV/f.u.
  value as an order-of-magnitude estimate. FHI-aims all-electron is the more reliable source.

---

## 7. hBN and Graphite

### 7a. Hexagonal boron nitride (hBN)

**Structure:** layered hexagonal. In-plane B–N covalent; interlayer van der Waals.
Experimental in-plane a ≈ 2.504 Å; interlayer c ≈ 6.656 Å.

| Method | a (Å) | E_bind (meV/atom) | Code | Source |
|--------|--------|-------------------|------|--------|
| PBE | — | ~0 (no binding) | — | standard |
| RPA@PBE | ~2.50 | ~30–40 | VASP/PAW | Harl et al. PRB 81, 115126 (2010) |
| **Experiment** | 2.504 | ~30–50 | — | exfoliation/AFM data |

**Notes:** hBN interlayer binding is a vdW benchmark. RPA captures it; PBE fails (near-zero
binding). The Pitts et al. (2025) self-consistent RPA paper includes BN in the test set.

### 7b. Graphite (interlayer binding)

**Key result:** Lebègue et al., PRL 105, 196401 (2010) — VASP/PAW RPA@PBE.

| Method | d_inter (Å) | E_bind (meV/atom) | Code | PP | k-mesh | Notes | Source |
|--------|-------------|-------------------|------|----|--------|-------|--------|
| PBE | — | ~2 (near zero) | VASP | PAW | — | no vdW | standard |
| RPA@PBE | 3.34 | **48** | VASP | PAW | dense | asymptotic C₃/d³ correct | Lebègue et al., PRL 105, 196401 (2010) |
| vdW-DF | — | ~35–50 | — | — | — | depends on variant | various |
| QMC | — | ~56 | — | — | — | may include higher-order | Spanu et al., PRL 2009 |
| **Experiment** | **3.354** | **35–52** | — | — | — | spread from exfoliation/torsion balance | standard |

**Notes (Lebègue 2010):**
- This is the landmark RPA-for-graphite paper. Uses VASP/PAW, RPA@PBE.
- 48 meV/atom interlayer binding energy agrees well with experiments (~35–52 meV/atom).
- The RPA captures the correct ~1/d³ (van der Waals) asymptotics; PBE gives essentially
  zero interlayer binding, showing the necessity of RPA (or vdW DFT) for layered materials.
- Convergence is demanding: dense in-plane k-mesh, many bands to describe interlayer
  correlation. The paper reports careful convergence with respect to vacuum spacing, k-mesh,
  and number of bands.
- The DOI is 10.1103/PhysRevLett.105.196401 (not PRL 103 which is a different paper by
  Zacharia et al. on "Nature and strength of interlayer binding").

---

## 8. Noble-Gas Solids

**Source:** Harl & Kresse, PRB 77, 045136 (2008) — the foundational RPA-for-solids paper.
Systems: Ne, Ar, Kr (fcc crystal). VASP/PAW, RPA@LDA (and RPA@PBE).

| System | Method | a (Å) | E_coh (meV/atom) | Code | Notes | Source |
|--------|--------|--------|-----------------|------|-------|--------|
| Ne | PBE | — | ~0 (unbound) | VASP/PAW | no vdW | Harl & Kresse PRB 77, 045136 (2008) |
| Ne | RPA@LDA | ~4.46 | ~27 | VASP/PAW | correct binding | Harl & Kresse PRB 77, 045136 (2008) |
| Ar | RPA@LDA | ~5.26 | ~88 | VASP/PAW | correct vdW | Harl & Kresse PRB 77, 045136 (2008) |
| Kr | RPA@LDA | ~5.64 | ~120 | VASP/PAW | correct trend | Harl & Kresse PRB 77, 045136 (2008) |
| Ne | **Experiment** | 4.430 | 27 | — | | Kittel |
| Ar | **Experiment** | 5.256 | 88 | — | | Kittel |
| Kr | **Experiment** | 5.646 | 116 | — | | Kittel |

**Notes:**
- This paper established that RPA (via ACFDT) correctly describes van der Waals cohesion in
  noble-gas solids — something PBE completely fails at (zero or near-zero binding).
- Agreement with experiment is within ~5% for Ne and Ar cohesive energies and lattice constants.
- Convergence requires careful k-mesh (8×8×8 minimum), large NBANDS (~300), and
  basis-set extrapolation E^{-3/2}. NOMEGA = 16 imaginary-frequency points.
- This work validated the VASP ACFDT-RPA implementation and established it as the reference.
- DOI: 10.1103/PhysRevB.77.045136

---

## 9. Metals

**Source:** Schimka et al., Nat. Mater. 9, 741 (2010) — cohesive energies and surface energies.
Also: PRB 87, 214102 (2013) for alkali/alkaline-earth/transition metals.

| System | Method | E_coh (eV/atom) | Code | Notes | Source |
|--------|--------|----------------|------|-------|--------|
| Cu | PBE | 3.49 | VASP/PAW | underbinds | Schimka et al. Nat. Mater. 9, 741 (2010) |
| Cu | RPA@PBE | ~3.5 | VASP/PAW | similar to PBE | Schimka et al. Nat. Mater. 9, 741 (2010) |
| Fe | PBE | 4.28 | VASP/PAW | — | Schimka et al. Nat. Mater. 9, 741 (2010) |
| Fe | RPA@PBE | ~4.3 | VASP/PAW | — | Schimka et al. Nat. Mater. 9, 741 (2010) |
| Na | RPA@PBE | ~1.1 | VASP/PAW | soft metal | PRB 87, 214102 (2013) |
| Mg | RPA@PBE | ~1.5 | VASP/PAW | hcp | PRB 87, 214102 (2013) |
| Al | RPA@PBE | ~3.4 | VASP/PAW | fcc | PRB 87, 214102 (2013) |

**Notes:**
- Schimka et al. (2010) is primarily about surface and adsorption energies (Nat. Mater. "Accurate
  surface and adsorption energies from many-body perturbation theory"). The main finding is that
  RPA@PBE significantly improves surface energies and CO adsorption on Rh(111) over PBE.
- For bulk metals, RPA@PBE cohesive energies are comparable in accuracy to PBE — no dramatic
  improvement. The gain is primarily in surface/interface energetics.
- PRB 87, 214102 (2013) by Schimka et al. provides the systematic metal cohesive energy and
  lattice constant table (alkali, alkaline-earth, transition metals). Values are available in
  that paper's Tables II–IV.
- DOI Nat. Mater.: 10.1038/nmat2806; DOI PRB metals: 10.1103/PhysRevB.87.214102

---

## 10. Lattice Constants and Bulk Moduli Summary

Compiled from Harl, Schimka, Kresse PRB 81, 115126 (2010) [VASP/PAW, RPA@PBE] and
Pitts, Contant, Hellgren PRB 112, 085137 (2025) [QE/ONCV, RPA@PBE], with experimental
references.

| Material | a_PBE (Å) | a_RPA (Å) | a_Expt (Å) | B_PBE (GPa) | B_RPA (GPa) | B_Expt (GPa) | Source(s) |
|----------|-----------|-----------|------------|------------|------------|--------------|-----------|
| C (diamond) | 3.572 | 3.553 | **3.553** | 420 | ~443 | 443 | Harl 2010; Pitts 2025 |
| Si | 5.468 | 5.421 | **5.431** | 90 | ~98 | 99 | Harl 2010; Pitts 2025 |
| BN (zinc-blende) | 3.626 | 3.604 | **3.592** | — | — | — | Harl 2010 |
| MgO | 4.258 | 4.196 | **4.211** | 152 | ~162 | 160 | Harl 2010; Pitts 2025 |
| LiF | 4.073 | 3.986 | **3.972** | 62 | ~68 | 67 | Harl 2010; Pitts 2025 |
| NaCl | 5.694 | 5.573 | **5.595** | 22 | ~26 | 25 | Harl 2010 |
| ZnO (wurtzite a) | 3.283 | 3.254 | **3.250** | — | — | 143 | Harl 2010 |
| ZnO (wurtzite c) | 5.304 | 5.262 | **5.207** | — | — | — | Harl 2010 |

**Statistical summary (Harl 2010, 20 solids):**
- PBE: MAE lattice constant ~0.05 Å (systematic overestimate)
- RPA@PBE: MAE lattice constant ~0.02 Å (mixed over/under)
- PBE: MAE bulk modulus ~9 GPa
- RPA@PBE: MAE bulk modulus ~5 GPa
- RPA@PBE is a clear improvement over PBE for structural properties.

---

## 11. Convergence Parameters Across Codes

### VASP/PAW (Harl & Kresse 2008, 2010; Kaltak et al. 2014)

| Parameter | Recommended value | Notes |
|-----------|------------------|-------|
| ALGO = ACFDT | — | activates RPA |
| NOMEGA | 12–16 | imaginary-frequency points; 8 sufficient for large-gap insulators, 16 for semiconductors/metals |
| ENCUTGW | = ENCUT (default 2/3 ENCUT) | must be converged; increase ENCUT and repeat |
| NBANDS | ~3–5 × N_occ | must be extrapolated; E(N)→∞ via E_cut^{-3/2} or N_bands^{-3/2} law |
| k-mesh | 8×8×8 (Si, MgO); 4×4×4 minimum | dense mesh critical for correlation energy |
| Basis extrapolation | E_RPA(∞) = E_RPA(E_cut) − A·E_cut^{-3/2} | standard; Harl 2008 established this scaling |
| Freq. scheme | imaginary axis (Gauss–Legendre) | default in VASP ACFDT |

**Kaltak, Klimeš, Kresse, JCTC 10, 2498 (2014):**
- Established imaginary-time + Laplace transform for O(N³) scaling RPA
- Minimax quadrature for imaginary-frequency grid: ~6–12 points sufficient for high accuracy
- Same ENCUTGW and NBANDS convergence requirements as conventional VASP RPA
- Validated on C, Si, LiH, Ne, Ar with < 1 meV/atom error vs conventional O(N⁴) RPA

### GPAW/PAW (Olsen & Thygesen 2012, 2013)

| Parameter | Value used | Notes |
|-----------|-----------|-------|
| Cutoff energy (ecut) | 80–164 eV (χ₀ cutoff) | "changing from 80 to 164 eV changes E_corr by > 1 eV" for Si — must extrapolate |
| Extrapolation | E_RPA(∞) = E(E_cut) + A·E_cut^{-3/2} | same scaling as VASP |
| Frequency points | 16 Gauss-Legendre (default) | 8 tested; 16 converged to < 0.1 meV |
| k-mesh | 4×4×4 (Si tutorial) | coarser than VASP recommendations |
| NBANDS | ~200% of default | set equal to cutoff-determined N_pw |
| Code | GPAW (PAW, plane-wave mode) | |

**Note on GPAW vs VASP:** GPAW and VASP use the same PAW formalism but different PAW
potentials and plane-wave implementations; they do not give identical RPA energies even for
the same material and nominally equivalent settings. Differences of 10–50 meV/atom are
expected due to different partial-wave completeness.

### FHI-aims / all-electron NAO (Ren et al. 2012)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Basis | numeric atom-centered orbitals (NAO), tier-2 | all-electron; no pseudopotential |
| k-mesh | 6×6×6 or 8×8×8 | |
| RI-V / RI-LVL | resolution-of-identity | for 2-electron integrals |
| Freq. points | 40–60 imaginary frequencies | Gauss-Legendre |
| Extrapolation | via basis-set size | not E_cut but NAO tier extrapolation |
| Code | FHI-aims | all-electron, localized basis |

**Advantage of all-electron:** No PAW augmentation or pseudopotential error; basis-set
completeness controlled by NAO tier. The RPA correlation energy is basis-set converged at
tier-2/tier-3 for most main-group solids. For Zn (3d), tier-2 may need augmentation.

### QE / ONCV (Pitts et al. 2025)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Code | Quantum ESPRESSO | plane-wave, ONCV pseudopotentials |
| ecut (wave) | 110 Ry (lattice constants) | |
| ecut (χ₀) | ~80 Ry | |
| k-mesh | 8×8×8 shifted | |
| Freq. points | 8 imaginary | system-adapted grid |
| N_bands | ~5×N_e | |

---

## 12. What Drives Spread Across Codes

Typical code-to-code spread for RPA cohesive energies: **20–80 meV/atom** for simple
semiconductors/insulators. Larger for systems with shallow d states (ZnO, Ga compounds).

1. **Basis-set completeness / PAW partial waves.** The RPA requires a complete virtual space.
   PAW potentials with few partial waves per l-channel converge slowly; ONCV and all-electron
   NAO converge differently. Basis-set extrapolation (E_cut^{-3/2} law) removes most of this
   if applied consistently.

2. **Number of imaginary-frequency points.** 8–16 points sufficient for insulators; metals
   need ~24. Gauss-Legendre vs Minimax grids give slightly different values at the same N_omega.

3. **k-mesh density.** The RPA correlation energy converges ~1/N_k; a 4×4×4 mesh can be
   off by 50–100 meV/atom for Si vs the converged value. Must use twist-average or dense mesh.

4. **DFT starting-point orbital energy.** RPA correlation energy depends on KS eigenvalues
   and orbitals. RPA@LDA vs RPA@PBE shifts cohesive energies by ~0.1–0.3 eV/atom for
   covalent solids (LDA overbinds → larger RPA@LDA E_corr). For ionic solids the difference
   is smaller.

5. **Semicore states (ZnO, GaAs, etc.).** Including Zn 3s,3p,3d in valence increases the
   virtual-state space required for convergence dramatically. With Zn 3d as core, the RPA
   cohesive energy of ZnO is qualitatively wrong (analogous to the GW gap problem).

6. **Basis-set correction / finite-basis error.** The ~E_cut^{-3/2} tail of the correlation
   energy is large. Without extrapolation, a single-cutoff calculation can be off by
   0.3–0.5 eV/atom for Si at typical cutoffs (400–600 eV). Consistent extrapolation is
   essential for comparing codes.

---

## 13. Key References

| Citation | DOI | Annotation |
|----------|-----|------------|
| Harl & Kresse, PRB **77**, 045136 (2008) | 10.1103/PhysRevB.77.045136 | Foundational: VASP/PAW ACFDT-RPA for noble-gas solids (Ne, Ar, Kr). Establishes E_cut^{-3/2} basis extrapolation. First demonstration that RPA captures vdW cohesion in solids. |
| Harl & Kresse, PRB **79**, 045117 (2009) | 10.1103/PhysRevB.79.045117 | VASP ACFDT-RPA total energies for many solids. Cohesive energies and lattice constants. Precursor to the full 2010 benchmark. |
| Harl, Schimka, Kresse, PRB **81**, 115126 (2010) | 10.1103/PhysRevB.81.115126 | **Primary RPA benchmark for solids.** VASP/PAW, RPA@PBE. Lattice constants, bulk moduli, atomization energies for 20 solids (Si, C, MgO, LiF, NaCl, ZnO, GaAs, metals, noble gases). MAE lattice constant 0.02 Å. |
| Schimka et al., Nat. Mater. **9**, 741 (2010) | 10.1038/nmat2806 | RPA@PBE for metals + insulators cohesive energies AND surface/adsorption energies. Shows RPA needed for surfaces; for bulk metals RPA ≈ PBE accuracy. |
| Lebègue et al., PRL **105**, 196401 (2010) | 10.1103/PhysRevLett.105.196401 | VASP/PAW RPA@PBE for graphite. Interlayer binding 48 meV/atom; correct vdW asymptotics. Landmark for layered materials. |
| Olsen & Thygesen, PRB **86**, 081103(R) (2012) | 10.1103/PhysRevB.86.081103 | GPAW RPA benchmark. Cohesive energies of graphite and covalent solids. RPA@LDA underbinds covalent solids (Si: 4.32 eV vs expt 4.62 eV). Rapid communication. |
| Olsen & Thygesen, PRB **87**, 075111 (2013) | 10.1103/PhysRevB.87.075111 | GPAW extended RPA benchmark: solids, molecules, graphene-metal interfaces; van der Waals to covalent. Full convergence study. |
| Kaltak, Klimeš, Kresse, JCTC **10**, 2498 (2014) | 10.1021/ct5001268 | Low-scaling O(N³) RPA via imaginary time + Laplace transforms. Minimax quadrature. Validated on Si, C, MgO, Ne, Ar. Same accuracy as O(N⁴) at < 1 meV/atom. |
| Ren, Rinke, Joas, Scheffler, J. Mater. Sci. **47**, 7447 (2012) | 10.1007/s10853-012-6570-4 | Review of RPA in computational chemistry and materials science. FHI-aims all-electron NAO results for 11 solids (C, Si, SiC, BN, AlP, LiH, LiF, LiCl, MgO). Cohesive energies and lattice constants. arXiv:1203.5536 |
| Booth, Grüneis, Kresse, Alavi, Nature **493**, 365 (2013) | 10.1038/nature11770 | FCIQMC for solids — exact wavefunction benchmark. Table 1: cohesive energies of LiH, C, BN, AlP from CCSD(T) and FCIQMC, compared to RPA. CCSD(T) within < 0.05 eV/atom of experiment; RPA less accurate for covalent systems. |
| Schimka et al., PRB **87**, 214102 (2013) | 10.1103/PhysRevB.87.214102 | Lattice constants and cohesive energies of alkali, alkaline-earth, transition metals. VASP/PAW RPA@PBE. Systematic metal benchmarks. |
| Pitts, Contant, Hellgren, PRB **112**, 085137 (2025) | 10.1103/PhysRevB.112.085137 | Self-consistent RPA and optimized hybrid functionals for solids. QE/ONCV; systems: C, Si, BN, LiF, MgO, TiO₂. Lattice constants. arXiv:2504.12768 |
| Zhang et al., New J. Phys. **21**, 013025 (2019) | 10.1088/1367-2630/aafcf6 | MSE test set: cohesive energy, lattice constant, bulk modulus for 10–20 solids; FHI-aims RPA@PBE and RPA@PBE0. Includes C, Si, MgO, LiF, BN. arXiv:1808.09780 |

---

## 14. Notes for CoQui PAW+ISDF-THC Validation

### Priority order for validation

1. **Silicon (Si)** — simplest, best-converged reference. Target: RPA@PBE cohesive energy
   ~4.6 eV/atom (VASP PAW) or ~4.3 eV (GPAW, RPA@LDA, note starting-point dependence).
   Lattice constant: 5.421 Å (VASP PAW RPA@PBE).

2. **MgO and LiF** — ionic solids where RPA performs well. Target: a(MgO) ≈ 4.196–4.200 Å,
   a(LiF) ≈ 3.986–3.996 Å. These are good tests of the Hartree/exchange balance in the
   augmentation because the Mg 2p and Li 1s core contributions are small but non-zero.

3. **Diamond (C)** — covalent, small cell, well-studied. Target: a ≈ 3.553 Å,
   E_coh ≈ 7.4–7.5 eV/atom (RPA@PBE, VASP).

4. **ZnO** — semicore stress test. Must include Zn 3s,3p,3d in valence. Convergence is
   demanding: large NBANDS, large ENCUTGW/ecut, and basis extrapolation required. No single
   well-accepted open-access RPA cohesive energy; use lattice constants (a ≈ 3.254 Å) as
   first check.

### What to watch for with PAW augmentation

- **Basis-set extrapolation is mandatory.** The RPA correlation energy has a leading
  ~A·E_cut^{-3/2} tail. Without extrapolation, a single-cutoff result cannot be compared
  across codes or against experiment. Always extrapolate or use a large enough cutoff
  that the tail is small (< 5 meV/atom target).

- **PAW partial-wave completeness.** The ISDF-THC augmentation modifies the two-electron
  integrals. For RPA the key quantity is the irreducible polarizability χ₀(q,G,G',iω).
  The on-site (augmentation) contribution to χ₀ comes from the density matrix in the
  PAW sphere. If the PAW partial-wave set is incomplete for high-energy unoccupied states,
  χ₀ is underestimated at high |G|, leading to underbinding. Compare against VASP PAW
  and FHI-aims all-electron — if CoQui underbinds systematically relative to VASP, check
  augmentation completeness.

- **NOMEGA convergence.** For Si and MgO, 12–16 Gauss-Legendre points on the imaginary
  axis is sufficient. For metals or ZnO (narrow d resonance), 20–24 may be needed.
  Cross-check by doubling NOMEGA and verifying < 1 meV/atom change.

- **k-mesh.** Use at least 6×6×6 for Si and MgO. The RPA correlation energy is smooth in
  k but the sum converges slowly for metals. For Si (semiconductor), 8×8×8 is well
  converged; 4×4×4 can give errors of ~50–100 meV/atom.

- **Starting-point sensitivity.** CoQui presumably uses PBE orbitals. The VASP reference
  numbers in Harl 2010 are RPA@PBE. The GPAW numbers from the tutorial are RPA@LDA (4.32
  eV for Si). Do not compare RPA@LDA values to RPA@PBE references without noting the
  difference (~0.2–0.3 eV/atom for covalent solids).

### Quick diagnostic table

| Material | VASP PAW RPA@PBE a (Å) | VASP PAW RPA@PBE E_coh (eV/atom) | Experiment a (Å) | Experiment E_coh (eV/atom) |
|----------|----------------------|--------------------------------|-----------------|--------------------------|
| Si | 5.421 | ~4.6 | 5.431 | 4.62 |
| C | 3.553 | ~7.4–7.5 | 3.553 | 7.37 |
| MgO | 4.196 | ~5.0–5.1 | 4.211 | 5.20 |
| LiF | 3.986 | ~4.3–4.4 | 3.972 | 4.46 |
| NaCl | 5.573 | ~3.3 | 5.595 | 3.34 |
| ZnO (a) | 3.254 | — | 3.250 | ~3.60 |
| Graphite (d_int) | 3.34 | 48 meV/atom interlayer | 3.354 | ~35–52 meV/atom interlayer |
| Ne (fcc) | ~4.46 | ~27 meV/atom | 4.430 | 27 meV/atom |
| Ar (fcc) | ~5.26 | ~88 meV/atom | 5.256 | 88 meV/atom |

