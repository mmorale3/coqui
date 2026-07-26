# abinit2coqui: COMPLETE inventory of the output bdft HDF5 file

Source tree: `/Users/mmorales/Software/CoQui_Separate_Development/src/python/mean_field/abinit_interface/`
Entry point: `abinit2coqui.py` (`main()` → `convert()`; abinit2coqui.py:632-657, 419-621).
All paths below are absolute; anchors are `file:line`.

Files that write to h5:
- `abinit2coqui.py` — /System, /System/BZ, /Orbitals, empty /Hamiltonian fallback
- `abinit_hamiltonian.py` — /Hamiltonian (NC path, `write_hamiltonian_nc`)
- `abinit_paw_hamiltonian.py` — /Hamiltonian (PAW path: `write_hamiltonian_paw`,
  `write_paw_augmentation`, `write_species_block`)

Inputs: ABINIT WFK netCDF (positional), POT netCDF (`--pot`), DEN netCDF (`--den`),
psp8 files (`--psp8`), PAW-XML datasets (`--pawxml`), atompaw `.corewf.xml` (`--corewf`).

Write conventions (abinit2coqui.py:93-124, abinit_paw_hamiltonian.py:319-330):
- int attrs `np.int32`, float attrs `np.float64`.
- Complex arrays: nda/TRIQS layout, float64 with trailing axis 2 + scalar vlen-str
  dataset attribute `__complex__ = "1"` (`_w_complex`, abinit2coqui.py:114-119).
- Bool arrays: numpy bool → HDF5 char enum (`_w_bool`, abinit2coqui.py:108-111).
- Strings: vlen str (`_w_strings`, abinit2coqui.py:122-124).

Global unit statement: ABINIT is natively Hartree; `schema_version = 2` declares
"HARTREE on disk for all energy-valued datasets", so **no Ry↔Ha factors anywhere**
(abinit_hamiltonian.py:213-216, abinit_paw_hamiltonian.py:525-529).

Symbols: `nk` = nkpt (nosym: all k are IBZ), `nspin` = nsppol, `npol` = 1 always,
`nbnd` = mband or `--nbnd` truncation, `nat`, `nsp` = nspecies,
`ngm_wfc` = union wfc G-sphere size, `ngm` = dense-sphere size,
`nh` = Σ_b(2l_b+1) projectors per species, `nhm` = max nh, `nkb` = Σ_atoms nh,
`nbeta` = ns = number of valence partial-wave channels, `nr` = radial mesh size,
`nqlc` = 2·lmax_beta+1, `nij_beta` = nbeta(nbeta+1)/2, `nij_proj` = nh(nh+1)/2.

---

## 1. `/System` (always written) — abinit2coqui.py:526-568

### Attributes

| name | type | source | notes |
|---|---|---|---|
| `number_of_atoms` | i32 | WFK `reduced_atom_positions` shape (abinit2coqui.py:150,153,527) | |
| `number_of_species` | i32 | WFK `atomic_numbers` (fallback `znuclpsp`) length (abinit2coqui.py:151,154,528) | |
| `number_of_dimensions` | i32 | constant 3 (abinit2coqui.py:529) | |
| `number_of_spins` | i32 | WFK `eigenvalues` axis 0 = nsppol (abinit2coqui.py:163-165,530) | |
| `number_of_spins_in_basis` | i32 | same (abinit2coqui.py:531) | |
| `number_of_polarizations` | i32 | constant 1 (abinit2coqui.py:477,532) | nspinor>1 aborts (abinit2coqui.py:423-424) |
| `number_of_polarizations_in_basis` | i32 | constant 1 (abinit2coqui.py:533) | |
| `number_of_bands` | i32 | WFK mband, min with `--nbnd` (abinit2coqui.py:479,534) | |
| `number_of_elec` | f64 | WFK `nelect`/`number_of_electrons`; fallback `round(occ[0].sum()/nkpt)` (abinit2coqui.py:185-190,535) | |
| `madelung_constant` | f64 | **computed**: `-2.0 * madelung_bdft(rprimd, recv, kp_grid, fft_mesh, 1e-10)` (abinit2coqui.py:539-542) | exact python port of CoQui `utils::madelung` (lattice_sums.py:102-138), incl. alpha/rcut estimation and fixed nimg=8 real-space box; Ha. Non-zero on purpose (0 would trigger bdft_system's compute-when-0 fallback — cf. madelung=0 EXX bug fix) |
| `nuclear_energy` | f64 | **computed**: `ewald_energy(rprimd, at_pos_bohr, zion_at)` (abinit2coqui.py:546-553) | Ha, e²=1; ions = valence point charges + neutralizing background (lattice_sums.py:31-86, QE `ewald(...)/e2` convention). zion per species: PAW → `round(Z − ∫ae_core·4πr²/√(4π))` cross-checked vs Σf (abinit2coqui.py:445-456); NC → psp8 `zion` (abinit2coqui.py:457-461). **Without `--psp8`/`--pawxml`: writes 0.0 + warning** (abinit2coqui.py:549-552) |
| `exx_core_core` | f64 | PAWXML `<exact_exchange core-core=...>` per species, summed over atoms: `sum(paw_parses[t-1]["exx_core_core"] for t in typat)` (abinit2coqui.py:555-560; abinit_pawxml.py:144-145) | **only** with `--pawxml` AND sum ≠ 0. Ha, frozen additive constant (plan B2) |
| `fermi_energy` | f64 | WFK `fermi_energy` (Ha), else 0.0 (abinit2coqui.py:191,561) | |

### Datasets

| name | shape/type | source | notes |
|---|---|---|---|
| `species` | (nsp,) vlen str | WFK znucl → element symbols via `_z_to_symbol()` (abinit2coqui.py:484-486,562,624-629) | unknown Z → "X%d" |
| `atomic_id` | (nat,) i32 | WFK `atom_species` (typat, 1-based) **minus 1** → 0-based (abinit2coqui.py:487,563) | |
| `atomic_positions` | (nat,3) f64 | **computed**: `xred @ rprimd` — crystal → cartesian Bohr (abinit2coqui.py:488,564) | |
| `lattice_vectors` | (3,3) f64 | WFK `primitive_vectors` unchanged, rows aₙ, Bohr (abinit2coqui.py:148,565) | |
| `reciprocal_vectors` | (3,3) f64 | **computed**: `2π·inv(rprimd).T`, rows bₙ with a·b=2π (abinit2coqui.py:198,566) | |
| `kpoint_weights` | (nk,) f64 | WFK `kpoint_weights` **renormalized**: `wtk / wtk.sum()` (abinit2coqui.py:157,567) | |

---

## 2. `/System/BZ` (always) — `write_bz` abinit2coqui.py:382-413; arrays from `build_bz_nosym` abinit2coqui.py:312-379

All **computed in converter** for the identity-only (nosym) case, mirroring
`src/mean_field/symmetry/bz_symmetry.hpp` with nsym=1, use_trev=False, Γ-centered
grid (Qpts ≡ kpts). k representatives are ABINIT's own (deliberately NOT folded;
momentum matching uses a mod-1 canonical key — abinit2coqui.py:326-338).

### Attributes
| name | value | anchor |
|---|---|---|
| `number_of_kpoints` | nk (i32) | abinit2coqui.py:384 |
| `number_of_kpoints_ibz` | nk | :385 |
| `number_of_trev_kpoint_pairs` | 0 | :386 |
| `number_of_qpoints` | nq = nk | :401 |
| `number_of_qpoints_ibz` | nq | :402 |

### Datasets
| name | shape/type | content | anchor |
|---|---|---|---|
| `kp_grid` | (3,) i32 | WFK `monkhorst_pack_folding` if present, else inferred from unique fractional coords (`_mp_grid_dims`); asserts ∏=nk | :159-160, :302-309, :320-323, :387 |
| `kpoints` | (nk,3) f64 | cartesian: `kpts_crys @ recv` | :370, :388 |
| `kpoints_crystal` | (nk,3) f64 | WFK `reduced_coordinates_of_kpoints` unchanged | :156, :330, :389 |
| `kp_symm` | (nk,) i32 | zeros | :341, :390 |
| `kp_to_ibz` | (nk,) i32 | arange(nk) | :344, :391 |
| `kp_trev` | (nk,) bool enum | all False | :342, :392 |
| `kp_trev_pair` | (nk,) i32 | all −1 | :343, :393 |
| `Symmetries` attr `number_of_symmetries` | 1 (i32) | :395-396 |
| `Symmetries/s0/R` | (3,3) f64 | identity | :397-398 |
| `Symmetries/s0/ft` | (3,) f64 | zeros | :399 |
| `qpoints` | (nq,3) f64 | cartesian Q = kpts (Γ-centered) | :347, :371, :403 |
| `qk_to_k2` | (nq,nk) i32 | index map kpts[b] = kpts[a] − Qpts[q] (mod 1) | :349-352, :404 |
| `qminus` | (nq,) i32 | index of −Q (mod 1) | :353-355, :405 |
| `qp_symm` | (nq,) i32 | zeros | :357, :406 |
| `qp_trev` | (nq,) bool enum | all False | :358, :407 |
| `qp_to_ibz` | (nq,) i32 | arange(nq) | :359, :408 |
| `qsymms` | (1,) i32 | [0] | :362, :409 |
| `nq_per_s` | (1,) i32 | [nq] | :363, :410 |
| `ks_to_k` | (1,nk) i32 | arange | :364, :411 |
| `Qs` | (1,2·nq) i32 | first nq = arange, rest 0 | :366-367, :412 |
| `qs_to_q` | (1,nq) i32 | arange | :365, :413 |

---

## 3. `/Orbitals` (always) — abinit2coqui.py:570-594

### Attributes
| name | type | source | anchor |
|---|---|---|---|
| `number_of_spins`, `number_of_spins_in_basis` | i32 | nsppol | :572-573 |
| `number_of_polarizations`, `number_of_polarizations_in_basis` | i32 | 1 | :574-575 |
| `number_of_kpoints`, `number_of_kpoints_ibz` | i32 | nk (nosym: equal) | :576-577 |
| `number_of_bands` | i32 | nbnd | :578 |
| `number_of_aux_bands` | i32 | 0 | :579 |
| `ecutrho` | f64 | **computed**: `2.0*wfc_ecut` without `--pot`; **`4.0*wfc_ecut` with `--pot`** (abinit2coqui.py:507-513, :580) |

### Datasets
| name | shape/type | source/derivation | anchor |
|---|---|---|---|
| `fft_mesh` | (3,) i32 | **computed** smooth grid: `2*max|miller_wfc_d|+2` per dim (even, anti-aliasing); max with WFK ngfft if present (always None from `read_wfk`) | :276-282, :502, :581 |
| `fft_mesh_aug` | (3,) i32 | with `--pot`: **POT `vtrial` grid dims** `vtrial.shape[1:4]` (matches the /Hamiltonian dense grid); else = fft_mesh | :506-512, :582 |
| `eigval` | (nspin,nk,nbnd) f64 | WFK `eigenvalues`, band-truncated, **Ha unchanged** | :163, :497, :583 |
| `occ` | (nspin,nk,nbnd) f64 | WFK `occupations`; **nspin==1: divided by 2** (ABINIT occ∈[0,2] → CoQui per-spin-channel occ∈[0,1]): `if nspin == 1: occ = occ / 2.0` | :497-500, :584 |
| `miller_wfc` | (ngm_wfc,3) i32 | **computed**: union of per-k Miller lists from WFK `reduced_coordinates_of_plane_waves` (first-appearance order), `number_of_coefficients` = true npw per k | :168-173, :235-282, :586 |
| `wfc_ecut` | scalar f64 | **computed**: `0.5*max|G_cart|²·(1+1e-8)` over the union | :271-274, :587 |
| `wfc_fft_grid` | (3,) i32 | = fft_mesh (same `wfc_mesh` array) | :588 |
| `wfc_ngm` | scalar i32 | ngm_wfc | :589 |
| `psi_s{is}_k{ik}` | (nbnd,ngm_wfc) complex (stored f64 …,2 + `__complex__`) | WFK `coefficients_of_wavefunctions` (nsppol,nkpt,mband,nspinor,mpw,2): nspinor slot 0, first npw(k) coeffs scattered onto the union grid via `kg_idx`, zeros where k has no plane wave; **no rescaling, no phase change** | :285-296, :591-594 |

One entry per (is,ik): is∈[0,nspin), ik∈[0,nk).

---

## 4. `/Hamiltonian` — three variants

- `--pot` + `--pawxml` → PAW block (abinit2coqui.py:601-606 → `write_hamiltonian_paw`)
- `--pot` + `--psp8`  → NC block (abinit2coqui.py:607-614 → `write_hamiltonian_nc`)
- otherwise → **empty group** `/Hamiltonian` with no attrs (abinit2coqui.py:615-616)

WFK usepaw=1 without `--pawxml` is a hard abort (abinit2coqui.py:425-428).
POT `vtrial` is read fresh in both branches (abinit2coqui.py:603, :609).

### 4a. Root attributes (both NC and PAW)

| name | value | anchor |
|---|---|---|
| `schema_version` | **2** (i32) — "HARTREE on disk for all energy-valued datasets… ABINIT is natively Ha, so no conversion factors anywhere below" | abinit_paw_hamiltonian.py:525-528; abinit_hamiltonian.py:213-216 |
| `pp_type` | vlen str `"paw"` / `"ncpp"` | abinit_paw_hamiltonian.py:529; abinit_hamiltonian.py:217 |

---

### 4b. `/Hamiltonian/paw` (PAW path) — `write_hamiltonian_paw`, abinit_paw_hamiltonian.py:441-618

Dense grid: `miller_g, (I1,I2,I3) = ah.dense_sphere(ngfft, rprimd, recv)` with
ngfft = POT vtrial dims — the largest G-sphere inscribed in the FFT box
(`Gn < gmax·(1−1e-9)`, gmax = min over d of (n_d/2)·2π/|a_d|), matching QE's
ecutrho-sphere miller_g convention, never the raw box (abinit_hamiltonian.py:142-155;
abinit_paw_hamiltonian.py:468-471).

#### Group attributes (all i32) — abinit_paw_hamiltonian.py:531-536
| name | value |
|---|---|
| `number_of_nspins` | nsppol |
| `number_of_polarizations` | 1 |
| `number_of_kpoints` | nk |
| `max_npw` | max over k of WFK `number_of_coefficients` |
| `number_of_atoms` | nat |
| `number_of_species` | nsp |
| `total_num_of_proj` | nkb = Σ_atoms nh(species) |
| `max_proj_per_atom` | nhm = max nh |
| `ngm` | dense-sphere count |
| `lspinorbit_nl`, `lspinorbit_loc` | 0, 0 |

#### Datasets

**`miller_g`** — (ngm,3) i32. Computed dense sphere (above). abinit_paw_hamiltonian.py:537.

**`pp_local_component`** — (ngm,) complex. **Computed** local ionic potential,
Hartree (schema 2). Assembly per atom (abinit_paw_hamiltonian.py:492-517), sources:
PAWXML `<blochl_local_ionic_potential>` (`vbar`, abinit_pawxml.py:129) +
PAWXML `<ae_core_density>`:
```python
vion = vbar[:nn].astype(float)
ncore = p["ae_core"][:nn] / np.sqrt(4.0 * np.pi)   # L=0 moment -> number density
vion = vion + _radial_hartree(ncore, r)            # -Z/r tail -> -Zval/r
Qtail = -float((r * vion)[-1])                     # asymptotic ionic charge from r*vion plateau
rvsr = r * vion + Qtail                            # short-ranged
sr = (4*np.pi/vol) * trapezoid(rvsr*sin(G r)/G, r)
sr -= 4.0*np.pi*Qtail/(vol*Gs**2)                  # G!=0 analytic -Q/r long range
sr[small] = (4*np.pi/vol) * trapezoid(rvsr * r, r) # G=0 finite "alpha" reference
vloc_tot += np.exp(-1j*(Gcart @ tau[a])) * sr
```
This is THE Coulomb/alpha split fix (memory `project_abinit_converter_vloc_e1e_fix`):
Bloechl's v_H[ñ_Zc] carries the bare −Z/r tail; adding frozen-core Hartree V_H[n_core]
screens it to −Zval/r, then the standard split makes the sin-transform integrand
short-ranged with the analytic −4πQ/(vol·G²) restored and a finite G=0 alpha term.
Comment records the pre-fix failure: FFT'ing the bare −Z/r gave a divergent V(G=0)
→ e_1e ≈ −37000 Ha (abinit_paw_hamiltonian.py:490-497). Caveat (same memory):
Qtail is read off the plateau, ≈4.18 vs Zval=4 for the Si test dataset.
Structure-factor sign: `e^{−iG·τ}`. No e² factor.

**`scf_local_potential`** — (nspin, npol², ngm) complex. POT `vtrial[0,:,:,:,0]`
(total KS potential, Ha) → `Vg = fftn(vloc)/vloc.size`, sampled at the sphere bins
`Vg[I1,I2,I3]`, reshaped (abinit_paw_hamiltonian.py:472-475, :539). Forward FFT with
1/N normalization; no unit conversion. nspin>1 vtrial layout NOT handled (index 0 only).

**`vxc`, `vxc_with_nlcc`** — (nspin, npol², ngm) complex.
- With `--den`: **computed** vxc = FFT of `xc_functionals.vxc_grid(rho_DEN, recv, xc_name)`
  sampled at sphere bins (abinit_paw_hamiltonian.py:547-581). rho_DEN = ABINIT smooth
  rhor = ñ + n̂ (read_den, abinit2coqui.py:208-229; nsppol=1 only, hard abort otherwise;
  DEN grid must equal POT grid, abinit_paw_hamiltonian.py:553-555).
  `vxc_with_nlcc` = vxc_grid(rho_DEN + Σ_a core_r) where core_g is the spherical FT of
  PAWXML `<pseudo_core_density>`/√(4π) with structure factor e^{−iG·τ}
  (abinit_paw_hamiltonian.py:559-576), inverse-FFT'd to the box (`_scatter_to_box`,
  :433-438). Functional: `--xc` or WFK `ixc` via `functional_from_ixc`
  (abinit2coqui.py:466-474; xc_functionals.py:127-145; PW92/PBE only, analytic e(ρ,σ) +
  finite-difference derivatives + spectral div, xc_functionals.py:94-124).
  `vxc_grid` returns **Hartree**. NOTE for the data contract: the code comment claims
  "vxc / vxc_with_nlcc (Ry on disk, add_vxc scales x0.5)" (abinit_paw_hamiltonian.py:540)
  while the written values are the Ha-native vxc_grid output — comment vs value
  discrepancy to reconcile against CoQui's add_vxc/schema-2 read path.
- Without `--den`: **zeros** of the same shape (abinit_paw_hamiltonian.py:582-584).

**`proj_per_atom`** — (nsp,) i32 = nh per **species** (QE nh(1:nsp) convention,
plan B2/F10a), nh = Σ_b(2l_b+1) from PAWXML valence-state l's
(abinit_paw_hamiltonian.py:479-483, :585).

**`projector_offset`** — (nat,) i32, cumulative per-atom projector offsets
(abinit_paw_hamiltonian.py:484, :586).

**`npw`** — (nk,) i32 = WFK `number_of_coefficients` (abinit_paw_hamiltonian.py:587).

**`atomic_id`** — (nat,) i32 = typat−1 (abinit_paw_hamiltonian.py:520, :588).

**`dion`** — (nsp, nhm, nhm) f64, **Hartree**, m-diagonal expansion of the per-species
beta-basis D⁰ (abinit_paw_hamiltonian.py:589-597):
```python
for I in range(nh):
    for J in range(nh):
        if ch["nhtolm"][I] == ch["nhtolm"][J]:
            dion[isp, I, J] = s["dij0"][ch["indv"][I]-1, ch["indv"][J]-1]
```
The beta-basis `dij0` is **assembled in the converter** (`assemble_dij0`,
abinit_paw_hamiltonian.py:152-201), reproducing ABINIT `atompaw_dij0`
(m_paw_atom.F90, opt_init==0). Ingredients, exactly:
1. `ked = parse["dij0"]` = PAWXML `<kinetic_energy_differences>` (ns×ns; this is the
   ONLY D⁰ piece the XML stores) — abinit_pawxml.py:134-136; abinit_paw_hamiltonian.py:179-181.
2. AE ionic Hartree: `vhnzc = _radial_hartree(ncore, r) − Z/r`, ncore = ae_core/√(4π),
   value at r<1e-12 zeroed (abinit_paw_hamiltonian.py:110-114).
3. PS ionic Hartree: `vhtnzc = reconstruct_vhtnzc(...)` = PAWXML `<zero_potential>`
   + poisson[ tncore·4πr² + g0·r²·(qcore − Z) ]/r, with
   qcore = ∫(ncore−tncore)·4πr² dr and g0 the l=0 compensation shape normalized to a
   unit monopole (abinit_paw_hamiltonian.py:60-90; `_poisson_over_r` :48-57). This is
   the electrostatic part of ABINIT m_pawpsp.F90 vlocopt==0 'Vbare'; **XC-FREE ON
   PURPOSE** — ABINIT's stored vlocr = vbare + vh − vxc1 subtracts a frozen atomic XC
   for its DFT SCF; CoQui's frozen D⁰ excludes vxc1 (GW/HF baseline, matches QE dion
   convention; including it would double-count XC vs exact exchange)
   (abinit_paw_hamiltonian.py:70-76, :169-178).
4. Assembly (same-l pairs only, trapezoid on r cut at kkbeta = index of paw_radius):
```python
intvh = np.trapezoid((vhtnzc * g0r2)[:k], r[:k])       # g0r2 unit monopole
D[i, j] += np.trapezoid((aa * vhnzc)[:k], r[:k])       # aa = u_ae_i * u_ae_j
D[i, j] -= np.trapezoid((pp * vhtnzc)[:k], r[:k])      # pp = u_ps_i * u_ps_j
D[i, j] -= intvh * np.trapezoid((aa - pp)[:k], r[:k])  # compensation-moment term
```
(abinit_paw_hamiltonian.py:186-201). u = r·R form (phi·r, :220-222). No moment
terms beyond the l=0 `intvh` piece; no v_H[valence] (deeq is CoQui-native);
verified vs ABINIT dumped D⁰ to ~4 digits (comment :166-168). Fallback:
kinetic-only `ked` when zero_potential/core densities are missing (:183-184).
NC contrast: NC `dion` = psp8 `ekb` diagonal, Ha as-is (abinit_hamiltonian.py:277-284).

**`miller_k{ik}`** — (npw_k,3) i32 per k: WFK `reduced_coordinates_of_plane_waves[ik,:npw_k]`
unchanged (abinit_paw_hamiltonian.py:599-609).

**`projector_k{ik}`** — (nkb, npw_k) complex per k (**transposed to projectors-as-rows**,
matching CoQui's hyperslab read; abinit_hamiltonian.py:295-298, abinit_paw_hamiltonian.py:610).
**Computed** β_i(k+G) via `build_beta_k` (abinit_hamiltonian.py:99-125) with PAW channels
`chi = r * proj` (PAWXML `<projector_function>` R-form times r; `_paw_channels`,
abinit_paw_hamiltonian.py:419-431):
- radial transform `F_l(q) = ∫ chi(r) j_l(qr) r^_R_POWER dr`, `_R_POWER = 1`
  (validated; chi already r-weighted ⇒ physically ∫proj j_l r² dr) — abinit_hamiltonian.py:23-28, :72-78
- phase `(-1j)**l` (QE init_us_2 `(0,-1)**l`), `_USE_MINUS_I = True` — :27, :117
- normalization `pref = 4π/√Ω` — :110
- structure factor `sk = exp(−1j*(kpg @ tau))` — :111
- **real spherical harmonics with QE ylmr2 odd-m signs** (`real_ylm`,
  abinit_hamiltonian.py:31-69 — commit 3956b45 fix): lm order m = 0, +1, −1, (+2, −2);
  QE builds real pairs from Condon-Shortley (−1)^m complex harmonics so every ODD-m
  component carries an extra minus sign:
  ```python
  out[:, 1] = -c * x           # m=+1 (QE: -px)
  out[:, 2] = -c * y           # m=-1 (QE: -py)
  ...
  out[:, 1] = -c1 * x * z      # m=+1  (QE: -xz)
  out[:, 2] = -c1 * y * z      # m=-1  (QE: -yz)
  out[:, 3] = c2 * (x*x - y*y) # m=+2  (unchanged)
  out[:, 4] = c1 * x * y       # m=-2  (unchanged)
  ```
  Comment records the measured −51.7 mHa exchange decoherence the plain-harmonics
  version caused (abinit_hamiltonian.py:38-44). `l>2` raises NotImplementedError (:68).
  Eigenvalue validation 5.6e-9 Ha (:23).
Column order per atom: species channel order × m; atoms concatenated in typat order
(abinit_paw_hamiltonian.py:602-608). `_paw_channels` ekb = diag(dij0) is metadata only,
not written.

**`ijtoh`** — (nsp,nhm,nhm) i32, `write_paw_augmentation` (abinit_paw_hamiltonian.py:621-637).
**Computed**: QE 1-based upper-triangle pair packing from `enumerate_channels`
(paw_radial.py:24-54); zero-padded to nhm.

**`qq_nt`** — (nsp,nhm,nhm) f64. **Computed**: `qq_nt(ih,jh) = qqq(indv(ih),indv(jh))`
gated by `nhtolm(ih)==nhtolm(jh)` (paw_radial.py:136-144), with
`qqq(nb,mb) = q_ij^{L=0}` = L=0 multipole moment of the AE−PS pair density
(paw_radial.py:125-133; moments below). Zero-padded to nhm.
(abinit_paw_hamiltonian.py:631-637).

**`augmentation_function_isp{nt}`** — (nij_proj, ngm) complex per species
(0-based species index in the name; abinit_paw_hamiltonian.py:639-641).
**Computed** Q^IJ(G) = QE qvan2 port (paw_qvan.py:263-302):
`q(G,ijh) = Σ_lp (−i)^L ap(lp,ivl,jvl) Y_lp(Ĝ) qrad(|G|,ijv,L)` with
- `qrad(iG,ijv,L) = (4π/Ω)·∫ qfuncl_{L,ijv}(r) j_L(|G|r) dr`, QE `simpson` with the
  `rab = dr/di` log-grid measure, cut at kkbeta, QE L-selection rule
  (|li−lj| ≤ L ≤ li+lj, L+li+lj even) (paw_qvan.py:219-257)
- `ylmr2` = exact port of upflib/ylmr2.f90 (QE ordering lm = l²+1+{2m−1 cos, 2m sin},
  QE sign/normalization) (paw_qvan.py:42-105)
- `ap/lpx/lpl` real-Gaunt coefficients by QE `aainit`'s direction-sampling +
  matrix-inversion recipe (sample-independent; paw_qvan.py:111-149)
- |G| cartesian Bohr⁻¹ from `miller_g @ recv` — no tpiba (paw_qvan.py:20-22, :272-273)

`qfuncl` itself (input to qrad, also written under Species): **computed, NOT from a
PAWXML qfuncl** (PAW-XML has none): `qfuncl[L,ijv,r] = q_ij^L · g_L(r)` with
- moments `q_ij^L = ∫(pfunc − ptfunc)·r^L dr` (QE simpson/rab), pfunc = u_ae,i·u_ae,j,
  ptfunc = u_ps,i·u_ps,j, u = r·R (paw_radial.py:60-90; adapter r-multiplication
  abinit_paw_hamiltonian.py:220-222)
- g_L = analytic shape per PAWXML `shape_function type=`: `bessel` (per-L 2-term
  j_L sum with g_L(rc)=0, g_L'(rc)=0; paw_radial.py:165-194), `gauss`, `sinc`
  (L-independent; :153-162), each in the g_L·r² convention, renormalized to unit
  L-moment ∫g_L r^L dr = 1 (paw_radial.py:96-119); selection
  abinit_paw_hamiltonian.py:222-230, unknown type → NotImplementedError.
- Hard cross-check vs the XML's tabulated `<shape_function>` values when present
  (rel L-inf > 1e-5 → RuntimeError; `check_shape_function`,
  abinit_paw_hamiltonian.py:120-149, :232-233).

Aggregation: `build_paw_augmentation` (abinit_paw_hamiltonian.py:281-315).

---

### 4c. `/Hamiltonian/Species` (PAW path only) — `write_species_block`, abinit_paw_hamiltonian.py:333-416

Group attr: `number_of_species` (i32) — :338.

Per species `nt{i}` (0-based):

#### `/Hamiltonian/Species/nt{i}` attributes
| name | type | value/source | anchor |
|---|---|---|---|
| `species_kind` | vlen str | `"paw"` | :341 |
| `mesh` | i32 | nr (PAWXML radial_grid iend+1) | :344; abinit_pawxml.py:74-82 |
| `kkbeta` | i32 | **computed**: `searchsorted(r, paw_radius)+1` (inclusive of rc); paw_radius from PAWXML `<paw_radius rc=>` | abinit_paw_hamiltonian.py:219, :344; abinit_pawxml.py:109 |
| `nbeta` | i32 | ns = # PAWXML `<valence_states>` | :343-344 |
| `nh` | i32 | Σ_b(2l_b+1) | :345 |
| `lmax` | i32 | max valence l | :345 |
| `lmax_rho` | i32 | 2·lmax | :217, :346 |
| `nqlc` | i32 | 2·lmax+1 | :217, :346 |
| `q_with_l` | i32 | 1 (constant) | :347 |
| `nqf` | i32 | 0 (constant) | :347 |
| `zp` | f64 | Σ state occupations f (PAWXML `<state f=>`) | :231, :349 |
| `exx_core_core` | f64 | PAWXML `<exact_exchange core-core=>`; **only when ≠ 0** | :396-398; abinit_pawxml.py:144-145 |

#### `/Hamiltonian/Species/nt{i}` datasets
| name | shape | source | anchor |
|---|---|---|---|
| `r` | (nr,) f64 | PAWXML `<radial_grid>` values r = a(e^{di}−1) (and other eq forms) | :350; abinit_pawxml.py:40-57 |
| `rab` | (nr,) f64 | **computed** dr/di = a·d·e^{di} | abinit_paw_hamiltonian.py:213-214, :351 |
| `lll` | (nbeta,) i32 | valence-state l's | :352 |
| `indv` | (nh,) i32 | **computed** channel map, 1-based beta index per projector | :353; paw_radial.py:37-52 |
| `nhtol` | (nh,) i32 | **computed**, l per projector | :354 |
| `nhtolm` | (nh,) i32 | **computed**, 1-based lm = l²+1+m (QE Ylm order) | :355; paw_radial.py:43 |
| `kbeta` | (nbeta,) i32 | **computed**: every entry = kkbeta | :356 |
| `aewfc` | (nbeta,nr) f64 | PAWXML `<ae_partial_wave>` × r (**R → u = r·R**) | :221, :357 |
| `pswfc` | (nbeta,nr) f64 | PAWXML `<pseudo_partial_wave>` × r | :222, :358 |
| `qqq` | (nbeta,nbeta) f64 | **computed** L=0 moments (see qq_nt above) | :359 |
| `qfuncl` | (nqlc,nij_beta,nr) f64 | **computed** moment × normalized shape (see 4b) | :360 |
| `beta` | (nbeta,nr) f64 | PAWXML `<projector_function>` × r (QE u-form, full mesh) | :249-250, :361-362 |
| `dion` | (nbeta,nbeta) f64 | **computed** `assemble_dij0` output (beta basis, Ha) — see the /Hamiltonian/paw/dion entry; on-disk pair consistent with ae_vloc/vloc_ps by construction | :234, :363-364 |

#### `/Hamiltonian/Species/nt{i}/paw` subgroup — :365-384
Attrs (read by pseudopot.cpp read_vnl_h5:765-767, per comment :367-369):
`lmax_aug` = lmax_rho (i32), `raug` = **−1.0** (f64 sentinel → use iraug),
`iraug` = kkbeta (i32) — :369-371.

| name | shape | source | anchor |
|---|---|---|---|
| `pfunc` | (nbeta,nbeta,nr) f64 | **computed** u_ae,i·u_ae,j outer products | :293-298, :372; paw_radial.py:60-62 |
| `ptfunc` | (nbeta,nbeta,nr) f64 | **computed** u_ps,i·u_ps,j | :373 |
| `augmom` | (nqlc,nbeta,nbeta) f64 | **computed**: moments unpacked from triangular ijv to the symmetric (nb,mb) matrix | :374-381 |
| `ae_vloc` | (nr,) f64 | **computed** vhnzc = v_H[n_core] − Z/r (AE frozen ionic Hartree, **Hartree on disk**, schema 2 — deliberately deviates from QE UPF ae_vloc which carries frozen atomic XC; XC-free per plan I2) | :253-262, :382-384 |
| `vloc_ps` | (nr,) f64 | **computed** vhtnzc = reconstruct_vhtnzc (PS frozen ionic Hartree, Ha) | :263, :382-384 |
| `ae_rho_atc` | (nr,) f64 | PAWXML `<ae_core_density>` **/ √(4π)** (L=0 moment → number density) | :236-238, :382-384 |
| `rho_atc_ps` | (nr,) f64 | PAWXML `<pseudo_core_density>` / √(4π) | :239, :382-384 |
| `oc` | (nbeta,) f64 | PAWXML valence-state occupations f | :251-252, :382-384 |

ae_vloc/vloc_ps written only when reconstructable (needs znucl + ae_core [+
zero_potential + ps_core for vloc_ps]); ae_rho_atc/rho_atc_ps only when the XML
has the densities (:246-247, :259-263, :382-384).

#### `/Hamiltonian/Species/nt{i}/Core` (conditional) — :385-394
Written only when core AE orbitals exist: from the PAWXML's own
`<core_states>`/`<ae_core_wavefunction>` blocks (abinit_pawxml.py:147-177) **or**
attached from an atompaw `.corewf.xml` via `--corewf` (`attach_corewf`,
abinit_pawxml.py:189-222 — radial grid must match, id-matched with document-order
fallback since atompaw ids differ between blocks). Missing (n,l) metadata →
RuntimeError (abinit_paw_hamiltonian.py:266-272).

| name | shape | source | anchor |
|---|---|---|---|
| attr `ncore_orbitals` | i32 | count | :390-391 |
| `n` | (ncore,) **f64** | core-state n (−1 when unknown) — pw2coqui GIPAW schema uses float | :273-275, :392 |
| `l` | (ncore,) f64 | core-state l | :276, :393 |
| `ae_wfc` | (ncore,nr) f64 | core R(r) × r (**u = r·R form**) | :277, :394 |

This is the **input** for CoQui's native ex_cvij builder (plan B3), not ex_cvij itself.

#### `/Hamiltonian/Species/nt{i}/Onecenter` — :399-416
| name | shape | provenance |
|---|---|---|
| `deltaC` | (nh,nh,nh,nh) f64 | **COMPUTED in converter** (`compute_deltaC`, paw_deltaC.py:111-142) — QE PAW_init_fock_kernel port, **NOT** the PAWXML `exact_exchange_X_matrix`. deltaC = K_ae − K_ps in **Hartree** with K[ij,ou] = Σ_LM ∫ V_LM^{ij}(r)·ρ_LM^{ou}(r) dr; ρ_LM from real-Gaunt-contracted pfunc (AE) / ptfunc + qfuncl (PS) pair densities (paw_deltaC.py:80-108); V_LM = multipole radial Hartree (4π/(2L+1))[r^{−(L+1)}∫₀^r ρ r'^L + r^L ∫_r^∞ ρ r'^{−(L+1)}] (:48-77); e² bookkeeping: "QE stores paw_fockrnl = e2*kexx … deltaC = ke/e2² = K_ae − K_ps (my V0 has no e2)" (:139-141), i.e. no e² factors appear — Ha directly. lmax_rho = 2·lmax; integral weight = QE simpson weights over the full mesh (kkbeta=None default → mesh = nr; call site abinit_paw_hamiltonian.py:310-311) |
| `ex_cvij` | (nh,nh) f64 | **PAWXML `<exact_exchange_X_matrix>`** (ns×ns, abinit_pawxml.py:139-143) expanded ln → nh: `ex_cv[I,J] = exxln[indv[I]-1, indv[J]-1]` **only when `nhtolm[I]==nhtolm[J]`** (exactly ABINIT's `if(ilm==jlm) ex_cvij=...`); contracts linearly with becsum, factor 1 (abinit_paw_hamiltonian.py:404-416). **Conditional**: only when the XML has the matrix. NOT from `.corewf` (that feeds Core/ instead). No unit conversion |

---

### 4d. `/Hamiltonian/ncpp` (NC path) — `write_hamiltonian_nc`, abinit_hamiltonian.py:158-299

Same root attrs (schema_version=2, pp_type="ncpp"). Group attrs identical in name/
meaning to the paw group list (abinit_hamiltonian.py:219-224); nh per species =
Σ_l nproj[l]·(2l+1) from psp8 (abinit_psp8.py:99-101).

| name | shape | source | anchor |
|---|---|---|---|
| `miller_g` | (ngm,3) i32 | dense_sphere on POT ngfft | :193-196, :225 |
| `pp_local_component` | (ngm,) complex | **computed** Σ_a e^{−iG·τ_a}·V_loc,sp(G); `vloc_of_g`: psp8 Vloc(r), Coulomb split `rvsr = r*vloc + zion` (short-ranged), sr transform − 4π·zion/(vol·G²), G=0 = (4π/vol)∫rvsr·r dr; Ha, e²=1 | abinit_hamiltonian.py:81-96, :202-209, :226-229 |
| `pp_local_component_nc` | (nspin,npol²,ngm) complex | same values tiled (SO/newer-build layout) — **NC path only**, PAW path has no _nc variant | :230-231 |
| `scf_local_potential` | (nspin,npol²,ngm) complex | POT vtrial FFT (as PAW) | :197-199, :232-233 |
| `vxc`, `vxc_with_nlcc` | (nspin,npol²,ngm) complex | **only with `--den`** (no else branch — datasets absent otherwise, unlike PAW's zeros); vxc = vxc_grid(rho_DEN); nlcc adds psp8 model core charge `rhoc/(4π)` (psp8 stores 4π·n_c(r); convention flagged unverified, abinit_psp8.py:85-94) via spherical FT when fchrg>0, else vxc_with_nlcc = vxc | :234-272 |
| `proj_per_atom` | (nsp,) i32 | psp8 nh per species | :186-188, :273 |
| `projector_offset` | (nat,) i32 | cumulative | :189, :274 |
| `npw` | (nk,) i32 | WFK number_of_coefficients | :275 |
| `atomic_id` | (nat,) i32 | typat−1 | :276 |
| `dion` | (nsp,nhm,nhm) f64 | psp8 `ekb` (KB energies, **Ha as-is**) on the diagonal, m-expanded via build_beta_k meta | :277-284; abinit_psp8.py:62 |
| `miller_k{ik}` | (npw_k,3) i32 | WFK per-k G list | :286-295 |
| `projector_k{ik}` | (nkb,npw_k) complex | build_beta_k with psp8 `chi` (already r-weighted like UPF beta → `_R_POWER=1`), (−i)^l, 4π/√Ω, e^{−i(k+G)·τ}, QE-sign real_ylm; transposed to rows | :23-28, :99-125, :286-298 |

No Species / Onecenter / augmentation datasets in the NC path.

---

## 5. Computed-vs-copied summary (the data-contract hot list)

| quantity | provenance |
|---|---|
| `dion` (both Species beta-basis and paw m-expanded) | **ASSEMBLED**: XML kinetic_energy_differences + ⟨u_ae u_ae\|v_H[n_core]−Z/r⟩ − ⟨u_ps u_ps\|v_H[ñ_Zc]⟩ − intvh·⟨u_ae²−u_ps²⟩ (same-l, trapezoid to kkbeta); XC-free by design (abinit_paw_hamiltonian.py:152-201). This is the "ABINIT converter adds ionic" risk item: QE UPF dion already ships the full frozen D⁰, the XML does not |
| `deltaC` | **COMPUTED** (QE paw_exx port), never read from XML (paw_deltaC.py) |
| `ex_cvij` | **COPIED** from PAWXML exact_exchange_X_matrix, lm-gated ln→nh expansion (abinit_paw_hamiltonian.py:404-416) |
| `Core/ae_wfc` | **COPIED** from XML core blocks or `--corewf` file, ×r (abinit_pawxml.py:147-222) |
| `qfuncl` / `augmom` / `qqq` / `qq_nt` | **COMPUTED** from AE−PS partial-wave moments × analytic shape (bessel/gauss/sinc per XML shape_type), hard-checked vs tabulated `<shape_function>` (paw_radial.py; abinit_paw_hamiltonian.py:120-149) |
| `augmentation_function_isp{nt}` (Q^IJ(G)) | **COMPUTED**: qvan2 port, (−i)^L·ap·Y_lp(Ĝ)·(4π/Ω)∫qfuncl j_L, QE simpson/rab (paw_qvan.py) |
| `projector_k{ik}` | **COMPUTED**: (−i)^l·(4π/√Ω)·Y_lm^{QE-signs}(k+G)·∫(r·proj) j_l r dr·e^{−i(k+G)·τ} (abinit_hamiltonian.py:99-125; odd-m sign fix 3956b45 at :31-69) |
| `pp_local_component` (PAW) | **COMPUTED**: FT[vbar + V_H[ae_core]] with Qtail Coulomb/alpha split (abinit_paw_hamiltonian.py:492-517) |
| `scf_local_potential` | **COPIED** (FFT of POT vtrial, spin 0 comp 0) |
| `vxc`/`vxc_with_nlcc` | **COMPUTED** (own PW92/PBE on DEN rhor [+ FT'd frozen PS-core/model-core]) or zeros(PAW)/absent(NC) |
| `ae_vloc`/`vloc_ps` | **COMPUTED** (vhnzc/vhtnzc reconstruction, Ha, XC-free — same arrays dion integrates) |
| `madelung_constant`, `nuclear_energy` | **COMPUTED** lattice sums (lattice_sums.py) |
| `exx_core_core` | **COPIED** from XML attribute (per-species attr + atom-summed /System attr) |
| eigval/occ/psi/miller/k-data | **COPIED** from WFK (occ ÷2 at nspin=1; ψ scattered to union grid; kpoint_weights renormalized; everything else unchanged) |

Real-harmonic conventions: two independent implementations, both QE-convention —
`real_ylm` (abinit_hamiltonian.py:45-69, l≤2, explicit odd-m minus signs, feeds
projector_k) and `ylmr2` (paw_qvan.py:42-105, full recursive upflib port, feeds
Q^IJ(G) and the real-Gaunt table). qvan/deltaC pair densities use the same
`real_gaunt` ap table (paw_qvan.py:111-149).

## 6. CLI flags and gating (abinit2coqui.py:632-653)

| flag | gates |
|---|---|
| `wfk` (positional) | everything (WFK netCDF; prtwf 1, iomode 3, istwfk *1, nsym 1 expected) |
| `--outdir`, `--prefix` | output path `{outdir}/{prefix}.h5` (abinit2coqui.py:515-517) |
| `--nbnd` | band truncation of eigval/occ/psi and number_of_bands |
| `--pot` | any real /Hamiltonian block; also switches fft_mesh_aug to the POT grid and ecutrho 2→4·wfc_ecut (abinit2coqui.py:506-513) |
| `--pawxml` (1 per species, znucl order) | PAW path (+ required when WFK usepaw=1); Species/Onecenter blocks; zval for nuclear_energy; /System exx_core_core |
| `--psp8` (1 per species) | NC path; zval for nuclear_energy |
| `--den` | real vxc/vxc_with_nlcc (else zeros in PAW, absent in NC) |
| `--xc {pbe,lda_pw}` | functional for `--den` (default: WFK ixc via `functional_from_ixc`; missing both → exit) |
| `--corewf` (1 per --pawxml) | attaches atompaw core orbitals → Species/nt{i}/Core (count mismatch → exit, abinit2coqui.py:439-444) |

No `--pot`: `/Hamiltonian` is written as an **empty group without even
schema_version/pp_type attrs** (abinit2coqui.py:615-616).

## 7. Known caveats recorded in code/comments (contract-relevant)

- vxc "Ry on disk, add_vxc scales x0.5" comments (abinit_paw_hamiltonian.py:540,
  abinit_hamiltonian.py:234) vs Ha-valued vxc_grid output + schema_version-2
  "no conversion factors" statement — reconcile against CoQui's reader.
- PAW pp_local_component Qtail read from the r·vion plateau (Si dataset: 4.18 vs
  Zval=4; memory `project_abinit_converter_vloc_e1e_fix`).
- psp8 NLCC rhoc = 4π·n_c convention "unverified against a real NLCC psp8"
  (abinit_psp8.py:88-89).
- deltaC integrates over the full mesh (kkbeta=None default), while dion/qrad cut
  at kkbeta.
- `real_ylm` hard-fails for l>2 (projector path); ylmr2/qvan path is general.
- nspinor>1, nsppol>1 DEN, and non-first-spin vtrial components unsupported.
