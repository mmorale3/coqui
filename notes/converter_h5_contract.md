# CoQui converter h5 contract — conventions, expectations, and backend parity

Status: AUDIT DELIVERABLE, 2026-07-25. Describes the schema_version-2 files as
actually written by `qe_converter/pw2coqui.f90` (QE) and
`src/python/mean_field/abinit_interface/` (ABINIT) and as actually consumed by
CoQui (`pseudopot::read_vnl_h5`, `add_vxc`, `mean_field/qe/qe_interface.cpp`,
`mean_field/bdft/`). Line anchors refer to the `paw` branch at 9a424c7 (plus
the D2 core-density fix in flight). Companion working inventories (per-dataset,
per-line): `notes/converter_h5_inventory_pw2coqui.md` /
`notes/converter_h5_inventory_abinit2coqui.md`. Normative statements are marked CONTRACT;
observed deviations are marked DEVIATION; recommendations live in the audit
report, not here.

---

## 0. Container models (the two routes are structurally different)

- **QE route (3 inputs)**: CoQui's `qe` backend reads (i) `data-file-schema.xml`
  (metadata), (ii) QE-native `wfc*.hdf5` (orbitals + per-k Miller), and
  (iii) `prefix.coqui.h5` from pw2coqui (`/System`, `/Orbitals` sans
  wavefunctions, `/Hamiltonian`). pw2coqui writes NO orbitals — the `add_orbs`
  namelist flag is dead. `/System` and `/Orbitals` of the coqui.h5 ARE consumed
  (qe_interface.cpp:310–492: species/positions/lattice/BZ kpoints, fft_mesh,
  fft_mesh_aug, npw, eigval, occ) — they duplicate XML content, with the h5
  authoritative for CoQui.
- **ABINIT route (self-contained bdft)**: abinit2coqui writes ONE file holding
  `/System` (+ full `/System/BZ` symmetry/q-point tables, nosym-only today),
  `/Orbitals` including `psi_s{is}_k{ik}` on a union wfc G-sphere, and
  `/Hamiltonian`. Read by the `bdft` backend.
- CONTRACT: a pw2coqui h5 is a *qe-backend companion*, never a standalone bdft
  file (it has no psi/miller_wfc/wfc_ecut). Feeding it to the bdft reader is a
  user error that today produces an HDF5 crash cascade (STATUS hardening 4b).

## 1. Global conventions

### 1.1 Units
- CONTRACT (`/Hamiltonian@schema_version = 2`, both converters): every
  energy-valued dataset under `/Hamiltonian` is in **Hartree on disk**:
  `dion[_so]`, Species `dion`, `ae_vloc`, `vloc_ps`, `pp_local_component[_nc]`,
  `scf_local_potential`, `vxc[_with_nlcc]`, `deltaC`, `ex_cvij`. Readers scale
  by `ry2ha = (schema_version >= 2 ? 1.0 : 0.5)` (pseudopot.cpp:359;
  add_vxc via `h5_pp_ry2ha`) — version <2 files are legacy-Ry.
- pw2coqui achieves this by explicit `/e2` at write (QE internals are Ry);
  `deltaC` needs `/e2²` because QE's `PAW_init_fock_kernel` returns e2²-scaled
  values (pw2coqui.f90:999–1011). abinit2coqui is Ha-native (no factors).
- `/System@nuclear_energy` (Ewald), `qe_*` attrs, `eigval`, `fermi_energy`,
  `exx_core_core`: Hartree.
- Lengths in Bohr; reciprocal in Bohr⁻¹; `kpoints` are Cartesian (QE ×tpiba;
  AB `kcrys @ recv`).
- DEVIATION (unresolved unit pin): `/Orbitals@ecutrho` — pw2coqui writes QE's
  `ecutrho` **in Ry, unconverted** (pw2coqui.f90:305); abinit2coqui writes
  `2·wfc_ecut` (no `--pot`) or `4·wfc_ecut` (with `--pot`) with `wfc_ecut` in
  Ha (abinit2coqui.py:507–513). The attribute's unit is therefore
  backend-dependent unless the AB factor 2 is read as a deliberate Ha→Ry
  conversion — the code does not say. Consumers include the THC ecut plumbing
  (`mfobj.ecutrho()`). Must be pinned explicitly in the next schema rev.

### 1.2 Complex storage
- CONTRACT: complex arrays are stored as float64 with a trailing (C-order)
  axis of length 2 plus a string dataset attribute `__complex__ = "1"`.
  Both converters conform (pw2coqui via Fortran leading dim 2 = C trailing;
  abinit2coqui `_w_complex`).

### 1.3 Index conventions
- 0-based on-disk names and ids: `atomic_id = typat−1`, group suffixes
  `s{i}`, `nt{i}`, `isp{i}`, `miller_k{ik}`, `projector_k{ik}`, `psi_s{is}_k{ik}`.
- 1-based Fortran-style channel tables: `ijtoh`, `indv`, `nhtolm`. CONTRACT
  (hard-checked at read, pseudopot.cpp:674–693): `ijtoh(nt,ih,jh)` is the QE
  `init_us_1` sequential upper-triangle packing `for ih: for jh>=ih: ++ij`,
  1-based, symmetric; only the active `nh(nt)` block is constrained (QE pads
  −1, AB pads 0).
- `proj_per_atom` is per-SPECIES (length nsp) despite the name — hard-checked
  (plan A5; pseudopot.cpp:527–533).
- Projector ordering within `projector_k`: atoms in `typat` order at offsets
  `projector_offset`; within an atom, species channel order × m (QE vkb
  layout). `nh(nt) = Σ_b (2 l_b + 1)`.

### 1.4 Angular/radial conventions
- Real spherical harmonics: QE `ylmr2` convention everywhere, including the
  odd-m sign structure (real pairs built from Condon–Shortley (−1)^m complex
  harmonics ⇒ odd-m components carry an extra minus). The abinit converter
  ports this twice: `real_ylm` (l ≤ 2, projector path; sign fix 3956b45) and
  a full recursive `ylmr2` port (Q(G)/Gaunt path). Any new Ylm consumer MUST
  match ylmr2 signs — channel-diagonal tests are blind to this; only
  off-diagonal (k,k−q) pair quantities expose it.
- Radial functions are stored in QE u-form, `u(r) = r·R(r)`: `beta`, `aewfc`,
  `pswfc`, `Core/ae_wfc`. `qfuncl` is the l-decomposed r²-weighted
  `Q^L_ij(r)`; `rab = dr/di` is the log-mesh measure; quadrature is QE
  `simpson` cut at `kkbeta` where applicable.
- Core densities `ae_rho_atc` / `rho_atc_ps` are **proper number densities**
  (no r², no √4π): AB divides the XML L=0-moment tables by √(4π)
  (abinit_paw_hamiltonian.py:236–239); QE copies UPF fields that are already
  in this convention.
- Projector G-space build (`projector_k`, and any native rebuild):
  `β(k+G) = (4π/√Ω) · (−i)^l · Y_lm^{ylmr2}(k+G) · [∫ u_β(r) j_l(|k+G|r) r dr] · e^{−i(k+G)·τ}`.
- Structure-factor sign: `e^{−iG·τ}` on atom-centered G-space objects
  (`pp_local_component`, compensation charge, `projector_k`).
- `augmentation_function_isp{nt}` = Q^IJ(G) per species with **no structure
  factor** (consumers add `e^{−iG·τ}`), dense-sphere order, QE `qvan2`
  convention: `Σ_lp (−i)^L ap(lp,ivl,jvl) Y_lp(Ĝ) qrad(|G|,ijv,L)`.

### 1.5 Occupations, weights, spin
- CONTRACT: `/Orbitals/occ` ∈ [0,1] per band per spin channel (QE writes
  `wg/wk`; AB divides ABINIT's occ ∈ [0,2] by 2 at nspin=1). CoQui applies the
  spin factor `n_s = 2/nspin` internally (`ns_scl` in v_h/becsum — plan I1).
  A converter writing occ that already includes spin degeneracy would silently
  double the density.
- `kpoint_weights` sums to 1 (QE ×0.5 at nspin=1; AB `wtk/Σwtk`). Written to
  `/System` (not `/System/BZ`).
- `eigval`/`occ` logical shape (nspin, nk, nbnd).

### 1.6 G-vector tables and meshes
- `miller_g` (dense): the ecutrho/dense **sphere**, not a box; the AB
  converter constructs the largest sphere inscribed in the POT FFT box.
  CONTRACT: every `miller_g` entry must be representable in `fft_mesh_aug`
  (single-wrap indexing). Note (pitfall): the current consumers
  (`compute_rho_aug_density_r`, `compute_int_VQ`) silently `continue` on
  out-of-box G's rather than aborting.
- `miller_k{ik}` per-k wfc sphere in the SOURCE code's own ordering; this
  ordering defines the coefficient order of `projector_k{ik}` (and, on the AB
  route, is mapped into the union `miller_wfc`). The set is recomputable, the
  order is not — the two datasets stand or fall together.
- `fft_mesh` (smooth/wfc) and `fft_mesh_aug` (dense/rho) may differ (AB
  split-mesh 14³/36³ exercises this; QE fixtures typically have them equal —
  historically a blind spot: mesh-mixup bugs are invisible on QE fixtures).

## 2. The D-matrix partition contract (plan I1–I5 → datasets)

The PAW/USPP non-local D decomposes as
`D = D_static + D_H_dynamic[n]`, with exchange NEVER in D (I4).

### 2.1 `D_static = dion + α_x·ex_cvij` (I2) — built once, never mutated
- `/Hamiltonian/{pp}/dion` is the **full frozen ionic D⁰, XC-free**:
  AE−PS kinetic difference + frozen-core-screened ionic Hartree terms
  (incl. the compensation-moment piece on the PS side), Eq. (d0) of the plan.
  - QE: `uspp%dvan` copied verbatim (÷e2). The UPF generator did the
    descreening; CoQui never re-derives it.
  - ABINIT: **assembled in the converter** (`assemble_dij0`,
    abinit_paw_hamiltonian.py:152–201): XML `<kinetic_energy_differences>` +
    `⟨u_ae u_ae|v_H[n_core]−Z/r⟩ − ⟨u_ps u_ps|v_H[ñ_Zc]⟩ −
    (∫v_H[ñ_Zc] g₀r²)·⟨u_ae²−u_ps²⟩`, same-l pairs, trapezoid to kkbeta,
    deliberately EXCLUDING ABINIT's frozen atomic V_xc1 (vbare, not vloc).
    Validated ≈4 digits against ABINIT's dumped D⁰.
  - CONTRACT: whatever the backend, `dion` must be this full frozen XC-free
    D⁰. CoQui adds nothing to it at read (`compute_paw_static_D` exists only
    as a cross-check tool; adding it double-counts — validated on LiH,
    paw_onecenter.hpp build_paw_scf_caches note).
- `ex_cvij` (frozen core–valence exact exchange, lm-diagonal, contracted with
  factor 1 through H₀/e_1e — never through ½Tr[DK]):
  - Detection order at read (pseudopot.cpp:907–946): `Onecenter/ex_cvij` if
    present → native build from `Core/ae_wfc` (`compute_ex_cvij_from_core`,
    plan B3; validated ~1e-7 vs ABINIT-XML) → absent with loud warning.
  - ABINIT writes `ex_cvij` (XML `<exact_exchange_X_matrix>`, ln→nh expansion
    gated on `nhtolm` equality); QE writes only `Core/` (GIPAW) and relies on
    the native builder. Two sources of truth — see audit report.
- `exx_core_core` (AB only): frozen core–core exchange constant, per-species
  attr + atom-summed `/System` attr. Additive constant to total energies;
  QE-sourced files have no counterpart.
- Static `H₀` additionally uses `pp_local_component` (frozen V_loc) and the
  frozen `∫V_loc·Q̂` term built at runtime (`static_h0_D`).

### 2.2 `D_H_dynamic[n]` (I3) — rebuilt from the current density at every
evaluation; **valence only**
- `D_H_a(I,J)[n] = ∫ v_H[ρ̄[n]]·Q̂_a,IJ + ΔD^H_a(I,J)[becsum[n]]` where ρ̄ is
  smooth + compensation density and ΔD^H is the radial AE−PS one-center
  valence Hartree (`compute_paw_hartree_atom`).
- CONTRACT (D2 resolution, 2026-07-25): NO frozen-core density enters the
  dynamic radial Hartree. The core–valence one-center electrostatics is inside
  `dion` (I2); injecting ρ_core here double-counts (+19.98 Ha V_H trace on the
  AB semicore fixture — invisible on QE fixtures whose species carry no core
  fields). The plan tex I3 parenthetical "frozen-core AE/PS core densities
  included" is superseded by this contract (flagged in the audit report).
- Inputs consumed: `qq_nt` (S-matrix / becsum charges), `augmentation_function`
  (∫V·Q term), `aewfc`/`pswfc`/`qfuncl`/`r`/`rab`/`kkbeta` + channel tables
  (radial ΔD^H), becsum per I1 (full-BZ lift, `n_s/N_k`).

### 2.3 Exchange (I4) — never in D
- Valence–valence: v_x / THC ERI with `moment` (Q̂ pair density + `deltaC`
  one-center, prefactor −1/N_k) or `shape` (full AE−PS form factors, deltaC
  dropped) modes.
- `Onecenter/deltaC(nh,nh,nh,nh)`: one-center Coulomb-kernel residual
  K_AE − K_PS in the projector-channel basis, proper Hartree.
  - QE: computed by `PAW_init_fock_kernel` at conversion, ÷e2².
  - ABINIT: computed by a python port of the same kernel (paw_deltaC.py) —
    NOT taken from the XML X-matrix (that is `ex_cvij`).
  - CoQui's radial machinery reproduces its Hartree contraction to 1e-7
    (LiH) / 1e-4 (Si — quadrature-limited); see redundancy discussion.
  - DEVIATION: quadrature domains differ — QE kernel integrates per its own
    convention, the AB port integrates the FULL mesh (kkbeta=None), CoQui's
    radial builder cuts at kkbeta. Part of the observed 1e-4 floor.

### 2.4 Backend parity (I6)
Same datasets, same units, same index conventions; the ONLY tolerated
per-backend divergence is dataset *presence* (`ex_cvij`, `Core/`,
`exx_core_core`, `madelung_constant`, `pp_local_component_nc`). A QE- and an
ABINIT-converted h5 for the same system must drive the identical CoQui code
path. Route-equivalence on an AB mf (D2 test matrix) is the acceptance gate.

## 3. Group-by-group expectations (consumer-verified)

Legend: REQ = reader hard-errors when absent (USPP/PAW), OPT = optional with
defined fallback, DEAD = written today, consumed by nothing in CoQui.

### `/System` (+`/System/BZ`)
Consumed by qe_interface (QE route) and bdft_system (AB route): species,
atomic_id, atomic_positions, lattice/reciprocal_vectors, kpoint_weights,
BZ kpoints (+ on the bdft route: kp_grid, kpoints_crystal, kp_symm/kp_to_ibz/
kp_trev tables, q-point tables, Symmetries). Attrs: electron/spin counts,
`nuclear_energy`, `fermi_energy`; `madelung_constant` (AB only — bdft_system
computes it natively when absent/0); `qe_ehart/qe_etxc/qe_vtxc/qe_epaw`
(QE only; reference-energy provenance, no production consumer);
`exx_core_core` (AB only). AB writes `nuclear_energy = 0.0` + warning when
run without `--psp8`/`--pawxml` — a trap for partial conversions.

### `/Orbitals`
- Both routes: `fft_mesh`, `fft_mesh_aug` (OPT on QE route with warning),
  `npw`, `eigval` (Ha), `occ` (§1.5), `ecutrho` (§1.1 DEVIATION).
- bdft route only: `wfc_ecut` (Ha), `wfc_fft_grid`, `wfc_ngm`, `miller_wfc`
  (union sphere), `psi_s{is}_k{ik}` (nbnd × ngm_wfc, no rescaling).

### `/Hamiltonian` and `/Hamiltonian/{pp_type}`
- Attrs: `schema_version` (REQ semantics, §1.1), `pp_type`; counts
  (`number_of_*`, `total_num_of_proj`, `max_proj_per_atom`, `ngm`,
  `max_npw`), `lspinorbit_nl/loc`.
- REQ datasets (all pp types): `miller_g`, `scf_local_potential`,
  `pp_local_component` (`_nc` variant when `lspinorbit_loc`), `proj_per_atom`,
  `projector_offset`, `npw`, `atomic_id`, `dion` (`dion_so` for SOC),
  `miller_k{ik}`, `projector_k{ik}`.
- REQ USPP/PAW: `ijtoh`, `qq_nt`, `augmentation_function_isp{nt}`.
- `vxc_with_nlcc`: consumed by `add_vxc` (GW@DFT vxc subtraction).
  `vxc` (core-zeroed variant): DEAD.
  AB writes vxc datasets as zeros without `--den` (PAW) or omits them (NC) —
  zeros are indistinguishable from "XC-free source"; see audit report.
- `qq_so` (QE SOC), `deeq`: deeq is deliberately NOT exported (I2/I3 —
  QE-screened deeq carries V_xc and the QE valence Hartree; CoQui builds its
  own). Any file still carrying `deeq` is legacy (schema <1).

### `/Hamiltonian/Species/nt{i}`
- REQ attrs: `mesh`, `nbeta`, `kkbeta`, `nh`; REQ datasets: `r`, `rab`,
  `lll`, `nhtol`, `nhtolm`, `indv`, `qfuncl` (aug species).
- PAW: `aewfc`, `pswfc` (partial waves; REQ for the one-center machinery);
  `/paw` attrs `lmax_aug`, `raug` (AB writes the −1.0 sentinel — use
  `iraug`), `iraug`; `ae_vloc`, `vloc_ps` (REQ at schema ≥2 per B4 — but see
  §4/asymmetry: semantic content differs between backends and the only
  consumer is out of production); `ae_rho_atc`, `rho_atc_ps` (silent-OPT;
  post-D2 no production consumer).
- `/Onecenter`: `deltaC` (OPT; §2.3), `ex_cvij` (OPT; §2.1).
- `/Core`: `ncore_orbitals`, `n` (f64), `l` (f64), `ae_wfc` (u-form) —
  OPT; feeds native ex_cvij (and nothing else since the D2 fix).
- DEAD at species level (written by one or both converters, consumed by
  nothing): `beta`, `kbeta`, `qqq`, species-level `dion` (duplicate of the
  channel-level one), `qfunc`, `q_with_l`, `nqf`, `nqlc`, `lmax`, `lmax_rho`,
  `zp`, `augshape`, `oc`, `pfunc`, `ptfunc`, `augmom`, `jjj`/`nhtoj`/
  `pfunc_rel`/`aewfc_rel` (SOC futures). `beta` is dead TODAY but is the
  enabler for a native projector build (audit report recommendation).

## 4. Known backend asymmetries (beyond dataset presence)

1. `dion` provenance: QE verbatim-from-UPF vs AB converter-assembled
   (≈4-digit agreement vs ABINIT's own D⁰) — a precision and convention
   surface that exists only on the AB side.
2. `ae_vloc`/`vloc_ps` SEMANTICS differ: QE copies UPF fields (which carry
   the generator's frozen atomic XC screening); AB writes XC-free
   reconstructions (`vhnzc`/`vhtnzc`) consistent with its dion assembly.
   Harmless today (consumer `compute_paw_static_D` is out of production) but
   the B4 schema requires the datasets — a semantic trap.
3. `ex_cvij` source: stored (AB) vs native-from-Core (QE). Native validated
   against the AB-stored convention to ~1e-7.
4. `deltaC` quadrature domain (full mesh AB vs QE-kernel convention vs
   CoQui-radial kkbeta).
5. `/System/BZ`: full nosym symmetry/q tables (AB) vs minimal (QE; symmetry
   from XML; `t_rev` never exported by pw2coqui).
6. `madelung_constant`, `exx_core_core`, `pp_local_component_nc`(NC),
   `psi/miller_wfc/wfc_*`: single-backend datasets.
7. `pp_local_component` G=0/tail handling: AB uses the Coulomb/alpha split
   with Qtail read off the r·v plateau (Si: 4.18 vs Zval 4); QE FFTs QE's
   vltot (QE handled the split internally).
8. `vxc`/`vxc_with_nlcc`: QE recomputes with QE's own functional stack
   (meta-GGA skipped); AB recomputes with its own PW92/PBE (+ finite-diff
   derivatives) or writes zeros/omits without `--den`.
9. AB projector-path `real_ylm` hard-fails for l>2; QE `init_us_2` is
   general. (CoQui's synthetic l=3 aug test does not cover AB-converted
   f-projectors.)
10. Converter-side wart (QE): `lspinorbit_loc` attribute written outside the
    ionode guard (pw2coqui.f90:616); harmless serially.

## 5. Validated recomputability (basis for any future slimming)

| stored quantity | native CoQui recomputation | validation |
|---|---|---|
| `augmentation_function_isp*` | `evaluate_Q` (qrad + ylmr2 + Gaunt) | 2e-11 vs stored, incl. AB fixture (paw_aug_q_eval_at_q0) |
| `deltaC` (Hartree contraction) | `compute_paw_hartree_atom` from aewfc/pswfc/qfuncl | 1e-7 (LiH) / 1e-4 (Si, quadrature) |
| `ex_cvij` | `compute_ex_cvij_from_core` from Core/ae_wfc | ~1e-7 vs ABINIT XML; 3e-11 vs hydrogenic analytics |
| `qq_nt` | L=0 moment of qfuncl (= what the AB converter itself does) | AB converter path is already this computation |
| `projector_k`/`miller_k` | NOT yet — needs a native init_us_2 analog from `beta` (currently DEAD data) | converter-side python build validated at 5.6e-9 Ha eigenvalues |

Everything in the DEAD list (§3) is removable with no CoQui-side change at
all. Recommendations, priorities, and schema-v3 proposals: see the audit
report (chat deliverable, 2026-07-25) — this document only records the
contract as it stands.
