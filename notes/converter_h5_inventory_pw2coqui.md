# pw2coqui.f90 → CoQui HDF5 output: complete write inventory

Source: `/Users/mmorales/Software/CoQui_Separate_Development/qe_converter/pw2coqui.f90` (1431 lines, branch `paw`, HEAD 9a424c7).
All line anchors below refer to that file unless another file is named.

## 0. Program flow, output file, and gating flags

- Output file name: `TRIM(tmp_dir)//TRIM(fname)`; default `fname = TRIM(prefix)//'.coqui.h5'` (line 61); opened `'write'` on ionode only (lines 82–85), closed at line 113.
- Namelist `input_pw2coqui` (line 41): `prefix, outdir, fname, add_system, add_pp, add_orbs`.
  - `add_system = .true.` default (line 51) → gates `write_system` (line 107) = everything under `/System` and `/Orbitals`.
  - `add_pp = .true.` default (line 52) → gates `write_pp` + `write_species` (lines 108–111) = everything under `/Hamiltonian`.
  - `add_orbs = .false.` default (line 53): read and broadcast (line 70) but **never used** — this converter writes NO wavefunctions/orbitals datasets; CoQui reads the QE `wfc*.dat`/`.hdf5` files directly.
- Global helper quantities computed in the main program (lines 89–104): `npw_g(nkstot)` = per-k global PW count from `ngk`, pool-reduced and `/nbgrp` (line 102); `npwx_g = MAXVAL(npw_g)` (line 103); `maxg` = max global G index from `ig_l2g(igk_k)` (line 97).
- `e2` is QE `constants` `e2 = 2.0` (Ry: e²=2), so every `/e2` below is an exact Ry→Hartree conversion.

### On-disk layout conventions (writer helpers, lines 1044–1365)

- Scalar-typed writers `h5_write_vector_{int,r}`, `h5_write_mat_{int,r}`, `h5_write_tensor_{int,r}`, `h5_write_tensor4_r` pass the Fortran dims to `qeh5_set_space` (e.g. lines 1056, 1118, 1182, 1246–1247). The HDF5 Fortran API reverses axis order on disk, so a dataset declared `[d1,d2,...]` in Fortran appears as `(...,d2,d1)` to C/h5py. Dims quoted below are the **Fortran dims as passed**.
- Complex writers `h5_write_vector_c` / `h5_write_mat_c` / `h5_write_tensor_c` / `h5_write_tensor4_c` (lines 1084–1104, 1146–1168, 1210–1230, 1254–1274) store complex as real with an extra **leading Fortran dim 2** (fastest index → h5py sees trailing axis 2), e.g. `DIMENSIONS=[2,size(v,1),size(v,2)]` (line 1160), then stamp a string attribute `__complex__ = "1"` on the dataset via `add_complex` (lines 1276–1318). Every complex dataset below therefore carries this attribute.
- `h5_write_tensor4_r` uses raw `H5Dwrite_f` via `write_real_data_raw` (lines 1232–1252, 1320–1343) because `qeh5_write_dataset` only overloads rank ≤ 3.
- No conjugation, sign flips, or explicit transposes are applied anywhere in this file; the only value transformations are the `/e2`, `/(e2*e2)`, `×alat`, `×tpiba`, `×0.5` (weights), `wg/wk`, `ityp−1` operations quoted per-item below.

---

## 1. Group `/System` (subroutine `write_system`, lines 130–355; ionode-only writes; gated by `add_system`)

Opened line 181. Note: `v_of_rho` (line 213) and `PAW_potential` (line 216) are called on **all ranks** (outside the ionode guard) to refresh the stale `ener`-module globals before the `qe_*` attributes are written; `epaw = 0` when not `okpaw` (lines 215–219).

### Attributes on `/System`

| Attribute | QE source | Type/conv | Conditional | Line |
|---|---|---|---|---|
| `number_of_atoms` | `ions_base%nat` | int, verbatim | always | 186 |
| `number_of_species` | `ions_base%nsp` | int, verbatim | always | 187 |
| `number_of_spins` | `ns = 2` if `lsda` else 1 (lines 178–179) | int, computed | always | 188 |
| `number_of_polarizations` | `noncollin_module%npol` | int, verbatim | always | 190 |
| `number_of_elec` | `klist%nelec` | real, verbatim | always | 192 |
| `noinv` | `control_flags%noinv` | logical→int 1/0 | always | 193–197 |
| `lspinorbit` | `noncollin_module%lspinorb` | logical→int 1/0 | always | 198–202 |
| `nuclear_energy` | `ewald(alat,nat,...,strf)` (QE Ewald function) | Ry→Ha: `enuc = ewald(...)/e2` (lines 204–205) | always | 204–206 |
| `qe_ehart` | `ener%ehart` (recomputed by `v_of_rho`, line 213) | `ehart/e2` (Ry→Ha) | always | 225 |
| `qe_etxc` | `ener%etxc` (recomputed) | `etxc/e2` | always | 226 |
| `qe_vtxc` | `ener%vtxc` (recomputed) | `vtxc/e2` | always | 227 |
| `qe_epaw` | `ener%epaw` via `PAW_potential(rho%bec, ddd_paw, epaw)` (line 216); 0 if not PAW | `epaw/e2` | always (value 0 for non-PAW) | 228 |

### Datasets in `/System`

| Dataset | QE source | Fortran dims | Conversion (exact line) | Conditional | Lines |
|---|---|---|---|---|---|
| `lattice_vectors` | `cell_base%at`, `alat` | (3,3) real | `v3(:,:) = at(:,:)*alat` (230) → columns aᵢ in Bohr | always | 230–231 |
| `reciprocal_vectors` | `cell_base%bg`, `tpiba` | (3,3) real | `v3(:,:) = bg(:,:)*tpiba` (232) → columns bᵢ in Bohr⁻¹ | always | 232–233 |
| `atomic_id` | `ions_base%ityp` | (nat) int | `ityp_s(i) = ityp(i)-1` (237): 0-based species index per atom | always | 235–239 |
| `atomic_positions` | `ions_base%tau`, `alat` | (3,nat) real | `vn(1:3,1:nat) = tau(1:3,1:nat)*alat` (242) → Cartesian Bohr | always | 241–243 |
| `species` | `ions_base%atm(1:nsp)` | (nsp) fixed-len-4 strings | `atm_(i) = TRIM(atm(i))//char(0)` (250); hand-built H5T_FORTRAN_S1 UTF-8 null-terminated type (252–270) | always | 246–272 |
| `kpoint_weights` | `klist%wk` | (nk = nkstot/ns) real | `wgt = 0.5d0*wk` if `nspin==1`, else `wgt = wk` (296–300) — collinear-unpolarized weights renormalized so Σ=1 | always | 295–301 |

Note: `kpoint_weights` is written to `/System` (handle `h5_s`, line 301), **not** `/System/BZ`.

---

## 2. Group `/System/BZ` (opened line 183)

### Attributes

| Attribute | QE source | Notes | Line |
|---|---|---|---|
| `number_of_kpoints` | `start_k`: `nk1*nk2*nk3` if > 0, else `nks_start` (279–283) | full-grid k count (may differ from IBZ) | 284 |
| `number_of_kpoints_ibz` | `nk = nkstot/ns` (286) | IBZ count | 287 |

### Datasets

| Dataset | QE source | Fortran dims | Conversion | Conditional | Lines |
|---|---|---|---|---|---|
| `kp_grid` | `start_k%nk1,nk2,nk3` | (3) int | verbatim; may be all-zero (comment line 274: CoQui infers the grid) | always | 275–278 |
| `kpoints` | `klist%xk` | (3,nk) real | `vn(:,1:nk) = xk(:,1:nk)*tpiba` (292) → Cartesian Bohr⁻¹ | always | 290–293 |

---

## 3. Group `/System/BZ/Symmetries` (opened line 184) and per-symmetry subgroups

- Attribute `number_of_symmetries` = `symm_base%nsym`, line 338.
- Per symmetry i = 1..nsym, subgroup `s{i-1}` (0-based name, e.g. `/System/BZ/Symmetries/s0`), opened line 341:

| Dataset | QE source | Fortran dims | Conversion | Lines |
|---|---|---|---|---|
| `R` | `symm_base%s(1:3,1:3,i)` (INTEGER rotation matrix, crystal axes) | (3,3) real | `v3(:,:) = s(1:3,1:3,i)` (342) — int→real cast, otherwise verbatim | 342–343 |
| `ft` | `symm_base%ft(1:3,i)` (fractional translation, crystal coords) | (3) real | verbatim | 344 |

`symm_base%t_rev` is imported (line 147) but **not written**.

---

## 4. Group `/Orbitals` (opened line 182; written inside `write_system`)

### Attributes

| Attribute | QE source | Notes | Line |
|---|---|---|---|
| `number_of_spins` | `ns` | 1/2 | 189 |
| `number_of_polarizations` | `npol` | | 191 |
| `number_of_kpoints` | full-grid count (see §2) | | 285 |
| `number_of_kpoints_ibz` | `nkstot/ns` | | 288 |
| `npwx` | `npwx_g` (computed, main lines 89–103) | global max PW per k | 289 |
| `number_of_bands` | `wvfct%nbnd` | | 304 |
| `ecutrho` | `gvect%ecutrho` | **Ry, no conversion** | 305 |

### Datasets

| Dataset | QE source | Fortran dims | Conversion (exact line) | Conditional | Lines |
|---|---|---|---|---|---|
| `npw` | computed `npw_g(1:nk)` (main lines 89–103) | (nk) int | verbatim | always | 308 |
| `fft_mesh` | `fft_base%dffts%nr1,nr2,nr3` (smooth/wfc grid) | (3) int | verbatim | always | 311–314 |
| `fft_mesh_aug` | `fft_base%dfftp%nr1,nr2,nr3` (dense/rho grid) | (3) int | verbatim | always | 315–318 |
| `eigval` | `wvfct%et(nbnd,nkstot)` | rank-3 `[nbnd,nk,ns]` (spin slowest via nkstot=nk·ns flattening) | `vn(:,:) = et(:,:)/e2` (322) — Ry→Ha | always | 320–326 |
| `occ` | `wvfct%wg`, `klist%wk` | rank-3 `[nbnd,nk,ns]` | `vn(:,i) = wg(:,i)/wk(i)` (330) — k-weight divided out ⇒ pure occupation numbers, dimensionless | always | 328–336 |

---

## 5. Group `/Hamiltonian` (subroutine `write_pp`, lines 358–831; gated by `add_pp`)

Opened line 421. `pp_type` selection lines 413–419: `"paw"` if `okpaw`, else `"uspp"` if `okvan`, else `"ncpp"`.

### Attributes on `/Hamiltonian`

| Attribute | Value | Line |
|---|---|---|
| `schema_version` | **2** (integer literal) | 429 |
| `pp_type` | string `"paw"`/`"uspp"`/`"ncpp"` | 430 |

Schema contract (comment lines 422–428): *(absent)* = legacy (QE deeq, Ry); *1* = deeq-free, still Ry on disk; *2* = deeq-free **and Hartree on disk** for all `/Hamiltonian` energy-valued datasets (`dion[_so]`, Species `dion`, `ae_vloc`, `vloc_ps`, `pp_local_component`, `scf_local_potential`, `vxc[_with_nlcc]`). Readers apply the 0.5 Ry→Ha scale only for version < 2.

---

## 6. Group `/Hamiltonian/{pp_type}` (i.e. `/Hamiltonian/paw`, `/Hamiltonian/uspp`, or `/Hamiltonian/ncpp`; opened line 431)

### Attributes

| Attribute | QE source | Conditional | Line |
|---|---|---|---|
| `number_of_nspins` | `ns` (1/2) | always | 433 |
| `number_of_polarizations` | `npol` | always | 434 |
| `number_of_kpoints` | `nk = nkstot/ns` (409) | always | 435 |
| `number_of_atoms` | `nat` | always | 436 |
| `number_of_species` | `nsp` | always | 437 |
| `total_num_of_proj` | `uspp%nkb` | always | 438 |
| `max_proj_per_atom` | `uspp_param%nhm` | always | 439 |
| `ngm` | `gvect%ngm_g` (global G count) | always | 443 |
| `lspinorbit_nl` | `lspinorb` → 1/0 | always | 448–452 |
| `max_npw` | `npwx_g` | always | 455 |
| `lspinorbit_loc` | hard-coded 0 (comment line 615: change if local potential becomes polarization-dependent) | always | 616 (NB: this call sits **outside** the `if(ionode)` guard, unlike every other attribute — latent non-ionode hazard, harmless in serial) |

### Datasets — projector bookkeeping and D/q matrices (ionode block, lines 411–480)

| Dataset | QE source | Fortran dims | Conversion (exact line) | Conditional | Lines |
|---|---|---|---|---|---|
| `proj_per_atom` | `uspp_param%nh(1:nsp)` (projectors **per species**, despite the name) | (nsp) int | verbatim | always | 440 |
| `projector_offset` | `uspp%ofsbeta` (per-atom offset into the nkb projector list) | (nat) int | verbatim | always | 441 |
| `ijtoh` | `uspp%ijtoh` (packed (ih,jh)→ijh map) | (nhm,nhm,nsp) int | verbatim | always | 442 |
| `atomic_id` | `ityp` | (nat) int | `ityp_s(:) = ityp(:)-1` (445): 0-based | always | 444–446 |
| `npw` | `npw_g(1:nk)` (computed) | (nk) int | verbatim (duplicate of `/Orbitals/npw`) | always | 456 |
| `dion_so` | `uspp%dvan_so` (bare/descreened D, spin-orbit) | complex (nhm,nhm,nspin,nsp) → on-disk rank-5 `[2,...]` | `dvan_so_ha = dvan_so / e2` (461) — Ry→Ha, schema 2 | `lspinorb` only | 458–463 |
| `qq_so` | `uspp%qq_so` (spinor-coupled augmentation overlap ∫Q) | complex (nhm,nhm,4,nsp) → rank-5 `[2,...]` | verbatim (dimensionless charge) | `lspinorb .and. allocated(qq_so)` | 464–465 |
| `dion` | `uspp%dvan` (bare/descreened D-matrix; species-resolved) | (nhm,nhm,nsp) real | `dvan_ha = dvan / e2` (469) — Ry→Ha, schema 2 | `.not.lspinorb` | 466–471 |
| `qq_nt` | `uspp%qq_nt` (species-resolved q_ij = ∫Q_ij; S = 1 + Σ|β⟩q⟨β|) | (nhm,nhm,nsp) real | verbatim | `allocated(qq_nt)` (i.e. USPP/PAW; also allocated for NCPP in QE but ≈0) | 473–474 |

Deliberate omission (comment lines 475–479): QE's SCF-screened `deeq`/`deeq_nc` are **NOT** exported — CoQui builds the density-dependent D natively (plan I2/I3; would double-count and carry V_xc).

### Datasets — augmentation functions Q_ij(G) (COMPUTED; lines 482–544)

| Dataset | Source | Fortran dims | Conversion | Conditional | Lines |
|---|---|---|---|---|---|
| `augmentation_function_isp{nt-1}` (one per augmented species, 0-based name, e.g. `augmentation_function_isp0`) | **Computed**: `qvan2(ngm_l, ih, jh, nt, qmod, qgm, ylmk0)` (513) with `ylmk0` from `ylmr2` (488) and `qmod = SQRT(gg)*tpiba` (490); assembled into global-G order via `ig_l2g` (515) and `mp_sum` over `world_comm` (529). **No structure factor** (comment 503–504). | complex (ngm_g, nij), nij = nh(nt)·(nh(nt)+1)/2 packed (ih≤jh) pairs (500) → rank-3 on disk `[2,ngm_g,nij]` | verbatim (no scaling; Q(G) dimensionless Fourier moments) | `okvan` outer (482); per species `upf(nt)%tvanp` (496) — true for both USPP and PAW species | 482–544, write at 531–533 |

(Commented-out sibling `r_augmentation_function_isp*` — real-space Q — lines 518–524, 534: dead code, not written.)

### Datasets — G-vector tables and local potentials (lines 550–723)

All FFTs use `fwfft('Rho', psic, dfftp)` on the **dense** grid; results scattered to global-G order via `ig_l2g` and `mp_sum(intra_bgrp_comm)`; writes ionode-only.

| Dataset | QE source | Fortran dims | Conversion (exact line) | Conditional | Lines |
|---|---|---|---|---|---|
| `miller_g` | `gvect%mill` gathered to global order via `ig_l2g` (552–556), `mp_sum` (557) | (3, ngm_g) int | verbatim (Miller indices) | always | 550–557, write 599 |
| `scf_local_potential` | `scf%v%of_r + scf%vltot` (total local KS potential incl. Hartree+XC+local PP), FFT→G. Collinear: `psic(ir) = v%of_r(ir,is) + vltot(ir)` (589), stored `vloc(ig_l2g(ig),1,is)` (593). Noncollinear (nspin=4): 2×2 spin-matrix packing — is=1: `v1+v4+vltot` (568), is=2: `v2 − i·v3` (572), is=3: `v2 + i·v3` (576), is=4: `v1−v4+vltot` (580), stored `vloc(ig_l2g(ig),is,1)` (585) | complex (ngm_g, npol·npol, ns) → rank-4 `[2,ngm_g,npol²,ns]` | `vloc = vloc / e2` — “Hartree (schema 2)” (600) | always | 560–602 |
| `pp_local_component` | `scf%vltot` (bare local pseudopotential, includes NLCC-free ionic local part), FFT→G (607–613) | complex (ngm_g) → rank-2 `[2,ngm_g]` | `vloc = vloc / e2` — “Hartree (schema 2)” (619) | always | 604–621 (comment 622–623: a future `pp_local_component_nc` (ngm,npol²,nspin) is reserved, not written) |
| `vxc_with_nlcc` | recomputed `v_xc(rho, rho_core, rhog_core, etxc, vtxc, vxc)` (631) **with** core density (NLCC included), FFT→G with same collinear/noncollinear packing as above (633–667) | complex (ngm_g, npol·npol, ns) → rank-4 `[2,...]` | `vloc = vloc / e2` (670) | **skipped for meta-GGA** (`xclib_dft_is('meta')`, 626–627 prints “Skipping v_xc”); otherwise always | 626–672 |
| `vxc` | recomputed `v_xc` after `rho_core(:) = 0`, `rhog_core(:) = 0` (676–678) — valence-only XC potential (destructively zeroes the in-memory core density) | complex (ngm_g, npol·npol, ns) → rank-4 `[2,...]` | `vloc = vloc / e2` (717) | skipped for meta-GGA | 674–719 |

### Datasets — per-k-point tables (loop `do ik = 1, nk`, lines 728–820)

| Dataset | QE source | Fortran dims | Conversion | Conditional | Lines |
|---|---|---|---|---|---|
| `miller_k{ik-1}` (one per k, 0-based, e.g. `miller_k0`) | global igk reconstruction: `itmp(l2g)=l2g` from `ig_l2g(igk_k)` (736–742), `mp_sum` + `/nbgrp` (743–744), sorted ascending-global-G `igk_g` (746–752, consistency check 753); `mill_k(:,ig) = mill_g(:,igk_g(ig))` (757) | (3, npw_g(ik)) int | verbatim | always (every pp_type) | 728–761 |
| `projector_k{ik-1}` (one per k, 0-based) | **Computed**: `init_us_2(npw, igk_k(1,ik_loc), xk(1,ik), vkb)` (741) → β_{κ}(k+G) tables (all nkb projectors, QE ordering: atoms via ofsbeta, structure phase e^{−i(k+G)·τ} included by init_us_2); gathered to global-G order with `mergewf` (795–796 pooled / 804–805 serial) and pool transfer `mp_get` (799–800) | complex (npw_g(ik), nkb) → rank-3 `[2,npw_g(ik),nkb]` | verbatim (QE vkb normalization, no scaling) | always (dataset exists even for NCPP; second dim 0 if nkb=0) | 763–814 |

---

## 7. Group `/Hamiltonian/Species` (subroutine `write_species`, lines 834–1041; ionode-only, early return line 862; gated by `add_pp`)

`/Hamiltonian` reopened idempotently (872–874); `Species` opened 875. `PAW_init_fock_kernel()` called once up front if any species is PAW (864–870); `PAW_clean_fock_kernel()` at 1039.

### 7.1 Per-species group `/Hamiltonian/Species/nt{nt-1}` (0-based name, e.g. `nt0`; opened 877–879)

Written for **every** species regardless of kind.

Attributes:

| Attribute | QE source | Conditional | Line |
|---|---|---|---|
| `species_kind` | string `"paw"` if `upf(nt)%tpawp`, `"uspp"` if `tvanp`, else `"ncpp"` | always | 882–888 |
| `mesh` | `rgrid(nt)%mesh` (`atom` module, aliased `g`) | always | 890–891 |
| `kkbeta` | `upf(nt)%kkbeta` | always | 892 |
| `lmax` | `upf(nt)%lmax` | always | 893 |
| `lmax_rho` | `upf(nt)%lmax_rho` | always | 894 |
| `nbeta` | `upf(nt)%nbeta` | always | 895 |
| `nh` | `uspp_param%nh(nt)` | always | 896 |
| `zp` | `upf(nt)%zp` (valence charge) | always | 897 |
| `q_with_l` | `merge(1,0,upf(nt)%q_with_l)` | `tvanp .or. tpawp` | 926–927 |
| `nqf` | `upf(nt)%nqf` | `tvanp .or. tpawp` | 928 |
| `nqlc` | `upf(nt)%nqlc` | `tvanp .or. tpawp` | 929 |

Datasets:

| Dataset | QE source | Fortran dims | Conversion | Conditional | Lines |
|---|---|---|---|---|---|
| `r` | `rgrid(nt)%r(1:mesh)` (radial grid, Bohr) | (mesh) real | verbatim | always | 900 |
| `rab` | `rgrid(nt)%rab(1:mesh)` (dr/di weights) | (mesh) real | verbatim | always | 901 |
| `lll` | `upf(nt)%lll(1:nbeta)` (l of each β) | (nbeta) int | verbatim | always | 904 |
| `kbeta` | `upf(nt)%kbeta(1:nbeta)` (cutoff grid index per β) | (nbeta) int | verbatim | always | 905 |
| `beta` | `upf(nt)%beta(1:mesh,1:nbeta)` (r·β(r), UPF convention, Ry-based) | (mesh,nbeta) real | verbatim (**no** /e2) | always | 906 |
| `dion` | `upf(nt)%dion(1:nbeta,1:nbeta)` (bare/frozen UPF D⁰ in β-channel basis) | (nbeta,nbeta) real | `dion_ha = upf(nt)%dion(...)/e2` — “Ha (schema 2)” (908) | always | 907–909 |
| `nhtolm` | `uspp%nhtolm(1:nh(nt),nt)` (ih→combined lm) | (nh) int | verbatim | `allocated(nhtolm)` | 913–914 |
| `nhtol` | `uspp%nhtol(1:nh(nt),nt)` (ih→l) | (nh) int | verbatim | `allocated(nhtol)` | 915–916 |
| `indv` | `uspp%indv(1:nh(nt),nt)` (ih→β channel) | (nh) int | verbatim | `allocated(indv)` | 917–918 |
| `nhtoj` | `uspp%nhtoj(1:nh(nt),nt)` (ih→total j) | (nh) real | verbatim | `lspinorb .and. allocated(nhtoj)` | 919–920 |
| `qqq` | `upf(nt)%qqq(1:nbeta,1:nbeta)` (∫Q_ij charges) | (nbeta,nbeta) real | verbatim | `tvanp .or. tpawp` | 923–925 |
| `qfuncl` | `upf(nt)%qfuncl` (l-decomposed r²·Q_ij^l(r)) | rank-3, full allocated dims (mesh, nbeta(nbeta+1)/2, 2·lmax+1 as stored in UPF) | verbatim | `(tvanp.or.tpawp) .and. allocated(qfuncl)` | 930–931 |
| `qfunc` | `upf(nt)%qfunc` (l-independent r²·Q_ij(r), legacy USPP) | rank-2, full allocated dims | verbatim | `(tvanp.or.tpawp) .and. allocated(qfunc)` | 932–933 |
| `jjj` | `upf(nt)%jjj(1:nbeta)` (j of each β) | (nbeta) real | verbatim | `lspinorb .and. allocated(jjj)` | 937–938 |
| `aewfc` | `upf(nt)%aewfc(1:mesh,1:nbeta)` (AE partial waves, r·φ(r)) | (mesh,nbeta) real | verbatim | `(tpawp .or. has_wfc) .and. allocated(aewfc)` — PAW always; USPP only if generated `--with-ae-wfc` | 941–944 |
| `pswfc` | `upf(nt)%pswfc(1:mesh,1:nbeta)` (PS partial waves, r·φ̃(r)) | (mesh,nbeta) real | verbatim | same as `aewfc` | 945–947 |

### 7.2 Subgroup `/Hamiltonian/Species/nt{n}/paw` (PAW species only, `upf(nt)%tpawp`; opened 952)

Attributes: `raug` (953), `iraug` (954), `lmax_aug` (955), `augshape` string `TRIM(upf(nt)%paw%augshape)` (956–957) — all verbatim from `upf(nt)%paw`.

| Dataset | QE source | Fortran dims | Conversion | Conditional | Lines |
|---|---|---|---|---|---|
| `pfunc` | `upf(nt)%paw%pfunc` (AE products φ_iφ_j·r² incl.) | rank-3 (mesh,nbeta,nbeta) | verbatim | `allocated(pfunc)` | 959–960 |
| `ptfunc` | `upf(nt)%paw%ptfunc` (PS products) | rank-3 | verbatim | `allocated(ptfunc)` | 961–962 |
| `augmom` | `upf(nt)%paw%augmom` (multipole moments of AE−PS pair densities) | rank-3 (nbeta,nbeta,0:2l) | verbatim | `allocated(augmom)` | 963–964 |
| `ae_vloc` | `upf(nt)%paw%ae_vloc` (AE local/−Z_ae potential on radial grid, Ry) | (mesh) real | `v_ha = upf(nt)%paw%ae_vloc / e2` — “Ha (schema 2)” (966) | `allocated(ae_vloc)` | 965–969 |
| `ae_rho_atc` | `upf(nt)%paw%ae_rho_atc` (AE core density) | (mesh) real | verbatim | `allocated(ae_rho_atc)` | 970–971 |
| `oc` | `upf(nt)%paw%oc` (partial-wave occupations) | (nbeta) real | verbatim | `allocated(oc)` | 972–973 |
| `vloc_ps` | `upf(nt)%vloc` (**top-level** UPF radial local channel, Ry; co-located here because only the PAW path consumes it — comment 974–980; needed for CoQui `paw_init_keeq` static D⁰ without QE `ddd_paw`) | (mesh) real | `v_ha = upf(nt)%vloc / e2` — “Ha (schema 2)” (982) | `allocated(upf%vloc)` | 981–985 |
| `rho_atc_ps` | `upf(nt)%rho_atc` (top-level UPF smooth NLCC core density) | (mesh) real | verbatim | `allocated(rho_atc)` | 986–987 |
| `pfunc_rel` | `upf(nt)%paw%pfunc_rel` (small-component AE products) | rank-3 | verbatim | `lspinorb .and. allocated` | 990–992 |
| `aewfc_rel` | `upf(nt)%paw%aewfc_rel` (small-component AE waves) | rank-2 | verbatim | `lspinorb .and. allocated` | 993–994 |

### 7.3 Subgroup `/Hamiltonian/Species/nt{n}/Onecenter` (PAW species only; opened 1005)

| Dataset | Source | Fortran dims | Conversion (exact) | Lines |
|---|---|---|---|---|
| `deltaC` | **Computed**: raw `ke(nt)%k` from `paw_exx%PAW_init_fock_kernel` (called 870) — one-center Coulomb residual ΔC = K_AE − K_PS in the partial-wave quadruplet basis | rank-4, dims = `SIZE(ke(nt)%k, 1..4)` at write time (1006–1007) | `deltaC_Ha = ke(nt)%k / (e2*e2)` (1008). Rationale comment (999–1004): QE returns ke%k in (e2)²×Ha (paw_exx.f90:388 ×e2 on top of paw_onecenter.f90:1204 pre-×e2), so ÷e2² gives proper Hartree on disk | 999–1011 |

(Comment lines 546–548 in `write_pp`: local-channel fits are deliberately NOT computed/written; consumers build them from the raw `ke%k` tensor.)

### 7.4 Subgroup `/Hamiltonian/Species/nt{n}/Core` (GIPAW core orbitals; opened 1019)

Conditional: `upf(nt)%has_gipaw .and. upf(nt)%gipaw_ncore_orbitals > 0` (1017) — requires pseudo generated `--with-gipaw` (comment 1014–1016); needed for explicit core-valence/core-core exchange.

- Attribute `ncore_orbitals` = `upf(nt)%gipaw_ncore_orbitals` (1020).

| Dataset | QE source | Fortran dims | Conversion | Conditional | Lines |
|---|---|---|---|---|---|
| `n` | `upf(nt)%gipaw_core_orbital_n(1:ncore)` (principal quantum numbers) | (ncore) real | verbatim | `allocated` | 1021–1023 |
| `l` | `upf(nt)%gipaw_core_orbital_l(1:ncore)` | (ncore) real | verbatim | `allocated` | 1024–1026 |
| `ae_wfc` | `upf(nt)%gipaw_core_orbital(1:mesh,1:ncore)` (AE core radial orbitals, r·φ_core) | (mesh,ncore) real | verbatim | `allocated` | 1027–1029 |

---

## 8. Requested summaries

### (a) schema_version

Single write: `/Hamiltonian@schema_version = 2`, line 429 (`call qeh5_add_attribute(h5_h%id,"schema_version",2)`). Contract comment lines 422–428. No other group carries a version; no dataset is gated *on* schema_version at write time (version 2 semantics are baked into the unconditional `/e2` scalings listed above).

### (b) Quantities COMPUTED in the converter (not verbatim QE-array copies)

1. `nuclear_energy` — Ewald energy via QE `ewald(...)` function call, ÷e2 (204–206).
2. `qe_ehart/qe_etxc/qe_vtxc/qe_epaw` — refreshed by explicit `v_of_rho` (213) + `PAW_potential` (216) because `read_file` leaves the `ener` globals stale (comment 210–212).
3. `npw_g`, `npwx_g`, `maxg` — global PW-per-k bookkeeping assembled across pools/band groups (main, 89–104).
4. `occ` — `wg/wk` normalization (330).
5. `kpoint_weights` — ×0.5 renormalization for nspin==1 (297).
6. `augmentation_function_isp{n}` — Q_ij(G) built via `qvan2` + `ylmr2` real spherical harmonics + `qmod=√g·tpiba`, globally assembled, no structure factor (482–544).
7. `miller_g` / `miller_k{ik}` — global Miller tables and per-k G-sphere orderings reconstructed via `ig_l2g` gathers (550–557, 728–761).
8. `scf_local_potential`, `pp_local_component`, `vxc_with_nlcc`, `vxc` — dense-grid `fwfft('Rho')` transforms of `v%of_r+vltot`, `vltot`, and two fresh `v_xc` evaluations (with-core and core-zeroed), incl. noncollinear 2×2 packing (560–723). `vxc` recomputation destructively zeroes `rho_core`/`rhog_core` in memory (676–677).
9. `projector_k{ik}` — per-k β(k+G) tables generated by `init_us_2` (741) and pool-merged with `mergewf`/`mp_get` (790–809).
10. `deltaC` — one-center ΔC kernel from `PAW_init_fock_kernel` (870), rescaled ÷(e2·e2) (1008).
11. 0-basing conventions — `atomic_id = ityp−1` (237, 445); 0-based group/dataset suffixes for symmetries `s{i-1}` (340), species `isp{nt-1}` (532) / `nt{nt-1}` (878), k-points `miller_k{ik-1}` / `projector_k{ik-1}` (759, 812).

### (c) Per-species PAW/radial data: verbatim vs transformed

Copied VERBATIM from `upf(nt)` / `rgrid(nt)`: `r`, `rab`, `lll`, `kbeta`, `beta`, `nhtolm`, `nhtol`, `indv`, `nhtoj`, `qqq`, `qfuncl`, `qfunc`, `jjj`, `aewfc`, `pswfc`, `pfunc`, `ptfunc`, `augmom`, `ae_rho_atc`, `oc`, `rho_atc_ps`, `pfunc_rel`, `aewfc_rel`, GIPAW `n`/`l`/`ae_wfc`, all `paw` attrs (`raug`, `iraug`, `lmax_aug`, `augshape`).

TRANSFORMED at write time (all Ry→Ha ÷e2 per schema 2): Species `dion` (908), `paw/ae_vloc` (966), `paw/vloc_ps` (982); plus the computed `Onecenter/deltaC` ÷e2² (1008). At the `/Hamiltonian/{pp_type}` level: `dion`/`dion_so` ÷e2 (461, 469); `scf_local_potential`/`pp_local_component`/`vxc_with_nlcc`/`vxc` ÷e2 (600, 619, 670, 717).

### Additional observations (contract-relevant)

- `add_orbs` namelist flag is dead — no orbital data is written by this converter.
- `kpoint_weights` sits on `/System`, not `/System/BZ` (301).
- `npw` is written twice: `/Orbitals/npw` (308) and `/Hamiltonian/{pp_type}/npw` (456).
- `lspinorbit_loc` attribute call (616) is outside the ionode guard — every other h5 call is ionode-only.
- `t_rev` (imported, line 147) is not exported; `ecutwfc` is imported but never written (only `ecutrho` is, in Ry).
- Every complex dataset carries the string attribute `__complex__="1"`; complex data stored as real with leading Fortran dim 2 (h5py: trailing axis of length 2).
- `gamma_only` runs are refused outright (line 73).
