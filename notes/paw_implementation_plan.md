# USPP/PAW Implementation Plan

Companion to `notes/paw_isdf_thc_prb.tex`. The .tex defines the formalism;
this file defines the engineering plan for getting it into CoQui.

## Goal

Extend CoQui beyond norm-conserving pseudopotentials (NCPP) to support:

- Ultrasoft pseudopotentials (USPP)
- Projector-augmented-wave (PAW)

Both starting from Quantum ESPRESSO mean-field calculations, with the
PAW-ISDF-THC formulation of the .tex used to construct all-electron ERIs
in the same THC structure already consumed by HF / RPA / GW / embedding.

## Locked decisions

1. **Where fits live.** Local fits — the channel basis `U_{a,λα}`, the local
   atomic auxiliary functions `η_{a,λ}`, and the residual-fit kernel
   `K_{a,λξ}` — are computed inside CoQui from raw radial data exported by
   `pw2coqui`. They are cached to disk keyed by species + fit parameters.
   `pw2coqui` stays physics-free except for one-center radial Coulomb
   integrals already provided by QE machinery (see decision 3).
2. **Compensation-charge grid.** `Q̂_{a,αβ}` lives on the **dense** FFT grid
   (`dfftp` in QE; `ngm_g` in pw2coqui). This matches the existing
   `augmentation_function_isp{nt}` convention in `pw2coqui.f90:450-477` and
   avoids under-resolving high-ℓ multipole channels.
3. **Core-valence and core-core exchange in scope.** Requires PAW datasets
   generated `--with-gipaw` so that `upf%gipaw_core_orbital*` is populated.
   `pw2coqui` computes the raw one-center radial integrals
   `K^{cv}_{a,αβ,cd}` and `K^{cc}_{a,cd,ef}` (using the same Gaunt + radial
   Hartree machinery as valence `ke%k`) and writes them to h5. CoQui appends
   core orbitals as additional atom-local rows in `X` per .tex §7.
4. **SOC and noncollinear in v1.** Per .tex §7 the spatial 𝒱^q kernel is
   unchanged; only `X^k_{Λ,iσ}` carries spinor structure. `pw2coqui` extends
   to export `jjj`, `nhtoj`, `qq_so`, `paw%aewfc_rel`, `paw%pfunc_rel`.
   Reader gets a spinor branch in the per-species PAW path mirroring the
   existing `dion_so` branch (`pseudopot.cpp:502-513`).

## HDF5 schema (output of Phase 0)

Layout under `/Hamiltonian/` in the `.coqui.h5` file. Existing groups are
preserved; new ones are additive.

```
/Hamiltonian/
  pp_type             : attribute, "ncpp" | "uspp" | "paw"   [exists]
  {ncpp,uspp,paw}/                                            [exists]
    nh, ofsbeta, ijtoh, atomic_id                             [exists]
    dion | dion_so                                            [exists]
    miller_g, miller_k{ik}, projector_k{ik}                   [exists]
    scf_local_potential, pp_local_component[_nc]              [exists]
    vxc, vxc_with_nlcc                                        [exists]
    augmentation_function_isp{nt}        (USPP/PAW)           [exists]
    lspinorbit_nl, lspinorbit_loc                             [exists]

  Species/{nt}/                                               [NEW, all phases]
    species_kind     : "ncpp" | "uspp" | "paw"
    mesh, kkbeta, lmax, lmax_rho
    r(mesh), rab(mesh)
    nbeta, lll(nbeta), kbeta(nbeta), jjj(nbeta)               (jjj if SOC)
    beta(mesh, nbeta)
    aewfc(mesh, nbeta)         (PAW)
    pswfc(mesh, nbeta)         (PAW)
    qfuncl(mesh, nbeta(nbeta+1)/2, 0:2*lmax)
    qqq(nbeta, nbeta), q_with_l, nqf, nqlc, rinner(0:2*lmax)
    nhtolm, nhtol, nhtoj
    qq_so(...)                 (SOC)

    paw/                       (PAW only)
      pfunc(mesh, nbeta, nbeta)
      ptfunc(mesh, nbeta, nbeta)
      pfunc_rel(...)           (SOC, small component)
      aewfc_rel(...)           (SOC)
      augmom(nbeta, nbeta, 0:2*lmax)
      raug, iraug, lmax_aug, augshape
      ae_vloc(mesh)
      ae_rho_atc(mesh)

    Onecenter/                 (PAW only — raw integrals from QE)
      deltaC(nh, nh, nh, nh)        # = ke(nt)%k from PAW_init_fock_kernel
      Kcv(nh, nh, ncore, ncore)     # core-valence radial Coulomb
      Kcc(ncore, ncore, ncore, ncore)

    Core/                      (PAW + GIPAW only)
      ncore_orbitals
      n(ncore), l(ncore), label(ncore)
      ae_wfc(mesh, ncore)
```

### Schema contract (plan B4, 2026-07-24 — single source of truth)

**`schema_version`** (int attribute on `/Hamiltonian`):

| version  | meaning |
|----------|---------|
| (absent) | legacy export: carried QE `deeq`/`deeq_nc`; energy-valued datasets in **Ry** (QE) or Ry-convention ×2 (abinit2coqui ≤ B2) |
| 1        | deeq-free (B1); units unchanged (Ry convention) |
| 2        | deeq-free + **Hartree on disk** for every energy-valued `/Hamiltonian` dataset |

**Units.** Energy-valued datasets covered by the rule: `dion`, `dion_so`,
`Species/{nt}/dion`, `Species/{nt}/paw/{ae_vloc, vloc_ps}`,
`pp_local_component[_nc]`, `scf_local_potential`, `vxc`, `vxc_with_nlcc`.
Readers apply the 0.5 Ry→Ha scale **only when `schema_version < 2`**
(`hamilt::h5_pp_ry2ha`). Always-Hartree regardless of version (never
scaled): `Onecenter/deltaC`, `Onecenter/ex_cvij`, `/System` energy
attributes (`nuclear_energy`, `madelung_constant`, `fermi_energy`,
`exx_core_core`), `/Orbitals/eigval`. Unit-free: projectors/`beta`,
partial waves, `qfuncl`/`qqq`/`qq_nt`/`qq_so` (charges), densities
(`*rho_atc*`), channel maps. In-memory convention after read is Hartree
for everything, including `ae_vloc`/`vloc_ps` (read-time scaled).
`pseudopot_to_h5.hpp` (pw2bgw plot-file import) still emits legacy-Ry
files without the attribute — intentionally valid as "legacy".

**`miller_g`** is an **ecutrho G-sphere** on both converters (QE: the
native `ngm_g` gvect sphere; abinit2coqui ≥ B4: the `ecutrho` sphere
carved from the FFT box — no longer the full box). Readers map it onto
`fft_mesh_aug` via `generate_k2g` and make no completeness assumption.

**`ijtoh`** has shape `(nsp, nhm, nhm)` (h5/row-major), 1-based values:
`ijtoh(nt, ih, jh)` = position of pair (ih, jh) in the sequential
upper-triangle enumeration `for ih: for jh >= ih: ++ij` (QE `init_us_1`,
verified against QE 7.4.1). Symmetric; padding beyond `nh(nt)` is
UNCONSTRAINED (QE writes −1, abinit2coqui writes 0 — readers must only
touch the active `nh(nt)` block). `read_vnl_h5` recomputes the expected
packing per species over the active block and hard-errors on mismatch (B4).

CoQui-internal caches (Phase 3+) live under a separate root, e.g.
`/PAW_Cache/Species/{nt}/{U, eta, K_fit}` keyed by fit parameters. Not part
of the converter contract.

## Phase 0 — Schema and converter cleanup

Foundation. Locks the converter ↔ reader contract. ~1 PR.

**Touch points**
- `qe_converter/pw2coqui.f90`
  - Fix the silent-stub bug in `write_factorized_paw_one_center` (lines
    833–843): `L` is allocated, computed, and immediately deallocated
    without any `h5_write_*` call. Either drop the factorization (we agreed
    fits live in CoQui) and write only the raw `ke%k` as `Onecenter/deltaC`,
    or remove the routine entirely and rewrite as `write_paw_onecenter`
    that emits raw integrals.
  - New routine `write_species(h5_n)` exporting per-species radial-grid
    fields enumerated in the schema above. All ionode-only; arrays are
    small.
  - New routine `write_paw_onecenter(h5_n)`:
    - Calls `PAW_init_fock_kernel` (already used) and writes raw
      `ke(nt)%k` as `Species/{nt}/Onecenter/deltaC` per species.
    - For `--with-gipaw` datasets, computes `Kcv` and `Kcc` using a
      generalized variant of `PAW_fock_onecenter` (in QE
      `PW/src/paw_exx.f90:297`) that takes (any radial pair, any radial
      pair) instead of always using `upf%paw%pfunc / ptfunc`. For
      core-core/core-valence, the second pair reads from
      `gipaw_core_orbital(:,c)`. No PS counterpart for core: frozen-core
      densities vanish outside the augmentation sphere.
    - Hard error if PAW + core-valence requested but
      `upf(nt)%has_gipaw == .false.`. Friendly message pointing at
      `--with-gipaw`.
  - Extend SOC writes: `jjj`, `nhtoj`, `qq_so`, relativistic small
    components.
- `qe_converter/README.md` — document new fields and the `--with-gipaw`
  requirement for core-valence.

**Tests (Phase 0)**
- New CTest under `qe_converter/tests/`: minimal Si PAW input, run
  `pw2coqui`, assert h5 schema matches a fixture file.
- Round-trip check for `Onecenter/deltaC`: re-symmetrize and compare to a
  recompute from `pfunc/ptfunc/qfuncl` to within `1e-10`.

**Out of scope this phase:** any changes to the CoQui reader or
downstream code.

## Phase 1 — USPP minimum-viable in CoQui

Get the USPP path through HF/RPA running, with augmentation handled in the
existing smooth pair-density code path. Shakes out plumbing PAW also needs.
2–3 PRs.

**Touch points**
- `src/hamiltonian/pseudo/pseudopot.{h,cpp}`
  - Replace `utils::check(false,"finish")` at `pseudopot.cpp:526` with a
    real `pp_uspp_t` branch. Read `dion`/`dion_so`, `qq_at`, `qq_nt`,
    `augmentation_function_isp{nt}` into `qgm` (currently a dead
    placeholder at `pseudopot.h:237`).
  - Populate the `ijtoh` field that's already commented out at
    `pseudopot.cpp:478, 482`.
- New `src/hamiltonian/overlap.{h,hpp}` (or similar): introduce an `S_op`
  abstraction for the USPP/PAW overlap
  `S = 1 + Σ_{a,IJ} |β_aI⟩ q_{IJ} ⟨β_aJ|`. Wire through the few places
  that assume `⟨ψ_i|ψ_j⟩ = δ_{ij}` (smoke test: orthogonality assertion on
  outputs of `Pskna` consumers).
- `src/methods/HF/`, `src/methods/SCF/`: smoke-test eigenvalue sums and
  Hartree assembly with augmentation included via direct
  `Σ_{IJ} ρ^{ij}_{aIJ} Q^{IJ}(G)` injection into the smooth pair density
  before Coulomb convolution. This is a temporary equivalent to the
  compensation-charge formulation (Phase 3) — same numerics, preserves the
  existing pair-density code path, lets us validate USPP independently of
  PAW.

**Tests (Phase 1)**
- Bulk Si with USPP at Γ-only: Hartree energy and exchange energy from
  CoQui must match `pw.x` total to within k-point integration error.
- Orthogonality of `S^{1/2} ψ_i` recovered to `1e-10` after `S_op`
  application.

## Phase 2 — PAW one-center kernel ingest

Get `K_a` (the .tex's `ΔC_{a,αβγδ}`) into CoQui memory. Lowest-risk PAW
addition: tensor is dense, atom-local, tiny.
1 PR.

**Touch points**
- `src/hamiltonian/pseudo/pseudopot.{h,cpp}`
  - Read `Species/{nt}/Onecenter/{deltaC, Kcv, Kcc}` for `pp_paw_t`.
  - Read `Species/{nt}/{paw,Core}/...` for radial data needed in
    Phase 3.
  - Store as `std::vector<species_paw_data>` on the pseudopot object
    (new POD struct, header-only).

**Tests (Phase 2)**
- Symmetry assertions on `deltaC` (the .tex's `K_{a,λξ}^* = K_{a,ξλ}` after
  fit; for the raw tensor, the four-fold permutation symmetries used in
  `write_factorized_paw_one_center:816-820`).
- Echo round-trip: read → write a copy → byte-compare.

## Phase 3 — Compensation charges and Y-features

Build the `(𝒢, 𝒜)` composite auxiliary index from .tex Eq.
(lambda-composite). 2 PRs.

**Touch points**
- New `src/hamiltonian/paw/compensation.{h,hpp}`: construct
  `Q̂_{a,αβ}(s)` on the dense FFT grid for each species. QE's `qvan2`
  output (already on disk per Phase 0) gives this; validate against
  `upf%paw%augmom` — multipoles must reproduce `ΔQ_{αβ}` through
  `lmax_aug`.
- New `src/hamiltonian/paw/local_isdf.{h,hpp}`: solve the local LS
  problem of .tex Eq. (eta-fit) for each species. Output:
  `U_{a,λα}` and `η_{a,λ}(s)` on the radial grid. Cache per-species under
  `/PAW_Cache/...` in the h5 file.
- `src/methods/ERI/thc.h` / `thc.icc`: add a `Y^k_{aλ,i}` builder. This is
  a per-atom dense gemm against `Pskna` (already in pseudopot). Slots
  alongside the existing `X^k_{μi}` collocation matrix.
- For SOC/noncollinear: spinor index σ on `Y` and on `X`. Kernel
  unchanged.
- Core orbitals (per decision 3): append additional local rows in `X`
  representing core wavefunctions evaluated as fixed atom-local features.
  No grid evaluation needed — they live entirely inside the augmentation
  sphere.

**Tests (Phase 3)**
- Multipole reproduction: built `Q̂_{a,αβ}` integrated against `|s|^ℓ
  Y_{ℓm}` matches `upf%paw%augmom` to `1e-8`.
- Local fit accuracy: `η_{a,λ}` reproduces the species training set
  `{φ_α^* φ_β, φ̃_α^* φ̃_β, Q̂_{αβ}, ΔQ_{αβ}}` of .tex Eq.
  (local-training-set) within configurable `eps_local`.

## Phase 4 — PAW-ISDF in the THC builder

Surface the boxed equations of .tex §IV out of `methods::make_isdf` for
PAW inputs. 2–3 PRs.

**Touch points**
- `src/methods/ERI/thc.{h,cpp,icc}`
  - Generalize the collocation matrix `X` to carry `N_μ + N_A` rows. The
    smooth ISDF block `μ ∈ 𝒢` is built as today; the augmentation block
    `(a, λ) ∈ 𝒜` comes from Phase 3.
  - Compute the projected Coulomb matrix `𝒱^q_{ΛΣ}` block-by-block per
    .tex Eq. (block-gg, block-gl, block-ll):
    - smooth-smooth: existing FFT path
    - smooth-local + local-local: FFT-on-deposited-η, or analytic radial
      transform of the Bloch-summed `η^q_{aλ}`
  - Add `K_{a,λξ}` to the same-atom local-local block (.tex Eq.
    `kernel-final`), where `K` is the LS-fit refinement of the raw
    `deltaC` from Phase 2 in the local channel basis.
  - Verify Hermiticity (.tex Eq. `eri-hermitian`) at `q ≠ 0`. The `q = 0`
    `G = 0` singularity treatment matches the existing NCPP path because
    compensation charges already make the pair density neutral (.tex §6
    "Discussion").
- `src/methods/ERI/eri_utils.hpp`: add USPP/PAW dispatch in `make_thc`
  (currently `eri_utils.hpp:59`) and `make_isdf`.
- `src/methods/ERI/thc_reader_t.hpp`: aux-index dimensions become
  `N_μ + N_A`. Verify all consumers (HF, RPA, GW, embedding) handle the
  enlarged auxiliary index transparently — by the .tex's design, they
  should.

**Tests (Phase 4)**
- **Reference benchmark.** Bulk Si + one transition-metal oxide (e.g.,
  NiO) with both NCPP and PAW pseudos. Sweep `α = N_μ/N_orb` and the local
  rank in `K_a`. Converge to all-electron ERIs from an independent code
  (PySCF GTH or VASP) within `eps_isdf + eps_local`.
- Hermiticity check on assembled `𝒱^q` at multiple `q ≠ 0`.
- Compare to direct (un-factorized) PAW Hartree from QE for a few
  selected `(ij, kl)` ERIs.

## Phase 5 — Downstream methods + tests

HF / RPA / GW / embedding consume any THC kernel by construction
(`src/methods/{HF,GW,SCF,embedding}`), so once Phase 4 lands they should
work on the new aux index without code changes. Rolling tests:

- HF and RPA correlation energy on Si and one TMO against PAW VASP.
- Spot-check `downfold_2e`, `gw_downfold` for regressions on the existing
  NCPP suite, then add PAW cases.
- Add `examples/toml_input_interface/paw/` with a runnable Si example.

## Test strategy summary

| Phase | What we're proving | Reference |
|-------|--------------------|-----------|
| 0     | Schema is stable; converter exports what reader will need | h5 fixture |
| 1     | USPP path runs end-to-end; S overlap correct | QE total energy |
| 2     | PAW one-center tensors land in memory undamaged | Symmetry + round-trip |
| 3     | Compensation charges and Y features are well-fit | Multipole moments + LS residual |
| 4     | Full PAW-ISDF-THC ERIs are accurate all-electron | VASP / PySCF GTH ERIs |
| 5     | Downstream methods (HF/RPA/GW) reproduce PAW reference | VASP totals + spectra |

## Energy regression tests (Phase 1.5, completed)

`src/hamiltonian/tests/test_hamilt.cpp::test_hartree_energy` and
`test_vxc_rho_integral` exercise CoQui's smooth-grid pipeline against QE
reference values exported by `pw2coqui` (attributes `qe_ehart`, `qe_etxc`,
`qe_vtxc`, `qe_epaw` on `/System`):

| Fixture | E_H (CoQui − QE) | vtxc (CoQui − QE) |
|---|---|---|
| NCPP | 1e-6 Ha | 1e-6 Ha |
| USPP | 0 (machine) | 2e-14 Ha |
| PAW  | 0 (machine) | 2e-14 Ha |

**Bug fixed during this work**: my initial standalone `build_total_density_r`
in `src/hamiltonian/paw/hartree_xc_energy.hpp` was using
`wfc_g->gv_to_fft()` directly to place ψ(G) on the FFT mesh. This array is
encoded on the WFC grid (e.g. 19³), not the dense FFT grid (e.g. 36³). The
remap is done by `map_truncated_grid_to_fft_grid` (called inside
`make_wfc_to_rho`) and stored as `pseudopot::swfc_to_rho`. Added a
`pseudopot::swfc_to_rho_view()` accessor and switched both energy tests to
use it. The production v_h_paw and add_Hartree paths were already using
`swfc_to_rho` internally, so the eigenvalue test was unaffected.

## RESOLVED: PAW eigenvalue residual was an SCF convergence issue

**Final root cause**: the original `lih_kp222_nbnd16_paw` fixture was generated
with `conv_thr=1e-10`. For PAW with Li 1s as semicore valence (deepest band at
ε ≈ -44 eV, projector overlap S_pw ≈ 0.36), this convergence is *insufficient*
because tiny becsum changes get amplified through `ddd_paw` (which depends
nonlinearly on `V_AE_oc[ρ_AE_oc]`, ~singular near r=0). The result: stored et
differed from h_psi(ρ_saved) by 0.146 Ha — purely an SCF residual, not a CoQui
bug.

The diagnostic chain that pinpointed this:
1. Hartree/vtxc tests showed CoQui's smooth-grid pipeline matches QE to 1e-14
2. Per-band decomposition in QE Fortran (T+V_loc+V_H+V_xc+V_NL) summed to
   -1.654 Ha, **matching CoQui exactly**, NOT QE's stored et = -1.508 Ha
3. Calling QE's own `h_psi(saved_ψ)` after `set_vrs` gave -1.654 Ha,
   confirming the discrepancy is between *stored* eigenvalues and h_psi at
   the saved density
4. Re-running SCF with `conv_thr=1e-14, mixing_beta=0.3` brought the saved
   eigenvalue to -1.626 Ha (close to -1.654), and CoQui's eigenvalue test
   residual dropped from 0.146 Ha to **6.2e-9 Ha**.

**Resolution**: replaced the PAW fixture with the tightly-converged version.
All eigenvalue and energy tests pass at 5e-5 Ha (eigenvalue) and 1e-8 Ha
(E_H, vtxc) tolerances. PAW path is now fully verified end-to-end.

**Lesson for future PAW fixtures**: use `conv_thr ≤ 1e-12` (1e-14 ideal) and
a small `mixing_beta` (0.3) when the dataset includes deep semicore states
as valence. The standard Hartree/total-energy convergence test (dr2 < conv_thr)
does *not* guarantee tight eigenvalue convergence for these states — `ddd_paw`
sensitivity demands roughly two extra digits of density-mixing precision.

## Open issues (none currently — all PAW eigenvalue/energy tests pass)

### Historical: PAW eigenvalue residual on `test_dft_eigenvalues` (RESOLVED)

`src/hamiltonian/tests/test_hamilt.cpp::test_dft_eigenvalues` recomputes
`H_nn = ⟨ψ_n | (T + V_loc + V_H + V_NL) | ψ_n⟩ + ⟨ψ_n|V_xc|ψ_n⟩` and
divides by `S_nn` to recover ε_n via the generalized eigenvalue equation
`H|ψ⟩ = ε S|ψ⟩`. NCPP fixtures pass at 5e-7 Ha. The PAW LiH 222 fixture
USED to fail at the **0.146 Ha** level on the deepest Li-1s-like band
— now passes at 6e-9 Ha after tightening conv_thr (see RESOLVED section
above).

**What we know** (from instrumented decomposition T / V_loc+V_NL / V_H /
V_xc / S_pw / S_full per band, plus offline diagonalization of the 16×16
Hk/Sk):

- For NCPP at k=0: every off-diagonal `H[a,b]` for a≠b is numerically
  zero ⇒ QE band basis diagonalizes CoQui's H ⇒ H_CoQui = H_QE.
- For PAW at k=0: off-diagonals are *not* zero —
  H[0,1] ≈ -0.035, H[0,2] ≈ -0.033 Ha — even though S off-diagonals are
  numerically zero. Diagonalizing CoQui's H/S gives ε_CoQui[0] = -1.656
  vs ε_QE[0] = -1.508 (diff -0.148). So **CoQui's H matrix differs from
  QE's H matrix**, this is not a basis-rotation artifact.
- The pattern by band: deepest band (S_pw ≈ 0.36, mostly Li 1s β
  projector) has the largest residual; the next valence band (mostly H
  1s, S_pw ≈ 1.0) has 5e-3 Ha residual; degenerate p-state bands
  (S_pw ≈ 1, l=1 projectors only) match to 5e-5 Ha. Residual is
  k-independent (0.146 Ha at every k for the deepest band).
- V_H smooth + augmentation has been verified band-by-band against QE's
  smooth-grid integral (the `ns_scl × Ω` scale in `v_h_paw.hpp` was
  derived from this match). V_NL formula matches QE's `add_vuspsi_k`
  structurally (verified by direct comparison). `deeq` exported by
  `pw2coqui` after `read_file_new` includes `dvan + ddd_paw + ∫V_eff Q`
  per QE's `newd_acc` (verified by an instrumented dump that's been
  reverted).
- becsum from CoQui matches QE to 3 sig figs (`becsum_(Li,0,0) = 2.02`
  vs QE 2.03), so projector overlaps are correctly normalized.
- FFT mesh ruled out: `fft_mesh = dffts = dfftp = 36³` for this fixture
  (ecutrho = 4×ecutwfc default), so smooth/dense grid mismatch can't
  explain the residual.

**USPP isolation experiment** (decisive finding): Adding the
`lih_kp222_nbnd16_uspp` fixture (USPP, no ddd_paw) shows the eigenvalue
test **passes at 3e-6 Ha** with the same projector / V_H_aug / V_NL /
add_S pipeline that PAW uses. So all the augmentation machinery is
correct; the bug is specifically in the **ddd_paw application**.

**ddd_paw scaling experiment** (instrumented `pw2coqui` to multiply the
ddd_paw contribution to deeq before export, then re-tested):

| ddd_paw multiplier | PAW residual (Ha) | Δ vs prev |
|---|---|---|
| 0× (subtracted) | 0.212 | — |
| 1× (production) | 0.146 | -0.066 |
| 2× | 0.079 | -0.067 |
| 3× | 0.012 | -0.067 |

Each ddd_paw addition shifts `H_nn` by **exactly +0.066 Ha** for the
deepest Li-1s-like band — perfectly linear. The residual hits zero at
**~3.21× ddd_paw**. So the bug is "ddd_paw effective contribution is
~3× smaller than it should be"; the ddd_paw-to-V_NL coupling has a
missing factor of ~3.

**What to check next**: trace through the becsum-packing convention all
the way from QE's `compute_becsum`/`PAW_potential` (where ddd_paw is
defined as ∂E_paw_oc/∂becsum_packed, with becsum_packed[off-diag] =
2×Re[β*β]) to CoQui's V_NL formula `Σ ⟨ψ|β⟩ × deeq × ⟨β|ψ⟩`. The
factor-of-3 (not 2 or 4) suggests a subtle interaction: spin-degeneracy
factor + packed-Hermitian factor + something else.

**Test status**: `test_dft_eigenvalues` PAW section passes at tol = 0.2
Ha; tighten to 5e-5 once resolved. USPP and NCPP sections pass at 5e-5.

## Open follow-ups (not blocking v1)

- Frozen-core extension when `--with-gipaw` data is unavailable: fall back
  to core-density Hartree only (no exchange). Requires a runtime mode flag
  and clear logging.
- Adaptive ISDF-on-real-space-grids (Zhu/Yeh/Morales/Greengard/Jiang/Kaye,
  arXiv:2510.20826, cited in .tex) — orthogonal axis of compression that
  composes with PAW-ISDF.
- Coulomb-metric refinement of `η_{a,λ}` after L²/pivoted-Cholesky
  selection (.tex §6 final paragraph). Add as an optional refinement step
  to Phase 3.
