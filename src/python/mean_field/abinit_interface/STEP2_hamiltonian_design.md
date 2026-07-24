# abinit2coqui Step 2 — `/Hamiltonian` (norm-conserving) design spec

Goal: emit the `/Hamiltonian` block so CoQuí's `bdft` backend can build the pseudopotential /
non-local Hamiltonian for an ABINIT NC run. Schema below is taken verbatim from what
`hamilt::pseudopot::read_vnl_h5` (src/hamiltonian/pseudo/pseudopot.cpp) reads on the
`pp_ncpp_t` path; PAW/USPP-only fields are explicitly excluded.

## What the NC path reads (must emit)

`/Hamiltonian`  attr `pp_type = "ncpp"`  (string)
`/Hamiltonian/ncpp`  attrs (all I32 unless noted):
- `number_of_nspins`, `number_of_polarizations`, `number_of_kpoints` (=nkpts_ibz),
  `max_npw`, `number_of_atoms`, `number_of_species`,
  `total_num_of_proj` (nkb = Σ_atoms nh), `max_proj_per_atom` (nhm), `ngm` (dense-grid size),
  `lspinorbit_nl` (0), `lspinorbit_loc` (0)

datasets:
- `miller_g` (ngm, 3) int — Miller indices of the DENSE (dfftp/aug) G-grid used for V_loc/V_scf
- `pp_local_component_nc` (ngm) real — V_loc(G) on the dense grid (NC local pseudopotential)
- `scf_local_potential` (nspin, npol², ngm) — V_scf(G) = total local KS potential
  (V_loc + V_H + V_xc) on the dense grid, complex (nda `(...,2)` layout)
- `dion` (nsp, nhm, nhm) real — the KB coefficients D_nn per species (column-major in pw2coqui;
  match its ordering — see pseudopot.cpp:532-560)
- `proj_per_atom` (nat) int = nh per atom; `projector_offset` (nat) int = running Σ nh;
  `npw` (nk) int = per-k projector count (== per-k wfc npw); `atomic_id` (nat) int (species idx)
- per k in IBZ: `miller_k{ik}` (npw_k, 3) int — the projector G-grid (= wfc per-k Miller list);
  `projector_k{ik}` (npw_k, nkb) complex — β_i(k+G), the KB projectors in G-space (QE `vkb`)

NOT needed for NC (gated by `if(ptype != pp_ncpp_t)`): `deeq`, `qq_nt`, `ijtoh`,
`augmentation_function_isp*`, `vxc_with_nlcc`, `/Hamiltonian/Species/*` radial partial waves,
`pp_local_component` (the non-`_nc` variant is USPP/PAW).

## Data sources & recomputation (the real work)

| field | source | method |
|---|---|---|
| dense `miller_g` | build from ecutrho + recv (Γ-sphere) | same as Step-1 wfc grid but on dfftp mesh |
| `dion` (D_nn) | ABINIT psp8 file | parse the KB energies (ekb) per projector |
| β radial `β_l(r)` | ABINIT psp8 file | parse projector radial functions + l, r-grid |
| `projector_k{ik}` β_i(k+G) | RECOMPUTE (init_us_2 analog) | β_lm,i(k+G) = (4π/√Ω)(-i)^l Y_lm(k̂+G) ∫β_l(r) j_l(|k+G|r) r²dr · e^{-i(k+G)·τ_a}; scipy `spherical_jn`, `sph_harm` |
| `V_loc(G)` = `pp_local_component_nc` | ABINIT psp8 local pot | Vloc(G)=(4π/Ω)∫[r·Vloc(r)+Z·erf-tail]·sin(Gr)/G dr (QE `vloc_of_g`) |
| `V_scf(G)` = `scf_local_potential` | ABINIT POT file (add `prtpot 1`, `iomode 3`) | read total V_KS(r) on dfftp grid → FFT → G on `miller_g` |

## psp8 parsing note
ABINIT NC pseudo `Si_GGA_noNLCC.psp8` is a plain-text format: header (zatom, zion, pspcod=8,
lmax, lloc, mmax r-grid size), then per-l blocks of projectors (with ekb = D_nn diagonal), then
the local potential column. NC KB is diagonal in n per l → `dion` is diagonal. Parse directly.

## Build/validation order
1. Emit attrs + dion + proj metadata + `miller_g` + `V_loc(G)` (psp8 only; no ABINIT re-run).
2. Recompute `projector_k{ik}` β(k+G); validate against a QE `vkb` for the same system, and via
   CoQuí overlap/H0 (NC ovlp short-circuits to I; check ⟨ψ|H0|ψ⟩ ≈ eigenvalues).
3. `V_scf(G)` from a POT file (needs ABINIT re-run with `prtpot 1`); or reconstruct
   V_H[n]+V_xc[n]+V_loc from the DEN file. Validate H0 diagonal = KS eigenvalues.
4. End-to-end: run CoQuí (bdft) HF/RPA on the ABINIT NC h5, compare to QE NC.

## Open questions to resolve during impl
- exact `dion` index ordering pw2coqui expects (row/col-major; pseudopot.cpp:532-560 reorders).
- projector normalization/phase convention vs QE `init_us_2` (structure-factor sign, i^l vs (-i)^l).
- whether CoQuí wants V_scf including or excluding V_loc (pseudopot.cpp: scf_local_potential is the
  FULL local KS potential; V_loc is provided separately as pp_local_component_nc). Confirm by
  reading how they're combined at H0 build time.
