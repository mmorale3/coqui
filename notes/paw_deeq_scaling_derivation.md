# PAW deeq scaling — resolved (rad_fac = 1)

The PAW one-center deeq (and the PAW V_H / V_X matrix elements) must be a
**functional of CoQui's self-consistent one-body density matrix `n`**, carrying
the *same normalization convention CoQui already uses for that density matrix in
the smooth-grid Hartree/exchange*. It must **not** depend on the QE mean-field
occupations `mf.occ()` — those only seed the initial state of a self-consistent
HF/GW calculation; every evaluated quantity is a function of the current `n`.

This reframing resolves the long-standing "what is `rad_fac`?" question. The
answer is **`rad_fac = 1` for every k-mesh**, and the apparent need for
`rad_fac = ns_scl² = 4` (at 2×2×2) was a normalization bug in the
full-density-matrix becsum masquerading as a convention constant.

## 1. The convention anchor: CoQui's smooth-grid density

`hamilt::detail::v_h_impl` (`src/hamiltonian/v_h.hpp`) builds the smooth density
the same way for both the diagonal `nii(s,k,n)` and the full `nij(s,k,a,b)`
overloads:

```
ρ(r) = (ns_scl / N_k) · Σ_{s,k,a,b} conj(ψ_{ska}(r)) · n_{skab} · ψ_{skb}(r)
```

- `ns_scl = 2` for `nspin=1, npol=1` (spin degeneracy), else `1` — `v_h.hpp:129`
  (nii path) and `v_h.hpp:375` (nij path).
- The `1/N_k` is applied **externally** via `nrm = ns_scl/(vol·N_k)`
  (`v_h.hpp:565`); the accumulation loop carries **no** per-k weight, i.e. `n`
  itself is NOT pre-weighted by k.

So the physical augmentation occupations (the becsum that parameterizes the
one-center density `ρ_AE = Σ becsum·pfunc` and the compensation charge
`ρ_aug = Σ becsum·Q^IJ`) must carry the identical normalization:

```
becsum_a(I,J) = (ns_scl / N_k) · Σ_{s,k,a,b} P*_{a,aI} · n_{skab} · P_{a,aJ}
```

This is pinned entirely by CoQui's own density-matrix convention. QE's `wg`,
`becsum`, `ddd_paw`, `e2=2`, and `mf.occ()` never enter the *definition* — they
only provide an external cross-check at the special point where CoQui's `n`
equals QE's converged MF density matrix.

## 2. `compute_paw_hartree_atom` solves Poisson in proper Hartree ⇒ rad_fac = 1

`compute_paw_hartree_atom` (`paw/paw_onecenter.hpp`) solves the radial Poisson
equation directly in Hartree (`radial_hartree_multipole_u_form`, no `e²`
factor) on a density parameterized by the becsum above, and returns the V_H
self-energy matrix element `dDeeq_H` in proper Hartree. Because the becsum
already carries the full `(ns_scl/N_k)` normalization, `dDeeq_H` *is* the
per-channel one-center Hartree deeq with **no further multiplier**:

```
rad_fac = 1
```

Cross-check at the consistency point: with `n` = QE's converged MF density
matrix, this reproduces QE's loaded per-channel `Dnn_atom` (= `deeq/2` in Ha,
which after QE's `newd` is `dvan + ∫V_eff·Q + ddd_paw`) to ~1e-6 Ha on the
LiH kp222 PAW HF fixture, across all diagonal channels (Li 1s/2s/2p/3d + H
1s/2s). This match is a *validation checkpoint*, not the definition.

## 3. Root cause of the spurious "rad_fac = 4 / 32"

There are two becsum builders and they were **not** normalized consistently:

| builder | k-weight | spin factor | result |
|---|---|---|---|
| `compute_becsum_diagonal` + caller | `wk = 1/N_k` (internal) | `× ns_scl` (caller) | physical becsum ✓ |
| `compute_becsum_full` (before fix)  | **none** | **none** | off by `N_k/ns_scl` ✗ |

The diagonal builder is used by every validated path (`compute_paw_deeq`, the
smooth-Hartree `nii` overload, the energy path in `hartree_xc_energy.hpp`, and
`test_paw_onecenter`), all of which apply `× ns_scl` explicitly (or, in the
smooth path, via `nrm`). The full builder is used by the *self-consistent*
paths (`compute_deeq_scf(nij)` and the smooth-Hartree `nij` overload) — which
no test exercised — and it dropped both the `1/N_k` and the `ns_scl`.

For a diagonal density matrix `n_{skab} = δ_{ab} occ_{ska}`, the full builder
therefore produced `N_k/ns_scl ×` the diagonal builder's becsum:

- 2×2×2: `N_k/ns_scl = 8/2 = 4`  — coincidentally equal to `ns_scl² = 4`.
- 4×4×4: `N_k/ns_scl = 64/2 = 32`.

This is exactly the empirical pattern recorded in the original audit: a V_H/THC
oracle that "needed" `rad_fac = 4` at 222 and `32` at 444. It was never a
rad_fac — it was the missing `(ns_scl/N_k)` normalization, an N_k-dependent
fudge, which is precisely why a single constant could not work at both meshes.

## 4. Fix

Make the full path carry the identical physical becsum as the diagonal path:

1. `compute_becsum_full` (`paw/v_h_paw.hpp`): apply `wk = 1/N_k`, matching
   `compute_becsum_diagonal`. (Fixes both `compute_deeq_scf` and the
   smooth-Hartree `nij` overload, which otherwise had the compensation charge
   a factor `N_k` too large.)
2. `compute_deeq_scf` (`paw/paw_onecenter.hpp`): apply `becsum *= ns_scl`,
   matching `compute_paw_deeq` and the energy path. (The smooth-Hartree `nij`
   overload gets `ns_scl` from `nrm`, so it needs only the `wk` fix.)

After the fix, all four consumers — {diagonal, full} × {radial-direct,
smooth-grid} — use `becsum = (ns_scl/N_k) Σ P* n P`, and `rad_fac = 1`
holds universally and MF-independently.

Both builders still assume **uniform k-weights** (nosym / full BZ); a
symmetry-reduced IBZ would need real per-k weights threaded through both
helpers. This is a pre-existing limitation shared by the diagonal helper, not
introduced here.

## 5. Validation that is MF-independent

The right oracle is **builder-to-builder consistency on a self-consistent
density matrix**, not a comparison against `mf.occ()` as if it were truth:

- For a diagonal `nij = diag(occ)`, `compute_deeq_scf(nij)` must equal
  `compute_paw_deeq(occ, /*Vloc_r=*/empty, include_static=true)` element-wise
  (both = static baseline + one-center radial Hartree, no G-space term).
- More generally, with `n` the idempotent projector onto the occupied manifold
  (the converged HF density matrix), the deeq must be invariant to a unitary
  rotation within the occupied subspace and to the choice of MF that seeded it.

## 6. References

- `src/hamiltonian/v_h.hpp:129,375,565` — `ns_scl` and `nrm = ns_scl/(vol·N_k)`;
  the smooth-density convention that anchors becsum.
- `src/hamiltonian/paw/v_h_paw.hpp` — `compute_becsum_diagonal` (has `wk`),
  `compute_becsum_full` (now has `wk`), `compute_rho_aug_density_r`.
- `src/hamiltonian/paw/paw_onecenter.hpp` — `compute_paw_hartree_atom`,
  `compute_deeq_scf`, `compute_paw_deeq` (rad_fac = 1).
- `src/hamiltonian/paw/hartree_xc_energy.hpp` — energy path (diagonal becsum
  + explicit `ns_scl`), an independent confirmation of the convention.
- `src/hamiltonian/pseudo/pseudopot.cpp:563,604` — Ry→Ha (÷2) of QE's loaded
  `dion`/`deeq`, used only for the QE cross-check at the consistency point.
