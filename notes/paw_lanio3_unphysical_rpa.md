# LaNiO3 PAW path: unphysical positive RPA correlation energy

**Status:** open bug (2026-05-07).
**Path affected:** PAW augmented-THC ERI → RPA correlation, specifically for
metallic systems with f-electron PAW datasets.
**Other PPs unaffected:** ONCV, USPP, ccECP all give physical (negative)
RPA correlation on the same LaNiO3 cell at the same parameters.

## Symptom

Cubic LaNiO3 (Pm-3m, a = 3.838 Å, 5 atoms, 2x2x2 k-grid) at PBE level,
PSlibrary kjpaw `La.pbe-spfn-kjpaw_psl.1.0.0.UPF` (11 valence) +
`Ni.pbe-spn-kjpaw` (18 valence) + `O.pbe-n-kjpaw` (6 valence):

```
One-electron energy:       -96.02347634736492 a.u.
Hartree-Fock energy:       +61.89525784365933 a.u.
RPA energy:                +22.78788751604479 a.u.   ← wrong sign
Total energy:              -11.34033098766080 a.u.
```

RPA correlation must be ≤ 0; we get +22.8 Ha. The other PPs at
the same (ecutwfc=70 Ry, nbnd=100, k=2x2x2) give:

| pseudo | 1e (Ha) | HF (Ha) | RPA (Ha) |
|---|---|---|---|
| ONCV  (SG15 sr)  | -118.985 | +46.234 | -1.723 |
| USPP  (rrkjus)   |  -99.559 | +60.216 | -6.259 |
| ccECP (AREP La)  | (similar) | (similar) | -1.674 |
| **PAW (kjpaw)**  | -96.023  | +61.895 | **+22.788** |

USPP -6.26 vs ONCV/ccECP -1.7 is a separate question (3.6× factor; metal
sensitive to mean-field starting point), but USPP's sign is at least
correct.

## Diagnostic ruling out: low-cutoff augmentation truncation

The La spfn UPF declares `rho_cutoff = 537 Ry`. The base run uses
`ecutrho = 4×ecutwfc = 280 Ry`, well below the suggested cutoff →
augmentation Q^IJ(G) is Fourier-truncated, plausibly losing PSD
of the augmented Coulomb V_full. Hypothesis: the +22 Ha is a low-cutoff
artifact that disappears once ecutrho ≥ 537.

Re-ran at `ecutwfc=200, ecutrho=800` (well above 537):

```
One-electron energy:       -95.97940514747080 a.u.
Hartree-Fock energy:       +61.89932011617702 a.u.
RPA energy:                +22.69053237986750 a.u.   ← still wrong sign
Total energy:              -11.38955265142628 a.u.
```

ΔRPA = +0.10 Ha (0.4% change) for a 3× ecutrho jump — well below the
~25 Ha discrepancy with the correct sign. **The +22 Ha is not an
augmentation truncation artifact.** Hypothesis falsified.

## What we know

- It's PAW-path specific (USPP, ONCV, ccECP all give −1.7 to −6.3 Ha).
- It's robust to ecutrho.
- SCF converges normally (E = -556.37 Ry, 66 iter, T_F = 12.78 eV,
  conv_thr=1d-12 reached on iter 66). The mean-field is fine.
- The augmented-THC Hartree (E_H_thc) test_hartree_thc_paw_aug case in
  `src/hamiltonian/tests/test_hamilt.cpp` *passes* on Si and LiH PAW
  fixtures with deltaC stored in proper Ha (after e42fe6a). So the
  smooth+aug Hartree path is internally consistent on those fixtures.
- ZnO/paw (4 atoms, Zn-spn 20-val, semicore d) gives **physical** RPA
  ≈ -1.34 Ha and converges normally — so PAW + semicore-d isn't enough
  to break things.
- LaNiO3/uspp at nbnd=250 jumps from -6.26 Ha (n=100) to -21.3 Ha — a
  3.4× change that's also unphysically large. May be a separate
  instability of the augmented-THC ERI for metals at high nbnd, or
  may share a root cause with the PAW issue.

## Hypotheses worth checking next

1. **f-electron projector handling.** La spfn dataset has the highest
   l_max=3 + l_max_rho=6 of any pseudo in this campaign. Possible bug
   in `paw_aug_q_eval` / `compute_becsum_diagonal` / `evaluate_Q_IJ_at_K`
   for l ≥ 3 on partially-occupied bands.
2. **Smearing × becsum.** The other PAW fixtures we've validated are
   insulators (Si, LiH) with full occupations or trivial smearing.
   Metallic occupations give fractional becsum entries — verify
   `compute_becsum_diagonal` handles fractional `nii` correctly with
   `npol=1, nspin=1, ns_scl=2.0` for closed-shell metal.
3. **gygi divergence treatment for metals on coarse k-mesh.** With
   2x2x2 (3 IBZ k-points for LaNiO3 cubic after symmetry) the gygi q=0
   regularization may be undersampled. Try alternative div_treatment
   in rpa.toml (e.g. spencer-alavi).
4. **Augmented-Π eigenvalue spectrum.** Add diagnostic logging to
   the RPA driver (`src/methods/SCF/rpa.cpp` near `e_rpa = ... rpa_energy(...)`)
   that prints max eigenvalue of (vΠ) at q=0 — if any > 1, RPA
   trace integral diverges and may flip sign through quadrature.

## Reproduction recipe

```
ssh rusty
cd ~/ceph/CoQui/PAW_comparisons/runs/LaNiO3/paw/w70_n100
cat scf/scf.in nscf/nscf.in rpa.toml run.sbatch
# Already run; see rpa.out for the +22.79 Ha output.
```

Cross-check at higher ecutrho (also +22.69 Ha):
```
cd ../w200_n100
cat rpa.out  # diagnostic resubmit, job 6361717
```

## Action items

- [ ] Add Π eigenvalue logging to RPA driver behind a verbosity flag.
- [ ] Verify `compute_becsum_diagonal` on a metallic fixture (could
      add a metallic-Si-like test fixture, or use existing Si fixture
      with fractional occupations forced via cold smearing).
- [ ] Test alternative `div_treatment` in rpa.toml on this fixture.
- [ ] Once fixed, re-run the 12 LaNiO3/paw grid points (currently
      blocked behind `runs/LaNiO3/paw/SKIP` marker on ceph).

## Sibling finding: CoQui THC build memory blow-up at large N_aux

While running the same campaign, the high-nbnd LaNiO3 and ZnO runs
(nbnd ∈ {250, 500} with semicore PSlibrary `-spn` USPP/PAW datasets)
consistently OOM at the augmented-THC build stage. The RPA driver
prints right before the kill:

```
ISDF - fitting interpolating vectors to pair densities
 - Number of interpolating points: 10600
 - Estimated minimum memory requirement for this step,
     per (current type of) node: 11.247... GB
paw_aug: N_smooth=10600, N_aug=1540, N_total=12140
paw_aug: built qrad table for 3 species
```

then SIGKILL on at least one rank. The 11.25 GB estimate
underpredicts actual usage by >10× — even with `-N 4 -n 64
--mem=128G` (= 512 GB total), `storage = "outcore"`, OOM still
hits. `storage="outcore"` apparently only affects post-build ERI
storage, not intermediate collocation tensors during THC build.

**Affected:** ZnO/{paw,uspp,oncv,ccecp} and LaNiO3/{uspp,oncv,ccecp}
at nbnd ∈ {250, 500} — 46 of 168 in-scope grid points
(see `summary.csv` on ceph for the exact set).

**Workarounds tried that did not help:**
- `--mem-per-cpu=8G` × 16 ranks (=128 G/node)
- `--mem=128G` per node × multiple nodes (-N 1 → -N 4)
- `storage = "outcore"` in rpa.toml
- `cg` instead of `david` diagonalization (resolved a different,
  earlier error: too-many-bands with David at high nbnd / small cells)

**Hypotheses:**
- Likely a non-distributed temporary buffer in `methods::isdf_t`
  or `paw_aug_q_eval` of size O(N_total × N_grid) replicated per
  rank, instead of O(N_total × N_grid_local).
- Or a 4-q-point worth of complex collocation arrays held
  simultaneously in `thc_reader_t::augment_thc_with_paw`.

**Action item:** profile memory usage during the augmented-THC
build with `mpirun -np 1` on a small test (e.g. ZnO nbnd=250) to
identify the largest tensor holding the resident set. If found,
either distribute it across MPI or write it to disk (true
out-of-core).
