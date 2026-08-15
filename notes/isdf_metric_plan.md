# ISDF metric project — plan and status

Goal: reduce `c_mu = N_mu / N_orb` (interpolation points per orbital at fixed accuracy)
in CoQui's ISDF/THC ERI factorization by changing *which* points pivoted Cholesky
selects — not the ansatz, not the zeta solve. Three composable knobs, all controlled
from the TOML input. Deliverable: error in **Hartree (Coulomb), exchange, and RPA
energies vs N_mu** on Si, plus a PRB article.

Authoritative spec: `notes/isdf_metric_knobs_spec.md` (in this repo).
Code anchors: `notes/code_map/{A_selection,B_orbitals_vG,C_input_drivers,D_tests_build}.md`.

## Session protocol (token discipline)

Each milestone is one session. At session start read ONLY: this file,
`notes/isdf_metric_decisions.md`, and the code_map file(s) the milestone touches.
Grep-first discovery; do not re-read the whole THC stack. At session end: update the
STATUS table + decisions log, commit (`isdf_metric M<k>: <summary>`), push.
Delegate mechanical edits to Opus subagents; keep numerics design in the main context.
Develop and unit-test LOCALLY (Mac); rusty ONLY for production curves (M5).

## STATUS

| Milestone | State | Commit |
|---|---|---|
| M0 scaffolding + instrumentation | pending | |
| M1 knob 2: filtered-orbital surrogate | pending | |
| M2 knob 1: separable pair weights | pending | |
| M3 knob 3: Coulomb-metric re-ranking (M3b: ibz mirror) | pending | |
| M4 sweep harness + local Si curves | pending | |
| M5 rusty production runs (Si 444) | pending | |
| M6 PRB article | pending | |

## Where each knob goes (from code_map A/B)

All selection work happens inside `thc::chol_metric_impl` (`thc.icc:763`, general)
and `thc::chol_metric_impl_ibz` (`thc.icc:43`, Gamma+IBZ). The grid-orbital arrays
(`distPsia/distPsib`) are read INSIDE these functions (Timer "DistOrbs",
`thc.icc:806-901`) and are separate from the arrays used later by the zeta solve
(`get_ZquG_Cquv_*` re-reads orbitals). Hence: selection-side changes CANNOT
contaminate the solve; the only outputs are `IPts` + `Xskau/Xskbu`.

- **Knob 2 (filter, selection-only surrogate):** multiply orbital G-coefficients by
  `exp(-alpha |k+G|^2 / Gc^2)` before the backward FFT in the DistOrbs block.
  `Gc^2 = 2*ecutwfc` (a.u.; VERIFY unit convention in-session against
  `truncated_g_grid::ecut()`). When `alpha>0` force the 'w'-read + scatter
  (`swfc_to_rho`) + FFT path even if `!custom_grid`. Nothing after selection changes.
- **Knob 1 (pair weights):** per-orbital scalars `sqrt(f_p(n))` folded into the
  orbital arrays / gathered pivot columns. Rank-1 weights = scale `distPsia/b` once
  after reading (diagonal + columns then automatically consistent). Laplace
  (`N_L`-term) weights need the multi-term Gram column: weights applied to the
  gathered `Paki/Pbki` block (`thc.icc:1113-1121`) and the diagonal build
  (`thc.icc:960-1006`), summed with coefficients `c_p >= 0`. PSD requires
  `f_p, g_p, c_p >= 0` — enforce. Orbital energies from the MF object (`mf`
  member); reference to chemical potential mu. Single-set (a_range==b_range)
  subtlety: 1/(eps_a-eps_i) is not sign-definite over all pairs — use the
  symmetric surrogate f_p(n) = exp(-t_p * max(eps_n - mu, 0)),
  g_p likewise (suppresses high-virtual pairs; document in decisions when tested).
- **Knob 3 (Coulomb re-ranking, two-pass):** run the EXISTING pivot loop to inflated
  rank `N1 = ceil(s * N_mu)` (`s = isdf_pool_factor`); the loop already maintains the
  Cholesky rows `R[0:nchol, r_local]` and pivot indices `rn`. Then:
  ell = R[:, rn] (N1 x N1, gather); FFT each row of R to G (rho_g grid,
  `gv_to_fft` map), scale by `sqrt(vbar(G))`; Y = R_G ell^H; K = Y^H Y (N1 x N1,
  replicate — N1 is small); serial pivoted Cholesky on K down to N_mu on rank 0 +
  broadcast; subset/reorder `rn` AND the accumulated collation matrices
  `Pskau/Pskbu` (u axis) before the Xskau assembly at `thc.icc:1372-1450`.
  vbar options: `bare` = 4pi/|G|^2 (via `pots::coulomb_t::evaluate` on
  `rho_g.g_vectors()`, G=0 zeroed by its `cutoff`), `attenuated` =
  bare * exp(-|G|^2/(4 omega^2)). q-average over `mf->Qpts()` when
  `isdf_metric_qavg` (points are shared across q — the reader calls
  `interpolating_points(0, ...)` once). M3a: `chol_metric_impl`; M3b: mirror in
  `chol_metric_impl_ibz` (needed for symmetry-reduced production meshes).

Constraint checks baked into the spec: the grid metric cancels from the zeta solve
exactly (never touch the solve); no ALS; never form C or K over the full grid.
`s=1` and `isdf_metric="l2"` and `alpha=0` and `weight="none"` must reproduce the
current selection BITWISE.

## TOML surface (all under `[interaction.thc]` / `[isdf]`; read in `thc` ctor
`thc.cpp:105-139`, echoed by `print_metadata` `thc.cpp:149-159`)

| key | default | knob | notes |
|---|---|---|---|
| `isdf_filter_alpha` | 0.0 | 2 | 0 = off (bitwise baseline) |
| `isdf_pair_weight` | "none" | 1 | "none" \| "gap" (Laplace 1/denominator) |
| `isdf_laplace_terms` | 4 | 1 | |
| `isdf_eta` | 0.01 | 1 | denominator regularizer (Ha) |
| `isdf_metric` | "l2" | 3 | "l2" \| "bare" \| "attenuated" |
| `isdf_pool_factor` | 2.0 | 3 | s; s=1 must be a no-op |
| `isdf_metric_gcut` | 0.0 | 3 | 0 = full rho_g list; else |G|^2/2 <= gcut |
| `isdf_metric_qavg` | true | 3 | q-averaged vbar over Qpts |
| `isdf_metric_omega` | 0.0 | 3 | attenuation width; 0 = auto (~2pi/L) |

Unknown TOML keys are SILENTLY IGNORED by CoQui (pull-based ptree reads) — therefore
every key above MUST appear in the `print_metadata` echo so runs are auditable.
Sweeps drive `nIpts` directly (hard count; thresh sentinel auto-resolves to 1e-13).

## Milestones

**M0 — scaffolding + instrumentation.** Add all TOML keys + members + echo (no
behavior change). Add end-of-selection summary line to BOTH impls at app_log(2):
final `nchol`, final `max|D|` residual, effective c_mu = nchol/nbnd (the non-ibz
impl currently reports NOTHING; the ibz impl only warns on early thresh stop).
Audit `build_isdf_only(check_accuracy=true)` (`thc_reader_t.hpp:181`) and report
what it measures. Accept: defaults bitwise-identical (test_methods_eri +
test_methods_hf green, 279s/np2 local baseline for eri), new keys echoed.

**M1 — knob 2.** Filter in DistOrbs as above, both impls. Accept: `alpha=0`
bitwise identical; alpha scan {0,0.25,0.5,1,2} on `qe_si222_ncpp` produces
monotone-ish selection changes and no degenerate-pivot aborts; log smallest
accepted pivot.

**M2 — knob 1.** Weight interface (f/g/c >= 0 enforced), rank-1 energy taper +
Laplace "gap" weight with NNLS-or-geometric fallback generator (reuse a minimax
generator if one exists in the DLR/IAFT code — check `numerics/imag_axes_ft`).
Selection-only (solve stays unweighted). Accept: `weight="none"` bitwise; N_L
convergence flat by 5-6; min-pivot logged, no PSD violation.

**M3 — knob 3.** Two-pass re-rank as above. Accept: `l2` bitwise; `s=1` no-op vs
pool-truncated baseline; `bare` vs `attenuated` both run on all three Si fixtures;
gcut and s scans converge (curves move < target between last two values).

**M4 — sweep harness + local curves.** `notes/sweeps/gen_sweep.py` + `collect.py`
(precedent: the paw notes' harvest scripts; parse stdout + `<prefix>.mbpt.h5` RPA
group {1e,hf,rpa}_energy; Hartree & Exchange parsed from the rpa driver's printed
lines `rpa.cpp:104-119`). Reference energies: NCPP Si — `[interaction.cholesky]`
tol 1e-10 (exact route); USPP/PAW Si — THC self-convergence at nIpts_c ~ 20.
Sweep nIpts_c in {4,6,8,10,12} x knob configs on `qe_si222_{ncpp,uspp,paw}`.
Deliverable per config: |Delta E_H|, |Delta E_x|, |Delta E_RPA| vs c_mu; report
c_mu* = smallest c_mu meeting 0.1 mHa/atom (tunable). Accept: baseline (all knobs
off) curves reproducible; at least one knob beats baseline c_mu* on one system.

**M5 — rusty production.** Push branch; pull + sbatch build in
`~/ceph/ISDF_metric/coqui` (scripts in `~/ceph/ISDF_metric/sbatch/`). Copy Si
kp444 saves from the isdf_vertex workspace into `~/ceph/ISDF_metric/runs/`
(read-only copy, never touch the originals) or regenerate with QE
(`~/Devel/QEF/q-e_7.0/build/CPU/bin`). Constraints (learned prior project):
MPI ranks <= nbnd (read_vnl_h5 reader limit); `-p ccq -C genoa --mem=0`;
OMP_NUM_THREADS=1. Same sweep harness, bigger meshes/bands. Accept: curves for
kp222 + kp444 Si at nbnd >= 60 for the winning configs + composition.

**M6 — article.** `notes/isdf_metric_prb.tex` (revtex4-2; precedent
`notes/paw_isdf_thc_prb.tex` on origin/unit_tests). Motivation (O(N_mu^2) storage /
O(N_mu^3) linalg in the auxiliary index), formulation (Sec 1 of the spec: metric
cancellation from the zeta solve, pivot objective K = C v C, the three knobs +
exactness of the pooled re-ranking), results (M4/M5 curves; c_mu* table), and the
PAW-consistency remark (aug channels already Coulomb-ranked). Figures from
`notes/sweeps/` data via matplotlib; build PDF locally (check `tectonic` or
`pdflatex` availability).

## Validation invariants (every milestone)

1. Default-config selection identical: same pivot list (compare the level-3
   per-pivot log on `qe_lih222` at fixed seed-free determinism) and pinned
   test_methods_hf / test_methods_gw energies unchanged.
2. `ctest -R "test_methods_eri|test_methods_hf"` green locally (np=2,
   KMP_DUPLICATE_LIB_OK=TRUE). Run test_methods_gw before each push (longer).
3. No new abort paths in PAW route: `qe_si222_paw` smoke run each milestone.

## Local run facts

Build: `coqui/build/` (clang 21 + project-local nda at `../nda`; recipe mined from
ISDF-Vertex notes; KMP_DUPLICATE_LIB_OK=TRUE required at runtime; <= 2 MPI ranks
for THC tests on macOS). `make -j4`. Fixture TOMLs for sweeps run `coqui` directly
against `tests/unit_test_files/qe/si_kp222_*` (mean_field prefix `pwscf`,
outdir = fixture dir; `filetype="h5"` uses `pwscf.coqui.h5`).
