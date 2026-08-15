# ISDF metric project — decisions log

Append-only. Newest at the bottom. Each entry: date, decision, why.

- **2026-08-14 — Branch base.** `isdf_metric` cut from `uspp-paw-isdf` @ 7cdd567
  (user directive). Rationale: the USPP/PAW THC route is the production ERI path
  and its augmentation-channel selection already uses a Coulomb metric
  (`paw_isdf_metric`, `thc_reader_t.hpp:162`) — point selection should become
  consistent with it.
- **2026-08-14 — Docs live in-repo under `notes/`** (repo has no `docs/`; the
  `notes/*_plan.md` + `notes/*_prb.tex` convention exists on origin/unit_tests
  and origin/paw). Master working copies also under `~/Projects/ISDF_metric/notes/`
  (outside the repo) for the non-code assets.
- **2026-08-14 — Dev local / production rusty** (user directive). Local Mac build
  uses a project-local nda clone (`~/Projects/ISDF_metric/nda`, triqs/nda branch
  `tensor` @ 624fe5c0) because the shared `~/Software/nda_mmorales` checkout is too
  old for this branch (`nda::tensor::op::RCP` missing) and belongs to other projects.
- **2026-08-14 — All knob options in TOML under `[interaction.thc]`/`[isdf]` with
  `isdf_` prefix**, defaults chosen so every default is bitwise-identical to the
  current code. CoQui silently ignores unknown TOML keys, so every new key is
  echoed via `print_metadata` (auditability requirement).
- **2026-08-14 — Selection-only scope.** Weights/metric affect POINT SELECTION
  only; the zeta solve stays unweighted and unfiltered (spec Sec 1.2: the grid
  metric cancels identically from the solve). `isdf_weight_solve` deferred
  indefinitely.
- **2026-08-14 — Si test matrix.** Local: `qe_si222_{ncpp,uspp,paw}` in-repo
  fixtures; NCPP reference = Cholesky ERI tol 1e-10 (Cholesky hard-aborts for
  USPP/PAW: `cholesky.cpp:73-98`), USPP/PAW reference = THC self-convergence at
  nIpts_c ~ 20. Production (M5): Si kp444, nbnd >= 60 on rusty.
- **2026-08-14 — Energy observables** = the `[rpa]` driver's printed/stored set:
  Hartree + Exchange (printed, `rpa.cpp:104-119`), 1e/HF/RPA (h5 `RPA` group).
  One driver gives all three requested errors (Coulomb, exchange, RPA) per run.
- **2026-08-14 — Knob 2 must not touch the collation matrices.** First M1 cut
  leaked the filter into the returned Xskau/Xskbu (accumulated from the filtered
  Gram arrays), contaminating Theta and every downstream contraction (E_x drifted
  ~0.16 Ha on Si at alpha=2). Fix: selection keeps FILTERED arrays for the Gram
  (diagonal + pivot columns) and UNFILTERED twins for the collation values
  (comm_buff carries an extra X section, xdup=2, when the knob is active).
  Same treatment in chol_metric_impl and chol_metric_impl_ibz. Twin b-side gets
  the identical q-phase factor. Cost when active: 2x the selection-side orbital
  memory + 2x the per-pivot gather volume; zero when alpha=0.
- **2026-08-14 — Bitwise regression caveat.** The 'w'+FFT selection path uses
  FFTW FFT_MEASURE plans (runtime-tuned), so run-to-run results differ at the
  1e-11 level even on identical binaries. "Bitwise" acceptance is therefore
  checked as: identical point count + energies within 1e-9 across a re-run pair.
- **2026-08-15 — M3b (ibz mirror) descoped to follow-up.** The ibz factor lives
  on the irreducible grid where the FFT-based exact K assembly does not apply
  directly; a correct mirror needs full-grid pool columns with symmetry-star
  handling (see m3_implementation_notes.md). Since symmetry-free Si production
  saves exist (nscf_kp{222,444}_nbnd256_nosym, copied into our mf_saves), the
  paper's production data uses the exact main-impl path; the symmetry-adapted
  selection guards isdf_metric/pair_weight != defaults with a clear message and
  the limitation is documented in the manuscript. Knob 2 (filter) IS available
  in the ibz path.
- **2026-08-15 — knob-3 bring-up findings.** (i) FFTW MEASURE planning would
  garble already-loaded data; pass 2 uses ESTIMATE. (ii) The dense pivoted
  Cholesky must update the FULL trailing block (not just the lower triangle)
  because full row/column pivot swaps otherwise mix stale entries — its PSD
  guard caught both bugs. (iii) USPP/PAW pools can stop at thresh before
  s*nIpts; re-ranking proceeds with the reached pool (warned).
- **2026-08-15 — Scoring directive (user).** The Hartree energy is cheap to
  evaluate directly from the density without the ERI factorization, so the fit
  does NOT need to target it. The observables that matter — the slow ones the
  factorization exists for — are EXCHANGE and DYNAMIC (screened/RPA-type)
  correlation. All scheme evaluation, c_mu* scoring, recommended defaults, and
  the manuscript's results framing use E_x and E_RPA; E_H is reported only as a
  consistency check. This materially upgrades the pair weight (its only bad
  number was Hartree at small c) and the filter (same).
- **2026-08-15 — Weight-form optimization added (user directive).** Any
  nonnegative separable function of the pair (via eigen-energies) is admissible
  as a weight; rather than fixing the 1/x Laplace form, parametrize f(eps) and
  optimize the parameters to minimize selection cost (rank-1 preferred: N_L=1
  makes weighted selection as cheap as unweighted) and maximize c_mu reduction
  scored on |dE_x|+|dE_RPA|. New plan item M4b; TOML surface will gain
  `isdf_weight_params` and additional `isdf_pair_weight` family names.
