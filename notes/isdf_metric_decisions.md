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
