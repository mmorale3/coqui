# L0 miniapp — the expensive step of the Γ₁ vertex

## Why this kernel

Measured on the Si 4³ production runs (`notes/vertex_perf_plan.md`, 2026‑09‑24), a chain
iteration with Γ₁ in both P and Σ spends **~97 %** of its wall in the Σ‑side dynamic‑vertex
solve, and the solver's own timers put **95.9 %** of that in `t_l0`:

| phase | wall | share |
|---|---|---|
| L0 (pair-pole applications) | 14 270 s | 95.9 % |
| rung applications | 349.7 s | 2.4 % |
| refit | 93.6 s | 0.6 % |
| T_s gemms / Arnoldi / rest | 162 s | 1.1 % |

Every unit converges in **one** operator application ("1 applications, converged true"), so
there is no Krylov iteration count to cut — `l0_apply` *is* the cost.

## What it reproduces

`dynbse.hpp::l0_apply_shift_cols` — the `inu != 0` twisted {U,T} path that all but one of the
79 bosonic nodes take. Per k-point:

- **A** pack the input into `Vt(nc, ncomp, nR, nc)`, `ncomp = 1 + 2·np` (constant, `U_a`, `T_a`);
- **B** per G pole: `Pj = gjᵀ Vt`, `Qj = Pj Ĝ_j`, `Bj = Pj gkq_jᵀ`, and per `l`: `Pl = G̃_l Vt`,
  `Rl = Pl gkq_lᵀ` — five skinny gemms with `K = N = nc = 8`;
- **C** `mulU`/`mulT`: scatter-accumulate each `(pole, component)` block into the five
  node-resolved accumulators `AU, AT, M2, A1, A3` of shape `(2, np, nc, nR, nc)`;
- **D** assemble into `F` and `Fsum`, the confluent terms through the sparse `D²`/`D³` tables.

Production shape (Si 4³, C = [0,8), DLR prec high, Σ-side union grid):
`nk 64, nc 8, nR 32, np 159 (fit 79), ng 80, ncomp 319` → **~1.1 TFLOP of gemm and ~0.3 TB of
scatter traffic per application**. Phase B is BLAS3 but tall-and-skinny; phase C has no reuse.

## Build and run

```
make cpu                                  # OpenMP reference
make cpu BLAS=1 BLAS_LIBS="-lmkl_rt"      # with MKL zgemm (what the rusty numbers use)
make gpu                                  # nvcc + cuBLAS (sm_80 and sm_90)
./l0_cpu  [--nk=64 --nR=32 --np=159 --ng=80 --threads=N]
./l0_gpu  [same flags]
```

Both print the cost model, per-application wall, achieved GFLOP/s and GB/s, and `|F|²` —
compare that scalar between the two binaries to check the port.

The data is deterministic pseudo-random with the production shapes and sparsity; the miniapp
checks kernels against each other, never against physics.

## Deliberately standalone

No CoQuí headers, no CMake wiring: it has to build on a GPU node without the tree, and it must
not be able to break the main build.
