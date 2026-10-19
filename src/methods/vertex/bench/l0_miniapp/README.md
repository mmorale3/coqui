# L0 miniapp — the pair-propagator kernel of the dynamic vertex

## Purpose

In the dynamic-rung Bethe–Salpeter solver (`dynbse.hpp`), each operator application is dominated by the pair
propagator L0 acting on a vector in the pole-family representation (`dynbse.hpp::l0_apply_shift_cols`). This
standalone miniapp reproduces that kernel with production-like shapes so that CPU and GPU implementations can be
developed and compared in isolation.

## What it reproduces

`l0_apply_shift_cols` — the `i nu != 0` path with the twisted {U, T} pole families. Per k-point:

- **A** pack the input into `Vt(nc, ncomp, nR, nc)`, `ncomp = 1 + 2·np` (constant, `U_a`, `T_a`);
- **B** per G pole: `Pj = gjᵀ Vt`, `Qj = Pj Ĝ_j`, `Bj = Pj gkq_jᵀ`, and per `l`: `Pl = G̃_l Vt`,
  `Rl = Pl gkq_lᵀ` — skinny GEMMs with `K = N = nc`;
- **C** `mulU`/`mulT`: scatter-accumulate each `(pole, component)` block into the five node-resolved
  accumulators `AU, AT, M2, A1, A3` of shape `(2, np, nc, nR, nc)`;
- **D** assemble into `F` and `Fsum`, with the confluent terms through the sparse `D²`/`D³` tables.

Phase B is BLAS3 but tall-and-skinny; phase C has no data reuse and is bandwidth-bound. For a typical shape
(`nk 64, nc 8, nR 32, np 159, ng 80`) one application is of order 1 TFLOP of GEMM and 0.3 TB of scatter traffic.

## Build and run

```
make cpu                                  # OpenMP reference
make cpu BLAS=1 BLAS_LIBS="-lmkl_rt"      # with an external zgemm (e.g. MKL)
make gpu                                  # nvcc + cuBLAS (sm_80 and sm_90)
./l0_cpu  [--nk=64 --nR=32 --np=159 --ng=80 --threads=N]
./l0_gpu  [same flags]
```

Both binaries print the cost model, the wall time per application, the achieved GFLOP/s and GB/s, and the scalar
`|F|²`; comparing that scalar between the two binaries checks the GPU port.

The input data are deterministic pseudo-random arrays with production-like shapes and sparsity: the miniapp checks
kernels against each other, not against physics.

## Standalone by design

The miniapp uses no CoQui headers and no CMake wiring, so it builds on a GPU node without the full tree and cannot
affect the main build.
