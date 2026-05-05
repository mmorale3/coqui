# Real-axis GW: GPU port status and Rusty handoff

State after the structural-templating commits on `real_axis`, plus the
2026-04-29 cuFINUFFT validation pass on a Rusty A100. The conv-class gates
(steps 1–4 below) are now fully validated end-to-end; the remaining
kernel-level work (steps 5–9) still needs porting.

## Validation status (Rusty, A100, CUDA 12.8, 2026-04-29)

- `test_math_finufft_nda` — 10 cases / 983 assertions, all pass.
  - Includes the new `cufinufft_nda_{1d,2d,RAII}_device_vs_host_double`
    cases (3 / 674 assertions) — cuFINUFFT bindings layer matches host
    FINUFFT element-wise to NUFFT eps.
- `test_real_axis_conv` — 6 cases / 1202 assertions, all pass.
  - Host-side benchmarks (Gaussian xcorr, Lorentzian Hilbert, H² = -1).
  - New `real_axis_conv_device_vs_host_{cross_correlate,convolve,hilbert}`
    cases (3 / 800 assertions) — `real_axis_conv_mem_t<DEVICE_MEMORY>`
    matches the host engine to NUFFT eps.

Real bugs found and fixed during the validation pass:

1. `cufinufft.cpp` was casting to `std::complex<float>*` /
   `std::complex<double>*` for the cuFINUFFT execute calls, but the
   API expects `cuFloatComplex*` / `cuDoubleComplex*`. The double path
   compiled by accident (implicit conversion); the float path didn't.
2. FINUFFT v2.4.x builds the `cufinufft` static lib with
   `CUDA_SEPARABLE_COMPILATION ON`. Without `CUDA_RESOLVE_DEVICE_SYMBOLS`
   the consumer must be linked with nvcc (`LINKER_LANGUAGE CUDA`); CoQui
   builds plain C++ test binaries. `cmake/finufft.cmake` now sets
   `CUDA_RESOLVE_DEVICE_SYMBOLS ON` on the cufinufft target.
3. The original conv-class lift used `nda::map` lazy assignments
   ("MEM-agnostic via nda::map") for Hadamard / scalar-scale / type lifts.
   That works on host (nda's host fallback hand-loops the assignment) but
   breaks on device — `tensor::assign` requires a `MemoryArray` rhs and
   rejects `expr_call`, AND the actual GPU runtime path will segfault on
   any lazy-expression rhs. Replaced every device branch with explicit
   `tensor::scale` + `tensor::elementwise` (with `if constexpr` host
   branches kept for the type-changing real⇄complex lifts, which have no
   nda::tensor primitive — those stage through host).

---

## What's already in place (built locally, no GPU)

### Memory-space dispatch in the (cu)FINUFFT NDA wrapper

- `numerics/fft/finufft_define.hpp` now has the `NUFFT_BACKEND_CUFINUFFT`
  enum tag in `nuplan_t::bend`.
- `numerics/fft/cufinufft.h` declares the device-side namespace
  `math::nufft::impl::dev`, mirroring `impl::host` one-for-one.
- `numerics/fft/cufinufft.cpp` is gated on `#ifdef COQUI_HAVE_CUFINUFFT`.
  Without that flag, the file compiles to runtime-aborting stubs (clear
  "compiled without cuFINUFFT" message). With it, the body calls
  `cufinufft_makeplan` / `setpts` / `execute` / `destroy` directly.
- `numerics/fft/finufft_nda.hpp`:
  - `create_plan<MEMORY_SPACE MEM = HOST_MEMORY>(...)` selects FINUFFT (host)
    or cuFINUFFT (device).
  - `setpts` / `fwdnufft` / `invnufft` dispatch on `nda::mem::on_host<>`.
  - `destroy_plan` dispatches on `nuplan_t::bend`.
  - `detail::check_dimensions` enforces array-vs-plan memory-space match.
- Wrapper class is now `math::nda::nufft_t<MEMORY_SPACE MEM = HOST_MEMORY>`.
  `math::nda::nufft` is preserved as `nufft_t<HOST_MEMORY>` alias for
  backward compatibility.

### Templated real-axis stack

- `methods/GW_real_axis/real_axis_conv.hpp`:
  - `grid_kind` lifted to namespace level.
  - Engine class renamed to `detail::real_axis_conv_base_t<MEMORY_SPACE MEM>`.
  - Aliases: `real_axis_conv_t = base_t<HOST_MEMORY>` (drop-in for
    existing host code) and `template<MEM> using real_axis_conv_mem_t =
    base_t<MEM>` (explicit-MEM device code).
  - Constructor: `_x_w`, `_x_Omega`, `_sgn_t` allocated in MEM via
    `memory::to_memory_space<MEM>`. Plans use `nufft_t<MEM>`.
  - Hilbert-kernel `_sgn_t = i*sgn(t_k)` is precomputed once at
    construction; the device kernel becomes a single fused multiply.
- All kernel functions (`accumulate_ImPi_one_kq`, `accumulate_ImSigma_one_kq{,_nufft}`,
  `RePi_from_ImPi`, `ReSigma_from_ImSigma_aux`, `primary_to_aux_one_k`,
  `aux_to_primary_one_k`) are templated on `MEMORY_SPACE MEM = HOST_MEMORY`
  and take `detail::real_axis_conv_base_t<MEM>&`.
- All driver functions (`evaluate_serial`, `evaluate_thc_serial`,
  `evaluate_Sigma_x_serial`, `run_scgw_serial`) are templated on
  `MEMORY_SPACE MEM = HOST_MEMORY` and take `memory::array<MEM, T, N>`.

### Inner ops already MEM-agnostic via `nda::map`

These convert host-only loops into nda lazy expressions that evaluate on
host or device depending on the array memory space. No `if constexpr`
gating — works for both:

- `cross_correlate`'s 2-arg Hadamard `Hhat = conj(F) * G`
- `convolve`'s 2-arg Hadamard `Hhat = F * G`
- `accumulate_ImPi_one_kq`'s 4-arg Hadamard
  `Hhat = conj(F<) * G> - conj(F>) * G<`
- `accumulate_ImSigma_one_kq_nufft`'s 4-arg Hadamard
  `Hhat = F1*G1 + F2*G2`
- `evaluate_serial` Step 5: `B = -Im W / pi`
- `run_scgw_serial`'s linear mix on A: `A_old = (1-a)*A_old + a*A_full`

### CMake plumbing for cuFINUFFT

- Top-level `CMakeLists.txt`: new option `ENABLE_CUFINUFFT` (OFF by
  default). When ON, sets `COQUI_HAVE_CUFINUFFT` compile flag and asserts
  `ENABLE_FINUFFT` is also ON.
- `cmake/finufft.cmake`: when `ENABLE_CUFINUFFT` is on, sets
  `FINUFFT_USE_CUDA=ON` in the FetchContent superbuild so cuFINUFFT is
  built alongside FINUFFT. Aliases the `cufinufft` target.
- `src/numerics/fft/CMakeLists.txt`: links `cufinufft` into `fft_lib` when
  `ENABLE_CUFINUFFT=ON`. Otherwise `cufinufft.cpp` builds as runtime-aborting
  stubs.

---

## What needs to happen on Rusty

### 1. Build with cuFINUFFT

```bash
# In the build dir:
cmake -DENABLE_CUFINUFFT=ON \
      -DENABLE_DEVICE=ON \
      -DCMAKE_CUDA_ARCHITECTURES=80 \   # adjust for the GPU on the node
      <other usual flags> \
      <path-to-source>
make -j fft_lib test_methods_gw_real_axis
```

If the build fails with `cufinufft target not found`, check the FINUFFT
version pinned in `cmake/finufft.cmake` (`FINUFFT_GIT_TAG`) and confirm
that release supports `FINUFFT_USE_CUDA=ON`.

### 2. Validate the cuFINUFFT bindings layer

The simplest check: a small unit test that creates a `nufft_t<DEVICE_MEMORY>`
plan, runs a known transform, and compares against the host result. The
existing host tests in `test_finufft_nda.cpp` give the templates; copying
them with `nufft_t<DEVICE_MEMORY>` and `nda::cuarray<...>` arrays should
produce a matching answer.

`test_finufft_nda.cpp` now ships with three such cases gated on
`#if defined(COQUI_HAVE_CUFINUFFT)`:
`cufinufft_nda_1d_device_vs_host_double`, `..._2d_...`, and
`cufinufft_nda_RAII_device_vs_host_double`. Each runs a host plan and a
device plan on the same random input, pulls the device output back via
`nda::to_host`, and asserts element-wise agreement to NUFFT eps. On
host-only builds these cases are excluded from the suite at compile
time. On Rusty with `-DENABLE_CUFINUFFT=ON` they should be the first
green light before any of the kernel-level work.

### 3. Implement the per-element device kernels

These are the remaining bodies that are still host-only with
`utils::check(false, ...)` device branches. Order of payoff (largest
kernel-call counts first):

#### High-priority (in `real_axis_conv_base_t<MEM>`)

These run inside the hot loop of `evaluate_serial` Steps 2 and 6.

- **Weight broadcast** in `cross_correlate` / `convolve`:
  ```cpp
  F(b, j) = F_in(b, j) * wq(j);   // 1D wq broadcast over 2D F
  ```
  Candidates: `nda::tensor::elementwise(...,"j", ..., "bj", op::MUL)` or a
  small CUDA kernel. The 1D × 2D broadcast pattern appears in
  `methods/ERI/thc.icc:1712` for sqrtVg × Z; copy that idiom.

- **`hilbert` weight + sgn(t) multiply**:
  ```cpp
  C(b, j) = ImX(b, j) * wq(j) + 0i;          // 1D × 2D broadcast
  Hhat(b, k) = _sgn_t(k) * Chat(b, k);       // 1D × 2D broadcast
  ReX(b, j) = s * Rraw(b, j).real();         // scalar*array, plus .real() lift
  ```
  All 1D × 2D, same pattern as above. `_sgn_t` is already in MEM-space.

#### Medium-priority (in the kernel functions)

- **`accumulate_ImPi_one_kq`**: weighted-spectra construction
  ```cpp
  Aless_k(b, j) = f(w(j)) * wq(j) * A_PQ_k(P, Q, j);
  ```
  This is index-tied (P, Q indices map into batch b) and includes a 1D
  scalar function call (`grid.fermi`). A device kernel that takes the
  per-band weight arrays plus `A_PQ_k` is the clean port.

- **`accumulate_ImSigma_one_kq_nufft`**: similar to Im Pi above, plus the
  `resample_bosonic_to_fermionic` interpolation.

- **`RePi_from_ImPi` / `ReSigma_from_ImSigma_aux`**: (P, Q) ↔ batch (B = Naux²)
  gather/scatter. These are pure copies — `nda::reshape` + assignment
  should work on both host and device.

- **`primary_to_aux_one_k` / `aux_to_primary_one_k`**: intermediate
  permutations between two GEMMs. The GEMMs themselves go through cuBLAS
  automatically. The permutations `(mu, iw, nu) <-> (P, iw, nu)` and
  `(P, Q, iw) <-> (P, iw, Q)` are also pure copies — likely doable with
  `nda::permutation` views or an explicit kernel.

#### Lower-priority (driver-body conversions)

These are inside `evaluate_serial`, used in cheap steps:

- **Step 4 (Dyson Pi assembly)**: `Pi(P, Q) = Re Pi + i*Im Pi`. Per-(q, iO)
  loop with a Naux × Naux solve. Solve goes through cuBLAS LAPACK; the
  Pi assembly is one nda::map.

- **Step 7 / Step 8**: pack/unpack between (P, Q, iw) and (Naux², iw)
  layouts for the batched Hilbert. Pure copies.

- **`evaluate_Sigma_x_serial`**: `n_skij(s,k,mu,nu) += f(w) * wq * A(s,k,iw,mu,nu)`
  reduction over iw, plus a Hadamard `V(q, P, Q) * n_aux(s,k-q,P,Q)`.

- **`run_scgw_serial`**: `dyson_update_A`, `find_mu_chem`, DIIS mixer.
  These are all per-element nda::map candidates plus a per-(s,k,iw)
  matrix invert (cuBLAS / cuSOLVER).

### 4. Lift the static_asserts

Once a method's inner work is MEM-agnostic, drop the
`static_assert(MEM == HOST_MEMORY)` at the top. The plan was to do this
incrementally so each lift is its own commit and the host suite stays
green at every step.

Recommended order:
1. `real_axis_conv_base_t<MEM>::cross_correlate` first (the simplest:
   weight broadcast + Hadamard already in `nda::map`). **DONE.**
2. `real_axis_conv_base_t<MEM>::convolve` (same). **DONE.**
3. `real_axis_conv_base_t<MEM>::hilbert` (sgn-multiply + weight; harder).
   **DONE.**
4. `real_axis_conv_base_t<MEM>::apply_weights` (both overloads). **DONE.**
5. `accumulate_ImPi_one_kq`, `accumulate_ImSigma_one_kq_nufft`.
6. `RePi_from_ImPi`, `ReSigma_from_ImSigma_aux` (gather/scatter).
7. `primary_to_aux_one_k`, `aux_to_primary_one_k` (permutations).
8. `evaluate_serial` body (Steps 4, 5, 7, 8 elementwise ops).
9. `evaluate_Sigma_x_serial`, `run_scgw_serial`, DIIS mixer.

### Conv-class lift details (steps 1–4 above)

The four `real_axis_conv_base_t` methods now compile and dispatch on both
host and device:

- The 1D × 2D weight broadcast (`F(b, j) = F_in(b, j) * wq(j)`) goes
  through `nda::tensor::elementwise(... "j", ..., "bj", op::MUL)` on
  device — the same idiom used by `methods/ERI/thc.icc` for `sqrtVg × Z`.
  On host the original double-loop is preserved (cuTENSOR's host fallback
  does not broadcast).
- Complex MEM-side copies of the trapezoidal weights (`_w_weights_c`,
  `_Omega_weights_c`) are precomputed once at construction so the device
  broadcast can match value-types without per-call allocs. The imag slot
  is zero; the real path keeps reading `_grid->w_weights()` directly.
- The `(dt / 2π)` post-NUFFT scaling and the `Re(Rraw)` extraction in
  `hilbert` use single-arg `nda::map` lambdas, which are MEM-agnostic.
- `apply_weights(rarray_t<2>&)` stages through a complex tmp buffer for
  the device path (`real -> cval_t(*, 0) -> broadcast -> .real()`) so the
  cuTENSOR call still sees same-typed operands. Only ever called by the
  conv class itself; no external callers in the current tree.

Tests on host: full `test_methods_gw_real_axis` suite (44 cases / 16838
assertions) bit-identical after the lift; FFT suite (309 assertions /
7 cases) untouched.

### 5. End-to-end test on a real fixture

The `test_real_axis_thc_g0w0_lih222_kspace_vs_rspace` regression test
already validates k-space vs R-space at machine precision. A natural
`MEM=DEVICE_MEMORY` regression test compares host vs device outputs at
the same precision target.

The xvalidate test (real-axis vs Matsubara) gives a separate sanity
check: agreement to ~3e-3 over the lowest 8 Matsubara points.

---

## Validation strategy on Rusty

1. **Unit-level**: copy each MEM-templated test to also instantiate
   `MEM=DEVICE_MEMORY`, compare against host at machine precision.
2. **Kernel-level**: a `MEM=DEVICE_MEMORY` instantiation of
   `accumulate_ImPi_one_kq` on a synthetic fixture; compare the resulting
   `ImPi_PQ_O` against the host result element-wise.
3. **End-to-end**: re-run the LiH222 G0W0 e2e test with
   `MEM=DEVICE_MEMORY`. Expected: same outputs as host within NUFFT eps.
4. **Perf**: nsight-compute on the inner kernels; nsight-systems on the
   full LiH222 e2e to identify any host↔device transfer hot spots.

The pre-existing R-space win on host (4.4× single-rank) is a useful
baseline. A1100/H100 should give another 5-10× on the NUFFT-dominated
steps, putting LiH222 G0W0 in the sub-second territory per iteration.

---

## Cross-cutting status (2026-04-29 evening)

The MEM-templated kernel layer is complete in **both** the real-axis
and imag-axis stacks:

- Real-axis: `update_w<MEM>` and `gw_t::evaluate<MEM>` are end-to-end
  MEM-aware, with on-device caches for the conv plan, THC factor X,
  R-space FT factors, and BZ kpq map. SCF / QP-SCF drivers thread MEM
  through. ENABLE_DEVICE-gated `host_vs_device` tests added for both
  the kernel and the full SCF loop. Validated on Rusty A100 at the
  kernel level (9 cases / 4496 assertions). End-to-end SCF timing on
  A100 pending an interactive session.

- Imag-axis: `eval_Pi_rpa_Rspace<MEM>`, `eval_Sigma_all_Rspace<MEM,
  Winp_in_R, Wout_in_R>` MEM-templated and host-bit-identical
  (44 + 12 cases / 16838 + 67 assertions). The k-space variants and
  `thc_hf_Xqindep` are templated with `static_assert(MEM ==
  HOST_MEMORY)` for now (R-space is the GPU default; HF is
  sub-leading). What remains: a `scr_coulomb_t::update_w<MEM>` and
  `gw_t::evaluate<MEM>` overload that build device-side dW/Sigma and
  copy back to host MBState, plus the slate Dyson refactor
  (`dyson_W_in_place`'s slate dPi/dZ/dA triplet is HOST-hardcoded).
  See `notes/gpu_port_status_2026-04-29.md` for the detailed plan.
