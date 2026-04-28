# Real-axis GW: GPU port status and Rusty handoff

State after the structural-templating commits on `real_axis`. Everything in
this document is the framework — execution validation needs a Rusty session
with cuFINUFFT installed.

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
   weight broadcast + Hadamard already in `nda::map`).
2. `real_axis_conv_base_t<MEM>::convolve` (same).
3. `real_axis_conv_base_t<MEM>::hilbert` (sgn-multiply + weight; harder).
4. `accumulate_ImPi_one_kq`, `accumulate_ImSigma_one_kq_nufft`.
5. `RePi_from_ImPi`, `ReSigma_from_ImSigma_aux` (gather/scatter).
6. `primary_to_aux_one_k`, `aux_to_primary_one_k` (permutations).
7. `evaluate_serial` body (Steps 4, 5, 7, 8 elementwise ops).
8. `evaluate_Sigma_x_serial`, `run_scgw_serial`, DIIS mixer.

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
