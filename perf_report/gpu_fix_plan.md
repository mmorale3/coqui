# CoQuí GPU: plan to a working, performant, efficient device path

> **READ FIRST — status as of 2026-08-02.** Since this plan was written, a **one-line bug in
> the GPU ERI builder** was found and fixed, and it changes the priorities below. See
> **§A. Handover** at the end of this file for the current state, what is committed, what is
> in flight, and exactly what to do next. §0-§6 remain valid as the SLATE/device analysis, with
> one correction: §0's framing of the block-cyclic port as the way to fix reproducibility was
> wrong — the reproducibility failure was the ERI bug, not the layout.

Status date: 2026-07-30. Branch `gpu`. Baseline: 8×A100-80GB, Si kp222/500b, SCF loop
**141 s/iter** (was 244 s/iter before this July's work).

Evidence tags used below: **[proven]** = reproduced with a stack trace or bit-level A/B on
rusty; **[measured]** = timing/number from an instrumented run; **[hypothesis]** = reasoned
from source, not yet tested.

---

## 0. The one structural finding that reorganises everything

CoQuí hands SLATE **views** of nda device blocks (`to_slate_view` → `tileInsert`). Two
independent facts make that unworkable as written:

1. `tileInsert(i,j,ptr,ld)` defaults to `device=HostNum`, so SLATE believed every device
   block was host memory and ran CPU LAPACK/BLAS and non-GPU-aware MPI directly on device
   pointers. **[proven]** — pre-fix backtrace:
   `slate::gesv → getrf → getrf_panel<HostTask> → tile::getrf → Tile::operator()(i,j)`,
   UCX "invalid permissions for mapped object". Fixed by tagging the tile with its device.

2. Even correctly tagged, SLATE's **batched** device path cannot use tiles shaped like ours.
   This is not a SLATE scalability limit — SLATE does distributed GPU linear algebra at
   scale by design. It is that we are using a mode SLATE's batched path cannot express.

   **The distinction that matters is who owns the device tiles.**

   *SLATE's canonical large-scale mode* is a **host-origin** matrix (`fromScaLAPACK`, or
   SLATE-allocated host tiles) run with `Target::Devices`. Per operation SLATE works out
   which tiles each device needs, allocates **its own** tiles from its per-device pool,
   copies in, runs batched cuBLAS, copies results back to the host origin. Every device tile
   SLATE touches is one it allocated, so all of them are compact — `stride == mb`. Uniform
   by construction.

   *Our mode* hands SLATE tiles that are **views into a larger nda block**, whose leading
   dimension is the local block's row count, not the tile height (`stride == lld != mb`).
   With defect 1 fixed these are honest **device-origin** tiles. Now:

   - `internal::gemm<Target::Devices>` builds its batch argument arrays via
     `device_regions_build`.
   - Regions come from `device_regions_range`, which scans **tile dimensions only**
     (`tileMb`/`tileNb`) and groups runs of equal-sized tiles. **Stride is not part of the
     grouping key.**
   - Within a region it records the first tile's stride per operand and asserts the rest
     match: `assert(group.ld[m] == Mij.stride())` (`internal_batch.hh:290`). That assert is
     enforcing a genuine constraint of batched BLAS — one batched cuBLAS call takes a
     *single* `ld` per operand.
   - On a device-origin matrix a rank's own tiles are origin tiles (`stride = lld`), while
     tiles it needs from other ranks arrive as SLATE workspace copies (`stride = mb`). A
     distributed gemm's k-loop spans the whole matrix, so any rank needs both kinds in the
     same operand ⇒ two strides in one region ⇒ assert.

   On a host-origin matrix this cannot happen, because *every* device tile is a workspace
   copy. That is precisely why the canonical mode scales. Note SLATE's own device-origin
   constructor `fromDevices` (single uniform `lda`) would hit the same wall as soon as a rank
   holds more than one tile row — so this is a property of device-origin matrices in the
   batched path, not something peculiar to CoQuí. SLATE version here is 2025.05.28
   (ABI 2.0.0), i.e. current; this is not an old-version artefact.

   **[proven]** — assert fires at 6 and 12 ranks from a top-level `slate::gemm`
   (`slate_ops::multiply(Z_quG, dagger(Z_quG), C_quv)`); CPU at the same (4,3,1) grid passes.

**Consequence:** to get a distributed device solve we must use SLATE's canonical mode one
level down — let SLATE **own** the device tiles, and copy our block in and the result out.
That is the same staging a `fromScaLAPACK` user pays, except ours is D2D (~1.5 TB/s) rather
than over PCIe. Uniform strides by construction, full batched GPU execution, and it scales.
This is §1.2c, and it is the canonical usage rather than a workaround.

*Rejected no-copy alternative:* make `distributed_array` store its local block in tile-major
order so `lld == mb`. That would remove the copy but changes the memory layout every other
consumer of `.local()` depends on (it is treated as a plain contiguous 2D array throughout).
Not worth it for a copy that is negligible against O(n³).

Corollary that pays for itself immediately: defect **B3** ("device ISDF/THC builder
segfaults at ≥16 ranks") is not about scale. The ERI grid comes from
`find_proc_grid_max_npools(np, nqpts_ibz, distr_tol)`, and SLATE is only entered when the
leftover ranks distribute the (u,v) axes:

| np | grid | inner comm | SLATE entered? |
|---|---|---|---|
| 4  | (4,1,1) | 1 | no — `comm.size()==1` shortcut to nda/cuBLAS |
| 8  | (8,1,1) | 1 | no |
| 6  | (2,3,1) | 3 | yes |
| 12 | (4,3,1) | 3 | yes |
| 16 | (8,2,1) | 2 | yes |

16 is just the smallest power-of-two rank count that distributes those axes. **B3
reproduces on one node at 6 or 12 ranks**, which is how the stack trace above was obtained.
Fold this into the harness; stop needing 4-node jobs to debug it.

---

## 1. Phase 1 — make the device SLATE path correct (blocks all large GPU runs)

### 1.1 Landed and validated
- `make_slate`: tag device tiles with their device. **[proven]** numerically neutral —
  4-rank device ERI build + 2 GW iterations **bit-identical** old vs new (6 ERI + 37 GW
  datasets, worst diff 0.0); shared-ERI GW matches the published reference to 4e-15 (4
  ranks) and 7e-15 (12 ranks).
- ERI class: retired the `unified_array` hack in `intvec_impl` (the
  `// MAM: temporary HACK until I figure out why slate::lu_solve seg faults with device
  memory` alias) plus the host element-writes it was covering for. Bit-identical.
- `slate_ops::inverse()`: in-place `slate::getri` is host-only by SLATE's own admission
  ("This routine is in-place and does not support GPUs"), as is the `trtri` it calls.
  Replaced with out-of-place `getri` (= `set(I)` + `getrs`, both device-capable) +
  `slate::copy` back. **Compiled but not exercised** — the Pi/W grid is (1,4,1,1) at 4
  ranks and (3,4,1,1) at 12, so `wq_intra_comm.size()==1` and `inverse()` takes the serial
  `nda::lapack` path. Needs a `np_P*np_Q > 1` configuration (see §4.2).

### 1.2a `Target::HostTask` — rejected

**P1a — `Target::HostTask` fallback: REJECTED, do not implement.** It was in an earlier
draft of this plan. Two reasons it is wrong:

1. It does exactly what the name says — the factorisation runs on the **CPU** even though a
   GPU is sitting there. For the ERI build that is a one-time cost, but it is still throwing
   away the machine.
2. Worse, it is **silently incorrect**. With `Target::HostTask` on device-origin tiles,
   SLATE fetches each tile into a *host workspace* copy, computes there, marks the host
   instance Modified and the device instance Invalid — and then never copies back, because
   **none of `gesv`, `getrs`, `gels`, `getri`, `trsm` calls `tileUpdateAllOrigin()`**
   (verified in the SLATE source; only `gemmC` does). The caller then reads `A.local()` on
   the device and gets **stale data with no error**. That is precisely the class of bug this
   whole exercise is removing. Only `slate_ops::multiply` would survive.

So there is no cheap correctness stopgap through SLATE. Use §1.2b for the ERI build and
§1.2c for the SCF loop.

### 1.2b Two solve regimes, selected by Np

`Np` ranges from ~1.3k (nbnd=100 test) through ~4.5k (kp222/500b) to a target of
**10k–40k**. `Ciq` is Np × Np complex double:

| Np | `Ciq` global | replicated per rank | distributed over 64 GPUs |
|---|---|---|---|
| 1 275 | 26 MB | 26 MB | — |
| 4 483 | 321 MB | 321 MB | 5 MB |
| 10 000 | 1.6 GB | 1.6 GB | 25 MB |
| 40 000 | **25.6 GB** | **impossible** | 400 MB |

So both paths are needed, exactly as you say, and the crossover is a memory argument:

- **Regime A — replicate (small/moderate Np).** Cheapest possible: no communication in the
  factorisation at all. Viable to ~10k (1.6 GB/rank), which covers everything run today.
- **Regime B — genuine distributed device solve (large Np).** Mandatory from ~10k up; at
  40k replication is off the table. This is §1.2c, and it is where SLATE earns its keep.

Select on `Np` against available device memory, with an override, and log which path was
taken. Regime A is also the natural fallback if B misbehaves, so the two are complementary
rather than redundant.

#### Regime A: replicate the small matrix, solve on device, no SLATE

The ERI solve has structure worth exploiting, and `thc.icc:1576` already carries the note
`// MAM: leave option for round-robin serial solves!`.

In `intvec_impl` the system is `Ciq · X = Ziq` with `Ciq` **nIpts × nIpts** and `Ziq`
**nIpts × nG**. At Np=4483 that is 321 MB for `Ciq` versus multi-GB for `Ziq` — the operator
is small and the right-hand side is huge. So do not factor `Ciq` in a distributed way at
all:

1. gather/replicate `Ciq` within the q-pool (321 MB per rank at Np=4483, 26 MB at nbnd=100),
2. factor it **on the device** with cuSOLVER (`nda::lapack::getrf`, the path
   `slate_ops::lu_solve` already takes when `comm.size()==1`),
3. apply `getrs` locally to each rank's own **column block** of `Ziq`.

Step 3 needs `Ziq` distributed by columns only; it is currently 2D over (u, G), so it needs
one `math::nda::redistribute` each way — a primitive that already exists and already has a
device-direct path. **[hypothesis]** on the redistribute cost; the algorithmic shape (small
operator, wide RHS ⇒ replicate the factorisation, parallelise the apply) is not in doubt and
is strictly less communication than a distributed LU.

This removes SLATE from the ERI build entirely, keeps everything on the GPU, and closes B3
without depending on §1.2c.

#### Regime B / §1.2c: SLATE device mirror — the real distributed device solve

This is the general fix and it serves **both** the large-Np ERI solve and the SCF loop
(`dyson_W_in_place`'s `inverse()`, where `A = I − Z·Π` is the same NP×NQ as everything else
and so has no small-operator structure to exploit).

Give SLATE tiles it allocated itself:
a `slate_mirror` RAII helper that, for device memory, does
`insertLocalTiles(Target::Devices)` + copy-in, and on `write_back()` copies out; for host
memory it degenerates to today's zero-copy view. Then all of `multiply`, `lu_solve`,
`least_squares_solve`, `inverse` run on the GPU with uniform-stride tiles.
- Cost: one D2D copy of the local block each way. At kp222/500b that is ~80 MB at
  ~1.5 TB/s ≈ 0.05 ms — negligible versus an O(n³) solve. **[hypothesis]** on the exact
  numbers, but the order of magnitude is not in doubt.
- Extra device memory: one more copy of the local block, from SLATE's own pool.
- Touches ~5 call sites in `slate_ops.hpp` plus the direct `to_slate_view` use in
  `thc.icc:1889`.
- **Bonus:** the mirror can convert layout during copy-in, which removes the blocker that
  killed **P7** (LU-solve instead of explicit inverse in `dyson_W_in_place`). P7 was backed
  out because `slate_ops::lu_solve` requires F-order while `dA_PQ`/`dZ_PQ`/`dPi_PQ` are
  C-order, and converting that buffer trio is invasive. With the mirror the conversion is
  local and free-ish. That is **−4 s/iter** [measured 8.3 s inverse + 3.7 s gemm → ~8/3 n³].

### 1.2d Unified memory as the alternative — legitimate, and it works

Worth stating plainly: **genuine unified memory solves defects 1 and 3, and it does so by
putting us in exactly SLATE's canonical mode.** An earlier draft of this plan called the
unified workaround "working only by accident"; that was wrong and is corrected here. The
accident was the *mislabelling*, not the concept.

Why it works. With `nda::heap<mem::Unified>`, `on_host` is false so `make_slate` picks
`dev` for `tileDevice`, but `on_device` is also false so the tile is inserted at **HostNum**
— a *host-origin* matrix whose pointer happens to be host-dereferenceable. Meanwhile
`slate_ops`' `_dev_` (= `have_device_compatible_addr_space`) is true, so execution is
`Target::Devices`. That is host-origin + Target::Devices: SLATE allocates its own compact
device workspace tiles, so strides are uniform (**no §0 defect 3**), and every host-side
touch — `getrf` panels, `getri`, `trtri`, non-GPU-aware MPI — is legal (**no §0 defect 1**).

**This path is intact in the current code**: `tiles_on_device = on_device<Array_t>`
deliberately excludes Unified, so nothing needs reverting to use it.

**The one thing that must change to do it correctly:** the arrays must be *actually*
unified where they are allocated — `local_3Array_t` in `get_ZquG_Cquv_fft`
(`thc_aux.icc:1107`) — **not** a unified-typed `darray_view_t` over device memory. The old
code did the latter, and that lie is what produced the `tile::getrf` segfault: a HostNum tile
holding a `cudaMalloc` pointer. `thc_aux.icc:811-813` shows this was already partly reverted.
The `distributed_array_view` static_assert in §1.3 is what makes the distinction
unmissable, and is worth adding whichever route is taken.

**Trade-offs versus the §1.2c mirror.**

| | unified | mirror (SLATE-owned device tiles) |
|---|---|---|
| code change | smallest — change array types, no `slate_ops` work | ~150 lines in `slate_ops`/`slate_aux` |
| where the matrix lives | managed; migrates on demand | resident on the GPU |
| data movement | implicit, fault-driven page migration | explicit bulk D2D, ~1.5 TB/s |
| profileability | poor — migration is invisible in most traces | explicit, shows up as copies |
| worst case | host/device ping-pong every `getrf` panel (≈93 panels at Np=40k, nb≈430) | panel round-trip only, same as any SLATE user |
| tuning needed | `cudaMemAdvise`/`cudaMemPrefetchAsync` to avoid fault-driven migration, which erodes the simplicity | none |
| footprint | can oversubscribe device memory ⇒ OOM becomes silent thrashing | fails loudly, predictable |
| blast radius | `C_quv`/`Z_quG` are produced by heavy cuTENSOR contractions in the Zqur stage and only *then* solved, so making them unified changes the whole builder's characteristics, not just the solve | confined to the SLATE boundary |

**Recommendation: unified is the better *short* path, the mirror the better *end state*.**
If a working large-Np distributed device solve is wanted with minimum risk and code, go
unified — it is a few type changes and it lands in SLATE's supported mode. Prefer the mirror
for the durable version, because it keeps the hot arrays in plain device memory, confines the
cost to the solve, and is visible in a profiler.

**One caution specific to this codebase:** **B6** (device ISDF selects zero Cholesky vectors,
sm90/H100 only, A100 and CPU fine) is still unexplained, and managed-memory behaviour is one
of the things that differs most across architectures and drivers. `chol_metric_impl` already
uses a `memory::unified_array` for `Abb`. Broadening unified usage while an
architecture-specific mystery is open risks entangling the two. Either understand B6 first,
or keep unified confined to the arrays that reach SLATE and record that choice.

### 1.2e A dedicated SLATE-backed distributed array — RECOMMENDED

Miguel's proposal, and it is better than §1.2c. Add a **new distributed-array type used only
for matrices that feed SLATE's distributed linear algebra**, laid out so the stride problem
cannot arise, device-resident, and *redistributed into* rather than copied into.

**Why this dominates the mirror.** `C_quv` and `Z_quG` are **already redistributed at the end
of construction** — `get_ZquG_Cquv_fft` does
`redistribute_in_place(C_quv, …)` / `redistribute_in_place(Z_qug, …)` (`thc_aux.icc:1716-1717`)
immediately before returning them to `intvec_impl` for the solve. So if `redistribute` can
target the new layout, **the copy-in is absorbed into a data movement that already happens**
and costs nothing extra. The mirror's copy-in is pure overhead by comparison.

**Two storage variants; prefer (ii).**

- (i) Wrap `slate::Matrix` with `insertLocalTiles(Target::Devices)` — SLATE allocates and
  owns the tiles from its per-device pool. Least code, but the local data is then scattered
  across pool blocks, which makes a `redistribute` target awkward.
- (ii) **Own one contiguous device buffer in tile-major order** (tiles laid out back to back,
  each stored `mb × nb`), and `tileInsert` pointers into it with `stride == mb`. Then *our*
  tiles and SLATE's workspace tiles both have `stride == mb` ⇒ uniform ⇒ the
  `device_regions_build` assert is satisfied by construction. Edge tiles carry their own
  smaller `mb`, which lands them in a different region anyway, so that is consistent too.
  Single contiguous buffer keeps `redistribute` simple and reuses the existing device-direct
  pairwise exchange.

This is the "tile-major storage" idea an earlier draft rejected — the rejection was wrong,
because it assumed changing `distributed_array` itself. Confining it to a **separate type**
leaves every existing `.local()` consumer untouched, which is the move that makes it work.

**What it buys beyond correctness**
- Type-level guarantee: it becomes impossible to hand SLATE a view with a foreign leading
  dimension. Strictly stronger than the §1.3 static_assert.
- No unified memory ⇒ no page-migration unpredictability, and no entanglement with B6.
- `make_slate`'s device-view path becomes unnecessary for these arrays, so the device
  tile-tagging in §1.1 can eventually be retired and `to_slate_view` simplified back to
  host-only. §1.1 becomes a stepping stone rather than the end state.
- The class controls its own layout, so it can be Fortran-ordered internally — which
  removes the F-layout blocker that killed **P7** without converting the
  `dPi_PQ`/`dZ_PQ`/`dA_PQ` trio in place.
- Serves regime B of §1.2b (large-Np ERI solve) and `dyson_W_in_place` with one mechanism.

**The one real constraint.** Tile-major storage is not a plain contiguous 2D block, so this
type **cannot expose `.local()` as an `nda` 2D view**. Anything that wants that needs a
conversion. In `intvec_impl` the post-solve code does exactly that — `Zloc = Z_quG.local()`
for the G=0 extraction and the `sqrtVg` `tensor::elementwise` — so expect one copy/redistribute
back to a normal device darray after the solve. Net versus the mirror: copy-in free, copy-out
still paid. Design the interface with that boundary explicit (e.g. `to_darray()` /
`from_darray()`), rather than pretending the type is a drop-in `distributed_array`.

**Effort:** the class plus a `redistribute` overload targeting it, plus `slate_ops` accepting
it. Estimate ~300-400 lines. The redistribute change is confined to the unpack index math —
`redistribute_alltoallv` already computes explicit per-peer chunk indices, so a tile-major
destination is a variation on that loop, not a rewrite.

**Fix while building it — latent `mt` bug.** `make_slate` computes `mt = m/mb` (**floor**,
`slate_aux.hpp:90`) whereas SLATE's `Matrix` uses `ceil`. Whenever `m % mb != 0` SLATE has one
more tile row than the `tileRank` lambda was built for, so the last partial tile row is
assigned by extrapolation plus `std::min(p-1, …)` clamping and can disagree with where the
data actually is. This is **live in the failing configuration** (Np=1292, block 430 ⇒ floor 3
vs SLATE's 4). Not the cause of the stride assert — that mechanism is proven — but a real
second-order bug in the same function, and the new class must get this right.

### 1.3 Two latent traps to close while in this code

- `tileDevice` returns **one constant device for every tile**, from `cudaGetDevice()`,
  while SLATE loops `device = 0 … num_devices()-1` selecting tiles with
  `device == tileDevice(i,j)`. This is only self-consistent when each process sees exactly
  one GPU.

  **It is not a test artifact — it is live in production.** Job 6711719 (12 ranks, 3 nodes,
  `--ntasks-per-node=4 --gpus-per-node=4`, i.e. the designed 1-rank-per-GPU layout, same
  convention as `bench_gpu_4n/run.sbatch`) reports
  `CUDA_VISIBLE_DEVICES=0,1,2,3` **for every rank**. So `num_devices()==4` and
  `cuda_init.cpp:72`'s `cudaSetDevice(node.rank() % num_devices)` gives a *different* `dev`
  per rank on the same node, in exactly the way the oversubscribed runs did.

  Failure mode tracks the visible device count, in both packings **[proven]**:

  | visible GPUs/rank | ranks × nodes | pre-fix binary | post-fix binary | job |
  |---|---|---|---|---|
  | 4 (`--gpus-per-node=4`, 1 rank/GPU) | 12 × 3 | segfault in `tile::getrf` | `cudaErrorIllegalAddress` | 6711719 |
  | 2 (oversubscribed) | 12 × 1 | segfault in `tile::getrf` | `cudaErrorIllegalAddress` | 6710805 |
  | 1 | 12 × 1, 6 × 1 | segfault in `tile::getrf` | clean `group.ld` stride assert | 6710883 |
  | 1 via `--gpu-bind=single:1` | 12 × 1 | segfault in `tile::getrf` | clean `group.ld` stride assert | 6711771 |

  Three conclusions, all **[proven]**. (a) The root-cause diagnosis holds in every packing —
  the pre-fix binary segfaults in SLATE's host panel kernel in all four. (b) With more than
  one visible device there is a *second*, independent defect that masks the stride assert.
  (c) Adding `--gpu-bind=single:1` collapses row 1 onto row 4: `CUDA_VISIBLE_DEVICES=0` for
  every rank, and the illegal access is replaced by the honest stride assert.

  **Fix, two parts.** First, **bind one GPU per rank at launch**
  (`srun --gpu-bind=single:1`, or a wrapper setting `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`)
  so `num_devices()==1` and `dev==0` everywhere. This matches the code's own design
  assumption, is a one-line change per script, and **every `run.sbatch` in `GPU_PORT_run`
  currently lacks it** — including the production `bench_gpu_*` scripts. Make it the
  documented launch convention. Second, make the glue defend itself: `utils::check` in
  `make_slate` that either `num_devices()==1` or all node-local ranks agree on `dev`, with a
  message naming the binding flag, instead of degrading into an illegal memory access.
- `distributed_array_view` (`nda_matrix.hpp:388-434`) **silently reinterprets the address
  space**: `OwningPolicy = borrowed<get_addr_space<Base_t>>` comes from the template
  argument while the constructor just takes `A_.data()`. That is how the unified hack was
  written, and it means `darray_view_t<unified_array<…>>` over a device array compiles and
  lies. Add a `static_assert` that the argument's address space matches — and fix the
  remaining sites that rely on the reinterpretation (`thc_aux.icc:970`,
  `pproc/hamiltonians.cpp:211`).

### 1.4 Do not re-attempt
- `SLATE_GPU_AWARE_MPI=1` segfaults in `slate::Tile::recv` (UCX) in this environment.
  **[proven]** Leave unset; SLATE's host staging is correct now. Consistent with the
  earlier NCCL result (NCCL 2-4× slower than MPI here; MPI already uses NVLink).
- The unused `to_slate()` copying path is now `static_assert`-ed host-only. P1b supersedes
  it — implement the mirror rather than reviving that function.

---

## 2. Phase 2 — close the remaining device-path defects

| id | defect | proposed fix | effort |
|---|---|---|---|
| **B3** | device ISDF/THC builder fails once the grid distributes (u,v) | §1.2 P1a, then P1b. Re-test at 6/12/16 ranks. | follows from §1 |
| **B6** | device ISDF selects **zero** Cholesky vectors on sm90/H100; A100 and CPU fine | Not SLATE — `chol_metric_impl` uses cuTENSOR + `utils::chol` on a unified `Abb`, no SLATE. Bisect properly: dump the residual diagonal and `Abb` at iteration 0 on A100 vs H100 and diff. Prime suspect is a cuTENSOR 2.2 contraction on sm90 (`contract(conj(Lr),"skpur",Lr,"skpur",Tab,"ur")`); second is `utils::max_element_multi`. **Do not guess — instrument.** | 1 job + fix |
| **R1** | THC point selection depends on the processor grid (Np = 4471…4483; ~1e-4 a.u. spread) | Root cause is the greedy pivot search: `find_distributed_maximum` breaks ties by rank, so the selection order is distribution-dependent. Make it deterministic (global index ordering with a fixed tie-break). Then cross-rank-count comparison becomes possible and the "always share a saved ERI" rule can be relaxed. | moderate |
| **B4** | µ-search bracketing walk silent and uncapped (0.2 a.u. steps, no limit) — turned a wrong Σ into a 4 h wall-clock hang | Iteration cap + bracket sanity check + `app_log` of each step. Trivial and pure diagnosability win. | small |
| **B5** | `compute_eigenspectra` allocates the full Σ(ω) *per rank* — 35 GB/rank at kp444/500b | Chunk or distribute over ω. Currently forces kp444 CPU jobs down to 24 ranks/node. | moderate |
| — | Zqur stage is ~390 lines with **no logging** between two `app_log(2)` calls, so any failure in it is unlocalisable (cost real time this session) | Add stage markers + `memory_report` at verbosity 3. | small |
| — | `TimerManager::elapsed()`/`number_of_calls()` APP_ABORT on a name that was never `start()`ed, so any one-sided sub-timer crashes the other branch at the first dump | Make them return 0 for unknown names instead of aborting. Removes a whole class of self-inflicted build-cycle losses. | small |

---

## 3. Phase 3 — performance, in measured priority order

Per-iteration budget now (8×A100, kp222/500b, 141 s/iter) **[measured]**:
`update_w 49.5 | Σ(R)+div_corr 25.2 | write 30.2 (→~17 steady-state with async) | Dyson ~18 | mixing 6.0 | HF 3.6 | eigenspectra 2.5 | energies 1.7 | unaccounted 5.3`

1. **G/Σ device residency — the structural item.** `G_tskij` and `Sigma_tskij` (4.4 GB each)
   live in **host shared memory** and are the currency of the SCF loop; every device kernel
   pulls slices up and pushes results back. Target: device-resident for the whole iteration,
   host sees them only for the HDF5 checkpoint. This is the enabler for items 2 and 3 below
   and removes the remaining per-(t,s,k) H2D/D2H traffic. Biggest single win left and the
   most invasive — do it deliberately, behind a flag, with the bit-level harness.
2. **Σ_div_correction, remaining half** — move the per-(t,s,k) gemms to the device. Needs
   G device-resident (item 1). [Its first half already landed: one gemm per (s,k) for the
   T-build, and skipping the Σ-sized staging buffer on one node.]
3. **Device Dyson (P6)** — batched LU priced at **0.27 ms per 500×500 block**
   [measured, `cublasZgetrfBatched`]. Worth ~3 s/iter now that the Dyson loop is 6.4 s/2
   iters and the rest is FT + irreducible communication.
4. **P7 LU-solve in `dyson_W_in_place`** — **−4 s/iter**, unblocked by §1.2 P1b's layout
   conversion. 8/3 n³ instead of 4 n³.
5. **Hunt the remaining `to_host`-the-whole-tensor callers.** The same pattern in
   `IAFT::check_leakage` cost **28 s/iter** before it was fixed (817df17). Still present in
   `simple_dyson`, `scf_common.hpp`, `thc_sosex.icc`, `embedding/`. Cheap, high-yield, and
   the audit is mechanical.
6. **Write path.** Async checkpoint landed (`COQUI_ASYNC_CHKPT=1`, off by default;
   ~17 s/iter in steady state). Remaining: (a) skip the async path for the pre-loop
   `dump_scf(0)` whose copy is pure waste since it is joined immediately; (b) **decision
   needed:** `G_tskij` is ~4.4 GB of the 8.9 GB checkpoint and `read_scf` never reads it
   back — but `com_diis_residual::upload_g_mu`, `pproc_t::analyt_cont`,
   `local_density_of_state` and the GW+EDMFT driver all read it with no fallback, so
   skipping it must be a **user-facing option** (e.g. keep only the final iteration), not a
   silent default. The path is filesystem-bound at ~0.40-0.44 GB/s measured in-run — **do
   not tune HDF5**; the only levers are write less, write per-rank files, or overlap.
7. **Do not re-attempt:** NCCL for `update_w`'s redistribute (measured 2-4× slower than
   MPI, which already runs at NVLink speed); software pipelining of the exchange (measured
   no effect — it is bandwidth-bound); porting `geigenvalues` to the GPU (`Xgeev` has no
   batched form; 16 host threads beat 4-stream cuSOLVER by 2.4×, and threading already took
   it 44.6 → 2.5 s/iter).

---

## 4. Phase 4 — efficiency, memory footprint, and test coverage

### 4.1 Memory
- Peak footprint is what shelved the **device pool (P2)**: reserving 3 GB inside the SCF
  loop OOMs iteration 2 at kp222/500b (an 18.5 GB tensor alloc fails with 36.5 GB "free").
  Revisit only after items 3.1 and B5 shrink the peak.
- Once G/Σ are device-resident, re-derive the per-rank footprint and re-check the
  large-system feasibility numbers (kp888/1000b needs k-symmetry).

### 4.2 Test coverage gaps that let all of this through
These are the reason two of this session's bugs survived since May. Fix the harness, not
just the code.
- **No ERI-build A/B existed** — every validation loaded a prebuilt shared ERI, which
  bypasses the entire ERI construction path. Added
  `correctness/eri_slate_check/` (builds the ERI old-vs-new in one job and diffs the h5).
  Promote it to the standard set.
- **No configuration exercises `slate_ops` on device data.** At the rank counts anyone runs
  (4, 8) the inner communicator is size 1 and every slate op takes its serial shortcut.
  Add a 6- or 12-rank single-node case to the standard harness, and one with
  `np_P*np_Q > 1` so `dyson_W_in_place`'s `inverse()` is covered.
- Keep the validation discipline that has been working: snapshot the binary
  (`cp bin/coqui bin/coqui.pre_<change>`) **before** rebuilding, run old and new in the
  **same job on the same node**, diff the h5 elementwise, and always force the old path
  explicitly. Cross-job comparisons drift ~1e-14 and are worthless.
- `rsync -a` preserves local mtime and silently skips recompiles. Always `touch` on rusty
  (or `--no-t`) and verify with `strings`.

---

## 5. Suggested order of work

| step | content | why here |
|---|---|---|
| 0 | **Add `--gpu-bind=single:1` to every GPU `run.sbatch`** | one line per script, no rebuild, removes a whole failure mode (§1.3) that is live in production today |
| 1 | §1.3 `make_slate` guard + §1.2b ERI replicate-and-solve | unblocks B3 and large-rank ERI builds, on the GPU, without touching SLATE |
| 2 | Re-test the ERI build at 6/12/16 ranks with binding on | confirms §1.2b end-to-end before anything is built on it |
| 3 | §2 **B4**, timer trap, Zqur logging | hours of work, removes whole classes of wasted debug cycles |
| 4 | §1.2e SLATE-backed array + `redistribute` overload, then **P7** | the durable distributed device solve; copy-in absorbed into an existing redistribute; unblocks P7 (−4 s/iter) |
| 5 | §2 **B6** instrumentation → fix | unblocks H100/sm90, currently a hard blocker there |
| 6 | §3.1 G/Σ device residency, then §3.2, §3.3 | the big performance item and its two dependents |
| 7 | §3.5 `to_host` audit, §3.6 write path | cheap wins, and the write path is the largest remaining item after update_w |
| 8 | §2 **R1**, **B5**, §4.1 pool | reproducibility and footprint; unblocks kp444/kp888 |

## 6. Decisions needed from Miguel
1. **Which mechanism for the distributed device solve?** Three viable, in increasing effort
   and increasing quality:
   - **§1.2d unified memory** — a few type changes, lands in SLATE's canonical mode, but pays
     fault-driven migration, is hard to profile, and would want `cudaMemAdvise` tuning. Fine
     as an interim.
   - **§1.2c mirror** — ~150 lines, explicit D2D copy in and out at the SLATE boundary.
   - **§1.2e dedicated SLATE-backed distributed array — recommended.** ~300-400 lines, but the
     copy-in is *free* because `C_quv`/`Z_quG` are already redistributed at that exact point,
     it makes the bug class unrepresentable at the type level, it unblocks P7's layout
     constraint, and it lets the §1.1 device tile-tagging be retired later.

   §1.2e supersedes §1.2c; they are the same idea with the copy absorbed into an existing
   redistribute. (`Target::HostTask` is rejected outright — §1.2a.)
2. **`G_tskij` in the checkpoint** — add a user-facing option to skip it (final iteration
   only), given the four readers that have no fallback?
3. **R1**: is making THC point selection distribution-independent worth the change, or is
   "always share a saved ERI" an acceptable permanent rule?

---

# A. Handover — state at 2026-08-02 and what to do next

## A.1 The headline result: a GPU-only ERI corruption bug, fixed

`src/methods/ERI/thc_aux.icc:1450`, in `get_ZquG_Cquv_fft` (device path only). The padding of
`iu_for_Xb` indexed the **global** array absolutely instead of this rank's slice, and the
reduction below it is a **sum**, so any rank holding a partial block silently subtracted 1 from
one of rank 0's interpolating-point indices. At nIpts=1275 on 4 ranks this turned
`iu_for_Xb[318]` from 318 into 317, so **row 318 of C_quv duplicated row 317**.

Fix: pad `iu_loc(...)`, not `iu_for_Xb(...)`. GPU-only because the host path goes through
`get_ZquG_Cquv_fft_shared_memory`, which pads a correctly node-indexed slice.

Verified at nbnd=100 / 4 ranks: `max|C - C^T|` 7.17e3 → 4.66e-10 (CPU 3.49e-10); cond(C)
1.38e19 → 4.50e11 (= CPU exactly); `max|C_gpu - C_cpu|` relative 4.8e-16. Rank scan: GPU now
matches CPU to 4.7e-14 / 3.4e-13 / 2.4e-13 at 1 / 2 / 4 ranks.

**Consequences you must act on**
1. **Every multi-rank GPU-built THC/ISDF ERI is invalid** (the bug triggered whenever nIpts did
   not divide evenly across ranks — nearly always).
   **RESOLVED 2026-08-03 — no production ERI was ever affected, nothing needs regenerating.**
   Settled by provenance, since the bug is device-path-only. The key mechanic is that
   `save = <path>` **reuses** an existing file: the builder logs "CoQuí will compute THC
   integrals and save to:", every later run logs "Reading the precomputed THC integrals from
   file:". So the many GPU benches whose toml has `save=` the same path all *read* it, and the
   file mtimes never moved.
   - `correctness/eri_shared.h5` — built by `correctness/cpu_1r/`, i.e. `mpirun -np 1
     build/cpu/bin/coqui`. Doubly immune (host path, and one rank cannot trigger it).
     **The published reference E=0.5146018573397235 stands.**
   - `si_kp222_n500_e125/eri_n500.h5` — built by **CPU** job 6676685 (`eri_build/`, `mpirun`,
     8 nodes / 192 ranks). Every perf benchmark read this file.
   - `si_kp444_n500_e125/eri_n500.h5` — built by **CPU** job 6676686 the same way.
     (`bench_gpu_16n` / `bench_gpu_6n_h200` name it in `save=` but never ran — no logs.)
   - Corrupt files are confined to the disposable `eri_slate_check` diagnostics.

   **Do NOT try to detect this bug from a saved ERI with a self-contained invariant** — two
   attempts both failed. `rel max|C-C^T|` really measures `2|Im C|` (the ISDF overlap is
   *hermitian*, not symmetric); it only seems to work on kp222, whose overlaps are nearly real
   (relImC ~1e-8..1e-11), and it produced a **false positive on kp444** (q=53 has relImC = 0.11
   legitimately ⇒ rel|C-C^T| = 0.213). `rel max|C-C^H|` detects nothing at all: the corruption
   duplicates an interpolating *point*, hence a row **and** its column, so C stays exactly
   hermitian — measured `rel|C-C^H| = 0` for all 50 saved ERIs, corrupt diagnostics included.
   cond(C) and collocation overlaps do not separate `eri_rs_BUGGY_4` from `eri_rs_FIXED_4`
   either. The only thing that ever established corruption was an **elementwise diff against a
   CPU reference** (`C_gpu[318,:] == C_cpu[317,:]`) plus the 3.4e-4 8-rank energy error.
2. **R1 was mostly this bug.** Rank-count energy spread 3.39e-4 → 3.12e-5 (11x). The residual
   3.1e-5 is the *point selection*, which varies with rank count on **CPU too** (Np = 1289 /
   1279 / 1275 / 1290 at 1/2/4/8), so it is algorithmic: the blocked greedy pivoted Cholesky in
   `chol_metric_impl` amplifies roundoff from different reduction orders. Needs reproducible
   reductions or a decomposition-independent tie-break.
3. C is **not** singular (sigma_min 9.78e-6, cond 4.5e11). Any earlier note in this file
   suggesting the ISDF solve is ill-posed was based on the corrupted matrix.

## A.2 Working tree — nothing is committed

Modified: `methods/ERI/thc_aux.icc` (**the fix**), `methods/ERI/thc.{h,icc}` (block-cyclic
branch + `COQUI_THC_SOLVE_DEBUG`/`_DUMP` instrumentation), `numerics/distributed_array/
{detail/slate_aux.hpp, slate_ops.hpp, tests/CMakeLists.txt}`.
New: `numerics/distributed_array/{matrix_array.hpp, slate_ops_matrix_array.hpp,
matrix_array_redistribute.hpp}`, `tests/test_matrix_array{,_ops}.cpp`, and the two docs.

**The ERI fix is independent of everything else and should land on its own** — one line plus a
comment in `thc_aux.icc`.

## A.3 Do this next, in order

1. **Land the `thc_aux.icc` fix** and regenerate every GPU-built ERI. Highest value by far, and
   it is one line. Re-validate `eri_shared.h5` against a CPU build before trusting any
   published number.
2. **Collect the missing CPU 8-rank point** — `cpu_rank_scan.sbatch` in
   `correctness/eri_slate_check/` (the last session lost network before it finished). Completes
   the CPU-vs-GPU table.
3. **Add `--gpu-bind=single:1`** to every GPU `run.sbatch` in `GPU_PORT_run` (§1.3). No rebuild;
   removes a live failure mode.
4. **Re-run the ctest suite at 2 ranks.** `test_slate.cpp:232` has been failing there all along,
   hidden by `CTEST_NPROC=4`. Either fix it via the block-cyclic path or record it.
5. **Finish the block-cyclic port** (task 6) — now worth doing on its merits: with C correct it
   reproduces legacy to **5.9e-13**. Convert the remaining SLATE call sites (`multiply` at
   thc.icc ~1617/1666/1724, the `buffer` solve at ~1650), then retest 12 ranks for B3.
6. Residual R1 (§A.1 item 2), then the performance items in §3.

## A.4 How to work on this efficiently

- **Fast unit-test loop: `src/build/gpu/run_dtest.sbatch`** — ~84 s build, ~4 min cycle. Keep
  `#SBATCH -N 1`. The build **must** gate the run: a failed build leaves the previous binary and
  the suite happily reports "All tests passed" for stale code.
- Instrumentation left in place: `COQUI_THC_SOLVE_DEBUG=1` (prints max|C|, max|C-C^T|, and the
  solve residual for q0; valid only when `pgrid3D[1]==pgrid3D[2]==1`) and
  `COQUI_THC_SOLVE_DUMP=<file.h5>` (dumps C and 64 columns of Z and X). Harnesses and analysis
  scripts are in `~/ceph/CoQui/GPU_PORT_run/correctness/eri_slate_check/`.
- **The method that worked: dump the matrices from a CPU and a GPU build and diff them in
  numpy.** Scalar summaries (norms, residuals, energies) supported three different plausible-
  but-wrong diagnoses over several hours. The elementwise diff showed the corruption was ONE row
  out of 1275 and that `C_gpu[318,:] == C_cpu[317,:]`, which named the mechanism immediately.
  Split CPU vs GPU early; it localizes a device defect in a single run.
