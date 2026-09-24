# CoQuí GPU: plan to a working, performant, efficient device path

> **READ FIRST — status as of 2026-08-03.** The **correctness backlog is closed**; see
> **§B. Correctness closeout (2026-08-03)** at the end of this file for the item-by-item
> disposition, and **§B.4** for a measurement caveat that invalidates any wall-clock number
> taken on rusty on 2026-08-03. Before that, a **one-line bug in the GPU ERI builder** was found
> and fixed; see **§A. Handover**. §0-§6 remain valid as the SLATE/device analysis, with one
> correction: §0's framing of the block-cyclic port as the way to fix reproducibility was
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
2. **R1 was mostly this bug.** Rank-count energy spread 3.39e-4 → 3.12e-5 (11x).
   **RESOLVED 2026-08-03 (job 6744072) — no code change needed; it is an input convention.**
   Running the 1/2/4/8-rank CPU scan with `nIpts` pinned instead of threshold-driven collapses
   the spread from **3.122e-05 to 1.277e-14**:

   | scan | Np at 1/2/4/8 | E1 spread |
   |---|---|---|
   | `thresh = 1e-5` | 1289 / 1279 / 1275 / 1290 | 3.122e-05 |
   | `nIpts = 1275`, `thresh = 0.0` | 1275 / 1275 / 1275 / 1275 | **1.277e-14** |

   So the pivot *sequence* is reproducible across decompositions to roundoff; what varies with
   rank count is only **where the threshold cuts off**, because the residual at the margin
   wobbles at roundoff and crosses `thresh` differently. The earlier note here and in my memory
   — that the blocked greedy pivoted Cholesky reorders near-degenerate pivots — is too
   pessimistic and should not drive work.
   **Recommendation: set `nIpts` explicitly for any run whose number must be compared against
   another rank count** (scaling studies, A/B benchmarks, published values). `thresh` remains
   fine for exploratory work, but its Np — and hence the energy at the 1e-5 level — is
   decomposition dependent.
   Note the two routes are not interchangeable at equal Np: `thresh = 1e-5` at 4 ranks also
   lands on Np = 1275 but gives E1 = 0.5145880920947619, versus 0.5146250400187667 for pinned
   `nIpts = 1275`, a 3.7e-5 difference. Exhausting a threshold part-way through a
   `chol_block_size` block selects a different point set than taking exactly 1275 pivots. The
   3e-5 scale is the ISDF truncation error at this threshold, i.e. the method's own error bar
   rather than a defect.

   (Superseded reasoning, kept for context: the residual was thought to be *point selection*
   varying with rank count on **CPU too** (Np = 1289 / 1279 / 1275 / 1290 at 1/2/4/8), hence
   algorithmic — the blocked greedy pivoted Cholesky in
   `chol_metric_impl` amplifies roundoff from different reduction orders, needing reproducible
   reductions or a decomposition-independent tie-break.)
3. C is **not** singular (sigma_min 9.78e-6, cond 4.5e11). Any earlier note in this file
   suggesting the ISDF solve is ill-posed was based on the corrupted matrix.

## A.2 Committed on branch `gpu` (2026-08-02/03), not pushed

- `6d85ff4` — **the ERI fix**, one line in `thc_aux.icc` plus the mechanism in the message.
- `909927c` — block-cyclic `distributed_matrix_array` (container, `slate_ops_matrix_array`,
  `matrix_array_redistribute`, tests) and every SLATE call site in `thc::intvec_impl` converted,
  plus the `slate_aux` device-address-space fix and the out-of-place `getri` in
  `slate_ops::inverse`. Gated by `COQUI_SLATE_BLOCK_CYCLIC`, **default OFF**.
- `26e1b36` — share one pair of block-cyclic containers across the ZBAR chain (redistributes
  9 → 5, verified bit-identical), and the ERI blast-radius correction in this file.

Still uncommitted: the `force_sync` iteration-0 checkpoint change (`chkpt_utils.{h,cpp}`,
`scf_driver.cpp`, `scf_common.cpp`) and this file's R1 update.

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

---

# B. Correctness closeout (2026-08-03)

Every open correctness item in §1.3, §2 and §A.3 is now closed, **including B6**. Three defects were
found during the work that were not on the list: one by a test written to close another item
(§B.2), and two that together *were* B6 (§B.3.2) — a build flag silently overridden, and CUB launch
failures silently discarded. Validation: full `build/gpufix` build clean (0 errors, job 6745231) —
which is what exercises the new `static_assert`s across every translation unit — plus
`test_math_distributed_nda` at 1, 2 and 4 ranks, host and device, and the B6 reproducer on H100.

## B.1 Disposition of every item

| id | item | disposition |
|---|---|---|
| **B3** | device ISDF/THC builder fails once the grid distributes (u,v) | **CLOSED.** The block-cyclic path fixes it (job 6743779: 12 ranks completes where legacy aborts in `device_regions_build`). It was gated OFF behind a stale note; see §B.2. Now selected automatically. |
| **B4** | µ-search bracketing walk silent and uncapped | **CLOSED.** `scf_common.cpp` `update_mu`: both bracketing walks and the bisection are capped (200 steps each) with messages naming the bracket, each bracketing step is logged, and non-finite `nelec` is rejected at entry *and* at exit. The exit check matters on its own: every loop here exits on a comparison, and every comparison against NaN is false, so a NaN appearing mid-search used to walk out as the answer. |
| **B5** | `compute_eigenspectra` allocates the full Σ(ω) per rank | **CLOSED** by `342f09a`. The only Σ(ω)-sized array is now `Sigmaw_wskij(nwl, …)`, sized by this rank's frequency count (4.4 GB → ~0.3 GB/rank at 8 ranks). `spectra` is ~8.8 MB. Nothing else in the routine is Σ(ω)-sized. |
| **B6** | device ISDF selects zero Cholesky vectors on sm90/H100 | **ROOT-CAUSED AND FIXED — it was two build/error-handling bugs, not a numerical one.** `CMakeLists.txt` silently overrode `-DCMAKE_CUDA_ARCHITECTURES`, so the "sm90" build contained sm_80 code only; on H100 every CoQuí kernel failed to launch, and CUB's discarded return value made that silent. See §B.3.2. |
| **R1** | THC point selection depends on the processor grid | **CLOSED as an input convention** (job 6744072, already in §A.1). Now documented where users meet it: the `thc` constructor options block in `thc.h`, with both measured tables and the "pin `nIpts` for anything compared across rank counts" rule. |
| §1.3 trap 1 | `tileDevice` assumes one visible GPU per rank | **CLOSED.** The guard that existed tested `dev < R.num_devices()`, which *passes* in exactly the broken case (4 visible GPUs, ranks setting dev=0..3) and so never fired. Replaced with `R.num_devices() == 1`, the assumption the code actually makes, with a message naming `--gpu-bind=single:1`. Launch side was already done (49 `srun` lines, 2026-08-02). |
| §1.3 trap 2 | `distributed_array_view` silently reinterprets the address space | **CLOSED.** `static_assert` in the constructor tying the template argument's address space to the argument's. The one live reinterpretation (`hamiltonians.cpp`, `unified_array` over a device slice) is fixed to `memory::array<MEM,…>`; it was dead code in practice since `add_thc_hamiltonian_components` is instantiated HOST-only. `thc_aux.icc:970` had already been fixed. The clean full build is the proof no other site relied on it. |
| §A.3 item 4 | `test_slate.cpp:232` fails at 2 ranks, hidden by `CTEST_NPROC=4` | **CHARACTERISED, deliberately not "fixed".** It is the legacy single-block container hitting the same `group.ld == Mij.stride()` assert as B3, reproduced on demand by `run_dtest.sbatch`'s witness leg. The fix is to route that call site through the block-cyclic container; the assert is the correct, loud behaviour meanwhile. Recorded rather than silenced. |
| — | Zqur stage unlogged for ~390 lines | **CLOSED.** `get_ZquG_Cquv_rspace` now logs the nnr-block loop, per-spin T(k,u,r) and Z(q,u,r) stages, and the C_quv build, with `memory_report` at each allocation, all at verbosity 3. |
| — | `TimerManager::elapsed()`/`number_of_calls()` abort on an unregistered name | **CLOSED.** Read accessors return 0 for unknown names via a new `findPos`; `start`/`stop` still abort, since stopping an unstarted timer is a real error. Removes the class of crash where a one-sided sub-timer killed the *working* branch at the first dump. |
| — | `intvec_impl(int iq, …)` still calls raw SLATE with the legacy layout | **CLOSED as a guarded constraint.** It is reachable only through `__eval_ls__(HOST_MEMORY)` (LS-THC), never on device, so it cannot hit the device batched path. A `static_assert(MEM == HOST_MEMORY)` plus a comment now makes that a compile-time contract instead of an accident, and points at the port to do if it is ever wanted on device. |
| **NEW** | `slate_ops::lu_solve` serial shortcut ignored `hermitian` | **FOUND AND FIXED.** See §B.2. |

## B.2 B3: the orientation question was settled, and it had been blocking the fix

`matrix_array.hpp` kept `COQUI_SLATE_BLOCK_CYCLIC` **default OFF** citing a 2026-07-31 measurement
(job 6719571) where the two paths disagreed by ~6e1 on the ERI and ~8e-5 on the energy. **That
diagnosis was wrong, and it was the only thing keeping the B3 fix switched off.**

The two paths solve the same system whenever `A` is hermitian. The legacy path stores `A` in C order,
so `hermitian=true` conjugates in place and hands slate the transposed view — slate sees
`conj(A)^T = A^H`. The block-cyclic container is column-major and hands slate `A`. For
`C_quv = Z Z^H` those are the same matrix. The 07-31 measurement predates the `iu_for_Xb` fix
(`6d85ff4`), and the corrupted `C` it ran on had a duplicated interpolating point — hence a
duplicated row **and** column, i.e. exactly rank deficient, cond 1.4e19. On a singular matrix the two
factorization orders pick different solutions out of the null space; that is where 6e1 came from.
With `C` correct (cond 4.5e11) job 6743779 measures the 4-rank energies agreeing to **5.8e-13**.

**Selection is now automatic**, and the rule reproduces the measured boundary exactly:
block-cyclic when the matrix grid is non-trivial (`pgrid3D[1]*pgrid3D[2] > 1`), legacy otherwise.
At 4 ranks the grid is (4,1,1) → legacy, which takes its serial local `getrf`/`getri` shortcut: no
slate, no redistribute, cheaper, and bit-compatible with every validated number to date. At 12 ranks
it is (4,3,1) → block-cyclic, the configuration where legacy aborts. `COQUI_SLATE_BLOCK_CYCLIC=1/0`
still forces either way for A/B work. Validated by `autodefault_validate.sbatch`, which pins all
three branches (auto@4 == legacy reference, auto@12 == block-cyclic reference, forced-legacy@12
still aborts).

**The new defect, found by the test written to pin the convention.** The convention is now pinned by
`matrix_array_hermitian_solve_orientation` on a deliberately **non-hermitian** `A` — the boundary the
hermitian agreement tests cannot see, since `A^H == A` hides any orientation change. It failed
immediately, at 1 rank only: `slate_ops::lu_solve`'s serial shortcut (`comm.size()==1`) ignored the
`hermitian` template parameter entirely, solving `A X = B` while the distributed branch solved
`A^H X = B` (measured `||A X - B||` = 7e-16 vs `||A^H X - B||` = 0.097 at 1 rank; the reverse at 2 and
4). `least_squares_solve`'s serial shortcut already handled it, which is what marks this an oversight
rather than a convention. Fixed by mirroring that. **No production result is affected** — every caller
passes a hermitian `A`, where the two coincide — but the flag had been meaning different things at
different rank counts, which is exactly the kind of thing that makes a future bug unfindable.

Method note worth keeping: the useful test was not another agreement check, it was a test on the
input class where the two paths are *supposed* to differ. That is what turned a documentation task
into finding a real bug.

**Numerical footprint of the `lu_solve` fix, measured deliberately** (job 6745450), because it does
touch a production path: at 4 ranks the ERI grid is (4,1,1), so the matrix communicator has size 1
and the legacy ISDF solve goes through exactly the shortcut that changed. Both legs forced to legacy,
pre-fix vs post-fix binary, same node:

| dataset | worst \|diff\| | relative |
|---|---|---|
| `collocation_matrix`, `kpts`, `qpts` | 0.0 | bit-identical |
| `coulomb_matrix` | 2.79e-09 | 5.5e-08 |
| `dual_interpolating_vectors_G0` | 9.13e-10 | 9.1e-10 |
| `interpolating_vectors_G0` | 2.56e-07 | 1.7e-06 |
| **total energy** | **1.45e-13** | — |

Only the ISDF vectors move, at 1.7e-06 relative — two orders inside the eps·cond(C) ≈ 1e-4 bound —
and the energy moves 1.45e-13, eight orders below the ISDF truncation error (3e-5). Accepted.
Note this also explains the one line the `autodefault_validate` verdict flagged: its 4-rank leg was
compared against an ERI built by a *pre-fix* binary, and the difference it reported (2.5585e-07)
matches the table above to the digit. The auto-selection itself is exact — the 12-rank leg is
**bit-identical** to the block-cyclic reference.

## B.3 B6 — reproduced, narrowed, root-caused and fixed

**It still reproduces against a current sm90 build** (job 6745246, H100 PCIe, compute_cap 9.0),
at **1 rank and at 4 ranks**, with the abort exactly as reported in `b257c45`. Reproducing at one
rank is the useful part: there is no MPI, so `find_distributed_maximum` and every reduction-order
argument are **excluded**, and the whole reproducer is a single-GPU, nbnd=100 job — a fast, cheap
debug cycle instead of a multi-node one. `--gpu-bind=single:1` verified (`CUDA_VISIBLE_DEVICES=0`
for every rank), so §1.3's device-binding trap is also excluded.


Device ISDF point selection aborts on sm90/H100 with "Current number of cholesky vectors = 0"
(`thc.icc:1073`), i.e. `max|Diag|` is already ≤ 1e-14 at iteration 0, while identical source and
input on sm80/A100 gives Np=1275. It is not SLATE: `chol_metric_impl` builds `Diag` with cuTENSOR
contractions and finds the pivot with `utils::max_element_multi`, no slate involved.

The abort condition localises it usefully: with `nchol = 0`, the residual diagonal is already empty
on entry, so the fault is upstream of the iteration in one of four places —
`Psia` itself, the `contract(conj(Psia),"skapr",Psia,"skapr",Lr,"skpr")` batched dot
(`thc.icc:970`), the `Diag(r) += conj(Lr)*Lr` accumulation (`thc.icc:975-977`, an unusual
no-reduction cuTENSOR pattern and the prime suspect), or `max_element_multi`/
`find_distributed_maximum`. `b6_sm90_repro.sbatch` (in `correctness/eri_slate_check/`) establishes
whether it still reproduces against a current sm90 build — the ERI path has changed substantially
since the 07-27 binary that produced the original report. **Instrument before fixing:** print
`sum|Psia|`, `sum|Lr|` and a host-computed `max|Diag|` on A100 and H100 and diff; each of the four
candidates gives a different signature, and one run separates them. Workaround remains: build the
ERI on A100 or CPU with `save=` and load it on H100.

**The instrumentation is now in the tree**, gated by `COQUI_CHOL_DIAG_DEBUG=1` (zero cost otherwise),
printing `max|Psia|`, `max|Lr|`, a **host-computed** `max|Diag|` and what `max_element_multi`
returned — the host reduction is the point, since it makes `max_element_multi` falsifiable
independently — plus stage prints along the orbital path. `b6_instrument_h100.sbatch` runs sm80 and
sm90 legs and prints them side by side.

### B.3.1 RESULT: localised to one call, `nda::copy_select` on sm90

Ran it (job 6745552, H100 PCIe, 1 rank). Every suspect above is **exonerated** — the fault is four
stages earlier:

```
custom_grid=1  C_skai=null  single_psi=1  mesh=(39,39,39)  wfc_grid=1
max|distPsia after read_distributed_orbital_set|   = 9.5013249059e-01   <-- healthy
max|psir after copy_select (G-space, pre-FFT)|     = 0.0000000000e+00   <-- BROKEN HERE
max|psir after fft::backward (r-space)|            = 0.0000000000e+00
max|distPsia after Znorm scale|                    = 0.0000000000e+00
max|Psia| = 0  max|Lr| = 0  max|Diag| = 0  lmax_res_val[0] = 0
```

The orbitals are **read correctly** (0.950, identical to A100). The next operation,

```cpp
nda::copy_select(true, 1, wfc_to_rho, ComplexType(1.0), distPsia.local(), ComplexType(0.0), psir);
```

(`thc.icc`, the `custom_grid` branch — scattering the wavefunction G-vectors onto the density FFT
grid) **produces all zeros on sm90 and the correct values on sm80.** Everything downstream is a
faithful transform of zeros, which is why the abort surfaces 250 lines later with `nchol=0`.

### B.3.2 ROOT CAUSE — two bugs, both fixed. It was never cuTENSOR and never sm90 codegen.

**Correcting two of my own intermediate conclusions, because both were wrong and instructively so:**
I wrote "it is not a failed launch" (because `arch::synchronize_if_set()` runs
`cuda_check(cudaGetLastError())` and no error appeared) and "not codegen, since the sm90 binary is
correct on A100". Both inferences were unsound. `cuobjdump` settles it:

```
build/gpu90/bin/coqui :  10 x "arch = sm_80"   ptx: (none)
build/gpufix/bin/coqui:  10 x "arch = sm_80"   ptx: (none)
```

**The sm90 build was never compiled for sm90.** `CMakeLists.txt` had

```cmake
if(DEFINED CUDA_ARCH)
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
else()
  set(CMAKE_CUDA_ARCHITECTURES 80)   # silently overwrites the user's flag
endif()
```

so the project honoured only `-DCUDA_ARCH=`, and `build/gpu90/build.bash`'s
`-DCMAKE_CUDA_ARCHITECTURES=90` — the standard variable anyone reaches for first, and the one the
CMakeCache dutifully recorded as `UNINITIALIZED=90` — was **overwritten with 80**. The binary
therefore held sm_80 cubins and no PTX to JIT forward from. That is why the "sm90" binary ran fine on
an A100: it *is* an sm80 binary.

**On H100 every CoQuí kernel then failed to launch** with `cudaErrorNoKernelImageForDevice`. Vendor
libraries are fat binaries covering sm_90, so cuTENSOR / cuBLAS / cuFFT kept working perfectly — on
zeros. Only CoQuí's own kernels died, `copy_select` among them.

**And the failure was silent, which is the second and more serious bug.** Every
`cub::DeviceFor::Bulk` call in `device_kernels/cuda/` **discarded the `cudaError_t` it returns**, and
CUB *consumes* the sticky error to produce that return value — so by the time
`arch::synchronize_if_set()` called `cudaGetLastError()` there was nothing left to find and the check
passed. `psir` had been explicitly zeroed the line before, so it simply stayed zero. Then:
FFT of zeros → `Diag` zero → `max_element_multi` (which for device input just does `nda::to_host` and
runs the **host** algorithm, so no kernel and no error) faithfully reported 0 → abort 250 lines later
blaming the ISDF threshold, with cuTENSOR the natural suspect. Every observation reconciles.

**Fixes applied:**
1. `CMakeLists.txt` now honours `CMAKE_CUDA_ARCHITECTURES`, keeps `CUDA_ARCH` as an alias, warns if
   both are set and disagree, and prints the architecture it chose.
2. `kernels::device::check_launch(status, what)` added in `cuda_aux.hpp` and applied to **all 28**
   live CUB launch sites across `copy_select.cu`, `copy_cast.cu`, `complex_tools.cu`,
   `kpoint_tools.cu`, `symmetry_tools.cu`, `potentials.cu`. On `cudaErrorNoKernelImageForDevice` it
   additionally reports the device's actual `sm_XX` and tells you to rebuild and how to verify with
   `cuobjdump`. Verified there are no CoQuí kernel launches outside that directory, and no raw
   `<<<>>>` launches anywhere; `argmax_min.cu`'s thrust calls use the `thrust::device` policy, which
   *throws* on error rather than returning a code, so they were never silent.

**Lessons worth keeping.** (a) A discarded status from a library that consumes the underlying error is
worse than no error handling at all — it makes the downstream `cudaGetLastError()` actively
misleading, which is what defeated me for a round. (b) `-DCMAKE_CUDA_ARCHITECTURES` being silently
overridden is the kind of build trap that produces wrong *numbers*, not build errors; the new
`message(STATUS "CUDA architectures: ...")` makes it visible. (c) Check what a binary actually
contains (`cuobjdump | grep 'arch ='`) before theorising about a hardware-specific numerical bug.
(d) `b6_instrument.sbatch` without `--constraint=h100-80gb` gets handed an A100 and silently passes —
that cost a round trip, and the accidental A100 run is what exposed the whole thing.

## B.4 MEASUREMENT CAVEAT: do not trust any rusty wall-clock number from 2026-08-03

New GPU timings were collected and are **not usable**. The 8-rank kp222/500b SCF loop measured
1127 s where 07-30 measured 278 s for two iterations, and the sub-timers show the cause precisely:

| per iteration, 8×A100 | 07-30 (job 6705038) | 08-03 (job 6745221) | ratio |
|---|---|---|---|
| `gemm Z*Pi -> A` | 3.97 | 3.88 | 1.0 |
| `inverse(A)` [SLATE LU] | 8.26 | 7.87 | 0.95 |
| FT gemm (PHsym) | 0.16 | 0.16 | 1.0 |
| **redistribute** | **17.5** | **245.8** | **14×** |

Compute is unchanged to a few percent; only the inter-rank exchange collapsed. Energies were
bit-identical throughout, so this is purely a rate effect.

### THE FIX FOR IT: `export UCX_TLS=^cuda_ipc`

**Measured (job 6746573), same allocation, one SCF iteration.** Excluding the broken transport
recovers essentially all the performance:

| leg | tau_to_w | w_to_tau | redistribute | SCF total | intra-node BW |
|---|---|---|---|---|---|
| default (broken `cuda_ipc`) | 127.1 | 138.1 | 251.8 | 404.9 | 0.22 GB/s |
| `COQUI_REDIST_DEVICE=0` (host staged) | 40.2 | 43.4 | 76.5 | 223.6 | — |
| **`UCX_TLS=^cuda_ipc`** | **11.9** | **11.5** | **19.9** | **164.5** | **14.59 GB/s** |
| 07-30 reference, healthy `cuda_ipc` | 10.2 | 10.9 | 17.5 | — | 88.75 GB/s |

Within ~15% of the healthy reference even though intra-node bandwidth is still 6x down — at
14.6 GB/s the exchange has stopped being the bottleneck. Note that host staging (40/43) also beats
the default (127/138) by 3x, so *any* route around `cuda_ipc` is better than leaving it enabled.
**Set `UCX_TLS=^cuda_ipc` in the run scripts until the site fixes CUDA IPC**, and re-measure before
making it permanent.

**It is site-wide, not a few bad nodes.** `bench_comm` on workergpu042-043 measured **intra-node
GPU-GPU 0.20 GB/s** against a recorded 07-30 baseline of **88.75 GB/s** — a 444x collapse — while
*inter-node* IB was healthy (8.95 vs 7.89 GB/s). Re-running the 8-rank benchmark with
`--exclude=workergpu042,043,044` landed on **workergpu057,059** and was *equally* slow (tau_to_w
125.4 s, redistribute 248.3 s for one iteration). So excluding nodes does not help. The signature —
intra-node GPU peer-to-peer dead, inter-node fine, every a100-80gb node affected, the same binary
that was fast on 07-30 — points at **CUDA IPC / peer-to-peer no longer working after a site-side
driver or UCX change**. That is a ticket for the Flatiron admins, not a CoQuí fix. `transport_matrix.sbatch`
measures whether routing around it helps (host-staged redistribute via `COQUI_REDIST_DEVICE=0`, and
`UCX_TLS=^cuda_ipc`); on 07-30 host staging cost 52/57 s per iteration, which would be **better than
the ~125/136 device-direct costs today**, so it is plausibly the right temporary default.

**Established as environmental, not a code regression, two independent ways.** First, no commit
after the 07-30 measurement touches the redistribute path at all — `git log fcc81ca..HEAD --
nda_utils.hpp nda_matrix.hpp` returns only `9f41dd6`, which is *in* the 07-30 binary. Second and
decisively, `binary_bisect.sbatch` (job 6745271) ran five snapshots back to back **inside one
allocation**, holding nodes, env, ERI and input fixed:

| binary | date | tau_to_w (1 iter) | w_to_tau |
|---|---|---|---|
| `coqui.syncwrite` | 07-30 09:58 | 125.3 | 138.6 |
| `coqui.pre_blockcyclic` | 07-30 16:50 | 126.4 | 139.4 |
| `coqui.pre_iufix` | 08-01 16:20 | 126.7 | 137.6 |
| `coqui.pre_fuse` | 08-03 10:38 | 126.2 | 137.0 |
| `coqui.perf0803` | 08-03 11:28 | *(see log)* | |

Flat. `coqui.syncwrite` is the **very binary** that measured 10.2 s/iter on 07-30 and it is now 12×
slower, so the machine changed, not CoQuí.

**Protocol lessons, both of which cost real time here:**
- **Always run the old binary alongside the new one, in the same allocation.** Cross-day comparison
  on this cluster is worthless at the factor-of-ten level. Snapshot binaries
  (`cp bin/coqui bin/coqui.<tag>`) exist for exactly this; use them for *performance*, not only for
  bit-level correctness A/B.
- **The launcher environment is part of the measurement.** `bench_gpu_4n/run.sbatch` and
  `gw_gpu_8r/run.sbatch` still set `OMPI_MCA_pml=ucx` and a restricted
  `UCX_TLS=cuda_copy,cuda_ipc,sm,self,rc,ud,dc`, while every post-P1 measurement
  (`bench_gpu_2n_opt/opt2/redist`) sets only `UCX_MEMTYPE_CACHE=n`. On the same two nodes minutes
  apart these gave SCF 703.6 s vs 1127.3 s — the settings are worth a factor of 1.6 here and the
  scripts disagree with each other. Settle on one convention (`UCX_MEMTYPE_CACHE=n`, no `UCX_TLS`)
  and record it in every script. Note this also means the **original 16-rank baseline of 631.9 s was
  measured with the restricted-`UCX_TLS` environment** and the 8-rank 1008.7 s was not, so those two
  baselines are not on equal footing either.
- `homework/bench_comm.cu` is the right first instrument when the exchange looks wrong: it reports
  intra- and inter-node GPU MPI bandwidth in about a minute, against a recorded 07-30 baseline of
  **88.75 GB/s intra-node, 7.89 GB/s inter-node**.

**To actually get the numbers:** rerun `bench_gpu_2n_0803`, `bench_gpu_4n_0803` and
`bench_gpu_2n_0803_async` (niter=3, shared ERI, `UCX_MEMTYPE_CACHE=n`) once `bench_comm` reports
**intra-node** bandwidth back at ~88 GB/s, and include a `coqui.syncwrite` leg in the same
allocation so the result is anchored to the 07-30 measurement rather than to a date.

### B.4.1 Two jobs were still queued when this session ended — collect them first

Network access to rusty dropped (DNS) before these finished. They were left **queued, not
cancelled**, so their logs should exist:

| job | what it answers | where the log lands |
|---|---|---|
| **6745423** | `bench_comm` with `--exclude=workergpu042,043,044`. Is the 0.20 GB/s intra-node collapse specific to those three nodes, or cluster-wide? | `GPU_PORT_run/homework/benchcomm_6745423.log` |
| **6745309** | 8-rank kp222/500b, niter=3, `coqui.perf0803`, same exclusion. The clean headline number, if the nodes it lands on are healthy. | `GPU_PORT_run/si_kp222_n500_e125/bench_gpu_2n_0803_excl/run_6745309.log` |

**Read 6745423 first.** If intra-node is back at ~88 GB/s there, the degradation is node-local, the
partition should simply be run with those three excluded, and 6745309 is a usable measurement — check
its `redistribute` sub-timer is back near 17 s/iter before trusting the total. If intra-node is still
~0.2 GB/s on other nodes, the problem is broader, 6745309 is *also* invalid, and this needs a ticket
to the Flatiron admins rather than another benchmark.

Extraction helper written for this: `si_kp222_n500_e125/extract_timings.sh <log>...` prints the
Dyson-SCF totals, the `update_w` sub-timers, Np, niter, redistribute mode and energies in a fixed
format. Baselines to compare against are in §B.4 above and the tables in §3.

---

# C. VALIDATED TIMINGS (2026-08-03, with `UCX_TLS=^cuda_ipc`)

Obtained once the broken transport was routed around (§B.4). **Every leg below is validated by an
anchor binary in the same allocation**: `coqui.syncwrite` (07-30) reproduced its 07-30 transform cost
to within 3% (tau_to_w 31.47 s over 3 iterations = 10.5 s/iter vs 10.2 s/iter on 07-30), and the
job's own `bench_comm` step reported intra-node 14.67 GB/s. So the allocation was behaving and the
numbers mean something. Job 6746623, workergpu[057,059].

## C.1 Si 2x2x2 / 500 bands, 8xA100 (2 nodes), shared ERI, niter=3

Metric is the `Dyson-SCF timers -> Total` line, i.e. the SCF loop excluding ERI construction.

| configuration | SCF loop, 3 iterations | vs original GPU |
|---|---|---|
| **CPU 1 node / 96 ranks** (original) | 1423.2 s | — |
| **CPU 2 nodes / 192 ranks** (original) | 1017.8 s | — |
| **GPU 8xA100, original code** (07-25, `bench_gpu_2n`) | 1008.7 s | 1.00x |
| GPU 8xA100, 07-30 code (anchor, measured today) | 416.96 s | 2.42x |
| **GPU 8xA100, current code** | **407.81 s** | **2.47x** |

**Current GPU vs CPU: 2.50x faster than 192 CPU ranks (2 full nodes), 3.49x faster than 96.**
Energies bit-identical across the anchor and current legs
(0.4830503807840806 / 0.5300765029178806 / 0.5341906406075212), and identical to every earlier run
in this series.

Anchor vs current on identical hardware, same allocation:

| | anchor (07-30) | current | |
|---|---|---|---|
| SCF loop, 3 iters | 416.96 | **407.81** | −2.2% |
| tau_to_w, 3 iters | 31.47 | 31.63 | unchanged |
| redistribute, 3 iters | 53.44 | 54.03 | unchanged |

The transform and exchange phases are unchanged, which is exactly right — nothing since 07-30 touched
them (`git log fcc81ca..HEAD -- nda_utils.hpp nda_matrix.hpp` is empty of functional change). The
−2.2% comes from the host-side work committed after 07-30 (eigenspectra threading, the
`Sigma_div_correction` staging buffer, mixing keeping F/Sigma in memory, the THC transform gemms).

## C.2 Where the 2.47x came from

Against the original 8xA100 run, in the order the work landed:

| change | effect on the 8xA100 iteration |
|---|---|
| P1: device-direct redistribute (`9860888`, `c2f8e71`) | tau_to_w+w_to_tau 109 -> 46 s/iter |
| `IAFT::check_leakage` on device (`817df17`) | −28 s/iter (a *diagnostic* was the second-largest cost) |
| eigenspectra threading + omega-local transform (`342f09a`) | 44.6 -> 2.5 s/iter |
| `Sigma_div_correction`, mixing, THC transform gemms | the residual −2.2% measured in C.1 |

## C.3 Still to collect — three jobs left queued, all self-diagnosing

The `gpu` partition was saturated (110 running / 34 pending) when this session ended, so these were
left **queued, not cancelled**. Each carries its own validity check, so they can be read cold.

| job | what it answers | log |
|---|---|---|
| **6746624** (`-p gpu`) and **6746696** (`-p gpupreempt`) | 16xA100, niter=3, **anchored**. Same script submitted to both queues because 4 nodes are scarce — whichever runs first wins, cancel the other. | `anch16_6746624.log` / `anch16p_6746696.log` |
| **6746697** | Async checkpoint A/B: sync vs `COQUI_ASYNC_CHKPT=1`, current binary, same allocation. This is now the **highest-value remaining perf item** (C.4: the write is 20-31% of the loop). | `asyncab_*.log` |

Reading the 16-rank result: compare to the original `bench_gpu_4n` **631.9 s**, but that baseline was
measured with the restricted `UCX_TLS` that pins the now-broken `cuda_ipc` (§B.4), so it is *not* on
equal footing with the 8-rank 1008.7 s. The anchor leg is what makes the comparison sound — if
`anchor_0730` reproduces ~31 s tau_to_w over 3 iterations, trust the `current` number.

Every script sets `UCX_TLS=^cuda_ipc` and prints `bench_comm` before measuring; if intra-node
bandwidth is not O(10 GB/s) or the anchor is several times its 07-30 cost, discard and see §B.4.

**To read the results, run one command:**

```
~/ceph/CoQui/GPU_PORT_run/si_kp222_n500_e125/collect_pending_results.sh
```

It finds whichever logs exist, judges each run's fabric health for you (flagging a discard if
intra-node bandwidth came back under 1 GB/s), prints the anchor-vs-current comparison with the pass
criterion spelled out, computes the async saving, and checks the async legs' energies match. It also
reminds you to cancel the duplicate 16-rank submission. Per-phase detail for any single log:
`./extract_timings.sh <log>`.

Beyond these: rerun the **16-rank and H100 ERI builds on device** now that B6 is fixed and the
block-cyclic path is auto-selected (§B.2, §B.3.2) — that combination has never been exercised
end-to-end, and it is the configuration that used to abort.

## C.4 Si 2x2x2 / 500 bands, 8xH100 (1 node), shared ERI, niter=3

**The first trustworthy H100 measurement.** Until today `build/gpu90` contained sm_80 code despite
being configured for 90 (§B.3.2), so on H100 every CoQuí kernel silently failed to launch. Binary
verified here as `10 arch = sm_90`. Job 6746651, workergpu160, `UCX_TLS=^cuda_ipc`.

| | original (job 6678589) | current | |
|---|---|---|---|
| SCF loop, 3 iterations | 941.36 s | **497.16 s** | **1.89x** |

Energies match the reference to ~2e-15 relative
(0.48305038078409046 / 0.5300765029178796 / 0.534190640607511).

**Side by side with 8xA100, same input, same 3 iterations — and the difference is not the GPU:**

| phase | 8xA100 (SXM4) | 8xH100 (PCIe) |
|---|---|---|
| SCF loop total | 407.81 | 497.16 |
| Dyson | 50.08 | 53.99 |
| MBPT solvers | 240.65 | 246.80 |
| Iterative alg | 18.90 | 18.70 |
| **checkpoint write** | **81.67** | **155.36** |
| Energies | 5.15 | 5.28 |
| unaccounted | 11.36 | 17.03 |

Compute is a wash (MBPT 240.7 vs 246.8, Dyson 50.1 vs 54.0). **Essentially the whole 89 s gap is the
checkpoint write**, +73.7 s, i.e. filesystem variance rather than hardware. Note the write is now
**20% of the loop on A100 and 31% on H100** — at this problem size the SCF loop is no longer
GPU-limited, which makes the async checkpoint (`COQUI_ASYNC_CHKPT=1`, §3.6) the obvious next win and
promotes the write path above further device work.

**Inside `update_w` the two machines differ in an instructive way:**

| | 8xA100 (SXM4, NVLink) | 8xH100 (PCIe) |
|---|---|---|
| `eval_Pi_qdep` | 26.50 | 37.12 |
| tau_to_w + w_to_tau | 65.42 | 103.98 |
| `dyson_W_in_place` (SLATE LU) | 55.17 | **35.75** |

The H100 wins the compute — the W-Dyson LU is **1.54x faster** — and loses the exchange, because
these are **H100 PCIe** parts with no NVLink between GPUs, so all eight ranks talk over PCIe where the
A100-SXM4 node has NVLink. Measured intra-node bandwidth 8.59 GB/s on the H100 node vs 88.75 GB/s on
a healthy A100-SXM4 node; roughly PCIe-gen5 speed, i.e. *normal for this hardware* and not the
`cuda_ipc` fault of §B.4. Net effect for this phase, the exchange penalty outweighs the compute gain.
**Do not treat "H100" as strictly faster than "A100" here — ask whether the part is SXM or PCIe
first**, and prefer the SXM4 A100 nodes for exchange-heavy runs at this size.

---

# D. FINAL RESULTS (2026-08-04) — and a correction to how §C.1 framed them

The queued runs landed. `si_kp222_n500_e125/collect_pending_results.sh` reproduces everything below.

## D.1 The SCF loop total is dominated by checkpoint-write variance

**This corrects §C.1, which quoted a single sample as though it were the number.** Across five runs of
the *same code*, 3 iterations each:

| run | SCF | Write | **SCF − Write** |
|---|---|---|---|
| 8xA100 (anchored, job 6746623) | 407.8 | 81.7 | **326.1** |
| 8xA100 (async A/B sync leg, 6746697) | 362.0 | 30.2 | **331.8** |
| 8xA100 (async A/B async leg) | 355.7 | 19.4 | **336.3** |
| 8xH100 (6746651) | 497.2 | 155.4 | **341.8** |
| 16xA100 (6746624/96) | 312.0 | 73.6 | 238.3 |

At 8 GPUs, **SCF − Write is 326–342 s (±2.4%) across every run and across both machines** — A100-SXM4
and H100-PCIe agree to 5% — while the totals span **356–497 s**. So essentially all the spread in the
headline figure is the checkpoint write, which itself ranged 19–155 s depending on filesystem load.

**Quote the 8xA100 result as a range: 356–408 s for 3 iterations, i.e. 2.47–2.83x over the original
1008.7 s.** A single number invites the reader to over-trust it, and the run-to-run spread here is
larger than most of the code improvements measured this session.

## D.2 16xA100 — two independent runs

Both submissions (`-p gpu` and `-p gpupreempt`) ran, on the same four nodes, fabric OK (14.6 / 13.8
GB/s intra-node):

| job | anchor_0730 | current |
|---|---|---|
| 6746624 | 316.93 | 324.52 |
| 6746696 | 312.85 | 311.97 |

**16xA100 current ≈ 312–325 s / 3 iterations, versus the original 631.9 s: ~1.95–2.03x.** The anchor
and current legs are indistinguishable here — run-to-run spread is ~4%, larger than the ~2% the
post-07-30 commits are worth — so do not read a difference between them as signal at this rank count.
Scaling 8→16 GPUs on the compute part is 326 → 238 s, i.e. **1.37x for 2x the GPUs**; the exchange
(`redistribute` 45 s / 3 iters at 16 ranks vs 54 at 8) does not shrink, which is the expected
communication-bound behaviour and where further scaling work would have to go.

*Caveat on the setup:* both 16-rank jobs shared the `anchored16/` output directory, so its
`log_*.txt` belong to whichever finished last. The per-job summaries above are independent (each job
printed its own). Give parallel submissions distinct output directories.

## D.3 Async checkpoint — works, and its value scales with how bad the filesystem is

Same allocation, sync vs `COQUI_ASYNC_CHKPT=1`, current binary:

| | SCF | Write |
|---|---|---|
| sync | 362.0 | 30.2 |
| async | 355.7 | 19.4 |

**It removes 36% of the write time**, and energies are **bit-identical** between the legs
(0.4830503807840806 / 0.5300765029178806 / 0.5341906406075212), which is the thing that had to be
confirmed before recommending it.

The headline "1.7% of the loop" understates it, because this run drew a *fast* write (30 s). Scaled by
the same 36% against the writes actually observed today, async is worth ~29 s when the write is 82 s
and ~56 s when it is 155 s — i.e. **up to ~11% of the loop.** Recommend turning it on by default
after one more A/B on a run that draws a slow write; the correctness question is settled.

## D.4 Where the remaining time is

At 8xA100 with a fast write, the ~332 s of non-write loop time over 3 iterations breaks down roughly
as `update_w` ~121 (of which the W-Dyson LU ~55, transforms ~65), Sigma ~120, Dyson ~50, mixing ~19,
energies ~5. The next targets in order of measured size: the **write path** (§3.6, and now D.3), the
**W-Dyson LU** (device batched LU, §3.3/P7), and **G/Sigma device residency** (§3.1). Note §3.1's
premise should be re-checked against D.1 — with compute now machine-insensitive to 5%, the case for
further device porting is weaker than the case for writing less.
