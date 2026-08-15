# A: point selection / zeta solve

## 1. ISDF interpolation point selection (pivoted Cholesky on the Gram/metric matrix)

### 1d. Call chain (driver -> kernel)
- Public entry A: `thc::interpolating_points<MEM>(int iq, int max, nda::range a_range, nda::range b_range)`
  -> `/Users/mmorales/Projects/ISDF_metric/coqui/src/methods/ERI/thc.cpp:163`. Decl `/Users/mmorales/Projects/ISDF_metric/coqui/src/methods/ERI/thc.h:125-132`.
  Returns `tuple<array<MEM,long,1> ipts, darray4 Xskau, optional<darray4> Xskbu>` = (point indices, phi_a(r_mu), phi_b(r_mu)).
- Public entry B (rotated orbitals): `thc::interpolating_points<MEM>(C_skai, iq, max)` -> `thc.cpp:198` (thc.h:134).
- Dispatch at `thc.cpp:186-194`: if (a_range==b_range && q==Gamma && nkpts!=nkpts_ibz) -> `chol_metric_impl_ibz<MEM,true,true>(iq,max,a,b,default_cholesky_block_size)` (`thc.icc:43`);
  else -> `chol_metric_impl<MEM,true,true>(iq,max,a,b,default_cholesky_block_size, C_skai)` (`thc.icc:763`).
- Template params: `<MEMORY_SPACE MEM, bool Ipts_only, bool return_Ruv>`; decl `thc.h:582-584` (main) and `thc.h:617-618` (ibz). Both are private.
- Callers of the driver: `thc_reader_t.hpp:432`, `:1708`, `:1870` (`_thc_builder_opt.value().interpolating_points<MEM>(0,_Np,x_range,y_range)`), and `check_eri_symmetry.cpp:249`.

### 1a. Gram diagonal d_g and columns C_{:,g}
- The "grid orbital" arrays are `distPsia`/`distPsib` (`thc.icc:809`, `:869-892`), read via `mf::read_distributed_orbital_set`,
  reshaped as `Psia(is,k,a,pol,r)` / `Psib(is,k,b,pol,r)` at `thc.icc:915-916`. Grid axis `r` is the DISTRIBUTED axis (`r_range = distPsia.local_range(1)`, `thc.icc:910`).
- Normalization `Znorm = 1/sqrt(sqrt(na*nb*ns*nk*npol))` applied to both orbital sets (`thc.icc:796`, `:859`, `:896`); removed later at `thc.icc:1385`,`:1424`.
- Non-Gamma q gets an r-space phase factor multiplied into Psib (`thc.icc:921-947`, `utils::rspace_phase_factor`).
- DIAGONAL d_g = Diag(r) = sum_{s,k,pol} conj(L_rr) * R_rr, where L_rr = <phi_a(:,r),phi_a(:,r)> — i.e. |sum_a |phi_a(r)|^2|^2 in the single_psi case.
  HOST path `thc.icc:960-977` (nda::blas::dotc per grid point); GPU path `thc.icc:978-1006` (nda::tensor::contract building Lr/Rr then contracting to Diag).
  This is exactly the Hadamard structure: Diag = (A o conj(A)) summed over orbital pairs.
- COLUMN C_{:,g}: after the pivot r_mu is chosen, the owning rank gathers `Paki(:,n) = conj(distPsia.local()(:,lr))` (and Pbki for b) at `thc.icc:1113-1121`
  — the orbital values at the pivot point, shape (ns*nk*na*npol, block_size).
  The Gram column is then built by GEMM against the full local grid block: `Lr(is,k,p,u,r) = Paki5d^T * Psia` (`thc.icc:1145-1149` HOST; `thc.icc:1167-1172` GPU tensor::contract),
  followed by the Hadamard/dot product `Tab(u,r) = sum_{skp} conj(Lr) * Rr` (`thc.icc:1153-1173`). Tab is the (block_size x nr_local) Gram column block.

### 1b. Pivot loop
- Main loop `while(true)` at `thc.icc:1081-1337`, wrapped by timer `IpIter` (`thc.icc:1070`,`:1338`).
- argmax: `utils::max_element_multi(Diag, lmax_res_val, lmax_res_indx)` (local, `thc.icc:1008-1009` initial, `:1320-1321` in-loop)
  then `utils::find_distributed_maximum(mpi->comm, lmax_res_val, gmax_res)` (global, `thc.icc:1015`, `:1323`) inside timer "COMM"/"ip_COMM".
- gather column: owner-only fill of `comm_buff` (holding Paki|Pbki|Rc) then `mpi->comm.all_reduce_in_place_n(ur,...)` and
  `all_reduce_in_place_n(comm_buff, block_size*(ns*nk*(na+nb)*npol + nchol), plus)` — `thc.icc:1128-1130`. One Allreduce per pivot block.
- downdate: `Tab -= dagger(Rc[0:nchol,:]) * R[0:nchol,:]` GEMM at `thc.icc:1177-1180`; then block matrix `Abb(n,:) = Tab(:,ip)` (`thc.icc:1183-1189`),
  `mpi->comm.reduce_in_place_n(Abb, block_size^2, plus)` to root (`thc.icc:1193`), rank-0 serial `utils::chol<false,W_type>(Abb,piv,thresh)` + `nda::inverse_in_place`
  (`thc.icc:1196-1237`, timer "ip_SERIAL"), then two broadcasts of `piv` and `Abb` (`thc.icc:1240-1241`).
- new Cholesky rows: `Rn = Abb[0:newv,0:newv] * Tab[0:newv,:]` GEMM (`thc.icc:1295-1296`); pivot global index recorded `rn(nchol+n) = ur(piv(n))` (`thc.icc:1299-1300`).
- diagonal update: `Diag(r) -= conj(Rn(v,r))*Rn(v,r)` (`thc.icc:1311-1317`), timer "ip_update_res".
- MPI collectives per pivot block: 2 Allreduce (ur, comm_buff), 1 Reduce (Abb), 2 Broadcast (piv, Abb), plus the max-reduction inside `find_distributed_maximum`.
- Collation matrices `Pskau`/`Pskbu` accumulated per accepted pivot (`thc.icc:1252-1269`), assembled into distributed `Xskau`/`Xskbu` at `thc.icc:1372-1450` with a k-point phase factor applied.

### 1c. Rank / threshold controls
- `thc::thresh` member, default 1e-5, ctor `thc.cpp:120` from ptree key "thresh"; printed at `thc.cpp:155`. Validated `thresh==0.0 || thresh>1e-14` (`thc.icc:776-777`).
- `nIpts` (arg `max`) is the hard count. Guard `thresh>0 || nIpts>0` (`thc.icc:774`). Stopping: `if(thresh>0 && thresh>old_max) break` (`thc.icc:1086`, `:1335`) and `if(nIpts>0 && nchol>=nIpts) break` (`thc.icc:1307`).
- Initial storage guess when nIpts<=0: `nmax = 6*sqrt(na*nb)` (`thc.icc:788`), grown by `2*sqrt(na*nb)` on overflow (`thc.icc:1272-1287`).
- Small-pivot cutoff: `utils::check(old_max > 1e-14, ...)` aborts (`thc.icc:1083`); also monotonicity check `old_max >= curr_max` (`thc.icc:1327`) and isfinite checks (`thc.icc:1088`,`:1330`).
- `block_size` = `default_cholesky_block_size` (ptree "chol_block_size", default 8, `thc.cpp:119`); FORCED to 1 when thresh==0.0 with an app_warning (`thc.icc:904-907`).
- The N_mu = c*N_orb convention lives in the caller/CLI: `check_eri_symmetry.cpp:79` option "nIpts_c" default 4.0; see also eri_utils.hpp:43-44 ("nIpts": "0", "thresh": "0.0").
- Last-block reordering when nIpts truncates mid-block: keeps the nv largest |W(i,i)|^2 pivots (`thc.icc:1202-1229`).

### 1e. Selection helper utilities
- `utils::max_element_multi(a, maxs, indx)` — `/Users/mmorales/Projects/ISDF_metric/coqui/src/utilities/functions.hpp:37`. Local top-N argmax over the real part; GPU arrays are copied to host first (`functions.hpp:59-63`, explicit "temporary hack").
- `utils::find_distributed_maximum(gcomm, l, g)` — `functions.hpp:73`. Implemented as Gather(N per rank) -> root argmax -> Broadcast (NOT an Allreduce/MPI_MAXLOC). Returned `g(i).second` encodes owner = idx/N and local slot = idx%N.
- `utils::chol<false,W_type>(Abb,piv,cut)` — `/Users/mmorales/Projects/ISDF_metric/coqui/src/utilities/functions.hpp:188`. Serial pivoted Cholesky of the small block matrix; drops columns with |v| <= cut, returns conj-permuted W and `piv(n)=nc` = accepted count.
- `thc::chol(Arr&,piv,cut)` at `/Users/mmorales/Projects/ISDF_metric/coqui/src/methods/ERI/thc_aux.icc:2366` is a near-duplicate member version (appears unused; `utils::chol` is the one called).
- Same pivoted-Cholesky pattern is reused in `/Users/mmorales/Projects/ISDF_metric/coqui/src/methods/ERI/cholesky.icc:636,840,961,1126`, `/Users/mmorales/Projects/ISDF_metric/coqui/src/utilities/distributed_cholesky.hpp:106,220`, `/Users/mmorales/Projects/ISDF_metric/coqui/src/methods/embedding/cholesky.hpp:106`.

### 1f. Symmetry-adapted variant
- `thc::chol_metric_impl_ibz` at `/Users/mmorales/Projects/ISDF_metric/coqui/src/methods/ERI/thc.icc:43` — same algorithm restricted to Gamma/IBZ (Matthews, JCTC 2020, 16, 1382 per doc at `thc.h:588-590`). Mirror line anchors: diagonal `thc.icc:267`, pivot loop start `thc.icc:316-343`, chol `thc.icc:485`, last-block reorder `thc.icc:487-506`, diag update/argmax `thc.icc:597`.
- N_mu resolution in the reader: `_Np` from ptree "nIpts" (`thc_reader_t.hpp:93`); thresh sentinel auto-resolve at `thc_reader_t.hpp:108-113` (thresh<0 -> 1e-13 if nIpts>0 else 1e-5); `_Np` is RESET to the actual returned count `_rp.size()` at `thc_reader_t.hpp:434`, `:1710`, `:1872`.
- ptree builders: `make_thc_ptree` / `make_thc_reader_ptree` at `/Users/mmorales/Projects/ISDF_metric/coqui/src/methods/ERI/eri_utils.hpp:134-184` (thresh default 1e-10 there); check `nIpts>0 or thresh>0.0` at `eri_utils.hpp:175`.

## 2. Zeta (interpolation vector) solve and Coulomb assembly

### 2a. Entry
- `thc::intvec_impl<MEM,return_coul_matrix>(IPts, Xa, Xb*, return_Ivec, a_range, b_range, pgrid3D)` — `/Users/mmorales/Projects/ISDF_metric/coqui/src/methods/ERI/thc.icc:1517-1778` (decl `thc.h:408-413`).
  Reached from public `thc::evaluate(...)` `thc.cpp:248` (ISDF), `thc.cpp:278` (with C_skai rotation), `thc::evaluate_isdf_only` `thc.cpp:363`.
- LS-THC variant (fit against a DF/Cholesky tensor B instead of pair densities): `thc::intvec_impl(int iq, IPts, a_range, b_range, B)` at `thc.icc:1781`, entered from `thc::evaluate(int iq, ri, B, ...)` `thc.cpp:228`.

### 2b. Theta = Z[:, I] and the normal equations
- `get_ZquG_Cquv<MEM>(IPts,Xa,Xb,a_range,b_range,pgrid3D)` at `thc.icc:1570` builds BOTH sides in one pass; dispatcher `/Users/mmorales/Projects/ISDF_metric/coqui/src/methods/ERI/thc_aux.icc:761` picks
  `get_ZquG_Cquv_fft_shared_memory` (HOST, `thc_aux.icc:1725`), `get_ZquG_Cquv_fft<MEM>` (device, `thc_aux.icc:1098`) or `get_ZquG_Cquv_rspace<MEM>` (no wfc grid, `thc_aux.icc:803`).
- Z^q_u(r) = sum_{k,ab} phi_a(r_u)phi_b*(r_u) phi_a*(r) phi_b(r) is built in real space, then:
  - C_quv (= Theta Theta^H, the collocation Gram) is literally the COLUMN SELECTION of Z at the interpolation points: comment "4d. C(q,u,v) = Z(q,u,rv)",
    implemented as `nda::copy_select(false,1,IPts,...,C_quv.local()(iq,Irng,all))` — device path `thc_aux.icc:1653-1661`, shared-memory path `thc_aux.icc:2286-2296`.
  - Z_qug is obtained by an r->G FFT of the phase-corrected Z (steps 4e/4f, `thc_aux.icc:1663-1699`; FFT under timers FFTPLAN/FFT) then mapped to the truncated grid via `rho_g.gv_to_fft()`.
  - r-space (no-FFT) path returns Z_qur/C_quv at `thc_aux.icc:985-1071`.
- SOLVE: per q-point, distributed (SLATE) solve of C_quv * zeta = Z_quG in `thc.icc:1587-1610`, timer "LSSolve".
  `use_least_squares` (ptree flag, default false, `thc.cpp:124`) picks `math::nda::slate_ops::least_squares_solve<true>(Ciq,Ziq)` (`thc.icc:1603`) else `slate_ops::lu_solve<true>(Ciq,Ziq)` (`thc.icc:1606`).
  NOTE: this is an LU / QR-least-squares solve of the normal equations — there is NO pinv / truncated-SVD / Cholesky-of-Theta*Theta^H regularization anywhere in this path. `info` is only checked for ==0; no condition number or rank diagnostic.
- Result overwrites Z_quG in place, so afterwards Z_quG holds zeta^q_u(G).

### 2c. Overlap inverse and G=0 heads (timer "ZBAR")
- `slate_ops::multiply(Z_quG, dagger(Z_quG), C_quv)` at `thc.icc:1618` recomputes S^q_uv = <zeta_u|zeta_v>; then S^{-1} is obtained by solving against an explicit identity buffer with the same lu_solve/least_squares_solve (`thc.icc:1619-1652`).
- Chi^q_u(G=0) = Z_qu extracted at `thc.icc:1666-1676` (Allreduce over the Z_quG communicator); dual Zbar_qu = S^{-1} Z_qu via `nda::blas::gemv` + Allreduce (`thc.icc:1679-1688`). These are saved as `interpolating_vectors_G0` / `dual_interpolating_vectors_G0` (`thc.cpp:399-400`).

### 2d. Coulomb kernel contraction V_uv = zeta^H v zeta (timer "VCoul")
- `thc.icc:1690-1725`: for each q, `vG.evaluate(sqrtVg, lattv, rho_g.g_vectors(), 0, Q(q,:))`, take elementwise sqrt, scale Z_quG by sqrt(v(G)) (`thc.icc:1714-1722`),
  then `math::nda::slate_ops::multiply(Z_quG, dagger(Z_quG), C_quv)` (`thc.icc:1724`) and `nda::tensor::scale(1/volume/nkpts, C_quv.local())` (`thc.icc:1725`).
  So V(q,u,v) = (1/(Omega*Nk)) sum_G zeta*_u(G) v(G) zeta_v(G); C_quv is REUSED as the output V buffer.
- If `return_Ivec`, the sqrt(v) factor is divided back out and the v(G)=0 component restored from `Zloc_store` (`thc.icc:1727-1755`); GPU branch is unimplemented (`utils::check(false,"finish, need kernel or a solution")` at `thc.icc:1748`, and `thc.icc:1719`).
- Saved by `thc::save` -> "coulomb_matrix" (`thc.cpp:403`) or "interpolating_vectors" (`thc.cpp:440-448`).

## 3. Diagnostics, logging, timers

### 3a. Logging infrastructure
- `/Users/mmorales/Projects/ISDF_metric/coqui/src/IO/app_loggers.h`: `app_log(int level, fmt, ...)` gated by global `__app_output_level__` (`app_loggers.h:45-47`), `app_debug` gated by `__app_debug_level__` (`:101`), plus `app_warning`, `app_log_flush` (`:117`). Setup via `setup_loggers(root, output_level=2, debug_level=0)` (`:39`).
- Level usage in the THC files: 1 = headers/summary (`thc.cpp:149-159`, `thc.icc:1529-1531`, `:1564-1566`); 2 = timers/block sizes/memory (`thc.cpp:462-489`, `thc_aux.icc:777,791,1329`); 3 = per-pivot trace `nchol, max |D|` and per-pivot |W(v,v)|^2 (`thc.icc:1080`, `:1227`, `:1233`; ibz `:329`, `:512`, `:518`); 4 = detailed interpolating-point timer block (`thc.icc:1347-1357`, ibz `:629-641`).
- Warnings: block_size forced to 1 (`thc.icc:906`, ibz `:202`), unused processors (`thc_aux.icc:1146`, `:1774`), insufficient-memory banner (`thc_aux.icc:1309-1315`, `:1907-1913`).

### 3b. Timers
- `utils::TimerManager Timer` member (`thc.h:329`); names pre-registered in the ctor at `thc.cpp:136-139`. `TimerManager::start(str)` uses `getOrAdd` (`/Users/mmorales/Projects/ISDF_metric/coqui/src/utilities/Timer.hpp:138`), so unregistered names (e.g. "ZBAR", "ip_setup_comm", "ip_chol", "ip_update_res", "ip_chol_gemm", "ip_chol_hadd") are created on first use.
- Aggregate report `thc::print_timers()` at `/Users/mmorales/Projects/ISDF_metric/coqui/src/methods/ERI/thc.cpp:460-490` (level 2), with `utils::memory_report(2)`. `thc::reset_timers()` at `thc.h:305`.
- Selection-side timers: TOTAL, IntPts/IntVecs, DistOrbs, IpIter, ip_setup_comm, ip_COMM, ip_chol, ip_SERIAL, ip_update_res, COMM.
- Solve/assembly timers: IntVecs, LSSolve (`thc.icc:1582`), ZBAR (`thc.icc:1617`), VCoul (`thc.icc:1690`), ALLOC, GEMM, shmX, TUR, ZUR, EXTRA, FFT, FFTPLAN, IO_SAVE, IO_ORBS.
- `utils::memory_report(level, tag)` sprinkled at `thc.cpp:174,209,238,262,293,393,416` and `thc.icc:1568`.

### 3c. Existing fit-error / quality diagnostics — SUMMARY: essentially NONE in production code
- The ONLY quality signal emitted during point selection is the running Cholesky residual `max |D|` at app_log level 3 (`thc.icc:1080`) and the per-pivot diagonal `|W(v,v)|^2` (`thc.icc:1227,1233`). There is no summary line reporting the final residual/thresh actually achieved for `chol_metric_impl`.
- `chol_metric_impl_ibz` DOES emit one extra diagnostic that the non-ibz path lacks: "[WARNING] thresh={} reached after {} interpolating points, before reaching requested nIpts={}" at `thc.icc:337` and `:614`. The main `chol_metric_impl` has no equivalent (compare `thc.icc:1086` and `:1335`).
- After the zeta solve there is NO fit-error diagnostic at all: `lu_solve`/`least_squares_solve` only have `info==0` checks (`thc.icc:1604`, `:1607`, `:1645`, `:1648`); the residual ||Z - Theta^H zeta|| is never formed, no condition number, no rank report. This is the natural insertion point for an ISDF-fit metric.
- Sanity guards that indirectly detect a bad fit: residual monotonicity `old_max >= curr_max` (`thc.icc:1327`), `isfinite` (`thc.icc:1088,1330`), `old_max > 1e-14` floor (`thc.icc:1083`).
- Out-of-band (offline/test) error measurement exists and is the template for any new diagnostic:
  - `/Users/mmorales/Projects/ISDF_metric/coqui/src/methods/ERI/tests/test_thc.cpp:59` `detail::eval_V_thc(Xa,Xb,Xc,Xd,Vuv)` reconstructs (ab|cd) from the THC factors; the lambda at `test_thc.cpp:95-156` reports per-q mean/max abs difference (`app_log(3,"q:{}, ME:{}, Max:{}")`), and TEST_CASEs at `:345`, `:414` log "avE/mxE".
  - `/Users/mmorales/Projects/ISDF_metric/coqui/src/methods/ERI/check_eri_symmetry.cpp:297-331` — mean_abs_diff / max_abs_diff over symmetry-related ERIs; CLI option "nIpts_c" (default 4.0) at `:79`.
  - `/Users/mmorales/Projects/ISDF_metric/coqui/src/methods/ERI/compare_eri.cpp:331-354` — cross-decomposition mean/max abs and scaled diffs; reports nIpts at `:418`.
- PAW/augmentation-specific completeness gates and their measured residuals are documented (not computed at runtime) in `/Users/mmorales/Projects/ISDF_metric/coqui/src/methods/ERI/thc_reader_t.hpp:330-365`; a q-aware channel-ranking selector `select_aug_channels_qaware` is invoked at `thc_reader_t.hpp:377`. Diagnostic-only gates `_paw_vgl`, `_paw_onsite` at `thc_reader_t.hpp:133-143`, `:2409`.
- Repo context: current branch is `isdf_metric` (also `origin/isdf_vertex`, `origin/isdf_vertex_leanW`); HEAD `7cdd567`. No fit-error code present on it yet.
