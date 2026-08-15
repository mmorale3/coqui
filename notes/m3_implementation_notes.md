# M3 (knob 3) implementation notes — two-pass Coulomb re-ranking

Insertion point: in `chol_metric_impl` (and later `_ibz`), AFTER the pivot loop
(`Timer.stop("IpIter")` / selection summary), BEFORE the Xskau assembly.

Existing objects at that point (verify names in code):
- `R` : (nmax, r_loc) = L^H rows; L(g,j) = conj(R(j,g)). Distributed over r (columns).
- `rn` : global grid indices of accepted pivots, length nchol.
- `Pskau/Pskbu` : per-pivot collation accumulations; u axis indexed in pool order.

Algorithm (all only when `isdf_metric != "l2"`):
0. Require nIpts>0 (runtime check). Pool target N1 = ceil(isdf_pool_factor*nIpts);
   run the existing loop with nIpts_pool = N1 (thresh may stop it earlier; if
   final nchol <= nIpts, warn and skip pass 2 = no-op). `nmax` sizing must use N1.
1. ell = L[P,:] = (R[:, rn])^H : each rank owns the columns of R at its local
   pivot grid indices -> assemble (nchol x nchol) ell by Allreduce of a zero-filled
   buffer (owner fills). Small.
2. Row-redistribute R: (nchol, N_g) col-dist -> row-dist ({comm.size(),1}) using
   darray redistribute (same pattern as DistOrbs). Each rank: for each local row j,
   scatter conj(R(j,:)) full grid vector into the rho_g FFT box? NO conj+scatter
   subtlety: row already holds full grid after redistribute; reshape to
   (rows_loc, m0,m1,m2), batched forward FFT (math::nda::fft, F.forward) of
   CONJUGATED rows -> hatL rows. Then map FFT box -> truncated G list via
   rho_g.gv_to_fft() gather; apply sqrt(vbar(G)) (and gcut mask) -> V rows
   (rows_loc x N_G).
   FFT normalization irrelevant (scale-invariant pivoting).
3. Redistribute V to G-distributed (all nchol rows, G slab) -> M_loc = V_loc V_loc^H
   via ZHERK/GEMM (nchol x nchol), comm.Allreduce -> M = L^H vbar L (replicated).
4. K = ell * M * ell^H (replicated, two small GEMMs).
5. Serial full pivoted Cholesky on K (all ranks identically, deterministic;
   reuse utils::chol pattern but need PIVOT ORDER + rank cap nIpts: write a
   small local routine `pivoted_chol_order(K, nIpts) -> sel[nIpts]` returning
   pool positions in elimination order).
6. Subset/reorder: rn_final(i) = rn(sel(i)); Pskau columns u -> keep sel columns
   in order (same for Pskbu); nchol = nIpts. Then fall through to existing
   Xskau assembly unchanged.

vbar(G): via `vG.evaluate(Vg, mf->lattv(), rho_g.g_vectors(), 0, q)`:
- "bare": q-average over mf->Qpts() when isdf_metric_qavg else q=0 row;
  kernel zeroes |q+G|^2 <= cutoff (handles G=0).
- "attenuated": multiply bare by exp(-|G|^2/(4 omega^2)); omega =
  isdf_metric_omega>0 ? that : 2*pi/L with L = cbrt(volume) (auto).
- gcut: isdf_metric_gcut>0 -> zero vbar where |G|^2/2 > gcut.

Edge cases:
- thresh reached before N1: pool = nchol_reached (warn); still re-rank if > nIpts.
- s=1: N1 = nIpts -> pass 2 selects all pool points; ORDER may change but the SET
  is identical -> Xskau column order permutes. Zeta solve is invariant under
  point permutation, but bitwise l2-default MUST bypass pass 2 entirely
  (isdf_metric=="l2" -> never enter).
- ibz mirror (M3b): same recipe; its R/rn structures are the mirror anchors at
  thc.icc:~330-640.

Memory: V row-dist = N1 x N_G complex (prod: 6k x 30k = 2.9 GB spread over ranks
-> fine); M,K,ell = N1^2 replicated (6k -> 576 MB complex each; acceptable on
--mem=0 genoa; tests trivial).

## M3b (ibz mirror) — corrected strategy

The ibz impl's Cholesky factor R lives on the IRREDUCIBLE grid subset
(ir_rgrid), where an FFT is not meaningful, and its Gram is the symmetrized
C (Matthews). The ell L^H identity therefore cannot be pushed through the
FFT route directly. Instead, pass 2 for ibz:

1. Pool pass exactly as main impl (existing loop, rank N1, gives pool rn).
2. Re-read orbitals on the FULL grid (r-distributed, like the main impl's
   DistOrbs) — extra read, acceptable at s~2.
3. Build the pool's full-grid Gram columns C_full(j, g) for the N1 pool
   points with main-impl-style gather+GEMM+Hadamard (cost ~ one extra pass-1,
   i.e. ~s x — same scaling).
4. FFT the N1 rows, weight by sqrt(vbar), K = V V^H (+ Allreduce), dense
   pivoted Cholesky, subset rn + Pskau as in the main impl.
   NOTE: this K is C v C with the UNSYMMETRIZED full-grid C restricted to the
   pool — the correct Coulomb objective; symmetry only reduced the SEARCH in
   pass 1, and the pool subset ordering does not need the symmetrized form.

Anchor: main impl `int nmax = (nIpts>0 ? nIpts : 6*sqrt(na*nb))` (thc.icc ~913).
M3 wiring: define `nIpts_sel = (isdf_metric=="l2" ? nIpts : ceil(isdf_pool_factor*nIpts))`
right there; use nIpts_sel for nmax, for the `nchol>=nIpts` stop, and in the min()
at the newv clamp; pass 2 then reduces the pool back to nIpts. Require nIpts>0
when isdf_metric!="l2" (utils::check at entry).
