# B: grid orbitals / Coulomb kernel / USPP-PAW

## 1. Grid orbitals u_n(r) for ISDF

### Producer / owner
- `/Users/mmorales/Projects/ISDF_metric/coqui/src/mean_field/distributed_orbital_readers.hpp:57` `mf::read_distributed_orbital_set<local_Array_t>(mfobj, comm, OT, pgrid_out, ispin, kp, orb, block_size)` is THE grid-orbital factory. `OT`: `'w'` = truncated wfc G-grid (ecutwfc PW list), `'g'` = density FFT G-grid, `'r'` = real-space density FFT grid. Ranks 2/4/5 supported -> shapes (skap,nnr) / (s,k,b,npol*nnr) / (s,k,b,npol,nnr).
- Same file :167,:190,:214 — when `OT=='r'` and `mf.orb_on_fft_grid()`, the batched inverse FFT `math::fft::invfft_many(Offt)` is applied in place on the local slab reshaped to (nbatch, m0,m1,m2). So "grid orbitals" = read G-coeffs from h5 into an FFT box + batched inverse FFT. No distributed FFT: the G/r axis is forced un-split (`pgrid[rank-1]==1`) when OT=='r'.
- `:244` `read_distributed_orbital_set_ibz` = same but k range limited to IBZ.
- There is NO persistent `distributed_orbital_set` class. The owner is a `math::nda::distributed_array` (`memory::darray_t<memory::array<MEM,ComplexType,N>, mpi3::communicator>`) returned by value; each caller owns its own copy.
- Backend dispatch: `mf::MF::get_orbital_set` (`src/mean_field/MF.hpp:366-376`, std::visit) -> `qe_readonly::get_orbital_set` (`src/mean_field/qe/qe_readonly.hpp:240,251,265`) -> `orbital_set_from_h5`; bdft variant at `src/mean_field/bdft/bdft_readonly.hpp:502,522,545`.

### Wavefunction cutoff / G-vector infrastructure
- `/Users/mmorales/Projects/ISDF_metric/coqui/src/grids/g_grids.hpp:44` `class grids::truncated_g_grid` holds ecut_, fft_mesh(3), recv(3,3), ngm, `gvecs(ngm,3)` (cartesian G), `g2fft(ngm)` (G index -> flat FFT index) and optional `fft2g`. Accessors: `ecut()`, `mesh()`, `size()`, `g_vectors()`, `gv_to_fft()`, `fft_to_gv()`, `reciprocal_vectors()`.
- Ctors: from ecut+mesh+recv (enumerates G inside 2*ecut), or from an explicit Miller-index array (QE case).
- `g_grids.hpp:250` `map_truncated_grids(full,g_in,g_out,map)` and `:311` `map_truncated_grid_to_fft_grid` build index maps between wfc and rho grids.
- MF side: `mf.wfc_truncated_grid()` (`MF.hpp:179`, qe at `qe/qe_readonly.hpp:98`, built by `detail::wfc_grid_from_h5(sys)`); `mf.ecutrho()`, `mf.fft_grid_dim()` (smooth/dffts), `mf.fft_grid_dim_aug()` (dense/dfftp), `mf.nnr()`, `mf.recv()`.
- k+G: G-list is k-independent; the k-dependence enters as phase factors on the r-grid (`utils::rspace_phase_factor`, thc.icc:936) and via q in the Coulomb kernel, not via per-k G lists.

### THC's own grid
- `/Users/mmorales/Projects/ISDF_metric/coqui/src/methods/ERI/thc.cpp:54` `detail::make_grid(comm, ecut, mf)` builds `thc::rho_g` (a `truncated_g_grid`). Default `thc::ecut = 1.4*ecutwfc` (or 0.4*ecutrho); PAW/NCPP note in the comment: with no explicit ecut it uses `mf.ecutrho()` on `fft_grid_dim_aug()` (dense mesh) so the rho_g G-list and any augmentation extension are sized consistently.
- `thc.cpp:73` `detail::make_wfc_to_rho(...)` -> `thc::swfc_to_rho`, a shared_array<long,1> mapping each wfc-grid G to its flat index in the rho_g FFT box. Used everywhere to scatter 'w'-grid coefficients into the (larger) rho_g FFT box before the inverse FFT.
- `thc::rho_g` exposed via `thc::g_grid()` (`thc.h:311`).

### Pipeline: build vs. selection vs. zeta solve (orbitals are REBUILT, not reused)
- Interpolation-point selection: `thc::interpolating_points` (`thc.cpp:163`, `:198`) -> `thc::chol_metric_impl` (`thc.icc:763`) or `chol_metric_impl_ibz` (`thc.icc:43`). Inside `chol_metric_impl` at `thc.icc:806-901` (Timer "DistOrbs") it reads orbitals itself: if `custom_grid` (rho_g mesh != mf fft mesh) it reads `'w'`, scatters via `swfc_to_rho` with `nda::copy_select`, and does its own `math::nda::fft<true> F(...); F.backward(P4d)` (thc.icc:823-826, and again at :877-880 for the b-side); otherwise it reads `'r'` directly. Then pivoted-Cholesky on the ZZ^H metric selects the points.
- Zeta solve / ERI assembly: `thc::evaluate` -> `thc::get_ZquG_Cquv` (`thc_aux.icc:761`) -> `get_ZquG_Cquv_fft` (`thc_aux.icc:1098`) / `_fft_shared_memory` (`:1725`) / `_rspace` (`:803`). These RE-READ the orbitals from disk: `thc_aux.icc:1225,1230` (and `:1861,:1866`) call `read_distributed_orbital_set<local_5Array_t>(...,'w',...)` on the IBZ k's and re-do FFTs internally (they keep them in 'w'/G form and FFT per interpolation-point block).
- Consequence for a metric change: the grid-orbital array used for point selection lives only inside `chol_metric_impl`; it is a *separate* array from the one used in the zeta solve. A new selection metric only needs to touch `chol_metric_impl`/`chol_metric_impl_ibz`, but any change to what "u_n(r)" means would have to be mirrored in `get_ZquG_Cquv_*`.
- The only things handed from selection to the solve are `IPts` (long,1 grid indices) and `Xa`/`Xb` (orbitals evaluated at those points, rank-4 distributed arrays).

## 2. Coulomb kernel v(q+G) in G-space

### The reusable kernel evaluator (canonical)
- `/Users/mmorales/Projects/ISDF_metric/coqui/src/potentials/coulomb.hpp:57` `pots::coulomb_t`. Two entry points:
  - `:111` `evaluate(V, lattv, gv, kp, kq)` — fills `V(g)` for a *precomputed G list* `gv(ngm,3)` (cartesian), with the shift `dk = kp - kq`. This is the "v(q+G) on the truncated G list" routine used by THC/Cholesky (pass `kp = 0`, `kq = Q`).
  - `:157` `evaluate_in_mesh(g_rng, V, mesh, lattv, recv, kp, kq)` — same but enumerating G directly from an FFT mesh + recv, so no G list needed.
- Options from ptree: `ndim` (2 or 3), `cutoff` (default 1e-8; |q+G|^2 <= cutoff -> v = 0, i.e. the *default divergence treatment inside the kernel is simply "zero out G=0"*), `screen_type` ({none,yukawa,erfc,erf} in 3d; {none,tanh} in 2d), `screen_length`.
- Scalar formula: `/Users/mmorales/Projects/ISDF_metric/coqui/src/potentials/potentials_impl.hpp:193` `pots::detail::eval_3d_impl::operator()(ig)` -> `V(g) = 4*pi/(g2+screen_length)` etc.; 2d at `:43` `eval_mesh_2d_impl` / `eval_2d_impl`. CUDA mirrors in `numerics/device_kernels/kernels.h` (`kernels::device::eval_3d`, `eval_mesh_3d`, ...).
- Type-erased wrapper: `/Users/mmorales/Projects/ISDF_metric/coqui/src/potentials/potentials.hpp:54` `pots::potential_t` (std::variant<coulomb_t>), built from ptree subnode `"potential"`.
- Legacy/duplicate: `/Users/mmorales/Projects/ISDF_metric/coqui/src/hamiltonian/potentials.hpp:47` `hamilt::potential_g(V,gv,kp,kq,type)` and `:81` `hamilt::potential_full_g(...)` — hardcoded 4pi/|G+kp-kq|^2 with a 1e-8 G=0 cut, no screening. Still used by some PAW/pseudo code.

### Where THC applies it (the Coulomb tensor from zeta)
- Member: `/Users/mmorales/Projects/ISDF_metric/coqui/src/methods/ERI/thc.h:341` `pots::potential_t vG;` constructed at `thc.cpp:117` from ptree child `"potential"`.
- **The assembly step: `/Users/mmorales/Projects/ISDF_metric/coqui/src/methods/ERI/thc.icc:1690-1756`** (inside `thc::intvec_impl`, Timer "VCoul"):
  1. `vG.evaluate(sqrtVg, mf->lattv(), rho_g.g_vectors(), v_zero, Q(q,all))` -> v(q+G) on the *rho_g* truncated G list (thc.icc:1700).
  2. element-wise `sqrt` -> sqrt(v).
  3. scale `Z_quG(q,u,G) *= sqrtVg(G)` (thc.icc:1717 host / :1720 device `nda::tensor::elementwise`).
  4. `math::nda::slate_ops::multiply(Z_quG, dagger(Z_quG), C_quv)` then `scale by 1/(volume*nkpts)` (thc.icc:1724-1725). => `V_{uv}^q = (1/(Omega Nk)) sum_G zeta*_u(q+G) v(q+G) zeta_v(q+G)`.
  5. If `return_Ivec`, the sqrt(v) factor is divided back out (thc.icc:1729-1753) and the G with v=0 restored from `Zloc_store`, so the returned `Z_quG` is the bare zeta(G).
- Cholesky analogue: `/Users/mmorales/Projects/ISDF_metric/coqui/src/methods/ERI/cholesky.icc:72` and `:204` use the identical `vG.evaluate(sqrtVg, lattv, rho_g.g_vectors(), v_zero, Q)` pattern.
- Also `thc.icc:1656-1676`: `Z_qu` = zeta at G=0 (found by searching `rho_g.gv_to_fft()` for index 0) and `Zbar_qu = S^{-1} Z_qu`; these are the head terms consumed by the GW divergence treatment.

### Divergence treatment ("gygi", ignore_g0, ...) — NOT in the kernel
- The kernel itself always sets v(q+G=0)=0. The `div_treatment` string is a *post-hoc* q=0 head correction handled downstream:
  - `/Users/mmorales/Projects/ISDF_metric/coqui/src/methods/GW/g0_div_utils.hpp:44` `solvers::div_utils` — `extrapolate_eps_inv_q0` (:74), `filter_qpts` (:47), `eps_inv_head_t` (:237/:280). Choices parsed by substring: `gygi` (new: polynomial extrapolation to q=0 over closest q-points), `gygi_extrplt` (deprecated), `gygi_average`, `gygi_smallest_q`, `ignore_g0`, plus a `2d` and `order_{N}` suffix.
  - Plumbed from `/Users/mmorales/Projects/ISDF_metric/coqui/src/methods/MBPT_drivers.cpp:129-132` (`div_treatment`, `hf_div_treatment`, `bare_div_treatment`, all default "gygi") into `solvers::gw_t`, `solvers::scr_coulomb_t` (`methods/scr_coulomb/scr_coulomb_t.h:87,261,270`), `hf_t`, and `embed_eri_t` (`methods/embedding/embed_eri_t.h:76-90`).
  - HF exchange divergence uses a Madelung-type constant: `/Users/mmorales/Projects/ISDF_metric/coqui/src/utilities/madelung_utils.hpp:70` `utils::madelung(lattv, recv, mp_mesh, fft_mesh, prec)`.
- => a change of ISDF selection metric that introduces a v(G)-weighted Gram matrix should reuse `pots::coulomb_t::evaluate` on `thc::rho_g.g_vectors()`; the q=0 zero is baked in via `cutoff`.

## 3. USPP/PAW additions of this branch around THC/ISDF

Branch `isdf_metric`, tip commits `7cdd567` (QE/ABINIT augmented-PP converters), `d114c1d` ("Ultrasoft and PAW pseudopotentials in the THC/ISDF two-electron integrals"), `2534861` (QE fixtures) on top of `7e27355` (merge from develop).

### New files under src/hamiltonian/paw/ (all branch-added)
- `local_isdf.hpp` (393 L) — `hamilt::paw::species_local_isdf`; per-species full-rank symmetric-pair factorization of the augmentation pair density Qhat_{a,IJ}(s) = sum_lambda U_{a,lambda I} U_{a,lambda J} eta_{a,lambda}(s), nlambda = nh^2 (1 row per diagonal (I,I); +/- pair of rows per off-diagonal I<J). Also `compute_K_a` giving K_{lambda xi} = sign*sign*DeltaC[i(l),j(l),i(x),j(x)] (the PAW on-site kernel).
- `local_isdf_compress.hpp` (774 L) — **`enum class hamilt::paw::isdf_metric { L2, Coulomb }`** (`:~54`), `detail::make_metric_weights(mill, recv, omega, metric)` (`:~118`, w(G)=1 or 4pi/(Omega|G|^2), 0 at G=0), `detail::gram_column(qgm,nt,pivot,w,...)`, `pivoted_cholesky_qgm_pairs(psp,nt,recv,omega,metric,tol)`, `build_local_isdf_compressed_by_norm(...)`. This is a *pair-channel* pivoted Cholesky in a selectable metric — NOT the global interpolation-point selection.
- `local_isdf_h5.hpp` (247 L) — cache I/O; `kLocalISDFGroup`, `kISDFSelectionTag` (selection-rule version tag, cache is rejected if the tag differs).
- `paw_aug_thc.hpp` (662 L) — the THC-side glue. Key symbols: `paw_aug_layout` / `make_paw_aug_layout(psp,isdf,N_mu)` (`:45`,`:53`; composite row index Lambda = N_mu + atom_aug_offset[a] + lambda), `build_eta_on_rho_g_at_q` (`:93`) and `..._chunk` (`:234`) giving eta_{a,lambda}^q(G) with structure factor e^{-i(q+G).tau_a} on the *thc rho_g grid*, `add_K_a_to_tile` (`:404`), **`coulomb_weights_on_rho_g` (`:449`) and `coulomb_weights_on_rho_g_at_q` (`:468`)** — local re-implementations of 4pi/(Omega|q+G|^2) with G=0 -> 0 (they do NOT go through `pots::coulomb_t`), `compute_VGL_q0_on_rho_g` (`:538`), `compute_VLL_q0_on_rho_g` (`:568`), `add_K_a_to_LL` (`:584`), `fill_Y_rows_for_sk` (`:617`) building Y_{a lambda,n}^k = sum_I U_{a,lambda I} P^k_{n,aI}.
- Also new: `paw_aug_q_eval.hpp` (CoQui-side qvan2 / qrad tables), `paw_onecenter.hpp`, `paw_radial.hpp`, `paw_symmetry.hpp`, `paw_runtime_caches.hpp`, `v_h_paw.hpp`, `v_x_paw.hpp`, `hartree_xc_energy.hpp`, `paw_energy_check.hpp`.

### Changes in methods/ERI
- `thc.h` / `thc.icc` diff vs main is SMALL and does not touch selection: (a) `return_Sinv_Ivec` renamed to `return_Ivec` and the optional 4th return value changed from IquG = S^{-1}Z to the **bare zeta_{qu}(G)** (`thc.icc:1697-1755`: the sqrt(v) factor is divided back out and the v=0 G restored); (b) new public accessors `thc::g_grid()`, `thc::volume()`, `thc::get_mf()` (`thc.h:304-320`) explicitly "needed by external callers that wish to evaluate auxiliary functions (e.g. PAW augmentation eta^q(G)) on the same grid".
- `paw_exx_options_parse.hpp` (new, 58 L) — shared toml -> `hamilt::paw_exx_options` parser (`vv_compensation`, `aug_lmax`, `qfac_cache_mb`), used by both `[interaction.thc]` and `[interaction.hamilt]`.
- `hamilt_eval_t.hpp` (new, 132 L) — direct (non-THC) Hamiltonian evaluator carrying the same exx options.
- `cholesky.cpp:73-98` — Cholesky ERIs **hard-abort** for USPP/PAW (no augmentation path); override knob `allow_smooth_only_aug_pp` for diagnostics. So THC is the only augmented ERI route.
- `thc_reader_t.hpp` (+1497 L) — where everything is assembled.

### Answer to the selection question
**Interpolation-point selection is UNCHANGED and is NOT augmentation-aware.**
- `thc_reader_t::build()` (`thc_reader_t.hpp:419-450`) calls the stock `thc::interpolating_points<MEM>(0, _Np, x_range, y_range)` — i.e. `chol_metric_impl` on the *smooth pseudo-orbitals only*, plain L2/overlap Gram, no augmentation contribution. It then calls `thc::evaluate(...)` with `ret_zeta_in_eval = host_mem && any_aug_species` to get zeta(G) back.
- The augmentation enters **only after**, in `thc_reader_t::augment_thc_with_paw<MEM>(dzeta_quG)` (`thc_reader_t.hpp:599`):
  1. `make_paw_aug_layout` -> N_total = N_smooth + N_aug (`:612`).
  2. **X row augmentation** (`:694-730`, Timer "PAW_AUG.X_aug"): `_X_shm` grows from (s,k,N_smooth,nbnd) to (s,k,N_total,nbnd); the extra rows are `fill_Y_rows_for_sk` (projector overlaps), NOT collocation at any grid point. So the "interpolating basis" is enlarged by atom-local channels that were never selected by a pivoting procedure over r.
  3. **_dZ grows** to (nq_ibz, N_total, N_total) and the smooth GG block is embedded (`:737-760`).
  4. **New Coulomb blocks assembled in G space** (`:988-1100`, Timers PAW_AUG.V_GL / V_LL): with w(g) = `coulomb_weights_on_rho_g_at_q(rho_g, q_cart, omega)` and q_cart = -Qpts(iq),
     - `V_GL(mu,lambda) = Omega * sum_g zeta(mu,g) conj(eta_w(lambda,g))`, `V_LG = conj(V_GL)^T`,
     - `V_LL(lambda,xi) = Omega^2 * sum_g eta(lambda,g) w(g) conj(eta(xi,g))` (conjugate on the SECOND index — there is a long comment warning that the earlier code stored the transpose and broke ln|det(I-Pi Z)|),
     - plus the grid-free on-site `add_K_a_to_LL` for PAW species only.
  5. `dense_LL` variant runs the LL sum on the dense augmentation sphere instead of rho_g when eta is not band-limited to rho_g.
- Gates/knobs (`thc_reader_t.hpp:130-168`): `paw_aug` (master), `paw_onsite` (K_a, default true for PAW), `paw_vgl`, `paw_vll`, `paw_isdf_tol` (1e-12), `paw_isdf_cache_h5`, `paw_aug_ecut` (truncate eta at |q+G|^2/2, applied to eta not to the kernel so Z stays a true Gram matrix / PSD), **`paw_isdf_metric` (default "coulomb", alternative "l2")**.

### The one augmentation-aware "selection" that DOES exist (channel level, not point level)
- `thc_reader_t::select_aug_channels_qaware(recv, omega)` (`thc_reader_t.hpp:1256`, called from `prepare_paw_isdf` `:379`). It builds its own `grids::truncated_g_grid rho_g(_MF->ecutrho(), _MF->fft_grid_dim_aug(), recv)`, evaluates eta at every q in `_MF->Qpts()` (with q_cart = -Q), and ranks each lambda row by its **worst-case-over-q Coulomb norm** sum_G 4pi/(Omega|q+G|^2) |eta_lambda(q+G)|^2, collapsing lambda -> (I,J) pair, then keeps pairs above a relative `paw_isdf_tol`. The comment at `:355-375` documents why the previous q=0-only Coulomb ranking was biased (L=0 pairs get the full 1/G^2 enhancement; l_i != l_j pairs vanish as G^L).
- => The augmentation **channel** selection already uses a Coulomb metric and is q-aware; the augmentation **row** set is then fixed and appended. A change of the *global interpolation-point* selection metric only has to touch `thc::chol_metric_impl` / `chol_metric_impl_ibz` (`thc.icc:763`, `:43`); it does not need to add augmentation Gram contributions for correctness of the current pipeline, because the augmentation rows bypass point selection entirely. But note the inconsistency: points are chosen with an L2/overlap metric on smooth orbitals while the eta channels are chosen with a Coulomb metric — a Coulomb-metric point selection would make the two consistent, and `hamilt::paw::isdf_metric` + `make_metric_weights` + `coulomb_weights_on_rho_g_at_q` are the existing primitives to reuse.
