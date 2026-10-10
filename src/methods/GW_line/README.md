# `gw_line`: self-consistent GW on a tilted frequency line

Module `src/methods/GW_line` (driver, kernels, closure) with the numerics in `src/numerics/line_dlr`. It is separate from the
imaginary-axis GW (`src/methods/GW`, `SCF`) and the real-axis GW (`src/methods/GW_real_axis`) and is selected by a
`[gw_line]` block in the CoQui input. Theory, equations and measured accuracies: `notes/line_gw/line_gw_notes.pdf` (source
`line_gw_notes.tex`) of the Cayley_real_axis_scGW project; implementation plan and status: `notes/line_gw_cpp_plan.md`;
log of every run and measurement: `notes/line_gw_progress.md`. The python prototype `coqui/cayley/` is the numerical
oracle of this code (`coqui/cayley/README.md`).

Contents: [1 What it computes](#1-what-gw_line-computes) · [2 Requirements](#2-requirements) ·
[3 Input](#3-input-minimal-example-and-the-complete-gw_line-reference) · [4 Outputs](#4-outputs) ·
[5 Restart and optics-only reruns](#5-restart-and-optics-only-reruns) · [6 Convergence](#6-convergence-guidance) ·
[7 Performance](#7-performance) · [8 GPU](#8-gpu) · [9 Validation](#9-validation) · [10 Limitations](#10-limitations-and-status)

## 1. What `gw_line` computes

Fully self-consistent GW (G and W both updated) at zero temperature for gapped, spin-restricted crystals, with the frequency
dependence handled on a straight line `zeta = mu + |zeta| e^{+-i theta}` through the chemical potential, tilted by
`theta_deg` (default 20 degrees) from the real axis, instead of the Matsubara axis. On the line every dynamical quantity is a
sum of real poles: G is stored as Lehmann poles `(e_m, v_m)` per k, the polarization and the screened interaction W(q) are
fitted at bosonic line nodes to an odd-symmetric real-pole basis, and Sigma^{>/<}(k) is sampled at the dense fermionic nodes of
the line and fitted to real-pole bases of each sector. The convolutions Pi = G G and Sigma = G W are done on complex-time rays
`t = s e^{-i theta/2}` with time nodes chosen by an interpolative decomposition (about 100 to 170 nodes per ray instead of the
~1000 of a generic quadrature).

The new G of every iteration comes from the Cayley closure: the spectral moments of Sigma_c (obtained exactly from its
real-pole fit) are folded into a finite block-Toeplitz (unitary) realization, which yields a Hermitian upfolded Hamiltonian,
hence a Lehmann representation of G with positive weights, the chemical potential from the electron count, and the static
part F = V_H + Sigma_x from the density matrix. No analytic continuation is used anywhere: the real-axis spectral function
A(k, w) and the optical response follow directly from the same pole representations.

After the SCF loop the driver writes A(k, w) on a real-frequency grid and, on request, the real-axis optics of the RPA
polarization built from the self-consistent G: eps^-1_00(q, w), loss function, eps1, eps2, refractive index, absorption,
reflectivity and conductivity for q -> 0 and every q of the mesh, optionally sharpened by one extra Pi -> W pass on a flatter
line (smaller angle). Notes sections: representation and rays (sec:rep, sec:time), GW equations (sec:gw), q -> 0 head and
IBZ (sec:head, sec:ibz), closure (sec:closure), spectra (sec:spectra), optics (sec:optics), validation (sec:valid), cost
(sec:cost).

## 2. Requirements

- Interaction: THC (`[interaction.thc]`, CoQui's ISDF/THC reader); the `[gw_line]` key `interaction` names that block.
  Cholesky interactions abort with "gw_line: requires a THC interaction".
- Mean field: QE (`[mean_field.qe]`, h5 from pw2coqui or xml; the only reader exercised by the tests and runs); the THC and
  the mean field must have the same number of bands (`nbnd`).
- Symmetry: a symmetric mean field (nkpts_ibz < nkpts) runs on the IBZ path (`ibz = true`, the default: poles, Sigma, closure,
  checkpoint at the IBZ k; Pi and W on IBZ union -IBZ); a nosym mean field runs the full-BZ path. `ibz = false` with a symmetric
  mean field aborts. Si 4x4x4 with QE `force_symmorphic`: 13 IBZ k of 64.
- Zero temperature, gapped systems: the KS spectrum must have a gap and the electron count must be even (2 x occupied bands);
  metals and finite temperature are designed (notes sec:finiteT, sec:metals) but not implemented.
- Spin-restricted, collinear (nspin = 1, npol = 1).
- Builds: host (MPI, BLAS/LAPACK, FFTW recommended: the k-mesh FFT convolutions need it) and CUDA (`-DENABLE_CUDA=ON`, see
  section 8).

## 3. Input: minimal example and the complete `[gw_line]` reference

Minimal input (defaults otherwise; complete commented examples in `examples/`):

```toml
[mean_field.qe]
name   = "mf_qe"
prefix = "si"
outdir = "/path/to/qe/out/"
nbnd   = 60
[interaction.thc]
name       = "eri"
mean_field = "mf_qe"
thresh     = 1e-4
save       = "si.thc.h5"        # written on the first run, read afterwards
[gw_line]
interaction = "eri"
prefix      = "si"              # checkpoint ./si.gw_line.h5; prefix (or output) is required by the executable
div_treatment = "gygi"          # CoQui's q -> 0 treatment, used by the production runs (default "ignore_g0")
```

Run: `mpirun -np <ranks> <build>/bin/coqui --filenames input.toml` with `OMP_NUM_THREADS=1` and one rank per core (host);
GPU runs see section 8.

| example | what it shows |
|---|---|
| `examples/lih222_smoke.toml` | LiH 2x2x2 unit-test fixture, 2 iterations, small spectra + optics (paths relative to `tests/unit_test_files`; the `[gw_line][examples]` test runs it) |
| `examples/si444_scgw.toml` | production scGW, Si 4x4x4, 60 bands, IBZ, gygi head, mixing 1.0 + damped tail, conv_thr 3e-5, memory keys |
| `examples/si444_g0w0.toml` | one-shot G0W0@KS (niter 1 from the KS start); what `start = "qp_diag"` does instead |
| `examples/si444_optics.toml` | optics-only rerun from the SCF checkpoint (restart, niter 0) with 10 and 5 degree lines |
| `examples/restart.toml` | continuing / extending a run |
| `examples/gpu.toml` | device run: budgets, closure on the GPU, launch line |

TOML notes: nested tables can be written inline on ONE line (`spectra = { eta = [0.004], nw = 601 }`) or as sub-tables
(`[gw_line.optics]` after the `[gw_line]` keys); multi-line inline tables are not valid TOML. Integer-typed keys (niter, K,
nw, ...) must be written as integers. String values are case-insensitive where the table says so (they are lower-cased).

### Complete key reference

Generated from `gw_line_params_t::from_ptree` (`driver.cpp`), `optics_params_t::from_ptree` (`optics.hpp`) and the
defaults of `driver.hpp`, `spectra.hpp`, `optics.hpp`, `scf_mixing.hpp`; consistency checked by
`coqui/cayley/scripts/check_gw_line_readme.py` (every key read by the code is in this table and vice versa). Energies in
Hartree. "auto" defaults are computed at run time as described.

<!-- keys:begin -->
**Line and real-pole bases**

| key | type | default | meaning | when to change |
|---|---|---|---|---|
| `interaction` | string | (required) | name of the THC `[interaction]` block (read by `main.cpp`) | always set |
| `theta_deg` | double | `20.0` | line angle (deg), in (0, 90); time rays at theta_t_frac x theta | convergence studies only (15-25 deg measured) |
| `eps` | double | `1e-10` | tolerance of every real-pole basis (fermionic, bosonic) and default of `time_eps` | 1e-8 for quick tests (10x worse moments) |
| `lam` | double | `6.0` | fermionic pole range (Ha) of the Sigma / G bases | if QP poles beyond 6 Ha matter (logged as dropped weight) |
| `lam_b` | double | auto: 2 `g_emax` (lehmann) / 2 `lam` (compressed) when <= 0 | bosonic pole range (Ha); must cover Pi's spectrum (sum of particle and hole pole energies) | leave auto; a narrower range made the W residues 1e5x ill-conditioned |
| `sigma_gap` | double | `0.02` | gap of the Sigma bases on each side of mu (Ha); < 0: auto 0.8 (QP edge + bos gap) | small-gap systems (< 0.04 Ha QP gap): use < 0 |
| `bos_gap` | double | `0.02` | gap of the bosonic basis (Ha); < 0: auto 0.5 x QP gap | as `sigma_gap` |
| `g_gap` | double | `0.0` | gap of the G compression bases, [0, lam) (`g_repr = "compressed"` only) | keep 0 |
| `g_repr` | string | `"lehmann"` | G between iterations: `"lehmann"` (factorized v v^dagger Lehmann poles of the closure, pruned) or `"compressed"` (per-sector gapless refit, S6 path) | `"compressed"` only for python parity |
| `g_emax` | double | auto: = lam (any value < 0; code -1) | lehmann pruning: drop poles with abs(e) > g_emax (moment-truncation artefacts, logged as "dropped") | with `lam` |
| `g_wtol` | double | `1e-12` | lehmann pruning: drop poles of weight abs(v)^2 < g_wtol | rarely |
| `g_emin_frac` | double | `0.5` | lehmann pruning: drop in-gap poles with abs(e) < g_emin_frac x QP half gap AND weight < `g_wsmall` (0 = off) | rarely (they set the ray length) |
| `g_wsmall` | double | `1e-4` | weight threshold of the in-gap pruning | rarely |
| `nodes_per_ray` | long | `120` | dense fermionic nodes per ray (log grid) where Sigma is sampled | convergence studies |
| `node_tmin` | double | `1e-3` | smallest abs(zeta - mu) of the fermionic nodes (Ha) | rarely |
| `node_tmax` | double | `60.0` | largest abs(zeta - mu) of the fermionic nodes; also tmax of the fermionic bases | rarely |

**Time grids of the ray products**

| key | type | default | meaning | when to change |
|---|---|---|---|---|
| `time_grid` | string | `"id"` | `"id"`: time-node ID grids rebuilt every iteration from the current poles (~100-170 nodes); `"gl"`: generic Gauss-Legendre rays (~1000 nodes, the python reference) | `"gl"` only for reference checks |
| `time_eps` | double | = `eps` | ID tolerance | rarely |
| `time_pad` | double | `1.25` | ID energy-range margin [Emin / pad, pad Emax] | rarely |
| `time_oversample` | double | `1.0` | ID node oversampling (>= 1) | rarely |
| `time_snap` | double | `0.0` | > 0: ID energy ranges snapped to a geometric grid of this many points per octave (grids stable under roundoff changes of the poles) | diagnostics |
| `ray_decades` | double | `36.0` | GL rays only: ray length e^{-decades} | `time_grid = "gl"` only |
| `t_chunk` | long | `0` (auto) | time nodes per chunk of the ray products; 0: host 32 (env `COQUI_GWLINE_HOST_TCHUNK`), device from the free memory | memory pressure (smaller) |

**Closure (Cayley moments, upfolding, Lehmann G)**

| key | type | default | meaning | when to change |
|---|---|---|---|---|
| `wp` | double | `0.11` | Cayley scale omega_p (Ha); resolution near mu ~ (pi / K) [(w - mu)^2 + wp^2] / wp | convergence studies |
| `K` | long | `24` | number of moments | 16 / 32 in convergence studies; 8 for tests |
| `tol_gram` | double | `1e-10` | Gram eigenvalue cut of the moment problem | rarely |
| `tol_gram_eps` | double | `1.0` | the cut used is max(tol_gram, tol_gram_eps x eps) (moments carry eps noise); 0 = off | keep |
| `nphi` | long | `8` | phases of the terminal-block scan | rarely |
| `tol_svd` | double | `1e-12` | relative SVD cutoff of the upfolding | rarely |
| `closure_cut` | string | `"hard"` | Gram cut placement `"hard"` / `"gap"` (largest eigenvalue ratio in a window) / `"smooth"` (smooth-step weights) | studies of the closure noise |
| `closure_svd_cut` | string | `"hard"` | SVD cut `"hard"` / `"gap"` | as above |
| `closure_cut_window` | double | `10.0` | window factor of the gap / smooth cuts (>= 1) | as above |
| `phase_keep` | double | `0.0` | > 0: keep the previous phase basin within this factor (phase continuity) | as above |
| `closure_threads` | long | `-1` (auto) | BLAS threads of the per-k host closure: -1 = the rank's cores in device runs (SLURM_CPUS_PER_TASK, else affinity / node ranks), untouched in host runs; 0 = untouched; n > 0 = n | device runs with unusual binding |
| `closure_k_workers` | long | `-1` (auto) | concurrent host threads over the rank's k (>= 1 or -1): auto = closure_threads / 2 in device runs, 1 in host runs | rarely |
| `closure_svd` | string | `"gesdd"` | SVD driver of the upfolding: `"gesdd"` or `"gesvd"` (python / pre-S7g) | A/B only |
| `closure_ueig` | string | `"cayley"` | eigenvectors of the unitary U: `"cayley"` (Hermitian Cayley image + Rayleigh-Ritz, Schur fallback) or `"schur"` | A/B only |
| `closure_device` | string | `"auto"` | cuSOLVER Gram / Cayley / Lehmann eigensolvers and SVD: `"auto"` (device runs) / `"on"` / `"off"`; each call falls back to host LAPACK on a device failure | `"off"` to A/B |
| `closure_dev_svd` | string | `"gesvdp"` | device SVD `"gesvdp"` (polar) or `"gesvd"` | A/B only |

**SCF: iterations, mixing, start, multilevel schedule**

| key | type | default | meaning | when to change |
|---|---|---|---|---|
| `niter` | long | `12` | TOTAL number of iterations (a restart continues up to niter; 0 = only spectra / optics) | per run |
| `mixing` | double | `1.0` | linear mixing of Sigma^{>/<} at the nodes, (0, 1] (F is not mixed) | rarely (0.5 halves the error per iteration only) |
| `conv_thr` | double | `3e-5` | stop when max abs(dSigma) at the nodes after mixing < conv_thr (never in an iteration without a previous Sigma); with DIIS also mixing x residual < conv_thr | 5e-5 on Si 4x4x4 (floor ~1e-4 of the residual) |
| `mixing_alg` | string | `"linear"` | `"linear"` or `"diis"` (Anderson/Pulay on Sigma at the nodes of all k) | DIIS was worse than linear 1.0 on Si |
| `diis_hist` | long | `6` | DIIS history length (>= 1) | with DIIS |
| `diis_start` | long | `2` | first iteration that extrapolates (earlier: linear `mixing`) | with DIIS |
| `diis_beta` | double | `1.0` | DIIS step x = sum c (x_i + beta r_i) | with DIIS |
| `diis_reg` | double | `1e-10` | Tikhonov regularization of B (relative) | with DIIS |
| `diis_cmax` | double | `10.0` | reset to a linear step when max abs(c) exceeds it | with DIIS |
| `diis_grow` | double | `10.0` | reset when the new residual > grow x the best | with DIIS |
| `diis_mix_F` | bool | `false` | include F in the DIIS vector (the next closure uses the extrapolated F) | with DIIS |
| `diis_wF` | double | `-1.0` | weight of the F elements (< 0: the number of fermionic nodes) | with DIIS |
| `damp_below` | double | `3e-4` | damped tail: once the residual max abs(Sigma[G] - Sigma_in) < damp_below, linear steps with `damp_mixing` (sticky, also across restarts); 0 = off | keep (undamped steps hop at the closure noise) |
| `damp_mixing` | double | `0.5` | mixing of the damped tail, (0, 1] | rarely |
| `start` | string | `"ks"` | initial G: `"ks"` (KS poles, mu = KS mid-gap) / `"qp_diag"` (one extra Pi -> W -> Sigma pass on the KS poles + diagonal G0W0 QP equation; KS vectors with the QP energies) / `"qp_file"` (QP energies read from `start_file`) | warm starts save <= 1 iteration on Si |
| `start_file` | string | `""` | h5 file of `start = "qp_file"` (e.g. a CoQui Matsubara `mbpt.h5`) | with qp_file |
| `start_dataset` | string | `""` | dataset of absolute QP energies (Ha, band order); empty: `scf/iter<final_iter>/qp_approx/E_ska`, else `E_ska` / `qp_energies` at the root | with qp_file |
| `start_eta` | double | `1e-3` | broadening (Ha) of the diagonal QP equation of `qp_diag` | rarely |
| `coarse.niter` | long | `0` | multilevel schedule: iterations 1..coarse.niter at coarse settings, then production (niter >= coarse.niter + 2; incompatible with `bases_file`) | not recommended (net gain ~3% on Si) |
| `coarse.eps` | double | `1e-8` | coarse-level basis tolerance | with coarse |
| `coarse.K` | long | `16` | coarse-level moments | with coarse |
| `coarse.nodes_per_ray` | long | `80` | coarse-level fermionic nodes per ray | with coarse |
| `coarse.time_eps` | double | = `coarse.eps` | coarse-level ID tolerance | with coarse |

**q -> 0 divergence and head of W**

| key | type | default | meaning | when to change |
|---|---|---|---|---|
| `div_treatment` | string | `"ignore_g0"` | Sigma_c head term: `"ignore_g0"` (no G = 0 term) or a CoQui gygi variant: `"gygi"`, `"gygi_order_N"`, `"gygi_perdir"`, `"gygi_2d"`, `"gygi_smallest_q"`, `"gygi_average"` (extrapolation of eps^-1_00(q) - 1 to q -> 0); `gygi_extrplt` and metal variants are rejected; nqpts = 1 forces ignore_g0 for Sigma_c | `"gygi"` for production (matches CoQui's scGW) |
| `hf_div_treatment` | string | follows div_treatment: "ignore_g0" if that is ignore_g0, else "gygi" | exchange Madelung term (`"ignore_g0"` or `"gygi"`) | rarely |
| `head_extrapolation` | string | `"gygi"` | q -> 0 variant of the head data (checkpoint, eps_inf) when div_treatment = ignore_g0 (else div_treatment itself) | rarely |

**Spectra (A(k, w) after the loop)**

| key | type | default | meaning | when to change |
|---|---|---|---|---|
| `spectra.enable` | bool | `true` | write `spectra/` | false for optics-only reruns |
| `spectra.eta` | list of double | [0.004, 0.01] | Lorentzian broadenings (Ha) | per plot |
| `spectra.wmin` | double | `-0.45` | grid start, w - mu (Ha) | per plot |
| `spectra.wmax` | double | `0.45` | grid end, w - mu (Ha) | per plot |
| `spectra.nw` | long | `601` | grid points | per plot |

**Optics (after the loop; `optics = { ... }` or `[gw_line.optics]`)**

| key | type | default | meaning | when to change |
|---|---|---|---|---|
| `optics.enable` | bool | true if the optics table is present, else false | run the optics | set |
| `optics.wmin` | double | `0.0` | excitation-energy grid start (Ha, >= 0) | per plot |
| `optics.wmax` | double | `1.5` | grid end (Ha) | per plot |
| `optics.nw` | long | `1501` | grid points | per plot |
| `optics.eta` | list of double | [0.01] | constant broadenings (Ha) | per plot |
| `optics.eta_rel` | list of double | [0.05] | energy-proportional broadenings eta = c w | per plot |
| `optics.scales` | "auto" or list | "auto" | scales (Ha, increasing) of the multi-scale response closure | rarely |
| `optics.nscales` | long | `4` | number of automatic scales | rarely |
| `optics.K` | "auto" or long | "auto" (K rule) | moments of the response closure | rarely |
| `optics.q0` | bool | `true` | q -> 0 (optical limit) | |
| `optics.finite_q` | bool | `true` | every mesh q != Gamma (loss, plasmon dispersion) | false to save time |
| `optics.theta_deg` | double or list | none | flatter final line(s) (deg): one extra Pi -> W -> head pass each, own bosonic basis, time rays at theta / 2 | 10 and 5 deg for sharp spectra |
| `optics.q0_variants` | list of string | ["gygi_perdir", "gygi_smallest_q", "gygi_average"] | other q -> 0 extrapolations evaluated for q0 (sensitivity groups `q0_<variant>`) | rarely |
| `optics.nnls_n` | long | `1500` | NNLS grid of the cross-check fit (error bars) | rarely |
| `optics.nline` | long | auto (when < 0): 1200 above 7.5 deg, 3000 below | candidate grid of the flatter line's bosonic basis | below 5 deg |
| `optics.npole` | long | auto (when < 0): 800 above 7.5 deg, 2400 below | pole grid of the flatter line's bosonic basis | below 5 deg |
| `optics.time_grid` | string | `"id"` | time grid of the flatter passes | rarely |
| `optics.mem_gb` | double | `2.0` | host budget per rank (GB) of a flatter pass's Pi group | memory pressure |
| `optics.delta_floor` | double | `1e-10` | floor of the fit residual used by the K rule | rarely |
| `optics.poles` | string | `"final"` | G of every optics line: `"final"` (final poles) or `"initial"` (iteration 0: KS or the qp start), e.g. RPA@PBE next to G0W0 | RPA@KS comparisons |

**Memory, distribution, symmetry**

| key | type | default | meaning | when to change |
|---|---|---|---|---|
| `ibz` | bool | `true` | symmetry reduction of a symmetric mean field (env `COQUI_GWLINE_IBZ` overrides) | false only with a nosym mean field (A/B) |
| `mem_budget_gb` | double | `0.0` | host GB per rank for the q plan (0 = `mem_frac` x MemAvailable / cgroup limit per rank) | shared nodes |
| `dev_mem_budget_gb` | double | `0.0` | device GB per rank (0 = `mem_frac` x the free device memory) | other libraries on the GPU |
| `mem_frac` | double | `0.8` | fraction of the available memory the q plan may fill, (0, 1] | OOM: lower |
| `q_group_size` | long | `0` | > 0 fixes the largest q-group size of the Pi -> W stage (env `COQUI_GWLINE_QGROUP` wins) | tests / A/B |
| `sigma_kdist` | bool | `true` | Sigma at the nodes k-distributed over the ranks (owner k mod np); false = replicated | keep |

**Checkpoint, output, restart**

| key | type | default | meaning | when to change |
|---|---|---|---|---|
| `restart` | bool | `false` | resume from `<output>.gw_line.h5:/scf_line/final_iter` (bitwise continuation); a missing file starts from the KS poles | restarts / optics reruns |
| `checkpoint_sigma` | string | `"last"` | Sigma at the nodes: `"last"` = only the last iteration's, in `<output>.gw_line.sigma.h5` (rewritten every iteration); `"all"` = every iteration's in `iter<N>/Sigma_{p,h}` | `"all"` for analysis of small systems |
| `outdir`, `prefix` | string | "./", required | checkpoint stem `<outdir>/<prefix>` (the executable requires `prefix` unless `output` is given) | per run |
| `output` | string | outdir + "/" + prefix | explicit checkpoint stem (wins over outdir / prefix); a direct C++ call without any of them uses "./gw_line" | scripted runs |

**Finite temperature (S8b; notes sec:finiteT; an iteration is thermal iff a pole lies within E_T = ln(1/thermal_tol)/beta of mu)**

| key | type | default | meaning | when to change |
|---|---|---|---|---|
| `theta_t_frac` | double | `0.5` | ray angle theta_t = theta_t_frac theta (T = 0 too) | flatter rays lower the finite-T node floor (rho = sin(theta - theta_t)/sin(theta_t)) |
| `beta` | double | `0.0` | inverse temperature (1/Ha); 0 = T = 0. Thermal iterations need `g_repr = "lehmann"`, `start = "ks"`, no `coarse`, no optics | set for finite T |
| `thermal_tol` | double | `1e-8` | thermal tolerance tau_T: window E_T = ln(1/tau_T)/beta; far poles keep weight 1 (error <= tau_T) | 1e-10..1e-12 for tight comparisons |
| `thermal_floor` | double | `30.0` | c_zeta: bosonic line nodes with rho beta abs(zeta) < c_zeta are excluded; wedge band bottom | keep |
| `thermal_floor_f` | double | = `thermal_floor` | c_f: fermionic node floor of the thermal closure fit (c_zeta + ln(1/tau_T) = the strict option) | keep |
| `wp_floor` | double | `15.0` | omega_p := max(`wp`, wp_floor zeta_T), zeta_T = c_zeta/(rho beta), in thermal iterations | keep (E3 study) |
| `mu_rule` | string | `"auto"` | `"auto"` (gap midpoint iff abs(dN) <= mu_dn_max and abs(dN) > mu_th_factor n_th, else N(mu) = N_el) / `"gap"` / `"number"` | metals: auto gives "number" |
| `mu_dn_max` | double | `0.1` | rule auto: largest thermal count mismatch at the gap midpoint read as a closure weight error | keep |
| `mu_th_factor` | double | `10.0` | rule auto: abs(dN) must exceed this times the thermal carriers n_th(mu_g) | keep |
| `band_heights` | long | `8` | heights of the wedge band of the bosonic data set D | keep |
| `band_x` | long | `21` | points per band height (odd: symmetric under z -> -conj z) | keep |
| `band_top` | double | `4.0` | band top / x extent = band_top zeta_T sin(theta) | keep |
| `mats_factor` | double | `4.0` | Matsubara points i nu_n of D, n = 1..ceil(mats_factor zeta_T beta / 2 pi) | keep |
| `bos_eps_T` | double | `1e-12` | tolerance of the D-selected bosonic basis (pivoted QR on D) | keep |
| `bos_line_eps` | double | `1e-10` | eps of the gapless bosonic line basis whose unmasked nodes enter D | keep |
| `cut_odd` | double | `1e-13` | relative SVD cutoff of the odd sector of the split pair fit | keep |
| `cut_even` | double | `1e-10` | relative SVD cutoff of the even sector | keep |
| `tau_grid` | string | `"gl"` | tau-leg nodes on [0, beta/2]: `"gl"` (composite GL, ~200-250) / `"id"` (finite-interval ID, ~30-40) | `"id"` for cost |
| `tau_eps` | double | `1e-12` | tolerance of the tau ID | keep |
| `spectra.occupation` | bool | `false` | also write f(w - mu) on the spectra grid (spectra/fermi_w; f A = the occupied spectrum) | finite-T plots |
| `thermal_bases_file` | string | `""` | parity: D (D_zeta_re/_im, D_kind), nu_b and the two-sided Sigma basis (sigma_basis_w) read from this h5 (the python finite-T SCF reference) | parity tests |

**Diagnostics**

| key | type | default | meaning | when to change |
|---|---|---|---|---|
| `debug_noise_h0` | double | `0.0` | relative Hermitian Gaussian noise on H0 (noise-floor meter) | diagnostics |
| `debug_noise_seed` | long | `0` | seed of the noise | diagnostics |
| `debug_noise_sigma` | double | `0.0` | relative noise on Sigma at the nodes (before mixing) ... | diagnostics |
| `debug_noise_iter` | long | `1` | ... in this iteration | diagnostics |
| `bases_file` | string | `""` | real-pole bases read from this h5 file instead of built (keys sigma_{particle,hole}_w, g_{particle,hole}_w, bos_nu, bos_zeta_nodes): portable python parity | parity tests |
<!-- keys:end -->

## 4. Outputs

All files are written by rank 0. Complex arrays are stored with a trailing dimension 2 (real, imaginary) (nda h5 layout).
Energies in Ha; `omega` of `spectra/` is relative to mu, that of `optics/` is the excitation energy.

`<output>.gw_line.h5` (written after every iteration):

```
system/      nkpts, nkpts_ibz, nbnd, Np, nelec, mu0 (KS mid-gap), H0 (nk, nb, nb), eigval (nk, nb), kpoints (nk, 3),
             qk_to_k2 (nq, nk), kp_to_ibz, kp_trev
input/       the parameters of the run (theta_deg, eps, lam, lam_b, K, wp, mixing, div_treatment, ...) and
             fermionic_nodes (the dense nodes, mu-relative; checked on restart)
scf_line/    final_iter, start_done
  iter0/     the initial state (KS or the qp start): mu, mu_sigma, dmu, e_homo, e_lumo, F, F_closure, has_sigma, poles/
  iter<N>/   N >= 1, after iteration N:
    mu, mu_sigma (centre at which Sigma was sampled), dmu, e_homo, e_lumo (QP edges, mu-relative)
    F            V_H + Sigma_x[D_N] (nk, nb, nb)
    F_closure    the F of the closure that built these poles (= F[D_{N-1}]; the spectra use it)
    poles/       {particle,hole}_counts (nk), {particle,hole}_e (sum M), {particle,hole}_v (sum M, nb): Lehmann poles
                 (mu-relative) and vectors per k, concatenated over k (g_repr = "compressed": {s}_coef (sum M, nb, nb))
    history/     iter, dSigma, resid, residF, mu, dmu, gap, e_homo, e_lumo, nelec, nelec_lehmann, N_mu, dropped_weight,
                 heldout_max, npoles_min/max, ng_min/max, g_emin, pruned_*, bos_gap, sigma_gap_p/h, time, time_grid,
                 nt_pi_p/h, nt_sigma_p/h, g_repr, mix, ndiis, level
    head/        eps^-1_00(q, zeta) - 1 of W: h_nodes (nq, nz_b), h_res / h_res_hole (nq, r_b) residues at +nu / -nu,
                 h0_nodes, h0_res, h0_res_hole (q -> 0 extrapolation with q_weights (nq)), zeta (nz_b), nu (r_b),
                 qpts (nq, 3), madelung, eps_inf = 1 / (1 + Re h0(0)), extrapolation, div_treatment, hf_div_treatment
    closure_phi  terminal phases of the closure per k
    Sigma_p, Sigma_h   only with checkpoint_sigma = "all": (nk, n_nodes, nb, nb)
spectra/     mu, omega (nw, w - mu), eta (n_eta), A_k_w_diag (n_eta, nk, nw, nb) = diagonal of A in the KS band basis,
             A_k_w_trace (n_eta, nk, nw), e_homo, e_lumo (mu-relative), vbm, cbm (absolute), gap
optics/      omega (nw), omega_eV, eta, eta_rel, wp2_valence (4 pi N / Omega), nelec, volume (bohr^3); attributes units,
             method, physics_scope, broadening_order
  theta<deg>/   one group per line, e.g. theta20.0 (the SCF angle, first) and theta10.0: theta_deg, source (which G /
                head), time_grid, pass_time, nt_pi_p/h, zeta, nu, h_nodes (nq, nz), q_weights, qpts, lattv
    q0/          q -> 0 with the run's extrapolation; q0_<variant>/ the other q0_variants; iq<n>/ mesh q n != Gamma
                 curves (n_broad, nw), rows = the eta values first, then eta_rel: eps1, eps2, n, kappa, alpha (1/bohr),
                 alpha_cm (1/cm), R, loss, sigma1, sigma2 (a.u.), each with *_err (abs(MB - NNLS) of the same channel);
                 eps1_R1, eps2_R1, mismatch_eps, mismatch_loss (the two closure routes); scalars eps_inf, eps_inf_m,
                 eps_inf_nnls_h/m, fsum_h/m, fsum_nnls_h/m, fsum_mb_h/m, delta_h/m, delta_complex_fit_h/m, K_h/m,
                 tol_gram_h/m, wneg_h/m, nnls_resid_h/m, h_asymmetry, time, iq; scales_h/m; h_nodes; poles/ (fit and
                 NNLS poles)
```

The k index of `scf_line/` and `spectra/` is the IBZ k on the IBZ path (`system/kp_to_ibz` maps every k of the mesh to it).

`<output>.gw_line.sigma.h5` (checkpoint_sigma = "last", rewritten every iteration): `iter`, `Sigma_p`, `Sigma_h`
(nk, n_nodes, nb, nb): the mixed Sigma^{>/<} at the fermionic nodes of the last iteration; needed by a restart.

What to plot: the trace of A(k, w) per k (band structure as a spectral map, QP peaks and satellites); `spectra/gap` and
`history/gap` per iteration (SCF convergence); `optics/theta<deg>/q0/eps2` and `loss` with their `_err` bands; the finite-q
loss for the plasmon dispersion.

```python
import h5py, numpy as np, matplotlib.pyplot as plt
HA = 27.211386
with h5py.File("si444.gw_line.h5", "r") as f:
    fi = int(f["scf_line/final_iter"][()])
    print("iterations", fi, "QP gap (eV)", f[f"scf_line/iter{fi}/history/gap"][()] * HA)   # h5 values in Ha
    w, eta, A = f["spectra/omega"][:], f["spectra/eta"][:], f["spectra/A_k_w_trace"][:]   # A: (n_eta, nk, nw)
    plt.figure(); plt.plot(w * HA, A[0, 0] / HA); plt.xlabel("w - mu (eV)"); plt.ylabel("Tr A(k=0, w) (1/eV)")
    th = sorted(k for k in f["optics"] if k.startswith("theta"))        # e.g. ['theta10.0', 'theta20.0', 'theta5.0']
    wo = f["optics/omega_eV"][:]
    plt.figure()
    for t in th:
        e2, err = f[f"optics/{t}/q0/eps2"][0], f[f"optics/{t}/q0/eps2_err"][0]   # row 0 = first constant eta
        plt.plot(wo, e2, label=t); plt.fill_between(wo, e2 - err, e2 + err, alpha=0.3)
    plt.legend(); plt.title("eps2, q -> 0"); plt.xlabel("omega (eV)")
    # a complex dataset: last dimension (re, im)
    F = f[f"scf_line/iter{fi}/F"][...]; F = F[..., 0] + 1j * F[..., 1]
plt.show()
```

## 5. Restart and optics-only reruns

- `restart = true` resumes from `scf_line/final_iter` of `<output>.gw_line.h5` (+ the Sigma file) and runs until `niter`
  iterations are done in total; the continuation is bitwise identical to an uninterrupted run with the same binary and
  ranks. Keep the line / basis / closure keys of the original run (the fermionic nodes are compared with `input/`). The DIIS
  history is not checkpointed (rebuilt: the first step after a restart is x + beta r); the damped-tail state is.
  A qp_diag start interrupted before its start pass completed redoes it (`scf_line/start_done`).
- Nothing left to iterate (`restart = true` and `niter` <= the iterations done, e.g. `niter = 0`): the driver goes straight
  to the spectra and optics, i.e. an "optics only" rerun (`examples/si444_optics.toml`). The SCF-angle optics line uses the
  head of the last checkpointed iteration if the run had converged, otherwise it is recomputed from the final poles (one
  Pi -> W pass; logged in `optics/theta<deg>/source`); every `optics.theta_deg` adds one pass on a flatter line.
- `niter = 0` without restart: the spectra / optics of the initial G (KS, or the qp start).
- `optics.poles = "initial"`: all optics lines from the iteration-0 poles (RPA@KS next to the GW optics of the same run).
- Reruns replace the `spectra/` and `optics/theta<deg>/` groups. Every `optics/theta<deg>/` group carries its own grid
  (`omega`, `omega_eV`, `eta`, `eta_rel`); the common `optics/omega`, `omega_eV`, `eta`, `eta_rel` are rewritten on every pass
  and describe the newest one. After changing `optics.wmin / wmax / nw / eta / eta_rel`, read the grid of the theta group.

## 6. Convergence guidance

- Defaults (perf 7.2): linear mixing 1.0, then 0.5 once the residual max abs(Sigma[G] - Sigma_in) < 3e-4 (damped tail),
  `conv_thr = 3e-5` on dSigma (= residual 6e-5 in the damped tail). The SCF map contracts by 0.15-0.25 per step for Si and
  LiH, so undamped steps converge in a few iterations: iterations to residual 2e-4 (old linear 0.5 -> new defaults) Si 2x2x2
  gygi 21 -> 7, Si 4x4x4 gygi 16 -> 8 (gap stable to 0.3 meV from iteration 8).
- The floor: undamped steps carry the closure's discrete decisions (Gram cut, phase basin) into the next input; the residual
  floors at 2e-5..2e-4 (Si 4x4x4: ~1e-4, use `conv_thr = 5e-5` there or stop by `niter` ~ 10). Near the floor the gap is
  reproducible to +-0.5-1 meV (two closure basins on Si 2x2x2 gygi: 4.0025 vs 4.0016-4.0019 eV); continuing at the floor can
  hop between them. This is the closure noise (S7f: a continuous ~1e5 amplification of roundoff in Sigma, no decision flips
  at the production settings), not the mixing.
- DIIS (`mixing_alg = "diis"`) was WORSE than linear 1.0 on Si (overshoots of the gap by 3 meV, stalls at 5e-4..2e-3: the
  residual is noise-dominated below ~1e-3); kept for systems with a slowly contracting map (untested). Warm starts
  (`qp_diag`, `qp_file`) save <= 1 iteration; the multilevel schedule (`coarse`) ~3% net.
- Run-to-run floor: the same binary on two nodes differs by ~3e-10 Ha in mu at iteration 3; host CPU types (AMD vs Intel MKL
  code paths) by up to 2e-6 Ha (closure amplification); ranks / GPUs on the same CPU type <= 4e-11 Ha.
- Convergence in the method parameters (S8a, notes sec:s8a; Si 4x4x4 IBZ, every setting converged from scratch): the defaults
  (theta_deg 20, eps 1e-10, K 24, wp 0.11) are within 0.4 meV of the tightest setting (K 40, eps 1e-12) on the gap, 1.2 meV on
  the band edges at G / X / L and the direct gap, 3 meV on levels 3 eV from the gap, 0.003% on eps_inf; the spread over K 16-40,
  eps 1e-8..1e-12, theta 20-25, wp 0.08-0.15 is <= 3.4 meV and 0.04%. K = 32 (+8% per iteration) brings the deeper levels below
  1 meV; theta_deg = 25 or eps = 1e-8 are ~24% cheaper per iteration at <= 1 meV. NOT resolved: the occupied band width (band
  bottom 13 eV from the Cayley centre; +-0.45 eV with the settings) and the loss-function maximum (set by the optics closure; use
  its error bar). Do not run the SCF below theta_deg = 20: 17.5 and 15 deg end in a limit cycle (dSigma 1e-3..8e-3); flatter
  lines are for the one-pass optics only. The k mesh is not part of this study.
- Spectra resolution near mu: delta_w ~ (pi / K) [(w - mu)^2 + wp^2] / wp; errors outside the window stay at the 10% level
  (notes sec:spectra). For optics, flatter lines (`optics.theta_deg = [10, 5]`) sharpen the spectra at one Pi -> W pass each.

## 7. Performance

Cost per iteration: Pi and Sigma (ray products, ~N_t N_k N_q Np^2), W (Dyson and residue fit per q and bosonic node,
N_zeta N_q Np^3), closure (per k, (K nb)^3 plus the phase scan). Measured, Si 4x4x4 IBZ (13 of 64 k, Np 741, 60 bands),
iteration 3 of a 3-iteration benchmark (div_treatment ignore_g0); an SCF needs ~8 iterations to a stable gap:

| machine | ranks | s / iteration | breakdown (s) | vs CoQui Matsubara iteration | commit |
|---|---|---|---|---|---|
| rome, 1 node | 128 | 56.5 | Pi 12.1, W 19.2, Sigma 20.2, closure 3.9 | 1.68x (33.6 s) | 7aec58a |
| genoa, 4 nodes | 384 | 15.0 | Pi 3.5, W 3.9, Sigma 5.2, closure 2.1 | 0.45x | 7aec58a |
| 2 x A100 80 GB | 2 | 18.8 | | 0.56x | perf 7.4b |
| 1 x A100 80 GB | 1 | 46.4 | | 1.38x | perf 7.4b |

Larger meshes: Si 6x6x6 IBZ 201 s for iteration 3 on 1 rome node (perf 7.5c); Si 8x8x8 G0W0 on 4 rome nodes 607 s / iteration
(7.4b, before the FFT transforms; did not fit before the q plan).

Knobs, in order of relevance:
- IBZ (`ibz`, default on with a symmetric mean field): poles, Sigma and the closure at the IBZ k, Pi / W on IBZ union -IBZ,
  Sigma by q-class sums; Si 4x4x4: 2.3x on the kernels, 15x on the closure.
- q groups (q_plan.hpp): the Pi -> W stage holds Pi(q, zeta) for a group of q; the largest pair-closed group whose memory
  model fits `mem_frac` of the free memory per rank (host MemAvailable / cgroup; device free memory) is chosen, with relief
  levels (Sigma's real-space residues in place of w, then the host t_chunk halved twice) when nothing fits; the log line
  "q plan: ..." prints the choice and the model. Overrides: `mem_budget_gb`, `dev_mem_budget_gb`, `q_group_size`, env
  `COQUI_GWLINE_QGROUP`, `COQUI_GWLINE_QPLAN_LEVEL`. If even the smallest group does not fit, the run aborts with the number
  of ranks that would fit.
- `t_chunk`: time nodes per chunk (host 32; device from the free memory). Larger chunks = larger gemms, more memory.
- Time grids: `time_grid = "id"` (default) uses ~100-170 nodes per ray (one shared particle grid); `"gl"` ~1000.
- Closure: host runs keep one rank per core (`OMP_NUM_THREADS=1`); ranks beyond N_k lend their cores to the k owners
  (socket-local contiguous core blocks, one core per BLAS thread: env `COQUI_GWLINE_CLOSURE_PLACE` / `_PIN`); the k with a
  free terminal block are scanned in parallel over ranks. Device runs: `closure_threads` = the rank's cores,
  `closure_k_workers` = threads / 2 concurrent k, dense eigensolvers on the GPU (`closure_device`).
- W node path (host runs): the Dyson matrices are exchanged through per-rank `/dev/shm` segments mapped by the whole node
  (kept between iterations), instead of MPI all-to-alls: rome 1 node W redistribute 9 -> 0.5 s. The segments are NOT in the
  q-plan memory model: ~21 GB per node for Si 4x4x4 IBZ, ~60 GB full BZ; the run falls back to the MPI redistribute if
  `/dev/shm` cannot hold them; `COQUI_GWLINE_W_REDIST=old` forces the old path. Device runs use the MPI redistribute.
- k-mesh convolutions: Pi and Sigma are convolutions over the k mesh done in real space; the transforms are 3-D FFTs of the
  mesh (FFTW on the host for N_k >= 8, cuFFT on the device for N_k >= 100, else gemms; env `COQUI_GWLINE_KFT`): Si 4x4x4 rome
  transforms 16.8 -> 6.2 s, 6x6x6 131 -> 20 s.
- Multi-node: blocks of the 2-D (P, Q) process grid shrink as 1 / ranks; the closure's k owners are currently all on node 0
  in multi-node runs (open item).

<!-- env:begin -->
Environment variables, user-facing (read at run time; unset = default):

| variable | default | effect |
|---|---|---|
| `COQUI_GWLINE_IBZ` | (key `ibz`) | 0 / 1 overrides the `ibz` key |
| `COQUI_GWLINE_KFT` | auto | k-mesh transforms: `gemm` (dense gemms), `fft` (3-D FFTs; error if the build has no FFT library), unset / `auto` (FFT from N_k >= `COQUI_GWLINE_KFT_NMIN`) |
| `COQUI_GWLINE_QGROUP` | 0 | > 0 fixes the largest q-group size (wins over `q_group_size`) |
| `COQUI_GWLINE_QPLAN_LEVEL` | -1 (auto) | forces a relief level of the q plan (0 = none, 1 = w^(R) in place / host t_chunk / 2, ...) |
| `COQUI_GWLINE_HOST_TCHUNK` | 32 | host t_chunk when the key `t_chunk` = 0 |
| `COQUI_GWLINE_CLOSURE_PIN` | core | BLAS team placement on the lent cores: `core` (thread t on core t), `mask` (perf 7.1c; caused a regression), `none` |
| `COQUI_GWLINE_CLOSURE_PLACE` | block | core blocks of the k owners: `block` (socket-local contiguous) or `interleave` |
| `COQUI_GWLINE_W_REDIST` | node | W exchange: `node` (/dev/shm node path, host runs) or `old` (MPI redistribute) |
| `COQUI_GWLINE_MEMTRACE` | 0 | 1: log VmRSS / VmHWM per rank and the node MemAvailable at points of the iteration |
| `COQUI_GWLINE_FTPROF` | 0 | 1: log the IBZ class-sum time per class (Sigma, level 1) |

Developer / A-B switches (defaults = the production paths; for benchmarks and bisection):

| variable | default | effect |
|---|---|---|
| `COQUI_GWLINE_BOS_NODES` | 1.5 | bosonic node factor (n1 = ceil(f r / 2) ray-1 nodes); <= 0: the python nodes |
| `COQUI_GWLINE_CLOSURE_BORROW` | 1 | 0: ranks beyond N_k do not lend cores to the closure |
| `COQUI_GWLINE_CLOSURE_KTABLE` | 0 | > 0: closure profile line for every k |
| `COQUI_GWLINE_CLOSURE_SCAN` | parallel | terminal-phase scan `parallel` (over ranks, eigensolve-free held-out error) or `serial` (pre-7.1c) |
| `COQUI_GWLINE_CLOSURE_WEIGHTS` | 0 | > 0: core blocks weighted by the previous closure's per-k cost |
| `COQUI_GWLINE_SCAN_MB` | 512 | broadcast budget per rank (MB) of the parallel scan |
| `COQUI_GWLINE_SCAN_THREADS` | 16 | cores per busy rank in the scan (0 = no cap) |
| `COQUI_GWLINE_UEIG_CUT` | inf | first cut of the Cayley U-eigen path: `inf` or `mu` |
| `COQUI_GWLINE_UEIG_ACCEPT` | 10 | residual acceptance factor of the Cayley U-eigen path |
| `COQUI_GWLINE_CONTRACT_RIGHT` | auto (nP < nQ) | Sigma orbital contraction order: 1 right factor first, 0 left |
| `COQUI_GWLINE_DYSON_BATCHED` | auto (Np <= 1024) | device Dyson: 1 batched LU (cuBLAS), 0 per-matrix cuSOLVER |
| `COQUI_GWLINE_DYSON_NBAT` | 256 | largest device Dyson sub-batch |
| `COQUI_GWLINE_FFT_CB` | 16384 / N_k in [16, 1024] | host FFT column block |
| `COQUI_GWLINE_FFT_DEV_CB` | 2^24 / N_k | device cuFFT column block |
| `COQUI_GWLINE_FFT_PLAN` | measure | `estimate`: FFTW_ESTIMATE plans |
| `COQUI_GWLINE_KFT_NMIN` | 8 host / 100 device | N_k from which `auto` uses FFTs |
| `COQUI_GWLINE_FUSED` | 1 | device fused Hadamard kernels; 0: cuTENSOR elementwise (bring-up path) |
| `COQUI_GWLINE_FUSED_VARIANT` | 0 | device slab-kernel variant |
| `COQUI_GWLINE_GT_CACHE` | -1 (auto) | cache of the transformed G~ shared by Pi and the Sigma hole leg: 0 off, 1 on, auto = on when <= 1.5 GB per rank (host) / 15% of the free device memory |
| `COQUI_GWLINE_GT_XV` | -1 (auto) | host G~ form: 1 XV (L ph R^dagger), 0 C(t), auto = the cheaper by flop count |
| `COQUI_GWLINE_IBZ_CB` | 512 | host column block of the IBZ class sums |
| `COQUI_GWLINE_IBZ_CSHARE` | 1 | 0: C(t) of G~ not shared per star (IBZ) |
| `COQUI_GWLINE_IBZ_CSHARE_MB` | 5% of the free device memory | device budget of the shared C(t) |
| `COQUI_GWLINE_PI_MIRROR` | 1 | 0: explicit hole leg of Pi (no mirror relation) |
| `COQUI_GWLINE_PI_QFOLD` | 1 | device Pi: one fused launch over all q (0: per q) |
| `COQUI_GWLINE_RSPACE` | 1 | 0: k-space Hadamards instead of real-space convolutions |
| `COQUI_GWLINE_SIGMA_KOUTER` | 1 | device Sigma: W(q, chunk) of all q by one strided gemm (0: per q) |
| `COQUI_GWLINE_SIGMA_QGROUP` | 0 (all q) | Sigma q-group size when the W residues live on the host |
| `COQUI_GWLINE_TGRID_SHARED` | 1 | 0: four separate ID grids instead of one shared particle grid |
| `COQUI_GWLINE_W_HOST` | -1 (plan) | device runs: 1 / 0 forces the W residues on the host / device (host builds: only 1 has an effect) |
| `COQUI_GWLINE_W_KEEP` | 1 | 0: the /dev/shm W node buffer is released after every call |
| `COQUI_GWLINE_W_KEEP_FRAC` | 0.1 | keep the node buffer only while it is <= this fraction of MemAvailable |
| `COQUI_GWLINE_W_MIRROR` | 1 | 0: W without the mirror (ray-1) reduction |
| `COQUI_GWLINE_W_NODE_SIZE` | 0 | > 0: virtual nodes of this many ranks in the W node path (tests) |
| `COQUI_GWLINE_W_PREFAULT` | none | `owner`: segments pre-faulted by their owners |
| `COQUI_GWLINE_W_PROFILE` | 0 | 1: barrier-profiled MPI redistribute path |
| `COQUI_GWLINE_W_SHM` | posix | node buffer backend: `posix` (per-rank shm_open) or `mpi` (MPI_Win_allocate_shared) |
| `COQUI_GWLINE_W_XCHUNK_MB` | 16 | cross-node message chunk (MB) of the W node path |
| `COQUI_GWLINE_W_ZSUB` | auto | forces the number of bosonic nodes per W sub-step |
| `COQUI_GWLINE_WR_CB` | 2^22 | element budget (divided by N_k) of the column blocks of the real-space residue transform |
| `COQUI_GWLINE_WR_INPLACE` | -1 (plan) | Sigma's real-space residues in place of w: 1 force, 0 off, -1 relief level of the q plan |

Test-only variables (unit tests and benchmarks; `GW_LINE_TEST_KEEP=1` keeps the test checkpoints):

| variable | default | effect |
|---|---|---|
| `COQUI_GWLINE_BENCH_DIR` | project si_kp222_nbnd60 data | [.bench] input directory |
| `COQUI_GWLINE_BENCH_PREFIX` | si | [.bench] prefix |
| `COQUI_GWLINE_BENCH_NP` | 640 | [.bench] THC nIpts |
| `COQUI_GWLINE_BENCH_TCHUNK` | 0 | [.bench] t_chunk |
| `COQUI_GWLINE_BENCH_HOST` | 0 | 1: [.bench] in host memory |
| `COQUI_GWLINE_BENCH_NCOLS` | 32 x 4300 | [.kft_bench] columns |
| `COQUI_GWLINE_DEV_TCHUNK` | 0 | [device] tests: device t_chunk |
| `COQUI_GWLINE_IBZ_MF` | | [.ibz_tables]: extra mean field "outdir|prefix" |
<!-- env:end -->

## 8. GPU

- Build CoQui with `-DENABLE_CUDA=ON`; the default compute space of such a build is the device (`coqui --compute cpu` forces
  the host path). One MPI rank per GPU with several cores per rank (the closure uses them): e.g. `srun -n 2 -c 16
  --cpu-bind=none` with `CUDA_VISIBLE_DEVICES` = rank mod GPUs per node (`examples/gpu.toml`).
- On the device: G~ and the Pi / Sigma Hadamard products (fused CUDA kernels in `cuda/`, cuBLAS gemms), the k-mesh transforms
  (gemms; cuFFT from N_k >= 100), W's Dyson solves (batched LU for Np <= 1024) and residue fits, the closure's Gram / Cayley /
  Lehmann eigenproblems and SVD (cuSOLVER, `closure_device = "auto"`, host LAPACK fallback per call). On the host: the
  real-pole fits, moments, upfolding bookkeeping, mixing, checkpoint. The /dev/shm W node path is host-only.
- Memory: `dev_mem_budget_gb` (0 = `mem_frac` x the free device memory at the plan) and `mem_budget_gb` (host, where the
  driver arrays and, if needed, the W residues live) drive the q plan; `t_chunk = 0` sizes the time chunk from the free
  device memory. The q-plan log line reports the device and host models.
- Measured: Si 4x4x4 IBZ 46.4 s / iteration on 1 A100, 18.8 s on 2 A100 (80 GB); host vs device A/B tests agree to <= 1.3e-13
  (kernels) and 5e-11 Ha (driver, iteration 2).
- Known CoQui bug (outside this module): `transform_k2g` (`numerics/device_kernels/cuda/symmetry_tools.cu`) did not zero its
  device error flag, so building a symmetric mean field in a CUDA build could abort at random. Fixed on the
  `cayley-line-scgw` branch (373a07b, one `cudaMemset`); not yet in CoQui's main branch.

## 9. Validation

Test executables (Catch2, `build/tests/bin`, MPI-aware; tags in brackets):

| executable | tags | what is checked, against which oracle |
|---|---|---|
| `test_line_dlr` | `[numerics][line_dlr]`, `[time_id]` | real-pole bases, fits, ray transforms, time-node ID vs python references (`tests/unit_test_files/gw_line/*.h5`) |
| `test_cayley` | `[numerics][cayley]`, `[scan]` | moments, upfolding, Lehmann G, chemical potential vs exact finite models / python |
| `test_response_closure` | `[response_closure]` | optics closure (MB, NNLS, K rule) vs python and exact models |
| `test_gw_line_kernels` | `[gw_line]`: `[V0]` `[V1]` `[V2]` `[V3]` `[V6]` `[head]` `[optics]` `[flat]` `[time_id]` `[lehmann]` `[s7e]` `[perf71]` `[w]` `[kft]` `[device]` | V0: V_H + Sigma_x vs CoQui hf_t; V1: Pi on the line vs the Casida transition sum; V2: W vs Dyson / fit of gathered matrices, CoQui's W^c(q, i nu) and Casida; V3: Sigma_c vs the exact Casida Sigma_c and CoQui's iteration-1 Sigma_c(i w_n); V6: the Coulomb head and the optics vs Casida (exact) and CoQui; A/B of every performance path (bitwise or <= 1e-12); host vs device |
| `test_gw_line_scf` | `[gw_line][scf]`: `[closure]` `[restart]` `[parity]` `[id_vs_gl]` `[gygi]` `[optics]` `[mixing]` `[qp_start]` `[multilevel]` `[qplan]` ...; `[gw_line][examples]` | closure on exact toy models; restart bitwise; python driver parity (gates 10x the measured noise floor); ID vs GL grids (5x floor); the examples of this README (parse + lih222 end-to-end + h5 layout) |
| `test_gw_line_ibz` | `[gw_line][ibz]` | IBZ path vs the full-BZ path (<= 1e-12 with trivial tables; symmetric vs nosym fixtures) |

Hidden cases (`[.name]`) are diagnostics and benchmarks. Fixtures: `qe_lih222`, `qe_si211`, `qe_lih223`, the `_sym` variants
(tests/unit_test_files/qe), stored THC `tests/unit_test_files/gw_line/lih222_thc/`.

Run on a laptop (macOS: `KMP_DUPLICATE_LIB_OK=TRUE`), <= 2 ranks:

```
cd coqui/build && make -j4 test_line_dlr test_cayley test_response_closure test_gw_line_kernels test_gw_line_scf test_gw_line_ibz
cd tests/bin && export OMP_NUM_THREADS=1
mpirun -np 2 ./test_gw_line_kernels "[gw_line]" && mpirun -np 2 ./test_gw_line_scf "[gw_line]" && mpirun -np 2 ./test_gw_line_ibz "[gw_line]"
./test_line_dlr && ./test_cayley && ./test_response_closure
./test_gw_line_scf "[examples]"          # the examples of this README, ~1.5 min
```

On a cluster: the same binaries inside an allocation (srun / mpirun; never on a login node); device builds additionally run
the `[device]` cases. Suites at 7aec58a, 1 and 2 ranks (Mac) and 2 ranks (rusty CPU, MKL): kernels 37/37, scf 18/18, ibz 5/5,
cayley 6/6; on 2 A100: kernels 40/40, ibz 5/5, scf 18/18, cayley 6/6.

Physics validation (notes sec:valid and the progress log; CoQui data on the same THC): Pi on the line vs Casida 2e-13, W 5e-14,
Sigma_c 2e-7 at the nodes (Si 2x2x2, python); flatter-line head kernels vs Casida <= 2.1e-11 at 10 / 5 deg (gate 1e-9);
converged C++ line scGW vs CoQui's converged Matsubara scGW at the fixed point (V4, Sigma_c(i w_n) relative): Si 2x2x2 gygi
2.5e-4 (2.5e-5 Ha), Si 4x4x4 gygi 6.9e-4 (8.2e-5 Ha), electron count 7.997 / 7.996, eps_inf within 3e-4 relative. QP gaps
(Si 2x2x2 / 4x4x4 gygi: 4.0025 / 2.1326 eV) are within 0.6 / 0.5 meV of Pade continuations of the Matsubara runs (Pade is a
consistency check only, not a reference).

## 10. Limitations and status

- Gapped systems (even electron count, KS gap); finite temperature is implemented (S8b, `beta`; notes sec:finiteT): thermal
  sector lists, guarded rays, the bosonic data set D with the tau leg, the split pair fit, Bose-weighted W(t), the total-Sigma
  closure and the mu rule; the near-mu resolution of thermal spectra is ~60/(rho beta) Ha (notes sec:fT_res). Metals (notes
  sec:metals, plan S8c) are designed, not implemented. The time-ray angle is theta_t_frac x theta (default 1/2).
- Spin-restricted collinear; THC interactions only.
- Optics: RPA polarization of the self-consistent G (no vertex corrections, no excitons; absorption onset = direct QP gap);
  q -> 0 from the extrapolation of the finite-q heads (variant dependent; the q0_<variant> groups quantify it).
- Reproducibility of converged gaps +-0.5-1 meV (closure basins, section 6); parameter convergence (S8a): section 6; the SCF does
  not converge below theta_deg = 20 (open).
- Open performance items: the W node path's /dev/shm segments are not in the q-plan memory model and stay allocated through
  Sigma; closure k owners all on node 0 in multi-node runs; the W node path is host-only; the device FFT threshold is modelled,
  not measured.
