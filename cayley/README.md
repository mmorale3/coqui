# cayley — real-axis spectra from Cayley-transformed self-energy moments

Standalone prototype (python, numpy/scipy/h5py) kept in its own folder of the CoQui tree.
Nothing here is wired into the CoQui build.

Idea (Allen & Booth, arXiv:2609.29271, adapted): the Cayley map u(ω) = (ω−μ+iω_p)/(ω−μ−iω_p) sends the real axis to
the unit circle; the moments C^(n) = ∫ A_Σ(ω) u(ω)^n dω of the correlation self-energy are bounded (‖C^(n)‖ ≤ ‖C^(0)‖),
and a finite set of them has an exact unitary realization C^(n) = R Uⁿ R† (block Toeplitz Gram factorization) that
upfolds Σ_c into a pole representation with real poles d_l = μ + ω_p cot(θ_l/2); the spectral function follows from
the Dyson equation with f + Σ_∞ and that pole set — no analytic continuation, no broadening in the equations.

Modules (`cayley/`):
- `maps.py`      Cayley map and inverse.
- `moments.py`   moments from pole representations (exact for Lehmann/DLR-type data), from a tilted line (lens Cauchy
                 integral), and from the imaginary axis + spectral-gap strip (gap Laplace identity).
- `upfold.py`    block Toeplitz-as-Gram unitary realization with Gram-eigenvalue truncation and terminal-phase fit.
- `spectral.py`  Σ(z) from poles, G and A(ω) from f + Σ_∞ + Σ_c(z).
- `dlr.py`       light DLR (geometric fine grids, pivoted QR) to refit CoQui τ-grid data (copied from the user's
                 real_axis_GW/toys/dlr.py).
- `coqui_io.py`  readers for CoQui `*.mbpt.h5` checkpoints and `*.thc.h5` files.
- `casida_g0w0.py` exact G0W0 Σ_c pole structure from a Casida RPA solution in the THC basis (exact moments, exact
                 reference spectra on small k-meshes).
- `finite_t.py`  finite-temperature oracles (S8b, notes section 11), independent of the line code: finite-T transition sum
                 for Pi(q, zeta), the imaginary-time (Matsubara) route for Pi(q, i nu_n), finite-T Casida W, Eq. fT_sigma from the
                 exact Casida poles (values and Cayley moments), the Matsubara nu_0 term, KS mu_0 ("auto" rule); S8b.3 hybrid:
                 `matsubara_set`, `density_matsubara` (Matsubara Dyson at fixed Sigma_c, subtracted free reference + analytic
                 1/w^4 tail + w^-6 elimination, N(mu) = N_el by bisection), `density_upfold` (exact oracle, small pole lists).
- `metal.py`     S8c helpers (oracles, no line code): finite-T Casida through the Hermitian form of the RPA problem (+ disk
                 cache), Eq. fT_sigma values / Cayley moments in blocked gemm form (+ a binned copy for dense real-axis grids),
                 `upfold_block_fast` (same realization and phase objective as `upfold_block`, no eigensolve per phase), the PH
                 asymmetry max|Pi - Pi^T|/max|Pi|. Metals go/no-go study: `dev/s8c_gonogo.py` (progress entry "S8c.0").
- `line/`        the line scGW prototype (`thc_gw.py` kernels, `timeray.py`, `line_dlr.py` bases, `closure.py`, `driver.py`).
                 Finite T: `LineGW.set_poles(e, v, beta=B, thermal_tol=1e-8, thermal_floor=30)` builds the thermal sector lists
                 (window poles in both sectors with (1-f)/f residues), the guarded rays (s <= beta/sin theta_t) and the node
                 floor; `sigma()` adds the Bose-weighted W(t) terms; `closure.chemical_potential_T / _auto`, `fit_sigma_total`,
                 `wp_thermal`. beta=None (default) runs the T = 0 code unchanged (bitwise). S8b.3 hybrid:
                 `LineGW.sigma_tau_leg` (Sigma_c(i w_n) from tau products with all poles, Filon-GL `timeray.tau_fourier`),
                 `LineSCGW(..., beta=B, scf_density="matsubara", theta_t_frac=...)` (mu, D, N from the Matsubara Dyson
                 equation; the closure supplies only the next poles); refs `scripts/gen_finiteT_hybrid_ref.py`.
                 NOTE (S8b.1): the bosonic fit restricted to the unmasked nodes does NOT determine W below zeta_T, and Sigma_c
                 at every node depends on it (progress entry "S8b.1"); the line Sigma_c matches Eq. fT_sigma to 1e-10 only
                 with W known on all nodes.

Tests: `tests/` (run `python3 -m pytest tests` or the files directly). Scripts: `scripts/` (finite-T references for the C++
tests: `scripts/gen_finiteT_ref.py <lih222|lih223>`; svo222: `scripts/gen_metal_ref.py casida|ref|merge svo222 ...`; finite-T Casida: `scripts/make_casida.py ... --beta B`).
