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

Tests: `tests/` (run `python3 -m pytest tests` or the files directly). Scripts: `scripts/`.
