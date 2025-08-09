CoQui: Correlated Quantum Interface
-----------------------------------------------
**Last Updated:** January 30, 2025

CoQui is a scientific software project designed for 
***Co**rrelated **Qu**antum **I**nterface* that goes beyond the scope 
of density functional theory (DFT). Starting with a single-particle basis 
set and a mean-field solution, typically DFT or Hartree-Fock, CoQui employs 
tensor hypercontraction (THC) decomposition to efficiently process two-electron 
operators. This sophisticated approach enables CoQui to achieve exceptionally 
low-scaling algorithms in subsequent many-body calculations. 

## What does CoQui do?
CoQui utilizes distributed linear algebra to enable high-performance
*ab-initio* calculations applicable to:
- Both k-point (periodic) and molecular systems
- Generic single-particle basis sets, such as Kohn-Sham (KS) orbitals,
  Gaussian-type orbitals, and their mixtures.

Currently, `CoQui` interfaces with the following backends 
(see [examples/dft_converter](examples/dft_converter) for input preparation): 
- [Quantum ESPRESSO](https://www.quantum-espresso.org)
- [PySCF](https://pyscf.org)

Below are some key features of `CoQui`. For more detailed examples, 
please visit our [examples](examples/README.md) page.
#### Compressed Representation for Many-Body Hamiltonians
- THC representation for two-electron Coulomb integrals 
  [[ref1](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.3c00615),
  [ref2](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.4c00085)]. 
- Cholesky decomposition for two-electron Coulomb integrals. 

#### Many-Body Perturbation Theory
- Hartree-Fock [[ref](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.3c00615)]
- RPA correlation energy [[ref](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.3c00615)]
- GW approximation [[ref](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.4c00085)]
- Second-order exchange (SOX) diagram [[ref](https://arxiv.org/abs/2404.17744)]
- Self-consistency with quasiparticle approximation 
- Self-consistency with full frequency dependence

#### Downfolding for effective low-energy Hamiltonians
- Constrained RPA to calculate screened interactions [[ref](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.4c00085)]
- Local effective low-energy Hamiltonian for further correlated calculations
  [[ref](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.4c00085)]

## Getting started with `CoQui` 
### Prerequisites
- C++ compiler that supports at least C++20.
- CMake >= 3.2.0
- MPI Library: openmpi >= 4 or mpich >= ?. 
- HDF5 >= 1.8.2 for checkpoint file I/O. 
- BLAS Library: OpenBLAS or Intel MKL. 
- LAPACK Library: OpenBLAS or Intel MKL. 
- [SLATE](https://github.com/icl-utk-edu/slate/tree/master) Library for distributed linear algebra.  
- Boost >= 1.68.0
- FFTW >= 3.2

### Installation
`CoQui` uses `CMake` to configure the build process. Please follow 
the instructions below step-by-step, and replace the placeholders in 
square brackets (`[]`) with your local settings.

```shell
# Step 1: Clone the git repository of CoQui
git clone https://github.com/mmorale3/BeyondDFT.git coqui.src

# Step 2: Create working directory for CMake to build in
mkdir -p coqui.build && cd coqui.build

# Step 3: Use cmake to configure the coqui build process
# Replace `[YOUR_INSTALL_PREFIX]` with the directory where you want CoQui installed.
# Replace `[NCORES]` with the number of cores you want to use for the test processes.
# Replace `[SLATE_INSTALL_PATH]` with your SLATE installation path. 
export slate_ROOT=[SLATE_INSTALL_PATH]
cmake \
        -DCMAKE_INSTALL_PREFIX=[YOUR_INSTALL_PREFIX] \
        -DCTEST_NPROC=[NCORES] \
        ../coqui.src

# Step 4: Build, test and install coqui
# Replace `[NCORES_MAKE] with the number of cores you want to use for the build processes. 
# The ctests will be executed in parallel using `[NCORES]` processors.
make -j[NCORES_MAKE] && make test && make install

# Step 5: Verify the installation
# Check if the executable `coqui` was successfully installed in `[YOUR_INSTALL_PREFIX]/bin`.

# For convenience, you can add `[YOUR_INSTALL_PREFIX]/bin` to your `PATH` environment variable: 
export PATH="[YOUR_INSTALL_PREFIX]/bin:$PATH"
```

### Tutorials and Examples 
For guidance on preparing inputs for `CoQui`, please refer to the 
[Examples](examples/README.md) section. 

## Citation
If you use `CoQui` in your research, please consider supporting our developers 
by citing the following papers:

[1] C.-N. Yeh, M. Morales, Low-Scaling Algorithm for the Random Phase
Approximation Using Tensor Hypercontraction with k-point Sampling,
[J. Chem. Theory Comput. 19, 18, 6197–6207 (2023)](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.3c00615).

[2] C.-N. Yeh, M. Morales, Low-Scaling Algorithms for GW and Constrained Random Phase
Approximation Using Symmetry-Adapted Interpolative Separable Density Fitting,
[J. Chem. Theory Comput. 20, 8, 3184–3198 (2024)](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.4c00085). 
