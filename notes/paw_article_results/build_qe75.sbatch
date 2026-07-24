#!/bin/bash
#SBATCH -J qe75_build
#SBATCH -p ccq
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -c 1
#SBATCH -t 02:00:00
#SBATCH -o /mnt/home/mmorales/Devel/QEF/q-e-7.5/build/CPU/build.%j.out
#SBATCH -e /mnt/home/mmorales/Devel/QEF/q-e-7.5/build/CPU/build.%j.err
set -e
source /etc/profile.d/modules.sh
# mirror ~/Devel/QEF/q-e_7.0/build/CPU/env.bash
module purge
module load modules
module load cmake gcc openmpi hdf5 boost intel-oneapi-mkl python3 lib/fftw3
export PATH=/mnt/home/mmorales/libexec/git-core/:${PATH}
export BOOST_ROOT=${BOOST_BASE}
export BLAS_ROOT=${MKLROOT}
export LAPACK_ROOT=${MKLROOT}

cd /mnt/home/mmorales/Devel/QEF/q-e-7.5/build/CPU
echo "=== cmake configure $(date) ==="
cmake \
  -DCMAKE_Fortran_COMPILER=mpif90 \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_INSTALL_PREFIX=./ \
  -DQE_ENABLE_HDF5=ON \
  -DCMAKE_BUILD_TYPE=Release \
  ../..
echo "=== make pw $(date) ==="
make -j 32 pw || make -j 32
echo "=== build done $(date) ==="
ls -la bin/pw.x
