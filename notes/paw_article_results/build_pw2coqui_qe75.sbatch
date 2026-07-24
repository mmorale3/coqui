#!/bin/bash
#SBATCH -J pw2coqui75
#SBATCH -p ccq
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -c 1
#SBATCH -t 01:00:00
#SBATCH -o /mnt/home/mmorales/Devel/QEF/q-e-7.5/build/CPU/pw2coqui_build.%j.out
#SBATCH -e /mnt/home/mmorales/Devel/QEF/q-e-7.5/build/CPU/pw2coqui_build.%j.err
set -e
source /etc/profile.d/modules.sh
module purge
module load modules
module load cmake gcc openmpi hdf5 boost intel-oneapi-mkl python3 lib/fftw3
export PATH=/mnt/home/mmorales/libexec/git-core/:${PATH}
export BOOST_ROOT=${BOOST_BASE}
export BLAS_ROOT=${MKLROOT}
export LAPACK_ROOT=${MKLROOT}

QE=/mnt/home/mmorales/Devel/QEF/q-e-7.5
COQUI_SRC=/mnt/home/mmorales/ceph/CoQui/Si_eos_rpa_hf/coqui/qe_converter/pw2coqui.f90

echo "=== install canonical pw2coqui.f90 (md5 $(md5sum $COQUI_SRC | cut -d" " -f1)) ==="
cp $COQUI_SRC $QE/PP/src/pw2coqui.f90

echo "=== add CMake target (idempotent) ==="
if ! grep -q qe_pp_pw2coqui_exe $QE/PP/CMakeLists.txt; then
cat >> $QE/PP/CMakeLists.txt <<'EOF'

###########################################################
# pw2coqui.x  (added for CoQui)
###########################################################
set(src_pw2coqui_x src/pw2coqui.f90)
qe_add_executable(qe_pp_pw2coqui_exe ${src_pw2coqui_x})
set_target_properties(qe_pp_pw2coqui_exe PROPERTIES OUTPUT_NAME pw2coqui.x)
target_link_libraries(qe_pp_pw2coqui_exe
    PRIVATE
        qe_pw
        qe_modules
        qe_pp
        qe_upflib
        qe_fftx
        qe_mpi_fortran
        qe_xclib)
EOF
echo "  appended pw2coqui target"
else
echo "  target already present"
fi

cd $QE/build/CPU
echo "=== reconfigure $(date) ==="
cmake -DCMAKE_Fortran_COMPILER=mpif90 -DCMAKE_C_COMPILER=mpicc \
      -DCMAKE_INSTALL_PREFIX=./ -DQE_ENABLE_HDF5=ON -DCMAKE_BUILD_TYPE=Release ../..
echo "=== build pw2coqui $(date) ==="
make -j 32 qe_pp_pw2coqui_exe
echo "=== done $(date) ==="
ls -la bin/pw2coqui.x
