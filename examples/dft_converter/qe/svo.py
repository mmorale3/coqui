#!/usr/bin/python
# pyright: reportUnusedExpression=false

import os

from ase import io
from ase.dft.kpoints import bandpath

from py_w90_driver.kpts_utils import gen_kpts_list
from py_w90_driver.qe import run_pwscf, run_pw2coqui, setup_nscf_nosym, copy_scf, read_kpoints
from py_w90_driver.wan90 import run_w90, wan2h5
from py_w90_driver.coqui import run_unfold_wfc


"""Common Parameters"""
mpi_exe = "mpirun"       # mpi executable
n_cores = 12             # customized number of processors for mpi, otherwise all processors will be used.
mpi_nk = 3               # number of k-pools in parallelization in QE
qe_bin_dir = None
coqui_bin_dir = None
run_wan = True

seedname = 'svo'
outdir = './out'
basename = os.path.basename(outdir.rstrip("/"))
parent_dir = os.path.dirname(outdir.rstrip("/"))
outdir = os.path.join(parent_dir, basename)

""" Atomic Structure read from cif """
atoms = io.read('./inputs/svo.cif', format='cif')
special_points = {
        'R': [0.50, 0.50, 0.50],
        'M': [0.50, 0.50, 0.00],
        'X': [0.00, 0.50, 0.00],
        'G': [0.00, 0.00, 0.00],
}
symm_kpath = bandpath(path=['R', 'G', 'G', 'X', 'X', 'M', 'M', 'G'], special_points=special_points, cell=atoms.cell)
mp_scf = (14, 14, 14)
mp_nscf = (7, 7, 7)
nbnd_coqui = 40
nbnd_wan = nbnd_coqui

""" Quantum Espresso Processes """

pseudopotentials = {'Sr': 'Sr_ONCV_PBE-1.2.upf',
                    'V': 'V.ccECP.upf',
                    'O': 'O.ccECP.upf'}
path_to_pseudopotentials = './pseudo'

# parameters written to espresso.pwi
input_data = {
    'outdir': outdir,
    'prefix': seedname,
    'verbosity': 'high',
    'system': {
        'ibrav': 0,
        'nat': 5,
        'ntyp': 3,
        'ecutwfc': 90.0,
        'occupations': 'smearing',
        'smearing': 'm-p',
        'degauss': 0.0015,
        'input_dft':  'pbe'
    },
    'electrons': {'conv_thr': 1e-10, 'mixing_beta': 0.7, 'diagonalization': 'david'},
}

# scf
qe_params = {'atoms': atoms,
             'pseudopotentials': pseudopotentials,
             'path_pseudo': path_to_pseudopotentials,
             'input_data': input_data,
             'mp_grid': mp_scf}
run_pwscf(ase_atoms=qe_params['atoms'], input_data=qe_params['input_data'],
          pseudo_dir=qe_params['path_pseudo'],
          pseudopotentials=qe_params['pseudopotentials'],
          kpts=qe_params['mp_grid'],
          mpi_exe=mpi_exe, n_cores=n_cores, mpi_nk=mpi_nk, bin_dir=qe_bin_dir)

# nscf w/ symmetries
input_data.update({'calculation': 'nscf'})
input_data['system']['nbnd'] = nbnd_coqui
input_data['system']['force_symmorphic'] = True
qe_params['mp_grid'] = mp_nscf
run_pwscf(ase_atoms=qe_params['atoms'], input_data=qe_params['input_data'],
          pseudo_dir=qe_params['path_pseudo'],
          pseudopotentials=qe_params['pseudopotentials'],
          kpts=qe_params['mp_grid'],
          mpi_exe=mpi_exe, n_cores=n_cores, mpi_nk=mpi_nk, bin_dir=qe_bin_dir)

# pw2coqui
# set n_cores=1 explicitly since pw2coqui does not support mpi yet
run_pw2coqui(qe_params['input_data']['prefix'], qe_params['input_data']['outdir'],
             mpi_exe=mpi_exe, n_cores=1, bin_dir=qe_bin_dir)

""" Wannier90 Process """

# preparing input for wan90
outdir_nosym = os.path.join(outdir, "nosym_tmp")
copy_scf(qe_params, outdir_nosym)
kpts_list = gen_kpts_list(mp_nscf)
qe_params_nosym = setup_nscf_nosym(qe_params, outdir_nosym, mp_grid=mp_nscf, kpoints_nscf=kpts_list,
                                   nbnd=nbnd_wan, conv_thr=1e-2)
run_pwscf(ase_atoms=qe_params_nosym['atoms'], input_data=qe_params_nosym['input_data'],
          pseudo_dir=qe_params_nosym['path_pseudo'],
          pseudopotentials=qe_params_nosym['pseudopotentials'],
          kpts=qe_params_nosym['kpoints'],
          mpi_exe=mpi_exe, n_cores=n_cores, mpi_nk=mpi_nk, bin_dir=qe_bin_dir)
# unfold wfc
run_unfold_wfc(seedname, outdir, seedname, outdir_nosym, 
               mpi_exe=mpi_exe, n_cores=n_cores, bin_dir=coqui_bin_dir)
kpoints, k_weights = read_kpoints(os.path.join(outdir_nosym, f"{seedname}.xml"))

# Wannier90 
projections = [{'site': 'V', 'ang_mtm': 'dxz,dyz,dxy', 'xaxis': (1, 0, 0)}]
exclude_bands = [list(range(1, 21)), list(range(24, nbnd_wan+1))]
pp_params = {'num_wann': 3, 
             'projections': projections,
             'exclude_bands': exclude_bands,
             'kpoint_path': symm_kpath,
             'num_bands': 3,
             'bands_plot': True,                      # default = false
             'bands_num_points': 100,                 # default = 100
             'bands_plot_format': 'gnuplot',          # default = "gnuplot"
             'write_hr': True,                        # default = false
             'write_u_matrices': True,                # default = false
             'write_xyz': True,
             'translate_home_cell': False,
             'dis_win_min': 2.0,               # default = lowest eigenvalue in the system
             'dis_win_max': 18.0,              # default = highest eigenvalue in the given states
             'dis_froz_min': 2.0,              # default = if dis_froz_max is given, then default is dis_win_min
             'dis_froz_max': 18.0,             # no default
             'dis_num_iter': 10000,            # default = 200
             'dis_conv_tol': 1.0e-14,          # default = 1.0E-10
             'dis_mix_ratio': 0.5,             # default = 0.5
             'num_iter': 10000, 
             'conv_window': 5, 
             'conv_tol': 1e-12
             }

overlap_params = {'write_mmn': True,
                  'write_amn': True,
                  'write_spn': False,
                  'write_unk': False}

win_context = {"pp_params": pp_params, "overlap_params": overlap_params}

run_w90(dft_context=qe_params_nosym, win_context=win_context, 
        job_wan_pp=True, job_dft2wan=True, job_w90=True, 
        mpi_exe=mpi_exe, n_cores=1, w90_bin_dir=qe_bin_dir, dft2wan_bin_dir=qe_bin_dir)

""" TRIQS Wannier90 Converter """
imp_context = {'num_imps': 1, 'occ': 1, 'imp0': [0, 0, 2, 3, 0, 0]}
wan2h5(dft_context=qe_params_nosym, imp_context=imp_context, rot_mat_type='hloc_diag', bloch_basis=True, w90zero=1e-5)

