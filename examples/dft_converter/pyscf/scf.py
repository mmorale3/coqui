#!/usr/bin/env python

'''
Converter for a PySCF mean-field object with periodic boundary condition.

Our converter uses the Python interface for hdf5 from nda (https://github.com/TRIQS/nda), rather than h5py.
Before running the script, make sure to install nda and to add 

"{BeyondDFT_source_dir}/src/mean_field/pyscf/pyscf_interface"
"{nda_build_dir}/deps/h5/python"

in PYTHONPATH. 
'''

from pyscf.pbc import gto, scf, dft
from pyscf.gto.basis import parse_nwchem

import pyscf_interface as pyscf_api

cell = gto.M(
    a = '''0.0,  2.715, 2.715
           2.715, 0.0,  2.715
           2.715, 2.715, 0.0''',
    atom = '''Si  0.      0.      0.
              Si 1.3575 1.3575 1.3575''',
    basis = 'gth-dzvp-molopt-sr',
    pseudo = 'gth-pbe',
    verbose = 7,
)

nk = [2,2,2] 
kpts = cell.make_kpts(nk)

kmf = scf.KRKS(cell, kpts)
kmf.xc = 'pbe'
kmf.kernel()
kmf.analyze()

#
# dump_to_h5() function extracts necessary information form kmf into h5 files which will then be read by BeyondDFT. 
# The outputs include: 
#   a) a h5 file for metadata: "outdir/prefix.h5"
#   b) a folder for wfc on a FFT mesh: "outdir/Orb_fft"
#
if pyscf_api.dump_to_h5(kmf, mo=False, outdir='./', prefix='pyscf') != 0: 
    raise ValueError("pyscf_api.dump() fail!")

# Alternatively, one can convert all the data to MO basis on-the-fly
#if pyscf_api.dump_to_h5(kmf, mo=True, outdir='./', prefix='pyscf') != 0:
#    raise ValueError("pyscf_api.dump() fail!")
