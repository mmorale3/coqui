#!/usr/bin/env python

import os
from pyscf.pbc import gto, scf, dft
from pyscf.gto.basis import parse_nwchem

import pyscf_interface as pyscf_api


basname = 'sto-3g'

basis =  {
'H': basname}

print(basis)

cell = gto.M(
    a = '''
  3.76700   0.00000   0.00000
 -1.88350   3.26232   0.00000
 -0.00000  -0.00000   6.13600''',
    atom = '''
H      -0.00020      2.17500      1.53400
H       1.88350      1.08770      4.60200''',
    basis = basis,
    verbose = 7
)

nk = [2,2,2]  # 2 k-poins for each axis
kpts = cell.make_kpts(nk)

#kmf = scf.KRKS(cell, kpts)
kmf = scf.KRHF(cell, kpts)
#kmf.xc = 'pbe'
kmf.kernel()
kmf.analyze()

if pyscf_api.dump_to_h5(kmf, True, './', 'pyscf') != 0: 
    raise ValueError("pyscf_api.dump() fail!")

