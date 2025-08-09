#!/usr/bin/env python

import os
from pyscf.pbc import gto, scf, dft
from pyscf.pbc import df, tools

from pyscf.gto.basis import parse_nwchem

import pyscf_interface as pyscf_api



cell = gto.M(
    a = '''
  3.76700   0.00000   0.00000
 -1.88350   3.26232   0.00000
 -0.00000  -0.00000   6.13600
    ''',
    atom = '''
Li      0.000   0.0000      0.0000
           ''',
    basis = 'sto-3g', 
    verbose = 7,
    spin = 8
)

nk = [2,2,2] 
kpts = cell.make_kpts(nk)

mydf   = df.GDF(cell)
mydf.auxbasis = df.aug_etb(cell, beta=2.0)
mydf.kpts = kpts

kmf = scf.KUHF(cell, kpts)
kmf.with_df = mydf
kmf.kernel()

kmf.analyze()

if pyscf_api.dump_to_h5(kmf, mo=False, outdir='./', prefix='pyscf') != 0: 
    raise ValueError("pyscf_api.dump() fail!")



