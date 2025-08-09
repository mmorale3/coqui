#!/usr/bin/env python

import os 
import numpy
from pyscf import lib
from pyscf.pbc import gto, scf, df
from pyscf.pbc.df import ft_ao
import pyscf.lib.chkfile as chk
import h5._h5py as h5

import pyscf_interface as pyscf_api

cell = gto.M(
    a = '''0.0,  2.715, 2.715
           2.715, 0.0,  2.715
           2.715, 2.715, 0.0''',
    atom = '''Si  0.      0.      0.
              Si 1.3575 1.3575 1.3575''',
    basis = 'gth-szv', 
    pseudo = 'gth-pade',
    verbose = 7,
    precision = 1e-12,
    mesh = numpy.array([36,36,36])
)

mp_mesh = [2,2,2]  # 2 k-poins for each axis, 2^3=8 kpts in total
kpts = cell.make_kpts(mp_mesh)

#mydf   = df.RSGDF(cell, kpts)
#mydf.auxbasis = df.aug_etb(cell, beta=1.5)
#mydf.auxbasis = 'def2-svp-ri'
#mydf._cderi_to_save = "cderi.h5"
#mydf.build()

kmf = scf.KRHF(cell, kpts)
#kmf.with_df = mydf
kmf.kernel()
kmf.analyze()

if pyscf_api.dump_to_h5(kmf, mp_mesh, False, './', 'pyscf') != 0:
    raise ValueError("pyscf_api.dump() fail!")

dm = kmf.make_rdm1()
vj, vk = kmf.get_jk(cell, dm)
vk *= -0.5
nkpts, nbnd = vj.shape[:2]
vj = vj.reshape((1, nkpts, nbnd, nbnd))
vk = vk.reshape((1, nkpts, nbnd, nbnd))

f = h5.File("pyscf.h5", 'a')
g = h5.Group(f)
scf_g = g.open_group("SCF")
h5.h5_write(scf_g, 'J', vj)
h5.h5_write(scf_g, 'K', vk)

