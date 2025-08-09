#!/usr/bin/env python

import numpy as np
from pyscf import gto, df, scf, dft
import pyscf_gdf_interface as bdft
import pyscf_interface as bdft_mf

mol = gto.M(atom='O 0 0 0; H  0 1 0; H 0 0 1', basis='ccpvdz', verbose=7)

mydf = df.DF(mol)
mydf._cderi_to_save = "cderi.h5"
mydf.build()
bdft.mol_gdf_dump_to_h5(mydf, mo=False, outdir="gdf_eri")

mf = scf.RHF(mol).density_fit()
mf.with_df = mydf
mf.kernel()
mf.analyze()
bdft_mf.mol_dump_to_h5(mf, becke_grid_level=5, mo=False, outdir='./', prefix='pyscf')
