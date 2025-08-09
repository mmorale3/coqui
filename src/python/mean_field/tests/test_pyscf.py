from mpi4py import MPI
import pytest

import coqui
from coqui.utils.tests.test_coqui_env import mpi
from coqui._lib.utils_module import utest_filename


def construct_pyscf_mf(mpi_context, system_id: str):
    outdir, prefix = utest_filename(system_id)
    mf_params = {
        "prefix": prefix,
        "outdir": outdir
    }
    qe_mf = coqui.make_mf(mpi_context, mf_params, "pyscf")
    assert isinstance(qe_mf, coqui._lib.mf_module.Mf)
    if mpi_context.root():
        print(qe_mf)


def test_pyscf_mf(mpi):
    construct_pyscf_mf(mpi, "pyscf_si222")
    construct_pyscf_mf(mpi, "pyscf_h2o_mol")