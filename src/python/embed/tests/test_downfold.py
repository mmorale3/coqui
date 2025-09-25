"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==========================================================================
"""

from mpi4py import MPI
import os
import pytest
import numpy as np

import coqui
from coqui.utils.tests.test_coqui_env import mpi
from coqui.mean_field.tests.test_qe import construct_qe_mf


def test_downfold(mpi):
    mf = construct_qe_mf(mpi, "qe_lih222_sym")
    eri_params = {
        "storage": "incore",
        "nIpts": mf.nbnd() * 10,
        "thresh": 1e-10,
        "chol_block_size": 1,
        "init": True
    }
    thc = coqui.make_thc_coulomb(mf, eri_params)

    gw_params = {
        "restart": False, "output": "gw", "niter": 1,
        "beta": 300, "wmax": 4.0, "iaft_prec": "medium",
        "iter_alg": {"alg": "damping", "mixing": 0.7}
    }
    coqui.run_gw(gw_params, h_int=thc)
    mpi.barrier()

    #proj_info = read_proj_info(mf.outdir()+"/lih_wan.h5")

    # downfold the local Green's function
    gloc_params = {
        "outdir": "./", "prefix": "gw",
        "input_type": "scf", "input_iter": 1,
        "wannier_file": mf.outdir() + "/lih_wan.h5",
        "force_real": True
    }
    Gloc_t = coqui.downfold_local_gf(mf, gloc_params)#, projector_info=proj_info)

    assert np.allclose(Gloc_t.imag, 0.0, atol=1e-10), "Imaginary part of Gloc(t) is not negligible"
    assert Gloc_t[-1,0,0,0] == pytest.approx(-0.9921589287308954, abs=1e-10)
    assert Gloc_t[-1,0,1,0] == pytest.approx(-2.3062352653523378e-05, abs=1e-10)
    assert Gloc_t[-1,0,1,1] == pytest.approx(-0.9677107811482517, abs=1e-10)

    # downfold the local screened interaction
    wloc_params = {
        "outdir": "./", "prefix": "gw",
        "screen_type": "gw_edmft_density",
        "input_type": "scf", "input_iter": 1,
        "wannier_file": mf.outdir() + "/lih_wan.h5",
        "permut_symm": True, "force_real": True
    }
    Vloc, Wloc_t = coqui.downfold_local_coulomb(thc, wloc_params)#, projector_info=proj_info)

    assert np.allclose(Vloc.imag, 0.0, atol=1e-12), "Imaginary part of Vloc is not negligible"
    assert Vloc[0,0,0,0] == pytest.approx(1.4160723518754155, abs=1e-12)
    assert Vloc[0,0,1,1] == pytest.approx(0.25499347676713535, abs=1e-12)
    assert Vloc[0,1,0,1] == pytest.approx(4.286546964169289e-05, abs=1e-12)
    assert Vloc[1,1,1,1] == pytest.approx(0.5557140951494038, abs=1e-12)

    assert np.allclose(Wloc_t.imag, 0.0, atol=1e-12), "Imaginary part of Wloc(t) is not negligible"
    assert Wloc_t[0,0,0,0,0] == pytest.approx(-0.2211111829995033, abs=1e-12)
    assert Wloc_t[0,0,0,1,1] == pytest.approx(-0.04914695530722674, abs=1e-12)
    assert Wloc_t[0,0,1,0,1] == pytest.approx(0.0, abs=1e-12)
    assert Wloc_t[0,1,1,1,1] == pytest.approx(-0.10222080762940529, abs=1e-12)

    # downfold the cRPA local screened interaction
    wloc_params["screen_type"] = "crpa"
    Vloc, Uloc_t = coqui.downfold_local_coulomb(thc, wloc_params)#, projector_info=proj_info)

    assert np.allclose(Uloc_t.imag, 0.0, atol=1e-12), "Imaginary part of Uloc(t) is not negligible"
    assert Uloc_t[0,0,0,0,0] == pytest.approx(-0.2152982864315092, abs=1e-12)
    assert Uloc_t[0,0,0,1,1] == pytest.approx(-0.04781796525814118, abs=1e-12)
    assert Uloc_t[0,0,1,0,1] == pytest.approx(-6.839510334289843e-06, abs=1e-12)
    assert Uloc_t[0,1,1,1,1] == pytest.approx(-0.09688743039217523, abs=1e-12)

    if mpi.root():
        os.remove("./gw.mbpt.h5")
    mpi.barrier()

def test_local_coulomb_from_mf(mpi):
    mf = construct_qe_mf(mpi, "qe_lih222_sym")
    eri_params = {
        "storage": "incore",
        "nIpts": mf.nbnd() * 10,
        "thresh": 1e-10,
        "chol_block_size": 1,
        "init": True
    }
    thc = coqui.make_thc_coulomb(mf, eri_params)

    # downfold the local screened interaction
    wloc_params = {
        "outdir": "./", "prefix": "crpa",
        "screen_type": "crpa",
        "input_type": "mf",
        "wannier_file": mf.outdir() + "/lih_wan.h5",
        "permut_symm": True, "force_real": True
    }
    Vloc, Wloc_t = coqui.downfold_local_coulomb(thc, wloc_params)#, projector_info=proj_info)

    assert np.allclose(Vloc.imag, 0.0, atol=1e-12), "Imaginary part of Vloc is not negligible"
    assert Vloc[0,0,0,0] == pytest.approx(1.4160723518754155, abs=1e-12)
    assert Vloc[0,0,1,1] == pytest.approx(0.25499347676713535, abs=1e-12)
    assert Vloc[0,1,0,1] == pytest.approx(4.286546964169289e-05, abs=1e-12)
    assert Vloc[1,1,1,1] == pytest.approx(0.5557140951494038, abs=1e-12)

    assert np.allclose(Wloc_t.imag, 0.0, atol=1e-12), "Imaginary part of Wloc(t) is not negligible"
    assert Wloc_t[0,0,0,0,0] == pytest.approx(-0.2058301133749745, abs=1e-12)
    assert Wloc_t[0,0,0,1,1] == pytest.approx(-0.04205916393585766, abs=1e-12)
    assert Wloc_t[0,0,1,0,1] == pytest.approx(-6.881435931595962e-06, abs=1e-12)
    assert Wloc_t[0,1,1,1,1] == pytest.approx(-0.08689366256595832, abs=1e-12)

    if mpi.root():
        os.remove("./crpa.mbpt.h5")
    mpi.barrier()
