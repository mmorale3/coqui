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

import json

from coqui._lib import post_proc_module as pproc_mod


def ac(mf, params):
    pproc_mod.ac(mf, json.dumps(params))


def band_interpolation(mf, params):
    pproc_mod.band_interpolation(mf, json.dumps(params))


def spectral_interpolation(mf, params):
    pproc_mod.spectral_interpolation(mf, json.dumps(params))


def local_dos(mf, params):
    pproc_mod.local_dos(mf, json.dumps(params))


def unfold_bz(mf, params):
    pproc_mod.unfold_bz(mf, json.dumps(params))


def dump_vxc(mf, params):
    pproc_mod.dump_vxc(mf, json.dumps(params))


def dump_hartree(mf, params):
    pproc_mod.dump_hartree(mf, json.dumps(params))

