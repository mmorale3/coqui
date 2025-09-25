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
import numpy as np

import coqui._lib.embed_module as embed_cxx


def downfold_local_gf(mf, params, *, projector_info=None):
    if projector_info is None:
        return embed_cxx.downfold_gloc(mf, json.dumps(params))[:,:,0]
    else:
        proj_mat = projector_info.get("proj_mat")
        band_window = projector_info.get("band_window")
        kpts_w90 = projector_info.get("kpts_w90")
        # ignore the number of impurities for now
        return embed_cxx.downfold_gloc(
            mf, json.dumps(params), proj_mat, band_window, kpts_w90
        )[:,:,0]


def downfold_local_coulomb(h_int, params, *, projector_info=None, local_polarizabilities=None):
    if local_polarizabilities is not None:
        required_keys = {"imp", "dc"}
        missing = required_keys - local_polarizabilities.keys()
        if missing:
            raise ValueError(f"Missing keys: {missing}")
    else:
        local_polarizabilities = None

    if projector_info is None:
        return embed_cxx.downfold_wloc(
            h_int, json.dumps(params), local_polarizabilities=local_polarizabilities
        )
    else:
        proj_mat = projector_info.get("proj_mat")
        band_window = projector_info.get("band_window")
        kpts_w90 = projector_info.get("kpts_w90")
        return embed_cxx.downfold_wloc(
            h_int, json.dumps(params), proj_mat, band_window, kpts_w90,
            local_polarizabilities=local_polarizabilities
        )


def downfold_1e(mf, params,
                *, projector_info = None, local_selfenergies = None):
    if local_selfenergies is not None:
        required_keys = {"sigma_imp", "sigma_dc", "vhf_imp", "vhf_dc"}
        missing = required_keys - local_selfenergies.keys()
        if missing:
            raise ValueError(f"Missing keys: {missing}")
    else:
        local_selfenergies = None

    if projector_info is not None:
        proj_mat = projector_info.get("proj_mat")
        band_window = projector_info.get("band_window")
        kpts_w90 = projector_info.get("kpts_w90")
        embed_cxx.downfold_1e(mf, json.dumps(params),
                              proj_mat, band_window, kpts_w90,
                              local_selfenergies = local_selfenergies)
    else:
        embed_cxx.downfold_1e(mf, json.dumps(params),
                              local_selfenergies = local_selfenergies)


def downfold_2e(h_int, params,
                *, projector_info = None, pi_imp_and_dc = None):
    if pi_imp_and_dc is None:
        pi_imp, pi_dc = None, None
    else:
        pi_imp = pi_imp_and_dc.get("pi_imp", None)
        pi_dc = pi_imp_and_dc.get("pi_dc", None)

    if projector_info is not None:
        proj_mat = projector_info.get("proj_mat")
        band_window = projector_info.get("band_window")
        kpts_w90 = projector_info.get("kpts_w90")
        embed_cxx.downfold_2e(h_int, json.dumps(params),
                              proj_mat, band_window, kpts_w90,
                              pi_imp_opt=pi_imp, pi_dc_opt=pi_dc)
    else:
        embed_cxx.downfold_2e(h_int, json.dumps(params),
                              pi_imp_opt=pi_imp, pi_dc_opt=pi_dc)


def dmft_embed(mf, params, *, projector_info,
               local_hf_potentials, local_sigma_dynamic):

    required_keys = {"imp", "dc"}
    missing = required_keys - local_sigma_dynamic.keys()
    if missing:
        raise ValueError(f"Missing keys in local_hf_potentials: {missing}")
    missing = required_keys - local_hf_potentials.keys()
    if missing:
        raise ValueError(f"Missing keys in local_sigma_dynamic: {missing}")

    # Append additional axis for the number of impurities
    for key in required_keys:
        if len(local_hf_potentials[key].shape) == 3:
            local_hf_potentials[key] = np.expand_dims(local_hf_potentials[key], axis=1)
        if len(local_sigma_dynamic[key].shape) == 4:
            local_sigma_dynamic[key] = np.expand_dims(local_sigma_dynamic[key], axis=2)

    proj_mat = projector_info.get("proj_mat")
    band_window = projector_info.get("band_window")
    kpts_w90 = projector_info.get("kpts_w90")

    embed_cxx.dmft_embed(mf, json.dumps(params), proj_mat, band_window, kpts_w90,
                         local_hf_potentials, local_sigma_dynamic)
