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

import sys
import numpy as np
from scipy.constants import physical_constants
Hartree_eV = physical_constants['Hartree energy in eV'][0]
import matplotlib.pyplot as plt
from h5 import HDFArchive

def band_plot(ax, coqui_h5, iteration=-1,
              fontsize=16, label="", verbal=True,
              **kwargs):
    with HDFArchive(coqui_h5, 'r') as ar:
        if iteration == -1:
            iteration = ar["scf/final_iter"]

        if "qp_approx" in ar[f"scf/iter{iteration}"]:
            qp_grp = ar[f"scf/iter{iteration}/qp_approx"]
        else:
            qp_grp = ar[f"scf/iter{iteration}"]

        mu = qp_grp["mu"] * Hartree_eV
        E_ska = qp_grp["wannier_inter/E_ska"] * Hartree_eV
        label_idx = qp_grp["wannier_inter/kpt_label_idx"]
        kpt_label_str = qp_grp["wannier_inter/kpt_labels"]

    kpt_label = [letter if letter != 'G' else '$\Gamma$' for letter in kpt_label_str]

    E_ska -= mu
    ns, nkpts, nbnd = E_ska.shape
    if verbal:
        print("  Plotting QP Band Structure")
        print("  --------------------------")
        print("  CoQui h5           = {}".format(coqui_h5))
        if iteration == 0:
            print("  Iteration          = {} (i.e. DFT bands)".format(iteration))
        else:
            print("  Iteration          = {}".format(iteration))
        print("  Number of spins    = {}".format(ns))
        print("  Number of k-points = {}".format(nkpts))
        print("  Number of bands    = {}".format(nbnd))
        print(f"  Chemical potential = {mu:.3f} (eV)\n")
        sys.stdout.flush()

    for i in range(nbnd):
        ax.plot(np.arange(nkpts), E_ska[0, :, i],
                label=label if i == 0 else None, **kwargs)

    ticks_pos = label_idx - 1
    ax.set_xticks(ticks_pos, kpt_label)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_ylabel('$\\epsilon - \\mu$ (eV)', fontsize=fontsize)
    ax.set_xlim(0, nkpts)
    ax.legend(fontsize=fontsize)


def spectral_plot(ax, coqui_h5, calc_type, iteration=-1, orb_list=None,
                  vmax=30, fontsize=16, cmap="viridis", abs_A=False, verbal=True):
    if calc_type not in ["mbpt", "dmft"]:
        raise ValueError(f"Unknown calc_type = {calc_type}. \n"
                         "Acceptable options are 'mbpt' for many-body perturbation theory "
                         "and 'dmft' for dmft embedding results.")

    h5_grp = "scf" if calc_type == "mbpt" else "embed"
    with HDFArchive(coqui_h5, 'r') as ar:
        if iteration == -1:
            iteration = ar[f"{h5_grp}/final_iter"]
        G_wska = ar[f"{h5_grp}/iter{iteration}/ac/G_wskab_inter/output"]
        w_mesh = ar[f"{h5_grp}/iter{iteration}/ac/G_wskab_inter/w_mesh"].real
        label_idx = ar[f"{h5_grp}/iter{iteration}/wannier_inter/kpt_label_idx"]
        kpt_label_str = ar[f"{h5_grp}/iter{iteration}/wannier_inter/kpt_labels"]

    kpt_label = [letter if letter != 'G' else '$\Gamma$' for letter in kpt_label_str]

    nw, ns, nkpts, nbnd = G_wska.shape
    if verbal:
        print("  Plotting spectral function")
        print("  --------------------------")
        print("  CoQuí h5                   = {}".format(coqui_h5))
        print("  Calculation type           = {}".format(calc_type))
        print("  Iteration                  = {}".format(iteration))
        print("  Number of real frequencies = {}".format(nw))
        print("  Number of spins            = {}".format(ns))
        print("  Number of k-points         = {}".format(nkpts))
        if orb_list is None:
            print("  Number of bands            = {}".format(nbnd))
        else:
            print("  Orbital list               = {}".format(orb_list))
        print("  Abs. A(k,w)                = {}".format(abs_A))
        sys.stdout.flush()

    spin_factor = 2.0 if ns == 1 else 1.0
    if orb_list is None:
        orb_list = np.arange(nbnd)

    A_wsk = -1.0 * np.sum(G_wska[:, :, :, orb_list].imag, axis=3)
    A_wk = np.sum(A_wsk, axis=1) * spin_factor

    _spectral_plot(ax, A_wk, w_mesh, kpt_label, label_idx, vmax, fontsize, cmap, abs_A)


def spectral_plot_maxent(ax, coqui_h5, iteration=-1, eta_for_A=0.001,
                         vmax=30, fontsize=16, cmap="viridis", abs_A=False, verbal=True):
    import py2aimb.dmft.pproc.spectral as spec

    with HDFArchive(coqui_h5, 'r') as ar:
        if iteration == -1:
            iteration = ar["embed/final_iter"]
        Simp_wsa = ar[f"embed/iter{iteration}/ac/Sigma_imp_wsa/output"]
        w_mesh = ar[f"embed/iter{iteration}/ac/Sigma_imp_wsa/w_mesh"]
        label_idx = ar[f"embed/iter{iteration}/wannier_inter/kpt_label_idx"]
        kpt_label_str = ar[f"embed/iter{iteration}/wannier_inter/kpt_labels"]
        kpts = ar[f"embed/iter{iteration}/wannier_inter/kpts"]

    kpt_label = [letter if letter != 'G' else '$\Gamma$' for letter in kpt_label_str]

    nw, ns, nbnd = Simp_wsa.shape
    nkpts = kpts.shape[0]
    if verbal:
        print("  Plotting spectral function")
        print("  --------------------------")
        print("  CoQuí h5                   = {}".format(coqui_h5))
        print("  Calculation type           = dmft w/ maxent")
        print("  Iteration                  = {}".format(iteration))
        print("  Number of real frequencies = {}".format(nw))
        print("  Number of spins            = {}".format(ns))
        print("  Number of k-points         = {}".format(nkpts))
        print("  Number of bands            = {}".format(nbnd))
        print("  Abs. A(k,w)                = {}".format(abs_A))
        sys.stdout.flush()

    spin_factor = 2.0 if ns == 1 else 1.0
    A_wska = spec.spectral_kpath(coqui_h5, iteration, Simp_wsa, w_mesh, eta=eta_for_A, verbal=verbal)
    A_wsk = np.sum(A_wska, axis=3)
    A_wk = np.sum(A_wsk, axis=1) * spin_factor

    _spectral_plot(ax, A_wk, w_mesh, kpt_label, label_idx, vmax, fontsize, cmap, abs_A)


def _spectral_plot(ax, A_wk, w_mesh, kpt_label, label_idx,
                   vmax=30, fontsize=16, cmap="viridis", abs_A=False):

    nw, nkpts = A_wk.shape

    neg_value = np.min(A_wk)
    if abs(neg_value) > 1e-3:
        print(f"[WARNING] The spectral has negative values with maximum ~ {neg_value:.3f}. \n"
              f"          Please double check your AC setup. Otherwise, you can set abs_A=True \n"
              f"          to plot |A(k,w)| and compare it w/ A(k,w). \n")

    cax = ax.pcolormesh(np.arange(nkpts), w_mesh*Hartree_eV,
                        np.abs(A_wk)/Hartree_eV if abs_A else A_wk/Hartree_eV,
                        shading='auto', cmap=cmap, vmin=0.0, vmax=vmax)

    # Add a colorbar to show the scale, and set the size of the colar bar tick labels
    cbar = plt.colorbar(cax, ax=ax)
    cbar.set_label("$A(k,\\omega)$" if not abs_A else "$|A(k,\omega)$|", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_ylabel('$\\epsilon - \\mu$ (eV)', fontsize=fontsize)
    ax.set_xlim(0, nkpts)

    # Set custom ticks
    ticks_pos = label_idx - 1
    ax.set_xticks(ticks_pos, kpt_label)
