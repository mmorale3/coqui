"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team

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
from functools import partial
import numpy as np
from h5 import HDFArchive

import triqs_modest as modest

import coqui
from coqui.utils.imag_axes_ft import IAFT
import coqui.dmft as coqui_dmft
from coqui.dmft.io import convert_gw_edmft_params, _normalize_solver_params_list
# Q5 (notes/q5_option2_outer_loop_spec.md): the Option-2 outer-loop diagnostics.
# Imported by module path, like coqui.dmft.io above -- outer_loop.py is numpy-only
# and pulls in nothing from this package.
import coqui.dmft.outer_loop as outer_loop_diag

Hartree_eV = 27.211386245988


def run_gw_edmft(h_int, embedding, inner_loop_alg=1, *, proj_info=None, params: dict):
    """
    Run the GW+EDMFT self-consistency workflow.

    Parameters are passed as a dictionary via ``params``.  

    Parameters
    ----------
    h_int : ThcCoulomb
        Coulomb interaction object for the full system, used in GW and
        downfolding/upfolding steps. Obtained from ``make_thc_coulomb``.
    embedding : triqs_modest.embedding
        Embedding object from TRIQS/ModEST
        (https://github.com/TRIQS/modest) defining mappings between local MLWF
        orbitals and impurity subspaces.
    inner_loop_alg : int, optional
        EDMFT inner-loop mode.

        - ``1``: recompute ``G_loc`` and ``W_loc`` each EDMFT iteration.
        - ``2``: keep ``G_loc`` and ``W_loc`` fixed inside one EDMFT inner loop.

        Default is ``1``.
    proj_info : dict, optional
        Projector metadata for the correlated subspace. If ``None``,
        ``wannier_file`` must be provided in ``params``.
    params : dict
        GW+EDMFT control parameter dictionary.

        Required top-level keys
        ~~~~~~~~~~~~~~~~~~~~~~~
            - ``niter`` (int): total number of outer GW+EDMFT cycles. Must be a positive integer.
            - ``edmft`` (dict): EDMFT subsection containing impurity controls.
            - ``edmft.impurity`` (dict or list[dict]): impurity solver parameter set(s).
            - ``wannier_file`` (str), only when ``proj_info is None``:
                Wannier90 HDF5 file used to build projectors.

        Optional top-level keys and defaults
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            - ``lattice_solver`` (``"gw"``, ``"qpgw"``; default ``"gw"``): solver used
                for the lattice stage.

                - ``"gw"``: the pre-Q4 workflow. ``coqui.run_gw`` is called
                    ``gw_iter_per_loop`` times inside every GW+EDMFT cycle.
                - ``"qpgw"``: the Q4 skeleton with a **frozen** effective Hamiltonian
                    (Option 1, ruling R-Q4-4 of ``notes/q4_edmft_skeleton_spec.md``).
                    ``coqui.run_qpgw`` runs **once**, before the outer loop, with
                    ``projector_info`` and the current
                    ``local_polarizabilities = dmft_state.local_pi_w`` attached; every
                    subsequent outer cycle is EDMFT-only and the GW stage is skipped
                    (the freeze is logged each cycle). ``gw_iter_per_loop`` is ignored
                    and ``edmft_iter_per_loop`` must be ``>= 1``. Q5 unfreezes this by
                    moving the qpGW stage back inside the cycle.

                    **Where the fermionic double counting is set.** Not by a downfold
                    knob: this workflow never calls ``coqui.downfold_1e``, so the
                    ``qp_selfenergy`` / ``dc_type`` parameters of the C++ QP-downfold
                    route (``MBPT_drivers.cpp`` ``downfolding_1e`` /
                    ``embed_t::downfolding``) are inert here and are deliberately not
                    set. The DC is built in python by
                    :func:`coqui.dmft.weiss.solve_gw_dc`:
                    ``Σ_dc = −G_loc·W_loc(τ)`` with the **full dynamical** ``W_loc``,
                    and ``Vhf_dc = eval_hf_dc(dm, V, U(0)+V)`` (Hartree at ``U(0)+V``,
                    exchange at ``V``). That is PDF eq 12 verbatim, and it is the unique
                    choice under which the Q4-b clean-limit gate is exact -- accepted as
                    the skeleton's fermionic DC by the R-Q4-1 AMENDMENT of
                    ``notes/q4_edmft_skeleton_spec.md`` §2. The original R-Q4-1 static-U
                    ``dc_type="gw"`` level applies only to the C++ ``downfold_1e``
                    route (one-shot / model workflows).
            - ``outer_loop`` (``"option1"``, ``"option2"``; default ``"option1"``;
                requires ``lattice_solver="qpgw"``): which outer loop of
                ``notes/q5_option2_outer_loop_spec.md`` to run.

                - ``"option1"``: the Q4 frozen-H_eff stage above, wired byte-identically
                    to the pre-Q5 workflow.
                - ``"option2"``: H_eff is **re-derived every outer cycle** from
                    ``Sigma^GW[G_latt, W_corr]`` via the mode-A map (PDF eq 3-4). The
                    qpGW+BSE stage moves INSIDE the cycle, with ``restart=True``,
                    ``local_polarizabilities = dmft_state.local_pi_w`` and
                    ``greens_func_source="embed"`` pointing at the previous cycle's
                    upfolded lattice G -- the C++ re-QP-ization step, in which iteration 1
                    of the qp loop consumes that G (and its density matrix) instead of the
                    restart-H_eff's analytic one. The first cycle of a fresh run has no
                    ``embed`` group and falls back to the frozen-stage behaviour (no
                    injection). Outer H_eff damping IS the qp loop's own ``iter_alg``
                    mixing against the checkpointed H_eff (ruling R-Q5-1; PDF §7 asks for
                    a conservative ``mixing ~ 0.3`` near a transition, which is the
                    workflow default). Each cycle also emits the Q5-b Mott-feedback-chain
                    log block and stores its trail under ``q5_outer_loop`` in the impurity
                    checkpoint.
            - ``outer_qpgw_niter`` (int, default ``1``; ``outer_loop="option2"`` only):
                qp iterations of the per-cycle lattice stage. ``1`` is the pure Option-2
                one-shot re-QP step -- the outer loop supplies the outer iteration.
            - ``qpgw`` (dict, only used when ``lattice_solver="qpgw"``): knobs forwarded
                verbatim to :func:`coqui.run_qpgw` (``qp_map``, ``off_diag_mode``,
                ``eta``, ``Nfit``, the ``qp_modea_*`` family, and the BSE/ladder
                ``pol_vertex_*`` family). ``outdir``, ``prefix``, ``restart``,
                ``screen_type``, ``div_treatment``, ``iter_alg`` and ``niter`` (=10;
                the frozen stage must reach its own qp fixed point) are
                filled in from the top-level settings and may be overridden here.
            - ``gw_iter_per_loop`` (int, default ``1``): number of GW updates per
                GW+EDMFT cycle. Ignored when ``lattice_solver="qpgw"``.
            - ``edmft_iter_per_loop`` (int, default ``1``): number of EDMFT updates
                per GW+EDMFT cycle.
            - ``outdir`` (str, default ``"./"``): output/checkpoint directory.
            - ``prefix`` (str, default ``"coqui"``): checkpoint file prefix.
            - ``restart`` (bool, default ``True``; ``False`` is not implemented):
                continue from an existing GW checkpoint.
            - ``screen_type`` (``"rpa"``, ``"gw_edmft"``; default ``"gw_edmft"``): 
                screening prescription for the lattice irreducible polarization. 
            - ``div_treatment`` (str, default ``"gygi"``): treatment for the
                ``q -> 0`` divergence.
            - ``corr_only`` (bool, default ``True``): embmeding dynamic part of 
                the self-energy only. 

        ``iter_alg`` section (optional): Iterative controls for GW and EDMFT inner loops.
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ``iter_alg`` defaults to ``{"alg": "damping", "mixing": 0.3}``.
        Currently, only ``alg="damping"`` is supported for GW+EDMFT.

            - ``mixing`` (float, default ``0.3``): fallback damping value.
            - ``gw_mixing`` (float, default ``mixing``): GW-specific damping.
            - ``edmft_mixing`` (float, default ``mixing``): EDMFT-specific damping.
            - ``edmft_mix_in_first_iter`` (bool, default ``True``): whether to mix 
                the EDMFT solution in the first iteration.

        ``edmft`` section
        ~~~~~~~~~~~~~~~~~
            - ``chkpt_h5`` (str, default ``{outdir}/{prefix}.mbpt.h5``): impurity
                solver checkpoint file path. If the file exists and ``restart=True``,
                previous EDMFT impurity results are loaded as the initial guess.
            - ``iaft`` (dict, optional): impurity DLR mesh controls.
                - ``wmax`` (float, default GW ``wmax``): impurity DLR frequency cutoff.
                - ``eps`` (float, default GW ``eps``): impurity DLR precision.
            - ``impurity`` (dict or list[dict], required): per-impurity solver
                configuration. A single dict applies to one impurity; use a list for
                multiple impurities.

                Per-impurity solver parameters (``edmft.impurity``)
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                Only TRIQS/CT-SEG is currently supported. Post-processing options are
                part of the same impurity-solver parameter block.

                Required CT-SEG run controls:

                    - ``n_cycles`` (int): total Monte Carlo measurement cycles across MPI.
                    - ``n_warmup_cycles`` (int): warmup Monte Carlo cycles.
                    - ``length_cycle`` (int): updates per Monte Carlo cycle.

                Optional CT-SEG mesh/solver controls:

                    - ``n_iw`` (default ``2 * max_dlr_idx``): Internal uniform Matsubara 
                        mesh size within CT-SEG. 
                    - ``n_tau`` (default ``6 * n_iw + 1``): Internal uniform imaginary-time 
                        mesh size within CT-SEG.
                    - ``n_tau_bosonic`` (default ``n_tau``): Internal bosonic imaginary-time 
                        mesh size within CT-SEG. 
                    - ``perform_tail_fit`` (default ``False``): enable self-energy tail fit.
                    - ``fit_max_moment`` (default ``3``): highest fitted self-energy moment.
                    - ``fit_min_w`` / ``fit_max_w`` / ``fit_min_n`` / ``fit_max_n``
                        (default ``None``): tail-fit window controls.
                    - ``analytic_hf`` (default ``True``): evaluate static self-energy 
                        analytically.
                    - ``truncate_uchi`` (default ``False``): stabilize ``U*Chi`` inversion i
                        n DLR post-processing.

                Additional DMFT workflow options in the same impurity block:

                     - ``retardation`` (``"dynamic"``, ``"static_u_zb"``; default
                         ``"dynamic"``): impurity retardation policy (ruling R-Q4-5).
                         ``"dynamic"`` (default, pre-Q4 behaviour) hands the solver the
                         full retarded ``U(iν)``. ``"static_u_zb"`` is **impurity mode
                         (a)**: the retarded part is dropped, the static interaction is
                         taken from ``static_u_source``, and the hybridization is
                         renormalized as ``Δ → Z_B Δ`` with the Casula-Werner bandwidth
                         factor (:func:`coqui.dmft.retardation.casula_werner_zb`). The
                         double-counting terms and the checkpointed solver inputs are
                         always built from the *unmodified* ``Vloc`` / ``u_weiss_iw``.
                     - ``static_u_source`` (``"u0"``, ``"u_inf"``; default ``"u0"``):
                         which column of ``U(iν)`` becomes the static interaction in
                         ``retardation="static_u_zb"``. ``"u0"`` is the screened
                         ``U(iν = 0)`` (the Casula-Werner standard); ``"u_inf"`` is the
                         unscreened ``U(iν → ∞) = Vloc``, the PDF §3.3 literal. ⚠ The
                         default deliberately contradicts the PDF §3.3 text as written
                         (R-Q4-5 AMENDMENT): pairing the bare interaction with
                         ``Z_B < 1`` double-counts screening, because ``Z_B`` comes from
                         integrating out the *screening* bosons. Set ``"u_inf"``
                         explicitly if the PDF literal was intended.
                     - ``init_imp_results`` (str, default ``"dc"``): initialization strategy
                         for impurity self-energies (``"dc"`` or ``"zero"``).
                     - ``degenerate_blk`` (list[list[int]], default ``None``): explicit
                         orbital-degeneracy groups for symmetry enforcement.
                     - ``degenerate_blk_thresh`` (float, default ``None``): infer
                         ``degenerate_blk`` from hybridization when explicit blocks are absent.
                     - ``screen_j`` (bool, default ``False``): include screened Hund's
                         coupling in density-density mapping.
                     - ``causal_projection`` (dict, default ``None``): options forwarded to
                         bosonic causal projection utilities.
                     - ``chemical_potential`` (dict, optional): auxiliary impurity chemical-
                         potential solver controls:
                         - ``tolerance`` (float, default ``0.1``)
                         - ``suppress_solver_output`` (bool, default ``True``)
                         - ``solver_output_file`` (str)
                         - ``n_cycles`` / ``n_warmup_cycles`` / ``length_cycle`` (int)

                         If omitted, defaults are derived from the main impurity solver setup:
                         ``n_cycles = int(solver_n_cycles * 0.05)``,
                         ``n_warmup_cycles = solver_n_warmup_cycles``, and
                         ``length_cycle = solver_length_cycle``.

    Returns
    -------
    None
        Results are written to checkpoint files (``{outdir}/{prefix}.mbpt.h5`` 
        and the impurity ``chkpt_h5``).

    Examples
    --------
    Minimal flat ``params`` dictionary:

    .. code-block:: python

        params = {
            "niter": 2,
            "gw_iter_per_loop": 1,
            "edmft_iter_per_loop": 1,
            "outdir": "./",
            "prefix": "svo",
            "screen_type": "gw_edmft",
            "div_treatment": "gygi",
            "wannier_file": "svo.mlwf.h5",
            "iter_alg": {
                "alg": "damping",
                "mixing": 0.4,
                "edmft_mixing": 0.2,
                "edmft_mix_in_first_iter": True,
            },
            "edmft": {
                "chkpt_h5": "./svo.dmft.h5",
                "iaft": {"wmax": 5.0, "eps": 1e-10},
                "impurity": {
                    "init_imp_results": "dc",
                    "n_iw": 3000,
                    "n_tau": 96001,
                    "length_cycle": 60,
                    "n_warmup_cycles": 6000,
                    "n_cycles": 2000000,
                    "chemical_potential": {
                        "tolerance": 0.1,
                        "n_cycles": 100000,
                    },
                },
            },
        }
        run_gw_edmft(h_int, embedding, params=params)
    """
    # Convert parameters to the internal format and set defaults
    params = convert_gw_edmft_params(params)
    coqui_mpi = h_int.mpi()
    mf = h_int.mf()

    if not isinstance(embedding, modest.embedding.Embedding):
        raise TypeError(f"Expected a modest.embedding object for 'embedding', got {type(embedding)}")

    if proj_info is None:
        obe = modest.make_one_body_elements_gw(params.pop("wannier_file"))
        proj_info = coqui_dmft.get_proj_info(obe.P)

    # Convert to the internal format
    niter, gw_iter_per_loop, edmft_iter_per_loop = (
        params.pop('niter'), params.pop('gw_iter_per_loop'), params.pop('edmft_iter_per_loop')
    )
    # Q4 (ruling R-Q4-4): lattice stage selector. "gw" reproduces the pre-Q4 workflow.
    lattice_solver = params.pop('lattice_solver', 'gw')
    qpgw_params = params.pop('qpgw', None)
    # Q5 (notes/q5_option2_outer_loop_spec.md): outer-loop selector. "option1" is the Q4
    # frozen-H_eff stage; "option2" re-derives H_eff every cycle (PDF eq 3-4 + §7).
    outer_loop = params.pop('outer_loop', 'option1')
    params.pop('outer_qpgw_niter', None)   # consumed by convert_gw_edmft_params (qpgw.niter)
    option2 = (lattice_solver == "qpgw" and outer_loop == "option2")

    # http://patorjk.com/software/taag/#p=display&f=Calvin+S&t=COQUI+GW%2BEDMFT&x=none&v=4&h=4&w=80&we=false
    coqui.app_log(1, "╔═╗╔═╗╔═╗ ╦ ╦╦  ╔═╗┬ ┬╔═╗┌┬┐┌┬┐┌─┐┌┬┐\n"
                     "║  ║ ║║═╬╗║ ║║  ║ ╦│││║╣  │││││├┤  │ \n"
                     "╚═╝╚═╝╚═╝╚╚═╝╩  ╚═╝└┴┘╚═╝─┴┘┴ ┴└   ┴ \n")
    coqui.app_log(1, f"  Total GW+EDMFT cycles (niter)       = {niter}")
    coqui.app_log(1, f"  Lattice solver                      = {lattice_solver}")
    if lattice_solver == "qpgw":
        coqui.app_log(1, f"  Outer loop                          = {outer_loop}")
    if option2:
        coqui.app_log(1, f"  qpGW iterations per GW+EDMFT cycle  = "
                         f"{qpgw_params.get('niter') if qpgw_params else '?'} "
                         f"(Option 2: H_eff re-derived every cycle)")
    elif lattice_solver == "qpgw":
        coqui.app_log(1,  "  GW iterations per GW+EDMFT cycle    = 0 (frozen H_eff, Option 1)")
    else:
        coqui.app_log(1, f"  GW iterations per GW+EDMFT cycle    = {gw_iter_per_loop}")
    coqui.app_log(1, f"  EDMFT iterations per GW+EDMFT cycle = {edmft_iter_per_loop}")
    coqui.app_log(1, f"    - Fix Gloc and Wloc during EDMFT iterations = {inner_loop_alg==2}\n")

    embedding_2e = embedding.merge_embed_block_by_imp.slice_sigma
    coqui.app_log(1, embedding.description(True))

    try:
        gw_params        = params.pop('gw', None)
        wloc_params      = params.pop('wloc')
        gloc_params      = params.pop('gloc')
        embed_params     = params.pop('dmft_embed')
        impurity_params  = params.pop('impurity')
        imp_iaft_params  = impurity_params.pop('iaft', {})
        # ``convert_gw_edmft_params`` always populates 'iter_alg' (io.py supplies the
        # {"alg": "damping", "mixing": 0.3} default), but a caller that hands
        # ``run_gw_edmft`` an already-internal params dict without the section used to
        # leave this None and blow up much later inside the EDMFT inner loop with a bare
        # ``'NoneType' object has no attribute 'get'``. Degrade to an empty mapping so the
        # per-call-site defaults below (mixing=0.7) apply instead.
        iterative_params = impurity_params.pop('iter_alg', None) or {}
    except KeyError as e:
        raise KeyError(f"run_gw_edmft: Missing required params key: {e.args[0]}")

    # Scale Monte-Carlo cycle counts by MPI communicator size.
    impurity_params['solver'] = _normalize_solver_params_list(
        impurity_params['solver'], coqui_mpi.comm_size()
    )

    coqui_chkpt_h5 = embed_params['outdir']+"/"+embed_params['prefix']+".mbpt.h5"
    solver_chkpt_h5 = impurity_params.pop('chkpt_h5', coqui_chkpt_h5)

    # DMFT state container
    dmft_state = coqui_dmft.DMFTState.make_dmft_state(
        coqui_chkpt_h5, embedding, embedding_2e, 
        wmax_imp=imp_iaft_params.get('wmax', None),
        eps_imp=imp_iaft_params.get('eps', None),
        spin_average=mf.nspin()==1,
        screen_type=wloc_params['screen_type'],
        verbal=coqui_mpi.root()
    )
    if impurity_params.pop('restart', True):
        dmft_state.load(solver_chkpt_h5)

    coqui_mpi.barrier()

    if lattice_solver == "qpgw" and not option2:
        # No qp_selfenergy/dc_type knob is set here: the python EDMFT loop never routes
        # through coqui.downfold_1e, so both keys would be inert (R-Q4-1 AMENDMENT).
        # The loop's fermionic DC is weiss.solve_gw_dc, called in the inner loop below.
        #
        # The frozen-H_eff lattice stage: ONE qpGW+BSE solve before the outer loop.
        _qpgw_lattice_stage(h_int, proj_info, dmft_state, qpgw_params)
        coqui_mpi.barrier()

    # Q5 gate Q5-b / R-Q5-2: per-cycle diagnostics carried across the outer loop.
    diag_prev = {'sigma_dc': None, 'pi_dc': None, 'mo_skia': None, 'proj_mo_c': None}
    diag_trail = []

    for iteration in range(niter):

        if option2:
            # Option 2 (Q5): H_eff is RE-DERIVED from Sigma^GW[G_latt, W_corr] this cycle.
            # The lattice G of the previous cycle -- the embedded one when an "embed" group
            # exists -- is injected into iteration 1 of the qp loop through
            # greens_func_source; the first cycle has no embed group and therefore falls
            # back to the frozen-stage behaviour (no injection, C = empty set).
            _qpgw_lattice_stage(h_int, proj_info, dmft_state, qpgw_params,
                                coqui_chkpt_h5=coqui_chkpt_h5,
                                cycle=iteration + 1, niter=niter)
            coqui_mpi.barrier()
        elif lattice_solver == "qpgw":
            # Option 1 freeze (R-Q4-4): H_eff, the qpGW+BSE W and the checkpoint were
            # produced once before the loop and are NOT refreshed per cycle.
            coqui.app_log(1, f"[GW+EDMFT cycle {iteration+1}/{niter}] lattice_solver = \"qpgw\": "
                             f"H_eff is FROZEN (Option 1, ruling R-Q4-4).\n"
                             f"  --> skipping the per-cycle lattice update; "
                             f"the qpGW+BSE stage ran once before the outer loop.\n")
        elif gw_params is not None and gw_iter_per_loop >= 1:
            # update GW solution with fixed impurity self-energies and polarizabilities
            _gw_loop(
                mf, h_int, proj_info, 
                dmft_state, coqui_chkpt_h5, 
                gw_params, embed_params, gw_iter_per_loop
            )

        if edmft_iter_per_loop >= 1:
            # Set the Green's function for the non-local RPA polarizability
            if lattice_solver == "qpgw":
                # The qp SCF loop writes "scf/iter{N}" through chkpt::dump_scf
                # (tools/chkpt_utils.cpp:108-130) with Dm_skij/Heff_skij/MO_skia/E_ska/mu
                # and stores NEITHER "greens_func_source" NOR the legacy "input_grp". The
                # G that W_loc's RPA bubble must use is the qpGW lattice G itself, which
                # read_greens_function rebuilds on the fly from (MO_skia, E_ska, mu)
                # (SCF/scf_common.cpp:440-459).
                with HDFArchive(coqui_chkpt_h5, 'r') as ar:
                    gf_for_wloc_source = "scf"
                    gf_for_wloc_iteration = ar["scf/final_iter"]
            else:
                with HDFArchive(coqui_chkpt_h5, 'r') as ar:
                    mbpt_final_iter = ar["scf/final_iter"]
                    try:
                        gf_for_wloc_source = ar[f"scf/iter{mbpt_final_iter}/greens_func_source"]
                        gf_for_wloc_iteration = ar[f"scf/iter{mbpt_final_iter}/greens_func_iteration"]
                    except KeyError:
                        gf_for_wloc_source = ar[f"scf/iter{mbpt_final_iter}/input_grp"]
                        gf_for_wloc_iteration = ar[f"scf/iter{mbpt_final_iter}/input_iter"]
            wloc_params["greens_func_source"] = gf_for_wloc_source
            wloc_params["greens_func_iteration"] = gf_for_wloc_iteration
            coqui_mpi.barrier()

            # inner EDMFT loop
            edmft_alg = {1: _edmft_loop, 2: _edmft_loop_fixed_gloc_and_wloc}
            try:
                edmft_impl = edmft_alg[inner_loop_alg]
            except KeyError:
                raise ValueError(f"Unknown inner_loop_alg={inner_loop_alg!r} (expected 1, or 2)")
        
            edmft_impl(
                mf, h_int, proj_info, dmft_state, solver_chkpt_h5, coqui_chkpt_h5, 
                gloc_params, wloc_params, impurity_params['solver'], embed_params, 
                iterative_params, edmft_iter_per_loop
            )

        if option2:
            # Q5-b: ONE consolidated Mott-feedback-chain block + checkpoint trail per cycle.
            diag_trail.append(_option2_cycle_diagnostics(
                mf, proj_info, dmft_state, coqui_chkpt_h5,
                impurity_params['solver'], diag_prev,
                cycle=iteration + 1, niter=niter, verbose=coqui_mpi.root()
            ))
            coqui_mpi.barrier()

    if option2 and coqui_mpi.root():
        _save_mott_chain_trail(solver_chkpt_h5, diag_trail)
    if option2:
        coqui_mpi.barrier()


def qpgw_stage_greens_func_source(coqui_chkpt_h5):
    """
    Q5 (spec §1 piece 2): pick the checkpoint group whose Green's function ITERATION 1 of
    the per-cycle qpGW stage must consume.

    ``"embed"`` -- the upfolded lattice G of the previous outer cycle, i.e. the object eq 3
    calls ``G_latt`` -- as soon as ``coqui.dmft_embed`` has written one. Before that (the
    first cycle of a fresh run) there is no embed group, and the stage falls back to the
    frozen-stage behaviour: NO injection, the qp loop builds its own analytic G. That
    fallback is the C = empty-set limit, and it is what Q5-g1/Q5-g2 pin.

    Returns ``(source, iteration)`` with ``source = None`` meaning "no injection".
    """
    try:
        with HDFArchive(coqui_chkpt_h5, 'r') as ar:
            if "embed" in ar.keys():
                return "embed", ar["embed/final_iter"]
    # RuntimeError is what TRIQS's h5 raises for an unopenable archive -- the C++ h5
    # layer's error, NOT a python OSError. Without it in the tuple this guard never fired
    # on a real TRIQS host and a missing/unreadable checkpoint aborted the Option-2 cycle
    # with a raw HDF5 traceback instead of falling back to "no injection".
    except (OSError, RuntimeError, KeyError):
        pass
    return None, -1


def _qpgw_lattice_stage(h_int, proj_info, dmft_state, qpgw_params,
                        coqui_chkpt_h5=None, cycle=None, niter=None):
    """
    Run the qpGW+BSE lattice stage of the GW+EDMFT skeleton.

    ``outer_loop = "option1"`` (default, ruling R-Q4-4) calls this ONCE before the outer
    loop: the quasiparticle Hamiltonian, the qpGW+BSE screened interaction and the
    ``scf/iter{N}`` checkpoint entry are produced here and never refreshed while the EDMFT
    cycles run. Passing ``coqui_chkpt_h5`` switches on the Q5 **Option-2** behaviour: the
    stage runs INSIDE every outer cycle and iteration 1 consumes the previous cycle's
    lattice G through ``greens_func_source`` (the C++ re-QP-ization step,
    ``notes/q5_option2_outer_loop_spec.md`` §1 piece 1).

    The impurity correction enters the lattice polarization only through
    ``local_polarizabilities`` (``P_latt = P^RPA[G_latt] + P^lad +
    P_C[P_imp - P_dc]P_C^dag``); on the first, from-scratch cycle
    ``dmft_state.local_pi_w`` is ``None``, which is the C = empty-set limit
    gated by Q4-a.

    Parameters
    ----------
    h_int : ThcCoulomb
        Coulomb interaction object for the full system.
    proj_info : dict
        Projector metadata (``proj_mat``, ``band_window``, ``kpts_w90``).
    dmft_state : DMFTState
        Supplies ``local_pi_w`` (``None`` before any impurity solution exists).
    qpgw_params : dict
        Parameter block forwarded verbatim to :func:`coqui.run_qpgw`.
    coqui_chkpt_h5 : str, optional
        Option-2 only: the CoQuí checkpoint from which the external G is taken.
    cycle, niter : int, optional
        Option-2 only: outer-cycle counters, for the log header.
    """
    if qpgw_params is None:
        raise KeyError(
            "run_gw_edmft: lattice_solver=\"qpgw\" requires a qpGW parameter block. "
            "This is built by coqui.dmft.io.convert_gw_edmft_params."
        )

    coqui_mpi = h_int.mpi()
    option2 = coqui_chkpt_h5 is not None
    if option2:
        qpgw_params = dict(qpgw_params)
        gf_source, gf_iter = qpgw_stage_greens_func_source(coqui_chkpt_h5)
        # Absent => the C++ knob stays inert and the loop builds its own analytic QP G.
        qpgw_params.pop("greens_func_source", None)
        qpgw_params.pop("greens_func_iteration", None)
        if gf_source is not None:
            qpgw_params["greens_func_source"] = gf_source
            qpgw_params["greens_func_iteration"] = gf_iter
        coqui.app_log(1, f"Lattice stage [cycle {cycle}/{niter}]: qpGW+BSE, "
                         f"H_eff RE-DERIVED (Option 2 / Q5)")
        coqui.app_log(1, "-------------------------------------------------------------------")
        coqui.app_log(1, f"  external G (iteration 1)   = "
                         f"{'none (first cycle: analytic QP G)' if gf_source is None else f'{gf_source}/iter{gf_iter}'}")
        coqui.app_log(1, f"  qp iterations this cycle   = {qpgw_params.get('niter')}")
        coqui.app_log(1, f"  H_eff damping (iter_alg)   = "
                         f"{qpgw_params.get('iter_alg', {}).get('mixing')}  (R-Q5-1)")
    else:
        coqui.app_log(1, "Lattice stage: qpGW+BSE, run ONCE (frozen H_eff, Option 1 / R-Q4-4)")
        coqui.app_log(1, "-------------------------------------------------------------------")
    coqui.app_log(1, f"  screen_type                = {qpgw_params.get('screen_type')}")
    coqui.app_log(1, f"  qp_map                     = {qpgw_params.get('qp_map', 'ac_pade')}")
    coqui.app_log(1, f"  pol_vertex                 = {qpgw_params.get('pol_vertex', 'none')}")
    coqui.app_log(1, f"  local polarizabilities     = "
                     f"{'attached' if dmft_state.local_pi_w is not None else 'absent (C = empty set)'}\n")

    coqui.run_qpgw(
        qpgw_params, h_int = h_int, projector_info = proj_info,
        local_polarizabilities = dmft_state.local_pi_w
    )
    coqui_mpi.barrier()


def _option2_cycle_diagnostics(mf, proj_info, dmft_state, coqui_chkpt_h5,
                               solver_params_list, prev, cycle, niter, verbose=True):
    """
    Gate Q5-b: assemble and log ONE Mott-feedback-chain block for this outer cycle, and
    return its fixed-layout trail row (:data:`coqui.dmft.outer_loop.MOTT_CHAIN_TRAIL_LABELS`).

    Where each field comes from:

    ============================  =========================================================
    ``gap_eV``                    ``scf/iter{N}/E_ska`` of the qpGW stage just run
    ``epsilon_inf``               ``scf/iter{N}/epsilon_inf`` (``scr_coulomb_t.cpp:1526``)
    ``lambda_nu0``                the eq-6 ladder watchdog, ``scf/iter{N}/lambda_nu0``.
                                  Increment Q6 §1.4(a) PERSISTS it from the C++ stage
                                  (``scr_coulomb_t.cpp``, the Q4 checkpoint-write block),
                                  so it is a real number whenever the ladder was injected;
                                  before Q6 it was permanently "not measured".
    ``*_imp_minus_dc``            ``dmft_state.local_{sigma,pi}_w`` on the tau axis
                                  (the ``dmft_state.py:267-289`` metric)
    ``u_bar_0`` / ``z_b``         impurity 0's ``Vloc + u_weiss_iw``; ``z_b`` only in
                                  retardation mode (a) (``static_u_zb``, R-Q4-5)
    ``dc_*_staleness``            this cycle's DC against the previous cycle's
    ``band_reorder_count``        maximal-overlap continuation meter on ``MO_skia``
    ``o_c``                       C-window MO character retention (R-Q5-2)
    ``r_nu0``/``r_mid``/``r_top`` Q6 §1.1 (PDF §8.3): the cancellation load
                                  ``||P_imp - P_dc||/||P_dc||`` per nu band, from the same
                                  ``dmft_state.local_pi_w`` that feeds ``pi_imp_minus_dc``
    ``lad_over_dc``               Q6 §1.1 / C3b: ``||P^lad_loc,orb||/||P_dc||`` from
                                  ``scf/iter{N}/pi_lad_loc_orb_wabcd``
    ============================  =========================================================

    Every step is guarded: a diagnostic must never take a production run down, and any
    field it cannot source stays at the finite ``MISSING`` sentinel.
    """
    ol = outer_loop_diag
    iaft = dmft_state.iaft
    fields = {}

    def _tau_f(d):
        return iaft.w_to_tau(d, stats='f')

    def _tau_b(d):
        return iaft.w_to_tau_phsym(d, stats='b')

    # ---- the lattice stage: gap(H_eff), eps_inf, and the MO set for the trackers -------
    e_ska = mo_skia = ovlp = pi_lad_orb = None
    try:
        with HDFArchive(coqui_chkpt_h5, 'r') as ar:
            it = ar["scf/final_iter"]
            grp = ar[f"scf/iter{it}"]
            e_ska = np.asarray(grp["E_ska"])
            mo_skia = np.asarray(grp["MO_skia"])
            if "epsilon_inf" in grp.keys():
                fields['epsilon_inf'] = float(np.real(grp["epsilon_inf"]))
            # Q6 §1.4(a): the Q3 injection meters, now persisted by the C++ stage. Absent
            # whenever the ladder was not injected -- the field then stays at MISSING.
            if "lambda_nu0" in grp.keys():
                fields['lambda_nu0'] = float(np.real(grp["lambda_nu0"]))
            if "pi_lad_loc_orb_wabcd" in grp.keys():
                pi_lad_orb = np.asarray(grp["pi_lad_loc_orb_wabcd"])
            ovlp = np.asarray(ar["system/S_skij"])
    # RuntimeError: TRIQS's h5 raises it (not OSError) when the archive or a dataset in it
    # cannot be read. A diagnostic must never take a production run down.
    except (OSError, RuntimeError, KeyError, ValueError, TypeError) as e:
        coqui.app_log(2, f"[Q5-b] lattice-stage fields not available this cycle: {e}")
    fields['gap_eV'] = ol.heff_gap_eV(e_ska, mf.nelec())

    # ---- R-Q5-2 subspace tracking ------------------------------------------------------
    proj_mo_c = None
    if mo_skia is not None:
        fields['band_reorder_count'] = ol.count_band_reorderings(
            mo_skia, prev.get('mo_skia'), ovlp)
        try:
            proj_mo_c = ol.project_mo_on_c(
                mo_skia, proj_info['proj_mat'], proj_info['band_window'])
            fields['o_c'] = ol.c_window_overlap(proj_mo_c, prev.get('proj_mo_c'))
        except (KeyError, ValueError, TypeError) as e:
            coqui.app_log(2, f"[Q5-b] o_C not available this cycle: {e}")

    # ---- the impurity/DC channel -------------------------------------------------------
    fields['sigma_imp_minus_dc'] = ol.imp_minus_dc(dmft_state.local_sigma_w, transform=_tau_f)
    fields['pi_imp_minus_dc'] = ol.imp_minus_dc(dmft_state.local_pi_w, transform=_tau_b)

    sigma_dc = dmft_state.local_sigma_w["dc"] if dmft_state.local_sigma_w else None
    pi_dc = dmft_state.local_pi_w["dc"] if dmft_state.local_pi_w else None
    fields['dc_sigma_staleness'] = ol.dc_staleness(sigma_dc, prev.get('sigma_dc'),
                                                   transform=_tau_f)
    fields['dc_pi_staleness'] = ol.dc_staleness(pi_dc, prev.get('pi_dc'), transform=_tau_b)

    # ---- Q6 §1.1 (PDF §8.3): the R(inu) cancellation load ------------------------------
    # Same object as pi_imp_minus_dc above, but NORMALISED by ||P_dc|| and resolved per nu
    # band -- that normalisation is what makes it a cancellation meter rather than a
    # magnitude. Measured on the nu axis the arrays already live on (no tau transform: the
    # ratio is taken node by node, and a tau image would mix the nodes it is separating).
    fields['r_nu0'], fields['r_mid'], fields['r_top'] = \
        ol.r_cancellation_load(dmft_state.local_pi_w)
    fields['lad_over_dc'] = ol.ladder_over_dc(pi_lad_orb, pi_dc)

    # ---- Ubar(0) and Z_B (impurity mode (a), R-Q4-5) ------------------------------------
    try:
        inp = dmft_state.solver_inputs[0]
        v_loc, u_weiss = inp['Vloc'], inp['u_weiss_iw']
        nu = _bosonic_nu_mesh(iaft)
        u_bar = coqui_dmft.total_density_channel(
            coqui_dmft.combine_static_and_retarded_u(v_loc, u_weiss))
        fields['u_bar_0'] = float(np.real(u_bar[0]))
        if solver_params_list[0].get('retardation', 'dynamic') == 'static_u_zb':
            fields['z_b'] = float(coqui_dmft.casula_werner_zb(u_bar, nu))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        pass

    trail = ol.mott_chain_trail(**fields)
    ol.log_mott_chain(cycle, niter, trail, verbose=verbose)

    prev['sigma_dc'] = None if sigma_dc is None else np.array(sigma_dc, copy=True)
    prev['pi_dc'] = None if pi_dc is None else np.array(pi_dc, copy=True)
    prev['mo_skia'] = mo_skia
    prev['proj_mo_c'] = proj_mo_c
    return trail


def _save_mott_chain_trail(solver_chkpt_h5, trail):
    """Store the Q5-b trail (one row per outer cycle) in the impurity checkpoint."""
    if not trail:
        return
    try:
        with HDFArchive(solver_chkpt_h5, 'a') as ar:
            if "q5_outer_loop" not in ar.keys():
                ar.create_group("q5_outer_loop")
            grp = ar["q5_outer_loop"]
            grp["mott_chain_trail"] = np.asarray(trail, dtype=float)
            grp["mott_chain_labels"] = list(outer_loop_diag.MOTT_CHAIN_TRAIL_LABELS)
    # RuntimeError: TRIQS's h5 error class for an unwritable/unopenable archive.
    except (OSError, RuntimeError, KeyError) as e:
        coqui.app_log(1, f"[Q5-b] could not store the Mott-chain trail: {e}")


def _gw_loop(mf, h_int, proj_info,
             dmft_state, coqui_chkpt_h5,
             gw_params, embed_params, gw_iter_per_loop):
    if gw_iter_per_loop < 1:
        return

    coqui_mpi = h_int.mpi()
    for gw_iteration in range(gw_iter_per_loop):
        with HDFArchive(coqui_chkpt_h5, 'r') as ar:
            greens_func_source = "embed" if "embed" in ar.keys() else "scf"
            greens_func_iteration = ar[f"{greens_func_source}/final_iter"]
        coqui_mpi.barrier()

        # GW if gw_params presents
        gw_params["greens_func_source"] = greens_func_source
        gw_params["greens_func_iteration"] = greens_func_iteration
        coqui.run_gw(
            gw_params, h_int = h_int, projector_info = proj_info,
            local_polarizabilities = dmft_state.local_pi_w
        )
        coqui_mpi.barrier()

        # Don't upfold the results if gw_iter_per_loop==1. 
        # Not sure if this is the best choice, but it allows us to skip one upfolding in the common case of 
        # doing just one GW iteration per GW+EDMFT loop, which can save some disk space in the checkpoint h5.
        if gw_iter_per_loop > 1: 
            # Updates GW+EDMFT solution with the latest GW results while keeping the impurity solutions fixed.
            # Upfolding
            coqui.dmft_embed(
                mf, embed_params,
                projector_info = proj_info,
                local_hf_potentials = dmft_state.local_sigma_infty,
                local_sigma_dynamic = dmft_state.local_sigma_w
            )
            coqui_mpi.barrier()


def _edmft_loop(mf, h_int, proj_info, dmft_state, solver_chkpt_h5, coqui_chkpt_h5, 
               gloc_params, wloc_params, solver_params_list, embed_params,
               iterative_params, num_iter):

    coqui_mpi = mf.mpi()

    for iteration in range(num_iter):
        with HDFArchive(coqui_chkpt_h5, 'r') as ar:
            greens_func_source = "embed" if "embed" in ar.keys() else "scf"
            greens_func_iteration = ar[f"{greens_func_source}/final_iter"]

        # downfold for W_loc
        # greens_func_source and greens_func_iteration should be fixed during the inner loop
        Vloc, Wloc_t = coqui.downfold_coulomb(
            h_int, wloc_params,
            projector_info=proj_info,
            local_polarizabilities=dmft_state.local_pi_w
        )

        # downfold for G_loc
        gloc_params["greens_func_source"] = greens_func_source
        gloc_params["greens_func_iteration"] = greens_func_iteration
        Gloc_t = coqui.downfold_local_gf(mf, gloc_params, projector_info=proj_info)

        if coqui_mpi.root():
            dmft_state.iaft.check_leakage(Gloc_t, stats='f', name='Gloc in the full MLWF space')
            dmft_state.iaft.check_leakage_phsym(Wloc_t, stats='b', name='Wloc in the full MLWF space')

        # Convert spin axis → list of length nspin
        Vloc, Wloc_t = [Vloc], [Wloc_t]
        Gloc_t = [Gloc_t[:, s] for s in range(Gloc_t.shape[1])]
        if mf.nspin() == 1:
            Gloc_t = [Gloc_t[0], Gloc_t[0].copy()]
        
        # Extract local Green's function and screened interactions for each impurity
        Gloc_C    = dmft_state.embedding['1e'].extract(Gloc_t)   # block matrix
        Vloc_C    = dmft_state.embedding['2e'].extract(Vloc)
        Wloc_C    = dmft_state.embedding['2e'].extract(Wloc_t)
        Vloc_C    = [ V[0] for V in Vloc_C ]      # spinless
        Wloc_C    = [ W_t[0] for W_t in Wloc_C ]  # spinless

        for imp_index, (G_t, W_t, V) in enumerate(zip(Gloc_C, Wloc_C, Vloc_C)):
            coqui_dmft.print_title_box(f"IMPURITY {imp_index}")

            solver_params = solver_params_list[imp_index]
            Res, Input = dmft_state.solver_results[imp_index], dmft_state.solver_inputs[imp_index]
            Input['Gloc_t'] = G_t
            Input['Wloc_t'] = coqui_dmft.chemistry_to_product_basis(W_t)
            Input['Vloc'] = coqui_dmft.chemistry_to_product_basis(V)

            coqui_dmft.fit_local_results_boson(
                Input, dmft_state.iaft, solver_params.get("causal_projection")
            )

            # Save previous Weiss fields before updating (for convergence check)
            prev_g_weiss_iw = Input.get('g_weiss_iw')
            prev_u_weiss_iw = Input.get('u_weiss_iw')

            # Fermionic and bosonic Weiss fields
            Input['g_weiss_iw'], Input['u_weiss_iw'] = _compute_weiss_fields(
                coqui_mpi, Res, Input, solver_params, dmft_state.iaft
            )
            Input['u_weiss_iw'] = coqui_dmft.fit_u_weiss(
                Input['u_weiss_iw'], dmft_state.iaft, solver_params.get("causal_projection")
            )

            # Q4-c causality monitor: meters U(inu) = Vloc + u_weiss_iw AFTER the causal
            # projection of fit_u_weiss (bath_fit.py:175-190), so it reports the object the
            # solver actually receives. Non-fatal.
            causality = coqui_dmft.monitor_u_causality(
                Input['Vloc'], Input['u_weiss_iw'], _bosonic_nu_mesh(dmft_state.iaft),
                imp_index=imp_index, verbose=coqui_mpi.root()
            )

            # h0: (nspin, norb, norb), delta_iw: (niw, nspin, norb, norb)
            Input['h0'], Input['delta_iw'] = coqui_dmft.extract_h0_and_delta(
                Input['g_weiss_iw'], dmft_state.iaft
            )

            # Impurity retardation policy (R-Q4-5). "dynamic" (default) is a pass-through.
            delta_iw_solver, Vloc_solver, u_weiss_iw_solver, z_b = \
                coqui_dmft.apply_impurity_retardation_mode(
                    Input['delta_iw'], Input['u_weiss_iw'], Input['Vloc'],
                    _bosonic_nu_mesh(dmft_state.iaft),
                    mode=solver_params.get('retardation', 'dynamic'),
                    static_u_source=solver_params.get('static_u_source', 'u0'),
                    name=f"impurity {imp_index}"
                )

            Ub, Ubp, Jb_spin, Jb_pair = coqui_dmft.hubbard_kanamori_coulomb(Input['Vloc'])
            U, Up, J_spin, J_pair = coqui_dmft.hubbard_kanamori_coulomb(Input['Vloc']+Input['u_weiss_iw'][0])
            gloc_t_arr = coqui_dmft.blk_arr_to_arr(Input['Gloc_t'], Input["gf_struct"])
            dm = -dmft_state.iaft.tau_interpolate(gloc_t_arr, dmft_state.iaft.beta, 'f')[0]
            g_beta_half = -dmft_state.iaft.tau_interpolate(gloc_t_arr, dmft_state.iaft.beta/2, 'f')[0]
            Input['density'] = (np.diag(dm[0]).sum() + np.diag(dm[1]).sum()).real
            coqui.app_log(1, "Bare/static orbital-averaged interactions for the impurity")
            coqui.app_log(1, "----------------------------------------------------------")
            coqui.app_log(1, f"  intra-orbital                  = {Ub*Hartree_eV:.4f}, {U*Hartree_eV:.4f} eV")
            coqui.app_log(1, f"  inter-orbital                  = {Ubp*Hartree_eV:.4f}, {Up*Hartree_eV:.4f} eV")
            coqui.app_log(1, f"  Hund's coupling (spin-flip)    = {Jb_spin*Hartree_eV:.4f}, {J_spin*Hartree_eV:.4f} eV")
            coqui.app_log(1, f"  Hund's coupling (pair-hopping) = {Jb_pair*Hartree_eV:.4f}, {J_pair*Hartree_eV:.4f} eV\n")

            # For metals, beta * G(beta/2) saturates with increasing beta
            # For insulators, beta * G(beta/2) decays exponentially to zero with increasing beta
            coqui.app_log(1, "Spectral weight proxy at Fermi level (eV^-1): A(w) ~ -beta/pi * G_loc(tau=beta/2)")
            coqui.app_log(1, "--------------------------------------------------------------")
            coqui.app_log(1, f"  Spin up:   {(dmft_state.iaft.beta/np.pi) * np.diag(g_beta_half[0]).real / Hartree_eV}")
            coqui.app_log(1, f"  Spin down: {(dmft_state.iaft.beta/np.pi) * np.diag(g_beta_half[1]).real / Hartree_eV}\n")

            coqui.app_log(1, "Local densities ")
            coqui.app_log(1, "-------------------")
            coqui.app_log(1, f"  Total: {Input['density']:.4f}")
            coqui.app_log(1, f"  Spin up: {np.diag(dm[0]).real}")
            coqui.app_log(1, f"  Spin down: {np.diag(dm[1]).real}\n")

            dmft_state.save_impurity_inputs(solver_chkpt_h5, imp_index)

            # Convert CoQuí outputs to TRIQS containers
            h0, delta_iw, h_int, u_weiss_iw = coqui_dmft.to_triqs_containers(
                Input['h0'], delta_iw_solver, Vloc_solver, u_weiss_iw_solver,
                dmft_state.iaft, gf_struct = Res['gf_struct'],
                triqs_iw_mesh = {"fermion": Res['iw_mesh_f'], "boson": Res['iw_mesh_b']},
                density_hamiltonian = True, real_hamiltonian = True,
                screen_j_in_u_dd = solver_params.get('screen_j', False)
            )

            # Analyze block symmetry
            if solver_params.get('degenerate_blk') is None and solver_params.get('degenerate_blk_thresh'):
                coqui.app_log(2, "Analyzing block symmetries via the hybridization function...\n")
                # Cache the result so subsequent EDMFT iterations skip re-analysis
                solver_params['degenerate_blk'] = modest.analyze_degenerate_blocks(
                    delta_iw, threshold=solver_params['degenerate_blk_thresh']
                )
            degenerate_blk = solver_params.get('degenerate_blk')
            if degenerate_blk is not None:
                coqui_dmft.print_degenerate_blks(degenerate_blk, Res['gf_struct'])
                delta_iw   = modest.symmetrize(delta_iw, degenerate_blk)
                h0         = coqui_dmft.symmetrize_h0_op(h0, degenerate_blk, Res['gf_struct'])
                h_int      = coqui_dmft.symmetrize_h_int_op(h_int, degenerate_blk, Res['gf_struct'])
                u_weiss_iw = coqui_dmft.symmetrize_blk2_gf(u_weiss_iw, degenerate_blk, Res['gf_struct'])

            # Call impurity solver, and store sigma_imp, vhf_imp, and pi_imp in "Res"
            Res.update(
                _solver_inner_loop(coqui_mpi, h0, delta_iw, u_weiss_iw, h_int, Input['density'], **solver_params)
            )
            # convert from triqs Gf to numpy arrays and ir mesh
            Res.update(coqui_dmft.imp_results_to_raw_data(
                Res['G_iw'], Res['Sigma_iw'], Res['W_iw'], Res['Pi_iw'], dmft_state.iaft)
            )
            # Causal projection
            coqui_dmft.fit_impurity_results_boson(
                Res, dmft_state.iaft, solver_params.get("causal_projection"))

            conv_metrics = _edmft_convergence_check(coqui_mpi, imp_index, Input, Res, dmft_state.iaft,
                                                    prev_g_weiss_iw, prev_u_weiss_iw)

            # Store convergence metrics array: [U_w0, A_w0, diff_g, diff_g_weiss, diff_u_weiss, diff_w, Sigma_w1]
            if coqui_mpi.root():
                A_w0  = float(np.mean((dmft_state.iaft.beta / np.pi) * np.diag(g_beta_half[0]+g_beta_half[1]).real)) * 0.5

                w1_idx = np.where(dmft_state.iaft.wn_mesh(stats='fermion') == 1)[0][0]
                sigma_w0 = 0.0
                norb_tot = 0
                for sigma_blk in Res['Sigma_iw_data']:
                    sigma_w0 += np.sum(np.diag(sigma_blk[w1_idx]).imag)
                    norb_tot += sigma_blk.shape[1]  # number of orbitals in the block
                sigma_w0 = sigma_w0 / norb_tot  # average over all orbitals

                Sigma_imp_iw0 = Res['Sigma_iw_data']
                Res['convergence'] = np.array([
                    U, A_w0,
                    conv_metrics['diff_g'], conv_metrics['diff_g_weiss'],
                    conv_metrics['diff_u_weiss'], conv_metrics['diff_w'],
                    sigma_w0
                ])
                # Q4-c trail: [hermiticity_max, dd_monotonicity_flips,
                #              min eig(U(0)-U(inu_max)), max eig(U(0)-U(inu_max))]
                Res['causality'] = coqui_dmft.causality_trail(causality)

            # GW double counting contributions (current implementation uses Gloc/Wloc inputs)
            Res.update(
                coqui_dmft.solve_gw_dc(
                    coqui_dmft.blk_arr_to_arr(Input['Gloc_t'], Res['gf_struct']),
                    Input['Vloc'], Input['Wloc_t'], Input['u_weiss_iw'],
                    dmft_state.iaft, density_only=True,
                    gf_struct=Res['gf_struct']
                )
            )

            # mixing impurity and dc solutions to facilitate convergence
            dmft_state.damp_impurity_results(
                solver_chkpt_h5, mixing=iterative_params.get('mixing', 0.7), impurity_indices=[imp_index],
                mix_in_first_iter=iterative_params.get('mix_in_first_iter', False)
            )

            # save solver results for current impurity
            dmft_state.save_impurity_results(solver_chkpt_h5, imp_index)
            
        # Embed impurity results
        dmft_state.embed_impurity_results()

        # Upfolding
        coqui.dmft_embed(
            mf, embed_params,
            projector_info = proj_info,
            local_hf_potentials = dmft_state.local_sigma_infty,
            local_sigma_dynamic = dmft_state.local_sigma_w
        )
        dmft_state.iteration += 1
        coqui_mpi.barrier()


def _edmft_loop_fixed_gloc_and_wloc(
        mf, h_int, proj_info, dmft_state, solver_chkpt_h5, coqui_chkpt_h5, 
        gloc_params, wloc_params, solver_params_list, embed_params,
        iterative_params, num_iter):
    coqui_mpi = mf.mpi()
    with HDFArchive(coqui_chkpt_h5, 'r') as ar:
        greens_func_source = "embed" if "embed" in ar.keys() else "scf"
        greens_func_iteration = ar[f"{greens_func_source}/final_iter"]

    # downfold for W_loc
    # greens_func_source and greens_func_iteration should be fixed during the inner loop
    Vloc, Wloc_t = coqui.downfold_coulomb(
        h_int, wloc_params,
        projector_info=proj_info,
        local_polarizabilities=dmft_state.local_pi_w
    )

    # downfold for G_loc
    gloc_params["greens_func_source"] = greens_func_source
    gloc_params["greens_func_iteration"] = greens_func_iteration
    Gloc_t = coqui.downfold_local_gf(mf, gloc_params, projector_info=proj_info)

    if coqui_mpi.root():
        dmft_state.iaft.check_leakage(Gloc_t, stats='f', name='Gloc in the full MLWF space')
        dmft_state.iaft.check_leakage_phsym(Wloc_t, stats='b', name='Wloc in the full MLWF space')

    # Convert spin axis → list of length nspin
    Gloc_t = [Gloc_t[:, s] for s in range(Gloc_t.shape[1])]
    if mf.nspin() == 1:
        Gloc_t = [Gloc_t[0], Gloc_t[0].copy()]

    # Extract local Green's function and screened interactions for each impurity
    Gloc_C    = dmft_state.embedding['1e'].extract(Gloc_t)   # block matrix
    Vloc_C    = dmft_state.embedding['2e'].extract([Vloc])    # (norb, norb, norb, norb)
    Wloc_C    = dmft_state.embedding['2e'].extract([Wloc_t]) # (nts, norb, norb, norb, norb)
    Vloc_C    = [ V[0] for V in Vloc_C ]      # spinless
    Wloc_C    = [ W_t[0] for W_t in Wloc_C ]  # spinless

    for iteration in range(num_iter):
        for imp_index, (G_t, W_t, V) in enumerate(zip(Gloc_C, Wloc_C, Vloc_C)):
            coqui_dmft.print_title_box(f"IMPURITY {imp_index}")

            solver_params = solver_params_list[imp_index]
            Res, Input = dmft_state.solver_results[imp_index], dmft_state.solver_inputs[imp_index]
            Input['Gloc_t'] = G_t
            Input['Wloc_t'] = coqui_dmft.chemistry_to_product_basis(W_t)
            Input['Vloc'] = coqui_dmft.chemistry_to_product_basis(V)

            coqui_dmft.fit_local_results_boson(
                Input, dmft_state.iaft, solver_params.get("causal_projection"))

            # Save previous Weiss fields before updating (for convergence check)
            prev_g_weiss_iw = Input.get('g_weiss_iw')
            prev_u_weiss_iw = Input.get('u_weiss_iw')

            # Fermionic and bosonic Weiss fields
            Input['g_weiss_iw'], Input['u_weiss_iw'] = _compute_weiss_fields(
                coqui_mpi, Res, Input, solver_params, dmft_state.iaft
            )
            Input['u_weiss_iw'] = coqui_dmft.fit_u_weiss(
                Input['u_weiss_iw'], dmft_state.iaft, solver_params.get("causal_projection")
            )

            # Q4-c causality monitor (see _edmft_loop for the seam note). Non-fatal.
            causality = coqui_dmft.monitor_u_causality(
                Input['Vloc'], Input['u_weiss_iw'], _bosonic_nu_mesh(dmft_state.iaft),
                imp_index=imp_index, verbose=coqui_mpi.root()
            )

            # h0: (nspin, norb, norb), delta_iw: (niw, nspin, norb, norb)
            Input['h0'], Input['delta_iw'] = coqui_dmft.extract_h0_and_delta(
                Input['g_weiss_iw'], dmft_state.iaft
            )

            # Impurity retardation policy (R-Q4-5). "dynamic" (default) is a pass-through.
            delta_iw_solver, Vloc_solver, u_weiss_iw_solver, z_b = \
                coqui_dmft.apply_impurity_retardation_mode(
                    Input['delta_iw'], Input['u_weiss_iw'], Input['Vloc'],
                    _bosonic_nu_mesh(dmft_state.iaft),
                    mode=solver_params.get('retardation', 'dynamic'),
                    static_u_source=solver_params.get('static_u_source', 'u0'),
                    name=f"impurity {imp_index}"
                )

            Ub, Ubp, Jb_spin, Jb_pair = coqui_dmft.hubbard_kanamori_coulomb(Input['Vloc'])
            U, Up, J_spin, J_pair = coqui_dmft.hubbard_kanamori_coulomb(Input['Vloc']+Input['u_weiss_iw'][0])
            dm = -dmft_state.iaft.tau_interpolate(
                coqui_dmft.blk_arr_to_arr(Input['Gloc_t'], Input["gf_struct"]),
                dmft_state.iaft.beta, 'f')[0]
            g_beta_half = -dmft_state.iaft.tau_interpolate(
                coqui_dmft.blk_arr_to_arr(Input['Gloc_t'], Input["gf_struct"]),
                dmft_state.iaft.beta/2, 'f')[0]
            Input['density'] = (np.diag(dm[0]).sum() + np.diag(dm[1]).sum()).real
            coqui.app_log(1, "Bare/static orbital-averaged interactions for the impurity")
            coqui.app_log(1, "----------------------------------------------------------")
            coqui.app_log(1, f"  intra-orbital                  = {Ub*Hartree_eV:.4f}, {U*Hartree_eV:.4f} eV")
            coqui.app_log(1, f"  inter-orbital                  = {Ubp*Hartree_eV:.4f}, {Up*Hartree_eV:.4f} eV")
            coqui.app_log(1, f"  Hund's coupling (spin-flip)    = {Jb_spin*Hartree_eV:.4f}, {J_spin*Hartree_eV:.4f} eV")
            coqui.app_log(1, f"  Hund's coupling (pair-hopping) = {Jb_pair*Hartree_eV:.4f}, {J_pair*Hartree_eV:.4f} eV\n")

            # For metals, beta * G(beta/2) saturates with increasing beta
            # For insulators, beta * G(beta/2) decays exponentially to zero with increasing beta
            coqui.app_log(1, "Spectral weight proxy at Fermi level: A(w) ~ -beta/pi * G_loc(tau=beta/2)")
            coqui.app_log(1, "---------------------------------------------------------")
            coqui.app_log(1, f"  Spin up:   {(dmft_state.iaft.beta/np.pi) * np.diag(g_beta_half[0]).real}")
            coqui.app_log(1, f"  Spin down: {(dmft_state.iaft.beta/np.pi) * np.diag(g_beta_half[1]).real}\n")

            coqui.app_log(1, "Local densities ")
            coqui.app_log(1, "-------------------")
            coqui.app_log(1, f"Total: {Input['density']:.4f}")
            coqui.app_log(1, f"Spin up: {np.diag(dm[0]).real}")
            coqui.app_log(1, f"Spin down: {np.diag(dm[1]).real}\n")

            dmft_state.save_impurity_inputs(solver_chkpt_h5, imp_index)

            # Convert CoQuí outputs to TRIQS containers
            h0, delta_iw, h_int, u_weiss_iw = coqui_dmft.to_triqs_containers(
                Input['h0'], delta_iw_solver, Vloc_solver, u_weiss_iw_solver,
                dmft_state.iaft, gf_struct = Res['gf_struct'],
                triqs_iw_mesh = {"fermion": Res['iw_mesh_f'], "boson": Res['iw_mesh_b']},
                density_hamiltonian = True, real_hamiltonian = True,
                screen_j_in_u_dd = solver_params.get('screen_j', False)
            )

            # Analyze block symmetry
            if solver_params.get('degenerate_blk') is None and solver_params.get('degenerate_blk_thresh'):
                coqui.app_log(2, "Analyzing block symmetries via the hybridization function...\n")
                # Cache the result so subsequent EDMFT iterations skip re-analysis
                solver_params['degenerate_blk'] = modest.analyze_degenerate_blocks(
                    delta_iw, threshold=solver_params['degenerate_blk_thresh']
                )
            degenerate_blk = solver_params.get('degenerate_blk') 
            if degenerate_blk is not None:
                coqui_dmft.print_degenerate_blks(degenerate_blk, Res['gf_struct'])
                delta_iw   = modest.symmetrize(delta_iw, degenerate_blk)
                h0         = coqui_dmft.symmetrize_h0_op(h0, degenerate_blk, Res['gf_struct'])
                h_int      = coqui_dmft.symmetrize_h_int_op(h_int, degenerate_blk, Res['gf_struct'])
                u_weiss_iw = coqui_dmft.symmetrize_blk2_gf(u_weiss_iw, degenerate_blk, Res['gf_struct'])

            # Call impurity solver, and store sigma_imp, vhf_imp, and pi_imp in "Res"
            Res.update(
                _solver_inner_loop(coqui_mpi, h0, delta_iw, u_weiss_iw, h_int, Input['density'], **solver_params)
            )
            # convert from triqs Gf to numpy arrays and ir mesh
            Res.update(coqui_dmft.imp_results_to_raw_data(
                Res['G_iw'], Res['Sigma_iw'], Res['W_iw'], Res['Pi_iw'], dmft_state.iaft)
            )
            # Causal projection
            coqui_dmft.fit_impurity_results_boson(
                Res, dmft_state.iaft, solver_params.get("causal_projection"))

            conv_metrics = _edmft_convergence_check(coqui_mpi, imp_index, Input, Res, dmft_state.iaft,
                                                      prev_g_weiss_iw, prev_u_weiss_iw)

            # Store convergence metrics array: [U_w0, A_w0, diff_g, diff_g_weiss, diff_u_weiss, diff_w]
            if coqui_mpi.root():
                A_w0  = float(np.mean((dmft_state.iaft.beta / np.pi) * np.diag(g_beta_half[0]+g_beta_half[1]).real)) * 0.5
                Res['convergence'] = np.array([
                    U, A_w0,
                    conv_metrics['diff_g'], conv_metrics['diff_g_weiss'],
                    conv_metrics['diff_u_weiss'], conv_metrics['diff_w']
                ])
                # Q4-c trail: [hermiticity_max, dd_monotonicity_flips,
                #              min eig(U(0)-U(inu_max)), max eig(U(0)-U(inu_max))]
                Res['causality'] = coqui_dmft.causality_trail(causality)

            # GW double counting contributions
            Res.update(
                coqui_dmft.solve_gw_dc(
                    coqui_dmft.blk_arr_to_arr(Input['Gloc_t'], Res['gf_struct']),
                    Input['Vloc'], Input['Wloc_t'], Input['u_weiss_iw'],
                    dmft_state.iaft, density_only=True,
                    gf_struct=Res['gf_struct']
                )
            )

            # mixing impurity and dc solutions to facilitate convergence
            dmft_state.damp_impurity_results(
                solver_chkpt_h5, mixing = iterative_params.get('mixing', 0.7), impurity_indices=[imp_index],
                mix_in_first_iter=iterative_params.get('mix_in_first_iter', True)
            )

            # save solver results for current impurity
            dmft_state.save_impurity_results(solver_chkpt_h5, imp_index)

        # Embed impurity results
        dmft_state.embed_impurity_results()

        dmft_state.iteration += 1
        coqui_mpi.barrier()

    # Upfolding
    coqui.dmft_embed(
        mf, embed_params,
        projector_info = proj_info,
        local_hf_potentials = dmft_state.local_sigma_infty,
        local_sigma_dynamic = dmft_state.local_sigma_w
    )
    coqui_mpi.barrier()


def _bosonic_nu_mesh(iaft):
    """
    Non-negative bosonic Matsubara frequencies nu_n = 2*pi*n/beta matching the
    ph-symmetric arrays produced by ``iaft.tau_to_w_phsym(..., stats='b')``.

    Same construction as ``bath_fit.causal_projection_boson`` (bath_fit.py:105).
    """
    return iaft.wn_mesh('b', positive_only=True) * np.pi / iaft.beta


def _compute_weiss_fields(coqui_mpi, imp_results, imp_inputs, solver_params, iaft):

    gloc_t_mat = coqui_dmft.blk_arr_to_arr(imp_inputs['Gloc_t'], imp_inputs['gf_struct'])

    if imp_results['Sigma_iw_data'] is not None:
        if solver_params.get('set_sigma_infty_to_dc', False):
            vhf_imp =  imp_results['Sigma_infty_dc']
        else:
            vhf_imp =  imp_results['Sigma_infty']

        if imp_inputs['screen_type'] == 'rpa':
            coqui.app_log(2, "screen_type = \"rpa\" -> "
                             "Set impurity polarizability to RPA for bosonic Weiss field.\n")
            # eval Pi_dc using the current Gloc
            pi_imp = iaft.tau_to_w_phsym(
                coqui_dmft.eval_pi_rpa(gloc_t_mat, density_only=True), stats='b'
            )
        else:
            pi_imp = imp_results['Pi_iw_data'][0] if imp_results['Pi_iw_data'] else None

        return (
            coqui_dmft.compute_weiss_fields_w(
                iaft = iaft,
                local_gf = {
                    "Gloc_t": gloc_t_mat,
                    "Wloc_t": imp_inputs['Wloc_t'],
                    "Vloc": imp_inputs['Vloc']
                },
                impurity_selfenergies = {
                    "Vhf_imp": coqui_dmft.blk_arr_to_arr(vhf_imp, imp_results['gf_struct']),
                    "Sigma_imp_w": coqui_dmft.blk_arr_to_arr(imp_results['Sigma_iw_data'], imp_results['gf_struct']),
                    "Pi_imp_w": pi_imp
                },
                density_only=True
            )
        )
    else:
        return (
            coqui_dmft.init_weiss_fields_w(
                iaft = iaft,
                local_gf = {
                    "Gloc_t": gloc_t_mat,
                    "Wloc_t": imp_inputs['Wloc_t'],
                    "Vloc": imp_inputs['Vloc']
                },
                init_imp_results = solver_params.get('init_imp_results', 'dc'),
                density_only=True
            )
        )


def _solver_inner_loop(coqui_mpi, h0, delta_iw, u_weiss_iw, h_int,
                       target_density, **solver_params):

    solver_params.pop('degenerate_blk_thresh', None)
    solver_params.pop('set_sigma_infty_to_dc', None)
    solver_params.pop('init_imp_results', None)
    solver_params.pop("causal_projection", None)
    solver_params.pop("screen_j", None)
    # Q4 R-Q4-5: consumed by apply_impurity_retardation_mode before the solver call.
    solver_params.pop("retardation", None)
    solver_params.pop("static_u_source", None)
    mu_params = solver_params.pop('chemical_potential', None)

    if mu_params is not None:
        dens_solver_params = solver_params.copy()
        dens_solver_params['verbosity'] = 0
        dens_solver_params['suppress_solver_output'] = mu_params.get('suppress_solver_output', True)
        if mu_params.get('solver_output_file'):
            dens_solver_params['solver_output_file'] = mu_params.get('solver_output_file')
        if mu_params.get('n_warmup_cycles'):
            dens_solver_params['n_warmup_cycles'] = mu_params.get('n_warmup_cycles')
        if mu_params.get('length_cycle'):
            dens_solver_params['length_cycle'] = mu_params.get('length_cycle')
        dens_solver_params['n_cycles'] = mu_params.get('n_cycles', solver_params.get('n_cycles')*0.05)

        gf_struct = [(bl, gf.target_shape[0]) for (bl, gf) in delta_iw]
        h0_sab = coqui_dmft.h0_operator_to_array(h0, gf_struct)
        compute_nelec_fcn = partial(
            coqui_dmft.compute_nelec_from_solver,
            gf_struct=gf_struct, h0_sab=h0_sab,
            delta_iw=delta_iw, u_weiss_iw=u_weiss_iw, h_int=h_int,
            **dens_solver_params
        )
        mu_imp, imp_density = coqui_dmft.compute_mu_impurity(
            target_density, compute_nelec_fcn,
            tolerance=mu_params.get('tolerance', 0.1), 
            mu0=0.0, # always start from mu=0 s.t. mu_imp falls back to 0 at convergence
        )
        # update h0 = h0 - mu_imp
        h0_mat_shifted = np.array([ h0_mat - np.eye(h0_mat.shape[0])*mu_imp for h0_mat in h0_sab ])
        h0 = coqui_dmft.h0_operator(h0_mat_shifted, gf_struct, force_real=True)
    else:
        mu_imp = 0.0

    solver_results = coqui_dmft.ctseg.solve_dynamic_imp(delta_iw, h0, u_weiss_iw, h_int, **solver_params)
    solver_results['mu_imp'] = mu_imp
    # impurity total density
    imp_density = 0.0
    for blk_name, occ in solver_results['orbital_occupations'].items():
        imp_density += occ.sum()
    solver_results['density'] = imp_density

    coqui.app_log(1, f"Total impurity densities = {imp_density}")
    coqui.app_log(1, f"Convergence of impurity density: {imp_density - target_density}\n")

    return solver_results


def _edmft_convergence_check(coqui_mpi, imp_index, Input, Res, iaft,
                              prev_g_weiss_iw=None, prev_u_weiss_iw=None):
    """Compute EDMFT self-consistency metrics and return them as a dict.

    Returns
    -------
    dict with keys:
        diff_g       : |Gloc_tau - Gimp_tau| (always computed)
        diff_g_weiss : |g_weiss_tau - g_weiss_prev_tau| (-1.0 if prev not available)
        diff_w       : |Wloc_tau - Wimp_tau| dd (-1.0 if W_iw_data is None)
        diff_u_weiss : |u_weiss_tau - u_weiss_prev_tau| dd (-1.0 if prev not available)
    """
    metrics = {'diff_g': -1.0, 'diff_g_weiss': -1.0, 'diff_w': -1.0, 'diff_u_weiss': -1.0}

    if not coqui_mpi.root():
        return metrics

    gf_struct = Res['gf_struct']

    # |Gloc - Gimp| on the imaginary-time axis
    gloc_t      = coqui_dmft.blk_arr_to_arr(Input['Gloc_t'], gf_struct)
    gimp_iw_mat = coqui_dmft.blk_arr_to_arr(Res['G_iw_data'], gf_struct)
    gimp_t      = iaft.w_to_tau(gimp_iw_mat, stats='f')
    norm_grid   = np.linalg.norm(gloc_t - gimp_t, axis=tuple(range(2, gloc_t.ndim)))
    metrics['diff_g'] = float(np.max(np.abs(norm_grid)))

    coqui.app_log(1, f"EDMFT self-consistency check for impurity {imp_index}:")
    coqui.app_log(1, f"  |Gloc_tau - Gimp_tau|                   = {metrics['diff_g']}")

    if prev_g_weiss_iw is not None:
        g_weiss_t      = iaft.w_to_tau(Input['g_weiss_iw'], stats='f')
        g_weiss_prev_t = iaft.w_to_tau(prev_g_weiss_iw, stats='f')
        norm_grid_gw   = np.linalg.norm(g_weiss_t - g_weiss_prev_t,
                                        axis=tuple(range(2, g_weiss_t.ndim)))
        metrics['diff_g_weiss'] = float(np.max(np.abs(norm_grid_gw)))
        coqui.app_log(1, f"  |g_weiss_tau - g_weiss_prev_tau|        = {metrics['diff_g_weiss']}")

    if Res['W_iw_data'] is not None:
        # |Wloc - Wimp| restricted to density-density components (always computed)
        wloc_dd  = coqui_dmft.product_basis_to_density_density(Input['Wloc_t'])
        wimp_raw = iaft.w_to_tau_phsym(Res["W_iw_data"][0], stats='b')
        wimp_dd  = wimp_raw if wimp_raw.ndim == 3 else coqui_dmft.product_basis_to_density_density(wimp_raw)
        norm_grid_w = np.linalg.norm(wloc_dd - wimp_dd, axis=tuple(range(1, wloc_dd.ndim)))
        metrics['diff_w'] = float(np.max(np.abs(norm_grid_w)))
        # Only print when screen_type != 'rpa' since diff_w is not expected to converge there
        if Input['screen_type'] != 'rpa':
            coqui.app_log(1, f"  |Wloc_tau - Wimp_tau| (density-density) = {metrics['diff_w']}")

    if prev_u_weiss_iw is not None:
        u_weiss_t      = iaft.w_to_tau_phsym(Input['u_weiss_iw'], stats='b')
        u_weiss_prev_t = iaft.w_to_tau_phsym(prev_u_weiss_iw, stats='b')
        if u_weiss_t.ndim == 3:
            u_weiss_dd      = u_weiss_t
            u_weiss_prev_dd = u_weiss_prev_t
        else:
            u_weiss_dd      = coqui_dmft.product_basis_to_density_density(u_weiss_t)
            u_weiss_prev_dd = coqui_dmft.product_basis_to_density_density(u_weiss_prev_t)
        norm_grid_uw = np.linalg.norm(u_weiss_dd - u_weiss_prev_dd, axis=tuple(range(1, u_weiss_dd.ndim)))
        metrics['diff_u_weiss'] = float(np.max(np.abs(norm_grid_uw)))
        coqui.app_log(1, f"  |u_weiss_tau - u_weiss_prev_tau| (dd)   = {metrics['diff_u_weiss']}")

    coqui.app_log(1, "")
    return metrics


def solve_impurities_from_chkpt(coqui_mpi, *, dmft_iteration=-1, imp_indices=None, 
                                params: dict, save_results=False):
    """
    Re-solve EDMFT impurity problems from a saved checkpoint.

    This helper reads previously stored impurity inputs from ``chkpt_h5`` and
    reruns only the impurity-solver stage, without performing GW updates,
    downfolding, or embedding/upfolding. It accepts the same flat GW+EDMFT
    parameter layout as :func:`run_gw_edmft`, but only the EDMFT/impurity
    subsection is used.

    Parameters
    ----------
    coqui_mpi : coqui.MpiHandler
        MPI handler used for communicator size, barriers, and root-only output.
    dmft_iteration : int, optional
        DMFT iteration to read from the impurity checkpoint.

        - ``-1``: use the latest stored iteration.
        - ``>= 0``: read the specified iteration explicitly.

        Default is ``-1``.
    imp_indices : list[int], optional
        Subset of impurity indices to solve. If ``None``, all impurities stored
        in the checkpoint are processed.
    params : dict
        GW+EDMFT parameter dictionary in the same format as :func:`run_gw_edmft`.
        For the ``edmft`` subsection, including ``edmft.impurity`` solver
        options, refer directly to the docstring of :func:`run_gw_edmft`.

        In this helper, only the EDMFT/impurity part is used. In particular, 
        ``edmft.chkpt_h5`` selects the impurity checkpoint to read, ``edmft.iaft`` 
        may override the impurity DLR mesh, and ``edmft.impurity`` provides 
        the solver settings for the re-solve.

    Returns
    -------
    list[dict]
        List of impurity solver result dictionaries, one per processed impurity,
        containing TRIQS solver outputs together with raw-data conversions such
        as ``Sigma_iw_data``, ``Pi_iw_data``, and ``W_iw_data``.
    """

    params = convert_gw_edmft_params(params)
    
    imp_params = params.pop('impurity')
    # Scale Monte-Carlo cycle counts by MPI communicator size.
    solver_params_list = _normalize_solver_params_list(
        imp_params['solver'], coqui_mpi.comm_size()
    )

    iaft = IAFT.from_coqui_chkpt(imp_params['chkpt_h5'], verbose=coqui_mpi.root())
    imp_iaft_params = imp_params.pop('iaft', {})

    solver_inputs = coqui_dmft.read_impurity_chkpt(
        imp_params['chkpt_h5'], dmft_iteration, read="inputs", impurity_indices=imp_indices
    )
    solver_results = []
    for imp_index, inputs in enumerate(solver_inputs):
        coqui_dmft.print_title_box(f"IMPURITY {imp_index}")
        solver_params = solver_params_list[imp_index]
        Input = solver_inputs[imp_index]

        Input['u_weiss_iw'] = coqui_dmft.fit_u_weiss(
            Input['u_weiss_iw'], iaft, solver_params.get("causal_projection")
        )

        Ub, Ubp, Jb_spin, Jb_pair = coqui_dmft.hubbard_kanamori_coulomb(Input['Vloc'])
        U, Up, J_spin, J_pair = coqui_dmft.hubbard_kanamori_coulomb(Input['Vloc']+Input['u_weiss_iw'][0])
        dm = -iaft.tau_interpolate(
            coqui_dmft.blk_arr_to_arr(Input['Gloc_t'], Input["gf_struct"]),
            iaft.beta, 'f')[0]
        Input['density'] = (np.diag(dm[0]).sum() + np.diag(dm[1]).sum()).real
        coqui.app_log(1, "Bare/static orbital-averaged interactions for the impurity")
        coqui.app_log(1, "----------------------------------------------------------")
        coqui.app_log(1, f"  intra-orbital                  = {Ub*Hartree_eV:.4f}, {U*Hartree_eV:.4f} eV")
        coqui.app_log(1, f"  inter-orbital                  = {Ubp*Hartree_eV:.4f}, {Up*Hartree_eV:.4f} eV")
        coqui.app_log(1, f"  Hund's coupling (spin-flip)    = {Jb_spin*Hartree_eV:.4f}, {J_spin*Hartree_eV:.4f} eV")
        coqui.app_log(1, f"  Hund's coupling (pair-hopping) = {Jb_pair*Hartree_eV:.4f}, {J_pair*Hartree_eV:.4f} eV\n")

        coqui.app_log(1, "Local densities ")
        coqui.app_log(1, "-------------------")
        coqui.app_log(1, f"Total: {Input['density']:.4f}")
        coqui.app_log(1, f"Spin up: {np.diag(dm[0]).real}")
        coqui.app_log(1, f"Spin down: {np.diag(dm[1]).real}\n")

        if coqui_mpi.root():
            iaft.check_leakage(Input['delta_iw'], 'f', 'delta', w_input=True)
            iaft.check_leakage_phsym(Input['u_weiss_iw'], 'b', 'u_weiss', w_input=True)

        # Convert CoQuí outputs to TRIQS containers
        h0, delta_iw, h_int, u_weiss_iw = coqui_dmft.to_triqs_containers(
            Input['h0'], Input['delta_iw'], Input['Vloc'], Input['u_weiss_iw'],
            iaft, gf_struct = Input['gf_struct'],
            triqs_iw_mesh = {
                "dlr_wmax": imp_iaft_params.get('wmax', iaft.wmax), 
                "dlr_eps": imp_iaft_params.get('eps', iaft.eps)
            },
            density_hamiltonian = True, real_hamiltonian = True,
            screen_j_in_u_dd = solver_params.get('screen_j', False)
        )

        # Analyze block symmetry
        if solver_params.get('degenerate_blk') is None and solver_params.get('degenerate_blk_thresh'):
            coqui.app_log(2, "Analyzing block symmetries via the hybridization function...\n")
            solver_params['degenerate_blk'] = modest.analyze_degenerate_blocks(
                delta_iw, threshold=solver_params['degenerate_blk_thresh']
            )
        degenerate_blk = solver_params.get('degenerate_blk')
        if degenerate_blk is not None:
            coqui_dmft.print_degenerate_blks(degenerate_blk, Input['gf_struct'])
            delta_iw   = modest.symmetrize(delta_iw, degenerate_blk)
            h0         = coqui_dmft.symmetrize_h0_op(h0, degenerate_blk, Input['gf_struct'])
            h_int      = coqui_dmft.symmetrize_h_int_op(h_int, degenerate_blk, Res['gf_struct'])
            u_weiss_iw = coqui_dmft.symmetrize_blk2_gf(u_weiss_iw, degenerate_blk, Input['gf_struct'])

        # Call impurity solver, and store sigma_imp, vhf_imp, and pi_imp in "Res"
        Res = _solver_inner_loop(coqui_mpi, h0, delta_iw, u_weiss_iw, h_int, Input['density'], **solver_params)

        # convert from triqs Gf to numpy arrays and ir mesh
        Res.update(coqui_dmft.imp_results_to_raw_data(
            Res['G_iw'], Res['Sigma_iw'], Res['W_iw'], Res['Pi_iw'], iaft)
        )

        # Causal projection
        coqui_dmft.fit_impurity_results_boson(
            Res, iaft, solver_params.get("causal_projection"))

        solver_results.append(Res)

        if save_results:
            # FIXME This will write final_iter +1 if dmft_iteration=-1
            coqui_dmft.save_impurity_results(
                solver_results, imp_params['chkpt_h5'], iteration=dmft_iteration, impurity_index=imp_index
            ) 

    return solver_results
