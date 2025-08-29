"""
Utility functions for TRIQS interface to EDMFT.

This module provides helper routines that facilitate the conversion and
manipulation of data between CoQuí and the TRIQS ecosystem in the context
of EDMFT, including:

- Constructing TRIQS Green’s functions (Gf, BlockGf, Block2Gf) from raw data
  such as Weiss fields, hybridization functions, and interaction tensors.
- Handling mappings between orbital subspaces, solver structures, and block
  Green’s functions.
- Extracting and reducing interaction tensors (e.g. to Hubbard–Kanamori form).
- Enforcing density–density or real-Hamiltonian approximations as required by
  TRIQS solvers (e.g. CT-SEG).
- Miscellaneous utilities for mesh handling, and TRIQS container construction.

Dependencies
------------
This module relies heavily on the TRIQS ecosystem, in particular:
- `triqs.gf` for Green’s function containers and operations
- `triqs.operators` for operator algebra
- `triqs.utility.mpi` for parallelism
- `h5` (HDFArchive) for reading and writing HDF5 data
"""
import triqs.utility.mpi as mpi
from h5 import HDFArchive
import numpy as np
import itertools

from triqs.gf import Gf, make_gf_dlr, BlockGf, Block2Gf
from triqs.operators import c_dag, c, Operator, util

def read_proj_info(wannier_h5):
  with HDFArchive(wannier_h5, 'r') as ar:
    C_ksIai = ar['dft_input/proj_mat']
    band_window = ar['dft_misc_input/band_window']
    kpts_w90 = ar['dft_input/kpts']

  return {'proj_mat': C_ksIai, 'band_window': band_window, 'kpts_w90': kpts_w90}


def set_n_iw(ir_kernel):
    iw_idx_f = ir_kernel.wn_mesh('f', False)
    iw_idx_b = ir_kernel.wn_mesh('b', False)
    max_idx = max(abs(iw_idx_f[0]), abs(iw_idx_f[-1]), abs(iw_idx_b[0]), abs(iw_idx_b[-1]))
    return int(max_idx + 1)


def get_c_to_solver_mapping(solver_struct):
    # Ex: 
    # C space = [ ('up', nbnd), ('down', nbnd) ]
    # solver  = [ ('up_0', a), ('up_1', b), ('down_0', a), ('down_1', b) ]
    #             where a + b = nbnd
    c_to_solver = {}
    o1_up, o1_dn = 0, 0
    for blk_name, blk_dim in solver_struct:
        if blk_name[:2] == "up":
            for i in range(blk_dim):
                c_to_solver[("up", i+o1_up)] = (blk_name, i)
            o1_up += blk_dim
        else:
            for i in range(blk_dim):
                c_to_solver[("down", i+o1_dn)] = (blk_name, i)
            o1_dn += blk_dim

    return c_to_solver


def Vijkl_in_triqs_notation(V_ijkl):
    # switch inner two indices to match triqs notation
    V = np.zeros(V_ijkl.shape, dtype=V_ijkl.dtype)
    nbnd = V.shape[0]
    for or1, or2, or3, or4 in itertools.product(range(nbnd), repeat=4):
        V[or1, or2, or3, or4] = V_ijkl[or1, or3, or2, or4]
    return V


def h_int_density_density(V_abcd, gf_struct, force_real=True):    
    c_to_solver = get_c_to_solver_mapping(gf_struct)
    V, Vp = util.reduce_4index_to_2index(
        Vijkl_in_triqs_notation(V_abcd.real if force_real else V_abcd)
    )
    h_int = util.h_int_density(['up', 'down'], V.shape[0], U=V, Uprime=Vp, 
                               map_operator_structure=c_to_solver)
    return h_int


def h_int_slater(V_abcd, gf_struct, force_real=True):
    c_to_solver = get_c_to_solver_mapping(gf_struct)
    return util.h_int_slater(['up', 'down'], V_abcd.shape[-1], 
                              Vijkl_in_triqs_notation(V_abcd.real if force_real else V_abcd), 
                              map_operator_structure=c_to_solver)
    

def h0_operator(h0_sab, gf_struct, *, diagonal=True, force_real=True):
    assert len(h0_sab.shape) == 3, "incorrect h0_sab.shape"
    H0 = Operator()
    o1_up, o1_dn = 0, 0
    for blk_name, blk_dim in gf_struct:
        s = 0 if blk_name[:2] == "up" else 1
        o1 = o1_up if blk_name[:2] == "up" else o1_dn
        for i in range(blk_dim):
            if force_real:
                H0 += h0_sab[s, o1+i, o1+i].real * c_dag(blk_name, i) * c(blk_name, i)
            else:
                H0 += h0_sab[s, o1+i, o1+i] * c_dag(blk_name, i) * c(blk_name, i)
    return H0


def u_weiss_full_to_density_density(U_wabcd):
    nw, nbnd = U_wabcd.shape[:2]
    u_wab = np.zeros((nw, nbnd, nbnd), dtype=U_wabcd.dtype)
    for a in range(nbnd):
        for b in range(nbnd):
            u_wab[:, a, b] = U_wabcd[:, a, a, b, b]

    return u_wab        
    


def to_block_gf(giw_ir, ir_kernel, gf_struct, mesh_iw):

    assert mesh_iw.statistic == 'Fermion', "Only mesh_iw.statistic == Fermion is supported."
    assert len(giw_ir.shape) == 4, "giw_ir needs to have dimensions (nw, nspins, nbnd, nbnd)"
    assert giw_ir.shape[2] == giw_ir.shape[3], "giw_ir needs to have dimensions (nw, nspins, nbnd, nbnd)"
    assert giw_ir.shape[0] == ir_kernel.nw_f, "giw_ir.shape[0] != ir_kernel.nw_f"
    
    # giw_ir = (nw, nspin, nbnd, nbnd)
    nbnd = giw_ir.shape[-1]

    blk_gf = BlockGf(mesh=mesh_iw, gf_struct=gf_struct)
    mesh_iw_idx = np.array([iwn.index for iwn in mesh_iw])

    # E.g. block structure = [ ("up_0", 1), ("down_0", 1), ("up_1", 2), ("down_1", 2) ]
    offset_up, offset_dn = 0, 0
    for blk_name, blk_dim in gf_struct:
        if blk_name[:2] == "up":
            blk_gf[blk_name].data[:] = ir_kernel.w_interpolate(
                giw_ir[:, 0, offset_up:offset_up+blk_dim, offset_up:offset_up+blk_dim], 
                mesh_iw_idx, 
                stats="f", 
                ir_notation=False
            )
            offset_up += blk_dim
            assert offset_up <= nbnd, "Spin up block excceds band range"
            offset_up = offset_up % nbnd
        elif blk_name[:4] == "down":
            blk_gf[blk_name].data[:] = ir_kernel.w_interpolate(
                giw_ir[:, 1, offset_dn:offset_dn+blk_dim, offset_dn:offset_dn+blk_dim], 
                mesh_iw_idx, 
                stats="f", 
                ir_notation=False
            )
            offset_dn += blk_dim
            assert offset_dn <= nbnd, "Spin down block excceds band range"
            offset_dn = offset_dn % nbnd

    return blk_gf


def to_block2_gf(Diw_ir, ir_kernel, gf_struct, mesh_iw):

    assert mesh_iw.statistic == 'Boson', "Only mesh_iw.statistic == Boson is supported."
    assert len(Diw_ir.shape) == 3, "Diw_ir needs to have dimensions (nw, nbnd, nbnd)"
    assert Diw_ir.shape[1] == Diw_ir.shape[2], "Diw_ir needs to have dimensions (nw, nbnd, nbnd)"
    nw_half = ir_kernel.nw_b//2 if ir_kernel.nw_b%2==0 else ir_kernel.nw_b//2 + 1
    assert Diw_ir.shape[0] == nw_half, "Diw_ir.shape[0] != nw_b_half"
    
    # Diw_ir = (nw, nbnd, nbnd)
    nbnd = Diw_ir.shape[-1]

    mesh_iw_idx = np.array([iwn.index for iwn in mesh_iw])
    Diw_data = ir_kernel.w_interpolate_phsym(Diw_ir, mesh_iw_idx, stats="b", ir_notation=False)

    gf_array = []
    o1_up, o1_dn = 0, 0
    for name1, dim1 in gf_struct:
        o1 = o1_up if name1[:2] == "up" else o1_dn
        
        gf_list = []
        o2_up, o2_dn = 0, 0
        for name2, dim2 in gf_struct:
            o2 = o2_up if name2[:2] == "up" else o2_dn
            
            gf = Gf(mesh = mesh_iw, target_shape = (dim1, dim2))
            gf.data[:] = Diw_data[:, o1:o1+dim1, o2:o2+dim2]
            
            if name2[:2] == "up":
                o2_up = o2 + dim2
                assert o2_up <= nbnd, "Spin up block excceds band range"
                o2_up = o2_up % nbnd
            else:
                o2_dn = o2 + dim2
                assert o2_dn <= nbnd, "Spin down block excceds band range"
                o2_dn = o2_dn % nbnd
                
            gf_list.append(gf)
        
        if name1[:2] == "up":
            o1_up = o1 + dim1
            assert o1_up <= nbnd, "Spin up block excceds band range"
            o1_up = o1_up % nbnd
        else:
            o1_dn = o1 + dim1
            assert o1_dn <= nbnd, "Spin down block excceds band range"
            o1_dn = o1_dn % nbnd
        
        gf_array.append(gf_list)

    names = [name for name, _ in gf_struct]
    return Block2Gf(names, names, gf_array)


def to_triqs_containers(h0, delta_iw, Vimp, u_weiss_iw, ir_kernel, 
                        gf_struct, triqs_iw_mesh, 
                        density_hamiltonian, real_hamiltonian=True):
    """
    Convert raw CoQui outputs (NumPy arrays) into TRIQS containers 
    (e.g. `triqs.operators.many_body_operator`, `BlockGf`, and `Block2Gf`).

    This function provides a bridge between CoQui’s raw many-body data and 
    TRIQS’ Green’s function representation. It constructs the one-particle and 
    two-particle objects required for TRIQS-based impurity solvers or DMFT-like 
    workflows.

    Parameters
    ----------
    h0 : np.ndarray
        One-particle Hamiltonian matrix in the impurity subspace.
    delta_iw : np.ndarray
        Hybridization function Δ(iωₙ) on the Matsubara frequency axis, in raw array form.
    Vimp : np.ndarray
        Bare Coulomb interaction tensor in the impurity basis.
    u_weiss_iw : np.ndarray
        Dynamical screened interaction U(iωₙ) from cRPA or EDMFT preprocessing, 
        given as raw arrays on the Matsubara axis.
    ir_kernel : object
        Imaginary-time/frequency transform kernel (e.g. IR/IAFT object) used for 
        Fourier transforms between τ and iωₙ.
    gf_struct : dict
        Block structure of Green’s functions, mapping orbital/spin indices to block labels.
    triqs_iw_mesh : dict
        Dictionary of TRIQS mesh objects:
        - `"fermion"` : Matsubara mesh for fermionic objects (Δ, G).
        - `"boson"`   : Matsubara mesh for bosonic objects (U).
    density_hamiltonian : bool
        Whether to use the density-density approximation for the Coulomb interaction. 
        This will also enforce the non-interacting Hamiltonian to density operator only. 
        Currently must be `True`. Non-density-density interactions are not implemented.
    real_hamiltonian : bool, optional
        If `True`, enforce the bare Hamiltonian (h0, Vimp) to be real-valued. 
        Default is `True`.

    Returns
    -------
    h0 : triqs.operators.many_body_operator
        One-particle Hamiltonian in TRIQS operator form.
    delta_iw : triqs.gf.BlockGf
        Hybridization function Δ(iωₙ) as a TRIQS block Green’s function.
    h_int : triqs.operators.many_body_operator
        Local interaction Hamiltonian in density-density approximation.
    u_weiss_iw : triqs.gf.Block2Gf
        Dynamical screened interaction U(iωₙ) as a TRIQS two-particle block Green’s function.

    Notes
    -----
    - This conversion assumes the density-density approximation for Coulomb 
      interactions; a full four-index mapping is not yet implemented.
    """

    assert density_hamiltonian==True, (
        "Convertion to non-density-density Hamiltonian is not implemented yet."
    )

    # one-particle
    h0 = h0_operator(h0, gf_struct, diagonal=density_hamiltonian, 
                     force_real=real_hamiltonian)
    delta_iw = to_block_gf(delta_iw, ir_kernel, 
                           gf_struct, triqs_iw_mesh["fermion"])
    
    # two-particle
    u_weiss_iw = to_block2_gf(
        u_weiss_full_to_density_density(u_weiss_iw), 
        ir_kernel, gf_struct, triqs_iw_mesh["boson"]
    )
    h_int = h_int_density_density(
        Vimp, gf_struct, 
        force_real=real_hamiltonian
    )
    
    return h0, delta_iw, h_int, u_weiss_iw


def Gf_dlr_from_ir(Giw_ir, ir_kernel, mesh_dlr_iw, dim=2):
    
    nbnd = Giw_ir.shape[-1]
    stats = 'f' if mesh_dlr_iw.statistic == 'Fermion' else 'b'

    if stats == 'b':
        Giw_ir_pos = Giw_ir.copy()
        Giw_ir = np.zeros([Giw_ir_pos.shape[0] * 2 - 1] + [nbnd] * dim, dtype=complex)
        Giw_ir[: Giw_ir_pos.shape[0]] = Giw_ir_pos[::-1]
        Giw_ir[Giw_ir_pos.shape[0] :] = Giw_ir_pos[1:]
    
    Gf_dlr_iw = Gf(mesh=mesh_dlr_iw, target_shape=[nbnd] * dim)

    # prepare idx array for spare ir
    if stats == 'f':
        mesh_dlr_iw_idx = np.array([iwn.index for iwn in mesh_dlr_iw])
    else:
        mesh_dlr_iw_idx = np.array([iwn.index for iwn in mesh_dlr_iw])

    Gf_dlr_iw.data[:] = ir_kernel.w_interpolate(Giw_ir, mesh_dlr_iw_idx, stats=stats, ir_notation=False)

    return make_gf_dlr(Gf_dlr_iw)


def estimate_zero_moment(Aw, iw_mesh):
    """
    Estimate the zeroth moment (high-frequency constant term) of A(iω). 

    Assumes that at large Matsubara frequencies the function behaves as

        A(iω) ≈ t + B / (iω) + O(1/ω²),

    where `t` is the zeroth moment (constant term). The estimate uses the values
    of A(iω) at the last two frequencies in `iw_mesh` to extrapolate to the
    ω → ∞ limit.

    Parameters
    ----------
    Aw : array of complex, shape (nw, ...)
        Values of A(iω) on the Matsubara frequency mesh. Must be ordered such
        that the last entries correspond to the largest |ω|.
    iw_mesh : array of complex, shape (nw)
        Matsubara frequency mesh corresponding to `Aw`. The last two entries are
        used for the extrapolation.

    Returns
    -------
    t : complex
        Estimated zeroth moment (constant term) of A(iω) in the high-frequency
        expansion.

    Notes
    -----
    - Accuracy depends on how far into the asymptotic regime the last two
      frequencies are. If the frequencies are not sufficiently large, the
      estimate may be biased.
    """
    iw_m1 = iw_mesh[-1]
    iw_m2 = iw_mesh[-2]
    t = Aw[-1].real - (Aw[-1] - Aw[-2]).real * iw_m2 ** 2 / (
           iw_m2 ** 2 - iw_m1 ** 2)
    t = t.astype(complex)

    return t


def extract_h0_and_delta(g_weiss_wsab, ir_kernel, high_freq_multiplier=10):
    """
    Estimate the static one-body term h₀ (as t_sIab) and the hybridization function Δ(iω)
    from a Weiss Green's function G₀(iω) sampled on a fermionic Matsubara mesh.

    The method:
      1) Interpolate G₀(iω) to a few very large Matsubara frequencies (scaled by
         `high_freq_multiplier`) to probe the asymptotic regime.
      2) Construct W(iω) = iω·I - [G₀(iω)]⁻¹ and estimate its zeroth moment
         t_sIab = lim_{|ω|→∞} W(iω) via `estimate_zero_moment`.
      3) Build Δ(iω) from the Dyson-like relation:
            Δ(iω) = iω·I - t_sIab - [G₀(iω)]⁻¹.

    Parameters
    ----------
    g_weiss_wsab : ndarray, complex, shape (nw, nspin, nbnd, nbnd)
        Weiss Green's function G₀(iωₙ) on the fermionic Matsubara mesh returned by `ir_kernel`.
        The leading dimension is frequency index; s,a,b are spin and orbital indices.
    ir_kernel : IAFT object
    high_freq_multiplier : float, default 10
        Multiplier applied to the last few (three) IR fermionic frequencies (in IR notation)
        before converting to physical Matsubara frequencies, to push evaluation deep into
        the asymptotic region for a more stable moment estimate.

    Returns
    -------
    t_sIab_estimate : ndarray, complex, shape (nspin, nbnd, nbnd)
        Estimate of the static one-body term (zeroth moment) per spin block.
    delta_estimate : ndarray, complex, shape (nw, nspin, nbnd, nbnd)
        Estimated hybridization function Δ(iωₙ) on the original fermionic mesh.

    Notes
    -----
    - Accuracy of `t_sIab_estimate` depends on how large the interpolated frequencies are.
    """
    nspin = g_weiss_wsab.shape[1]
    
    # 1) Interpolate G0 to very high fermionic frequencies to improve the accuracy of high-frequency fitting
    iwn_interp = ir_kernel.wn_mesh('f', ir_notation=False)[-3:] * high_freq_multiplier
    g_weiss_interp = ir_kernel.w_interpolate(g_weiss_wsab, iwn_interp, 'f', ir_notation=False)
    iwn_interp = (2*iwn_interp.astype(float) + 1) * np.pi / ir_kernel.beta
    weiss_tmp = np.zeros(g_weiss_interp.shape, dtype=complex)
    for n, g in enumerate(g_weiss_interp):
        for s in range(nspin):
            weiss_tmp[n, s] = 1j * iwn_interp[n] - np.linalg.inv(g[s])

    # 2) Fitting the zeroth moment as the non-interacting Hamiltonian
    t_sIab_estimate = estimate_zero_moment(weiss_tmp, iwn_interp)

    # 3) Construct Δ(iω) = iω·I - t_sIab - [G0(iω)]^{-1} on the original mesh
    iwn_mesh_imp = ir_kernel.wn_mesh('f') * np.pi / ir_kernel.beta
    delta_estimate = np.zeros(g_weiss_wsab.shape, dtype=complex)
    nbnd = t_sIab_estimate.shape[-1]
    for n in range(delta_estimate.shape[0]):
        for s in range(nspin):
            g_weiss_inv = np.linalg.inv(g_weiss_wsab[n, s])
            delta_estimate[n, s] = 1j * iwn_mesh_imp[n] * np.eye(nbnd) - t_sIab_estimate[s] - g_weiss_inv

    # 4) checking the leakage of the resulting Δ(iω)
    if mpi.is_master_node():
        ir_kernel.check_leakage(delta_estimate, 'f', 'delta_estimate', w_input=True)
    mpi.barrier()

    return t_sIab_estimate, delta_estimate


def compute_weiss_fields_w(*, ir_kernel, local_gf, impurity_selfenergies=None, density_only=False):
    # checking inputs 
    missing = {"Gloc_t", "Wloc_t", "Vloc"} - local_gf.keys()
    if missing:
        raise ValueError(f"Missing keys in local_gf: {missing}")

    if impurity_selfenergies is not None:
        missing = {"Vhf_imp", "Sigma_imp_w", "Pi_imp_w"} - impurity_selfenergies.keys()
        if missing:
            raise ValueError(f"Missing keys in impurity_selfenergies: {missing}")
    else:
        impurity_selfenergies = {"Vhf_imp": None, "Sigma_imp_w": None, "Pi_imp_w": None}

    # bosonic first 
    if impurity_selfenergies["Pi_imp_w"] is not None:
        mpi.report("Evaluate the bosonic Weiss field using the provided impurity polarizability.")
        Pi_imp_w = impurity_selfenergies["Pi_imp_w"]
    else:
        mpi.report("Evaluate the bosonic Weiss field at the RPA level.")
        Pi_imp_w = eval_pi_rpa(local_gf["Gloc_t"], density_only=density_only, w_out=True, ft=ir_kernel)
        
    u_weiss_w = compute_weiss_boson_w(
        local_gf["Vloc"],
        ir_kernel.tau_to_w_phsym(local_gf["Wloc_t"], stats='b'),
        Pi_imp_w
    )
    # fermionic 
    if impurity_selfenergies["Vhf_imp"] is not None and impurity_selfenergies["Sigma_imp_w"] is not None:
        mpi.report("Evaluate the fermionic Weiss field using the provided impurity self-energy.\n")
        Vhf_imp = impurity_selfenergies["Vhf_imp"]
        Sigma_imp_w = impurity_selfenergies["Sigma_imp_w"]
    else:
        mpi.report("Evaluate the fermionic Weiss field using the local GW self-energy.\n")
        Vhf_imp = eval_hf_dc(
            -ir_kernel.tau_interpolate(local_gf["Gloc_t"], [ir_kernel.beta], stats='f')[0], 
            local_gf["Vloc"], 
            u_weiss_w[0]+local_gf["Vloc"]
        )
        Sigma_imp_w = ir_kernel.tau_to_w(eval_gw_dc_t(local_gf["Gloc_t"], local_gf["Wloc_t"]), stats='f')
                   
    g_weiss_w = compute_weiss_fermion_w(
        ir_kernel.tau_to_w(local_gf["Gloc_t"], stats='f'),
        Vhf_imp, 
        Sigma_imp_w
    )
    return g_weiss_w, u_weiss_w


def compute_weiss_fermion_w(G_wsab, vhf_sab, sigma_wsab):
    nw, nspin, nbnd = G_wsab.shape[:3]
    g_wsab = np.zeros(G_wsab.shape, dtype=G_wsab.dtype)

    #  g(w) = [G(w)^{-1} + Sigma_imp]^{-1]
    for ws in range(nw*nspin):
        w = ws // nspin
        s = ws % nspin
        X_inv = np.linalg.solve(G_wsab[w, s], np.eye(nbnd)) + vhf_sab[s] + sigma_wsab[w, s]
        g_wsab[w, s] = np.linalg.solve(X_inv, np.eye(nbnd))

    return g_wsab


def compute_weiss_boson_w(V_abcd, W_wabcd, Pi_wabcd):
    nbnd = V_abcd.shape[0]
    nbnd2 = nbnd*nbnd
    Wfull_pb = (W_wabcd + V_abcd).reshape(-1,nbnd2,nbnd2)
    Pi_pb = Pi_wabcd.reshape(-1,nbnd2,nbnd2)

    # U(w) = W(w)[I + Pi(w)*W(w)]^{-1}
    U_pb = np.zeros(Wfull_pb.shape, dtype=Wfull_pb.dtype)
    for n, W in enumerate(Wfull_pb):
        X = np.eye(nbnd2) + Pi_pb[n] @ W
        X_inv = np.linalg.solve(X, np.eye(nbnd2))
        U_pb[n] = W @ X_inv

    return U_pb.reshape(W_wabcd.shape) - V_abcd


def eval_pi_rpa(G_tsab, density_only=False, *, w_out=False, ft=None):
    nts, nspin, nbnd = G_tsab.shape[:3]
    nts_half = nts//2 if nts%2==0 else nts//2 + 1
    pi_t = np.zeros((nts_half, nbnd, nbnd, nbnd, nbnd), dtype=complex)
    spin_factor = -2.0 if nspin == 1 else -1.0
    if not density_only:
        for t in range(nts_half):
            mt = nts-t-1
            pi_t[t] += spin_factor * np.einsum(
                'sbd,sca->abcd',
                G_tsab[t], G_tsab[mt]
            )
    else:
        for t in range(nts_half):
            mt = nts-t-1
            for s in range(nspin):
                for a in range(nbnd):
                    for b in range(nbnd):
                        pi_t[t, a, a, b, b] += spin_factor * G_tsab[t, s, a, b] * G_tsab[mt, s, b, a]

    return ft.tau_to_w_phsym(pi_t, stats='b') if w_out else pi_t


def eval_hf_dc(Dm_sab, V_abcd, U0_abcd):
    """
    Evaluate the Hartree and exchange contributions to the density matrix.

    Parameters:
    - Dm_sab: Density matrix in spin and band indices.
    - V_abcd: Bare interaction on a product basis.
    - U0_abcd: Static screened interaction on a product basis.

    Returns:
    - Hartree-Fock potential for an impurity with dynamic interactions.
    """

    Vhf_sab = np.zeros(Dm_sab.shape, dtype=Dm_sab.dtype)

    # Hartree contribution
    spin_factor = 2.0 if Dm_sab.shape[0] == 1 else 1.0
    for s in range(Dm_sab.shape[0]):
        Vhf_sab += spin_factor * np.einsum('dc,bacd->ab', Dm_sab[s], U0_abcd)

    # Exchange contribution
    Vhf_sab -= np.einsum('sab,aibj->sij', Dm_sab, V_abcd)

    return Vhf_sab


def eval_gw_dc_t(G_tsab, W_tabcd):
    """
    Evaluate the GW self-energy contribution to the impurity Green's function.

    Parameters:
    - G_tsab: Impurity Green's function in time, spin, and band indices.
    - W_tabcd: Dynamic interaction on a product basis.

    Returns:
    - GW self-energy contribution to the impurity Green's function.
    """

    nts, nts_half = G_tsab.shape[0], W_tabcd.shape[0]
    sigma_tsab = np.zeros(G_tsab.shape, dtype=G_tsab.dtype)

    for t in range(nts_half):
        mt = nts-t-1
        sigma_tsab[t] = -1 * np.einsum('sab,aibj->sij', G_tsab[t], W_tabcd[t])
        if mt != t:
            sigma_tsab[mt] = -1 * np.einsum('sab,aibj->sij', G_tsab[mt], W_tabcd[t])

    return sigma_tsab


def solve_gw_dc(G_t, V, W_t, u_weiss_iw, ir_kernel, density_only=True):

    Pi_dc_t = eval_pi_rpa(G_t, density_only=True)

    Dm = -ir_kernel.tau_interpolate(G_t, [ir_kernel.beta], stats='f')[0]
    Vhf_dc = eval_hf_dc(Dm, V, u_weiss_iw[0]+V)
    
    Sigma_dc_t = eval_gw_dc_t(G_t, W_t)

    return {
        "Pi_dc_iw_mat": ir_kernel.tau_to_w_phsym(Pi_dc_t, stats='b'), 
        "Vhf_dc_mat": Vhf_dc, 
        "Sigma_dc_iw_mat": ir_kernel.tau_to_w(Sigma_dc_t, stats='f')
    }


def impurity_results_to_coqui(solver_res):
    """
    Convert the impurity results in TRIQS Gf containers to raw data in the IR basis 

    solver_res: [INPUT/OUTPUT] SolverResults
        
        TRIQS impurity results are stored in 
        
            - Sigma_Hartree: block matrix 
              Static impurity self-energy 

            - Sigma_dynamic: TRIQS BlockGf
              Dynamic impurity self-energy on the Matsubara axis 

            - Pi_iw: TRIQS Gf
              Impurity polarizability on the Matsubara axis 

        Raw data in the IR basis are updated to 

            - Vhf_imp_mat: numpy.ndarray(nspin, n_orb, n_orb)
              Static impurity self-energy

            - Sigma_imp_iw_mat: numpy.ndarray(niw, nspin, n_orb, n_orb)
              Dynamic impurity self-energy on the IR Matsubara points

            - Pi_imp_iw_mat: numpy.ndarray(niw_half, n_orb, n_orb, n_orb, n_orb)
              Impurity polarizability on the positive IR Matsubara points
    
    """
    # convert solver_res.Sigma_Hartree from the blk structure to a numpy array
    solver_res.Vhf_imp_mat = block_mat_to_mat(solver_res.Sigma_Hartree, solver_res.gf_struct)
    n_orb = solver_res.Vhf_imp_mat.shape[-1]
    
    # interpolate solver_res.Sigma_dynamic to ir grid and convert from blk structure to numpy array
    Sigma_dyn = block_gf_to_gf(solver_res.Sigma_dynamic, solver_res.gf_struct)
    
    ir_idx_f = solver_res.ir_kernel.wn_mesh(stats='f', ir_notation=False)
    nw_f_half = len(ir_idx_f) // 2
    solver_res.Sigma_imp_iw_mat = np.zeros((len(ir_idx_f), 2, n_orb, n_orb), dtype=complex)
    for idx in range(nw_f_half):
        iw_pos = nw_f_half + idx
        iw_neg = nw_f_half - idx - 1
        data_idx = solver_res.iw_mesh_f.to_data_index(ir_idx_f[iw_pos])
        solver_res.Sigma_imp_iw_mat[iw_pos] = Sigma_dyn.data[data_idx]
        solver_res.Sigma_imp_iw_mat[iw_neg] = Sigma_dyn.data[data_idx].conj()

    solver_res.Sigma_dynamic_gf = Sigma_dyn
    
    # interpolate solver_res.Pi_iw_pb to ir grid
    ir_idx_b = solver_res.ir_kernel.wn_mesh(stats='b', ir_notation=False)
    nw_b_half = len(ir_idx_b)//2
    solver_res.Pi_imp_iw_mat = np.zeros((nw_b_half+1, n_orb, n_orb, n_orb, n_orb), dtype=complex)
    for idx in range(nw_b_half+1):
        data_idx = solver_res.iw_mesh_b.to_data_index(ir_idx_b[nw_b_half+idx])
        solver_res.Pi_imp_iw_mat[idx] = solver_res.Pi_iw.data[data_idx]
    


def block_gf_to_gf(block_gf, gf_struct):
    
    n_orb = sum(dim for name, dim in gf_struct if name[:2]=="up")
    gf_sab = Gf(mesh = block_gf.mesh, target_shape=(2, n_orb, n_orb))
    gf_sab.data[:] = 0.0
    
    offsets = [0, 0]
    for blk_name, dim in gf_struct:
        s = 0 if blk_name[:2] == "up" else 1
        gf_sab.data[:, s, offsets[s]:offsets[s]+dim, offsets[s]:offsets[s]+dim] = block_gf[blk_name].data[:]
        offsets[s] += dim
    
    return gf_sab


def blk_arr_to_arr(blk_array, gf_struct):
    """
    Convert a list of spin/orbital block arrays into a single dense array.

    Parameters
    ----------
    blk_array : list of ndarray
        List of block data arrays. Each block corresponds to one entry in `gf_struct`,
        and all blocks must have the same leading dimensions (e.g. frequency, k-points).
        The last two dimensions of each block are the orbital indices for that block.
    gf_struct : list of (str, int)
        Structure definition for the blocks. Each element is a tuple `(block_name, dim)`,
        where `block_name` is a string like `"up_0"`, `"down_1"`, and `dim` is the orbital
        dimension of that block.

    Returns
    -------
    array : ndarray
        Combined dense array with shape
        ``arr_shape[:-2] + (nspin, n_orb, n_orb)``,
        where `arr_shape` is the shape of the first element of `blk_array`.
        The extra dimension `nspin` is the number of spin blocks (e.g. 2 for "up"/"down"),
        and `n_orb` is the total number of orbitals per spin.
        Each block from `blk_array` is placed into the correct slice of the full array
        according to `gf_struct`.

    Notes
    -----
    - Spin blocks are identified by the prefix of `block_name` before the underscore
      (e.g. `"up"` in `"up_0"`).
    - Orbital sub-blocks are placed contiguously along the orbital axes, with offsets
      determined by the cumulative dimensions of the blocks.
    - Assumes each spin has the same total orbital count; raises if not.
    """

    assert len(blk_array) == len(gf_struct),  (
        f"Inconistent number of blocks between block_array ({len(blk_array)}) "
        f"and gf_struct ({len(gf_struct)})."
    )

    spin_blk = []
    for name, _ in gf_struct:
        spin = name.split('_', 1)[0]
        if spin not in spin_blk:
            spin_blk.append(spin)
    spin_to_idx = {s: i for i, s in enumerate(spin_blk)}
    nspin = len(spin_blk)

    # Compute per-spin orbital totals and validate they match
    per_spin_counts = {s: 0 for s in spin_blk}
    for name, dim in gf_struct:
        s = name.split('_', 1)[0]
        per_spin_counts[s] += dim
    counts = set(per_spin_counts.values())
    assert len(counts) == 1, (
        "Per-spin orbital totals must match across spins; got "
        f"{per_spin_counts}"
    )
    n_orb = counts.pop()

    # Infer leading shape/dtype from first block
    arr_shape = blk_array[0].shape
    leading_shape = arr_shape[:-2]

    # Allocate dense array
    array = np.zeros(leading_shape+(nspin, n_orb, n_orb), dtype=blk_array[0].dtype)
    
    # Fill with block data
    offsets = [0] * nspin
    for (blk_name, dim), blk_data in zip(gf_struct, blk_array):
        s_name = blk_name.split('_', 1)[0]
        s = spin_to_idx[s_name]

        assert blk_data.shape[:-2] == leading_shape, (
            f"Leading dims mismatch for block {blk_name}: "
            f"{blk_data.shape[:-2]} vs {leading_shape}"
        )

        start, stop = offsets[s], offsets[s]+dim
        array[..., s, start:stop, start:stop] = blk_data
        offsets[s] += dim

    return array


def arr_to_blk_arr(array, gf_struct):
    """
    Convert a dense array into a list of spin/orbital block arrays per `gf_struct`.

    Parameters
    ----------
    array : ndarray
        Dense data with shape: leading_shape + (nspin, n_orb, n_orb).
    gf_struct : list[tuple[str, int]]
        List of (block_name, dim), e.g. [("up_0", d0), ("up_1", d1), ("down_0", d2), ...].
        The spin label is the prefix before the first underscore.

    Returns
    -------
    blk_array : list of ndarray
        Blocks in the same order as `gf_struct`. Each has shape: leading_shape + (dim, dim).

    Raises
    ------
    AssertionError
        If shapes are inconsistent with `gf_struct` or per-spin orbital totals
        do not match the dense array’s orbital size.
    """
    assert array.ndim >= 3, "array must have at least 3 dimensions (…, nspin, n_orb, n_orb)"
    leading_shape = array.shape[:-3]
    nspin, n_orb_0, n_orb_1 = array.shape[-3:]
    assert n_orb_0 == n_orb_1, f"Last two dims must be square; got {(n_orb_0, n_orb_1)}"

    # Preserve spin order as first seen in gf_struct
    spin_blk = []
    for name, _ in gf_struct:
        s = name.split('_', 1)[0]
        if s not in spin_blk:
            spin_blk.append(s)
    assert len(spin_blk) == nspin, (
        f"Spin count mismatch: array has {nspin}, gf_struct implies {len(spin_blk)}"
    )
    spin_to_idx = {s: i for i, s in enumerate(spin_blk)}

    # Check per-spin total orbitals vs array
    per_spin_counts = {s: 0 for s in spin_blk}
    for name, dim in gf_struct:
        s = name.split('_', 1)[0]
        per_spin_counts[s] += dim
    totals = set(per_spin_counts.values())
    assert len(totals) == 1 and totals.pop() == n_orb_0, (
        "Per-spin orbital totals must match the dense array's orbital size; "
        f"got per-spin {per_spin_counts}, array n_orb={n_orb_0}"
    )

    # Extract blocks
    offsets = [0] * nspin
    blk_array = []
    for blk_name, dim in gf_struct:
        s_name = blk_name.split('_', 1)[0]
        s = spin_to_idx[s_name]

        start, stop = offsets[s], offsets[s]+dim
        assert stop <= n_orb_0, (
            f"Block {blk_name} (dim={dim}) exceeds spin-orbital range "
            f"[{start}:{stop}) with n_orb={n_orb_0}"
        )

        blk_array.append(array[..., s, start:stop, start:stop].copy())
        offsets[s] += dim

    for s_idx, off in enumerate(offsets):
        assert off == n_orb_0, (
            f"Unused orbital slots remain for spin index {s_idx}: "
            f"filled {off} / {n_orb_0}"
        )

    return blk_array


def block_mat_to_mat(block_mat, gf_struct):
    assert len(block_mat) == len(gf_struct), f"Inconistent number of blocks from block_mat ({len(block_mat)}) and gf_struct ({len(gf_struct)})."

    n_orb = sum(dim for name, dim in gf_struct if name[:2]=="up")
    
    # assume nspin = 2
    mat = np.zeros((2, n_orb, n_orb), dtype=block_mat[0].dtype)
    offsets = [0, 0]
    for (blk_name, dim), blk_data in zip(gf_struct, block_mat):
        s = 0 if blk_name[:2] == "up" else 1
        mat[s, offsets[s]:offsets[s]+dim, offsets[s]:offsets[s]+dim] = blk_data
        offsets[s] += dim

    return mat
        

def print_title_box(name, box_width=19):
    top_left = '╔'
    top_right = '╗'
    bottom_left = '╚'
    bottom_right = '╝'
    horizontal = '═'
    vertical = '║'

    title = f"{vertical}{name.center(box_width - 2)}{vertical}"
    top_border = f"{top_left}{horizontal * (box_width - 2)}{top_right}"
    bottom_border = f"{bottom_left}{horizontal * (box_width - 2)}{bottom_right}"

    mpi.report("\n"+top_border)
    mpi.report(title)
    mpi.report(bottom_border+"\n")


