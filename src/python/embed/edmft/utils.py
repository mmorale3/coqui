from h5 import HDFArchive
import numpy as np

from triqs.gf import Gf, make_gf_dlr

def read_proj_info(wannier_h5):
  with HDFArchive(wannier_h5, 'r') as ar:
    C_ksIai = ar['dft_input/proj_mat']
    band_window = ar['dft_misc_input/band_window']
    kpts_w90 = ar['dft_input/kpts']

  return {'proj_mat': C_ksIai, 'band_window': band_window, 'kpts_w90': kpts_w90}


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


def compute_weiss_fields_w(*, ir_kernel, local_gf, impurity_selfenergies=None, density_only=False):
    missing = {"Gloc_t", "Wloc_t", "Vloc"} - local_gf.keys()
    if missing:
        raise ValueError(f"Missing keys in local_gf: {missing}")

    if impurity_selfenergies is not None:
        missing = {"Vhf_imp", "Sigma_imp_t", "Pi_imp_t"} - impurity_selfenergies.keys()
        if missing:
            raise ValueError(f"Missing keys in impurity_selfenergies: {missing}")

    if impurity_selfenergies is None:
        print("Using RPA to compute Weiss fields")
        # compute Weiss fields with impurity self-energies approximated as GW
        u_weiss_w = compute_weiss_boson_w(local_gf["Vloc"],
                                          ir_kernel.tau_to_w_phsym(local_gf["Wloc_t"], stats='b'),
                                          eval_pi_rpa(local_gf["Gloc_t"], density_only=density_only, w_out=True, ft=ir_kernel))

        vhf_sab = eval_hf_dc(-ir_kernel.tau_interpolate(local_gf["Gloc_t"], [ir_kernel.beta], stats='f')[0],
                             local_gf["Vloc"], u_weiss_w[0]+local_gf["Vloc"])
        sigma_gw_tsab = eval_gw_dc_t(local_gf["Gloc_t"], local_gf["Wloc_t"])
        g_weiss_w = compute_weiss_fermion_w(ir_kernel.tau_to_w(local_gf["Gloc_t"], stats='f'), vhf_sab,
                                            ir_kernel.tau_to_w(sigma_gw_tsab, stats='f'))

        return g_weiss_w, u_weiss_w
    else:
        print("Using impurity self-energies to compute Weiss fields")
        g_weiss_w = compute_weiss_fermion_w(ir_kernel.tau_to_w(local_gf["Gloc_t"], stats='f'),
                                            impurity_selfenergies["Vhf_imp"],
                                            ir_kernel.tau_to_w(impurity_selfenergies["Sigma_imp_t"], stats='f'))

        u_weiss_w = compute_weiss_boson_w(local_gf["Vloc"],
                                          ir_kernel.tau_to_w_phsym(local_gf["Wloc_t"], stats='b'),
                                          ir_kernel.tau_to_w_phsym(impurity_selfenergies["Pi_imp_t"], stats='b'))

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

