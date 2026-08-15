#!/usr/bin/env python3.14
"""Independent Hartree probe for the AB Si PAW fixture.

Computes, straight from bdft.h5 with numpy only:
  E_H(rho_tilde)            -- smooth density alone
  E_H(rho_tilde + rho_hat)  -- with compensation charge from becsum x Q(G)
Ground truth: ABINIT smooth-grid hartree = 58.4283179 Ha (si_paw.abo).
CoQui direct route printed 45.7378 Ha; THC implied 58.43 + one-center.
"""
import numpy as np
import h5py

F = "/Users/mmorales/Software/CoQui_Separate_Development/tests/unit_test_files/bdft/si_kp222_paw_abinit/bdft.h5"
f = h5py.File(F, "r")

latt = f["System/lattice_vectors"][...]          # rows = a_i (a.u.)
recv = f["System/reciprocal_vectors"][...]       # rows = b_i
tau  = f["System/atomic_positions"][...]         # cartesian a.u. (check)
wk   = f["System/kpoint_weights"][...]
occ  = f["Orbitals/occ"][...]                    # (ns, nk, nb)
mesh = f["Orbitals/fft_mesh_aug"][...]
Om   = abs(np.linalg.det(latt))
nk   = 8
nb   = occ.shape[-1]
print(f"Omega={Om:.6f}  mesh={mesh}  occ sum={occ.sum()*wk[0] if wk.sum()>0.99 else occ.sum()/nk}")
print("tau =", tau)

# ---- smooth density on the aug mesh ----
mwfc = f["Orbitals/miller_wfc"][...]              # shared wfc sphere (ngw, 3)
N1, N2, N3 = (int(x) for x in mesh)
rho = np.zeros((N1, N2, N3))
for ik in range(nk):
    psi = f[f"Orbitals/psi_s0_k{ik}"][...]
    psi = psi[..., 0] + 1j * psi[..., 1]         # (nb, ngw)
    box = np.zeros((N1, N2, N3), dtype=complex)
    for ib in range(nb):
        o = occ[0, ik, ib]
        if o < 1e-12: continue
        box[...] = 0.0
        box[mwfc[:, 0] % N1, mwfc[:, 1] % N2, mwfc[:, 2] % N3] = psi[ib]
        # psi(r) = sum_G c_G e^{iGr}; ifft with norm='forward' gives that sum
        pr = np.fft.ifftn(box, norm="forward")
        rho += wk[ik] * o * np.abs(pr) ** 2
rho /= Om                                        # |psi|^2/Om, sum|c|^2=norm
nG = np.fft.fftn(rho) / (N1 * N2 * N3)           # n_G with n(r)=sum n_G e^{iGr}
print(f"integral rho_tilde = {nG[0,0,0].real * Om:.6f} (AE would be 24)")

# ---- becsum from stored projectors ----
nh = 12
pofs = f["Hamiltonian/paw/projector_offset"][...]
becsum = np.zeros((2, nh, nh), dtype=complex)
for ik in range(nk):
    P = f[f"Hamiltonian/paw/projector_k{ik}"][...]
    P = P[..., 0] + 1j * P[..., 1]               # (nkb, npw_k)
    psi = f[f"Orbitals/psi_s0_k{ik}"][...]
    psi = psi[..., 0] + 1j * psi[..., 1]
    # projectors stored on their own per-k sphere; overlap <beta|psi> needs
    # matching G lists: miller_k == subset/order of miller_wfc? Compare.
    mk = f[f"Hamiltonian/paw/miller_k{ik}"][...]
    # build index of mk rows inside mwfc
    key = {tuple(g): i for i, g in enumerate(mwfc)}
    idx = np.array([key.get(tuple(g), -1) for g in mk])
    ok = (idx >= 0)
    Pk = P[:, ok]
    Ck = psi[:, idx[ok]]
    # <beta_I | psi_b> = sum_G conj(beta_I(G)) c_b(G)  (convention guess)
    ov = Pk.conj() @ Ck.T                        # (nkb, nb)
    for ia in range(2):
        sl = slice(pofs[ia], pofs[ia] + nh)
        for ib in range(nb):
            o = occ[0, ik, ib]
            if o < 1e-12: continue
            v = ov[sl, ib]
            becsum[ia] += wk[ik] * o * np.outer(v, v.conj())
becsum = becsum.real
print("becsum diag atom0:", np.diag(becsum[0])[:6].round(4))

# ---- rho_hat on miller_g via stored augmentation function ----
mg  = f["Hamiltonian/paw/miller_g"][...]
Qg  = f["Hamiltonian/paw/augmentation_function_isp0"][...]
Qg  = Qg[..., 0] + 1j * Qg[..., 1]               # (nij=78, ngm)
ijtoh = f["Hamiltonian/paw/ijtoh"][0]            # (12,12) -> 0-based? check
Gcart = mg @ recv                                # (ngm, 3)
ngm = mg.shape[0]
print("ijtoh[0,:4] =", ijtoh[0, :4], " min/max:", ijtoh.min(), ijtoh.max())

rho_hat = np.zeros(ngm, dtype=complex)
for ia in range(2):
    sf = np.exp(-1j * (Gcart @ tau[ia]))
    acc = np.zeros(ngm, dtype=complex)
    for I in range(nh):
        for J in range(nh):
            ij = ijtoh[I, J] - (1 if ijtoh.min() >= 1 else 0)
            acc += becsum[ia, I, J] * Qg[ij]
    rho_hat += acc * sf
print(f"integral rho_hat = {rho_hat[np.all(mg==0,axis=1)].real * Om}")

# ---- Hartree energies ----
G2 = np.einsum("ij,ij->i", Gcart, Gcart)
nz = G2 > 1e-12
# smooth nG on the miller_g list
n_s = nG[mg[:, 0] % N1, mg[:, 1] % N2, mg[:, 2] % N3]
EH_s  = 0.5 * Om * np.sum(4 * np.pi / G2[nz] * np.abs(n_s[nz]) ** 2)
n_t = n_s + rho_hat
EH_t  = 0.5 * Om * np.sum(4 * np.pi / G2[nz] * np.abs(n_t[nz]) ** 2)
print(f"E_H(rho_tilde)          = {EH_s:.6f} Ha")
print(f"E_H(rho_tilde+rho_hat)  = {EH_t:.6f} Ha   [ABINIT smooth hartree = 58.428318]")
