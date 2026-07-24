#!/usr/bin/env python3
"""Compute the PAW projector-overlap amplitude  P_{i,a alpha}^k = <beta_{a alpha}^k | psi_i^k>
per band, directly from existing CoQui/QE h5 (no rerun).  For each band i we report
max_{a,alpha} |P| (and the l-channel that attains it), aggregated as the max over k-points.

Inputs per run dir <rd>:
  <rd>/nscf/out/si.coqui.h5         -> /Hamiltonian/paw/projector_k{ik} {nproj,npw,2}
                                       /Hamiltonian/paw/miller_k{ik}    {npw,3}
                                       /Hamiltonian/Species/nt0/nhtol   {nh}
                                       /Hamiltonian/paw/proj_per_atom, projector_offset
                                       /Orbitals/eigval {1,nk,nband}, /Orbitals/occ
  <rd>/nscf/out/si.save/wfc{ik+1}.hdf5 -> /evc {nband,2*npw}, /MillerIndices {npw,3}

Usage: proj_amplitude.py <run_dir> <pseudo_label> <out_csv>
"""
import sys, h5py, numpy as np

rd, label, out = sys.argv[1], sys.argv[2], sys.argv[3]
Ha2eV = 27.211386

ch = h5py.File(f"{rd}/nscf/out/si.coqui.h5", "r")
paw = ch["Hamiltonian/paw"]
nhtol_sp = ch["Hamiltonian/Species/nt0/nhtol"][:]      # l of each of nh projectors on the species
ppa = int(paw["proj_per_atom"][0])                     # projectors per atom
off = paw["projector_offset"][:]                        # start index of each atom in the proj block
natom = len(off)
# l for each row of the (natom*ppa) projector block
lrow = np.concatenate([nhtol_sp[:ppa] for _ in range(natom)])

eig = ch["Orbitals/eigval"][0]    # {nk, nband}, Hartree
occ = ch["Orbitals/occ"][0]       # {nk, nband}
nk, nband = eig.shape
vbm = np.max(eig[occ > 0.5]) if np.any(occ > 0.5) else np.max(eig[:, :4])

def miller_key(M):
    return {tuple(int(x) for x in M[g]): g for g in range(M.shape[0])}

Ls = [0, 1, 2]
maxPl = {l: np.zeros(nband) for l in Ls}               # per-l max|P| per band (max over k)
norm0 = norm_hi = None
for ik in range(nk):
    proj = paw[f"projector_k{ik}"][:]                  # {nproj, npw, 2}
    Pc = proj[..., 0] + 1j * proj[..., 1]              # {nproj, npw}
    Mc = paw[f"miller_k{ik}"][:]                        # {npw,3}
    w = h5py.File(f"{rd}/nscf/out/si.save/wfc{ik+1}.hdf5", "r")
    evc = w["evc"][:]                                   # {nband, 2*npw}
    Mq = w["MillerIndices"][:]                          # {npw,3}
    psi = evc[:, 0::2] + 1j * evc[:, 1::2]              # {nband, npw_q}
    # reorder QE columns to the CoQui projector G-ordering via Miller indices
    kq = miller_key(Mq)
    idx = np.array([kq.get(tuple(int(x) for x in Mc[g]), -1) for g in range(Mc.shape[0])])
    good = idx >= 0
    psi_r = np.zeros((psi.shape[0], Mc.shape[0]), dtype=complex)
    psi_r[:, good] = psi[:, idx[good]]
    if ik == 0:
        nn = np.sum(np.abs(psi_r) ** 2, axis=1)         # plane-wave norm per band
        norm0, norm_hi = nn[0], nn[-1]
    P = np.conjugate(Pc) @ psi_r.T                      # {nproj, nband}
    aP = np.abs(P)                                      # {nproj, nband}
    for l in Ls:
        rows = np.where(lrow == l)[0]
        if len(rows) == 0: continue
        pk = aP[rows].max(axis=0)
        upd = pk > maxPl[l]
        maxPl[l][upd] = pk[upd]
    w.close()

with open(out, "w") as fo:
    fo.write("pseudo,band,eig_eV_rel_vbm,maxP_s,maxP_p,maxP_d\n")
    for i in range(nband):
        e = (np.min(eig[:, i]) - vbm) * Ha2eV          # lowest eig of band i across k, rel VBM
        fo.write(f"{label},{i},{e:.4f},{maxPl[0][i]:.6f},{maxPl[1][i]:.6f},{maxPl[2][i]:.6f}\n")
sp = np.maximum(maxPl[0], maxPl[1])                     # shared s+p channels
print(f"{label}: nk={nk} nband={nband} nproj={Pc.shape[0]} ppa={ppa} natom={natom} "
      f"lrows={list(map(int,lrow))}")
print(f"  evc PW-norm: band0={norm0:.4f} bandHi={norm_hi:.4f}")
print(f"  s+p max|P|:  occ(0-3)={sp[:4].max():.3f}  low-virt(4-30)={sp[4:30].max():.3f}  "
      f"high-virt(50+)={sp[50:].max():.3f}")
print(f"  d   max|P|:  occ(0-3)={maxPl[2][:4].max():.3f}  high-virt(50+)={maxPl[2][50:].max():.3f}")
