"""
validate_paw_overlap.py -- PAW overlap (S) diagnostic for the ABINIT converter.

For a converted PAW bdft h5, computes  <psi~_nk | S | psi~_nk>  per occupied state,
where  S = 1 + sum_a sum_ij qq_nt[nt_a](i,j) |beta_ai><beta_aj| .  For a correct PAW
mean field this must equal 1 for every occupied band; the k-summed  sum f_nk <psi|S|psi>
must equal the physical electron count.  A deficit localizes the nelec bug to the
augmentation overlap (qq_nt / projector normalization) vs the smooth-grid truncation.

Self-contained: reads only the converter's h5 (orbitals, projectors, qq_nt, occ).
Usage: python validate_paw_overlap.py abinit_paw.h5
"""
import sys
import numpy as np
import h5py

H5 = sys.argv[1] if len(sys.argv) > 1 else "abinit_paw.h5"


def cplx(a):
    return a[..., 0] + 1j * a[..., 1]


def main():
    f = h5py.File(H5, "r")
    O = f["Orbitals"]; P = f["Hamiltonian/paw"]
    nsp_spin = int(O.attrs["number_of_spins"])
    nk = int(O.attrs["number_of_kpoints"])
    nbnd = int(O.attrs["number_of_bands"])
    occ = O["occ"][:]                                   # (nspin,nk,nbnd)
    miller_wfc = O["miller_wfc"][:]                     # (ngm_wfc,3)
    wfc_index = {(int(g[0]), int(g[1]), int(g[2])): i for i, g in enumerate(miller_wfc)}

    nh = P["proj_per_atom"][:]                          # (nsp,)
    ofs = P["projector_offset"][:]                      # (nat,)
    atomic_id = P["atomic_id"][:]                       # (nat,) 0-based species
    qq_nt = P["qq_nt"][:]                               # (nsp,nhm,nhm)
    nat = len(atomic_id)
    nkb = int(P.attrs["total_num_of_proj"])

    tot = 0.0
    print("per-k <psi~|S|psi~> for occupied bands (should be ~1.0):")
    for ik in range(nk):
        psi = cplx(O["psi_s0_k%d" % ik][:])            # (nbnd, ngm_wfc)
        beta = cplx(P["projector_k%d" % ik][:])        # (nkb, npw_k) = stored beta
        mk = P["miller_k%d" % ik][:]                   # (npw_k,3)
        # map projector G-list into the shared wfc grid columns
        col = np.array([wfc_index.get((int(g[0]), int(g[1]), int(g[2])), -1) for g in mk])
        good = col >= 0
        colg = col[good]
        psi_on_k = psi[:, colg]                        # (nbnd, npw_k_good)
        beta_g = beta[:, good]                         # (nkb, npw_k_good)
        # <beta_i|psi_n> = sum_G conj(beta_i(G)) psi_n(G)
        Bproj = np.conj(beta_g) @ psi_on_k.T            # (nkb, nbnd)
        # smooth norm on the (full) wfc grid
        smooth = np.einsum("ng,ng->n", np.conj(psi), psi).real  # (nbnd,)
        # augmentation per band
        aug = np.zeros(nbnd)
        for a in range(nat):
            nt = int(atomic_id[a]); nha = int(nh[nt]); o = int(ofs[a])
            q = qq_nt[nt, :nha, :nha]                   # (nha,nha)
            Ba = Bproj[o:o + nha, :]                    # (nha, nbnd)
            # <psi|beta_i> qq_ij <beta_j|psi> = conj(Bproj_i) q_ij Bproj_j
            aug += np.einsum("in,ij,jn->n", np.conj(Ba), q, Ba).real
        S = smooth + aug
        occk = occ[0, ik]
        nocc = int(round(occk.sum() * 2)) if occk.max() <= 0.51 else int(round(occk.sum()))
        # report the lowest ~4 bands (occupied for Si)
        sel = np.where(occk > 1e-6)[0]
        Ssel = S[sel]
        print("  k%d: nocc=%d  smooth[occ]=%s  S[occ]=%s"
              % (ik, len(sel),
                 np.array2string(smooth[sel], precision=3, max_line_width=200),
                 np.array2string(Ssel, precision=4, max_line_width=200)))
        tot += float(np.sum(occk * S))
    # CoQui divides occ by 2 for nspin=1; total electron count comparison
    print("\nSum_nk f_nk <psi|S|psi> = %.5f  (CoQui per-spin target; nelec_reported=3.75)" % tot)
    print("  smooth-only Sum f<psi|psi> = %.5f" % _smooth_total(f, wfc_index))
    f.close()


def _smooth_total(f, wfc_index):
    O = f["Orbitals"]
    nk = int(O.attrs["number_of_kpoints"]); occ = O["occ"][:]
    t = 0.0
    for ik in range(nk):
        psi = cplx(O["psi_s0_k%d" % ik][:])
        sm = np.einsum("ng,ng->n", np.conj(psi), psi).real
        t += float(np.sum(occ[0, ik] * sm))
    return t


if __name__ == "__main__":
    main()
