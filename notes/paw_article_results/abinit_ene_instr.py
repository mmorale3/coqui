#!/usr/bin/env python3
"""Env-gated full-precision energy-ledger instrumentation for ABINIT 10.6.7.

Purpose: give a term-by-term ABINIT reference for CoQui's E_1e / E_H / E_x on a
volume series (the Si PAW EOS exchange defect, 2026-07-29).  ABINIT's .abo
prints its PAW energy as a single lumped 'PAW spherical_terms', and prints
nothing at all for e_fock0 / e_fock, so the .abo alone cannot be mapped onto
CoQui's decomposition.  Both dumps below are pure `write` statements behind an
environment variable -- they cannot change any computed value.

Two insertion sites:

  1. 67_common/m_common.F90 :: prtene
     Dumps every field of the `energies` container at es24.16 (kinetic,
     hartree, xc, xcdc, localpsp, corepsp, nlpsp_vfock, fock, fock0, fockdc,
     eigenvalues, ewald, epaw, epaw_dc, epaw_xc, epaw_core, entropy) plus the
     two assembled totals.  Gate: ABI_DUMP_ENE.

     Why e_fock0 matters: at the FIRST step of a one-shot hybrid (nstep2 1,
     getwfk from a converged PBE run) the Fock operator is built from the input
     PBE orbitals, so e_fock0 is exactly the smooth plane-wave Fock energy
     evaluated ON PBE ORBITALS.  The .abo's DC total is NOT usable for this --
     it goes through e_eigenvalues, which comes from one partially-converged
     diagonalization and is noisy at the 100 mHa level across a volume series
     (measured: eos_exx500 etotal2 = -9.3556, -9.3332, -9.2244, -9.3353 for
     a = 10.15..10.45, i.e. non-monotone garbage).

  2. 65_paw/m_paw_denpot.F90 :: pawdenpot
     Dumps the individual on-site pieces that get summed into
     paw_energies%epaw: e1t10 (= sum_ij rho_ij dij0, the frozen one-centre
     D^0 contraction -- the direct analogue of CoQui's Tr[gamma dion]), eh2
     (one-centre Hartree; note epaw takes HALF of it), e1xc, etild1xc,
     exccore, efock, efockdc, ehnzc, ekincore, eh2dc.  Gate: ABI_DUMP_PAWENE.

     These are subroutine locals, not fields of any type, so they are summed
     over my_comm_atom here when paral_atom is on -- otherwise a run with atom
     parallelism would silently report one group's partial sums.

Usage (on the machine holding the ABINIT source tree):
    python3 abinit_ene_instr.py --root ~/abinit_build/abinit-10.6.7
    python3 abinit_ene_instr.py --root ~/abinit_build/abinit-10.6.7 --revert
    python3 abinit_ene_instr.py --root ~/abinit_build/abinit-10.6.7 --check
"""
import argparse
import os
import sys

MARKER = "COQUI-INSTR"
BAKSUF = ".coqui_orig"

# --- site 1: m_common.F90 :: prtene ----------------------------------------

C_DECL_ANCHOR = " real(dp) :: eent,enevalue,etotal,etotaldc,exc_semilocal,el_temp\n"
C_DECL_ADD = (
    "!" + MARKER + ": locals for the env-gated energy dump\n"
    " character(len=16) :: coqui_env_\n"
    " integer :: coqui_st_\n"
)

C_DUMP_ANCHOR = " call energies%eval_eint(dtset,usepaw,optdc,etotal,etotaldc)\n"


def _emit(label, expr, unit="iout"):
    """One dumped scalar: a write into msg followed by wrtout."""
    return ("   write(msg,'(a,es24.16)') ' COQUI_ENE %-14s = ',%s\n"
            "   call wrtout(%s,msg)\n" % (label, expr, unit))


C_FIELDS = [
    ("kinetic",      "energies%e_kinetic"),
    ("hartree",      "energies%e_hartree"),
    ("xc",           "energies%e_xc"),
    ("xcdc",         "energies%e_xcdc"),
    ("localpsp",     "energies%e_localpsp"),
    ("corepsp",      "energies%e_corepsp"),
    ("corepspdc",    "energies%e_corepspdc"),
    ("nlpsp_vfock",  "energies%e_nlpsp_vfock"),
    ("fock",         "energies%e_fock"),
    ("fock0",        "energies%e_fock0"),
    ("fockdc",       "energies%e_fockdc"),
    ("eigenvalues",  "energies%e_eigenvalues"),
    ("ewald",        "energies%e_ewald"),
    ("epaw",         "energies%paw%epaw"),
    ("epaw_dc",      "energies%paw%epaw_dc"),
    ("epaw_xc",      "energies%paw%epaw_xc"),
    ("epaw_core",    "energies%paw%epaw_core"),
    ("entropy",      "energies%e_entropy"),
    ("etotal",       "etotal"),
    ("etotaldc",     "etotaldc"),
]

C_DUMP_ADD = (
    "\n!" + MARKER + ": full-precision energy ledger, env-gated (ABI_DUMP_ENE).\n"
    "!Pure output -- no computed quantity is touched.\n"
    " call get_environment_variable(\"ABI_DUMP_ENE\",coqui_env_,status=coqui_st_)\n"
    "!status: 0 = found, 1 = not set, -1 = found but value longer than coqui_env_.\n"
    "!Accept <=0 so that an over-long value still enables the dump.\n"
    " if (coqui_st_<=0) then\n"
    "   write(msg,'(a,i3)') ' COQUI_ENE optdc          = ',optdc\n"
    "   call wrtout(iout,msg)\n"
    + "".join(_emit(lab, ex) for lab, ex in C_FIELDS) +
    " end if\n"
)

# --- site 2: m_paw_denpot.F90 :: pawdenpot ---------------------------------

P_DECL_ANCHOR = " real(dp) :: intvh,intg,eshift,eh2dc,ehpw\n"
P_DECL_ADD = (
    "!" + MARKER + ": locals for the env-gated on-site energy / kernel dumps\n"
    " character(len=16) :: coqui_env_\n"
    " integer :: coqui_st_\n"
    " integer :: coqui_iu_,coqui_k1_,coqui_k2_\n"
    " real(dp) :: coqui_arr_(10)\n"
)

P_DUMP_ANCHOR = "!In case we have an entropy associated with PAW contribution\n"

P_FIELDS = [
    ("e1t10",    1),   # sum_ij rho_ij dij0   <-> CoQui Tr[gamma dion]
    ("eh2",      2),   # one-centre Hartree (epaw uses HALF of this)
    ("e1xc",     3),
    ("etild1xc", 4),
    ("exccore",  5),
    ("efock",    6),   # one-centre Fock (vv + cv)
    ("efockdc",  7),
    ("ehnzc",    8),
    ("ekincore", 9),
    ("eh2dc",   10),
]

P_DUMP_ADD = (
    "\n!" + MARKER + ": env-gated dump of the individual on-site energy pieces\n"
    "!(ABI_DUMP_PAWENE).  These are subroutine locals, so they must be reduced\n"
    "!over the atom communicator when atom parallelism is active.\n"
    " if (option/=1.and.ipert==0) then\n"
    "   call get_environment_variable(\"ABI_DUMP_PAWENE\",coqui_env_,status=coqui_st_)\n"
    "   if (coqui_st_<=0) then\n"
    "     coqui_arr_(1)=e1t10    ; coqui_arr_(2)=eh2      ; coqui_arr_(3)=e1xc\n"
    "     coqui_arr_(4)=etild1xc ; coqui_arr_(5)=exccore  ; coqui_arr_(6)=efock\n"
    "     coqui_arr_(7)=efockdc  ; coqui_arr_(8)=ehnzc    ; coqui_arr_(9)=ekincore\n"
    "     coqui_arr_(10)=eh2dc\n"
    "     if (paral_atom) call xmpi_sum(coqui_arr_,my_comm_atom,ierr)\n"
    + "".join(
        "     write(msg,'(a,es24.16)') ' COQUI_PAWENE %-9s = ',coqui_arr_(%d)\n"
        "     call wrtout(std_out,msg,'COLL')\n" % (lab, idx)
        for lab, idx in P_FIELDS
    ) +
    "   end if\n"
    " end if\n\n"
)

# --- site 3: m_paw_denpot.F90 :: pawdenpot, one-center Fock KERNEL ----------
#
# `efock`/`efockdc` give the on-site Fock ENERGY split (efockdc = the vv
# one-centre exchange, efock = vv + cv), but not why it differs from CoQui's.
# This dumps the operator itself:
#
#   pawtab%eijkl(klmn,klmn1) = <phi_i phi_j|v|phi_k phi_l>^AE
#                            - <phit_i phit_j + Qhat|v|phit_k phit_l + Qhat>
#
# built in 65_paw/m_paw_init.F90:586-655 -- the SAME tensor as CoQui's
# `Onecenter/deltaC`, on packed pairs klmn = jlmn*(jlmn-1)/2 + ilmn (ilmn<=jlmn)
# instead of the unpacked (nh,nh,nh,nh).  NOTE pawinit fills only klmn<=klmn1
# (`k1min=klmn`), so the lower triangle reads back as zero -- symmetrize before
# comparing.  `ex_cvij` (core-valence) and ABINIT's own `rhoijp` are dumped
# alongside so the kernel can be contracted with ABINIT's density matrix, which
# separates a kernel error from a becsum error.
#
# Gate: ABI_DUMP_PAWKERNEL.  Writes PAWKERNEL.dat in the run directory.

K_DUMP_ANCHOR = ("       paw_ij(iatom)%dijfock(:,:)="
                 "dijfock_vv(:,:)+dijfock_cv(:,:)\n")

K_DUMP_ADD = (
    "!" + MARKER + ": env-gated dump of the one-centre Fock kernel\n"
    "!(ABI_DUMP_PAWKERNEL).  Pure output.\n"
    "       call get_environment_variable(\"ABI_DUMP_PAWKERNEL\","
    "coqui_env_,status=coqui_st_)\n"
    "       if (coqui_st_<=0) then\n"
    "         open(newunit=coqui_iu_,file='PAWKERNEL.dat',"
    "position='append',action='write')\n"
    "         write(coqui_iu_,'(a,4i8)') 'ATOM ',iatom,itypat,"
    "pawtab(itypat)%lmn_size,pawtab(itypat)%lmn2_size\n"
    "         do coqui_k1_=1,pawtab(itypat)%lmn_size\n"
    "           write(coqui_iu_,'(a,7i6)') 'INDLMN ',coqui_k1_,"
    "pawtab(itypat)%indlmn(1:6,coqui_k1_)\n"
    "         end do\n"
    "         do coqui_k1_=1,pawtab(itypat)%lmn2_size\n"
    "           write(coqui_iu_,'(a,3i6,es24.16)') 'PAIR ',coqui_k1_,"
    "pawtab(itypat)%indklmn(7,coqui_k1_),pawtab(itypat)%indklmn(8,coqui_k1_),"
    "pawtab(itypat)%dltij(coqui_k1_)\n"
    "         end do\n"
    "         do coqui_k1_=1,pawtab(itypat)%lmn2_size\n"
    "           do coqui_k2_=1,pawtab(itypat)%lmn2_size\n"
    "             write(coqui_iu_,'(a,2i6,es24.16)') 'EIJKL ',coqui_k1_,"
    "coqui_k2_,pawtab(itypat)%eijkl(coqui_k1_,coqui_k2_)\n"
    "           end do\n"
    "         end do\n"
    "         if (allocated(pawtab(itypat)%ex_cvij)) then\n"
    "           do coqui_k1_=1,pawtab(itypat)%lmn2_size\n"
    "             write(coqui_iu_,'(a,i6,es24.16)') 'EXCVIJ ',coqui_k1_,"
    "pawtab(itypat)%ex_cvij(coqui_k1_)\n"
    "           end do\n"
    "         end if\n"
    "         write(coqui_iu_,'(a,2i8)') 'NRHOIJSEL ',"
    "pawrhoij(iatom)%nrhoijsel,pawrhoij(iatom)%cplex_rhoij\n"
    "         do coqui_k1_=1,pawrhoij(iatom)%nrhoijsel\n"
    "           write(coqui_iu_,'(a,2i6,es24.16)') 'RHOIJ ',coqui_k1_,"
    "pawrhoij(iatom)%rhoijselect(coqui_k1_),"
    "pawrhoij(iatom)%rhoijp(pawrhoij(iatom)%cplex_rhoij*(coqui_k1_-1)+1,1)\n"
    "         end do\n"
    "         close(coqui_iu_)\n"
    "       end if\n"
)

SITES = [
    ("src/67_common/m_common.F90",
     [(C_DECL_ANCHOR, C_DECL_ADD, "after"),
      (C_DUMP_ANCHOR, C_DUMP_ADD, "after")]),
    ("src/65_paw/m_paw_denpot.F90",
     [(P_DECL_ANCHOR, P_DECL_ADD, "after"),
      (P_DUMP_ANCHOR, P_DUMP_ADD, "before"),
      (K_DUMP_ANCHOR, K_DUMP_ADD, "after")]),
]


def apply(root, revert=False, check=False):
    rc = 0
    for rel, edits in SITES:
        path = os.path.join(root, rel)
        bak = path + BAKSUF
        if not os.path.isfile(path):
            print("MISSING %s" % path)
            return 2
        txt = open(path).read()
        patched = MARKER in txt

        if check:
            print("%-40s patched=%s backup=%s" % (rel, patched, os.path.isfile(bak)))
            continue

        if revert:
            if not os.path.isfile(bak):
                print("%-40s no backup -- nothing to revert" % rel)
                continue
            open(path, "w").write(open(bak).read())
            os.remove(bak)
            print("%-40s REVERTED" % rel)
            continue

        if patched:
            print("%-40s already patched -- skipped" % rel)
            continue

        if not os.path.isfile(bak):
            open(bak, "w").write(txt)

        for anchor, add, where in edits:
            n = txt.count(anchor)
            if n != 1:
                # A silently-missed anchor would produce a binary that looks
                # instrumented but prints nothing. Refuse instead.
                print("ANCHOR count %d (want 1) in %s for:\n%s" % (n, rel, anchor))
                return 3
            txt = txt.replace(anchor, anchor + add if where == "after" else add + anchor)
        open(path, "w").write(txt)
        print("%-40s PATCHED (%d edits)" % (rel, len(edits)))
    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="ABINIT source root (contains src/)")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sys.exit(apply(os.path.expanduser(a.root), a.revert, a.check))
