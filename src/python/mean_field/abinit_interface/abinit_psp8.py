"""
Parser for ABINIT norm-conserving pseudopotentials in psp8 format (pspcod=8,
ONCVPSP output).  Used by abinit2coqui to build the norm-conserving `/Hamiltonian` block:
the KB coefficients (dion), the radial nonlocal projectors chi_ln(r) (for beta(G)),
and the local potential Vloc(r) (for pp_local_component_nc / V_loc(G)).

psp8 layout (ONCVPSP-3.x):
  line1: title  (element  code  r_core=...)
  line2: zatom zion pspd
  line3: pspcod pspxc lmax lloc mmax r2well
  line4: rchrg fchrg qchrg           (NLCC; fchrg==0 -> no core charge)
  line5: nproj(0..4)                 (# projectors per angular momentum l)
  line6: extension_switch [, ...]
  then, for each l with nproj[l] > 0:
     header row:  l   ekb_1 ekb_2 ... ekb_{nproj[l]}     (KB energies = dion diag)
     mmax rows:   idx  r   chi_{l,1}(r) ... chi_{l,nproj[l]}(r)
  then the local potential block:
     header row:  lloc
     mmax rows:   idx  r   Vloc(r)
  (if fchrg>0 a model-core-charge block follows; not needed for Si_GGA_noNLCC)
"""

import numpy as np


def _f(tok):
    """psp8 uses Fortran 'D' exponents."""
    return float(tok.replace("D", "E").replace("d", "E"))


def parse_psp8(path):
    """Parse a psp8 file. Returns a dict:
      zatom, zion, lmax, lloc, mmax, nproj (list, len 5),
      r (mmax,), vloc (mmax,),
      channels: list of dicts {l, ekb (nproj_l,), chi (mmax, nproj_l)}
      nh: total lm*n projector count per atom, and ln->(l) expansion helpers.
    """
    with open(path) as fh:
        lines = fh.readlines()

    # header
    zatom, zion = _f(lines[1].split()[0]), _f(lines[1].split()[1])
    t3 = lines[2].split()
    pspcod, pspxc, lmax, lloc, mmax = (int(t3[0]), int(t3[1]), int(t3[2]),
                                       int(t3[3]), int(t3[4]))
    assert pspcod == 8, "not a psp8 file (pspcod=%d)" % pspcod
    t4 = lines[3].split()
    fchrg = _f(t4[1]) if len(t4) > 1 else 0.0
    nproj = [int(x) for x in lines[4].split()[:5]]

    # locate the first data block: line 5 is extension_switch; blocks start at line 6
    idx = 6
    channels = []
    for l in range(len(nproj)):
        npl = nproj[l]
        if npl == 0:
            continue
        # header row: "l  ekb_1 ... ekb_npl"
        hdr = lines[idx].split()
        # first token is the l index (int); the rest are ekb (may be fewer tokens
        # if wrapped, but ONCVPSP writes them on one line)
        ekb = np.array([_f(x) for x in hdr[1:1 + npl]], dtype=float)
        idx += 1
        chi = np.empty((mmax, npl), dtype=float)
        r = np.empty(mmax, dtype=float)
        for i in range(mmax):
            tok = lines[idx + i].split()
            r[i] = _f(tok[1])
            for j in range(npl):
                chi[i, j] = _f(tok[2 + j])
        idx += mmax
        channels.append(dict(l=l, ekb=ekb, chi=chi, r=r))

    # local potential block: header row (lloc) then mmax rows (idx, r, Vloc)
    # idx currently points at the vloc header row
    idx += 1
    rloc = np.empty(mmax, dtype=float)
    vloc = np.empty(mmax, dtype=float)
    for i in range(mmax):
        tok = lines[idx + i].split()
        rloc[i] = _f(tok[1])
        vloc[i] = _f(tok[2])

    # model core charge (NLCC) block, present when fchrg > 0: mmax rows of
    # idx, r, rhoc(, derivatives). psp8 stores 4*pi*n_c(r) (ABINIT psp8
    # convention; NOT r^2-weighted) -- consumers divide by 4*pi for the
    # number density. NOTE (plan B2): convention unverified against a real
    # NLCC psp8 on this machine; recheck on the first NLCC pseudo.
    rhoc = None
    if fchrg > 0.0:
        idx += mmax
        rhoc = np.empty(mmax, dtype=float)
        for i in range(mmax):
            rhoc[i] = _f(lines[idx + i].split()[2])

    # shared radial grid (all channels + vloc share the same r in psp8)
    r = channels[0]["r"] if channels else rloc

    # projector bookkeeping: expand (l, n) -> lm with m-degeneracy (2l+1)
    # nh = Σ_l nproj[l] * (2l+1)
    nh = sum(nproj[l] * (2 * l + 1) for l in range(len(nproj)))

    return dict(zatom=zatom, zion=zion, lmax=lmax, lloc=lloc, mmax=mmax,
                fchrg=fchrg, nproj=nproj, r=r, vloc=vloc, channels=channels,
                nh=nh, rhoc=rhoc)


def dion_diag(psp):
    """Return the NC KB coefficients as a diagonal per-(l,n) list, in the (l,n)
    order used to lay out projectors.  For NC pseudos dion is diagonal:
    D = diag over all (l, n) channels of ekb_{l,n}.  Length = Σ_l nproj[l].
    """
    d = []
    for ch in psp["channels"]:
        d.extend(list(ch["ekb"]))
    return np.array(d, dtype=float)


if __name__ == "__main__":
    import sys
    psp = parse_psp8(sys.argv[1])
    print("zatom=%g zion=%g lmax=%d lloc=%d mmax=%d nproj=%s nh(per atom)=%d"
          % (psp["zatom"], psp["zion"], psp["lmax"], psp["lloc"], psp["mmax"],
             psp["nproj"], psp["nh"]))
    for ch in psp["channels"]:
        print("  l=%d nproj=%d ekb=%s  chi range [%.3e, %.3e]"
              % (ch["l"], ch["chi"].shape[1], ch["ekb"],
                 ch["chi"].min(), ch["chi"].max()))
    print("  r: [%.4g .. %.4g] (%d pts);  Vloc(r->0)=%.4f  Vloc(r=rmax)=%.4f"
          % (psp["r"][0], psp["r"][-1], len(psp["r"]), psp["vloc"][0], psp["vloc"][-1]))
    print("  dion diag =", dion_diag(psp))
