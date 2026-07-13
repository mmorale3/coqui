"""
Parser for ABINIT PAW-XML datasets (PAW-XML spec / ESL, `<paw_dataset version="0.7">`).
Parser for abinit2coqui's PAW path: extracts the partial waves, projectors,
core densities, local ionic potential, kinetic-energy differences (Dij0), the
compensation shape function, and -- crucially -- the `exact_exchange_X_matrix`
(= the one-center exact-exchange kernel deltaC / K_a that QE mishandles and ABINIT
provides directly), plus the core-core exchange energy.

Everything sits on a single logarithmic radial grid  r = a*(exp(d*i)-1), i=0..iend.

Returned dict keys:
  r (nr,), grid params (a,d,nr)
  states: list of {n,l,f,e,id}  (the valence partial-wave channels)
  phi_ae (nstate, nr)   AE partial waves  phi_i(r)
  phi_ps (nstate, nr)   PS partial waves  tilde-phi_i(r)
  proj   (nstate, nr)   projector functions  tilde-p_i(r)
  shape_type, shape_rc, paw_radius
  vbar (nr,)            blochl_local_ionic_potential
  ae_core, ps_core (nr,)   core densities
  dij0 (nstate,nstate)  kinetic_energy_differences (symmetric)
  exx_X (nstate,nstate)  exact_exchange_X_matrix (one-center exact exchange)
  exx_core_core (float)  core-core exchange energy (Ha)
"""

import os
import numpy as np
import xml.etree.ElementTree as ET


def _floats(text):
    return np.array([float(x) for x in text.split()], dtype=float)


def parse_pawxml(path):
    root = ET.parse(path).getroot()

    def find(tag):
        # tags are unnamespaced in PAW-XML
        return root.find(tag)

    def findall(tag):
        return root.findall(tag)

    # --- atomic number (nuclear charge Z) ---
    atom = find("atom")
    znucl = float(atom.get("Z")) if atom is not None and atom.get("Z") else None

    # --- radial grid (assume single grid 'log1') ---
    rg = find("radial_grid")
    a = float(rg.get("a")); d = float(rg.get("d"))
    istart = int(rg.get("istart")); iend = int(rg.get("iend"))
    i = np.arange(istart, iend + 1)
    r = a * (np.exp(d * i) - 1.0)
    nr = r.size

    # --- valence states ---
    states = []
    for s in find("valence_states").findall("state"):
        states.append(dict(n=(int(s.get("n")) if s.get("n") else None),
                           l=int(s.get("l")),
                           f=(float(s.get("f")) if s.get("f") else 0.0),
                           e=float(s.get("e")), id=s.get("id")))
    ns = len(states)
    id2idx = {st["id"]: k for k, st in enumerate(states)}

    def collect(tag):
        """Radial functions tagged by state=..., in valence-state order."""
        out = np.zeros((ns, nr), dtype=float)
        for el in findall(tag):
            k = id2idx[el.get("state")]
            out[k] = _floats(el.text)[:nr]
        return out

    phi_ae = collect("ae_partial_wave")
    phi_ps = collect("pseudo_partial_wave")
    proj = collect("projector_function")

    # --- shape function / radii ---
    sf = find("shape_function")
    shape_type = sf.get("type"); shape_rc = float(sf.get("rc"))
    paw_radius = float(find("paw_radius").get("rc"))

    # --- local ionic potential + core densities ---
    def radial(tag):
        el = find(tag)
        return _floats(el.text)[:nr] if el is not None else None
    vbar = radial("blochl_local_ionic_potential")
    ae_core = radial("ae_core_density")
    ps_core = radial("pseudo_core_density")

    # --- Dij0 (kinetic energy differences) : ns*ns matrix ---
    ked = _floats(find("kinetic_energy_differences").text)
    dij0 = ked.reshape(ns, ns) if ked.size == ns * ns else ked

    # --- one-center exact exchange ---
    xel = find("exact_exchange_X_matrix")
    exx_X = None
    if xel is not None and xel.text and xel.text.strip():
        v = _floats(xel.text)
        exx_X = v.reshape(ns, ns) if v.size == ns * ns else v
    exx_cc_el = find("exact_exchange")
    exx_core_core = float(exx_cc_el.get("core-core")) if exx_cc_el is not None else 0.0

    # --- optional companion file: PS ionic Hartree v_H[tilde-n_Zc] (vhtnzc) ---
    # Not in the PAW-XML; supplied out-of-band as `<pawxml>.vhtnzc` (one line of nr
    # floats on this radial grid, from an instrumented ABINIT run).  Enables the
    # full frozen-D^0 assembly in the converter; absent -> kinetic-only D^0.
    vhtnzc = None
    vpath = path + ".vhtnzc"
    if os.path.exists(vpath):
        vhtnzc = _floats(open(vpath).read())[:nr]

    return dict(r=r, a=a, d=d, nr=nr, states=states, ns=ns, znucl=znucl,
                phi_ae=phi_ae, phi_ps=phi_ps, proj=proj,
                shape_type=shape_type, shape_rc=shape_rc, paw_radius=paw_radius,
                vbar=vbar, ae_core=ae_core, ps_core=ps_core, vhtnzc=vhtnzc,
                dij0=dij0, exx_X=exx_X, exx_core_core=exx_core_core)


if __name__ == "__main__":
    import sys
    p = parse_pawxml(sys.argv[1])
    print("PAW-XML: nr=%d  r=[%.3e..%.3f]  paw_radius=%.3f  shape=%s(rc=%.3f)"
          % (p["nr"], p["r"][0], p["r"][-1], p["paw_radius"], p["shape_type"], p["shape_rc"]))
    print("valence states (n,l,f,e):")
    for st in p["states"]:
        print("   ", st["id"], "n=%s l=%d f=%.2f e=%.4f" % (st["n"], st["l"], st["f"], st["e"]))
    print("partial-wave shapes: phi_ae%s phi_ps%s proj%s"
          % (p["phi_ae"].shape, p["phi_ps"].shape, p["proj"].shape))
    print("phi_ae[0] range [%.3f, %.3f]; proj[0] range [%.3f, %.3f]"
          % (p["phi_ae"][0].min(), p["phi_ae"][0].max(), p["proj"][0].min(), p["proj"][0].max()))
    print("Dij0 =\n", np.round(p["dij0"], 5))
    print("exact_exchange_X_matrix =\n", None if p["exx_X"] is None else np.round(p["exx_X"], 4))
    print("core-core exchange = %.5f Ha" % p["exx_core_core"])
    # sanity: AE partial wave should differ from PS inside paw_radius, match outside
    rc = p["paw_radius"]; out = p["r"] > rc
    print("phi_ae==phi_ps beyond rc? max|diff| = %.2e"
          % np.max(np.abs(p["phi_ae"][0][out] - p["phi_ps"][0][out])))
