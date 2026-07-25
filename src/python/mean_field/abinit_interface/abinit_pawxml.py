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
  shape_tab (nr,) | None   tabulated <shape_function> values on the main
                           grid (density convention g(r), no r^2), when the
                           XML tabulates them -- used by the plan-B2
                           analytic-vs-tabulated cross-check
  vbar (nr,)            blochl_local_ionic_potential
  ae_core, ps_core (nr,)   core densities
  dij0 (nstate,nstate)  kinetic_energy_differences (symmetric)
  exx_X (nstate,nstate)  exact_exchange_X_matrix (one-center exact exchange)
  exx_core_core (float)  core-core exchange energy (Ha)
  core_states: list of {n,l,f,e,id} | None   AE core orbitals metadata
  core_ae_wfc (ncore, nr) | None   AE core wavefunctions R(r) (not r*R),
                                   when the XML provides them (plan B3 input)
"""

import numpy as np
import xml.etree.ElementTree as ET


def _floats(text):
    return np.array([float(x) for x in text.split()], dtype=float)


def _grid_r(rg):
    """Radial grid values from a <radial_grid> element (PAW-XML eq forms)."""
    eq = rg.get("eq", "r=a*(exp(d*i)-1)")
    istart = int(rg.get("istart", 0))
    iend = int(rg.get("iend"))
    i = np.arange(istart, iend + 1).astype(float)
    if eq == "r=a*(exp(d*i)-1)":
        a = float(rg.get("a")); d = float(rg.get("d"))
        return a * (np.exp(d * i) - 1.0)
    if eq == "r=d*i":
        return float(rg.get("d")) * i
    if eq == "r=a*i/(n-i)":
        a = float(rg.get("a")); n = float(rg.get("n"))
        return a * i / (n - i)
    if eq == "r=a*(exp(d*i)-1)/(exp(d)-1)":   # rare variant
        a = float(rg.get("a")); d = float(rg.get("d"))
        return a * (np.exp(d * i) - 1.0) / (np.exp(d) - 1.0)
    raise NotImplementedError("abinit_pawxml: radial grid eq '%s' not supported" % eq)


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

    # --- radial grids (main = first; keep all by id for tagged functions) ---
    grids = {}
    for rg_el in findall("radial_grid"):
        grids[rg_el.get("id", "log1")] = _grid_r(rg_el)
    rg = find("radial_grid")
    a = float(rg.get("a", 0.0)); d = float(rg.get("d", 0.0))
    main_grid_id = rg.get("id", "log1")
    r = grids[main_grid_id]
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
    # tabulated shape values (type="num", or generators that tabulate the
    # analytic shape as well). Interpolate onto the main grid if tagged with
    # a different radial grid. Density convention g(r) -- no r^2.
    shape_tab = None
    if sf.text and sf.text.strip():
        vals = _floats(sf.text)
        gid = sf.get("grid", main_grid_id)
        rg_tab = grids.get(gid)
        if rg_tab is None or vals.size != rg_tab.size:
            raise RuntimeError(
                "abinit_pawxml: tabulated <shape_function> has %d values but "
                "grid '%s' has %s points" % (vals.size, gid,
                "?" if rg_tab is None else rg_tab.size))
        shape_tab = vals if gid == main_grid_id else np.interp(r, rg_tab, vals)

    # --- local ionic potential + core densities ---
    def radial(tag):
        el = find(tag)
        return _floats(el.text)[:nr] if el is not None else None
    vbar = radial("blochl_local_ionic_potential")
    zero_potential = radial("zero_potential")
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

    # --- AE core orbitals (plan B2/B3): optional, generator-dependent ---
    # Metadata from <core_states><state .../></core_states> when present;
    # radial functions from <ae_core_wavefunction state=...> elements
    # (R-form, like the valence partial waves). Both absent -> None.
    core_states = None
    cs_el = find("core_states")
    if cs_el is not None:
        core_states = [dict(n=(int(s.get("n")) if s.get("n") else None),
                            l=int(s.get("l")),
                            f=(float(s.get("f")) if s.get("f") else 0.0),
                            e=(float(s.get("e")) if s.get("e") else 0.0),
                            id=s.get("id"))
                       for s in cs_el.findall("state")]
    core_ae_wfc = None
    cw_els = findall("ae_core_wavefunction")
    if cw_els:
        if core_states is not None and len(core_states) == len(cw_els):
            cid = {st["id"]: k for k, st in enumerate(core_states)}
            core_ae_wfc = np.zeros((len(cw_els), nr))
            for el in cw_els:
                core_ae_wfc[cid[el.get("state")]] = _floats(el.text)[:nr]
        else:
            # no (usable) metadata block: keep document order, synthesize ids
            core_ae_wfc = np.stack([_floats(el.text)[:nr] for el in cw_els])
            if core_states is None:
                core_states = [dict(n=(int(el.get("n")) if el.get("n") else None),
                                    l=(int(el.get("l")) if el.get("l") else None),
                                    f=0.0, e=0.0, id=el.get("state"))
                               for el in cw_els]

    return dict(r=r, a=a, d=d, nr=nr, states=states, ns=ns, znucl=znucl,
                phi_ae=phi_ae, phi_ps=phi_ps, proj=proj,
                shape_type=shape_type, shape_rc=shape_rc, paw_radius=paw_radius,
                shape_tab=shape_tab,
                vbar=vbar, zero_potential=zero_potential,
                ae_core=ae_core, ps_core=ps_core,
                dij0=dij0, exx_X=exx_X, exx_core_core=exx_core_core,
                core_states=core_states, core_ae_wfc=core_ae_wfc)


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
