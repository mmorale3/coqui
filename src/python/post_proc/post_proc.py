import json

from coqui._lib import post_proc_module as pproc_mod


def ac(mf, params):
    pproc_mod.ac(mf, json.dumps(params))


def band_interpolation(mf, params):
    pproc_mod.band_interpolation(mf, json.dumps(params))


def spectral_interpolation(mf, params):
    pproc_mod.spectral_interpolation(mf, json.dumps(params))


def local_dos(mf, params):
    pproc_mod.local_dos(mf, json.dumps(params))


def unfold_bz(mf, params):
    pproc_mod.unfold_bz(mf, json.dumps(params))


def dump_vxc(mf, params):
    pproc_mod.dump_vxc(mf, json.dumps(params))


def dump_hartree(mf, params):
    pproc_mod.dump_hartree(mf, json.dumps(params))

