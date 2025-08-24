import json

from coqui._lib.mf_module import Mf


def make_mf(mpi, params, mf_type):
    return Mf(mpi, json.dumps(params), mf_type)
