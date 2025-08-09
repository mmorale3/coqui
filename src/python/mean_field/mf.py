import json

from coqui._lib.mf_module import Mf


def make_mf(mpi, mf_params, mf_type):
    return Mf(mpi, json.dumps(mf_params), mf_type)
