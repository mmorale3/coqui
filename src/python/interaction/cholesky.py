import json

from coqui._lib.eri_module import CholCoulomb

def make_chol_coulomb(mf, params):
  return CholCoulomb(mf, json.dumps(params))
