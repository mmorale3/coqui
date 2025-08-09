import json

from coqui._lib.eri_module import ThcCoulomb

def thc_default_params():
  """
  Default parameters for the THC Coulomb interaction handler.

  Returns
  -------
  dict
      Default parameters for the THC Coulomb interaction.
  """
  return {
      "init": True,
      "nIpts": 0,
      "thresh": 0.0,
      "storage": "incore",
      "save": "",
      "cd_dir": "",
      "chol_block_size": 8
  }

def make_thc_coulomb(mf, eri_params):
  """
  Create a THC Coulomb interaction handler using interpolative separable density fitting (ISDF).

  Parameters
  ----------
  mf : coqui Mf class
    The mean-field handler for the target system.

  eri_params: dict
    Parameters for the THC Coulomb interaction. Supported keys include:

      - init: bool, default=True
        If True, initializes the THC computation at construction.
        If False, defer initialization until `.init()` is called.

      - nIpts: int
        Number of THC interpolation points.
        Acts as one of the stopping criteria for the THC decomposition.

      - thresh: float
        Threshold for the THC decomposition.
        Also acts as a stopping criterion.

        Note: If both `nIpts` and `thresh` are provided, the THC algorithm terminates when either condition is met.

      - ecut: float, default=mf.ecutrho()
        Plane wave cutoff used for the evaluation of coulomb matrix elements.

      - storage: str, default="incore"
        How THC integrals are stored and accessed.
        Options:
          - "incore": kept in memory and reused on-the-fly.
          - "outcore": read from the HDF5 file as needed.

      - save: str, default=""
        HDF5 file to save the THC integrals.
        If empty, integrals are not saved.

      - cd_dir: str, default=""
        Directory to precomputed Cholesky-decomposed Coulomb integrals.
        If provided, least-square THC algorithm will be performed instead of ISDF.

      - chol_block_size: int, default=8
        Block size for Cholesky decomposition

  Returns
  -------
  CoQui ThcCoulomb class
  """
  return ThcCoulomb(mf, json.dumps(eri_params))
