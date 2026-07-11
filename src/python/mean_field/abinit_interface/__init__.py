"""
ABINIT -> CoQuí mean-field converter.

Produces a self-contained HDF5 file in the CoQuí "bdft" backend schema from an
ABINIT WFK netCDF file, so an ABINIT SCF can drive CoQuí (RPA/GW/...) via
`mf_source = bdft` without QE or pw2coqui.

Norm-conserving and PAW pseudopotentials, no symmetry reduction.
See abinit2coqui.py and README.md.
"""

from .abinit2coqui import convert, read_wfk

__all__ = ["convert", "read_wfk"]
