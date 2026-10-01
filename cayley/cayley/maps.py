import numpy as np

def cayley(w, wp, mu=0.0):
    """u(w) = (w - mu + i wp)/(w - mu - i wp): real axis -> unit circle, mu -> -1, +-inf -> +1, mu+wp -> +i (|u|<=1 for Im w<=0)."""
    return (w - mu + 1j * wp) / (w - mu - 1j * wp)

def inv_cayley(u, wp, mu=0.0):
    """real pole positions from unit-circle points: mu + wp cot(theta/2), u = e^{i theta}."""
    th = np.angle(u)
    return mu + wp / np.tan(th / 2)

def disk_variable(zeta, wp, mu=0.0):
    """z = (zeta - mu - i wp)/(zeta - mu + i wp): upper half plane -> unit disk, z=0 <-> zeta = mu + i wp."""
    return (zeta - mu - 1j * wp) / (zeta - mu + 1j * wp)

def zeta_from_disk(z, wp, mu=0.0):
    return mu + 1j * wp * (1 + z) / (1 - z)
