import numpy as np


def sigma_from_poles(d, W):
    """Sigma_c(z) = sum_l W[:,l] W[:,l]^dagger / (z - d_l) as a callable z -> (nz, N, N)."""
    def S(z):
        z = np.atleast_1d(np.asarray(z, complex))
        return np.einsum('zl,il,jl->zij', 1.0 / (z[:, None] - d[None, :]), W, W.conj())
    return S


def greens_function(H, Sfun, z, S_ovlp=None):
    """G(z) = [z S - H - Sigma_c(z)]^-1 with H = h0 + Sigma_inf (f + Sigma_inf in the paper); returns (nz, N, N)."""
    z = np.atleast_1d(np.asarray(z, complex))
    N = H.shape[0]
    Sm = np.eye(N) if S_ovlp is None else S_ovlp
    Sig = Sfun(z)
    return np.linalg.inv(z[:, None, None] * Sm - H - Sig)


def spectral_function(H, Sfun, om, eta, S_ovlp=None, trace=False):
    """A(w) = (i/2pi)[G(w+i eta) - G(w+i eta)^dagger]; (nw, N, N) or its trace."""
    G = greens_function(H, Sfun, om + 1j * eta, S_ovlp)
    A = (G - np.conj(np.transpose(G, (0, 2, 1)))) * (1j / (2 * np.pi))
    return np.trace(A, axis1=1, axis2=2).real if trace else A


def upfolded_hamiltonian(H, d, W):
    """Hermitian upfolded matrix [[H, W],[W^dagger, diag d]] whose eigen-decomposition gives G exactly in Lehmann form."""
    N, Np = W.shape
    Ht = np.zeros((N + Np, N + Np), complex)
    Ht[:N, :N] = H; Ht[:N, N:] = W; Ht[N:, :N] = W.conj().T; Ht[N:, N:] = np.diag(d)
    return Ht


def lehmann_from_upfolded(H, d, W):
    """Poles e_m and residues |<i|m>|^2 (N x M) of G from the upfolded Hamiltonian (exact for the pole Sigma)."""
    e, V = np.linalg.eigh(upfolded_hamiltonian(H, d, W))
    N = H.shape[0]
    return e, V[:N, :]
