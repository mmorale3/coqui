"""Readers for CoQui checkpoints (*.mbpt.h5) and THC files (*.thc.h5). Complex arrays are stored with a trailing dim 2."""
import numpy as np, h5py


def _c(a):
    a = np.asarray(a)
    return a[..., 0] + 1j * a[..., 1] if a.shape[-1] == 2 else a


class Checkpoint:
    """Dyson scGW checkpoint: system/, imaginary_fourier_transform/, mean_field/, scf/iterN/{G_tskij, Sigma_tskij, F_skij, Dm_skij, mu}.
    Conventions (verified in the real_axis_GW project): Sigma_tskij is the correlation part only; H0 + F_iter0 = H_KS
    (F0 contains Vxc); for it >= 1 F = Hartree + exchange (no H0); tau mesh stored as x in [-1,1], tau = (x+1) beta/2."""
    def __init__(self, fname):
        self.fname = fname
        with h5py.File(fname, 'r') as f:
            g = f['imaginary_fourier_transform']
            self.beta = float(g['beta'][()]); self.wmax = float(g['wmax'][()]); self.eps = float(g['eps'][()])
            self.lam = float(g['lambda'][()]); self.basis = g['basis'][()]
            self.x_f = np.array(g['tau_mesh/fermion']); self.tau_f = (self.x_f + 1) * self.beta / 2
            self.x_b = np.array(g['tau_mesh/boson']); self.tau_b = (self.x_b + 1) * self.beta / 2
            self.iwn_f = np.array(g['iwn_mesh/fermion']); self.iwn_b = np.array(g['iwn_mesh/boson'])
            s = f['system']
            self.nk = int(s['number_of_kpoints'][()]); self.nk_ibz = int(s['number_of_kpoints_ibz'][()])
            self.nb = int(s['number_of_orbitals'][()]); self.ns = int(s['number_of_spins'][()])
            self.kpts = np.array(s['kpoints']); self.kpts_crys = np.array(s['kpoints_crys']); self.k_weight = np.array(s['k_weight'])
            self.qk_to_k2 = np.array(s['qk_to_k2']); self.qminus = np.array(s['qminus']); self.kp_to_ibz = np.array(s['kp_to_ibz'])
            self.H0 = _c(s['H0_skij']); self.S = _c(s['S_skij'])
            self.eig = np.array(f['mean_field/eigvals'])
            self.final_iter = int(f['scf/final_iter'][()])
            self.iters = sorted(int(k[4:]) for k in f['scf'].keys() if k.startswith('iter'))
            self.mu = {it: float(f[f'scf/iter{it}/mu'][()]) for it in self.iters}
            if 'scf/mu_history' in f: self.mu_history = np.array(f['scf/mu_history'])

    def _read(self, it, name):
        with h5py.File(self.fname, 'r') as f:
            return _c(f[f'scf/iter{it}/{name}'])

    def Sigma(self, it=None):  return self._read(self.final_iter if it is None else it, 'Sigma_tskij')   # (nt, ns, nk, nb, nb)
    def G(self, it=None):      return self._read(self.final_iter if it is None else it, 'G_tskij')
    def F(self, it=None):      return self._read(self.final_iter if it is None else it, 'F_skij')
    def Dm(self, it=None):     return self._read(self.final_iter if it is None else it, 'Dm_skij')
    def mu_of(self, it=None):  return self.mu[self.final_iter if it is None else it]

    def static_hamiltonian(self, it=None):
        """H0 + F_it (= h0 + Hartree + exchange = f + Sigma_inf of the paper) for it >= 1; (ns, nk, nb, nb)."""
        it = self.final_iter if it is None else it
        assert it >= 1, "F_iter0 contains Vxc; use it >= 1"
        return self.H0 + self.F(it)


class THC:
    def __init__(self, fname):
        with h5py.File(fname, 'r') as f:
            self.Np = int(f['Np'][()])
            self.X = _c(f['collocation_matrix'])          # (ns, nk, Np, nb)
            self.Z = _c(f['coulomb_matrix'])              # (nq, Np, Np)
            self.kpts = np.array(f['kpts']); self.qpts = np.array(f['qpts'])
            self.nb = int(f['number_of_bands'][()]) if 'number_of_bands' in f else self.X.shape[-1]
