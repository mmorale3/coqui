import sys
import numpy as np
import sparse_ir

"""
Fourier transform on the imaginary axis based on IR basis and the sparse sampling technique.  
"""


def set_precision(precision):
    if isinstance(precision, str):
        if precision == "high":
            return 1e-15
        elif precision == "medium":
            return 1e-10
        elif precision == "low":
            return 1e-6
        else:
            raise ValueError("Unknown precision value: {}. ")
    return precision


def set_lambda(lmbda, coqui_cxx_style=False):
    if coqui_cxx_style:
        if lmbda <= 1000:
            return 1000
        elif lmbda > 1000 and lmbda <= 10000:
            return 10000
        elif lmbda > 10000 and lmbda <= 100000:
            return 100000
        elif lmbda > 100000 and lmbda <= 1000000:
            return 1000000
        else:
            raise ValueError("Invalid lambda value: {}. "
                             "Acceptable range is [1000, 1000000]".format(lmbda))
    return lmbda



class IAFT(object):
    """
    Driver for FT on the imaginary axis.
    Given inverse temperature, lambda and precision, the IAFT class evaluate the corresponding
    IR basis and sparse sampling points on-the-fly.

    Dependency:
        sparse-ir with xprec supports (https://sparse-ir.readthedocs.io/en/latest/)
        To install sparse-ir with xprec supports: "pip install sparse-ir[xprec]".

    Attributes:
    beta: float
        Inverse temperature (a.u.)
    lmbda: float
        Dimensionless lambda parameter for constructing the IR basis
    prec: float
        Precision for IR basis
    bases: sparse-ir.FiniteTempBasisSet
        IR basis instance
    tau_mesh_f: numpy.ndarray(dim=1)
        Fermionic tau sampling points
    tau_mesh_b: numpy.ndarray(dim=1)
        Bosonic tau sampling points
    wn_mesh_f: numpy.ndarray(dim=1)
        Fermionic Matsubara "indices" sampling points. NOT PHYSICAL FREQUENCIES.
        Physical Matsubara frequencies are wn_mesh_f * numpy.pi / beta
    wn_mesh_b: numpy.ndarray(dim=1)
        Bosonic Matsubara "indices" sampling points. NOT PHYSICAL FREQUENCIES.
        Physical Matsubara frequencies are wn_mesh_f * numpy.pi / beta
    nt_f: int
        Number of fermionic tau sampling points
    nt_b: int
        Number of bosonic tau sampling points
    nw_f: int
        Number of fermionic frequency sampling points
    nw_b: int
        Number of bosonic frequency sampling points
    """
    def __init__(self, beta: float, lmbda: float, prec = 1e-15, verbose: bool = True,
                 *, coqui_cxx_style: bool = False):
        """
        :param beta: float
            Inverse temperature (a.u.)
        :param lmbda: float
            Lambda parameter for constructing IR basis.
        :param prec: float
            Precision for IR basis
        """
        self.beta  = beta
        self.lmbda = set_lambda(lmbda, coqui_cxx_style=coqui_cxx_style)
        self.prec  = set_precision(prec)
        self.wmax  = self.lmbda / self.beta
        self.statisics = {'f', 'b'}

        self.bases = sparse_ir.FiniteTempBasisSet(beta=self.beta, wmax=self.wmax, eps=self.prec)
        self.tau_mesh_f = self.bases.smpl_tau_f.sampling_points
        self.tau_mesh_b = self.bases.smpl_tau_b.sampling_points
        self._wn_mesh_f = self.bases.smpl_wn_f.sampling_points
        self._wn_mesh_b = self.bases.smpl_wn_b.sampling_points
        self.nt_f, self.nw_f = self.tau_mesh_f.shape[0], self._wn_mesh_f.shape[0]
        self.nt_b, self.nw_b = self.tau_mesh_b.shape[0], self._wn_mesh_b.shape[0]

        Ttl_ff = self.bases.basis_f.u(self.tau_mesh_f).T
        Twl_ff = self.bases.basis_f.uhat(self._wn_mesh_f).T
        Ttl_bb = self.bases.basis_b.u(self.tau_mesh_b).T
        Twl_bb = self.bases.basis_b.uhat(self._wn_mesh_b).T

        self.Tlt_ff = np.linalg.pinv(Ttl_ff)
        self.Tlt_bb = np.linalg.pinv(Ttl_bb)
        self.Tlw_ff = np.linalg.pinv(Twl_ff)
        self.Tlw_bb = np.linalg.pinv(Twl_bb)

        # Ttw_ff = Ttl_ff * [Twl_ff]^{-1}
        self.Ttw_ff = np.dot(Ttl_ff, self.Tlw_ff)
        self.Twt_ff = np.dot(Twl_ff, self.Tlt_ff)
        self.Ttw_bb = np.dot(Ttl_bb, self.Tlw_bb)
        self.Twt_bb = np.dot(Twl_bb, self.Tlt_bb)

        if verbose:
            print(self)
            sys.stdout.flush()

    def __str__(self):
        return ("Mesh details on the imaginary axis\n" \
                "----------------------------------\n" \
                "Intermediate Representation\n" \
                "precision = {}\n" \
                "beta = {}\n" \
                "lambda = {}\n" \
                "nt_f, nw_f = {}, {}\n" \
                "nt_b, nw_b = {}, {}\n".format(self.prec, self.beta, self.lmbda, self.nt_f, self.nw_f,
                                                self.nt_b, self.nw_b))

    def __eq__(self, other):
        if not isinstance(other, IAFT):
            return NotImplemented

        return (
                self.beta == other.beta and
                self.lmbda == other.lmbda and
                self.prec == other.prec
        )

    def wn_mesh(self, stats: str, ir_notation: bool = True, *, positive_only=False):
        """
        Return Matsubara frequency indices.
        :param stats: str
            statistics: 'f' for fermions and 'b' for bosons
        :param ir_notation: bool
            Whether wn_mesh_interp is in sparse_ir notation where iwn = n*pi/beta for both fermions and bosons.
            Otherwise, iwn = (2n+1)*pi/beta  for fermions and 2n*pi/beta for bosons.

        :return: numpy.ndarray(dim=1)
            Matsubara frequency indices
        """
        if stats not in self.statisics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))
        wn_mesh = np.array(self._wn_mesh_f, dtype=int) if stats == 'f' else np.array(self._wn_mesh_b, dtype=int)
        if not ir_notation:
            wn_mesh = (wn_mesh-1)//2 if stats == 'f' else wn_mesh//2

        if positive_only:
            nw_half = wn_mesh.shape[0]//2
            return wn_mesh[nw_half:]
        else:
            return wn_mesh

    def tau_to_w(self, Ot, stats: str):
        """
        Fourier transform from imaginary-time axis to Matsubara-frequency axis
        :param Ot: numpy.ndarray
            imaginary-time object with dimensions (nts, ...)
        :param stats: str
            statistics: 'f' for fermions and 'b' for bosons

        :return: numpy.ndarray
            Matsubara-frequency object with dimensions (nw, ...)
        """
        if stats not in self.statisics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))
        Twt = self.Twt_ff if stats == 'f' else self.Twt_bb
        if Ot.shape[0] != Twt.shape[1]:
            raise ValueError(
                "tau_to_w: Number of tau points are inconsistent: {} and {}".format(Ot.shape[0], Twt.shape[1]))

        Ot_shape = Ot.shape
        Ot = Ot.reshape(Ot.shape[0], -1)
        Ow = np.dot(Twt, Ot)

        Ot = Ot.reshape(Ot_shape)
        Ow = Ow.reshape((Twt.shape[0],) + Ot_shape[1:])
        return Ow

    def tau_to_w_phsym(self, Ot, stats: str):
        """
        Fourier transform from imaginary-time axis to Matsubara-frequency axis w/ particle-hole symmetry
        :param Ot: numpy.ndarray
            imaginary-time object with dimensions (nts, ...)
        :param stats: str
            statistics: 'f' for fermions and 'b' for bosons

        :return: numpy.ndarray
            Matsubara-frequency object with dimensions (nw, ...)
        """
        if stats != 'b':
            raise ValueError("FT w/ particle-hole symmetry only support bosonic correlation functions")

        nw_half = self.nw_b // 2 if self.nw_b % 2 == 0 else self.nw_b // 2 + 1
        nt_half = self.nt_b // 2 if self.nt_b % 2 == 0 else self.nt_b // 2 + 1
        if Ot.shape[0] != nt_half:
            raise ValueError(
                "tau_to_w_phsym: Number of tau points are inconsistent: {} and {}".format(Ot.shape[0], nt_half))

        Twt_pos = np.zeros((nw_half, nt_half), dtype=self.Twt_bb.dtype)
        for n in range(nw_half):
            iw = self.nw_b // 2 + n
            for it in range(nt_half):
                imt = self.nt_b - it - 1
                Twt_pos[n, it] = self.Twt_bb[iw, it] if it == imt else self.Twt_bb[iw, it] + self.Twt_bb[iw, imt]

        Ot_shape = Ot.shape
        Ot = Ot.reshape(Ot.shape[0], -1)
        Ow = np.dot(Twt_pos, Ot)

        Ot = Ot.reshape(Ot_shape)
        Ow = Ow.reshape((Twt_pos.shape[0],) + Ot_shape[1:])
        return Ow

    def w_to_tau(self, Ow, stats):
        """
        Fourier transform from Matsubara-frequency axis to imaginary-time axis.

        :param Ow: numpy.ndarray
            Matsubara-frequency object with dimensions (nw, ...)
        :param stats: str
            statistics, 'f' for fermions and 'b' for bosons

        :return: numpy.ndarray
            Imaginary-time object with dimensions (nt, ...)
        """
        if stats not in self.statisics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))
        Ttw = self.Ttw_ff if stats == 'f' else self.Ttw_bb
        if Ow.shape[0] != Ttw.shape[1]:
            raise ValueError(
                "w_to_tau: Number of w points are inconsistent: {} and {}".format(Ow.shape[0], Ttw.shape[1]))

        Ow_shape = Ow.shape
        Ow = Ow.reshape(Ow.shape[0], -1)
        Ot = np.dot(Ttw, Ow)

        Ow = Ow.reshape(Ow_shape)
        Ot = Ot.reshape((Ttw.shape[0],) + Ow_shape[1:])
        return Ot

    def w_to_tau_phsym(self, Ow, stats):
        """
        Fourier transform from Matsubara-frequency axis to imaginary-time axis w/ particle-hole symmetry.

        :param Ow: numpy.ndarray
            Matsubara-frequency object with dimensions (nw, ...)
        :param stats: str
            statistics, 'f' for fermions and 'b' for bosons

        :return: numpy.ndarray
            Imaginary-time object with dimensions (nt, ...)
        """
        if stats != 'b':
            raise ValueError("FT w/ particle-hole symmetry only support bosonic correlation functions")

        nw_half = self.nw_b // 2 if self.nw_b % 2 == 0 else self.nw_b // 2 + 1
        nt_half = self.nt_b // 2 if self.nt_b % 2 == 0 else self.nt_b // 2 + 1
        if Ow.shape[0] != nw_half:
            raise ValueError(
                "w_to_tau_phsym: Number of w points are inconsistent: {} and {}".format(Ow.shape[0], nw_half))

        Ttw_pos = np.zeros((nt_half, nw_half), dtype=self.Ttw_bb.dtype)
        for it in range(nt_half):
            for n in range(nw_half):
                iw = self.nw_b // 2 + n
                imw = self.nw_b // 2 - n
                Ttw_pos[it, n] = self.Ttw_bb[it, iw] if iw == imw else self.Ttw_bb[it, iw] + self.Ttw_bb[it, imw]

        Ow_shape = Ow.shape
        Ow = Ow.reshape(Ow.shape[0], -1)
        Ot = np.dot(Ttw_pos, Ow)

        Ow = Ow.reshape(Ow_shape)
        Ot = Ot.reshape((Ttw_pos.shape[0],) + Ow_shape[1:])
        return Ot


    def w_interpolate(self, Ow, target, stats: str, ir_notation: bool = True):
        """
        Interpolate a dynamic object to arbitrary points on the Matsubara axis.

        :param Ow: numpy.ndarray
            Dynamic object on the Matsubara sampling points, self.wn_mesh.
        :param target: IAFT or numpy.ndarray(dim=1, dtype=int)
        :param stats: str
            Statistics, 'f' for fermions and 'b' for bosons.
        :param ir_notation: bool
            Whether wn_mesh_interp is in sparse_ir notation where iwn = n*pi/beta for both fermions and bosons.
            Otherwise, iwn = (2n+1)*pi/beta  for fermions and 2n*pi/beta for bosons.

        :return: numpy.ndarray
            Matsubara-frequency object with dimensions (nw_interp, ...)
        """
        if isinstance(target, IAFT):
            return self._w_interpolate(Ow, target.wn_mesh(stats, ir_notation), stats, ir_notation)
        else:
            return self._w_interpolate(Ow, target, stats, ir_notation)


    def _w_interpolate(self, Ow, wn_mesh_interp, stats: str, ir_notation: bool = True):
        """
        Interpolate a dynamic object to arbitrary points on the Matsubara axis.

        :param Ow: numpy.ndarray
            Dynamic object on the Matsubara sampling points, self.wn_mesh.
        :param wn_mesh_interp: numpy.ndarray(dim=1, dtype=int)
            Target frequencies "INDICES".
            The physical Matsubara frequencies are wn_mesh_interp * pi/beta.
        :param stats: str
            Statistics, 'f' for fermions and 'b' for bosons.
        :param ir_notation: bool
            Whether wn_mesh_interp is in sparse_ir notation where iwn = n*pi/beta for both fermions and bosons.
            Otherwise, iwn = (2n+1)*pi/beta  for fermions and 2n*pi/beta for bosons.

        :return: numpy.ndarray
            Matsubara-frequency object with dimensions (nw_interp, ...)
        """
        if stats not in self.statisics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))
        if ir_notation:
            wn_indices = np.asarray(wn_mesh_interp)
        else:
            wn_indices = np.array([2*n+1 if stats == 'f' else 2*n for n in wn_mesh_interp], dtype=int)
        Tlw = self.Tlw_ff if stats == 'f' else self.Tlw_bb
        if Ow.shape[0] != Tlw.shape[1]:
            raise ValueError(
                "w_interpolate: Number of w points are inconsistent: {} and {}".format(Ow.shape[0], Tlw.shape[1]))

        Twl_interp = self.bases.basis_f.uhat(wn_indices).T if stats == 'f' else self.bases.basis_b.uhat(wn_indices).T
        Tww = np.dot(Twl_interp, Tlw)

        Ow_shape = Ow.shape
        Ow = Ow.reshape(Ow.shape[0], -1)
        Ow_interp = np.dot(Tww, Ow)

        Ow = Ow.reshape(Ow_shape)
        Ow_interp = Ow_interp.reshape((wn_indices.shape[0],) + Ow_shape[1:])
        return Ow_interp


    def w_interpolate_phsym(self, Ow, target, stats: str, ir_notation: bool = True):
        """
        Interpolate a dynamic object to arbitrary points on the Matsubara axis.

        :param Ow: numpy.ndarray
            Dynamic object on the Matsubara sampling points, self.wn_mesh.
        :param target: IAFT or numpy.ndarray(dim=1, dtype=int)
            Target frequencies "INDICES".
            The physical Matsubara frequencies are wn_mesh_interp * pi/beta.
        :param stats: str
            Statistics, 'f' for fermions and 'b' for bosons.
        :param ir_notation: bool
            Whether wn_mesh_interp is in sparse_ir notation where iwn = n*pi/beta for both fermions and bosons.
            Otherwise, iwn = (2n+1)*pi/beta  for fermions and 2n*pi/beta for bosons.

        :return: numpy.ndarray
            Matsubara-frequency object with dimensions (nw_interp, ...)
        """
        if isinstance(target, IAFT):
            nw_half_offset = target.nw_b // 2
            iw_mesh = target.wn_mesh(stats, ir_notation)[nw_half_offset:]
            return self._w_interpolate_phsym(Ow, iw_mesh, stats, ir_notation)
        else:
            return self._w_interpolate_phsym(Ow, target, stats, ir_notation)

    def _w_interpolate_phsym(self, Ow, wn_mesh_interp, stats: str, ir_notation: bool = True):
        """
        Interpolate a dynamic object to arbitrary points on the Matsubara axis.

        :param Ow: numpy.ndarray
            Dynamic object on the Matsubara sampling points, self.wn_mesh.
        :param wn_mesh_interp: numpy.ndarray(dim=1, dtype=int)
            Target frequencies "INDICES".
            The physical Matsubara frequencies are wn_mesh_interp * pi/beta.
        :param stats: str
            Statistics, 'f' for fermions and 'b' for bosons.
        :param ir_notation: bool
            Whether wn_mesh_interp is in sparse_ir notation where iwn = n*pi/beta for both fermions and bosons.
            Otherwise, iwn = (2n+1)*pi/beta  for fermions and 2n*pi/beta for bosons.

        :return: numpy.ndarray
            Matsubara-frequency object with dimensions (nw_interp, ...)
        """
        if stats != 'b':
            raise ValueError("FT w/ particle-hole symmetry only support bosonic correlation functions")

        nw_half = self.nw_b // 2 if self.nw_b % 2 == 0 else self.nw_b // 2 + 1
        nt_half = self.nt_b // 2 if self.nt_b % 2 == 0 else self.nt_b // 2 + 1
        if Ow.shape[0] != nw_half:
            raise ValueError(
                "w_interpolate_phsym: Number of w points are inconsistent: {} and {}".format(Ow.shape[0], nw_half))

        if ir_notation:
            wn_indices = np.asarray(wn_mesh_interp)
        else:
            wn_indices = np.array([2*n for n in wn_mesh_interp], dtype=int)
        Tlw = self.Tlw_bb
        Tlw_pos = np.zeros((Tlw.shape[0], nw_half), dtype=Tlw.dtype)
        for l in range(Tlw.shape[0]):
            for n in range(nw_half):
                iw = self.nw_b // 2 + n
                imw = self.nw_b // 2 - n
                Tlw_pos[l, n] = Tlw[l, iw] if iw == imw else Tlw[l, iw] + Tlw[l, imw]

        Twl_interp = self.bases.basis_b.uhat(wn_indices).T
        Tww = np.dot(Twl_interp, Tlw_pos)

        Ow_shape = Ow.shape
        Ow = Ow.reshape(Ow.shape[0], -1)
        Ow_interp = np.dot(Tww, Ow)

        Ow = Ow.reshape(Ow_shape)
        Ow_interp = Ow_interp.reshape((wn_indices.shape[0],) + Ow_shape[1:])
        return Ow_interp


    def _tau_interpolate(self, Ot, tau_mesh_interp, stats: str):
        """
         Interpolate a dynamic object to arbitrary points on the imaginary-time axis.

        :param Ot: numpy.ndarray
            Dynamic object on the imaginary-time sampling points, self.tau_mesh.
        :param tau_mesh_interp: numpy.ndarray(dim=1, dtype=float)
            Target tau points.
        :param stats: str
            Statistics, 'f' for fermions and 'b' for bosons

        :return: numpy.ndarray
            Imaginary-time object with dimensions (nt_interp, ...)
        """
        if stats not in self.statisics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))
        Tlt = self.Tlt_ff if stats == 'f' else self.Tlt_bb
        if Ot.shape[0] != Tlt.shape[1]:
            raise ValueError(
                "t_interpolate: Number of tau points are inconsistent: {} and {}".format(Ot.shape[0], Tlt.shape[1]))

        Ttl_interp = self.bases.basis_f.u(tau_mesh_interp).T if stats == 'f' else self.bases.basis_b.u(tau_mesh_interp).T
        Ttt = np.dot(Ttl_interp, Tlt)

        Ot_shape = Ot.shape
        Ot = Ot.reshape(Ot.shape[0], -1)
        Ot_interp = np.dot(Ttt, Ot)

        Ot = Ot.reshape(Ot_shape)
        Ot_interp = Ot_interp.reshape((np.shape(tau_mesh_interp)[0],) + Ot_shape[1:])
        return Ot_interp


    def tau_interpolate(self, Ot, target, stats: str):
        """
         Interpolate a dynamic object to arbitrary points on the imaginary-time axis.

        :param Ot: numpy.ndarray
            Dynamic object on the imaginary-time sampling points, self.tau_mesh.
        :param tau_mesh_interp: numpy.ndarray(dim=1, dtype=float)
            Target tau points.
        :param stats: str
            Statistics, 'f' for fermions and 'b' for bosons

        :return: numpy.ndarray
            Imaginary-time object with dimensions (nt_interp, ...)
        """
        if isinstance(target, IAFT):
            return self._tau_interpolate(Ot, target.tau_mesh_f if stats=='f' else target.tau_mesh_b, stats)
        else:
            return self._tau_interpolate(Ot, target, stats)


    def _tau_interpolate_phsym(self, Ot, tau_mesh_interp, stats: str):
        """
         Interpolate a dynamic object to arbitrary points on the imaginary-time axis.

        :param Ot: numpy.ndarray
            Dynamic object on the imaginary-time sampling points, self.tau_mesh.
        :param tau_mesh_interp: numpy.ndarray(dim=1, dtype=float)
            Target tau points.
        :param stats: str
            Statistics, 'f' for fermions and 'b' for bosons

        :return: numpy.ndarray
            Imaginary-time object with dimensions (nt_interp, ...)
        """
        if stats != 'b':
            raise ValueError("FT w/ particle-hole symmetry only support bosonic correlation functions")

        nw_half = self.nw_b // 2 if self.nw_b % 2 == 0 else self.nw_b // 2 + 1
        nt_half = self.nt_b // 2 if self.nt_b % 2 == 0 else self.nt_b // 2 + 1
        if Ot.shape[0] != nt_half:
            raise ValueError(
                "tau_interpolate_phsym: Number of tau points are inconsistent: {} and {}".format(Ot.shape[0], nw_half))

        Tlt = self.Tlt_ff if stats == 'f' else self.Tlt_bb
        Tlt_pos = np.zeros((Tlt.shape[0], nt_half), dtype=Tlt.dtype)
        for l in range(Tlt.shape[0]):
            for it in range(nt_half):
                imt = self.nt_b - it - 1
                Tlt_pos[l, it] = Tlt[l, it] if it == imt else Tlt[l, it] + Tlt[l, imt]

        Ttl_interp = self.bases.basis_b.u(tau_mesh_interp).T
        Ttt = np.dot(Ttl_interp, Tlt_pos)

        Ot_shape = Ot.shape
        Ot = Ot.reshape(Ot.shape[0], -1)
        Ot_interp = np.dot(Ttt, Ot)

        Ot = Ot.reshape(Ot_shape)
        Ot_interp = Ot_interp.reshape((np.shape(tau_mesh_interp)[0],) + Ot_shape[1:])
        return Ot_interp


    def tau_interpolate_phsym(self, Ot, target, stats: str):
        """
         Interpolate a dynamic object to arbitrary points on the imaginary-time axis.

        :param Ot: numpy.ndarray
            Dynamic object on the imaginary-time sampling points, self.tau_mesh.
        :param tau_mesh_interp: numpy.ndarray(dim=1, dtype=float)
            Target tau points.
        :param stats: str
            Statistics, 'f' for fermions and 'b' for bosons

        :return: numpy.ndarray
            Imaginary-time object with dimensions (nt_interp, ...)
        """
        if isinstance(target, IAFT):
            nt_half_target = target.nt_b // 2 if target.nt_b % 2 == 0 else target.nt_b // 2 + 1
            return self._tau_interpolate_phsym(Ot, target.tau_mesh_b[:nt_half_target], stats)
        else:
            return self._tau_interpolate_phsym(Ot, target, stats)


    def check_leakage(self, Ot, stats: str, name: str = "", w_input: bool = False):
        """
        Check decay of the IR coefficients to assess the quality of IR basis for the beta and lambda.
        The coefficients should decay exponentially, and the leakage is defined as:
            leakage = the smallest coefficients / the largest coefficients
        :param Ot:
        :param stats:
        :param name:
        :param w_input:
        :return:
        """
        if w_input:
            Ot_ = self.w_to_tau(Ot, stats)
            self.check_leakage(Ot_, stats, name, w_input=False)
            return

        if stats not in self.statisics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))
        nts = self.nt_f if stats == 'f' else self.nt_b
        Tlt = self.Tlt_ff if stats == 'f' else self.Tlt_bb
        if nts != Ot.shape[0]:
            raise ValueError("Inconsistency between nts = {} and Ot.shape[0] = {}".format(nts, Ot.shape[0]))

        # coeff_first
        O_l0_i = np.einsum('t,ti->i', Tlt[0], Ot.reshape(nts, -1))
        coeff_first = np.max(np.abs(O_l0_i))

        # coeff_last
        O_lm2_t = np.einsum('lt,ti->li', Tlt[-2:], Ot.reshape(nts, -1))
        coeff_last = np.max(np.abs(O_lm2_t))

        leakage = coeff_last/coeff_first
        print("IAFT leakage of {}: {}".format(name, leakage))
        if leakage >= 1e-5:
            print("[WARNING] check_leakage: coeff_last/coeff_first = {} >= 1e-5; "
                  "coeff_last = {}, coeff_first = {}".format(leakage, coeff_last, coeff_first))
        sys.stdout.flush()

    def check_leakage_phsym(self, Ot, stats: str, name: str = "", w_input: bool = False):
        """
        Check decay of the IR coefficients to assess the quality of IR basis for the beta and lambda.
        The coefficients should decay exponentially, and the leakage is defined as:
            leakage = the smallest coefficients / the largest coefficients
        :param Ot:
        :param stats:
        :param name:
        :param w_input:
        :return:
        """
        if stats != 'b':
            raise ValueError("FT w/ particle-hole symmetry only support bosonic correlation functions")

        if w_input:
            Ot_ = self.w_to_tau_phsym(Ot, stats)
            self.check_leakage_phsym(Ot_, stats, name, w_input=False)
            return

        if stats not in self.statisics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))

        nts = self.nt_b
        nt_half = self.nt_b // 2 if self.nt_b % 2 == 0 else self.nt_b // 2 + 1
        Tlt = self.Tlt_bb
        if nt_half != Ot.shape[0]:
            raise ValueError("Inconsistency between nts_half = {} and Ot.shape[0] = {}".format(nt_half, Ot.shape[0]))

        # coeff_first
        Tl0_t_pos = np.zeros(nt_half, dtype=complex)
        for it in range(nt_half):
            imt = nts-it-1
            Tl0_t_pos[it] = Tlt[0,it] if it == imt else Tlt[0,it] + Tlt[0,imt]
        O_l0_i = np.einsum('t,ti->i', Tl0_t_pos, Ot.reshape(nt_half, -1))
        coeff_first = np.max(np.abs(O_l0_i))

        # coeff_last
        Tlm2_t_pos = np.zeros((2,nt_half), dtype=complex)
        nl = Tlt.shape[0]
        for it in range(nt_half):
            imt = nts-it-1
            Tlm2_t_pos[0,it] = Tlt[nl-2,it] if it==imt else Tlt[nl-2,it] + Tlt[nl-2,imt]
            Tlm2_t_pos[1,it] = Tlt[nl-1,it] if it==imt else Tlt[nl-1,it] + Tlt[nl-1,imt]
        O_lm2_t = np.einsum('lt,ti->li', Tlm2_t_pos, Ot.reshape(nt_half,-1))
        coeff_last = np.max(np.abs(O_lm2_t))

        leakage = coeff_last/coeff_first
        print("IAFT leakage of {}: {}".format(name, leakage))
        if leakage >= 1e-5:
            print("[WARNING] check_leakage_phsym: coeff_last/coeff_first = {} >= 1e-5; "
                  "coeff_last = {}, coeff_first = {}".format(leakage, coeff_last, coeff_first))
        sys.stdout.flush()


if __name__ == '__main__':
    # Initialize IAFT object for given inverse temperature, lambda and precision
    ft = IAFT(1000, 1e4, 1e-6)

    print(ft.wn_mesh('f', True))

    Gt = np.zeros((ft.nt_f, 2, 2, 2))
    Gw = ft.tau_to_w(Gt, 'f')
    print(Gw.shape)

    # Interpolate to arbitrary tau point
    tau_interp = np.array([0.0, ft.beta])
    Gt_interp = ft.tau_interpolate(Gt, tau_interp, 'f')
    print(Gt_interp.shape)

    # wn in spare_ir notation
    w_interp = np.array([-1,1,3,5], dtype=int)
    Gw_interp = ft.w_interpolate(Gw, w_interp, 'f', True)
    print(Gw_interp.shape)

    # wn in physical notation
    w_interp = np.array([-1,0,1,2,3,4], dtype=int)
    Gw_interp = ft.w_interpolate(Gw, w_interp, 'f', False)
    print(Gw_interp.shape)

    Gt2 = ft.w_to_tau(Gw, 'f')
    print(Gt2.shape)
