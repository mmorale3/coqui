import os
import numpy as np
import h5py

'''
For debugging purposes, here are a collection of functions for the python version of 
imag_axes_ft/IAFT class based on the IR grids 
'''

def tau_to_w(Ot, beta, lamb, prec='high', stats='f', debug=False):
    '''
    Fourier transform from imaginary-time axis to Matsubara-frequency axis
    :param Ot: imaginary-time object with dimensions (nts, ...)
    :param beta: inverse temperature
    :param lamb: dimensionless parameter lambda
    :param stats: statistics. 'f' for fermions; 'b' for bosons
    :return: Matsubara-frequency object with dimensions (nw, ...)
    '''
    home_dir = os.path.dirname(os.path.abspath(__file__))
    if prec == "high":
        filename = home_dir + "/" + lamb + ".1e-15.h5"
    elif prec == "medium":
        filename = home_dir + "/" + lamb + ".1e-10.h5"
    else:
        filename = home_dir + "/" + lamb + ".1e-6.h5"
    stats_type = "fermion" if stats == 'f' else "boson"

    if debug:
        print("Read the {} IR grids from {}".format(stats_type, filename))

    f = h5py.File(filename, 'r')
    Twt = f[stats_type+"/Twt"][()].view(complex)[...,0]
    Twt *= (beta/np.sqrt(2))
    f.close()

    if Ot.shape[0] != Twt.shape[1]:
        raise ValueError("number of tau points are inconsistent: {} and {}".format(Ot.shape[0], Twt.shape[1]))

    Ot_shape = Ot.shape
    Ot = Ot.reshape(Ot.shape[0], -1)
    Ow = np.dot(Twt, Ot)

    Ot = Ot.reshape(Ot_shape)
    Ow = Ow.reshape((Twt.shape[0],)+Ot_shape[1:])
    return Ow

def w_to_tau(Ow, beta, lamb, prec='high', stats='f', debug=False):
    '''
    Fourier transform from Matsubara-frequency axis to imaginary-time axis
    :param Ow: Matsubara-frequency object with dimensions (nw, ...)
    :param beta: inverse temperature
    :param lamb: dimensionless parameter lambda
    :param stats: statistics. 'f' for fermions; 'b' for bosons
    :return: Imaginary-time object with dimensions (nts, ...)
    '''
    home_dir = os.path.dirname(os.path.abspath(__file__))
    if prec == "high":
        filename = home_dir + "/" + lamb + ".1e-15.h5"
    elif prec == "medium":
        filename = home_dir + "/" + lamb + ".1e-10.h5"
    else:
        filename = home_dir + "/" + lamb + ".1e-6.h5"
    stats_type = "fermion" if stats == 'f' else "boson"

    if debug:
        print("Read the {} IR grids from {}".format(stats_type, filename))

    f = h5py.File(filename, 'r')
    Ttw = f[stats_type+"/Ttw"][()].view(complex)[...,0]
    Ttw *= (np.sqrt(2)/beta)
    f.close()

    if Ow.shape[0] != Ttw.shape[1]:
        raise ValueError("number of w points are inconsistent: {} and {}".format(Ow.shape[0], Ttw.shape[1]))

    Ow_shape = Ow.shape
    Ow = Ow.reshape(Ow.shape[0], -1)
    Ot = np.dot(Ttw, Ow)

    Ow = Ow.reshape(Ow_shape)
    Ot = Ot.reshape((Ttw.shape[0],)+Ow_shape[1:])
    return Ot

def check_leakage(Ot, dataset, beta, lamb, prec="high", stats='f', debug=False):
    home_dir = os.path.dirname(os.path.abspath(__file__))
    if prec == "high":
        filename = home_dir + "/" + lamb + ".1e-15.h5"
    elif prec == "medium":
        filename = home_dir + "/" + lamb + ".1e-10.h5"
    else:
        filename = home_dir + "/" + lamb + ".1e-6.h5"
    stats_type = "fermion" if stats == 'f' else "boson"

    if debug:
        print("Read the {} IR grids from {}".format(stats_type, filename))

    f = h5py.File(filename, 'r')
    Tct = f[stats_type+"/Tct"][()].view(complex)[...,0]
    Tct *= np.sqrt(2.0/beta)
    f.close()

    Ot_shape = Ot.shape
    Ot = Ot.reshape(Ot.shape[0], -1)

    Oc0 = np.dot(Ot.T, Tct[0])
    c_first = np.max(np.abs(Oc0))

    Ocm2 = np.dot(Tct[-2:], Ot)
    c_last = np.max(np.abs(Ocm2))

    leakage = c_last/c_first
    print("IAFT leakage of {}: {}".format(dataset, leakage))

class iaft(object):
    def __init__(self, beta, lamb, prec):
        self.beta = beta
        self.lamb = lamb
        self.prec = prec
        home_dir = os.path.dirname(os.path.abspath(__file__))
        if prec == "high":
            self.filename = home_dir + "/" + lamb + ".1e-15.h5"
        elif prec == "medium":
            self.filename = home_dir + "/" + lamb + ".1e-10.h5"
        else:
            self.filename = home_dir + "/" + lamb + ".1e-6.h5"

        f = h5py.File(self.filename, 'r')
        self.nts = f["fermion/nt"][()]
        self.nw  = f["fermion/nw"][()]
        self.tau_mesh = f["fermion/tau_mesh"][()]
        self.wsample = f["fermion/wn_mesh"][()]
        f.close()

        self.tau_mesh = (self.tau_mesh + 1) * self.beta/2.0
        self.wsample = self.wsample.astype(float)
        for n in range(self.wsample.shape[0]):
            self.wsample[n] *= np.pi / self.beta

        print(self)

    def __str__(self):
        return "*******************************\n" \
               "Imaginary-Axis Fourier Transform:\n" \
               "*******************************\n" \
               "    - beta = {}\n" \
               "    - lamb = {}\n" \
               "    - nt, nw = {}, {}\n" \
               "    - file = {}\n" \
               "*******************************".format(self.beta, self.lamb, self.nts, self.nw, self.filename)

    def tau_to_w(self, Ot):
        return tau_to_w(Ot, self.beta, self.lamb, self.prec, stats='f', debug=False)
    def w_to_tau(self, Ow):
        return w_to_tau(Ow, self.beta, self.lamb, self.prec, stats='f', debug=False)


if __name__ == '__main__':
    beta = 1000
    lamb = "1e4"

    Gt = np.zeros((108, 2, 2, 2))
    Gw = tau_to_w(Gt, beta, lamb, "f")
    print(Gw.shape)

    Gt2 = w_to_tau(Gw, beta, lamb, "f")
    print(Gt2.shape)
