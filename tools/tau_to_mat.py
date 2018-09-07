import numpy as np
import scipy.integrate as sI
import sys

fname = sys.argv[1]
beta = float(sys.argv[2])

gf_vec_tau = np.loadtxt(fname)
gf_vec_tau_exp = np.zeros(gf_vec_tau.shape, dtype=complex)
gf_vec_tau_exp = gf_vec_tau + 0.0j
gf_vec_iwn = np.zeros(gf_vec_tau.shape, dtype=complex)
NMAT = int(100 * beta / (2.0 * np.pi))

for nn in range(NMAT):
    iwn = 1.0j * (2.0 * nn + 1.0) * np.pi / beta
    gf_vec_iwn[nn, 0] = iwn.imag
    for tt in range(gf_vec_tau_exp.shape[0]):
        tau = gf_vec_tau_exp[tt, 0]
        gf_vec_tau_exp[tt:, 1::] = np.exp(iwn * tau) * gf_vec_tau[tt, 1::]

    for ii in range(gf_vec_tau_exp.shape[1] - 1):
        gf_vec_iwn[nn, ii + 1] = \
            sI.simps(gf_vec_tau_exp[:, ii + 1], gf_vec_tau[:, 0])


np.savetxt("giwn_python.dat", np.transpose(
    [gf_vec_iwn[:, 0].real, gf_vec_iwn[:, 3].imag]))
