from numpy import mean
from numpy import std
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot
import numpy as np


def cov_mat_adj(par, cov_mat_orig):

    D = par.shape[1]
    # adjust the ratio of the proposal variances based on the covariance of the parameters

    def_by = 1

    chaincorr = np.corrcoef(par.T)
    chaincorr[np.isnan(chaincorr)] = 0
    new_stepcor_chain = np.zeros((D, D))
    temp_stepvar_chain = np.zeros((D, D))

    for rr in range(D):
        for cc in range(D):
            if rr == cc:
                new_stepcor_chain[rr, cc] = 1
            elif np.abs(chaincorr[rr, cc]) < 0.6:
                new_stepcor_chain[rr, cc] = 0
            else:
                new_stepcor_chain[rr, cc] = np.sign(chaincorr[rr, cc]) *\
                    (np.abs(chaincorr[rr, cc]) - 0.4)


    pvars = np.diag(cov_mat_orig)/def_by

    for rr in range(D):
        for cc in range(D):
            temp_stepvar_chain[rr, cc] = new_stepcor_chain[rr, cc] * np.sqrt(pvars[rr] * pvars[cc])

    tempvar = np.diag(temp_stepvar_chain)

    for dd in range(D):
        if tempvar[dd] <= 0:
            temp_stepvar_chain[dd, dd] = cov_mat_orig[dd, dd]


    cov_mat_new = temp_stepvar_chain

    return cov_mat_new

