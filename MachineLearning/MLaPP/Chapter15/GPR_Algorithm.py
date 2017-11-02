import numpy as np
import scipy.linalg as sl

'''
numerical stable method to calculate GPR using Cholesky Decomposition
'''

def fit_gpr(K, Ks, Kss, y):
    y = y.reshape(-1, 1)
    N = len(K)
    Ky = K
    L = sl.cholesky(Ky, lower=True)
    L_inv = sl.inv(L)  # inverse of lower triangular matrix, numeric stable
    alpha = np.dot(L_inv.T, np.dot(L_inv, y))  # N * 1
    f_mu = np.dot(Ks.T, alpha)   # N2 * 1
    v = np.dot(L_inv, Ks)        # N * N2
    f_var = Kss - np.dot(v.T, v) # N2 * N2
    f_var[f_var < 0] = 0
    log_likelihood = -0.5 * np.dot(y.T, alpha) - np.sum(np.log(np.diag(L))) - (N / 2) * np.log(2 * np.pi)

    return f_mu, f_var, log_likelihood
