import numpy as np
import scipy.linalg as sl
import sklearn.metrics.pairwise as smp

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
    log_likelihood = -0.5 * np.dot(y.T, alpha) - np.sum(np.log(np.diag(L))) - (N / 2) * np.log(2 * np.pi)

    if type(Ks) != type(np.array([1])):
        return log_likelihood  # Log likelihood don't depend on Ks & Kss

    f_mu = np.dot(Ks.T, alpha)   # N2 * 1
    v = np.dot(L_inv, Ks)        # N * N2
    f_var = Kss - np.dot(v.T, v) # N2 * N2
    f_var[f_var < 0] = 0

    return f_mu, f_var, log_likelihood

# squared-exponential (SE) kernel for the noisy observations
def SE(X, Y, v_scale, h_scale, noise_sigma):
    N, D = X.shape
    gamma = 1 / (2 * h_scale**2)
    gram = (v_scale**2) * smp.rbf_kernel(X, Y, gamma=gamma)
    if len(X) == len(Y):
        gram += (noise_sigma**2) * np.eye(N)

    return gram