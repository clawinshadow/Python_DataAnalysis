import numpy as np
import scipy.linalg as sl

def kalman_filter(y, A, C, Q, R, init_mu, init_V):
    '''
    ignore control signal B, D, u
    A should be a square matrix K * K, C should be a transform matrix D * K
    Q.shape = A.shape, R.shape: D * D
    '''

    init_mu = init_mu.reshape(-1, 1)
    K = len(init_mu)   # dimension of hidder state
    if y.ndim == 1:
        N = 1
    else:
        N, D = y.shape
    mus = np.zeros((N, K))
    covs = np.zeros((N, K, K))
    for i in range(N):
        if i == 0:
            pre_mu = init_mu
            pre_cov = init_V
            mpred = pre_mu
            vpred = pre_cov
        else:
            pre_mu = mus[i - 1].reshape(-1, 1)      # μ(t-1)
            pre_cov = covs[i - 1]                   # Σ(t-1)
            mpred = np.dot(A, pre_mu)                    # μ(t|t-1)
            vpred = np.dot(A, np.dot(pre_cov, A.T)) + Q  # Σ(t|t-1)

        yi = y[i].reshape(-1, 1)
        ye = np.dot(C, mpred)                       # D * 1
        r = yi - ye                                 # D * 1
        S = np.dot(C, np.dot(vpred, C.T)) + R       # D * D
        KM = np.dot(vpred, np.dot(C.T, sl.inv(S)))  # K * D
        new_mu = mpred + np.dot(KM, r)              # K * 1
        new_cov = np.dot(np.eye(K) - np.dot(KM, C), vpred)  # K * K

        mus[i] = new_mu.ravel()
        covs[i] = new_cov

    return mus, covs