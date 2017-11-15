import numpy as np
import scipy.linalg as sl
from kalmanFilter import *

def kalman_smoothing(y, A, C, Q, R, init_mu, init_V):
    init_mu = init_mu.reshape(-1, 1)
    K = len(init_mu)  # dimension of hidder state
    if y.ndim == 1:
        N = 1
    else:
        N, D = y.shape
    smooth_mus = np.zeros((N, K))
    smooth_covs = np.zeros((N, K, K))
    # forwards algorithm
    filter_mus, filter_covs = kalman_filter(y, A, C, Q, R, init_mu, init_V)

    smooth_mus[-1] = filter_mus[-1]
    smooth_covs[-1] = filter_covs[-1]
    # tracing backwards
    for i in range(N - 2, -1, -1):
        msmooth_future = smooth_mus[i + 1].reshape(-1, 1)
        Vsmooth_future = smooth_covs[i + 1]
        mfilt = filter_mus[i].reshape(-1, 1)          # μt
        Vfilt = filter_covs[i]                        # Σt
        mfilt_pre = np.dot(A, mfilt)                  # μ(t+1|t)
        Vfilt_pre = np.dot(A, np.dot(Vfilt, A.T)) + Q # Σ(t+1|t)

        J = np.dot(Vfilt, np.dot(A.T, sl.inv(Vfilt_pre)))
        smooth_mu = mfilt + np.dot(J, (msmooth_future - mfilt_pre))
        smooth_cov = Vfilt + np.dot(J, np.dot((Vsmooth_future - Vfilt_pre), J.T))

        smooth_mus[i] = smooth_mu.ravel()
        smooth_covs[i] = smooth_cov

    return smooth_mus, smooth_covs