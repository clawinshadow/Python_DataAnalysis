import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
from pfSLD import *

'''
Rao-Blackwellised particle filtering (RBPF)

与普通PF不同的地方主要在于使用了Kalman Filtering来更新模型参数，详细算法参考 Page.833
'''

def rbpfSLD(NSamples, y, u, *args):
    # parse args
    A, B = args[0], args[1]
    C, D = args[2], args[3]
    E, F = args[4], args[5]
    G, T = args[6], args[7]

    ny, t = y.shape  # ny is dimension of each observation, t is the number of time steps
    ny, nx, nz = C.shape  # nx, ny, nz are all dimensions

    # initialise all the containers
    z_rbpf = np.zeros((NSamples, t), dtype='int16') # each row will be a sample of z[1:T], z is the discrete hidden state
    z_rbpf_pred = np.zeros(z_rbpf.shape, dtype='int16')
    mu = 0.01 * np.random.randn(NSamples, nx, t)    # kalman mean of x, μx
    mu_pred = 0.01 * np.random.randn(NSamples, nx)
    sigma = np.zeros((NSamples, nx, nx))            # kalman covariance of x, Σx
    sigma_pred = np.zeros((NSamples, nx, nx))
    S = np.zeros((NSamples, ny, ny))                # kalman predictive covariance of y,
    y_pred = 0.01 * np.random.randn(NSamples, ny, t)  # p(yt|y(t-1)), Ck * xt + Gk * u(t), mean
    w = np.ones((NSamples, t))  # each row is the weights of a sample, w[1:T]
    xest = np.zeros((nx, t))    # most important return, is the estimation of x[1:T]
    zest = np.zeros((nz, t))
    initz = np.ones(nz) / nz    # initial multinomial distribution of z

    # initial the first state of z, and kalman params
    for i in range(NSamples):
        sigma[i] = np.eye(nx)
        sigma_pred[i] = sigma[i]
        z_rbpf[i, 0] = (int)(sample_from_multinomial(initz))
        S[i] = np.dot(np.dot(C[:, :, z_rbpf[i, 0]], sigma_pred[i]), C[:, :, z_rbpf[i, 0]].T) + \
               np.dot(D[:, :, z_rbpf[i, 0]], D[:, :, z_rbpf[i, 0]].T)

    # sampling
    for i in range(1, t):
        '''注意外层循环是t，一步一步来的，每一步采满NSamples个'''
        for j in range(NSamples):
            currentTM = T[(int)(z_rbpf[j, i - 1])]  # current transition distribution of z

            # sample z(t)~p(z(t)|z(t-1))
            z_next = sample_from_multinomial(currentTM)
            z_rbpf_pred[j, i] = (int)(z_next)

            # Kalman prediction
            mu_pred[j] = (np.dot(A[:, :, z_next], mu[j, :, i - 1].reshape(-1, 1)) + \
                         np.dot(F[:, :, z_next], u[:, i]).reshape(-1, 1)).ravel()
            sigma_pred[j] = np.dot(np.dot(A[:, :, z_next], sigma[j]), A[:, :, z_next].T) + \
                            np.dot(B[:, :, z_next], B[:, :, z_next].T)
            S[j] = np.dot(np.dot(C[:, :, z_next], sigma_pred[j]), C[:, :, z_next].T) + \
                   np.dot(D[:, :, z_next], D[:, :, z_next].T)
            y_pred[j, :, i] = (np.dot(C[:, :, z_next], mu_pred[j].reshape(-1, 1)) + \
                              np.dot(G[:, :, z_next], u[:, i]).reshape(-1, 1)).ravel()

        # evaluate importance weights
        for j in range(NSamples):
            yt = y[:, i].ravel()
            w[j, i] = ss.multivariate_normal.pdf(yt, mean=y_pred[j, :, i], cov=S[j]) + 1e-99  # for numeric stable

        w[:, i] = w[:, i] / np.sum(w[:, i])

        # selection step
        outIndex = resampling(np.arange(0, NSamples, 1), w[:, i])
        z_rbpf[:, i] = z_rbpf_pred[outIndex, i]
        mu_pred = mu_pred[outIndex]
        sigma_pred = sigma_pred[outIndex]
        S = S[outIndex]
        y_pred[:, :, i] = y_pred[outIndex, :, i]

        # updating step
        for j in range(NSamples):
            # Kalman update
            K = np.dot(sigma_pred[j], np.dot(C[:, :, z_rbpf[j, i]].T, sl.inv(S[j])))  # nx * ny
            mu[j, :, i] = mu_pred[j] + np.dot(K, (y[:, i] - y_pred[j, :, i]).reshape(-1, 1)).ravel() # nx * 1
            sigma[j] = sigma_pred[j] - np.dot(K, np.dot(C[:, :, z_rbpf[j ,i]], sigma_pred[j]))

        xest[:, i] = np.mean(np.squeeze(mu[:, :, i]), axis=0)
        hists, edges = np.histogram(z_rbpf[:, i], np.arange(0, nz + 1, 1))
        zest[:, i] = hists / np.sum(hists)

    zsamples = z_rbpf

    return xest, zest, zsamples, w
