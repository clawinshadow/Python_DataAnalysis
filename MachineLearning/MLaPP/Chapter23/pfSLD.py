import numpy as np
import scipy.stats as ss
import scipy.linalg as sl

'''
Particle Filtering for SLDS model, SLDS definition refer to Chapter 18.6

Parameters:
  NSamples: number of samples
  **args: parameters of SLDS, details refer to rbpfManeuverDemo.py
  y: observations, ny * T
  u: signal control, nu * T
  
Returns:
  xest: estimation of x[1:T], the same shape of x, nx * T, x is continuous states
  zest: estimation of z[1:T], the same shape of z, nz * T, z is discrete states, for SLDS only
  zsamples: nz * T * N, samples of z[1:T], 
  xsamples: nx * T * N, samples of x[1:T]
  
本例中 nx = ny = 4, nz = nu = 1
'''

def sample_from_multinomial(distribution):
    res = np.random.multinomial(1, distribution)
    return np.asscalar(np.flatnonzero(res))

def resampling(inIndex, q):
    q = q.reshape(-1, 1)
    S = len(q)

    N_babies = np.zeros(S, dtype='int16')
    u = np.zeros(S)

    cumDist = np.cumsum(q)
    aux = np.random.rand()
    u = np.linspace(aux, S - 1 + aux, S)
    u = u / S
    j = 0
    for i in range(S):
        while u[i] > cumDist[j]:
            j = j + 1
        N_babies[j] += 1

    index = 0
    outIndex = np.zeros(inIndex.shape, dtype='int16')
    for i in range(S):
        if (N_babies[i] > 0):
            for j in range(index, index + N_babies[i]):
                outIndex[j] = (int)(inIndex[i])
        index = index + N_babies[i]

    return outIndex

def pfSLD(NSamples, y, u, *args):
    # parse args
    A, B = args[0], args[1]
    C, D = args[2], args[3]
    E, F = args[4], args[5]
    G, T = args[6], args[7]

    ny, t = y.shape  # ny is dimension of each observation, t is the number of time steps
    ny, nx, nz = C.shape  # nx, ny, nz are all dimensions

    # initialise all the containers
    z_pf = np.zeros((NSamples, t), dtype='int16')   # each row will be a sample of z[1:T], z is the discrete hidden state
    z_pf_pred = np.zeros(z_pf.shape, dtype='int16') # q(zt|z(t-1)), use markov chain transition matrix
    x_pf = 10 * np.random.randn(NSamples, nx, t) # each nx * T matrix is a sample of x[1:T], x is the continuous hidden state
    x_pf_pred = x_pf                 # p(xt|x(t-1)), Ak * x(t-1) + Fk * u(t) + Bk * randn(nx, 1)
    y_pred = 10 * np.random.randn(NSamples, ny, t) # p(yt|y(t-1)), Ck * xt + Gk * u(t), mean
    w = np.ones((NSamples, t))       # each row is the weights of a sample, w[1:T]
    xest = np.zeros((nx, t))         # most important return, is the estimation of x[1:T]
    zest = np.zeros((nz, t))
    initz = np.ones(nz) / nz         # initial multinomial distribution of z

    # initial the first state of z
    for i in range(NSamples):
        z_pf[i, 0] = (int)(sample_from_multinomial(initz))

    # sampling
    for i in range(1, t):
        '''注意外层循环是t，一步一步来的，每一步采满NSamples个'''
        for j in range(NSamples):
            currentTM = T[(int)(z_pf[j, i - 1])]  # current transition distribution of z
            # % sample z(t)~p(z(t)|z(t-1))
            z_next = sample_from_multinomial(currentTM)
            z_pf_pred[j, i] = (int)(z_next)
            # sample x(t)~p(x(t)|z(t|t-1),x(t-1))
            x_pf_pred[j, :, i] = (np.dot(A[:, :, z_next], x_pf[j, :, i - 1].reshape(-1, 1)) + \
                                 np.dot(B[:, :, z_next], np.random.randn(nx, 1)) + \
                                 np.dot(F[:, :, z_next], u[:, i]).reshape(-1, 1)).ravel()

        for j in range(NSamples):
            zt = z_pf_pred[j, i]
            xt = x_pf_pred[j, :, i].reshape(-1, 1)

            y_pred[j, :, i] = (np.dot(C[:, :, zt], xt) + np.dot(G[:, :, zt], u[:, i]).reshape(-1, 1)).ravel()
            cov = np.dot(D[:, :, zt], D[:, :, zt].T)
            yt = y[:, i].ravel()
            w[j, i] = ss.multivariate_normal.pdf(yt, mean=y_pred[j, :, i], cov=cov) + 1e-99  # for numeric stable

        w[:, i] = w[:, i] / np.sum(w[:, i])

        outIndex = resampling(np.arange(0, NSamples, 1), w[:, i])
        z_pf[:, i] = z_pf_pred[outIndex, i]
        x_pf[:, :, i] = x_pf_pred[outIndex, :, i]
        xest[:, i] = np.mean(np.squeeze(x_pf[:, :, i]), axis=0)
        hists, edges = np.histogram(z_pf[:, i], np.arange(0, nz + 1, 1))
        zest[:, i] = hists / np.sum(hists)

    zsamples = z_pf
    xsamples = x_pf

    return xest, zest, xsamples, zsamples, w
