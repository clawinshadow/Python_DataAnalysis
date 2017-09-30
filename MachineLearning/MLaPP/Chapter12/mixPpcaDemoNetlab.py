import numpy as np
import scipy.linalg as sl
import scipy.stats as ss
import matplotlib.pyplot as plt

np.random.seed(0)

'''
MFA：mixture of Factor Analysis
     用于拟合高维实数类型的数据，是个很好的模型，比GMM要少了不少参数，一个是KLD，一个是KDD，一般而言D要远大于L
p(xi|zi, qi = k, θ) = N(xi|μk +Wk*zi, Ψ)
p(zi|θ) = N(zi|0, I)
p(qi|θ) = Cat(qi|π)

Ψ 是共享的，只是 μk 和 Wk 不一样
'''

# pi: 1 * K, mu: K * D, W: K * D * L, psi: D * D
def GetR(X, pi, mu, W, psi):
    K = len(pi)
    N, D = X.shape
    if K == 1:
        return np.ones((N, K))
    R = np.zeros((N, K))
    for i in range(K):
        mui = mu[i].ravel()
        sigma = np.dot(W[i].T, W[i]) + psi
        R[:, i] = pi[i] * ss.multivariate_normal(mui, sigma).pdf(X)

    rowsum = np.sum(R, axis=1).reshape(-1, 1)
    R = R / rowsum
    return R

def UpdateTheta(X, rk, mu, W, psi):
    N = len(X)
    D, L = W.shape
    t = np.dot(W.T, sl.inv(psi))
    gap = X - mu  # N * D
    sigma_c = sl.inv(np.eye(L) + np.dot(t, W))  # L * L
    mu_c = np.dot(sigma_c, np.dot(t, gap.T))  # L * N
    mu_c = mu_c.T  # N * L
    b_c = np.c_[mu_c, np.ones((N, L))] # N * 2L
    C_c = np.zeros((N, 2*L, 2*L))    # N * 2L
    for i in range(N):
        mu_ic = mu_c[i].reshape(-1, 1) # L * 1
        mat_1 = np.dot(mu_ic, mu_ic.T) # L * L
        mat_2 = np.tile(mu_ic, L).reshape(L, L) # L * L
        C_ic = np.c_[np.r_[mat_1, mat_2], np.r_[mat_2, np.eye(L)]] # 2L * 2L
        C_c[i] = C_ic

    Wic = np.zeros((D, 2 * L))
    Wic_1 = np.zeros((D, 2 * L))
    Wic_2 = np.zeros((2 * L, 2 * L))
    for i in range(N):
        xi = X[i].reshape(-1, 1)  # D * 1
        Wic_1 += rk[i] * np.dot(xi, b_c[i].reshape(1, -1))  # D * 2L
        Wic_2 += rk[i] * C_c[i]
    Wic = np.dot(Wic_1, sl.inv(Wic_2))

    psi_temp = np.zeros((D, D))
    for i in range(N):
        xi = X[i].reshape(-1, 1)
        gap = xi - np.dot(Wic, b_c[i].reshape(-1, 1))
        psi_temp += rk[i] * np.dot(gap, xi.T)

    psi = np.diag(np.diag(psi_temp)) / N
    W_new = Wic[:, :L]  # D * L
    mu_new = Wic[0, L:] # L * 1

    return W_new, mu_new, psi
    

def MStep(X, R, pi, mu, W, psi):
    K, D, L = W.shape
    pi_new = np.mean(R, axis=0)
    W_new = np.zeros(W.shape)
    mu_new = np.zeros(mu.shape)
    psi_temp = psi
    for k in range(len(pi)):
        rk = R[:, k]
        wk_new, muk_new, psi_new = UpdateTheta(X, rk, mu[k], W[k], psi_temp)
        W_new[k] = wk_new
        mu_new[k] = muk_new
        psi_temp = psi_new

    return pi_new, mu_new, W_new, psi_temp

def FitMFA(X, K=3, L=1, maxIter=100):
    N, D = X.shape
    pi = np.tile(1/K, K)
    mu = np.random.randn(K, 1, D)
    W = np.random.randn(K, D, L)
    psi = np.eye(D)
    for i in range(maxIter):
        R = GetR(X, pi, mu, W, psi)
        pi_new, mu_new, W_new, psi_new = MStep(X, R, pi, mu, W, psi)
        print('Pi_new: ', pi_new)
        print('mu_new: \n', mu_new)
        print('W_new: \n', W_new)
        print('psi_temp: ', psi_new)
        pi, mu = pi_new, mu_new
        W, psi = W_new, psi_new

    return pi, mu, W, psi
    

# Generate Data
N = 500
r = np.random.rand(N) + 1
theta = np.random.rand(N) * 2 * np.pi

x1 = r * np.sin(theta)
x2 = r * np.cos(theta)
X = np.c_[x1.reshape(-1, 1), x2.reshape(-1, 1)]
print('X.shape: ', X.shape)

# Fit MFA
FitMFA(X)

# plots
fig = plt.figure(figsize=(13, 5))
fig.canvas.set_window_title('mixPpcaDemoNetlab')

ax = plt.subplot(121)
plt.axis([-2.5, 2.5, -2, 2])
plt.xticks(np.linspace(-2, 2, 9))
plt.yticks(np.linspace(-1.5, 1.5, 7))
plt.plot(X[:, 0], X[:, 1], 'ro', fillstyle='none', linestyle='none', mew=1)

plt.show()

