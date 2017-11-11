import numpy as np

'''
HMM中常见的一种算法，Forward Algorithm, 前向算法，也叫Filter Method, 用于求解 P(Zt | X_1:t), 是一种
online的方法，根据过去从time 1 到 最新的time t的所有观测值X_1:t，来计算当前最新的一个隐藏状态Zt的概率分布
假设总共有K个状态，则结果就是一个长度为K的向量

Notations:
αt ∝ ψt * (Ψ.T * αt − 1)
ψt(j) = p(xt|zt = j)
Ψ(i, j) = p(zt = j|z(t−1) = i), that is the transition matrix

refer to Algorithm 17.1: Page. 610

本例中 p(xt|zt = j) 的概率模型就采取casinoDemo中的obsModel这个muti-bernoulli的模型
'''

def normalize(a):
    return a / np.sum(a)

def hmm_filter(X, A, obsModel, pi):
    '''
    :param X: observations
    :param A: transition matrix
    :param obsModel: P(x | z = j)
    :param pi: initial distribution of states
    :return: Z, matrix with shape (N, K), that is P(Z = j | X), probs of using fair and loaded dices
    '''
    N = len(X)
    assert N > 0
    pi = pi.reshape(-1, 1)
    Z = np.zeros((N, 2))
    phi1 = np.array([obsModel[0, X[0] - 1], obsModel[1, X[0] - 1]])  # local evidence P(x1 | z = j)
    phi1 = phi1.reshape(-1, 1)
    alpha1 = normalize(phi1 * pi)
    Z[0] = alpha1.ravel()
    if N == 1:
        return Z
    else:
        for i in range(1, N):
            xi = X[i]
            phi = np.array([obsModel[0, xi - 1], obsModel[1, xi - 1]]).reshape(-1, 1)
            alpha_i = normalize(phi * np.dot(A.T, Z[i-1].reshape(-1, 1)))
            Z[i] = alpha_i.ravel()

    return Z
