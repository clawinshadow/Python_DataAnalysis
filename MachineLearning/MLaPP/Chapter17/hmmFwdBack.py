import numpy as np

'''
HMM中常见的另一种算法，Forwards-Backwards Algorithm, 前向-后向算法，也叫Smoothing Method, 
用于求解 P(Zt | X_1:T), 这个与filter method不同的地方在于filter method是online的，只能使用现在和过去
的数据来预测，但是smoothing method可以适用未来的数据，所以除了Forward之外我们还要使用Backward来将未来的
数据包括进去，所以他是offline的

Notations:
γt(j) ∝ αt(j) * βt(j)

βt(j) means p(x(t+1):T |zt = j)
要注意的是与αt不一样，βt这个向量的和不需要等于1，它不是一个针对state的概率分布
αt 与 hmmFilter 里面的一样
β(t−1) = np.dot(Ψ, (ψt * βt))
最右边的βT(i) = 1， 所以 βT = np.ones(K)
'''

def normalize(a):
    return a / np.sum(a)

def hmm_smoothing(X, A, obsModel, pi):
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
    alpha = np.zeros((N, 2))
    beta = np.zeros((N, 2))

    # initial alpha[0]
    phi1 = np.array([obsModel[0, X[0] - 1], obsModel[1, X[0] - 1]])  # local evidence P(x1 | z = j)
    phi1 = phi1.reshape(-1, 1)
    alpha1 = normalize(phi1 * pi)
    alpha[0] = alpha1.ravel()

    # initial beta[0]
    K = len(A)
    beta[-1] = np.ones(K)

    # calc all alphas
    for i in range(1, N):
        pre_alpha = alpha[i - 1].reshape(-1, 1)
        xi = X[i]
        phi = np.array([obsModel[0, xi - 1], obsModel[1, xi - 1]]).reshape(-1, 1)
        alpha_i = normalize(phi * np.dot(A.T, pre_alpha))
        alpha[i] = alpha_i.ravel()

    # calc all betas
    for j in range(N - 2, -1, -1):
        pre_beta = beta[j + 1].reshape(-1, 1)
        xj = X[j + 1]   # pay attention, local evidence use j + 1 not j
        phi = np.array([obsModel[0, xj - 1], obsModel[1, xj - 1]]).reshape(-1, 1)
        # beta_j = np.dot(A, phi * pre_beta)          # don't need to normalize beta
        beta_j = normalize(np.dot(A, phi * pre_beta)) # but matlab codes still normalize...
        beta[j] = beta_j.ravel()

    Z = alpha * beta
    rowsum = np.sum(Z, axis=1).reshape(-1, 1)  # normalize Z

    return Z / rowsum