import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
import sklearn.cluster as sc

'''
包含MLE和MAP的EM算法，对于GMM的Fit.

因为EM算法只保证收敛到一个局部最小值，所以千万不要忽视初始值和是否有Prior的重要性！
初始值不一样，或者是否给一个prior，得出来的结果往往有很大的差别。
'''

# 判断一个矩阵是否是正定矩阵，最好的数值方法是对它进行楚列斯基分解，如果不报错就是正定的
# 比逐一计算特征值要快得多
def IsPosDef(mat):
    try:
        sl.cholesky(mat)
        return True
    except Exception:
        print('Not a Positive Definite Matrix.')
        return False

def KMeansSigma(data):
    N, D = data.shape[0], data.shape[1]
    mean = np.mean(data, axis=0)
    gap = data - mean
    S = np.dot(gap.T, gap) / (N - 1)
    Sbar = np.dot(gap.T, gap) / N
    Vars = np.zeros((D, D))
    for i in range(N):
        xi = gap[i].reshape(-1, 1)
        Vars += (np.dot(xi, xi.T) - Sbar)**2
    VarS = (N / ((N - 1)**3)) * Vars
    
    I = sl.triu(np.ones((D, D)))
    for i in range(D):
        I[i, i] = 0

    I = I == 1
    Lambda = np.sum(VarS[I]) / np.sum(S[I]**2)
    Lambda = np.min([1, Lambda])
    Lambda = np.max([0, Lambda])
    C = Lambda * np.diag(np.diag(S)) + (1-Lambda)*S

    return C

def KMeansInit(X, K):
    res = sc.KMeans(K).fit(X)
    mu = res.cluster_centers_  # K * D
    labels = res.labels_
    sigmas = np.zeros((K, X.shape[1], X.shape[1]))
    counts = np.zeros(K)
    for i in range(K):
        indices = labels==i
        datai = X[indices]
        counts[i] = len(datai)
        S = KMeansSigma(datai)
        if not IsPosDef(S):
            S = GetSigmaInit(X.shape[1], 0, 1)[0]
        sigmas[i] = S

    pi = counts / np.sum(counts)
    return pi, mu, sigmas

def GetParamInitial(D, K=2, P=2):
    '''P is regularizer'''
    mu = GetMuInit(D, K)
    sigma = GetSigmaInit(D, P, K)
    pi = GetPiInit(P, K)

    return pi, mu, sigma

def GetMuInit(D, K=2):
    return np.random.randn(K, D)

def GetPiInit(P=2, K=2):
    pi = np.random.rand(K) + P
    return pi / np.sum(pi)

def GetSigmaInit(D, P=2, K=2):
    sigma = np.zeros((K, D, D))
    for i in range(K):
        a = np.random.randn(D).reshape(-1, 1)
        randpd = np.dot(a, a.T) + np.diag((P + 0.001) * np.ones(D))
        sigma[i] = randpd
    return sigma

def GetResponsibility(X, pi, mu, cov):
    r = np.zeros((X.shape[0], len(pi)))
    for k in range(len(pi)):
        r[:, k] = pi[k] * ss.multivariate_normal(mu[k], cov[k]).pdf(X)

    total = np.sum(r, axis=1)
    for i in range(r.shape[1]):
        r[:, i] = r[:, i] / total

    return r

def GetCovK(data, muk, rk):
    # old method, slow
    # dim = len(muk.ravel())
    # cov = np.zeros((dim, dim))
    # for i in range(len(data)):
    #     rik = rk.ravel()[i]
    #     gap = data[i].reshape(-1, 1) - muk.reshape(-1, 1)
    #     cov_i = rik * np.dot(gap, gap.T)
    #     cov += cov_i
    #     return cov / np.sum(rk)

    # 用矩阵的方式来进行计算，3d array, fast
    muk, rk = muk.ravel(), rk.ravel()
    gap = data - muk
    X_3d = gap.reshape(gap.shape[0], gap.shape[1], 1)     # N * D * 1
    X_3d_t = gap.reshape(gap.shape[0], 1, gap.shape[1])   # N * 1 * D
    outers = X_3d * X_3d_t                                   # N * D * D，注意这里不能使用np.dot()，否则结果完全不一样
    rk = rk.reshape(len(rk), 1, 1)  # N * 1 * 1
    outers = outers * rk     # N * D * D
    sumOuters = np.sum(outers, axis=0) # D * D

    return sumOuters / np.sum(rk)

# MLE, 当D过大时，会有严重的过拟合，以及病态矩阵，数值不稳定性  
def EM(data, pi, mu, cov):
    # E step, 主要是计算r(ik)
    r = GetResponsibility(data, pi, mu, cov)
    # r = []
    # for i in range(len(data)):
    #    xi = data[i]
    #    r.append(GetResponsibility(xi, pi, mu, cov))

    # M step, 重新计算pi, mu, cov
    pi_new = np.mean(r, axis=0)
    mu_new = []
    cov_new = []
    for k in range(len(mu)):
        rk = r[:, k].reshape(1, -1)
        mu_k = np.dot(rk, data) / np.sum(rk)
        mu_new.append(mu_k.ravel())
        cov_new.append(GetCovK(data, mu_k, rk))

    # print('r: \n', r)
    return r, pi_new, np.array(mu_new), np.array(cov_new)

def MakeNIWPrior(data, K=3):
    '''dependent on data'''
    D = data.shape[1]  # 数据的维度
    s = np.var(data, axis=0)
    
    m0 = np.zeros(D)
    k = 0.01
    dof = D + 2
    S0 = ((1/K)**(1/D)) * np.diag(s)

    return m0, k, dof, S0

# MAP，当D过大时依然有很好的数值稳定性
def EM_MAP(data, pi, mu, cov, prior):
    # E step, 主要是计算r(ik)
    r = GetResponsibility(data, pi, mu, cov)

    # M step, 重新计算pi, mu, cov
    m0 = prior[0]
    k0 = prior[1]
    dof = prior[2]
    S0 = prior[3] # prior ~ NIW(μk,Σk|m0, κ0, ν0, S0)
    
    pi_new = np.mean(r, axis=0)   # 书中给pi也分配了狄利克雷分布的prior，但这里省略
    mu_new = []
    cov_new = []
    for k in range(len(mu)):
        rk = r[:, k].reshape(1, -1)
        xk = np.dot(rk, data) / np.sum(rk)
        mu_k = ((np.dot(rk, data) + k0 * m0) / (np.sum(rk) + k0)).ravel()
        Sk = np.sum(rk) * GetCovK(data, xk, rk)
        ratio = k0 * np.sum(rk) / (k0 + np.sum(rk))
        gap = (xk - m0).reshape(-1, 1)
        cov_k = (S0 + Sk + ratio * np.dot(gap, gap.T)) / (dof + np.sum(rk) + data.shape[1] + 2)
        
        mu_new.append(mu_k)
        cov_new.append(cov_k)

    return r, pi_new, np.array(mu_new), np.array(cov_new)

