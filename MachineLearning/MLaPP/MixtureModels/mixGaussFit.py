import numpy as np
import scipy.stats as ss

'''
包含MLE和MAP的EM算法，对于GMM的Fit.

因为EM算法只保证收敛到一个局部最小值，所以千万不要忽视初始值和是否有Prior的重要性！
初始值不一样，或者是否给一个prior，得出来的结果往往有很大的差别。
'''

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

def GetResponsibility(x, pi, mu, cov):
    r = []
    total = 0
    for k in range(len(pi)):
        pi_k = pi[k]
        gaussianProb_k = ss.multivariate_normal(mu[k], cov[k]).pdf(x)
        rk = pi_k * gaussianProb_k
        
        r.append(rk)
        total += rk

    return r / total

def GetCovK(data, xk, rk):
    dim = len(xk.ravel())
    cov = np.zeros((dim, dim))
    for i in range(len(data)):
        rik = rk.ravel()[i]
        gap = data[i].reshape(-1, 1) - xk.reshape(-1, 1)
        cov_i = rik * np.dot(gap, gap.T)
        cov += cov_i

    return cov / np.sum(rk)

# MLE, 当D过大时，会有严重的过拟合，以及病态矩阵，数值不稳定性  
def EM(data, pi, mu, cov):
    # E step, 主要是计算r(ik)
    r = []
    for i in range(len(data)):
        xi = data[i]
        r.append(GetResponsibility(xi, pi, mu, cov))

    r = np.array(r)
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
    r = []
    for i in range(len(data)):
        xi = data[i]
        r.append(GetResponsibility(xi, pi, mu, cov))

    r = np.array(r)
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

