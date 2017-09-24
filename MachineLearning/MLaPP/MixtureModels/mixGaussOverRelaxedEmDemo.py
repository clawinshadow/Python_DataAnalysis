import math
import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
import scipy.special as ssp
import matplotlib.pyplot as plt
from mixGaussFit import *

'''
这个采用的算法是Adaptive的OverRelax EM, A-OREM, 不是书里面讲的一般的OREM，因为eta在迭代过程中会做一些自适应的变化，
而一般的OREM中eta是固定的，可能会造成收敛的失败。
'''

def SetPrior(data, K=3):
    '''dependent on data'''
    D = data.shape[1]  # 数据的维度
    
    m0 = np.zeros(D)
    k = 0.01
    dof = D + 1
    S0 = 0.1 * np.eye(D)

    return m0, k, dof, S0

# 计算数据集的log-likelihood
def LogLikelihood(X, pi, mu, sigma, prior):
    K = len(sigma)
    N = len(X)
    probs = np.zeros((N, K))
    for i in range(K):
        probs[:, i] = math.log(pi[i]) + ss.multivariate_normal(mu[i], sigma[i], allow_singular=True).logpdf(X)

    total = ssp.logsumexp(probs, axis=1)
    dataLL = np.sum(total)

    # calculate prior ll
    priorLL = 0
    prior_mu, prior_k, prior_dof, prior_S = prior
    for i in range(K):
        priorLL += ss.multivariate_normal(prior_mu, sigma[i]/prior_k,  allow_singular=True).logpdf(mu[i])
        priorLL += ss.invwishart(df=prior_dof, scale=prior_S).logpdf(sigma[i])

    return dataLL + priorLL

def OverrelaxEM(X, pi, mu, sigma, prior, eta):
    r, pi_new, mu_new, sigma_new = EM_MAP(X, pi, mu, sigma, prior)  # 普通EM算法得出来的新参数
    # update pi in Overrelax way
    pi_or = pi * (pi_new / pi)**eta
    pi_or = pi_or / np.sum(pi_or)
    # update mu and sigma in Overrelax way
    mu_or = np.zeros(mu.shape)
    sigma_or = np.zeros(sigma.shape)
    K = len(sigma)
    valid = True
    for i in range(K):
        mu_or[i] = mu[i] + eta * (mu_new[i] - mu[i])
        try:
            logSigma = sl.logm(sigma[i])
            logSigmaNew = sl.logm(sigma_new[i])
            logSigma = logSigma + eta * (logSigmaNew - logSigma)
            sigma_or[i] = sl.expm(logSigma)
        except Exception as e:
            print(e)
            valid = False
            break
        if not IsPosDef(sigma_or[i]):
            valid = False
            break

    return valid, [pi_or, mu_or, sigma_or], [pi_new, mu_new, sigma_new]
    
def AdaptiveOR_EM(X, eta, K=10, maxIter=30):
    LLs = []
    D = X.shape[1]
    # pi, mu, sigma = GetParamInitial(D, K)
    pi, mu, sigma = KMeansInit(X, K)
    prior = SetPrior(X, K)
    for i in range(maxIter):
        LL_current = LogLikelihood(X, pi, mu, sigma, prior)
        print('LL: ', LL_current)
        LLs.append(LL_current)
        valid, model_or, model_new = OverrelaxEM(X, pi, mu, sigma, prior, eta)
        if valid:
            LL_new = LogLikelihood(X, model_or[0], model_or[1], model_or[2], prior)
            valid = LL_new > LL_current   # 至少要满足新的LL大于旧的LL
        if valid:
            eta = eta * 2                 # 如果上一步要求满足，则加大步长，乘2后再看看
        else:
            eta = 1
            model_or = model_new          # 恢复到标准EM

        pi = model_or[0]
        mu = model_or[1]
        sigma = model_or[2]

    return np.array(LLs)

def GenerateData(trial):
    np.random.seed(trial)
    D = 15
    N = 5000
    K = 10
    pi, mu, sigma = GetParamInitial(D, K, 0)
    mu = np.random.rand(K, D)
    counts = ss.multinomial(N, pi).rvs(1).ravel()
    print('counts: ', counts)
    data = np.zeros((1, D))
    for i in range(len(counts)):
        count_i = counts[i]
        mu_i = mu[i]
        sigma_i = sigma[i]
        samples_i = ss.multivariate_normal(mu_i, sigma_i).rvs(count_i)
        data = np.vstack((data, samples_i))
    data = data[1:]   # delete the first row
    np.random.shuffle(data)

    return data

# plots
fig = plt.figure(figsize=(11, 5))
fig.canvas.set_window_title('mixGaussOverRelaxedEmDemo')

def Plot(trial):
    plt.title('K=5, D=15, N=5000')
    plt.xlabel('iterations')
    plt.ylabel('loglik')
    N = 24
    plt.xticks(np.arange(0, N, 2))
    X = GenerateData(trial)
    eta = [1, 1.25, 2, 5]
    colors = ['r', 'k', 'g', 'c']
    markers = ['x', 'd', '>', '<']
    for i in range(len(eta)):
        etai = eta[i]
        LLs = AdaptiveOR_EM(X, etai, 5, N)
        LLs = LLs[1:]  # delete the first LL

        if eta[i] == 1:
            plt.plot(np.arange(1, N, 1), LLs, color='b', marker='o', markersize=5, label='EM', fillstyle='none')
        labelStr = 'OR({0})'.format(str(etai))
        plt.plot(np.arange(1, N, 1), LLs, color=colors[i], marker=markers[i], markersize=6, label=labelStr, fillstyle='none')
    plt.legend()

plt.subplot(121)
Plot(1)
plt.subplot(122)
Plot(2)

plt.show()



