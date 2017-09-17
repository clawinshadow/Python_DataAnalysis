import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt
from mixGaussFit import *

np.random.seed(0)

# Generate Data, fixed K = 3
def MakeCov(origMat, D):
    m11 = origMat
    m21 = np.zeros((D - 2, 2))  # origMat必须为(2, 2)
    m12 = np.zeros((2, D - 2))
    m22 = np.eye(D - 2)
    return np.r_[np.c_[m11, m12], np.c_[m21, m22]]

def GetInitial(D, K=3):
    mu_init = np.random.rand(K, D)
    mixWeights_init = np.tile(1/K, K)
    return mu_init, mixWeights_init
    
def Sample(D, N=100):
    K = 3  # 固定的
    mean_1 = np.r_[-1, 1, np.zeros(D-2)]
    mean_2 = np.r_[1, -1, np.zeros(D-2)]
    mean_3 = np.r_[3, -1, np.zeros(D-2)]
    cov_1 = MakeCov([[1, -0.7],
                     [-0.7, 1]], D)
    cov_2 = MakeCov([[1, 0.7],
                     [0.7, 1]], D)
    cov_3 = MakeCov([[1, 0.9],
                     [0.9, 1]], D)
    n = [0.5, 0.3, 0.2]  # 采样的数量
    x1 = ss.multivariate_normal(mean_1, cov_1).rvs((int)(n[0] * N))
    x2 = ss.multivariate_normal(mean_2, cov_2).rvs((int)(n[1] * N))
    x3 = ss.multivariate_normal(mean_3, cov_3).rvs((int)(n[2] * N))
    
    x = np.r_[x1, x2, x3]
    sigma_init = np.array([cov_1, cov_2, cov_3])
    return x, sigma_init  # 返回的是数据样本集， 以及协方差的初始值(固定的)

# Fit model with MLE or MAP 
def Fit(x, pi, mu, cov, isMAP=False):
    success = True
    try:
        maxIter = 30
        cov_old = cov
        pi_old = pi
        mu_old = mu
        prior = MakeNIWPrior(x)
        for i in range(maxIter):
            if isMAP:
                r, pi_new, mu_new, cov_new = EM_MAP(x, pi_old, mu_old, cov_old, prior)
            else:
                r, pi_new, mu_new, cov_new = EM(x, pi_old, mu_old, cov_old)
            #print('{0:-^60}'.format('Iteration: ' + str(i + 1)))
            #print('pi: ', pi_new)
            if np.allclose(pi_new, pi):
                print('converged')
                break
            pi_old = pi_new
            mu_old = mu_new
            cov_old = cov_new
    except Exception as e:
        print(e)
        success = False
    return success

# Fit with several trials
def GetFailRatio(D, trials=10):
    print('D = ', D)
    x, cov = Sample(D)
    x = sp.StandardScaler().fit_transform(x)
    MLE_fail, MAP_fail = 0, 0
    for i in range(trials):
        mu, pi = GetInitial(D)    # 每次尝试，不一样的初始值
        if not Fit(x, pi, mu, cov, True):
            MAP_fail += 1
        if not Fit(x, pi, mu, cov, False):
            MLE_fail += 1
    print('MLE_fail, MAP_fail: ', MLE_fail, MAP_fail)
    return [MLE_fail / trials, MAP_fail / trials]

D = np.arange(10, 101, 10)
ratios = []
for i in range(len(D)):
    Di = D[i]
    ratios.append(GetFailRatio(Di))
ratios = np.array(ratios)
print('ratios: \n', ratios)

# plots
fig = plt.figure()
fig.canvas.set_window_title("mixGaussMLvsMAP")

plt.subplot()
plt.axis([5, 105, -0.04, 1.04])
plt.xticks(np.arange(10, 101, 10))
plt.yticks(np.arange(0, 1.01, 0.1))
plt.xlabel('dimensionality')
plt.ylabel('fraction of times EM for GMM fails')
plt.plot(D, ratios[:, 0], 'r-', marker='o', fillstyle='none', label='MLE')
plt.plot(D, ratios[:, 1], 'k:', marker='s', fillstyle='none', label='MAP')

plt.legend()
plt.show()
