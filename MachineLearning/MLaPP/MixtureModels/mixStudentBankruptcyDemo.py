import math
import numpy as np
import scipy.io as sio
import scipy.stats as ss
import scipy.linalg as sl
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt

np.random.seed(0)

def pdf(x, mu, sigma, dof):
    '''
    多元T分布的概率密度函数
    mu: 均值向量，d维
    sigma: 类似于协方差矩阵
    dof: 自由度
    '''
    d = sigma.shape[0]
    gap = (x - mu).reshape(-1, 1)
    val_1 = math.gamma((dof + d) / 2)
    val_2 = math.gamma(dof/2) * (dof * np.pi)**d/2 * sl.det(sigma)**0.5
    val_3 = 1 + (1/dof) * np.dot(gap.T, np.dot(sl.inv(sigma), gap))

    return (val_1 / val_2) * val_3**(-(dof + d) / 2) 

def GetR(x, pi, mu, sigma, dof):
    r = []
    total = 0
    for i in range(len(pi)):
        prob = pi[i] * pdf(x, mu[i], sigma[i], dof[i])
        total += prob
        r.append(prob)
    return r / total

# load data
data = sio.loadmat("bankruptcy.mat")
x = data['data'][:, 1:]
x_train = sp.StandardScaler().fit_transform(x)  # 标准化
y_train = data['data'][:, 0]

# Fit Mix-Student model
K = 2
mu_init = np.r_[np.random.randn(1, 2), np.random.randn(1, 2)]
sigma_init = np.array([np.eye(2), np.eye(2)])
dof_init = np.array([10, 10])
