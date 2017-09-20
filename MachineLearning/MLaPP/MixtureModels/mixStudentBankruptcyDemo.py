import math
import numpy as np
import scipy.io as sio
import scipy.stats as ss
import scipy.linalg as sl
import scipy.optimize as so
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.special import digamma, gammaln
from mixGaussFit import *

'''
mixStudent计算起来比mixGauss复杂一点，但是算法是差不多的，简单写一下EM的逻辑：
所有的参数θ包括：
  pi：混合系数，1 * K
  mu：每个分类中student分布的均值向量，K * D
  sigma: 每个分类中student分布的协方差矩阵，K * D * D
  dof: 每个分类中student分布的自由度，可以是已知的，也可以是未知的。未知的计算起来会更复杂一点，本例中只给出dof已知的算法

E step:
  1. 与mixGauss完全相同，只是T分布给出的概率密度不一样而已，计算 r 矩阵，N * K
  2. 计算每个数据点xi与当前分类的mu的马氏距离，记为 delta_i_k = (xi - mu).T * sl.inv(sigma_k) * (xi - mu)
  
                        dof_k + D
  3. 计算 z_i_k = ---------------------
                    dof_k + delta_i_k

M step: 遍历分类数量K来进行计算，对每个k来说：
                    
                   Σ(r_i_k * z_i_k * xi)
  1. 更新 mu_k = --------------------------, Σ 为 i: 1 ~ N
                      Σ(r_i_k * z_i_k)

                      1
  2. 更新 sigma_k = ----- * [Σ(r_i_k * z_i_k * xi * xi.T) - Σ(r_i_k * z_i_k) * mu_k * mu_k.T]
                     r_k

     其中 r_k = Σ(r_i_k), i: 1 ~ N

  3. 更新pi: pi_k = r_k / N
'''

np.random.seed(0)

def pdf(x, mu, sigma, dof, inv_sigma):
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
    val_3 = 1 + (1/dof) * np.dot(gap.T, np.dot(inv_sigma, gap))

    return (val_1 / val_2) * val_3**(-(dof + d) / 2) 

# 计算单个数据点的R值， K * 1
def GetR_Z(x, pi, mu, sigma, dof, inv_sigma):
    r, z = [], []
    total = 0
    D = len(x)
    for i in range(len(pi)):
        prob = pi[i] * pdf(x, mu[i], sigma[i], dof[i], inv_sigma[i])
        total += prob
        r.append(prob)

        distance = mahalanobis(x.ravel(), mu[i].ravel(), inv_sigma[i])
        zik = (dof[i] + D) / (dof[i] + distance)
        z.append(zik)

    r, z = np.array(r), np.array(z)   
    return (r / total).ravel(), z.ravel()

# 计算R矩阵, Z矩阵
def EStep(X, pi, mu, sigma, dof):
    # 为了提高效率，提前计算好每个sigma的逆矩阵
    inv_sigma = []
    for i in range(len(sigma)):
        inv_sigma.append(sl.inv(sigma[i]))
        
    rMat, zMat = [], []
    for i in range(len(X)):
        xi = X[i]
        r, z = GetR_Z(xi, pi, mu, sigma, dof, inv_sigma)
        rMat.append(r)
        zMat.append(z)
        
    return np.array(rMat), np.array(zMat)  # N * K

# M step 中就不再需要pi, mu ... 这些参数了，已经拿到期望值r和z了
def MStep(X, r, z, dof, updateDof=False):
    K, N = r.shape[1], X.shape[0]
    mu = []
    sigma = []
    pi = []
    dof_new = []
    for i in range(K):
        rk = r[:, i]
        zk = z[:, i]
        w = (rk * zk).reshape(-1, 1)  # N * 1
        mu_k = np.sum(w * X, axis=0) / np.sum(w)
        sigma_k = UpdateSigma(X, mu_k, r, z, i)
        pi_k = np.sum(rk) / N

        mu.append(mu_k.ravel())
        sigma.append(sigma_k)
        pi.append(pi_k)

    # update dof
    if updateDof:
        L = np.zeros((N, K))
        for i in range(K):
            sigma_k = sigma[i]
            logdet = 0.5 * math.log(sl.det(sigma_k))
            logmix = math.log(pi[i])
            distances = []
            inv_sigma_k = sl.inv(sigma_k)
            for j in range(len(X)):
                xj = X[j]
                distances.append(mahalanobis(xj.ravel(), mu[i].ravel(), inv_sigma_k))
            distances = np.array(distances).ravel()
            L[:, i] = GetLk(dof[i], logmix, logdet, distances, X.shape[1])

        print('L: \n', L)

        for i in range(K):
            sigma_k = sigma[i]
            logdet = 0.5 * math.log(sl.det(sigma_k))
            logmix = math.log(pi[i])
            distances = []
            inv_sigma_k = sl.inv(sigma_k)
            for j in range(len(X)):
                xj = X[j]
                distances.append(mahalanobis(xj.ravel(), mu[i].ravel(), inv_sigma_k))
            distances = np.array(distances).ravel()
            
            bnds = ((0, None),)
            x_init = (dof[i],)
            res = so.minimize(dofFunc, x0=x_init, args=(logmix, logdet, distances, X.shape[1], L, i), bounds=bnds)
            dof_new.append(res.x)
    else:
        dof_new = dof

    return np.array(mu), np.array(sigma), np.array(pi), np.array(dof_new)

def UpdateSigma(X, mu, r, z, k):
    rk = np.sum(r[:, k])
    val_1 = np.zeros((X.shape[1], X.shape[1]))
    for i in range(len(X)):
        xi = X[i].reshape(-1, 1)
        rik, zik = r[i, k], z[i, k]
        val_1 += rik * zik * np.dot(xi, xi.T)
    mu_k = mu.reshape(-1, 1)
    sumrz = np.sum(r[:, k] * z[:, k])
    val_2 = sumrz * np.dot(mu_k, mu_k.T)
    
    return (1/rk) * (val_1 - val_2)

def GetLk(x, logmix, logdets, distances, D):
    val1 = logmix + gammaln((x+D)/2) - gammaln(x/2) - logdets - D * math.log(x) / 2
    val2 = 0.5 * (x + D) * np.log1p(distances/x)
    gap = val1 - val2
    return gap

def dofFunc(x, logmix, logdets, distances, D, L, k):
    val1 = logmix + gammaln((x+D)/2) - gammaln(x/2) - logdets - D * math.log(x) / 2
    val2 = 0.5 * (x + D) * np.log1p(distances/x)
    gap = val1 - val2
    L[:, k] = gap
    lse = np.log(np.sum(np.exp(L), axis=1))
    return -np.sum(lse)

    # return N * gammaln(x/2) - (N*x/2) * math.log(x/2) - (x/2) * val
    # return N * digamma(x/2) - N * math.log(x/2) - N - val

def mixStudentFit(X, K=2, maxIter=50):
    mu = np.r_[np.random.randn(1, 2), np.random.randn(1, 2)]
    sigma = GetSigmaInit(X.shape[1], 2, K)
    dof = np.tile(10, K)
    pi = GetPiInit(2, K)

    for i in range(maxIter):
        r, z = EStep(X, pi, mu, sigma, dof)
        mu_new, sigma_new, pi_new, dof_new = MStep(X, r, z, dof, True)
        print('{0:-^60}'.format('Iteration ' + str(i)))
        print('mu: \n', mu_new)
        print('sigma: \n', sigma_new)
        print('pi: ', pi_new)
        print('dof: ', dof_new)
        pi = pi_new
        mu = mu_new
        sigma = sigma_new
        dof = dof_new

    return pi, mu, sigma, dof

def mixGaussFit(X, K=2, maxIter=50):
    mu = np.r_[np.random.randn(1, 2), np.random.randn(1, 2)]
    sigma = GetSigmaInit(X.shape[1], 2, K)
    pi = GetPiInit(2, K)
    prior_mu = np.zeros(X.shape[1])
    prior_k = 0.01
    prior_dof = X.shape[1] + 1
    prior_s = 0.1 * np.eye(X.shape[1])
    prior = [prior_mu, prior_k, prior_dof, prior_s]

    for i in range(maxIter):
        r, pi_new, mu_new, sigma_new = EM_MAP(X, pi, mu, sigma, prior)
        print('{0:-^60}'.format('Iteration ' + str(i)))
        print('mu: \n', mu_new)
        print('sigma: \n', sigma_new)
        print('pi: ', pi_new)
        pi = pi_new
        mu = mu_new
        sigma = sigma_new

    return pi, mu, sigma

# load data
data = sio.loadmat("bankruptcy.mat")
x = data['data'][:, 1:]
x_train = sp.StandardScaler().fit_transform(x)  # 标准化
y_train = data['data'][:, 0]
Bankrupt = x_train[y_train == 0]
Solvent = x_train[y_train == 1]

# plots
fig = plt.figure(figsize=(11, 5))
fig.canvas.set_window_title("mixStudentBankruptcyDemo")

plt.subplot(121)
plt.title('14 errors using gauss (red=error)')
plt.axis([-5, 2, -7, 3])
plt.xticks(np.linspace(-5, 2, 8))
plt.yticks(np.linspace(-7, 3, 11))
plt.plot(Bankrupt[:, 0], Bankrupt[:, 1], 'bo', ls='none', fillstyle='none', label='Bankrupt')
plt.plot(Solvent[:, 0], Solvent[:, 1], 'b^', ls='none', fillstyle='none', label='Solvent')
plt.legend()

# Fit Mix-Student model
res = mixStudentFit(x_train)
# res2 = mixGaussFit(x_train)


plt.show()
