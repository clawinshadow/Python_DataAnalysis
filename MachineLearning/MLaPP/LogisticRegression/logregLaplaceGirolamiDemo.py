import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
import scipy.optimize as so
import matplotlib.pyplot as plt

np.random.seed(0)

# generate data
mu_0 = [-5, 1]   # 分类0的数据
cov_0 = 1.5**2 * np.eye(2)

mu_1 = [1, 5]    # 分类1的数据
cov_1 = np.eye(2)

N = 30           # 每个分类的样本点数
rv0 = ss.multivariate_normal(mean=mu_0, cov=cov_0)
rv1 = ss.multivariate_normal(mean=mu_1, cov=cov_1)

X = np.vstack((rv0.rvs(N), rv1.rvs(N)))
y = np.concatenate((np.zeros(N), np.ones(N)))

isZero = y == 0
points_0 = X[isZero]
points_1 = X[~isZero]

# plot the points
fig = plt.figure(figsize=(11, 10))
fig.canvas.set_window_title('logregLaplaceGirolamiDemo')

plt.subplot(221)
plt.axis([-10, 5, -8, 8])
plt.xticks(np.linspace(-10, 5, 4))
plt.yticks(np.arange(-8, 9, 2))
plt.title('data')
plt.plot(points_0[:, 0], points_0[:, 1], 'bo', ls='none', fillstyle='none', ms=5)
plt.plot(points_1[:, 0], points_1[:, 1], 'ro', ls='none', fillstyle='none', ms=1)

# plot the candidate decision boundary
x = np.arange(-8, 8, 0.2)
w = [-2, -2.5, -3, -3.5]
plt.plot(x, w[0] * x, 'k-')
plt.plot(x, w[1] * x, 'r-')
plt.plot(x, w[2] * x, 'g-')
plt.plot(x, w[3] * x, 'b-')

# Log Likelihood
def logistic(x):
    return 1 / (1 + np.exp(-x))

def GetNLL(X, y, w):
    '''计算训练集的Negative - log - likelihood'''
    probs_0 = np.log(1 - logistic(np.dot(points_0, w.reshape(-1, 1))))
    probs_1 = np.log(logistic(np.dot(points_1, w.reshape(-1, 1))))
    probs = np.concatenate((probs_0.ravel(), probs_1.ravel()))

    return -1 * np.sum(probs)

w1, w2 = np.meshgrid(np.arange(-8, 8, 0.11), np.arange(-8, 8, 0.11))
w1_ravel = w1.ravel()
w2_ravel = w2.ravel()
likelihood = []
for i in range(len(w1_ravel)):
    w = np.array([w1_ravel[i], w2_ravel[i]])
    likelihood.append(GetNLL(X, y, w))
likelihood = np.array(likelihood).reshape(w1.shape)

# plot Negative log likelihood
# 当w偏离太远时，NLL会有很多时候算出来是inf，这个在计算机系统中因为是数值计算的限制，无法解决
# 反映在图形中的时候，inf是不画的，所以有一块地方都是空白
plt.subplot(222)
plt.axis([-8, 8, -8, 8])
plt.xticks(np.arange(-8, 9, 2))
plt.yticks(np.arange(-8, 9, 2))
plt.title('Log-Likelihood')
plt.contour(w1, w2, likelihood, 20, cmap='jet')

# 利用IRLS计算w的轨迹
def IRLS(X, y, w0, maxIter=100):
    i = 0
    w = np.array(w0).reshape(-1, 1)
    ws = [w.ravel()]
    while i < maxIter:
        eta = np.dot(X, w)
        mu = logistic(eta)
        S = np.diag((mu * (1 - mu)).ravel())
        part_1 = np.dot(sl.inv(np.dot(X.T, np.dot(S, X))), X.T)
        part_2 = np.dot(S, np.dot(X, w)) + y.reshape(-1, 1) - mu.reshape(-1, 1)
        w_next = np.dot(part_1, part_2)

        ws.append(w_next.ravel())
        i += 1

    return np.array(ws)

# plot the trace of w
w0 = [-2, -2]
ws = IRLS(X, y, w0)
print('ws.shape: ', ws.shape)
plt.plot(ws[:, 0], ws[:, 1], color='black') # 一条直线，没有尽头

# Posterio P(w|D), 这里只能计算unnormalised的后验概率图，p(D)没法计算
# 假设权重 w 服从 先验概率分布 N(w|0, 100I)
alpha = 100
prior = ss.multivariate_normal([0, 0], alpha * np.eye(2))
prior_probs = prior.logpdf(np.dstack((w1, w2)))
posterior_likelihood = likelihood - prior_probs

# plot the Log−Unnormalised Posterior
plt.subplot(223)
plt.axis([-8, 8, -8, 8])
plt.xticks(np.arange(-8, 9, 2))
plt.yticks(np.arange(-8, 9, 2))
plt.title('Log−Unnormalised Posterior')
plt.contour(w1, w2, posterior_likelihood, 20, cmap='jet')

# 计算MAP
def NLL_Posterior(w, X, y, priorModel):
    return GetNLL(X, y, w) - priorModel.logpdf(w)

w0 = [0, 0]
res = so.minimize(NLL_Posterior, w0, args=(X, y, prior))
print('MAP : ', res.x)
w_MAP = res.x

# plot MAP
plt.plot(w_MAP[0], w_MAP[1], ls='none', marker='o', color='midnightblue', ms=10)

# calc Laplace Approximation of posterior, 也叫做 Gauss Approximation
# 计算Hessian矩阵先, NLL 的 H = X.T * S * X
mu = logistic(np.dot(X, w_MAP.reshape(-1, 1)))
S = np.diag((mu * (1 - mu)).ravel())
H_NLL = np.dot(X.T, np.dot(S, X))
# 加入L2的penalize后，lambda = 1 / alpha, H矩阵变为：
H = H_NLL + (1 / alpha) * np.eye(len(w_MAP))
cov = sl.inv(H)
laplaceApprx = ss.multivariate_normal(w_MAP, cov)

# plot Laplace Approximation
Z = laplaceApprx.pdf(np.dstack((w1, w2)))

plt.subplot(224)
plt.axis([-8, 8, -8, 8])
plt.xticks(np.arange(-8, 9, 2))
plt.yticks(np.arange(-8, 9, 2))
plt.title('Laplace Approximation to Posterior')
plt.plot(w_MAP[0], w_MAP[1], ls='none', marker='o', color='midnightblue', ms=5)
plt.contour(w1, w2, Z, 20, cmap='jet')

plt.show()

