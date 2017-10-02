import numpy as np
import scipy.linalg as sl
import scipy.stats as ss
import matplotlib.pyplot as plt

'''
有时间可以改成一个Animation
'''

np.random.seed(10)

def basisAttr():
    plt.axis([-3, 3, -3, 2.5])
    plt.xticks(np.linspace(-3, 3, 7))
    plt.yticks(np.arange(-3, 2.6, 0.5))

def plotGauss(mu, sigma):
    D = len(sigma)
    assert D == 2  # 只画2维的高斯分布
    X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-2.5, 2.5, 100))
    Z = np.dstack((X, Y))
    rv = ss.multivariate_normal(mu.ravel(), sigma)
    pdf = rv.pdf(Z)
    level = pdf.min() + (pdf.max() - pdf.min()) * 0.2
    plt.contour(X, Y, pdf, levels=[level], colors='red')
    
def getXrecon(X, W, Z, L):
    N = X.shape[1]
    vals, vecs = sl.eig(np.dot(Z.T, Z) / N)
    sortedIndices = np.argsort(vals)[::-1] # descend order
    L_indices = sortedIndices[:L]           # top L indices
    vr = (vecs.T[L_indices]).T            # top L eigen vectors, D * L
    West = W * vr # 2 * 1
    Z2 = np.dot(X.T, West) # N * 1
    x_recon = np.dot(Z2, West.T)

    return x_recon.T
    
# generate data
N, d = 25, 2
mu = ss.multivariate_normal([1, 0], np.eye(d)).rvs(1)
print('true mu: ', mu)
sigma = np.array([[1, -0.7],
                  [-0.7, 1]]) # true sigma
X = ss.multivariate_normal(mu, sigma).rvs(N)
print(X.shape)

k = 1
mu = np.mean(X, axis=0)
X = X - mu   # center the data
X = X.T      # 书中用的是 D * N 的矩阵， 2 * 25

# 在svd的返回值中，v中每一行是个奇异值对应的向量，不是每一列。u则是每一列
u, s, v = sl.svd(sigma)
w_true = v[0] # real principal component

u, s, v = sl.svd(np.cov(X, bias=True))
w_data = v[0] # principal component from data

w = np.random.rand(X.shape[0], k)  # initial value, 2 * 1

fig = plt.figure(figsize=(10, 8))
fig.canvas.set_window_title('pcaEmStepByStep')

# setup EM
iterCount = 1
maxIter = 100
converged = False
while (not converged) and (iterCount <= maxIter):
    # E Step
    Z = np.dot(sl.inv(np.dot(w.T, w)), np.dot(w.T, X))  # 1 * N
    X_recon = np.dot(w, Z)  # 2 * N
    w_orth = sl.orth(w)
    if iterCount <= 2:
        # plot E Step
        plt.subplot((int)('22' + str(2 * iterCount - 1)))
        basisAttr()
        plotGauss(np.zeros(X.shape[0]), sigma)
        plt.plot(X[0], X[1], 'gx', ls='none', ms=5)
        plt.plot(X_recon[0], X_recon[1], 'ko', ls='none', fillstyle='none', ms=10)
        plt.plot([-3, 3], [-3 * w_orth[1] / w_orth[0], 3 * w_orth[1] / w_orth[0]], 'c-')
        for i in range(X.shape[1]):
            xi, x_reconi = X[:, i], X_recon[:, i]
            plt.plot([xi[0], x_reconi[0]], [xi[1], x_reconi[1]], 'k-', lw=0.5)
        plt.title('E Step {0}'.format(iterCount))

    # M Step
    w_new = np.dot(np.dot(X, Z.T), sl.inv(np.dot(Z, Z.T))) # 2 * 1
    converged = np.allclose(w, w_new)

    w_orth_new = sl.orth(w_new)  # 2 * 1
    Z_M = np.dot(X.T, w_orth_new)  # N * 1
    x_recon_M = getXrecon(X, w_new, Z_M, k)
    if iterCount <= 2:
        # plot M Step
        plt.subplot((int)('22' + str(2 * iterCount)))
        basisAttr()
        plotGauss(np.zeros((1, X.shape[0])), sigma)
        plt.plot(X[0], X[1], 'gx', ls='none', ms=5)
        plt.plot(x_recon_M[0], x_recon_M[1], 'ko', ls='none', fillstyle='none', ms=10)
        plt.plot([-3, 3], [-3 * w_orth_new[1] / w_orth_new[0], 3 * w_orth_new[1] / w_orth_new[0]], 'c-')
        for i in range(X.shape[1]):
            xi, x_reconi = X[:, i], x_recon_M[:, i]
            plt.plot([xi[0], x_reconi[0]], [xi[1], x_reconi[1]], 'k-', lw=0.5)
        plt.title('M Step {0}'.format(iterCount))

    w = w_new
    iterCount += 1

plt.show()





