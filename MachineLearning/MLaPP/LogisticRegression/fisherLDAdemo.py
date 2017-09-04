import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
import sklearn.linear_model as slm
import sklearn.decomposition as sd
import matplotlib.pyplot as plt

def GetSW(X):
    mu = np.mean(X, axis=0)
    sw = np.zeros((X.shape[1], X.shape[1]))
    for i in range(len(X)):
        gap = (X[i] - mu).reshape(-1, 1)
        sw += np.dot(gap, gap.T)

    return sw

def GetY(center, x, w):
    return center[1] + (w[1]/w[0]) * (x - center[0])

# Generate data
np.random.seed(0)

mu = np.array([[1, 3], [3, 1]])
cov = np.array([[4, 0.01], [0.01, 0.1]])
n_samples = 200
points_0 = ss.multivariate_normal(mu[0], cov).rvs(100)   # blue +
points_1 = ss.multivariate_normal(mu[1], cov).rvs(100)   # red circle
X = np.vstack((points_0, points_1))
y = np.vstack((np.zeros(len(points_0)).reshape(-1, 1), np.ones(len(points_1)).reshape(-1, 1)))

# fit with Fisher LDA
mu0 = np.mean(points_0, axis=0)
mu1 = np.mean(points_1, axis=0)
center = 0.5 * (mu0 + mu1)
print('mean 0: ', mu0)
print('mean 1: ', mu1)
meanGap = (mu0 - mu1).reshape(-1, 1)
SB = np.dot(meanGap, meanGap.T)  # 2 * 2 矩阵
SW = GetSW(points_0) + GetSW(points_1)
eigVals, eigVectors= sl.eig(np.dot(sl.inv(SW), SB))
print('eigVals: ', eigVals)
print('eigVectors: ', eigVectors)
w = eigVectors[:, -1].reshape(-1, 1)

# fit with PCA
PCA = sd.PCA(n_components=2)
PCA.fit(X)
w_PCA = PCA.components_[0].reshape(-1, 1)
print('w_PCA: ', w_PCA.ravel())

# calculate points projections
LDA_Projections = np.dot(X, w)
PCA_Projections = np.dot(X, w_PCA)
print('LDA_Projections: ', LDA_Projections)
print('PCA_Projections: ', PCA_Projections)

# plots
x = np.linspace(-4, 8, 200)
y_lda = GetY(center, x, w)
y_pca = GetY(center, x, w_PCA)

fig = plt.figure(figsize=(11, 10))
fig.canvas.set_window_title('fisherLDAdemo')

plt.subplot(221)
plt.axis([-4, 8, 0, 4])
plt.xticks(np.arange(-4, 9, 2))
plt.yticks(np.arange(0, 4.1, 1))
plt.plot(points_0[:, 0], points_0[:, 1], 'b+', ls='none')
plt.plot(points_1[:, 0], points_1[:, 1], 'ro', ls='none', fillstyle='none')
plt.plot(x, y_lda, 'r-', label='fisher')
plt.plot(x, y_pca, 'g:', label='pca')

plt.legend()
plt.show()
