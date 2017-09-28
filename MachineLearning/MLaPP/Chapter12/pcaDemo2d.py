import numpy as np
import scipy.linalg as sl
import sklearn.preprocessing as spp
import sklearn.decomposition as sd
import matplotlib.pyplot as plt

np.random.seed(0)

def PCA(X, L=1):
    x = spp.StandardScaler().fit_transform(X)   # 对PCA来说数据必须先标准化，否则得到的结果会很不一样
    N, D = X.shape
    assert D >= L
    x_3d = x.reshape(x.shape[0], x.shape[1], 1)
    x_3d_t = x.reshape(x.shape[0], 1, x.shape[1])
    S = x_3d * x_3d_t
    S = np.sum(S, axis=0) / N
    
    w, vr = sl.eig(S)                    # 求解特征值和特征向量
    sortedIndices = np.argsort(w)[::-1]  # 给特征值排序, 从大到小排列
    L_indices = sortedIndices[0:L]
    w = w[L_indices]
    vr = vr.T[L_indices]                 # vr中默认每列是特征向量，不是每行

    return w, vr
    
# generate data
n = 5
x = np.r_[np.random.randn(n, 2) + 2 * np.ones((n, 2)),\
          2 * np.random.randn(n, 2) - 2 * np.ones((n, 2))]
print(x)

# Fit PCA
L = 1
w, vecs = PCA(x, L)
print('w_diy: ', w)
print('vecs_diy: ', vecs)

x_samples = np.linspace(-5, 5, 100)
y_samples = x_samples * vecs[0, 1] / vecs[0, 0] # vecs is a 2d array
z = np.dot(x, vecs.reshape(-1, 1))  # latent variable z
print(z)
x_recon = np.dot(z, vecs)           # 重构x
x_recon_mean = np.mean(x_recon, axis=0)
print(x_recon)

# plot data
fig = plt.figure()
fig.canvas.set_window_title('pcaDemo2d')

plt.subplot()
plt.axis([-5, 5, -4, 5])
plt.axis('equal')  # 保证坐标系xlim, ylim内的图像是正方形的，两边留多少它会自己调整
plt.xticks([-5, 0, 5])
plt.yticks(np.linspace(-4, 5, 10))
plt.plot(x[:, 0], x[:, 1], 'ro', mew=2, fillstyle='none')
plt.plot(x_recon[:, 0], x_recon[:, 1], 'g+', mew=2, ms=10)
plt.plot(x_samples, y_samples, color='purple')
plt.plot(x_recon_mean[0], x_recon_mean[1], 'r*', ms=10)
for i in range(len(x)):
    plt.plot([x[i, 0], x_recon[i, 0]], [x[i, 1], x_recon[i, 1]], color='darkblue')

plt.show()
