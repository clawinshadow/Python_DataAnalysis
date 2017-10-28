import numpy as np
import scipy.io as sio
import scipy.linalg as sl
import sklearn.decomposition as sd
import matplotlib.pyplot as plt

'''
书中的这个示例是有问题的，kpca的图形在matlab code中也无法重现
并且matlab code中画kpca的代码非常奇怪，本例中是使用sklearn画的, 看上去跟书里面的完全吻合
'''

# load data
data = sio.loadmat('kpcaDemo2.mat')
print(data.keys())
X = data['patterns']
print(X.shape)
N, D = X.shape
rbf_var = 0.01

# reconstruct with PCA
pca = sd.PCA(D).fit(X)
X_pca = pca.transform(X)

# reconstruct with KPCA
kpca = sd.KernelPCA(N, kernel='rbf', gamma=1/rbf_var).fit(X)
X_kpca = kpca.transform(X)
print(X_kpca.shape)

# plots
fig = plt.figure(figsize=(11, 5))
fig.canvas.set_window_title('kpcaDemo2')

plt.subplot(121)
plt.axis([-0.6, 0.8, -0.8, 0.6])
plt.title('pca')
plt.xticks(np.linspace(-0.6, 0.8, 8))
plt.yticks(np.linspace(-0.8, 0.6, 8))
plt.plot(X_pca[:, 0], X_pca[:, 1], marker='x', color='darkblue', ls='none', mew=2)

plt.subplot(122)
plt.axis([-0.8, 0.8, -0.8, 0.6])
plt.title('kpca')
plt.xticks(np.linspace(-0.8, 0.8, 9))
plt.yticks(np.linspace(-0.8, 0.6, 8))
plt.plot(X_kpca[:, 0], X_kpca[:, 1], marker='x', color='darkblue', ls='none', mew=2)

plt.tight_layout()
plt.show()