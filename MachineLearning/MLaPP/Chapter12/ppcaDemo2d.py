import numpy as np
import scipy.linalg as sl
import sklearn.preprocessing as spp
import matplotlib.pyplot as plt
from pcaFit import *

'''
要注意为什么ppca和pca的图形中不同的地方在于投影不再垂直于w的方向，就是在于 w -> z 的转化过程中，
1. PCA是 zi = w.T * xi, 它的w是正交的，norm(w) = 1, 所以zi就是xi在w方向上的垂直投影 (简单的线性代数)
2. 而在 PPCA 中， 算出MLE的 w 之后，虽然此时w是正交的，但还要经过一些另外的变换，最终 zi = W_after.T * xi, 此时的W_after的范数就
不再是1了，所以PPCA中 zi 不是 xi在w方向上的垂直投影，而是都向着mean偏移，被mean吸引过去的感觉

原始数据 X 是 N*D 的，假如 N>D，并且是满秩矩阵，那么X就是一个D维的线性空间
而 W 是 N*L的，L < D， 所以 W 就是一个L维的线性子空间，一个L维的manifold，
隐藏变量 Z 都在W形成的这个L维流形上面，这就是PCA的几何意义
'''

np.random.seed(0)

# generate data
n = 5
x = np.r_[np.random.randn(n, 2) + 2 * np.ones((n, 2)),\
          2 * np.random.randn(n, 2) - 2 * np.ones((n, 2))]

# fit model
L = 1
mu, W, Z, x_recon = PPCA(x, L)
print('W: \n', W)
print(x_recon)
Z2 = np.array([-5, 5])
x_recon2 = np.dot(Z2.reshape(-1, 1), W.T) + mu

# plot data
fig = plt.figure()
fig.canvas.set_window_title('ppcaDemo2d')

plt.subplot()
plt.axis([-5, 5, -4, 5])
plt.xticks([-5, 0, 5])
plt.yticks(np.linspace(-4, 5, 10))
plt.plot(x[:, 0], x[:, 1], 'ro', mew=2, fillstyle='none')
plt.plot(x_recon[:, 0], x_recon[:, 1], 'g+', mew=2, ms=10)
plt.plot(x_recon2[:, 0], x_recon2[:, 1], color='m')
plt.plot(mu[0], mu[1], 'r*', ms=10)
for i in range(len(x)):
    plt.plot([x[i, 0], x_recon[i, 0]], [x[i, 1], x_recon[i, 1]], color='darkblue')

plt.show()
