import numpy as np
import scipy.linalg as sl
import sklearn.preprocessing as spp
import matplotlib.pyplot as plt
from pcaFit import *

np.random.seed(0)
    
# generate data
n = 5
x = np.r_[np.random.randn(n, 2) + 2 * np.ones((n, 2)),\
          2 * np.random.randn(n, 2) - 2 * np.ones((n, 2))]

# Fit PCA
L = 1
mu, vr, W, z, x_recon = PCA(x, L)
print('W:\n', W)
Z2 = np.array([-5, 5])
x_recon2 = np.dot(Z2.reshape(-1, 1), W.T) + mu
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
plt.plot(x_recon2[:, 0], x_recon2[:, 1], color='m')
plt.plot(mu[0], mu[1], 'r*', ms=10)
for i in range(len(x)):
    plt.plot([x[i, 0], x_recon[i, 0]], [x[i, 1], x_recon[i, 1]], color='darkblue')

plt.show()
