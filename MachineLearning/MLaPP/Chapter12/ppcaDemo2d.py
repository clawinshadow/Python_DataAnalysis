import numpy as np
import scipy.linalg as sl
import sklearn.preprocessing as spp
import matplotlib.pyplot as plt
from pcaFit import *

'''与书中算出来的不一样，很困惑，不知道哪里错了，看了半天无果，待以后再查'''

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
