import numpy as np
import scipy.io as sio
import sklearn.linear_model as slm
import matplotlib.pyplot as plt
from kalmanSmooth import *

# load data
data = sio.loadmat('linregOnlineDemoKalman.mat')
print(data.keys())
xtrain = data['xtrain']
ytrain = data['ytrain']
print(xtrain.shape, ytrain.shape)
N = len(xtrain)
xx = np.c_[np.ones(N), xtrain]

# fit with OLS
ols = slm.LinearRegression(False).fit(xx, ytrain.ravel())
print(ols.coef_)

# fit with online kalman, one pass through data will converge
w = np.zeros((N + 1, 2))
V = 10 * np.eye(2)
stderr = np.zeros((N + 1, 2))
stderr[0] = np.array([np.sqrt(V[0, 0]), np.sqrt(V[1, 1])])
sigma2 = 1
for i in range(N):
    A = np.eye(2)
    C = xx[i].reshape(1, -1)
    Q = np.zeros((2, 2))
    R = sigma2
    y = ytrain[i]
    mus, covs = kalman_smoothing(y, A, C, Q, R, w[i], V)
    V = covs[0]
    w[i + 1] = mus[0]
    stderr[i + 1] = np.array([np.sqrt(V[0, 0]), np.sqrt(V[1, 1])])

print('trace of w: \n', w)

xs = np.linspace(1, N + 1, N + 1)

# plots
fig = plt.figure()
fig.canvas.set_window_title('linregOnlineDemoKalman')

ax = plt.subplot()
ax.tick_params(direction='in')
plt.axis([0, 25, -8, 4])
plt.xticks(np.linspace(0, 25, 6))
plt.yticks(np.linspace(-8, 4, 7))
plt.xlabel('time')
plt.ylabel('weights')
plt.title('online linear regression')
plt.plot(xs, w[:, 0], 'ko-', fillstyle='none', lw=1, label='w0')
plt.plot(xs, w[:, 1], 'r*-', fillstyle='none', lw=1, label='w1')
# fmt='none' means only plot the errorbars, exclude the lines connecting points
plt.errorbar(xs, w[:, 0], yerr=stderr[:, 0], ecolor='k', fmt='none', elinewidth=1, capsize=3, label='w0 batch')
plt.errorbar(xs, w[:, 1], yerr=stderr[:, 1], ecolor='r', fmt='none', elinewidth=1, capsize=3, label='w1 batch')
plt.plot(xs, ols.coef_[0] * np.ones(len(xs)), 'k-', lw=2)
plt.plot(xs, ols.coef_[1] * np.ones(len(xs)), 'r:', lw=2)
plt.legend()

plt.tight_layout()
plt.show()