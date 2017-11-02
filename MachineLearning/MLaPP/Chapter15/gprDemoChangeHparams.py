import numpy as np
import scipy.io as sio
import scipy.linalg as sl
import sklearn.metrics.pairwise as smp
import matplotlib.pyplot as plt
from GPR_Algorithm import *

# squared-exponential (SE) kernel for the noisy observations
def SE(X, Y, v_scale, h_scale, noise_sigma):
    N, D = X.shape
    gamma = 1 / (2 * h_scale**2)
    gram = (v_scale**2) * smp.rbf_kernel(X, Y, gamma=gamma)
    if len(X) == len(Y):
        gram += (noise_sigma**2) * np.eye(N)

    return gram

# prepare data
data = sio.loadmat('gprDemoChangeHparams.mat')
print(data.keys())
x = data['x']
y = data['y']

hparams = np.array([[1, 1, 0.1],
                    [0.3, 1.08, 5e-5],
                    [3, 1.16, 0.89]])
xtest = np.linspace(-7.5, 7.5, 201)

# Fit with GPR
fig = plt.figure(figsize=(13, 4))
fig.canvas.set_window_title('gprDemoChangeHparams')

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
xtest = xtest.reshape(-1, 1)
for i in range(len(hparams)):
    params = hparams[i]
    K = SE(x, x, params[1], params[0], params[2])
    Ks = SE(x, xtest, params[1], params[0], params[2])
    Kss = SE(xtest, xtest, params[1], params[0], params[2])
    mu, var, ll = fit_gpr(K, Ks, Kss, y)
    mu = mu.ravel()
    var = var - params[2] ** 2      # remove observation noise
    std = np.sqrt(np.diag(var))
    print(mu.shape, std.shape)

    plt.subplot((int)('13' + str(i + 1)))
    plt.title(r'$(l, \sigma_f, \sigma_y): ({0}, {1}, {2})$'.format(params[0], params[1], params[2]))
    plt.axis([-8, 8, -3, 3])
    plt.xticks(np.linspace(-8, 8, 9))
    plt.yticks(np.linspace(-3, 3, 7))
    plt.plot(x.ravel(), y.ravel(), 'k+', linestyle='none', ms=17)
    plt.plot(xtest.ravel(), mu, 'k-', lw=2)
    plt.fill_between(xtest.ravel(), mu - 2 * std, mu + 2 * std, edgecolors='gray', color='gray', alpha=0.5)

plt.tight_layout()
plt.show()