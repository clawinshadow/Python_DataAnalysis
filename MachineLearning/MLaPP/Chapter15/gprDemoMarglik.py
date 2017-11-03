import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from GPR_Algorithm import *

# prepare data
data = sio.loadmat('gprDemoMarglik.mat')
print(data.keys())
x = data['xs']
y = data['fs']
print(x.shape, y.shape)

N = 41
l_space = np.linspace(np.log(0.1), np.log(80), N)
s_space = np.linspace(np.log(0.03), np.log(3), N)  # log-space
xx, yy = np.exp(np.meshgrid(l_space, s_space))
params = np.c_[xx.ravel(), yy.ravel()]

zz = np.zeros(len(params))
for i in range(len(params)):
    l, sigma = params[i][0], params[i][1]
    K = SE(x, x, 1, l, sigma)
    LL = fit_gpr(K, None, None, y)
    zz[i] = LL
zz[zz < -100] = -100
zz = zz.reshape(xx.shape)

# plots
fig = plt.figure(figsize=(13, 4))
fig.canvas.set_window_title('gprDemoMarglik')

plt.subplot(131)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('characteristic lengthscale')
plt.ylabel('noise standard deviation')
levels = -1 * np.array([8.3, 8.5, 8.9, 9.3, 9.8, 11.5, 15])
plt.contour(xx, yy, zz, levels=levels[::-1], cmap='jet')
plt.plot([1, 10], [0.2, 0.8], 'b+', ms=10, linestyle='none')

def plot(index, l, sigma):
    xtest = np.linspace(-7.5, 7.5, 141).reshape(-1, 1)
    K = SE(x, x, 1, l, sigma)
    Ks = SE(x, xtest, 1, l, sigma)
    Kss = SE(xtest, xtest, 1, l, sigma)
    mu, cov, ll = fit_gpr(K, Ks, Kss, y)
    mu = mu.ravel()
    cov = cov - sigma ** 2  # remove observation noise
    std = np.sqrt(np.diag(cov))

    plt.subplot(index)
    plt.axis([-7.5, 7.5, -2, 2.5])
    plt.xticks(np.linspace(-5, 5, 3))
    plt.yticks(np.linspace(-2, 2, 5))
    plt.xlabel('input, x')
    plt.ylabel('output, y')
    plt.plot(x.ravel(), y.ravel(), 'k+', linestyle='none', ms=10)
    plt.plot(xtest, mu, 'k-', lw=2)
    plt.fill_between(xtest.ravel(), mu - 2 * std, mu + 2 * std, color='gray', alpha=0.5)

plot(132, 1, np.sqrt(0.2))
plot(133, 10, np.sqrt(0.8))

plt.tight_layout()
plt.show()
