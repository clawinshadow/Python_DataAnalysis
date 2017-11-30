import numpy as np
import scipy.io as sio
import scipy.stats as ss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''simulated annealing, 模拟退火法，既可用于寻找全局最优解，也可用于multi-modal分布的采样'''

np.random.seed(2)

# load data
data = sio.loadmat('peaks.mat')
print(data.keys())
XX = data['XX']
YY = data['YY']
Z = data['Z']
m = np.min(Z)
Zpos = Z + np.abs(m) + 1

# plot surface at different temperatures
fig = plt.figure(figsize=(10, 9))
fig.canvas.set_window_title('saDemoPeaks_1')

temps = [2, 1.5, 1, 0.2]
for i in range(len(temps)):
    t = temps[i]
    Zt = Zpos**(1/t)   # p(x) ∝ exp(−f(x)/T) = exp(-f(x))^(1/t)
    titleStr = 'temp {0:.3f}'.format(t)
    # draw 3d surface plot
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    ax.set_title(titleStr)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 60)
    if t >= 1:
        ax.set_zlim(0, 10)
    ax.set_xticks(np.linspace(0, 50, 6))
    ax.set_yticks(np.linspace(0, 60, 4))
    ax.plot_surface(XX, YY, Zt, cmap='jet', edgecolors='k')

plt.tight_layout()

# simulated annealing
def target(x):
    neg_peaks = data['Z'] * -1
    row = (int)(np.round(x[0]))
    col = (int)(np.round(x[1]))
    if row >= 0 and row < neg_peaks.shape[0] and col >= 0 and col < neg_peaks.shape[1]:
        p = neg_peaks[row, col]
    else:
        p = np.inf  # invalid

    return p

NSamples = 1000
xinit = [34, 24]
T = 1.0  # initial temperature
cov_prop = 4 * np.eye(2)
samples = np.zeros((NSamples, len(xinit)))
temps = np.zeros(NSamples)
energies = np.zeros(NSamples)
for i in range(NSamples):
    if i == 0:
        xprev = xinit
    else:
        xprev = samples[i - 1]

    xnext = ss.multivariate_normal.rvs(mean=xprev, cov=cov_prop, size=1)
    e_old = target(xprev)
    e_new = target(xnext)
    if e_new == np.inf:
        alpha = 0
    else:
        alpha = np.exp((e_old - e_new) / T)

    r = np.min([1, alpha])
    u = np.random.rand()
    if u < r:
        samples[i] = xnext
        energies[i] = e_new
    else:
        samples[i] = xprev
        energies[i] = e_old

    T = 0.995 * T     # T = const * T0^k, decrease temperature
    temps[i] = T

# plots
fig2 = plt.figure(figsize=(10, 9))
fig2.canvas.set_window_title('saDemoPeaks_2')

ax1 = plt.subplot(221)
ax1.tick_params(direction='in')
plt.axis([0, 1200, 0, 1])
plt.xticks(np.linspace(0, 1200, 7))
plt.yticks(np.linspace(0, 1, 11))
plt.title('temperature vs iteration')
plt.plot(np.linspace(1, NSamples, NSamples), temps, '-', lw=1)

ax2 = plt.subplot(222)
ax2.tick_params(direction='in')
plt.xlim(0, 1000)
plt.xticks(np.linspace(0, 1000, 6))
plt.title('energy vs iteration')
plt.plot(np.linspace(1, NSamples, NSamples), energies, '-', lw=1)

'''
# matplotlib中并没有原生的histogram3d方法，自己写的这个bar3d不仅没法使用jet的cmap，并且画起来特别慢。。100个bins都得花几十秒
# 还是matlab中的hist3方法高效的多，也漂亮的多
def histogram3d(index, data, titleStr):
    ax = fig2.add_subplot(2, 2, index, projection='3d')
    ax.set_title(titleStr)
    hist, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=100)

    xpos, ypos = np.meshgrid(xedges[:-1] + 0.5, yedges[:-1] + 0.5)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)

    dx = 1 * np.ones_like(zpos)
    dy = dx.copy()
    dz = hist.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, cmap='jet')

sample1 = samples[:550]
sample2 = samples
titleStr1 = 'iter {0}, temp {1:.3f}'.format(550, temps[549])
titleStr2 = 'iter {0}, temp {1:.3f}'.format(1000, temps[-1])
histogram3d(3, sample1, titleStr1)
histogram3d(4, sample2, titleStr2)
'''

plt.tight_layout()
plt.show()
