import numpy as np
import scipy.io as sio
import scipy.linalg as sl
import matplotlib.pyplot as plt
from pfSLD import *
from rbpfSLD import *

'''大概要运行十几秒钟'''

np.random.seed(0)

# load data & params
data = sio.loadmat('rbpfManeuverDemo.mat')
print(data.keys())
x, y = data['x'], data['y']
z, u = data['z'], data['u']
par = data['par'][0][0]
# print(par)
A = par[0]   # continues states matrix， K * K * N, 为什么后面还多了个N，因为还有个discrete的state，共有N个状态
B = par[1]   # noise matrix for continuous states, K * K * N
C = par[2]   # transition matrix from continuous states to obs, D * K * N
D = par[3]   # noise matrix for obs, D * D * N
E = par[4]   # 没什么用处，没有用到
F = par[5]   # input / control vectors, K * 1 * N, ut is a scalar
G = par[6]   # input / control vector for obs, D * 1 * N
T = par[7]   # transition matrix of markov chain for discrete states, specially for SLDS models
pz0 = par[8]
mu0 = par[9] # initial Gaussian mean
S0 = par[10] # initial Gaussian cov
print(A[:, :, 0])

# Fit with Particle Filtering
NSamples = 500
res = pfSLD(NSamples, y, u, A, B, C, D, E, F, G, T)
xest, zest = res[0], res[1]
max_zest = np.argmax(zest, axis=0)
mse_pf = np.mean((xest[0] - x[0])**2) + np.mean((xest[2] - x[2])**2)
titleStr_pf = 'pf, mse {0:.3f}'.format(mse_pf)

# Fit with RBPF
res_rbpf = rbpfSLD(NSamples, y, u, A, B, C, D, E, F, G, T)
xest_rbpf, zest_rbpf = res_rbpf[0], res_rbpf[1]
max_zest_rbpf = np.argmax(zest_rbpf, axis=0)
mse_rbpf = np.mean((xest_rbpf[0] - x[0])**2) + np.mean((xest_rbpf[2] - x[2])**2)
titleStr_rbpf = 'rbpf, mse {0:.3f}'.format(mse_rbpf)

# plot maneuver trajectory data
fig = plt.figure(figsize=(13, 4))
fig.canvas.set_window_title('rbpfManeuverDemo')

def plot(index, xdata, zdata, title):
    ax = plt.subplot(index)
    ax.tick_params(direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title(title)
    plt.axis([-90, 0, -250, 50])
    plt.xticks(np.linspace(-90, 0, 10))
    plt.yticks(np.linspace(-250, 50, 7))
    plt.plot(y[0], y[2], '.', )
    colors = ['b', 'r', 'k']
    symbols = ['o', 'x', '*']
    zs = np.unique(zdata)
    for i in range(len(zs)):
        zi = zs[i]
        idx = zdata == zi
        plt.plot(xdata[0, idx.ravel()], xdata[2, idx.ravel()], marker=symbols[i], color=colors[i], fillstyle='none', linestyle='none')

plot(131, x, z, 'data')
plot(132, xest, max_zest, titleStr_pf)
plot(133, xest_rbpf, max_zest_rbpf, titleStr_rbpf)

plt.tight_layout()
plt.show()

# prepare belief state data
zs = np.unique(z)
N = z.shape[1]
dummyZ = np.zeros((N, len(zs)))
for i in range(N):
    zi = z[:, i]
    idx = (int)(np.flatnonzero(zs == zi)[0])
    dummyZ[i, idx] = 1

zest_labels = max_zest + 1    # (0, 1, 2) -> (1, 2, 3)
diffs = np.count_nonzero(z != zest_labels)
err = diffs / N
titleStr_pf_z = 'pf, error rate {0:.3f}'.format(err)

zest_labels_rbpf = max_zest_rbpf + 1
err_rbpf = np.count_nonzero(z != zest_labels_rbpf) / N
titleStr_rbpf_z = 'rbpf, error rate {0:.3f}'.format(err_rbpf)

# plot belief state
fig = plt.figure()
fig.canvas.set_window_title('rbpfManeuverBelZ')

def plotZ(index, title, data, isInterpolation=False):
    ax1 = plt.subplot(index)
    ax1.tick_params(bottom=False, left=False)  # hide tick spines
    plt.title(title)
    plt.xticks(np.linspace(1, 3, 3))
    plt.yticks(np.linspace(100, 20, 5))
    if isInterpolation:
        plt.imshow(data, cmap='jet', aspect='auto', extent=[1, 3, 100, 0], interpolation='nearest')
    else:
        plt.imshow(data, cmap='jet', aspect='auto', extent=[1, 3, 100, 0])

plotZ(131, 'truth', dummyZ)
plotZ(132, titleStr_pf_z, zest.T, False)
plotZ(133, titleStr_rbpf_z, zest_rbpf.T, True)

plt.tight_layout()
plt.show()
