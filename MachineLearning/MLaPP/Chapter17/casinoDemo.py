import numpy as np
import scipy.io as sio
import scipy.linalg as sl
import matplotlib.pyplot as plt
from hmmFilter import *
from hmmFwdBack import *

# load data
data = sio.loadmat('casinoDemo.mat')
print(data.keys())
hidden = data['hidden']     # 1 * 300
observed = data['observed'].ravel()
transMat = data['transmat']
obsModel = data['obsModel']
pi = data['pi'].ravel()
print(np.unique(hidden), np.unique(observed))
print('observe mdoel: \n', obsModel)
print('initial pi: ', pi)
N = hidden.shape[1]
x = np.linspace(1, N, N)
gray = hidden == 2

# Fit with Filter, Smoothing and Viterbi
Z1 = hmm_filter(observed, transMat, obsModel, pi)
Z2 = hmm_smoothing(observed, transMat, obsModel, pi)

# plots
fig = plt.figure(figsize=(13, 4))
fig.canvas.set_window_title('casinoDemo')

def plot(index, title, data):
    ax = plt.subplot(index)
    ax.tick_params(direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title(title)
    plt.xlabel('roll number')
    plt.ylabel('p(loaded)')
    plt.axis([0, 300, 0, 1])
    plt.xticks(np.linspace(0, 300, 7))
    plt.yticks(np.linspace(0, 1, 3))
    plt.vlines(x[gray.ravel()], 0, 1, colors='silver')
    plt.plot(x, data[:, 1], color='midnightblue', lw=2)

plot(131, 'filtered', Z1)
plot(132, 'smoothed', Z2)

plt.tight_layout()
plt.show()
