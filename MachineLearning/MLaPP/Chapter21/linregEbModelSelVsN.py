import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from linregFitEmpiricalBayes import *
from linregFitVariationalBayes import *

'''
这个图画的是model selection，是用哪一个degree来对x进行多项式扩展
'''

def polyExpansion(xbase, deg):
    assert deg > 0
    N, D = xbase.shape
    X = np.ones((N, deg * D + 1))
    for i in range(deg):
        X[:, i:(i + 1) * D] = xbase ** (i + 1)

    return X

# load data
data = sio.loadmat('linregEbModelSelVsN.mat')     # 30 points
data2 = sio.loadmat('linregEbModelSelVsN2.mat')   # 5 points
print(data.keys())
x_base = data['x1d']
ytrain = data['ytrain']
x_base2 = data2['x1d']
ytrain2 = data2['ytrain']
print(x_base2)
print(ytrain2)

# Fit with EB & VB,
degs = [1, 2, 3]
PPs = []
prob_ebs = np.zeros(len(degs))  # N = 30, EB, P(D|m)
prob_vbs = np.zeros(len(degs))  # N = 30, VB
prob_ebs2 = np.zeros(len(degs)) # N = 5, EB
prob_vbs2 = np.zeros(len(degs)) # N = 5, VB
for i in range(len(degs)):
    # N = 5
    xtrain2 = polyExpansion(x_base2, degs[i])
    res_eb2= linregFitEB(xtrain2, ytrain2)
    res_vb2 = linregFitVB(xtrain2, ytrain2)
    prob_ebs2[i] = (np.exp(res_eb2[0]))
    prob_vbs2[i] = (np.exp(res_vb2[-1]))

    # N = 30
    xtrain = polyExpansion(x_base, degs[i])
    res_eb = linregFitEB(xtrain, ytrain)
    res_vb = linregFitVB(xtrain, ytrain)
    prob_ebs[i] = (np.exp(res_eb[0]))
    prob_vbs[i] = (np.exp(res_vb[-1]))

# P(m|D) = P(m) * p(D|m) / P(D), p(m) is an uniform, so ignore it
post_ebs = prob_ebs / np.sum(prob_ebs)
post_vbs = prob_vbs / np.sum(prob_vbs)
post_ebs2 = prob_ebs2 / np.sum(prob_ebs2)
post_vbs2 = prob_vbs2 / np.sum(prob_vbs2)

# plots
fig = plt.figure(figsize=(10, 9))
fig.canvas.set_window_title('linregEbModelSelVsN')

def plot(index, title, probs):
    ax = plt.subplot(index)
    ax.tick_params(direction='in')
    plt.title(title)
    plt.axis([-0.5, 4, 0, 1])
    plt.xticks(np.linspace(0, 3, 4))
    plt.yticks(np.linspace(0, 1, 6))
    plt.xlabel('M')
    plt.ylabel('P(M|D)')
    print(probs)
    plt.bar(np.linspace(1, 3, 3), probs, color='midnightblue', edgecolor='none', align='center')

plot(221, 'N=5, method=VB', post_vbs2)
plot(222, 'n=5, method=EB', post_ebs2)
plot(223, 'N=30, method=VB', post_vbs)
plot(224, 'n=30, method=EB', post_ebs)

plt.tight_layout()
plt.show()