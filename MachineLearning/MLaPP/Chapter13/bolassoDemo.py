import numpy as np
import scipy.io as sio
import sklearn.linear_model as slm
import matplotlib.pyplot as plt
from GenerateDataForBoLasso import *

'''
different with figure in book, need further check.
'''

np.random.seed(0)

# prepare data
n = 1000
d = 16
r = 8
nDS = 256
nDS_2 = 128
X, y, w = bolassoMakeData(n, d, r, nDS, True)       # consistent data
X2, y2, w2 = bolassoMakeData(n, d, r, nDS_2, False) # inconsistent data

lambdas = np.logspace(-15, 0, 76, base=np.e)[::-1]

# Fit with boLasso
def boLassoFit(X_ds, y_ds, l):
    n_ds, n, d = X_ds.shape
    P = np.zeros(n_ds)
    S = np.zeros((n_ds, d))
    for i in range(n_ds):
        X = X_ds[i]
        y = y_ds[i]
        lasso = slm.Lasso(alpha=l, fit_intercept=True).fit(X, y)
        w = lasso.coef_.ravel()
        S[i] = np.array(w != 0, dtype='int8')  # calculate w == 0
    P = np.count_nonzero(S, axis=0) / n_ds
    print(P)
    return P

def Fit(X_ds, y_ds, lambdas):
    n_lambdas = len(lambdas)
    n_ds, n, d = X_ds.shape
    Probs = np.zeros((n_lambdas, d))
    for i in range(n_lambdas):
        l = lambdas[i]
        Probs[i] = boLassoFit(X_ds, y_ds, l)
        print('lambda {0} ended: '.format(i))

    return Probs

probs = Fit(X, y, lambdas)
probs_2 = Fit(X2, y2, lambdas)
consistent_dict = {'consistent': probs}
inconsistent_dict = {'inconsistent': probs_2}
sio.savemat('consistent_dict.mat', consistent_dict)
sio.savemat('inconsistent_dict.mat', inconsistent_dict)

# plots
dataDict = sio.loadmat('consistent_dict.mat')
data = dataDict['consistent']
print('data.shape: ', data.shape)
dataDict2 = sio.loadmat('inconsistent_dict.mat')
data2 = dataDict2['inconsistent']

fig = plt.figure(figsize=(13, 4))
fig.canvas.set_window_title('bolassoDemo')

plt.subplot(131)
#plt.axis([0, 15, 16, 0])
plt.title('lasso on sign inconsistent data')
plt.xlabel(r'$-log(\lambda)$')
plt.ylabel('variable index')
plt.xticks(np.linspace(0, 15, 4))
plt.yticks(np.linspace(0, 16, 9))
plt.imshow(data.T, cmap='gray', aspect='auto', extent=[0, 15, 16, 0])   # 这个参数很重要aspect='auto'
plt.colorbar(ticks=np.linspace(0, 1, 11))

plt.subplot(132)
plt.title('bolasso on sign inconsistent data 128 bootstraps')
plt.xlabel(r'$-log(\lambda)$')
plt.ylabel('variable index')
plt.xticks(np.linspace(0, 15, 4))
plt.yticks(np.linspace(0, 16, 9))
plt.imshow(data2.T, cmap='gray', aspect='auto', extent=[0, 15, 16, 0])  # 这个参数很重要aspect='auto'
plt.colorbar(ticks=np.linspace(0, 1, 11))

plt.tight_layout()
plt.show()
    
    
