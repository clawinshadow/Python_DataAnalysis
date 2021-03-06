import numpy as np
import scipy.io as sio
import sklearn.preprocessing as sp
import sklearn.metrics as sm
import sklearn.linear_model as slm
import matplotlib.pyplot as plt
from subsets import *

# prepare data
data = sio.loadmat('prostate.mat')
print(data.keys())
print(data['X'].shape)
print(data['y'].shape)

X, y = data['X'], data['y']
X = sp.StandardScaler().fit_transform(X)
y = sp.StandardScaler(with_std=False).fit_transform(y)

# Fit models use Orthogonal Least Squares
N, D = X.shape
models = subsets(D)
mses = dict()
min_mses = np.zeros(len(models))
for k, v in models.items():
    ms = []
    if k == 0:
        y_predict = np.zeros((N, 1))
        mse = sm.mean_squared_error(y, y_predict)
        ms.append(mse)
    else:
        for i in v:
            cols = np.array(i)  # convert tuple to ndarray, for indexing
            X_train = X[:, cols]
            LR = slm.LinearRegression(fit_intercept=False).fit(X_train, y)
            y_predict = LR.predict(X_train)
            mse = sm.mean_squared_error(y, y_predict)
            ms.append(mse)

    mses[k] = ms
    min_mses[k] = np.min(ms)

print(min_mses)

# plots
fig = plt.figure(figsize=(8, 6))
fig.canvas.set_window_title('prostateSubsets')

plt.subplot()
plt.title('all subsets on prostate cancer')
plt.xlabel('subset size')
plt.ylabel('training set error')
plt.axis([0, 8, 0.4, 1.4])
plt.xticks(np.linspace(0, 8, 9))
plt.yticks(np.linspace(0.4, 1.4, 6))
for k, v in mses.items():
    count = len(v)
    xs = np.tile(k, count)
    plt.plot(xs, v, color='midnightblue', marker='o', ms=4, mec='none', linestyle='none')

plt.plot(list(models.keys()), min_mses, 'ro-', mec='r', lw=2)

plt.show()
