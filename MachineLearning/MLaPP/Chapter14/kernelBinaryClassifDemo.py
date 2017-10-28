import h5py
import numpy as np
import sklearn.linear_model as slm
import sklearn.metrics.pairwise as smp
import matplotlib.pyplot as plt
from plotDecisionBoundary import *

# prepare data
# this is as matlab v7.3 data, should install 'h5py' package to read it
with h5py.File('bishop2class.mat', 'r') as data:
    print('keys of data: ', list(data.keys()))
    X = np.array(data['X']).T
    y = np.array(data['Y']).T

print(X.shape, y.shape)

# don't need to standardize, just kernelise is enough
rbf_scale = 0.3
gamma = 1 / (2 * rbf_scale**2)
x_train = smp.rbf_kernel(X, gamma=gamma)     # kernelise
x_train = x_train / np.sqrt(2 * np.pi * rbf_scale**2)

# Fit with L2VM
lambda_ = 5
C = 1 / (2 * lambda_)  # convert lambda_ in matlab code to C in sklearn.logisicRegression
y = y.ravel()
LogReg = slm.LogisticRegression(penalty='l2', C=C, tol=1e-03, solver='lbfgs', max_iter=200, fit_intercept=False).fit(x_train, y)
print(LogReg.coef_)  # exactly the same as in matlab codes
y_predict_L2 = LogReg.predict(x_train)
nerr_l2 = np.count_nonzero(y_predict_L2 == y)
print('nerr of L2VM: ', nerr_l2)

# plots
fig = plt.figure(figsize=(9, 8))
fig.canvas.set_window_title('kernelBinaryClassifDemo')

args = dict(subplotIndex=221, title='logregL2, nerr='+str(nerr_l2), tickRange=[-2, 3, -3, 3],\
            centers=X, rbfscale=rbf_scale)
plot(X, y, LogReg, smp.rbf_kernel, **args)

plt.tight_layout()
plt.show()

