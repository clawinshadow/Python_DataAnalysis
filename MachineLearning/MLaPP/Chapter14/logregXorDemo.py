import numpy as np
import scipy.io as sio
import sklearn.linear_model as slm
import sklearn.preprocessing as sp
import sklearn.metrics.pairwise as smp
import matplotlib.pyplot as plt
from plotDecisionBoundary import *

'''slightly different with in books, but it doesn't matter'''

# generate data
def poly(X, deg=10):
    '''
    这个是多元的X的指数扩展，与之前一维的X是不一样的，与sklearn里面的PolynomialFeatures也不一样
    以如下X为例，deg=2
       1  0            1  0  1  0
    X: 2  4   -> φ(X): 2  4  4  16
       3  5            3  5  9  25 
    '''
    N, D = X.shape
    result = np.zeros((N, D * deg))
    for i in range(deg):
        result[:, i*D : (i+1)*D] = X**(i + 1)

    return result
    
data = sio.loadmat('XOR.mat')
X = data['X']
y = data['y'].ravel()
print(X.shape, y.shape)

# Fit with poly 10 expaned logistic regression
scaler = sp.MinMaxScaler(feature_range=(-1, 1)).fit(X) # because number is too large, so rescale to (-1, 1)
X_scaled = scaler.transform(X)
X_train = poly(X_scaled) # polynomial expansion with deg=10
LG = slm.LogisticRegression(C=50, tol=1e-03, solver='lbfgs', max_iter=200, fit_intercept=True).fit(X_train, y)
print('w by poly expansion: ', LG.coef_)  # 误差在千分之一以内

# Fit with 4-centroids kernelised model with scale=1 RBF kernel
N, D = X.shape
sscaler = sp.StandardScaler().fit(X)
x_standard = sscaler.transform(X)
# sigh.. spend 2 hours to figure out that variance in StandardScaler is divided by (N - 1)
# while in matlab codes, it's divided by N
factor = np.sqrt((N - 1) / N)
x_standard = x_standard * factor  # keep the same as in matlab codes
print(np.var(x_standard, axis=0, ddof=1))        # ddof = 1 means biased variance, should be 1

rbf_scale = 1
gamma = 1 / (2 * rbf_scale**2)
mus = np.array([[1, 1], [1, 5], [5, 1], [5, 5]])
centers = sp.StandardScaler().fit_transform(mus)
centers = centers * np.sqrt(3/4)
x_train2 = smp.rbf_kernel(x_standard, centers, gamma=gamma)     # kernelise
x_train2 = x_train2 / np.sqrt(2 * np.pi * rbf_scale**2)

LG2 = slm.LogisticRegression(C=50, tol=1e-03, solver='lbfgs', max_iter=200, fit_intercept=True).fit(x_train2, y)
y_predict2 = LG2.predict(x_train2)
print('w by kernel with 4 centroids: ', LG2.coef_)

# plots
fig = plt.figure(figsize=(10.5, 5))
fig.canvas.set_window_title('logregXorDemo')

args1 = dict(subplotIndex=121, deg=10, title='poly10', tickRange=[0, 6, -1, 7], scaler=scaler)
plot(X, y, LG, poly, **args1)
args2 = dict(title='rbf prototypes', tickRange=[0, 6, -1, 7], sscaler=sscaler, factor=factor, \
            centers=centers, rbfscale=rbf_scale, subplotIndex=122)
plot(X, y, LG2, smp.rbf_kernel, **args2)

plt.tight_layout()
plt.show()
