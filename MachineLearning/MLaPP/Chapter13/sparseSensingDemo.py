import numpy as np
import scipy.linalg as sl
import sklearn.linear_model as slm
import matplotlib.pyplot as plt

np.random.seed(0)

# generate data
N = 2**10  # number of observataions
D = 2**12  # dimension of data
n_spikes= 160 # -1 or +1
w = np.zeros(D)
w[:n_spikes] = np.random.choice([-1, 1], size=n_spikes)
np.random.shuffle(w)
w = w.reshape(-1, 1)

X = np.random.randn(N, D)
# orthonormalize rows, 为什么不是正交化列，可能是因为D > N的关系，如果直接orth(X)会返回D*D的矩阵
X = sl.orth(X.T).T

sigma = 0.01
y = np.dot(X, w) + sigma * np.random.randn(N, 1)  # N * 1, add noise
print(y.shape)

# Fit with lasso
max_lambda = np.max(np.abs(np.dot(X.T, y)))
tau = 0.1 * max_lambda / (2 * N)  # lambda 一定要rescale后才能传入sklearn.Lasso
print('tau: ', tau)
lasso = slm.Lasso(alpha=tau).fit(X, y)
print('w by l1: ', lasso.coef_)
w_l1 = lasso.coef_
print('length of w: {0} nonzero count: {1} '.format(len(w_l1), np.count_nonzero(w_l1)))
