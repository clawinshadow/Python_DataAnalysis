import numpy as np
import scipy.io as sio
import scipy.linalg as sl
import scipy.stats as ss
import sklearn.linear_model as slm
import matplotlib.pyplot as plt

np.random.seed(0)

def logistic(x):
    return 1 / (1 + np.exp(-x))

def loglikelihood(w, X, y):
    w = w.reshape(-1, 1)
    y = y.reshape(-1, 1)
    mu = logistic(np.dot(X, w))
    p = np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))

    return p

# load sat data
data = sio.loadmat('sat.mat')
print(data.keys())
sat = data['sat']
# print(sat)  # 3, 4 column is X, the first column is target labels
X = sat[:, 2:4]
y = sat[:, 0]

C = 1e-8
LGR = slm.LogisticRegression(C=1/C, fit_intercept=False, tol=1e-6).fit(X, y)
w = LGR.coef_.ravel()
print(w)

# Get D= diag(μi(1 − μi)) and μi = sigmoid(wˆT * x)
N, D = X.shape
S = np.zeros((N, N))
for i in range(N):
    xi = X[i]
    mu = logistic(np.sum(w * xi))
    S[i, i] = mu * (1 - mu)

# Get hessian matrix
invV0 = C * np.eye(D)
V0 = sl.inv(invV0)
H = invV0 + np.dot(X.T, np.dot(S, X))
invH = sl.inv(H)
print(invH)

# Get covariance of proposal distribution, refer to equation 24.53
cov_proposal = 2.38**2 * invH / D

# sampling with MH algorithm
NSamples = 5000
xinit = w
samples = np.zeros((NSamples, D))
for i in range(NSamples):
    if i == 0:
        x_prev = xinit
    else:
        x_prev = samples[i - 1]

    x_next = ss.multivariate_normal.rvs(mean=x_prev, cov=cov_proposal, size=1)
    alpha = np.exp(loglikelihood(x_next, X, y) - loglikelihood(x_prev, X, y))
    r = np.min([1, alpha])
    u = np.random.rand()
    if u < r:
        samples[i] = x_next
    else:
        samples[i] = x_prev

# plots
fig = plt.figure(figsize=(13, 4))
fig.canvas.set_window_title('logregSatMhDemo')

ax1 = plt.subplot(131)
ax1.tick_params(direction='in')
plt.axis([-120, 0, 0, 0.2])
plt.xticks(np.linspace(-120, 0, 7))
plt.yticks(np.linspace(0, 0.2, 11))
plt.xlabel('w0')
plt.ylabel('w1')
plt.plot(samples[:, 0], samples[:, 1], '.', ms=1, linestyle='none')

ax2 = plt.subplot(132)
ax2.tick_params(direction='in')
plt.axis([-120, 0, 0, 1600])
plt.xticks(np.linspace(-120, 0, 7))
plt.yticks(np.linspace(0, 1500, 4))
plt.title('w0 intercept')
plt.hist(samples[:, 0], 10, edgecolor='k')

ax2 = plt.subplot(133)
ax2.tick_params(direction='in')
plt.axis([0, 0.2, 0, 1700])
plt.xticks(np.linspace(0, 0.2, 5))
plt.yticks(np.linspace(0, 1500, 4))
plt.title('w1 slope')
plt.hist(samples[:, 1], 10, edgecolor='k')

plt.tight_layout()
plt.show()