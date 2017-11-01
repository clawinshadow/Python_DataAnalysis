import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
import sklearn.metrics.pairwise as smp
import sklearn.gaussian_process as sgp
import sklearn.gaussian_process.kernels as kernels
import matplotlib.pyplot as plt

'''
GPR 里面的 predict() 方法要慎用，它返回的不是posterior mean..
'''

np.random.seed(0)

# GP prior, only need a mu(x) and a kernel func K(x, x)
xtest = np.linspace(-5, 5, 101).reshape(-1, 1)
N, D = xtest.shape
mu = np.zeros(N)
sigma = smp.rbf_kernel(xtest, gamma=0.5)
# if using this sigma directly, then it will be a singular matrix, invalid for a cov matrix
sigma = sigma + 1e-8 * np.eye(N)  # for numerical stability
prior_samples = ss.multivariate_normal(mu, sigma).rvs(3)
print(prior_samples.shape)

# Fit with GP
xtrain = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
ytrain = np.sin(xtrain)

alpha = 1e-8  # for numerical stability
K = smp.rbf_kernel(xtrain, gamma=0.5) + alpha * np.eye(xtrain.shape[0])        # 5 * 5
Ks = smp.rbf_kernel(xtrain, xtest, gamma=0.5)    # 5 * 101
print(Ks.shape)
Kss = smp.rbf_kernel(xtest, xtest, gamma=0.5)    # 101 * 101
post_sigma = Kss - np.dot(Ks.T, np.dot(sl.inv(K), Ks))
post_mu = np.dot(Ks.T, np.dot(sl.inv(K), ytrain.reshape(-1, 1))).ravel()  # 101 * 1
print('post_mu by self: ', post_mu)  # same with in matlab codes
post_samples = ss.multivariate_normal(post_mu, post_sigma, allow_singular=True).rvs(3)
std = np.sqrt(np.diag(post_sigma))

# Fit with sklearn.GP
kernel = 1.0 * kernels.RBF(length_scale=1.0) + 1.0 * kernels.WhiteKernel(noise_level=alpha)
GP = sgp.GaussianProcessRegressor(kernel=kernel, alpha=0.0).fit(xtrain, ytrain)
y_predict = GP.predict(xtest)  # be careful to use predict..
print('post_mu by sklearn.GP: ', y_predict.ravel())

# plot prior
fig = plt.figure(figsize=(12, 5))
fig.canvas.set_window_title('gprDemoNoiseFree')

ax = plt.subplot(121)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.axis([-5, 5, -3, 3])
plt.xticks(np.linspace(-5, 5, 3))
plt.yticks(np.linspace(-3, 3, 13))
plt.plot(xtest.ravel(), prior_samples.T, 'k-')

# plot posterior
ax = plt.subplot(122)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.axis([-5, 5, -2, 2.5])
plt.xticks(np.linspace(-5, 5, 3))
plt.yticks(np.linspace(-2, 2.5, 10))
plt.plot(xtest.ravel(), post_samples.T, 'k-')
plt.plot(xtrain.ravel(), ytrain, 'rx', ms=8, mew=2, linestyle='none')
plt.fill_between(xtest.ravel(), post_mu - 2 * std, post_mu + 2 * std, color='gray')

plt.tight_layout()
plt.show()