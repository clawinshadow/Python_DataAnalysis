import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
import matplotlib.pyplot as plt

# The data is generated from yi ∼ N(x,Σy), where x = [0.5, 0.5] and Σy = 0.1[2, 1; 1, 1]).
mu_x = np.array([0.5, 0.5])
sigma_y = 0.1 * np.array([[2, 1],
                          [1, 1]])
noisy_y = ss.multivariate_normal(mu_x, sigma_y)
samples = noisy_y.rvs(10)

def GraphAttr():
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, -0.5, 0, 0.5, 1])

fig = plt.figure(figsize=(13, 4))
fig.canvas.set_window_title('gaussInferParamsMean2d')

plt.subplot(131)
plt.title('data')
plt.plot(samples[:, 0], samples[:, 1], 'bo')
plt.plot(0.5, 0.5, 'kX', markersize=10)
GraphAttr()

# The prior is p(x) = N(x|0, 0.1I2)
prior_mu = np.array([0, 0])
prior_cov = 0.1 * np.eye(2)
prior = ss.multivariate_normal(prior_mu, prior_cov)
x, y = np.mgrid[-1:1:0.01, -1:1:0.01]
z = np.dstack((x, y))
probs = prior.pdf(z)

plt.subplot(132)
plt.title('prior')
plt.contour(x, y, probs, cmap='jet')
GraphAttr()

# calculate posterior
avg_y = np.average(samples, axis=0).reshape(-1, 1)
N = len(samples)
post_cov = sl.inv(sl.inv(prior_cov) + N * sl.inv(sigma_y))
print('post_cov: \n', post_cov)
post_mu = np.dot(post_cov, np.dot(sl.inv(sigma_y), N * avg_y) + np.dot(sl.inv(prior_cov), prior_mu.reshape(-1, 1)))
print('post_mu: ', post_mu)

post = ss.multivariate_normal(post_mu.ravel(), post_cov)
probs2 = post.pdf(z)

plt.subplot(133)
plt.title('post after 10 obs')
plt.contour(x, y, probs2, cmap='jet')
GraphAttr()

plt.show()
