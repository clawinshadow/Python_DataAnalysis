import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

'''并没有真正的采样，就是画个示例图而已，就当是练习画图了'''

# gaussian parameters
mu = np.array([0, 0])
cov = np.array([[1, 0.99],
                [0.99, 1]])
gauss = ss.multivariate_normal(mu, cov)

xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
xs = np.c_[xx.ravel(), yy.ravel()]
zz = gauss.pdf(xs).reshape(xx.shape)
level = zz.min() + 0.05 * (zz.max() - zz.min())

fig = plt.figure()
fig.canvas.set_window_title('gibbsGaussDemo')

ax = plt.subplot()
ax.tick_params(direction='in')
plt.axis([-3, 3, -3, 3])
plt.xticks(np.linspace(-3, 3, 7))
plt.yticks(np.linspace(-3, 3, 7))
plt.contour(xx, yy, zz, levels=[level], colors='r', linewidths=0.5)
plt.plot(mu[0], mu[1], 'rx', ms=15, mew=0.5)

# dummy sample points
blueSeq = np.array([
                       [-1 / 2, -1],
                       [-1 / 2, 0],
                       [1, 0],
                       [1, 1],
                       [-1 / 2, 1],
                       [-1 / 2, 1 / 2],
                       [1.5, 1 / 2],
                       [1.5, 1.5],
                   ]) / 3

for i in range(len(blueSeq) - 1):
    point1 = blueSeq[i]
    point2 = blueSeq[i + 1]
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'b-', lw=0.5)

# conditional gaussian
xs = np.linspace(-2, 2, 401)
x2obs = 0.7
mu_x1_cond = mu[0] + (cov[0, 1] / cov[1, 1]) * (x2obs - mu[1])  # mean of p(x1|x2=0.7)
cov_x1_cond = cov[0, 0] - cov[0, 1]**2 / cov[1, 1]              # covariance of p(x1|x2=0.7)

gauss_cond = ss.norm(mu_x1_cond, np.sqrt(cov_x1_cond))
ys = gauss_cond.pdf(xs)
ys = ys * 0.3 - 3   # for plot visuality
plt.plot(xs, ys, 'g-', lw=0.5)

# arrows and notations
plt.plot([-2.5, 2.5], [1.5, 1.5], 'k-', lw=1);
plt.plot([-2.5, -2.3], [1.5, 1.6], 'k-', lw=1);
plt.plot([-2.5, -2.3], [1.5, 1.4], 'k-', lw=1);
plt.plot([2.5, 2.3], [1.5, 1.6], 'k-', lw=1);
plt.plot([2.5, 2.3], [1.5, 1.4], 'k-', lw=1);
plt.text(0, 1.7, 'L', color='k');

plt.plot([0.5, 1], [-2, -2], 'k-', lw=1);
plt.plot([0.5, 0.6], [-2, -1.9], 'k-', lw=1);
plt.plot([0.5, 0.6], [-2, -2.1], 'k-', lw=1);
plt.plot([1, 0.9], [-2, -1.9], 'k-', lw=1);
plt.plot([1, 0.9], [-2, -2.1], 'k-', lw=1);
plt.text(1.2, -2, 'L', color='k');

plt.tight_layout()
plt.show()