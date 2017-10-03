import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

'''
sub-Gaussian / super-Gaussian 的区别在于峰态是大于还是小于零，大于零的，则尖峰比正态分布高，
相应的具有更长的尾巴，是super-Gaussian，小于零的则比正态分布要更平缓，是sub-Gaussian，在ICA中
p(z)是sub还是super的Gaussian是很重要的，至于是具体哪种sub或者super的分布就不重要了
'''

np.random.seed(0)

# generate data and samples
x = np.linspace(-4, 4, 500)
rv1 = ss.norm()
rv2 = ss.laplace(0, 1)
rv3 = ss.uniform(-2, 4)  # uniform的参数有点奇怪，比如这里就表示取值范围是(-2, -2+4)

pdf1 = rv1.pdf(x)
pdf2 = rv2.pdf(x)
pdf3 = rv3.pdf(x)

N = 5000  # Monte Carlo No.
gaussian_x1, gaussian_x2 = rv1.rvs(N), rv1.rvs(N)
laplace_x1, laplace_x2 = rv2.rvs(N), rv2.rvs(N)
uniform_x1, uniform_x2 = rv3.rvs(N), rv3.rvs(N)

# plots
fig = plt.figure(figsize=(11,9))
fig.canvas.set_window_title('subSuperGaussPlot')

ax = plt.subplot(221)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.axis([-4, 4, 0, 0.5])
plt.xticks(np.linspace(-4, 4, 9))
plt.yticks(np.arange(0, 0.51, 0.05))
plt.plot(x, pdf1, 'b-', lw=2)
plt.plot(x, pdf2, 'r:', lw=2)
plt.plot(x, pdf3, 'g--', lw=2)

plt.subplot(222)
plt.title('Gaussian')
plt.axis([-4.5, 4.5, -4, 4])
plt.xticks(np.linspace(-4, 4, 9))
plt.yticks(np.linspace(-3, 3, 7))
plt.plot(gaussian_x1, gaussian_x2, 'b.', ms=1)

plt.subplot(223)
plt.title('Laplace')
plt.axis([-11, 11, -8, 10])
plt.xticks(np.linspace(-10, 10, 5))
plt.yticks(np.linspace(-8, 10, 10))
plt.plot(laplace_x1, laplace_x2, 'r.', ms=1)

plt.subplot(224)
plt.title('Uniform')
plt.axis([-2.5, 2.5, -2, 2])
plt.xticks(np.linspace(-2, 2, 5))
plt.yticks(np.linspace(-1.5, 1.5, 7))
plt.plot(uniform_x1, uniform_x2, 'g.', ms=1)

plt.show()
