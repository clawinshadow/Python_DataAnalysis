import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

mu = np.array([0, 2])
sigma = np.array([1, 0.05])
# use Mixture Gaussian model to build a bimodal distribution
rv1 = ss.norm(mu[0], sigma[0])
rv2 = ss.norm(mu[1], sigma[1])
x = np.linspace(-2, 4, 1000)
y = 0.5 * rv1.pdf(x) + 0.5 * rv2.pdf(x)
mean = np.sum(x*y) / len(x)  # Monto Carlo 方法来估计次概率分布的均值
print('mean: ', mean)

plt.figure()
plt.subplot()
plt.xlim(-2, 4)
plt.ylim(0, 4.5)
plt.plot(x, y, 'k-')
plt.vlines(mean, 0, 4.5, color='darkblue', linewidth=3)

plt.show()
