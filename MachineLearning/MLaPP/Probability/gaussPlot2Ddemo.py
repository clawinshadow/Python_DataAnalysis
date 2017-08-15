import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

samples = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(samples, samples)  # meshgrid用于生成多元高斯分布的样本点集
Z = np.dstack((X, Y))    # shape, 200 * 200 * 2

mu = [0, 0] # 均值向量
cov_full = [[2.5, 2], [2, 2.5]]
cov_diagonal = [[1, 0], [0, 3]]
cov_spherical = np.eye(2)  # isotropic

MVN_full = ss.multivariate_normal(mu, cov_full)
MVN_diagonal = ss.multivariate_normal(mu, cov_diagonal)
MVN_spherical = ss.multivariate_normal(mu, cov_spherical)

plt.figure(figsize=(8, 7.8))
plt.subplot(221)
plt.contour(X, Y, MVN_full.pdf(Z))
plt.title('full')

plt.subplot(222)
plt.contour(X, Y, MVN_diagonal.pdf(Z))
plt.title('diagonal')

plt.subplot(223)
plt.contour(X, Y, MVN_spherical.pdf(Z))
plt.title('spherical')

ax = plt.subplot(224, projection='3d')
ax.plot_surface(X, Y, MVN_spherical.pdf(Z), rstride=2, cstride=2, cmap='jet')

plt.show()
