import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# sample data definition
mu1 = [0.22, 0.45]
mu2 = [0.5, 0.5]
mu3 = [0.77, 0.55]
cov1 = [[0.011, -0.01], [-0.01, 0.018]]
cov2 = [[0.018, 0.01], [0.01, 0.011]]
cov3 = cov1;
mixCoef = [0.5, 0.3, 0.2]

rv1 = ss.multivariate_normal(mu1, cov1)
rv2 = ss.multivariate_normal(mu2, cov2)
rv3 = ss.multivariate_normal(mu3, cov3)

# generate plot datas
X, Y = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
Z = np.dstack((X, Y))
Z1 = rv1.pdf(Z)
Z2 = rv2.pdf(Z)
Z3 = rv3.pdf(Z)

mix_Z = mixCoef[0] * Z1 + mixCoef[1] * Z2 + mixCoef[2] * Z3

# plots
fig = plt.figure(figsize=(11, 5))
fig.canvas.set_window_title('mixGaussPlotDemo')

plt.subplot(121)
plt.axis([0, 1, 0.16, 0.85])
plt.xticks(np.arange(0, 1.01, 0.1))
plt.yticks(np.arange(0.2, 0.85, 0.1))
plt.contour(X, Y, Z1, 8, colors='r')
plt.contour(X, Y, Z2, 8, colors='g')
plt.contour(X, Y, Z3, 8, colors='b')

ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X, Y, mix_Z, color='saddlebrown', edgecolors='saddlebrown')

plt.show()
