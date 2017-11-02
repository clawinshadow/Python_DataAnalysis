import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
随手写的一个示例，并不一定与书中的完全吻合
画3D的surface的时候，X, Y, Z 依然必须都是2D的数组，这个一定要注意
'''

# Multidimensional SE kernel
def multidimension_se(X, M, v_scale=1, noise_sigma=1e-8):
    N, D = X.shape
    assert D == len(M)

    distance_matrix = sm.pairwise_distances(X, metric='mahalanobis', VI=sl.inv(M))
    distance_matrix = distance_matrix**2
    gram = v_scale * np.exp(-0.5 * distance_matrix) + noise_sigma * np.eye(N)

    return gram

xx, yy = np.meshgrid(np.linspace(-3, 3, 31), np.linspace(-3, 3, 31))
xx_ravel = xx.ravel()
yy_ravel = yy.ravel()
X = np.c_[xx_ravel, yy_ravel]

N, D = X.shape
M1 = np.eye(D)
M2 = np.diag([1, 1/3])**2
M3 = np.array([[1, -1], [-1, 1]]) + np.diag([1/6, 1/6])**2
M = [M1, M2, M3]

# plots
fig = plt.figure(figsize=(13, 4))
fig.canvas.set_window_title('gprDemoArd')

for i in range(len(M)):
    Mi = M[i]
    K = multidimension_se(X, Mi)
    mu = np.zeros(N)
    Z = ss.multivariate_normal(mu, K, allow_singular=True).rvs(1)
    Z = Z.reshape(xx.shape)

    index = (int)('13' + str(i + 1))
    ax = fig.add_subplot(index, projection='3d')
    ax.grid(False)
    ax.set_xlabel('input x1')
    ax.set_ylabel('input x2')
    ax.set_zlabel('output y')
    ax.plot_surface(xx, yy, Z, cmap='jet')

plt.tight_layout()
plt.show()