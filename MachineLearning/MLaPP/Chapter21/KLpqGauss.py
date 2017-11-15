import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

mu = np.array([0, 0])
sigma = np.array([[1, 0.97],
                  [0.97, 1]])

sigmaKLa = np.eye(2) / 25
sigmaKLb = np.eye(2)

xx, yy = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200))
xs = np.c_[xx.ravel(), yy.ravel()]

z1 = ss.multivariate_normal(mu, sigma).pdf(xs).reshape(xx.shape)
z2 = ss.multivariate_normal(mu, sigmaKLa).pdf(xs).reshape(xx.shape)
z3 = ss.multivariate_normal(mu, sigmaKLb).pdf(xs).reshape(xx.shape)

# plots
fig = plt.figure(figsize=(10.5, 5))
fig.canvas.set_window_title('KLpqGauss')

def plot(index, data):
    ax = plt.subplot(index, aspect='equal')
    ax.tick_params(direction='in')
    plt.axis([-1, 1, -1, 1])
    plt.xticks(np.linspace(-1, 1, 11))
    plt.yticks(np.linspace(-1, 1, 11))
    plt.contour(xx, yy, z1, colors='b', linewidths=1)
    plt.contour(xx, yy, data, colors='r', linewidths=1)

plot(121, z2)
plot(122, z3)

plt.tight_layout()
plt.show()