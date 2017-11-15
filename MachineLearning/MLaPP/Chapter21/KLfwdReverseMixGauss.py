import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

'''
几个简单的图而已，并没有用到两个优化 KL 散度的算法来求解，会很复杂
'''

mus = np.array([[-1, -1],
                [1, 1]])
sigmas = np.zeros((2, 2, 2))
sigmas[0] = np.array([[0.5, 0.25],
                      [0.25, 1]])
sigmas[1] = np.array([[0.5, -0.25],
                      [-0.25, 1]])
sigmaKL = np.array([[3, 2],
                    [2, 3]])

xx, yy = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
xs = np.c_[xx.ravel(), yy.ravel()]

f1 = ss.multivariate_normal(mus[0], sigmas[0]).pdf(xs)
f2 = ss.multivariate_normal(mus[1], sigmas[1]).pdf(xs)
klf = ss.multivariate_normal(np.zeros(2), sigmaKL).pdf(xs).reshape(xx.shape)
kll = ss.multivariate_normal(mus[0], sigmas[0] * 0.6).pdf(xs).reshape(xx.shape)  # more compact
klr = ss.multivariate_normal(mus[1], sigmas[1] * 0.6).pdf(xs).reshape(xx.shape)

f = (f1 + f2).reshape(xx.shape)

# plots
fig = plt.figure(figsize=(13, 4))
fig.canvas.set_window_title('KLfwdReverseMixGauss')

plt.subplot(131, aspect='equal')  # make plot square
plt.title('KL Forward')
plt.axis('off')                   # hide axis
plt.contour(xx, yy, f, colors='b')
plt.contour(xx, yy, klf, colors='r')

plt.subplot(132, aspect='equal')  # make plot square
plt.title('KL Reverse 1')
plt.axis('off')                   # hide axis
plt.contour(xx, yy, f, colors='b')
plt.contour(xx, yy, kll, colors='r')

plt.subplot(133, aspect='equal')  # make plot square
plt.title('KL Reverse 2')
plt.axis('off')                   # hide axis
plt.contour(xx, yy, f, colors='b')
plt.contour(xx, yy, klr, colors='r')

plt.tight_layout()
plt.show()