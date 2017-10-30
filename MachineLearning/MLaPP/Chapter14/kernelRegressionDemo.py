import numpy as np
import scipy.io as sio
import scipy.linalg as sl
import matplotlib.pyplot as plt

# prepare data
data = sio.loadmat('kernelRegressionDemo.mat')
print(data.keys())
x = data['x']
y = data['y']
ytrue = data['ytrue']

# Fit with kernel regression
def get_h(x, y):
    x = x.ravel()
    y = y.ravel()
    N = len(x)
    mad_x = np.median(np.abs(x - np.median(x)))
    mad_y = np.median(np.abs(y - np.median(y)))
    sigma_x = mad_x / 0.6745
    sigma_y = mad_y / 0.6745
    h_x, h_y = sigma_x * (4 / (3 * N))**0.2, sigma_y * (4 / (3 * N))**0.2
    h = np.sqrt(h_x * h_y)

    return h

def gaussianKernel(x, centers, h):
    N, D = x.shape
    C = len(centers)
    weights = np.zeros((N, C))
    for i in range(C):
        center = centers[i]
        val1 = 1 / np.sqrt(2 * np.pi)
        val2 = np.exp(-1 * sl.norm(x - center, axis=1)**2 / (2 * h**2))
        weights[:, i] = (1/h) * val1 * val2
    sum = np.sum(weights, axis=1).reshape(-1, 1)
    weights = weights / sum

    return weights

h = get_h(x, y)
weights = gaussianKernel(x, x, h)
y_kr = np.dot(weights, y)

# plots
fig = plt.figure()
fig.canvas.set_window_title('kernelRegressionDemo')

plt.subplot()
plt.title('Gaussian kernel regression')
plt.axis([-2, 2, -0.4, 1.2])
plt.xticks(np.linspace(-2, 2, 9))
plt.yticks(np.linspace(-0.4, 1.2, 9))
plt.plot(x, y, 'kx', ms=3, linestyle='none', label='data')
plt.plot(x, ytrue, 'b-', label='true')
plt.plot(x, y_kr, 'r--', lw=2, label='estimate')

plt.legend()
plt.tight_layout()
plt.show()

