import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt

def randRange(vmin, vmax, size=20):
    return vmin + (vmax - vmin) * np.random.rand(20)

def SSE(w, x_train, y_train):
    return np.sum(np.power(y_train - w[0] - w[1] * x_train, 2))

# generate data
w_true = [1, 1]                   # 真实的权重向量
x = randRange(-4, 4, 20)
y = w_true[0] + w_true[1] * x + 0.8 * np.random.randn(20)    # N(0, 0.8**2)的噪声

w0 = [1, 1]
res = so.minimize(SSE, w0, args=(x, y))
print('Optimization Success: ', res.success)
print('w by scipy.optimize.minimize(): ', res.x)

fig = plt.figure(figsize=(7, 6))
fig.canvas.set_window_title('contoursSSEdemo')
plt.axis([-1, 3, -1, 3])
plt.xticks(np.arange(-1, 3.1, 1))
plt.yticks(np.arange(-1, 3.1, 0.5))
plt.xlabel('w0')
plt.ylabel('w1')
plt.title('Sum of squares error contours for linear regression', fontdict={ 'fontsize': 10 })
plt.plot([res.x[0]], [res.x[1]], 'rx', markersize=10, markeredgewidth=2)

X, Y = np.meshgrid(np.linspace(-1, 3, 200), np.linspace(-1, 3, 200))
X_flat = X.ravel()
Y_flat = Y.ravel()
Z_flat = []
for i in range(len(X_flat)):
    Z_flat.append(SSE([X_flat[i], Y_flat[i]], x, y))

Z = np.array(Z_flat).reshape(X.shape)

plt.contour(X, Y, Z, cmap='jet')

plt.show()

