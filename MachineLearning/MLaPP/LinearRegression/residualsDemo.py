import numpy as np
import scipy.linalg as sl
import scipy.optimize as so
import sklearn.linear_model as slm
import matplotlib.pyplot as plt

# generate data
w_true = [1, 1]                   # 真实的权重向量
x = np.linspace(-3.5, 3.5, 200)
y = w_true[0] + w_true[1] * x

# plot truth line
fig = plt.figure(figsize=(7, 6))
fig.canvas.set_window_title('residualsDemo')

ax = plt.subplot()
ax.grid(True, linestyle=':')
plt.plot(x, y, color='lightblue', linestyle=':', label='truth')
plt.ylim(-3, 6)
plt.xlim(-4, 4)
plt.yticks(np.arange(-3, 5.1, 1))
plt.xticks(np.arange(-4, 4.1, 1))

# generate noisy data, 任意抽取20个x值，然后再加上N(0, 0.8**2)的噪声
randIndices = np.random.randint(0, 200, size=20)
x_noisy = x[randIndices]
y_noisy = y[randIndices] + 0.8 * np.random.randn(20)

# plot noisy data, empty circle
plt.plot(x_noisy, y_noisy, color='r', ls='None', marker='o', markeredgewidth=1.5, fillstyle='none') # ls='None'，不要画直线

# 计算MLE的权重向量
dm = np.c_[np.ones(len(x_noisy)).reshape(-1, 1), x_noisy.reshape(-1, 1)]
y_vector = y_noisy.reshape(-1, 1)
w_MLE = np.dot(sl.inv(np.dot(dm.T, dm)), np.dot(dm.T, y_vector))
print('W_MLE: ', w_MLE.ravel())

# 原先的样本点在w_MLE下的估计值
y_MLE = w_MLE[0] + w_MLE[1] * x
y_noisy_MLE = w_MLE[0] + w_MLE[1] * x_noisy

# plot estimated values
plt.title('w_MLE: {0}'.format(w_MLE.ravel()))
plt.plot(x_noisy, y_noisy_MLE, color='midnightblue', ls='None', marker='x')
plt.plot(x, y_MLE, 'r-', label='prediction', lw=1.5)
for i in range(len(x_noisy)):
    plt.plot([x_noisy[i], x_noisy[i]], [y_noisy[i], y_noisy_MLE[i]], color='midnightblue', lw=1.5)

plt.legend()
plt.show()


