import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
import matplotlib.pyplot as plt
import sklearn.preprocessing as sp
from mixexpFit import *

'''
!!!此处需要用到同级目录中的mixexpFit.py文件!!!

需要牢记的是在MoE模型中，给定K个专家，那么 W 和 V 参数矩阵虽然shape都是K*D，但是其功能和意义都完全不一样
V 矩阵是给softmax函数使用的，W 矩阵相当于是线性回归的参数，
V 矩阵用于计算每个数据点属于每个K分类的概率， W 矩阵用于计算数据点在每个分类中的回归模型中的预测概率
最后就是一维数组的参数，每个K分类的方差
'''

np.random.seed(0)
# generate data
w = 0.01 * np.random.randn(3)
b = [-1, 1, -1]
x = np.linspace(-1, 1, 50)
y = []
for i in range(len(x)):
    xi = x[i]
    yi = 0
    if xi <= -0.5:
        yi = w[0] * xi + b[0]
    elif xi <= 0.5:
        yi = w[1] * xi + b[1]
    else:
        yi = w[2] * xi + b[2]

    y.append(yi)

y = np.array(y) + 0.2 * np.random.randn(len(x))

# Fit Model
# LG = slm.LogisticRegression(C=1e10, fit_intercept=False, multi_class='multinomial')
x_train = x.reshape(-1, 1)
SS = sp.StandardScaler()
SS.fit(x_train)    # 记住参数
x_train = SS.transform(x_train)
x_train = np.c_[np.ones(len(x_train)).reshape(-1, 1), x_train]
y_train = y.reshape(-1, 1)

K = 3
mixCoef, weightMat, sigmaMat = FitMixExp(x_train, y_train, K)

# plot training points
fig = plt.figure(figsize=(11, 10))
fig.canvas.set_window_title('mixexpDemo')

plt.subplot(221)
plt.title('expert predictions, fixed mixing weights: False')
plt.axis([-1, 1, -1.5, 1.5])
plt.xticks(np.linspace(-1, 1, 5))
plt.yticks(np.linspace(-1.5, 1.5, 7))
plt.plot(x, y, 'bo', ls='none', fillstyle='none')

# plot the regression line
xx = np.linspace(-1, 1, 200)
xx_standard = SS.transform(xx.reshape(-1, 1))
xx_standard = np.c_[np.ones(len(xx_standard)).reshape(-1, 1), xx_standard]
yy = np.dot(xx_standard, weightMat.T)
plt.plot(xx, yy[:, 0], 'r:')
plt.plot(xx, yy[:, 1], 'b-')
plt.plot(xx, yy[:, 2], 'k-.')

# plot the gating functions
yy2 = []
for i in range(len(xx_standard)):
    xi = xx_standard[i]
    yy2.append(Softmax(xi, mixCoef))
yy2 = np.array(yy2).reshape(yy.shape)

plt.subplot(222)
plt.title('gating functions, fixed mixing weights: False')
plt.axis([-1.1, 1.1, -0.2, 1.2])
plt.xticks(np.linspace(-1, 1, 5))
plt.yticks(np.linspace(0, 1, 11))
plt.plot(xx, yy2[:, 0], 'r:')
plt.plot(xx, yy2[:, 1], 'b-')
plt.plot(xx, yy2[:, 2], 'k-.')

# plot the predicts
muk = np.dot(x_train, weightMat.T)
mus = []
sigmas = []
sigmaMat = sigmaMat.ravel() ** 0.5
for i in range(len(x_train)):
    pi = Softmax(x_train[i], mixCoef).ravel()
    muki = muk[i].ravel()
    mui = np.sum(muki * pi)
    mus.append(mui)  # 均值
    sigmas.append(np.sum(pi * (sigmaMat + muki**2)) - mui**2)

sigmas = 0.5 * np.array(sigmas) ** 0.5

plt.subplot(223)
plt.title('predicted mean and var, fixed mixing weights: False')
plt.axis([-1.1, 1, -2, 1.5])
plt.xticks(np.linspace(-1, 1, 5))
plt.yticks(np.linspace(-2, 1.5, 8))
plt.plot(x, y, 'bo', ls='none', fillstyle='none')
plt.errorbar(x, mus, yerr=sigmas, capsize=2, barsabove=True, elinewidth=1)
plt.plot(x, mus, 'r-')

plt.show()
