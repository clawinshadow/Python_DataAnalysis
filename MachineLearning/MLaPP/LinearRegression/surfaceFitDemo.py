import numpy as np
import scipy.linalg as sl
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator

def Transform_Linear(x):
    # 扩展为 [1, x1, x2]
    a = np.ones(len(x)).reshape(-1, 1)
    return np.hstack((a, x))

def Transform_Nonlinear(x):
    # 扩展为 [1, x1, x2, x1**2, x2**2]
    a = np.ones(len(x)).reshape(-1, 1) 
    return np.c_[a, x, x**2]                # 类似于np.hstack()

def GetWeights(X, y):
    # X是design matrix, y是目标向量，本例中直接使用公式求解(最小二乘法)
    return np.dot(sl.inv(np.dot(X.T, X)), np.dot(X.T, y))

data = sio.loadmat("moteData.mat")
print("keys of data: ", list(data.keys()))
print('data.X.shape: ', data["X"].shape)
print('data.y.shape: ', data['y'].shape)
print('top 10 data: ', [data["X"][0:10], data["y"][0:10]])

dm1 = Transform_Linear(data['X'])
dm2 = Transform_Nonlinear(data['X'])
w1 = GetWeights(dm1, data['y'])
w2 = GetWeights(dm2, data['y'])
print('w.shape: ', w1.shape)

X, Y = np.meshgrid(np.linspace(-2, 42, 200), np.linspace(-2, 33, 200))
# 注意下面如何重构数组的结构以方便计算Z
# 1. 先将X Y展开成一维数组
# 2. 将1, X, Y按列组合在一起，形成 N * 3 的矩阵
# 3. 将此矩阵与权重向量相乘 (N * 3) * (3 * 1) = (N * 1)
# 4. 再将Z矩阵reshape成X，Y原始的形状
combine = np.c_[X.ravel().reshape(-1, 1), Y.ravel().reshape(-1, 1)]
combine1 = Transform_Linear(combine)
combine2 = Transform_Nonlinear(combine)
Z1 = np.dot(combine1, w1).reshape(X.shape)
Z2 = np.dot(combine2, w2).reshape(X.shape)
print('X.shape: ', X.shape)
print('Y.shape: ', Y.shape)
print('Z.shape: ', Z1.shape)

fig = plt.figure(figsize=(11, 5))
fig.canvas.set_window_title('surfaceFitDemo')
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(data["X"][:, 0], data["X"][:, 1], data['y'].ravel(), color='red')
# 注意edgecolors的使用，是给网格线的颜色赋值
ax1.plot_surface(X, Y, Z1, rstride=10, cstride=10, linewidth=1, edgecolors='k', cmap='jet')
ax1.set_xlim(-2, 42)
ax1.set_xticks(np.arange(0, 41, 10))
ax1.set_ylim(-2, 33)
ax1.set_yticks(np.arange(0, 32, 5))
ax1.set_zlim(Z1.min() - 1, Z1.max() + 1)
ax1.set_zticks(np.arange(15.5, 18.5, 0.5))

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(data["X"][:, 0], data["X"][:, 1], data['y'].ravel(), color='red')
ax2.plot_surface(X, Y, Z2, rstride=10, cstride=10, linewidth=1, edgecolors='k', cmap='jet')
ax2.set_xlim(-2, 42)
ax2.set_xticks(np.arange(0, 41, 10))
ax2.set_ylim(-2, 33)
ax2.set_yticks(np.arange(0, 32, 10))
ax2.set_zlim(14.5, 18.5)
ax2.set_zticks(np.arange(15, 18.5, 0.5))

plt.show()

