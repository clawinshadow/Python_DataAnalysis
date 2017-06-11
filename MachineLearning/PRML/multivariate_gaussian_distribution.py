import numpy as np
import scipy.linalg as sl
import scipy.stats as ss
import matplotlib.pyplot as plt

'''
记一些多元高斯分布的性质，会很常用。
1. 多元高斯分布的相关矩阵 Σ 为一个实对称矩阵，它的特征向量ui, uj, ... un构成一组规范正交基, 即

   Σ*ui = λi*ui
   
                1,     if i = j
   ui.T * uj =
                0,     otherwise

   Σ = Σ(λi*ui*ui.T)

   Σ.inv = Σ(1/λi * ui*ui.T)

   多元高斯分布的相关矩阵 Σ 至少是半正定的，一般都是正定的，每个特征值都大于零
'''

# 验证 Σ = Σ(λi*ui*ui.T)
cov = np.array([[1, 1, 2],
                [1, 4, 3],
                [2, 3, 6]])
vals, vecs = sl.eig(cov)           # 特征向量是按列排列的
print('covariance matrix: \n', cov)
print('eigen values: ', vals)
print('eigen vectors: \n', vecs) 
print('u0.T * u0: ', np.dot(vecs[:, 0], vecs[:, 0]))  
print('u1.T * u1: ', np.dot(vecs[:, 1], vecs[:, 1]))
print('u2.T * u2: ', np.dot(vecs[:, 2], vecs[:, 2]))
print('u0.T * u1: ', np.dot(vecs[:, 0], vecs[:, 1]))
print('u0.T * u2: ', np.dot(vecs[:, 0], vecs[:, 2]))
print('u1.T * u2: ', np.dot(vecs[:, 1], vecs[:, 2]))
tempMat = vals[0] * np.dot(vecs[:, 0].reshape(-1, 1), vecs[:, 0].reshape(1, -1)) + \
          vals[1] * np.dot(vecs[:, 1].reshape(-1, 1), vecs[:, 1].reshape(1, -1)) + \
          vals[2] * np.dot(vecs[:, 2].reshape(-1, 1), vecs[:, 2].reshape(1, -1))
print(tempMat)
print('cov == tempMat: ', np.allclose(cov, tempMat))

# visualize, 三种不同性质的协方差矩阵
mean = np.array([0, 0])                  # 均值向量
cov = np.array([[1.0, 0.3], [0.3, 0.5]]) # 一般的协方差矩阵
# x, y 都是 N * N 的矩阵，pos是 2 * N * N
x, y = np.meshgrid(np.arange(-2, 2, 0.01), np.arange(-2, 2, 0.01)) # 构造画图需要的点集
pos = np.dstack((x, y))
rv = ss.multivariate_normal(mean, cov)   # 默认协方差矩阵不能为奇异矩阵
probs = rv.pdf(pos)                      # probs也是 N * N

fig = plt.figure(figsize=(13, 4))
fig.canvas.set_window_title('Multivariate Gaussian Distribution')
plt.subplot(131)
plt.contour(x, y, probs, colors='red')

cov = np.diag([0.5, 0.9])      # 协方差矩阵为对角矩阵
rv = ss.multivariate_normal(mean, cov)
probs2 = rv.pdf(pos)
plt.subplot(132)
plt.contour(x, y, probs2, colors='red')

cov = 0.7 * np.eye(2)           # 协方差矩阵为 系数* I
rv = ss.multivariate_normal(mean, cov)
probs3 = rv.pdf(pos)
plt.subplot(133)
plt.contour(x, y, probs3, colors='red')
plt.show()

