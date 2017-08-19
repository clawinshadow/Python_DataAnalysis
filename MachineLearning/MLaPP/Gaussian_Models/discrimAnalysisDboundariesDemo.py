import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

# generate sample points
mu_r = np.array([-1, -1])  # 红色点集的均值向量
mu_b = np.array([1, 1])    # 蓝色点集的均值向量
sigma_b = np.array([[2, 0],
                    [0, 0.8]])  # 红色点集的协方差矩阵，diagonal
sigma_r = np.array([[1.2, 0],
                    [0, 0.7]])  # 蓝色点集的协方差矩阵，一样是diagonal

mvn_r = ss.multivariate_normal(mu_r, sigma_r)
mvn_b = ss.multivariate_normal(mu_b, sigma_b)

points_r = mvn_r.rvs(13)   # 红色的样本点集
points_b = mvn_b.rvs(17)

x, y = np.mgrid[-3:3:0.01, -3:2.5:0.01]
data = np.dstack((x, y))
probs_r = mvn_r.pdf(data)  # 红色的概率密度样本
probs_b = mvn_b.pdf(data)  # 蓝色的概率密度样本

# calculate the boundary, in the MC way
def getBoundary(data, probs_1, probs_2, atol):
    
    # 第一步，重构3维的样本点集data，降一维
    elementCount = len(data.ravel())  # data数组里面元素的总数
    rows = (int)(elementCount / 2)    # 重构后二维数组的行数，即样本点总数，每一行就是每个样本点的坐标
    points = data.reshape((rows, 2))  # N * 2
    print(points.shape)
    print(probs_1.min(), probs_2.max())

    boundaryPoints = []
    # 遍历每个样本点，如果它在两个高斯分布中的概率密度大致相等，即在边界线上
    rs = probs_1.ravel()
    rb = probs_2.ravel()                # 这两个的长度应该是与points的长度一致的
    gap = np.abs(rs - rb)               # 计算两者之间的绝对值
    indices = np.argwhere(gap <= atol)  # 筛选边界上的点集
    boundaryPoints = points[indices].reshape((len(indices), 2))
    print(len(boundaryPoints))

    return boundaryPoints

# Plots
plt.figure(figsize=(13, 5.5))
plt.subplot(121)
plt.xlim(-3, 3)
plt.ylim(-3, 2.5)
plt.xticks([-3, -2, 0, 2, 3])
plt.yticks([-3, -2, 0, 2, 2.5])
plt.title('Parabolic boundary')
plt.plot(points_r[:, 0], points_r[:, 1], 'ro', fillstyle='none')
plt.plot(points_b[:, 0], points_b[:, 1], 'b+')
plt.contour(x, y, probs_r, colors='r')
plt.contour(x, y, probs_b, colors='b')
# 样本点太少，得到的边界点不多，所以画出来的图象有锯齿，但没办法，采样太多的话跑起来太慢
boundaryPoints = getBoundary(data, probs_r, probs_b, 1e-05)
plt.plot(boundaryPoints[:, 0], boundaryPoints[:, 1], 'k-', linewidth=2)
plt.fill_between(boundaryPoints[:, 0], -3, boundaryPoints[:, 1], color='orange', alpha='0.1')
plt.fill_between(boundaryPoints[:, 0], boundaryPoints[:, 1], 2.5, color='lightblue', alpha='0.1')

x, y = np.mgrid[-3:7:0.01, -3:8:0.01]
data = np.dstack((x, y))

mu_r = np.array([-0.5, 5])
mu_g = np.array([4.5, 5])
mu_b = np.array([2, 0])

sigma_r = np.array([[1, 0],
                    [0, 0.7]])
sigma_g = np.array([[1, 0],
                    [0, 0.7]])  # red和green的协方差矩阵时一样的，所以它们的边界是线性的，LDA
sigma_b = np.array([[3, 0],
                    [0, 0.9]])

mvn_r = ss.multivariate_normal(mu_r, sigma_r)
mvn_g = ss.multivariate_normal(mu_g, sigma_g)
mvn_b = ss.multivariate_normal(mu_b, sigma_b)

points_r = mvn_r.rvs(7)
points_g = mvn_g.rvs(12)
points_b = mvn_b.rvs(10)

probs_r = mvn_r.pdf(data)
probs_g = mvn_g.pdf(data)
probs_b = mvn_b.pdf(data)

plt.subplot(122)
plt.xlim(-3, 7)
plt.ylim(-3, 8)
plt.xticks([-2, 0, 2, 4, 6])
plt.yticks([-2, 0, 2, 4, 6, 8])
plt.title('Some Linear, Some Quadratic')
plt.plot(points_r[:, 0], points_r[:, 1], 'ro', fillstyle='none')
plt.plot(points_g[:, 0], points_g[:, 1], 'g>', fillstyle='none')
plt.plot(points_b[:, 0], points_b[:, 1], 'b+')
plt.contour(x, y, probs_r, colors='r')
plt.contour(x, y, probs_g, colors='g')
plt.contour(x, y, probs_b, colors='b')

def getBoundary2(data, probs_1, probs_2, probs_3, atol):
    
    # 第一步，重构3维的样本点集data，降一维
    elementCount = len(data.ravel())  # data数组里面元素的总数
    rows = (int)(elementCount / 2)    # 重构后二维数组的行数，即样本点总数，每一行就是每个样本点的坐标
    points = data.reshape((rows, 2))  # N * 2

    boundaryPoints = []
    # 遍历每个样本点，如果它在两个高斯分布中的概率密度大致相等，即在边界线上
    r1 = probs_1.ravel()
    r2 = probs_2.ravel()
    r3 = probs_3.ravel()                     # 这三个的长度应该是与points的长度一致的
    gap = np.abs(r1 - r2)                    # 计算两者之间的绝对值
    equal_indices = gap <= atol              # probs_1 = probs_2 的点集
    greater_indices = r1 > r3                # 剔除掉equal_indices中多余的点集
    final_indices = equal_indices & greater_indices
    print(final_indices)
    
    boundaryPoints = points[final_indices].reshape((len(points[final_indices]), 2))
    print(len(boundaryPoints))

    return boundaryPoints

boundaryPoints_rg = getBoundary2(data, probs_r, probs_g, probs_b, 1e-08)
boundaryPoints_rb = getBoundary2(data, probs_r, probs_b, probs_g, 1e-05)
boundaryPoints_bg = getBoundary2(data, probs_b, probs_g, probs_r, 1e-05)
plt.plot(boundaryPoints_rg[:, 0], boundaryPoints_rg[:, 1], 'k-', linewidth=2)
plt.plot(boundaryPoints_rb[:, 0], boundaryPoints_rb[:, 1], 'k-', linewidth=2)
plt.plot(boundaryPoints_bg[:, 0], boundaryPoints_bg[:, 1], 'k-', linewidth=2)

plt.show()

