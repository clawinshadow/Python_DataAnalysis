import numpy as np
import matplotlib.pyplot as plt

samples = np.random.rand(6000, 2) # 生成一个6000行2列的随机数矩阵，取值范围都是[0, 1]之间
samples = samples * 4 - 2         # 调整取值范围到 [-2, 2]

inCircle = np.sum(samples**2, axis=1) <= 4 # bool 矩阵，筛选出在半径为2的圆内的样本点
pi = np.sum(inCircle) / len(samples)  # Monte Carlo方法来估计pi
print('the estimate of pi: ', pi)
print('the standard of pi: ', np.pi)

plt.figure(figsize=(8, 8))
plt.subplot()
plt.plot(samples[inCircle, 0], samples[inCircle, 1], 'bo')     # 园内的点用蓝色标识
plt.plot(samples[~inCircle, 0], samples[~inCircle, 1], 'ro')   # 注意使用bool矩阵来筛选数组元素的index trick, 以及~的用法
plt.xticks([-2, -1, 0, 1, 2])
plt.yticks(np.arange(-2, 2.5, 0.5))

plt.show()
