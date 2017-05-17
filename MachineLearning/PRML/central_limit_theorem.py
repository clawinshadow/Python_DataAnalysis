import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

'''
关于中心极限定理的一个简单示例，取N个服从uniform分布的随机变量{x1, x2, ..., xn}，重复20次，计算每次抽样的均值，得出
{mean1, mean2, ..., mean20}，然后取N分别等于1, 2, 5, 10, 会发现N越大，图形越接近于正态分布
'''

def GenerateData(N):
    dataList = []
    for i in range(N):
        data = ss.uniform().rvs(500)
        dataList.append(data)
    dataList = np.array(dataList)
    mean = np.mean(dataList, axis=0)  # 计算每列的平均值
    # print(mean)
    return mean

N_1 = GenerateData(1)
N_2 = GenerateData(2)
N_5 = GenerateData(5)
N_10 = GenerateData(10)

plt.figure(figsize=(11, 7))
ax = plt.subplot(221)
n, bins, patches = plt.hist(N_1, 20, rwidth=0.9, normed=1)   # 将500个数据分成20个组来展示成直方图
print(n)            # 经过normed之后的每个直方图中的值，不一定小于1
print(bins)         # 每个划分区间段的边界
print(patches)
print(np.sum(n))    # 每个n之和再乘以 1/20 == 1
plt.text(0.1, 0.9, 'N = 1', transform=ax.transAxes)

ax2 = plt.subplot(222)
plt.hist(N_2, 20, rwidth=0.9, normed=1)   # 将500个数据分成20个组来展示成直方图
plt.text(0.1, 0.9, 'N = 2', transform=ax2.transAxes)

ax3 = plt.subplot(223)
plt.hist(N_5, 20, rwidth=0.9, normed=1)   # 将500个数据分成20个组来展示成直方图
plt.text(0.1, 0.9, 'N = 5', transform=ax3.transAxes)

ax4 = plt.subplot(224)
plt.hist(N_10, 20, rwidth=0.9, normed=1)   # 将500个数据分成20个组来展示成直方图
plt.text(0.1, 0.9, 'N = 10', transform=ax4.transAxes)
plt.show()
