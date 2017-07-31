import sklearn.datasets as sd
import matplotlib.pyplot as plt
import matplotlib.colors as mc

'''
make_gaussian_quantiles: 这个比较奇葩，用于从一个isotropic的高斯分布中采样，相当于划分出一个同心圆，然后每个类的数据
均匀的分布在相邻两个圆环之间
    mean: 用于指定高斯分布的均值向量
    cov: 用于指定高斯分布的协方差矩阵，因为是isotropic的，所以这个参数只是一个标量，而不是一个矩阵
    n_samples & n_features & shuffle & random_state: 略
'''

plt.figure()
plt.subplot()

# 自定义colormap
colors = ['r','g','b']
myCMap=mc.ListedColormap(colors)

plt.title('Gaussian divided into three quantiles')
X1, Y1 = sd.make_gaussian_quantiles(n_features=2, n_classes=3)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, cmap=myCMap, edgecolors='face')

plt.show()
