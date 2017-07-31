import sklearn.datasets as sd
import matplotlib.pyplot as plt
import matplotlib.colors as mc

'''
这两个都是用于特定算法的演示，使用不是很广泛
make_circles: 大圆套小圆
    n_samples: 略
    noise: 噪声的标准差，可选参数，默认是None
    facotr: 大圆和小圆的半径之比，默认是0.8
make_moons: 形如两个互相咬合的半圆，参数定义同上
'''

colors = ['r','g']
cmap = mc.ListedColormap(colors)

X1, y1 = sd.make_circles(factor=0.7)

plt.figure(figsize=(12, 10))
plt.subplot(221)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=y1, cmap=cmap)

# 加入噪声，标准差为0.1
X2, y2 = sd.make_circles(factor=0.7, noise=0.1)
plt.subplot(222)
plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=y2, cmap=cmap)

X3, y3 = sd.make_moons()
plt.subplot(223)
plt.scatter(X3[:, 0], X3[:, 1], marker='o', c=y3, cmap=cmap)

# 加入噪声，标准差为0.1
X4, y4 = sd.make_moons(noise=0.1)
plt.subplot(224)
plt.scatter(X4[:, 0], X4[:, 1], marker='o', c=y4, cmap=cmap)

plt.show()
