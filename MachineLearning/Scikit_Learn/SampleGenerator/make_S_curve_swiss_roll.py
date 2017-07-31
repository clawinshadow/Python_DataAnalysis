import numpy as np
import sklearn.datasets as sd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from mpl_toolkits.mplot3d import Axes3D

'''
这两个数据集生成器主要用于流行学习算法的测试，manifold learning
注意这两个数据集都是三维的

make_s_curve: S型的数据集
    n_samples: 略
    noise: 噪声的标准差
    random_state: 略
make_swiss_roll: 形似瑞士卷的数据集，见图形，参数同上
'''
# S曲线
n_points = 1000
X, color = sd.make_s_curve(n_points, random_state=0)

fig = plt.figure(figsize=(12, 6))
plt.suptitle("Datasets for manifold learning.")

ax = fig.add_subplot(121, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(4, -72)

# Swiss Roll
noise=0.05
X2, labels = sd.make_swiss_roll(n_points, noise=noise, random_state=0)
ax = fig.add_subplot(122, projection='3d')
# ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], c=color, cmap=plt.cm.Spectral)
# 离散化的colormap
for l in np.unique(labels):
    ax.plot3D(X2[labels == l, 0], X2[labels == l, 1], X2[labels == l, 2],
              'o', color=plt.cm.jet(np.float(l) / np.max(labels + 1)))
ax.view_init(7, -80)

plt.show()
