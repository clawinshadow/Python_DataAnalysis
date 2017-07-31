import matplotlib.pyplot as plt
import matplotlib.colors as mc
import sklearn.datasets.samples_generator as sds

'''
make_blobs: blob就是团的意思，几乎相当于cluster，生成几团数据，每一团都服从isotropic的高斯分布
    n_samples: 数据点的总数
    n_features: 每个数据点的维度，默认是2
    centers: 每团数据的中心点的个数，或者是具体的中心坐标，对应于每个高斯分布的均值向量
    cluster_std: 每团数据的标准差，默认是1
    center_box: 每个中心点的边界， (min, max) 数据对
    shuffle: 是否打乱分类数据，默认是True
    randome_state: 略

    Returns: X, y . X 是每个数据点的坐标，y是每个数据点对应的分类标签
'''

X, y = sds.make_blobs(n_samples=100, centers=3, n_features=2, random_state=0)
print('Generate 3 clusters.')
print('X: \n', X)
print('y: ', y)

# 自定义一个离散的colormap，因为y的取值都是0， 1， 2， 对应这个colormap的每一种颜色
colors = ['r','g','b']
myCMap=mc.ListedColormap(colors)

plt.figure(figsize=(12, 6))
plt.subplot(121)
# 此时c参数是一个长度为n_samples的序列，标注了每个数据点的颜色
# edgecolors='face'表明不要边缘颜色，与圈内填充的颜色一致
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, cmap=myCMap, edgecolors='face')

# 自定义中心点和标准差，让每个cluster之间的间隔更清晰
centers = [[-3, 3], [0, 5], [3, 3]]
stds = [0.5, 0.5, 0.5]
X2, y2 = sds.make_blobs(centers=centers, cluster_std=stds, n_features=2, random_state=0)

plt.subplot(122)
plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=y2, cmap=myCMap, edgecolors='face')

plt.show()
