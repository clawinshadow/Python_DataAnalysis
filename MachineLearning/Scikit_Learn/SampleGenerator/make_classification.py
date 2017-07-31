import numpy as np
import scipy.linalg as sl
import sklearn.datasets as sd
import matplotlib.pyplot as plt

'''
make_classification: 生成分类数据
    n_samples: 生成数据集的样本总数
    n_features: 特征列的总数，等于 n_informative + n_redundant + n_repeated
    n_informative: 相互独立的特征列个数，线性不相关，特征向量子空间中的一组基
    n_redundant: 冗余特征列的个数，每一个冗余特征等于所有独立特征的线性组合, 这个是必选参数
    n_repeated: 重复列的个数，前面任一列的简单重复
    n_classes: 分类的个数，默认是2
    n_clusters_per_class: 每个分类中的cluster数量
    weights: 各分类向量占总数量的比重，默认是均分的
    flip_y: 翻转分类标签的比例，用于制造离群点
    class_sep: 不明
    hypercube: 不明，可能与cluster的具体生成方式有关
    shift: 平移数据
    scale: 缩放数据
    shuffle & random_state: 略
'''

X1, y1 = sd.make_classification(n_samples=10, n_features=3, n_informative=3, n_redundant=0, flip_y=0, random_state=0)
print('X1: \n', X1)
print('y1: ', y1)
# 每一列都是相互独立的，所以X1的rank应该等于3，即 X1.T * X1 的det不等于零
print('det(X1.T*X1) == 0: ', np.allclose(sl.det(np.dot(X1.T, X1)), 0))

X2, y2 = sd.make_classification(n_samples=10, n_features=3, n_informative=2, n_redundant=1, flip_y=0, random_state=0)
print('X2: \n', X2)
print('y2: ', y2)
# 有一列是冗余的，那么X2的rank应该等于2
print('det(X2.T*X2) == 0: ', np.allclose(sl.det(np.dot(X2.T, X2)), 0))
# 相应的SVD后的值，会有一个非常接近于零
print('SVD values of X2: ', sl.svdvals(X2))

X3, y3 = sd.make_classification(n_samples=10, n_features=3, n_informative=2, n_redundant=0, \
                                n_repeated=1, flip_y=0, random_state=0)
print('X3: \n', X3)
print('y3: ', y3)
# 有一列是重复列，那么X2的rank也应该等于2
print('det(X3.T*X3) == 0: ', np.allclose(sl.det(np.dot(X3.T, X3)), 0))
# 相应的SVD后的值，会有一个几乎等于零
print('SVD values of X3: ', sl.svdvals(X3))

X4, y4 = sd.make_classification(n_samples=10, n_features=3, n_informative=3, n_redundant=0, \
                                shift=1.0, flip_y=0, random_state=0)
print('X4: \n', X4)
print('y4: ', y4)
# 与X1的唯一不同之处在于所有数据都平移了1.0
print('X4 - X1: \n', np.subtract(X4, X1))

X5, y5 = sd.make_classification(n_samples=10, n_features=3, n_informative=3, n_redundant=0, \
                                scale=2.0, flip_y=0, random_state=0)
print('X5: \n', X5)
print('y5: ', y5)
# 与X1的唯一不同之处在于所有数据都乘以了2.0
print('X5 / X1: \n', np.divide(X5, X1))

X6, y6 = sd.make_classification(n_samples=10, n_features=3, n_informative=3, n_redundant=0, \
                                weights=[0.7], flip_y=0, random_state=0)
print('X6: \n', X6)
print('y6: ', y6) # 与X1的唯一不同之处在于分类标签中0和1的比例不再是1:1了，而是7:3
