import sklearn.datasets as sd
import matplotlib.pyplot as plt

'''
make_regression: 生成用于回归算法测试的数据集，就是特征列的线性组合+高斯噪声
    n_samples: 略
    n_features: 略
    n_informative: 有效特征列的数目，即彼此线性不相关的
    n_targets: 目标向量的维度，一般都是标量，默认也是1
    bias: 是否有截距
    effective_rank: 与n_informative类似，有效的秩是多少，即线性不相关的特征列有多少个，默认是0，表示是特征列是满秩的
    tail_strength: 参考make_low_rank_matrix中的这个参数，只有当effective_rank不为None时才起作用
    noise:噪声的标准差
    coef: 当这个为True时，返回值中会有生成这个模型的系数
    random_state: 略
Return: X, y, coef
'''

# 首先是没有截距的数据集
X, y, coef = sd.make_regression(n_features=1, n_informative=1, noise=0.1, coef=True, random_state=0)
print('X : \n', X)
print('Y without bias : \n', y)
print('coefficiens: ', coef)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Regression datasets without bias')
plt.scatter(X, y, marker='o', c='r')

# 加入截距 5
X2, y2, coef2 = sd.make_regression(n_features=1, n_informative=1, noise=0.1, bias=20, coef=True, random_state=0)
print('X2 : \n', X2)
print('Y2 with bias : \n', y2)
print('coefficiens: ', coef2)

plt.subplot(122)
plt.title('Regression datasets with bias')
plt.scatter(X2, y2, marker='o', c='r')

plt.show()
