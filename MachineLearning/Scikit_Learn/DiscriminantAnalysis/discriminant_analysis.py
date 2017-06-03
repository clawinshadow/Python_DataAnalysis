import numpy as np
import scipy.linalg as sl
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
import sklearn.discriminant_analysis as sda

'''
参考资料：1. 《多元统计分析》 第四章 判别分析
          2. 维基百科：https://en.wikipedia.org/wiki/Linear_discriminant_analysis
          3. sklearn doc: http://scikit-learn.org/stable/modules/lda_qda.html

判别分析的核心是马氏距离，假设待判别向量为 x, 样本中共有k个分类，每个分类的数据集为 Gi, i = 1, 2, ..., k;
每个分类的样本均值为μ1, μ2, ..., μk 协方差矩阵为Σ1, Σ2, ..., Σk

则 x 到各个样本均值的马氏距离为 M_d = (x - μi).T * Σi.inv * (x - μi), 判别分析的核心思想就是计算x到每个分类的
样本均值的马氏距离，选取最近的一个作为它的分类

1. 假如各分类的协方差矩阵均不相同，则每两个分类的马氏距离之差:

   W(x) = (x - μi).T * Σi.inv * (x - μi) - (x - μj).T * Σj.inv * (x - μj)
   
   i, j 是 k 中任意两个分类，那么所有情况有 k(k-1)/2 种，从中选取一个i，当所有j不等于i的时候，W(x)均小于零，
   意味着 x 到 μi 的距离是最近的，所以与别的均值的距离之差总是小于零，这个i分类就是判别分析的分类结果。

   因为判别函数W(x)是一个二次型函数，所以此时的判别分析也称为二次判别分析，Quadratic Disciminant Analysis(QDA)

2. 假如各分类的协方差矩阵相等，均为Σ，则任意两个分类的马氏距离之差为：

   W(x) = M_d(i) - M_d(j) = 2*(x - (μi+μj)/2).T * Σ.inv * (μi - μj)

   注意这个i, j的顺序会颠倒过来，所以这时候选取一个i作为分类结果时，它需要满足的条件是，对于所有j不等于i，
   W(x)均要大于零，与第一种情况恰恰相反

   因为此时判别函数W(x)变成了一个线性函数，所以称为线性判别分析，Linear Discriminant Analysis (LDA)

3. 还有一种判别分析称之为费歇判别 Fisher's Discriminant Analysis, 它的核心思想与一元方差分析类似，将K组p维数据
   投影到某一个方向，使得组与组之间的投影尽可能的分开，而每组之内的方差又要尽可能的小，换言之就是组间平方和要大
   组内平方和要小。

   假设方向向量为 a, 样本都是p维向量，则组间平方和和组内平方和分别用a.T * B * a 和 a.T * E * a 来表示，B与E都是
   n*n矩阵，具体算法不赘述

   那么就转化成一个最优化问题的求解
           a.T * B * a
   Δ(a) = -------------
           a.T * E * a

   可以转化成只含等式约束的最优化问题，即 constraint: a.T * E * a = 1, 求 Maximize(a.T * B * a)
   根据二阶充分必要条件，求导可得a为 E.inv * B的最大特征值对应的特征向量 
'''
# a simple binary-classification example
X = np.array([[-1, -1],
              [-2, -1],
              [-3, -2],
              [1, 1],
              [2, 1],
              [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = sda.LinearDiscriminantAnalysis(store_covariance=True)   # 两个分类的协方差矩阵相同，所以使用LDA
clf.fit(X, y)
print('training dataset D: \n', X)
print('training target: ', y)
print('LinearDiscriminantAnalysis class object: \n', clf)
print('weights vector - clf.coef_: ', clf.coef_)  # 权重向量，不包括bias
print('bias - clf.intercept_: ', clf.intercept_)  # 截距，即bias
# 训练集每个分类的协方差矩阵，对LDA来说都是一样的, store_covariance必须=True
# 注意这个协方差矩阵是标准化之后的协方差阵，也是相关阵
# 直接计算的协方差阵应该是np.cov([[-1, -1], [-2, -1], [-3, -2]]) = [[1, 0.5], [0.5, 0.333333]]
print('covariance matrix with default solver: \n', clf.covariance_) 
print('individual class means: ', clf.means_)     # 每个分类的样本均值
print('overall mean of D: ', clf.xbar_)           # 整个训练集的样本均值
print('unique targets: ', clf.classes_)           # 训练集的所有不同分类

x = np.array([-0.8, -1])                          # 待预测的向量x
mds = []
for i in range(len(clf.means_)):
    # 计算x与各分类均值向量的距离
    mds.append(ssd.mahalanobis(x, clf.means_[i], sl.inv(clf.covariance_)))
sortedIndices = np.argsort(mds)
predictResult = clf.classes_[sortedIndices[0]]    # 选取其中最近的一个作为分类结果
print('vector to be predicted: ', x)
print('distances to means: ', mds)
print('predict result: ', predictResult)
print('predect result by sklearn: ', clf.predict(x.reshape(1, -1)))

def generate_dataset(fixedCov=True):
    '''
    生成两组二元高斯分布的数据集，每一组带有自己的分类标签
    '''
    mu1 = np.array([0, 0])                    # 分类一的均值向量
    mu2 = np.array([1, 1])                    # 分类二的均值向量
    C = np.array([[0., -0.23], [0.83, .23]])  # 一个线性变换的矩阵，赋予各特征列相关关系，不再是相互独立的高斯分布
    n = 100                                   # 每一组中的样本数量
    dim = 2                                   # 特征列的数目，本例中是二元分布
    # 生成多元高斯分布样本数据的步骤
    # 1. 每一列生成独立的一元高斯分布数据
    # 2. 组合起来后，乘以预先定义好的线性变换矩阵。 n*p * p*p = n*p
    group_1 = np.dot(np.random.randn(n, dim), C) + mu1 # 生成分类一的样本
    group_2 = []
    if fixedCov:
        group_2 = np.dot(np.random.randn(n, dim), C) + mu2   # 生成分类二的样本, 与分类一共享协方差矩阵，只是均值不一样
    else:
        group_2 = np.dot(np.random.randn(n, dim), C.T) + mu2 
    # 将两个分类的数据组合起来，各赋予分类标签0和1
    X = np.r_[group_1, group_2]  # 按第一个维度连接起来，对一维数组来说就是append，对二维数组来说就是按行堆叠
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y

# QDA sample
print('{0:-^60}'.format('Quadratic Discriminant Analysis'))
X, y = generate_dataset(False)
qda = sda.QuadraticDiscriminantAnalysis(store_covariances=True)
y_pred = qda.fit(X, y).predict(X)                            
X0, X1 = X[y == 0], X[y == 1]                               # X中属于两个分类的点集
alpha = 0.5 
plt.plot(X0[:, 0], X0[:, 1], 'o', alpha=alpha, color='red')
plt.plot(X1[:, 0], X1[:, 1], 'o', alpha=alpha, color='blue')
# class 0 and 1 : areas
nx, ny = 200, 100
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
Z = qda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)
plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')
plt.show()
    
    
    
