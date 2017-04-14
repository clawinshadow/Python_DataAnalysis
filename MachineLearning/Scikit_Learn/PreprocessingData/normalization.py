import numpy as np
import sklearn.preprocessing as sp

'''
normalization是将数据归一化，比如在计算标准正交基的时候，将每个列向量归一化
令列向量 A: [a1, a2, a3, ..., an], 则算法为：

a[i] = a[i] / norm(A)

这个norm可以是向量的任意范数，但一般是欧几里得范数
'''

X = np.array([[1, -1, 2],
              [2, 0, 0],
              [0, 1, -1]], dtype='float64')
X_normalized = sp.normalize(X, norm='l2') # 用L2范数来计算向量的范数
print('X: \n', X)
print('X after normalization: \n', X_normalized)
print('{0:-^70}'.format('Seperate Line'))

# 也可以使用Normalizer, 但是与StandardScaler/MinMaxScaler..等有状态的类不同
# Normalizer是一个无状态的类，它的fit什么都没做，不记录任何训练集的状态
normalizer = sp.Normalizer().fit(X)         # fit does nothing
print('Normalizer: ', normalizer)
X_normalized = normalizer.transform(X)
print('X after normalization use Normalizer class: \n', X_normalized)
X_test = np.array([[-1, 1, 0]], dtype='float64')
X_test_scale = normalizer.transform(X_test) # 只根据当前测试数据的范数来归一化，与训练集X无关
print('X_test: \n', X_test)
print('Normalize X_test: \n', X_test_scale)
