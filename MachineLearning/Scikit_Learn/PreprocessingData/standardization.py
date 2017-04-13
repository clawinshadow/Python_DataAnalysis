import numpy as np
from sklearn import preprocessing

'''
Standardization是将一组数据按高斯分布标准化，不管它的实际分布是什么样的，都幻化成高斯分布的典型特征
即均值为0，标准差为1。所以这个标准化过程也称之为mean removal & variance scaling, 即均值移除与方差缩放
假设数组A：[a1, a2, a3, ..., an], A的均值为Mean(A), 方差为Var(A), 具体算法如下：
          a[i] - Mean(A)
a[i] = ---------------------
         math.sqrt(Var(A))

if with_mean = False, 则 a[i] = a[i] / math.sqrt(Var(A))
elif with_std = False, 则 a[i] = a[i] - Mean(A)

doc: http://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
'''

X = np.array([[1, -1, 2],
              [2, 0, 0],
              [0, 1, -1]], dtype='float64')
X_scaled = preprocessing.scale(X)
print('X: \n', X)
print('X after standardization: \n', X_scaled)
# verify standardization process
print('mean of each column of standardized X: ', X_scaled.mean(axis=0))
print('std of each column of standardized X: ', X_scaled.std(axis=0))

# Use StandardScaler instead, 能保存训练集中均值和方差等状态，再将其用于测试集数据的标准化
# 可用于pipeline，并且功能更为丰富
print('{0:-^60}'.format('SeperateLine'))
scaler = preprocessing.StandardScaler().fit(X)
print(scaler)
print('scaler.mean_:', scaler.mean_)   # 训练集的均值
print('scaler.scale_:', scaler.scale_) # 训练集的方差
print('Use scaler to transform X: \n', scaler.transform(X))
testData = np.array([[-1, 1, 0]], dtype='float64')
print('test data: ', testData)
print('Use scaler to transform test data: ', scaler.transform(testData)) # 标准化测试数据

print('{0:-^60}'.format('SeperateLine'))

# with_mean = False, 不管均值，只将方差缩放
scaler_without_mean = preprocessing.StandardScaler(with_mean=False)
print('scaler without mean: ', scaler_without_mean)
print('Use scaler without mean to transform X: \n', scaler_without_mean.fit_transform(X))
print('{0:-^60}'.format('SeperateLine'))

# with_std = False, 不管方差，只移除均值
scaler_without_std = preprocessing.StandardScaler(with_std=False)
print('scaler without std: ', scaler_without_std)
print('Use scaler without std to transform X: \n', scaler_without_std.fit_transform(X))
