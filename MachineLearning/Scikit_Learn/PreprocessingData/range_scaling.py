import numpy as np
import sklearn.preprocessing as sp

'''
Scaling features to a range:
1. MinMaxScaler:
   将一组观测值散布到[min, max]区间内，一般情况下是[0, 1]区间
   令 A:[a1, a2, a3, ..., an], Min(A)是A的最小值，Max(A)是A的最大值，则算法为：
   a[i] = (a[i] - Min(A)) / (Max(A) - Min(A))

   如果区间不是[0, 1]，则：
   a[i] = (a[i] in [0, 1]) * (max - min) + min

2. MaxAbsScaler:
   将一组观测值散布到[-1, 1]区间内，以0为中心分别散布到[-1, 0]和[0, 1]区间内
   取最大的绝对值为Max值，然后以0为最小值分别进行正负值的标准化
   令 A: [a1, a2, a3, ..., an], Max(Abs(A))为A每个元素取绝对值后的最大值

   a[i] = a[i] / Max(Abs(A))

3. 如果想要剔除某些离群点(outlier)后再进行标准化，可以考虑使用RobustScaler
'''

X = np.array([[1, -1, 2],
              [2, 0, 0],
              [0, 1, -1]], dtype='float64')
scaler = sp.MinMaxScaler() # 默认是[0, 1]区间
print('X: \n', X)
print('Scale X to range[0, 1]: ', scaler.fit_transform(X))
# 上面的X相当于training data, fit完了后scaler会记住每个feature的min和max
# 然后这个scaler再用于标准化其它数据时，将不保证所得的结果均在[0, 1]范围内
X_test = np.array([[-3, -1, 4]], dtype='float64')
X_test_scale = scaler.transform(X_test)
print('X_test: \n', X_test)
print('Scale X_test based on training data X: \n', X_test_scale)
print('Undo the scaling: ', scaler.inverse_transform(X_test_scale))
print('data_min_: ', scaler.data_min_)      # 每一列的最小值
print('data_max_: ', scaler.data_max_)      # 每一列的最大值
print('data_range_: ', scaler.data_range_)  # 每一列的取值范围
print('scale_: ', scaler.scale_)            # 每一列的缩放比例 1/data_range_

print('{0:-^90}'.format('Seperate Line'))

# 通过设置feature_range参数将range改为[0, 2]
scaler2 = sp.MinMaxScaler(feature_range=(0, 2))
print('Scale X to range[0, 2]: \n', scaler2.fit_transform(X)) # should be 2 times greater than scaler1
print('{0:-^90}'.format('Seperate Line'))

# MaxAbsScaler
max_abs_scaler = sp.MaxAbsScaler()
X_MaxAbs = max_abs_scaler.fit_transform(X)
print('MaxAbs Scale X to range[-1, 1]: ', X_MaxAbs)
print('max_abs_: ', max_abs_scaler.max_abs_)   # 每一列的最大绝对值
print('scale_: ', max_abs_scaler.scale_)       # 每一列的缩放比例 = max_abs_
