import numpy as np
import sklearn.preprocessing as sp

'''
Imputation of missing values, 是将某些invalid的数据用更为合理的数据来替代，比如将NaN替代为本列其它观测值的
平均值，或者中位数...等等
'''

X = np.array([[1, 2],
              [np.nan, 3],
              [7, 6]])
# missing_values参数指定待替换的值, 必须为一个数值或者'NaN'
# strategy参数指定用何种策略来替换invalid values, 有'mean', 'median', 'most_frequent'
# axis指定维度
imp = sp.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
print('X: \n', X)
print('Imputation of X: \n', imp.transform(X)) # np.nan -> (1 + 7) / 2

# 用其余观测值的中位数来替代'inf'
X = np.array([[1, 2],
              [np.inf, 3],
              [2, 6],
              [3, 7],
              [5, 10]])
print('{0:-^60}'.format('Seperate Line'))
imp = sp.Imputer(missing_values=np.inf, strategy='median', axis=0)
imp.fit(X)
print('X: \n', X)
print('Imputation of X: \n', imp.transform(X)) # np.nan -> (2 + 3) / 2
print(imp.get_params())  # a sample of get_params() method which belongs to every estimator
