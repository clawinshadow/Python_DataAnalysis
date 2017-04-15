import numpy as np
import sklearn.preprocessing as sp

'''
这是一个相对较复杂的处理数据的方式，将特征列按多项式展开，比如：
1. degree = 2, 则：
   (X_1, X_2) to (1, X_1, X_2, X_1^2, X_1X_2, X_2^2).
   如果interaction_only=True, 则剔除所有自己跟自己相乘的成员，比如X1**2, X2**2等等

2. degree = 3, interaction_only=True
   (X_1, X_2, X_3) to (1, X_1, X_2, X_3, X_1X_2, X_1X_3, X_2X_3, X_1X_2X_3)

'''

X = np.arange(6).reshape(3, 2)
print('X: \n', X)
poly = sp.PolynomialFeatures(degree=2)
print('Poly X: \n', poly.fit_transform(X))

print('{0:-^60}'.format('Seperate Line'))

X = np.arange(9).reshape(3, 3)
poly = sp.PolynomialFeatures(degree=3, interaction_only=True)
print('Poly X: \n', poly.fit_transform(X))
