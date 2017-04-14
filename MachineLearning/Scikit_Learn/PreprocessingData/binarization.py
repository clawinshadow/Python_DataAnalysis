import numpy as np
import sklearn.preprocessing as sp

'''
Binarization相当于某种两极分化，它有一个参数叫threshold, 整个数据集里高于这个阈值的都格式化为1
低于这个阈值的都格式化为0，简单粗暴.
Binarizer类也是无状态的类，fit does nothing
'''

X = np.array([[1, -1, 2],
              [2, 0, 0],
              [0, 1, -1]], dtype='float64')
binarizer = sp.Binarizer().fit(X) # 默认的threshold=0
print('binarizer: ', binarizer)
print('X: \n', X)
print('Binarize X: \n', binarizer.transform(X))

binarizer = sp.Binarizer(threshold=1.1)
print('Binarize X with threshold=1.1: \n', binarizer.transform(X))
