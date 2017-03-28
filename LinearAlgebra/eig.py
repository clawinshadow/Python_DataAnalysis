import numpy as np
import scipy.linalg as sl

'''
计算n*n矩阵的特征值和特征向量，eig方法中提供了两个可选参数left和right
分别用于计算矩阵的左特征向量和右特征向量

1. 右特征向量：
   即 A*Xr = λr*Xr, 此时Xr位于A的右边，是一个n维列向量，所以称为右特征向量
   此时 det(A - λr*I) = 0
   绝大多数情况下我们都是求的此种特征向量
   
2. 左特征向量：
   即 XL*A = λr*XL, 此时XL位于A的左边，是一个n维行向量，称为左特征向量
   将方程左右边同时取转置，可得到 det(A.T - λr*I) = 0
   
'''

A = np.array([[3, 2], [3, -2]])
print('A: \n', A)
w, vr = sl.eig(A)
print('eigen values: ', w)
# 默认返回的是标准化后的特征向量，即2-范数为1
print('normalized right eigenvectors: \n', vr)

B = A.T
print('B: \n', B)
w, vl = sl.eig(B, left=True, right=False)
print('eigen values: ', w)
# 求左特征向量，因为B是A的转置，结果应该与A一样
print('normalized left eigenvectors: \n', vl)

# 可用于复特征值
C = np.array([[1, 2], [-2, 1]])
print('C: \n', C)
w, vr = sl.eig(C)
print('eigen values: ', w)
print('normalized right eigenvectors: \n', vr)
