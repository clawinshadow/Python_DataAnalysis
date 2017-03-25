import numpy as np
import math
from numpy import linalg as LA

'''
关于norm()函数的demo

ord	norm for matrices	    norm for vectors
None	Frobenius norm	                2-norm
‘fro’	Frobenius norm	                  –
‘nuc’	nuclear norm	                  –
inf	max(sum(abs(x), axis=1))	max(abs(x))
-inf	min(sum(abs(x), axis=1))	min(abs(x))
0	–	sum(x != 0)
1	max(sum(abs(x), axis=0))	as below
-1	min(sum(abs(x), axis=0))	as below
2	2-norm (largest sing. value)	as below
-2	smallest singular value	as below

 Frobenius norm：矩阵中每个元素的平方和，对于向量来说就是二维范数
 Nuclear norm: 矩阵奇异值之和
'''

a = np.arange(9) - 4
print('a: ', a)
b = a.reshape((3, 3))
print('b: \n', b)

print('LA.norm(a): ', LA.norm(a))
print('math.sqrt(np.sum(a**2)): ', math.sqrt(np.sum(a**2)))
print('LA.norm(b): ', LA.norm(b))
print('LA.norm(a, ord=\'fro\'): ', LA.norm(b, ord='fro')) # these four are equal

# 向量的无穷范数即向量的切比雪夫距离，取每一个分量绝对值的最大值
print('LA.norm(a, np.inf): ', LA.norm(a, np.inf))
# 矩阵的无穷范数是对于每一行的绝对值之和，取最大值 4 + 3 + 2
print('LA.norm(b, np.inf): ', LA.norm(b, np.inf)) 

# 负无穷的范数貌似只是np的一个diy功能，数学定义里面没看到过
# 与正无穷范数的概念相反，都是取最小值
print('LA.norm(a, -np.inf): ', LA.norm(a, -np.inf))
print('LA.norm(b, -np.inf): ', LA.norm(b, -np.inf))

# 向量的一维范数即曼哈顿距离，各分量的绝对值之和
# 矩阵的一维范数是对于每一列的绝对值之和，取最大值 4 + 1 + 2
print('LA.norm(a, 1): ', LA.norm(a, 1))
print('LA.norm(b, 1): ', LA.norm(b, 1))

# 向量的-1维范数即每个分量的-1次方之和再-1次方
# 矩阵的-1维范数与一维范数相反，每一列的绝对值之和的最小值 3 + 3
print('LA.norm(a, -1): ', LA.norm(a, -1))
print('LA.norm(b, -1): ', LA.norm(b, -1))

# 向量的n维范数即每个分量绝对值的n次方之和再开n次方
print('LA.norm(a, 2): ', LA.norm(a, 2))
print('math.pow(np.sum(abs(a)**2), 1/2): ', math.pow(np.sum(abs(a)**2), 1/2))
print('LA.norm(a, 3): ', LA.norm(a, 3))
print('math.pow(np.sum(abs(a)**3), 1/3): ', math.pow(np.sum(abs(a)**3), 1/3))

# using the axis argument
c = np.array([[ 1, 2, 3],
              [-1, 1, 4]])
print('LA.norm(c, axis=0): ', LA.norm(c, axis=0)) # 计算每一列的范数
print('LA.norm(c, axis=1): ', LA.norm(c, axis=1)) # 计算每一行的范数


