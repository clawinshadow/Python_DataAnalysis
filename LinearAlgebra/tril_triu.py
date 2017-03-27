import numpy as np
import scipy.linalg as sl

A = np.arange(16).reshape(4, 4)
print('A: \n', A)
# k = 0 时，保留对角线元素，对角线上方的所有元素格式化为0
print('tril(A, 0): \n', sl.tril(A))
# k = -1 时，包括对角线元素都被格式化为0
print('tril(A, -1): \n', sl.tril(A, -1))
# k = 1 时，边界上移到对角线上方的一格
print('tril(A, 1): \n', sl.tril(A, 1))

# triu 与 tril相反
# k = 0 时，保留对角线元素，对角线下方的所有元素格式化为0
print('triu(A, 0): \n', sl.triu(A))
# k = 1 时，包括对角线元素都被格式化为0
print('triu(A, 1): \n', sl.triu(A, 1))
# k = -1 时，边界下移到对角线下方的一格
print('triu(A, -1): \n', sl.triu(A, -1))

# 不管是tril还是triu，k的正值永远是把边界往上移，为负时往下移
