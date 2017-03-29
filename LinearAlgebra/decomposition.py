import numpy as np
import scipy.linalg as sl

# 应该是使用的改进的Gram-Schmidt正交化过程
A = np.array([[1, -1, 4], [1, 4, -2], [1, 4, 2], [1, -1, 0]])
print('A: \n', A)
Q, R = sl.qr(A, mode='economic')
print('QR decomposition of A: ')
print('Q: \n', Q)
print('R: \n', R)
print('{0:-^60}'.format('Seperate Line'))

# LU分解，scipy里面的lu分解并不是朴素的高斯消元法
# 而是改进后的部分主元高斯消去法，包含置换矩阵P
# 每进行下一步消去操作前，必须检查每一行主元的大小然后进行必要的置换
# 即PA = LU分解，详见 数值分析 一书 Page.86
# 而scipy里面的P是放在等式右边的，即A = PLU，那么这里面的P应该是P的逆矩阵，
# 即P的转置
A = np.array([[2, 1, 5], [4, 4, -4], [1, 3, 1]])
print('A: \n', A)
p, l, u = sl.lu(A)
print('LU decomposition of A: ')
print('P: \n', p)
print('L: \n', l)
print('U: \n', u)
print('A == PLU: ', np.allclose(A, np.dot(np.dot(p, l), u)))
print('{0:-^60}'.format('Seperate Line'))
# permute_l = True时，返回结果将P和L合并了
pl, u = sl.lu(A, permute_l=True)
print('LU decomposition of A: ')
print('P*L: \n', pl)
print('U: \n', u)
print('A == PLU: ', np.allclose(A, np.dot(pl, u)))
print('{0:-^60}'.format('Seperate Line'))

# Cholesky分解，用于分解对称正定阵
# 1. 将A进行普通的LU分解， A = LU
# 2. 将U分解为对角阵D乘以L.T, A = L*D*L.T
# 3. 将D开方得到D`，那么A = L*D`*D`*L.T
#    因为D`为对角阵，所以D`=D`.T, 令L`=L*D`,则 A = L`*L`.T
# 这个L`是一个三角矩阵，即为楚列斯基分解的返回值
A = np.array([[4, 2, -2], [2, 10, 2], [-2, 2, 5]])
print('A: \n', A)
c = sl.cholesky(A)
print('Cholesky decomposition of A: \n', c)

# SVD, 奇异值分解，假设A为 m * n矩阵，Rank(A) = r
# A = U * s * V.T,
# 1. 先计算A.T*A 的特征值与特征向量
# 2. V为标准化的 n*n 特征向量矩阵
# 3. s为分解出来的奇异值矩阵，shape与A相同, 每个奇异值si为特征值的开方
# 4. U最复杂，前 r 个分量是 1/si * A * Vi
#    后面 m - r 个分量是将N(A.T)标准正交化后得来
A = np.array([[1, 1], [1, 1], [0, 0]])
print('A: \n', A)
U, s, Vh = sl.svd(A)
print('Singular values decomposition of A: ')
print('U: \n', U)
print('s: \n', s) # s is an 1-D array
print('V: \n', Vh)

# diagsvd 会返回与A一致的shape
S = sl.diagsvd(s, A.shape[0], A.shape[1])
print('S: \n', S)
print('A == U*S*V.T: ', np.allclose(A, np.dot(np.dot(U, S), Vh)))


