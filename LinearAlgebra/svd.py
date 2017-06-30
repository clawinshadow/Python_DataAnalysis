import numpy as np
import scipy.linalg as sl

a = np.array([[1, 2, 3],
              [3, 3, 1]])
print('A: \n', a)
eigVals, U = sl.eig(np.dot(a, a.T)) # 这里得出的特征值是无序的，但我们一般要从大到小排序
print('U: \n', U) 
indices = np.argsort(eigVals)[::-1]
U = np.take(U, indices)
eigVals = np.take(eigVals, indices.reverse()) 
print('eigVals of A*A.T: ', eigVals) # 中间的 D 矩阵就是特征值再开方所得
print('U: \n', U)                    # eigen vectors of A*A.T, m * m

D = np.c_[np.diag(np.sqrt(eigVals)), np.zeros(2).reshape(-1, 1)]
print('D: \n', D)

eigVals2, V = sl.eig(np.dot(a.T, a))
indices = np.argsort(eigVals2)
V = np.take(V, indices)
eigVals2 = np.take(eigVals2, indices) 
V = V.T                              # eigen vectors of A.T*A, n * n, but need to transpose then match the SVD element
print('V.T: \n', V)
print('U*D*V.T: ', np.dot(np.dot(U, D), V))

u, d, v = sl.svd(a)
print('u by sl.svd(a): \n', u)
print('d by sl.svd(a): \n', d)
print('v by sl.svd(a): \n', v)

ad = np.c_[np.diag(d), np.zeros(2).reshape(-1, 1)] # 这里的d只是一维的数组，特征值的简单排列，实际上应该是2*3的矩阵才对

print('u * ad * v: \n', np.dot(np.dot(u, ad), v))  # 应该是等于原矩阵a的
