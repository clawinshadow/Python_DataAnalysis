import numpy as np
import scipy.sparse as sp

'''
关于scipy中处理稀疏矩阵的一些方法，具体原理见.\Translation\SparseMatrix.md
'''

# 关于CSC算法，原理不赘述
# 优点：
#     1. 高效的算术运算: CSC + CSC, CSC * CSC
#     2. 高效的column slicing, 列切片
#     3. 快速的矩阵向量内积(但是CSR, BSR或许要更快)
# 缺点：
#     1. row slicing 行切片会很慢
#     2. 转化为LIL或者DOK时代价会比较昂贵
print('{0:-^70}'.format('CSC'))
M = np.array([[0, 0, 0, 0],
              [5, 8, 0, 0],
              [0, 0, 3, 0],
              [0, 6, 0, 0]])
M_csc = sp.csc_matrix(M)
print('M: \n', M)
print('M_csc object: ', M_csc)
print('shape of M_csc: ', M_csc.shape)
print('nnz of M: ', M_csc.nnz)              # 非零元素的个数
print('data of M_csc: ', M_csc.data)        # 所有的非零元素     => A数组
print('indices of M_csc: ', M_csc.indices)  # 非零元素的行索引   => JA数组
print('indptr of M_csc: ', M_csc.indptr)    # 非零元素的索引指针 => IA数组
print('convert M_csc back to M: \n', M_csc.toarray())

# 第二种构造方式，M[row[i], col[i]] = data[i]
# CSR也有此种构造方式
row = np.array([0, 2, 2, 0, 1, 2])
col = np.array([0, 0, 1, 2, 2, 2])
data = np.array([1, 2, 3, 4, 5, 6])
M2 = sp.csc_matrix((data, (row, col)), shape=(3, 3)).toarray()
print('row: ', row)
print('col: ', col)
print('data: ', data)
print('Construct sparst matrix with row, col and data: \n', M2)

# 关于CSR算法，原理也不赘述
# 优点：
#     1. 高效的算术运算: CSR + CSR, CSR * CSR
#     2. 高效的row slicing, 行切片
#     3. 快速的矩阵向量内积
# 缺点：
#     1. column slicing 列切片会很慢
#     2. 转化为LIL或者DOK时代价会比较昂贵
print('{0:-^70}'.format('CSR'))
M_csr = sp.csr_matrix(M)
print('M: \n', M)
print('M_csr object: ', M_csr)
print('shape of M_csr: ', M_csr.shape)
print('nnz of M: ', M_csr.nnz)              # 非零元素的个数
print('data of M_csr: ', M_csr.data)        # 所有的非零元素     => A数组
print('indices of M_csr: ', M_csr.indices)  # 非零元素的列索引   => JA数组
print('indptr of M_csr: ', M_csr.indptr)    # 非零元素的索引指针 => IA数组
print('convert M_csr back to M: \n', M_csr.toarray())
# 每种算法之间都可以互相转化的，例如CSR转化成CSC
M_csc = M_csr.tocsc(copy=True)
print('data of M_csc converted from M_csr: ', M_csc.data)       
print('indices of M_csc converted from M_csr: ', M_csc.indices)  
print('indptr of M_csc converted from M_csr: ', M_csc.indptr)

# Dok基本表示法
# 优点是访问元素特别快，O(1)，并且能很高效的转化为coo_matrix
print('{0:-^70}'.format('DOK'))
M_dok = sp.dok_matrix(M)
print('M: \n', M)
print('dok of M: ', M_dok)

# LIL基本表示法
# 优点：
#     1. 支持很灵活的切片
#     2. 转化为其它各类稀疏矩阵结构都很方便
# 缺点：
#     1. 算术运算比如 LIL + LIL 很慢 （要快的话考虑CSC和CSR）
#     2. 很慢的列切片 （考虑CSC）
#     3. 矩阵内积运算很慢 （还是考虑CSR和CSC）
# 主要运用目的：
#     1. 很方便用于构建稀疏矩阵
#     2. LIL创建好后，转化成CSR和CSC用于算术运算也很高效
#     3. 当需要创建非常大的矩阵时，要考虑使用COO格式
print('{0:-^70}'.format('LIL'))
M_lil = sp.lil_matrix(M)
print('M: \n', M)
print('lil of M: \n', M_lil)
print('rows of M_lil: ', M_lil.rows)  # 存储每一行非零元素的列索引
print('data of M_lil: ', M_lil.data)  # 存储每一行的非零元素

# COO基本表示法，其实很简单的，就相当于CSC的第二种构造形式，一样的
# 优点：
#     1. 转化为其它各类稀疏矩阵结构都很方便
#     2. 支持重复的入口，DOK不支持，因为后者是key-value形式的
#     3. 转化成CSR/CSC会非常快
# 缺点：
#     1. 不支持算术运算
#     2. 不支持切片
print('{0:-^70}'.format('COO'))
row  = np.array([0, 3, 1, 0])
col  = np.array([0, 3, 1, 2])
data = np.array([4, 5, 7, 9])
coo = sp.coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
print('row: ', row)
print('col: ', col)
print('data: ', data)
print('Construct sparst matrix through coo_matrix: \n', coo)
