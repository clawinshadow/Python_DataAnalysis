import numpy as np

'''
numpy.nonzero(a): 返回数组a中不为零的所有元素的索引，这个索引的shape并不与a完全一致，而是采取tuple(arrays)的方式，
tuple里面的每个array都是一维数组，存储a中每一个维度不为零元素的索引，比如a为二维数组，则tuple里面有两个一维数组，
第一个为行索引，第二个为列索引

numpy.flatnonzero(a): 将n维数组a展开后所有不为零的元素的索引，比如3*3单位矩阵，展开后就是[1,0,0,0,1,0,0,0,1],
所有不为零的元素的索引即是[0, 4, 8]

'''

x = np.eye(3)
nonzeroIndices = np.nonzero(x)
print('x: \n', x)
print('np.nonzero(x): \n', nonzeroIndices) # tuple(row indices, column indices)
print('np.flatnonzero(x): \n', np.flatnonzero(x))
print('nonzero counts: ', np.count_nonzero(x))

