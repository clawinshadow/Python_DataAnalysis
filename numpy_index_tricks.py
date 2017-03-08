import numpy as np

a = np.mgrid[-1:1:5j]  # j means total number of generated values between start and stop
print('a = np.mgrid[-1:1:5j]: ', a)
a = np.mgrid[-1:1:0.2] # 0.2 means step
print('a = np.mgrid[-1:1:0.2]: ', a)

a = np.mgrid[0:5,0:5]
print('a = np.mgrid[0:5,0:5]: \n', a)  # 一个横向的，一个纵向的

a = np.ogrid[-1:1:5j]
print('a = np.ogrid[-1:1:5j]: ', a) # same as mgrid when it's 1-D array

a = np.ogrid[0:5,0:5]
print('a = np.ogrid[0:5,0:5]: \n', a) # don't expand 
print('a[0].shape: ', a[0].shape)
print('a[1].shape: ', a[1].shape)

a = np.arange(10).reshape(2, 5)
print('a: \n', a)

b = np.ix_([0, 1], [1, 3])
print('b: \n', b)
print('a[b]: \n', a[b])  # use b as an index to fetch values in matrix a

# Translates slice objects to concatenation along the first axis.
a = np.r_[np.array([1,2,3]), 0, 0, np.array([4,5,6])]
b = np.r_[-1:1:6j, [0]*3, 5, 6]
print('a = np.r_[np.array([1,2,3]), 0, 0, np.array([4,5,6])]:\n', a)
print('b = np.r_[-1:1:6j, [0]*3, 5, 6]:\n', b)

a = np.array([[0, 1, 2], [3, 4, 5]])
print('a: \n', a)
b = np.r_['1', a, a]   # concatenate along second axis
print('b = np.r_[''1'', a, a]: \n', b)
b = np.r_['0', a, a]   # concatenate along first axis
print('b = np.r_[''0'', a, a]: \n', b)

print(np.r_['0', [1,2,3],[4,5,6]])
print(np.r_['0, 2', [1,2,3],[4,5,6]])  # concatenate along first axis, force dim>=2
print(np.r_['0, 3', [1,2,3],[4,5,6]])  # force dim >= 3

print(np.r_['r',[1,2,3], [4,5,6]])  # 1 x N (row) matrix is produced
print(np.r_['c',[1,2,3], [4,5,6]])  # N x 1 (column) matrix is produced

di = np.diag_indices(4)
print('di = np.diag_indices(4): \n ', di)
a = np.arange(16).reshape(4, 4)
a[di] = 100
print('a: \n', a)


