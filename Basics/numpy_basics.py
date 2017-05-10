import numpy as np

a = np.arange(5)
print(a)
print('array element data type: ', a.dtype)
print('array shape: ', a.shape)

b = np.array([np.arange(3), np.arange(3)])
print('shape of \n{0} is: {1}'.format(b, b.shape))

c = np.array([[1, 2], [3, 4]])
print(c)
print(c[0, 0], c[0, 1], c[1, 0], c[1, 1])

print(np.float64(64))
d = np.arange(3, dtype=np.int64)
print('item size of int64: ', d.itemsize)

d = np.arange(3, dtype='f')
print(d)
print(np.sctypeDict.keys())

t = np.dtype('Float64')
print('code of "Float64" is: ', t.char)
print('str of "Float64" is: ', t.str) # > or < means big-endian or little-endian

# slicing
a = np.arange(10)
print(a[3:7])
print(a[::-1])
print(a[3:7:-1]) # empty array, because start index 3 < end index 7
print(a[7:3:-1]) # when step < 0, start index must greater than end index
print(a[:7:2])

b = np.arange(2*3*4).reshape((2, 3, 4))
print(b)
#  flatten always returns a copy and ravel returns a view of the original array whenever possible
print(b.ravel())
print(b.flatten())
c = b.ravel()
c[0] = 100
print(b)

b[0, 0, 0] = 0
print(b)

c = b.flatten()
c[0] = 100
print(b)

b.shape = (6, 4)
print(b)
print(b.transpose())

# reshape don't change the original array, but resize() does
print(b.reshape((3, 8)))
print(b)
b.resize((3, 8))  # resize() return None, reshape() return a copy of array
print(b)
