import numpy as np

a = np.arange(24).reshape(2, 12)
print('a: \n', a)
print('a.ndim: ', np.ndim(a))
print('a.size: ', np.size(a))
print('a.itemsize: ', a.itemsize)
print('a.nbytes: ', a.nbytes)  # a.size * a.itemsize

a.resize((4, 6))
print('a: \n', a)
print('a.T: \n', a.T)

b = np.array([1 + 2j, 3 + 4j])
print('b: \n', b)
print('b.real: ', b.real)
print('b.imag: ', b.imag)
print('b.dtype: ', b.dtype)
print('b.tolist: ', b.tolist())
print('b.astype(int): ', b.astype(int))  # discard the imag part

c = np.arange(4).reshape(2, 2)
print('c: \n', c)
c.flat = 7
print('c after c.flat = 7: \n', c)