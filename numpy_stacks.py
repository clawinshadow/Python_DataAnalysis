import numpy as np

# hstack() horizontal stack, equals to concatenate(tup, axis = 1)
a = np.arange(9).reshape(3, 3)
b = 2 * a
c = np.hstack((a, b))
print('a: \n', a)
print('b: \n', b)
print('np.hstack(a, b): \n{0}\nshape: {1}'.format(c, c.shape))
print('{0:-^60}'.format('Seperate'))

a = np.arange(2 * 3 * 4).reshape(2, 3, 4)
b = 2 * a
c = np.hstack((a, b))
print('a: \n', a)
print('b: \n', b)
print('np.hstack(a, b): \n{0}\nshape: {1}'.format(c, c.shape))

# vstack() vertical stack, equals to concatenate(tup, axis = 0)
print('{0:-^60}'.format('Seperate'))
a = np.arange(6).reshape(2, 3)
b = 2 * a
c = np.vstack((a, b))
print('a: \n', a)
print('b: \n', b)
print('np.vstack(a, b): \n{0}\nshape: {1}'.format(c, c.shape))

print(
    '''
    hstack() and vstack() are retained for backward compatibility,
    it's recommended to use concatenate() instead
    for axis, if a.shape = b.shape = (m * n * t), then
    concatenate((a,b), axis=0) => result.shape: (2m * n * t)
    concatenate((a,b), axis=1) => result.shape: (m * 2n * t)
    concatenate((a,b), axis=2) => result.shape: (m * n * 2t)
    ''')
a = np.arange(2 * 2 * 2).reshape(2, 2, 2)
b = a * 2
c1 = np.concatenate((a, b), axis=0)
c2 = np.concatenate((a, b), axis=1)
c3 = np.concatenate((a, b), axis=2)
print('a: \n', a)
print('b: \n', b)
print('np.concatenate((a, b), axis=0): \n{0}\nshape: {1}'.format(c1, c1.shape))
print('np.concatenate((a, b), axis=1): \n{0}\nshape: {1}'.format(c2, c2.shape))
print('np.concatenate((a, b), axis=2): \n{0}\nshape: {1}'.format(c3, c3.shape))

print('''
dstack(): Takes a sequence of arrays and stack them along the third axis to make a single array.
All of them must have the same shape along all but the third axis.
a.shape = b.shape = (m * n * t)， then dstack(a, b).shape = (m * n * t * 2)
''')
a = np.array((1, 2, 3))
b = np.array((2, 3, 4))
c = np.dstack((a, b))
print('a: \n', a)
print('b: \n', b)
print('np.dstack(a, b): \n{0}\nshape: {1}'.format(c, c.shape))

# column_stack() and row_stack
a = np.array((1, 2, 3))
b = np.array((2, 3, 4))
c = np.column_stack((a, b))
print('a: \n', a)
print('b: \n', b)
print('np.column_stack(a, b): \n{0}\nshape: {1}'.format(c, c.shape))

d = np.row_stack((a, b))
d2 = np.vstack((a, b))
print('np.row_stack(a, b): \n{0}\nshape: {1}'.format(d, d.shape))
print('row_stack <=> vstack \n', d == d2)