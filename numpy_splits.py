import numpy as np

a = np.arange(9).reshape(3, 3)
b = np.hsplit(a, 3)
print('a: \n', a)
print('np.hsplit(a, 3): \n{0}\n'.format(b))

b = np.vsplit(a, 3)
print('a: \n', a)
print('np.vsplit(a, 3): \n{0}\n'.format(b))

print('''
hsplit() and vsplit() can be replaced by split(), just different by axis parameter,
hspliat => axis = 1, vsplit => axis = 0''')
b = np.split(a, 3, axis=1)
print('np.split(a, 3, axis=1): \n', b)

b = np.split(a, 3, axis=0)
print('np.split(a, 3, axis=0): \n', b)

print('''
np.dsplit(): if A.shape:(m * n * t), then after dsplit(p), it's (m * n * t / p)\n''')

a = np.arange(27).reshape(3, 3, 3)
b = np.dsplit(a, 3)
print('a: \n', a)
print('np.dsplit(a, 3): \n', b)
