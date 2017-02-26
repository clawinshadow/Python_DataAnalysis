import numpy as np

print('''
When operating on two arrays, NumPy compares their shapes element-wise.
It starts with the trailing dimensions, and works its way forward.
Two dimensions are compatible when
1. they are equal,
2. one of them is 1


''')

x = np.arange(4)
xx = x.reshape(4,1)
y = np.ones(5)
z = np.ones((3,4))

print('x : ', x)
print('xx : \n', xx)
print('y : ', y)
print('z : \n', z)

try:
    x + y
except ValueError as err:
    print('x + y: ', err)

print('xx + y: \n',  xx + y)
print('x + z: \n', x + z)

print('{0:-^60}'.format('Seperate Line'))
a = np.array([0.0, 10.0, 20.0, 30.0])
b = np.array([1.0, 2.0, 3.0])
print('a : ', a)
print('b : ', b)
# (4, 1) + (3) => (4, 3)
print('a[:, np.newaxis] + b: \n', a[:, np.newaxis] + b)