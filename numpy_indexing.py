import numpy as np

x = np.arange(10)
print('x : ', x)
print('x[2]: ', x[2])
print('x[-2]: ', x[-2])
print('x[2:5]: ', x[2:5])
print('x[:-3]: ', x[:-3])
print('x[1:8:2]: ', x[1:8:2])
print('x[np.array([3,3,-3,5])]: ', x[np.array([3,3,-3,5])])

print('{0:-^60}'.format('Seperate Line'))
x.shape = (2, 5)
print('x : \n', x)
print('x[1,2]: ', x[1, 2])
print('x[0]: ', x[0])  # for A(m * n * t), A[0]: the first (n * t)
print('x[1, 0:4:2]: ', x[1, 0:4:2])

print('{0:-^60}'.format('Seperate Line'))
y = np.arange(35).reshape(5, 7)
print('y : \n', y)
print('y[np.array([0, 2, 4])]: \n', y[np.array([0, 2, 4])])
# select 1#, 2#, 3# elements from 0#, 2#, 4# rows, respectively
print('y[np.array([0, 2, 4]), np.array([1, 2, 3])]: ', y[np.array([0, 2, 4]), np.array([1, 2, 3])])

try:
    print('y[np.array([0, 2, 4]), np.array([1, 2])]: ', y[np.array([0, 2, 4]), np.array([1, 2])])
except IndexError as err:
    print('y[np.array([0, 2, 4]), np.array([1, 2])]: \n', str(err))

print('y[np.array([0, 2, 4]), 1]: ', y[np.array([0, 2, 4]), 1])
print('y[np.array([0, 2, 4]), 1:3]: \n', y[np.array([0, 2, 4]), 1:3])


