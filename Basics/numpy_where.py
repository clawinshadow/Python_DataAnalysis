import numpy as np

'''
np.where(condition[, x, y]): 相当于数组层面上的三元表达式，condition不必多说是bool型的条件，x和y都是可选参数，如果都没有，则该方法
等同于nonzero，返回满足condition的元素索引，也是以tuple(array)的方式来返回，如果x和y都有，则相当于 condition ?? x : y
如果满足condition则输出x指定的数据变换，否则输出y中指定的变换
'''

condition = [[True, False], [True, True]]
x = [[1, 2], [3, 4]]
y = [[9, 8], [7, 6]]
print('condition: ', condition)
print('x: ', x)
print('y: ', y)
print('np.where(condition, x, y): \n', np.where(condition, x, y))

x = np.arange(9).reshape(3, 3)
print('x: ', x)
print('np.where(x > 5): ', np.where(x > 5))  # 完全等同于np.nonzero
print('np.where(x < 5, x, nan): \n', np.where(x < 5, x, np.NaN)) # 小于5的原样输出，否则都是nan

