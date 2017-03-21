import scipy.linalg as sl
import numpy as np
'''
a : (N, N) array_like
    Square input data
    
b : (N, NRHS) array_like
    Input data for the right hand side.
    
sym_pos : bool, optional
    Assume a is symmetric and positive definite.
    This key is deprecated and assume_a = ‘pos’ keyword is recommended instead.
    The functionality is the same. It will be removed in the future.
    
lower : bool, optional
    If True, only the data contained in the lower triangle of a.
    Default is to use upper triangle. (ignored for 'gen')
    
overwrite_a : bool, optional
    Allow overwriting data in a (may enhance performance). Default is False.
    
overwrite_b : bool, optional
    Allow overwriting data in b (may enhance performance). Default is False.

unit_diagonal : bool, optional
    If True, diagonal elements of a are assumed to be 1 and will not be referenced.

check_finite : bool, optional
    Whether to check that the input matrices contain only finite numbers.
    Disabling may give a performance gain,
    but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs.
'''

def printv(A, b, x):
    print('{0:-^60}'.format('Seperate Line'))
    print('A: \n', A)
    print('b: ', b)
    print('Solve Ax = b: ', x)
    

A = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
b = np.array([2, 4, -1])
x = sl.solve(A, b)
printv(A, b, x)

A = np.array([[5, 3], [3, 6]])  # A是一个对称正定矩阵
b = np.array([8, 9])
# 当sym_pos被设置成true时，应该会使用楚列斯基分解来求解，提高速度
x = sl.solve(A, b, sym_pos=True) 
printv(A, b, x)

A = np.array([[1, 1, 1], [2, 1, 1], [3, 0, 1]])
b = np.array([3, 2, 1])
# 仅使用矩阵的上三角部分求解，如果lower=True则为下半部分，其余的忽略
x = sl.solve_triangular(A, b)  
printv(A, b, x)

A = np.array([[3, 1, 1], [0, 3, 1], [0, 0, 3]])
b = np.array([3, 2, 1])
# 当unit_diagonal为True时，所有对角元素均被替换成1然后进行求解
x = sl.solve_triangular(A, b, unit_diagonal=True)  
printv(A, b, x)
