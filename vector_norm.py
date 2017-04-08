import numpy as np

'''
demos about vector's norm: ||x||
L1: x各分量的绝对值之和
L2: x各分量的平方和再开方
Lp: x各分量的p次方和再开p次方
Linf: x各分量中绝对值最大的那个
'''

def norm(x, p=2):
    if p == 1:
        return np.sum(np.absolute(x))
    elif p == 2:
        return np.sqrt(np.sum(x**2)) # Euclidean Norm or Frobenius Norm
    elif p == np.inf:
        return np.max(np.absolute(x))
    else:
        return np.power(np.sum(x**p), 1/p)
    

x = np.array([2, 1, 3, 4])
print('x: ', x)
print('norm(x, 1): ', norm(x, 1))
print('norm(x, 2): ', norm(x, 2))
print('norm(x, 3): ', norm(x, 3))
print('norm(x, np.inf): ', norm(x, np.inf))
