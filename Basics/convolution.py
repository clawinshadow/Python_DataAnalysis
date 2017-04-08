import numpy as np
'''
离散卷积：向量f为m维向量，g为n维向量，则f*g为m+n-1维向量，
          若m与n不相等，则少的一方用零补齐

f(t):
    f(0) = 1, f(1) = 2, f(2) = 3
g(t):
    g(0) = 0, g(1) = 1, g(2) = 0.5

a(t) = f(t) * g(t)

a(0) = f(0) * g(0)
a(1) = f(0) * g(1) + f(1) * g(0)
a(2) = f(0) * g(2) + f(1) * g(1) + f(2) * g(0)
a(3) = f(1) * g(2) + f(2) * g(1)
a(4) = f(2) * g(2)

(a * v)[n] = \sum_{m = -\infty}^{\infty} a[m] v[n - m]
'''

f = [1, 2, 3]
g = [0, 1, 0.5]
a = np.convolve(f, g)
print('f: ', f)
print('g: ', g)
print('f*g:', a)

