import scipy.optimize as so

'''
黄金分割法来进行标量函数的最小化问题, 也称为一维搜索
无约束的，unconstrained.
'''

# scipy里面的golden method即用黄金分割法来进行求解的
def fx(x):
    return x**4 - 14*x**3 + 60*x**2 -70*x

# 黄金分割法的唯一前提是f(x)在起始区间[a0, b0]内是单峰的，即只有一个最小值
# bracket为起始的搜索区间，这个可以给3个值的tuple也可以给两个值的tuple
# 如果是三个值(a, b, c)， 则必须保证f(b) < f(a), f(b) < f(c)， 即f(b)是最小的一个
# 这样就不需要算法再去搜索合理的起始区间了，最终结果一定在(a, c)内
# 如果是两个值(a, b), 那么只是给区间搜索提供了一个良好的起始值
# 然后算法需要再去使用划界法来确定合理的起始区间，保证f(x)是单峰的
# 那么显然在这种情况下，最终的x不一定在(a, b)内
# tol即停止条件的阈值大小
res = so.minimize_scalar(fx, bracket=(0, 2), method='golden', tol=1.0e-6)
print(res)

# 最终结果x在bracket之外
print('{0:-^60}'.format('Use golden method'))
res = so.minimize_scalar(fx, bracket=(0, 0.7), method='golden', tol=1.0e-6)
print(res)

# Brent方法是Golden方法的改良版，具体算法不知，但它加快了golden方法的收敛速度
print('{0:-^60}'.format('Use brent method'))
res = so.minimize_scalar(fx, bracket=(0, 2), method='brent', tol=1.0e-6)
print(res)

# Bounded方法是求区间内的函数最小值，必须提供边界参数bounds，此时tol参数不能使用
print('{0:-^60}'.format('Use bounded method'))
res = so.minimize_scalar(fx, bounds=(0, 0.6), method='bounded')
print(res)

