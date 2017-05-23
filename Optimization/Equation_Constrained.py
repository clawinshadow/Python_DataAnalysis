import numpy as np
import scipy.optimize as so

'''
理论上来说，在求解含等式约束的极小值问题时，可以用拉格朗日定理，二阶充分条件来求解：
1. Df(x*) + λ*Dh(x*) = 0
2. 对于所有 y 属于h(X)在点x*处的切线空间 T(x*)，并且y不等于0，都有 y.T * L(x*, λ*) * y > 0
其中 L(x*, λ*) = F(x*) + [λH(x*)] 名为拉格朗日函数，需要用到f和h的二阶导数矩阵，算出来是个n*n的矩阵

满足这两个条件的话，x*就是f(x)在约束h(x)=0下的严格局部最小点，同理，如果条件2中的不等式是大于0，则为严格局部极大点

e.g. minimize    f(x) = x1**2 + 2*x1*x2 + 3*x2**2 + 4*x1 + 5*x2 + 6*x3
     subject to  x1 + 2*x2 = 3
                 4*x1 + 5*x3 = 6

scipy.optimize 用起来十分方便，如下：
手算的照片见同级目录
'''


def func(x, sign=1.0):
    '''Objective function 目标函数'''
    return sign * (x[0]**2 + 2*x[0]*x[1] + 3*x[1]**2 + 4*x[0] + 5*x[1] + 6*x[2])

def func_deriv(x, sign=1.0):
    '''Derivative of objective function, D[f(x)]'''
    df_dx0 = sign*(2*x[0] + 2*x[1] + 4)
    df_dx1 = sign*(2*x[0] + 6*x[1] + 5)
    df_dx2 = sign*(6)
    return np.array([df_dx0, df_dx1, df_dx2])

cons = ({'type': 'eq',
         'fun': lambda x: np.array([x[0] + 2*x[1] - 3]),
         'jac': lambda x: np.array([1, 2, 0])},
        {'type': 'eq',
         'fun': lambda x: np.array([4*x[0] + 5*x[2] - 6]),
         'jac': lambda x: np.array([4, 0, 5])})

# docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
# 第二个参数是随便给定的一个起始解x, 这个解的维度一定不能错
# 如果选定的是SLSQP参数，一定要给定一阶导数矩阵jac参数，同理选定其他方法时也要查看文档，赋予相应的参数
res = so.minimize(func, [-1.0, 1.0, 1.0], args=(1.0,), jac=func_deriv, constraints=cons, method='SLSQP', options={'disp': True})
print(res.x)
