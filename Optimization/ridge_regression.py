import numpy as np
import scipy.optimize as so
import sklearn.linear_model as slm

def func(x, alpha):
    return x[0]**2 + (x[0] - 0.1)**2 + (x[0] + x[1] + x[2] - 1)**2 + alpha * (x[0]**2 + x[1]**2 + x[2]**2)

def func_der(x, alpha):
    df_x0 = (2*alpha + 6)*x[0] - 0.2 + 2*(x[1] + x[2] -1)
    df_x1 = 2*(x[0] + x[1] + x[2] - 1) + 2 * alpha * x[1]
    df_x2 = 2*(x[0] + x[1] + x[2] - 1) + 2 * alpha * x[2]
    return np.array([df_x0, df_x1, df_x2])

# 使用BFGS需要一阶导数矩阵
res = so.minimize(func, [1.0, 1.0, 1.0], args=(.5,), method='BFGS', jac=func_der, tol=0.0000001, options = {'disp': True})

# 单纯形下山法，不需要一阶导数矩阵
res_nm = so.minimize(func, [1.0, 1.0, 1.0], args=(.5,), method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
print(res.x)
print(res_nm.x)

# 使用sklearn中的岭回归类，但是算出来的结果不一样，不晓得问题出在哪里，回头看看源代码
X = [[0, 0],
     [0, 0],
     [1, 1]]  # 模拟两个共线性的特征列，如果用OLE，系数肯定很难看
y = [0, 0.1, 1]
reg = slm.Ridge(alpha = .5)
reg.fit (X, y) 

print('coefficients(w1, w2) by linear_model.Ridge in sklearn: ', reg.coef_)
print('intercept(w0) by linear_model.Ridge in sklearn: ', reg.intercept_) 
