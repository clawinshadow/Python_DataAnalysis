import numpy as np
import scipy.linalg as sl
import scipy.optimize as so
import sklearn.linear_model as slm

'''
验证scipy.optimize.minimize()这个方法，与sklearn和公式相比较，理论上要是一样的.
这个方法的定义如下：

minimize f(x) subject to

g_i(x) >= 0,  i = 1,...,m
h_j(x)  = 0,  j = 1,...,p

这个方法最通用，只要把函数f(x)和约束条件定义好，那么几乎可以求解所有最优化问题
但难就难在有时候f(x)不太好定义，比如在曲线拟合中，函数的定义就得把所有数据点的残差平方和都加起来
显然没有curve_fit()和leastsq()方便
Returns: 返回一个类 OptimizeResult，属性如下， 一般我们只关心x和success与否
        x	(ndarray) The solution of the optimization.
        success	(bool) Whether or not the optimizer exited successfully.
        status	(int) Termination status of the optimizer. Its value depends on the underlying solver. Refer to message for details.
        message	(str) Description of the cause of the termination.
        fun, jac, hess: ndarray	Values of objective function, its Jacobian and its Hessian (if available).
                        The Hessians may be approximations, see the documentation of the function in question.
        hess_inv	(object) Inverse of the objective function’s Hessian; may be an approximation. Not available for all solvers.
                        The type of this attribute may be either np.ndarray or scipy.sparse.linalg.LinearOperator.
        nfev, njev, nhev	(int) Number of evaluations of the objective functions and of its Jacobian and Hessian.
        nit	(int) Number of iterations performed by the optimizer.
        maxcv	(float) The maximum constraint violation.
'''

def randRange(vmin, vmax, size=20):
    return vmin + (vmax - vmin) * np.random.rand(20)

def SSE(w, x_train, y_train):
    return np.sum(np.power(y_train - w[0] - w[1] * x_train, 2))

# generate data
w_true = [1, 1]                   # 真实的权重向量
x = randRange(-4, 4, 20)
y = w_true[0] + w_true[1] * x + 0.8 * np.random.randn(20)                          # N(0, 0.8**2)的噪声

w0 = [1, 1]                       # 需要给定一个初始值
res = so.minimize(SSE, w0, args=(x, y))
print('Optimization Success: ', res.success)
print('w by scipy.optimize.minimize(): ', res.x)

reg = slm.LinearRegression()
reg.fit(x.reshape(-1, 1), y)
print('w by sklean.linear_model.LinearRegression:', [ reg.intercept_, reg.coef_])  # 系数和截距是分开的

# 用公式计算
dm = np.c_[np.ones(len(x)).reshape(-1, 1), x.reshape(-1, 1)]                       # design matrix
y_vector = y.reshape(-1, 1)
w_MLE = np.dot(sl.inv(np.dot(dm.T, dm)), np.dot(dm.T, y_vector))

print('w calculated by equation: ', w_MLE.ravel())


