import numpy as np
import scipy.linalg as sl
import scipy.optimize as so
import sklearn.linear_model as slm

'''
验证scipy.optimize.curve_fit()这个方法，与sklearn和公式相比较，理论上要是一样的
'''

def randRange(vmin, vmax, size=20):
    return vmin + (vmax - vmin) * np.random.rand(20)

def func(x, w0, w1):
    '''
    curve_fit()中使用的函数f(*)定义很严格:
    1. 签名里面第一个参数x对应于xdata
    2. 余下的每一个参数都是待估计的参数，并且均为标量

    Assume：ydata = f(xdata, *params) + eps
    '''
    return w0 + w1 * x

# generate data
w_true = [1, 1]                   # 真实的权重向量
x = randRange(-4, 4, 20)
y = w_true[0] + w_true[1] * x + 0.8 * np.random.randn(20)  # N(0, 0.8**2)的噪声

popt, pcov = so.curve_fit(func, x, y)   # pcov 是 一个协方差矩阵，暂时不明到底是干嘛的
print('w by scipy.optimize.curve_fit(): ', popt)

reg = slm.LinearRegression()
reg.fit(x.reshape(-1, 1), y)
print('w by sklean.linear_model.LinearRegression:', [ reg.intercept_, reg.coef_])  # 系数和截距是分开的

# 用公式计算
dm = np.c_[np.ones(len(x)).reshape(-1, 1), x.reshape(-1, 1)]  # design matrix
y_vector = y.reshape(-1, 1)
w_MLE = np.dot(sl.inv(np.dot(dm.T, dm)), np.dot(dm.T, y_vector))

print('w calculated by equation: ', w_MLE.ravel())


