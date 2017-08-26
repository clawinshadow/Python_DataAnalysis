import numpy as np
import scipy.linalg as sl
import scipy.optimize as so
import sklearn.linear_model as slm

'''
验证scipy.optimize.leastsq()这个方法，与sklearn和公式相比较，理论上要是一样的.
这个方法中 函数的定义如下：
x = arg min(sum(func(y)**2,axis=0))
         y

在曲线拟合中，这个func(y)定义的是残差，即func(y) = ydata - f(xdata, *params)
Returns: 如果full_output=False时，返回的是精简后的结果，只有参数x和标志位ier
         否则返回所有的信息，一般不使用
'''

def randRange(vmin, vmax, size=20):
    return vmin + (vmax - vmin) * np.random.rand(20)

def Residual(w, x, y):
    '''
    和curve_fit()不一样的是，这个待估计的参数要放在第一个参数..并且支持数组，
    后面的参数不管多少个，调用的时候都放在args=tuple()里面去
    '''
    return w[0] + w[1] * x - y

# generate data
w_true = [1, 1]                   # 真实的权重向量
x = randRange(-4, 4, 20)
y = w_true[0] + w_true[1] * x + 0.8 * np.random.randn(20)                          # N(0, 0.8**2)的噪声

w0 = [1, 1]                       # 需要给定一个初始值
result = so.leastsq(Residual, w0, args=(x, y))          # curve_fit()的参数和函数设定比较奇葩，这个是比较通用的格式    
print('w by scipy.optimize.leastsq(): ', result[0])

reg = slm.LinearRegression()
reg.fit(x.reshape(-1, 1), y)
print('w by sklean.linear_model.LinearRegression:', [ reg.intercept_, reg.coef_])  # 系数和截距是分开的

# 用公式计算
dm = np.c_[np.ones(len(x)).reshape(-1, 1), x.reshape(-1, 1)]                       # design matrix
y_vector = y.reshape(-1, 1)
w_MLE = np.dot(sl.inv(np.dot(dm.T, dm)), np.dot(dm.T, y_vector))

print('w calculated by equation: ', w_MLE.ravel())


