import numpy as np
import scipy.optimize as so
import scipy.linalg as sl
import sklearn.linear_model as slm
import matplotlib.pyplot as plt
import sklearn.preprocessing as ss

'''
！！写在最前面：当使用L2的regularization时，fit数据前一定要把design matrix标准化先，否则得出来的结果天壤之别
                即便把fit_intercept设置成True也没用，X标准化，但是y不用。在预测时，x也要先标准化
'''


def DM(x, addones=False):
    if addones:
        return np.c_[np.ones(len(x)).reshape(-1, 1), x.reshape(-1, 1), (x**2).reshape(-1, 1)] # 适用于有截距的情况
    else:
        return np.c_[x.reshape(-1, 1), (x**2).reshape(-1, 1)]  # 生成design matrix

def DM2(x, degree):
    if degree < 1:
        return x
    else:
        result = x.reshape(-1, 1)
        i = 2
        while i <= degree:
            result = np.c_[result, (x**i).reshape(-1, 1)]
            i += 1
        print('DM with degree - {0}: {1}'.format(degree, result.shape))
        return result

# Generate data
w_true = [-1.5, 1/9]
sigma = 2                                                # 噪声的标准差
x_sample = np.linspace(0, 20, 21)
y_true = w_true[0] * x_sample + w_true[1] * x_sample**2  # real func，没有带截距
y_sample = y_true + sigma * np.random.randn(21)          

# Fit Model
# 第一种方法，直接用公式计算 Wˆridge = (λ*I + X.T*X).inv * X.T * y
Lambda = np.exp(-20.135)
dm = DM(x_sample)
y_vec = y_sample.reshape(-1, 1)
w_ridge = np.dot(sl.inv(Lambda * np.eye(2) + np.dot(dm.T, dm)), np.dot(dm.T, y_vec))
print('w_ridge calculated by equation: ', w_ridge.ravel())

# 第二种方法，使用sklearn.linear_model里面的Ridge类， alpha参数代表Lambda
ridge = slm.Ridge(alpha=Lambda, fit_intercept=False)
ridge.fit(dm, y_sample)
print('w_ridge calculated by sklearn.linear_model.Ridge: ', ridge.coef_)

# 第三种方法， Naive method，用so.minimize()来求解
def Loss(w, alpha, x_train, y_train, fit_intercept=False):
    SSE = []
    for i in range(len(x_train)):
        SSE.append((y_train[i] - np.sum(w * x_train[i]))**2)

    if fit_intercept:
        return np.sum(SSE) + alpha * np.sum(w[1:]**2)
    else:
        return np.sum(SSE) + alpha * np.sum(w**2)

w0 = [1, 1]
res = so.minimize(Loss, w0, args=(Lambda, dm, y_sample))
print('w_ridge by minimize(): ', res.x)     # 这三个结果应该都是相等的

# Fit Model with intercept(做个测试，与本例的图形没什么关系)
# method 1
dm2 = DM(x_sample, addones=True)
w_ridge2 = np.dot(sl.inv(Lambda * np.eye(3) + np.dot(dm2.T, dm2)), np.dot(dm2.T, y_vec))
print('w_ridge_intercept calculated by equation: ', w_ridge2.ravel())

# method 2
ridge2 = slm.Ridge(alpha=Lambda, fit_intercept=True)
ridge2.fit(dm2, y_sample)
print('w_ridge and intercept calculated by sklearn.linear_model.Ridge: ', ridge2.coef_, ridge2.intercept_)

# method 3
w1 = [1, 1, 1]
res2 = so.minimize(Loss, w1, args=(Lambda, dm2, y_sample, True))
# 这三个结果依然是相等的，证明不管是公式还是sklearn，都没有惩罚截距w0，都是自动化的，不用特别关注w0是否被惩罚的问题
print('w_ridge_intercept by minimize(): ', res2.x)
print('----------------------------------------------------------------------')

# 回到本例中的图形上来，用14阶多项式来回归
degree = 14
dm3 = DM2(x_sample, degree)
standard_dm3 = ss.StandardScaler().fit_transform(dm3)
ridge3 = slm.Ridge(alpha=Lambda, fit_intercept=False)
ridge3.fit(standard_dm3, y_sample)
w_ridge3 = ridge3.coef_
print('w with degree 14: ', w_ridge3)
yst = ridge3.predict(dm3)
SSE = np.sum((yst - y_sample)**2)

# generate test data
x_test = np.linspace(0, 20, 200)
# y_predict = np.dot(DM2(x_test, degree), w_ridge3.reshape(-1, 1))
x_test_standard = ss.StandardScaler().fit_transform(DM2(x_test, degree))
y_predict = ridge3.predict(x_test_standard)
print("y_predict, min, max: ", y_predict.min(), y_predict.max())

# 计算每个点的预测方差 VN = σ2*(λ*I + X.T * X).inv, sigma_N**2 = σ2 + x.T * VN * x
VN = sigma**2 * sl.inv(Lambda * np.eye(degree) + np.dot(standard_dm3.T, standard_dm3))
sig = []
for i in range(len(x_test_standard)):
    xi = x_test_standard[i]
    sig.append((4 + np.dot(xi, np.dot(VN, xi.T)))**0.5)
print('predict standard variance: ', sig)

# plot
fig = plt.figure(figsize=(12, 10))
fig.canvas.set_window_title('linregPolyVsRegDemo')

plt.subplot(221)
plt.axis([0, 20, -10, 20])
plt.xticks(np.arange(0, 21, 5))
plt.yticks(np.arange(-10, 21, 5))
plt.title(r'$ln\lambda = -20.135$')
plt.plot(x_sample, y_sample, color='midnightblue', marker='o', ls='none')
plt.plot(x_test, y_predict, 'k-', lw=2)
plt.plot(x_test, y_predict + sig, 'b:', lw=1)
plt.plot(x_test, y_predict - sig, 'b:', lw=1)

plt.show()





