import math
import numpy as np
import scipy.optimize as so
import scipy.linalg as sl
import scipy.stats as ss
import sklearn.linear_model as slm
import matplotlib.pyplot as plt
import sklearn.preprocessing as sp

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
standard_dm3 = sp.StandardScaler().fit_transform(dm3)

# plot
fig = plt.figure(figsize=(12, 10))
fig.canvas.set_window_title('linregPolyVsRegDemo')

def Fit(x_train, y_train, Lambda, degree):
    ridge3 = slm.Ridge(alpha=Lambda, fit_intercept=False)
    ridge3.fit(x_train, y_train)
    w_ridge3 = ridge3.coef_
    print('w with degree {0}: {1}'.format(degree, w_ridge3))

    return ridge3    # 返回岭回归的模型

def MSE(model, X, y):
    y_predict = model.predict(X)
    SSE = np.sum((y_predict - y)**2)
    MSE = SSE / len(X)

    return MSE

def sig_SE(model, X, y):
    y_predict = model.predict(X)
    SSE_Var = np.var((y_predict - y)**2)
    sig = SSE_Var ** 0.5

    return sig

def Draw(index, x_train, y_train, Lambda, degree): 
    model = Fit(x_train, y_train, Lambda, degree)
    sig = sig_SE(model, x_train, y_train)
    
    # generate test data
    x_test = np.linspace(0, 20, 200)
    # y_predict = np.dot(DM2(x_test, degree), w_ridge3.reshape(-1, 1))
    x_test_standard = sp.StandardScaler().fit_transform(DM2(x_test, degree))
    y_predict = model.predict(x_test_standard)
    print("y_predict, min, max: ", y_predict.min(), y_predict.max())
    
    plt.subplot(index)
    plt.axis([0, 20, -15, 20])
    plt.xticks(np.arange(0, 21, 5))
    plt.yticks(np.arange(-15, 21, 5))
    plt.title(r'$ln\lambda = $' + str(np.log(Lambda)))
    plt.plot(x_sample, y_sample, color='midnightblue', marker='o', ls='none')
    plt.plot(x_test, y_predict, 'k-', lw=2)
    plt.plot(x_test, y_predict + sig, 'b:', lw=1)
    plt.plot(x_test, y_predict - sig, 'b:', lw=1)

Draw(221, standard_dm3, y_sample, np.exp(-20.135), degree)
Draw(222, standard_dm3, y_sample, np.exp(-8.571), degree)

# plot train MSE & test MSE with lambda increasing
log_lambdas = np.linspace(-23, 3, 10)
lambdas = np.exp(log_lambdas)

x_test = np.linspace(0, 20, 200)
x_test_standard = sp.StandardScaler().fit_transform(DM2(x_test, degree))
y_test_true = w_true[0] * x_test + w_true[1] * x_test**2
train_MSEs = []
test_MSEs = []
for i in range(len(lambdas)):
    model = Fit(standard_dm3, y_sample, lambdas[i], degree)
    trainMSE = MSE(model, standard_dm3, y_sample)
    testMSE = MSE(model, x_test_standard, y_test_true)
    train_MSEs.append(trainMSE)
    test_MSEs.append(testMSE)

# 这里要备注一下，因为不清楚书中用的测试集是什么，所以画的图形跟书里面的有所出入
# 我们这里的测试集有200个点，MSE非常的小，比测试集的MSE要小，但是总体趋势是一致的
plt.subplot(223)
plt.xlim(-25, 5)
plt.xticks(np.arange(-25, 5.5, 5))
plt.xlabel(r'$log \lambda$')
plt.title('mean squared error')
plt.plot(log_lambdas, train_MSEs, ls=':', marker='s', color='midnightblue', \
         fillstyle='none', mew=2, label='train mse')
plt.plot(log_lambdas, test_MSEs, ls='-', marker='x', color='red', \
         fillstyle='none', mew=2, label='test mse')

# plot performance estimate using training set
def NLML(x_train, y_train, Lambda, noise_sig, degree):
    model = Fit(x_train, y_train, Lambda, degree)
    w = model.coef_
    rv_w = ss.multivariate_normal(np.tile(0, len(w)), (noise_sig**2 / Lambda) * np.eye(len(w)))
    logpdf_w = rv_w.logpdf(w)
    
    likelihood = []
    for i in range(len(x_train)):
        x = x_train[i]
        y = y_train[i]
        logpdf_y = ss.norm(np.sum(w*x), noise_sig).logpdf(y)
        likelihood.append(logpdf_w + logpdf_y)

    return -1 * np.sum(likelihood)

log_lambdas = np.linspace(-18, 2, 8)
lambdas = np.exp(log_lambdas)

NLMLs = []
for i in range(len(lambdas)):
    NLMLs.append(NLML(standard_dm3, y_sample, lambdas[i], sigma, degree))

# rescale to (0, 1)
print('origian NLMLs: ', NLMLs)
NLMLs = sp.MinMaxScaler().fit_transform(np.array(NLMLs).reshape(-1, 1)).ravel()
print('NLMLs: ', NLMLs)

plt.subplot(224)
plt.axis([-20, 5, 0, 1])
plt.yticks(np.linspace(0.1, 0.9, 9))
plt.xticks(np.linspace(-20, 5, 6))
plt.plot(log_lambdas, NLMLs, ls='-', marker='o', fillstyle='none', color='k', label='negative log marg. likelihood')

plt.show()





