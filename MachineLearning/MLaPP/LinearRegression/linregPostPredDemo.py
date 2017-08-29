import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
import sklearn.linear_model as slm
import matplotlib.pyplot as plt

'''
response variable: y ~ N(w.T * x, sigma**2)
prior of w: w ~ N(W0, V0)
so:
  posterior of w (using Bayes theory):
  p(w|D) ~ N(WN, VN)
  WN = VN * inv(V0) * W0 + 1/sigma**2 * VN * X.T * y
  VN.inv = V0.inv + 1/sigma**2 * X.T * X

  prediction distribution of x: integrate out w
  p(y|x, D, sigma**2) = N(y|WN.T * x, sigma_N(x) ** 2)
  WN.T is the mean of posterior p(w|D)
  sigma_N(x) ** 2 = sigma**2 + x.T * VN * x, 每个点都不一样

  prediction of distribution of x (plug-in approximation): 不再是积掉w,
      而是用w的MLE来代替积分里面w的后验概率 N(w|WN, VN), 得出来的结果就是
      每个点的预测均值与前面的一样，但是方差不再随样本点的变化而变化，每个点的预测方差都是一样的

      sigma_N(x) ** 2 = sigma**2

  当先验分布 w0 = 0, V0 = delta**2 * I 时，降格为岭回归，lambda = sigma**2 / delta**2
  VN = sigma**2(lambda * I + X.T * X).inv
  WN = (lambda * I + X.T * X).inv * X.T * y
'''

def DM(x, degree):
    result = np.ones(len(x)).reshape(-1, 1)
    for i in range(degree):
        result = np.c_[result, (x**(i+1)).reshape(-1, 1)]

    return result

# Generate data
np.random.seed(0)
w_true = [-1.5, 1/9]
sigma = 5            # standard variance of noise
x_sample = np.array([-3, -2, 0, 2, 3])
y_true = w_true[0] * x_sample + w_true[1] * x_sample**2 # real func, 没有带截距
y_sample = y_true + sigma * np.random.randn(len(x_sample))

degree = 2
x_test = np.linspace(-7, 7, 43)
x_train = DM(x_sample, degree)
x_test_dm = DM(x_test, degree)

# Fit model
alpha = 0.001
model = slm.Ridge(alpha=alpha, fit_intercept=False)
model.fit(x_train, y_sample)
w = model.coef_
print('w: ', w)

# posterior w
WN = w
VN = sigma**2 * sl.inv(alpha * np.eye(len(w)) + np.dot(x_train.T, x_train))
postW = ss.multivariate_normal(mean=WN, cov=VN)
ws = postW.rvs(10)

# predict
y_train_predict = model.predict(x_train)
fixSigma2 = np.var(y_train_predict - y_sample, ddof=degree+1)
fixSigma = fixSigma2**0.5
sigma_N = []
for i in range(len(x_test_dm)):
    x = x_test_dm[i].reshape(-1, 1)
    sigma_N.append((sigma ** 2 + np.dot(x.T, np.dot(VN, x))) ** 0.5)
sigma_N = np.array(sigma_N).ravel()

y_test_predict = model.predict(x_test_dm)
y_predicts = []
for i in range(len(ws)):
    w = ws[i]
    y_predicts.append(np.dot(x_test_dm, w.reshape(-1, 1)).ravel())
y_predicts = np.array(y_predicts).T

# plots
fig = plt.figure(figsize=(11, 10))
fig.canvas.set_window_title('linregPostPredDemo')

plt.subplot(221)
plt.axis([-8, 8, 0, 60])
plt.title('plugin approximation (MLE)')
plt.xticks(np.arange(-8, 8.1, 2))
plt.yticks(np.arange(0, 61, 10))
plt.errorbar(x_test, y_test_predict, fixSigma, ecolor='midnightblue', capsize=2, label='prediction')
plt.plot(x_sample, y_sample, ls='none', color='r', marker='o', ms=8,\
         mew=2, fillstyle='none', label='training data')
plt.legend()

plt.subplot(222)
plt.axis([-8, 8, -10, 80])
plt.title('Posterior predicive (known variance)')
plt.xticks(np.arange(-8, 8.1, 2))
plt.yticks(np.arange(10, 81, 10))
plt.errorbar(x_test, y_test_predict, sigma_N, ecolor='midnightblue', capsize=2, label='prediction')
plt.plot(x_sample, y_sample, ls='none', color='r', marker='o', ms=8,\
         mew=2, fillstyle='none', label='training data')
plt.legend()

plt.subplot(223)
plt.axis([-8, 8, 0, 50])
plt.title('functions sampled from plugin approximation to posterior')
plt.xticks(np.arange(-8, 8.1, 2))
plt.yticks(np.arange(0, 51, 5))
plt.plot(x_sample, y_sample, ls='none', color='r', marker='o', ms=8, mew=2, fillstyle='none')
plt.plot(x_test, y_test_predict, 'k-', lw=2)

plt.subplot(224)
plt.axis([-8, 8, -20, 100])
plt.title('functions sampled from posterior')
plt.xticks(np.arange(-8, 8.1, 2))
plt.yticks(np.arange(-20, 101, 20))
plt.plot(x_sample, y_sample, ls='none', color='r', marker='o', ms=8, mew=2, fillstyle='none')
plt.plot(x_test, y_predicts, 'k-', lw=1)

plt.show()
