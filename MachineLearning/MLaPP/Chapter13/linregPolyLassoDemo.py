import numpy as np
import scipy.linalg as sl
import sklearn.preprocessing as sp
import sklearn.linear_model as slm
import sklearn.metrics as sm
import matplotlib.pyplot as plt

#rs = np.random.RandomState(654321)
np.random.seed(654321)

# prepare data
deg = 14
w = np.array([-1.5, 1/9])  # true w
sigma = 2 # noise std
x_train = np.linspace(0, 20, 21)
x_test = np.arange(0, 20, 0.1)
y_train = w[0] * x_train + w[1] * x_train**2 + sigma * np.random.randn(len(x_train))
y_test_noisyfree = w[0] * x_test + w[1] * x_test**2
y_test = y_test_noisyfree + sigma * np.random.randn(len(x_test))

# remove the mean of response
y_train = (y_train - np.mean(y_train)).reshape(-1, 1)
y_test = (y_test - np.mean(y_test)).reshape(-1, 1)
# preprocessing X
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
ss = sp.StandardScaler().fit(x_train)                    # 1. standardize first
x_train = ss.transform(x_train)
mm = sp.MinMaxScaler(feature_range=(-1, 1)).fit(x_train) # 2. rescale to [-1, 1], 参数却是(0, 1)
x_train = mm.transform(x_train)
po = sp.PolynomialFeatures(degree=deg).fit(x_train)      # 3. polynomial expansion
x_train = po.transform(x_train)[:, 1:]                   # remove the first column [1, 1, 1, 1...]

print(x_train.shape)
x_test = ss.transform(x_test)
x_test = mm.transform(x_test)
po = sp.PolynomialFeatures(degree=deg).fit(x_test)
x_test = po.transform(x_test)[:, 1:]
print(x_test.shape)

lambda_max = sl.norm(np.dot(x_train.T, y_train), np.inf)
print(lambda_max)

lambdas = np.array([lambda_max, 2, 1, 0.5, 0.1, 0.01, 0.0001, 0])
count = len(lambdas)
trainMses = np.zeros(count)
testMses = np.zeros(count)
path = slm.lars_path(x_train, y_train.ravel())
print('lasso path by LARS: ', path[0])
for i in range(count):
    l = lambdas[i]
    # lasso = slm.Lasso(alpha=l, fit_intercept=False).fit(x_train, y_train)
    # slm.Lasso是用coordinate descent的算法计算的，非常慢并且在lambda过小时可能不会收敛
    lasso = slm.LassoLars(alpha=l, fit_intercept=False, normalize=False).fit(x_train, y_train)
    print(lasso.coef_)
    y_predict_train = lasso.predict(x_train)
    y_predict_test = lasso.predict(x_test)
    trainMses[i] = sm.mean_squared_error(y_train, y_predict_train)
    testMses[i] = sm.mean_squared_error(y_test, y_predict_test)

print(trainMses)
print(testMses)

# plots
fig = plt.figure()
fig.canvas.set_window_title('linregPolyLassoDemo')

ax = plt.subplot()
plt.axis([count - 1, -0.2, 0, 45])
ax.set_xticklabels(lambdas)
plt.xticks(np.arange(0, count, 1)[::-1])
plt.yticks(np.arange(0, 46, 5))
plt.xlabel('lambda')
plt.ylabel('mse')
plt.plot(np.arange(0, count, 1)[::-1], trainMses, 'b:', marker='s', fillstyle='none', label='train')
plt.plot(np.arange(0, count, 1)[::-1], testMses, 'r-', marker='x', fillstyle='none', label='test')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.legend()
plt.show()

