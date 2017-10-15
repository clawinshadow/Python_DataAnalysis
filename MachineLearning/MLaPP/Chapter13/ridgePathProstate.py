import numpy as np
import scipy.io as sio
import sklearn.metrics as sm
import sklearn.preprocessing as sp
import sklearn.linear_model as slm
import sklearn.model_selection as sms
import matplotlib.pyplot as plt

np.random.seed(0)

# prepare data
data = sio.loadmat('prostateStnd.mat')
print(data['names'])
print(data['X'].shape)
print(data['y'].shape)

X, y = data['X'], data['y']
print('mean(X), mean(Y): ', np.mean(X), np.mean(y))  # mean(y) not equal to 0, didn't been standardized

legends = np.array(['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45'])

# 注意logspace的用法，默认是10为底的，其实就是10**(np.linspace(0.8, 4, 30))
# 下面的起始值0.8和4完全只能靠我自己猜，书中没写，pmtk里面的的代码我也看不懂，无语
lambdas = np.logspace(0.8, 4, 30, base=10)[::-1]  # reverse order

# fit models with lambdas
N, D = X.shape
K = 5  # for CV
w_mat = np.zeros((len(lambdas), D))  # exclude intercept
scoreDict = dict()
for i in range(len(lambdas)):
    l = lambdas[i]
    X_y = np.c_[X, y]
    np.random.shuffle(X_y)
    x_train, y_train = X_y[:, :D], X_y[:, -1].reshape(-1, 1)
    ridge = slm.Ridge(alpha=l).fit(x_train, y_train)
    print(ridge.coef_, ridge.intercept_)
    w_mat[i] = ridge.coef_
    ridgeCV = slm.Ridge(alpha=l)
    #scores_i = -1 * sms.cross_val_score(ridgeCV, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
    scores_i = sms.cross_val_score(ridgeCV, X, y, cv=5) # R2 score的结果更接近于图形
    mean_score = np.mean(scores_i)
    scoreDict[mean_score] = i

print(scoreDict)

# use 1SE rule, 就是CV的最小值加上一个标准差，取这个值作为最优的参数
scores = np.array(list(scoreDict.keys()), dtype='float64')
std_err = np.var(scores)**0.5
print('std_err: ', std_err)
s = np.sort(scores)
thresh = s.min() + std_err  # 1 SE rule
s_se = s[s > thresh][0]    # 比它大的里面挑个最小的
index = scoreDict[s_se]
print('best lambda by CV: ', index)

# select the first lambda to verify the correctness of method cross_val_score()
# 需要关闭上面的 np.random.shuffle(X_y)
kf = sms.KFold(n_splits=5)
scores_2 = []
for train_indices, test_indices in kf.split(X):
    x_train, x_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    l_tmp = lambdas[0]
    ridge = slm.Ridge(alpha=l_tmp)
    ridge.fit(x_train, y_train)
    y_predict = ridge.predict(x_test)
    score = sm.mean_squared_error(y_predict, y_test)
    scores_2.append(score)

print('scores_2: ', scores_2)

# plots
fig = plt.figure()
fig.canvas.set_window_title('ridgePathProstate')

plt.subplot()
plt.axis([0, 30, -0.2, 0.6])
plt.xticks(np.linspace(0, 30, 7))
plt.yticks(np.linspace(-0.2, 0.6, 9))
plt.plot(np.linspace(1, len(lambdas), len(lambdas)), w_mat, marker='o', fillstyle='none')
plt.axvline(index + 1, color='r', lw=2) # index 是从0开始计数的，这边的坐标是从1开始的
plt.legend(legends)

plt.show()

