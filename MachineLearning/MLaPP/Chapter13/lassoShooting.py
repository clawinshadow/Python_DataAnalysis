import math
import numpy as np
import scipy.io as sio
import scipy.linalg as sl
import sklearn.linear_model as slm

'''
最简单朴素的 Coordinate descent 方法来求解BPDN的问题

一个要特别注意的地方就是：
书中的Coordinate Descent中使用的λ 来自于目标函数 ： RSS(w) + λ*||w||_1
而sklearn.Lasso中使用的目标函数是: 1/(2*N) * RSS(w) + λ * ||w||_1
所以我们要验证结果是，给sklearn中传入的alpha参数必须要除以 2 * N， N为sample的数量
'''

def soft(a, b):
    val_1 = max(abs(a) - b, 0)
    sign_a = a / abs(a)
    return sign_a * val_1

# prepare data
data = sio.loadmat('prostateStnd.mat')
X, y = data['Xtrain'], data['ytrain']  
y = y - np.mean(y)    # center the response variable

# Fit models with sklearn
N, D = X.shape
larsRes = slm.lars_path(X, y.ravel())    # 先用LARS计算lasso的path，看lambda的最大值和最小值是多少
lambda_path = larsRes[0]    # 9个
coef_path = larsRes[2]      # (8, 9) every col is a w vector
print('lambda path: ', lambda_path)
print('coef path: ', coef_path)

l = lambda_path[3] * 2 * N # 任选一个来作为lambda

# Fit model with Coordinate descent
tmp = sl.inv(np.dot(X.T, X) + l * np.eye(D))
w = np.dot(tmp, np.dot(X.T, y))  # ridge estimate to initialize w
print(w.shape)
max_iter = 200
for i in range(max_iter):
    w_new = np.copy(w)
    for j in range(D):
        xj = X[:, j].reshape(-1, 1)
        wj = w_new[j]
        aj = 2 * np.sum(xj**2)
        vec_1 = xj * (y - np.dot(X, w_new) + wj * xj)  # N * 1
        cj = 2 * np.sum(vec_1)
        w_new[j] = soft(cj / aj, l / aj)

    gap = sl.norm(w_new - w, 1)
    print('gap: ', gap)
    if np.allclose(w_new, w, rtol=1e-10):
        w = w_new
        break

    w = w_new.reshape(-1, 1)

# 要注意我们自己算的lambda和sklearn.Lasso中用的lambda的单位是不一样的，所以有l/(2*N)
lasso = slm.Lasso(alpha=l/(2*N), fit_intercept=False).fit(X, y)  
print('w by Lasso: ', lasso.coef_)
print('w by coordinate descent： ', w)
print('w by lars_path: ', coef_path[:, 4])
