import numpy as np
import scipy.stats as ss
import scipy.io as sio
import scipy.linalg as sl
import sklearn.linear_model as slm
import matplotlib.pyplot as plt

# extract data
data = sio.loadmat('sat.mat')
sat = data['sat']
X, y = sat[:,3], sat[:,0]
X = X.reshape(-1, 1)

# plot data
fig = plt.figure(figsize=(6, 5))
fig.canvas.set_window_title('logregSATdemoBayes')

plt.subplot()
plt.axis([455, 655, -0.02, 1.02])
plt.xticks(np.arange(460, 655, 20))
plt.yticks(np.arange(0, 1.02, 0.1))
plt.plot(X.ravel(), y, 'ko')

# 利用IRLS来计算w
def logistic(x):
    return 1 / (1 + np.exp(-x))

def IRLS(X, y, maxIter=100):
    i = 0
    w0 = np.array([0, 0]).reshape(-1, 1)
    X = np.c_[np.ones(len(X)), X]        # 因为有截距，所以第一列填充1
    H = np.zeros((2, 2))                 # 二阶导数矩阵，后面要用
    while i < maxIter:
        eta = np.dot(X, w0)
        mu = logistic(eta)
        S = np.diag((mu * (1 - mu)).ravel())
        part_1 = np.dot(sl.inv(np.dot(X.T, np.dot(S, X))), X.T)
        part_2 = np.dot(S, eta) + y.reshape(-1, 1) - mu.reshape(-1, 1)
        w_next = np.dot(part_1, part_2)
        H = np.dot(X.T, np.dot(S, X))
        if np.allclose(w0, w_next):
            print('break')
            break
        w0 = w_next
        i += 1

    return np.array(w0).ravel(), H

w_IRLS, H = IRLS(X, y)
print('w_IRLS: ', w_IRLS)

logReg = slm.LogisticRegression(C=1e15, tol=1e-6)   # C值非常大时，相当于关闭了penalization
logReg.fit(X, y)
w_sklearn = np.concatenate((logReg.intercept_, logReg.coef_.ravel())) 
print('logReg.w: ', w_sklearn)

# 计算Gaussian Approximation
mu = w_IRLS
cov = sl.inv(H)
posterior = ss.multivariate_normal(mu, cov)

X = np.c_[np.ones(len(X)), X]
mean = logistic(np.dot(X, w_IRLS.reshape(-1, 1))).ravel()

# 利用MC来计算posterior的median和quantiles
w_samples = posterior.rvs(1000)
predict_probs = logistic(np.dot(X, w_samples.T))

median = np.median(predict_probs, axis=1)
quantile_5 = np.percentile(predict_probs, 5, axis=1)
quantile_95 = np.percentile(predict_probs, 95, axis=1)

print('mean: ', mean)
print('median: ', median)
print('quantile_5: ', quantile_5)
print('quantile_95: ', quantile_95)

# plot these statistics
plt.plot(X[:, 1], mean, 'ro', fillstyle='none')
plt.plot(X[:, 1], median, 'bx', ms=4)
plt.errorbar(X[:, 1], mean, yerr=np.vstack((quantile_95 - mean, mean - quantile_5)), ls='none', elinewidth=0.5)

plt.show()

