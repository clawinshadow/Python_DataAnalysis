import numpy as np
import scipy.io as sio
import sklearn.linear_model as slm
import sklearn.preprocessing as spp
import sklearn.metrics.pairwise as smp
import sklearn.model_selection as sms
import sklearn.svm as svm
import matplotlib.pyplot as plt

'''sparsity跟书中的结论不太一样，要详细核对下两边使用的算法，以后再查，意思理解就行了'''

# prepare data
data = sio.loadmat('kernelRegrDemo.mat')
print(data.keys())
X = data['X']
y = data['y'].ravel()
print(X.shape, y.shape)

lambda_ = 0.5
rbfscale = 0.3
gamma = 1 / (2 * rbfscale**2)
xtest = np.linspace(-10, 10, 201).reshape(-1, 1)
xtest_standard = spp.StandardScaler().fit_transform(xtest)
xtest_standard = xtest_standard * np.sqrt(200/201)

x_train = smp.rbf_kernel(X, gamma=gamma)
x_train = x_train / np.sqrt(2 * np.pi * rbfscale**2)
x_test = smp.rbf_kernel(xtest_standard, X, gamma=gamma)
x_test = x_test / np.sqrt(2 * np.pi * rbfscale**2)

# Fit with Ridge
ridge = slm.Ridge(alpha=lambda_, fit_intercept=False).fit(x_train, y)
print('w by L2: \n', ridge.coef_)
y_predict_L2 = ridge.predict(x_test)

# Fit with L1
N, D = x_train.shape
lambda_L1 = lambda_/(2 * N)
lasso = slm.LassoLars(alpha=lambda_L1, fit_intercept=False).fit(x_train, y)
print('w by L1: \n', lasso.coef_)
y_predict_L1 = lasso.predict(x_test)
sv_indices = lasso.coef_ != 0

# Fit with ARD
ARD = slm.ARDRegression(fit_intercept=False).fit(x_train, y)
print('w by ARD: \n', ARD.coef_)
y_predict_rvm = ARD.predict(x_test)
sv_indices_rvm = ARD.coef_ != 0

# Fit with SVR
scaler = spp.StandardScaler().fit(X)
x_standard = scaler.transform(X)
N, D = X.shape
factor = np.sqrt((N - 1) / N)
x_standard = x_standard * factor
K = 5
Cs = 2**(np.linspace(-5, 5, 10))
scoreMat = np.zeros((len(Cs), K))
for i in range(len(Cs)):
    Ci = Cs[i]
    svr = svm.SVR(C=Ci, gamma=gamma)
    scoreMat[i] = sms.cross_val_score(svr, x_standard, y, cv=K)
scores = np.mean(scoreMat, axis=1)
bestIndex = np.argmax(scores)
print(bestIndex)
C_best = Cs[bestIndex]

svr = svm.SVR(C=C_best, gamma=gamma).fit(x_standard, y)
y_predict_svm = svr.predict(xtest_standard)

# plots
fig = plt.figure(figsize=(9, 8))
fig.canvas.set_window_title('kernelRegrDemo')

def plot(idx, title, xdata, ydata, sv='none'):
    plt.subplot(idx)
    plt.axis([-2, 2, -0.4, 1.2])
    plt.xticks(np.linspace(-2, 2, 9))
    plt.yticks(np.linspace(-0.4, 1.2, 9))
    plt.plot(X, y, marker='*', color='midnightblue', ms=8, linewidth=2, linestyle='none')
    plt.title(title)
    plt.plot(xdata, ydata, 'g-', lw=2)
    if not sv == 'none':
        plt.plot(X[sv], y[sv], 'ro', fillstyle='none', linestyle='none', ms=10, mew=2)

plot(221, 'linregL2', xtest_standard, y_predict_L2)
plot(222, 'linregL1', xtest_standard, y_predict_L1, sv_indices)
plot(223, 'RVM', xtest_standard, y_predict_rvm, sv_indices_rvm)
plot(224, 'SVM', xtest_standard, y_predict_svm, svr.support_)


plt.tight_layout()
plt.show()