import h5py
import numpy as np
import sklearn.svm as svm
import sklearn.linear_model as slm
import sklearn.metrics.pairwise as smp
import sklearn.model_selection as sms
import sklearn.preprocessing as spp
import matplotlib.pyplot as plt
from plotDecisionBoundary import *

'''
RVM for classification can not use ARDRegression, because it's used for regression
need to use Iterative L1 regularization, refer to 13.7.4.3, ARD for logistic Regression

!!SVM is different from in book, matlab code debug fail, need further check
'''

# prepare data
# this is as matlab v7.3 data, should install 'h5py' package to read it
with h5py.File('bishop2class.mat', 'r') as data:
    print('keys of data: ', list(data.keys()))
    X = np.array(data['X']).T
    y = np.array(data['Y']).T

print(X.shape, y.shape)

# don't need to standardize, just kernelise is enough
rbf_scale = 0.3
gamma = 1 / (2 * rbf_scale**2)
x_train = smp.rbf_kernel(X, gamma=gamma)     # kernelise
x_train = x_train / np.sqrt(2 * np.pi * rbf_scale**2)

# Fit with L2VM
lambda_ = 5
C = 1 / (2 * lambda_)  # convert lambda_ in matlab code to C in sklearn.logisicRegression
y = y.ravel()
LogReg = slm.LogisticRegression(penalty='l2', C=C, tol=1e-03, solver='lbfgs', max_iter=200, fit_intercept=False).fit(x_train, y)
print('w of L2VM: ', LogReg.coef_)  # exactly the same as in matlab codes
y_predict_L2 = LogReg.predict(x_train)
nerr_l2 = np.count_nonzero(y_predict_L2 == y)
print('nerr of L2VM: ', nerr_l2)

# Fit with L1VM
C_L1 = 1
LG_L1 = slm.LogisticRegression(penalty='l1', C=C_L1, tol=1e-05, solver='liblinear', max_iter=200, fit_intercept=False).fit(x_train, y)
print('w of L1VM: ', LG_L1.coef_)
y_predict_L1 = LG_L1.predict(x_train)
nerr_L1 = np.count_nonzero(y_predict_L1 == y)
print('nerr of L1VM: ', nerr_L1)

# Fit with RVM/SBL
#ARD = slm.ARDRegression().fit(x_train, y)
#print('w of RVM: ', ARD.coef_)
#print('alpha of priors: ', ARD.lambda_)

# Fit with SVM CV
scaler = spp.StandardScaler().fit(X)
x_standard = scaler.transform(X)
N, D = X.shape
factor = np.sqrt((N - 1) / N)
x_standard = x_standard * factor
print(x_standard)
K = 5
Cs = 2**(np.linspace(-5, 5, 10))
scoreMat = np.zeros((len(Cs), K))
for i in range(len(Cs)):
    Ci = Cs[i]
    SVC = svm.SVC(Ci, gamma=gamma)
    scoreMat[i] = sms.cross_val_score(SVC, x_standard, y, cv=K)

print(scoreMat)

scores = np.mean(scoreMat, axis=1)
bestIndex = np.argmax(scores)
print(bestIndex)
C_best = Cs[bestIndex]

svc = svm.SVC(C=C_best, gamma=gamma).fit(x_standard, y)
print('support vectors number: ', svc.n_support_)
y_predict_svc = svc.predict(x_standard)
nerr_svc = np.count_nonzero(y_predict_svc == y)
print('nerr of SVM: ', nerr_svc)

# plots
fig = plt.figure(figsize=(9, 8))
fig.canvas.set_window_title('kernelBinaryClassifDemo')

args = dict(subplotIndex=221, title='logregL2, nerr='+str(nerr_l2), tickRange=[-2, 3, -3, 3],\
            centers=X, rbfscale=rbf_scale, markers=['+', 'x'])
plot(X, y, LogReg, smp.rbf_kernel, **args)

args2 = dict(subplotIndex=222, title='logregL1, nerr='+str(nerr_L1), tickRange=[-2, 3, -3, 3],\
            centers=X, rbfscale=rbf_scale, markers=['+', 'x'], drawSV=True)
plot(X, y, LG_L1, smp.rbf_kernel, **args2)

args3 = dict(subplotIndex=224, title='SVM, nerr='+str(nerr_svc), tickRange=[-2, 3, -3, 3], \
             sscaler=scaler, factor=factor, markers=['+', 'x'], drawSV=True, svIndices=svc.support_)
plot(X, y, svc, 'none', **args3)

plt.tight_layout()
plt.show()

