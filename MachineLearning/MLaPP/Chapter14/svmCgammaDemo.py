import numpy as np
import scipy.io as sio
import sklearn.svm as svm
import sklearn.model_selection as sms
import sklearn.preprocessing as spp
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
svm.SVC, SVR is exactly the same as in matlab, because they both based on libsvm
'''

# prepare data
data = sio.loadmat('hastieMixture.mat')
print(data.keys())
X, y = data['X'], data['y']
print(X.shape, y.shape)
x_standard = spp.StandardScaler().fit_transform(X)
N, D = X.shape
factor = np.sqrt((N - 1) / N)
x_standard = x_standard * factor
y = y.ravel()

# Fit with SVM
# method 1: use KFold explictly
gamma = 5.0
C_range = np.logspace(-1, 3.5, 15)
K = 5
kf = sms.KFold(n_splits=K, shuffle=False)
scores = np.zeros(len(C_range))
scores_se = np.zeros(len(C_range))
for i in range(len(C_range)):
    C = C_range[i]
    losses = np.zeros(len(x_standard))
    for train_index, test_index in kf.split(x_standard):
        X_train, X_test = x_standard[train_index], x_standard[test_index]
        y_train, y_test = y[train_index], y[test_index]
        svc = svm.SVC(C=C, gamma=gamma, shrinking=False).fit(X_train, y_train)
        y_predict = svc.predict(X_test)
        losses[test_index] = np.mean(y_test != y_predict)  # 0-1 loss
    scores[i] = np.mean(losses)
    scores_se[i] = np.std(losses) / np.sqrt(N)

best_index = np.argmin(scores)
best_C = C_range[best_index]
print('Best C: ', best_C)
print('CV scores: ', scores)
print('Standard Erros of CV scores: ', scores_se)

# method 2: use cross_val_score
def one_zero_loss(ground_truth, predictions):
    return np.mean(ground_truth != predictions)

# greater_is_better should be False, but result will be negative numbers
# CV=5 will be a StratifiedKFold, but we need common KFold actually
loss = sm.make_scorer(one_zero_loss, greater_is_better=True)
scoreMat = np.zeros((len(C_range), K))
for i in range(len(C_range)):
    Ci = C_range[i]
    SVC = svm.SVC(Ci, gamma=gamma, shrinking=False)
    scoreMat[i] = sms.cross_val_score(SVC, x_standard, y, cv=kf, scoring=loss)

scores2 = np.mean(scoreMat, axis=1)
print(scoreMat)
print('CV scores by cross_val_score: \n', scores2) # the same as in method 1

# Tuning hyper-parameters using GridSearchCV
print('{0:-^60}'.format('Tuning parameters by GridSearchCV'))
C_space = np.logspace(-1, 3.5, 10)
gamma_space = np.logspace(-1, 1, 10)
paramSpace = dict(C=C_space, gamma=gamma_space)
svc = svm.SVC(shrinking=False)
clf = sms.GridSearchCV(svc, param_grid=paramSpace, scoring=loss, cv=kf)
clf.fit(x_standard, y)
print('GridSearchCV scores: \n', clf.cv_results_['mean_test_score'])  # should be 100
print('related params: \n', clf.cv_results_['params'])

params = clf.cv_results_['params']
scores3 = clf.cv_results_['mean_test_score']
assert len(scores3) == len(params)
xx = np.zeros(len(params))
yy = np.zeros(len(params))
zz = np.zeros(len(params))
for i in range(len(params)):
    xx[i] = params[i]['C']
    yy[i] = params[i]['gamma']
    zz[i] = scores3[i]
xx = xx.reshape(len(C_space), len(gamma_space))
yy = yy.reshape(len(C_space), len(gamma_space))
zz = zz.reshape(len(C_space), len(gamma_space))


# plots
fig = plt.figure(figsize=(10, 4))
fig.canvas.set_window_title('svmCgammaDemo')

ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(xx, yy, zz, cmap='jet')  # 3D 的图不是很会画。。但数据应该是一样的

plt.subplot(122)
plt.axis([10**-2, 10**4, 0.185, 0.37])
plt.xticks(np.logspace(-2, 4, 7))
plt.yticks(np.linspace(0.15, 0.35, 5))
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('CV Error')
plt.title(r'$\gamma = {0}$'.format(gamma))
plt.errorbar(C_range, scores, scores_se, ecolor='b', elinewidth=1, capsize=3)
plt.plot(C_range, scores, marker='o', mfc='darkorange', mec='k', mew=2, ms=8, linestyle='none')
plt.axhline(0.21, color='r', linestyle='--', lw=2)  # bayes optimal error
plt.axvline(best_C, color='g', linestyle='--', lw=2)

plt.tight_layout()
plt.show()