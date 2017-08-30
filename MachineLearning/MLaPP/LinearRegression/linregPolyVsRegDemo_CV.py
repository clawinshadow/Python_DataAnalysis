import numpy as np
import sklearn.linear_model as slm
import sklearn.preprocessing as sp
import sklearn.model_selection as sml
import matplotlib.pyplot as plt

def DM(x, degree):
    if degree < 1:
        return x
    else:
        result = x.reshape(-1, 1)
        i = 2
        while i <= degree:
            result = np.c_[result, (x**i).reshape(-1, 1)]
            i += 1
        return result

def Noisy(x):
    sigma = 2         # 噪声的标准差 
    w_true = [-1.5, 1/9]
    y_true = w_true[0] * x + w_true[1] * x**2       # real func，没有带截距

    return y_true + sigma * np.random.randn(len(x))

def GenerateTrainingData(n):
    x_train = np.linspace(0, 20, n)
    y_train = Noisy(x_train)

    return x_train, y_train

def GetNegMSE(x, y, x_test, y_test, Lambda):
    model = slm.Ridge(alpha=Lambda, fit_intercept=False)
    model.fit(x, y)
    y_predict = model.predict(x_test)
    return -np.mean((y_predict - y_test)**2)

x_train, y_train = GenerateTrainingData(21)
x_train_dm = DM(x_train, 14)
x_train_dm = sp.StandardScaler().fit_transform(x_train_dm)  # center the data

def CV_MSE(Lambda):
    ridge = slm.Ridge(alpha=Lambda, fit_intercept=False)
    # MSE * -1 : Neg MSE
    cv = sml.cross_val_score(ridge, x_train_dm, y_train, cv=5, scoring='neg_mean_squared_error')
    print('cv with lambda - {0}: {1}'.format(Lambda, cv))
    return cv

# verify CV_MSE, 得出来的结果应该是一样的
def CV_Verify(Lambda):
    kf = sml.KFold(5)
    MSEs = []
    for train_indices, test_indices in kf.split(x_train_dm):
        x_train_i = x_train_dm[train_indices]
        x_test_i = x_train_dm[test_indices]
        y_train_i = y_train[train_indices]
        y_test_i = y_train[test_indices]
        MSEs.append(GetNegMSE(x_train_i, y_train_i, x_test_i, y_test_i, Lambda))
    print('cv_verify : ', MSEs)
    return -1 * np.mean(MSEs)

log_ls = np.linspace(-23, 3, 10)
ls = np.exp(log_ls)

mses = []
for i in range(len(ls)):
    Lambda = ls[i]
    mses.append(CV_Verify(Lambda))
    # mses.append(CV_plot(Lambda))

print('mses: ', mses)

sortIndices = np.argsort(mses)
maxLambda = log_ls[sortIndices[0]]
print('maxLambda: ', maxLambda)
print(np.max(mses))

plt.figure(figsize=(7, 6))
plt.subplot()
plt.semilogy(log_ls, mses, color='k', marker='o', ms=10, mew=2, fillstyle='none')
plt.axvline(maxLambda, color='midnightblue', lw=2)  # 使用axvline, 可以不用指定起点和终点，比vlines()方便
plt.title('5−fold cross validation, ntrain = 21')
plt.xlabel('log lambda')
plt.ylabel('mse')
plt.xlim(-25, 5)
plt.xticks(np.arange(-25, 6, 5))

plt.show()

# CV_MSE(ls[0])
# CV_Verify(ls[0])
