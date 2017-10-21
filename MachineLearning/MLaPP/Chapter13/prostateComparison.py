import pprint
import warnings
import numpy as np
import scipy.io as sio
import scipy.linalg as sl
import sklearn.metrics as sm
import sklearn.model_selection as sms
import sklearn.linear_model as slm
from subsets import *

np.random.seed(0)

# prepare data
data = sio.loadmat('prostateStnd.mat')
print(data.keys())
X, y = data['Xtrain'], data['ytrain']
xtest, ytest = data['Xtest'], data['ytest']
legends = np.array(['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45'])

def OneSE_Rule(scoreDict, useRule=True):
    # use 1SE rule, 就是CV的最小值加上一个标准差，取这个值作为最优的参数
    scores = np.array(list(scoreDict.keys()), dtype='float64')
    std_err = np.var(scores)**0.5
    print('ridge cv std_err: ', std_err)
    s = np.sort(scores)
    if useRule:
        thresh = s.min() + std_err  # 1 SE rule
        s_se = s[s <= thresh][-1]    # 比它小的里面挑个最大的
        #s_se = s[s > thresh][0]        # 比它大的里面挑个最小的
        index = scoreDict[s_se]
        print('best lambda by Ridge CV: ', index)
    else:
        index = scoreDict[s[0]]

    return index

def RidgeCV(X, y, xtest, ytest, K):
    print('{0:-^100}'.format('Ridge CV'))
    maxLambda = sl.norm(2 * np.dot(X.T, y), np.inf)
    maxLambda = np.log10(maxLambda)
    lambdaRange = np.logspace(-2, maxLambda, 30)
    N, D = X.shape
    w_mat = np.zeros((len(lambdaRange), D))
    intercept = np.zeros(len(lambdaRange))
    scoreDict = dict()
    for i in range(len(lambdaRange)):
        l = lambdaRange[i]
        X_y = np.c_[X, y]
        np.random.shuffle(X_y)
        x_train, y_train = X_y[:, :D], X_y[:, -1].reshape(-1, 1)
        ridge = slm.Ridge(alpha=l, fit_intercept=True).fit(x_train, y_train)
        w_mat[i] = ridge.coef_
        intercept[i] = ridge.intercept_
        
        ridgeCV = slm.Ridge(alpha=l, fit_intercept=True)
        scores_i = -1 * sms.cross_val_score(ridgeCV, x_train, y_train, cv=K, scoring='neg_mean_squared_error')
        mean_score = np.mean(scores_i)
        scoreDict[mean_score] = i

    print('ridge cv scores:')
    pprint.pprint(scoreDict)

    index = OneSE_Rule(scoreDict)
    
    w, icpt = w_mat[index], intercept[index]
    print('w by Ridge CV: ', w)
    print('intercept by Ridge CV: ', icpt)

    y_test_predict = np.dot(xtest, w.reshape(-1, 1)) + icpt
    mse_ridge_cv = sm.mean_squared_error(ytest, y_test_predict)
    print('mse of Ridge CV: ', mse_ridge_cv)

    return w, icpt, mse_ridge_cv

def LassoCV(X, y, xtest, ytest, K):
    print('{0:-^100}'.format('Lasso CV'))
    larsRes = slm.lars_path(X, y.ravel())  # 根据lars_path来得出最大的maxLambda
    lambda_path = larsRes[0]
    print('maxLambda of Lasso: ', lambda_path[0])
    maxLambda = np.log10(lambda_path[0])
    lambdaRange = np.logspace(-2, maxLambda, 30)
    N, D = X.shape
    w_mat = np.zeros((len(lambdaRange), D))
    intercept = np.zeros(len(lambdaRange))
    scoreDict = dict()
    for i in range(len(lambdaRange)):
        l = lambdaRange[i]
        X_y = np.c_[X, y]
        np.random.shuffle(X_y)
        x_train, y_train = X_y[:, :D], X_y[:, -1].reshape(-1, 1)
        lasso = slm.Lasso(alpha=l, fit_intercept=True).fit(x_train, y_train)
        w_mat[i] = lasso.coef_
        intercept[i] = lasso.intercept_
        
        lassoCV = slm.Lasso(alpha=l, fit_intercept=True)
        scores_i = -1 * sms.cross_val_score(lassoCV, x_train, y_train, cv=K, scoring='neg_mean_squared_error')
        mean_score = np.mean(scores_i)
        scoreDict[mean_score] = i

    print('Lasso cv scores:')
    pprint.pprint(scoreDict)

    #index = OneSE_Rule(scoreDict)
    index = OneSE_Rule(scoreDict, False)  # Lasso CV 不需要使用1SE Rule
    
    w, icpt = w_mat[index], intercept[index]
    print('w by Lasso CV: ', w)
    print('intercept by Lasso CV: ', icpt)

    y_test_predict = np.dot(xtest, w.reshape(-1, 1)) + icpt
    mse_lasso_cv = sm.mean_squared_error(ytest, y_test_predict)
    print('mse of Lasso CV: ', mse_lasso_cv)

    return w, icpt, mse_lasso_cv

# customized estimator
class SubsetSelection:
    def __init__(self, cols):
        self.cols = np.array(cols)  # convert tuple to ndarray, for indexing

    def predict(self, X):
        X_cut = X[:, self.cols]
        return self.model.predict(X_cut)

    def fit(self, X, y): 
        X_cut = X[:, self.cols]
        LR = slm.LinearRegression(fit_intercept=True).fit(X_cut, y)
        self.model = LR
        D = X.shape[1]
        w = np.zeros(D)
        w[self.cols] = LR.coef_
        self.w = w
        self.icpt = LR.intercept_

        return self

    def get_params(self, deep=False):
        return {'cols': self.cols}

def getAllSubsets(N):
    ss = subsets(N)
    result = []
    for k, v in ss.items():
        if k == 0:
            continue  # 剔除空集
        for i in v:
            result.append(i)

    return result

def SubsetCV(X, y, xtest, ytest, K):
    print('{0:-^100}'.format('Best Subset CV'))
    N, D = X.shape
    subsets = getAllSubsets(D)
    w_mat = np.zeros((len(subsets), D))
    intercept = np.zeros(len(subsets))
    scoreDict = dict()
    for i in range(len(subsets)):
        cols = subsets[i]
        X_y = np.c_[X, y]
        np.random.shuffle(X_y)
        x_train, y_train = X_y[:, :D], X_y[:, -1].reshape(-1, 1)
        ss = SubsetSelection(cols).fit(x_train, y_train)
        w_mat[i] = ss.w
        intercept[i] = ss.icpt
        
        ssCV = SubsetSelection(cols)
        scores_i = -1 * sms.cross_val_score(ssCV, x_train, y_train, cv=K, scoring='neg_mean_squared_error')
        mean_score = np.mean(scores_i)
        scoreDict[mean_score] = i

    print('dont print Best Subset cv scores:')
    # pprint.pprint(scoreDict)

    index = OneSE_Rule(scoreDict, False)  # Subset CV 也不使用1SE Rule
    
    w, icpt = w_mat[index], intercept[index]
    print('w by Best Subset CV: ', w)
    print('intercept by Best Subset CV: ', icpt)

    y_test_predict = np.dot(xtest, w.reshape(-1, 1)) + icpt
    mse_ss_cv = sm.mean_squared_error(ytest, y_test_predict)
    print('mse of Best Subset CV: ', mse_ss_cv)

    return w, icpt, mse_ss_cv


# OLS
ols = slm.LinearRegression(fit_intercept=True).fit(X, y)
w_ols = ols.coef_.ravel()
icpt_ols = np.asscalar(ols.intercept_)
print('w_ols: ', w_ols)
print('intercept of ols: ', ols.intercept_)
y_test_predict = ols.predict(xtest)
mse_ols = sm.mean_squared_error(ytest, y_test_predict)
print('mse by ols: ', mse_ols)

# Ridge, Lasso, SS with CV
w_ridge, icpt_ridge, mse_ridge_cv = RidgeCV(X, y, xtest, ytest, 10)
w_lasso, icpt_lasso, mse_lasso_cv = LassoCV(X, y, xtest, ytest, 10)
warnings.filterwarnings("ignore")  # Suppress warnings from sklearn
w_ss, icpt_ss, mse_ss_cv = SubsetCV(X, y, xtest, ytest, 10)

print('\n\n')
print('{0:<20}{1:<20}{2:<20}{3:<20}{4:<20}'.format('Term', 'LS', 'Best Subset', 'Ridge', 'Lasso'))
print('_'*100)
print('{0:<20}{1:<20.4f}{2:<20.4f}{3:<20.4f}{4:<20.4f}'.format('Intercept', icpt_ols, icpt_ss, icpt_ridge, icpt_lasso))
for i in range(len(legends)):
    olsi = np.asscalar(w_ols[i])
    ssi = np.asscalar(w_ss[i])
    ridgei = np.asscalar(w_ridge[i])
    lassoi = np.asscalar(w_lasso[i])
    print('{0:<20}{1:<20.4f}{2:<20.4f}{3:<20.4f}{4:<20.4f}'.format(legends[i], olsi, ssi, ridgei, lassoi))
print('_'*100)
print('{0:<20}{1:<20.4f}{2:<20.4f}{3:<20.4f}{4:<20.4f}'.format('Test Error', mse_ols, mse_ss_cv, mse_ridge_cv, mse_lasso_cv))

    
