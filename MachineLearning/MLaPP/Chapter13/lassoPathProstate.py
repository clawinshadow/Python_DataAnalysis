import numpy as np
import scipy.io as sio
import scipy.linalg as sl
import sklearn.metrics as sm
import sklearn.preprocessing as sp
import sklearn.linear_model as slm
import sklearn.model_selection as sms
import matplotlib.pyplot as plt

'''
要注意书中的两个术语，一个是Lambda，lambda就是|w|前面的系数，跟ridge里面的含义是一样的
另一个是B，B表示目标函数去掉lambda之后的另一种表示方式，s.t. |w| < B
B 和 lambda是反比例的关系，但是没有一定的公式可以互相转换。这三个图有的是用lambda来画的
有的是用B来画的，这个一定要弄清楚

ps: looks slightly different with the figures in book, but essence is the same
'''

np.random.seed(0)

# prepare data
data = sio.loadmat('prostateStnd.mat')
print(data.keys())
X, y = data['Xtrain'], data['ytrain']  # xtrain更接近于书里面的值，不要用全部的X
#X, y = data['X'], data['y']
legends = np.array(['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45'])

# Fit models with OLS
y = y - np.mean(y)                           # 这里如果不center y，后面的lars_path的结果会不准确
ols = slm.LinearRegression(False).fit(X, y)  # fit without intercept
print('ols coefficients: ', ols.coef_)
B_max = np.sum(np.abs(ols.coef_))
# tmax = sl.norm(ols.coef_, 1)
print('l1 norm of ols coefficients: ', B_max)

# Fit models with Lasso-LARS
NL = 24
N, D = X.shape

larsRes = slm.lars_path(X, y.ravel())    # 先用LARS计算lasso的path，看lambda的最大值和最小值是多少
lambda_path = larsRes[0]    # 9个
coef_path = larsRes[2]      # (8, 9) every col is a w vector
B = np.sum(np.abs(coef_path), axis=0)
print('lambda path: ', lambda_path)
print('coef path: ', coef_path)  # 可以看到最后的一列，就是前面的 w_ols，代表无约束的w向量
print('B in critical values: ', B)
maxLambda = lambda_path[0]
lambdas = np.logspace(np.log10(maxLambda), -3.3, NL)  # 非线性的划分区间
print('lambdas: ', lambdas)
w_mat = np.zeros((NL, D))
K = 5
scores_cv = np.zeros((NL, K))
for i in range(NL):
    li = lambdas[i]
    lasso = slm.Lasso(alpha=li, fit_intercept=False).fit(X, y)
    w_mat[i] = lasso.coef_
    lassoCV = slm.Lasso(alpha=li, fit_intercept=False)
    scores_cv[i] = sms.cross_val_score(lassoCV, X, y, cv=K)
    
mean_scores = np.mean(scores_cv, axis=1)
print('cv scores: ', mean_scores)
min_idx = np.argsort(mean_scores)[-1]  # 这里不再使用1SE rule

print('coefficients: ', w_mat)  # for plot 1

# plots
fig = plt.figure(figsize=(9, 7))
fig.canvas.set_window_title('lassoPathProstate')

plt.subplot(221)
plt.axis([0, 25, -0.3, 0.7])
plt.xticks(np.linspace(0, 25, 6))
plt.yticks(np.linspace(-0.3, 0.7, 11))
plt.plot(np.linspace(1, NL, NL), w_mat, marker='o', fillstyle='none')
plt.axvline(min_idx, color='red', lw=2)
plt.legend(legends)

plt.subplot(223)
plt.axis([0, 2.5, -0.3, 0.7])
plt.xticks(np.linspace(0, 2.5, 6))
plt.yticks(np.linspace(-0.3, 0.7, 11))
plt.plot(B, coef_path.T, marker='o', ms=5, fillstyle='none')
plt.legend(legends)

plt.subplot(224)
plt.axis([1, 9.1, -0.3, 0.7])
LN = len(lambda_path)
plt.xticks(np.linspace(1, LN, LN))
plt.yticks(np.linspace(-0.3, 0.7, 11))
plt.plot(np.linspace(1, LN, LN), coef_path.T, marker='o', ms=5, fillstyle='none')
plt.legend(legends)

plt.tight_layout()
plt.show()

