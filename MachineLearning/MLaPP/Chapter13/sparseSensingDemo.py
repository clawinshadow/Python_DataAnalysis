import numpy as np
import scipy.linalg as sl
import sklearn.linear_model as slm
import sklearn.metrics as sm
import matplotlib.pyplot as plt

np.random.seed(0)

# generate data
N = 2**10  # number of observataions
D = 2**12  # dimension of data
n_spikes= 160 # -1 or +1
w = np.zeros(D)
w[:n_spikes] = np.random.choice([-1, 1], size=n_spikes)
np.random.shuffle(w)
w = w.reshape(-1, 1)

X = np.random.randn(N, D)
# orthonormalize rows, 为什么不是正交化列，可能是因为D > N的关系，如果直接orth(X)会返回D*D的矩阵
X = sl.orth(X.T).T

sigma = 0.01
y = np.dot(X, w) + sigma * np.random.randn(N, 1)  # N * 1, add noise
print(y.shape)

# Fit with lasso
max_lambda = np.max(np.abs(np.dot(X.T, y)))
tau = 0.1 * max_lambda / (2 * N)  # lambda 一定要rescale后才能传入sklearn.Lasso
print('tau: ', tau)
lasso = slm.Lasso(alpha=tau).fit(X, y)
print('w by l1: ', lasso.coef_)
w_l1 = lasso.coef_
print('length of w: {0} nonzero count: {1} '.format(len(w_l1), np.count_nonzero(w_l1)))
#y_predict = lasso.predict(X)
mse_l1 = sm.mean_squared_error(w, w_l1)  # 注意mse是根据w来算的，不是根据y
print('mse of L1 reconstruction: ', mse_l1)

# debiasing
max_wj = np.max(np.abs(w_l1))
idx = np.abs(w_l1) > 0.01 * max_wj
print(idx.shape)
X_shrinked = X[:, idx]  # 小于0.01*max_wj的feature视为被错误的收缩了的，要用OLS恢复
print('shape of X_shrinked: ', X_shrinked.shape)
w_debiased = np.zeros(D)
ols = slm.LinearRegression().fit(X_shrinked, y)
w_debiased[idx] = ols.coef_.ravel()
# y_predict_debiased = np.dot(X, w_debiased.reshape(-1, 1))
mse_l1_debiased = sm.mean_squared_error(w, w_debiased)  # 注意mse是根据w来算的，不是根据y
print('mse_l1_debiased: ', mse_l1_debiased)

# OLS
ols2 = slm.LinearRegression().fit(X, y)
w_ols = ols2.coef_.ravel()
print(w_ols.min(), w_ols.max())
mse_ols = sm.mean_squared_error(w, w_ols)
print('mse_ols: ', mse_ols)

# plots
def plot(index, title, w):
    plt.subplot(index)
    #ax.spines['bottom'].set_position('center')
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    plt.title(title)
    if index == 414:
        plt.axis([0, 4100, -0.6, 0.6])
        plt.xticks(np.linspace(0, 4000, 5))
        plt.yticks([-0.5, 0, 0.5])
    else:
        plt.axis([0, 4100, -1.05, 1.05])
        plt.xticks(np.linspace(0, 4000, 5))
        plt.yticks([-1, 0, 1])
    plt.axhline(y=0, color='midnightblue', lw=1)

    condlist = [w <= 0, w > 0]
    min_choicelist = [w, 0]
    max_choicelist = [0, w]
    wmin = np.select(condlist, min_choicelist)
    wmax = np.select(condlist, max_choicelist)
    plt.vlines(np.arange(0, len(w), 1), wmin, wmax, colors='midnightblue', lw=0.5)
    # 用循环的方式画4096条线超级无敌慢。。
    #for i in range(len(w)):
    #    plt.plot([i, i], [0, w[i]], color='midnightblue', lw=0.5)
    
fig = plt.figure(figsize=(9, 6))
fig.canvas.set_window_title('sparseSensingDemo')

plot(411, 'Original (D = 4096, number of nonzeros = 160)', w)

titleStr = 'L1 reconstruction (K0 = 1024, lambda = {0:.4f}, MSE = {1:.6f})'.format(tau * 2 * N, mse_l1)
plot(412, titleStr, w_l1)

titleStr2 = 'Debiased (MSE = {0:e})'.format(mse_l1_debiased)
plot(413, titleStr2, w_debiased)

titleStr3 = 'Minimum norm solution (MSE = {0:.5f})'.format(mse_ols)
plot(414, titleStr3, w_ols)

plt.tight_layout()
plt.show()
